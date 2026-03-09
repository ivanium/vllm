# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Disk transfer backend for direct GPU↔Disk via internal CPU staging."""

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload.backends.base import (  # noqa: E501
    TransferBackend,
)
from vllm.logger import init_logger
from vllm.utils.platform_utils import is_pin_memory_available

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)


class _DiskEvent:
    """Adapter wrapping :class:`threading.Event` for the query/sync pattern.

    This mirrors the interface used by :class:`torch.cuda.Event` so that
    callers can use ``query()`` / ``wait()`` uniformly regardless of the
    backend.
    """

    def __init__(self) -> None:
        self._event = threading.Event()

    def set(self) -> None:
        self._event.set()

    def query(self) -> bool:
        return self._event.is_set()

    def wait(self) -> None:
        self._event.wait()


class DiskTransferBackend(TransferBackend):
    """Transfer backend for direct GPU↔Disk via internal CPU staging.

    Stores use GPU→staging(CPU)→Disk; loads use Disk→staging(CPU)→GPU.
    The staging buffers are internal and not visible to the scheduler.
    """

    DEFAULT_NUM_STAGING_BLOCKS = 64

    def __init__(
        self,
        disk_path: str,
        num_workers: int = 4,
        num_staging_blocks: int = DEFAULT_NUM_STAGING_BLOCKS,
    ) -> None:
        self._disk_path = disk_path
        self._num_workers = num_workers
        self._num_staging_blocks = num_staging_blocks

        # Populated by setup()
        self._fd: int | None = None
        self._gpu_kv_caches: dict[str, torch.Tensor] | None = None
        self._staging: dict[str, torch.Tensor] | None = None
        self._num_disk_blocks: int = 0
        self._bytes_per_block: int = 0
        self._num_layers: int = 0
        self._pool: ThreadPoolExecutor | None = None
        self._last_event: _DiskEvent | None = None

    # ------------------------------------------------------------------
    # TransferBackend interface
    # ------------------------------------------------------------------

    @property
    def is_initialized(self) -> bool:
        return (
            self._fd is not None
            and self._gpu_kv_caches is not None
            and self._staging is not None
            and self._pool is not None
        )

    def setup(
        self,
        src_caches: dict[str, torch.Tensor],
        capacity_bytes: int,
        kv_cache_config: "KVCacheConfig | None",
    ) -> int:
        self._gpu_kv_caches = src_caches
        self._num_layers = len(src_caches)

        first = next(iter(src_caches.values()))
        self._bytes_per_block = first.stride(0) * first.element_size()

        self._num_disk_blocks = max(
            1, capacity_bytes // (self._bytes_per_block * self._num_layers)
        )

        total_bytes = self._bytes_per_block * self._num_layers * self._num_disk_blocks

        logger.info(
            "DiskTransferBackend: %d layers, %d disk blocks (%.2f GB), "
            "%d staging blocks, file=%s",
            self._num_layers,
            self._num_disk_blocks,
            total_bytes / (1024**3),
            self._num_staging_blocks,
            self._disk_path,
        )

        # Allocate pinned CPU staging buffers.
        pin = is_pin_memory_available()
        self._staging = {}
        for name, gpu_tensor in src_caches.items():
            shape = (self._num_staging_blocks,) + gpu_tensor.shape[1:]
            self._staging[name] = torch.zeros(
                shape,
                dtype=gpu_tensor.dtype,
                device="cpu",
                pin_memory=pin,
            )

        # Open backing file.
        self._fd = os.open(
            self._disk_path,
            os.O_RDWR | os.O_CREAT | os.O_TRUNC,
            0o644,
        )
        try:
            os.posix_fallocate(self._fd, 0, total_bytes)
        except (OSError, AttributeError):
            os.ftruncate(self._fd, total_bytes)

        self._pool = ThreadPoolExecutor(max_workers=self._num_workers)
        return self._num_disk_blocks

    def copy_blocks(
        self,
        src_block_ids: list[int],
        dst_block_ids: list[int],
        is_store: bool,
    ) -> None:
        assert self._fd is not None
        assert self._gpu_kv_caches is not None
        assert self._staging is not None
        assert self._pool is not None

        event = _DiskEvent()
        self._last_event = event

        n = len(src_block_ids)
        if n == 0:
            event.set()
            return

        # Process in chunks of staging capacity.
        chunk_size = self._num_staging_blocks
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            src_chunk = src_block_ids[start:end]
            dst_chunk = dst_block_ids[start:end]
            is_last_chunk = end == n

            if is_store:
                self._store_chunk(
                    src_chunk,
                    dst_chunk,
                    event if is_last_chunk else None,
                )
            else:
                self._load_chunk(
                    src_chunk,
                    dst_chunk,
                    event if is_last_chunk else None,
                )

    def record_event(self) -> _DiskEvent:
        assert self._last_event is not None
        event = self._last_event
        self._last_event = None
        return event

    def query_event(self, event: Any) -> bool:
        return event.query()

    def sync_event(self, event: Any) -> None:
        event.wait()

    def sync_all(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = ThreadPoolExecutor(max_workers=self._num_workers)

    def validate_block_ids(self, block_ids: list[int], is_src: bool) -> None:
        if not block_ids:
            return
        if is_src:
            assert self._gpu_kv_caches is not None
            num = next(iter(self._gpu_kv_caches.values())).shape[0]
            label = "Source (GPU)"
        else:
            num = self._num_disk_blocks
            label = "Dest (Disk)"
        lo, hi = min(block_ids), max(block_ids)
        if lo < 0 or hi >= num:
            bad = lo if lo < 0 else hi
            raise ValueError(f"{label} block ID {bad} out of bounds [0, {num})")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        if os.path.exists(self._disk_path):
            os.unlink(self._disk_path)

    # ------------------------------------------------------------------
    # Internal: chunked store / load
    # ------------------------------------------------------------------

    def _store_chunk(
        self,
        gpu_block_ids: list[int],
        disk_block_ids: list[int],
        event: _DiskEvent | None,
    ) -> None:
        """GPU → staging → Disk for one chunk."""
        assert self._gpu_kv_caches is not None
        assert self._staging is not None
        assert self._pool is not None

        n = len(gpu_block_ids)
        staging_ids = list(range(n))

        # GPU → staging (indexed copy via torch).
        for name in self._gpu_kv_caches:
            self._staging[name][staging_ids] = self._gpu_kv_caches[name][gpu_block_ids]

        # staging → Disk (async pwrite via thread pool).
        sem = threading.Semaphore(0)
        num_ios = n * self._num_layers
        assert self._fd is not None
        fd = self._fd

        for layer_idx, name in enumerate(self._staging):
            for i, disk_id in enumerate(disk_block_ids):
                data = self._staging[name][i].contiguous().numpy().tobytes()
                offset = self._file_offset(layer_idx, disk_id)
                self._pool.submit(self._pwrite_task, fd, data, offset, sem)

        if event is not None:
            self._pool.submit(self._waiter_task, sem, num_ios, event)
        else:
            # Must wait before reusing staging buffers for the next chunk.
            for _ in range(num_ios):
                sem.acquire()

    def _load_chunk(
        self,
        disk_block_ids: list[int],
        gpu_block_ids: list[int],
        event: _DiskEvent | None,
    ) -> None:
        """Disk → staging → GPU for one chunk."""
        assert self._gpu_kv_caches is not None
        assert self._staging is not None
        assert self._pool is not None

        n = len(disk_block_ids)
        staging_ids = list(range(n))

        # Disk → staging (sync pread, must complete before GPU copy).
        sem = threading.Semaphore(0)
        num_ios = n * self._num_layers
        assert self._fd is not None
        fd = self._fd

        for layer_idx, name in enumerate(self._staging):
            tensor = self._staging[name]
            for i, disk_id in enumerate(disk_block_ids):
                offset = self._file_offset(layer_idx, disk_id)
                self._pool.submit(
                    self._pread_task,
                    fd,
                    tensor,
                    i,
                    offset,
                    self._bytes_per_block,
                    sem,
                )

        for _ in range(num_ios):
            sem.acquire()

        # staging → GPU (indexed copy via torch).
        for name in self._gpu_kv_caches:
            self._gpu_kv_caches[name][gpu_block_ids] = self._staging[name][staging_ids]

        if event is not None:
            event.set()

    def _file_offset(self, layer_idx: int, block_id: int) -> int:
        return (
            layer_idx * self._num_disk_blocks * self._bytes_per_block
            + block_id * self._bytes_per_block
        )

    # ------------------------------------------------------------------
    # I/O helpers (run in thread pool)
    # ------------------------------------------------------------------

    @staticmethod
    def _pwrite_task(
        fd: int,
        data: bytes,
        offset: int,
        sem: threading.Semaphore,
    ) -> None:
        try:
            os.pwrite(fd, data, offset)
        finally:
            sem.release()

    @staticmethod
    def _pread_task(
        fd: int,
        tensor: torch.Tensor,
        staging_idx: int,
        offset: int,
        nbytes: int,
        sem: threading.Semaphore,
    ) -> None:
        try:
            data = os.pread(fd, nbytes, offset)
            block = torch.frombuffer(bytearray(data), dtype=tensor.dtype)
            tensor[staging_idx].copy_(block.view(tensor[staging_idx].shape))
        finally:
            sem.release()

    @staticmethod
    def _waiter_task(
        sem: threading.Semaphore,
        count: int,
        event: _DiskEvent,
    ) -> None:
        for _ in range(count):
            sem.acquire()
        event.set()
