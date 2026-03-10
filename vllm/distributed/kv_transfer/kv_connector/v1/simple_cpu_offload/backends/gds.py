# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GDS transfer backend for direct GPU↔NVMe via GPUDirect Storage."""

import os
from typing import TYPE_CHECKING, Any

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload.backends.base import (  # noqa: E501
    TransferBackend,
)
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)

_GDS_ALIGNMENT = 4096


class _GDSEvent:
    """Completion event wrapping a list of kvikio IOFutures.

    Matches the query()/wait() interface used by _DiskEvent and
    torch.cuda.Event so callers can use them uniformly.
    """

    def __init__(self, futures: list) -> None:
        self._futures = futures
        self._done = False

    def query(self) -> bool:
        if self._done:
            return True
        if all(f.done() for f in self._futures):
            self._done = True
            return True
        return False

    def wait(self) -> None:
        if self._done:
            return
        for f in self._futures:
            f.get()
        self._done = True


class GDSTransferBackend(TransferBackend):
    """Transfer backend for direct GPU↔NVMe via GPUDirect Storage.

    Uses kvikio CuFile for DMA between GPU memory and NVMe storage,
    bypassing CPU staging buffers entirely.  Falls back to kvikio
    compatibility mode (POSIX I/O) when block sizes are not 4KB-aligned.
    """

    def __init__(self, disk_path: str) -> None:
        try:
            import kvikio  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "GDSTransferBackend requires kvikio. "
                "Install with: pip install kvikio-cu12"
            ) from e

        self._disk_path = disk_path
        self._cufile: Any | None = None
        self._gpu_kv_caches: dict[str, torch.Tensor] | None = None
        self._num_disk_blocks: int = 0
        self._bytes_per_block: int = 0
        self._num_layers: int = 0
        self._last_event: _GDSEvent | None = None

    # ------------------------------------------------------------------
    # TransferBackend interface
    # ------------------------------------------------------------------

    @property
    def is_initialized(self) -> bool:
        return self._cufile is not None and self._gpu_kv_caches is not None

    def setup(
        self,
        src_caches: dict[str, torch.Tensor],
        capacity_bytes: int,
        kv_cache_config: "KVCacheConfig | None",
    ) -> int:
        import kvikio

        self._gpu_kv_caches = src_caches
        self._num_layers = len(src_caches)

        first = next(iter(src_caches.values()))
        self._bytes_per_block = first.stride(0) * first.element_size()

        if self._bytes_per_block % _GDS_ALIGNMENT != 0:
            logger.warning(
                "GDSTransferBackend: bytes_per_block=%d is not 4KB-aligned. "
                "GPUDirect Storage will use compatibility mode (POSIX "
                "fallback). Performance will be similar to "
                "backend_type='disk'.",
                self._bytes_per_block,
            )
            kvikio.defaults.compat_mode_set(True)

        self._num_disk_blocks = max(
            1, capacity_bytes // (self._bytes_per_block * self._num_layers)
        )
        total_bytes = self._bytes_per_block * self._num_layers * self._num_disk_blocks

        logger.info(
            "GDSTransferBackend: %d layers, %d disk blocks (%.2f GB), file=%s",
            self._num_layers,
            self._num_disk_blocks,
            total_bytes / (1024**3),
            self._disk_path,
        )

        # Pre-allocate backing file.
        fd = os.open(self._disk_path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
        try:
            os.posix_fallocate(fd, 0, total_bytes)
        except (OSError, AttributeError):
            os.ftruncate(fd, total_bytes)
        finally:
            os.close(fd)

        self._cufile = kvikio.CuFile(self._disk_path, "r+")
        return self._num_disk_blocks

    def copy_blocks(
        self,
        src_block_ids: list[int],
        dst_block_ids: list[int],
        is_store: bool,
    ) -> None:
        assert self._cufile is not None
        assert self._gpu_kv_caches is not None

        futures: list = []
        for layer_idx, name in enumerate(self._gpu_kv_caches):
            tensor = self._gpu_kv_caches[name]
            for src_id, dst_id in zip(src_block_ids, dst_block_ids):
                if is_store:
                    futures.append(
                        self._cufile.pwrite(
                            tensor[src_id],
                            self._bytes_per_block,
                            self._file_offset(layer_idx, dst_id),
                        )
                    )
                else:
                    futures.append(
                        self._cufile.pread(
                            tensor[dst_id],
                            self._bytes_per_block,
                            self._file_offset(layer_idx, src_id),
                        )
                    )

        self._last_event = _GDSEvent(futures)

    def record_event(self) -> _GDSEvent:
        assert self._last_event is not None
        event = self._last_event
        self._last_event = None
        return event

    def query_event(self, event: Any) -> bool:
        return event.query()

    def sync_event(self, event: Any) -> None:
        event.wait()

    def sync_all(self) -> None:
        if self._last_event is not None:
            self._last_event.wait()

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
        if self._cufile is not None:
            self._cufile.close()
            self._cufile = None
        if os.path.exists(self._disk_path):
            os.unlink(self._disk_path)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _file_offset(self, layer_idx: int, block_id: int) -> int:
        return (
            layer_idx * self._num_disk_blocks * self._bytes_per_block
            + block_id * self._bytes_per_block
        )
