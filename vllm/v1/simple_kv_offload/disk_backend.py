# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Disk I/O backend for GPU<->NVMe block transfers via pinned staging buffers.

Uses separate IO threads for store and load so that loads (latency-critical)
never block behind stores (background work). Each thread owns its own pinned
staging buffers to avoid contention.

The disk file mirrors the GPU pool: block ``b`` of a group is slot ``b`` of
the group's slot array. A block's content is a single contiguous byte range
on both sides (see ``CopyGeometry``), so each staged block is one DMA and one
pwrite/pread.
"""

from __future__ import annotations

import contextlib
import os
import queue
import threading
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.math_utils import round_up
from vllm.v1.kv_cache_interface import large_page_block_range, large_page_of_block
from vllm.v1.simple_kv_offload.cuda_mem_ops import (
    CU_MEMCPY_SRC_ACCESS_ORDER_ANY,
    CU_MEMCPY_SRC_ACCESS_ORDER_STREAM,
    CopyGeometry,
    DriverMemcpyParams,
    driver_memcpy_params,
    launch_batch,
    pin_tensor,
)

logger = init_logger(__name__)

O_DIRECT = getattr(os, "O_DIRECT", 0)
_ALIGNMENT = 4096


def _alloc_aligned(num_slots: int, slot_bytes: int) -> torch.Tensor:
    """Allocate a staging buffer whose base address is O_DIRECT aligned.

    The CPU allocator only guarantees 64-byte alignment, so over-allocate by
    one alignment unit and return an aligned view. The view keeps the backing
    storage alive.
    """
    nbytes = num_slots * slot_bytes
    raw = torch.zeros(nbytes + _ALIGNMENT, dtype=torch.int8, device="cpu")
    offset = -raw.data_ptr() % _ALIGNMENT
    return raw[offset : offset + nbytes].view(num_slots, slot_bytes)


class DiskSlotLayout(NamedTuple):
    """Disk file layout: pool page ``p`` is file bytes ``[p * page_bytes,
    (p + 1) * page_bytes)``, so pages owned by different groups never overlap.

    A block sits in its page's file range at its slot index in the page times
    ``slot_bytes[g]``. In O_DIRECT mode a slot is the block padded to the
    driver alignment, so every file offset, length and staging buffer address
    is aligned; the pad bytes are never read back into GPU memory. Buffered
    I/O through the page cache uses dense slots.
    """

    large_page_bytes: int
    block_bytes: tuple[int, ...]
    slot_bytes: tuple[int, ...]
    page_bytes: int
    buf_slot_bytes: int

    def block_offset(self, group_id: int, block_id: int) -> int:
        block_bytes = self.block_bytes[group_id]
        page = large_page_of_block(block_id, self.large_page_bytes, block_bytes)
        first = large_page_block_range(page, self.large_page_bytes, block_bytes).start
        return page * self.page_bytes + (block_id - first) * self.slot_bytes[group_id]


def plan_disk_layout(
    large_page_bytes: int, block_bytes: Sequence[int], use_page_cache: bool
) -> DiskSlotLayout:
    """Derive the disk slot layout from the per-group block bytes.

    Callers sizing the pool (scheduler and worker) must use the returned
    ``page_bytes``: in O_DIRECT mode every slot is padded to the driver
    alignment, so the GPU page size would undercount the file size.
    """
    slot_bytes = tuple(
        b if use_page_cache else round_up(b, _ALIGNMENT) for b in block_bytes
    )
    # A page holds at most large_page_bytes // b blocks of a group.
    page_bytes = max(
        large_page_bytes // b * s for s, b in zip(slot_bytes, block_bytes, strict=True)
    )
    return DiskSlotLayout(
        large_page_bytes=large_page_bytes,
        block_bytes=tuple(block_bytes),
        slot_bytes=slot_bytes,
        page_bytes=page_bytes,
        buf_slot_bytes=max(slot_bytes),
    )


class DiskBackend:
    """Async disk offload backend with pipelined GPU DMA and interleaved IO.

    Architecture:
    - Separate coordinator threads for store and load (never block each other)
    - Interleaved pipeline: DMA slot N while pread/pwrite slot N-1
    - O_DIRECT by default; page cache is opt-in via use_page_cache

    Same launch_copy interface as DmaCopyBackend so the worker can swap
    backends without changing calling code.
    """

    def __init__(self) -> None:
        self._geometry: CopyGeometry | None = None
        self._layout: DiskSlotLayout | None = None
        self._store_driver: DriverMemcpyParams | None = None
        self._load_driver: DriverMemcpyParams | None = None
        self._load_stream: torch.cuda.Stream | None = None
        self._store_stream: torch.cuda.Stream | None = None
        self._store_queue: queue.SimpleQueue = queue.SimpleQueue()
        self._load_queue: queue.SimpleQueue = queue.SimpleQueue()
        self._store_thread: threading.Thread | None = None
        self._load_thread: threading.Thread | None = None
        self._shutdown: bool = False
        self._fd: int = -1
        self._disk_path: str = ""
        self._num_buffer_slots: int = 0
        self._store_buf: torch.Tensor | None = None
        self._load_buf: torch.Tensor | None = None
        self._store_views: list[memoryview] = []
        self._load_views: list[memoryview] = []

    def init(
        self,
        geometry: CopyGeometry,
        device: torch.device,
        load_stream: torch.cuda.Stream,
        store_stream: torch.cuda.Stream,
        disk_path: str,
        layout: DiskSlotLayout,
        num_pages: int,
        num_buffer_slots: int = 2,
        use_page_cache: bool = False,
    ) -> None:
        self._load_stream = load_stream
        self._store_stream = store_stream
        self._geometry = geometry
        self._layout = layout
        self._num_buffer_slots = num_buffer_slots

        # Stores read the live KV cache -> STREAM (paired with the compute-done
        # wait in get_finished); loads read stable pinned host memory -> ANY.
        self._store_driver = driver_memcpy_params(
            store_stream, CU_MEMCPY_SRC_ACCESS_ORDER_STREAM
        )
        self._load_driver = driver_memcpy_params(
            load_stream, CU_MEMCPY_SRC_ACCESS_ORDER_ANY
        )

        # Separate buffer pools for store and load threads.
        self._store_buf = _alloc_aligned(num_buffer_slots, self._layout.buf_slot_bytes)
        pin_tensor(self._store_buf)
        self._load_buf = _alloc_aligned(num_buffer_slots, self._layout.buf_slot_bytes)
        pin_tensor(self._load_buf)
        self._store_views = [
            memoryview(self._store_buf[slot].numpy())
            for slot in range(num_buffer_slots)
        ]
        self._load_views = [
            memoryview(self._load_buf[slot].numpy()) for slot in range(num_buffer_slots)
        ]

        os.makedirs(os.path.dirname(disk_path) or ".", exist_ok=True)
        # Slot contents never outlive the process, so unlink then O_EXCL rather
        # than reopening: a pre-existing file would otherwise keep its own
        # (possibly world-readable) mode, and blocks may encode user prompts.
        with contextlib.suppress(FileNotFoundError):
            os.unlink(disk_path)
        # O_DIRECT by default: page cache would consume the very host DRAM this
        # backend exists to conserve, and doubles the copy on the store path.
        flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
        if not use_page_cache:
            flags |= O_DIRECT
        self._fd = os.open(disk_path, flags, 0o600)
        self._disk_path = disk_path
        os.ftruncate(self._fd, num_pages * self._layout.page_bytes)

        logger.info(
            "DiskBackend: path=%s, pages=%d, total=%.2f GB, buf=%dx%d bytes"
            " (page_cache=%s)",
            disk_path,
            num_pages,
            (num_pages * self._layout.page_bytes) / (1024**3),
            num_buffer_slots,
            self._layout.buf_slot_bytes,
            use_page_cache,
        )

        self._store_thread = threading.Thread(
            target=self._store_loop,
            args=(device, store_stream),
            daemon=True,
        )
        self._load_thread = threading.Thread(
            target=self._load_loop,
            args=(device, load_stream),
            daemon=True,
        )
        self._store_thread.start()
        self._load_thread.start()

    def launch_copy(
        self,
        src_blocks: list[int],
        dst_blocks: list[int],
        group_ids: list[int],
        is_store: bool,
        event_idx: int,
        events_list: list[tuple[int, torch.Event]],
        wait_event: torch.Event | None = None,
    ) -> None:
        q = self._store_queue if is_store else self._load_queue
        q.put((src_blocks, dst_blocks, group_ids, event_idx, events_list, wait_event))

    def shutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        self._store_queue.put(None)
        self._load_queue.put(None)
        if self._store_thread is not None:
            self._store_thread.join(timeout=10.0)
        if self._load_thread is not None:
            self._load_thread.join(timeout=10.0)
        if self._fd < 0:
            return
        # Slot contents can encode user prompts, so drop the name now rather
        # than leaving them readable until the next run overwrites the file.
        # Unlinking only removes the directory entry: any thread still holding
        # the fd keeps writing to the (now anonymous) inode, which the kernel
        # frees once the last fd goes away.
        with contextlib.suppress(OSError):
            os.unlink(self._disk_path)
        # Closing under a still-running IO thread would let the fd number be
        # reused by an unrelated open(), turning its next pwrite into a write
        # into that file. Leaking one fd for the remaining process lifetime is
        # the cheaper failure.
        if any(
            t is not None and t.is_alive()
            for t in (self._store_thread, self._load_thread)
        ):
            logger.warning(
                "IO thread still running after shutdown timeout; leaking fd %d",
                self._fd,
            )
            return
        os.close(self._fd)
        self._fd = -1

    def _gpu_block_span(self, group_id: int, block_id: int) -> tuple[int, int]:
        """(address, nbytes) of one block's contiguous GPU content."""
        geometry = self._geometry
        assert geometry is not None
        block_bytes = int(geometry.block_bytes[group_id])
        return int(geometry.gpu_bases[group_id]) + block_id * block_bytes, block_bytes

    def _dma(
        self,
        dst_addr: int,
        src_addr: int,
        nbytes: int,
        driver: DriverMemcpyParams,
    ) -> None:
        launch_batch(
            np.asarray([dst_addr], dtype=np.uint64),
            np.asarray([src_addr], dtype=np.uint64),
            np.asarray([nbytes], dtype=np.uint64),
            driver,
        )

    def _store_loop(
        self,
        device: torch.device,
        stream: torch.cuda.Stream,
    ) -> None:
        current_platform.set_device(device)
        while True:
            item = self._store_queue.get()
            if item is None:
                return
            (src_blocks, dst_blocks, group_ids, event_idx, events_list, wait_event) = (
                item
            )
            if wait_event is not None:
                stream.wait_event(wait_event)
            self._do_store(src_blocks, dst_blocks, group_ids, stream)
            event = torch.Event()
            event.record(stream)
            events_list.append((event_idx, event))

    def _write_slot(self, buf_slot: int, file_offset: int, nbytes: int) -> None:
        written = os.pwritev(
            self._fd, [self._store_views[buf_slot][:nbytes]], file_offset
        )
        if written < nbytes:
            raise OSError(f"Short write: expected {nbytes} bytes, wrote {written}")

    def _read_slot(self, buf_slot: int, file_offset: int, nbytes: int) -> None:
        bytes_read = os.preadv(
            self._fd, [self._load_views[buf_slot][:nbytes]], file_offset
        )
        if bytes_read < nbytes:
            raise OSError(f"Short read: expected {nbytes} bytes, read {bytes_read}")

    def _load_loop(
        self,
        device: torch.device,
        stream: torch.cuda.Stream,
    ) -> None:
        current_platform.set_device(device)
        while True:
            item = self._load_queue.get()
            if item is None:
                return
            (src_blocks, dst_blocks, group_ids, event_idx, events_list, wait_event) = (
                item
            )
            if wait_event is not None:
                stream.wait_event(wait_event)
            self._do_load(src_blocks, dst_blocks, group_ids, stream)
            event = torch.Event()
            event.record(stream)
            events_list.append((event_idx, event))

    def _do_store(
        self,
        gpu_blocks: list[int],
        disk_blocks: list[int],
        group_ids: list[int],
        stream: torch.cuda.Stream,
    ) -> None:
        """GPU -> buffer (DMA) -> disk (pwrite), interleaved double-buffer."""
        assert self._store_driver is not None
        assert self._layout is not None and self._store_buf is not None
        geometry = self._geometry
        assert geometry is not None
        n = self._num_buffer_slots
        # (DMA event, file offset, length) of the block staged in each slot.
        pending: list[tuple[torch.Event, int, int] | None] = [None] * n

        i = 0
        for gpu_blk, disk_blk, g in zip(gpu_blocks, disk_blocks, group_ids):
            if not geometry.has_region[g]:
                continue  # PP: this rank holds no layer of the group.
            buf_slot = i % n
            prev = pending[buf_slot]
            if prev is not None:
                prev[0].synchronize()
                self._write_slot(buf_slot, prev[1], prev[2])

            src_addr, nbytes = self._gpu_block_span(g, gpu_blk)
            dst_addr = (
                self._store_buf.data_ptr() + buf_slot * self._layout.buf_slot_bytes
            )
            self._dma(dst_addr, src_addr, nbytes, self._store_driver)
            ev = torch.Event()
            ev.record(stream)
            pending[buf_slot] = (
                ev,
                self._layout.block_offset(g, disk_blk),
                self._layout.slot_bytes[g],
            )
            i += 1

        for slot, last in enumerate(pending):
            if last is not None:
                last[0].synchronize()
                self._write_slot(slot, last[1], last[2])

    def _do_load(
        self,
        disk_blocks: list[int],
        gpu_blocks: list[int],
        group_ids: list[int],
        stream: torch.cuda.Stream,
    ) -> None:
        """Disk (pread) -> buffer -> GPU (DMA), interleaved double-buffer."""
        assert self._load_driver is not None
        assert self._layout is not None and self._load_buf is not None
        geometry = self._geometry
        assert geometry is not None
        n = self._num_buffer_slots
        prev_dma_events: list[torch.Event | None] = [None] * n

        i = 0
        for disk_blk, gpu_blk, g in zip(disk_blocks, gpu_blocks, group_ids):
            if not geometry.has_region[g]:
                continue  # PP: this rank holds no layer of the group.
            buf_slot = i % n
            prev = prev_dma_events[buf_slot]
            if prev is not None:
                prev.synchronize()

            self._read_slot(
                buf_slot,
                self._layout.block_offset(g, disk_blk),
                self._layout.slot_bytes[g],
            )
            dst_addr, nbytes = self._gpu_block_span(g, gpu_blk)
            src_addr = (
                self._load_buf.data_ptr() + buf_slot * self._layout.buf_slot_bytes
            )
            self._dma(dst_addr, src_addr, nbytes, self._load_driver)
            ev = torch.Event()
            ev.record(stream)
            prev_dma_events[buf_slot] = ev
            i += 1

        for ev in prev_dma_events:
            if ev is not None:
                ev.synchronize()
