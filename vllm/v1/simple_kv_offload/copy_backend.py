# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DMA copy backend for GPU<->CPU block transfers."""

from __future__ import annotations

import queue
import threading
import time

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.simple_kv_offload.cuda_mem_ops import (
    BatchMemcpyParams,
    build_params,
    copy_blocks,
)

logger = init_logger(__name__)


class DmaCopyBackend:
    """cuMemcpyBatchAsync copy backend (background thread)."""

    def __init__(self) -> None:
        self._store_params: BatchMemcpyParams | None = None
        self._load_params: BatchMemcpyParams | None = None
        self._load_stream: torch.cuda.Stream | None = None
        self._store_stream: torch.cuda.Stream | None = None
        self._queue: queue.SimpleQueue | None = None
        self._thread: threading.Thread | None = None
        self._shutdown: bool = False
        # Transfer stats: list of (num_bytes, duration_s, is_store)
        self._transfer_stats: list[tuple[int, float, bool]] = []
        self._stats_lock = threading.Lock()

    def init(
        self,
        gpu_caches: dict[str, torch.Tensor],
        cpu_caches: dict[str, torch.Tensor],
        device: torch.device,
        load_stream: torch.cuda.Stream,
        store_stream: torch.cuda.Stream,
    ) -> None:
        self._load_stream = load_stream
        self._store_stream = store_stream

        self._store_params = build_params(gpu_caches, cpu_caches, store_stream)
        self._load_params = build_params(cpu_caches, gpu_caches, load_stream)

        self._queue = queue.SimpleQueue()
        self._thread = threading.Thread(
            target=self._copy_loop,
            args=(self._queue, device, load_stream, store_stream,
                  self._transfer_stats, self._stats_lock),
            daemon=True,
        )
        self._thread.start()

    def drain_transfer_stats(self) -> list[tuple[int, float, bool]]:
        """Return and clear collected transfer stats."""
        with self._stats_lock:
            stats = self._transfer_stats
            self._transfer_stats = []
            return stats

    def launch_copy(
        self,
        src_blocks: list[int],
        dst_blocks: list[int],
        is_store: bool,
        event_idx: int,
        events_list: list[tuple[int, torch.Event]],
    ) -> None:
        params = self._store_params if is_store else self._load_params
        assert params is not None and self._queue is not None
        self._queue.put(
            (src_blocks, dst_blocks, params, is_store, event_idx, events_list)
        )

    def shutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        if self._queue is not None:
            self._queue.put(None)
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    @staticmethod
    def _copy_loop(
        q: queue.SimpleQueue,
        device: torch.device,
        load_stream: torch.cuda.Stream,
        store_stream: torch.cuda.Stream,
        transfer_stats: list[tuple[int, float, bool]],
        stats_lock: threading.Lock,
    ) -> None:
        current_platform.set_device(device)
        while True:
            item = q.get()
            if item is None:
                return
            src_blocks, dst_blocks, params, is_store, event_idx, events_list = item
            num_bytes = int(sum(params.bpb)) * len(src_blocks)
            t0 = time.perf_counter()
            copy_blocks(src_blocks, dst_blocks, params)
            elapsed = time.perf_counter() - t0
            with stats_lock:
                transfer_stats.append((num_bytes, elapsed, is_store))
            stream = store_stream if is_store else load_stream
            event = torch.Event()
            event.record(stream)
            events_list.append((event_idx, event))
