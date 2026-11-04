# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side handler for SimpleCPUOffloadConnector."""

from typing import TYPE_CHECKING

import numpy as np
import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.utils import group_pool_storage
from vllm.logger import init_logger
from vllm.utils.torch_utils import PIN_MEMORY
from vllm.v1.simple_kv_offload.copy_backend import DmaCopyBackend
from vllm.v1.simple_kv_offload.cuda_mem_ops import CopyGeometry, pin_tensor
from vllm.v1.simple_kv_offload.disk_backend import DiskBackend, plan_disk_layout
from vllm.v1.simple_kv_offload.metadata import (
    SimpleCPUOffloadMetadata,
    SimpleCPUOffloadWorkerMetadata,
)
from vllm.v1.simple_kv_offload.sizing import sync_num_offload_blocks_across_workers

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)


class SimpleCPUOffloadWorker:
    """Worker-side handler for CPU offloading transfers."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: "KVCacheConfig | None",
        cpu_capacity_bytes: int,
        kv_offload_backend: str = "cpu",
        disk_path: str | None = None,
        disk_capacity_bytes: int = 0,
        disk_buffer_slots: int = 2,
        use_page_cache: bool = False,
        num_cpu_blocks: int | None = None,
    ):
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        self.cpu_capacity_bytes = cpu_capacity_bytes
        self._num_cpu_blocks_override = num_cpu_blocks
        self.disk_path = disk_path
        self.disk_capacity_bytes = disk_capacity_bytes
        self.disk_buffer_slots = disk_buffer_slots
        self.use_page_cache = use_page_cache
        self.disk_mode = kv_offload_backend == "disk"

        self.device: torch.device | None = None
        self.num_cpu_blocks: int = 0

        # CUDA streams for the async transfers
        self.load_stream: torch.cuda.Stream | None = None
        self.store_stream: torch.cuda.Stream | None = None

        self._backend: DmaCopyBackend | DiskBackend | None = None

        # Ordered (event_idx, Event). Events pre-allocated on main thread.
        self._load_events: list[tuple[int, torch.Event]] = []
        self._store_events: list[tuple[int, torch.Event]] = []
        # High-water marks: highest event_idx completed per stream.
        # When the event list is empty, the hwm covers all prior events.
        self._load_hwm: int = -1
        self._store_hwm: int = -1

        # Metadata for the current step
        self._connector_metadata: SimpleCPUOffloadMetadata | None = None

        # Compute-done event recorded before each store; reused across steps
        # (get_finished runs once per step, copy queue is FIFO).
        self._store_compute_done: torch.Event | None = None

        # Pending event index sets, populated in bind_connector_metadata
        self._pending_load_event_indices: set[int] = set()
        self._pending_store_event_indices: set[int] = set()
        # Completed store events to report via build_connector_worker_meta
        self._completed_store_events: dict[int, int] = {}

        # The CPU arena (CPU backend only).
        self._cpu_arena: torch.Tensor | None = None

    def register_kv_caches(
        self,
        kv_caches: dict[str, torch.Tensor],
    ) -> None:
        """Register GPU KV caches and allocate the offload target.

        Every layer view is a stride into one backing allocation. A manager
        block of group g is one contiguous byte range of ``block_bytes_g`` at
        ``base_g + b * block_bytes_g``. CPU blocks live at the same offsets in
        one pinned arena of GPU-sized large pages; the disk backend stages
        blocks through pinned buffers into the slot layout on disk.

        Args:
            kv_caches: Per-layer GPU KV caches. Each group's layers share one
                backing storage; only the base address of each group is used.

        """
        if not kv_caches:
            logger.warning("No KV caches to offload.")
            return

        self.device = next(iter(kv_caches.values())).device

        config = self.kv_cache_config
        assert config is not None
        groups = config.kv_cache_groups

        gpu_bases = np.zeros(len(groups), dtype=np.int64)
        has_region = np.zeros(len(groups), dtype=bool)
        for group_id in range(len(groups)):
            storage = group_pool_storage(kv_caches, config, group_id)
            if storage is None:
                continue  # PP: this rank holds no layer of the group.
            gpu_bases[group_id] = storage.data_ptr()
            has_region[group_id] = True

        block_bytes = np.array(
            [config.group_geometry(g).block_bytes for g in range(len(groups))],
            dtype=np.int64,
        )
        if self.disk_mode:
            # O_DIRECT pads every slot to the driver alignment (see
            # plan_disk_layout); sizing with the GPU page would make the
            # backing file outgrow disk_capacity_bytes.
            layout = plan_disk_layout(
                config.large_page_bytes, block_bytes.tolist(), self.use_page_cache
            )
            page_bytes = layout.page_bytes
        else:
            page_bytes = config.large_page_bytes
        capacity = (
            self.disk_capacity_bytes if self.disk_mode else self.cpu_capacity_bytes
        )
        self.num_cpu_blocks = self._resolve_num_cpu_blocks(capacity, page_bytes)

        # Use lowest priority so KV cache I/O yields to compute streams.
        low_pri, _ = torch.cuda.Stream.priority_range()
        self.load_stream = torch.cuda.Stream(priority=low_pri)
        self.store_stream = torch.cuda.Stream(priority=low_pri)

        geometry = CopyGeometry(
            gpu_bases=gpu_bases,
            has_region=has_region,
            cpu_base=0,
            block_bytes=block_bytes,
        )

        if self.disk_mode:
            logger.info(
                "SimpleCPUOffloadWorker [DISK]: %d groups, %d disk pages (%.2f GB)",
                int(has_region.sum()),
                self.num_cpu_blocks,
                (self.num_cpu_blocks * page_bytes) / (1024**3),
            )
            assert self.disk_path is not None
            rank_path = f"{self.disk_path}.rank_{self.device.index or 0}"
            self._backend = DiskBackend()
            self._backend.init(
                geometry,
                self.device,
                self.load_stream,
                self.store_stream,
                rank_path,
                layout,
                self.num_cpu_blocks,
                self.disk_buffer_slots,
                self.use_page_cache,
            )
            return

        logger.info(
            "SimpleCPUOffloadWorker [CPU]: %d groups, %d CPU pages (%.2f GB)",
            int(has_region.sum()),
            self.num_cpu_blocks,
            (self.num_cpu_blocks * page_bytes) / (1024**3),
        )

        self._cpu_arena = torch.zeros(
            self.num_cpu_blocks * page_bytes, dtype=torch.uint8, device="cpu"
        )
        if PIN_MEMORY:
            pin_tensor(self._cpu_arena)

        self._backend = DmaCopyBackend()
        self._backend.init(
            geometry._replace(cpu_base=self._cpu_arena.data_ptr()),
            self.device,
            self.load_stream,
            self.store_stream,
        )

    def _resolve_num_cpu_blocks(self, capacity_bytes: int, bytes_per_block: int) -> int:
        local_capacity = capacity_bytes // bytes_per_block
        requested = self._num_cpu_blocks_override
        if requested is None:
            return sync_num_offload_blocks_across_workers(max(1, local_capacity))
        if type(requested) is not int or not 0 < requested <= local_capacity:
            raise ValueError(
                f"num_cpu_blocks={requested!r} must be between 1 and "
                f"local offload capacity {local_capacity}"
            )
        return requested

    def bind_connector_metadata(self, metadata: SimpleCPUOffloadMetadata) -> None:
        self._connector_metadata = metadata
        if metadata.load_event >= 0:
            self._pending_load_event_indices.add(metadata.load_event)
        if metadata.store_event >= 0:
            self._pending_store_event_indices.add(metadata.store_event)

    def clear_connector_metadata(self) -> None:
        self._connector_metadata = None

    def start_load_kv(self) -> None:
        # Invoked after the forward launch on steps without sync loads
        # (SchedulerOutput.has_sync_kv_loads), so the CPU-side block copy
        # op overhead (~5ms) stays hidden behind GPU compute. Stores are
        # issued in wait_for_save().
        metadata = self._connector_metadata
        if metadata is not None and metadata.load_cpu_blocks:
            backend = self._backend
            assert backend is not None
            backend.launch_copy(
                metadata.load_cpu_blocks,
                metadata.load_gpu_blocks,
                metadata.load_group_ids,
                is_store=False,
                event_idx=metadata.load_event,
                events_list=self._load_events,
            )

    def wait_for_save(self) -> None:
        """Submit async stores.

        Stores (GPU->CPU) read the live KV cache, which the compute stream may
        still be writing under v1 overlapped execution, so they are ordered
        after a compute-done event recorded on the current stream. See
        #45704 for the bug and #39306 for the srcAccessOrder rationale.
        """
        metadata = self._connector_metadata
        if metadata is not None and metadata.store_gpu_blocks:
            backend = self._backend
            assert backend is not None
            if self._store_compute_done is None:
                self._store_compute_done = torch.Event()
            self._store_compute_done.record(torch.cuda.current_stream())
            backend.launch_copy(
                metadata.store_gpu_blocks,
                metadata.store_cpu_blocks,
                metadata.store_group_ids,
                is_store=True,
                event_idx=metadata.store_event,
                events_list=self._store_events,
                wait_event=self._store_compute_done,
            )

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        """Report completed transfer events to the scheduler.

        Loads are submitted in start_load_kv() and stores in wait_for_save().

        Returns:
            tuple of (finished_sending, finished_recving).
            - finished_sending: always None (stores use worker metadata).
            - finished_recving: req_ids whose loads have completed.

        """
        metadata = self._connector_metadata
        finished_recving: set[str] = set()

        if self._pending_load_event_indices:
            load_wm = self._poll_stream_events(is_store=False)
            for j in [j for j in self._pending_load_event_indices if j <= load_wm]:
                self._pending_load_event_indices.discard(j)
                req_ids = (
                    metadata.load_event_to_reqs.get(j) if metadata is not None else None
                )
                if req_ids:
                    finished_recving.update(req_ids)

        if self._pending_store_event_indices:
            store_wm = self._poll_stream_events(is_store=True)
            for j in [j for j in self._pending_store_event_indices if j <= store_wm]:
                self._pending_store_event_indices.discard(j)
                self._completed_store_events[j] = 1

        return None, finished_recving or None

    def build_connector_worker_meta(self) -> SimpleCPUOffloadWorkerMetadata | None:
        """Return completed store events since the last call."""
        if not self._completed_store_events:
            return None
        meta = SimpleCPUOffloadWorkerMetadata(
            completed_store_events=self._completed_store_events,
        )
        self._completed_store_events = {}
        return meta

    def handle_preemptions(
        self, kv_connector_metadata: SimpleCPUOffloadMetadata
    ) -> None:
        """Sync all in-flight transfers before preempted blocks are reused."""
        if not kv_connector_metadata.need_flush:
            return
        self._flush_and_sync_all()

    def _flush_and_sync_all(self) -> None:
        """Synchronize all in-flight transfer events."""
        for event_idx, event in self._load_events:
            event.synchronize()
            self._load_hwm = event_idx
        self._load_events.clear()

        for event_idx, event in self._store_events:
            event.synchronize()
            self._store_hwm = event_idx
        self._store_events.clear()

    def _poll_stream_events(self, is_store: bool) -> int:
        """Non-blocking poll for completed events and return the high-water mark."""
        events = self._store_events if is_store else self._load_events
        hwm = self._store_hwm if is_store else self._load_hwm
        while events:
            event_idx, event = events[0]
            if not event.query():
                break
            hwm = event_idx
            events.pop(0)
        if is_store:
            self._store_hwm = hwm
        else:
            self._load_hwm = hwm
        return hwm
