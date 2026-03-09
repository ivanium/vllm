# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side handler for SimpleCPUOffloadConnector."""

from typing import TYPE_CHECKING, Any

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload.backends import (
    CudaTransferBackend,
    TransferBackend,
)
from vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload.metadata import (
    SimpleCPUOffloadMetadata,
)
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)


class SimpleCPUOffloadWorker:
    """Worker-side handler for CPU offloading transfers."""

    def __init__(
        self,
        vllm_config: "VllmConfig",
        kv_cache_config: "KVCacheConfig | None",
        capacity_bytes: int,
        backend: TransferBackend,
    ):
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        self.capacity_bytes = capacity_bytes
        self.backend = backend

        # Ordered (event_idx, event) -- stream ordering lets us break early
        self._load_events: list[tuple[int, Any]] = []
        self._store_events: list[tuple[int, Any]] = []

        # Deferred stores: queued in wait_for_save(), flushed in start_load_kv()
        self._pending_store_events: list[tuple[int, list[int], list[int]]] = []
        self._connector_metadata: SimpleCPUOffloadMetadata | None = None

        # Pending event index sets, populated in bind_connector_metadata
        self._pending_load_event_indices: set[int] = set()
        self._pending_store_event_indices: set[int] = set()

    @property
    def _is_initialized(self) -> bool:
        """Whether KV caches are registered and ready for transfers."""
        return self.backend.is_initialized

    def register_kv_caches(
        self,
        kv_caches: dict[str, torch.Tensor],
        kv_cache_raw_tensors: dict[str, torch.Tensor] | None = None,
    ) -> None:
        """Register GPU KV caches and set up the transfer backend.

        Args:
            kv_caches: Reshaped per-layer GPU KV caches.
            kv_cache_raw_tensors: Raw int8 tensors before reshape. If provided,
                used for transfers (HMA-safe). Falls back to kv_caches if None.
        """
        raw = kv_cache_raw_tensors if kv_cache_raw_tensors is not None else kv_caches

        # Deduplicate shared tensors (multiple layers may share the same tensor)
        seen_ptrs: dict[int, tuple[str, torch.Tensor]] = {}
        for name, tensor in raw.items():
            ptr = tensor.data_ptr()
            if ptr not in seen_ptrs:
                seen_ptrs[ptr] = (name, tensor)

        # Build ordered dict of unique raw tensors for the backend
        unique_gpu_caches: dict[str, torch.Tensor] = {}
        for name, tensor in seen_ptrs.values():
            # Raw tensors are 1D int8. Reshape to [num_blocks, page_size_bytes]
            # so stride(0) gives page_size_bytes for the Triton kernel.
            if tensor.dim() == 1:
                assert self.kv_cache_config is not None
                num_blocks = self.kv_cache_config.num_blocks
                tensor = tensor.view(num_blocks, -1)
            unique_gpu_caches[name] = tensor

        self.backend.setup(unique_gpu_caches, self.capacity_bytes, self.kv_cache_config)

    def bind_connector_metadata(self, metadata: SimpleCPUOffloadMetadata) -> None:
        self._connector_metadata = metadata
        if metadata.load_event >= 0:
            self._pending_load_event_indices.add(metadata.load_event)
        if metadata.store_event >= 0:
            self._pending_store_event_indices.add(metadata.store_event)

    def clear_connector_metadata(self) -> None:
        self._connector_metadata = None

    def start_load_kv(self) -> None:
        """Flush deferred stores, then start async loads from CPU to GPU."""
        if not self._is_initialized:
            logger.warning("KV caches not registered, skipping load")
            return

        self._submit_pending_stores()

        if self._connector_metadata is None:
            return

        metadata = self._connector_metadata
        if not metadata.load_gpu_blocks:
            return

        backend = self.backend
        if isinstance(backend, CudaTransferBackend):
            assert backend.load_stream is not None
            with torch.cuda.stream(backend.load_stream):
                backend.copy_blocks(
                    metadata.load_cpu_blocks,
                    metadata.load_gpu_blocks,
                    is_store=False,
                )
                event = backend.record_event()
        else:
            backend.copy_blocks(
                metadata.load_cpu_blocks,
                metadata.load_gpu_blocks,
                is_store=False,
            )
            event = backend.record_event()

        self._load_events.append((metadata.load_event, event))
        logger.debug(
            "Started loading %d blocks from CPU (event_idx=%d)",
            len(metadata.load_gpu_blocks),
            metadata.load_event,
        )

    def wait_for_save(self) -> None:
        """Queue store events; actual submission deferred to start_load_kv()."""
        if self._connector_metadata is None:
            return

        if not self._is_initialized:
            return

        metadata = self._connector_metadata
        if not metadata.store_gpu_blocks:
            return

        self._pending_store_events.append(
            (metadata.store_event, metadata.store_gpu_blocks, metadata.store_cpu_blocks)
        )
        logger.debug(
            "Queued storing %d blocks to CPU (event_idx=%d)",
            len(metadata.store_gpu_blocks),
            metadata.store_event,
        )

    def _submit_pending_stores(self) -> None:
        if not self._pending_store_events or not self._is_initialized:
            return

        all_src: list[int] = []
        all_dst: list[int] = []
        for _, src, dst in self._pending_store_events:
            all_src.extend(src)
            all_dst.extend(dst)

        backend = self.backend
        if isinstance(backend, CudaTransferBackend):
            assert backend.store_stream is not None
            with torch.cuda.stream(backend.store_stream):
                if all_src:
                    backend.copy_blocks(all_src, all_dst, is_store=True)
                event = backend.record_event()
        else:
            if all_src:
                backend.copy_blocks(all_src, all_dst, is_store=True)
            event = backend.record_event()

        for event_idx, _, _ in self._pending_store_events:
            self._store_events.append((event_idx, event))
            logger.debug("Submitted deferred store to CPU (event_idx=%d)", event_idx)

        self._pending_store_events.clear()

    def get_finished(
        self,
        finished_req_ids: set[str],
    ) -> tuple[set[str] | None, set[str] | None]:
        """Updates from worker to scheduler on completed transfer events.

        Returns:
            tuple of (finished_sending, finished_recving).
            - finish_sending is only used by connector scheduler, and we use it to
            store the finished store event ids rather than req_ids.
            - finished_recving still tracks the req_ids that have finished loading.
        """
        finished_recving: set[str] = set()
        finished_sending: set[str] = set()

        load_wm = self._drain_stream_events(self._load_events)
        meta = self._connector_metadata
        for j in [j for j in self._pending_load_event_indices if j <= load_wm]:
            req_ids = meta.load_event_to_reqs.get(j) if meta is not None else None
            if req_ids:
                finished_recving.update(req_ids)
                self._pending_load_event_indices.discard(j)

        store_wm = self._drain_stream_events(self._store_events)

        for j in [j for j in self._pending_store_event_indices if j <= store_wm]:
            self._pending_store_event_indices.discard(j)
            finished_sending.add(f"__store_done_{j}")

        return finished_sending or None, finished_recving or None

    def _drain_stream_events(self, events: list[tuple[int, Any]]) -> int:
        watermark = -1
        while events:
            event_idx, event = events[0]
            if self.backend.query_event(event):
                watermark = event_idx
                events.pop(0)
            else:
                break  # Stream ordering: nothing after this can have fired.
        return watermark

    def handle_preemptions(self) -> None:
        """Sync all in-flight transfers before preempted blocks are reused."""
        self._submit_pending_stores()

        for _, event in self._load_events:
            self.backend.sync_event(event)
        self._load_events.clear()

        for _, event in self._store_events:
            self.backend.sync_event(event)
        self._store_events.clear()
