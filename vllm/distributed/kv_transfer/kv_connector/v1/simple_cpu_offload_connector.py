# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
SimpleCPUOffloadConnector: A minimal, efficient CPU offloading connector.

This connector provides:
- Triton kernel-based GPU<->CPU block transfers
- LRU eviction using BlockPool for CPU block management
- Scheduler-side hash->CPU_block_id mapping
- Async transfers with CUDA streams
- Hybrid KV cache manager support (SupportsHMA)
"""

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_events import KVCacheEvent
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload.manager import (
    SimpleCPUOffloadScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload.metadata import (
    SimpleCPUOffloadMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload.worker import (
    SimpleCPUOffloadWorker,
)
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import KVConnectorOutput

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
    from vllm.v1.attention.backend import AttentionBackend, AttentionMetadata
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)

# Default CPU capacity: 8 GB
DEFAULT_CPU_CAPACITY_BYTES = 8 * (1024**3)


class SimpleCPUOffloadConnector(KVConnectorBase_V1, SupportsHMA):
    """
    Simple CPU offloading connector with Triton kernel transfers.

    This connector offloads KV cache blocks to CPU memory using:
    - Triton kernels for efficient GPU<->CPU transfers
    - BlockPool for LRU-based CPU block management
    - Scheduler-side hash mapping for cache lookup
    - Async CUDA streams for non-blocking transfers

    Features:
    - Minimal code, maximum reuse of existing infrastructure
    - Async transfers overlap with model computation
    - Pinned CPU memory for high bandwidth
    """

    @property
    def prefer_cross_layer_blocks(self) -> bool:
        """Use per-layer blocks for standard KV cache layout."""
        return False

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig | None" = None,
    ):
        """
        Initialize the SimpleCPUOffloadConnector.

        Args:
            vllm_config: vLLM configuration
            role: SCHEDULER or WORKER role
            kv_cache_config: KV cache configuration
        """
        super().__init__(vllm_config, role, kv_cache_config)

        extra_config = self._kv_transfer_config.kv_connector_extra_config or {}
        cpu_capacity_bytes = int(
            extra_config.get("cpu_bytes_to_use", DEFAULT_CPU_CAPACITY_BYTES)
        )
        offload_decode_blocks = bool(extra_config.get("offload_decode_blocks", False))

        logger.info(
            "CPUOffloadConnector: Initializing with role=%s, cpu_capacity=%.2f GB",
            role.name,
            cpu_capacity_bytes / (1024**3),
        )

        # Role-specific initialization
        self.scheduler_manager: SimpleCPUOffloadScheduler | None = None
        self.worker_handler: SimpleCPUOffloadWorker | None = None

        if role == KVConnectorRole.SCHEDULER:
            self.scheduler_manager = SimpleCPUOffloadScheduler(
                vllm_config,
                kv_cache_config,
                cpu_capacity_bytes,
                offload_decode_blocks=offload_decode_blocks,
            )
        elif role == KVConnectorRole.WORKER:
            self.worker_handler = SimpleCPUOffloadWorker(
                vllm_config,
                kv_cache_config,
                cpu_capacity_bytes,
            )

    # ==============================
    # Worker-side methods
    # ==============================

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        """Register per-layer KV caches with the connector."""
        if self.worker_handler is not None:
            self.worker_handler.register_kv_caches(kv_caches)

    def register_cross_layers_kv_cache(
        self,
        kv_cache: torch.Tensor,
        attn_backend: type["AttentionBackend"],
    ) -> None:
        """Register cross-layer KV cache tensor."""
        if self.worker_handler is not None:
            self.worker_handler.register_cross_layers_kv_cache(kv_cache, attn_backend)

    def bind_connector_metadata(
        self,
        connector_metadata: KVConnectorMetadata,
    ) -> None:
        """Bind connector metadata for the current step."""
        super().bind_connector_metadata(connector_metadata)
        if self.worker_handler is not None:
            assert isinstance(connector_metadata, SimpleCPUOffloadMetadata)
            self.worker_handler.bind_connector_metadata(connector_metadata)

    def clear_connector_metadata(self) -> None:
        """Clear connector metadata after the step."""
        super().clear_connector_metadata()
        if self.worker_handler is not None:
            self.worker_handler.clear_connector_metadata()

    def handle_preemptions(self, preempted_req_ids: set[str]) -> None:
        """Handle preempted requests before their blocks are overwritten."""
        if self.worker_handler is not None:
            self.worker_handler.handle_preemptions(preempted_req_ids)

    def start_load_kv(
        self,
        forward_context: "ForwardContext",
        **kwargs: Any,
    ) -> None:
        """Start async loading KV cache from CPU to GPU."""
        if self.worker_handler is not None:
            self.worker_handler.start_load_kv()

    def wait_for_layer_load(self, layer_name: str) -> None:
        """Block until the KV for a specific layer is loaded."""
        if self.worker_handler is not None:
            self.worker_handler.wait_for_layer_load(layer_name)

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> None:
        """
        Save KV cache for a layer.

        Note: For cross-layer cache, saving happens in wait_for_save()
        after all layers are computed.
        """
        # Not used for cross-layer cache
        pass

    def wait_for_save(self) -> None:
        """Start async storing KV cache from GPU to CPU."""
        if self.worker_handler is not None:
            self.worker_handler.wait_for_save()

    def get_finished(
        self,
        finished_req_ids: set[str],
    ) -> tuple[set[str] | None, set[str] | None]:
        """Get request IDs that have finished async transfers."""
        if self.worker_handler is not None:
            return self.worker_handler.get_finished(finished_req_ids)
        return None, None

    # ==============================
    # Scheduler-side methods
    # ==============================

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        """Get number of tokens that can be loaded from CPU cache."""
        if self.scheduler_manager is not None:
            return self.scheduler_manager.get_num_new_matched_tokens(
                request, num_computed_tokens
            )
        return 0, False

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        """Update connector state after GPU block allocation."""
        if self.scheduler_manager is not None:
            self.scheduler_manager.update_state_after_alloc(
                request, blocks, num_external_tokens
            )

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        """Build connector metadata for this step."""
        if self.scheduler_manager is not None:
            return self.scheduler_manager.build_connector_meta(scheduler_output)
        return SimpleCPUOffloadMetadata()

    def update_connector_output(
        self,
        connector_output: KVConnectorOutput,
    ) -> None:
        """Update connector state from worker output."""
        if self.scheduler_manager is not None:
            self.scheduler_manager.update_connector_output(connector_output)

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        """Called when a request has finished."""
        if self.scheduler_manager is not None:
            return self.scheduler_manager.request_finished(request, block_ids)
        return False, None

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        """Called when a request has finished for all KV cache groups."""
        if self.scheduler_manager is not None:
            return self.scheduler_manager.request_finished_all_groups(
                request, block_ids
            )
        return False, None

    def take_events(self) -> Iterable[KVCacheEvent]:
        """Return KV cache events for telemetry."""
        if self.scheduler_manager is not None:
            return self.scheduler_manager.take_events()
        return []
