# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterator
from typing import TYPE_CHECKING

import torch

from vllm.config import VllmConfig
from vllm.platforms import current_platform
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.kv_offload.abstract import LoadStoreSpec, OffloadingManager
from vllm.v1.kv_offload.arc_manager import ARCOffloadingManager
from vllm.v1.kv_offload.backends.cpu import CPUBackend
from vllm.v1.kv_offload.hybrid_manager import HybridOffloadingManager
from vllm.v1.kv_offload.lru_manager import LRUOffloadingManager
from vllm.v1.kv_offload.mediums import CPULoadStoreSpec, GPULoadStoreSpec
from vllm.v1.kv_offload.spec import OffloadingSpec
from vllm.v1.kv_offload.worker.cpu_gpu import CpuGpuOffloadingHandlers
from vllm.v1.kv_offload.worker.hybrid_cpu_gpu import HybridCpuGpuOffloadingHandlers
from vllm.v1.kv_offload.worker.worker import OffloadingHandler

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheConfig


class CPUOffloadingSpec(OffloadingSpec):
    def __init__(self, vllm_config: VllmConfig):
        super().__init__(vllm_config)

        num_cpu_blocks = self.extra_config.get("num_cpu_blocks")
        if not num_cpu_blocks:
            raise Exception(
                "num_cpu_blocks must be specified in kv_connector_extra_config"
            )
        self.num_cpu_blocks: int = num_cpu_blocks

        # scheduler-side
        self._manager: OffloadingManager | None = None
        self._hybrid_manager: HybridOffloadingManager | None = None

        # worker-side
        self._handlers: CpuGpuOffloadingHandlers | None = None
        self._hybrid_handlers: HybridCpuGpuOffloadingHandlers | None = None

        self.eviction_policy: str = self.extra_config.get("eviction_policy", "lru")

    def _get_enable_events(self) -> bool:
        """Check if KV cache events are enabled."""
        kv_events_config = self.vllm_config.kv_events_config
        return kv_events_config is not None and kv_events_config.enable_kv_cache_events

    def _create_single_manager(self, enable_events: bool) -> OffloadingManager:
        """Create a single OffloadingManager with CPU backend."""
        backend = CPUBackend(
            block_size=self.offloaded_block_size, num_blocks=self.num_cpu_blocks
        )
        return self._create_manager_from_backend(backend, enable_events)

    def _create_manager_from_backend(
        self, backend: CPUBackend, enable_events: bool
    ) -> OffloadingManager:
        """Create an OffloadingManager with the specified backend."""
        if self.eviction_policy == "lru":
            return LRUOffloadingManager(backend=backend, enable_events=enable_events)
        elif self.eviction_policy == "arc":
            return ARCOffloadingManager(backend=backend, enable_events=enable_events)
        else:
            raise ValueError(
                f"Unknown eviction policy: {self.eviction_policy}. "
                f"Supported policies: lru, arc"
            )

    def get_manager(self, num_groups: int = 1) -> OffloadingManager:
        """
        Get an OffloadingManager for the specified number of KV cache groups.

        Args:
            num_groups: Number of KV cache groups (default 1 for non-HMA models).
                       Use > 1 for HMA models with multiple cache types.

        Returns:
            HybridOffloadingManager that handles per-group tracking.
            Even for single group, returns HybridOffloadingManager for
            consistent interface (accepts group_id parameter).
        """
        enable_events = self._get_enable_events()

        # Always use HybridOffloadingManager for consistent interface
        # (all manager calls accept group_id parameter)
        if not self._hybrid_manager:

            def backend_factory(group_id: int) -> CPUBackend:
                return CPUBackend(
                    block_size=self.offloaded_block_size,
                    num_blocks=self.num_cpu_blocks,
                )

            def manager_factory(
                backend: CPUBackend, enable_events: bool
            ) -> OffloadingManager:
                return self._create_manager_from_backend(backend, enable_events)

            self._hybrid_manager = HybridOffloadingManager(
                num_groups=num_groups,
                backend_factory=backend_factory,
                manager_factory=manager_factory,
                enable_events=enable_events,
            )
        return self._hybrid_manager

    def get_handlers(
        self,
        kv_caches: dict[str, torch.Tensor],
        attn_backends: dict[str, type[AttentionBackend]],
        kv_cache_config: "KVCacheConfig | None" = None,
    ) -> Iterator[
        tuple[type[LoadStoreSpec], type[LoadStoreSpec], OffloadingHandler]
        | tuple[type[LoadStoreSpec], type[LoadStoreSpec], OffloadingHandler, int]
    ]:
        """
        Get offloading handlers along with their respective src and dst types.

        Args:
            kv_caches: A dictionary of layer_name -> gpu_kv_cache tensor.
            attn_backends: A dictionary of layer_name -> AttentionBackend.
            kv_cache_config: Optional KV cache config for HMA support.
                            When provided with multiple groups, yields 4-tuples
                            (src_type, dst_type, handler, group_id).

        Yields:
            3-tuples (src_type, dst_type, handler) for single-group mode.
            4-tuples (src_type, dst_type, handler, group_id) for HMA mode.
        """
        if not current_platform.is_cuda_alike():
            raise Exception(
                "CPU Offloading is currently only supported on CUDA-alike GPUs"
            )

        # Check if HMA mode (multiple groups)
        num_groups = len(kv_cache_config.kv_cache_groups) if kv_cache_config else 1

        if num_groups <= 1 or kv_cache_config is None:
            # Original single-group path (backward compatible)
            if not self._handlers:
                self._handlers = CpuGpuOffloadingHandlers(
                    attn_backends=attn_backends,
                    gpu_block_size=self.gpu_block_size,
                    cpu_block_size=self.offloaded_block_size,
                    num_cpu_blocks=self.num_cpu_blocks,
                    gpu_caches=kv_caches,
                )
            assert self._handlers is not None
            yield (
                GPULoadStoreSpec,
                CPULoadStoreSpec,
                self._handlers.gpu_to_cpu_handler,
            )
            yield (
                CPULoadStoreSpec,
                GPULoadStoreSpec,
                self._handlers.cpu_to_gpu_handler,
            )
            return

        # HMA path: create per-group handlers
        if not self._hybrid_handlers:
            self._hybrid_handlers = HybridCpuGpuOffloadingHandlers(
                kv_cache_config=kv_cache_config,
                gpu_block_size=self.gpu_block_size,
                cpu_block_size=self.offloaded_block_size,
                num_cpu_blocks_per_group=self.num_cpu_blocks,
                gpu_caches=kv_caches,
                attn_backends=attn_backends,
            )

        # Yield 4-tuples with group_id for HMA mode
        for group_id in self._hybrid_handlers.get_group_ids():
            yield (
                GPULoadStoreSpec,
                CPULoadStoreSpec,
                self._hybrid_handlers.get_gpu_to_cpu_handler(group_id),
                group_id,
            )
            yield (
                CPULoadStoreSpec,
                GPULoadStoreSpec,
                self._hybrid_handlers.get_cpu_to_gpu_handler(group_id),
                group_id,
            )
