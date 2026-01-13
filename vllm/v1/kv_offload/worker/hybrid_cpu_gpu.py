# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Per-group CPU-GPU handlers for HMA (Hybrid Memory Allocator) support.

This module provides HybridCpuGpuOffloadingHandlers which creates separate
CpuGpuOffloadingHandlers for each KV cache group. This enables per-group
CPU offloading for hybrid models with multiple KV cache types.
"""

from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.kv_offload.worker.cpu_gpu import CpuGpuOffloadingHandlers

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)


class HybridCpuGpuOffloadingHandlers:
    """
    Per-group CpuGpuOffloadingHandlers for HMA (Hybrid Memory Allocator) support.

    Creates separate handler pairs (GPU->CPU and CPU->GPU) for each KV cache
    group. Each group's handlers only know about that group's layer tensors,
    enabling independent offloading of different cache types.

    This design allocates CPU memory separately per group. Each group gets:
    - Its own CPU tensor storage (one tensor per layer in the group)
    - Its own handlers for GPU<->CPU transfers
    - Its own block ID namespace

    Note: This design is simpler but uses more memory than a shared design.
    A future optimization could share CPU blocks across groups for blocks with
    the same content hash.

    Args:
        kv_cache_config: KV cache configuration with group specifications.
        gpu_block_size: GPU block size in tokens.
        cpu_block_size: CPU block size in tokens.
        num_cpu_blocks_per_group: Number of CPU blocks allocated per group.
        gpu_caches: Dictionary of layer names to GPU KV cache tensors.
        attn_backends: Dictionary of layer names to attention backends.
    """

    def __init__(
        self,
        kv_cache_config: "KVCacheConfig",
        gpu_block_size: int,
        cpu_block_size: int,
        num_cpu_blocks_per_group: int,
        gpu_caches: dict[str, torch.Tensor],
        attn_backends: dict[str, type[AttentionBackend]],
    ):
        self.kv_cache_config = kv_cache_config
        self.gpu_block_size = gpu_block_size
        self.cpu_block_size = cpu_block_size
        self.num_cpu_blocks_per_group = num_cpu_blocks_per_group

        # Create per-group handlers
        self.group_handlers: dict[int, CpuGpuOffloadingHandlers] = {}

        for group_id, group_spec in enumerate(kv_cache_config.kv_cache_groups):
            # Filter to only this group's layers
            group_layer_names = group_spec.layer_names

            # Get GPU caches for this group's layers
            group_gpu_caches: dict[str, torch.Tensor] = {}
            for layer_name in group_layer_names:
                if layer_name in gpu_caches:
                    group_gpu_caches[layer_name] = gpu_caches[layer_name]

            # Get attention backends for this group's layers
            group_attn_backends: dict[str, type[AttentionBackend]] = {}
            for layer_name in group_layer_names:
                if layer_name in attn_backends:
                    group_attn_backends[layer_name] = attn_backends[layer_name]

            # Skip empty groups (e.g., no layers on this PP rank)
            if not group_gpu_caches:
                logger.debug(
                    "Skipping group %d: no GPU caches for layer names %s",
                    group_id,
                    group_layer_names,
                )
                continue

            logger.info(
                "Creating handlers for group %d with %d layers: %s",
                group_id,
                len(group_gpu_caches),
                list(group_gpu_caches.keys()),
            )

            # Create handlers for this group
            # Each group gets its own CPU tensors and handlers
            self.group_handlers[group_id] = CpuGpuOffloadingHandlers(
                gpu_block_size=gpu_block_size,
                cpu_block_size=cpu_block_size,
                num_cpu_blocks=num_cpu_blocks_per_group,
                gpu_caches=group_gpu_caches,
                attn_backends=group_attn_backends,
            )

    def get_gpu_to_cpu_handler(self, group_id: int):
        """
        Get the GPU->CPU handler for a specific group.

        Args:
            group_id: KV cache group ID.

        Returns:
            The SingleDirectionOffloadingHandler for GPU->CPU transfers.

        Raises:
            KeyError: If no handler exists for the group.
        """
        return self.group_handlers[group_id].gpu_to_cpu_handler

    def get_cpu_to_gpu_handler(self, group_id: int):
        """
        Get the CPU->GPU handler for a specific group.

        Args:
            group_id: KV cache group ID.

        Returns:
            The SingleDirectionOffloadingHandler for CPU->GPU transfers.

        Raises:
            KeyError: If no handler exists for the group.
        """
        return self.group_handlers[group_id].cpu_to_gpu_handler

    def get_group_ids(self) -> list[int]:
        """
        Get list of group IDs that have handlers.

        Returns:
            List of group IDs with registered handlers.
        """
        return list(self.group_handlers.keys())

    def __len__(self) -> int:
        """
        Get number of groups with handlers.

        Returns:
            Number of groups.
        """
        return len(self.group_handlers)
