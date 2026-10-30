# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared sizing helpers for SimpleCPUOffloadConnector."""

from __future__ import annotations

import torch
import torch.distributed as dist

from vllm.logger import init_logger

logger = init_logger(__name__)


def sync_num_offload_blocks_across_workers(num_offload_blocks: int) -> int:
    """All-reduce MIN so every rank allocates the same offload pool size."""
    if not dist.is_initialized():
        return num_offload_blocks

    from vllm.distributed.parallel_state import get_world_group

    world_group = get_world_group()
    if world_group.world_size <= 1:
        return num_offload_blocks

    blocks_tensor = torch.tensor([num_offload_blocks], dtype=torch.int64, device="cpu")
    dist.all_reduce(blocks_tensor, group=world_group.cpu_group, op=dist.ReduceOp.MIN)
    synced = int(blocks_tensor.item())
    if synced != num_offload_blocks:
        logger.info(
            "SimpleCPUOffload: aligned num_offload_blocks from %d to %d "
            "across %d workers",
            num_offload_blocks,
            synced,
            world_group.world_size,
        )
    return synced
