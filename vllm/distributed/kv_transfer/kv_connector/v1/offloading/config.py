# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Translate vLLM KV cache metadata for native offloading backends."""

from typing import TYPE_CHECKING

from vllm.utils.math_utils import round_up
from vllm.v1.core.kv_cache_utils import resolve_dcp_kv_block_size
from vllm.v1.kv_cache_interface import KVCacheGroupRole
from vllm.v1.kv_offload.config import (
    OffloadingCacheConfig,
    OffloadingConfig,
    OffloadingGroupConfig,
    OffloadingModelConfig,
    OffloadingParallelConfig,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig


def get_offloading_group_ids(kv_cache_config: "KVCacheConfig") -> tuple[int, ...]:
    if kv_cache_config.hisparse_host_num_blocks is None:
        return kv_cache_config.prefix_cacheable_group_ids
    return tuple(
        group_id
        for group_id, group in enumerate(kv_cache_config.kv_cache_groups)
        if group.role is KVCacheGroupRole.HISPARSE_INDEXER
    )


def build_offloading_config(
    vllm_config: "VllmConfig",
    kv_cache_config: "KVCacheConfig",
) -> OffloadingConfig:
    """Translate vLLM configuration into the native offloading boundary."""
    kv_transfer_config = vllm_config.kv_transfer_config
    assert kv_transfer_config is not None
    extra_config = kv_transfer_config.kv_connector_extra_config
    assert kv_transfer_config.engine_id is not None
    engine_id = kv_transfer_config.engine_id

    parallel_config = vllm_config.parallel_config
    selected_groups = tuple(
        (group_id, kv_cache_config.kv_cache_groups[group_id])
        for group_id in get_offloading_group_ids(kv_cache_config)
    )
    if not selected_groups:
        raise ValueError("KV offloading found no eligible cache groups.")
    groups = tuple(
        OffloadingGroupConfig(
            group_id=group_id,
            tokens_per_block=resolve_dcp_kv_block_size(
                group.kv_cache_spec,
                parallel_config.decode_context_parallel_size,
            ),
            layer_names=tuple(group.layer_names),
        )
        for group_id, group in selected_groups
    )

    tokens_per_hash = kv_cache_config.block_size
    for group in groups:
        assert group.tokens_per_block % tokens_per_hash == 0, (
            f"tokens_per_block={group.tokens_per_block} not divisible by "
            f"tokens_per_hash={tokens_per_hash}. "
            f"Hybrid models (e.g. Mamba+Attention) need "
            f"--enable-prefix-caching to align block sizes."
        )

    blocks_per_chunk = 1
    blocks_per_chunk_config = extra_config.get("blocks_per_chunk")
    tokens_per_chunk = extra_config.get("block_size")

    if blocks_per_chunk_config is not None and tokens_per_chunk is not None:
        raise ValueError(
            "Specify only one of 'block_size' or 'blocks_per_chunk' "
            "in kv_connector_extra_config."
        )

    if blocks_per_chunk_config is not None:
        blocks_per_chunk = int(blocks_per_chunk_config)

        if blocks_per_chunk <= 0:
            raise ValueError("'blocks_per_chunk' must be greater than 0.")

    elif tokens_per_chunk is not None:
        tokens_per_chunk_int = int(tokens_per_chunk)

        unique_tokens_per_block = {group.tokens_per_block for group in groups}

        assert len(unique_tokens_per_block) == 1, (
            "If 'block_size' is specified in kv_connector_extra_config, "
            "there must be at least one KV cache group, "
            "and all groups must have the same block size."
        )

        tokens_per_block = unique_tokens_per_block.pop()
        if tokens_per_chunk_int % tokens_per_block == 0:
            blocks_per_chunk = tokens_per_chunk_int // tokens_per_block
        else:
            raise ValueError(
                f"'block_size'={tokens_per_chunk_int} in kv_connector_extra_config "
                f"must be a multiple of the GPU KV cache block size "
                f"({tokens_per_block} tokens). Use "
                f"{round_up(tokens_per_chunk_int, tokens_per_block)} instead, or set "
                f"'blocks_per_chunk' to express the chunk size in blocks."
            )

    worker_kv_bytes_per_block = 0
    if kv_cache_config.num_pages > 0:
        # This rank's bytes of one manager block across the offloaded groups:
        # R_g kernel rows of row_g bytes each, summed over selected groups.
        worker_kv_bytes_per_block = sum(
            kv_cache_config.group_geometry(group_id).block_bytes
            for group_id, _ in selected_groups
        )

    replicated_layout = (
        vllm_config.model_config.use_mla
        and parallel_config.tensor_parallel_size > 1
        and kv_cache_config.kv_tp_replicas == parallel_config.tensor_parallel_size
        and worker_kv_bytes_per_block > 0
        # Safe MVP boundary: TP-only, no other parallel axes.
        and parallel_config.pipeline_parallel_size == 1
        and parallel_config.prefill_context_parallel_size == 1
        and parallel_config.decode_context_parallel_size == 1
        and parallel_config.world_size == parallel_config.tensor_parallel_size
        # Shared /dev/shm mmap layout is single-node mp only.
        and parallel_config.distributed_executor_backend == "mp"
        and parallel_config.nnodes_within_dp == 1
    )

    canonical_layout = bool(extra_config.get("canonical_layout", False))

    # Whole manager blocks are portable across parallel topologies only when
    # every rank holds byte-identical KV: TP-replicated groups, no context
    # parallelism, no PP. Row-interleaved head shards cannot be recombined
    # across topologies by concatenation, so the head-sharded case of the
    # per-layer page encoding has no whole-block analog.
    tp_size = parallel_config.tensor_parallel_size
    is_parallelism_agnostic = (
        kv_cache_config.kv_tp_replicas == tp_size
        and parallel_config.decode_context_parallel_size == 1
        and parallel_config.prefill_context_parallel_size == 1
        and parallel_config.world_size == tp_size
    )
    if canonical_layout:
        # Canonical pages are whole group blocks; TP-replicated groups map
        # identically with writer rotation, so the shared /dev/shm layout
        # dedups them.
        replicated_layout = (
            is_parallelism_agnostic
            and parallel_config.nnodes_within_dp == 1
            and (
                parallel_config.world_size == 1
                or parallel_config.distributed_executor_backend == "mp"
            )
        )

    kv_events_config = vllm_config.kv_events_config
    cache_dtype = (
        vllm_config.model_config.dtype
        if vllm_config.cache_config.cache_dtype == "auto"
        else vllm_config.cache_config.cache_dtype
    )

    return OffloadingConfig(
        groups=groups,
        worker_kv_bytes_per_block=worker_kv_bytes_per_block,
        enable_kv_cache_events=(
            kv_events_config is not None and kv_events_config.enable_kv_cache_events
        ),
        extra_config=extra_config,
        engine_id=engine_id,
        model=OffloadingModelConfig(
            name=vllm_config.model_config.model,
            dtype=str(cache_dtype).removeprefix("torch."),
        ),
        cache=OffloadingCacheConfig(
            tokens_per_hash=tokens_per_hash,
            blocks_per_chunk=blocks_per_chunk,
        ),
        parallel=OffloadingParallelConfig(
            rank=parallel_config.rank,
            world_size=parallel_config.world_size,
            tp_size=parallel_config.tensor_parallel_size,
            pp_size=parallel_config.pipeline_parallel_size,
            pcp_size=parallel_config.prefill_context_parallel_size,
            dcp_size=parallel_config.decode_context_parallel_size,
            data_parallel_index=parallel_config.data_parallel_index,
            data_parallel_size=parallel_config.data_parallel_size,
            data_parallel_rank_local=parallel_config.data_parallel_rank_local,
            is_parallelism_agnostic=is_parallelism_agnostic,
        ),
        replicated_layout=replicated_layout,
        canonical_layout=canonical_layout,
        kv_cache_layout=vllm_config.cache_config.kv_cache_layout,
    )
