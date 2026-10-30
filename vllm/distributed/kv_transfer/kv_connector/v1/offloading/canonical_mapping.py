# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Derivation of canonical page mappings for KV offloading.

The only place in the offloading stack that reasons about parallelism
(TP/DCP/PCP); everything downstream consumes byte mappings. The offload
unit is a group's whole manager block (all layers, row-interleaved), so
mappings are derived per group: the canonical page of a group is the
full offloaded manager block without parallelism. Groups whose bytes are
TP-replicated (and not context-sharded) map identically with writer
rotation; all others get an opaque fallback mapping (fail closed).
"""

from typing import TYPE_CHECKING

from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheLayout
from vllm.v1.kv_offload.base import CanonicalPageMapping, CopyRun

if TYPE_CHECKING:
    from vllm.config import VllmConfig

# Version of the canonical byte format; bump on any layout change.
# v2: canonical pages are whole group manager blocks (large-page allocator);
# v1 pages were per-layer.
CANONICAL_FORMAT_VERSION = 2


def canonical_format_id(kv_cache_layout: str) -> str:
    """Identity of the canonical byte format, for namespacing persisted KV.
    Canonical pages keep the worker's KV layout family, so the id couples the
    format version with that family; consumers must match it exactly. The family
    keeps its historical NHD/HND spelling so ids stay stable for KV persisted
    before the layout enum existed."""
    layout = KVCacheLayout[kv_cache_layout]
    legacy = {KVCacheLayout.LBNHC: "nhd", KVCacheLayout.LBHNC: "hnd"}
    family = legacy.get(layout, layout.name.lower())
    return f"v{CANONICAL_FORMAT_VERSION}-{family}"


def _identity_mapping(
    block_bytes: int, num_writers: int, writer_index: int
) -> CanonicalPageMapping:
    """Byte-identical block on every writer rank; writers rotate per block."""
    return CanonicalPageMapping(
        canonical_page_size_bytes=block_bytes,
        local_page_size_bytes=block_bytes,
        runs=(CopyRun(0, 0, block_bytes, 1, block_bytes, block_bytes),),
        num_writers=num_writers,
        writer_index=writer_index,
        parallelism_agnostic=True,
    )


def _opaque_fallback_mapping(
    page_size_bytes: int, num_ranks: int, rank: int
) -> CanonicalPageMapping:
    """Fallback: place the worker's page whole at a worker-exclusive offset."""
    run = CopyRun(
        0, rank * page_size_bytes, page_size_bytes, 1, page_size_bytes, page_size_bytes
    )
    return CanonicalPageMapping(
        canonical_page_size_bytes=num_ranks * page_size_bytes,
        local_page_size_bytes=page_size_bytes,
        runs=(run,),
        num_writers=1,
        writer_index=0,
        parallelism_agnostic=False,
    )


def derive_canonical_mappings(
    vllm_config: "VllmConfig",
    kv_cache_config: KVCacheConfig,
    group_ids: tuple[int, ...],
) -> dict[int, CanonicalPageMapping]:
    """Per-group canonical page mappings for this worker, keyed by group id.

    Empty when the offload workers are not exactly the TP group; groups
    absent from the result have no canonical representation.
    """
    parallel_config = vllm_config.parallel_config
    tp_size = parallel_config.tensor_parallel_size
    if (
        parallel_config.world_size != tp_size
        or parallel_config.decode_context_parallel_size != 1
        or parallel_config.prefill_context_parallel_size != 1
    ):
        return {}

    rank = parallel_config.rank
    replicated = kv_cache_config.kv_tp_replicas == tp_size
    mappings: dict[int, CanonicalPageMapping] = {}
    for group_id in group_ids:
        block_bytes = kv_cache_config.group_geometry(group_id).block_bytes
        if replicated:
            mappings[group_id] = _identity_mapping(block_bytes, tp_size, rank)
        else:
            mappings[group_id] = _opaque_fallback_mapping(block_bytes, tp_size, rank)
    return mappings
