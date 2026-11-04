# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from unittest.mock import MagicMock

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.canonical_mapping import (
    CANONICAL_FORMAT_VERSION,
    canonical_format_id,
    derive_canonical_mappings,
)
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, MLAAttentionSpec
from vllm.v1.kv_offload.base import CopyRun

NUM_PAGES = 3
ROW_BYTES = 256
ROWS_PER_BLOCK = 2
BLOCK_BYTES = ROWS_PER_BLOCK * ROW_BYTES


def _group(name: str, row_bytes: int = ROW_BYTES) -> KVCacheGroupSpec:
    spec = MLAAttentionSpec(block_size=4, num_kv_heads=1, head_size=64, dtype="uint8")
    return KVCacheGroupSpec(
        [name],
        spec,
        num_blocks=NUM_PAGES,
        blocks_per_page=1,
        kernel_rows_per_block=ROWS_PER_BLOCK,
        row_bytes=row_bytes,
    )


def _vllm_config(tp=1, dcp=1, pcp=1, pp=1, rank=0):
    config = MagicMock()
    config.parallel_config.tensor_parallel_size = tp
    config.parallel_config.decode_context_parallel_size = dcp
    config.parallel_config.prefill_context_parallel_size = pcp
    config.parallel_config.world_size = pp * tp * pcp
    config.parallel_config.rank = rank
    return config


def _kv_cache_config(groups, kv_tp_replicas: int = 1) -> KVCacheConfig:
    return KVCacheConfig(
        num_pages=NUM_PAGES,
        kv_cache_tensors=[],
        kv_cache_groups=list(groups),
        kv_tp_replicas=kv_tp_replicas,
        large_page_bytes=max(g.row_bytes for g in groups),
        block_size=4,
    )


def test_canonical_format_version_covers_whole_block_pages():
    # v2: canonical pages are whole group manager blocks (large-page
    # allocator); v1 pages were per-layer.
    assert CANONICAL_FORMAT_VERSION == 2
    assert canonical_format_id("LBHNC") == "v2-hnd"
    assert canonical_format_id("LBNHC") == "v2-nhd"


def test_replicated_group_gets_identity_mapping_with_writer_rotation():
    groups = [_group("a"), _group("b", row_bytes=512)]
    config = _kv_cache_config(groups, kv_tp_replicas=4)

    rank0 = derive_canonical_mappings(_vllm_config(tp=4, rank=0), config, (0, 1))
    rank3 = derive_canonical_mappings(_vllm_config(tp=4, rank=3), config, (0, 1))

    for mappings in (rank0, rank3):
        assert set(mappings) == {0, 1}
        assert mappings[0].canonical_page_size_bytes == BLOCK_BYTES
        assert mappings[0].local_page_size_bytes == BLOCK_BYTES
        assert mappings[0].parallelism_agnostic
        # Group bytes differ: one 512B-block group, one 1024B-block group.
        assert mappings[1].canonical_page_size_bytes == 2 * 512
    # Identical bytes on every rank: replicas take turns writing blocks.
    assert rank0[0].runs == rank3[0].runs
    assert (rank0[0].is_writer(0), rank3[0].is_writer(0)) == (True, False)
    assert (rank0[0].is_writer(3), rank3[0].is_writer(3)) == (False, True)


def test_sharded_group_gets_opaque_mapping():
    groups = [_group("a")]
    config = _kv_cache_config(groups, kv_tp_replicas=1)

    mappings = derive_canonical_mappings(_vllm_config(tp=4, rank=2), config, (0,))

    mapping = mappings[0]
    assert not mapping.parallelism_agnostic
    # Rank-exclusive offsets: rank r's page lands whole at r * page_size.
    assert mapping.runs == (
        CopyRun(
            local_offset=0,
            canonical_offset=2 * BLOCK_BYTES,
            fragment_size=BLOCK_BYTES,
            num_fragments=1,
            local_stride=BLOCK_BYTES,
            canonical_stride=BLOCK_BYTES,
        ),
    )
    # Every rank writes its own copy.
    assert mapping.num_writers == 1
    assert mapping.local_page_size_bytes == BLOCK_BYTES
    assert mapping.canonical_page_size_bytes == 4 * BLOCK_BYTES


def test_identity_mapping_round_trips_across_tp_sizes():
    """Bytes stored under one TP size are the bytes another TP size loads."""
    groups = [_group("a")]
    config = _kv_cache_config(groups)
    reference = bytes((5 + 11 * i) % 256 for i in range(BLOCK_BYTES))
    for tp in (4, 2, 1):
        config.kv_tp_replicas = tp
        mappings = derive_canonical_mappings(_vllm_config(tp=tp), config, (0,))
        mapping = mappings[0]
        page = bytearray(mapping.local_page_size_bytes)
        for run in mapping.runs:
            for i in range(run.num_fragments):
                start = run.local_offset + i * run.local_stride
                page[start : start + run.fragment_size] = reference[
                    run.canonical_offset
                    + i * run.canonical_stride : run.canonical_offset
                    + i * run.canonical_stride
                    + run.fragment_size
                ]
        assert bytes(page) == reference


def test_derive_refuses_non_tp_worker_grids():
    groups = [_group("a")]
    config = _kv_cache_config(groups, kv_tp_replicas=2)
    for kwargs in ({"tp": 2, "pp": 2}, {"tp": 2, "dcp": 2}, {"tp": 2, "pcp": 2}):
        assert derive_canonical_mappings(_vllm_config(**kwargs), config, (0,)) == {}
