# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_store_coordinator import (  # noqa: E501
    ExternalCachedBlockPool,
    MooncakeStoreCoordinator,
)
from vllm.v1.core.kv_cache_utils import BlockHash, BlockHashListWithBlockSize
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheGroupSpec,
    SlidingWindowSpec,
)

# ----- ExternalCachedBlockPool -----


def test_external_cached_block_pool_tautological_returns_present_for_any_hash():
    cmap = ExternalCachedBlockPool()
    h = BlockHash(b"\xaa" * 4)
    res = cmap.get_cached_block(h, [0, 1])
    assert res is not None
    assert len(res) == 2
    assert res[0] is not cmap.null_block
    assert res[1] is not cmap.null_block


def test_external_cached_block_pool_hit_all_groups():
    h = BlockHash(b"\x11\x22\x33\x44")
    cmap = ExternalCachedBlockPool({(0, bytes(h)), (1, bytes(h))})
    res = cmap.get_cached_block(h, [0, 1])
    assert res is not None
    assert len(res) == 2
    assert res[0] is not cmap.null_block
    assert res[1] is not cmap.null_block


def test_external_cached_block_pool_miss_one_group():
    h = BlockHash(b"\x11\x22\x33\x44")
    cmap = ExternalCachedBlockPool({(0, bytes(h))})
    assert cmap.get_cached_block(h, [0, 1]) is None


def test_external_cached_block_pool_unknown_hash():
    h_known = BlockHash(b"\x01" * 4)
    h_unknown = BlockHash(b"\x02" * 4)
    cmap = ExternalCachedBlockPool({(0, bytes(h_known))})
    assert cmap.get_cached_block(h_unknown, [0]) is None


# ----- Helpers -----


def _full(block_size=16, sliding_window=None):
    return FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=8,
        head_size=64,
        dtype=None,
        sliding_window=sliding_window,
    )


def _swa(block_size=16, sliding_window=32):
    return SlidingWindowSpec(
        block_size=block_size,
        num_kv_heads=8,
        head_size=64,
        dtype=None,
        sliding_window=sliding_window,
    )


def _hashes(n: int) -> list[BlockHash]:
    return [BlockHash(bytes([i + 1]) * 4) for i in range(n)]


# ----- Single-group coordinator -----


def test_coordinator_single_full_attention_all_hits():
    groups = [KVCacheGroupSpec(["L0"], _full(16))]
    coord = MooncakeStoreCoordinator(groups, hash_block_size=16)
    hs = _hashes(4)
    cmap = ExternalCachedBlockPool({(0, bytes(h)) for h in hs})
    masks, hit = coord.find_longest_cache_hit(hs, max_length=64, cached_block_pool=cmap)
    assert hit == 64
    assert masks[0] == [True, True, True, True]


def test_coordinator_single_full_attention_partial_prefix():
    groups = [KVCacheGroupSpec(["L0"], _full(16))]
    coord = MooncakeStoreCoordinator(groups, hash_block_size=16)
    hs = _hashes(4)
    cmap = ExternalCachedBlockPool({(0, bytes(hs[0])), (0, bytes(hs[1]))})
    masks, hit = coord.find_longest_cache_hit(hs, max_length=64, cached_block_pool=cmap)
    assert hit == 32
    assert masks[0] == [True, True]


def test_coordinator_single_full_attention_no_hits():
    groups = [KVCacheGroupSpec(["L0"], _full(16))]
    coord = MooncakeStoreCoordinator(groups, hash_block_size=16)
    hs = _hashes(4)
    cmap = ExternalCachedBlockPool(set())
    masks, hit = coord.find_longest_cache_hit(hs, max_length=64, cached_block_pool=cmap)
    assert hit == 0
    assert masks[0] == []


def test_coordinator_single_swa_tautological_pool_masks_pre_window():
    """SWA tautological-pool: hit_length spans full prefix, mask is
    tail-window only."""
    groups = [KVCacheGroupSpec(["L0"], _swa(block_size=16, sliding_window=32))]
    coord = MooncakeStoreCoordinator(groups, hash_block_size=16)
    hs = _hashes(4)  # 4 chunks * 16 tokens
    cmap = ExternalCachedBlockPool()
    masks, hit = coord.find_longest_cache_hit(hs, max_length=64, cached_block_pool=cmap)
    assert hit == 64
    # ceil((sw-1)/block_size) = ceil(31/16) = 2 tail blocks.
    assert masks[0][-2:] == [True, True]
    assert all(not m for m in masks[0][:-2])


# ----- Hybrid coordinator (single-group worker, multi-group coordinator) -----


def test_coordinator_hybrid_full_plus_swa_all_hit():
    groups = [
        KVCacheGroupSpec(["L0"], _full(16)),
        KVCacheGroupSpec(["L1"], _swa(16, 32)),
    ]
    coord = MooncakeStoreCoordinator(groups, hash_block_size=16)
    hs = _hashes(4)
    cmap = ExternalCachedBlockPool({(g, bytes(h)) for g in (0, 1) for h in hs})
    _masks, hit = coord.find_longest_cache_hit(
        hs, max_length=64, cached_block_pool=cmap
    )
    assert hit == 64


def test_coordinator_hybrid_hole_in_full_clips_both():
    groups = [
        KVCacheGroupSpec(["L0"], _full(16)),
        KVCacheGroupSpec(["L1"], _swa(16, 32)),
    ]
    coord = MooncakeStoreCoordinator(groups, hash_block_size=16)
    hs = _hashes(4)
    exists = {(0, bytes(hs[0])), (0, bytes(hs[2])), (0, bytes(hs[3]))}
    exists |= {(1, bytes(h)) for h in hs}
    cmap = ExternalCachedBlockPool(exists)
    _masks, hit = coord.find_longest_cache_hit(
        hs, max_length=64, cached_block_pool=cmap
    )
    assert hit == 16


def test_coordinator_group_block_size_double_hash():
    """Group block_size=32 over hash_block_size=16 hashes: adjacent
    hashes merge before pool lookup."""
    groups = [
        KVCacheGroupSpec(["L0"], _full(16)),
        KVCacheGroupSpec(["L1"], _full(32)),
    ]
    coord = MooncakeStoreCoordinator(groups, hash_block_size=16)
    hs = _hashes(4)
    big_hashes = list(BlockHashListWithBlockSize(hs, 16, 32))
    exists = {(0, bytes(h)) for h in hs}
    exists |= {(1, bytes(bh)) for bh in big_hashes}
    cmap = ExternalCachedBlockPool(exists)
    _masks, hit = coord.find_longest_cache_hit(
        hs, max_length=64, cached_block_pool=cmap
    )
    assert hit == 64
    assert hit % 32 == 0
