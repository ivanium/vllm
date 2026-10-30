# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.coordinator import (  # noqa: E501
    ExternalCachedBlockPool,
    MooncakeStoreCoordinator,
)
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheGroupSpec,
    MambaSpec,
    SlidingWindowSpec,
)


def _mamba_align(block_size=32):
    return MambaSpec(
        block_size=block_size,
        shapes=((1, 1),),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
    )


def _group(layer_names, spec, *, is_eagle_group=False):
    """KVCacheGroupSpec with the large-page geometry fields the planner fills
    in. The coordinator is token-space only and never reads them."""
    return KVCacheGroupSpec(
        layer_names,
        spec,
        is_eagle_group=is_eagle_group,
        num_blocks=64,
        blocks_per_page=1,
        kernel_rows_per_block=1,
        row_bytes=1024,
    )


def _make_coord(groups, use_eagle=False, retention_interval=None):
    """Construct a coordinator whose groups all use the global block size B,
    which is also the hash and scheduler block size, as the planner fixes."""
    block_size = groups[0].kv_cache_spec.block_size
    return MooncakeStoreCoordinator(
        groups,
        scheduler_block_size=block_size,
        block_size=block_size,
        use_eagle=use_eagle,
        retention_interval=retention_interval,
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
    groups = [_group(["L0"], _full(16))]
    coord = _make_coord(groups)
    hs = _hashes(4)
    cmap = ExternalCachedBlockPool({(0, bytes(h)) for h in hs})
    masks, hit = coord.find_longest_cache_hit(hs, max_length=64, cached_block_pool=cmap)
    assert hit == 64
    assert masks[0] == [True, True, True, True]


def test_coordinator_single_full_attention_partial_prefix():
    groups = [_group(["L0"], _full(16))]
    coord = _make_coord(groups)
    hs = _hashes(4)
    cmap = ExternalCachedBlockPool({(0, bytes(hs[0])), (0, bytes(hs[1]))})
    masks, hit = coord.find_longest_cache_hit(hs, max_length=64, cached_block_pool=cmap)
    assert hit == 32
    assert masks[0] == [True, True]


def test_coordinator_single_full_attention_no_hits():
    groups = [_group(["L0"], _full(16))]
    coord = _make_coord(groups)
    hs = _hashes(4)
    cmap = ExternalCachedBlockPool(set())
    masks, hit = coord.find_longest_cache_hit(hs, max_length=64, cached_block_pool=cmap)
    assert hit == 0
    assert masks[0] == []


def test_coordinator_single_swa_tautological_pool_masks_pre_window():
    """SWA tautological-pool: hit_length spans full prefix, mask is
    tail-window only."""
    groups = [_group(["L0"], _swa(block_size=16, sliding_window=32))]
    coord = _make_coord(groups)
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
        _group(["L0"], _full(16)),
        _group(["L1"], _swa(16, 32)),
    ]
    coord = _make_coord(groups)
    hs = _hashes(4)
    cmap = ExternalCachedBlockPool({(g, bytes(h)) for g in (0, 1) for h in hs})
    _masks, hit = coord.find_longest_cache_hit(
        hs, max_length=64, cached_block_pool=cmap
    )
    assert hit == 64


def test_coordinator_hybrid_hole_in_full_clips_both():
    groups = [
        _group(["L0"], _full(16)),
        _group(["L1"], _swa(16, 32)),
    ]
    coord = _make_coord(groups)
    hs = _hashes(4)
    exists = {(0, bytes(hs[0])), (0, bytes(hs[2])), (0, bytes(hs[3]))}
    exists |= {(1, bytes(h)) for h in hs}
    cmap = ExternalCachedBlockPool(exists)
    _masks, hit = coord.find_longest_cache_hit(
        hs, max_length=64, cached_block_pool=cmap
    )
    assert hit == 16


# ----- store_mask -----


def test_store_mask_full_attention_all_true():
    groups = [_group(["L0"], _full(16))]
    coord = _make_coord(groups)
    masks = coord.store_mask(64)
    assert masks == (None,)


def test_store_mask_zero_aligned_returns_empty_per_group():
    groups = [
        _group(["L0"], _full(16)),
        _group(["L1"], _swa(16, 32)),
    ]
    coord = _make_coord(groups)
    masks = coord.store_mask(0)
    assert masks == (None, None)


def test_store_mask_fast_path_all_block_sizes_equal_lcm():
    """When every non-full-attn group already aligns to lcm_block_size, the
    fast path returns all-True without invoking find_longest_cache_hit."""
    full = _full(block_size=64)
    swa = _swa(block_size=64, sliding_window=128)
    groups = [_group(["L0"], full), _group(["L1"], swa)]
    coord = _make_coord(groups)
    assert coord.lcm_block_size == 64
    masks = coord.store_mask(256)
    # Every block in every group is True — no sub-lcm filtering possible.
    assert masks == (None, None)


def test_store_mask_fast_path_single_attention_group():
    """Two groups sharing the same SWA spec collapse to one attention group;
    no lcm filter applies, every chunk is True."""
    swa = _swa(block_size=16, sliding_window=32)
    groups = [_group(["L0"], swa), _group(["L1"], swa)]
    coord = _make_coord(groups)
    assert len(coord.attention_groups) == 1
    masks = coord.store_mask(64)
    assert masks == (None, None)


# ----- store_mask with retention_interval (DSV4 sparse SWA checkpointing) -----


def _retention_groups():
    """Hybrid full-attn + SWA(sw=8) at block size 8; the SWA group needs one
    tail block per hit boundary."""
    full = _full(8)
    swa = _swa(block_size=8, sliding_window=8)
    return [_group(["L0"], full), _group(["L1"], swa)]


def test_store_mask_retention_interval_sparsifies_swa_tails():
    """retention_interval=64 keeps an SWA tail once per 64-token segment
    (chunks 7 and 15) instead of every 32 tokens, dropping the mid-segment
    boundaries at 32 and 96."""
    coord = _make_coord(_retention_groups(), retention_interval=64)
    masks = coord.store_mask(128)
    assert masks[0] is None  # full attn unaffected
    assert masks[1] == [i in (7, 15) for i in range(16)]


def test_store_mask_retention_interval_zero_keeps_only_replay_boundary():
    """retention_interval=0 drops all segment tails; only the latest replay
    boundary (capped at num_prompt-1, aligned down to the block) is retained."""
    coord = _make_coord(_retention_groups(), retention_interval=0)
    # No replay info -> nothing reachable for the SWA group.
    assert coord.store_mask(128)[1] == [False] * 16
    # num_prompt=100 -> latest hit boundary = (100-1)//32*32 = 96 -> chunk 11.
    masks = coord.store_mask(128, num_prompt_tokens=100)
    assert masks[1] == [i == 11 for i in range(16)]


def test_store_mask_retention_interval_zero_eagle_keeps_lookup_tail():
    """With EAGLE the replay boundary sits one scheduler block lower, as in
    core's ``replay_boundaries``: the lookup matches that block and drops
    back from it. The retained SWA tail must span the ``need`` contiguous
    blocks the lookup requires (cdiv(8 - 1, 8) + 1 = 2 here)."""
    coord = _make_coord(_retention_groups(), use_eagle=True, retention_interval=0)
    # num_prompt=100 -> boundary 96 - 8 = 88 -> chunks 10, 11 (11 is the
    # EAGLE peek).
    masks = coord.store_mask(96, num_prompt_tokens=100)
    assert masks[0] is None
    assert masks[1] == [i in (10, 11) for i in range(12)]
    # Aligned num_prompt=96 -> boundaries 80 (resend) and 88 (longer sibling).
    masks = coord.store_mask(96, num_prompt_tokens=96)
    assert masks[1] == [i in (9, 10, 11) for i in range(12)]


@pytest.mark.parametrize("num_prompt_tokens", [100, 96])
def test_store_mask_retention_interval_zero_eagle_round_trip(num_prompt_tokens):
    """Everything store_mask keeps is in the store; the EAGLE lookup of the
    same prompt must then reach the block below the one it drops (88 of
    96)."""
    groups = _retention_groups()
    coord = _make_coord(groups, use_eagle=True, retention_interval=0)
    hashes = _hashes(96 // coord.block_size)
    exists: set[tuple[int, bytes]] = set()
    store_masks = coord.store_mask(96, num_prompt_tokens=num_prompt_tokens)
    for gid, (group, mask) in enumerate(zip(groups, store_masks, strict=True)):
        for chunk_id, block_hash in enumerate(hashes):
            if mask is None or mask[chunk_id]:
                exists.add((gid, bytes(block_hash)))

    _masks, hit = coord.find_longest_cache_hit(
        hashes,
        max_length=96,
        cached_block_pool=ExternalCachedBlockPool(exists),
    )
    assert hit == 88


@pytest.mark.parametrize("extra_retained_tokens", [0, 3])
def test_load_mask_restores_mtp_replay_tail(extra_retained_tokens):
    """An external SWA hit allocates the MTP re-prefill tail below the window,
    so the store must keep it and the load must restore it; otherwise those
    blocks hold stale KV under valid prefix hashes."""
    swa = SlidingWindowSpec(
        block_size=8,
        num_kv_heads=8,
        head_size=64,
        dtype=None,
        sliding_window=8,
        extra_retained_tokens=extra_retained_tokens,
    )
    groups = [_group(["L0"], _full(8)), _group(["L1"], swa)]
    coord = _make_coord(groups, retention_interval=0)
    hashes = _hashes(96 // coord.block_size)
    exists: set[tuple[int, bytes]] = set()
    store_masks = coord.store_mask(96, num_prompt_tokens=96)
    for gid, (group, mask) in enumerate(zip(groups, store_masks, strict=True)):
        for chunk_id, block_hash in enumerate(hashes):
            if mask is None or mask[chunk_id]:
                exists.add((gid, bytes(block_hash)))

    _masks, hit = coord.find_longest_cache_hit(
        hashes,
        max_length=96,
        cached_block_pool=ExternalCachedBlockPool(exists),
    )
    assert hit == 88
    loaded = [i for i, v in enumerate(coord.load_mask(hashes, hit)[1]) if v]
    # add_local_computed_blocks pads only the blocks below the retained window.
    first_allocated = (hit - swa.sliding_window - extra_retained_tokens + 1) // 8
    assert loaded == list(range(first_allocated, hit // 8))


def test_store_mask_retention_interval_keeps_segment_and_replay_tails():
    """Sparse segment tails (interval=64 -> chunks 7,15) plus the replay
    boundary tail (num_prompt=100 -> chunk 11) coexist."""
    coord = _make_coord(_retention_groups(), retention_interval=64)
    masks = coord.store_mask(128, num_prompt_tokens=100)
    assert masks[1] == [i in (7, 11, 15) for i in range(16)]


def test_store_mask_suffix_matches_full_mask_tail():
    coord = _make_coord(_retention_groups(), retention_interval=64)
    full = coord.store_mask(128, num_prompt_tokens=100)
    suffix = coord.store_mask(128, start_token=64, num_prompt_tokens=100)

    for g_idx, cache_group in enumerate(coord.kv_cache_groups):
        block_size = cache_group.kv_cache_spec.block_size
        start_chunk = 64 // block_size
        end_chunk = 128 // block_size
        full_mask = full[g_idx]
        if full_mask is None:
            assert suffix[g_idx] is None
        else:
            assert suffix[g_idx] == full_mask[start_chunk:end_chunk]


def test_store_mask_retention_prefix_stable_as_aligned_length_grows():
    coord = _make_coord(_retention_groups(), retention_interval=0)

    shorter = coord.store_mask(64, num_prompt_tokens=100)[1]
    longer = coord.store_mask(128, num_prompt_tokens=100)[1]

    assert shorter is not None
    assert longer is not None
    assert longer[: len(shorter)] == shorter


def test_store_mask_excludes_mamba_groups_lookup_unaffected():
    """Align-mode mamba block tables are not append-only (interior states are
    nulled/freed; speculative blocks relocate), so the positional normal save
    must never cover mamba chunks — regardless of retention. Lookups still
    probe mamba boundaries: their keys come from the pinned snapshot/CoW
    hand-off path instead."""
    groups = [
        _group(["L0"], _full(16)),
        _group(["L1"], _mamba_align(16)),
    ]
    coord = _make_coord(groups)
    masks = coord.store_mask(64)
    assert masks[0] is None  # full attn stays dense
    assert masks[1] == [False] * 4

    coord = _make_coord(groups, retention_interval=32)
    assert coord.store_mask(64)[1] == [False] * 4

    assert coord.lookup_mask(64)[1] is None


# ----- Eagle / MTP interaction with load_mask -----


def test_lookup_with_eagle_pops_last_full_attention_block():
    """Sanity: with use_eagle, find_longest_cache_hit drops the last block.
    Pairs with the load_mask test below to lock the round-trip contract."""
    groups = [_group(["L0"], _full(16))]
    coord = _make_coord(groups, use_eagle=True)
    hs = _hashes(4)
    cmap = ExternalCachedBlockPool({(0, bytes(h)) for h in hs})
    _masks, hit = coord.find_longest_cache_hit(
        hs, max_length=64, cached_block_pool=cmap
    )
    # 4 blocks present, eagle pops 1 → 3 blocks = 48 tokens.
    assert hit == 48


def test_load_mask_with_eagle_does_not_double_prune_full_attention():
    """Regression for silent KV corruption with MTP/EAGLE-3.

    The recv side calls ``load_mask(block_hashes, token_len)`` where
    ``token_len`` is already the eagle-pruned hit length from ``lookup``.
    A second eagle pop here used to shorten the mask by one extra block;
    ``process_tokens`` then yielded a chunk past the mask, which the worker
    silently skipped — leaving the trailing block of the loaded prefix
    uninitialized in local KV.
    """
    groups = [_group(["L0"], _full(16))]
    coord = _make_coord(groups, use_eagle=True)
    hs = _hashes(4)
    cmap = ExternalCachedBlockPool({(0, bytes(h)) for h in hs})
    _masks, hit = coord.find_longest_cache_hit(
        hs, max_length=64, cached_block_pool=cmap
    )
    assert hit == 48  # eagle popped 1 block

    masks = coord.load_mask(hs, token_len=hit)
    # Every chunk that process_tokens(token_len=48, ...) would yield must
    # have a corresponding mask slot. process_tokens emits chunk_id 0..2
    # (start=0, 16, 32), so the mask must be length 3, all True.
    assert masks[0] == [True, True, True]


def test_load_mask_with_eagle_hybrid_full_plus_swa():
    """Hybrid (FullAttn + SWA) with eagle: load_mask must cover every chunk
    in [0, token_len) for the FullAttn group; SWA group keeps its
    tail-window mask."""
    groups = [
        _group(["L0"], _full(16)),
        _group(["L1"], _swa(16, 32)),
    ]
    coord = _make_coord(groups, use_eagle=True)
    hs = _hashes(4)
    exists = {(g, bytes(h)) for g in (0, 1) for h in hs}
    cmap = ExternalCachedBlockPool(exists)
    _masks, hit = coord.find_longest_cache_hit(
        hs, max_length=64, cached_block_pool=cmap
    )
    # FullAttn dictates the convergence; eagle pops one block off it.
    assert hit == 48

    masks = coord.load_mask(hs, token_len=hit)
    # FullAttn: all chunks populated locally.
    assert masks[0] == [True, True, True]
    # SWA: tail-window only (ceil((32-1)/16) = 2 trailing blocks).
    assert masks[1][-2:] == [True, True]


def test_load_mask_without_eagle_unchanged():
    """Sanity: when eagle is off, load_mask is identical to the pre-fix path."""
    groups = [_group(["L0"], _full(16))]
    coord = _make_coord(groups, use_eagle=False)
    hs = _hashes(4)
    cmap = ExternalCachedBlockPool({(0, bytes(h)) for h in hs})
    _masks, hit = coord.find_longest_cache_hit(
        hs, max_length=64, cached_block_pool=cmap
    )
    assert hit == 64
    masks = coord.load_mask(hs, token_len=hit)
    assert masks[0] == [True, True, True, True]


def _mamba(block_size=16):
    return MambaSpec(
        block_size=block_size,
        shapes=((1, 1),),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
    )


def test_lookup_with_eagle_hybrid_full_plus_mamba_no_overrun():
    """Full+Mamba with eagle must not overrun the attention-verified hit.

    ``MambaManager`` ignores ``drop_eagle_block`` (a Mamba block at position
    p IS the recurrent state after (p + 1) * block_size tokens; there is
    nothing to recompute), so granting the Mamba group the one-block eagle
    peek margin lets it match one block PAST the eagle-pruned full-attention
    hit, and adopting that length resumes the recurrent state ahead of the
    verified token prefix (#43559). Gating the margin on ``not
    isinstance(spec, MambaSpec)`` pins the hit to the attention-verified 48.
    """
    groups = [
        _group(["L0"], _full(16)),
        _group(["L1"], _mamba(16)),
    ]
    coord = _make_coord(groups, use_eagle=True)
    hs = _hashes(4)
    exists = {(g, bytes(h)) for g in (0, 1) for h in hs}
    cmap = ExternalCachedBlockPool(exists)
    _masks, hit = coord.find_longest_cache_hit(
        hs, max_length=64, cached_block_pool=cmap
    )
    # FullAttn matches 4 blocks, eagle pops 1 -> 48 verified tokens. The
    # Mamba group must serve its state@48 snapshot, not peek to state@64.
    assert hit == 48


def test_eagle_flag_propagates_to_all_merged_swa_groups():
    """Regression for MTP x PD external-store 0% prefix hit.

    DSV4 splits SWA layers into several KV cache groups sharing one spec, and
    only the group containing the MTP layer is annotated ``is_eagle_group``.
    The lookup merges equal-spec groups, applies the eagle drop to the merged
    group, and requires each chunk hash to exist in EVERY member group — so
    the save-side masks must eagle-shift every member, not just the annotated
    one. Without propagation the non-annotated groups never store the eagle
    proof-run chunks and every external lookup returns 0.
    """
    swa = _swa(block_size=16, sliding_window=32)
    groups = [
        _group(["L0"], _full(16)),
        _group(["L1"], swa),
        _group(["L2"], swa, is_eagle_group=True),
    ]
    coord = _make_coord(groups, use_eagle=True, retention_interval=0)
    assert coord.eagle_group_ids == {1, 2}

    # Save side: both SWA groups must produce identical (eagle-shifted) masks.
    masks = coord.store_mask(128, num_prompt_tokens=130)
    assert masks[1] == masks[2]

    # Round trip: everything store_mask kept is in the store; the eagle
    # lookup must then serve a non-zero hit (it was 0 before the fix).
    hs = _hashes(128 // 16)
    exists = set()
    for g_idx, mask in enumerate(masks):
        for i, h in enumerate(hs):
            if mask is None or mask[i]:
                exists.add((g_idx, bytes(h)))
    _masks, hit = coord.find_longest_cache_hit(
        hs, max_length=128, cached_block_pool=ExternalCachedBlockPool(exists)
    )
    # Replay boundary 128 - 16 = 112: the eagle lookup matches 128 and drops
    # back to it.
    assert hit == 112
