# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for MooncakeLookupCoordinator.

These tests build the synthetic block pool directly from per-(chunk, group)
exists results and verify that ``find_longest_cache_hit`` produces the
expected hit length. No vllm scheduler / worker / Mooncake instances
required.
"""

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_lookup_coordinator import (  # noqa: E501
    MooncakeLookupCoordinator,
)
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    SlidingWindowSpec,
)


def _make_block_hashes(n: int) -> list[BlockHash]:
    """Distinct 8-byte block hashes for testing concat behavior."""
    return [BlockHash(i.to_bytes(8, "big")) for i in range(n)]


def _fa_spec(block_size: int = 16) -> FullAttentionSpec:
    return FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=4,
        head_size=16,
        dtype=torch.float16,
    )


def _swa_spec(block_size: int = 16, sliding_window: int = 64) -> SlidingWindowSpec:
    return SlidingWindowSpec(
        block_size=block_size,
        num_kv_heads=4,
        head_size=16,
        dtype=torch.float16,
        sliding_window=sliding_window,
    )


def _make_config(specs: list) -> KVCacheConfig:
    return KVCacheConfig(
        num_blocks=100,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec([f"layer{i}"], spec) for i, spec in enumerate(specs)
        ],
    )


# ---------------------------------------------------------------------------
# FA-only: aggregator should match a flat "first-miss in chunk space" walk.
# ---------------------------------------------------------------------------


@pytest.mark.cpu_test
def test_fa_only_full_hit():
    """All chunks present → hit length == max."""
    block_size = 16
    config = _make_config([_fa_spec(block_size=block_size)])
    coord = MooncakeLookupCoordinator(config, hash_block_size=block_size)
    hashes = _make_block_hashes(8)  # 8 chunks * 16 tokens = 128 tokens
    exists = {(i, 0): 1 for i in range(8)}
    pool = coord.build_block_pool(exists, hashes)
    coord.block_pool = pool
    _hit_blocks, hit = coord.find_longest_cache_hit(hashes, max_cache_hit_length=128)
    assert hit == 128


@pytest.mark.cpu_test
def test_fa_only_first_chunk_miss():
    """Miss at chunk 0 caps hit at 0."""
    block_size = 16
    config = _make_config([_fa_spec(block_size=block_size)])
    coord = MooncakeLookupCoordinator(config, hash_block_size=block_size)
    hashes = _make_block_hashes(8)
    exists = {(i, 0): (1 if i > 0 else 0) for i in range(8)}
    pool = coord.build_block_pool(exists, hashes)
    coord.block_pool = pool
    _hit_blocks, hit = coord.find_longest_cache_hit(hashes, max_cache_hit_length=128)
    assert hit == 0


@pytest.mark.cpu_test
def test_fa_only_mid_chunk_miss():
    """Miss at chunk 3 → hit length = 3 * block_size."""
    block_size = 16
    config = _make_config([_fa_spec(block_size=block_size)])
    coord = MooncakeLookupCoordinator(config, hash_block_size=block_size)
    hashes = _make_block_hashes(8)
    exists = {(i, 0): (0 if i == 3 else 1) for i in range(8)}
    pool = coord.build_block_pool(exists, hashes)
    coord.block_pool = pool
    _hit_blocks, hit = coord.find_longest_cache_hit(hashes, max_cache_hit_length=128)
    # FA's manager walks 0..max, breaks on first miss → 3 chunks hit.
    assert hit == 3 * block_size


# ---------------------------------------------------------------------------
# FA + SWA: SWA's window shifts as FA shrinks max_length (the bug fix).
# ---------------------------------------------------------------------------


@pytest.mark.cpu_test
def test_fa_plus_swa_all_hit():
    """All FA chunks + last 3 SWA chunks present → full hit."""
    block_size = 16
    sliding_window = 32  # → blocks_per_sw = ceil((32-1)/16) = 2 contiguous blocks
    config = _make_config(
        [
            _fa_spec(block_size=block_size),
            _swa_spec(block_size=block_size, sliding_window=sliding_window),
        ]
    )
    coord = MooncakeLookupCoordinator(config, hash_block_size=block_size)
    hashes = _make_block_hashes(8)
    exists = {}
    for cid in range(8):
        exists[(cid, 0)] = 1
        exists[(cid, 1)] = 1
    pool = coord.build_block_pool(exists, hashes)
    coord.block_pool = pool
    _hit_blocks, hit = coord.find_longest_cache_hit(hashes, max_cache_hit_length=128)
    assert hit == 128


@pytest.mark.cpu_test
def test_fa_plus_swa_fa_miss_caps_hit():
    """FA misses at chunk 5; SWA window for the FA-clipped length must hit too."""
    block_size = 16
    sliding_window = 32
    config = _make_config(
        [
            _fa_spec(block_size=block_size),
            _swa_spec(block_size=block_size, sliding_window=sliding_window),
        ]
    )
    coord = MooncakeLookupCoordinator(config, hash_block_size=block_size)
    hashes = _make_block_hashes(8)
    exists = {}
    for cid in range(8):
        exists[(cid, 0)] = 1 if cid != 5 else 0
        exists[(cid, 1)] = 1  # SWA: every chunk has the byte
    pool = coord.build_block_pool(exists, hashes)
    coord.block_pool = pool
    _hit_blocks, hit = coord.find_longest_cache_hit(hashes, max_cache_hit_length=128)
    # FA caps to 5 chunks (80 tokens). SWA window at 80 has all chunks present.
    assert hit == 5 * block_size


@pytest.mark.cpu_test
def test_fa_plus_swa_swa_window_miss_at_clipped_length():
    """The bug-fix scenario:

    FA hits everything (8 chunks). At max_length=128, SWA's window is
    last 2 contiguous chunks (chunks 6, 7). Suppose SWA chunks 6, 7 hit
    but chunks 4, 5 miss. The single-pass aggregation today would say
    "all queried hits, return 128." With the iterative coordinator,
    the answer should still be 128 because SWA's window at 128 only
    needs chunks 6, 7 — earlier chunks aren't in window.
    """
    block_size = 16
    sliding_window = 32
    config = _make_config(
        [
            _fa_spec(block_size=block_size),
            _swa_spec(block_size=block_size, sliding_window=sliding_window),
        ]
    )
    coord = MooncakeLookupCoordinator(config, hash_block_size=block_size)
    hashes = _make_block_hashes(8)
    exists = {}
    for cid in range(8):
        exists[(cid, 0)] = 1  # FA all hit
        # SWA: chunks 4, 5 absent (they aren't in window at full max_length)
        exists[(cid, 1)] = 1 if cid not in (4, 5) else 0
    pool = coord.build_block_pool(exists, hashes)
    coord.block_pool = pool
    _hit_blocks, hit = coord.find_longest_cache_hit(hashes, max_cache_hit_length=128)
    assert hit == 128


@pytest.mark.cpu_test
def test_fa_plus_swa_iterative_shrink():
    """The exact scenario the iterative loop is for.

    FA misses at chunk 5 (hit = 80 tokens after FA).
    SWA window at 80: last 2 chunks of cdiv(80,16)=5 chunks → chunks 3, 4.
    But SWA chunks 3, 4 are absent (only chunks 6, 7 saved).
    Coordinator must converge to a smaller hit_length where SWA window
    is fully populated, not return 80.
    """
    block_size = 16
    sliding_window = 32
    config = _make_config(
        [
            _fa_spec(block_size=block_size),
            _swa_spec(block_size=block_size, sliding_window=sliding_window),
        ]
    )
    coord = MooncakeLookupCoordinator(config, hash_block_size=block_size)
    hashes = _make_block_hashes(8)
    exists = {}
    for cid in range(8):
        exists[(cid, 0)] = 1 if cid != 5 else 0  # FA miss at chunk 5
        # SWA: only the LAST two chunks (6, 7) are saved, simulating
        # full-prompt save with no chunked-prefill backfill.
        exists[(cid, 1)] = 1 if cid in (6, 7) else 0
    pool = coord.build_block_pool(exists, hashes)
    coord.block_pool = pool
    _hit_blocks, hit = coord.find_longest_cache_hit(hashes, max_cache_hit_length=128)
    # FA caps at 80; SWA window at 80 needs chunks 3, 4 which aren't there.
    # Shrinks further. SWA window at 64 needs chunks 2, 3, ... none there
    # either. Eventually 0.
    assert hit == 0


# ---------------------------------------------------------------------------
# Concat-hash mapping for HMA (group_block_size > hash_block_size).
# ---------------------------------------------------------------------------


@pytest.mark.cpu_test
def test_concat_hash_lookup_for_larger_group_block_size():
    """An HMA group whose block_size is 2 × hash_block_size needs concat hashes.

    Build 16-byte block_hashes at hash_bs=8. A group with block_size=16
    will probe with concat(h0+h1), concat(h2+h3), etc. Our pool must
    answer 'cached' for those concat hashes when both halves were
    saved by Mooncake.
    """
    hash_bs = 8
    fa_block_size = 16  # 2 × hash_bs
    config = _make_config([_fa_spec(block_size=fa_block_size)])
    coord = MooncakeLookupCoordinator(config, hash_block_size=hash_bs)
    # 8 hashes at hash_bs → 4 chunks at FA's block_size=16.
    hashes = _make_block_hashes(8)
    # Mooncake: chunks 0, 1 saved; chunks 2, 3 missing.
    exists = {
        (0, 0): 1,
        (1, 0): 1,
        (2, 0): 0,
        (3, 0): 0,
    }
    pool = coord.build_block_pool(exists, hashes)
    # Token length = 4 chunks × 16 tokens = 64. FA breaks at first miss → 32.
    coord.block_pool = pool
    _hit_blocks, hit = coord.find_longest_cache_hit(hashes, max_cache_hit_length=64)
    assert hit == 2 * fa_block_size


# ---------------------------------------------------------------------------
# Edge cases.
# ---------------------------------------------------------------------------


@pytest.mark.cpu_test
def test_empty_block_hashes_returns_zero():
    config = _make_config([_fa_spec()])
    coord = MooncakeLookupCoordinator(config, hash_block_size=16)
    pool = coord.build_block_pool({}, [])
    coord.block_pool = pool
    _hit_blocks, hit = coord.find_longest_cache_hit([], max_cache_hit_length=0)
    assert hit == 0


@pytest.mark.cpu_test
def test_eagle_group_ids_preserved_through_unwrap():
    """``is_eagle_group`` flags must propagate from the input config
    into ``self.eagle_group_ids`` and ``self.eagle_attn_group_indices``.

    Regression: previously the unwrap dropped the flag and __init__
    set both eagle sets empty, silently disabling the parent's
    extra-block match-and-drop and over-reporting hit length by one
    block under spec-decode.
    """
    block_size = 16
    fa = _fa_spec(block_size=block_size)
    config = KVCacheConfig(
        num_blocks=100,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["layer0"], fa, is_eagle_group=False),
            KVCacheGroupSpec(["layer1"], fa, is_eagle_group=True),
        ],
    )
    coord = MooncakeLookupCoordinator(config, hash_block_size=block_size)
    assert coord.eagle_group_ids == {1}
    # Both groups have identical specs → merge into one attention_group;
    # that attention_group contains group_id 1 → flagged eagle.
    assert coord.eagle_attn_group_indices == {0}


@pytest.mark.cpu_test
def test_eagle_preserved_through_uniform_type_wrap():
    """Unwrap from UniformTypeKVCacheSpecs MUST preserve is_eagle_group
    on the new KVCacheGroupSpec — otherwise spec-decode workloads
    silently lose EAGLE handling on the worker side.
    """
    from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs

    block_size = 16
    fa = _fa_spec(block_size=block_size)
    wrapped = UniformTypeKVCacheSpecs(
        block_size=block_size, kv_cache_specs={"layer1": fa}
    )
    config = KVCacheConfig(
        num_blocks=100,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["layer0"], fa, is_eagle_group=False),
            KVCacheGroupSpec(["layer1"], wrapped, is_eagle_group=True),
        ],
    )
    coord = MooncakeLookupCoordinator(config, hash_block_size=block_size)
    assert coord.eagle_group_ids == {1}


@pytest.mark.cpu_test
def test_use_eagle_flags_all_groups_when_none_annotated():
    """Mirror parent ``KVCacheCoordinator.__init__`` fallback: when
    ``use_eagle=True`` is passed and no group is explicitly flagged
    via ``is_eagle_group``, treat all groups as eagle.

    Without this, an EAGLE-enabled config without per-group annotations
    would over-report hit length by one block (the parent's
    extra-block match-and-drop wouldn't fire).
    """
    block_size = 16
    config = _make_config([_fa_spec(block_size=block_size)])
    # use_eagle=True, no is_eagle_group annotations
    coord = MooncakeLookupCoordinator(
        config, hash_block_size=block_size, use_eagle=True
    )
    # Expect ALL groups treated as eagle.
    assert coord.eagle_group_ids == {0}
    # use_eagle=False (default): no auto-annotation
    coord2 = MooncakeLookupCoordinator(config, hash_block_size=block_size)
    assert coord2.eagle_group_ids == set()


@pytest.mark.cpu_test
def test_hash_block_size_misalignment_raises():
    """``HybridKVCacheCoordinator.__init__`` asserts every group's
    ``block_size`` is divisible by ``hash_block_size``. Skipping the
    parent's ``__init__`` means we have to recreate the invariant —
    otherwise ``scale = block_size // hash_block_size`` silently
    rounds and the concat hash mapping doesn't match vllm's
    ``BlockHashListWithBlockSize``.
    """
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_lookup_coordinator import (  # noqa: E501
        HashBlockSizeMisalignedError,
    )

    config = _make_config([_fa_spec(block_size=15)])  # 15 % 4 != 0
    with pytest.raises(HashBlockSizeMisalignedError):
        MooncakeLookupCoordinator(config, hash_block_size=4)


@pytest.mark.cpu_test
def test_unknown_manager_spec_raises():
    """Unknown spec types fail closed at construction.

    Caller (worker) catches ``UnknownManagerSpecError`` and falls back
    to the legacy aggregation. Silently skipping the unknown group
    would let the coordinator return a hit length that ignored that
    group, producing a false external hit when its data isn't in the
    store.
    """
    from dataclasses import dataclass

    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_lookup_coordinator import (  # noqa: E501
        UnknownManagerSpecError,
    )
    from vllm.v1.kv_cache_interface import KVCacheSpec

    @dataclass(frozen=True, kw_only=True)
    class _UnknownSpec(KVCacheSpec):
        block_size: int
        # Not registered in spec_manager_map → coordinator must refuse.

    block_size = 16
    config = KVCacheConfig(
        num_blocks=100,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["layer0"], _fa_spec(block_size=block_size)),
            KVCacheGroupSpec(["layer1"], _UnknownSpec(block_size=block_size)),
        ],
    )
    with pytest.raises(UnknownManagerSpecError):
        MooncakeLookupCoordinator(config, hash_block_size=block_size)


@pytest.mark.cpu_test
def test_uniform_type_wrapped_specs_unwrap():
    """Worker-side configs wrap groups in UniformTypeKVCacheSpecs.

    Regression: previously the coordinator did
    ``spec_manager_map[type(spec)]`` directly and crashed with
    ``KeyError: <class UniformTypeKVCacheSpecs>`` on DSV4-Flash
    startup. The fix drills into ``kv_cache_specs`` to reach a
    concrete inner spec.
    """
    from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs

    block_size = 16
    inner_fa = _fa_spec(block_size=block_size)
    wrapped = UniformTypeKVCacheSpecs(
        block_size=block_size,
        kv_cache_specs={"layer0": inner_fa},
    )
    config = KVCacheConfig(
        num_blocks=100,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["layer0"], wrapped)],
    )

    # Should NOT raise KeyError. Coordinator unwraps and finds
    # FullAttentionManager for the inner FullAttentionSpec.
    coord = MooncakeLookupCoordinator(config, hash_block_size=block_size)
    assert len(coord.attention_groups) == 1
    spec, group_ids, _manager_cls = coord.attention_groups[0]
    assert group_ids == [0]
    # Probe spec is the unwrapped inner spec, not the wrapper.
    assert spec is inner_fa


@pytest.mark.cpu_test
def test_error_exists_treated_as_miss():
    """exists == -1 (Mooncake error) folds to miss — never a false hit."""
    block_size = 16
    config = _make_config([_fa_spec(block_size=block_size)])
    coord = MooncakeLookupCoordinator(config, hash_block_size=block_size)
    hashes = _make_block_hashes(4)
    exists = {(0, 0): 1, (1, 0): -1, (2, 0): 1, (3, 0): 1}
    pool = coord.build_block_pool(exists, hashes)
    coord.block_pool = pool
    _hit_blocks, hit = coord.find_longest_cache_hit(hashes, max_cache_hit_length=64)
    # Chunk 1 errored → treated as miss → hit = 1 chunk.
    assert hit == block_size
