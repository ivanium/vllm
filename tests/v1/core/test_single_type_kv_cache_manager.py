# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import random

import pytest
import torch

from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    KVCacheBlock,
    get_request_block_hasher,
    init_none_hash,
    make_block_hash_with_group_id,
)
from vllm.v1.core.single_type_kv_cache_manager import (
    ChunkedLocalAttentionManager,
    FullAttentionManager,
    SlidingWindowManager,
)
from vllm.v1.kv_cache_interface import (
    ChunkedLocalAttentionSpec,
    FullAttentionSpec,
    SlidingWindowSpec,
)
from vllm.v1.request import Request

pytestmark = pytest.mark.cpu_test


def get_sliding_window_manager(sliding_window_spec, block_pool, enable_caching=True):
    # Tests don't exercise admission gating; pass a large cap that is a no-op.
    return SlidingWindowManager(
        sliding_window_spec,
        block_pool=block_pool,
        enable_caching=enable_caching,
        kv_cache_group_id=0,
        scheduler_block_size=sliding_window_spec.block_size,
        max_admission_blocks_per_request=10**9,
    )


def get_chunked_local_attention_manager(
    chunked_local_attention_spec, block_pool, enable_caching=True
):
    return ChunkedLocalAttentionManager(
        chunked_local_attention_spec,
        block_pool=block_pool,
        enable_caching=enable_caching,
        kv_cache_group_id=0,
        scheduler_block_size=chunked_local_attention_spec.block_size,
        max_admission_blocks_per_request=10**9,
    )


def test_chunked_local_attention_possible_cached_prefix():
    block_size = 2
    chunked_local_attention_spec = ChunkedLocalAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        attention_chunk_size=4,
    )

    block_pool = BlockPool(
        num_gpu_blocks=100, enable_caching=True, hash_block_size=block_size
    )
    manager = get_chunked_local_attention_manager(
        chunked_local_attention_spec, block_pool
    )

    def run_one_case(block_is_cached, tail_token, expect_length):
        block_hash_list = [
            BlockHash(str(i).encode()) for i in range(len(block_is_cached))
        ]

        block_pool.cached_block_hash_to_block._cache.clear()

        # Mock the block pool with the cached blocks
        for i, (block_hash, is_cached) in enumerate(
            zip(block_hash_list, block_is_cached)
        ):
            if is_cached:
                block_pool.cached_block_hash_to_block.insert(
                    make_block_hash_with_group_id(block_hash, 0),
                    block_pool.blocks[i + 10],
                )

        computed_blocks = manager.find_longest_cache_hit(
            block_hashes=block_hash_list,
            max_length=len(block_hash_list) * block_size + tail_token,
            kv_cache_group_ids=[0],
            block_pool=block_pool,
            kv_cache_spec=chunked_local_attention_spec,
            drop_eagle_block=False,
            alignment_tokens=block_size,
        )[0][0]
        assert len(computed_blocks) == expect_length

        assert all(
            block == block_pool.null_block
            for block in computed_blocks[: (expect_length - 1) // 2]
        )

    run_one_case([True], 0, 1)
    run_one_case([True], 1, 1)
    run_one_case([True, False], 0, 2)
    run_one_case([True, False], 1, 2)
    run_one_case([True, True], 0, 2)
    run_one_case([True, True], 1, 2)
    run_one_case([True, True, False], 0, 2)
    run_one_case([True, True, False], 1, 2)
    run_one_case([True, True, True], 0, 3)
    run_one_case([True, True, True], 1, 3)
    run_one_case([True, True, True, False], 0, 4)
    run_one_case([True, True, True, False], 1, 4)
    run_one_case([random.choice([True, False])] * 8 + [True], 1, 9)
    run_one_case([random.choice([True, False])] * 8 + [False], 1, 8)
    run_one_case([random.choice([True, False])] * 8 + [True, True], 1, 10)
    run_one_case([random.choice([True, False])] * 8 + [True, False], 0, 10)
    run_one_case([random.choice([True, False])] * 8 + [True, False], 1, 10)
    run_one_case([random.choice([True, False])] * 8 + [False, True], 0, 10)
    run_one_case([random.choice([True, False])] * 8 + [False, True], 1, 10)
    run_one_case([random.choice([True, False])] * 8 + [False, False], 0, 10)
    run_one_case([random.choice([True, False])] * 8 + [False, False], 1, 10)


def test_sliding_window_possible_cached_prefix():
    block_size = 2
    sliding_window_spec = SlidingWindowSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        sliding_window=4,
    )

    block_pool = BlockPool(
        num_gpu_blocks=100, enable_caching=True, hash_block_size=block_size
    )
    manager = get_sliding_window_manager(sliding_window_spec, block_pool)

    def run_one_case(block_is_cached, expect_length):
        block_hash_list = [
            BlockHash(str(i).encode()) for i in range(len(block_is_cached))
        ]

        block_pool.cached_block_hash_to_block._cache.clear()

        # Mock the block pool with the cached blocks
        for i, (block_hash, is_cached) in enumerate(
            zip(block_hash_list, block_is_cached)
        ):
            if is_cached:
                block_pool.cached_block_hash_to_block.insert(
                    make_block_hash_with_group_id(block_hash, 0),
                    block_pool.blocks[i + 10],
                )

        computed_blocks = manager.find_longest_cache_hit(
            block_hashes=block_hash_list,
            max_length=len(block_hash_list) * block_size,
            kv_cache_group_ids=[0],
            block_pool=block_pool,
            kv_cache_spec=sliding_window_spec,
            drop_eagle_block=False,
            alignment_tokens=block_size,
        )[0][0]
        assert len(computed_blocks) == expect_length

        assert all(
            block == block_pool.null_block
            for block in computed_blocks[: expect_length - 2]
        )
        for i in range(2):
            if i < expect_length:
                block_index = expect_length - i - 1
                assert computed_blocks[block_index].block_id == block_index + 10

    run_one_case([False] * 10, 0)
    run_one_case([True], 1)
    run_one_case([True, False], 1)
    run_one_case([True, True], 2)
    run_one_case([True, True, False], 2)
    run_one_case([True, True, True], 3)
    run_one_case([True, True, True, False], 3)
    run_one_case(
        [True, True, False, True, False, False, True, True, False, True, True, True], 12
    )
    run_one_case(
        [True, True, False, True, False, False, True, True, False, False, False], 8
    )
    run_one_case(
        [True, True, False, True, False, False, True, True, False, False, False, True],
        8,
    )


def test_chunked_local_attention_remove_skipped_blocks():
    attention_spec = ChunkedLocalAttentionSpec(
        block_size=2,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        attention_chunk_size=4,
    )

    block_pool = BlockPool(num_gpu_blocks=2000, enable_caching=True, hash_block_size=2)

    manager = get_chunked_local_attention_manager(attention_spec, block_pool)

    null_block_id = block_pool.null_block.block_id

    def id_to_block_table(ids) -> list[KVCacheBlock]:
        return [
            KVCacheBlock(id_) if id_ != null_block_id else block_pool.null_block
            for id_ in ids
        ]

    def assert_block_id(block_table: list[KVCacheBlock], ids: list[int]):
        for block, id_ in zip(block_table, ids):
            if id_ == null_block_id:
                assert block == block_pool.null_block
            else:
                assert block.block_id == id_

    original_block_ids = [
        1000,
        1001,
        1002,
        1003,
        1004,
        1005,
        1006,
        1007,
        1008,
        1009,
        1010,
    ]
    block_table = id_to_block_table(original_block_ids)
    manager.req_to_blocks["test"] = block_table

    manager.remove_skipped_blocks("test", 0)
    assert_block_id(block_table, original_block_ids)

    # For 4th token (0-indexed), token 0-3 is out of the local attention window.
    manager.remove_skipped_blocks("test", 4)
    assert_block_id(block_table, [null_block_id] * 2)

    # For 6th token (0-indexed), token 4 - 6 are in local attention window,
    # token 0 - 3 are out, 2 blocks can be removed.
    manager.remove_skipped_blocks("test", 6)
    assert_block_id(block_table, [null_block_id] * 2 + original_block_ids[2:])
    # For 12th token (0-indexed),
    # token 0-11 are out, 6 block can be removed.
    manager.remove_skipped_blocks("test", 12)
    assert_block_id(block_table, [null_block_id] * 6)


def test_sliding_window_remove_skipped_blocks():
    sliding_window_spec = SlidingWindowSpec(
        block_size=2,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        sliding_window=4,
    )

    block_pool = BlockPool(num_gpu_blocks=2000, enable_caching=True, hash_block_size=2)

    manager = get_sliding_window_manager(sliding_window_spec, block_pool)

    null_block_id = block_pool.null_block.block_id

    def id_to_block_table(ids) -> list[KVCacheBlock]:
        return [
            KVCacheBlock(id_) if id_ != null_block_id else block_pool.null_block
            for id_ in ids
        ]

    def assert_block_id(block_table: list[KVCacheBlock], ids: list[int]):
        for block, id_ in zip(block_table, ids):
            if id_ == null_block_id:
                assert block == block_pool.null_block
            else:
                assert block.block_id == id_

    original_block_ids = [
        1000,
        1001,
        1002,
        1003,
        1004,
        1005,
        1006,
        1007,
        1008,
        1009,
        1010,
    ]
    block_table = id_to_block_table(original_block_ids)
    manager.req_to_blocks["test"] = block_table

    manager.remove_skipped_blocks("test", 0)
    assert_block_id(block_table, original_block_ids)

    # 4 tokens are computed. Only token 0 is out of the sliding window. As
    # block 1000 also contains token 1 that is in the sliding window, block 1000
    # cannot be removed.
    manager.remove_skipped_blocks("test", 4)
    assert_block_id(block_table, original_block_ids)

    # 5 tokens are computed. Token 0 & 1 are out of the sliding window.
    # Block 1000 can be removed.
    manager.remove_skipped_blocks("test", 5)
    assert_block_id(block_table, [null_block_id] + original_block_ids[1:])

    # 6 tokens are computed. Token 0-2 are out of the sliding window.
    # Cannot remove new block as the block 1001 is still used by token 3.
    manager.remove_skipped_blocks("test", 6)
    assert_block_id(block_table, [null_block_id] + original_block_ids[1:])

    # 7 tokens are computed. Token 0-3 are out of the sliding window.
    # Block 1001 can be removed and block 1000 is already removed.
    manager.remove_skipped_blocks("test", 7)
    assert_block_id(block_table, [null_block_id] * 2 + original_block_ids[2:])

    # 11 tokens are computed. Token 0-7 are out of the sliding window.
    # Block 1002 & 1003 can be removed now. Block 1003 represents a longer
    # sequence, and is expected to be evicted earlier than 1002, so the order
    # of removed blocks should be [1003, 1002].
    manager.remove_skipped_blocks("test", 11)
    assert_block_id(block_table, [null_block_id] * 4 + original_block_ids[4:])


def test_sliding_window_fine_grained_cache_hit():
    block_size = 4
    hash_block_size = 2
    sliding_window_spec = SlidingWindowSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        sliding_window=8,
    )
    block_pool = BlockPool(
        num_gpu_blocks=100, enable_caching=True, hash_block_size=hash_block_size
    )

    # 7 hash blocks = 14 tokens. Full-block keys are the hashes at each
    # block's last hash boundary (indices 1, 3, 5 -> blocks 0, 1, 2); index 6
    # is a partial prompt tail at 14 tokens inside block 3.
    hashes = [BlockHash(str(i).encode()) for i in range(7)]

    def insert(hash_idx: int, pool_block_id: int) -> None:
        block_pool.cached_block_hash_to_block.insert(
            make_block_hash_with_group_id(hashes[hash_idx], 0),
            block_pool.blocks[pool_block_id],
        )

    for hash_idx, pool_block_id in ((1, 10), (3, 11), (5, 12), (6, 13)):
        insert(hash_idx, pool_block_id)

    def find(max_length: int, drop_eagle_block: bool = False):
        blocks, hit_length = SlidingWindowManager.find_longest_cache_hit(
            block_hashes=hashes,
            max_length=max_length,
            kv_cache_group_ids=[0],
            block_pool=block_pool,
            kv_cache_spec=sliding_window_spec,
            drop_eagle_block=drop_eagle_block,
            alignment_tokens=hash_block_size,
        )
        return [b.block_id for b in blocks[0]], hit_length

    null_id = block_pool.null_block.block_id

    # The partial tail at 14 plus the window blocks before it.
    assert find(15) == ([null_id, 11, 12, 13], 14)
    # Eagle drops one hash unit; the tail block no longer serves the claim
    # but the window at 12 is still covered.
    assert find(15, drop_eagle_block=True) == ([null_id, 11, 12], 12)
    # Capped below the tail: falls back to the block-boundary entry at 12.
    assert find(13) == ([null_id, 11, 12], 12)
    # Evicting block 1 breaks window coverage for 14 and 12; the longest hit
    # becomes the standalone block-0 entry (4 tokens cover its own window).
    block_pool.cached_block_hash_to_block.pop(
        make_block_hash_with_group_id(hashes[3], 0), 11
    )
    assert find(15) == ([10], 4)


def test_sliding_window_mask_replay_anchor_at_hash_granularity():
    """With partial hits, the replay boundary sits on a hash boundary that may
    fall inside a block; the mask must keep that block plus enough full blocks
    to cover the whole window."""
    spec = SlidingWindowSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        sliding_window=4,  # need = cdiv(3, 4) = 1 contiguous block per hit
    )

    def mask(num_prompt_tokens: int, partial_anchor_tokens: int | None):
        return SlidingWindowManager.reachable_block_mask(
            start_block=0,
            end_block=4,
            alignment_tokens=8,  # lcm of a hybrid with a block-8 FA group
            kv_cache_spec=spec,
            use_eagle=False,
            retention_interval=None,
            num_prompt_tokens=num_prompt_tokens,
            partial_anchor_tokens=partial_anchor_tokens,
        )

    # Without partial hits: only the dense segment tails (odd blocks, one
    # `need`-sized tail per 8-token segment).
    assert mask(15, None) == [False, True, False, True]
    # Block-aligned fine anchor (prompt 13 -> latest boundary 12): keep the
    # `need` blocks before block 3 on top of the segment tails.
    assert mask(13, 2) == [False, True, True, True]
    # Mid-block fine anchor (prompt 15 -> latest boundary 14 inside block 3):
    # keep the partial tail block itself plus a full window before it.
    assert mask(15, 2) == [False, True, True, True]


def test_full_attention_fine_grained_eagle_drop():
    block_size = 4
    hash_block_size = 2
    full_spec = FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
    )
    block_pool = BlockPool(
        num_gpu_blocks=100, enable_caching=True, hash_block_size=hash_block_size
    )
    hashes = [BlockHash(str(i).encode()) for i in range(7)]
    for hash_idx, pool_block_id in ((1, 10), (3, 11), (5, 12), (6, 13)):
        block_pool.cached_block_hash_to_block.insert(
            make_block_hash_with_group_id(hashes[hash_idx], 0),
            block_pool.blocks[pool_block_id],
        )

    def find(drop_eagle_block: bool):
        blocks, hit_length = FullAttentionManager.find_longest_cache_hit(
            block_hashes=hashes,
            max_length=15,
            kv_cache_group_ids=[0],
            block_pool=block_pool,
            kv_cache_spec=full_spec,
            drop_eagle_block=drop_eagle_block,
            alignment_tokens=hash_block_size,
        )
        return [b.block_id for b in blocks[0]], hit_length

    assert find(drop_eagle_block=False) == ([10, 11, 12, 13], 14)
    # Eagle drops one hash unit (14 -> 12); the partial tail block is trimmed.
    assert find(drop_eagle_block=True) == ([10, 11, 12], 12)
    # Without the partial tail the longest hit is 12; the eagle drop lands
    # mid-block (10) and keeps the last full block as the CoW source.
    block_pool.cached_block_hash_to_block.pop(
        make_block_hash_with_group_id(hashes[6], 0), 13
    )
    assert find(drop_eagle_block=True) == ([10, 11, 12], 10)


def test_get_num_blocks_to_allocate():
    block_size = 2
    sliding_window_spec = SlidingWindowSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        sliding_window=4,  # Placeholder value, not related to test result
    )

    block_pool = BlockPool(
        num_gpu_blocks=100, enable_caching=True, hash_block_size=block_size
    )
    manager = get_sliding_window_manager(sliding_window_spec, block_pool)
    cached_blocks_1 = [KVCacheBlock(i + 1) for i in range(10)]
    cached_blocks_2 = [block_pool.null_block for _ in range(5)] + [
        KVCacheBlock(i + 1) for i in range(5)
    ]

    assert (
        manager.get_num_blocks_to_allocate(
            "1", 20 * block_size, cached_blocks_1, 0, 0, 20 * block_size
        )
        == 20
    )
    assert (
        manager.get_num_blocks_to_allocate(
            "2", 20 * block_size, cached_blocks_2, 0, 0, 20 * block_size
        )
        == 15
    )


def test_evictable_cached_blocks_not_double_allocated():
    block_size = 2
    sliding_window_length = 2 * block_size
    sliding_window_spec = SlidingWindowSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        sliding_window=sliding_window_length,
    )

    block_pool = BlockPool(
        num_gpu_blocks=100, enable_caching=True, hash_block_size=block_size
    )
    manager = get_sliding_window_manager(sliding_window_spec, block_pool)

    request_id = "req"
    evictable_block = block_pool.blocks[1]  # ref_cnt == 0, eviction candidate

    num_blocks_to_allocate = manager.get_num_blocks_to_allocate(
        request_id=request_id,
        num_tokens=2 * block_size,
        new_computed_blocks=[evictable_block],
        total_computed_tokens=block_size,
        num_local_computed_tokens=block_size,
        num_tokens_main_model=2 * block_size,
    )
    # Free capacity check should count evictable cached blocks, but allocation
    # should only allocate the truly new block.
    assert num_blocks_to_allocate == 2

    manager.add_local_computed_blocks(
        request_id,
        [evictable_block],
        num_local_computed_tokens=block_size,
        num_external_computed_tokens=0,
    )
    new_blocks = manager.allocate_new_blocks(
        request_id, num_tokens=4, num_tokens_main_model=4
    )
    assert len(new_blocks) == 1
    assert len(manager.req_to_blocks[request_id]) == 2


def test_chunked_local_attention_get_num_blocks_to_allocate():
    block_size = 2
    attention_spec = ChunkedLocalAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        attention_chunk_size=4,  # Placeholder value, not related to test result
    )

    block_pool = BlockPool(
        num_gpu_blocks=100, enable_caching=True, hash_block_size=block_size
    )
    manager = get_chunked_local_attention_manager(attention_spec, block_pool)
    cached_blocks_1 = [KVCacheBlock(i + 1) for i in range(10)]
    cached_blocks_2 = [block_pool.null_block for _ in range(5)] + [
        KVCacheBlock(i + 1) for i in range(5)
    ]

    assert (
        manager.get_num_blocks_to_allocate(
            "1", 20 * block_size, cached_blocks_1, 0, 0, 20 * block_size
        )
        == 20
    )
    assert (
        manager.get_num_blocks_to_allocate(
            "2", 20 * block_size, cached_blocks_2, 0, 0, 20 * block_size
        )
        == 15
    )


def test_predictor_matches_allocator_blocks_calculation_with_admission_cap():
    """In forward steps, `get_num_blocks_to_allocate` must return exactly what
    `allocate_new_blocks` will pull; otherwise `block_pool.get_new_blocks`
    raises `ValueError: Cannot get N free blocks from the pool`.
    """
    block_size = 2
    sliding_window = 8  # 4-block live window
    cap = sliding_window // block_size

    spec = SlidingWindowSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        sliding_window=sliding_window,
    )
    block_pool = BlockPool(
        num_gpu_blocks=100, enable_caching=True, hash_block_size=block_size
    )
    manager = SlidingWindowManager(
        spec,
        block_pool=block_pool,
        enable_caching=False,
        kv_cache_group_id=0,
        scheduler_block_size=spec.block_size,
        max_admission_blocks_per_request=cap,
    )

    request_id = "req"
    total_computed = 0
    # Walk through request forward steps. Check num_blocks returned by
    # `get_num_blocks_to_allocate` matches what `allocate_new_blocks` pulls
    for num_tokens in (4, 8, 12, 16):
        predicted = manager.get_num_blocks_to_allocate(
            request_id=request_id,
            num_tokens=num_tokens,
            new_computed_blocks=[],
            total_computed_tokens=total_computed,
            num_local_computed_tokens=0,
            num_tokens_main_model=num_tokens,
        )
        new_blocks = manager.allocate_new_blocks(
            request_id, num_tokens=num_tokens, num_tokens_main_model=num_tokens
        )
        assert predicted == len(new_blocks), (
            f"num_tokens={num_tokens}: predictor returned {predicted} "
            f"but allocator pulled {len(new_blocks)}"
        )
        total_computed = num_tokens


def test_full_attention_partial_tail_same_step_guard():
    """A partial tail entry registered this step covers KV that is only
    written during this step's execution, while a consumer's CoW copy runs at
    step start. Consumers hitting the fresh entry must be deferred one step
    (mirrors the mamba guard in ``MambaManager.get_num_blocks_to_allocate``).
    """
    init_none_hash(sha256)
    hash_block_size = 2
    block_size = 4
    block_pool = BlockPool(
        num_gpu_blocks=20, enable_caching=True, hash_block_size=hash_block_size
    )
    spec = FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
    )
    manager = FullAttentionManager(
        spec,
        block_pool=block_pool,
        enable_caching=True,
        kv_cache_group_id=0,
        scheduler_block_size=block_size,
    )

    sampling_params = SamplingParams(max_tokens=17)
    sampling_params.update_from_generation_config({}, eos_token_id=100)
    producer = Request(
        request_id="producer",
        prompt_token_ids=[0, 0, 1, 1, 2, 2],
        sampling_params=sampling_params,
        pooling_params=None,
        block_hasher=get_request_block_hasher(hash_block_size, sha256),
    )
    blocks = manager.allocate_new_blocks("producer", 6, 6)
    assert len(blocks) == 2
    # Caches block 0 as full and freshly registers the partial tail at 6
    # tokens on block 1.
    manager.cache_blocks(producer, 6)

    hit_blocks = list(manager.req_to_blocks["producer"])
    # Same step: a consumer hitting the fresh partial tail must be deferred.
    num_blocks = manager.get_num_blocks_to_allocate("consumer", 8, hit_blocks, 6, 6, 8)
    assert num_blocks > block_pool.num_gpu_blocks

    # Next step the entry's KV has been written; the hit is consumable.
    manager.new_step_starts()
    num_blocks = manager.get_num_blocks_to_allocate("consumer", 8, hit_blocks, 6, 6, 8)
    assert num_blocks <= block_pool.num_gpu_blocks
