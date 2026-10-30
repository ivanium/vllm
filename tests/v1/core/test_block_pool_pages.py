# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""BlockPool with groups that hold more than one block per page."""

import math
import random

import pytest
import torch

from vllm.distributed.kv_events import BlockRemoved, BlockStored
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import (
    KVCacheBlock,
    get_request_block_hasher,
    init_none_hash,
    maybe_convert_block_hash,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec
from vllm.v1.request import Request

pytestmark = pytest.mark.cpu_test

BLOCK_SIZE = 4


@pytest.fixture(autouse=True)
def _auto_init_hash_fn():
    init_none_hash(sha256)


def make_pool(num_pages: int, blocks_per_page: tuple[int, ...]) -> BlockPool:
    # Group g's blocks tile a page in blocks_per_page[g] pieces.
    page_bytes = math.lcm(*blocks_per_page)
    spec = FullAttentionSpec(
        block_size=BLOCK_SIZE, num_kv_heads=1, head_size=16, dtype=torch.float16
    )
    groups = [
        KVCacheGroupSpec(
            layer_names=[],
            kv_cache_spec=spec,
            num_blocks=num_pages * n,
            blocks_per_page=n,
            kernel_rows_per_block=1,
            row_bytes=page_bytes // n,
        )
        for n in blocks_per_page
    ]
    return BlockPool(
        num_pages=num_pages,
        enable_caching=True,
        block_size=BLOCK_SIZE,
        enable_kv_cache_events=True,
        large_page_bytes=page_bytes,
        groups=groups,
    )


def make_request(request_id: str, prompt_token_ids: list[int]) -> Request:
    sampling_params = SamplingParams(max_tokens=1)
    sampling_params.update_from_generation_config({}, 100)
    return Request(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        mm_features=None,
        sampling_params=sampling_params,
        pooling_params=None,
        block_hasher=get_request_block_hasher(BLOCK_SIZE, sha256),
    )


def cache(
    pool: BlockPool,
    request: Request,
    blocks: list[KVCacheBlock],
    g: int,
    num_cached_blocks: int = 0,
):
    pool.cache_full_blocks(
        request=request,
        blocks=blocks,
        num_cached_blocks=num_cached_blocks,
        num_full_blocks=len(blocks),
        kv_cache_group_id=g,
    )


def cached_free_blocks(
    pool: BlockPool, request_id: str, num_blocks: int, g: int
) -> list[KVCacheBlock]:
    """Allocate blocks of a group, cache them under a fresh prompt, free them."""
    start = 1000 * len(request_id) + 100 * num_blocks
    request = make_request(request_id, list(range(start, start + num_blocks * 4)))
    blocks = pool.get_new_blocks(num_blocks, g)
    cache(pool, request, blocks, g)
    pool.free_blocks(blocks)
    return blocks


def test_group_fills_its_pages_before_taking_a_page():
    pool = make_pool(num_pages=3, blocks_per_page=(2,))
    assert pool.get_num_free_blocks(0) == 4

    first = pool.get_new_blocks(1, 0)
    second = pool.get_new_blocks(1, 0)

    # Block b is at index b % 2 of page b // 2; page 0 holds the null block.
    assert [b.block_id for b in first + second] == [2, 3]
    assert pool.get_num_free_blocks(0) == 2
    assert pool.can_allocate((2,), {0: []}, 0)
    assert not pool.can_allocate((3,), {0: []}, 0)


def test_block_that_crosses_a_page_end_is_never_allocated():
    """Blocks of 300 B in 1000 B pages: block 3 (bytes 900-1200) crosses the
    end of page 0 and block 6 the end of page 1, so no page holds them."""
    spec = FullAttentionSpec(
        block_size=BLOCK_SIZE, num_kv_heads=1, head_size=16, dtype=torch.float16
    )
    pool = BlockPool(
        num_pages=3,
        enable_caching=True,
        block_size=BLOCK_SIZE,
        large_page_bytes=1000,
        groups=[
            KVCacheGroupSpec(
                layer_names=[],
                kv_cache_spec=spec,
                num_blocks=10,
                blocks_per_page=2,
                kernel_rows_per_block=1,
                row_bytes=300,
            )
        ],
    )
    # Pages 1 and 2 hold at least 2 blocks each; page 0 is the null page.
    assert pool.get_num_free_blocks(0) == 4

    blocks = pool.get_new_blocks(4, 0)

    assert sorted(b.block_id for b in blocks) == [4, 5, 7, 8]
    assert [pool.get_block(0, b.block_id) for b in blocks] == blocks
    pool.free_blocks(blocks)
    assert pool.get_num_free_blocks(0) == 4


def test_group_reuses_the_free_blocks_of_its_fullest_page_first():
    pool = make_pool(num_pages=4, blocks_per_page=(4, 1))
    blocks = pool.get_new_blocks(8, 0)
    # Page 1 keeps one block in use and page 2 keeps three.
    pool.free_blocks(blocks[:3] + blocks[4:5])

    assert pool.get_new_blocks(1, 0) == [blocks[4]]

    # Page 1 empties, so another group can take it.
    pool.free_blocks(blocks[3:4])
    assert pool.get_num_free_blocks(1) == 2


@pytest.mark.parametrize("retained_by", ["prefix_cache", "unpin"])
def test_clean_page_is_taken_before_retained_blocks_are_evicted(retained_by):
    """Prefix-cached blocks and blocks released with unpin_blocks (HiSparse
    resident pages still read) both outlive the clean pages."""
    pool = make_pool(num_pages=4, blocks_per_page=(2,))
    retained, kept = pool.get_new_blocks(2, 0)
    request = make_request("a", list(range(4)))
    reused: list[KVCacheBlock] = []
    if retained_by == "prefix_cache":
        cache(pool, request, [retained], 0)
        pool.free_blocks([retained])
    else:
        pool.unpin_blocks([retained], reused.append)

    # Page 1 is in use, so its retained block survives while clean pages last.
    assert [b.block_id for b in pool.get_new_blocks(2, 0)] == [4, 5]
    assert [b.block_id for b in pool.get_new_blocks(2, 0)] == [6, 7]
    assert not reused
    if retained_by == "prefix_cache":
        assert pool.get_cached_block(request.block_hashes[0], [0]) == [retained]

    (evicted,) = pool.get_new_blocks(1, 0)

    assert evicted is retained and evicted.block_hash is None
    assert pool.get_cached_block(request.block_hashes[0], [0]) is None
    assert reused == ([retained] if retained_by == "unpin" else [])
    assert kept.ref_cnt == 1


def test_cached_blocks_do_not_spare_the_clean_page_for_later_groups():
    pool = make_pool(num_pages=3, blocks_per_page=(2, 1))
    cached, _ = pool.get_new_blocks(2, 0)
    cache(pool, make_request("a", list(range(4))), [cached], 0)
    pool.free_blocks([cached])

    # Group 0 takes the clean page instead of evicting its cached block, so
    # group 1 finds no page left.
    assert not pool.can_allocate((1, 1), {0: []}, 0)
    pool.get_new_blocks(1, 0)
    with pytest.raises(ValueError):
        pool.get_new_blocks(1, 1)


def test_cached_page_goes_to_another_group_last():
    pool = make_pool(num_pages=3, blocks_per_page=(2, 1))
    blocks = cached_free_blocks(pool, "a", 2, g=0)
    pool.take_events()

    # Group 1 takes the clean page first, then the page of cached blocks.
    assert [b.block_id for b in pool.get_new_blocks(1, 1)] == [2]
    assert pool.take_events() == []
    assert [b.block_id for b in pool.get_new_blocks(1, 1)] == [1]

    assert all(b.block_hash is None for b in blocks)
    removed = pool.take_events()
    assert len(removed) == 2 and all(isinstance(e, BlockRemoved) for e in removed)


def test_older_cached_page_is_evicted_before_a_recent_cached_block():
    """With no clean page left, a group evicts whichever was released first:
    its cached blocks on in-use pages or the front page of the free queue."""
    pool = make_pool(num_pages=4, blocks_per_page=(2,))
    old = cached_free_blocks(pool, "old", 2, g=0)  # Page 1, fully free.
    recent, kept = pool.get_new_blocks(2, 0)  # Page 2.
    request = make_request("recent", list(range(500, 504)))
    cache(pool, request, [recent], 0)
    pool.free_blocks([recent])
    pool.get_new_blocks(2, 0)  # Page 3: no clean page is left.

    (block,) = pool.get_new_blocks(1, 0)

    assert block in old and all(b.block_hash is None for b in old)
    assert pool.get_cached_block(request.block_hashes[0], [0]) == [recent]
    assert kept.ref_cnt == 1


def test_admission_holds_when_group_allocations_interleave():
    """A request's later Mamba blocks and other requests' reservations can be
    allocated after another group's blocks, in any order."""
    pool = make_pool(num_pages=6, blocks_per_page=(1, 2))
    # Pages 1 and 2 hold one cached free and one in-use block of group 1.
    blocks = pool.get_new_blocks(4, 1)
    cache(pool, make_request("a", list(range(16))), blocks, 1)
    pool.free_blocks(blocks[::2])

    # Group-order allocation would fit, but this interleaving does not: group
    # 1 takes a clean page before evicting its cached blocks.
    assert not pool.can_allocate((3, 2), {1: []}, 0)
    pool.get_new_blocks(1, 0)
    pool.get_new_blocks(1, 1)
    with pytest.raises(ValueError):
        pool.get_new_blocks(2, 0)


def test_admission_counts_the_free_blocks_of_pinned_hit_pages():
    """A hit pins its page, and the page's other blocks become free blocks of
    the group, so a request that fits there is admitted even with no free
    page left."""
    pool = make_pool(num_pages=3, blocks_per_page=(4, 4))
    hit_0 = cached_free_blocks(pool, "a", 4, g=0)[0]
    hit_1 = cached_free_blocks(pool, "bb", 4, g=1)[0]

    assert pool.can_allocate((2, 2), {0: [hit_0], 1: [hit_1]}, 0)
    pool.touch([hit_0, hit_1])
    pool.get_new_blocks(1, 0)
    pool.get_new_blocks(1, 1)


def test_hit_on_a_free_page_keeps_its_other_cached_blocks():
    pool = make_pool(num_pages=3, blocks_per_page=(2,))
    hit, other = cached_free_blocks(pool, "a", 2, g=0)
    other_hash = other.block_hash

    pool.touch([hit])

    assert pool.get_num_free_blocks(0) == 3  # A clean page plus `other`.
    assert [b.block_id for b in pool.get_new_blocks(2, 0)] == [4, 5]
    assert other.block_hash == other_hash
    assert pool.get_new_blocks(1, 0) == [other]
    assert other.block_hash is None


def test_block_ids_and_eviction_are_per_group():
    pool = make_pool(num_pages=3, blocks_per_page=(2, 1))
    (block_0,) = cached_free_blocks(pool, "a", 1, g=0)
    pool.touch([block_0])
    (block_1,) = cached_free_blocks(pool, "bb", 1, g=1)

    # Both groups hand out block id 2: index 0 of page 1 and page 2.
    assert block_0.block_id == block_1.block_id == 2
    assert pool.get_block(0, 2) is block_0
    assert pool.get_block(1, 2) is block_1
    assert pool.get_block(0, 0) is pool.get_block(1, 0) is pool.null_block

    pool.evict_blocks([(1, 2)])

    assert block_1.block_hash is None
    assert block_0.block_hash is not None


def test_reset_prefix_cache_clears_every_page():
    pool = make_pool(num_pages=4, blocks_per_page=(2, 1))
    blocks_0 = cached_free_blocks(pool, "a", 2, g=0)
    blocks_1 = cached_free_blocks(pool, "bb", 1, g=1)
    held = pool.get_new_blocks(1, 1)

    assert not pool.reset_prefix_cache()
    pool.free_blocks(held)
    assert pool.reset_prefix_cache()

    assert all(b.block_hash is None for b in blocks_0 + blocks_1)
    assert pool.get_num_free_blocks(0) == 6
    assert pool.get_usage() == 0.0


@pytest.mark.parametrize("blocks_per_page", [(1, 1), (1, 2, 4), (3, 2)])
@pytest.mark.parametrize("seed", [0, 1])
def test_admitted_sessions_always_allocate(blocks_per_page, seed):
    """Sessions with shared prefixes come and go; a session runs only if
    `can_allocate` admits it, and then its touches and allocations in group
    order must succeed. With one block per page, the test is the legacy
    `sum(demand) <= free blocks` check."""
    rng = random.Random(seed)
    num_pages = 24
    num_groups = len(blocks_per_page)
    pool = make_pool(num_pages, blocks_per_page)
    prefixes = [[rng.randrange(10**6) for _ in range(n * BLOCK_SIZE)] for n in (2, 4)]
    sessions: list[list[list[KVCacheBlock]]] = []
    num_admitted = 0

    for step in range(400):
        if sessions and rng.random() < 0.4:
            for g, blocks in enumerate(sessions.pop(rng.randrange(len(sessions)))):
                pool.free_blocks(reversed(blocks))
        tokens = rng.choice(prefixes) + [
            10**6 + step * 100 + i for i in range(rng.randrange(4) * BLOCK_SIZE)
        ]
        request = make_request(f"r{step}", tokens)
        num_blocks = len(tokens) // BLOCK_SIZE
        hits: list[list[KVCacheBlock]] = []
        for g in range(num_groups):
            group_hits: list[KVCacheBlock] = []
            for block_hash in request.block_hashes:
                hit = pool.get_cached_block(block_hash, [g])
                if hit is None:
                    break
                group_hits.append(hit[0])
            hits.append(group_hits)
        demand = tuple(
            num_blocks - len(group_hits) + sum(b.ref_cnt == 0 for b in group_hits)
            for group_hits in hits
        )
        hit_blocks = {g: hits[g] for g in range(num_groups) if blocks_per_page[g] > 1}

        admitted = pool.can_allocate(demand, hit_blocks, 0)

        if all(n == 1 for n in blocks_per_page):
            assert admitted == (sum(demand) <= pool.get_num_free_blocks())
        if not admitted:
            continue
        num_admitted += 1
        for group_hits in hits:
            pool.touch(group_hits)
        session = []
        for g, group_hits in enumerate(hits):
            new_blocks = pool.get_new_blocks(num_blocks - len(group_hits), g)
            assert all(b.group_id == g and b.ref_cnt == 1 for b in new_blocks)
            cache(pool, request, group_hits + new_blocks, g, len(group_hits))
            session.append(group_hits + new_blocks)
        sessions.append(session)

    assert num_admitted > 100
    for session in sessions:
        for blocks in session:
            pool.free_blocks(blocks)
    assert pool.get_usage() == 0.0
    assert pool.get_num_free_blocks(0) == (num_pages - 1) * blocks_per_page[0]


# ---------------------------------------------------------------------------
# cache_partial_block: registration at an interior (mamba-align checkpoint)
# boundary, its events, and its interaction with eviction, reset and
# promotion.
# ---------------------------------------------------------------------------


def test_partial_block_registration_emits_block_stored_with_parent():
    pool = make_pool(num_pages=3, blocks_per_page=(1,))
    (block,) = pool.get_new_blocks(1, 0)
    request = make_request("a", list(range(8)))
    pool.take_events()

    key = pool.cache_partial_block(request, block, 2 * BLOCK_SIZE, 0)

    assert key is not None
    assert pool.get_cached_block(request.block_hashes[1], [0]) == [block]
    (event,) = pool.take_events()
    assert isinstance(event, BlockStored)
    assert event.block_hashes == [maybe_convert_block_hash(request.block_hashes[1])]
    assert event.parent_block_hash == maybe_convert_block_hash(request.block_hashes[0])
    assert event.token_ids == [4, 5, 6, 7]
    assert event.block_size == BLOCK_SIZE
    assert event.group_idx == 0


def test_partial_block_replacement_emits_remove_then_store():
    """Re-keying a cached block retires its old entries before storing the
    new one."""
    pool = make_pool(num_pages=3, blocks_per_page=(1,))
    (block,) = pool.get_new_blocks(1, 0)
    request = make_request("a", list(range(8)))
    cache(pool, request, [block], 0)
    pool.take_events()

    key = pool.cache_partial_block(
        request, block, 2 * BLOCK_SIZE, 0, replace_existing_hashes=True
    )

    assert key is not None
    assert pool.get_cached_block(request.block_hashes[0], [0]) is None
    assert pool.get_cached_block(request.block_hashes[1], [0]) == [block]
    removed, stored = pool.take_events()
    assert isinstance(removed, BlockRemoved)
    assert removed.block_hashes == [maybe_convert_block_hash(request.block_hashes[0])]
    assert removed.group_idx == 0
    assert isinstance(stored, BlockStored)
    assert stored.block_hashes == [maybe_convert_block_hash(request.block_hashes[1])]
    assert stored.group_idx == 0


def test_eviction_removes_partial_block_entries():
    pool = make_pool(num_pages=2, blocks_per_page=(1,))
    (block,) = pool.get_new_blocks(1, 0)
    request = make_request("a", list(range(4)))
    pool.cache_partial_block(request, block, BLOCK_SIZE, 0)
    pool.free_blocks([block])
    pool.take_events()

    (recycled,) = pool.get_new_blocks(1, 0)

    assert recycled is block and block.block_hash is None
    assert pool.get_cached_block(request.block_hashes[0], [0]) is None
    (removed,) = pool.take_events()
    assert isinstance(removed, BlockRemoved)
    assert removed.block_hashes == [maybe_convert_block_hash(request.block_hashes[0])]


def test_reset_prefix_cache_clears_partial_block_entries():
    pool = make_pool(num_pages=3, blocks_per_page=(1, 2))
    (block,) = pool.get_new_blocks(1, 1)
    request = make_request("a", list(range(8)))
    pool.cache_partial_block(request, block, BLOCK_SIZE, 1)
    pool.free_blocks([block])
    pool.take_events()

    assert pool.reset_prefix_cache()

    assert pool.get_cached_block(request.block_hashes[0], [1]) is None
    assert block.block_hash is None


def test_full_block_caching_replaces_the_partial_entry():
    """Promoting a partially registered block to its full-block hash retires
    the partial entry (remove-then-store)."""
    pool = make_pool(num_pages=3, blocks_per_page=(1,))
    b0, b1 = pool.get_new_blocks(2, 0)
    partial_req = make_request("p", [10, 11, 12, 13, 14, 15, 16, 17])
    pool.cache_partial_block(partial_req, b1, BLOCK_SIZE, 0)
    pool.take_events()

    request = make_request("a", list(range(8)))
    cache(pool, request, [b0, b1], 0)

    assert pool.get_cached_block(partial_req.block_hashes[0], [0]) is None
    assert pool.get_cached_block(request.block_hashes[0], [0]) == [b0]
    assert pool.get_cached_block(request.block_hashes[1], [0]) == [b1]
    removed, stored = pool.take_events()
    assert isinstance(removed, BlockRemoved)
    assert removed.block_hashes == [
        maybe_convert_block_hash(partial_req.block_hashes[0])
    ]
    assert isinstance(stored, BlockStored)
    assert stored.block_hashes == [
        maybe_convert_block_hash(request.block_hashes[0]),
        maybe_convert_block_hash(request.block_hashes[1]),
    ]


def test_secondary_hash_survives_eviction_of_same_id_block_of_another_group():
    """Partial entries are keyed by (group, block id): evicting one group's
    block must not remove another group's entry that shares the bare id."""
    pool = make_pool(num_pages=5, blocks_per_page=(1, 2))
    (g1_block,) = pool.get_new_blocks(1, 1)  # index 0 of page 1: id 2
    (g0_block,) = pool.get_new_blocks(1, 0)  # page 2: id 2
    assert g1_block.block_id == g0_block.block_id == 2

    # The group-1 block carries a primary hash plus a secondary partial entry
    # at a shorter prefix of another request (the mamba-align re-key shape).
    req_a = make_request("a", list(range(8)))
    cache(pool, req_a, [g1_block], 1)
    req_b = make_request("b", [10, 11, 12, 13, 14, 15, 16, 17])
    pool.cache_partial_block(req_b, g1_block, BLOCK_SIZE, 1)
    assert pool.get_cached_block(req_b.block_hashes[0], [1]) == [g1_block]

    # Cache and free the group-0 block, then force its eviction: the pool
    # takes the clean pages 3 and 4 first, then evicts page 2.
    cache(pool, make_request("c", [20, 21, 22, 23]), [g0_block], 0)
    pool.free_blocks([g0_block])
    pool.take_events()
    pool.get_new_blocks(2, 0)
    (recycled,) = pool.get_new_blocks(1, 0)
    assert recycled is g0_block and g0_block.block_hash is None

    # The group-1 entries survive the group-0 eviction untouched.
    assert pool.get_cached_block(req_b.block_hashes[0], [1]) == [g1_block]
    assert pool.get_cached_block(req_a.block_hashes[0], [1]) == [g1_block]
    req_b_event_hash = maybe_convert_block_hash(req_b.block_hashes[0])
    for event in pool.take_events():
        assert isinstance(event, BlockRemoved)
        assert req_b_event_hash not in event.block_hashes


def test_taking_a_fresh_page_keeps_the_same_id_block_entries():
    """A page no group has owned yet shares its id with a block of a
    group with more blocks per page; taking it must not touch that block."""
    pool = make_pool(num_pages=4, blocks_per_page=(2,))
    b2, _ = pool.get_new_blocks(2, 0)  # page 1 holds ids 2 and 3
    cache(pool, make_request("a", list(range(4))), [b2], 0)
    # A second entry at a prefix of another request (the mamba-align re-key).
    request = make_request("b", [10, 11, 12, 13])
    pool.cache_partial_block(request, b2, BLOCK_SIZE, 0)
    pool.take_events()

    pool.get_new_blocks(2, 0)  # takes page 2, whose first block is id 2

    assert pool.get_cached_block(request.block_hashes[0], [0]) == [b2]
    assert not pool.take_events()


def test_evicting_a_read_block_keeps_it_until_reuse():
    """A block released with unpin_blocks stays readable after its prefix
    cache entry is evicted, and its page still drains cleanly."""
    pool = make_pool(num_pages=3, blocks_per_page=(2,))
    read, kept = pool.get_new_blocks(2, 0)
    request = make_request("a", list(range(4)))
    cache(pool, request, [read], 0)
    reused: list[KVCacheBlock] = []
    pool.unpin_blocks([read], reused.append)

    pool.evict_blocks([(0, read.block_id)])
    pool.free_blocks([kept])

    assert pool.get_cached_block(request.block_hashes[0], [0]) is None
    assert pool.get_num_free_blocks(0) == 4
    assert [b.block_id for b in pool.get_new_blocks(2, 0)] == [4, 5]
    assert not reused


def test_read_block_of_a_one_block_group_is_reused_last():
    """A block still read after unpin_blocks outlives clean pages even when
    its last reference is released through free_blocks."""
    pool = make_pool(num_pages=4, blocks_per_page=(1,))
    (read,) = pool.get_new_blocks(1, 0)
    pool.touch([read])
    reused: list[KVCacheBlock] = []
    pool.unpin_blocks([read], reused.append)
    pool.free_blocks([read])

    clean = pool.get_new_blocks(2, 0)

    assert read not in clean and not reused
    assert pool.get_new_blocks(1, 0) == [read]
    assert reused == [read]
