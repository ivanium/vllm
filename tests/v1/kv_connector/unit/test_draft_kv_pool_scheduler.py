# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A prefill instance (NIXL for the decode side's reads, a MooncakeStore prefix
cache with async loads) whose DFlash/DSpark drafter group has its own block
pool (`VLLM_DRAFT_KV_POOL`): async loads are reserved in each pool, store saves
pin each block in its own pool, and no pool size that fits a request stalls.

Reuses the scheduler loop of test_async_kv_load_window_group_reservation (the
admission fix's test) with the drafter group in pool 1.
"""

import random
from dataclasses import replace
from unittest.mock import patch

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake_store import (
    scheduler as store_scheduler,
)
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import DFlashSlidingWindowManager
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.request import RequestStatus

from . import test_async_kv_load_window_group_reservation as base
from .utils import create_scheduler, create_vllm_config, make_kv_cache_group

pytestmark = pytest.mark.cpu_test


def _pool_scheduler_factory(pool_blocks: int):
    def _scheduler(num_blocks: int, hybrid: bool):
        assert hybrid
        vllm_config = create_vllm_config(
            kv_connector="MultiConnector",
            kv_role="kv_both",
            kv_connector_extra_config=base.PREFILL_CONNECTORS,
            kv_load_failure_policy="recompute",
            max_num_seqs=16,
            max_num_batched_tokens=128,
            block_size=base.BLOCK_SIZE,
        )
        vllm_config.scheduler_config.long_prefill_token_threshold = 64
        vllm_config.scheduler_config.max_num_inflight_kv_loads = 8
        vllm_config.scheduler_config.max_async_kv_loads_per_step = 4
        groups = [
            make_kv_cache_group(["target"], base._spec()),
            replace(
                make_kv_cache_group(
                    ["drafter"],
                    base._spec(),
                    DFlashSlidingWindowManager,
                    DFlashSlidingWindowManager.config(sliding_window=base.WINDOW),
                ),
                pool_id=1,
            ),
        ]
        kv_cache_config = KVCacheConfig(
            num_blocks=num_blocks,
            kv_cache_tensors=[],
            kv_cache_groups=groups,
            pool_num_blocks={1: pool_blocks},
        )
        store = base._StoreStub()
        with patch.object(store_scheduler, "LookupKeyClient", return_value=store):
            scheduler = create_scheduler(
                vllm_config, num_blocks=num_blocks, kv_cache_config=kv_cache_config
            )
        assert isinstance(scheduler, Scheduler)
        return scheduler, store

    return _scheduler


def _run(num_blocks, pool_blocks, prompts, steps):
    with patch.object(base, "_scheduler", _pool_scheduler_factory(pool_blocks)):
        return base._run(num_blocks, True, prompts, steps)


def _drafter_cap(scheduler) -> int:
    drafter = scheduler.kv_cache_manager.coordinator.single_type_managers[1]
    return drafter.max_admission_blocks_per_request


def test_parked_load_reserves_in_each_pool():
    scheduler, requests, _ = _run(200, 40, base.SHORT_TAILS, steps=1)
    head = requests[0]
    assert head.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    shared, private = scheduler.kv_cache_manager.coordinator.block_pools
    target, drafter = scheduler.kv_cache_manager.coordinator.single_type_managers
    assert target.block_pool is shared and drafter.block_pool is private
    # The 48-token tail: 3 blocks in each group, each in its own pool.
    assert scheduler._parked_reservations[head] == 3
    assert scheduler._parked_private_reservations[head] == [3]
    assert scheduler._spec_decode_step_blocks_by_pool == [0, 0]
    # The head's drafter blocks were allocated from the drafter's pool.
    held = [b for b in drafter.req_to_blocks[head.request_id] if not b.is_null]
    assert held and all(b.pool() is private for b in held)


def _assert_drained(scheduler, requests):
    """Every request finished, every reservation and pin released, and every
    block of both pools free again (cached blocks are free)."""
    assert all(r.request_id not in scheduler.requests for r in requests)
    assert scheduler._parked_reserved_blocks == 0
    assert scheduler._parked_private_reserved_blocks == [0]
    assert not scheduler._parked_private_reservations
    store = scheduler.connector.sub_connectors[1].connector_scheduler
    assert not store._pinned_saves and not store._pinned_private_blocks
    for pool in scheduler.kv_cache_manager.coordinator.block_pools:
        assert pool.get_num_free_blocks() == pool.num_gpu_blocks - 1


@pytest.mark.parametrize(
    "prompts",
    [
        base.SHORT_TAILS,
        [(640 + 48, 640), (640 + 16, 640), (320 + 100, 320), (960 + 30, 960)],
        [(1600 + 16, 1600), (800 + 200, 800), (400 + 8, 400)],
        [(1900 + 120, 1900), (64 + 300, 64), (1200 + 64, 1200), (500 + 40, 480)],
    ],
)
def test_store_hits_never_stall_with_a_drafter_pool(prompts):
    """Every shared pool that fits the longest prompt, with every drafter pool
    from the smallest that admits a request at its cap, serves every prompt."""
    smallest = max(-(-n // base.BLOCK_SIZE) for n, _ in prompts) + 16
    probe, _, _ = _run(smallest, 1000, prompts, steps=0)
    cap = _drafter_cap(probe)
    stalled = []
    for num_blocks in range(smallest, smallest + 40, 3):
        for pool_blocks in (1 + cap, 2 + cap, 2 * cap, 4 * cap, 1000):
            scheduler, requests, stall = _run(num_blocks, pool_blocks, prompts, 160)
            if stall:
                stalled.append((num_blocks, pool_blocks))
            else:
                _assert_drained(scheduler, requests)
    assert stalled == []


def test_seeded_store_hit_mixes_never_stall_with_a_drafter_pool():
    rng = random.Random(82)
    stalled = []
    for _ in range(20):
        prompts = []
        for _ in range(rng.randint(2, 5)):
            num_hit_tokens = rng.randrange(128, 1800, base.BLOCK_SIZE)
            prompts.append((num_hit_tokens + rng.randint(1, 64), num_hit_tokens))
        if rng.random() < 0.5:
            prompts.append((rng.randint(64, 600), 0))
        smallest = max(-(-n // base.BLOCK_SIZE) for n, _ in prompts) + 16
        probe, _, _ = _run(smallest, 1000, prompts, steps=0)
        cap = _drafter_cap(probe)
        for num_blocks in range(smallest, smallest + 30, 5):
            pool_blocks = rng.choice([1 + cap, 2 * cap, 3 * cap + 7])
            scheduler, requests, stall = _run(num_blocks, pool_blocks, prompts, 160)
            if stall:
                stalled.append((prompts, num_blocks, pool_blocks))
            else:
                _assert_drained(scheduler, requests)
    assert stalled == []


def test_store_saves_pin_blocks_in_their_own_pool():
    """A store job references the drafter's blocks in the drafter's pool: one
    reference each while the job runs, released when every rank reports it."""
    scheduler, store = _pool_scheduler_factory(40)(200, True)
    shared, private = scheduler.kv_cache_manager.coordinator.block_pools
    store_scheduler_ = scheduler.connector.sub_connectors[1].connector_scheduler
    request_meta = store_scheduler.ReqMeta(
        req_id="r",
        token_len_chunk=0,
        block_ids=([1, 2], [1, 3]),
        block_hashes=[],
        can_save=True,
        boundary_puts=[store_scheduler.BoundaryPut(1, 5, 32)],
    )
    request_meta.token_len_chunk = 32
    store_scheduler_._reference_save_blocks([request_meta])
    job = request_meta.store_job_id
    assert store_scheduler_._pinned_saves[job][0] == [1, 2]
    pinned = store_scheduler_._pinned_private_blocks[job]
    assert sorted(b.block_id for b in pinned) == [1, 3, 5]
    assert all(b.pool() is private for b in pinned)
    assert [shared.blocks[i].ref_cnt for i in (1, 2, 3)] == [1, 1, 0]
    assert [private.blocks[i].ref_cnt for i in (1, 2, 3, 5)] == [1, 0, 1, 1]
    store_scheduler_.update_connector_output(
        type(
            "Out",
            (),
            {
                "kv_connector_worker_meta": store_scheduler.MooncakeStoreWorkerMetadata(
                    completed_saves={job: store_scheduler_._num_workers}
                )
            },
        )()
    )
    assert job not in store_scheduler_._pinned_saves
    assert job not in store_scheduler_._pinned_private_blocks
    assert all(private.blocks[i].ref_cnt == 0 for i in (1, 3, 5))
    assert all(shared.blocks[i].ref_cnt == 0 for i in (1, 2))
    assert private.get_num_free_blocks() == private.num_gpu_blocks - 1
    assert shared.get_num_free_blocks() == shared.num_gpu_blocks - 1
