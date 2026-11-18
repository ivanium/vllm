# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Async KV loads on a prefill instance whose KV cache has a sliding-window
group beside full attention (a DFlash/DSpark drafter).

A store hit with a long hit and a short tail (an agentic next turn) is parked
while its KV loads, and its reservation keeps later loads from taking the
blocks its promotion needs. The sliding-window group's share of that
reservation was the admission cap less the length of the request's block
table, which is null below the window: past the cap it was 0. A later load
could then take the blocks the head's tail needed in that group. A completed
load that cannot be allocated stops the waiting loop, and with nothing running
no block is ever freed, so the instance stalled for good.
"""

import random
from unittest.mock import patch

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake_store import (
    scheduler as store_scheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake_store.data import (
    MooncakeLookupResult,
    MooncakeStoreWorkerMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
    MultiKVConnectorWorkerMetadata,
)
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import DFlashSlidingWindowManager
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheConfig
from vllm.v1.request import Request, RequestStatus

from .utils import (
    create_model_runner_output,
    create_request,
    create_scheduler,
    create_vllm_config,
    make_kv_cache_group,
)

pytestmark = pytest.mark.cpu_test

BLOCK_SIZE = 16
WINDOW = 64
# A prefill instance's connectors: NIXL for the decode side's reads, and a
# MooncakeStore prefix cache whose hits load asynchronously.
PREFILL_CONNECTORS = {
    "connectors": [
        {
            "kv_connector": "NixlConnector",
            "kv_role": "kv_both",
            "kv_load_failure_policy": "fail",
            "kv_buffer_device": "cuda",
            "kv_connector_extra_config": {"backends": ["UCX"]},
        },
        {
            "kv_connector": "MooncakeStoreConnector",
            "kv_role": "kv_both",
            "kv_load_failure_policy": "recompute",
            "kv_connector_extra_config": {
                "load_async": True,
                "lookup_async": True,
                "enable_offload": False,
            },
        },
    ]
}


class _StoreStub:
    """MooncakeStore lookups that hit a fixed number of tokens per request."""

    def __init__(self) -> None:
        self.hits: dict[str, int] = {}

    def prefetch(self, req_id, num_tokens, block_hashes) -> bool:
        return False

    def discard(self, req_id) -> None:
        pass

    def close(self) -> None:
        pass

    def lookup(self, req_id, num_tokens, block_hashes, non_block=False):
        return MooncakeLookupResult(self.hits.get(req_id, 0))


def _spec() -> AttentionSpec:
    return AttentionSpec(
        block_size=BLOCK_SIZE, num_kv_heads=1, head_size=1, dtype=torch.float32
    )


def _scheduler(num_blocks: int, hybrid: bool) -> tuple[Scheduler, _StoreStub]:
    vllm_config = create_vllm_config(
        kv_connector="MultiConnector",
        kv_role="kv_both",
        kv_connector_extra_config=PREFILL_CONNECTORS,
        kv_load_failure_policy="recompute",
        max_num_seqs=16,
        max_num_batched_tokens=128,
        block_size=BLOCK_SIZE,
        disable_hybrid_kv_cache_manager=not hybrid,
    )
    vllm_config.scheduler_config.long_prefill_token_threshold = 64
    vllm_config.scheduler_config.max_num_inflight_kv_loads = 8
    vllm_config.scheduler_config.max_async_kv_loads_per_step = 4
    if hybrid:
        groups = [
            make_kv_cache_group(["target"], _spec()),
            make_kv_cache_group(
                ["drafter"],
                _spec(),
                DFlashSlidingWindowManager,
                DFlashSlidingWindowManager.config(sliding_window=WINDOW),
            ),
        ]
    else:
        # Without the hybrid KV cache manager the drafter's layers are full
        # attention in the target's group.
        groups = [make_kv_cache_group(["target", "drafter"], _spec())]
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks, kv_cache_tensors=[], kv_cache_groups=groups
    )
    store = _StoreStub()
    with patch.object(store_scheduler, "LookupKeyClient", return_value=store):
        scheduler = create_scheduler(
            vllm_config, num_blocks=num_blocks, kv_cache_config=kv_cache_config
        )
    assert isinstance(scheduler, Scheduler)
    return scheduler, store


def _run(
    num_blocks: int, hybrid: bool, prompts: list[tuple[int, int]], steps: int
) -> tuple[Scheduler, list[Request], bool]:
    """Serve `prompts` ((prompt tokens, store-hit tokens) each) for `steps`
    steps. A load completes one step after it starts, store saves complete at
    once, and the decode side reads a finished request's KV one step after it
    finishes. Returns whether the instance stalled: requests left, a completed
    load waiting, and no token scheduled in the last 30 steps."""
    scheduler, store = _scheduler(num_blocks, hybrid)
    store_connector = scheduler.connector.sub_connectors[1].connector_scheduler
    requests = []
    for num_tokens, num_hit_tokens in prompts:
        request = create_request(
            num_tokens=num_tokens, block_size=BLOCK_SIZE, do_remote_decode=True
        )
        store.hits[request.request_id] = num_hit_tokens
        scheduler.add_request(request)
        requests.append(request)
    loading: dict[str, int] = {}
    finished: dict[str, int] = {}
    seen_finished: set[str] = set()
    scheduled_tokens: list[int] = []
    output = scheduler.schedule()
    for step in range(1, steps + 1):
        scheduled = [scheduler.requests[i] for i in output.num_scheduled_tokens]
        received = {i for i, start in loading.items() if step - start >= 1}
        sent = {i for i, end in finished.items() if step - end >= 1}
        for request_id in received:
            del loading[request_id]
        for request_id in sent:
            del finished[request_id]
        saves = MooncakeStoreWorkerMetadata(
            completed_saves={
                job: n for job, (_, n) in store_connector._pinned_saves.items()
            }
        )
        runner_output = create_model_runner_output(
            scheduled,
            finished_recving=received or None,
            finished_sending=sent or None,
            kv_connector_worker_meta=MultiKVConnectorWorkerMetadata((None, saves)),
        )
        runner_output.sampled_token_ids = [
            [] if request.is_prefill_chunk else tokens
            for request, tokens in zip(scheduled, runner_output.sampled_token_ids)
        ]
        scheduler.update_from_output(output, runner_output)
        for request in requests:
            if request.is_finished() and request.request_id not in seen_finished:
                seen_finished.add(request.request_id)
                finished[request.request_id] = step
        output = scheduler.schedule()
        for request in requests:
            if (
                request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
                and request.request_id not in loading
            ):
                loading[request.request_id] = step
        scheduled_tokens.append(sum(output.num_scheduled_tokens.values()))
    stalled = (
        any(request.request_id in scheduler.requests for request in requests)
        and len(scheduler.kv_holding_waiting) > 0
        and not any(scheduled_tokens[-30:])
    )
    return scheduler, requests, stalled


# Long store hits with short tails, as in agentic sessions' next turns.
SHORT_TAILS = [(640 + 48, 640), (640 + 16, 640)]


def test_parked_load_reserves_the_window_group_tail():
    """A parked load reserves its tail's blocks in both groups, though its
    block table in the sliding-window group is past the admission cap."""
    scheduler, requests, _ = _run(200, True, SHORT_TAILS, steps=1)
    head = requests[0]
    assert head.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    managers = scheduler.kv_cache_manager.coordinator.single_type_managers
    drafter = managers[1]
    drafter_blocks = drafter.req_to_blocks[head.request_id]
    assert drafter.max_admission_blocks_per_request is not None
    assert len(drafter_blocks) > drafter.max_admission_blocks_per_request

    # What promoting the load allocates: the 48-token tail in each group.
    def tail_blocks(manager) -> int:
        return manager.get_num_blocks_to_allocate(
            head.request_id,
            head.num_tokens,
            [],
            head.num_computed_tokens,
            head.num_computed_tokens,
            head.num_tokens,
        )

    assert [tail_blocks(manager) for manager in managers] == [3, 3]
    assert scheduler._parked_reservations[head] == 6


@pytest.mark.parametrize("hybrid", [True, False], ids=["hybrid", "single-group"])
@pytest.mark.parametrize(
    "prompts",
    [
        SHORT_TAILS,
        [(640 + 48, 640), (640 + 16, 640), (320 + 100, 320), (960 + 30, 960)],
        [(1600 + 16, 1600), (800 + 200, 800), (400 + 8, 400)],
        [
            (1000 + 8, 1000),
            (900 + 24, 900),
            (700 + 40, 700),
            (600 + 16, 600),
            (300 + 64, 300),
        ],
        [(1700 + 30, 1700), (1500 + 30, 1500), (200 + 500, 200)],
        [(1900 + 120, 1900), (64 + 300, 64), (1200 + 64, 1200), (500 + 40, 480)],
    ],
)
def test_store_hits_with_short_tails_never_stall(hybrid, prompts):
    """Every pool that fits the longest prompt serves every prompt."""
    smallest = max(-(-num_tokens // BLOCK_SIZE) for num_tokens, _ in prompts) + 16
    stalled = [
        num_blocks
        for num_blocks in range(smallest, smallest + 100)
        if _run(num_blocks, hybrid, prompts, steps=100)[2]
    ]
    assert stalled == []


@pytest.mark.parametrize("hybrid", [True, False], ids=["hybrid", "single-group"])
def test_seeded_store_hit_mixes_never_stall(hybrid):
    """Mixes of 2-5 store hits with tails of 1-64 tokens, and sometimes a store
    miss, over pools from the smallest that fits the longest prompt."""
    rng = random.Random(48)
    stalled = []
    for _ in range(30):
        prompts = []
        for _ in range(rng.randint(2, 5)):
            num_hit_tokens = rng.randrange(128, 1800, BLOCK_SIZE)
            prompts.append((num_hit_tokens + rng.randint(1, 64), num_hit_tokens))
        if rng.random() < 0.5:
            prompts.append((rng.randint(64, 600), 0))
        smallest = max(-(-n // BLOCK_SIZE) for n, _ in prompts) + 16
        stalled += [
            (prompts, num_blocks)
            for num_blocks in range(smallest, smallest + 60)
            if _run(num_blocks, hybrid, prompts, steps=100)[2]
        ]
    assert stalled == []
