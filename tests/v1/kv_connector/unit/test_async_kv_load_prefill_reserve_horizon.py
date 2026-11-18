# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_ASYNC_KV_LOAD_PREFILL_RESERVE_CHUNKS: against async KV loads (here a
prefill instance's KV cache store hits), an in-flight prefill reserves only the
blocks its next chunks need, not its whole remaining sequence.

The scheduler is the real one, with a prefill instance's NIXL +
MooncakeStoreConnector MultiConnector; store lookups are stubbed."""

from dataclasses import dataclass, field
from types import SimpleNamespace
from unittest.mock import patch

import pytest

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
from vllm.v1.core.sched import scheduler as core_scheduler
from vllm.v1.metrics.stats import KV_FETCH_STAGES
from vllm.v1.request import RequestStatus

from .utils import (
    assert_scheduler_empty,
    create_model_runner_output,
    create_request,
    create_scheduler,
    create_vllm_config,
)

pytestmark = pytest.mark.cpu_test

ENV = "VLLM_ASYNC_KV_LOAD_PREFILL_RESERVE_CHUNKS"
BLOCK = 16
# A prefill instance's shape: a long prefill is capped at CHUNK tokens per step
# (long_prefill_token_threshold) while other requests wait, out of a BUDGET of
# max_num_batched_tokens. A reserved "chunk" is a whole BUDGET (8 blocks).
CHUNK = 64
BUDGET = 128

# A prefill instance's kv-transfer-config, minus the HBM pool.
P_CONNECTORS = {
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
    """LookupKeyClient stand-in: a fixed store hit per request."""

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


def _scheduler(num_blocks: int, spec_tokens: int = 0):
    vllm_config = create_vllm_config(
        kv_connector="MultiConnector",
        kv_role="kv_both",
        kv_connector_extra_config=P_CONNECTORS,
        kv_load_failure_policy="recompute",
        max_num_seqs=16,
        max_num_batched_tokens=BUDGET,
        block_size=BLOCK,
    )
    vllm_config.scheduler_config.long_prefill_token_threshold = CHUNK
    vllm_config.scheduler_config.max_num_inflight_kv_loads = 8
    vllm_config.scheduler_config.max_async_kv_loads_per_step = 4
    stub = _StoreStub()
    with patch.object(store_scheduler, "LookupKeyClient", return_value=stub):
        scheduler = create_scheduler(vllm_config, num_blocks=num_blocks)
    if spec_tokens:
        # The CPU model runner rejects a speculative config; emulate the
        # scheduler's spec decode block accounting (EAGLE-style lookahead).
        scheduler.num_spec_tokens = scheduler.num_lookahead_tokens = spec_tokens
        scheduler._spec_decode_step_blocks = -(-(1 + 2 * spec_tokens) // BLOCK)
    return scheduler, stub


def _p_request(stub, num_tokens: int, hit: int):
    """A prefill-instance request (NIXL do_remote_decode, max_tokens 1)."""
    request = create_request(
        num_tokens=num_tokens, block_size=BLOCK, do_remote_decode=True
    )
    stub.hits[request.request_id] = hit
    return request


@dataclass
class _Run:
    """What a scenario did, step by step, by request in arrival order."""

    scheduler: object
    long_prefill: object
    hits: list
    waiting_to_start: list[int] = field(default_factory=list)
    preempted: list[str] = field(default_factory=list)
    load_start_step: dict[str, int] = field(default_factory=dict)
    finish_step: dict[str, int] = field(default_factory=dict)
    lends_by_step: list[int] = field(default_factory=list)
    backoff_by_step: list[bool] = field(default_factory=list)
    free_by_step: list[int] = field(default_factory=list)
    shortfalls_by_step: list[int] = field(default_factory=list)

    @property
    def prefill_end(self) -> int | None:
        return self.finish_step.get(self.long_prefill.request_id)

    @property
    def hit_ends(self) -> list[int | None]:
        return [self.finish_step.get(r.request_id) for r in self.hits]

    @property
    def hit_load_starts(self) -> list[int | None]:
        return [self.load_start_step.get(r.request_id) for r in self.hits]


class _Driver:
    """Steps the scheduler like a prefill instance's engine: a load takes
    `load_steps` steps, store saves finish within a step, and the decode side
    reads (frees) a finished request's KV `pull_delay` steps after it finishes
    (None: never)."""

    def __init__(self, scheduler, load_steps: int, pull_delay: int | None):
        self.scheduler = scheduler
        self.store = scheduler.connector.sub_connectors[1].connector_scheduler
        self.load_steps = load_steps
        self.pull_delay = pull_delay
        self.loading: dict[str, int] = {}
        self.finished: dict[str, int] = {}
        self.seen_finished: set[str] = set()
        self.step = 0

    def __call__(self, output, watched):
        """Run one step after `output`; return the next output."""
        self.step += 1
        step, scheduler = self.step, self.scheduler
        scheduled = [scheduler.requests[i] for i in output.num_scheduled_tokens]
        recving = {i for i, s in self.loading.items() if step - s >= self.load_steps}
        sending = {
            i
            for i, s in self.finished.items()
            if self.pull_delay is not None and step - s >= self.pull_delay
        }
        for i in recving:
            del self.loading[i]
        for i in sending:
            del self.finished[i]
        # Every rank finished the store saves the last step pinned blocks for.
        saves = MooncakeStoreWorkerMetadata(
            completed_saves={job: n for job, (_, n) in self.store._pinned_saves.items()}
        )
        model_output = create_model_runner_output(
            scheduled,
            finished_recving=recving or None,
            finished_sending=sending or None,
            kv_connector_worker_meta=MultiKVConnectorWorkerMetadata((None, saves)),
        )
        # A step that leaves a prefill unfinished samples no token for it.
        model_output.sampled_token_ids = [
            [] if r.is_prefill_chunk else tokens
            for r, tokens in zip(scheduled, model_output.sampled_token_ids)
        ]
        scheduler.update_from_output(output, model_output)
        for request in watched:
            if request.is_finished() and request.request_id not in self.seen_finished:
                self.seen_finished.add(request.request_id)
                self.finished[request.request_id] = step
        output = scheduler.schedule()
        for request in watched:
            if (
                request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
                and request.request_id not in self.loading
            ):
                self.loading[request.request_id] = step
        return output


def _warm_up(scheduler, stub, drive):
    """One store hit loads, computes and has its KV read, so the scheduler has
    measured how long a load holds its blocks."""
    request = _p_request(stub, 80, 64)
    scheduler.add_request(request)
    output = scheduler.schedule()
    for _ in range(200):
        if request.request_id not in scheduler.requests:
            return
        output = drive(output, [request])
    raise AssertionError("the warm-up request never returned its KV")


def _run(
    monkeypatch,
    chunks: str | None,
    *,
    warmup: bool = True,
    warmup_pull_delay: int | None = None,
    num_blocks: int = 84,
    long_tokens: int = 1280,
    num_hits: int = 4,
    arrivals: int = 0,
    hit_tokens: int = 64,
    tail_tokens: int = 16,
    load_steps: int = 1,
    pull_delay: int | None = 1,
    steps: int = 30,
    spec_tokens: int = 0,
) -> _Run:
    """A long store-miss prefill runs at the chunk cap; store hits arrive after
    its first chunk, and `arrivals` more each step. A load takes `load_steps`
    steps, and a store save one. The decode instance reads (frees) a finished
    request's blocks `pull_delay` steps after it finishes (None: never).
    `warmup`: one store hit is served first, its KV read `warmup_pull_delay`
    (default `pull_delay`, or 1 if that is None) steps after it finishes."""
    if chunks is None:
        monkeypatch.delenv(ENV, raising=False)
    else:
        monkeypatch.setenv(ENV, chunks)
    scheduler, stub = _scheduler(num_blocks, spec_tokens)
    if warmup:
        if warmup_pull_delay is None:
            warmup_pull_delay = 1 if pull_delay is None else pull_delay
        _warm_up(scheduler, stub, _Driver(scheduler, load_steps, warmup_pull_delay))
    drive = _Driver(scheduler, load_steps, pull_delay)
    long_prefill = _p_request(stub, long_tokens, hit=0)
    scheduler.add_request(long_prefill)
    output = scheduler.schedule()

    def new_hits(n: int) -> list:
        hits = [
            _p_request(stub, hit_tokens + tail_tokens, hit_tokens) for _ in range(n)
        ]
        for request in hits:
            scheduler.add_request(request)
        return hits

    hits = new_hits(num_hits)
    run = _Run(scheduler, long_prefill, hits)
    for step in range(1, steps + 1):
        hits.extend(new_hits(arrivals))
        output = drive(output, [long_prefill, *hits])
        assert drive.step == step
        for request in (long_prefill, *hits):
            if request.is_finished() and request.request_id not in run.finish_step:
                run.finish_step[request.request_id] = step
        run.preempted.extend(output.preempted_req_ids or ())
        for request in hits:
            if (
                request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
                and request.request_id not in run.load_start_step
            ):
                run.load_start_step[request.request_id] = step
        stages = dict(zip(KV_FETCH_STAGES, scheduler._kv_fetch_counts.values()))
        run.waiting_to_start.append(stages["waiting_to_start"])
        # getattr: the default arms also run on a scheduler without the switch.
        run.lends_by_step.append(getattr(scheduler, "_num_kv_load_lends", 0))
        run.backoff_by_step.append(getattr(scheduler, "_kv_load_lend_backoff", False))
        run.free_by_step.append(
            scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
        )
        run.shortfalls_by_step.append(
            getattr(scheduler, "_num_prefill_kv_shortfalls", 0)
        )
    return run


# ------------------------------------------------------------------ the switch


@pytest.mark.parametrize(
    "value, expected_chunks",
    [(None, 0), ("", 0), ("0", 0), ("-3", 0), ("x", 0), ("1", 1), ("8", 8)],
)
def test_switch_and_startup_line(monkeypatch, value, expected_chunks):
    infos: list[str] = []
    monkeypatch.setattr(
        core_scheduler.logger, "info", lambda msg, *a: infos.append(msg % a)
    )
    if value is None:
        monkeypatch.delenv(ENV, raising=False)
    else:
        monkeypatch.setenv(ENV, value)
    scheduler, _ = _scheduler(num_blocks=100)
    # A reserved chunk is the most one step schedules (max_num_batched_tokens).
    assert scheduler._prefill_reserve_horizon_tokens == expected_chunks * BUDGET
    lines = [m for m in infos if ENV in m]
    assert len(lines) == (1 if expected_chunks else 0)
    if expected_chunks:
        tokens = expected_chunks * BUDGET
        assert f"next {expected_chunks} chunks ({tokens} tokens)" in lines[0]


def test_switch_needs_a_connector_and_no_hisparse(monkeypatch):
    monkeypatch.setenv(ENV, "4")
    scheduler, _ = _scheduler(num_blocks=100)
    get = core_scheduler.Scheduler._get_prefill_reserve_horizon
    assert get(scheduler, SimpleNamespace(hisparse_host_num_blocks=None)) == 4 * BUDGET
    assert get(scheduler, SimpleNamespace(hisparse_host_num_blocks=8)) == 0
    scheduler.connector = None
    assert get(scheduler, SimpleNamespace(hisparse_host_num_blocks=None)) == 0


# ------------------------------------------------------------- the mechanism
# 1,280-token prefill (80 blocks) on an 83-block pool; store hits of 64 tokens
# (4 blocks) plus a 16-token tail. `_run` serves one store hit first (warm-up),
# so the scheduler has measured how long a load holds its KV.


def test_full_reservation_refuses_store_hits_beside_a_long_prefill(monkeypatch):
    """Default: the long prefill's whole remaining sequence is reserved, so the
    store hit at the queue head waits (waiting_to_start) with most of the KV
    free, the hits behind it wait on it, and the prefill runs at the chunk cap
    because they count as eligible."""
    monkeypatch.setattr(
        core_scheduler.Scheduler,
        "_request_horizon_blocks",
        lambda *_: pytest.fail("the default path computes no horizon"),
        raising=False,
    )
    run = _run(monkeypatch, None)
    end = run.prefill_end
    assert end == 19  # 128 tokens alone, then 64 per step
    assert run.free_by_step[0] == 71  # 14% of 83 blocks used
    assert min(run.load_start_step.values()) == end + 1
    assert run.waiting_to_start[: end - 1] == [1] * (end - 1)
    assert not run.preempted
    assert run.lends_by_step[-1] == 0


@pytest.mark.parametrize("spec_tokens, num_blocks", [(0, 84), (7, 86)])
@pytest.mark.parametrize("pull_delay", [1, 4])
def test_lending_admits_store_hits_beside_a_long_prefill(
    monkeypatch, spec_tokens, num_blocks, pull_delay
):
    """N=2, loads measured to hold their KV for a few steps: the store hits load
    at once into blocks the long prefill needs only for later chunks, compute,
    and are read back before it needs them. It is never short of KV or
    preempted, and once nothing waits it takes the whole budget, so it finishes
    sooner."""
    kwargs = dict(
        spec_tokens=spec_tokens,
        num_blocks=num_blocks,
        pull_delay=pull_delay,
        warmup_pull_delay=1,
    )
    base = _run(monkeypatch, None, **kwargs)
    run = _run(monkeypatch, "2", **kwargs)
    s = run.scheduler
    assert run.prefill_end < base.prefill_end == 19
    assert run.hit_load_starts == [1] * 4
    assert s._num_kv_load_lends == 4
    assert run.waiting_to_start[0] == 0
    assert run.hit_ends == [3] * 4
    # Default: they load once the prefill's KV has been read, then finish.
    assert base.hit_ends == [21 + pull_delay] * 4
    assert not run.preempted
    assert s._num_prefill_kv_shortfalls == 0
    assert s._num_kv_load_lend_backoffs == 0
    assert not s._kv_load_starts and s._num_kv_loans_out == 0
    assert_scheduler_empty(s)


def test_lending_with_steady_hits_and_slower_reads(monkeypatch):
    """A store hit arrives every step and the decode side reads a finished
    request's KV 3 steps later: some hits load beside the prefill instead of
    after it, with no shortfall, back-off or preemption."""
    base = _run(monkeypatch, None, arrivals=1, pull_delay=3, steps=40)
    run = _run(monkeypatch, "2", arrivals=1, pull_delay=3, steps=40)
    assert run.prefill_end == base.prefill_end == 19
    assert min(base.load_start_step.values()) == 22
    early = [step for step in run.load_start_step.values() if step < 19]
    assert len(early) == run.scheduler._num_kv_load_lends == 5
    assert sum(e is not None for e in run.hit_ends) >= sum(
        e is not None for e in base.hit_ends
    )
    assert not run.preempted
    assert run.scheduler._num_prefill_kv_shortfalls == 0
    assert run.scheduler._num_kv_load_lend_backoffs == 0


def test_no_lending_before_a_load_has_returned(monkeypatch):
    """Without a measured load hold time, nothing is lent."""
    run = _run(monkeypatch, "2", warmup=False)
    assert run.lends_by_step[-1] == 0
    assert run.prefill_end == 19
    assert run.scheduler._kv_load_hold_estimate() is not None  # hits returned


@pytest.mark.parametrize("chunks", ["2", "16"])
@pytest.mark.parametrize("warmup", [False, True])
@pytest.mark.parametrize("pull_delay", [16, 30])
def test_lagging_reads_finish_as_many_hits(monkeypatch, pull_delay, warmup, chunks):
    """A store hit every step for 150 steps, the decode side
    reading KV 16 or 30 steps late. Lending must finish at least as many hits
    as the default, and must not preempt the long prefill."""
    kwargs = dict(arrivals=1, pull_delay=pull_delay, steps=150, warmup=warmup)
    base = _run(monkeypatch, None, **kwargs)
    run = _run(monkeypatch, chunks, **kwargs)
    done = sum(e is not None for e in run.hit_ends)
    base_done = sum(e is not None for e in base.hit_ends)
    assert base_done == {16: 112, 30: 64}[pull_delay]
    assert done >= base_done
    assert run.long_prefill.request_id not in run.preempted
    assert run.prefill_end <= base.prefill_end


@pytest.mark.parametrize("pull_delay", [16, 30])
def test_reads_slowing_down_preempt_a_prefill_at_most_once(monkeypatch, pull_delay):
    """Loads were measured to return in a few steps, then the decode side
    falls 16 or 30 steps behind: loans made on the fast estimate outlive the
    runway and the prefill preempts once. It then keeps its full reservation, so
    no loan borrows from it again (no repeated self-preemption)."""
    run = _run(
        monkeypatch,
        "2",
        arrivals=1,
        pull_delay=pull_delay,
        warmup_pull_delay=1,
        steps=150,
    )
    lp = run.long_prefill
    assert run.preempted.count(lp.request_id) == 1
    assert lp.num_preemptions == 1
    assert run.prefill_end is not None
    s = run.scheduler
    assert s._num_prefill_kv_shortfalls_with_loans == s._num_prefill_kv_shortfalls == 1


@pytest.mark.parametrize("chunks", ["2", "4"])
def test_lending_backs_off_before_a_shortfall_when_kv_is_never_read(
    monkeypatch, chunks
):
    """Lent blocks return only once the decode side reads them. When it stops
    reading, lending stops (the runway no longer fits a load), then backs off,
    before the prefill runs short. The guarantee is not by construction: with
    nothing ever returned, the prefill then preempts, once."""
    run = _run(monkeypatch, chunks, arrivals=1, pull_delay=None, steps=24)
    s = run.scheduler
    last_lend = run.lends_by_step.index(s._num_kv_load_lends)
    backoff = run.backoff_by_step.index(True)
    shortfall = run.shortfalls_by_step.index(1)
    assert last_lend < backoff <= shortfall
    assert s._num_kv_load_lend_backoffs == 1
    assert run.preempted == [run.long_prefill.request_id]


def test_lending_identical_when_the_full_reservation_fits(monkeypatch):
    """With room for the full reservation, every decision is the default's."""
    base = _run(monkeypatch, None, num_blocks=200)
    run = _run(monkeypatch, "2", num_blocks=200)
    assert run.hit_load_starts == base.hit_load_starts
    assert run.hit_ends == base.hit_ends
    assert run.prefill_end == base.prefill_end
    assert run.waiting_to_start == base.waiting_to_start
    assert run.free_by_step == base.free_by_step
    assert run.scheduler._num_kv_load_lends == 0


def test_parked_and_preempted_keep_their_whole_reservation(monkeypatch):
    """Lending covers in-flight prefills only, never parked loads or a prefill
    that has been preempted; nothing is lent before a hold time is measured."""
    monkeypatch.setenv(ENV, "1")
    scheduler, stub = _scheduler(num_blocks=100)
    scheduler._parked_reserved_blocks = 7
    assert scheduler._inflight_prefill_reserved_blocks() == 7
    long_prefill = _p_request(stub, 640, hit=0)
    scheduler.add_request(long_prefill)
    scheduler.schedule()  # alone: the whole 128-token budget (8 blocks)
    assert scheduler._request_remaining_blocks(long_prefill) == 32
    assert scheduler._inflight_prefill_reserved_blocks() == 7 + 32
    scheduler._kv_load_hold_steps.append(0)
    # 7 parked + the long prefill's next chunk of 128 tokens (8 blocks).
    assert scheduler._inflight_prefill_reserved_blocks() == 7 + 8
    assert scheduler._kv_load_full_reserved_blocks == 7 + 32
    # Loads measured to hold their KV for 2 steps: 3 chunks (24 blocks).
    scheduler._kv_load_hold_steps.append(2)
    assert scheduler._inflight_prefill_reserved_blocks() == 7 + 24
    long_prefill.num_preemptions = 1
    assert scheduler._inflight_prefill_reserved_blocks() == 7 + 32


def test_no_lending_while_a_preempted_prefill_is_in_flight(monkeypatch):
    """A preempted prefill could still be the `running[-1]`
    victim of another prefill's late loans, so nothing is lent on the rank
    while one is in flight."""
    monkeypatch.setenv(ENV, "1")
    scheduler, stub = _scheduler(num_blocks=200)
    scheduler._kv_load_hold_steps.append(0)
    first, second = _p_request(stub, 640, hit=0), _p_request(stub, 640, hit=0)
    scheduler.add_request(first)
    scheduler.add_request(second)
    scheduler.schedule()  # 64 + 64 tokens: both capped, 36 blocks left each
    assert scheduler._inflight_prefills == {first, second}
    assert scheduler._inflight_prefill_reserved_blocks() == 8 + 8
    second.num_preemptions = 1
    assert scheduler._inflight_prefill_reserved_blocks() == 36 + 36
    assert scheduler._kv_load_horizon_tokens == 0


def test_hold_estimate_counts_loads_still_out(monkeypatch):
    monkeypatch.setenv(ENV, "1")
    scheduler, _ = _scheduler(num_blocks=100)
    assert scheduler._kv_load_hold_estimate() is None
    scheduler._kv_load_hold_steps.extend([3, 5, 4])
    assert scheduler._kv_load_hold_estimate() == 5
    scheduler._kv_load_starts["a"] = (scheduler.current_step, 0.0, True)
    scheduler.current_step += 9
    assert scheduler._kv_load_hold_estimate() == 9


def test_default_reservation_unchanged(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    scheduler, stub = _scheduler(num_blocks=100)
    long_prefill = _p_request(stub, 640, hit=0)
    scheduler.add_request(long_prefill)
    scheduler.schedule()
    assert scheduler._inflight_prefill_reserved_blocks() == 32


def test_shortfalls_once_per_step_and_split_by_loans(monkeypatch):
    """A shortfall the default also has (a second, non-load prefill crowds the
    long one out) counts once per step, as without loans; the preemptions are
    the default's."""
    kwargs = dict(num_hits=1, hit_tokens=0, tail_tokens=640, steps=40)
    base = _run(monkeypatch, None, **kwargs)
    run = _run(monkeypatch, "2", **kwargs)
    s = run.scheduler
    assert len(run.preempted) == len(base.preempted) == 1
    assert run.prefill_end == base.prefill_end
    assert s._num_kv_load_lends == 0
    assert s._num_prefill_kv_shortfalls == 1
    assert s._num_prefill_kv_shortfalls_with_loans == 0

    s._inflight_prefills.add(run.long_prefill)
    for _ in range(3):
        s._record_prefill_kv_shortfall(run.long_prefill)
    assert s._num_prefill_kv_shortfalls == 2
    s._num_kv_loans_out = 1
    s.current_step += 1
    s._record_prefill_kv_shortfall(run.long_prefill)
    assert s._num_prefill_kv_shortfalls == 3
    assert s._num_prefill_kv_shortfalls_with_loans == 1


# ------------------------------------------------------------------ reporting


def test_engagement_and_periodic_lending_lines(monkeypatch):
    infos: list[str] = []
    monkeypatch.setattr(
        core_scheduler.logger, "info", lambda msg, *a: infos.append(msg % a)
    )
    run = _run(monkeypatch, "2", steps=6)
    s = run.scheduler
    engaged = [m for m in infos if m.startswith("Admitted an async KV load")]
    assert len(engaged) == 1
    assert not [m for m in infos if m.startswith("Async KV loads admitted")]
    s._kv_load_lend_log_time -= 61
    s._maybe_log_kv_load_lending()
    (line,) = [m for m in infos if m.startswith("Async KV loads admitted")]
    assert "last 61 s: 4 (most blocks borrowed: " in line
    # Scheduler steps from the load's start until the read frees its KV; the
    # warm-up load (whose KV was read a step later) set the estimate.
    assert "loans returned: 4, lifetime mean 2.0 / max 2 steps" in line
    assert "loans outstanding: 0; load hold estimate: 3 steps, runway 512" in line
    assert "shortfalls: 0 with loans outstanding, 0 without" in line
    assert "(since start: 4 loans, 0 back-offs, 0 shortfalls with loans" in line


def test_lending_line_on_a_rank_that_lends_nothing(monkeypatch):
    """When the measured runway vetoes every loan, the
    per-minute line still shows the hold estimate and runway (here with the
    decode side reading 16 steps late, one simulated second per step)."""
    infos: list[str] = []
    monkeypatch.setattr(
        core_scheduler.logger, "info", lambda msg, *a: infos.append(msg % a)
    )
    clock = [0.0]
    monkeypatch.setattr(core_scheduler.time, "monotonic", lambda: clock[0])
    step = _Driver.__call__

    def timed_step(self, output, watched):
        clock[0] += 1.0
        return step(self, output, watched)

    monkeypatch.setattr(_Driver, "__call__", timed_step)
    run = _run(monkeypatch, "2", arrivals=1, pull_delay=16, steps=150)
    assert run.scheduler._num_kv_load_lends == 0
    lines = [m for m in infos if m.startswith("Async KV loads admitted")]
    assert len(lines) >= 2  # 150 s of steps, at most one line a minute
    assert all(": 0 (most blocks borrowed: 0); loans returned: 0" in m for m in lines)
    hold = run.scheduler._kv_load_hold_estimate()
    assert hold is not None and hold > 16
    assert "load hold estimate: " in lines[-1] and " steps, runway " in lines[-1]


def test_backoff_warning_once(monkeypatch):
    warnings: list[str] = []
    monkeypatch.setattr(
        core_scheduler.logger, "warning", lambda msg, *a: warnings.append(msg % a)
    )
    _run(monkeypatch, "2", arrivals=1, pull_delay=None, steps=24)
    lines = [m for m in warnings if m.startswith("Async KV loads stop using")]
    assert len(lines) == 1
