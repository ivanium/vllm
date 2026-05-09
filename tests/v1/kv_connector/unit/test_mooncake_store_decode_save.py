# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit test for the decode-tail KV offload fix in MooncakeStoreScheduler.

Bug: Gate B in build_connector_meta filtered all decode-phase tokens via
    if num_computed_token >= len(request.prompt_token_ids):
        continue
so decode-extended blocks never produced ReqMeta -> worker never called
batch_put -> response KV stayed on GPU only.

Fix: replace Gate B with an async-safe block-hash clamp + "no new full block"
short-circuit. Decode-phase block boundaries now emit ReqMeta naturally; the
ReqMeta.from_request_tracker chunk_boundary already handles sub-block
partials.

This test exercises the build_connector_meta path with a request whose
num_computed_tokens has already passed prompt length (pure decode), and
verifies that:
  * post-fix: meta.requests is non-empty + can_save=True for the decode req
  * pre-fix (regression): would have returned empty
  * negative: kv_role="kv_consumer" still skips (Gate A path unchanged)
"""

from unittest.mock import MagicMock

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_store_data import (
    RequestTracker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_store_scheduler import (
    MooncakeStoreScheduler,
)


def _make_scheduler(kv_role: str, block_size: int = 16) -> MooncakeStoreScheduler:
    """Build a MooncakeStoreScheduler without running __init__ (skips ZMQ + Mooncake C++ setup)."""
    sched = MooncakeStoreScheduler.__new__(MooncakeStoreScheduler)
    sched.kv_role = kv_role
    sched.load_async = False
    sched.client = MagicMock()
    sched.load_specs = {}
    sched.pcp_size = 1
    sched.dcp_size = 1
    sched.original_block_size = block_size
    sched._block_size = block_size
    sched._request_trackers = {}
    sched._preempted_req_ids = set()
    sched._discard_partial_chunks = True
    sched._unfinished_requests = {}
    sched._unfinished_request_ids = set()
    return sched


def _make_decode_request(
    req_id: str,
    prompt_len: int = 32,
    decode_len: int = 16,
    block_size: int = 16,
):
    """Build a fake Request with prompt + decode tokens already appended.

    block_hashes is pre-populated to cover (prompt_len + decode_len), simulating
    the natural extension that Request.append_output_token_ids does in
    vllm/v1/request.py:200-216.
    """
    total_tokens = prompt_len + decode_len
    num_blocks = total_tokens // block_size

    request = MagicMock()
    request.request_id = req_id
    request.prompt_token_ids = list(range(prompt_len))
    # all_token_ids = prompt + generated tokens
    request.all_token_ids = list(range(total_tokens))
    request.num_tokens = total_tokens
    # block_hashes auto-extends on output token append (vllm/v1/request.py:211)
    request.block_hashes = [f"hash-{i}".encode() for i in range(num_blocks)]
    request.num_computed_tokens = total_tokens - 1
    return request


def _make_cached_request_data(
    req_ids: list[str],
    num_computed_tokens: list[int],
    new_block_ids: list[list[int]],
):
    cached = MagicMock()
    cached.req_ids = req_ids
    cached.num_computed_tokens = num_computed_tokens
    cached.new_block_ids = new_block_ids
    cached.resumed_req_ids = set()
    cached.new_token_ids = [[] for _ in req_ids]
    cached.all_token_ids = {}
    cached.num_output_tokens = [0 for _ in req_ids]
    return cached


def _make_scheduler_output(
    cached_reqs,
    num_scheduled_tokens: dict[str, int],
):
    so = MagicMock()
    so.scheduled_new_reqs = []
    so.scheduled_cached_reqs = cached_reqs
    so.num_scheduled_tokens = num_scheduled_tokens
    so.finished_req_ids = []
    so.preempted_req_ids = []
    return so


def test_decode_phase_request_emits_reqmeta_post_fix():
    """Post-fix: a request whose num_computed_tokens > len(prompt) (pure decode)
    should still produce a save-enabled ReqMeta when new blocks have crossed.

    Pre-fix this returned empty because Gate B at scheduler:281-283 had a
    `continue` once past prompt length.
    """
    block_size = 16
    prompt_len = 32  # 2 blocks of prompt
    decode_len = 16  # +1 full decode block

    sched = _make_scheduler(kv_role="kv_both", block_size=block_size)
    req = _make_decode_request(
        "decode-req-1",
        prompt_len=prompt_len,
        decode_len=decode_len,
        block_size=block_size,
    )
    sched._unfinished_requests[req.request_id] = (req, [0, 1, 2])
    sched._unfinished_request_ids.add(req.request_id)
    # Prefill saved 2 prompt blocks (32 tokens) earlier
    sched._request_trackers[req.request_id] = RequestTracker(
        req_id=req.request_id,
        token_len=prompt_len,
        allocated_block_ids=[0, 1],
        num_saved_tokens=prompt_len,
        token_ids=list(range(prompt_len)),
    )

    cached = _make_cached_request_data(
        req_ids=[req.request_id],
        num_computed_tokens=[prompt_len],  # already past prompt, in decode
        new_block_ids=[[2]],  # one new block from decode
    )
    so = _make_scheduler_output(
        cached_reqs=cached,
        num_scheduled_tokens={req.request_id: decode_len},
    )

    meta = sched.build_connector_meta(so)

    assert len(meta.requests) == 1, (
        f"Expected 1 ReqMeta from decode-phase request, got {len(meta.requests)}. "
        "If 0, Gate B fix did not apply — the decode-phase 'continue' is back."
    )
    rm = meta.requests[0]
    assert rm.req_id == req.request_id
    assert rm.can_save is True, "decode-phase ReqMeta must have can_save=True"
    # token_len_chunk should be block-aligned across prompt + decode
    assert rm.token_len_chunk == prompt_len + decode_len
    # block_ids should include the new decode block
    assert 2 in rm.block_ids


def test_kv_consumer_role_still_skips_decode_save():
    """Negative test: kv_consumer role (decode-only worker, no save) must
    still skip — Gate A at scheduler:212 (`if not force_skip_save:`) is
    unchanged by the fix.
    """
    block_size = 16
    sched = _make_scheduler(kv_role="kv_consumer", block_size=block_size)
    req = _make_decode_request("consumer-req-1", 32, 16, block_size)
    sched._unfinished_requests[req.request_id] = (req, [0, 1, 2])
    sched._unfinished_request_ids.add(req.request_id)
    sched._request_trackers[req.request_id] = RequestTracker(
        req_id=req.request_id,
        token_len=32,
        allocated_block_ids=[0, 1],
        num_saved_tokens=32,
        token_ids=list(range(32)),
    )

    cached = _make_cached_request_data(
        req_ids=[req.request_id],
        num_computed_tokens=[32],
        new_block_ids=[[2]],
    )
    so = _make_scheduler_output(
        cached_reqs=cached,
        num_scheduled_tokens={req.request_id: 16},
    )

    meta = sched.build_connector_meta(so)
    assert len(meta.requests) == 0, (
        "kv_consumer must not emit save ReqMeta — Gate A path was unintentionally relaxed."
    )


def test_async_clamp_when_block_hashes_lag():
    """Async-safe clamp: when scheduler ticks before Request.block_hashes has
    been extended (the upstream commit 7ee5d5093b race), token_len must be
    clamped to len(block_hashes) * block_size to avoid out-of-bounds reads in
    ReqMeta.from_request_tracker.
    """
    block_size = 16
    prompt_len = 32

    sched = _make_scheduler(kv_role="kv_both", block_size=block_size)
    req = _make_decode_request("lag-req-1", prompt_len=prompt_len, decode_len=16, block_size=block_size)
    # Simulate the lag: only 2 block_hashes (prompt) even though 16 decode
    # tokens are claimed by num_scheduled_tokens.
    req.block_hashes = req.block_hashes[:2]
    sched._unfinished_requests[req.request_id] = (req, [0, 1, 2])
    sched._unfinished_request_ids.add(req.request_id)
    sched._request_trackers[req.request_id] = RequestTracker(
        req_id=req.request_id,
        token_len=prompt_len,
        allocated_block_ids=[0, 1],
        num_saved_tokens=prompt_len,
        token_ids=list(range(prompt_len)),
    )

    cached = _make_cached_request_data(
        req_ids=[req.request_id],
        num_computed_tokens=[prompt_len],
        new_block_ids=[[2]],
    )
    so = _make_scheduler_output(
        cached_reqs=cached,
        num_scheduled_tokens={req.request_id: 16},
    )

    # Should not raise IndexError; should short-circuit the save (no new block
    # has hash yet).
    meta = sched.build_connector_meta(so)
    # No new save was possible because hashed_cap = 2 * 16 = 32 = num_saved_tokens
    assert len(meta.requests) == 0, (
        "When block_hashes lag, scheduler must clamp + skip rather than emit "
        "a ReqMeta that the worker cannot resolve."
    )
