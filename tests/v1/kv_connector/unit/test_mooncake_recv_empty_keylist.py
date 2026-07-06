# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression test: the Mooncake store recv thread must report a load as
finished even when there is nothing to fetch on this rank.

A WAITING_FOR_REMOTE_KVS request is only promoted once *every* TP rank reports
it via the KVOutputAggregator. If a rank's recv thread returns without calling
``set_finished_request`` (previously: a ``ZeroDivisionError`` from
``tp_rank % len(key_list)`` on an empty ``key_list``), the request is stranded
forever and its blocks leak, deadlocking admission under TP>1 (Running=0,
Waiting>0).
"""

import queue
import threading
from types import SimpleNamespace

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.worker import (
    KVCacheStoreRecvingThread,
    LookupKeyClient,
)


def _make_thread(tp_rank: int) -> KVCacheStoreRecvingThread:
    # Bypass __init__ (needs a real MooncakeDistributedStore); wire only the
    # attributes _handle_request touches.
    t = KVCacheStoreRecvingThread.__new__(KVCacheStoreRecvingThread)
    t.tp_rank = tp_rank
    t.block_size = 16
    t.token_databases = []  # no groups -> key_list stays empty
    t.coord = SimpleNamespace(load_mask=lambda block_hashes, token_len: ())
    t.usable_disk_offload_buffer_budget_bytes = None
    t.done_task_lock = threading.Lock()
    t.finished_requests = set()
    t.request_queue = queue.Queue()
    return t


def test_empty_keylist_reports_finished():
    for tp_rank in (0, 1, 3):
        t = _make_thread(tp_rank)
        req_meta = SimpleNamespace(
            req_id="req-abc",
            block_hashes=[],
            block_ids=(),
            load_spec=SimpleNamespace(token_len=64, vllm_cached_tokens=0),
        )
        # Mirror the real run() loop bookkeeping so task_done() is balanced.
        t.request_queue.put(req_meta)
        t.request_queue.get()

        # Must not raise (previously ZeroDivisionError on empty key_list).
        t._handle_request(req_meta)

        assert "req-abc" in t.get_and_clear_finished_requests()
        # task_done() was called exactly once for the single queued item.
        t.request_queue.join()


def _make_lookup_client(timeout_s: float) -> LookupKeyClient:
    # Bypass __init__ (needs real ZMQ context/subprocess); wire only what
    # lookup()/_collect() touch. The fake socket never returns a reply, so an
    # async lookup stays pending until the fail-open deadline.
    c = LookupKeyClient.__new__(LookupKeyClient)
    c._pending = {}
    c._results = {}
    c._pending_since = {}
    c._lookup_seq = 0
    c._block_timeout_s = timeout_s
    c._proc_dead = False
    c._proc = SimpleNamespace(is_alive=lambda: True)
    c._sock = SimpleNamespace(
        send_multipart=lambda *a, **k: None,
        poll=lambda timeout_ms: 0,  # never any reply
    )
    return c


def test_async_lookup_fails_open_when_subprocess_unresponsive():
    import time

    c = _make_lookup_client(timeout_s=0.05)
    block_hashes = [b"\x00" * 8]

    # First async call sends the request and returns None (reply pending).
    assert c.lookup("req-x", 64, block_hashes, non_block=True) is None
    assert "req-x" in c._pending

    # Still within the deadline: keeps waiting (None), does not fail open.
    assert c.lookup("req-x", 64, block_hashes, non_block=True) is None
    assert "req-x" in c._pending

    # Past the deadline with no reply: fail open with 0 hits so the scheduler
    # can admit the request instead of stalling admission forever.
    time.sleep(0.06)
    assert c.lookup("req-x", 64, block_hashes, non_block=True) == 0
    assert "req-x" not in c._pending
    assert "req-x" not in c._pending_since
