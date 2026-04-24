# DEBUG STALL — DP engine instrumentation

Temporary instrumentation added to `vllm/v1/engine/core.py` (search for
`[DEBUG STALL]`) to diagnose the full-pipeline freeze observed in
`pd_dep_mooncake_offload_nixl_400G` 1P2D @ concurrency=256. Revert before
upstreaming.

## Observation prompting this change

In logs/pd_dep_mooncake_offload_nixl_400G/2026-04-23/20260423_230640 the
entire PD pipeline froze in lockstep:

| Minute | prefill NIXL-NOTIF | decode NIXL-NOTIF | router 200 OK |
|---|---|---|---|
| 23:15 | 333 | ~170 | 171 |
| 23:16 | 56 | ~30 | 29 |
| 23:17–23:44 | **0** | **0** | **0** |
| 23:45 | 52 | ~25 | 25 |

~28 minutes of total silence across all three signals (not just router). No
configured timeout matches 28 min (UCX_RC=5s, NCCL=300s). This rules out
"prefill busy, decode idle" — it's a coordinated wait. The instrumentation
below aims to identify which subsystem is blocking.

## What was added

### 1. DP heartbeat thread (default ON, 5 s interval)

In `DPEngineCoreProc.__init__`, a daemon thread that periodically logs, per
DP rank:

```
[DP-HB] dp_rank=N step=S age_since_step=X.Xs age_since_exec=Y.Ys \
    steps_since_exec=Z running=R waiting=W engines_running=B wave=V
```

- `step` — monotonic step counter (goes up whenever the busy loop completes
  a pass, whether it ran a real batch or a dummy one).
- `age_since_step` — seconds since the loop last exited `_process_engine_step`.
  Stuck ≥ 10 s ⇒ loop is blocked in a native call (NCCL, UCX, Mooncake).
- `age_since_exec` — seconds since the last pass that actually executed the
  model (not a dummy batch). Stuck while `age_since_step` is small ⇒ scheduler
  is admitting no work even though the engine is alive.
- `steps_since_exec` — count of consecutive dummy passes. Rising fast while
  `running>0` ⇒ requests are admitted but not dispatched (prefix-cache /
  KV-remote-wait starvation).
- `running / waiting` — scheduler queue counts from `get_request_counts`.
- `engines_running` — the post-allreduce global flag.

Override via `VLLM_DP_HEARTBEAT_INTERVAL_S=<sec>` (0 disables).

### 2. Step-time + execution-time bookkeeping

`run_busy_loop` now updates two timestamps after each `_process_engine_step`
call so the heartbeat thread can report them. One change only.

### 3. DP all-reduce timing

In `_has_global_unfinished_reqs`, the `ParallelConfig.has_unfinished_dp`
all-reduce (fires every 32 steps) is wrapped in a timer. Slow ones are
logged:

```
[DP-ALLREDUCE] dp_rank=N step=S returned=B took=T.Ts local_unfinished=L
```

- Threshold default: 0.5 s. Below that it is silent.
- Override via `VLLM_DP_LOG_ALLREDUCE_THRESHOLD_S=<sec>`.
  - `0` — log every all-reduce (entry + return).
  - large — effectively disable.

An all-reduce that takes minutes is the EP-lockstep-straggler signature: one
rank is blocked in a step and all other ranks wait here.

## How to read the resulting logs during a stall

| Heartbeat state | Interpretation |
|---|---|
| No `[DP-HB]` lines at all on a rank | Python-level deadlock in that engine (GIL never released). Py-spy to narrow further. |
| `[DP-HB]` alive, `age_since_step` ticking upward | Busy loop blocked in a native call. Inspect the most recent log lines before the freeze for the last subsystem touched. |
| `[DP-HB]` alive, `step` incrementing, `steps_since_exec` growing | Loop healthy, scheduler cannot admit work. Likely prefix cache / KV-remote-wait starvation. Check `running/waiting`. |
| `[DP-ALLREDUCE]` with `took>=N*60s` | DP finish-sync all-reduce hung. One peer rank was stuck in a step; check which heartbeat's `age_since_step` matches. |

## Reverting

All three chunks are marked `# [DEBUG STALL]`. `git grep "DEBUG STALL"
vllm/v1/engine/core.py` finds them. Delete and rebuild.

## Update (round 2) — expanding `_maybe_log_stall_heartbeat`

First instrumented run confirmed the stall at 1P2D × 128 concurrency (a
passing config on a prior day; reproducibility is not concurrency-gated).
Heartbeat showed:

- Prefill all 4 ranks stepping, `age_since_step≈0.1s` (loop alive, native
  calls not wedged)
- `age_since_exec` climbing 200–400 s (no model execution for 3–6 min)
- `steps_since_exec` up to ~2700 dummy batches
- `running=0` everywhere, `waiting=11 on DP0, 9 on DP3`, `unfinished=11/9`
- `status={'WAITING': N}` (the original instrumentation only)
- Gap between `unfinished` and visible status: 2 requests per stuck rank
  were not in `waiting` or `running`

That 2-request gap is the stall's engine side. Scheduler code shows that
when `load_kv_async=True` (our Mooncake config), a request is transitioned
to `RequestStatus.WAITING_FOR_REMOTE_KVS` and **moved to
`self.skipped_waiting`** (`vllm/v1/core/sched/scheduler.py` ~L790). It's
promoted back only when `_try_promote_blocked_waiting_request` sees its ID
in `self.finished_recving_kv_req_ids` — which is populated from the
worker's `KVConnectorOutput.finished_recving` signal.

So the round-2 instrumentation adds to `_maybe_log_stall_heartbeat`:

- `skipped` queue length
- Per-status histogram now iterates `waiting + running + skipped_waiting`,
  exposing `WAITING_FOR_REMOTE_KVS` counts
- `finished_recv_kv` = `len(self.finished_recving_kv_req_ids)`
- `failed_recv_kv` = `len(self.failed_recving_kv_req_ids)`
- Per-connector attribute sweep (probes common names like
  `_reqs_need_recv`, `_async_load_tasks`, `_pending_loads`, …) — reports
  whichever ones exist with their length
- Up to 3 sample stuck request IDs + their status, for cross-referencing
  with NIXL / Mooncake logs

### Expected signatures to look for

| Pattern | Likely cause |
|---|---|
| `status={'WAITING_FOR_REMOTE_KVS': N}`, `finished_recv_kv=0`, `failed_recv_kv=0` | Connector never fires completion callback — Mooncake async load silently dead. |
| `WAITING_FOR_REMOTE_KVS=N`, `finished_recv_kv=N` | Promotion not happening: `_try_promote_blocked_waiting_request` logic bug. |
| `WAITING_FOR_REMOTE_KVS=N`, `failed_recv_kv=N` | Loads failed but error propagation stalled. |
| `MooncakeStoreConnector(_async_load_tasks=N)` growing without bound | Mooncake RPC queue backed up. |

## Update (round 3) — worker-side KV transfer thread instrumentation

Run 034112 confirmed:

- `status={'WAITING_FOR_REMOTE_KVS': 1}, finished_recv_kv=0, failed_recv_kv=0` fires
  repeatedly, and eventually one request times out to the router as HTTP 500
  "Connection reset by peer". So the async load completion signal for that
  request never reaches `scheduler.finished_recving_kv_req_ids`.
- Scheduler-side `connectors=[NixlConnector(-), MooncakeStoreConnector(-)]` shows
  no in-flight attributes — the completion-tracking state lives on the worker
  side, not the scheduler-side `MooncakeStoreConnector`.

### Where the signal actually lives

`MooncakeStoreConnector.start_load_kv` is a **no-op**. Async loads are issued
inside `MooncakeStoreWorker.get_finished()`:

1. `self.kv_recv_thread.add_request(request)` enqueues the load onto the
   background `KVCacheStoreRecvingThread`.
2. Later calls to `get_finished()` drain `self.kv_recv_thread.finished_requests`
   via `get_and_clear_finished_requests()` and return it as `done_recving`,
   which the scheduler uses to populate `finished_recving_kv_req_ids`.

The completion hand-off depends on `_handle_request` in the background thread
calling `self.set_finished_request(req_id)`.

### The candidate bug

`KVTransferThread._worker_loop` (`mooncake_store_worker.py:260`) wraps
`_handle_request` in a bare `except Exception` that only logs and loops:

```python
while True:
    try:
        request_data = self.request_queue.get()
        ...
        self._handle_request(request_data)
    except Exception as e:
        logger.error("Error in %s: %s", thread_name, e)
```

If any exception fires inside `_handle_request` *before* the call to
`self.set_finished_request(req_id)` (line 753), the request is dequeued,
logged as an error, and permanently dropped. The scheduler-side
`WAITING_FOR_REMOTE_KVS` entry then has no mechanism to recover — match the
observed symptom.

One concrete candidate: in `KVCacheStoreRecvingThread._handle_request`,
`self.tp_rank % len(key_list)` (line 655) would raise `ZeroDivisionError`
whenever `token_database.process_tokens(...)` yields zero keys (e.g. when
`vllm_cached_tokens == kvpool_cached_tokens` so there is nothing to load from
Mooncake). That `ZeroDivisionError` is caught in `_worker_loop` and the
request is dropped.

### What round-3 instrumentation adds in `mooncake_store_worker.py`

All marked `[DEBUG STALL]`.

1. Three counters on `KVTransferThread`:
   - `_dropped_request_ids: set[str]` — req_ids of requests whose dispatch
     threw.
   - `_total_handled`, `_total_dropped`.
2. Periodic `[KVTHREAD-HB]` log from inside `_worker_loop`, every ~5 s. Shows
   `qsize=`, `finished=`, `total_handled=`, `total_dropped=`,
   `dropped_sample=[...]`. Silent during healthy operation once the queue
   drains.
3. Enriched exception branch: before calling `_handle_request`, remember the
   `req_id` of the dequeued `ReqMeta`. If the handler throws, log as
   `[KVTHREAD-DROP] name=... req_id=<id> err=<exc>` with full traceback and
   add the req_id to `_dropped_request_ids`.

With these in place, a stalled run should surface the exact req_id being
dropped and the exact traceback (confirming or disproving the zero-keys
hypothesis above).

## Update (round 4) — fixing the truncation that faked a stall signal

Run 20260424_052821 (1P2D, conc 512, 20 multi-turn convos) showed
`stuck_sample=['chatcmpl-___prefill_addr_gb200-rack1-16::WAITING_FOR_REMOTE_KVS']`
appearing in the SAME form across many heartbeats. I initially read that as
"one request stuck for 60+ s". It isn't — that string is the PD composite
req_id (`chatcmpl-<uuid>___prefill_addr_<host>:<port>___decode_addr_<host>:<port>_<hash>`)
truncated to 40 chars, which cuts *before* the distinguishing UUID. Every
request parked under that prefill node renders identically after the slice,
so the heartbeat couldn't tell churn from persistence.

### Changes

`vllm/v1/engine/core.py` — `_maybe_log_stall_heartbeat`:

- Drop the `rid[:40]` slice. Log full request IDs.
- Maintain `self._stall_skipped_first_seen: dict[req_id, monotonic_ts]` and
  emit `dwell=<seconds>` per stuck entry. A truly stuck request shows
  `dwell` growing monotonically across heartbeats; churn shows many
  different req_ids each at low dwell.
- Cumulative `total_skipped_seen` counter — if this climbs by ~K per
  heartbeat, K requests are flowing through WAITING_FOR_REMOTE_KVS each
  window (not stuck).
- Raise sample cap from 3 → 10 and sort oldest-dwell-first so the worst
  offender is always at the top.

`vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_worker.py`
— `_worker_loop`:

- Change `self.request_queue.get()` to `get(timeout=1.0)` + `queue.Empty`
  continue. Heartbeat now fires on schedule whether the queue is busy or
  idle. Previously the loop could silently block inside `get()` forever,
  making "silence" ambiguous (idle vs wedged).
- Track `handling_req_id` and `handling_start` across the `_handle_request`
  call. Heartbeat reports `handling=<req_id>@<elapsed>s` when a request is
  currently in flight, so mid-native-call hangs name the victim.
- Emit `[KVTHREAD-END-SLOW] req_id=... elapsed_s=X.X` at WARNING when a
  single `_handle_request` takes >=5 s. Normal completions are kept at
  DEBUG to avoid flooding.

### How round-4 output distinguishes stall from churn

| Pattern | Interpretation |
|---|---|
| `stuck_sample` entries with growing `dwell` on the SAME `rid` across heartbeats | Actual stall of that request |
| `stuck_sample` entries with `dwell<heartbeat_period` and different `rid` each heartbeat, `total_skipped_seen` climbing fast | Churn through WAITING_FOR_REMOTE_KVS; not a stall |
| `[KVTHREAD-HB] handling=<rid>@N.Ns` with N growing across heartbeats | Specific request wedged inside a native Mooncake call |
| `[KVTHREAD-END-SLOW]` firing | Request completed but took too long — latency regression, not a hang |

## Round 5 — scheduler-side promote/allocate trace for candidate (e)

Round 4 revealed `[KV-RECV-APPLY]` firing for the stuck rid (so the
completion *was* delivered to the right scheduler) but the request
staying in `WAITING_FOR_REMOTE_KVS` for 13+ min with `finished_recv_kv=1`
pinned. No `[SCHED-ALLOC-BLOCK]` entries for the stuck rid either — so
allocate_slots is never reached for it. This round pinpoints WHICH of
three mechanisms is firing:

1. **Key mismatch** — `_try_promote_blocked_waiting_request` called,
   rid is NOT in `finished_recving_kv_req_ids` despite the RECV-APPLY add
   (impl bug around how the set is keyed).
2. **Promote OK but allocate fails** — already traced by
   `[SCHED-ALLOC-BLOCK]`.
3. **Peek never selects the rid** — the queue never returns this rid
   from `peek_request()`, so promote is never even attempted.

### Changes

`vllm/v1/core/sched/scheduler.py` — `_try_promote_blocked_waiting_request`:

- `[PROMOTE-TRY] rid=<...> in_finished_set=<bool> set_size=N ...` emitted
  on every call for a `WAITING_FOR_REMOTE_KVS` request. Rate-limited per
  rid to 1 line per 5 s to keep the happy path quiet; ALWAYS emits on
  `in_finished_set=True` so the promote→allocate sequence is never
  sampled out.
- `[PROMOTE-OK] rid=<...> new_status=<...> num_computed=X num_tokens=Y`
  on the return-True path. A `[PROMOTE-OK]` with no matching
  `[SCHED-ALLOC-BLOCK]` on the same rid means the request was scheduled
  into `running` and the stall is elsewhere.

### How round-5 output discriminates candidate (e)

| Pattern | Interpretation |
|---|---|
| No `[PROMOTE-TRY]` lines for the stuck rid | Peek never selects it — queue-state bug (e)-3 |
| `[PROMOTE-TRY] in_finished_set=False` repeatedly for a rid whose `[KV-RECV-APPLY]` fired earlier | Key mismatch (e)-1; compare rid strings byte-for-byte |
| `[PROMOTE-TRY] in_finished_set=True` → `[PROMOTE-OK]` → `[SCHED-ALLOC-BLOCK]` on same rid, repeated | Allocation pressure loop (e)-2 |
| `[PROMOTE-OK]` then silence (no ALLOC-BLOCK) but rid still stuck | Downstream of allocate — e.g., blocks weren't actually cached, or status reverted somewhere |
