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
