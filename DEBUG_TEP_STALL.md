# DEBUG_TEP_STALL — KV-retention stall on prefill (TEP+DEP)

Companion to [`ROOT_CAUSE_STALL.md`](ROOT_CAUSE_STALL.md). The earlier doc
covers a **control-flow** deadlock on the prefill scheduler
(head-of-line blocking in `skipped_waiting`, fixed via lateral-preempt +
`has_executed`). This doc covers a **different bug**: with heterogeneous
TP (prefill TP=4, decode DP=8/TP=1) and MLA, the connector's broadcast
notification path uses the wrong request id, so 3 of 4 prefill ranks
never receive a "done-reading" notif and only release their pinned KV
blocks via the 300s `VLLM_NIXL_ABORT_REQUEST_TIMEOUT` drain.

The two stalls share symptoms ("rate=0/min, requests stuck") but the
internal state is distinct, so the fixes are different too.

## Recipe under test

`recipes/crusoe/kimik25/low_latency/dao/pd_tep_dep_mooncake_offload_nixl_400G.yaml`,
GB200 NVL72, Kimi-K2.5-NVFP4:

- 1 prefill node, 4 GPUs, `-tp 4 -ep`, MultiConnector
  (NixlConnector + MooncakeStoreConnector{`load_async: true`})
- 1 decode replica, 2 nodes × 4 GPUs, `-dp {dp_size} -ep`, NixlConnector only
- `block_size: 64`, `max_model_len: 118400`, `gpu-memory-utilization: 0.85`,
  `--enforce-eager` on prefill (intentional), `--kv-cache-dtype fp8`
- Bench: `vllm-bench` synthetic_sharegpt_v3, 20 conversations, multi-turn,
  `--max-concurrency 256`

Env knobs that matter for this stall:

- `VLLM_NIXL_ABORT_REQUEST_TIMEOUT=300`
- `VLLM_NIXL_DEBUG_INTERVAL_S=30`
- `VLLM_NIXL_NOTIF_LOG=1`

## Root cause — wrong request id in MLA broadcast notif

`_read_blocks_for_req` in `nixl_connector.py` does the right thing for
the rank decode actually pulls from: it calls `_read_blocks(...)` with
`remote_request_id=meta.remote.request_id`, and the NIXL transfer is
prepped with an auto-notif tagged `f"{remote_request_id}:{world_size}"`.
That tag matches the key prefill stores in `_reqs_to_send`, so the rank
that served the read drains correctly.

For MLA with `tp_ratio < 0` (KV is replicated across prefill TP ranks,
so we only read from one of them), the connector also has to *manually*
notify the other prefill ranks so they can release their copy of the
blocks. That broadcast was sending the **wrong** id:

```python
# BEFORE (buggy)
notif_id = f"{req_id}:{self.world_size}".encode()
```

`req_id` here is the **decode-side** request id. But every prefill rank
keys `_reqs_to_send` by its own (prefill-side) request id —
`meta.remote.request_id`. The format `"<uuid32>-<suffix>"` shows the
mismatch directly: prefill UUID `…6819f-bbf939e4` vs. decode UUID
`…6819f-be0f86d8`. The notif arrives, `_get_new_notifs` parses out the
decode id, fails to find it in `_reqs_to_send` / `_reqs_to_process`, and
logs "Potentially invalid KV blocks for unrecognized request" — then
drops the notif.

So for every request:
- 1 prefill rank (the one decode reads from) gets a valid auto-notif and
  drains immediately.
- The other 3 prefill ranks get a bogus broadcast notif that gets
  rejected; their `_reqs_to_send` entries sit until the 300s timer expires.

The fix is one line — use the prefill-side id in the broadcast, matching
what `_read_blocks` already does:

```python
# AFTER
notif_id = f"{meta.remote.request_id}:{self.world_size}".encode()
```

## Evidence — run `20260425_<post-instrumentation>`

Instrumentation added to `nixl_connector.py` for this run:

- Per-request dwell timestamps on prefill (`_reqs_to_send_added_at`)
  and decode (`_recving_added_at`).
- Cumulative counters: `dbg_send_added_total`,
  `dbg_send_completed_via_notif_total`,
  `dbg_send_completed_via_timeout_total`,
  `dbg_recv_added_total`, `dbg_recv_completed_total`,
  `dbg_recv_failed_total`.
- `inspect_state()` exposes oldest / p50 / p90 dwell + counters; the
  periodic `[NIXL-DEBUG]` reporter therefore prints rate-deltas directly.

Per-rank counter snapshot at end of run (from prefill engine log):

| rank | send_added | send_completed_via_notif | send_completed_via_timeout |
|------|-----------:|-------------------------:|---------------------------:|
| TP0  |        511 |                      511 |                          0 |
| TP1  |        587 |                        0 |                        577 |
| TP2  |        587 |                        0 |                        577 |
| TP3  |        587 |                        0 |                        577 |

Decode side: 587 reads posted, all to prefill `remote_rank=0`. So TP0
sees the auto-notif from `_read_blocks` (correct id, drains via notif).
TP1–TP3 see only the broadcast notif from `_read_blocks_for_req`
(wrong id, rejected → drained via 300s timeout).

Each of TP1–TP3 logged exactly 587 "Potentially invalid KV blocks for
unrecognized request" warnings — one per read decode posted.

Bench impact: TTFT mean=340s, p99=597s (≈ the 300s abort timeout
dominates, plus queueing). TPOT (~14ms) is healthy *when* a request
actually makes it through. Output throughput collapses to ~16 tok/s,
limited by the rate at which prefill blocks free up (timeout-bound).

## Why this is *not* the head-of-line deadlock

The head-of-line deadlock requires a `WAITING_FOR_REMOTE_KVS` victim in
`running` that the scheduler keeps skipping over. Here `running=0` and
free blocks are insufficient to admit any waiter — the scheduler isn't
deadlocked on choice, it has nothing it can do until a block-holder
releases. The block-holders are completed prefills sitting in
`_reqs_to_send`, and they release on either:

- A NIXL `done-reading` notif from decode (the success path), or
- The 300s `VLLM_NIXL_ABORT_REQUEST_TIMEOUT` expiration drain.

With the bug above, 3-out-of-4 prefill ranks always take the timeout
path. With the fix, all 4 ranks take the notif path.

## Fix shipped

- `vllm/distributed/kv_transfer/kv_connector/v1/nixl_connector.py`,
  `_read_blocks_for_req`: replace `req_id` with
  `meta.remote.request_id` when constructing the MLA broadcast
  `notif_id`. This matches the format used by `_read_blocks` for the
  auto-notif on the rank that actually served the read.
- One-line change; no behavior change on non-MLA / non-`tp_ratio<0`
  paths (the broadcast block is gated on `self.use_mla and tp_ratio < 0`).

## Instrumentation kept in tree

The dwell timestamps and cumulative counters added during diagnosis are
left in place. They're cheap (a few dict updates per request) and gated
on `VLLM_NIXL_NOTIF_LOG=1` for the per-request dwell log lines. The
counters in `inspect_state()` cost nothing extra and are useful for
spotting future imbalances of the same shape (one rank healthy, others
draining via timeout).

No recipe changes — the recipe is correct.

## Verification

Re-run the same recipe with `NUM_PROMPTS=20`, `MAX_CONCURRENCY=256`,
`PREFILL_COUNT=1`, `DECODE_COUNT=1`. Expected signals:

- Prefill TP1–TP3 `dbg_send_completed_via_notif_total` ≈ TP0's value
  (no longer 0).
- Prefill TP1–TP3 `dbg_send_completed_via_timeout_total` ≈ 0.
- "Potentially invalid KV blocks for unrecognized request" warnings
  drop to 0 (or to noise from genuinely-stale post-abort notifs).
- TTFT drops from ~340s back to single-digit seconds; output throughput
  recovers.
