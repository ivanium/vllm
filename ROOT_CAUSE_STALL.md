# Root cause: 1P2D PD stall under high concurrency with MultiConnector

Observed in `pd_dep_mooncake_offload_nixl_400G` on GB200, 1 prefill node
× 2 decode nodes, concurrency ≥ 128, multi-turn sharegpt. Symptom:
throughput drops to 0/min, rate stays 0 for 20–30 minutes before the
router times out requests with HTTP 500 `Connection reset by peer`.

## TL;DR

Two requests, each holding ~50% of the prefill rank's KV cache, deadlock
the scheduler. One is `WAITING_FOR_REMOTE_KVS` and would complete almost
instantly if promoted. The other is `WAITING` at the queue head and
can't allocate blocks. The scheduler's outer `break` on allocation
failure never lets it peek past the head, so the blocked one never gets
promoted and never releases its blocks. **Head-of-line blocking in
`self.skipped_waiting` iteration.**

## The deadlock

Two requests both sit in `self.skipped_waiting` at the moment of freeze.
Both own ~1000 blocks. The pool has ~2150 blocks total. Roughly 13 free.

```
skipped_waiting queue (head → tail):

  [head]  Owner A   rid=…880bc4db985…
          status=WAITING  (promoted; mid chunked-prefill)
          owns 1072 blocks
          nc=68608, nt=69521   → needs ~26 more blocks to finish

          Owner B   rid=…94a12dc16bc…
          status=WAITING_FOR_REMOTE_KVS  (Mooncake load done)
          in_finished_recving_kv_req_ids = True
          owns 1057 blocks
          nc=67648, nt=67758   → would need ~1 more block if promoted
```

Scheduler loop each iteration:

1. `peek_request()` returns Owner A.
2. `_is_blocked_waiting_status(WAITING)` → `False`, **skip promote branch**.
3. Try `reserve_full_isl` / `allocate_slots` for Owner A → fail (need 26,
   have 13) → **`break`** exits the outer while loop.
4. Owner B never peeked. Its status-change to `WAITING` (which would
   happen via `_try_promote_blocked_waiting_request` if called) never
   fires. Its blocks never release.

Repeat every schedule() call. Neither request moves. System stalls until
the router's HTTP timeout fires, aborts the requests, and eventually
frees the blocks by error path.

## Why is this a deadlock and not just a slow scheduler

Owner B would unblock itself and Owner A if promoted:

- Promote `B` → status=WAITING.
- Allocate ~1 extra block for its final 110 tokens (fits in 13 free).
- Finish prefill → hand off KV to decode via NIXL → blocks released.
- 1057 blocks return to the free pool.
- Owner A can now allocate its 26 → finish prefill → release 1072.
- Rank recovers fully.

The only thing standing in the way is that `break` in step 3 above.

## Concrete trace (run 20260424_172233, prefill DP1)

```
17:31:33  [PROMOTE-TRY]   rid=…94a12dc16bc…  in_finished_set=False  set_size=0
17:31:33  [KV-RECV-APPLY] rid=…94a12dc16bc…  queued_finished_before=0
17:31:33  [PROMOTE-TRY]   rid=…880bc4db985…  in_finished_set=False  set_size=0
17:31:33  [KV-RECV-APPLY] rid=…880bc4db985…  queued_finished_before=1
17:31:34  [PROMOTE-TRY]   rid=…880bc4db985…  in_finished_set=True   set_size=2
17:31:34  [PROMOTE-OK]    rid=…880bc4db985…  new_status=WAITING
--- 94a12dc16bc never appears in PROMOTE-TRY again ---
17:35:53  [SCHED-ALLOC-BLOCK] branch=reserve_full_isl
              req_id=…880bc4db985…  status=WAITING
              needed_blocks=35  free_blocks=13
              waiting=19  skipped=2  running=0
17:35:53  [ALLOC-OWNERS]  free_blocks=13  total_owners=2  total_owned_blocks=2129
              top=[{n=1072, status=WAITING, …880bc4db985…},
                   {n=1057, status=WAITING_FOR_REMOTE_KVS, in_fin=True,
                    …94a12dc16bc…}]
--- repeats every 5 s for the remaining 14+ min of the run ---
```

Owner B's ID was added to `finished_recving_kv_req_ids` at 17:31:33
(`[KV-RECV-APPLY] queued_finished_before=0`) and stayed there. The
completion path works. The promote path never runs on it again.

## Why it's concentrated on 1P2D high-concurrency

The deadlock needs two concurrent requests that each want ~half the pool.
Conditions that make that likely:

- **Tight KV headroom on prefill ranks.** GB200 NVL72 topology gives
  local-rank-0/1/3 ~1.02× admission concurrency at `max_model_len=118400`
  (`Maximum concurrency for 118400 tokens per request: 1.02x` in vLLM's
  own init log). Any two requests near `max_model_len` fill the pool.
- **Multi-turn sharegpt.** Prompts grow turn by turn, so by the time the
  conversation reaches turn 20+, prompts approach `max_model_len` and
  every request is "big".
- **Concurrency ≥ 128.** Enough in-flight requests that two of them end
  up both at the front of `skipped_waiting` at the same time.

At 2P2D the pool has twice the admission capacity and the two-big-requests
collision is much rarer, which is why `20260423_224115` (2P2D × 256)
passed while `20260423_230640` (1P2D × 256) stalled.

Mooncake is *not* the cause. The completion signal flows correctly end
to end. Removing Mooncake only helps because it reduces per-request
latency, making the two-concurrent-giants collision less likely. The
deadlock is a vLLM scheduler property that also bites pure NIXL when the
timing aligns — it's just much rarer there.

## Fix directions

The bug is the outer `break` on `allocate_slots` failure in
`vllm/v1/core/sched/scheduler.py`. Three options:

### 1. Two-phase schedule loop (preferred)

Before the main admission loop, run one pass that promotes every
request in `self.skipped_waiting` whose ID is in
`finished_recving_kv_req_ids`. This is a queue-state update, not a
scheduling decision — it consumes no blocks. After that pass, the main
loop's `break` behavior is safe because every ready-to-run request has
the chance to contribute its blocks back to the pool on completion.

```python
# In schedule(), before the main while loop:
for req in list(self.skipped_waiting):
    if (req.status == RequestStatus.WAITING_FOR_REMOTE_KVS
            and req.request_id in self.finished_recving_kv_req_ids):
        self._try_promote_blocked_waiting_request(req)
```

This is minimally invasive and matches the semantic intent.

### 2. Replace `break` with `continue`

Walk past the head on allocation failure. Needs a per-iteration
visited-set to avoid retrying the same rid in the same schedule() call.
More correct in general but larger behavior change.

### 3. Separate queue for blocked-waiting requests

Don't mix `WAITING_FOR_REMOTE_KVS` requests into `skipped_waiting`
alongside `WAITING` requests. Give blocked-waiting its own queue and
promote from it separately. Biggest refactor.

## What fixed this without a code change

- **Remove Mooncake** (pure NIXL). Works because the collision window
  closes. Costs 6× E2EL.
- **2P2D instead of 1P2D.** Works because the pool doubles.
- **Concurrency ≤ 128.** Reduces the chance of two concurrent giants.
  Actually still hit the bug in one run at conc 128 — timing-dependent.

All three are workarounds. The real fix is in the scheduler.
