# Root cause: 1P2D PD stall under high concurrency with MultiConnector

Observed in `pd_dep_mooncake_offload_nixl_400G` on GB200, 1 prefill node
× 2 decode nodes, concurrency ≥ 128, multi-turn sharegpt. Symptom:
throughput drops to 0/min, rate stays 0 for 20–30 minutes before the
router times out requests with HTTP 500 `Connection reset by peer`.

## TL;DR

Two requests, each holding ~50% of the prefill rank's KV cache, deadlock
the scheduler. One is `WAITING_FOR_REMOTE_KVS` and would complete almost
instantly if promoted + admitted. The other is `WAITING` at the queue
head and can't allocate blocks. The scheduler's outer `break` on
allocation failure (in [`Scheduler.schedule()`](vllm/v1/core/sched/scheduler.py#L348),
at [line 773](vllm/v1/core/sched/scheduler.py#L773) and
[line 827](vllm/v1/core/sched/scheduler.py#L827)) never lets it peek
past the head, so the blocked one never gets promoted and never
releases its blocks. **Head-of-line blocking in
[`self.skipped_waiting`](vllm/v1/core/sched/scheduler.py#L169)
iteration.**

**Fix: replace the two `break`s with a bounded `continue` so the loop
walks past an allocation-failed head to requests behind it that might
succeed.** See [Fix directions](#fix-directions) below for the full
rationale (a pre-pass promoter is useful defense-in-depth but
insufficient on its own).

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

Scheduler loop each iteration — [`Scheduler.schedule()` outer `while`](vllm/v1/core/sched/scheduler.py#L567):

1. `peek_request()` returns Owner A.
2. [`_is_blocked_waiting_status(WAITING)`](vllm/v1/core/sched/scheduler.py#L1622)
   → `False`, **skip the promote branch** (the
   [promote + move-to-skipped dispatch](vllm/v1/core/sched/scheduler.py#L578)).
3. Try
   [`reserve_full_isl`](vllm/v1/core/sched/scheduler.py#L729-L773)
   then
   [`allocate_slots`](vllm/v1/core/sched/scheduler.py#L779-L827)
   for Owner A → fail (need 26, have 13) → **`break`** exits the outer
   while loop at either
   [scheduler.py:773](vllm/v1/core/sched/scheduler.py#L773) or
   [scheduler.py:827](vllm/v1/core/sched/scheduler.py#L827).
4. Owner B never peeked. Its status-change to `WAITING` (which would
   happen via
   [`_try_promote_blocked_waiting_request`](vllm/v1/core/sched/scheduler.py#L2213)
   if called) never fires. Its blocks never release.

Repeat every `schedule()` call. Neither request moves. System stalls
until the router's HTTP timeout fires, aborts the requests, and
eventually frees the blocks by error path.

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

Owner B's ID was added to
[`finished_recving_kv_req_ids`](vllm/v1/core/sched/scheduler.py#L182)
at 17:31:33 (`[KV-RECV-APPLY]` logged inside
[`update_from_output`](vllm/v1/core/sched/scheduler.py#L2300);
`queued_finished_before=0`) and stayed there. The completion path works.
The promote path ([`_try_promote_blocked_waiting_request`](vllm/v1/core/sched/scheduler.py#L2213))
never runs on it again — no subsequent `[PROMOTE-TRY]` line for this
rid.

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

The bug is the outer `break` on allocation failure in
[`Scheduler.schedule()`](vllm/v1/core/sched/scheduler.py#L348) at
[scheduler.py:773](vllm/v1/core/sched/scheduler.py#L773) and
[scheduler.py:827](vllm/v1/core/sched/scheduler.py#L827).

**Necessary fix: option B. Option A is useful but insufficient alone.**

### Walk through why

With the stall state
`skipped_waiting = [A:WAITING, B:WAITING_FOR_REMOTE_KVS (in_fin=True)]`:

| Change | Main loop iter 1 | Iter 2 | Result |
|---|---|---|---|
| **A only (pre-pass promote)** | peek A → not blocked → allocate fails → `break` | — | **Still stuck.** B got pre-promoted to WAITING but never admitted; A at head still breaks the loop. Pre-promotion doesn't free blocks or change queue order. |
| **B only (`break`→`continue`)** | peek A → not blocked → allocate fails → **skip A, continue** | peek B → blocked → `_try_promote` succeeds → allocate 1 block (fits in 13 free) → admit to running | **Fixes it.** B runs its last 110 tokens, finishes prefill, releases 1057 blocks. A's next `schedule()` call allocates and proceeds. |
| **A + B** | same as B-only | same as B-only | **Fixes it.** Pre-pass is redundant here but provides defense in depth for cases where the blocked rid is never peeked for other reasons (LoRA cap, token budget, etc.). |

So option B is the load-bearing change. Option A is a correctness-preserving
cleanup, not a fix on its own.

### Option A — two-phase schedule loop (defense in depth)

Before the main admission loop at
[scheduler.py:567](vllm/v1/core/sched/scheduler.py#L567), run one pass
that promotes every request in
[`self.skipped_waiting`](vllm/v1/core/sched/scheduler.py#L169) whose ID
is in
[`finished_recving_kv_req_ids`](vllm/v1/core/sched/scheduler.py#L182).
This is a queue-state update, not a scheduling decision — consumes no
blocks.

```python
# In schedule(), before the main while loop at scheduler.py:567
for req in list(self.skipped_waiting):
    if (req.status == RequestStatus.WAITING_FOR_REMOTE_KVS
            and req.request_id in self.finished_recving_kv_req_ids):
        self._try_promote_blocked_waiting_request(req)
```

Reuses the existing
[`_try_promote_blocked_waiting_request`](vllm/v1/core/sched/scheduler.py#L2213)
untouched. **By itself it does not break the head-of-line deadlock**
(see table above), but pairs well with option B — it guarantees
completed requests become promotable regardless of whether the main
loop reaches them in the current `schedule()` call.

### Option B — replace `break` with bounded `continue` (required)

Walk past the head on allocation failure at
[scheduler.py:773](vllm/v1/core/sched/scheduler.py#L773) and
[scheduler.py:827](vllm/v1/core/sched/scheduler.py#L827). Requires a
per-`schedule()` visited-set to avoid infinite loop when every head
fails.

```python
# Near the top of the `if not preempted_reqs and ... UNPAUSED:` block
attempted_in_this_call: set[str] = set()

# At each break site, replace `break` with:
attempted_in_this_call.add(request_id)
request_queue.pop_request()
step_skipped_waiting.prepend_request(request)  # park at end of pass
continue

# And at the top of the loop body, after peek:
if request_id in attempted_in_this_call:
    # Already tried this call; park and move on.
    request_queue.pop_request()
    step_skipped_waiting.prepend_request(request)
    continue
```

Complexity: O(|queue|) per `schedule()` call instead of O(1) early
termination. With the existing `step_skipped_waiting` merge-back at
[scheduler.py:912](vllm/v1/core/sched/scheduler.py#L912), the loop is
guaranteed to terminate each call.

### Option C — separate queue for blocked-waiting requests (bigger refactor)

Don't mix `WAITING_FOR_REMOTE_KVS` requests into `skipped_waiting`
alongside preempted `WAITING` requests. Today they share the queue via
[`step_skipped_waiting.prepend_request`](vllm/v1/core/sched/scheduler.py#L587)
and
[`self.skipped_waiting.prepend_requests`](vllm/v1/core/sched/scheduler.py#L912).
Give blocked-waiting its own queue and promote from it separately —
then head-of-line blocking is structurally impossible because WAITING
requests never sit in front of blocked-waiting ones.
Largest surface-area change; most durable.

### Recommendation

**Ship option A + option B together.** A costs ~5 lines and defends
against edge cases that B alone misses; B is the actual fix for the
observed deadlock. C is the cleaner long-term answer if this class of
bug reappears in other queue-mixing paths.

## Applied fix

Landed in `vllm/v1/core/sched/scheduler.py`, marked
`# [BUG FIX #headofline]`:

1. [**Pre-pass promoter**](vllm/v1/core/sched/scheduler.py#L567) —
   before the admission loop, walk `self.skipped_waiting` and call
   `_try_promote_blocked_waiting_request` for every req whose ID is in
   `finished_recving_kv_req_ids`. Pure status bookkeeping; consumes no
   blocks.
2. [**`reserve_full_isl` break→park**](vllm/v1/core/sched/scheduler.py#L785)
   — replace `break` with `request_queue.pop_request();
   step_skipped_waiting.prepend_request(request); continue`. Same
   pattern as the existing blocked-waiting handler at
   [scheduler.py:587](vllm/v1/core/sched/scheduler.py#L587).
3. [**`allocate_slots` break→park**](vllm/v1/core/sched/scheduler.py#L849)
   — same replacement at the second failure site.

Encoder cleanup (`encoder_cache_manager.free(request)`) is preserved
before the park, matching the original `break` path. No infinite loop:
each failed request is popped from the active `request_queue` into
`step_skipped_waiting`, which merges back to `self.skipped_waiting`
only at the end of `schedule()` via
[`prepend_requests`](vllm/v1/core/sched/scheduler.py#L912), so the
within-call queue is strictly monotonic.

## What fixed this without a code change

- **Remove Mooncake** (pure NIXL). Works because the collision window
  closes. Costs 6× E2EL.
- **2P2D instead of 1P2D.** Works because the pool doubles.
- **Concurrency ≤ 128.** Reduces the chance of two concurrent giants.
  Actually still hit the bug in one run at conc 128 — timing-dependent.

All three are workarounds. The real fix is in the scheduler.

## Code reference index

### Scheduler — the deadlock site

- [`Scheduler.schedule()`](vllm/v1/core/sched/scheduler.py#L348) — entry
- [Outer admission `while`](vllm/v1/core/sched/scheduler.py#L567) — the loop that stalls
- [Blocked-status dispatch](vllm/v1/core/sched/scheduler.py#L578) — promote-or-skip for `WAITING_FOR_REMOTE_KVS`
- [`reserve_full_isl` branch](vllm/v1/core/sched/scheduler.py#L729-L773) — fails with too-few-blocks; `break` at [L773](vllm/v1/core/sched/scheduler.py#L773)
- [`allocate_slots` branch](vllm/v1/core/sched/scheduler.py#L779-L827) — same failure mode; `break` at [L827](vllm/v1/core/sched/scheduler.py#L827)
- [`step_skipped_waiting.prepend_request`](vllm/v1/core/sched/scheduler.py#L587) — how blocked/preempted reqs end up at the head
- [`self.skipped_waiting.prepend_requests`](vllm/v1/core/sched/scheduler.py#L912) — merge-back at end of schedule()
- [`_select_waiting_queue_for_scheduling`](vllm/v1/core/sched/scheduler.py#L1707) — which queue `peek` reads from
- [`_is_blocked_waiting_status`](vllm/v1/core/sched/scheduler.py#L1622) — returns False for plain `WAITING`, skipping promote
- [`_try_promote_blocked_waiting_request`](vllm/v1/core/sched/scheduler.py#L2213) — the promotion never called on Owner B
- [`finished_recving_kv_req_ids`](vllm/v1/core/sched/scheduler.py#L182) and [`failed_recving_kv_req_ids`](vllm/v1/core/sched/scheduler.py#L183) — the sets driving promotion
- [`[KV-RECV-APPLY]` log](vllm/v1/core/sched/scheduler.py#L2300) — inside `update_from_output`, populates `finished_recving_kv_req_ids`

### Scheduler — diagnostic logs added in rounds 5–6

- [`[SCHED-ALLOC-BLOCK]` reserve_full_isl](vllm/v1/core/sched/scheduler.py#L743)
- [`[SCHED-ALLOC-BLOCK]` allocate_slots](vllm/v1/core/sched/scheduler.py#L789)
- [`[PROMOTE-TRY]` / `[PROMOTE-OK]`](vllm/v1/core/sched/scheduler.py#L2213)
- [`_log_block_owners`](vllm/v1/core/sched/scheduler.py#L1634) → emits `[ALLOC-OWNERS]`

### Engine core — scheduler-state visibility

- [`_maybe_log_stall_heartbeat`](vllm/v1/engine/core.py#L1200) → emits `[STALL-HEARTBEAT]` at [L1316](vllm/v1/engine/core.py#L1316)
- [DP heartbeat thread `[DP-HB]`](vllm/v1/engine/core.py#L1772)
- [`DPEngineCoreProc.run_busy_loop`](vllm/v1/engine/core.py#L1864) — where step_counter / exec timestamps are maintained

### MooncakeStoreConnector — proven healthy in this stall

- [`MooncakeStoreConnector`](vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_connector.py#L86)
- [`start_load_kv` is a no-op](vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_connector.py#L197) — loads issued in `get_finished()` instead
- [`MooncakeStoreWorker.get_finished`](vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_worker.py#L1110) — enqueues loads, drains finished set
- [`[KV-RECV-DONE]` log](vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_worker.py#L1172) — every non-empty `done_recving`
- [`KVTransferThread._worker_loop`](vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_worker.py#L264) — the background loop; `[KVTHREAD-HB]` at [L291](vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_worker.py#L291)
- [`KVCacheStoreRecvingThread._handle_request`](vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_worker.py#L719) — performs the actual Mooncake `batch_get`

### Companion doc

- [`DEBUG_STALL.md`](DEBUG_STALL.md) — chronological rounds 1–6, with the per-round hypothesis/verdict matrix.
