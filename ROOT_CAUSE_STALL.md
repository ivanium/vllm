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
          status=WAITING  (promoted from WAITING_FOR_REMOTE_KVS;
                           never reached self.running)
          owns 1072 blocks (allocated for Mooncake prefix-cache load)
          nc=68608, nt=69521   → full-ISL estimate needs ~26 more blocks
                                  to cover the rest of the prompt

          Owner B   rid=…94a12dc16bc…
          status=WAITING_FOR_REMOTE_KVS  (Mooncake load done)
          in_finished_recving_kv_req_ids = True
          owns 1057 blocks (allocated for Mooncake prefix-cache load)
          nc=67648, nt=67758   → would need ~1 more block if promoted
```

Both requests got their blocks and `num_computed_tokens` from Mooncake
prefix-cache loads, not from any forward pass. `num_computed_tokens` is
set by
[`_update_waiting_for_remote_kv`](vllm/v1/core/sched/scheduler.py#L2211)
at promotion time based on how many tokens the connector reported as
loaded. Neither request has ever been in `self.running`; after promotion
Owner A's very first admission check fails on
[`reserve_full_isl`](vllm/v1/core/sched/scheduler.py#L729-L773) because
its full-sequence estimate doesn't fit in the 13 free blocks, and it's
been stuck there ever since.

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

## Limit case: what if the pool were fully saturated?

The observed trace has `free_blocks=13` — the traversal fix works
because B only needs 1 more block to succeed and the 13 free slots
cover it. If the two giants together owned the *entire* pool
(`free=0`), no scheduler traversal fix resolves the stall, because
there is no block for B to allocate even after promotion:

1. Option B (`continue` past A): loop walks past A, peeks B, promotes
   B to `WAITING`, tries to allocate 1 block → **fails (free=0)** →
   continues past B. Queue exhausted. Same state next step.
2. Option A (pre-promote pass): same — B is flipped to `WAITING` but
   still can't allocate.
3. Option C (separate queue): same — B is reachable but still can't
   allocate.

This is a different class of failure: a **resource deadlock**, not a
control-flow deadlock. Two requests each own a non-releasable fraction
of a shared pool, and each needs a block the other holds. On the
prefill rank neither is preemptible — A has finished its Mooncake load
(no safe point to abort), B's blocks are the active target of an
in-flight NIXL write (freeing them mid-transfer is memory corruption).

The robust fix for this class is **admission-time block reservation**,
not traversal: refuse to call `_try_promote_blocked_waiting_request`
(or refuse to start a new prefix-cache load) if the summed full-ISL
reservation across currently-admitted requests would exceed the pool
minus a safety margin. Today `reserve_full_isl` checks the current
single request against current free blocks — it has no notion of
aggregate commitment. Two requests can each pass independently at
their own admission moment, each leaving just enough room for itself,
with no headroom for either to top up later.

This is out of scope for the traversal fix above but is the real
robustness story. The traversal fix handles the common case (some
headroom exists); reservation handles the adversarial case (pool
fully committed).

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

Mooncake is *not* the cause — the completion signal flows correctly end
to end. But this deadlock is also not a generic "high-latency connector"
bug. It specifically needs a connector that plays the **prefill-side
async prefix-loader role**: one that returns `load_kv_async=True` from
`get_num_new_matched_tokens()`, causing the scheduler to allocate
blocks, set `status=WAITING_FOR_REMOTE_KVS`, and park the request in
`skipped_waiting` with blocks already held. `MultiConnector` with
Mooncake (or LMCache, or any future L1/L2 prefix cache) fills that
role.

### Pure NIXL does not hit this exact wedge

On the **prefill rank**, pure NIXL is a sender — it doesn't do prefix
cache loads. Its
[`get_num_new_matched_tokens`](vllm/distributed/kv_transfer/kv_connector/v1/nixl_connector.py)
returns 0 external tokens, `load_kv_async=False`, and requests never
enter `WAITING_FOR_REMOTE_KVS`. No Owner-B-shape exists, so this wedge
cannot form.

On the **decode rank**, pure NIXL does use `WAITING_FOR_REMOTE_KVS`
(awaiting the prefill rank's push). The same head-of-line traversal bug
is structurally reachable there. But the self-healing dynamic that
makes this a *wedge* rather than slow-down breaks down on decode:

- **Promoting B doesn't free blocks on decode.** Decode holds blocks for
  its whole lifetime and only releases at completion. Promote-and-run
  converts one pool-holder into another. It does not unblock A.
- **Decode has a preemption victim pool.** The running loop at
  [scheduler.py:385](vllm/v1/core/sched/scheduler.py#L385) preempts
  running requests on allocation failure via
  [`_preempt_request`](vllm/v1/core/sched/scheduler.py#L1019), which
  frees blocks and re-queues the victim in `self.waiting` with
  `num_computed_tokens=0`. Under decode-side concurrency,
  `self.running` is always large → preemption can always find a victim
  → pool always has a release valve. On the prefill rank in this trace,
  `running=0` (every candidate stalled in `skipped_waiting` before ever
  being admitted), so the valve has no target.

Net: the wedge specifically requires (a) a prefill-side prefix-loader
connector, (b) two requests whose full-ISL estimates together exceed
the pool, (c) `running≈0` because post-promote admission keeps failing.
Pure NIXL fails to meet (a) on the prefill rank and (c) on the decode
rank. Removing Mooncake "fixes" the stall by removing condition (a);
2P2D fixes (b); lower concurrency reduces the probability of (b).

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

### Option F2 — lateral preemption for blocked-waiting holders (principled alternative)

Options A/B/C are traversal fixes — they restructure how the queue is
iterated. F2 fixes the same wedge at a higher level of abstraction:
**treat block-holding requests without forward-pass progress as
preemptible the same way `self.running` is already preemptible.**

The observation: Owner B in the trace holds 1057 blocks, its Mooncake
load has completed, and it has generated zero tokens. A `self.running`
request mid-decode that needs more blocks can already be preempted by
[`_preempt_request`](vllm/v1/core/sched/scheduler.py#L1019) (frees
blocks, resets `num_computed_tokens=0`, re-queues to `self.waiting`).
Owner B has *less* progress than that running request (pure KV load,
no forward pass), so if running is preemptible, B should be too. The
existing valve at [scheduler.py:385](vllm/v1/core/sched/scheduler.py#L385)
just can't see into `skipped_waiting`.

#### Candidate set

```python
# Admission target = the request this iteration is trying to schedule
target = request  # peeked from active queue at top of outer while

# Lateral preemption pool — only requests at the same "tier" as target
# (block-holding, KV loaded, no forward pass yet). RUNNING is
# deliberately excluded: running has strictly more sunk work (prefix +
# generated tokens), and robbing it to admit a zero-compute request is
# priority inversion.
candidates = [
    r for r in self.skipped_waiting
    if r is not target and (
        (r.status == RequestStatus.WAITING_FOR_REMOTE_KVS
         and r.request_id in self.finished_recving_kv_req_ids)
        or (r.status == RequestStatus.WAITING
            and r.num_computed_tokens > 0)   # promoted-from-remote-kv signature
    )
]

# Least-progressed first; ties broken by block count (bigger victim = more relief)
victim = pick_least_progressed(candidates)
```

`WAITING + nc>0` identifies a request that was promoted out of
`WAITING_FOR_REMOTE_KVS` and still holds its prefix blocks.
`_preempt_request` resets `nc=0` on the way back, and a freshly-admitted
`WAITING` request has `nc=0` too, so the predicate is a reliable
signature for "promoted, not yet run."

#### Trace timeline under F2

State at the moment of wedge (from §Concrete trace):
`free=13`, `skipped_waiting = [A(WAITING, 1072 blk, needs 26),
B(WAITING_FOR_REMOTE_KVS recv-done, 1057 blk)]`, `running=0`.

1. Outer `while` peeks A → `allocate_slots(26)` fails (13 free < 26 needed).
2. Today: `break`. Under F2: consult lateral preemption pool.
3. Candidates = `{B}` (A excluded as target; no other skipped_waiting entries).
4. Victim = B. `_preempt_request(B)`: free 1057 blocks, `B.nc=0`,
   `B.status=WAITING_FOR_REMOTE_KVS` re-queued, discard from
   `finished_recving_kv_req_ids` (will re-fetch on next admission).
   `free` jumps 13 → 1070.
5. Retry `allocate_slots(A, 26)` → succeeds. A joins `self.running`.
6. A prefills, hands off to decode via NIXL, releases 1072 + 26.
7. B's next admission triggers a fresh Mooncake load (~500ms redundant
   work; no wedge).

#### Why F2 is sufficient

The wedge *definitionally* requires a recv-done (or promoted-not-run)
request behind the failing head — that's what makes it a wedge rather
than normal backpressure. An in-flight-only scenario isn't a wedge: A
waits for some transfer to finish naturally, at which point F2's
candidate set becomes non-empty on the next tick. F2 therefore covers
every wedge state without needing to cancel in-flight transfers or
touch running.

#### Why RUNNING is excluded

A running request has `prefix + generated_tokens` worth of sunk work; A
(or any F2 target) has `prefix` only. Preempting running to admit A
throws away compute to benefit a less-progressed request — priority
inversion. vLLM's running loop at
[scheduler.py:385](vllm/v1/core/sched/scheduler.py#L385) already does
**within-tier** RUNNING-to-RUNNING preemption on its own allocation
pressure; F2 doesn't extend or interfere with that path. Two clean
tiers:

| Loop | Victim pool | Beneficiary |
|---|---|---|
| Running loop (existing) | RUNNING | RUNNING needing more blocks |
| Admission loop (F2) | block-holding non-running (recv-done + promoted-not-run) | WAITING-ish admission target |

#### Limitations

- **In-flight transfers are not preempted.** If every block-holder
  behind A is still receiving, F2 does nothing this tick; A waits for
  natural completion (~500ms for Mooncake). This is backpressure, not
  wedge — correctness preserved, latency bump only.
- **Redundant load on victim re-admission.** B will re-fetch its
  prefix from Mooncake on next admission. For <500ms transfers this
  is acceptable; for multi-second transfers F2's advantage over A+B
  thins out.
- **Fairness/starvation.** A victim that keeps getting preempted can
  starve. Standard mitigations apply (FCFS ordering, age-based
  preemption bias, priority hints); not unique to F2.

### Recommendation

**Short term: ship option A + option B together.** A costs ~5 lines
and defends against edge cases that B alone misses; B is the actual
fix for the observed deadlock. This is the minimum viable patch.

**Medium term: replace A + B with option F2.** F2 encodes the right
invariant — "a request holding blocks without forward-pass progress is
preemptible the same as a running request" — instead of patching
queue traversal. It composes cleanly with the existing running-loop
preemption valve, handles the `free=0` saturation case that A+B
cannot (A+B walks past the head but still fails to allocate; F2 frees
blocks directly), and removes head-of-line as a structural concern
rather than a traversal concern.

**Option C is orthogonal.** Splitting the queue is a separate
refactor motivated by code clarity; it doesn't subsume F2 and F2
doesn't need it.

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
