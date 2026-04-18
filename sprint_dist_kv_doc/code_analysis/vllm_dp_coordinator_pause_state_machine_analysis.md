# DPCoordinator Pause-Start-Wait State Machine Analysis for DP>1 in vLLM

Based on current workspace code:

- `<repo-root>`
- Focus on analysis `vllm/v1/engine/*`, `vllm/v1/core/sched/*`

---

## 1. Let’s talk about the conclusion first

When vLLM currently has `DP > 1`, there are actually two sets of control mechanisms that are orthogonal to each other but superimpose each other:

1. **Global DP wave state machine**
   - Responsible for maintaining a global `running / paused` state between multiple DP ranks.
   - The core objects are `DPCoordinatorProc` and `DPEngineCoreProc`.
   - This mechanism only truly enables wave coordination when **MoE + DP>1**.

2. **Local scheduler pause state machine**
   - Responsible for `UNPAUSED / PAUSED_NEW / PAUSED_ALL` inside a single engine process.
   - The core objects are `Scheduler.pause_state`, `EngineCoreProc.pause_scheduler()`, `resume_scheduler()`.
   - `pause_generation(mode="abort"|"wait"|"keep")` uses this mechanism.

These two sets of state machines are not the same thing:

- `START_DP_WAVE` / `wave_complete` is the **global DP running state**.
- `pause_scheduler` / `resume_scheduler` is **local scheduling state**.

When actually running, both sides must be allowed before the real request can proceed.

---

## 2. When did DPCoordinator exist?

Whether `DPCoordinator` is required is determined by `VllmConfig.needs_dp_coordinator`:

- `vllm/config/vllm.py:462-482`
- `vllm/v1/engine/utils.py:968-980`

The code semantics are:

1. **MoE + DP>1**
   - Even external LB requires a coordinator.
   - The reason is that wave coordination relies on coordinator.

2. **Non-MoE + DP>1**
   - Coordinator is only required for internal/hybrid LB.
   - At this time, the coordinator mainly does stats summary and does not do wave coordination.

In addition, `DPEngineCoreProc` is only used by **MoE**:

- `vllm/v1/engine/core.py:1071-1082`

Non-MoE DP ranks are treated as "mutually independent DP=1 engines" in v1, without the global wave logic of `DPEngineCoreProc`.

---

## 3. ASCII architecture diagram

This section only addresses one thing:

- Draw the relationship between "front-end process/coordinator process/multiple DP engine processes" clearly first

### 3.1 internal / hybrid LB, and MoE + DP>1

This is currently the **most important to understand** form because:

- The front end will do internal LB
- coordinator manages both stats and wave coordination
- engine uses `DPEngineCoreProc````text
                                shared frontend / api server process

    +-----------------------------------------------------------------------------------+
    | AsyncLLM / DPLBAsyncMPClient                                                      |
    |                                                                                   |
    |  local cached state:                                                              |
    |    - current_wave                                                                 |
    |    - engines_running                                                              |
    |    - lb_engines[dp_rank] = [waiting, running]                                     |
    |                                                                                   |
    |  outbound:                                                                        |
    |    - ROUTER -------------- ADD / ABORT / UTILITY --------------------------+      |
    |    - PAIR ---------------- FIRST_REQ / SCALE_ELASTIC_EP ----------------+  |      |
    |                                                                         |  |      |
    |  inbound:                                                                         |
    |    - XSUB <------------- (counts, current_wave, engines_running) -------+  |      |
    +-----------------------------------------------------------------------------------+
                                                                              |  |
                                                                              v  v
                                     +--------------------------------------------------+
                                     |                 DPCoordinatorProc                 |
                                     |                                                  |
                                     |  global control state:                           |
                                     |    - current_wave                                |
                                     |    - engines_running                             |
                                     |    - engine[i].request_counts                    |
                                     |                                                  |
                                     |  inbound from frontend:                          |
                                     |    - FIRST_REQ                                   |
                                     |    - SCALE_ELASTIC_EP                            |
                                     |                                                  |
                                     |  inbound from engines:                           |
                                     |    - scheduler_stats                             |
                                     |    - wave_complete                               |
                                     |    - start_wave                                  |
                                     |                                                  |
                                     |  outbound:                                       |
                                     |    - publish stats/wave to frontend              |
                                     |    - broadcast START_DP_WAVE to engines          |
                                     +--------------------------+-----------------------+
                                                                ^
                                                                |
                                     EngineCoreOutputs          | START_DP_WAVE
                           (stats / wave_complete / start_wave) |
                                                                |
     +---------------------------------+   +---------------------------------+   +---------------------------------+
     |        engine proc dp=0         |   |        engine proc dp=1         |   |        engine proc dp=N-1       |
     |---------------------------------|   |---------------------------------|   |---------------------------------|
     | input_thread                    |   | input_thread                    |   | input_thread                    |
     |   ROUTER <- frontend            |   |   ROUTER <- frontend            |   |   ROUTER <- frontend            |
     |   XSUB   <- coordinator         |   |   XSUB   <- coordinator         |   |   XSUB   <- coordinator         |
     |          -> input_queue         |   |          -> input_queue         |   |          -> input_queue         |
     |                                 |   |                                 |   |                                 |
     | busy_loop: DPEngineCoreProc     |   | busy_loop: DPEngineCoreProc     |   | busy_loop: DPEngineCoreProc     |
     |   - Scheduler                   |   |   - Scheduler                   |   |   - Scheduler                   |
     |   - ModelExecutor               |   |   - ModelExecutor               |   |   - ModelExecutor               |
     |   - current_wave                |   |   - current_wave                |   |   - current_wave                |
     |   - engines_running             |   |   - engines_running             |   |   - engines_running             |
     |                                 |   |                                 |   |                                 |
     | output_thread                   |   | output_thread                   |   | output_thread                   |
     |   PUSH -> frontend outputs      |   |   PUSH -> frontend outputs      |   |   PUSH -> frontend outputs      |
     |   PUSH -> coordinator control   |   |   PUSH -> coordinator control   |   |   PUSH -> coordinator control   |
     +---------------------------------+   +---------------------------------+   +---------------------------------+
                       ^                                 ^                                         ^
                       |                                 |                                         |
                       +---------------------------------+-----------------------------------------+
                                         torch.distributed dp_group
                                      all_reduce(has_unfinished_dp)
```
The most critical things in this picture are the two lines:

1. **Front end <-> coordinator**
   - Control plane cache synchronization is used
   - The front end gets `current_wave/engines_running/lb_engines`

2. **engine <-> coordinator**
   - Use global wave advancement
   - Both `wave_complete` and `start_wave` are reported from here

### 3.2 external LB, and MoE + DP>1

In this form, the front end does not use shared internal LB. Instead, each API server only manages its own local DP rank.```text
     api server 0                    api server 1                    api server N-1
     +----------------------+        +----------------------+        +----------------------+
     | AsyncLLM             |        | AsyncLLM             |        | AsyncLLM             |
     | DPAsyncMPClient      |        | DPAsyncMPClient      |        | DPAsyncMPClient      |
     +----------+-----------+        +----------+-----------+        +----------+-----------+
                |                               |                               |
                | ROUTER / PUSH                 | ROUTER / PUSH                 | ROUTER / PUSH
                v                               v                               v
     +----------------------+        +----------------------+        +----------------------+
     | engine proc dp=0     |        | engine proc dp=1     |        | engine proc dp=N-1   |
     | DPEngineCoreProc     |        | DPEngineCoreProc     |        | DPEngineCoreProc     |
     +----------+-----------+        +----------+-----------+        +----------+-----------+
                \                               |                               /
                 \                              |                              /
                  +-----------------------------+-----------------------------+
                                                |
                                                v
                               +--------------------------------------+
                               |          DPCoordinatorProc           |
                               |  mainly for MoE wave coordination    |
                               |  stats publication may be absent      |
                               +--------------------------------------+
```
The most easily misunderstood point here is:

- **coordinator still exists**
- But `pause_generation()` is no longer naturally a "global pause on all API servers"
- Whether it takes effect globally depends on whether someone fanout the pause command to all front ends.

### 3.3 Internal structure of a single engine process

What is drawn above is the relationship between processes; inside a single `engine proc dp=k`, it can be further reduced to look like this:```text
engine process (dp=k)

    +------------------------------+
    | input_thread                 |
    |  - recv from frontend        |
    |  - recv START_DP_WAVE        |
    |  - preprocess request        |
    |  - push into input_queue     |
    +--------------+---------------+
                   |
                   v
    +------------------------------+
    | DPEngineCoreProc busy_loop   |
    |  - _process_input_queue()    |
    |  - Scheduler                 |
    |  - ModelExecutor             |
    |  - maybe dummy batch         |
    |  - all_reduce unfinished     |
    +--------------+---------------+
                   |
                   v
    +------------------------------+
    | output_thread                |
    |  - send normal outputs       |
    |  - send scheduler_stats      |
    |  - send wave_complete        |
    |  - send start_wave           |
    +------------------------------+
```
This is why it is best to divide the engine process into three layers when reading the source code:

1. `input_thread`
2. `busy_loop`
3. `output_thread`

Otherwise, it is easy to confuse "when the message enters the input_queue" and "when the status takes effect in the busy_loop".

---

## 4. Component division of labor

### 3.1 Front-end side: `DPAsyncMPClient` / `DPLBAsyncMPClient`

Key code:

- `vllm/v1/engine/core_client.py:1109-1286`
- `vllm/v1/engine/core_client.py:1322-1362`

Responsibilities:

1. Cache the triples sent by the coordinator:
   - `current_wave`
   - `engines_running`
   - `lb_engines` (`[waiting, running]` per engine)

2. Mark each new request with `request.current_wave`
   - `vllm/v1/engine/core_client.py:1268-1279`
   - `vllm/v1/engine/__init__.py:87-90`

3. When engines are paused, additionally notify the coordinator through `FIRST_REQ` to wake up other DP ranks.

4. In internal LB mode, use `lb_engines` for engine selection
   - `vllm/v1/engine/core_client.py:1322-1350`

### 3.2 Central control plane: `DPCoordinatorProc`

Key code:

- `vllm/v1/engine/coordinator.py:151-459`

Responsibilities:

1. Receive `scheduler_stats` reported by engine
2. Maintain the overall situation:
   - `current_wave`
   - `engines_running`
3. Broadcast to the front end:
   - Load statistics
   - Global wave / running status
4. Broadcast to engine:
   - `START_DP_WAVE`

### 3.3 engine side: `DPEngineCoreProc`

Key code:

- `vllm/v1/engine/core.py:1579-1758`

Responsibilities:

1. Receive the request and write it to the local scheduler
2. After the local wave ends, use all-reduce to determine whether there are still unfinished requests globally.
3. Report when necessary:
   - `wave_complete`
   - `start_wave`
4. Execute dummy batch when there is no real request but the whole system is still running, driving EP / DP to advance in alignment

### 3.4 Local scheduling plane: `Scheduler.pause_state`

Key code:

- `vllm/v1/core/sched/interface.py:22-33`
- `vllm/v1/core/sched/scheduler.py:368-370`
- `vllm/v1/core/sched/scheduler.py:1842-1858`

Status:

- `UNPAUSED`
- `PAUSED_NEW`
- `PAUSED_ALL`

Semantics:

- `PAUSED_NEW`: existing running requests continue to run, new requests are not scheduled
- `PAUSED_ALL`: No requests are scheduled

---

## 5. Global DP wave state machine

### 4.1 Core status

Coordinator maintenance:

- `current_wave`
- `engines_running`Initialization:

- `vllm/v1/engine/coordinator.py:203-205`
- Initially `current_wave = 0, engines_running = False`

You can think of it as a binary state machine:```text
State A: paused  = (wave = W, running = False)
State B: running = (wave = W, running = True)
```
### 4.2 paused -> running

There are two entrances.

#### Entry 1: The front end sends a new request when it finds paused

Path:

1. The front end tags the request with `request.current_wave = self.current_wave`
   - `core_client.py:1268-1272`
2. If `engines_running == False` in local cache
   - The front end sends additional `FIRST_REQ`
   - `core_client.py:1275-1279`
3. stats task forwards it to coordinator
   - `core_client.py:1227-1236`
4. The coordinator broadcasts `START_DP_WAVE` when `not engines_running`
   - `coordinator.py:348-366`

#### Entrance 2: engine receives the stale-wave request and reports `start_wave` on its own

Path:

1. After the engine receives the request, if `request_wave != self.current_wave`
   - `core.py:1640-1654`
2. If it finds that it is no longer running and the scheduler does not have a local pause
   - It will report `EngineCoreOutputs(start_wave=self.current_wave)`
3. The coordinator broadcasts `START_DP_WAVE` after receiving it
   - `coordinator.py:428-443`

This path is what the code calls "race condition handling".

### 4.3 running -> paused

Path:

1. Each DP rank calculates local `local_unfinished_reqs` in busy loop
   - `core.py:1715`
2. Do `all_reduce(MAX)` every 32 steps
   - `core.py:1752-1758`
   - `config/parallel.py:576-584`
3. If there are no unfinished requests globally:
   - `self.engines_running = False`
   - rank0 sends `wave_complete=self.current_wave` to coordinator
   - `core.py:1730-1745`
4. Execute engine locally:
   - `self.current_wave += 1`
   - `self.step_counter = 0`
   - `core.py:1746-1748`
5. After the coordinator receives `wave_complete=W`, it switches to:
   - `current_wave = W + 1`
   - `engines_running = False`
   - `coordinator.py:414-427`

### 4.4 How does the engine change after receiving `START_DP_WAVE`?

Key logic:

- `core.py:1668-1681`

Semantics:

1. Receive `(new_wave, exclude_eng_index)`
2. If you are not an excluded engine and `new_wave >= self.current_wave`
3. Update `self.current_wave = new_wave`
4. If you are not currently running, switch to running

The `exclude_engine_index` here is used to prevent the engine that has already received the request from being awakened repeatedly.

---

## 6. Local pause / wait / keep / resume state machine### 5.1 API entrance

User visible entrance:

- `vllm/v1/engine/async_llm.py:726-775`

Among them:

- `pause_generation(mode="abort"|"wait"|"keep")`
- `resume_generation()`
- `is_paused()`

The call eventually falls to:

- `core_client.py:1039-1048`
- `core.py:1507-1546`

Note:

- The comments in `core.py` describe the completion semantics of `abort`/`keep` more towards "output has been sent"
- But in **actual implementation**, the completion conditions of future are unified and closer to:
  - `_idle_state_callbacks` is triggered after `has_work() == False`
  - See `core.py:1539-1545` and `core.py:1149-1153` for the code

The following analysis is based on **real behavior**, not the annotation text.

### 5.2 Three pause modes

#### `abort`

Logic:

- `core.py:1531-1537`

Behavior:

1. Direct `finish_requests(..., FINISHED_ABORTED)`
2. Send abort outputs
3. `pause_state = PAUSED_NEW`
4. Wait for the engine to enter idle and then complete the future.

#### `wait`

Logic:

- `core.py:1507-1546`
- `scheduler.py:1848-1858`

Behavior:

1. `pause_state = PAUSED_NEW`
2. New requests can still enter the queue, but will not be scheduled.
3. The currently running request continues to run.
4. When `len(running) == 0`, future is completed

Note that "wait completed" here does not mean "the system has no requests at all", but "the current in-flight running requests have been cleared."

#### `keep`

Logic:

- `core.py:1537-1546`
- `scheduler.py:368-370`
- `scheduler.py:1848-1850`

Behavior:

1. `pause_state = PAUSED_ALL`
2. All requests are frozen and will no longer be scheduled.
3. `get_num_unfinished_requests()` returns 0 directly under `PAUSED_ALL`
4. The engine will regard itself as "accessible to idle"
5. Then use `resume_scheduler()` to resume

### 5.3 resume

Logic:

- Normal engine: `core.py:637-639`
- DP MoE engine: `core.py:1656-1666`

One more thing has been done under DP MoE:

1. `pause_state -> UNPAUSED`
2. If the current global `engines_running == False`
3. There are still unfinished requests in the local scheduler
4. Actively report `start_wave=self.current_wave`

That is to say:

- `resume_scheduler()` only restores local scheduling permissions
- But `DPEngineCoreProc.resume_scheduler()` will also try to restore the global wave

---

## 7. How to superimpose two sets of state machines

In actual operation, at least the following two levels of "pause" must be distinguished:

### 6.1 Global wave pause

Source:

- All DP ranks have no unfinished requests globally

Reflection:

- `engines_running = False`
- coordinator/front-end/DPEngineCoreProc's `current_wave` forward### 6.2 Local scheduling pause

Source:

- Explicitly call `pause_generation()`

Reflection:

- `scheduler.pause_state != UNPAUSED`

### 6.3 The true meaning of the combination of the two

It can be roughly understood as:```text
Whether the request can actually be advanced
= local scheduler allows scheduling
AND global wave is already running

But is the engine busy loop itself still spinning?
= engines_running
  OR scheduler.has_requests()
OR batch_queue is not empty
```
Code location:

- `core.py:1124-1130`

This explains two confusing points:

1. `resume_generation()` does not mean starting calculation immediately
   - If the global wave has not been started yet, `start_wave` is also needed

2. `engines_running=False` does not mean that there is no local request
   - Under `keep` / `wait`, there may still be frozen requests in the local queue

---

## 8. Race conditions that have been explicitly handled in the code

This part is the race that is "already aware and handled" in the current design.

### 7.1 Front-end current_wave expired

Processing methods:

- Each request comes with `request.current_wave`
  - `engine/__init__.py:87-90`
- The engine can report `start_wave` when receiving a stale wave
  - `core.py:1640-1654`
- The coordinator broadcasts `START_DP_WAVE` after receiving it
  - `coordinator.py:428-443`

### 7.2 The front end sends the first request during paused period

Processing methods:

- `FIRST_REQ`
  - `core_client.py:1276-1279`
- The coordinator will start this wave after receiving the notification in the paused state.
  - `coordinator.py:355-366`

### 7.3 stats out of order

Processing methods:

- coordinator performs monotonicity check on `(stats_wave, stats_step)`
  - `coordinator.py:383-406`

These mechanisms indicate that the current implementation is not "completely streaking", but uses the `wave` number as a lightweight epoch to repair the disorder caused by asynchronous messages on the control plane.

---

## 9. From the perspective of competition, the weakest part of the current implementation

The following is written in the order of "I think the most worthy of vigilance".

### 8.1 The completion time of `pause_generation()` is not equal to "all output has been processed by the front end"

Key code:

- `core.py:1526-1545`
- `async_llm.py:760-767`

Phenomenon:

1. The pause future on the engine side is completed when `_idle_state_callbacks` is triggered.
2. This time point only indicates that the engine enters idle
3. But the link output_queue -> output thread -> client output task -> output_processor may not be completely drained.
4. Therefore, a special addition is added to `AsyncLLM.pause_generation()`:
   - `await asyncio.sleep(0.02)`

This shows that the current implementation itself recognizes:

- There is a window between "pause completed" and "the final output from the user perspective is stably visible"
- The current patching method is empirical sleep instead of strict barrier

This is a typical race condition where the control plane is completed first and the data plane is finished later.

### 8.2 `wait_for_requests_to_drain()` determines `engines_running`, not "the system is really empty"

Key code:

- `async_llm.py:950-964`
- `core_client.py:1248-1251`
- `scheduler.py:1848-1858`

The problem is:

1. `wait_for_requests_to_drain()` only see `dp_engines_running()`
2. `dp_engines_running()` is essentially the global running flag broadcast by the coordinator
3. It is not equivalent to:
   - The front-end output queue has been drained
   - The local waiting queue is empty
   - Frozen requests in `keep`/`wait` mode have disappearedEspecially in:

- `PAUSED_NEW`: `get_num_unfinished_requests()` only counts `running`
- `PAUSED_ALL`: Return 0 directly

So the real semantics of "drain completion" are closer:

- **Global wave has stopped**

instead of:

- **There is no request status at all in the system**

This is ambiguous for high-level control actions such as "capacity expansion" and "hot update".

### 8.3 The global finished detection is only synchronized every 32 steps, and there is a natural delay window.

Key code:

- `core.py:1752-1758`

The logic is:```text
step_counter += 1
Only do all_reduce(MAX) once when step_counter % 32 == 0
Otherwise, return True directly
```
This means:

1. Even if there is no unfinished request globally
2. The system may also continue to run for up to 31 more "thinking it is still running" cycles.
3. During these cycles:
   - The coordinator doesn't know yet that the wave has ended
   - The front-end `engines_running` has not changed yet
   - `wait_for_requests_to_drain()` will continue to wait
   - engine may continue to do dummy batch

This is not a correctness bug, but it significantly increases the race window and control surface lag.

### 8.4 internal LB relies on the coordinator’s 100ms level stats. Multiple front-ends will concurrently grab the same engine.

Key code:

- `coordinator.py:263-285`
- `core_client.py:1329-1345`

Current status:

1. The coordinator publishes stats every 100ms by default.
2. `DPLBAsyncMPClient` uses the `lb_engines` received last time to select rank
3. To alleviate the stale stats problem, it only increments a local waiting count inside this client:
   - `current_counts[eng_index][0] += self.client_count`

Weak points:

- This "optimistic increment" is not shared globally
- When multiple API servers see the same old stats at the same time, they may still send requests to the same engine together.

Therefore, this is a problem of weak load balancing consistency. It is not necessarily wrong, but it will cause instantaneous tilt.

### 8.5 `run_engine_stats_update_task()`’s processing of `FIRST_REQ` depends on the poll return sequence, and the writing method is relatively brittle

Key code:

- `core_client.py:1178-1245`

Especially this condition:```python
if (
    not self.engines_running
    and len(events) == 2
    or (events[0][0] == first_req_rcv_socket)
):
```
Its features are:

1. The behavior partially depends on who `events[0]` is
2. `FIRST_REQ` and stats on two different sockets
3. Only one `FIRST_REQ` is explicitly processed in a loop
4. stats is "drain to the latest", but `FIRST_REQ` is not symmetrically processed

This is not to say that it is necessarily wrong, but it is to say:

- It is sensitive to event interleaving
- Poor readability
- Once more control messages (such as expansion and contraction, additional barriers) are mixed in later, it is easy to create boundary bugs

This is an obvious control plane logic vulnerability.

### 8.6 During Elastic EP scale-up, the `current_wave` initialization of the new engine is out of sync with the existing network wave, and subsequent traffic is required to remedy the problem.

Key code:

- `coordinator.py:321-333`

This annotation already directly acknowledges:

- `current_wave = 0` for newly started engine
- But existing engines may already be in higher waves

The current implementation does not do an explicit "wave bootstrap/snapshot sync" when scale-up completes, but instead relies on:

1. `request.current_wave` carried in subsequent requests
2. Or the coordinator will send `START_DP_WAVE` later.

To bring the new engine back to the correct epoch.

This shows that the control surface of Elastic EP is still **eventually consistent**, not **strongly consistent**.

### 8.7 Whether pause/resume is "globally effective" depends on the client topology and is not guaranteed by the coordinator.

Key code:

- `core_client.py:1039-1048`
- `core_client.py:1352-1362`

Current status:

1. `DPLBAsyncMPClient.call_utility_async()` will `gather` to all engines managed by this client
2. But `DPAsyncMPClient` does not; it only adjusts the engine it manages

Meaning:

- Under **internal LB**, `pause_generation()` is basically equivalent to "broadcasting all engines in this front-end pipe"
- Under **external LB**, `pause_generation()` is not a global pause at the coordinator level, it is just a local pause for this front-end / this rank

Therefore, in the current implementation, the question of "whether pause is a global action" is not provided by the coordinator, but is determined by the deployment topology.

This is a semantic crack that is easy to step on for the operation and maintenance layer.

### 8.8 `has_work()` bears two semantics: "there is really work" and "although it is frozen, the loop must be maintained", making it difficult for high-level managers to understand idle accurately.

Key code:

- `core.py:1124-1130`
- `scheduler.py:1848-1858`

`has_work()` depends on:

- `engines_running`
- `scheduler.has_requests()`
- `batch_queue`

But `scheduler.has_requests()` has been rewritten by pause state to have semantics:

- `PAUSED_NEW`: New requests in waiting are not counted as unfinished
- `PAUSED_ALL`: direct 0

So "idle" can mean three completely different things:

1. There is really no request
2. There are still requests, but they are all frozen.
3. There are still requests, but the global wave is stopped

This makes the upper-level control logic very easy to misjudge.

---

## 10. I think the most important boundary judgment in current design

### 9.1 `START_DP_WAVE` is not `resume_scheduler`They control respectively:

- `START_DP_WAVE`: DP global running/paused
- `resume_scheduler`: whether the local scheduler allows scheduling

Any analysis that mixes these two concepts will quickly look at the wrong code.

### 9.2 `wave` is a lightweight epoch, not a strict barrier token

It can handle common stale messages, but it is not strictly distributed transaction id.

So the current implementation is essentially:

- **Fix race conditions with monotonic wave numbering + eventually consistent message propagation**

instead of:

- **Rely on strong barrier/ack mechanism to eliminate race conditions**

### 9.3 The "completion" semantics of `wait`/`keep` are engineering semantics, not mathematical semantics

Specifically:

- `wait` completed: running cleared
- `keep` completed: engine can be stopped to idle

Neither of them equals "there is no request state inside the system".

---

## 11. If I want to reinforce, what will I do first?

### 10.1 Replace `sleep(0.02)` with explicit output-drain barrier

Goal:

- Before the pause future is completed, ensure that the engine output has been completely consumed by the client output task and output_processor

### 10.2 Split "wave idle" and "request state empty" into two independent signals

It is recommended to distinguish at least:

- `global_wave_running`
- `local_scheduler_empty`
- `frontend_output_drained`
- `paused_with_frozen_requests`

### 10.3 Make 32-step finished sync configurable, or downsample the interval at the end

The current fixed 32 steps is too hard and will directly increase the convergence delay.

### 10.4 Add explicit wave snapshot synchronization to elastic scale-up

Do not let the new engine enter the system with `current_wave=0` and then rely on traffic to repair it.

### 10.5 Rewrite event handling of `run_engine_stats_update_task()`

Goal:

- Does not depend on `events[0]`
- `FIRST_REQ` / stats / scale signals are drained respectively
- Clarify the priority of each message

### 10.6 Provide a true "global pause/resume" control surface under external LB

Otherwise the current pause semantics can easily be mistaken for cluster-wide, when in fact it is just local.

---

## 12. Final summary

Currently, vLLM actually relies on the following set of mechanisms to work together under DP>1, especially **MoE + DP>1**:

1. `DPCoordinatorProc`
   - Maintain global `current_wave/engines_running`
   - Broadcast `START_DP_WAVE`
   - Aggregate LB stats

2. `DPEngineCoreProc`
   - Accept requests locally
   - Use all-reduce to determine whether there are unfinished requests globally
   - Push back `start_wave` in stale-wave scenario

3. `Scheduler.pause_state`
   - Provide `UNPAUSED / PAUSED_NEW / PAUSED_ALL`
   - Implement `abort/wait/keep/resume`

Its advantages are:

- The structure is not too complicated
- Already have patch paths for common stale-wave races
- No need for heavy distributed barrier protocolsIts weaknesses are:

- Many places are **eventually consistent** rather than **strongly consistent**
- Mixed semantics of high-level "pause/drain/idle"
- At least one place explicitly relies on `sleep(20ms)` to compensate for race
- There are obvious vulnerabilities in the control surfaces under Elastic EP and multi-frontend

If you want to dig deeper later, I suggest you draw two pictures directly in the next step:

1. Timing diagram of `FIRST_REQ / start_wave / wave_complete / START_DP_WAVE`
2. Timing diagram of `pause_generation(wait|keep) -> idle callback -> output processor`

These two pictures will expose the remaining race conditions faster than continuing to pile up source code.