# vLLM Data Parallel Load Balancing Deep Dive

> Code analysis based on vllm-project/vllm main branch (2026-04)
> Related Issues: https://github.com/vllm-project/vllm/issues/24461
> Related documents: https://docs.vllm.ai/en/stable/serving/data_parallel_deployment/

---

## 1. Data Parallel Architecture Overview

### 1.1 What is Data Parallel (DP)

The Data Parallel deployment method of vLLM is to copy model weights to multiple GPUs/instances, and each instance independently processes different batches of requests.

Core architectural points:
- Each DP rank is an independent **core engine process** that communicates with the front-end process through ZMQ socket
- Each DP engine has **independent KV cache**
- For the MoE model, DP ranks are **not completely independent** — the expert layer needs to be synchronized across ranks (the forward pass must be aligned, and the idle rank needs to perform dummy forward pass)
- For dense models, each rank is **completely independent** and does not require synchronization.

### 1.2 The relationship between DP and P/D separation

| Concept | What to do | Granularity |
|------|--------|------|
| **P/D separation** | Split the Prefill and Decode stages into different instances | Divided by inference stage |
| **DP LB** | Distribute requests among multiple replicas of the same role | Distribute by replica |

The two are orthogonal and can be used in combination. For example, if there are 4 Prefill instances + 8 Decode instances, DP LB is used to offload the prefill instances, DP LB is also used to offload the decode instances, and KV transfer is used to transfer the KV cache between P/D.

---

## 2. Three load balancing modes

### 2.1 Internal LB — "One front desk, multiple kitchens"
```
User request ──→ [Unique API Server] ──→ Engine 0 (GPU 0)
                    │                ──→ Engine 1 (GPU 1)
                    │                ──→ Engine 2 (GPU 2)
                    │                ──→ Engine 3 (GPU 3)
└─ Internal decision on who to send to
```
**Start everything with one command**:```bash
vllm serve $MODEL --data-parallel-size 4
```
- Only **one HTTP port** (e.g. 8000)
- The API Server process inside vLLM sees the queue lengths of all engines and decides to distribute them by itself.
- The user only sees one service address and has no idea how many engines are behind it.
- **Suitable**: Single machine with multiple cards, small-scale deployment

Multiple nodes:```bash
# Node 0 (head, ip 10.99.48.128)
vllm serve $MODEL --data-parallel-size 4 --data-parallel-size-local 2 \
    --data-parallel-address 10.99.48.128 --data-parallel-rpc-port 13345
# Node 1
vllm serve $MODEL --headless --data-parallel-size 4 --data-parallel-size-local 2 \
    --data-parallel-start-rank 2 \
    --data-parallel-address 10.99.48.128 --data-parallel-rpc-port 13345
```
**Limitations**: API Server is a single node, and the DP size becomes a bottleneck (it can be alleviated with `--api-server-count`, but it is still limited to a single node).

### 2.2 Hybrid LB — "There is a front desk on each floor and a security diversion at the gate."
```
                           ┌─→ [ Node 0 API Server ] ──→ Engine 0, Engine 1
User request ──→ [External LB] ──┤
                           └─→ [ Node 1 API Server ] ──→ Engine 2, Engine 3
Each node schedules itself internally
```
**Each node starts its own API Server**:```bash
# Node 0
vllm serve $MODEL --data-parallel-size 4 --data-parallel-size-local 2 \
    --data-parallel-start-rank 0 --data-parallel-hybrid-lb
# Node 1
vllm serve $MODEL --data-parallel-size 4 --data-parallel-size-local 2 \
    --data-parallel-start-rank 2 --data-parallel-hybrid-lb
```
- Each node has **its own HTTP port**
- The internal scheduling of the node is still done by vLLM itself (the same as Internal LB)
- Traffic distribution between nodes is handled by **external LB** (Nginx/K8s Ingress)
- **Suitable**: Multi-node medium-sized, want to reduce cross-node communication

### 2.3 External LB — "Each chef opens a store independently and is diverted to external platforms"
```
                           ┌─→ [ vllm serve :8000 ] ── Engine 0 (GPU 0)
                           ├─→ [ vllm serve :8001 ] ── Engine 1 (GPU 1)
User request ──→ [External Router] ──┤
                           ├─→ [ vllm serve :8002 ] ── Engine 2 (GPU 2)
                           └─→ [ vllm serve :8003 ] ── Engine 3 (GPU 3)
Each is an independent process
```
**Each rank starts independently**:```bash
CUDA_VISIBLE_DEVICES=0 vllm serve $MODEL --data-parallel-size 4 --data-parallel-rank 0 --port 8000
CUDA_VISIBLE_DEVICES=1 vllm serve $MODEL --data-parallel-size 4 --data-parallel-rank 1 --port 8001
CUDA_VISIBLE_DEVICES=2 vllm serve $MODEL --data-parallel-size 4 --data-parallel-rank 2 --port 8002
CUDA_VISIBLE_DEVICES=3 vllm serve $MODEL --data-parallel-size 4 --data-parallel-rank 3 --port 8003
```
- Each engine has its own HTTP port and a completely independent `vllm serve` process
- **All scheduling decisions are external**: The external router can do intelligent routing based on real-time telemetry
- **Suitable**: Large-scale production deployment, K8s environment (each rank = one Pod)

**For dense model**: No need for `--data-parallel-size/rank`, just start N completely independent `vllm serve`.
**For MoE model**: `--data-parallel-size/rank` is still required because the expert layer has synchronization requirements across ranks.

### 2.4 Comparison summary

| Mode | Who schedules | Number of ports | Startup method | Scale |
|------|--------|--------|---------|------|
| Internal | vLLM Internal | 1 | One command | Small |
| Hybrid | Intra-node vLLM + Inter-node external | 1 per node | One command per node | Medium |
| External | Completely external | 1 per rank | One command per rank | Large |

---

## 3. Detailed explanation of Internal LB strategy

### 3.1 Data flow
```
Engine 0 ──scheduler_stats──→                    ┌──→ API Server 0
Engine 1 ──scheduler_stats──→ DP Coordinator ──┤ (lb_engines local cache)
Engine 2 ──scheduler_stats──→ (summary release every 100ms) └──→ API Server 1
Engine 3 ──scheduler_stats──→ (lb_engines local cache)
```
Each Engine's Scheduler reports two numbers after each step:
- `num_waiting_reqs` — Number of requests waiting in queue
- `num_running_reqs` — Number of requests being executed

The DP Coordinator process summarizes the statistics of all engines and pushes them to all API Servers through ZMQ XPUB **every 100ms**.

Critical code path:
- Coordinator released: `vllm/v1/engine/coordinator.py:282`
- API Server receives: `vllm/v1/engine/core_client.py:1248-1259`
- Routing decision: `vllm/v1/engine/core_client.py:1322-1350`

### 3.2 Core selection algorithm: Weighted Least-Loaded```python
# vllm/v1/engine/core_client.py:1322-1345
def get_core_engine_for_request(self, request):
    current_counts = self.lb_engines    # [[waiting, running], ...]
    num_engines = len(current_counts)
    min_score = sys.maxsize
    eng_index = 0

    for i in range(num_engines):
idx = (self.eng_start_index + i) % num_engines # Rotation starting point
        waiting, running = current_counts[idx]
score = waiting * 4 + running # Weighted score
        if score < min_score:
            min_score = score
            eng_index = idx

# Local optimistic update to avoid the thunder group effect between two coordinator pushes
    current_counts[eng_index][0] += self.client_count
    return self.core_engines[eng_index]
```
### 3.3 Scoring formula
```
score = waiting x 4 + running
```
- **waiting weight 4x**: Queued requests are much more important than running requests. Waiting means that the KV cache is already tight or the batch is full, and new requests will wait longer.
- **running weight 1x**: Although running requests occupy resources, they are consumed normally and do not represent congestion.
- Select the engine with the **smallest** score

**Example**:
| Engine | waiting | running | score |
|--------|---------|---------|-------|
| 0 | 2 | 30 | 2x4+30 = **38** |
| 1 | 0 | 40 | 0x4+40 = **40** |
| 2 | 5 | 10 | 5x4+10 = **30** <-- Select this |
| 3 | 3 | 25 | 3x4+25 = **37** |

### 3.4 Rotation starting point (eng_start_index)```python
idx = (self.eng_start_index + i) % num_engines
```
When multiple engine scores are the same (for example, they are all 0), the traversal order determines who is selected. `eng_start_index` allows **different API Server processes to start scanning** from different engines to prevent all API Servers from hitting Engine 0 when no load occurs.

### 3.5 Local Optimistic Update (Lightning Protection Group)```python
current_counts[eng_index][0] += self.client_count
```
Coordinator's statistics push interval is 100ms. Between two pushes, if multiple requests arrive, they see the same snapshot and may all choose the same engine (thundering herd).

Solution: Each time an engine is selected, **locally immediately sets the waiting +N** of that engine** (N = the total number of API Servers, that is, `client_count`). Subsequent requests will naturally be distributed to other engines. When the coordinator pushes next time, the local cache will be overwritten with the real value.

### 3.6 Future evolution TODO```python
# TODO use P2C alg for larger DP sizes
```
Currently it is O(N) full scan to select the smallest one. For large-scale DP (64+ ranks), the plan is to switch to the **Power-of-2-Choices (P2C)** algorithm - randomly select 2 engines and take the one with the lower score. P2C is close to optimal in theory and practice, and has a complexity of O(1).

---

## 4. Cache-Aware / Session-Aware routing: Current status

### 4.1 Conclusion: vLLM currently does not

The API Server process (`DPLBAsyncMPClient`) does not maintain any cache-aware or session-aware state**.

All routing related status held by API Server:

| Field | Type | Purpose |
|------|------|------|
| `lb_engines` | `list[list[int]]` | `[waiting, running]` for each engine, pushed by Coordinator every 100ms |
| `reqs_in_flight` | `dict[str, EngineIdentity]` | Track which engine the request is in, **only used for abort routing** |

Does not include:
- No prefix tree / radix tree
- No per-engine cache hit rate statistics
- There is no mapping table for prompt hash -> engine
- No sticky binding for session ID -> engine
- No KV cache utilization information

Original document (`docs/serving/data_parallel_deployment.md`):
> Currently, the internal DP load balancing is done within the API server process(es) and is based on the running and waiting queues in each of the engines. This could be made more sophisticated in future by incorporating KV cache aware logic.

### 4.2 X-data-parallel-rank Header — External injection mechanism

API Server provides an entry (`vllm/entrypoints/openai/engine/serving.py:766-778`):```python
def _get_data_parallel_rank(raw_request):
    rank_str = raw_request.headers.get("X-data-parallel-rank")
    return int(rank_str) if rank_str else None
```
The external router can pass the HTTP header `X-data-parallel-rank: 2` **forcibly specify** which engine the request is sent to. Highest priority during LB selection:```python
# core_client.py:1324 — If rank is specified externally, skip the LB logic directly
if (eng_index := request.data_parallel_rank) is None and ...
# Only then enter the weighted least-loaded selection
```
If you need cache-aware routing, you can implement the logic in the external router and then inject decisions through this header.

### 4.3 cache_salt — Cache isolation within a single Engine

There is a `cache_salt` field in `EngineCoreRequest`, but its function is to isolate the prefix cache within a single engine (for example, caches of different tenants do not cross each other), and has nothing to do with cross-engine routing.

### 4.4 Summary of routing decision inputs
```
Inputs to vLLM API Server routing decisions:

[Y] Number of waiting / running requests for each engine (Coordinator push)
[Y] Externally injected X-data-parallel-rank header (bypass LB)
[N] Prefix cache status of each engine
[N] KV cache utilization of each engine
[N] prompt -> historical mapping of engine
[N] session / conversation affinity
```
### 4.5 SGLang comparison: Cache-Aware implemented

SGLang's sgl-model-gateway (Rust implementation) provides multiple strategies:

| Strategy | Description |
|------|------|
| `cache_aware` | Maintain the radix prefix tree, track which prefixes are cached by each worker, and route requests with the same prefix to the same worker |
| `consistent_hashing` | hash ring does session affinity, `X-SMG-Routing-Key` header controls sticky |
| `prefix_hash` | Lightweight version cache-aware, consistent hash + bounded load, O(log n) |
| `power_of_two` | P2C algorithm |
| `round_robin` / `random` | Basic strategy |

### 4.6 If you need the option of Cache-Aware routing

1. **Wait for vLLM to implement** cache-aware internal LB (currently just roadmap)
2. **External LB + implement routing logic by yourself** (based on consistent hashing of prompt hash + `X-data-parallel-rank` header injection)
3. **Refer to SGLang’s sgl-model-gateway** as an external router (already has a mature `cache_aware` strategy)

---

## 5. Issue #24461: DP performance optimization for non-MoE models

### 5.1 Problem description

> **[BugFix]: Avoid unnecessary coordination for non-MoE data parallel**
> Author: Nick Hill (njhill) | Status: CLOSED | Fixed PR: #30739

When DP is enabled, the implementation of vLLM makes the unified assumption that all models are MoE models, and therefore is performed for all DP ranks:

1. **Unnecessary cross-rank synchronization** — all-reduce operation of shared metadata before each forward pass
2. **Unnecessary dummy forward pass** — idle ranks are forced to perform dummy operations to stay synchronized
3. **Unnecessary step/wave coordination** — DP Coordinator forces all ranks to be in step

For **dense models** (such as Qwen3-8B, Llama, etc.), these synchronization operations are completely unnecessary.
```
MoE model: There are dependencies between DP ranks -> need to be synchronized (reasonable)
Dense model: DP ranks are completely independent -> synchronization is pure overhead (bug)
```
Nick Hill also pointed out: Even if synchronization is removed for dense models, the DP Coordinator should still be left running because it is responsible for propagating load balancing statistics.

### 5.2 Fix (PR #30739)

**"[BugFix] Support online dense model DP without overhead"** | 2026-01-02 merged

Core changes:
- Set the parallel config of the dense model to `DP=1` at the worker level, so that each rank **runs completely independently**
- When using internal LB, DP Coordinator still runs to propagate load balancing statistics
- But **step/wave synchronization logic is disabled** (`enable_wave_coordination=False`)
- Only supports online / AsyncLLM scenarios
- For the combination of offline DP + dense model, fail directly at startup

Key documents involved:
- `vllm/v1/engine/coordinator.py` — Coordinator logic decoupling
- `vllm/v1/engine/core.py` — engine core runs logic independently
- `vllm/config/parallel.py`, `vllm/config/vllm.py` — The configuration layer determines whether MoE coordination is required
- `vllm/distributed/parallel_state.py` — parallel state adjustment
- `vllm/v1/worker/gpu_worker.py`, `gpu/model_runner.py` — Remove unnecessary synchronization at the worker level

### 5.3 Performance improvement (4xH100, Qwen3-8B, DP=4)

| Metrics | Before | After | Improvement |
|------|--------|-------|------|
| Request throughput (req/s) | 38.31 | **40.31** | +5.2% |
| Output token throughput (tok/s) | 19,615 | **20,637** | +5.2% |
| Mean TTFT (ms) | 131.78 | **88.94** | **-32.5%** |
| Median TTFT (ms) | 124.31 | **74.67** | **-39.9%** |
| Mean TPOT (ms) | 9.93 | **9.50** | -4.3% |
| P99 ITL (ms) | 15.93 | **12.86** | -19.3% |

The most significant improvement is in TTFT (first token delay), which is reduced by about 30-40% because the synchronization wait before each forward pass is removed.

---

## 6. Key code index

| Components | Files | Key Lines |
|------|------|--------|
| DP Coordinator process | `vllm/v1/engine/coordinator.py` | `DPCoordinatorProc` L151 |
| Stats release | `vllm/v1/engine/coordinator.py` | L282 |
| Stats receiving | `vllm/v1/engine/core_client.py` | L1248-1259 |
| LB routing decision | `vllm/v1/engine/core_client.py` | `get_core_engine_for_request` L1322 |
| X-data-parallel-rank parsing | `vllm/entrypoints/openai/engine/serving.py` | `_get_data_parallel_rank` L766 |
| EngineCoreRequest definition | `vllm/v1/engine/__init__.py` | L72-94 |
| SchedulerStats | `vllm/v1/metrics/stats.py` | L171-198 |
| Internal LB test | `tests/v1/distributed/test_internal_lb_dp.py` | |
| External LB test | `tests/v1/distributed/test_external_lb_dp.py` | |