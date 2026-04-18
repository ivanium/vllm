# vLLM Data Parallel Load Balancing Deep Dive

> 基于 vllm-project/vllm main 分支 (2026-04) 代码分析
> 相关 Issue: https://github.com/vllm-project/vllm/issues/24461
> 相关文档: https://docs.vllm.ai/en/stable/serving/data_parallel_deployment/

---

## 1. Data Parallel 架构概述

### 1.1 什么是 Data Parallel (DP)

vLLM 的 Data Parallel 部署方式是将**模型权重复制到多个 GPU/实例**上，每个实例独立处理不同批次的请求。

核心架构点:
- 每个 DP rank 是一个独立的 **core engine 进程**，通过 ZMQ socket 与前端进程通信
- 每个 DP engine 有**独立的 KV cache**
- 对于 MoE 模型，DP ranks **不完全独立** — expert 层需要跨 rank 同步（forward pass 必须对齐，空闲 rank 需执行 dummy forward pass）
- 对于 dense 模型，各 rank **完全独立**，无需同步

### 1.2 DP 与 P/D 分离的关系

| 概念 | 做什么 | 粒度 |
|------|--------|------|
| **P/D 分离** | Prefill 和 Decode 阶段拆到不同实例 | 按推理阶段分 |
| **DP LB** | 同一角色的多个副本之间分配请求 | 按副本分 |

两者正交，可以组合使用。比如 4 个 Prefill 实例 + 8 个 Decode 实例，Prefill 之间用 DP LB 分流，Decode 之间也用 DP LB 分流，P/D 之间用 KV transfer 传递 KV cache。

---

## 2. 三种负载均衡模式

### 2.1 Internal LB — "一个前台，多个厨房"

```
用户请求 ──→ [ 唯一的 API Server ] ──→ Engine 0 (GPU 0)
                    │                ──→ Engine 1 (GPU 1)
                    │                ──→ Engine 2 (GPU 2)
                    │                ──→ Engine 3 (GPU 3)
                    └─ 内部决定发给谁
```

**一条命令启动所有东西**:
```bash
vllm serve $MODEL --data-parallel-size 4
```

- 只有**一个 HTTP 端口**（比如 8000）
- vLLM 内部的 API Server 进程看到所有 engine 的队列长度，自己决定分发
- 用户只看到一个服务地址，完全不知道后面有几个 engine
- **适合**: 单机多卡、小规模部署

多节点:
```bash
# Node 0 (head, ip 10.99.48.128)
vllm serve $MODEL --data-parallel-size 4 --data-parallel-size-local 2 \
    --data-parallel-address 10.99.48.128 --data-parallel-rpc-port 13345
# Node 1
vllm serve $MODEL --headless --data-parallel-size 4 --data-parallel-size-local 2 \
    --data-parallel-start-rank 2 \
    --data-parallel-address 10.99.48.128 --data-parallel-rpc-port 13345
```

**局限**: API Server 是单点，DP size 大了成瓶颈（可以用 `--api-server-count` 缓解，但仍限于单节点）。

### 2.2 Hybrid LB — "每层楼有个前台，大门口有个保安分流"

```
                           ┌─→ [ Node 0 API Server ] ──→ Engine 0, Engine 1
用户请求 ──→ [ 外部 LB ] ──┤
                           └─→ [ Node 1 API Server ] ──→ Engine 2, Engine 3
                                各节点内部自己调度
```

**每个节点启动自己的 API Server**:
```bash
# Node 0
vllm serve $MODEL --data-parallel-size 4 --data-parallel-size-local 2 \
    --data-parallel-start-rank 0 --data-parallel-hybrid-lb
# Node 1
vllm serve $MODEL --data-parallel-size 4 --data-parallel-size-local 2 \
    --data-parallel-start-rank 2 --data-parallel-hybrid-lb
```

- 每个节点有**自己的 HTTP 端口**
- 节点内部的调度还是 vLLM 自己做（和 Internal LB 一样）
- 节点之间的流量分配由**外部 LB**（Nginx/K8s Ingress）负责
- **适合**: 多节点中等规模，想减少跨节点通信

### 2.3 External LB — "每个厨师独立开店，外部平台分流"

```
                           ┌─→ [ vllm serve :8000 ] ── Engine 0 (GPU 0)
                           ├─→ [ vllm serve :8001 ] ── Engine 1 (GPU 1)
用户请求 ──→ [ 外部 Router ] ──┤
                           ├─→ [ vllm serve :8002 ] ── Engine 2 (GPU 2)
                           └─→ [ vllm serve :8003 ] ── Engine 3 (GPU 3)
                                 每个都是独立进程
```

**每个 rank 独立启动**:
```bash
CUDA_VISIBLE_DEVICES=0 vllm serve $MODEL --data-parallel-size 4 --data-parallel-rank 0 --port 8000
CUDA_VISIBLE_DEVICES=1 vllm serve $MODEL --data-parallel-size 4 --data-parallel-rank 1 --port 8001
CUDA_VISIBLE_DEVICES=2 vllm serve $MODEL --data-parallel-size 4 --data-parallel-rank 2 --port 8002
CUDA_VISIBLE_DEVICES=3 vllm serve $MODEL --data-parallel-size 4 --data-parallel-rank 3 --port 8003
```

- 每个 engine 有**自己的 HTTP 端口**，完全独立的 `vllm serve` 进程
- **所有调度决策都在外部**: 外部 router 可以根据实时遥测做智能路由
- **适合**: 大规模生产部署、K8s 环境（每个 rank = 一个 Pod）

**对 dense 模型**: 不需要 `--data-parallel-size/rank`，就是启动 N 个完全独立的 `vllm serve`。
**对 MoE 模型**: 仍需要 `--data-parallel-size/rank`，因为 expert 层跨 rank 有同步需求。

### 2.4 对比总结

| 模式 | 谁调度 | 端口数 | 启动方式 | 规模 |
|------|--------|--------|---------|------|
| Internal | vLLM 内部 | 1 个 | 一条命令 | 小 |
| Hybrid | 节点内 vLLM + 节点间外部 | 每节点 1 个 | 每节点一条命令 | 中 |
| External | 完全外部 | 每 rank 1 个 | 每 rank 一条命令 | 大 |

---

## 3. Internal LB 策略详解

### 3.1 数据流

```
Engine 0 ──scheduler_stats──→                    ┌──→ API Server 0
Engine 1 ──scheduler_stats──→  DP Coordinator  ──┤    (lb_engines 本地缓存)
Engine 2 ──scheduler_stats──→  (每100ms汇总发布)  └──→ API Server 1
Engine 3 ──scheduler_stats──→                         (lb_engines 本地缓存)
```

每个 Engine 的 Scheduler 在每个 step 后上报两个数字:
- `num_waiting_reqs` — 排队等待的请求数
- `num_running_reqs` — 正在执行的请求数

DP Coordinator 进程汇总所有 engine 的统计，**每 100ms** 通过 ZMQ XPUB 推送给所有 API Server。

关键代码路径:
- Coordinator 发布: `vllm/v1/engine/coordinator.py:282`
- API Server 接收: `vllm/v1/engine/core_client.py:1248-1259`
- 路由决策: `vllm/v1/engine/core_client.py:1322-1350`

### 3.2 核心选择算法: Weighted Least-Loaded

```python
# vllm/v1/engine/core_client.py:1322-1345
def get_core_engine_for_request(self, request):
    current_counts = self.lb_engines    # [[waiting, running], ...]
    num_engines = len(current_counts)
    min_score = sys.maxsize
    eng_index = 0

    for i in range(num_engines):
        idx = (self.eng_start_index + i) % num_engines   # 轮转起点
        waiting, running = current_counts[idx]
        score = waiting * 4 + running                     # 加权评分
        if score < min_score:
            min_score = score
            eng_index = idx

    # 本地乐观更新，避免两次 coordinator 推送之间的雷群效应
    current_counts[eng_index][0] += self.client_count
    return self.core_engines[eng_index]
```

### 3.3 评分公式

```
score = waiting x 4 + running
```

- **waiting 权重 4x**: 排队请求比正在运行的请求重要得多。waiting 意味着 KV cache 已经吃紧或 batch 已满，新请求会等更久
- **running 权重 1x**: running 的请求虽然占用资源，但它们在正常消耗中，不代表拥堵
- 选 score **最小**的 engine

**举例**:
| Engine | waiting | running | score |
|--------|---------|---------|-------|
| 0 | 2 | 30 | 2x4+30 = **38** |
| 1 | 0 | 40 | 0x4+40 = **40** |
| 2 | 5 | 10 | 5x4+10 = **30** <-- 选这个 |
| 3 | 3 | 25 | 3x4+25 = **37** |

### 3.4 轮转起点 (eng_start_index)

```python
idx = (self.eng_start_index + i) % num_engines
```

当多个 engine 分数相同（比如都是 0）时，遍历顺序决定谁被选中。`eng_start_index` 让**不同的 API Server 进程从不同的 engine 开始扫描**，避免所有 API Server 在空载时全部打到 Engine 0。

### 3.5 本地乐观更新（防雷群）

```python
current_counts[eng_index][0] += self.client_count
```

Coordinator 的统计推送间隔是 100ms。在两次推送之间，如果多个请求到达，它们看到的是**同一份快照**，可能都选同一个 engine (thundering herd)。

解决办法: 每选中一个 engine，**本地立即把该 engine 的 waiting +N**（N = API Server 总数，即 `client_count`）。后续请求会自然分散到其他 engine。等下次 coordinator 推送到来，本地缓存会被真实值覆盖。

### 3.6 未来演进 TODO

```python
# TODO use P2C alg for larger DP sizes
```

当前是 O(N) 全扫描选最小。对于大规模 DP（64+ ranks），计划换成 **Power-of-2-Choices (P2C)** 算法 — 随机选 2 个 engine，取分数低的那个。P2C 在理论和实践中接近最优，且复杂度 O(1)。

---

## 4. Cache-Aware / Session-Aware 路由: 现状

### 4.1 结论: vLLM 当前没有

API Server 进程 (`DPLBAsyncMPClient`) 内**不维护任何 cache-aware 或 session-aware 的状态**。

API Server 持有的全部路由相关状态:

| 字段 | 类型 | 用途 |
|------|------|------|
| `lb_engines` | `list[list[int]]` | 每个 engine 的 `[waiting, running]`，Coordinator 每 100ms 推送 |
| `reqs_in_flight` | `dict[str, EngineIdentity]` | 跟踪请求在哪个 engine，**仅用于 abort 路由** |

不包含:
- 没有 prefix tree / radix tree
- 没有 per-engine cache 命中率统计
- 没有 prompt hash -> engine 的映射表
- 没有 session ID -> engine 的 sticky 绑定
- 没有 KV cache 利用率信息

文档原话 (`docs/serving/data_parallel_deployment.md`):
> Currently, the internal DP load balancing is done within the API server process(es) and is based on the running and waiting queues in each of the engines. This could be made more sophisticated in future by incorporating KV cache aware logic.

### 4.2 X-data-parallel-rank Header — 外部注入机制

API Server 提供了一个入口 (`vllm/entrypoints/openai/engine/serving.py:766-778`):

```python
def _get_data_parallel_rank(raw_request):
    rank_str = raw_request.headers.get("X-data-parallel-rank")
    return int(rank_str) if rank_str else None
```

外部 router 可以通过 HTTP header `X-data-parallel-rank: 2` **强制指定**请求发到哪个 engine。在 LB 选择时优先级最高:

```python
# core_client.py:1324 — 如果外部指定了 rank，直接跳过 LB 逻辑
if (eng_index := request.data_parallel_rank) is None and ...
    # 才进入 weighted least-loaded 选择
```

如果需要 cache-aware 路由，可以在外部 router 实现逻辑，然后通过这个 header 注入决策。

### 4.3 cache_salt — 单 Engine 内的 Cache 隔离

`EngineCoreRequest` 里有个 `cache_salt` 字段，但它的作用是在**单个 engine 内部**隔离 prefix cache（比如不同租户的 cache 不互串），和跨 engine 路由无关。

### 4.4 路由决策输入总结

```
vLLM API Server 路由决策的输入:

  [Y] 每个 engine 的 waiting / running 请求数  (Coordinator 推送)
  [Y] 外部注入的 X-data-parallel-rank header   (bypass LB)
  [N] 各 engine 的 prefix cache 状态
  [N] 各 engine 的 KV cache 利用率
  [N] prompt -> engine 的历史映射
  [N] session / conversation 亲和性
```

### 4.5 SGLang 的对比: 已实现 Cache-Aware

SGLang 的 sgl-model-gateway (Rust 实现) 提供了多种策略:

| 策略 | 描述 |
|------|------|
| `cache_aware` | 维护 radix prefix tree，跟踪每个 worker 缓存了哪些 prefix，将相同 prefix 的请求路由到同一 worker |
| `consistent_hashing` | hash ring 做 session affinity，`X-SMG-Routing-Key` header 控制 sticky |
| `prefix_hash` | 轻量版 cache-aware，consistent hash + bounded load，O(log n) |
| `power_of_two` | P2C 算法 |
| `round_robin` / `random` | 基础策略 |

### 4.6 如果需要 Cache-Aware 路由的选项

1. **等 vLLM 实现** cache-aware internal LB（目前只是 roadmap）
2. **External LB + 自己实现路由逻辑**（基于 prompt hash 的 consistent hashing + `X-data-parallel-rank` header 注入）
3. **参考 SGLang 的 sgl-model-gateway** 作为外部 router（已有成熟的 `cache_aware` 策略）

---

## 5. Issue #24461: 非 MoE 模型的 DP 性能优化

### 5.1 问题描述

> **[BugFix]: Avoid unnecessary coordination for non-MoE data parallel**
> 作者: Nick Hill (njhill) | 状态: CLOSED | 修复 PR: #30739

当启用 DP 时，vLLM 的实现**统一假设**所有模型都是 MoE 模型，因此对所有 DP rank 都执行了:

1. **不必要的跨 rank 同步** — 每次 forward pass 前共享 metadata 的 all-reduce 操作
2. **不必要的 dummy forward pass** — 空闲 rank 被强制执行空操作以保持同步
3. **不必要的 step/wave 协调** — DP Coordinator 强制所有 rank 步调一致

对于 **dense 模型**（如 Qwen3-8B、Llama 等），这些同步操作完全没有必要。

```
MoE 模型: DP rank 之间有依赖 -> 需要同步（合理）
Dense 模型: DP rank 之间完全独立 -> 同步是纯粹的开销（bug）
```

Nick Hill 同时指出: 即使对 dense 模型移除同步，**DP Coordinator 仍然应该保留运行**，因为它负责传播负载均衡统计数据。

### 5.2 修复方案 (PR #30739)

**"[BugFix] Support online dense model DP without overhead"** | 2026-01-02 merged

核心改动:
- 在 worker 层面将 dense 模型的 parallel config 等效设为 `DP=1`，使每个 rank **完全独立运行**
- 当使用 internal LB 时，DP Coordinator 仍然运行以传播负载均衡统计
- 但 **step/wave 同步逻辑被禁用** (`enable_wave_coordination=False`)
- 仅支持 online / AsyncLLM 场景
- 对 offline DP + dense 模型的组合，启动时直接 fail

涉及的关键文件:
- `vllm/v1/engine/coordinator.py` — Coordinator 逻辑解耦
- `vllm/v1/engine/core.py` — engine core 独立运行逻辑
- `vllm/config/parallel.py`, `vllm/config/vllm.py` — 配置层判断是否需要 MoE 协调
- `vllm/distributed/parallel_state.py` — parallel state 调整
- `vllm/v1/worker/gpu_worker.py`, `gpu/model_runner.py` — worker 层面去除不必要同步

### 5.3 性能提升 (4xH100, Qwen3-8B, DP=4)

| 指标 | Before | After | 提升 |
|------|--------|-------|------|
| 请求吞吐 (req/s) | 38.31 | **40.31** | +5.2% |
| 输出 token 吞吐 (tok/s) | 19,615 | **20,637** | +5.2% |
| Mean TTFT (ms) | 131.78 | **88.94** | **-32.5%** |
| Median TTFT (ms) | 124.31 | **74.67** | **-39.9%** |
| Mean TPOT (ms) | 9.93 | **9.50** | -4.3% |
| P99 ITL (ms) | 15.93 | **12.86** | -19.3% |

最显著的改善在 **TTFT（首 token 延迟）**，降低约 30-40%，因为移除了每次 forward pass 前的同步等待。

---

## 6. 关键代码索引

| 组件 | 文件 | 关键行 |
|------|------|--------|
| DP Coordinator 进程 | `vllm/v1/engine/coordinator.py` | `DPCoordinatorProc` L151 |
| Stats 发布 | `vllm/v1/engine/coordinator.py` | L282 |
| Stats 接收 | `vllm/v1/engine/core_client.py` | L1248-1259 |
| LB 路由决策 | `vllm/v1/engine/core_client.py` | `get_core_engine_for_request` L1322 |
| X-data-parallel-rank 解析 | `vllm/entrypoints/openai/engine/serving.py` | `_get_data_parallel_rank` L766 |
| EngineCoreRequest 定义 | `vllm/v1/engine/__init__.py` | L72-94 |
| SchedulerStats | `vllm/v1/metrics/stats.py` | L171-198 |
| Internal LB 测试 | `tests/v1/distributed/test_internal_lb_dp.py` | |
| External LB 测试 | `tests/v1/distributed/test_external_lb_dp.py` | |
