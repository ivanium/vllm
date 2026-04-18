# Mooncake Store Connector 深度解析

围绕 vLLM 中 Mooncake store connector 的架构设计、核心机制和底层原理的专题整理。

---

## 目录

- [第一部分：架构设计原则](#第一部分架构设计原则)
  - [1.1 控制面 vs 数据面的分离](#11-控制面-vs-数据面的分离)
  - [1.2 mooncake_store_worker.py 的数据面架构](#12-mooncake_store_workerpy-的数据面架构)
- [第二部分：Scheduler 侧（控制面）](#第二部分scheduler-侧控制面)
  - [2.1 新 scheduler 的 schedule() 机制](#21-新-scheduler-的-schedule-机制)
  - [2.2 RequestTracker 为什么要独立抽象](#22-requesttracker-为什么要独立抽象)
  - [2.3 两阶段提交在 lookup/alloc 中的应用](#23-两阶段提交在-lookupalloc-中的应用)
  - [2.4 request_finished 的延迟释放机制](#24-request_finished-的延迟释放机制)
- [第三部分：Worker 侧（数据面）](#第三部分worker-侧数据面)
  - [3.1 背压控制机理](#31-背压控制机理)
  - [3.2 Offload 分批与 Mooncake eviction 粒度](#32-offload-分批与-mooncake-eviction-粒度)
- [第四部分：可观测性](#第四部分可观测性)
  - [4.1 KV Events 链与外部交互](#41-kv-events-链与外部交互)

---

# 第一部分：架构设计原则

## 1.1 控制面 vs 数据面的分离

### 核心定义

- **控制面**：决定 "要不要做、做什么、给谁做"——产出是 **决策**
- **数据面**：执行 "怎么做、搬多少字节、搬到哪个地址"——产出是 **数据搬运**

### 在 Mooncake Store Connector 中的映射

```
控制面 (mooncake_store_scheduler.py)    数据面 (mooncake_store_worker.py)
─────────────────────────────           ─────────────────────────────
运行在: scheduler 进程                   运行在: worker 进程
持有:   无 store 句柄，无 GPU 访问        持有:   MooncakeDistributedStore + GPU 显存

做什么:                                  做什么:
  "req_A store 命中 768 token"            store.batch_get(keys, gpu_addrs)
  "本地已有 256，需额外加载 512"           store.batch_put(keys, gpu_addrs)
  "block 分配成功，can_load=True"          store.register_buffer(gpu_ptr)
  "本轮 req_B 有新完整 chunk 可保存"       store.batch_is_exist(keys)

产出: MooncakeStoreConnectorMetadata     产出: GPU 上 KV 数据就位/写入 store
      (工作单)                                  done_sending / done_recving
```

### 设计时的三个判断标准

**标准 1：这个操作需要什么资源？**

| 操作 | 需要的资源 | 放哪 |
|------|-----------|------|
| "这个 request 命中了多少" | store 查询 | 数据面 |
| "命中数 > 本地已算数？" | 两个整数比较 | 控制面 |
| "分配 block 槽位" | KV cache manager | 控制面 |
| "把 KV 从 store 搬到 GPU" | store 句柄 + GPU 地址 | 数据面 |
| "这轮哪些 request 要 save" | scheduler output 信息 | 控制面 |
| "按 hash 去重，跳过已存在" | store.batch_is_exist | 数据面 |

**规则：需要 I/O 资源（store 句柄、GPU 显存、网络）的放数据面，只需要逻辑判断和状态的放控制面。**

但这里有个微妙点：`lookup` 操作需要 store 句柄（数据面资源），但结果是给 scheduler 做决策用的（控制面需求）。怎么办？

```
Scheduler (控制面)              Worker (数据面)
     │                              │
     │  "帮我查一下命中了多少"         │
     │────── ZMQ IPC ──────────────→│
     │                              │ store.batch_is_exist()
     │                              │ (真正访问 store)
     │←───── "768" ─────────────────│
     │                              │
     │  拿着 768 做调度决策            │
```

**Lookup 的执行在数据面，但结果回传控制面做决策。用 IPC 做桥梁。**

**标准 2：这个操作的调用频率和延迟容忍度是什么？**

| 操作 | 频率 | 延迟容忍 | 设计决策 |
|------|------|----------|----------|
| 调度决策 | 每 step 同步 | 微秒级 | 控制面必须快，不能等 I/O |
| batch_put/get | 每 step 异步 | 毫秒级可接受 | 数据面可以后台线程跑 |
| prefix lookup | 每新请求一次 | 亚毫秒级 | 虽是数据面操作但走同步 IPC |

这直接决定了数据面为什么要用后台线程 + 队列：

```
如果数据面是同步的：
  scheduler.schedule()
    → get_num_new_matched_tokens()   (控制面，快)
    → update_state_after_alloc()     (控制面，快)
    → build_connector_meta()         (控制面，快)
    → worker.batch_get()             (数据面，要等 100ms!)
    → worker.batch_put()             (数据面，又要等 100ms!)
    整个调度循环被 I/O 阻塞

实际设计（异步数据面）：
  scheduler.schedule()               (控制面，全部微秒级完成)
    → 产出 metadata（工作单）

  worker.get_finished()              (把工作单丢进后台队列)
    → kv_recv_thread.add_request()   (入队就返回)
    → kv_send_thread.add_request()   (入队就返回)
    → 收集上一轮的完成结果

  GPU: [====== forward ======]       (和 store I/O 并行)
  Store: [=== batch_get ===]         (后台线程在跑)
```

**标准 3：这个状态属于谁，生命周期跟谁走？**

| 状态 | 生命周期 | 放哪 |
|------|----------|------|
| `LoadSpec` (命中多少，能否加载) | 单轮 step | 控制面 |
| `RequestTracker` (累积 token) | 跨 step，preempt 销毁 | 控制面 |
| store 句柄/连接 | 进程级 | 数据面 |
| GPU base address / block_len | 进程级 | 数据面 |
| `stored_requests` 引用计数 | 跨 step | 数据面 |
| 背压状态 (`_store_pressure`) | 跨 step | 数据面 |

### 设计步骤总结

| 规则 | 怎么判断 | Mooncake 中的体现 |
|------|---------|------------------|
| **按资源归属分** | 需要 I/O/硬件资源 → 数据面；纯逻辑 → 控制面 | store 句柄/GPU 地址全在 worker；调度决策全在 scheduler |
| **按延迟要求分** | 快路径 → 控制面同步；慢路径 → 数据面异步 | scheduler 全是微秒级；worker 用后台线程 + 队列 |
| **接口最小化** | 两面之间只传"做什么"（工作单）和"做完了吗"（通知） | Metadata 只描述 req_id + token 范围；不包含 GPU 地址、store key |

最后一条最重要：**Metadata 里没有 GPU 地址，也没有 store key 字符串**。控制面只说"req_A 需要加载前 768 个 token"，数据面自己用 `ChunkedTokenDatabase.process_tokens()` + `prepare_value()` 算出具体的 key 和地址。

---

## 1.2 mooncake_store_worker.py 的数据面架构

### 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        Worker 进程                               │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              MooncakeStoreWorker (总控)                   │   │
│  │                                                          │   │
│  │  ┌───────────────────┐  ┌────────────────────────────┐   │   │
│  │  │MooncakeDistributed│  │ ChunkedTokenDatabase       │   │   │
│  │  │Store (C++ 库)      │  │ (逻辑 chunk → GPU 地址映射) │   │   │
│  │  └────────┬──────────┘  └────────────────────────────┘   │   │
│  │           │                                              │   │
│  │  ┌────────┴──────────────────────────────────────┐       │   │
│  │  │            GPU KV Cache Memory                 │       │   │
│  │  │  register_buffer() 注册后 store 才能 DMA 读写  │       │   │
│  │  └────────────────────────────────────────────────┘       │   │
│  └──────────┬─────────────────────────┬───────────────────┬──┘   │
│             │                         │                   │      │
│    ┌────────▼────────┐       ┌────────▼─────────┐         │      │
│    │ SendingThread   │       │ RecvingThread    │         │      │
│    │ (后台 put)       │       │ (后台 get)        │         │      │
│    │ - hash 去重      │       │ - offload 分批    │         │      │
│    │ - TP stride 跳过 │       │ - TP rotate 负载 │         │      │
│    │ - CUDA event 同步│       │ - 失败检测        │         │      │
│    │ - 背压控制       │       │                  │         │      │
│    └─────────────────┘       └──────────────────┘         │      │
│                                                           │      │
│                                           ┌───────────────▼──┐   │
│                                           │ LookupKeyServer  │   │
│                                           │ (prefix 查询服务) │   │
│                                           └──────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
                      ↑
              ZMQ IPC: Scheduler 进程查询
```

### 关键设计 1: TP stride 去重写入

当模型的 KV head 数量 **少于** TP 并行度时（比如 GQA 模型，4 个 KV head 但 TP=8），多个 TP rank 会持有完全相同的 KV 数据。如果每个 rank 都写 store，就是重复写。

```python
if self.num_kv_head < self.tp_size:
    self.put_step = self.tp_size // self.num_kv_head
    self.head_or_tp_rank = self.tp_rank // self.put_step
```

```
TP=8, KV heads=4 → put_step=2

TP rank:    0  1  2  3  4  5  6  7
KV head:    0  0  1  1  2  2  3  3
是否写:     Y  N  Y  N  Y  N  Y  N

rank 0 写 head 0 的数据，rank 1 跳过（和 rank 0 一样）
rank 2 写 head 1 的数据，rank 3 跳过
```

### 关键设计 2: GPU 内存布局探测与注册

不同 attention backend 的 KV cache tensor 布局不同。Worker 必须自动探测布局：

```
FlashAttention / ROCm:
  tensor shape: (2, num_blocks, num_heads, head_size)
                 ↑
                K/V 维度在最外层 → 两个 segment

FlashInfer / MLA:
  tensor shape: (num_blocks, 2, num_heads, head_size)
  或:           (num_blocks, latent_dim)
                 ↑
                block 维度在最外层 → 一个 segment
```

探测方法：

```python
page_size_bytes = storage.nbytes() // self.num_blocks
outer_dims = [
    d for d in range(first_kv_cache.ndim)
    if first_kv_cache.stride(d) * el > page_size_bytes
]
```

如果某个维度的 stride（字节）大于单个 block 的大小，说明这个维度跨过了整个 block 页面——它一定是外层 segment 维度。这保证了后续 `prepare_value()` 里 `base_addr + block_id * block_len` 的地址算术对任何 backend 都正确。

### 关键设计 3: SendingThread 的三层过滤

```
ReqMeta 进入
    │
    ▼
┌─ 过滤 1: TP stride ─────────────────────────────┐
│  starts[tp_rank % put_step :: put_step]          │
│  多个 rank 共享同一 KV head 时，只让部分 rank 写  │
│  例: put_step=2, rank 1 跳过全部 key              │
└────────────────────────┬─────────────────────────┘
                         │
                         ▼
┌─ 过滤 2: Hash 去重 ─────────────────────────────┐
│  exists_states = store.batch_is_exist(keys)     │
│  已存在的 block 不重复写                          │
│  例: 10 个 chunk，store 里已有 7 个 → 只写 3 个    │
└────────────────────────┬─────────────────────────┘
                         │
                         ▼
┌─ 过滤 3: 背压控制 ──────────────────────────────┐
│  if _should_skip_request(req_id):                │
│    跳过写入                                       │
│                                                  │
│  触发: batch_put 返回 NO_AVAILABLE_HANDLE        │
│  恢复: 后续某次 batch_put 成功                    │
└────────────────────────┬─────────────────────────┘
                         │
                         ▼
              CUDA event synchronize
              store.batch_put_from_multi_buffers()
```

### 关键设计 4: RecvingThread 的 TP 轮转负载均衡

```python
key_list_c = (
    key_list[self.tp_rank % len(key_list) :]
    + key_list[: self.tp_rank % len(key_list)]
)
```

同一 request 的多个 chunk 要从 store 读取。如果所有 TP rank 都按同样顺序请求，它们会同时争抢 chunk_0，造成热点：

```
轮转后:
rank 0: [chunk_0, chunk_1, chunk_2, chunk_3]
rank 1: [chunk_1, chunk_2, chunk_3, chunk_0]
rank 2: [chunk_2, chunk_3, chunk_0, chunk_1]
rank 3: [chunk_3, chunk_0, chunk_1, chunk_2]
```

每个时刻不同 rank 在读不同 chunk，分散了 store 的并发压力。

### 关键设计 5: Lookup 全 TP/PP shard 一致性检查

Prefix 命中不能只看当前 rank 自己，必须所有 TP/PP rank 同时有才算命中：

```python
# 展开所有 TP rank 的 key
for i in range(1, min(self.tp_size, self.num_kv_head)):
    for item in keys:
        new_str = item.replace("@tp_rank:0", f"@tp_rank:{i}", 1)
        multi_tp_keys.append(new_str)

# 一次 batch_is_exist 查询所有
res = self.store.batch_is_exist(multi_tp_keys)
```

找最短木板：

```
chunk:      0   1   2   3   4   5
tp_rank 0:  Y   Y   Y   Y   N   N   (前 4 个命中)
tp_rank 1:  Y   Y   Y   N   N   N   (前 3 个命中)
pp_rank 1:  Y   Y   Y   Y   Y   N   (前 5 个命中)
                        ^
                 最小首次缺失 = chunk 3
                 → 只有前 3 个 chunk 全部 shard 都有
                 → 返回 starts[3] 作为命中长度
```

**任何一个 shard 缺失某个 chunk，这个 chunk 之后的所有 prefix 都不能用。** 因为加载时需要所有 shard 的数据凑齐才能做正确计算。

### 关键设计 6: get_finished 的 I/O 与计算重叠

```
时间轴 →

GPU compute:  [====== step N 的 forward ======][=== step N+1 ===]
                                               ↑
                                        get_finished() 在这里调用
                                               │
Store I/O:                  [== load job 入队 ==][= 后台执行 =]
                            [== save job 入队 ==][= 后台执行 =]
                                                   ↑
                                            和 step N+1 的
                                            GPU 计算重叠
```

`start_load_kv()` 和 `wait_for_save()` 都是空操作。所有 I/O 都集中在 `get_finished()` 里发起，因为此时 GPU 已经开始跑下一轮 forward，store I/O 可以和 GPU 计算并行。

---

# 第二部分：Scheduler 侧（控制面）

## 2.1 新 scheduler 的 schedule() 机制

### 旧 scheduler（v0，~2024 年）

旧 scheduler 有显式的 prefill 阶段和 decode 阶段：

```
旧 scheduler 的一轮调度：

Step 1: 有 WAITING 请求吗？
        ├─ 有 → 组一个 prefill batch，所有 decode 请求本轮不调度
        └─ 没有 → 组一个 decode batch

核心规则：prefill 和 decode 互斥，prefill 优先
```

问题：
- 一个 8K token 的 prompt 做 prefill 时，所有正在 decode 的请求都要停下来等
- GPU 利用率低：prefill 时 decode 空闲
- 延迟不可预测：decode 的 TPOT 会突然飙高

### 新 scheduler（v1）

代码开头的注释就是宣言：

```
There's no "decoding phase" nor "prefill phase" in the scheduler.
Each request just has num_computed_tokens and num_tokens_with_spec.
At each step, the scheduler tries to assign tokens to the requests
so that each request's num_computed_tokens can catch up its
num_tokens_with_spec.
```

**不再区分 prefill 和 decode。** 每个 request 只有一个问题："还差多少 token 没算？"

### schedule() 完整流程

```
schedule()
│
├── 全局资源: token_budget = max_num_scheduled_tokens (比如 8192)
│
├── 【阶段 1】调度 RUNNING 队列（decode + chunked prefill 的后续 chunk）
│   │
│   │  遍历 self.running 中每个 request:
│   │
│   │  ┌─ 计算 num_new_tokens = num_tokens_with_spec - num_computed_tokens
│   │  │   (decode 通常 = 1，chunked prefill 后续可能 > 1)
│   │  │
│   │  ├─ num_new_tokens = min(num_new_tokens, token_budget)
│   │  │
│   │  ├─ long_prefill_token_threshold 截断
│   │  │   (防止单个 chunked prefill 吃掉全部 budget)
│   │  │
│   │  ├─ allocate_slots → 成功？
│   │  │   ├─ 成功 → token_budget -= num_new_tokens
│   │  │   └─ 失败 → 抢占最低优先级 request，腾出 block，重试
│   │  │            └─ 抢占到自己都不够 → break
│   │  │
│   │  └─ 加入 scheduled_running_reqs
│   │
│   │  关键特征:
│   │  - decode 请求每个只消耗 1 个 token 的 budget
│   │  - chunked prefill 的后续 chunk 也在这里被调度
│   │  - 不会因为"有 prefill"就跳过 decode
│
├── 【阶段 2】调度 WAITING 队列（新 prefill）
│   │
│   │  前置条件:
│   │  - 阶段 1 没有发生抢占
│   │  - 还有剩余 token_budget
│   │  - running 数量 < max_num_running_reqs
│   │
│   │  遍历 waiting + skipped_waiting 中每个 request:
│   │
│   │  ┌─ 本地 prefix cache 查找 → num_local_computed
│   │  ├─ 外部 store 查找 → num_external_computed
│   │  ├─ num_new_tokens = num_tokens - (local + external)
│   │  │
│   │  ├─ chunked prefill 截断
│   │  │
│   │  ├─ allocate_slots → 成功？
│   │  │   ├─ 成功 + 同步加载 → 加入 running + scheduled_new_reqs
│   │  │   ├─ 成功 + 异步加载 → WAITING_FOR_REMOTE_KVS, 不进 running
│   │  │   └─ 失败 → break（不抢占）
│   │  │
│   │  └─ token_budget -= num_new_tokens
│   │
│
├── 【构建输出】
│   │
│   └─ SchedulerOutput {
│        scheduled_new_reqs:    阶段 2 同步路径的 request
│        scheduled_cached_reqs: 阶段 1 的全部 + 阶段 2 恢复的
│        num_scheduled_tokens:  每个 request 本轮算多少 token
│      }
│
└── 发给 worker 执行 forward pass
```

### 五个关键设计点

**1. 统一 token budget——prefill 和 decode 混合调度**

```
token_budget = 8192

阶段 1 (running 队列):
  req_A (decode):           消耗 1    → 剩余 8191
  req_B (decode):           消耗 1    → 剩余 8190
  req_C (chunked prefill):  消耗 2048 → 剩余 6142
  req_D (decode):           消耗 1    → 剩余 6141

阶段 2 (waiting 队列):
  req_E (新 prefill, 4096 token, 本地命中 2048):
                            消耗 2048 → 剩余 4093
  req_F (新 prefill, 6000 token):
                            消耗 4093 → 剩余 0, break
```

**decode 和 prefill 在同一个 batch 里共存。**

**2. Running 优先——decode 延迟有保障**

阶段 1 先处理 running 队列（主要是 decode），阶段 2 再处理 waiting 队列。对比旧架构：旧架构是 prefill 优先、decode 被阻塞；新架构反过来，**decode 优先、prefill 用剩余容量**。

**3. Chunked prefill——不存在"一个 prefill 独占全部 GPU"**

一个 8000 token 的 prompt 不再一次性占满 GPU：

```
Step 0: req_E prefill chunk 1 (token 0-2047)    + req_A/B/D decode
Step 1: req_E prefill chunk 2 (token 2048-4095) + req_A/B/D decode
Step 2: req_E prefill chunk 3 (token 4096-6143) + req_A/B/D decode
Step 3: req_E prefill chunk 4 (token 6144-7999) + req_A/B/D decode
```

**每个 step 里 decode 请求都能正常推进。**

**4. 没有"prefill/decode"的概念——只有 num_new_tokens**

```python
num_new_tokens = request.num_tokens_with_spec - request.num_computed_tokens
```

对所有 request 一视同仁：

| 场景 | num_tokens_with_spec | num_computed_tokens | num_new_tokens |
|------|---------------------|---------------------|----------------|
| 新 prefill, 4096 token | 4096 | 0 | 4096 |
| chunked prefill 第二轮 | 4096 | 2048 | 2048 |
| 普通 decode | 101 | 100 | 1 |
| speculative decode (3 draft) | 104 | 101 | 3 |
| prefix cache 命中 3000 | 4096 | 3000 | 1096 |

**5. 抢占策略——只在阶段 1 发生**

- **Running 请求已占用 GPU 资源，有权抢占其他 running**（否则会死锁）
- **Waiting 请求还没开始，没资格抢占**（只能等有空位再进来）

### 新旧对比

| 维度 | 旧 | 新 |
|------|----|----|
| prefill vs decode | 互斥，prefill 优先 | 混合，decode 先调度 |
| 大 prompt | 独占整个 step | chunked，每轮最多 threshold 个 token |
| decode 延迟 | 遇到 prefill 就停 | 几乎不受 prefill 影响 |
| 调度粒度 | request 级别 | token 级别（统一 budget） |
| 概念模型 | "这个 request 是在 prefill 还是 decode" | "这个 request 还差多少 token" |

### 什么是 scheduled_new_reqs / scheduled_cached_reqs

它们和 `WAITING` / `RUNNING` 是不同维度的东西：

- **`WAITING` / `RUNNING`** 是 request 的持久状态
- **`scheduled_new_reqs` / `scheduled_cached_reqs`** 是本轮 step 的调度输出分类

```
scheduled_new_reqs:     本轮第一次被送上 GPU 的 request
                        worker 需要为它们创建全新的运行时状态
scheduled_cached_reqs:  已经在 GPU 上的 request（上一轮就在 running）
                        worker 只需要增量更新
两个都不在:              本轮不参与 GPU forward
                        (WAITING_FOR_REMOTE_KVS 的 request 就在这里)
```

---

## 2.2 RequestTracker 为什么要独立抽象

### 问题：为什么不直接放在 Request 里

`Request` 是 vLLM 核心对象，`RequestTracker` 是 mooncake store connector 私有对象。为什么要拆开？

### 原因 1：生命周期不同

```
Request 的生命周期:
  创建 ──────────────── preempt ──── resume ──────── 结束
  │                       │            │              │
  同一个 Python 对象，始终存在，字段被修改但对象不销毁

RequestTracker 的生命周期:
  创建 ──── 累积状态 ──── 销毁    重新创建 ──── 累积 ── 销毁
  │                       │      │                      │
  preempt 时 pop 掉              resume 时 new 一个全新的
  因为旧 block_ids 已失效         num_saved_tokens 从 0 开始
```

如果把 `num_saved_tokens` 和 `allocated_block_ids` 放在 `Request` 上，preempt 时需要手动清零这些字段。但 `Request` 上没有"你被 preempt 了，请重置 store 相关字段"这个钩子。

用独立对象就很干净——pop 掉整个 tracker，所有 store 状态一次性消失。

### 原因 2：Request 是全局共享的，RequestTracker 是 connector 私有的

`Request` 被 scheduler、worker、engine、API server 等十几个模块引用。vLLM 现在有十几种 connector（nixl、moriio、lmcache、offloading、p2p_nccl、mooncake、mooncake_store...）。

如果每种 connector 都往 `Request` 上加字段：

```python
class Request:
    # 核心字段
    request_id: str
    num_computed_tokens: int
    ...
    # mooncake store 专用
    mooncake_num_saved_tokens: int
    mooncake_allocated_block_ids: list[int]
    # nixl 专用
    nixl_transfer_handle: ...
    # moriio 专用
    moriio_state: ...
```

`Request` 会变成所有 connector 实现细节的垃圾场。**RequestTracker 让 mooncake store connector 的状态完全封闭在自己的文件内。**

### 原因 3：一份 Request 对应两个不同视角的 token 进度

```
Request.num_computed_tokens:
  "这个 request 的 KV 在 GPU 上准备好了多少"
  由 scheduler 主循环管理
  在 async load 完成前就已经被设置
  是面向 调度决策 的

RequestTracker.token_len:
  "这个 request 在 store connector 视角下累积到了多少 token"
  由 build_connector_meta 管理
  每轮 step 增量更新
  是面向 store I/O 决策 的
```

### 总结

**Request 描述的是"这个请求在 vLLM 系统中的状态"，RequestTracker 描述的是"这个请求在 mooncake store 视角下的 I/O 进度"。** 两者是不同关注点的不同抽象。

---

## 2.3 两阶段提交在 lookup/alloc 中的应用

### 两阶段提交在防什么

**两个资源池各管各的，任何一个说"有"都不算数，必须两个同时说"有"才能真干活。**

```
资源池 A: Mooncake Store (外部，可能在远端内存/NVMe 上)
资源池 B: 本地 GPU KV cache block 槽位
```

- Store 说"我有前 768 token 的 KV" → 但本地没有空 block 可以接收，没用
- 本地分配了 block → 但 store 里并没有命中，也没用

### 两阶段流程

```
阶段 1 (intent):
    scheduler 调用 lookup()
    store 回答 "命中 768 token"
    scheduler 记 LoadSpec(can_load=False)
    ↑ 此时只是记了一个"意向"，什么都没承诺

中间: vLLM 正常走 block 分配流程

阶段 2 (confirm):
    看本地 block 分配结果
    分到了 → can_load = True   两边都有，可以干
    没分到 → can_load = False  放弃，当这轮没命中
```

### 完整时序图

```
┌─────────────────────────────────────────────────────────────────┐
│                     Scheduler 调度循环                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  (1) get_num_new_matched_tokens()                               │
│     ┌──────────┐    ZMQ IPC     ┌──────────┐                    │
│     │Scheduler │ ──────────────→│  Worker   │                    │
│     │          │ "前768个token   │ (lookup)  │                    │
│     │          │←──命中"────────│           │                    │
│     └────┬─────┘                └──────────┘                    │
│          │                                                      │
│          │  LoadSpec { can_load: false }  ← 记录意图，不做承诺    │
│          │  return (512, async)           ← 告诉调度器要分配多少   │
│          ▼                                                      │
│  (2) vLLM KVCacheManager.allocate_blocks()                      │
│          │                                                      │
│          │ 可能成功（分到 block）                                  │
│          │ 可能失败（显存不足，分到 0 个 block）                    │
│          ▼                                                      │
│  (3) update_state_after_alloc()                                 │
│          │                                                      │
│          ├─ num_external_tokens > 0 → can_load = true           │
│          │                                                      │
│          └─ num_external_tokens == 0 → can_load = false         │
│             (store 有数据但本地没地方放，放弃本轮加载)              │
│          │                                                      │
│          ▼                                                      │
│  (4) build_connector_meta()                                     │
│     只有 can_load=true 的 LoadSpec 才会传给 worker 执行           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 关键代码

```python
# 阶段 1: get_num_new_matched_tokens — 只做查询，记录意图
self.load_specs[request.request_id] = LoadSpec(
    vllm_cached_tokens=num_computed_tokens,
    kvpool_cached_tokens=num_external_hit_tokens,
    can_load=False,                    # 初始"未确认"
)

# 阶段 2: update_state_after_alloc — block 分配成功后才确认
self.load_specs[request.request_id].can_load = True   # 翻转为"已确认"
```

### 边界情况：intent 时 store 有，confirm 时 store 里数据被 evict 了

这个场景 scheduler 的两阶段提交**防不住**。两阶段提交解决的是"本地 GPU block 分配失败"的问题，不是"store 数据消失"的问题。store 的 eviction 发生在 scheduler 的控制范围之外。

实际的防线在 **worker 侧 + scheduler 的失败恢复机制**，分三层：

```
Scheduler                    Worker                     Mooncake Store
    │                           │                            │
    │ (1) lookup("有多少?")      │                            │
    │──────── ZMQ ─────────────→│                            │
    │                           │── query ──────────────────→│
    │                           │←── "768 token 命中" ───────│
    │←──── "768" ──────────────│                            │
    │                           │                            │
    │ (2) LoadSpec(can_load=F)  │                            │
    │ (3) 分配 block → 成功     │                            │
    │ (4) can_load = True       │                            │
    │                           │                            │
    │ (5) build_meta+get_finished                            │
    │──── ReqMeta(load) ───────→│                            │
    │                           │                            │
    │                           │   ┌─── 竞态窗口 ────┐      │
    │                           │   │ store evict 了  │      │
    │                           │   │ 这批 KV 数据    │      │
    │                           │   └─────────────────┘      │
    │                           │                            │
    │                           │── batch_get ──────────────→│
    │                           │←── [-1, -1, ...] 失败 ────│
    │                           │                            │
    │                           │ (6) 标记 invalid_block_ids │
    │                           │     set_finished_request   │
    │                           │                            │
    │←── KVConnectorOutput ────│                            │
    │    finished_recving={req} │                            │
    │    invalid_blocks={b1,b2} │                            │
    │                           │                            │
    │ (7) _handle_invalid_blocks│                            │
    │   截断 num_computed_tokens│                            │
    │                           │                            │
    │ (8) _update_waiting_for_remote_kv                      │
    │   保留脏 block 之前的有效部分                             │
    │   释放脏 block                                          │
    │   request → WAITING                                    │
    │                           │                            │
    │ (9) 下一轮正常调度         │                            │
    │   从截断点开始 prefill      │                            │
```

### 失败处理总结

| 失败类型 | 防线在哪 | 处理方式 |
|---------|---------|---------|
| 本地 GPU 显存不够 | scheduler 的两阶段提交（`can_load`） | 直接放弃 load，`can_load=False` |
| Store 数据被 evict / 读取失败 | worker 检测 + scheduler `_handle_invalid_blocks` | 截断 `num_computed_tokens`，回退重算 |

**两阶段提交只管"本地有没有地方放"这一个问题。** store 侧的数据一致性不在它的职责范围内——那是 worker 的 `batch_get` 返回值 + scheduler 的失败恢复路径兜底。

---

## 2.4 request_finished 的延迟释放机制

### 核心代码

```python
def request_finished(self, request, block_ids):
    if self.kv_role == "kv_consumer":
        return False, None                    # consumer 从不写回，直接释放

    tracker = self._request_trackers.get(request.request_id)
    if tracker is not None and tracker.num_saved_tokens <= 0:
        return False, None                    # 从未保存过，不延迟

    delay_free_blocks = len(block_ids) > 0    # 有 block 且正在异步写回 → 延迟释放
    return delay_free_blocks, None
```

### 为什么需要延迟释放

```
Worker 线程:   ───── async put(block_data) to store ─────────→ done
                     ↑                                         ↑
Scheduler:     request_finished()                         block 才能安全释放
                     │
                     └─ 如果立即释放 block，worker 正在读取的
                        GPU 内存可能已经被新请求覆盖
                        → 写入 store 的数据是脏的
```

`delay_free_blocks = True` 告诉 vLLM 的 block manager：**不要立即回收这些 block**，等 worker 的异步 store 写入完成后再释放。

### 作用范围：per-request

这个方法是 **per-request** 调用的。每个 request 结束时，scheduler 单独调用一次 `request_finished(request_A, block_ids_of_A)`。

**不会出现"一次 load 涵盖多个 request 的 block"的情况。** 每个 request 的 load 和 save 都是独立的：

```
Request A: prompt=[The cat sat ...]
  → tracker_A, block_ids=[b0, b1, b2]
  → 异步 batch_put(keys_A, addrs_from_b0_b1_b2)

Request B: prompt=[The dog ran ...]
  → tracker_B, block_ids=[b5, b6, b7]
  → 异步 batch_put(keys_B, addrs_from_b5_b6_b7)
```

当 A 结束时：`request_finished(A, [b0, b1, b2])` → A 的 batch_put 可能还在异步执行 → `delay=True`

### 共享 block 怎么办

vLLM 的 prefix caching 支持 block 共享。如果 A 和 B 有相同前缀，它们可能共享 b0。但共享 block 的生命周期管理不在这里——**block manager 有引用计数，只有当最后一个引用者释放时 block 才真正回收。** `request_finished` 里的 `block_ids` 参数已经是 block manager 判断过的"这个 request 结束后可以考虑释放的 block"，共享 block 不会出现在这个列表里。

### Worker 侧配合：引用计数释放

```
stored_requests[req_id] = 挂起的 store job 数量

场景：request 结束了，但后台还有 2 个 store job 在跑
  stored_requests["req_A"] = 2
  finished_req_ids = {"req_A"}

  本轮：req_A 已 finished，但计数 ≠ 0 → 放入待完成集合
    self.finished_store_req.add("req_A")

  ...后台线程每完成一个 job 就 dec_stored_request...

  未来某轮：计数归零
    stored_requests["req_A"] = 0 and "req_A" in finished_store_req
    → 终于返回给 scheduler → scheduler 释放 block
```

**block 的释放不是在 request 结束时，而是在所有异步 store job 完成时。**

---

# 第三部分：Worker 侧（数据面）

## 3.1 背压控制机理

### 两种不同的资源

```
资源 1: Store 存储空间 (数据最终存放的地方)
  - 远端内存 / NVMe / 分布式存储
  - 容量大，可以 evict 旧数据腾空间
  - "旧的出去新的进来"就是这一层

资源 2: Transfer staging buffer (数据搬运过程中的临时中转区)
  - 本地 CPU 内存中的一小块 buffer
  - 容量小且固定（默认 1280MB）
  - 不能 evict，因为正在被使用中
  - NO_AVAILABLE_HANDLE 报的就是这一层
```

类比：

```
Store 存储空间  = 仓库（很大，可以扔掉旧货腾地方）
Staging buffer = 仓库门口的卸货台（很小，同时只能停几辆卡车）

问题不是仓库满了，而是卸货台满了——
所有卸货台都被正在卸货的卡车占着，
新卡车到了只能在门口等。
```

### 数据搬运的实际路径

```
GPU 显存                  CPU staging buffer              Store (远端)
┌──────────┐             ┌──────────────┐               ┌──────────┐
│ KV block │ ── DMA ────→│  临时 copy   │ ── RDMA/TCP ─→│  持久化   │
│ (源数据)  │             │  (中转站)    │               │  (目标)   │
└──────────┘             └──────────────┘               └──────────┘
                              ↑
                     这块内存是有限的
                     默认 1280MB
                     一次 put 占一份
                     put 完成后才释放
```

Staging buffer 存的是 KV 数据的完整拷贝。一个 chunk 的 KV 数据有多大，staging buffer 就要占多大。

### NO_AVAILABLE_HANDLE 触发时机

```
时间线：

t0: batch_put(req_A, 10 chunks)   → 占用 staging buffer 约 400MB
    正在通过 RDMA 传输中...

t1: batch_put(req_B, 8 chunks)    → 占用 staging buffer 约 320MB
    正在通过 RDMA 传输中...

t2: batch_put(req_C, 15 chunks)   → 占用 staging buffer 约 600MB
    累计: 400 + 320 + 600 = 1320MB > 1280MB 预算
    → Mooncake 返回 NO_AVAILABLE_HANDLE (-200)
    → "我没有足够的 transfer handle 给你用了"

t3: req_A 的传输完成 → 释放 400MB staging buffer
    此时新的 put 才能成功
```

**Store 有无限空间没用——卸货台就这么大。**

### 为什么 eviction 帮不上忙

```
数据流: GPU → [staging buffer] → 网络 → [store 存储]
                    ↑                        ↑
              瓶颈在这里               这里可以 evict
              (传输资源耗尽)            (但数据还没到这里)
```

Eviction 解决的是"store 存储层满了"的问题。但 `NO_AVAILABLE_HANDLE` 发生时，数据**还没离开本机 CPU**，根本没走到 store 存储层。

### 背压控制状态机

```
              batch_put 返回
              NO_AVAILABLE_HANDLE
正常 ──────────────────────────────→ 压力状态
 ↑                                      │
 │                                      │ 后续同一 request 的
 │                                      │ store job → 直接跳过
 │                                      │ (dec_stored_request,
 │                                      │  不调 batch_put)
 │                                      │
 │     某次 batch_put 成功                │
 └──────────────────────────────────────┘
   _clear_store_pressure()
   所有 skip 标记清空
```

### 关键：压力状态怎么恢复

跳过条件是 AND 关系：

```python
return self._store_pressure_active and req_id in self._skip_store_requests
```

必须同时满足"压力状态激活" **且** "这个 req_id 在跳过名单里"。

```
                   _skip_store_requests 名单

时间    req_A    req_B    req_C    req_D    压力状态
─────   ──────   ──────   ──────   ──────   ────────
t0      失败→加入                            激活
t1               不在名单                    激活
                 → 尝试 put
                 → 成功!
                 → 清除全部                   关闭

或者:
t0      失败→加入                            激活
t1               不在名单                    激活
                 → 尝试 put
                 → 也失败→加入                激活
t2                        不在名单            激活
                          → 尝试 put
                          → 成功!
                          → 清除全部           关闭
```

**核心机制：只跳过已经失败过的 request 的后续 job，新 request 永远会尝试。** 新 request 充当了"探针"的角色——每来一个新的 req_id 就尝试一次 put。

### 为什么只对 NO_AVAILABLE_HANDLE 触发背压

代码里只对 `-200` 触发背压，不对 `-800` (TRANSFER_FAIL) 触发：

```python
if MOONCAKE_NO_AVAILABLE_HANDLE in failed_codes:
    self._mark_request_skipped_for_pressure(req_id)
```

两者的根因不同：

- `NO_AVAILABLE_HANDLE`：本地资源不够 → 减少并发可以缓解 → 背压有效
- `TRANSFER_FAIL`：远端网络问题 → 减少并发也没用 → 背压无意义

### batch_put 失败的所有可能原因

| 错误码 | 含义 | 根因 |
|--------|------|------|
| `-200` NO_AVAILABLE_HANDLE | staging buffer 耗尽 | 并发 put 太多 |
| `-10` BUFFER_OVERFLOW | 本地缓冲区溢出 | 单次 put 数据量超过 local_buffer_size |
| `-800` TRANSFER_FAIL | RDMA/TCP 传输失败 | 网络抖动、对端宕机、RDMA QP 异常 |
| `-900` RPC_FAIL | 元数据 RPC 调用失败 | metadata server 不可达 |
| `-500` WRITE_FAIL | 写入失败 | 远端存储后端拒绝写入 |
| `-700` INVALID_WRITE | 无效写入 | key 已存在且不允许覆写 |
| `-703` REPLICA_IS_NOT_READY | 副本未就绪 | 多副本模式下副本还在同步 |
| `-101` SEGMENT_NOT_FOUND | 段未找到 | 源 GPU 内存未注册到 store |
| `-400` INVALID_KEY | 无效 key | key 格式不合法 |

---

## 3.2 Offload 分批与 Mooncake eviction 粒度

### Offload 分批是什么

**不是分批发 RPC。** 是分批调用 Mooncake 的 `batch_get_into_multi_buffers` 这个本地 API。

启用 `enable_offload=true` 时，数据可能在本地 NVMe 磁盘上。从磁盘读到 GPU 的路径是：

```
NVMe 磁盘 → CPU staging buffer (direct I/O) → GPU 显存 (DMA)
                    ↑
              这块 buffer 是固定大小的
              默认 1280MB
```

### 为什么要分批

```
不分批:
  batch_get_into_multi_buffers(20 keys)
  → Mooncake 内部需要 2GB staging buffer
  → 超出 1280MB → NO_AVAILABLE_HANDLE

分批后:
  batch_get_into_multi_buffers(8 keys)  → ~800MB staging → 成功 → 释放 staging
  batch_get_into_multi_buffers(8 keys)  → ~800MB staging → 成功 → 释放 staging
  batch_get_into_multi_buffers(4 keys)  → ~400MB staging → 成功 → 释放 staging
```

三次调用是 **串行的**（在同一个 RecvingThread 里顺序执行），前一批完成释放 staging buffer 后，后一批才开始。

### 两个限制维度

```python
if (
    total_staging_bytes > self.usable_disk_offload_buffer_budget_bytes   # 字节数超了
    or len(key_list_c) > usable_batch_keys                               # key 数量超了
):
    load_batches = _split_disk_offload_load_batches(...)
```

- **字节数**：staging buffer 的物理容量限制
- **key 数量**：即使每个 key 很小，key 太多也会带来元数据开销和系统调用放大

### 三层防线

```
                     staging buffer 保护的三层防线

┌─ 第 1 层：事前拆批 (预防) ─────────────────────────────────────┐
│  预估 staging 字节数，超出预算时主动拆成多个小 batch               │
│  触发位置: RecvingThread._handle_request (load 路径)            │
│  效果: 每批 staging 用量控制在预算内                             │
└───────────────────────────────────────────────────────────────┘
                          │
                   如果预估不准或并发太高
                          ▼
┌─ 第 2 层：背压控制 (事后应对) ─────────────────────────────────┐
│  batch_put 返回 NO_AVAILABLE_HANDLE 时                        │
│  标记该 request 后续 store job 跳过                            │
│  触发位置: SendingThread._handle_request (save 路径)           │
│  效果: 减少并发 put 数量，等已有传输完成释放 staging             │
└───────────────────────────────────────────────────────────────┘
                          │
                   如果单个 key 就超出预算
                          ▼
┌─ 第 3 层：放弃 (兜底) ────────────────────────────────────────┐
│  单个 key 的 staging 需求 > 总预算                             │
│  直接放弃该 request 的 load，set_finished_request              │
│  上层 scheduler 通过 invalid_block 机制回退到本地 recompute     │
│  触发位置: RecvingThread._handle_request (oversized_key)       │
│  效果: 避免永远无法完成的死循环                                  │
└───────────────────────────────────────────────────────────────┘
```

### Mooncake 的 eviction 粒度

**每个 TP rank 独立 put，key 互不相同。** 对 Mooncake store 来说，这些是完全独立的 key-value 对：

```
"Qwen2-7B@tp_rank:0@pcp0@dcp0@pp_rank:0@a3f8c2"
"Qwen2-7B@tp_rank:1@pcp0@dcp0@pp_rank:0@a3f8c2"
"Qwen2-7B@tp_rank:2@pcp0@dcp0@pp_rank:0@a3f8c2"
"Qwen2-7B@tp_rank:3@pcp0@dcp0@pp_rank:0@a3f8c2"
```

Mooncake 不知道它们之间有逻辑关联——它不理解"这 4 个 key 其实是同一个 chunk 在 4 个 TP rank 上的分片"。

**Mooncake 的管理和 eviction 粒度是单个 key：**

```
Mooncake store 视角：

key_1 → value_1 (一坨字节)     热度高，保留
key_2 → value_2 (一坨字节)     热度低，evict
key_3 → value_3 (一坨字节)     热度中，保留

它不知道 key_1 和 key_2 "必须同时存在才有意义"
```

### 导致的一致性问题

```
chunk_hash="a3f8c2"

tp_rank:0 的 key → 还在 store 里
tp_rank:1 的 key → 被 evict 了
tp_rank:2 的 key → 还在 store 里
tp_rank:3 的 key → 还在 store 里
```

只有 1 个 rank 的数据丢了，但这个 chunk 对 vLLM 来说就是**完全不可用的**——forward pass 需要所有 TP rank 的 KV 同时就绪。

### 为什么 Mooncake 不做"按组 evict"

```
方案 A (当前): Mooncake 不感知 TP/PP 关系，按单 key evict
  优点: store 层通用，不需要理解 vLLM 的并行拓扑
  缺点: 可能 evict 不一致，浪费残留 shard 的存储空间
  应对: vLLM 侧做全 shard 一致性检查

方案 B (假设): Mooncake 感知 TP/PP 关系，按组 evict
  优点: 要么全有要么全无，不会浪费空间
  缺点: store 需要理解 vLLM 的并行维度、key 命名约定
        store 和推理框架强耦合
```

当前选的是方案 A——**store 保持通用，把一致性问题推到上层（vLLM lookup）解决。**

---

# 第四部分：可观测性

## 4.1 KV Events 链与外部交互

### 事件类型

```python
class BlockStored(KVCacheEvent):     # 一个 block 被存入（store/cache）
    block_hashes: list[ExternalBlockHash]
    parent_block_hash: ExternalBlockHash | None
    token_ids: list[int]
    block_size: int
    medium: str | None               # "cpu", "GPU" 等

class BlockRemoved(KVCacheEvent):     # 一个 block 被移除
    block_hashes: list[ExternalBlockHash]
    medium: str | None

class AllBlocksCleared(KVCacheEvent): # 全部清空
    pass
```

### parent_block_hash 链表

SendingThread 在写入时生成事件链：

```python
prev_key = None
for index, start in enumerate(starts):
    stored_event = BlockStored(
        block_hashes=[new_block_hashes[index]],
        parent_block_hash=prev_key,          # 指向前一个 chunk
        token_ids=token_ids,
    )
    prev_key = new_block_hashes[index]       # 更新链表指针
```

生成的事件链：

```
请求: "The quick brown fox jumps over the lazy dog ..."

BlockStored {
  block_hash: "a1b2",
  parent: None,           ← 第一个 chunk，没有 parent
  token_ids: [The, quick, brown, ...128个 token],
}
    │
    ▼
BlockStored {
  block_hash: "c3d4",
  parent: "a1b2",         ← 指向第一个 chunk
  token_ids: [fox, jumps, over, ...128个 token],
}
    │
    ▼
BlockStored {
  block_hash: "e5f6",
  parent: "c3d4",         ← 指向第二个 chunk
  token_ids: [the, lazy, dog, ...128个 token],
}
```

### 多 request 共享前缀时是一棵树

```
"The quick brown fox jumps"  → chunk a1b2 → chunk c3d4
"The quick brown fox sleeps" → chunk a1b2 → chunk g7h8
"The quick brown cat runs"   → chunk a1b2 → chunk i9j0

                a1b2 (共享前缀)
               / | \
            c3d4 g7h8 i9j0 (各自的第二个 chunk)
```

外部消费者收到事件后，可以重建整棵 prefix tree，知道哪些 block 是共享的、哪些是某个 request 独有的。

### 完整数据流

```
Worker 进程                     Scheduler 进程                     外部消费者
─────────────                  ────────────────                   ───────────

SendingThread
  │ batch_put 成功
  │ 生成 BlockStored 事件
  ▼
KVCacheStoreSendingThread
  .update_kv_event(events)
  │
  ▼
get_kv_connector_kv_cache_events()
  │ 收集事件 → KVConnectorOutput
  │
  │ ═══ KVConnectorOutput 跨进程传回 ═══
  │                                    ▼
  │                            KVEventAggregator
  │                              .add_events()
  │                              │
  │                              │ 所有 TP worker 都报告了？
  │                              │  Yes → get_common_events()
  │                              │  No  → 继续等
  │                              ▼
  │                            connector.take_events()
  │                              │
  │                            kv_cache_manager.take_events()
  │                              │
  │                            合并两个来源的事件
  │                              │
  │                              ▼
  │                            KVEventBatch(ts=now, events=[...])
  │                              │
  │                            kv_event_publisher.publish(batch)
  │                              │
  │                              ▼
  │                         ┌─────────────────┐
  │                         │ ZmqEventPublisher│
  │                         │ (后台线程)        │
  │                         └────────┬────────┘
  │                                  │
  │                       ┌──────────┴──────────┐
  │                       │                     │
  │                    ZMQ PUB              ZMQ ROUTER
  │                    (实时推送)            (历史重放)
  │                       │                     │
  │                       ▼                     ▼
  │                 ┌───────────┐         ┌───────────┐
  │                 │ 外部消费者  │         │ 外部消费者  │
  │                 │ (SUB 订阅) │         │ (REQ 请求  │
  │                 │            │         │  历史事件)  │
  │                 └───────────┘         └───────────┘
```

### ZMQ PUB/SUB：实时推送

```python
# Publisher 端 (vLLM 内部)
self._pub = self._ctx.socket(zmq.PUB)
self._pub.bind("tcp://*:5557")

# 每条消息格式: [topic, sequence_number, msgpack_payload]
self._pub.send_multipart((topic_bytes, seq_bytes, payload))
```

外部消费者：

```python
import zmq, msgspec

ctx = zmq.Context()
sub = ctx.socket(zmq.SUB)
sub.connect("tcp://vllm-host:5557")
sub.subscribe(b"")  # 订阅所有 topic

while True:
    topic, seq_bytes, payload = sub.recv_multipart()
    seq = int.from_bytes(seq_bytes, "big")
    batch = msgspec.msgpack.decode(payload, type=KVEventBatch)

    for event in batch.events:
        if isinstance(event, BlockStored):
            print(f"Block {event.block_hashes} stored, "
                  f"parent={event.parent_block_hash}")
```

### ZMQ ROUTER/REQ：历史重放

PUB/SUB 无法保证可靠投递。Publisher 维护了一个环形 buffer（默认 10000 条）：

```python
self._buffer = deque[tuple[int, bytes]](maxlen=buffer_steps)
```

消费者发现 sequence number 不连续时，可以通过 ROUTER socket 请求重放：

```python
# 消费者发现丢了 seq 100-200 之间的事件
req = ctx.socket(zmq.REQ)
req.connect("tcp://vllm-host:5558")   # replay 端口 = pub 端口 + 1

req.send(100 .to_bytes(8, "big"))    # 从 seq=100 开始重放

while True:
    seq_bytes, payload = req.recv_multipart()
    seq = int.from_bytes(seq_bytes, "big", signed=True)
    if seq == -1:       # END_SEQ 标记
        break
    batch = msgspec.msgpack.decode(payload)
```

### KVEventAggregator 的跨 worker 聚合

```python
class KVEventAggregator:
    def get_common_events(self) -> list[KVCacheEvent]:
        return [
            event
            for event, count in self._event_counter.items()
            if count == self._num_workers    # 所有 worker 都报告了才算
        ]
```

为什么需要聚合？因为 TP 下每个 worker 独立 put，但从外部视角，一个 block "真正存好"意味着所有 TP rank 的 shard 都写入成功。如果只有 3/4 的 rank 写成功了，这个 block 对外部是不可用的——所以只有所有 worker 都报告了 `BlockStored` 才发布。

**这和 lookup 全 shard 一致性检查是同一个问题的两面：**

- **Lookup**：读之前检查是否所有 shard 都有
- **Events**：写之后通知外部只有所有 shard 都写了才算

### DP 多实例隔离

每个 DP rank 的 Publisher 绑定不同端口：

```python
@staticmethod
def offset_endpoint_port(endpoint, data_parallel_rank):
    # tcp://*:5557 + dp_rank=2 → tcp://*:5559
    base_port = int(endpoint[last_colon_idx + 1:])
    new_port = base_port + data_parallel_rank
    return f"{base_addr}:{new_port}"
```

```
DP rank 0 → tcp://*:5557
DP rank 1 → tcp://*:5558
DP rank 2 → tcp://*:5559
```

外部消费者连不同端口就能区分不同 DP 实例的事件，不会混在一起。

### 消费者是谁

事件通过 `EventPublisher` 发出后，可能的消费者：

- 外部 KV cache 管理系统（知道什么数据在 store 里）
- 监控/Grafana（统计 cache 命中率、eviction 速率）
- 另一个 vLLM 实例（据此决定是否可以复用远端 KV）
