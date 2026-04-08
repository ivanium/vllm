# Mooncake KV Transfer 核心机制精读笔记

> 适用代码库：`/home/aoshen/setup_new_cluster/vllm/`
> 本文档整理了对 Mooncake + vLLM KV Transfer 代码的深度讨论，按理解难度递进排列。
> 每节都给出"为什么这样设计"的解释，而不只是描述"是什么"。

---

## 目录

1. [request_finished() —— 延迟释放的精髓](#1-request_finished----延迟释放的精髓)
2. [RDMA Datapath —— 零 CPU 拷贝](#2-rdma-datapath----零-cpu-拷贝)
3. [PD 分离部署（Disaggregated Serving）完整流程](#3-pd-分离部署disaggregated-serving完整流程)
4. [两个服务端口的分工](#4-两个服务端口的分工)
5. [Benchmark 流程解析](#5-benchmark-流程解析)
6. [Native CPU Offload：Simple vs OffloadingConnector](#6-native-cpu-offloadsimple-vs-offloadingconnector)
7. [ARC 算法——精准理解四个队列](#7-arc-算法精准理解四个队列)
8. [FilterReusedOffloadingManager —— 频率过滤](#8-filterreusedoffloadingmanager----频率过滤)
9. [Chain Hash —— 为什么相同前缀有相同哈希](#9-chain-hash----为什么相同前缀有相同哈希)
10. [GQA put_step —— 跨 TP rank 去重](#10-gqa-put_step----跨-tp-rank-去重)
11. [精华代码模式汇总](#11-精华代码模式汇总)
12. [完整心智模型](#12-完整心智模型)

---

## 1. request_finished() —— 延迟释放的精髓

### 接口定义

```
base.py:530-549
request_finished(request, block_ids) → (bool, dict|None)
```

返回值语义：
- `(False, None)` —— 请求结束，立刻释放 GPU blocks（大多数 connector 的默认行为）
- `(True, data)` —— 先**不**释放 GPU blocks，等异步传输完成后再释放

### 为什么需要延迟释放？

P2P 模式里，Prefiller 做完 prefill 之后，Consumer（Decoder）才来通过 RDMA 拉取 KV cache。
如果 Prefiller 在 `request_finished()` 时立刻释放 GPU blocks，
那块内存可能被新请求覆盖，Consumer 拉到的就是垃圾数据。

所以：**Prefiller 必须持有 GPU blocks，直到 RDMA 传输完成**。

### Prefiller 侧的完整调用栈

```
1. 推理结束（output.finish_reason == FINISHED_LENGTH_CAPPED 即单 token 输出）
   scheduler._free_request(request)
   ↓
2. scheduler._connector_finished(request, block_ids)
   → connector.request_finished() 返回 (True, {transfer_id, block_ids})
   ↓
3. scheduler 检查 delay_free_blocks=True → 跳过 _free_blocks()
   同时把 block_ids 存入待发送队列：
   reqs_need_send[transfer_id] = SendBlockMeta(block_ids=..., ready=asyncio.Event())
   ↓
4. Worker 侧 start_load_kv() 收到 build_connector_meta 下发的 reqs_to_send
   → SendBlockMeta.ready.set()  (通知发送线程可以开始发了)
   ↓
5. Consumer 发来 ZMQ 拉取请求 MooncakeXferMetadata（含 dst_ptrs）
   → _sender_worker 等到 ready.set() 后调用 send_kv_to_decode()
   → engine.batch_transfer_sync_write(src_ptrs, dst_ptrs, lengths)  [RDMA 传输]
   ↓
6. RDMA 完成后，Producer Worker 把 transfer_id 放入 finished_sending set
   ↓
7. 下一个 step 调用 get_finished() 时，Scheduler 收到 finished_sending
   → scheduler._update_from_kv_xfer_finished()
   → _free_blocks(block_ids)  ← GPU blocks 此时才真正释放
```

**关键文件位置：**
- `scheduler.py:1818-1838` — `_free_request` 检查 `delay_free_blocks`
- `scheduler.py:2111-2138` — `_update_from_kv_xfer_finished` 触发延迟释放
- `mooncake_connector.py:584-635` — `request_finished` Prefiller 路径

### Consumer（Decoder）侧的行为

Consumer 的 `request_finished()` 返回 `(False, None)`。

原因：Consumer 在拿到 KV cache 之前就已经把 block 分配好了（`get_num_new_matched_tokens` 返回 True，表示 blocks 会被异步填充）。请求结束时 KV 已经传完，block 里的内容也不再需要，可以立刻释放。

### FINISHED_LENGTH_CAPPED 是什么意思？

Proxy 给 Prefiller 的请求设置 `max_tokens=1`（只需要输出 1 个 token），Prefiller 输出完这 1 个 token 后，finish reason 变成 `FINISHED_LENGTH_CAPPED`（达到长度上限）。

这是 PD 分离部署里约定的**正常结束信号**，不是错误。

---

## 2. RDMA Datapath —— 零 CPU 拷贝

### 传输路径

```
Prefiller GPU 内存
    ↓  (RDMA NIC DMA 读)
Prefiller RDMA NIC
    ↓  (InfiniBand / RoCE)
Decoder  RDMA NIC
    ↓  (RDMA NIC DMA 写)
Decoder  GPU 内存
```

**CPU 全程不参与数据搬运**，只负责下发 `(src_ptr, dst_ptr, length)` 三元组给 RDMA 引擎。

### GPU 内存注册

RDMA 要访问 GPU 内存，必须先向 RDMA NIC "注册"该内存区域：

```python
# mooncake_connector.py:1215-1287
def register_kv_caches(kv_caches):
    for layer_name, kv_cache in kv_caches.items():
        ptr = kv_cache.data_ptr()   # GPU 内存物理地址
        size = kv_cache.nbytes()
        engine.batch_register_memory([ptr], [size])
```

注册后，Mooncake C++ 引擎持有这段内存的 MR（Memory Region），可对其发起 RDMA Read/Write。

### asyncio + ThreadPoolExecutor 模式

RDMA 传输是**同步阻塞**操作（等待 ACK），但 vLLM Worker 是 asyncio 事件循环。
为了不阻塞事件循环，用 `run_in_executor` 把阻塞操作放到线程池：

```python
# mooncake_connector.py
await loop.run_in_executor(
    self._executor,
    self._send_blocks,  # 同步阻塞函数
    src_ptrs, dst_ptrs, lengths
)
```

这是 Python asyncio 里处理同步阻塞 I/O 的标准模式。

---

## 3. PD 分离部署（Disaggregated Serving）完整流程

### 两个角色

| 角色 | 配置 | 职责 |
|------|------|------|
| Prefiller | `kv_role=kv_producer` | 只做 prefill，产出 KV cache，通过 RDMA 推送给 Decoder |
| Decoder   | `kv_role=kv_consumer` | 不做 prefill，等 KV cache 从 Prefiller 到达后开始 decode |

### Proxy 路由逻辑

Proxy 接收到一个用户请求后，发出**两个独立请求**：

```
Proxy → Prefiller:
    prompt=<原始 prompt>
    max_tokens=1            ← 只需 prefill，不需要真正 decode
    kv_transfer_params={
        "transfer_id": "uuid-xxx",
        "do_remote_prefill": false,
        "do_remote_decode": true,   ← 通知 Prefiller 把 KV 传给 Decoder
        "decoder_engine_id": "engine-yyy",
    }

Proxy → Decoder:
    prompt=<原始 prompt>
    max_tokens=<真实生成长度>
    kv_transfer_params={
        "transfer_id": "uuid-xxx",  ← 相同的 transfer_id！
        "do_remote_prefill": true,  ← 通知 Decoder 等 KV 从 Prefiller 到来
        "do_remote_decode": false,
        "prefill_engine_id": "engine-zzz",
    }
```

两个请求共享同一个 `transfer_id`，这是 Prefiller 和 Decoder 匹配传输的唯一标识。

### ZMQ 通信角色

- Prefiller：ZMQ **ROUTER** socket（被动监听，可接受多个 Consumer 的拉取请求）
- Decoder：ZMQ **DEALER** socket（主动连接 Prefiller 的 ROUTER，发出拉取请求）

Consumer 主动拉（Pull），Producer 被动提供（Push-on-demand）。

---

## 4. 两个服务端口的分工

### Port 8080 — HTTP Bootstrap 服务（RDMA 节点发现）

**作用**：RDMA 通信建立前的"地址簿"。

每个 Worker 启动时向 Prefiller 的 Bootstrap 服务注册自己的 RDMA 地址：
```
POST /register
{engine_id, tp_rank, pp_rank, addr: "192.168.1.100:12345"}
```

Consumer 在第一次发起拉取时查询：
```
GET /query?engine_id=xxx
→ {tp_rank: {pp_rank: "addr"}}
```

**特点**：只用于连接建立阶段（一次性），不参与每次传输。

### Port 50051 — gRPC Master 服务（KV block 元数据目录）

**作用**：Store 模式的"总账本"，跟踪哪些 block hash 在哪台机器的 CPU 内存里。

关键 RPC：
```
batch_is_exist(keys[])  → 哪些 key 已存在（避免重复写入）
batch_put(keys[], ...)  → 注册 block 位置
batch_get(keys[], ...)  → 查询 block 在哪里，返回 RDMA 地址
```

驱逐策略：
- `MC_EVICT_HI=0.95` — 内存使用率超过 95% 触发 LRU 驱逐
- `MC_EVICT_RATIO=0.1` — 每次驱逐释放 10% 容量
- `MC_LEASE_TTL=30000ms` — block 的 TTL（30秒未访问自动过期）

**关键区别**：
- 8080 是 Prefiller-specific 的（每个 Prefiller 自己跑一个）
- 50051 是全局唯一的（整个集群共享一个 Master）

---

## 5. Benchmark 流程解析

### 基本骨架（两个 benchmark 脚本的共同模式）

```bash
for backend in baseline native simple mooncake; do
    1. 启动 vllm serve（含 backend 对应的 offload 参数）
    2. 等待 /health 端点响应（最多 180s）
    3. 运行 vllm-bench --dataset-name random ...
    4. kill server
done

python compare_results.py $RESULT_DIR   # 对比各 backend 结果
```

### benchmark_cpu_offloading.sh —— 纯 overhead 测量

**关键设计**：`--seed 42` + `--dataset-name random` 生成随机唯一 prompt，确保**没有任何 prefix 命中**。

目的：隔离"offload 本身的 store/load overhead"，排除 prefix cache 节省计算量的干扰。

四个 backend：
| Backend | 启动参数 | 说明 |
|---------|---------|------|
| baseline | 无 offload | 纯 GPU cache，性能基线 |
| native | `--kv-offloading-backend native` + `--kv-offloading-size 80` | vLLM 内置 native offload |
| simple | 同 native + `VLLM_USE_SIMPLE_KV_OFFLOAD=1` | 简化实现，无策略 |
| mooncake | `--kv-transfer-config '{"kv_connector":"MooncakeStoreConnector",...}'` | 分布式 store |

### benchmark_multi_turn.sh —— Prefix Reuse 效益测量

**关键设计**：构造多轮对话 prompt，有大量共享前缀（global prefix + conv prefix），确保有 prefix 命中。

目的：测量 Mooncake 在真实前缀复用场景下带来的吞吐提升。

---

## 6. Native CPU Offload：Simple vs OffloadingConnector

### 选择逻辑

```python
# vllm/config/vllm.py:668-676
if kv_offloading_backend == "native":
    if os.environ.get("VLLM_USE_SIMPLE_KV_OFFLOAD", "0") == "1":
        connector = SimpleCPUOffloadConnector    # 简化实现
    else:
        connector = OffloadingConnector           # 完整实现（含 ARC/LRU）
```

### Simple CPU Offload（SimpleCPUOffloadConnector）

**特点**：极简，几乎没有策略。

核心 Worker 方法都是空操作：
```python
def start_load_kv(self): pass
def save_kv_layer(self): pass
def wait_for_save(self): pass
```

所有 I/O 都在 `get_finished()` 里驱动。

**两种模式**：
- **Eager 模式**：每个请求的 full block 算完就立刻 `cudaMemcpyAsync` 到 pinned CPU 内存
- **Lazy 模式**：用游标扫描 `FreeKVCacheBlockQueue`，发现即将被驱逐的 cached block，提前搬到 CPU

**限制**：没有驱逐策略，没有频率过滤，CPU 内存满了就直接跳过。

### OffloadingConnector（完整实现）

**特点**：完整的插件化架构，支持可替换驱逐策略。

关键抽象层：`OffloadingManager`（在 `v1/kv_offload/abstract.py`）
- `lookup(block_hashes)` — 查询哪些 block 在 CPU 里
- `prepare_store(block_hashes)` — 返回哪些 block 应该被存储
- `touch(block_hashes)` — 更新访问时间（LRU/ARC 用）
- `evict(n)` — 驱逐 n 个 block

两种驱逐策略（通过 `eviction_policy` 参数选择）：
- `lru` — 标准 LRU
- `arc` — ARC（Adaptive Replacement Cache）

额外装饰器：`FilterReusedOffloadingManager`（频率过滤，见第 8 节）

**`block_size_factor`**：支持跨层合并 block，减少元数据开销：
```
block_size_factor=4 意味着：每 4 个 GPU block 合并为 1 个 CPU block
→ 减少 store 里的 entry 数量，降低管理开销
```

### 对比表

| 维度 | SimpleCPUOffloadConnector | OffloadingConnector |
|------|--------------------------|---------------------|
| 驱逐策略 | 无（满了跳过）| LRU 或 ARC（自适应）|
| 频率过滤 | 无 | FilterReusedOffloadingManager（可选）|
| block 合并 | 无 | block_size_factor |
| 实现复杂度 | 极简 | 完整插件化架构 |
| 适用场景 | 测试/调试/简单场景 | 生产环境 |

---

## 7. ARC 算法——精准理解四个队列

### 背景：为什么不用 LRU？

LRU 的问题：**一次性扫描型访问**会污染 cache。

例如：一个超长 prompt 产生大量 block，把整个 LRU 队列顶掉，
导致其他请求频繁命中的 block 被驱逐。

ARC 的目标：**自适应地平衡"最近访问"和"频繁访问"**，避免被一次性访问冲垮。

### 四个队列的含义

```
T1（Recently Used Once）   = "最近只用过一次的 block"
T2（Frequently Used）      = "多次访问过的 block"  ← 这里的 block 不容易被驱逐
B1（Ghost of T1 evictions）= "T1 里被驱逐 block 的记录（只有 hash，没有数据）"
B2（Ghost of T2 evictions）= "T2 里被驱逐 block 的记录（只有 hash，没有数据）"
```

**用图书馆打比方：**
```
T1 = 新书架（你昨天借过一次的书）
T2 = 常借书架（你借过多次的书）
B1 = T1 退回记录本（记着你退了哪些新书，但书已经被别人拿走了）
B2 = T2 退回记录本（记着你退了哪些常借书，但书已经被别人拿走了）
```

### 状态机：block 的流转

```
初次看到 hash → 加入 T1 尾部（"刚来的新书"）
    ↓ T1 满了且需要驱逐
从 T1 头部驱逐 → 删除数据，在 B1 里留记录（"新书被还走，留个记录"）
    ↓ 下次又用这个 hash（命中 B1）
B1 hit → 说明 T1 太小了，这个 block 在 T1 里还没熬到第二次访问就被驱逐了
    → target_t1_size 增大  → 重新加入 T2（"这本书其实是常借书，错放在新书架了"）
    ↓ T2 满了且需要驱逐
从 T2 头部驱逐 → 删除数据，在 B2 里留记录
    ↓ 下次又用这个 hash（命中 B2）
B2 hit → 说明 T2 太小了，这个 block 被驱逐了但其实是常用的
    → target_t1_size 减小（给 T2 更多空间）
```

### 核心公式

```python
# v1/kv_offload/cpu/policies/arc.py

# T1 命中：从 T1 提升到 T2（直接升级为"常借书"）
if hash in T1:
    T1.pop(hash)
    T2[hash] = value

# B1 命中：T1 太小 → 扩大 T1 的目标大小
elif hash in B1:
    delta = max(1, len(B2) // len(B1))
    target_t1_size += delta   # 允许 T1 变大
    B1.pop(hash)
    T2[hash] = value          # 这次直接进 T2

# B2 命中：T2 太小 → 缩小 T1（相当于给 T2 腾空间）
elif hash in B2:
    delta = max(1, len(B1) // len(B2))
    target_t1_size -= delta   # 缩小 T1，T2 自然得到更多空间
    B2.pop(hash)
    T2[hash] = value
```

### "为什么 B1 命中要扩大 T1，而不是扩大 T2？"

**误解**：B1 说明 block 在 T1 被驱逐了，是因为 T2 太大占了 T1 的空间。
所以应该扩大 T2 让 block 直接进 T2？

**正确理解**：
- B1 命中说明这个 block 在 T1 里还没等到**第二次访问**就被驱逐了。
- 它第一次进来时只被访问一次，ARC 把它放 T1。
- 但 T1 太小，它被驱逐了，没等到第二次访问就消失了。
- 现在第二次访问来了（命中 B1），证明"它确实需要被访问两次"。
- **根本原因**：T1 不够大，block 在 T1 里熬不到升 T2 就被踢走了。
- **解决方案**：扩大 T1，让下一批类似 block 能在 T1 里多活一段时间，等到第二次访问后正常升 T2。

### 虚拟驱逐（virtual_t1_size）

`evict(n)` 不是逐个驱逐再重新计算，而是**先模拟驱逐**，然后一次性提交：

```python
def evict(n):
    virtual_t1 = len(T1)  # 模拟 T1 的当前大小
    candidates = []
    for i in range(n):
        if virtual_t1 > target_t1_size:
            # T1 过大，从 T1 驱逐
            candidates.append(("T1", T1.oldest_key()))
            virtual_t1 -= 1
        else:
            # T1 合适，从 T2 驱逐
            candidates.append(("T2", T2.oldest_key()))
    # 确认候选后，再真正修改 T1/T2/B1/B2
    for source, key in candidates:
        _do_evict(source, key)
```

**意义**：原子性地选出 n 个候选，避免驱逐过程中 target_t1_size 状态不一致。

---

## 8. FilterReusedOffloadingManager —— 频率过滤

### 作用

只存储"被多次使用的 block"，过滤掉一次性访问的 block（如长 prompt 的唯一内容）。

```python
# v1/kv_offload/reuse_manager.py
class FilterReusedOffloadingManager:
    def __init__(self, inner_manager, store_threshold=0):
        self.inner = inner_manager
        self.store_threshold = store_threshold  # 默认 0 = 禁用
        self.counts = defaultdict(int)          # block_hash → 访问次数

    def lookup(self, block_hashes):
        for h in block_hashes:
            self.counts[h] += 1   # 每次查询都计数
        return self.inner.lookup(block_hashes)

    def prepare_store(self, block_hashes):
        filtered = [h for h in block_hashes
                    if self.counts[h] >= self.store_threshold]
        return self.inner.prepare_store(filtered)
```

### 默认行为

`store_threshold=0`（从 `v1/kv_offload/cpu/spec.py:56-81` 读取）意味着：
- **Filter 被禁用**，所有 block 都进入 ARC。
- 激活 Filter 需要手动设置 `store_threshold >= 2`。

### 计数的粒度

**跨请求共享计数**：`counts` 以 `block_hash` 为 key，不区分来自哪个 `request_id`。

这意味着：
- 请求 A 的 prompt 前 1024 token（hash=0xABCD）→ `counts[0xABCD] = 1`
- 请求 B 的 prompt 前 1024 token（hash=0xABCD，因为 token 相同）→ `counts[0xABCD] = 2`
- 第二次后，这个 block 通过 Filter → 存入 ARC

注意：这里判断的是"跨不同请求的 block 复用"，而不是同一个请求重复 prefill。
同一个请求重复 prefill 两次，每次 hash 相同，计数会增加到 2，也能通过 Filter——
但这和"两个不同请求用了同一个前缀"在机制上是等价的，Filter 无法区分也不需要区分。

---

## 9. Chain Hash —— 为什么相同前缀有相同哈希

### 公式

```python
# v1/core/kv_cache_utils.py
block_hash = hash(
    parent_block_hash      # 前一个 block 的哈希
    + token_ids_bytes      # 本 block 的 token ids
    + extra_keys_bytes     # 额外的 key（如 LoRA id）
)
```

### 关键性质

**链式计算**：每个 block 的 hash 依赖于它前面所有 block 的内容。

这保证了：
- 两个请求只要共享相同的 token 前缀（前 N 个 token 完全一样），
  它们对应 block 的 hash 就完全相同。
- 与请求 ID 无关，与到达时间无关。

**实际例子**：
```
请求 A：[token1, token2, token3, token4]  → block_hash_A = hash(0 + [t1,t2,t3,t4])
请求 B：[token1, token2, token3, token4, token5, token6]
                                           block 0: hash(0 + [t1,t2,t3,t4]) = block_hash_A （相同！）
                                           block 1: hash(block_hash_A + [t5,t6,...])
```

→ 请求 A 算出来的 KV cache，请求 B 的 block 0 可以**零成本复用**。

### 这和 Store 模式的关系

Store 模式的 `PoolKey.chunk_hash` 就是这个 chain hash。
所以任何机器上的任何请求，只要 token 前缀相同，就会命中同一个 store 里的 KV cache。
这是 Mooncake Store 实现跨机器 prefix sharing 的核心基础。

---

## 10. GQA put_step —— 跨 TP rank 去重

### 问题背景

Grouped Query Attention（GQA）下，`num_kv_heads` < `num_attention_heads`。

当 `num_kv_heads` < `tp_size` 时，例如 `num_kv_heads=4, tp_size=8`：
- TP rank 0 和 TP rank 4 持有**完全相同的 KV heads**（head 0~3 被两个 rank 共享）
- 如果两个 rank 都往 store 里写，就会重复写入相同数据

### put_step 解决方案

```python
# mooncake_store_worker.py（send 线程）
put_step = tp_size // num_kv_heads  # = 8//4 = 2

# 只有满足条件的 rank 才写入
if tp_rank % put_step == 0:
    # 只有 rank 0, 2, 4, 6 写入（步长为 put_step=2）
    do_put()
```

实际上，使用了 slice 写法 `[tp_rank % put_step :: put_step]`，
保证每个物理 KV head 恰好被**一个 TP rank 负责写入**，其他 rank 跳过。

**结果**：store 里每个 KV block 只存一份，不会因为 GQA 导致 2x 重复写入。

---

## 11. 精华代码模式汇总

### 1. asyncio.Event 协调 Scheduler↔Worker 跨线程同步

```python
# SendBlockMeta.ready = asyncio.Event()
# Scheduler 侧（主线程）：
send_meta.ready.set()   # "block 准备好了，可以发了"

# Worker 侧（asyncio 协程里）：
await send_meta.ready.wait()  # 等 Scheduler 通知
```

用 asyncio.Event 代替锁，零阻塞地协调两个异步方向的状态。

### 2. ZMQ ROUTER/DEALER 多路复用

```
Prefiller           多个 Decoder
ZMQ ROUTER   ←──── ZMQ DEALER (Decoder 1)
             ←──── ZMQ DEALER (Decoder 2)
             ←──── ZMQ DEALER (Decoder 3)
```

ROUTER socket 自动为每个 DEALER 分配 identity，回复时带上 identity 即可路由回正确的 Decoder。
一个 ROUTER 可同时服务任意数量的 Decoder，无需为每个 Decoder 维护独立 socket。

### 3. 相邻 block 合并（coalesce）减少 RDMA 次数

```python
# mooncake_connector.py:_build_transfer_params
# 如果两个 block 在 GPU 内存里地址连续：
if block_i_end_addr == block_{i+1}_start_addr:
    merge_into_single_transfer_region()
```

GPU 内存按 page 分配，如果恰好连续，就合并成一次大传输，
把 N 次 RDMA 变成 1 次，显著降低传输延迟。

### 4. 请求队列 + 背压（backpressure）

```python
# KVTransferThread 基类
while True:
    req = queue.get(timeout=1.0)
    self._handle_request(req)
    # 如果 store handle 不足（返回 MOONCAKE_NO_AVAILABLE_HANDLE）：
    # → _mark_request_skipped_for_pressure()
    # → 等 store 压力释放（_clear_store_pressure）后重试
```

这是生产者-消费者模式的标准背压实现：
Worker 把请求投入队列（不阻塞），Send 线程从队列消费，
压力大时 Skip，压力小时 Recover，整个系统不会因为 store 暂时不可用而崩溃。

---

## 12. 完整心智模型

读完这份笔记后，对着下图能说清楚每个箭头代表什么：

```
                    ┌─────────────────────────────────────────────┐
                    │              Prefiller（Producer）            │
                    │  Scheduler                   Worker          │
                    │  ─────────                   ──────          │
                    │  request_finished() ──────→ SendBlockMeta   │
                    │  (返回 True,延迟释放)   ready.set()          │
                    │                              ↓               │
                    │                         RDMA engine          │
                    └──────────────────────────────┬──────────────┘
                                                   │ batch_transfer_sync_write
                                                   │ (RDMA，零 CPU 拷贝)
                    ┌──────────────────────────────▼──────────────┐
                    │              Decoder（Consumer）              │
                    │  Worker                   Scheduler          │
                    │  ──────                   ─────────          │
                    │  receive_kv ──完成──→ get_finished()         │
                    │                       ↓                      │
                    │                  _update_from_kv_xfer        │
                    │                  → 开始 decode              │
                    └─────────────────────────────────────────────┘

                    ┌─────────────────────────────────────────────┐
                    │              Store 模式（任意实例）            │
                    │                                              │
                    │  Token hash → PoolKey → Master(50051)查询   │
                    │  → 命中 → RDMA 直接从持有者 GPU 拉取到本 GPU │
                    │  → 未命中 → 正常计算，计算完存入 Store        │
                    │                                              │
                    │  Chain Hash 保证：相同 token 前缀 = 相同 key  │
                    │  put_step 保证：GQA 下不重复写入              │
                    │  ARC 保证：CPU 缓存优先保留高频 block          │
                    └─────────────────────────────────────────────┘
```
