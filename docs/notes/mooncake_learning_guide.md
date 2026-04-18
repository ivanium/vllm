# Mooncake KV Transfer 代码精读学习指南

> 适用代码库：`/home/aoshen/setup_new_cluster/vllm/`
> 涵盖模块：`vllm/scripts/mooncake/` + `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/`

---

## 一、全局架构速览

### 两种工作模式

```
模式 A：MooncakeConnector（P2P 直连）
  Prefiller (Producer) ──RDMA──> Decoder (Consumer)
  特点：点对点，每次请求单独传输，适合 Disaggregated Serving

模式 B：MooncakeStoreConnector（共享池）
  所有实例 <──> MooncakeDistributedStore（CPU/Disk 共享内存池）
  特点：哈希去重，多实例共享前缀 cache，适合多轮对话/prefix reuse
```

### 三层分工

```
┌─────────────────────────────────────────────────────┐
│  Scheduler 层（CPU，主进程）                          │
│  职责：元数据管理、状态跟踪、block 分配决策              │
│  接口：get_num_new_matched_tokens / build_connector_meta │
└──────────────────────────┬──────────────────────────┘
                           │ KVConnectorMetadata（每 step 下发）
┌──────────────────────────▼──────────────────────────┐
│  Worker 层（GPU 进程 + 异步线程）                      │
│  职责：实际 KV 搬运、注册 RDMA buffer、异步等待         │
│  接口：start_load_kv / save_kv_layer / wait_for_save  │
└──────────────────────────┬──────────────────────────┘
                           │ GPU 内存地址 / RDMA
┌──────────────────────────▼──────────────────────────┐
│  Mooncake TransferEngine（C++ RDMA 引擎）              │
│  职责：零拷贝内存传输，RDMA/TCP 协议                    │
└─────────────────────────────────────────────────────┘
```

### 一次 Step 的完整调用顺序（核心心智模型）

```
每个推理 step，Scheduler 侧依次调用：
  1. get_num_new_matched_tokens()   ← "能从外部 KV store 命中多少 token？"
  2. update_state_after_alloc()     ← "block 分好了，记录下来"
  3. build_connector_meta()         ← "生成这一步的传输指令"

Worker 侧在 forward pass 中依次调用：
  4. start_load_kv()                ← "开始异步拉取（非阻塞）"
  5. [逐层 attention 计算...]
  6. save_kv_layer()                ← "每层算完后异步保存（非阻塞）"
  7. wait_for_save()                ← "等所有层保存完成（阻塞点）"

请求结束时：
  8. request_finished()             ← "决定是否触发 P2P 发送"
```

**学习任何文件，都要对应到这 8 步里的某一步。**

---

## 二、文件总览

### Native KV Cache 层（前置知识）

| 文件 | 行数 | 核心职责 |
|------|------|----------|
| `v1/core/kv_cache_utils.py` | 1700+ | `KVCacheBlock`、`BlockHash`、`FreeKVCacheBlockQueue`（LRU 双向链表）|
| `v1/core/block_pool.py` | 511 | Block 分配/驱逐/prefix cache 命中查找 |
| `v1/core/kv_cache_manager.py` | 553 | 调度器与 block pool 的高层接口 |
| `v1/simple_kv_offload/manager.py` | 740 | Native CPU offload 调度侧（eager/lazy 模式）|
| `v1/simple_kv_offload/worker.py` | 350+ | Native CPU offload Worker 侧（pinned mem + CUDA stream）|
| `v1/kv_offload/abstract.py` | ~163 | Native offload 抽象接口（与 KVConnectorBase_V1 对比）|

### Mooncake 层

| 文件 | 行数 | 所属模式 | 核心职责 |
|------|------|----------|----------|
| `vllm/config/kv_transfer.py` | 123 | 通用 | 配置数据结构 KVTransferConfig |
| `kv_connector/v1/base.py` | 663 | 通用 | 抽象基类，定义所有接口契约 |
| `mooncake/mooncake_utils.py` | 127 | P2P | Bootstrap 服务，worker 注册/发现 |
| `mooncake/mooncake_connector.py` | 1661 | P2P | P2P 传输完整实现（Scheduler+Worker）|
| `mooncake/mooncake_store_data.py` | 262 | Store | 数据结构定义（PoolKey/ReqMeta/LoadSpec）|
| `mooncake/mooncake_store_connector.py` | 214 | Store | Store 模式外观类，委托给 Scheduler/Worker |
| `mooncake/mooncake_store_scheduler.py` | 404 | Store | Store 模式调度侧实现 |
| `mooncake/mooncake_store_worker.py` | 1040 | Store | Store 模式 Worker 侧（收发线程）|
| `scripts/mooncake/README.md` | 189 | 通用 | 部署和 benchmark 指南 |
| `scripts/mooncake/mooncake_config.json` | 8 | 通用 | Mooncake store 配置 |
| `tests/.../test_mooncake_connector.py` | 757 | P2P | P2P 模式单元测试 |
| `tests/.../test_mooncake_store_worker.py` | 348 | Store | Store Worker 单元测试 |

---

## 三、详细阅读顺序

### 第零步：GPU KV Cache 基本单元（1 小时，必读）

这是所有后续内容的地基。Mooncake 里的 `block_ids`、`block_hashes`、`allocate_slots` 都直接来自这一层，不懂这层就看不懂 `ReqMeta` 里的字段从哪来。

#### 0.1 精读 `v1/core/kv_cache_utils.py`（聚焦 Lines 49-156）

**文件路径**：`vllm/vllm/v1/core/kv_cache_utils.py`

```
Lines 49-68   BlockHashWithGroupId
              = BlockHash(bytes) + group_id.to_bytes(4, "big")
              含义：prefix cache 里每个 block 的唯一标识
              group_id 区分"全量 attention"与"滑动窗口 attention"两种 cache 组

Lines 109-156 KVCacheBlock (dataclass, slots=True)
              关键字段：
              - block_id: int           ← GPU 内存里的 page 索引（0~N-1）
              - ref_cnt: int            ← 当前有多少个请求在用这个 block
              - _block_hash: BlockHashWithGroupId | None
                                        ← 内容哈希，None=还没算完/已被驱逐
              - prev_free_block / next_free_block
                                        ← 在 LRU 双向链表里的前后指针
              - is_null: bool           ← 占位 block（滑动窗口窗口外不存储）

Lines 158-299 FreeKVCacheBlockQueue（LRU 双向链表）
              - popleft()    ← 驱逐时从头部取（最近最少使用）
              - append()     ← 释放时追加到尾部（变成最近使用）
              - remove()     ← O(1) 从链表中摘除（前缀命中时复用）
              核心不变式：ref_cnt == 0 的 block 才在此队列里
```

**边读边做**：画出一个 block 的生命周期：
```
分配（get_new_blocks）
  → ref_cnt++ → 从 free_queue 移出
  → [请求使用中]
  → ref_cnt-- → ref_cnt==0 时 append 到 free_queue 尾部
  → [等待驱逐，可被 prefix cache 命中复用]
  → popleft() 驱逐 → _block_hash 清空 → 重新分配给新请求
```

---

#### 0.2 精读 `v1/core/block_pool.py`（聚焦 Lines 183-423）

**文件路径**：`vllm/vllm/v1/core/block_pool.py`

```
Lines 33-127  BlockHashToBlockMap
              dict: BlockHashWithGroupId → KVCacheBlock
              作用：prefix cache 命中查找表
              get_one_block(hash) → block 或 None

Lines 183-208 get_cached_block(block_hash, kv_cache_group_ids)
              → 所有 group 都命中则返回 block 列表，否则 None
              含义：这是 prefix cache 的"查"操作

Lines 210-318 cache_full_blocks(...)
              → 将刚算完的 full block 写入 BlockHashToBlockMap
              触发时机：allocate_slots() 结束后，block 已填满 block_size 个 token
              含义：这是 prefix cache 的"存"操作

Lines 320-350 get_new_blocks(num_blocks) → list[KVCacheBlock]
              核心逻辑：
              1. 先从 free_queue.popleft_n(num) 取 LRU block
              2. 若有 _block_hash → 从 BlockHashToBlockMap 删除（驱逐）
              3. 重置 block，返回给调用方
              这里是 LRU 驱逐真正发生的地方

Lines 409-423 free_blocks(ordered_blocks)
              → ref_cnt -= 1，归零时 append 到 free_queue 尾部
```

---

#### 0.3 精读 `v1/core/kv_cache_manager.py`（聚焦 Lines 176-450）

**文件路径**：`vllm/vllm/v1/core/kv_cache_manager.py`

```
Lines 176-255 get_computed_blocks(request) → (blocks, num_cached_tokens)
              含义：查 prefix cache，返回已缓存的 block 列表和 token 数
              → 这是 Mooncake 里 get_num_new_matched_tokens() 的"本地版本"
              → Mooncake 在此基础上再加一层：去远端 store 查更多命中

Lines 257-427 allocate_slots(request, num_new_tokens, ...)
              含义：分配新 block，返回 KVCacheBlocks 对象
              → 这个返回值就是 update_state_after_alloc(request, blocks, ...) 里的 blocks 参数
              → blocks.get_unhashed_block_ids() 拿到的 block_id 列表，
                就是后来 ReqMeta.block_ids 的来源

Lines 429-450 free(request_id)
              → 调用 free_blocks()，ref_cnt 归零
              → 这就是 request_finished() 里 delay_free=True 时推迟的操作
```

**读完后能回答的问题：**
- `block_id` 是什么？（GPU tensor 的 page 索引，0~N-1 的整数）
- GPU 内存里一个 block 的物理大小？（`2 × block_size × num_kv_heads × head_size × dtype_size` 字节）
- 为什么 prefix cache 命中的 block 不需要重算？（block 的内容由 token ids 哈希确定，相同 token 序列 → 相同 hash → 复用同一个 block）
- `allocate_slots` 之后 Mooncake 的 `update_state_after_alloc` 收到什么？（`KVCacheBlocks` 对象，内含 `block_ids: list[int]`）

---

### 第零B步：Native CPU Offload——读对照组（45 分钟，选读）

不需要深读，**以对比为目的**，理解 Mooncake 要解决哪些 native 方案的局限。

#### 0B.1 阅读 `v1/simple_kv_offload/manager.py`（重点三段）

**文件路径**：`vllm/vllm/v1/simple_kv_offload/manager.py`

```
Lines 67-150  SimpleCPUOffloadScheduler.__init__
              关键：维护独立的 cpu_block_pool（与 GPU block_pool 平行）
              CPU pool 容量 = cpu_mem_size_gib / page_size

Lines 375-441 _prepare_lazy_store_specs()  ← Lazy 模式
              逻辑：用游标扫描 GPU free_queue
              → 发现快被驱逐的 cached block（有 hash）
              → 提前异步搬到 CPU，避免直接丢失

Lines 443-579 _prepare_eager_store_specs() ← Eager 模式
              逻辑：请求的每个 full block 刚算完就立刻异步存 CPU
              → 不等驱逐，主动 offload
              → 需要 CPU pool 有空间才行（否则跳过）
```

#### 0B.2 阅读 `v1/simple_kv_offload/worker.py`（Lines 25-100）

**文件路径**：`vllm/vllm/v1/simple_kv_offload/worker.py`

```
Lines 25-100  初始化
              - 分配 pinned CPU tensor（page-locked，支持高速 DMA）
              - 两条 CUDA stream：load_stream / store_stream（互不阻塞）
              - 每个 store/load event 预分配 CUDA Event 对象
```

#### 0B.3 阅读 `v1/kv_offload/abstract.py`（Lines 69-163）

**文件路径**：`vllm/vllm/v1/kv_offload/abstract.py`

```
Lines 69-163  OffloadingManager 抽象接口
              - lookup(block_hashes) → int | None   ← 查询 offload 命中
              - prepare_load(block_hashes)           ← 准备加载
              - prepare_store(block_hashes)          ← 准备存储
              - complete_load / complete_store       ← 完成回调
              - touch(block_hashes)                  ← 更新 LRU 时间戳
```

**对比表：读完后填写**

| 维度 | Native simple offload | Mooncake Store |
|------|----------------------|----------------|
| 存储介质 | 本机 pinned CPU 内存 | 分布式共享 CPU 内存池（多机）|
| 传输机制 | `cudaMemcpyAsync`（CUDA DMA）| RDMA |
| 去重 | 无 | `batch_is_exist` 跳过已存在 block |
| 跨实例共享 | 不支持 | 支持（任何实例的 prefix cache 对所有人可见）|
| 调度接口 | `SimpleCPUOffloadScheduler`（内嵌）| `KVConnectorBase_V1`（标准化）|
| 驱逐策略 | LRU（随 GPU block_pool）| LRU（Master 控制，TTL=30s，高水位=95%）|

---

### 概念对应关系（读完零和零B后填写）

```
Native 概念                        Mooncake Store 对应代码位置
─────────────────────────────────────────────────────────────────
KVCacheBlock.block_id          →   ReqMeta.block_ids
                                   mooncake_store_data.py:165

BlockHash（token 内容哈希）     →   PoolKey.chunk_hash
                                   mooncake_store_data.py:32

allocate_slots() 返回的 blocks →   update_state_after_alloc() 的 blocks 参数
                                   mooncake_store_scheduler.py:106

vLLM prefix cache 命中 token 数 →  LoadSpec.vllm_cached_tokens
                                   mooncake_store_data.py:130

Store 命中 token 数             →  LoadSpec.kvpool_cached_tokens
                                   mooncake_store_data.py:130

kv_cache_manager.free(req_id)  →   request_finished() 返回 True → 延迟调用
                                   mooncake_store_scheduler.py:345
                                   scheduler.py:1831

block_pool.get_new_blocks LRU  →   Mooncake Master 的 eviction_high_watermark + LRU
                                   start_mooncake_master.sh:55-57
```

---

### 第一步：建立配置认知（30 分钟）

#### 1.1 阅读 `scripts/mooncake/README.md`

**目标**：搞清楚这套系统干什么，怎么部署

重点关注：
- 两种 connector 的使用场景区别（P2P vs Store）
- 启动命令格式（`--kv-connector`, `--kv-role`）
- 网络要求（RDMA vs TCP）

---

#### 1.2 阅读 `vllm/config/kv_transfer.py`（全文 123 行）

**文件路径**：`vllm/vllm/config/kv_transfer.py`

关键代码段：

```
Lines 11-13  类型别名：KVProducer / KVConsumer / KVRole
Lines 22-61  KVTransferConfig 字段定义
             - kv_connector: str         ← 选择哪种 connector
             - kv_role: str              ← "kv_producer" / "kv_consumer" / "kv_both"
             - engine_id: str            ← 本实例的唯一 UUID
             - kv_connector_extra_config ← connector 专属配置
Lines 93-107 __post_init__() 验证逻辑
Lines 110-119 is_kv_producer / is_kv_consumer 属性
```

**边读边做**：在纸上记录 `kv_role` 三种取值的含义，后面所有代码都会用到。

---

#### 1.3 阅读 `scripts/mooncake/mooncake_config.json`（全文 8 行）

```json
{
    "metadata_server": "http://127.0.0.1:8080/metadata",
    "master_server_address": "127.0.0.1:50051",
    "global_segment_size": "600GB",   // CPU 内存池大小
    "local_buffer_size": "4GB",       // 本地 staging buffer
    "protocol": "rdma",               // rdma 或 tcp
    "device_name": ""                 // 留空=自动检测 RDMA 网卡
}
```

---

### 第二步：理解接口契约（45 分钟）

#### 2.1 精读 `kv_connector/v1/base.py`（全文 663 行）

**文件路径**：`vllm/vllm/distributed/kv_transfer/kv_connector/v1/base.py`

**分三轮读**：

**第一轮（10 min）：只看 Scheduler 侧抽象方法**

```
Lines 450-482  get_num_new_matched_tokens(request, num_computed_tokens)
               → 返回 (int|None, bool)
               含义：(能从外部 cache 命中的 token 数, 是否异步加载)
               None = 本实例处理，不走外部传输
               int  = 从外部加载这么多 token 的 KV

Lines 485-503  update_state_after_alloc(request, blocks, num_external_tokens)
               含义：块分配完成后，更新 connector 内部状态

Lines 506-518  build_connector_meta(scheduler_output) → KVConnectorMetadata
               含义：生成这个 step 的传输指令，下发给 Worker

Lines 530-549  request_finished(request, block_ids) → (bool, dict|None)
               含义：请求结束，决定是否要异步发送 KV 给其他实例
               返回 True = 需要异步发送
```

**第二轮（10 min）：只看 Worker 侧抽象方法**

```
Lines 257-265  register_kv_caches(kv_caches: dict[str, Tensor])
               含义：Worker 初始化时，注册所有 layer 的 KV cache tensor 地址到 RDMA

Lines 298-314  start_load_kv(forward_context, **kwargs)
               含义：开始异步加载（非阻塞），返回立即

Lines 316-328  wait_for_layer_load(layer_name: str)
               含义：等待特定层的 KV 加载完成（可以 pipeline 异步）

Lines 330-350  save_kv_layer(layer_name, kv_layer, attn_metadata, **kwargs)
               含义：每层 attention 计算完毕后，异步保存该层 KV

Lines 352-361  wait_for_save()
               含义：等待所有层 KV 保存完成（真正的阻塞点）
```

**第三轮（5 min）：理解元数据传递机制**

```
Lines 140-146  KVConnectorMetadata (抽象基类)
               → 每个 step 从 Scheduler 传给 Worker 的"指令包"

Lines 149-168  KVConnectorWorkerMetadata
               → Worker 聚合多个 TP rank 的数据用
               → aggregate() 方法合并多 rank 信息

Lines 217-255  bind/clear/get_connector_metadata()
               → Worker 每 step 从这里取出 Scheduler 下发的元数据
```

**边读边做**：画出 Scheduler→Worker 数据流：
```
Scheduler                    Worker
    │                           │
    │  build_connector_meta()   │
    │ ──────────────────────>   │  (KVConnectorMetadata)
    │                           │
    │                           │  bind_connector_metadata()
    │                           │  ↓
    │                           │  start_load_kv()  ← 内部取元数据
```

---

### 第三步：P2P 模式——从小到大（2 小时）

#### 3.1 阅读 `mooncake_utils.py`（全文 127 行）

**文件路径**：`vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_utils.py`

这是最短的文件，但理解它是读 P2P 模式的前提。

```
Lines 14      WorkerAddr = str  ← 就是 "ip:port" 字符串

Lines 19-24   RegisterWorkerPayload (Pydantic Model)
              字段：engine_id, dp_rank, tp_rank, pp_rank, addr
              含义：每个 Worker 启动时向 Bootstrap 服务注册自己的地址

Lines 27-31   EngineEntry (dataclass)
              结构：{tp_rank: {pp_rank: worker_addr}}
              含义：一个 engine 里所有 (tp, pp) 组合的地址映射

Lines 34-127  MooncakeBootstrapServer
              ├─ Lines 40-48   __init__: FastAPI app + 注册表 dict
              ├─ Lines 58-70   start(): 在独立线程里跑 uvicorn
              ├─ Lines 80-123  register_worker(): POST /register 处理
              │                核心：engine_id → {tp_rank → {pp_rank → addr}}
              └─ Lines 125-126 query(): GET /query?engine_id=xxx 返回地址表
```

**关键概念**：Bootstrap 服务只在 Producer 侧运行（Prefiller），Consumer 在 start_load_kv 时查询它，从而知道每个 TP rank 的 Prefiller 地址。

---

#### 3.2 精读 `mooncake_connector.py`（全文 1661 行，分 5 节读）

**文件路径**：`vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py`

---

**第 3.2.1 节：工具函数与数据结构（Lines 1-329，40 min）**

```
Lines 1-65    imports — 注意引入了 zmq, msgspec, httpx，是系统通信基础

Lines 67-68   类型别名
              ReqId = str     ← 调度器内部的请求 ID
              TransferId = str ← KV 传输的协调 ID（通常是 decode 侧生成的 UUID）

Lines 71-75   TransferRegion (dataclass)
              字段：src_ptr, dst_ptr, length, is_driver_rank
              含义：一次 KV block 传输的"源-目的-长度"三元组

Lines 78-96   _get_tp_ratio(local_tp, remote_tp) → (int, int, bool)
              含义：计算本地 TP 和远端 TP 的比值（支持异构 TP）
              例：local=8, remote=4 → ratio=(2,1), is_sender_larger=True

Lines 99-127  _expand_transfer_regions(...)
              含义：把 1:N 或 N:1 的 TP 映射展开为具体传输列表

Lines 130-164 _compute_sender_transfer_plan(...)
              含义：决定每个 sender rank 负责传输哪些 receiver rank 的数据
              关键：异构 TP 下，一个 sender 可能要负责多个 receiver 的部分数据

Lines 167-180 _can_coalesce_block_transfers(...)
              含义：判断相邻两个 block 是否可以合并成一次传输（内存连续性检查）

Lines 183-234 _validate_asymmetric_region_lengths(...)
              含义：校验异构 TP 下传输区域长度是否一致

Lines 244-273 MooncakeXferMetadata / MooncakeXferResponse (msgspec.Struct)
              含义：通过 ZMQ 传递的"拉取请求"消息格式
              MooncakeXferMetadata 字段：
              - transfer_id: str
              - dst_ptrs: list[int]    ← Consumer 侧 GPU 内存地址
              - lengths: list[int]
              - dst_tp_rank: int
              - dst_pp_rank: int

Lines 276-298 PullReqMeta / SendBlockMeta (dataclass)
              PullReqMeta：Consumer 侧记录"我要拉取谁的哪些 block"
              SendBlockMeta：Producer 侧记录"我有哪些 block 要发送，ready 没"

Lines 301-327 MooncakeConnectorMetadata
              ├─ reqs_to_recv: dict[EngineId, dict[ReqId, PullReqMeta]]
              │  Consumer 要从哪些 Prefiller 拉取哪些请求
              └─ reqs_to_send: dict[ReqId, (TransferId, list[int])]
                 Producer 这步要发送哪些请求的 block
```

**边读边做**：在纸上画 PullReqMeta 和 SendBlockMeta 的关系：
```
Consumer Scheduler        Producer Scheduler
PullReqMeta               SendBlockMeta
  transfer_id ────────────→ transfer_id (匹配！)
  remote_engine_id          (属于这个 engine)
  local_block_ids           local_block_ids (对应 block)
```

---

**第 3.2.2 节：MooncakeConnector 外观类（Lines 330-440，15 min）**

```
Lines 330-350 __init__
              根据 kv_role 决定创建 Scheduler 实例还是 Worker 实例（或两者）

Lines 352-368 get_required_kvcache_layout() (classmethod)
              返回 "NHD" 布局要求（N=num_blocks, H=num_heads, D=head_dim）

Lines 374-403 Scheduler 委托方法（每个方法只有 1-2 行，全部委托给 _scheduler）
              get_num_new_matched_tokens → self._scheduler.get_num_new_matched_tokens
              update_state_after_alloc  → self._scheduler.update_state_after_alloc
              build_connector_meta      → self._scheduler.build_connector_meta
              request_finished          → self._scheduler.request_finished

Lines 408-439 Worker 委托方法（全部委托给 _worker）
              register_kv_caches → self._worker.register_kv_caches
              start_load_kv      → self._worker.start_load_kv
              save_kv_layer      → self._worker.save_kv_layer
              wait_for_save      → self._worker.wait_for_save
```

**关键理解**：`MooncakeConnector` 只是一个外观（Facade），真正逻辑全在 `MooncakeConnectorScheduler` 和 `MooncakeConnectorWorker` 里。

---

**第 3.2.3 节：MooncakeConnectorScheduler（Lines 442-636，30 min）**

```
Lines 445-464 __init__
              关键字段：
              - reqs_need_send: dict[TransferId, SendBlockMeta]
                "已完成 prefill，等待被 Consumer 拉取的 block 集合"
              - active_send_meta: dict[ReqId, SendBlockMeta]
                "正在发送中的请求"
              - reqs_to_recv: dict[ReqId, PullReqMeta]
                "Consumer 侧：正在等待接收的请求"

Lines 466-504 get_num_new_matched_tokens(request, num_computed_tokens)
              ┌─ Producer 侧：返回 (None, False) — 自己做 prefill，不需要外部 KV
              └─ Consumer 侧：
                 - 从 request.kv_transfer_params 拿到 transfer_id, remote_engine_id
                 - 返回 (所有 prompt token 数, True)
                   ↑ True = 这些 token 的 KV 会异步从 Producer 拉取

Lines 506-542 update_state_after_alloc(request, blocks, num_external_tokens)
              ┌─ Producer 侧：不做任何事
              └─ Consumer 侧：
                 - 把 (transfer_id, local_block_ids) 存入 reqs_to_recv
                 - 记录 remote_engine_id 和 bootstrap_addr

Lines 552-582 build_connector_meta(scheduler_output)
              ┌─ Producer 侧：把 reqs_need_send 里准备好的请求放入 metadata
              └─ Consumer 侧：把 reqs_to_recv 里所有待接收请求放入 metadata
              → 返回 MooncakeConnectorMetadata（下发给 Worker）

Lines 584-635 request_finished(request, block_ids)
              ┌─ Consumer 侧：清理 reqs_to_recv，返回 (False, None)
              └─ Producer 侧：
                 - 把完成的请求加入 reqs_need_send（transfer_id → block_ids）
                 - 返回 (True, {"transfer_id": ..., "block_ids": ...})
                   ↑ True = 通知 Worker 这些 block 准备好发送了
```

---

**第 3.2.4 节：MooncakeConnectorWorker 初始化（Lines 638-820，20 min）**

```
Lines 641-770 __init__
              重点理解初始化顺序：

              Lines 651-672  初始化 Mooncake TransferEngine
                             engine = MooncakeTransferEngine(config)
                             ← 这是 C++ RDMA 引擎的 Python 绑定

              Lines 674-695  初始化 ZMQ Sender Socket（Producer 侧）
                             socket = zmq.Context().socket(ROUTER)
                             ← ROUTER socket 可以同时接收多个 Consumer 的请求

              Lines 697-720  初始化 ZMQ Receiver（Consumer 侧）
                             ← DEALER socket，主动连接到 Producer 的 ROUTER

              Lines 722-745  启动 Bootstrap Server（仅 Producer 侧）
                             bootstrap_server = MooncakeBootstrapServer()
                             bootstrap_server.start()

              Lines 747-770  初始化 TP 拓扑信息
                             local_tp_size, local_pp_size, 用于传输计划计算

Lines 791-819 register_worker_with_bootstrap(engine_id, addr)
              向 Producer 的 Bootstrap 服务注册自己（Consumer 侧调用）
              使用 httpx 发送 POST /register 请求
```

---

**第 3.2.5 节：发送与接收核心逻辑（Lines 821-1661，40 min）**

```
Lines 821-858 _mooncake_sender_listener() (async)
              Producer 侧的 ZMQ ROUTER 监听循环：
              - 接收 Consumer 发来的 MooncakeXferMetadata 消息
              - 提交到 sender thread pool 处理
              - 关键：这是一个 while True 循环，Producer 全程在线监听

Lines 860-900 _sender_worker() (async)
              从 queue 取出一个 Consumer 的拉取请求，调用 send_kv_to_decode

Lines 882-1051 send_kv_to_decode(transfer_id, xfer_meta, zmq_socket, identity)
               Producer 侧处理单个 Consumer 拉取请求的完整逻辑：
               Lines 900-950  等待对应 block 的 ready 事件
                              (需要等 Scheduler 通知 SendBlockMeta.ready.set())
               Lines 951-1000 调用 _build_transfer_params 构建传输参数
               Lines 1001-1040 调用 Mooncake engine 执行实际传输
                               engine.batch_transfer_sync_write(src_ptrs, dst_ptrs, lengths)
               Lines 1041-1051 发送响应给 Consumer（传输完成通知）

Lines 1063-1194 _build_transfer_params(transfer_id, xfer_meta)
                构建传输参数的核心：
                Lines 1080-1120 获取 Producer 侧 block 的 GPU 地址（src_ptrs）
                Lines 1121-1160 根据 TP 比例计算传输计划
                Lines 1161-1194 合并相邻连续 block（减少传输次数）

Lines 1215-1287 register_kv_caches(kv_caches)
                Worker 初始化时调用：
                - 遍历所有 layer 的 KV tensor
                - 调用 engine.register_memory_region(ptr, size, "cpu"/"gpu")
                - 注册后 Mooncake 可以对这些地址做 RDMA 操作

Lines 1322-1352 get_finished()
                每 step 调用，返回：
                - (sending_done_req_ids, None) 如果是 Producer
                - (None, receiving_done_req_ids) 如果是 Consumer
                Consumer 侧等待 receive_kv 完成后标记为 done

Lines 1354-1408 receive_kv_from_single_worker(engine_id, requests)
                Consumer 侧拉取 KV 的完整逻辑：
                Lines 1360-1390 向 Bootstrap 查询 Producer 各 rank 地址
                Lines 1391-1408 构建 MooncakeXferMetadata，发送给 Producer ZMQ

Lines 1462-1515 start_load_kv() / _start_load_kv()
                Consumer 侧：从 metadata 取出 reqs_to_recv，启动异步接收
                Producer 侧：从 metadata 取出 reqs_to_send，标记 block ready

Lines 1428-1432 save_kv_layer() / wait_for_save()
                Producer 侧：这两个方法在 P2P 模式下是空操作
                （P2P 模式里 Producer 不需要"保存"，Consumer 主动来拉）
```

---

### 第四步：Store 模式——数据结构驱动（2 小时）

Store 模式比 P2P 复杂，必须先读懂数据结构，再读逻辑。

#### 4.1 精读 `mooncake_store_data.py`（全文 262 行，30 min）

**文件路径**：`vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_data.py`

```
Lines 21-29   KeyMetadata (dataclass)
              字段：model_name, tp_rank, pcp_rank, dcp_rank, pp_rank
              含义：KV cache 存入 store 时的"维度标签"
              区分不同模型、不同 TP/PP 配置下的 KV cache

Lines 32-59   PoolKey (有序 dataclass)
              字段：key_metadata (KeyMetadata), chunk_hash (int/tuple)
              方法：
              - __hash__(): 基于所有字段的哈希（用作 dict key）
              - to_string(): 序列化为字符串（实际写入 store 的 key）
              含义：store 里每个 KV cache 条目的唯一标识符

Lines 62-127  ChunkedTokenDatabase
              这是将 KV blocks 转换为 store key 的核心工具：
              Lines 65-78   __init__: 存储 base_addr 和 block_len
              Lines 80-97   prepare_value(block_id) → (addr_list, size_list, block_id)
                            给定 block_id，返回 GPU 内存地址 + 大小
              Lines 99-127  process_tokens(token_ids, block_hashes) → iter(start, end, pool_key)
                            关键：将 token 序列按 block 分块，每块生成对应的 PoolKey
                            这个迭代器是 Send/Recv 线程的核心输入

Lines 130-137 LoadSpec (dataclass)
              字段：vllm_cached_tokens, kvpool_cached_tokens, can_load, token_len
              含义：调度器告诉 Worker "这个请求需要从 store 加载多少 token 的 KV"
              - vllm_cached_tokens: vLLM 本地 prefix cache 已命中多少
              - kvpool_cached_tokens: store 里命中多少
              - can_load: 是否可以异步加载

Lines 140-162 RequestTracker (dataclass)
              字段：req_id, token_len, allocated_block_ids, num_saved_tokens, token_ids
              方法：update() — 更新已保存的 token 数
              含义：Worker 侧追踪一个请求的保存进度

Lines 165-245 ReqMeta (dataclass)
              这是从 Scheduler 传给 Worker 的核心数据结构：
              字段：
              - req_id: str
              - token_len_chunk: int      ← 这个 chunk 里有多少 token
              - block_ids: list[int]      ← 对应哪些 KV block
              - block_hashes: list[BlockHash]  ← 每个 block 的内容哈希
              - can_save: bool            ← 是否要存入 store（Producer 侧）
              - load_spec: LoadSpec       ← 加载规格（Consumer 侧）
              Lines 182-245 from_request_tracker() 构造方法（复杂，先跳过）

Lines 248-261 MooncakeStoreConnectorMetadata
              字段：requests: list[ReqMeta], unfinished_request_ids, preempted_req_ids
              含义：每个 step 从 Scheduler 传给 Worker 的"任务包"
```

**边读边做**：画出 Store 模式的 key 结构：
```
一个 KV cache 条目的完整 key：
  PoolKey {
    key_metadata: {model="llama3-8b", tp_rank=0, pp_rank=0, ...}
    chunk_hash: 0xABCD1234  ← 这段 token 的内容哈希
  }
  → to_string() → "llama3-8b|tp0|pp0|0xABCD1234"
```

---

#### 4.2 阅读 `mooncake_store_connector.py`（全文 214 行，20 min）

**文件路径**：`vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_connector.py`

这是外观类，结构和 `MooncakeConnector` 类似。

```
Lines 43-73   MooncakeStoreKVEvents
              作用：聚合多个 Worker 进程的 KV cache 事件（BlockStored/BlockRemoved）
              Lines 49-50  add_events(): 添加事件列表
              Lines 52-57  aggregate(): 合并多 Worker 数据（去重）
              Lines 59-60  increment_workers(): 记录已汇报的 Worker 数

Lines 76-101  MooncakeStoreConnector.__init__
              关键：创建 MooncakeStoreScheduler 和 MooncakeStoreWorker 实例

Lines 107-163 Scheduler 委托方法（与 P2P 模式结构相同，全部委托给 _scheduler）

Lines 169-213 Worker 委托方法
              注意 save_kv_layer (Lines 181-189)：
              这里不是空操作！Store 模式每层算完就触发异步保存
              → self._worker.save_kv_layer(layer_name, kv_layer, attn_metadata)

Lines 203-213 get_kv_connector_kv_cache_events()
              返回 MooncakeStoreKVEvents 对象（供 vLLM 调度器知道哪些 block 已缓存）
```

---

#### 4.3 精读 `mooncake_store_scheduler.py`（全文 404 行，30 min）

**文件路径**：`vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_scheduler.py`

```
Lines 34-59   __init__
              关键字段：
              - lookup_client: LookupKeyClient  ← ZMQ 客户端，查询 store 命中情况
              - req_tracker: dict[str, RequestTracker]  ← 每个活跃请求的追踪器
              - load_specs: dict[str, LoadSpec]  ← 每个请求的加载规格

Lines 61-104  get_num_new_matched_tokens(request, num_computed_tokens)
              这是 Store 模式最复杂的方法：
              Lines 70-80   调用 lookup_client.lookup(block_hashes) 查询 store
                            → 返回 store 里命中了多少连续 prefix token
              Lines 81-95   计算 kvpool_cached_tokens（store 命中数）
              Lines 96-104  返回 (kvpool_cached_tokens - already_computed, True)
                            True = 这些 KV 需要异步从 store 加载

Lines 106-142 update_state_after_alloc(request, blocks, num_external_tokens)
              Lines 115-130 创建/更新 RequestTracker，记录分配的 block_ids
              Lines 131-142 记录 load_spec（加载规格），供 build_connector_meta 使用

Lines 144-343 build_connector_meta(scheduler_output)
              最长的方法（200 行），构建每个 step 的传输指令：
              Lines 160-200 遍历所有活跃请求
              Lines 201-250 对 Producer 侧：构建 can_save=True 的 ReqMeta
              Lines 251-300 对 Consumer 侧：构建含 load_spec 的 ReqMeta
              Lines 301-343 处理预占（preemption）和未完成请求
              → 返回 MooncakeStoreConnectorMetadata

Lines 345-363 request_finished(request, block_ids)
              清理 req_tracker 和 load_specs

Lines 366-392 LookupKeyClient
              Lines 369-378 __init__: 连接到 Worker 侧的 ZMQ 服务器
              Lines 380-388 lookup(block_hashes) → int
                            发送 ZMQ 请求，返回 store 里命中的 token 数
              Lines 390-391 close()

Lines 394-403 get_zmq_rpc_path_lookup()
              返回 ZMQ 地址（ipc:// Unix socket 路径）
```

---

#### 4.4 精读 `mooncake_store_worker.py`（全文 1040 行，45 min）

**文件路径**：`vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_worker.py`

**分三节读**：

**第 4.4.1 节：配置与工具函数（Lines 1-330，10 min）**

```
Lines 54-77   全局常量
              MOONCAKE_NO_AVAILABLE_HANDLE = -200  ← store 压力过大时的错误码
              DEFAULT_GLOBAL_SEGMENT_SIZE = 4GB
              DEFAULT_LOCAL_BUFFER_SIZE = 4GB

Lines 85-147  MooncakeStoreConfig (dataclass)
              Lines 110-134 from_file(): 从 mooncake_config.json 读取
              Lines 137-147 load_from_env(): 从环境变量读取

Lines 155-325 Disk Offload 相关工具函数（先跳过，只在测试时回来看）
              _get_disk_offload_buffer_budget_bytes() (Lines 155-167)
              _parse_size()                           (Lines 170-209)
              _split_disk_offload_load_batches()      (Lines 251-325)
```

**第 4.4.2 节：KVTransferThread 基类（Lines 333-432，10 min）**

```
Lines 333-432 KVTransferThread (threading.Thread 子类)
              这是 Send 和 Recv 线程的共同基类

              Lines 348-378 __init__
                            关键字段：
                            - request_queue: queue.Queue  ← 接收来自 Worker 的请求
                            - finished_requests: set      ← 已完成的请求 ID
                            - kv_events: list             ← KV cache 事件

              Lines 380-382 add_request(req_meta): 向队列添加请求
              Lines 384-392 get_and_clear_finished_requests(): 取出已完成请求 ID

              Lines 399-416 run() — 线程主循环
                            while True:
                              req = queue.get(timeout=1.0)
                              self._handle_request(req)   ← 子类实现

              Lines 418-420 _handle_request(req): 抽象方法，子类实现
              Lines 422-425 update_kv_event(event): 追加 KV 事件
```

**第 4.4.3 节：Send 和 Recv 线程（Lines 435-884，25 min）**

```
Lines 435-711 KVCacheStoreSendingThread (KVTransferThread 子类)
              处理"将 KV cache 保存到 store"的请求

              Lines 452-478 __init__
                            关键字段：
                            - store: MooncakeDistributedStore
                            - db: ChunkedTokenDatabase   ← token→地址 转换工具
                            - stored_requests: dict       ← 引用计数，防止提前释放

              Lines 509-512 _should_skip_request(req_meta) → bool
                            检查请求是否因为压力而需要跳过

              Lines 514-524 _mark_request_skipped_for_pressure(req_meta)
                            标记请求为"因压力跳过"，等压力释放后重试

              Lines 526-537 _clear_store_pressure()
                            当 store 恢复正常时，清除压力标记，重新处理跳过的请求

              Lines 539-710 _handle_request(req_meta) — 核心逻辑
                            Lines 545-580  遍历 req 里的所有 block hashes
                            Lines 581-620  跳过已存在于 store 的 block（去重！）
                            Lines 621-660  调用 store.batch_put_from_multi_buffers()
                                           ← 实际将 GPU KV cache 写入 store
                            Lines 661-700  处理 MOONCAKE_NO_AVAILABLE_HANDLE 错误
                                           触发 _mark_request_skipped_for_pressure()
                            Lines 701-710  更新 KV 事件（BlockStored）

Lines 713-884 KVCacheStoreRecvingThread (KVTransferThread 子类)
              处理"从 store 加载 KV cache 到 GPU"的请求

              Lines 729-755 __init__
                            关键字段：
                            - store: MooncakeDistributedStore
                            - db: ChunkedTokenDatabase
                            - disk_offload_budget: int  ← 限制单次 disk I/O 量

              Lines 757-884 _handle_request(req_meta) — 核心逻辑
                            Lines 763-800  收集所有需要加载的 (pool_key, block_id) 对
                            Lines 801-840  如果有 disk offload：按预算分批（_split_disk_offload_load_batches）
                            Lines 841-870  调用 store.batch_get_into_multi_buffers()
                                           ← 实际从 store 读取 KV cache 到 GPU
                            Lines 871-884  更新已加载的 token 数

Lines 892-1040 MooncakeStoreWorker — 整体 Worker 类
              Lines 911-994  __init__ 第一部分：初始化 rank 信息、ZMQ 服务器
              Lines 999-1026 __init__ 第二部分：初始化 MooncakeDistributedStore
                             store = MooncakeDistributedStore(config)
                             store.initialize()
```

---

### 第五步：单元测试——边测边理解（1 小时）

#### 5.1 运行基础测试

```bash
cd /home/aoshen/setup_new_cluster/vllm

# 运行全部 unit tests，确认环境正常
python -m pytest tests/v1/kv_connector/unit/ -v 2>&1 | tee test_baseline.log

# 只跑 P2P 测试
python -m pytest tests/v1/kv_connector/unit/test_mooncake_connector.py -v -s

# 只跑 Store Worker 测试
python -m pytest tests/v1/kv_connector/unit/test_mooncake_store_worker.py -v -s
```

#### 5.2 精读 `test_mooncake_connector.py`（按测试顺序读）

**文件路径**：`tests/v1/kv_connector/unit/test_mooncake_connector.py`

| 测试名 | 行号 | 学习目标 |
|--------|------|----------|
| `test_basic_interface` | 55-100 | 理解 Scheduler 调用序列：先 get_num_matched → update_alloc → build_meta |
| `test_prompt_less_than_block_size` | 103-138 | 边界情况：prompt 小于一个 block 时如何处理 |
| `test_bootstrap_server` | 152-209 | Bootstrap 服务注册/查询流程 |
| `test_scheduler_request_finished` | 212-242 | request_finished 的两种返回值 |
| `test_kv_producer` | 303-458 | **最重要**：完整的 Producer 发送流程，包含 ZMQ mock |
| `test_kv_consumer` | 461-523 | Consumer 接收流程 |
| `test_worker_get_finished_timeout` | 526-560 | 超时后 abort 行为 |
| `test_kv_producer_heterogeneous_tp` | 612-756 | 异构 TP 传输计划（参数化测试） |

**边读边做**：每读一个测试，在对应的实现代码里找到被测试的代码路径，用注释标注"此处被 test_xxx 覆盖"。

#### 5.3 精读 `test_mooncake_store_worker.py`（按测试顺序读）

**文件路径**：`tests/v1/kv_connector/unit/test_mooncake_store_worker.py`

| 测试名 | 行号 | 学习目标 |
|--------|------|----------|
| `test_store_sending_thread_skips_request_during_cpu_pressure` | 107-139 | Backpressure 机制：handle 不够时跳过请求 |
| `test_store_sending_thread_only_skips_on_no_available_handle` | 142-161 | 只有 -200 错误触发 skip，其他错误直接失败 |
| `test_recv_thread_uses_single_batch_when_no_disk_offload_budget` | 188-208 | 无 disk offload 时，所有 block 一次性加载 |
| `test_recv_thread_splits_disk_offload_loads_by_budget` | 241-281 | disk offload 按预算分批加载 |
| `test_recv_thread_stops_after_first_failing_disk_offload_sub_batch` | 284-300 | 第一批失败就停止，不继续加载 |

---

## 四、边改边学——三个递进实验

### 实验一：添加传输追踪日志（简单，30 min）

**目标**：理解每次 step 实际触发了哪些 KV 操作

在 `mooncake_store_scheduler.py` 的 `get_num_new_matched_tokens` 开头加：

```python
# Lines 61-62 之后插入：
import logging
_logger = logging.getLogger("mooncake_trace")

def get_num_new_matched_tokens(self, request, num_computed_tokens):
    _logger.debug(
        f"[TRACE] req={request.request_id[:8]}, "
        f"num_computed={num_computed_tokens}, "
        f"num_tokens={request.num_tokens}"
    )
    # ... 原有代码
```

运行时加 `VLLM_LOGGING_LEVEL=DEBUG` 观察每个请求的 token 命中情况。

**验证方式**：
```bash
VLLM_LOGGING_LEVEL=DEBUG python -m pytest tests/v1/kv_connector/unit/test_mooncake_connector.py::test_basic_interface -v -s 2>&1 | grep TRACE
```

---

### 实验二：修改超时阈值并观察 abort 行为（中等，1 小时）

**目标**：理解 `VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT` 超时机制

在 `mooncake_connector.py` 里找到超时相关逻辑（搜索 `ABORT_REQUEST_TIMEOUT`），将测试中的超时改小：

**验证方式**：运行 `test_worker_get_finished_timeout`（Lines 526-560），在测试里修改 `timeout` 参数，观察行为变化：
```bash
python -m pytest tests/v1/kv_connector/unit/test_mooncake_connector.py::test_worker_get_finished_timeout -v -s
```

---

### 实验三：理解 Backpressure 机制（较难，1-2 小时）

**目标**：修改 `_should_skip_request` 的判断条件，观察测试失败

在 `mooncake_store_worker.py` Lines 509-512：

```python
# 原始代码：
def _should_skip_request(self, req_meta):
    return req_meta.req_id in self._skipped_for_pressure

# 实验：改成基于队列深度
def _should_skip_request(self, req_meta):
    if self.request_queue.qsize() > 10:  # 队列积压超过 10 个
        return True
    return req_meta.req_id in self._skipped_for_pressure
```

然后运行：
```bash
python -m pytest tests/v1/kv_connector/unit/test_mooncake_store_worker.py::test_store_sending_thread_skips_request_during_cpu_pressure -v -s
```

观察测试是否还能通过，理解原有设计的意图。

---

## 五、常见概念对照表

| 概念 | P2P 模式 | Store 模式 |
|------|----------|------------|
| 传输标识 | `TransferId`（UUID） | `BlockHash`（内容哈希） |
| 命中判断 | 无（Producer 必须先完成 prefill） | ZMQ 查询 store 命中 token 数 |
| 去重 | 无（每请求独立传输） | 有（相同 hash 的 block 只存一份） |
| 传输方向 | Producer → Consumer（Push 触发 Pull） | 双向（put/get） |
| 控制面通信 | ZMQ ROUTER/DEALER + Bootstrap HTTP | ZMQ（lookup RPC） |
| 数据面传输 | Mooncake TransferEngine（RDMA） | MooncakeDistributedStore（RDMA） |
| 适用场景 | Disaggregated Serving（分离部署） | Multi-turn / Prefix Sharing |

---

## 六、调试速查

### 查看哪些 env var 影响行为

```bash
grep -r "VLLM_MOONCAKE\|MOONCAKE_" vllm/vllm/ --include="*.py" | grep "os.environ\|getenv"
```

### 运行单个测试（带详细日志）

```bash
cd /home/aoshen/setup_new_cluster/vllm
python -m pytest tests/v1/kv_connector/unit/test_mooncake_connector.py::test_kv_producer -v -s --log-cli-level=DEBUG
```

### 查看 P2P 模式的请求状态流转

```bash
grep -n "reqs_need_send\|reqs_to_recv\|active_send_meta" \
  vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py
```

### 查看 Store 模式的 KV 事件触发点

```bash
grep -n "BlockStored\|BlockRemoved\|update_kv_event" \
  vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_worker.py
```

---

## 七、学习进度检查

完成以下每项，说明对应阶段已掌握：

- [ ] 能用一句话说清楚 P2P 和 Store 模式的核心区别
- [ ] 能从 `base.py` 里背出 8 个核心方法的名字和职责
- [ ] 能解释 `MooncakeXferMetadata` 为什么要传 `dst_ptrs` 而不是 block_ids
- [ ] 能解释 Store 模式的 `PoolKey.to_string()` 里为什么要包含 tp_rank
- [ ] 运行 `test_kv_producer` 不报错
- [ ] 运行 `test_store_sending_thread_skips_request_during_cpu_pressure` 不报错
- [ ] 能解释 `_should_skip_request` 和 `_clear_store_pressure` 的协作关系
- [ ] 能在代码里找到 `batch_transfer_sync_write` 的调用点，说清楚它的参数从哪来
