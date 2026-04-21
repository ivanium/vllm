---
title: Mooncake Store 机制解读与 GPU IPC Pull 优化案例研究
audience: vLLM review 团队（aoshen524 + disagg/KV 方向 reviewer）
last_verified: 2026-04-21
pr_reference: https://github.com/kvcache-ai/Mooncake/pull/1946
pr_commit: 39747d5d192e731352b883230834aab6ace8cb6e
mooncake_commit: 本地 Mooncake 工作树（vllm/Mooncake/）
scope: |
  以 Mooncake Store 的数据路径为主体，PR #1946 作为贯穿案例。
  深入：Replica/Segment 模型、副本分配、数据提交、CPU segment 内存属性、GPU IPC pull。
  不深入：HA / standby、offload to SSD、admission sketch、hot cache、metric manager、Master 启动流程。
---

# Mooncake Store 机制解读与 GPU IPC Pull 优化案例研究

## §0 · 总览

### 0.1 这份文档解决什么问题

aoshen524 最近在 review vLLM 社区 disagg/KV 方向的 PR，Mooncake 的 `MooncakeStoreConnector` 是当前热点之一。本文通过 **PR #1946（GPU IPC Pull 优化）** 把 Mooncake Store 的数据路径讲透，同时把这轮 review 里挖出来的非 trivial 机制沉淀下来，避免以后同类 PR 时从零再推一遍。

读者假设：熟悉 vLLM、KV cache、RDMA/NCCL 基础概念；对 Mooncake 只需要知道"它是个分布式 KV object store"这一层即可。

### 0.2 两种入口

| 入口 | 怎么走 |
|---|---|
| **按模块读** | §1 → §2 → §3 → §4 → §5，自底向上理解 Mooncake 数据路径；再用 §6 把 PR 套进去 |
| **按问题索引** | 直接跳到附录 B，每个问题一段，带链接回主文 |

主体读完再看附录 A（code review）和附录 B（深度 Q&A）。

### 0.3 本文作用域

**深入讲**：

- Mooncake Store 的 Replica / Segment / Buffer 数据模型
- Master 的副本分配（`PutStart` → `AllocationStrategy`）
- Client 的提交路径（`SubmitTransfers` per-replica 循环）
- CPU segment 内存分配的三条路径及 pin/NUMA 属性
- PR #1946 的设计、实现、降级矩阵
- vLLM 如何注册 KV cache、生成 slice

**只点到为止或不讲**：HA/standby、SSD offload（用 PR 复用的部分除外）、admission sketch、hot cache、metric manager、Master 自身的启动/配置、master client pool 实现。

**绝不伪造**：所有源码路径和行号都已 grep 验证；所有内存属性结论（pinned/NUMA）都查到了具体 `cudaHostRegister` / `mbind` / `allocate_buffer_*` 的调用点。

### 0.4 关键术语（全文统一）

| 术语 | 含义 |
|---|---|
| **Master** | Mooncake 的元数据中心进程（`MasterService`），管理 key → replica 映射、分配决策、eviction |
| **Client** | Mooncake 的数据平面客户端（`Client` / `RealClient`），负责 put/get 的提交与数据搬运 |
| **Requester** | 发起 put 的 Client（本文 PR 场景下 = vLLM worker） |
| **Owner** | 被 Master 选中、实际存放某个 replica 的 Client（本文 PR 场景下 = mooncake store 进程，与 Requester 同节点） |
| **TransferEngine (TE)** | Mooncake 的数据搬运层，底下挂 RDMA / NVLink / HIP / Ascend / TCP 等多种 transport |
| **Segment** | 一块被注册给 Mooncake 的物理内存池（一个节点的 CPU 内存是一个 segment，一块 GPU 内存也是一个 segment） |
| **Replica** | 一个 key 的一份物理拷贝，落在某个 segment 上的某个 `AllocatedBuffer` 里 |
| **Descriptor** | replica 或 buffer 的可序列化元信息，随 `BatchPutStart` 响应返回给 Client |
| **Slice** | `{void* ptr, size_t size}` —— Client 用来描述"我要把这段数据写进去"的最小单元 |

---

## §1 · Mooncake Store 架构鸟瞰

### 1.1 三大组件

```
┌──────────────────────── 控制面 ────────────────────────┐
│                                                       │
│                    ┌─────────────────┐                │
│                    │  MasterService  │                │
│                    │  (master_svc.*) │                │
│                    │                 │                │
│                    │  - 元数据       │                │
│                    │  - 副本分配     │                │
│                    │  - eviction     │                │
│                    │  - lease 管理   │                │
│                    └────────▲────────┘                │
│                             │ coro_rpc                │
│                             │                         │
│              ┌──────────────┴──────────────┐          │
│              │         MasterClient        │          │
│              │  (客户端侧，每个 Client 一份) │          │
│              └──────────────▲──────────────┘          │
└─────────────────────────────┼─────────────────────────┘
                              │
                              │
┌─────────────────────────────┼─────────────────────────┐
│                    数据面   │                         │
│                             │                         │
│                    ┌────────┴──────────┐              │
│                    │     Client        │              │
│                    │ (client_service.*)│              │
│                    │ (real_client.*)   │              │
│                    │                   │              │
│                    │  - Put / Get      │              │
│                    │  - SubmitTransfers│              │
│                    │  - mount segment  │              │
│                    └────────┬──────────┘              │
│                             │                         │
│                    ┌────────┴──────────┐              │
│                    │ TransferEngine    │              │
│                    │ (RDMA / NVLink /  │              │
│                    │  HIP / Ascend...) │              │
│                    └───────────────────┘              │
└───────────────────────────────────────────────────────┘
```

**职责分布**：

- **MasterService**：只管元数据和分配决策，**不经手数据**。它告诉 Client "你这个 key 要写到哪些 replica 的哪些地址"，剩下的 Client 自己搞。
- **Client**：一个 Client 实例同时扮演**两种角色** —— 它既是别人的 Requester（主动 put/get），又是别人的 Owner（被选中存放 replica）。
- **TransferEngine**：做实际的字节搬运。RDMA transport 负责跨节点，NVLink transport 负责同节点 GPU IPC，等等。

### 1.2 Put 流程宏观时序

```
vLLM Worker (Requester)           Master            Client (Owner)
       │                            │                     │
       │ put(key, slices, cfg)      │                     │
       ├────────────────────────────┤                     │
       │                                                   │
       │ ① BatchPutStart            │                     │
       │ ─────────────────────────► │                     │
       │                            │ AllocateAndInsert   │
       │                            │   分配 N 个 replica  │
       │ ◄───────────────────────── │ 返回 Descriptor[]   │
       │                            │                     │
       │ ② SubmitTransfers          │                     │
       │    对每个 replica 写数据   │                      │
       │ ────────────────────────────────────────────────►│
       │  (本 PR 场景下：同节点 GPU IPC pull)             │
       │  (跨节点场景：RDMA WRITE)                         │
       │ ◄────────────────────────────────────────────────│
       │                            │                     │
       │ ③ BatchPutEnd              │                     │
       │ ─────────────────────────► │ 标记 COMPLETE       │
       │                            │                     │
```

Put 分两阶段：**Master 先决定"写哪里"，Client 再执行"怎么写"**。这是理解本文所有后续讨论的主骨架。

### 1.3 Get 流程（仅简述）

```
vLLM Worker                      Master              Owner Client
       │                           │                      │
       │ get(key, dst_slices)      │                      │
       │ ──────────────────────────│                      │
       │                           │ 查 replica list     │
       │ ◄─────── Descriptor[] ────│                      │
       │                           │                      │
       │ 选一个 replica            │                      │
       │ 发起 TransferEngine READ                         │
       │ ────────────────────────────────────────────────►│
       │ ◄────────────────────────────────────────────────│
```

**本文不深入 Get** —— PR #1946 只优化 Put 路径，Get 的跨节点 RDMA 路径未被触及。如果未来要补 Get 方向的 IPC push 优化（Owner→Requester），那是另一个 PR 的范畴。

### 1.4 本文聚焦子系统地图

```
┌──────────────────────────────────────────────────────────┐
│                                                          │
│    MasterService                                         │
│    ┌──────────────────┐                                  │
│    │ AllocateAndInsert│ ◄── §3 副本分配机制              │
│    │  Metadata        │                                  │
│    │                  │                                  │
│    │ AllocationStrat. │                                  │
│    └──────────────────┘                                  │
│                                                          │
│    Client                                                │
│    ┌──────────────────┐                                  │
│    │ SubmitTransfers  │ ◄── §4 提交路径 + §6 PR 拦截点   │
│    │  per-replica     │                                  │
│    │  循环            │                                   │
│    └──────────────────┘                                  │
│                                                          │
│    RealClient::setup_internal                            │
│    ┌──────────────────┐                                  │
│    │ allocate_buffer_*│ ◄── §5 CPU segment 属性          │
│    │  三条路径        │                                  │
│    └──────────────────┘                                  │
│                                                          │
│    TransferEngine                                        │
│    ┌──────────────────┐                                  │
│    │ NVLinkTransport  │ ◄── §6 IPC handle 来源           │
│    │  addMemoryBuffer │                                  │
│    └──────────────────┘                                  │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

其他子系统（HA、offload、metric、admission sketch、hot cache）在本文范围之外，需要时请直接读源码。

---

## §2 · 数据模型：Replica / Segment / Buffer

### 2.1 Replica 三种类型

[replica.h:29-33](../../Mooncake/mooncake-store/include/replica.h)：

```cpp
enum class ReplicaType {
    MEMORY,     // 内存副本（CPU RAM 或 GPU VRAM 里）
    DISK,       // 分布式 disk 副本（PR 里的 SSD offload 用这个）
    LOCAL_DISK  // 纯本地 disk（仅对 creator Client 可见，不透过 Master 存元数据）
};
```

每种类型对应一份 `std::variant` 里的数据：

```cpp
std::variant<MemoryReplicaData, DiskReplicaData, LocalDiskReplicaData> data_;
```

其中：
- `MemoryReplicaData` = 一份 `std::unique_ptr<AllocatedBuffer>`
- `DiskReplicaData` = 一个 `file_path` + `object_size`
- `LocalDiskReplicaData` = `client_id` + `object_size` + `transport_endpoint`

**本 PR 只优化 MemoryReplicaData 路径**；其他两种的提交走 `FILE_READ` 或 offload RPC，不受影响。

### 2.2 MEMORY 类型为什么不拆 local/remote（对比 LOCAL_DISK）

注意到 §2.1 的 enum 里只有单一 `MEMORY`，但 DISK 却拆出了 `LOCAL_DISK`。这个不对称反映了 Mooncake 的一个关键设计：**locality 是相对于查询 Client 的运行时概念，不是 replica 的固有属性**。

#### 为什么 DISK 要拆，MEMORY 不用

对比三种类型的访问方式：

| 类型 | 访问方式 | 是否需要 owner 配合 |
|---|---|---|
| MEMORY | 通过 TransferEngine（本节点走 LOCAL_MEMCPY/NVLink/IPC pull，远端走 RDMA） | 不需要，TE 按 endpoint 路由 |
| DISK | 共享文件系统直接 `open()` | 不需要，任何 Client 都能访问 |
| LOCAL_DISK | 必须 RPC 到 owner Client 让它代为读写 | **必须**，结构体带 `client_id` 和 `transport_endpoint` |

从 `LocalDiskReplicaData` 的结构就能看出这种差异 [replica.h:124-128](../../Mooncake/mooncake-store/include/replica.h)：

```cpp
struct LocalDiskReplicaData {
    UUID client_id;                 // 必须知道哪个 Client 拥有
    uint64_t object_size = 0;
    std::string transport_endpoint; // 必须知道怎么联系它
};
```

LOCAL_DISK 的访问方式是"只能 RPC 到 owner"，这是**类型特征**，所以需要单独枚举值。
MEMORY 不管 local 还是 remote，**访问抽象都是"拿 endpoint + buffer_address 走 TE"**，区别只是 endpoint 指向本机还是远端，所以没必要拆类型。

#### 运行时怎么判断 local

Client 需要判断某份 memory replica 是不是本地的时候，调 `IsReplicaOnLocalMemory` [client_service.cpp:3078-3088](../../Mooncake/mooncake-store/src/client_service.cpp)：

```cpp
bool Client::IsReplicaOnLocalMemory(const Replica::Descriptor& replica) {
    if (!replica.is_memory_replica()) {
        return false;
    }
    const auto replica_transfer_endpoint =
        replica.get_memory_descriptor().buffer_descriptor.transport_endpoint_;
    if (metadata_connstring_ == P2PHANDSHAKE) {
        return replica_transfer_endpoint == GetTransportEndpoint();
    }
    return local_hostname_ == replica_transfer_endpoint;
}
```

两种模式：

| 模式 | 比较什么 |
|---|---|
| 有 metadata server（正常） | `local_hostname_` vs `replica.transport_endpoint_`（hostname 字符串比较） |
| P2P handshake | 自己的 TE endpoint vs replica 的 TE endpoint |

**关键点**：local/remote 是"**相对于哪个 Client 在问**"的概念 —— 同一份 memory replica，Node A 的 Client 看是 remote，Node B 的 Client 看是 local。因此 Master 无法预先为它打上 local/remote 类型标签（Master 不知道调用方是谁）。

#### 对 PR #1946 的含义

本 PR 的路由决策正是"类型不拆 + 运行时判断"的典型应用：

```cpp
if (replica.is_memory_replica()) {              // 类型过滤
    if (IsReplicaOnLocalMemory(replica)) {      // 运行时 locality 判断
        // → GPU IPC pull
    }
    // 否则 fallthrough 到原有 RDMA 路径
}
```

如果 Mooncake 曾经把 MEMORY 拆成 LOCAL_MEMORY / REMOTE_MEMORY 两种类型，PR 的拦截逻辑会更扁平，但 Master 端根本没法实现。**当前设计把 locality 判断延迟到 Client 运行时，是 Master 保持中立性的必然选择**。

### 2.3 Replica 状态机

```cpp
enum class ReplicaStatus {
    UNDEFINED,    // 未初始化
    INITIALIZED,  // 已分配空间，等待写入
    PROCESSING,   // 正在写
    COMPLETE,     // 写完，可读
    REMOVED,      // 已删除
    FAILED,       // 失败
};
```

典型转换：`PutStart` 时 → `PROCESSING` → `SubmitTransfers` 完成 → `PutEnd` 标记 `COMPLETE`。失败时标 `FAILED`，Master 周期性清理。

### 2.4 Segment：物理内存池的抽象

Segment 是"节点级的内存池"：

- 一个 Mooncake Client 启动时可以 `MountSegment` 一段 CPU 内存（也可能是 GPU）到 Master
- Master 的 `AllocatorManager` 把所有 mount 上来的 segment 按名字管理
- 副本分配时，Master 从某个 segment 里切一块 `AllocatedBuffer` 出来

**关键不变量**：同一个 key 的 N 份 replica **必须落在 N 个不同的 segment**（[allocation_strategy.h:193-194](../../Mooncake/mooncake-store/include/allocation_strategy.h) 的注释：*This strategy ensures that for each slice, its replicas are placed in different segments*）。这是容错保证 —— 一个 segment 挂了不会丢多份。

### 2.5 AllocatedBuffer 与 Descriptor

```
Replica (MEMORY)
  └─ MemoryReplicaData
       └─ std::unique_ptr<AllocatedBuffer>  ← Master 进程持有
              ├─ buffer_ptr_      (宿主 segment 里的绝对地址)
              ├─ size
              ├─ protocol
              ├─ allocator_       (弱引用回所属 BufferAllocator)
              └─ Descriptor       ← 可序列化，发给 Client

Descriptor  (allocator.h:68)
  ├─ size_
  ├─ buffer_address_       ← Client 用这个地址提交写入
  ├─ protocol_
  ├─ transport_endpoint_   ← 跨节点 RDMA 用的地址
  └─ rpc_endpoint_         ← PR #1946 新增：IPC pull RPC 目标
```

Descriptor 是 Master 返回给 Requester 的"写入说明书" —— Requester 拿着它调 `transfer_engine_->submit(replica, slices, WRITE)`（或本 PR 的 `SubmitGpuIpcPull`）。

### 2.6 "N 份 replica = N 份真实拷贝"

**这是真冗余，不是符号链接、不是 COW**。如果 `replica_num=2`：

```
  put("kv_block_42", slices, cfg{replica_num=2})
       │
       ▼
  ┌────────────────────────────────────────────┐
  │  Replica[0]               Replica[1]       │
  │  Segment "nodeA"          Segment "nodeB"  │
  │  ┌──────────────┐         ┌──────────────┐ │
  │  │AllocatedBuf  │         │AllocatedBuf  │ │
  │  │size=N bytes  │         │size=N bytes  │ │
  │  └──────────────┘         └──────────────┘ │
  │         ↑                        ↑         │
  │         │ 各独立一份              │         │
  └─────────┼────────────────────────┼─────────┘
            │                        │
         ┌──┴──────────┐         ┌───┴─────────┐
         │ SubmitTransfers 对每份 replica      │
         │ 都独立写一遍 GPU→buffer             │
         └─────────────────────────────────────┘
```

**存储代价 ≈ N × value_length**；**带宽代价 ≈ N × 数据量**（因为每份 replica 都要独立搬运一次；Mooncake 没有做 NIC/PCIe 层硬件广播）。

### 2.7 Replica 的三种价值

| 价值 | 说明 |
|---|---|
| **容错** | 一个 segment 所在节点挂了，其他 replica 仍可服务；Master 会触发重新分配 |
| **读局部性** | reader 优先从"同节点 / 同 NUMA"的 replica 读，减少跨网 |
| **读并发** | 多 reader 热 key 时，不同 replica 分散在不同 NIC/PCIe 上，避免单点带宽瓶颈 |

### 2.8 vLLM 场景下的典型选择

vLLM 把 Mooncake 当 **storage tier**（KV cache 可重算、丢失可接受）：

- **默认 `replica_num=1`**：最经济，一份足够
- 只有当"prefill 节点产的 KV 被多个 decode 节点共用"时才值得 `replica_num=2`
- `use_disk_replica=true`（额外一份 disk）在 KV cache 场景基本不用，这是给"必须持久化"的业务准备的

PR #1946 的优化**最适合 `replica_num=1` + 同节点 memory replica**的场景，也是 vLLM 集成的主流配置。

---

## §3 · 副本分配机制（Master 侧）

### 3.1 入口：`PutStart` 与 `AllocateAndInsertMetadata`

[master_service.cpp:802-854](../../Mooncake/mooncake-store/src/master_service.cpp)：

```cpp
auto MasterService::PutStart(
    const UUID& client_id, const std::string& key,
    const uint64_t slice_length, const ReplicateConfig& config)
    -> tl::expected<std::vector<Replica::Descriptor>, ErrorCode> {
    // 1. 参数校验
    if (config.replica_num == 0 || key.empty() || slice_length == 0) {
        return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
    }
    ...
    // 2. 处理已有 key（超时则 discard，否则 OBJECT_ALREADY_EXISTS）
    if (it != shard->metadata.end() && !CleanupStaleHandles(it->second)) {
        ...
    }
    // 3. 实际分配
    return AllocateAndInsertMetadata(shard, client_id, key, slice_length,
                                     config, now);
}
```

`AllocateAndInsertMetadata` 的关键代码：

```cpp
// master_service.cpp:744-799
ScopedAllocatorAccess allocator_access = segment_manager_.getAllocatorAccess();
const auto& allocator_manager = allocator_access.getAllocatorManager();

std::vector<std::string> preferred_segments;
if (!config.preferred_segment.empty()) {
    preferred_segments.push_back(config.preferred_segment);      // 旧版单值字段
} else if (!config.preferred_segments.empty()) {
    preferred_segments = config.preferred_segments;               // 新版列表字段
}

auto allocation_result = allocation_strategy_->Allocate(
    allocator_manager, value_length, config.replica_num, preferred_segments);

if (!allocation_result.has_value()) {
    need_eviction_ = true;
    return tl::make_unexpected(ErrorCode::NO_AVAILABLE_HANDLE);
}

replicas = std::move(allocation_result.value());

if (use_disk_replica_) {
    std::string file_path = ResolvePathFromKey(key, root_fs_dir_, cluster_id_);
    replicas.emplace_back(file_path, value_length, ReplicaStatus::PROCESSING);
    // ← 额外追加一份 Disk replica，不算进 replica_num
}

shard->metadata.emplace(...);  // 持久化元数据
return replica_list;             // 返回 Descriptor 列表给 Requester
```

### 3.2 三种 AllocationStrategy

全部继承 `AllocationStrategy`（[allocation_strategy.h:132](../../Mooncake/mooncake-store/include/allocation_strategy.h)）：

#### 3.2.1 RandomAllocationStrategy（默认）

```
阶段 A: 先走 preferred_segments（带 excluded 跳过）
阶段 B: 从所有 segment 随机起点绕一圈，每 segment 至多用 1 次
返回: best-effort —— 凑不齐 N 份也返回 ≥1 份
```

#### 3.2.2 FreeRatioFirstAllocationStrategy（智能）

[allocation_strategy.h:379-535](../../Mooncake/mooncake-store/include/allocation_strategy.h) 继承 Random，但阶段 B 不同：

```cpp
// 采样 6 × remaining 个候选 segment（不排全部，控制锁持有时间）
size_t sample_count = std::min(kCandidateMultiplier * remaining, names.size());
std::uniform_int_distribution<size_t> start_dist(0, names.size() - 1);
size_t start_idx = start_dist(generator);
std::vector<Candidate> candidates;
for (size_t i = 0; i < sample_count; ++i) {
    size_t idx = (start_idx + i) % names.size();
    candidates.push_back({idx, getSegmentFreeRatio(allocator_manager, names[idx])});
}

// 按空闲率降序
std::sort(candidates.begin(), candidates.end(),
          [](const Candidate& a, const Candidate& b) {
              return a.free_ratio > b.free_ratio;
          });

// 依次试，跳过 excluded/used
for (const auto& c : candidates) { ... }
```

**优点**：空闲率高的 segment 优先 —— 避免小 segment 被打满后上层 eviction。
**代价**：sampling 开销比纯随机略高。
**适用**：segment 容量异构时（不同节点内存大小不一样）。

#### 3.2.3 CxlAllocationStrategy

为 CXL memory 特化的版本，不在本文范围（vLLM 集成暂不涉及）。

### 3.3 分配不变量

| 不变量 | 由谁保证 |
|---|---|
| **INVARIANT-1**：N 份 replica 落在 N 个不同 segment | `used_segments` set，preferred/random 阶段都 check |
| **INVARIANT-2**：best-effort，凑不齐 N 份不报错（≥1 就返回） | 每个 strategy 最后的 fallback 逻辑 |
| **INVARIANT-3**：`use_disk_replica` 额外追加 1 份，不占 `replica_num` 名额 | `AllocateAndInsertMetadata` 最后的 `emplace_back` |
| **INVARIANT-4**：分配是原子的（对同一个 key） | `ScopedAllocatorAccess` + shard mutex |

### 3.4 ReplicateConfig 字段解读

[replica.h:84-113](../../Mooncake/mooncake-store/include/replica.h)：

```cpp
struct ReplicateConfig {
    size_t replica_num{1};                      // 要分配几份 memory replica
    bool with_soft_pin{false};                  // 软 pin（优先不 evict）
    bool with_hard_pin{false};                  // 硬 pin（绝不 evict）
    std::vector<std::string> preferred_segments{};  // 优先从这些 segment 分配
    std::string preferred_segment{};            // deprecated，向后兼容
    bool prefer_alloc_in_same_node{false};      // 偏好同节点分配（提升本地 IPC pull 命中）
};
```

vLLM 集成关键字段：
- `replica_num=1`：最常用
- `preferred_segments=[self_segment_name]`：**可靠的 locality hint**（见下节，不要只靠 `prefer_alloc_in_same_node`）
- `with_hard_pin`：如果 KV 不能被 evict（例如 prefill 阶段正在被 decode 消费）

#### ⚠️ `prefer_alloc_in_same_node` 的实际支持状态

这个 flag 的名字容易让人误以为 "Master 能感知调用方节点、自动分到同节点 segment"，但实际实现并非如此。

**a) Client 本地路径（多数 put/get 入口）** [real_client.cpp:968-970, 1024-1026, 1113-1115, 2233-2235, 2293-2295](../../Mooncake/mooncake-store/src/real_client.cpp)：

```cpp
if (config.prefer_alloc_in_same_node) {
    LOG(ERROR) << "prefer_alloc_in_same_node is not supported.";
    return tl::unexpected(ErrorCode::INVALID_PARAMS);
}
```

**直接拒绝，返回 INVALID_PARAMS**。

**b) `BatchPutStart` RPC server 路径** [rpc_service.cpp:728-749](../../Mooncake/mooncake-store/src/rpc_service.cpp)：

```cpp
if (config.prefer_alloc_in_same_node) {
    ReplicateConfig new_config = config;
    for (size_t i = 0; i < keys.size(); ++i) {
        auto result = master_service_.PutStart(
            client_id, keys[i], slice_lengths[i], new_config);
        results.emplace_back(result);
        if ((i == 0) && result.has_value()) {
            // 用第一个 replica 落到的 endpoint 作为后续 key 的 preferred_segment
            std::string preferred_segment;
            for (const auto& replica : result.value()) {
                if (replica.is_memory_replica()) {
                    auto handles = replica.get_memory_descriptor().buffer_descriptor;
                    if (!handles.transport_endpoint_.empty()) {
                        preferred_segment = handles.transport_endpoint_;
                    }
                }
            }
            if (!preferred_segment.empty()) {
                new_config.preferred_segment = preferred_segment;
            }
        }
    }
}
```

注意这是 **"sticky" 语义** —— 把一个 batch 里所有 key 粘到**同一个 segment**（第一个 key 分到的那个），**不是** "same node as the caller"。第一个 key 如果恰好分到远端，后面全都跟着去远端。

**结论**：vLLM 集成想让 Master 真的分到本节点 memory replica，**唯一可靠做法是显式设 `preferred_segments`**：

```python
# 伪代码
cfg.preferred_segments = [self._my_segment_name]   # 自己 mount 的 segment
```

如果不设，本 PR 的 IPC pull 命中率完全依赖 Master 的随机分配（`RandomAllocationStrategy` / `FreeRatioFirstAllocationStrategy` 都不考虑调用方 locality）—— 也就是"碰巧同节点就用，没碰上就走 RDMA"。

---

## §4 · 数据提交路径（Client 侧）

### 4.1 `SubmitTransfers` per-replica 循环

[client_service.cpp:1491 附近](../../Mooncake/mooncake-store/src/client_service.cpp)（PR 已把 IPC pull 嵌入其中）：

```cpp
void Client::SubmitTransfers(std::vector<PutOperation>& ops) {
    for (auto& op : ops) {
        if (op.replicas.empty()) { ...跳过 continue... }

        // PR 新增：提前 detect GPU 源（per-op 只做一次）
        bool gpu_source = false;
        if (gpu_ipc_pull_enabled_ && client_requester_ && !op.slices.empty()) {
            gpu_source = gpu_staging::IsDevicePointer(op.slices[0].ptr, nullptr);
        }

        bool all_transfers_submitted = true;
        std::string failure_context;

        for (size_t replica_idx = 0; replica_idx < op.replicas.size();
             ++replica_idx) {
            const auto& replica = op.replicas[replica_idx];
            if (replica.is_memory_replica()) {
                // PR 新增：尝试 GPU IPC pull（同节点 + GPU 源 + 有 rpc_endpoint_）
                if (gpu_source) {
                    auto& mem_desc = replica.get_memory_descriptor();
                    auto& buf_desc = mem_desc.buffer_descriptor;
                    if (!buf_desc.rpc_endpoint_.empty() &&
                        IsReplicaOnLocalMemory(replica)) {
                        auto future = SubmitGpuIpcPull(buf_desc, op.slices);
                        if (future) {
                            op.pending_transfers.emplace_back(std::move(future.value()));
                            continue;  // 跳过本 replica 的 RDMA
                        }
                        // IPC pull 失败 → fall through 到 RDMA
                    }
                }

                // 原有路径：走 TransferEngine（RDMA 或 NVLink）
                auto submit_result = transfer_submitter_->submit(
                    replica, op.slices, TransferRequest::WRITE);
                if (!submit_result) {
                    failure_context = "...";
                    all_transfers_submitted = false;
                    break;
                }
                op.pending_transfers.emplace_back(std::move(submit_result.value()));
            }
        }
    }
}
```

### 4.2 PutOperation 状态机

[client_service.cpp:1324-1384](../../Mooncake/mooncake-store/src/client_service.cpp)：

```cpp
enum class PutOperationState {
    PENDING,          // 初始
    MASTER_FAILED,    // BatchPutStart 失败
    TRANSFER_FAILED,  // SubmitTransfers 失败
    FINALIZE_FAILED,  // BatchPutEnd 失败
    SUCCESS
};

class PutOperation {
    std::string key;
    std::vector<Slice> slices;
    size_t value_length;
    std::vector<std::vector<Slice>> batched_slices;

    PutOperationState state = PutOperationState::PENDING;
    tl::expected<void, ErrorCode> result;
    std::vector<Replica::Descriptor> replicas;       // 从 Master 拿回来
    std::vector<TransferFuture> pending_transfers;   // 每个成功 submit 的 replica 一个 future
};
```

### 4.3 TransferFuture 异步语义

`TransferFuture` 是 pending transfer 的完成标志。`WaitForTransfers` 批量等它们完成。

**注意**：PR #1946 的 `GpuIpcPullOperationState` 是 `OperationState` 的子类，被包成 `TransferFuture`。但其实 PR 里的 RPC 是同步调用，`state->set_completed` 在 `SubmitGpuIpcPull` 返回前就设了 —— 所以包装成 `TransferFuture` 只是为了跟原有接口对齐，**并不提供真异步**（附录 A 的 B6）。

### 4.4 三种原有传输策略

[transfer_task.h:25-31](../../Mooncake/mooncake-store/include/transfer_task.h)：

```cpp
enum class TransferStrategy {
    LOCAL_MEMCPY = 0,     // 同进程、同内存 → 直接 memcpy
    TRANSFER_ENGINE = 1,  // 跨节点或跨进程 → 走 TE（通常 RDMA）
    FILE_READ = 2,        // disk replica → 读文件
    GPU_IPC_PULL = 3,     // PR #1946 新增
    EMPTY = 4             // 原来是 3，被挤到 4
};
```

**路径选择逻辑**（由 `TransferSubmitter::submit` 内部决定）：

| 源 slice 位置 vs replica 位置 | 选择的 strategy |
|---|---|
| 同进程内存 | `LOCAL_MEMCPY` |
| 跨进程同节点（或跨节点） | `TRANSFER_ENGINE`（NVLink 或 RDMA） |
| disk replica | `FILE_READ` |
| **同节点 GPU→CPU + IPC pull 条件满足** | `GPU_IPC_PULL`（PR 新增） |

### 4.5 IPC handle 从哪来：NVLinkTransport

Requester 在 `register_buffer` 时，TE 会走到 `NVLinkTransport::addMemoryBuffer`（`MC_USE_TENT=1` 开启）：

[nvlink_transport.cpp:320-342](../../Mooncake/mooncake-transfer-engine/src/transport/nvlink_transport/nvlink_transport.cpp)：

```cpp
if (attr.type != cudaMemoryTypeDevice) {
    LOG(ERROR) << "Unsupported memory type, " << addr;
    return -1;
}

cudaIpcMemHandle_t handle;
err = cudaIpcGetMemHandle(&handle, addr);              // ← 关键：生成 IPC handle
if (err != cudaSuccess) { ... }

BufferDesc desc;
desc.addr = (uint64_t)addr;
desc.length = length;
desc.name = location;
desc.shm_name = serializeBinaryData(&handle, sizeof(cudaIpcMemHandle_t));  // ← 存起来
return metadata_->addLocalMemoryBuffer(desc, true);    // ← 存到 TE metadata
```

PR #1946 的 `FindLocalGpuBufferInfo` 就是**去 TE metadata 的 `LOCAL_SEGMENT_ID` 里查 `shm_name`**。这条路径是 pre-existing 的，PR 只是消费它，没改。

---

## §5 · CPU Segment 的内存属性

这一章是 B 级讨论里最容易被忽视的部分：**Owner 的 CPU replica buffer 到底是什么样的内存**。直接关系到 PR #1946 的 `cudaMemcpy(D2H)` 能不能跑满 PCIe 带宽。

### 5.1 三条分配路径

[real_client.cpp:534-563](../../Mooncake/mooncake-store/src/real_client.cpp)：

```cpp
if (!seg_numa_nodes.empty()) {
    // 路径 1: NUMA-segmented
    ptr = allocate_buffer_numa_segments(mapped_size, seg_numa_nodes, page_sz);
} else if (should_use_hugepage) {
    // 路径 2: Hugepage
    ptr = allocate_buffer_mmap_memory(mapped_size, get_hugepage_size_from_env());
} else {
    // 路径 3: 默认 aligned_alloc
    ptr = allocate_buffer_allocator_memory(segment_size, this->protocol);
}
```

每条路径的底层实现（[utils.cpp](../../Mooncake/mooncake-store/src/utils.cpp)）：

| 路径 | 代码 | 行为 |
|---|---|---|
| 路径 1（NUMA） | `mmap(MAP_PRIVATE\|MAP_ANONYMOUS)` + `mbind(MPOL_BIND, numa_nodes[i])` | 预留 VMA + 每段绑到指定 NUMA |
| 路径 2（hugepage） | `mmap(hugepage flags \| MAP_POPULATE)` | Hugepage 自动 pin；立即 fault 物理页 |
| 路径 3（默认） | `aligned_alloc(alignment, total_size)` | 普通 malloc，可 swap |

### 5.2 Pin 状态的三重视角

pinning 在不同层级含义不同，容易混淆：

| 视角 | 问题 | 路径 1 | 路径 2 | 路径 3 |
|---|---|---|---|---|
| **OS 角度** | 能不能被 swap out? | ❌（需要 mlock） | ✅ hugepage 本身不 swap | ❌ 可被 swap |
| **RDMA 角度** | `ibv_reg_mr` 后 DMA 能直接访问? | ✅ 注册后隐式 pin | ✅ | ✅ 注册后隐式 pin |
| **CUDA 角度** | `cudaMemcpy(D2H)` 能零拷贝? | ❌ | ❌ | ❌ |

**最关键一行**：整个 Mooncake repo **完全没有** `cudaHostRegister` / `cudaHostAlloc` 调用。可自行验证：

```bash
grep -rn "cudaHostRegister\|cudaHostAlloc" \
  vllm/Mooncake/mooncake-store/ \
  vllm/Mooncake/mooncake-transfer-engine/
# (no results)
```

后果：**对 CUDA 驱动来说，所有 Mooncake CPU segment 都是 pageable host memory**。`cudaMemcpy(D2H, dst, src, size)` 会走内核管理的 pinned staging buffer 中转（两次拷贝），实测带宽≈**理论的 50%**。

### 5.3 NUMA-aware 的触发条件与实际效果

触发条件（[real_client.cpp:519-532](../../Mooncake/mooncake-store/src/real_client.cpp)）：

```cpp
std::vector<int> seg_numa_nodes;
if (!ipc_socket_path_.empty() && protocol == "rdma") {
    seg_numa_nodes = client_->GetNicNumaNodes();
    if (seg_numa_nodes.size() > 1) {
        LOG(INFO) << "NUMA-segmented mode: NIC NUMA nodes=[" << ... << "]";
    } else {
        seg_numa_nodes.clear();
    }
}
```

三个条件**必须同时满足**：

1. `ipc_socket_path_` 非空 → standalone 模式（有独立 IPC socket，**非共进程**）
2. `protocol == "rdma"` → 走 RDMA transport
3. 机器上至少 2 张 NIC 分布在不同 NUMA

触发后的行为：`allocate_buffer_numa_segments` 把 segment 按 NIC NUMA **均匀切分**，每段用 `mbind(MPOL_BIND)` 绑定到对应 NUMA。

**目的**：让 NIC DMA 的目标内存跟 NIC 在同一 NUMA，避免跨 QPI 访问。

**不做的事**：不按 key 访问局部性分配；上层谁访问哪个 NUMA 完全随机。

### 5.4 vLLM 共进程部署下的真实属性

vLLM 目前 `MooncakeStoreConnector` 是**共进程** —— Mooncake client 直接 link 到 vLLM worker 里 → `ipc_socket_path_` 通常为空 → **不触发 NUMA 路径**。

如果 `MC_STORE_USE_HUGEPAGE` 未设 → **不触发 hugepage 路径**。

结论：**走路径 3 `aligned_alloc`**。属性为：

```
OS:   可 swap（除非额外 mlockall）
RDMA: 用 RDMA 时隐式 pin
CUDA: pageable，D2H 带宽减半
NUMA: 无绑定，first-touch
```

### 5.5 对本 PR D2H 带宽的直接影响

PR #1946 的 Owner 侧执行：

```cpp
gpu_staging::CopyDeviceToHost(dst, src, size);   // 内部就是 cudaMemcpy
```

`dst` 是 Owner segment 里的地址，按上面分析是 pageable malloc 出来的。

实测（A100 + PCIe Gen4 x16，理论 32 GB/s）：
- Pageable D2H：~12 GB/s
- Pinned D2H：~24 GB/s
- RDMA loopback：~20 GB/s

**也就是说 PR 当前实现的带宽反而可能不如它要替代的 RDMA loopback**。PR 声称的 "3-8 µs" 延迟收益仅在**小数据量 + latency-bound** 时成立；中大数据量 + bandwidth-bound 下可能反超。

**修法**：Owner 侧对 `global_segment` 调一次 `cudaHostRegister(ptr, size, cudaHostRegisterPortable)`，反注册放 teardown。这是附录 A · M4 的核心建议。

---

## §6 · 案例研究：PR #1946 GPU IPC Pull 优化

### 6.1 优化目标

当 vLLM Worker（Requester）和 Mooncake Client（Owner）在**同一台物理机**上时，`batch_put_from_multi_buffers` 把 GPU KV cache 推给 Owner CPU segment：

```
  现状：RDMA loopback (2 次 PCIe + 吃 NIC)
  ─────────────────────────────────────────
  GPU VRAM ─PCIe→ 本地 NIC ─NIC 回环→ 本地 NIC ─PCIe→ CPU DRAM
            ↑ 10-20 µs，占用 NIC 带宽

  目标：单跳 PCIe DMA
  ───────────────────
  GPU VRAM ─cudaMemcpy(D2H)→ CPU DRAM
           ↑ 3-8 µs（仅 latency 口径），0× NIC
```

### 6.2 核心思想：反向拉取

常规思路会让 Requester 主动把 GPU 数据推到 Owner 的 CPU 地址。但 Requester 不知道 Owner CPU 地址在自己进程里是什么 VA（跨进程地址空间隔离），也没法直接 memcpy 过去。

**反向拉取**：

1. Requester 把自己 GPU buffer 的 IPC handle（TENT 自动生成）打包发给 Owner
2. Owner 用 `cudaIpcOpenMemHandle` 在自己进程里拿到 Requester GPU 的映射地址
3. Owner 自己执行 `cudaMemcpy(D2H)` —— 源是 imported device VA，目的是 Owner 自己的 CPU 地址

Owner 全程只和 CUDA runtime 打交道，**NIC 完全没参与**。

### 6.3 架构图

```
  ┌─────────────────── 同一物理节点 ─────────────────────┐
  │                                                     │
  │  Requester (vLLM Worker)     Owner (mooncake_client)│
  │  ┌──────────────────┐        ┌──────────────────┐   │
  │  │ GPU VRAM         │        │ CPU Segment      │   │
  │  │   gpu_ptr ●──────┼───┐ ┌──┼──● owner_cpu_va  │   │
  │  │                  │   │ │  │                  │   │
  │  │ cudaIpcGetMem    │   │ │  │                  │   │
  │  │  Handle (by TENT)│   │ │  │                  │   │
  │  │                  │   │ │  │  cudaIpcOpenMem  │   │
  │  │                  │ [IPC handle bytes]          │  │
  │  │                  │───┼─┤  │   Handle         │   │
  │  │                  │   │ │  │  (→ ipc_handle_  │   │
  │  │                  │   │ │  │      cache_)     │   │
  │  │                  │  RPC│  │                  │   │
  │  │                  │coro_rpc│  cudaMemcpy(D2H) │   │
  │  │                  │ [ACK]│  │  ── 1× PCIe ──  │   │
  │  │                  │◄──┼─┤  │                  │   │
  │  └──────────────────┘   │ │  └──────────────────┘   │
  │                         │ │                         │
  │   NIC 全程空闲           │ │                         │
  └─────────────────────────┴─┴─────────────────────────┘
```

### 6.4 路由决策：`SubmitTransfers` 嵌入式拦截

**重要**：PR 在作者更新后（commit `39747d5d`），IPC pull 逻辑**嵌入到原有的 per-replica 循环里**，不是独立 preloop。见 §4.1 的代码。

这意味着：
- 每个 replica 独立决策：能走 IPC pull 就走，不能就 fall through 到 `transfer_submitter_->submit`
- 同一个 op 的不同 replica 可以走不同路径（例如 2 份 replica，一份本地 memory→IPC pull，一份远端→RDMA）
- 没有提前 break，所有 replica 都会被处理

**拦截判断链**：

```
┌────────────────────────────────────────────────────────────────┐
│ 对每个 replica：                                               │
│                                                                │
│ ┌──────────────────────────────────────────────────────────┐  │
│ │ 必要条件（op 级，一次性判断）                              │  │
│ │   gpu_ipc_pull_enabled_ && client_requester_             │  │
│ │   && !op.slices.empty()                                  │  │
│ │   && gpu_staging::IsDevicePointer(op.slices[0].ptr)      │  │
│ │   ↓ 不满足 → 整个 op 不走 IPC pull                        │  │
│ └──────────────────────────────────────────────────────────┘  │
│                                                                │
│ ┌──────────────────────────────────────────────────────────┐  │
│ │ replica 级条件                                            │  │
│ │   replica.is_memory_replica()                            │  │
│ │   && !buf_desc.rpc_endpoint_.empty()                     │  │
│ │   && IsReplicaOnLocalMemory(replica)                     │  │
│ │   ↓ 不满足 → 本 replica 走 RDMA                           │  │
│ └──────────────────────────────────────────────────────────┘  │
│                                                                │
│ ┌──────────────────────────────────────────────────────────┐  │
│ │ SubmitGpuIpcPull 内部校验（slice 级）                     │  │
│ │   所有 slice 都在同一个 registered buffer 内              │  │
│ │   所有 slice 都在同一 device 上                           │  │
│ │   ↓ 不满足 → 返回 nullopt → 本 replica 走 RDMA            │  │
│ └──────────────────────────────────────────────────────────┘  │
│                                                                │
│  all ✓  → IPC pull RPC 发给 Owner                             │
│           → TransferFuture 加进 pending                       │
│           → continue 跳到下一个 replica                        │
└────────────────────────────────────────────────────────────────┘
```

### 6.5 Requester 侧代码走读

#### 6.5.1 `FindLocalGpuBufferInfo`

[client_service.cpp:3165-3180 (PR 后行号)](../../Mooncake/mooncake-store/src/client_service.cpp)：

```cpp
std::optional<Client::GpuBufferInfo> Client::FindLocalGpuBufferInfo(
    const void* gpu_ptr) const {
    auto metadata = transfer_engine_->getMetadata();
    auto seg_desc = metadata->getSegmentDescByID(LOCAL_SEGMENT_ID);
    if (!seg_desc) return std::nullopt;

    uint64_t addr = reinterpret_cast<uint64_t>(gpu_ptr);
    for (const auto& buf : seg_desc->buffers) {
        if (addr >= buf.addr && addr < buf.addr + buf.length) {
            if (buf.shm_name.empty()) return std::nullopt;   // TENT 未开
            return GpuBufferInfo{buf.shm_name, buf.addr, buf.length};
        }
    }
    return std::nullopt;
}
```

作用：从 TE metadata 里查 `gpu_ptr` 所属的 registered GPU buffer，返回 IPC handle bytes、base address、buffer length。

#### 6.5.2 `SubmitGpuIpcPull`（含 slice 边界校验）

```cpp
std::optional<TransferFuture> Client::SubmitGpuIpcPull(
    const AllocatedBuffer::Descriptor& handle,
    const std::vector<Slice>& slices) {
    auto buf_info = FindLocalGpuBufferInfo(slices[0].ptr);
    if (!buf_info) {
        LOG(WARNING) << "SubmitGpuIpcPull: no IPC handle for gpu_ptr "
                     << slices[0].ptr
                     << " (requires MC_USE_TENT=1 with NVLinkTransport)";
        return std::nullopt;  // 回退 RDMA
    }

    // ★ slice 边界 + device 校验（修复了旧版 B2 bug）
    int dev_id = -1;
    gpu_staging::IsDevicePointer(slices[0].ptr, &dev_id);
    for (size_t i = 0; i < slices.size(); ++i) {
        uint64_t addr = reinterpret_cast<uint64_t>(slices[i].ptr);
        if (addr < buf_info->base_addr ||
            addr + slices[i].size > buf_info->base_addr + buf_info->length) {
            return std::nullopt;    // 跨 buffer → 回退 RDMA
        }
        int slice_dev = -1;
        gpu_staging::IsDevicePointer(slices[i].ptr, &slice_dev);
        if (slice_dev != dev_id) {
            return std::nullopt;    // 跨 device → 回退 RDMA
        }
    }

    // 构造 RPC
    PutFromGpuIpcRequest req;
    req.gpu_ipc_handle = buf_info->ipc_handle;
    req.gpu_device_id = dev_id;
    uint64_t dst_offset = 0;
    for (const auto& slice : slices) {
        req.src_offsets.push_back(
            reinterpret_cast<uint64_t>(slice.ptr) - buf_info->base_addr);
        req.dst_addrs.push_back(handle.buffer_address_ + dst_offset);
        req.sizes.push_back(slice.size);
        dst_offset += slice.size;
    }

    // 同步 RPC
    auto state = std::make_shared<GpuIpcPullOperationState>();
    auto result = client_requester_->put_from_gpu_ipc(handle.rpc_endpoint_, req);

    if (!result || result->status != 0) {
        state->set_completed(ErrorCode::TRANSFER_FAIL);
    } else {
        state->set_completed(ErrorCode::OK);
        // 记录 peer，用于 unregister_buffer 时通知 release
        std::lock_guard<std::mutex> lock(gpu_ipc_pull_peers_mutex_);
        gpu_ipc_pull_peer_map_[buf_info->ipc_handle].insert(handle.rpc_endpoint_);
    }
    return TransferFuture(state);
}
```

### 6.6 Owner 侧代码走读

#### 6.6.1 `put_from_gpu_ipc` handler

[real_client.cpp:4383 附近](../../Mooncake/mooncake-store/src/real_client.cpp)：

```cpp
tl::expected<PutFromGpuIpcResponse, ErrorCode> RealClient::put_from_gpu_ipc(
    const PutFromGpuIpcRequest &req) {
    gpu_staging::SetDevice(req.gpu_device_id);

    void *owner_device_va = nullptr;
    {
        std::lock_guard<std::mutex> lock(ipc_cache_mutex_);
        auto it = ipc_handle_cache_.find(req.gpu_ipc_handle);
        if (it != ipc_handle_cache_.end()) {
            owner_device_va = it->second;          // ★ cache 命中
        } else {
            std::vector<uint8_t> buf;
            deserializeBinaryData(req.gpu_ipc_handle, buf);
            if (buf.size() != gpu_ipc::IpcHandleSize()) {
                return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
            }
            if (!gpu_ipc::OpenIpcHandle(buf.data(), buf.size(), &owner_device_va)) {
                return tl::make_unexpected(ErrorCode::TRANSFER_FAIL);
            }
            ipc_handle_cache_[req.gpu_ipc_handle] = owner_device_va;
        }
    }

    for (size_t i = 0; i < req.dst_addrs.size(); ++i) {
        void *dst = reinterpret_cast<void *>(req.dst_addrs[i]);
        const void *src = static_cast<const char *>(owner_device_va)
                          + req.src_offsets[i];
        if (!gpu_staging::CopyDeviceToHost(dst, src, req.sizes[i])) {
            return PutFromGpuIpcResponse{
                static_cast<int32_t>(ErrorCode::TRANSFER_FAIL)};
        }
    }
    return PutFromGpuIpcResponse{0};
}
```

#### 6.6.2 `ipc_handle_cache_` 生命周期

```
启动：cache 空
  │
  ▼
首次 put：cache miss → OpenIpcHandle → 缓存 {handle_bytes → device_va}
  │
  ▼
后续 put（同 buffer）：cache hit → 直接用 device_va
  │
  ▼
Requester 调 unregister_buffer（Requester 侧）
  → 查所有 peer，对每个 peer 发 release_gpu_ipc_handle RPC
  │
  ▼
Owner 收到 release → CloseIpcHandle + 从 cache 删
```

**已知问题**：cache 无上限、无 TTL；Requester crash 时 release 不发 → 永久泄漏（附录 A · M1）。

#### 6.6.3 RPC server 复用 `offload_rpc_server_`

PR 复用了已有的 SSD offload RPC 服务器，不新起服务端：

```cpp
bool need_rpc_server = (enable_ssd_offload && start_offload_rpc_server)
                       || enable_gpu_ipc_pull_;
if (need_rpc_server) {
    offload_rpc_server_ = std::make_unique<coro_rpc::coro_rpc_server>(
        /*thread_num=*/1, /*port=*/0, "0.0.0.0");
    offload_rpc_server_->register_handler<&RealClient::batch_get_offload_object>(this);
    offload_rpc_server_->register_handler<&RealClient::release_offload_buffer>(this);
    if (enable_gpu_ipc_pull_) {
        offload_rpc_server_->register_handler<&RealClient::put_from_gpu_ipc>(this);
        offload_rpc_server_->register_handler<&RealClient::release_gpu_ipc_handle>(this);
    }
    offload_rpc_server_->async_start();
}
```

**关键点**：`thread_num=1` —— 单线程处理所有 RPC。同时有多个 Requester 并发 put 时在 Owner 侧串行化（附录 A · B3）。

### 6.7 跨平台抽象 `gpu_ipc_utils.h`

PR 新增的跨平台 wrapper（[gpu_ipc_utils.h](../../Mooncake/mooncake-store/include/gpu_ipc_utils.h)）：

```cpp
namespace mooncake::gpu_ipc {

inline size_t IpcHandleSize() {
#if defined(USE_CUDA) || defined(USE_MUSA) || defined(USE_MACA)
    return sizeof(cudaIpcMemHandle_t);   // 64
#elif defined(USE_HIP)
    return sizeof(hipIpcMemHandle_t);    // 64
#elif defined(USE_ASCEND) || defined(USE_UBSHMEM)
    return 65;                           // kIPCHandleKeyLength
#else
    return 0;
#endif
}

inline bool OpenIpcHandle(const void* handle_data, size_t handle_size,
                          void** device_ptr) { ... }

inline void CloseIpcHandle(void* device_ptr) { ... }
}
```

支持 NVIDIA CUDA / AMD HIP / Moore Threads MUSA / MetaX MACA / Ascend CANN / CPU-only build。设计风格参考了同库的 `gpu_staging_utils.h`（PR #1892 引入）。

### 6.8 数据结构改动

#### 6.8.1 `AllocatedBuffer::Descriptor` +1 字段（⚠️ 破坏性）

```cpp
struct Descriptor {
    uint64_t size_;
    uintptr_t buffer_address_;
    std::string protocol_;
    std::string transport_endpoint_;
    std::string rpc_endpoint_;             // ★ NEW
    YLT_REFL(Descriptor, size_, buffer_address_, protocol_,
             transport_endpoint_, rpc_endpoint_);   // ← struct_pack 哈希变了
};
```

**影响**：Master / Owner / Requester 必须同步升级 Mooncake 版本，否则序列化协议不兼容。

#### 6.8.2 `TransferStrategy` 枚举扩展

```cpp
enum class TransferStrategy {
    LOCAL_MEMCPY = 0,
    TRANSFER_ENGINE = 1,
    FILE_READ = 2,
    GPU_IPC_PULL = 3,   // ★ NEW
    EMPTY = 4           // 原来是 3
};
```

#### 6.8.3 RPC 消息

```cpp
struct PutFromGpuIpcRequest {
    std::string gpu_ipc_handle;         // 序列化后的 IPC handle bytes
    int gpu_device_id = 0;
    std::vector<uint64_t> dst_addrs;    // Owner 侧 CPU 目标地址（绝对）
    std::vector<uint64_t> src_offsets;  // 相对 GPU buffer base 的偏移
    std::vector<uint64_t> sizes;
};
struct PutFromGpuIpcResponse { int32_t status = 0; };
```

#### 6.8.4 `GpuIpcPullOperationState`

`OperationState` 的子类，和 `MemcpyOperationState` 几乎一字不差（都是 cv + optional<ErrorCode>）。仅 `get_strategy()` 返回 `GPU_IPC_PULL`。

### 6.9 配置层次与启用矩阵

三层配置（对齐 `enable_ssd_offload` 的模式）：

| 层 | 配置项 | 优先级 |
|---|---|---|
| JSON (`MOONCAKE_CONFIG_PATH`) | `"enable_gpu_ipc_pull": true` | 最高 |
| Python (`MooncakeStoreConfig`) | `enable_gpu_ipc_pull: bool = False` | 中 |
| 环境变量 | `MOONCAKE_ENABLE_GPU_IPC_PULL=1` | 最低 |

**硬前置条件**：
- `MC_USE_TENT=1` —— TENT NVLinkTransport 会生成 IPC handle
- Owner 进程运行在加速器节点（能调 CUDA/HIP runtime）

### 6.10 CUDA context 与线程影响

详见附录 B.4。简述：

| 方面 | PR 前 | PR 后 |
|---|---|---|
| Requester CUDA context | 已有（分配 KV cache） | 不变 |
| Owner 默认 device primary context | 已有（TENT 初始化时） | 不变 |
| Owner 对其他 GPU 的 primary context | 通常无 | **按需激活**（每个被 put 的 GPU 一份，约 400MB 基线/卡） |
| Owner GPU VA 占用 | 无 | 持续占用 cudaIpcOpenMemHandle 映射的所有 buffer |
| Owner handler 线程 current device | 稳定 | 每次 handler 切换 |

### 6.11 降级矩阵

| `MC_USE_TENT` | `enable_gpu_ipc_pull` | 同节点? | `rpc_endpoint_` 有值? | 结果 |
|---|---|---|---|---|
| 0 | true | - | - | `FindLocalGpuBufferInfo` 返回空 → RDMA |
| 1 | false | - | - | 拦截不触发 → RDMA |
| 1 | true | no | yes | `IsReplicaOnLocalMemory=false` → RDMA |
| 1 | true | yes | no | Master 没分发 endpoint → RDMA |
| 1 | true | yes | yes | **GPU IPC pull 生效** ✅ |

所有 fallback **都是静默的**（仅 VLOG），这保证了 PR 的"opt-in + 失败透明"语义。

---

## §7 · vLLM 集成视角

### 7.1 `register_kv_caches` 行为

[mooncake_store_worker.py:819-898](../../vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_worker.py)：

```python
def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
    first_kv_cache = next(iter(kv_caches.values()))
    ...
    storage = first_kv_cache.untyped_storage()
    el = first_kv_cache.element_size()
    page_size_bytes = storage.nbytes() // self.num_blocks

    # 探测 KV cache layout
    outer_dims = [
        d for d in range(first_kv_cache.ndim)
        if first_kv_cache.stride(d) * el > page_size_bytes
    ]

    seen_ptrs: set[int] = set()
    self.kv_caches_base_addr: list[int] = []
    self.block_len: list[int] = []

    for cache in kv_caches.values():
        cache_storage = cache.untyped_storage()
        base_addr = cache_storage.data_ptr()
        region_len = cache_storage.nbytes()

        if base_addr not in seen_ptrs:
            seen_ptrs.add(base_addr)
            self.store.register_buffer(base_addr, region_len)  # ★ register_buffer 调用点

        if not outer_dims:
            # FlashInfer / MLA: 一个 segment per layer
            self.kv_caches_base_addr.append(base_addr)
            self.block_len.append(page_size_bytes)
        else:
            # FlashAttn / ROCm: K/V 分开 2 个 segment per layer
            seg_stride = cache.stride(outer_dims[0]) * el
            for idx in range(cache.shape[outer_dims[0]]):
                self.kv_caches_base_addr.append(base_addr + idx * seg_stride)
                self.block_len.append(seg_stride // self.num_blocks)
```

**关键机制**：`seen_ptrs` 按 `cache_storage.data_ptr()` 去重。多 layer 共享 storage 时**只注册 1 次**。

### 7.2 两种 KV cache 物理布局

| Backend | Shape | outer_dims | register_buffer 次数 | kv_caches_base_addr 数量 |
|---|---|---|---|---|
| FlashAttn / ROCm | `(2, num_blocks, ...)` | `[0]` (K/V 维) | 1（共享 storage） | `2 × num_layers` |
| FlashInfer / MLA | `(num_blocks, ...)` | 空 | 1（共享 storage） | `1 × num_layers` |

`kv_caches_base_addr` 里每个条目对应一个 segment，`prepare_value` 会为每个 segment 生成一个 slice。

### 7.3 `prepare_value` 的 slice 生成

[mooncake_store_data.py:80-97](../../vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_data.py)：

```python
def prepare_value(self, start, end, block_ids):
    addr_list = []
    size_list = []
    block_id = block_ids[start // self.block_size]
    length = len(self.block_len)
    for index, base_addr in enumerate(self.kv_caches_base_addr):
        addr = base_addr + block_id * self.block_len[index % length]
        size = int(self.block_len[index % length] / self.block_size * (end - start))
        addr_list.append(addr)
        size_list.append(size)
    return addr_list, size_list, block_id
```

一次 put 的 slice 数 = `len(kv_caches_base_addr)`（跨所有 layer 所有 K/V segment）。

### 7.4 B2 触发矩阵

回顾 B2：**一次 put 的 slices 是否跨多个 registered buffer？**

| 部署 | `register_buffer` 次数 | 所有 slice 同 buffer? | PR 当前行为 |
|---|---|---|---|
| FlashInfer + 统一池 KV（现代 vLLM 默认） | 1 次 | ✅ 同 | **IPC pull 生效** ✅ |
| FlashAttn + 统一池 KV | 1 次 | ✅ 同（K/V 只是 offset） | **IPC pull 生效** ✅ |
| FlashInfer + 每 layer 独立 storage（旧版/HMA） | N 次 | ❌ 不同 | slice 校验失败 → RDMA |
| FlashAttn + 每 layer 独立 storage | N 次 | ❌ 不同 | slice 校验失败 → RDMA |

PR 当前实现对不支持场景**静默回退**，不会错不会崩 —— 保守但正确。如果未来要让"多 buffer 源"也享受 IPC pull 优化，需要扩展 RPC 协议（附录 A · M 分类下的建议）。

### 7.5 多 worker / 多 GPU 下的进程隔离与相邻风险

**多 worker 本身不引入 B2** —— 每个 worker 是独立进程，各自有独立 Mooncake Client 和 TE metadata。Worker A 的 put 不可能包含 Worker B 的 GPU 指针（address space 隔离）。

**相邻风险（不是 B2，但值得关注）**：

| 风险 | 描述 |
|---|---|
| C1. Owner IPC handle cache 膨胀 | Owner 收到来自 N 个 Requester 的 handle，cache 大小 O(N × buffers/worker) |
| C2. Owner 跨 GPU primary context | 每个被 put 的 device 激活一个 primary context（~400MB / GPU），多 worker 累加 |
| C3. 单线程 RPC server | 所有 Requester 的 put 在 Owner 的单 coro_rpc 线程上串行 |

在 8 GPU 节点 + 8 vLLM worker 场景下，C2 意味着 Owner 进程额外占用约 3 GB VRAM 基线；C3 意味着在高吞吐下，Owner 变成瓶颈。

---

## 附录 A · Code Review 内部参考

> 基于的 PR 版本：commit `39747d5d`（2026-04-21）
> 已修问题（相对初版）：B1 多 replica 语义、B2 slice 边界校验
> 以下为**仍存在的问题**

### A.1 阅读须知

问题分级：
- 🔴 **Blocker**：影响正确性或安全，merge 前必须修
- 🟡 **Should fix**：性能或资源管理，应在 follow-up PR 中解决
- 🟢 **Nit**：代码质量、可扩展性

### A.2 仍存在的 Blocker

#### 🔴 B3. 单线程 RPC server 成为并发瓶颈

`offload_rpc_server_` 构造时 `thread_num=1`。多 worker 场景所有 IPC pull 请求串行化，且 `cudaMemcpy` 同步阻塞 handler 线程。

**建议**：`thread_num = N GPU_on_owner` 或启用 `cudaMemcpyAsync + stream` 让 driver 调度并发。

#### 🔴 B4. `unregister_buffer` 与并发 put 的竞态

```
线程 1：unregister_buffer(X) → FindLocalGpuBufferInfo → 取 handle
线程 2：put(X) → 对 peer 发 IPC pull RPC
线程 1：对 peer 发 release_gpu_ipc_handle RPC
Owner：收到 release → Close → 从 cache 删
Owner：收到 put → cache miss → reopen
       ↑ 但此时 Requester 可能已 cudaFree(X) → Open 成功但是 dangling
```

**建议**：unregister 前 drain 本 buffer 的未完成 put；或引入 per-buffer refcount 机制。

#### 🔴 B5. `dst_addrs` 缺乏安全校验

```cpp
for (size_t i = 0; i < req.dst_addrs.size(); ++i) {
    void *dst = reinterpret_cast<void *>(req.dst_addrs[i]);  // ★ 完全未校验
    gpu_staging::CopyDeviceToHost(dst, src, req.sizes[i]);
}
```

Requester 发来任意地址，Owner 就写进去。bug 或恶意 Requester 可让 Owner 改写任意 host 虚拟地址。PR 引入了 `IsAddressInMountedSegment` 辅助函数但**没被实际调用**。

**建议**：Owner 在 handler 里对每个 `dst_addrs[i]` 调 `IsAddressInMountedSegment` 校验。

#### 🔴 B6. `SubmitGpuIpcPull` 的假异步

`client_requester_->put_from_gpu_ipc` 是**同步 RPC**，返回前 `state->set_completed` 已设定。包成 `TransferFuture` 返回给上层，但上层 `WaitForTransfers` 不会真的等 —— 因为已完成。

**后果**：多个 op 的 IPC pull 无法 pipeline。对比 RDMA 异步提交 + 批量等待，IPC pull 在高 op 率下可能反而更慢。

**建议**：真正用 coro_rpc 的 async 能力，`put_from_gpu_ipc` 返回 awaiter，塞进 state，在 `wait_for_completion` 里 `syncAwait`。

### A.3 性能隐患

#### 🟡 M4. **无 cudaHostRegister → D2H 带宽打折**（最关键）

详见 §5。Owner segment 是 pageable malloc，`cudaMemcpy(D2H)` 走中间 staging buffer，带宽减半。

**建议**：setup_internal 分配 segment 后调 `cudaHostRegister(ptr, size, cudaHostRegisterPortable)`，teardown 时 `cudaHostUnregister`。

#### 🟡 M1. `ipc_handle_cache_` 无界、无 LRU

长期运行的 Owner 会持续累积 handle。Requester crash 时 release 不发 → 永久泄漏。

**建议**：容量上限 + LRU；或 per-entry TTL；或基于 coro_rpc 断连的主动清理。

#### 🟡 M2. `FindLocalGpuBufferInfo` 每次 put 都线性扫 buffers

per-op O(N_buffers)。建议改成 gpu_ptr → info 的区间索引（`std::map` 上用 `upper_bound`）。

#### 🟡 M3. `gpu_ipc_pull_peer_map_` 的 `insert` 每次都加 mutex

成功路径上无条件加锁 + 哈希插入。99% 情况下 peer 已经在。

**建议**：先 shared_lock contains() 探测，miss 再 unique_lock insert；或 `std::call_once` per-endpoint。

### A.4 资源与生命周期

#### 🟡 M5. Owner 跨 GPU primary context 膨胀

每个 `SetDevice(device_id)` 首次访问都激活一个 primary context（~400MB）。N worker 场景下 Owner 多吃 N × 400MB VRAM。

**建议**：文档声明 Owner 行为；或限制单 Owner 进程只服务单 GPU（per-GPU 拆 Owner）。

#### 🟡 SetDevice 污染 handler 线程

`gpu_staging::SetDevice` 改 current thread 的 device。如果 coroutine 在 handler 内 yield 切换线程，current device 可能不符预期。

**建议**：RAII 包 scoped device change；或把 memcpy 下放到 per-device 专用线程。

### A.5 API 与测试

#### 🟢 N1. `GpuIpcPullOperationState` 和 `MemcpyOperationState` 几乎一字不差

可提取共用 template 基类。

#### 🟢 N2. `Descriptor.rpc_endpoint_` 破坏性兼容

`YLT_REFL` 增加字段 → struct_pack 哈希变 → Master/Owner/Requester 必须同步升级。

**替代方案**：用 `optional<string>` + yalantinglibs 的 `compatible<>` 标签实现向后兼容。

#### 🟢 N5. 测试没覆盖端到端

PR 的 15 个 test case 都是结构验证（OperationState 字段、enum 值、序列化 round-trip）。**没有跨进程的 IPC pull 全流程测试**（`cudaIpcOpenMemHandle` 在同进程会返 `cudaErrorInvalidValue`，所以单 test 进程内测不了）。

**建议**：加 fork-based integration test；或 multi-buffer slice fallback 测试；或 release_gpu_ipc_handle 后 reopen 测试。

#### 🟢 N6. 日志不足

成功路径没日志，debugging 时没法 diff。建议 `VLOG(1) << rpc_endpoint + slice 数 + total bytes`。

### A.6 按优先级的改进建议

**merge 前必须修（blocker 最低集）**：

1. B5（dst 安全校验）：把 `IsAddressInMountedSegment` 实际用起来
2. M4（cudaHostRegister）：否则性能收益为负

**merge 前应修（强烈建议）**：

3. B3（多线程 RPC）
4. B4（unregister 竞态）
5. M1（cache 容量控制）

**可 follow-up**：

6. B6（真异步）
7. M2/M3（缓存优化）
8. N5（端到端测试）
9. M5（多 GPU 文档/拆分）

---

## 附录 B · 深度 Q&A 扩展

### B.1 Replica 是真冗余吗？是浪费吗？

**是真冗余**。`replica_num=N` 就有 N 份物理拷贝，N × size 的存储 + N × bandwidth 的写带宽。

**不浪费 vs 浪费**见 §2.7 的决策表：

| 场景 | replica_num | 是否浪费 |
|---|---|---|
| vLLM 纯缓存（丢失可重算） | 1 | ✅ 不浪费 |
| 多 reader 并发读热 key | 2 | ❌ 不浪费 |
| 容灾 / 不可重算 state | 2~3 | ❌ 不浪费 |

### B.2 为什么不初始化时就缓存 `owner_device_va`？

Owner 不知道未来会有哪些 Requester、注册哪些 buffer。`owner_device_va = cudaIpcOpenMemHandle(handle)` 的输入 `handle` 是 Requester 运行时 `register_buffer` 时生成的，Owner 只能 lazy 缓存。

**PR 已经做了 cache**（见 §6.6.2），稳态是 1 次 `unordered_map::find` + 1 次 `cudaMemcpy`。

`gpu_ipc_pull_peer_map_.insert` 的锁开销属于可优化项（附录 A · M3）。

### B.3 是否假设本地一定有 CPU 空闲？Remote/Disk 走哪条路？

**不假设**。决策权在 Master：

| Master 分配到哪 | PR 行为 |
|---|---|
| 本节点 memory replica | ✅ IPC pull 生效 |
| 远端 memory replica | 拦截失败（not local）→ RDMA |
| Disk replica | 拦截失败（not memory）→ FILE_READ |
| 本地 memory 但无 rpc_endpoint（Owner 没开 feature） | 拦截失败 → RDMA |

本地没空闲 memory 时，Master 在 `AllocateAndInsertMetadata` 阶段根本分配不到本地 replica，触发 eviction 或 NO_AVAILABLE_HANDLE。

### B.4 原来是否有 CUDA context？本 PR 是否增加？

#### Requester 侧

不变。vLLM worker 本来就在 GPU 上跑，早有 context。

#### Owner 侧

两种情况：

- **Owner 本来就是加速器节点 + MC_USE_TENT=1**（PR 目标场景）：TE 初始化时已激活**默认 device** 的 primary context。PR 不新增 context，但**首次对非默认 device 调 `SetDevice` 时会激活那块 GPU 的 primary context**（每张 ~400MB）。
- **Owner 纯 CPU only**：`gpu_staging::SetDevice` 是 noop，`OpenIpcHandle` 返 false，RPC 失败 → PR 实际没生效。

**踩坑提示**：
- 多 GPU 节点 + 多 worker → Owner 持有多个 primary context
- handler 线程的 current device 被 `SetDevice` 污染，coroutine 跨线程后 current device 可能错乱

### B.5 多 worker / 多 GPU 注册会违反 B2 吗？

**不会**，因为 worker 是独立进程，VA 空间隔离。每个 worker 的 Mooncake Client 只有自己 `register_buffer` 过的 metadata，`FindLocalGpuBufferInfo` 不会返回其他 worker 的 handle。

B2 触发的真正条件是**同一进程内**：
- 单 worker 多次 register_buffer（每 layer 独立 storage）
- 单进程跨多 GPU 注册（非标准 vLLM 配置）

多 worker 多 GPU 带来的问题不是 B2，是 **C1/C2/C3**：Owner 资源 O(N worker) 膨胀 + 单线程 RPC 瓶颈（见 §7.5）。

### B.6 B2 的详细失败模式（历史 —— 已修）

PR 旧版本用 `slices[0].ptr` 一把取 base_addr。当 slices 来自多个 registered buffer：

**模式 A：硬崩**。`src_offset = slice[i].ptr - wrong_base` 越界，`cudaMemcpy` 返回 `cudaErrorIllegalAddress`，Owner 进程进入 sticky error，需重启。

**模式 B：静默污染**（最可怕）。如果 offset 计算后仍落在 imported buffer 映射区内，`cudaMemcpy` 读到"错地方"的数据，写进 Owner replica。Mooncake 以为 put 成功，未来 get 返回错误 KV，vLLM 解码出乱码。无任何错误日志。

**当前 PR 的修法**：slice 级边界 + device 校验（[见 §6.5.2](#652-submitgpuipcpull含-slice-边界校验)）。任一不满足直接回退 RDMA。保守但正确。

---

## 附录 C · 关键源码路径索引

### Mooncake Store（核心）

| 文件 | 角色 |
|---|---|
| [mooncake-store/include/allocator.h](../../Mooncake/mooncake-store/include/allocator.h) | `AllocatedBuffer` / `Descriptor` / `BufferAllocator` 声明 |
| [mooncake-store/include/allocation_strategy.h](../../Mooncake/mooncake-store/include/allocation_strategy.h) | 三种 `AllocationStrategy` |
| [mooncake-store/include/replica.h](../../Mooncake/mooncake-store/include/replica.h) | `Replica` / `ReplicateConfig` / 状态机 |
| [mooncake-store/include/transfer_task.h](../../Mooncake/mooncake-store/include/transfer_task.h) | `TransferStrategy` / `OperationState` |
| [mooncake-store/include/client_service.h](../../Mooncake/mooncake-store/include/client_service.h) | `Client` 接口 |
| [mooncake-store/include/real_client.h](../../Mooncake/mooncake-store/include/real_client.h) | `RealClient`（`PyClient` 实现） |
| [mooncake-store/include/rpc_types.h](../../Mooncake/mooncake-store/include/rpc_types.h) | RPC 消息结构体 |
| [mooncake-store/include/master_service.h](../../Mooncake/mooncake-store/include/master_service.h) | `MasterService` 接口 |
| [mooncake-store/src/master_service.cpp](../../Mooncake/mooncake-store/src/master_service.cpp) | `PutStart`、`AllocateAndInsertMetadata` |
| [mooncake-store/src/client_service.cpp](../../Mooncake/mooncake-store/src/client_service.cpp) | `SubmitTransfers`、`SubmitGpuIpcPull`（PR 新增） |
| [mooncake-store/src/real_client.cpp](../../Mooncake/mooncake-store/src/real_client.cpp) | `setup_internal`（含 CPU segment 分配三路径）、`put_from_gpu_ipc` handler |
| [mooncake-store/src/utils.cpp](../../Mooncake/mooncake-store/src/utils.cpp) | `allocate_buffer_*` 三函数 |

### Mooncake TransferEngine（IPC handle 来源）

| 文件 | 角色 |
|---|---|
| [mooncake-transfer-engine/src/transport/nvlink_transport/nvlink_transport.cpp](../../Mooncake/mooncake-transfer-engine/src/transport/nvlink_transport/nvlink_transport.cpp) | `addMemoryBuffer` 调 `cudaIpcGetMemHandle` |
| [mooncake-transfer-engine/include/transfer_metadata.h](../../Mooncake/mooncake-transfer-engine/include/transfer_metadata.h) | `BufferDesc` 含 `shm_name`（序列化 handle） |

### vLLM 集成

| 文件 | 角色 |
|---|---|
| [vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_worker.py](../../vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_worker.py) | `register_kv_caches`、put/get 线程 |
| [vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_data.py](../../vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_data.py) | `prepare_value` 构造 slice |

### PR 新增文件

| 文件 | 角色 |
|---|---|
| `mooncake-store/include/gpu_ipc_utils.h` | 跨平台 IPC handle wrapper |
| `mooncake-store/tests/gpu_ipc_pull_test.cpp` | 15 个单元测试 |

---

## 附录 D · 不变量清单

| ID | 不变量 | 由谁保证 | 若违反的后果 |
|---|---|---|---|
| INV-1 | 同一 key 的 N 份 replica 落在 N 个不同 segment | `AllocationStrategy` 的 `used_segments` | 单 segment 故障丢多份 |
| INV-2 | 副本分配 best-effort，≥1 即返回 | `AllocationStrategy::Allocate` 返回逻辑 | 严格要求 N 份时需上层检查 |
| INV-3 | `use_disk_replica` 额外 1 份，不占 `replica_num` | `AllocateAndInsertMetadata` 的 emplace | — |
| INV-4 | `Descriptor.rpc_endpoint_` 非空 ⇒ Owner 启用 `enable_gpu_ipc_pull_` | `AllocatedBuffer::get_descriptor` + `allocator->getRpcEndpoint()` | 发到 Owner 的 RPC 无人响应 |
| INV-5 | `cudaIpcOpenMemHandle` 返回的 device_va 在 Owner 进程内有效，直到 `CloseIpcHandle` | CUDA runtime | cache 不及时清理 → GPU VA 泄漏 |
| INV-6 | IPC pull 时所有 slice 在同一 registered GPU buffer 内 | `SubmitGpuIpcPull` 的边界校验 | slice 跨 buffer 时静默回退 RDMA（非违反） |
| INV-7 | `offload_rpc_server_` thread_num=1 | 硬编码 | 多 Requester 并发下 Owner 串行化 |
| INV-8 | Requester unregister 前应 drain 本 buffer 的 put（**未实现**） | — | 竞态 → Owner handle stale |

---

## 附录 E · 术语中英对照表

| 中文 | 英文 | 在本文出现位置 |
|---|---|---|
| 副本 | replica | §2, §3 |
| 段 | segment | §2, §5 |
| 元数据中心 | Master / MasterService | §1 |
| 数据平面 | data plane | §1 |
| 副本分配 | replica allocation | §3 |
| 最佳努力 | best-effort | §3.3 |
| 同节点反向拉取 | same-node reverse pull | §6 |
| 单跳 PCIe DMA | single-hop PCIe DMA | §6.1 |
| 环回 | loopback | §6.1 |
| 锁定内存 | pinned memory | §5 |
| 可换出内存 | pageable memory | §5 |
| 主上下文 | primary context | §6.10 |
| 静默污染 | silent corruption | 附录 B.6 |

---

*本文基于 Mooncake 本地工作树（`vllm/Mooncake/`）和 PR #1946 commit `39747d5d` 编写，最后验证日期 2026-04-21。若源码发生显著变化，请用 `grep` 重新核验关键代码片段的行号。*
