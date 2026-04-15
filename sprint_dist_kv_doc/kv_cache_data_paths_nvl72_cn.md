# Complete Analysis of KV Cache Data Paths on NVL72 GB200

## 1. 概述

在 NVL72 GB200 上，vLLM + Mooncake（L2 CPU Memory + L3 NVMe SSD）场景下，GPU 需要的 KV Cache block 可能来自 **6 种位置**。本文自上而下拆解：每条路径的物理 datapath、Mooncake 内部的 replica 选择逻辑、RDMA 批量调用机制、以及数据放置的随机性问题。

---

## 2. 数据来源总览

```
请求方 GPU 需要一个 KV block，可能来自：

  ① 本地 GPU HBM     不经过 Mooncake，vLLM prefix caching 直接复用
  ② 本地 CPU Memory   Mooncake MEMORY replica，RDMA loopback
  ③ 本地 NVMe SSD     Mooncake LOCAL_DISK replica，io_uring + RDMA loopback
  ④ 远程 CPU Memory   Mooncake 远程 MEMORY replica，RDMA 或 NVLink 中转
  ⑤ 远程 NVMe SSD     远端 FileStorage RPC + RDMA
  ⑥ 远程 GPU HBM      P2P connector (NixlConnector / MooncakeConnector)，不经过 Store
```

| # | 来源 | Connector | Transport | 典型带宽 |
|---|------|-----------|-----------|---------|
| ① | 本地 GPU HBM | vLLM 自管 | — | 8 TB/s |
| ② | 本地 CPU Memory | MooncakeStoreConnector | RDMA loopback | RNIC 依赖 |
| ③ | 本地 SSD | MooncakeStoreConnector | io_uring + RDMA | ~25 GB/s (SSD 瓶颈) |
| ④ | 远程 CPU Memory | MooncakeStoreConnector | RDMA (RoCE) / NVLink 中转 | 50 / 450 GB/s |
| ⑤ | 远程 SSD | MooncakeStoreConnector | FileStorage RPC + RDMA | SSD+网络 |
| ⑥ | 远程 GPU HBM | NixlConnector / MooncakeConnector | UCX cuda_ipc / RDMA GPUDirect | NVLink ~900 GB/s |

---

## 3. 两种 Mooncake Connector

| | MooncakeConnector | MooncakeStoreConnector |
|---|---|---|
| 角色 | 跨实例 P→D KV 直传（替代 NixlConnector） | KV Cache 分层缓存（CPU/SSD offload + 跨节点复用） |
| 底层 | TransferEngine 直接 P2P | MooncakeDistributedStore（Master 协调） |
| 数据流 | Prefill GPU → RDMA → Decode GPU | GPU ↔ CPU Memory Pool ↔ SSD |
| 缓存 | 不做缓存 | 支持 prefix caching（hash 去重） |

生产配置用 MultiConnector 组合：save 时写入所有 sub-connector，load 时按顺序尝试。

---

## 4. 注册拓扑：谁在 Master 的分配池中

每个 vLLM GPU worker 进程独立初始化，向 Master 注册两类内存：

```
GPU VRAM:  register_buffer() → RegisterLocalMemory(remote_accessible=false)
           → 只注册到 Transfer Engine（使 RDMA 可访问）
           → 不上报 Master，不进入 allocator pool

CPU Mem:   setup() → allocate_buffer_*() → MountSegment()
           → 上报 Master，addAllocator() 进入分配池
           → segment.name = local_hostname_（同节点 worker 共享同一 name）
```

**Master 只从 CPU global_segment 中分配。GPU VRAM 不在分配池中。** GPU VRAM 仅作为 RDMA 传输的源（put）/ 目的（get）端点。

2 节点 TP4 的完整拓扑：

```
Master Server
  │
  │  AllocatorManager.names_ = ["10.x.x.11", "10.x.x.14"]
  │  allocators_ = {
  │    "10.x.x.11": [CPU_seg_5GB × 4],   ← Node A 的 4 个 worker 各 5GB
  │    "10.x.x.14": [CPU_seg_5GB × 4],   ← Node B 的 4 个 worker 各 5GB
  │  }
  │
  ├─ Node A (10.x.x.11), Prefill, TP=4
  │   Worker 0~3: 各自 GPU VRAM (仅 Transfer Engine) + CPU 5GB (Master pool)
  │
  └─ Node B (10.x.x.14), Decode, TP=4
      Worker 0~3: 各自 GPU VRAM (仅 Transfer Engine) + CPU 5GB (Master pool)
```

---

## 5. 数据放置：Put 时写到哪

### 5.1 分配策略：随机，不感知本地性

vLLM 调用 `batch_put_from_multi_buffers(keys, addrs, sizes)` 时不传 `ReplicateConfig`，使用默认值：`replica_num=1`，`preferred_segments=[]`。

Master 的 `RandomAllocationStrategy::Allocate()`（allocation_strategy.h:271-297）：
1. 从 `names_` 中随机选一个起点（`std::uniform_int_distribution`）
2. 顺序扫描直到找到有空间的 segment
3. **不感知调用者在哪个节点** — Master 只知道有哪些 segment，不知道 put 来自哪里

**结论：数据可能被写到任何节点的 CPU memory。** Prefill 在 Node A 产出 KV → Master 可能把 replica 分配到 Node B 的 CPU segment → RDMA 跨节点写入 Node B。

### 5.2 举例

```
Prefill (Node A) 写 100 个 KV block:
  → Master 对每个 key 随机分配: ~50 个在 Node A CPU，~50 个在 Node B CPU
  → 50 个 RDMA loopback 写本地，50 个 RDMA 远程写到 Node B
```

### 5.3 物理数据流

```
Put:  GPU VRAM (源) ──RDMA Write──→ CPU global_segment (目标, Master 分配)
                                    ├─ 可能在本地 (loopback)
                                    └─ 可能在远端 (跨节点 RDMA)
```

---

## 6. 数据读取：Get 时从哪拿

### 6.1 Replica 选择优先级

Master 返回 replica 列表按插入顺序：**MEMORY 在前，DISK/LOCAL_DISK 在后**。

vLLM 的 `batch_get_into_multi_buffers` 只传 3 个参数，`prefer_alloc_in_same_node` 默认 `false`（store_py.cpp:2226），走 `FindFirstCompleteReplica`——**取列表中第一个 COMPLETE 状态的 replica，不检查本地性**。

```
实际优先级：
  1. 第一个 COMPLETE 的 MEMORY replica（不区分本地/远程，取决于 Master 列表顺序）
  2. DISK replica（所有 MEMORY 都不可用时）
  3. LOCAL_DISK replica（仅当只剩 LOCAL_DISK 时，走 FileStorage RPC 专用路径）
```

**本地 CPU 和远程 CPU 之间没有优先级。** 拿到哪个取决于 PutStart 时随机分配到了哪个节点。

### 6.2 会从别的节点的 CPU Memory 拿吗？

**会。** 因为 put 和 get 两端都没有本地性偏好：

```
场景：Decode 在 Node B，需要一个 KV block

如果 PutStart 时这个 key 被随机分配到 Node A 的 CPU segment：
  → Master 返回 replica: MEMORY on Node A
  → Node B batch_get → FindFirstCompleteReplica → 选到 Node A 的 MEMORY
  → RDMA 远程读: Node A CPU → RoCE → Node B GPU

如果 PutStart 时这个 key 被分配到 Node B 的 CPU segment：
  → Master 返回 replica: MEMORY on Node B
  → Node B batch_get → 选到 Node B 的 MEMORY
  → RDMA loopback: Node B CPU → Node B GPU

哪种情况发生完全取决于 PutStart 的随机分配结果。
```

### 6.3 MEMORY 和 DISK replica 的共存

```
场景 1: CPU 没满 → MEMORY(COMPLETE) 在前，DISK(COMPLETE) 在后 → 选 MEMORY
场景 2: CPU 满，MEMORY 被驱逐 → 只剩 LOCAL_DISK(COMPLETE) → 走 SSD 路径
场景 3: 多节点，Node A MEMORY 被驱逐但 Node B MEMORY 还在
       → [MEMORY_B(COMPLETE), LOCAL_DISK_A(COMPLETE)] → 选远程 MEMORY_B
       → 远程 CPU memory 优先于本地 SSD
```

---

## 7. RDMA 批量调用机制

**不是每个 key 一次 RDMA，也不是每个远程节点一次 RDMA。** 是 Slice 级别批量 + watermark 流控。

### 7.1 调用链

```
vLLM batch_get_into_multi_buffers(100 keys)
  │
  ├─ LOCAL_DISK keys → FileStorage RPC 专用路径（串行两阶段）
  │
  └─ MEMORY keys → Client::BatchGet()
       │ 对每个 key 独立提交 async transfer（不等完成）
       ▼
     TransferSubmitter → 每个 key 拆成 64KB Slice
       ▼
     RdmaTransport::submitTransferTask()
       │ 按 RdmaContext 分组: slices_to_post[context].push_back(slice)
       │ 累积到 watermark (~512 slice) → 一次 ibv_post_send() 批量提交
       │ 不同远端节点的 slice 混在同一个 post_send 中
       ▼
     RNIC DMA engine 按 QP 并行执行
```

### 7.2 具体例子

```
100 个 MEMORY keys: 40 在 Node A CPU, 30 在 Node B CPU, 30 在本地 CPU
  → 每个 key 256KB → 每个 key 4 个 64KB slice → 共 400 slice
  → 400 < watermark(512) → 一次 ibv_post_send()
  → RNIC 内部并行处理去 3 个目标的 slice
```

| 参数 | 默认值 | 含义 |
|------|--------|------|
| Slice 大小 | 64 KB (`MC_SLICE_SIZE`) | 每个 RDMA 请求粒度 |
| Watermark | ~512 (`max_wr × num_qp_per_ep`) | 累积多少 slice 后批量提交 |

---

## 8. 每条数据路径的物理 Datapath

### 路径 ②：本地 CPU Memory → 本地 GPU

```
GPU→CPU offload: GPU VRAM → IBV_WR_RDMA_WRITE → RNIC loopback → CPU global_segment
CPU→GPU load:    CPU global_segment → IBV_WR_RDMA_READ → RNIC loopback → GPU VRAM

带宽: RNIC loopback 能力（非 NVLink-C2C — Mooncake 默认 protocol="rdma"）
延迟: sub-ms（RDMA bucket 125-1000 μs）
开销: 零 CPU/GPU SM 占用（DMA engine 驱动）
```

### 路径 ③：本地 SSD → 本地 CPU → 本地 GPU

```
NVMe SSD → io_uring (O_DIRECT) → ClientBuffer (CPU staging ~1GiB) → RDMA loopback → GPU VRAM

SSD 带宽: ~25+ GB/s (RAID0 4盘)
限制: 单节点多 GPU 场景 disk offload 当前不工作（Kimi TP4 受影响）
```

### 路径 ④：远程 CPU Memory → 本地 GPU

```
方案 A（当前默认）: 远端 CPU → RNIC → RoCE 网络 → 本地 RNIC → 本地 GPU   (~50 GB/s)
方案 B（柜内可选）: 远端 CPU → C2C → 远端 GPU → NVLink → 本地 GPU        (~450 GB/s)
  前提: protocol="nvlink" + USE_MNNVL 编译 flag + 同一 NVL72 柜内
  限制: protocol 是全局配置，改了 nvlink 跨柜会 break
```

### 路径 ⑤：远程 SSD → 远程 CPU → 本地 GPU

```
远端 NVMe → io_uring → 远端 ClientBuffer → (RDMA/NVLink) → 本地 GPU

实现: FileStorage RPC 两阶段
  1. RPC: batch_get_offload_object → 远端读 SSD 到 ClientBuffer
  2. Transfer Engine: RDMA 从远端 ClientBuffer 拉到本地 GPU
  两阶段串行，ClientBuffer 是有限资源（~1GiB）
```

### 路径 ⑥：远程 GPU → 本地 GPU（P2P 直传，不经过 Store）

```
NixlConnector:      UCX cuda_ipc (MNNVL NVLink 直通, 柜内 ~900 GB/s)
                    UCX cuda_copy+tcp (跨柜 ~50 GB/s)
MooncakeConnector:  RDMA GPUDirect Write (~50 GB/s) 或 NVLink
```

---

## 9. 判断流程图

```
GPU 需要 KV block
  │
  ├─ vLLM prefix caching 命中本地 GPU HBM?
  │   └─ YES → 路径 ① 直接复用，不经过 Mooncake
  │
  └─ NO → batch_get_into_multi_buffers(key)
       │
       ▼
     Master BatchQuery(key) → replica 列表
       │
       ├─ 仅有 LOCAL_DISK replica?
       │   └─ YES → FileStorage RPC → SSD → staging → RDMA → GPU (路径 ③/⑤)
       │
       └─ NO → FindFirstCompleteReplica → 第一个 COMPLETE 的 (通常 MEMORY)
            │
            ├─ replica 在本地节点 CPU? → RDMA loopback (路径 ②)
            └─ replica 在远程节点 CPU? → RDMA 远程读 (路径 ④)
                                         ↑ 本地/远程由 PutStart 随机分配决定
```

---

## 10. 当前局限与优化方向

| 问题 | 现状 | 潜在优化 |
|------|------|---------|
| Put 随机分配，数据可能跨节点 | `preferred_segments=[]`, Master 随机选 | vLLM 传 `preferred_segments=[local_hostname]` |
| Get 不区分本地/远程 | `prefer_alloc_in_same_node=false` | vLLM 传 `true`，启用 endpoint 分组批量 |
| 柜内走 RDMA 而非 NVLink | `protocol="rdma"` 全局配置 | 改为 `"nvlink"` (但跨柜会 break) |
| Disk offload 多 GPU 不可用 | Mooncake 当前实现限制 | 等上游修复 |
| 无 NUMA / 拓扑感知 | replica 选择不考虑距离 | 需 Mooncake 侧改造 |
| `GetPreferredReplica` 不用于 BatchGet | 只有单 key Get 有本地优先 | 需 Mooncake 或 vLLM 侧改造 |

---

## 附录 A：关键源码文件

| 文件 | 内容 |
|------|------|
| `vllm/.../mooncake_store_worker.py` | vLLM Worker: send/recv thread, register_kv_caches, lookup |
| `vllm/.../mooncake_store_connector.py` | vLLM Scheduler: build_connector_meta |
| `vllm/.../mooncake_connector.py` | P2P connector: ZMQ 握手 + TransferEngine |
| `Mooncake/.../real_client.cpp` | C++ Client: batch_get/put, replica query |
| `Mooncake/.../client_service.cpp` | Client: BatchGet, GetPreferredReplica, MountSegment |
| `Mooncake/.../replica.h` | Replica 类型: MEMORY, DISK, LOCAL_DISK |
| `Mooncake/.../allocation_strategy.h` | 分配策略: RandomAllocationStrategy |
| `Mooncake/.../segment.cpp` | Master: MountSegment → addAllocator |
| `Mooncake/.../multi_transport.cpp` | Transport 选择: selectTransport() |
| `Mooncake/.../rdma_transport.cpp` | RDMA 批量: Slice + watermark |
| `Mooncake/.../store_py.cpp` | Python binding: 默认参数 |

## 附录 B：NVL72 柜内互联拓扑

```
┌──────────────── NVL72 柜 ────────────────────┐
│  9 × NVLink Switch Tray, 72 GPU 全互联        │
│                                                │
│  Node (Compute Tray):                          │
│    GPU0 ←(C2C 900GB/s)→ Grace CPU0 ←(C2C)→ GPU1   │
│    GPU2 ←(C2C 900GB/s)→ Grace CPU1 ←(C2C)→ GPU3   │
│                                                │
│  GPU↔GPU (柜内):  NVLink 1.8 TB/s per GPU     │
│  GPU↔CPU (同 SC):  NVLink-C2C 900 GB/s 双向   │
│  CPU↔CPU (跨 SC):  无直连，需 GPU 或 RDMA 中转  │
│  Node↔Node (跨柜): RoCE RDMA                  │
└────────────────────────────────────────────────┘
```
