# Mooncake 同节点 GPU↔CPU 传输优化：完整分析文档

**日期**: 2026-04-17
**作者**: aoshen524 + Claude
**相关 PR**:
- kvcache-ai/Mooncake#1913 (POSIX SHM segment — 方案 A，我们提交，未合并)
- kvcache-ai/Mooncake#1892 (GPU staging via cudaMemcpy D2H — 社区 OPEN，修磁盘路径 segfault，与本文主线正交，提供 `gpu_staging_utils`/`PinnedBufferPool` 工具可复用)
- kvcache-ai/Mooncake#1890 (TENT cudaMemcpyBatchAsync 批量优化 — OPEN)
- kvcache-ai/Mooncake#1580, #1838 (NUMA-aware segment allocation — 已合并)
- kvcache-ai/Mooncake#711 (NVLink local transport — 关闭)
- kvcache-ai/Mooncake#823 (vLLM NUMA binding — 已合并)
- kvcache-ai/Mooncake#1535 (Dummy→Real memfd SCM_RIGHTS — 已合并，方案 A 的 ShmHelper 基石)
- ivanium/vllm#19 (vLLM `--tent` flag — 暂存)

---

## 目录

1. [问题场景与背景](#1-问题场景与背景)
2. [当前数据路径分析](#2-当前数据路径分析)
3. [关键知识点：为什么 cudaMemcpy 不能替代 RDMA](#3-关键知识点为什么-cudamemcpy-不能替代-rdma)
4. [Mooncake 已有基础设施盘点](#4-mooncake-已有基础设施盘点)
5. [社区相关工作](#5-社区相关工作)
6. [方案 A：POSIX SHM segment（PR #1913）](#6-方案-aposix-shm-segment-pr-1913)
7. [方案 B：GPU IPC handle 反向 pull](#7-方案-bgpu-ipc-handle-反向-pull)
8. [方案对比与最佳实践建议](#8-方案对比与最佳实践建议)
9. [附录：完整代码引用](#9-附录完整代码引用)

---

## 1. 问题场景与背景

### 1.1 vLLM + Mooncake Store 的部署架构

vLLM 的 MooncakeStoreConnector 当前使用 **owner/requester 分离架构**：

```
┌─────────────────────────────────────────────────────────┐
│                     同一物理节点                          │
│                                                          │
│  ┌────────────────────────┐   ┌──────────────────────┐  │
│  │  vLLM Worker           │   │  Mooncake Owner       │  │
│  │  (Requester 进程)       │   │  (独立后台进程)       │  │
│  │                        │   │                      │  │
│  │  GPU KV Cache          │   │  CPU Segment (80GB)  │  │
│  │  [register_buffer]     │   │  [MountSegment]      │  │
│  │                        │   │                      │  │
│  │  Python MooncakeStore  │   │  mooncake_client     │  │
│  │  Requester setup       │   │  --global_segment    │  │
│  │  (global_segment=0)    │   │                      │  │
│  └────────────┬───────────┘   └──────────┬───────────┘  │
│               │                           │              │
│               └──────────┬────────────────┘              │
│                          │                               │
│                          ▼                               │
│              ┌───────────────────────┐                   │
│              │  Mooncake Master      │                   │
│              │  (元数据、副本分配)    │                   │
│              └───────────────────────┘                   │
└─────────────────────────────────────────────────────────┘
```

- **Owner**: 持有大量 CPU 内存，作为 Mooncake Store 的 segment 提供方
- **Requester**: vLLM Worker，持有 GPU KV cache，向 Store put/get
- **Master**: 元数据服务，协调 segment 分配

### 1.2 核心 API 调用

```python
# vLLM Worker 侧
store.register_buffer(gpu_kv_cache_ptr, total_size)   # 注册 GPU 内存为 RDMA MR

# Prefill 之后
store.batch_put_from_multi_buffers(
    keys=["req1/layer0", ...],
    all_buffers=[[gpu_ptr_0], ...],
    all_sizes=[[131072], ...])   # 把 GPU 数据写到 owner 的 CPU segment

# Decode 之前
store.batch_get_into_multi_buffers(
    keys=[...],
    all_buffers=[[gpu_dst], ...],   # 读回 owner 的数据到 GPU
    all_sizes=[[131072], ...])
```

### 1.3 要解决的问题

**同节点 GPU→CPU 传输当前走 RDMA loopback**，浪费网卡带宽：
- 2 次 PCIe 穿越（GPU→NIC + NIC→CPU）
- 占用 NIC 带宽（本来应该留给真正的跨节点传输）
- 延迟 ~10-20us（相比直接 PCIe DMA 的 ~2-5us）

---

## 2. 当前数据路径分析

### 2.1 数据流（RDMA loopback）

```
  Requester (vLLM) 进程                      Owner 进程
  ┌───────────────────────┐                 ┌─────────────────────┐
  │  GPU VRAM             │                 │  CPU DRAM           │
  │  [KV cache]           │                 │  [Segment]          │
  │  gpu_ptr              │                 │  0x7f8000000000     │
  └───────────┬───────────┘                 └──────────▲──────────┘
              │                                        │
              │ PCIe DMA (读)                 PCIe DMA (写)
              ▼                                        │
  ┌───────────────────────────────────────────────────┴────────┐
  │                      RDMA NIC                               │
  │  ibv_post_send(WRITE, src=gpu_ptr, dst=remote_va+rkey)      │
  │                                                             │
  │  硬件 loopback：NIC 发出的包立即被自己收回                     │
  │  (不走网线，但仍消耗 NIC 队列、完成队列、带宽)                  │
  └─────────────────────────────────────────────────────────────┘

  总代价：
  - 2 × PCIe DMA
  - 1 × NIC loopback（NIC 硬件路径）
  - 占用 NIC queue pair、completion queue
  - 总延迟：10-20us
```

### 2.2 调用链（代码层面）

```
[Requester 进程]

  Python: store.batch_put_from_multi_buffers(keys, gpu_addrs, sizes)
    ↓
  [mooncake-integration/store/store_py.cpp]
  pybind11 → Slice 转换
    ↓
  [real_client.cpp]
  RealClient::batch_put_from_multi_buffers
    ↓
  [client_service.cpp]
  Client::BatchPut → 4 阶段协议：
    ├── StartBatchPut        → RPC to Master, 获得 Replica::Descriptor
    ├── SubmitTransfers      → TransferSubmitter::submit
    ├── WaitForTransfers     → 等待 future
    └── FinalizeBatchPut     → RPC to Master (PutEnd)
                  ↓
  [transfer_task.cpp]
  TransferSubmitter::selectStrategy
    ├── MC_STORE_MEMCPY=1 且 isLocalTransfer → LOCAL_MEMCPY
    └── 否则 → TRANSFER_ENGINE (RDMA)    ← 默认走这里
                  ↓
  [transfer_engine.cpp / rdma_transport]
  TransferEngine::submitTransfer
    ↓
  ibv_post_send(WRITE, source=gpu_ptr, target_addr=owner_va, rkey=...)
    ↓
  NIC 硬件 DMA：GPU VRAM → (PCIe) → NIC → (loopback) → NIC → (PCIe) → CPU DRAM
```

---

## 3. 关键知识点：为什么 cudaMemcpy 不能替代 RDMA

### 3.1 RDMA 和 cudaMemcpy 的本质差异

| | RDMA (ibv_post_send) | cudaMemcpy |
|-----|---------------------|----------|
| 执行者 | **NIC 硬件** | **本进程 CPU / GPU** |
| 地址空间要求 | 只要 NIC 能解析到物理地址（通过 rkey） | 两端地址必须都在**本进程虚拟地址空间**中 |
| 跨进程能力 | 原生支持（这就是它的目的） | 需要 `cudaIpcGetMemHandle` 或 shared memory |

### 3.2 RDMA loopback 为什么能工作（requester 不需要访问 owner 的 CPU）

**一句话**：RDMA 是让 NIC 硬件代你搬数据。你只需要告诉 NIC "从哪读、往哪写"，NIC 自己去读物理内存，不需要你的 CPU 解引用任何地址。

具体到我们场景：

```
  Requester 调用 ibv_post_send 时传 3 样东西给 NIC：
    ① source    = gpu_ptr（本进程 GPU 地址，已注册过）
    ② dest_addr = 0x7f800...（Owner 进程里的虚拟地址）
    ③ rkey      = Owner 注册时 NIC 发的"通行证"

  然后：
    - NIC 用 rkey 和 dest_addr 在自己的表里查到 Owner 那边的物理地址
    - NIC 用 source 地址在自己的表里查到 Requester 这边 GPU 的物理地址
    - NIC 直接 DMA 搬运，两个物理地址之间点对点
    - Requester CPU 从头到尾没碰过 0x7f800...
```

**关键**：`dest_addr` 对 requester 来说只是个"标签"，不是需要访问的地址。Owner 进程里它是真实地址，但 requester 从不用它去解引用，只是传给 NIC 当 token。

### 3.3 cudaMemcpy 为什么不能这么玩

**一句话**：cudaMemcpy 不是硬件代劳，是 **当前进程自己**去搬数据。既然是自己搬，两头地址都必须是本进程能寻址到的 —— 否则就是访问野指针。

对比：

```
  RDMA          cudaMemcpy
  ──────        ──────────
  NIC 搬        本进程搬
  看物理地址     看虚拟地址
  跨进程天然支持  跨进程必须先把地址映射进来
```

当我们尝试在 requester 里写：

```
  cudaMemcpy(
      dst = 0x7f800...,    ← Owner 进程的虚拟地址
      src = gpu_ptr,
      size, cudaMemcpyDeviceToHost)
```

CUDA 在 requester 进程里查 `0x7f800...` 是什么：
- 查不到（requester 的页表里根本没这个映射）
- 要么 SIGSEGV，要么 `cudaErrorInvalidValue`

所以要用 cudaMemcpy 替代 RDMA，**必须先让 requester 把 owner 的 CPU 地址"拉进"自己的地址空间** —— 这正是后面两个方案要解决的核心问题。

### 3.4 结论：要用 cudaMemcpy 替代 RDMA loopback，必须让两端在同一进程中可访问

有两条路：

1. **让 Owner 的 CPU 在 Requester 进程里可访问** → POSIX SHM (shm_open + MAP_SHARED)
   - Owner 用 `shm_open` 分配 → 写入 `/dev/shm/mooncake_seg_xxx`
   - Requester `shm_open` + `mmap` 同一文件 → 两进程指向同一物理页
   - Requester 的 cudaMemcpy(gpu_ptr → requester_mmap_va) 写到物理页，Owner 立即可见
   - **这是 PR #1913 的思路**

2. **让 Requester 的 GPU 在 Owner 进程里可访问** → CUDA IPC
   - Requester `cudaIpcGetMemHandle(gpu_ptr)` → 得到 handle
   - handle 通过 metadata 发给 owner
   - Owner `cudaIpcOpenMemHandle(handle)` → 在 owner 进程获得同一块 GPU 内存的虚拟地址
   - Owner 的 cudaMemcpy(owner_gpu_va → owner_cpu_va, DeviceToHost) 完成传输
   - **这是方案 B 的思路**

---

## 4. Mooncake 已有基础设施盘点

### 4.1 传输层已有能力

| Transport | 适用场景 | GPU↔GPU | GPU↔CPU 跨进程 | 同进程 CPU↔CPU |
|-----------|---------|---------|----------------|---------------|
| **RDMA** | 通用跨节点 | ✅ GPUDirect RDMA | ✅（走 NIC） | ✅（走 NIC） |
| **TCP** | 无 RDMA 环境 | ✅（含 cudaMemcpy staging，见 tcp_transport.cpp:122-137） | ✅ | ✅ |
| **NVLinkTransport (legacy)** | 机内 GPU-GPU | ✅ cudaIpcMemHandle + cudaMemcpy | ❌（只处理 cuda location） | ❌ |
| **IntraNodeNvlinkTransport** | 机内 GPU-GPU | ✅ cudaIpcMemHandle + cudaMemcpy | ❌ | ❌ |
| **TENT NVLinkTransport** | 机内 GPU-GPU/CPU | ✅ | 部分（只 host_register 但不跨进程映射） | ❌ |
| **TENT ShmTransport** | 机内 CPU-CPU | ❌ | ⚠️ 仅当 shm_path 已填入时 | ✅ |
| **TENT TcpTransport** | 同 TCP | ✅（含 cudaMemcpy） | ✅ | ✅ |

**关键发现**：**没有任何一个现成 transport 能原生做 "同节点 GPU(requester) → CPU(owner 跨进程)"**。

### 4.2 已有的跨进程共享基础设施

#### 4.2.1 CUDA IPC（用于 GPU 内存）

```cpp
// NVLinkTransport::addMemoryBuffer (nvlink_transport.cpp:210-266)
if (location.type() == "cuda") {
    CUdeviceptr base_ptr;
    cuMemGetAddressRange(&base_ptr, &alloc_size, (CUdeviceptr)desc.addr);
    cudaIpcMemHandle_t handle;
    cudaIpcGetMemHandle(&handle, (void*)base_ptr);
    desc.shm_path = serializeBinaryData(&handle, sizeof(handle));
    // handle 通过 metadata 广播给其他进程
}

// NVLinkTransport::relocateSharedMemoryAddress (nvlink_transport.cpp:299)
// 对端进程收到 handle 后：
deserializeBinaryData(buffer->shm_path, output_buffer);
cudaIpcMemHandle_t handle;
memcpy(&handle, output_buffer.data(), sizeof(handle));
cudaIpcOpenMemHandle(&shm_addr, handle, cudaIpcMemLazyEnablePeerAccess);
// shm_addr 是对端进程里可访问的 GPU 虚拟地址
```

**当前用在**：机内 GPU↔GPU。

#### 4.2.2 POSIX SHM + SCM_RIGHTS（用于 CPU 内存，Dummy→Real）

```cpp
// shm_helper.h (PR #1535 merged)
class ShmHelper {
    void *allocate(size_t size) {
        // memfd_create + mmap(MAP_SHARED)
    }
    // 通过 Unix socket SCM_RIGHTS 传递 fd
    int ipc_send_fd(int socket, int fd, ...);
    int ipc_recv_fd(int socket, ...);
};
```

**当前用在**：Dummy Client ↔ Real Client（不是 Owner ↔ Requester）。

#### 4.2.3 TENT ShmTransport（用于 CPU 内存跨进程）

```cpp
// shm_transport.cpp:179
Status ShmTransport::allocateLocalMemory(void** addr, size_t size, MemoryOptions& options) {
    options.shm_path = randomFileName();
    *addr = createSharedMemory(options.shm_path, size);  // shm_open + mmap
}

// shm_transport.cpp:148
Status ShmTransport::addMemoryBuffer(BufferDesc& desc, const MemoryOptions& options) {
    if (options.shm_path.empty())
        return Status::OK();  // ← 必须有 shm_path 才处理！
    desc.shm_path = options.shm_path;
    desc.transports.push_back(SHM);
}

// shm_transport.cpp:243
Status ShmTransport::relocateSharedMemoryAddress(uint64_t& dest_addr, ...) {
    // 对端进程用 shm_open + mmap 打开同一个 shm 文件
    int shm_fd = shm_open(buffer->shm_path.c_str(), O_RDWR, 0644);
    shm_addr = mmap(nullptr, buffer->length, PROT_READ|PROT_WRITE, MAP_SHARED, shm_fd, 0);
    dest_addr = dest_addr - buffer->addr + (uint64_t)shm_addr;
}
```

**当前状态**：代码已存在，但 `allocateLocalMemory(SHM)` 不被 mooncake-store 的 segment 分配路径调用。现有 segment 分配走 `malloc`/`mmap(MAP_PRIVATE)`，不是 SHM。

### 4.3 NUMA 感知能力

| 组件 | NUMA-aware? | 机制 |
|------|-------------|------|
| `allocate_buffer_numa_segments` (PR #1580 merged) | ✅ | `mbind(MPOL_BIND)` 按 NIC NUMA 分段绑定 |
| 普通 `malloc` / `mmap` | ❌ | Kernel first-touch |
| POSIX SHM (`shm_open + mmap MAP_SHARED`) | ❌ | First-touch（可以加 `mbind`） |
| RDMA Worker 线程 | ✅ | `bindToSocket(numa_socket_id_)` |
| TCP Worker / Memcpy Worker | ❌ | 通用 ThreadPool |

**关键**：如果新增 SHM segment 分配路径，需要也做 NUMA mbind，否则所有物理页落在一个 NUMA 节点，丢失 NUMA 优化。

---

## 5. 社区相关工作

### 5.1 已合并 PR

| PR | 内容 | 意义 |
|----|------|-----|
| #1580 | NUMA-aware global segment allocation (standalone mode) | 多 NIC 场景下 NIC 带宽打满 |
| #1838 | 扩展 #1580 到 RealClient-only mode | 覆盖 vLLM 场景 |
| #1468 | TENT NUMA prefault + RDMA MR warm-up | 启动加速 |
| #1535 | Dummy→Real memfd + SCM_RIGHTS 共享 hot cache | 为跨进程共享 CPU 内存奠定基础（ShmHelper） |
| #823 | vLLM+lmcache NUMA binding | P99 TTFT 大幅改善 |
| #1086 | Topology PCI 距离考虑 NUMA affinity | NIC 选择更智能 |

### 5.2 开放/关闭 PR

| PR | 状态 | 内容 |
|----|------|------|
| #1892 | OPEN | GPU pointer staging（PR #1913 以外另一条思路） |
| #1890 | OPEN | TENT: cudaMemcpyBatchAsync 批量优化 |
| #1913 | **OPEN（我们的）** | POSIX SHM segment 分配 + shm_path 透传 |
| #711 | CLOSED | 早期 NVLink transport auto-route（被关闭，换成了新的 IntraNodeNvlinkTransport） |

### 5.3 相关 Issue

| Issue | 内容 |
|-------|------|
| #162 | "Refactoring TE" — roadmap 明确列出 "SHM transport for same-node" |
| #68 | "PD same node protocol" — 团队承诺支持 SHM transport |
| #90 | "Currently Mooncake supports cudaMemcpy if same process. We are investigating cross-process same-machine." |
| #708 | "CPU↔GPU in machine" — 确认 GPUDirect RDMA 是当前方案 |
| #1058 | TENT Roadmap — "Runtime integration with Mooncake Store" |
| #1163 | "Decouple TransferTask from TE" — ITransferEngine 抽象 |
| #1883 | Overall Roadmap |

---

## 6. 方案 A：POSIX SHM segment（PR #1913）

### 6.1 思路

让 Owner 用 `shm_open + mmap(MAP_SHARED)` 分配 CPU segment，让 Requester 也能 `shm_open` 打开同一块内存，然后 Requester 侧直接 `cudaMemcpy(gpu_ptr → owner_shm_addr_in_requester)`。

### 6.2 数据流

```
  启动阶段:
    Owner 进程:
      ptr = shm_open("/dev/shm/mooncake_seg_abc") + ftruncate + mmap
      MountSegment(ptr, size, shm_path="mooncake_seg_abc")
      → registerLocalMemory(ptr, size, shm_path="mooncake_seg_abc")
      → ShmTransport::addMemoryBuffer(desc.shm_path = "mooncake_seg_abc")
      → Master 广播 segment 描述符（含 shm_path）

    Requester 进程（首次传输时）:
      收到 BufferDesc（包含 shm_path）
      ShmTransport::relocateSharedMemoryAddress:
        fd = shm_open("mooncake_seg_abc", O_RDWR)
        requester_va = mmap(NULL, size, PROT_RW, MAP_SHARED, fd, 0)
        [requester_va 和 owner_va 指向同一块物理页]

  传输阶段:
    Requester 调 batch_put(gpu_ptr → owner_remote_va)
    TransferSubmitter 选择 ShmTransport（或新的逻辑）
    cudaMemcpy(
        dst = requester_va + offset,    ← shared mmap，requester 进程能访问
        src = gpu_ptr,                  ← 本进程 GPU
        size, cudaMemcpyDeviceToHost)
    → 一次 PCIe DMA 完成，物理页直接被 owner 看到
```

### 6.3 架构图

```
  Requester (vLLM) 进程              Owner 进程
  ┌─────────────────────────┐        ┌──────────────────────┐
  │  GPU VRAM               │        │                      │
  │  gpu_ptr                │        │                      │
  │  [本进程可访问]          │        │                      │
  │           │             │        │                      │
  │           │ cudaMemcpy  │        │                      │
  │           ▼ D2H         │        │                      │
  │  requester_shm_va       │        │  owner_shm_va         │
  │  (mmap MAP_SHARED)      │        │  (mmap MAP_SHARED)   │
  │           │             │        │           │           │
  └───────────┼─────────────┘        └───────────┼──────────┘
              │                                   │
              └────── 同一块物理页 ────────────────┘
                 /dev/shm/mooncake_seg_abc

  每次传输代价：
  - 1 × PCIe DMA (GPU → CPU)
  - 0 × NIC usage
  - 延迟：2-5us
```

### 6.4 优点

- ✅ 不依赖 owner 进程链接 CUDA runtime（owner 可以是纯 CPU 进程）
- ✅ 完全复用 Mooncake 已有 ShmTransport 和 shm_path 机制
- ✅ POSIX SHM 是稳定成熟的 Linux API
- ✅ 跨进程共享非常直观

### 6.5 缺点

- ❌ Owner 必须改 segment 分配路径（从 malloc → shm_open）
- ❌ Requester 侧 CUDA runtime 要能访问这块 shared memory —— 需要 `cudaHostRegister` 才能得到最佳 D2H 性能（可选）
- ❌ 需要 NUMA-aware：shm_open 分配的内存默认 kernel first-touch，需要加 `mbind`，否则丢失 PR #1580 的 NUMA 优化
- ❌ `/dev/shm` 默认大小限制（tmpfs size），需要足够大
- ❌ 进程崩溃后 `/dev/shm/mooncake_seg_*` 需要清理（shm_unlink）

### 6.6 已完成的代码

kvcache-ai/Mooncake#1913 包含：
- `TransferEngine::registerLocalMemory` 增加 `shm_path` 参数
- `Client::MountSegment` / `Client::RegisterLocalMemory` 增加 `shm_path`
- `real_client.cpp` 增加 `allocate_shm_segment()` + `ShmSegmentDeleter`
- TENT 路径自动透传 `MemoryOptions.shm_path`
- 通过 `MC_USE_TENT=1` 或 `MC_STORE_SHM_SEGMENTS=1` 启用

---

## 7. 方案 B：GPU IPC handle 反向 pull

### 7.1 思路

**反转数据流方向**：不让 Requester 把数据"推"给 Owner，而是让 Owner 主动从 Requester 的 GPU "拉"数据。

- Requester: `register_buffer(gpu_ptr)` 触发 `cudaIpcGetMemHandle(gpu_ptr)`（NVLinkTransport 已自动做）
- 通过新 RPC `put_from_gpu_ipc` 通知 Owner："这是 handle，你去拉"
- Owner: `cudaIpcOpenMemHandle(handle)` + `cudaMemcpy(owner_cpu_va, owner_gpu_va, size, DeviceToHost)`

### 7.2 重大发现：已有 70% 的基础设施可直接复用

深入调研后发现，**Mooncake 已经有完整的 client-to-client 数据面 RPC 基础设施**，原本为 SSD offload 设计，完全可以套用到方案 B：

**`batch_get_offload_object` 链路**（已存在，生产级代码）：
- `mooncake-store/include/pyclient.h:61-128` — `ClientRequester` 类
- `mooncake-store/src/real_client.cpp:3649-3756` — 完整 RPC 实现
- `mooncake-store/include/rpc_types.h:146-150` — `BatchGetOffloadObjectResponse`

**如何工作**：
- 发起方直接调 peer 的 `local_rpc_addr`（不走 Master）
- RPC 通道用 `coro_rpc`，每个 RealClient 都有 `offload_rpc_server_`
- 已支持 TCP 和 RDMA（`init_ibv()`）
- 清理机制 `release_offload_buffer()` 已做成 fire-and-forget

**这就是"反向 pull"的现成模型**，当前用在 "remote SSD pull"，只要加一条新 RPC method 就能做 "remote GPU IPC pull"。

### 7.3 原语清单（全部已存在）

| 组件 | 现状 | 文件位置 |
|------|-----|---------|
| Client-to-Client RPC 通道 | ✅ 生产可用 | `pyclient.h:ClientRequester`, `real_client.cpp:offload_rpc_server_` |
| `cudaIpcGetMemHandle` 生成 | ✅ 已用于 GPU↔GPU | `nvlink_transport.cpp:245-250` |
| `cudaIpcOpenMemHandle` 对端打开 | ✅ 已用于 GPU↔GPU | `nvlink_transport.cpp:344` |
| P2P bidirectional access | ✅ 已启用 | `intranode_nvlink_transport.cpp:57-95` |
| GPU pointer 检测 + SetDevice 工具 | ✅ PR #1892 提供 | `gpu_staging_utils.h` (OPEN PR) |
| `cudaMemcpyBatchAsync` 批量优化 | ✅ PR #1890 提供 | (OPEN PR) |
| 4 阶段 put 协议 | ✅ 不需要改 | `client_service.cpp:Client::BatchPut` |
| Master 元数据广播 | ✅ IPC handle 已在 BufferDesc.shm_path | 不需要改 |

**真正需要新写的**：一条 RPC method（约 100 行 C++）。

### 7.4 数据流

```
  启动阶段（零新代码，全部复用）:
    Requester 进程:
      register_buffer(gpu_ptr, size)
        → NVLinkTransport::addMemoryBuffer (location="cuda")
        → cudaIpcGetMemHandle(&handle, gpu_ptr)     ← 已有代码
        → BufferDesc.shm_path = serialize(handle)   ← 已有代码
        → Master 广播到所有 peer                     ← 已有代码

    Owner 进程: 不需要启动时做任何事，handle 按需首次使用时才 open

  传输阶段（Put）:
    Requester: batch_put_from_multi_buffers(gpu_ptr, size → key)
      → Client::BatchPut
      → StartBatchPut (Master 分配 owner 侧空间)           ← 已有
      → SubmitTransfers:
          检测: local + GPU source + CPU dest → 走新路径     ← 新 ~40 行
          ClientRequester::put_from_gpu_ipc(owner_rpc_addr, ← 新 ~30 行
              {keys, gpu_ipc_handle, src_offset, dst_addr, size})
      → WaitForTransfers: 等 RPC 返回
      → FinalizeBatchPut (PutEnd)                           ← 已有

    Owner: RPC handler RealClient::put_from_gpu_ipc()       ← 新 ~60 行
      → 查 IPC handle cache，首次调 cudaIpcOpenMemHandle
      → cudaMemcpy(owner_cpu_va, owner_gpu_va, size, D2H)
      → 返回 per-key status
```

### 7.5 架构图

```
  Requester (vLLM) 进程                 Owner 进程
  ┌──────────────────────┐             ┌─────────────────────────────┐
  │  GPU VRAM            │             │  CPU Segment (普通 malloc)  │
  │  gpu_ptr (cudaMalloc)│             │  owner_cpu_va              │
  │                      │             │                             │
  │  cudaIpcGetMemHandle │ handle 通    │  [首次]                     │
  │  (在 register_buffer │ 过 metadata  │  cudaIpcOpenMemHandle       │
  │   时自动生成)         │ 传播         │     ↓                       │
  │                      │─────────────>│  owner_gpu_va (IPC opened)  │
  │                      │             │  [之后缓存复用]               │
  │                      │             │     │                       │
  │                      │  put_from_   │     │                       │
  │                      │  gpu_ipc RPC │     │                       │
  │                      │  ──────────> │     │ cudaMemcpy(D2H)       │
  │                      │  (coro_rpc)  │     ▼                       │
  │                      │             │  owner_cpu_va ← owner_gpu_va │
  │                      │             │  (Owner 进程内完成, GPU DMA) │
  │                      │<── ACK ───── │                             │
  └──────────────────────┘             └─────────────────────────────┘

  每次传输代价：
  - 1 × PCIe DMA (GPU → CPU)
  - 1 × coro_rpc round-trip（~几十微秒）
  - 0 × NIC 带宽占用
  - 延迟：3-8us（含 RPC）
```

### 7.6 代码改动清单（最简实践）

总改动量 **~180 行 C++ + 20 行 header**。

#### 改动 1：`rpc_types.h` 新增消息（~20 行）

```cpp
// mooncake-store/include/rpc_types.h
struct PutFromGpuIpcRequest {
    std::vector<std::string> keys;
    std::string gpu_ipc_handle;     // 序列化的 cudaIpcMemHandle_t
    int gpu_device_id;
    std::vector<uint64_t> src_offsets;
    std::vector<uint64_t> dst_addrs;  // owner 自己的 CPU 地址
    std::vector<uint64_t> sizes;
};
struct PutFromGpuIpcResponse {
    std::vector<int32_t> per_key_status;  // 0=OK, <0=errorcode
};
```

#### 改动 2：Owner 侧 RPC handler（~60 行）

```cpp
// mooncake-store/src/real_client.cpp
// 使用 unordered_map 缓存已打开的 handle，避免重复 cudaIpcOpenMemHandle
tl::expected<PutFromGpuIpcResponse, ErrorCode>
RealClient::put_from_gpu_ipc(const PutFromGpuIpcRequest& req) {
    std::lock_guard<std::mutex> lock(ipc_cache_mutex_);
    void* owner_gpu_va = nullptr;
    auto it = ipc_handle_cache_.find(req.gpu_ipc_handle);
    if (it == ipc_handle_cache_.end()) {
        cudaIpcMemHandle_t h;
        deserializeBinaryData(req.gpu_ipc_handle, h);
        gpu_staging::SetDevice(req.gpu_device_id);   // 复用 PR #1892
        cudaError_t err = cudaIpcOpenMemHandle(
            &owner_gpu_va, h, cudaIpcMemLazyEnablePeerAccess);
        if (err != cudaSuccess) return tl::unexpected(ErrorCode::TRANSFER_FAIL);
        ipc_handle_cache_[req.gpu_ipc_handle] = owner_gpu_va;
    } else {
        owner_gpu_va = it->second;
    }
    PutFromGpuIpcResponse resp;
    for (size_t i = 0; i < req.keys.size(); ++i) {
        cudaError_t err = cudaMemcpy(
            reinterpret_cast<void*>(req.dst_addrs[i]),
            static_cast<char*>(owner_gpu_va) + req.src_offsets[i],
            req.sizes[i], cudaMemcpyDeviceToHost);
        resp.per_key_status.push_back(err == cudaSuccess ? 0 : toInt(ErrorCode::TRANSFER_FAIL));
    }
    return resp;
}

// 注册：setup_internal 中
offload_rpc_server_->register_handler<&RealClient::put_from_gpu_ipc>(this);
```

#### 改动 3：Requester 侧 ClientRequester（~30 行）

```cpp
// 同 batch_get_offload_object 模式，一行 invoke_rpc 模板即可
tl::expected<PutFromGpuIpcResponse, ErrorCode>
ClientRequester::put_from_gpu_ipc(const std::string& owner_rpc_addr,
                                   const PutFromGpuIpcRequest& req) {
    return invoke_rpc<&WrappedRealClientService::put_from_gpu_ipc,
                      PutFromGpuIpcResponse>(owner_rpc_addr, req);
}
```

#### 改动 4：`SubmitTransfers` 路由（~40 行）

```cpp
// mooncake-store/src/client_service.cpp
void Client::SubmitTransfers(std::vector<PutOperation>& ops) {
    for (auto& op : ops) {
        if (op.IsResolved()) continue;
        if (op.replicas.empty()) { op.SetError(...); continue; }

        // 新增：检测同节点 GPU→CPU + source 有 IPC handle → 反向 pull
        if (ShouldUseGpuIpcPull(op)) {
            op.pending_transfers.emplace_back(SubmitGpuIpcPull(op).value());
            continue;
        }
        // ... existing RDMA/memcpy path ...
    }
}
// ShouldUseGpuIpcPull 的条件：
//   1. MC_STORE_GPU_IPC_PULL=1 启用
//   2. op.slices[0].ptr 是 GPU 地址（cudaPointerGetAttributes）
//   3. op.replicas[0].transport_endpoint IP == local IP
//   4. local BufferDesc 里有 shm_path（IPC handle）
```

#### 改动 5：IPC handle 读取（~10 行）

Requester 侧查询本地 segment 描述符拿 `shm_path`（handle 序列化）。NVLinkTransport 已填好，**不需要新代码**。

### 7.7 优点

- ✅ **最小新代码**：~180 行，相比方案 A（~250 行）更少
- ✅ **复用生产级基础设施**：`offload_rpc_server_` + `ClientRequester` 模式已在 SSD 路径验证
- ✅ **完全复用 NVLinkTransport 的 handle 生成**（零改动）
- ✅ **复用 PR #1892 的 `gpu_staging_utils`**（可先本地 cherry-pick）
- ✅ Owner 的 CPU segment **不需要改分配方式**（仍然 malloc）
- ✅ GPU DMA 引擎做 D2H，性能接近硬件上限
- ✅ 不污染 `/dev/shm`，不引入 POSIX SHM 依赖
- ✅ NUMA 友好（CUDA DMA 跨 NUMA 透明）

### 7.8 缺点和风险

- ❌ **Owner 进程必须链接 CUDA runtime**（关键约束）
  - Owner 在 GPU 节点一般都有 libcuda，但纯 CPU 节点部署不可行
  - CUDA context 初始化占 ~200MB GPU memory
- ⚠️ **IPC handle 缓存需要淘汰**：`unregister_buffer` 时要给 peer 发 `ipc_close` 通知
- ⚠️ **CUDA IPC per-process 单次 open 限制**：同一 handle 在 owner 进程只能 open 一次 —— 用缓存即可
- ⚠️ **cudaIpcOpenMemHandle 首次开销**（~几百 us）：缓存后仅首次传输承担
- ⚠️ **RPC 是串行的**：一次 `put_from_gpu_ipc` 串行执行一批 cudaMemcpy —— 但已经是 batch，和 RDMA 的 batch 量级相当

### 7.9 OPEN PR 协同

**可以直接依赖的 OPEN PR**：

| PR | 作用 | 方案 B 需要的部分 |
|----|-----|----------------|
| #1892 | `gpu_staging_utils.h` + `PinnedBufferPool` | 复用 `SetDevice`, `IsDevicePointer` |
| #1890 | `cudaMemcpyBatchAsync` 批量 | 替换 for 循环里的 `cudaMemcpy`，进一步提速 |

**这两个合入后，方案 B 代码量可以更少**（直接 include 新 header）。

### 7.10 为什么方案 B 比早先估计的"大手术"要小

当时认为方案 B 需要：
1. 给 Owner 加数据平面 RPC（不只是 Master 控制面）
2. 改架构 push→pull

调研结果：**两项都已经在 Mooncake 中存在**，只是当前用在 SSD offload 场景：
- `offload_rpc_server_` 就是 owner 侧的数据平面 RPC 服务器
- `batch_get_offload_object` 已经是 "owner-pull-from-remote" 模式（只是为 remote file storage 做的）

我们只是在同一个 RPC 框架里加**一个新 method**。

---

## 8. 方案对比与最佳实践建议

### 8.1 方案对比矩阵（基于方案 B 深度调研后更新）

| 维度 | 方案 A (SHM) | 方案 B (GPU IPC reverse pull) |
|------|-------------|-----------------------------|
| Owner 是否需要 CUDA runtime | ❌ 不需要 | ✅ 必须 |
| Owner segment 分配方式改动 | ✅ malloc → shm_open | ❌ 不改（仍然 malloc） |
| 架构层面改动 | Segment 分配链路 + shm_path 透传 | **加一条 RPC method**（复用 offload RPC 框架） |
| GPU DMA 引擎利用 | ⚠️（Requester CPU 线程 cudaMemcpy） | ✅（Owner 内部 cudaMemcpy，GPU DMA 引擎） |
| 复用已有基础设施 | TENT ShmTransport | **NVLinkTransport IPC + offload RPC**（~70%） |
| 代码改动量 | ~250 行 | **~180 行** |
| NUMA 友好性 | 需要自己加 mbind | 自动（CUDA DMA 跨 NUMA 透明） |
| 进程崩溃清理 | 需要 shm_unlink | CUDA IPC 自动（进程退出释放 handle） |
| Owner 为纯 CPU 节点可用 | ✅ | ❌ |
| vLLM 当前部署契合度 | ✅（部署灵活） | ✅（vLLM owner 通常在 GPU 节点） |
| 依赖其他 OPEN PR | 无 | 可选依赖 #1892 (utils) / #1890 (batch memcpy) |

### 8.2 性能预估

| 方案 | PCIe 跳数 | NIC 使用 | 线程执行者 | 延迟预估 |
|------|----------|---------|----------|----------|
| 当前 RDMA loopback | 2 | 是（loopback） | NIC 硬件 | 10-20us |
| 方案 A (SHM) | 1 | 否 | Requester 的 MemcpyWorkerPool | 2-5us |
| 方案 B (GPU IPC pull) | 1 | 否 | Owner 的 CUDA runtime | 3-8us（含 RPC） |

### 8.3 最佳实践建议

**短期（立即可做）**：

1. **先在不改代码的情况下跑基线**：用 `MC_USE_TENT=1` + `MC_INTRANODE_NVLINK=1` 测量当前同节点 GPU→CPU 传输性能，确认 RDMA loopback 真的是瓶颈
2. **等待 PR #1892 合并**：它解决 GPU pointer + CPU 线程 memcpy segfault 问题（磁盘 offload 路径），是独立于本问题的 bug 修复

**中期（3-4 周）**：选一个方案推进。

**推荐方案 B（GPU IPC reverse pull）** 的理由（基于深度调研更新）：
- **代码量更少**（~180 行 vs ~250 行）
- **复用已有生产级 RPC 基础设施**（offload_rpc_server_ + ClientRequester，已在 SSD 路径验证）
- **完全复用 NVLinkTransport 的 IPC handle 生成**，零改动
- Owner 的 CPU segment 不需要改分配方式（调试容易、部署简单）
- 不引入 POSIX SHM 依赖，不污染 `/dev/shm`
- vLLM 场景下 Owner 通常在 GPU 节点，CUDA runtime 本来就有
- GPU DMA 引擎做 D2H，理论性能更好

**仍选 方案 A（SHM）** 的情况：
- Owner 必须部署在纯 CPU 节点（无 libcuda 环境）
- 不想引入新的数据面 RPC method
- Mooncake roadmap 对 SHM transport 已有长期规划（issue #162）

**不推荐同时做两个**：重复造轮子，增加维护负担。

### 8.4 长期（6+ 周）

向 Mooncake 上游提 RFC，提议：

1. 让 `TransferEngine::registerLocalMemory` 的 TENT 路径自动调用 `allocateLocalMemory(SHM)` 返回的 shm_path（需要 Mooncake 的 segment 分配链集成到 TENT）
2. 补全 `MemcpyWorkerPool` 的 NUMA 绑核（对标 PR #823 在 vLLM 做的事）
3. `ShmTransport` 分配时走 `mbind` 做 NUMA 分段（对标 PR #1580 对 malloc 段的做法）

这些工作的前置条件是 **issue #1163 (Decouple TransferTask from TE)** 的 ITransferEngine 抽象——等它完成，Mooncake Store 可以通过统一接口使用 TENT 的所有能力。

---

## 9. 附录：完整代码引用

### 9.1 RDMA loopback 的关键代码

```cpp
// mooncake-transfer-engine/src/transport/rdma_transport/rdma_endpoint.cpp:155-174
// 同 NIC 自连接的 loopback handshake
if (context_.nicPath() == peer_nic_path_) {
    auto segment_desc = context_.engine().meta()->getSegmentDescByID(LOCAL_SEGMENT_ID);
    if (segment_desc) {
        for (auto &nic : segment_desc->devices)
            if (nic.name == context_.deviceName())
                return doSetupConnection(nic.gid, nic.lid, qpNum());
    }
}
```

### 9.2 isLocalTransfer 判定

```cpp
// mooncake-store/src/transfer_task.cpp:784-796
bool TransferSubmitter::isLocalTransfer(
    const AllocatedBuffer::Descriptor& handle) const {
    std::string local_ep = engine_.getLocalIpAndPort();
    std::string local_ip = extractIpAddress(local_ep);
    if (!local_ep.empty()) {
        std::string handle_ip = extractIpAddress(handle.transport_endpoint_);
        return !handle.transport_endpoint_.empty() && handle_ip == local_ip;
    }
    return false;
}
```

### 9.3 TENT 的 transport 选择逻辑

```cpp
// mooncake-transfer-engine/tent/src/runtime/transfer_engine_impl.cpp:799-837
TransportType TransferEngineImpl::getTransportType(const Request& request, int priority) {
    // ...
    for (auto type : entry->transports) {
        if ((type == NVLINK || type == SHM) && !same_machine) continue;
        if (checkAvailability(transport_list_[type], local_mtype, remote_mtype)) {
            if (priority-- == 0) return type;
        }
    }
    return UNSPEC;
}
```

### 9.4 ShmTransport 的 shm_path 依赖

```cpp
// mooncake-transfer-engine/tent/src/transport/shm/shm_transport.cpp:148-159
Status ShmTransport::addMemoryBuffer(BufferDesc &desc, const MemoryOptions &options) {
    if (options.shm_path.empty())
        return Status::OK();  // 不注册 → entry->transports 里不会有 SHM
    desc.shm_path = options.shm_path;
    desc.transports.push_back(TransportType::SHM);
    return Status::OK();
}
```

### 9.5 NVLinkTransport 对 CPU location 的处理

```cpp
// mooncake-transfer-engine/tent/src/transport/nvlink/nvlink_transport.cpp:210-265
Status NVLinkTransport::addMemoryBuffer(BufferDesc& desc, const MemoryOptions& options) {
    LocationParser location(desc.location);
    if (location.type() == "cuda") {
        // GPU 路径：cudaIpcGetMemHandle → desc.shm_path
        cudaIpcMemHandle_t handle;
        cudaIpcGetMemHandle(&handle, (void*)base_ptr);
        desc.shm_path = serializeBinaryData(&handle, sizeof(handle));
        desc.transports.push_back(NVLINK);
    } else if (location.type() == "cpu" || location.type() == kWildcardLocation) {
        // CPU 路径：只做 cudaHostRegister（本进程 pin 内存）
        if (host_register_)
            cudaHostRegister(((void*)desc.addr), desc.length, cudaHostRegisterDefault);
        // ← 不调用 cudaIpcGetMemHandle！不添加跨进程共享机制！
    }
    desc.transports.push_back(NVLINK);
    return Status::OK();
}
```

**关键发现**：NVLinkTransport 虽然给 CPU 加了 NVLINK tag，但没有给跨进程共享能力。当 requester 要传数据到 owner 的 CPU 时，NVLinkTransport 的 `relocateSharedMemoryAddress` 会在 `location.type() != "cuda"` 时返回 InvalidArgument（line 333-336），导致选其他 transport（最终落到 RDMA）。

### 9.6 方案 B 需要的 NVLinkTransport 扩展（伪代码）

```cpp
// addMemoryBuffer for CPU location: 也生成 cudaHostGetDevicePointer + IPC handle
else if (location.type() == "cpu") {
    // 方法 1: cudaHostAlloc(cudaHostAllocPortable) 分配 + cudaIpcGetMemHandle？
    //         实际上 cudaHostMalloc 的 CPU 地址不能直接 IPC
    // 方法 2: 让 Owner 主动从 requester GPU IPC 拉数据（反向）
    //         此时 owner 的 CPU segment 不需要任何特殊处理
    //         但需要 Owner 进程级别的数据面 RPC 接收 "pull" 指令
    // ...
}
```

### 9.7 CUDA IPC 的跨进程使用模式

```cpp
// 进程 A (Requester，持有 GPU 内存)
void *gpu_ptr;
cudaMalloc(&gpu_ptr, size);
cudaIpcMemHandle_t handle;
cudaIpcGetMemHandle(&handle, gpu_ptr);
// 将 handle（64 字节）通过 IPC/socket 发给进程 B

// 进程 B (Owner)
cudaSetDevice(0);  // 必须和进程 A 的 GPU 一致
void *shared_gpu_ptr;
cudaIpcOpenMemHandle(&shared_gpu_ptr, handle, cudaIpcMemLazyEnablePeerAccess);
// shared_gpu_ptr 在进程 B 中有效，指向进程 A 的同一块 GPU 内存
cudaMemcpy(owner_cpu_va, shared_gpu_ptr, size, cudaMemcpyDeviceToHost);
// ... 使用后:
cudaIpcCloseMemHandle(shared_gpu_ptr);
```

---

## 附注

**本文档基于 Mooncake commit `0a9c993`** (`[PG][TENT] Fix CUDA collective wait semantics and NVLink small-transfer completion (#1863)`) 分析。后续 Mooncake 主线变化可能影响具体行号和 API 签名，但主要架构判断应保持稳定。

**讨论记录的关键澄清**：
1. Requester **不需要**在 RDMA 路径下访问 Owner 的 CPU（NIC 硬件代办）
2. Requester **必须**访问 Owner 的 CPU，**仅当**想用 cudaMemcpy 替代 RDMA（这是 cudaMemcpy 的限制）
3. 方案 A 和方案 B 都解决"同节点避免 NIC loopback"问题，但方向不同
4. Mooncake 已有基础设施中，**没有任何 transport** 原生支持 "同节点 GPU(requester) → CPU(owner 跨进程)"，这正是我们要填补的空白
