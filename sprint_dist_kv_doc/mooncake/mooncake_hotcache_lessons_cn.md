# Mooncake Local Hot Cache 集成教训

**日期**: 2026-04-24
**作者**: aoshen（调研记录）
**主题**: 启用 `MC_STORE_LOCAL_HOT_CACHE_SIZE > 0` 时 vLLM engine 静默冻结的复现、两条 mitigation 与 page-fault storm 假设

**相关实验材料**:
- 实验流水账: `solo_session/hot_cache_debug/FINAL_hotcache.md`、`solo_session/hot_cache_debug/progress.md`、`solo_session/hot_cache_debug/results.tsv`
- batch1 日志: `/home/aoshen/vigil_logs/hotcache_debug_batch1_1777045754/`
- iter1 (local) 日志: `/home/aoshen/vigil_logs/hotcache_debug_iter1_1777045513/`

**Mooncake 源码引用** (本地路径 `Mooncake/`):
- `mooncake-store/include/client_service.h:533-541`（`ShouldAdmitToHotCache` 内联实现） + `client_service.cpp:811, 981, 1112`（admission 调用点）
- `mooncake-store/src/client_service.cpp:635`（`InitLocalHotCache` 在 Client 构造时被无条件调）
- `mooncake-store/src/client_service.cpp:3061-3122`（`InitLocalHotCache` 实现起点 3061，env 读取段在 3081-3122）
- `mooncake-store/include/count_min_sketch.h:14-86`（CMS 实现）
- `mooncake-store/src/local_hot_cache.cpp:19-65`（`LocalHotCache::LocalHotCache` 构造函数 malloc/memfd 二分支）
- `mooncake-store/src/local_hot_cache.cpp:161`（`LocalHotCache::GetFreeBlock`）
- `mooncake-store/src/local_hot_cache.cpp:279`（`LocalHotCacheHandler::SubmitPutTask`）
- `mooncake-store/src/local_hot_cache.cpp:344`（`LocalHotCacheHandler::workerThread`）
- `mooncake-store/include/shm_helper.h:33-72` + `mooncake-store/src/shm_helper.cpp`（ShmHelper memfd 分配）

---

> 本文是本目录里 **Mooncake local hot cache** 的 canonical 文档。其它部署 / YAML / 数据路径笔记只保留简述和链接，机制细节统一放这里。

## 0. 机制总览

Hot cache 是 Mooncake 里一个容易被误解的子系统。它不是 Mooncake memory replica，也不受 master 管理；它是 client 进程内的本地缓存层。

它的工作可以拆成三步：

1. Get 一个 key 时，先看这个 key 是否已经在本 client 的 hot cache 里。
2. 如果没命中，就正常从 Mooncake replica 读；读完后用 CMS 判断这个 key 是否已经“热到值得缓存”。
3. 如果后续命中 hot cache，就把读路径改成本进程内存拷贝，不再走远端 replica / RDMA。

### 0.1 它不是 Mooncake replica，master 完全不感知

| 维度 | Mooncake replica (MEMORY/DISK) | Hot cache block |
|---|---|---|
| master 知道它存在吗？ | 是，走 `MountSegment` 注册 | 否，client 私有，从不上报 |
| 跨 client 可见？ | 是，其他 client 查 master 能发现 | 否，默认进程私有 |
| 生命周期 | 租约 + eviction 管理 | client 进程退出即消失 |
| 出现在 `GetReplicaList` 返回里？ | 是 | 否 |
| `ReplicaType` 枚举涵盖？ | `MEMORY` / `DISK` / `LOCAL_DISK` | 无 HOT_CACHE 类型 |
| 容量控制 | `MOONCAKE_GLOBAL_SEGMENT_SIZE` | `MC_STORE_LOCAL_HOT_CACHE_SIZE` |

判断它是不是 replica，最直接看两点：master 不知道它，其他 client 默认也查不到它。所以它不能提供副本冗余，也不能替代 Mooncake 的全局 memory replica。

### 0.2 启用条件与配置面

`Client` 构造时会无条件调用 `InitLocalHotCache()`，但 `MC_STORE_LOCAL_HOT_CACHE_SIZE` 默认为 0；size 为 0 时 early return，`hot_cache_` / `admission_sketch_` / handler 全部为空。因此 hot cache **默认关闭**，只有显式设置容量才启用。

| env / 参数 | 作用 |
|---|---|
| `MC_STORE_LOCAL_HOT_CACHE_SIZE` | hot cache 总容量，单位字节；0 表示关闭 |
| `MC_STORE_LOCAL_HOT_BLOCK_SIZE` | 单个 hot cache block 大小，默认 16 MiB |
| `MC_STORE_LOCAL_HOT_ADMISSION_THRESHOLD` | CMS 准入阈值，默认 2 |
| `MC_STORE_LOCAL_HOT_CACHE_USE_SHM` / `use_shm` | 用 memfd shared memory 分配 hot cache region |

`use_shm=true` 要分部署形态看：

- 在 daemon + dummy client 模式下，Mooncake daemon 是一个独立进程，vLLM worker / dummy client 是另一个进程。普通 `malloc` 出来的内存只能被 daemon 自己看到；如果希望 dummy client 也能映射同一块 hot cache，就需要 shared memory。`use_shm=true` 的本意就是做这件事。
- 在 vLLM in-process linked client 模式下，Mooncake client 直接跑在 vLLM 进程里，没有另一个进程要共享这块 hot cache。因此这里并不需要 shared memory。

实验里 `use_shm=true` 能缓解 stall，关键不是“共享”本身，而是它这条分配路径顺手用了 `MAP_POPULATE`：内存映射创建时就让内核把页准备好，后面并发写 hot cache 时不再集中触发大量 page fault。

### 0.3 热化触发：CMS 频率准入，不是访问一次就缓存

CMS 是 Count-Min Sketch。放在这里可以先把它理解成一个**省内存的访问次数估计器**。

它不为每个 key 单独存一个精确 counter，而是维护几排小 counter。每次访问一个 key 时，用几组 hash 函数把这个 key 映射到每一排里的一个 counter，然后把这些 counter 都加 1。查询这个 key 的访问次数时，再用同样的 hash 位置读出这些 counter，并取里面的最小值作为估计次数。

为什么取最小值：不同 key 可能 hash 到同一个 counter，导致某些 counter 被别的 key 一起加大。取最小值可以尽量避开被碰撞污染得最严重的 counter。这个估计值可能比真实访问次数偏大，但通常不会偏小。

它在 hot cache 里的作用只有一个：决定一个 key 是否够热、值不值得放进 hot cache。

默认 `admission_threshold=2` 的意思是：

1. 第一次从 remote / replica 读到某个 key：只记一次访问，不放 hot cache。
2. 第二次再读到同一个 key：CMS 估计次数达到阈值，才允许放进 hot cache。

所以 hot cache 默认不是“读一次就缓存”，而是“至少重复访问过才缓存”。

```cpp
bool ShouldAdmitToHotCache(const std::string& key, bool cache_used) {
    if (!(hot_cache_ && !cache_used)) return false;
    if (admission_sketch_ == nullptr) return true;
    return admission_sketch_->increment(key) >= admission_threshold_;
}
```

语义：

- `cache_used=true`：本次已经从 hot cache 命中，不 re-promote，也不动 CMS counter。
- `cache_used=false`：本次从 remote / replica 拉取，才让 CMS 计一次访问。
- `admission_sketch_ == nullptr`：未启 CMS 时每次都准入。

核心设计是“频繁访问的 key 才缓存”，天然过滤 one-shot key，避免 cache 污染。

### 0.4 完整热化路径

```text
Client::Get(key)
  ├─ FindFirstCompleteReplica(replicas, &replica)
  ├─ if (hot_cache_ && replica.is_memory_replica())
  │    cache_used = RedirectToHotCache(key, replica)
  ├─ TransferRead(replica, slices)                  # hit = local memcpy, miss = RDMA
  │
  └─ if (ShouldAdmitToHotCache(key, cache_used))
        ProcessSlicesAsync(key, slices, replica)
          ├─ skip if IsReplicaOnLocalMemory && !IsShm
          └─ for each slice:
               hot_cache_handler_->SubmitPutTask(key, slice)
                 ├─ TouchHotKey: 已在 cache 就只摸 LRU
                 └─ otherwise:
                      a. GetFreeBlock，满了就淘汰 LRU tail
                      b. memcpy(block->addr, slice.ptr, slice.size)
                      c. task_queue_.push(task)
```

关键点：数据 memcpy 是同步发生在 Get 调用栈里的；“异步”的主要是后续 worker 线程插入 / 更新 LRU 索引。

### 0.5 命中路径：`RedirectToHotCache` 改写 replica descriptor

```cpp
bool Client::RedirectToHotCache(const std::string& key,
                                Replica::Descriptor& replica) {
    HotMemBlock* blk = hot_cache_->GetHotKey(key);
    if (blk == nullptr) return false;

    mem_desc.buffer_descriptor.transport_endpoint_ = GetTransportEndpoint();
    mem_desc.buffer_descriptor.buffer_address_ = (uintptr_t)blk->addr;
    return true;
}
```

技巧是把 replica descriptor 里的 `transport_endpoint_` 改成本 client，把 `buffer_address_` 改成 hot cache block 地址。后续 `TransferRead` 会把它当成本地 memory replica，走 local memcpy，不再发 RDMA。

`HotMemBlock` 有 `ref_count`，`GetHotKey` / `ReleaseHotKey` 配对，防止 block 正在读时被 LRU 淘汰后复用。

### 0.6 对 vLLM in-process 模式的当前判断

在本文实验的 vLLM `MooncakeStoreConnector` in-process 部署形态下，hot cache 有两个现实问题：

- 它默认不是生产常用路径，上游 YAML / daemon setup 都没有把它作为常规配置面使用。
- 在 16 并发 × 70k input × 10 turn 这个 workload 下，hot cache 既可能触发静默 stall，也没有表现出性能收益。

因此当前建议是：除非专门验证 hot cache，否则 vLLM + MooncakeStoreConnector 路径不要打开 `MC_STORE_LOCAL_HOT_CACHE_SIZE`；如果看到 engine 静默冻结，第一件事先关 hot cache 复测。

---

## 1. 现象与最小复现配置

### 复现配置（最小）

| 项 | 值 |
|---|---|
| 推理引擎 | 单 vLLM (`MooncakeStoreConnector` in-process) |
| 并行度 | tp=4，单节点 GB200 4 GPU |
| Bench | 32 prompts × 70k input × 10 multi-turn × c=16 |
| 关键 env | `MC_STORE_LOCAL_HOT_CACHE_SIZE=8589934592`（8 GiB；512 MiB / 1 GiB 也复现） |
| 默认未改 | `admission_threshold=2`（CMS 二阶 admit）+ malloc backing（`use_shm=false`） |

### 症状

- vLLM engine log 周期性打印 `Running: 0, Waiting: N` ≥15 min 静默冻结
- **完全没有 ERROR / 没有 Mooncake 报错 / 没有进程崩溃 / 没有 OOM**
- bench 一直挂着等不到响应，必须手动 cancel

### 验证矩阵（来自 `FINAL_hotcache.md`）

| Tag | 改动 vs 默认 | 结果 |
|---|---|---|
| iter1 | 8 GiB local 模式 | STALL |
| iter2 | 1 GiB | STALL |
| iter3 | 8 GiB + block 1 MiB | STALL |
| iter4 | 8 GiB + `MC_STORE_LOCAL_HOT_ADMISSION_THRESHOLD=1` | **PASS** ✓ |
| iter5 | 8 GiB + `MC_STORE_LOCAL_HOT_CACHE_USE_SHM=1` | **PASS** ✓ |
| iter6 | 512 MiB | STALL |

→ 缩 size、改 block 都无效；只有动 admission 阈值或换内存分配方式才能修。

---

## 2. 两条独立 mitigation

### A. `MC_STORE_LOCAL_HOT_ADMISSION_THRESHOLD=1`

**机制**：把 CMS admission 阈值从默认 2 降到 1。也就是把“第二次访问才允许进 hot cache”改成“第一次访问就允许进 hot cache”。CMS 的含义见 §0.3。

源码路径：
- `client_service.h:533-541`（头文件内联）`Client::ShouldAdmitToHotCache(key, cache_used)`：决定一个 Get 命中后 KV 是否入 hot cache。默认 `admission_threshold_ = 2`（`client_service.h:768`）
- `client_service.cpp:811, 981, 1112` 三个 admission 调用点（Get 单 key + 两个 BatchGet overload）
- `client_service.cpp:3111-3122` `InitLocalHotCache()`：从 `MC_STORE_LOCAL_HOT_ADMISSION_THRESHOLD` 读阈值
- `count_min_sketch.h:14-86`：4 行 × 4096 列 `uint8_t` 表 + 全局 `std::mutex` 保护 increment/count，total_increments ≥ width × depth (16384) 时整表右移衰减

**threshold=2 行为（默认）**：
1. 第一次 Get 一个新 key → CMS 记一次访问，**不入** cache
2. 第二次 Get 同 key → 达到阈值，admit → `SubmitPutTask` 入队
3. 实际 cache 物化要等 worker 线程取出 task + GetFreeBlock + memcpy + LRU insert

**threshold=1 行为**：第一次 Get 就 admit；worker queue 立刻有任务，hot cache 早早有 entries。

### B. `MC_STORE_LOCAL_HOT_CACHE_USE_SHM=1`

**机制**：把 hot cache region 分配方式从 `std::malloc(total_size)` 改为 memfd + `mmap(MAP_SHARED|MAP_POPULATE)`。

源码路径：
- `local_hot_cache.cpp:19-65` LocalHotCache ctor：根据 `use_shm_` 走两条分支
  - false：`bulk_memory_standard_ = std::malloc(total_size)`
  - true：`ShmHelper::getInstance()->allocate(total_size)`
- `shm_helper.cpp` allocate：
  1. `memfd_create(MOONCAKE_SHM_NAME, flags)`
  2. `ftruncate(fd, size)`
  3. `mmap(NULL, size, PROT_READ|PROT_WRITE, MAP_SHARED | MAP_POPULATE, fd, 0)` ← **关键差异**

### 这两条都修同一症状的含义

- A 动的是 **CMS decision logic**（决定哪些 key 进 hot cache）
- B 动的是 **物理内存分配**（malloc anonymous mmap → memfd tmpfs）
- 完全不同 layer 都能独立修同一个 stall → 要么是多根因，要么有一个底层原因穿过两层

实验上看 B（USE_SHM=1）阻断 stall 的效果更直接（不依赖 CMS 决策路径），A（admission=1）相当于改变 SubmitPutTask 的时间分布。下一节给出**底层原因穿过两层**的假设。

---

## 3. page-fault storm 假设（未严格证伪，但与现象 100% 吻合）

这里的 page fault 指的是：程序先申请一大段虚拟内存，但 Linux 不一定马上分配真实物理页。等线程第一次写到某个页时，内核才补上真实物理页和页表。这次“第一次写触发内核补页”的过程就是 page fault。

这里的 storm 指的是：不是一个线程偶尔触发 page fault，而是很多 worker 在同一段时间里一起触发，导致内核忙着处理内存页，业务线程长时间没有进展。

本文的假设是：

1. hot cache 用默认 malloc 路径申请了很大的内存区。
2. 申请完成时，这块内存的大量页面还没有真正被写过。
3. benchmark 开始后，很多 worker 几乎同时往 hot cache 的不同位置写数据。
4. 这些写入同时触发大量 page fault，内核要忙着给这些页面补真实物理页和页表。
5. 这个过程会争用进程级内存管理锁；并发越高，越容易把整个 vLLM 进程拖到长时间无进展。

所以我们看到的是：没有 Mooncake ERROR，没有 OOM，没有崩溃，但 vLLM engine 长时间停在 `Running: 0, Waiting: N`。这不像普通业务逻辑报错，更像进程里的很多线程都在等内核处理内存页。

### 为什么 `USE_SHM=1` 能绕开

`MC_STORE_LOCAL_HOT_CACHE_USE_SHM=1` 走 memfd + `MAP_POPULATE`。重点是 `MAP_POPULATE`：它让 mmap 创建时就尽量把页面准备好。

这样 benchmark 真正开始写 hot cache 时，就不会集中触发那么多“第一次写才补页”的工作。因此 stall 消失。

### 为什么 `admission=1` 也可能缓解

默认 threshold=2 时，很多 key 第一次访问只计数、不入 cache；到了第二轮复用时，大量 key 同时达到阈值，于是大量 hot cache 写入集中发生。

threshold=1 时，key 第一次访问就开始入 cache。写入更早发生，也更分散，不容易在某一刻集中触发一大批 page fault。

### 标记

**这是 hypothesis 不是 fact**。我们还没有用 perf / ftrace 把内核等待栈抓出来。但：
- 与症状 100% 吻合（静默 + 无 ERROR + 整进程冻结）
- A/B 两条 mitigation 都能用这个 hypothesis 一致解释
- 没有反证（未发现其它能同时解释 A 和 B 的 hypothesis）

如果要进一步确证：在 stall 时抓一次 perf / ftrace，看线程是否大量卡在内核内存页分配和页表更新相关路径上。

---

## 4. 绕过 stall 后性能仍崩

即使 admission=1 / USE_SHM=1 修好 stall，hot cache **整体性能反而比不开 hot cache 更差**：

| 指标 | 不开 hot cache（baseline，session 2） | iter4（admission=1） | iter5（USE_SHM=1） |
|---|---:|---:|---:|
| throughput | 58,464 tok/s | 30,568 tok/s（**-48%**） | 19,100 tok/s（**-67%**） |
| TTFT mean | 13,183 ms | 19,871 ms（+51%） | 21,217 ms（+61%） |
| TPOT mean | 19.25 ms | 45.43 ms（**+136%**） | 44.83 ms（+133%） |
| e2el mean | 18,938 ms | 36,425 ms（+92%） | 58,523 ms（+209%） |
| External hit rate | 79.3% | 13.9% | 11.1% |

External hit rate 还从 79.3% 跌到 11-14%，说明**绝大部分 Get 仍走到 mooncake store**，hot cache 在路径上变成"噪声层"——每次 Get 都要先查它再 fall through。

候选解释（**未独立验证**）：
- 16 MiB block size 默认值对 KV-cache 工作负载太大（每个 KV 块 ~36 KB）
- LRU 访问全局 mutex contention（多 TP rank 共享）
- CMS increment 全局 mutex contention（每个 Get 都要进）
- `drainDeferredTouches` 持 `lru_mutex_` unique_lock 阻塞其它路径

### 实操结论

> **本 workload (16 conc × 70k input × 10 turn) 下 hot cache 没有性能理由开**。

baseline 不开 hot cache 已经能拿 79.3% external hit + 58k tok/s。要再榨性能应该从**降低 mooncake store Get 延迟**入手（rail-aligned QP、更大 batch RDMA），不是加一层 client 端 cache。

---

## 5. "0 用户 codepath"

为什么这种 bug 在 Mooncake 上游 main 长期没有人踩？因为 hot cache 这条路在生产形态里**0 用户**：

| 检查项 | 结果 |
|---|---|
| `Mooncake/mooncake-standalone.yaml` 是否设 `MC_STORE_LOCAL_HOT_CACHE_SIZE` | ❌ 不设（默认 0） |
| `Mooncake/mooncake_high_availablity.yaml` 是否设 | ❌ 不设 |
| Mooncake repo 任何 .md / .yaml / .json / .py 提到 hot cache | grep `HOT_CACHE` / `_USE_SHM` / `_ADMISSION_THRESHOLD` 全 0 命中 |
| 官方 store daemon `mooncake_store_service.py` 启动时读 hot cache env | ❌ `setup()` 6 个基础参数里没有 |

代码层面的"默认死"机制：
- `client_service.cpp:635`：每个 Client 构造时无条件调 `InitLocalHotCache()`
- `client_service.cpp:3081-3098`：函数内首先读 `MC_STORE_LOCAL_HOT_CACHE_SIZE`，**默认 0**
- size==0 时 early return → `hot_cache_` / `admission_sketch_` / `handler` 全 nullptr → 整个 codepath 默认关闭

也就是说：

> Bug 在生产形态里 0 用户，长期没人触发。我们这次踩到是因为用了"in-process linked client + size>0"的组合 —— vLLM `MooncakeStoreConnector` 是这条路的主流用法，但官方 yaml 从未启用，因此从未在这种 in-process 形态下被验证过。

---

## 6. `use_shm=true` 的真实设计动机

这条对**给 mooncake 上游反馈**很重要。

### 为什么 USE_SHM 看起来像 "hot cache 工作模式开关"

USE_SHM 关时 malloc，开时 memfd —— 表面看就是分配方式选择。**但这不是它的设计意图**。

### 真实设计意图（从 ShmHelper 实现倒推）

`shm_helper.cpp` 的 `allocate(size)` 链路会创建一块 memfd-backed shared memory，然后用 `mmap(MAP_SHARED | MAP_POPULATE, ...)` 映射进进程。

这里有两个性质要分开看：

- `MAP_SHARED`：让这块内存可以被多个进程映射到各自地址空间里，用来共享同一份 hot cache 数据。
- `MAP_POPULATE`：映射时提前准备页面，减少后续第一次写时的 page fault。

这是为 daemon 模式设计的：
- `mooncake_store_service` daemon 持一个全节点共享的 hot cache region（memfd）
- 本节点的多个 vLLM 进程是 **dummy_client**（也 link mooncake-store，但走 UDS socket 接 daemon）
- daemon 把 memfd 交给 dummy client → 后者 mmap 到自己进程 → 多个进程看到同一块 hot cache 数据
- 这才是 `LocalHotCache(use_shm=true)` 的**真正用户场景**

### 我们触碰到的是什么

vLLM 用的是 **in-process linked client**：vLLM 进程内部直接 link mooncake-store C++ 库，没有外部 daemon，也没有 dummy client 需要共享同一块 hot cache。也就是说，我们这条路径不需要 `MAP_SHARED` 提供的跨进程共享。

USE_SHM 在我们这种部署下能修 stall，主要是因为它同时带了 `MAP_POPULATE`，提前把页面准备好了。它修的是 page fault 集中爆发，不是因为我们真的需要 shared memory。

### 反馈材料里要写清的事

跟官方反馈时要说明：

> 我们的部署形态是 vLLM 进程内 link mooncake-store（`MooncakeStoreConnector`），不是 daemon + dummy_client 模式。`MC_STORE_LOCAL_HOT_CACHE_USE_SHM=1` 在我们这条路里能修 stall，但主要原因是 `MAP_POPULATE` 提前准备页面，不是 shared memory 本身。如果只是为了修 in-process 模式的 stall，更合适的 fix 是给默认 malloc 路径也做 prefault，或者新增一个语义明确的 prefault 开关，而不是让用户打开 use_shm。

---

## 7. 可观测性补丁

下游分支 `aoshen524/Mooncake:feat/observability-logs` 已加（PR #1 to `ivanium/Mooncake:yifan/dev`）：

- `transfer_metadata.cpp`：segment descriptor 404 时多打 `metadata_key='<key>'`
- `http_metadata_server.cpp`：每次 GET/PUT/DELETE 打一行 access log（`metadata GET MISS / PUT NEW / PUT UPDATE / DELETE OK / DELETE MISS`）
- `transfer_task.cpp` + `client_service.cpp`：openSegment 失败 / replica 提交失败时打 `endpoint='...' replica_idx=...`
- `client_service.cpp WaitForTransfers`：从"任一 replica 失败 = 整 op 失败"改为 quorum 写（至少一个成功 = op 成功）

这些不是修 hot cache 的，但 hot cache stall 调试中起辅助作用：master 端的 PUT/GET access log 让你能确认"client 在 stall 期间是否还在 ping master"——可以排除 client 整个进程死了的可能（如果 metric heartbeat 还在 → 进程活着 → bug 是局部 deadlock 而非崩溃）。

安装 + 验证流程见 `mooncake_local_install_verify_cn.md`。

---

## 8. 教训汇总（给下一个集成 vLLM + Mooncake 的人）

1. **"engine 静默冻结"先排查 hot cache**：如果 yaml 里有 `MC_STORE_LOCAL_HOT_CACHE_SIZE > 0`，第一件事关掉它复测。本 workload 下 hot cache 既容易触发 stall，又没性能收益。
2. **Hot cache 是默认关的**：上游 yaml + 整 repo 都没人开。开它意味着你在 0 用户 codepath 上，不要假设有 production 验证。
3. **stall ≠ 崩溃**：mooncake bug 表现可以是无声的 mm_lock contention，不一定有 ERROR。`Running:0 Waiting:N` ≥ 几分钟 + 无报错 = 高概率 mm 层卡住。
4. **修 stall 的 env 不是修 perf 的 env**：admission=1 / USE_SHM=1 修了进程冻结但 throughput 仍降 50%+。两个问题分别处理。
5. **报上游时区分"症状层"和"根因层"**：USE_SHM=1 修 stall 是症状层（副作用 prefault），根因层是 malloc 路径不 prefault → 反馈材料要把这点点出来，避免官方误以为加 USE_SHM 文档/默认就够。
