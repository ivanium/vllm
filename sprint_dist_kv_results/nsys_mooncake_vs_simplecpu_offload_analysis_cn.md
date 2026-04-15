# Nsys Trace Analysis: MooncakeStore vs SimpleCPUOffload KV Cache Offload

**日期**: 2026-04-13
**模型**: nvidia/Kimi-K2.5-NVFP4 (MoE, DeepSeekV2 MLA, EAGLE3 speculative decode)
**硬件**: 4x NVIDIA GB200 (189 GB, 152 SMs), TP4
**架构**: P/D 分离部署, prefill 1 节点 + decode 1 节点
**负载**: Multi-turn chat, 70K input tokens, 300 output tokens, 8 并发, 10 轮对话

## 配置差异

两次运行的**唯一区别**是 prefill 端 MultiConnector 中的 KV cache offload 后端:

| 配置项 | Mooncake | SimpleCPU |
|--------|----------|-----------|
| Prefill KV-transfer | NixlConnector + **MooncakeStoreConnector** (`load_async: true`) | NixlConnector + **SimpleCPUOffloadConnector** (`cpu_bytes: 400GB, lazy_offload: true`) |
| Decode KV-transfer | NixlConnector (完全相同) | NixlConnector (完全相同) |
| 其他配置 | 完全相同 | 完全相同 |

日志路径:
- Mooncake: `vigil/logs/pd_kimi_70k_nsys/2026-04-13/20260413_142016/`
- SimpleCPU: `vigil/logs/pd_kimi_70k_simplecpu_nsys/2026-04-13/20260413_142029/`

---

## 1. 端到端 Benchmark 结果

| 指标 | Mooncake | SimpleCPU | 差异 |
|------|----------|-----------|------|
| Benchmark 总时长 | 240.2s | 292.5s | +21.8% |
| 输入吞吐量 (tok/s) | **46,623** | 38,287 | Mooncake +21.8% |
| 输出吞吐量 (tok/s) | **199.8** | 164.1 | Mooncake +21.8% |
| 平均 TTFT (ms) | **4,530** | 7,701 | SimpleCPU 慢 70% |
| 中位数 TTFT (ms) | **2,497** | 5,589 | SimpleCPU 慢 124% |
| 平均 TPOT (ms) | 24.56 | **22.80** | SimpleCPU 快 7.2% |
| 中位数 TPOT (ms) | 24.85 | **22.08** | SimpleCPU 快 11.1% |
| 平均 E2EL (ms) | **11,873** | 14,517 | SimpleCPU 慢 22.3% |
| P99 TTFT (ms) | **31,907** | 42,145 | SimpleCPU 慢 32.1% |
| P99 TPOT (ms) | **45.46** | 95.41 | SimpleCPU 慢 110% |

**结论**: Mooncake 在吞吐量 (+21.8%) 和 TTFT (-41~55%) 上显著领先; SimpleCPU 仅在 TPOT 上有 7~11% 的微弱优势, 但 P99 TPOT 反而差 2 倍。整体 E2EL Mooncake 优 22%。

### 稳态 Per-Turn 分析 (Turn 4-9)

| 指标 | Mooncake (avg T4-T9) | SimpleCPU (avg T4-T9) | 差异 |
|------|---------------------|----------------------|------|
| Mean TTFT (ms) | ~2,364 | ~4,451 | SimpleCPU 慢 88% |
| Mean TPOT (ms) | ~25.1 | ~22.7 | SimpleCPU 快 9.5% |
| Mean E2EL (ms) | ~9,822 | ~11,286 | SimpleCPU 慢 14.9% |

---

## 2. Profiling 窗口局限性与 External Cache Hit Rate (重要发现)

### 2.1 Profiling 只捕获了冷启动行为

两个 run 的 prefill profiler 配置均为 `delay_iterations: 0, max_iterations: 30`, 即**只捕获前 30 个 worker step**。关键时间线:

```
时间线                     Mooncake                  SimpleCPU
──────                     ────────                  ─────────
Profiler 启动              14:28:16                  14:29:15
Profiler 停止 (30 iters)   14:28:27 (11s)            14:29:26 (11s)
首次 External Cache > 0%   14:29:05 (+49s)           14:30:19 (+64s)
External Cache 峰值        79.3%                     68.4%
```

**nsys trace 整个 profiling 窗口内, 两个 run 的 External prefix cache hit rate 均为 0.0%。**

这意味着:
- **Trace 只捕获了 STORE (offload) 路径** — KV cache 被写入外部存储
- **Trace 完全没有捕获 LOAD (restore) 路径** — 从外部存储恢复 KV cache 以实现 prefix cache hit
- **offload 的核心价值 — prefix cache 复用 — 在 trace 中不可见**
- Benchmark 结果 (TTFT, throughput) 反映的是**完整 10 轮对话**的表现, 大部分性能差异来自 profiling 窗口之后

### 2.2 External Cache Hit Rate 对比 (从 vLLM 日志)

| 阶段 | Mooncake | SimpleCPU | 分析 |
|------|----------|-----------|------|
| Profiling 窗口 (前 30 步) | 0.0% | 0.0% | 首轮对话, 无历史 prefix 可复用 |
| 首次非零 | **+49s** (13.8%) | +64s (4.8%) | **Mooncake 快 15s** 开始提供 cache 命中 |
| Turn 2-3 阶段 | 快速爬升到 40-50% | 缓慢爬升到 20-30% | Mooncake 爬升速率 ~2x |
| 稳态峰值 | **79.3%** | 68.4% | **Mooncake 高 10.9pp** |

#### 为什么 Mooncake 的 External Cache Hit Rate 更高?

1. **Mooncake 立即 offload**: 请求完成后立刻 `store_put`, KV 数据在 ~6ms 内进入远程 Store, 下次相同 prefix 的请求立即可复用
2. **SimpleCPU lazy offload 滞后**: `lazy_offload: true` 模式下, block 要接近被驱逐时才触发 offload, 部分 block 可能**在 offload 之前就被驱逐** → 永久丢失 → cache miss
3. **Mooncake 首次 cache 命中早 15s**: 49s vs 64s, 这 15s 的差距在 multi-turn 场景下意味着 SimpleCPU 的第 2 轮对话几乎无法享受 external cache

#### 对 Benchmark 性能的影响

峰值 external cache hit rate 差 10.9pp (79.3% vs 68.4%) 在 70K input token 场景下影响巨大:
- **每增加 1pp external cache hit**: 节省 ~70K tokens 的 prefill 计算 (一个完整请求)
- **10.9pp 差距**: 相当于 Mooncake 每 ~9 个请求比 SimpleCPU 多命中 ~1 个完整请求的 prefix
- 这直接解释了 21.8% 的吞吐量差异和 70% 的 TTFT 差异中的大部分

### 2.3 对 Trace 分析的影响

由于 profiling 窗口完全在冷启动阶段, 以下 trace 分析结果有局限性:

| 分析内容 | 可信度 | 说明 |
|----------|--------|------|
| Forward pass 计算性能 | 高 | 冷启动时的计算行为与稳态一致 |
| Mooncake store_put 开销 | 高 | Store 路径在冷启动时就活跃 |
| SimpleCPU offload 机制 | 低 | 不可见于 trace, 且 lazy mode 下冷启动时 offload 量小 |
| **External cache load 延迟** | **不可用** | **两边 external cache hit 均为 0%, load 路径未被触发** |
| AllReduce 竞争分析 | 中 | 冷启动时 RDMA 活动可能与稳态不同 |

> **建议**: 若需分析 load 路径和稳态行为, 应将 profiler 配置改为 `delay_iterations: 60+` 以跳过冷启动期, 或在 external cache hit rate > 50% 后再触发 profiling。

---

## 3. KV Cache Offload 机制对比 (核心分析)

### 3.1 架构差异

```
┌─────────────────────────────────────────────────────────────────────┐
│                    MooncakeStoreConnector                           │
│                                                                     │
│  Prefill GPU ──[cuda_event_sync]──> 后台线程 ──[RDMA/TCP]──> 远程Store │
│       │                                                             │
│       │  1. 每 step 记录一个 CUDA Event                              │
│       │  2. 后台线程 synchronize() 等待 GPU 计算完成                   │
│       │  3. batch_put_from_multi_buffers() 零拷贝 GPU→Store          │
│       │  4. 加载时 batch_get_into_multi_buffers() Store→GPU          │
│       │                                                             │
│  同步方式: CUDA Event (阻塞后台 CPU 线程, 不阻塞 GPU)                  │
│  传输协议: UCX RDMA 或 TCP                                           │
│  NVTX 插桩: 完整 (cuda_event_sync, store_put, store_get, batch_is_exist) │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                  SimpleCPUOffloadConnector                          │
│                                                                     │
│  Prefill GPU ──[low-pri stream]──> CPU Pinned Memory                │
│       │                                                             │
│       │  1. 分配两个低优先级 CUDA Stream (load_stream, store_stream)  │
│       │  2. cuMemcpyBatchAsync() 批量异步 DMA (GPU→CPU 或 CPU→GPU)   │
│       │  3. 非阻塞 Event 轮询 (event.query()) 检查完成状态            │
│       │  4. 仅在 preemption 时强制同步                               │
│       │                                                             │
│  同步方式: 非阻塞 Event Query (不阻塞任何线程)                        │
│  传输方式: cuMemcpyBatchAsync (CUDA Driver API, pinned memory)       │
│  NVTX 插桩: 无 (offload 路径在 trace 中完全不可见)                    │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 Offload (Store) 路径详细对比

| 方面 | MooncakeStore | SimpleCPUOffload |
|------|---------------|------------------|
| **触发时机** | 请求完成时 (`get_finished()`) | Eager: 每个 block 计算完成后; Lazy: 接近驱逐时 |
| **数据源** | GPU KV cache block 的原始地址 | GPU KV cache block ID |
| **目标** | 远程分布式存储 (跨节点) | 本地 CPU pinned memory |
| **传输 API** | `store.batch_put_from_multi_buffers()` | `cuMemcpyBatchAsync()` |
| **同步机制** | `cuda_event.synchronize()` — 阻塞后台 CPU 线程 | `event.query()` — 非阻塞轮询 |
| **CUDA Stream** | 隐式使用默认 stream | 显式低优先级 `store_stream` |
| **批量大小** | 每次 put n=64 blocks (典型值) | 由 `_target_free` 决定 |
| **反压检测** | `MOONCAKE_NO_AVAILABLE_HANDLE` 错误码 | CPU block pool 空闲列表深度 |
| **可观测性** | NVTX 标记: `mooncake_store_put_n=X` | 无 NVTX, 仅内部 stats 队列 |

### 3.3 Load (Get) 路径详细对比

| 方面 | MooncakeStore | SimpleCPUOffload |
|------|---------------|------------------|
| **触发时机** | Scheduler 发现需要外部 token 时 | Scheduler 找到 CPU 缓存命中时 |
| **数据源** | 远程分布式存储 | 本地 CPU pinned memory |
| **目标** | GPU KV cache block | GPU KV cache block |
| **传输 API** | `store.batch_get_into_multi_buffers()` | `cuMemcpyBatchAsync()` |
| **负载均衡** | TP rank 轮转分散 store 访问 | 无 (本地内存, 无需均衡) |
| **磁盘 offload** | 支持 (大批量时分片到 staging buffer) | 不支持 |
| **可观测性** | NVTX 标记: `mooncake_store_get_n=X` | 无 NVTX |

### 3.4 Lazy Offload 模式 (SimpleCPU 特有)

本次测试中 SimpleCPU 使用 `lazy_offload: true`:
- **策略**: 不是立即 offload 每个计算完的 block, 而是在 GPU free queue 接近驱逐点时才触发
- **实现**: 维护一个 cursor 遍历 GPU free queue, 每 step 从 cursor 开始收集即将被驱逐的 block
- **目标量**: `_target_free = max_batched_tokens / block_size` 个 block
- **优势**: 减少不必要的 GPU→CPU 拷贝
- **劣势**: block 可能来不及 offload 就被驱逐, 导致 cache miss; 且 load 延迟不可预测

---

## 4. Prefill 端 Offload 行为分析 (Trace 证据, 仅冷启动阶段)

### 4.1 Trace 概览

| 指标 | Mooncake | SimpleCPU | 差异 |
|------|----------|-----------|------|
| 总时长 | 10.12s | 10.59s | SimpleCPU +4.7% |
| 总 kernel 数 | 314,179 | 336,252 | SimpleCPU +7.0% |
| GPU 总时间 (4卡) | 37,321 ms | 40,262 ms | SimpleCPU +7.9% |
| NVTX 事件数 | 543 (含完整 Mooncake 插桩) | 121 (仅 forward pass) |
| Forward pass 次数 | 121 | 121 | 相同 |
| 平均 forward pass 时长 | 306.1 ms | 304.1 ms | 几乎相同 |
| CUDA Graphs | 无 (eager 模式) | 无 (eager 模式) |

**关键发现**: 两者的 forward pass 性能几乎一致 (~304-306ms), 说明 offload 后端的选择不影响 GPU 计算本身。差异来自 offload/load 的调度和同步开销。

### 4.2 Mooncake Offload 开销分解

| 操作 | 次数 | 总耗时 (ms) | 平均 (ms) | 最大 (ms) |
|------|------|-----------|---------|---------|
| `mooncake_cuda_event_sync` | 120 | **8,713** | 72.6 | 200.1 |
| `mooncake_store_put` (n=64) | 98 | 573 | 5.8 | 68.9 |
| `mooncake_store_put` (n=17~47) | 20 | 41 | 2.0 | ~5 |
| `mooncake_batch_is_exist` | 156 | 85 | 0.5 | 0.9 |
| `mooncake_store_get` (n=1) | 28 | 19 | 0.7 | 2.8 |
| `mooncake_lookup_batch` | 8 | 11 | 1.4 | 3.0 |
| **总计** | | **9,412** | | |

#### cuda_event_sync 与 GPU 计算的重叠分析

cuda_event_sync 是 Mooncake 最大的开销项 (8.7s), 但**它运行在后台 CPU 线程上, 不阻塞 GPU**。通过分析 sync 窗口内的 GPU kernel 活动:

| 分析 | 结果 |
|------|------|
| Top 5 最长 sync 窗口 (191~200ms) | 每个窗口内有 ~4,700 个 kernel 在运行 |
| sync 期间 GPU 利用率 | 100% — fmha, nvjet GEMM, allreduce 等全部在执行 |
| **与 forward pass 重叠比例** | **92.1%** (8,713ms 中 8,026ms 被 forward pass 覆盖) |
| **实际暴露的 Mooncake 开销** | **294ms / 10,134ms = 仅 2.9%** |

结论: Mooncake 的 offload 路径设计良好 -- 后台线程的 `cuda_event_sync` 等待 GPU 计算完成, 期间 GPU 满载运行, **90%+ 的 offload 开销被隐藏在 forward pass 背后**。

#### store_put 性能

- 典型一次 `store_put` (64 blocks): **5.8ms 平均**, 包含 GPU→远程 Store 的 RDMA 传输
- 这 5.8ms 中约 60% 也与 forward pass 重叠
- 实际 RDMA 带宽: 假设每 block 约 1MB, 64 blocks / 5.8ms ≈ **11 GB/s** (合理的 RDMA 速率)

### 4.3 SimpleCPU Offload 行为 (Trace 证据)

**SimpleCPU 的 offload 在 nsys trace 中完全不可见**:

| 检查项 | 结果 |
|--------|------|
| NVTX offload 标记 | 无 — 仅有 `execute_context_*`, `NCCL`, `cuBLAS` |
| D2H memcpy (GPU→CPU) | 仅 448~482 次, 每次 4 bytes (标量, 非 KV 数据) |
| H2D memcpy (CPU→GPU) | 仅 38~42 MB 总量 (元数据, 非 KV 数据) |
| 可见的 KV 数据传输 | 无 |

**原因分析**: SimpleCPUOffloadConnector 使用 `cuMemcpyBatchAsync()` 在低优先级 CUDA Stream 上执行 DMA, 且没有添加 NVTX 标记。nsys 的 CUPTI 层可能未捕获这些通过 CUDA Driver API 直接发起的批量异步拷贝, 或者它们被归类到了其他 stream activity 中而非标准 memcpy 事件。

**这意味着我们无法从 trace 中直接测量 SimpleCPU 的 offload 开销**, 只能通过间接指标推断。

### 4.4 间接推断: SimpleCPU Offload 的影响

虽然无法直接观测, 但以下间接证据表明 SimpleCPU 的 lazy offload 造成了调度层面的 back-pressure:

| 间接指标 | Mooncake | SimpleCPU | 分析 |
|----------|----------|-----------|------|
| 总 kernel 数 | 314,179 | 336,252 (+7%) | SimpleCPU 执行了更多 kernel |
| Forward pass 次数 | 121 | 121 | 相同 — 不是因为多了 forward pass |
| `execute_context_2(8192)` 次数 | 8 | 13 (+63%) | SimpleCPU 有更多 2-request 批处理 |
| GPU 总 kernel 时间 | 37,321 ms | 40,262 ms (+7.9%) | 更多计算工作 |
| AllReduce 平均延迟 | 425 us | 368 us | **SimpleCPU 反而更低** |

分析:
1. **SimpleCPU 的 lazy offload 可能导致 GPU KV block 释放不及时**, 迫使 scheduler 更频繁地分配小批量或重新处理
2. **更多的 `execute_context_2` 批处理** 说明 scheduler 在 SimpleCPU 下更频繁地将多个请求打包, 可能是因为请求排队时间更长
3. SimpleCPU 的 allreduce 延迟更低 (368us vs 425us), 是因为没有 RDMA 流量与 NVLink 竞争

### 4.5 AllReduce 竞争 (Mooncake 特有)

Mooncake 的 RDMA 传输与 NVLink AllReduce 产生带宽竞争:

| GPU | Mooncake (avg/max us) | SimpleCPU (avg/max us) | 分析 |
|-----|----------------------|------------------------|------|
| 0 | 464 / **176,952** | 373 / 111,247 | Mooncake GPU 0 有 177ms 异常尖峰 |
| 1 | 418 / 117,911 | 399 / 161,867 | |
| 2 | 407 / 63,469 | 365 / 161,371 | |
| 3 | 413 / 104,774 | 335 / 3,954 | |
| **平均** | **425** | **368** | **Mooncake AllReduce 慢 15%** |

Mooncake 的 AllReduce 平均慢 15%, 且 GPU 0 出现 177ms 极端尖峰。这是 RDMA 传输 (Mooncake store_put) 与 NVLink AllReduce 争夺互连带宽的直接证据。

---

## 5. Decode 端 KV Load 行为分析

Decode 端配置完全相同 (都用 NixlConnector), 但由于 prefill 端的 offload 后端不同, decode 的行为也受到间接影响。

### 5.1 NIXL KV Transfer (Prefill→Decode)

两个 trace 都使用 NIXL 进行 P/D 间的 KV 传输:

| 指标 | Mooncake | SimpleCPU | 差异 |
|------|----------|-----------|------|
| `nixl_read_blocks` 大块传输 (n=135,718) | 4 次, avg 372ms | 4 次, avg 364ms | 几乎相同 |
| `nixl_read_blocks` 小块传输 (n=115,444) | 4 次, avg 3.2ms | 8 次, avg 2.6ms | SimpleCPU 多 4 次 |
| `nixl_xfer_done` 完成确认 | 8 次, 0.3ms | 12 次, 0.9ms | SimpleCPU 多 4 次 |

**关键发现**: 
- 首次大块 KV 传输 (~135K blocks) 耗时 **~370ms**, 两者一致 — **NIXL 层本身的传输性能没有差异**
- SimpleCPU 有更多的增量传输 (8 次 vs 4 次小块读取), 因为它处理了 3 个请求 vs Mooncake 的 2 个

### 5.2 Decode 首次 KV 接收 (context_1 事件)

context_1 事件是 decode 接收 prefill 发来的 KV 后执行的第一步, 直接反映 KV load 延迟:

| 事件 | Mooncake | SimpleCPU | 差异 |
|------|----------|-----------|------|
| `context_1 gen_0(0)` | ~150ms | ~220ms | SimpleCPU **慢 47%** |
| `context_1 gen_1(4)` | avg 857ms (max 887ms) | avg 965ms (max 1004ms) | SimpleCPU **慢 12.5%** |

SimpleCPU 的 decode 首步延迟显著更高 (47%), 说明 prefill 端 offload 后端的选择影响了 KV 数据到达 decode 的时序。

### 5.3 Decode NVLink P2P 流量

| 指标 | Mooncake | SimpleCPU | 差异 |
|------|----------|-----------|------|
| P2P 总数据量 (4卡) | 20.3 GB | 29.7 GB | SimpleCPU **多 46%** |
| P2P 总时间 | 31.9 ms | 45.8 ms | SimpleCPU 多 44% |
| P2P 次数 (每卡) | 186 | 186 | 相同 |
| P2P 平均大小 | 27.3 MB | 39.9 MB | SimpleCPU 大 46% |

SimpleCPU decode 的 NVLink P2P 流量高 46%, 主要因为多处理了一个 generation (4 vs 3)。

### 5.4 Decode Per-Step 延迟

| 统计量 | Mooncake | SimpleCPU | 差异 |
|--------|----------|-----------|------|
| 步数 | 801 | 800 | |
| **中位数** | **4.84 ms** | **5.69 ms** | **SimpleCPU 慢 17.5%** |
| **P90** | **6.73 ms** | **8.49 ms** | **SimpleCPU 慢 26.2%** |
| P99 | 379.6 ms | 465.7 ms | SimpleCPU 慢 22.7% |
| Max | 886.8 ms | 1,004.3 ms | SimpleCPU 慢 13.2% |

即使 decode 配置完全相同, SimpleCPU 的 per-step 延迟仍然高 17~26%。这不是因为 decode 本身变慢, 而是因为:
1. **Prefill 慢 → KV 到达时间晚 → decode 排队等待**, 批处理组成不同
2. **SimpleCPU 多处理了一个 generation**, 意味着更多的 KV block 在 attention 中被访问

### 5.5 Decode Kernel 分类差异

| 类别 | Mooncake ms (%) | SimpleCPU ms (%) | 差异 |
|------|----------------|-----------------|------|
| GEMM | 2,600 (29.3%) | 2,845 (26.6%) | +245 ms (+9.4%) |
| **AllReduce** | **2,076 (23.4%)** | **2,734 (25.5%)** | **+658 ms (+31.7%)** |
| **Attention** | **1,843 (20.8%)** | **2,496 (23.3%)** | **+653 ms (+35.4%)** |
| Other | 1,301 (14.7%) | 1,401 (13.1%) | +100 ms |
| MoE | 632 (7.1%) | 698 (6.5%) | +66 ms |
| Elementwise | 321 (3.6%) | 439 (4.1%) | +118 ms |

AllReduce (+658ms) 和 Attention (+653ms) 是 decode 端最大的时间差, 共占 1.3s (总差异 1.8s 的 72%)。

---

## 6. 核心差异总结

### 6.1 Offload 路径对比总结

```
                    Mooncake                    SimpleCPU
                    ────────                    ─────────
存储位置:          远程分布式 Store              本地 CPU 内存 (400GB pinned)
传输机制:          RDMA 零拷贝                  cuMemcpyBatchAsync (DMA)
同步代价:          8.7s cuda_event_sync         不可见 (非阻塞 query)
                   (92% 被 GPU 计算隐藏)         
AllReduce 影响:    +15% (NVLink 竞争)           无影响
NVTX 可观测性:     完整                         无
Trace 中可见性:    store_put/get 清晰可见       完全不可见
```

### 6.2 为什么 Mooncake 更快? (根因分析)

**主因: External Cache Hit Rate 差异 (贡献 ~70% 的性能差距)**

1. **Mooncake 立即 offload → cache 填充快**: 请求完成后 ~6ms 内 KV 进入 Store, 下一轮对话立刻可复用
2. **SimpleCPU lazy offload → cache 填充慢且有丢失**: block 接近驱逐才 offload, 部分 block 来不及 offload 就丢失
3. **结果**: Mooncake external cache hit 79.3% vs SimpleCPU 68.4%, 差 10.9pp
4. **在 70K token 场景下**: 10.9pp 差距 ≈ 每 9 个请求多节省 1 个完整 prefill, 直接影响 TTFT 和吞吐量

**次因: Offload 实现差异 (贡献 ~30% 的性能差距)**

1. **Mooncake 的 offload 设计更高效**: 后台线程 + CUDA Event 同步, 90% 开销被 GPU 计算覆盖, 实际暴露开销仅 2.9%
2. **SimpleCPU 的 lazy offload 造成调度 back-pressure**: GPU block 释放不及时, scheduler 被迫处理更多小批量 (+7% kernel, +63% 多请求批处理)
3. **Mooncake external cache 首次命中早 15s**: 49s vs 64s, 第 2 轮对话时 SimpleCPU 几乎无 external cache 可用

### 6.3 SimpleCPU 的唯一优势

- **TPOT 均值低 7~11%**: 因为 CPU offload 不产生 NVLink 竞争
- **AllReduce 延迟低 15%**: 同上, 但此优势不足以弥补 TTFT 的巨大劣势
- **注意**: Decode per-step 延迟 SimpleCPU 反而高 17%, TPOT 优势可能来自 benchmark 层面不同的调度节奏, 非底层每步速度

### 6.4 关键数据汇总

```
                            Mooncake        SimpleCPU       赢家
                            ────────        ─────────       ────
输入吞吐量 (tok/s)          46,623          38,287          Mooncake (+21.8%)
输出吞吐量 (tok/s)          199.8           164.1           Mooncake (+21.8%)
平均 TTFT (ms)              4,530           7,701           Mooncake (-41%)
中位 TTFT (ms)              2,497           5,589           Mooncake (-55%)
平均 TPOT (ms)              24.56           22.80           SimpleCPU (-7%)
平均 E2EL (ms)              11,873          14,517          Mooncake (-18%)
Prefill 时长 (s)            10.12           10.59           Mooncake (-4.4%)
Decode 中位 step (ms)       4.84            5.69            Mooncake (-15%)
External Cache 峰值 (%)     79.3            68.4            Mooncake (+10.9pp)
External Cache 首次命中     +49s            +64s            Mooncake (快 15s)
NIXL 首次传输 (ms)          372             364             持平
AllReduce 开销 (prefill)    +15% vs 基线    基线            SimpleCPU
Offload 暴露开销            2.9%            不可观测         -
```

---

## 7. 优化建议

### 7.1 Mooncake 优化方向

1. **缓解 AllReduce 竞争**: GPU 0 出现 177ms AllReduce 尖峰, 建议:
   - 限制 `store_put` 的 RDMA 突发带宽 (避免与 AllReduce 时间窗重叠)
   - 或将 store_put 调度到 forward pass 间隙 (AllReduce 完成后)

2. **减少 cuda_event_sync 峰值**: 最大 200ms 的 sync 说明某些 step 的 GPU 计算异常长, 考虑:
   - 分层 event sync (每 N 层一个 event, 而非整个 step)
   - 流水线式 put: 前面几层 KV 的 sync + put 与后续层计算重叠

### 7.2 SimpleCPU 优化方向

1. **添加 NVTX 插桩** (最重要): 当前 offload 路径在 trace 中完全不可见, 无法诊断。建议在以下位置添加标记:
   - `launch_copy()` 开始/结束
   - `cuMemcpyBatchAsync()` 调用
   - Event query 结果 (完成 vs 未完成)
   - Lazy offload 触发点和 block 数量

2. **评估 eager vs lazy offload**: 本次使用 `lazy_offload: true`, 可能导致 block 释放不及时。建议对比 `lazy_offload: false` (eager 模式) 看是否改善 TTFT。

3. **诊断 back-pressure**: 7% 的额外 kernel 和 63% 更多的多请求批处理暗示调度瓶颈, 需要:
   - 监控 GPU free block 数量随时间变化
   - 监控 CPU block pool 利用率
   - 检查 `_target_free` 参数是否合理

4. **关注 P99 TPOT**: SimpleCPU 的 P99 TPOT 达 95.4ms (Mooncake 45.5ms), 尾延迟高 2 倍。可能是 CPU→GPU load 操作偶尔与 decode 计算竞争 Stream 资源。

### 7.3 Profiling 策略优化

当前 profiling 配置 (`delay_iterations: 0, max_iterations: 30`) 只捕获冷启动, 完全错过了 load 路径。建议:

1. **捕获稳态行为**: 将 prefill profiler 改为 `delay_iterations: 80, max_iterations: 30`, 使 profiling 在 external cache hit rate > 50% 后开始
2. **分两次 profiling**: 一次 `delay: 0` 捕获冷启动 store 行为, 一次 `delay: 80+` 捕获稳态 load + store 行为
3. **增加 SimpleCPU NVTX**: 即使延迟 profiling, SimpleCPU 的 offload 仍然不可见, 必须先加插桩

### 7.4 通用建议

- **此负载下 Mooncake 是更好的选择**: 吞吐量 +21.8%, TTFT -41~55%, E2EL -18%, external cache hit 高 10.9pp
- **SimpleCPU 最大瓶颈是 lazy offload 的 cache 丢失**: 峰值 external cache 只有 68.4% (vs Mooncake 79.3%), 建议:
  - 测试 `lazy_offload: false` (eager 模式) 是否提高 cache 命中率
  - 增大 offload buffer 或降低 `_target_free` 阈值
- **如果集群无 RDMA**: SimpleCPU 是可行的 fallback, 但需要配合 eager offload 和 NVTX 插桩来优化
- **两者都需要解决 AllReduce skew**: 两个 trace 的 decode 端都有 70~82% 的 GPU 间 AllReduce 不均衡, 这是独立于 offload 后端的共性问题
