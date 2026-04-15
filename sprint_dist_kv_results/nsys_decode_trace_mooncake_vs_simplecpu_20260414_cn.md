# Nsys Decode Trace Analysis: MooncakeStore vs SimpleCPUOffload (2026-04-14)

**日期**: 2026-04-14
**模型**: nvidia/Kimi-K2.5-NVFP4 (MoE, DeepSeekV2 MLA, EAGLE3 speculative decode, 3 spec tokens)
**硬件**: 4x NVIDIA GB200 (189 GB, 152 SMs), TP4, NVLink
**架构**: P/D 分离部署, prefill 1 节点 + decode 1 节点, round-robin router
**负载**: Random 70K input + 300 output (multi-turn 10 轮, 并发 1/2/4/8) + ShareGPT (并发 4/8)
**Profiling**: Nsys CUDA trace, decode 端, `delay_iterations=5800, max_iterations=300`
**vLLM 版本**: 0.1.dev17+ga1517daa6.d20260412

---

## 配置差异

两次运行的**唯一区别**是 prefill 端 MultiConnector 中的 KV cache offload 后端:

| 配置项 | Mooncake | SimpleCPU |
|--------|----------|-----------|
| Prefill KV-transfer | NixlConnector + **MooncakeStoreConnector** (`load_async: true`) | NixlConnector + **SimpleCPUOffloadConnector** (`cpu_bytes: 400GB, lazy_offload: true`) |
| Decode KV-transfer | NixlConnector (完全相同) | NixlConnector (完全相同) |
| 其他所有配置 | 完全相同 | 完全相同 |

> Decode 端配置完全一致, 差异仅在 prefill 端。下文分析的是 offload 后端选择对 decode 端的**间接影响**。

日志路径:
- Mooncake: `vigil/logs/pd_kimi_70k_nsys/2026-04-14/20260414_020701/`
- SimpleCPU: `vigil/logs/pd_kimi_70k_simplecpu_nsys/2026-04-14/20260414_020703/`

---

## 1. 端到端 Benchmark 对比

### 1.1 Random 70K Multi-turn (10 轮对话)

| 指标 | 并发 | Mooncake | SimpleCPU | 差异 |
|------|------|----------|-----------|------|
| Mean TTFT (ms) | c=1 | 1152 | **1106** | SimpleCPU 快 4% |
| | c=2 | 1142 | **1117** | SimpleCPU 快 2% |
| | c=4 | 1766 | **1456** | SimpleCPU 快 18% |
| | c=8 | **2473** | 2987 | Mooncake 快 17% |
| Mean TPOT (ms) | c=1 | 6.33 | **6.13** | SimpleCPU 快 3% |
| | c=2 | **7.33** | 7.38 | ~持平 |
| | c=4 | **15.61** | 15.92 | ~持平 |
| | c=8 | 21.85 | **20.41** | SimpleCPU 快 7% |
| Mean ITL (ms) | c=1 | 10.16 | **10.08** | ~持平 |
| | c=8 | 21.96 | **21.18** | SimpleCPU 快 4% |
| Mean E2EL (ms) | c=1 | 3046 | **2938** | SimpleCPU 快 4% |
| | c=8 | **9006** | 9091 | ~持平 |
| Output throughput (tok/s) | c=1 | 98.5 | **102.1** | SimpleCPU 高 4% |
| | c=4 | 182.1 | **188.4** | SimpleCPU 高 3% |
| | c=8 | **264.1** | 261.7 | ~持平 |
| P99 TTFT (ms) | c=1 | 4607 | **4180** | SimpleCPU 快 9% |
| | c=4 | 12832 | **7807** | SimpleCPU 快 39% |
| | c=8 | 13322 | **12383** | SimpleCPU 快 7% |

### 1.2 ShareGPT (真实对话数据, 412 请求)

| 指标 | 并发 | Mooncake | SimpleCPU | 差异 |
|------|------|----------|-----------|------|
| Mean TTFT (ms) | c=4 | 693 | **697** | ~持平 |
| | c=8 | 786 | **745** | SimpleCPU 快 5% |
| Mean TPOT (ms) | c=4 | 13.08 | **12.84** | SimpleCPU 快 2% |
| | c=8 | 18.03 | **15.46** | SimpleCPU 快 14% |
| Mean E2EL (ms) | c=4 | 4524 | **4483** | SimpleCPU 快 1% |
| | c=8 | 6114 | **5309** | SimpleCPU 快 13% |
| Output throughput (tok/s) | c=4 | 249.0 | **255.9** | SimpleCPU 高 3% |
| | c=8 | 311.7 | **341.3** | SimpleCPU 高 10% |
| P99 TTFT (ms) | c=4 | 1417 | **1367** | SimpleCPU 快 4% |
| | c=8 | 1706 | **1438** | SimpleCPU 快 16% |

### 1.3 Benchmark 结论

与 4/13 的实验结果不同, **本次 (4/14) 运行中 SimpleCPU 在大多数指标上略优或持平**:

- **低并发 (c=1~4)**: SimpleCPU TTFT 好 2~18%, TPOT 持平, 整体 E2EL 略优
- **高并发 (c=8)**: SimpleCPU 在 ShareGPT 场景显著领先 (throughput +10%, TPOT -14%), Random 70K 场景两者持平
- **P99 尾延迟**: SimpleCPU 的 P99 TTFT 在 c=4 时比 Mooncake 快 39%, 本次没有出现上次的 P99 TPOT 退化

> 注意: 4/13 run 中 Mooncake 大幅领先 (吞吐 +22%, TTFT -55%), 4/14 反转。需排查是否由节点间网络差异 (rack1-07/08 vs rack1-03/04) 或 prefill 侧 mooncake transfer service 状态引起。

---

## 2. Decode 端 Nsys Trace 深度分析

### 2.1 Profiling 窗口说明

| 参数 | 值 | 说明 |
|------|-----|------|
| delay_iterations | 5800 | 前 5800 个 scheduler step 不 profiling |
| max_iterations | 300 | 捕获 300 个 step |
| 实际捕获时长 | ~5.4-5.5s | GPU kernel 跨度 |
| 对应负载阶段 | Random 70K, c=1~2 | multi-turn warm-up 后期, prefix cache ~82% |

Profiling 窗口期间:
- **Mooncake**: GPU KV cache 0~24%, prefix cache hit 81~82%, 1~2 并发请求
- **SimpleCPU**: GPU KV cache 0~24%, prefix cache hit 81~82%, 1~2 并发请求
- 两者在 profiling 窗口内的负载特征相似

### 2.2 GPU Kernel 时间总量

| 指标 | Mooncake | SimpleCPU | 差异 |
|------|----------|-----------|------|
| 总 GPU kernel 时间 | **20,393 ms** | 22,485 ms | SimpleCPU 多 10.2% |
| 总 kernel 调用次数 | 2,239,116 | 2,244,300 | ~持平 |
| 平均 kernel 时长 | **9.11 us** | 10.02 us | SimpleCPU 慢 10% |
| Decode step 总数 | 1,200 | 1,201 | ~相同 |
| 平均 step 时长 | **9.86 ms** | 10.35 ms | SimpleCPU 慢 5% |

### 2.3 Top Kernel Hotspots 对比

| 排名 | Kernel | Mooncake (ms) | SimpleCPU (ms) | 差异 |
|------|--------|---------------|----------------|------|
| 1 | fmhaSm100f (MLA Decode Attention) | 5,184 (25.4%) | 5,408 (24.1%) | +4.3% |
| 2 | **allreduce_fusion_kernel_oneshot_lamport** | **1,547 (7.6%)** | **3,118 (13.9%)** | **+101%** |
| 3 | bmm (FP4 MoE GEMM) | 2,120 (10.4%) | 1,659+534 (9.8%) | ~持平 |
| 4 | bmm (BF16 MLA Up-proj) | 1,245 (6.1%) | 652+342+346 (6.0%) | ~持平 |
| 5 | fmhaSm100f (Prefill/Verification Attn) | 1,018 (5.0%) | 1,261 (5.6%) | +24% |
| 6 | router_gemm (MoE routing) | 593 (2.9%) | 630 (2.8%) | +6% |
| 7 | fused_a_gemm | 572 (2.8%) | 591 (2.6%) | +3% |
| 8 | quantize_with_block_size | 437 (2.1%) | 441 (2.0%) | ~持平 |
| 9 | elementwise_kernel | 460 (2.3%) | 555 (2.5%) | +21% |
| 10 | concat_and_cache_mla_kernel | 199 (1.0%) | — | — |

**关键发现**: AllReduce 是两者之间最大的差异来源, SimpleCPU 的 AllReduce 总耗时是 Mooncake 的 **2 倍**。其他 kernel (Attention, GEMM, MoE) 差异在 ±10% 以内。

### 2.4 AllReduce 延迟分布 (核心差异)

| 延迟区间 | Mooncake 次数 | Mooncake 时间 | SimpleCPU 次数 | SimpleCPU 时间 |
|----------|---------------|---------------|----------------|----------------|
| < 10 us | 121,172 | 969 ms | 118,279 | 961 ms |
| 10-50 us | 33,529 | 419 ms | 36,326 | 447 ms |
| 50-100 us | 9 | 0.5 ms | 38 | 2.9 ms |
| 100 us - 1 ms | 52 | 26 ms | 105 | 39 ms |
| 1-10 ms | 36 | 110 ms | 49 | 153 ms |
| **> 10 ms** | **2** | **22 ms** | **3** | **1,514 ms** |

**关键发现**:

- 正常 AllReduce (< 50us) 两者几乎一致, 说明基线通信性能相同
- SimpleCPU 有 **3 次极端 AllReduce stall (> 10ms)**, 总计 **1,514 ms**, 平均每次 **~505 ms**
- Mooncake 仅 2 次 > 10ms stall, 总计 22 ms, 平均每次 ~11 ms
- 这 3 次极端 stall **贡献了 SimpleCPU AllReduce 总开销差异的 95%** (1514/1571)

### 2.5 Memory Copy 对比

| 类型 | 指标 | Mooncake | SimpleCPU | 差异 |
|------|------|----------|-----------|------|
| **P2P (NVLink/NIXL)** | 次数 | 36,704 | **49,352** | +34% |
| | 总数据量 | 4.499 GB | **5.624 GB** | +25% |
| | 总耗时 | 176 ms | **189 ms** | +7% |
| | 每设备次数 | 9,176 | 12,338 | +34% |
| | 每设备数据 | 1.125 GB | 1.406 GB | +25% |
| H2D | 次数 | 27,848 | 27,883 | ~持平 |
| | 总耗时 | 50.9 ms | 51.4 ms | ~持平 |
| | **总数据量** | **0.042 GB** | **0.045 GB** | **非 KV cache** |
| D2D | 次数 | 12,000 | 12,000 | 相同 |
| | 总数据量 | 7.973 GB | 8.258 GB | +4% |
| D2H | 次数 | 2,432 | 2,440 | ~持平 |

**H2D 数据详解 (重要)**:

H2D 的 ~50ms / 42MB **不是 KV cache CPU→GPU load**, 而是 decode 引擎的常规小数据传输:

| H2D 大小区间 | Mooncake 次数 | SimpleCPU 次数 | 说明 |
|-------------|---------------|----------------|------|
| < 1 KB (avg 56B) | 26,632 | 26,662 | 采样参数、token IDs、indices |
| 1-64 KB (avg 27KB) | 1,200 | 1,201 | scheduler metadata (≈1 per step) |
| 64KB-1MB (512KB) | 16 | 20 | NIXL block table / 大 metadata |

所有 H2D copy 都在 stream 19 (主 compute stream) 上, 两个 run 完全一致。**真正的 KV cache 数据走 P2P (NIXL 跨节点 NVLink/RDMA), 不走 H2D。**

**分析**: SimpleCPU 的 P2P 传输比 Mooncake 多 34% (次数) / 25% (数据量)。P2P 传输主要来自 NIXL 跨节点 KV cache transfer (streams 7120-7124, 多通道并发)。更多的 P2P 传输意味着 SimpleCPU prefill 端向 decode 端推送了更多 KV cache 数据, 这与不同的 prefix cache 行为一致。

### 2.6 NIXL Transfer NVTX 对比

| NVTX Range | Mooncake | SimpleCPU |
|------------|----------|-----------|
| nixl_read_blocks 次数 | 16 (4 requests × 4 TP) | 20 (5 requests × 4 TP) |
| nixl_read_blocks 平均 | 10.65 ms | 11.62 ms |
| nixl_read_blocks 总计 | 170 ms | 232 ms |
| 每次读取 blocks | n=6944 | n=6944 |

SimpleCPU 在 profiling 窗口内多了 1 个 NIXL transfer request, 且每次传输平均慢 ~1ms。每次传输的 block 数量相同 (6944), 说明单次传输的 KV cache 大小一致 (70K tokens → 6944 blocks)。

### 2.7 Decode Step 分布

| Step 类型 | Mooncake | SimpleCPU | 说明 |
|-----------|----------|-----------|------|
| context_0 generation_2 (8 tokens) | 168 步 | 12 步 | CUDAGraph 小 batch |
| context_0 generation_3 (12 tokens) | 532 步 | **700 步** | CUDAGraph 中 batch |
| context_0 generation_4 (16 tokens) | 484 步 | 469 步 | CUDAGraph 大 batch |
| context_1 generation_2 (非 graph) | 4 步 | 4 步 | 验证/冷启动 |
| context_1 generation_3 (非 graph) | 12 步 | 16 步 | 验证/冷启动 |
| **总 step** | **1,200** | **1,201** | ~相同 |

generation 后的数字对应 EAGLE3 speculative decode 的 batch size: `(accepted_tokens + 1) × TP_tokens`。Mooncake 有更多 generation_2 (小 batch) 步, 说明在 profiling 窗口内 spec decode acceptance rate 分布略有不同。

---

## 3. KV Transfer 指标分析 (从 vLLM 日志)

### 3.1 全量统计 (整个 run)

| 指标 | Mooncake | SimpleCPU |
|------|----------|-----------|
| KV Transfer 报告次数 | 755 | 760 |
| 平均 xfer time | 40.90 ms | 41.86 ms |
| 中位数 xfer time | 11.76 ms | 13.02 ms |
| P90 xfer time | 163 ms | 182 ms |
| 最大 xfer time | 385 ms | **575 ms** |
| 平均 post time | 32.07 ms | 30.57 ms |
| 平均 throughput | 11,424 MB/s | 12,086 MB/s |
| 中位数 throughput | 5,093 MB/s | 5,984 MB/s |
| 平均 descriptors | 7,507 | 8,237 |

### 3.2 稳态 (低并发, 前 20 轮 multi-turn)

在低并发 multi-turn warm-up 阶段 (prefix cache 逐步建立), 两者 KV transfer 特征:

| 指标 | Mooncake | SimpleCPU |
|------|----------|-----------|
| 典型 xfer time (prefix cache > 60%) | 2.8-3.5 ms | **2.6-2.8 ms** |
| 典型 throughput | 76-95 GB/s | **95-104 GB/s** |
| 首次冷传输 (70K tokens) | 95.4 ms | 101.6 ms |

**SimpleCPU 在稳态小量传输时 throughput 更高** (~100 GB/s vs ~85 GB/s), 可能因为 SimpleCPU prefill 端的 KV cache 在 CPU 内存中的布局更有利于 NIXL 传输。

### 3.3 Decode 端稳态吞吐

| 指标 | Mooncake | SimpleCPU |
|------|----------|-----------|
| Generation throughput (>=2 reqs, mean) | 280.6 tok/s | **291.8 tok/s** |
| Generation throughput (>=2 reqs, median) | 267.9 tok/s | **279.7 tok/s** |
| 最终 prefix cache hit rate | 92.3% | 83.7% |
| 最终 external prefix cache hit rate | 100% | 100% |

SimpleCPU decode 端在高并发时的 generation throughput 高 ~4%, 与 benchmark 结果一致。

---

## 4. Prefix Cache Hit Rate 对比

| 阶段 | Mooncake | SimpleCPU |
|------|----------|-----------|
| 初始 (首请求) | 0% | 0% |
| 5 轮 multi-turn 后 | ~57% | ~57% |
| 10 轮 multi-turn 后 | ~78% | ~78% |
| 最终稳态 | **92.3%** | 83.7% |
| External prefix cache hit rate | 100% | 100% |

Mooncake 的最终 prefix cache hit rate 高 ~8.6 个百分点。但在 profiling 窗口 (~82%) 时两者相近, 差异主要出现在后续高并发 ShareGPT 阶段。

---

## 5. CPU→GPU Load 路径源码分析 (重要: Profiling 盲区)

> 本次 nsys trace 仅捕获 **decode 端**。KV cache 的 CPU offload/load 发生在 **prefill 端** (prefill traces 目录为空, 未捕获)。以下分析 prefill 端两种 backend 的 load 机制, 解释为何 nsys 无法完整观测 CPU→GPU 的实际耗时。

### 5.1 SimpleCPUOffload Load 路径

**源码**: `simple_cpu_offload_connector.py` → `worker.py` → `copy_backend.py` → `cuda_mem_ops.py`

```
Scheduler step
  │
  ├─ build_connector_meta()     # 构建 (gpu_block, cpu_block) 映射
  │
  ├─ start_load_kv()            # ← 空操作! 故意延迟到 model forward 之后
  │
  ├─ [GPU model forward]        # 主 compute stream 执行模型推理
  │
  └─ get_finished()             # model forward 结束后:
       │
       ├─ _backend.launch_copy()    # 将 copy 任务放入 Python Queue (非阻塞, ~微秒)
       │    │
       │    └─ [Background Thread: _copy_loop]
       │         │
       │         ├─ copy_blocks()         # 调用 cuMemcpyBatchAsync()
       │         │    └─ _batch_memcpy_fn(dst_gpu, src_cpu, sizes, n,
       │         │                         attrs, stream=load_stream)
       │         │    # ↑ 异步! 立即返回, DMA 在 load_stream 上排队
       │         │
       │         ├─ event = torch.Event()
       │         └─ event.record(load_stream)  # 记录完成事件
       │
       └─ _poll_stream_events()     # 非阻塞轮询 event.query()
            └─ 返回已完成的 request IDs
```

**关键设计**:
1. **Compute-I/O overlap**: `start_load_kv()` 故意什么都不做, load 推迟到 `get_finished()` (model forward 之后), 用 GPU compute 时间掩盖 CPU 端 queue 开销 (~5ms)
2. **低优先级 CUDA stream**: `load_stream = torch.cuda.Stream(priority=low_pri)` — 让 compute 优先
3. **Pinned memory**: CPU tensor 通过 `cudaHostRegister` 注册为 pinned memory, 支持 DMA
4. **`cuMemcpyBatchAsync`**: 批量异步 H2D copy (driver API), 一次调用可传 `n × num_layers` 个 block
5. **完成跟踪**: CUDA event (`event.query()`) 非阻塞轮询, 不等待

**Profiling 盲区**:
- `perf_counter()` 在 `_copy_loop` 中测的是 `cuMemcpyBatchAsync` 的 **提交时间**, 不是 DMA 完成时间
- 实际 H2D DMA 在 `load_stream` 上异步执行, 完成时间取决于数据量和 PCIe/NVLink 带宽
- nsys **理论上可以捕获** `load_stream` 上的 H2D memcpy (因为走标准 CUDA API), 但需要 prefill 端 trace

### 5.2 MooncakeStore Load 路径

**源码**: `mooncake_store_connector.py` → `mooncake_store_worker.py` → Mooncake C++ 库

```
Scheduler step
  │
  ├─ build_connector_meta()     # 构建 LoadSpec (token range, block hashes)
  │
  ├─ [GPU model forward]
  │
  └─ get_finished()
       │
       ├─ kv_recv_thread.add_request()  # 放入队列 (非阻塞)
       │    │
       │    └─ [Background Thread: KVCacheStoreRecvingThread]
       │         │
       │         ├─ token_database.process_tokens()  # 计算 key 和 GPU 地址
       │         │
       │         ├─ store.batch_get_into_multi_buffers(
       │         │      keys,         # Mooncake store 中的 key
       │         │      gpu_addrs,    # 目标: GPU 显存地址 (指针)
       │         │      sizes         # 每个 block 的字节数
       │         │  )
       │         │  # ↑ Mooncake C++ 引擎: RDMA/TCP → GPU DMA
       │         │  # ↑ 绕过 CUDA API, 直接写入 GPU 显存
       │         │  # ↑ 对 nsys 完全不可见!
       │         │
       │         └─ set_finished_request(req_id)
       │
       └─ get_and_clear_finished_requests()  # 收集完成的 requests
```

**关键设计**:
1. **`load_async = True` (强制)**: worker.py line 1013 硬编码, 不可关闭
2. **零 CUDA stream**: Mooncake load **不使用任何 CUDA stream**, 不调用任何 CUDA memcpy API
3. **RDMA 直写 GPU**: `batch_get_into_multi_buffers` 接收 GPU 显存指针, 由 Mooncake 的 RDMA 引擎直接 DMA 写入
4. **同步语义**: RPC 返回 = 数据到达 GPU 显存 (从 Mooncake 引擎的角度)

**Profiling 盲区**:
- `perf_counter()` 测的是 `batch_get_into_multi_buffers` 的 **RPC 调用时间** (包含网络 + DMA)
- nsys **完全无法捕获** Mooncake 的数据传输 — 不走 CUDA API, RDMA DMA 对 CUPTI 不可见
- NVTX range `mooncake_store_get_n=X` 只标记 RPC 调用边界

### 5.3 两种 Backend Load 对比

| 特性 | SimpleCPU | MooncakeStore |
|------|-----------|---------------|
| 数据来源 | 本节点 CPU pinned memory | 远程 Mooncake transfer store |
| 传输 API | `cuMemcpyBatchAsync` (CUDA driver) | `batch_get_into_multi_buffers` (RDMA) |
| CUDA stream | `load_stream` (low priority) | 无 (不走 CUDA) |
| nsys 可观测性 | **部分可观测** (H2D memcpy 可见, 但需 prefill trace) | **完全不可观测** (RDMA 绕过 CUDA) |
| 同步方式 | CUDA event 非阻塞轮询 | RPC 返回 = 完成 |
| Compute-I/O overlap | load_stream 低优先级, 自动让步 compute | 后台线程, 与 GPU 完全解耦 |
| 数据路径 | CPU RAM → PCIe → GPU HBM | Remote GPU/CPU → RDMA NIC → GPU HBM |

### 5.4 对 Decode 端的间接影响

虽然 CPU→GPU load 发生在 **prefill 端**, 但通过以下链路间接影响 decode:

```
Prefill CPU→GPU load 完成
  → Prefill 可以开始处理下一个请求
  → NIXL 将 KV cache 从 Prefill GPU 传输到 Decode GPU (P2P)
  → Decode 端收到 KV cache, 开始 decode
```

- **SimpleCPU**: `cuMemcpyBatchAsync` 在低优先级 stream 上 → 可能与 prefill compute 竞争 PCIe 带宽 → 影响 NIXL transfer 启动时机
- **MooncakeStore**: RDMA 直写 → 与 CUDA compute 完全独立 → 但受网络带宽限制

decode trace 中观测到的差异 (AllReduce stall, P2P 传输量) 是这些 prefill 端行为差异的**下游效应**。

---

## 6. 根因分析

### 6.1 AllReduce 极端 Stall 根因

SimpleCPU decode trace 中出现 3 次 > 10ms 的 AllReduce stall (总计 1514ms, 最大 ~505ms), 而 Mooncake 仅 2 次 (总计 22ms)。

**推测根因**: NIXL P2P 传输与 AllReduce 的 **NVLink 带宽竞争**。

1. Decode 端的 TP AllReduce 和 NIXL KV transfer 都使用 NVLink
2. SimpleCPU 在 profiling 窗口内有 5 个 NIXL transfer (vs Mooncake 4 个), 且 P2P 传输总量多 34%
3. 当一个大的 NIXL transfer 与 AllReduce 时间重叠时, AllReduce 的某个 rank 被阻塞等待 NVLink 带宽, 导致其他 rank 空转等待同步
4. 这解释了为什么 stall 是极端值 (> 500ms) 而非均匀退化 — 只有 transfer 和 AllReduce 恰好重叠时才触发

**佐证**:
- AllReduce 正常区间 (< 50us) 的次数和时间两者几乎一致, 排除了 NCCL 配置差异
- P2P 传输量多 34% 增加了时间重叠的概率
- 极端 stall 在 Mooncake 中也存在但量级小得多 (11ms vs 505ms), 可能因为 MooncakeStore 的异步 load 机制减少了传输突发

### 6.2 P2P 传输量差异原因

SimpleCPU P2P 传输多 34% 可能由以下因素导致:

1. **Prefill 端 offload 行为差异**: SimpleCPU 使用 lazy offload, 可能在特定时间点触发批量 offload/restore, 影响了 NIXL transfer 的时序
2. **Profiling 窗口内多一个请求**: SimpleCPU 捕获了 5 个 NIXL transfer (vs 4 个), 每个 transfer 的 P2P copy 数量相同 (~6944 blocks × 4 TP), 多出的 1 个 request 贡献了 ~12,000 次额外 P2P copy
3. **请求调度时序**: 不同的 prefill 端 offload 延迟导致请求到达 decode 端的时序不同

### 6.3 Benchmark 结果与 Trace 分析的一致性

| 观测 | Trace 数据 | Benchmark 表现 | 一致性 |
|------|-----------|---------------|--------|
| SimpleCPU AllReduce stall | +1571ms | — | AllReduce stall 不直接影响吞吐, 因为是低并发窗口 |
| SimpleCPU kernel 时间 +10% | 22.5s vs 20.4s | — | trace 仅 5s, 需更长窗口验证 |
| SimpleCPU P2P 传输多 34% | 49K vs 37K | SimpleCPU throughput 略高 | 反直觉: 更多传输但更高吞吐 |
| SimpleCPU NIXL 传输更快 (稳态) | 2.6ms vs 3.0ms | SimpleCPU TPOT 更低 | 一致: 更快传输 → 更快恢复 decode |
| Mooncake prefix cache 更高 | 92% vs 84% | Mooncake 高并发 TTFT 有时更好 | 部分一致 |

### 6.4 与 4/13 实验结果差异说明

| 指标 | 4/13 结论 | 4/14 结论 | 可能原因 |
|------|-----------|-----------|----------|
| 吞吐量 | Mooncake +22% | SimpleCPU +3~10% | 不同节点 (rack1-07/08 vs 03/04) |
| TTFT | Mooncake 快 55% | SimpleCPU 快 2~18% | 4/13 高并发 (8), 4/14 sweep 1~8 |
| TPOT | SimpleCPU 快 7% | SimpleCPU 快 2~14% | 一致趋势, SimpleCPU TPOT 持续略优 |
| P99 TPOT | SimpleCPU 慢 110% | SimpleCPU 持平或略优 | 4/13 可能有网络抖动 |

---

## 7. 结论与建议

### 7.1 结论

1. **本次 (4/14) 运行中 SimpleCPU 整体性能略优于 Mooncake**, 尤其在 ShareGPT c=8 场景 (throughput +10%, TPOT -14%)。这与 4/13 的结果相反。

2. **Decode trace 揭示的主要差异是 AllReduce stall**: SimpleCPU 的 AllReduce 总耗时 2x (3208ms vs 1633ms), 由 3 次极端 stall 驱动。但这些 stall 发生在低并发 profiling 窗口, 对整体 benchmark 影响有限。

3. **KV Transfer 稳态性能**: SimpleCPU 在小量传输时吞吐更高 (~100 GB/s vs ~85 GB/s), 这直接有利于 multi-turn 场景下的 TTFT。

4. **CPU→GPU load 在当前 trace 中不可见**: 两种 backend 的 load 路径都是异步的, 且发生在 prefill 端 (未被 trace)。SimpleCPU 用 `cuMemcpyBatchAsync` (低优先级 CUDA stream), MooncakeStore 用 RDMA 直写 GPU (完全绕过 CUDA API)。Decode trace 中的 H2D copy (~50ms / 42MB) 仅是常规 metadata 传输, 不是 KV cache load。

5. **两次实验结果不一致, 需要更多控制变量的 A/B 测试** 来隔离节点间差异 (rack1-03/04 vs 07/08)。

### 7.2 建议

1. **同节点 A/B 测试**: 在相同节点对 (如 rack1-03/04) 上交替运行两种配置, 消除节点间网络差异
2. **Prefill 端 nsys trace**: 当前只有 decode 端 trace (prefill traces 为空)。要观测 CPU→GPU load 的实际耗时:
   - SimpleCPU: 需要在 prefill 端启用 nsys, 捕获 `load_stream` 上的 `cuMemcpyBatchAsync` (标准 CUDA API, nsys 可见)
   - MooncakeStore: nsys 无法捕获 RDMA 传输, 需要用 Mooncake 自身的 profiling 工具或 `perf_counter()` 日志
3. **延长 profiling 窗口到高并发阶段**: 当前 `delay_iterations=5800` 捕获的是低并发 warm-up 阶段, 考虑增加 delay 到 ShareGPT c=8 阶段 (约 iteration 8000+) 以捕获高压力下的 offload/load 行为
4. **关注 AllReduce-NIXL 冲突**: 如果在更多实验中复现极端 AllReduce stall, 考虑:
   - 将 NIXL transfer 与 AllReduce 分配到不同 NVLink 通道 (如果硬件支持)
   - 在 decode 端增加 NIXL transfer 的 rate limiting, 避免突发传输
5. **MooncakeStore 状态排查**: 检查 4/14 run 中 MooncakeStore transfer service 是否正常运行, mooncake 的 prefix cache hit rate 更高但 throughput 反而不如 SimpleCPU, 可能存在 mooncake 服务端的性能回归

---

## 附录: 数据源

| 数据 | Mooncake 路径 | SimpleCPU 路径 |
|------|--------------|----------------|
| Decode log | `decode-0/decode-0-gb200-rack1-08.log` | `decode-0/decode-0-gb200-rack1-04.log` |
| Nsys trace | `decode-0/traces/pd_dev_nsys_decode.nsys-rep` | `decode-0/traces/pd_dev_nsys_decode.nsys-rep` |
| SQLite export | `/tmp/mooncake_decode.sqlite` | `/tmp/simplecpu_decode.sqlite` |
| Benchmark JSONs | `openai-chat-infqps-*.json` | `openai-chat-infqps-*.json` |
| Results summary | `results.json` | `results.json` |
| Prefill log | `prefill-0/prefill-0-gb200-rack1-07.log` | `prefill-0/prefill-0-gb200-rack1-03.log` |
