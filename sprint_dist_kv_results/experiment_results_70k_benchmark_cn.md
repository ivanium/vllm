# Experiment Results: Kimi NVFP4 PD 1P1D TP4 - 70K Long-Context Multi-Turn Benchmark

## 实验概要

- **模型**: nvidia/Kimi-K2.5-NVFP4, TP4, Eagle3 speculative decoding
- **负载**: random 70K input tokens, 300 output tokens, 10 multi-turn, sweep concurrency 1/2/4/8
- **硬件**: GB200 NVL72, 2 nodes × 4 GPUs (1 prefill + 1 decode)
- **日期**: 2026-04-11
- **配置差异**: 仅 offload backend 不同，其余配置完全对齐（FA4, fp8 KV cache, MoE FP4, NCCL/GLOO 完整配置）

## Update: 双 SimpleCPU 口径

- 文档现在统一展示两个 SimpleCPU 版本：
  - **SimpleCPU (lazy)**: `20260411_034812`
  - **SimpleCPU (eager)**: `20260411_135828`
- `eager` 这组已确认是 **真正的 SimpleCPU**：`SimpleCPUOffloadConnector ... mode=eager`
- cache hit 口径：
  - **Mooncake**: prefix `44.1%`, external `74.8%`
  - **SimpleCPU (lazy/eager)**: prefix `42.8%`, external `0.0%`
- 结论上，`eager` 比 `lazy` 更公平，尤其在 `c=2 / c=4` 更强；但 **Mooncake 的高并发优势仍然成立**

## 三组配置

| 配置 | Prefill Offload | Decode Offload | CPU Memory |
|------|----------------|----------------|-----------|
| **MooncakeStore** | MultiConnector(Nixl + MooncakeStoreConnector) | 纯 Nixl | 100 GB/worker × 4 = 400 GB |
| **SimpleCPU (lazy)** | MultiConnector(Nixl + SimpleCPUOffloadConnector, lazy) | 纯 Nixl | 400 GB total |
| **SimpleCPU (eager)** | MultiConnector(Nixl + SimpleCPUOffloadConnector, eager) | 纯 Nixl | 400 GB total |

---

## 1. 性能对比

### Sweep Summary

| Concurrency | Mooncake Req/s | SimpleCPU Req/s (lazy) | SimpleCPU Req/s (eager) |
|-------------|---------------|------------------------|-------------------------|
| 1 | 0.29 | 0.29 | **0.32** |
| 2 | 0.54 | 0.49 | **0.55** |
| 4 | **0.60** | 0.52 | 0.56 |
| 8 | **0.76** | 0.45 | 0.44 |
| **Best** | **0.76 @c=8** | 0.52 @c=4 | 0.56 @c=4 |

### TTFT (P50)

| Concurrency | Mooncake | SimpleCPU (lazy) | SimpleCPU (eager) |
|-------------|---------|------------------|-------------------|
| 1 | 828 ms | 809 ms | 814 ms |
| 2 | 978 ms | 979 ms | **966 ms** |
| 4 | **1476 ms** | 2960 ms | 2472 ms |
| 8 | **2688 ms** | **14336 ms** | 14587 ms |

### TPOT (P50)

| Concurrency | Mooncake | SimpleCPU (lazy) | SimpleCPU (eager) |
|-------------|---------|------------------|-------------------|
| 1 | 6.64 ms | 6.60 ms | 6.49 ms |
| 2 | 7.64 ms | 7.78 ms | 7.65 ms |
| 4 | 15.93 ms | 15.53 ms | 14.66 ms |
| 8 | 22.98 ms | 12.25 ms | 12.03 ms |

### E2EL (P50)

| Concurrency | Mooncake | SimpleCPU (lazy) | SimpleCPU (eager) |
|-------------|---------|------------------|-------------------|
| 1 | 2811 ms | 2783 ms | **2769 ms** |
| 2 | 3283 ms | 3301 ms | **3225 ms** |
| 4 | **6382 ms** | 7571 ms | 6970 ms |
| 8 | **9648 ms** | **17938 ms** | 18151 ms |

---

## 2. 缓存命中率

| 指标 | Mooncake | SimpleCPU (lazy) | SimpleCPU (eager) |
|------|---------|------------------|-------------------|
| Prefix cache hit rate | 44.1% | 42.8% | 42.8% |
| **External prefix cache hit rate** | **74.8%** | 0.0% | 0.0% |

Mooncake 依赖 external prefix cache hit；两个 SimpleCPU 版本都没有 external cache hit。`eager` 和 `lazy` 的差别主要不是 cache hit，而是 offload 行为本身带来的中低并发开销。

---

## 3. 传输带宽分析

### Nixl P→D 传输（decode 端测量）

每个 70K 请求需要传 ~2.2 GB KV 到 decode 端。

| 指标 | Mooncake | SimpleCPU (lazy) | SimpleCPU (eager) |
|------|---------|------------------|-------------------|
| Avg xfer time | 292 ms | 357 ms | 358 ms |
| Avg post time | 229 ms | 269 ms | 310 ms |
| **Avg total Nixl latency** | **521 ms** | **626 ms** | **668 ms** |
| Avg throughput | 5.0-7.9 GB/s | 4.7-4.8 GB/s | 4.7-4.8 GB/s |
| Max xfer time | 808 ms | 1386 ms | 478 ms |

Mooncake 的 Nixl 传输仍更轻，因为 prefill 端有 external cache hit。两个 SimpleCPU 版本在高并发下的 Nixl 开销量级接近，`eager` 的收益主要体现在 c=1/2/4 的整体请求路径上。

### Nixl 占 TTFT 的比重

| Concurrency | Mooncake Nixl/TTFT | SimpleCPU (lazy) Nixl/TTFT | SimpleCPU (eager) Nixl/TTFT |
|-------------|-------------------|----------------------------|-----------------------------|
| c=1 | **63%** | **77%** | **82%** |
| c=2 | 53% | 64% | 69% |
| c=4 | 35% | 21% | 27% |
| c=8 | 19% | 4% | 5% |

**低并发时 Nixl 是 TTFT 的主要成分**（63-77%）。高并发时排队成为主因。

### Mooncake Store offload/fetch 带宽

| 方向 | 吞吐量 | 平均延迟 | 平均批大小 |
|------|--------|---------|-----------|
| **GPU→CPU put** | 11.2-11.6 GiB/s | 5.2 ms | 59.7 MiB |
| **CPU→GPU get** | 9.2 GiB/s | 194.5 ms | 1789.5 MiB (1.75 GiB) |

Put（offload）小而快（60 MiB/5ms），get（prefix load）大而慢（1.75 GiB/195ms）。

### C++ RDMA 传输累计量（Mooncake Store）

| 方向 | 总量 | 次数 | p95 延迟 |
|------|------|------|---------|
| Total Read (CPU→GPU) | 292.11 GB | 299 batch | p95 < 200 ms |
| Total Write (GPU→CPU) | 24.63 GB | 424 batch | p95 < 5 ms |

Read >> Write（12:1 比例），说明大量 prefix 被多次复用（从 Store 读回）。

---

## 4. Decode 端负载分析

c=8 时 decode 端的关键差异：

| 指标 | Mooncake | SimpleCPU (lazy) | SimpleCPU (eager) |
|------|---------|------------------|-------------------|
| Running reqs | **5-6** | **2** | **1-2** |
| Waiting reqs | 1-2 | 0 | 0-1 |
| **KV cache usage** | **89.5%** | **31.6%** | **31.5-31.6%** |
| Gen throughput | 101-378 tok/s (波动) | 75-188 tok/s | 61-190 tok/s |

**Mooncake 的 TPOT 更高（22.98 vs 12.25/12.03ms）是因为 decode 端同时服务更多请求**：
- Mooncake prefill 端有 74.8% external cache hit → 请求更快完成 prefill → 更快涌入 decode
- 两个 SimpleCPU prefill 端都没有 external cache hit → 每个请求都要完整 prefill 70K tokens → 天然限流
- decode 端 5-6 个 70K 请求 KV cache 占 89.5% HBM → 大 batch attention → TPOT 更高

**这不是 Mooncake 的缺陷，是优势的副作用**——prefill 更高效导致 decode 端更忙。总 E2EL 仍然优于两个 SimpleCPU 版本（9648 vs 17938/18151 ms @c=8）。

---

## 5. Mooncake Master 状态

| 指标 | 值 |
|------|---|
| batch_put_start_requests_total | 32,362 |
| successful_evictions_total | **167** |

有 167 次 LRU 驱逐——确认 offload 发生了。

---

## 6. 结论

1. **Mooncake Store 在高并发下吞吐领先明显**：相对 `SimpleCPU (lazy)` 为 `0.76 vs 0.52 req/s`，相对 `SimpleCPU (eager)` 为 `0.76 vs 0.44 req/s`
2. **真正的 SimpleCPU（eager）在 c=2 / c=4 更强**：`c=2` 基本追平 Mooncake 吞吐，`c=4` 也明显优于 `lazy`
3. **代价是 Mooncake 的 TPOT 更高**（22.98 vs 12.25/12.03 ms @c=8），因为 decode 端更忙——这是 prefill 更高效的副作用
4. **External cache hit rate 74.8%** 是核心优势——大量 prefix 从 CPU memory 直接加载，避免重复 prefill
5. **Nixl P→D 传输**在低并发时占 TTFT 的 53-82%，是主要延迟瓶颈之一

---

## 7. 日志路径

| 配置 | 日志 |
|------|------|
| Mooncake Store | `vigil/logs/pd_kimi_bench_a_70k/2026-04-11/20260411_034826` |
| SimpleCPU (lazy) | `vigil/logs/pd_kimi_bench_a_70k_baseline_simplecpu/2026-04-11/20260411_034812` |
| SimpleCPU (eager) | `vigil/logs/pd_kimi_bench_a_70k_baseline_simplecpu/2026-04-11/20260411_135828` |
