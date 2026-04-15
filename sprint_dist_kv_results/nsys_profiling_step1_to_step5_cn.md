# Nsys Profiling Results: Step 1 - Step 6 (Final)

## 实验概要

- **日期**: 2026-04-12
- **nsys 版本**: 2026.2.1 (`nsight-systems-cli-2026.2.1`)
- **硬件**: GB200 NVL72, 2 nodes × 4 GPUs (1 prefill + 1 decode)
- **目标**: 建立 PD 分离 + Mooncake 配置的 nsys profiling 能力，拿到 baseline kernel 分布

---

## Step 1-3: Qwen3-0.6B (Dense 小模型验证)

**目的**: 验证 nsys 在 slurm remote + PD 模式下工作

| Step | 配置 | 节点 | nsys 端 | Trace 大小 | 状态 |
|------|------|------|---------|-----------|------|
| 1 | `pd_profile_step1_prefill_only.yaml` | rack1-07/08 | prefill | 9.2 MB | ✅ |
| 2 | `pd_profile_step2_decode_only.yaml` | rack1-07/08 | decode | 17 MB | ✅ |
| 3 | `pd_profile_step3_both.yaml` | rack1-15/16 | both | 9.3 + 17 MB | ✅ |

**Capture 方式**: 全生命周期（无 delay/duration），进程退出时 flush

**Trace 路径**:
- Step 1: `vigil/logs/pd_profile_step1_prefill_only/2026-04-12/20260412_074935/prefill-0/traces/nsys_profile.nsys-rep`
- Step 2: `vigil/logs/pd_profile_step2_decode_only/2026-04-12/20260412_075425/decode-0/traces/nsys_profile.nsys-rep`
- Step 3: `vigil/logs/pd_profile_step3_both/2026-04-12/20260412_075942/{prefill,decode}-0/traces/nsys_profile.nsys-rep`

### Qwen3-0.6B Kernel 概览

**Prefill 端** (Step 1, 112 kernel instances):
- 41.8% — `vectorized_elementwise_kernel<FillFunctor>` (内存初始化)
- 23.3% — `_topk_topp_kernel` (top-k/top-p 采样)
- 6.8% — `rms_norm_kernel`
- 3.3% — `nvjet_sm100_tst_128x256` (SM100 GEMM)
- 0.9% — `fmhaSm100fKernel...PagedKv...Persistent` (Flash Attention)

**Decode 端** (Step 2, 14364 kernel instances):
- 15.0% — `nvjet_sm100_tst_32x64` (decode GEMM)
- 8.9% — `rms_norm_kernel`
- 8.9% — `FillFunctor` (内存初始化)
- 8.6% — `nvjet_sm100_tst_8x64` (小 GEMM)
- 8.0% — `fmhaSm100f...PagedKv...PersistentSwapsAbForGen` (decode attention)

---

## Step 5: Kimi-K2.5-NVFP4 (MoE 大模型)

### Step 5a: Prefill 端

**配置**: PD 1P1D, TP=4, MultiConnector(Nixl + MooncakeStore), enforce-eager, 8K input × 2 prompts (c=1)
**Capture 方式**: `--capture-range=cudaProfilerApi` + `vllm-bench --profile`
**节点**: rack1-15/16
**Trace**: 59 MB
**路径**: `vigil/logs/pd_kimi_profile_step5_basic/2026-04-12/20260412_083746/prefill-0/traces/nsys_profile.nsys-rep`

| % Time | Instances | Kernel | 类型 |
|--------|-----------|--------|------|
| **94.0%** | 984 | `cross_device_reduce_1stage` | TP all-reduce (vLLM 自定义) |
| 0.8% | 240 | `bmm_E2m1...FP4` (512u2) | MoE FP4 GEMM |
| 0.5% | 240 | `bmm_Bfloat16_E2m1` (512) | MoE gate/up proj |
| 0.4% | 120 | `bmm_E2m1...FP4` (512) | MoE GEMM variant |
| 0.4% | 120 | `bmm_E2m1...FP4` (16x512) | MoE GEMM variant |
| 0.2% | 8 | `ncclAllGather_RING_LL` | NCCL AllGather |
| 0.2% | 488 | `nvjet_sm100_tst_64x16` (splitK) | attention KQV projection |
| 0.2% | 488 | `fmhaSm100f...Persistent` | Flash Attention (SM100) |
| 0.1% | 976 | `fused_add_rms_norm` | RMSNorm |
| 0.1% | 480 | `routingDeepSeek` | MoE routing (expert selection) |

**NVTX Ranges**:
- `execute_context_1(26)_generation_0(0)`: 8 instances, avg 375ms — 主 forward pass
- `execute_context_0(0)_generation_0(0)`: 16 instances, avg 0.39ms
- `NCCL:ncclAllGather`: 8 instances, avg 0.11ms

### Step 5b: Decode 端

**配置**: 同上但 decode only profiling, NixlConnector only
**Trace**: 661 MB
**路径**: `vigil/logs/pd_kimi_profile_step5_decode_only/2026-04-12/20260412_085429/decode-0/traces/nsys_profile.nsys-rep`

#### Kernel 时间分布

| % Time | Instances | Avg (μs) | Kernel | 类型 |
|--------|-----------|----------|--------|------|
| **94.2%** | 251,904 | **1,116** | `cross_device_reduce_1stage` | TP all-reduce (vLLM 自定义) |
| 0.3% | 124,928 | 7.8 | `fused_a_gemm_kernel<1,2112,7168>` | MoE FP4 GEMM |
| 0.3% | 124,928 | 7.7 | `nvjet_sm100_tst_64x8` | decode GEMM (TNT) |
| 0.3% | 124,928 | 7.0 | `fmhaSm100f...FP8...PersistentSwaps` | Flash Attention (MLA, FP8 KV) |
| 0.3% | 249,856 | 3.4 | `fused_add_rms_norm` | RMSNorm + residual |
| 0.3% | 124,928 | 6.6 | `nvjet_sm100_tst_24x64` | decode GEMM (TNN) |
| 0.3% | 92,160 | 8.8 | `bmm_Bfloat16_E2m1` | MoE GEMM |
| 0.2% | 2,048 | **359** | `ncclAllGather_RING_LL` | NCCL AllGather |
| 0.2% | 61,440 | 11.3 | `bmm_E2m1_FP4` | MoE FP4 GEMM |
| 0.2% | 251,904 | 2.6 | `rms_norm_kernel` | RMSNorm |
| 0.2% | 122,880 | 5.1 | `routingIndicesClusterKernel` | MoE routing (expert selection) |
| 0.2% | 122,880 | 4.4 | `routingMainKernel` | MoE routing (main) |
| 0.2% | 249,856 | 2.1 | `cvt_fp16_to_fp4` | FP16→FP4 conversion |
| 0.2% | 122,880 | 4.3 | `finalizeKernel` | MoE finalize |

#### NVTX Ranges

| Range | Instances | Avg (ms) | 说明 |
|-------|-----------|----------|------|
| `execute_context_0(0)_generation_1(1)` | 2,032 | **198** | decode generation 步（主循环，即 TPOT） |
| `execute_context_1(1)_generation_0(0)` | 16 | 329 | 首次 prefill context（从 Nixl 收到 KV 后） |
| `execute_context_0(0)_generation_0(0)` | 808 | 0.82 | 轻量 context |
| `NCCL:ncclAllGather` | 2,048 | 0.07 | AllGather |

#### Memory Operations

| 操作 | Count | % Time | 说明 |
|------|-------|--------|------|
| Device-to-Device | 260,080 | 91.0% | KV cache 内部搬运 |
| Host-to-Device | 14,368 | 5.8% | 可能是 Nixl KV transfer 到 GPU |
| Device-to-Host | 2,048 | 1.6% | token output |
| Peer-to-Peer | 976 | 1.2% | TP 间数据交换 |

#### Decode 端深入分析

1. **`cross_device_reduce_1stage` 平均 1.1ms/次，共 251,904 次**
   - Kimi 有 61 层 transformer，每层 MoE 后需要 all-reduce
   - 4 个请求 × 128 output tokens = 512 decode steps
   - 251,904 / 512 ≈ **492 次 all-reduce/step** — 远超 61 层，说明每层有多次 all-reduce（MoE shared expert + routed expert 各一次，加上 attention QKV projection 的 all-reduce）
   - 每 step 的 all-reduce 总耗时：492 × 1.1ms ≈ **541ms**，但 NVTX 显示 TPOT=198ms，说明 all-reduce 和 compute 有并行重叠

2. **TPOT = 198ms（NVTX `generation_1(1)` avg）**
   - 对 c=1, 8K context 的 decode 来说偏高
   - 同事配置用了 `VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm`，可能显著降低 all-reduce 延迟

3. **Flash Attention 只有 7μs/次**
   - MLA attention kernel (`HQk576 HV512`, FP8 KV, Persistent) 极快
   - Decode 阶段 attention 不是瓶颈

4. **NCCL AllGather avg 359μs，只有 2,048 次**
   - 这不是 per-layer 的 all-reduce，而是 Nixl handshake 或 metadata 交换
   - 实际的 TP all-reduce 走的是 vLLM 自定义 `cross_device_reduce_1stage`，不走 NCCL

5. **MoE 全流程**: routing (5.1μs + 4.4μs) → FP4 GEMM (7.8μs) → finalize (4.3μs) → all-reduce (1,116μs)
   - MoE 计算只有 ~21μs，但 all-reduce 要 1,116μs — **通信计算比 53:1**

---

## Step 6: 对齐 pd_kimi_bench_a_70k.yaml 的完整配置 (Final)

**配置**: 完全对齐 `pd_kimi_bench_a_70k.yaml`（`-O3`, speculative Eagle3, `FULL_DECODE_ONLY` CUDA graph, `fuse_allreduce_rms`, Mooncake + Nixl）
**负载**: 70K random input × 2 prompts, 300 output, c=1
**nsys**: `cudaProfilerApi` + `repeat:1:sync`, `-t cuda,nvtx,cublas`

### Step 6a: Prefill 端（`-O3 --enforce-eager`）

**Trace**: 53 MB
**路径**: `vigil/logs/pd_kimi_70k_nsys_prefill/2026-04-12/20260412_152635/prefill-0/traces/nsys_profile.nsys-rep`

| % Time | Instances | Avg (μs) | Kernel | 类型 |
|--------|-----------|----------|--------|------|
| **93.2%** | 1,056 | **1,521** | `cross_device_reduce_1stage` | TP all-reduce (vLLM 自定义) |
| 1.7% | 480 | 59.5 | `bmm_E2m1...FP4` (16x512) | MoE FP4 GEMM |
| 0.7% | 360 | 34.6 | `bmm_Bfloat16_E2m1` (16x512) | MoE GEMM |
| 0.4% | 32 | 194 | `ncclAllGather_RING_LL` | NCCL AllGather |
| 0.3% | 488 | 9.0 | `nvjet_sm100_tst_64x16` (splitK) | attention GEMM |
| 0.2% | 488 | 6.6 | `fmhaSm100f...Persistent` | Flash Attention (SM100) |

**Prefill 端仍然 93% all-reduce** — 因为 prefill 用 `--enforce-eager`，不走 torch.compile，所以 `fuse_allreduce_rms` 和 trtllm allreduce 都不生效。这是 `pd_kimi_bench_a_70k.yaml` 的原始设计。

### Step 6b: Decode 端（`-O3` + `FULL_DECODE_ONLY` CUDA graph + `fuse_allreduce_rms`）

**Trace**: 98 MB
**路径**: `vigil/logs/pd_kimi_70k_nsys_decode/2026-04-12/20260412_152636/decode-0/traces/nsys_profile.nsys-rep`

| % Time | Instances | Avg (μs) | Kernel | 类型 |
|--------|-----------|----------|--------|------|
| **27.1%** | 72,704 | **28.7** | `allreduce_fusion_kernel_oneshot_lamport` | **flashinfer trtllm allreduce** |
| 4.8% | 34,160 | 10.7 | `fmhaSm100f...FP8...MultiCtas...Static` | Flash Attention (MLA, FP8 KV, multi-CTA) |
| 3.7% | 38,056 | 7.6 | `nvjet_sm100_tst_64x8` | decode GEMM (TNT) |
| 3.4% | 34,648 | 7.6 | `fused_a_gemm_kernel<1,2112,7168>` | MoE FP4 GEMM |
| 2.8% | 34,648 | 6.1 | `nvjet_sm100_tst_24x64` | decode GEMM (TNN) |
| 2.7% | 568 | 368 | `allreduce_fusion_kernel_oneshot_lamport` (variant) | trtllm allreduce (large) |
| 2.6% | 2,272 | 89.0 | `nvjet_sm100_tst_192x8` (splitK) | large GEMM |
| 2.4% × 4 | 8,520 each | 20-22 | `bmm_E2m1...FP4` | MoE FP4 GEMMs (多种配置) |
| 2.3% | 34,080 | 5.3 | `routingIndicesCluster` | MoE routing |
| 2.2% | 68,160 | 2.5 | `cvt_fp16_to_fp4` | FP16→FP4 conversion |

**Decode 端 all-reduce 从 94% 降到 ~30%！** 关键变化：
- `cross_device_reduce_1stage` (avg 1.1ms) → `allreduce_fusion_kernel_oneshot_lamport` (avg **28.7μs**)
- **all-reduce 延迟降低 40x**（1,100μs → 28.7μs）
- 这是 `-O3` 的 `fuse_allreduce_rms` pass 把 allreduce 和 RMSNorm fuse 到了 flashinfer 的 lamport allreduce kernel

### NVTX Annotations — Nixl P→D 传输（decode 端可见）

| NVTX Range | Instances | Avg | 说明 |
|-----------|-----------|-----|------|
| `nixl_read_blocks_for_req=chatcmpl` | 8 | **66.8ms** | per-request per-TP-rank KV 传输入口 |
| `nixl_read_blocks_n=62` | 8 | **66.7ms** | 单次 RDMA READ（62 blocks × 4 TP ranks × 2 requests） |
| `nixl_xfer_done_req=chatcmpl-...` | 8 | **29μs** | 传输完成确认 |

**Nixl P→D 传输**：每次 ~67ms，传输 62 个 KV cache blocks（70K tokens / block_size）。传输完成确认只需 29μs（只是 check state + release handle）。

**Mooncake Store NVTX 未触发** — 2 个 70K 请求 c=1 串行，GPU 显存足够不需要 offload 到 CPU。需要更高并发（c=4+）才能触发 Mooncake 的 batch_put/batch_get。

---

## 关键对比：enforce-eager vs -O3

| 维度 | enforce-eager (Step 5) | -O3 (Step 6) |
|------|----------------------|--------------|
| All-reduce kernel | `cross_device_reduce_1stage` | `allreduce_fusion_kernel_oneshot_lamport` |
| All-reduce 占比 | 94% | 30% |
| All-reduce 平均延迟 | 1,116 μs | 28.7 μs |
| Attention kernel | `fmhaSm100f...Persistent` | `fmhaSm100f...MultiCtas...Static` |
| Attention 占比 | 0.3% | 4.8% |
| CUDA graph | 无 | FULL_DECODE_ONLY |
| Speculative | 无 | Eagle3 |

**`-O3` + `fuse_allreduce_rms` 是性能关键**，不是 env var。Prefill 端的 `enforce-eager` 是有意为之（避免 compile 开销影响 TTFT），所以 prefill 端 93% all-reduce 是预期行为。

---

## P→D 传输延迟 Roofline 分析

### 硬件参数

| 链路 | 带宽 | 说明 |
|------|------|------|
| NVLink 5th gen (柜内 GPU-GPU) | 900 GB/s (单向) | 72 GPU 全互联，1.8 TB/s 双向 |
| 4 × 200Gb NIC (RDMA/RoCE) | **100 GB/s** (4 × 25 GB/s) | mlx5_1~4, 跨节点 PD 传输走这条路 |
| NVLink-C2C (GPU-CPU) | 450 GB/s (单向) | Mooncake CPU offload 走这条 |

### KV Cache 数据量（Kimi MLA, 70K tokens）

```
block_size           = 32 tokens
per_block_per_layer  = 32 × 576 × fp8 = 18,432 bytes
layers               = 62 (61 MLA + 1 non-MLA)
per_block_all_layers = 18,432 × 61 ≈ 1.07 MiB

70K tokens → 2,188 blocks
Total per TP rank = 2,188 × 1.07 MiB = 2.28 GiB
Total all 4 ranks = 9.13 GiB
```

### 实测 vs 理论

| | 值 | 来源 |
|---|---|---|
| **实测传输时间** | **66.7 ms** per TP rank | NVTX `nixl_read_blocks_for_req` |
| **实测 sub-transfer** | 3.1 ms, 1.2 MB, 62 descriptors | Nixl metrics |
| **实测 throughput** | 385 MB/s (per sub-transfer) | Nixl metrics |

### Roofline 对比

| 假设数据量 | NVLink (900 GB/s) | 4×200Gb NIC (100 GB/s) | 实测 | Gap vs NIC |
|-----------|-------------------|----------------------|------|-----------|
| 解读 A: 66 MiB (62 blocks) | 0.07 ms | 0.65 ms | 66.7 ms | **103x** |
| **解读 B: 2.28 GiB (2188 blocks)** | **2.5 ms** | **22.9 ms** | **66.7 ms** | **2.9x** |

**解读 B 更合理**：
- Nixl 对 70K 请求传输 ~2.28 GiB KV cache（per TP rank）
- 有效吞吐 **34.3 GB/s**，NIC 利用率 **34%**
- 和理论 100 GB/s 有 ~3x gap

### Gap 分析

2.9x gap 的可能原因：
1. **Nixl 串行化 overhead**：2,188 blocks 分 ~35 批次（每批 62 blocks），串行发起 RDMA READ，每批 3.1ms overhead
2. **小块传输效率低**：每个 sub-transfer 只有 1.2 MB，RDMA 小消息 latency dominated
3. **UCX 协议栈开销**：UCX → verbs → NIC DMA，每层有 μs 级 overhead
4. **TP 同步点**：4 个 TP rank 的传输可能不完全并行，有 barrier 等待

### Nixl 传输在 E2E 中的占比（c=1, 70K）

```
vllm-bench 结果（c=1, 70K input, 300 output, 2 prompts）:
  TTFT:  2,540 ms (mean)
  TPOT:  52.6 ms (mean)
  E2EL:  ~18s (含 300 output tokens)

Nixl 传输: 66.7 ms (NVTX)
传输占 TTFT: 66.7 / 2540 = 2.6%
```

**c=1 下 Nixl P→D 传输不是瓶颈**。TTFT 2.5s 主要是 70K tokens 的 chunked prefill（~9 chunks × 8192 tokens）。但 c=4/8 下多个请求并发传输可能改变这个比例。

### 优化空间

如果能把有效吞吐从 34 GB/s 提到 70+ GB/s（NIC 利用率 70%+）：
- 传输时间从 66.7ms 降到 ~33ms
- 对 70K 请求的 TTFT 影响：减少 ~33ms

关键手段：
- **增大 batch size**：每次传输更多 blocks，减少 per-transfer overhead
- **Pipeline 传输**：prefill 每完成一个 chunk 就开始传，不等全部完成
- **多 NIC 并行**：确认 4 张网卡是否都被利用（`--nic-metrics=hf` 可观测）

---

## 可疑发现

### 1. TP all-reduce 占 94% — 异常高

Prefill 和 decode 两端的 `cross_device_reduce_1stage` 都占了 **94%** 的 GPU 时间。这不正常。

**可能原因**:
- **这是 vLLM 的自定义 all-reduce，不走 NCCL**。在 GB200 NVL72 的 MNNVL 拓扑下，可能没有正确利用 NVLink 带宽
- **缺少 `NCCL_CUMEM_ENABLE=1` 和 `NCCL_NVLS_ENABLE=1`** — 同事的 `pd_dev_nsys.yaml` 有这两个 env var，我们没加。这两个启用了 NCCL 的 CUDA memory 和 NVLink SHARP 优化
- **缺少 `VLLM_FLASHINFER_ALLREDUCE_BACKEND: "trtllm"`** — 同事用了 trtllm allreduce backend
- **缺少 `VLLM_NVFP4_GEMM_BACKEND: "flashinfer-trtllm"`** — 同事用了 flashinfer-trtllm GEMM backend
- 也可能是 `enforce-eager` 模式下 all-reduce 没有被 fuse 到其他 kernel

**对比同事配置缺少的 env vars**:
```yaml
NCCL_CUMEM_ENABLE: 1           # 我们没加
NCCL_NVLS_ENABLE: 1            # 我们没加
VLLM_FLASHINFER_ALLREDUCE_BACKEND: "trtllm"  # 我们没加
VLLM_NVFP4_GEMM_BACKEND: "flashinfer-trtllm"  # 我们没加
PYTORCH_ALLOC_CONF: "expandable_segments:True"  # 我们没加
```

### 2. Mooncake Store 操作未触发

Step 5 的 benchmark 是 8K input × 2-4 prompts (c=1)。Mooncake 的 `batch_put`/`batch_get` 完全没触发——没有 NVTX 标注，没有 KV Transfer metrics。这说明 workload 太轻，GPU 显存足够，不需要 offload。

要验证 NVTX 标注，需要更重的 workload（70K input, 多并发）来触发 Mooncake Store 操作。

### 3. Decode trace 661 MB 过大

4 个 8K 请求的 decode trace 就有 661 MB。如果换成 70K × 8 prompts（Step 6 目标），trace 可能到 10+ GB。需要用 `delay_iterations`/`max_iterations` 精确控制 capture window。

---

## 实践经验总结

| 问题 | 原因 | 解决 |
|------|------|------|
| `--delay 120` trace 为空 | benchmark 在 delay 前就结束了 | 小模型去掉 delay，用全生命周期捕获 |
| TP>1 nsys 卡住 | TP worker 孤儿进程，nsys `--wait=all` 阻塞 | 加 `--wait=primary` |
| Kimi shutdown segfault | Mooncake/UCX C++ 析构顺序 bug，和 nsys 无关 | 用 `cudaProfilerApi` 模式，trace 在 benchmark 期间写完 |
| 同时 profile 两端 trace 为空 | scancel 32s KillWait 不够两个 nsys flush | 分别 profile；或用 `cudaProfilerApi` |
| router 不转发 `/start_profile` 给两端 | PD router 行为限制 | 分别跑 prefill-only 和 decode-only 配置 |
| `delay_iterations=30` prefill 端不触发 | PD 模式 prefill 只有几个 iteration，不到 30 | prefill 用 `delay_iterations=0`，decode 用 30 |

---

## 下一步

1. **修复 all-reduce 94% 的问题**: 加上 `NCCL_CUMEM_ENABLE=1`, `NCCL_NVLS_ENABLE=1`, `VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm` 等同事配置中的 env vars
2. **用 `delay_iterations` 方式重跑**: prefill `delay_iterations=0`, decode `delay_iterations=30, max_iterations=100`
3. **增大 workload 验证 NVTX**: 70K input, 更多 prompts, 触发 Mooncake Store put/get
4. **Step 6**: 加 NCCL CE tracing, GPU hw counters, NIC hf metrics
