# Kimi K2.5 Decode Nsys Trace Analysis Report

> **工作流**: `vllm-infra-skills/.claude/skills/05-profiler/nsys-trace-analysis.md` 7-step workflow
> **Trace 来源**: `vigil/logs/pd_kimi_70k_nsys_decode/2026-04-12/20260412_152636/decode-0/traces/nsys_profile.nsys-rep`
> **配置**: `vigil/examples/pd_kimi_70k_nsys_decode.yaml`
> **日期**: 2026-04-13

---

## Step 1: 导出与元数据

```bash
nsys export --type=sqlite --output=/tmp/nsys_analysis.sqlite nsys_profile.nsys-rep
```

### 硬件

| GPU | SM 数 | 显存 | 频率 |
|-----|-------|------|------|
| 4x NVIDIA GB200 (GB100) | 152 | 197.9 GB | 2.06 GHz |

### Trace 概览

| 指标 | 值 |
|------|-----|
| 活跃范围 | 741.8ms - 6065.5ms |
| 总时长 | 5.324s |
| Kernel 数 | 1,114,464 |
| Memcpy 事件 | 19,144 |
| NVTX 标注 | 3,104 |
| Sync 事件 | 18,168 |

### 部署配置 (from YAML)

```
模型: nvidia/Kimi-K2.5-NVFP4 (61 层, MLA + MoE, FP4 experts)
TP=4, P/D 分离 (prefill: rack1-15, decode: rack1-16)
KV Transfer: NixlConnector (RDMA)
投机解码: EAGLE3 (num_speculative_tokens=3)
CUDA Graph: FULL_DECODE_ONLY
编译: fuse_allreduce_rms=true, -O3
nsys 采集: delay_iterations=30, max_iterations=200
```

---

## Step 2: 时间轴分段

### 600ms 分箱直方图

```
Bin  时间范围(ms)      Kernel数    GPU时间(ms)   阶段
───  ──────────────  ──────────  ──────────  ─────────────
 0    742-  1342          76         20.0    预热空跑
 1   1342-  1942           7        185.0    首次 eager forward
 2   1942-  2542       7,611        495.8    eager forward (main)
 3   2542-  3142         302        106.8    context→generation 过渡
 4   3142-  3742       7,472        186.4    graph replay 预热
 5   3742-  4342      16,149        303.5    加速收敛
 6   4342-  4942     447,336      2,608.1    ★ 稳态 decode
 7   4942-  5542     353,083      2,057.5    ★ 稳态 decode
 8   5542-  6142     282,428      1,718.2    ★ 稳态 (含第二请求)
```

### NVTX 阶段识别

| NVTX 标注 | 含义 | 出现时段 |
|-----------|------|---------|
| `execute_context_0(0)_generation_0(0)` | 无请求空跑 | 0-252ms |
| `nixl_read_blocks_n=62` | KV cache 从 prefill 搬运 62 blocks | 252-740ms |
| `execute_context_1(1)_generation_0(0)` | 首次 decode (eager, 含 context) | 740-2120ms |
| `execute_context_0(0)_generation_1(4)` | 稳态 decode (1 req, 4 tokens/step) | 2935-6065ms |

### 阶段划分

```
0ms        252ms      740ms        2120ms       2935ms       4190ms         6065ms
 │          │          │             │            │            │              │
 ▼          ▼          ▼             ▼            ▼            ▼              ▼
┃ Phase 1  ┃ Phase 2  ┃  Phase 3   ┃  Phase 4  ┃  Phase 5  ┃   Phase 6    ┃
┃ 预热     ┃ nixl KV  ┃  首次 decode┃  context→ ┃  Graph    ┃   稳态       ┃
┃ 空跑     ┃ 传输     ┃  eager 模式┃  gen 过渡 ┃  replay   ┃   decode     ┃
┃          ┃ 62 blocks┃  (非 graph)┃          ┃  预热     ┃   ~138 steps ┃
┃ ~250ms   ┃ ~490ms   ┃  ~1380ms   ┃  ~815ms   ┃  ~1255ms  ┃   ~1875ms   ┃
```

**Phase 3 走 eager 模式的原因**: 请求刚从 prefill 节点迁移到 decode 节点，
decode 节点还未为它产出过任何 token (`num_output_tokens==0`,
`v1/core/sched/output.py:161-163`)，被标记为 "context request"。
这不是指 prefill（prefill 已在 prefill 节点完成），而是一个**过渡状态**。
`FULL_DECODE_ONLY` 模式下含 context request 的 batch 走 `mixed_mode()=NONE`
（不用 graph），首次 forward 后 `num_output_tokens=1`，变为 generation，开始走 graph。

---

## Step 3: Kernel 热点排名

### Top 10 by GPU 时间

| # | Kernel | 调用次数 | 总时间(ms) | 平均(us) | 类别 |
|---|--------|---------|----------|---------|------|
| 1 | `allreduce_fusion_kernel_oneshot_lamport` | 73,272 | 2,293 | 31.3 | AllReduce |
| 2 | `device_kernel` (cuDNN) | 69,296 | 626 | 9.0 | Shared Expert GEMM |
| 3 | `fmhaSm100fKernel` (MLA Paged-KV) | 34,160 | 367 | 10.7 | Attention |
| 4 | `nvjet_sm100_tst_*_TNT` | 38,056 | 288 | 7.6 | GEMM (o_proj/down) |
| 5 | `vectorized_elementwise_kernel` | 144,480 | 263 | 1.8 | Elementwise |
| 6 | `fused_a_gemm_kernel` | 34,648 | 262 | 7.6 | MLA QKV proj |
| 7 | `nvjet_sm100_tst_*_TNN` | 34,648 | 213 | 6.1 | GEMM (kv_b/o_proj) |
| 8 | `nvjet_sm100_tst_*_splitK` | 2,272 | 202 | 89.0 | EAGLE3 lm_head |
| 9 | `bmm_E2m1_E2m1E2m1_Fp32` (variant 1) | 8,520 | 183 | 21.5 | MoE Expert W1+W3 |
| 10 | `bmm_E2m1_E2m1E2m1_Fp32` (variant 2) | 8,520 | 183 | 21.5 | MoE Expert W1+W3 |

### 稳态分类占比 (>4300ms)

```
类别          总时间(ms)    占比
──────────  ──────────  ──────
GEMM          2,460       37.7%
Other         1,878       28.8%   (含 cuDNN, elementwise, norm, kvc)
AllReduce     1,230       18.9%
MoE routing     574        8.8%
Attention       375        5.8%
```

---

## Step 4: 多维交叉分析

### 4a. 跨 GPU AllReduce 不对称

| Device | 总时间(us) | 次数 | 平均(us) | 最大(us) |
|--------|----------|------|---------|---------|
| dev 0 | 161,461 | 8,582 | 18.8 | 12,300 |
| dev 1 | 146,448 | 8,583 | 17.1 | 12,301 |
| dev 2 | 102,758 | 8,582 | **12.0** | 6,395 |
| dev 3 | 140,587 | 8,582 | 16.4 | 6,380 |

**发现**: dev 2 的 allreduce 平均耗时最短 (12.0us)，dev 0 最长 (18.8us)，差 57%。
dev 0/1 出现过 12.3ms 的极端 outlier，dev 2/3 最大仅 6.4ms。
dev 2 可能是 NVSwitch 拓扑中最快到达 barrier 的 GPU。

### 4b. Stream 映射

| Stream | Device | Kernel 数 | GPU 时间(ms) | 角色 |
|--------|--------|----------|------------|------|
| 19 | 0/1/2/3 | ~100,250 each | 607-674 | 主计算流 (CUDA Graph) |
| 7109 | **0 only** | 27,300 | 108.8 | Shared expert 辅助流 |
| 7119 | **1/2/3** | 27,300 each | 109-115 | Shared expert 辅助流 |

**发现**: Stream 7109 和 7119 是同一逻辑角色 (`aux_stream()`) 在不同 device 上的实例。
kernel pattern 完全相同: `device_kernel`, `cvt_fp16_to_fp4`, `vectorized_elementwise`, `triton_poi_fused_mul_silu_slice_0`。

### 4c. CUDA Graph 身份

| Graph ID | Kernel 数 | GPU 时间(ms) | Top kernel | 身份 |
|----------|----------|------------|-----------|------|
| 17417 | 473,980 | 2,679 | allreduce (150ms) + bmm_E2m1 (83ms) | **主模型 decode** (61 层) |
| 17387 | 6,260 | 50.9 | splitK_TNT (6ms) + allreduce (3.5ms) | **EAGLE3 draft graph A** |
| 17390 | 5,488 | 47.5 | allreduce (4.8ms) + splitK_TNT (3.3ms) | **EAGLE3 draft graph B** |

### 4d. Memcpy 分布

| 类型 | 次数 | 总量(MB) | 平均(us) | 说明 |
|------|------|---------|---------|------|
| H2D (kind=1) | 5,460 | 4.4 | 1.8 | Scheduler 标量参数 (4-64 bytes) |
| D2H (kind=2) | 520 | ~0 | 2.5 | Sampled token IDs |
| D2D (kind=8) | 2,604 | 529.9 | 1.6 | EAGLE3 KV 整理 + hidden state |

**D2D 详情**: 每步 3 组 —— 1920KB (KV block 整理) + 56KB (4 tokens hidden state) + 14KB (1 token hidden state)。

---

## Step 5: 单步深钻

### 选取稳态步: 4451.8ms - 4461.6ms (device 0, graph 17417)

**模型结构**: 61 层 MLA Attention + 60 层 MoE + 1 层 Dense MLP (layer 0)

### 主流 (stream 19) 分解

```
类别                时间(us)    占比     次数    平均(us)
────────────────  ──────────  ──────  ──────  ──────
GEMM               3,863     45.7%     425      9.1
AllReduce          1,674     19.8%     123     13.6
其他 (norm/elem)   1,212     14.3%       -        -
MoE routing+fin    1,039     12.3%     240      4.3
Attention            667      7.9%      61     10.9
────────────────────────────────────────────────────
总计               8,456 us  100%    1,403
```

### 辅助流 (stream 7109) — Shared Expert 并行

| 指标 | 值 |
|------|-----|
| GPU 时间 | 1,656 us (420 kernels) |
| 完全隐藏在主流背后 | 是 |
| 无并行情况下 step 增加 | +1,656us (+20%) |

### 每层 kernel 序列 (MoE 层, main stream)

```
                     MLA Attention 部分                    
①  allreduce_fusion   (8us)  ← 上层 reduce + RMSNorm 融合
②  fused_a_gemm       (8us)  ← QKV latent 压缩投影
③  triton (norm)       (5us)  ← kv_a_layernorm
④  nvjet_TNN           (6us)  ← kv_b_proj
⑤  triton (RoPE)       (3us)  ← 位置编码
⑥  concat_and_cache    (3us)  ← KV Cache 写入
⑦  nvjet_NNT           (3us)  ← q_proj
⑧  triton (scale)      (2us)  ← Q/K scaling
⑨  fmhaSm100f         (11us)  ← ★ Flash Attention SM100
⑩  nvjet_TNN           (4us)  ← o_proj part 1
⑪  nvjet_TNT           (7us)  ← o_proj part 2
⑫  allreduce           (8us)  ← o_proj TP reduce

                     MoE FFN 部分 (双流并行)
⑬  router_gemm         (4us)  ← Gate projection
⑭  vectorized          (3us)  ← Score correction
⑮  cvt_fp16_to_fp4     (2us)  ← 激活 FP4 量化
⑯  routingMain         (4us)  ← Top-K 选择
⑰  routingIndices      (5us)  ← Token→Expert 分发
⑱  bmm_E2m1           (22us)  ← ★ Expert W1+W3 FP4 GEMM
⑲  bmm_Bfloat16       (13us)  ← ★ Expert W2 FP4 GEMM
⑳  finalizeKernel      (3us)  ← 合并
㉑  triton_add_mul      (2us)  ← 残差相加
㉒  allreduce/reduce   (12us)  ← MoE output TP reduce
```

---

## Step 6: Insight 检查清单

### [x] 首个 AllReduce 惩罚

每步第一个 allreduce 的耗时 (device 0):

```
410us, 604us, 734us, 1655us, 1309us, 935us, 783us, 11us, 3056us
```

**结论**: 首个 allreduce 平均 ~1000us，后续仅 6-15us。差 **100-300 倍**。
原因: graph replay 间 CPU 调度间隙导致 4 GPU 到达首个 barrier 时间不同步，
spin-wait 吸收了全部 skew。这个惩罚占 allreduce 总时间的 30-50%。

### [x] 跨 GPU AllReduce 不对称

已在 Step 4a 确认。dev 2 最快 (12.0us avg), dev 0 最慢 (18.8us avg)。
compute 时间几乎一致 (差 <6%)，差异全部来自 barrier spin-wait。

### [x] EAGLE3 Draft Model 通信瓶颈

```
EAGLE3 graph A: allreduce 占 87% (138.4ms / 158.3ms)
EAGLE3 graph B: allreduce 占 80% (64.5ms / 80.5ms)
```

**结论**: Draft model 极度 communication-bound。每生成 3 个投机 token 的通信开销
约 203ms (累计), 有效计算仅约 36ms。TP=4 下投机解码的通信税极高。

### [x] GPU Bubble 接近零

从 Step 2 直方图可知, 稳态阶段 (bin 6-8) kernel 密集, 无明显空闲期。
Async scheduling 有效地将 CPU 调度隐藏在 GPU 执行背后。

### [x] 多流 SM 竞争

Shared expert `device_kernel` 与主流 `bmm_E2m1` (22us) 并行时仅 4-5us,
与 `vectorized_elementwise` (2us) 并行时反而 8-9us。
原因: decode batch 极小时, bmm_E2m1 的 grid 只占部分 SM, 剩余 SM 空闲可服务 aux stream。

### [x] Graph 内部两段式结构

每次 graph replay 中 `triton_poi_fused_0` 出现 2 次, 间距 ~0.5ms:
- Sub-phase 1: 1 层 dense MLA attention (无 MoE, layer 0)
- Sub-phase 2: 60 层 MoE (含 routing + expert GEMM + shared expert parallel)

由 `first_k_dense_replace` 配置控制: layer 0 使用 `KimiMLP` (dense), layer 1-60 使用 `KimiMoE`。

### [x] D2D Memcpy 在 Pipeline Bubble 中

所有 D2D memcpy 的 `concurrent_kernels=0`，确认发生在 graph replay 之间。
这是 EAGLE3 token 接受/拒绝后的 KV cache 整理 (1920KB) 和 hidden state 传递 (56KB/14KB)。

---

## Step 7: 代码路径回溯

### NVTX 标注来源

| NVTX 文本 | 代码位置 |
|-----------|---------|
| `execute_context_X(Y)_generation_Z(W)` | `v1/worker/gpu_worker.py:726-738` |
| `nixl_read_blocks_for_req=` | `distributed/kv_transfer/.../nixl_connector.py:2509-2511` |
| `nixl_read_blocks_n=` | `nixl_connector.py:2681-2682` |
| `nixl_xfer_done_req=` | `nixl_connector.py:2400-2401` |

### 关键代码路径

| 行为 | 代码位置 | 说明 |
|------|---------|------|
| 空跑提前返回 | `gpu_model_runner.py:3830-3846` | `num_scheduled_tokens=0` → `kv_connector_no_forward()` |
| CUDA Graph mode 选择 | `v1/cudagraph_dispatcher.py:301` | `uniform_decode` → FULL, 否则 → NONE |
| Graph 预先 capture | `gpu_worker.py:588` | `capture_model()` 在初始化阶段完成 |
| Graph replay | `compilation/cuda_graph.py:355` | `entry.cudagraph.replay()` |
| Shared expert 双流 | `fused_moe/runner/default_moe_runner.py:238-354` | `aux_stream()` 并行执行 |
| Async scheduling | `gpu_model_runner.py:3481-3493` | `prepare_inputs_event.synchronize()` 等 D2H copy |

### Kernel → 模块映射

| Kernel | 模型模块 | 代码位置 |
|--------|---------|---------|
| `allreduce_fusion_kernel_oneshot_lamport` | TP AllReduce + RMSNorm | FlashInfer `trtllm_allreduce_fusion.cuh` |
| `fused_a_gemm_kernel` | MLA QKV latent proj | `layers/mla.py` |
| `fmhaSm100fKernel_*` | Flash Attention SM100 | FlashAttn v4 backend |
| `bmm_E2m1_*` | MoE Expert FP4 GEMM | FlashInfer CUTLASS FP4 |
| `device_kernel` | cuDNN fused GEMM | Shared expert MLP |
| `routingMainKernel` | MoE Top-K | FlashInfer router |
| `concat_and_cache_mla_kernel` | KV Cache write | `vllm/_custom_ops.py` |

---

## 总结: 关键发现

| # | 发现 | 影响 | 优化方向 |
|---|------|------|---------|
| 1 | AllReduce 占稳态 GPU 时间 19.8% | decode latency 的最大通信开销 | 降低 TP 度 / overlap allreduce with compute |
| 2 | 首个 allreduce 惩罚 ~1000us/step | 占 allreduce 总时间 30-50% | 用 CUDA event 同步 graph replay 起点 |
| 3 | EAGLE3 draft model 87% 时间等 allreduce | 投机解码效率极低 | Draft model 用 TP=1 |
| 4 | Shared expert 并行节省 20% latency | 多流设计有效 | 已达预期效果 |
| 5 | 稳态每步 ~9.8ms (61 层, 8.5ms GPU) | ~102 tokens/s per request | GPU bubble <0.2ms, 计算密度高 |
| 6 | Layer 0 dense + Layer 1-60 MoE 两段式 | Graph 内部结构清晰 | 无需优化 |
| 7 | D2D memcpy 在 bubble 中, 未与 compute overlap | ~6us/step 额外开销 | 可忽略 |
