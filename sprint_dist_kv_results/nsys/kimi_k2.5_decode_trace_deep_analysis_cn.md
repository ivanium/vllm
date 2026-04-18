# Deep Analysis of the Kimi K2.5 Decode Trace

> **Trace**: `vigil/logs/pd_kimi_70k_nsys_decode/2026-04-12/20260412_152636/decode-0/traces/nsys_profile.nsys-rep`
> **日期**: 2026-04-12
> **硬件**: 4× NVIDIA GB200 (GB100 chip, SM 10.0, 192GB HBM3e), NVL72 NVSwitch
> **模型**: nvidia/Kimi-K2.5-NVFP4 (61 layers, MLA + MoE, FP4 experts)
> **部署**: P/D 分离 (prefill: gb200-rack1-15, decode: gb200-rack1-16), TP=4
> **KV Transfer**: NixlConnector (RDMA/NVSwitch)
> **投机解码**: EAGLE3 (num_speculative_tokens=3)
> **CUDA Graph**: `FULL_AND_PIECEWISE`（decode worker 实际生效配置；`REPRODUCE.md` 中命令写的是 `FULL_DECODE_ONLY`，但与 worker log 不一致）
> **编译优化**: `fuse_allreduce_rms=true`, `enable_sp=false`, `fuse_gemm_comms=false`, `-O3`

## 目录

1. [Trace 总览与时序分解](#1-trace-总览与时序分解)
2. [单层 Forward Pass 的 Kernel 映射](#2-单层-forward-pass-的-kernel-映射)（含 §2.5 FP4 量化跟踪）
3. [机制一：Lamport 负零哨兵协议](#3-机制一lamport-负零哨兵协议)
4. [机制二：CUDA Graph 捕获跨 GPU 通信](#4-机制二cuda-graph-捕获跨-gpu-通信)
5. [关键配置参数](#5-关键配置参数)
6. [Trace-Derived Insights](#6-trace-derived-insights)
7. [附录：配置校正与问题定位](#7-附录配置校正与问题定位)


---

## 1. Trace 总览与时序分解

**总体统计**：GPU 活动窗口 742 ms – 6065 ms（5.32 s），1,114,464 kernel（95% 在 CUDA Graph 内），19,144 memcpy，3,104 NVTX 标注。

### 1.1 六阶段时序

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

CUDA Graph 已在初始化阶段（`gpu_worker.py:577-588` 的 `_dummy_run + kernel_warmup + capture_model`）为所有 `cudagraph_capture_sizes` 预先 capture，Phase 3/4 走 eager 是因为混合 batch 含 context 请求（`uniform_decode=False` → dispatcher 返回 `mixed_mode()=NONE`）。

| Phase | NVTX 与关键事件 |
|---|---|
| 1 (0–252 ms) | `execute_context_0(0)_generation_0(0)`：空跑，`num_scheduled_tokens=0`，走 `kv_connector_no_forward()`（`gpu_model_runner.py:3830-3846`）轮询 prefill KV |
| 2 (252–740 ms) | `nixl_read_blocks_n=62`：NixlConnector RDMA READ 拉 62 个 KV blocks（`nixl_connector.py:2509-2694`）。首批 360 ms，3 组并发补传 53–58 ms |
| 3 (740–2120 ms) | `execute_context_1(1)_generation_0(0)`：**decode 端首次 forward，num_output_tokens=0** 被标为 context request——prefill 产出的首 token 由 router 直接返回用户，不回填到 decode（`request.py:122, 185-201`）。此 batch 走 eager（61 层无 graph 优化），耗时 1378 ms 还叠加了 185 ms 首次 Lamport barrier 冷启动 + cuBLAS/cuDNN plan 重选 |
| 4 (2120–2935 ms) | 过渡期：302 kernel 来自后续几轮 eager decode + Triton JIT 编译 + EAGLE3 draft 首次运行；batch 变纯 generation 后 dispatcher 返回 `decode_mode()=FULL` |
| 5 (2935–4190 ms) | `execute_context_0(0)_generation_1(4)`（1 verified + 3 spec token）：CUDAGraphWrapper（`compilation/cuda_graph.py:341-356`）走 replay 路径，首批 ~760 ms 仍需 GPU 侧 JIT/Tensor Core/L2 预热，4190 ms 后收敛到 ~9.8 ms/step |
| 6 (4190–6065 ms) | 稳态：`graph.replay()` → 297 kernel(stream 19) + 420 kernel(stream 7109) ~8.5 ms GPU + 1.3 ms CPU，async scheduling 流水线化。第二请求 5632 ms 到达时 nixl 仅 1.4 ms（KV 已 warm）、复用已捕获 graph，`execute_context_1` 仅 41 ms |

### 1.2 稳态单步性能分解（device 0, graph 17417）

| 类别 | 时间 (μs) | 占比 | 次数 | 平均 (μs) |
|---|---:|---:|---:|---:|
| AllReduce | 1,674 | 19.8% | 120 | 13.6 |
| GEMM | 3,863 | 45.7% | 425 | 9.1 |
| Attention (fmha) | 667 | 7.9% | 61 | 10.9 |
| MoE routing+finalize | 1,039 | 12.3% | 240 | 4.3 |
| 其他 (norm/elem/kvc) | 1,212 | 14.3% | — | — |
| **主流总 GPU 时间** | **8,456** | **100%** | **—** | **—** |
| Shared expert（stream 7109，并行隐藏） | 1,656 | — | 420 | — |

Wall-clock ~9.8 ms/step（含 CPU 调度间隙）。

---

## 2. 单层 Forward Pass 的 Kernel 映射

### 2.1 代码结构

模型入口: `vllm/model_executor/models/kimi_linear.py`

```
KimiLinearForCausalLM.forward()
  └→ KimiLinearModel.forward()          ← 61 层循环
       └→ KimiDecoderLayer.forward()    ← 每层: MLA + MoE
            ├→ input_layernorm()         ← RMSNorm
            ├→ KimiMLAAttention()        ← Multi-head Latent Attention
            ├→ post_attention_layernorm() ← RMSNorm
            └→ KimiMoE()                ← Mixture of Experts (60/61 层)
                                            (第 1 层是 dense MLP)
```

### 2.2 完整单层 Kernel 序列 (device 0, stream 19)

以下是 trace 中一个 MoE 层的实测 kernel 序列 (23 kernels on main stream + 7 on aux stream):

```
┌────────────────────────────────────────────────────────────────────────────┐
│                                                                            │
│  ① input_layernorm (RMSNorm + fused residual add + 上层 allreduce)         │
│     代码: kimi_linear.py:369                                               │
│                                                                            │
│     triton_poi_fused_0           (3.0us)  ← 残差提取                       │
│     triton_poi_fused_1           (1.2us)  ← RMSNorm pointwise             │
│     allreduce_fusion_kernel_     (8.0us)  ← 上层output reduce + norm 融合  │
│       oneshot_lamport                        (fuse_allreduce_rms=true)     │
│                                                                            │
│  ② MLA Attention                                                           │
│     代码: model_executor/layers/mla.py MultiHeadLatentAttentionWrapper      │
│                                                                            │
│     fused_a_gemm_kernel          (7.5us)  ← QKV latent 压缩投影            │
│     triton_poi_fused_2           (2.6us)  ← kv_a_layernorm                │
│     triton_red_fused_3           (2.7us)  ← 归一化 reduce                  │
│     nvjet_sm100_tst_24x64_TNN   (6.1us)  ← kv_b_proj (KV latent→full KV) │
│     triton_poi_fused_add_clone_  (2.6us)  ← RoPE 位置编码                  │
│       copy_expand_index_mul_neg_                                           │
│       slice_split_stack                                                    │
│     concat_and_cache_mla_kernel  (2.7us)  ← KV Cache 写入 (paged)         │
│     nvjet_sm100_tst_64x8_NNT    (3.0us)  ← q_proj (Q latent→full Q)      │
│     triton_poi_fused__to_copy_   (1.7us)  ← Q/K scaling + concat          │
│       cat_clamp_mul_reciprocal_                                            │
│       view_0                                                               │
│     fmhaSm100fKernel_QkvE4m3O  (10.7us)  ← ★ Flash Attention SM100       │
│       Bfloat16HQk576HV512                    Paged-KV, FP8 QKV, BF16 out │
│     nvjet_sm100_tst_16x64_TNN   (3.8us)  ← o_proj (part 1)               │
│     nvjet_sm100_tst_64x8_TNT    (7.3us)  ← o_proj (part 2) + fused       │
│     allreduce_fusion_kernel      (8.0us)  ← o_proj TP AllReduce            │
│                                                                            │
│  ③ post_attention_layernorm                                                │
│     (fused into allreduce above by fuse_allreduce_rms)                     │
│                                                                            │
│  ④ MoE FFN ← 双流并行!                                                    │
│     代码: kimi_linear.py:162-177                                           │
│                                                                            │
│     ┌────────── Stream 19 (主流) ──────────┬─── Stream 7109 (辅助流) ───┐  │
│     │                                      │                           │  │
│     │ router_gemm_kernel_     (4.2us) Gate │ vectorized_elem  (2.1us)  │  │
│     │   float_output                       │ cvt_fp16_to_fp4  (2.5us)  │  │
│     │ vectorized_elementwise  (2.5us) Bias │ device_kernel    (9.0us)  │  │
│     │ cvt_fp16_to_fp4_sf_    (2.3us) Quant│   (shared gate_up GEMM)   │  │
│     │   major                              │ triton_poi_fused_(2.3us)  │  │
│     │ routingMainKernel      (4.2us) TopK  │   mul_silu_slice_0        │  │
│     │ routingIndicesCluster  (5.3us) 分发  │   (SiLU activation)       │  │
│     │                                      │ vectorized_elem  (1.2us)  │  │
│     │ bmm_E2m1_E2m1E2m1_   (22.0us) W1+W3│ cvt_fp16_to_fp4  (2.5us)  │  │
│     │   Fp32                   FP4 GEMM    │ device_kernel    (5.0us)  │  │
│     │ bmm_Bfloat16_E2m1E2m1 (13.0us) W2   │   (shared down GEMM)     │  │
│     │   _Fp32                  FP4 GEMM    │                           │  │
│     │ finalizeKernel         (3.4us) 合并  │ ← wait_stream 汇合       │  │
│     └──────────────────────────────────────┴───────────────────────────┘  │
│                                                                            │
│  ⑤ 残差相加 + TP AllReduce                                                 │
│     triton_poi_fused_add_mul_0   (1.5us)  ← shared + routed 加权求和      │
│     allreduce_fusion_kernel /    (12.0us) ← MoE output reduce             │
│       cross_device_reduce_1stage            (+ next layer RMSNorm fused)  │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

### 2.3 时序图: 主流 vs 辅助流并行

```
时间 ─────────────────────────────────────────────────────────────────────→

Stream 19:  ┃ar ┃fused_a┃norm┃kv_b┃rope┃kvc┃q_proj┃scale┃ fmha ┃o_p┃o_p┃ar ┃
(MLA部分)   ┃8us┃ 7.5us ┃5us ┃6us ┃3us ┃3us┃ 3us  ┃2us  ┃11us  ┃4us┃7us┃8us┃

Stream 19:  ┃gate┃bias┃fp4 ┃topK┃idx ┃  bmm_E2m1   ┃ bmm_Bf16  ┃fin┃add┃ allreduce ┃
(MoE部分)   ┃4us ┃3us ┃2us ┃4us ┃5us ┃    22us     ┃   13us    ┃3us┃2us┃   12us    ┃
                  ↓ wait_stream                                        ↑ wait_stream
Stream 7109:     ┃vec┃fp4┃ device_kernel ┃silu┃vec┃fp4┃devkernel┃────┘
(Shared exp)     ┃2us┃3us┃    9us        ┃2us ┃1us┃3us┃  5us    ┃
                 └── 总计 ~25us, 完全隐藏在主流 ~75us 后面 ──────┘
```

### 2.4 Kernel → 代码位置 映射表

| Kernel | 代码位置 | 模块 |
|--------|---------|------|
| `allreduce_fusion_kernel_oneshot_lamport` | FlashInfer `trtllm_allreduce_fusion.cuh` | TP 通信 |
| `fused_a_gemm_kernel` | `layers/mla.py` → fused QKV A-proj | MLA |
| `nvjet_sm100_tst_*_TNN` | CUTLASS SM100 GEMM (TNN layout) | 各种 Linear |
| `nvjet_sm100_tst_*_TNT` | CUTLASS SM100 GEMM (TNT layout) | o_proj / down_proj |
| `nvjet_sm100_tst_*_NNT` | CUTLASS SM100 GEMM (NNT layout) | q_proj |
| `fmhaSm100fKernel_*` | Flash Attention SM100 (Paged-KV) | MLA Attention |
| `concat_and_cache_mla_kernel` | `vllm/_custom_ops.py` | KV Cache |
| `device_kernel` | cuDNN fused GEMM (shared expert) | Shared MLP |
| `router_gemm_kernel_float_output` | FlashInfer MoE router | MoE Gate |
| `routingMainKernel` | FlashInfer top-K routing | MoE Dispatch |
| `routingIndicesClusterKernel` | FlashInfer token→expert 索引 | MoE Dispatch |
| `bmm_E2m1_E2m1E2m1_Fp32` | CUTLASS FP4 BMM (FP4×FP4→FP32) | MoE Expert W1+W3 |
| `bmm_Bfloat16_E2m1E2m1_Fp32` | CUTLASS FP4 BMM (BF16×FP4→FP32) | MoE Expert W2 |
| `finalizeKernel` | FlashInfer MoE combine | MoE Reduce |
| `cvt_fp16_to_fp4_sf_major` | `csrc/quantization/fp4/nvfp4_quant_kernels.cu` | 激活量化 |
| `triton_poi_fused_add_mul_0` | Triton JIT (残差相加) | 残差连接 |
| `triton_red_fused_2/3` | Triton JIT (RMSNorm reduce) | LayerNorm |
| `cross_device_reduce_1stage` | `csrc/custom_all_reduce.cuh` | vLLM 自定义 AR |

---

### 2.5 FP4 E2M1 量化跟踪：从 kernel 格式转换看两阶段 GEMM 设计

#### 2.5.1 E2M1 格式

4-bit 浮点: 1 位符号 + 2 位指数 + 1 位尾数, 无 inf/NaN:

```
bits  sign exp  man    值     计算
0000   +   00   0    +0.0    subnormal
0001   +   00   1    +0.5    subnormal: 0.5 × 2^0
0010   +   01   0    +1.0    normal: 1.0 × 2^0
0011   +   01   1    +1.5    normal: 1.5 × 2^0
0100   +   10   0    +2.0    normal: 1.0 × 2^1
0101   +   10   1    +3.0    normal: 1.5 × 2^1
0110   +   11   0    +4.0    normal: 1.0 × 2^2
0111   +   11   1    +6.0    normal: 1.5 × 2^2

可表示值 (含负数): ±{0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}
存储: 2 个 FP4 值打包到 1 个 uint8 (nibble pair)
```

代码定义: `vllm/scalar_type.py:345`:
```python
float4_e2m1f = ScalarType.float_(2, 1, True, NanRepr.NONE)
```

#### 2.5.2 Block Scale Factor 机制

8 个离散值的动态范围只有 [0, 6.0]，直接量化会崩溃。解法是 **block-wise scaling**:

```
原始数据 (BF16): [0.12, -3.7, 8.2, 0.001, ..., -15.3]  (每 16 个值一组)

Step 1: max_abs = max(|values|) = 15.3

Step 2: scale = global_scale × (max_abs / 6.0) = global_scale × 2.55

Step 3: scale → FP8 E4M3 格式 (8-bit, 省存储)

Step 4: 每个值量化:
         0.12 / scale → 0.047 → round → 0    (0000)
        -3.7 / scale → -1.45  → round → -1.5 (1011)
         8.2 / scale →  3.22  → round →  3.0 (0101)

存储:  ┌────────────────────┬──────────────┐
       │ 16 × FP4 = 8 bytes │ 1 × FP8 SF  │  ← "sf_major" 布局:
       │ (nibble-packed)     │ (E4M3 格式)  │    SF 连续存储, 数据连续存储
       └────────────────────┴──────────────┘    (对 Tensor Core MMA 更友好)
```

**block-wise（每 16 值 1 FP8 scale）** 是精度与元数据开销的折中：per-tensor 会被 outlier 拖坏大量中小值，per-element 元数据开销过大且不匹配 Tensor Core block-scaled GEMM 输入格式。vLLM 实现：`CVT_FP4_SF_VEC_SIZE=16`，`cvt_warp_fp16_to_fp4()` 对 16 值取 `max_abs` 写出 1 个 FP8 SFout。代码：`csrc/libtorch_stable/quantization/fp4/nvfp4_quant_kernels.cu`。

#### 2.5.3 两阶段 GEMM: 为什么 W1+W3 用 FP4 input 而 W2 用 BF16 input

```
                MoE Expert FFN 数据流

Hidden states (BF16)
        │
        ├──→ cvt_fp16_to_fp4_sf_major ──→ FP4 (给 routed experts)
        │                                    │
        │         ┌─────────────────────────┘
        │         ↓
        │    bmm_E2m1_E2m1E2m1_Fp32           ★ W1+W3: FP4 × FP4 → FP32
        │    Kernel含义: input=E2M1, weight=E2M1, output=FP32
        │         │
        │         ↓
        │    SiLU(gate) × up = activated       此时数据回到 BF16
        │         │
        │         ↓
        │    bmm_Bfloat16_E2m1E2m1_Fp32       ★ W2: BF16 × FP4 → FP32
        │    Kernel含义: input=BF16, weight=E2M1, output=FP32
        │         │
        │         ↓
        │    finalizeKernel → BF16 output
```

**为什么不对称量化？**

这是基于误差传播路径的精度权衡:

| 阶段 | Input 格式 | Weight 格式 | 理由 |
|------|-----------|-----------|------|
| W1+W3 (gate+up) | **FP4** | FP4 | 进入 SiLU 非线性前，FP4 精度损失对 gate 值影响有限 |
| W2 (down) | **BF16** | FP4 | SiLU 输出即将回到主残差路径，精度损失会被后续所有层放大 |

**关键**：W2 output 直接通过 `finalizeKernel` 回主残差路径，若 W2 input 也 FP4 则 SiLU 值域压缩 + FP4 只有 8 离散值，两轮量化误差叠加后被后续 60 层放大。高精度 accumulator 只能减少加法误差，**无法修复乘法输入本身的量化误差**（`W2 · Q(x) = W2·x + W2·e`，`W2·e` 在 partial product 阶段就已注入，FP32 累加保护不了它）。所以 W2 采用 BF16 input + FP4 weight 的非对称组合。

#### 2.5.4 Tensor Core FP4 指令

GB200 (SM 10.0, Blackwell) 原生支持 `mma.sync.aligned.m16n8k64.f32.e2m1.e2m1.f32`，K 维度 64 vs BF16 `mma.m16n8k16` 的 K=16，**FP4 算力 4× BF16**，这是 NVFP4 对 MoE expert 吞吐的核心价值。

#### 2.5.5 端到端数据格式转换时序

从 trace 中 MoE 层的 kernel 可精确追溯每次格式转换:

```
时间(us)  Kernel                      Input → Output         说明
──────────────────────────────────────────────────────────────────────
 0.0     router_gemm_kernel           BF16 → FP32           门控 logit
 4.2     vectorized_elementwise       FP32 → FP32           score bias
 6.7     cvt_fp16_to_fp4_sf_major     BF16 → FP4+SF(FP8)   ★ 激活量化
 9.0     routingMainKernel            FP32 → INT32          Top-K 索引
13.2     routingIndicesCluster        INT32 → INT32         分发表
18.5     bmm_E2m1_E2m1E2m1_Fp32      FP4 × FP4 → FP32     ★ W1+W3 GEMM
40.5     bmm_Bfloat16_E2m1E2m1_Fp32  BF16 × FP4 → FP32    ★ W2 GEMM
53.5     finalizeKernel               FP32 → BF16           加权合并
56.9     triton_poi_fused_add_mul_0   BF16 + BF16 → BF16   残差相加
58.4     allreduce_fusion_kernel      BF16 → BF16           TP reduce
```

注意: **量化发生在 routing 之前** (`cvt_fp16_to_fp4` 在 `routingMainKernel` 之前)。FP4 数据直接被 dispatch 到 expert，expert GEMM 直接消费 FP4 input，省掉 dispatch 后再量化的开销。

---

## 3. 机制一：Lamport 负零哨兵协议

**代码**：`csrc/custom_all_reduce.cuh`、FlashInfer `trtllm_allreduce_fusion.cuh`、编译 pass `allreduce_rms_fusion.py`。

### 3.1 为什么不用 NCCL

decode 每次 allreduce 的 payload 只有 `hidden × sizeof(bf16) = 7168 × 2 = 14 KB`。NVSwitch 跑这点数据理论上 0.016 μs 就够，但 **NCCL 一次 ~15 μs**——几乎全是 kernel launch、协议协商、barrier 开销。每 step 120 次 allreduce × 15 μs = **1.8 ms 纯通信开销**。问题是延迟，不是带宽。

### 3.2 Lamport oneshot allreduce

做法：**每个 GPU 通过 `cudaIpcOpenMemHandle` 拿到其他所有 GPU 的 peer buffer 虚拟地址，直接读对方内存，不做 barrier**。用 IEEE 754 的负零（`-0.0 == +0.0` 但 bit pattern 不同）做哨兵判数据是否就绪——接收方轮询 peer memory，只要读到非 `-0.0` 就是真实数据。

每次 allreduce 的执行（伪代码，省略 block/stride）：

```cuda
// PUSH: 写新数据，把两轮前的旧 buf 清成 -0.0 表示未就绪
data_buf[flag % 3][off] = my_data;
data_buf[(flag+2) % 3][off] = -0.0f;

// POLL + REDUCE: 原地读所有 peer 的 buffer
float sum = 0;
for (int r = 0; r < 4; r++) {
    float v;
    do { v = load_volatile(peer_data_buf[r][flag % 3][off]); }
    while (is_negative_zero(v));   // -0.0 = 未就绪，继续自旋
    sum += v;
}
output[off] = sum;
```

用**三缓冲轮换**（`flag % 3` 写 / `(flag+2) % 3` 清）防 ABA：当本 GPU 开始下一轮写 buf[0] 时，别的 GPU 可能还在读上一轮的 buf[0]，三个 buffer 保证"正在写的"和"还可能被读的"永远不是同一个。

### 3.3 与 NCCL 的差别

| | NCCL allreduce | Lamport oneshot |
|---|---|---|
| kernel launch | 调度新 kernel | 融合进已在跑的 kernel 内 |
| 拓扑协商 | ring/tree | 直接 P2P |
| 内存拷贝 | 数据拷到 NCCL buffer | 原地读 peer buffer |
| 同步机制 | collective barrier | spin on 负零 |
| 最小延迟 | ~10–15 μs | ~5–8 μs |
| 适用范围 | 任意大小 | < 512 KB (4 GPU) |

### 3.4 融合优化：allreduce + RMSNorm

编译 pass `allreduce_rms_fusion.py` 把两步合成一个 kernel：一次 global memory 往返内完成 `poll_and_reduce → rsqrt(mean²) → scale`，trace 里每层开头的 `allreduce_fusion_kernel_oneshot_lamport` 就是这个融合版本。

---

## 4. 机制二：CUDA Graph 捕获跨 GPU 通信

### 4.1 核心矛盾

CUDA Graph 是"录制一次，重放多次"——所有 kernel 参数在 capture 时固定。但 allreduce 需要：
1. 每次读到**新鲜数据**（不是 capture 时的）
2. 自旋等待其他 GPU **本次执行** 的 flag（不是 capture 时的）

### 4.2 解法：两层指针间接寻址 + IPC 事后注册

核心做法是 **固定"地址表的地址"，而不是 peer buffer 地址本身**：

1. **Capture 时**固定的只是第一层指针 `&d_rank_data_base_[N]`——它是 device memory 上一个 `RankData` 数组项（结构体字段 `ptrs[0..7]` 用于放 8 个 rank 的 peer buffer 指针）。此时 `ptrs[]` 内容未初始化。
2. **Capture 结束后**通过 `cudaIpcOpenMemHandle` 打开其他 rank 的远端显存，把真实指针回填到 `d_rank_data_base_[N].ptrs[rank]`。
3. **Replay 时** kernel 先读 `rank_data->ptrs[rank]` 再解引用 peer live buffer，天然拿到最新数据，不需要重新 capture。

Lamport flag 的跨 replay 正确性靠把 flag 计数器放到 `Signal`（IPC 共享内存），不是 graph 参数：`self_sg->_flag[blockIdx.x] += 1` 每次 replay 自增，replay N 看到 flag=N，不会与前一轮混淆。

代码位置：`csrc/custom_all_reduce.cuh`（`RankData` struct、capture 检测与 `d_rank_data_base_` 分配）、`csrc/custom_all_reduce.cu`（`register_graph_buffers()` 在 capture 结束后 `cudaIpcOpenMemHandle` + H2D memcpy 回填 `RankData`）、`custom_all_reduce.py:213-230`（Python 侧通过 `get_graph_buffer_ipc_meta` → `dist.all_gather_object` → `register_graph_buffers` 跨 rank 交换 IPC handle）。

---

## 5. 关键配置参数

### 5.1 Decode 端 vLLM 配置

来自 `pd_kimi_70k_nsys_decode.yaml`:

```yaml
# CUDA Graph
--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
--compilation_config.pass_config.fuse_allreduce_rms true  # allreduce+RMSNorm 融合
-O3                                                        # 最高编译优化等级

# MoE
VLLM_USE_FLASHINFER_MOE_FP4: "1"                 # 启用 FlashInfer FP4 MoE kernel
VLLM_FLASHINFER_MOE_BACKEND: latency             # 延迟优化 backend
VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE: "8192"       # 每 expert 最大 token 数
VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD: "8192" # 共享专家并行流阈值

# 投机解码
--speculative-config '{"model": "lightseekorg/kimi-k2.5-eagle3",
                       "method": "eagle3",
                       "num_speculative_tokens": 3}'

# Attention
--attention_config.flash_attn_version 4           # Flash Attention v4 (SM100)
--kv-cache-dtype fp8                              # KV Cache FP8 量化

# nsys 采集
--profiler-config '{"profiler":"cuda","delay_iterations":30,"max_iterations":200}'
```

### 5.2 Graph ID 与身份

| Graph ID | Kernel 数 | 内容 | 身份 |
|----------|----------|------|------|
| 17417 | 1,020,880 | 61层完整 forward (2 streams) | 主模型 decode |
| 17390 | 11,928 | allreduce + TNT GEMM + splitK | EAGLE3 验证头 |
| 17387 | 13,632 | allreduce + splitK GEMM | EAGLE3 draft model |

### 5.3 活跃 Stream

| Stream ID | 角色 | 每 step kernel 数 (dev0) |
|-----------|------|------------------------|
| 19 | 主计算流 (所有 CUDA Graph) | ~1,403 |
| 7109 | 共享专家辅助流 (aux_stream) | ~420 |
| 7119 | 共享专家辅助流 (dev 1/2/3 上的 aux_stream, 见 §7.0) | ~420 |
| async_output_copy | D2H 结果拷贝 | 少量 memcpy |

---

## 6. Trace-Derived Insights

以下发现**只能从 nsys trace 数据中观察到**，无法通过阅读代码得出。

> **nsys 陷阱**：`streamId` 是 per-process 本地编号，不是跨进程 ID。4 个 worker 的 main stream 都是 `19` 是巧合——`current_stream()`（`vllm/utils/torch_utils.py:528`）第一次创建的 dedicated stream（CUDA Graph capture 不能用 null stream）在各进程都落到本地编号 19；aux stream `7109`（dev 0） vs `7119`（dev 1/2/3）的差异来自 worker 初始化阶段前置 stream 创建顺序不同。真正唯一标识一条 CUDA stream 需要 (pid, contextId, streamId) 三元组，而不是裸 streamId。

### 6.1 AllReduce 跨 GPU 不对称: dev 2 等待时间是 dev 3 的 2 倍

**数据** (单步 4451-4462ms):
```
                allreduce 总时间   compute 总时间    allreduce 占比
  dev 0:          1,674 us          6,781 us          20%
  dev 1:          1,219 us          6,400 us          16%
  dev 2:          2,193 us          6,675 us          25%  ← 最慢
  dev 3:          1,083 us          6,661 us          14%  ← 最快
```

compute 部分几乎相同 (6400-6780us, 差异 < 6%)，但 allreduce 时间差 **2 倍**。

**首个 allreduce 的到达时序揭示了根本原因**:
```
                start time     end time      duration    等了谁?
  dev 0:        4451.815ms     4452.419ms     604us      等 dev 1,3
  dev 2:        4451.999ms     4452.420ms     420us      等 dev 1,3
  dev 1:        4452.361ms     4452.419ms      58us      等 dev 3
  dev 3:        4452.412ms     4452.419ms       7us      最后到, 不等
```

所有 GPU 在 ~4452.419ms 同时完成（end time 相差 < 1us），证明 Lamport 协议正确同步了。
但 **dev 0 比 dev 3 早到 597us**，这 597us 全部浪费在 spin-wait 上。

**原因分析**:

这不是 NVSwitch 拓扑问题（每层后续 allreduce 仅 6-10us）。根本原因是
**graph replay 之间的 CPU 调度间隙在各 GPU 上长度不同**:
- 每次 `graph.replay()` 由 CPU 发出，但 4 个 GPU 各有独立的 CUDA stream queue
- CPU 对 4 个 GPU 的 `replay()` 调用有微秒级时差
- 第一个 allreduce 吸收了这个初始偏差
- 一旦同步完成，后续 allreduce 几乎免费 (6-10us)，因为 GPU 已 lockstep

**影响**: 首个 allreduce "吸收偏差" 的代价约 200-600us/step，占 allreduce 总时间的 **30-50%**。
如果能让 4 个 GPU 的 graph replay 更同步地启动（例如用 CUDA event 同步 replay 起点），
可以将这部分开销压缩到接近零。

---

### 6.2 EAGLE3 Draft Model: 87% 时间在等 AllReduce (通信瓶颈)

**数据**:
```
                         总 GPU 时间   allreduce    allreduce 占比   非 AR kernel
主模型 (17417, 61层):     8,456 us      1,674 us       20%           6,782 us
EAGLE3 graph A (17387):    每次 ~370us   ~324 us       87%             ~46 us
EAGLE3 graph B (17390):    每次 ~189us   ~151 us       80%             ~38 us
```

EAGLE3 graph 17387 的完整 kernel profile (dev 0):
```
allreduce_fusion (324.8us avg)        ← 87% 的时间!
nvjet_sm100_tst splitK_TNT (31.2us)   ← lm_head GEMM
triton_red_fused_2 (3.7us)            ← RMSNorm reduce
triton_poi_fused_*                     ← 各种 pointwise
splitKreduce_kernel (3.1us)           ← splitK 归约
```

**这意味着什么**:

EAGLE3 draft model 只有 ~2 层 (lm_head + 投影层)，每层计算仅 ~20us。但每层仍需
TP allreduce，且 allreduce 的 latency floor (Lamport spin-wait ~8us + 首个 barrier ~300us)
不随计算量缩减。

```
Draft model 有效计算率:
  graph A: 46us / 370us = 12%   ← 88% 时间在通信!
  graph B: 38us / 189us = 20%

对比主模型:
  main:   6782us / 8456us = 80%  ← 20% 在通信
```

**结论**: EAGLE3 在 TP=4 下的投机解码是**极度通信瓶颈**的。每生成 3 个投机 token 的
开销 ~560us (graph A + B)，其中 475us 是纯 allreduce 等待。

**优化方向**:
- Draft model 用 TP=1 (不做 allreduce): 560us → ~84us, 节省 85%
- 或用更小的 draft model 跑在单 GPU 上，只在 verify 时做 TP

---

### 6.3 D2D Memcpy 揭示 EAGLE3 的 Token 接受/拒绝机制

**数据** (每步 3 组 D2D memcpy，发生在 graph replay 之间):
```
memcpy 1: 1920 KB × 4 GPU  (concurrent_kernels=0, 即 graph 之间)
memcpy 2:   56 KB × 4 GPU  = 4 tokens × 7168 × 2 bytes
memcpy 3:   14 KB × 4 GPU  = 1 token  × 7168 × 2 bytes
```

**解读**:

```
主模型 graph 17417 (verify + generate)
  │
  ├→ D2D 1920KB = 240 × 8KB: KV cache block 整理
  │   (EAGLE3 的 3 个投机 token 被 verify 后，接受的 token 的
  │    KV cache 需要从临时 slot 搬到正式 slot)
  │
  ├→ D2D 56KB = 4 tokens hidden states:
  │   (1 verified + 3 speculative 的 hidden_states 传给 EAGLE3 draft model)
  │
  ├→ D2D 14KB = 1 token hidden states:
  │   (最终确定的 1 个新 token 的 hidden state)
  │
  ↓
EAGLE3 graph 17390 → 17387 (draft 3 new speculative tokens)
```

**关键发现**: 这些 memcpy 发生在 graph replay 之间 (concurrent_kernels=0)，
是**串行的 GPU idle 时间**。虽然每次只有 ~2us，但 3 组 × 4 GPU = 24 次，加上
CPU 调度开销，这构成了步间 bubble 的一部分。

---

### 6.4 Shared Expert 的 SM 竞争: 与 bmm_E2m1 并行时反而更快

**数据** (stream 7109 device_kernel 时延 vs 主流并发 kernel):

```
主流并发 kernel                  aux device_kernel 耗时
─────────────────────────────────────────────────────
vectorized_elementwise (2us)         8-9 us  ← 慢
bmm_E2m1 (22us)                     4-5 us  ← 快!
bmm_Bfloat16 (13us)                11-12 us ← 慢
无 (gap)                             9 us   ← 基准
```

时延分布直方图:
```
 4us: ████████████ (12)         ← 与 bmm_E2m1 并行
 5us: █████████████████████ (21) ← 与 bmm_E2m1 并行
 8us: ███████████████████████████████████████████ (43) ← 与 small kernel 并行
 9us: █████████████████ (17)
11us: █████████ (9)              ← 与 bmm_Bfloat16 并行
```

**违反直觉**: 与大 kernel (bmm_E2m1, 22us) 并行时 shared expert 反而**更快** (4-5us vs 8-9us)。

**原因**: GB200 有 152 个 SM。`bmm_E2m1` 的 gridDim 可能只用了部分 SM（decode batch=1
时 expert GEMM 的矩阵很小），剩余 SM 可以充分服务 aux stream 的 `device_kernel`。
而 `vectorized_elementwise` 虽然执行快 (2us)，但它可能用了很多 SM blocks 来做
elementwise 操作（因为 elementwise kernel 通常 launch 很多 blocks 来保证延迟），
反而和 aux stream 抢 SM。

**推论**: 多流并行的效率取决于**并发 kernel 的 SM 占用率**，而不是单个 kernel 的执行时间。
小而宽的 kernel (多 block, 少计算) 反而比大而窄的 kernel (少 block, 多计算) 更影响并行效率。

---

## 7. 附录：配置校正与问题定位

> 本附录澄清本 trace 运行的实际生效配置——与 `REPRODUCE.md` 里命令中写的略有不符，影响对 §1/§3 某些现象的解读。阅读正文时若发现和预期不符，先查本附录。

这份 trace **没有启用 sequence parallelism**。因此你期待看到的 `all_reduce -> reduce_scatter + all_gather` 改写，本次运行里根本没有前置条件。

### 7.1 生效配置（以 decode worker log 为准）

`decode-0/decode-0-gb200-rack1-16.log` 中的最终 `VllmConfig` 明确显示：

```text
compilation_config.pass_config = {
  'fuse_norm_quant': False,
  'fuse_act_quant': True,
  'fuse_attn_quant': False,
  'enable_sp': False,
  'fuse_gemm_comms': False,
  'fuse_allreduce_rms': True,
}
cudagraph_mode = FULL_AND_PIECEWISE
data_parallel_size = 1
```

对应源码路径：

- `vllm/config/vllm.py`: `PostGradPassManager` 只有在 `pass_config.enable_sp=True` 时才会注册 `SequenceParallelismPass`
- `vllm/compilation/passes/pass_manager.py`: 本次实际只注册了 `AllReduceFusionPass`，没有注册 `SequenceParallelismPass`
- `vllm/config/vllm.py`: `-O3` 的默认值当前把 `enable_sp` 绑定到 `IS_DENSE`，而该常量在当前源码里被硬编码为 `False`

### 7.2 为什么是 allreduce 而不是 RS/AG

启用的是 FlashInfer **allreduce+rms fusion**（trtllm backend），不是 sequence parallelism。日志：`Enabled custom fusions: act_quant, allreduce_rms`。trace 中 `allreduce_fusion_kernel_oneshot_lamport` 主导（73,272 次，2,293 ms），完全无 `ReduceScatter`；仅有 2,272 次 `ncclDevKernel_AllGather_RING_LL`（47 ms），但不是 SP 成对出现的 AG/RS。

### 7.3 模型侧也没走 MoE sequence parallel

`KimiK25ForConditionalGeneration` 文本主干是 `DeepseekV2ForCausalLM`（`vllm/model_executor/models/kimi_k25.py`），而 `DeepseekV2MoE.is_sequence_parallel` 要求 `enable_expert_parallel=True` + `data_parallel_size>1`。本次 `data_parallel_size=1`、未启用 `--enable-expert-parallel`，所以模型侧 SP 路径也未触发。另外 `DeepseekV2MoE.forward()` 源码注释显示 "replace the all_reduce at the end of attn with reduce_scatter" 仍是 TODO，当前实现尚未做此替换。

### 7.4 对本文其余分析的影响

- `REPRODUCE.md`/`slurm.out` 中虽然同时出现了：
  - `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'`
  - `--compilation_config.pass_config.fuse_allreduce_rms true`
- 但 `vllm/utils/argparse_utils.py:parse_args()` 会把同一根键
  `--compilation-config` 的 dotted 形式重新组装成一个新的 JSON 参数，
  并在参数列表末尾追加回去；这样在 argparse 的最终结果里，后追加的
  `{"pass_config":{"fuse_allreduce_rms":true}}` 会整体覆盖前面的
  `{"cudagraph_mode":"FULL_DECODE_ONLY"}`，而不是做 deep merge。
- 这正好解释了为什么 `slurm.out` 里命令看似带了 `FULL_DECODE_ONLY`，
  但 worker 的 `non-default args` 和最终 `VllmConfig` 里它消失了。

- "`FULL_DECODE_ONLY` + mixed batch 必然 eager" 这一前提需要重审。
- decode worker 实际日志显示本次运行捕获了：
  - `Capturing CUDA graphs (mixed prefill-decode, PIECEWISE): 17`
  - `Capturing CUDA graphs (decode, FULL): 9`
- 因此 §1 的 Phase 3/4 中把 mixed batch 全部解释为 eager 的部分，和实际生效配置不完全一致。

**结论**: 这次 trace 的"异常点"不是"SP 开了但没拆成 RS/AG"，而是 **SP 根本没开，开的是 allreduce+rms 融合**；同时文档里对 cudagraph mode 的前提也写偏了。

