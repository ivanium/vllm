# Deep Analysis of the Kimi K2.5 Decode Trace

> **Trace**: `vigil/logs/pd_kimi_70k_nsys_decode/2026-04-12/20260412_152636/decode-0/traces/nsys_profile.nsys-rep`
> **Date**: 2026-04-12
> **Hardware**: 4× NVIDIA GB200 (GB100 chip, SM 10.0, 192GB HBM3e), NVL72 NVSwitch
> **Model**: nvidia/Kimi-K2.5-NVFP4 (61 layers, MLA + MoE, FP4 experts)
> **Deployment**: P/D disaggregation (prefill: gb200-rack1-15, decode: gb200-rack1-16), TP=4
> **KV Transfer**: NixlConnector (RDMA/NVSwitch)
> **Speculative Decoding**: EAGLE3 (num_speculative_tokens=3)
> **CUDA Graph**: `FULL_AND_PIECEWISE` (the config actually in effect on the decode worker; the command in `REPRODUCE.md` says `FULL_DECODE_ONLY`, but that is inconsistent with the worker log)
> **Compilation**: `fuse_allreduce_rms=true`, `enable_sp=false`, `fuse_gemm_comms=false`, `-O3`

## Table of Contents

1. [Trace Overview and Phase Breakdown](#1-trace-overview-and-phase-breakdown)
2. [Kernel Mapping of a Single Forward Pass](#2-kernel-mapping-of-a-single-forward-pass) (includes §2.5 FP4 quantization trace)
3. [Mechanism 1: Lamport Negative-Zero Sentinel Protocol](#3-mechanism-1-lamport-negative-zero-sentinel-protocol)
4. [Mechanism 2: CUDA Graph Capture of Cross-GPU Communication](#4-mechanism-2-cuda-graph-capture-of-cross-gpu-communication)
5. [Key Configuration Parameters](#5-key-configuration-parameters)
6. [Trace-Derived Insights](#6-trace-derived-insights)
7. [Appendix: Configuration Correction and Issue Localization](#7-appendix-configuration-correction-and-issue-localization)


---

## 1. Trace Overview and Phase Breakdown

**Overall statistics**: GPU activity window 742 ms – 6065 ms (5.32 s), 1,114,464 kernels (95% inside CUDA Graph), 19,144 memcpys, 3,104 NVTX annotations.

### 1.1 Six-Phase Timeline

```
0ms        252ms      740ms        2120ms       2935ms       4190ms         6065ms
 │          │          │             │            │            │              │
 ▼          ▼          ▼             ▼            ▼            ▼              ▼
┃ Phase 1  ┃ Phase 2  ┃  Phase 3   ┃  Phase 4  ┃  Phase 5  ┃   Phase 6    ┃
┃ warmup   ┃ nixl KV  ┃  first     ┃  context→ ┃  Graph    ┃   steady-    ┃
┃ idle     ┃ transfer ┃  decode,   ┃  gen      ┃  replay   ┃   state      ┃
┃          ┃ 62 blocks┃  eager mode┃  transition┃  warmup  ┃   decode     ┃
┃          ┃          ┃  (no graph)┃           ┃           ┃  ~138 steps  ┃
┃ ~250ms   ┃ ~490ms   ┃  ~1380ms   ┃  ~815ms   ┃  ~1255ms  ┃   ~1875ms   ┃
```

CUDA Graphs are pre-captured for all `cudagraph_capture_sizes` during initialization (`gpu_worker.py:577-588`, `_dummy_run + kernel_warmup + capture_model`). Phases 3/4 fall back to eager because the mixed batch contains a context request (`uniform_decode=False` → dispatcher returns `mixed_mode()=NONE`).

| Phase | NVTX and key events |
|---|---|
| 1 (0–252 ms) | `execute_context_0(0)_generation_0(0)`: idle run, `num_scheduled_tokens=0`, goes through `kv_connector_no_forward()` (`gpu_model_runner.py:3830-3846`) polling prefill KV |
| 2 (252–740 ms) | `nixl_read_blocks_n=62`: NixlConnector RDMA READ pulls 62 KV blocks (`nixl_connector.py:2509-2694`). First batch 360 ms, 3 concurrent supplementary transfers of 53–58 ms |
| 3 (740–2120 ms) | `execute_context_1(1)_generation_0(0)`: **first forward on the decode side, num_output_tokens=0** is marked as a context request — the first token produced by prefill is returned to the user directly by the router and is not backfilled to decode (`request.py:122, 185-201`). This batch runs eager (no graph optimization for 61 layers), costing 1378 ms plus a 185 ms cold-start of the first Lamport barrier and cuBLAS/cuDNN plan re-selection |
| 4 (2120–2935 ms) | Transition: 302 kernels from several more eager decodes + Triton JIT compilation + first run of EAGLE3 draft; once the batch becomes pure generation, dispatcher returns `decode_mode()=FULL` |
| 5 (2935–4190 ms) | `execute_context_0(0)_generation_1(4)` (1 verified + 3 spec tokens): CUDAGraphWrapper (`compilation/cuda_graph.py:341-356`) takes the replay path, first batch ~760 ms still needs GPU-side JIT/Tensor Core/L2 warmup, converging to ~9.8 ms/step after 4190 ms |
| 6 (4190–6065 ms) | Steady state: `graph.replay()` → 297 kernels (stream 19) + 420 kernels (stream 7109) ~8.5 ms GPU + 1.3 ms CPU, async scheduling pipelined. When the second request arrives at 5632 ms, nixl only takes 1.4 ms (KV is warm), the captured graph is reused, and `execute_context_1` only costs 41 ms |

### 1.2 Steady-State Per-Step Performance Breakdown (device 0, graph 17417)

| Category | Time (μs) | Share | Count | Avg (μs) |
|---|---:|---:|---:|---:|
| AllReduce | 1,674 | 19.8% | 120 | 13.6 |
| GEMM | 3,863 | 45.7% | 425 | 9.1 |
| Attention (fmha) | 667 | 7.9% | 61 | 10.9 |
| MoE routing+finalize | 1,039 | 12.3% | 240 | 4.3 |
| Other (norm/elem/kvc) | 1,212 | 14.3% | — | — |
| **Main-stream total GPU time** | **8,456** | **100%** | **—** | **—** |
| Shared expert (stream 7109, hidden in parallel) | 1,656 | — | 420 | — |

Wall-clock ~9.8 ms/step (including CPU scheduling gaps).

---

## 2. Kernel Mapping of a Single Forward Pass

### 2.1 Code Structure

Model entry: `vllm/model_executor/models/kimi_linear.py`

```
KimiLinearForCausalLM.forward()
  └→ KimiLinearModel.forward()          ← 61-layer loop
       └→ KimiDecoderLayer.forward()    ← per layer: MLA + MoE
            ├→ input_layernorm()         ← RMSNorm
            ├→ KimiMLAAttention()        ← Multi-head Latent Attention
            ├→ post_attention_layernorm() ← RMSNorm
            └→ KimiMoE()                ← Mixture of Experts (60 of 61 layers)
                                            (layer 1 is dense MLP)
```

### 2.2 Full Per-Layer Kernel Sequence (device 0, stream 19)

Below is an observed kernel sequence for one MoE layer in the trace (23 kernels on main stream + 7 on aux stream):

```
┌────────────────────────────────────────────────────────────────────────────┐
│                                                                            │
│  ① input_layernorm (RMSNorm + fused residual add + previous-layer AR)      │
│     code: kimi_linear.py:369                                               │
│                                                                            │
│     triton_poi_fused_0           (3.0us)  ← residual extraction            │
│     triton_poi_fused_1           (1.2us)  ← RMSNorm pointwise              │
│     allreduce_fusion_kernel_     (8.0us)  ← prev-layer output reduce +     │
│       oneshot_lamport                        norm fusion                   │
│                                              (fuse_allreduce_rms=true)     │
│                                                                            │
│  ② MLA Attention                                                           │
│     code: model_executor/layers/mla.py MultiHeadLatentAttentionWrapper     │
│                                                                            │
│     fused_a_gemm_kernel          (7.5us)  ← QKV latent compression proj    │
│     triton_poi_fused_2           (2.6us)  ← kv_a_layernorm                 │
│     triton_red_fused_3           (2.7us)  ← normalization reduce           │
│     nvjet_sm100_tst_24x64_TNN   (6.1us)  ← kv_b_proj (KV latent→full KV)  │
│     triton_poi_fused_add_clone_  (2.6us)  ← RoPE position encoding         │
│       copy_expand_index_mul_neg_                                           │
│       slice_split_stack                                                    │
│     concat_and_cache_mla_kernel  (2.7us)  ← KV Cache write (paged)         │
│     nvjet_sm100_tst_64x8_NNT    (3.0us)  ← q_proj (Q latent→full Q)       │
│     triton_poi_fused__to_copy_   (1.7us)  ← Q/K scaling + concat           │
│       cat_clamp_mul_reciprocal_                                            │
│       view_0                                                               │
│     fmhaSm100fKernel_QkvE4m3O  (10.7us)  ← ★ Flash Attention SM100         │
│       Bfloat16HQk576HV512                    Paged-KV, FP8 QKV, BF16 out   │
│     nvjet_sm100_tst_16x64_TNN   (3.8us)  ← o_proj (part 1)                │
│     nvjet_sm100_tst_64x8_TNT    (7.3us)  ← o_proj (part 2) + fused        │
│     allreduce_fusion_kernel      (8.0us)  ← o_proj TP AllReduce            │
│                                                                            │
│  ③ post_attention_layernorm                                                │
│     (fused into the allreduce above by fuse_allreduce_rms)                 │
│                                                                            │
│  ④ MoE FFN ← dual-stream parallelism!                                      │
│     code: kimi_linear.py:162-177                                           │
│                                                                            │
│     ┌────────── Stream 19 (main) ─────────┬─── Stream 7109 (aux) ──────┐   │
│     │                                      │                           │   │
│     │ router_gemm_kernel_     (4.2us) Gate │ vectorized_elem  (2.1us)  │   │
│     │   float_output                       │ cvt_fp16_to_fp4  (2.5us)  │   │
│     │ vectorized_elementwise  (2.5us) Bias │ device_kernel    (9.0us)  │   │
│     │ cvt_fp16_to_fp4_sf_    (2.3us) Quant│   (shared gate_up GEMM)   │   │
│     │   major                              │ triton_poi_fused_(2.3us)  │   │
│     │ routingMainKernel      (4.2us) TopK  │   mul_silu_slice_0        │   │
│     │ routingIndicesCluster  (5.3us) dispat│   (SiLU activation)       │   │
│     │                                      │ vectorized_elem  (1.2us)  │   │
│     │ bmm_E2m1_E2m1E2m1_   (22.0us) W1+W3│ cvt_fp16_to_fp4  (2.5us)  │   │
│     │   Fp32                   FP4 GEMM    │ device_kernel    (5.0us)  │   │
│     │ bmm_Bfloat16_E2m1E2m1 (13.0us) W2   │   (shared down GEMM)     │   │
│     │   _Fp32                  FP4 GEMM    │                           │   │
│     │ finalizeKernel         (3.4us) combi│ ← wait_stream join        │   │
│     └──────────────────────────────────────┴───────────────────────────┘   │
│                                                                            │
│  ⑤ Residual add + TP AllReduce                                             │
│     triton_poi_fused_add_mul_0   (1.5us)  ← weighted sum shared + routed   │
│     allreduce_fusion_kernel /    (12.0us) ← MoE output reduce              │
│       cross_device_reduce_1stage            (+ next layer RMSNorm fused)   │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

### 2.3 Timing Diagram: Main Stream vs Aux Stream Parallelism

```
time ─────────────────────────────────────────────────────────────────────→

Stream 19:  ┃ar ┃fused_a┃norm┃kv_b┃rope┃kvc┃q_proj┃scale┃ fmha ┃o_p┃o_p┃ar ┃
(MLA part)  ┃8us┃ 7.5us ┃5us ┃6us ┃3us ┃3us┃ 3us  ┃2us  ┃11us  ┃4us┃7us┃8us┃

Stream 19:  ┃gate┃bias┃fp4 ┃topK┃idx ┃  bmm_E2m1   ┃ bmm_Bf16  ┃fin┃add┃ allreduce ┃
(MoE part)  ┃4us ┃3us ┃2us ┃4us ┃5us ┃    22us     ┃   13us    ┃3us┃2us┃   12us    ┃
                  ↓ wait_stream                                        ↑ wait_stream
Stream 7109:     ┃vec┃fp4┃ device_kernel ┃silu┃vec┃fp4┃devkernel┃────┘
(Shared exp)     ┃2us┃3us┃    9us        ┃2us ┃1us┃3us┃  5us    ┃
                 └── total ~25us, fully hidden behind main-stream ~75us ──┘
```

### 2.4 Kernel → Code Location Mapping Table

| Kernel | Code location | Module |
|--------|---------|------|
| `allreduce_fusion_kernel_oneshot_lamport` | FlashInfer `trtllm_allreduce_fusion.cuh` | TP communication |
| `fused_a_gemm_kernel` | `layers/mla.py` → fused QKV A-proj | MLA |
| `nvjet_sm100_tst_*_TNN` | CUTLASS SM100 GEMM (TNN layout) | various Linear |
| `nvjet_sm100_tst_*_TNT` | CUTLASS SM100 GEMM (TNT layout) | o_proj / down_proj |
| `nvjet_sm100_tst_*_NNT` | CUTLASS SM100 GEMM (NNT layout) | q_proj |
| `fmhaSm100fKernel_*` | Flash Attention SM100 (Paged-KV) | MLA Attention |
| `concat_and_cache_mla_kernel` | `vllm/_custom_ops.py` | KV Cache |
| `device_kernel` | cuDNN fused GEMM (shared expert) | Shared MLP |
| `router_gemm_kernel_float_output` | FlashInfer MoE router | MoE Gate |
| `routingMainKernel` | FlashInfer top-K routing | MoE Dispatch |
| `routingIndicesClusterKernel` | FlashInfer token→expert indexing | MoE Dispatch |
| `bmm_E2m1_E2m1E2m1_Fp32` | CUTLASS FP4 BMM (FP4×FP4→FP32) | MoE Expert W1+W3 |
| `bmm_Bfloat16_E2m1E2m1_Fp32` | CUTLASS FP4 BMM (BF16×FP4→FP32) | MoE Expert W2 |
| `finalizeKernel` | FlashInfer MoE combine | MoE Reduce |
| `cvt_fp16_to_fp4_sf_major` | `csrc/quantization/fp4/nvfp4_quant_kernels.cu` | activation quantization |
| `triton_poi_fused_add_mul_0` | Triton JIT (residual add) | residual connection |
| `triton_red_fused_2/3` | Triton JIT (RMSNorm reduce) | LayerNorm |
| `cross_device_reduce_1stage` | `csrc/custom_all_reduce.cuh` | vLLM custom AR |

---

### 2.5 FP4 E2M1 Quantization Trace: Two-Stage GEMM Design Viewed from Kernel Format Conversions

#### 2.5.1 The E2M1 Format

4-bit float: 1 sign bit + 2 exponent bits + 1 mantissa bit, no inf/NaN:

```
bits  sign exp  man    value     computation
0000   +   00   0    +0.0    subnormal
0001   +   00   1    +0.5    subnormal: 0.5 × 2^0
0010   +   01   0    +1.0    normal: 1.0 × 2^0
0011   +   01   1    +1.5    normal: 1.5 × 2^0
0100   +   10   0    +2.0    normal: 1.0 × 2^1
0101   +   10   1    +3.0    normal: 1.5 × 2^1
0110   +   11   0    +4.0    normal: 1.0 × 2^2
0111   +   11   1    +6.0    normal: 1.5 × 2^2

Representable values (including negatives): ±{0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}
Storage: 2 FP4 values packed into 1 uint8 (nibble pair)
```

Code definition: `vllm/scalar_type.py:345`:
```python
float4_e2m1f = ScalarType.float_(2, 1, True, NanRepr.NONE)
```

#### 2.5.2 Block Scale Factor Mechanism

The dynamic range of 8 discrete values is only [0, 6.0], so direct quantization collapses. The fix is **block-wise scaling**:

```
Raw data (BF16): [0.12, -3.7, 8.2, 0.001, ..., -15.3]  (group of 16 values)

Step 1: max_abs = max(|values|) = 15.3

Step 2: scale = global_scale × (max_abs / 6.0) = global_scale × 2.55

Step 3: scale → FP8 E4M3 format (8-bit, saves storage)

Step 4: quantize each value:
         0.12 / scale → 0.047 → round → 0    (0000)
        -3.7 / scale → -1.45  → round → -1.5 (1011)
         8.2 / scale →  3.22  → round →  3.0 (0101)

Storage: ┌────────────────────┬──────────────┐
         │ 16 × FP4 = 8 bytes │ 1 × FP8 SF   │  ← "sf_major" layout:
         │ (nibble-packed)    │ (E4M3 format)│    SF contiguous, data contiguous
         └────────────────────┴──────────────┘    (Tensor-Core-MMA friendly)
```

**Block-wise (1 FP8 scale per 16 values)** is a tradeoff between precision and metadata overhead: per-tensor is wrecked by outliers that trash many small/medium values; per-element has too much metadata overhead and does not match the Tensor Core block-scaled GEMM input format. vLLM implementation: `CVT_FP4_SF_VEC_SIZE=16`; `cvt_warp_fp16_to_fp4()` takes `max_abs` over 16 values and writes one FP8 SFout. Code: `csrc/libtorch_stable/quantization/fp4/nvfp4_quant_kernels.cu`.

#### 2.5.3 Two-Stage GEMM: Why W1+W3 Use FP4 Input but W2 Uses BF16 Input

```
                MoE Expert FFN Data Flow

Hidden states (BF16)
        │
        ├──→ cvt_fp16_to_fp4_sf_major ──→ FP4 (feeds routed experts)
        │                                    │
        │         ┌─────────────────────────┘
        │         ↓
        │    bmm_E2m1_E2m1E2m1_Fp32           ★ W1+W3: FP4 × FP4 → FP32
        │    Kernel meaning: input=E2M1, weight=E2M1, output=FP32
        │         │
        │         ↓
        │    SiLU(gate) × up = activated       data returns to BF16 here
        │         │
        │         ↓
        │    bmm_Bfloat16_E2m1E2m1_Fp32       ★ W2: BF16 × FP4 → FP32
        │    Kernel meaning: input=BF16, weight=E2M1, output=FP32
        │         │
        │         ↓
        │    finalizeKernel → BF16 output
```

**Why asymmetric quantization?**

This is a precision tradeoff based on the error-propagation path:

| Stage | Input format | Weight format | Rationale |
|------|-----------|-----------|------|
| W1+W3 (gate+up) | **FP4** | FP4 | Before entering the SiLU nonlinearity, FP4 precision loss has limited impact on gate values |
| W2 (down) | **BF16** | FP4 | SiLU output is about to return to the main residual path; precision loss is amplified by all subsequent layers |

**Key point**: W2 output flows directly through `finalizeKernel` back to the main residual path. If W2 input were also FP4, the SiLU value range compression + only 8 discrete FP4 values means the two rounds of quantization error compound and then get amplified by the following 60 layers. A high-precision accumulator can only reduce addition error; it **cannot fix the multiplication input quantization error itself** (`W2 · Q(x) = W2·x + W2·e`; `W2·e` is already injected at the partial product stage, and FP32 accumulation cannot protect against it). Hence W2 uses the asymmetric combination of BF16 input + FP4 weight.

#### 2.5.4 Tensor Core FP4 Instruction

GB200 (SM 10.0, Blackwell) natively supports `mma.sync.aligned.m16n8k64.f32.e2m1.e2m1.f32`; the K dimension of 64 vs BF16 `mma.m16n8k16`'s K=16 means **FP4 compute throughput is 4× that of BF16**, which is the core value of NVFP4 for MoE expert throughput.

#### 2.5.5 End-to-End Data Format Conversion Timeline

From the MoE-layer kernels in the trace we can exactly trace each format conversion:

```
time(us) Kernel                      Input → Output         Note
──────────────────────────────────────────────────────────────────────
 0.0     router_gemm_kernel           BF16 → FP32           gate logit
 4.2     vectorized_elementwise       FP32 → FP32           score bias
 6.7     cvt_fp16_to_fp4_sf_major     BF16 → FP4+SF(FP8)   ★ activation quant
 9.0     routingMainKernel            FP32 → INT32          Top-K indices
13.2     routingIndicesCluster        INT32 → INT32         dispatch table
18.5     bmm_E2m1_E2m1E2m1_Fp32      FP4 × FP4 → FP32     ★ W1+W3 GEMM
40.5     bmm_Bfloat16_E2m1E2m1_Fp32  BF16 × FP4 → FP32    ★ W2 GEMM
53.5     finalizeKernel               FP32 → BF16           weighted combine
56.9     triton_poi_fused_add_mul_0   BF16 + BF16 → BF16   residual add
58.4     allreduce_fusion_kernel      BF16 → BF16           TP reduce
```

Note: **quantization happens before routing** (`cvt_fp16_to_fp4` comes before `routingMainKernel`). FP4 data is dispatched directly to experts, and expert GEMMs consume FP4 input directly, eliminating the cost of quantizing again after dispatch.

---

## 3. Mechanism 1: Lamport Negative-Zero Sentinel Protocol

**Code**: `csrc/custom_all_reduce.cuh`, FlashInfer `trtllm_allreduce_fusion.cuh`, compile pass `allreduce_rms_fusion.py`.

### 3.1 Why Not NCCL

Each decode allreduce has a payload of only `hidden × sizeof(bf16) = 7168 × 2 = 14 KB`. Theoretically NVSwitch needs only 0.016 μs for that amount of data, but **NCCL takes ~15 μs per call** — almost all of it is kernel launch, protocol negotiation, and barrier overhead. 120 allreduces per step × 15 μs = **1.8 ms of pure communication overhead**. The problem is latency, not bandwidth.

### 3.2 Lamport Oneshot AllReduce

Approach: **each GPU obtains the peer buffer virtual addresses of every other GPU via `cudaIpcOpenMemHandle`, and directly reads the peer's memory without a barrier**. IEEE 754 negative zero (`-0.0 == +0.0` but with a different bit pattern) is used as a sentinel indicating whether data is ready — the receiver polls peer memory and as long as it reads a value that is not `-0.0`, it is real data.

Each allreduce's execution (pseudo-code, block/stride omitted):

```cuda
// PUSH: write new data, clear the buffer from two rounds ago to -0.0 to mark not ready
data_buf[flag % 3][off] = my_data;
data_buf[(flag+2) % 3][off] = -0.0f;

// POLL + REDUCE: read all peers' buffers in place
float sum = 0;
for (int r = 0; r < 4; r++) {
    float v;
    do { v = load_volatile(peer_data_buf[r][flag % 3][off]); }
    while (is_negative_zero(v));   // -0.0 = not ready, keep spinning
    sum += v;
}
output[off] = sum;
```

It uses **triple-buffer rotation** (`flag % 3` write / `(flag+2) % 3` clear) to prevent ABA: when this GPU begins the next round of writes to buf[0], other GPUs may still be reading the prior round's buf[0]; three buffers guarantee that "the one being written" and "the one that might still be read" are never the same.

### 3.3 Differences from NCCL

| | NCCL allreduce | Lamport oneshot |
|---|---|---|
| kernel launch | schedules a new kernel | fused into a kernel already running |
| topology negotiation | ring/tree | direct P2P |
| memory copies | data copied to NCCL buffer | in-place read of peer buffer |
| synchronization | collective barrier | spin on negative zero |
| minimum latency | ~10–15 μs | ~5–8 μs |
| applicable size range | any size | < 512 KB (4 GPU) |

### 3.4 Fusion Optimization: AllReduce + RMSNorm

The compile pass `allreduce_rms_fusion.py` merges the two steps into one kernel: within a single global memory round-trip it completes `poll_and_reduce → rsqrt(mean²) → scale`. The `allreduce_fusion_kernel_oneshot_lamport` at the start of each layer in the trace is this fused version.

---

## 4. Mechanism 2: CUDA Graph Capture of Cross-GPU Communication

### 4.1 The Core Contradiction

CUDA Graph is "record once, replay many times" — all kernel parameters are fixed at capture time. But allreduce requires:
1. Reading **fresh data** each time (not what was present at capture)
2. Spinning on the flag of **this execution** of the other GPUs (not the one at capture)

### 4.2 Solution: Two-Level Pointer Indirection + Post-Capture IPC Registration

The core idea is to **fix "the address of the address table", not the peer buffer address itself**:

1. **At capture time**, what is fixed is only the first-level pointer `&d_rank_data_base_[N]` — it points to a `RankData` array slot in device memory (the struct field `ptrs[0..7]` holds the peer buffer pointers for 8 ranks). At this point the contents of `ptrs[]` are uninitialized.
2. **After capture**, `cudaIpcOpenMemHandle` opens the remote memory of other ranks, and the real pointers are written back into `d_rank_data_base_[N].ptrs[rank]`.
3. **At replay time**, the kernel first reads `rank_data->ptrs[rank]` and then dereferences the peer's live buffer, naturally getting the latest data, without re-capture.

The cross-replay correctness of the Lamport flag relies on placing the flag counter into `Signal` (IPC shared memory), not into a graph parameter: `self_sg->_flag[blockIdx.x] += 1` increments on every replay; replay N sees flag=N, and does not get confused with the previous round.

Code locations: `csrc/custom_all_reduce.cuh` (`RankData` struct, capture detection, and `d_rank_data_base_` allocation); `csrc/custom_all_reduce.cu` (`register_graph_buffers()` does `cudaIpcOpenMemHandle` + H2D memcpy to backfill `RankData` after capture ends); `custom_all_reduce.py:213-230` (Python side exchanges IPC handles across ranks via `get_graph_buffer_ipc_meta` → `dist.all_gather_object` → `register_graph_buffers`).

---

## 5. Key Configuration Parameters

### 5.1 Decode-Side vLLM Configuration

From `pd_kimi_70k_nsys_decode.yaml`:

```yaml
# CUDA Graph
--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
--compilation_config.pass_config.fuse_allreduce_rms true  # allreduce+RMSNorm fusion
-O3                                                        # highest compile optimization level

# MoE
VLLM_USE_FLASHINFER_MOE_FP4: "1"                 # enable FlashInfer FP4 MoE kernel
VLLM_FLASHINFER_MOE_BACKEND: latency             # latency-optimized backend
VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE: "8192"       # max tokens per expert
VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD: "8192" # shared-expert parallel-stream threshold

# Speculative decoding
--speculative-config '{"model": "lightseekorg/kimi-k2.5-eagle3",
                       "method": "eagle3",
                       "num_speculative_tokens": 3}'

# Attention
--attention_config.flash_attn_version 4           # Flash Attention v4 (SM100)
--kv-cache-dtype fp8                              # KV Cache FP8 quantization

# nsys capture
--profiler-config '{"profiler":"cuda","delay_iterations":30,"max_iterations":200}'
```

### 5.2 Graph IDs and Identity

| Graph ID | Kernel count | Content | Identity |
|----------|----------|------|------|
| 17417 | 1,020,880 | 61-layer full forward (2 streams) | main model decode |
| 17390 | 11,928 | allreduce + TNT GEMM + splitK | EAGLE3 verification head |
| 17387 | 13,632 | allreduce + splitK GEMM | EAGLE3 draft model |

### 5.3 Active Streams

| Stream ID | Role | Kernels/step (dev0) |
|-----------|------|------------------------|
| 19 | Main compute stream (all CUDA Graphs) | ~1,403 |
| 7109 | Shared-expert aux stream (aux_stream) | ~420 |
| 7119 | Shared-expert aux stream (aux_stream on dev 1/2/3, see §7.0) | ~420 |
| async_output_copy | D2H result copy | few memcpys |

---

## 6. Trace-Derived Insights

The findings below are **only observable from the nsys trace data**; they cannot be derived by reading the code.

> **nsys pitfall**: `streamId` is a per-process local number, not a cross-process ID. The fact that all 4 workers' main streams are `19` is a coincidence — the first dedicated stream created by `current_stream()` (`vllm/utils/torch_utils.py:528`) (CUDA Graph capture cannot use the null stream) happens to land on local number 19 in each process; the difference between aux streams `7109` (dev 0) vs `7119` (dev 1/2/3) stems from different prefix-stream creation orders during worker initialization. Uniquely identifying a CUDA stream truly requires the (pid, contextId, streamId) triple, not the bare streamId.

### 6.1 AllReduce Is Asymmetric Across GPUs: dev 2 Waits 2× as Long as dev 3

**Data** (single step, 4451–4462 ms):
```
                total allreduce    total compute     allreduce share
  dev 0:          1,674 us          6,781 us          20%
  dev 1:          1,219 us          6,400 us          16%
  dev 2:          2,193 us          6,675 us          25%  ← slowest
  dev 3:          1,083 us          6,661 us          14%  ← fastest
```

Compute is nearly identical (6400–6780us, < 6% variance), but allreduce time differs by **2×**.

**The arrival timing of the first allreduce reveals the root cause**:
```
                start time     end time      duration    waited for whom?
  dev 0:        4451.815ms     4452.419ms     604us      waits for dev 1,3
  dev 2:        4451.999ms     4452.420ms     420us      waits for dev 1,3
  dev 1:        4452.361ms     4452.419ms      58us      waits for dev 3
  dev 3:        4452.412ms     4452.419ms       7us      last to arrive, no wait
```

All GPUs finish at ~4452.419 ms simultaneously (end times differ < 1us), confirming the Lamport protocol synchronized correctly.
But **dev 0 arrived 597us earlier than dev 3**, and those 597us are all burned on spin-wait.

**Root cause analysis**:

This is not an NVSwitch topology issue (subsequent allreduces after every layer are only 6–10us). The real cause is that
**the CPU scheduling gaps between graph replays have different lengths on each GPU**:
- Each `graph.replay()` is issued by CPU, but each of the 4 GPUs has its own CUDA stream queue
- The CPU's `replay()` calls to the 4 GPUs have microsecond-level skew
- The first allreduce absorbs this initial skew
- Once synchronized, subsequent allreduces are nearly free (6–10us) because the GPUs are already in lockstep

**Impact**: The cost of the first allreduce "absorbing the skew" is about 200–600us/step, which is **30–50%** of total allreduce time.
If the 4 GPUs' graph replays could be launched more synchronously (e.g. using a CUDA event to align replay start points),
this overhead could be compressed to near zero.

---

### 6.2 EAGLE3 Draft Model: 87% of Time Waiting on AllReduce (Communication-Bound)

**Data**:
```
                         total GPU time   allreduce    allreduce share   non-AR kernels
main model (17417, 61 layers): 8,456 us      1,674 us       20%           6,782 us
EAGLE3 graph A (17387):         ~370us each  ~324 us        87%             ~46 us
EAGLE3 graph B (17390):         ~189us each  ~151 us        80%             ~38 us
```

Full kernel profile of EAGLE3 graph 17387 (dev 0):
```
allreduce_fusion (324.8us avg)        ← 87% of the time!
nvjet_sm100_tst splitK_TNT (31.2us)   ← lm_head GEMM
triton_red_fused_2 (3.7us)            ← RMSNorm reduce
triton_poi_fused_*                     ← various pointwise
splitKreduce_kernel (3.1us)           ← splitK reduction
```

**What this means**:

The EAGLE3 draft model has only ~2 layers (lm_head + a projection), with only ~20us of compute per layer. But each layer still needs
a TP allreduce, and the allreduce latency floor (Lamport spin-wait ~8us + first-barrier ~300us)
does not shrink with compute.

```
Draft model effective compute ratio:
  graph A: 46us / 370us = 12%   ← 88% of time in communication!
  graph B: 38us / 189us = 20%

For comparison, the main model:
  main:   6782us / 8456us = 80%  ← 20% in communication
```

**Conclusion**: EAGLE3 speculative decoding under TP=4 is **extremely communication-bound**. The cost of generating 3 speculative tokens
is ~560us (graph A + B), of which 475us is pure allreduce wait.

**Optimization directions**:
- Run draft model at TP=1 (no allreduce): 560us → ~84us, saves 85%
- Or use a smaller draft model on a single GPU and only do TP during verification

---

### 6.3 D2D Memcpy Reveals EAGLE3's Token Accept/Reject Mechanism

**Data** (3 groups of D2D memcpy per step, between graph replays):
```
memcpy 1: 1920 KB × 4 GPU  (concurrent_kernels=0, i.e., between graphs)
memcpy 2:   56 KB × 4 GPU  = 4 tokens × 7168 × 2 bytes
memcpy 3:   14 KB × 4 GPU  = 1 token  × 7168 × 2 bytes
```

**Interpretation**:

```
Main model graph 17417 (verify + generate)
  │
  ├→ D2D 1920KB = 240 × 8KB: KV cache block compaction
  │   (after EAGLE3's 3 speculative tokens are verified, the
  │    accepted tokens' KV cache needs to be moved from temp slots
  │    to permanent slots)
  │
  ├→ D2D 56KB = 4 tokens hidden states:
  │   (hidden_states of 1 verified + 3 speculative passed to EAGLE3 draft)
  │
  ├→ D2D 14KB = 1 token hidden state:
  │   (hidden state of the finally committed 1 new token)
  │
  ↓
EAGLE3 graph 17390 → 17387 (draft 3 new speculative tokens)
```

**Key finding**: These memcpys occur between graph replays (concurrent_kernels=0) and are
**serial GPU idle time**. Although each one is only ~2us, 3 groups × 4 GPUs = 24 instances, plus
CPU scheduling overhead, this contributes to part of the inter-step bubble.

---

### 6.4 Shared Expert SM Contention: Actually Faster When Running in Parallel with bmm_E2m1

**Data** (stream 7109 `device_kernel` latency vs concurrent kernel on main stream):

```
Concurrent kernel on main stream   aux device_kernel duration
─────────────────────────────────────────────────────────────
vectorized_elementwise (2us)         8-9 us  ← slow
bmm_E2m1 (22us)                     4-5 us  ← fast!
bmm_Bfloat16 (13us)                11-12 us ← slow
none (gap)                           9 us   ← baseline
```

Latency distribution histogram:
```
 4us: ████████████ (12)         ← parallel with bmm_E2m1
 5us: █████████████████████ (21) ← parallel with bmm_E2m1
 8us: ███████████████████████████████████████████ (43) ← parallel with small kernels
 9us: █████████████████ (17)
11us: █████████ (9)              ← parallel with bmm_Bfloat16
```

**Counterintuitive**: Running in parallel with a large kernel (bmm_E2m1, 22us) makes the shared expert **faster** (4-5us vs 8-9us).

**Reason**: GB200 has 152 SMs. `bmm_E2m1`'s gridDim likely uses only part of the SMs (expert GEMM matrices are small at decode
batch=1), and the remaining SMs can fully serve the aux stream's `device_kernel`.
In contrast, `vectorized_elementwise` executes quickly (2us), but it likely uses many SM blocks for the
elementwise operation (because elementwise kernels often launch many blocks to guarantee low latency),
which instead contends for SMs with the aux stream.

**Corollary**: The efficiency of multi-stream parallelism depends on the **SM occupancy of concurrently running kernels**, not
on the execution time of any single kernel. Small, wide kernels (many blocks, little compute) hurt parallelism efficiency more
than large, narrow kernels (few blocks, lots of compute).

---

## 7. Appendix: Configuration Correction and Issue Localization

> This appendix clarifies the configuration actually in effect for this trace run — it diverges slightly from the command in `REPRODUCE.md`, which affects how some phenomena in §1/§3 are interpreted. If anything in the main text feels off, check this appendix first.

This trace **did not enable sequence parallelism**. So the `all_reduce -> reduce_scatter + all_gather` rewrite you might have been expecting had no prerequisite satisfied in this run.

### 7.1 Effective Configuration (per the decode worker log)

The final `VllmConfig` in `decode-0/decode-0-gb200-rack1-16.log` explicitly shows:

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

Corresponding source paths:

- `vllm/config/vllm.py`: `PostGradPassManager` only registers `SequenceParallelismPass` when `pass_config.enable_sp=True`
- `vllm/compilation/passes/pass_manager.py`: this run actually only registered `AllReduceFusionPass`, with no `SequenceParallelismPass`
- `vllm/config/vllm.py`: the default for `-O3` currently binds `enable_sp` to `IS_DENSE`, which is hard-coded to `False` in the current source

### 7.2 Why AllReduce and Not RS/AG

What is enabled is the FlashInfer **allreduce+rms fusion** (trtllm backend), not sequence parallelism. Log: `Enabled custom fusions: act_quant, allreduce_rms`. In the trace, `allreduce_fusion_kernel_oneshot_lamport` dominates (73,272 invocations, 2,293 ms), with no `ReduceScatter` at all; there are only 2,272 invocations of `ncclDevKernel_AllGather_RING_LL` (47 ms), but they are not paired with RS as SP requires.

### 7.3 The Model Side Also Did Not Take the MoE Sequence Parallel Path

The `KimiK25ForConditionalGeneration` text backbone is `DeepseekV2ForCausalLM` (`vllm/model_executor/models/kimi_k25.py`), and `DeepseekV2MoE.is_sequence_parallel` requires `enable_expert_parallel=True` + `data_parallel_size>1`. In this run `data_parallel_size=1` and `--enable-expert-parallel` was not set, so the SP path on the model side was not triggered either. Additionally, the source comment in `DeepseekV2MoE.forward()` — "replace the all_reduce at the end of attn with reduce_scatter" — is still a TODO; the current implementation has not made that replacement.

### 7.4 Impact on the Rest of This Document

- In `REPRODUCE.md`/`slurm.out`, both of the following appear at the same time:
  - `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'`
  - `--compilation_config.pass_config.fuse_allreduce_rms true`
- But `vllm/utils/argparse_utils.py:parse_args()` reassembles the dotted form of the same root key
  `--compilation-config` into a new JSON argument, and appends it at the end of the argument list;
  this way, in the final argparse result, the appended
  `{"pass_config":{"fuse_allreduce_rms":true}}` overwrites the prior
  `{"cudagraph_mode":"FULL_DECODE_ONLY"}` wholesale instead of deep-merging.
- This exactly explains why, although `slurm.out` contains `FULL_DECODE_ONLY`
  in the command, it disappears from the worker's `non-default args` and final `VllmConfig`.

- The premise that "`FULL_DECODE_ONLY` + mixed batch must be eager" needs to be reviewed.
- The decode worker log actually shows that this run captured:
  - `Capturing CUDA graphs (mixed prefill-decode, PIECEWISE): 17`
  - `Capturing CUDA graphs (decode, FULL): 9`
- Therefore, the parts of §1 Phase 3/4 that explain mixed batches entirely as eager are not fully consistent with the effective configuration.

**Conclusion**: The "anomaly" in this trace is not "SP was enabled but was not split into RS/AG"; it is that **SP was never enabled; what was enabled is the allreduce+rms fusion**. At the same time, the document's premise about the cudagraph mode is slightly off.
