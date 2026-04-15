# HBM Storage Modeling for the Kimi K2.5 NVFP4 Inference System

> Based on actual configuration and log data of experimental `pd_kimi_bench_a_70k` (2026-04-11 01:24)

---

## 1. Overview of experimental environment

| dimensions | values |
|------|-----|
| Model | nvidia/Kimi-K2.5-NVFP4 (based on DeepSeek V3 architecture) |
| GPU | NVIDIA B200 (GB200 NVL72) |
| Memory per GPU | 192 GB HBM3e (CUDA visible 184.31 GiB) |
| Parallel Strategy | TP=4, PP=1, DP=1 (4 GPU/node, 1 node/instance) |
| Deployment mode | PD separation: 1 prefill node + 1 decode node |
| gpu_memory_utilization | 0.85 |
| max_model_len | 131,072 tokens |
| kv_cache_dtype | fp8 |
| Attention Backend | FLASHINFER_MLA (HND layout) |
| Speculative decoding | Eagle3 (kimi-k2.5-eagle3), 3 speculative tokens |

---

## 2. Model architecture parameters

### 2.1 Core structure (text_config, DeepSeek V3 architecture)

| Parameter | Value | Description |
|------|-----|------|
| hidden_size | 7,168 | Hidden layer dimension |
| num_hidden_layers | 61 | Transformer layer number |
| num_attention_heads | 64 | Number of attention heads |
| num_key_value_heads | 64 | Number of KV heads (the KV cache under MLA actually only has 1 latent) |
| vocab_size | 163,840 | Vocabulary size |
| max_position_embeddings | 262,144 | Maximum position encoding |

### 2.2 MLA (Multi-head Latent Attention) parameters

| Parameter | Value | Description |
|------|-----|------|
| kv_lora_rank | 512 | KV compressed latent dimension |
| q_lora_rank | 1,536 | Q compressed latent dimension |
| qk_nope_head_dim | 128 | Per-head dimension of Q/K non-RoPE part |
| qk_rope_head_dim | 64 | Per-head dimension of Q/K RoPE section |
| v_head_dim | 128 | Value per-head dimension |
| **KV cache head_size** | **576** | = kv_lora_rank(512) + qk_rope_head_dim(64) |

**The core idea of MLA**: Instead of storing the complete K/V (128 dimensions per head × 64 heads = 16,384 dimensions), a compressed latent vector (512 dimensions) plus the uncompressible RoPE part (64 dimensions) is stored, for a total of 576 dimensions. The complete K/V is recovered through up-projection during calculation.

### 2.3 MoE (Mixture of Experts) parameters

| Parameter | Value | Description |
|------|-----|------|
| n_routed_experts | 384 | Number of routing experts |
| num_experts_per_tok | 8 | Number of experts activated per token |
| n_shared_experts | 1 | Number of shared experts |
| moe_intermediate_size | 2,048 | Intermediate dimensions per routing specialist |
| intermediate_size | 18,432 | Intermediate layer dimensions for shared expert/dense layers |
| first_k_dense_replace | 1 | The first 1 layer is dense FFN, and the last 60 layers are MoE |### 2.4 Quantitative configuration

| dimensions | values |
|------|-----|
| Weight quantization | NVFP4 (4-bit float, group_size=16) |
| KV cache quantization | FP8 (8-bit float, `--kv-cache-dtype fp8`) |
| Quantization ignore | self_attn*, lm_head, mm_projector*, vision_tower* for all 61 layers |

Note: **Attention weights are not quantized** (keep bfloat16), only the Linear layer of MoE experts uses NVFP4.

---

## 3. Single GPU memory layout

### 3.1 Initial state of video memory (before model loading)

Extracted from the log (prefill and decode nodes are consistent):
```
total_memory = 184.31 GiB (GPU physical total memory, CUDA visible part)
cuda_memory = 2.47 GiB (CUDA runtime + driver overhead)
torch_memory = 0.02 GiB (PyTorch initial usage)
non_torch_memory = 2.45 GiB (non-PyTorch CUDA allocation)
free_memory = 181.84 GiB (available video memory)
torch_peak       =   0.01 GiB
```
### 3.2 Requested Memory (the upper limit of video memory controlled by vLLM)
```
requested_memory = gpu_memory_utilization × total_memory
                 = 0.85 × 184.31 GiB
= 156.66 GiB (log confirmation)
```
The remaining 15% (27.65 GiB) is reserved for the OS, CUDA runtime, other processes, etc.

### 3.3 Model weight occupation

Total model safetensors disk size: **550.24 GB** (590,779,912,064 bytes, 119 shards)

| Weight category | Number of tensors | Description |
|---------|---------|------|
| MoE Expert (gate/up/down_proj) | 276,604 | NVFP4 quantization, accounting for the majority |
| Shared Expert | 720 | NVFP4 Quantization |
| Attention (self_attn) | 549 | **bfloat16 (unquantized)** |
| Embedding + LM head | 5 | bfloat16 |
| Others (LayerNorm, etc.) | 463 | bfloat16 |

**TP=4 Order GPU Weight Estimation**:
```
Weight per GPU ≈ total weight / TP_size
            ≈ 590,779,912,064 / 4 bytes
            ≈ 147,694,978,016 bytes
            ≈ 137.5 GiB
```
In addition, the weight of the **Eagle3 speculative decoding model** needs to be loaded. The measured steady-state increment is **~0.87 GiB/GPU** (the temporary peak value during loading is ~5.08 GiB, and the temporary memory is released after the loading is completed).

### 3.4 Overview of video memory allocation (log measured data)

#### Prefill node (actual measurement)
```
Initial free memory:     181.84 GiB
Requested memory:        156.66 GiB (0.85 × 184.31)
weights memory: 140.39 GiB (main model 139.52 + Eagle3 0.87, steady state)
torch peak increase: 2.41 GiB (activation peak)
non-torch forward: -2.99 GiB (temporary memory released during profiling)
Total non-KV-cache:      139.81 GiB
CUDA Graph: 0 GiB (enforce_eager disabled)
─────────────────────────────────────
Available KV Cache: 16.85 GiB ← Actual Available
num_gpu_blocks:           14,413
GPU KV cache size:       461,216 tokens (14,413 × 32)
Max concurrency @131K:     3.52x
```
#### Decode node (actual measurement)
```
Initial free memory:     181.84 GiB
Requested memory:        156.66 GiB (0.85 × 184.31)
weights memory: 140.36 GiB (main model 139.49 + Eagle3 0.87, steady state)
torch peak increase: 2.66 GiB (activation peak)
non-torch forward:        -1.40 GiB
Total non-KV-cache:      141.62 GiB
CUDA Graph (actual): 0.75 GiB (PIECEWISE=17, FULL=9, 13s capture)
CUDA Graph (estimated): 0.59 GiB (difference 0.16 GiB / 21.5%)
─────────────────────────────────────
Available KV Cache: 15.04 GiB ← profiling calculation (⚠ not deducting CUDA Graph)
Effective KV Cache: 14.14 GiB ← Actual available at runtime (CUDA Graph squeezes 0.75 GiB)
num_gpu_blocks:           12,866
GPU KV cache size:       411,712 tokens (12,866 × 32)
Max concurrency @131K:     3.14x
```
#### Video memory layout diagram
```
┌─────────────────── Single GPU Memory Layout (184.31 GiB) ────────────────────┐
│                                                                    │
│  ┌──────────────────────────────────────────────┐                 │
│ │ Reserved (15%) 27.65 GiB │ ← Not managed by vLLM│
│ │ (OS, CUDA runtime, driver, other processes) │ │
│  ├──────────────────────────────────────────────┤                 │
│  │                                              │                 │
│ │ vLLM Controlled Area (85%) 156.66 GiB │ │
│  │  ┌──────────────────────────────────────┐    │                 │
│ │ │ Model weight (main + Eagle3) │ │ │
│  │  │ 140.36~140.39 GiB                   │    │                 │
│  │  ├──────────────────────────────────────┤    │                 │
│ │ │ Activation peak (forward calculation) │ │ │
│  │  │ Prefill: 2.41 GiB / Decode: 2.66 GiB│    │                 │
│  │  ├──────────────────────────────────────┤    │                 │
│ │ │ CUDA Graph (decode only) │ │ │
│  │  │ Prefill: 0 / Decode: 0.75 GiB       │    │                 │
│  │  ├──────────────────────────────────────┤    │                 │
│  │  │ KV Cache Pool (FP8)                  │    │                 │
│  │  │ Prefill: 16.85 GiB / Decode: 15.04  │    │                 │
│  │  └──────────────────────────────────────┘    │                 │
│  │                                              │                 │
│  └──────────────────────────────────────────────┘                 │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```
### 3.5 Model weight loading details

| Stage | Prefill | Decode | Description |
|------|---------|--------|------|
| Peak after main model loading | 139.52 GiB | 139.49 GiB | 119 shards, about 26-29 minutes |
| Eagle3 peak value after loading | 144.60 GiB | 144.57 GiB | Including **Temporary memory ~4.21 GiB** during loading |
| **Final weight occupation (steady state)** | **140.39 GiB** | **140.36 GiB** | After profiling, temporary memory has been released |
| Eagle3 Steady State Increment | +0.87 GiB | +0.87 GiB | = Final Steady State - Main Model Peak |

Note: The difference between the peak value during loading of Eagle3 (144.57 GiB) and the steady state (140.36 GiB) is 4.21 GiB. This part is the temporary buffer loaded by safetensors, which is automatically released after the loading is completed. The **actual weight occupation of the Eagle3 model is only 0.87 GiB**.

### 3.6 Differences between Prefill vs Decode nodes

| Dimensions | Prefill node | Decode node |
|------|-------------|-------------|
| Compile mode | `--enforce-eager` (no compilation) | `-O3` (torch.compile) |
| CUDA Graph | **Disable** | **Enable** (`FULL_DECODE_ONLY`) |
| CUDA Graph memory | 0 GiB | **0.75 GiB** (actual measurement) |
| KV Transfer | MultiConnector (NIXL + Mooncake) | NixlConnector |
| Available KV Cache | **16.85 GiB** | **15.04 GiB** |
| num_gpu_blocks | **14,413** | **12,866** |
| KV cache tokens | **461,216** | **411,712** |
| Concurrency@131K | **3.52x** | **3.14x** |
| Activation Peak | 2.41 GiB | 2.66 GiB |
| Compilation optimization | `norm_quant`, `act_quant`, `allreduce_rms` | `act_quant`, `allreduce_rms` |

---

## 4. KV Cache storage modeling

### 4.1 MLA KV Cache structure

MLA does not store the traditional K and V matrices, but the compressed latent:
```
Traditional MHA/GQA KV Cache (per token per layer):
K: [num_kv_heads, head_dim] e.g., [64, 128] = 8,192 elements
V: [num_kv_heads, head_dim] e.g., [64, 128] = 8,192 elements
Total: 16,384 elements

MLA KV Cache (per token per layer):
latent: [1, kv_lora_rank + qk_rope_head_dim] = [1, 576] = 576 elements
(kv_lora_rank=512 stores the compressed KV, qk_rope_head_dim=64 stores the RoPE part)

Compression ratio: 16,384 / 576 ≈ 28.4x
```
### 4.2 Single KV Cache Block size

KV cache in vLLM is managed in "block" (page) units:

| Parameter | Value |
|------|-----|
| block_size | **32 tokens** (FLASHINFER_MLA kernel requirement, see description below) |
| num_kv_heads (effective) | 1 (MLA latent is shared across heads) |
| head_size | 576 (kv_lora_rank + qk_rope_head_dim) |
| dtype | fp8 (1 byte/element) |

**Why block_size = 32 instead of the default 16?**

vLLM defaults to `DEFAULT_BLOCK_SIZE = 16` (`vllm/config/cache.py:34`), but the kernel of FLASHINFER_MLA backend only supports block_size=[32, 64] (`vllm/v1/attention/backends/mla/flashinfer_mla.py:49-50`). vLLM in `get_preferred_block_size()` detected that 16 is not supported, and automatically upgraded to the minimum supported value of 32. This is a **kernel hard requirement** and has nothing to do with Eagle3 speculative decoding.

**Calculation (Source: `vllm/v1/kv_cache_interface.py` `MLAAttentionSpec.real_page_size_bytes`)**:```python
page_size_bytes = block_size × num_kv_heads × head_size × dtype_size
               = 32 × 1 × 576 × 1
               = 18,432 bytes
               = 18 KiB per block per layer
```
### 4.3 Full-layer KV Cache Block size

Theoretical per-block size of main model 61 layers:
```
total_page_size (main model) = page_size_per_layer × num_layers
                         = 18,432 × 61
                         = 1,124,352 bytes
≈ 1.07 MiB per block (main model 61 layers only)
```
However, the actual block overhead of this experiment does not need to be estimated through "group_size inversion". It is clearly registered in the Decode log:```text
language_model.model.layers.0..60.self_attn.attn   -> torch.Size([12866, 32, 576])
model.layers.61.self_attn.attn                     -> torch.Size([12866, 2, 32, 16, 128])
```
This shows that a block index actually contains two types of pages:

- **61 master model MLA layers**: `18,432 B` per layer
- **1 Eagle3 draft attention layer**: `2 × 32 × 16 × 128 × 1 = 131,072 B`

Therefore, the "total cost of a single block across all layers" in this experiment is:
```
total_page_size (this experiment)
  = 61 × 18,432 + 131,072
  = 1,255,424 bytes
  ≈ 1.20 MiB per block
```
This result is completely consistent with the log:
```
Decode: 12,866 × 1,255,424 B = 16,152,285,184 B = 15.04 GiB
Prefill: 14,413 × 1,255,424 B = 18,094,426,112 B = 16.85 GiB
```
In other words, **Eagle3 brought an additional 4,096 B KV overhead** to each token in this experiment, corresponding to 1 draft attention layer, instead of "about 7 layers" as written in the previous document.

### 4.4 KV Cache capacity (log measurement)

| Node | Available KV Cache | num_gpu_blocks | block_size | Total token capacity | Concurrency@131K |
|------|-------------|---------------|-----------|---------------|----------|
| **Prefill** | **16.85 GiB** | **14,413** | 32 | **461,216** | **3.52x** |
| **Decode** | **15.04 GiB** | **12,866** | 32 | **411,712** | **3.14x** |

block_size = 32 is a hard requirement of **FLASHINFER_MLA kernel** (only supports [32, 64], see section 4.2 for explanation).

The Decode node log shows `Overriding num_gpu_blocks=0 with num_gpu_blocks_override=128`, which is a **temporary override** (`gpu_model_runner.py:5827-5836`) of the CUDA graph profiling stage, which is only used to allocate the minimum KV cache to measure graph memory. The final num_gpu_blocks is recalculated after profiling is completed.

This experiment hits the special case of `len(kv_cache_groups) == 1 && UniformTypeKVCacheSpecs` (`vllm/v1/core/kv_cache_utils.py:1098-1119`), so:```python
num_blocks = available_memory // total_page_size_per_block
```
Among them, `total_page_size_per_block` is not 18,432 B in a single layer, but:```python
total_page_size_per_block = sum(spec.page_size_bytes for spec in kv_cache_specs.values())
                          = 61 * 18,432 + 131,072
                          = 1,255,424 bytes
```
So the more accurate calculation method in this experiment is:

- `available_memory` = `requested_memory - non_kv_cache_memory`
- `per_block_total_bytes` = `1,255,424`
- `num_gpu_blocks` = `available_memory // per_block_total_bytes`

**Decode node verification**:
```
available = 156.66 - 141.62 = 15.04 GiB                       ✓
num_gpu_blocks = floor(15.04 GiB / 1,255,424 B) = 12,866      ✓
token_capacity = 12,866 × 32 = 411,712                        ✓
concurrency@131K = 411,712 / 131,072 = 3.14x                  ✓
```
### 4.5 KV Cache overhead per token

| Indicators | Main model 61 layers | Contains Eagle3 draft 1 layer |
|------|-------------|----------------------|
| Master model MLA per token | 576 bytes/layer | — |
| Full layer of main model per token | 576 × 61 = 35,136 bytes ≈ 34.3 KiB | — |
| per token draft layer | — | 2 × 16 × 128 × 1 = 4,096 bytes |
| Total of all layers per token | — | **39,232 bytes ≈ 38.3 KiB** |
| 1K tokens all tiers | — | **~37.4 MiB** |
| 131K tokens all tiers | — | **4.789 GiB** |

If you consider the 32-token page granularity of vLLM, the common length in the bench can also be written as:

- `70,000 tokens` -> `ceil(70000 / 32) = 2,188 blocks` -> `2.558 GiB`
- `7,000 tokens` -> `219 blocks` -> `256.1 MiB`
- `10,500 tokens` -> `329 blocks` -> `384.7 MiB`
- `52,500 tokens` -> `1,641 blocks` -> `1.919 GiB`

**Key insights**: Thanks to MLA compression + FP8 quantization, the 61-layer KV cache of the main model is already very economical; in this experiment, the additional Eagle3 draft layer only added **4 KiB/token**, so a complete request for a `131K` context only requires a total of **4.789 GiB** HBM KV capacity.

### 4.6 Combined effect of MLA compression + FP8
```
Traditional MHA + bfloat16 per token per layer:
  = 2 × num_kv_heads × head_dim × 2 bytes
  = 2 × 64 × 128 × 2 = 32,768 bytes

MLA + FP8 per token per layer:
  = 1 × 576 × 1 = 576 bytes

Total compression ratio = 32,768 / 576 ≈ 56.9x
```
This 56.9x compression breaks down to:
- MLA latent compression: ~28.4x (16,384 dimensions → 576 dimensions)
- FP8 vs bfloat16: 2x (2 bytes → 1 byte)

### 4.7 HBM working set aligned with this benchmark

The corresponding KV footprint of the bench parameters (`70k input`, `300 output`, `10 turns`, `global prefix 15%`, `conversation prefix 75%`) is as follows:

| Composition | tokens | blocks | HBM occupation |
|------|--------|--------|----------|
| Global prefix | 10,500 | 329 | **0.385 GiB** |
| Conversation prefix | 52,500 | 1,641 | **1.919 GiB/conversation** |
| Unique segment | 7,000 | 219 | **0.256 GiB/request** |
| Full 70k turn | 70,000 | 2,188 | **2.558 GiB / Request** |
| 300 decode tokens | 300 | 10 | **0.0117 GiB / request** |

If all conversations in the benchmark are regarded as the theoretical worst case scenario of "conversation prefix needs to reside at the same time", then the prefix-only working set is:

| Sweep Concurrency | Number of Conversations | Global + Conversation Prefix | Turn 1 Full Working Set |
|-----------|-------|----------------------------------|-------------------|
| 1 | 2 | **4.22 GiB** | **4.73 GiB** |
| 2 | 4 | **8.06 GiB** | **9.08 GiB** |
| 4 | 8 | **15.73 GiB** | **17.78 GiB** |
| 8 | 16 | **31.08 GiB** | **35.18 GiB** |

After comparing this with the HBM capacity in the log, we can directly draw several conclusions:

- The 16.85 GiB KV pool of **Prefill node** can theoretically barely cover the prefix working set when `C=4`, but it is already close to the limit.
- **Decode node** If the 0.75 GiB of CUDA Graph is also included, the effective KV is only **14.14 GiB**. Then when `C=4`, even "global + 8 conversation prefix" cannot be installed.
- When `C=8`, the theoretical prefix working set reaches **31.08 GiB**, which is far beyond the prefill/decode HBM KV budget of a single GPU. It must rely on eviction, prefix cache deactivation recycling, or lower-level KV media.

---

## 5. Summary of actual video memory measurements

### 5.1 Prefill node (single GPU, actual measurement)

| Component | Size | Data Source |
|------|------|---------|
| GPU total physical memory | 184.31 GiB | `total_memory` |
| vLLM governance limit (85%) | 156.66 GiB | `requested_memory` |
| Model weights (main model + Eagle3) | **140.39 GiB** | `weights_memory` (profiling log) |
| Activation peak | **2.41 GiB** | `torch_peak_increase` |
| Non-torch forward | **-2.99 GiB** | Temporary memory released during profiling |
| CUDA Graph | **0 GiB** | `enforce_eager` disabled |
| **Available KV Cache** | **16.85 GiB** | `Available KV cache memory` |
| num_gpu_blocks | **14,413** | `cache_config_info` |
| Token capacity | **461,216** | `GPU KV cache size` |
| Concurrency@131K | **3.52x** | `Maximum concurrency` |### 5.2 Decode node (single GPU, actual measurement)

| Component | Size | Data Source |
|------|------|---------|
| GPU total physical memory | 184.31 GiB | `total_memory` |
| vLLM governance limit (85%) | 156.66 GiB | `requested_memory` |
| Model weights (main model + Eagle3) | **140.36 GiB** | `weights_memory` |
| Activation Peak | **2.66 GiB** | `torch_peak_increase` |
| Non-torch forward | **-1.40 GiB** | profiling period |
| CUDA Graph | **0.75 GiB** | `Graph capturing took 0.75 GiB` |
| **Available KV Cache** | **15.04 GiB** | `Available KV cache memory` |
| Effective KV Cache (with CUDA Graph) | **14.14 GiB** | `--kv-cache-memory=15185736397` |
| Effective num_gpu_blocks (if graph withheld) | **12,096** | `14.14 GiB / 1,255,424 B` |
| Effective token capacity (if graph withheld) | **387,072** | `12,096 × 32` |
| num_gpu_blocks | **12,866** | `cache_config_info` |
| Token capacity | **411,712** | `GPU KV cache size` |
| Concurrency@131K | **3.14x** | `Maximum concurrency` |

### 5.3 Single node (4 GPU) - Note that KV cache is synchronized when TP=4

| Metrics | Prefill node | Decode node |
|------|-------------|-------------|
| Total GPU Memory | 4 × 184.31 = 737.2 GiB | 737.2 GiB |
| Total model weights (across 4 GPUs) | ~550 GB (~140 GiB per GPU) | ~550 GB |
| KV Cache / GPU | **16.85 GiB** | **15.04 GiB** |
| Token capacity (limited to single GPU) | **461,216** | **411,712** |
| 131K context concurrency number | **3.52** | **3.14** |

**Important**: When TP=4, each GPU stores **the same** KV cache blocks (1/4 of all layers are on different GPUs, but the number of blocks is the same). Therefore, the concurrency capability is based on the num_gpu_blocks of **single GPU**, and the KV cache of 4 GPUs cannot be simply added.

---

## 6. vLLM KV Cache allocation mechanism

### 6.1 Memory Profiling process (`vllm/v1/worker/gpu_worker.py`)
```
1. Record the initial video memory snapshot (init_memory_snapshot) [line 284]
   → torch_peak=0.01, free_memory=181.84, total_memory=184.31 GiB

2. Calculate requested_memory = gpu_memory_utilization × total_memory [line 285]
   = 0.85 × 184.31 = 156.66 GiB

3. Assign model weights (cumem allocator)
→ Allocate to GPU block by layer, main model ~139.5 GiB + Eagle3 ~0.87 GiB

4. Execute profile_run (dummy forward pass) [line 366]
→ Measure activation peak memory

5. Calculate non_kv_cache_memory: [line 387-391]
   non_kv_cache_memory = non_torch_increase + torch_peak_increase + weights_memory
   Decode: -1.40 + 2.66 + 140.36 = 141.62 GiB

6. Calculate available KV cache: [line 417-421]
   available_kv_cache = requested_memory - non_kv_cache_memory - cudagraph_estimate_applied
   Decode: 156.66 - 141.62 - 0 = 15.04 GiB

⚠ cudagraph_estimate_applied = 0 because VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS
It is turned off by default. The 0.75 GiB of CUDA Graph is actually occupied during the capture phase, which will occupy the KV cache space.

7. Calculate num_gpu_blocks:
This experiment uses the `UniformTypeKVCacheSpecs` special case
   num_gpu_blocks = available_memory // total_page_size_per_block
                  = 15.04 GiB // 1,255,424 B
                  = 12,866
```
### 6.2 Block Management

This experiment is not a general case of "single page size × group_size", but a special case of `UniformTypeKVCacheSpecs`:```python
# Main model MLA layer
mla_page_size = 32 × 1 × 576 × 1 = 18,432 bytes

# Eagle3 draft layer (the log shows cache shape = [num_blocks, 2, 32, 16, 128])
draft_page_size = 2 × 32 × 16 × 128 × 1 = 131,072 bytes

# Total per-block size after merging all layers
total_page_size_per_block = 61 × 18,432 + 131,072
                          = 1,255,424 bytes

num_blocks = available_memory // total_page_size_per_block
```
### 6.3 Special processing of CUDA Graph graphics memory

CUDA Graph memory of Decode node (0.75 GiB) **Does not participate** KV cache budget calculation:
```
When available_kv_cache is calculated:
cudagraph_estimate_applied = 0 (VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS default False)

CUDA Graph capture after KV cache allocation:
  Graph capturing finished in 13 secs, took 0.75 GiB (actual)
Graph estimated: 0.59 GiB, 21.5% difference

Actual effect: KV cache is nominally 15.04 GiB, but CUDA Graph will occupy 0.75 GiB of it during runtime.
Effective KV cache ≈ 14.14 GiB (Log suggestion: --kv-cache-memory=14.14 GiB)

vLLM v0.19 will enable VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1, by default
At that time, CUDA Graph memory will be deducted before KV cache calculation.
```
### 6.4 KV Cache memory pool

In this experiment, vLLM will allocate a KV tensor to each attention layer separately, but all layers share the same `num_gpu_blocks`:```python
# 61 master model MLA layers
per_mla_layer_shape = [num_gpu_blocks, 32, 576]      # fp8
per_mla_layer_size  = num_gpu_blocks * 18,432 bytes

# 1 Eagle3 draft layer
per_draft_layer_shape = [num_gpu_blocks, 2, 32, 16, 128]  # fp8
per_draft_layer_size  = num_gpu_blocks * 131,072 bytes

total_kv_pool = 61 * per_mla_layer_size + per_draft_layer_size
```
Take the decode node as an example:```text
61 × (12,866 × 18,432) + 1 × (12,866 × 131,072)
= 16,152,285,184 bytes
= 15.04 GiB
```
---

## 7. Cheat Sheet of Important Numerical Values
```
┌──────────────────────────────────────────────────────────────────┐
│ GPU physical memory: 192 GB (184.31 GiB CUDA visible) │
│ vLLM Control (85%): 156.66 GiB / GPU │
│ Model weight / GPU (TP=4, actual measurement): 140.36~140.39 GiB │
│ └ Main model: 139.49~139.52 GiB │
│ └ Eagle3: ~0.87 GiB (steady state) │
│Activation peak: Prefill 2.41 / Decode 2.66 GiB │
│ CUDA Graph:                      Prefill 0 / Decode 0.75 GiB     │
│                                                                   │
│ KV Cache block_size: 32 tokens (actual) │
│ KV Cache head_size:              576 (kv_lora_rank + rope_dim)    │
│ KV Cache num_kv_heads:           1 (MLA latent)                   │
│ KV Cache dtype:                  fp8 (1 byte)                     │
│                                                                   │
│ Available KV Cache: Prefill 16.85 / Decode 15.04 GiB│
│ num_gpu_blocks:                  Prefill 14,413 / Decode 12,866   │
│Token capacity: Prefill 461,216 / Decode 411,712 │
│ Concurrency@131K: Prefill 3.52x / Decode 3.14x │
│                                                                   │
│ Main model 61 layers per token: 35,136 bytes (34.3 KiB) │
│ Each token contains Eagle3 1 layer: 39,232 bytes (38.3 KiB) │
│ 131K context (including Eagle3): 4.789 GiB │
│ Decode effective token capacity (deducting graph): 387,072 tokens / 2.95x @131K │
│ MLA+FP8 combined compression ratio (vs MHA+bf16): 56.9x │
└──────────────────────────────────────────────────────────────────┘
```
---

## 8. Limitations and follow-up work

### 8.1 Data Integrity

All HBM data in this document comes from actual logs (642K lines of decode log, ~95K lines of prefill log), including:
- Model weight, activation, CUDA graph, and KV cache allocation are all **profiling measured values**
- `num_gpu_blocks` and token capacity come from vLLM EngineCore logs

### 8.2 Confirmed vs Pending Confirmation

| Project | Status | Source |
|------|------|------|
| block_size = 32 | **Confirmed** | FLASHINFER_MLA kernel requirements [32,64], source code `flashinfer_mla.py:44-46` |
| num_kv_heads = 1 (MLA) | **Confirmed** | Main model cache shape `[num_blocks, 32, 576]` |
| head_size = 576 | **Confirmed** | Model configuration + log `AttentionSelectorConfig(head_size=576)` |
| Main model page_size = 18,432 bytes/layer | **Confirmed** | `kv_cache_interface.py:182-189` |
| Eagle3 extra draft layer = 1 layer | **Confirmed** | Log registration `model.layers.61.self_attn.attn` |
| draft layer page_size = 131,072 bytes | **Confirmed** | cache shape `[num_blocks, 2, 32, 16, 128]` |
| total_page_size_per_block = 1,255,424 bytes | **Confirmed** | `61×18,432 + 131,072` |
| weights_memory | **Confirmed** | Log profiling measured value |
| available_kv_cache | **Confirmed** | Log `Available KV cache memory` |
| num_gpu_blocks | **Confirmed** | log `cache_config_info` |
| Eagle3 Weight 0.87 GiB | **Confirmed** | Log Steady-State Difference (140.36 - 139.49) |
| MLA compression ratio 56.9x | **Confirmed** | Theoretical calculation, source code verification |
| CUDA Graph is not deducted from KV cache | **Confirmed** | `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS` Default False |

### 8.3 Known issues

1. **non-torch forward negative value**: profiling shows a negative non-torch forward increase (Prefill: -2.99 GiB, Decode: -1.40 GiB). This is normal behavior — the temporary buffer allocated by CUDA during model loading is released after profiling is completed. (`vllm/utils/mem_utils.py:271-274`)

2. **CUDA Graph graphics memory occupies KV cache**: Decode node has a nominal 15.04 GiB KV cache, but CUDA Graph actually occupies 0.75 GiB (vs estimated 0.59 GiB, deviation 21.5%), effective KV cache ≈ 14.14 GiB, corresponding to an effective block number of approximately 12,096, and an effective token capacity of approximately 387,072. vLLM v0.19 will enable `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1` by default to resolve this issue.3. **Benchmark prefix working set exceeds HBM** under high concurrency: when `C=4`, only the global prefix + 8 conversation prefixes reach 15.73 GiB; when `C=8`, it reaches 31.08 GiB. A single decode TP group cannot fully preserve all hot prefixes with HBM alone.

### 8.4 Subsequent modeling direction

- [ ] Modeling the CPU DDR (L2) layer: offload strategy and capacity planning for SimpleCPUOffloadConnector
- [ ] Modeling the NVMe (L3) layer: Capacity planning for local disk cache
- [ ] Modeling cross-node KV cache transmission bandwidth: actual throughput of NIXL (NVLink) vs Mooncake (RDMA)
- [ ] Consider the impact of prefix caching on effective KV cache utilization
- [ ] KV cache stress test in multi-user concurrent scenario

---

## Appendix A: Hierarchical analysis of model weights (TP=4 single GPU)

### Attention weight (bfloat16, not quantized)

Main parameters of each layer of Attention (approximately):

| Matrix | Shape (full amount) | Parameter amount | After TP=4 / GPU |
|------|-------------|--------|--------------|
| q_a_proj | [7168, 1536] | 11.0M | 11.0M (not split) |
| q_a_layernorm | [1536] | 1.5K | 1.5K |
| q_b_proj | [1536, 64×192] | 18.9M | 4.7M |
| kv_a_proj_with_mqa | [7168, 576] | 4.1M | 4.1M (not split) |
| kv_a_layernorm | [512] | 512 | 512 |
| kv_b_proj | [512, 64×256] | 8.4M | 2.1M |
| o_proj | [64×128, 7168] | 58.7M | 14.7M |

Each layer Attention / GPU ≈ 37M params × 2 bytes (bf16) ≈ 74 MB
61 layers total ≈ **4.4 GiB / GPU**

### MoE weights (NVFP4, quantized)

Every routing expert at every layer:
- gate_proj + up_proj + down_proj: 3 × 7168 × 2048 = 44.0M params
- NVFP4 (0.5 bytes + scales): ≈ 0.5625 bytes/param → ~24.8 MB / expert

TP=4: 384/4 = 96 routing experts per GPU

MoE/GPU per layer:
- 96 Routing Expert: 96 × 24.8 MB ≈ 2.38 GB
- 1 shared expert (TP split): 3 × 7168 × 18432 / 4 × 0.5625 bytes ≈ 55.9 MB
- Gate: 7168 × 384 × 2 bytes ≈ 5.3 MB
- MoE/GPU ≈ ~2.44 GB per layer

Total 60 MoE tiers ≈ **~146.4 GB ≈ 136.3 GiB / GPU**

(Note: The above MoE estimates are based on theoretical calculations, and may actually differ slightly due to padding, quantification metadata, etc.)

### Others

| Components | Size / GPU |
|------|-----------|
| Embedding (vocab_parallel) | 163,840 × 7,168 / 4 × 2 bytes ≈ 561 MB |
| LM Head | ~561 MB (may be shared with embedding) |
| LayerNorm (61 layers) | Smaller (~a few MB) |---

## Appendix B: KV Cache capacity and concurrency relationship

Actual measurement based on Decode node: 12,866 blocks × 32 tokens = 411,712 tokens total pool:

| Scenario | Tokens per request | Blocks per request | KV cache per request | Maximum concurrency |
|------|-------------|-------------|---------------|---------|
| 131K context (max) | 131,072 | 4,096 | **4.789 GiB** | **3.14** (measured) |
| 70K context (bench) | 70,000 | 2,188 | **2.558 GiB** | ~5.88 |
| 32K context | 32,000 | 1,000 | **1.169 GiB** | ~12.87 |
| 8K context | 8,000 | 250 | **0.292 GiB** | ~51.47 |

Actual measurement based on Prefill node: 14,413 blocks × 32 tokens = 461,216 tokens total pool:

| Scenario | Maximum concurrency |
|------|---------|
| 131K context (max) | **3.52** (actual measurement) |
| 70K context (bench) | ~6.59 |
| 32K context | ~14.41 |
| 8K context | ~57.65 |

Note: In practice, prefix caching hit rate, multi-round dialogue sharing prefixes, and KV cache block fragmentation need to be considered.

---

## Appendix C: Comparison of different KV Cache quantification strategies

| Strategy | Per token per layer | Full 61 layers per token | 131K context | Compression ratio vs MHA+bf16 |
|------|-------------|------------------|-------------|-------------------|
| MHA + bfloat16 | 32,768 B | 1,998,848 B (1.9 MiB) | ~249 GiB | 1x |
| MHA + fp8 | 16,384 B | 999,424 B (976 KiB) | ~125 GiB | 2x |
| **MLA + bfloat16** | 1,152 B | 70,272 B (68.6 KiB) | ~8.76 GiB | 28.4x |
| **MLA + fp8 (main model 61 layers)** | **576 B** | **35,136 B (34.3 KiB)** | **~4.38 GiB** | **56.9x** |
| **MLA + fp8 + Eagle3 1 layer (total for this experiment)** | — | **39,232 B (38.3 KiB)** | **4.789 GiB** | — |
| MLA + fp8_ds_mla | 656 B | 40,016 B (39.1 KiB) | ~4.99 GiB | 49.9x |

---

*Document generation time: 2026-04-11*
*Data source: vigil/logs/pd_kimi_bench_a_70k/2026-04-11/20260411_012402*