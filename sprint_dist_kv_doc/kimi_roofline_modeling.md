# Kimi K2.5 NVFP4 Main Model Roofline Modeling

> Goal: Starting from the main model `nvidia/Kimi-K2.5-NVFP4`, estimate its theoretical computing intensity on the current `GB200/B200 + TP=4` machine, determine at what batch size it will switch from HBM bandwidth limited to computing power limited, and map this conclusion to the `prefill/decode` configuration of the current `pd_kimi_bench_a_70k`.

---

## 1. Let’s look at the conclusion first

Let’s first give some of the most important conclusions this time:

- **Machine roofline**: According to the current machine documentation and NVIDIA official `GB200 NVL72` specification page, for a single GPU, **`HBM = 8 TB/s`**, **`BF16 dense = 2.5 PFLOPS`**, so **ridge point = `312.5 FLOP/B`**.
- **Main model structure**: The backbone of `Kimi-K2.5-NVFP4` is **`61` layer**, of which **the `0` layer is dense FFN**, and the remaining **`60` layers are MoE**. `self_attn` weight remains **BF16**, MoE/FFN Linear weight goes **NVFP4**.
- **Single layer matmul FLOPs are neat**: under `TP=4`,
  - The attention projection part is about **`73.27M FLOPs/token/GPU`**
  - dense FFN about **`198.18M FLOPs/token/GPU`**
  - `8 routed + 1 shared` in the MoE layer adds up to about **`198.18M FLOPs/token/GPU`**
  - So the total amount of matmul in any layer is about **`271.45M FLOPs/token/GPU`**
- **If you only look at the dense first layer**, the overall AI can touch the BF16 roofline at around **`M ~= 170`**.
- **But if you look at the real whole model**, the MoE token will be dispersed to many experts, making it difficult for the expert GEMM to grow during decoding. After approximating with occupancy, the ridge crossing of the full model is at `M ~= 4554 tokens/step`**, which is much larger than the realistic reachable range of decode.
- **Operating point of current benchmark**:
  - decode: `max_num_seqs = 16`, the full model AI is only **`~5.8 FLOP/B`**
  - prefill: `max_num_batched_tokens = 8192`, the full model AI is about **`429 FLOP/B`**
  - In other words, **The current Kimi main model decode is obviously dominated by HBM/weight traffic, and the prefill large chunk has entered near or even above the compute-bound interval**

In other words:

- If the goal is to optimize **decode**, the main contradiction is not "adding a little bit to batch can fully consume the computing power", but **MoE small GEMM + weight traffic + routing dispersion**.
- If the goal is to optimize **prefill**, the chunk size of `8192` is already not low from a pure roofline perspective. The marginal benefit of increasing the batch size will be more reflected in **reducing the number of chunks and scheduling overhead**, rather than continuing to increase the arithmetic intensity of the main GEMM.

---

## 2. Machine Roofline parameters

### 2.1 HBM bandwidth

It is documented in the local machine documentation:

- Model: `NVIDIA GB200 NVL72`
- Single GPU HBM: `192 GB HBM3e`
- Single GPU HBM bandwidth modeled at **`8 TB/s`**

Local reference:- `vllm/sprint_dist_kv_doc/machine_doc.md`

### 2.2 GB200/B200 theoretical computing power

The `GB200 Grace Blackwell Superchip` specifications given on NVIDIA’s official `GB200 NVL72` page are:

- `FP16/BF16 Tensor Core = 10 PFLOPS`
- `GPU Memory Bandwidth = 16 TB/s`

Footnote on the same page:

- `Specification in sparse. Dense is one-half sparse spec shown.`

Therefore, converted to **dense** and divided by **`2 GPU / superchip`**, we get single GPU:

| Metrics | Nominal per Superchip | dense per Superchip | dense per GPU |
|------|-------------------|--------------------|-----------------|
| BF16 Tensor | `10 PFLOPS` | `5 PFLOPS` | **`2.5 PFLOPS`** |
| HBM Bandwidth | `16 TB/s` | `16 TB/s` | **`8.0 TB/s`** |

Official source:

- https://www.nvidia.com/en-us/data-center/gb200-nvl72/

### 2.3 Ridge Point

Defined by roofline:```text
ridge_point = peak_compute / peak_bandwidth
            = 2.5e15 / 8.0e12
            = 312.5 FLOP/B
```
Therefore, this article adopts uniformly:

- **`BF16 ridge = 312.5 FLOP/B`**

---

## 3. Main model structure and TP4 local shape

The objects of this modeling are:

- `/mnt/lustre/hf-models/hub/models--nvidia--Kimi-K2.5-NVFP4/snapshots/c0285e649c34d4386b01e38abca642c06cbe014e`

Configuration from:

- `config.json`
- `vllm/model_executor/models/deepseek_v2.py`

### 3.1 Key model parameters

| Parameter | Value |
|------|----|
| `hidden_size` | `7168` |
| `num_hidden_layers` | `61` |
| `num_attention_heads` | `64` |
| `q_lora_rank` | `1536` |
| `kv_lora_rank` | `512` |
| `qk_nope_head_dim` | `128` |
| `qk_rope_head_dim` | `64` |
| `v_head_dim` | `128` |
| `intermediate_size` | `18432` |
| `moe_intermediate_size` | `2048` |
| `n_routed_experts` | `384` |
| `num_experts_per_tok` | `8` |
| `n_shared_experts` | `1` |
| `first_k_dense_replace` | `1` |

### 3.2 Structure explanation

It can be seen directly from `DeepseekV2DecoderLayer`:

- Layer `0` uses `DeepseekV2MLP(intermediate_size=18432)`, which is **dense FFN**
- Layers `1~60` use `DeepseekV2MoE`
- In `DeepseekV2MoE`:
  - routed expert's `intermediate_size = moe_intermediate_size = 2048`
  - `intermediate_size = moe_intermediate_size * n_shared_experts = 2048` for shared expert

This is important because it means that shared expert is not 18432 wide, but only 2048 wide.

### 3.3 Main GEMM shapes per GPU under TP=4

For the current `TP=4`, the main matmul of the main model on each GPU can be approximately written as:

| Module | Local `K x N` | Weight type | Remarks |
|------|--------------|----------|------|
| `fused_qkv_a_proj` | `7168 x 2112` | BF16 | replicated |
| `q_b_proj` | `1536 x 3072` | BF16 | column parallel |
| `kv_b_proj` | `512 x 4096` | BF16 | column parallel |
| `o_proj` | `2048 x 7168` | BF16 | row parallel |
| `dense gate_up_proj` | `7168 x 9216` | NVFP4 | `2 * 18432 / 4` |
| `dense down_proj` | `4608 x 7168` | NVFP4 | `18432/4` |
| `expert gate_up_proj` | `7168 x 1024` | NVFP4 | `2 * 2048 / 4` |
| `expert down_proj` | `512 x 7168` | NVFP4 | `2048/4` |Among them:

-The attention weight is not quantized, so press **BF16 weight traffic**
- FFN/MoE weight is NVFP4, this article uses an approximate weight byte cost of **`0.5625 B/param`**

---

## 4. Theoretical FLOPs for each module

For GEMM `A[M,K] x B[K,N]`, according to the most common approximation:```text
FLOPs = 2 * M * K * N
```
Let’s first look at the matmul FLOPs of **per token/per GPU**, that is, let `M = 1`.

### 4.1 Attention projection

The four main projections combined:```text
fused_qkv_a_proj = 2 * 7168 * 2112   = 30,261,248
q_b_proj         = 2 * 1536 * 3072   =  9,437,184
kv_b_proj        = 2 *  512 * 4096   =  4,194,304
o_proj           = 2 * 2048 * 7168   = 29,376,512
---------------------------------------------------
attention total                        73,269,248 FLOPs/token/GPU
```
### 4.2 Layer 0 dense FFN```text
gate_up = 2 * 7168 * 9216 = 132,120,576
down    = 2 * 4608 * 7168 =  66,060,288
----------------------------------------
dense FFN                  198,180,864 FLOPs/token/GPU
```
### 4.3 Single routed/shared expert```text
gate_up = 2 * 7168 * 1024 = 14,680,064
down    = 2 *  512 * 7168 =  7,340,032
---------------------------------------
one expert                 22,020,096 FLOPs/token/GPU
```
Currently activated per token:

- `8` routed experts
- `1` shared expert

So the total FFN matmul for a single MoE layer is:```text
9 * 22,020,096 = 198,180,864 FLOPs/token/GPU
```
This is exactly the same as dense FFN.

### 4.4 Total FLOPs of single layer and whole model

Therefore:```text
per-layer total
= attention + FFN
= 73,269,248 + 198,180,864
= 271,450,112 FLOPs/token/GPU
```
Full model `61` layers:```text
61 * 271,450,112
= 16,558,456,832 FLOPs/token/GPU
= 0.01656 TFLOP/token/GPU
```
This is just the theoretical amount of backbone matmul, **not yet taken into account**:

- attention score/value kernel
-softmax
-RMSNorm
- routing/topk
- all-reduce / all-to-all / other communications
- KV cache read and write

---

## 5. Arithmetic Intensity modeling method

### 5.1 AI of a single GEMM

For an `A[M,K] x B[K,N]`, this article uses the simplest roofline approximation:```text
FLOPs = 2 * M * K * N
Bytes = a * (M*K + M*N) + w * (K*N)
AI    = FLOPs / Bytes
```
Among them:

- `a = 2 B` means activation is calculated in BF16
- `w = 2 B` means BF16 weight
- `w = 0.5625 B` means NVFP4 weight approximation

### 5.2 Critical batch `M*`

When `AI(M*) = 312.5 FLOP/B`, `M*` can be regarded as the theoretical inflection point when the module switches from bandwidth-bound to compute-bound.

---

## 6. Theoretical calculation intensity of each module

### 6.1 `M*` of a single kernel

| Module | `AI(∞)` | `M*_BF16-ridge` |
|------|---------|------------------|
| `fused_qkv_a_proj` | `1631.34` | `386.55` |
| `q_b_proj` | `1024.00` | `449.75` |
| `kv_b_proj` | `455.11` | `997.27` |
| `o_proj` | `1592.89` | `388.77` |
| `dense gate_up_proj` | `4032.00` | `95.27` |
| `dense down_proj` | `2804.87` | `98.91` |
| `expert gate_up_proj` | `896.00` | `134.96` |
| `expert down_proj` | `477.87` | `253.98` |

You can see it directly here:

- **The most difficult attention kernel to fill is `kv_b_proj`**
- **The most difficult expert kernel to fill up is `expert_down_proj`**
- Both dense GEMMs of NVFP4 FFN are relatively easy to enter compute-bound

### 6.2 AI aggregated by module

A closer approach to system modeling is to combine multiple kernels by module:

| Module | `M=1` | `M=8` | `M=16` | `M=32` | `M=64` | `M=128` | `M=256` | `M*=312.5` |
|------|-------|-------|--------|--------|--------|---------|---------|------------|
| Attention projection sum | `1.00` | `7.95` | `15.81` | `31.24` | `61.04` | `116.70` | `214.47` | **`409.24`** |
| Dense FFN sum | `3.55` | `28.22` | `55.98` | `110.21` | `213.73` | `402.99` | `718.95` | **`96.46`** |
| Sum of individual expert | `3.54` | `27.32` | `52.58` | `97.75` | `171.35` | `274.81` | `393.67` | **`159.95`** |
| Tier 0 overall | `2.10` | `16.72` | `33.21` | `65.52` | `127.59` | `242.45` | `440.89` | **`170.44`** |

The explanation of this table is:

- **If you only look at the 0th dense layer**, you will encounter the BF16 ridge at about `M ~= 170`
- **attention projection itself is more difficult to fill** and requires `M ~= 409`
- A single expert GEMM also requires `M ~= 160`

But Kimi’s real bottleneck is not that “an expert’s GEMM is too small”, but rather:

- During decoding, tokens in a batch are dispersed to many experts
- The actual `M_expert` received by each expert is often close to `1`So MoE occupancy also needs to be modeled separately.

---

## 7. MoE Occupancy Modeling

### 7.1 Why MoE needs to be calculated separately

Although a token will theoretically go through 8 routed + 1 shared experts, and the total FLOPs look the same as dense FFN, what is actually executed during decoding is not "a large FFN GEMM", but:

- Do routing first
- Then distribute the tokens in the batch to different experts
- Each expert makes a smaller GEMM

If the tokens in the batch are dispersed to many experts, then the `M` of a single expert will be very small, and the arithmetic strength will not increase.

### 7.2 Simplified occupancy model

For a global batch `M`, the total number of token slots of routed expert in the current configuration is:```text
slots = 8 * M
```
For `384` experts, using independent uniform landing point approximation, the number of active experts can be obtained:```text
E_active(M) ~= 384 * ( 1 - (1 - 1/384)^(8M) )
```
Then the average number of tokens for a single active expert is:```text
m_expert(M) ~= 8M / E_active(M)
```
Therefore, the AI of the MoE layer can no longer be estimated using "a large FFN", but must be estimated using:

- `attention(M)`
- `E_active` routed expert small GEMM
- `1` shared expert GEMM

Together we find the total `FLOPs / Bytes`.

### 7.3 AI of MoE layer and whole model

| `M` | `E_active` | Average `m_expert` | MoE layer AI | Full model AI |
|-----|------------|------------------|--------------|---------------|
| `1` | `7.9` | `1.01` | `2.11` | `2.11` |
| `4` | `30.7` | `1.04` | `4.00` | `4.04` |
| `8` | `59.0` | `1.08` | `4.85` | `4.91` |
| `16` | `109.0` | `1.17` | `5.72` | `5.79` |
| `32` | `187.0` | `1.37` | `6.96` | `7.06` |
| `64` | `283.0` | `1.81` | `9.37` | `9.52` |
| `128` | `357.4` | `2.87` | `14.87` | `15.10` |
| `256` | `382.2` | `5.36` | `27.43` | `27.86` |
| `512` | `384.0` | `10.67` | `52.80` | `53.62` |
| `8192` | `384.0` | `170.67` | `423.40` | `429.03` |

Using the same method to find ridge crossing, we can get:

- **MoE layer**: `M ~= 4659`
- **Full model**: `M ~= 4554`

This result is actually the core conclusion of this article:

- **Kimi’s dense first layer is not difficult to eat**
- **What really keeps the entire model in the memory-bound range is that MoE expert occupancy is too broken during decoding**

---

## 8. Map to current `pd_kimi_bench_a_70k`

The key scheduling parameters of this benchmark come from:

- `vigil/examples/pd_kimi_bench_a_70k.yaml`
- `workload_analysis/workload_analysis_pd_kimi_70k.md`
- Actual prefill/decode logs from 2026-04-11

### 8.1 Current configuration

| Side | Key Parameters | Value |
|----|----------|----|
| Prefill | `max_num_batched_tokens` | `8192` |
| Prefill | `max_num_seqs` | `16` |
| Decode | `max_num_batched_tokens` | `8192` |
| Decode | `max_num_seqs` | `16` |
| Both sides | `tensor_parallel_size` | `4` |
| Both sides | `pipeline_parallel_size` | `1` |

Among them:

- The prefill side log can directly see `BatchDescriptor(num_tokens=8192, ...)`
- The decode side is a token-level step, and the theoretical upper limit is mainly constrained by `max_num_seqs = 16`

### 8.2 Current operating point

Plot these batch points onto the AI curve above:

| Scenario | Represents `M` | Full model AI | Relationship with ridge `312.5` |
|------|----------|---------------|--------------------------|
| small decode | `4` | `4.04` | **much lower than ridge** |
| decode upper limit | `16` | `5.79` | **much lower than ridge** |
| Medium prefill | `128` | `15.10` | **much lower than ridge** |
| Current prefill chunk | `8192` | `429.03` | **Above ridge** |So you can get it directly:

1. **Decode is almost impossible to enter compute-bound based on the current batch**
   Even if `max_num_seqs` is raised from `16` to `64`, full-model AI is only `9.52 FLOP/B`; raising `128` is only `15.10 FLOP/B`. It is still more than an order of magnitude away from `312.5`.

2. **Prefill’s large chunk is already large enough**
   When `M = 8192`, the main model has entered the compute-bound region from the pure GEMM roofline. Continuing to increase the prefill batch will not significantly increase the arithmetic intensity like a small batch.

3. **Therefore the optimization directions of prefill and decode should be separated**
   - prefill is more like **chunk number, scheduling, KV/activation/compilation overhead** issues
   - decode is more like **MoE small GEMM, poor weight reuse, HBM traffic** problems

---

## 9. Direct inspiration for current systems

### 9.1 Decode side

If the problem is slow decoding, then look at this modeling:

- **Simply increasing `max_num_seqs` will not make Kimi decode hit the computing power wall quickly**
- The current decode is still deeply **memory-bound**
- The main contradiction is more likely to be:
  - MoE expert occupancy fragmentation
  - NVFP4 weighted traffic
  - attention / MoE kernel launch and scheduling overhead
  - Communication and KV reading and writing

Therefore, the direction worth looking at on the decode side is:

- expert packing/grouped dispatch
- Higher expert occupancy
- Less MoE launch/dispatch overhead
- Whether speculative decode really improves target-side effective occupancy

### 9.2 Prefill side

If the problem is that prefill is slow, then look at the roofline:

- `8192` is big enough
- Further increase the income of `max_num_batched_tokens`, more may come from:
  - Fewer chunks
  - Fewer scheduler rotations
  - Lower per-chunk fixed overhead

Instead of continuing to improve AI to gain more computing power.

This is consistent with your previous workload analysis:

- `70K` tokens/turn requires about `9` prefill chunks under `8192` chunk
- So the bottleneck of prefill has probably shifted from "not strong enough" to "too many chunks"

---

## 10. Modeling boundaries and considerations

This document is a **backbone GEMM + HBM roofline** model that deliberately ignores many real system factors. The following expenses have not yet been taken into account:

-attention score / value kernel
- softmax/norm/activation
- MoE routing and topk
- TP all-reduce
- expert parallel/all-to-all
- KV cache read and write
- CUDA Graph / compile shape constraints
- kernel fusion and launch overhead

Therefore:

- `M*` here should be understood as **"the first turning point in the pure roofline sense"**
- It should not be understood as "the real system will definitely no longer increase in speed by this batch"
- Especially in decoding scenarios, the real bottleneck is usually more memory-bound than this model.

---

## 11. Final summaryFor the current `Kimi-K2.5-NVFP4` main model on `GB200/B200 + TP=4`:

- Single GPU machine roofline takes **`2.5 PFLOPS BF16 dense`** and **`8 TB/s HBM`**
- Corresponding to **ridge point = `312.5 FLOP/B`**
- The overall roofline inflection point of **layer 0 dense layer** is at **`M ~= 170`**
- The inflection point of **attention projection sum** is at **`M ~= 409`**
- **Real Whole Model** Because of MoE expert occupancy dispersion, the inflection point is postponed to **`M ~= 4554`**

Map to current benchmark:

- **decode (`M <= 16`) obviously memory-bound**
- **prefill (`M = 8192`) has entered the compute-bound area**

Therefore, one of the most practical judgments is:

- **Kimi’s main model currently does not lack prefill batch strength, but what it lacks is effective expert occupancy and lower memory traffic on the decode side**