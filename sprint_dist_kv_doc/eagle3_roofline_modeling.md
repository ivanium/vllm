# Kimi K2.5 Thinking-Eagle3 Roofline Modeling

> Goal: Estimate the batch size at which `Thinking-Eagle3` draft/MTP calculations will switch from HBM bandwidth limited to computing power limited on the current GB200/B200 machine, which is the theoretical upper limit of "continuing to increase the batch size will basically have no roofline benefits".

---

## 1. Let’s look at the modeling objects and conclusions first.

The draft checkpoint we are referring to this time is:

- `/mnt/lustre/hf-models/hub/models--nvidia--Kimi-K2.5-Thinking-Eagle3/snapshots/0b0c6ac039089ad2c2418c91c039553381a302d9`

From `config.json` you can read directly:

- `architectures = Eagle3DeepseekV2ForCausalLM`
- `torch_dtype = bfloat16`
- `num_hidden_layers = 1`
- `hidden_size = 7168`
- `intermediate_size = 18432`
- `q_lora_rank = 1536`
- `kv_lora_rank = 512`
- `qk_nope_head_dim = 128`
- `qk_rope_head_dim = 64`
- `v_head_dim = 128`
- `num_attention_heads = 64`
- `eagle_aux_hidden_state_layer_ids = [1, 29, 57]`

Under the current assumptions of `TP=4`, `B200 HBM≈8 TB/s`, `BF16 dense tensor≈2.5 PFLOPS/GPU`:

- **The overall roofline inflection point of Eagle3 step** is approximately **`M ~= 360 tokens`**
- If `fc(aux hidden state combine)` is also included in the first draft start, the inflection point is approximately **`M ~= 345 tokens`**
- If **each master GEMM is required to reach the ridge point** individually, the most conservative bottleneck is `kv_b_proj`, which requires **`M ~= 998 tokens`**

So a more practical conclusion is:

- **Optimistic**: After `MTP batch size` reaches around **`~350`**, the overall calculation will be as strong as BF16 roofline
- **Conservative**: It needs to be around **`~1000`**, even the most difficult `kv_b_proj` has only reached the ridge point

In the current experimental configuration:

- `max_num_seqs = 16`
- `num_speculative_tokens = 3`
- `parallel_drafting = False` (vLLM default and not explicitly turned on for the current configuration)

Therefore, the actual batch limit of the current Eagle3 draft GEMM is basically only `16`**; even if the speculative positions are completely combined into `16 × 3 = 48` in the future, it is still far lower than `~350`. In other words, **the main GEMM of Thinking-Eagle3 under the current configuration is obviously still in the HBM bandwidth dominant range**.

---

## 2. Machine Roofline parameters

### 2.1 HBM bandwidth

It has been recorded in the local machine documentation:

- Single GPU HBM bandwidth modeled at **`8 TB/s`**

Reference:

- `vllm/sprint_dist_kv_doc/machine_doc.md`

### 2.2 B200 theoretical computing power

The `GB200 Grace Blackwell Superchip` specifications given on NVIDIA’s official `GB200 NVL72` page are:- `FP16/BF16 Tensor Core: 10 PFLOPS`
- `FP8/FP6 Tensor Core: 20 PFLOPS`
- `GPU Memory Bandwidth: 16 TB/s`

But the footer on the page states:

- `Specification in sparse. Dense is one-half sparse spec shown.`

So scaling to **dense** and dividing by **2 GPU / superchip**, we get per GPU:

| Metrics | per Superchip | dense per Superchip | dense per GPU |
|------|--------------|--------------------|--------------|
| BF16 Tensor | 10 PFLOPS | 5 PFLOPS | **2.5 PFLOPS** |
| FP8 Tensor | 20 PFLOPS | 10 PFLOPS | **5.0 PFLOPS** |
| HBM Bandwidth | 16 TB/s | 16 TB/s | **8.0 TB/s** |

Official source:

- https://www.nvidia.com/en-us/data-center/gb200-nvl72/

### 2.3 Ridge Point

Defined by roofline:```text
ridge_point = peak_compute / peak_bandwidth
```
Get:

| Accuracy | Peak computing power | HBM bandwidth | ridge point |
|------|----------|----------|-------------|
| BF16 dense | 2.5e15 FLOP/s | 8.0e12 B/s | **312.5 FLOP/B** |
| FP8 dense | 5.0e15 FLOP/s | 8.0e12 B/s | **625 FLOP/B** |

Because the checkpoint configuration of `Thinking-Eagle3` is `torch_dtype=bfloat16` and there is no quantitative configuration, the main analysis of this article is based on **BF16 ridge = 312.5 FLOP/B**.

---

## 3. Draft Layer structure and GEMM shape under TP4

### 3.1 Why is this “1-layer Eagle3”

In `config.json` of `Thinking-Eagle3`:

- `num_hidden_layers = 1`

At the same time vLLM's `DeepseekV2Eagle3DecoderLayer` uses:

- The input dimension of the first layer of attention is `2 * hidden_size`, because `embed + hidden_state` will be put together
- MLP is still dense `DeepseekV2MLP`
- There is an additional `fc`, which concatenates the three aux hidden states returned by the target model and presses them back to the draft hidden size

### 3.2 Current draft/MTP batch definition

The `MTP batch size = M` here refers to the number of token rows actually processed simultaneously by **one draft GEMM**.

Under current configuration:

- `parallel_drafting = False`
- So in each speculative step, the draft model only processes the "current number of active requests" tokens each time
- Therefore, the effective upper limit of `M` in the current experiment is basically `max_num_seqs = 16`

If it is changed to parallel drafting in the future, it will be closer:```text
M_effective ~= active_requests × num_speculative_tokens
```
---

## 4. Main GEMM list under TP4

Based on vLLM source code:

- `DeepSeekV2FusedQkvAProjLinear(..., disable_tp=True)` → replicated
- `q_b_proj`, `kv_b_proj` → column parallel
- `o_proj`, `down_proj` → row parallel
- `gate_up_proj` → merged column parallel
- `fc` → replicated

For the current `TP=4`, the main GEMM on each GPU can be approximately written as:

| Module | Local GEMM Shape (K x N) | Notes |
|------|------------------------|------|
| `fused_qkv_a_proj` | `14336 x 2112` | The first layer input is `2 * hidden_size`, replicated |
| `q_b_proj` | `1536 x 3072` | `64 * 192 / 4` |
| `kv_b_proj` | `512 x 4096` | `64 * (128+128) / 4` |
| `o_proj` | `2048 x 7168` | `64 * 128 / 4 = 2048` |
| `gate_up_proj` | `7168 x 9216` | local output after merged `2 * 18432 / 4` |
| `down_proj` | `4608 x 7168` | `18432 / 4 = 4608` |
| `fc(aux combine)` | `21504 x 7168` | `3 * target_hidden_size -> hidden_size`, replicated |

After adding up all these GEMMs, the total GEMM calculation amount per GPU per token is approximately:```text
F_token ~= 610,009,088 FLOPs / token / GPU
```
The corresponding total weight reading volume is approximately:```text
W_bytes ~= 610,009,088 bytes / GPU
        ~= 0.568 GiB / GPU
```
Therefore, when `M=1`, the overall AI is very close to `1 FLOP/B`, which is also the fundamental reason why small batch drafts are easily memory-bound.

---

## 5. Arithmetic Intensity formula

For any GEMM:```text
A: [M, K]
B: [K, N]
C: [M, N]
```
Using the simplest roofline approximation:

- FLOPs: `2 * M * K * N`
- Bytes: `s * (M*K + K*N + M*N)`
- where `s = 2 bytes` (BF16)

Then:```text
AI(M) = 2 M K N / ( s (M K + K N + M N) )
```
When `AI(M) = ridge_point`, the critical batch can be solved:```text
M* = ridge * s * K * N / ( 2 K N - ridge * s * (K + N) )
```
If the denominator is less than or equal to 0, it means that the module cannot touch the ridge point even if `M -> inf`.

---

## 6. `M*` of each main module

### 6.1 BF16 ridge: `312.5 FLOP/B`

| Module | `AI(∞)` | `M*_BF16` |
|------|---------|-----------|
| `fused_qkv_a_proj` | 1840.8 | **376** |
| `q_b_proj` | 1024.0 | **450** |
| `kv_b_proj` | 455.1 | **998** |
| `o_proj` | 1592.9 | **389** |
| `gate_up_proj` | 4032.0 | **339** |
| `down_proj` | 2804.9 | **352** |
| `fc(aux combine)` | 5376.0 | **332** |

Conclusion:

- From the most conservative perspective of a single module, the bottleneck is `kv_b_proj`
- So in order for all main GEMMs to reach the BF16 ridge, the batch size needs to be at least **`~1000`**

### 6.2 FP8 ridge: `625 FLOP/B`

| Module | `M*_FP8` |
|------|----------|
| `fused_qkv_a_proj` | **946** |
| `q_b_proj` | **1604** |
| `kv_b_proj` | **Unreachable** (`AI(∞)=455.1 < 625`) |
| `o_proj` | **1029** |
| `gate_up_proj` | **740** |
| `down_proj` | **804** |
| `fc(aux combine)` | **707** |

This shows that if roofline is upgraded to the FP8 level in the future, then just looking at these GEMMs, `kv_b_proj` will always be memory-bound.

---

## 7. Roofline inflection point of the overall Step

### 7.1 Count `fc` into the first draft start

Adding up all the above 7 GEMMs, the overall AI is approximately:

| `M` | `AI_total(M)` |
|-----|---------------|
| 16 | 15.92 |
| 32 | 31.70 |
| 48 | 47.32 |
| 64 | 62.79 |
| 128 | 123.26 |
| 256 | 237.70 |
| 345 | **312.57** |
| 512 | 443.69 |

Therefore:

- **The overall inflection point for the first draft is about `M ~= 345`**

### 7.2 Only look at recurrent Eagle3 draft step (excluding `fc`)

If you remove `fc(aux combine)`, you only see the draft steps that will actually be executed repeatedly:

| `M` | `AI_step(M)` |
|-----|--------------|
| 16 | 15.89 |
| 32 | 31.58 |
| 48 | 47.06 |
| 64 | 62.33 |
| 128 | 121.50 |
| 256 | 231.26 |
| 360 | **312.93** |

Therefore:

- **The inflection point of the overall recurrent draft step is approximately `M ~= 360`**

---

## 8. The meaning of the current experimental configuration

Current `pd_kimi_bench_a_70k` configuration:

- `TP=4`
- `max_num_seqs=16`
- `num_speculative_tokens=3`
- `parallel_drafting=False`So the current judgment closest to the real execution path is:

### 8.1 The current actual draft batch is very small

Because `parallel_drafting=False`, the effective batch of each speculative step is basically:```text
M_actual <= active_decode_requests <= max_num_seqs = 16
```
At this time, the overall AI is only:```text
AI_total(16) ~= 15.9 FLOP/B
```
Relative to BF16 ridge:```text
15.9 / 312.5 ~= 5.1%
```
In other words, the main GEMM of the current Eagle3 draft is still far from being "full of computing power", and it is obviously dominated by HBM/weight traffic**.

### 8.2 Even the complete merging of speculative positions is not enough

If the three speculative positions are completely combined in the future, the ideal approximation will be:```text
M_effective ~= 16 * 3 = 48
```
So:```text
AI_total(48) ~= 47.3 FLOP/B
```
It only reaches BF16 ridge:```text
47.3 / 312.5 ~= 15.1%
```
Therefore, from a roofline perspective, the current configuration is still very far from the range of "there will be no benefit if the batch is increased further"**.

---

## 9. Theoretical throughput upper limit

If we still only look at these 7 main GEMMs and ignore overheads such as attention/KV reading and writing, norm, softmax, collective, kernel launch, etc., then:

### 9.1 Compute ceiling```text
tokens/s_compute_ceiling
  = peak_compute / FLOPs_per_token
  = 2.5e15 / 610,009,088
  ≈ 4.10e6 tokens/s/GPU
```
This is just a very loose upper bound on the roofline and should not be taken as a true reachability value.

### 9.2 Roofline lower bound on time

For batch `M`:```text
t_min(M) = max( FLOPs(M) / P_peak, Bytes(M) / BW_peak )
```
Under this simplified model:

| `M` | Dominant term | roofline throughput upper bound |
|-----|--------|-------------------|
| 1 | memory | ~13.1 K tok/s/GPU |
| 16 | memory | ~208.8 K tok/s/GPU |
| 48 | memory | ~620.5 K tok/s/GPU |
| 128 | memory | ~1.62 M tok/s/GPU |
| 345 | compute | ~4.10 M tok/s/GPU |
| 512 | compute | ~4.10 M tok/s/GPU |

So it can be seen intuitively from this approximate model:

- When `M < ~350`, increasing the batch size is mainly to increase weight reuse, and the benefits come from **increasing computing intensity**
- After `M >= ~350`, the overall main GEMM starts hitting the **BF16 compute roofline**
- Increasing the batch size further will theoretically not bring the same level of benefits to these main GEMMs.

---

## 10. How to explain “no profit”

The "no benefit" here should be understood at two levels:

1. **Overall Level**
   When the overall `AI_total(M)` reaches the ridge point, continue to increase the batch size without changing the roofline dominant term, and the benefits will become significantly smaller.

2. **Module-by-module level**
   If you require every critical GEMM to go from bandwidth-bound to compute-bound, look at the slowest module.
   In the current `Thinking-Eagle3`, the conservative bottleneck is `kv_b_proj`, which is about `M ~= 998`.

Therefore, a more secure expression is:

- **Overall no-benefit point**: `M ~= 350`
- **The most conservative fully-compute-bound point**: `M ~= 1000`

---

## 11. Limitations

This document is an **optimistic** roofline modeling that has not yet taken into account:

- MLA attention's own QK/AV kernel
- KV cache read and write and HBM access
- CUDA graph/launch/scheduler overhead
- TP all-reduce / all-gather communication
- Runtime factors such as prefix cache, paged attention, fragmentation, etc.

Therefore:

- The `M ~= 350` given here should be more like the optimistic lower bound where the main GEMM starts to hit the BF16 ridge**
- The true system-level point of "there is no benefit in increasing the batch size" is usually **bigger, not smaller**

---

## 12. Final Conclusion

For current `Kimi-K2.5-Thinking-Eagle3` on `GB200/B200 + TP4`:

- **It is recommended to regard `M ~= 350` as the first inflection point of the overall GEMM roofline**
- **It is recommended to consider `M ~= 1000` as a conservative upper limit**
- The actual `M <= 16` under the current experimental configuration, even if idealized and speculative positions are only `M ~= 48`
- So **The current Eagle3/MTP is far from reaching the range of "there will be no profit if the batch is increased"**

In other words, looking at the roofline alone, the current MTP/Eagle3 is still deep in the memory-bound range.