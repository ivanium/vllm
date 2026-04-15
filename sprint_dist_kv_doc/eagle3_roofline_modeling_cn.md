# Kimi K2.5 Thinking-Eagle3 Roofline Modeling

> 目标: 估算在当前 GB200/B200 机器上，`Thinking-Eagle3` draft/MTP 计算在多大 batch size 下会从 HBM 带宽受限切换到算力受限，也就是“继续增大 batch 基本没有 roofline 意义上的收益”的理论上限。

---

## 1. 建模对象与结论先看

这次我们参考的 draft checkpoint 是:

- `/mnt/lustre/hf-models/hub/models--nvidia--Kimi-K2.5-Thinking-Eagle3/snapshots/0b0c6ac039089ad2c2418c91c039553381a302d9`

从 `config.json` 可以直接读到:

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

在当前 `TP=4`、`B200 HBM≈8 TB/s`、`BF16 dense tensor≈2.5 PFLOPS/GPU` 的假设下:

- **整体 Eagle3 step 的 roofline 拐点**大约在 **`M ~= 360 tokens`**
- 如果把 `fc(aux hidden state combine)` 也算进第一次 draft 启动，则拐点大约在 **`M ~= 345 tokens`**
- 如果要求**每个主 GEMM 都 individually 到达 ridge point**，最保守瓶颈是 `kv_b_proj`，需要 **`M ~= 998 tokens`**

所以一个比较实用的结论是:

- **乐观口径**: `MTP batch size` 到 **`~350`** 左右后，整体算强基本摸到 BF16 roofline
- **保守口径**: 要到 **`~1000`** 左右，连最难吃满的 `kv_b_proj` 也才到 ridge point

而当前实验配置里:

- `max_num_seqs = 16`
- `num_speculative_tokens = 3`
- `parallel_drafting = False`（vLLM 默认值，且当前配置没有显式打开）

所以当前 Eagle3 draft GEMM 的**实际 batch 上限基本只有 `16`**；即使未来把 speculative positions 完全并起来看成 `16 × 3 = 48`，也还远低于 `~350`。也就是说，**当前配置下 Thinking-Eagle3 的主 GEMM 明显仍处在 HBM 带宽主导区间**。

---

## 2. 机器 Roofline 参数

### 2.1 HBM 带宽

本地机器文档里已经记录:

- 单 GPU HBM 带宽按 **`8 TB/s`** 建模

参考:

- `vllm/sprint_dist_kv_doc/machine_doc_cn.md`

### 2.2 B200 理论算力

NVIDIA 官方 `GB200 NVL72` 页面给出的 `GB200 Grace Blackwell Superchip` 规格是:

- `FP16/BF16 Tensor Core: 10 PFLOPS`
- `FP8/FP6 Tensor Core: 20 PFLOPS`
- `GPU Memory Bandwidth: 16 TB/s`

但页面脚注说明:

- `Specification in sparse. Dense is one-half sparse spec shown.`

因此换算到 **dense** 且再除以 **2 GPU / superchip**，得到每 GPU:

| 指标 | 每 Superchip | dense 每 Superchip | dense 每 GPU |
|------|--------------|--------------------|--------------|
| BF16 Tensor | 10 PFLOPS | 5 PFLOPS | **2.5 PFLOPS** |
| FP8 Tensor | 20 PFLOPS | 10 PFLOPS | **5.0 PFLOPS** |
| HBM 带宽 | 16 TB/s | 16 TB/s | **8.0 TB/s** |

官方来源:

- https://www.nvidia.com/en-us/data-center/gb200-nvl72/

### 2.3 Ridge Point

按 roofline 定义:

```text
ridge_point = peak_compute / peak_bandwidth
```

得到:

| 精度 | 峰值算力 | HBM 带宽 | ridge point |
|------|----------|----------|-------------|
| BF16 dense | 2.5e15 FLOP/s | 8.0e12 B/s | **312.5 FLOP/B** |
| FP8 dense | 5.0e15 FLOP/s | 8.0e12 B/s | **625 FLOP/B** |

因为 `Thinking-Eagle3` 的 checkpoint 配置是 `torch_dtype=bfloat16`，且没有量化配置，所以本文主分析以 **BF16 ridge = 312.5 FLOP/B** 为准。

---

## 3. Draft Layer 结构与 TP4 下的 GEMM 形状

### 3.1 为什么这是 “1 层 Eagle3”

`Thinking-Eagle3` 的 `config.json` 里:

- `num_hidden_layers = 1`

同时 vLLM 的 `DeepseekV2Eagle3DecoderLayer` 使用:

- 第一层 attention 输入维度是 `2 * hidden_size`，因为会把 `embed + hidden_state` 拼起来
- MLP 仍然是 dense `DeepseekV2MLP`
- 额外还有一个 `fc`，把 target model 返回的 3 个 aux hidden states 拼接后压回 draft hidden size

### 3.2 当前 draft/MTP 的 batch 定义

这里的 `MTP batch size = M`，指的是**一次 draft GEMM 实际同时处理的 token 行数**。

在当前配置下:

- `parallel_drafting = False`
- 所以每个 speculative step 里，draft model 每次只处理“当前活跃请求数”个 token
- 因此当前实验的有效 `M` 上限基本就是 `max_num_seqs = 16`

如果未来改成 parallel drafting，才更接近:

```text
M_effective ~= active_requests × num_speculative_tokens
```

---

## 4. TP4 下的主 GEMM 列表

基于 vLLM 源码:

- `DeepSeekV2FusedQkvAProjLinear(..., disable_tp=True)` → replicated
- `q_b_proj`, `kv_b_proj` → column parallel
- `o_proj`, `down_proj` → row parallel
- `gate_up_proj` → merged column parallel
- `fc` → replicated

对当前 `TP=4`，每 GPU 上的主要 GEMM 可近似写成:

| 模块 | 本地 GEMM 形状 (K x N) | 备注 |
|------|------------------------|------|
| `fused_qkv_a_proj` | `14336 x 2112` | 第一层输入是 `2 * hidden_size`，replicated |
| `q_b_proj` | `1536 x 3072` | `64 * 192 / 4` |
| `kv_b_proj` | `512 x 4096` | `64 * (128+128) / 4` |
| `o_proj` | `2048 x 7168` | `64 * 128 / 4 = 2048` |
| `gate_up_proj` | `7168 x 9216` | merged 后本地输出 `2 * 18432 / 4` |
| `down_proj` | `4608 x 7168` | `18432 / 4 = 4608` |
| `fc(aux combine)` | `21504 x 7168` | `3 * target_hidden_size -> hidden_size`，replicated |

把这些 GEMM 全部加总后，每 GPU 每 token 的总 GEMM 计算量约为:

```text
F_token ~= 610,009,088 FLOPs / token / GPU
```

对应总权重读取量约为:

```text
W_bytes ~= 610,009,088 bytes / GPU
        ~= 0.568 GiB / GPU
```

所以在 `M=1` 时，整体 AI 非常接近 `1 FLOP/B`，这也是小 batch draft 很容易强 memory-bound 的根本原因。

---

## 5. Arithmetic Intensity 公式

对任意一个 GEMM:

```text
A: [M, K]
B: [K, N]
C: [M, N]
```

采用最简单的 roofline 近似:

- FLOPs: `2 * M * K * N`
- Bytes: `s * (M*K + K*N + M*N)`
- 其中 `s = 2 bytes`（BF16）

则:

```text
AI(M) = 2 M K N / ( s (M K + K N + M N) )
```

当 `AI(M) = ridge_point` 时，可以解出临界 batch:

```text
M* = ridge * s * K * N / ( 2 K N - ridge * s * (K + N) )
```

若分母小于等于 0，则说明该模块即使 `M -> inf` 也碰不到该 ridge point。

---

## 6. 各主模块的 `M*`

### 6.1 BF16 ridge: `312.5 FLOP/B`

| 模块 | `AI(∞)` | `M*_BF16` |
|------|---------|-----------|
| `fused_qkv_a_proj` | 1840.8 | **376** |
| `q_b_proj` | 1024.0 | **450** |
| `kv_b_proj` | 455.1 | **998** |
| `o_proj` | 1592.9 | **389** |
| `gate_up_proj` | 4032.0 | **339** |
| `down_proj` | 2804.9 | **352** |
| `fc(aux combine)` | 5376.0 | **332** |

结论:

- 从**单模块最保守**口径看，瓶颈是 `kv_b_proj`
- 所以要让所有主 GEMM 都达到 BF16 ridge，batch 至少需要 **`~1000`**

### 6.2 FP8 ridge: `625 FLOP/B`

| 模块 | `M*_FP8` |
|------|----------|
| `fused_qkv_a_proj` | **946** |
| `q_b_proj` | **1604** |
| `kv_b_proj` | **无法达到** (`AI(∞)=455.1 < 625`) |
| `o_proj` | **1029** |
| `gate_up_proj` | **740** |
| `down_proj` | **804** |
| `fc(aux combine)` | **707** |

这说明如果将来把 roofline 提升到 FP8 档位，那么单看这些 GEMM，`kv_b_proj` 永远还是 memory-bound。

---

## 7. 整体 Step 的 Roofline 拐点

### 7.1 把 `fc` 也算入第一次 draft 启动

把上面的 7 个 GEMM 全部相加，整体 AI 近似为:

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

因此:

- **整体首次 draft 启动的拐点约为 `M ~= 345`**

### 7.2 只看 recurrent Eagle3 draft step（不含 `fc`）

如果把 `fc(aux combine)` 去掉，只看真正会重复执行的 draft step:

| `M` | `AI_step(M)` |
|-----|--------------|
| 16 | 15.89 |
| 32 | 31.58 |
| 48 | 47.06 |
| 64 | 62.33 |
| 128 | 121.50 |
| 256 | 231.26 |
| 360 | **312.93** |

因此:

- **整体 recurrent draft step 的拐点约为 `M ~= 360`**

---

## 8. 对当前实验配置的含义

当前 `pd_kimi_bench_a_70k` 配置:

- `TP=4`
- `max_num_seqs=16`
- `num_speculative_tokens=3`
- `parallel_drafting=False`

所以当前最接近真实执行路径的判断是:

### 8.1 当前实际 draft batch 很小

因为 `parallel_drafting=False`，每个 speculative step 的有效 batch 基本就是:

```text
M_actual <= active_decode_requests <= max_num_seqs = 16
```

这时整体 AI 只有:

```text
AI_total(16) ~= 15.9 FLOP/B
```

相对 BF16 ridge:

```text
15.9 / 312.5 ~= 5.1%
```

也就是说，**当前 Eagle3 draft 的主 GEMM 离“算力吃满”还差得很远，明显是 HBM/weight traffic 主导**。

### 8.2 即使把 speculative positions 完全并起来也不够

如果未来把 3 个 speculative positions 完全并起来，理想化地近似成:

```text
M_effective ~= 16 * 3 = 48
```

那么:

```text
AI_total(48) ~= 47.3 FLOP/B
```

也只达到 BF16 ridge 的:

```text
47.3 / 312.5 ~= 15.1%
```

所以从 roofline 视角看，**当前这套配置离“batch 再增大就没收益”的区间还非常远**。

---

## 9. 理论吞吐上限

如果仍只看这 7 个主 GEMM，忽略 attention/KV 读写、norm、softmax、collective、kernel launch 等开销，则:

### 9.1 Compute ceiling

```text
tokens/s_compute_ceiling
  = peak_compute / FLOPs_per_token
  = 2.5e15 / 610,009,088
  ≈ 4.10e6 tokens/s/GPU
```

这只是一个**极松的 roofline 上界**，不应当被当作真实可达值。

### 9.2 Roofline lower bound on time

对 batch `M`:

```text
t_min(M) = max( FLOPs(M) / P_peak, Bytes(M) / BW_peak )
```

在这个简化模型下:

| `M` | 主导项 | roofline 吞吐上界 |
|-----|--------|-------------------|
| 1 | memory | ~13.1 K tok/s/GPU |
| 16 | memory | ~208.8 K tok/s/GPU |
| 48 | memory | ~620.5 K tok/s/GPU |
| 128 | memory | ~1.62 M tok/s/GPU |
| 345 | compute | ~4.10 M tok/s/GPU |
| 512 | compute | ~4.10 M tok/s/GPU |

所以从这个近似模型可以很直观看到:

- `M < ~350` 时，增大 batch 主要是在提升权重复用，收益来自 **提升计算强度**
- `M >= ~350` 后，整体主 GEMM 开始撞上 **BF16 compute roofline**
- 再往上增大 batch，理论上就不会再给这些主 GEMM 带来同等级别的收益

---

## 10. 如何解释 “没有收益”

这里的“没有收益”要分两层理解:

1. **整体层面**
   当整体 `AI_total(M)` 达到 ridge point 后，继续增大 batch，不再改变 roofline 主导项，收益会明显变小。

2. **逐模块层面**
   如果要求每个关键 GEMM 都从 bandwidth-bound 变成 compute-bound，则要看最慢模块。
   在当前 `Thinking-Eagle3` 里，保守瓶颈是 `kv_b_proj`，大约要到 `M ~= 998`。

因此更稳妥的表达是:

- **整体意义上的 no-benefit 点**: `M ~= 350`
- **最保守的 fully-compute-bound 点**: `M ~= 1000`

---

## 11. 局限性

本文档是一个**偏乐观**的 roofline 建模，尚未计入:

- MLA attention 本身的 QK/AV kernel
- KV cache 读写与 HBM 访问
- CUDA graph / launch / scheduler 开销
- TP all-reduce / all-gather 通信
- prefix cache、paged attention、碎片化等运行时因素

因此:

- 这里给出的 `M ~= 350` 应更像是**主 GEMM 开始撞到 BF16 ridge 的 optimistic lower bound**
- 真正系统级的“batch 再加大没有收益”的点，通常只会 **更大，不会更小**

---

## 12. 最终结论

对当前 `Kimi-K2.5-Thinking-Eagle3` 在 `GB200/B200 + TP4` 上:

- **推荐把 `M ~= 350` 视为整体 GEMM roofline 的第一拐点**
- **推荐把 `M ~= 1000` 视为保守上限**
- 当前实验配置下实际 `M <= 16`，即使理想化并 speculative positions 也只有 `M ~= 48`
- 所以 **当前 Eagle3/MTP 远没有达到“batch 再增大就没收益”的区间**

换句话说，单从 roofline 看，**当前 MTP/Eagle3 仍然深处 memory-bound 区间**。

