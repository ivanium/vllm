# Kimi-K2.5-NVFP4 主模型 Roofline 建模

> 目标: 从主模型 `nvidia/Kimi-K2.5-NVFP4` 出发，估算它在当前 `GB200/B200 + TP=4` 机器上的理论计算强度，判断在什么 batch 规模下会从 HBM 带宽受限切到算力受限，并把这个结论映射到当前 `pd_kimi_bench_a_70k` 的 `prefill/decode` 配置。

---

## 1. 结论先看

先给出这次最重要的几个结论:

- **机器 roofline**: 按当前机器文档和 NVIDIA 官方 `GB200 NVL72` 规格页，单 GPU 取 **`HBM = 8 TB/s`**、**`BF16 dense = 2.5 PFLOPS`**，所以 **ridge point = `312.5 FLOP/B`**。
- **主模型结构**: `Kimi-K2.5-NVFP4` 的主干是 **`61` 层**，其中 **第 `0` 层是 dense FFN**，其余 **`60` 层是 MoE**。`self_attn` 权重保持 **BF16**，MoE/FFN Linear 权重走 **NVFP4**。
- **单层 matmul FLOPs 很整齐**: 在 `TP=4` 下，
  - attention 投影部分约 **`73.27M FLOPs/token/GPU`**
  - dense FFN 约 **`198.18M FLOPs/token/GPU`**
  - MoE 层里 `8 routed + 1 shared` 合起来也约 **`198.18M FLOPs/token/GPU`**
  - 所以任意一层的 matmul 总量都约 **`271.45M FLOPs/token/GPU`**
- **如果只看 dense first layer**，整体 AI 在 **`M ~= 170`** 左右就能摸到 BF16 roofline。
- **但如果看真实的整模型**，MoE token 会分散到很多 expert，导致 decode 时专家 GEMM 很难做大。用 occupancy 近似后，**全模型的 ridge crossing 在 `M ~= 4554 tokens/step`**，远大于 decode 现实可达范围。
- **当前 benchmark 的 operating point**:
  - decode: `max_num_seqs = 16`，全模型 AI 只有 **`~5.8 FLOP/B`**
  - prefill: `max_num_batched_tokens = 8192`，全模型 AI 约 **`429 FLOP/B`**
  - 也就是说，**当前 Kimi 主模型 decode 明显是 HBM/权重流量主导，而 prefill 大 chunk 已经进入 compute-bound 区间附近甚至之上**

换句话说:

- 如果目标是优化 **decode**，主矛盾不是“batch 再加一点就能吃满算力”，而是 **MoE 小 GEMM + weight traffic + routing 分散**。
- 如果目标是优化 **prefill**，`8192` 这个 chunk size 从纯 roofline 看已经不低了，再增大 batch 的边际收益更多会体现在 **减少 chunk 数和调度开销**，而不是继续提升主 GEMM 的算术强度。

---

## 2. 机器 Roofline 参数

### 2.1 HBM 带宽

本地机器文档中已经记录:

- 机型: `NVIDIA GB200 NVL72`
- 单 GPU HBM: `192 GB HBM3e`
- 单 GPU HBM 带宽按 **`8 TB/s`** 建模

本地参考:

- `vllm/sprint_dist_kv_docs/doc.md`

### 2.2 GB200/B200 理论算力

NVIDIA 官方 `GB200 NVL72` 页面给出的 `GB200 Grace Blackwell Superchip` 规格是:

- `FP16/BF16 Tensor Core = 10 PFLOPS`
- `GPU Memory Bandwidth = 16 TB/s`

同页脚注说明:

- `Specification in sparse. Dense is one-half sparse spec shown.`

因此换算成 **dense**，再除以 **`2 GPU / superchip`**，得到单 GPU:

| 指标 | 每 Superchip 标称 | dense 每 Superchip | dense 每 GPU |
|------|-------------------|--------------------|--------------|
| BF16 Tensor | `10 PFLOPS` | `5 PFLOPS` | **`2.5 PFLOPS`** |
| HBM 带宽 | `16 TB/s` | `16 TB/s` | **`8.0 TB/s`** |

官方来源:

- https://www.nvidia.com/en-us/data-center/gb200-nvl72/

### 2.3 Ridge Point

按 roofline 定义:

```text
ridge_point = peak_compute / peak_bandwidth
            = 2.5e15 / 8.0e12
            = 312.5 FLOP/B
```

所以本文统一采用:

- **`BF16 ridge = 312.5 FLOP/B`**

---

## 3. 主模型结构与 TP4 本地形状

这次建模的对象是:

- `/mnt/lustre/hf-models/hub/models--nvidia--Kimi-K2.5-NVFP4/snapshots/c0285e649c34d4386b01e38abca642c06cbe014e`

配置来自:

- `config.json`
- `vllm/model_executor/models/deepseek_v2.py`

### 3.1 关键模型参数

| 参数 | 值 |
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

### 3.2 结构解释

从 `DeepseekV2DecoderLayer` 可以直接看出:

- 第 `0` 层使用 `DeepseekV2MLP(intermediate_size=18432)`，是 **dense FFN**
- 第 `1~60` 层使用 `DeepseekV2MoE`
- `DeepseekV2MoE` 里:
  - routed expert 的 `intermediate_size = moe_intermediate_size = 2048`
  - shared expert 的 `intermediate_size = moe_intermediate_size * n_shared_experts = 2048`

这点很重要，因为它意味着 **shared expert 不是 `18432` 宽，而是也只有 `2048` 宽**。

### 3.3 TP=4 下每 GPU 的主要 GEMM 形状

对当前 `TP=4`，每 GPU 上主模型的主要 matmul 可近似写成:

| 模块 | 本地 `K x N` | 权重类型 | 备注 |
|------|--------------|----------|------|
| `fused_qkv_a_proj` | `7168 x 2112` | BF16 | replicated |
| `q_b_proj` | `1536 x 3072` | BF16 | column parallel |
| `kv_b_proj` | `512 x 4096` | BF16 | column parallel |
| `o_proj` | `2048 x 7168` | BF16 | row parallel |
| `dense gate_up_proj` | `7168 x 9216` | NVFP4 | `2 * 18432 / 4` |
| `dense down_proj` | `4608 x 7168` | NVFP4 | `18432 / 4` |
| `expert gate_up_proj` | `7168 x 1024` | NVFP4 | `2 * 2048 / 4` |
| `expert down_proj` | `512 x 7168` | NVFP4 | `2048 / 4` |

其中:

- attention 权重未量化，所以按 **BF16 weight traffic**
- FFN/MoE 权重是 NVFP4，本文采用 **`0.5625 B/param`** 的近似权重字节成本

---

## 4. 每个模块的理论 FLOPs

对 GEMM `A[M,K] x B[K,N]`，按最常见近似:

```text
FLOPs = 2 * M * K * N
```

这里先看 **每 token / 每 GPU** 的 matmul FLOPs，也就是令 `M = 1`。

### 4.1 Attention 投影

四个主投影合起来:

```text
fused_qkv_a_proj = 2 * 7168 * 2112   = 30,261,248
q_b_proj         = 2 * 1536 * 3072   =  9,437,184
kv_b_proj        = 2 *  512 * 4096   =  4,194,304
o_proj           = 2 * 2048 * 7168   = 29,376,512
---------------------------------------------------
attention total                        73,269,248 FLOPs/token/GPU
```

### 4.2 第 0 层 dense FFN

```text
gate_up = 2 * 7168 * 9216 = 132,120,576
down    = 2 * 4608 * 7168 =  66,060,288
----------------------------------------
dense FFN                  198,180,864 FLOPs/token/GPU
```

### 4.3 单个 routed/shared expert

```text
gate_up = 2 * 7168 * 1024 = 14,680,064
down    = 2 *  512 * 7168 =  7,340,032
---------------------------------------
one expert                 22,020,096 FLOPs/token/GPU
```

当前每个 token 激活:

- `8` 个 routed experts
- `1` 个 shared expert

所以单个 MoE 层的 FFN matmul 总量是:

```text
9 * 22,020,096 = 198,180,864 FLOPs/token/GPU
```

这和 dense FFN 完全相同。

### 4.4 单层与整模型总 FLOPs

因此:

```text
per-layer total
= attention + FFN
= 73,269,248 + 198,180,864
= 271,450,112 FLOPs/token/GPU
```

整模型 `61` 层:

```text
61 * 271,450,112
= 16,558,456,832 FLOPs/token/GPU
= 0.01656 TFLOP/token/GPU
```

这只是主干 matmul 的理论量，**尚未计入**:

- attention score/value kernel
- softmax
- RMSNorm
- routing/topk
- all-reduce / all-to-all / 其他通信
- KV cache 读写

---

## 5. Arithmetic Intensity 建模方法

### 5.1 单个 GEMM 的 AI

对一个 `A[M,K] x B[K,N]`，本文采用最简单的 roofline 近似:

```text
FLOPs = 2 * M * K * N
Bytes = a * (M*K + M*N) + w * (K*N)
AI    = FLOPs / Bytes
```

其中:

- `a = 2 B` 表示 activation 用 BF16 计
- `w = 2 B` 表示 BF16 权重
- `w = 0.5625 B` 表示 NVFP4 权重近似

### 5.2 临界 batch `M*`

当 `AI(M*) = 312.5 FLOP/B` 时，可以把 `M*` 看成该模块从 bandwidth-bound 切到 compute-bound 的理论拐点。

---

## 6. 每个模块的理论计算强度

### 6.1 单个 kernel 的 `M*`

| 模块 | `AI(∞)` | `M*_BF16-ridge` |
|------|---------|-----------------|
| `fused_qkv_a_proj` | `1631.34` | `386.55` |
| `q_b_proj` | `1024.00` | `449.75` |
| `kv_b_proj` | `455.11` | `997.27` |
| `o_proj` | `1592.89` | `388.77` |
| `dense gate_up_proj` | `4032.00` | `95.27` |
| `dense down_proj` | `2804.87` | `98.91` |
| `expert gate_up_proj` | `896.00` | `134.96` |
| `expert down_proj` | `477.87` | `253.98` |

这里可以直接看到:

- **最难吃满的 attention kernel 是 `kv_b_proj`**
- **最难吃满的 expert kernel 是 `expert_down_proj`**
- NVFP4 FFN 的两个 dense GEMM 都比较容易进入 compute-bound

### 6.2 按模块聚合后的 AI

更接近系统建模的口径，是把多个 kernel 按模块合起来看:

| 模块 | `M=1` | `M=8` | `M=16` | `M=32` | `M=64` | `M=128` | `M=256` | `M*=312.5` |
|------|-------|-------|--------|--------|--------|---------|---------|------------|
| Attention 投影总和 | `1.00` | `7.95` | `15.81` | `31.24` | `61.04` | `116.70` | `214.47` | **`409.24`** |
| Dense FFN 总和 | `3.55` | `28.22` | `55.98` | `110.21` | `213.73` | `402.99` | `718.95` | **`96.46`** |
| 单个 expert 总和 | `3.54` | `27.32` | `52.58` | `97.75` | `171.35` | `274.81` | `393.67` | **`159.95`** |
| 第 0 层整体 | `2.10` | `16.72` | `33.21` | `65.52` | `127.59` | `242.45` | `440.89` | **`170.44`** |

这个表的解释是:

- **如果只看第 0 层 dense layer**，大概 `M ~= 170` 就会碰到 BF16 ridge
- **attention 投影本身更难吃满**，需要 `M ~= 409`
- 单个 expert GEMM 也需要 `M ~= 160`

但 Kimi 的真实瓶颈不在“一个 expert 的 GEMM 太小”，而在于:

- decode 时一个 batch 里的 token 被分散到很多 expert
- 每个 expert 实际吃到的 `M_expert` 往往接近 `1`

所以还需要单独建模 MoE occupancy。

---

## 7. MoE Occupancy 建模

### 7.1 为什么 MoE 需要单独算

虽然一个 token 理论上会经过 `8 routed + 1 shared` 个 expert，总 FLOPs 看起来和 dense FFN 一样，但 decode 时真正执行的不是“一个大 FFN GEMM”，而是:

- 先做 routing
- 再把 batch 内 token 分发到不同 expert
- 每个 expert 各自做一个更小的 GEMM

如果 batch 内 token 被分散到很多 expert，那么单 expert 的 `M` 会很小，算术强度就上不去。

### 7.2 简化 occupancy 模型

对一个全局 batch `M`，当前配置中 routed expert 的总 token slot 数是:

```text
slots = 8 * M
```

对 `384` 个 expert，采用独立均匀落点近似，可得活跃 expert 数:

```text
E_active(M) ~= 384 * ( 1 - (1 - 1/384)^(8M) )
```

那么单个活跃 expert 的平均 token 数就是:

```text
m_expert(M) ~= 8M / E_active(M)
```

于是 MoE 层的 AI 不能再用“一个大 FFN”去估，而要用:

- `attention(M)`
- `E_active` 个 routed expert 小 GEMM
- `1` 个 shared expert GEMM

一起求总 `FLOPs / Bytes`。

### 7.3 MoE 层与整模型的 AI

| `M` | `E_active` | 平均 `m_expert` | MoE layer AI | Full model AI |
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

用同样的方法求 ridge crossing，可得:

- **MoE layer**: `M ~= 4659`
- **Full model**: `M ~= 4554`

这个结果其实就是本文最核心的结论:

- **Kimi 的 dense first layer 并不难吃满**
- **真正把整模型一直拖在 memory-bound 区间里的，是 decode 时 MoE expert occupancy 太碎**

---

## 8. 映射到当前 `pd_kimi_bench_a_70k`

本 benchmark 的关键调度参数来自:

- `vigil/examples/pd_kimi_bench_a_70k.yaml`
- `workload_analysis_pd_kimi_70k.md`
- 2026-04-11 的 prefill / decode 实际日志

### 8.1 当前配置

| 侧 | 关键参数 | 值 |
|----|----------|----|
| Prefill | `max_num_batched_tokens` | `8192` |
| Prefill | `max_num_seqs` | `16` |
| Decode | `max_num_batched_tokens` | `8192` |
| Decode | `max_num_seqs` | `16` |
| 两侧 | `tensor_parallel_size` | `4` |
| 两侧 | `pipeline_parallel_size` | `1` |

其中:

- prefill 侧日志可以直接看到 `BatchDescriptor(num_tokens=8192, ...)`
- decode 侧是 token 级 step，理论上限主要受 `max_num_seqs = 16` 约束

### 8.2 当前 operating point

把这些 batch 点投到上面的 AI 曲线上:

| 场景 | 代表 `M` | Full model AI | 与 ridge `312.5` 的关系 |
|------|----------|---------------|--------------------------|
| 小 decode | `4` | `4.04` | **远低于 ridge** |
| decode 上限 | `16` | `5.79` | **远低于 ridge** |
| 中等 prefill | `128` | `15.10` | **远低于 ridge** |
| 当前 prefill chunk | `8192` | `429.03` | **高于 ridge** |

于是可以直接得到:

1. **Decode 几乎不可能靠当前 batch 进入 compute-bound**  
   即使把 `max_num_seqs` 从 `16` 提到 `64`，full-model AI 也只有 `9.52 FLOP/B`；提到 `128` 也才 `15.10 FLOP/B`。离 `312.5` 还差一个数量级以上。

2. **Prefill 的大 chunk 已经足够大**  
   `M = 8192` 时，主模型从纯 GEMM roofline 看已经进入 compute-bound 区域。继续增大 prefill batch，不会像小 batch 那样显著提升算术强度。

3. **因此 prefill 和 decode 的优化方向应该分开**  
   - prefill 更像是 **chunk 数、调度、KV/activation/编译开销** 问题  
   - decode 更像是 **MoE 小 GEMM、权重复用差、HBM traffic** 问题

---

## 9. 对当前系统的直接启发

### 9.1 Decode 侧

如果问题是 decode 慢，那么从这份建模看:

- **单纯调大 `max_num_seqs` 不会让 Kimi decode 很快撞上算力墙**
- 当前 decode 仍然深处 **memory-bound**
- 主矛盾更可能是:
  - MoE expert occupancy 碎片化
  - NVFP4 权重流量
  - attention / MoE kernel launch 与调度开销
  - 通信与 KV 读写

因此 decode 侧更值得看的方向是:

- expert packing / grouped dispatch
- 更高 expert occupancy
- 更少的 MoE launch/dispatch overhead
- speculative decode 是否真的提升了 target-side 有效 occupancy

### 9.2 Prefill 侧

如果问题是 prefill 慢，那么从 roofline 看:

- `8192` 已经足够大
- 再往上增大 `max_num_batched_tokens` 的收益，更多可能来自:
  - 更少的 chunk 数
  - 更少的 scheduler 轮转
  - 更低的 per-chunk 固定开销

而不是继续通过提高 AI 去吃满更多算力。

这和你们前面的 workload 分析是对得上的:

- `70K` tokens/turn 在 `8192` chunk 下需要大约 `9` 个 prefill chunk
- 所以 prefill 的瓶颈很可能已经从“算强不够”转移到了“chunk 太多”

---

## 10. 建模边界与注意事项

这份文档是一个 **主干 GEMM + HBM roofline** 模型，故意忽略了很多真实系统因素。下面这些开销都还没有算进去:

- attention score / value kernel
- softmax / norm / activation
- MoE routing 和 topk
- TP all-reduce
- expert parallel / all-to-all
- KV cache 读写
- CUDA Graph / compile 形状约束
- kernel fusion 和 launch overhead

因此:

- 这里的 `M*` 应该理解为 **“纯 roofline 意义上的第一拐点”**
- 不应被理解为“真实系统到这个 batch 一定不再涨速”
- 特别是 decode 场景，真实瓶颈通常会比这份模型更偏 memory-bound

---

## 11. 最终总结

对当前 `Kimi-K2.5-NVFP4` 主模型在 `GB200/B200 + TP=4` 上:

- 单 GPU 机器 roofline 取 **`2.5 PFLOPS BF16 dense`** 和 **`8 TB/s HBM`**
- 对应 **ridge point = `312.5 FLOP/B`**
- **第 0 层 dense layer** 的整体 roofline 拐点在 **`M ~= 170`**
- **attention 投影总和** 的拐点在 **`M ~= 409`**
- **真实整模型** 因为 MoE expert occupancy 分散，拐点被推迟到 **`M ~= 4554`**

映射到当前 benchmark:

- **decode (`M <= 16`) 明显深处 memory-bound**
- **prefill (`M = 8192`) 已经进入 compute-bound 区域**

因此一个最实用的判断是:

- **Kimi 主模型当前不缺 prefill batch 强度，缺的是 decode 侧的有效 expert occupancy 和更低的 memory traffic**

