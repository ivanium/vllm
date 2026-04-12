# Kimi-K2.5-NVFP4 推理系统 HBM 存储建模

> 基于实验 `pd_kimi_bench_a_70k` (2026-04-11 01:24) 的实际配置和日志数据

---

## 1. 实验环境总览

| 维度 | 值 |
|------|-----|
| 模型 | nvidia/Kimi-K2.5-NVFP4 (基于 DeepSeek V3 架构) |
| GPU | NVIDIA B200 (GB200 NVL72) |
| 每 GPU 显存 | 192 GB HBM3e (CUDA 可见 184.31 GiB) |
| 并行策略 | TP=4, PP=1, DP=1 (4 GPU / 节点, 1 节点 / 实例) |
| 部署模式 | PD 分离: 1 prefill 节点 + 1 decode 节点 |
| gpu_memory_utilization | 0.85 |
| max_model_len | 131,072 tokens |
| kv_cache_dtype | fp8 |
| Attention Backend | FLASHINFER_MLA (HND layout) |
| 投机解码 | Eagle3 (kimi-k2.5-eagle3), 3 speculative tokens |

---

## 2. 模型架构参数

### 2.1 核心结构 (text_config, DeepSeek V3 架构)

| 参数 | 值 | 说明 |
|------|-----|------|
| hidden_size | 7,168 | 隐层维度 |
| num_hidden_layers | 61 | Transformer 层数 |
| num_attention_heads | 64 | 注意力头数 |
| num_key_value_heads | 64 | KV 头数 (MLA 下 KV cache 实际只有 1 个 latent) |
| vocab_size | 163,840 | 词表大小 |
| max_position_embeddings | 262,144 | 最大位置编码 |

### 2.2 MLA (Multi-head Latent Attention) 参数

| 参数 | 值 | 说明 |
|------|-----|------|
| kv_lora_rank | 512 | KV 压缩后的 latent 维度 |
| q_lora_rank | 1,536 | Q 压缩后的 latent 维度 |
| qk_nope_head_dim | 128 | Q/K 非 RoPE 部分的 per-head 维度 |
| qk_rope_head_dim | 64 | Q/K RoPE 部分的 per-head 维度 |
| v_head_dim | 128 | Value per-head 维度 |
| **KV cache head_size** | **576** | = kv_lora_rank(512) + qk_rope_head_dim(64) |

**MLA 的核心思想**: 不存储完整的 K/V (每头 128 维 × 64 头 = 16,384 维), 而是存储一个压缩的 latent 向量 (512 维) 加上无法压缩的 RoPE 部分 (64 维), 总共 576 维. 在计算时通过 up-projection 恢复出完整的 K/V.

### 2.3 MoE (Mixture of Experts) 参数

| 参数 | 值 | 说明 |
|------|-----|------|
| n_routed_experts | 384 | 路由专家数 |
| num_experts_per_tok | 8 | 每 token 激活的专家数 |
| n_shared_experts | 1 | 共享专家数 |
| moe_intermediate_size | 2,048 | 每个路由专家的中间层维度 |
| intermediate_size | 18,432 | 共享专家 / dense 层的中间层维度 |
| first_k_dense_replace | 1 | 前 1 层为 dense FFN, 后 60 层为 MoE |

### 2.4 量化配置

| 维度 | 值 |
|------|-----|
| 权重量化 | NVFP4 (4-bit float, group_size=16) |
| KV cache 量化 | FP8 (8-bit float, `--kv-cache-dtype fp8`) |
| 量化 ignore | 所有 61 层的 self_attn*, lm_head, mm_projector*, vision_tower* |

注意: **Attention 权重不量化** (保持 bfloat16), 仅 MoE 专家的 Linear 层使用 NVFP4.

---

## 3. 单 GPU 显存布局

### 3.1 显存初始状态 (模型加载前)

从日志提取 (prefill 和 decode 节点一致):

```
total_memory     = 184.31 GiB   (GPU 物理总显存, CUDA 可见部分)
cuda_memory      =   2.47 GiB   (CUDA runtime + driver 开销)
torch_memory     =   0.02 GiB   (PyTorch 初始占用)
non_torch_memory =   2.45 GiB   (非 PyTorch 的 CUDA 分配)
free_memory      = 181.84 GiB   (可用显存)
torch_peak       =   0.01 GiB
```

### 3.2 Requested Memory (vLLM 管控的显存上限)

```
requested_memory = gpu_memory_utilization × total_memory
                 = 0.85 × 184.31 GiB
                 = 156.66 GiB   (日志确认)
```

剩余 15% (27.65 GiB) 预留给 OS、CUDA runtime、其他进程等.

### 3.3 模型权重占用

模型 safetensors 磁盘总大小: **550.24 GB** (590,779,912,064 bytes, 119 个 shard)

| 权重类别 | 张量数量 | 说明 |
|---------|---------|------|
| MoE 专家 (gate/up/down_proj) | 276,604 | NVFP4 量化, 占大头 |
| 共享专家 | 720 | NVFP4 量化 |
| Attention (self_attn) | 549 | **bfloat16 (未量化)** |
| Embedding + LM head | 5 | bfloat16 |
| 其他 (LayerNorm 等) | 463 | bfloat16 |

**TP=4 下单 GPU 权重估算**:

```
每 GPU 权重 ≈ 总权重 / TP_size
            ≈ 590,779,912,064 / 4 bytes
            ≈ 147,694,978,016 bytes
            ≈ 137.5 GiB
```

此外还需加载 **Eagle3 投机解码模型** 的权重, 实测稳态增量 **~0.87 GiB/GPU** (加载期间临时峰值 ~5.08 GiB, 加载完成后释放临时内存).

### 3.4 显存分配概览 (日志实测数据)

#### Prefill 节点 (实测)

```
Initial free memory:     181.84 GiB
Requested memory:        156.66 GiB (0.85 × 184.31)
weights memory:          140.39 GiB  (主模型 139.52 + Eagle3 0.87, 稳态)
torch peak increase:       2.41 GiB  (activation 峰值)
non-torch forward:        -2.99 GiB  (profiling 期间释放的临时内存)
Total non-KV-cache:      139.81 GiB
CUDA Graph:                0    GiB  (enforce_eager 禁用)
─────────────────────────────────────
Available KV Cache:       16.85 GiB  ← 实际可用
num_gpu_blocks:           14,413
GPU KV cache size:       461,216 tokens (14,413 × 32)
Max concurrency @131K:     3.52x
```

#### Decode 节点 (实测)

```
Initial free memory:     181.84 GiB
Requested memory:        156.66 GiB (0.85 × 184.31)
weights memory:          140.36 GiB  (主模型 139.49 + Eagle3 0.87, 稳态)
torch peak increase:       2.66 GiB  (activation 峰值)
non-torch forward:        -1.40 GiB
Total non-KV-cache:      141.62 GiB
CUDA Graph (actual):       0.75 GiB  (PIECEWISE=17, FULL=9, 13s 捕获)
CUDA Graph (estimated):    0.59 GiB  (差异 0.16 GiB / 21.5%)
─────────────────────────────────────
Available KV Cache:       15.04 GiB  ← profiling 计算值 (⚠ 未扣除 CUDA Graph)
Effective KV Cache:       14.14 GiB  ← 运行时实际可用 (CUDA Graph 挤占 0.75 GiB)
num_gpu_blocks:           12,866
GPU KV cache size:       411,712 tokens (12,866 × 32)
Max concurrency @131K:     3.14x
```

#### 显存布局图

```
┌─────────────────── 单 GPU 显存布局 (184.31 GiB) ──────────────────┐
│                                                                    │
│  ┌──────────────────────────────────────────────┐                 │
│  │ 预留区 (15%)                    27.65 GiB     │ ← 不由 vLLM 管理│
│  │ (OS, CUDA runtime, driver, 其他进程)          │                 │
│  ├──────────────────────────────────────────────┤                 │
│  │                                              │                 │
│  │  vLLM 管控区 (85%)            156.66 GiB     │                 │
│  │  ┌──────────────────────────────────────┐    │                 │
│  │  │ 模型权重 (main + Eagle3)             │    │                 │
│  │  │ 140.36~140.39 GiB                   │    │                 │
│  │  ├──────────────────────────────────────┤    │                 │
│  │  │ Activation 峰值 (前向计算)            │    │                 │
│  │  │ Prefill: 2.41 GiB / Decode: 2.66 GiB│    │                 │
│  │  ├──────────────────────────────────────┤    │                 │
│  │  │ CUDA Graph (仅 decode)               │    │                 │
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

### 3.5 模型权重加载详情

| 阶段 | Prefill | Decode | 说明 |
|------|---------|--------|------|
| 主模型加载后峰值 | 139.52 GiB | 139.49 GiB | 119 shards, 约 26-29 分钟 |
| Eagle3 加载后峰值 | 144.60 GiB | 144.57 GiB | 含加载期间 **临时内存 ~4.21 GiB** |
| **最终权重占用 (稳态)** | **140.39 GiB** | **140.36 GiB** | profiling 后, 临时内存已释放 |
| Eagle3 稳态增量 | +0.87 GiB | +0.87 GiB | = 最终稳态 - 主模型峰值 |

注意: Eagle3 加载期间的峰值 (144.57 GiB) 与稳态 (140.36 GiB) 差距 4.21 GiB, 这部分是 safetensors 加载的临时缓冲区, 加载完成后自动释放. Eagle3 模型的 **实际权重占用仅 0.87 GiB**.

### 3.6 Prefill vs Decode 节点的差异

| 维度 | Prefill 节点 | Decode 节点 |
|------|-------------|-------------|
| 编译模式 | `--enforce-eager` (无编译) | `-O3` (torch.compile) |
| CUDA Graph | **禁用** | **启用** (`FULL_DECODE_ONLY`) |
| CUDA Graph 显存 | 0 GiB | **0.75 GiB** (实测) |
| KV Transfer | MultiConnector (NIXL + Mooncake) | NixlConnector |
| 可用 KV Cache | **16.85 GiB** | **15.04 GiB** |
| num_gpu_blocks | **14,413** | **12,866** |
| KV cache tokens | **461,216** | **411,712** |
| 并发@131K | **3.52x** | **3.14x** |
| Activation 峰值 | 2.41 GiB | 2.66 GiB |
| 编译优化 | `norm_quant`, `act_quant`, `allreduce_rms` | `act_quant`, `allreduce_rms` |

---

## 4. KV Cache 存储建模

### 4.1 MLA KV Cache 结构

MLA 不存储传统的 K 和 V 矩阵, 而是存储压缩后的 latent:

```
传统 MHA/GQA KV Cache (per token per layer):
  K: [num_kv_heads, head_dim]   e.g., [64, 128] = 8,192 元素
  V: [num_kv_heads, head_dim]   e.g., [64, 128] = 8,192 元素
  总计: 16,384 元素

MLA KV Cache (per token per layer):
  latent: [1, kv_lora_rank + qk_rope_head_dim] = [1, 576] = 576 元素
  (kv_lora_rank=512 存储压缩的 KV, qk_rope_head_dim=64 存储 RoPE 部分)

压缩比: 16,384 / 576 ≈ 28.4x
```

### 4.2 单个 KV Cache Block 大小

vLLM 中 KV cache 以 "block" (page) 为单位管理:

| 参数 | 值 |
|------|-----|
| block_size | **32 tokens** (FLASHINFER_MLA kernel 要求, 见下方说明) |
| num_kv_heads (effective) | 1 (MLA latent 是跨头共享的) |
| head_size | 576 (kv_lora_rank + qk_rope_head_dim) |
| dtype | fp8 (1 byte/element) |

**为什么 block_size = 32 而非默认 16?**

vLLM 默认 `DEFAULT_BLOCK_SIZE = 16` (`vllm/config/cache.py:34`), 但 FLASHINFER_MLA 后端的 kernel 仅支持 block_size=[32, 64] (`vllm/v1/attention/backends/mla/flashinfer_mla.py:49-50`). vLLM 在 `get_preferred_block_size()` 中检测到 16 不被支持, 自动升级到最小支持值 32. 这是 **kernel 硬性要求**, 与 Eagle3 投机解码无关.

**计算 (来源: `vllm/v1/kv_cache_interface.py` `MLAAttentionSpec.real_page_size_bytes`)**:

```python
page_size_bytes = block_size × num_kv_heads × head_size × dtype_size
               = 32 × 1 × 576 × 1
               = 18,432 bytes
               = 18 KiB per block per layer
```

### 4.3 全层 KV Cache Block 大小

主模型 61 层的理论 per-block 大小:
```
total_page_size (主模型) = page_size_per_layer × num_layers
                         = 18,432 × 61
                         = 1,124,352 bytes
                         ≈ 1.07 MiB per block (仅主模型 61 层)
```

但这次实验的实际 block 开销并不需要通过“group_size 反推”来估算。Decode 日志里明确注册了：

```text
language_model.model.layers.0..60.self_attn.attn   -> torch.Size([12866, 32, 576])
model.layers.61.self_attn.attn                     -> torch.Size([12866, 2, 32, 16, 128])
```

这说明一个 block index 上实际包含两类页:

- **61 个主模型 MLA 层**: 每层 `18,432 B`
- **1 个 Eagle3 draft attention 层**: `2 × 32 × 16 × 128 × 1 = 131,072 B`

因此，这次实验中“跨全部层的单 block 总开销”是:
```
total_page_size (本次实验)
  = 61 × 18,432 + 131,072
  = 1,255,424 bytes
  ≈ 1.20 MiB per block
```

这个结果和日志完全一致:
```
Decode: 12,866 × 1,255,424 B = 16,152,285,184 B = 15.04 GiB
Prefill: 14,413 × 1,255,424 B = 18,094,426,112 B = 16.85 GiB
```

也就是说，**Eagle3 在这次实验里给每个 token 额外带来 4,096 B 的 KV 开销**，对应 1 个 draft attention 层，而不是之前文档里写的“约 7 层”。

### 4.4 KV Cache 容量 (日志实测)

| 节点 | 可用 KV Cache | num_gpu_blocks | block_size | 总 token 容量 | 并发@131K |
|------|-------------|---------------|-----------|--------------|----------|
| **Prefill** | **16.85 GiB** | **14,413** | 32 | **461,216** | **3.52x** |
| **Decode** | **15.04 GiB** | **12,866** | 32 | **411,712** | **3.14x** |

block_size = 32 是 **FLASHINFER_MLA kernel 硬性要求** (仅支持 [32, 64], 见 4.2 节说明).

Decode 节点日志显示 `Overriding num_gpu_blocks=0 with num_gpu_blocks_override=128`, 这是 CUDA graph profiling 阶段的 **临时 override** (`gpu_model_runner.py:5827-5836`), 仅用于分配最小 KV cache 来测量 graph 显存. 最终 num_gpu_blocks 是 profiling 完成后重新计算的.

这次实验命中的是 `len(kv_cache_groups) == 1 && UniformTypeKVCacheSpecs` 这个 special case (`vllm/v1/core/kv_cache_utils.py:1098-1119`), 因此:

```python
num_blocks = available_memory // total_page_size_per_block
```

其中 `total_page_size_per_block` 不是单层的 18,432 B，而是:

```python
total_page_size_per_block = sum(spec.page_size_bytes for spec in kv_cache_specs.values())
                          = 61 * 18,432 + 131,072
                          = 1,255,424 bytes
```

所以这次实验里更准确的计算方式是:

- `available_memory` = `requested_memory - non_kv_cache_memory`
- `per_block_total_bytes` = `1,255,424`
- `num_gpu_blocks` = `available_memory // per_block_total_bytes`

**Decode 节点验证**:
```
available = 156.66 - 141.62 = 15.04 GiB                       ✓
num_gpu_blocks = floor(15.04 GiB / 1,255,424 B) = 12,866      ✓
token_capacity = 12,866 × 32 = 411,712                        ✓
concurrency@131K = 411,712 / 131,072 = 3.14x                  ✓
```

### 4.5 每 token KV Cache 开销

| 指标 | 主模型 61 层 | 含 Eagle3 draft 1 层 |
|------|-------------|----------------------|
| 每 token 主模型 MLA | 576 bytes/layer | — |
| 每 token 主模型全层 | 576 × 61 = 35,136 bytes ≈ 34.3 KiB | — |
| 每 token draft 层 | — | 2 × 16 × 128 × 1 = 4,096 bytes |
| 每 token 全部层合计 | — | **39,232 bytes ≈ 38.3 KiB** |
| 1K tokens 全部层 | — | **~37.4 MiB** |
| 131K tokens 全部层 | — | **4.789 GiB** |

如果考虑 vLLM 的 32-token page 粒度，bench 里的常见长度还可以写成:

- `70,000 tokens` -> `ceil(70000 / 32) = 2,188 blocks` -> `2.558 GiB`
- `7,000 tokens` -> `219 blocks` -> `256.1 MiB`
- `10,500 tokens` -> `329 blocks` -> `384.7 MiB`
- `52,500 tokens` -> `1,641 blocks` -> `1.919 GiB`

**关键洞察**: 得益于 MLA 的压缩 + FP8 量化，主模型 61 层的 KV cache 已经非常省；本次实验里额外的 Eagle3 draft 层只再增加了 **4 KiB/token**，于是一个 `131K` context 的完整请求总共只需要 **4.789 GiB** HBM KV 容量。

### 4.6 MLA 压缩 + FP8 的联合效果

```
传统 MHA + bfloat16 per token per layer:
  = 2 × num_kv_heads × head_dim × 2 bytes
  = 2 × 64 × 128 × 2 = 32,768 bytes

MLA + FP8 per token per layer:
  = 1 × 576 × 1 = 576 bytes

总压缩比 = 32,768 / 576 ≈ 56.9x
```

这个 56.9x 的压缩分解为:
- MLA latent 压缩: ~28.4x (16,384 维 → 576 维)
- FP8 vs bfloat16: 2x (2 bytes → 1 byte)

### 4.7 与本 benchmark 对齐的 HBM working set

bench 参数 (`70k input`, `300 output`, `10 turns`, `global prefix 15%`, `conversation prefix 75%`) 对应的 KV footprint 如下:

| 组成 | tokens | blocks | HBM 占用 |
|------|--------|--------|----------|
| Global prefix | 10,500 | 329 | **0.385 GiB** |
| Conversation prefix | 52,500 | 1,641 | **1.919 GiB / 对话** |
| Unique segment | 7,000 | 219 | **0.256 GiB / 请求** |
| Full 70k turn | 70,000 | 2,188 | **2.558 GiB / 请求** |
| 300 decode tokens | 300 | 10 | **0.0117 GiB / 请求** |

如果把 benchmark 中的对话都视为“conversation prefix 需要同时驻留”的理论最坏情形，那么仅前缀工作集就是:

| Sweep 并发 | 对话数 | Global + Conversation Prefix | Turn 1 全量工作集 |
|-----------|-------|------------------------------|-------------------|
| 1 | 2 | **4.22 GiB** | **4.73 GiB** |
| 2 | 4 | **8.06 GiB** | **9.08 GiB** |
| 4 | 8 | **15.73 GiB** | **17.78 GiB** |
| 8 | 16 | **31.08 GiB** | **35.18 GiB** |

这和日志中的 HBM 容量对比后，可以直接得到几个结论:

- **Prefill 节点** 的 16.85 GiB KV pool，理论上还能勉强覆盖 `C=4` 时的 prefix 工作集，但已经接近极限。
- **Decode 节点** 如果把 CUDA Graph 的 0.75 GiB 也算进去，有效 KV 只剩 **14.14 GiB**，那么 `C=4` 时连“global + 8 个 conversation prefix”都已经装不下。
- `C=8` 时理论前缀工作集达到 **31.08 GiB**，远超 prefill/decode 单 GPU 的 HBM KV budget，必须依赖 eviction、prefix cache 失活回收、或更低层级的 KV 介质。

---

## 5. 显存实测汇总

### 5.1 Prefill 节点 (单 GPU, 实测)

| 组件 | 大小 | 数据来源 |
|------|------|---------|
| GPU 物理总显存 | 184.31 GiB | `total_memory` |
| vLLM 管控上限 (85%) | 156.66 GiB | `requested_memory` |
| 模型权重 (主模型 + Eagle3) | **140.39 GiB** | `weights_memory` (profiling 日志) |
| Activation 峰值 | **2.41 GiB** | `torch_peak_increase` |
| Non-torch forward | **-2.99 GiB** | profiling 期间释放的临时内存 |
| CUDA Graph | **0 GiB** | `enforce_eager` 禁用 |
| **可用 KV Cache** | **16.85 GiB** | `Available KV cache memory` |
| num_gpu_blocks | **14,413** | `cache_config_info` |
| Token 容量 | **461,216** | `GPU KV cache size` |
| 并发@131K | **3.52x** | `Maximum concurrency` |

### 5.2 Decode 节点 (单 GPU, 实测)

| 组件 | 大小 | 数据来源 |
|------|------|---------|
| GPU 物理总显存 | 184.31 GiB | `total_memory` |
| vLLM 管控上限 (85%) | 156.66 GiB | `requested_memory` |
| 模型权重 (主模型 + Eagle3) | **140.36 GiB** | `weights_memory` |
| Activation 峰值 | **2.66 GiB** | `torch_peak_increase` |
| Non-torch forward | **-1.40 GiB** | profiling 期间 |
| CUDA Graph | **0.75 GiB** | `Graph capturing took 0.75 GiB` |
| **可用 KV Cache** | **15.04 GiB** | `Available KV cache memory` |
| 有效 KV Cache (扣 CUDA Graph) | **14.14 GiB** | `--kv-cache-memory=15185736397` |
| 有效 num_gpu_blocks (若预扣 graph) | **12,096** | `14.14 GiB / 1,255,424 B` |
| 有效 token 容量 (若预扣 graph) | **387,072** | `12,096 × 32` |
| num_gpu_blocks | **12,866** | `cache_config_info` |
| Token 容量 | **411,712** | `GPU KV cache size` |
| 并发@131K | **3.14x** | `Maximum concurrency` |

### 5.3 单节点 (4 GPU) — 注意 TP=4 时 KV cache 是同步的

| 指标 | Prefill 节点 | Decode 节点 |
|------|-------------|-------------|
| 总 GPU 显存 | 4 × 184.31 = 737.2 GiB | 737.2 GiB |
| 总模型权重 (跨 4 GPU) | ~550 GB (每 GPU ~140 GiB) | ~550 GB |
| KV Cache / GPU | **16.85 GiB** | **15.04 GiB** |
| Token 容量 (受限于单 GPU) | **461,216** | **411,712** |
| 131K context 并发数 | **3.52** | **3.14** |

**重要**: TP=4 时每个 GPU 存储 **相同的** KV cache blocks (全部层的 1/4 在不同 GPU 上, 但 block 数相同). 因此并发能力以**单 GPU** 的 num_gpu_blocks 为准, 不能将 4 GPU 的 KV cache 简单相加.

---

## 6. vLLM KV Cache 分配机制

### 6.1 Memory Profiling 流程 (`vllm/v1/worker/gpu_worker.py`)

```
1. 记录初始显存快照 (init_memory_snapshot)                       [line 284]
   → torch_peak=0.01, free_memory=181.84, total_memory=184.31 GiB

2. 计算 requested_memory = gpu_memory_utilization × total_memory  [line 285]
   = 0.85 × 184.31 = 156.66 GiB

3. 分配模型权重 (cumem allocator)
   → 按 layer 逐块分配到 GPU, 主模型 ~139.5 GiB + Eagle3 ~0.87 GiB

4. 执行 profile_run (dummy forward pass)                         [line 366]
   → 测量 activation 峰值显存

5. 计算 non_kv_cache_memory:                                      [line 387-391]
   non_kv_cache_memory = non_torch_increase + torch_peak_increase + weights_memory
   Decode: -1.40 + 2.66 + 140.36 = 141.62 GiB

6. 计算可用 KV cache:                                             [line 417-421]
   available_kv_cache = requested_memory - non_kv_cache_memory - cudagraph_estimate_applied
   Decode: 156.66 - 141.62 - 0 = 15.04 GiB

   ⚠ cudagraph_estimate_applied = 0 因为 VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS
   默认关闭. CUDA Graph 的 0.75 GiB 在 capture 阶段才实际占用, 会挤占 KV cache 空间.

7. 计算 num_gpu_blocks:
   本次实验走的是 `UniformTypeKVCacheSpecs` special case
   num_gpu_blocks = available_memory // total_page_size_per_block
                  = 15.04 GiB // 1,255,424 B
                  = 12,866
```

### 6.2 Block 管理

这次实验并不是“单一 page size × group_size”的 general case，而是 `UniformTypeKVCacheSpecs` 的 special case:

```python
# 主模型 MLA 层
mla_page_size = 32 × 1 × 576 × 1 = 18,432 bytes

# Eagle3 draft 层（日志里显示 cache shape = [num_blocks, 2, 32, 16, 128]）
draft_page_size = 2 × 32 × 16 × 128 × 1 = 131,072 bytes

# 所有层合并后的 per-block 总大小
total_page_size_per_block = 61 × 18,432 + 131,072
                          = 1,255,424 bytes

num_blocks = available_memory // total_page_size_per_block
```

### 6.3 CUDA Graph 显存的特殊处理

Decode 节点的 CUDA Graph 显存 (0.75 GiB) **不参与** KV cache 预算计算:

```
available_kv_cache 计算时:
  cudagraph_estimate_applied = 0  (VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS 默认 False)

CUDA Graph capture 在 KV cache 分配之后:
  Graph capturing finished in 13 secs, took 0.75 GiB (actual)
  Graph estimated: 0.59 GiB, 差异 21.5%

实际效果: KV cache 标称 15.04 GiB, 但运行时 CUDA Graph 会占用其中 0.75 GiB,
         有效 KV cache ≈ 14.14 GiB (日志建议: --kv-cache-memory=14.14 GiB)

vLLM v0.19 将默认启用 VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1,
届时 CUDA Graph 显存将在 KV cache 计算前被扣除.
```

### 6.4 KV Cache 内存池

这次实验里，vLLM 会为每个 attention layer 单独分配一个 KV tensor，但所有 layer 共享同一个 `num_gpu_blocks`:

```python
# 61 个主模型 MLA 层
per_mla_layer_shape = [num_gpu_blocks, 32, 576]      # fp8
per_mla_layer_size  = num_gpu_blocks * 18,432 bytes

# 1 个 Eagle3 draft 层
per_draft_layer_shape = [num_gpu_blocks, 2, 32, 16, 128]  # fp8
per_draft_layer_size  = num_gpu_blocks * 131,072 bytes

total_kv_pool = 61 * per_mla_layer_size + per_draft_layer_size
```

以 decode 节点为例:

```text
61 × (12,866 × 18,432) + 1 × (12,866 × 131,072)
= 16,152,285,184 bytes
= 15.04 GiB
```

---

## 7. 重要数值速查表

```
┌──────────────────────────────────────────────────────────────────┐
│ GPU 物理显存:                     192 GB (184.31 GiB CUDA 可见)   │
│ vLLM 管控 (85%):                 156.66 GiB / GPU                │
│ 模型权重 / GPU (TP=4, 实测):     140.36~140.39 GiB               │
│   └ 主模型:                      139.49~139.52 GiB               │
│   └ Eagle3:                      ~0.87 GiB (稳态)                 │
│ Activation 峰值:                 Prefill 2.41 / Decode 2.66 GiB  │
│ CUDA Graph:                      Prefill 0 / Decode 0.75 GiB     │
│                                                                   │
│ KV Cache block_size:             32 tokens (实际)                 │
│ KV Cache head_size:              576 (kv_lora_rank + rope_dim)    │
│ KV Cache num_kv_heads:           1 (MLA latent)                   │
│ KV Cache dtype:                  fp8 (1 byte)                     │
│                                                                   │
│ 可用 KV Cache:                   Prefill 16.85 / Decode 15.04 GiB│
│ num_gpu_blocks:                  Prefill 14,413 / Decode 12,866   │
│ Token 容量:                      Prefill 461,216 / Decode 411,712 │
│ 并发@131K:                       Prefill 3.52x / Decode 3.14x    │
│                                                                   │
│ 每 token 主模型 61 层:            35,136 bytes (34.3 KiB)          │
│ 每 token 含 Eagle3 1 层:         39,232 bytes (38.3 KiB)          │
│ 131K context (含 Eagle3):        4.789 GiB                        │
│ Decode 有效 token 容量(扣 graph): 387,072 tokens / 2.95x @131K     │
│ MLA+FP8 联合压缩比 (vs MHA+bf16): 56.9x                           │
└──────────────────────────────────────────────────────────────────┘
```

---

## 8. 局限性与后续工作

### 8.1 数据完整性

本文档所有 HBM 数据均来自实际日志 (642K 行 decode log, ~95K 行 prefill log), 包括:
- 模型权重、activation、CUDA graph、KV cache 分配均为 **profiling 实测值**
- `num_gpu_blocks` 和 token 容量来自 vLLM EngineCore 日志

### 8.2 已确认 vs 待确认

| 项目 | 状态 | 来源 |
|------|------|------|
| block_size = 32 | **已确认** | FLASHINFER_MLA kernel 要求 [32,64], 源码 `flashinfer_mla.py:44-46` |
| num_kv_heads = 1 (MLA) | **已确认** | 主模型 cache shape `[num_blocks, 32, 576]` |
| head_size = 576 | **已确认** | 模型配置 + 日志 `AttentionSelectorConfig(head_size=576)` |
| 主模型 page_size = 18,432 bytes/layer | **已确认** | `kv_cache_interface.py:182-189` |
| Eagle3 额外 draft 层 = 1 层 | **已确认** | 日志注册 `model.layers.61.self_attn.attn` |
| draft layer page_size = 131,072 bytes | **已确认** | cache shape `[num_blocks, 2, 32, 16, 128]` |
| total_page_size_per_block = 1,255,424 bytes | **已确认** | `61×18,432 + 131,072` |
| weights_memory | **已确认** | 日志 profiling 实测值 |
| available_kv_cache | **已确认** | 日志 `Available KV cache memory` |
| num_gpu_blocks | **已确认** | 日志 `cache_config_info` |
| Eagle3 权重 0.87 GiB | **已确认** | 日志 稳态差值 (140.36 - 139.49) |
| MLA 压缩比 56.9x | **已确认** | 理论计算, 源码验证 |
| CUDA Graph 未从 KV cache 扣除 | **已确认** | `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS` 默认 False |

### 8.3 已知问题

1. **non-torch forward 负值**: profiling 显示 non-torch forward increase 为负数 (Prefill: -2.99 GiB, Decode: -1.40 GiB). 这是正常行为 — 模型加载期间 CUDA 分配的临时缓冲区在 profiling 完成后被释放. (`vllm/utils/mem_utils.py:271-274`)

2. **CUDA Graph 显存挤占 KV cache**: Decode 节点标称 15.04 GiB KV cache, 但 CUDA Graph 实际占用 0.75 GiB (vs 估算 0.59 GiB, 偏差 21.5%), 有效 KV cache ≈ 14.14 GiB，对应有效 block 数约 12,096、有效 token 容量约 387,072。vLLM v0.19 将默认启用 `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1` 解决此问题。

3. **benchmark 前缀工作集在高并发下超过 HBM**: `C=4` 时仅全局前缀 + 8 个 conversation prefix 就达到 15.73 GiB；`C=8` 时达到 31.08 GiB。单个 decode TP 组无法仅靠 HBM 完整保留所有热前缀。

### 8.4 后续建模方向

- [ ] 建模 CPU DDR (L2) 层: SimpleCPUOffloadConnector 的 offload 策略和容量规划
- [ ] 建模 NVMe (L3) 层: 本地磁盘缓存的容量规划
- [ ] 建模跨节点 KV cache 传输带宽: NIXL (NVLink) vs Mooncake (RDMA) 的实际吞吐
- [ ] 考虑 prefix caching 对有效 KV cache 利用率的影响
- [ ] 多用户并发场景下的 KV cache 压力测试

---

## 附录 A: 模型权重分层解析 (TP=4 单 GPU)

### Attention 权重 (bfloat16, 未量化)

每层 Attention 主要参数 (近似):

| 矩阵 | Shape (全量) | 参数量 | TP=4 后 / GPU |
|------|-------------|--------|--------------|
| q_a_proj | [7168, 1536] | 11.0M | 11.0M (不切分) |
| q_a_layernorm | [1536] | 1.5K | 1.5K |
| q_b_proj | [1536, 64×192] | 18.9M | 4.7M |
| kv_a_proj_with_mqa | [7168, 576] | 4.1M | 4.1M (不切分) |
| kv_a_layernorm | [512] | 512 | 512 |
| kv_b_proj | [512, 64×256] | 8.4M | 2.1M |
| o_proj | [64×128, 7168] | 58.7M | 14.7M |

每层 Attention / GPU ≈ 37M params × 2 bytes (bf16) ≈ 74 MB
61 层总计 ≈ **4.4 GiB / GPU**

### MoE 权重 (NVFP4, 量化)

每层每个路由专家:
- gate_proj + up_proj + down_proj: 3 × 7168 × 2048 = 44.0M params
- NVFP4 (0.5 bytes + scales): ≈ 0.5625 bytes/param → ~24.8 MB / expert

TP=4: 每 GPU 负责 384/4 = 96 个路由专家

每层 MoE / GPU:
- 96 路由专家: 96 × 24.8 MB ≈ 2.38 GB
- 1 共享专家 (TP 切分): 3 × 7168 × 18432 / 4 × 0.5625 bytes ≈ 55.9 MB
- Gate: 7168 × 384 × 2 bytes ≈ 5.3 MB
- 每层 MoE / GPU ≈ ~2.44 GB

60 MoE 层总计 ≈ **~146.4 GB ≈ 136.3 GiB / GPU**

(注: 上述 MoE 估算基于理论计算, 实际可能因 padding、量化 metadata 等略有出入)

### 其他

| 组件 | 大小 / GPU |
|------|-----------|
| Embedding (vocab_parallel) | 163,840 × 7,168 / 4 × 2 bytes ≈ 561 MB |
| LM Head | ~561 MB (可能与 embedding 共享) |
| LayerNorm (61 层) | 较小 (~几 MB) |

---

## 附录 B: KV Cache 容量与并发关系

基于 Decode 节点实测: 12,866 blocks × 32 tokens = 411,712 tokens 总池:

| 场景 | 每请求 tokens | 每请求 blocks | 每请求 KV cache | 最大并发 |
|------|-------------|-------------|---------------|---------|
| 131K context (max) | 131,072 | 4,096 | **4.789 GiB** | **3.14** (实测) |
| 70K context (bench) | 70,000 | 2,188 | **2.558 GiB** | ~5.88 |
| 32K context | 32,000 | 1,000 | **1.169 GiB** | ~12.87 |
| 8K context | 8,000 | 250 | **0.292 GiB** | ~51.47 |

基于 Prefill 节点实测: 14,413 blocks × 32 tokens = 461,216 tokens 总池:

| 场景 | 最大并发 |
|------|---------|
| 131K context (max) | **3.52** (实测) |
| 70K context (bench) | ~6.59 |
| 32K context | ~14.41 |
| 8K context | ~57.65 |

注: 实际还需考虑 prefix caching 命中率, 多轮对话共享前缀, 以及 KV cache block 碎片化.

---

## 附录 C: 不同 KV Cache 量化策略对比

| 策略 | 每 token 每层 | 每 token 全 61 层 | 131K context | 压缩比 vs MHA+bf16 |
|------|-------------|-----------------|-------------|-------------------|
| MHA + bfloat16 | 32,768 B | 1,998,848 B (1.9 MiB) | ~249 GiB | 1x |
| MHA + fp8 | 16,384 B | 999,424 B (976 KiB) | ~125 GiB | 2x |
| **MLA + bfloat16** | 1,152 B | 70,272 B (68.6 KiB) | ~8.76 GiB | 28.4x |
| **MLA + fp8 (主模型 61 层)** | **576 B** | **35,136 B (34.3 KiB)** | **~4.38 GiB** | **56.9x** |
| **MLA + fp8 + Eagle3 1 层 (本实验总计)** | — | **39,232 B (38.3 KiB)** | **4.789 GiB** | — |
| MLA + fp8_ds_mla | 656 B | 40,016 B (39.1 KiB) | ~4.99 GiB | 49.9x |

---

*文档生成时间: 2026-04-11*
*数据来源: vigil/logs/pd_kimi_bench_a_70k/2026-04-11/20260411_012402*
