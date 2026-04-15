# Workload Analysis Report for the Kimi K2.5 NVFP4 PD-Disaggregated Inference System

- **日期**: 2026-04-11
- **配置文件**: `vigil/examples/pd_kimi_bench_a_70k.yaml`
- **日志目录**: `vigil/logs/pd_kimi_bench_a_70k/2026-04-11/20260411_012402/`
- **模型**: nvidia/Kimi-K2.5-NVFP4
- **硬件**: GB200 NVL72 (2 节点, 8 GPU)
- **vLLM commit**: `f05fce059` (branch: `feat/mooncake-store-int-notes-20260408`)
- **vigil commit**: `284b3fa` (branch: `feat/vigil-recipe-updates-20260408`)

---

## 1. 系统概述

### 1.1 硬件平台

本次测试在 NVIDIA GB200 NVL72 集群上进行，使用其中 2 个节点共 8 块 GPU。

| 资源 | 单 GPU | 单节点 (4 GPU) | 本次测试 (8 GPU) |
|:-----|:------:|:--------------:|:----------------:|
| GPU 型号 | NVIDIA B200 (Blackwell) | — | — |
| GPU HBM3e | 192 GB | 768 GB | 1,536 GB |
| CPU DDR (LPDDR5X) | — | 882 GiB | 1,764 GiB |
| 本地 NVMe RAID0 | — | ~12 TB | ~24 TB |
| GPU-CPU 带宽 (NVLink-C2C) | 225 GB/s | 900 GB/s | — |
| GPU-GPU 带宽 (NVLink 5) | 1.8 TB/s | — | — |

集群总规模: 18 节点, 72 GPU, 13.8 TB HBM3e, 共享 Lustre 存储 35 TB。

### 1.2 模型架构

| 参数 | 值 |
|:-----|:---|
| 模型名称 | Kimi K2.5 NVFP4 |
| 基础架构 | DeepSeek V3 |
| Transformer 层数 | 61 |
| Hidden size | 7,168 |
| Attention heads / KV heads | 64 / 64 (MHA) |
| MoE 专家总数 / 激活数 / 共享数 | 384 / 8 / 1 |
| MoE intermediate size (per expert) | 2,048 |
| FFN intermediate size | 18,432 |
| 词表大小 | 163,840 |
| 最大上下文长度 | 262,144 tokens (256K) |
| 权重量化 | NVFP4 (4-bit, group_size=16) |
| 激活量化 | NVFP4 (4-bit, group_size=16) |
| KV Cache 量化 | FP8 |
| 位置编码 | YaRN (rope_theta=50000, factor=64) |
| 注意力特性 | Q LoRA rank=1536, KV LoRA rank=512, QK nope_dim=128, QK rope_dim=64, V head_dim=128 |

本次测试仅使用语言模型部分 (`--language-model-only`)，不加载视觉编码器。

### 1.3 Serving 架构 (PD 分离)

```
                    ┌─────────────────────────────────────────┐
                    │  Router (round_robin, PD disaggregation)│
                    │  Port: 37029                            │
                    └───────┬─────────────────┬───────────────┘
                            │                 │
                ┌───────────▼───────┐ ┌───────▼───────────────┐
                │  Prefill Node     │ │  Decode Node          │
                │  gb200-rack1-09   │ │  gb200-rack1-10       │
                │  4x B200 (TP4)   │ │  4x B200 (TP4)        │
                │  Port: 8000      │ │  Port: 8000           │
                └───────────────────┘ └───────────────────────┘
```

**Prefill vs Decode 配置对比:**

| 配置项 | Prefill | Decode |
|:-------|:--------|:-------|
| 节点 | gb200-rack1-09 | gb200-rack1-10 |
| TP | 4 | 4 |
| 编译模式 | enforce-eager | O3 + cudagraph FULL_DECODE_ONLY |
| KV Transfer | MultiConnector (Nixl + MooncakeStore) | NixlConnector |
| Chunked prefill | 启用 | 启用 |
| Prefix caching | 启用 | 启用 |
| Sleep mode | 启用 | 启用 |
| Flash Attention | v4, disable_flashinfer_prefill | v4, disable_flashinfer_prefill |
| **共同配置** | | |
| max-model-len | 131,072 | 131,072 |
| max-num-seqs | 16 | 16 |
| max-num-batched-tokens | 8,192 | 8,192 |
| KV cache dtype | fp8 | fp8 |
| GPU memory utilization | 0.85 | 0.85 |
| Speculative decoding | Eagle3 (3 tokens, 0.6 synthetic acceptance) | Eagle3 (3 tokens, 0.6 synthetic acceptance) |

关键设计选择:
- **Prefill 使用 enforce-eager** 而非 O3 编译，因为 prefill 阶段 input shape 变化大，编译收益有限且有 overhead
- **Prefill 使用 MultiConnector** (Nixl + MooncakeStore) 支持多层级 KV 缓存; Decode 仅使用 NixlConnector 接收 KV
- **Eagle3 投机解码**在两端均启用，synthetic acceptance rate 0.6，每次投机 3 个 token

---

## 2. 基准测试配置分析

### 2.1 工具链

- **vigil**: Pipeline 编排器，管理 serving 启动、router、benchmark 执行、指标收集
- **vllm-bench**: Rust 实现的基准测试工具，支持多轮对话、并发扫描、前缀共享模拟
- **vmon**: 后台指标采集器 (2s interval)，从两个节点采集 GPU/vLLM 指标
- **Slurm**: 远程作业调度 (job_id: 2956)

### 2.2 请求生成参数

| 参数 | 值 | 说明 |
|:-----|:---|:-----|
| Backend | openai-chat | OpenAI Chat Completions API |
| Dataset | random | 随机生成 token |
| Input length | 70,000 tokens/turn | 每轮固定 |
| Output length | 300 tokens/turn | 每轮固定 |
| Multi-turn | 10 turns/conversation | |
| Inter-turn delay | 250 ms | 模拟用户思考时间 |
| History accumulation | No | 每轮独立发送完整 prompt，不累积历史 |
| Prefix: global | 15% = 10,500 tokens | 所有对话共享 |
| Prefix: conversation | 75% = 52,500 tokens | 同一对话内共享 |
| Prefix: unique | 10% = 7,000 tokens | 每次请求独特 |
| Concurrency sweep | 1, 2, 4, 8 | |
| sweep-num-prompts-factor | 2 | conversations = 2 × concurrency |

### 2.2.1 Workload 模式定义

当前 Kimi 70K 分析主要覆盖的是 **Simple test mode**，但在本项目里我们约定有三类 workload，需要统一记录并对照使用。

#### A. Simple test mode

目标是把复杂流量均摊成一个可重复、可对比的平均 case，方便做系统基线分析与参数扫描。

| 维度 | 定义 |
|:-----|:-----|
| 输入长度 | 固定 `70,000` tokens |
| 输出长度 | 固定 `200` tokens；本次 Kimi 70K recipe 实际使用 `300` tokens |
| 全局共享前缀 | 首 `12K` tokens 跨所有 conversations 共享 |
| 会话内前缀命中 | 目标为约 `90%` intra-conversation prefix cache hit |
| 数据集 | 一般使用 `random` |
| 适用场景 | 基线性能、前缀缓存收益评估、PD 参数扫描 |

这类模式的核心特点是：**长度固定、复用比例固定、结果稳定，适合比较不同配置和实现。**

#### B. Loadtest mode

目标是尽量贴近真实业务流量分布，而不是只跑单一平均 case。

| 维度 | 定义 |
|:-----|:-----|
| 输入长度分布 (ISL) | `P50 70K`, `P90 90K`, `P99 120K` |
| 输出长度分布 (OS) | `P50 200`, `P90 300`, `P99 1000` |
| 会话内缓存命中 | 从约 `60%` 逐步增长到 `95%` |
| 跨会话缓存命中 | 通常只有首轮 global/shared 部分接近 `100%` |
| 数据集 | `synthetic_sharegpt_v3.json` |
| 适用场景 | 更真实的吞吐-延迟权衡、容量规划、尾延迟分析 |

这类模式的核心特点是：**长度和缓存命中都服从真实分布，更适合验证系统在生产风格负载下的行为。**

#### C. Shadow mode

目标是按真实评测流量做 1:1 replay，而不是用 synthetic 分布采样。

| 维度 | 定义 |
|:-----|:-----|
| 流量来源 | 真实 `swebenchpro` eval traffic dump |
| 数据集 | `codex_swebenchpro.json` |
| 输入/输出长度 | 跟随原始 trace |
| 前缀模式 | 跟随原始 trace 中的上下文与会话结构 |
| 适用场景 | 真实 replay、问题复现、博客/结果复核 |

这类模式的核心特点是：**最接近真实 workload，但可重复性和参数可控性弱于 simple/loadtest。**

#### D. 三类 workload 与本文关系

- 本文分析的主数据来自 `random + fixed 70K/300 + prefix sharing`，属于 **Simple test mode**。
- 后续若将同一套 PD recipe 扩展为 `sharegpt + synthetic_sharegpt_v3.json`，则属于 **Loadtest mode**。
- 若直接回放 `codex_swebenchpro.json`，则应单独标记为 **Shadow mode**，不与 simple/loadtest 混合统计。

### 2.3 负载计算

| 并发度 | 对话数 | 总请求数 | 总 Input Tokens | 总 Output Tokens | 总 Tokens |
|:------:|:------:|:--------:|:--------------:|:----------------:|:---------:|
| 1 | 2 | 20 | 1,400,000 | 6,000 | 1,406,000 |
| 2 | 4 | 40 | 2,800,002 | 12,000 | 2,812,002 |
| 4 | 8 | 80 | 5,600,000 | 24,000 | 5,624,000 |
| 8 | 16 | 160 | 11,200,000 | 48,000 | 11,248,000 |

每个对话: 10 turns × 70,000 = 700,000 input tokens + 10 turns × 300 = 3,000 output tokens。

整个 benchmark 总耗时 515.1 秒 (~8.6 分钟)，含所有 4 个并发级别的顺序执行。Pipeline 总耗时 2,413.8 秒 (~40 分钟，含 serving 启动)。

---

## 3. 输入/输出模式分析

### 3.1 Token 分布特征

本负载具有**极端的 prefill-dominated 特征**:

- **Input:Output 比例 = 233:1** (70,000 : 300)
- 吞吐量中 input tokens 占绝对主导 (例如 C=1: input 21,180 tok/s vs output 91 tok/s)
- 每轮请求中，prefill 计算量远大于 decode 计算量

这意味着:
- **Prefill 节点是系统的关键路径**，其计算能力直接决定了系统吞吐上限
- Decode 节点在大部分时间处于等待状态，利用率较低
- 优化 prefill 效率（如 prefix caching、chunked prefill 参数调优）是提升整体性能的最高杠杆

### 3.2 前缀共享分析

每轮 70,000 tokens 的组成结构:

```
├── Global Prefix    ─── 10,500 tokens (15%) ─── 所有对话共享，全局缓存
├── Conversation Prefix ─ 52,500 tokens (75%) ─── 同一对话内 10 轮共享
└── Unique Tokens    ─── 7,000 tokens (10%)  ─── 每次请求独特，必须计算
```

**Turn 1 (冷启动)**:
- 全局前缀首次出现 → 需要完整计算 10,500 tokens
- 对话前缀首次出现 → 需要完整计算 52,500 tokens
- 唯一部分 → 需要计算 7,000 tokens
- **总计: 70,000 tokens 全部需要计算**

**Turn 2-10 (热缓存)**:
- 全局前缀 → **缓存命中** (已由 Turn 1 或其他对话缓存)
- 对话前缀 → **缓存命中** (已由本对话 Turn 1 缓存)
- 唯一部分 → 需要计算 7,000 tokens
- **仅需计算 7,000 tokens (10%)，单个热请求的理论总可复用比例为 90%**

这一设计直接解释了 Turn 1 到 Turn 2 之间 TTFT 的戏剧性下降 (详见第 5 节)。

### 3.2.1 为什么工程日志里本地 Prefix Cache Hit 可能只有 15%

上面的 `90%` 指的是**单个已预热请求**从请求语义上可复用的 token 比例，不等于工程日志里的**本地 GPU prefix cache hit rate**。这两个口径需要分开看:

- 本 workload 启用了 prefix-sharing mode，且 **No history accumulation**。也就是说，每一轮只发送当前轮的固定长度 user message，上一轮生成的 `300` 个 output tokens **不会**拼接到下一轮输入中。
- 因此，对任意会话 `i` 的第 `t` 轮请求，其输入始终可以写成: `G(10.5K) + C_i(52.5K) + U_{i,t}(7K)`。
- 对单个已经预热的 Turn 2-10 请求，确实只有 `7K` unique suffix 需要新算，所以该请求的**总可复用比例**是 `90%`。
- 但按整段 10 轮对话平均时，还要计入每个会话的首轮冷启动: `1` 轮冷 + `9` 轮热，对应平均总可复用比例约为 `(10 x 10.5K + 9 x 52.5K) / (10 x 70K) = 82.5%`。
- 更重要的是，工程日志通常会把复用拆成两类: 本地 GPU 命中的 `Prefix cache hit rate`，以及通过 Mooncake / NIXL / KV connector 取回的 `External prefix cache hit rate`。这两个指标的分母不同，**不能直接相加**。
- 在 `C=8` 这组负载里，benchmark 实际生成的是 `16` 个会话，而不是 `8` 个。这样会话级前缀工作集大小为 `16 x 52.5K + 10.5K = 850.5K tokens`。
- 结合 [hbm_storage_modeling.md](../memory_analysis/hbm_storage_modeling_cn.md) 的 KV 容量建模，prefill 侧本地 GPU KV 容量通常约为 `46万~59万 tokens`，不足以长期容纳全部 `16` 份会话前缀。
- 结果就是: 最热、所有会话共享的 `10.5K global prefix` 更容易长期驻留在本地，因此本地 `Prefix cache hit rate` 常接近 `15%`；而 `52.5K conversation prefix` 虽然同样被复用，但更可能通过外部 KV cache 命中，或者在会话轮转过程中被挤出本地后重新加载。

换句话说，`本地约 15%` 与 `单个热请求理论 90% 可复用` 并不矛盾。前者描述的是**复用发生在本地 GPU 上的比例**，后者描述的是**从请求内容上有多少 token 可以不重新计算**。

### 3.3 Token 组成可视化说明

建议绘制堆叠柱状图:

- **X 轴**: Turn index (1-10)
- **Y 轴**: Token 数量 (0-70,000)
- **颜色**: 绿色 = Global Prefix, 蓝色 = Conversation Prefix, 红色 = Unique
- **标注**: Turn 1 全部为"需计算"; Turn 2+ 中绿色+蓝色标注为"缓存命中"

---

## 4. 并发与到达模式分析

### 4.1 并发级别总览

| 并发度 | 对话数 | 总请求数 | 持续时间 (s) | 总 Input Tokens | 总 Output Tokens | 成功率 |
|:------:|:------:|:--------:|:----------:|:--------------:|:----------------:|:------:|
| 1 | 2 | 20 | 66.1 | 1,400,000 | 6,000 | 100% |
| 2 | 4 | 40 | 71.7 | 2,800,002 | 12,000 | 100% |
| 4 | 8 | 80 | 145.6 | 5,600,000 | 24,000 | 100% |
| 8 | 16 | 160 | 226.5 | 11,200,000 | 48,000 | 100% |

所有并发级别下 0 失败，说明系统在此负载范围内具有良好的稳定性。

### 4.2 请求到达模式

本 benchmark 使用**闭环 (closed-loop)** 模式，而非开环 (open-loop):

- 每个对话的下一轮请求在上一轮完成后延迟 250ms 才发出
- 并发度控制的是同时运行的对话数，而非同时在途的请求数
- 这意味着实际的请求到达率受系统响应延迟制约

**闭环模式的含义**:
- 当系统变慢时，请求到达率自动降低（自适应背压）
- 实际 request throughput 是系统响应能力的反映，而非外部施加的负载
- 这与生产环境中多用户并发访问的场景更为接近

### 4.3 请求重叠与 Prefill 竞争

在不同并发度下，请求在时间轴上的重叠模式不同:

**C=1 (2 对话)**:
- 同一时刻最多 2 个对话在进行，但由于 250ms delay，请求基本串行
- Prefill 节点几乎不存在竞争

**C=2 (4 对话)**:
- 最多 2 个请求可能同时到达 prefill 节点
- 但由于 E2EL ~3.5s >> 250ms delay，实际重叠概率中等

**C=4 (8 对话)**:
- Turn 1 时 8 个对话同时发出首轮请求，造成 prefill 严重拥堵
- 后续轮次中，4 个对话可能同时等待 prefill
- max-num-seqs=16 尚未成为瓶颈，但 max-num-batched-tokens=8192 限制了并行 prefill 的效率

**C=8 (16 对话)**:
- Turn 1 时 16 个对话同时发出请求，远超 prefill 节点处理能力
- 后续轮次中，8 个对话可能同时竞争 prefill 资源
- max-num-seqs=16 接近饱和，排队延迟显著
- E2EL 膨胀到 11.2s，形成严重的排队效应

---

## 5. 逐轮次时序分析

### 5.1 TTFT 逐轮次变化

TTFT (Time to First Token) 是衡量 prefill 性能的核心指标。以下矩阵展示了各并发度下每轮的平均 TTFT (ms):

| Turn | C=1 | C=2 | C=4 | C=8 |
|:----:|:---:|:---:|:---:|:---:|
| 1 | **5,143** | **3,112** | **4,644** | **7,333** |
| 2 | 848 | 973 | 2,765 | 6,170 |
| 3 | 850 | 976 | 2,088 | 3,771 |
| 4 | 839 | 1,001 | 2,184 | 3,655 |
| 5 | 828 | 1,026 | 2,450 | 3,039 |
| 6 | 833 | 972 | 1,923 | 3,675 |
| 7 | 836 | 970 | 1,898 | 3,458 |
| 8 | 842 | 948 | 1,633 | 3,193 |
| 9 | 832 | 984 | 2,051 | 3,625 |
| 10 | 852 | 972 | 1,490 | 3,422 |
| **均值** | **1,271** | **1,193** | **2,312** | **4,134** |

**关键观察**:

1. **Turn 1 冷启动惩罚**: 所有并发度下 Turn 1 的 TTFT 显著高于后续轮次
   - C=1: 5,143ms → ~840ms (Turn 2+)，降幅 **83%**
   - C=2: 3,112ms → ~975ms，降幅 **69%**
   - 这直接验证了高比例 prefix 复用的效果; 对热请求而言，只有 7K unique tokens 需要新算

2. **C=1/C=2 稳态非常稳定**: Turn 2-10 的 TTFT 标准差极小
   - C=1: 828-852ms，波动 < 3%
   - C=2: 948-1,026ms，波动 < 8%

3. **C=4/C=8 稳态 TTFT 显著升高且波动大**:
   - C=4: 1,490-2,765ms，波动范围 ~1,300ms
   - C=8: 3,039-6,170ms，波动范围 ~3,100ms
   - 这反映了 prefill 节点排队造成的不确定延迟

4. **C=2 Turn 1 (3,112ms) 低于 C=1 Turn 1 (5,143ms)**: 可能因为 C=2 有 4 个对话，全局前缀被更早的对话预热，部分对话受益于已有缓存

### 5.2 TPOT 逐轮次变化

TPOT (Time per Output Token, 排除首 token) 反映 decode 性能:

| Turn | C=1 | C=2 | C=4 | C=8 |
|:----:|:---:|:---:|:---:|:---:|
| 1 | 8.40 | 10.09 | 12.83 | 17.40 |
| 2 | 6.15 | 7.69 | 13.44 | 22.72 |
| 3 | 6.62 | 7.48 | 17.35 | 24.63 |
| 4 | 7.22 | 7.73 | 16.39 | 24.71 |
| 5 | 6.49 | 8.11 | 18.17 | 24.56 |
| 6 | 6.36 | 7.67 | 17.41 | 24.87 |
| 7 | 6.61 | 7.70 | 17.85 | 24.78 |
| 8 | 6.62 | 7.73 | 16.54 | 24.75 |
| 9 | 6.50 | 7.32 | 16.27 | 25.01 |
| 10 | 7.08 | 7.94 | 15.31 | 22.23 |
| **均值** | **6.80** | **7.95** | **16.16** | **23.57** |

**观察**:
- TPOT 在同一并发度内较为稳定（排除 Turn 1）
- C=1→C=2 增幅 17%，C=2→C=4 增幅 103%，C=4→C=8 增幅 46%
- C=4 出现阶跃式增长，说明 decode 节点在 C=4 时批处理 size 显著增大

### 5.3 ITL P99 尾延迟分析

ITL (Inter-token Latency) 的 P99 是衡量用户体验一致性的关键指标:

| 并发度 | Mean ITL (ms) | Median ITL (ms) | P99 ITL (ms) | P99/Median |
|:------:|:------------:|:---------------:|:------------:|:----------:|
| 1 | 10.98 | 10.56 | 14.45 | 1.4x |
| 2 | 12.85 | 12.15 | 25.75 | 2.1x |
| 4 | 16.86 | 14.19 | 123.88 | **8.7x** |
| 8 | 23.54 | 15.95 | 379.45 | **23.8x** |

**分析**:
- C=1/C=2 的 P99/Median 比值在 1.4-2.1x，分布紧凑，用户体验一致
- **C=4 P99 ITL 跳升至 124ms**，P99/Median 达 8.7x，出现明显尾延迟
- **C=8 P99 ITL 爆炸至 379ms**，P99/Median 达 23.8x，表明少数 token 生成遭遇严重延迟

可能原因:
- Eagle3 投机解码在高并发下验证失败率上升，导致 token 重新生成
- KV cache transfer 与 decode 计算的资源竞争
- Scheduler 在 batch 切换时的调度延迟
- 高并发下 prefill 请求可能中断正在进行的 decode batch (chunked prefill 干扰)

### 5.4 E2EL 分解

E2EL (End-to-End Latency) 可近似分解为: E2EL ≈ TTFT + output_tokens × TPOT

| 并发度 | Mean TTFT | 300 × Mean TPOT | 预估 E2EL | 实际 Mean E2EL | 误差 |
|:------:|:---------:|:---------------:|:---------:|:-------------:|:----:|
| 1 | 1,271 | 2,040 | 3,311 | 3,305 | 0.2% |
| 2 | 1,193 | 2,384 | 3,577 | 3,569 | 0.2% |
| 4 | 2,312 | 4,847 | 7,159 | 7,143 | 0.2% |
| 8 | 4,134 | 7,071 | 11,205 | 11,181 | 0.2% |

分解模型高度准确 (误差 < 1%)。

**E2EL 中 TTFT 占比**:
- C=1: 38% (TTFT) + 62% (Decode)
- C=2: 33% (TTFT) + 67% (Decode)
- C=4: 32% (TTFT) + 68% (Decode)
- C=8: 37% (TTFT) + 63% (Decode)

TTFT 和 decode 时间对 E2EL 的贡献比较稳定，约 1:2。但注意在用户体感上，TTFT 对应"等待响应开始"的时间，对用户体验影响更大。

---

## 6. 性能扩展分析

### 6.1 吞吐量扩展

| 并发度 | Total Throughput (tok/s) | Input Throughput (tok/s) | Output Throughput (tok/s) | Request Throughput (req/s) | 吞吐扩展比 |
|:------:|:------------------------:|:------------------------:|:-------------------------:|:--------------------------:|:---------:|
| 1 | 21,271 | 21,180 | 91 | 0.30 | 1.00x |
| 2 | 39,242 | 39,074 | 167 | 0.56 | 1.84x |
| 4 | 38,623 | 38,458 | 165 | 0.55 | 1.82x |
| 8 | 49,660 | 49,448 | 212 | 0.71 | 2.33x |

**关键发现**:

1. **C=1→C=2 近线性扩展 (1.84x)**: 系统在 C=1 时有大量空闲容量，加倍并发几乎加倍吞吐
2. **C=2→C=4 吞吐持平甚至微降 (39,242 → 38,623)**: 这是 **prefill 饱和的明确信号**
   - 请求量翻倍，但吞吐不增反降
   - Request throughput 从 0.56 降至 0.55 req/s
   - 额外的并发只增加了排队时间
3. **C=4→C=8 吞吐增长至 49,660**: 但这主要是因为同时在途请求数翻倍，throughput 提升 (1.28x) 远低于并发提升 (2x)
   - 实际上是"更多请求在更长的队列中等待"的表象

### 6.2 延迟扩展

| 并发度 | Mean TTFT (ms) | Mean TPOT (ms) | Mean ITL (ms) | Mean E2EL (ms) | P99 ITL (ms) |
|:------:|:--------------:|:--------------:|:-------------:|:--------------:|:------------:|
| 1 | 1,271 | 6.80 | 10.98 | 3,305 | 14.45 |
| 2 | 1,193 | 7.95 | 12.85 | 3,569 | 25.75 |
| 4 | 2,312 | 16.16 | 16.86 | 7,143 | 123.88 |
| 8 | 4,134 | 23.57 | 23.54 | 11,181 | 379.45 |

**延迟扩展特征**:
- **C=1→C=2**: TTFT 几乎不变 (1,271→1,193ms)，E2EL 仅增 8%。这是"甜区"配置
- **C=2→C=4 (拐点)**: TTFT 翻倍 (1,193→2,312ms)，E2EL 翻倍 (3,569→7,143ms)，但吞吐无增长
- **C=4→C=8**: TTFT 再次接近翻倍 (2,312→4,134ms)，P99 ITL 增长 3x (124→379ms)

### 6.3 吞吐-延迟权衡

建议绘制双轴图:
- **X 轴**: Concurrency (1, 2, 4, 8)
- **左 Y 轴**: Total Throughput (tok/s) — 柱状图
- **右 Y 轴**: Mean TTFT (ms) — 折线图

数据点:

| C | Throughput | TTFT |
|:-:|:----------:|:----:|
| 1 | 21,271 | 1,271 |
| 2 | 39,242 | 1,193 |
| 4 | 38,623 | 2,312 |
| 8 | 49,660 | 4,134 |

图表将清晰展示: **C=2 是最优操作点** — 吞吐已接近系统上限，而延迟尚未显著恶化。C=4 以上，延迟持续膨胀但吞吐增长微弱。

### 6.4 每 GPU 效率

| 并发度 | Total Throughput | 每 GPU 效率 (tok/s/GPU) | 效率提升 (vs C=1) |
|:------:|:----------------:|:----------------------:|:-----------------:|
| 1 | 21,271 | 2,659 | 1.00x |
| 2 | 39,242 | 4,905 | 1.84x |
| 4 | 38,623 | 4,828 | 1.82x |
| 8 | 49,660 | 6,208 | 2.33x |

从 C=2 到 C=8，效率仅提升 27% (4,905→6,208 tok/s/GPU)，但平均延迟增加了 213% (3,569→11,181ms)。在 SLA 敏感的场景下，这不是一个有利的权衡。

---

## 7. 关键发现与瓶颈识别

### 7.1 Prefill 是主要瓶颈

**证据**:
- 70K tokens/turn 在 TP4 下需要大量计算; max-num-batched-tokens=8192 意味着每个 70K 请求需要 ~9 个 chunk 的 chunked prefill
- 吞吐在 C=2 即饱和 (39K→38K tok/s @ C=2→C=4)，而延迟翻倍
- Request throughput 在 C=2→C=4 不升反降 (0.56→0.55 req/s)
- 所有并发度下，Turn 1 (需完整 prefill 70K tokens) 的 TTFT 远高于后续轮次

**影响**: Prefill 节点的处理能力直接决定了系统的有效并发上限

### 7.2 Prefix Caching 效果显著

**证据**:
- C=1 Turn 1 TTFT 5,143ms → Turn 2 TTFT 848ms，降幅 **83%** (6.1x 加速)
- C=2 Turn 1 TTFT 3,112ms → Turn 2 TTFT 973ms，降幅 **69%** (3.2x 加速)
- Turn 2-10 在 C=1/C=2 下 TTFT 极其稳定 (波动 < 3%)，说明 cache hit 路径性能一致

**验证**: 高比例 prefix reuse 设计 (15% global + 75% conversation) 在实际测试中得到了验证。对单个已预热请求而言，从 70K tokens 的完整 prefill 缩减到仅 7K unique tokens 的计算，是 TTFT 大幅下降的根本原因; 但工程上观测到的本地 GPU prefix cache hit rate 可能显著低于 90%，因为其中一大部分复用会通过外部 KV cache 实现。

### 7.3 Decode 尾延迟在高并发下严重恶化

**证据**:
- P99 ITL: C=1 14ms → C=8 379ms，恶化 **26x**
- P99/Median 比值: C=1 1.4x → C=8 23.8x，分布极度右偏
- 每轮 P99 ITL 在 C=8 持续高于 350ms (Turn 2-9)

**可能原因**:
- Eagle3 投机解码在高负载下验证失败带来的 penalty
- 多个 decode 请求在同一 batch 中竞争 GPU 计算资源
- KV transfer (从 prefill 到 decode via Nixl) 与 decode 计算的资源干扰
- Chunked prefill 可能在 decode 节点上产生干扰 (decode 节点也启用了 chunked-prefill)

### 7.4 KV Transfer 开销是稳态 TTFT 的主要组成

**证据**:
- C=1 Turn 2-10 稳态 TTFT ~840ms，但实际仅需计算 7,000 unique tokens
- 7K tokens 在 TP4 B200 上的纯 prefill 计算时间估计约 100-200ms
- 剩余 ~640ms 主要来自:
  - Prefix cache 查找与 KV 加载 (从 CPU DDR 或 NVMe 到 GPU HBM)
  - KV transfer: prefill 到 decode 通过 Nixl (跨节点 via RoCE)
  - Scheduling overhead (chunked prefill 分片调度)

**意义**: 在 prefix caching 生效的稳态下，KV 数据搬运而非计算成为 TTFT 的主要组成部分。这指向 KV transfer 优化作为下一步性能提升的方向。

### 7.5 max-num-seqs=16 可能限制高并发性能

**证据**:
- Prefill 和 Decode 均设置 max-num-seqs=16
- C=8 时有 16 个对话同时运行，Turn 1 时 16 个请求同时到达
- 虽然请求是通过 router 分发的 (round_robin)，但 prefill 节点最多同时处理 16 个 sequence
- C=8 的 Turn 1 TTFT 高达 7,333ms，部分原因可能是 scheduler 排队

---

## 8. 优化建议

### 8.1 短期优化 (不改硬件拓扑)

| 优化项 | 当前值 | 建议值 | 预期效果 |
|:------|:------|:------|:--------|
| Prefill 副本数 | 1P1D | 2P1D 或 2P2D | 倍增 prefill 吞吐，降低排队延迟 |
| max-num-batched-tokens | 8,192 | 16,384 或 32,768 | 减少 chunked prefill 分片数 (9→5 或 3)，降低调度开销 |
| max-num-seqs | 16 | 32 或 64 | 减少高并发排队 |
| Eagle3 num_speculative_tokens | 3 | 2 (高并发时) | 降低 P99 ITL 尾延迟，牺牲少量平均 TPOT |

### 8.2 中期优化 (架构调整)

1. **Prefill TP8**: 将 prefill 从 TP4 (1 node) 扩展到 TP8 (2 nodes)，单请求 prefill 延迟降低约 50%
2. **Prefill 编译优化**: 评估 O3 编译在 prefill 端的收益 (当前使用 enforce-eager)
3. **Load-aware 路由**: 从 round_robin 切换到负载感知路由，尤其在多 prefill 节点场景下
4. **KV Transfer Pipeline**: 将 KV transfer 与 decode 计算 overlap，减少 TTFT 中的 transfer 等待

### 8.3 长期优化 (系统级)

1. **DP+TP 混合并行**: 使用 DP2×TP2 替代 TP4 作为 prefill 配置，获得 2x 请求并行度
2. **大规模扩展测试**: 当前仅使用 2/18 节点 (8/72 GPU)。需要在更大规模下验证扩展性
3. **真实 Production Trace Replay**: 70K random tokens 是极端场景。真实工作负载的 token 长度分布、对话轮次数、并发模式可能显著不同。建议使用 production access logs 进行 replay 测试
4. **Mooncake 多级缓存调优**: 评估 L1(GPU HBM) → L2(CPU DDR via NVLink-C2C) → L3(NVMe RAID0) 三级缓存在不同 working set size 下的命中率

---

## 9. 附录

### 9.1 完整逐轮次指标

#### 9.1.1 Concurrency = 1 (2 对话, 20 请求)

| Turn | TTFT Mean | TTFT Median | TTFT P99 | TPOT Mean | ITL Mean | ITL P99 | E2EL Mean | E2EL P99 |
|:----:|:---------:|:-----------:|:--------:|:---------:|:--------:|:-------:|:---------:|:--------:|
| 1 | 5,143.4 | 5,143.4 | 7,181.9 | 8.40 | 14.11 | 12.88 | 7,654.3 | 10,332.8 |
| 2 | 847.7 | 847.7 | 860.2 | 6.15 | 10.53 | 14.12 | 2,685.8 | 2,698.1 |
| 3 | 850.2 | 850.2 | 856.2 | 6.62 | 11.18 | 17.37 | 2,828.6 | 2,837.5 |
| 4 | 839.2 | 839.2 | 846.7 | 7.22 | 10.51 | 13.25 | 2,998.3 | 3,191.3 |
| 5 | 828.1 | 828.1 | 842.6 | 6.49 | 10.51 | 13.02 | 2,767.1 | 2,784.9 |
| 6 | 833.4 | 833.4 | 842.2 | 6.36 | 10.50 | 11.53 | 2,734.4 | 2,798.3 |
| 7 | 836.1 | 836.1 | 846.9 | 6.61 | 10.54 | 12.78 | 2,812.3 | 2,837.5 |
| 8 | 842.4 | 842.4 | 855.5 | 6.62 | 10.50 | 11.92 | 2,822.1 | 2,909.6 |
| 9 | 832.4 | 832.4 | 847.6 | 6.50 | 10.73 | 15.47 | 2,775.1 | 2,881.2 |
| 10 | 852.0 | 852.0 | 862.5 | 7.08 | 10.89 | 17.06 | 2,970.4 | 3,141.4 |

#### 9.1.2 Concurrency = 2 (4 对话, 40 请求)

| Turn | TTFT Mean | TTFT Median | TTFT P99 | TPOT Mean | ITL Mean | ITL P99 | E2EL Mean | E2EL P99 |
|:----:|:---------:|:-----------:|:--------:|:---------:|:--------:|:-------:|:---------:|:--------:|
| 1 | 3,111.8 | 3,320.4 | 4,548.9 | 10.09 | 16.82 | 20.83 | 6,127.5 | 6,929.0 |
| 2 | 973.3 | 991.8 | 1,096.2 | 7.69 | 12.16 | 19.62 | 3,271.1 | 3,390.0 |
| 3 | 975.9 | 995.8 | 1,001.5 | 7.48 | 12.24 | 25.87 | 3,212.4 | 3,443.8 |
| 4 | 1,001.1 | 1,002.2 | 1,016.4 | 7.73 | 12.63 | 55.91 | 3,312.6 | 3,376.8 |
| 5 | 1,026.4 | 1,022.3 | 1,061.6 | 8.11 | 12.73 | 42.85 | 3,452.3 | 3,597.8 |
| 6 | 971.9 | 982.0 | 1,010.3 | 7.67 | 12.18 | 25.92 | 3,263.9 | 3,586.7 |
| 7 | 969.9 | 968.8 | 1,001.8 | 7.70 | 12.40 | 19.20 | 3,272.8 | 3,443.5 |
| 8 | 948.4 | 953.7 | 993.0 | 7.73 | 12.28 | 23.72 | 3,259.3 | 3,314.5 |
| 9 | 984.0 | 948.2 | 1,134.3 | 7.32 | 12.59 | 26.58 | 3,171.7 | 3,332.7 |
| 10 | 972.1 | 954.6 | 1,063.3 | 7.94 | 12.58 | 19.38 | 3,347.2 | 3,572.5 |

#### 9.1.3 Concurrency = 4 (8 对话, 80 请求)

| Turn | TTFT Mean | TTFT Median | TTFT P99 | TPOT Mean | ITL Mean | ITL P99 | E2EL Mean | E2EL P99 |
|:----:|:---------:|:-----------:|:--------:|:---------:|:--------:|:-------:|:---------:|:--------:|
| 1 | 4,644.0 | 3,276.2 | 10,657.0 | 12.83 | 17.02 | 125.36 | 8,481.2 | 17,223.8 |
| 2 | 2,764.9 | 2,679.1 | 6,587.1 | 13.44 | 14.95 | 34.96 | 6,784.2 | 9,693.2 |
| 3 | 2,087.8 | 1,972.5 | 2,930.9 | 17.35 | 18.17 | 125.56 | 7,276.3 | 8,217.1 |
| 4 | 2,183.6 | 2,228.0 | 3,077.7 | 16.39 | 16.94 | 123.82 | 7,085.2 | 7,854.6 |
| 5 | 2,449.6 | 2,544.7 | 3,180.5 | 18.17 | 18.13 | 126.76 | 7,883.0 | 8,620.5 |
| 6 | 1,923.3 | 1,622.3 | 2,982.1 | 17.41 | 17.35 | 126.33 | 7,127.6 | 7,436.0 |
| 7 | 1,897.8 | 1,945.7 | 2,621.1 | 17.85 | 17.85 | 126.28 | 7,236.1 | 7,727.4 |
| 8 | 1,632.7 | 1,620.5 | 2,132.7 | 16.54 | 16.58 | 122.02 | 6,577.0 | 7,160.8 |
| 9 | 2,051.3 | 1,995.1 | 2,659.1 | 16.27 | 16.25 | 101.42 | 6,917.0 | 7,427.2 |
| 10 | 1,489.7 | 1,280.6 | 2,572.6 | 15.31 | 15.29 | 36.07 | 6,066.5 | 6,808.1 |

#### 9.1.4 Concurrency = 8 (16 对话, 160 请求)

| Turn | TTFT Mean | TTFT Median | TTFT P99 | TPOT Mean | ITL Mean | ITL P99 | E2EL Mean | E2EL P99 |
|:----:|:---------:|:-----------:|:--------:|:---------:|:--------:|:-------:|:---------:|:--------:|
| 1 | 7,332.8 | 6,101.8 | 13,782.6 | 17.40 | 17.46 | 127.73 | 12,536.1 | 20,010.7 |
| 2 | 6,170.4 | 4,789.1 | 14,101.3 | 22.72 | 22.67 | 321.68 | 12,964.3 | 19,986.2 |
| 3 | 3,771.2 | 4,035.8 | 7,457.4 | 24.63 | 24.61 | 372.80 | 11,134.9 | 14,869.2 |
| 4 | 3,655.1 | 3,162.9 | 7,567.0 | 24.71 | 24.63 | 380.76 | 11,044.3 | 14,857.2 |
| 5 | 3,039.4 | 2,261.9 | 6,647.6 | 24.56 | 24.48 | 387.08 | 10,384.1 | 13,927.9 |
| 6 | 3,675.4 | 3,515.1 | 6,741.2 | 24.87 | 24.78 | 349.91 | 11,110.4 | 13,950.1 |
| 7 | 3,458.3 | 2,970.4 | 6,647.9 | 24.78 | 24.75 | 405.73 | 10,866.8 | 13,664.8 |
| 8 | 3,193.1 | 2,393.0 | 6,318.3 | 24.75 | 24.70 | 405.15 | 10,594.7 | 13,368.8 |
| 9 | 3,625.4 | 3,613.3 | 6,291.8 | 25.01 | 25.07 | 413.08 | 11,102.8 | 13,420.5 |
| 10 | 3,421.9 | 3,082.5 | 5,932.1 | 22.23 | 22.17 | 356.16 | 10,068.8 | 13,151.6 |

### 9.2 配置文件引用

- Benchmark 配置: `vigil/examples/pd_kimi_bench_a_70k.yaml`
- 模型配置: `/mnt/lustre/hf-models/hub/models--nvidia--Kimi-K2.5-NVFP4/snapshots/c0285e649c34d4386b01e38abca642c06cbe014e/config.json`
- 量化配置: 同目录 `hf_quant_config.json`
- 机器文档: `vllm/sprint_dist_kv_doc/machine_doc_cn.md`

### 9.3 数据来源

所有性能数据来自以下 JSON 文件:
- `openai-chat-infqps-concurrency1-Kimi-K2.5-NVFP4-20260411-015541.json`
- `openai-chat-infqps-concurrency2-Kimi-K2.5-NVFP4-20260411-015541.json`
- `openai-chat-infqps-concurrency4-Kimi-K2.5-NVFP4-20260411-015541.json`
- `openai-chat-infqps-concurrency8-Kimi-K2.5-NVFP4-20260411-015541.json`
- `results.json` (pipeline 汇总)

vmon 指标: `vmon/vmon_vllm-bench.json` (258 samples, 2s interval, ~514s 覆盖全部 benchmark)

### 9.4 名词解释

| 缩写 | 全称 | 说明 |
|:-----|:-----|:-----|
| TTFT | Time to First Token | 从请求发出到收到第一个 token 的时间 |
| TPOT | Time per Output Token | 每个输出 token 的平均生成时间 (排除首 token) |
| ITL | Inter-token Latency | 相邻两个 token 之间的时间间隔 |
| E2EL | End-to-End Latency | 从请求发出到最后一个 token 生成完毕的总时间 |
| PD | Prefill-Decode | 预填充-解码分离架构 |
| TP | Tensor Parallelism | 张量并行 |
| MoE | Mixture of Experts | 混合专家模型 |
| KV | Key-Value | 注意力机制中的键值缓存 |
| NVFP4 | NVIDIA 4-bit Floating Point | NVIDIA 4位浮点量化格式 |
| HBM3e | High Bandwidth Memory 3e | 高带宽显存 |
| NVLink-C2C | NVLink Chip-to-Chip | 芯片间高速互联 |
