# L2 CPU DDR Offload Storage Modeling

> 基于 2026-04-09 Distributed KV Cache Store 讨论 (Yifan Qiao / Zijing Liu / Woosuk Kwon)、
> GB200 集群架构文档 (`machine_doc_cn.md`)、HBM 存储建模 (`hbm_storage_modeling.md`) 综合分析

---

## 1. 问题陈述：为什么需要 L2 CPU Offload

### 1.1 HBM KV Cache 容量瓶颈（已建模结论）

在 Kimi-K2.5-NVFP4 + TP=4 + GB200 的实测环境下：

| 指标 | Prefill 节点 | Decode 节点 |
|------|-------------|-------------|
| 可用 KV Cache / GPU | **16.85 GiB** | **15.04 GiB** (有效 14.14 GiB) |
| Token 容量 | 461,216 | 411,712 |
| 并发 @131K | 3.52x | 3.14x |

### 1.2 Agentic 工作负载的压力

从 Crusoe 的实际 coding application 数据：

- ISL (Input Sequence Length): P50 70K, P90 90K, P99 120K
- OSL (Output Sequence Length): P50 200, P90 300, P99 1000
- 多轮对话 10-20 turn 后，总 context 达到 50K-200K tokens
- **前缀大量重复**：system prompt + skills/memory (15-40K) 在整个对话中固定不变

### 1.3 HBM 无法承载的并发工作集

benchmark 参数 (70K input, 300 output, 10 turns, global prefix 15%, conversation prefix 75%)：

| 并发数 C | 对话数 | Global + Conv Prefix 工作集 | Turn 1 全量工作集 |
|----------|-------|---------------------------|-----------------|
| 1 | 2 | 4.22 GiB | 4.73 GiB |
| 2 | 4 | 8.06 GiB | 9.08 GiB |
| **4** | **8** | **15.73 GiB** | **17.78 GiB** |
| **8** | **16** | **31.08 GiB** | **35.18 GiB** |

**关键结论**：
- C=4 时前缀工作集 (15.73 GiB) 已超过 Decode 节点有效 KV cache (14.14 GiB)
- C=8 时 (31.08 GiB) 远超任何单 GPU 的 HBM KV 预算
- **必须将不活跃的 KV cache 卸载到 L2 CPU DDR**

### 1.4 外部 prefix cache pool 的必要性

来自 Distributed KV Cache Store 讨论的关键数据：

| 模型 | 每 token KV cache 大小 | 70K 序列大小 | 1000 请求数据集总需求 |
|------|----------------------|-------------|---------------------|
| Llama3 8B (bf16) | 128 KB | 8.53 GB | — |
| Kimi-K2.5 (bf16) | 68.625 KB | 4.57 GB | — |
| **Kimi-K2.5 (fp8)** | **39.1 KB** | **2.67 GB** | **2.67 TB** |

即使经过 MLA + FP8 的 56.9x 压缩，1000 个 70K 请求的 prefix cache 仍需 **2.67 TB**，远超单节点 HBM 容量。

---

## 2. L2 CPU DDR 硬件特性

### 2.1 GB200 NVLink-C2C：不是传统 PCIe

GB200 Superchip 中 GPU 与 CPU 之间的互联是 **NVLink-C2C**，这是理解 L2 offload 性能的关键：

| 维度 | NVLink-C2C (GB200) | PCIe Gen5 (传统 x86) |
|------|--------------------|-----------------------|
| 带宽 | **900 GB/s 双向** (450 GB/s 单向) | 128 GB/s (64 单向) |
| 倍数 | **~7x PCIe Gen5** | 基准 |
| 内存一致性 | **硬件级 cache coherent** | 无 |
| 地址空间 | **CPU/GPU 统一虚拟地址空间** | 独立 |
| 数据搬运 | 无需显式 memcpy，硬件自动 | 需要 cudaMemcpy |
| 能效 | 比 PCIe Gen5 高 25x | 基准 |

### 2.2 单节点 CPU 内存拓扑

```
┌────────────── Compute Tray (单节点) ──────────────────────┐
│                                                           │
│  Superchip 0                    Superchip 1               │
│  ┌─────────────────────┐       ┌─────────────────────┐    │
│  │ Grace CPU 0         │       │ Grace CPU 1         │    │
│  │ ~441 GiB LPDDR5X    │       │ ~441 GiB LPDDR5X    │    │
│  │   ↕ C2C 900GB/s     │       │   ↕ C2C 900GB/s     │    │
│  │ GPU 0    GPU 1      │       │ GPU 2    GPU 3      │    │
│  │ 192G     192G       │       │ 192G     192G       │    │
│  └─────────────────────┘       └─────────────────────┘    │
│                                                           │
│  节点总 CPU 内存: ~882 GiB LPDDR5X                         │
│  /dev/shm: 442G (约一半物理内存)                            │
└───────────────────────────────────────────────────────────┘
```

### 2.3 关键带宽约束

```
GPU ←→ CPU DDR 路径上的带宽瓶颈分析:

  GPU HBM 内部:        8,000 GB/s   ← 不是瓶颈
  NVLink-C2C:            450 GB/s   ← 单向, 2 GPU 共享 900 GB/s
  CPU LPDDR5X 本身:      ~500 GB/s   ← 可能成为瓶颈 (与 C2C 接近)
  
  实际可用带宽 ≈ min(C2C 单向, DDR 带宽 / 2)
                ≈ min(450, 250) 
                ≈ ~250 GB/s per GPU (保守估计, 考虑 2 GPU 共享 DDR)
```

**重要约束**：每个 Grace CPU 连接 2 个 GPU，900 GB/s C2C 带宽和 ~500 GB/s DDR 带宽由 2 个 GPU 共享。在两个 GPU 同时做 KV offload/reload 时，单 GPU 实际可用带宽约为峰值的一半。

---

## 3. L2 CPU DDR 容量建模

### 3.1 可用容量估算

| 项目 | 大小 | 说明 |
|------|------|------|
| 节点物理 CPU 内存 | ~882 GiB | 2 个 Grace CPU, LPDDR5X |
| OS + 系统服务 | ~20-40 GiB | 内核、systemd、多用户 (4-7 users) |
| vLLM 进程 (非 KV) | ~10-20 GiB | Python runtime、scheduler、ZMQ 等 |
| PyTorch CPU tensors | ~5-10 GiB | 模型加载临时缓冲等 |
| **可用于 KV Cache** | **~800-840 GiB** | 保守按 **800 GiB** 计算 |

### 3.2 KV Cache 容量 (L2)

基于 Kimi-K2.5 (MLA + FP8 + Eagle3) 每 token 39,232 bytes = 38.3 KiB：

| 指标 | 值 |
|------|-----|
| L2 可用空间 | ~800 GiB |
| 可缓存 tokens | 800 GiB / 38.3 KiB ≈ **21,370,000 tokens** (2137 万) |
| 等效 131K 请求数 | 2137 万 / 131,072 ≈ **163 个** |
| 等效 70K 请求数 | 2137 万 / 70,000 ≈ **305 个** |

### 3.3 L1 HBM vs L2 CPU DDR 容量对比

| 层级 | 单 GPU 容量 | 单节点 (4 GPU) 等效 | 131K 并发数 | 70K 并发数 |
|------|-----------|-------------------|-----------|----------|
| L1 HBM (Prefill) | 16.85 GiB | 16.85 GiB (TP 复制) | 3.52 | 6.59 |
| L1 HBM (Decode) | 14.14 GiB (有效) | 14.14 GiB (TP 复制) | 2.95 | 5.53 |
| **L2 CPU DDR** | — | **~800 GiB** | **~163** | **~305** |
| **L1 + L2 合计** | — | **~815 GiB** | **~166** | **~311** |

**关键洞察**：L2 CPU DDR 的 KV cache 容量是 L1 HBM 的 **~47x**（800 / 16.85），是扩展 prefix cache 命中率的最大杠杆。

### 3.4 对比 1000 请求数据集需求

| 存储层 | 可缓存 70K 请求数 | 占 1000 请求比例 | 说明 |
|-------|-----------------|----------------|------|
| L1 HBM (单 GPU) | ~6 | 0.6% | 仅活跃请求 |
| **L2 CPU DDR (单节点)** | **~305** | **30.5%** | 温缓存, 高命中潜力 |
| L3 NVMe RAID0 (单节点) | ~4,700 | 470% (全覆盖) | 冷缓存, 12T |
| **整柜 L2 (18 节点 RDMA)** | **~5,490** | **549%** (全覆盖) | 分布式跨节点 |

单节点 L2 即可覆盖 1000 请求数据集的 30.5%，如果考虑 prefix 共享 (global prefix 15% + conversation prefix 75%)，实际 unique KV 量远小于 1000 × 2.67 GB。

---

## 4. 传输延迟建模

### 4.1 KV Cache Load 延迟

从 TTFT 角度，TTFT = max(GPU prefill 计算时间, KV cache 加载时间)。

来自 Distributed KV Cache Store 讨论中的 Llama-3.1-8B-Instruct 在 H100 上的实测数据（附件图表）：

```
Prompt Length (K tokens)  |  GPU Compute (ms)  |  Load from CPU (ms)
           10             |       ~200         |       ~200
           20             |       ~400         |       ~350
           30             |       ~700         |       ~500
           40             |      ~1100         |       ~650
           50             |      ~1500         |       ~900
           60             |      ~2100         |      ~1200
           70             |      ~3000         |      ~1500
           80             |      ~3800         |      ~2000
           90             |      ~5200         |      ~2700
          100             |      ~7000         |      ~3500
```

**关键观察**：CPU load 时间随 prompt length 线性增长（受限于带宽），而 GPU compute 近似二次增长（attention 是 O(n²)）。**30K tokens 以上时，从 CPU 加载 KV 比重新计算更快。**

### 4.2 GB200 上的带宽优势

H100 使用 PCIe Gen5 (~64 GB/s 单向) 连接 CPU；GB200 使用 NVLink-C2C (~450 GB/s 单向)，**带宽提升 ~7x**。

以 Kimi-K2.5 70K tokens 为例：

| 路径 | 数据量 | 带宽 | 传输时间 |
|------|--------|------|---------|
| H100 PCIe Gen5 | 2.67 GB | ~50 GB/s (实测) | ~53 ms |
| **GB200 C2C (理想)** | 2.67 GB | ~250 GB/s (保守) | **~11 ms** |
| **GB200 C2C (2 GPU 并发)** | 2.67 GB × 2 | ~500 GB/s (共享) | **~11 ms / GPU** |

与 GB200 上 Kimi-K2.5 的 prefill 计算时间（70K tokens 估计数百 ms 量级）相比，L2 加载延迟 **远低于** GPU 计算时间，几乎可以完全隐藏在计算后面。

### 4.3 不同场景下的 L2 加载延迟

| 场景 | 数据量 | GB200 C2C 传输时间 (保守 250 GB/s) |
|------|--------|----------------------------------|
| 7K unique segment | 262 MiB | **~1 ms** |
| 10.5K global prefix | 384 MiB | **~1.5 ms** |
| 52.5K conv prefix | 1.92 GiB | **~7.7 ms** |
| 70K full turn | 2.56 GiB | **~10.2 ms** |
| 131K max context | 4.79 GiB | **~19.2 ms** |

---

## 5. Offload 策略分析

### 5.1 当前 vLLM Connector 架构

```
┌─────────────────────────────────────────────────────────┐
│                   MultiConnector                         │
│  ┌──────────────────┐  ┌─────────────────────────────┐  │
│  │ NixlConnector    │  │ MooncakeStoreConnector       │  │
│  │ (P→D 直传)       │  │ (L2 CPU + L3 NVMe offload) │  │
│  │ NVLink / RDMA    │  │ + 跨节点分布式 prefix cache  │  │
│  └──────────────────┘  └─────────────────────────────┘  │
│                                                         │
│  save: 写入所有 sub-connector                            │
│  load: 按顺序尝试 (NixlConnector 优先)                   │
└─────────────────────────────────────────────────────────┘
```

MooncakeStoreConnector 通过 MooncakeDistributedStore + TransferEngine 实现 KV cache 的分层存储：

```
Mooncake Transfer Engine
├── GPU HBM (L1) ←→ CPU DDR (L2): NVLink-C2C, 900 GB/s
│   硬件 cache coherent, 可用 unified memory
│
├── CPU DDR (L2) ←→ 远端 CPU DDR / GPU: RDMA 或 NVLink
│   跨节点 KV Cache 复用 (柜内可走 NVLink 全互联)
│
├── CPU DDR (L2) ←→ 本地 NVMe (L3): PCIe / DMA
│   本节点 disk cache 读写
│
└── 远端 NVMe (L3) 跨节点访问:
    远端 NVMe → 远端 CPU DDR → RDMA/NVLink → 本地 GPU
```

### 5.2 Eviction 策略考量

当 L1 HBM KV pool 接近满载时，需要将不活跃的 KV blocks 卸载到 L2：

| 策略 | 描述 | 适用场景 |
|------|------|---------|
| **LRU** | 最近最少使用的 block 优先卸载 | 通用, 对多轮对话友好 |
| **Prefix-aware** | 共享前缀保留在 L1, 非共享部分优先卸载 | Agentic workload (大量 prefix 复用) |
| **Proactive offload** | Prefill 完成后主动将 KV cache 下推到 L2 | PD 分离架构, prefill 节点主动清理 |
| **Frequency-based** | 按 prefix hash 的命中频率决定驻留层级 | 有稳定请求分布的生产环境 |

### 5.3 GB200 上 Unified Memory 的两种使用模式

1. **隐式访问（Unified Memory）**：GPU 访问 CPU 内存时，硬件通过 NVLink-C2C 自动拉取数据页。适合细粒度、按需加载的场景。
   - 优点：无需管理数据搬运，编程模型简单
   - 缺点：page fault 带来不确定延迟，难以控制预取

2. **显式传输（DMA Copy Engine）**：应用主动调度批量传输，利用 Grace CPU 的 DMA 引擎。
   - 优点：吞吐可控，适合提前预取
   - 缺点：需要调度逻辑
   - **Mooncake 当前采用这种方式** (TransferEngine zero-copy RDMA/DMA)

---

## 6. 容量规划：单节点 L1+L2 联合预算

### 6.1 Benchmark 场景分析

以 `pd_kimi_bench_a_70k` 参数为例（70K input, 300 output, 10 turns, global 15%, conv 75%）：

#### 假设：完美 prefix caching + L1/L2 分层

```
L1 HBM (Decode, 有效): 14.14 GiB
  → 活跃请求 KV cache (正在 decode 的请求)
  → 约 3 个 131K 或 5 个 70K 并发请求

L2 CPU DDR: ~800 GiB
  → 不活跃但可能复用的 prefix KV cache
  → global prefix (10.5K) + conversation prefixes
```

#### 各并发级别的 L1+L2 联合覆盖

| 并发 C | 活跃请求 KV (L1 需求) | Prefix 工作集 (L2 候选) | L1 能否容纳活跃 | L2 能否容纳 prefix |
|--------|---------------------|----------------------|---------------|------------------|
| 1 | 2.56 GiB (1×70K) | 2.30 GiB (1 global + 2 conv) | ✅ 充裕 | ✅ 充裕 |
| 2 | 5.12 GiB (2×70K) | 4.22 GiB | ✅ 可以 | ✅ 充裕 |
| 4 | 10.24 GiB (4×70K) | 8.06 GiB | ✅ 刚好 | ✅ 充裕 |
| 8 | 20.48 GiB (8×70K) | 15.73 GiB | ❌ 超出 L1 | ✅ 充裕 |
| 16 | 40.96 GiB | 31.08 GiB | ❌ 远超 | ✅ 充裕 |

**结论**：L2 的 800 GiB 容量对于 prefix 缓存**完全足够**。瓶颈在于 L1 HBM 能同时服务的活跃请求数（受 decode 并发限制）。

### 6.2 C=8 场景的分层策略

```
C=8 时，8 个活跃请求需要 20.48 GiB KV cache (全部 70K tokens):

方案 A: 纯 HBM (不可行)
  需要: 20.48 GiB
  可用: 14.14 GiB
  缺口: -6.34 GiB ❌

方案 B: L1+L2 分层 (可行)
  L1 保留: 每请求 unique segment (7K tokens) × 8 = 2.05 GiB
           + 正在 decode 的 output tokens
  L2 缓存: global prefix (0.385 GiB, 只需 1 份)
           + 8 个 conversation prefix (8 × 1.919 = 15.35 GiB)
  L2 总需: ~15.73 GiB (仅占 L2 可用 800 GiB 的 2%)
  
  L1 实际需求: ~2.05 GiB + decode buffers ≈ 3-5 GiB
  L1 可用: 14.14 GiB → ✅ 充裕

方案 C: 激进 offload — 只在 L1 保留当前 decode step 所需
  L1: 仅保留 1-2 个请求的完整 KV + 当前 batch 的 attention 计算所需
  L2: 其余全部 offload
  优势: 更高并发, 但受限于 L2→L1 加载延迟
```

### 6.3 整柜 (18 节点) 的 L2 总容量

| 指标 | 单节点 | 整柜 (18 节点) |
|------|-------|---------------|
| CPU DDR 可用 | ~800 GiB | **~14.4 TiB** |
| 可缓存 70K 请求 | ~305 个 | **~5,490 个** |
| 可缓存 131K 请求 | ~163 个 | **~2,934 个** |
| 占 1000 请求数据集 | 30.5% | **549%** (远超全覆盖) |

通过 RDMA 跨节点访问 L2，整柜可以构建一个 **14.4 TiB 的分布式 prefix cache pool**，完全覆盖 2.67 TB 的 1000 请求数据集。

---

## 7. 传输带宽瓶颈分析

### 7.1 L2→L1 加载的带宽限制

当多个 GPU 同时从 L2 加载 KV cache 时：

```
                     每节点带宽分配
┌──────────────────────────────────────────────────┐
│                                                  │
│  Grace CPU 0 (LPDDR5X ~500 GB/s)                │
│  ├── GPU 0: C2C ≤ 450 GB/s ──┐                  │
│  └── GPU 1: C2C ≤ 450 GB/s ──┼── 共享 DDR 带宽   │
│  实际: 每 GPU ~200-250 GB/s    │                  │
│                               │                  │
│  Grace CPU 1 (LPDDR5X ~500 GB/s)                │
│  ├── GPU 2: C2C ≤ 450 GB/s ──┐                  │
│  └── GPU 3: C2C ≤ 450 GB/s ──┼── 共享 DDR 带宽   │
│  实际: 每 GPU ~200-250 GB/s    │                  │
│                               │                  │
│  节点总 L2→L1 聚合带宽: ~800-1000 GB/s (4 GPU 并发)│
└──────────────────────────────────────────────────┘
```

### 7.2 跨节点 L2 访问带宽

| 路径 | 带宽 | 延迟 | 说明 |
|------|------|------|------|
| 本地 L2 (NVLink-C2C) | ~250 GB/s / GPU | ~μs | 最优路径 |
| 柜内远程 L2 (via NVLink 中转) | ~450 GB/s (GPU-GPU) | ~10μs | 走 NVLink Switch 全互联 |
| 柜内远程 L2 (RDMA) | ~50 GB/s (RoCE) | ~10μs | 走 RoCE RDMA |
| **跨柜远程 L2 (RDMA)** | ~50 GB/s (RoCE) | ~50-100μs | 最慢路径 |

### 7.3 加载延迟 vs Decode 迭代间隔

Decode 阶段每个迭代（step）需要读取所有活跃请求的 KV cache。如果部分 KV 在 L2：

| decode batch size | 每 step 需读取的 KV 总量 | L2 传输时间 (250 GB/s) |
|-------------------|------------------------|----------------------|
| 1 (131K) | 4.79 GiB (全在 L2 时) | ~19 ms |
| 4 (各 70K) | 10.24 GiB | ~41 ms |
| 8 (各 70K) | 20.48 GiB | ~82 ms |

**注意**：实际中 decode 不需要每 step 都从 L2 全量加载。KV cache 在 decode 期间会驻留在 L1 HBM，只有：
1. **新请求调入**时需要从 L2 加载 prefix
2. **eviction 发生**时需要在 L1 和 L2 之间搬移

所以 L2 传输主要影响的是**新请求的 TTFT**，而非 decode 吞吐。

---

## 8. 全层级存储对比总览

```
速度快 ──────────────────────────────────────────── 速度慢
容量小 ──────────────────────────────────────────── 容量大

┌──────────────────────────────────────────────────────────────┐
│  L1 · GPU HBM (每 GPU 184.31 GiB, 有效 KV ~14-17 GiB)       │
│  带宽: 8 TB/s (HBM 内部)  |  用途: 活跃 KV, 正在计算          │
│  容量: 3-6 个 70K 请求 / GPU                                  │
│  TP=4 时 KV 同步, 并发以单 GPU 为准                            │
├──────────────────────────────────────────────────────────────┤
│  L2 · CPU DDR (每节点 ~882 GiB, 可用 ~800 GiB)               │
│  带宽: 250 GB/s/GPU (NVLink-C2C, 保守)  |  延迟: ~μs          │
│  用途: 温 KV Cache (prefix, 最近不活跃的请求)                   │
│  容量: ~305 个 70K 请求 / 节点                                │
│  优势: GB200 NVLink-C2C 使其接近 HBM 扩展层                   │
├──────────────────────────────────────────────────────────────┤
│  L3 · 本地 NVMe RAID0 (每节点 ~12T)                           │
│  带宽: 25+ GB/s (4 盘 RAID0)  |  延迟: ~100μs                │
│  用途: 冷 KV Cache, 大容量 prefix 持久缓存                     │
│  容量: ~4,700 个 70K 请求 / 节点                              │
│  路径: /mnt/data (xfs)                                       │
├──────────────────────────────────────────────────────────────┤
│  分布式 · 整柜 L2 (18 节点, RDMA/NVLink)                      │
│  带宽: 50 GB/s (RDMA) 或 450 GB/s (NVLink 中转)              │
│  容量: ~14.4 TiB → ~5,490 个 70K 请求                        │
│  覆盖: 2.67 TB 数据集 → 完全覆盖                              │
└──────────────────────────────────────────────────────────────┘
```

---

## 9. 建议与后续方向

### 9.1 近期优先级

1. **验证 NVLink-C2C 实际带宽**：用 `cuda-memcpy-bench` 或 Mooncake 自带的 benchmark 测量 GPU↔CPU DDR 的实际传输带宽。标称 450 GB/s 单向，实际在 KV cache block 粒度 (18 KiB/layer) 下的有效带宽需要实测。

2. **确定 CPU DDR 可用量**：在 vLLM 运行时，实际测量 Python 进程 + Mooncake store 注册后的 CPU 内存剩余量。当前 800 GiB 是估算值。

3. **测量 prefix cache hit rate**：在 agentic workload (SWEBench-pro replay) 下，统计 L1 miss + L2 hit 的比例，量化 L2 offload 的实际收益。

### 9.2 性能优化方向

- **Prefetch pipeline**：scheduler 在分配请求时，提前将所需 prefix 从 L2 预取到 L1，隐藏传输延迟
- **Prefix pinning**：高频共享 prefix (如 system prompt) 常驻 L1，不参与 eviction
- **Hierarchical eviction**：L1 evict → L2（不直接丢弃），L2 evict → L3 或丢弃

### 9.3 待建模

- [ ] L3 NVMe RAID0 层的实际读写延迟和吞吐建模
- [ ] 跨节点 RDMA vs NVLink 中转的实际延迟对比
- [ ] 不同 eviction 策略对 TTFT P99 的影响
- [ ] DP4+EP 场景下 KV cache 不需要 TP 复制时的容量变化

---

## 附录 A: 数值速查表

```
┌──────────────────────────────────────────────────────────────────┐
│ 每 token KV cache (Kimi-K2.5, MLA+FP8+Eagle3):  38.3 KiB       │
│ 每 token KV cache (Kimi-K2.5, MLA+FP8, 无 Eagle3): 34.3 KiB    │
│ 每 token KV cache (Kimi-K2.5, MLA+bf16):         68.625 KB      │
│ 每 token KV cache (Llama3 8B, MHA+bf16):         128 KB         │
│                                                                  │
│ L1 HBM 可用 KV / GPU:     Prefill 16.85 / Decode 14.14 GiB     │
│ L2 CPU DDR 可用 / 节点:    ~800 GiB (估算)                       │
│ L3 NVMe RAID0 / 节点:     ~12 TiB                               │
│                                                                  │
│ L2 容量 vs L1:            ~47x (800 / 16.85)                     │
│ L2 带宽 (C2C, 保守):      ~250 GB/s / GPU (单向)                 │
│ L2 带宽 (C2C, 峰值):      ~450 GB/s / GPU (单向, 独占)           │
│ L2→L1 70K token 加载:     ~10-20 ms (2.56 GiB, 250 GB/s)       │
│                                                                  │
│ 整柜 L2 总容量:            ~14.4 TiB (18 × 800 GiB)             │
│ 整柜 L2 可缓存 70K 请求:   ~5,490 个                              │
│ 整柜 L2 vs 1000 请求集:   549% (远超全覆盖)                       │
│                                                                  │
│ MLA+FP8 联合压缩比:        56.9x (vs MHA+bf16)                   │
│ NVLink-C2C vs PCIe Gen5:  ~7x 带宽, 25x 能效                    │
└──────────────────────────────────────────────────────────────────┘
```

---

## 附录 B: 来源文档

| 文档 | 内容 | 日期 |
|------|------|------|
| `2026-04-09-distributed-kv-cache-store.md` (zip) | Distributed KV Cache Store 讨论: agentic workload 动机、KV 大小计算、Connector API、Mooncake 架构 | 2026-04-09 |
| `machine_doc_cn.md` | NVL72 GB200 集群存储与互联架构、KV Cache 分层总览、NVLink-C2C 特性 | 2026-04-09 |
| `hbm_storage_modeling.md` | Kimi-K2.5-NVFP4 HBM 存储建模: 显存分配、KV cache 结构、并发容量计算 | 2026-04-11 |

---

*文档生成时间: 2026-04-14*
*分析方法: 基于三份源文档的数据交叉验证与延伸建模*
