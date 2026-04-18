# L2 CPU DDR Offload Storage Modeling

> Based on 2026-04-09 Distributed KV Cache Store discussion (Yifan Qiao / Zijing Liu / Woosuk Kwon),
> Comprehensive analysis of GB200 cluster architecture document (`machine_doc.md`), HBM storage modeling (`hbm_storage_modeling.md`)

---

## 1. Problem statement: Why L2 CPU Offload is needed

### 1.1 HBM KV Cache capacity bottleneck (modeled conclusion)

In the actual measurement environment of Kimi-K2.5-NVFP4 + TP=4 + GB200:

| Metrics | Prefill node | Decode node |
|------|-------------|-------------|
| Available KV Cache / GPU | **16.85 GiB** | **15.04 GiB** (effective 14.14 GiB) |
| Token Capacity | 461,216 | 411,712 |
| Concurrency @131K | 3.52x | 3.14x |

### 1.2 Pressure of Agentic Workload

Actual coding application data from Crusoe:

- ISL (Input Sequence Length): P50 70K, P90 90K, P99 120K
- OSL (Output Sequence Length): P50 200, P90 300, P99 1000
- After 10-20 rounds of dialogue, the total context reaches 50K-200K tokens
- **Lots of repeated prefixes**: system prompt + skills/memory (15-40K) are fixed throughout the conversation

### 1.3 HBM cannot carry concurrent working sets

benchmark parameters (70K input, 300 output, 10 turns, global prefix 15%, conversation prefix 75%):

| Number of concurrency C | Number of conversations | Global + Conv Prefix working set | Turn 1 full working set |
|----------|-------|---------------------------|----------------|
| 1 | 2 | 4.22 GiB | 4.73 GiB |
| 2 | 4 | 8.06 GiB | 9.08 GiB |
| **4** | **8** | **15.73 GiB** | **17.78 GiB** |
| **8** | **16** | **31.08 GiB** | **35.18 GiB** |

**Key Conclusions**:
- Prefix working set (15.73 GiB) at C=4 exceeds Decode node effective KV cache (14.14 GiB)
- At C=8 (31.08 GiB) far exceeds any single GPU HBM KV budget
- **Inactive KV cache must be offloaded to L2 CPU DDR**

### 1.4 The necessity of external prefix cache pool

Key figures from the Distributed KV Cache Store discussion:

| Model | KV cache size per token | 70K sequence size | 1000 request data set total requirements |
|------|----------------------|-------------|---------------------|
| Llama3 8B (bf16) | 128 KB | 8.53 GB | — |
| Kimi-K2.5 (bf16) | 68.625 KB | 4.57 GB | — |
| **Kimi-K2.5 (fp8)** | **39.1 KB** | **2.67 GB** | **2.67 TB** |Even after 56.9x compression of MLA + FP8, the prefix cache for 1000 70K requests still requires **2.67 TB**, which is far more than the single-node HBM capacity.

---

## 2. L2 CPU DDR hardware features

### 2.1 GB200 NVLink-C2C: Not traditional PCIe

The interconnection between GPU and CPU in GB200 Superchip is **NVLink-C2C**, which is the key to understanding L2 offload performance:

| Dimensions | NVLink-C2C (GB200) | PCIe Gen5 (legacy x86) |
|------|--------------------|-----------------------|
| Bandwidth | **900 GB/s bidirectional** (450 GB/s one-way) | 128 GB/s (64 one-way) |
| Multiples | **~7x PCIe Gen5** | Benchmarks |
| Memory consistency | **Hardware level cache coherent** | None |
| Address Space | **CPU/GPU Unified Virtual Address Space** | Standalone |
| Data handling | No need for explicit memcpy, hardware automatically | Requires cudaMemcpy |
| Energy Efficiency | 25x better than PCIe Gen5 | Benchmarks |

### 2.2 Single node CPU memory topology
```
┌─────────────── Compute Tray (single node) ─────────────────────┐
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
│ Total Node CPU Memory: ~882 GiB LPDDR5X │
│ /dev/shm: 442G (about half of the physical memory) │
└───────────────────────────────────────────────────────────┘
```
### 2.3 Key bandwidth constraints
```
Bandwidth bottleneck analysis on GPU ←→ CPU DDR path:

GPU HBM internal: 8,000 GB/s ← not a bottleneck
NVLink-C2C: 450 GB/s ← One-way, 2 GPUs shared 900 GB/s
CPU LPDDR5X itself: ~500 GB/s ← May be a bottleneck (close to C2C)
  
Actual available bandwidth ≈ min (C2C one-way, DDR bandwidth / 2)
                ≈ min(450, 250) 
≈ ~250 GB/s per GPU (conservative estimate, consider 2 GPUs sharing DDR)
```
**Important Constraint**: Each Grace CPU is connected to 2 GPUs, 900 GB/s C2C bandwidth and ~500 GB/s DDR bandwidth are shared by the 2 GPUs. When two GPUs perform KV offload/reload at the same time, the actual available bandwidth of a single GPU is about half of the peak value.

---

## 3. L2 CPU DDR capacity modeling

### 3.1 Available capacity estimation

| Item | Size | Description |
|------|------|------|
| Node Physical CPU Memory | ~882 GiB | 2 Grace CPUs, LPDDR5X |
| OS + System Services | ~20-40 GiB | Kernel, systemd, multi-user (4-7 users) |
| vLLM process (non-KV) | ~10-20 GiB | Python runtime, scheduler, ZMQ, etc. |
| PyTorch CPU tensors | ~5-10 GiB | Temporary buffer for model loading, etc. |
| **Available for KV Cache** | **~800-840 GiB** | Conservatively calculated at **800 GiB** |

### 3.2 KV Cache capacity (L2)

Based on Kimi-K2.5 (MLA + FP8 + Eagle3) 39,232 bytes = 38.3 KiB per token:

| Indicators | Values |
|------|-----|
| L2 free space | ~800 GiB |
| Cacheable tokens | 800 GiB / 38.3 KiB ≈ **21,370,000 tokens** (21.37 million) |
| Equivalent to 131K requests | 21.37 million / 131,072 ≈ **163** |
| Equivalent to 70K requests | 21.37 million / 70,000 ≈ **305** |

### 3.3 L1 HBM vs L2 CPU DDR capacity comparison

| Tier | Single GPU capacity | Single node (4 GPU) equivalent | 131K concurrency | 70K concurrency |
|------|-----------|-----------|-----------|----------|
| L1 HBM (Prefill) | 16.85 GiB | 16.85 GiB (TP Replication) | 3.52 | 6.59 |
| L1 HBM (Decode) | 14.14 GiB (Effective) | 14.14 GiB (TP Replication) | 2.95 | 5.53 |
| **L2 CPU DDR** | — | **~800 GiB** | **~163** | **~305** |
| **L1 + L2 Total** | — | **~815 GiB** | **~166** | **~311** |

**Key Insight**: The KV cache capacity of L2 CPU DDR is **~47x** (800 / 16.85) of L1 HBM, which is the largest leverage to expand the prefix cache hit rate.

### 3.4 Comparison of 1000 request data set requirements

| Storage layer | Number of cacheable 70K requests | Proportion of 1000 requests | Description |
|-------|------------------|----------------|------|
| L1 HBM (single GPU) | ~6 | 0.6% | Active requests only |
| **L2 CPU DDR (single node)** | **~305** | **30.5%** | Warm cache, high hit potential |
| L3 NVMe RAID0 (single node) | ~4,700 | 470% (full coverage) | Cold cache, 12T |
| **Full cabinet L2 (18 nodes RDMA)** | **~5,490** | **549%** (full coverage) | Distributed across nodes |A single node L2 can cover 30.5% of the 1000 request data set. If prefix sharing (global prefix 15% + conversation prefix 75%) is taken into account, the actual unique KV amount is much less than 1000 × 2.67 GB.

---

## 4. Transmission delay modeling

### 4.1 KV Cache Load Latency

From a TTFT perspective, TTFT = max (GPU prefill calculation time, KV cache loading time).

Measured data of Llama-3.1-8B-Instruct on H100 from the Distributed KV Cache Store discussion (attached chart):
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
**Key observation**: CPU load time grows linearly with prompt length (limited by bandwidth), while GPU compute grows approximately quadratically (attention is O(n²)). **Above 30K tokens, loading KV from CPU is faster than recalculating. **

### 4.2 Bandwidth Advantage on GB200

H100 uses PCIe Gen5 (~64 GB/s one-way) to connect to the CPU; GB200 uses NVLink-C2C (~450 GB/s one-way), which increases bandwidth by ~7x**.

Take Kimi-K2.5 70K tokens as an example:

| Path | Data volume | Bandwidth | Transmission time |
|------|--------|------|---------|
| H100 PCIe Gen5 | 2.67 GB | ~50 GB/s (tested) | ~53 ms |
| **GB200 C2C (ideal)** | 2.67 GB | ~250 GB/s (conservative) | **~11 ms** |
| **GB200 C2C (2 GPU concurrent)** | 2.67 GB × 2 | ~500 GB/s (shared) | **~11 ms / GPU** |

Compared to the prefill computation time of Kimi-K2.5 on GB200 (estimated to be on the order of hundreds of ms for 70K tokens), the L2 loading latency is **much lower** than the GPU computation time and can be almost completely hidden behind the computation.

### 4.3 L2 loading delay in different scenarios

| Scenario | Data volume | GB200 C2C transfer time (conservative 250 GB/s) |
|------|--------|----------------------------------|
| 7K unique segment | 262 MiB | **~1 ms** |
| 10.5K global prefix | 384 MiB | **~1.5 ms** |
| 52.5K conv prefix | 1.92 GiB | **~7.7 ms** |
| 70K full turn | 2.56 GiB | **~10.2 ms** |
| 131K max context | 4.79 GiB | **~19.2 ms** |

---

## 5. Offload strategy analysis

### 5.1 Current vLLM Connector Architecture
```
┌─────────────────────────────────────────────────────────┐
│                   MultiConnector                         │
│  ┌──────────────────┐  ┌─────────────────────────────┐  │
│  │ NixlConnector    │  │ MooncakeStoreConnector       │  │
│ │ (P→D direct transfer) │ │ (L2 CPU + L3 NVMe offload) │ │
│ │ NVLink / RDMA │ │ + cross-node distributed prefix cache │ │
│  └──────────────────┘  └─────────────────────────────┘  │
│                                                         │
│ save: write all sub-connectors │
│ load: try in order (NixlConnector takes precedence) │
└─────────────────────────────────────────────────────────┘
```
MooncakeStoreConnector implements hierarchical storage of KV cache through MooncakeDistributedStore + TransferEngine:
```
Mooncake Transfer Engine
├── GPU HBM (L1) ←→ CPU DDR (L2): NVLink-C2C, 900 GB/s
│ Hardware cache coherent, unified memory available
│
├── CPU DDR (L2) ←→ Remote CPU DDR / GPU: RDMA or NVLink
│ Cross-node KV Cache reuse (removable NVLink full interconnection within the cabinet)
│
├── CPU DDR (L2) ←→ Local NVMe (L3): PCIe / DMA
│ This node disk cache read and write
│
└── Remote NVMe (L3) cross-node access:
Remote NVMe → Remote CPU DDR → RDMA/NVLink → Local GPU
```
### 5.2 Eviction strategy considerations

When the L1 HBM KV pool is nearly full, inactive KV blocks need to be offloaded to L2:

| Strategy | Description | Applicable Scenarios |
|------|------|---------|
| **LRU** | The least recently used blocks are uninstalled first | Universal, friendly to multiple rounds of dialogue |
| **Prefix-aware** | The shared prefix is retained in L1, and the non-shared part is uninstalled first | Agentic workload (a large number of prefix reuse) |
| **Proactive offload** | After Prefill is completed, actively push the KV cache down to L2 | PD separation architecture, prefill nodes actively clean up |
| **Frequency-based** | Determine the residency level based on the hit frequency of prefix hash | A production environment with stable request distribution |

### 5.3 Two usage modes of Unified Memory on GB200

1. **Implicit Access (Unified Memory)**: When the GPU accesses the CPU memory, the hardware automatically pulls the data page through NVLink-C2C. Suitable for fine-grained, on-demand loading scenarios.
   - Advantages: No need to manage data transfer, simple programming model
   - Disadvantages: page faults bring uncertain delays and it is difficult to control prefetching

2. **Explicit transfer (DMA Copy Engine)**: The application actively schedules batch transfers and utilizes the DMA engine of Grace CPU.
   - Advantages: controllable throughput, suitable for early prefetching
   - Disadvantages: Requires scheduling logic
   - **Mooncake currently uses this method** (TransferEngine zero-copy RDMA/DMA)

---

## 6. Capacity planning: single node L1+L2 joint budget

### 6.1 Benchmark scenario analysis

Take the `pd_kimi_bench_a_70k` parameter as an example (70K input, 300 output, 10 turns, global 15%, conv 75%):

#### Assumption: perfect prefix caching + L1/L2 layering
```
L1 HBM (Decode, valid): 14.14 GiB
→ Active requests to KV cache (requests being decoded)
→ About 3 131K or 5 70K concurrent requests

L2 CPU DDR: ~800 GiB
→ Inactive but potentially reusable prefix KV cache
  → global prefix (10.5K) + conversation prefixes
```
#### L1+L2 joint coverage of each concurrency level

| Concurrency C | Active request KV (L1 requirement) | Prefix working set (L2 candidate) | Can L1 accommodate active | Can L2 accommodate prefix |
|--------|---------------------|----------------------|---------------|------------------|
| 1 | 2.56 GiB (1×70K) | 2.30 GiB (1 global + 2 conv) | ✅ plenty | ✅ plenty |
| 2 | 5.12 GiB (2×70K) | 4.22 GiB | ✅ Yes | ✅ Plenty |
| 4 | 10.24 GiB (4×70K) | 8.06 GiB | ✅ Just enough | ✅ Plenty |
| 8 | 20.48 GiB (8×70K) | 15.73 GiB | ❌ Beyond L1 | ✅ Sufficient |
| 16 | 40.96 GiB | 31.08 GiB | ❌ Much More | ✅ Ample |

**Conclusion**: L2's 800 GiB capacity is **more than enough** for the prefix cache. The bottleneck is the number of active requests that the L1 HBM can serve simultaneously (limited by decode concurrency).

### 6.2 Layered strategy for C=8 scenario
```
With C=8, 8 active requests require 20.48 GiB KV cache (all 70K tokens):

Option A: Pure HBM (not feasible)
Requires: 20.48 GiB
Available: 14.14 GiB
Gap: -6.34 GiB ❌

Option B: L1+L2 layering (feasible)
L1 reserved: unique segment (7K tokens) × 8 = 2.05 GiB per request
+ output tokens being decoded
L2 cache: global prefix (0.385 GiB, only 1 copy)
+ 8 conversation prefixes (8 × 1.919 = 15.35 GiB)
Total L2 requirement: ~15.73 GiB (only 2% of the 800 GiB available for L2)
  
L1 actual requirement: ~2.05 GiB + decode buffers ≈ 3-5 GiB
L1 available: 14.14 GiB → ✅ plenty

Solution C: Aggressive offload — only keep current decode step required in L1
L1: Only retain the complete KV of 1-2 requests + required for the attention calculation of the current batch
L2: All others offload
Advantages: Higher concurrency, but limited by L2→L1 loading delay
```
### 6.3 Total L2 capacity of the entire cabinet (18 nodes)

| Metrics | Single node | Full cabinet (18 nodes) |
|------|-------|---------------|
| CPU DDR available | ~800 GiB | **~14.4 TiB** |
| Cacheable 70K requests | ~305 | **~5,490** |
| Cacheable 131K requests | ~163 | **~2,934** |
| 1000 requested data set | 30.5% | **549%** (much more than full coverage) |

Through RDMA cross-node access to L2, the entire cabinet can build a **14.4 TiB distributed prefix cache pool**, completely covering 2.67 TB of 1000 request data sets.

---

## 7. Transmission bandwidth bottleneck analysis

### 7.1 L2→L1 loaded bandwidth limit

When multiple GPUs load KV cache from L2 at the same time:
```
Bandwidth allocation per node
┌──────────────────────────────────────────────────┐
│                                                  │
│  Grace CPU 0 (LPDDR5X ~500 GB/s)                │
│  ├── GPU 0: C2C ≤ 450 GB/s ──┐                  │
│ └── GPU 1: C2C ≤ 450 GB/s ──┼── Shared DDR bandwidth │
│ Actual: ~200-250 GB/s per GPU │ │
│                               │                  │
│  Grace CPU 1 (LPDDR5X ~500 GB/s)                │
│  ├── GPU 2: C2C ≤ 450 GB/s ──┐                  │
│ └── GPU 3: C2C ≤ 450 GB/s ──┼── Shared DDR Bandwidth │
│ Actual: ~200-250 GB/s per GPU │ │
│                               │                  │
│ Total node L2→L1 aggregate bandwidth: ~800-1000 GB/s (4 GPU concurrent)│
└──────────────────────────────────────────────────┘
```
### 7.2 Cross-node L2 access bandwidth

| Path | Bandwidth | Delay | Description |
|------|------|------|------|
| Native L2 (NVLink-C2C) | ~250 GB/s / GPU | ~μs | Optimal path |
| Remote L2 in the cabinet (via NVLink transit) | ~450 GB/s (GPU-GPU) | ~10μs | Full interconnection via NVLink Switch |
| Remote L2 in Cabinet (RDMA) | ~50 GB/s (RoCE) | ~10μs | Go RoCE RDMA |
| **Cross-cabinet remote L2 (RDMA)** | ~50 GB/s (RoCE) | ~50-100μs | Slowest path |

### 7.3 Loading delay vs Decode iteration interval

Each iteration (step) of the Decode phase requires reading the KV cache of all active requests. If part of the KV is at L2:

| decode batch size | Total amount of KV to be read per step | L2 transfer time (250 GB/s) |
|-------------------|------------------------|-----------------------|
| 1 (131K) | 4.79 GiB (all at L2) | ~19 ms |
| 4 (70K each) | 10.24 GiB | ~41 ms |
| 8 (70K each) | 20.48 GiB | ~82 ms |

**Note**: In practice, decode does not need to be fully loaded from L2 in every step. The KV cache will reside in the L1 HBM during decoding, only:
1. **New requests need to be loaded from L2 when loading**
2. When an **eviction occurs**, it needs to be moved between L1 and L2

Therefore, L2 transmission mainly affects the TTFT of new requests, not decode throughput.

---

## 8. Full-level storage comparison overview
```
Fast ─────────────────────────────────────────── Slow
Small capacity ──────────────────────────────────────────── Large capacity

┌──────────────────────────────────────────────────────────────┐
│ L1 · GPU HBM (184.31 GiB per GPU, effective KV ~14-17 GiB) │
│ Bandwidth: 8 TB/s (HBM internal) | Purpose: Active KV, calculating │
│ Capacity: 3-6 70K requests / GPU │
│ KV synchronization when TP=4, concurrency is subject to single GPU │
├──────────────────────────────────────────────────────────────┤
│ L2 · CPU DDR (~882 GiB per node, ~800 GiB available) │
│ Bandwidth: 250 GB/s/GPU (NVLink-C2C, conservative) | Latency: ~μs │
│ Purpose: Warm KV Cache (prefix, recently inactive requests) │
│ Capacity: ~305 70K requests/node │
│ Advantages: GB200 NVLink-C2C brings it close to HBM extension layer │
├──────────────────────────────────────────────────────────────┤
│ L3 · Local NVMe RAID0 (~12T per node) │
│ Bandwidth: 25+ GB/s (4-disk RAID0) | Latency: ~100μs │
│ Purpose: Cold KV Cache, large-capacity prefix persistent cache │
│ Capacity: ~4,700 70K requests/node │
│ Path: /mnt/data (xfs) │
├──────────────────────────────────────────────────────────────┤
│ Distributed · Full Cabinet L2 (18 nodes, RDMA/NVLink) │
│ Bandwidth: 50 GB/s (RDMA) or 450 GB/s (NVLink transit) │
│ Capacity: ~14.4 TiB → ~5,490 70K requests │
│ Coverage: 2.67 TB dataset → Full coverage │
└──────────────────────────────────────────────────────────────┘
```
---

## 9. Suggestions and follow-up directions

### 9.1 Recent Priority

1. **Verify the actual bandwidth of NVLink-C2C**: Use `cuda-memcpy-bench` or the benchmark that comes with Mooncake to measure the actual transmission bandwidth of GPU↔CPU DDR. The nominal 450 GB/s one-way, the actual effective bandwidth at the KV cache block granularity (18 KiB/layer) needs to be measured.

2. **Determine CPU DDR available amount**: When vLLM is running, actually measure the remaining amount of CPU memory after the Python process + Mooncake store is registered. The current 800 GiB is an estimate.

3. **Measure prefix cache hit rate**: Under agentic workload (SWEBench-pro replay), count the ratio of L1 miss + L2 hit to quantify the actual benefits of L2 offload.

### 9.2 Performance optimization direction

- **Prefetch pipeline**: When the scheduler allocates requests, it prefetches the required prefix from L2 to L1 in advance, hiding the transmission delay.
- **Prefix pinning**: High-frequency shared prefix (such as system prompt) is resident in L1 and does not participate in eviction
- **Hierarchical eviction**: L1 evict → L2 (not discarded directly), L2 evict → L3 or discarded

### 9.3 To be modeled

- [ ] Actual read and write latency and throughput modeling of L3 NVMe RAID0 layer
- [ ] Actual latency comparison of cross-node RDMA vs NVLink transit
- [ ] Impact of different eviction strategies on TTFT P99
- [ ] Capacity change when KV cache does not require TP replication in DP4+EP scenario

---

## Appendix A: Numerical Cheat Sheet
```
┌──────────────────────────────────────────────────────────────────┐
│ KV cache per token (Kimi-K2.5, MLA+FP8+Eagle3): 38.3 KiB │
│ KV cache per token (Kimi-K2.5, MLA+FP8, without Eagle3): 34.3 KiB │
│ KV cache per token (Kimi-K2.5, MLA+bf16): 68.625 KB │
│ KV cache per token (Llama3 8B, MHA+bf16): 128 KB │
│                                                                  │
│ L1 HBM available KV / GPU: Prefill 16.85 / Decode 14.14 GiB │
│ L2 CPU DDR available/node: ~800 GiB (estimated) │
│ L3 NVMe RAID0 / Node: ~12 TiB │
│                                                                  │
│ L2 capacity vs L1: ~47x (800 / 16.85) │
│ L2 bandwidth (C2C, conservative): ~250 GB/s / GPU (unidirectional) │
│ L2 bandwidth (C2C, peak): ~450 GB/s / GPU (unidirectional, exclusive) │
│ L2→L1 70K token loading: ~10-20 ms (2.56 GiB, 250 GB/s) │
│                                                                  │
│ Total L2 capacity of the entire cabinet: ~14.4 TiB (18 × 800 GiB) │
│ The entire cabinet L2 can cache 70K requests: ~5,490 │
│ Full cabinet L2 vs 1000 request set: 549% (far more than full coverage) │
│                                                                  │
│ MLA+FP8 combined compression ratio: 56.9x (vs MHA+bf16) │
│ NVLink-C2C vs PCIe Gen5: ~7x bandwidth, 25x energy efficiency │
└──────────────────────────────────────────────────────────────────┘
```
---

## Appendix B: Source Documentation

| Document | Content | Date |
|------|------|------|
| `2026-04-09-distributed-kv-cache-store.md` (zip) | Distributed KV Cache Store Discussion: agentic workload motivation, KV size calculation, Connector API, Mooncake architecture | 2026-04-09 |
| `machine_doc.md` | NVL72 GB200 cluster storage and interconnection architecture, KV Cache layered overview, NVLink-C2C features | 2026-04-09 |
| `hbm_storage_modeling.md` | Kimi-K2.5-NVFP4 HBM storage modeling: video memory allocation, KV cache structure, concurrent capacity calculation | 2026-04-11 |

---

*Document generation time: 2026-04-14*
*Analysis method: Data cross-validation and extended modeling based on three source documents*