# Workload Analysis Report for the Kimi K2.5 NVFP4 PD-Disaggregated Inference System

- **Date**: 2026-04-11
- **Configuration file**: `vigil/examples/pd_kimi_bench_a_70k.yaml`
- **Log directory**: `vigil/logs/pd_kimi_bench_a_70k/2026-04-11/20260411_012402/`
- **Model**: nvidia/Kimi-K2.5-NVFP4
- **Hardware**: GB200 NVL72 (2 nodes, 8 GPU)
- **vLLM commit**: `f05fce059` (branch: `feat/mooncake-store-int-notes-20260408`)
- **vigil commit**: `284b3fa` (branch: `feat/vigil-recipe-updates-20260408`)

---

## 1. System Overview

### 1.1 Hardware platform

This test was conducted on an NVIDIA GB200 NVL72 cluster, using 2 nodes and a total of 8 GPUs.

| Resources | Single GPU | Single node (4 GPU) | This test (8 GPU) |
|:-----|:------:|:--------------:|:----------------:|
| GPU model | NVIDIA B200 (Blackwell) | — | — |
| GPU HBM3e | 192 GB | 768 GB | 1,536 GB |
| CPU DDR (LPDDR5X) | — | 882 GiB | 1,764 GiB |
| Local NVMe RAID0 | — | ~12 TB | ~24 TB |
| GPU-CPU Bandwidth (NVLink-C2C) | 225 GB/s | 900 GB/s | — |
| GPU-GPU Bandwidth (NVLink 5) | 1.8 TB/s | — | — |

Total cluster size: 18 nodes, 72 GPUs, 13.8 TB HBM3e, 35 TB shared Luster storage.

### 1.2 Model architecture

| Parameter | Value |
|:-----|:---|
| Model name | Kimi K2.5 NVFP4 |
| Infrastructure | DeepSeek V3 |
| Transformer layers | 61 |
| Hidden size | 7,168 |
| Attention heads / KV heads | 64 / 64 (MHA) |
| Total number of MoE experts / number of activations / number of shares | 384 / 8 / 1 |
| MoE intermediate size (per expert) | 2,048 |
| FFN intermediate size | 18,432 |
| Vocabulary size | 163,840 |
| Maximum context length | 262,144 tokens (256K) |
| Weight quantization | NVFP4 (4-bit, group_size=16) |
| Activation quantization | NVFP4 (4-bit, group_size=16) |
| KV Cache Quantization | FP8 |
| Position encoding | YaRN (rope_theta=50000, factor=64) |
| Attention characteristics | Q LoRA rank=1536, KV LoRA rank=512, QK nope_dim=128, QK rope_dim=64, V head_dim=128 |

This test only uses the language model part (`--language-model-only`) and does not load the visual encoder.

### 1.3 Serving architecture (PD separation)
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
**Prefill vs Decode configuration comparison:**

| Configuration items | Prefill | Decode |
|:-------|:--------|:-------|
| Node | gb200-rack1-09 | gb200-rack1-10 |
| TP | 4 | 4 |
| compile mode | enforce-eager | O3 + cudagraph FULL_DECODE_ONLY |
| KV Transfer | MultiConnector (Nixl + MooncakeStore) | NixlConnector |
| Chunked prefill | enable | enable |
| Prefix caching | Enable | Enable |
| Sleep mode | Enable | Enable |
| Flash Attention | v4, disable_flashinfer_prefill | v4, disable_flashinfer_prefill |
| **COMMON CONFIGURATION** | | |
| max-model-len | 131,072 | 131,072 |
| max-num-seqs | 16 | 16 |
| max-num-batched-tokens | 8,192 | 8,192 |
| KV cache dtype | fp8 | fp8 |
| GPU memory utilization | 0.85 | 0.85 |
| Speculative decoding | Eagle3 (3 tokens, 0.6 synthetic acceptance) | Eagle3 (3 tokens, 0.6 synthetic acceptance) |

Key design choices:
- **Prefill uses enforce-eager** instead of O3 compilation, because the input shape changes greatly in the prefill stage, the compilation benefits are limited and there is overhead
- **Prefill uses MultiConnector** (Nixl + MooncakeStore) to support multi-level KV caching; Decode only uses NixlConnector to receive KV
- **Eagle3 speculative decoding** is enabled on both ends, synthetic acceptance rate 0.6, 3 tokens per speculation

---

## 2. Benchmark configuration analysis

### 2.1 Toolchain

- **vigil**: Pipeline orchestrator, manages serving startup, router, benchmark execution, and indicator collection
- **vllm-bench**: A benchmark testing tool implemented in Rust, supporting multi-round dialogue, concurrent scanning, and prefix sharing simulation
- **vmon**: Background indicator collector (2s interval), collects GPU/vLLM indicators from two nodes
- **Slurm**: Remote job scheduling (job_id: 2956)

### 2.2 Request generation parameters

| Parameter | Value | Description |
|:-----|:---|:-----|
| Backend | openai-chat | OpenAI Chat Completions API |
| Dataset | random | Randomly generate token |
| Input length | 70,000 tokens/turn | Fixed per round |
| Output length | 300 tokens/turn | Fixed per round |
| Multi-turn | 10 turns/conversation | |
| Inter-turn delay | 250 ms | Simulate user thinking time |
| History accumulation | No | Send a complete prompt independently in each round and do not accumulate history |
| Prefix: global | 15% = 10,500 tokens | Shared across all conversations |
| Prefix: conversation | 75% = 52,500 tokens | Shared within the same conversation |
| Prefix: unique | 10% = 7,000 tokens | Unique per request |
| Concurrency sweep | 1, 2, 4, 8 | |
| sweep-num-prompts-factor | 2 | conversations = 2 × concurrency |### 2.2.1 Workload mode definition

The current Kimi 70K analysis mainly covers **Simple test mode**, but in this project we have agreed on three types of workloads, which need to be uniformly recorded and used comparatively.

#### A. Simple test mode

The goal is to evenly divide complex traffic into a repeatable and comparable average case to facilitate system baseline analysis and parameter scanning.

| Dimensions | Definition |
|:-----|:-----|
| Input length | Fixed `70,000` tokens |
| Output length | Fixed `200` tokens; this Kimi 70K recipe actually uses `300` tokens |
| Global shared prefix | First `12K` tokens shared across all conversations |
| Intra-conversation prefix cache hit | Target is approximately `90%` intra-conversation prefix cache hit |
| Dataset | Generally use `random` |
| Applicable scenarios | Baseline performance, prefix cache revenue evaluation, PD parameter scan |

The core features of this type of model are: fixed length, fixed reuse ratio, and stable results, making it suitable for comparing different configurations and implementations. **

#### B. Loadtest mode

The goal is to get as close to the real business traffic distribution as possible, rather than just running a single average case.

| Dimensions | Definition |
|:-----|:-----|
| Input length distribution (ISL) | `P50 70K`, `P90 90K`, `P99 120K` |
| Output length distribution (OS) | `P50 200`, `P90 300`, `P99 1000` |
| Intra-session cache hits | Gradually increased from about `60%` to `95%` |
| Cross-session cache hits | Typically only the first global/shared portion is close to `100%` |
| Dataset | `synthetic_sharegpt_v3.json` |
| Applicable scenarios | More realistic throughput-delay trade-off, capacity planning, tail delay analysis |

The core feature of this type of mode is that both the ** length and cache hits obey the real distribution, which is more suitable for verifying the behavior of the system under production-style load. **

#### C. Shadow mode

The goal is to do a 1:1 replay based on real evaluation traffic, rather than using synthetic distribution sampling.

| Dimensions | Definition |
|:-----|:-----|
| Traffic source | Real `swebenchpro` eval traffic dump |
| Dataset | `codex_swebenchpro.json` |
| input/output length | follow original trace |
| prefix pattern | follows the context and session structure in the original trace |
| Applicable scenarios | Real replay, problem reproduction, blog/result review |

The core characteristics of this type of mode are: ** closest to the real workload, but the repeatability and parameter controllability are weaker than simple/loadtest. **

#### D. The relationship between the three types of workload and this article

- The main data analyzed in this article comes from `random + fixed 70K/300 + prefix sharing`, which belongs to **Simple test mode**.
- If the same PD recipe is subsequently expanded to `sharegpt + synthetic_sharegpt_v3.json`, it will belong to **Loadtest mode**.
- If `codex_swebenchpro.json` is played back directly, it should be marked as **Shadow mode** separately and not mixed with simple/loadtest statistics.

### 2.3 Load calculation

| Concurrency | Number of conversations | Total requests | Total Input Tokens | Total Output Tokens | Total Tokens |
|:------:|:------:|:--------:|:---------------:|:----------------:|:----------:|
| 1 | 2 | 20 | 1,400,000 | 6,000 | 1,406,000 |
| 2 | 4 | 40 | 2,800,002 | 12,000 | 2,812,002 |
| 4 | 8 | 80 | 5,600,000 | 24,000 | 5,624,000 |
| 8 | 16 | 160 | 11,200,000 | 48,000 | 11,248,000 |Each conversation: 10 turns × 70,000 = 700,000 input tokens + 10 turns × 300 = 3,000 output tokens.

The entire benchmark took a total of 515.1 seconds (~8.6 minutes), including sequential execution of all 4 concurrency levels. The pipeline took a total of 2,413.8 seconds (~40 minutes, including serving startup).

---

## 3. Input/output mode analysis

### 3.1 Token distribution characteristics

This payload has **extreme prefill-dominated characteristics**:

- **Input:Output ratio = 233:1** (70,000 : 300)
- Input tokens dominate the throughput (for example, C=1: input 21,180 tok/s vs output 91 tok/s)
- In each round of requests, the amount of prefill calculation is much greater than the amount of decode calculation.

This means:
- **Prefill node is the critical path of the system**, and its computing power directly determines the upper limit of system throughput
- The Decode node is in a waiting state most of the time and has low utilization rate
- Optimizing prefill efficiency (such as prefix caching, chunked prefill parameter tuning) is the highest lever to improve overall performance

### 3.2 Prefix sharing analysis

The composition of 70,000 tokens per round:
```
├── Global Prefix ─── 10,500 tokens (15%) ─── Shared by all conversations, global cache
├── Conversation Prefix ─ 52,500 tokens (75%) ─── 10 rounds of sharing within the same conversation
└── Unique Tokens ─── 7,000 tokens (10%) ─── Each request is unique and must be calculated
```
**Turn 1 (cold start)**:
- First occurrence of global prefix → requires 10,500 tokens for full calculation
- First occurrence of dialogue prefix → requires 52,500 tokens for full calculation
- Only part → 7,000 tokens need to be calculated
- **Total: 70,000 tokens all need to be calculated**

**Turn 2-10 (hot cache)**:
- global prefix → **cache hit** (cached by Turn 1 or other dialogue)
- Conversation Prefix → **Cache Hit** (cached by this conversation Turn 1)
- Only part → 7,000 tokens need to be calculated
- **Only 7,000 tokens (10%) are calculated, and the theoretical total reusability ratio of a single hot request is 90%**

This design directly explains the dramatic drop in TTFT between Turn 1 and Turn 2 (see Section 5 for details).

### 3.2.1 Why the local Prefix Cache Hit in the project log may only be 15%

The `90%` above refers to the proportion of tokens that can be reused in a single preheated request from the request semantics, which is not equal to the **local GPU prefix cache hit rate** in the project log. These two calibers need to be looked at separately:

- This workload has prefix-sharing mode enabled and **No history accumulation**. In other words, each round only sends the fixed-length user message of the current round, and the `300` output tokens generated in the previous round will not be spliced ​​into the next round of input.
- Therefore, the input for round `t` of any session `i` can always be written as: `G(10.5K) + C_i(52.5K) + U_{i,t}(7K)`.
- For a single warmed-up Turn 2-10 request, indeed only `7K` unique suffix needs to be newly calculated, so the **total reusable ratio** of this request is `90%`.
- But when averaging over the entire 10-round conversation, the first cold start of each session is also included: `1` cold round + `9` hot round, corresponding to an average total reusable ratio of approximately `(10 x 10.5K + 9 x 52.5K) / (10 x 70K) = 82.5%`.
- More importantly, project logs usually split reuse into two categories: `Prefix cache hit rate` hit by local GPU, and `External prefix cache hit rate` retrieved through Mooncake / NIXL / KV connector. The denominators of these two indicators are different and cannot be added directly.
- In the `C=8` set of workloads, the benchmark actually generated `16` sessions instead of `8`. This results in a session-level prefix working set size of `16 x 52.5K + 10.5K = 850.5K tokens`.
- Combined with the KV capacity modeling of [hbm_storage_modeling.md](../memory_analysis/hbm_storage_modeling.md), the local GPU KV capacity on the prefill side is usually about `460,000~590,000 tokens`, which is not enough to accommodate all `16` session prefixes in the long term.
- The result is: the hottest `10.5K global prefix` shared by all sessions is more likely to reside locally for a long time, so the local `Prefix cache hit rate` is often close to `15%`; while the `52.5K conversation prefix`, although also reused, is more likely to be hit through the external KV cache, or be squeezed out of the local and reloaded during session rotation.

In other words, `about 15% local` is not inconsistent with `single hot request theory 90% reusable`. The former describes the proportion of reuse that occurs on the local GPU, and the latter describes how many tokens can be retrieved from the request content without recalculation.

### 3.3 Token composition visual description

It is recommended to draw a stacked column chart:

- **X axis**: Turn index (1-10)
- **Y axis**: Token quantity (0-70,000)
- **Color**: Green = Global Prefix, Blue = Conversation Prefix, Red = Unique
- **Mark**: Turn 1 is all "requires calculation"; Turn 2+ is marked in green + blue as "cache hit"---

## 4. Concurrency and arrival pattern analysis

### 4.1 Overview of concurrency levels

| Concurrency | Number of conversations | Total requests | Duration (s) | Total Input Tokens | Total Output Tokens | Success rate |
|:------:|:------:|:--------:|:----------:|:---------------:|:----------------:|:------:|
| 1 | 2 | 20 | 66.1 | 1,400,000 | 6,000 | 100% |
| 2 | 4 | 40 | 71.7 | 2,800,002 | 12,000 | 100% |
| 4 | 8 | 80 | 145.6 | 5,600,000 | 24,000 | 100% |
| 8 | 16 | 160 | 226.5 | 11,200,000 | 48,000 | 100% |

0 failures at all concurrency levels indicate that the system has good stability within this load range.

### 4.2 Request arrival mode

This benchmark uses **closed-loop** mode instead of open-loop:

- The next round of requests for each conversation is delayed by 250ms after the previous round is completed.
- Concurrency controls the number of conversations running at the same time, not the number of requests in progress at the same time
- This means that the actual request arrival rate is subject to system response latency

**Meaning of closed loop mode**:
- When the system slows down, the request arrival rate automatically decreases (adaptive backpressure)
- Actual request throughput is a reflection of system responsiveness, not externally imposed load
- This is closer to the scenario of concurrent access by multiple users in a production environment

### 4.3 Request overlap and competition with Prefill

Under different concurrency levels, the overlap pattern of requests on the timeline is different:

**C=1 (2 conversations)**:
- Up to 2 conversations are ongoing at the same time, but due to the 250ms delay, the requests are basically serial
- There is almost no competition for Prefill nodes

**C=2 (4 conversations)**:
- Up to 2 requests may arrive at the prefill node at the same time
- But due to E2EL ~3.5s >> 250ms delay, the actual overlap probability is medium

**C=4 (8 dialogues)**:
- During Turn 1, 8 conversations issued the first round of requests at the same time, causing serious congestion in prefill.
- In subsequent rounds, 4 dialogues may be waiting for prefill at the same time
- max-num-seqs=16 has not become a bottleneck, but max-num-batched-tokens=8192 limits the efficiency of parallel prefill

**C=8 (16 conversations)**:
- During Turn 1, 16 conversations issued requests at the same time, which far exceeded the processing capacity of the prefill node.
- In subsequent rounds, 8 dialogues may compete for prefill resources at the same time
- max-num-seqs=16 is close to saturation and the queuing delay is significant
- E2EL expands to 11.2s, forming a serious queuing effect

---

## 5. Round-by-round timing analysis

### 5.1 TTFT changes from round to round

TTFT (Time to First Token) is the core indicator for measuring prefill performance. The following matrix shows the average TTFT (ms) per round at various concurrency levels:

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
| **Mean** | **1,271** | **1,193** | **2,312** | **4,134** |**Key Observations**:

1. **Turn 1 cold start penalty**: The TTFT of Turn 1 is significantly higher than that of subsequent rounds under all concurrency levels
   - C=1: 5,143ms → ~840ms (Turn 2+), a decrease of **83%**
   - C=2: 3,112ms → ~975ms, a decrease of **69%**
   - This directly verifies the effect of high proportion of prefix reuse; for hot requests, only 7K unique tokens need to be newly calculated

2. **C=1/C=2 steady state is very stable**: the TTFT standard deviation of Turn 2-10 is extremely small
   - C=1: 828-852ms, fluctuation < 3%
   - C=2: 948-1,026ms, fluctuation < 8%

3. **C=4/C=8 Steady-state TTFT increases significantly and fluctuates greatly**:
   - C=4: 1,490-2,765ms, fluctuation range ~1,300ms
   - C=8: 3,039-6,170ms, fluctuation range ~3,100ms
   - This reflects non-deterministic delays caused by queuing up prefill nodes

4. **C=2 Turn 1 (3,112ms) is lower than C=1 Turn 1 (5,143ms)**: It may be because C=2 has 4 conversations, the global prefix is warmed up by earlier conversations, and some conversations benefit from the existing cache

### 5.2 TPOT changes from round to round

TPOT (Time per Output Token, excluding the first token) reflects decoding performance:

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
| **Mean** | **6.80** | **7.95** | **16.16** | **23.57** |

**Observation**:
- TPOT is relatively stable within the same concurrency (excluding Turn 1)
- C=1→C=2 increases by 17%, C=2→C=4 increases by 103%, C=4→C=8 increases by 46%
- C=4 shows a step-like growth, indicating that the batch size of the decode node increases significantly when C=4

### 5.3 ITL P99 tail delay analysis

P99 of ITL (Inter-token Latency) is a key indicator to measure the consistency of user experience:

| Concurrency | Mean ITL (ms) | Median ITL (ms) | P99 ITL (ms) | P99/Median |
|:------:|:----------:|:----------:|:----------:|:----------:|
| 1 | 10.98 | 10.56 | 14.45 | 1.4x |
| 2 | 12.85 | 12.15 | 25.75 | 2.1x |
| 4 | 16.86 | 14.19 | 123.88 | **8.7x** |
| 8 | 23.54 | 15.95 | 379.45 | **23.8x** |

**Analysis**:
- The P99/Median ratio of C=1/C=2 is 1.4-2.1x, the distribution is compact, and the user experience is consistent
- **C=4 P99 ITL jumps to 124ms**, P99/Median reaches 8.7x, and there is obvious tail delay
- **C=8 P99 ITL exploded to 379ms**, P99/Median reached 23.8x, indicating that a few token generation encountered serious delaysPossible reasons:
- The verification failure rate of Eagle3 speculative decoding increases under high concurrency, causing tokens to be regenerated.
- KV cache transfer competes with decode calculation resources
- Scheduler's scheduling delay when switching batches
- Prefill requests under high concurrency may interrupt the ongoing decode batch (chunked prefill interference)

### 5.4 E2EL decomposition

E2EL (End-to-End Latency) can be approximately decomposed into: E2EL ≈ TTFT + output_tokens × TPOT

| Concurrency | Mean TTFT | 300 × Mean TPOT | Estimated E2EL | Actual Mean E2EL | Error |
|:----------:|:----------:|:----------:|:----------:|:-------------:|:----:|
| 1 | 1,271 | 2,040 | 3,311 | 3,305 | 0.2% |
| 2 | 1,193 | 2,384 | 3,577 | 3,569 | 0.2% |
| 4 | 2,312 | 4,847 | 7,159 | 7,143 | 0.2% |
| 8 | 4,134 | 7,071 | 11,205 | 11,181 | 0.2% |

The decomposition model is highly accurate (error < 1%).

**TTFT proportion in E2EL**:
- C=1: 38% (TTFT) + 62% (Decode)
- C=2: 33% (TTFT) + 67% (Decode)
- C=4: 32% (TTFT) + 68% (Decode)
- C=8: 37% (TTFT) + 63% (Decode)

The contribution of TTFT and decode time to E2EL is relatively stable, about 1:2. But note that in terms of user experience, TTFT corresponds to the time of "waiting for response to start", which has a greater impact on user experience.

---

## 6. Performance expansion analysis

### 6.1 Throughput Scaling

| Concurrency | Total Throughput (tok/s) | Input Throughput (tok/s) | Output Throughput (tok/s) | Request Throughput (req/s) | Throughput expansion ratio |
|:------:|:------------------------:|:------------------------:|:--------------------------:|:--------------------------:|:----------:|
| 1 | 21,271 | 21,180 | 91 | 0.30 | 1.00x |
| 2 | 39,242 | 39,074 | 167 | 0.56 | 1.84x |
| 4 | 38,623 | 38,458 | 165 | 0.55 | 1.82x |
| 8 | 49,660 | 49,448 | 212 | 0.71 | 2.33x |

**Key Findings**:

1. **C=1→C=2 near linear expansion (1.84x)**: The system has a lot of free capacity when C=1, doubling the concurrency almost doubles the throughput
2. **C=2→C=4 throughput is flat or even slightly decreased (39,242 → 38,623)**: This is a clear signal of **prefill saturation**
   - The request volume doubled, but the throughput decreased instead of increasing.
   - Request throughput reduced from 0.56 to 0.55 req/s
   - Additional concurrency only increases queuing time
3. **C=4→C=8 throughput increases to 49,660**: But this is mainly because the number of simultaneous requests in transit is doubled, and the throughput improvement (1.28x) is much lower than the concurrency improvement (2x)
   - is actually the appearance of "more requests waiting in a longer queue"

### 6.2 Delayed expansion| Concurrency | Mean TTFT (ms) | Mean TPOT (ms) | Mean ITL (ms) | Mean E2EL (ms) | P99 ITL (ms) |
|:------:|:---------------:|:---------------:|:-------------:|:---------------:|:----------:|
| 1 | 1,271 | 6.80 | 10.98 | 3,305 | 14.45 |
| 2 | 1,193 | 7.95 | 12.85 | 3,569 | 25.75 |
| 4 | 2,312 | 16.16 | 16.86 | 7,143 | 123.88 |
| 8 | 4,134 | 23.57 | 23.54 | 11,181 | 379.45 |

**Delayed Expansion Features**:
- **C=1→C=2**: TTFT is almost unchanged (1,271→1,193ms) and E2EL only increases by 8%. This is the "sweet spot" configuration
- **C=2→C=4 (inflection point)**: TTFT doubled (1,193→2,312ms), E2EL doubled (3,569→7,143ms), but throughput did not increase
- **C=4→C=8**: TTFT nearly doubled again (2,312→4,134ms), P99 ITL grew 3x (124→379ms)

### 6.3 Throughput-Latency Tradeoff

It is recommended to draw a two-axis chart:
- **X-axis**: Concurrency (1, 2, 4, 8)
- **Left Y-axis**: Total Throughput (tok/s) — Bar chart
- **Right Y-Axis**: Mean TTFT (ms) — Line Chart

Data points:

| C | Throughput | TTFT |
|:-:|:----------:|:----:|
| 1 | 21,271 | 1,271 |
| 2 | 39,242 | 1,193 |
| 4 | 38,623 | 2,312 |
| 8 | 49,660 | 4,134 |

The chart will clearly show: **C=2 is the optimal operating point** — throughput is close to the system limit, but latency has not worsened significantly. Above C=4, latency continues to expand but throughput increases slightly.

### 6.4 Per-GPU efficiency

| Concurrency | Total Throughput | Per-GPU efficiency (tok/s/GPU) | Efficiency improvement (vs C=1) |
|:------:|:----------------:|:-----------------------:|:------------------:|
| 1 | 21,271 | 2,659 | 1.00x |
| 2 | 39,242 | 4,905 | 1.84x |
| 4 | 38,623 | 4,828 | 1.82x |
| 8 | 49,660 | 6,208 | 2.33x |

From C=2 to C=8, efficiency only increases by 27% (4,905→6,208 tok/s/GPU), but average latency increases by 213% (3,569→11,181ms). In SLA-sensitive scenarios, this is not a favorable trade-off.

---

## 7. Key findings and bottleneck identification

### 7.1 Prefill is the main bottleneck

**Evidence**:
- 70K tokens/turn requires a lot of calculations under TP4; max-num-batched-tokens=8192 means that each 70K request requires ~9 chunks of chunked prefill
- Throughput saturates at C=2 (39K→38K tok/s @ C=2→C=4), while latency doubles
- Request throughput does not rise but falls at C=2→C=4 (0.56→0.55 req/s)
- Under all concurrency levels, the TTFT of Turn 1 (requires full prefill of 70K tokens) is much higher than that of subsequent rounds

**Impact**: The processing capability of the Prefill node directly determines the effective concurrency upper limit of the system### 7.2 Prefix Caching has a significant effect

**Evidence**:
- C=1 Turn 1 TTFT 5,143ms → Turn 2 TTFT 848ms, a decrease of **83%** (6.1x acceleration)
- C=2 Turn 1 TTFT 3,112ms → Turn 2 TTFT 973ms, a decrease of **69%** (3.2x acceleration)
- Turn 2-10 TTFT is extremely stable (fluctuation < 3%) under C=1/C=2, indicating that the cache hit path performance is consistent

**Validation**: The high proportion of prefix reuse design (15% global + 75% conversation) was verified in actual testing. For a single preheated request, the calculation of the complete prefill from 70K tokens to only 7K unique tokens is the root cause of the significant drop in TTFT; however, the local GPU prefix cache hit rate observed in the project may be significantly lower than 90%, because a large part of the reuse will be implemented through the external KV cache.

### 7.3 Decode tail latency is seriously worsened under high concurrency

**Evidence**:
- P99 ITL: C=1 14ms → C=8 379ms, deterioration **26x**
- P99/Median ratio: C=1 1.4x → C=8 23.8x, the distribution is extremely right-skewed
- P99 ITL per round at C=8 for more than 350ms (Turn 2-9)

**Possible reasons**:
- Penalty caused by Eagle3 speculative decoding failure to verify under high load
- Multiple decode requests compete for GPU computing resources in the same batch
- Resource interference between KV transfer (from prefill to decode via Nixl) and decode calculation
- Chunked prefill may cause interference on the decode node (the decode node also has chunked-prefill enabled)

### 7.4 KV Transfer overhead is the main component of steady-state TTFT

**Evidence**:
- C=1 Turn 2-10 Steady-state TTFT ~840ms, but only 7,000 unique tokens are actually calculated
- The pure prefill calculation time of 7K tokens on TP4 B200 is estimated to be about 100-200ms.
- The remaining ~640ms mainly comes from:
  - Prefix cache lookup and KV loading (from CPU DDR or NVMe to GPU HBM)
  - KV transfer: prefill to decode via Nixl (cross-node via RoCE)
  - Scheduling overhead (chunked prefill shard scheduling)

**Significance**: In the steady state with prefix caching in effect, KV data handling rather than computation becomes the main component of TTFT. This points to KV transfer optimization as the next step for performance improvement.

### 7.5 max-num-seqs=16 may limit high concurrency performance

**Evidence**:
- Both Prefill and Decode set max-num-seqs=16
- When C=8, 16 conversations are running at the same time, and when Turn 1, 16 requests arrive at the same time.
- Although requests are distributed through the router (round_robin), the prefill node can handle up to 16 sequences at the same time
- Turn 1 TTFT of C=8 is as high as 7,333ms, partly due to scheduler queuing

---

## 8. Optimization suggestions

### 8.1 Short-term optimization (no changes to hardware topology)

| Optimization item | Current value | Recommended value | Expected effect |
|:------|:------|:------|:--------|
| Number of Prefill copies | 1P1D | 2P1D or 2P2D | Double prefill throughput and reduce queuing delay |
| max-num-batched-tokens | 8,192 | 16,384 or 32,768 | Reduce the number of chunked prefill shards (9→5 or 3) and reduce scheduling overhead |
| max-num-seqs | 16 | 32 or 64 | Reduce high concurrency queuing |
| Eagle3 num_speculative_tokens | 3 | 2 (when concurrency is high) | Reduce P99 ITL tail latency, sacrificing a small amount of average TPOT |### 8.2 Mid-term optimization (architectural adjustment)

1. **Prefill TP8**: Expand prefill from TP4 (1 node) to TP8 (2 nodes), and reduce the single request prefill delay by about 50%
2. **Prefill compilation optimization**: Evaluate the benefits of O3 compilation on the prefill side (currently using enforce-eager)
3. **Load-aware routing**: Switch from round_robin to load-aware routing, especially in multi-prefill node scenarios
4. **KV Transfer Pipeline**: Calculate the overlap between KV transfer and decode to reduce transfer waiting in TTFT

### 8.3 Long-term optimization (system level)

1. **DP+TP hybrid parallelism**: Use DP2×TP2 instead of TP4 as the prefill configuration to obtain 2x request parallelism
2. **Massive Scaling Test**: Currently using only 2/18 nodes (8/72 GPU). Need to verify scalability at larger scale
3. **Real Production Trace Replay**: 70K random tokens is an extreme scenario. The distribution of token lengths, number of dialogue rounds, and concurrency patterns of real workloads may be significantly different. It is recommended to use production access logs for replay testing
4. **Mooncake multi-level cache tuning**: Evaluate the hit rate of L1 (GPU HBM) → L2 (CPU DDR via NVLink-C2C) → L3 (NVMe RAID0) three-level cache under different working set sizes

---

## 9. Appendix

### 9.1 Complete round-by-round indicators

#### 9.1.1 Concurrency = 1 (2 conversations, 20 requests)

| Turn | TTFT Mean | TTFT Median | TTFT P99 | TPOT Mean | ITL Mean | ITL P99 | E2EL Mean | E2EL P99 |
|:----:|:---------:|:----------:|:--------:|:---------:|:--------:|:-------:|:----------:|:--------:|
| 1 | 5,143.4 | 5,143.4 | 7,181.9 | 8.40 | 14.11 | 12.88 | 7,654.3 | 10,332.8 |
| 2 | 847.7 | 847.7 | 860.2 | 6.15 | 10.53 | 14.12 | 2,685.8 | 2,698.1 |
| 3 | 850.2 | 850.2 | 856.2 | 6.62 | 11.18 | 17.37 | 2,828.6 | 2,837.5 |
| 4 | 839.2 | 839.2 | 846.7 | 7.22 | 10.51 | 13.25 | 2,998.3 | 3,191.3 |
| 5 | 828.1 | 828.1 | 842.6 | 6.49 | 10.51 | 13.02 | 2,767.1 | 2,784.9 |
| 6 | 833.4 | 833.4 | 842.2 | 6.36 | 10.50 | 11.53 | 2,734.4 | 2,798.3 |
| 7 | 836.1 | 836.1 | 846.9 | 6.61 | 10.54 | 12.78 | 2,812.3 | 2,837.5 |
| 8 | 842.4 | 842.4 | 855.5 | 6.62 | 10.50 | 11.92 | 2,822.1 | 2,909.6 |
| 9 | 832.4 | 832.4 | 847.6 | 6.50 | 10.73 | 15.47 | 2,775.1 | 2,881.2 |
| 10 | 852.0 | 852.0 | 862.5 | 7.08 | 10.89 | 17.06 | 2,970.4 | 3,141.4 |#### 9.1.2 Concurrency = 2 (4 conversations, 40 requests)

| Turn | TTFT Mean | TTFT Median | TTFT P99 | TPOT Mean | ITL Mean | ITL P99 | E2EL Mean | E2EL P99 |
|:----:|:---------:|:----------:|:--------:|:---------:|:--------:|:-------:|:----------:|:--------:|
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

#### 9.1.3 Concurrency = 4 (8 conversations, 80 requests)

| Turn | TTFT Mean | TTFT Median | TTFT P99 | TPOT Mean | ITL Mean | ITL P99 | E2EL Mean | E2EL P99 |
|:----:|:---------:|:----------:|:--------:|:---------:|:--------:|:-------:|:----------:|:--------:|
| 1 | 4,644.0 | 3,276.2 | 10,657.0 | 12.83 | 17.02 | 125.36 | 8,481.2 | 17,223.8 |
| 2 | 2,764.9 | 2,679.1 | 6,587.1 | 13.44 | 14.95 | 34.96 | 6,784.2 | 9,693.2 |
| 3 | 2,087.8 | 1,972.5 | 2,930.9 | 17.35 | 18.17 | 125.56 | 7,276.3 | 8,217.1 |
| 4 | 2,183.6 | 2,228.0 | 3,077.7 | 16.39 | 16.94 | 123.82 | 7,085.2 | 7,854.6 |
| 5 | 2,449.6 | 2,544.7 | 3,180.5 | 18.17 | 18.13 | 126.76 | 7,883.0 | 8,620.5 |
| 6 | 1,923.3 | 1,622.3 | 2,982.1 | 17.41 | 17.35 | 126.33 | 7,127.6 | 7,436.0 |
| 7 | 1,897.8 | 1,945.7 | 2,621.1 | 17.85 | 17.85 | 126.28 | 7,236.1 | 7,727.4 |
| 8 | 1,632.7 | 1,620.5 | 2,132.7 | 16.54 | 16.58 | 122.02 | 6,577.0 | 7,160.8 |
| 9 | 2,051.3 | 1,995.1 | 2,659.1 | 16.27 | 16.25 | 101.42 | 6,917.0 | 7,427.2 |
| 10 | 1,489.7 | 1,280.6 | 2,572.6 | 15.31 | 15.29 | 36.07 | 6,066.5 | 6,808.1 |#### 9.1.4 Concurrency = 8 (16 conversations, 160 requests)

| Turn | TTFT Mean | TTFT Median | TTFT P99 | TPOT Mean | ITL Mean | ITL P99 | E2EL Mean | E2EL P99 |
|:----:|:---------:|:----------:|:--------:|:---------:|:--------:|:-------:|:----------:|:--------:|
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

### 9.2 Configuration file reference

- Benchmark configuration: `vigil/examples/pd_kimi_bench_a_70k.yaml`
- Model configuration: `/mnt/lustre/hf-models/hub/models--nvidia--Kimi-K2.5-NVFP4/snapshots/c0285e649c34d4386b01e38abca642c06cbe014e/config.json`
- Quantification configuration: same directory `hf_quant_config.json`
- Machine documentation: `vllm/sprint_dist_kv_doc/machine_doc.md`

### 9.3 Data source

All performance data comes from the following JSON file:
- `openai-chat-infqps-concurrency1-Kimi-K2.5-NVFP4-20260411-015541.json`
- `openai-chat-infqps-concurrency2-Kimi-K2.5-NVFP4-20260411-015541.json`
- `openai-chat-infqps-concurrency4-Kimi-K2.5-NVFP4-20260411-015541.json`
- `openai-chat-infqps-concurrency8-Kimi-K2.5-NVFP4-20260411-015541.json`
- `results.json` (pipeline summary)vmon indicator: `vmon/vmon_vllm-bench.json` (258 samples, 2s interval, ~514s covering all benchmarks)

### 9.4 Glossary

| Abbreviation | Full name | Description |
|:-----|:-----|:-----|
| TTFT | Time to First Token | The time from when the request is sent to when the first token is received |
| TPOT | Time per Output Token | Average generation time of each output token (excluding the first token) |
| ITL | Inter-token Latency | The time interval between two adjacent tokens |
| E2EL | End-to-End Latency | The total time from the request to the last token generation |
| PD | Prefill-Decode | Prefill-Decode separation architecture |
| TP | Tensor Parallelism | Tensor Parallelism |
| MoE | Mixture of Experts | Mixed expert model |
| KV | Key-Value | Key-value cache in attention mechanism |
| NVFP4 | NVIDIA 4-bit Floating Point | NVIDIA 4-bit floating point quantization format |
| HBM3e | High Bandwidth Memory 3e | High bandwidth memory |
| NVLink-C2C | NVLink Chip-to-Chip | High-speed interconnection between chips |