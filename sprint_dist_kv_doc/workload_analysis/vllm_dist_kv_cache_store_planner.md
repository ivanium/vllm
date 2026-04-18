# [Internal] vLLM Distributed KV Cache Store Planner

- Source: https://docs.google.com/document/d/1xMBWD_KFnroOHiD2gHg0sk3L1DufTqiRGqegYduFngs/edit?pli=1&tab=t.b4tw2dvgiuxz
- Synced to repo: 2026-04-14
- Note: This is a local Markdown snapshot derived from the connected Google Doc for easier browsing inside the repo.

## Planning

### vLLM Distributed KV Cache Store Planning

#### Overall goal

The overall goal is to enable vLLM runs with a multi-tier and distributed KV cache store to avoid any prefix cache miss for multi-turn and agentic workloads. In other words, system prompts and ALL session contexts should get cache hits.

The sprint is planned tightly for a single month, with a focus on integrating Mooncake-store into vLLM and optimizing the performance. At the end of the sprint, we will deliver:

- open-source code
- representative agentic workloads
- and a blog post:
  - summarize and highlight the techniques and optimizations
  - our performance numbers
  - and instructions for people to try out the code and reproduce the results

#### Non-goals

- We mainly focus on vLLM-side optimizations rather than Mooncake internals unless they unlock key performance improvements.
- We will seek help from the Mooncake team for Mooncake-side optimizations.

#### Targeted hardware setup and performance goals

Hardware:

- Mainly optimize for GB200 with multi-node NVLink

Setup:

- Single node DP
- PD
- vLLM router + Multi-node DP

Performance goals:

- Metrics: throughput, latency, prefix cache hit rate
- SLO for distributed KV store:
  - Zero overhead when no prefix cache hit
  - Cache all useful content per space permits
  - Caches stored by one node should be visible ASAP to the other nodes
- E2E SLA target: Crusoe workload

#### Key models to support

- DSV3.2
- Kimi 2.5
- (Optional) Qwen3.5

### Key Milestones & Timeline

#### 0.5: Simulated agentic workload and benchmark tool

Zijing has already collected the agentic trace and set this up.

#### 1: Basic Mooncake store integration into vLLM

- Implement the Mooncake store connector
- Support RDMA and TCP and disk backend
- Validate with single-node DP
- Verify DP+EP and multiple vLLM DP instances can share KV cache content
- Start with Llama and Qwen3, then try Kimi-2.5 NVFP4

#### 2: Distributed KV store + PD

- Good integration with vLLM MultiConnector
- Decode nodes can get partial cache hits from the distributed store and fetch remaining KV from prefill nodes
- Support different TP and KV cache layouts for prefill/decode nodes

#### 3: Distributed KV store + vLLM router for multi-node DP

- Router queries Mooncake master for prefix cache location
- Needs a cache-aware load-balancing policy

#### 4: Performance optimization

- Profile and optimize:
  - control-path communication
  - datapath communication
  - router integration
  - cache eviction policies

## Dev Results

### Apr 9, 2026

Kimi-k2.5, DP2 TP4: benchmark results showed Mooncake improving throughput and mean latency, while some tail metrics remained worse or flat.

### Apr 8, 2026

Single-node DP2 and TP2 experiments:

- baseline vs Mooncake memory offload
- with and without cross-layer
- substantial TTFT and E2EL improvements in the reported runs

## Meeting Notes

### Apr 13, 2026 | Dist KV Store Standup

Key points:

- Eagle speculative decoding accept rate is low on random datasets
- NIXL uses UCX for PD data transfer
- There were segfaults during NIXL connector initialization, not specific to offloading
- Workload definition needs to be unified and calibrated
- Plan to benchmark larger PD and MNDP deployments

### Apr 9, 2026

Notes on CPU-GPU datapath:

- C2C bandwidth is high, but latency for small transfers may not be better than PCIe
- When protocol is RDMA, even local CPU-originating cache loads may still involve the NIC path
- More benchmarking is needed to quantify CPU-GPU communication latency

### Apr 8, 2026 | Dist Mooncake Store Sync

Current limits:

- Single node, multi-GPU: CPU memory offloading works, disk offloading does not
- Multi-node, 1 GPU per node: CPU and disk offloading work
- Multi-node, multi-GPU: CPU memory offloading works

Additional notes:

- Need runnable setups without crash
- Need better observability for PD bandwidth and transfer behavior
- Need benchmark settings large enough to trigger offloading and external cache hits

## Dev RunBook

### Setup

Dev branch:

- https://github.com/ivanium/vllm/tree/feat/mooncake-store-int

Basic setup summary:

- clone `vigil`
- clone `ivanium/vllm` as `vllm-mooncake`
- checkout `feat/mooncake-store-int`
- create `uv` Python 3.12 environment
- install editable vLLM
- install Mooncake transfer engine wheel
- install Kimi sprint dependencies:
  - `nixl[cu13]`
  - `flashinfer-cubin==0.6.7`
  - `flashinfer-jit-cache==0.6.7`
  - `fastsafetensors`
- clone `router-internal`
- clone `crusoe-inference`
- clone and install `vllm-bench`

### PD disaggregation setup

- Create `dist_kv/` recipes under `recipes/crusoe/kimik25/`
- Use `pd_dev.yaml` as a setup reference
- Add more detailed instructions for Mooncake env vars and master launch

### Multi-node DP setup

- Create `dist_kv/` recipes for DP under `recipes/crusoe/kimik25/`
- Use `tp4_eagle_fa4_offloading_c8.yaml` as a reference
- Add more detailed instructions for Mooncake env vars and master launch
