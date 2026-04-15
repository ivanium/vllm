# Complete Analysis of KV Cache Data Paths on NVL72 GB200

## 1. Overview

On NVL72 GB200, in the vLLM + Mooncake (L2 CPU Memory + L3 NVMe SSD) scenario, the KV Cache block required by the GPU may come from **6 locations**. This article dismantles from top to bottom: the physical datapath of each path, the replica selection logic inside Mooncake, the RDMA batch calling mechanism, and the randomness of data placement.

---

## 2. Overview of data sources
```
The requesting GPU requires a KV block, which may come from:

① Local GPU HBM is reused directly without going through Mooncake and vLLM prefix caching.
② Local CPU Memory Mooncake MEMORY replica, RDMA loopback
③ Local NVMe SSD Mooncake LOCAL_DISK replica, io_uring + RDMA loopback
④ Remote CPU Memory Mooncake remote MEMORY replica, RDMA or NVLink transfer
⑤ Remote NVMe SSD remote FileStorage RPC + RDMA
⑥ Remote GPU HBM P2P connector (NixlConnector / MooncakeConnector), without going through Store
```
| # | Source | Connector | Transport | Typical Bandwidth |
|---|------|-----------|-----------|----------|
| ① | Local GPU HBM | vLLM self-managed | — | 8 TB/s |
| ② | Local CPU Memory | MooncakeStoreConnector | RDMA loopback | RNIC dependency |
| ③ | Local SSD | MooncakeStoreConnector | io_uring + RDMA | ~25 GB/s (SSD bottleneck) |
| ④ | Remote CPU Memory | MooncakeStoreConnector | RDMA (RoCE) / NVLink Transit | 50 / 450 GB/s |
| ⑤ | Remote SSD | MooncakeStoreConnector | FileStorage RPC + RDMA | SSD+Network |
| ⑥ | Remote GPU HBM | NixlConnector / MooncakeConnector | UCX cuda_ipc / RDMA GPUDirect | NVLink ~900 GB/s |

---

## 3. Two types of Mooncake Connector

| | MooncakeConnector | MooncakeStoreConnector |
|---|---|---|
| Role | Cross-instance P→D KV direct transfer (replaces NixlConnector) | KV Cache hierarchical cache (CPU/SSD offload + cross-node reuse) |
| Bottom layer | TransferEngine direct P2P | MooncakeDistributedStore (Master coordination) |
| Data flow | Prefill GPU → RDMA → Decode GPU | GPU ↔ CPU Memory Pool ↔ SSD |
| Caching | No caching | Support prefix caching (hash deduplication) |

Production configuration uses MultiConnector combination: all sub-connectors are written when saving, and tried in order when loading.

---

## 4. Registration topology: who is in the Master’s allocation pool

Each vLLM GPU worker process is initialized independently and registers two types of memory with the Master:
```
GPU VRAM:  register_buffer() → RegisterLocalMemory(remote_accessible=false)
→ Only register to Transfer Engine (make RDMA accessible)
→ Do not report to Master, do not enter allocator pool

CPU Mem:   setup() → allocate_buffer_*() → MountSegment()
→ Report to Master, addAllocator() enters the allocation pool
→ segment.name = local_hostname_ (workers on the same node share the same name)
```
**Master is only allocated from CPU global_segment. GPU VRAM is not in the allocation pool. **GPU VRAM only serves as the source (put)/destination (get) endpoint for RDMA transfers.

Complete topology of 2-node TP4:
```
Master Server
  │
  │  AllocatorManager.names_ = ["10.x.x.11", "10.x.x.14"]
  │  allocators_ = {
│ "10.x.x.11": [CPU_seg_5GB × 4], ← Node A's 4 workers are 5GB each
│ "10.x.x.14": [CPU_seg_5GB × 4], ← Node B's 4 workers are 5GB each
  │  }
  │
  ├─ Node A (10.x.x.11), Prefill, TP=4
│ Worker 0~3: Each GPU VRAM (Transfer Engine only) + CPU 5GB (Master pool)
  │
  └─ Node B (10.x.x.14), Decode, TP=4
Worker 0~3: Each GPU VRAM (Transfer Engine only) + CPU 5GB (Master pool)
```
---

## 5. Data placement: where to write when putting

### 5.1 Allocation strategy: random, not aware of locality

When vLLM calls `batch_put_from_multi_buffers(keys, addrs, sizes)`, `ReplicateConfig` is not passed and the default values are used: `replica_num=1`, `preferred_segments=[]`.

Master's `RandomAllocationStrategy::Allocate()` (allocation_strategy.h:271-297):
1. Randomly select a starting point from `names_` (`std::uniform_int_distribution`)
2. Scan sequentially until a segment with space is found.
3. **Not aware of which node the caller is on** — Master only knows which segments there are, but does not know where the put comes from.

**Conclusion: Data may be written to the CPU memory of any node. ** Prefill generates KV in Node A → Master may allocate replica to Node B’s CPU segment → RDMA writes to Node B across nodes.

### 5.2 Example
```
Prefill (Node A) writes 100 KV blocks:
→ Master randomly allocates each key: ~50 in Node A CPU, ~50 in Node B CPU
→ 50 RDMA loopbacks write locally, and 50 RDMA loopbacks write remotely to Node B
```
### 5.3 Physical data flow
```
Put: GPU VRAM (source) ──RDMA Write──→ CPU global_segment (destination, Master allocation)
├─ May be local (loopback)
└─ Possibly at the remote end (cross-node RDMA)
```
---

## 6. Data reading: where to get it from when getting

### 6.1 Replica select priority

Master returns the replica list in insertion order: **MEMORY first, DISK/LOCAL_DISK last**.

vLLM's `batch_get_into_multi_buffers` only passes 3 parameters, `prefer_alloc_in_same_node` defaults to `false` (store_py.cpp:2226), use `FindFirstCompleteReplica` - **Get the first replica in the COMPLETE state in the list, without checking locality**.
```
Actual priority:
1. The MEMORY replica of the first COMPLETE (does not differentiate between local/remote, depends on the order of the Master list)
2. DISK replica (when all MEMORY is unavailable)
3. LOCAL_DISK replica (only when there is only LOCAL_DISK left, take the FileStorage RPC dedicated path)
```
**There is no priority between local CPU and remote CPU. ** Which one you get depends on which node is randomly assigned during PutStart.

### 6.2 Will it be obtained from the CPU Memory of other nodes?

**meeting. ** Because neither put nor get has locality preference:
```
Scenario: Decode is in Node B and requires a KV block

If this key is randomly assigned to the CPU segment of Node A during PutStart:
→ Master returns replica: MEMORY on Node A
→ Node B batch_get → FindFirstCompleteReplica → Select the MEMORY of Node A
→ RDMA remote read: Node A CPU → RoCE → Node B GPU

If this key is assigned to the CPU segment of Node B during PutStart:
→ Master returns replica: MEMORY on Node B
→ Node B batch_get → Select the MEMORY of Node B
  → RDMA loopback: Node B CPU → Node B GPU

Which happens depends entirely on PutStart's random assignment.
```
### 6.3 Coexistence of MEMORY and DISK replica
```
Scenario 1: CPU is not full → MEMORY(COMPLETE) first, DISK(COMPLETE) second → select MEMORY
Scenario 2: CPU is full, MEMORY is evicted → only LOCAL_DISK(COMPLETE) remains → take the SSD path
Scenario 3: Multiple nodes, Node A MEMORY is evicted but Node B MEMORY is still there
→ [MEMORY_B(COMPLETE), LOCAL_DISK_A(COMPLETE)] → Select remote MEMORY_B
→ Remote CPU memory takes priority over local SSD
```
---

## 7. RDMA batch calling mechanism

**Not one RDMA per key, nor one RDMA per remote node. ** It is Slice level batching + watermark flow control.

### 7.1 Call chain
```
vLLM batch_get_into_multi_buffers(100 keys)
  │
├─ LOCAL_DISK keys → FileStorage RPC private path (serial two-phase)
  │
  └─ MEMORY keys → Client::BatchGet()
│ Submit async transfer independently for each key (not waiting for completion)
       ▼
TransferSubmitter → Split each key into 64KB Slices
       ▼
     RdmaTransport::submitTransferTask()
│ Group by RdmaContext: slices_to_post[context].push_back(slice)
│ Accumulated to watermark (~512 slice) → One ibv_post_send() batch submission
│ Slices from different remote nodes are mixed in the same post_send
       ▼
RNIC DMA engine executes in parallel according to QP
```
### 7.2 Specific examples
```
100 MEMORY keys: 40 in Node A CPU, 30 in Node B CPU, 30 in local CPU
→ 256KB per key → 4 64KB slices per key → 400 slices in total
→ 400 < watermark(512) → one ibv_post_send()
→ RNIC internally processes slices to 3 targets in parallel
```
| Parameters | Default value | Meaning |
|------|--------|------|
| Slice size | 64 KB (`MC_SLICE_SIZE`) | Per RDMA request granularity |
| Watermark | ~512 (`max_wr × num_qp_per_ep`) | How many slices to accumulate before batch submission |

---

## 8. Physical Datapath of each data path

### Path ②: Local CPU Memory → Local GPU
```
GPU→CPU offload: GPU VRAM → IBV_WR_RDMA_WRITE → RNIC loopback → CPU global_segment
CPU→GPU load:    CPU global_segment → IBV_WR_RDMA_READ → RNIC loopback → GPU VRAM

Bandwidth: RNIC loopback capability (non-NVLink-C2C — Mooncake default protocol="rdma")
Latency: sub-ms (RDMA bucket 125-1000 μs)
Overhead: Zero CPU/GPU SM usage (DMA engine driver)
```
### Path ③: Local SSD → Local CPU → Local GPU
```
NVMe SSD → io_uring (O_DIRECT) → ClientBuffer (CPU staging ~1GiB) → RDMA loopback → GPU VRAM

SSD bandwidth: ~25+ GB/s (RAID0 4 drives)
Limitation: disk offload currently does not work in single-node multi-GPU scenarios (Kimi TP4 is affected)
```
### Path ④: Remote CPU Memory → Local GPU
```
Scenario A (current default): Remote CPU → RNIC → RoCE network → Local RNIC → Local GPU (~50 GB/s)
Solution B (optional in the cabinet): Remote CPU → C2C → Remote GPU → NVLink → Local GPU (~450 GB/s)
Prerequisite: protocol="nvlink" + USE_MNNVL compilation flag + the same NVL72 cabinet
Limitation: protocol is a global configuration. If nvlink is changed, it will break across cabinets.
```
### Path ⑤: Remote SSD → Remote CPU → Local GPU
```
Remote NVMe → io_uring → Remote ClientBuffer → (RDMA/NVLink) → Local GPU

Implementation: FileStorage RPC two-phase
1. RPC: batch_get_offload_object → read SSD remotely to ClientBuffer
2. Transfer Engine: RDMA pulls from the remote ClientBuffer to the local GPU
Two-stage serialization, ClientBuffer is a limited resource (~1GiB)
```
### Path ⑥: Remote GPU → Local GPU (P2P direct transmission, without going through the Store)
```
NixlConnector: UCX cuda_ipc (MNNVL NVLink passthrough, in-cabinet ~900 GB/s)
UCX cuda_copy+tcp (cross-cabinet ~50 GB/s)
MooncakeConnector: RDMA GPUDirect Write (~50 GB/s) or NVLink
```
---

## 9. Judgment flow chart
```
GPU requires KV block
  │
├─ vLLM prefix caching hits local GPU HBM?
│ └─ YES → Path ① Directly reuse without going through Mooncake
  │
  └─ NO → batch_get_into_multi_buffers(key)
       │
       ▼
Master BatchQuery(key) → replica list
       │
├─ Only LOCAL_DISK replica?
│ └─ YES → FileStorage RPC → SSD → staging → RDMA → GPU (path ③/⑤)
       │
└─ NO → FindFirstCompleteReplica → first COMPLETE (usually MEMORY)
            │
├─ replica on local node CPU? → RDMA loopback (path ②)
└─ replica in remote node CPU? → RDMA remote read (path ④)
↑ Local/remote is randomly assigned by PutStart
```
---

## 10. Current limitations and optimization directions

| Problem | Current Situation | Potential Optimizations |
|------|------|---------|
| Put is randomly allocated, data may span nodes | `preferred_segments=[]`, Master is randomly selected | vLLM passes `preferred_segments=[local_hostname]` |
| Get does not distinguish between local/remote | `prefer_alloc_in_same_node=false` | vLLM passes `true` to enable endpoint grouping batching |
| Use RDMA instead of NVLink in the cabinet | `protocol="rdma"` global configuration | Change to `"nvlink"` (but it will break across cabinets) |
| Disk offload is not available for multiple GPUs | Mooncake’s current implementation limitations | Waiting for upstream fixes |
| No NUMA/topology awareness | Replica selection does not consider distance | Mooncake side modification required |
| `GetPreferredReplica` is not used for BatchGet | Only single key Get has local priority | Requires Mooncake or vLLM side modification |

---

## Appendix A: Key source code files

| Documentation | Content |
|------|------|
| `vllm/.../mooncake_store_worker.py` | vLLM Worker: send/recv thread, register_kv_caches, lookup |
| `vllm/.../mooncake_store_connector.py` | vLLM Scheduler: build_connector_meta |
| `vllm/.../mooncake_connector.py` | P2P connector: ZMQ handshake + TransferEngine |
| `Mooncake/.../real_client.cpp` | C++ Client: batch_get/put, replica query |
| `Mooncake/.../client_service.cpp` | Client: BatchGet, GetPreferredReplica, MountSegment |
| `Mooncake/.../replica.h` | Replica type: MEMORY, DISK, LOCAL_DISK |
| `Mooncake/.../allocation_strategy.h` | Allocation strategy: RandomAllocationStrategy |
| `Mooncake/.../segment.cpp` | Master: MountSegment → addAllocator |
| `Mooncake/.../multi_transport.cpp` | Transport selection: selectTransport() |
| `Mooncake/.../rdma_transport.cpp` | RDMA batch: Slice + watermark |
| `Mooncake/.../store_py.cpp` | Python binding: default parameters |

## Appendix B: NVL72 in-cabinet interconnection topology
```
┌───────────────── NVL72 cabinet ────────────────────┐
│ 9 × NVLink Switch Tray, 72 GPU fully interconnected │
│                                                │
│  Node (Compute Tray):                          │
│    GPU0 ←(C2C 900GB/s)→ Grace CPU0 ←(C2C)→ GPU1   │
│    GPU2 ←(C2C 900GB/s)→ Grace CPU1 ←(C2C)→ GPU3   │
│                                                │
│ GPU↔GPU (inside the cabinet): NVLink 1.8 TB/s per GPU │
│ GPU↔CPU (same as SC): NVLink-C2C 900 GB/s bidirectional │
│CPU↔CPU (cross-SC): No direct connection, requires GPU or RDMA transfer │
│ Node↔Node (cross-counter): RoCE RDMA │
└────────────────────────────────────────────────┘
```
