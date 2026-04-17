# feat/mooncake-store-int — Summary

Branch compared against: `main`
Date: 2026-04-13

---

## What's New

The branch introduces **MooncakeStoreConnector**, a new KV cache connector alongside the existing `MooncakeConnector`. The key architectural difference:

| | MooncakeConnector (existing) | MooncakeStoreConnector (new) |
|---|---|---|
| Transfer model | Direct P2P (prefiller → decoder) | Shared distributed store (hash-based) |
| Prefix caching | No | Yes — content-addressed deduplication |
| Disk offloading | No | Yes |
| DP sharing | No | Yes (via PYTHONHASHSEED=0) |

---

## New Files

| File | Purpose |
|---|---|
| [`mooncake_store_connector.py`](../vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_connector.py) | vLLM `KVConnectorBase_V1` interface; dispatches to Scheduler/Worker |
| [`mooncake_store_scheduler.py`](../vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_scheduler.py) | Prefix cache lookup via ZMQ RPC, builds per-step load/save specs |
| [`mooncake_store_worker.py`](../vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_worker.py) | Async send/recv threads, KV layout detection, disk offload logic |
| [`mooncake_store_data.py`](../vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_data.py) | `PoolKey`, `ChunkedTokenDatabase`, `ReqMeta`, `LoadSpec` data structures |
| [`mooncake_utils.py`](../vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_utils.py) | Bootstrap server (FastAPI), config loading from `MOONCAKE_CONFIG_PATH` |
| [`scripts/mooncake/`](../scripts/mooncake/) | README, benchmark scripts, config examples |

---

## Architecture

```
Scheduler process                    Worker processes
─────────────────────                ────────────────────────────────
MooncakeStoreScheduler               MooncakeStoreWorker
 ├─ ZMQ lookup client  ──RPC──────►  LookupKeyServer (serves block hash hits)
 ├─ RequestTracker (per req)          ├─ KVCacheStoreSendingThread  ──put_async()──► MooncakeDistributedStore
 └─ build_connector_meta()            └─ KVCacheStoreRecvingThread  ◄─get_async()──  (CPU/disk tiered)
      ↓ MooncakeStoreConnectorMetadata (via shared memory / IPC)
```

**Store key format:** `model@tp:X@pcp:Y@dcp:Z@pp:W@<chunk_hash>` — enables cross-DP deduplication when `PYTHONHASHSEED=0`.

---

## Key Features (by commit)

1. **Core store integration** — `put_async`/`get_async` to `MooncakeDistributedStore`; compute/I/O overlap via `get_finished()` issuing I/O after GPU compute launch

2. **Disk offloading** — Tiered GPU → CPU → disk; `MOONCAKE_ENABLE_OFFLOAD`, per-rank buffer size; direct-IO aligned batch splitting to fit Mooncake's stage buffer

3. **CPU/disk pressure handling** — `stop_storing` logic: when `put_async()` returns `NO_AVAILABLE_HANDLE (-200)`, pauses saves for affected requests while continuing loads; auto-recovers when pressure subsides

4. **Chunked load batches** — Splits large load batches into smaller chunks to respect Mooncake stage buffer limits (commit `dcb203478`)

5. **Async enforcement** — All I/O is async; `load_async=True` is enforced, no synchronous blocking path (commit `cebbf1bcb`)

6. **Single-node DP** — `local_engines_only=True` uses `data_parallel_rank_local` for within-node cross-engine KV sharing (commit `e2eb23ba8`)

7. **Cross-layer blocks** — `prefer_cross_layer_blocks` property from `enable_cross_layers_blocks` extra config (commit `7bd7f7607`)

---

## Integration with vLLM v1 KV Connector Protocol

The connector implements the full `KVConnectorBase_V1` interface:

- **`get_num_new_matched_tokens()`** → ZMQ lookup for prefix hits
- **`update_state_after_alloc()`** → tracks allocated GPU blocks per request
- **`build_connector_meta()`** → produces `ReqMeta` (save/load specs) per step
- **`get_finished()`** → issues all async I/O after compute, returns completed request IDs
- **`request_finished()`** → delays GPU block free if async save still pending

---

## Data Structures

### PoolKey
Store address: `model@tp:X@pcp:Y@dcp:Z@pp:W@<chunk_hash>`
Enables hash-based deduplication: same token sequence hash → same store entry.

### ChunkedTokenDatabase
Token-to-GPU-address mapping. `process_tokens(token_len, block_hashes)` yields
`(start, end, PoolKey)` tuples; `prepare_value()` returns GPU tensor addresses/sizes.

### ReqMeta
Per-request I/O spec: `can_save`, `load_spec`, `is_last_chunk`, `block_hashes`.

### LoadSpec
Load operation parameters: `vllm_cached_tokens`, `kvpool_cached_tokens`, `can_load`, `token_len`.

### RequestTracker (scheduler-side)
Tracks allocated block IDs, token lengths, and saved token counts across scheduler ticks.
Supports request resumption after preemption.

### MooncakeStoreConnectorMetadata
Scheduler→Worker payload: batch of `ReqMeta`, unfinished req IDs, preempted req IDs.

---

## Worker Threading Model

Two background threads in `MooncakeStoreWorker`:

**KVCacheStoreSendingThread (save/put):**
- Processes `ReqMeta` queue
- TP-rank striding: each rank stores every `put_step`-th chunk to avoid redundant cross-TP storage
- Detects pressure via `NO_AVAILABLE_HANDLE` and activates `stop_storing` flag
- Async `put_async()` with CUDA events

**KVCacheStoreRecvingThread (load/get):**
- Processes `LoadSpec` queue
- Splits batches to fit Mooncake stage buffer limits
- Disk offload path: `_split_disk_offload_load_batches()` with direct-IO alignment
- Cyclic key-list rotation by TP rank for load balancing
- Async `get_async()` with `mask_num` to skip already-cached tokens

---

## Deployment

```bash
MOONCAKE_CONFIG_PATH=scripts/mooncake/mooncake_config.json
PYTHONHASHSEED=0               # required for cross-DP hash consistency
MOONCAKE_ENABLE_OFFLOAD=1      # optional disk offload
MOONCAKE_OFFLOAD_FILE_STORAGE_PATH=/nvme/mooncake
MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES=1342177280  # 1.25 GB per rank

# Connector selection:
--kv-transfer-config '{"kv_connector": "MooncakeStoreConnector", ...}'
```

### mooncake_config.json structure
```json
{
  "metadata_server": "http://IP:8080/metadata",
  "master_server_address": "IP:50051",
  "global_segment_size": "600GB",
  "local_buffer_size": "4GB",
  "protocol": "rdma",
  "device_name": ""
}
```
All sizes are **per-GPU (per-rank)**, not cluster-wide.

---

## Benchmarks

Two benchmark modes in [`scripts/mooncake/`](../scripts/mooncake/):

- **`benchmark_cpu_offloading.sh`** — single-turn random prompts; stresses raw I/O throughput (no prefix reuse)
- **`benchmark_multi_turn.sh`** — multi-turn conversations with shared global + per-conversation prefixes; measures prefix cache hit rate

Backends compared: `baseline`, `native`, `simple`, `mooncake`, `mooncake-mem`.

---

## Commit Log

```
4d8520e51 Merge pull request #15 from aoshen524/pr/mooncake-config-json
cd9ff11c3 docs(mooncake): add gb200 rack1 config example
21e84aa0f fix (simple_cpu_backend): allow multi-connector to pass bind_gpu_block_pool() to all connectors
e2eb23ba8 feat/fix: support single-node DP
7bd7f7607 feat: add prefer_cross_layer_blocks property to MooncakeStoreConnector
9998c1db5 doc: update README to recommend pre-compiled wheel
87e434c33 doc: update README with latest instructions
081a89c89 chore: update env vars and hyper params for disk offload
12b8e3eba feat: split load batch to chunks to adapt to mooncake stage buffer
dcb203478 feat: stop storing current reqs when CPU/disk offloading is under pressure
4a98f1ecd feat: enforce load_async and better overlapping
cebbf1bcb feat: support disk offloading
39ebab358 chore: update README and config to use RDMA
bf0d8b031 fix (worker): identify different kv cache layouts
426d5021f chore (scripts): add running scripts and instructions
3c305908a fix (worker): adapt to new kv_cache type applied in #37484
c3c6d629e style: fix pre-commit issues after mooncake connector
1434f5164 Add mooncake store connector
```
