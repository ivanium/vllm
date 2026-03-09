# Disk Backend V2: MultiConnector Composition

**Date**: 2026-03-10
**Status**: Implemented (Option A)

## Problem

The current disk backend implementation embeds ~200 lines of disk-specific logic directly into `SimpleCPUOffloadConnector`: two-hop state machines, disk manager/worker composition, sentinel remapping, and metadata field merging. This makes the connector complex and hard to maintain.

## Design

Use `MultiConnector` to compose two independent `SimpleCPUOffloadConnector` instances — one for GPU↔CPU, one for GPU↔Disk. Each connector stays single-purpose.

### Architecture

```text
MultiConnector
├── SimpleCPUOffloadConnector (backend_type=cuda)
│   └── GPU ↔ CPU via Triton kernels + CUDA streams
└── SimpleCPUOffloadConnector (backend_type=disk)
    └── GPU ↔ Disk via torch .to() + pwrite/pread thread pool
        (internal pinned CPU staging buffers)
```

Both connectors independently scan the GPU block pool for LRU eviction candidates (lazy store) and check their respective caches for prefix hits (load). MultiConnector picks the first hit — CPU is checked first (faster).

### Configuration

```python
# Example: tiered GPU → CPU + Disk caching
engine_args = {
    "kv_connector": "MultiConnector",
    "kv_connector_extra_config": {
        "connectors": [
            {
                "kv_connector": "SimpleCPUOffloadConnector",
                "kv_connector_extra_config": {
                    "cpu_bytes_to_use": 8_000_000_000,
                    "lazy_offload": True,
                },
            },
            {
                "kv_connector": "SimpleCPUOffloadConnector",
                "kv_connector_extra_config": {
                    "backend_type": "disk",
                    "disk_bytes_to_use": 100_000_000_000,
                    "disk_path": "/mnt/nvme/vllm_kv_cache",
                    "lazy_offload": True,
                },
            },
        ],
    },
}
```

### Key Design Decisions

| Decision | Choice | Rationale |
| ---------- | -------- | ----------- |
| Composition | MultiConnector, not embedded in connector | Each connector stays single-purpose |
| Store path | Both connectors independently scan GPU pool | Simple, no cross-connector dependencies |
| Load path | Direct disk→GPU via internal staging | No two-hop state machine needed |
| Disk I/O | torch `.to("cuda", non_blocking=True)` + pwrite/pread | Simple, torch handles DMA + scatter |
| Staging buffers | Fixed small pool inside DiskTransferBackend | Low memory, sufficient for batched transfers |
| Future cascading (Option B) | Deferred to follow-up PR | Clean additive change via `source_connector_index` |

## Changes

### 1. SimpleCPUOffloadConnector — Parameterize Backend

Add `backend_type` config (`"cuda"` default, `"disk"` option). ~15 lines added:

```python
backend_type = str(extra_config.get("backend_type", "cuda"))
if backend_type == "disk":
    capacity_bytes = int(extra_config.get("disk_bytes_to_use", 0))
    disk_path = str(extra_config.get("disk_path", "/tmp/vllm_kv_cache"))
    backend = DiskTransferBackend(disk_path=disk_path)
else:
    capacity_bytes = int(extra_config.get("cpu_bytes_to_use",
                         DEFAULT_CPU_CAPACITY_BYTES))
    backend = CudaTransferBackend()
```

Remove all disk-specific code (~200 lines): `_TwoHopLoadState`, `_pending_disk_loads`, `_two_hop_loads`, `_setup_two_hop_load()`, `_queue_hop2()`, `_cleanup_two_hop_state()`, disk sentinel handling, disk field merging, `disk_manager`, `disk_worker`.

Add `get_offload_block_pool()` method (hook for future Option B cross-wiring):

```python
def get_offload_block_pool(self) -> "BlockPool | None":
    if self.scheduler_manager is not None:
        return self.scheduler_manager.cpu_block_pool
    return None
```

### 2. DiskTransferBackend — Rewrite for Direct GPU↔Disk

The current implementation assumes CPU tensors as source. Rewrite for direct GPU↔Disk with internal staging:

**State:**

- GPU KV caches (reference, from `setup()`)
- Pre-allocated flat disk file
- Fixed-size pinned CPU staging buffer pool (default 64 blocks)
- Dedicated low-priority CUDA stream for GPU↔staging copies (never touches the compute stream)
- `ThreadPoolExecutor` for disk I/O

**Store (GPU→Disk):**

1. On the dedicated CUDA stream: `staging[:n] = gpu_kv_caches[src_ids]` — GPU→staging via torch
2. Record CUDA event on the dedicated stream, synchronize only that event (not the compute stream)
3. Submit `pwrite(staging_buffer, offset)` to thread pool (reads directly from pinned staging)
4. Return `_DiskEvent` set when all pwrite ops complete

**Load (Disk→GPU):**

1. Submit `pread` to thread pool, reading directly into the pinned staging buffer (zero-copy into pre-allocated tensor)
2. Wait for I/O completion
3. On the dedicated CUDA stream: `gpu_kv_caches[dst_ids] = staging[:n].to("cuda", non_blocking=True)`
4. Return `torch.cuda.Event` recorded on the dedicated stream after the copy

**Staging buffer pool lifecycle:**

- Allocated once in `setup()`: a dict of pinned CPU tensors mirroring GPU KV cache shapes but with `num_staging_blocks` blocks (default 64, configurable).
- Staging blocks are a shared pool used by both loads and stores. Since `copy_blocks` is called synchronously per batch (loads and stores don't overlap within a single `start_load_kv` call), no concurrent allocation contention occurs.
- If a batch has more blocks than staging capacity, `copy_blocks` processes them in chunks of `num_staging_blocks`, synchronizing between chunks.
- Staging buffers are never exposed to the scheduler or block pool — they are purely internal to the backend.

### 3. SimpleCPUOffloadMetadata — Remove Disk Fields

Delete all `disk_load_*` and `disk_store_*` fields. Each connector instance uses the standard `load_*`/`store_*` fields independently via `MultiKVConnectorMetadata`.

### 4. MultiConnector — Cross-Wiring Infrastructure (for future Option B)

Add `source_connector_index` config support and `get_offload_block_pool()` to base class. Not wired up in this PR — just the hooks.

## Behavioral Invariants

**Store deduplication via ordering:** `MultiConnector.build_connector_meta` calls each connector's `build_connector_meta` sequentially. The CPU connector runs first, touching GPU blocks (ref_cnt 0→1). When the disk connector runs second, those blocks are no longer at ref_cnt=0 and won't appear in `get_eviction_candidates`. This naturally deduplicates — each block is stored to at most one tier per scheduling step.

**Load priority via config order:** `MultiConnector.get_num_new_matched_tokens` returns the first hit. CPU connector is listed first in config, so CPU cache is always preferred over disk. A block may exist in both tiers (e.g., stored to both before one was evicted), but CPU is always checked first.

**`register_kv_caches` wiring:** `MultiConnector` calls `register_kv_caches(gpu_kv_caches)` on all connectors uniformly. The disk connector receives GPU KV caches directly — this is intentional for direct GPU↔Disk transfers. No special wiring needed (unlike the old two-hop approach).

**`bind_connector_metadata` simplification:** `MultiConnector` handles per-connector metadata binding via `MultiKVConnectorMetadata.metadata` tuple. The current `SimpleCPUOffloadConnector.bind_connector_metadata` override that remaps disk fields into a synthetic metadata is deleted entirely.

**`SupportsHMA`:** `MultiConnector` needs `request_finished_all_groups` when composing HMA-supporting connectors. Added to file changes.

**Disk file management:** The file is created with `O_CREAT | O_TRUNC` (fresh each run). `shutdown()` closes the fd and unlinks the file. Multiple vLLM instances must use different `disk_path` values.

## Future: Option B (Cascading Stores)

A follow-up PR adds cascading: disk connector scans CPU block pool instead of GPU pool.

Changes needed:

- Manager: add `_source_block_pool` field + `bind_source_block_pool()` (~5 lines)
- MultiConnector: `bind_gpu_block_pool()` cross-wires based on `source_connector_index`
- DiskTransferBackend: `bind_store_source_caches()` to read from CPU tensors for stores

No rewrites — purely additive.

## File Changes

| File | Change |
| ------ | -------- |
| `simple_cpu_offload_connector.py` | Remove disk code (~200 lines), add backend_type (~15 lines), add `get_offload_block_pool()` |
| `simple_cpu_offload/backends/base.py` | `TransferBackend` ABC |
| `simple_cpu_offload/backends/cuda.py` | `CudaTransferBackend` (pinned CPU + Triton kernels) |
| `simple_cpu_offload/backends/disk.py` | `DiskTransferBackend` for direct GPU↔Disk |
| `simple_cpu_offload/metadata.py` | Remove disk fields |
| `simple_cpu_offload/worker.py` | No changes |
| `simple_cpu_offload/manager.py` | No changes |
| `multi_connector.py` | Add cross-wiring infrastructure (future Option B), add `request_finished_all_groups` |
| `base.py` | Add `get_offload_block_pool()` default |
| Tests | Update for new architecture, remove two-hop tests, clean up |
