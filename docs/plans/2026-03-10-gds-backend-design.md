# GDS Transfer Backend: GPUDirect Storage for KV Cache Offloading

**Date**: 2026-03-10
**Status**: Implemented

## Problem

The current `DiskTransferBackend` routes all GPU-to-disk transfers through CPU staging buffers: GPU -> pinned CPU -> `pwrite()` for stores, and `pread()` -> pinned CPU -> GPU for loads. This double-copy through the CPU is the primary bottleneck, especially on systems with fast NVMe SSDs where the disk I/O itself is not the limiting factor.

## Design

Add a new `GDSTransferBackend` that uses NVIDIA GPUDirect Storage (GDS) via the [kvikio](https://github.com/rapidsai/kvikio) library to perform direct DMA between GPU memory and NVMe storage, bypassing the CPU entirely.

### Architecture

`GDSTransferBackend` is a new `TransferBackend` implementation selected via `backend_type: "gds"`. It uses kvikio's `CuFile` to read/write directly between GPU tensors and a flat file on an NVMe filesystem.

```text
Current (DiskTransferBackend):
  Store: GPU -> pinned CPU staging -> pwrite() -> NVMe
  Load:  NVMe -> pread() -> pinned CPU staging -> GPU

New (GDSTransferBackend):
  Store: GPU -> DMA -> NVMe  (via kvikio CuFile.pwrite)
  Load:  NVMe -> DMA -> GPU  (via kvikio CuFile.pread)
```

No CPU staging buffers, no thread pool, no CUDA streams managed by the backend. kvikio handles parallelism and DMA internally.

### Data Path

**Store (GPU -> Disk):**

1. For each layer and block, call `CuFile.pwrite(gpu_tensor[block_id], size, file_offset)` (non-blocking, returns `IOFuture`)
2. Collect all `IOFuture` objects into a `_GDSEvent`
3. Return the event for the worker's async completion tracking

**Load (Disk -> GPU):**

1. For each layer and block, call `CuFile.pread(gpu_tensor[block_id], size, file_offset)` (non-blocking, returns `IOFuture`)
2. Collect all `IOFuture` objects into a `_GDSEvent`
3. Return the event for the worker's async completion tracking

### File Layout

Identical to `DiskTransferBackend` (layer-major flat file):

```text
offset = layer_idx * num_blocks * bytes_per_block + block_id * bytes_per_block
```

Same file format means the two disk backends are interchangeable at the config level.

### Alignment and Fallback

GDS requires 4KB-aligned file offsets and sizes for optimal direct DMA. At setup time:

- Check if `bytes_per_block % 4096 == 0`
- If not aligned: log a warning and set kvikio to compatibility mode (POSIX fallback). Performance will be similar to `DiskTransferBackend` but the backend still functions correctly.
- If aligned: GDS direct DMA path is used automatically.

In practice, KV cache block sizes are almost always 4KB-aligned for common models and page sizes.

### Event Model

`_GDSEvent` wraps a list of `kvikio.IOFuture` objects:

- `query()` -> `True` if all futures report `done()`
- `wait()` -> calls `.get()` on each future (blocks until complete)

This matches the `_DiskEvent` / `torch.cuda.Event` interface the worker already uses.

### kvikio Dependency

kvikio is an **optional** dependency (lazy import). `GDSTransferBackend.__init__` imports kvikio at runtime and raises a clear error if not installed:

```text
GDSTransferBackend requires kvikio. Install with: pip install kvikio-cu12
```

No changes to requirements files. The user installs kvikio independently when they want GDS support.

### Configuration

Same config keys as the `disk` backend — only `backend_type` changes:

```python
{
    "kv_connector": "SimpleCPUOffloadConnector",
    "kv_connector_extra_config": {
        "backend_type": "gds",
        "disk_bytes_to_use": 100_000_000_000,
        "disk_path": "/mnt/nvme/vllm_kv_cache",
    },
}
```

Works with MultiConnector tiered composition the same way as `disk`.

### Key Design Decisions

| Decision | Choice | Rationale |
| -------- | ------ | --------- |
| New backend vs extend existing | New `GDSTransferBackend` class | Clean separation, no runtime detection complexity |
| kvikio dependency | Optional (lazy import) | Not all users have GDS-capable hardware |
| Alignment handling | Warning + compat mode fallback | No worse than current disk backend when unaligned |
| File layout | Same as DiskTransferBackend | Interchangeable, no migration needed |
| I/O parallelism | kvikio internal thread pool | No need to manage our own ThreadPoolExecutor |
| Staging buffers | None | GDS DMA bypasses CPU entirely |

## File Changes

| File | Change |
| ---- | ------ |
| `simple_cpu_offload/backends/gds.py` | New: `_GDSEvent` + `GDSTransferBackend` |
| `simple_cpu_offload/backends/__init__.py` | Add `GDSTransferBackend` export |
| `simple_cpu_offload_connector.py` | Add `"gds"` case to `backend_type` switch |
| `benchmarks/benchmark_cpu_offloading.sh` | Add `gds` to `OFFLOAD_MODE` |
| `benchmarks/benchmark_simple_cpu_offload.py` | Add `gds` to `--backend` choices |
| Tests | Mock-based unit test + gated integration test |

## What Does Not Change

- `TransferBackend` ABC — no new abstract methods
- `worker.py` — already backend-agnostic
- `manager.py` — unchanged
- `metadata.py` — unchanged

## Testing

- **Unit tests:** Mock kvikio to verify correct API calls (setup, pread/pwrite, event lifecycle). No GPU or NVMe required.
- **Integration tests:** Gated behind `@pytest.mark.skipif(not has_kvikio, ...)` for environments with GDS hardware.
