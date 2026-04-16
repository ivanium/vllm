# RFC: Reduce CPU Overhead in Mooncake Store Put Path

## Status

Draft

## Summary

Two optimizations to reduce CPU overhead in the Mooncake Store sending
thread: (1) merge multiple requests' keys into a single batch RPC call,
and (2) bind Python transfer threads to the GPU's NUMA node.

## Problem 1: Serial Per-Request RPCs

`KVCacheStoreSendingThread` processes requests one at a time. Each
`_handle_request` call makes 3 synchronous Master RPCs:

```
_handle_request(req_meta)
├── batch_is_exist(keys)                     ← RPC #1
├── current_event.synchronize()              ← CPU wait
└── batch_put_from_multi_buffers(keys, ...)  ← RPC #2 (PutStart) + RPC #3 (PutEnd)
```

With 8 concurrent requests × 4 chunks each, that is **24 Master RPCs
executed serially** (8×exist + 8×start + 8×end).

## Problem 2: No CPU Affinity for Transfer Threads

The Python `KVCacheStoreSendingThread` and `KVCacheStoreRecvingThread`
are plain `threading.Thread` with no CPU affinity. The OS may schedule
them on a NUMA node far from the RDMA NIC, causing:

- Cross-NUMA memory access for Slice construction
- Cache misses on the Python→C++ boundary
- Suboptimal PCIe routing for GPU memory access

Note: Mooncake's C++ RDMA worker threads already bind to the NIC's NUMA
node (`bindToSocket` in `worker_pool.cpp:385`), but the Python calling
threads are not covered. The nixl connector already does this
(`os.sched_setaffinity` at `nixl_connector.py:1134`).

## Solution 1: Cross-Request Batch Merging

### Design

Override `run()` in `KVCacheStoreSendingThread` to drain all queued
requests and merge their keys before issuing RPCs:

```
Before (serial):  req1: exist→sync→put | req2: exist→sync→put | ...
After  (merged):  drain all → merged_exist → sync → merged_put
```

### Mooncake Per-Key Error Isolation

A merged `batch_put_from_multi_buffers(32_keys)` does NOT cause
all-or-nothing failure. Mooncake's 4-phase protocol provides per-key
error isolation:

1. **PutStart** (`client_service.cpp:1500`): Each key is independently
   allocated. Key A failing does not affect Key B.
2. **SubmitTransfers** (`client_service.cpp:1572`): Keys that failed
   PutStart are skipped; only successful keys get RDMA transfers.
3. **FinalizeBatchPut** (`client_service.cpp:1710`): Successful keys
   go to `BatchPutEnd`; failed keys go to `BatchPutRevoke`.
4. **CollectResults** (`client_service.cpp:1908`): Returns per-key
   status codes in a `vector<expected<void, ErrorCode>>`.

The Python `batch_put_from_multi_buffers` returns `list[int]` where
each element is the status for the corresponding key (0=success,
negative=error code).

### Expected Improvement

| Metric | Before | After |
|--------|--------|-------|
| Master RPCs per step (8 req × 4 chunks) | 24 | 3 |
| CUDA event synchronize calls | 8 | 1 |
| TransferEngine submit calls | 8 | 1 (better doorbell coalescing) |

## Solution 2: NUMA CPU Affinity

### Design

Add `_try_bind_numa()` to `KVTransferThread.run()` that binds the
thread to CPU cores on the same NUMA node as the current GPU, using
`os.sched_setaffinity()` and `discover_numa_topology()`.

Graceful fallback: if the platform does not support affinity or NUMA
discovery fails, log a debug message and continue without binding.

## Future Work

- Apply similar batching to the receiving thread
- Expose batch size limits as configuration
- GPU-to-NIC NUMA affinity (bind to NIC's NUMA node, not just GPU's)
