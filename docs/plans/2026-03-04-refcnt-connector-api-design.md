# Ref_cnt-Based Connector API for CPU Offloading

## Problem

The current SimpleCPUOffload connector ties request lifecycle to block lifecycle via a delayed-freeing protocol:

1. `request_finished()` returns `(True, None)` when stores are in-flight
2. The scheduler delays calling `_free_blocks()` and keeps the request in `self.requests`
3. The worker-side `get_finished()` uses two-phase timing logic to track store completion vs request completion
4. Only when `finished_sending` arrives does the scheduler free blocks

This creates tight coupling between the scheduler and connector, complex timing state (`_stores_completed_reqs`, `_finished_reqs_waiting_for_store`), and requires the scheduler to understand connector-internal async behavior.

## Solution

Use BlockPool's existing reference counting to decouple block lifecycle from request lifecycle. The connector increments `ref_cnt` on GPU blocks when submitting a store, and decrements when the store completes. The scheduler always frees blocks immediately.

## Design

### Core Mechanism

**Store submission** (`_prepare_store_specs()`):
```python
if src_gpu_blocks and self._gpu_block_pool is not None:
    gpu_blocks = [self._gpu_block_pool.blocks[bid] for bid in src_gpu_blocks]
    self._gpu_block_pool.touch(gpu_blocks)
    self._pending_gpu_store_blocks[req_id].extend(src_gpu_blocks)
```

**Request finished** (`request_finished()`):
```python
def request_finished(self, request, block_ids):
    req_id = request.request_id
    is_storing = req_id in self._storing_requests
    if not is_storing:
        self._cleanup_request(req_id)
    # Always False — scheduler frees immediately, ref_cnt keeps blocks alive
    return False, None
```

**Store completion** (`update_connector_output()`):
```python
# After caching CPU blocks...
gpu_block_ids = self._pending_gpu_store_blocks.pop(req_id, [])
if gpu_block_ids and self._gpu_block_pool is not None:
    self._gpu_block_pool.free_blocks(
        self._gpu_block_pool.blocks[bid] for bid in gpu_block_ids
    )
self._cleanup_request(req_id)
```

### Ref_cnt Trace

```
Step N:   _prepare_store_specs() → touch(gpu_blocks) → ref_cnt: 1 → 2
Step N+M: Request finishes → kv_cache_manager.free() → ref_cnt: 2 → 1 (block stays alive)
Step N+K: Store completes → connector free_blocks() → ref_cnt: 1 → 0 (block freed to pool)
```

### BlockPool.touch() Handles Both ref_cnt States

For blocks with ref_cnt > 0 (active request blocks):
- `touch()` increments ref_cnt (block already not in free queue)

For blocks with ref_cnt == 0 (prefix-cached blocks, lazy mode):
- `touch()` removes from free queue, then increments to 1
- Block is protected from allocation/eviction during copy
- `free_blocks()` decrements to 0, adds back to free queue

### get_finished() Simplification

The two-phase timing logic is eliminated. `get_finished()` no longer needs `finished_req_ids` for store tracking:

```python
def get_finished(self, finished_req_ids):
    # Load completions (unchanged)
    for job_idx in [j for j in self._pending_load_wm_jobs if j <= load_wm]:
        finished_recving.update(self._pending_load_wm_jobs.pop(job_idx))

    # Store completions (simplified — no two-phase timing)
    for job_idx in [j for j in self._pending_store_wm_jobs if j <= store_wm]:
        fired_store_reqs.update(self._pending_store_wm_jobs.pop(job_idx))

    still_pending = set().union(*self._pending_store_wm_jobs.values()) if ... else set()

    for req_id in fired_store_reqs:
        if req_id not in still_pending:
            finished_sending.add(req_id)
            # No two-phase: emit immediately, connector handles internally

    # finished_req_ids parameter is IGNORED for stores
    return finished_sending or None, finished_recving or None
```

Removed state:
- `_stores_completed_reqs`
- `_finished_reqs_waiting_for_store`

### Scheduler Changes

`_update_from_kv_xfer_finished()` (scheduler.py:2091-2094):
```python
for req_id in kv_connector_output.finished_sending or ():
    if req_id not in self.requests:
        continue  # Already freed via ref_cnt (SimpleCPUOffload)
    self._free_blocks(self.requests[req_id])  # NIXL path (unchanged)
```

This is backward-compatible: NIXL returns `(True, ...)` from `request_finished()`, so blocks stay in `self.requests` and the existing path works. SimpleCPUOffload returns `(False, None)`, so blocks are freed immediately and the `finished_sending` loop skips them.

### Lazy Mode

Lazy mode stores blocks selected from GPU eviction candidates (ref_cnt == 0). The same mechanism applies:

1. `_prepare_lazy_store_specs()`: touch GPU blocks → ref_cnt: 0 → 1, removed from free queue
2. Copy executes (blocks can't be evicted)
3. `update_connector_output()`: free GPU blocks → ref_cnt: 1 → 0, back in free queue

New state for lazy mode: track GPU block IDs per lazy job for ref_cnt decrement on completion.

### Cleanup Safety

`_cleanup_request()` handles GPU ref_cnt as a safety net (for aborts/preemption):
```python
gpu_block_ids = self._pending_gpu_store_blocks.pop(req_id, [])
if gpu_block_ids and self._gpu_block_pool is not None:
    self._gpu_block_pool.free_blocks(
        self._gpu_block_pool.blocks[bid] for bid in gpu_block_ids
    )
```

### Load Blocks — No Change

Loads don't need this mechanism. The request waits in `WAITING_FOR_REMOTE_KVS` until the load completes — no block lifecycle conflict. CPU block `touch()` during loads (existing behavior) is unchanged.

## Ordering Guarantees

Touch always happens before free, guaranteed by the scheduler's step-based execution:

- Touch: during `build_connector_meta()` (end of scheduling step N)
- Free: during `update_from_output()` (beginning of step N+M, M >= 1)

A request cannot finish in the same step its store is submitted, because request completion comes from model output which arrives at the start of the NEXT step.

No double-free: each decrement comes from a different source (kv_cache_manager vs connector), each happens exactly once.

No double-use: when kv_cache_manager decrements ref_cnt, block stays alive (ref_cnt > 0), so it's not allocated to new requests.

## Preemption

When a request is preempted with in-flight stores:

1. Scheduler calls `kv_cache_manager.free()` → ref_cnt: 2 → 1 (block stays alive)
2. Worker calls `handle_preemptions()` → syncs streams
3. Store eventually completes → `update_connector_output()` → ref_cnt: 1 → 0 (block freed)

If the request is re-scheduled before the store completes, it gets new blocks. The old blocks remain alive until the connector releases them.

## NIXL / PD Compatibility

This design is fully backward-compatible:

- Base class API unchanged
- `request_finished()` return value semantics unchanged: `(True, ...)` = delay freeing, `(False, ...)` = free immediately
- NIXL continues to use `(True, ...)` and the existing delayed-freeing path
- The scheduler change at `_update_from_kv_xfer_finished()` gracefully handles both patterns

For future NIXL adoption:
1. Add `bind_gpu_block_pool()` to NIXL connector
2. Touch GPU blocks when starting P→D sends
3. Return `(False, ...)` from `request_finished()`
4. Decrement ref_cnt when sends complete

## Scope

- **In scope:** SimpleCPUOffload connector (eager + lazy mode), scheduler compatibility
- **Out of scope:** NIXL migration, unified GPU+CPU block pool manager, GPU↔CPU block correspondence tracking

## Files Changed

| File | Change |
|------|--------|
| `simple_cpu_offload/manager.py` | Add `_pending_gpu_store_blocks`, touch/free GPU blocks, simplify `request_finished()` |
| `simple_cpu_offload_connector.py` | Remove `_stores_completed_reqs`, `_finished_reqs_waiting_for_store`, simplify `get_finished()` |
| `v1/core/sched/scheduler.py` | Handle missing req_ids in `_update_from_kv_xfer_finished()` |
