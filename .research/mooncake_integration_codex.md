# `feat/mooncake-store-int` vs `main`: Mooncake KV Cache Integration

This note compares branch `feat/mooncake-store-int` against `main` and explains what was added to integrate vLLM with Mooncake as a shared KV cache store.

At a high level, this branch does **not** just tweak the existing Mooncake connector. It adds a second integration path, `MooncakeStoreConnector`, that treats Mooncake as a **content-addressed distributed KV store** instead of a direct peer-to-peer transport. That shift is what enables external prefix-cache lookup, deduplicated storage by block hash, CPU+disk offload tiers, and reuse across data-parallel engines.

## Diff Summary

Branch-only work is concentrated in Mooncake-specific connector code, tests, and operational scripts.

- 21 files changed
- ~3.8k lines added
- 4 new core implementation files:
  - `mooncake_store_connector.py`
  - `mooncake_store_scheduler.py`
  - `mooncake_store_data.py`
  - `mooncake_store_worker.py`
- Supporting changes:
  - connector registration in `kv_connector/factory.py`
  - `MultiConnector.bind_gpu_block_pool()` forwarding
  - single-node DP helper in `mooncake_utils.py`
  - tests for connector behavior and worker logic
  - `scripts/mooncake/` setup, benchmark, and config assets

## What Was Added

### 1. A new connector: `MooncakeStoreConnector`

The branch registers a new connector name, `MooncakeStoreConnector`, in the vLLM KV connector factory. This is the main integration point that allows the runtime to select Mooncake as a KV transfer backend via:

```json
{
  "kv_connector": "MooncakeStoreConnector",
  "kv_role": "kv_both"
}
```

Unlike the pre-existing `MooncakeConnector`, which is organized around direct engine-to-engine send/receive, `MooncakeStoreConnector` splits into:

- a **scheduler-side** component (`MooncakeStoreScheduler`)
- a **worker-side** component (`MooncakeStoreWorker`)
- a structured metadata payload (`MooncakeStoreConnectorMetadata`)

The connector implements the v1 connector protocol methods that matter for external KV reuse:

- `get_num_new_matched_tokens()`
- `update_state_after_alloc()`
- `build_connector_meta()`
- `request_finished()`
- `get_finished()`

Two methods are intentionally turned into no-ops:

- `start_load_kv()`
- `wait_for_save()`

That is deliberate. This branch moves both loads and stores into `get_finished()` so I/O can be issued after compute launch and overlap with GPU work.

### 2. A scheduler-side prefix lookup path

`MooncakeStoreScheduler` is responsible for deciding whether a request can reuse KV already stored in Mooncake.

It does that by:

- normalizing the effective block size for PCP/DCP
- querying a local ZMQ RPC endpoint through `LookupKeyClient`
- recording per-request `LoadSpec`
- building `ReqMeta` objects for each scheduler step
- tracking unfinished and preempted requests across ticks

The lookup path uses request `block_hashes` and asks, effectively: "how many prompt tokens already exist in the Mooncake-backed KV pool?"

If there is a hit, the scheduler:

- computes how many tokens can be loaded externally
- asks vLLM to allocate enough local blocks for those tokens
- marks the request as loadable only after allocation succeeds

### 3. A worker-side Mooncake store engine

`MooncakeStoreWorker` owns the actual Mooncake integration:

- loads config from `MOONCAKE_CONFIG_PATH`
- creates a `MooncakeDistributedStore`
- registers KV cache memory buffers with Mooncake
- starts one background receive thread
- optionally starts one background send thread
- hosts the lookup logic that checks store existence by key

This worker is where the store semantics live:

- `batch_is_exist()` for dedup/prefix lookup
- `batch_put_from_multi_buffers()` for saving KV chunks
- `batch_get_into_multi_buffers()` for restoring KV chunks

### 4. A content-addressed key model

The new data model in `mooncake_store_data.py` builds store keys out of:

- model name
- TP rank or KV-head-derived rank
- PCP rank
- DCP rank
- PP rank
- block hash

The resulting string looks like:

```text
<model>@tp_rank:<x>@pcp<y>@dcp<z>@pp_rank:<w>@<chunk_hash>
```

That matters because the branch is really storing by **hash identity**, not by request identity. Two requests with the same prefix hash chain can reuse the same Mooncake entry.

### 5. CPU and disk offload support

The worker config adds optional disk offload on top of CPU-backed Mooncake buffers. The branch introduces:

- `MOONCAKE_ENABLE_OFFLOAD`
- `MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES`
- disk staging budget calculations
- direct-I/O-aware batching and padding logic
- per-batch splitting when a load would overflow Mooncake's staging buffer

This is one of the most substantial pieces of the branch beyond the initial connector bring-up.

### 6. Pressure handling when Mooncake offload resources fill up

When store operations return `NO_AVAILABLE_HANDLE (-200)`, the send thread interprets that as CPU/disk-offload pressure. Instead of failing the whole request path, it:

- marks the current request as temporarily skip-save
- stops enqueuing further store writes for that request
- keeps the system alive for loads and other traffic
- clears the pressure state after a later successful store batch

This is a pragmatic reliability feature: under pressure, the system prefers degraded caching over hard failure.

### 7. Operational tooling and documentation

The branch adds a full Mooncake scripts directory for setup and benchmarking:

- master startup script
- env setup helper
- single-turn benchmark
- multi-turn benchmark
- comparison script
- sample JSON configs
- a bundled Mooncake wheel for ARM64 GB200/GH platforms

This is not just code integration; it is an attempt to make the feature runnable and measurable.

## Architecture Compared to `main`

Before this branch:

- vLLM already had `MooncakeConnector`
- that path was oriented around direct transfer between engines
- there was no Mooncake-backed shared KV-store connector in the factory

After this branch:

- `MooncakeStoreConnector` becomes a first-class connector
- the scheduler can query external prefix hits from Mooncake
- workers can save/load KV chunks through the store API
- identical prefixes can be deduplicated by block hash
- DP engines can share hits if their hash chains match
- Mooncake can serve as a CPU or CPU+disk-backed KV pool

In other words, the branch turns Mooncake from a transport mechanism into a reusable external KV memory tier.

## Key Files and Responsibilities

| File | Responsibility |
| --- | --- |
| `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_connector.py` | vLLM-facing connector class; scheduler/worker dispatch; event collection |
| `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_scheduler.py` | external prefix lookup, per-request tracking, scheduler metadata construction |
| `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_data.py` | store key format, request metadata, load/save specs |
| `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_worker.py` | Mooncake store setup, KV memory registration, async load/store threads, lookup server |
| `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_utils.py` | shared Mooncake helper for DP engine indexing |
| `vllm/distributed/kv_transfer/kv_connector/factory.py` | connector registration |
| `vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py` | forwards `bind_gpu_block_pool()` to all child connectors |
| `scripts/mooncake/*` | setup, config, launch, and benchmarking |

## Life of a Request

This is the most useful way to understand what the branch actually does.

### Phase 0: process startup

When vLLM starts with `MooncakeStoreConnector`:

1. The scheduler process builds `MooncakeStoreScheduler`.
2. Each worker builds `MooncakeStoreWorker`.
3. The worker loads Mooncake config from `MOONCAKE_CONFIG_PATH`.
4. The worker creates and configures `MooncakeDistributedStore`.
5. Once KV cache tensors are allocated, the worker registers their backing memory with Mooncake.
6. The worker starts:
   - a receive thread for loads
   - a send thread for stores, if the role permits saving
7. Worker rank 0 also starts a `LookupKeyServer` over ZMQ so the scheduler can query prefix hits.

The system is now ready to treat Mooncake as an external KV pool.

### Phase 1: a new request arrives

Suppose a prompt enters the scheduler.

The request already has block hashes from vLLM's prefix-cache hashing logic. `MooncakeStoreScheduler.get_num_new_matched_tokens()` does the first important piece of work:

1. It computes the effective chunkable token length.
   - By default it discards partial chunks, so it rounds down to full block multiples.
   - If the prompt is shorter than one effective block, Mooncake is skipped.
2. It sends the request's block-hash chain over ZMQ to the lookup server.
3. The worker-side lookup server asks `MooncakeStoreWorker.lookup()`.
4. The worker converts hashes into Mooncake keys and calls `batch_is_exist()`.
5. The lookup expands keys across TP ranks and PP ranks before deciding how much prefix is fully present.
6. The scheduler gets back a token count representing reusable external KV.

At this point, the scheduler knows whether Mooncake can satisfy any prefix portion of the request.

### Phase 2: scheduler decides how much to allocate

If Mooncake has reusable KV:

1. The scheduler computes `need_to_allocate = external_hit_tokens - already_computed_tokens`.
2. It stores a `LoadSpec` for the request.
3. vLLM allocates local GPU KV blocks for those externally available tokens.
4. `update_state_after_alloc()` marks the request as eligible for load only if the allocation matches the expected number of external tokens.

This is important: Mooncake data is not loaded directly into nowhere. The branch still relies on normal vLLM block allocation, then fills those blocks from the external store.

### Phase 3: scheduler builds per-step metadata

On each scheduler tick, `build_connector_meta()` builds a `MooncakeStoreConnectorMetadata` object.

For each active request, the scheduler may create a `ReqMeta` containing:

- request id
- total chunkable token length for this step
- allocated block ids
- block hashes
- optional `LoadSpec`
- whether the request can save
- whether this is the last chunk
- optional token ids for event emission

The scheduler also keeps track of:

- unfinished requests
- preempted requests
- resumed requests
- how many tokens have already been saved

That tracking is what lets the connector continue working across chunked prefills, decode continuation, and preemption/restart.

### Phase 4: model compute is launched

This branch deliberately does **not** issue Mooncake I/O during `start_load_kv()` or `wait_for_save()`.

Instead, when the worker later enters `get_finished()`:

1. it receives the metadata for the current scheduler step
2. it enqueues any necessary loads
3. it enqueues any necessary stores
4. it polls for completed prior work

The intent is to launch the GPU compute first, then push store I/O in parallel for better overlap.

This is one of the branch's defining design choices.

### Phase 5: loading KV from Mooncake

For requests whose `LoadSpec.can_load` is true:

1. `get_finished()` finalizes the load token length.
   - There is special handling for the "all but one token" edge case around block boundaries.
2. The request is handed to `KVCacheStoreRecvingThread`.
3. The receive thread walks the request's hashes via `ChunkedTokenDatabase.process_tokens()`.
4. For each chunk, it computes:
   - store key
   - local GPU destination address
   - copy size
5. It masks already local tokens using `vllm_cached_tokens`.
6. It rotates the key order by TP rank to spread read load.
7. If disk offload is enabled and the batch is too large, it splits the load into smaller sub-batches.
8. Each sub-batch is issued with `batch_get_into_multi_buffers()`.
9. When the load path finishes, the request id is marked complete on the receive side.

The core mapping work is done by `ChunkedTokenDatabase`:

- `process_tokens()` maps token ranges to hash-derived store keys
- `prepare_value()` maps those token ranges to exact GPU memory addresses

That is the bridge between logical token chunks and physical KV memory.

### Phase 6: saving KV into Mooncake

For requests whose `ReqMeta.can_save` is true:

1. `get_finished()` creates a CUDA event once for the batch of store work.
2. That event is attached to each save request.
3. The send thread waits on the event so the store only reads KV after GPU writes are visible.
4. The send thread converts the request into chunk keys and source addresses.
5. It applies TP striding via `put_step` so not every TP rank redundantly stores the same content.
   - This is especially relevant when KV heads are fewer than TP ranks.
6. It checks `batch_is_exist()` first.
   - already-present chunks are skipped
   - only missing chunks are written
7. Missing chunks are written with `batch_put_from_multi_buffers()`.
8. If KV cache event emission is enabled, `BlockStored` events are recorded.
9. The request's outstanding store-job count is decremented.

This is where deduplicated storage really happens. The branch is not blindly pushing all KV into Mooncake; it first asks whether the content-addressed chunk already exists.

### Phase 7: completion and delayed free

As requests finish:

- the receive thread reports which requests finished loading
- the send thread tracks outstanding async store work
- `MooncakeStoreWorker._get_and_clear_finished_sending()` only reports send completion after all outstanding store jobs for a request are done

`MooncakeStoreScheduler.request_finished()` may delay block free when a request still has pending save work. That avoids releasing GPU blocks before the async store path has consumed them.

### Phase 8: preemption and resumption

The scheduler explicitly tracks preempted requests.

When a request is preempted:

- tracker state is removed
- unfinished request bookkeeping is updated
- save completion state is cleared on the worker side

When it resumes:

- a new `RequestTracker` is built
- new block ids are attached
- load/save metadata is rebuilt for the resumed request

So the branch is trying to keep the Mooncake integration compatible with vLLM's scheduler realities, not just the happy path.

## How Lookup and Deduplication Work

The branch's reuse story depends on three pieces fitting together.

### Hash-based addressing

Each chunk is addressed by its block hash, not by request id. That allows:

- the same request to resume later
- a different request with the same prefix to reuse the same KV
- data-parallel engines to share the same external cache entries

### Cross-rank key expansion during lookup

Worker lookup expands the key set across:

- TP ranks
- PP ranks

Then it checks whether each chunk exists everywhere it needs to exist. It returns the earliest missing chunk boundary. That gives the scheduler a prefix length that is safe to treat as externally reusable.

### Stable hashing across DP engines

The scripts README calls out a practical requirement: `PYTHONHASHSEED=0` should be fixed for reliable cross-DP reuse. If each engine seeds prefix hashing differently, identical prompts can generate different hash chains, and Mooncake deduplication becomes ineffective across engines.

## KV Memory Layout Handling

One subtle but important part of the branch is `register_kv_caches()`.

The Mooncake worker cannot assume one fixed KV tensor layout. The code detects layout by inspecting stride and storage size, then derives:

- base addresses to register with Mooncake
- per-block byte lengths for each segment

It supports at least two broad cases:

- blocks-first layout, such as FlashInfer/MLA style tensors
- K/V-outermost layout, such as FlashAttention-style tensors

It also supports a cross-layer KV cache registration path, where a single tensor packs multiple layers together.

Without this layout detection, the store would compute the wrong source/destination addresses and silently corrupt loads/stores.

## Offload-Specific Behavior

This branch grows beyond pure prefix caching by using Mooncake as a larger offload tier.

### CPU-only mode

Mooncake can act as a CPU-backed external KV store. In the scripts this is often labeled `mooncake-mem`.

### CPU + disk mode

When offload is enabled:

- Mooncake uses disk as another storage tier
- vLLM sets Mooncake offload env vars through `setup_vllm_env.sh`
- the receive path becomes careful about staging buffer size
- oversized loads are split into smaller batches

### Pressure backoff

When Mooncake reports `NO_AVAILABLE_HANDLE`, the send thread interprets that as temporary saturation of CPU/disk offload resources. The branch chooses to:

- keep serving requests
- skip additional store batches for affected requests
- resume normal saving later after a successful batch

This is a resilience optimization aimed at real deployment conditions.

## Integration Points Outside the New Connector

The branch also makes a few smaller but meaningful changes around the new core code.

### Factory registration

`KVConnectorFactory` now knows the new connector name, making the feature selectable from config.

### `MultiConnector` support

`MultiConnector.bind_gpu_block_pool()` now forwards block-pool binding to child connectors that implement it. This is a compatibility fix so Mooncake store integration can coexist with multi-connector setups.

### Single-node DP engine indexing

`get_mooncake_dp_engine_index()` centralizes Mooncake's notion of DP rank:

- use `data_parallel_rank_local` when `local_engines_only` is set
- otherwise use `data_parallel_index`

That helper is wired into both the old `MooncakeConnector` and the new store scheduler/worker lookup path. The effect is better support for single-node DP sharing and correct side-channel addressing.

### Cross-layer block preference

`MooncakeStoreConnector.prefer_cross_layer_blocks` is sourced from `enable_cross_layers_blocks` in connector extra config. This lets the connector participate in vLLM's cross-layer KV block mode when enabled.

## Tests Added

The branch adds targeted unit coverage rather than just smoke tests.

`test_mooncake_store_connector.py` covers:

- scheduler vs worker initialization
- rank-0 lookup server creation
- DP/local-engine lookup socket path rules
- worker method delegation
- KV event wrapping and aggregation
- cross-layer block preference

`test_mooncake_store_worker.py` covers:

- store-pressure skip logic for `NO_AVAILABLE_HANDLE`
- disk-offload buffer budgeting
- sub-batch splitting for large loads
- failure handling in load sub-batches
- unsplittable oversized keys
- KV cache layout detection
- cross-layer cache registration

That test coverage maps closely to the risky parts of the branch.

## Operational Story

The new `scripts/mooncake/README.md` makes the intended deployment story pretty clear.

The expected flow is:

1. Install a Mooncake build or wheel.
2. Start the Mooncake master service.
3. Point vLLM to `MOONCAKE_CONFIG_PATH`.
4. Source `setup_vllm_env.sh` for CPU or CPU+disk configurations.
5. Launch vLLM with `MooncakeStoreConnector`.
6. Benchmark either:
   - single-turn random prompts for raw offload overhead
   - multi-turn shared-prefix traffic for reuse/prefix-hit behavior

The benchmark split is useful:

- `benchmark_cpu_offloading.sh` isolates offload overhead because prompts are random and do not reuse prefix KV
- `benchmark_multi_turn.sh` is where Mooncake's external prefix cache can actually pay off

## Commit Progression

Looking at the branch history, the implementation appears to have evolved in stages:

1. Initial connector introduction
2. Fixes for style and kv-cache layout handling
3. Scripts and setup documentation
4. RDMA-oriented config updates
5. Disk offload support
6. Async-overlap enforcement through `get_finished()`
7. Backpressure handling under offload pressure
8. Load-batch splitting for Mooncake stage-buffer limits
9. Single-node DP support
10. Cross-layer block preference support
11. Final README/config polishing

That history lines up with what the code shows: the branch started as basic store integration, then hardened into something more deployment-aware.

## Bottom Line

The branch integrates Mooncake into vLLM by adding a new store-based KV connector, not by lightly modifying the old P2P Mooncake path.

The main technical outcomes are:

- Mooncake becomes an external shared KV memory tier for vLLM
- external prefix-cache lookup is added on the scheduler side
- KV chunks are stored and loaded by content hash
- identical prefixes can be deduplicated and reused
- async load/store is overlapped with compute through `get_finished()`
- CPU and disk offload tiers are supported
- store-pressure handling is added so overload degrades caching instead of crashing
- DP/local-engine sharing and cross-layer block support are wired in
- scripts and tests are added so the feature is runnable and diagnosable

If I had to summarize the branch in one sentence: it teaches vLLM to use Mooncake as a hash-addressed external KV cache service, with enough scheduler state, worker plumbing, and operational guardrails to make that viable for real prefix reuse and offloading.
