# Source Walkthrough for PD / Native / Simple / MultiConnector

Update time: 2026-04-11

This article focuses on answering four questions:

1. What are `native offload`, `simple offload` and `MultiConnector` respectively in vLLM v1.
2. What is the relationship between `/vllm/v1/kv_offload/cpu/manager.py` and these connectors.
3. How `MultiConnector` is initialized and how to distribute requests in the P/D disaggregation scenario.
4. How does `lazy_offload` of `simple` work? Is it scanned in every step?

This article only covers the KV connector path of vLLM v1, and does not expand on other implementations such as v0, LMCache, Mooncake, and MoRIIO.

---

## TL;DR

- `native offload` corresponds to `OffloadingConnector`.
- `simple offload` corresponds to `SimpleCPUOffloadConnector`.
- The two are not aliases, nor are they the same implementation with a different name.
- `CPUOffloadingManager` in `/vllm/v1/kv_offload/cpu/manager.py` only serves `OffloadingConnector`, not `SimpleCPUOffloadConnector`.
- The core rules of `MultiConnector` are:
  - `load` selects only the first child connector that "claims a hit".
  - `save` will fan-out all child connectors.
- In the configuration `MultiConnector([NixlConnector, OffloadingConnector])`:
  - `OffloadingConnector` must be initialized.
  - `OffloadingConnector` will definitely participate in store/save.
  - Whether `load` falls into the offload connector depends on whether `NixlConnector` has hit first.
- `lazy_offload` of `SimpleCPUOffloadConnector` is a strategy of "incrementally scan the GPU free queue frontier every step".
  - Store planning will be done at each step.
  - But not every step scans the entire free queue from the beginning.

---

## 0. Key file index

If you just want to quickly trace the source code a second time, look at these files first:

- Top-level configuration mapping
  - `vllm/config/cache.py`
  - `vllm/config/vllm.py`
- connector factory and registration
  - `vllm/distributed/kv_transfer/kv_connector/factory.py`
  - `vllm/distributed/kv_transfer/kv_transfer_state.py`
- `MultiConnector`
  - `vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py`
- native offload
  - `vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py`
  - `vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py`
  - `vllm/distributed/kv_transfer/kv_connector/v1/offloading/worker.py`
  - `vllm/v1/kv_offload/factory.py`
  - `vllm/v1/kv_offload/cpu/spec.py`
  - `vllm/v1/kv_offload/cpu/manager.py`
  - `vllm/v1/kv_offload/worker/cpu_gpu.py`
- simple offload
  - `vllm/distributed/kv_transfer/kv_connector/v1/simple_cpu_offload_connector.py`
  - `vllm/v1/simple_kv_offload/manager.py`
  - `vllm/v1/simple_kv_offload/worker.py`
- scheduler/worker access point
  - `vllm/v1/core/sched/scheduler.py`
  - `vllm/v1/worker/gpu/kv_connector.py`
  - `vllm/v1/worker/gpu/model_runner.py`-GPU free queue/BlockPool
  - `vllm/v1/core/block_pool.py`
  - `vllm/v1/core/kv_cache_utils.py`

---

## 1. Configure entry and default behavior

### 1.1 Non-P/D separation scenario

If you are using a top-level cache configuration:```bash
vllm serve <model> \
  --kv-offloading-size 80 \
  --kv-offloading-backend native
```
Then the `native` path will be taken by default.

Corresponding source code logic:

- `CacheConfig.kv_offloading_backend` defaults to `"native"`.
- offloading is only really enabled if `kv_offloading_size` is set.
- `VllmConfig._post_init_kv_transfer_config()` will translate this cache configuration into `kv_transfer_config.kv_connector`.

The simplified mapping rules are:```python
if kv_offloading_size is None:
    offload_disabled()
elif kv_offloading_backend == "native":
    if VLLM_USE_SIMPLE_KV_OFFLOAD:
        kv_connector = "SimpleCPUOffloadConnector"
    else:
        kv_connector = "OffloadingConnector"
elif kv_offloading_backend == "lmcache":
    kv_connector = "LMCacheConnectorV1"
```
So:

- Default native = `OffloadingConnector`
- `native + VLLM_USE_SIMPLE_KV_OFFLOAD=1` = `SimpleCPUOffloadConnector`

### 1.2 P/D separation scene

If you have written `--kv-transfer-config` by hand, especially:```json
{
  "kv_connector": "MultiConnector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "connectors": [...]
  }
}
```
Then you should write the child connector you want to use directly into the `connectors` list instead of relying on the "default native mapping".

The reason is:

- `--kv-transfer-config` directly determines the top-level connector.
- `KVConnectorFactory.create_connector()` will instantiate the connector according to `kv_transfer_config.kv_connector`.
- `MultiConnector` will continue to instantiate its child connectors.

For P/D, the most common way to write it is:```text
prefill: MultiConnector(NixlConnector, OffloadingConnector)
decode : NixlConnector
```
or:```text
prefill: MultiConnector(NixlConnector, SimpleCPUOffloadConnector)
decode : NixlConnector
```
### 1.3 An important note

If you have handwritten in the P/D scene:```json
{"kv_connector":"MultiConnector", ...}
```
Then it is best not to use additional:

- `--kv-offloading-size`
- `--kv-offloading-backend`

The reason is not "theoretically incompatible", but that they belong to two sets of entries on the code path:

- `--kv-transfer-config` directly specifies the top-level connector
- `kv_offloading_size/backend` will overwrite the top-level `kv_transfer_config.kv_connector` again in `VllmConfig._post_init_kv_transfer_config()`

Mixing can therefore significantly increase ambiguity when reading and maintaining configurations.

The rest of this article will be understood according to the following conventions:

- Non-P/D: use `kv_offloading_size/backend`
- P/D + `MultiConnector`: write only `kv-transfer-config`

---

## 2. Positioning of the three main lines```mermaid
flowchart TB
    CacheCfg["CacheConfig\nkv_offloading_size / kv_offloading_backend"] --> PostInit["VllmConfig._post_init_kv_transfer_config()"]
    PostInit -->|native| OffConn["OffloadingConnector"]
    PostInit -->|native + VLLM_USE_SIMPLE_KV_OFFLOAD=1| SimpleConn["SimpleCPUOffloadConnector"]

    OffConn --> SpecFactory["OffloadingSpecFactory"]
    SpecFactory --> CPUSpec["CPUOffloadingSpec"]
    CPUSpec -->|scheduler side| CPUManager["CPUOffloadingManager\nv1/kv_offload/cpu/manager.py"]
    CPUSpec -->|worker side| Handlers["CpuGpuOffloadingHandlers"]
    Handlers --> OffWorker["OffloadingWorker"]

    SimpleConn --> SimpleSched["SimpleCPUOffloadScheduler"]
    SimpleConn --> SimpleWorker["SimpleCPUOffloadWorker"]

    Multi["MultiConnector"] --> Nixl["NixlConnector"]
    Multi --> OffConn
    Multi --> SimpleConn
```
The essential differences between the three main lines:

- `OffloadingConnector`
  - A more general offload framework.
  - There is `OffloadingManager` abstraction on the scheduler side.
  - `CPUOffloadingManager` is the CPU backend manager on this chain.
- `SimpleCPUOffloadConnector`
  - A completely independent set of "lightweight CPU offload" implementation.
  - No need for `OffloadingSpec` or `CPUOffloadingManager`.
  - Maintain CPU block pool, GPU free queue scanning, and DMA backend by yourself.
- `MultiConnector`
  - Not an offload backend.
  - It is just a wrapper used to combine multiple connectors.

---

## 3. What does `/vllm/v1/kv_offload/cpu/manager.py` care about?

### 3.1 What it is not

`CPUOffloadingManager` is not:

- top level connector
- worker side copy engine
- Part of `MultiConnector`
- Internal components of `SimpleCPUOffloadConnector`

### 3.2 What is it

`CPUOffloadingManager` is a CPU implementation of `OffloadingManager`, running on the **scheduler side**.

Its responsibilities are very clear:

- Use block hash to track "which logical blocks have been offloaded to the CPU"
- Manage CPU offload capacity
- Determine whether eviction is required when storing
- Determine which CPU block IDs are returned during load/store
- Maintain cache policies such as LRU / ARC
- Generate offload event

But it doesn't do actual data copying.

---

## 4. The relationship between `CPUOffloadingManager` and `OffloadingConnector`

### 4.1 Relationship diagram```mermaid
flowchart LR
    OffConn["OffloadingConnector"] --> SpecFactory["OffloadingSpecFactory.create_spec()"]
    SpecFactory --> CPUSpec["CPUOffloadingSpec"]
    CPUSpec -->|get_manager()| CPUManager["CPUOffloadingManager"]
    CPUSpec -->|get_handlers()| CpuGpuHandlers["CpuGpuOffloadingHandlers"]
    CpuGpuHandlers --> OffWorker["OffloadingWorker"]
```
### 4.2 Key links

`OffloadingConnector.__init__()` will first create a `spec`:```python
spec = OffloadingSpecFactory.create_spec(vllm_config, kv_cache_config)
```
The default `spec_name` is `CPUOffloadingSpec`, so in the CPU offload scenario:```python
spec = CPUOffloadingSpec(...)
```
Then:

- Create `OffloadingConnectorScheduler(spec)` on the scheduler side
- Create `OffloadingConnectorWorker(spec)` on the worker side

In other words, `CPUOffloadingSpec` is the "shared configuration and assembly layer" of native offload.

### 4.3 The precise location of `CPUOffloadingManager`

In `CPUOffloadingSpec.get_manager()`:```python
self._manager = CPUOffloadingManager(
    block_size=offloaded_block_size,
    num_blocks=self.num_blocks,
    cache_policy=self.eviction_policy,
    enable_events=enable_events,
)
```
So `CPUOffloadingManager` is created by `CPUOffloadingSpec` and used by `OffloadingConnectorScheduler`.

The relationship can be written as:```text
OffloadingConnector
  -> OffloadingConnectorScheduler
      -> OffloadingManager interface
          -> CPUOffloadingManager
```
### 4.4 Why is it only on the scheduler side?

Because it only cares about "whether the logical block is on the CPU, whether it can be loaded, and whether it should be eliminated", regardless of tensor copy.

The worker side is actually responsible for copying:```text
CPUOffloadingSpec.get_handlers()
  -> CpuGpuOffloadingHandlers
  -> SingleDirectionOffloadingHandler(GPU->CPU / CPU->GPU)
  -> OffloadingWorker
```
Therefore, the design of the native path is a classic two-stage formula:

- scheduler: calculate address, calculate hit, calculate capacity, calculate eviction
- Worker: perform asynchronous transmission according to spec

---

## 5. Responsibility boundaries of `CPUOffloadingManager`

It can be understood as the "CPU version of prefix-cache block manager", but it manages offloaded medium rather than GPU pool.

### 5.1 The interfaces it directly exposes

`OffloadingManager` abstractly defines these primitives:

- `lookup(block_hashes)`
- `prepare_load(block_hashes)`
- `touch(block_hashes)`
- `complete_load(block_hashes)`
- `prepare_store(block_hashes)`
- `complete_store(block_hashes)`

Under the native path, the CPU implementation of these primitives is `CPUOffloadingManager`.

### 5.2 What it returns

What it returns is not a tensor, nor a DMA command, but a `CPULoadStoreSpec`.

`CPULoadStoreSpec` is essentially CPU block IDs:```python
class CPULoadStoreSpec(BlockIDsLoadStoreSpec):
    @staticmethod
    def medium() -> str:
        return "CPU"
```
Then on the worker side:

- `CPULoadStoreSpec`
- `GPULoadStoreSpec`

Leave it to `CpuGpuOffloadingHandlers` to do the real CPU<->GPU copy.

### 5.3 The nature of its store decision

`prepare_store()` does three things:

1. Filter block hashes that already exist in the CPU cache
2. If there is insufficient space, do eviction according to the strategy
3. Assign CPU block IDs to "the block that needs to be written to the CPU this time"

The pseudocode is as follows:```python
def prepare_store(block_hashes):
    to_store = [bh for bh in block_hashes if not exists_in_cpu_cache(bh)]

    if not to_store:
        return empty_store_output()

    need_evict = len(to_store) - num_free_cpu_blocks()
    if need_evict > 0:
        evicted = policy.evict(need_evict, protected=set(block_hashes))
        if evicted is None:
            return None
        free_evicted_blocks(evicted)

    cpu_blocks = allocate_blocks_for(to_store)
    policy.insert(to_store, cpu_blocks)
    return PrepareStoreOutput(
        block_hashes_to_store=to_store,
        store_spec=CPULoadStoreSpec(cpu_blocks),
        block_hashes_evicted=evicted_hashes,
    )
```
It's all "logical addressing + capacity management", no copy.

---

## 6. `SimpleCPUOffloadConnector` Why not use `CPUOffloadingManager`

Because the simple path implements a complete set of scheduling and block management logic:

- scheduler side: `SimpleCPUOffloadScheduler`
- worker side: `SimpleCPUOffloadWorker`

Key differences:

- The native path is managed by `OffloadingManager` abstraction for block hash / medium spec
- The simple path is done directly by `KVCacheCoordinator + BlockPool + DmaCopyBackend`

Comparison table:

| dimensions | native/OffloadingConnector | simple/SimpleCPUOffloadConnector |
| --- | --- | --- |
| scheduler core status | `CPUOffloadingManager` | `cpu_coordinator + cpu_block_pool` |
| worker copy framework | `OffloadingWorker + OffloadingHandlers` | `DmaCopyBackend` |
| Whether to use `OffloadingSpec` | Yes | No |
| store strategy | per-request / per-hash | eager or lazy |
| Whether to use `CPUOffloadingManager` | Yes | No |

This is why `/vllm/v1/kv_offload/cpu/manager.py` is not directly related to `SimpleCPUOffloadConnector`.

---

## 7. Initialization logic of `MultiConnector`

### 7.1 Top-level principles

`MultiConnector` is a wrapper that instantiates the `connectors` list in the configuration into child connectors one by one.

Key rules:

- The top-level `kv_connector` is a `MultiConnector`
- The sub-connector is configured in `kv_connector_extra_config.connectors`
- Each child connector will get:
  - own child `KVTransferConfig`
  - Same `role`
  - Same `kv_cache_config`

### 7.2 Initialization flow chart```mermaid
sequenceDiagram
    participant S as Scheduler process
    participant W as Worker process

    S->>S: KVConnectorFactory.create_connector(role=SCHEDULER)
    S->>S: create MultiConnector
    S->>S: for child in connectors: instantiate child scheduler connectors

    W->>W: ensure_kv_transfer_initialized()
    W->>W: KVConnectorFactory.create_connector(role=WORKER)
    W->>W: create MultiConnector
    W->>W: for child in connectors: instantiate child worker connectors

    W->>W: GPUModelRunner.initialize_kv_cache()
    W->>W: ActiveKVConnector(register_kv_caches on all child connectors)
```
### 7.3 Scheduler side initialization

In `Scheduler.__init__()`:```python
self.connector = KVConnectorFactory.create_connector(
    config=self.vllm_config,
    role=KVConnectorRole.SCHEDULER,
    kv_cache_config=self.kv_cache_config,
)
```
If the top level is a `MultiConnector`, it will be in its own `__init__()`:```python
for connector_cls, temp_config in self._get_connector_classes_and_configs(vllm_config):
    self._connectors.append(connector_cls(temp_config, role, kv_cache_config))
```
`_get_connector_classes_and_configs()` will first turn each child's JSON configuration into a temporary `VllmConfig.kv_transfer_config`, and then hand it to `KVConnectorFactory` to get the class.

pseudocode:```python
def multiconnector_init(vllm_config, role, kv_cache_config):
    self._connectors = []
    for child_cfg in vllm_config.kv_transfer_config.extra["connectors"]:
        temp_config = shallow_copy(vllm_config)
        temp_config.kv_transfer_config = KVTransferConfig(**child_cfg)
        child_cls = KVConnectorFactory.get_connector_class(temp_config.kv_transfer_config)
        child = child_cls(temp_config, role, kv_cache_config)
        self._connectors.append(child)
```
### 7.4 Worker side initialization

The worker side is not built in the scheduler, but in:```python
ensure_kv_transfer_initialized(self.vllm_config, kv_cache_config)
```
The same goes internally:```python
_KV_CONNECTOR_AGENT = KVConnectorFactory.create_connector(
    config=vllm_config,
    role=KVConnectorRole.WORKER,
    kv_cache_config=kv_cache_config,
)
```
If the top level is still a `MultiConnector`, it will instantiate a copy of the worker-side child connectors again.

So you end up with two symmetrical trees:

- scheduler process: `MultiConnector(SCHEDULER)` -> child scheduler connectors
- worker process: `MultiConnector(WORKER)` -> child worker connectors

### 7.5 KV cache registration

After the worker side `GPUModelRunner` initializes the KV cache, it will be called through `ActiveKVConnector`:```python
self.kv_connector.register_kv_caches(kv_caches_dict)
```
And `MultiConnector.register_kv_caches()` will broadcast this call to all child connectors:```python
def register_kv_caches(self, kv_caches):
    for c in self._connectors:
        c.register_kv_caches(kv_caches)
```
That's why under `MultiConnector(Nixl, OffloadingConnector)`:

- `NixlConnector` will get KV cache registration
- `OffloadingConnector` will also get KV cache registration

Both children will actually be initialized.

---

## 8. Runtime distribution logic of `MultiConnector`

### 8.1 One-sentence version of the rules

- `load`: selects the first child connector claiming a hit
- `save`: all child connectors will participate

### 8.2 Timing diagram```mermaid
sequenceDiagram
    participant Sch as Scheduler
    participant MCs as MultiConnector(scheduler)
    participant W as Worker/ActiveKVConnector
    participant MCw as MultiConnector(worker)

    Sch->>MCs: get_num_new_matched_tokens(req, num_computed_tokens)
    MCs->>MCs: query child connectors in order
    Note over MCs: first child with hit wins load path

    Sch->>MCs: update_state_after_alloc(req, blocks, num_external_tokens)
    Note over MCs: chosen child gets real blocks; others get empty blocks

    Sch->>MCs: build_connector_meta(step)
    Note over MCs: every child builds its own metadata

    W->>MCw: bind_connector_metadata(meta)
    W->>MCw: handle_preemptions(meta)
    W->>MCw: start_load_kv()
    W->>MCw: wait_for_save()
    W->>MCw: get_finished()
    MCw-->>Sch: KVConnectorOutput

    Sch->>MCs: update_connector_output(output)
    Note over MCs: save completion is merged back to all children
```
### 8.3 `load` Why is "the first one hit first"

The core logic of `MultiConnector.get_num_new_matched_tokens()` is:```python
to_return = (0, False)
for i, c in enumerate(self._connectors):
    toks, load_async = c.get_num_new_matched_tokens(request, num_computed_tokens)
    if toks is None:
        return (None, False)
    if to_return[0] == 0 and toks > 0:
        self._requests_to_connector[request.request_id] = i
        to_return = (toks, load_async)
return to_return
```
So:

- child order matters
- Once the first child is able to load, subsequent children will not be selected as the load source even if they are also hit.

### 8.4 `update_state_after_alloc()` How to make only one child really load

`MultiConnector.update_state_after_alloc()` will only pass the real `blocks` to the selected child.

Other children get `empty_blocks` and `0` external tokens:```python
chosen = requests_to_connector.get(req_id, -1)
for i, c in enumerate(children):
    if i == chosen:
        c.update_state_after_alloc(request, blocks, num_external_tokens)
    else:
        c.update_state_after_alloc(request, empty_blocks, 0)
```
So:

- load path has only one child, so it works really well
- This is not the case with save path. The metadata of save is that each child will be built independently.

### 8.5 Why does `save` fan-out to all children?

`MultiConnector.build_connector_meta()` is:```python
metadata = tuple(c.build_connector_meta(scheduler_output) for c in self._connectors)
```
So each child has the opportunity to generate its own store / load metadata in this step.

This means:

- `NixlConnector` can prepare its own send metadata
- `OffloadingConnector` or `SimpleCPUOffloadConnector` can also prepare its own offload store metadata

So common configuration:```text
MultiConnector(
  NixlConnector,
  OffloadingConnector
)
```
The real semantics of is not "choose one of the two", but:

- load looks at Nixl first
- save writes Nixl and OffloadingConnector simultaneously

---

## 9. Core mechanism of `simple lazy_offload`

### 9.1 Say something first

`lazy_offload` does not immediately save the "just calculated block" to the CPU upon request.

What it does is:

- Look at the GPU free queue frontier at each step
- Find free prefix blocks that are likely to be reused or evicted soon
- If there are no corresponding copies in the CPU, add them to the CPU

### 9.2 Dependent data structures

It relies on three things:

- GPU `BlockPool.free_block_queue`
- CPU `cpu_block_pool`
- `_cursor`

The semantics of `free_block_queue` are very critical:

- The head of the team is LRU and the free block that will be taken away by the allocator at the earliest
- The block will be removed from the free queue when it is touched again
- After the block is released, it will be appended back to the end of the queue according to the eviction order.

So lazy store is essentially patrolling the "eviction frontier".

### 9.3 When to execute

Each scheduler step will be executed when `build_connector_meta()`:```python
store_gpu, store_cpu, store_req_ids = self.prepare_store_specs(scheduler_output)
```
If it is lazy mode, enter:```python
self._prepare_lazy_store_specs()
```
So the answer is:

- Each step will do lazy store planning
- But not full rescan of free queue at every step

### 9.4 How much does it scan?

The upper limit of scanning is determined by two conditions:

- `covered < _target_free`
- `len(gpu_ids) < num_cpu_free`

where `_target_free` is a "target safe window":

- Full attention: press `max_num_batched_tokens / block_size`
- Sliding window: by window size
- Mamba: give fixed reserve
-Finally multiplied by an additional water level coefficient

So a scan will only try to cover the "safe window ahead", not the entire free queue.

### 9.5 Why not scan from the beginning every step of the way?

Because it has `_cursor`.

Starting point rules:

- cursor is invalid or empty: start from the head of the queue
- Otherwise: start with `cursor.next_free_block`

The cursor will only fall back to the head when it fails, for example:

- The block pointed to by cursor is no longer free
- or its `ref_cnt > 0`

### 9.6 Core pseudocode```python
def prepare_lazy_store_specs():
    if gpu_pool is None or target_free <= 0:
        return [], [], []

    if cursor is stale:
        cursor = None

    node = head.next if cursor is None else cursor.next
    covered = 0
    gpu_ids = []
    block_hashes = []

    while (
        node is not tail
        and covered < target_free
        and len(gpu_ids) < num_cpu_free
    ):
        last_visited = node
        bhash = node.block_hash

        if bhash exists and not null and bhash not already in cpu cache:
            gpu_ids.append(node.block_id)
            block_hashes.append(bhash)

        covered += 1
        node = node.next

    cursor = last_visited

    if gpu_ids:
        cpu_blocks = cpu_pool.get_new_blocks(len(gpu_ids))
        stamp_block_hash(cpu_blocks, block_hashes)
        gpu_pool.touch(gpu_ids)
        return gpu_ids, cpu_ids, []
    else:
        return [], [], []
```
### 9.7 A common misunderstanding

"Scan every step" does not mean "traverse all GPU blocks every step".

A more accurate statement is:

- Each step will continue an **incremental inspection** from where it stopped last time.
- The inspection object is the free block near the eviction frontier
- Only the hit block will be taken offload

---

## 10. Key differences between native and simple

### 10.1 Architectural differences```mermaid
flowchart LR
    subgraph Native["native / OffloadingConnector"]
        N1["OffloadingConnectorScheduler"] --> N2["CPUOffloadingManager"]
        N3["OffloadingConnectorWorker"] --> N4["OffloadingWorker"]
        N4 --> N5["CpuGpuOffloadingHandlers"]
    end

    subgraph Simple["simple / SimpleCPUOffloadConnector"]
        S1["SimpleCPUOffloadScheduler"] --> S2["cpu_coordinator + cpu_block_pool"]
        S3["SimpleCPUOffloadWorker"] --> S4["DmaCopyBackend"]
    end
```
### 10.2 Differences in scheduling ideas

- native
  - Centered on block hash / offloaded medium
  - scheduler does `lookup / prepare_load / prepare_store` through `CPUOffloadingManager`
  - worker executes copy according to medium spec through `OffloadingWorker`

-simple
  - Centered on `BlockPool / KVCacheCoordinator / GPU free queue`
  - The scheduler maintains CPU block mapping and load/store events by itself
  - worker uses more direct DMA copy backend

### 10.3 Why both can be put into `MultiConnector`

Because they all implement the same `KVConnectorBase_V1` interface.

So `MultiConnector` doesn't care about child which is:

- `OffloadingConnector`
- `SimpleCPUOffloadConnector`
- `NixlConnector`

It only cares that these children can answer:

- `get_num_new_matched_tokens`
- `update_state_after_alloc`
- `build_connector_meta`
- `update_connector_output`
- `request_finished`

---

## 11. Go back to your two copies of YAML

### 11.1 native.yaml

The prefill side is:```text
MultiConnector(
  NixlConnector,
  OffloadingConnector
)
```
The decode side is:```text
NixlConnector
```
Therefore:

- `OffloadingConnector` will be instantiated
- `CPUOffloadingManager` will also be instantiated
- prefill's save will write Nixl and OffloadingConnector
- decode does not participate in CPU offload

### 11.2 simplecpu.yaml

The prefill side is:```text
MultiConnector(
  NixlConnector,
  SimpleCPUOffloadConnector(lazy_offload=true)
)
```
The decode side still only has:```text
NixlConnector
```
Therefore:

- `SimpleCPUOffloadConnector` will be instantiated
- But `CPUOffloadingManager` will not be used
- prefill's save will write Nixl and SimpleCPUOffloadConnector
- decode does not participate in CPU offload

### 11.3 One of the most easily misjudged points

In these two YAMLs, the child order of prefill is:```text
NixlConnector first
Offload connector at the back
```
This means:

-CPU offload child will definitely store
- But the load is not guaranteed to fall to the CPU offload child
- Because the load of `MultiConnector` is "first hit first"

So a more accurate conclusion is:

- "native/simple connector is indeed used" is correct for save
- For load, it is also necessary to consider whether `NixlConnector` hits first

---

## 12. Practical suggestions

- If your goal is "verify that native/simple is indeed initialized and participates in save", the current configuration is sufficient.
- If your goal is to "strictly compare native/simple load behavior", the current `MultiConnector` order is not pure enough.
- If you want the offload connector to bear the load first, you can change the child order to:```text
MultiConnector(
  OffloadingConnector or SimpleCPUOffloadConnector,
  NixlConnector
)
```
Or temporarily remove `NixlConnector` in a separate experiment.

---

## 13. The last sentence to summarize

The three most memorable sentences:

- `CPUOffloadingManager` only belongs to the native path, not the simple path.
- `MultiConnector` is not an offload backend, it is just a "load select, save broadcast" combiner.
- `simple lazy_offload` is "incremental inspection of the eviction frontier every step", not "full scan of all GPU blocks every step".