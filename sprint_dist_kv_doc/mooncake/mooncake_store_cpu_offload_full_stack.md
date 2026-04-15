# Mooncake Store CPU Offload Full Stack

This article compiles the complete call stack of `MooncakeStoreConnector` as CPU offload in `vllm/distributed/kv_transfer/kv_connector/v1/mooncake`, and disassembles and explains the key branch points.

Area of concern:

- How vLLM V1 worker/scheduler generates and consumes Mooncake metadata
- How MooncakeStoreConnector sends load/store requests to the store on the worker side
- What are the branches inside the Mooncake store layer?
- How to choose transport for TransferEngine
- What primitive finally falls to the lowest level?

Unfocused expansion:

- direct P2P path for `MooncakeConnector`
- Complete background state machine for SSD offload
- Multi-machine / multi-card / Ascend specialized branch

## Global architecture overview
```
┌─────────────────────────────────────────────────────────────────────────┐
│                          vLLM V1 Worker                                 │
│                                                                         │
│   Scheduler                          GPUModelRunner                     │
│     │  build_connector_meta()          │  forward()                     │
│ │ Generate ReqMeta (load/save spec) │ │
│     ▼                                  ▼                                │
│   MooncakeStoreConnector                                                │
│ │ get_finished() ─── Really issue I/O │
│     ├─ kv_send_thread ─── GPU→CPU offload                               │
│     └─ kv_recv_thread ─── CPU→GPU load                                  │
└─────┬───────────────────────────────────────────────────────────────────┘
      │  register_buffer / batch_put / batch_get
      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     Mooncake Store (C++ layer)                          │
│                                                                         │
│   RealClient                                                            │
│     │                                                                   │
│     ├─ Master (RPC) ─── metadata / allocation / eviction                │
│     │                                                                   │
│     └─ TransferSubmitter                                                │
│          │                                                              │
│ ├─ LOCAL_MEMCPY ──── std::memcpy (when MC_STORE_MEMCPY=1) │
│          │                                                              │
│ └─ TRANSFER_ENGINE ─── Default path │
│               │                                                         │
│               ▼                                                         │
│          TransferEngine → MultiTransport                                │
│               │                                                         │
│ ├─ rdma ──── ibv_post_send (default) │
│               ├─ tcp  ──── socket + cudaMemcpy (fallback)               │
│ ├─ nvlink ── cudaMemcpy (NVLink branch) │
│               └─ ...                                                    │
└─────────────────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        Storage Segments                                  │
│                                                                         │
│   ┌──────────────────┐     ┌──────────────────┐                         │
│   │  GPU VRAM        │     │  CPU Host Memory  │                        │
│   │  (vLLM KV cache) │     │  (global segment) │                        │
│   │  ibv_reg_dmabuf  │     │  ibv_reg_mr       │                        │
│   └──────────────────┘     └──────────────────┘                         │
└─────────────────────────────────────────────────────────────────────────┘
```
## Scene conclusion

For the following scenario:

- Standalone
- Single GPU
- `MooncakeStoreConnector`
- CPU offload enabled
-disk offload off

The default main path is not `cudaMemcpy`.

The default main path is:

- GPU -> CPU offload: `IBV_WR_RDMA_WRITE` -> `ibv_post_send(...)`
- CPU -> GPU load: `IBV_WR_RDMA_READ` -> `ibv_post_send(...)`

`cudaMemcpy` will only appear in certain NVLink transports, not in this default Mooncake store CPU offload path.

In addition, there is a local optimization branch `LOCAL_MEMCPY` in the Mooncake store, but its bottom layer is `std::memcpy`, not `cudaMemcpy`, and it is closed by `MC_STORE_MEMCPY=0` by default; for the GPU pointer passed in by vLLM, this branch is not originally the target path we want.

###Default physical data path

In the same machine CPU offload scenario, the default physical DMA path is:
```
  GPU → CPU offload (IBV_WR_RDMA_WRITE):

    GPU VRAM
      │
      ▼  cuMemGetHandleForAddressRange(DMA_BUF_FD)
    ibv_reg_dmabuf_mr() ──► GPU Memory Region (MR)
      │
      │  IBV_WR_RDMA_WRITE
      ▼
    RNIC / ConnectX DMA engine  (loopback mode, same-node)
      │
      │  PCIe P2P / Data Direct
      ▼
    ibv_reg_mr() ──► host Memory Region (MR)
      │
      ▼
    CPU Host DRAM  (Mooncake global segment)


  CPU → GPU load (IBV_WR_RDMA_READ):

    CPU Host DRAM  (Mooncake global segment)
      │
      ▼
    ibv_reg_mr() ──► host Memory Region (MR)
      │
      │  IBV_WR_RDMA_READ
      ▼
    RNIC / ConnectX DMA engine  (loopback mode, same-node)
      │
      │  PCIe P2P / Data Direct
      ▼
    ibv_reg_dmabuf_mr() ──► GPU Memory Region (MR)
      │
      ▼
    GPU VRAM  (vLLM KV cache tensor)
```
Key source code location:
- GPU MR registration: `Mooncake/mooncake-transfer-engine/src/transport/rdma_transport/rdma_context.cpp:245-258`
- Host MR registration: `Mooncake/mooncake-transfer-engine/src/transport/rdma_transport/rdma_context.cpp:229-233`
- Same machine loopback: `Mooncake/mooncake-transfer-engine/src/transport/rdma_transport/rdma_endpoint.cpp:161-173`
- RDMA post: `Mooncake/mooncake-transfer-engine/src/transport/rdma_transport/rdma_endpoint.cpp:496-505`

### What the default path is not

To avoid misunderstandings, explicitly list the mechanisms that the default path does not involve:

- **not** CPU thread `std::memcpy` — `MC_STORE_MEMCPY` is off by default
- **Not** `tcp -> DRAM bounce buffer -> cudaMemcpy` — TCP is fallback, not default
- **Not** GPU kernel does `ld/st` on CPU memory in SM
- **Not** explicit `NVLink-C2C` local copy path

### Relationship between GB200 and NVLink-C2C

If you are running on GB200 (Grace + Blackwell), the machine itself supports C2C:

- `nvidia-smi -q` shows `GPU C2C Mode: Enabled`
- `nvidia-smi -q` shows `Addressing Mode: ATS`

However, judging from the current implementation of Mooncake, there is no special branch for "switching to NVLink-C2C copy when the same machine CPU replica → GPU buffer". The default is still:

- `protocol = "rdma"`
- `RdmaTransport`
- same-node `loopback mode`

> GB200 has C2C capabilities, but MooncakeStoreConnector does not explicitly use this local C2C copy path by default. The default Grace CPU → adjacent GPU datapath is an RNIC/Data Direct-driven RDMA loopback that moves the CPU memory replica to the GPU VRAM via PCIe P2P.

## 1. Let’s make it clear first: Who created the CPU KV cache pool?

The most confusing point here is that "CPU offload" does not mean "vLLM created a CPU KV cache tensor by itself".

For `MooncakeStoreConnector`, the actual situation is:

- The vLLM side only explicitly registers the GPU KV cache
- The CPU side undertakes offload not vLLM's CPU tensor, but Mooncake store's own host memory segment
- This host memory segment forms the CPU KV cache pool / memory replica pool mentioned in this article

In other words, the two ends of this path are actually:

- GPU side: vLLM’s GPU KV cache tensor
-CPU side: Mooncake’s global segment (host memory pool)

instead of:

- GPU side: vLLM GPU KV cache
- CPU side: vLLM CPU KV cache### 1. What is registered on the vLLM side?

The job of `MooncakeStoreWorker.register_kv_caches()` is straightforward:

1. Traverse `kv_caches`
2. Get `data_ptr()` and `nbytes()` of each tensor
3. Adjust `self.store.register_buffer(base_addr, region_len)`

In other words, what vLLM registers with Mooncake is the device pointer corresponding to the GPU KV cache.

Related code:

- `vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_worker.py:1174-1257`
- `Mooncake/mooncake-store/src/real_client.cpp:1970-1977`
- `Mooncake/mooncake-store/src/client_service.cpp:2148-2159`

### 2. Who created the CPU pool?

The CPU pool is allocated and mounted by Mooncake store itself according to `global_segment_size` in the `setup(...)` phase.

The main line is:

1. vLLM first reads `global_segment_size`, `local_buffer_size`, `protocol`, `master_server_address` and other fields from the JSON pointed to by `MOONCAKE_CONFIG_PATH`
2. `MooncakeStoreWorker.__init__()` assembles these fields into `config_dict`
3. `self.store.setup(config_dict)` Enter Mooncake
4. `RealClient::setup_internal(...)` reads `global_segment_size`
5. Allocate host memory for global segment
6. Call `client_->MountSegment(ptr, mapped_size, protocol, seg_location)` for each segment
7. `Client::MountSegment(...)` internally calls `transfer_engine_->registerLocalMemory(...)`
8. This segment is published to master/metadata, and the memory replica of subsequent objects is allocated from here.

Related code:

- `vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_worker.py:160-170`
- `vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_worker.py:1113-1130`
- `Mooncake/mooncake-store/src/real_client.cpp:623-678`
- `Mooncake/mooncake-store/src/real_client.cpp:437-540`
- `Mooncake/mooncake-store/src/client_service.cpp:2039-2098`

Therefore, the source of "CPU KV cache pool" is not vLLM opening another set of CPU KV tensors, but Mooncake's own global segment.

### 3. Which parameter controls the CPU pool size?

The main parameters are:

- `global_segment_size`It represents the CPU memory pool size contributed by the current node to Mooncake. For a single-machine single-card scenario, it can be roughly understood as "how much host memory pool this machine provides for Mooncake CPU offload."

Do not confuse this with this parameter:

- `local_buffer_size`

`local_buffer_size` is the local transfer buffer of Mooncake transfer engine, not the CPU KV cache pool capacity.

Related code:

- `Mooncake/mooncake-store/src/real_client.cpp:641-677`
- `Mooncake/mooncake-store/src/real_client.cpp:420-427`
- `Mooncake/mooncake-store/src/real_client.cpp:486-533`

### 4. What environment variables/configurations are passed to vLLM?

In this chain, the most critical thing is not a single "CPU KV cache size env var", but:

1. Environment variable `MOONCAKE_CONFIG_PATH`
2. The JSON configuration file pointed to by this path
3. `global_segment_size` in JSON

What `setup_vllm_env.sh` does is:

- export `MOONCAKE_CONFIG_PATH`
- Change `global_segment_size` in JSON to the value specified by `--cpu-mem-size <GB>`
- Reserve `metadata_server`, `master_server_address`, `protocol`, `device_name`, `local_buffer_size` and other fields

Script and example configuration:

- `vllm/scripts/mooncake/setup_vllm_env.sh`
- `vllm/scripts/mooncake/mooncake_config.json`

Among them:

- `--cpu-mem-size 80` will change `global_segment_size` to `80GB`
- This is where the size of the Mooncake CPU pool comes from
- `--disk-size` only affects disk offload and does not determine the CPU pool size

### 5. A minimalist mental model

You can think of `MooncakeStoreConnector` as the picture below:
```
  vLLM GPU KV cache tensor
      │  register_buffer(device_ptr, size)
      ▼
  Mooncake TransferEngine / transport registry
      │
│ There are only 5 interfaces:
      │    register_buffer / unregister_buffer
      │    batch_put_from_multi_buffers
      │    batch_get_into_multi_buffers
      │    batch_is_exist
      │
      │  put/get using (key, gpu_addr, size)
      ▼
  Mooncake object store
      │  objects stored in memory replicas
      ▼
  Mooncake global segment (host memory pool)
      │
│ NOTE: This is not a CPU tensor allocated by vLLM
│ Instead, Mooncake uses global_segment_size itself
│ Allocated host memory, registered through ibv_reg_mr
      ▼
CPU DRAM (managed by Mooncake)
```
Therefore, if you look for the place where "vLLM creates CPU KV cache tensor" in the code, you will basically not find it, because this path is not designed at all.

## 2. Complete main stack: GPU -> CPU offload

The "GPU -> CPU offload" mentioned here refers to: vLLM stores the calculated GPU KV cache block into Mooncake's memory replica, which is the CPU memory pool.
```
GPU → CPU Offload call chain overview:

  GPUModelRunner                MooncakeStoreWorker           Mooncake C++
  ────────────                  ───────────────────           ──────────
       │                              │                           │
forward() completed │ │
       │                              │                           │
  post_forward()                      │                           │
       └─► get_finished() ──────────►│                           │
                                      │                           │
Record CUDA event │
Send to kv_send_thread │
                                      │                           │
                               event.synchronize()               │
(Wait until GPU calculation is completed) │
                                      │                           │
                               batch_put_from_multi_buffers ────►│
                                                                  │
                                                           BatchPut
                                                             │
                                                           SubmitTransfers
                                                             │
                                                           TransferSubmitter
                                                             │
                                                      ┌──────┴──────┐
                                                      │             │
                                                 MC_STORE       TRANSFER
                                                 _MEMCPY=1      _ENGINE
│ (default)
                                                 std::memcpy      │
                                                           MultiTransport
                                                             │
                                                      selectTransport
                                                      (by segment protocol)
                                                             │
rdma (default)
                                                             │
                                                       IBV_WR_RDMA_WRITE
                                                       ibv_post_send()
```
### 1. vLLM initialization phase: first register the GPU KV cache

1. After `GPUModelRunner.initialize_kv_cache()` allocates the KV cache tensor, call `register_kv_caches(...)` of the KV transfer group.
   `vllm/vllm/v1/worker/gpu_model_runner.py:6858-6870`
2. `ActiveKVConnector.__init__()` passes `kv_caches_dict` to the connector again and sets host xfer ops.
   `vllm/vllm/v1/worker/gpu/kv_connector.py:48-58`
3. `MooncakeStoreConnector.register_kv_caches()` is transferred to `MooncakeStoreWorker.register_kv_caches()`.
   `vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_connector.py:169-171`
4. `MooncakeStoreWorker.register_kv_caches()` traverses each KV tensor, takes `data_ptr()` / `nbytes()`, and calls `self.store.register_buffer(base_addr, region_len)`.
   `vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_worker.py:1174-1257`
5. `RealClient.register_buffer_internal()` calls `client_->RegisterLocalMemory(buffer, size, kWildcardLocation, false, true)`.
   `Mooncake/mooncake-store/src/real_client.cpp:1970-1977`
6. `Client::RegisterLocalMemory()` calls `transfer_engine_->registerLocalMemory(...)`.
   `Mooncake/mooncake-store/src/client_service.cpp:2148-2159`
7. `TransferEngineImpl::registerLocalMemory()` registers this address to all installed transports.
   `Mooncake/mooncake-transfer-engine/src/transfer_engine_impl.cpp:475-495`
8. If location is `*` in RDMA transport `registerLocalMemoryInternal()`, `getMemoryLocation(...)` will be called; for CUDA device pointer, `cudaPointerGetAttributes(...)` will be run first to identify it as GPU memory.
   `Mooncake/mooncake-transfer-engine/src/transport/rdma_transport/rdma_transport.cpp:283-299`
   `Mooncake/mooncake-transfer-engine/src/memory_location.cpp:33-52`

The meaning of this step is:

- What vLLM hands to Mooncake is not an abstract tensor, but a GPU virtual address
- Mooncake first registers these GPU addresses to TransferEngine/transport
- Subsequent put/get no longer needs to find the tensor again, only `(key, addr, size)` is passed### 2. Scheduler side: decide which requests to load/save in this round

1. The scheduler calls `connector.build_connector_meta(scheduler_output)` before the end of each round of schedule.
   `vllm/vllm/v1/core/sched/scheduler.py:936-954`
2. `MooncakeStoreScheduler.get_num_new_matched_tokens()` first uses `LookupKeyClient.lookup(...)` to query how many prefix hits there are in the remote store and generate `LoadSpec`.
   `vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_scheduler.py:61-104`
3. `update_state_after_alloc()` After block allocation, record the local block id into `_unfinished_requests` and set `LoadSpec.can_load` to true.
   `vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_scheduler.py:106-125`
4. `build_connector_meta()` traverses:
   - `scheduled_new_reqs`
   - `scheduled_cached_reqs`
   - "There is a pending load spec but there is no actual scheduling in this round" request
5. For each request, call `ReqMeta.from_request_tracker(...)` to generate `ReqMeta`, which contains:
   - `req_id`
   - `block_ids`
   - `block_hashes`
   - `load_spec`
   - `can_save`
   - `token_len_chunk`
6. These `ReqMeta` are packaged into `MooncakeStoreConnectorMetadata`.
   `vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_scheduler.py:144-330`

### 3. Worker side: before and after this round of forward

1. `ActiveKVConnector.pre_forward()` binds metadata, and then calls `start_load_kv(...)`.
   `vllm/vllm/v1/worker/gpu/kv_connector.py:62-76`
2. For `MooncakeStoreConnector`, `start_load_kv()` is no-op.
   `vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_connector.py:173-176`
3. `ActiveKVConnector.post_forward()` is called after forward:
   - `wait_for_save()`, also no-op here
   - `get_finished(finished_req_ids)`
   `vllm/vllm/v1/worker/gpu/kv_connector.py:78-95`
4. `MooncakeStoreConnector.get_finished()` enters `MooncakeStoreWorker.get_finished(finished_req_ids, metadata)`.
   `vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_connector.py:195-201`
5. `MooncakeStoreWorker.get_finished()`:
   - Throw loadable requests to `kv_recv_thread`
   - If there is a request that can be saved, first record a shared `torch.cuda.Event()`, and then throw the request to `kv_send_thread`
   - Collect the load/store results completed in the previous round or earlier
   `vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_worker.py:1331-1431`Here is a very critical design point:

- `start_load_kv()` / `save_kv_layer()` do not actually do I/O
- The actual I/O issuance is deferred to `get_finished()`
- The purpose is to make compute and I/O overlap

### 4. Sending thread: writing to Mooncake store from GPU KV block

1. `KVCacheStoreSendingThread._handle_request(req_meta)` receives a `ReqMeta`.
   `vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_worker.py:562-760`
2. `token_database.process_tokens(...)` changes the token chunk into `(start, end, key)`.
3. `batch_is_exist(keys)` does dedupe and only retains blocks that do not yet exist in the store.
4. `token_database.prepare_value(...)` calculates the GPU address and shard size of each block.
   `vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_data.py:80-97`
5. Wait for the shared `current_event.synchronize()` to ensure that the GPU calculation has been completed and the KV data is stable and readable.
6. Call `self.store.batch_put_from_multi_buffers(keys, addrs, sizes)`.
   `vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_worker.py:652-707`

### 5. Mooncake store layer: enter transfer_submitter from batch_put

1. `RealClient::batch_put_from_multi_buffers(...)`
2. `RealClient::batch_put_from_multi_buffers_internal(...)`
3. `client_->BatchPut(keys, batched_slices, config)`
   `Mooncake/mooncake-store/src/real_client.cpp:2985-3042`
4. `Client::BatchPut(...)`
   - Construct `PutOperation`
   - `StartBatchPut(ops, client_cfg)`
   - `SubmitTransfers(ops)`
   - `WaitForTransfers(ops)`
   - `FinalizeBatchPut(ops)`
   `Mooncake/mooncake-store/src/client_service.cpp:1946-1978`
5. `SubmitTransfers(ops)` calls `transfer_submitter_->submit(replica, slices, TransferRequest::WRITE)` for each memory replica.
   `Mooncake/mooncake-store/src/client_service.cpp:1497-1560`
6. `TransferSubmitter::submit(...)` Enter the memory replica branch and continue to choose `LOCAL_MEMCPY` or `TRANSFER_ENGINE`.
   `Mooncake/mooncake-store/src/transfer_task.cpp:438-460`### 6. Mooncake store layer branch: the real decision to use memcpy or transport

The branch tree of `TransferSubmitter::submit(...)` is as follows:

1. If `replica.is_memory_replica()` is false:
   Go `submitFileReadOperation(...)`
   This is the disk replica path, not the main path of this article.
2. If it is replica memory:
   - `validateTransferParams(...)`
   - `selectStrategy(handle, slices)`
3. `selectStrategy(...)`:
   - If `MC_STORE_MEMCPY` is not enabled, return directly to `TRANSFER_ENGINE`
   - If enabled and `isLocalTransfer(handle)` is true, returns `LOCAL_MEMCPY`
   - Otherwise return `TRANSFER_ENGINE`
   `Mooncake/mooncake-store/src/transfer_task.cpp:681-696`
4. The `LOCAL_MEMCPY` branch will enter `submitMemcpyOperation(...)`, and the bottom layer is `std::memcpy(...)`.
   `Mooncake/mooncake-store/src/transfer_task.cpp:545-588`
5. The `TRANSFER_ENGINE` branch will enter `submitTransferEngineOperation(...)`, construct `TransferRequest`, and then hand it over to `engine_.submitTransfer(...)`.
   `Mooncake/mooncake-store/src/transfer_task.cpp:625-661`

For vLLM's GPU KV cache, the default hits are:

- `MC_STORE_MEMCPY=0`
- So just go to `TRANSFER_ENGINE`

In other words, `std::memcpy` will not fall here, let alone `cudaMemcpy`.

### 7. TransferEngine: Select a specific transport

1. `submitTransfer(requests)` -> `engine_.submitTransfer(batch_id, requests)`
   `Mooncake/mooncake-store/src/transfer_task.cpp:590-623`
2. `MultiTransport::submitTransfer(...)` calls `selectTransport(request, transport)` for each request.
   `Mooncake/mooncake-transfer-engine/src/multi_transport.cpp:103-140`
3. The basis of `selectTransport(...)` is:
   - Check the `target_segment_desc` corresponding to `entry.target_id`
   - Read `target_segment_desc->protocol`
   - Select the corresponding `transport_map_[proto]`
   `Mooncake/mooncake-transfer-engine/src/multi_transport.cpp:351-372`

In other words, here is not "choose the transport based on whether the request is READ or WRITE", but:

- The transport is first determined by the `protocol` of the target segment
- opcode only determines whether the transport uses READ semantics or WRITE semantics internally### 8. protocol installation branch

When TransferEngine is initialized, which transports will be installed first depends on the compilation conditions and environment:

1. If `USE_MNNVL` / `USE_INTRA_NVLINK`:
   - `MC_INTRANODE_NVLINK` -> `nvlink_intra`
   - `MC_FORCE_MNNVL` or no HCA -> `nvlink`
   - else -> `rdma`
2. Otherwise general CUDA branch:
   - If HCA is detected and there is no `MC_FORCE_TCP`, install `rdma` (or `barex`)
   - Otherwise install `tcp`
   `Mooncake/mooncake-transfer-engine/src/transfer_engine_impl.cpp:240-323`

Therefore, in a conventional x86 + CUDA + RoCE network card environment, the stand-alone single-card CPU offload is usually `rdma`.

### 9. The lowest level: RDMA verbs, not cudaMemcpy

When `rdma` is selected for protocol:

1. `RdmaTransport::submitTransfer(...)`
2. `RdmaTransport::submitTransferTask(...)`
3. Convert each slice into an `ibv_send_wr` in `RdmaEndPoint`
4. `opcode == READ` -> `IBV_WR_RDMA_READ`
5. `opcode == WRITE` -> `IBV_WR_RDMA_WRITE`
6. Final `ibv_post_send(...)`
   `Mooncake/mooncake-transfer-engine/src/transport/rdma_transport/rdma_transport.cpp:430-530`
   `Mooncake/mooncake-transfer-engine/src/transport/rdma_transport/rdma_endpoint.cpp:488-515`

The main path for this article:

- offload store is `TransferRequest::WRITE`
- So it ends up being `IBV_WR_RDMA_WRITE`

## 3. Complete main stack: CPU -> GPU load

The "CPU -> GPU load" mentioned here refers to the GPU KV block that reads the KV cache back to the vLLM from the Mooncake memory replica.
```
CPU → GPU Load call chain overview:

  Scheduler                     MooncakeStoreWorker           Mooncake C++
  ─────────                     ───────────────────           ──────────
       │                              │                           │
  build_connector_meta()              │                           │
Generate ReqMeta │ │
  (load_spec.can_load=True)           │                           │
       │                              │                           │
       └──► metadata ──────────────►│                           │
                                      │                           │
Send to kv_recv_thread │
                                      │                           │
                               prepare_value()                   │
(Calculate target GPU block address) │
                                      │                           │
                               batch_get_into_multi_buffers ───►│
                                                                  │
                                                           BatchGet
                                                             │
                                                           transfer_submitter
                                                             │
                                                        TRANSFER_ENGINE
                                                             │
rdma (default)
                                                             │
                                                       IBV_WR_RDMA_READ
                                                       ibv_post_send()
                                                             │
                                                     RNIC DMA: CPU MR → GPU MR
```
### 1. Scheduler generates load spec first

Still from:

- `get_num_new_matched_tokens()`
- `update_state_after_alloc()`
- `build_connector_meta()`

Start with these three steps. The difference is that this time `ReqMeta.load_spec` is not empty and `can_load=True`.

Corresponding code:

- `mooncake_store_scheduler.py:61-125`
- `mooncake_store_scheduler.py:144-330`

### 2. The worker issues load in `get_finished()`

1. `ActiveKVConnector.post_forward()` -> `kv_connector.get_finished(...)`
2. `MooncakeStoreWorker.get_finished(...)`
3. Found some `request.load_spec.can_load` to be true
4. Call `kv_recv_thread.add_request(request)`
   `vllm/vllm/v1/worker/gpu/kv_connector.py:78-95`
   `vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_worker.py:1359-1379`

### 3. The receiving thread calculates the target GPU address

1. `KVCacheStoreRecvingThread._handle_request(req_meta)`
2. `token_database.process_tokens(...)` gets the key to be loaded
3. `prepare_value(...)` calculates the target GPU block address
4. Adjust `self.store.batch_get_into_multi_buffers(batch_keys, batch_addrs, batch_sizes)`
   `vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_worker.py:817-990`
   `vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_data.py:80-97`

### 4. Mooncake store layer: Enter transfer_submitter from batch_get

1. `RealClient::batch_get_into_multi_buffers(...)`
2. `RealClient::batch_get_into_multi_buffers_internal(...)`
3. `client_->BatchGet(batch_keys, batch_query_results, batch_slices, prefer_alloc_in_same_node)`
   `Mooncake/mooncake-store/src/real_client.cpp:3045-3195`
4. `Client::BatchGet(...)`
   - If `prefer_alloc_in_same_node`, do `BatchGetWhenPreferSameNode(...)`
   - Otherwise, find the replica for each key, and then call `transfer_submitter_->submit(replica, slices, TransferRequest::READ)`
   `Mooncake/mooncake-store/src/client_service.cpp:949-1045`### 5. store layer branch

Similar to put, `transfer_submitter_->submit(...)` still has the following branches:

1. `memory replica` vs `disk replica`
2. `LOCAL_MEMCPY` vs `TRANSFER_ENGINE`

For this scenario, the default is still:

-memory replica
- `TRANSFER_ENGINE`

### 6. TransferEngine and the lowest layer

1. `submitTransferEngineOperation(...)` constructs `TransferRequest::READ`
   `Mooncake/mooncake-store/src/transfer_task.cpp:625-661`
2. `MultiTransport::selectTransport(...)` selects transport according to the protocol of the target segment
   `Mooncake/mooncake-transfer-engine/src/multi_transport.cpp:351-372`
3. If it is `rdma` transport:
   `opcode == READ` -> `IBV_WR_RDMA_READ`
   `ibv_post_send(...)`
   `Mooncake/mooncake-transfer-engine/src/transport/rdma_transport/rdma_endpoint.cpp:496-515`

So the lowest level of the load path is:

- `IBV_WR_RDMA_READ`
- not `cudaMemcpy`

## 4. Overview of branch trees
```
Complete branch decision tree:

  MooncakeStoreWorker.get_finished()
    │
    ├─ load_spec.can_load ──► kv_recv_thread
    │                              │
    │                        batch_get_into_multi_buffers
    │                              │
└─ request.can_save ──► kv_send_thread (record CUDA event)
                                   │
                             batch_put_from_multi_buffers
                                   │
                                   ▼
                          TransferSubmitter.submit()
                                   │
                          ┌────────┴────────┐
                          │                 │
                    disk replica?      memory replica?
                          │                 │
                  submitFileRead     selectStrategy()
                                           │
                                  ┌────────┴────────┐
                                  │                 │
                           MC_STORE_MEMCPY=1   MC_STORE_MEMCPY=0
and isLocal? (default)
                                  │                 │
                           LOCAL_MEMCPY      TRANSFER_ENGINE
                           std::memcpy             │
                                           MultiTransport
                                           selectTransport()
                                                   │
                            ┌──────────┬───────────┼──────────┐
                            │          │           │          │
                          rdma       tcp       nvlink     nvlink_intra
(default) (fallback)
                            │          │           │          │
                       ibv_post    socket +    cudaMemcpy  cudaMemcpy
                       _send()    cudaMemcpy
```
### A. vLLM connector layer branch

1. `Scheduler.build_connector_meta(...)`
   - New requests `scheduled_new_reqs`
   - Cached requests `scheduled_cached_reqs`
   - Pending load spec but requests that are not actually scheduled this round
2. `MooncakeStoreConnector`
   - `start_load_kv()` is no-op
   - `save_kv_layer()` is no-op
   - `wait_for_save()` is no-op
   - The real I/O is in `get_finished()`
3. `MooncakeStoreWorker.get_finished(...)`
   - `load_spec.can_load` -> Send to recv thread
   - `request.can_save` -> Record the CUDA event and send it to the send thread

### B. Mooncake store layer branch

1. `BatchPut(...)`
   - `prefer_alloc_in_same_node` -> `BatchPutWhenPreferSameNode(...)`
   - Otherwise -> `StartBatchPut -> SubmitTransfers -> WaitForTransfers -> FinalizeBatchPut`
   `Mooncake/mooncake-store/src/client_service.cpp:1946-1978`
2. `BatchGet(...)`
   - `prefer_alloc_in_same_node` -> `BatchGetWhenPreferSameNode(...)`
   - Otherwise -> normal batch get
   `Mooncake/mooncake-store/src/client_service.cpp:949-978`
3. `TransferSubmitter::submit(...)`
   - `disk replica` -> `submitFileReadOperation(...)`
   - `memory replica` -> `selectStrategy(...)`
4. `selectStrategy(...)`
   - `MC_STORE_MEMCPY=1` and `isLocalTransfer` -> `LOCAL_MEMCPY`
   - Otherwise -> `TRANSFER_ENGINE`
5. `LOCAL_MEMCPY`
   - The lowest level is `std::memcpy`
6. `TRANSFER_ENGINE`
   - Leave it to the specific transport

### C. protocol / transport layer branch

1. Install transport first during the initialization phase.
   - `rdma`
   - `tcp`
   - `nvlink`
   - `nvlink_intra`
   - `barex`
   - and other compiler-specific transports
2. When actually submitting the transfer:
   - Check the target segment descriptor
   - Read `target_segment_desc->protocol`
   - `transport_map_[proto]`
3. After entering transport:
   - `READ` / `WRITE` only determine the data direction inside the transport
   - does not determine transport type

## 5. Under what circumstances will it really fall into cudaMemcpy?

In Mooncake, explicit `cudaMemcpy(...)` does exist, but only in the following transport branches:
```
The path where cudaMemcpy appears:

  ┌─────────────────────────────────────────────────────────────────┐
  │  1. NVLink transport (GPU↔GPU)                                 │
│ protocol = "nvlink" or "nvlink_intra" │
  │     IntraNodeNvlinkTransport::submitTransfer()                 │
  │     NvlinkTransport::submitTransfer()                          │
  │     → cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice)    │
  │                                                                 │
  │  2. TCP transport (fallback)                                    │
  │     protocol = "tcp"                                            │
│ Data path: │
│ Sender: GPU → cudaMemcpy → DRAM bounce buffer → socket │
│ Receiving end: socket → DRAM bounce buffer → cudaMemcpy → GPU │
  │     tcp_transport.cpp:121-135, 178-225                         │
  └─────────────────────────────────────────────────────────────────┘
```
- `IntraNodeNvlinkTransport::submitTransfer(...)`
  `Mooncake/mooncake-transfer-engine/src/transport/intranode_nvlink_transport/intranode_nvlink_transport.cpp:198-204`
- `NvlinkTransport::submitTransfer(...)`
  `Mooncake/mooncake-transfer-engine/src/transport/nvlink_transport/nvlink_transport.cpp:226-231`
- `TcpTransport` (triggered only during fallback)
  `Mooncake/mooncake-transfer-engine/src/transport/tcp_transport/tcp_transport.cpp:121-135`
  `Mooncake/mooncake-transfer-engine/src/transport/tcp_transport/tcp_transport.cpp:178-225`

In other words, only when the protocol finally selects these transports will you see `cudaMemcpy(...)` at the bottom.

For the CPU offload scenario in this article, this is not the default path.

## 6. Why is the default not cudaMemcpy?

There are two key reasons:

1. vLLM registers the GPU device pointer for Mooncake, not the host pinned buffer.
2. Mooncake store turns off "local memcpy optimization" by default to avoid handing over the GPU pointer to CPU copy paths such as `std::memcpy(...)`.

Related code:

- `MC_STORE_MEMCPY` is turned off by default
  `Mooncake/mooncake-store/src/transfer_task.cpp:412-435`
- `selectStrategy()` goes directly to `TRANSFER_ENGINE` on shutdown
  `Mooncake/mooncake-store/src/transfer_task.cpp:681-696`
- Mooncake's script also clearly states: Do not enable direct disk write, because it will directly touch GPU-resident data during put and segfault; disk offload should take the background path of CPU memory -> SSD, not GPU -> SSD inline.
  `vllm/scripts/mooncake/start_mooncake_master.sh:110-116`

## 7. Final judgment on “single machine, single GPU, no disk offload”

In this specific scenario, the most common and important hit branches are:

1. vLLM scheduler generates `ReqMeta`
2. The worker issues load/store in `get_finished()`
3. send / recv thread to calculate GPU block address
4. Mooncake store enters `BatchPut` / `BatchGet`
5. Enter `transfer_submitter_->submit(...)`
6. `MC_STORE_MEMCPY=0`, so don’t go to `LOCAL_MEMCPY`
7. Go to `TRANSFER_ENGINE`
8. TransferEngine selects transport based on the `protocol` of the target segment
9. Usually choose `rdma` on machines with HCA.
10. Final:
    - store: `IBV_WR_RDMA_WRITE` -> `ibv_post_send(...)`
    - load: `IBV_WR_RDMA_READ` -> `ibv_post_send(...)`So, if your goal is to find the "real bottom copy primitive", then the answer in this default main path is not `cudaMemcpy`, but:

- `ibv_post_send`
- In conjunction with `IBV_WR_RDMA_WRITE / IBV_WR_RDMA_READ`

And `cudaMemcpy` just belongs to another type of transport branch.