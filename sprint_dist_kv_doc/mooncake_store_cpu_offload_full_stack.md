# Mooncake Store CPU Offload Full Stack

本文整理 `vllm/distributed/kv_transfer/kv_connector/v1/mooncake` 中，`MooncakeStoreConnector` 作为 CPU offload 时的完整调用栈，并把关键分支点拆开说明。

关注范围：

- vLLM V1 worker / scheduler 如何生成并消费 Mooncake metadata
- MooncakeStoreConnector 在 worker 侧如何把 load/store 请求下发到 store
- Mooncake store 层内部有哪些分支
- TransferEngine 如何选 transport
- 最底层最终落到什么原语

不重点展开：

- `MooncakeConnector` 的 direct P2P 路径
- SSD offload 的完整后台状态机
- 多机 / 多卡 / Ascend 特化分支

## 全局架构总览

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          vLLM V1 Worker                                 │
│                                                                         │
│   Scheduler                          GPUModelRunner                     │
│     │  build_connector_meta()          │  forward()                     │
│     │  生成 ReqMeta (load/save spec)   │                                │
│     ▼                                  ▼                                │
│   MooncakeStoreConnector                                                │
│     │  get_finished() ─── 真正下发 I/O                                  │
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
│          ├─ LOCAL_MEMCPY ──── std::memcpy  (MC_STORE_MEMCPY=1 时)       │
│          │                                                              │
│          └─ TRANSFER_ENGINE ─── 默认路径                                │
│               │                                                         │
│               ▼                                                         │
│          TransferEngine → MultiTransport                                │
│               │                                                         │
│               ├─ rdma ──── ibv_post_send (默认)                         │
│               ├─ tcp  ──── socket + cudaMemcpy (fallback)               │
│               ├─ nvlink ── cudaMemcpy (NVLink 分支)                     │
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

## 场景结论

对下面这个场景：

- 单机
- 单 GPU
- `MooncakeStoreConnector`
- CPU offload 开启
- disk offload 关闭

默认主路径不是 `cudaMemcpy`。

默认主路径是：

- GPU -> CPU offload: `IBV_WR_RDMA_WRITE` -> `ibv_post_send(...)`
- CPU -> GPU load: `IBV_WR_RDMA_READ` -> `ibv_post_send(...)`

`cudaMemcpy` 只会出现在某些 NVLink transport 里，而不是这条默认 Mooncake store CPU offload 路径里。

另外，Mooncake store 里还有一个本地优化分支 `LOCAL_MEMCPY`，但它底层是 `std::memcpy`，不是 `cudaMemcpy`，而且默认由 `MC_STORE_MEMCPY=0` 关闭；对 vLLM 传入的 GPU 指针，这条分支本来也不是我们要的目标路径。

### 默认物理数据路径

同机 CPU offload 场景下，默认的物理 DMA 路径：

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

关键源码位置：
- GPU MR 注册：`Mooncake/mooncake-transfer-engine/src/transport/rdma_transport/rdma_context.cpp:245-258`
- host MR 注册：`Mooncake/mooncake-transfer-engine/src/transport/rdma_transport/rdma_context.cpp:229-233`
- 同机 loopback：`Mooncake/mooncake-transfer-engine/src/transport/rdma_transport/rdma_endpoint.cpp:161-173`
- RDMA post：`Mooncake/mooncake-transfer-engine/src/transport/rdma_transport/rdma_endpoint.cpp:496-505`

### 默认路径不是什么

为避免误解，明确列出默认路径**不**涉及的机制：

- **不是** CPU 线程 `std::memcpy` — `MC_STORE_MEMCPY` 默认关闭
- **不是** `tcp -> DRAM bounce buffer -> cudaMemcpy` — TCP 是 fallback，不是默认
- **不是** GPU kernel 在 SM 里对 CPU memory 做 `ld/st`
- **不是** 显式的 `NVLink-C2C` 本地 copy 路径

### GB200 上和 NVLink-C2C 的关系

如果你在 GB200 (Grace + Blackwell) 上跑，机器本身支持 C2C：

- `nvidia-smi -q` 显示 `GPU C2C Mode : Enabled`
- `nvidia-smi -q` 显示 `Addressing Mode : ATS`

但从 Mooncake 当前实现看，没有"同机 CPU replica → GPU buffer 时切到 NVLink-C2C copy"的专门分支。默认还是：

- `protocol = "rdma"`
- `RdmaTransport`
- same-node `loopback mode`

> GB200 具备 C2C 能力，但 MooncakeStoreConnector 默认并没有显式使用这条本地 C2C copy 路径。默认的 Grace CPU → 邻近 GPU datapath 是 RNIC/Data Direct 驱动的 RDMA loopback，经 PCIe P2P 把 CPU memory replica 搬到 GPU VRAM。

## 一、先说清楚：CPU KV cache pool 到底是谁创建的

这里最容易混淆的点是，“CPU offload” 不等于 “vLLM 自己创建了一份 CPU KV cache tensor”。

对 `MooncakeStoreConnector` 来说，实际情况是：

- vLLM 侧只显式注册 GPU KV cache
- CPU 侧承接 offload 的不是 vLLM 的 CPU tensor，而是 Mooncake store 自己的 host memory segment
- 这块 host memory segment 组成了本文说的 CPU KV cache pool / memory replica pool

也就是说，这条路径的两端其实是：

- GPU 端：vLLM 的 GPU KV cache tensor
- CPU 端：Mooncake 的 global segment（host memory pool）

而不是：

- GPU 端：vLLM GPU KV cache
- CPU 端：vLLM CPU KV cache

### 1. vLLM 侧到底注册了什么

`MooncakeStoreWorker.register_kv_caches()` 的工作很直接：

1. 遍历 `kv_caches`
2. 取每个 tensor 的 `data_ptr()` 和 `nbytes()`
3. 调 `self.store.register_buffer(base_addr, region_len)`

也就是说，vLLM 注册给 Mooncake 的是 GPU KV cache 对应的 device pointer。

相关代码：

- `vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_worker.py:1174-1257`
- `Mooncake/mooncake-store/src/real_client.cpp:1970-1977`
- `Mooncake/mooncake-store/src/client_service.cpp:2148-2159`

### 2. CPU pool 到底是谁创建的

CPU pool 是 Mooncake store 在 `setup(...)` 阶段按照 `global_segment_size` 自己分配并挂载的。

主线是：

1. vLLM 先从 `MOONCAKE_CONFIG_PATH` 指向的 JSON 读出 `global_segment_size`、`local_buffer_size`、`protocol`、`master_server_address` 等字段
2. `MooncakeStoreWorker.__init__()` 把这些字段组装成 `config_dict`
3. `self.store.setup(config_dict)` 进入 Mooncake
4. `RealClient::setup_internal(...)` 读取 `global_segment_size`
5. 为 global segment 分配 host memory
6. 对每一段调用 `client_->MountSegment(ptr, mapped_size, protocol, seg_location)`
7. `Client::MountSegment(...)` 内部再调用 `transfer_engine_->registerLocalMemory(...)`
8. 这块 segment 被发布到 master / metadata，后续对象的 memory replica 就从这里分配

相关代码：

- `vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_worker.py:160-170`
- `vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_worker.py:1113-1130`
- `Mooncake/mooncake-store/src/real_client.cpp:623-678`
- `Mooncake/mooncake-store/src/real_client.cpp:437-540`
- `Mooncake/mooncake-store/src/client_service.cpp:2039-2098`

所以，“CPU KV cache pool” 的来源不是 vLLM 另开了一套 CPU KV 张量，而是 Mooncake 自己的 global segment。

### 3. 到底哪个参数控制 CPU pool 大小

主参数是：

- `global_segment_size`

它表示当前节点贡献给 Mooncake 的 CPU memory pool 大小。对单机单卡场景，可以近似理解为“这台机器给 Mooncake CPU offload 提供了多大的 host memory 池”。

不要把它和下面这个参数混掉：

- `local_buffer_size`

`local_buffer_size` 是 Mooncake transfer engine 的本地传输缓冲区，不是 CPU KV cache pool 容量。

相关代码：

- `Mooncake/mooncake-store/src/real_client.cpp:641-677`
- `Mooncake/mooncake-store/src/real_client.cpp:420-427`
- `Mooncake/mooncake-store/src/real_client.cpp:486-533`

### 4. vLLM 到底传了哪些环境变量 / 配置

这条链里，最关键的不是某个单独的 “CPU KV cache size env var”，而是：

1. 环境变量 `MOONCAKE_CONFIG_PATH`
2. 这个路径指向的 JSON 配置文件
3. JSON 里的 `global_segment_size`

`setup_vllm_env.sh` 做的事情是：

- 导出 `MOONCAKE_CONFIG_PATH`
- 把 JSON 里的 `global_segment_size` 改成 `--cpu-mem-size <GB>` 指定的值
- 保留 `metadata_server`、`master_server_address`、`protocol`、`device_name`、`local_buffer_size` 等其他字段

脚本和示例配置：

- `vllm/scripts/mooncake/setup_vllm_env.sh`
- `vllm/scripts/mooncake/mooncake_config.json`

其中：

- `--cpu-mem-size 80` 会把 `global_segment_size` 改成 `80GB`
- 这就是 Mooncake CPU pool 的大小来源
- `--disk-size` 只影响 disk offload，不决定 CPU pool 大小

### 5. 一个最简 mental model

可以把 `MooncakeStoreConnector` 想成下面这张图：

```
  vLLM GPU KV cache tensor
      │  register_buffer(device_ptr, size)
      ▼
  Mooncake TransferEngine / transport registry
      │
      │  接口只有 5 个:
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
      │  注意: 这不是 vLLM 分配的 CPU tensor
      │  而是 Mooncake 自己用 global_segment_size
      │  分配的 host memory, 通过 ibv_reg_mr 注册
      ▼
  CPU DRAM (由 Mooncake 管理)
```

所以，如果你在代码里找”vLLM 创建 CPU KV cache tensor”的地方，基本找不到，因为这条路径压根不是这个设计。

## 二、完整主栈：GPU -> CPU offload

这里说的 “GPU -> CPU offload” 指的是：vLLM 把已经算好的 GPU KV cache block 存进 Mooncake 的 memory replica，也就是 CPU memory pool。

```
  GPU → CPU Offload 调用链概览:

  GPUModelRunner                MooncakeStoreWorker           Mooncake C++
  ────────────                  ───────────────────           ──────────
       │                              │                           │
  forward() 完成                      │                           │
       │                              │                           │
  post_forward()                      │                           │
       └─► get_finished() ──────────►│                           │
                                      │                           │
                               录 CUDA event                     │
                               下发到 kv_send_thread              │
                                      │                           │
                               event.synchronize()               │
                               (等 GPU 计算完成)                   │
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
                                                      │         (默认)
                                                 std::memcpy      │
                                                           MultiTransport
                                                             │
                                                      selectTransport
                                                      (by segment protocol)
                                                             │
                                                          rdma (默认)
                                                             │
                                                       IBV_WR_RDMA_WRITE
                                                       ibv_post_send()
```

### 1. vLLM 初始化阶段：先注册 GPU KV cache

1. `GPUModelRunner.initialize_kv_cache()` 分配 KV cache tensor 后，调用 KV transfer group 的 `register_kv_caches(...)`。
   `vllm/vllm/v1/worker/gpu_model_runner.py:6858-6870`
2. `ActiveKVConnector.__init__()` 再次把 `kv_caches_dict` 交给 connector，并设置 host xfer ops。
   `vllm/vllm/v1/worker/gpu/kv_connector.py:48-58`
3. `MooncakeStoreConnector.register_kv_caches()` 调到 `MooncakeStoreWorker.register_kv_caches()`。
   `vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_connector.py:169-171`
4. `MooncakeStoreWorker.register_kv_caches()` 遍历每个 KV tensor，取 `data_ptr()` / `nbytes()`，调用 `self.store.register_buffer(base_addr, region_len)`。
   `vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_worker.py:1174-1257`
5. `RealClient.register_buffer_internal()` 调 `client_->RegisterLocalMemory(buffer, size, kWildcardLocation, false, true)`。
   `Mooncake/mooncake-store/src/real_client.cpp:1970-1977`
6. `Client::RegisterLocalMemory()` 调 `transfer_engine_->registerLocalMemory(...)`。
   `Mooncake/mooncake-store/src/client_service.cpp:2148-2159`
7. `TransferEngineImpl::registerLocalMemory()` 把这段地址注册给所有已安装 transport。
   `Mooncake/mooncake-transfer-engine/src/transfer_engine_impl.cpp:475-495`
8. RDMA transport 在 `registerLocalMemoryInternal()` 里如果 location 是 `*`，会调用 `getMemoryLocation(...)`；对 CUDA device pointer，会先跑 `cudaPointerGetAttributes(...)`，识别成 GPU memory。
   `Mooncake/mooncake-transfer-engine/src/transport/rdma_transport/rdma_transport.cpp:283-299`
   `Mooncake/mooncake-transfer-engine/src/memory_location.cpp:33-52`

这一步的意义是：

- vLLM 交给 Mooncake 的不是抽象 tensor，而是 GPU 虚拟地址
- Mooncake 先把这些 GPU 地址注册到 TransferEngine / transport
- 后续 put/get 不再需要重新找 tensor，只传 `(key, addr, size)`

### 2. scheduler 侧：决定本轮哪些 request 要 load / save

1. scheduler 在每轮 schedule 结束前调用 `connector.build_connector_meta(scheduler_output)`。
   `vllm/vllm/v1/core/sched/scheduler.py:936-954`
2. `MooncakeStoreScheduler.get_num_new_matched_tokens()` 先通过 `LookupKeyClient.lookup(...)` 查询远端 store 里已有多少前缀命中，生成 `LoadSpec`。
   `vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_scheduler.py:61-104`
3. `update_state_after_alloc()` 在 block 分配后，把本地 block id 记进 `_unfinished_requests`，并把 `LoadSpec.can_load` 置真。
   `vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_scheduler.py:106-125`
4. `build_connector_meta()` 遍历：
   - `scheduled_new_reqs`
   - `scheduled_cached_reqs`
   - “有 pending load spec 但本轮没实际调度”的请求
5. 对每个请求，调用 `ReqMeta.from_request_tracker(...)` 生成 `ReqMeta`，其中包含：
   - `req_id`
   - `block_ids`
   - `block_hashes`
   - `load_spec`
   - `can_save`
   - `token_len_chunk`
6. 这些 `ReqMeta` 被打包进 `MooncakeStoreConnectorMetadata`。
   `vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_scheduler.py:144-330`

### 3. worker 侧：本轮 forward 前后

1. `ActiveKVConnector.pre_forward()` 绑定 metadata，然后调用 `start_load_kv(...)`。
   `vllm/vllm/v1/worker/gpu/kv_connector.py:62-76`
2. 对 `MooncakeStoreConnector` 来说，`start_load_kv()` 是 no-op。
   `vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_connector.py:173-176`
3. `ActiveKVConnector.post_forward()` 在 forward 后调用：
   - `wait_for_save()`，这里也是 no-op
   - `get_finished(finished_req_ids)`
   `vllm/vllm/v1/worker/gpu/kv_connector.py:78-95`
4. `MooncakeStoreConnector.get_finished()` 进入 `MooncakeStoreWorker.get_finished(finished_req_ids, metadata)`。
   `vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_connector.py:195-201`
5. `MooncakeStoreWorker.get_finished()`：
   - 把可 load 的请求扔给 `kv_recv_thread`
   - 如果有可 save 的请求，先录一个共享 `torch.cuda.Event()`，再把请求扔给 `kv_send_thread`
   - 收取上一轮或更早完成的 load/store 结果
   `vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_worker.py:1331-1431`

这里是一个非常关键的设计点：

- `start_load_kv()` / `save_kv_layer()` 不真正做 I/O
- 真正的 I/O 下发被推迟到 `get_finished()`
- 目的是让 compute 和 I/O overlap

### 4. 发送线程：从 GPU KV block 写入 Mooncake store

1. `KVCacheStoreSendingThread._handle_request(req_meta)` 收到一个 `ReqMeta`。
   `vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_worker.py:562-760`
2. `token_database.process_tokens(...)` 把 token chunk 变成 `(start, end, key)`。
3. `batch_is_exist(keys)` 做 dedupe，只保留 store 里还不存在的 block。
4. `token_database.prepare_value(...)` 计算每个 block 的 GPU 地址和分片大小。
   `vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_data.py:80-97`
5. 等共享的 `current_event.synchronize()`，保证 GPU 计算已完成，KV 数据稳定可读。
6. 调 `self.store.batch_put_from_multi_buffers(keys, addrs, sizes)`。
   `vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_worker.py:652-707`

### 5. Mooncake store 层：从 batch_put 进入 transfer_submitter

1. `RealClient::batch_put_from_multi_buffers(...)`
2. `RealClient::batch_put_from_multi_buffers_internal(...)`
3. `client_->BatchPut(keys, batched_slices, config)`
   `Mooncake/mooncake-store/src/real_client.cpp:2985-3042`
4. `Client::BatchPut(...)`
   - 构造 `PutOperation`
   - `StartBatchPut(ops, client_cfg)`
   - `SubmitTransfers(ops)`
   - `WaitForTransfers(ops)`
   - `FinalizeBatchPut(ops)`
   `Mooncake/mooncake-store/src/client_service.cpp:1946-1978`
5. `SubmitTransfers(ops)` 对每个 memory replica 调 `transfer_submitter_->submit(replica, slices, TransferRequest::WRITE)`。
   `Mooncake/mooncake-store/src/client_service.cpp:1497-1560`
6. `TransferSubmitter::submit(...)` 进入 memory replica 分支，继续选择是 `LOCAL_MEMCPY` 还是 `TRANSFER_ENGINE`。
   `Mooncake/mooncake-store/src/transfer_task.cpp:438-460`

### 6. Mooncake store 层分支：真正决定走 memcpy 还是 transport

`TransferSubmitter::submit(...)` 的分支树如下：

1. 如果 `replica.is_memory_replica()` 为假：
   走 `submitFileReadOperation(...)`
   这是 disk replica 路径，不是本文主路径。
2. 如果是 memory replica：
   - `validateTransferParams(...)`
   - `selectStrategy(handle, slices)`
3. `selectStrategy(...)`：
   - 若 `MC_STORE_MEMCPY` 未开启，直接返回 `TRANSFER_ENGINE`
   - 若开启且 `isLocalTransfer(handle)` 为真，返回 `LOCAL_MEMCPY`
   - 否则返回 `TRANSFER_ENGINE`
   `Mooncake/mooncake-store/src/transfer_task.cpp:681-696`
4. `LOCAL_MEMCPY` 分支会进入 `submitMemcpyOperation(...)`，底层是 `std::memcpy(...)`。
   `Mooncake/mooncake-store/src/transfer_task.cpp:545-588`
5. `TRANSFER_ENGINE` 分支会进入 `submitTransferEngineOperation(...)`，构造 `TransferRequest`，再交给 `engine_.submitTransfer(...)`。
   `Mooncake/mooncake-store/src/transfer_task.cpp:625-661`

对 vLLM 的 GPU KV cache 来说，默认命中的是：

- `MC_STORE_MEMCPY=0`
- 所以直接走 `TRANSFER_ENGINE`

也就是说，这里不会落到 `std::memcpy`，更不会落到 `cudaMemcpy`。

### 7. TransferEngine：选择具体 transport

1. `submitTransfer(requests)` -> `engine_.submitTransfer(batch_id, requests)`
   `Mooncake/mooncake-store/src/transfer_task.cpp:590-623`
2. `MultiTransport::submitTransfer(...)` 对每个 request 调 `selectTransport(request, transport)`。
   `Mooncake/mooncake-transfer-engine/src/multi_transport.cpp:103-140`
3. `selectTransport(...)` 的依据是：
   - 查 `entry.target_id` 对应的 `target_segment_desc`
   - 读取 `target_segment_desc->protocol`
   - 选出相应 `transport_map_[proto]`
   `Mooncake/mooncake-transfer-engine/src/multi_transport.cpp:351-372`

也就是说，这里不是“看 request 是 READ 还是 WRITE 来选 transport”，而是：

- transport 先由目标 segment 的 `protocol` 决定
- opcode 只决定 transport 内部用 READ 语义还是 WRITE 语义

### 8. protocol 安装分支

TransferEngine 初始化时，会先安装哪些 transport，取决于编译条件和环境：

1. 若是 `USE_MNNVL` / `USE_INTRA_NVLINK`：
   - `MC_INTRANODE_NVLINK` -> `nvlink_intra`
   - `MC_FORCE_MNNVL` 或无 HCA -> `nvlink`
   - 否则 -> `rdma`
2. 否则的通用 CUDA 分支：
   - 如果检测到 HCA 且没 `MC_FORCE_TCP`，安装 `rdma`（或 `barex`）
   - 否则安装 `tcp`
   `Mooncake/mooncake-transfer-engine/src/transfer_engine_impl.cpp:240-323`

所以在常规 x86 + CUDA + 有 RoCE 网卡的环境里，单机单卡 CPU offload 通常就是 `rdma`。

### 9. 最底层：RDMA verbs，而不是 cudaMemcpy

当 protocol 选中 `rdma` 后：

1. `RdmaTransport::submitTransfer(...)`
2. `RdmaTransport::submitTransferTask(...)`
3. `RdmaEndPoint` 里把每个 slice 变成一个 `ibv_send_wr`
4. `opcode == READ` -> `IBV_WR_RDMA_READ`
5. `opcode == WRITE` -> `IBV_WR_RDMA_WRITE`
6. 最终 `ibv_post_send(...)`
   `Mooncake/mooncake-transfer-engine/src/transport/rdma_transport/rdma_transport.cpp:430-530`
   `Mooncake/mooncake-transfer-engine/src/transport/rdma_transport/rdma_endpoint.cpp:488-515`

对本文这条主路径：

- offload store 是 `TransferRequest::WRITE`
- 所以最终是 `IBV_WR_RDMA_WRITE`

## 三、完整主栈：CPU -> GPU load

这里说的 “CPU -> GPU load” 指的是：从 Mooncake memory replica 里把 KV cache 读回 vLLM 的 GPU KV block。

```
  CPU → GPU Load 调用链概览:

  Scheduler                     MooncakeStoreWorker           Mooncake C++
  ─────────                     ───────────────────           ──────────
       │                              │                           │
  build_connector_meta()              │                           │
  生成 ReqMeta                        │                           │
  (load_spec.can_load=True)           │                           │
       │                              │                           │
       └──► metadata ──────────────►│                           │
                                      │                           │
                               下发到 kv_recv_thread              │
                                      │                           │
                               prepare_value()                   │
                               (计算目标 GPU block 地址)          │
                                      │                           │
                               batch_get_into_multi_buffers ───►│
                                                                  │
                                                           BatchGet
                                                             │
                                                           transfer_submitter
                                                             │
                                                        TRANSFER_ENGINE
                                                             │
                                                          rdma (默认)
                                                             │
                                                       IBV_WR_RDMA_READ
                                                       ibv_post_send()
                                                             │
                                                     RNIC DMA: CPU MR → GPU MR
```

### 1. scheduler 侧先生成 load spec

仍然从：

- `get_num_new_matched_tokens()`
- `update_state_after_alloc()`
- `build_connector_meta()`

这三步开始。区别是这次 `ReqMeta.load_spec` 不为空，且 `can_load=True`。

对应代码：

- `mooncake_store_scheduler.py:61-125`
- `mooncake_store_scheduler.py:144-330`

### 2. worker 侧在 `get_finished()` 下发 load

1. `ActiveKVConnector.post_forward()` -> `kv_connector.get_finished(...)`
2. `MooncakeStoreWorker.get_finished(...)`
3. 发现某些 `request.load_spec.can_load` 为真
4. 调 `kv_recv_thread.add_request(request)`
   `vllm/vllm/v1/worker/gpu/kv_connector.py:78-95`
   `vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_worker.py:1359-1379`

### 3. 接收线程计算目标 GPU 地址

1. `KVCacheStoreRecvingThread._handle_request(req_meta)`
2. `token_database.process_tokens(...)` 得到待加载 key
3. `prepare_value(...)` 计算目标 GPU block 地址
4. 调 `self.store.batch_get_into_multi_buffers(batch_keys, batch_addrs, batch_sizes)`
   `vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_worker.py:817-990`
   `vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_data.py:80-97`

### 4. Mooncake store 层：从 batch_get 进入 transfer_submitter

1. `RealClient::batch_get_into_multi_buffers(...)`
2. `RealClient::batch_get_into_multi_buffers_internal(...)`
3. `client_->BatchGet(batch_keys, batch_query_results, batch_slices, prefer_alloc_in_same_node)`
   `Mooncake/mooncake-store/src/real_client.cpp:3045-3195`
4. `Client::BatchGet(...)`
   - 若 `prefer_alloc_in_same_node`，走 `BatchGetWhenPreferSameNode(...)`
   - 否则对每个 key 找 replica，然后调用 `transfer_submitter_->submit(replica, slices, TransferRequest::READ)`
   `Mooncake/mooncake-store/src/client_service.cpp:949-1045`

### 5. store 层分支

与 put 类似，`transfer_submitter_->submit(...)` 仍然有下面这些分支：

1. `memory replica` vs `disk replica`
2. `LOCAL_MEMCPY` vs `TRANSFER_ENGINE`

对本文场景，仍然默认走：

- memory replica
- `TRANSFER_ENGINE`

### 6. TransferEngine 与最底层

1. `submitTransferEngineOperation(...)` 构造 `TransferRequest::READ`
   `Mooncake/mooncake-store/src/transfer_task.cpp:625-661`
2. `MultiTransport::selectTransport(...)` 按目标 segment 的 protocol 选 transport
   `Mooncake/mooncake-transfer-engine/src/multi_transport.cpp:351-372`
3. 如果是 `rdma` transport：
   `opcode == READ` -> `IBV_WR_RDMA_READ`
   `ibv_post_send(...)`
   `Mooncake/mooncake-transfer-engine/src/transport/rdma_transport/rdma_endpoint.cpp:496-515`

所以 load 路径的最底层是：

- `IBV_WR_RDMA_READ`
- 不是 `cudaMemcpy`

## 四、分支树总览

```
  完整分支决策树:

  MooncakeStoreWorker.get_finished()
    │
    ├─ load_spec.can_load ──► kv_recv_thread
    │                              │
    │                        batch_get_into_multi_buffers
    │                              │
    └─ request.can_save ──► kv_send_thread (录 CUDA event)
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
                           且 isLocal?          (默认)
                                  │                 │
                           LOCAL_MEMCPY      TRANSFER_ENGINE
                           std::memcpy             │
                                           MultiTransport
                                           selectTransport()
                                                   │
                            ┌──────────┬───────────┼──────────┐
                            │          │           │          │
                          rdma       tcp       nvlink     nvlink_intra
                         (默认)    (fallback)
                            │          │           │          │
                       ibv_post    socket +    cudaMemcpy  cudaMemcpy
                       _send()    cudaMemcpy
```

### A. vLLM connector 层分支

1. `Scheduler.build_connector_meta(...)`
   - 新请求 `scheduled_new_reqs`
   - 已缓存请求 `scheduled_cached_reqs`
   - 挂起 load spec 但本轮未实际调度的请求
2. `MooncakeStoreConnector`
   - `start_load_kv()` 是 no-op
   - `save_kv_layer()` 是 no-op
   - `wait_for_save()` 是 no-op
   - 真正 I/O 都在 `get_finished()`
3. `MooncakeStoreWorker.get_finished(...)`
   - `load_spec.can_load` -> 下发到 recv thread
   - `request.can_save` -> 录 CUDA event 后下发到 send thread

### B. Mooncake store 层分支

1. `BatchPut(...)`
   - `prefer_alloc_in_same_node` -> `BatchPutWhenPreferSameNode(...)`
   - 否则 -> `StartBatchPut -> SubmitTransfers -> WaitForTransfers -> FinalizeBatchPut`
   `Mooncake/mooncake-store/src/client_service.cpp:1946-1978`
2. `BatchGet(...)`
   - `prefer_alloc_in_same_node` -> `BatchGetWhenPreferSameNode(...)`
   - 否则 -> 普通 batch get
   `Mooncake/mooncake-store/src/client_service.cpp:949-978`
3. `TransferSubmitter::submit(...)`
   - `disk replica` -> `submitFileReadOperation(...)`
   - `memory replica` -> `selectStrategy(...)`
4. `selectStrategy(...)`
   - `MC_STORE_MEMCPY=1` 且 `isLocalTransfer` -> `LOCAL_MEMCPY`
   - 否则 -> `TRANSFER_ENGINE`
5. `LOCAL_MEMCPY`
   - 最底层是 `std::memcpy`
6. `TRANSFER_ENGINE`
   - 交给具体 transport

### C. protocol / transport 层分支

1. 初始化阶段先安装 transport
   - `rdma`
   - `tcp`
   - `nvlink`
   - `nvlink_intra`
   - `barex`
   - 以及其他编译特化 transport
2. 真正提交传输时：
   - 查目标 segment descriptor
   - 读取 `target_segment_desc->protocol`
   - `transport_map_[proto]`
3. 进入 transport 后：
   - `READ` / `WRITE` 只决定 transport 内部的数据方向
   - 不决定 transport 类型

## 五、什么情况下才会真的落到 cudaMemcpy

在 Mooncake 里，显式 `cudaMemcpy(...)` 确实存在，但只在以下 transport 分支：

```
  cudaMemcpy 出现的路径:

  ┌─────────────────────────────────────────────────────────────────┐
  │  1. NVLink transport (GPU↔GPU)                                 │
  │     protocol = "nvlink" 或 "nvlink_intra"                      │
  │     IntraNodeNvlinkTransport::submitTransfer()                 │
  │     NvlinkTransport::submitTransfer()                          │
  │     → cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice)    │
  │                                                                 │
  │  2. TCP transport (fallback)                                    │
  │     protocol = "tcp"                                            │
  │     数据路径:                                                    │
  │       发送端: GPU → cudaMemcpy → DRAM bounce buffer → socket   │
  │       接收端: socket → DRAM bounce buffer → cudaMemcpy → GPU   │
  │     tcp_transport.cpp:121-135, 178-225                         │
  └─────────────────────────────────────────────────────────────────┘
```

- `IntraNodeNvlinkTransport::submitTransfer(...)`
  `Mooncake/mooncake-transfer-engine/src/transport/intranode_nvlink_transport/intranode_nvlink_transport.cpp:198-204`
- `NvlinkTransport::submitTransfer(...)`
  `Mooncake/mooncake-transfer-engine/src/transport/nvlink_transport/nvlink_transport.cpp:226-231`
- `TcpTransport` (fallback 时才触发)
  `Mooncake/mooncake-transfer-engine/src/transport/tcp_transport/tcp_transport.cpp:121-135`
  `Mooncake/mooncake-transfer-engine/src/transport/tcp_transport/tcp_transport.cpp:178-225`

也就是说，只有当 protocol 最终选到了这些 transport，才会在底层看到 `cudaMemcpy(...)`。

对本文的 CPU offload 场景，默认不是这条路。

## 六、为什么默认不是 cudaMemcpy

关键原因有两个：

1. vLLM 给 Mooncake 注册的是 GPU device pointer，不是 host pinned buffer。
2. Mooncake store 默认把 “本地 memcpy 优化” 关掉，避免把 GPU 指针交给 `std::memcpy(...)` 这种 CPU copy 路径。

相关代码：

- `MC_STORE_MEMCPY` 默认关闭
  `Mooncake/mooncake-store/src/transfer_task.cpp:412-435`
- `selectStrategy()` 在关闭时直接走 `TRANSFER_ENGINE`
  `Mooncake/mooncake-store/src/transfer_task.cpp:681-696`
- Mooncake 的脚本也明确写了：不要启用 direct disk write，因为那会在 put 时直接碰 GPU-resident data 而 segfault；disk offload 要走 CPU memory -> SSD 的后台路径，而不是 GPU -> SSD inline。
  `vllm/scripts/mooncake/start_mooncake_master.sh:110-116`

## 七、对“单机、单 GPU、无 disk offload”的最终判断

在这个具体场景里，最常见且最重要的命中分支是：

1. vLLM scheduler 生成 `ReqMeta`
2. worker 在 `get_finished()` 下发 load/store
3. send / recv thread 计算 GPU block 地址
4. Mooncake store 进入 `BatchPut` / `BatchGet`
5. 进入 `transfer_submitter_->submit(...)`
6. `MC_STORE_MEMCPY=0`，所以不走 `LOCAL_MEMCPY`
7. 走 `TRANSFER_ENGINE`
8. TransferEngine 根据目标 segment 的 `protocol` 选 transport
9. 常规有 HCA 的机器上通常选 `rdma`
10. 最终：
    - store: `IBV_WR_RDMA_WRITE` -> `ibv_post_send(...)`
    - load: `IBV_WR_RDMA_READ` -> `ibv_post_send(...)`

所以，如果你的目标是找到 “真正的最底层 copy primitive”，那么这条默认主路径里答案不是 `cudaMemcpy`，而是：

- `ibv_post_send`
- 配合 `IBV_WR_RDMA_WRITE / IBV_WR_RDMA_READ`

而 `cudaMemcpy` 只属于另一类 transport 分支。
