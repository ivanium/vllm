# LMCache 接入 vLLM 以替代 `MooncakeStoreConnector` 调研

## 结论先行

如果你的目标是：

- 继续保留 Mooncake 作为远端 KV cache 池
- 但希望 vLLM 侧不要直接绑死 `MooncakeStoreConnector`
- 并且后续能平滑切换到别的远端后端，或者增加本地 CPU / 本地盘 / 压缩 / 控制面能力

那么更推荐的路线是：

`vLLM -> LMCacheConnectorV1 -> LMCache -> Mooncake Store`

而不是：

`vLLM -> MooncakeStoreConnector -> Mooncake Store`

一句话概括：

- 直接 `MooncakeStoreConnector`：Mooncake 既是远端存储池，也是 vLLM 直接对接的 KV connector。
- 走 `LMCacheConnectorV1`：LMCache 成为 vLLM 的 KV cache 管理层，Mooncake 退到 LMCache 背后，充当远端存储 / 远端 lookup / RDMA 数据面。

## 现在其实有两条完全不同的接法

### 路线 A：vLLM 直接接 Mooncake

本地代码里，vLLM 同时注册了 `LMCacheConnectorV1` 和 `MooncakeStoreConnector`，说明它们是并列方案，而不是一回事：

- `LMCacheConnectorV1`：[vllm/vllm/distributed/kv_transfer/kv_connector/factory.py](/home/aoshen/setup_new_cluster/vllm/vllm/distributed/kv_transfer/kv_connector/factory.py#L164)
- `MooncakeStoreConnector`：[vllm/vllm/distributed/kv_transfer/kv_connector/factory.py](/home/aoshen/setup_new_cluster/vllm/vllm/distributed/kv_transfer/kv_connector/factory.py#L212)

`MooncakeStoreConnector` 的本地注释写得很明确：它是“使用 `MooncakeDistributedStore` 作为 shared KV cache pool”的 connector：

- 说明位置：[vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_connector.py](/home/aoshen/setup_new_cluster/vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_connector.py#L3)

这条路线的特点是：

- 结构更直接
- 组件更少
- vLLM 直接理解 Mooncake 的共享 KV pool
- 但 vLLM 侧就直接绑定到了 Mooncake 的语义和配置模型

### 路线 B：vLLM 先接 LMCache，再由 LMCache 接 Mooncake

LMCache 官方文档已经明确给出 Mooncake 作为其 storage backend 的接法，启动 vLLM 时使用的是：

```bash
--kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'
```

对应本地文档位置：

- [LMCache/docs/source/kv_cache/storage_backends/mooncake.rst](/home/aoshen/setup_new_cluster/LMCache/docs/source/kv_cache/storage_backends/mooncake.rst#L98)

LMCache 这时通过：

- `remote_url: "mooncakestore://host:port"` 连接 Mooncake
- `extra_config` 继续把 Mooncake 的参数透传进去

对应本地文档 / 代码位置：

- `remote_url` 和 `external_lookup_client` 配置项：[LMCache/docs/source/api_reference/configurations.rst](/home/aoshen/setup_new_cluster/LMCache/docs/source/api_reference/configurations.rst#L46)
- Mooncake backend 配置样例：[LMCache/docs/source/kv_cache/storage_backends/mooncake.rst](/home/aoshen/setup_new_cluster/LMCache/docs/source/kv_cache/storage_backends/mooncake.rst#L72)
- LMCache 内置 Mooncake connector 实现：[LMCache/lmcache/v1/storage_backend/connector/mooncakestore_connector.py](/home/aoshen/setup_new_cluster/LMCache/lmcache/v1/storage_backend/connector/mooncakestore_connector.py#L94)

## LMCache 相比直接 `MooncakeStoreConnector` 的优势

这里的“优势”不是说 Mooncake 不好，而是两者所在层次不同。Mooncake 更像高性能远端存储 / 传输底座；LMCache 更像 KV cache 管理层。

### 1. 抽象层更高，vLLM 不再直接绑定某个远端后端

LMCache 的架构是：

- GPU working set
- CPU DRAM 热层
- 本地盘 / GDS
- 远端后端（Redis / Mooncake / InfiniStore 等）

本地架构文档：

- [LMCache/docs/source/developer_guide/architecture.rst](/home/aoshen/setup_new_cluster/LMCache/docs/source/developer_guide/architecture.rst#L8)

这意味着你把 vLLM 接到 LMCache 后，后面到底接 Mooncake、Redis、文件系统，还是多层组合，vLLM 侧接口都不需要变。

直接收益：

- 后端替换成本更低
- 便于 A/B 测试不同 remote backend
- 后面做多层缓存时，不需要再改 vLLM connector

### 2. LMCache 天生是分层缓存，不只是“远端 KV 池”

LMCache 文档里明确写了它支持：

- GPU -> CPU offload
- CPU -> 本地盘 / 远端异步写入
- 热数据预取回 CPU
- 按需搬回 GPU

对应位置：

- [LMCache/docs/source/developer_guide/architecture.rst](/home/aoshen/setup_new_cluster/LMCache/docs/source/developer_guide/architecture.rst#L19)

这和“vLLM 直接把 KV 存到 Mooncake Store”相比，差别在于：

- 直接 Mooncake 路线主要是“共享远端 KV 池”
- LMCache 路线是“分层缓存编排 + Mooncake 作为其中一层”

对长上下文、多轮对话、RAG 这种冷热分布明显的场景，LMCache 这层通常更有价值。

### 3. LMCache 有更完整的 cache 管理面

LMCache 不只是存和取，它还提供了控制面能力：

- lookup
- clear
- compress / decompress
- move
- pin / unpin
- health / finish checks

对应位置：

- [LMCache/docs/source/developer_guide/architecture.rst](/home/aoshen/setup_new_cluster/LMCache/docs/source/developer_guide/architecture.rst#L98)

这类能力对于生产环境非常关键，因为你后面常常会需要：

- 预热某批 prompt 的 KV
- 固定某些热点 cache 不被淘汰
- 观察某类 cache 命中率
- 做 cache-aware routing

直接 `MooncakeStoreConnector` 更偏“数据面 connector”，LMCache 更偏“数据面 + 控制面”。

### 4. LMCache 的 connector 能力更丰富，和 vLLM 的解耦更彻底

LMCache 的技术报告把自己的核心价值总结为三点：

- 高性能 KV cache 数据搬运
- 模块化 connector，尽量跟推理引擎演进解耦
- 一等公民的 control API

官方技术报告：

- https://lmcache.ai/tech_report.pdf

这一点对你现在这个需求很贴切：你本来就想把 “vLLM 直接接 Mooncake” 改成 “vLLM 接一个更中立的 cache 层”。LMCache 正好就是这一层。

### 5. 当前本地代码里，LMCache connector 支持 layerwise/load-save pipeline，而 `MooncakeStoreConnector` 没有

在本地 checkout 里：

- `LMCacheConnectorV1` 暴露了 `start_load_kv`、`wait_for_layer_load`、`save_kv_layer`、`wait_for_save`
- 并且还显式处理了 `use_layerwise`

对应位置：

- [vllm/vllm/distributed/kv_transfer/kv_connector/v1/lmcache_connector.py](/home/aoshen/setup_new_cluster/vllm/vllm/distributed/kv_transfer/kv_connector/v1/lmcache_connector.py#L72)

而 `MooncakeStoreConnector` 当前实现里：

- `wait_for_layer_load` 是 no-op
- `save_kv_layer` 是 no-op
- 注释直接写了 “No layerwise support”

对应位置：

- [vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_connector.py](/home/aoshen/setup_new_cluster/vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_connector.py#L173)

这说明在你当前本地版本里，LMCache 路线在 cache pipeline 能力上更完整。

### 6. LMCache 可以把 Mooncake 当成远端后端，而不是唯一后端

LMCache 的 remote backend 是插件式 / 适配器式设计，Mooncake 只是其中一种。

对应位置：

- [LMCache/docs/source/developer_guide/architecture.rst](/home/aoshen/setup_new_cluster/LMCache/docs/source/developer_guide/architecture.rst#L95)

这带来的工程优势是：

- 你可以先保留 Mooncake
- 后面再引入本地 disk / Redis / 其他远端层
- 或者把 Mooncake 只作为 cold tier，而 CPU / disk 作为更热的 tier

### 7. 官方协同路线已经明确在推 “LMCache + Mooncake”

Mooncake 官方站和 LMCache 官方文档都已经把这条路线写成正式集成方案：

- LMCache 官方 Mooncake backend 文档：https://docs.lmcache.ai/kv_cache/storage_backends/mooncake.html
- Mooncake 官方 LMCache 集成页：https://kvcache-ai.github.io/Mooncake/getting_started/examples/lmcache-integration.html

Mooncake 官方还给出了联合 benchmark，展示了 `vLLM + LMCache + Mooncake Store` 的 cache-hit 改善结果。

## 当 LMCache 接入 Mooncake 时，LMCache 的角色是什么

这是这次调研里最重要的问题。

答案是：**LMCache 不再是“另一个远端存储”，而是 vLLM 和 Mooncake 之间的 KV cache 管理层 / 编排层。**

可以这样理解：

- Mooncake：远端共享 KV pool，偏数据面，负责分布式内存池、RDMA、多 NIC 带宽聚合、远端对象读写
- LMCache：缓存语义层，负责 chunking、索引、命中判断、冷热分层、异步 offload、回填、控制面、策略编排
- vLLM：推理执行层，负责真正的 prefill / decode 计算

更直白一点：

- 没有 LMCache 时，vLLM 直接“看见” Mooncake
- 有 LMCache 时，vLLM 不直接操作 Mooncake，而是只和 LMCache 说话
- LMCache 再决定哪些 KV 留在本地 CPU，哪些刷到 Mooncake，哪些从 Mooncake 拉回

### 这时 Mooncake 主要扮演什么角色

Mooncake 在 LMCache 背后通常扮演 3 个角色：

- 远端存储层：作为 `remote_url: mooncakestore://...` 指向的远端 KV backend
- 高性能远端数据面：通过 Transfer Engine / RDMA 提供高带宽传输
- 可选的外部 lookup 服务：LMCache 配置项里还有 `external_lookup_client: mooncakestore://...`

对应位置：

- `external_lookup_client` 说明：[LMCache/docs/source/api_reference/configurations.rst](/home/aoshen/setup_new_cluster/LMCache/docs/source/api_reference/configurations.rst#L76)

### 这时 LMCache 主要扮演什么角色

LMCache 在 Mooncake 前面通常扮演 6 个角色：

- vLLM connector 层：对接 `LMCacheConnectorV1`
- KV 索引层：维护 token/chunk 到 cache 对象位置的映射
- 分层缓存编排层：管理 CPU / disk / remote 的冷热分层
- 搬运与序列化层：负责 chunk 化、异步 load/store、serde、零拷贝协同
- 控制面：提供 lookup / clear / move / pin 等能力
- 策略层：决定 store / retrieve / eviction / min_retrieve_tokens / priority 等行为

## 一个很关键的边界：LMCache + Mooncake 不等于 LMCache 的 PD 模式

这里很容易混淆。

LMCache 文档明确写了：

- 当 `enable_pd=true` 时，`remote_url` 必须为 null

对应位置：

- [LMCache/docs/source/api_reference/configurations.rst](/home/aoshen/setup_new_cluster/LMCache/docs/source/api_reference/configurations.rst#L219)

所以：

- `LMCache + remote_url: mooncakestore://...` 这条路，本质上是 **storage/offload/reuse** 路线
- 它不是 LMCache 自己的 PD transport 路线
- 如果你要的是 prefill/decode 之间的实时跨节点 KV 传输，那要看 Mooncake Transfer Engine connector 或 LMCache 自己的 PD/NIXL 路线

也就是说：

- 替代 `MooncakeStoreConnector`：LMCache 很适合
- 替代 `MooncakeConnector` / Transfer Engine 这类 PD connector：要单独评估，不应直接画等号

## 对你这个场景的建议

### 推荐方案

如果你的目标是“用更通用的 cache 层替换 vLLM 对 Mooncake 的直接耦合”，推荐：

1. vLLM 侧统一切到 `LMCacheConnectorV1`
2. LMCache 侧先保留 Mooncake 作为 remote backend
3. 先启用 `local_cpu + Mooncake remote`
4. 后续再视命中率和内存占用决定是否加 `local_disk`

这样做的好处是：

- 改动边界清晰
- 先不丢掉 Mooncake 的 RDMA / distributed pool 优势
- 同时拿到 LMCache 的分层缓存和控制面能力
- 后面切其他 backend 或做多层缓存都更容易

### 不那么推荐直接迁到 LMCache 的场景

如果你现在最关心的是：

- 最少组件数
- 最短链路
- 只需要 Mooncake 这个共享 KV pool
- 暂时不关心 CPU / disk / cache control / backend 可替换性

那么直接 `MooncakeStoreConnector` 仍然有其简单性优势。

也就是说，LMCache 的收益来自“能力更强、层次更高”，代价是“系统里多了一层”。

## 一个比较实用的目标架构

```text
vLLM
  -> LMCacheConnectorV1
    -> LMCache
      -> Local CPU cache
      -> Local disk cache (optional)
      -> Mooncake Store remote backend
```

对应启动形态通常是：

```bash
LMCACHE_CONFIG_FILE=lmcache_mooncake.yaml \
vllm serve <model> \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
```

配置核心类似：

```yaml
local_cpu: true
max_local_cpu_size: 20
remote_url: "mooncakestore://127.0.0.1:50051/"
numa_mode: "auto"

extra_config:
  local_hostname: "127.0.0.1"
  metadata_server: "http://127.0.0.1:8080/metadata"
  master_server_address: "127.0.0.1:50051"
  protocol: "rdma"
  device_name: ""
  global_segment_size: 21474836480
  local_buffer_size: 0
  mooncake_prefer_local_alloc: true
  save_chunk_meta: false
```

## 最后的判断

如果只用一句话回答你的问题：

**LMCache 的优势，不在于“它比 Mooncake 更会做 RDMA”，而在于它把 Mooncake 这类远端能力包装进一个更完整的 KV cache 管理体系里。**

因此，当 LMCache 再去接 Mooncake 时：

- Mooncake 负责“远端 KV 存储和高性能传输”
- LMCache 负责“缓存语义、分层管理、命中复用、控制面和策略编排”

所以此时 LMCache 的角色不是 Mooncake 的替代品，而是 Mooncake 之上的 **cache management layer / orchestration layer**。

## 参考资料

- LMCache 技术报告：https://lmcache.ai/tech_report.pdf
- LMCache Mooncake backend 文档：https://docs.lmcache.ai/kv_cache/storage_backends/mooncake.html
- LMCache storage backends 总览：https://docs.lmcache.ai/kv_cache/storage_backends/index.html
- Mooncake 官方首页：https://kvcache-ai.github.io/Mooncake/
- Mooncake x LMCache 官方集成页：https://kvcache-ai.github.io/Mooncake/getting_started/examples/lmcache-integration.html

