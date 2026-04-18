# LMCache Integration Research for Replacing MooncakeStoreConnector in vLLM

## Conclusion first

If your goal is:

- Continue to retain Mooncake as the remote KV cache pool
- But I hope the vLLM side will not directly tie `MooncakeStoreConnector`
- And can smoothly switch to other remote backends later, or increase local CPU/local disk/compression/control plane capabilities

Then the more recommended route is:

`vLLM -> LMCacheConnectorV1 -> LMCache -> Mooncake Store`

instead of:

`vLLM -> MooncakeStoreConnector -> Mooncake Store`

In one sentence:

- Direct `MooncakeStoreConnector`: Mooncake is both a remote storage pool and a KV connector that vLLM directly connects to.
- Go to `LMCacheConnectorV1`: LMCache becomes the KV cache management layer of vLLM, and Mooncake retreats behind LMCache to act as remote storage/remote lookup/RDMA data plane.

## Now there are actually two completely different methods of connection.

### Route A: vLLM directly to Mooncake

In the local code, vLLM registered `LMCacheConnectorV1` and `MooncakeStoreConnector` at the same time, indicating that they are parallel solutions, not the same thing:

- `LMCacheConnectorV1`: [vllm/vllm/distributed/kv_transfer/kv_connector/factory.py](../../vllm/distributed/kv_transfer/kv_connector/factory.py#L164)
- `MooncakeStoreConnector`: [vllm/vllm/distributed/kv_transfer/kv_connector/factory.py](../../vllm/distributed/kv_transfer/kv_connector/factory.py#L212)

The local annotation of `MooncakeStoreConnector` is very clear: it is a connector that "uses `MooncakeDistributedStore` as a shared KV cache pool":

- Description location: [vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_connector.py](../../vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_connector.py#L3)

Characteristics of this route are:

- More straightforward structure
- Fewer components
- vLLM directly understands Mooncake’s shared KV pool
- But the vLLM side is directly bound to Mooncake’s semantics and configuration model

### Route B: vLLM first connects to LMCache, and then LMCache connects to Mooncake

The official LMCache documentation has clearly given Mooncake as the connection method for its storage backend. When starting vLLM, the following is used:```bash
--kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'
```
Corresponding local document location:

- `LMCache/docs/source/kv_cache/storage_backends/mooncake.rst#L98`

LMCache now passes:

- `remote_url: "mooncakestore://host:port"` connect to Mooncake
- `extra_config` continues to transparently pass Mooncake parameters into it

Corresponding local documentation/code location:

- `remote_url` and `external_lookup_client` configuration items: `LMCache/docs/source/api_reference/configurations.rst#L46`
- Mooncake backend configuration example: `LMCache/docs/source/kv_cache/storage_backends/mooncake.rst#L72`
- LMCache built-in Mooncake connector implementation: `LMCache/lmcache/v1/storage_backend/connector/mooncakestore_connector.py#L94`

## Advantages of LMCache over direct `MooncakeStoreConnector`

The "advantage" here is not that Mooncake is bad, but that the two are on different levels. Mooncake is more like a high-performance remote storage/transmission base; LMCache is more like a KV cache management layer.

### 1. The abstraction layer is higher, vLLM is no longer directly bound to a remote backend

The architecture of LMCache is:

-GPU working set
- CPU DRAM thermal layer
- Local disk/GDS
- Remote backend (Redis/Mooncake/InfiniStore, etc.)

Local schema documentation:

- `LMCache/docs/source/developer_guide/architecture.rst#L8`

This means that after you connect vLLM to LMCache, whether it is connected to Mooncake, Redis, file system, or a multi-layer combination, the vLLM side interface does not need to change.

Direct benefits:

- Lower backend replacement costs
- Facilitates A/B testing of different remote backends
- When doing multi-layer caching later, there is no need to change the vLLM connector.

### 2. LMCache is inherently a hierarchical cache, not just a "remote KV pool"

The LMCache documentation clearly states that it supports:

- GPU -> CPU offload
- CPU -> local disk / remote asynchronous writing
- Hot data prefetching to CPU
- Move back GPU on demand

Corresponding location:

- `LMCache/docs/source/developer_guide/architecture.rst#L19`Compared with "vLLM directly saves KV to Mooncake Store", the difference is:

- Direct Mooncake route is mainly "shared remote KV pool"
- The LMCache route is "layered cache orchestration + Mooncake as one of the layers"

For scenarios such as long contexts, multi-turn conversations, and RAG, where hot and cold distribution is obvious, the LMCache layer is usually more valuable.

### 3. LMCache has a more complete cache management surface

LMCache is not just about storing and retrieving, it also provides control plane capabilities:

-lookup
- clear
- compress/decompress
- move
-pin/unpin
- health/finish checks

Corresponding location:

- `LMCache/docs/source/developer_guide/architecture.rst#L98`

This type of capability is critical in a production environment because you will often need to:

- Preheat the KV of a certain batch of prompts
- Fixed some hotspot caches not being eliminated
- Observe a certain type of cache hit rate
- Do cache-aware routing

Directly `MooncakeStoreConnector` is more of a "data plane connector", while LMCache is more of a "data plane + control plane".

### 4. LMCache’s connector has richer capabilities and is more completely decoupled from vLLM

LMCache’s technical report summarizes its core values into three points:

- High-performance KV cache data transfer
- Modular connector, try to decouple it from the evolution of the inference engine
- Control API for first class citizens

Official technical report:

- https://lmcache.ai/tech_report.pdf

This is very appropriate for your current needs: you originally wanted to change "vLLM directly connected to Mooncake" to "vLLM connected to a more neutral cache layer". LMCache is exactly this layer.

### 5. In the current local code, LMCache connector supports layerwise/load-save pipeline, but `MooncakeStoreConnector` does not

In local checkout:

- `LMCacheConnectorV1` exposes `start_load_kv`, `wait_for_layer_load`, `save_kv_layer`, `wait_for_save`
- and also explicitly handles `use_layerwise`

Corresponding location:

- [vllm/vllm/distributed/kv_transfer/kv_connector/v1/lmcache_connector.py](../../vllm/distributed/kv_transfer/kv_connector/v1/lmcache_connector.py#L72)

In the current implementation of `MooncakeStoreConnector`:

- `wait_for_layer_load` is no-op
- `save_kv_layer` is no-op
- The comment directly says "No layerwise support"

Corresponding location:

- [vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_connector.py](../../vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_connector.py#L173)This shows that in your current local version, the LMCache route is more complete in terms of cache pipeline capabilities.

### 6. LMCache can use Mooncake as a remote backend instead of the only backend

LMCache's remote backend is a plug-in/adapter design, and Mooncake is just one of them.

Corresponding location:

- `LMCache/docs/source/developer_guide/architecture.rst#L95`

The engineering advantages this brings are:

- You can keep Mooncake first
- Local disk / Redis / other remote layers will be introduced later
- Or make Mooncake only the cold tier and CPU/disk the hotter tier

### 7. The official collaboration route has clearly promoted “LMCache + Mooncake”

Both the Mooncake official website and LMCache official documents have written this route into a formal integration solution:

- LMCache official Mooncake backend documentation: https://docs.lmcache.ai/kv_cache/storage_backends/mooncake.html
- Mooncake official LMCache integration page: https://kvcache-ai.github.io/Mooncake/getting_started/examples/lmcache-integration.html

Mooncake officials also gave a joint benchmark, showing the cache-hit improvement results of `vLLM + LMCache + Mooncake Store`.

## What is the role of LMCache when it is connected to Mooncake?

This is the most important question in this survey.

The answer is: **LMCache is no longer "another remote storage", but the KV cache management/orchestration layer between vLLM and Mooncake. **

It can be understood like this:

- Mooncake: Remote shared KV pool, partial to the data side, responsible for distributed memory pool, RDMA, multi-NIC bandwidth aggregation, remote object reading and writing
- LMCache: cache semantic layer, responsible for chunking, indexing, hit judgment, hot and cold stratification, asynchronous offload, backfill, control plane, and policy orchestration
- vLLM: reasoning execution layer, responsible for real prefill/decode calculations

To be more straightforward:

- Without LMCache, vLLM "sees" Mooncake directly
- When there is LMCache, vLLM does not directly operate Mooncake, but only talks to LMCache
- LMCache then decides which KVs stay in the local CPU, which ones are flushed to Mooncake, and which ones are pulled back from Mooncake

### What is Mooncake’s main role at this time?

Mooncake usually plays 3 roles behind LMCache:

- Remote storage layer: as the remote KV backend pointed to by `remote_url: mooncakestore://...`
- High-performance remote data plane: Provides high-bandwidth transmission through Transfer Engine / RDMA
- Optional external lookup service: There is also `external_lookup_client: mooncakestore://...` in the LMCache configuration item

Corresponding location:

- `external_lookup_client` Description: `LMCache/docs/source/api_reference/configurations.rst#L76`### What is the main role of LMCache at this time?

LMCache usually plays 6 roles in front of Mooncake:

- vLLM connector layer: docking `LMCacheConnectorV1`
- KV index layer: maintains the mapping of token/chunk to cache object location
- Hierarchical cache orchestration layer: manages hot and cold stratification of CPU/disk/remote
- Transport and serialization layer: responsible for chunking, asynchronous load/store, serde, and zero-copy collaboration
- Control surface: Provides capabilities such as lookup / clear / move / pin etc.
- Strategy layer: determines behaviors such as store / retrieve / eviction / min_retrieve_tokens / priority etc.

## A very critical boundary: LMCache + Mooncake is not equal to LMCache’s PD mode

It's easy to get confused here.

The LMCache documentation clearly states:

- `remote_url` must be null when `enable_pd=true`

Corresponding location:

- `LMCache/docs/source/api_reference/configurations.rst#L219`

So:

- `LMCache + remote_url: mooncakestore://...` This path is essentially the **storage/offload/reuse** path
- It is not LMCache's own PD transport route
- If you want real-time cross-node KV transfer between prefill/decode, then look at the Mooncake Transfer Engine connector or LMCache's own PD/NIXL route

That is to say:

- Alternative to `MooncakeStoreConnector`: LMCache is suitable
- Replace PD connectors such as `MooncakeConnector` / Transfer Engine: to be evaluated separately, the equal sign should not be drawn directly

## Suggestions for your scenario

### Recommended plan

If your goal is to "replace vLLM's direct coupling to Mooncake with a more general cache layer", it is recommended:

1. vLLM side unified switch to `LMCacheConnectorV1`
2. The LMCache side first retains Mooncake as the remote backend.
3. First enable `local_cpu + Mooncake remote`
4. We will then decide whether to add `local_disk` based on the hit rate and memory usage.

The benefits of doing this are:

- Change boundaries are clear
- Don’t throw away Mooncake’s RDMA / distributed pool advantages yet
- Get LMCache’s hierarchical caching and control plane capabilities at the same time
- It will be easier to cut other backends or do multi-layer caching later

### It is not recommended to migrate directly to LMCache.

If what you are most concerned about right now is:

- Minimum number of components
- shortest link
- Just need Mooncake this shared KV pool
- Don't care about CPU/disk/cache control/backend replaceability yet

Well the direct `MooncakeStoreConnector` still has its simplicity advantage.

In other words, LMCache's benefits come from "stronger capabilities and higher levels", at the cost of "an extra layer in the system."

## A more practical target architecture```text
vLLM
  -> LMCacheConnectorV1
    -> LMCache
      -> Local CPU cache
      -> Local disk cache (optional)
      -> Mooncake Store remote backend
```
The corresponding startup form is usually:```bash
LMCACHE_CONFIG_FILE=lmcache_mooncake.yaml \
vllm serve <model> \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
```
Configuring the core is similar to:```yaml
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
## Final judgment

If I could answer your question in just one sentence:

**The advantage of LMCache is not that "it is better at RDMA than Mooncake", but that it packages remote capabilities such as Mooncake into a more complete KV cache management system. **

Therefore, when LMCache picks up Mooncake again:

- Mooncake is responsible for "remote KV storage and high-performance transmission"
- LMCache is responsible for "cache semantics, hierarchical management, hit multiplexing, control plane and policy orchestration"

So the role of LMCache at this time is not a replacement for Mooncake, but a **cache management layer / orchestration layer** on top of Mooncake.

## References

- LMCache technical report: https://lmcache.ai/tech_report.pdf
- LMCache Mooncake backend documentation: https://docs.lmcache.ai/kv_cache/storage_backends/mooncake.html
- LMCache storage backends overview: https://docs.lmcache.ai/kv_cache/storage_backends/index.html
- Mooncake official homepage: https://kvcache-ai.github.io/Mooncake/
- Mooncake x LMCache official integration page: https://kvcache-ai.github.io/Mooncake/getting_started/examples/lmcache-integration.html