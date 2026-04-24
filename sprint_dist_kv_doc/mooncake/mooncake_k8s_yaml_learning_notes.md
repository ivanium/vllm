---
title: Mooncake k8s Deployment YAML Learning Notes (HA vs Standalone Comparison + Source Verification)
audience: Engineers trying to understand Mooncake deployment and runtime mechanisms
last_verified: 2026-04-23
sources:
  - vllm/mooncake-standalone.yaml           (minimal single-master deployment)
  - vllm/mooncake_high_availablity.yaml     (production deployment with HA + tiered store)
  - vllm/third_partys/Mooncake/             (Mooncake source, verified with grep)
related_docs:
  - mooncake_k8s_deployment_advanced.md        (advanced HA master / etcd / health probe notes)
  - sglang_mooncake_env_config_deep_dive.md   (full env variable overview)
  - k8s_mooncake_ha_deployment_notes.md       (higher-level HA deployment notes; referenced by the source doc but not present in this checkout)
---

# Mooncake k8s Deployment YAML Learning Notes

This document compares two real YAML files, **minimal standalone** and **production HA**, layer by layer. It uses source-code evidence to explain the runtime behavior behind each configuration item. Goal: after reading these notes, you should be able to look at any Mooncake deployment YAML and quickly tell which tradeoffs it chose.

---

## § 1 · Side-by-Side View of the Two YAMLs

### 1.1 Skeleton comparison

| Dimension | standalone.yaml | HA.yaml |
|---|---|---|
| `kind` | `RoleBasedGroup` | `RoleBasedGroup` (same) |
| Number of roles | 2 (`master` + `store`) | **4** (`master` + `store` + `store-prefill` + `store-decode`) |
| master replicas | **1** | **3** |
| master HA | off (no `--enable_ha`) | on (`--enable_ha=true` + etcd) |
| master podAntiAffinity | none | yes (`topologyKey: hostname`) |
| store replicas | 4, single container | `store` 12 / `store-prefill` 6 / `store-decode` 2, **two containers per pod (dual NUMA)** |
| store segment size | 60 GB | 1000 GB (dedicated) / 500 GB (colocated) |
| store NICs | 1 NIC (`mlx5_5`) | 8 NICs (`mlx5_10..17`) |
| `hostNetwork` | master=false, store=true | same |
| `numactl --membind` | **none** | **yes** (dual NUMA binding) |
| `MC_ENABLE_DEST_DEVICE_AFFINITY` | `"1"` | `"1"` |

### 1.2 One-line positioning

- **standalone.yaml**: a "development / small-scale test" template. One master plus several stores. It runs, but does not tolerate a single master failure, and uses one NIC.
- **HA.yaml**: a "large-scale production" template. Three masters with etcd leader election, tiered stores, dual-NUMA optimization, and a full 8-NIC configuration.

### 1.3 Full configuration comparison, grouped into 8 categories

The skeleton above only lists the most visible differences. The tables below include **every YAML field plus the mechanism behind it**, except the disk eviction layer, which is left for later documentation. The "one-line mechanism" column points to the corresponding section or source location.

#### A. Architecture & Scheduling

| Dimension | standalone | HA | One-line mechanism |
|---|---|---|---|
| `kind` | `RoleBasedGroup` | same | **sgl-project/rbg** (github.com/sgl-project/rbg), apiVersion `workloads.x-k8s.io/v1alpha1`. It gives "multi-role systems" capabilities that native StatefulSet lacks: cross-role startup ordering, shared upgrade semantics, and cross-role readiness checks. |
| master replicas | 1 | **3 (odd quorum)** | HA uses an odd replica count plus podAntiAffinity to spread replicas across machines, so losing any one machine still leaves quorum. See [advanced doc §1.2](./mooncake_k8s_deployment_advanced.md). |
| master `podAntiAffinity` | none | `required + hostname` | `required` is a hard constraint. If the cluster does not have enough nodes satisfying the label constraints, the pod stays Pending. `preferred` is a soft constraint. |
| `nodeAffinity` by role | none | `required` + label match | Different store tiers are pinned to the corresponding node pools. |

#### B. Metadata & HA Layer

| Dimension | standalone | HA | One-line mechanism |
|---|---|---|---|
| `MOONCAKE_TE_META_DATA_SERVER` | `P2PHANDSHAKE` | `P2PHANDSHAKE` | TransferEngine metadata mode. It does **not** use a standalone metadata server. See §1.4. |
| `MOONCAKE_MASTER` format | `host:port` | `etcd://...` | The latter lets the client discover the leader through etcd. See [advanced doc §2](./mooncake_k8s_deployment_advanced.md). |
| `--enable_ha` | absent | `true` | On startup, the master registers into etcd and joins Raft leader election. |
| `--etcd_endpoints` | absent | external etcd cluster | etcd itself needs a separately deployed 3- or 5-node HA cluster. It is **not covered by the Mooncake YAML**. |
| `--rpc_address` | defaults to localhost | `$(POD_IP)` | This address is written into etcd so clients can discover the leader. |

#### C. Memory & Buffer

| Dimension | standalone | HA | One-line mechanism |
|---|---|---|---|
| `MOONCAKE_GLOBAL_SEGMENT_SIZE` | `60gb` | `1000gb` (dedicated) / `500gb` (colocated) | DRAM contributed to the global pool, registered to master through `MountSegment`. |
| `MOONCAKE_LOCAL_BUFFER_SIZE` | default 16 MB | `67108864` (64 MB) | **Not the same memory as the previous item**. See §1.5. |
| Client local hot cache (L1.5) | on by default | on | **Not a Mooncake replica**. The master is unaware of it. It is local cache inside the client process, with **CMS frequency admission**, LRU eviction, and optional shm sharing. See §1.7. |
| Object replication factor | 1 (default) | 1 (default) | `ReplicateConfig` can specify N replicas per object. Neither YAML changes it. |
| Hugepage | off by default | off by default | `should_use_hugepage_`, controlled by internal config or env. |
| `max_mr_size` (single MR limit) | 1 TiB (default) | 1 TiB (default) | `config.h:38` `0x10000000000`; can be overridden by `MC_MAX_MR_SIZE`. |

#### D. Eviction & Lease

| Dimension | standalone | HA | One-line mechanism |
|---|---|---|---|
| `--eviction_high_watermark_ratio` | default `0.95` | `0.9` (more aggressive) | Background eviction starts when usage reaches this ratio. See §1.6. |
| Eviction algorithm | **near-LRU** | same | `master_service.h:555` comments: "BatchEvict evicts objects in a near-LRU way". `eviction_strategy.h` also defines `LRUEvictionStrategy` and `FIFOEvictionStrategy`. **Note**: `SIEVE = 1` is `EndpointStoreType` for the TransferEngine endpoint cache, **not** master eviction. |
| `--default_kv_lease_ttl` (ms) | `300000` (5 min) | `10000` (10 s) | Mooncake default is `DEFAULT_DEFAULT_KV_LEASE_TTL = 5000 ms` (`types.h:84`). |
| Lease renewal mechanism | implicit (Get/Put auto-refresh) | same | On every `Get` / `Put`, master calls `metadata.GrantLease(default_kv_lease_ttl_, ...)` automatically. There is **no separate heartbeat RPC**. Keys with no access for longer than TTL become evictable. See §1.6. |

#### E. Identity & Ports

| Dimension | standalone | HA | One-line mechanism |
|---|---|---|---|
| `MOONCAKE_LOCAL_HOSTNAME` source | Downward API `status.podIP` | same | With `hostNetwork: true`, this is the host IP. RDMA peers need this address to connect back. |
| Client identity | `client_id_` UUID + `local_hostname_` (includes port) | same | UUID is constructed by `generate_uuid()`. Hostname is reported to the master during `MountSegment`. |
| Segment identity | each `MountSegment` generates `segment.id` UUID | same | **One Client can own N Segments** (`mounted_segments_` is a map). See §1.8. |
| AutoPortBinder port range | `[12300, 14300]` random | same | `utils.cpp:51-82`. `bind()` reserves the port for mutual exclusion; retry limit is 20 (`MC_STORE_CLIENT_SETUP_RETRIES`). |
| Ports occupied per single container | 3 to 4 | same | See §1.9. |

#### F. NIC / NUMA / Affinity

| Dimension | standalone | HA | One-line mechanism |
|---|---|---|---|
| `MOONCAKE_DEVICE` | `mlx5_5` (1 NIC) | `mlx5_10..17` (8 NICs) | Node IB NIC list, comma-separated. It also supports a JSON dict for explicit GPU-to-NIC mapping. |
| NUMA binding | none | external `numactl --membind=N` | Does not use Mooncake's internal NUMA splitting path. That path is only active in daemon mode. See §4.4. |
| `MC_ENABLE_DEST_DEVICE_AFFINITY` | `"1"` | `"1"` | When sending data, prefer the NIC closest to the destination GPU. This is a key optimization for rail-aligned topology. |
| NIC-NUMA topology discovery | `TransferEngine::getLocalTopology()` | same | Reads sysfs plus `ibv_query_gid` to build a `cpu:N -> preferred_hca[]` matrix. |
| `MC_MS_AUTO_DISC=1` | unset | unset | If set, it **forcibly overrides** `MOONCAKE_DEVICE` and switches to auto-discovery. |

#### G. Network & DNS

| Dimension | standalone | HA | One-line mechanism |
|---|---|---|---|
| master `hostNetwork` | `false` | `false` | Control plane uses pod network. |
| store `hostNetwork` | `true` | `true` | Required because RDMA GIDs are bound to physical NICs. |
| `dnsPolicy` | `ClusterFirstWithHostNet` | same | With `hostNetwork`, DNS defaults to the host resolver. This setting lets the pod still resolve `*.svc.cluster.local`. |
| `dnsConfig.nameservers` | custom `10.60.0.2/3` | default | standalone adds nameservers for a special DNS environment. |
| `/dev/infiniband` mount | **explicit** `hostPath` + `volumeMounts` | implicit through `privileged: true` | Explicit mounting better matches the production principle of least privilege. |

#### H. Protocol & Metrics

(Health probe rows have moved to [advanced doc §3](./mooncake_k8s_deployment_advanced.md).)

| Dimension | standalone | HA | One-line mechanism |
|---|---|---|---|
| `MOONCAKE_PROTOCOL` | `rdma` | `rdma` | The code also supports `tcp / cxl / ascend / ubshmem / rpc_only`; these YAMLs only use rdma. |
| `MC_TE_METRIC` | unset | `'true'` | Enables a background thread that **periodically logs glog INFO**: throughput and latency histogram. It is **not** a Prometheus endpoint, just logs ([transfer_engine_impl.cpp:722-758](../../third_partys/Mooncake/mooncake-transfer-engine/src/transfer_engine_impl.cpp#L722)). |
| C++ HTTP server (master `/metrics`, client `/health`) | enabled by default | enabled by default | **Another layer**: master defaults to 9003, client defaults to 9300. Prometheus scrape uses this. It is **unrelated to `MC_TE_METRIC`**. |

### 1.4 What is `P2PHANDSHAKE`?

[transfer_metadata.cpp:143-146](../../third_partys/Mooncake/mooncake-transfer-engine/src/transfer_metadata.cpp#L143):
```cpp
if (conn_string == P2PHANDSHAKE) {
    p2p_handshake_mode_ = true;
    ...
}
```

- This literal string is the TransferEngine switch for **"no central metadata server"** mode.
- Alternatives are `http://host:8080/metadata` (standalone metadata service) or `etcd://...` (etcd-backed).
- In P2PHANDSHAKE mode, the first communication between two nodes uses a **direct RPC handshake** to exchange RDMA endpoint / GID / qp_num and related information. It does not depend on shared metadata.
- The handshake port defaults to `12001` and is overridden by env `MC_HANDSHAKE_PORT` ([config.cpp:146](../../third_partys/Mooncake/mooncake-transfer-engine/src/config.cpp#L146)).
- The `p2p_handshake_mode_` boolean is used in 7 branches in `transfer_metadata.cpp`; discovery / handshake is implemented through the `HandShakePlugin` plugin mechanism ([transfer_metadata.cpp:137](../../third_partys/Mooncake/mooncake-transfer-engine/src/transfer_metadata.cpp#L137)).
- **Production recommendation**: one fewer metadata service pod and one fewer failure point, at the cost of more code complexity.
- The SGLang side must keep `MOONCAKE_TE_META_DATA_SERVER=P2PHANDSHAKE` to hit the 4-condition TransferEngine reuse path. See env-config doc §4.5.

### 1.5 `registerLocalMemory` vs `MountSegment`: two kinds of "memory registration"

The API names look similar, but they do **completely different things**. Both are called during setup:

| API | Memory purpose | Size source | Visibility |
|---|---|---|---|
| `registerLocalMemory(ptr, size, ...)` | **staging buffer for this client's Put/Get** | `MOONCAKE_LOCAL_BUFFER_SIZE` (typically 16 to 64 MB) | only this client uses it; not shared |
| `MountSegment(ptr, size, protocol, location)` | **DRAM contributed to the global pool** | `MOONCAKE_GLOBAL_SEGMENT_SIZE` (GB to TB scale) | registered into master; accessible by all clients in the cluster |

The code calls both ([real_client.cpp:473](../../third_partys/Mooncake/mooncake-store/src/real_client.cpp#L473) and [:581](../../third_partys/Mooncake/mooncake-store/src/real_client.cpp#L581)).

**Direct consequence**: even when `global_segment_size=0` (pure consumer mode, contributes no memory), the `local_buffer_size` buffer still exists and is not zero. Pure store pods can set `local_buffer_size` to 0 to save memory, but the HA YAML conservatively gives it 64 MB.

### 1.6 Combined semantics of Lease + Eviction

**Lease** - implicit renewal, not active heartbeat:
- On `Put(key, value)`, master calls `metadata.GrantLease(default_kv_lease_ttl_, default_kv_soft_pin_ttl_)` to issue a lease ([master_service.cpp:456](../../third_partys/Mooncake/mooncake-store/src/master_service.cpp)).
- **Every `Get` also calls GrantLease to renew** ([master_service.cpp:770](../../third_partys/Mooncake/mooncake-store/src/master_service.cpp#L770)), effectively refreshing on LRU access.
- `GetReplicaListResponse` returns `lease_ttl_ms` to the client, so the client can calculate a deadline, but there is **no independent renew RPC**.
- If a key is not accessed for longer than `default_kv_lease_ttl`, `IsLeaseExpired()` becomes true, and eviction may clear it ([master_service.cpp:608, 1799, 1844, 1911](../../third_partys/Mooncake/mooncake-store/src/master_service.cpp)).
- HA chooses a short 10-second lease: keys with no access lose protection within 10 s, so master memory can be reclaimed faster.
- Standalone chooses 5 minutes: development workloads have fewer keys, so it matters less.

**Eviction**:
- Two triggers:
  1. `PutStart` fails due to insufficient space, triggering immediate eviction.
  2. `EvictionThreadFunc` background thread ([master_service.cpp:2210+](../../third_partys/Mooncake/mooncake-store/src/master_service.cpp)) polls every `kEvictionThreadSleepMs` milliseconds. If usage >= `eviction_high_watermark_ratio`, it proactively evicts.
- Algorithm = **near-LRU**: comment in `master_service.h:555` says "BatchEvict evicts objects in a near-LRU way".
- `eviction_strategy.h` defines `LRUEvictionStrategy` and `FIFOEvictionStrategy`, both subclasses of abstract base `EvictionStrategy`.
- **Note**: the code also has `enum EndpointStoreType { FIFO = 0, SIEVE = 1 }` (`mooncake-transfer-engine/include/config.h:28-30`). This is a **TransferEngine endpoint-cache enum**, **not** master-side KV eviction. Do not confuse them.

**Key semantic point: lease expiration does not mean immediate deletion; it means preferred deletion.**

A lease-expired key only becomes an **eviction candidate**. It is not removed immediately. If memory is plentiful (usage < high_watermark), an expired key **can remain alive indefinitely** until `EvictionThreadFunc` is actually triggered.

Three protection layers ([master_service.cpp:3518-3583](../../third_partys/Mooncake/mooncake-store/src/master_service.cpp#L3518)):

| Object state | When it can be evicted |
|---|---|
| `IsHardPinned()` = true | **never** evicted |
| `IsSoftPinned(now)` = true | only in the "second pass" aggressive mode, when `allow_evict_soft_pinned_objects_=true` |
| normal + lease expired | preferred candidate, oldest `lease_timeout` first |
| normal + lease not expired | not evicted in this round |

**BatchEvict two-pass strategy**:
- **First pass**: clear only objects with "expired lease + no pin". It uses `std::nth_element` to select the oldest N and evicts down to the `evict_ratio_target` watermark.
- **Second pass**: if the first round is not enough (`evicted_count < evict_ratio_lowerbound`), it adds the remaining `no_pin_objects` and, if allowed, `soft_pin_objects` to the candidate set and continues eviction.

**Practical implication**: setting `default_kv_lease_ttl=10000` (10 s) in HA YAML **does not mean a key disappears after 10 seconds**. It means "10 seconds without access -> enters candidate pool and will be preferred next time memory is tight". To force a key to stay, use `with_hard_pin` in `ReplicateConfig` during Put.

HA uses `eviction_high_watermark_ratio=0.9` rather than the default 0.95. **Earlier eviction reduces hard `PutStart` failures caused by memory fragmentation**, at the cost of a slightly lower hit rate.

### 1.7 Full mechanism of client local hot cache (L1.5)

Hot cache is an easily misunderstood subsystem in Mooncake. Many people confuse it with Mooncake memory replicas. This section clarifies three points: **it is not a replica, how KV becomes hot, and how a hit skips RDMA**.

#### 1.7.1 It is not a Mooncake replica; master is completely unaware

| Dimension | Mooncake replica (MEMORY/DISK) | Hot cache block |
|---|---|---|
| Does master know it exists? | yes, registered through `MountSegment` | **no**, private to this client and never reported |
| Visible across clients? | yes, other clients can discover it through master | **no**, process-private |
| Lifecycle | lease + eviction management | disappears when the client process exits |
| Appears in `GetReplicaList` response? | yes | **no** |
| Covered by `ReplicaType` enum? | `MEMORY` / `DISK` / `LOCAL_DISK` | **no HOT_CACHE type** |
| Capacity control | `MOONCAKE_GLOBAL_SEGMENT_SIZE` | `MC_STORE_LOCAL_HOT_CACHE_SIZE` |

Analogy: **hot cache is like a CDN edge cache or Linux page cache**, while the Mooncake global pool is like the backend object store. One is "local/private"; the other is "cluster-shared".

#### 1.7.2 Heating trigger: CMS frequency admission, not cache-once

[client_service.h:522-530](../../third_partys/Mooncake/mooncake-store/include/client_service.h#L522):

```cpp
bool ShouldAdmitToHotCache(const std::string& key, bool cache_used) {
    if (!(hot_cache_ && !cache_used)) return false;
    if (admission_sketch_ == nullptr) return true;              // no CMS: admit every time
    return admission_sketch_->increment(key) >= admission_threshold_;
}
```

Semantics:
- **`cache_used=true`** (this request already used hot cache): **do not re-promote**, and do not touch the CMS counter. This avoids a "hot -> even hotter" feedback loop.
- **`cache_used=false`** (this request pulled from remote): increment the **Count-Min Sketch** counter, and only promote after it reaches `admission_threshold_`.
- `admission_sketch_ == nullptr` (CMS disabled): admit every time.

**Core design**: "only frequently accessed keys are cached", not "cache after one access". This naturally filters one-shot keys and prevents cache pollution.

#### 1.7.3 Full heating path

[client_service.cpp:811 + 3095-3116](../../third_partys/Mooncake/mooncake-store/src/client_service.cpp#L811):

```
Client::Get(key)
  +- FindFirstCompleteReplica(replicas, &replica)
  +- if (hot_cache_ && replica.is_memory_replica())
  |    cache_used = RedirectToHotCache(key, replica)   <- hit directly rewrites replica descriptor
  +- TransferRead(replica, slices)                     <- hit = local memcpy; miss = RDMA
  |
  +- if (ShouldAdmitToHotCache(key, cache_used))       <- miss and CMS threshold reached
        ProcessSlicesAsync(key, slices, replica)
          +- (skip if IsReplicaOnLocalMemory && !IsShm)
          |       ^ local segment and non-shm mode: no need to cache
          +- for each slice:
               hot_cache_handler_->SubmitPutTask(key, slice)
                 +- TouchHotKey: if already cached, only touch LRU and return
                 +- otherwise:
                 |    a. GetFreeBlock (evict LRU tail if full)
                 |    b. memcpy(block->addr, slice.ptr, slice.size)  <- synchronous copy
                 |    c. task_queue_.push(task)
                 +- worker thread async: task.hot_cache->PutHotKey(block)
                                            <- async part only inserts LRU index
```

**Key observations**:
- `memcpy` is **synchronous** and happens in the Get call stack, not asynchronously. This matters if you care about Get latency.
- The only "async" part is LRU index insertion. It does not affect when data is in place.
- Local replica + non-shm -> **skip**, because local access is already fast enough and should not consume cache quota.

#### 1.7.4 Hit path: RedirectToHotCache rewrites the replica descriptor

[client_service.cpp:1148-1171](../../third_partys/Mooncake/mooncake-store/src/client_service.cpp#L1148):

```cpp
bool Client::RedirectToHotCache(const std::string& key, Replica::Descriptor& replica) {
    HotMemBlock* blk = hot_cache_->GetHotKey(key);
    if (blk == nullptr) return false;

    mem_desc.buffer_descriptor.transport_endpoint_ = GetTransportEndpoint();  // this client
    mem_desc.buffer_descriptor.buffer_address_ = (uintptr_t)blk->addr;        // hot cache address
    return true;
}
```

**Trick**: it changes `transport_endpoint_` inside the replica descriptor to the client itself, and changes `buffer_address_` to the hot cache block address. The later `TransferRead` believes this is a "local memory replica", takes the local memcpy path, and uses **no RDMA**.

`HotMemBlock` has `ref_count` ([local_hot_cache.h:24-33](../../third_partys/Mooncake/mooncake-store/include/local_hot_cache.h#L24)). `GetHotKey` and `ReleaseHotKey` are paired so a block cannot be evicted by LRU and reused while it is being read.

#### 1.7.5 Configuration quick reference

| env | Purpose |
|---|---|
| `MC_STORE_LOCAL_HOT_CACHE_SIZE` | total hot cache capacity in bytes |
| `MC_STORE_LOCAL_HOT_BLOCK_SIZE` | size of one block, default 16 MB |
| `use_shm` parameter | constructor parameter for `LocalHotCache`; when enabled, blocks are allocated from memfd and **can be shared by dummy clients on the same node** |

shm mode is the only way to make hot cache visible across processes, and it is limited to dummy clients on the same node.

### 1.8 Identity triple: client_id / segment_id / local_hostname

```
One Client process (example: NUMA0 container)
+-------------------------------------------------+
| Client object                                    |
|   client_id_      = UUID-A  (generated at construction)  <- globally unique Client identity
|   local_hostname_ = "10.1.2.3:12847"                     <- externally reachable address (IP + AutoPortBinder port)
|   transfer_engine_ (1 instance, with separate TE rpc port)|
|                                                   |
|   mounted_segments_: map<UUID, Segment> {         |
|     UUID-alpha -> { name="10.1.2.3:12847", base, size=931 GB } |
|     UUID-beta  -> { name="10.1.2.3:12847", base, size=... }   |  <- if multiple Segments exist
|     ...                                           |
|   }                                               |
+-------------------------------------------------+
```

- `client_id_` is the identity master uses to identify the "data owner".
- `local_hostname_` is the address that lets others **connect back to you**. It is written into `segment.name`.
- `segment.id` is produced by each `MountSegment`. **One Client can own N Segments**. See §3.5.

This YAML defaults to 1 Client = 1 Segment, because 931 GB < 1 TiB `max_mr_size`, but the source-level model is one-to-many.

### 1.9 Port overview: how many ports does one store container actually occupy?

Use the `store-numa0` container in HA YAML as an example:

| Port source | Typical value | Default | Purpose | Who connects |
|---|---|---|---|---|
| Python `--port` | 8099/8100 in YAML | argparse `default=8080` ([mooncake_store_service.py:301](../../third_partys/Mooncake/mooncake-wheel/mooncake/mooncake_store_service.py#L301)) | REST API (application-level PUT/GET HTTP) | debugging / test tools |
| `AutoPortBinder` random | a dynamically chosen port in `[12300, 14300]` | same (`utils.h:376`) | **Client identity port**, the port suffix in `local_hostname` | only `bind()`, no `listen()`; pure reservation for mutual exclusion |
| TransferEngine handshake | dynamically chosen by default, optionally fixed | `MC_HANDSHAKE_PORT=12001` ([config.h:46](../../third_partys/Mooncake/mooncake-transfer-engine/include/config.h#L46)) | RDMA P2P handshake RPC | peer client during first communication |
| C++ client HTTP server | `FLAGS_http_port` | **`9300`** ([real_client.cpp:38](../../third_partys/Mooncake/mooncake-store/src/real_client.cpp#L38) `DEFINE_int32(http_port, 9300, ...)`) | `/health` `/metrics` Prometheus endpoint | k8s probe / monitoring |
| master metrics port, for comparison | 9003 in YAML | **`9003`** ([master_config.h:262](../../third_partys/Mooncake/mooncake-store/include/master_config.h#L262) `uint16_t http_port = 9003`) | master metrics, not client | Prometheus |

So one dual-NUMA pod in HA YAML occupies about 8 host ports (2 containers x 4 ports), because `hostNetwork: true` places everything on the host NIC. This is why AutoPortBinder needs process-level mutual exclusion.

### 1.10 Quick reference for the `MC_*` env family

All of these are read directly by Mooncake C++ code and **do not pass through** SGLang Python:

| env | Default | Purpose | standalone | HA |
|---|---|---|---|---|
| `MC_TE_METRIC` | unset | If set to `1`/`true`/`yes`/`on`, enables a background thread that **periodically prints glog INFO logs**: throughput MB/s + latency histogram. It is **not** a Prometheus endpoint. See [transfer_engine_impl.cpp:722-758](../../third_partys/Mooncake/mooncake-transfer-engine/src/transfer_engine_impl.cpp#L722). | — | `true` |
| `MC_TE_METRIC_INTERVAL_SECONDS` | `5` | print interval in seconds for the metric logs above | — | default |
| `MC_ENABLE_DEST_DEVICE_AFFINITY` | unset | choose the NIC closest to the destination when sending data | `1` | `1` |
| `MC_MS_AUTO_DISC` | 0 | 1 = **forcibly override** `MOONCAKE_DEVICE` and use auto-discovery | unset | unset |
| `MC_MAX_MR_SIZE` | 1 TiB | maximum size of a single MR; setup while loop splits into multiple Segments if exceeded | default | default |
| `MC_HANDSHAKE_PORT` | `12001` | RPC port for P2PHANDSHAKE | default | default |
| `MC_STORE_CLIENT_SETUP_RETRIES` | `20` | AutoPortBinder retry limit ([real_client.cpp:419](../../third_partys/Mooncake/mooncake-store/src/real_client.cpp#L419)) | default | default |
| `MC_STORE_CLIENT_METRIC` | **enabled** | client-side metric switch; set `0/false` to disable ([client_metric.h:672](../../third_partys/Mooncake/mooncake-store/include/client_metric.h#L672)) | default | default |
| `MC_STORE_CLIENT_METRIC_INTERVAL` | `0` | client metric reporting interval in seconds; 0 = collect only, do not report | default | default |
| `MC_STORE_CLUSTER_ID` | `"mooncake"` | cluster namespace for the etcd key in HA mode ([etcd_leader_coordinator.cpp:368](../../third_partys/Mooncake/mooncake-store/src/ha/leadership/backends/etcd/etcd_leader_coordinator.cpp#L368)) | — | default |
| `MC_STORE_USE_HUGEPAGE` | unset | enable hugepage allocation ([utils.h:285](../../third_partys/Mooncake/mooncake-store/include/utils.h#L285)) | unset | unset |
| `MC_STORE_HUGEPAGE_SIZE` | — | hugepage size | — | — |
| `MC_STORE_LOCAL_HOT_CACHE_SIZE` | — | client local hot cache size in bytes | — | — |
| `MC_STORE_LOCAL_HOT_BLOCK_SIZE` | — | hot cache block size in bytes | — | — |
| `MC_RPC_PROTOCOL` | — | RPC protocol selection ([master_client.h:64](../../third_partys/Mooncake/mooncake-store/include/master_client.h#L64)) | — | — |
| `MC_USE_IPV6` | unset | IPv6 support switch | — | — |
| `MC_RETRY_CNT` | `9` | generic RDMA op retry count ([config.cpp:198](../../third_partys/Mooncake/mooncake-transfer-engine/src/config.cpp#L198)) | default | default |
| `MC_CXL_DEV_SIZE` | unset | CXL-mode device size in bytes. If unset with `protocol=cxl`, it FATALs. | — | — |
| `MC_MIN_REG_SIZE` | — | minimum registration block on EIC/Barex path (`eic_max_block_size`) | — | — |
| `MC_DISABLE_METACACHE` | unset | disable TE metadata cache, debug only | — | — |

---

## § 2 · Runtime Configuration Required by RDMA

This section covers **the low-level RDMA configuration every Mooncake k8s deployment must understand**. It can look like boilerplate in YAML, but missing any part can prevent startup.

> RBG operator structure, Downward API, and other advanced k8s mechanisms have moved to [advanced doc §4, Deployment Infrastructure](./mooncake_k8s_deployment_advanced.md).

### 2.1 Why store needs `hostNetwork: true`, but master does not

| Role | `hostNetwork` | Reason |
|---|---|---|
| master | `false` | Control plane only. It does not perform RDMA data transfer, so pod network is sufficient. It registers pod IP into etcd. |
| store | `true` | **RDMA GID is bound to the physical NIC**. Pod network / CNI overlay cannot run RDMA. |

The companion setting must be `dnsPolicy: ClusterFirstWithHostNet`: with hostNetwork, DNS normally uses the host resolver. This line lets the pod still resolve `*.svc.cluster.local`, such as etcd or master Service DNS.

### 2.2 The four RDMA essentials

Both YAMLs have this configuration for store containers:

```yaml
command:
- sh -c
- |
    ulimit -n 1048576        # many file descriptors
    ulimit -l unlimited      # memlock: RDMA MR must stay pinned and unswapped
securityContext:
  privileged: true           # access /dev/infiniband/* devices
  capabilities:
    add: [IPC_LOCK, SYS_RESOURCE]
    # IPC_LOCK     -> allow mlock / ibv_reg_mr
    # SYS_RESOURCE -> allow raising rlimit upper bounds
```

**Missing any one of these four pieces can break RDMA**. The error often looks like `ibv_reg_mr: Cannot allocate memory`. Use this block as a template.

standalone.yaml additionally mounts `hostPath: /dev/infiniband` with `volumeMounts`, which is more explicit. HA.yaml relies on `privileged: true` to mount it implicitly. The explicit approach better matches production least-privilege principles.

---

## § 3 · Store Role: Single NUMA vs Dual NUMA

This is the largest structural difference between the two YAMLs.

### 3.1 Standalone: one container, one NUMA

```yaml
- name: store
  replicas: 4
  template:
    spec:
      containers:
      - command: [sh, -c, "python3 -m mooncake.mooncake_store_service"]
        env:
        - name: MOONCAKE_GLOBAL_SEGMENT_SIZE
          value: 60gb
        - name: MOONCAKE_DEVICE
          value: "mlx5_5"       # <- 1 NIC
```

- 4 pods, 1 container per pod.
- Each container contributes 60 GB; total pool is 240 GB.
- Uses 1 NIC and does not care about NUMA.

### 3.2 HA: two containers per pod plus `numactl` NUMA binding

```yaml
containers:
- command: [sh, -c, "numactl --membind=0 python3 -m mooncake.mooncake_store_service --port=8099"]
  name: store-numa0
  env:
  - name: MOONCAKE_GLOBAL_SEGMENT_SIZE
    value: "1000gb"
  - name: MOONCAKE_DEVICE
    value: "mlx5_10,mlx5_11,mlx5_12,mlx5_13,mlx5_14,mlx5_15,mlx5_16,mlx5_17"

- command: [sh, -c, "numactl --membind=1 python3 -m mooncake.mooncake_store_service --port=8100"]
  name: store-numa1
  env: (same env)
```

### 3.3 Mechanism of two containers for dual NUMA, with source evidence

Each container independently runs `mooncake_store_service`, independently calls `store.setup(...)`, and then enters C++:

**Step 1 - Python entry** ([mooncake_store_service.py:108-117](../../third_partys/Mooncake/mooncake-wheel/mooncake/mooncake_store_service.py#L108)):
```python
store = MooncakeDistributedStore()
ret = store.setup(
    local_hostname,          # = "10.1.2.3" (status.podIP, no port!)
    metadata_server,         # "P2PHANDSHAKE"
    global_segment_size,     # 1000 * 10^9 (bytes)
    local_buffer_size,
    protocol,                # "rdma"
    device_name,             # 8 NICs
    master_server_address,   # "etcd://..."
)
```

**Step 2 - C++ AutoPortBinder** ([real_client.cpp:400-461](../../third_partys/Mooncake/mooncake-store/src/real_client.cpp#L400)):
```cpp
if (colon_pos == npos) {    // no port -> allocate random port automatically
    for (retry < 20) {
        port_binder_ = std::make_unique<AutoPortBinder>();  // random in [12300, 14300]
        this->local_hostname = hostname + ":" + to_string(port);
        // example: "10.1.2.3:12847"
        ...
    }
}
```

The two containers have the same `status.podIP`, but **AutoPortBinder chooses a separate random port for each**, so their final `local_hostname` values differ:
- container 1: `"10.1.2.3:12847"` + `client_id_` UUID-A
- container 2: `"10.1.2.3:14051"` + `client_id_` UUID-B

From the master's point of view, these are **two completely independent Clients** with non-conflicting identities.

**Step 3 - MountSegment** ([real_client.cpp:534-587](../../third_partys/Mooncake/mooncake-store/src/real_client.cpp#L534)):
```cpp
while (global_segment_size > 0) {
    size_t segment_size = std::min(global_segment_size, max_mr_size);
    ...
    void *ptr = allocate_buffer_allocator_memory(segment_size, this->protocol);
    // <- normal aligned_alloc. External numactl --membind=N already pins NUMA.
    client_->MountSegment(ptr, mapped_size, protocol, seg_location);
}
```

**Note**: Mooncake code itself is NUMA-unaware in this path. NUMA separation is entirely provided by external `numactl`. This is simple and reliable.

### 3.4 Two NUMA setup methods: multi-process + numactl vs single-process internal splitting

Mooncake source supports **two** NUMA setup paths. The chosen path depends on whether **`ipc_socket_path_` is empty** ([real_client.cpp:519](../../third_partys/Mooncake/mooncake-store/src/real_client.cpp#L519)):

```cpp
if (!ipc_socket_path_.empty() && protocol == "rdma") {  // <- key condition
    seg_numa_nodes = client_->GetNicNumaNodes();
    if (seg_numa_nodes.size() > 1) {
        // internal NUMA splitting path
    }
}
```

The two paths have mutually exclusive triggers.

#### Path A - multi-process plus external `numactl` (chosen by HA.yaml)

**Trigger**: the Python entry `mooncake_store_service` calls the pybind `setup(...)`, and **hardcodes `ipc_socket_path=""`** ([store_py.cpp:1572-1575](../../third_partys/Mooncake/mooncake-integration/store/store_py.cpp#L1572)).

**Steps**:
1. YAML starts one container per NUMA, using `numactl --membind=N`:
   ```yaml
   - command:
     - sh -c
     - "numactl --membind=0 python3 -m mooncake.mooncake_store_service --port=8099"
   - command:
     - sh -c
     - "numactl --membind=1 python3 -m mooncake.mooncake_store_service --port=8100"
   ```
2. Each process independently creates `MooncakeDistributedStore().setup(...)`.
3. Each process randomly chooses a `[12300, 14300]` identity port through `AutoPortBinder`. See §1.9.
4. In the `setup_internal` while loop (`ipc_socket_path_.empty() == true`), it directly uses normal **`aligned_alloc`** to allocate `global_segment_size` bytes ([real_client.cpp:561](../../third_partys/Mooncake/mooncake-store/src/real_client.cpp#L561)).
5. Because the outer `numactl --membind=N` sets process-level memory policy, all pages faulted by kernel `aligned_alloc` / `mmap` land on NUMA N. Mooncake code does not need to know.
6. `MountSegment` reports one full block to master, with the default `kWildcardLocation` for `seg_location`.

**Pros**: simple Mooncake logic plus **process-level failure isolation**. If NUMA0 crashes, NUMA1 can still work.
**Cons**: YAML needs an extra container layer, and each container needs its own env and probes.

#### Path B - single-process internal NUMA splitting (daemon mode; HA.yaml does **not** use this)

**Trigger**: all of the following must be true:
- Use the **ConfigDict overload** of `setup()` ([store_py.cpp:1582-1608](../../third_partys/Mooncake/mooncake-integration/store/store_py.cpp#L1582)) and pass `ipc_socket_path: /some/path` in the config.
- `protocol == "rdma"`.
- The node has **multiple NUMA nodes with NICs attached** (`client_->GetNicNumaNodes().size() > 1`).

This scenario corresponds to running `mooncake_client` / `mooncake_store_service` as a **standalone daemon**, then letting SGLang access it as a dummy client with `MOONCAKE_STANDALONE_STORAGE=1` plus `MOONCAKE_CLIENT=unix:/some/path`.

##### Flow from daemon startup to master registration

```
  setup(ipc_socket_path=..., protocol=rdma)
              |
              v
  +-------------------------------------------+
  |  1. Discover NUMA topology                |
  |     GetNicNumaNodes() -> [0, 1]           |
  +-------------------------------------------+
              |
              v
  +-------------------------------------------+
  |  2. Allocate contiguous VMA + mbind each region |
  |     mmap(size) -> one VMA                 |
  |     region[0] -> mbind(NUMA 0)            |
  |     region[1] -> mbind(NUMA 1)            |
  |     (no physical pages yet, only policy)  |
  +-------------------------------------------+
              |
              v
  +-------------------------------------------+
  |  3. Register MR with ibv_reg_mr           |
  |     -> page faults follow mbind policy    |
  |     -> NUMA-aware registered memory       |
  +-------------------------------------------+
              |
              v
  +-------------------------------------------+
  |  4. MountSegment reports to master        |
  |     Segment {                             |
  |       base, size,                         |
  |       location = "segments:4096:0,1" <-   |
  |       (encodes NUMA layout for readers)   |
  |     }                                     |
  +-------------------------------------------+
              |
              v
  +-------------------------------------------+
  |  5. start_ipc_server                      |
  |     UDS listen @ ipc_socket_path          |
  |     waits for dummy clients               |
  +-------------------------------------------+
```

**Core points**:
- **Contiguous VMA, physical pages split across NUMA**: user code sees one normal buffer, while the kernel memory policy distributes physical pages across NUMA nodes by region.
- **The location string carries NUMA metadata**: `"segments:4096:0,1"` is encoded into `Segment.location`, persisted by master, and later read by readers. `resolveSegmentsLocation(offset)` calculates which NUMA this access lands on and selects the corresponding NIC.
- **Only difference from normal mode (Path A)**: adds the "discover NUMA + mbind + location encoding" sequence. Everything else is the same.

##### Why does non-empty `ipc_socket_path` trigger NUMA splitting?

`ipc_socket_path` is not a prerequisite for NUMA splitting by itself. **It is the trigger for the whole daemon mode**, and NUMA splitting is an extra sub-behavior enabled in that mode.

**What a non-empty `ipc_socket_path` does** ([real_client.cpp:762-768](../../third_partys/Mooncake/mooncake-store/src/real_client.cpp#L762)):
```cpp
if (!ipc_socket_path_.empty()) {
    start_ipc_server();      // start Unix Domain Socket server thread
    LOG(INFO) << "Starting IPC server at " << ipc_socket_path_;
}
```

After startup, [`ipc_server_func`](../../third_partys/Mooncake/mooncake-store/src/real_client.cpp#L3976) `bind`s and `listen`s on a UDS in Linux abstract namespace, accepting same-node dummy clients for shared-memory transfer.

**Causal chain**:
1. Starting an IPC server means "I am a daemon serving all dummy clients on this node".
2. Same-node dummy clients may be attached to GPUs on **any NUMA**.
3. If daemon memory only lands on NUMA 0, a client on NUMA 1 has to cross QPI. More seriously, **a NIC on NUMA 1 may not directly send data from NUMA 0 MR** in the desired affinity path.
4. Therefore, daemon mode **must** allocate across NUMA, giving each NUMA local MR for the NIC on that NUMA.
5. Mooncake couples the two concerns in one `if` branch ([real_client.cpp:519](../../third_partys/Mooncake/mooncake-store/src/real_client.cpp#L519)): **"IPC server enabled => automatically allocate across NUMA"**.

Conversely, in non-daemon mode, the process is either pinned to one NUMA by outer `numactl` (Path A), or it runs on a single-NUMA node (standalone.yaml). Mooncake does not need to manage cross-NUMA allocation itself.

**Pathname convention**: `ipc_socket_path` is just a **path string used as an id**. The code uses **Linux abstract namespace** ([real_client.cpp:3987](../../third_partys/Mooncake/mooncake-store/src/real_client.cpp#L3987), writing into `&addr.sun_path[1]` with byte 0 as null), so it **does not create a real file**. Typical values include `/var/run/mooncake_client.sock` or `@mooncake_client_50052.sock` (Mooncake's own scheme in standalone_storage mode; see [store_py.cpp:1621](../../third_partys/Mooncake/mooncake-integration/store/store_py.cpp#L1621): `"@mooncake_client_" + port + ".sock"`). On the SGLang side, configure `MOONCAKE_CLIENT=host:port` or a unix path to connect.

**Steps** (`real_client.cpp:516-587`):
1. `GetNicNumaNodes()` reads NUMA nodes that have NICs from `TransferEngine::getLocalTopology()`, for example `[0, 1]`.
2. `allocate_buffer_numa_segments(total_size, numa_nodes, page_size)` ([utils.cpp:137-193](../../third_partys/Mooncake/mooncake-store/src/utils.cpp#L137)):
   ```cpp
   size_t n = numa_nodes.size();                        // 2
   size_t region_size = align_up(total_size / n, page_size);
   size_t map_size = region_size * n;

   // 1) reserve one contiguous VMA, no physical pages allocated yet
   void *ptr = mmap(nullptr, map_size, PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

   // 2) for each region in the VMA, call mbind(MPOL_BIND) to bind to the NUMA node
   for (size_t i = 0; i < n; ++i) {
       numa_bitmask_setbit(mask, numa_nodes[i]);
       char *region = (char*)ptr + i * region_size;
       mbind(region, region_size, MPOL_BIND, mask->maskp, mask->size, 0);
   }
   // 3) no prefault
   //    later ibv_reg_mr() triggers page faults, and physical pages land according to mbind
   ```
3. The result is **one contiguous virtual address space** that spans multiple NUMA nodes internally, while still looking like **one buffer** externally.
4. A **single `MountSegment`** reports the whole buffer to master. `seg_location = buildSegmentsLocation(...)` carries NUMA layout information so master can make NIC-affinity decisions later.
5. The process is not replicated, so **one daemon consumes all NUMA memory on the node**.

**Pros**: simpler YAML (single container); transparent to SGLang, which only connects through IPC.
**Cons**: one process crash loses all NUMA memory on the node; startup is slower because it must allocate and register one large VMA.

#### Quick decision table

| Scenario | Recommended approach |
|---|---|
| Production + need NUMA-level failure isolation + multiple NUMA nodes contribute memory | **Path A** (HA.yaml) |
| Want Mooncake cache to survive SGLang restarts independently | **Path B** (daemon + dummy client) |
| Single-NUMA node; no need to consider this complexity | single process + single container is sufficient (standalone.yaml) |
| Want host-level tooling such as monitoring and cgroup limits to control resources per NUMA | **Path A** |

### 3.5 How many segments does one container contribute?

Key source: [`client_service.h:642`](../../third_partys/Mooncake/mooncake-store/include/client_service.h#L642):
```cpp
std::unordered_map<UUID, Segment, boost::hash<UUID>> mounted_segments_;
```

**One Client can own N Segments**. The map type already shows this. N > 1 happens in these cases:

| Scenario | Trigger |
|---|---|
| `global_segment_size > max_mr_size` auto-split | default max_mr_size = **1 TiB** ([config.h:38](../../third_partys/Mooncake/mooncake-transfer-engine/include/config.h#L38) `0x10000000000`); splits only when exceeded |
| NUMA-segmented mode | single-process daemon mode + multiple NUMA NICs |
| User explicitly calls `MountSegment` multiple times | public API, for example mount host DRAM first and HBM later |
| Mixed memory types | hugepage / ascend / normal aligned_alloc have separate tracking |

**Actual deployment observation**:
- standalone.yaml: 60 GB < 1 TiB -> each container has **1 segment**. 4 pods x 1 container x 1 segment = 4 segments, total 240 GB.
- HA.yaml: 1000 GB < 1 TiB -> each container has **1 segment**. But 20 pods x 2 containers x 1 segment = **40 segments**, total ~32 TB (store=12x2x1000, store-prefill=6x2x500, store-decode=2x2x500).

Correct mental model: **1 process = 1 Client, and 1 Client owns a segment map that contains 1 entry in this deployment**.

### 3.6 `MC_ENABLE_DEST_DEVICE_AFFINITY=1`: two benefits at once

Both YAMLs enable it. Many people think this is only a "routing optimization" that selects the nearest NIC. In practice, it has **two stacked benefits**:
1. **Shorter path**: latency / bandwidth optimization through rail / NUMA alignment.
2. **QP count drops from O(N^2) to O(N)**: resource savings and more stable connections.

The second point is often missed, so this section expands it.

#### 3.6.1 Endpoint / QP is **created lazily**, keyed by `peer_nic_path`

[endpoint_store.cpp:49-76](../../third_partys/Mooncake/mooncake-transfer-engine/src/transport/rdma_transport/endpoint_store.cpp#L49):

```cpp
std::shared_ptr<RdmaEndPoint> FIFOEndpointStore::insertEndpoint(
    const std::string &peer_nic_path, RdmaContext *context) {
    if (endpoint_map_.find(peer_nic_path) != endpoint_map_.end()) {
        return endpoint_map_[peer_nic_path];             // reuse
    }
    auto endpoint = std::make_shared<RdmaEndPoint>(*context);
    endpoint->construct(context->cq(),
                        config.num_qp_per_ep,            // <- each endpoint constructs num_qp_per_ep QPs
                        ...);
    endpoint_map_[peer_nic_path] = endpoint;
}
```

**Semantics**:
- Endpoint is stored by `peer_nic_path` (= remote segment + remote NIC).
- Different `peer_nic_path` -> create a separate endpoint plus `num_qp_per_ep` QPs.
- Same key -> reuse the same endpoint.
- **Actual QP count = size of the visited `peer_nic_path` set x `num_qp_per_ep`**.

#### 3.6.2 The affinity switch changes `peer_nic_path` diversity

[topology.cpp:553-598](../../third_partys/Mooncake/mooncake-transfer-engine/src/topology.cpp#L553) `Topology::selectDevice`:

```cpp
// with hint (affinity ON):
if (!hint.empty()) {
    auto hca_idx = entry.getHcaIndex(std::string(hint));
    if (hca_idx != -1) return hca_idx;                   // <- deterministically return the matching NIC name
}
// without hint (affinity OFF):
rand_value = SimpleRandom::Get().next();                 // <- random
return entry.preferred_hca[rand_value % entry.preferred_hca.size()];
```

`worker_pool.cpp:113` decides where the hint comes from:
```cpp
auto hint = globalConfig().enable_dest_device_affinity
                ? context_.deviceName()                  // on: local NIC name
                : "";                                    // off: empty
```

#### 3.6.3 Concrete count comparison, with 8 local NICs x 8 remote NICs

| Metric | Affinity **OFF** (random) | Affinity **ON** (rail-aligned) |
|---|---|---|
| Maximum `peer_nic_path` diversity | 8 x 8 = **64** | 8 x 1 = **8** |
| Actual QPs, assuming `num_qp_per_ep=2` | **128** | **16** |
| endpoint_map peak | 64 | 8 |
| `FIFOEndpointStore` LRU eviction | frequent, may exceed `max_size_` | rare |
| first handshake (`MC_HANDSHAKE_PORT`) count | O(N^2) | O(N) |

**Reduction: total QP count is about 1/N**. With N=8, it drops from 128 to 16.

#### 3.6.4 Three downstream benefits

1. **Hardware resources**: CX-7 / BF3 HCAs have a total QP limit (`device_attr.max_qp`). Affinity lets the same budget cover more distinct cluster connections.
2. **Connection stability**: without affinity, `FIFOEndpointStore.max_size_` is easy to fill, causing endpoint eviction and rebuild on the next access, which **adds connection jitter and lowers throughput**.
3. **P2P handshake overhead**: first access to a new `peer_nic_path` uses `MC_HANDSHAKE_PORT=12001` to exchange GID / qp_num. Affinity reduces handshake count from O(N^2) to O(N).

Therefore, in HA YAML, `MC_ENABLE_DEST_DEVICE_AFFINITY=1` is **not an optional optimization**. It is **mandatory** for rail-aligned multi-NIC clusters. In standalone YAML with only 1 NIC, it makes no difference.

---

## § 4 · Tiered Store Design (HA Only)

HA.yaml splits store into 3 tiers. This is a fundamental architectural difference from standalone.

### 4.1 Configuration of the three store tiers

| Role | replicas | segment/container | Contribution | node label |
|---|---|---|---|---|
| `store` (dedicated) | 12 | 1000 GB | **24 TB** | `kvcache.ai/mooncake-store=true` |
| `store-prefill` (colocated prefill) | 6 | 500 GB | 6 TB | `...mooncake-store-prefill=true` |
| `store-decode` (colocated decode) | 2 | 500 GB | 2 TB | `...mooncake-store-decode=true` |
| **Total** | | | **~32 TB** | |

### 4.2 Design philosophy

**Why three tiers?** They are grouped by "how close storage is to the user":

- **Dedicated store nodes**: do not run inference. All node DRAM goes to Mooncake, so each pod contributes 2 TB (dual NUMA x 1 TB).
- **Colocated nodes (prefill/decode)**: most of the node DRAM is consumed by model weights, activations, and CPU offload, so only 1 TB/pod can be allocated to Mooncake.
- Benefit of colocation: **locality**. A decode node's own KV cache can live in Mooncake on the same node. GET can use same-node shared memory or NUMA-local RDMA loopback, which is an order of magnitude lower latency than cross-node access.

**Why should decode nodes also contribute storage?** A common misunderstanding is: "decode only reads KV from prefill, so it should not store anything." In reality:
- During decode, every generated token **appends one new KV row**. If local GPU HBM cannot hold it, it can be offloaded to same-node DRAM for local access.
- After the conversation round ends, KV remains on that node. If the user continues the conversation, a prefill node can read it back from the decode node for prefix-cache reuse.
- Without contributing storage, 2 TB of idle DRAM is wasted, reducing prefix-cache hit rate for the whole pool.

### 4.3 Remote RDMA DRAM vs local NVMe SSD: why Mooncake chooses a DRAM pool

| Tier | Bandwidth, sequential read | Latency, small block |
|---|---|---|
| Local DRAM | ~100 GB/s | ~80 ns |
| **Remote RDMA DRAM (200G IB)** | **~20-25 GB/s** | **~2-5 us** |
| Local NVMe SSD (Gen4) | ~7 GB/s | ~80 us |
| Local NVMe SSD (Gen5) | ~14 GB/s | ~60 us |

**In a cluster with high-speed IB, remote node DRAM is faster than local SSD**: latency is 10-40x better, and bandwidth is also higher. Mooncake therefore builds a distributed DRAM pool rather than an SSD-based tier. Local SSD only beats remote DRAM when the cluster does not have a high-speed network.

Full hierarchy:
```
local HBM > local DRAM same NUMA > local DRAM other NUMA > neighboring node DRAM in same rack (RDMA) > neighboring node DRAM across racks (RDMA) > local NVMe > remote SSD
```

Mooncake is an **L3 distributed DRAM pool**, filling the gap between "local DRAM is not enough" and "we must fall back to SSD".

### 4.3.1 Replica selection logic: the code has no explicit "memory > disk" priority

Question: if a key has both a **remote memory replica** and a **local SSD replica**, which one does the client choose?

**Answer: by default, it chooses the first COMPLETE replica in the list. Memory priority is implicitly guaranteed by write order.**

[client_service.cpp:2922-2935](../../third_partys/Mooncake/mooncake-store/src/client_service.cpp#L2922):

```cpp
ErrorCode Client::FindFirstCompleteReplica(
    const std::vector<Replica::Descriptor>& replica_list,
    Replica::Descriptor& replica) {
    for (size_t i = 0; i < replica_list.size(); ++i) {
        if (replica_list[i].status == ReplicaStatus::COMPLETE) {
            replica = replica_list[i];                // <- first COMPLETE, linear scan
            return ErrorCode::OK;
        }
    }
    return ErrorCode::INVALID_REPLICA;
}
```

**Key point**: `replica_list` order comes from the master's `VisitReplicas` traversal over internal `std::vector<Replica>` storage, i.e. **write order**. Mooncake's write flow:
1. Put writes memory replica first (PutStart -> PutEnd).
2. Background offload task later adds disk replica. See `NotifyOffloadSuccess`.

Therefore, the memory replica naturally appears earlier in the list, and `FindFirstCompleteReplica` naturally chooses memory first, **even if it is remote**. This matches the performance data in §4.3:

> remote RDMA DRAM (~25 GB/s, ~5 us) > local NVMe SSD (~7 GB/s, ~60 us)

### 4.3.2 Optional "locality optimization": GetPreferredReplica

[client_service.cpp:2937-2966](../../third_partys/Mooncake/mooncake-store/src/client_service.cpp#L2937) also has an **optional** function, which is not the default Get path:

```cpp
tl::expected<Replica::Descriptor, ErrorCode> Client::GetPreferredReplica(...) {
    for (const auto& rep : replica_list) {
        if (rep.is_memory_replica()) {
            if (local_endpoints.count(mem_desc.buffer_descriptor.transport_endpoint_)) {
                return rep;                         // <- prefer local memory replica
            }
        }
    }
    return replica_list[0];                         // <- fallback to first if no local one exists
}
```

With multiple memory replicas, this function prefers a "segment mounted by this client itself", but it **does not consider disk**, and it is **not the default Get path**.

---

## § 5 · Quick Answers to Common Questions

**Q: What happens if the master in standalone.yaml dies?**
A: The control plane stops, and all store data becomes temporarily inaccessible even though the data itself still exists. If the restarted master did not persist metadata, it loses all metadata, so **all previously stored KV is logically lost**. That is why standalone is only suitable for development.

**Q: What if etcd also dies in HA.yaml?**
A: The 3 masters lose their ability to elect a leader, and new requests fail. Already established data-path connections may keep working for a while, but not reliably. Therefore, etcd itself must also be deployed as a separate 3- or 5-node HA cluster.

**Q: Are more store replicas always better?**
A: No. More replicas increase master state-management pressure and create more RDMA connections. Size the replica count by actual required capacity in TB. A common starting point is 1-2 TB per node.

**Q: Why is the HA master's KV lease only 10 s, while standalone uses 5 minutes?**
A: A short lease prevents master memory from being occupied by zombie keys and is suitable for high churn. The cost is more frequent client renewal, which increases RPC pressure. In HA, there are more clients, and a short lease is safer. At standalone scale it does not matter much.

**Q: Which is better: "NUMA-segmented automatic splitting" or "external numactl"?**
A: For production, prefer external `numactl` plus multiple processes, which is the HA.yaml approach. It provides better failure isolation. NUMA-segmented mode is suitable for a compact deployment where one daemon process owns the whole node, but one process crash takes down the whole node's Mooncake memory.

---

## § 6 · Further Reading

- [**mooncake_k8s_deployment_advanced.md**](./mooncake_k8s_deployment_advanced.md) — companion document covering the three major "advanced deployment / operations" topics: **HA master details, etcd integration, and health probe strategy**
- [sglang_mooncake_env_config_deep_dive.md](./sglang_mooncake_env_config_deep_dive.md) — full view of the SGLang-side env variables for consuming Mooncake
- `k8s_mooncake_ha_deployment_notes.md` — higher-level HA deployment notes; referenced by the source doc but not present in this checkout
- Mooncake source: `vllm/third_partys/Mooncake/`
