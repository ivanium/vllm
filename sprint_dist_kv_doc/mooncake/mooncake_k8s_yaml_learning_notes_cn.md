---
title: Mooncake k8s 部署 YAML 学习笔记 (HA vs Standalone 对比 + 源码印证)
audience: 正在理解 Mooncake 部署 + 运行时机制的工程师
last_verified: 2026-04-23
sources:
  - vllm/mooncake-standalone.yaml           (单 master 极简部署)
  - vllm/mooncake_high_availablity.yaml     (HA + 分层 store 生产部署)
  - vllm/third_partys/Mooncake/             (Mooncake 源码, grep 验证)
related_docs:
  - mooncake_k8s_deployment_advanced_cn.md        (HA master / etcd / 健康探测 进阶)
  - sglang_mooncake_env_config_deep_dive_cn.md   (env 变量全景)
  - k8s_mooncake_ha_deployment_notes.md           (HA 部署上层说明)
---

# Mooncake k8s 部署 YAML 学习笔记

本文把两份真实 YAML (**standalone 极简** vs **HA 生产**) 放在一起逐层对比, 用源码印证每个配置项背后的运行时行为。目标: 读完这份笔记, 看到任何一份 Mooncake 部署 YAML 都能快速判断它选了哪些 tradeoff。

---

## § 1 · 两份 YAML 的对位看

### 1.1 骨架对比

| 维度 | standalone.yaml | HA.yaml |
|---|---|---|
| `kind` | `RoleBasedGroup` | `RoleBasedGroup` (同) |
| 角色数 | 2 (`master` + `store`) | **4** (`master` + `store` + `store-prefill` + `store-decode`) |
| master 副本 | **1** | **3** |
| master HA | 关 (无 `--enable_ha`) | 开 (`--enable_ha=true` + etcd) |
| master podAntiAffinity | 无 | 有 (`topologyKey: hostname`) |
| store 副本 | 4, 单 container | `store` 12 / `store-prefill` 6 / `store-decode` 2, **每 pod 双 container (双 NUMA)** |
| store segment 大小 | 60 GB | 1000 GB (专职) / 500 GB (混部) |
| store NIC | 1 张 (`mlx5_5`) | 8 张 (`mlx5_10..17`) |
| `hostNetwork` | master=false, store=true | 同 |
| `numactl --membind` | **无** | **有** (双 NUMA 绑定) |
| `MC_ENABLE_DEST_DEVICE_AFFINITY` | `"1"` | `"1"` |

### 1.2 一句话定位

- **standalone.yaml**: "开发/小规模测试" 模板。1 master + 几个 store, 能跑, 不抗单点故障, NIC 单张
- **HA.yaml**: "生产大规模" 模板。3 master + etcd 选主 + 分层 store + 双 NUMA 优化 + 8 NIC 满配

### 1.3 完整配置对比 (按 8 类别分组)

上面的骨架对比只挑了最显眼的差异。下面**把每个 YAML 字段 + 每个背后机制** 都放进对比表 —— 一个都不漏 (disk eviction 层除外, 后续文档处理)。每一行的 "机制一行" 列都指出对应 § 或源码位置。

#### A. 架构 & 调度

| 维度 | standalone | HA | 机制一行 |
|---|---|---|---|
| `kind` | `RoleBasedGroup` | 同 | **sgl-project/rbg** (github.com/sgl-project/rbg), apiVersion `workloads.x-k8s.io/v1alpha1`. 给"多角色系统"提供了原生 StatefulSet 没有的: 跨角色启动顺序、共享升级语义、跨角色 ready 检查 |
| master 副本 | 1 | **3 (奇数 quorum)** | HA 用奇数副本 + podAntiAffinity 分散到不同机器, 保证任 1 台挂掉仍有 quorum。详见 [advanced doc §1.2](./mooncake_k8s_deployment_advanced_cn.md) |
| master `podAntiAffinity` | 无 | `required + hostname` | `required` = 硬约束, 集群没足够满足 label 的节点时 pod 直接 Pending; `preferred` 是软约束 |
| `nodeAffinity` (按角色) | 无 | `required` + label match | 不同档 store 固定到对应 node pool |

#### B. 元数据 & HA 层

| 维度 | standalone | HA | 机制一行 |
|---|---|---|---|
| `MOONCAKE_TE_META_DATA_SERVER` | `P2PHANDSHAKE` | `P2PHANDSHAKE` | TransferEngine metadata 模式, **不是**走独立 metadata server。详见 §1.4 |
| `MOONCAKE_MASTER` 格式 | `host:port` | `etcd://...` | 后者让 client 通过 etcd 发现 leader。详见 [advanced doc §2](./mooncake_k8s_deployment_advanced_cn.md) |
| `--enable_ha` | 无 | `true` | master 启动时向 etcd 注册并参与 Raft 选主 |
| `--etcd_endpoints` | 无 | 指向外置 etcd 集群 | etcd 本身要**另外**部署 3/5 节点 HA 集群, **不在 Mooncake YAML 范围内** |
| `--rpc_address` | 默认 localhost | `$(POD_IP)` | 这个地址会写进 etcd 供 client 发现 leader |

#### C. 内存 & Buffer

| 维度 | standalone | HA | 机制一行 |
|---|---|---|---|
| `MOONCAKE_GLOBAL_SEGMENT_SIZE` | `60gb` | `1000gb` (专职) / `500gb` (混部) | 贡献给全局 pool 的 DRAM, 通过 `MountSegment` 注册给 master |
| `MOONCAKE_LOCAL_BUFFER_SIZE` | 默认 16 MB | `67108864` (64 MB) | **和上一项不是同一块内存**! 见 §1.5 |
| Client 本地 hot cache (L1.5) | 开 (默认) | 开 | **非 Mooncake replica** (master 不感知), 是 client 进程内的本地缓存; **CMS 频率准入** + LRU 淘汰 + 可选 shm 共享。详见新 §1.7 |
| Object replication factor | 1 (默认) | 1 (默认) | `ReplicateConfig` 可按对象指定 N 副本; 本 YAML 都没改 |
| Hugepage | 默认不开 | 默认不开 | `should_use_hugepage_`, 由内部配置或 env 控制 |
| `max_mr_size` (单 MR 上限) | 1 TiB (默认) | 1 TiB (默认) | `config.h:38` `0x10000000000`, 可 `MC_MAX_MR_SIZE` 覆盖 |

#### D. Eviction & Lease

| 维度 | standalone | HA | 机制一行 |
|---|---|---|---|
| `--eviction_high_watermark_ratio` | 默认 `0.95` | `0.9` (更激进) | 用量到此比例开始后台 evict。见 §1.6 |
| Eviction 算法 | **near-LRU** | 同 | `master_service.h:555` 注释 "BatchEvict evicts objects in a near-LRU way"。`eviction_strategy.h` 还定义了 `LRUEvictionStrategy` 和 `FIFOEvictionStrategy` 两个子类。**注意**: `SIEVE = 1` 是 `EndpointStoreType` (TransferEngine 的 endpoint 缓存), **不是** master eviction |
| `--default_kv_lease_ttl` (ms) | `300000` (5 min) | `10000` (10 s) | Mooncake 默认 `DEFAULT_DEFAULT_KV_LEASE_TTL = 5000 ms` (`types.h:84`) |
| Lease 续期机制 | 隐式 (Get/Put 自动刷新) | 同 | master 在每次 `Get` / `Put` 时调 `metadata.GrantLease(default_kv_lease_ttl_, ...)` 自动续。**没有独立的 heartbeat RPC**, 长期无访问的 key 到 TTL 就被 evict (详见 §1.6) |

#### E. 身份 & 端口

| 维度 | standalone | HA | 机制一行 |
|---|---|---|---|
| `MOONCAKE_LOCAL_HOSTNAME` 来源 | Downward API `status.podIP` | 同 | `hostNetwork: true` 下等于宿主机 IP, 对端 RDMA 回连要这个 |
| Client 身份 | `client_id_` UUID + `local_hostname_` (含 port) | 同 | UUID 构造时 `generate_uuid()`; hostname 在 `MountSegment` 时上报给 master |
| Segment 身份 | 每次 `MountSegment` 生成 `segment.id` UUID | 同 | **一个 Client 可持有 N 个 Segment** (`mounted_segments_` 是 map), 详见 §1.8 |
| AutoPortBinder 端口范围 | `[12300, 14300]` random | 同 | `utils.cpp:51-82`。bind() 占位互斥, 重试上限 20 次 (`MC_STORE_CLIENT_SETUP_RETRIES`) |
| 单 container 占的端口数 | 3~4 个 | 同 | 见 §1.9 |

#### F. NIC / NUMA / 亲和

| 维度 | standalone | HA | 机制一行 |
|---|---|---|---|
| `MOONCAKE_DEVICE` | `mlx5_5` (1 张) | `mlx5_10..17` (8 张) | 节点 IB NIC 列表, 逗号分隔;也支持 JSON dict 做 GPU→NIC 显式映射 |
| NUMA 绑定 | 无 | 外部 `numactl --membind=N` | 不走 Mooncake 内部 NUMA 切分 (那个只在 daemon 模式生效, 见 §4.4) |
| `MC_ENABLE_DEST_DEVICE_AFFINITY` | `"1"` | `"1"` | 发数据时优先选"离对端 GPU 最近的 NIC", rail-aligned 拓扑关键优化 |
| NIC-NUMA 拓扑发现 | `TransferEngine::getLocalTopology()` | 同 | 读 sysfs + `ibv_query_gid`, 构建 `cpu:N → preferred_hca[]` 矩阵 |
| `MC_MS_AUTO_DISC=1` | 未设 | 未设 | 设了会**强制覆盖** `MOONCAKE_DEVICE` 改走自动发现 |

#### G. 网络 & DNS

| 维度 | standalone | HA | 机制一行 |
|---|---|---|---|
| master `hostNetwork` | `false` | `false` | 控制面走 pod network |
| store `hostNetwork` | `true` | `true` | RDMA GID 绑物理 NIC 必需 |
| `dnsPolicy` | `ClusterFirstWithHostNet` | 同 | `hostNetwork` 下默认 DNS 走宿主机, 这行让 pod 仍能解析 `*.svc.cluster.local` |
| `dnsConfig.nameservers` | 自定义 `10.60.0.2/3` | 默认 | standalone 针对特殊 DNS 环境补了 nameserver |
| `/dev/infiniband` 挂载 | **显式** `hostPath` + `volumeMounts` | 依赖 `privileged: true` 隐式挂 | 显式方式更符合生产"最小权限"原则 |

#### H. Protocol & Metrics

(健康探测 probe 行已移至 [advanced doc §3](./mooncake_k8s_deployment_advanced_cn.md))

| 维度 | standalone | HA | 机制一行 |
|---|---|---|---|
| `MOONCAKE_PROTOCOL` | `rdma` | `rdma` | 代码还支持 `tcp / cxl / ascend / ubshmem / rpc_only`, 本 YAML 只用 rdma |
| `MC_TE_METRIC` | 未设 | `'true'` | 开后台线程**周期性打 glog INFO**: 吞吐 + 延迟直方图。**不是** Prometheus endpoint, 只是 log ([transfer_engine_impl.cpp:722-758](../../Mooncake/mooncake-transfer-engine/src/transfer_engine_impl.cpp#L722)) |
| C++ HTTP server (master `/metrics`, client `/health`) | 启 (默认) | 启 (默认) | **另一层**机制: master 默认 9003, client 默认 9300; Prometheus scrape 用这个, **和 `MC_TE_METRIC` 无关** |

### 1.4 `P2PHANDSHAKE` 是什么

[transfer_metadata.cpp:143-146](../../Mooncake/mooncake-transfer-engine/src/transfer_metadata.cpp#L143):
```cpp
if (conn_string == P2PHANDSHAKE) {
    p2p_handshake_mode_ = true;
    ...
}
```

- 这个字面字符串是 TransferEngine 的**"no central metadata server"** 模式开关
- 替代方案是 `http://host:8080/metadata` (独立 metadata service) 或 `etcd://...` (etcd-backed)
- 在 P2PHANDSHAKE 模式下, 两节点第一次通信时走**直接 RPC 握手**交换 RDMA endpoint / GID / qp_num 等信息, 不依赖共享元数据
- 握手端口默认 `12001`, 由 env `MC_HANDSHAKE_PORT` 覆盖 ([config.cpp:146](../../Mooncake/mooncake-transfer-engine/src/config.cpp#L146))
- `p2p_handshake_mode_` 布尔在 `transfer_metadata.cpp` 有 7 处分支使用, 通过 `HandShakePlugin` 插件机制 ([transfer_metadata.cpp:137](../../Mooncake/mooncake-transfer-engine/src/transfer_metadata.cpp#L137)) 实现发现/握手逻辑
- **生产推荐**: 少一个 metadata service pod / 少一个故障点, 代价是代码复杂度
- SGLang 侧必须保持 `MOONCAKE_TE_META_DATA_SERVER=P2PHANDSHAKE` 才能命中 TransferEngine 复用的 4-AND 条件 (见 env-config doc §4.5)

### 1.5 `registerLocalMemory` vs `MountSegment`:两种"内存注册"

这两个 API 名字像, 做的事**完全不同**。setup 流程里它们都会被调:

| API | 内存用途 | 大小来源 | 可见范围 |
|---|---|---|---|
| `registerLocalMemory(ptr, size, ...)` | **本 client Put/Get 时的 staging buffer** | `MOONCAKE_LOCAL_BUFFER_SIZE` (典型 16~64 MB) | 仅本 client 自己用, 不共享 |
| `MountSegment(ptr, size, protocol, location)` | **贡献给全局 pool 的 DRAM** | `MOONCAKE_GLOBAL_SEGMENT_SIZE` (GB~TB 级) | 注册进 master, 全集群 client 可访问 |

代码同时调两者 ([real_client.cpp:473](../../Mooncake/mooncake-store/src/real_client.cpp#L473) 和 [:581](../../Mooncake/mooncake-store/src/real_client.cpp#L581))。

**直接后果**: 即使 `global_segment_size=0` (纯消费者模式, 不贡献), 仍然会有 `local_buffer_size` 那块 buffer, 不会为 0。纯 store pod 可以把 `local_buffer_size` 设成 0 节省内存, 但 HA YAML 保守地给了 64 MB。

### 1.6 Lease + Eviction 组合语义

**Lease (租约)** — 隐式续期, 非主动 heartbeat:
- `Put(key, value)` 时 master 调 `metadata.GrantLease(default_kv_lease_ttl_, default_kv_soft_pin_ttl_)` 发放租约 ([master_service.cpp:456](../../Mooncake/mooncake-store/src/master_service.cpp))
- **每次 `Get` 也会 GrantLease 续期** ([master_service.cpp:770](../../Mooncake/mooncake-store/src/master_service.cpp#L770)), 相当于 LRU 访问即刷新
- `GetReplicaListResponse` 里带 `lease_ttl_ms` 返回给 client, client 可以拿这个计算 deadline, 但 **没有独立的 renew RPC**
- 如果一个 key 超过 `default_kv_lease_ttl` 没被访问 → `IsLeaseExpired()` = true → eviction 时允许被清 ([master_service.cpp:608, 1799, 1844, 1911](../../Mooncake/mooncake-store/src/master_service.cpp))
- HA 选 10s 短租约:未被访问的 key 10s 内失去保护, master 内存周转更快
- Standalone 选 5min:开发场景 key 少, 无所谓

**Eviction (淘汰)**:
- 两种触发:
  1. `PutStart` 因空间不足失败 → 立即触发 evict
  2. `EvictionThreadFunc` 后台线程 ([master_service.cpp:2210+](../../Mooncake/mooncake-store/src/master_service.cpp)) 每 `kEvictionThreadSleepMs` 毫秒轮询, 用量 ≥ `eviction_high_watermark_ratio` → 主动 evict
- 算法 = **near-LRU**: master_service.h:555 注释 "BatchEvict evicts objects in a near-LRU way"
- `eviction_strategy.h` 定义 `LRUEvictionStrategy` 和 `FIFOEvictionStrategy` 两个子类 (抽象基类 `EvictionStrategy`)
- ⚠️ **注意**: 代码里还有 `enum EndpointStoreType { FIFO = 0, SIEVE = 1 }` (`mooncake-transfer-engine/include/config.h:28-30`), 这是 **TransferEngine 的 endpoint 缓存枚举**, **不是** master 侧 KV eviction, 不要混淆

**⚠️ 关键语义:"Lease 过期 ≠ 立即删除,是优先删除"**

Lease 过期只让 key **变成 eviction 候选**, 不是立即被清。如果内存宽裕 (用量 < high_watermark), 过期的 key **可以继续存活任意久**, 直到 `EvictionThreadFunc` 真的被触发才删。

三层保护 ([master_service.cpp:3518-3583](../../Mooncake/mooncake-store/src/master_service.cpp#L3518)):

| 对象状态 | 何时可被 evict |
|---|---|
| `IsHardPinned()` = true | **永远不** evict |
| `IsSoftPinned(now)` = true | 仅"second pass"(激进模式)且 `allow_evict_soft_pinned_objects_=true` 时 |
| 普通 + lease 已过期 | 首选候选, 按 `lease_timeout` 最老优先 |
| 普通 + lease 未过期 | 本轮不会被 evict |

**BatchEvict 的两 pass 策略**:
- **First pass**: 只清"lease 过期 + 无 pin"的对象, 按 `std::nth_element` 挑最老的 N 个, 删到 `evict_ratio_target` 水位
- **Second pass**: 如果第一轮不够 (`evicted_count < evict_ratio_lowerbound`), 把未 evict 完的 `no_pin_objects` 和 (如果允许) `soft_pin_objects` 加入候选继续删

**实战含义**: HA YAML 设 `default_kv_lease_ttl=10000` (10s) **不代表 10 秒后 key 就没了**, 只代表 "10 秒没访问 → 进入候选池, 下次内存紧张时会优先被清"。想强行保证某个 key 不被清, 要用 `with_hard_pin` (Put 时设 `ReplicateConfig`)。

HA 用 `eviction_high_watermark_ratio=0.9` 而非默认 0.95 —— **越早 evict 越能避免因内存碎片导致的 `PutStart` 硬失败**, 代价是命中率略降。

### 1.7 Client 本地 hot cache (L1.5) 完整机制

Hot cache 是 Mooncake 里一个**非常容易被误解**的子系统 — 很多人把它和 Mooncake 的 memory replica 混为一谈。这一节讲清三件事:**它不是 replica / KV 怎么变热 / 命中怎么跳过 RDMA**。

#### 1.7.1 它不是 Mooncake replica, master 完全不感知

| 维度 | Mooncake replica (MEMORY/DISK) | Hot cache block |
|---|---|---|
| master 知道它存在吗? | 是 (走 `MountSegment` 注册) | **否** (本 client 私有, 从不上报) |
| 跨 client 可见? | 是 (其他 client 查 master 能发现) | **否** (进程私有) |
| 生命周期 | 租约 + eviction 管理 | client 进程退出即消失 |
| 出现在 `GetReplicaList` 返回里? | 是 | **不出现** |
| `ReplicaType` 枚举涵盖? | `MEMORY` / `DISK` / `LOCAL_DISK` | **无 HOT_CACHE 类型** |
| 容量控制 | `MOONCAKE_GLOBAL_SEGMENT_SIZE` | `MC_STORE_LOCAL_HOT_CACHE_SIZE` |

类比: **hot cache ≈ CDN edge cache / Linux page cache**, Mooncake 全局 pool ≈ 后端 object store。一个是"本地/私有",一个是"集群共享"。

#### 1.7.2 热化触发: CMS 频率准入, 不是一次就缓存

[client_service.h:522-530](../../Mooncake/mooncake-store/include/client_service.h#L522):

```cpp
bool ShouldAdmitToHotCache(const std::string& key, bool cache_used) {
    if (!(hot_cache_ && !cache_used)) return false;
    if (admission_sketch_ == nullptr) return true;              // 无 CMS: 每次都入
    return admission_sketch_->increment(key) >= admission_threshold_;
}
```

语义:
- **`cache_used=true`** (本次已从 hot cache 拿到): **不 re-promote**, 不动 CMS counter, 避免"热->更热"循环
- **`cache_used=false`** (本次从 remote 拉): increment **Count-Min Sketch** counter, 达到 `admission_threshold_` 才升级
- `admission_sketch_ == nullptr` (未启 CMS): 每次都准入

**核心设计**: "频繁访问的 key 才缓存", 不是"访问一次就缓存" — **天然过滤 one-shot key**, 防 cache 污染。

#### 1.7.3 完整热化路径

[client_service.cpp:811 + 3095-3116](../../Mooncake/mooncake-store/src/client_service.cpp#L811):

```
Client::Get(key)
  ├─ FindFirstCompleteReplica(replicas, &replica)
  ├─ if (hot_cache_ && replica.is_memory_replica())
  │    cache_used = RedirectToHotCache(key, replica)   ← 命中直接改 replica descriptor
  ├─ TransferRead(replica, slices)                     ← 命中 = 本地 memcpy; miss = RDMA
  │
  └─ if (ShouldAdmitToHotCache(key, cache_used))       ← miss 且 CMS 达阈值
        ProcessSlicesAsync(key, slices, replica)
          ├─ (skip if IsReplicaOnLocalMemory && !IsShm)
          │       ↑ 本地 segment 且非 shm 模式: 无必要 cache, 直接跳
          └─ for each slice:
               hot_cache_handler_->SubmitPutTask(key, slice)
                 ├─ TouchHotKey: 已在 cache 就只摸 LRU, 返回
                 ├─ 否则:
                 │    a. GetFreeBlock (LRU 满则淘汰尾部)
                 │    b. memcpy(block->addr, slice.ptr, slice.size)  ← 同步拷贝
                 │    c. task_queue_.push(task)
                 └─ worker 线程 async: task.hot_cache->PutHotKey(block)
                                            ← 异步: 只是插 LRU 索引
```

**关键观察**:
- memcpy **是同步** (发生在 Get 调用栈里), 不是异步 — 如果你担心 Get 延迟, 知道这点很重要
- "异步"的只是 LRU 索引插入, 不影响数据就位时机
- 本地 replica + 非 shm → **跳过**, 因为本地访问已经够快, 不值得占 cache 配额

#### 1.7.4 命中使用: RedirectToHotCache 改写 replica descriptor

[client_service.cpp:1148-1171](../../Mooncake/mooncake-store/src/client_service.cpp#L1148):

```cpp
bool Client::RedirectToHotCache(const std::string& key, Replica::Descriptor& replica) {
    HotMemBlock* blk = hot_cache_->GetHotKey(key);
    if (blk == nullptr) return false;

    mem_desc.buffer_descriptor.transport_endpoint_ = GetTransportEndpoint();  // 本 client
    mem_desc.buffer_descriptor.buffer_address_ = (uintptr_t)blk->addr;        // hot cache 地址
    return true;
}
```

**技巧**: 把 replica descriptor 里的 `transport_endpoint_` 改成 client 自己, `buffer_address_` 改成 hot cache block 的地址 — 后续 `TransferRead` 误以为这是一个"本地 memory replica", 走 local memcpy 路径, **没有 RDMA**。

`HotMemBlock` 有 `ref_count` ([local_hot_cache.h:24-33](../../Mooncake/mooncake-store/include/local_hot_cache.h#L24)), `GetHotKey`/`ReleaseHotKey` 配对使用, 防 block 被读时被 LRU evict 挪作他用。

#### 1.7.5 配置速查

| env | 作用 |
|---|---|
| `MC_STORE_LOCAL_HOT_CACHE_SIZE` | hot cache 总容量 (字节) |
| `MC_STORE_LOCAL_HOT_BLOCK_SIZE` | 单 block 尺寸 (默认 16MB) |
| `use_shm` 参数 | `LocalHotCache` 构造参数, 开启后 block 从 memfd 分配, **可被本节点 dummy client 共享访问** |

开 shm 模式是唯一让 hot cache 跨进程可见的方式, 且仅限同节点的 dummy client。

### 1.8 身份三元组:client_id / segment_id / local_hostname

```
一个 Client 进程 (例: NUMA0 container)
┌─────────────────────────────────────────────────┐
│ Client 对象                                      │
│   client_id_    = UUID-A  (构造时 generate)       │ ← 全局唯一 Client 身份
│   local_hostname_ = "10.1.2.3:12847"             │ ← 对外可达地址 (IP+AutoPortBinder port)
│   transfer_engine_ (1 个, 带独立 TE rpc port)      │
│                                                 │
│   mounted_segments_: map<UUID, Segment> {       │
│     UUID-α → { name="10.1.2.3:12847", base, size=931GB }   │
│     UUID-β → { name="10.1.2.3:12847", base, size=... }     │  ← 多 Segment 时
│     ...                                         │
│   }                                             │
└─────────────────────────────────────────────────┘
```

- `client_id_` 是 master 用来识别"数据 owner"的身份
- `local_hostname_` 是让别人能 **回连你** 的地址 (写进 `segment.name`)
- `segment.id` 是每次 `MountSegment` 的产物, **一个 Client 可有 N 个 Segment** (见 §3.5)

这份 YAML 默认 1 Client = 1 Segment (因为 931 GB < 1 TiB max_mr_size), 但源码层面是 1 对多关系。

### 1.9 端口全景:一个 store container 实际占几个端口

以 HA YAML 的 `store-numa0` container 为例:

| 端口来源 | 典型值 | 默认值 | 用途 | 谁会连 |
|---|---|---|---|---|
| Python `--port` | YAML 里 8099/8100 | argparse `default=8080` ([mooncake_store_service.py:301](../../Mooncake/mooncake-wheel/mooncake/mooncake_store_service.py#L301)) | REST API (应用层 PUT/GET HTTP) | 调试/测试工具 |
| `AutoPortBinder` 随机 | `[12300, 14300]` 某个 | 同 (`utils.h:376`) | **Client 身份端口**, 即 `local_hostname` 的 port 后缀 | 只 `bind()` 不 `listen()`, 纯占位互斥 |
| TransferEngine 握手 | 通常随机, 可固定 | `MC_HANDSHAKE_PORT=12001` ([config.h:46](../../Mooncake/mooncake-transfer-engine/include/config.h#L46)) | RDMA P2P handshake RPC | 对端 client 第一次通信时握手 |
| C++ client HTTP server | `FLAGS_http_port` | **`9300`** ([real_client.cpp:38](../../Mooncake/mooncake-store/src/real_client.cpp#L38) `DEFINE_int32(http_port, 9300, ...)`) | `/health` `/metrics` Prometheus endpoint | k8s probe / 监控 |
| (对比) master metrics port | YAML 里 9003 | **`9003`** ([master_config.h:262](../../Mooncake/mooncake-store/include/master_config.h#L262) `uint16_t http_port = 9003`) | master 的 metrics, 非 client | Prometheus |

所以 HA YAML 一个双 NUMA pod 总共占 ~8 个宿主机端口 (2 个 container × 4 端口), 因为 `hostNetwork: true` 全在宿主机网卡上。这也是 AutoPortBinder 进程间互斥的必要性。

### 1.10 `MC_*` env 家族速查

全部是 Mooncake C++ 侧直接读的, **不经** SGLang Python:

| env | 默认值 | 作用 | standalone | HA |
|---|---|---|---|---|
| `MC_TE_METRIC` | 未设 | 设 `1`/`true`/`yes`/`on` 后开后台线程**周期打 glog INFO 日志**(吞吐 MB/s + latency histogram),**不是** Prometheus endpoint。见 [transfer_engine_impl.cpp:722-758](../../Mooncake/mooncake-transfer-engine/src/transfer_engine_impl.cpp#L722) | — | `true` |
| `MC_TE_METRIC_INTERVAL_SECONDS` | `5` | 上面那个日志的打印间隔秒数 | — | 默认 |
| `MC_ENABLE_DEST_DEVICE_AFFINITY` | 未设 | 发数据时选离目标最近的 NIC | `1` | `1` |
| `MC_MS_AUTO_DISC` | 0 | 1 = **强制覆盖** `MOONCAKE_DEVICE` 走自动发现 | 未设 | 未设 |
| `MC_MAX_MR_SIZE` | 1 TiB | 单 MR 最大尺寸, 超过在 setup while 循环切多 Segment | 默认 | 默认 |
| `MC_HANDSHAKE_PORT` | `12001` | P2PHANDSHAKE 的 RPC 端口 | 默认 | 默认 |
| `MC_STORE_CLIENT_SETUP_RETRIES` | `20` | AutoPortBinder 失败重试上限 ([real_client.cpp:419](../../Mooncake/mooncake-store/src/real_client.cpp#L419)) | 默认 | 默认 |
| `MC_STORE_CLIENT_METRIC` | **enabled** | client 侧 metric 开关, 设 `0/false` 关闭 ([client_metric.h:672](../../Mooncake/mooncake-store/include/client_metric.h#L672)) | 默认 | 默认 |
| `MC_STORE_CLIENT_METRIC_INTERVAL` | `0` | client metric 上报间隔秒; 0 = 只采不报 | 默认 | 默认 |
| `MC_STORE_CLUSTER_ID` | `"mooncake"` | HA 模式下 etcd key 的 cluster namespace ([etcd_leader_coordinator.cpp:368](../../Mooncake/mooncake-store/src/ha/leadership/backends/etcd/etcd_leader_coordinator.cpp#L368)) | — | 默认 |
| `MC_STORE_USE_HUGEPAGE` | 未设 | 启用 hugepage 分配 ([utils.h:285](../../Mooncake/mooncake-store/include/utils.h#L285)) | 未设 | 未设 |
| `MC_STORE_HUGEPAGE_SIZE` | — | hugepage 尺寸 | — | — |
| `MC_STORE_LOCAL_HOT_CACHE_SIZE` | — | Client 本地 hot cache 字节数 | — | — |
| `MC_STORE_LOCAL_HOT_BLOCK_SIZE` | — | hot cache 单 block 字节数 | — | — |
| `MC_RPC_PROTOCOL` | — | RPC 协议选择 ([master_client.h:64](../../Mooncake/mooncake-store/include/master_client.h#L64)) | — | — |
| `MC_USE_IPV6` | 未设 | IPv6 支持开关 | — | — |
| `MC_RETRY_CNT` | `9` | 通用 RDMA op 重试次数 ([config.cpp:198](../../Mooncake/mooncake-transfer-engine/src/config.cpp#L198)) | 默认 | 默认 |
| `MC_CXL_DEV_SIZE` | 未设 | CXL 模式设备字节数 (不设 + protocol=cxl 会 FATAL) | — | — |
| `MC_MIN_REG_SIZE` | — | EIC/Barex 路径最小注册块 (`eic_max_block_size`) | — | — |
| `MC_DISABLE_METACACHE` | 未设 | 关闭 TE metadata cache (仅调试) | — | — |

---

## § 2 · RDMA 必备的运行时配置

这一节讲 **所有 Mooncake k8s 部署都要理解的 RDMA 底层配置** (YAML 里看起来像样板, 但缺一个就起不来)。

> RBG operator 结构 / Downward API 等 k8s 高阶机制已移至 [advanced doc §4 部署基础设施](./mooncake_k8s_deployment_advanced_cn.md)

### 2.1 为什么 store 要 `hostNetwork: true` 而 master 不要

| 角色 | `hostNetwork` | 原因 |
|---|---|---|
| master | `false` | 控制面, 不做 RDMA 数据传输, pod network 就够; 用 pod IP 注册到 etcd |
| store | `true` | **RDMA GID 绑物理网卡**, pod network (CNI overlay) 跑不了 RDMA |

配套必须加 `dnsPolicy: ClusterFirstWithHostNet`: hostNetwork 下默认 DNS 走宿主机, 这行让 pod 仍能解析 `*.svc.cluster.local` (比如 etcd / master 的 Service DNS)。

### 2.2 RDMA 必备的"四件套"

两份 YAML 的 store 容器都有这组配置:

```yaml
command:
- sh -c
- |
    ulimit -n 1048576        # 大量 fd
    ulimit -l unlimited      # memlock: RDMA MR 必须 pin 住不换出
securityContext:
  privileged: true           # 访问 /dev/infiniband/* 设备
  capabilities:
    add: [IPC_LOCK, SYS_RESOURCE]
    # IPC_LOCK     → 允许 mlock / ibv_reg_mr
    # SYS_RESOURCE → 允许提升 rlimit 上限
```

**四件套缺一个 RDMA 就挂**, 错误信息往往是 `ibv_reg_mr: Cannot allocate memory`。可以直接当模板套。

standalone.yaml 额外用了 `hostPath: /dev/infiniband` 挂载 + `volumeMounts` (更显式), HA.yaml 靠 `privileged: true` 隐式挂载。前者更符合生产"最小权限"原则。

---

## § 3 · Store 角色:单 NUMA vs 双 NUMA

这是两份 YAML 最大的结构差异。

### 3.1 Standalone: 单 container 单 NUMA

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
          value: "mlx5_5"       # ← 1 张 NIC
```

- 4 个 pod, 每 pod 1 个 container
- 每个 container 贡献 60 GB, 总池 240 GB
- 走 1 张 NIC, 不关心 NUMA

### 3.2 HA: 每 pod 双 container + `numactl` 绑 NUMA

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
  env: (同上 env)
```

### 3.3 双 NUMA 双 container 的机制 (源码印证)

每个 container 独立跑 `mooncake_store_service`, 独立调 `store.setup(...)`, 在 C++ 侧:

**第 1 步 — Python 入口** ([mooncake_store_service.py:108-117](../../Mooncake/mooncake-wheel/mooncake/mooncake_store_service.py#L108)):
```python
store = MooncakeDistributedStore()
ret = store.setup(
    local_hostname,          # = "10.1.2.3" (status.podIP, 没带端口!)
    metadata_server,         # "P2PHANDSHAKE"
    global_segment_size,     # 1000 * 10^9 (字节)
    local_buffer_size,
    protocol,                # "rdma"
    device_name,             # 8 张 NIC
    master_server_address,   # "etcd://..."
)
```

**第 2 步 — C++ AutoPortBinder** ([real_client.cpp:400-461](../../Mooncake/mooncake-store/src/real_client.cpp#L400)):
```cpp
if (colon_pos == npos) {    // 没带端口 → 自动随机分配
    for (retry < 20) {
        port_binder_ = std::make_unique<AutoPortBinder>();  // 在 [12300, 14300] 随机
        this->local_hostname = hostname + ":" + to_string(port);
        // 例如 "10.1.2.3:12847"
        ...
    }
}
```

两个 container 的 `status.podIP` 一样, 但 **AutoPortBinder 各自随机**, 所以最终 `local_hostname` 不同:
- container 1: `"10.1.2.3:12847"` + `client_id_` UUID-A
- container 2: `"10.1.2.3:14051"` + `client_id_` UUID-B

master 视角看到的是 **两个完全独立的 Client**, 身份互不冲突。

**第 3 步 — MountSegment** ([real_client.cpp:534-587](../../Mooncake/mooncake-store/src/real_client.cpp#L534)):
```cpp
while (global_segment_size > 0) {
    size_t segment_size = std::min(global_segment_size, max_mr_size);
    ...
    void *ptr = allocate_buffer_allocator_memory(segment_size, this->protocol);
    // ← 普通 aligned_alloc. 外部 numactl --membind=N 已经把 NUMA 钉死.
    client_->MountSegment(ptr, mapped_size, protocol, seg_location);
}
```

**注意**: Mooncake 代码本身对 NUMA 无感, NUMA 分离完全靠**外部 `numactl`**。优雅又可靠。

### 3.4 两种 NUMA setup 方式:多进程 + numactl vs 单进程内部切分

Mooncake 代码同时支持**两种** NUMA setup 路径, 选哪个由 **`ipc_socket_path_` 是否为空** 决定 ([real_client.cpp:519](../../Mooncake/mooncake-store/src/real_client.cpp#L519)):

```cpp
if (!ipc_socket_path_.empty() && protocol == "rdma") {  // ← 判断关键
    seg_numa_nodes = client_->GetNicNumaNodes();
    if (seg_numa_nodes.size() > 1) {
        // 走内部 NUMA 切分
    }
}
```

两条路径的触发条件互斥, 细节如下。

#### 路径 A — 多进程 + 外部 `numactl` (HA.yaml 选这个)

**触发**: `mooncake_store_service` Python 入口走 `setup(...)` pybind 调用, **硬编码 `ipc_socket_path=""`** ([store_py.cpp:1572-1575](../../Mooncake/mooncake-integration/store/store_py.cpp#L1572))。

**步骤**:
1. YAML 里每个 NUMA 开一个 container, command 用 `numactl --membind=N`:
   ```yaml
   - command:
     - sh -c
     - "numactl --membind=0 python3 -m mooncake.mooncake_store_service --port=8099"
   - command:
     - sh -c
     - "numactl --membind=1 python3 -m mooncake.mooncake_store_service --port=8100"
   ```
2. 每个进程独立起一个 `MooncakeDistributedStore().setup(...)`
3. 进程在 `AutoPortBinder` 随机选一个 `[12300, 14300]` 端口做身份 (见 §1.9)
4. 在 `setup_internal` 的 while 循环里 (`ipc_socket_path_.empty() == true`), 直接走**普通 `aligned_alloc`** 分配 `global_segment_size` 字节 ([real_client.cpp:561](../../Mooncake/mooncake-store/src/real_client.cpp#L561))
5. 因为外层 `numactl --membind=N` 设了进程级 memory policy, 内核的 `aligned_alloc` 调 `mmap` 时**所有页都 fault 到 NUMA N** —— Mooncake 代码完全无感
6. `MountSegment` 上报给 master 一整块, seg_location 用默认 `kWildcardLocation`

**优点**: Mooncake 代码逻辑简单 + **进程级故障隔离** (NUMA0 崩了, NUMA1 还在工作)
**缺点**: YAML 多一层 container, 每个 container 要单独配 env/探针

#### 路径 B — 单进程内部 NUMA 切分 (daemon 模式, HA.yaml **没用**)

**触发**: 需要同时满足:
- 用 `setup()` 的 **ConfigDict 重载** ([store_py.cpp:1582-1608](../../Mooncake/mooncake-integration/store/store_py.cpp#L1582)) 且 config 里传了 `ipc_socket_path: /some/path`
- `protocol == "rdma"`
- 本节点有 **多个 NUMA 都挂着 NIC** (`client_->GetNicNumaNodes().size() > 1`)

这个场景对应 "把 `mooncake_client` / `mooncake_store_service` 做成 **standalone daemon**, 然后 SGLang 用 `MOONCAKE_STANDALONE_STORAGE=1` + `MOONCAKE_CLIENT=unix:/some/path` 作为 dummy client 接入"。

##### Daemon 模式从启动到 master 注册的流程

```
  setup(ipc_socket_path=..., protocol=rdma)
              │
              ▼
  ┌───────────────────────────────────────────┐
  │  1. 探 NUMA 拓扑                           │
  │     GetNicNumaNodes() → [0, 1]            │
  └───────────────────────────────────────────┘
              │
              ▼
  ┌───────────────────────────────────────────┐
  │  2. 分配连续 VMA + 逐 region mbind          │
  │     mmap(size) → 一块 VMA                  │
  │     region[0] → mbind(NUMA 0)             │
  │     region[1] → mbind(NUMA 1)             │
  │     (还没物理页, 只是 policy)              │
  └───────────────────────────────────────────┘
              │
              ▼
  ┌───────────────────────────────────────────┐
  │  3. ibv_reg_mr 注册 MR                     │
  │     → page fault 按 mbind 落到对应 NUMA   │
  │     → 得到 NUMA-aware 的已注册内存         │
  └───────────────────────────────────────────┘
              │
              ▼
  ┌───────────────────────────────────────────┐
  │  4. MountSegment 上报 master               │
  │     Segment {                              │
  │       base, size,                          │
  │       location = "segments:4096:0,1"  ←   │
  │       (编码 NUMA 布局, 供 reader 用)       │
  │     }                                      │
  └───────────────────────────────────────────┘
              │
              ▼
  ┌───────────────────────────────────────────┐
  │  5. start_ipc_server                       │
  │     UDS listen @ ipc_socket_path           │
  │     等 dummy client 接入                   │
  └───────────────────────────────────────────┘
```

**核心要点**:
- **VMA 连续, 物理页分 NUMA**: 用户代码看到的是一块普通 buffer, 内核的 mempolicy 把物理页按 region 分散到不同 NUMA
- **location 字符串传递 NUMA 元信息**: `"segments:4096:0,1"` 编码进 `Segment.location`, master 持久化, reader 通过 `resolveSegmentsLocation(offset)` 算出本次访问落在哪个 NUMA, 选对应的 NIC
- **和普通模式 (路径 A) 的唯一差别**: 多了"探 NUMA + mbind + location 编码"这一串, 其它流程完全一样

##### 为什么 `ipc_socket_path` 非空才触发 NUMA 切分?

`ipc_socket_path` 不是 NUMA 切分的前提, **它是整个"daemon 模式"的触发器**, NUMA 切分只是 daemon 模式下顺带启用的一个子行为。

**`ipc_socket_path` 非空做的事** ([real_client.cpp:762-768](../../Mooncake/mooncake-store/src/real_client.cpp#L762)):
```cpp
if (!ipc_socket_path_.empty()) {
    start_ipc_server();      // 启动 Unix Domain Socket server 线程
    LOG(INFO) << "Starting IPC server at " << ipc_socket_path_;
}
```

启动后 [`ipc_server_func`](../../Mooncake/mooncake-store/src/real_client.cpp#L3976) 会 `bind` + `listen` 在一个 UDS (abstract namespace) 上, 接受本节点 dummy client 连进来做共享内存传输。

**因果链**:
1. 开 IPC server = "我是 daemon, 服务本节点所有 dummy client"
2. 本节点 dummy client 可能挂在**任何 NUMA** 的 GPU 上
3. 如果 daemon 内存只落在 NUMA 0, 挂在 NUMA 1 的 client 访问就跨 QPI; 更严重的是 **挂在 NUMA 1 的 NIC 没法直接发 NUMA 0 的 MR 数据** (需要 MR-NIC NUMA 亲和)
4. 所以 daemon 模式下 **必须** 跨 NUMA 分配, 让每个 NUMA 都有**本地 MR** 供本 NUMA 的 NIC 使用
5. Mooncake 把这两件事耦合在同一个 `if` 分支 ([real_client.cpp:519](../../Mooncake/mooncake-store/src/real_client.cpp#L519)): **"开了 IPC server ⇒ 自动跨 NUMA 分配"**

反过来: 非 daemon 模式下, 进程要么被外层 `numactl` 绑死单个 NUMA (路径 A), 要么就是单 NUMA 节点 (standalone.yaml), Mooncake 没必要自己折腾跨 NUMA。

**路径名约定**: `ipc_socket_path` 的值本身只是一个**路径字符串**做 id, 因为代码用 **Linux abstract namespace** ([real_client.cpp:3987](../../Mooncake/mooncake-store/src/real_client.cpp#L3987) `&addr.sun_path[1]` 写路径, 第 0 字节是 null), **不会真建文件**。典型值如 `/var/run/mooncake_client.sock` 或 `@mooncake_client_50052.sock` (Mooncake 自己在 standalone_storage 模式下用的 scheme, 见 [store_py.cpp:1621](../../Mooncake/mooncake-integration/store/store_py.cpp#L1621) `"@mooncake_client_" + port + ".sock"`)。SGLang 侧配 `MOONCAKE_CLIENT=host:port` 或 unix 路径连进来即可。

**步骤** (`real_client.cpp:516-587`):
1. `GetNicNumaNodes()` 从 `TransferEngine::getLocalTopology()` 里读出有 NIC 的 NUMA 列表, 如 `[0, 1]`
2. `allocate_buffer_numa_segments(total_size, numa_nodes, page_size)` ([utils.cpp:137-193](../../Mooncake/mooncake-store/src/utils.cpp#L137)):
   ```cpp
   size_t n = numa_nodes.size();                        // 2
   size_t region_size = align_up(total_size / n, page_size);
   size_t map_size = region_size * n;

   // 1) reserve 一块连续的 VMA, 不分配物理页
   void *ptr = mmap(nullptr, map_size, PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

   // 2) 对 VMA 里的每个 region, 调 mbind(MPOL_BIND) 绑到对应 NUMA
   for (size_t i = 0; i < n; ++i) {
       numa_bitmask_setbit(mask, numa_nodes[i]);
       char *region = (char*)ptr + i * region_size;
       mbind(region, region_size, MPOL_BIND, mask->maskp, mask->size, 0);
   }
   // 3) 不预分配物理页 (no prefault)
   //    后续 ibv_reg_mr() 会触发 page fault, 物理页按 mbind 策略落到对应 NUMA
   ```
3. 结果是 **一块连续的虚拟地址空间**, 内部跨多个 NUMA node, 但对外看仍是**单个 buffer**
4. **单次 `MountSegment`** 把这整块上报给 master, `seg_location = buildSegmentsLocation(...)` 带 NUMA 布局信息, master 后续可以做 NIC affinity 决策
5. 进程本身没开多份, 所以**一个 daemon 吃下本节点所有 NUMA 的内存**

**优点**: YAML 简单 (单 container); 对 SGLang 透明 (通过 IPC 连就行)
**缺点**: 单进程崩 = 本节点所有 NUMA 都掉; 启动慢 (要分配 + 注册整个大 VMA)

#### 快速决策表

| 场景 | 推荐方案 |
|---|---|
| 生产 + 要求 NUMA 级故障隔离 + 多 NUMA 都贡献 | **路径 A** (HA.yaml) |
| 想让 Mooncake cache 独立于 SGLang 重启存活 | **路径 B** (daemon + dummy client) |
| 单 NUMA 节点, 无需考虑这些 | 单进程 + 单 container 即可 (standalone.yaml) |
| 想用 host-level 工具 (monitoring, cgroup limit) 控制每个 NUMA 资源 | **路径 A** |

### 3.5 单个 container 贡献几个 segment?

关键: [`client_service.h:642`](../../Mooncake/mooncake-store/include/client_service.h#L642):
```cpp
std::unordered_map<UUID, Segment, boost::hash<UUID>> mounted_segments_;
```

**一个 Client 可以持有 N 个 Segment** (map 类型就说明了这点)。何时会 N > 1:

| 场景 | 触发条件 |
|---|---|
| `global_segment_size > max_mr_size` 自动切分 | 默认 max_mr_size = **1 TiB** ([config.h:38](../../Mooncake/mooncake-transfer-engine/include/config.h#L38) `0x10000000000`), 超过才切 |
| NUMA-segmented 模式 | 单进程 daemon 模式 + 多 NUMA NIC |
| 用户显式多次调用 `MountSegment` | API 公开 (e.g. 先 mount host DRAM 再 mount HBM) |
| 多种内存类型混合 | hugepage / ascend / 普通 aligned_alloc 各有追踪 |

**实际部署观察**:
- standalone.yaml: 60 GB < 1 TiB → 每 container **1 个 segment**。4 pod × 1 container × 1 segment = 4 segment, 总 240 GB
- HA.yaml: 1000 GB < 1 TiB → 每 container **1 个 segment**。但 20 pod × 2 container × 1 segment = **40 个 segment**, 总 ~32 TB (按 store=12×2×1000, store-prefill=6×2×500, store-decode=2×2×500 计)

正确心智模型: **1 进程 = 1 Client, 1 Client 持有一个 segment map, 默认填 1 条**。

### 3.6 `MC_ENABLE_DEST_DEVICE_AFFINITY=1`:双重收益

两份 YAML 都开了。很多人以为这只是"路由优化"(选最近 NIC), 实际上它有**两个叠加收益**:
1. **路径更短**: 延迟/带宽优化 (对齐 rail / NUMA)
2. **QP 总数从 O(N²) 降到 O(N)**: 资源节省 + 连接稳定

下面展开第 2 点, 因为它往往被忽略。

#### 3.6.1 Endpoint / QP 是**懒创建**的, 按 `peer_nic_path` 为 key

[endpoint_store.cpp:49-76](../../Mooncake/mooncake-transfer-engine/src/transport/rdma_transport/endpoint_store.cpp#L49):

```cpp
std::shared_ptr<RdmaEndPoint> FIFOEndpointStore::insertEndpoint(
    const std::string &peer_nic_path, RdmaContext *context) {
    if (endpoint_map_.find(peer_nic_path) != endpoint_map_.end()) {
        return endpoint_map_[peer_nic_path];             // 复用
    }
    auto endpoint = std::make_shared<RdmaEndPoint>(*context);
    endpoint->construct(context->cq(),
                        config.num_qp_per_ep,            // ← 每个 endpoint 构造 num_qp_per_ep 个 QP
                        ...);
    endpoint_map_[peer_nic_path] = endpoint;
}
```

**语义**:
- Endpoint 以 `peer_nic_path` (= 对端 segment + 对端 NIC) 为 key 存放
- 不同 `peer_nic_path` → 各自建 endpoint + `num_qp_per_ep` 个 QP
- 相同 → 复用同一 endpoint
- **QP 实际数 = "被访问过的 peer_nic_path 集合大小" × `num_qp_per_ep`**

#### 3.6.2 Affinity 开关改变 `peer_nic_path` 多样性

[topology.cpp:553-598](../../Mooncake/mooncake-transfer-engine/src/topology.cpp#L553) `Topology::selectDevice`:

```cpp
// 有 hint (affinity ON):
if (!hint.empty()) {
    auto hca_idx = entry.getHcaIndex(std::string(hint));
    if (hca_idx != -1) return hca_idx;                   // ← 确定性返回同名 NIC
}
// 无 hint (affinity OFF):
rand_value = SimpleRandom::Get().next();                 // ← 随机
return entry.preferred_hca[rand_value % entry.preferred_hca.size()];
```

而 `worker_pool.cpp:113` 决定 hint 的来源:
```cpp
auto hint = globalConfig().enable_dest_device_affinity
                ? context_.deviceName()                  // 开: 本端 NIC 名
                : "";                                    // 关: 空
```

#### 3.6.3 具体数量对比 (以本端 8 NIC × 对端 8 NIC 为例)

| 指标 | Affinity **OFF** (随机) | Affinity **ON** (rail-aligned) |
|---|---|---|
| `peer_nic_path` 最大多样性 | 8 × 8 = **64** | 8 × 1 = **8** |
| 实际 QP 数 (假设 `num_qp_per_ep=2`) | **128** | **16** |
| endpoint_map 峰值 | 64 | 8 |
| `FIFOEndpointStore` LRU 淘汰 | 频繁 (可能超 `max_size_`) | 稀有 |
| 首次握手 (`MC_HANDSHAKE_PORT`) 次数 | O(N²) | O(N) |

**降幅: 总 QP 数约 1/N**, N=8 时从 128 降到 16。

#### 3.6.4 三级连锁收益

1. **硬件资源**: CX-7 / BF3 的 HCA 有 QP 总数上限 (`device_attr.max_qp`), affinity 让预算里容纳更多不同集群连接
2. **连接稳定**: 无 affinity 时 `FIFOEndpointStore` 的 `max_size_` 容易被打满, 触发 endpoint evict + 下次重建 → **连接抖动拉低吞吐**
3. **P2P 握手开销**: 首次访问新 `peer_nic_path` 要走 `MC_HANDSHAKE_PORT=12001` 交换 GID / qp_num, affinity 把握手次数从 O(N²) 压到 O(N)

所以 HA YAML 开 `MC_ENABLE_DEST_DEVICE_AFFINITY=1` **不是可有可无的优化**,而是 rail-aligned 多 NIC 集群里**必开**的一项。standalone YAML 只有 1 NIC 时开不开都一样。

---

## § 4 · 分层 store 设计 (仅 HA)

HA.yaml 把 store 分成 3 档, 这是和 standalone 根本性的架构差异。

### 4.1 三档 store 的配置

| 角色 | replicas | segment/container | 贡献 | node 标签 |
|---|---|---|---|---|
| `store` (专职) | 12 | 1000 GB | **24 TB** | `kvcache.ai/mooncake-store=true` |
| `store-prefill` (混部 prefill) | 6 | 500 GB | 6 TB | `...mooncake-store-prefill=true` |
| `store-decode` (混部 decode) | 2 | 500 GB | 2 TB | `...mooncake-store-decode=true` |
| **总计** | | | **~32 TB** | |

### 4.2 设计哲学

**为什么要分 3 档?** 按"离使用者多近"分层:

- **专职 store 节点**: 不跑推理, 节点 DRAM 全部给 Mooncake → 每 pod 2 TB (双 NUMA × 1 TB)
- **混部节点 (prefill/decode)**: 节点 DRAM 大部分被模型权重 + activation + CPU offload 占掉, 只能挤出 1 TB/pod
- 混部的好处: **就近访问**。decode 节点自己的 KV cache 存在本节点 Mooncake 里, GET 走本机共享内存或 NUMA-local RDMA loopback, 延迟比跨节点低一个数量级

**为什么 decode 节点也要贡献存储?** 常见误解: "decode 只从 prefill 拿 KV, 不应该存东西"。实际上:
- decode 过程每生成 1 token 就**追加 1 行新 KV**, 本节点 GPU HBM 可能装不下 → offload 到本节点 DRAM (就近访问)
- 本轮对话结束后 KV 留在本节点, 下一轮用户继续对话时, prefill 节点可以从这个 decode 节点读回来复用 (prefix cache 命中)
- 不贡献就浪费了 2 TB 闲置 DRAM, 整个 pool 可缓存的 prefix 命中率下降

### 4.3 Remote RDMA DRAM vs Local NVMe SSD: 为什么 Mooncake 选 DRAM pool

| 层级 | 带宽 (顺序读) | 延迟 (小块) |
|---|---|---|
| Local DRAM | ~100 GB/s | ~80 ns |
| **Remote RDMA DRAM (200G IB)** | **~20-25 GB/s** | **~2-5 μs** |
| Local NVMe SSD (Gen4) | ~7 GB/s | ~80 μs |
| Local NVMe SSD (Gen5) | ~14 GB/s | ~60 μs |

**在有高速 IB 的集群里, 远端节点 DRAM 比本地 SSD 更快** (延迟快 10-40×, 带宽也更高)。所以 Mooncake 堆 DRAM 成分布式池, 而不是做 SSD-based tier。只有集群没有高速网络时 local SSD 才反超 remote DRAM。

完整层级:
```
local HBM > local DRAM 本 NUMA > local DRAM 另一 NUMA > 同机架邻居 DRAM (RDMA) > 跨机架邻居 DRAM (RDMA) > local NVMe > remote SSD
```

Mooncake 的定位是 **L3 分布式 DRAM pool**, 填补 "local DRAM 不够, 但还没到必须上 SSD" 的空缺。

### 4.3.1 Replica 选择逻辑:代码里没有显式"memory > disk"优先级

问题: 如果一个 key 同时有 **remote memory replica** 和 **local SSD replica**, client 会选哪个?

**答: 默认取 list 里第一个 COMPLETE 的,靠写入顺序隐式保证 memory 优先**。

[client_service.cpp:2922-2935](../../Mooncake/mooncake-store/src/client_service.cpp#L2922):

```cpp
ErrorCode Client::FindFirstCompleteReplica(
    const std::vector<Replica::Descriptor>& replica_list,
    Replica::Descriptor& replica) {
    for (size_t i = 0; i < replica_list.size(); ++i) {
        if (replica_list[i].status == ReplicaStatus::COMPLETE) {
            replica = replica_list[i];                // ← 第一个 COMPLETE 就用, 线性 scan
            return ErrorCode::OK;
        }
    }
    return ErrorCode::INVALID_REPLICA;
}
```

**关键**: `replica_list` 的顺序来自 master 的 `VisitReplicas` 遍历 (`std::vector<Replica>` 内部存储顺序), 也就是**写入顺序**。Mooncake 的写流程:
1. Put 先写 memory replica (PutStart → PutEnd)
2. 后台 offload 任务 (见 `NotifyOffloadSuccess`) 才添加 disk replica

**所以 memory replica 天然在 list 前面**, `FindFirstCompleteReplica` 自然优先选 memory — **即使它是 remote 的**。这符合 §4.3 的性能数据:

> remote RDMA DRAM (~25 GB/s, ~5μs) > local NVMe SSD (~7 GB/s, ~60μs)

### 4.3.2 可选的 "就近优化": GetPreferredReplica

[client_service.cpp:2937-2966](../../Mooncake/mooncake-store/src/client_service.cpp#L2937) 还有一个**可选**函数(不是默认 Get 路径):

```cpp
tl::expected<Replica::Descriptor, ErrorCode> Client::GetPreferredReplica(...) {
    for (const auto& rep : replica_list) {
        if (rep.is_memory_replica()) {
            if (local_endpoints.count(mem_desc.buffer_descriptor.transport_endpoint_)) {
                return rep;                         // ← 优先本地 memory replica
            }
        }
    }
    return replica_list[0];                         // ← 找不到本地的, 回退第一个
}
```

在多 memory replica 场景下, 这个函数会优先选"本 client 自己挂的 segment"—— 但**不考虑 disk**, 也**不是默认 Get 路径**。

---

## § 5 · 常见问题速答

**Q: standalone.yaml 里 master 挂了会怎样?**
A: 控制面停, 所有 store 的数据暂时不可访问 (但数据还在), master 重启后如果不是持久化的就丢所有元数据 → **所有已存的 KV 也就 "逻辑上丢了"**。所以 standalone 只适合开发。

**Q: HA.yaml 里如果 etcd 也挂了呢?**
A: 3 master 失去选主能力, 新请求失败; 已建立连接的数据路径可能还能工作一会儿但不可靠。所以 etcd 本身也要单独部署成 3/5 节点 HA 集群。

**Q: store 副本数是不是越多越好?**
A: 不是。多 replica 增加 master 的状态管理压力 + 更多 RDMA 连接。按实际需要的容量 / TB 算 replicas, 一般单节点给 1-2 TB 起步。

**Q: 为什么 HA master 的 KV 租约只有 10s, 而 standalone 给了 5 分钟?**
A: 短租约保证 master 内存不被僵尸 key 占据, 适合高周转。但代价是 client 要勤续期 (increases RPC pressure)。HA 场景 client 多, 用短租约更安全。standalone 小规模无所谓。

**Q: "NUMA-segmented 自动切分" 和 "外部 numactl" 哪个更好?**
A: 生产首选外部 `numactl` + 多进程 (HA.yaml 方案), 故障隔离更好。NUMA-segmented 适合"一个 daemon 进程吃满整个节点"的紧凑部署, 但单进程崩 = 全节点挂。

---

## § 6 · 延伸阅读

- [**mooncake_k8s_deployment_advanced_cn.md**](./mooncake_k8s_deployment_advanced_cn.md) — 姊妹文档, 收纳 **HA master 详解 / etcd 集成机制 / 健康探测策略** 三大块 "部署/运维进阶" 内容
- [sglang_mooncake_env_config_deep_dive_cn.md](./sglang_mooncake_env_config_deep_dive_cn.md) — SGLang 侧消费 Mooncake 的 env 变量全景
- [k8s_mooncake_ha_deployment_notes.md](./k8s_mooncake_ha_deployment_notes.md) — HA 部署上层说明
- Mooncake 源码: `vllm/third_partys/Mooncake/`
