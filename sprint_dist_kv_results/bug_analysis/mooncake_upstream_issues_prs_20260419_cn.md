# Mooncake 基础概念 + 上游 Issue / PR 匹配汇总

日期：2026-04-19
仓库：https://github.com/kvcache-ai/Mooncake
本地 Mooncake HEAD：`be75ca0`（2026-04-18）

## 用途

1. **第一部分**先把读懂 bug 家族所需的 Mooncake 基础概念讲清楚（`metadata not found`、两类 key、segment / device / buffer、`segment_name` 格式、local segment 创建流程、size 决定因素）
2. **第二部分**把 `sprint_dist_kv_results/bug_analysis/` 里观察到的四类 bug 家族与 Mooncake 上游已报告 / 已修复 / 在讨论中的工作对齐
3. **第三部分**列未被上游覆盖的项 + 下一步建议

相关材料：
- [metadata_not_found_root_causes_20260419_cn.md](metadata_not_found_root_causes_20260419_cn.md)：`metadata not found` 根因白皮书（9 条触发原因枚举）
- [decode5_mtc16_mooncake_correlation_analysis_20260415.md](decode5_mtc16_mooncake_correlation_analysis_20260415.md)：家族 1 源分析
- [batch_put_failed_global_analysis_20260415.md](batch_put_failed_global_analysis_20260415.md)：家族 4 源分析

---

# 第一部分：Mooncake 基础概念

## 1. 什么是 `metadata not found`

**一句话定义**：HTTP metadata server 收到 `GET /metadata?key=X`，本地 `std::unordered_map<string,string>` 里没有 key X，返回 HTTP 404 + body `metadata not found`。

**两处字面 404 抛出点**（仅此两处）：
- GET 路径：[mooncake-store/src/http_metadata_server.cpp:38-41](../../Mooncake/mooncake-store/src/http_metadata_server.cpp#L38-L41)
- DELETE 路径：[mooncake-store/src/http_metadata_server.cpp:100-104](../../Mooncake/mooncake-store/src/http_metadata_server.cpp#L100-L104)

**重要区分**：
| 错误 | 来源 | 含义 | 所属家族 |
| --- | --- | --- | --- |
| `metadata not found`（HTTP 404） | HTTP metadata server | **segment 元信息** 查不到 | 家族 1 |
| `OBJECT_NOT_FOUND(-704)` | master service | **object 元信息** 查不到 | 家族 3 |

两条链独立，不要混。

---

## 2. HTTP metadata server 里存的两类 key

HTTP metadata server 是**纯内存** `std::unordered_map<string,string>`（[http_metadata_server.cpp:12-18](../../Mooncake/mooncake-store/src/http_metadata_server.cpp#L12-L18)），无持久化、无 backup、无心跳。

每个 Transfer Engine 进程启动时会往里写**两条** key：

### 2.1 `mooncake/<cluster_id>/ram/<segment_name>` —— RDMA 数据面

结构（`encodeSegmentDesc`，[transfer_metadata.cpp:191-226](../../Mooncake/mooncake-transfer-engine/src/transfer_metadata.cpp#L191-L226)）：

```json
{
  "name": "192.168.0.103:12345",
  "protocol": "rdma",
  "tcp_data_port": 0,
  "timestamp": "2026-04-19 ...",
  "devices": [
    {"name": "mlx5_0", "lid": 12, "gid": "fe80::..."},
    {"name": "mlx5_1", "lid": 13, "gid": "fe80::..."}
  ],
  "buffers": [
    {
      "name": "kv_layer_0",
      "addr": 0x7f1234560000,
      "length": 2147483648,
      "rkey": [0xABCD0001, 0xABCD0002],
      "lkey": [0x12340001, 0x12340002]
    }
  ],
  "priority_matrix": { ... }
}
```

**作用**：给 peer 做 **RDMA 数据读写** 用。
- `devices` + `lid`/`gid` → 建 QP 握手需要
- `buffers[i].addr + length + rkey[]` → RDMA_WRITE / READ 的目标地址 + 权限
- `priority_matrix` → peer 选用哪块 NIC 走这个 buffer

### 2.2 `mooncake/<cluster_id>/rpc_meta/<segment_name>` —— 控制面

结构（`addRpcMetaEntry`，[transfer_metadata.cpp:865-867](../../Mooncake/mooncake-transfer-engine/src/transfer_metadata.cpp#L865-L867)）：

```json
{
  "ip_or_host_name": "192.168.0.103",
  "rpc_port": 12345
}
```

**作用**：给 peer 做**握手 / 通知 / 探活的 RPC 定位**用——"这个 `segment_name` 身份对应的 RPC 端点在哪"。

### 2.3 key 前缀

- `mooncake/` 固定前缀
- `<cluster_id>` 由 `MC_METADATA_CLUSTER_ID` 环境变量决定（为空则省略）—— 不同 cluster_id 的 client 互相查不到对方
- `<segment_name>` 一般是 `hostname:rpc_port`，详见 §4

---

## 3. segment / device / buffer 的关系是 1 : N : M

**一个 Transfer Engine 进程 = 一个 local segment**，不管注册多少块内存、有多少张网卡。

| 概念 | 个数 | 从哪来 |
| --- | --- | --- |
| **segment** | 进程级唯一（= 1） | `allocateLocalSegmentID()` 启动时创建 |
| **device** | N 张 | `ibv_get_device_list()` 枚举 + 用户 filter（[topology.cpp:150-197](../../Mooncake/mooncake-transfer-engine/src/topology.cpp#L150-L197)）|
| **buffer** | M 条 | 每次 `registerLocalMemory()` 追加一条 |

**每条 buffer 对每张 device 都有一对 `lkey/rkey`**：
```cpp
// rdma_transport.cpp:278-281
for (auto &context : context_list_) {
    buffer_desc.lkey.push_back(context->lkey(addr));
    buffer_desc.rkey.push_back(context->rkey(addr));
}
```
所以 `buffer_desc.lkey[]` 数组的**长度 = device 数**。

**注册 memory 不创建新 segment**——只是往 segment 的 `buffers` 数组追加一条。vLLM prefill worker 注册 N 层 KV cache ⇒ **1 个 segment + N 条 buffer**。

### 举个例子

`gb200-rack1-09` 上 1 个 vLLM prefill worker，挂 2 张 mlx5 网卡，注册 3 层 KV cache：

```
segment "10.0.0.9:12345"
├─ devices = [mlx5_0, mlx5_1]        ← 2 张网卡
└─ buffers = [
     {name: "layer_0", addr: A, length: L, lkey: [L_a0, L_a1], rkey: [R_a0, R_a1]},
     {name: "layer_1", addr: B, length: L, lkey: [L_b0, L_b1], rkey: [R_b0, R_b1]},
     {name: "layer_2", addr: C, length: L, lkey: [L_c0, L_c1], rkey: [R_c0, R_c1]}
   ]
```

---

## 4. `segment_name` 的格式：不一定是 `host:port`

`segment_name` 在**不同模式下由不同主体**决定。

| 模式 | `segment_name` 来自 | TE 是否强制格式 |
| --- | --- | --- |
| 一般 RDMA | 调用方原样传入（[transfer_engine_impl.cpp:107](../../Mooncake/mooncake-transfer-engine/src/transfer_engine_impl.cpp#L107) `local_server_name_ = local_server_name`）| **不强制**，理论上可以任意字符串 |
| Ascend | 从 `ip:port:npu_x` 解出再拼（[:102](../../Mooncake/mooncake-transfer-engine/src/transfer_engine_impl.cpp#L102)）| 格式固定 `host:port` |
| p2p handshake | TE 自己 `findAvailableTcpPort()` 拼（[:139-143](../../Mooncake/mooncake-transfer-engine/src/transfer_engine_impl.cpp#L139-L143)）| **强制** `host:port` |

**为什么一般模式下大家都写 `host:port`**：vLLM / Mooncake Store / SGLang 等上层调用 `TransferEngine::init` 时**按约定传 `host:port`**。TE 本身不关心——只是把它当成一个不透明的 key。

**为什么 p2p 模式必须是 `host:port`**：看 [transfer_metadata.cpp:896-899](../../Mooncake/mooncake-transfer-engine/src/transfer_metadata.cpp#L896-L899) 的 `getRpcMetaEntry` p2p 分支：
```cpp
if (p2p_handshake_mode_) {
    auto [ip, port] = parseHostNameWithPort(server_name);
    desc.ip_or_host_name = ip;
    desc.rpc_port = port;
}
```
p2p 模式**没有 HTTP metadata server，也就没有 `rpc_meta/` key**；ip 和 port 必须**编码在 name 里**，直接 parse 出来。

**对比总结**：
- 一般模式：`segment_name` 是**不透明键**，`rpc_meta/<name>` 存 `{ip, port}` 补信息
- p2p 模式：`segment_name` **自带** ip 和 port，无需 `rpc_meta/` 补充 —— 这就是 p2p 模式能豁免 rpc_meta 的原因

---

## 5. local segment 的完整创建流水

入口是 `TransferEngineImpl::init(metadata_conn_string, local_server_name, ip_or_host_name, rpc_port)`（[transfer_engine_impl.cpp:67](../../Mooncake/mooncake-transfer-engine/src/transfer_engine_impl.cpp#L67)）。按 RDMA 路径展开 8 步：

| 步骤 | 动作 | 源码 | 副作用 |
| --- | --- | --- | --- |
| ① | 决定 `local_server_name_` | [transfer_engine_impl.cpp:102, :107, :139-143](../../Mooncake/mooncake-transfer-engine/src/transfer_engine_impl.cpp#L102) | — |
| ② | 构造 `TransferMetadata` | [:182](../../Mooncake/mooncake-transfer-engine/src/transfer_engine_impl.cpp#L182) | 选 handshake plugin 或 storage plugin |
| ③ | `addRpcMetaEntry(local_server_name_, desc)` | [:192](../../Mooncake/mooncake-transfer-engine/src/transfer_engine_impl.cpp#L192) | **HTTP PUT `rpc_meta/<name>`** |
| ④ | `installTransport` + topology discover | [:186-243](../../Mooncake/mooncake-transfer-engine/src/transfer_engine_impl.cpp#L186-L243) | 枚举 HCA，填 `local_topology_` |
| ⑤ | `initializeRdmaResources()` | [rdma_transport.cpp:104](../../Mooncake/mooncake-transfer-engine/src/transport/rdma_transport/rdma_transport.cpp#L104) | 为每张 HCA open ibv context |
| ⑥ | `allocateLocalSegmentID()` | [rdma_transport.cpp:357-373](../../Mooncake/mooncake-transfer-engine/src/transport/rdma_transport/rdma_transport.cpp#L357-L373) | new `SegmentDesc`，填 `devices[]`，写进**进程内存** |
| ⑦ | `updateLocalSegmentDesc()` | [transfer_metadata.cpp:762-773](../../Mooncake/mooncake-transfer-engine/src/transfer_metadata.cpp#L762-L773) | **HTTP PUT `ram/<name>`**（此时 `buffers=[]`） |
| ⑧ | 每次 `registerLocalMemory()` | [rdma_transport.cpp:174-299](../../Mooncake/mooncake-transfer-engine/src/transport/rdma_transport/rdma_transport.cpp#L174-L299) | `ibv_reg_mr` + 追加 buffer + 重新 PUT `ram/<name>` |

**HTTP server 上的时间线**：
```
t0  init 开始
t1  第 ③ 步：HTTP server 上出现 rpc_meta/<name>      ← 第一次触达
t2  第 ⑦ 步：HTTP server 上出现 ram/<name>，buffers=[]
t3  第 ⑧ 步首次 register_buffer：ram/<name> 里 buffers=[layer_0]
t4+ 每次 register_buffer 都重写 ram/<name>
```

**关键观察**：
- "segment 被创建" 可以指**三个不同时刻**，依赖你关注哪一层——进程内存对象（⑥）、HTTP server 上可见（⑦）、可用数据面（⑧a）
- `rpc_meta/` **在第 ③ 步就写了**，远早于 `ram/`
- `ram/` 从 `buffers=[]` 到"有可用数据"之间存在短暂窗口，启动 race 可能命中这个窗口

---

## 6. local segment 的大小谁决定

**SegmentDesc 没有"size"字段**。所谓"大小"是 `sum(buffers[i].length)`，每条 buffer 的 `length` 来自 `registerLocalMemory(addr, length, ...)` 的入参。

源码路径：
```cpp
// rdma_transport.cpp:295-298
buffer_desc.addr   = (uint64_t)addr;
buffer_desc.length = length;               // 入参直接塞进去
metadata_->addLocalMemoryBuffer(buffer_desc, update_metadata);

// transfer_metadata.cpp:803
segment_desc->buffers.push_back(buffer_desc);   // 追加，永远不重算 size
```

**TE 不校验、不限制 `length`**（只检查 `length > 0` 和地址不重叠）。多大都收。

### 谁调 `registerLocalMemory` —— vLLM 这条栈上

[mooncake_store_worker.py:857-864](../../vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_worker.py#L857-L864)：
```python
for cache in kv_caches.values():
    cache_storage = cache.untyped_storage()
    base_addr  = cache_storage.data_ptr()
    region_len = cache_storage.nbytes()     # ← 每次 register 多少
    if base_addr not in seen_ptrs:
        seen_ptrs.add(base_addr)
        ret = self.store.register_buffer(base_addr, region_len)
```

`region_len = cache_storage.nbytes()` 来自 PyTorch tensor 的底层 storage，对 vLLM 来说就是**一层 KV cache 的字节数**。

所以 vLLM prefill worker 的 local segment 大小 =

```
Σ over KV layers (按 base_addr 去重后)  cache_storage.nbytes()
```

这个值由 vLLM engine 的配置决定：
- `num_kv_heads / head_dim / num_hidden_layers`（模型）
- `block_size`（`--block-size`）
- `gpu_memory_utilization` → `num_blocks`（`--gpu-memory-utilization`）
- `dtype`（`--kv-cache-dtype`）

### 别和另一个"segment"混

Mooncake 里还有**另一个同名概念** —— **Mooncake Store 的 mount segment**（[real_client.cpp:512-536](../../Mooncake/mooncake-store/src/real_client.cpp#L512-L536)）：

```cpp
while (global_segment_size > 0) {
    size_t segment_size = std::min(global_segment_size, max_mr_size);
    client_->MountSegment(ptr, segment_size, protocol);
}
```

那个 segment 有明确 size（`global_segment_size`，由 `MC_GLOBAL_SEGMENT_SIZE` / `MC_MAX_MR_SIZE` 两个 env 决定），是 **Mooncake Store 作为存储服务端**给 object 分配空间用的。

| "segment" 所指 | size 来源 | 配置位置 |
| --- | --- | --- |
| **TE local segment**（本文讨论的）| `sum(buffers[i].length)` | vLLM 侧：`cache_storage.nbytes()` |
| **Mooncake Store mount segment**（服务端存储单元）| `global_segment_size` | `MC_GLOBAL_SEGMENT_SIZE` / `MC_MAX_MR_SIZE` env |

vLLM 这侧关心的只是第一行。

---

## 7. 为什么 `ram/` 和 `rpc_meta/` 必须是两个 key

五个维度全都不同：

| 维度 | `rpc_meta/<name>` | `ram/<name>` |
| --- | --- | --- |
| **写入时机** | 第 ③ 步，RPC 端口 bind 后第一时间 | 第 ⑦ 步后 PUT 空 desc，⑧ 步起随 buffers 变化刷新 |
| **更新频率** | 进程存活期间不变 | 每次 register/unregister buffer 都重写 |
| **唯一性约束** | **拒绝重复 PUT**（[http_metadata_server.cpp:64-73](../../Mooncake/mooncake-store/src/http_metadata_server.cpp#L64-L73) 防身份冲突） | PUT 允许覆盖 |
| **访问路径** | 握手前查一次拿 `ip:port` | 通常握手消息直接带，本地 cache 失效才回源 |
| **p2p 模式** | 豁免，不写 storage（[transfer_metadata.cpp:843-863](../../Mooncake/mooncake-transfer-engine/src/transfer_metadata.cpp#L843-L863)）| 依然需要 |

### 唯一性约束是最致命的差异

看 [http_metadata_server.cpp:64-73](../../Mooncake/mooncake-store/src/http_metadata_server.cpp#L64-L73)：
```cpp
if (key.find("rpc_meta") != std::string::npos &&
    store_.find(std::string(key)) != store_.end()) {
    resp.set_status_and_content(status_type::bad_request,
                                "Duplicate rpc_meta key not allowed");
    return;
}
```

客户端异常退出没删 `rpc_meta/<name>`，下次同身份的新进程想 PUT，会被**拒绝**——新进程起不来。这就是为什么 PR #1363 那条心跳超时清理必须**同时**清 `ram/*` 和 `rpc_meta/*`：只清一条就会形成"握手能通但数据面 key 丢"或反过来的半状态。

### 两条 key 残留 / 缺失时的典型失败

| 缺失情况 | 后果 |
| --- | --- |
| `rpc_meta` 缺 | 连握手都发不出去（不知道 peer 的 RPC 端点） |
| `rpc_meta` 残留 | 握手打到**已死进程**，ETIMEDOUT ← **家族 2 症状** |
| `ram` 缺 | 握手成功但拿不到 `lkey/rkey/addr`，RDMA_WRITE 无法提交 ← **家族 1 主症状** |
| `ram` 残留（旧 `rkey/addr`） | RDMA_WRITE 写到**已释放的内存**，对端 NIC 回 remote access error |

---

# 第二部分：四类 bug 家族的上游匹配

## 家族 1 —— 元数据 / 过期 descriptor 可见性失败

**症状链**：`metadata not found` → `Failed to open segment` → `Transfer submission failed for key` → `TRANSFER_FAIL(-800)` → `batch_put failed`

**本地假设**：共享 HTTP metadata server 里残留了上一个 job 的 `transport_endpoint_` 记录。

| 编号 | 类型 | 状态 | 日期 | 内容 |
| --- | --- | --- | --- | --- |
| [#1115](https://github.com/kvcache-ai/Mooncake/issues/1115) | Issue | open | 2025-11-26 | 同一 Python 进程里多个 `MooncakeDistributedStore` 客户端会出现 `metadata not found`。症状一致，根因不同。 |
| [#1260](https://github.com/kvcache-ai/Mooncake/issues/1260) | Issue | open | 2025-12-23 | ADXL / Ascend transport 单机环境下的 `metadata not found`。同样错误串，不是同一 bug。 |
| [#1549](https://github.com/kvcache-ai/Mooncake/pull/1549) | PR | 已合入 2026-03-04 | 修磁盘淘汰后 master 未被通知、导致 `DiskDescriptor` 过期的路径。不覆盖跨 job endpoint 过期。 |
| [#1826](https://github.com/kvcache-ai/Mooncake/pull/1826) | PR | 已合入 2026-04-09 | **最接近的客户端侧修复**。详见下方 "PR #1826" 小节。 |
| [#1624](https://github.com/kvcache-ai/Mooncake/pull/1624) | PR | 已合入 2026-03-09 | 断连时移除缓存的 target segment descriptor。详见下方 "PR #1624" 小节。 |
| [#1485](https://github.com/kvcache-ai/Mooncake/pull/1485) | PR | 已合入 2026-02-04 | `unregisterLocalMemory` 后同步 metadata，避免 peer 看到过期 buffer descriptor。 |
| [#1219](https://github.com/kvcache-ai/Mooncake/pull/1219) | PR | open | 2025-12-16 | "Delete useless meta from http metadata server"（拉模型）。详见下方 "#1219 / #1363 详细对比"。 |
| [#1363](https://github.com/kvcache-ai/Mooncake/pull/1363) | PR | open (WIP) | 2026-01-12 | "HTTP metadata cleanup on client timeout"（推模型），**精确命中我们的根因假设**。详见下方 "#1219 / #1363 详细对比"。 |

### PR #1826 —— TENT 客户端缓存一致性重构（已合入）

- **作者**：caozhanhao　**提交**：2026-04-06　**合入**：2026-04-09
- **分支**：`pr/refetch_segment`，改动 20 个文件，+550 / −180
- **模块**：Transfer Engine 的 **TENT 子模块**（`mooncake-transfer-engine/tent/`）

**作者报告的场景**（与我们假设很像）：
- SGLang + Mooncake-PG + TENT，多个 ProcessGroup **顺序初始化**
- 每个 PG 注册内存后做 warm-up transfer，让 peer 的 segment 信息进本地缓存
- 下一个 PG 注册新内存时，之前缓存里的 segment 条目就**过期**了
- 新 PG warm-up 反复失败，`findBuffer()` 在过期缓存里查不到 peer 刚注册的 buffer

**修复机制**：新增 `SegmentManager::withCachedSegment()` 模板 helper，把"拿缓存 → 操作 → 失败时 invalidate 重试一次"封装成一个模式。
1. 从缓存拿 segment
2. 执行调用方传入的 lambda
3. 若 lambda 返回 `Status::NeedsRefreshCache`，就 invalidate 缓存、重新拉、再执行一次
4. 仍失败就转成 `InvalidEntry`

**关键附带收益**：segment 缓存 TTL 不再需要为了一致性而设得很短，可以放大甚至无限大——正确性不再依赖 TTL 过期。

**对我们的意义**：**限于 TENT 子模块**。`MooncakeStoreConnector` 背后走的是 TE（不是 TENT），所以此修复**不会自动生效**。只是证明"客户端缓存一致性"这一层上游已经在修。TTL 调参思路可借鉴。

### PR #1624 —— Ascend 断连清缓存（已合入）

- **作者**：ascend-direct-dev　**提交**：2026-03-06　**合入**：2026-03-09
- **分支**：`new`，改动 2 个文件，+27 / −22
- **模块**：**Ascend Direct Transport only**

**机制**：Ascend Direct Transport 在断开与 peer 的连接时，顺带把本地缓存里那个 peer 的 target segment descriptor 一并移除。peer 用不同参数重连时不会命中旧缓存。

**对我们的意义**：**和我们 bench 场景无直接关系**（我们走 GB200 + RDMA）。列在这里主要是因为它**证明了"断连清缓存"这一动作在上游被普遍认可**——先在 Ascend 上做，其他 transport 后来被 #1826 用更通用的 `withCachedSegment` 方式覆盖。

### #1219 / #1363 详细对比 —— 最重要的一对

两个 PR 针对的是**同一个问题**：HTTP metadata server 没有心跳机制，客户端异常退出后 `mooncake/ram/*` 和 `mooncake/rpc_meta/*` 会残留过期信息。新节点用这些信息去连，或同一节点用不同 RDMA 参数重启，就会失败——**这正是我们家族 1 的根因假设**。

`ykwd` 在 #1219 的 review 里已经点明两者的分歧：**#1219 走拉模型（periodic poll），#1363 走推模型（event-driven cleanup on timeout）**。

#### PR #1219 —— 拉模型：HTTP server 主动轮询 master 对账

- **作者**：stmatengss　**提交**：2025-12-16　**状态**：open
- **改动 8 个文件，+202 / −16**
- **机制**：HTTP metadata server 后台周期性调用 master service 的 RPC 拿权威 segment / client 列表，和本地缓存对账，删不一致的条目
- **改动文件**：
  - `mooncake-store/src/http_metadata_server.cpp`（+107）：周期性健康检查 worker
  - `mooncake-store/{include,src}/ha_helper.{h,cpp}`（+26）：HA 辅助逻辑
  - `mooncake-store/include/master_config.h`（+9）、`master.cpp`（+26/−16）
  - `mooncake-store/{include,src}/rpc_service.{h,cpp}`（+13）
- **已知 review 反馈**（作者自己引述）：
  - `is_segment_healthy` 对每个 segment 都调一次 `GetAllSegments`——O(N²) 嫌疑
  - 用 master 的 ping 路径做 client 健康检查太 hacky
- **派生小 PR**（已合入）：#1484（优化健康检查）、#1489（clang-format）、#1490（trailing whitespace）
- **当前阻塞**：路线之争，作者最后一条留言是向维护者询问"Which PR is better?"

#### PR #1363 —— 推模型：Master heartbeat 超时时通知 HTTP server 清理

- **作者**：chenkaiyue　**提交**：2026-01-12　**状态**：open (WIP)
- **改动 8 个文件，+144 / −3**
- **问题陈述**（直接引 PR body，与我们家族 1 症状一字对应）：
  > When a client **crashes or is forcefully terminated** (e.g., `kill -9`, OOM killed, node failure), the cleanup operations may fail or never execute. The Master Service can detect this through heartbeat timeout (`client_ttl`) and clean up its internal segment resources, but the **HTTP Metadata Server has no heartbeat mechanism** and cannot detect client failures.
  >
  > This leads to **stale metadata residue** on the HTTP Metadata Server:
  > - `mooncake/ram/{segment_name}` — Contains outdated RDMA buffer information
  > - `mooncake/rpc_meta/{segment_name}` — Contains stale RPC connection details
  >
  > These residual entries can cause issues when:
  > - New nodes attempt to connect using stale metadata
  > - The same node restarts and registers with different RDMA parameters
  > - Other nodes query and cache outdated peer information
- **机制**：复用 Master Service 已有的 client heartbeat + `client_ttl`；`ClientMonitorThreadMain()` 检测到客户端超时，进入原本的 `UnmountSegment` 流程时**额外调用** `HttpMetadataServer::removeKey()` / `removeKeys()` 把 `mooncake/ram/*` 和 `mooncake/rpc_meta/*` 删掉
- **部署约束**：HTTP Metadata Server 和 Master Service **同进程共部署**（`--enable_http_metadata_server=true`）。Master 直接调 C++ 方法，不走网络
- **开关**：新增 `--enable_metadata_cleanup_on_timeout`，默认关
- **改动文件**：
  - `mooncake-store/include/master_config.h`：新增 bool 配置字段
  - `mooncake-store/src/master.cpp`：命令行 flag 解析 + 校验 + 把 `HttpMetadataServer*` 传进 `WrappedMasterService`
  - `mooncake-store/{include,src}/http_metadata_server.{h,cpp}`：新增 `removeKey` / `removeKeys`
  - `mooncake-store/{include,src}/master_service.{h,cpp}`：持有 `HttpMetadataServer*` 成员，在 `ClientMonitorThreadMain()` 里调 `cleanupHttpMetadata(segment_name)`
  - `mooncake-store/{include,src}/rpc_service.{h,cpp}`：`WrappedMasterService` 构造函数加可选 `HttpMetadataServer*` 参数
- **典型用法**：
  ```bash
  mooncake_master \
      --enable_http_metadata_server=true \
      --enable_metadata_cleanup_on_timeout=true \
      --client_ttl=20 \
      --v=1
  ```
- **当前阻塞**：WIP + CI + 路线之争

#### 两者对比

| 维度 | #1219（poll-based）| #1363（event-driven）|
| --- | --- | --- |
| 触发时机 | 周期性，秒~分钟级延迟 | 客户端超时立即清理 |
| master 心跳依赖 | 需要，但通过 ping 绕一层 | 直接复用已有机制，路径最短 |
| 部署约束 | HTTP server 可独立部署 | HTTP server 必须和 master 同进程 |
| 实现复杂度 | 需要对账逻辑 + O(N²) 嫌疑 | 在现有 `ClientMonitorThreadMain` 里加一个调用 |
| 适用场景 | HA 场景下 HTTP server 与 master 分离部署 | 常见共部署场景 |
| **对我们 bench 的契合度** | 通用，但慢 | **我们就是共部署**，直接对症 |

**对我们的意义**：#1363 的 problem statement 和我们在 [decode5_mtc16_mooncake_correlation_analysis_20260415.md](decode5_mtc16_mooncake_correlation_analysis_20260415.md) 里写的"共享 metadata server 里残留了上一个 job 的 `transport_endpoint_`"完全对应；#1363 如果合入，家族 1 会从"强假设"升级为"已有上游修复"。**短期内没有现成补丁可以直接 cherry-pick**——需要自己把 #1363 打到本地 Mooncake，或者等上游路线定下来。

### 家族 1 小结

- 症状在上游有多处报告，但"跨 job 共享 metadata server 残留 `transport_endpoint_`"这个**根因**目前**无人明确命名**
- 最接近的两个 PR（#1219 / #1363）都没合入
- 客户端缓存侧已有部分修复（#1826 TENT、#1624 Ascend、#1485 unregister），但都不覆盖我们走的主 TE + RDMA 路径

---

## 家族 2 —— RDMA 握手 / 端点状态失败

**症状链**：`Failed to modify QP to RTR, Connection timed out [110]` (rdma_endpoint.cpp:646) → `packet mismatch` (rdma_endpoint.cpp:283) → `mark it inactive` → 60s `transfer timeout` (transfer_task.cpp:341) → `TRANSFER_FAIL(-800)` → `batch_put failed`

**本地假设**：启动窗口大量并发建 RDMA 连接，QP 状态机在 `INIT → RTR` 拿不到 peer 的 `mtu/gid/peer lid/peer qp num`，超时后端点被标 inactive；后续业务命中 inactive 端点时卡 60s 硬等待。

| 编号 | 类型 | 状态 | 日期 | 内容 |
| --- | --- | --- | --- | --- |
| [#204](https://github.com/kvcache-ai/Mooncake/issues/204) | Issue | open | 2025-04-03 | 最早的 `Failed to modify QP to RTR` ETIMEDOUT 报告。只是症状，未闭环。 |
| [#1066](https://github.com/kvcache-ai/Mooncake/issues/1066) | Issue | closed | 2025-11-17 | 推理时同样的 `QP to RTR ... Connection timed out`。关闭但没 merged 修复。 |
| [#1151](https://github.com/kvcache-ai/Mooncake/issues/1151) | Issue | open | 2025-12-02 | 多网卡 k8s 环境下 `packet mismatch` + `mark it inactive` 组合。 |
| [#1088](https://github.com/kvcache-ai/Mooncake/issues/1088) | Issue | open | 2025-11-20 | `Failed to complete transfers after 60 seconds`，60s 超时。 |
| [#919](https://github.com/kvcache-ai/Mooncake/issues/919) | Issue | open | 2025-10-13 | 60s timeout → `TRANSFER_FAIL` → `batch_put failed err -800`，**与我们日志一字不差**。未闭环。 |
| [#1766](https://github.com/kvcache-ai/Mooncake/issues/1766) | Issue | closed | 2026-04-03 | 云燧 (Yunsilicon) 网卡上的 `QP to RTR`，硬件特化问题，不是通用 bug。 |
| [#1560](https://github.com/kvcache-ai/Mooncake/pull/1560) | PR | 已合入 2026-02-27 | 修 bootstrap RPC 在握手阶段的 re-entrancy 死锁。 |
| [#1705](https://github.com/kvcache-ai/Mooncake/pull/1705) | PR | 已合入 2026-03-23 | **修"reusing connection"方向的根因**：并发 `connect()` 把已建立的 endpoint reset 掉。 |
| [#1733](https://github.com/kvcache-ai/Mooncake/pull/1733) | PR | 已合入 2026-03-30 | 修 RDMA simultaneous-open 竞态导致的 endpoint use-after-delete。与 #1705 同家族。 |
| [#1803](https://github.com/kvcache-ai/Mooncake/pull/1803) | PR | 已合入 2026-04-02 | 修握手路径 duplicate notify recv WR + 误导性 `Success [0]` 日志。家族 2 + 家族 4 都受益。 |
| [#1762](https://github.com/kvcache-ai/Mooncake/pull/1762) | PR | 已合入 ~2026-03-29 | 修握手连接建立时的死锁。相邻修复。 |
| [#398](https://github.com/kvcache-ai/Mooncake/pull/398) | PR | 已合入（较早）| 引入软件层 60s transfer 超时机制的原始 PR。60s 硬等待的出处。 |

### 家族 2 小结

- 症状报告密集，2026-Q1 连续合入多条根因补丁（#1560 / #1705 / #1733 / #1803 / #1762）
- 通用的"ETIMEDOUT on stale peer QP reused from previous job"场景在 #1088 / #919 / #1151 仍然 open
- **优先核对本地 Mooncake pin 是否已包含上述 2026-Q1 的合入**（本地 HEAD `be75ca0` 经 `git log` 核对：#1803 / #1733 / #1705 均已包含）

---

## 家族 3 —— processing 对象再 put / `OBJECT_NOT_FOUND(-704)` 放大

**症状链**：先 `TRANSFER_FAIL(-800)`，延迟到达的 `BatchPutEnd` 返回 `OBJECT_NOT_FOUND(-704)`；涉及 `put_start_discard_timeout_sec=30` 丢弃旧 processing metadata、`ExistKey` 语义歧义（absent vs processing 未完成）、`WaitForTransfers()` 在 `prefer_alloc_in_same_node` 合并写路径上的串行等待。

| 编号 | 类型 | 状态 | 日期 | 内容 |
| --- | --- | --- | --- | --- |
| [#974](https://github.com/kvcache-ai/Mooncake/issues/974) | Issue | closed | 2025-10 | **根因精确匹配**："Keys stuck in PROCESSING if PUT initiator crashes"。 |
| [#975](https://github.com/kvcache-ai/Mooncake/issues/975) | Issue (RFC) | closed | — | 针对 #974 的设计讨论：卡在 processing 的副本如何自动清理。 |
| [#849](https://github.com/kvcache-ai/Mooncake/issues/849) | Issue | open | 2025-09-17 | 同一假设：后续对同 key 的 `PutStart` 永远失败。 |
| [#727](https://github.com/kvcache-ai/Mooncake/issues/727) | Issue | open | 2025-08-08 | `object_already_exists` 然后 `CHECK_EQ(status_, PROCESSING)`。对应 `ExistKey` 语义歧义。 |
| [#571](https://github.com/kvcache-ai/Mooncake/issues/571) | Issue | closed | — | `BatchIsExist` 不一致地返回 -704。同样指向 `ExistKey` 语义。 |
| [#993](https://github.com/kvcache-ai/Mooncake/pull/993) | PR | 已合入 2025-11-06 | **实现 `put_start_discard_timeout_sec=30` + client_id 跟踪机制**。本家族主力修复。 |

### 家族 3 小结

- 根因和核心机制在上游**已明确并已合入主干**
- 但以下两条子假设在上游**找不到专门的 issue / PR**：
  - `ExistKey` 三态语义（absent / processing / complete）
  - `prefer_alloc_in_same_node` 合并写 + `WaitForTransfers()` 串行等待的放大效应
- 这两项值得我们向上游提新 issue

---

## 家族 4 —— Python 侧错误账本 / 可观测性缺口

**本地问题**：`-1` 被当作 `TRANSFER_FAIL` 计入统计（实际 -1 是 `INTERNAL_ERROR`，-800 才是 `TRANSFER_FAIL`）；`batch_put failed` 警告缺少 `req_id / tp_rank / elapsed / first_failed_key / readable code name`；decode 侧只看到 `External prefix cache hit rate: 100%`，无法知道外部 load 是否真的成功。

| 编号 | 类型 | 状态 | 日期 | 内容 |
| --- | --- | --- | --- | --- |
| [#1475](https://github.com/kvcache-ai/Mooncake/pull/1475) | PR | open | 2026-02-02 | "[TENT] Add logs for troubleshooting"：统一日志工具 + 限流。相邻但不是错误码可读化。 |
| [#1850](https://github.com/kvcache-ai/Mooncake/issues/1850) | Issue (RFC) | open | 2026-04-09 | "End-to-End Transfer Tracing for Mooncake"：上游最接近的可观测性总体方案。 |
| [#1851](https://github.com/kvcache-ai/Mooncake/pull/1851) | PR | open | 2026-04-09 | 实现 #1850 的 tracing pipeline。 |
| [#1529](https://github.com/kvcache-ai/Mooncake/pull/1529) | PR | 已合入 2026-02-10 | slog 日志等级通过 env var 控制。小改进。 |
| [#1803](https://github.com/kvcache-ai/Mooncake/pull/1803) | PR | 已合入 2026-04-02 | （家族 2 已列）修握手路径误导性的 `Success [0]` 日志。**唯一已合入的"错误可读化"类修复**。 |

### 家族 4 小结

可读错误码映射、结构化握手诊断字段（endpoint / NIC / MTU / GID / LID / QP）、metadata server 的 PUT / DELETE 审计日志都**在上游没有专门的工作**。#1850 / #1851 做的是整体 tracing，不是错误码可读化本身。

---

# 第三部分：总结与下一步

## 上游未报告的项（建议我们向上游提 issue）

1. **家族 1**：跨 job 的 `transport_endpoint_` 在共享 HTTP metadata server 中残留——未作为独立根因被命名。
2. **家族 3**：`ExistKey` 三态语义（absent / processing / complete）、`prefer_alloc_in_same_node` 合并写与 `WaitForTransfers()` 串行等待的相互作用——未报告。
3. **家族 4**：错误码可读名映射、握手结构化诊断字段、metadata server 的 PUT / DELETE 审计日志——没有专门的 issue / PR。

其余所有症状在上游都至少有症状级或相邻修复级的覆盖。

## 下一步建议

### 短期（我们自己能做）

1. **cherry-pick #1363 到本地 Mooncake**：最直接堵住家族 1 根因。需要 `mooncake_master --enable_metadata_cleanup_on_timeout=true --enable_http_metadata_server=true`。
2. **vLLM 侧在 `MooncakeStoreWorker` 补 `__del__` / 显式 `unregister_buffer`**：当前 [mooncake_store_worker.py:864](../../vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_worker.py#L864) 只调 `register_buffer`，进程正常退出时不会发对偶的 `removeSegmentDesc` —— 等于放大家族 1 的 tail。
3. **核对本地 Mooncake 是否已含 2026-Q1 合入**：#1705 / #1733 / #1803 / #993 本地 HEAD `be75ca0` 已包含；**#1826 未包含**（`withCachedSegment` 全树 0 匹配，但我们走 TE 不受影响）。
4. **bench 脚本加 metadata server uptime 校验**：堵 "HTTP server 重启导致整轮 404" 的诊断盲区。

### 中期（等上游）

1. **持续盯 #1219 / #1363 路线决策** —— 维护者定稿前不深投入。
2. **补 #1485 的 "DELETE 失败重试"** —— 向上游提 issue。

### 长期

1. **HTTP metadata server 持久化 / 重启恢复** —— 无上游 PR，值得新 issue。
2. **主 TE 客户端 cache 失效策略迁移到 `withCachedSegment` 模式** —— 当 #1826 合入稳定后向 TE 侧推广。

---

## 附录：本文论点的源码验证命令

所有源码行号基于本地 Mooncake HEAD `be75ca0`：

```bash
# 验证本地 HEAD
git -C /home/aoshen/setup_new_cluster/vllm/Mooncake log --oneline -1
# 预期：be75ca0 ...

# metadata not found 仅两处
grep -n "metadata not found" Mooncake/mooncake-store/src/http_metadata_server.cpp
# 预期：41, 104

# master 不通知 HTTP server（证明 #1363 未合入）
grep -rn "removeKey\|removeKeys" Mooncake/mooncake-store/
# 预期：仅 http_metadata_server.h 的实现，master_service.cpp 无匹配

grep -rn "enable_metadata_cleanup_on_timeout" Mooncake/mooncake-store/
# 预期：0 匹配

# #1826 未合入
grep -rn "withCachedSegment" Mooncake/
# 预期：0 匹配

# 本地已含的 PR（应全部命中）
git -C /home/aoshen/setup_new_cluster/vllm/Mooncake log --oneline --all | grep -E "1549|1624|1485|1705|1733|1803|993"

# Failed to open segment 触发点
grep -n "Failed to open segment" Mooncake/mooncake-store/src/transfer_task.cpp
# 预期：493, 535, 657

# vLLM 侧未调 unregister_buffer
grep -n "register_buffer\|__del__" vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_worker.py
# 预期：仅 :864 的 register_buffer，无 __del__ 或 unregister_buffer
```
