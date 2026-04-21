# `metadata not found` 根因白皮书

日期：2026-04-19
本地 Mooncake HEAD：`be75ca0`（2026-04-18，"client_service: quorum write on WaitForTransfers + richer failure logging"）
相关分析：[mooncake_upstream_issues_prs_20260419_cn.md](mooncake_upstream_issues_prs_20260419_cn.md)、[decode5_mtc16_mooncake_correlation_analysis_20260415.md](decode5_mtc16_mooncake_correlation_analysis_20260415.md)、[batch_put_failed_global_analysis_20260415.md](batch_put_failed_global_analysis_20260415.md)

## 目的

我们的 benchmark 报告已经把 `metadata not found → Failed to open segment → TRANSFER_FAIL(-800) → batch_put failed` 这条链定位为家族 1 根因，但只给了**症状级**描述。上游 PR #1219 / #1363 / #1549 / #1826 各自解决一类 stale metadata，没有一份文档把**所有可能的触发原因**列全。

本文做的事：**枚举本地 Mooncake 代码树里所有会让 HTTP metadata server 对 `GET /metadata?key=...` 返回 404 的独立触发路径**，每条带源码定位 + 机制 + 是否已有上游 PR + 是否影响我们 bench。

---

## 第 0 节：本地代码库版本

### 本地 HEAD 包含的相关 PR（已验证）

| PR | 标题 | commit |
| --- | --- | --- |
| [#1803](https://github.com/kvcache-ai/Mooncake/pull/1803) | [TENT] Fix duplicate notify recv WR posting and PLOG misuse | `a85800b` |
| [#1733](https://github.com/kvcache-ai/Mooncake/pull/1733) | [TE] Fix simultaneous open handshake in RdmaEndpoint | `4b3d44f` |
| [#1705](https://github.com/kvcache-ai/Mooncake/pull/1705) | [TENT] avoid resetting RDMA endpoint on duplicate concurrent bootstrap | `692ccda` |
| [#1624](https://github.com/kvcache-ai/Mooncake/pull/1624) | remove target segment desc cache when disconnect | `834c416` |
| [#1549](https://github.com/kvcache-ai/Mooncake/pull/1549) | [Store] Notify master on disk eviction to fix stale metadata | `a402dc7` |
| [#1485](https://github.com/kvcache-ai/Mooncake/pull/1485) | fix: unregisterLocalMemory syncs to metadata | `2e601a1` |
| [#993](https://github.com/kvcache-ai/Mooncake/pull/993) | [Store] Add Timeout Mechanism for Put Operations | `6d05c93` |

### 本地 HEAD **不**包含的相关 PR（已验证）

| PR | 标题 | 验证方式 |
| --- | --- | --- |
| [#1219](https://github.com/kvcache-ai/Mooncake/pull/1219) | [Store] Delete useless meta from http metadata server | `removeKey` / `removeKeys` 在 `mooncake-store/` 下无匹配 |
| [#1363](https://github.com/kvcache-ai/Mooncake/pull/1363) | [WIP][Store] Add HTTP metadata cleanup on client timeout | `enable_metadata_cleanup_on_timeout` 无匹配 |
| [#1826](https://github.com/kvcache-ai/Mooncake/pull/1826) | [TENT] withCachedSegment async invalidation | `withCachedSegment` 全树无匹配 |

---

## 第 1 节：什么是 `metadata not found`

**一句话定义**：HTTP metadata server 收到 `GET /metadata?key=X`，本地 `std::unordered_map<string,string>` 里没有 key X，返回 HTTP 404 + body `metadata not found`。

两处字面 404 抛出点（**仅此两处**）：
- GET 路径：[mooncake-store/src/http_metadata_server.cpp:38-41](../../Mooncake/mooncake-store/src/http_metadata_server.cpp#L38-L41)
- DELETE 路径：[mooncake-store/src/http_metadata_server.cpp:100-104](../../Mooncake/mooncake-store/src/http_metadata_server.cpp#L100-L104)

典型 key 命名：
- `mooncake/<cluster_id>/ram/<segment_name>` —— RDMA 数据面（`devices / buffers / lkey / rkey`）
- `mooncake/<cluster_id>/rpc_meta/<segment_name>` —— 控制面（`ip_or_host_name + rpc_port`）
- `<cluster_id>` 由 `MC_METADATA_CLUSTER_ID` 环境变量决定
- `<segment_name>` 一般是 `hostname:rpc_port`

**重要区分**：`metadata not found` ≠ `OBJECT_NOT_FOUND(-704)`。前者是**HTTP server 层**查不到 segment 元信息；后者是 **master service 层**查不到 object 元信息。两条是独立链路，属不同家族（家族 1 vs 家族 3）。

---

## 第 1.5 节：概念基础

读者若不熟 Mooncake 架构，先看这两点，否则后面每条触发原因都会看得糊里糊涂。

### 1.5.1 local segment 的完整创建流水

一个 Transfer Engine 进程 = **一个** local segment。但 segment 不是"一个动作"产生的，是 `init` 过程中的**多步**结果。按 RDMA 路径展开（入口 [transfer_engine_impl.cpp:67](../../Mooncake/mooncake-transfer-engine/src/transfer_engine_impl.cpp#L67)）：

1. **决定 `local_server_name_`**（[:102, :107, :139-143](../../Mooncake/mooncake-transfer-engine/src/transfer_engine_impl.cpp#L102)）
   - Ascend：从 `ip:port:npu_x` 解成 `host:port`
   - 一般 RDMA：直接用调用方传入的 `local_server_name`
   - p2p handshake：忽略传入的，自己 `findAvailableTcpPort()` 拼 `ip:port`

2. **构造 `TransferMetadata`**（[:182](../../Mooncake/mooncake-transfer-engine/src/transfer_engine_impl.cpp#L182)）：根据 `conn_string` 选 handshake plugin 或 storage plugin

3. **PUT `rpc_meta/<name>`**（[:192](../../Mooncake/mooncake-transfer-engine/src/transfer_engine_impl.cpp#L192) → `addRpcMetaEntry`）—— 此时 segment 还没造，但 RPC 身份已广播到 HTTP server

4. **`MultiTransport::installTransport` + topology discover**（[:186-243](../../Mooncake/mooncake-transfer-engine/src/transfer_engine_impl.cpp#L186-L243)）：枚举 `ibv_get_device_list()`（[topology.cpp:150-197](../../Mooncake/mooncake-transfer-engine/src/topology.cpp#L150-L197)），filter 后保存在 `local_topology_`

5. **`initializeRdmaResources()`**（[rdma_transport.cpp:104](../../Mooncake/mooncake-transfer-engine/src/transport/rdma_transport/rdma_transport.cpp#L104)）：对每张 HCA open ibv context，拿 `lid / gid / pd / cq`，填进 `context_list_`

6. **`allocateLocalSegmentID()`**（[:110, :357-373](../../Mooncake/mooncake-transfer-engine/src/transport/rdma_transport/rdma_transport.cpp#L357-L373)）：new 一个 `SegmentDesc`，`desc->name = local_server_name_`，遍历 `context_list_` 填 `desc->devices`，`addLocalSegment` 放进进程内存的 `segment_id_to_desc_map_[LOCAL_SEGMENT_ID]`。**此时还没 PUT 到 HTTP server**，`buffers=[]`

7. **`updateLocalSegmentDesc()`**（[rdma_transport.cpp:123](../../Mooncake/mooncake-transfer-engine/src/transport/rdma_transport/rdma_transport.cpp#L123) → [transfer_metadata.cpp:762-773](../../Mooncake/mooncake-transfer-engine/src/transfer_metadata.cpp#L762-L773)）：encode desc 成 JSON，调 `storage_plugin_->set()` PUT `ram/<name>` → **HTTP server 上第一次出现 `ram/` key**（buffers 为空）

8. **每次 `registerLocalMemory()`**（[rdma_transport.cpp:174-299](../../Mooncake/mooncake-transfer-engine/src/transport/rdma_transport/rdma_transport.cpp#L174-L299) → [transfer_metadata.cpp:795-807](../../Mooncake/mooncake-transfer-engine/src/transfer_metadata.cpp#L795-L807)）：对每张 HCA `ibv_reg_mr` 拿 `lkey[]/rkey[]`，`addLocalMemoryBuffer` 追加到 `desc->buffers`，若 `update_metadata=true` 再 PUT 一次 `ram/<name>`

**时间线（什么时刻 HTTP server 上能看到什么）**：
```
t0  init 开始
t1  第 3 步后：HTTP server 上有 rpc_meta/<name>
t2  第 7 步后：HTTP server 上有 ram/<name>，但 buffers=[]
t3  第 8 步首次 register_buffer 后：ram/<name> 里 buffers=[layer_0]
t4+ 每次 register_buffer 都重写 ram/<name>
```

**推论**：
- device 数 = 第 5 步枚举到的可用 HCA 数
- segment 数永远 = 1（per process）
- buffer 数 = 累计 `registerLocalMemory()` 次数
- 注册 memory **不创建新 segment，只追加 buffer**
- 家族 1 的 stale metadata 本质是：**同一个 `<name>` 身份在 HTTP server 上残留着已消失进程的 `rpc_meta/` 或 `ram/` 条目**，新进程用同一身份启动时，要么被 PUT 拒绝（见 1.5.2 唯一性约束），要么和 peer 的本地 cache 对不上

### 1.5.2 `ram/` 和 `rpc_meta/` 为何必须是两个 key

五个维度全不同：

| 维度 | `rpc_meta/<name>` | `ram/<name>` |
| --- | --- | --- |
| 写入时机 | TE 启动、RPC 端口 bind 成功时第一时间写（第 ③ 步） | 第 ⑦ 步后 PUT 空 desc，⑧ 步起随 buffers 变化刷新 |
| 更新频率 | 进程存活期间不变 | 每次 register/unregister buffer 都重写 |
| 唯一性约束 | **拒绝重复 PUT**（[http_metadata_server.cpp:64-73](../../Mooncake/mooncake-store/src/http_metadata_server.cpp#L64-L73) 防身份冲突） | PUT 允许覆盖 |
| 访问路径 | 握手前查一次拿 `ip:port` | 通常握手消息直接带，本地 cache 失效才回源 |
| p2p handshake 模式 | 豁免，不写 storage（[transfer_metadata.cpp:843-863](../../Mooncake/mooncake-transfer-engine/src/transfer_metadata.cpp#L843-L863)） | 依然需要 |

**关于 `segment_name` 格式**：`segment_name` 的值在不同模式下由**不同主体**决定。**不是一定要 `host:port`**：

- **一般 RDMA 模式**：`local_server_name_ = local_server_name`（[transfer_engine_impl.cpp:107](../../Mooncake/mooncake-transfer-engine/src/transfer_engine_impl.cpp#L107)），**原样用调用方传入的字符串**，TE 不强制格式。理论上可以是任意 key，但 vLLM / Mooncake Store / SGLang 等上层**按约定都传 `host:port`**。
- **p2p handshake 模式**：TE 自己 `findAvailableTcpPort()` 拼 `ip:port`（[:139-143](../../Mooncake/mooncake-transfer-engine/src/transfer_engine_impl.cpp#L139-L143)），**强制** `host:port` 格式。原因：p2p 模式没有 HTTP metadata server，也就没有 `rpc_meta/` key；控制面信息必须**编码在 name 本身**里。`getRpcMetaEntry` 的 p2p 分支直接 `parseHostNameWithPort(server_name)` 解 ip 和 port（[transfer_metadata.cpp:896-899](../../Mooncake/mooncake-transfer-engine/src/transfer_metadata.cpp#L896-L899)）。
- **Ascend 模式**：从 `ip:port:npu_x` 解出再拼 `host:port`。

**对比总结**：
- 一般模式：`segment_name` 是**不透明键**，`rpc_meta/<name>` 存 `{ip, port}` 补信息
- p2p 模式：`segment_name` **自带** ip 和 port，不需要 `rpc_meta/` 补充

这也解释了为什么 p2p 能豁免 rpc_meta：不是"不需要"，而是"信息已经编码在 name 里，无需单独存一条"。

**推论**：任何 metadata 清理机制（例如 PR #1363 的 `cleanupHttpMetadata()`）必须**同时**处理 `mooncake/ram/*` 和 `mooncake/rpc_meta/*`。只清一条就会形成"握手能通但数据面 key 丢"或反过来的半状态，病症会更诡异。

---

## 第 2 节：从 GET 到 404 的完整调用栈

自顶向下，一次 `batch_put failed` 的错误日志链可以回溯到这条栈：

```
vLLM MooncakeStoreConnector
  → store.batch_put(keys, values)
  → client_service.cpp::BatchPutStart / SubmitTransfers
  → transfer_task.cpp::SubmitTransfers → openSegment(transport_endpoint_)
       三处 "Failed to open segment" 日志点：
         - transfer_task.cpp:493     (Read path)
         - transfer_task.cpp:535     (Write path)
         - transfer_task.cpp:657     (Replica path)
  → TransferEngineImpl::openSegment  [tent/src/runtime/transfer_engine_impl.cpp:465-471]
  → SegmentManager::openRemote  → HTTP GET /metadata?key=mooncake/<cluster_id>/ram/<segment_name>
       ↓
  HttpMetadataServer GET handler  [http_metadata_server.cpp:26-47]
       key 不在 store_  → 返回 404 "metadata not found"
       ↓
  openRemote 得到 404  → 返回 ERR_INVALID_ARGUMENT
  openSegment 得到 ERR_INVALID_ARGUMENT  → Handle = (SegmentHandle)ERR_INVALID_ARGUMENT
  → transfer_task.cpp 检查到 seg == ERR_INVALID_ARGUMENT  → LOG(ERROR) "Failed to open segment"
  → 上层把该 slice 标记 TRANSFER_FAIL(-800)
  → client_service.cpp WaitForTransfers 收到 -800
  → Python 层 MooncakeStoreConnector.batch_put 聚合成 "batch_put failed"
```

一个 `metadata not found` 字面日志出现在 HTTP server 侧；对应的客户端侧日志是 `Failed to open segment endpoint='...'`。**两条日志通常同时出现**，排查时两侧都要看。

---

## 第 3 节：所有可能让 key 缺失的触发原因

按"从未写入" vs "写入后丢失"分成两类。每条给：机制 / 源码定位 / 触发条件 / 本地修复状态 / 对我们 bench 的影响。

### A. Key 从未写入

#### 原因 1：HTTP metadata server 进程重启 / 崩溃

- **机制**：HTTP server 的 `store_` 是纯内存 `std::unordered_map<string,string>`（[http_metadata_server.cpp:12-18](../../Mooncake/mooncake-store/src/http_metadata_server.cpp#L12-L18)），无持久化、无 backup。进程一重启**全部 key 丢失**，但客户端本地 `segment_id_to_desc_map_` cache 还记得那些 segment。下一次 `getSegmentDesc` 走回源 → 404。
- **触发条件**：HTTP server OOM / panic / deploy 重启 / 运维手动 kill
- **上游修复状态**：**无任何 PR 讨论持久化**。#1219 只能补周期性清理，不能恢复已丢失的数据。
- **对我们 bench**：如果 metadata server 容器 crash 过一次，整轮 bench 就会大面积 `metadata not found`。**建议排查时先查 HTTP server 进程的 uptime**。

#### 原因 2：客户端启动 race —— peer B 还没 register 完，client A 已经来查

- **机制**：worker A 从 master 拿到一组 endpoint，去 HTTP server GET `ram/<B>`；此时 worker B 的 TE 还在枚举 RDMA device / pin memory，尚未完成首次 `register_buffer`（`addLocalSegment` + HTTP PUT）。GET 先到 → 404。
- **源码**：[transfer_metadata.cpp:625-633](../../Mooncake/mooncake-transfer-engine/src/transfer_metadata.cpp#L625-L633) 的 `getSegmentDesc` 只做一次 GET，没有"重试 + 等待" 逻辑。
- **触发条件**：多 worker 并发启动 + master 对 segment 可见性没有 barrier
- **上游修复状态**：issue #1115（同一 Python 进程多客户端的 race）属这一类，open 未修。我们本地无此场景。
- **对我们 bench**：可能性存在，但瞬态 404 通常被上层重试吃掉，不是家族 1 的主因。

#### 原因 3：transport 不匹配 / cluster_id 错配

- **机制 a**：producer 和 consumer 的 `MC_METADATA_CLUSTER_ID` 环境变量不同。producer 写 `mooncake/A/ram/...`，consumer 读 `mooncake/B/ram/...` → 404。参考 [transfer_metadata.cpp:111-121](../../Mooncake/mooncake-transfer-engine/src/transfer_metadata.cpp#L111-L121)。
- **机制 b**：ADXL / Ascend 等非 RDMA transport 在某些场景下不写 `ram/` key（issue #1260），但 peer 以 RDMA 方式去查，返回 404。
- **对我们 bench**：若 vigil 的 rendered_config 正确一致，不应命中。**排查第一步应确认两端 `MC_METADATA_CLUSTER_ID` 相同**。

### B. Key 写入后丢失 / 过期

#### 原因 4：跨 job stale `transport_endpoint_`（**家族 1 的真正根因**）

- **机制**：前一轮 bench 的 prefill worker 正常或异常退出后，HTTP server 上的 `ram/<host:port>` 条目**没清干净**（见原因 6）。同时 master 里的 segment 状态在 TTL 到期后被清空。本轮启动后，某个新的 vLLM worker 尝试"恢复"它从 nixl/某种 discovery 机制拿到的 endpoint 列表，GET `ram/<oldhost:oldport>` → HTTP server 上**可能还有**但**值已过期**，或**已被覆盖但仍不匹配**，或**真的找不到**。
- **注**：具体命中 404 还是命中"值过期但对应 RDMA 资源不可用"取决于清理路径。前者归本原因，后者归家族 2。
- **源码定位**：HTTP server 缺 client-level 感知，没有代码路径在 master 侧通知它"这个 client 挂了，对应 key 可以删"。
- **上游修复状态**：**PR #1363 直接针对此场景，但未合入本地**。#1219 作为备选路线也未合入。
- **对我们 bench**：**这是 [decode5_mtc16_mooncake_correlation_analysis_20260415.md](decode5_mtc16_mooncake_correlation_analysis_20260415.md) 的家族 1 主因**。必须 patch 本地。

#### 原因 5：客户端正常 unmount 但 HTTP DELETE 失败

- **机制**：进程正常退出时会调 `removeSegmentDesc` / `removeRpcMetaEntry`（[transfer_metadata.cpp:358-378, 875-884](../../Mooncake/mooncake-transfer-engine/src/transfer_metadata.cpp#L358-L378)）对 HTTP server 发 DELETE。若网络瞬态 / HTTP server 高负载丢了这个请求，key 残留。
- **上游修复状态**：**PR #1485 已合入**（`unregisterLocalMemory` 会同步 metadata），但**只覆盖"注销 memory buffer"一侧，不覆盖"进程退出时完整清理"**。失败路径没有重试逻辑。
- **对我们 bench**：概率事件，不是主因；但贡献给家族 1 的 tail。

#### 原因 6：客户端 `kill -9` / OOM / 节点宕机

- **机制**：客户端来不及执行 `removeSegmentDesc` 就已经消失。Master Service 有心跳机制（[master_service.cpp:3644-3736](../../Mooncake/mooncake-store/src/master_service.cpp#L3644-L3736) 的 `ClientMonitorFunc`），TTL 过期后 `PrepareUnmountSegment + CommitUnmountSegment` **清自己的**内部 segment 状态。**但 master 不会通知 HTTP metadata server 删对应的 key**。
- **验证**：`master_service.cpp` 里搜 `removeKey` / `removeKeys` 0 匹配，`enable_metadata_cleanup_on_timeout` 0 匹配 —— 本地代码**没有**任何 master → HTTP server 的反向删除路径。
- **上游修复状态**：**PR #1363 正是为了填这个缺口，但未合入本地**。
- **对我们 bench**：vLLM worker 被 SLURM preempt / SIGKILL 的场景会直接命中。结合 benchmark 里 `prefill_9_mtc_144` 的 3 次 slurm preemption，这条**一定被命中过**。

#### 原因 7：disk eviction 导致 descriptor 失效

- **机制**：磁盘层淘汰某个 object 的 disk replica 后，master 侧的 `DiskDescriptor` 会变 stale，客户端 Get 时看到 `metadata not found`（但这条严格说不是 HTTP server 的 404）。
- **上游修复状态**：**PR #1549 已合入本地**。磁盘淘汰时会通知 master，fix 这一路径。
- **对我们 bench**：我们没开 disk tier（纯 DRAM），**不相关**。

#### 原因 8：客户端本地 segment cache 过期 / 不一致

- **机制**：客户端 `segment_id_to_desc_map_` 缓存了 peer 的 segment desc。peer 改变了 `buffers` 或 `devices`（比如重新 register），客户端仍用旧 cache 拿旧 `rkey` 去写 → RDMA 返回 remote access error。这**不是 HTTP 404**，但症状相似，容易混进家族 1。
- **上游修复状态**：**PR #1826（withCachedSegment） 未合入本地**（`withCachedSegment` 全树 0 匹配）。即便合入也**只影响 TENT 子模块**；主 TE 走的 `syncSegmentCache`（[transfer_metadata.cpp:639-673](../../Mooncake/mooncake-transfer-engine/src/transfer_metadata.cpp#L639-L673)）依赖 TTL。
- **对我们 bench**：vLLM `MooncakeStoreConnector` 走的是主 TE，本修复不直接受益。

#### 原因 9：同节点重启用不同 RDMA 参数

- **机制**：worker 被 kill 后重启，复用同一 `hostname:rpc_port` 身份，但 `gid / lid / rkey` 全变。HTTP server 上 key 还在，值是**新的**（PUT 覆盖了），但如果此前有 peer 在并发访问，那个 peer 会拿到**半新半旧的拼接**（极少见）或**新值但 cache 未刷**。
- **归类警告**：这条**表面像家族 1**（metadata 看起来"不对"），**实际属家族 2**（RDMA 握手 / endpoint state）。症状是 `Failed to modify QP to RTR` / `packet mismatch` 而非 `metadata not found`。
- **对我们 bench**：在 `autosweep_20260417_135116_bug_census_cn.md` 里观察到的就是这条。

---

## 第 4 节：原因到家族的映射

| 原因 | 字面产生 `metadata not found`？ | 实际归属家族 | 本地是否已修 |
| --- | :---: | :---: | :---: |
| 1. HTTP server 重启 | ✓ | 家族 1 | ✗（无任何 PR） |
| 2. 启动 race | ✓（瞬态） | 家族 1 | 部分（上层重试） |
| 3. cluster_id / transport 错配 | ✓ | 家族 1 | 部署层问题 |
| 4. 跨 job stale `transport_endpoint_` | ✓ | **家族 1 主因** | ✗（#1363 / #1219 未合） |
| 5. 正常 unmount 但 DELETE 失败 | ✓ | 家族 1 | 部分（#1485 ✓，但不重试） |
| 6. `kill -9` / OOM / 宕机 | ✓ | 家族 1 | ✗（#1363 未合） |
| 7. disk eviction | ✓ | 家族 1 | ✓（#1549 已合，但我们不开 disk） |
| 8. 客户端 cache 过期 | ✗（是错值而非 404） | 家族 1 相邻 | ✗（#1826 未合） |
| 9. 同节点重启 RDMA 参数变 | ✗（是 QP 握手错） | **家族 2** | ✓（#1705 / #1733 / #1803 已合） |

---

## 第 5 节：本地代码里没有的修复机制（缺口清单）

按优先级排：

1. **master TTL 过期 → 级联 DELETE HTTP server key**
   - 缺口对应：原因 6（最主要）
   - 上游路线：PR #1363，open（WIP）
   - 实现复杂度：中。需要 `master → HTTP server` 同进程部署前提下加一条调用。
   - **短期建议**：cherry-pick #1363 本地打 patch，`--enable_metadata_cleanup_on_timeout=true`。

2. **HTTP server 周期性自查 → 删除 master 不认识的 key**
   - 缺口对应：原因 6（冗余方案） + 原因 1 部分（可以删除 stale）
   - 上游路线：PR #1219，open，有 O(N²) 性能嫌疑
   - 与 #1363 是替代关系。维护者路线尚未定稿。

3. **HTTP server 持久化 / 重启恢复**
   - 缺口对应：原因 1
   - 上游路线：**无任何 PR**。需要考虑一致性模型（是否和 master 对账、LWW vs 强一致、崩溃一致性等）。
   - **建议向上游提 issue**。

4. **vLLM `MooncakeStoreWorker.__del__` / 显式 `unregister_buffer` 调用**
   - 缺口对应：原因 5（进程关闭不发 DELETE）
   - 本地位置：[vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_worker.py:864](../../vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_worker.py#L864) 只调 `register_buffer`，该文件**没有** `__del__` 或 `unregister_buffer` 调用。
   - 注：同目录 `mooncake_connector.py:771` 和 `mooncake_utils.py:60` 有 `__del__`，但都不清理 store worker 的 buffer。
   - **短期建议**：在 `MooncakeStoreWorker` 加一个 `__del__` / context manager，退出时显式 `unregister_buffer`。

5. **客户端 cache 失效策略**
   - 缺口对应：原因 8
   - 上游路线：#1826（TENT only），我们走 TE 不受益
   - 长期看主 TE 也应迁移到类似 `withCachedSegment` 的"失败即 invalidate + refetch"模式。

---

## 第 6 节：诊断 checklist

oncall 在一次 `metadata not found` 出现时按顺序检查：

1. **HTTP metadata server 进程是否最近重启过？**
   - `docker ps` / `ps -o etime= -p <pid>` 看 uptime
   - 若 uptime 短于 bench 起始时间 → 命中原因 1

2. **涉事的 segment_name 在 master `GetAllSegments` 输出里吗？**
   - 若**在** master 里但**不在** HTTP server 里 → 命中原因 6
   - 若**不在** master 里但**在** HTTP server 里 → 命中原因 6 的另一面（key 已 stale）
   - 两侧都**不在** → 命中原因 1 或原因 2

3. **直接 GET HTTP server 看得到这个 key 吗？**
   ```bash
   curl -s "http://<metadata_server>/metadata?key=mooncake/ram/<segment_name>"
   curl -s "http://<metadata_server>/metadata?key=mooncake/rpc_meta/<segment_name>"
   ```
   - 两条都 404 → 原因 1 / 2 / 6
   - 只有 `ram/` 404 → 原因 5 / 7
   - 只有 `rpc_meta/` 404 → 罕见，需看 `addRpcMetaEntry` 是否失败过

4. **上一次该 worker 退出是 normal 还是 SIGKILL？**
   - `kubectl events` / `dmesg` / `journalctl` 看 OOM / preempt 记录
   - SIGKILL / OOM → 命中原因 6

5. **master TTL 配置（`--client_ttl`）vs worker 启动间隔？**
   - TTL 默认 10s。若 worker 重启间隔 < TTL，会出现"master 认为 client 还活着但进程已换身份"的窗口 → 命中原因 9

6. **vLLM worker 是否调过 `register_buffer` 的对偶 `unregister_buffer`？**
   - `grep -rn unregister_buffer vllm/distributed/kv_transfer/kv_connector/v1/mooncake/`
   - 本地确认**没有** → 原因 5 的概率放大

7. **两端 `MC_METADATA_CLUSTER_ID` 是否一致？**
   - 错配则所有 GET 永远 404 → 原因 3

---

## 第 7 节：建议方向（仅记录，不在本文档落实）

### 短期（我们自己能做）

1. **vLLM `MooncakeStoreWorker` 加 `__del__` / 显式 `unregister_buffer`** —— 堵原因 5。
2. **给 bench 脚本加一步"metadata server 启动时间校验"** —— 堵原因 1 的诊断。
3. **cherry-pick #1363 打本地 patch** —— 堵原因 6。

### 中期（等上游）

1. **跟进 #1219 / #1363 路线决策** —— 维护者定稿前不深投入。
2. **把 #1485 的 "DELETE 失败重试" 补上** —— 向上游提 issue。

### 长期

1. **HTTP metadata server 持久化 / 重启恢复** —— 无上游 PR，值得新 issue。
2. **主 TE 客户端 cache 失效策略迁移到 `withCachedSegment` 模式** —— 当 #1826 合入稳定后向 TE 侧推广。

---

## 验证记录

本文论点的源码行号均来自本地 Mooncake HEAD `be75ca0`。核对方式：

- `grep -n "metadata not found" mooncake-store/src/http_metadata_server.cpp` → 仅两行命中（`:41`、`:104`）✓
- `grep -rn "removeKey\|removeKeys" mooncake-store/` → 仅 [`http_metadata_server.h`](../../Mooncake/mooncake-store/include/http_metadata_server.h) 的 `removeSegmentDesc` / `removeRpcMetaEntry` 实现，`master_service.cpp` 中**无**匹配 ✓（证明 master 不通知 HTTP server）
- `grep -rn "enable_metadata_cleanup_on_timeout" mooncake-store/` → 0 匹配 ✓（证明 #1363 未合入）
- `grep -rn "withCachedSegment" Mooncake/` → 0 匹配 ✓（证明 #1826 未合入）
- `git log --oneline --all | grep -E "1549|1624|1485|1705|1733|1803|993"` → 7 条全部命中（证明列表中已合入的 PR 确实在本地）
- `grep -n "Failed to open segment" mooncake-store/src/transfer_task.cpp` → `:493, :535, :657` ✓
- vLLM 端：`grep -n "register_buffer\|__del__" vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_worker.py` → 仅 `:864` 的 `register_buffer` 调用，**无** `__del__` / `unregister_buffer` ✓
