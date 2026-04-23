# Bug 类别 1：metadata not found（元数据 / 段描述符失效）

- **合并日期**：2026-04-22
- **本地 Mooncake HEAD**：`be75ca0`（2026-04-18, "client_service: quorum write on WaitForTransfers + richer failure logging"）
- **合并自**：
  - `metadata_not_found_root_causes_20260419_cn.md`（9 条触发原因白皮书）
  - `decode5_mtc16_mooncake_correlation_analysis_20260415.md` + `decode5_mtc16_mooncake_correlation_analysis_20260415_cn.md`（主证据 + 因果链）
  - `mooncake_upstream_issues_prs_20260419_cn.md` 家族 1 章节（PR 全景）
  - `prefill3_decode5_mooncake_deep_dive_20260415_cn.md`（交叉引用，本类非主轴）
- **证据权重**：**Strong hypothesis**（多 run 交叉 + 源码 grep-verified + 上游 PR body 一字对应）

### 在三层模型中的位置：Layer 2（Segment Layer）

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Layer 3:  OBJECT LAYER (业务数据：KV-cache value)         → 类别 3      │
│  错误    → OBJECT_NOT_FOUND(-704) / batch_put_failed 聚合               │
├─────────────────────────────────────────────────────────────────────────┤
│  Layer 2:  SEGMENT LAYER (peer 身份地址本)               ← 本文档        │
│           "RDMA 能通，但找不到 peer 的 gid/lid/ip:port 地址本"           │
│  错误    → HTTP 404 "metadata not found"                                │
│  归属    → HTTP metadata server store_ map（在 master 进程内）           │
├─────────────────────────────────────────────────────────────────────────┤
│  Layer 1:  CONNECTION LAYER (RDMA QP 物理握手)          → 类别 2         │
│  错误    → QP to RTR 失败 / packet mismatch / Connection timed out[110] │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 1. 摘要

**最直观的一句话**：**peer 想找 segment X，但 HTTP metadata server 上 X 的"地址本条目"没了**。

一个 Transfer Engine 进程 = 一个 segment（进程级唯一，详见 §3.4.5）。这个 segment 的"联系方式"被拆成 HTTP metadata server 上的**两条 key**：
- `ram/<segment_name>` = **数据面地址**（RDMA 建 QP 要的 `gid/lid` + 内存定位要的 `addr/rkey`）
- `rpc_meta/<segment_name>` = **控制面地址**（握手/通知要的 `ip:rpc_port`）

任意一条 key 缺失 → HTTP 404 "metadata not found" → peer 不知道往哪儿发数据 → 一整套错误链。

**技术一句话**：HTTP metadata server 对 `GET /metadata?key=mooncake/<cluster_id>/ram/<segment_name>` 返回 **HTTP 404 "metadata not found"**，触发自下而上的 6 层错误链，最终在 vLLM Python 层表现为 `batch_put failed` 聚合错误（`res[i] = -800`）。

**本地 bench 命中情况**：
- `decode_5_mtc_16` run 中，prefill-0 侧有 **69 个唯一失败 `req_id`** 命中此链，完整覆盖 turn-band 1 / 2 + turn-band 3 最早 5 个请求。
- 报错窗口（16:46:02–16:48:50，169 秒）内请求完成速率降到 `0.402 req/s`；窗口结束后跃升至 `1.565 req/s`，**3.9× 阶跃**。
- External prefix hit-rate 在邻居 run（decode_4 / decode_6）均达 79.3%；`decode_5_mtc_16` 仅 61.7%，且从未触达 70%。

**根因假设**：共享 HTTP metadata server 里残留了上一个 job 的 `transport_endpoint_`（即 `ram/<host:port>` / `rpc_meta/<host:port>`）。**9 条独立触发路径**中，本 bench 最可能命中原因 4（跨 job stale）+ 原因 6（`kill -9` / OOM / 宕机）。

**已排除的原因**（本 bench 特定）：
- **原因 1（HTTP server 重启）已排除**：HTTP metadata server 是 master 进程内的对象（非独立进程/容器），代码无自动重启机制（`master.cpp:14, :138-142, :316-324` + `http_metadata_server.cpp` 只有 `start()` / `stop()` 两个方法），用户确认 bench 期间未手动重启 master。
- **原因 7（disk eviction）已排除**：bench 未配置 disk tier（`rendered_config.yaml` 无 `local_buffer_dir` / `disk_dir` / `MC_STORE_ENABLE_DISK` 等参数）。

**修复状态一句话**：上游 PR **#1363 直接对症本 bug**，但 **open (WIP)**，本地未包含。短期需 cherry-pick；长期需要 HTTP server 持久化（无任何上游 PR）。

---

## 2. 症状链（日志指纹）

```
[HTTP metadata server]         "metadata not found" (404)
         ↓
[TransferEngine]               metadata get 失败 → openSegment 返回 ERR_INVALID_ARGUMENT(-1)
         ↓
[TransferSubmitter]            "Failed to open segment <endpoint>"
         ↓
                               submit_batch() 返回 nullopt
         ↓
[ClientService::SubmitTransfers] "Transfer submission failed for key ..."
         ↓                       SetError(ErrorCode::TRANSFER_FAIL)
[ClientService::finalize]       "Operation for key ... failed: TRANSFER_FAIL"
         ↓                       toString() 映射 → -800
[vLLM Python]                   "batch_put failed: ..."
```

**日志计数几乎相等**：decode_5 run 中 `metadata not found` / `Failed to open segment` / `Transfer submission failed` 计数都在 ~106994 量级——本质上是**同一事件的 4 个投影**。

**本链**不含：`handshake timeout` / `packet mismatch` / `mark it inactive` / `Failed to complete transfers after 60 seconds`（那些属 bug 类别 2 RDMA 握手）。也不含 `OBJECT_NOT_FOUND(-704)`（那属 bug 类别 3 master 层 object 元数据）。

---

## 3. 根因分析

### 3.0 名词澄清：三种 "metadata" 到底指什么

"metadata" 在 Mooncake 里被**不严谨地用于三样完全不同的东西**，混用会把 bug 归因带偏。本类的 "metadata not found" **只指第 1 种**。

| 种类 | 存在哪里 | 数据结构 | 写者 | 读者 | 缺失时的错误 | 归属 bug 类别 |
| --- | --- | --- | --- | --- | --- | --- |
| **① 段元数据**（segment metadata） | **HTTP metadata server 的内存 map** | `std::unordered_map<string,string> store_`（`http_metadata_server.h:48`），value 是 JSON 序列化的 `SegmentDesc`（含 protocol / device / buffer / topology） | 每个 Transfer Engine 进程启动时 HTTP PUT | 对端 TE 建 RDMA 连接前 HTTP GET | **HTTP 404 "metadata not found"**（`http_metadata_server.cpp:40-41`） | **本类（类别 1）** |
| **② 对象元数据**（object metadata） | **Master Service 的内存表** | 内部 KV 对象记录（per-key replica list、state、TTL） | master 内部 RPC 服务（`PutStart` / `PutEnd`） | 客户端 `GetReplicaList` 等 RPC | `ErrorCode::OBJECT_NOT_FOUND(-704)` | **类别 3** |
| **③ 本地客户端缓存** | **每个 TE 客户端的进程内存** | `segment_id_to_desc_map_` + `segment_name_to_id_map_`（`transfer_metadata.h:215-217`） | `getSegmentDescByName` / `getSegmentDescByID` 拉回后写入 | 同一 TE 客户端的所有传输操作 | 触发新 HTTP GET，若 HTTP server 也无 → 回落到种类 ① 的 404 | 跨类（TTL / cache 一致性） |

**关键事实：HTTP metadata server 不是独立进程**。它是 `mooncake_master` 二进制启动时的 **同进程组件**（`master.cpp:14, :138-142, :316-324`），由 `--enable_http_metadata_server=true` 开关决定是否启用。master 进程里同时跑着：
- Master Service RPC 线程组（处理对象元数据）
- HTTP metadata server 的 `coro_http_server` 异步线程组（处理段元数据）
- ClientMonitor 心跳线程（`master_service.cpp:3644-3736`）

**这解释了两个推论**：
1. 用户没重启 master → 种类 ① 的段元数据 map 不会丢（原因 1 排除）。
2. master TTL 触发 `UnmountSegment` 时，只清了 master 自己的种类 ②，**不会联动清 HTTP server 的种类 ①**——这正是 PR #1363 想补的缺口。

本文后续所有 "metadata not found" / "stale metadata" / "key 残留" 都是指**种类 ①**。种类 ② 和 ③ 仅在 §10 交叉引用里出现。

### 3.1 从 HTTP 404 到 vLLM `batch_put failed` 的完整调用栈

错误**自下而上**传播。一次 `batch_put failed` 对应一段确定的 C++ 错误链，产生 4 条不同文件的 ERROR 日志：

| # | 错误消息 | 源文件 | 行号 | 函数 / 条件 |
|---|---|---|---|---|
| 1 | `metadata not found` | `Mooncake/mooncake-store/src/http_metadata_server.cpp` | 39 / 87 | GET / DELETE handler，key 不在 `store_` map 里 |
| 1b | `metadata not found` | `Mooncake/mooncake-wheel/mooncake/http_metadata_server.py` | 80 / 100 | Python server（本 bench 用的是 HTTP C++ 版） |
| 2 | _(无日志)_ | `Mooncake/mooncake-transfer-engine/src/transfer_engine_impl.cpp` | 434-441 | `TransferEngineImpl::openSegment()` → 返回 `ERR_INVALID_ARGUMENT` |
| 2b | _(无日志)_ | `Mooncake/mooncake-transfer-engine/src/transfer_metadata.cpp` | 851-879 | `getSegmentDesc()` → `storage_plugin_->get()` 404 → `nullptr` |
| 3 | `Failed to open segment <endpoint>` | `Mooncake/mooncake-store/src/transfer_task.cpp` | **493** | `TransferSubmitter::submit_batch()` |
| 3b | `Failed to open segment <endpoint>` | `Mooncake/mooncake-store/src/transfer_task.cpp` | 535 | `submit_batch_get_offload_object()` |
| 3c | `Failed to open segment for endpoint='...'` | `Mooncake/mooncake-store/src/transfer_task.cpp` | **657** | `submitTransferEngineOperation()` |
| 4 | `Transfer submission failed for key <key>: <ctx>` | `Mooncake/mooncake-store/src/client_service.cpp` | **1556** | `SubmitTransfers()` → `SetError(TRANSFER_FAIL)` |
| 4b | `Transfer submission failed for key <key>: <ctx>` | `Mooncake/mooncake-store/src/client_service.cpp` | **1948** | 合并 upsert 路径 |
| 5 | `Transfer failed for key ... quorum=0/N (all replicas failed)` | `Mooncake/mooncake-store/src/client_service.cpp` | **1639** | finalize 阶段；`toString()` 查表 `types.cpp:38` → `-800`（**注**：本地 HEAD 已改为 quorum 语义，见类 2 §3.9.b） |
| 6 | `batch_put failed: ...` | `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_worker.py` | 480 | vLLM Python 层，`res[i] < 0` |

**Python 层信息受限**：只能看到 `-800`，中间所有 endpoint / replica / segment_name 信息都丢失。追溯源头必须看 C++ 侧日志。

### 3.2 让 key 缺失的 9 条独立触发原因

> 源自 `metadata_not_found_root_causes_20260419_cn.md §3`，**全量搬运**，按"从未写入" vs "写入后丢失"分类。

#### A. Key 从未写入

**原因 1：HTTP metadata server 进程重启 / 崩溃**
- **机制**：HTTP server 的 `store_` 是纯内存 `std::unordered_map<string,string>`（`http_metadata_server.cpp:12-18`），无持久化。进程一重启**全部 key 丢失**，但客户端本地 `segment_id_to_desc_map_` cache 还记得那些 segment。下一次 `getSegmentDesc` 走回源 → 404。
- **上游修复**：**无任何 PR 讨论持久化**。#1219 能补周期清理，不能恢复已丢。
- **对本 bench**：**已排除**。HTTP metadata server 是 `mooncake_master` 进程内的组件（`master.cpp:14, :138-142, :316-324`，受 `--enable_http_metadata_server` 控制），**不是独立进程也不是独立容器**。代码只有 `start()` / `stop()`，无 supervise / respawn / daemon 逻辑。用户确认 bench 期间未手动重启 master，因此种类 ① 段元数据 map 不会丢失。

**原因 2：客户端启动 race —— peer B 还没 register 完，client A 已经来查**
- **机制**：worker A 从 master 拿到 endpoint 列表后立刻 GET `ram/<B>`；此时 worker B 的 TE 仍在枚举 HCA / pin memory，尚未完成首次 `register_buffer`（`addLocalSegment` + HTTP PUT）。GET 先到 → 404。
- **源码**：`transfer_metadata.cpp:625-633` 的 `getSegmentDesc` 只做一次 GET，无 "重试+等待"。
- **相关 issue**：#1115（同一 Python 进程多 store 客户端的 race），open 未修，本 bench 无此场景。
- **对本 bench**：瞬态 404 通常被上层重试吃掉，不是主因。

**原因 3：transport 不匹配 / `cluster_id` 错配**
- **机制 a**：两端 `MC_METADATA_CLUSTER_ID` 不同。producer 写 `mooncake/A/ram/...`，consumer 读 `mooncake/B/ram/...` → 404。参 `transfer_metadata.cpp:111-121`。
- **机制 b**：ADXL / Ascend 在某些场景下不写 `ram/` key（issue #1260），peer RDMA 查 → 404。
- **对本 bench**：vigil rendered_config 正确则不应命中。**排查第一步应确认两端 `MC_METADATA_CLUSTER_ID` 相同**。

#### B. Key 写入后丢失 / 过期

**原因 4：跨 job stale `transport_endpoint_`（本 bench 家族 1 真正根因）**
- **机制**：前一轮 bench 的 prefill worker 正常 / 异常退出后，HTTP server 上的 `ram/<host:port>` 条目**没清干净**（见原因 6）。master 里 segment 状态 TTL 到期后被清空。本轮启动后，新 vLLM worker 尝试"恢复"从 discovery 拿到的 endpoint 列表，GET 命中 HTTP server 上已过期的值或直接 404。
- **源码定位**：HTTP server 缺 client-level 感知；**master 没有反向通知 HTTP server 删 key 的代码路径**。
- **上游修复**：**PR #1363 直接针对此场景**，但 open (WIP)，本地不含。#1219 备选路线也不含。
- **对本 bench**：**decode5_mtc16 家族 1 主因**。必须 patch 本地。

**原因 5：客户端正常 unmount，但 HTTP DELETE 失败**
- **机制**：进程正常退出会调 `removeSegmentDesc` / `removeRpcMetaEntry`（`transfer_metadata.cpp:358-378, 875-884`）对 HTTP server 发 DELETE。网络瞬态 / HTTP server 高负载丢了请求，key 残留。
- **上游修复**：**PR #1485 已合入**（`unregisterLocalMemory` 同步 metadata），但**只覆盖 "注销 memory buffer" 一侧，不覆盖 "进程退出完整清理"**。失败路径无重试。
- **对本 bench**：概率事件，贡献给 tail。

**原因 6：客户端 `kill -9` / OOM / 节点宕机**
- **机制**：客户端来不及 `removeSegmentDesc` 就消失。Master Service 有心跳（`master_service.cpp:3644-3736` 的 `ClientMonitorFunc`），TTL 过期后 `PrepareUnmountSegment + CommitUnmountSegment` **清自己内部状态**。**但 master 不会通知 HTTP metadata server**。
- **验证**：`grep removeKey\|removeKeys master_service.cpp` → 0 匹配。`grep enable_metadata_cleanup_on_timeout` → 0 匹配。
- **上游修复**：**PR #1363 正是填此缺口**，本地不含。
- **对本 bench**：vLLM worker 被 SLURM preempt / SIGKILL 即命中。`prefill_9_mtc_144` 记录到 3 次 slurm preemption，**此条一定被命中过**。

**原因 7：disk eviction 导致 descriptor 失效**
- **机制**：磁盘层淘汰某 object 的 disk replica 后，master 侧 `DiskDescriptor` 变 stale，Get 时看到 `metadata not found`（但严格说不是 HTTP server 的 404，属种类 ② 对象元数据）。
- **上游修复**：**PR #1549 已合入本地**。
- **对本 bench**：**已排除**。搜索 `rendered_config.yaml` / bench 脚本 / `mooncake_config.json` 均**无任何 disk tier 参数**（`local_buffer_dir` / `disk_dir` / `MC_STORE_ENABLE_DISK` / `kvcache_disk_path` 全部 0 匹配）。纯 DRAM 部署，不会命中。

**原因 8：客户端本地 segment cache 过期 / 不一致**
- **机制**：客户端 `segment_id_to_desc_map_` 缓存了 peer desc。peer 改变 `buffers` 或 `devices`（重新 register），客户端仍用旧 cache 拿旧 `rkey` 写 → RDMA 返回 remote access error。**不是 HTTP 404**，但症状相似，易混。
- **上游修复**：**PR #1826（`withCachedSegment`）未合入本地**（`grep withCachedSegment Mooncake/` → 0 匹配）。即便合入也**只影响 TENT 子模块**；主 TE 走 `syncSegmentCache`（`transfer_metadata.cpp:639-673`）依赖 TTL。
- **对本 bench**：vLLM `MooncakeStoreConnector` 走主 TE，本修复不直接受益。

**原因 9：同节点重启用不同 RDMA 参数**
- **机制**：worker 被 kill 后重启，复用同一 `hostname:rpc_port` 身份，但 `gid / lid / rkey` 全变。HTTP server 上 key 还在（**新**值），但 peer cache 里是**旧**值。
- **归类警告**：**表面像家族 1，实际属家族 2**。症状是 `Failed to modify QP to RTR` / `packet mismatch`，不是 `metadata not found`。
- **对本 bench**：在 `autosweep_20260417_135116_bug_census_cn.md` 观察到的是这条（→ 见 bug 类别 2）。

### 3.3 原因到家族映射

| 原因 | 字面 `metadata not found`？ | 实际归属 | 本地已修？ | 本 bench 命中？ |
| --- | :---: | :---: | :---: | :---: |
| 1. HTTP server 重启 | ✓ | 家族 1 | ✗（无 PR） | **排除**（master 未重启） |
| 2. 启动 race | ✓（瞬态） | 家族 1 | 部分（上层重试） | 贡献 tail |
| 3. cluster_id / transport 错配 | ✓ | 家族 1 | 部署层问题 | 需核查配置 |
| **4. 跨 job stale `transport_endpoint_`** | ✓ | **家族 1 主因** | ✗（#1363 / #1219 未合） | **强命中** |
| 5. 正常 unmount 但 DELETE 失败 | ✓ | 家族 1 | 部分（#1485 ✓，但不重试） | 贡献 tail |
| **6. `kill -9` / OOM / 宕机** | ✓ | **家族 1 强命中** | ✗（#1363 未合） | **强命中**（prefill_9 有 3 次 SLURM preempt） |
| 7. disk eviction | ✓（种类 ②） | 家族 1 | ✓（#1549 已合） | **排除**（无 disk tier） |
| 8. 客户端 cache 过期 | ✗（错值非 404） | 家族 1 相邻 | ✗（#1826 未合） | 不直接命中 |
| 9. 同节点重启 RDMA 参数变 | ✗（QP 错） | **家族 2** | ✓（#1705 / #1733 / #1803 已合） | 已修，见类别 2 |

### 3.4 概念基础：为什么会有 `ram/` 和 `rpc_meta/` 两条 key

> 读者若不熟 Mooncake HTTP metadata 的层次，先看这节，否则后面每条原因都会看糊。

**HTTP metadata server 是什么**：**`mooncake_master` 进程内的 `coro_http_server` 组件**（非独立进程，非独立容器），由 `--enable_http_metadata_server=true` 开关启用（`master.cpp:138-142, :316-324`）。核心数据是纯内存 `std::unordered_map<string,string>`（`http_metadata_server.h:48` / `cpp:12-18`），**无持久化、无 backup、无心跳、无自动重启**。master 进程挂了，这份 map 就全丢；master 不挂，这份 map 就在。每个 Transfer Engine 进程启动时通过 HTTP 向它写**两条** key。

#### 3.4.1 `mooncake/<cluster_id>/ram/<segment_name>` —— RDMA 数据面

结构（`encodeSegmentDesc`，`transfer_metadata.cpp:191-226`）：

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

**作用**：给 peer 做 **RDMA 数据读写** 用。`devices` + `lid/gid` → 建 QP 握手需要；`buffers[i].addr + length + rkey[]` → RDMA_WRITE/READ 的目标地址 + 权限；`priority_matrix` → peer 选用哪块 NIC 走这个 buffer。

#### 3.4.2 `mooncake/<cluster_id>/rpc_meta/<segment_name>` —— 控制面

结构（`addRpcMetaEntry`，`transfer_metadata.cpp:865-867`）：

```json
{ "ip_or_host_name": "192.168.0.103", "rpc_port": 12345 }
```

**作用**：给 peer 做**握手 / 通知 / 探活的 RPC 定位**用——"这个 `segment_name` 身份对应的 RPC 端点在哪"。

#### 3.4.3 key 前缀

- `mooncake/` 固定前缀
- `<cluster_id>` 由 `MC_METADATA_CLUSTER_ID` 环境变量决定（为空则省略）——不同 cluster_id 的 client 互相查不到对方
- `<segment_name>` 一般是 `hostname:rpc_port`，详见 §3.4.6

#### 3.4.4 五维差异

| 维度 | `rpc_meta/<name>` | `ram/<name>` |
| --- | --- | --- |
| 写入时机 | 启动第一时间（第 ③ 步） | 第 ⑦ 步 PUT 空 desc，⑧ 步起刷新 |
| 更新频率 | 存活期间不变 | 每次 register/unregister buffer 重写 |
| 唯一性约束 | **拒绝重复 PUT**（`http_metadata_server.cpp:64-73`） | 允许覆盖 |
| 访问路径 | 握手前查一次拿 `ip:port` | 通常握手消息直接带，cache 失效才回源 |
| p2p 模式 | 豁免不写（`transfer_metadata.cpp:843-863`） | 依然需要 |

**关键推论**：任何 metadata 清理机制（如 PR #1363 的 `cleanupHttpMetadata()`）必须**同时**清 `mooncake/ram/*` 和 `mooncake/rpc_meta/*`。只清一条会形成"握手通但数据面 key 丢"或反过来的半状态。

#### 3.4.5 segment / device / buffer 是 1 : N : M

**一个 Transfer Engine 进程 = 一个 local segment**，不管注册多少块内存、有多少张网卡。

| 概念 | 个数 | 从哪来 |
| --- | --- | --- |
| **segment** | 进程级唯一（= 1） | `allocateLocalSegmentID()` 启动时创建 |
| **device** | N 张 HCA | `ibv_get_device_list()` + filter（`topology.cpp:150-197`）|
| **buffer** | M 条 | 每次 `registerLocalMemory()` 追加 |

每条 buffer 对每张 device 各有一对 `lkey/rkey`（`rdma_transport.cpp:278-281`）：

```cpp
for (auto &context : context_list_) {
    buffer_desc.lkey.push_back(context->lkey(addr));
    buffer_desc.rkey.push_back(context->rkey(addr));
}
```

所以 `buffer_desc.lkey[]` 的长度 = device 数。注册 memory **不创建新 segment**——只是往 segment 的 `buffers` 数组追加一条。vLLM prefill worker 注册 N 层 KV cache ⇒ **1 个 segment + N 条 buffer**。

#### 3.4.6 `segment_name` 格式在不同模式下

**不是一定要 `host:port`**，值由不同主体决定：

- **一般 RDMA 模式**：`local_server_name_ = local_server_name`（`transfer_engine_impl.cpp:107`），**原样用调用方传入的字符串**，TE 不强制格式。理论上可以是任意 key，但 vLLM / Mooncake Store / SGLang 等上层按约定都传 `host:port`。
- **p2p handshake 模式**：TE 自己 `findAvailableTcpPort()` 拼 `ip:port`（`transfer_engine_impl.cpp:139-143`），**强制** `host:port` 格式。原因：p2p 模式没有 HTTP metadata server，也就没有 `rpc_meta/` key；控制面信息必须**编码在 name 本身**里。`getRpcMetaEntry` 的 p2p 分支直接 `parseHostNameWithPort(server_name)` 解析 ip 和 port（`transfer_metadata.cpp:896-899`）。
- **Ascend 模式**：从 `ip:port:npu_x` 解出再拼 `host:port`。

#### 3.4.7 local segment 创建 7 步时间线

一个 TE 进程 init 过程的多步流水（入口 `transfer_engine_impl.cpp:67`）：

1. **决定 `local_server_name_`**（`:102, :107, :139-143`）
2. **构造 `TransferMetadata`**（`:182`）：根据 `conn_string` 选 handshake / storage plugin
3. **PUT `rpc_meta/<name>`**（`:192` → `addRpcMetaEntry`）—— 此时 segment 还没造，但 RPC 身份已广播到 HTTP server
4. **`MultiTransport::installTransport` + topology discover**（`:186-243`）：枚举 `ibv_get_device_list()`，filter 后保存在 `local_topology_`
5. **`initializeRdmaResources()`**（`rdma_transport.cpp:104`）：对每张 HCA open ibv context，拿 `lid / gid / pd / cq`，填进 `context_list_`
6. **`allocateLocalSegmentID()`**（`rdma_transport.cpp:357-373`）：new 一个 `SegmentDesc`，`desc->name = local_server_name_`，遍历 `context_list_` 填 `desc->devices`，`addLocalSegment` 放进进程内存的 `segment_id_to_desc_map_[LOCAL_SEGMENT_ID]`。**此时还没 PUT 到 HTTP server**，`buffers=[]`
7. **`updateLocalSegmentDesc()`**（`rdma_transport.cpp:123` → `transfer_metadata.cpp:762-773`）：encode desc 成 JSON，调 `storage_plugin_->set()` PUT `ram/<name>` → **HTTP server 上第一次出现 `ram/` key**（buffers 为空）
8. **每次 `registerLocalMemory()`**（`rdma_transport.cpp:174-299` → `transfer_metadata.cpp:795-807`）：对每张 HCA `ibv_reg_mr` 拿 `lkey[]/rkey[]`，`addLocalMemoryBuffer` 追加到 `desc->buffers`，若 `update_metadata=true` 再 PUT 一次 `ram/<name>`

**时间线（什么时刻 HTTP server 上能看到什么）**：

```
t0  init 开始
t1  第 ③ 步后：HTTP server 上有 rpc_meta/<name>
t2  第 ⑦ 步后：HTTP server 上有 ram/<name>，但 buffers=[]
t3  第 ⑧ 步首次 register_buffer 后：ram/<name> 里 buffers=[layer_0]
t4+ 每次 register_buffer 都重写 ram/<name>
```

**stale metadata 本质**：同一 `<name>` 身份在 HTTP server 上残留着已消失进程的 `rpc_meta/` 或 `ram/` 条目，新进程用同一身份启动时要么被 PUT 拒绝（唯一性约束），要么和 peer 的 cache 对不上。

---

## 4. 证据（decode_5_mtc_16 主 run）

### 4.1 benchmark 相邻点对比

| Case | Req/s | Out tok/s | Mean TTFT | P90 TTFT | Mean E2E | P90 E2E | T1 TTFT | T2 TTFT | T3 TTFT | T4 TTFT | T5 TTFT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `decode_4_mtc_14` | 1.44 | 430.53 | 6.88 | 13.31 | 9.62 | 15.58 | 27.18 | **12.05** | 3.54 | 3.77 | 3.61 |
| `decode_5_mtc_16` | 0.96 | 288.63 | 13.73 | 35.03 | 16.48 | 37.26 | 28.04 | **35.81** | 33.29 | 12.53 | 4.62 |
| `decode_6_mtc_18` | 1.42 | 426.24 | 9.81 | 26.93 | 12.49 | 31.53 | 31.07 | **20.80** | 5.65 | 5.65 | 5.80 |

- **Turn 1 正常**，**Turn 2 / Turn 3 异常**，Turn 5+ 回到稳定区间 → 不是泛化扩容问题
- 源：`bench_results/kimi_pd_p1d1to17_mooncake_nsys_autosweep/decode_5_mtc_16/attempt_1/hook_post_serve_0_vllm-bench.log:29, :58, :87, :116, :174`

### 4.2 报错时间窗 vs 吞吐时间窗

- 首次失败：`16:46:02`；最后失败：`16:48:50`
- 失败窗口内完成速率：`68 / 169s = 0.402 req/s`
- 失败窗口结束后：`252 / 161s = 1.565 req/s`
- **3.9× 阶跃**，时间上完全对齐，不是巧合

关键日志位置：
- `prefill-0-gb200-rack1-09.log:648304, :648395, :648397`（最后仍在失败的一段）
- `prefill-0-gb200-rack1-09.log:1413, :1415, :1416`（启动后最早一批代表性日志）
- `prefill-0-gb200-rack1-09.log:1918, :2033`（第一批 batch 级失败）

### 4.3 69 个失败 `req_id` 的 turn-band 分布

> Benchmark 未开 `--save-detailed`，无法精确映射 `(conversation_id, turn_id, request_id)`，改用 "完成计数 ÷ 32 conversations" 切 turn-band：

| Turn-band | 失败请求数 |
| --- | ---: |
| 1 | **32** |
| 2 | **32** |
| 3 | 5 |

**最强结果**：每个 conversation 的 turn-1 + turn-2 全部 publish 失败。Turn 3 只截到最早 5 个就开始自愈。

### 4.4 5 个 decode 目标节点均匀分布

| Turn-band | rack1-10 | rack1-11 | rack1-12 | rack1-13 | rack1-14 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 7 | 7 | 6 | 6 | 6 |
| 2 | 7 | 6 | 7 | 6 | 6 |
| 3 | 1 | 1 | 1 | 1 | 1 |

→ 排除 "某台 decode 节点坏掉" 假设。router.log 显示 5 个 decode 节点调度完全均匀（320 个请求全部完成，每个 decode 节点收到 64 个）。

### 4.5 prefill failed-batch 指标对比

| Case | 出现 failed-metric 的秒数 | 首次失败 | 最后失败 | 最大 failed batches | 最大 failed keys |
| --- | ---: | --- | --- | ---: | ---: |
| `decode_4_mtc_14` | 0 | none | none | 0 | 0 |
| `decode_5_mtc_16` | **135** | `16:46:05` | `16:48:51` | **20** | **1761** |
| `decode_6_mtc_18` | 0 | none | none | 0 | 0 |

`decode_5` 代表性指标行：`prefill-0-gb200-rack1-09.log:38395, :52903, :648397`。

### 4.6 External prefix cache hit-rate 里程碑

| Case | 首次 ≥10% | 首次 ≥20% | 首次 ≥50% | 首次 ≥70% | 最终 |
| --- | --- | --- | --- | --- | ---: |
| `decode_4_mtc_14` | 16:32:29 (13.3%) | 16:32:53 (27.8%) | 16:33:05 (51.2%) | 16:33:34 (70.0%) | **79.3%** |
| `decode_5_mtc_16` | 16:49:57 (13.4%) | 16:50:02 (20.0%) | 16:50:44 (50.1%) | **从未达到** | **61.7%** |
| `decode_6_mtc_18` | 17:03:41 (16.0%) | 17:03:44 (24.6%) | 17:04:26 (52.4%) | 17:05:04 (70.1%) | **79.3%** |

关键行号：
- `decode_4` 达到 79.3%：`decode_4_mtc_14/attempt_1/prefill-0/prefill-0-gb200-rack1-09.log:5280`
- `decode_6` 达到 79.3%：`decode_6_mtc_18/attempt_1/prefill-0/prefill-0-gb200-rack1-01.log:5112`
- `decode_5` 初期仍为 0.0%：`decode_5_mtc_16/attempt_1/prefill-0/prefill-0-gb200-rack1-09.log:20662`
- `decode_5` 最终 61.7%：`decode_5_mtc_16/attempt_1/prefill-0/prefill-0-gb200-rack1-09.log:651071`

### 4.7 自愈形态

- 早期重灾：`88/113 keys failed`（`:158220`）、`97/108 keys failed`（`:158221`）
- 尾部轻微：`1/6 keys failed`（`:648306`）、`2/7 keys failed`（`:648304`）
- **不是二值开/关**，是长时间不稳定期的渐进自愈

### 4.7.1 失败 endpoint 的 IP 不属于本 job（家族 1 最强证据之一）

> 源自 `decode5_mtc16_mooncake_correlation_analysis_20260415.md §6.7`（仅英文版有该证据表）。这是"stale descriptor"假设**比简单 propagation delay 解释更强**的核心依据。

**本 job 运行节点**（`slurm.out`）：`gb200-rack1-[09-14]`，对应 IP 集合：
```
192.168.0.109  192.168.0.110  192.168.0.111  192.168.0.112  192.168.0.113  192.168.0.114
```

**本 job prefill 侧实际 listen 的 endpoint**（都在 `192.168.0.109`）：
```
192.168.0.109:16961 / :16663 / :16571 / :16725
```

**但触发 404 的 miss endpoint IP 集合**（**完全不在本 job 节点范围内**）：
```
192.168.0.101  192.168.0.103  192.168.0.104  192.168.0.107  192.168.0.117  192.168.0.118
```

**Top miss endpoints**（按次数降序）：

| Count | Endpoint |
| ---: | --- |
| 6412 | `192.168.0.117:16379` |
| 6225 | `192.168.0.104:16234` |
| 6176 | `192.168.0.103:16462` |
| 6089 | `192.168.0.118:15418` |
| 5990 | `192.168.0.118:15530` |
| 5982 | `192.168.0.117:16214` |
| 5958 | `192.168.0.101:15446` |
| 5951 | `192.168.0.101:15426` |
| 5948 | `192.168.0.107:15220` |
| 5900 | `192.168.0.117:15721` |
| 5879 | `192.168.0.101:15580` |
| 5860 | `192.168.0.101:16860` |
| 5822 | `192.168.0.107:15843` |
| 5806 | `192.168.0.107:16944` |
| 5804 | `192.168.0.117:15945` |
| 5747 | `192.168.0.107:16075` |
| 5724 | `192.168.0.118:16719` |
| 5721 | `192.168.0.118:15089` |

**关键结论**：这些 endpoint IP（`101/103/104/107/117/118`）**不在本 job 节点集合**（`109-114`）里——说明 Mooncake 客户端正在查询**上一轮 bench / 别的 job 残留的 segment 身份**。这正是原因 4（跨 job stale `transport_endpoint_`）的直接证据——**不是"本 job 有节点来不及 publish"的 race 问题（那样 miss IP 应该在 `109-114` 内）**。

**这也解释了本 bench 日志计数的量级**（~106994 次 metadata not found）：不是几个 endpoint 偶尔查一次，而是**十几个 stale endpoint 各被反复查几千次**（每次 decode 请求都触发一批查询，每个 endpoint 都要查）。

### 4.7.2 配置锚点（源码 404 语义）

Shared Mooncake config（来自 rendered config）：
- metadata server: `http://192.168.0.101:8080/metadata`
- master server: `192.168.0.101:50051`

两项都在 `192.168.0.101` 上——这台机**不在本 job 节点（109-114）范围内**，是**共享部署**。这正是"跨 job stale metadata"能发生的前提：
- 上一轮 bench 的 prefill worker 向 `192.168.0.101:8080` PUT 了自己的 `ram/<ip>:<port>` 条目
- 上一轮 job 退出（normal / SIGKILL / OOM），master 侧 TTL 过期清 segment 状态，但 HTTP metadata server 上的 key **残留**
- 本轮 job 启动，某个 discovery 路径拿到这批 stale endpoint → GET HTTP server → 404

**因果精准表述**（来自 decode5 英文版 §6.8）：
- **不是** "segment 存在但 transfer engine 打不开"
- **而是** "transfer engine 从共享 metadata store 里根本找不到 segment descriptor"

### 4.8 multi-turn 为何被放大

benchmark 配置：`multi-turn`、`concurrency 16`、`prefix-global-ratio 0.15`、`prefix-conversation-ratio 0.75`、`no_history_accumulation=true`。前几轮 KV publish 失败 → 下一轮拿不到本该命中的 external prefix → 继续失败 → 恶性循环。Turn 5+ publish 路径自愈后恢复。

---

## 5. 源码锚点（grep-verified）

### 5.1 Mooncake 侧

| 作用 | 文件:行 |
| --- | --- |
| HTTP 404 GET handler | [`mooncake-store/src/http_metadata_server.cpp:38-41`](../../../Mooncake/mooncake-store/src/http_metadata_server.cpp#L38-L41) |
| HTTP 404 DELETE handler | [`mooncake-store/src/http_metadata_server.cpp:100-104`](../../../Mooncake/mooncake-store/src/http_metadata_server.cpp#L100-L104) |
| `store_` 内存 map 定义（无持久化） | [`mooncake-store/src/http_metadata_server.cpp:12-18`](../../../Mooncake/mooncake-store/src/http_metadata_server.cpp#L12-L18) |
| `ram/` PUT 唯一性约束 | [`mooncake-store/src/http_metadata_server.cpp:64-73`](../../../Mooncake/mooncake-store/src/http_metadata_server.cpp#L64-L73) |
| `getSegmentDesc` (无重试) | [`mooncake-transfer-engine/src/transfer_metadata.cpp:625-633`](../../../Mooncake/mooncake-transfer-engine/src/transfer_metadata.cpp#L625-L633) |
| `getSegmentDesc` 404→nullptr | [`transfer_metadata.cpp:851-879`](../../../Mooncake/mooncake-transfer-engine/src/transfer_metadata.cpp#L851-L879) |
| `openSegment` 404→`ERR_INVALID_ARGUMENT` | [`transfer_engine_impl.cpp:434-441`](../../../Mooncake/mooncake-transfer-engine/src/transfer_engine_impl.cpp#L434-L441) |
| `TransferSubmitter::submit_batch` | [`transfer_task.cpp:493`](../../../Mooncake/mooncake-store/src/transfer_task.cpp#L493) |
| `submit_batch_get_offload_object` | [`transfer_task.cpp:535`](../../../Mooncake/mooncake-store/src/transfer_task.cpp#L535) |
| `submitTransferEngineOperation` | [`transfer_task.cpp:657`](../../../Mooncake/mooncake-store/src/transfer_task.cpp#L657) |
| `SubmitTransfers` (TRANSFER_FAIL 写出) | [`client_service.cpp:1556, :1948`](../../../Mooncake/mooncake-store/src/client_service.cpp#L1556) |
| finalize 汇总 (quorum=0 → `-800`) | [`client_service.cpp:1639`](../../../Mooncake/mooncake-store/src/client_service.cpp#L1639) + `types.cpp:38` |
| master ClientMonitor (不通知 HTTP server) | [`master_service.cpp:3644-3736`](../../../Mooncake/mooncake-store/src/master_service.cpp#L3644-L3736) |
| local segment 创建入口 | [`transfer_engine_impl.cpp:67, :192, :186-243`](../../../Mooncake/mooncake-transfer-engine/src/transfer_engine_impl.cpp#L67) |
| `allocateLocalSegmentID` | [`rdma_transport.cpp:357-373`](../../../Mooncake/mooncake-transfer-engine/src/transport/rdma_transport/rdma_transport.cpp#L357-L373) |
| `updateLocalSegmentDesc` (首次 PUT `ram/`) | [`transfer_metadata.cpp:762-773`](../../../Mooncake/mooncake-transfer-engine/src/transfer_metadata.cpp#L762-L773) |
| `registerLocalMemory` (PUT 刷新) | [`transfer_metadata.cpp:795-807`](../../../Mooncake/mooncake-transfer-engine/src/transfer_metadata.cpp#L795-L807) |
| `removeSegmentDesc` / `removeRpcMetaEntry` | [`transfer_metadata.cpp:358-378, :875-884`](../../../Mooncake/mooncake-transfer-engine/src/transfer_metadata.cpp#L358-L378) |
| 客户端 cache 同步 (依赖 TTL) | [`transfer_metadata.cpp:639-673`](../../../Mooncake/mooncake-transfer-engine/src/transfer_metadata.cpp#L639-L673) |
| Python metadata server (未启用) | [`Mooncake/mooncake-wheel/mooncake/http_metadata_server.py:80, :100`](../../../Mooncake/mooncake-wheel/mooncake/http_metadata_server.py#L80) |

### 5.2 vLLM 侧

| 作用 | 文件:行 |
| --- | --- |
| `batch_put failed` Python 聚合 | [`vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_worker.py:480`](../../distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_worker.py#L480) |
| `register_buffer` 唯一调用点 | [`mooncake_store_worker.py:864`](../../distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_worker.py#L864) |
| **缺：`__del__` / `unregister_buffer`** | 本文件 0 匹配 |

---

## 6. 上游 PR 状态（家族 1 相关）

### 6.1 本地 HEAD `be75ca0` **已包含**

| PR | 标题 | commit | 对本 bug 作用 |
| --- | --- | --- | --- |
| [#1549](https://github.com/kvcache-ai/Mooncake/pull/1549) | Notify master on disk eviction to fix stale metadata | `a402dc7` | 修原因 7，本 bench 不开 disk，不受益 |
| [#1485](https://github.com/kvcache-ai/Mooncake/pull/1485) | `unregisterLocalMemory` syncs to metadata | `2e601a1` | 部分修原因 5（不重试） |
| [#1624](https://github.com/kvcache-ai/Mooncake/pull/1624) | Remove target segment desc cache when disconnect | `834c416` | **Ascend Direct only**，本 bench 走 RDMA 不受益；但证明 "断连清缓存" 模式在上游受认可 |

### 6.2 本地 HEAD **不包含**（需要关注）

| PR | 标题 | 状态 | 验证 | 对本 bug 意义 |
| --- | --- | --- | --- | --- |
| [#1219](https://github.com/kvcache-ai/Mooncake/pull/1219) | Delete useless meta from http metadata server | open | `grep removeKey mooncake-store/` → 仅 header 匹配 | 拉模型，能补原因 6；O(N²) 嫌疑，路线未定 |
| [#1363](https://github.com/kvcache-ai/Mooncake/pull/1363) | HTTP metadata cleanup on client timeout | **open (WIP)** | `grep enable_metadata_cleanup_on_timeout` → 0 匹配 | **推模型，直接命中原因 4 / 6，problem statement 一字对应本 bench 根因假设** |
| [#1826](https://github.com/kvcache-ai/Mooncake/pull/1826) | TENT `withCachedSegment` async invalidation | 已合入上游 | `grep withCachedSegment Mooncake/` → 0 匹配（子模块路径未同步） | 仅影响 TENT，vLLM 走主 TE 不直接受益；但 TTL 调参思路可借鉴 |

**PR #1363 关键原文引用**（来自 PR body，与本 bench 假设完全对应）：

> When a client **crashes or is forcefully terminated** (e.g., `kill -9`, OOM killed, node failure), the cleanup operations may fail or never execute. The Master Service can detect this through heartbeat timeout (`client_ttl`) and clean up its internal segment resources, but the **HTTP Metadata Server has no heartbeat mechanism** and cannot detect client failures.
>
> This leads to **stale metadata residue** on the HTTP Metadata Server: `mooncake/ram/{segment_name}` / `mooncake/rpc_meta/{segment_name}`. These residual entries can cause issues when: new nodes attempt to connect using stale metadata; the same node restarts and registers with different RDMA parameters; other nodes query and cache outdated peer information.

**#1219 vs #1363 路线对比**：

| 维度 | #1219（poll-based） | #1363（event-driven） |
| --- | --- | --- |
| 触发时机 | 周期性，秒~分钟级延迟 | 客户端超时立即清理 |
| master 心跳依赖 | 需要，但通过 ping 绕一层 | 直接复用已有机制，路径最短 |
| 部署约束 | HTTP server 可独立部署 | HTTP server 必须和 master 同进程 |
| 实现复杂度 | 需要对账逻辑 + O(N²) 嫌疑 | 在现有 `ClientMonitorThreadMain` 里加一个调用 |
| 适用场景 | HA 场景 HTTP / master 分离部署 | 常见共部署 |
| **本 bench 契合度** | 通用但慢 | **我们就是共部署，直接对症** |

**#1363 典型用法**：

```bash
mooncake_master \
    --enable_http_metadata_server=true \
    --enable_metadata_cleanup_on_timeout=true \
    --client_ttl=20 \
    --v=1
```

### 6.3 相关 open issues

- [#1115](https://github.com/kvcache-ai/Mooncake/issues/1115)（同进程多客户端 race）：症状一致根因不同，本 bench 无此场景
- [#1260](https://github.com/kvcache-ai/Mooncake/issues/1260)（Ascend 单机 404）：同样错误串不同 bug

---

## 7. 本地修复状态缺口清单（按优先级）

| # | 缺口 | 对应原因 | 上游路线 | 复杂度 |
| --- | --- | --- | --- | --- |
| 1 | master TTL → 级联 DELETE HTTP server key | 6（主） | PR #1363 WIP | 中 |
| 2 | HTTP server 周期自查 → 删 master 不认识的 key | 6 冗余 + 1 部分 | PR #1219 WIP | 中 |
| 3 | HTTP server 持久化 / 重启恢复 | 1 | **无任何 PR** | 高 |
| 4 | vLLM `MooncakeStoreWorker.__del__` / 显式 `unregister_buffer` | 5 | N/A（vLLM 侧） | 低 |
| 5 | 主 TE 客户端 cache 失效策略 | 8 | #1826 TENT 模式推广 | 中 |

注：同目录 `mooncake_connector.py:771` 和 `mooncake_utils.py:60` 有 `__del__`，但都不清理 store worker 的 buffer。

---

## 8. 行动项

### 8.1 短期（我们自己能做）

1. **给 bench 脚本加 metadata server 启动时间校验** —— 堵原因 1 的诊断盲区。一行 `ps -o etime= -p <pid>` 即可。
2. **vLLM `MooncakeStoreWorker` 加 `__del__` / context manager** —— 退出时显式 `unregister_buffer`。当前文件无此代码（grep 确认）。堵原因 5。
3. **Cherry-pick PR #1363 本地打 patch** —— 启动时加 `--enable_metadata_cleanup_on_timeout=true` + `--client_ttl=20`。堵原因 4 / 6。

### 8.2 中期（等上游）

1. **跟进 #1219 / #1363 路线决策** —— 维护者定稿前不深投入。
2. **把 #1485 的 "DELETE 失败重试" 补上** —— 向上游提 issue。

### 8.3 长期

1. **HTTP metadata server 持久化 / 重启恢复** —— 无上游 PR，值得新 issue。需考虑一致性模型（与 master 对账 / LWW / 崩溃一致性）。
2. **主 TE 客户端 cache 失效策略迁移到 `withCachedSegment` 模式** —— 当 #1826 稳定后向 TE 侧推广。

---

## 9. 诊断 checklist

oncall 在新的 `metadata not found` 事件里按序检查：

1. **`mooncake_master` 进程最近重启过？**（HTTP metadata server 在 master 进程内，不是独立进程）`ps -o etime= -p $(pgrep -f mooncake_master)` 看 uptime。uptime 短于 bench 起始 → 原因 1。**本 bench 已排除此项**（用户确认 master 未重启）。
2. **涉事 `segment_name` 在 master `GetAllSegments` 输出里吗？** 在 master 但不在 HTTP → 原因 6。不在 master 但在 HTTP → 原因 6 另一面。两侧都不在 → 原因 1 / 2。
3. **直接 GET HTTP server 该 key？**
   ```bash
   curl -s "http://<metadata_server>/metadata?key=mooncake/ram/<segment_name>"
   curl -s "http://<metadata_server>/metadata?key=mooncake/rpc_meta/<segment_name>"
   ```
   两条都 404 → 原因 1 / 2 / 6。仅 `ram/` 404 → 原因 5 / 7。仅 `rpc_meta/` 404 → 罕见，需看 `addRpcMetaEntry` 失败历史。
4. **worker 上次退出是 normal 还是 SIGKILL？** `kubectl events` / `dmesg` / `journalctl` 看 OOM / preempt → 原因 6。
5. **`--client_ttl` vs worker 启动间隔？** 默认 10s。重启间隔 < TTL → 原因 9 窗口。
6. **vLLM 调过 `unregister_buffer` 吗？** `grep -rn unregister_buffer vllm/distributed/kv_transfer/kv_connector/v1/mooncake/`。本地确认**没有** → 原因 5 放大。
7. **两端 `MC_METADATA_CLUSTER_ID` 一致？** 错配则永远 404 → 原因 3。

---

## 10. 相关但不属于本类

- **`OBJECT_NOT_FOUND(-704)`**：master service 层 object 元数据查不到。与本类 HTTP 404 segment 元数据查不到是**两条独立链路**。属 bug 类别 3。见 `bug_class_batch_put_transfer_fail_cn.md`。
- **`Failed to modify QP to RTR [110]` / `packet mismatch`**：RDMA 握手 / QP 状态机失败。原因 9（同节点重启 RDMA 参数变）实际归属此家族。见 `bug_class_rdma_handshake_timeout_cn.md`。
- **可观测性缺口**（`-1 vs -800` 统计错误、缺 `req_id` / `tp_rank` / `elapsed`）：见 bug 类别 3 的 §观测方案。
- **`prefill3_decode5_mooncake_deep_dive` 的"重复 put + 30s discard"链**：属 bug 类别 3（processing object re-put）。本类只引用，不包含。

**本次异常不支持的解释**（对 decode_5_mtc_16 而言）：
1. `router` 调度失衡 —— 5 个 decode 节点分配完全均匀（320 请求 ÷ 5 = 64/每节点）
2. 某一台 decode 节点坏掉 —— 失败均匀覆盖全部 5 个 decode 目标
3. decode kernel / TPOT 回归 —— `decode_5` mean TPOT 仍接近相邻点
4. 普通负载扩展退化 —— 相邻 `decode_4` / `decode_6` 无同样 failed-batch 指标

---

## 11. 验证命令

```bash
cd Mooncake && git rev-parse --short HEAD                        # 预期 be75ca0
grep -n "metadata not found" mooncake-store/src/http_metadata_server.cpp  # 预期仅 41, 104
grep -rn "removeKey\|removeKeys" mooncake-store/                  # 预期仅 http_metadata_server.h 里的实现，master_service.cpp 无匹配
grep -rn "enable_metadata_cleanup_on_timeout" mooncake-store/     # 预期 0（证明 #1363 未合入）
grep -rn "withCachedSegment" Mooncake/                            # 预期 0（证明 #1826 未合入）
git log --oneline --all | grep -E "1549|1624|1485|1705|1733|1803|993"  # 7 条全部命中
grep -n "Failed to open segment" mooncake-store/src/transfer_task.cpp  # 预期 493, 535, 657
cd ../vllm && grep -n "register_buffer\|__del__" vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_worker.py
# 预期仅 :864 的 register_buffer，无 __del__ 或 unregister_buffer
```

---

## 12. 来源文档

| 文件 | 贡献章节 |
| --- | --- |
| `metadata_not_found_root_causes_20260419_cn.md` | 第 0-7 节全部 + 验证记录（权重最高，整份并入为 §3 / §5 / §7 / §9 / §11 骨架） |
| `decode5_mtc16_mooncake_correlation_analysis_20260415.md` + `decode5_mtc16_mooncake_correlation_analysis_20260415_cn.md` | 摘要 + §1-10 + 附录 A/B（主证据 run，整份并入为 §1 / §2 / §4） |
| `mooncake_upstream_issues_prs_20260419_cn.md` | 第一部分 §1-§3（概念基础部分合入 §3.4）+ 第二部分 家族 1 章节（§301-423 合入 §6） |
| `prefill3_decode5_mooncake_deep_dive_20260415_cn.md` | 仅 §10 交叉引用（本文非其主轴） |

**未来新证据追加指引**：
- 新 benchmark run 出现 `metadata not found` → 在 §4 添加证据小节 `§4.N: <run_name>`
- 新触发原因（第 10 条+） → 在 §3.2 添加，同步更新 §3.3 映射表
- 上游有新 PR → 更新 §6；若 cherry-pick 到本地，移到 §6.1 并写 commit 哈希
- 本类缺口被填上 → 更新 §7 + §8 相应行，保留历史记录
