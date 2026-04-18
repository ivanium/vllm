# Mooncake batch_put failed 可观测性改进方案

日期：2026-04-15

范围：针对 `batch_put failed` 根因定位能力，系统梳理当前 vLLM / Mooncake 日志体系的现状、缺口、改进方案。

相关文档（4 个独立问题）：
- 问题 1：[`decode5_mtc16_mooncake_correlation_analysis_20260415_cn.md`](decode5_mtc16_mooncake_correlation_analysis_20260415_cn.md) — metadata / segment 可见性
- 问题 2：[`prefill6_mooncake_error_analysis_20260415_cn.md`](prefill6_mooncake_error_analysis_20260415_cn.md) — RDMA 握手与端点状态
- 问题 3：[`prefill3_decode5_mooncake_deep_dive_20260415_cn.md`](prefill3_decode5_mooncake_deep_dive_20260415_cn.md) — processing 对象重复 put / `OBJECT_NOT_FOUND`
- 问题 4：[`batch_put_failed_global_analysis_20260415_cn.md`](batch_put_failed_global_analysis_20260415_cn.md) — 可观测性 / 错误归因

## 0. 4 个问题的错误起点速查表

下一次复现时，**根据首先看到的日志关键词**可以直接定位到对应问题和方案。

| 症状关键词 | 对应问题 | 起点源文件 | 对应观测方案 |
|---|---|---|---|
| `metadata not found` | 问题 1 | `http_metadata_server.cpp:39,87` / `http_metadata_server.py:80,100` | B + C |
| `Failed to open segment` | 问题 1 | `transfer_task.cpp:498,535,639` | B + C + D |
| `Transfer submission failed for key` | 问题 1 | `client_service.cpp:1590,1946` | D |
| `handshake timeout` | 问题 2 | `rdma_endpoint.cpp` | G |
| `packet mismatch` | 问题 2 | `rdma_endpoint.cpp:282` | G |
| `mark it inactive` | 问题 2 | `worker_pool.cpp:245` | G |
| `Failed to complete transfers after 60 seconds` | 问题 2 或 3 | `transfer_task.cpp:341` | G 或 H |
| `Transfer failed for key` | 问题 2 | `client_service.cpp:1681` | G + D |
| `Failed to finalize put for key: OBJECT_NOT_FOUND(-704)` | 问题 3 | `master_service.cpp` / `client_service.cpp` | H + I |
| `revoke` 批量覆盖更早 failed key | 问题 3 | `client_service.cpp:1941` 合并写 | H |
| Python 侧 `batch_put failed codes={-800}` | 聚合症状 | `mooncake_store_worker.py:480` | E |
| decode `100% hit rate` 但性能差 | 问题 4 | vLLM decode connector | J |

## 1. 问题背景

`batch_put failed` 的错误链从 metadata server 一直传到 Python，涉及 4~5 层组件：

```
[metadata server HTTP 404]       ← 根因发生地
        ↓
[TransferEngine openSegment]     ← 静默失败（无日志）
        ↓
[TransferSubmitter submit_batch] ← 有日志但缺 object key 关联
        ↓
[ClientService SubmitTransfers]  ← 有日志但缺 endpoint 关联
        ↓
[Python batch_put_from_multi_buffers] ← 只看到 -800
```

每一层丢一点信息，到 Python 层就只剩 "code=-800" 了。无法知道：
- 哪个 segment endpoint 被查 404
- 404 是时序问题（publish 没发生）还是永久问题（publish 丢了）
- 不同的 C++ 失败原因（metadata / openSegment / submit）全部折叠成 `TRANSFER_FAIL(-800)`

## 2. 当前可观测性盘点

### 2.1 Python 侧（vLLM）

| 位置 | 状态 | 内容 |
|---|---|---|
| `mooncake_store_worker.py:480` batch_put failed 警告 | ✅ 已增强 | req_id, tp_rank, elapsed, 分类计数 (transfer_fail/no_handle/other), failed_samples |
| `mooncake_store_worker.py` batch_get failed 警告 | ✅ 已增强 | 同上，含 elapsed 和可读错误码 |
| `mooncake_store_worker.py` exception handler | ✅ 已增强 | `logger.exception` 含完整 traceback |
| Prometheus metrics (`vllm:mooncake_store_put_*`) | ❌ 集成已 revert | `mooncake_store_metrics.py` 不再被 import，counter 不会被填充 |

### 2.2 Mooncake C++ 侧

| 位置 | 日志级别 | 内容 | 缺口 |
|---|---|---|---|
| `http_metadata_server.cpp:30-44` GET handler | **无日志** | 仅 HTTP 响应 | 看不到哪个 key 被查、hit/miss 结果 |
| `http_metadata_server.cpp:97-104` DELETE handler | **无日志** | 仅 HTTP 响应 | 看不到 DELETE 触发原因 |
| `transfer_metadata.cpp:878` storage_plugin_->get() 失败 | **无日志** | 静默 return nullptr | 无法定位"哪个 segment_name 引起 404" |
| `transfer_engine_impl.cpp:434-441` openSegment | `VLOG(1)` 成功轨迹 | 默认关闭 | — |
| `transfer_task.cpp:498/535/639` Failed to open segment | `LOG(ERROR)` | 只有 endpoint | 缺 object key，无法反查 |
| `client_service.cpp:1590/1946` Transfer submission failed | `LOG(ERROR)` | 有 key + failure_context | 缺 endpoint，无法看到具体是哪台机器 |
| `client_service.cpp:1876` Operation failed | `LOG(ERROR)` | 有 key + TRANSFER_FAIL 字符串 | 有 failure_context 但无 endpoint |
| `client_metric.cpp:127` Client Metrics Report | `LOG(INFO)` 每 5s | 总量级指标 | **纯污染**，对根因无帮助 |
| `transfer_engine_impl.cpp:722` MC_TE_METRIC | `LOG(INFO)` 每 5s | RDMA 总量 | **纯污染** |

### 2.3 Prometheus / Grafana

| 组件 | 状态 |
|---|---|
| Grafana 容器 | 在跑 |
| Prometheus 容器 | **没跑**（`http://localhost:9090` 不可达）|
| `mooncake-overview.json` dashboard | 只覆盖 `master_*` 指标，不覆盖 per-worker |
| vLLM `/metrics` endpoint | 存在，但没人抓 |

## 3. 识别出的缺口

按优先级：

### P0 — 根因定位的"最关键一条"

**问题**：metadata 404 时，C++ `storage_plugin_->get()` 返回 false 后静默丢弃，整条错误链最关键的信息源头**没有任何日志**。

**影响**：即使启用 `GLOG_v=2` 也抓不到哪个 segment_name 触发了 404。所有下游日志只能看到 "Failed to open segment <endpoint>"，但无法确定"这是第一次查还是反复查"、"这个 endpoint 是否曾经 publish 成功过"。

### P1 — metadata server 无 access log

**问题**：`mooncake_master` 进程自带 HTTP metadata server（端口 8080），但对 GET / PUT / DELETE 请求完全不记日志。

**影响**：没法回答"某个 key 在时间轴上是怎么被访问的":
- `publish` 时间（PUT）
- 第一次 `lookup` 时间（GET）
- 结果（hit / 404）

没有这个时间线就没法判断是 startup race 还是 publish 丢失。

### P2 — C++ 两条 ERROR 日志之间的断层

**问题**：
- `transfer_task.cpp:498` 的 `Failed to open segment <endpoint>` 只有 endpoint 没有 key
- `client_service.cpp:1590` 的 `Transfer submission failed for key <key>` 只有 key 没有 endpoint

**影响**：想把 Python `failed_samples=[(key, ...)]` 和具体 endpoint 对上，必须 grep 两次然后靠时间戳猜，容易对错。

### P3 — Prometheus 指标未对接

**问题**：我们已经在 `mooncake_store_metrics.py` 定义了正确的分类 counter（`put_transfer_fail_keys` 等），但 worker 没调用 `record_put_failures()`。

**影响**：运行时能拿到正确的 warning 日志，但拿不到可用于告警 / dashboard 的 Prometheus 时间序列。无法对"某段时间内 TRANSFER_FAIL 激增"设告警。

### P4 — 不同 C++ 错误在 Python 层无法区分

**问题**：C++ 代码 [`client_service.cpp:1592`](Mooncake/mooncake-store/src/client_service.cpp#L1592) 把"metadata not found / Failed to open segment / Transfer submission failed"三种根本不同的失败全部 `SetError(TRANSFER_FAIL)`，统一返回 `-800`。

**影响**：Python 侧无论怎么改都分不清这三类。必须通过 Mooncake C++ 日志才能定位。这是 **Mooncake 设计本身的信息损失**。

## 4. 推荐的改进方案

按 ROI 和代价排序，建议按顺序实施：

### 方案 A — 关闭总量级日志污染（零代价）

修改 vigil 配置 env：
```yaml
env:
  # 关掉两个周期性总量指标日志
  MC_STORE_CLIENT_METRIC: "0"
  MC_TE_METRIC: "0"
  # 删除 MC_STORE_CLIENT_METRIC_INTERVAL 和 MC_TE_METRIC_INTERVAL_SECONDS
```

**收益**：
- 135s 失败窗口少 27+ 条无关 INFO 日志
- `grep "batch_put failed"` 不被稀释
- 对调查零信息损失（这些指标本来就不区分失败原因）

### 方案 B — Mooncake 加 P0 日志（约 3 行代码）

在 `Mooncake/mooncake-transfer-engine/src/transfer_metadata.cpp:878` 附近：

```cpp
if (!storage_plugin_->get(getFullMetadataKey(segment_name), peer_json)) {
    LOG(WARNING) << "getSegmentDesc: metadata lookup failed for segment='"
                 << segment_name << "' (metadata_key='"
                 << getFullMetadataKey(segment_name) << "')";
    return nullptr;
}
```

**收益**：
- 静默失败变成可见信号
- 能直接回答"哪些 segment endpoint 在长期触发 404"
- 下次复现时能立即定位问题 segment 的分布

**代价**：
- Mooncake 是 submodule，需要本地 fork + 重编
- 改动小到可以上游 PR

### 方案 C — metadata server 加 access log（约 5 行代码）

在 `Mooncake/mooncake-store/src/http_metadata_server.cpp` GET handler 末尾：

```cpp
// GET /metadata?key=...
if (it == store_.end()) {
    LOG(WARNING) << "metadata GET MISS: key='" << key << "'";
    resp.set_status_and_content(status_type::not_found, "metadata not found");
    return;
}
// 可选：成功时用 VLOG(1) 记录，避免日志量爆炸
VLOG(1) << "metadata GET HIT: key='" << key << "' size=" << it->second.size();
```

类似地给 DELETE handler 也加一行。

**收益**：
- 直接看到"哪个 key 何时被查、结果如何"
- 配合 PUT 日志（需要另加）能完整复盘 startup race 时序
- 日志在 `scripts/mooncake/mooncake_master.log`，独立于 vLLM worker 日志

**关于 `--bg` 模式的澄清**：
- [`start_mooncake_master.sh:140-172`](scripts/mooncake/start_mooncake_master.sh#L140-L172) 已用 `-logtostderr` + stderr 重定向到 `mooncake_master.log`
- `--bg` 和前台启动**日志内容完全一样**，只是 `--bg` 写文件、前台写 terminal
- `--bg` 绝对不影响可观测性，不需要为此改成前台

### 方案 D — Mooncake C++ 错误链加上 key / endpoint 交叉引用（约 6 行代码）

在 `Mooncake/mooncake-store/src/transfer_task.cpp:498` 加上 key：

```cpp
// 当前：
LOG(ERROR) << "Failed to open segment " << handle.transport_endpoint_;
// 改为：
LOG(ERROR) << "Failed to open segment " << handle.transport_endpoint_
           << " (replica_idx=" << replica_idx << ")";
```

在 `Mooncake/mooncake-store/src/client_service.cpp:1590` 加上 endpoint：

```cpp
// 当前：
LOG(ERROR) << "Transfer submission failed for key " << op.key
           << ": " << failure_context;
// 改为（需要从 op.replicas[replica_idx] 拿 endpoint）：
LOG(ERROR) << "Transfer submission failed for key " << op.key
           << " (endpoint=" << op.replicas[replica_idx].get_memory_descriptor().buffer_descriptor.transport_endpoint_
           << "): " << failure_context;
```

**收益**：
- `grep <key>` 和 `grep <endpoint>` 命中的日志行能直接交叉引用
- 不用再对时间戳拼凑

### 方案 E — 恢复 Prometheus metrics 集成（约 10 行代码）

在 `mooncake_store_worker.py:477-502` 的失败处理路径加回：

```python
# 在 if failed: 块里
if self.xfer_stats is not None and self._stats_lock:
    with self._stats_lock:
        self.xfer_stats.record_put_failures(
            failed_batches=1,
            failed_keys=len(failed),
            transfer_fail_keys=transfer_fail_keys,
            no_available_handle_keys=no_handle_keys,
            other_failed_keys=other_keys,
        )
```

同时在 `__init__` 里恢复 `xfer_stats` / `_stats_lock` 参数和 `MooncakeStoreWorker.get_kv_connector_stats()` 方法。

**收益**：
- Prometheus counter `vllm:mooncake_store_put_transfer_fail_keys` 重新有数据
- 可以给"TRANSFER_FAIL 激增"配告警
- Grafana dashboard 能画时间序列

**代价**：
- 恢复被 revert 的代码
- 需要 Prometheus server 跑起来才有用

### 方案 F — 启动 Prometheus 容器

本地已有 Grafana，但 Prometheus 没跑。需要：
1. 启动 Prometheus 容器（scrape vLLM worker 的 `/metrics`）
2. 扩展 `mooncake-overview.json`，增加 panel：
   - `rate(vllm:mooncake_store_put_failed_batches[1m])`
   - `rate(vllm:mooncake_store_put_transfer_fail_keys[1m])`
   - `vllm:mooncake_store_put_other_failed_keys`

## 4.X 覆盖度审视（对照 4 个问题报告）

方案 A~F 主要覆盖了**问题 1**（metadata/segment 可见性）。对照 4 个问题文档后，发现对**问题 2、3、4 的 decode 侧**覆盖严重不足：

| 方案 | 问题 1 metadata 可见性 | 问题 2 RDMA 握手 | 问题 3 OBJECT_NOT_FOUND 放大 | 问题 4 decode 侧盲区 |
|---|:---:|:---:|:---:|:---:|
| A 关总量 metric | ✓ 减噪 | ✓ 减噪 | ✓ 减噪 | ✓ |
| B `transfer_metadata.cpp:878` 加 log | ✓✓ **主力** | ✗ | ✗ | 部分 |
| C metadata server access log | ✓✓ **主力** | ✗ | 部分 | 部分 |
| D C++ key/endpoint 交叉引用 | ✓ | 部分 | 部分 | ✓ |
| E 恢复 Prometheus 集成 | ✓ | ✓ | ✓ | ✓ |
| F 启动 Prometheus | ✓ | ✓ | ✓ | ✓ |

需要补充方案 G/H/I/J。

### 方案 G — RDMA 握手 & endpoint 状态日志（针对问题 2）

改动 `Mooncake/mooncake-transfer-engine/src/transport/rdma_transport/`：

**G-1：握手失败时打印完整 peer 信息**

`rdma_endpoint.cpp:282` 附近 `packet mismatch` 日志：
```cpp
// 当前（信息不足）：
LOG(ERROR) << "packet mismatch";
// 改为：
LOG(ERROR) << "packet mismatch: endpoint=" << endpoint_
           << " local_nic_path='" << local_nic_path_ << "'"
           << " peer_nic_path='" << peer.peer_nic_path << "'"
           << " local_gid=" << local_gid_ << " peer_gid=" << peer.gid
           << " local_lid=" << local_lid_ << " peer_lid=" << peer.lid
           << " local_qp_num=" << local_qp_num_ << " peer_qp_num=" << peer.qp_num
           << " local_active_mtu=" << local_active_mtu_
           << " configured_mtu=" << configured_mtu_
           << " effective_mtu=" << effective_mtu_;
```

**G-2：reuse / re-establish 分支打印状态机迁移**

`rdma_endpoint.cpp:254`：
```cpp
LOG(INFO) << "reuse existing connection: endpoint=" << endpoint_
          << " previous_state=" << previous_state
          << " new_state=" << new_state
          << " reuse_reason=" << reuse_reason;
```

**G-3：mark inactive 时打印原因和重试数**

`worker_pool.cpp:245`：
```cpp
LOG(ERROR) << "mark it inactive: endpoint=" << endpoint
           << " reason=" << reason
           << " retry_count=" << retry_count
           << " last_error=" << last_error;
```

**G-4：transfer 60s 超时日志加 replica 信息**

`transfer_task.cpp:341` 附近：
```cpp
LOG(ERROR) << "Failed to complete transfers after 60 seconds"
           << " pending_transfer_idx=" << i
           << " replica_idx=" << replica_idx
           << " strategy=" << transfer_strategy
           << " endpoint=" << replica.transport_endpoint_
           << " full_replica=" << replica.to_string();
```

**收益**：
- 区分候选 1（simultaneous-open / stale reuse）vs 候选 2（path mismatch）vs 候选 3（RDMA 参数不匹配）
- 围绕热点主机（如 `192.168.0.104`）构建 endpoint-pair 时间线
- 问题 2 文档 §11 明确列出的三个状态机候选能被证伪或锁定

### 方案 H — Mooncake 状态机级日志（针对问题 3）

改动 `Mooncake/mooncake-store/src/`：

**H-1：`ExistKey()` 区分三种情况**

`master_service.cpp:432`：
```cpp
ErrorCode MasterService::ExistKey(const std::string& key, bool& exists) {
    auto it = metadata_.find(key);
    if (it == metadata_.end()) {
        LOG(INFO) << "ExistKey MISS (metadata absent): key='" << key << "'";
        exists = false;
        return ErrorCode::OK;
    }
    bool has_completed = /* check completed replicas */;
    bool has_processing = /* check processing replicas */;
    if (has_completed) {
        exists = true;
        return ErrorCode::OK;
    }
    if (has_processing) {
        LOG(WARNING) << "ExistKey MISS (only processing replica): key='" << key
                     << "' processing_replicas=" << processing_count
                     << " put_start_age_sec=" << put_start_age_sec;
        exists = false;  // 当前语义
        return ErrorCode::OK;
    }
    // ...
}
```

**H-2：`PutStart()` discard 分支结构化日志**

`master_service.cpp:850` 附近 30s discard 分支：
```cpp
if (now - old_put_start_time >= put_start_discard_timeout_sec) {
    LOG(WARNING) << "PutStart discarding stale processing metadata:"
                 << " key='" << key << "'"
                 << " age_sec=" << (now - old_put_start_time)
                 << " old_client_id=" << old_client_id
                 << " new_client_id=" << new_client_id
                 << " discarded_replica_count=" << old_replicas.size();
    // ... 继续执行 discard ...
}
```

**H-3：`WaitForTransfers()` 进度日志**

`client_service.cpp:1633`：
```cpp
for (size_t i = 0; i < pending_transfers.size(); ++i) {
    LOG(INFO) << "WaitForTransfers: op_key='" << op.key << "'"
              << " future_idx=" << i
              << " future_total=" << pending_transfers.size();
    auto t0 = std::chrono::steady_clock::now();
    auto result = pending_transfers[i].get();
    auto elapsed_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    if (elapsed_s > 1.0) {
        LOG(WARNING) << "WaitForTransfers slow: op_key='" << op.key << "'"
                     << " future_idx=" << i << " elapsed_s=" << elapsed_s;
    }
}
```

**H-4：段合并写可观测性**

`client_service.cpp:1941` `BatchPutWhenPreferSameNode`：
```cpp
LOG(INFO) << "BatchPutWhenPreferSameNode:"
          << " segment_group_id=" << group_id
          << " endpoint=" << transport_endpoint
          << " logical_keys=" << logical_key_count
          << " merged_slices=" << merged_slice_count;
```

**收益**：
- 直接证明或证伪问题 3 的"processing object re-put"放大链
- 能看到"WaitForTransfers 串行等待 60s × N futures"的实际累计时间
- `PutStart discard` 事件能和延迟的 `BatchPutEnd OBJECT_NOT_FOUND` 在时间轴上对齐

### 方案 I — Python `batch_is_exist` 语义对齐日志（针对问题 3）

改动 `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_worker.py:397-455`：

```python
# 在 _handle_request 中调用 batch_is_exist 之后：
exists_results = self.store.batch_is_exist(keys)
missing_keys = [k for k, e in zip(keys, exists_results) if e != 1]
if missing_keys:
    logger.debug(
        "batch_is_exist MISS for req %s (tp_rank=%d): %d/%d keys missing, "
        "samples=%s",
        req_id, self.tp_rank, len(missing_keys), len(keys),
        missing_keys[:3],
    )
```

配合方案 H-1 的 Mooncake `ExistKey` 日志，可以交叉定位：
- **Python 侧**: "我认为 key X 是 missing，重新 put"
- **Mooncake 侧**: "key X 其实在 processing，还没完成"
- **后果**: 旧 processing metadata 被 30s discard 擦除 → 原始 BatchPutEnd 命中 OBJECT_NOT_FOUND

**收益**：问题 3 §10 明确需要"区分 processing vs missing"的证据，这个是 vLLM 侧的对应视角。

### 方案 J — decode 侧真实 load 结果指标（针对问题 4 §11.4）

改动 vLLM decode 侧的 KV connector（`mooncake_store_worker.py` + `mooncake_store_metrics.py`）：

**J-1：新增 Prometheus counter**
```python
# mooncake_store_metrics.py
vllm:mooncake_store_get_succeeded_blocks      # batch_get 返回值 ≥ 0 的 block 数
vllm:mooncake_store_get_failed_blocks         # 返回值 < 0 的 block 数
vllm:mooncake_store_get_fallback_blocks       # 因 load 失败而 fallback 到 recompute 的 block 数
vllm:mooncake_store_hit_but_load_failed_blocks  # hit 了但最终 load 失败的 block 数
```

**J-2：decode 侧 warning**

在 decode 侧 connector 层级，当 hit 但 load 失败触发 fallback 时：
```python
logger.warning(
    "external_kv_load_failed: req_id=%s, "
    "hit_blocks=%d, loaded_blocks=%d, failed_blocks=%d, "
    "fallback_recompute_blocks=%d",
    req_id, hit_count, loaded_count, failed_count, fallback_count,
)
```

**收益**：
- 独立证伪 decode 侧 `External prefix cache hit rate: 100.0%` 的假象
- 能看到"被记 hit 但实际 recompute"的 block 数量，这是问题 4 §11.4 的核心诉求
- 配合 Grafana dashboard 能直接监控真实 KV load 成功率

## 5. 实施路线图

按"首先看到什么症状"的 bootstrap checklist 分支执行。

### 5.1 复现前（必做，几分钟）

做：
- **方案 A**：关 `MC_STORE_CLIENT_METRIC` 和 `MC_TE_METRIC`
- **方案 E**：恢复 Prometheus metrics 集成

然后跑 `repro_batch_put_failed_x10.sh`。根据**首先看到的日志**分支：

### 5.2 分支 1 — 首先看到 `metadata not found` / `Failed to open segment`（→ 问题 1）

做：
- **方案 B**：`transfer_metadata.cpp:878` 加 getSegmentDesc 404 日志
- **方案 C**：`http_metadata_server.cpp` 加 GET miss access log
- **方案 D**：C++ key/endpoint 交叉引用

本地 fork Mooncake → 改代码 → 重编 → 重跑复现。

### 5.3 分支 2 — 首先看到 `handshake timeout` / `packet mismatch` / `mark it inactive`（→ 问题 2）

做：
- **方案 G**（G-1 到 G-4 全部）
- **方案 D**（交叉引用仍然有用）

关注热点主机的 endpoint-pair 时间线，用 G-1 的 MTU/GID/LID/QP 字段区分 3 个候选：simultaneous-open vs path mismatch vs RDMA 参数不匹配。

### 5.4 分支 3 — 首先看到 `OBJECT_NOT_FOUND(-704)` / revoke 链（→ 问题 3）

做：
- **方案 H**（H-1 到 H-4 全部）
- **方案 I**：Python `batch_is_exist` 语义对齐
- **方案 D**：交叉引用

关注：
- `WaitForTransfers slow` 日志揭示累计等待时间
- `PutStart discard` 日志揭示元数据被擦除时间点
- `ExistKey MISS (only processing replica)` 日志揭示语义不匹配

### 5.5 分支 4 — decode 侧 100% hit 但性能差（→ 问题 4 §11.4）

做：
- **方案 J**：decode 侧真实 load 结果指标

同时和分支 1~3 平行推进（因为 decode 侧盲区独立于 prefill 侧故障族）。

### 5.6 长期（跨所有分支）

- **方案 F**：启动 Prometheus + 扩展 Grafana dashboard
- 推方案 B/G/H 的改动给 Mooncake upstream
- 考虑给 Mooncake 加细分错误码：`METADATA_NOT_FOUND = -801`、`HANDSHAKE_FAIL = -802`，让 Python 能区分

## 6. 下次复现的交叉验证流程

假设方案 A+B+C 都做了：

```
Time   | Python vLLM worker              | prefill glog           | mooncake_master.log
-------+---------------------------------+------------------------+----------------------
16:46:02 | batch_put failed: req=X         | -                      | metadata GET MISS: key='<endpoint-Y>'
         | tp_rank=0 elapsed=0.003s        |                        |
         | failed_samples=[(<obj_key_1>,   |                        |
         |                  TRANSFER_FAIL)] |                        |
-------+---------------------------------+------------------------+----------------------
16:46:02 | -                               | getSegmentDesc:        | -
         |                                 | metadata lookup failed |
         |                                 | for segment='<endpoint-Y>' |
-------+---------------------------------+------------------------+----------------------
16:46:02 | -                               | Failed to open segment | -
         |                                 | <endpoint-Y>            |
         |                                 | (replica_idx=0)        |
-------+---------------------------------+------------------------+----------------------
16:46:02 | -                               | Transfer submission    | -
         |                                 | failed for key         |
         |                                 | <obj_key_1>            |
         |                                 | (endpoint=<endpoint-Y>)|
```

这样能直接判断：
- **是同一个 segment 持续 404？** → 某台 prefill 节点 publish 从未发生或过期
- **很多 segment 都 404？** → metadata server 侧有共性问题（TTL / 驱逐 / 清理）
- **在 publish PUT 日志之后还 404？** → metadata 写入和读取之间的一致性问题

## 7. 附：关键文件索引

**当前已有的**：
- `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_worker.py` — Python batch_put 日志增强
- `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_metrics.py` — Prometheus 定义（未被使用）
- `scripts/mooncake/start_mooncake_master.sh` — mooncake_master 启动（`--bg` 日志写 `scripts/mooncake/mooncake_master.log`）
- `scripts/mooncake/repro_batch_put_failed_x10.sh` — P1D5 复现脚本

**需要改动的（按问题分类）**：

问题 1（metadata 可见性）：
- `Mooncake/mooncake-store/src/http_metadata_server.cpp` — 方案 C：GET/DELETE handler 加 access log
- `Mooncake/mooncake-transfer-engine/src/transfer_metadata.cpp` — 方案 B：getSegmentDesc 加 404 日志
- `Mooncake/mooncake-store/src/transfer_task.cpp` — 方案 D：Failed to open segment 加 replica_idx
- `Mooncake/mooncake-store/src/client_service.cpp` — 方案 D：Transfer submission failed 加 endpoint

问题 2（RDMA 握手）：
- `Mooncake/mooncake-transfer-engine/src/transport/rdma_transport/rdma_endpoint.cpp` — 方案 G-1/G-2：packet mismatch / reuse 加完整 peer 字段
- `Mooncake/mooncake-transfer-engine/src/transport/rdma_transport/worker_pool.cpp` — 方案 G-3：mark inactive 加原因和重试数
- `Mooncake/mooncake-store/src/transfer_task.cpp`（已列）— 方案 G-4：60s 超时加 replica 信息

问题 3（processing re-put / OBJECT_NOT_FOUND）：
- `Mooncake/mooncake-store/src/master_service.cpp` — 方案 H-1/H-2：ExistKey 三分支 + PutStart discard
- `Mooncake/mooncake-store/src/client_service.cpp`（已列）— 方案 H-3/H-4：WaitForTransfers 进度 + 段合并
- `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_worker.py` — 方案 I：batch_is_exist 语义对齐 debug 日志

问题 4（decode 侧真实 load 结果）：
- `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_worker.py` — 方案 J-2：load 失败 warning
- `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_metrics.py` — 方案 J-1：新增 4 个 counter

**需要配置的**：
- vigil YAML（关 MC_STORE_CLIENT_METRIC / MC_TE_METRIC 环境变量）
- `examples/mooncake-prometheus-grafana-stack/` — 启动 Prometheus 容器并 scrape vLLM /metrics
