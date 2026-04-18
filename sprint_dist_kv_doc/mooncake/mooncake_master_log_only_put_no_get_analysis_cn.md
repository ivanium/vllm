# Mooncake master log 只有 PUT 没有 GET 的原因分析

**日期**: 2026-04-17
**作者**: aoshen（调研记录）
**相关实验**:
- 启动脚本: `scripts/mooncake/benchmark_kimi_pd_p1to17d1_mooncake_nsys_autosweep.sh`
- vLLM 日志: `bench_results/kimi_pd_p1to17d1_mooncake_nsys_autosweep/20260417_135116/`
- Mooncake 日志: `scripts/mooncake/mooncake_master.log`
- Mooncake 源码: `Mooncake/` (本地 submodule)

---

## 1. 现象

实验运行中:
- vLLM 侧 `External prefix cache hit rate` 稳定上升到 **20%–50%+**（从 0.0% 起），说明 external cache（Mooncake）**有命中**。
- 但 `mooncake_master.log` 里肉眼可见的条目几乎全是
  ```
  I20260417 ... http_metadata_server.cpp:78] metadata PUT NEW key='mooncake%2Frpc_meta%2F...'
  I20260417 ... http_metadata_server.cpp:78] metadata PUT UPDATE key='mooncake%2Fram%2F...'
  ```
  统计下来 15346 条 `http_metadata_server.cpp` 行中：
  - `PUT NEW`    472 条
  - `PUT UPDATE` 14868 条
  - `PUT DUP_RPC_META_REJECTED` 2 条
  - `DELETE OK`  2 条
  - **`GET` 0 条**

直观印象：有命中却完全看不到 Get 的日志。

---

## 2. 结论（先说答案）

这是 Mooncake **设计使然**，不是 bug。原因拆成三点：

| # | 观察 | 根因 |
|---|---|---|
| 1 | 日志里的 "PUT" 不是 KV 数据 Put | 它们来自 **HTTP metadata server (port 8080)**，记录的是 transfer engine 的 RPC 节点/段注册元数据（`rpc_meta`、`ram`），与 KV data-plane 完全无关 |
| 2 | HTTP metadata server 的 GET 只打 MISS | 源码显式只在 miss 时 `LOG(WARNING)`，命中路径无任何 LOG，避免每次握手刷屏 |
| 3 | 真正的 KV Get/Put（master RPC, port 50051）**压根不打 per-op log** | 成功路径只累加 metric 计数器；失败/miss 走 `VLOG(1)`，默认 glog verbosity=0 被抑制 |

所以 vLLM 侧的 "external cache hit" 对应的是 master RPC 上的 `ExistKey` + `GetReplicaList`，它们被**静默计数**到每 10s 一条的 `Master Admin Metrics` 汇总行里，而不是 per-op 日志。

---

## 3. 证据链

### 3.1 你看到的 PUT log 来自 HTTP metadata server

**源码位置**: [`Mooncake/mooncake-store/src/http_metadata_server.cpp:50-84`](../../Mooncake/mooncake-store/src/http_metadata_server.cpp)

```cpp
// PUT /metadata?key=<key>
server_->set_http_handler<PUT>(
    "/metadata", [this](coro_http_request& req, coro_http_response& resp) {
        ...
        LOG(INFO) << "metadata PUT " << (was_existing ? "UPDATE" : "NEW")
                  << " key='" << key << "'"
                  << " body_bytes=" << body.size()
                  << " store_size=" << store_size_after;
        ...
    });
```

**关键点**: HTTP metadata server 监听 port 8080（启动脚本中 `MC_HTTP_PORT=8080`），它只管 transfer engine 的**节点元数据**：
- `mooncake%2Frpc_meta%2F192.168.0.116%3A12390` → 某个 worker 的 RPC 地址
- `mooncake%2Fram%2F192.168.0.116%3A12390`       → 该 worker 注册的 RAM 段描述符

**这些 PUT 和 KV cache 数据毫无关系。** 每个 vLLM worker 起来时向 metadata server 注册一次 `rpc_meta`（→ `PUT NEW`），之后每次段增长或再心跳时 `PUT UPDATE`。

### 3.2 HTTP GET 只在 MISS 时打 log

**源码位置**: [`http_metadata_server.cpp:26-47`](../../Mooncake/mooncake-store/src/http_metadata_server.cpp)

```cpp
// GET /metadata?key=<key>
server_->set_http_handler<GET>(
    "/metadata", [this](coro_http_request& req, coro_http_response& resp) {
        ...
        auto it = store_.find(std::string(key));
        if (it == store_.end()) {
            LOG(WARNING) << "metadata GET MISS key='" << key
                         << "' store_size=" << store_.size();
            resp.set_status_and_content(status_type::not_found, ...);
            return;
        }

        // 命中路径：直接返回，完全没有 LOG
        resp.add_header("Content-Type", "application/json");
        resp.set_status_and_content(status_type::ok, it->second);
    });
```

**设计意图**：transfer engine 握手时会频繁 GET 节点信息（每次连接前查对端），打全量 log 会刷屏到不可用。

对比三个方法的日志策略：

| Method | 成功路径 | 失败路径 |
|---|---|---|
| PUT    | `LOG(INFO)` 全量打 | `LOG(WARNING)` 打 duplicate |
| GET    | **静默** | `LOG(WARNING) metadata GET MISS` |
| DELETE | `LOG(INFO)` 全量打 | `LOG(WARNING) metadata DELETE MISS` |

### 3.3 真正的 KV Get/Put 不打 per-op log

**源码位置**: [`Mooncake/mooncake-store/src/master_service.cpp`](../../Mooncake/mooncake-store/src/master_service.cpp)

#### `GetReplicaList`（KV 数据面的 Get，行 706）

```cpp
auto MasterService::GetReplicaList(const std::string& key)
    -> tl::expected<GetReplicaListResponse, ErrorCode> {
    std::shared_lock<std::shared_mutex> shared_lock(snapshot_mutex_);
    MetadataAccessorRO accessor(this, key);

    MasterMetricManager::instance().inc_total_get_nums();  // ← 只累加计数

    if (!accessor.Exists()) {
        VLOG(1) << "key=" << key << ", info=object_not_found";  // ← VLOG(1) 默认被吞
        return tl::make_unexpected(ErrorCode::OBJECT_NOT_FOUND);
    }
    ...
    if (replica_list[0].is_memory_replica()) {
        MasterMetricManager::instance().inc_mem_cache_hit_nums();  // ← 只累加计数
    } else if (replica_list[0].is_disk_replica()) {
        MasterMetricManager::instance().inc_file_cache_hit_nums();
    }
    MasterMetricManager::instance().inc_valid_get_nums();
    metadata.GrantLease(...);
    return GetReplicaListResponse(...);   // 命中：完全没有 LOG
}
```

#### `ExistKey`（vLLM 侧"查 external prefix cache 是否命中"走的路径，行 422）

```cpp
auto MasterService::ExistKey(const std::string& key)
    -> tl::expected<bool, ErrorCode> {
    std::shared_lock<std::shared_mutex> shared_lock(snapshot_mutex_);
    MetadataAccessorRO accessor(this, key);
    if (!accessor.Exists()) {
        VLOG(1) << "key=" << key << ", info=object_not_found";  // ← 默认吞
        return false;
    }
    ...
    return true;  // 命中：无 LOG
}
```

#### PutStart / PutEnd

- 成功路径：同样没有 per-key `LOG(INFO)`，只在错误/冲突时打（如 `master_service.cpp:847` 的 `object_already_exists`、`868` 的 `Illegal client ... to PutEnd key`）。

#### 启动脚本没打开 verbose

[`scripts/mooncake/start_mooncake_master.sh:103-119`](../../scripts/mooncake/start_mooncake_master.sh):

```bash
CMD=(
    mooncake_master
    -rpc_port="$MC_RPC_PORT"
    ...
    -logtostderr   # ← 只有这个
)
```

没有 `-v=1`，也没有 `GLOG_v=1` 环境变量 → glog 默认 verbosity=0 → 所有 `VLOG(1)` 被抑制。

### 3.4 Get 确实在发生 — 从 metric summary 能看到

master 每 10s 在 `rpc_service.cpp:108` 打一条 `Master Admin Metrics` 行，聚合所有计数器。实验运行后期的采样：

```
Get:(Req=9992/0/9992,  Item=7829000/7829000)       ← 10k 次 Get，780 万 item
ExistKey:(Req=843166/0/843166, Item=853344694/...)  ← 84 万次 ExistKey，8.5 亿 item
PutStart:(Req=100409/.../100409, Item=2842263/...)  ← 10 万次 Put 开始，284 万 item
PutEnd:(Req=100409/.../100409, Item=2842260/...)    ← 10 万次 Put 结束
```

- `ExistKey` 数量远大于 `Get`：vLLM 先批量查"这些 block 在不在"，再对命中的子集发 `Get`。这是正常模式。
- vLLM 侧 "External prefix cache hit rate ≈ 20%+" 与 master metric 中 `mem_cache_hit_nums / total_get_nums` 的比例一致。

---

## 4. 如何打开 per-op Get/Put log（如果真的需要）

### 方案 A: 提高 glog verbose 级别

修改 [`scripts/mooncake/start_mooncake_master.sh:104-119`](../../scripts/mooncake/start_mooncake_master.sh) 的 `CMD=(...)`：

```bash
CMD=(
    mooncake_master
    -rpc_port="$MC_RPC_PORT"
    ...
    -logtostderr
    -v=1                    # ← 加这行
)
```

或启动前：
```bash
export GLOG_v=1
bash start_mooncake_master.sh --bg
```

**代价**：
- 每个 Get miss、ExistKey miss、QueryIp miss 都会打一条。
- 你的实验里 `ExistKey Item=8.5 亿`，即使 miss 比例只有几个百分点，日志也会爆到百 MB+ 级别。**不要在全量 benchmark 里开**，只在小规模复现时用。

### 方案 B: 直接读 Prometheus metric（推荐）

master 在 `MC_METRICS_PORT=9003`（本实验中对应 `9103`，见启动 log `metrics_port=9103`）暴露 metric 端点。关键指标：

| Metric | 含义 |
|---|---|
| `master_total_get_nums`        | 累计 Get 请求数 |
| `master_valid_get_nums`        | 成功返回的 Get 数 |
| `master_mem_cache_hit_nums`    | 命中 mem replica 的 Get 数 |
| `master_file_cache_hit_nums`   | 命中 disk replica 的 Get 数 |
| `master_total_put_start_nums`  | Put 开始数 |
| `master_total_put_end_nums`    | Put 结束数 |

```bash
curl -s http://localhost:9103/metrics | grep -E "master_(total|valid|mem_cache|file_cache|total_put)"
```

这是**无日志代价的权威数据源**，也是 master.log 里那条超长 Metrics summary 行的底层来源。

### 方案 C: 解析 master.log 里的 Metrics summary

既有的 `Master Admin Metrics` 每 10s 一条，直接 grep 出来做 delta 就够了：

```bash
grep -oE "Get:\(Req=[^)]+\)" mooncake_master.log | head
grep -oE "ExistKey:\(Req=[^)]+\)" mooncake_master.log | tail
```

---

## 5. 附：三层 Mooncake 日志架构速查

```
┌──────────────────────────────────────────────────────────────┐
│ vLLM worker 进程                                              │
│  ├─ MooncakeStoreConnector  (Python)                          │
│  └─ 调用 ↓                                                    │
├──────────────────────────────────────────────────────────────┤
│ Mooncake Client (C++)                                         │
│  ├─ 连 master RPC 50051  (Get/Put/ExistKey 控制面)           │
│  └─ 连 peer TransferEngine (实际 KV 数据 RDMA/TCP 传输)      │
├──────────────────────────────────────────────────────────────┤
│ Mooncake Master (单进程, 本实验 = rack1-11)                  │
│  ├─ RPC service      port 50051                              │
│  │   └─ master_service.cpp                                    │
│  │       - PutStart/PutEnd/Get/ExistKey: **仅 metric 计数** │
│  │       - 错误/冲突才 LOG                                   │
│  ├─ HTTP metadata    port 8080                               │
│  │   └─ http_metadata_server.cpp                              │
│  │       - PUT/DELETE: LOG(INFO)                             │
│  │       - GET: LOG 仅限 MISS                                │
│  └─ Metrics endpoint port 9103 (Prometheus)                  │
└──────────────────────────────────────────────────────────────┘
```

**关键认知**：
- mooncake_master.log 里看到的 99% "PUT" 行都是 **transfer engine 节点发现**（HTTP metadata server 的 PUT），不是 KV 数据 PUT。
- KV 数据面的 Get/Put/ExistKey 默认**完全不打 per-op log**，监控靠 metrics，不靠 grep log。
- vLLM "External prefix cache hit rate" 非零 ⟺ master metric `mem_cache_hit_nums` 递增，与有没有 GET log **无关**。

---

## 6. 快速复现命令

```bash
# 查 HTTP metadata server 的操作类型分布
grep "http_metadata_server.cpp" scripts/mooncake/mooncake_master.log \
  | awk '{print $6, $7}' | sort | uniq -c

# 查 Get/ExistKey/PutEnd 计数器的演化
grep -oE "Get:\(Req=[^)]+\), ExistKey:\(Req=[^)]+\)" \
  scripts/mooncake/mooncake_master.log | sort -u | tail -10

# 查 vLLM 侧 external cache hit rate
grep "External prefix cache hit rate" \
  bench_results/kimi_pd_p1to17d1_mooncake_nsys_autosweep/20260417_135116/prefill_3_mtc_48/attempt_1/prefill-0/*.log \
  | awk -F'External prefix cache hit rate:' '{print $2}' | awk '{print $1}' | sort -u
```

---

## 7. 相关文件索引

- 源码：
  - [`Mooncake/mooncake-store/src/http_metadata_server.cpp`](../../Mooncake/mooncake-store/src/http_metadata_server.cpp) (行 26, 38, 50, 78, 87, 111)
  - [`Mooncake/mooncake-store/src/master_service.cpp`](../../Mooncake/mooncake-store/src/master_service.cpp) (行 422 `ExistKey`, 706 `GetReplicaList`, 285 `mount_segment`, 847/868 `PutEnd`)
- 启动脚本：[`scripts/mooncake/start_mooncake_master.sh`](../../scripts/mooncake/start_mooncake_master.sh) (行 104–119 CMD)
- Benchmark 脚本：[`scripts/mooncake/benchmark_kimi_pd_p1to17d1_mooncake_nsys_autosweep.sh`](../../scripts/mooncake/benchmark_kimi_pd_p1to17d1_mooncake_nsys_autosweep.sh)
- 相关文档：
  - [`mooncake_code_go_through.md`](mooncake_code_go_through.md) — Mooncake 代码整体走读
  - [`mooncake_store_cpu_offload_full_stack_cn.md`](mooncake_store_cpu_offload_full_stack_cn.md) — 全链路分析
