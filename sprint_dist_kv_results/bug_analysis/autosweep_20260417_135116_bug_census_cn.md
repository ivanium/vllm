# P1to17D1 autosweep @ 20260417_135116 的 bug 汇总

日期：2026-04-18
参照 run：`bench_results/kimi_pd_p1to17d1_mooncake_nsys_autosweep/20260417_135116/`

## 问题卡片

| 字段 | 值 |
| --- | --- |
| 问题编号 | 本轮新增观察汇总（跨原问题 1/2/3）|
| 当前状态 | `Confirmed` + `New mechanism` |
| 受影响 sweep 点 | `prefill_5_mtc_80`, `prefill_6_mtc_96`, `prefill_8_mtc_128` |
| 未受影响 sweep 点 | `prefill_3_mtc_48`, `prefill_4_mtc_64`, `prefill_7_mtc_112` |
| preempted 未跑到 bench | `prefill_9_mtc_144`（3 次 slurm preemption）|
| 一行结论 | 问题 2（RDMA 握手 / 端点状态故障）在多个 sweep 点**普遍发生**；本次在 **`Failed to modify QP to RTR: Connection timed out [110]`** 这层抓到了更上游的错误机制；是否升级为 `batch_put failed` 取决于 inactive endpoint 是否后续仍被业务踩到。|
| 相关文档 | [问题 2 prefill6 分析](prefill6_mooncake_error_analysis_20260415_cn.md), [问题 4 可观测性](batch_put_failed_global_analysis_20260415_cn.md), [可观测性方案](mooncake_observability_plan_20260415_cn.md) |

## 1. 观察到什么

与最初的 `prefill_6_mtc_18` bug 报告里的**问题 2 家族**对齐，但两点**新发现**：

1. **更上游的错误机制**：这次在 `rdma_endpoint.cpp:646` 抓到 kernel 级 `errno=110` (ETIMEDOUT)：
   ```
   [Handshake] Failed to modify QP to RTR, check mtu, gid, peer lid, peer qp num: Connection timed out [110]
   ```
   原来 `packet mismatch (rdma_endpoint.cpp:283)` 是**下游症状**，实际 QP 状态机在 `INIT → RTR` 这一步就没拿到 peer 的 `mtu / gid / peer lid / peer qp num` → kernel timeout。

2. **握手失败是否升级取决于是否再被业务路由踩到**：同一 run 里，握手失败（Group A）和 batch_put failed（Group B）节点**分离**。握手不一定导致业务错误；关键看 inactive endpoint 后续是否仍被业务命中。

## 2. 全体受影响 log 汇总

全 sweep 里扫描到 **8 份 prefill log** 有相关 signal，按是否有业务层错误分两组：

### Group A — 启动期握手打嗝，自愈成功，**数据面无损**（5 份 log）

| sweep 点 | prefill 索引 | 节点 | `Failed QP→RTR` | `mark inactive` | `packet mismatch` | `reusing connection` | `batch_put failed` |
| --- | :---: | --- | ---: | ---: | ---: | ---: | ---: |
| `prefill_5_mtc_80/attempt_2` | prefill-3 | rack1-12 | 3 | 3 | 0 | 8 | 0 |
| `prefill_6_mtc_96/attempt_1` | prefill-2 | rack1-10 | 0 | 1 | 1 | 2 | 0 |
| `prefill_6_mtc_96/attempt_1` | prefill-5 | rack1-17 | 7 | 6 | 0 | 16 | 0 |
| `prefill_8_mtc_128/attempt_1` | prefill-5 | rack1-12 | 4 | 2 | 0 | 21 | 0 |
| `prefill_8_mtc_128/attempt_1` | prefill-6 | rack1-16 | **11** | **11** | 0 | **22** | 0 |

特征：
- 握手失败集中在**启动后 ~12 分钟处的几百毫秒窗口**（engine 加载完 `mooncake.register_kv_caches` 触发大量 RDMA 建连）
- 失败后 `Successfully reset the endpoint (triggered by: failed connection setup (active))` → 之后 `reusing connection`
- 没有 60s timeout，没有 TRANSFER_FAIL，没有 batch_put failed
- **11 次 QP→RTR 失败的 rack1-16 都自愈了**，说明 Mooncake 的 reset + reuse 机制工作正常

### Group B — 握手失败 → 升级为 60s timeout → `batch_put failed`（3 份 log）

| sweep 点 | prefill 索引 | 节点 | `batch_put failed` | `TRANSFER_FAIL` | `60s timeout` | `packet mismatch` | `mark inactive` |
| --- | :---: | --- | ---: | ---: | ---: | ---: | ---: |
| `prefill_5_mtc_80/attempt_2` | prefill-4 | rack1-16 | 1 | 7 | 3 | 0 | 0 |
| `prefill_8_mtc_128/attempt_1` | prefill-1 | rack1-04 | 2 | 10 | 4 | 0 | 0 |
| `prefill_8_mtc_128/attempt_1` | prefill-0 | rack1-03 | **3** | **11** | **4** | **2** | **2** |

**关键观察**：Group B 里**只有 `prefill-0/rack1-03` 一份 log 同时有本地握手失败 + 业务失败**。其它 2 份 log（`rack1-16`、`rack1-04`）有 batch_put failed 却**没有本地 packet mismatch / mark inactive / QP→RTR**。

**解释**：这 2 份 log 是"**受害方**"——它们本地的 RDMA 端点正常，但在通过 RDMA 向 **别的节点上已 inactive 的 endpoint** 写入时，等不到 completion → Mooncake 在 `transfer_task.cpp:341` 做 60 秒硬等待 → timeout → TRANSFER_FAIL → Python 日志 `batch_put failed`。

换言之，**"握手失败的节点"和"业务请求失败的节点"是两个角色**。rack1-03 罕见地同时扮演两个角色（自己先握手失败，又被业务再踩到）。

## 3. 全局计数

| 错误类型 | 跨 8 份 log 合计 |
| --- | ---: |
| `Failed to modify QP to RTR: Connection timed out [110]` (新机制) | **25** |
| `mark it inactive` | 23 |
| `Successfully reset the endpoint (failed connection setup)` | 25 |
| `packet mismatch` (下游症状) | 3 |
| `Received same peer QP numbers, reusing connection` (自愈) | 80 |
| `complete transfers after 60 seconds` (硬超时) | 11 |
| `Transfer failed for key ... TRANSFER_FAIL` (C++) | 28 |
| `batch_put failed` (Python Worker 警告) | **6** |
| `OBJECT_NOT_FOUND` (问题 3 signature) | 0 |

**注意**：这次 **没有**观察到问题 1（`metadata not found` / `Failed to retrieve segment descriptor` / `Failed to open segment`）。所以**问题 1 在本 run 没复现**；**问题 2 在多个 sweep 点都出现了**。

## 4. Example 时间线

### Group B 最重灾：`prefill_8_mtc_128/attempt_1/prefill-0/rack1-03`

握手失败：
```
[16:19:28.812006] E rdma_endpoint.cpp:283  Invalid argument: received packet mismatch,
                  local.local_nic_path: 192.168.0.103:13018@rocep139s0,
                  local.peer_nic_path: 192.168.0.112:13960@rocep140s0,
                  peer.local_nic_path: (empty), peer.peer_nic_path: (empty)
[16:19:28.812055] E worker_pool.cpp:245    Worker: Cannot make connection for endpoint:
                  192.168.0.112:13960@rocep140s0, mark it inactive
[16:19:28.812070] E rdma_endpoint.cpp:283  Invalid argument: received packet mismatch,
                  local=192.168.0.103:12768@rocep195s0, peer=192.168.0.112:13048@rocep139s0
[16:19:28.812158] E worker_pool.cpp:245    mark it inactive: 192.168.0.112:13048@rocep139s0
[16:19:29-30]     multiple "Re-establish connection" attempts
```

业务升级（60s 后）：
```
[16:20:27.771656] E client_service.cpp:1611  Transfer failed for key
                  c0285e...@tp_rank:0@pcp0@dcp0@pp_rank:0@3af799...:
                  TRANSFER_FAIL (Transfer 0 failed)
[16:20:29.108819] E client_service.cpp:1611  Transfer failed for key ...@fed86dc2... TRANSFER_FAIL
```

Python `mooncake_store_worker` 聚合（我们新加的日志）：
```
[16:20:29] WARNING [mooncake_store_worker.py:479] batch_put failed: 1/32 keys failed
  for req chatcmpl-___prefill_addr_gb200-rack1-03:18000___decode_addr_gb200-rack1-18:18000_...
  (tp_rank=2, elapsed=61.349s,             ← 61 秒 ≈ 1 × 60s 硬超时
   codes={'TRANSFER_FAIL': 1},
   transfer_fail=1, no_handle=0, other=0,
   batch_bytes=71958528),
   failed_samples=[('c0285e...@fed86dc2...', 'TRANSFER_FAIL')]

[16:21:27] WARNING [mooncake_store_worker.py:479] batch_put failed: 2/32 keys failed
  (tp_rank=1, elapsed=120.012s,             ← 120 秒 = 2 × 60s 超时（印证 WaitForTransfers 串行等待）
   codes={'TRANSFER_FAIL': 2},
   ...)
```

### Group A 代表：`prefill_6_mtc_96/attempt_1/prefill-5/rack1-17`（握手失败 6 次但自愈）

全在 **112 毫秒**内爆发：
```
[15:13:03.766] E rdma_endpoint.cpp:646  [Handshake] Failed to modify QP to RTR ... Connection timed out [110]
[15:13:03.766] E rdma_endpoint.cpp:646  [Handshake] Failed to modify QP to RTR ... Connection timed out [110]
[15:13:03.769] I rdma_endpoint.cpp:433  Successfully reset the endpoint
[15:13:03.770] E worker_pool.cpp:245    mark it inactive: 192.168.0.101:13963@rocep196s0
[15:13:03.770] I rdma_endpoint.cpp:433  Successfully reset the endpoint
[15:13:03.770] E worker_pool.cpp:245    mark it inactive: 192.168.0.101:14108@rocep196s0
[15:13:03.813] E rdma_endpoint.cpp:646  [Handshake] Failed to modify QP to RTR ...
[15:13:03.841] E rdma_endpoint.cpp:646  [Handshake] Failed to modify QP to RTR ...
[15:13:03.844] E worker_pool.cpp:245    mark it inactive: 192.168.0.111:14037@rocep139s0
[15:13:03.868] E worker_pool.cpp:245    mark it inactive: 192.168.0.111:14037@rocep139s0  (4 个 worker 撞同一 peer)
[15:13:03.874] E worker_pool.cpp:245    mark it inactive: 192.168.0.111:14037@rocep139s0
[15:13:03.878] E worker_pool.cpp:245    mark it inactive: 192.168.0.111:14037@rocep139s0
[15:13:03.893 ... 15:13:04.214]  16× "Received same peer QP numbers, reusing connection." ← 自愈
```

## 5. 新日志（方案 B/C/D）在本次起了什么作用

| 日志 | 作用 | 证据 |
| --- | --- | --- |
| Python `batch_put failed` 含 `req_id / tp_rank / elapsed / codes / failed_samples` | 直接看出 `elapsed=60s vs 120s` 对应 WaitForTransfers 的 1x vs 2x 超时 | §4 example 时间线 |
| 失败 key 的完整 name（含 `tp_rank:0@pcp0@dcp0@pp_rank:0@<hash>`） | 能在 Mooncake C++ 日志里 grep 同一 key，打通 Python ↔ C++ 链路 | 3 个 batch_put failed 都能和上游 `Transfer failed for key` 一一对应 |
| `codes={TRANSFER_FAIL: N}` 可读错误码 | 不用查 `-800 = ?` 对照表 | 所有 batch_put failed 行都用人名 |
| `transfer_fail / no_handle / other` 三桶分类 | 本 run 全是 `TRANSFER_FAIL`，瞬间可判"不是 offload 背压" | 所有行 `no_handle=0` |
| 方案 B `Failed to retrieve segment descriptor` 加 `metadata_key` | 未触发（本 run 问题 1 没复现，方案 B 没有生效机会） | `grep -c 'Failed to retrieve segment descriptor'` 全是 0 |
| 方案 C HTTP metadata access log (`metadata GET MISS` 等) | 未用上（理由同上）| — |
| 方案 D `Failed to open segment` 加 `replica_idx / endpoint` | 未触发 | 同上 |

**关键**：方案 A（Python `batch_put failed` 增强）在本 run 发挥了**核心**作用，让我们能准确定位 Group B 的 3 个节点 + 每个请求的失败时序；方案 B/C/D 设计用于问题 1（metadata 可见性），这次没有复现场景，所以没派上用场。

## 6. 关键洞察

### 6.1 握手失败是普遍现象，不是异常

全 sweep 里 **6 / 7 个 sweep 点的 prefill worker 都出现了握手失败**（除了 P=3/4 和 P=7）。握手失败的次数和 **prefill 总数** 正相关：

| sweep 点 | P | `Failed QP→RTR` 总数 | `mark inactive` 总数 |
| --- | :---: | ---: | ---: |
| prefill_3 | 3 | 0 | 0 |
| prefill_4 | 4 | 0 | 0 |
| prefill_5 | 5 | 3 | 3 |
| prefill_6 | 6 | 7 | 7 |
| prefill_7 | 7 | 0 | 0 |
| prefill_8 | 8 | 15 | 15 |

→ **节点越多，启动风暴越猛，启动期握手失败越频繁**。但 P=7 又是 0（可能恰好节点组合较好 / 启动时序错开）。

### 6.2 握手失败是否升级，取决于业务路由

对比 `prefill_8_mtc_128`：

| 节点 | 握手失败 | `batch_put failed` |
| --- | ---: | ---: |
| rack1-03 | 有（pm=2, mi=2）| **3 条** ← 升级 |
| rack1-04 | **0** | **2 条** ← 被别人拖下水 |
| rack1-12 | 有（qptrt=4, mi=2）| 0 ← 自愈 |
| rack1-16 | 有（qptrt=11, mi=11）| 0 ← 自愈 |

握手失败最严重的 **rack1-16（11 次）反而 0 业务失败**；握手正常的 **rack1-04 却被 2 条 batch_put failed 打中**。说明业务失败的分布**不是看本地握手情况，而是看本地的写请求是否命中了别人家坏掉的 endpoint**。

### 6.3 ETIMEDOUT 是一个全新的直接机制

原 prefill6 bug 分析文档把 `packet mismatch` 作为上游原因。本次观察显示：

```
kernel QP INIT→RTR 失败（errno=110 ETIMEDOUT）  ← NEW 直接机制
        ↓
Mooncake worker reset endpoint + mark inactive
        ↓ （后续再踩到这个 endpoint）
等 60 秒没 completion
        ↓
TRANSFER_FAIL
        ↓
batch_put failed（Python 聚合）
```

这改写了原问题 2 分析文档里的"最强候选"：
- 不一定是 simultaneous-open / stale reuse 竞争
- 更可能是**启动惊群时 peer 端 QP 还没就绪，kernel 等不到 peer 的握手包超时**

## 7. 解决方案讨论

基于本次观察，**问题 2 的根因链有两段**，每段需要不同的修法：

### 7.1 上游修法：避免启动惊群

**现状**：P 个 prefill × 4 TP × （多 peer）同时发起 RDMA 握手，kernel 被淹 → ETIMEDOUT。

**可选方案**：

| 方案 | 代价 | 收益 |
| --- | :---: | --- |
| **A. Warmup 探测**：benchmark 在发业务流量前先让所有 prefill worker 互 ping 一次 RDMA endpoint，等所有 endpoint 都 reusing / RTS，再放业务 | 小（改 vigil post_serve） | 简单，能直接避开 benchmark 踩到握手失败窗口 |
| **B. Mooncake 侧加握手重试 with 指数退避**：而不是一次失败就 `mark inactive` | 中（改 Mooncake C++） | 根治启动惊群，但可能延长启动时间 |
| **C. 启动时序错开**：vigil 配置让 prefill 以 1-2 秒错峰启动 | 小（改 vigil serving） | 降低同时握手数 |

**推荐**：A + C 组合。A 兜底 workload 不踩；C 降低触发率。B 是上游修复，最干净但改 Mooncake。

### 7.2 下游修法：inactive endpoint 要真的不被用

**现状**：endpoint 被 `mark it inactive` 后，**后续业务请求仍可能被 router 到这个 endpoint**（证据：rack1-04 / rack1-16 无本地握手失败，却 batch_put failed）。

**可选方案**：

| 方案 | 代价 | 收益 |
| --- | :---: | --- |
| **D. Mooncake 侧`SubmitTransfers` 检查 endpoint inactive 标志**：如果 inactive，直接 fail fast，不要 60 秒傻等 | 中（改 Mooncake client_service.cpp） | **彻底消除 60s timeout 尾延迟**，失败要么秒失败要么成功 |
| **E. Mooncake 侧 inactive endpoint 后台自动重建**：不要等到业务踩到才发现还是坏的 | 大（改 Mooncake worker_pool.cpp） | 真正的自愈 |
| **F. vLLM 侧写路径对 inactive endpoint 主动跳过 replica**：`batch_put_from_multi_buffers` 返回值之外增加 "skip_inactive" | 中（改 vLLM mooncake connector + Mooncake API） | 在 mooncake 修之前的应急 |

**推荐**：D 最实际——让 60 秒超时变成秒级失败，虽然还是失败，但 Python 侧可以立即重试或换 replica，避免被单个坏 endpoint 阻塞整个 batch。E 是最终方案。

### 7.3 观测方面的不足

- 方案 B/C/D 在本 run **完全没有触发**（问题 1 没复现），所以没被验证到。要么启动 master log 看 `metadata PUT/GET` 流水（方案 C 埋的），要么等下一次问题 1 触发
- 如果要主动复现问题 1，可以**故意用旧的 stale mooncake_master**（把 segment metadata 灌了但不服务业务的那种）来构造 metadata lookup 失败场景
- **decode 侧 hit-rate 可观测性**（方案 J）在本 run 也没做，所以看不到 Turn 1 publish 失败如何传导到 Turn 2 的 reuse 丢失——下次跑 decode 扫描时应该先把 J 做了

### 7.4 短期行动清单（ROI 降序）

1. ✅ **本次发的 PR 合并**：`ivanium/Mooncake#1` 是可观测性，让下一次复现能看到更多细节
2. 🔜 **加 warmup（方案 A）**：在 `vigil` `pre_serve` 加一个 Mooncake RDMA 预热脚本（所有 prefill worker 互相 ping 一次 transfer engine），让握手失败都发生在业务流量之前
3. 🔜 **把 CONCURRENCY_PER_PREFILL 调回 ~3-4**：才能真正对标原 bug run 的负载
4. 🔜 **profiling 加回来**：出 nsys trace 才能看握手失败 → 业务升级的精确时序
5. 🔜 **Mooncake 侧方案 D（fail-fast on inactive）**：从"60s 超时"降到"秒级失败"
6. 📋 **文档化本次 ETIMEDOUT 机制**：已在本文 §2

## 7.5 "本次 6 个 batch_put failed 其实都可以不失败"

进一步读 Mooncake `client_service.cpp` 找到**两个都在 write path 的设计 bug**，让本次每个单 replica 超时都直接拉挂整个 op：

### Bug 7.5.a — `SubmitTransfers`: replica 0 submit 失败就 break（line 1531-1568）

```cpp
for (replica_idx = 0 .. op.replicas.size()) {
    auto submit_result = transfer_submitter_->submit(replica, ...);
    if (!submit_result) {
        all_transfers_submitted = false;
        break;                    // ← 不试 replica 1, 2
    }
    op.pending_transfers.emplace_back(submit_result.value());
}
if (!all_transfers_submitted) {
    op.SetError(ErrorCode::TRANSFER_FAIL, failure_context);
}
```

**正常语义**：3 份 replica 至少 1 份成功就算 put 成功（典型 replicated write semantics）。
**实际行为**：replica 0 endpoint 是 inactive → submit 直接 false → 整个 op 挂掉，replica 1/2 从没被试过。

### Bug 7.5.b — `WaitForTransfers`: 任一 future 失败就全挂（line 1572-1614）

```cpp
bool all_transfers_succeeded = true;
for (i = 0 .. op.pending_transfers.size()) {
    auto result = op.pending_transfers[i].get();    // blocking，可能 60s
    if (result != OK) {
        if (all_transfers_succeeded) { first_error = result; failed_transfer_idx = i; }
        all_transfers_succeeded = false;
        // Continue waiting for all transfers to avoid resource leaks
    }
}
if (!all_transfers_succeeded) op.SetError(first_error, "Transfer N failed");
```

**实际行为**：replica 0 的 RDMA 等不到 completion → 60 秒硬 timeout → 尽管 replica 1/2 可能都 OK，op 仍被 SetError(TRANSFER_FAIL)。

### 两个 bug 与本次 6 个 batch_put failed 的对应

每个 Python 侧的 `batch_put failed: 1/32 keys failed ... codes={TRANSFER_FAIL: 1}` 对应的 C++ ERROR 都是 `Transfer failed for key ... (Transfer 0 failed)` — **只有 replica 0 失败，replica 1/2 在 Mooncake 这两个设计 bug 下没有发挥作用**。

| sweep / 节点 | 失败 key 数 | 根因 |
| --- | --- | --- |
| p5/rack1-16 | 1/32 | replica 0 future 60s timeout |
| p8/rack1-04 | 1/32 + 1/32 | 同上 |
| p8/rack1-03 | 1/32 + 1/32 + 2/32 | 同上 |

若修好 7.5.a + 7.5.b：**本次 6 个 batch_put failed 预期全部消失**。

### 建议的 Mooncake 修法

**Fix A（`SubmitTransfers` 单 replica submit 失败不中断）**：

```cpp
size_t num_submitted = 0;
for (replica_idx = 0 .. op.replicas.size()) {
    auto submit_result = transfer_submitter_->submit(replica, ...);
    if (!submit_result) {
        LOG(WARNING) << "Skip replica " << replica_idx
                     << " endpoint=" << replica.transport_endpoint_
                     << " (submit failed)";
        continue;                   // ← 继续试下一个
    }
    op.pending_transfers.emplace_back(submit_result.value());
    num_submitted++;
}
if (num_submitted == 0) {
    op.SetError(ErrorCode::TRANSFER_FAIL, "No replica accepted submit");
}
```

**Fix B（`WaitForTransfers` 用 quorum 语义）**：

```cpp
size_t num_succeeded = 0;
ErrorCode last_error = ErrorCode::OK;
for (i = 0 .. op.pending_transfers.size()) {
    auto result = op.pending_transfers[i].get();
    if (result == ErrorCode::OK) num_succeeded++;
    else last_error = result;
}
if (num_succeeded >= 1) {
    LOG(INFO) << "Key " << op.key << " put succeeded on "
              << num_succeeded << "/" << op.pending_transfers.size() << " replicas";
} else {
    op.SetError(last_error, "All replicas failed");
}
```

两处改动加起来 ~30 行 C++。收益：

| 维度 | Fix 前 | Fix 后 |
| --- | --- | --- |
| 本次 6 个 batch_put failed | 发生 | 不发生 |
| 问题 2 升级链 | inactive endpoint → 60s timeout → batch_put failed | inactive endpoint → 换 replica 直接成功 |
| Tail 延迟（op 级别） | 60s（等最慢 replica） | 60s（同上，要 Fix C 才能消）|
| 失败语义 | 任一 replica 失败 = op 失败 | 所有 replica 失败 = op 失败 |

**Fix C（可选，消 tail 延迟）**：quorum 达到后 early-return + cancel 其它 in-flight 传输。改动较大，测试面更广，可以做第二 PR。

### 与上游 PR 的排序

1. **已提**：`ivanium/Mooncake#1` — 可观测性（B/C/D 日志增强）
2. **下一个 PR 候选**：Fix A + Fix B — write quorum 语义
3. **后续**：Fix C（early return）+ 方案 E（后台重建 inactive endpoint）

## 8. 稳定结论措辞

> `20260417_135116` 这次 P1to17D1 autosweep 里，**问题 2（RDMA 握手 / 端点状态故障）在多个 sweep 点普遍发生**。8 份 prefill log 里 5 份握手失败后自愈（无业务影响），3 份握手失败升级为 60s timeout → `TRANSFER_FAIL` → Python `batch_put failed`（共 6 条警告）。本次首次在 `rdma_endpoint.cpp:646` 抓到底层机制 `Failed to modify QP to RTR: Connection timed out [110]`，改写了原 prefill6 分析里"simultaneous-open / stale reuse 是最强候选"的判断——**更可能是启动惊群时 peer QP 还没就绪，kernel 等不到握手包超时**。握手失败是否升级为业务错误，取决于 inactive endpoint 是否后续仍被业务路由踩到，这说明 Mooncake 的 `mark inactive` 语义并不足以阻止 `SubmitTransfers` 继续尝试坏 endpoint。推荐修复顺序：(1) benchmark warmup 避开启动窗口、(2) Mooncake 客户端对 inactive endpoint fail-fast、(3) 后台自动重建 inactive endpoint。
