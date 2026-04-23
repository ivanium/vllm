# Bug 类别 3：Object 层失败（OBJECT_NOT_FOUND 放大链 + batch_put 账本错 + 可观测性缺口）

- **合并日期**：2026-04-22
- **本地 Mooncake HEAD**：`be75ca0`（2026-04-18）
- **合并自**：
  - `prefill3_decode5_mooncake_deep_dive_20260415.md` + `prefill3_decode5_mooncake_deep_dive_20260415_cn.md`（OBJECT_NOT_FOUND 放大链机制 + 08:07-08:11 时间链）
  - `batch_put_failed_global_analysis_20260415.md` + `batch_put_failed_global_analysis_20260415_cn.md`（`-1 vs -800` 账本 bug + batch_put_failed 聚合分析 + 完整错误码表）
  - `mooncake_observability_plan_20260415_cn.md`（观测方案 A-J 速查表）
  - `mooncake_upstream_issues_prs_20260419_cn.md` 家族 3 + 家族 4 章节
- **证据权重**：**Strong hypothesis → Confirmed**（prefill_3 主 run 的 08:07-08:11 时间链逐行可对；`-1 vs -800` 账本 bug 已明确修复）

### 在三层模型中的位置：Layer 3（Object Layer）

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Layer 3:  OBJECT LAYER (业务数据：KV-cache value)        ← 本文档       │
│           "地址都对、RDMA 通了，但这个具体的 KV 记录在 master 里没了"    │
│  错误    → OBJECT_NOT_FOUND(-704) / batch_put_failed 聚合错误           │
│  归属    → Master Service 的 object 元数据表 + vLLM Python 账本         │
├─────────────────────────────────────────────────────────────────────────┤
│  Layer 2:  SEGMENT LAYER (peer 身份地址本)              → 类别 1         │
│  错误    → HTTP 404 "metadata not found"                                │
├─────────────────────────────────────────────────────────────────────────┤
│  Layer 1:  CONNECTION LAYER (RDMA QP 物理握手)          → 类别 2         │
│  错误    → QP to RTR 失败 / packet mismatch / Connection timed out[110] │
└─────────────────────────────────────────────────────────────────────────┘

本类既包含真正属于 Layer 3 的根因（OBJECT_NOT_FOUND 放大链），
也合并了 跨三层都需要的"可观测性 / 错误归因缺口"——
因为后者的存在意义就是为了给前两类定位根因，读者不应分两份文档。
```

---

## 1. 摘要

**最直观的一句话**：**RDMA 连接是通的，peer 的地址本也在，但往这个 key 写（或读）时，Master Service 的 object 表里已经找不到对应记录了**。

本类覆盖**两类密切相关的症状**：
1. **OBJECT_NOT_FOUND(-704) 放大链**：最初的 `TRANSFER_FAIL(-800)`（类 1 或类 2 触发）→ 合并写 + `WaitForTransfers` 串行等待 → 60s×N 累积延迟 → vLLM 用 `ExistKey` 查对象误以为缺失 → 发新 `PutStart` → 30s 后擦掉旧 processing metadata → 原始延迟的 `BatchPutEnd` 最终命中 `OBJECT_NOT_FOUND`
2. **Python 账本 bug + 字段缺失**：`-1` 被误当 `TRANSFER_FAIL`（正确是 `-800`）；旧版 `batch_put_failed` 缺 `req_id / tp_rank / elapsed / readable_code / failed_samples`；decode 侧 `100% hit rate` 看不到真实 load 成功率

**本地 bench 命中情况**：

| Run | log | `batch_put failed` | `TRANSFER_FAIL` 行数 | 最可能问题 |
| --- | --- | ---: | ---: | --- |
| `decode_5_mtc_16` | `prefill-0-gb200-rack1-09.log` | 2205 | 106994 | 类 1（metadata not found） |
| `prefill_6_mtc_18` | 4 份 prefill 日志 | 6 | 18 | 类 2（RDMA 握手） |
| `prefill_3_mtc_12` | `prefill-2-gb200-rack1-13.log` | 2 | 16 | **类 3 主证据 run** |
| `prefill_5_mtc_16` | `prefill-1-gb200-rack1-07.log` | 2 | 12 | 类 2 或 3 较弱信号 |
| `prefill_4_mtc_14` | `prefill-0-gb200-rack1-01.log` | 1 | 4 | 类 2 较弱信号 |
| `p4_d4_mtc_14` | `prefill-0-gb200-rack1-06.log` | 1 | 2 | 类 2 较弱信号 |

**prefill_3 主证据（08:07-08:11，跨 5 分钟，放大链完整展示）**：5 轮 60s × TRANSFER_FAIL 累积 → 第 3 轮和第 5 轮命中 OBJECT_NOT_FOUND → 同线程 revoke 覆盖了 3+5 个更早失败的 key。同时 Mooncake transfer-engine 统计显示吞吐仍 `586.74 MB/s`、延迟分布以 `200-1000us` 为主——**不是网络慢，是状态机/生命周期 bug**。

**修复状态一句话**：
- **家族 3**：上游 **PR #993 已合入本地**（`put_start_discard_timeout_sec=30` + client_id 跟踪），这是本家族**主力修复**。但 `ExistKey` 三态语义（absent / processing / complete）和 `prefer_alloc_in_same_node` 合并写 + `WaitForTransfers` 串行等待的相互作用**在上游没有专门的 issue/PR**，需要我们向上游报告。
- **家族 4**：`-1 vs -800` 账本 bug **已在本地修复**（vLLM 侧）；`batch_put_failed` 字段增强 **已在本地完成**（`req_id / tp_rank / elapsed / failed_samples`）；握手字段增强 **已在本地完成**（`endpoint / NIC path / MTU / GID / LID / QP`）。**但 decode 侧真实 load 指标（方案 J）、`ExistKey` / `PutStart discard` / `WaitForTransfers` 进度日志（方案 H）均未实装**。

---

## 2. 症状链

### 2.1 OBJECT_NOT_FOUND 放大链日志指纹（家族 3）

```
Failed to complete transfers after 60 seconds       (transfer_task.cpp:341)
  -> Transfer failed for key: TRANSFER_FAIL         (client_service.cpp:1611)
  -> Continue waiting for other futures in same op  (client_service.cpp:1630)
  -> Later batches hit more timeouts                (60s × N 累积)
  -> Finalize put for key: OBJECT_NOT_FOUND         (client_service.cpp)
  -> Revoke earlier failed keys                     (同线程 batch 覆盖更早失败)
```

代表性关键词：
- `Failed to complete transfers after 60 seconds`
- `Transfer failed for key`
- `Failed to finalize put for key`
- `OBJECT_NOT_FOUND`
- `revoke`

### 2.2 batch_put_failed Python 聚合日志指纹（家族 4）

**增强后版本**（本地已实装，见 §3.7）：
```
[16:20:29] WARNING [mooncake_store_worker.py:479] batch_put failed: 1/32 keys failed
  for req chatcmpl-___prefill_addr_...___decode_addr_..._<uuid>
  (tp_rank=2, elapsed=61.349s,               ← elapsed ≈ 60n 直接看出 WaitForTransfers 超时
   codes={'TRANSFER_FAIL': 1},               ← 可读错误码名，不只是 -800
   transfer_fail=1, no_handle=0, other=0,    ← 三桶分类
   batch_bytes=71958528),
   failed_samples=[('c0285e...@fed86dc2...', 'TRANSFER_FAIL')]  ← 具体失败 key
```

**旧版缺失字段**（已修）：只有 `len(failed) / len(keys)`、`failed_codes`（原始整数）、`total_bytes`、`keys[0]`——且 `keys[0]` 不保证是失败 key。

### 2.3 最容易误导人的日志形态

| 日志形态 | 为什么误导 |
| --- | --- |
| `batch_put failed` | 聚合症状，不是根因。需要看 C++ 侧才知道真正发生什么 |
| `codes={-800}` | 没有 code-name 映射时意识不到是 `TRANSFER_FAIL` |
| `first_key=...` | 历史上它不一定是第一个失败 key，只是 batch 里的第一个 key |
| `External prefix cache hit rate: 100.0%` | 这只表示被计成 hit；不能证明 external KV load 真的成功了 |

### 2.4 本类不含

- `metadata not found` / `Failed to open segment` → 类 1（Layer 2）
- `handshake timeout` / `packet mismatch` / `mark it inactive` → 类 2（Layer 1）
- `NO_AVAILABLE_HANDLE(-200)`（offload 压力）→ 不属家族 1/2/3，归家族 4 的错误归因范畴

---

## 3. 根因分析

### 3.0 名词澄清：四种错误码区分

**混用四种错误码是诊断偏差的头号原因**。每个错误码属于哪一层、哪类 bug、对应哪种根因，必须先记住：

| 错误码 | 名字 | 含义 | 层级 / 类别 | 源码定义 |
| --- | --- | --- | --- | --- |
| `0` | `OK` | 成功 | — | `types.h:208+` |
| **`-1`** | `INTERNAL_ERROR` | **通用内部错误**（**不是** transfer 失败！） | 跨层 | `types.h:210` |
| `-200` | `NO_AVAILABLE_HANDLE` | offload 压力 / handle 分配失败 | Layer 3 | `types.h:215` |
| **`-704`** | **`OBJECT_NOT_FOUND`** | **master 里找不到 object 元数据** | **Layer 3（本类主角）** | `types.h:227` |
| **`-800`** | **`TRANSFER_FAIL`** | **transfer 操作失败** | **Layer 2 → Layer 3 触发器** | `types.h:241` |
| `-900` | `RPC_FAIL` | RPC 失败 | 跨层 | `types.h:243` |

**完整错误码表**（来自 `Mooncake/mooncake-store/include/types.h:208-274`）：

| Code | 名称 | 含义 |
| ---: | --- | --- |
| 0 | `OK` | 成功 |
| -1 | `INTERNAL_ERROR` | 内部错误 |
| -10 | `BUFFER_OVERFLOW` | 缓冲区不足 |
| -100 | `SHARD_INDEX_OUT_OF_RANGE` | shard 索引越界 |
| -101 | `SEGMENT_NOT_FOUND` | 没有找到可用 segment |
| -102 | `SEGMENT_ALREADY_EXISTS` | segment 已存在 |
| -103 | `CLIENT_NOT_FOUND` | client 不存在 |
| -200 | `NO_AVAILABLE_HANDLE` | handle 分配失败 / offload 压力 |
| -300 | `INVALID_VERSION` | 版本非法 |
| -400 | `INVALID_KEY` | key 非法 |
| -500 | `WRITE_FAIL` | 写失败 |
| -600 | `INVALID_PARAMS` | 参数非法 |
| -601 | `ILLEGAL_CLIENT` | client 非法 |
| -700 | `INVALID_WRITE` | 非法写入 |
| -701 | `INVALID_READ` | 非法读取 |
| -702 | `INVALID_REPLICA` | replica 操作非法 |
| -703 | `REPLICA_IS_NOT_READY` | replica 尚未 ready |
| **-704** | **`OBJECT_NOT_FOUND`** | **object 不存在（本类主角）** |
| -705 | `OBJECT_ALREADY_EXISTS` | object 已存在 |
| -706 | `OBJECT_HAS_LEASE` | object 持有 lease |
| -707 | `LEASE_EXPIRED` | 数据传输前 lease 已过期 |
| -708 | `OBJECT_HAS_REPLICATION_TASK` | object 有进行中的复制任务 |
| -709 | `OBJECT_NO_REPLICATION_TASK` | object 没有进行中的复制任务 |
| -710 | `REPLICA_NOT_FOUND` | replica 不存在 |
| -711 | `REPLICA_ALREADY_EXISTS` | replica 已存在 |
| -712 | `REPLICA_IS_GONE` | replica 曾存在但现已消失 |
| -713 | `REPLICA_NOT_IN_LOCAL_MEMORY` | replica 不在本地内存中 |
| -714 | `OBJECT_REPLICA_BUSY` | replica refcount 非零 |
| **-800** | **`TRANSFER_FAIL`** | **传输操作失败（本类上游触发器）** |
| -900 | `RPC_FAIL` | RPC 失败 |
| -1000~-1004 | `ETCD_*` | etcd 相关错误 |
| -1010 | `UNAVAILABLE_IN_CURRENT_STATUS` | 当前状态下不可用 |

### 3.1 OBJECT_NOT_FOUND 放大链完整机制（7 步）

> 源自 `prefill3_decode5_mooncake_deep_dive_20260415_cn.md §5`，**整块搬运**。

这条链的**核心问题不是**"为什么 transfer 一开始会失败"——那是类 1 或类 2 的工作。本类关心的是：**为什么 finalize 阶段会进一步暴露出一个看起来像独立 master 侧根因的 `OBJECT_NOT_FOUND`**。

**7 步机制**：

1. `BatchPutWhenPreferSameNode()` 按 `transport_endpoint_` 对多个逻辑 key 分组，并以 segment 为粒度提交 transfer。
2. 某个 segment 级 transfer future 超时（前置 bug：类 1 metadata not found 或类 2 握手失败），于是这次操作里记录了 `TRANSFER_FAIL(-800)`。
3. `WaitForTransfers()` 在第一次失败后**不会立刻停止**；它会继续串行等待同一 op 里的其他 future，每个 future 都可能再等 60 秒。
4. 因此，单次 op 的 finalize 延迟可能不是 60 秒，而是 **180、240 甚至 300 秒**。
5. 与此同时，vLLM 发送端会执行 `batch_is_exist(keys)`，而 Mooncake 的 `ExistKey()` 对"元数据还在，但没有任何 completed replica"的对象会**返回 `false`**。
6. 如果新的 `PutStart` 在 30 秒后再次打到这些 key，master 会**丢弃旧的 processing metadata**。
7. 原始 put 那个被**延迟的 `BatchPutEnd`** 最终才执行时，就会面对**已经被擦除的元数据**，从而失败为 `OBJECT_NOT_FOUND(-704)`。

**哪些是事实，哪些是推断**：
- 第 1、3、5、6 步都有源码直接支持（见 §3.2-3.5）
- 第 2、4、7 步与观察到的日志时间线高度吻合（见 §4）
- 目前还缺的最后一块证据，是 master 侧对精确 `PutStart → discard → delayed BatchPutEnd` 顺序的最终生命周期证明——这正是**方案 H**（§11.8）要补的

### 3.2 合并写 + 段级 transfer（`prefer_alloc_in_same_node`）

**源码**：`third_partys/Mooncake/mooncake-store/src/client_service.cpp:1897`

**重要逻辑**：
- key 按 `buffer_descriptor.transport_endpoint_` **分组**
- transfer 按**合并后的 segment group** 提交
- `WaitForTransfers()` 对合并后的 op 工作，然后再把结果写回原始逻辑 op

**放大效应**：
- N 个独立 future → **1 个合并 future**
- 某个合并 future 失败时，**覆盖的逻辑 key 范围扩大**
- 这解释了为什么日志里可能只看到少数几条 `Transfer failed for key`，但**更大一批逻辑 key 会在稍后的 finalize / revoke 阶段一起失败**

### 3.3 WaitForTransfers 串行等待放大 60s × N

**源码**：
- `third_partys/Mooncake/mooncake-store/src/client_service.cpp:1630`
- `third_partys/Mooncake/mooncake-store/src/transfer_task.cpp:317`
- `third_partys/Mooncake/mooncake-store/src/transfer_task.cpp:341`

**含义**：
- `WaitForTransfers()` 会**逐个**调用 `pending_transfers[i].get()`
- 即使前一个 future 已经失败，它**仍会继续等待**后面的 future
- 每个 future 最多可以再等 60 秒

**累积效应**：
- 一个失败 future → **60 秒**
- 三个 pending future → **180 秒**
- 五个 pending future → **300 秒**

**与类 2 的关系**：类 2 讨论的是"外层 `wait_for(60s)` 为什么走满"——机制在 `transfer_task.cpp:317`；本节讨论的是"op 内部 N 个 future **串行**累积到 60s×N"——机制在 `client_service.cpp:1630`。两处都是 60s，但是**两个独立的 bug**：
- 类 2 Bug 3.9.b（`WaitForTransfers` 任一 future 失败就全挂）说的是**失败语义**不对
- 本节说的是**等待策略**不对（串行而非并行 + 不 cancel）

### 3.4 ExistKey 三态语义缝隙

**源码**：`third_partys/Mooncake/mooncake-store/src/master_service.cpp:422`

**三态真实情况**：

| 状态 | 含义 | 当前 ExistKey() 返回 | vLLM 行为 |
| --- | --- | --- | --- |
| **absent** | 元数据完全不存在 | `false` | 上层重新 put（**正常**） |
| **processing** | 元数据存在但只有 processing replica（未 completed） | **`false`** ← | **上层误认为缺失而重新 put（语义缝隙！）** |
| **complete** | 至少有 1 个 completed replica | `true` | put 被拒（正常） |

**关键矛盾**：
- **对 vLLM 而言**：对象看起来是 missing（调 `batch_is_exist` 返回 `false`）
- **对 Mooncake 而言**：对象可能其实仍然存在，只是还没完成（processing 状态）

vLLM 端代码（`mooncake_store_worker.py:397-455`）任何 `exists != 1` 都会被当成缺失，从而允许 re-put。这就触发了第 5 → 6 → 7 步的放大链。

**修法方向**（§8）：`ExistKey()` 要区分三种情况，或 vLLM 要感知 processing 状态。

### 3.5 30s put_start_discard 机制

**源码**：`third_partys/Mooncake/mooncake-store/src/master_service.cpp:836`

**相关配置**（`scripts/mooncake/mooncake_master.log:2`）：
```
put_start_discard_timeout_sec=30
put_start_release_timeout_sec=600
```

**含义**：
- 如果**没有 completed replica**
- 且旧的 `put_start_time` 已经超过 **30 秒**
- 一个新的 `PutStart` 就可以**丢弃旧 processing metadata**，并继续执行

**为什么是 30 秒而不是 600 秒**：`release_timeout=600s` 是 put 正常释放的时间，若新 PutStart 等 600s 才能继续，benchmark 会卡死。`discard_timeout=30s` 是一个 "quick release" 机制——但副作用就是本类的放大链：如果一个 op 被 `WaitForTransfers` 串行等待拖到 30s 以后，新 PutStart 就能擦掉旧的 metadata，原始的延迟 BatchPutEnd 再回来就撞上 OBJECT_NOT_FOUND。

**上游修复**：PR #993 于 2025-11-06 合入，实现 `put_start_discard_timeout_sec=30` + `client_id` 跟踪机制——**但这本身就是本 bug 的机制来源，不是 bug 的修复**。真正的修复应该是让 `ExistKey` 能区分 processing 状态，或让合并写 + WaitForTransfers 不累积到 30s 以上。

### 3.6 网络慢假说证据不足

在 `08:09:25` 附近，transfer-engine 的统计**并不符合**"整个数据面都挂了"：
- 吞吐仍有 `586.74 MB/s`
- 延迟分布仍然以 `200-1000us` 为主，只有少数长尾离群值

→ 这不是"整个网络卡住了"，而是**狭义的状态机/生命周期问题**。

### 3.7 Python `-1 vs -800` 账本 bug（家族 4 最确定的修复）

> 源自 `batch_put_failed_global_analysis_20260415_cn.md §6.1`，**已确认并修复**。

**历史上的错误代码**：
```python
transfer_fail_keys = sum(1 for i in failed if res[i] == -1)
```

**正确应该是**：
```python
transfer_fail_keys = sum(1 for i in failed if res[i] == -800)  # -800 = TRANSFER_FAIL
```

**偏差**：
- `-1` 是 `INTERNAL_ERROR`（被错误统计）
- `-800` 才是 `TRANSFER_FAIL`（真实转移故障，**被漏掉**）

**影响的 metrics**：
- `vllm:mooncake_store_put_transfer_fail_keys` 统计的是错误的类别
- 真正的 `TRANSFER_FAIL(-800)` key 被归到了 `other_failed_keys` 桶

**状态**：本地已修，观测方案 §11.5 的 vLLM 侧 counter 定义（`put_transfer_fail_keys / put_no_available_handle_keys / put_other_failed_keys`）基于**修复后**的映射。

### 3.8 batch_put_failed 字段缺失（已修）

**旧版 warning 能稳定展示**：
- `len(failed) / len(keys)`
- `failed_codes`
- `total_bytes`
- `keys[0]`

**但缺少**：

| 缺失字段 | 为什么重要 |
| --- | --- |
| `req_id` | 把故障绑定到某次 request 生命周期 |
| `tp_rank` | 帮助区分按 rank 分裂的问题 |
| `elapsed` | 区分快速失败、60s 超时还是 finalize 阶段失败 |
| failed-key 样例 | 帮助定位具体 block / rank / hash 的失败 |
| 可读的 code name | 避免只看到原始 `codes={-800}` 的歧义 |

**最关键的坑**：`keys[0]` **不保证**一定是失败 key！它只是 batch 里的第一个 key。

**状态**：本地已修（vLLM 侧），§2.2 展示的增强格式是修复后的版本。

### 3.9 decode 侧可观测盲区

在 `decode_5_mtc_16` 中，decode 日志稳定显示：
- `External prefix cache hit rate: 100.0%`

**但不会显示**：
- `batch_get failed`
- `load error`
- `fallback`
- `recompute`

**关键问题**：这个 hit **最后到底有没有真的变成一次成功的 external KV load？**

当前可观测性回答不了这个问题——需要方案 J（§11.10）补的 4 个新 counter：
- `vllm:mooncake_store_get_succeeded_blocks`
- `vllm:mooncake_store_get_failed_blocks`
- `vllm:mooncake_store_get_fallback_blocks`
- `vllm:mooncake_store_hit_but_load_failed_blocks`

### 3.10 C++ 到 Python 的信息损失

**同一个底层故障可能表现成**：
- `metadata not found`（类 1）
- `Failed to open segment`（类 1）
- `Transfer submission failed for key`（类 2 或类 3 触发器）
- `TRANSFER_FAIL`（聚合）
- `batch_put_failed`（Python 聚合）

如果没有**稳定的症状层级**（见三层模型），这些东西很容易被误当成不同 bug。

**信息在 C++ 层已经被折叠**：`client_service.cpp:1592` 把 "metadata not found / Failed to open segment / Transfer submission failed" 三种根本不同的失败全部 `SetError(TRANSFER_FAIL)`，统一返回 `-800`。**Python 侧无论怎么改都分不清这三类**，必须通过 Mooncake C++ 日志才能定位。**这是 Mooncake 设计本身的信息损失**——长期看上游需要加细分错误码（如 `METADATA_NOT_FOUND=-801 / HANDSHAKE_FAIL=-802`）。

### 3.11 症状到问题的映射

替代旧的 "mode A / mode B" 混合叫法，本表作为默认分流参考：

| 症状 / 关键词 | 是否不应直接当成根因 | 最可能指向 |
| --- | --- | --- |
| `batch_put failed` | **是** | 类 1 / 2 / 3 的聚合症状 |
| `TRANSFER_FAIL(-800)` | **是** | 类 1 或类 2，有时也是类 3 的**上游触发器** |
| `OBJECT_NOT_FOUND(-704)` | **是** | 优先看**类 3** |
| `metadata not found` | 否 | 优先看**类 1** |
| `Failed to open segment` | 否 | 优先看**类 1** |
| `handshake timeout` | 否 | 优先看**类 2** |
| `packet mismatch` | 否 | 优先看**类 2** |
| `inactive endpoint` | 否 | 优先看**类 2** |
| decode 侧 `100% hit rate` | **是** | 只表示"被计成 hit"；必须结合本类来解读 |

---

## 4. 证据

### 4.1 prefill_3_mtc_12 主 run 的 08:07-08:11 累积时间链

来源：`bench_results/pd_kimi_nsys_prefill3_light/prefill_3_mtc_12/prefill-2/prefill-2-gb200-rack1-13.log`

| 时间 | 线程 / pid | 事件 | 含义 |
| --- | --- | --- | --- |
| `08:07:25` | `2137498` | timeout + `TRANSFER_FAIL` | **第一次 timeout** |
| `08:07:25` | `2137533` | timeout + `TRANSFER_FAIL` | 第二个线程也开始 timeout |
| `08:08:25` | `2137498` | timeout + `TRANSFER_FAIL` | 第二轮 |
| `08:08:25` | `2137533` | timeout + `TRANSFER_FAIL` | 第二轮 |
| `08:09:25` | `2137498` | timeout + `TRANSFER_FAIL` | 第三轮 |
| `08:09:25` | `2137533` | timeout + `TRANSFER_FAIL` | 第三轮 |
| `08:09:25` | `2137533` | **同 batch 命中 `OBJECT_NOT_FOUND`** | **finalize 已进入二次故障阶段** |
| `08:09:25` | `2137533` | **revoke 之前 3 个失败 key** | 覆盖了该线程里更早 timeout 的 key |
| `08:10:25` | `2137498` | timeout + `TRANSFER_FAIL` | 第四轮 |
| `08:11:25` | `2137498` | timeout + `TRANSFER_FAIL` | 第五轮 |
| `08:11:25` | `2137498` | **同 batch 命中 `OBJECT_NOT_FOUND`** | 第二次 finalize 异常 |
| `08:11:25` | `2137498` | **revoke 之前 5 个失败 key** | 覆盖了 `08:07` 到 `08:11` 的 key |

**关键结论**：这是最强的证据之一，**状态会跨过第一次 60 秒 timeout 持续存在**，并继续影响后续的 finalize / revoke 行为。

**代表性日志位置**：
- `08:07` 的第一次 timeout：`prefill-2-gb200-rack1-13.log:2490`
- `08:08` 的第二次 timeout：`prefill-2-gb200-rack1-13.log:3240`
- `08:09` 第一次出现混合 `TRANSFER_FAIL + OBJECT_NOT_FOUND` 的 batch：`prefill-2-gb200-rack1-13.log:3990`
- `08:10` 的第四次 timeout：`prefill-2-gb200-rack1-13.log:4808`
- `08:11` 第二次出现混合故障 batch：`prefill-2-gb200-rack1-13.log:5579`

### 4.2 跨 6 个 run 的 batch_put_failed 计数

来源：`batch_put_failed_global_analysis_20260415_cn.md §3`，已在 §1 摘要展示。关键观察：

- `decode_5_mtc_16` 的 2205 个 `batch_put failed` 对应 106994 行 `TRANSFER_FAIL` —— 典型类 1 场景（metadata not found）
- `prefill_3_mtc_12` 的 2 个 `batch_put failed` 对应 16 行 `TRANSFER_FAIL` —— 典型类 3 场景（OBJECT_NOT_FOUND 放大）
- 不同类的触发，都在 Python 侧折叠成同一个 `batch_put failed` 聚合

### 4.3 错误归因偏差（`-1 vs -800`）实际案例

修复前的 vLLM counter：
- `vllm:mooncake_store_put_transfer_fail_keys = 0`（实际有大量 `-800`，但代码查的是 `-1`）
- `vllm:mooncake_store_put_other_failed_keys = N`（`-800` 被错误归入这里）

修复后：
- `vllm:mooncake_store_put_transfer_fail_keys = N`
- `vllm:mooncake_store_put_other_failed_keys = ~0`

### 4.4 网络并非根因的对照证据

`prefill_3_mtc_12` 的 `08:09:25` 附近 Mooncake transfer-engine 统计：
- 吞吐：`586.74 MB/s`（健康）
- 延迟分布：以 `200-1000us` 为主，长尾离群值少数
- 结论：**不是数据面挂了**，是状态机/生命周期问题

---

## 5. 源码锚点（grep-verified）

### 5.1 Mooncake 侧

| 作用 | 文件:行 |
| --- | --- |
| **`BatchPutWhenPreferSameNode`（合并写）** | [`mooncake-store/src/client_service.cpp:1897`](../../../Mooncake/mooncake-store/src/client_service.cpp#L1941) |
| **`WaitForTransfers`（串行等待 60s×N）** | [`client_service.cpp:1630`](../../../Mooncake/mooncake-store/src/client_service.cpp#L1633) |
| Transfer submission failed ERROR | [`client_service.cpp:1590, :1946`](../../../Mooncake/mooncake-store/src/client_service.cpp#L1590) |
| `Operation failed` + `-800` 映射 | [`client_service.cpp:1876`](../../../Mooncake/mooncake-store/src/client_service.cpp#L1876) + `types.cpp:38` |
| **`TRANSFER_FAIL` 赋值点（9 处）** | `client_service.cpp:960, 982, 1102, 1623, 1994, 2323, 2396, 2608, 2628` |
| **`ExistKey` 三态实现** | [`master_service.cpp:422`](../../../Mooncake/mooncake-store/src/master_service.cpp#L432) |
| **`PutStart` 30s discard 分支** | [`master_service.cpp:836`](../../../Mooncake/mooncake-store/src/master_service.cpp#L850) |
| 60s `wait_for`（外层） | [`transfer_task.cpp:317`](../../../Mooncake/mooncake-store/src/transfer_task.cpp#L297) |
| 60s timeout（prefill3 引用） | [`transfer_task.cpp:341`](../../../Mooncake/mooncake-store/src/transfer_task.cpp#L341) |
| `Failed to open segment` | `transfer_task.cpp:498, :535, :639` |
| 完整错误码表 | [`mooncake-store/include/types.h:208-274`](../../../Mooncake/mooncake-store/include/types.h#L208-L274) |
| `-800` → `TRANSFER_FAIL` 字符串映射 | `types.cpp:38` |
| `metadata not found` 404 | `http_metadata_server.cpp:39, :87`（类 1 引用） |

### 5.2 vLLM 侧

| 作用 | 文件:行 |
| --- | --- |
| **`batch_put failed` Python 聚合**（增强后） | [`mooncake_store_worker.py:479, :480`](../../distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_worker.py#L479) |
| `batch_is_exist` 调用点 | [`mooncake_store_worker.py:397-455`](../../distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_worker.py#L397-L455) |
| Prometheus counter 定义 | [`mooncake_store_metrics.py`](../../distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_metrics.py) |
| 默认 transfer timeout（不相关的 30s） | `Mooncake/mooncake-integration/transfer_engine/transfer_engine_py.cpp`（`MC_TRANSFER_TIMEOUT=30`） |

### 5.3 配置文件

- `scripts/mooncake/mooncake_master.log:2` — `put_start_discard_timeout_sec=30`、`put_start_release_timeout_sec=600`
- `scripts/mooncake/start_mooncake_master.sh:140-172` — `--bg` 模式的日志输出说明

---

## 6. 上游 PR 状态

### 6.1 家族 3（OBJECT_NOT_FOUND 放大链）

#### 本地 HEAD `be75ca0` **已包含**

| PR | 标题 | 合入日期 | 对本 bug 作用 |
| --- | --- | --- | --- |
| [#993](https://github.com/kvcache-ai/Mooncake/pull/993) | [Store] Add Timeout Mechanism for Put Operations | 2025-11-06 | **本家族主力修复**——实现 `put_start_discard_timeout_sec=30` + `client_id` 跟踪机制 |

#### 相关 open issues（根因已明确，但修复路径各异）

| Issue | 标题 | 状态 | 对本 bug 意义 |
| --- | --- | --- | --- |
| [#974](https://github.com/kvcache-ai/Mooncake/issues/974) | Keys stuck in PROCESSING if PUT initiator crashes | closed 2025-10 | **根因精确匹配**，对应 #993 |
| [#975](https://github.com/kvcache-ai/Mooncake/issues/975) | RFC: 针对 #974 的设计讨论（processing 副本自动清理） | closed | 机制讨论 |
| [#849](https://github.com/kvcache-ai/Mooncake/issues/849) | 后续对同 key 的 `PutStart` 永远失败 | open 2025-09 | 同一假设 |
| [#727](https://github.com/kvcache-ai/Mooncake/issues/727) | `object_already_exists` 然后 `CHECK_EQ(status_, PROCESSING)` | open 2025-08 | 对应 `ExistKey` 语义歧义 |
| [#571](https://github.com/kvcache-ai/Mooncake/issues/571) | `BatchIsExist` 不一致地返回 -704 | closed | 同样指向 `ExistKey` 语义 |

### 6.2 家族 4（可观测性 / 错误归因缺口）

| PR / Issue | 标题 | 状态 | 对本 bug 意义 |
| --- | --- | --- | --- |
| [#1475](https://github.com/kvcache-ai/Mooncake/pull/1475) | [TENT] Add logs for troubleshooting | open 2026-02 | 统一日志工具 + 限流。相邻但**不是**错误码可读化 |
| [#1850](https://github.com/kvcache-ai/Mooncake/issues/1850) | RFC: End-to-End Transfer Tracing for Mooncake | open 2026-04 | 上游最接近的可观测性总体方案 |
| [#1851](https://github.com/kvcache-ai/Mooncake/pull/1851) | 实现 #1850 的 tracing pipeline | open 2026-04 | — |
| [#1529](https://github.com/kvcache-ai/Mooncake/pull/1529) | slog 日志等级通过 env var 控制 | 已合入 2026-02 | 小改进 |
| [#1803](https://github.com/kvcache-ai/Mooncake/pull/1803) | Fix duplicate notify recv WR + 误导性 `Success [0]` 日志 | 已合入 2026-04 | **唯一已合入的"错误可读化"类修复**。家族 2 + 家族 4 都受益 |

### 6.3 上游未报告（建议我们向上游提 issue）

以下 3 项本地实证确认但上游没有专门的 issue / PR：

1. **家族 3**：`ExistKey` 三态语义（absent / processing / complete）——`ExistKey` 对 processing 返回 false 造成 vLLM 误 re-put
2. **家族 3**：`prefer_alloc_in_same_node` 合并写 + `WaitForTransfers` 串行等待的相互作用——60s×N 累积延迟把 op 拖过 30s discard 窗口
3. **家族 4**：错误码可读名映射（C++ 侧）、握手结构化诊断字段（endpoint / NIC / MTU / GID / LID / QP）、metadata server 的 PUT / DELETE 审计日志——#1850/#1851 做的是整体 tracing，不覆盖这些

### 6.4 家族 3 + 4 小结

- **家族 3**：根因和核心机制在上游已明确并已合入主干（#993），但三态语义和合并写串行等待的副作用**没有专门修复**
- **家族 4**：上游只合入了 #1803（错误可读化 1 处）+ #1529（日志等级），大部分诊断字段和错误归因工作**都在本地做**

---

## 7. 本地修复状态

### 7.1 已完成

**vLLM 侧**（本地 HEAD 包含）：
- ✅ `batch_put failed` 日志输出 `req_id`
- ✅ `batch_put failed` 日志输出 `tp_rank`
- ✅ `batch_put failed` 日志输出 `elapsed`
- ✅ `batch_put failed` 日志输出 `first_failed_key`
- ✅ `batch_put failed` 日志输出 `failed_examples`
- ✅ `batch_put failed` 日志输出**可读错误码**（`codes={'TRANSFER_FAIL': N}`）
- ✅ `batch_put failed` 三桶分类（`transfer_fail / no_handle / other`）
- ✅ `batch_get failed` 日志增强（含 elapsed、可读错误码）
- ✅ exception handler 使用 `logger.exception` 含完整 traceback
- ✅ Prometheus counter 定义（`put_failed_batches / put_failed_keys / put_transfer_fail_keys / put_no_available_handle_keys / put_other_failed_keys`）
- ✅ **`-1 vs -800` 账本 bug 修复**（counter 定义基于修复后的映射）

**Mooncake 侧**（本地 HEAD 包含）：
- ✅ **PR #993**：`put_start_discard_timeout_sec=30` + `client_id` 跟踪机制
- ✅ 握手失败日志增强为包含 `endpoint / NIC path / MTU / GID / LID / QP`（家族 2 的方案 G）
- ✅ transfer 失败日志增强为包含 `pending_transfer_index / replica_index / strategy / full replica descriptor`
- ✅ PR #1803（家族 4 唯一已合入的"错误可读化"）

### 7.2 缺失（按优先级）

| # | 缺口 | 对应机制 / 对应方案 | 需本地改 / 向上游提 |
| --- | --- | --- | --- |
| 1 | **Prometheus metrics 集成已 revert**（`mooncake_store_metrics.py` 不再被 import） | 方案 E（§11.5） | vLLM 本地恢复 |
| 2 | **`ExistKey` 三态语义日志**（区分 absent vs processing） | 方案 H-1（§11.8）+ 向上游提 issue | Mooncake 本地 + 上游 |
| 3 | **`PutStart discard` 分支结构化日志**（记录 key / age / client_id） | 方案 H-2（§11.8） | Mooncake 本地 |
| 4 | **`WaitForTransfers` 进度日志**（每个 future idx + elapsed） | 方案 H-3（§11.8） | Mooncake 本地 |
| 5 | **段合并写可观测性**（`BatchPutWhenPreferSameNode` 日志） | 方案 H-4（§11.8） | Mooncake 本地 |
| 6 | **metadata server access log**（GET MISS / HIT / DELETE） | 方案 C（§11.3） | Mooncake 本地 |
| 7 | **`getSegmentDesc` 404 日志**（P0 最关键） | 方案 B（§11.2） | Mooncake 本地 |
| 8 | **C++ key/endpoint 交叉引用**（`Failed to open segment` 加 replica_idx） | 方案 D（§11.4） | Mooncake 本地 |
| 9 | **Python `batch_is_exist` 语义对齐 debug 日志** | 方案 I（§11.9） | vLLM 本地 |
| 10 | **decode 侧真实 load 结果指标（4 个新 counter）** | 方案 J（§11.10） | vLLM 本地 |
| 11 | **Prometheus 容器启动** + Grafana dashboard 扩展 | 方案 F（§11.6） | 运维 |
| 12 | **方案 A**（关 `MC_STORE_CLIENT_METRIC / MC_TE_METRIC` 减噪） | 方案 A（§11.1） | vigil YAML |
| 13 | **`ExistKey` 三态语义修复**（逻辑层面，不只是日志） | §8 长期 | Mooncake 本地 + 上游 |
| 14 | **`WaitForTransfers` 并行等待 + cancel**（消 60s×N 累积） | §8 长期 | Mooncake 本地 + 上游 |
| 15 | **Mooncake 细分错误码**（`METADATA_NOT_FOUND=-801` / `HANDSHAKE_FAIL=-802`） | §8 长期 | 上游 |

---

## 8. 行动项

### 8.1 短期（ROI 降序）

1. ✅ **已做**：`-1 vs -800` 修复 + `batch_put failed` 字段增强 + 握手字段增强（家族 4 核心修复）
2. 🔜 **方案 A**（零代价）：vigil YAML 关 `MC_STORE_CLIENT_METRIC / MC_TE_METRIC`，去掉 135s 失败窗口里 27+ 条噪声 INFO
3. 🔜 **方案 E**（~10 行）：恢复 `mooncake_store_metrics.py` 的 import 和 `record_put_failures()` 调用
4. 🔜 **方案 J**（~30 行）：decode 侧 4 个新 counter + `external_kv_load_failed` warning
5. 🔜 **方案 B**（~3 行 Mooncake）：`transfer_metadata.cpp:878` 加 getSegmentDesc 404 日志（类 1 收益最大，但也帮本类交叉定位）

### 8.2 中期

1. **方案 C**（~5 行）：`http_metadata_server.cpp` GET/DELETE access log
2. **方案 D**（~6 行）：`Failed to open segment` + `Transfer submission failed` 加 key/endpoint 交叉引用
3. **方案 H**（~20 行，核心）：`ExistKey` / `PutStart discard` / `WaitForTransfers` / 段合并的 4 组日志——**这是证伪 / 确认本类放大链的最后一块证据**
4. **方案 I**（~10 行）：Python `batch_is_exist` MISS 日志，配合 H-1 交叉定位
5. **方案 F**：启动 Prometheus 容器 + 扩展 Grafana dashboard
6. **向上游提 issue**：
   - `ExistKey` 三态语义歧义
   - 合并写 + WaitForTransfers 串行等待相互作用
   - 错误码可读名映射

### 8.3 长期

1. **`ExistKey` 三态语义修复**：区分"元数据不存在"和"仍在 processing 未完成"，`batch_is_exist` 上层 API 也要暴露
2. **`WaitForTransfers` 改为并行等待 + quorum early-return + cancel**（类 2 的 Fix C 同款路线）
3. **Mooncake 细分错误码**：加 `METADATA_NOT_FOUND=-801` / `HANDSHAKE_FAIL=-802`，让 Python 能从错误码直接区分三类
4. **端到端 tracing pipeline**（跟进上游 #1850 / #1851）

---

## 9. 诊断 checklist

oncall 看到 `OBJECT_NOT_FOUND` 或 `batch_put failed` 时按序检查：

1. **先看错误码分布**：
   - `codes={-800}` → 类 1 / 2 / 3 任一（聚合症状，继续下钻）
   - `codes={-704}` → **本类（家族 3 放大链）**
   - `codes={-200}` → offload 压力（NO_AVAILABLE_HANDLE）
   - `codes={-1}` → **检查是否 `-1 vs -800` 账本 bug 复发**（本地已修但要定期验证）
2. **看 `elapsed`**：
   - `61s` / `120s` / `180s` ... → 家族 3 的 60s × N 累积
   - `~毫秒` → 不属本类的快速失败
3. **看 `failed_samples` 错误码**：
   - 混合 `TRANSFER_FAIL` + `OBJECT_NOT_FOUND` → **典型本类放大链**，去看对应 log 文件
4. **grep Mooncake log 的 revoke 关键词**：
   - 有 `revoke` 覆盖更早失败 key → 本类 §6.1 时间链特征
5. **查 `PutStart discard` 日志**（方案 H-2 实装后）：
   - 有 `PutStart discarding stale processing metadata` → 直接对准本类的第 6 步机制
6. **查 `ExistKey` MISS 日志**（方案 H-1 实装后）：
   - `ExistKey MISS (only processing replica)` → 直接对准本类第 5 步（语义缝隙）
7. **如果是 decode 侧问题**：
   - `External prefix cache hit rate: 100%` + 性能差 → **必须看方案 J 的 4 个 counter**，否则无法区分"真 load 成功"vs"被计 hit 但 fallback"

---

## 10. 相关但不属于本类

- **`metadata not found` HTTP 404**：HTTP metadata server 段元数据查不到。Layer 2。见 [bug_class_metadata_not_found_cn.md](bug_class_metadata_not_found_cn.md)。
- **`Failed to modify QP to RTR` / `packet mismatch`**：RDMA 握手失败。Layer 1。见 [bug_class_rdma_handshake_timeout_cn.md](bug_class_rdma_handshake_timeout_cn.md)。
- **类 2 的 Bug 3.9.a / 3.9.b**（`client_service.cpp:1531-1568` / `:1572-1614`）：`SubmitTransfers` replica 0 失败就 break + `WaitForTransfers` 任一失败就全挂。这**两个放大 bug 和本类的 `:1941`（合并写）+ `:1633`（串行等待）是独立 4 处代码**，但都影响写路径的失败语义。见类 2 §3.9。
- **`NO_AVAILABLE_HANDLE(-200)`**：offload 压力，不属本家族。检查 `codes={'NO_AVAILABLE_HANDLE': N}` 是否 `>0`——如 `=0` 则排除。

---

## 11. 可观测性方案 A–J 完整清单

> 源自 `mooncake_observability_plan_20260415_cn.md`，**整份并入**。10 个方案按 ROI 和代价排序，覆盖本家族（类 3 + 家族 4 观测）+ 跨类协作（类 1、类 2）。

### 11.1 方案 A — 关闭总量级日志污染（零代价）

修改 vigil 配置 env：
```yaml
env:
  MC_STORE_CLIENT_METRIC: "0"
  MC_TE_METRIC: "0"
  # 删除 MC_STORE_CLIENT_METRIC_INTERVAL 和 MC_TE_METRIC_INTERVAL_SECONDS
```

**收益**：
- 135s 失败窗口少 27+ 条无关 INFO 日志
- `grep "batch_put failed"` 不被稀释
- 对调查零信息损失（这些指标本来就不区分失败原因）

**覆盖**：类 1 / 2 / 3 / 4（全类减噪）

### 11.2 方案 B — Mooncake 加 P0 日志（约 3 行代码）

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

**覆盖**：**类 1 主力**；本类间接受益（交叉定位）

### 11.3 方案 C — metadata server 加 access log（约 5 行代码）

在 `Mooncake/mooncake-store/src/http_metadata_server.cpp` GET handler 末尾：
```cpp
if (it == store_.end()) {
    LOG(WARNING) << "metadata GET MISS: key='" << key << "'";
    resp.set_status_and_content(status_type::not_found, "metadata not found");
    return;
}
VLOG(1) << "metadata GET HIT: key='" << key << "' size=" << it->second.size();
```
类似地给 DELETE handler 也加一行。

**收益**：
- 直接看到"哪个 key 何时被查、结果如何"
- 配合 PUT 日志（需要另加）能完整复盘 startup race 时序
- 日志在 `scripts/mooncake/mooncake_master.log`，独立于 vLLM worker 日志

**覆盖**：**类 1 主力**；本类间接受益

**关于 `--bg` 模式的澄清**：
- `start_mooncake_master.sh:140-172` 已用 `-logtostderr` + stderr 重定向到 `mooncake_master.log`
- `--bg` 和前台启动**日志内容完全一样**，不影响可观测性

### 11.4 方案 D — Mooncake C++ 错误链加上 key / endpoint 交叉引用（约 6 行代码）

在 `transfer_task.cpp:498` 加上 replica_idx：
```cpp
LOG(ERROR) << "Failed to open segment " << handle.transport_endpoint_
           << " (replica_idx=" << replica_idx << ")";
```

在 `client_service.cpp:1590` 加上 endpoint：
```cpp
LOG(ERROR) << "Transfer submission failed for key " << op.key
           << " (endpoint=" << op.replicas[replica_idx].get_memory_descriptor().buffer_descriptor.transport_endpoint_
           << "): " << failure_context;
```

**收益**：
- `grep <key>` 和 `grep <endpoint>` 命中的日志行能直接交叉引用
- 不用再对时间戳拼凑

**覆盖**：类 1 / 2 / 3 通用

### 11.5 方案 E — 恢复 Prometheus metrics 集成（约 10 行代码）

在 `mooncake_store_worker.py:477-502` 的失败处理路径加回：
```python
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

同时在 `__init__` 里恢复 `xfer_stats` / `_stats_lock` 参数和 `get_kv_connector_stats()` 方法。

**收益**：
- Prometheus counter `vllm:mooncake_store_put_transfer_fail_keys` 重新有数据
- 可以给"TRANSFER_FAIL 激增"配告警
- Grafana dashboard 能画时间序列

**覆盖**：类 1 / 2 / 3 通用；**本类需要此方案把已修好的 -1 vs -800 逻辑**暴露给 Prometheus

### 11.6 方案 F — 启动 Prometheus 容器

本地已有 Grafana（在跑），Prometheus 没跑（`http://localhost:9090` 不可达）。需要：
1. 启动 Prometheus 容器（scrape vLLM worker 的 `/metrics`）
2. 扩展 `mooncake-overview.json`，增加 panel：
   - `rate(vllm:mooncake_store_put_failed_batches[1m])`
   - `rate(vllm:mooncake_store_put_transfer_fail_keys[1m])`
   - `vllm:mooncake_store_put_other_failed_keys`

**覆盖**：类 1 / 2 / 3 通用

### 11.7 方案 G — RDMA 握手 & endpoint 状态日志（针对类 2）

> **已在类 2 §11.3（K-1~K-5）详细列出，本节只列摘要引用**。本地 HEAD 已部分实装（握手失败字段）。

- G-1：握手失败时打印完整 peer 信息（`rdma_endpoint.cpp:282` 附近）
- G-2：reuse / re-establish 分支打印状态机迁移（`rdma_endpoint.cpp:254`）
- G-3：mark inactive 时打印原因和重试数（`worker_pool.cpp:245`）
- G-4：transfer 60s 超时日志加 replica 信息（`transfer_task.cpp:341`）

**覆盖**：类 2 主力

### 11.8 方案 H — Mooncake 状态机级日志（针对本类家族 3）

> **本类的核心方案**。这 4 小节能把本类从 `Strong hypothesis` 升级到 `Confirmed`。

**H-1：`ExistKey()` 区分三种情况**

`master_service.cpp:422`：
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

`master_service.cpp:836` 附近 30s discard 分支：
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

`client_service.cpp:1630`：
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

`client_service.cpp:1897` `BatchPutWhenPreferSameNode`：
```cpp
LOG(INFO) << "BatchPutWhenPreferSameNode:"
          << " segment_group_id=" << group_id
          << " endpoint=" << transport_endpoint
          << " logical_keys=" << logical_key_count
          << " merged_slices=" << merged_slice_count;
```

**收益**：
- 直接证明或证伪本类的 "processing object re-put" 放大链
- 能看到 "WaitForTransfers 串行等待 60s × N futures" 的实际累计时间
- `PutStart discard` 事件能和延迟的 `BatchPutEnd OBJECT_NOT_FOUND` 在时间轴上对齐

### 11.9 方案 I — Python `batch_is_exist` 语义对齐日志（针对本类家族 3）

改动 `mooncake_store_worker.py:397-455`：

```python
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

**收益**：配合方案 H-1 的 Mooncake `ExistKey` 日志，可以交叉定位：
- **Python 侧**: "我认为 key X 是 missing，重新 put"
- **Mooncake 侧**: "key X 其实在 processing，还没完成"
- **后果**: 旧 processing metadata 被 30s discard 擦除 → 原始 BatchPutEnd 命中 OBJECT_NOT_FOUND

### 11.10 方案 J — decode 侧真实 load 结果指标（针对家族 4 decode 盲区）

**J-1：新增 4 个 Prometheus counter**

```python
# mooncake_store_metrics.py
vllm:mooncake_store_get_succeeded_blocks      # batch_get 返回值 ≥ 0 的 block 数
vllm:mooncake_store_get_failed_blocks         # 返回值 < 0 的 block 数
vllm:mooncake_store_get_fallback_blocks       # 因 load 失败 fallback 到 recompute 的 block 数
vllm:mooncake_store_hit_but_load_failed_blocks  # hit 但最终 load 失败的 block 数
```

**J-2：decode 侧 warning**

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
- 能看到"被记 hit 但实际 recompute"的 block 数量
- 配合 Grafana dashboard 能直接监控真实 KV load 成功率

### 11.11 方案覆盖度审视

| 方案 | 类 1 metadata | 类 2 RDMA | 类 3 OBJECT_NOT_FOUND 放大 | 家族 4 decode 盲区 |
|---|:---:|:---:|:---:|:---:|
| A 关总量 metric | ✓ 减噪 | ✓ 减噪 | ✓ 减噪 | ✓ |
| B getSegmentDesc 404 log | **✓✓ 主力** | ✗ | ✗ | 部分 |
| C metadata server access log | **✓✓ 主力** | ✗ | 部分 | 部分 |
| D C++ key/endpoint 交叉引用 | ✓ | 部分 | 部分 | ✓ |
| E 恢复 Prometheus 集成 | ✓ | ✓ | ✓ | ✓ |
| F 启动 Prometheus | ✓ | ✓ | ✓ | ✓ |
| G RDMA 握手日志 | ✗ | **✓✓ 主力** | ✗ | 部分 |
| H 状态机日志 | ✗ | ✗ | **✓✓✓ 核心** | ✗ |
| I batch_is_exist 对齐 | ✗ | ✗ | **✓✓ 主力** | ✗ |
| J decode 真实 load | ✗ | ✗ | ✗ | **✓✓✓ 主力** |

### 11.12 实施路线图（分支 1-4）

> 源自 `mooncake_observability_plan_20260415_cn.md §5`。按"首先看到什么症状"的 bootstrap checklist 分支执行。

**复现前（必做，几分钟）**：
- 方案 A（关 `MC_STORE_CLIENT_METRIC / MC_TE_METRIC`）
- 方案 E（恢复 Prometheus metrics 集成）

**然后跑 `repro_batch_put_failed_x10.sh`**，根据**首先看到的日志**分支：

**分支 1 — 首先看到 `metadata not found` / `Failed to open segment`（→ 类 1）**
- 方案 B、C、D → fork Mooncake → 改代码 → 重编 → 重跑复现

**分支 2 — 首先看到 `handshake timeout` / `packet mismatch` / `mark it inactive`（→ 类 2）**
- 方案 G（G-1~G-4 全部）+ 方案 D（交叉引用仍有用）
- 关注热点主机的 endpoint-pair 时间线，用 G-1 的 MTU/GID/LID/QP 字段区分 simultaneous-open / path mismatch / RDMA 参数不匹配

**分支 3 — 首先看到 `OBJECT_NOT_FOUND(-704)` / revoke 链（→ 本类）**
- 方案 H（H-1~H-4 全部）+ 方案 I + 方案 D
- 关注：
  - `WaitForTransfers slow` 日志揭示累计等待时间
  - `PutStart discard` 日志揭示元数据被擦除时间点
  - `ExistKey MISS (only processing replica)` 日志揭示语义不匹配

**分支 4 — decode 侧 100% hit 但性能差（→ 家族 4）**
- 方案 J

**长期（跨所有分支）**：
- 方案 F（Prometheus + Grafana）
- 推方案 B/G/H 的改动给 Mooncake upstream
- 考虑给 Mooncake 加细分错误码：`METADATA_NOT_FOUND=-801`、`HANDSHAKE_FAIL=-802`

### 11.13 下次复现的交叉验证流程（方案 A+B+C 都做后）

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

判断：
- **同一个 segment 持续 404？** → 某台 prefill 节点 publish 从未发生或过期
- **很多 segment 都 404？** → metadata server 侧有共性问题（TTL / 驱逐 / 清理）
- **在 publish PUT 日志之后还 404？** → metadata 写入和读取之间的一致性问题

---

## 12. 验证命令 + 来源文档

### 12.1 grep 验证命令

```bash
# 关键源码锚点
grep -n "BatchPutWhenPreferSameNode" Mooncake/mooncake-store/src/client_service.cpp
# 预期可见 :1941 附近

grep -n "WaitForTransfers" Mooncake/mooncake-store/src/client_service.cpp
# 预期可见 :1633 附近

grep -n "ExistKey" Mooncake/mooncake-store/src/master_service.cpp
# 预期可见 :432 附近

grep -n "put_start_discard_timeout_sec\|put_start_release_timeout_sec" Mooncake/mooncake-store/src/master_service.cpp
# 预期可见 :850 附近

# TRANSFER_FAIL 赋值点（9 处）
grep -n "SetError.*TRANSFER_FAIL\|ErrorCode::TRANSFER_FAIL" Mooncake/mooncake-store/src/client_service.cpp
# 预期：960, 982, 1102, 1623, 1994, 2323, 2396, 2608, 2628

# 完整错误码表
grep -n "INTERNAL_ERROR\|OBJECT_NOT_FOUND\|TRANSFER_FAIL" Mooncake/mooncake-store/include/types.h
# 预期可见 :208-274 范围

# PR #993 已合入本地
cd Mooncake && git log --oneline --all | grep -E "993"
# 预期 1 条命中

# vLLM 侧 batch_put failed 增强版本
grep -n "batch_put failed" vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_worker.py
# 预期 :479 或 :480（增强后的 WARNING）

grep -n "tp_rank\|req_id\|elapsed\|failed_samples" vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_worker.py
# 预期多行命中（已增强）
```

### 12.2 来源文档映射

| 文件 | 贡献章节 |
| --- | --- |
| `prefill3_decode5_mooncake_deep_dive_20260415.md` + `prefill3_decode5_mooncake_deep_dive_20260415_cn.md` | §1 摘要（prefill_3 主 run 命中）+ §2.1 症状链 + §3.1-3.6 全部根因机制（7 步 + 合并写 + WaitForTransfers + ExistKey + 30s discard + 网络非根因）+ §4.1 完整时间链表 + §4.4 网络证据对照 |
| `batch_put_failed_global_analysis_20260415.md` + `batch_put_failed_global_analysis_20260415_cn.md` | §1 摘要（跨 6 run 计数 + 修复状态）+ §2.3 易误导日志形态 + §3.0 四种错误码（含完整表）+ §3.7 `-1 vs -800` + §3.8 字段缺失 + §3.9 decode 盲区 + §3.10 C++ 信息损失 + §3.11 症状映射表 + §4.2-4.3 跨 run 证据 + §7.1 已完成清单 |
| `mooncake_observability_plan_20260415_cn.md` | §9 诊断 checklist（§0 速查表）+ §11.1-11.13 方案 A-J 全部（整份并入）+ §11.11 覆盖度审视 + §11.12 实施路线图 + §11.13 交叉验证 |
| `mooncake_upstream_issues_prs_20260419_cn.md` 家族 3 + 家族 4 章节 | §6 上游 PR 状态全部（家族 3 PR 表 + 家族 4 PR 表 + 上游未报告清单） |

**未来新证据追加指引**：
- 新 run 出现 `OBJECT_NOT_FOUND` / `batch_put failed` → 在 §4 添加证据小节，标注对应机制步骤
- 新诊断字段上线 → 更新 §3.7-3.9 和 §7.1
- 新方案实装 → 从 §11 的"未做"状态移到 §7.1
- 上游合入 `ExistKey` 三态 / 合并写串行修复 → 更新 §6.1 + §7.1
- 错误码表变更（新加 `METADATA_NOT_FOUND=-801` 等） → 更新 §3.0
