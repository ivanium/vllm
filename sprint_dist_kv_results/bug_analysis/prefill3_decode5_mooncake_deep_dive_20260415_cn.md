# 问题 3：处理中对象重复 Put 与 `OBJECT_NOT_FOUND` 放大链

日期：2026-04-16

## 问题卡片

| 字段 | 值 |
| --- | --- |
| 问题编号 | 问题 3 |
| 简称 | processing object re-put / `OBJECT_NOT_FOUND` |
| 当前状态 | `Strong hypothesis` |
| 受影响运行 | 主要是 `prefill_3_mtc_12`；在其他 run 中，`OBJECT_NOT_FOUND(-704)` 还没有被确认为主导性矛盾 |
| 一行结论 | 当前最强判断是：`prefill3` 中的 `OBJECT_NOT_FOUND(-704)` 并不是一个独立的 master 侧根因，而是由 `TRANSFER_FAIL`、segment 级等待放大、`ExistKey` 语义不匹配以及 30 秒过期 processing metadata 丢弃共同制造出来的二次故障。 |
| 相关问题 | [问题 1：元数据 / 过期描述符可见性故障](decode5_mtc16_mooncake_correlation_analysis_20260415_cn.md)；[问题 2：RDMA 握手与端点状态故障](prefill6_mooncake_error_analysis_20260415_cn.md)；[问题 4：可观测性、错误归因与诊断盲区](batch_put_failed_global_analysis_20260415_cn.md) |

## 1. 问题定义

这个问题指的是如下链条：
- 某些 put 操作先进入 `TRANSFER_FAIL(-800)`，或者在 transfer 阶段长时间 pending
- 上层的去重 / 重试逻辑把“仍处于 processing、但尚未完成”的对象当成“缺失”
- 在 `put_start_discard_timeout_sec=30` 之后到来的新 `PutStart` 会擦掉旧的 processing metadata
- 原始 put 的延迟 `BatchPutEnd` 随后执行时，就会因为元数据已被擦除而失败为 `OBJECT_NOT_FOUND(-704)`

这里真正关键的问题不是“为什么 transfer 一开始会失败？”

真正关键的问题是：
- 为什么 finalize 阶段会进一步暴露出一个看起来像独立 master 侧根因的 `OBJECT_NOT_FOUND`？

这个问题不包括：
- `metadata not found / Failed to open segment`，那属于 [问题 1](decode5_mtc16_mooncake_correlation_analysis_20260415_cn.md)
- `handshake timeout / packet mismatch / inactive endpoint`，那属于 [问题 2](prefill6_mooncake_error_analysis_20260415_cn.md)
- 日志与指标缺口，属于 [问题 4](batch_put_failed_global_analysis_20260415_cn.md)

## 2. 当前状态

- 证据强度：`Strong hypothesis`
- 优先级：中高
- 当前结论的单一事实来源：
  - `prefill3` 里的 `OBJECT_NOT_FOUND(-704)` 最好理解成 `TRANSFER_FAIL` 之后的二次状态机效应，而不是一个全新的、完全独立的 master bug。

已经稳定的事实：
- 从 `08:07` 到 `08:11` 出现了多次重复的 60 秒 timeout
- `OBJECT_NOT_FOUND` 只会在多个更早的 `TRANSFER_FAIL` 已经堆积之后才出现
- 同一线程后续的 revoke 行，覆盖了此前同线程里超时的那些 key
- Mooncake 源码里完整存在这条放大链所需的三个机制：
  - segment 级合并写
  - `WaitForTransfers()` 中的串行等待
  - `PutStart()` 里对旧 processing metadata 的 30 秒丢弃

## 3. 受影响范围

主运行：
- `bench_results/pd_kimi_nsys_prefill3_light/prefill_3_mtc_12/prefill-2/prefill-2-gb200-rack1-13.log`

主要场景：
- external KV 的多实例 prefill 写入
- `prefer_alloc_in_same_node` 合并写路径
- 某些 batch 或相邻 batch 中同时存在多个 pending / timeout 的 transfer

主要风险：
- `TRANSFER_FAIL` 之后又跟出了 `OBJECT_NOT_FOUND`
- 这很容易被误读成第二类完全独立的 master 侧 bug

## 4. 主要症状与日志指纹

这里最具识别度的不是某一行报错，而是一条时间链：

```text
Failed to complete transfers after 60 seconds
  -> Transfer failed for key: TRANSFER_FAIL
  -> Continue waiting for other futures in the same op
  -> Later batches hit more timeouts
  -> Finalize put for key: OBJECT_NOT_FOUND
  -> Revoke earlier failed keys
```

代表性关键词：
- `Failed to complete transfers after 60 seconds`
- `Transfer failed for key`
- `Failed to finalize put for key`
- `OBJECT_NOT_FOUND`
- `revoke`

代表性位置：
- `08:07` 的第一次 timeout：`.../prefill-2-gb200-rack1-13.log:2490`
- `08:08` 的第二次 timeout：`.../prefill-2-gb200-rack1-13.log:3240`
- `08:09` 第一次出现混合 `TRANSFER_FAIL + OBJECT_NOT_FOUND` 的 batch：`.../prefill-2-gb200-rack1-13.log:3990`
- `08:10` 的第四次 timeout：`.../prefill-2-gb200-rack1-13.log:4808`
- `08:11` 第二次出现混合故障 batch：`.../prefill-2-gb200-rack1-13.log:5579`

## 5. 当前最强的机制链

当前最强解释是：

1. `BatchPutWhenPreferSameNode()` 按 `transport_endpoint_` 对多个逻辑 key 分组，并以 segment 为粒度提交 transfer。
2. 某个 segment 级 transfer future 超时，于是这次操作里记录了 `TRANSFER_FAIL(-800)`。
3. `WaitForTransfers()` 在第一次失败后不会立刻停止；它会继续串行等待同一 op 里的其他 future，每个 future 都可能再等 60 秒。
4. 因此，单次 op 的 finalize 延迟可能不是 60 秒，而是 180、240 甚至 300 秒。
5. 与此同时，vLLM 发送端会执行 `batch_is_exist(keys)`，而 Mooncake 的 `ExistKey()` 对“元数据还在，但没有任何 completed replica”的对象会返回 `false`。
6. 如果新的 `PutStart` 在 30 秒后再次打到这些 key，master 会丢弃旧的 processing metadata。
7. 原始 put 那个被延迟的 `BatchPutEnd` 最终才执行时，就会面对已经被擦除的元数据，从而失败为 `OBJECT_NOT_FOUND(-704)`。

哪些是事实，哪些是推断：
- 第 1、3、5、6 步都有源码直接支持。
- 第 2、4、7 步与观察到的日志时间线高度吻合。
- 目前还缺的最后一块证据，是 master 侧对精确 `PutStart -> discard -> delayed BatchPutEnd` 顺序的最终生命周期证明。

## 6. 关键证据

### 6.1 `08:07` 到 `08:11` 这条链是累积性的，不是一组互相独立的失败

| 时间 | 线程 / pid | 事件 | 含义 |
| --- | --- | --- | --- |
| `08:07:25` | `2137498` | timeout + `TRANSFER_FAIL` | 第一次 timeout |
| `08:07:25` | `2137533` | timeout + `TRANSFER_FAIL` | 第二个线程也开始 timeout |
| `08:08:25` | `2137498` | timeout + `TRANSFER_FAIL` | 第二轮 |
| `08:08:25` | `2137533` | timeout + `TRANSFER_FAIL` | 第二轮 |
| `08:09:25` | `2137498` | timeout + `TRANSFER_FAIL` | 第三轮 |
| `08:09:25` | `2137533` | timeout + `TRANSFER_FAIL` | 第三轮 |
| `08:09:25` | `2137533` | 同 batch 命中 `OBJECT_NOT_FOUND` | finalize 已进入二次故障阶段 |
| `08:09:25` | `2137533` | revoke 之前 3 个失败 key | 覆盖了该线程里更早 timeout 的 key |
| `08:10:25` | `2137498` | timeout + `TRANSFER_FAIL` | 第四轮 |
| `08:11:25` | `2137498` | timeout + `TRANSFER_FAIL` | 第五轮 |
| `08:11:25` | `2137498` | 同 batch 命中 `OBJECT_NOT_FOUND` | 第二次 finalize 异常 |
| `08:11:25` | `2137498` | revoke 之前 5 个失败 key | 覆盖了 `08:07` 到 `08:11` 的 key |

这是最强的证据之一，说明状态会跨过第一次 60 秒 timeout 持续存在，并继续影响后续的 finalize / revoke 行为。

### 6.2 Mooncake 这里不是逐 key transfer，而是按 segment 合并

相关源码：
- `third_partys/Mooncake/mooncake-store/src/client_service.cpp:1941`

重要逻辑：
- key 按 `buffer_descriptor.transport_endpoint_` 分组
- transfer 按合并后的 segment group 提交
- `WaitForTransfers()` 对合并后的 op 工作，然后再把结果写回原始逻辑 op

这解释了为什么：
- 日志里可能只看到少数几条 `Transfer failed for key`
- 但更大一批逻辑 key 会在稍后的 finalize / revoke 阶段一起失败

### 6.3 `WaitForTransfers()` 会串行等待每一个 future

相关源码：
- `third_partys/Mooncake/mooncake-store/src/client_service.cpp:1633`
- `third_partys/Mooncake/mooncake-store/src/transfer_task.cpp:297`
- `third_partys/Mooncake/mooncake-store/src/transfer_task.cpp:341`

含义：
- `WaitForTransfers()` 会逐个调用 `pending_transfers[i].get()`
- 即使前一个 future 已经失败，它仍会继续等待后面的 future
- 每个 future 最多可以再等 60 秒

这就把：
- 一个失败 future 变成大约 60 秒
- 三个 pending future 变成大约 180 秒
- 五个 pending future 变成大约 300 秒

### 6.4 `batch_is_exist` 与 `ExistKey` 对“存在”的语义并不一致

vLLM 发送端行为：
- `mooncake_store_worker.py:397-455`
- 任何 `exists != 1` 的 key 都会被当成缺失，从而允许 re-put

Mooncake `ExistKey()` 行为：
- `master_service.cpp:432`
- 只有至少有一个 completed replica 时才返回 `true`
- 对“元数据仍存在，但只有 processing replica”的对象会返回 `false`

这个语义不一致非常关键：
- 对 vLLM 来说，对象看起来是 missing
- 对 Mooncake 来说，对象可能其实仍然存在，只是还没完成

### 6.5 30 秒 discard 分支会擦掉旧 processing metadata

相关源码：
- `third_partys/Mooncake/mooncake-store/src/master_service.cpp:850`

相关配置：
- `scripts/mooncake/mooncake_master.log:2`
- `put_start_discard_timeout_sec=30`
- `put_start_release_timeout_sec=600`

含义：
- 如果没有 completed replica
- 且旧的 `put_start_time` 已经超过 30 秒
- 一个新的 `PutStart` 就可以丢弃旧 processing metadata，并继续执行

### 6.6 纯粹用“网络慢”来解释，这个说法证据较弱

在 `08:09:25` 附近，transfer-engine 的统计并不符合“整个数据面都挂了”这种解释：
- 吞吐仍有 `586.74 MB/s`
- 延迟分布仍然以 `200-1000us` 为主，只有少数长尾离群值

因此，与其说“整个网络卡住了”，不如说“这里更像一个狭义的状态机 / 生命周期问题”，这个解释要强得多。

## 7. 这不是什么

当前证据不支持以下解释：

1. 这不是一个已经单独证实的全新 master bug 族。
2. 这不只是“网络慢了 60 秒”。
3. 这不是 [问题 1](decode5_mtc16_mooncake_correlation_analysis_20260415_cn.md) 的 metadata-404 可见性故障族。
4. 这不是 Python warning 自己就能解释清楚的问题。

## 8. 与其他问题的关系

- 与 [问题 1](decode5_mtc16_mooncake_correlation_analysis_20260415_cn.md) 的关系：
  - 问题 1 的核心是 metadata / stale descriptor 查找失败
  - 问题 3 则可以从多种上游 `TRANSFER_FAIL` 开始，其核心是后续状态机如何把它放大成 `OBJECT_NOT_FOUND`

- 与 [问题 2](prefill6_mooncake_error_analysis_20260415_cn.md) 的关系：
  - 问题 2 类型的 `TRANSFER_FAIL` 也可能成为问题 3 的上游触发器
  - 但 `prefill6` 当前并没有把 `OBJECT_NOT_FOUND` 暴露成主导性矛盾

- 与 [问题 4](batch_put_failed_global_analysis_20260415_cn.md) 的关系：
  - `ExistKey`、`PutStart discard` 与 `WaitForTransfers` 的 instrumentation，正是最可能把这个问题从 `Strong hypothesis` 升级到 `Confirmed` 的可观测性补充

## 9. 仍未证明的部分

目前缺的不是整体机制，而是最终的 master 侧证据闭环。

最有价值的缺失证据是：
- 这个 key 的第一次 `PutStart`
- 完成前对象的 processing state
- 之后进入 30 秒 discard 分支的那个精确 `PutStart`
- 元数据被擦除的事件本身
- 原始延迟 `BatchPutEnd` 随后因 `OBJECT_NOT_FOUND` 失败

这也是为什么这个问题现在仍然是 `Strong hypothesis`，而不是 `Confirmed`。

## 10. 修复方向

首先，加上刚好能闭合生命周期证据链的最小可观测性：
- 在 `ExistKey()` 中区分“元数据确实不存在”和“仍在 processing、但尚未完成”
- 在 `PutStart()` 中，当 discard 分支触发时记录 key、age、client ID 和 replica 数量
- 在 `WaitForTransfers()` 中记录该 op 的 future 总数和当前下标

然后，如果生命周期证据确认了这条机制，再考虑逻辑修复，例如：
- 阻止上层把 processing 对象当成普通 missing 对象
- 重新审视 `batch_is_exist` 与 `ExistKey` 之间的契约
- 改进合并 op 在 finalize 前允许被拖延的时长

## 11. 验证计划

要认为这个问题已经修复或被完全证实，至少应满足：

1. 复现同样 workload 时，不再出现 `TRANSFER_FAIL` batch 后续又演化成 `OBJECT_NOT_FOUND`。
2. `WaitForTransfers()` 日志能直接揭示每个 op 的 future 数量与累计等待行为。
3. `ExistKey()` 日志能区分“processing 未完成”和“元数据缺失”。
4. `PutStart discard` 日志能揭示旧 processing metadata 是否真的被覆盖。
5. 如果应用了逻辑修复，`BatchPutEnd` 不应再因为旧 metadata 被丢弃而命中 `OBJECT_NOT_FOUND`。

## 12. 稳定结论措辞

请以下面这段话作为这个问题的单一结论表述：

> 在 `prefill3` 中，`OBJECT_NOT_FOUND(-704)` 最合理的解释是 `TRANSFER_FAIL(-800)` 之后的二次状态机故障。segment 级合并写与 `WaitForTransfers()` 中串行的 60 秒等待会把 finalize 拖延到几分钟；与此同时，vLLM 会把“仍在 processing 但未完成”的对象当成缺失对象，而后续 `PutStart` 又可能在 30 秒后丢弃旧元数据。于是，原始延迟的 `BatchPutEnd` 最终就会撞上 `OBJECT_NOT_FOUND`。
