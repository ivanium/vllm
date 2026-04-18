# 问题 4：可观测性、错误归因与诊断盲区

日期：2026-04-16

## 问题卡片

| 字段 | 值 |
| --- | --- |
| 问题编号 | 问题 4 |
| 简称 | 可观测性 / 错误归因 |
| 当前状态 | `Confirmed` |
| 受影响运行 | 跨问题的共性问题；已确认影响 `decode_5_mtc_16`、`prefill_6_mtc_18`、`prefill_3/4/5` 与 `p4_d4_mtc_14` 的诊断与监控 |
| 一行结论 | 目前至少已经确认两个可观测性问题：Python 侧的 transfer-fail 统计原先是错的，而当前日志 / 指标仍然缺少把 `batch_put failed`、`TRANSFER_FAIL` 和 `OBJECT_NOT_FOUND` 正确划入各自问题族所需的关键上下文。 |
| 相关问题 | [问题 1：元数据 / 过期描述符可见性故障](decode5_mtc16_mooncake_correlation_analysis_20260415_cn.md)；[问题 2：RDMA 握手与端点状态故障](prefill6_mooncake_error_analysis_20260415_cn.md)；[问题 3：处理中对象重复 put 与 `OBJECT_NOT_FOUND` 放大链](prefill3_decode5_mooncake_deep_dive_20260415_cn.md) |

## 1. 问题定义

这个问题只讨论：为什么我们现在仍然难以观测并区分问题 1、2、3。它并不定义一个新的数据面根因。

它包含三类内容：
- 错误统计不正确
- warning 和指标里缺少关键信息
- decode 侧存在盲区，导致根因分析不得不依赖 prefill 侧日志

核心问题不是：
- 为什么某次 benchmark 变慢了
- 为什么某个具体端点失败了

核心问题是：
- 为什么现有日志和指标仍然会让不同故障族在表面上看起来非常相似，从而误导判断

## 2. 当前状态

- 证据强度：`Confirmed`
- 优先级：高，但属于支撑性问题
- 当前结论的单一事实来源：
  - 可观测性缺口不会直接制造数据面故障，但它们会扭曲监控、延长根因定位时间，并让一个症状族看起来像好几个互不相干的 bug。

已经稳定确认的事实：
- Python 侧过去把 `-1` 误当成 transfer failure，而真正的 transfer failure 应该是 `-800`
- 旧版 `batch_put failed` warning 缺少关键的 request、rank、耗时和 failed-key 上下文
- 在 `decode5` 里，decode 侧的 hit-rate 视角无法区分“真正成功加载”与“静默 fallback / 重算”

## 3. 受影响范围

在这批 benchmark 里，至少有 6 个 run 出现了 `batch_put failed` 或紧密相关的底层故障：

| Benchmark run | 受影响日志 | `batch_put failed` | `TRANSFER_FAIL` 相关行数 | 最可能映射的问题 |
| --- | --- | ---: | ---: | --- |
| `decode_5_mtc_16` | `prefill-0-gb200-rack1-09.log` | 2205 | 106994 | [问题 1](decode5_mtc16_mooncake_correlation_analysis_20260415_cn.md) |
| `prefill_6_mtc_18` | 4 份 prefill 日志 | 6 | 18 | [问题 2](prefill6_mooncake_error_analysis_20260415_cn.md) |
| `prefill_3_mtc_12` | `prefill-2-gb200-rack1-13.log` | 2 | 16 | [问题 3](prefill3_decode5_mooncake_deep_dive_20260415_cn.md) |
| `prefill_5_mtc_16` | `prefill-1-gb200-rack1-07.log` | 2 | 12 | 问题 2 或 3 的较弱信号 |
| `prefill_4_mtc_14` | `prefill-0-gb200-rack1-01.log` | 1 | 4 | 问题 2 的较弱信号 |
| `p4_d4_mtc_14` | `prefill-0-gb200-rack1-06.log` | 1 | 2 | 问题 2 的较弱信号 |

健康对照组：
- `decode_4_mtc_14`
- `decode_6_mtc_18`
- `prefill_2_mtc_*`

## 4. 主要症状与诊断指纹

这个问题最典型的症状不是某一条单独的报错，而是“歧义”本身。

最容易误导人的日志形态有：

1. `batch_put failed`
   - 这是聚合症状，不是根因

2. `codes={-800}`
   - 如果没有 code-name 映射，很容易意识不到它其实是 `TRANSFER_FAIL`

3. `first_key=...`
   - 历史上它不一定是第一个失败 key，只是 batch 里的第一个 key

4. `External prefix cache hit rate: 100.0%`
   - 这并不能证明 external KV load 真的成功了

## 5. 当前最强的诊断机制链

这里的机制是“诊断层面”的，而不是“数据面”的：

1. 问题 1、2 或 3 中的某个真实数据面故障先发生。
2. Python 和 C++ 日志没有携带足够的 request、key、endpoint 和 state 上下文。
3. 指标可能把这个故障归到了错误的错误桶里。
4. decode 侧可观测性无法告诉我们：某个 “hit” 最终是否真的变成了一次成功的 external KV load。
5. 因此，`batch_put failed`、`TRANSFER_FAIL` 和 `OBJECT_NOT_FOUND` 很容易被误读成彼此独立的根因，而不是同一条问题链上的不同层次症状。

## 6. 关键证据

### 6.1 已确认的 transfer-fail 统计 bug

历史上的 Python 代码里有：

```python
transfer_fail_keys = sum(1 for i in failed if res[i] == -1)
```

但 Mooncake C++ 里定义的是：
- `-1 = INTERNAL_ERROR`
- `-800 = TRANSFER_FAIL`

影响：
- `vllm:mooncake_store_put_transfer_fail_keys` 统计的是错误的类别
- 真正的 `TRANSFER_FAIL(-800)` key 被归到了 `other_failed_keys`

这是一个已经确认的统计问题，不是猜测。

### 6.2 `batch_put failed` 过去缺少最重要的上下文

旧版 warning 能稳定展示：
- `len(failed)` / `len(keys)`
- `failed_codes`
- `total_bytes`
- `keys[0]`

但缺少：
- `req_id`
- `tp_rank`
- `elapsed`
- failed-key 样例
- 可读的错误码名称

最关键的是：
- `keys[0]` 不保证一定是失败 key

### 6.3 decode 侧的 hit-rate 还不够

在 `decode_5_mtc_16` 中，decode 日志稳定会显示：
- `External prefix cache hit rate: 100.0%`

但不会显示：
- `batch_get failed`
- `load error`
- `fallback`
- `recompute`

这意味着 decode 侧仍然回答不了最关键的问题：
- 这个 hit 最后到底有没有真的变成一次成功的 external KV load？

### 6.4 C++ 与 Python 暴露的是同一条问题链的不同层次，但目前没有共享层级

同一个底层故障，可能表现成：
- `metadata not found`
- `Failed to open segment`
- `Transfer submission failed for key`
- `TRANSFER_FAIL`
- `batch_put failed`

如果没有稳定的症状层级，这些东西就很容易被误当成不同 bug。

### 6.5 完整的 Mooncake 错误码表应保持贴近文档

来源：
- `third_partys/Mooncake/mooncake-store/include/types.h:208-274`

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
| -704 | `OBJECT_NOT_FOUND` | object 不存在 |
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
| -800 | `TRANSFER_FAIL` | 传输操作失败 |
| -900 | `RPC_FAIL` | RPC 失败 |
| -1000 | `ETCD_OPERATION_ERROR` | etcd 操作失败 |
| -1001 | `ETCD_KEY_NOT_EXIST` | etcd key 缺失 |
| -1002 | `ETCD_TRANSACTION_FAIL` | etcd 事务失败 |
| -1003 | `ETCD_CTX_CANCELLED` | etcd context 被取消 |
| -1004 | `OPLOG_ENTRY_NOT_FOUND` | oplog 条目不存在 |
| -1010 | `UNAVAILABLE_IN_CURRENT_STATUS` | 当前状态下不可用 |

### 6.6 缺失的诊断字段应继续作为明确清单保留

旧版 `batch_put failed` 缺失字段：

| 缺失字段 | 为什么重要 |
| --- | --- |
| `req_id` | 把故障绑定到某次 request 生命周期 |
| `tp_rank` | 帮助区分按 rank 分裂的问题 |
| `elapsed` | 区分快速失败、60 秒超时还是 finalize 阶段失败 |
| failed-key 样例 | 帮助定位具体 block / rank / hash 的失败 |
| 可读的 code name | 避免只看到原始 `codes={-800}` 的歧义 |

旧版 `batch_get failed` 的缺口：
- `req_id`
- `tp_rank`
- `elapsed`
- failed key / 可读 code name

旧版 exception-handler 的缺口：
- 缺少 `req_id`
- 缺少 `tp_rank`
- 缺少 traceback

### 6.7 C++ 源码侧的报错起点仍值得统一放在一处

| 信号 | 源码位置 | 当前含义 |
| --- | --- | --- |
| `metadata not found` | `http_metadata_server.cpp:39,87` | 元数据存储里缺少这个 key |
| `Failed to open segment` | `transfer_task.cpp:498,535,639` | `openSegment(handle.transport_endpoint_)` 失败 |
| `Transfer submission failed for key` | `client_service.cpp:1621,1992` | transfer submit 失败并被记为 `TRANSFER_FAIL` |
| `TRANSFER_FAIL` 赋值点 | `client_service.cpp:960,982,1102,1623,1994,2323,2396,2608,2628` | 多条读写路径里统一归一化成 transfer failure |

## 7. 症状到问题的映射

这张表替代旧的 “mode A / mode B” 混合叫法，应该作为默认的分流参考。

| 症状 / 关键词 | 是否不应直接当成根因 | 最可能指向 |
| --- | --- | --- |
| `batch_put failed` | 是 | 问题 1 / 2 / 3 的聚合症状 |
| `TRANSFER_FAIL(-800)` | 是 | 问题 1 或 2，有时也是问题 3 的上游触发器 |
| `OBJECT_NOT_FOUND(-704)` | 是 | 优先看问题 3 |
| `metadata not found` | 否 | 优先看问题 1 |
| `Failed to open segment` | 否 | 优先看问题 1 |
| `handshake timeout` | 否 | 优先看问题 2 |
| `packet mismatch` | 否 | 优先看问题 2 |
| `inactive endpoint` | 否 | 优先看问题 2 |
| decode 侧 `100% hit rate` | 是 | 只表示“被计成 hit”；必须结合问题 4 来解读 |

## 8. 这不是什么

这个问题不是：

1. 一个新的数据面根因
2. `batch_put failed` 本身
3. 只在某一个 run 里出现的问题

它是一个跨问题解释：说明为什么诊断过程仍然慢、仍然容易出错。

## 9. 与其他问题的关系

- 与 [问题 1](decode5_mtc16_mooncake_correlation_analysis_20260415_cn.md) 的关系：
  - metadata `PUT/DELETE` 与 descriptor 生命周期日志，正是能最快把问题 1 从 `Strong hypothesis` 升级到 `Confirmed` 的证据

- 与 [问题 2](prefill6_mooncake_error_analysis_20260415_cn.md) 的关系：
  - 握手字段增强后，才能把 simultaneous-open、path mismatch 和真实 RDMA 参数不匹配区分开

- 与 [问题 3](prefill3_decode5_mooncake_deep_dive_20260415_cn.md) 的关系：
  - `ExistKey`、`PutStart discard` 和 `WaitForTransfers` 的结构性信息，是补齐 `OBJECT_NOT_FOUND` 放大链最终证据缺口的关键

## 10. 仍未证明的部分

对问题 4 来说，剩余未知点不是“这些盲区是否存在”，而是“哪些修复已经在线，哪些还只停留在本地改动”。

未决项：
- 这些可观测性补丁里，哪些已经出现在当前运行时构建里
- decode 侧的“真实 load 成功 / fallback”应当在 connector、scheduler 还是 metrics 层发出
- metadata 生命周期日志应放在 metadata server、transfer metadata 层，还是两边都放

## 11. 修复方向

### 11.1 错误统计

- 定义 `MOONCAKE_TRANSFER_FAIL = -800`
- 修正 `transfer_fail_keys` 的统计逻辑
- 除了原始整数外，同时导出可读的 code name

### 11.2 Python warning 上下文

至少，`batch_put failed` / `batch_get failed` 应包含：
- `req_id`
- `tp_rank`
- `elapsed`
- `first_failed_key`
- `failed_examples`
- 可读的 code name

### 11.3 C++ 状态点日志

最高优先级补充：
- `ExistKey()`：区分“元数据不存在”和“仍在 processing 但尚未完成”
- `PutStart()`：对 30 秒 discard 分支做结构化日志
- `WaitForTransfers()`：记录 pending future 总数与当前 future 下标
- metadata server：成功的 `PUT / DELETE`
- transfer metadata：成功的 descriptor 注册 / 删除

### 11.4 decode 侧真实结果可观测性

至少需要：
- 被计为 external-prefix hit 的 block 数
- load success 数
- load failure 数
- fallback / recompute 数

缺少这一层的话，`decode5` 这类问题仍然只能主要靠 prefill 日志来追根因。

### 11.5 已在本地加上的可观测性工作不要丢

vLLM 侧已经在本地加入的内容：
- `batch_put failed` 日志输出 `req_id`
- `batch_put failed` 日志输出 `tp_rank`
- `batch_put failed` 日志输出 `first_failed_key`
- `batch_put failed` 日志输出 `failed_examples`
- 计数器：`put_failed_batches / put_failed_keys / put_transfer_fail_keys / put_no_available_handle_keys / put_other_failed_keys`

Mooncake 侧已经在本地加入的内容：
- 握手失败日志增强为包含 `endpoint / NIC path / MTU / GID / LID / QP`
- transfer 失败日志增强为包含 `pending_transfer_index / replica_index / strategy / full replica descriptor`

### 11.6 验证与 dashboard 状态应继续挂在这个问题下

已经验证：
- `.venv/bin/python -m pytest tests/v1/kv_connector/unit/test_mooncake_store_worker.py -q`
- `.venv/bin/python -m pytest tests/v1/kv_connector/unit/test_mooncake_store_metrics.py -q`

尚未重新验证：
- 这一轮里还没有重编译并重跑本地的 Mooncake C++ 可观测性修改

分析时的 Grafana / Prometheus 状态：
- Grafana 容器在运行
- Prometheus 容器未运行
- Grafana datasource 指向 `http://localhost:9090`，当前不可达
- dashboard 搜索结果为空；`mooncake-overview` 未加载

这意味着：
- 就算代码级补丁存在，dashboard 级的可观测性目前也还没有达到稳定可用状态

## 12. 验证计划

要认为这个问题已经得到足够处理，以下条件都应成立：

1. `TRANSFER_FAIL` 的 Prometheus 统计与真实 `-800` 日志出现次数一致。
2. `batch_put failed` warning 能直接展示 `req_id / tp_rank / elapsed / failed key samples`。
3. `batch_get failed` warning 也带有同样的上下文。
4. `ExistKey`、`PutStart discard` 和 `WaitForTransfers` 日志足以直接证明或证伪问题 3。
5. metadata `PUT/DELETE` 与 descriptor 注册 / 删除日志足以直接证明或证伪问题 1。
6. decode 侧指标能够区分“被计作 hit”和“真实 load 成功”。

### 12.1 复现入口应继续保留在这里

主要复现：
- `decode_5_mtc_16`
  - 配置：`bench_results/kimi_pd_p1d1to17_mooncake_nsys_autosweep/decode_5_mtc_16/attempt_1/rendered_config.yaml`
  - 命令：`vigil -c bench_results/kimi_pd_p1d1to17_mooncake_nsys_autosweep/decode_5_mtc_16/attempt_1/rendered_config.yaml`

次要复现：
- `prefill_6_mtc_18`
  - 配置：`bench_results/pd_kimi_nsys_prefill6_light/prefill_6_mtc_18/rendered_config.yaml`
  - 命令：`vigil -c bench_results/pd_kimi_nsys_prefill6_light/prefill_6_mtc_18/rendered_config.yaml`

## 13. 稳定结论措辞

请以下面这段话作为这个问题的单一结论表述：

> 目前至少已经确认两个可观测性问题。第一，transfer-fail 统计此前把 `TRANSFER_FAIL(-800)` 归到了错误的错误桶。第二，Python、C++ 与 decode 侧日志仍然缺少把 object、segment、endpoint、request 以及真实 external-load 结果串起来所需的字段。这些问题不会直接制造数据面故障，但它们会扭曲监控，并延缓对问题 1、2、3 的根因隔离。
