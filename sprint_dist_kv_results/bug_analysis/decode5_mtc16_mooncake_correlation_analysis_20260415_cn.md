# Decode-5 MTC-16 Mooncake 报错与性能异常相关性分析

日期：2026-04-15

相关文档：
- `sprint_dist_kv_results/bug_analysis/prefill6_mooncake_error_analysis_20260415_cn.md`
- `sprint_dist_kv_results/bug_analysis/decode5_mtc16_mooncake_correlation_analysis_20260415.md`
- `bench_results/kimi_pd_p1d1to17_mooncake_nsys_autosweep/decode_5_mtc_16/attempt_1/*`

分析范围：
- 主分析对象：
  - `bench_results/kimi_pd_p1d1to17_mooncake_nsys_autosweep/decode_5_mtc_16`
- 对照组：
  - `bench_results/kimi_pd_p1d1to17_mooncake_nsys_autosweep/decode_4_mtc_14`
  - `bench_results/kimi_pd_p1d1to17_mooncake_nsys_autosweep/decode_6_mtc_18`
- 目标：
  - 判断 `decode_5_mtc_16` 中 Mooncake 报错是否和性能异常强相关
  - 尽可能将失败请求映射到 multi-turn benchmark 的具体阶段
  - 横向对比 `decode_4/5/6` 的 prefill 秒级指标

说明：
- 下文路径均为 repo 相对路径。
- 下文行号均基于当前工作区 2026-04-15 的本地日志副本。
- 这份文档结合了直接日志阅读和本地脚本统计。
- 本次 benchmark 没有开启 `--save-detailed`，因此无法恢复精确的 `conversation_id + turn_id + request_id` 三级映射。

## 执行摘要

`decode_5_mtc_16` 中的 Mooncake 报错与性能异常是强相关的。

当前最佳判断：
- 这不是一个泛化意义上的 `decode=5` 扩容性能问题。
- 主导问题是 prefill 侧出现了一段持续较长的 Mooncake 写入 / 发布失败窗口，核心报错链条是：
  - `metadata not found`
  - `Failed to open segment`
  - `Transfer submission failed for key`
  - `batch_put failed`
- 这些失败几乎精确覆盖了：
  - turn-band 1 的全部失败请求
  - turn-band 2 的全部失败请求
  - turn-band 3 最早进入的少数请求
- 这一时间段恰好对应 benchmark 最关键的“前几轮生成并复用外部 KV”的阶段。
- 对照组 `decode_4_mtc_14` 和 `decode_6_mtc_18` 没有出现相同的 failed-batch 指标，而且最终 external prefix hit-rate 明显更高。

最可能的因果链：

`prefill 侧 Mooncake metadata / segment 可见性异常 -> 早期请求的 KV publish 失败 -> 下一轮拿不到或拿不全应有的 external prefix reuse -> Turn 2 / Turn 3 的 TTFT 和 E2E 被显著拉高 -> 系统逐步自愈 -> 后续轮次恢复`

## 1. Benchmark 层面的异常现象

从 benchmark 总体结果看，`decode_5` 相比 `decode_4` 和 `decode_6` 是非常明显的异常点。

| Case | Req/s | Out tok/s | Mean TTFT (s) | P90 TTFT (s) | Mean E2E (s) | P90 E2E (s) | Turn 1 TTFT (s) | Turn 2 TTFT (s) | Turn 3 TTFT (s) | Turn 4 TTFT (s) | Turn 5 TTFT (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `decode_4_mtc_14` | 1.44 | 430.53 | 6.88 | 13.31 | 9.62 | 15.58 | 27.18 | 12.05 | 3.54 | 3.77 | 3.61 |
| `decode_5_mtc_16` | 0.96 | 288.63 | 13.73 | 35.03 | 16.48 | 37.26 | 28.04 | 35.81 | 33.29 | 12.53 | 4.62 |
| `decode_6_mtc_18` | 1.42 | 426.24 | 9.81 | 26.93 | 12.49 | 31.53 | 31.07 | 20.80 | 5.65 | 5.65 | 5.80 |

关键观察：
- `decode_5` 的吞吐相对前后两个点明显下降。
- `Turn 1` 的 TTFT 其实并不异常，和 `decode_4`、`decode_6` 在同一量级。
- 真正异常的是 `Turn 2` 和 `Turn 3`：
  - `decode_5 Turn 2 = 35.81s`
  - `decode_4 Turn 2 = 12.05s`
  - `decode_6 Turn 2 = 20.80s`
- 到 `Turn 5+`，`decode_5` 又回到了 `4.5-4.8s` 左右的稳定区间，说明它不是整个 run 全程都坏掉。

关键 benchmark 文件：
- `bench_results/kimi_pd_p1d1to17_mooncake_nsys_autosweep/decode_5_mtc_16/attempt_1/hook_post_serve_0_vllm-bench.log:29`
- `bench_results/kimi_pd_p1d1to17_mooncake_nsys_autosweep/decode_5_mtc_16/attempt_1/hook_post_serve_0_vllm-bench.log:58`
- `bench_results/kimi_pd_p1d1to17_mooncake_nsys_autosweep/decode_5_mtc_16/attempt_1/hook_post_serve_0_vllm-bench.log:87`
- `bench_results/kimi_pd_p1d1to17_mooncake_nsys_autosweep/decode_5_mtc_16/attempt_1/hook_post_serve_0_vllm-bench.log:116`
- `bench_results/kimi_pd_p1d1to17_mooncake_nsys_autosweep/decode_5_mtc_16/attempt_1/hook_post_serve_0_vllm-bench.log:174`

## 2. Decode-5 实际出现的错误类型

这次 `decode_5_mtc_16` 并没有复现 `prefill6` 文档里的那条握手超时链。

没有看到的报错：
- `handshake timeout`
- `packet mismatch`
- `mark it inactive`
- `Failed to complete transfers after 60 seconds`

实际出现的报错：
- `metadata not found`
- `Failed to open segment`
- `Transfer submission failed for key`
- `failed: TRANSFER_FAIL`
- `batch_put failed`

启动后最早一批代表性日志：
- `bench_results/kimi_pd_p1d1to17_mooncake_nsys_autosweep/decode_5_mtc_16/attempt_1/prefill-0/prefill-0-gb200-rack1-09.log:1413`
- `bench_results/kimi_pd_p1d1to17_mooncake_nsys_autosweep/decode_5_mtc_16/attempt_1/prefill-0/prefill-0-gb200-rack1-09.log:1415`
- `bench_results/kimi_pd_p1d1to17_mooncake_nsys_autosweep/decode_5_mtc_16/attempt_1/prefill-0/prefill-0-gb200-rack1-09.log:1416`

第一批 batch 级失败：
- `bench_results/kimi_pd_p1d1to17_mooncake_nsys_autosweep/decode_5_mtc_16/attempt_1/prefill-0/prefill-0-gb200-rack1-09.log:1918`
- `bench_results/kimi_pd_p1d1to17_mooncake_nsys_autosweep/decode_5_mtc_16/attempt_1/prefill-0/prefill-0-gb200-rack1-09.log:2033`

解释：
- 这次 `decode_5` 的故障形态，更像是“Mooncake metadata / segment 在需要的时候不可见”，而不是 `prefill6` 中那种“startup handshake timeout -> inactive endpoint”的链路。

## 3. 报错时间线与吞吐时间线的对应关系

prefill 侧失败窗口大致持续于：
- 首次失败：`16:46:02`
- 最后一次失败：`16:48:50`

在这个窗口内：
- `batch_put failed` 涉及 `69` 个唯一失败 `req_id`
- `router.log` 中仅完成了 `68` 个请求

基于 `router.log` 的完成时间统计：
- 失败窗口内完成速率：`68 / 169s = 0.402 req/s`
- 失败窗口结束后完成速率：`252 / 161s = 1.565 req/s`

也就是说，报错停止后，请求完成速率出现了接近 `3.9x` 的阶段跃迁。

解释：
- 这不是简单的“时间上同时发生”。
- 而是系统在 Mooncake 报错持续期间进入了一个明确的低吞吐阶段。
- 报错停掉以后，吞吐立刻显著上升。

关键日志位置：
- 最后仍在失败的一段：
  - `bench_results/kimi_pd_p1d1to17_mooncake_nsys_autosweep/decode_5_mtc_16/attempt_1/prefill-0/prefill-0-gb200-rack1-09.log:648304`
  - `bench_results/kimi_pd_p1d1to17_mooncake_nsys_autosweep/decode_5_mtc_16/attempt_1/prefill-0/prefill-0-gb200-rack1-09.log:648395`
- 失败计数仍然存在于：
  - `bench_results/kimi_pd_p1d1to17_mooncake_nsys_autosweep/decode_5_mtc_16/attempt_1/prefill-0/prefill-0-gb200-rack1-09.log:648397`

## 4. 69 个失败 `req_id` 的 turn-band 映射

### 4.1 重要限制

这次无法精确映射到 `(conversation_id, turn_id)`，原因是：
- benchmark 结果文件只有聚合指标
- proxy/router 将原始 benchmark request id 改写成了：
  - `chatcmpl-___prefill_addr_<...>___decode_addr_<...>_<uuid>`
- 原始的 benchmark `p1d5-mtc16-*` request id 没有保存在这些失败日志中

因此，这里采用的是 turn-band 重建方法：
- 看某个失败请求第一次出现时，系统已经累计完成了多少请求
- 在 `32` 个 conversation 的前提下：
  - 完成数 `0-31` 归为 turn-band 1
  - 完成数 `32-63` 归为 turn-band 2
  - 完成数 `64-95` 归为 turn-band 3

### 4.2 turn-band 结果

| Turn-band | 失败请求数 |
| --- | ---: |
| 1 | 32 |
| 2 | 32 |
| 3 | 5 |

这是整份分析里最强的一条结果。

解释：
- 每个 conversation 的第一轮请求，看起来都在 Mooncake publish 路径上失败了。
- 每个 conversation 的第二轮请求，也都在 Mooncake publish 路径上失败了。
- 然后失败窗口继续截断了最早进入的 `5` 个第三轮请求，直到系统开始自愈。

### 4.3 按 decode 目标节点拆分

| Turn-band | `rack1-10` | `rack1-11` | `rack1-12` | `rack1-13` | `rack1-14` |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 7 | 7 | 6 | 6 | 6 |
| 2 | 7 | 6 | 7 | 6 | 6 |
| 3 | 1 | 1 | 1 | 1 | 1 |

解释：
- 失败请求在 5 个 decode 目标之间分布得非常均匀。
- 这和“某一台 decode 节点异常导致整个 benchmark 变差”的解释不一致。

## 5. 为什么这件事会直接拖坏 multi-turn 性能

这个 benchmark 本身就是为 prefix reuse 设计的：
- `multi-turn`
- `multi-turn-concurrency 16`
- `multi-turn-prefix-global-ratio 0.15`
- `multi-turn-prefix-conversation-ratio 0.75`
- `no_history_accumulation=true`

也就是说：
- benchmark 会反复发送固定长度、但有大比例可复用前缀的 prompt
- 如果前几轮的 KV publish 失败，下一轮就拿不到本该命中的 external prefix reuse

这和分轮性能表现是严格对得上的：
- `Turn 1` 不比邻居明显更差
- `Turn 2 / Turn 3` 最差
- `Turn 5+` 在写路径开始自愈后恢复

因此，最合理的解释是：

`Turn 1 的 cache publish 失败 -> Turn 2 失去本应命中的 reuse -> Turn 2 的 publish 也继续失败 -> Turn 3 继续受影响 -> 系统开始自愈 -> Turn 5+ 基本恢复`

## 6. Prefill 指标横向对比：Decode-4 vs Decode-5 vs Decode-6

### 6.1 failed-batch 指标

| Case | 出现 failed-metric 的秒数 | 首次失败 | 最后失败 | 最大 failed batches | 最大 failed keys |
| --- | ---: | --- | --- | ---: | ---: |
| `decode_4_mtc_14` | 0 | none | none | 0 | 0 |
| `decode_5_mtc_16` | 135 | `16:46:05` | `16:48:51` | 20 | 1761 |
| `decode_6_mtc_18` | 0 | none | none | 0 | 0 |

这三组对比非常干净：
- `decode_4` 和 `decode_6` 的 prefill metrics 全程没有出现 `failed_batches > 0`
- `decode_5` 出现了长时间连续的 failed-batch 窗口

`decode_5` 的代表性指标行：
- `bench_results/kimi_pd_p1d1to17_mooncake_nsys_autosweep/decode_5_mtc_16/attempt_1/prefill-0/prefill-0-gb200-rack1-09.log:38395`
- `bench_results/kimi_pd_p1d1to17_mooncake_nsys_autosweep/decode_5_mtc_16/attempt_1/prefill-0/prefill-0-gb200-rack1-09.log:52903`
- `bench_results/kimi_pd_p1d1to17_mooncake_nsys_autosweep/decode_5_mtc_16/attempt_1/prefill-0/prefill-0-gb200-rack1-09.log:648397`

### 6.2 external prefix cache hit-rate 里程碑

| Case | 首次 `>=10%` | 首次 `>=20%` | 首次 `>=50%` | 首次 `>=70%` | 最终 / 最大值 |
| --- | --- | --- | --- | --- | ---: |
| `decode_4_mtc_14` | `16:32:29 (13.3%)` | `16:32:53 (27.8%)` | `16:33:05 (51.2%)` | `16:33:34 (70.0%)` | `79.3%` |
| `decode_5_mtc_16` | `16:49:57 (13.4%)` | `16:50:02 (20.0%)` | `16:50:44 (50.1%)` | 从未达到 | `61.7%` |
| `decode_6_mtc_18` | `17:03:41 (16.0%)` | `17:03:44 (24.6%)` | `17:04:26 (52.4%)` | `17:05:04 (70.1%)` | `79.3%` |

解释：
- `decode_5` 不仅一开始更差
- 它恢复得也更慢，而且最终 external hit-rate 也明显低于 `decode_4/6`

关键行号：
- `decode_4` 达到 `79.3%`：
  - `bench_results/kimi_pd_p1d1to17_mooncake_nsys_autosweep/decode_4_mtc_14/attempt_1/prefill-0/prefill-0-gb200-rack1-09.log:5280`
- `decode_6` 达到 `79.3%`：
  - `bench_results/kimi_pd_p1d1to17_mooncake_nsys_autosweep/decode_6_mtc_18/attempt_1/prefill-0/prefill-0-gb200-rack1-01.log:5112`
- `decode_5` 初期仍为 `0.0%`：
  - `bench_results/kimi_pd_p1d1to17_mooncake_nsys_autosweep/decode_5_mtc_16/attempt_1/prefill-0/prefill-0-gb200-rack1-09.log:20662`
- `decode_5` 晚得多才爬到 `13.4%`、`20.0%`、`50.1%`：
  - `bench_results/kimi_pd_p1d1to17_mooncake_nsys_autosweep/decode_5_mtc_16/attempt_1/prefill-0/prefill-0-gb200-rack1-09.log:649504`
  - `bench_results/kimi_pd_p1d1to17_mooncake_nsys_autosweep/decode_5_mtc_16/attempt_1/prefill-0/prefill-0-gb200-rack1-09.log:649590`
  - `bench_results/kimi_pd_p1d1to17_mooncake_nsys_autosweep/decode_5_mtc_16/attempt_1/prefill-0/prefill-0-gb200-rack1-09.log:650358`
- `decode_5` 最终只有 `61.7%`：
  - `bench_results/kimi_pd_p1d1to17_mooncake_nsys_autosweep/decode_5_mtc_16/attempt_1/prefill-0/prefill-0-gb200-rack1-09.log:651071`

## 7. 自愈过程的形态

这次并不是“完全失败然后瞬间恢复”，而更像是逐步自愈：
- 早期：
  - 单请求 failed-key 比例很高
  - external prefix hit-rate 接近 `0%`
- 中期：
  - failures 仍持续出现
  - 但已有部分请求开始完成
  - failed-key 比例依然高
- 失败窗口尾部：
  - failures 变轻
  - hit-rate 开始爬升
- 失败窗口结束后：
  - failed-batch 指标消失
  - 请求完成速率显著提升
  - external hit-rate 继续爬升，但最终仍没有追上 `decode_4/6`

早期重灾样本：
- `88/113 keys failed`
  - `bench_results/kimi_pd_p1d1to17_mooncake_nsys_autosweep/decode_5_mtc_16/attempt_1/prefill-0/prefill-0-gb200-rack1-09.log:158220`
- `97/108 keys failed`
  - `bench_results/kimi_pd_p1d1to17_mooncake_nsys_autosweep/decode_5_mtc_16/attempt_1/prefill-0/prefill-0-gb200-rack1-09.log:158221`

尾部较轻样本：
- `1/6 keys failed`
  - `bench_results/kimi_pd_p1d1to17_mooncake_nsys_autosweep/decode_5_mtc_16/attempt_1/prefill-0/prefill-0-gb200-rack1-09.log:648306`
- `2/7 keys failed`
  - `bench_results/kimi_pd_p1d1to17_mooncake_nsys_autosweep/decode_5_mtc_16/attempt_1/prefill-0/prefill-0-gb200-rack1-09.log:648304`

这说明它更像是 publish 路径经历了一段较长的不稳定期，而不是简单的二值式开/关。

## 8. 这次异常不支持哪些解释

以下解释不符合当前证据：

1. `router` 调度失衡
- `router.log` 显示 5 个 decode 节点分配完全均匀
- 本地统计结果是 `320` 个请求全部完成，且每个 decode 节点都收到 `64` 个请求

2. 某一台 decode 节点坏掉
- 失败请求均匀覆盖全部 5 个 decode 目标

3. decode kernel / TPOT 回归
- `decode_5` 的 `mean TPOT` 仍接近相邻点
- 后续轮次也能回到相对正常的 steady range

4. 普通的负载扩展退化
- `decode_4` 和 `decode_6` 没有相同的 failed-batch 指标
- 所以不能简单归因为“decode 变多、负载变大一点自然就变差”

## 9. 最可能的根因

当前最合理的判断是：

主问题在 prefill 侧的 Mooncake publish 路径，具体更接近 metadata / segment 可见性或 segment-open readiness 问题。

为什么这是最匹配的解释：
- 最早出现的是 `metadata not found`
- 再往下是 `Failed to open segment`
- 然后是 `Transfer submission failed for key`
- 最后 Python 层看到的是 `batch_put failed`
- 同一时间段内 throughput 明显受抑制，external hit-rate 接近 0
- 失败计数消失后，throughput 和 hit-rate 开始恢复

最简洁的根因描述可以写成：

`decode_5_mtc_16` 在 prefill 侧经历了一段较长的 Mooncake publish 失败窗口，导致前两轮无法稳定建立 external KV reuse，因此造成了 Turn 2 / Turn 3 的显著延迟异常。

## 10. 建议的后续动作

1. 在可能的情况下，使用相同节点组合重跑一次 `decode_5_mtc_16`
- 用于区分它到底是一次瞬时 metadata / readiness 抖动，还是 `decode=5` 这一拓扑稳定复现的问题

2. 如果成本允许，下一次打开详细结果保存
- 建议加 `--save-detailed`
- 这样就可以恢复精确的 `(conversation_id, turn_id, request_id)` 对应关系

3. 下一次继续抓同样的 prefill 秒级指标
- 重点关注：
  - `mooncake_store_put_failed_batches`
  - `mooncake_store_put_failed_keys`
  - `External prefix cache hit rate`

4. 深挖这 18 个 segment endpoint 为什么会长期返回 metadata 404
- 这是本次 run 中离“根因”最近的直接外部症状

5. 如果需要，补做 `decode_4/5/6` 的对照复跑
- 用来确认 `79.3%` external hit-rate 是否就是这类 benchmark 的健康上限

## 最终结论

可以明确地说：这次 `decode_5_mtc_16` 的 Mooncake 报错与性能异常是强相关的。

而且比“可能有影响”更强一些，当前最好的结论是：
- 错误几乎完整覆盖了前两个 turn-band
- 这正是 benchmark 需要“先 publish、再 reuse external KV”的关键阶段
- 相邻点 `decode_4/6` 没有同样的 failed-batch 指标
- 报错结束后，吞吐和后续轮次明显恢复

因此，目前最准确的总结是：

`decode_5_mtc_16` 并不是因为 decode=5 天然扩展差，而是因为前两轮发生了 Mooncake publish / reuse 失效，导致这一个 sweep 点成为吞吐和 TTFT 的异常点。`

## 附录 A：错误链源码映射

错误是**自下而上**传播的。写 KV cache 时，Python 侧 `batch_put_from_multi_buffers` 的返回数组 `res[i] = -800`，每个 `-800` 都对应一段完整的 C++ 错误链，会产生 4 条不同文件的 ERROR 日志。

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

### A.1 逐层源码位置

| # | 错误消息 | 源文件 | 行号 | 函数 / 触发条件 |
|---|---|---|---|---|
| 1 | `metadata not found` | `Mooncake/mooncake-store/src/http_metadata_server.cpp` | 39 / 87 | C++ metadata server GET / DELETE handler，key 不在 `store_` map |
| 1b | `metadata not found` | `Mooncake/mooncake-wheel/mooncake/http_metadata_server.py` | 80 / 100 | Python metadata server GET / DELETE handler（配置用的是 HTTP） |
| 2 | _(无日志)_ | `Mooncake/mooncake-transfer-engine/src/transfer_engine_impl.cpp` | 434-441 | `TransferEngineImpl::openSegment()` — metadata get 失败 → 返回 `ERR_INVALID_ARGUMENT` |
| 2b | _(无日志)_ | `Mooncake/mooncake-transfer-engine/src/transfer_metadata.cpp` | 851-879 | `getSegmentDesc()` → `storage_plugin_->get()` 404 → 返回 `nullptr` |
| 3 | `Failed to open segment <endpoint>` | `Mooncake/mooncake-store/src/transfer_task.cpp` | 498 | `TransferSubmitter::submit_batch()` |
| 3b | `Failed to open segment <endpoint>` | `Mooncake/mooncake-store/src/transfer_task.cpp` | 535 | `submit_batch_get_offload_object()` |
| 3c | `Failed to open segment for endpoint='...'` | `Mooncake/mooncake-store/src/transfer_task.cpp` | 639 | `submitTransferEngineOperation()` |
| 4 | `Transfer submission failed for key <key>: <failure_context>` | `Mooncake/mooncake-store/src/client_service.cpp` | 1590 | `ClientService::SubmitTransfers()` — `submit()` 返回 falsy → `SetError(TRANSFER_FAIL)` |
| 4b | `Transfer submission failed for key <key>: <failure_context>` | `Mooncake/mooncake-store/src/client_service.cpp` | 1946 | 合并 upsert 路径 |
| 5 | `Operation for key ... failed: TRANSFER_FAIL` | `Mooncake/mooncake-store/src/client_service.cpp` | 1876 | finalize 阶段汇总；`toString()` 查表 `Mooncake/mooncake-store/src/types.cpp:38` |
| 6 | `batch_put failed: ...` | `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_worker.py` | 480 | vLLM Python 层 — `res[i] < 0` |

### A.2 关键洞察

- **日志计数几乎相等**：`decode_5_mtc_16` 日志中 `metadata not found / Failed to open segment / Transfer submission failed` 都在 106994 次量级，本质上是同一事件的 4 个投影。
- **根因在第 1 层**：`metadata not found` 是整条链的源头。可能原因：
  1. prefill worker 还没完成 segment PUT（时序问题 / startup race）
  2. metadata key 被误删或 TTL 过期
  3. 部分 segment endpoint 长期返回 404（本次 run 涉及 18 个这样的 endpoint）
- **Python 层信息受限**：Python 只能看到 `-800`，中间所有的 endpoint / replica / segment_name 信息都丢失了。要追溯源头必须看 C++ 侧日志或接入额外可观测性。

## 附录 B：Mooncake 的可观测性现状

### B.1 C++ 侧日志

Mooncake 用 `glog`（Google Logging）：
- `LOG(ERROR)` — 默认打开
- `LOG(WARNING)` — 默认打开
- `LOG(INFO)` — 默认打开
- `VLOG(1)` / `VLOG(2)` — 默认**关闭**，需要 `GLOG_v=1` 或 `GLOG_v=2` 环境变量

关键 verbose 信号（当前看不到）：
- `client_service.cpp:1595` — `VLOG(1) << "Successfully submitted N transfers for key ..."` — 需要 `GLOG_v=1`
- `client_service.cpp:1883` — `VLOG(1) << "Operation for key ... completed successfully"`
- `transfer_metadata.cpp` 里 segment descriptor 的详细 get/put 轨迹基本都是 `VLOG`

要在下一次 benchmark 中抓到更深信号，可在 prefill worker 的启动环境加：
```bash
GLOG_v=1   # or GLOG_v=2 for even more detail
```

注意：`VLOG(2)` 会产生极大日志量（每次 RDMA / metadata sync 都会打），真实 workload 下可能需要磁盘几十 GB，建议先用 `v=1` 试。

### B.2 C++ 侧 Metrics（Prometheus）

Mooncake 提供了两套 metrics 开关，本次配置都开了：

| 环境变量 | 作用 | 本次状态 |
|---|---|---|
| `MC_STORE_CLIENT_METRIC=1` | Mooncake store 客户端侧指标 | 已开启（见 `rendered_config.yaml:68`） |
| `MC_STORE_CLIENT_METRIC_INTERVAL=5` | 上报间隔（秒） | 5s |
| `MC_TE_METRIC=1` | TransferEngine 指标（RDMA 带宽/延迟/重试） | 已开启（见 `rendered_config.yaml:70`） |

问题：
- `MC_STORE_CLIENT_METRIC` 目前只提供**总量级别**指标（put / get / hit），不区分失败原因，不区分 endpoint，不区分 segment_name。
- `MC_TE_METRIC` 提供 RDMA 传输统计，但不会关联到具体 key 或 segment。

**对于本次 bug，现有 metrics 不够用。**

### B.3 Grafana 状态

参考之前 `prefill6` 的分析：
- Grafana 容器在跑
- Prometheus 容器**没在跑**（`http://localhost:9090` 不可达）
- `mooncake-overview.json` 只覆盖 master metrics（`master_*`），不覆盖 per-worker `TRANSFER_FAIL / batch_put failed` 等指标

要启用需要：
1. 启动 Prometheus 容器（抓 vLLM worker 的 `/metrics` endpoint）
2. 扩展 dashboard 加入：
   - `vllm:mooncake_store_put_failed_batches`
   - `vllm:mooncake_store_put_failed_keys`
   - `vllm:mooncake_store_put_transfer_fail_keys`（修复 `-1 vs -800` bug 后才会正确上报）

### B.4 推荐的下一步观测手段（按 ROI 排序）

1. **最低代价** — 在下一次复现时，prefill worker 启动环境加 `GLOG_v=1`，配合刚恢复的 Python 日志（req_id / tp_rank / failed_samples），可以把失败 key 和具体 segment endpoint 一一对应。
2. **中代价** — 启动 Prometheus，把 vLLM + Mooncake 的指标对齐看时间线。
3. **高代价（需改 Mooncake 源码）** — 在 `transfer_metadata.cpp:getSegmentDesc()` 里给"metadata 404 失败"加一条 `LOG(WARNING)`，包含 `segment_name` 和 HTTP 响应，就不需要去对日志里查哪个 key 对应哪个 endpoint。
4. **最高代价** — 给 metadata server 加 access log，记录每个 GET 请求的 key 以及命中 / 未命中，能直接区分"是 publish 没发生"还是"publish 发生了但 key 不匹配"。
