# 问题 2：RDMA 握手与端点状态故障

日期：2026-04-16

## 问题卡片

| 字段 | 值 |
| --- | --- |
| 问题编号 | 问题 2 |
| 简称 | RDMA 握手 / 端点状态 |
| 当前状态 | `Strong hypothesis` |
| 受影响运行 | 主要是 `prefill_6_mtc_18`；在 `prefill3/4/5` 与 `p4_d4` 中也能看到同一故障族的较弱信号 |
| 一行结论 | 当前最强判断是：Mooncake 在启动窗口内的 RDMA 建连或端点状态管理出现故障，先表现为 `handshake timeout / packet mismatch / inactive endpoint`，之后再放大成 `60s transfer timeout -> TRANSFER_FAIL -> batch_put failed`。 |
| 相关问题 | [问题 1：元数据 / 过期描述符可见性故障](decode5_mtc16_mooncake_correlation_analysis_20260415_cn.md)；[问题 3：处理中对象重复 put 与 `OBJECT_NOT_FOUND` 放大链](prefill3_decode5_mooncake_deep_dive_20260415_cn.md)；[问题 4：可观测性、错误归因与诊断盲区](batch_put_failed_global_analysis_20260415_cn.md) |

## 1. 问题定义

这个问题指的是如下故障族：
- Mooncake 命中了底层建连错误，例如 `handshake timeout`、`packet mismatch` 和 `mark it inactive`
- 某些 endpoint 随后进入坏掉、inactive 或 stale 的连接状态
- 后续写入仍然会命中这些 endpoint，并在 `WaitForTransfers()` 中阻塞，直到 60 秒硬超时
- Mooncake 返回 `TRANSFER_FAIL(-800)`，Python 侧再把这些结果聚合成 `batch_put failed`

这个问题不包括：
- `metadata not found / Failed to open segment`，那属于 [问题 1](decode5_mtc16_mooncake_correlation_analysis_20260415_cn.md)
- 下游的 `OBJECT_NOT_FOUND(-704)` 放大链，属于 [问题 3](prefill3_decode5_mooncake_deep_dive_20260415_cn.md)
- 日志与指标的盲区，属于 [问题 4](batch_put_failed_global_analysis_20260415_cn.md)

## 2. 当前状态

- 证据强度：`Strong hypothesis`
- 优先级：高
- 当前结论的单一事实来源：
  - `prefill_6_mtc_18` 最合理的解释是“启动期的 RDMA 握手 / 端点状态问题”，而不是一个独立的 Python `batch_put` bug。

已经稳定的事实：
- `handshake timeout`、`packet mismatch` 和 `inactive endpoint` 都集中在启动窗口
- 后续的 `60s transfer timeout` 发生在这些启动失败之后
- 这个 run 中所有 Python 侧 `batch_put failed` warning 都携带 `-800`
- 当前观察到的模式不符合 offload 压力，也不符合问题 1 的 metadata-404 故障族

## 3. 受影响范围

主运行：
- `bench_results/pd_kimi_nsys_prefill6_light/prefill_6_mtc_18`

当前最明显的热点主机：
- `192.168.0.104` 占了 `11/27` 个 inactive-endpoint 事件

次级相关运行：
- `prefill_3_mtc_12`
- `prefill_4_mtc_14`
- `prefill_5_mtc_16`
- `p4_d4_mtc_14`

这些 run 里只出现了较弱的相关信号，例如 re-establish / same-peer-QP reuse，但没有像 `prefill6` 那样出现同样集中、同样完整可见的问题链。

## 4. 主要症状与日志指纹

这里最稳定的底层指纹是：

```text
handshake timeout
  -> received packet mismatch
  -> mark it inactive
  -> Re-establish connection / reuse existing connection
  -> Failed to complete transfers after 60 seconds
  -> Transfer failed for key: TRANSFER_FAIL
  -> batch_put failed
```

代表性关键词：
- `handshake timeout`
- `received packet mismatch`
- `mark it inactive`
- `Re-establish connection`
- `Received same peer QP numbers, reusing connection`
- `Failed to complete transfers after 60 seconds`
- `Transfer failed for key`

## 5. 当前最强的机制链

当前最强解释是：

1. 某些 RDMA endpoint 在启动期间进入了异常握手路径。
2. 这个异常路径表现为 `handshake timeout`、`packet mismatch`、连接复用 / re-establish 竞争，以及 endpoint 被标记为 inactive。
3. 后续 store 写入仍然会命中这些坏掉或陈旧的 endpoint 状态。
4. transfer completion 一直等不到，于是等待逻辑在 60 秒处超时。
5. Mooncake 把这些 key 标成 `TRANSFER_FAIL(-800)`。
6. vLLM 再把逐 key 结果聚合后记录成 `batch_put failed`。

哪些是事实，哪些是推断：
- 第 1、2、4、5、6 步都有日志和源码直接支持。
- 第 3 步里“坏掉 / stale 的 endpoint state”这一解释，仍然是当前最强但尚未完全证实的推断。

## 6. 关键证据

### 6.1 按日志文件统计的握手失败

| 日志文件 | `handshake timeout` 数量 | 代表性位置 |
| --- | ---: | --- |
| `prefill-1-gb200-rack1-04.log` | 17 | `:1340` |
| `prefill-0-gb200-rack1-03.log` | 5 | `:1286` |
| `prefill-3-gb200-rack1-08.log` | 3 | `:1912` |
| `prefill-4-gb200-rack1-10.log` | 1 | `:1741` |
| `prefill-5-gb200-rack1-11.log` | 1 | `:1583` |

这已经说明：
- 问题并不是平均分布在所有 prefill worker 上的。

### 6.2 错误计数汇总

| 类别 | 数量 | 时间 / 含义 |
| --- | ---: | --- |
| `handshake timeout` | 27 | 全部发生在 `2026-04-14 09:15:48-09:15:49` |
| `packet mismatch` | 14 | 握手描述符或路径不匹配信号 |
| `inactive endpoint` | 27 | endpoint 建立失败后，worker 将其标记为 inactive |
| `transfer timeout` | 9 | `Failed to complete transfers after 60 seconds` |
| `Transfer failed for key` | 9 | 9 个 key 命中 `TRANSFER_FAIL` |
| `batch_put failed` | 6 | Python batch warning 覆盖了这 9 个 key |

这个顺序很重要：
- 先出现握手失败
- 后出现 transfer timeout
- 最后才是 Python warning

### 6.3 热点主机分布

| Host | Count |
| --- | ---: |
| `192.168.0.104` | 11 |
| `192.168.0.107` | 4 |
| `192.168.0.108` | 4 |
| `192.168.0.103` | 4 |
| `192.168.0.110` | 3 |
| `192.168.0.111` | 1 |

这是最强的信号之一，说明这并不只是均匀分布的启动噪声。

### 6.4 所有 60 秒超时 batch 与失败 key

| Worker log | Timeout 行 | Batch ID | Failed key 行 | Failed key 后缀 |
| --- | --- | ---: | --- | --- |
| `prefill-0-gb200-rack1-03.log` | `:2163` | `80299849924864` | `:2164` | `16a380262664...` |
| `prefill-0-gb200-rack1-03.log` | `:2218` | `97737619846512` | `:2219` | `ac5bfcd52c34...` |
| `prefill-1-gb200-rack1-04.log` | `:2338` | `75355604427936` | `:2339` | `e4e2a74e4315...` |
| `prefill-1-gb200-rack1-04.log` | `:2340` | `82376130695408` | `:2341` | `9e80ff557a31...` |
| `prefill-1-gb200-rack1-04.log` | `:3100` | `75355605178320` | `:3101` | `e2fdd361cadd...` |
| `prefill-1-gb200-rack1-04.log` | `:3109` | `82376131834192` | `:3110` | `78548e37244b...` |
| `prefill-2-gb200-rack1-07.log` | `:2585` | `77278877289136` | `:2586` | `67fa0fb5de24...` |
| `prefill-3-gb200-rack1-08.log` | `:2941` | `103858685124736` | `:2942` | `3c35f017697c...` |
| `prefill-3-gb200-rack1-08.log` | `:3700` | `103858686344336` | `:3701` | `2800bcca4333...` |

这张表保留了 timeout-batch 与 failed-key 的精确映射，对后续整理 endpoint-pair 时间线很重要。

### 6.5 这里的 `TRANSFER_FAIL` 是严格意义上的 60 秒硬超时

相关源码：
- `third_partys/Mooncake/mooncake-store/src/transfer_task.cpp:341`
- `third_partys/Mooncake/mooncake-store/src/client_service.cpp:1681`

含义：
- 这不是一个随机的 Python 异常
- 它是 Mooncake transfer wait 超时的确定性结果

### 6.6 这里的 `batch_put failed` 是症状，不是根因

所有 Python 侧 warning 都带有：
- `codes={-800}`

含义：
- Python warning 只是做了聚合，没有补充根因信息
- 根因信号早已出现在更底层的 handshake 与 transfer 日志中

## 7. 当前最强的状态机解释

下一层更有用的拆解不是“网络坏了”，而是下面三个状态机候选。

### 7.1 候选 1：simultaneous-open 或 stale connection reuse

支持证据：
- 日志里出现了 `Received same peer QP numbers, reusing connection`
- 日志里出现了 `Re-establish connection`
- 问题高度集中在启动期，而不是在整个时间轴上均匀分布

当前判断：
- 最强候选

### 7.2 候选 2：多 NIC 路径不匹配或 endpoint advertisement 不匹配

支持证据：
- `packet mismatch` 是直接的握手阶段信号
- 某些日志里出现了空的或不一致的 `peer.local_nic_path` / `peer.peer_nic_path`

当前判断：
- 第二强候选

### 7.3 候选 3：真实 RDMA 参数不匹配

可能涉及的维度：
- `MTU`
- `GID`
- `LID`
- `QP`

当前判断：
- 不能排除
- 但目前弱于前两个状态机解释

## 8. 这不是什么

当前证据不支持以下解释：

1. 这不是 [问题 1](decode5_mtc16_mooncake_correlation_analysis_20260415_cn.md) 的 metadata-404 故障族。
2. 这不是一个独立的 Python `batch_put failed` bug。
3. 这不是 offload pressure，也不是 `NO_AVAILABLE_HANDLE(-200)`。
4. 这不是 decode 侧负载波动或 benchmark 侧噪声。

## 9. 与其他问题的关系

- 与 [问题 1](decode5_mtc16_mooncake_correlation_analysis_20260415_cn.md) 的关系：
  - 两者最终都可能落成 `TRANSFER_FAIL(-800)` 和 `batch_put failed`
  - 但问题 1 是由 metadata / segment descriptor 查找失败驱动的
  - 问题 2 是由握手 / endpoint state 故障驱动的

- 与 [问题 3](prefill3_decode5_mooncake_deep_dive_20260415_cn.md) 的关系：
  - 问题 3 解释了为什么某些 `TRANSFER_FAIL` 链之后会进一步放大成 `OBJECT_NOT_FOUND(-704)`
  - 但这不是 `prefill6` 这个 run 的主要矛盾

- 与 [问题 4](batch_put_failed_global_analysis_20260415_cn.md) 的关系：
  - 问题 4 里的可观测性工作，正是用来把 simultaneous-open、path mismatch 和真实 RDMA 参数不匹配干净地区分开的

## 10. 仍未证明的部分

主要还不确定的是：

1. 这里的主导因素到底更偏 simultaneous-open / stale reuse，还是更偏 path mismatch？
2. `192.168.0.104` 到底是主机局部热点、NIC 局部热点，还是仅仅是当前状态机 bug 最容易暴露出来的位置？
3. 某个 endpoint 被标成 inactive 之后，为什么后续写入仍然会走到一个会卡 60 秒的路径？

所以当前状态是：
- 我们已经知道根因不是 Python warning 本身
- 但还没把状态机级故障收敛到一个单一实现 bug

## 11. 修复方向

首先，继续强化状态机证据：
- 在握手失败日志里打印 `endpoint / NIC path / MTU / GID / LID / QP`
- 在 transfer 失败日志里打印 `replica_index / strategy / transport_endpoint`
- 围绕 `192.168.0.104` 这个热点构建 endpoint-pair 时间线

然后再逐步收窄逻辑修复方向，例如：
- stale connection reuse 的触发条件
- simultaneous-open 冲突处理
- path mismatch 后 endpoint reset / quarantine 的行为

同时把下面这张源码映射表继续贴在这个问题旁边：

| 信号 | 源码位置 | 角色 |
| --- | --- | --- |
| `packet mismatch` | `rdma_endpoint.cpp:282` | 握手描述符或 endpoint-path 不匹配 |
| `reuse existing connection` | `rdma_endpoint.cpp:254` | simultaneous-open / reuse 分支 |
| `mark it inactive` | `worker_pool.cpp:245` | endpoint 建立失败，worker 将其标记 inactive |
| `timeout after 60 seconds` | `transfer_task.cpp:341` | 硬性 transfer timeout |
| `Transfer failed for key` | `client_service.cpp:1681` | 逐 key transfer failure 上报 |

## 12. 验证计划

要认为这个问题已经修好，以下条件都应成立：

1. 启动窗口中不再出现成批的 `handshake timeout`。
2. `packet mismatch` 与 `mark it inactive` 基本消失。
3. `Failed to complete transfers after 60 seconds` 事件消失。
4. `TRANSFER_FAIL(-800)` 与 Python `batch_put failed` warning 同步大幅下降或一起消失。
5. 在相同 benchmark 形状下，`prefill_6_mtc_18` 不再进入同样的启动后写路径故障族。

## 13. 稳定结论措辞

请以下面这段话作为这个问题的单一结论表述：

> `prefill_6_mtc_18` 最合理的解释是启动期的 RDMA 握手 / 端点状态故障。底层问题链从 `handshake timeout / packet mismatch / inactive endpoint` 开始，随后放大成 `60s transfer timeout -> TRANSFER_FAIL(-800) -> batch_put failed`。在这个问题里，`batch_put failed` 只是下游聚合症状，不是根因。
