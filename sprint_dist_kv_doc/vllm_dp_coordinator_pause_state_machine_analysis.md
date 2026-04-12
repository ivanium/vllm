# vLLM 中 DP>1 的 DPCoordinator / pause-start-wait 状态机分析

基于当前工作区代码：

- `/home/aoshen/setup_new_cluster/vllm`
- 重点分析 `vllm/v1/engine/*`、`vllm/v1/core/sched/*`

---

## 1. 先说结论

当前 vLLM 在 `DP > 1` 时，实际上有两套彼此正交、但会互相叠加的控制机制：

1. **全局 DP wave 状态机**
   - 负责在多个 DP rank 之间维护一个全局的 `running / paused` 状态。
   - 核心对象是 `DPCoordinatorProc` 和 `DPEngineCoreProc`。
   - 这套机制只在 **MoE + DP>1** 时真正启用 wave coordination。

2. **本地 scheduler pause 状态机**
   - 负责单个 engine 进程内部的 `UNPAUSED / PAUSED_NEW / PAUSED_ALL`。
   - 核心对象是 `Scheduler.pause_state`、`EngineCoreProc.pause_scheduler()`、`resume_scheduler()`。
   - `pause_generation(mode="abort"|"wait"|"keep")` 走的是这套机制。

这两套状态机不是一回事：

- `START_DP_WAVE` / `wave_complete` 是 **全局 DP 运行态**。
- `pause_scheduler` / `resume_scheduler` 是 **本地调度态**。

实际运行时，**必须两边都允许，真实请求才会继续推进**。

---

## 2. DPCoordinator 到底什么时候存在

`DPCoordinator` 是否需要，由 `VllmConfig.needs_dp_coordinator` 决定：

- `vllm/config/vllm.py:462-482`
- `vllm/v1/engine/utils.py:968-980`

代码语义是：

1. **MoE + DP>1**
   - 即使是 external LB，也需要 coordinator。
   - 原因是 wave coordination 依赖 coordinator。

2. **非 MoE + DP>1**
   - 只有 internal/hybrid LB 时才需要 coordinator。
   - 此时 coordinator 主要做 stats 汇总，不做 wave coordination。

另外，`DPEngineCoreProc` 只给 **MoE** 用：

- `vllm/v1/engine/core.py:1071-1082`

非 MoE 的 DP rank 在 v1 里被当成“相互独立的 DP=1 engine”处理，不走 `DPEngineCoreProc` 那套全局 wave 逻辑。

---

## 3. ASCII 架构图

这一节只解决一件事：

- 把“前端进程 / coordinator 进程 / 多个 DP engine 进程”的关系先画清楚

### 3.1 internal / hybrid LB，且 MoE + DP>1

这是当前 **最值得重点理解** 的形态，因为：

- 前端会做 internal LB
- coordinator 既管 stats，也管 wave coordination
- engine 用的是 `DPEngineCoreProc`

```text
                                shared frontend / api server process

    +-----------------------------------------------------------------------------------+
    | AsyncLLM / DPLBAsyncMPClient                                                      |
    |                                                                                   |
    |  local cached state:                                                              |
    |    - current_wave                                                                 |
    |    - engines_running                                                              |
    |    - lb_engines[dp_rank] = [waiting, running]                                     |
    |                                                                                   |
    |  outbound:                                                                        |
    |    - ROUTER -------------- ADD / ABORT / UTILITY --------------------------+      |
    |    - PAIR ---------------- FIRST_REQ / SCALE_ELASTIC_EP ----------------+  |      |
    |                                                                         |  |      |
    |  inbound:                                                                         |
    |    - XSUB <------------- (counts, current_wave, engines_running) -------+  |      |
    +-----------------------------------------------------------------------------------+
                                                                              |  |
                                                                              v  v
                                     +--------------------------------------------------+
                                     |                 DPCoordinatorProc                 |
                                     |                                                  |
                                     |  global control state:                           |
                                     |    - current_wave                                |
                                     |    - engines_running                             |
                                     |    - engine[i].request_counts                    |
                                     |                                                  |
                                     |  inbound from frontend:                          |
                                     |    - FIRST_REQ                                   |
                                     |    - SCALE_ELASTIC_EP                            |
                                     |                                                  |
                                     |  inbound from engines:                           |
                                     |    - scheduler_stats                             |
                                     |    - wave_complete                               |
                                     |    - start_wave                                  |
                                     |                                                  |
                                     |  outbound:                                       |
                                     |    - publish stats/wave to frontend              |
                                     |    - broadcast START_DP_WAVE to engines          |
                                     +--------------------------+-----------------------+
                                                                ^
                                                                |
                                     EngineCoreOutputs          | START_DP_WAVE
                           (stats / wave_complete / start_wave) |
                                                                |
     +---------------------------------+   +---------------------------------+   +---------------------------------+
     |        engine proc dp=0         |   |        engine proc dp=1         |   |        engine proc dp=N-1       |
     |---------------------------------|   |---------------------------------|   |---------------------------------|
     | input_thread                    |   | input_thread                    |   | input_thread                    |
     |   ROUTER <- frontend            |   |   ROUTER <- frontend            |   |   ROUTER <- frontend            |
     |   XSUB   <- coordinator         |   |   XSUB   <- coordinator         |   |   XSUB   <- coordinator         |
     |          -> input_queue         |   |          -> input_queue         |   |          -> input_queue         |
     |                                 |   |                                 |   |                                 |
     | busy_loop: DPEngineCoreProc     |   | busy_loop: DPEngineCoreProc     |   | busy_loop: DPEngineCoreProc     |
     |   - Scheduler                   |   |   - Scheduler                   |   |   - Scheduler                   |
     |   - ModelExecutor               |   |   - ModelExecutor               |   |   - ModelExecutor               |
     |   - current_wave                |   |   - current_wave                |   |   - current_wave                |
     |   - engines_running             |   |   - engines_running             |   |   - engines_running             |
     |                                 |   |                                 |   |                                 |
     | output_thread                   |   | output_thread                   |   | output_thread                   |
     |   PUSH -> frontend outputs      |   |   PUSH -> frontend outputs      |   |   PUSH -> frontend outputs      |
     |   PUSH -> coordinator control   |   |   PUSH -> coordinator control   |   |   PUSH -> coordinator control   |
     +---------------------------------+   +---------------------------------+   +---------------------------------+
                       ^                                 ^                                         ^
                       |                                 |                                         |
                       +---------------------------------+-----------------------------------------+
                                         torch.distributed dp_group
                                      all_reduce(has_unfinished_dp)
```

这个图里最关键的是两条线：

1. **前端 <-> coordinator**
   - 走的是控制面缓存同步
   - 前端拿到 `current_wave/engines_running/lb_engines`

2. **engine <-> coordinator**
   - 走的是全局 wave 推进
   - `wave_complete` 和 `start_wave` 都从这里上报

### 3.2 external LB，且 MoE + DP>1

这个形态下，前端不做 shared internal LB，而是**每个 API server 只管自己本地的 DP rank**。

```text
     api server 0                    api server 1                    api server N-1
     +----------------------+        +----------------------+        +----------------------+
     | AsyncLLM             |        | AsyncLLM             |        | AsyncLLM             |
     | DPAsyncMPClient      |        | DPAsyncMPClient      |        | DPAsyncMPClient      |
     +----------+-----------+        +----------+-----------+        +----------+-----------+
                |                               |                               |
                | ROUTER / PUSH                 | ROUTER / PUSH                 | ROUTER / PUSH
                v                               v                               v
     +----------------------+        +----------------------+        +----------------------+
     | engine proc dp=0     |        | engine proc dp=1     |        | engine proc dp=N-1   |
     | DPEngineCoreProc     |        | DPEngineCoreProc     |        | DPEngineCoreProc     |
     +----------+-----------+        +----------+-----------+        +----------+-----------+
                \                               |                               /
                 \                              |                              /
                  +-----------------------------+-----------------------------+
                                                |
                                                v
                               +--------------------------------------+
                               |          DPCoordinatorProc           |
                               |  mainly for MoE wave coordination    |
                               |  stats publication may be absent      |
                               +--------------------------------------+
```

这里最容易误解的一点是：

- **coordinator 仍然存在**
- 但 `pause_generation()` 不再天然是“所有 API server 上的全局 pause”
- 它是否全局生效，要看是不是有人把 pause 指令 fanout 到所有前端

### 3.3 单个 engine 进程内部的结构

上面画的是进程间关系；单个 `engine proc dp=k` 内部，还可以再缩成下面这样看：

```text
engine process (dp=k)

    +------------------------------+
    | input_thread                 |
    |  - recv from frontend        |
    |  - recv START_DP_WAVE        |
    |  - preprocess request        |
    |  - push into input_queue     |
    +--------------+---------------+
                   |
                   v
    +------------------------------+
    | DPEngineCoreProc busy_loop   |
    |  - _process_input_queue()    |
    |  - Scheduler                 |
    |  - ModelExecutor             |
    |  - maybe dummy batch         |
    |  - all_reduce unfinished     |
    +--------------+---------------+
                   |
                   v
    +------------------------------+
    | output_thread                |
    |  - send normal outputs       |
    |  - send scheduler_stats      |
    |  - send wave_complete        |
    |  - send start_wave           |
    +------------------------------+
```

这也是为什么读源码时最好把 engine 进程分成三层去看：

1. `input_thread`
2. `busy_loop`
3. `output_thread`

否则很容易把“消息什么时候进 input_queue”和“状态什么时候在 busy_loop 里生效”混在一起。

---

## 4. 组件分工

### 3.1 前端侧：`DPAsyncMPClient` / `DPLBAsyncMPClient`

关键代码：

- `vllm/v1/engine/core_client.py:1109-1286`
- `vllm/v1/engine/core_client.py:1322-1362`

职责：

1. 缓存 coordinator 下发的三元组：
   - `current_wave`
   - `engines_running`
   - `lb_engines`（每个 engine 的 `[waiting, running]`）

2. 给每个新请求打上 `request.current_wave`
   - `vllm/v1/engine/core_client.py:1268-1279`
   - `vllm/v1/engine/__init__.py:87-90`

3. 在 engines 处于 paused 时，额外通过 `FIRST_REQ` 通知 coordinator 唤醒其它 DP rank

4. 在 internal LB 模式下，用 `lb_engines` 做 engine 选择
   - `vllm/v1/engine/core_client.py:1322-1350`

### 3.2 中央控制面：`DPCoordinatorProc`

关键代码：

- `vllm/v1/engine/coordinator.py:151-459`

职责：

1. 接收 engine 上报的 `scheduler_stats`
2. 维护全局：
   - `current_wave`
   - `engines_running`
3. 向前端广播：
   - 负载统计
   - 全局 wave / running 状态
4. 向 engine 广播：
   - `START_DP_WAVE`

### 3.3 engine 侧：`DPEngineCoreProc`

关键代码：

- `vllm/v1/engine/core.py:1579-1758`

职责：

1. 接收请求并写入本地 scheduler
2. 在本地 wave 结束后，通过 all-reduce 判断“全局是否还有 unfinished requests”
3. 在需要时上报：
   - `wave_complete`
   - `start_wave`
4. 在无真实请求但全局仍处于 running 时执行 dummy batch，驱动 EP / DP 对齐前进

### 3.4 本地调度面：`Scheduler.pause_state`

关键代码：

- `vllm/v1/core/sched/interface.py:22-33`
- `vllm/v1/core/sched/scheduler.py:368-370`
- `vllm/v1/core/sched/scheduler.py:1842-1858`

状态：

- `UNPAUSED`
- `PAUSED_NEW`
- `PAUSED_ALL`

语义：

- `PAUSED_NEW`：已有 running 请求继续跑，新请求不调度
- `PAUSED_ALL`：任何请求都不调度

---

## 5. 全局 DP wave 状态机

### 4.1 核心状态

Coordinator 维护：

- `current_wave`
- `engines_running`

初始化：

- `vllm/v1/engine/coordinator.py:203-205`
- 初始为 `current_wave = 0, engines_running = False`

可以把它看成一个二元状态机：

```text
State A: paused  = (wave = W, running = False)
State B: running = (wave = W, running = True)
```

### 4.2 paused -> running

有两条入口。

#### 入口 1：前端发现 paused 时送来新请求

路径：

1. 前端给请求打上 `request.current_wave = self.current_wave`
   - `core_client.py:1268-1272`
2. 如果本地缓存里 `engines_running == False`
   - 前端额外发 `FIRST_REQ`
   - `core_client.py:1275-1279`
3. stats task 把它转发给 coordinator
   - `core_client.py:1227-1236`
4. coordinator 在 `not engines_running` 时广播 `START_DP_WAVE`
   - `coordinator.py:348-366`

#### 入口 2：engine 收到 stale-wave 请求，自行上报 `start_wave`

路径：

1. engine 收到请求后，若 `request_wave != self.current_wave`
   - `core.py:1640-1654`
2. 如果它发现自己已经不在 running 状态，且 scheduler 没有本地 pause
   - 它会上报 `EngineCoreOutputs(start_wave=self.current_wave)`
3. coordinator 收到后广播 `START_DP_WAVE`
   - `coordinator.py:428-443`

这条路径就是代码里说的“race condition handling”。

### 4.3 running -> paused

路径：

1. 每个 DP rank 在 busy loop 中计算本地 `local_unfinished_reqs`
   - `core.py:1715`
2. 每 32 步做一次 `all_reduce(MAX)`
   - `core.py:1752-1758`
   - `config/parallel.py:576-584`
3. 如果全局没有 unfinished requests：
   - `self.engines_running = False`
   - rank0 向 coordinator 发送 `wave_complete=self.current_wave`
   - `core.py:1730-1745`
4. engine 本地再执行：
   - `self.current_wave += 1`
   - `self.step_counter = 0`
   - `core.py:1746-1748`
5. coordinator 收到 `wave_complete=W` 后，切到：
   - `current_wave = W + 1`
   - `engines_running = False`
   - `coordinator.py:414-427`

### 4.4 engine 收到 `START_DP_WAVE` 后怎么变

关键逻辑：

- `core.py:1668-1681`

语义：

1. 收到 `(new_wave, exclude_eng_index)`
2. 如果自己不是被排除的 engine，且 `new_wave >= self.current_wave`
3. 更新 `self.current_wave = new_wave`
4. 如果自己当前不是 running，就切到 running

这里的 `exclude_engine_index` 用来避免已经拿到请求的那个 engine 再被重复唤醒。

---

## 6. 本地 pause / wait / keep / resume 状态机

### 5.1 API 入口

用户可见入口：

- `vllm/v1/engine/async_llm.py:726-775`

其中：

- `pause_generation(mode="abort"|"wait"|"keep")`
- `resume_generation()`
- `is_paused()`

调用最终落到：

- `core_client.py:1039-1048`
- `core.py:1507-1546`

注意：

- `core.py` 里的注释把 `abort`/`keep` 的完成语义描述得更偏向“输出已发完”
- 但**实际实现**里，future 完成条件统一更接近：
  - `has_work() == False` 后触发 `_idle_state_callbacks`
  - 代码见 `core.py:1539-1545` 与 `core.py:1149-1153`

下面的分析都以**实现真实行为**为准，不以注释文字为准。

### 5.2 三种 pause 模式

#### `abort`

逻辑：

- `core.py:1531-1537`

行为：

1. 直接 `finish_requests(..., FINISHED_ABORTED)`
2. 发送 abort outputs
3. `pause_state = PAUSED_NEW`
4. 等 engine 进入 idle 后完成 future

#### `wait`

逻辑：

- `core.py:1507-1546`
- `scheduler.py:1848-1858`

行为：

1. `pause_state = PAUSED_NEW`
2. 新请求仍可进入队列，但不会被调度
3. 当前 running 的请求继续跑
4. 当 `len(running) == 0` 时，future 完成

注意这里的“wait 完成”不是“系统完全没有请求”，而是“当前 in-flight running 请求清空了”。

#### `keep`

逻辑：

- `core.py:1537-1546`
- `scheduler.py:368-370`
- `scheduler.py:1848-1850`

行为：

1. `pause_state = PAUSED_ALL`
2. 所有请求冻结，不再调度
3. `get_num_unfinished_requests()` 在 `PAUSED_ALL` 下直接返回 0
4. engine 会把自己视为“可进入 idle”
5. 之后 `resume_scheduler()` 再恢复

### 5.3 resume

逻辑：

- 普通 engine：`core.py:637-639`
- DP MoE engine：`core.py:1656-1666`

DP MoE 下多做了一件事：

1. `pause_state -> UNPAUSED`
2. 如果当前全局 `engines_running == False`
3. 且本地 scheduler 里还有 unfinished requests
4. 主动上报 `start_wave=self.current_wave`

也就是说：

- `resume_scheduler()` 只恢复本地调度权限
- 但 `DPEngineCoreProc.resume_scheduler()` 还会顺手尝试恢复全局 wave

---

## 7. 两套状态机如何叠加

实际运行中，至少要区分下面两层“暂停”：

### 6.1 全局波次暂停

来源：

- 所有 DP rank 全局都没有 unfinished requests

体现：

- `engines_running = False`
- coordinator / front-end / DPEngineCoreProc 的 `current_wave` 前进

### 6.2 本地调度暂停

来源：

- 显式调用 `pause_generation()`

体现：

- `scheduler.pause_state != UNPAUSED`

### 6.3 二者组合后的真实含义

可以粗略理解为：

```text
请求是否能真正推进
= 本地 scheduler 允许调度
  AND 全局 wave 已经 running

但 engine busy loop 本身是否还在转
= engines_running
  OR scheduler.has_requests()
  OR batch_queue 非空
```

代码位置：

- `core.py:1124-1130`

这解释了两个容易混淆的点：

1. `resume_generation()` 不等于立刻开始算
   - 如果全局 wave 还没被拉起，还需要 `start_wave`

2. `engines_running=False` 也不等于本地没有请求
   - `keep` / `wait` 下，本地队列里可能还冻结着请求

---

## 8. 代码里已经显式处理过的竞态

这部分是当前设计里“已经意识到并处理”的 race。

### 7.1 前端 current_wave 过期

处理手段：

- 每个请求都带 `request.current_wave`
  - `engine/__init__.py:87-90`
- engine 收到 stale wave 时可回报 `start_wave`
  - `core.py:1640-1654`
- coordinator 收到后广播 `START_DP_WAVE`
  - `coordinator.py:428-443`

### 7.2 前端在 paused 期间发第一条请求

处理手段：

- `FIRST_REQ`
  - `core_client.py:1276-1279`
- coordinator 在 paused 态下收到该通知后统一启动本轮 wave
  - `coordinator.py:355-366`

### 7.3 stats 乱序

处理手段：

- coordinator 对 `(stats_wave, stats_step)` 做单调性检查
  - `coordinator.py:383-406`

这些机制说明当前实现不是“完全裸奔”，而是用 `wave` 编号当成了一个轻量 epoch，去修补控制面异步消息导致的乱序。

---

## 9. 从竞态角度看，当前实现最薄弱的地方

下面按“我认为最值得警惕”的顺序来写。

### 8.1 `pause_generation()` 的完成时刻，不等于“所有输出都已经被前端处理完”

关键代码：

- `core.py:1526-1545`
- `async_llm.py:760-767`

现象：

1. engine 侧的 pause future，是在 `_idle_state_callbacks` 被触发时完成的
2. 这个时点只说明 engine 进入 idle
3. 但 output_queue -> output thread -> client output task -> output_processor 这条链路还可能没完全排空
4. 因此 `AsyncLLM.pause_generation()` 里专门补了一个：
   - `await asyncio.sleep(0.02)`

这说明当前实现自己都承认：

- “pause 完成” 和 “用户视角的最终输出都已稳定可见” 之间有窗口
- 现在的修补方式是经验型 sleep，而不是严格 barrier

这是一类典型的**控制面先完成、数据面后收尾**的竞态。

### 8.2 `wait_for_requests_to_drain()` 判断的是 `engines_running`，不是“系统真的空了”

关键代码：

- `async_llm.py:950-964`
- `core_client.py:1248-1251`
- `scheduler.py:1848-1858`

问题在于：

1. `wait_for_requests_to_drain()` 只看 `dp_engines_running()`
2. `dp_engines_running()` 本质上是 coordinator 广播过来的全局 running 标志
3. 它不等价于：
   - 前端输出队列已排空
   - 本地 waiting 队列为空
   - `keep`/`wait` 模式下被冻结的请求已经消失

特别是在：

- `PAUSED_NEW`：`get_num_unfinished_requests()` 只统计 `running`
- `PAUSED_ALL`：直接返回 0

于是“drain 完成”的真实语义更接近：

- **全局 wave 已经停了**

而不是：

- **系统里已经完全没有请求状态**

这对“扩缩容”“热更新”这类高层控制动作是有歧义的。

### 8.3 全局 finished 检测是每 32 步才同步一次，天生有延迟窗口

关键代码：

- `core.py:1752-1758`

逻辑是：

```text
step_counter += 1
只有 step_counter % 32 == 0 时，才做一次 all_reduce(MAX)
否则直接返回 True
```

这意味着：

1. 即使全局已经没有 unfinished request
2. 系统也可能继续多跑最多 31 个“认为自己还在 running”的周期
3. 在这些周期里：
   - coordinator 还不知道 wave 已结束
   - 前端 `engines_running` 还没变
   - `wait_for_requests_to_drain()` 会继续等
   - engine 可能继续做 dummy batch

这不是 correctness bug，但它明显扩大了竞态窗口和控制面滞后。

### 8.4 internal LB 依赖 coordinator 的 100ms 级别 stats，多个 front-end 会并发抢同一 engine

关键代码：

- `coordinator.py:263-285`
- `core_client.py:1329-1345`

现状：

1. coordinator 默认 100ms 一次 stats 发布
2. `DPLBAsyncMPClient` 用上一次收到的 `lb_engines` 选 rank
3. 为了减轻 stale stats 问题，它只在**本 client 内部**做了一个本地 waiting 计数递增：
   - `current_counts[eng_index][0] += self.client_count`

薄弱点：

- 这个“乐观递增”不是全局共享的
- 多个 API server 同时看见同一份旧 stats 时，仍可能一起把请求打到同一台 engine

因此这是一个**负载均衡一致性弱**的问题，不一定错，但会造成瞬时倾斜。

### 8.5 `run_engine_stats_update_task()` 对 `FIRST_REQ` 的处理依赖 poll 返回顺序，写法比较脆

关键代码：

- `core_client.py:1178-1245`

尤其是这个条件：

```python
if (
    not self.engines_running
    and len(events) == 2
    or (events[0][0] == first_req_rcv_socket)
):
```

它的特点是：

1. 行为部分依赖 `events[0]` 是谁
2. `FIRST_REQ` 和 stats 在两个不同 socket 上
3. 一次循环里只显式处理一条 `FIRST_REQ`
4. stats 是“drain 到最新”，但 `FIRST_REQ` 不是对应对称处理

这不是说它一定会错，而是说：

- 它对 event interleaving 很敏感
- 可读性差
- 后续一旦混入更多控制消息（例如扩缩容、额外 barrier），很容易出边界 bug

这是一个明显的**控制面逻辑脆弱点**。

### 8.6 Elastic EP scale-up 时，新 engine 的 `current_wave` 初始化与现网 wave 不同步，靠后续流量补救

关键代码：

- `coordinator.py:321-333`

这里注释已经直接承认：

- 新启动 engine 的 `current_wave = 0`
- 但已有 engine 可能已经在更高 wave

当前实现没有在 scale-up 完成时做一次显式 “wave bootstrap/snapshot sync”，而是依赖：

1. 后续请求携带的 `request.current_wave`
2. 或 coordinator 后续再发 `START_DP_WAVE`

来把新 engine 拉回正确 epoch。

这说明 Elastic EP 的控制面仍然是**最终一致**，不是**强一致**。

### 8.7 pause/resume 是否“全局生效”，取决于 client 拓扑，不是 coordinator 保证的

关键代码：

- `core_client.py:1039-1048`
- `core_client.py:1352-1362`

现状：

1. `DPLBAsyncMPClient.call_utility_async()` 会 `gather` 到所有本 client 管理的 engines
2. 但 `DPAsyncMPClient` 不会；它只调自己管理的那个 engine

含义：

- **internal LB** 下，`pause_generation()` 基本等价于“对本 front-end 管的所有 engines 做广播”
- **external LB** 下，`pause_generation()` 不是 coordinator 级别的全局 pause，它只是本 front-end / 本 rank 的本地 pause

所以“pause 是不是全局动作”这件事，在当前实现里不是由 coordinator 统一提供语义，而是由部署拓扑决定。

这对运维层是一个很容易踩坑的语义裂缝。

### 8.8 `has_work()` 同时承担“真的有工作”和“虽然冻结了但还要维持 loop”两种语义，导致高层很难把 idle 理解准确

关键代码：

- `core.py:1124-1130`
- `scheduler.py:1848-1858`

`has_work()` 依赖：

- `engines_running`
- `scheduler.has_requests()`
- `batch_queue`

但 `scheduler.has_requests()` 又被 pause state 改写了语义：

- `PAUSED_NEW`：waiting 中的新请求不算 unfinished
- `PAUSED_ALL`：直接 0

于是 “idle” 可能表示三种完全不同的事情：

1. 真的没有请求了
2. 还有请求，但都被冻结了
3. 还有请求，但全局 wave 已停

这会让上层控制逻辑非常容易误判。

---

## 10. 我认为当前设计里最重要的边界判断

### 9.1 `START_DP_WAVE` 不是 `resume_scheduler`

它们分别控制：

- `START_DP_WAVE`：DP 全局 running/paused
- `resume_scheduler`：本地 scheduler 是否允许调度

任何分析如果把这两个概念混在一起，都会很快看错代码。

### 9.2 `wave` 是轻量 epoch，不是严格 barrier token

它能处理常见 stale message，但它不是严格的 distributed transaction id。

因此当前实现本质上是：

- **靠单调 wave 编号 + 最终一致消息传播修补竞态**

而不是：

- **靠强 barrier/ack 机制消灭竞态**

### 9.3 `wait`/`keep` 的“完成”语义是工程语义，不是数学语义

具体来说：

- `wait` 完成：running 清空了
- `keep` 完成：engine 可以停到 idle 了

都不等于“系统内部已经完全无请求状态”。

---

## 11. 如果要加固，我会优先做什么

### 10.1 用显式 output-drain barrier 取代 `sleep(0.02)`

目标：

- pause future 完成前，确保 engine output 已被 client output task 和 output_processor 完整消费

### 10.2 把“wave idle”和“request state empty”拆成两个独立信号

建议至少区分：

- `global_wave_running`
- `local_scheduler_empty`
- `frontend_output_drained`
- `paused_with_frozen_requests`

### 10.3 把 32-step finished sync 做成可配置，或者在尾声阶段降采样间隔

现在的固定 32 步太硬，会直接拉大收敛延迟。

### 10.4 给 elastic scale-up 增加显式 wave snapshot 同步

不要让新 engine 以 `current_wave=0` 进入系统后再靠流量修复。

### 10.5 重写 `run_engine_stats_update_task()` 的事件处理

目标：

- 不依赖 `events[0]`
- `FIRST_REQ` / stats / scale 信号分别 drain
- 明确每种消息的优先级

### 10.6 在 external LB 下提供真正的“全局 pause/resume”控制面

否则现在的 pause 语义很容易被误以为是 cluster-wide，而实际上它只是 local。

---

## 12. 最后的总结

当前 vLLM 在 DP>1、尤其是 **MoE + DP>1** 下，实际是靠下面这组机制协同工作：

1. `DPCoordinatorProc`
   - 维护全局 `current_wave / engines_running`
   - 广播 `START_DP_WAVE`
   - 汇总 LB stats

2. `DPEngineCoreProc`
   - 在本地接受请求
   - 用 all-reduce 判断全局是否还有 unfinished requests
   - 在 stale-wave 场景下回推 `start_wave`

3. `Scheduler.pause_state`
   - 提供 `UNPAUSED / PAUSED_NEW / PAUSED_ALL`
   - 实现 `abort / wait / keep / resume`

它的优点是：

- 结构不算复杂
- 对常见 stale-wave 竞态已经有补丁路径
- 不需要重型分布式 barrier 协议

它的弱点是：

- 很多地方是**最终一致**而不是**强一致**
- 高层“pause / drain / idle”的语义混杂
- 至少有一个地方明确靠 `sleep(20ms)` 补 race
- Elastic EP 和 multi-frontend 下的控制面都还有明显脆弱点

如果后面要继续深挖，我建议下一步直接画两张图：

1. `FIRST_REQ / start_wave / wave_complete / START_DP_WAVE` 的时序图
2. `pause_generation(wait|keep) -> idle callback -> output processor` 的时序图

这两张图会比继续堆源码更快暴露剩余竞态。
