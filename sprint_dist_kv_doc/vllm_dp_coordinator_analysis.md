# vLLM DP Coordinator 深度分析

> 基于 vLLM v1 架构，分析 Data Parallel (DP > 1) 场景下的 DPCoordinator 组件、Wave 状态机、以及竞态条件。

---

## 1. 整体架构：进程关系图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        API Server (Front-End)                          │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  DPLBAsyncMPClient                                               │   │
│  │  ┌─────────────────────┐  ┌──────────────────────────────────┐  │   │
│  │  │ engines_running 状态 │  │ lb_engines: [[wait,run], ...]   │  │   │
│  │  │ current_wave        │  │ 负载均衡: score = wait*4 + run   │  │   │
│  │  └─────────────────────┘  └──────────────────────────────────┘  │   │
│  └───────┬───────────────────────────┬──────────────────────────────┘   │
│          │ ADD request               │ "FIRST_REQ" (当 engines_running  │
│          │ (ROUTER→DEALER)           │  == False 时发送)                │
│          │                           │ (PAIR socket, inproc)            │
└──────────┼───────────────────────────┼──────────────────────────────────┘
           │                           │
           │                     ┌─────▼──────────┐
           │                     │  Stats Update   │
           │                     │  Task (asyncio) │
           │                     └─────┬──────────┘
           │                           │ XSUB → XPUB
           │                           │
┌──────────┼───────────────────────────┼──────────────────────────────────┐
│          │              ┌────────────▼────────────┐                     │
│          │              │    DPCoordinatorProc     │    独立进程         │
│          │              │    (单线程 ZMQ poller)   │                     │
│          │              │                         │                     │
│          │              │ publish_front ◄──────── XPUB: 发布 stats     │
│          │              │                    给所有 Front-End            │
│          │              │                                               │
│          │              │ publish_back ────────► XPUB: 广播             │
│          │              │                    START_DP_WAVE 给所有 Engine │
│          │              │                                               │
│          │              │ output_back ◄──────── PULL: 收集 Engine 的    │
│          │              │                    stats / wave_complete /     │
│          │              │                    start_wave                  │
│          │              └────────────────────────┘                      │
│          │                     │  broadcast                              │
└──────────┼─────────────────────┼────────────────────────────────────────┘
           │                     │
     ┌─────┼─────────────────────┼─────────────────────────────┐
     │     │                     │                              │
     │     ▼                     ▼                              │
     │ ┌────────────────────────────────────────────────┐      │
     │ │         DPEngineCoreProc (dp_rank=0)           │      │
     │ │                                                 │      │
     │ │  ┌───────────┐  ┌───────────┐  ┌────────────┐ │      │
     │ │  │ Input     │  │ Busy Loop │  │ Output     │ │      │
     │ │  │ Thread    │──│ (主线程)   │──│ Thread     │ │      │
     │ │  │ ZMQ recv  │  │           │  │ ZMQ send   │ │      │
     │ │  └───────────┘  └─────┬─────┘  └────────────┘ │      │
     │ │                       │ all-reduce (每32步)     │      │
     │ └───────────────────────┼────────────────────────┘      │
     │                         │                                │
     │                    NCCL / Gloo                           │
     │                    all-reduce                            │
     │                         │                                │
     │ ┌───────────────────────┼────────────────────────┐      │
     │ │         DPEngineCoreProc (dp_rank=1)           │      │
     │ │                                                 │      │
     │ │  ┌───────────┐  ┌───────────┐  ┌────────────┐ │      │
     │ │  │ Input     │  │ Busy Loop │  │ Output     │ │      │
     │ │  │ Thread    │──│ (主线程)   │──│ Thread     │ │      │
     │ │  │ ZMQ recv  │  │           │  │ ZMQ send   │ │      │
     │ │  └───────────┘  └───────────┘  └────────────┘ │      │
     │ └────────────────────────────────────────────────┘      │
     │                         ...                              │
     │ ┌────────────────────────────────────────────────┐      │
     │ │         DPEngineCoreProc (dp_rank=N-1)         │      │
     │ └────────────────────────────────────────────────┘      │
     └─────────────────────────────────────────────────────────┘
```

**关键点：**

| 组件 | 进程模型 | 线程模型 |
|------|---------|---------|
| API Server (Front-End) | 独立进程 | asyncio 事件循环 |
| DPCoordinatorProc | 独立 daemon 进程 | 单线程 ZMQ poller |
| DPEngineCoreProc × N | 每个 DP rank 一个独立进程 | 3线程：Input Thread + Busy Loop + Output Thread |

---

## 2. Wave 状态机

### 2.1 核心概念

MoE (Mixture of Experts) 模型在 DP 模式下要求**所有 DP rank 同步执行 forward pass**（因为 Expert Parallelism 需要 all-to-all 通信）。Wave 机制解决的核心问题是：**当没有请求时，如何让所有 Engine 同步地进入/退出空闲状态**。

### 2.2 状态定义

```
┌─────────────────────────────────────────────────────────────┐
│                      全局 Wave 状态                          │
│                                                              │
│  变量:                                                       │
│    engines_running: bool    — 所有 Engine 是否在运行          │
│    current_wave: int        — 当前 wave 编号（单调递增）      │
│                                                              │
│  三处维护 (各自独立的副本):                                    │
│    1. DPCoordinatorProc.process_input_socket() 局部变量       │
│    2. DPEngineCoreProc.engines_running / current_wave        │
│    3. DPAsyncMPClient.engines_running / current_wave         │
└─────────────────────────────────────────────────────────────┘
```

### 2.3 状态转换图

```
                    ┌──────────────────────────────────────────┐
                    │                                          │
     新请求到达      │            ┌──────────────────┐          │
    ─────────────►  │            │                  │          │
                    ▼            ▼                  │          │
            ┌──────────────┐          ┌────────────────────┐  │
            │              │          │                    │  │
            │    PAUSED    │          │     RUNNING        │  │
            │              │──────►   │                    │  │
            │ engines_     │  触发:   │ engines_running    │  │
            │ running=F    │ START_   │ = True             │  │
            │              │ DP_WAVE  │                    │  │
            │ wave=N       │ 广播     │ 每步执行:          │  │
            │              │          │  model forward     │  │
            └──────────────┘          │  (含 dummy batch)  │  │
                    ▲                 │                    │  │
                    │                 │ 每32步:            │  │
                    │                 │  all-reduce 检查   │  │
                    │                 │  全局 unfinished   │  │
                    │                 └─────────┬──────────┘  │
                    │                           │              │
                    │      all-reduce 返回      │              │
                    │      全局无未完成请求       │              │
                    │ ◄──────────────────────── ┘              │
                    │                                          │
                    │  动作:                                    │
                    │    engines_running = False                │
                    │    current_wave++                         │
                    │    rank 0 发送 wave_complete              │
                    │    step_counter 重置为 0                  │
                    │                                          │
                    └──────────────────────────────────────────┘
```

### 2.4 详细转换流程

#### 转换 1: PAUSED → RUNNING（新请求触发）

```
时序图:

  Front-End            Coordinator          Engine[0]         Engine[1]
     │                     │                    │                 │
     │  1. add_request     │                    │                 │
     │  (engines_running   │                    │                 │
     │   == False)         │                    │                 │
     │                     │                    │                 │
     │  2. "FIRST_REQ"────►│                    │                 │
     │     (via PAIR→XPUB) │                    │                 │
     │                     │                    │                 │
     │     同时发送 ADD ────┼────────────────►  接收 request       │
     │     给选中的Engine   │                    │                 │
     │                     │ 3. START_DP_WAVE   │                 │
     │                     │   (wave, exclude=  │                 │
     │                     │    选中的Engine)    │                 │
     │                     │────────────────────┼────────►        │
     │                     │                    │          收到    │
     │                     │                    │          START   │
     │                     │                    │          _DP_    │
     │                     │                    │          WAVE    │
     │                     │                    │                 │
     │  4. stats update ◄──│                    │  engines_running │
     │  (wave, running=T)  │                    │  = True          │
     │                     │                    │                 │
```

**关键代码路径** (`core_client.py:1268-1279`):
```python
async def add_request_async(self, request):
    request.current_wave = self.current_wave
    chosen_engine = self.get_core_engine_for_request(request)
    self._send_input(EngineCoreRequestType.ADD, request, chosen_engine)
    if not self.engines_running:
        # 通知 Coordinator 唤醒其他 Engine
        req_msg = msgspec.msgpack.encode(("FIRST_REQ", chosen_engine))
        await self.first_req_send_socket.send(req_msg)
```

**注意**: `exclude_eng_index` 的作用是避免给已经收到请求的 Engine 重复发送 START_DP_WAVE。

#### 转换 2: RUNNING → PAUSED（全局完成）

```
  Engine[0]                  Engine[1]               Coordinator
     │                          │                        │
     │  local_unfinished=F      │  local_unfinished=F    │
     │                          │                        │
     │  ◄── all-reduce (MAX) ──►│                        │
     │      result = False       │                        │
     │                          │                        │
     │  engines_running=False   │  engines_running=False │
     │  current_wave++          │  current_wave++        │
     │                          │                        │
     │  (rank 0 only)           │                        │
     │  wave_complete ──────────┼───────────────────►    │
     │                          │                  更新状态:
     │                          │                  engines_running=F
     │                          │                  current_wave=wave+1
     │                          │                        │
     │                          │               广播给 Front-End
     │                          │                        │
```

**关键代码路径** (`core.py:1726-1748`):
```python
self.engines_running = self._has_global_unfinished_reqs(local_unfinished_reqs)
if not self.engines_running:
    if self.dp_rank == 0 or not self.has_coordinator:
        self.output_queue.put_nowait(
            (-1, EngineCoreOutputs(wave_complete=self.current_wave))
        )
    self.current_wave += 1
    self.step_counter = 0
```

---

## 3. 同步机制一览

```
┌─────────────────────────────────────────────────────────────────────┐
│                         同步原语总览                                 │
├─────────────────────┬──────────────┬────────────────────────────────┤
│ 原语                 │ 类型         │ 用途                           │
├─────────────────────┼──────────────┼────────────────────────────────┤
│ torch.distributed   │ 集体通信     │ Engine间全局 unfinished 检测    │
│ .all_reduce(MAX)    │ (NCCL/Gloo)  │ 语义: 任一 rank 有未完成 → True │
├─────────────────────┼──────────────┼────────────────────────────────┤
│ input_queue         │ Queue        │ Input Thread → Busy Loop       │
│                     │ (线程安全)    │ 请求传递                       │
├─────────────────────┼──────────────┼────────────────────────────────┤
│ output_queue        │ Queue        │ Busy Loop → Output Thread      │
│                     │ (线程安全)    │ 输出 + 控制信号传递             │
├─────────────────────┼──────────────┼────────────────────────────────┤
│ ZMQ XPUB/XSUB      │ Pub-Sub      │ Coordinator → Engines 广播     │
│                     │ (进程间)      │ START_DP_WAVE                  │
├─────────────────────┼──────────────┼────────────────────────────────┤
│ ZMQ PUSH/PULL       │ 管道         │ Engines → Coordinator 上报     │
│                     │ (进程间)      │ stats / wave_complete          │
├─────────────────────┼──────────────┼────────────────────────────────┤
│ ZMQ ROUTER/DEALER   │ Req-Rep      │ Front-End → Engine 请求分发    │
│                     │ (进程间)      │                                │
├─────────────────────┼──────────────┼────────────────────────────────┤
│ threading.Event     │ 事件         │ 启动握手同步                    │
│                     │ (线程间)      │                                │
├─────────────────────┼──────────────┼────────────────────────────────┤
│ multiprocessing.Pipe│ 管道         │ Coordinator → 父进程            │
│                     │ (进程间)      │ 传递 ZMQ 地址                  │
└─────────────────────┴──────────────┴────────────────────────────────┘
```

---

## 4. Engine 内部线程模型

```
┌─────────────────────────────────────────────────────────────────┐
│                    DPEngineCoreProc (每个 DP rank)               │
│                                                                  │
│  ┌──────────────────┐    input_queue    ┌────────────────────┐  │
│  │   Input Thread   │ ───────────────► │    Busy Loop       │  │
│  │                  │   (Queue, FIFO)   │    (主线程)         │  │
│  │ • ZMQ DEALER     │                   │                    │  │
│  │   recv (Front)   │                   │ 1. _process_       │  │
│  │ • ZMQ XSUB       │                   │    input_queue()   │  │
│  │   recv (Coord)   │                   │ 2. _process_       │  │
│  │                  │                   │    engine_step()   │  │
│  │ 解码消息 →       │                   │ 3. all-reduce      │  │
│  │ put(type, data)  │                   │    (每32步)        │  │
│  └──────────────────┘                   │ 4. 检查 running    │  │
│                                          │    状态             │  │
│                                          └──────────┬─────────┘  │
│                                                      │            │
│                                           output_queue│            │
│                                            (Queue)    │            │
│                                                      ▼            │
│                                          ┌────────────────────┐  │
│                                          │  Output Thread     │  │
│                                          │                    │  │
│                                          │ • client_index=-1  │  │
│                                          │   → ZMQ PUSH to   │  │
│                                          │     Coordinator    │  │
│                                          │ • client_index>=0  │  │
│                                          │   → ZMQ PUSH to   │  │
│                                          │     Front-End      │  │
│                                          └────────────────────┘  │
│                                                                  │
│  关键: GIL 释放                                                   │
│  ZMQ socket I/O 和 NCCL 通信都会释放 GIL，                        │
│  因此三个线程可以真正并行执行 I/O 操作。                            │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. 竞态条件分析

### 5.1 已处理的竞态：Stale Wave Request

**场景描述:**

```
时间线:
  t0: wave 0 正在运行
  t1: wave 0 完成，Coordinator 设置 current_wave=1, engines_running=F
  t2: Coordinator 发布 (wave=1, running=F) 给 Front-End
  t3: Front-End 收到更新，设置 current_wave=1
  t4: 新请求到达 Front-End，设置 request.current_wave=1，发给 Engine[0]
  t5: 但是！另一个 Front-End 还没收到更新（网络延迟），current_wave 还是 0
  t6: 它发送的请求带着 wave=0 给 Engine[1]
```

**Engine[1] 收到 wave=0 的请求，但自己已经在 wave=1：**

```python
# core.py:1640-1654
def add_request(self, request, request_wave=0):
    super().add_request(request, request_wave)
    if self.has_coordinator and request_wave != self.current_wave:
        if request_wave > self.current_wave:
            # 未来的 wave → 更新自己
            self.current_wave = request_wave
        elif (not self.engines_running
              and self.scheduler.pause_state == PauseState.UNPAUSED):
            # 过期的 wave + 当前空闲 → 需要唤醒所有人
            self.engines_running = True
            self.output_queue.put_nowait(
                (-1, EngineCoreOutputs(start_wave=self.current_wave))
            )
```

**Coordinator 端处理** (`coordinator.py:428-443`):
```python
elif (wave := outputs.start_wave) is not None and (
    wave > current_wave
    or (wave == current_wave and not engines_running)
):
    # 广播 START_DP_WAVE，排除发送者
    self._send_start_wave(publish_back, wave, eng_index)
```

**结论**: 已通过 `start_wave` 信号 + Coordinator 广播机制处理。

---

### 5.2 潜在风险 1：三方状态不一致窗口

**问题**: `engines_running` 和 `current_wave` 在三个地方独立维护：

```
Front-End ─── Coordinator ─── Engine[0..N]
   各自有                各自有           各自有
   engines_running      engines_running   engines_running
   current_wave         current_wave      current_wave
```

**不一致窗口示例:**

```
时间  Front-End          Coordinator        Engine[0]
────  ─────────────────  ─────────────────  ──────────────────
t0    running=T, w=0     running=T, w=0     running=T, w=0
t1    running=T, w=0     running=T, w=0     all-reduce→False
                                             running=F, w=1
                                             发送 wave_complete(0)
t2    running=T, w=0     收到 wave_complete
                         running=F, w=1
                         发布给 Front-End
t3    收到更新
      running=F, w=1
```

**风险**: 在 t1-t3 之间 (~数ms到100ms)，Front-End 认为还在运行，可能发送带 `wave=0` 的请求。这正是 5.1 中已处理的竞态。

**但更细微的问题是**: Front-End 在 `add_request_async` 中：
```python
# core_client.py:1276
if not self.engines_running:
    req_msg = msgspec.msgpack.encode(("FIRST_REQ", chosen_engine))
    await self.first_req_send_socket.send(req_msg)
```
`engines_running` 的检查和 `FIRST_REQ` 的发送**不是原子的**。在 asyncio 单线程模型下不会出问题（不会被抢占），但 `self.engines_running` 由两个 asyncio task 修改：
1. `add_request_async` 中设为 `True`
2. `_process_output` 中根据 `wave_complete` 设为 `False`
3. `run_engine_stats_update_task` 中根据 Coordinator 推送更新

由于这三者都在同一个 asyncio 事件循环中，**不存在真正的竞态**。但如果将来改为多线程或多协程并发处理，则需要注意。

---

### 5.3 潜在风险 2：all-reduce 32步节流的副作用

```python
# core.py:1752-1758
def _has_global_unfinished_reqs(self, local_unfinished):
    self.step_counter += 1
    if self.step_counter % 32 != 0:
        return True   # ← 不检查，假设还在运行
    return ParallelConfig.has_unfinished_dp(self.dp_group, local_unfinished)
```

**设计意图**: all-reduce 有开销，每步都做会严重影响性能。32步检查一次是权衡。

**风险场景:**

```
  Engine[0]                          Engine[1]
     │                                  │
     │  step 1: 最后一个请求完成         │  step 1: 还有请求
     │  local_unfinished = False        │  local_unfinished = True
     │                                  │
     │  step 2-31:                      │  step 5: 请求完成
     │  _has_global → True (跳过检查)   │  local_unfinished = False
     │  执行 dummy batch × 30 次！       │  step 6-31: dummy batch × 26 次
     │                                  │
     │  step 32: all-reduce             │  step 32: all-reduce
     │  全局 False → PAUSED             │  全局 False → PAUSED
```

**影响**:
- **GPU 浪费**: 最多浪费 31 步的 dummy batch（空跑 forward pass）
- **延迟**: 从全局完成到检测到完成，最多延迟 31 步（通常每步 ~10-50ms → 最多 ~0.3-1.5s 延迟）
- **不会导致正确性问题**，只是效率问题

**更严重的情况**: 如果一个 Engine 在第 1 步就完成了，但另一个 Engine 有很多请求，完成请求的 Engine 会持续执行 dummy batch 直到下一次 all-reduce 确认另一个 Engine 也完成。这是设计如此，因为 MoE 的 all-to-all 通信要求所有 rank 必须同步参与。

---

### 5.4 潜在风险 3：Coordinator 单点故障 + ZMQ 消息丢失

**问题**: Coordinator 是单进程单线程，如果挂了：

```
  Front-End                    Coordinator (DEAD)         Engines
     │                              ╳                       │
     │  发送 "FIRST_REQ" ──────►   (丢失)                   │
     │                                                      │
     │  Engine 收到 ADD request                             │
     │  但 engines_running=F                                │
     │  其他 Engine 不知道要开始                              │
     │                                                      │
     │                              结果:                    │
     │                              Engine[i] 独自运行       │
     │                              其他 Engine 卡在 PAUSED  │
     │                              all-reduce 永远不返回    │
     │                              (因为一个 rank 在 PAUSED │
     │                               不参与 all-reduce)      │
```

**缓解**: Coordinator 进程设为 daemon=True，如果父进程死了，Coordinator 也会退出。但 Coordinator 自身崩溃没有恢复机制。

**影响**: 集群级故障，需要重启整个服务。

---

### 5.5 潜在风险 4：PAUSED 状态下的 all-reduce 死锁

**最关键的竞态**:

当所有 Engine 处于 PAUSED 状态时，它们在 busy loop 的顶部检查：

```python
# core.py:1716-1719
if not executed:
    if not local_unfinished_reqs and not self.engines_running:
        continue  # ← 跳过整个循环体，不执行 all-reduce
    self.execute_dummy_batch()  # ← 只在 engines_running=True 时执行
```

**正常情况**: 所有 Engine 同步进入 PAUSED，都跳过 all-reduce，等待 START_DP_WAVE。

**危险场景**: 如果一个 Engine 收到请求但其他 Engine 还没收到 START_DP_WAVE：

```
  Engine[0]                    Engine[1]
     │                            │
     │  收到 ADD request          │
     │  (直接来自 Front-End)      │
     │  engines_running = True    │  engines_running = False
     │                            │
     │  执行 step()               │  continue (跳过)
     │  执行 dummy_batch()        │
     │  all-reduce... 等待        │  不参与 all-reduce
     │       ↓                    │       ↓
     │    HANG!                   │    不知道要参与
```

**实际防护**: 这个场景被 START_DP_WAVE 机制防护：
1. Front-End 发送 "FIRST_REQ" 给 Coordinator
2. Coordinator 广播 START_DP_WAVE 给所有 Engine（排除目标 Engine）
3. 所有 Engine 同时设置 `engines_running = True`

**但如果 START_DP_WAVE 消息到达晚于 Engine[0] 开始执行 step()?**

关键在于时序：Engine[0] 收到 ADD 后，需要在 `_process_input_queue()` 中处理它，然后在下一轮循环才执行 step()。而在这段时间内，Coordinator 的广播应该已经到达 Engine[1]。

**但这不是保证的** —— 取决于 ZMQ 消息延迟。实际代码中，Engine 收到请求后会设置 `engines_running = True`，然后在 busy loop 中执行 step 和 all-reduce。如果其他 Engine 还没收到 START_DP_WAVE，all-reduce 会阻塞。

**实际缓解**: 这个场景在实践中很少发生，因为：
1. ZMQ 在同一主机上的 IPC 延迟通常 < 1ms
2. Engine 的 step() 执行需要 10-50ms
3. 时间窗口非常短

但在**跨节点 DP 部署**中（使用 TCP 而非 IPC），这个风险更大。

---

### 5.6 潜在风险 5：负载均衡统计的延迟

```python
# coordinator.py:266
wait_for = self.stats_update_interval_ms if stats_changed else 5000
```

**问题**: Stats 更新有最小 100ms 间隔 + 50ms 聚合等待。Front-End 的负载均衡决策基于 100ms 前的数据。

```
  Front-End[A]           Front-End[B]          Engine[0]     Engine[1]
     │                      │                     │              │
     │  stats: E0=0, E1=0   │  stats: E0=0, E1=0 │              │
     │                      │                     │              │
     │  选 E0 ──────────────┼─────────────────►   │              │
     │                      │  选 E0 ──────────►  │              │
     │                      │  (同时选中同一个！)   │              │
     │                      │                     │              │
     │        100ms 后 stats 更新...               │              │
     │  stats: E0=2, E1=0   │                     │              │
```

**缓解措施**:
- Front-End 在本地立即增加计数: `current_counts[eng_index][0] += self.client_count`
- 但多个 Front-End 之间无法感知对方的本地增量

**影响**: 突发请求时可能短暂负载不均，但随着 stats 更新会自动纠正。

---

## 6. Pause/Resume 机制（Scheduler 级别）

这是区别于 DP Wave 的另一层暂停机制，主要用于模型权重更新等运维操作。

```
┌─────────────────────────────────────────────────────────┐
│                  Scheduler PauseState                    │
│                                                          │
│  UNPAUSED ◄───── resume_scheduler() ────── PAUSED_NEW  │
│     │                                          ▲         │
│     │  pause_scheduler(mode="wait")            │         │
│     │  pause_scheduler(mode="abort")           │         │
│     └──────────────────────────────────►       │         │
│                                                │         │
│  UNPAUSED ◄───── resume_scheduler() ────── PAUSED_ALL  │
│     │                                          ▲         │
│     │  pause_scheduler(mode="keep")            │         │
│     └──────────────────────────────────►       │         │
│                                                          │
│  语义:                                                   │
│  • UNPAUSED:   正常调度 waiting + running 请求            │
│  • PAUSED_NEW: 不调度 waiting 中的新请求，                │
│                running 中的请求继续执行到完成              │
│  • PAUSED_ALL: 完全停止调度（token_budget = 0）           │
│                running 中的请求也冻结                     │
└─────────────────────────────────────────────────────────┘
```

**DP 场景下的 resume 特殊处理** (`core.py:1656-1666`):

```python
def resume_scheduler(self):
    super().resume_scheduler()  # 设置 PauseState.UNPAUSED
    if (self.has_coordinator
        and not self.engines_running
        and self.scheduler.has_unfinished_requests()):
        # resume 后如果还有未完成请求，需要唤醒其他 DP Engine
        self.output_queue.put_nowait(
            (-1, EngineCoreOutputs(start_wave=self.current_wave))
        )
```

---

## 7. 关键代码索引

| 组件 | 文件 | 行号 |
|------|------|------|
| DPCoordinator (启动 + 地址管理) | `vllm/v1/engine/coordinator.py` | 23-143 |
| DPCoordinatorProc (核心逻辑) | `vllm/v1/engine/coordinator.py` | 151-465 |
| DPEngineCoreProc | `vllm/v1/engine/core.py` | 1579-1873 |
| run_busy_loop (DP版) | `vllm/v1/engine/core.py` | 1696-1750 |
| _has_global_unfinished_reqs | `vllm/v1/engine/core.py` | 1752-1758 |
| add_request (DP竞态处理) | `vllm/v1/engine/core.py` | 1640-1654 |
| resume_scheduler (DP版) | `vllm/v1/engine/core.py` | 1656-1666 |
| _handle_client_request (START_DP_WAVE) | `vllm/v1/engine/core.py` | 1668-1681 |
| pause_scheduler (多进程版) | `vllm/v1/engine/core.py` | 1507-1548 |
| DPAsyncMPClient (Front-End) | `vllm/v1/engine/core_client.py` | ~1090-1280 |
| DPLBAsyncMPClient (负载均衡) | `vllm/v1/engine/core_client.py` | 1289-1350 |
| has_unfinished_dp (all-reduce) | `vllm/config/parallel.py` | 576-584 |
| coordinate_batch_across_dp | `vllm/v1/worker/dp_utils.py` | 165-229 |
| PauseState 定义 | `vllm/v1/core/sched/interface.py` | 22-33 |

---

## 8. 总结：风险矩阵

| 风险 | 严重性 | 概率 | 已有防护 | 建议 |
|------|--------|------|----------|------|
| Stale wave request | 中 | 高 | start_wave 机制 | 已充分处理 |
| 三方状态不一致窗口 | 低 | 高 | asyncio 单线程 + start_wave | 多线程化时需重新评估 |
| 32步节流浪费 GPU | 低 | 中 | 设计如此 | 可调参数化 |
| Coordinator 单点故障 | 高 | 低 | daemon 进程 | 缺乏故障恢复机制 |
| PAUSED→RUNNING all-reduce 死锁 | 高 | 极低 | START_DP_WAVE 时序保证 | 跨节点部署需增加超时 |
| 负载均衡统计延迟 | 低 | 高 | 本地计数预增 | 多 Front-End 间无协调 |
