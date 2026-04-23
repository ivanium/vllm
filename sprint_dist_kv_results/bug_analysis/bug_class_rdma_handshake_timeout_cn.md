# Bug 类别 2：RDMA 握手 / QP 状态机失败（对外表现为 60s 超时）

- **合并日期**：2026-04-22
- **本地 Mooncake HEAD**：`be75ca0`（2026-04-18, "client_service: quorum write on WaitForTransfers + richer failure logging"）
- **合并自**：
  - `/home/aoshen/.claude/plans/sparkling-crafting-neumann.md`（60 秒超时根因分析 · 详细版 · 含 RDMA 基础 / 架构图 / 流程图 / 场景 A/B 区分）—— **作为本文骨架**
  - `prefill6_mooncake_error_analysis_20260415.md` + `prefill6_mooncake_error_analysis_20260415_cn.md`（prefill_6_mtc_18 的 27 个握手超时事件主证据）
  - `autosweep_20260417_135116_bug_census_cn.md`（跨 sweep 统计 + ETIMEDOUT 新机制 + `client_service.cpp` 两个放大 bug）
  - `mooncake_upstream_issues_prs_20260419_cn.md` 家族 2 章节（12 个 issue/PR + 本地已合入状态）
- **证据权重**：**Strong hypothesis → Confirmed**（多 sweep 点复现 + 源码 grep-verified + 底层 kernel 机制 ETIMEDOUT 在 `rdma_endpoint.cpp:646` 首次定位）

### 在三层模型中的位置：Layer 1（Connection Layer）

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Layer 3:  OBJECT LAYER (业务数据：KV-cache value)         → 类别 3      │
│  错误    → OBJECT_NOT_FOUND(-704) / batch_put_failed 聚合               │
├─────────────────────────────────────────────────────────────────────────┤
│  Layer 2:  SEGMENT LAYER (peer 身份地址本)               → 类别 1        │
│  错误    → HTTP 404 "metadata not found"                                │
├─────────────────────────────────────────────────────────────────────────┤
│  Layer 1:  CONNECTION LAYER (RDMA QP 物理握手)           ← 本文档        │
│           "peer QP 还没就绪 / 网络通路没建起来 / RNIC / 握手风暴"        │
│  错误    → Failed to modify QP to RTR / packet mismatch                 │
│            Connection timed out [110] (ETIMEDOUT)                       │
│  归属    → Kernel RDMA stack + Mooncake RdmaEndpoint 状态机              │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 1. 摘要

**最直观的一句话**：**RDMA 握手没走到 RTS → peer 的 QP 永远没准备好收数据 → ibv_post_send 从未成功 → Completion Queue 永远没有完成事件 → 外层 `wait_for(60s)` 必走满 → `TRANSFER_FAIL(-800)` → Python 看到 `batch_put failed, elapsed≈61s`**。

"60 秒超时" **不是 bug 本身**，而是握手失败后最外层的症状表现。真正的 bug 在 **QP 状态机 INIT → RTR 这一步**。

**三种独立根因场景**（本类**必须**先做这个区分，否则诊断会跑偏）：

| 维度 | **场景 A · Bench 起跑握手风暴** | **场景 B · 运行期 peer 持续失联** | **场景 C · Passive re-establish 丢在途 WR** |
|---|---|---|---|
| **发生时间** | bench 起跑后 [0, 120s] 内集中 | bench 启动 N 分钟后才首次出现 | 任意时刻 peer 触发 re-handshake |
| **持续性** | 风暴散去后自愈 | 每 60s 循环，直到 peer 恢复 | 受波及的在途 batch 都 60s 超时完后自愈 |
| **涉及 peer** | 多 peer 同时出错 | 单一 peer 反复出现 | 单一 peer，一次性事件 |
| **底层错误** | `packet mismatch` | `Connection timed out [110]` / ETIMEDOUT | **无握手错误**；`Outstanding work requests found, CQ will not be generated` |
| **故障比例** | 首轮几乎所有 batch 中招 | 只有路由到坏 peer 的 key 中招（eg 15/32） | 仅风暴瞬间的在途 batch 中招（eg 6 个） |
| **本机角色** | 客户端侧受害 | 客户端侧无异常，**对端才是真实故障点** | 本机 QP 被对端要求重建；在途 WR 静默丢失 |
| **自愈** | 自然自愈 | 不自愈 | 在途 batch 超时完后自愈；新 batch 恢复正常 |
| **修法方向** | 客户端 ramp-up / 显式 warmup / fail-fast | peer 健康检查 + circuit breaker | reset 时 markFailed 在途 slices；或开启 `slice_timeout` |

**本地 bench 命中情况**：

- **prefill_6_mtc_18 主 run**：27 次 `handshake timeout` 全部集中在 `2026-04-14 09:15:48-09:15:49`（启动窗口内 1 秒），9 个 key 升级为 `TRANSFER_FAIL`，6 条 Python `batch_put failed`。热点主机 `192.168.0.104` 占 `11/27` 事件。属**场景 A**。
- **P1to17D1 autosweep (20260417_135116)**：8 份 log 里 5 份 "握手失败 → 自愈成功"（Group A），3 份 "握手失败 → 升级为 60s timeout"（Group B）。首次在 `rdma_endpoint.cpp:646` 抓到**底层机制** `Failed to modify QP to RTR: Connection timed out [110]`，改写了原 prefill6 分析里"simultaneous-open 是最强候选"的判断——**更可能是启动惊群时 peer QP 还没就绪，kernel 等不到握手包超时**。6 个 sweep 点（P=3/4/5/6/8）都出现了握手失败，节点越多越严重（P=3/4 零次，P=8 十五次）。混合场景 A + 场景 B。
- **用户提供 log（prefill-2-gb200-rack1-13.log 08:11:25）**：bench 启动 07:56:32，首错 08:06:25（T+10min），每 60s 一次持续到 log 结束，全程仅一次 `mark it inactive` 指向同一 peer `192.168.0.113:13581@rocep195s0`，错误是 `Connection timed out [110]`，15/32 keys 中招——**纯场景 B**。

**修复状态一句话**：家族 2 在 2026-Q1 上游**已密集合入 5 条根因补丁**（#1560 / #1705 / #1733 / #1803 / #1762），本地 `be75ca0` 均已包含。但**场景 B 的 fail-fast / 自愈机制**和 **`client_service.cpp` 的两个写路径放大 bug**（§3.8）都**尚未有上游 PR**，需要本地改动。

---

## 2. 症状链（日志指纹）

### 2.1 场景 A 日志指纹（握手风暴）

```
handshake timeout                              (rdma_endpoint.cpp:283)
  -> received packet mismatch                  (rdma_endpoint.cpp:281-291)
  -> mark it inactive                          (worker_pool.cpp:244-248)
  -> Re-establish connection / reuse           (rdma_endpoint.cpp:254)
  -> Received same peer QP numbers, reusing    (自愈信号)
  -> Failed to complete transfers after 60 seconds  (transfer_task.cpp:317 若未自愈)
  -> Transfer failed for key: TRANSFER_FAIL    (client_service.cpp:1611)
  -> batch_put failed (elapsed≈61s)            (mooncake_store_worker.py:480)
```

特征：多 peer 同时出错，bench 起跑后短时间爆发，大部分自愈。

### 2.2 场景 B 日志指纹（peer 持续失联）

```
[Handshake] Failed to modify QP to RTR,        (rdma_endpoint.cpp:646)
  check mtu, gid, peer lid, peer qp num:
  Connection timed out [110]                   ← kernel ETIMEDOUT
  -> Successfully reset the endpoint           (rdma_endpoint.cpp:433)
  -> mark it inactive                          (worker_pool.cpp:245)
  -> (60 秒后)
  -> Failed to complete transfers after 60 seconds
  -> Transfer failed for key: TRANSFER_FAIL
  -> batch_put failed (elapsed=61s / 120s / ... = N×60s)
  -> (下一批请求到达)
  -> 新建 endpoint → 再次 ETIMEDOUT（peer 还是挂的）
  -> 循环直到 peer 恢复或 client 停发
```

特征：**单一** peer 反复出现在日志里（同一 `host:port@rocepXsY` 反复），`mark inactive` 次数少但每次都吃 60s，客户端侧无本地握手失败也可能命中（作为"受害方"向坏 peer 发数据）。

### 2.3 场景 C 日志指纹（Passive re-establish 丢在途 WR）

```
rdma_endpoint.cpp:325  Re-establish connection: EndPoint: local X, peer Y  (passive 由对端触发)
rdma_endpoint.cpp:406  Outstanding work requests found, CQ will not be generated
                       (在 disconnectUnlocked 里 line 406-407；:124 是 deconstruct 的同款日志，
                        本类大多是 :406 这路)
rdma_endpoint.cpp:433  Successfully reset the endpoint (triggered by: re-establishing connection (passive))
                       ↓ 注意：这条 log 只表示 resetConnection 成功；之后的 doSetupConnection
                         成功路径**无日志**，所以看不到"重建完成"的正面确认
(之后每 60s 一次 batch 超时，持续 N 分钟直到在途 batch 耗尽)
transfer_task.cpp:353  Failed to complete transfers after 60 seconds for batch X
                       (polling 分支，非 event-driven 的 :317；详见 §3.11)
```

**区分签名**（关键）：场景 C 日志里**不存在** `packet mismatch`、`mark it inactive`、`Failed to modify QP to RTR`、`Connection timed out [110]`。这是和场景 A/B 最硬的差异——**0 个握手错误签名 + 1 个 passive re-establish + 若干 Outstanding WR 警告 + 之后的 60s 超时**。

特征：对端 QP 换了（进程抖动 / 内部重建 / SIEVE eviction 等原因），主动过来要求本机 re-handshake；本机乖乖配合 reset + 重建到 RTS；但 reset 瞬间在 QP 上的在途 WR 被静默丢弃（因为 RTS→RESET 直接转，CQ 不产生 completion）。这些 WR 的 slice 永远 POSTED，它们的 batch 永远不 is_finished，只能等外层 60s 硬等到点。

**区分规则**：从 Python 侧 `batch_put failed` 日志的**时间戳分布**初判：
- 失败集中在 bench 起跑头 120s、之后消失 → 场景 A
- 失败从 bench 启动 N 分钟后才开始、之后每 60s 循环不止 → 场景 B
- 突发一次 passive re-establish、之后几分钟内每 60s 一次 batch 失败然后完全恢复 → 场景 C
- 多种混合 → 按 log 时间窗 + error 签名分开处理

**本类**不含：`metadata not found` HTTP 404（类别 1）、`OBJECT_NOT_FOUND(-704)`（类别 3）。

---

## 3. 根因分析

### 3.0 名词澄清：60s 超时不是 bug 本身

"60s 超时" 是**症状最外层表现**。真正的 bug 在 QP 状态机。倒推链：

```
batch_put failed                              ← Python 看到的表象
  ← TRANSFER_FAIL (-800)                      ← Mooncake 返回码
  ← wait_for(60s) 走满                         ← transfer_task.cpp:317
  ← is_finished 永远 false                    ← 没有 slice markSuccess
  ← CQ 永远没有 WC                             ← 没有 ibv_post_send
  ← QP 永远没到 RTS                            ← 握手失败
  ← INIT → RTR 失败                            ★ 真正的 bug
```

"60s" 这个数字来自 PR #398 引入的软件层超时（`transfer_task.cpp:317` 的 `completion_cv.wait_for(..., 60s, ...)`）。握手失败是**毫秒级**发生，但上层没有机制短路 `wait_for(60s)` 把这个早期失败冒上来，所以每个中招的 batch 都付完整的 60 秒代价。`elapsed=120s` 是同一 op 被连环触发多次或客户端层做一次 retry 的 2×60s。

### 3.1 RDMA 基础（读懂后面所有术语）

#### 3.1.1 RDMA 是什么

RDMA = Remote Direct Memory Access。本端网卡（NIC，叫 RNIC）直接把数据 DMA 到**远端进程的内存地址**，绕过远端 CPU、绕过 kernel，拿到的是微秒级延迟 + ≥100 Gbps 带宽。代价是：**建立连接前双方要交换一堆参数、把状态机走到位**；任何一步出错就不能发一个 byte。Mooncake 的 60s 超时就是卡在这个"状态机没走到位"。

#### 3.1.2 三个关键对象

```
     ┌──────────────────────────────────────────────────────────┐
     │  进程 A                                                  │
     │  ┌─────────────┐   send  ┌─────────────┐                 │
     │  │  Send Queue │ ──────► │             │                 │
     │  │   (SQ)      │         │   Queue     │    通过 RNIC    │
     │  └─────────────┘         │   Pair      │ ══════════════► 到远端 QP
     │  ┌─────────────┐   recv  │   (QP)      │                 │
     │  │ Recv Queue  │ ◄────── │             │                 │
     │  │   (RQ)      │         └─────────────┘                 │
     │  └─────────────┘                                         │
     │                                                          │
     │  ┌──────────────────┐                                    │
     │  │ Completion Queue │ ◄── RNIC post 完成事件（WC）       │
     │  │      (CQ)        │     成功 / 失败都会在这里出现      │
     │  └──────────────────┘                                    │
     └──────────────────────────────────────────────────────────┘
```

- **QP (Queue Pair)**：一对"发送队列 + 接收队列"。一条 RDMA "连接" = 一对 QP（本端 QP ↔ 对端 QP）。类比 TCP 的 socket。
- **WR / WQE (Work Request / Work Queue Element)**：一次"发这段内存到对端"的请求，用户 post 给 SQ，RNIC 异步执行。
- **CQ (Completion Queue)**：RNIC 执行完 WR 后，把一个 "WC（Work Completion）" 放到 CQ；用户轮询 CQ 拿到成功/失败。

**关键点**：如果一个 WR 根本没被 post 到 SQ（比如 QP 还没连上），**CQ 里永远不会出现这个 WR 的 WC**。Mooncake 就是卡在这里。

#### 3.1.3 QP 状态机 —— `RESET → INIT → RTR → RTS`（本次 bug 的核心）

RDMA 规范定义 QP 必须按顺序走完以下状态才能发数据：

```
  ┌─────┐   ibv_modify_qp   ┌──────┐   ibv_modify_qp   ┌─────┐   ibv_modify_qp   ┌─────┐
  │RESET│ ─────────────────►│ INIT │ ─────────────────►│ RTR │ ─────────────────►│ RTS │
  └─────┘   (本端独立配置)   └──────┘  (需要对端参数!)  └─────┘    (本端配置)    └─────┘
     ▲                                    ▲                                          │
     │                                    │                                          │
     │     任何出错 → ERR ─────────────────┘                                          │
     │                                                                                │
     └──── 本次 60s bug 的链条：RTR 转换失败 → 卡在 INIT 或 CONNECTING ←──────────────┘
                                                                     可以发/收数据
```

| 状态 | 含义 | 需要的参数 | 现实类比 |
|------|------|------------|---------|
| **RESET** | 刚 create_qp 出来 | — | 电话机刚插电 |
| **INIT** | 本端已配置 port/权限 | 本端 port_num、pkey | 听到拨号音 |
| **RTR** (Ready To Receive) | **已知道对端是谁**，能收包 | **对端 GID、LID、QPN、MTU**（必须通过 handshake 拿！） | 对方号码已拨通、对方在听 |
| **RTS** (Ready To Send) | 能发包 | 本端 PSN、retry_cnt、timeout | 双方都在线，可以说话 |

**最关键的是 `INIT → RTR` 这一步**：你必须先通过一个**带外通道**（Mooncake 里是 HTTP/TCP handshake RPC）拿到对端的：
- `peer_gid`（对端 NIC 的 GUID，类比 MAC 地址）
- `peer_lid`（对端 NIC 的 LID，InfiniBand subnet 内的短地址）
- `peer_qp_num`（对端这条连接使用的 QP 号）
- `path_mtu`（双方协商的 MTU）

拿不到 / 拿错 / 对端还没准备好 → `ibv_modify_qp(RTR)` 失败 → QP 永远停在 INIT → 之后 `ibv_post_send` 会被 RNIC 直接拒绝。**Mooncake 的 `packet mismatch` 就发生在这一步的握手校验阶段；`Connection timed out [110]` 就是 kernel 等 peer 握手包超时**。

### 3.2 Mooncake 整体架构（本类 bug 涉及的组件）

```
  ┌──────────────────────────────────────────────────────────────────────────────┐
  │                           vLLM Python 进程（客户端侧）                        │
  │                                                                              │
  │   batch_put(keys, buffers)                                                   │
  │            │                                                                 │
  │            ▼                                                                 │
  │   ┌───────────────────────────────┐                                          │
  │   │ Mooncake Python Client        │                                          │
  │   │ (mooncake-integration binding)│                                          │
  │   └────────┬──────────────────────┘                                          │
  └────────────┼─────────────────────────────────────────────────────────────────┘
               │  (C++ 调用)
               ▼
  ┌──────────────────────────────────────────────────────────────────────────────┐
  │                  Mooncake Store（libmooncake_store，同进程内）                │
  │                                                                              │
  │   Client::BatchPut()  ─►  SubmitTransfers  ─►  WaitForTransfers              │
  │                                                 (阻塞在 future.get())        │
  │                                                     │                        │
  │                                                     ▼                        │
  │   ┌──────────────────────────────────────────────────────────────────┐       │
  │   │            TransferFuture::wait_for_completion                   │       │
  │   │   completion_cv.wait_for(..., 60s, is_finished)  ◄──┐            │       │
  │   │                                                     │            │       │
  │   │   ★ 60s 超时来自这里 (transfer_task.cpp:317)        │            │       │
  │   └─────────────────────────────────────────────────────┼────────────┘       │
  │                                                         │                    │
  │                                      is_finished = true 才能解阻塞          │
  │                                      需要所有 slice markSuccess/Failed       │
  └─────────────────────────────────────────────────────────┼────────────────────┘
                                                            │
               ┌────────────────────────────────────────────┘
               │  (TransferEngine 把 batch 拆成 slices，交给 WorkerPool)
               ▼
  ┌──────────────────────────────────────────────────────────────────────────────┐
  │                     Mooncake TransferEngine (RDMA 传输)                       │
  │                                                                              │
  │   ┌──────────────┐     ┌────────────┐     ┌────────────┐                     │
  │   │ WorkerPool   │ ──► │RdmaEndpoint│ ──► │    QP      │  ──► RNIC ──► 对端  │
  │   │ (worker线程) │     │(per-peer)  │     │(RESET→INIT │                     │
  │   │              │     │            │     │ →RTR→RTS)  │                     │
  │   └──────┬───────┘     └─────┬──────┘     └─────┬──────┘                     │
  │          │                   │                   │                           │
  │          │ 轮询              │ 握手状态          │ post_send / recv          │
  │          ▼                   ▼                   ▼                           │
  │   ┌──────────────────────────────────────────────────────┐                   │
  │   │                Completion Queue (CQ)                 │ ◄── RNIC 完成事件│
  │   └──────────────────────────────────────────────────────┘                   │
  │                                                                              │
  │   worker_pool.cpp 循环：                                                     │
  │     1. 挑一个 endpoint → 检查 active()                                       │
  │     2. active=false → 进入 "mark it inactive" 分支（line 244）               │
  │     3. active=true  → post_send → 等 CQ → markSuccess/Failed                 │
  └──────────────────────────────────────────────────────────────────────────────┘
```

### 3.3 正常（稳态）情况：happy path

```
Time ────────────────────────────────────────────────────────────────────►

Client     batch_put([k1..k32])
             │
             ├─► submit_batch ────────────────────────────────────────┐
             │                                                        │
             └─► wait_for(60s, is_finished)                           │
                                                                      │
Worker                                                                │
                       endpoint.active()?  ─── yes ───┐               │
                                                      ▼               │
                                                 ibv_post_send ───┐   │
                                                                  │   │
                                                                  ▼   │
RNIC                                                      （数据通过  │
                                                          RDMA 发到   │
                                                          对端）      │
                                                                  │   │
                                                                  ▼   │
                                          ~几百微秒后 CQ 有 WC ───┘   │
                                                      │               │
Worker                                                ▼               │
                                          slice->markSuccess() ───┐   │
                                                                  │   │
                                          is_finished = true      │   │
                                          completion_cv.notify() ─┼───┘
                                                                  │
Client                                     wait_for() 立即返回 ◄──┘
             batch_put() 返回 OK   (elapsed ≈ 毫秒级)
```

### 3.4 Bench 起跑故障 path（场景 A 的代码级流程）

#### 3.4.0 关键前提：QP 是 lazy 建立的，失败后走 "delete-then-rebuild" 而不是"原地重试"

在看下面的时间轴前先明确两件事：

**(a) 请求本身就是握手的触发源（lazy）**

`RdmaEndpoint` 对象在 TransferEngine 初始化时只构造到 `UNCONNECTED` 状态（`rdma_endpoint.cpp:32-81`），**QP 状态机没跑**。只有当某个 worker 线程拿到 slice、发现 `endpoint->connected() == false` 时，才会调 `setupConnectionsByActive()` → `doSetupConnection()` → 走 RESET→INIT→RTR→RTS。所以：

> **"发起请求"和"触发握手"是同一件事，不是两件事。**
> bench 起跑 = 几十个请求同时到 = 几十个握手同时并发 = 风暴。

**(b) 失败后的 endpoint 不会原地重试，而是被销毁、下次重建**

```
t=0.00s   首请求触发握手 → packet mismatch (rdma_endpoint.cpp:283)
          握手失败 → ERR_REJECT_HANDSHAKE
t=0.00s   worker_pool.cpp:244
          endpoint.set_active(false)          ← mark inactive
          slice 进 failed_slice_list
          原始请求开始 wait_for(60s) 计时

t=1.00s   worker 循环检测到 endpoint 空闲满 1s
          context_.deleteEndpoint()           ← endpoint 对象被销毁
          (worker_pool.cpp:238-239)

          (此时那个触发握手的请求仍然卡在 wait_for 里)

t=60s     wait_for(60s) 到点 → TRANSFER_FAIL → batch_put failed
          Python 端看到 elapsed ≈ 61s

t>60s     bench 后续请求进来，目标 segment 没有 endpoint 对象
          → 新建 UNCONNECTED endpoint → 再次触发 doSetupConnection
          此时风暴已散、对端已 ready → 握手成功 → QP 到达 RTS ✓
          该请求毫秒级完成，稳态恢复
```

这解释了两个表象：

1. **为什么 bench 起跑 60-120s 后就自愈**：不是"等一会儿 endpoint 自己恢复"，而是"失败的 endpoint 被销毁，下一次请求新建的 endpoint 赶上了风暴散去后的良好窗口"
2. **为什么会看到 `elapsed≈120s`（2×60s）**：风暴期间同一 segment 可能被连环触发多次 "重建→再失败"，每波都吃一个 60s；或者客户端侧做简单重试也会累加一次 60s

#### 3.4.1 时间轴视图

```
T=0s       (bench 启动)
│
├── 一批 32 个 batch_put 同时涌入 Mooncake Store
│   所有请求的目标 endpoint 都还是首次使用 → 状态 = UNCONNECTED
│
├── Worker 线程 A/B/C/... 并发调用 doSetupConnection()
│   │
│   ├── RESET → INIT ✓   (本端独立，不会出错)
│   │
│   └── INIT → RTR ✗     ← 要拿对端 GID/LID/QPN，
│                           经 HTTP handshake RPC 问对端
│                           对端同时被几十个握手打爆 / 还没 ready
│                           或者 NIC path 元组在校验时对不上
│                           → 报 "packet mismatch" (rdma_endpoint.cpp:283)
│                           → return ERR_REJECT_HANDSHAKE
│
├── worker_pool.cpp:244  "mark it inactive"
│   endpoint.set_active(false)
│   这 32 个请求对应的 slice 被扔进 failed_slice_list
│
├── redispatch() 想换一条路
│   ✗ num_replica=1，该 segment 没别的 endpoint
│   ✗ 所有 NIC path 也同时冷 → 没活的备选
│   slice 原路返回 failed_slice_list
│
│   ♻ 重试循环：redispatch → 仍没活 endpoint → redispatch → ...
│     (worker_pool.cpp:174-268 的循环；retry_cnt 慢慢累加)
│
│   ⚠ 整段时间里：
│       * ibv_post_send 从来没成功出去过
│       * 因此 CQ 从不会 post 任何 WC
│       * 因此 slice->markSuccess()/markFailed() 不会被调
│       * 因此 is_finished 一直是 false
│
├── (这期间 vLLM 的 Python 端看到 batch_put 没返回，也在等)
│
... (60 秒过去) ...
│
T=60s      transfer_task.cpp:317 的 wait_for(60s) 到期
│          completed = false
│          ↓
│          check_task_status() 发现所有 slice 状态异常
│          → 设置 result = ErrorCode::TRANSFER_FAIL
│          ↓
│          client_service.cpp:1611
│          LOG "Transfer failed for key ... TRANSFER_FAIL"
│          ↓
│          Client::BatchPut() 返回 tl::unexpected(TRANSFER_FAIL)
│          ↓
│          Python 端聚合：
│          "batch_put failed: 32/32 keys, codes={'TRANSFER_FAIL':32},
│           elapsed=61.349s"
│
T=61s      (若客户端层面再做一次 retry，同路径再撞 60s → elapsed=120s)
│
... (过几十秒，握手风暴散去，对端 ready；endpoint 被 deleteEndpoint
     回收后，新请求触发的 doSetupConnection 握手成功 → 稳态恢复) ...
│
T=120s~    后续 batch_put 都在毫秒级完成 ✓
```

#### 3.4.2 故障路径的"CQ 永远不 completion"可视化

```
  正常情况：                                 故障情况（bench 起跑）：

  SQ (Send Queue)                            SQ (Send Queue)
  ┌─────────────┐                            ┌─────────────┐
  │ WR #1       │ ─► post_send ─► RNIC       │  (EMPTY)    │ ← 根本没 post 进去
  │ WR #2       │                            │             │   因为 QP 没到 RTS
  └─────────────┘                            └─────────────┘

      ▼ 数据发出                                  ▼

  CQ (Completion Queue)                      CQ (Completion Queue)
  ┌─────────────┐                            ┌─────────────┐
  │ WC #1 ✓     │ ← RNIC post 完成事件       │  (EMPTY)    │ ← 永远没有 WC
  │ WC #2 ✓     │                            │             │
  └─────────────┘                            └─────────────┘

      ▼                                          ▼

  markSuccess() 被调用                       markSuccess() 不会被调用
  is_finished = true                         is_finished = false 永远
  wait_for() 立即返回                        wait_for(60s) 必走满
```

#### 3.4.3 num_replica=1 是"100% 暴露器"而非"根因"

```
num_replica = 2:                              num_replica = 1:

 key ─┬─► endpoint_A (cold, 卡住)              key ───► endpoint_A (cold, 卡住)
      │                                                       │
      └─► endpoint_B (cold, 卡住 or 成功)                      └── 没有备选 ──►  60s 超时
           │                                                                    必现
           如果 B 成功 → key 整体成功
           如果 B 也失败 → 60s 超时
```

num_replica=1 不是 bug 的触发器，而是"**没有冗余能吸收单点冷启动抖动**"的放大器。本 bench 配置 `num_replica=1`，所以场景 A 的失败率从"偶发"提升到"必现"。

### 3.5 INIT → RTR → RTS 为什么在高并发下"必然"失败 —— 四个具体机制

问题不在"状态机本身慢"，而在**状态机要用的外部信息（peer 参数、RPC 回包）在并发风暴里拿不稳**。按贡献度排序：

#### 机制 ①：Metadata server 的 chicken-and-egg 竞争

跨节点握手需要的 `peer_gid / peer_lid / peer_qpn / nic_path` 等信息不是握手 RPC 自己凭空产生的，而是**每个节点把自己的 segment descriptor 推到 metadata server（etcd 或 HTTP metadata plugin），握手时另一方再查回来**。

```
Node A 启动:                              Node B 启动:
  - create QP (RESET)                       - create QP (RESET)
  - updateLocalSegmentDesc → push A         - updateLocalSegmentDesc → push B
                                    ▼
                    ┌───── Metadata Server (etcd / HTTP) ─────┐
                    │   A 的 segment desc 还在写入 / 复制中   │
                    └─────────────────────────────────────────┘
                                    ▲
  - 首请求触发 doSetupConnection             - 首请求触发 doSetupConnection
  - getSegmentDescByName(B) ✓                - getSegmentDescByName(A) ✗  ← A 还没落盘！
  - modify_qp_to_RTR(peer=B) ✓               - return nullptr → "Peer NIC not found"
                                             - 握手失败 → mark inactive
```

关键代码位：
- `rdma_transport.cpp` 中 `startHandshakeDaemon()` 调 `updateLocalSegmentDesc()` 发布本地 segment 到 metadata server
- `rdma_endpoint.cpp:293-309` 里 `getSegmentDescByName(peer_server_name)` 查 peer；查不到就直接报 "Peer NIC not found" 并失败
- `transfer_metadata_plugin.cpp:270-271` 里 HTTP metadata 查询的 `CURLOPT_TIMEOUT_MS = 3000`（3 秒！），metadata server 一过载这 3s 就超时

**bench 起跑这一秒**，几十个节点**同时** push + 同时 query，顺序不定；任何"先查后落"的组合都会命中这个 nullptr 分支。

#### 机制 ②：握手 RPC 服务端是**单线程 accept**

握手 RPC 实际承载在 `transfer_metadata_plugin.cpp` 的一个 listener 线程里（约 line 728-827）：

```cpp
listener_ = std::thread([this]() {
    while (listener_running_) {
        int conn_fd = accept(listen_fd_, ...);    //  ← 单线程 accept
        // JSON parse → on_connection_callback_ → setupConnectionsByPassive
        // 全部同步执行完才回 accept 下一个
    }
});
```

意味着：
- 整台机每秒只能串行处理 **1/T_handshake** 个握手请求
- listen backlog 默认 128（`listen_backlog_ = config.handshake_listen_backlog`）；超出部分**直接被 kernel 丢弃**
- 同一个 socket 的 `SO_RCVTIMEO = 60s`（`transfer_metadata_plugin.cpp:747/951`），对方连上但 60s 内 listener 没轮到它 → recv 超时 → RPC 失败

在 N 个节点同时起动的场景下，握手 RPC 数量是 O(N²) 级；单线程服务器 + 128 backlog = 必挤。

#### 机制 ③：Simultaneous-open 碰撞 → "packet mismatch"

RDMA handshake 有 active 和 passive 两路：
- Active 路径：本端主动 `sendHandshake()` 给对端（`rdma_endpoint.cpp:232`）
- Passive 路径：本端 listener 收到对端的 active 请求 → `setupConnectionsByPassive()` （`rdma_endpoint.cpp:312-381`）

如果 A→B 和 B→A 几乎同时发起（**bench 起跑期高度同步**），下面这串事件会发生：

```
t=0.00s  A 发 active handshake → B                B 发 active handshake → A
t=0.01s                          ↘               ↙
                                  交换中
t=0.05s                          passive 路径已抢先
                                  在两边 setup 了 endpoint
t=0.10s  A 的 active RPC 回包终于到了
         但 endpoint 已经被 passive 路径设置过
         A 比较 peer_desc.local_nic_path vs peer_nic_path_
         → 对不上（因为 passive 路径设的是另一套字段）
         → rdma_endpoint.cpp:281-291  "received packet mismatch"
         → disconnectUnlocked()
         → 返回 ERR_REJECT_HANDSHAKE
```

所以日志里的 "packet mismatch" **不是"网线接错"也不是"配置错"**，而是 **"active + passive 两路同时跑，回包时间差把 descriptor 比对搞坏了"**——这是典型的 simultaneous-open race。低并发下 active 总比 passive 快，碰撞概率低；高并发下两路几乎同步，碰撞变成常态。

**上游修复**：PR #1733（已合入本地 HEAD）修了 "RDMA simultaneous-open 竞态导致的 endpoint use-after-delete"；PR #1705（已合入）修了 "并发 `connect()` 把已建立的 endpoint reset 掉"。这两个补丁把机制 ③ 的严重度大幅降低——但问题并未消失，因为 ① 和 ② 仍然存在。

#### 机制 ④：握手链路**没有 RPC 级 retry**

`sendHandshake()` 失败后：
```cpp
int rc = context_.engine().sendHandshake(...);   // rdma_endpoint.cpp:232
if (rc) {
    resetConnection("handshake RPC failure");
    return rc;              // ← 直接返回失败，没有 backoff、没有 retry
}
```

向上冒泡到 `worker_pool.cpp:244`：
```cpp
endpoint->set_active(false);                     // mark inactive
```

之后只靠 **slice 级 retry**（换 NIC path）和 **1 秒后 deleteEndpoint**→下次请求重建来"恢复"。握手本身不重试。

所以高并发下只要某次握手运气不好（① 的 metadata 没落 / ② 的 backlog 挤满 / ③ 的 simultaneous-open 碰撞），**这一轮请求几乎必然 60s 超时**；恢复要等"风暴散 + endpoint 销毁 + 下次新建"。

#### 汇总：为什么高并发下 `INIT→RTR→RTS` 失败是"必然"而非"偶然"

| 问题 | 低并发 | 高并发 |
|------|--------|--------|
| Metadata server query | 有时间等 peer publish | O(N²) 并发查询，nullptr 分支概率大 |
| RPC listener | 秒级闲置 | backlog 128 极易挤满 |
| RPC `SO_RCVTIMEO` | 60s 用不到 | 排队时间 >60s 就丢 |
| active/passive 竞争 | active 总比 passive 快 | 几乎同步，simultaneous-open race 常态化 |
| 握手 RPC 重试 | 无需 | 无机制 |

**结论**：高并发不是"增加了一点延迟"，而是**把几条本来不触发的失败分支集体抬到必然**。

### 3.6 ETIMEDOUT：场景 A 的更底层定位 + 场景 B 的直接机制

**历史推断（prefill6 分析时）**：`packet mismatch (rdma_endpoint.cpp:283)` 是最强候选机制，指向 simultaneous-open 或 stale reuse。

**autosweep (20260417_135116) 的新发现**：在 `rdma_endpoint.cpp:646` 抓到更上游的 kernel 级错误：

```
[Handshake] Failed to modify QP to RTR,
  check mtu, gid, peer lid, peer qp num:
  Connection timed out [110]                    ← errno=110 (ETIMEDOUT)
```

意思是：QP 状态机在 `INIT → RTR` 这一步就没拿到 peer 的 `mtu / gid / peer lid / peer qp num`，kernel 等 peer 的握手包超时直接拒绝 modify。**`packet mismatch` 是下游症状，不是最源头**。

**两种含义**：
- **场景 A（握手风暴）**：ETIMEDOUT 是机制 ①+② 的表现——peer 还没 publish / RPC backlog 挤满 → peer QP 还没就绪 → kernel 等不到握手包。
- **场景 B（peer 持续失联）**：ETIMEDOUT 就是**真实的故障直接原因**——对端进程挂了 / RNIC 坏了 / 网络断了 → peer QP **永远**不会就绪 → 每次握手都 timeout。

所以同一个 ETIMEDOUT 日志行，场景 A 和场景 B 都会出现，但**持续性和单一 peer 特征**是区分两者的关键。

### 3.7 场景 A vs 场景 B —— 同一处代码，两种完全不同的根因

两个场景都触发同一处代码（`transfer_task.cpp:317 wait_for(60s)`），但根因完全不同。之前把它们混为一谈，这里明确区分：

| 维度 | 场景 A · Bench 起跑握手风暴 | 场景 B · 运行期 peer 持续失联 |
|------|----------------------------|------------------------------|
| **发生时间** | bench 起跑后 [0, 120s] 内集中 | bench 启动 N 分钟后才首次出现 |
| **持续性** | 风暴散去后自愈，稳态恢复 | 持续每分钟一次直到 peer 恢复或 client 停发 |
| **涉及 peer** | 多个 peer 同时出错 | 单一 peer（日志里 `mark it inactive` 永远指向同一个 endpoint） |
| **底层错误** | `packet mismatch`（NIC path 校验失败 / simultaneous-open race） | `Connection timed out [110]`（kernel ETIMEDOUT，QP 无法迁移到 RTR） |
| **故障比例** | 首轮几乎所有 batch 中招 | 只有路由到坏 peer 的部分 key 中招（eg 15/32） |
| **本机角色** | 客户端侧受害 | 客户端侧无异常，但**对端 peer 才是真实故障点** |
| **日志签名** | `packet mismatch` + `mark it inactive` + 多个 peer_nic_path | `Failed to modify QP to RTR ... Connection timed out [110]` + 同一 peer_nic_path 重复出现 |
| **诊断位置** | 本机（看并发压力） | 对端 peer（RNIC / 进程 / 网络通路） |
| **修法方向** | 客户端 ramp-up / 显式 warmup / fail-fast | 对端健康检查 + circuit breaker + 自愈 + 快速判死 |

**区分规则**：从 Python 侧 `batch_put failed` 日志的时间戳分布就能初判——
- 如果失败集中在 bench 起跑头 120s 内、之后完全消失 → **场景 A**
- 如果失败从 bench 启动 N 分钟后才开始、之后每 60s 一次持续不止 → **场景 B**
- 同时都有 → 可能是两种机制叠加，要按 log 时间窗分开处理

> **注**：这张表是 A vs B 的专题对比。本文档还有第三个独立场景 **C（Passive re-establish 丢在途 WR）**，日志签名完全不同（见 §2.3、§3.10），诊断入口 → §9 第 0 步先做三场景快速区分。§1 摘要里有完整 A/B/C 三列的并列对比表。

### 3.8 "重建 endpoint 不是万能药"（场景 B 专属）

早先说过"endpoint 失败后 1 秒销毁、下次新建时风暴已散就会成功"。这个说法**只适用于场景 A**，在场景 B 下不成立。

#### 重建能救的：瞬时失败

```
场景 A 时间线：
  t=0    首次 endpoint → 握手失败（并发风暴）
  t=1s   endpoint 销毁
  t=N    (N >> 1s，等风暴散去) 新请求 → 新建 endpoint → 握手成功 ✓
```

失败的原因是**瞬时条件**：并发风暴、metadata 未就绪、simultaneous-open race。等一会儿条件改变了，下次重建就过了。

#### 重建救不了的：持续性失败

```
场景 B 时间线：
  t=0    endpoint #1 → 握手失败 (peer ETIMEDOUT)
  t=1s   endpoint #1 销毁
  t=60s  新 batch 的 wait_for 到点 → TRANSFER_FAIL
         同时新 batch 来 → 新建 endpoint #2 → 再次失败（peer 还是挂的）
  t=61s  endpoint #2 销毁
  t=120s 新 batch wait_for 到点 → TRANSFER_FAIL
         同时新 batch 来 → 新建 endpoint #3 → 再次失败
  ...每 60s 循环，直到 peer 自己恢复或 client 停发
```

失败的原因是**持续性条件**：对端进程挂了 / RNIC 坏了 / 网络断了。重建只是在同一个条件下再试一遍，结果相同。

#### 为什么"节拍"是 60s 而不是更快

直觉以为："重建很快，失败应该秒级循环"。实际看 log：

```
 08:06:25.096   endpoint #1 握手失败（几百 ms 内）
 08:06:25.316   mark inactive
                ⟶ 握手很早就失败了
                ⟶ 但 slice 在 worker_pool 里循环 redispatch，找不到活 endpoint
                ⟶ batch_desc.is_finished 一直是 false
                ⟶ 外层 wait_for(60s) 慢慢计时
 08:07:25.151   wait_for(60s) 到点 → TRANSFER_FAIL
```

**卡 60 秒的不是握手本身，是外层 `wait_for(60s)`**。握手失败实际在毫秒/秒级就发生了，但上层没有机制把这个早期失败冒出来短路 `wait_for`。每个中招的 batch 都付完整的 60s 代价。

#### 这暴露的三个设计痛点

1. **Fail-fast 缺失**：slice 目标 endpoint 不可用时，没有机制短路 `wait_for(60s)`，只能干等 → 对应修法 **B-1**（submit 前 check `active()`，inactive 立刻返回）
2. **没有 circuit breaker / peer 黑名单**：对连续失败的 peer 仍然每次请求都新建 endpoint 再失败 → 应给 peer 打"最近 N 次都失败"标记，短期内直接 fail-fast 不再尝试握手（类似 HTTP 客户端的熔断器）
3. **没有连接健康检查**：Mooncake 不会主动 ping / heartbeat 检测 peer，只能靠业务请求"撞墙"才发现 → 应有独立的健康检查线程，peer 挂了直接标记，所有命中该 peer 的 batch 立刻 fail

场景 A 的修法（ramp-up / warmup）对场景 B **完全无效**——不管 client 多温柔，peer 挂了就是挂了。场景 B 的根本解是上面三个设计改动 + 查对端机器。

### 3.9 `client_service.cpp` 写路径两个放大 bug（所有场景都受影响）

读 Mooncake `client_service.cpp` 发现**两个都在写路径的设计 bug**，让本该部分成功的 put 变成完全失败：

#### Bug 3.9.a — `SubmitTransfers`: replica 0 submit 失败就 break（**仍存在**，`client_service.cpp:1503-1568`）

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

#### Bug 3.9.b — `WaitForTransfers`: 任一 future 失败就全挂（历史 bug，**本地 HEAD 已修**）

**历史描述**（autosweep_bug_census 报告时）：

```cpp
bool all_transfers_succeeded = true;
for (i = 0 .. op.pending_transfers.size()) {
    auto result = op.pending_transfers[i].get();    // blocking，可能 60s
    if (result != OK) {
        if (all_transfers_succeeded) { first_error = result; failed_transfer_idx = i; }
        all_transfers_succeeded = false;
    }
}
if (!all_transfers_succeeded) op.SetError(first_error, "Transfer N failed");
```

**历史行为**：replica 0 等不到 completion → 60s timeout → 尽管 replica 1/2 可能都 OK，op 仍被 `SetError(TRANSFER_FAIL)`。

**本地修复状态（2026-04-18, HEAD `be75ca0`）**：commit `client_service: quorum write on WaitForTransfers + richer failure logging` **已修复**。当前实际代码（`client_service.cpp:1592-1649`）：

```cpp
size_t num_succeeded = 0;
ErrorCode last_error = ErrorCode::OK;
std::vector<std::pair<size_t, ErrorCode>> failed_replicas;

for (size_t i = 0; i < op.pending_transfers.size(); ++i) {
    ErrorCode transfer_result = op.pending_transfers[i].get();
    if (transfer_result == ErrorCode::OK) {
        num_succeeded++;
    } else {
        last_error = transfer_result;
        failed_replicas.emplace_back(i, transfer_result);
    }
}

// Quorum semantics: if at least one replica succeeded, the put
// transfer phase is considered successful and finalization proceeds.
if (num_succeeded > 0) {
    // Transfer phase successful - continue to finalization
    if (!failed_replicas.empty()) {
        LOG(WARNING) << "WaitForTransfers partial success for key " << op.key
                     << " quorum=" << num_succeeded << "/" << op.pending_transfers.size()
                     << " failed_replicas=[...]";
    }
} else {
    LOG(ERROR) << "Transfer failed for key " << op.key
               << " quorum=0/" << op.pending_transfers.size()
               << " (all replicas failed)";
    op.SetError(last_error, "All replicas failed");
}
```

**当前语义**：quorum write——只要 ≥1 replica 成功就算成功。**本地剩余 caveat**（HEAD commit 注释原话）：
- `master-side replica tracking still assumes all-or-nothing`
- `Tracking per-replica final state at the master is a follow-up`

所以 `WaitForTransfers` 客户端路径的 "all-or-nothing" 已修，但 **master 侧的 per-replica 状态跟踪还没跟上**，这是下一个 PR 的工作。

#### 两个 bug 叠加的历史放大效应（Bug 3.9.b 修前）

每个 Python 侧的 `batch_put failed: 1/32 keys failed ... codes={TRANSFER_FAIL: 1}` 对应的 C++ ERROR 都是 `Transfer failed for key ... (Transfer 0 failed)`——**历史上只有 replica 0 失败，replica 1/2 在 Mooncake 这两个设计 bug 下没有发挥作用**。

| autosweep sweep / 节点 | 失败 key 数 | 历史根因 |
| --- | --- | --- |
| p5/rack1-16 | 1/32 | replica 0 future 60s timeout + WaitForTransfers all-or-nothing |
| p8/rack1-04 | 1/32 + 1/32 | 同上 |
| p8/rack1-03 | 1/32 + 1/32 + 2/32 | 同上 |

**修好后（HEAD `be75ca0`）**：Bug 3.9.b 的贡献消失；Bug 3.9.a 仍存在（`SubmitTransfers` replica 0 失败就 break）。所以若 autosweep 在 HEAD `be75ca0` 之后重跑，**剩余的 batch_put_failed 只来自 Bug 3.9.a**——而 Bug 3.9.a 只在 `num_replica ≥ 2` 时才能显现（单 replica 时无所谓顺序）。本 bench `num_replica=1`，**Bug 3.9.a 也不会放大失败**。

**结论**：写路径放大 bug 在 HEAD `be75ca0` 已基本消除；autosweep 观察到的 6 个 batch_put_failed 在现在重跑应该不会复现（同时期性能提升也会改变握手风暴窗口，需要新一轮 bench 验证）。

### 3.10 场景 C 的代码级机制（passive re-establishment 丢在途 WR）

场景 C 和 A/B 在代码路径上完全不同。A/B 是"握手从一开始就失败 → endpoint 到不了 RTS → 没有 WR 发出"；场景 C 是"**握手成功了**、QP 到了 RTS、新请求能正常跑，但 **reset 那一瞬间正在 SQ 里的在途 WR 被静默丢弃**，它们的 batch 只能卡 60s"。

#### 入口：对端要求 passive re-handshake

当对端的 QP 换了（新 qp_num 和本机缓存的 `peer_qp_num_list_` 不同），对端会发 active handshake RPC 到本机；本机 listener 收到后调 [`setupConnectionsByPassive`](../../../Mooncake/mooncake-transfer-engine/src/transport/rdma_transport/rdma_endpoint.cpp#L312-L381)（`rdma_endpoint.cpp:312-381`），进入 re-handshake 分支（line 325 的 `LOG(WARNING) << "Re-establish connection: ..."`）。

#### 关键步骤 ①：resetConnection → disconnectUnlocked 把 QP 直接从 RTS 打回 RESET

[`resetConnection`](../../../Mooncake/mooncake-transfer-engine/src/transport/rdma_transport/rdma_endpoint.cpp#L419-L437)（line 327）调 [`disconnectUnlocked`](../../../Mooncake/mooncake-transfer-engine/src/transport/rdma_transport/rdma_endpoint.cpp#L388-L417)。后者直接：

```cpp
// rdma_endpoint.cpp:394
attr.qp_state = IBV_QPS_RESET;
ret = ibv_modify_qp(qp_list_[i], &attr, IBV_QP_STATE);
```

**这是直接从 RTS 转 RESET，不过 ERROR 态**。在 Mellanox 驱动上，这种转法**不会**给在途 SQ 上的 WR 产生 `WR_FLUSH_ERR` 的 CQ 事件——WR 被静默丢弃。这是场景 C 的机械根因。

#### 关键步骤 ②：检测到在途 WR → 发 Outstanding 警告 + 清计数器

```cpp
// rdma_endpoint.cpp:404-412
if (wr_depth_list_[i] != 0) {
    if (!displayed) {
        LOG(WARNING) << "Outstanding work requests found, CQ will not be generated";
        displayed = true;
    }
    __sync_fetch_and_sub(cq_outstanding_, wr_depth_list_[i]);
    wr_depth_list_[i] = 0;
}
```

代码只做了两件事：打个警告 log，把 `cq_outstanding_` 和 `wr_depth_list_[i]` 这两个计数器扣回来。**没有对这些 WR 对应的 slice 调 `markFailed()`**——它们仍停在 `Transport::Slice::POSTED` 状态，对上层是"还在飞"。

#### 关键步骤 ③：在途 slice 的后续命运

在途 slice 既不会 `markSuccess` 也不会 `markFailed`：
- worker_pool 的 `performPollCq` 不会看到它们（CQ 里永远没有它们的 WC）
- 它们的 `batch_desc.is_finished` 永远不会被置 true
- batch 的 `wait_for(60s)` / polling 循环只能等**外层超时**（详见 §3.11），60s 到点后 `TRANSFER_FAIL`

#### 后续：新 batch 能正常跑

`setupConnectionsByPassive` 继续调 [`doSetupConnection`](../../../Mooncake/mooncake-transfer-engine/src/transport/rdma_transport/rdma_endpoint.cpp#L545-L567)（line 368），成功后 `status_ = CONNECTED`、`peer_qp_num_list_` 更新（line 564-565）。新 batch 路由进来，`endpoint->connected() == true`，走正常的 `submitPostSend` → CQ → `markSuccess`，毫秒级完成。

所以场景 C 的表象是：**一瞬间的 passive re-establish → N 个在途 batch 接下来几分钟里逐个 60s 超时 → 之后完全恢复正常**。受影响 batch 数量 = re-establish 瞬间 SQ 上的在途 WR 数量 / 每 batch 平均 slice 数。

#### 和场景 A/B 的根本区别

场景 A/B 的失败路径：握手到不了 RTS → slice 根本没被 post_send 过 → 没有 WR 被 RNIC 接手 → CQ 从一开始就不可能有完成。
场景 C 的失败路径：握手成功过、WR 被 post_send 过、RNIC 本来可能完成、但被 reset 瞬间夺走了 completion 权。

### 3.11 polling 分支 vs event-driven 分支 —— 两条 60s 计时路径

文档前文 §3.0 / §3.4.1 把 60s 硬等源头都标成 `transfer_task.cpp:317` 的 `completion_cv.wait_for`。实际二进制要看编译宏 `USE_EVENT_DRIVEN_COMPLETION` 走哪条：

| Build flag | 等待函数 | 60s 超时打印行 | is_finished 判定 |
|---|---|---|---|
| `USE_EVENT_DRIVEN_COMPLETION` **defined** | `completion_cv.wait_for(60s, ...)` at [`transfer_task.cpp:317`](../../../Mooncake/mooncake-store/src/transfer_task.cpp#L317) | [`transfer_task.cpp:341`](../../../Mooncake/mooncake-store/src/transfer_task.cpp#L341) | atomic `batch_desc.is_finished`（slice 的 markSuccess/Failed 里原子置位） |
| **未定义**（本 bench 的情况） | while 循环 polling at [`transfer_task.cpp:350-369`](../../../Mooncake/mooncake-store/src/transfer_task.cpp#L350-L369) | [`transfer_task.cpp:353`](../../../Mooncake/mooncake-store/src/transfer_task.cpp#L353) | 每轮调 `check_task_status()` → `engine_.getTransferStatus()` → 看 `success_slice_count + failed_slice_count == slice_count` |

本 bench 日志里看到的 `E transfer_task.cpp:353` 是 polling 分支，不是 :317 的 event-driven。两条路都受同一个 `timeout_seconds = 60` 常量约束，超时行为一致，只是打印行号不同。

实现细节看 [`check_task_status`](../../../Mooncake/mooncake-store/src/transfer_task.cpp#L225-L275)（`transfer_task.cpp:225-275`），它遍历每个 transfer 调 `getTransferStatus`；若任何一个 FAILED 立刻把 batch 结果设为 TRANSFER_FAIL，全部 COMPLETED 才 OK，否则（至少有一个 WAITING）继续循环。

### 3.12 `slice_timeout` —— 已经写好但默认关闭的 fail-fast 机制

这条是三个场景都卡 60s 的**共同上层根因**。§3.8（重建不是万能药）里说"Fail-fast 缺失"，其实 Mooncake 有一层现成的 slice 级 fail-fast，只是**默认没开**。

#### 位置和逻辑

[`MultiTransport::getTransferStatus`](../../../Mooncake/mooncake-transfer-engine/src/multi_transport.cpp#L143-L180)（`multi_transport.cpp:143-180`）在判定 `success + failed < slice_count`（即还没跑完）时，会走这段：

```cpp
// multi_transport.cpp:163-176
if (globalConfig().slice_timeout > 0) {
    auto current_ts = getCurrentTimeInNano();
    const int64_t kPacketDeliveryTimeout =
        globalConfig().slice_timeout * 1000000000;
    for (auto &slice : task.slice_list) {
        auto ts = slice->ts;
        if (ts > 0 && current_ts > ts &&
            current_ts - ts > kPacketDeliveryTimeout) {
            LOG(INFO) << "Slice timeout detected";
            status.s = Transport::TransferStatusEnum::TIMEOUT;
            return Status::OK();
        }
    }
}
status.s = Transport::TransferStatusEnum::WAITING;
```

`slice->ts` 是这个 slice 被 post_send 出去的时间戳（见 [`rdma_endpoint.cpp:505`](../../../Mooncake/mooncake-transfer-engine/src/transport/rdma_transport/rdma_endpoint.cpp#L505)：`slice->ts = getCurrentTimeInNano();`）。

#### 默认和开关

- **配置默认**：[`config.h:54`](../../../Mooncake/mooncake-transfer-engine/include/config.h#L54) `int64_t slice_timeout = -1`（负数即关闭）
- **环境变量**：`MC_SLICE_TIMEOUT`（单位：秒），见 [`config.cpp:241-245`](../../../Mooncake/mooncake-transfer-engine/src/config.cpp#L241-L245)
- **作用**：开启后（例如设 `MC_SLICE_TIMEOUT=10`），任何一个 slice 从 post_send 到现在超过 10 秒仍没 completion，整个 batch 立刻被判为 TIMEOUT，polling 分支里 `check_task_status` 看到后会把 batch 结果设成 FAILED → `wait_for` 循环立刻返回

#### 对三个场景的效果

| 场景 | `slice_timeout=10` 开启后 |
|---|---|
| A（起跑风暴） | 握手失败 → slice 其实没被 post（ts==0 不走超时），仍由 60s 硬等兜底。**不直接救场 A**。 |
| B（peer 挂了） | 同上：握手失败 → slice 没 post → `slice->ts == 0`，`ts > 0` 条件不满足，**不救场 B**。 |
| C（passive re-establish 丢 WR） | slice 曾被 post（`ts>0`）、之后被 reset 丢弃，10s 后 `TIMEOUT` 触发，batch 10s 左右失败而非 60s。**直接消除场 C 60s 等**。 |

所以 `MC_SLICE_TIMEOUT=10` 是**场景 C 的零代码改动治标方案**；对 A/B 无效（它们的失败 slice 从没到 `ts>0` 那一步）。相比之下，**真正通用的 fail-fast 仍需 §3.8 讨论的 "submit 前 check `active()`"** 之类的方案。

---

## 4. 证据

### 4.1 prefill_6_mtc_18 主 run（场景 A 集中爆发）

来源：`bench_results/pd_kimi_nsys_prefill6_light/prefill_6_mtc_18`

**按日志文件统计的握手失败**：

| 日志文件 | `handshake timeout` 数量 | 代表性位置 |
| --- | ---: | --- |
| `prefill-1-gb200-rack1-04.log` | 17 | `:1340` |
| `prefill-0-gb200-rack1-03.log` | 5 | `:1286` |
| `prefill-3-gb200-rack1-08.log` | 3 | `:1912` |
| `prefill-4-gb200-rack1-10.log` | 1 | `:1741` |
| `prefill-5-gb200-rack1-11.log` | 1 | `:1583` |

问题并不是平均分布在所有 prefill worker 上的。

**错误计数汇总**：

| 类别 | 数量 | 时间 / 含义 |
| --- | ---: | --- |
| `handshake timeout` | 27 | 全部发生在 `2026-04-14 09:15:48-09:15:49`（1 秒窗口） |
| `packet mismatch` | 14 | 握手描述符或路径不匹配信号 |
| `inactive endpoint` | 27 | endpoint 建立失败后 worker 将其标记为 inactive |
| `transfer timeout` | 9 | `Failed to complete transfers after 60 seconds` |
| `Transfer failed for key` | 9 | 9 个 key 命中 `TRANSFER_FAIL` |
| `batch_put failed` | 6 | Python batch warning 覆盖了这 9 个 key |

顺序很重要：**先握手失败 → 后 transfer timeout → 最后 Python warning**。

**热点主机分布**：

| Host | Count |
| --- | ---: |
| `192.168.0.104` | 11 |
| `192.168.0.107` | 4 |
| `192.168.0.108` | 4 |
| `192.168.0.103` | 4 |
| `192.168.0.110` | 3 |
| `192.168.0.111` | 1 |

不是均匀分布的启动噪声——某几台机器被集中踩到。

**所有 60 秒超时 batch 与失败 key 映射**：

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

### 4.2 autosweep 20260417_135116 跨 sweep 统计

来源：`bench_results/kimi_pd_p1to17d1_mooncake_nsys_autosweep/20260417_135116/`

**全 sweep 扫描结果**：8 份 prefill log 有相关 signal，按是否有业务层错误分两组。

#### 4.2.1 Group A — 启动期握手打嗝，自愈成功，**数据面无损**（5 份 log）

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

#### 4.2.2 Group B — 握手失败 → 升级为 60s timeout → `batch_put failed`（3 份 log）

| sweep 点 | prefill 索引 | 节点 | `batch_put failed` | `TRANSFER_FAIL` | `60s timeout` | `packet mismatch` | `mark inactive` |
| --- | :---: | --- | ---: | ---: | ---: | ---: | ---: |
| `prefill_5_mtc_80/attempt_2` | prefill-4 | rack1-16 | 1 | 7 | 3 | 0 | 0 |
| `prefill_8_mtc_128/attempt_1` | prefill-1 | rack1-04 | 2 | 10 | 4 | 0 | 0 |
| `prefill_8_mtc_128/attempt_1` | prefill-0 | rack1-03 | **3** | **11** | **4** | **2** | **2** |

**关键观察**：Group B 里**只有 `prefill-0/rack1-03` 一份 log 同时有本地握手失败 + 业务失败**。其它 2 份 log（`rack1-16`、`rack1-04`）有 batch_put failed 却**没有本地 packet mismatch / mark inactive / QP→RTR**。

**解释**：这 2 份 log 是"**受害方**"——它们本地的 RDMA 端点正常，但在通过 RDMA 向**别的节点上已 inactive 的 endpoint**写入时，等不到 completion → Mooncake 在 `transfer_task.cpp:341` 做 60 秒硬等待 → timeout → TRANSFER_FAIL → Python 日志 `batch_put failed`。

换言之，**"握手失败的节点"和"业务请求失败的节点"是两个角色**。rack1-03 罕见地同时扮演两个角色（自己先握手失败，又被业务再踩到）。

#### 4.2.3 全局计数（跨 8 份 log）

| 错误类型 | 合计 |
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

**注意**：这次 **没有**观察到类别 1（`metadata not found`）。所以**类别 1 在本 run 没复现**；**类别 2 在多个 sweep 点都出现了**。

#### 4.2.4 节点规模与握手失败次数正相关

| sweep 点 | P | `Failed QP→RTR` | `mark inactive` |
| --- | :---: | ---: | ---: |
| prefill_3 | 3 | 0 | 0 |
| prefill_4 | 4 | 0 | 0 |
| prefill_5 | 5 | 3 | 3 |
| prefill_6 | 6 | 7 | 7 |
| prefill_7 | 7 | 0 | 0 |
| prefill_8 | 8 | 15 | 15 |

→ **节点越多，启动风暴越猛，启动期握手失败越频繁**。但 P=7 又是 0（可能恰好节点组合较好 / 启动时序错开）。

#### 4.2.5 握手失败是否升级，取决于业务路由

对比 `prefill_8_mtc_128`：

| 节点 | 握手失败 | `batch_put failed` |
| --- | ---: | ---: |
| rack1-03 | 有（pm=2, mi=2） | **3 条** ← 升级 |
| rack1-04 | **0** | **2 条** ← 被别人拖下水 |
| rack1-12 | 有（qptrt=4, mi=2） | 0 ← 自愈 |
| rack1-16 | 有（qptrt=11, mi=11） | 0 ← 自愈 |

握手失败最严重的 **rack1-16（11 次）反而 0 业务失败**；握手正常的 **rack1-04 却被 2 条 batch_put failed 打中**。说明业务失败的分布**不是看本地握手情况，而是看本地的写请求是否命中了别人家坏掉的 endpoint**。

### 4.3 Group A 代表：prefill_6_mtc_96 / rack1-17（112 毫秒内 6 次失败然后自愈）

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

### 4.4 Group B 重灾：prefill_8_mtc_128 / rack1-03（握手失败 → 60s 后升级）

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

60 秒后升级：
```
[16:20:27.771656] E client_service.cpp:1611  Transfer failed for key
                  c0285e...@tp_rank:0@pcp0@dcp0@pp_rank:0@3af799...:
                  TRANSFER_FAIL (Transfer 0 failed)
[16:20:29.108819] E client_service.cpp:1611  Transfer failed for key ...@fed86dc2... TRANSFER_FAIL
```

Python `mooncake_store_worker` 聚合：
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

### 4.5 用户提供 log：纯场景 B 的典型

来源：`prefill-2-gb200-rack1-13.log` 08:11:25 失败的 batch `104982289024128`

- bench 启动 `07:56:32`，首次错误 `08:06:25`（T+10min），之后每 60s 一次持续到 log 结束
- 全程仅**一次** `mark it inactive`，目标 `192.168.0.113:13581@rocep195s0`
- 错误是 `Connection timed out [110]`，**不是** `packet mismatch`
- 只有 **15/32 keys** 中招，其他 17 个 keys 同一时间 BatchPut 毫秒级成功
- **结论**：这条 log 是**纯场景 B**（peer 持续失联），不是 prefill_6 的场景 A

这意味着需要去查 `192.168.0.113` 这台机器的：
1. `mooncake_master` / 业务进程是否还活着？
2. `rocep195s0` 这张 RNIC 的 `ibv_devinfo` + `ibstat` 状态？
3. 该 endpoint 的 QPN 是否和其他节点 cache 的值一致？

### 4.6 60s 其他来源排除表

| 来源 | 文件 | 是否本次元凶 | 原因 |
|------|------|-------------|------|
| **`transfer_task.cpp:317` / `:353`** | mooncake-store | **★ 是** | 与 `elapsed ≈ 60n` 吻合；bench 起跑窗口集中；polling 分支打 :353，event-driven 分支打 :317（详见 §3.11） |
| TCP handshake `SO_RCVTIMEO=60` | transfer_metadata_plugin.cpp:747/951 | 间接 | 会在握手链路里触发，但错误通过 §3.5 机制 ④ 路径冒泡 |
| `etcd snapshotTimeout=60` | etcd_wrapper.go:82 | 否 | 控制面快照路径 |
| `kConnectionIdleTimeout=60` | tcp_transport.h:145 | 否 | TCP 连接池空闲回收 |
| `client_live_ttl_sec=60` | master.yaml:21 | 否 | 是 client 租约，不会卡单次调用 |
| `MC_TRANSFER_TIMEOUT=30` | transfer_engine_py.cpp | 否 | 默认 30，与 60 不符 |
| `VLLM_ABORT_REQUEST_TIMEOUT=480+60` grace | vllm/envs.py:203 | 否 | 数量级不符 |

### 4.7 场景 C 代表：prefill-1-gb200-rack1-07.log（passive re-establish 丢在途 WR）

来源：`bench_results/pd_kimi_nsys_prefill5_light/prefill_5_mtc_16/prefill-1/prefill-1-gb200-rack1-07.log`（本机 `192.168.0.107`）

#### 时间线

- **bench 启动** `07:56:32`（log 开头的 APIServer pid=2658222）
- **T+11 分钟稳态运行**：无任何握手相关 ERROR/WARNING
- **08:57:56 passive re-establish 风暴**（全部在 ~14ms 内）：
  ```
  08:57:56.164186  rdma_endpoint.cpp:325  Re-establish connection:
                   local 192.168.0.107:12730@rocep195s0, peer 192.168.0.101:13199@rocep196s0
  08:57:56.164295  rdma_endpoint.cpp:325  Re-establish connection:
                   local 192.168.0.107:12531@rocep195s0, peer 192.168.0.101:13577@rocep196s0
  08:57:56.164837  rdma_endpoint.cpp:325  Re-establish connection:
                   local 192.168.0.107:12945@rocep139s0, peer 192.168.0.101:13199@rocep195s0
  08:57:56.167949  rdma_endpoint.cpp:124  Outstanding work requests found, CQ will not be generated
  08:57:56.168116  rdma_endpoint.cpp:124  Outstanding work requests found, CQ will not be generated
  08:57:56.169642  rdma_endpoint.cpp:124  Outstanding work requests found, CQ will not be generated
  08:57:56.169667  rdma_endpoint.cpp:124  Outstanding work requests found, CQ will not be generated
  08:57:56.171326  rdma_endpoint.cpp:433  Successfully reset the endpoint (triggered by: re-establishing connection (passive)).
  08:57:56.171520  rdma_endpoint.cpp:433  Successfully reset the endpoint (passive)
  08:57:56.171653  rdma_endpoint.cpp:433  Successfully reset the endpoint (passive)
  08:57:56.174367  rdma_endpoint.cpp:325  Re-establish connection: 192.168.0.107:12945@rocep139s0 ↔ 192.168.0.101:13199@rocep196s0  (第 4 个)
  08:57:56.175865  rdma_endpoint.cpp:124  Outstanding work requests found, CQ will not be generated
  08:57:56.177412  rdma_endpoint.cpp:124  Outstanding work requests found, CQ will not be generated
  08:57:56.178690  rdma_endpoint.cpp:433  Successfully reset the endpoint (passive)
  ```
- **共计**：4 条 `Re-establish connection`、6 条 `Outstanding work requests found`、4 条 `Successfully reset`，全部在 08:57:56.164-.178 的 14 毫秒窗口内，目标全指向 peer `192.168.0.101` 的 3 个不同 port/nic
- **08:58:55 开始逐个 60s 超时**：
  ```
  08:58:55.073  transfer_task.cpp:353  Failed to complete transfers after 60 seconds for batch 102649517889648 (pid 2661428)
  08:58:55.151  transfer_task.cpp:353  batch 77734613640592 (pid 2661393)
  08:59:55.073  transfer_task.cpp:353  batch 102649517912192 (pid 2661428)
  08:59:55.152  transfer_task.cpp:353  batch 77734613927744 (pid 2661393)
  09:00:55.152  transfer_task.cpp:353  batch 77734613699264 (pid 2661393)
  09:01:55.153  transfer_task.cpp:353  batch 77734613561712 (pid 2661393)
  ```
  **6 条 60s 超时**，时间节拍精确 60s 一条（因为 worker 串行：一 batch `wait_for` 阻塞 60s → fail → worker 立刻抓下一个同样打向坏 endpoint 的 batch → 再 60s）
- **log 末尾**：`BatchPut: count=258` 大部分成功，`BatchPutRevoke: count=1`，之后 `Shutting down`

#### 场景 C 的"阴性证据"

**全 log 完全不存在**这些签名（grep 验证）：
- `packet mismatch` — 0 次
- `mark it inactive` — 0 次
- `Failed to modify QP to RTR` — 0 次
- `Connection timed out [110]` — 0 次
- `Worker: Process failed for slice` — 0 次（= CQ 没产生任何 error completion，和 "CQ will not be generated" 注释吻合）
- `Worker: Received context async event` — 0 次（没 QP_FATAL）
- `Failed to ibv_post_send` — 0 次
- `evicted` — 0 次（SIEVE 没触发）
- `Peer NIC not found` — 0 次（metadata 查得到）

只有 §2.3 列的三类签名（`Re-establish connection` + `Outstanding work requests found` + `Successfully reset`）+ 后续 60s 超时。

#### 结论与修法

- 这是教科书级别的场景 C 案例：**peer 192.168.0.101 的某个进程 QP 换了（可能是 SIEVE eviction / 内部抖动 / 进程重启），主动发 passive re-establish；本机配合重建，但 reset 瞬间在 SQ 上的 6 个 WR 被静默丢弃**
- 这 6 个 WR 分别属于 6 个 batch，它们的 wait_for 从提交时刻 +60s 到点逐个 fail；worker 串行等待放大了总耗时
- **不是客户端 ramp-up / warmup 能救的**（问题不在起跑）
- **场景 A 的修法完全不适用**
- 治标方案：设 `MC_SLICE_TIMEOUT=10` 环境变量，把 60s 变 10s（见 §3.12）
- 治本方案：`disconnectUnlocked` 里 outstanding WR 清零时**对每个关联 slice 调 `markFailed()`**，让 batch 立刻失败而不是干等外层超时（见 §7.2 第 11 行）

---

## 5. 源码锚点（grep-verified）

### 5.1 Mooncake 侧

| 作用 | 文件:行 |
| --- | --- |
| **60s `wait_for` ★ event-driven 分支** | [`mooncake-store/src/transfer_task.cpp:317`](../../../Mooncake/mooncake-store/src/transfer_task.cpp#L317) |
| **60s polling 分支超时 log**（本 bench 实际走的路径） | [`transfer_task.cpp:353`](../../../Mooncake/mooncake-store/src/transfer_task.cpp#L353) |
| polling 分支 `check_task_status` 循环 | [`transfer_task.cpp:225-275`](../../../Mooncake/mooncake-store/src/transfer_task.cpp#L225-L275) |
| 60s timeout（prefill6 分析引用） | [`transfer_task.cpp:341`](../../../Mooncake/mooncake-store/src/transfer_task.cpp#L341) |
| 懒建立入口 `doSetupConnection` | [`rdma_endpoint.cpp:569-671`](../../../Mooncake/mooncake-transfer-engine/src/transport/rdma_transport/rdma_endpoint.cpp#L569-L671) |
| RTR 转换（要 peer 参数） | [`rdma_endpoint.cpp:607-649`](../../../Mooncake/mooncake-transfer-engine/src/transport/rdma_transport/rdma_endpoint.cpp#L607-L649) |
| **ETIMEDOUT 错误点** ★ | [`rdma_endpoint.cpp:646`](../../../Mooncake/mooncake-transfer-engine/src/transport/rdma_transport/rdma_endpoint.cpp#L646) |
| **packet mismatch** 错误点 | [`rdma_endpoint.cpp:281-291`](../../../Mooncake/mooncake-transfer-engine/src/transport/rdma_transport/rdma_endpoint.cpp#L281-L291) |
| `reuse existing connection` | [`rdma_endpoint.cpp:254`](../../../Mooncake/mooncake-transfer-engine/src/transport/rdma_transport/rdma_endpoint.cpp#L254) |
| `sendHandshake` 无重试 | [`rdma_endpoint.cpp:232`](../../../Mooncake/mooncake-transfer-engine/src/transport/rdma_transport/rdma_endpoint.cpp#L232) |
| `Successfully reset the endpoint` | [`rdma_endpoint.cpp:433`](../../../Mooncake/mooncake-transfer-engine/src/transport/rdma_transport/rdma_endpoint.cpp#L433) |
| **`disconnectUnlocked` RTS→RESET 直转 + Outstanding WR 日志** ★（场景 C 机械根因）| [`rdma_endpoint.cpp:388-417`](../../../Mooncake/mooncake-transfer-engine/src/transport/rdma_transport/rdma_endpoint.cpp#L388-L417) |
| **场景 C Re-establish 日志行** | [`rdma_endpoint.cpp:325`](../../../Mooncake/mooncake-transfer-engine/src/transport/rdma_transport/rdma_endpoint.cpp#L325) |
| **场景 C Outstanding WR 日志行**（disconnectUnlocked 里） | [`rdma_endpoint.cpp:406-407`](../../../Mooncake/mooncake-transfer-engine/src/transport/rdma_transport/rdma_endpoint.cpp#L406-L407) |
| `doSetupConnection` 成功后置 CONNECTED | [`rdma_endpoint.cpp:564-565`](../../../Mooncake/mooncake-transfer-engine/src/transport/rdma_transport/rdma_endpoint.cpp#L564-L565) |
| `slice->ts` 起点（slice 在 submitPostSend 里被打上时间戳） | [`rdma_endpoint.cpp:505`](../../../Mooncake/mooncake-transfer-engine/src/transport/rdma_transport/rdma_endpoint.cpp#L505) |
| `getSegmentDescByName` (chicken-and-egg 触发点) | [`rdma_endpoint.cpp:293-309`](../../../Mooncake/mooncake-transfer-engine/src/transport/rdma_transport/rdma_endpoint.cpp#L293-L309) |
| `setupConnectionsByPassive` (simultaneous-open + 场景 C 的入口) | [`rdma_endpoint.cpp:312-381`](../../../Mooncake/mooncake-transfer-engine/src/transport/rdma_transport/rdma_endpoint.cpp#L312-L381) |
| **`slice_timeout` 半成品 fail-fast 机制** ★ | [`multi_transport.cpp:163-176`](../../../Mooncake/mooncake-transfer-engine/src/multi_transport.cpp#L163-L176) |
| `slice_timeout` 默认 `-1`（关闭） | [`config.h:54`](../../../Mooncake/mooncake-transfer-engine/include/config.h#L54) |
| `MC_SLICE_TIMEOUT` 环境变量读取 | [`config.cpp:241-245`](../../../Mooncake/mooncake-transfer-engine/src/config.cpp#L241-L245) |
| `mark it inactive` | [`worker_pool.cpp:244-248`](../../../Mooncake/mooncake-transfer-engine/src/transport/rdma_transport/worker_pool.cpp#L244-L248) |
| 跳过 inactive + redispatch 循环 | [`worker_pool.cpp:236-268, :345-382`](../../../Mooncake/mooncake-transfer-engine/src/transport/rdma_transport/worker_pool.cpp#L236-L268) |
| `Transfer failed for key` ERROR | [`client_service.cpp:1611`](../../../Mooncake/mooncake-store/src/client_service.cpp#L1611) |
| prefill6 引用的 TRANSFER_FAIL 行 | [`client_service.cpp:1681`](../../../Mooncake/mooncake-store/src/client_service.cpp#L1681) |
| **Bug 3.9.a** `SubmitTransfers` replica 0 break | [`client_service.cpp:1531-1568`](../../../Mooncake/mooncake-store/src/client_service.cpp#L1531-L1568) |
| **Bug 3.9.b** `WaitForTransfers` 任一失败就全挂 | [`client_service.cpp:1572-1614`](../../../Mooncake/mooncake-store/src/client_service.cpp#L1572-L1614) |
| HTTP metadata 查询 timeout 3s | [`transfer_metadata_plugin.cpp:270-271`](../../../Mooncake/mooncake-transfer-engine/src/transfer_metadata_plugin.cpp#L270-L271) |
| 握手 RPC 单线程 listener | [`transfer_metadata_plugin.cpp:728-827`](../../../Mooncake/mooncake-transfer-engine/src/transfer_metadata_plugin.cpp#L728-L827) |
| 握手 RPC `SO_RCVTIMEO=60s` | [`transfer_metadata_plugin.cpp:747, :951`](../../../Mooncake/mooncake-transfer-engine/src/transfer_metadata_plugin.cpp#L747) |
| `SIEVEEndpointStore::evictEndpoint` | [`endpoint_store.cpp:210`](../../../Mooncake/mooncake-transfer-engine/src/transport/rdma_transport/endpoint_store.cpp#L210) |
| `SIEVEEndpointStore::deleteEndpoint` | [`endpoint_store.cpp:170`](../../../Mooncake/mooncake-transfer-engine/src/transport/rdma_transport/endpoint_store.cpp#L170) |

### 5.2 vLLM 侧

| 作用 | 文件:行 |
| --- | --- |
| `batch_put failed` Python 聚合 | [`mooncake_store_worker.py:480`](../../distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_worker.py#L480) |
| 60s 计时起点（elapsed 字段） | [`mooncake_store_worker.py:479`](../../distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_worker.py#L479) |
| `transfer_engine_py.cpp` 聚合 `{'TRANSFER_FAIL': N}` | `Mooncake/mooncake-integration/transfer_engine/transfer_engine_py.cpp` |

---

## 6. 上游 PR 状态（家族 2 相关）

### 6.1 本地 HEAD `be75ca0` **已包含**

| PR | 标题 | 合入日期 | 对本 bug 作用 |
| --- | --- | --- | --- |
| [#398](https://github.com/kvcache-ai/Mooncake/pull/398) | 引入软件层 60s transfer 超时机制的原始 PR | 较早 | **60s 硬等待的出处** |
| [#1560](https://github.com/kvcache-ai/Mooncake/pull/1560) | 修 bootstrap RPC 在握手阶段的 re-entrancy 死锁 | 2026-02-27 | 去掉启动窗口的一类死锁 |
| [#1705](https://github.com/kvcache-ai/Mooncake/pull/1705) | avoid resetting RDMA endpoint on duplicate concurrent bootstrap | 2026-03-23 | **修 "reusing connection" 方向的根因**：并发 `connect()` 把已建立的 endpoint reset 掉 |
| [#1733](https://github.com/kvcache-ai/Mooncake/pull/1733) | Fix simultaneous open handshake in RdmaEndpoint | 2026-03-30 | **修 RDMA simultaneous-open 竞态导致的 endpoint use-after-delete**（机制 ③ 的上游修复）|
| [#1762](https://github.com/kvcache-ai/Mooncake/pull/1762) | 修握手连接建立时的死锁 | ~2026-03-29 | 相邻修复 |
| [#1803](https://github.com/kvcache-ai/Mooncake/pull/1803) | Fix duplicate notify recv WR posting and PLOG misuse | 2026-04-02 | 修握手路径 duplicate notify recv WR + 误导性 `Success [0]` 日志。家族 2 + 家族 4 都受益 |

这 5 条 2026-Q1 的根因补丁**已集中在本地 HEAD**——意味着 prefill6 主 run（2026-04-14 采集）**应该已经有修复，但观察到的握手风暴仍然严重**。说明 ①②④ 三个机制不是靠 #1705/#1733 就能完全消除的。

### 6.2 本地 HEAD **不包含**（相关 open issues，未闭环）

| Issue | 标题 | 状态 | 与本 bench 关系 |
| --- | --- | --- | --- |
| [#204](https://github.com/kvcache-ai/Mooncake/issues/204) | 最早的 `Failed to modify QP to RTR` ETIMEDOUT 报告 | open 2025-04 | 症状，未闭环 |
| [#1066](https://github.com/kvcache-ai/Mooncake/issues/1066) | 推理时同样的 `QP to RTR ... Connection timed out` | closed 2025-11 | 关闭但**没 merged 修复** |
| [#1088](https://github.com/kvcache-ai/Mooncake/issues/1088) | `Failed to complete transfers after 60 seconds` | open 2025-11 | 60s 超时，未闭环 |
| [#919](https://github.com/kvcache-ai/Mooncake/issues/919) | 60s timeout → `TRANSFER_FAIL` → `batch_put failed err -800` | open 2025-10 | **与本 bench 日志一字不差**，未闭环 |
| [#1151](https://github.com/kvcache-ai/Mooncake/issues/1151) | 多网卡 k8s 环境下 `packet mismatch` + `mark it inactive` | open 2025-12 | 场景类似但多网卡特化 |
| [#1766](https://github.com/kvcache-ai/Mooncake/issues/1766) | 云燧 (Yunsilicon) 网卡上的 `QP to RTR` | closed 2026-04 | 硬件特化问题，不通用 |

**注**：#919 的 problem statement 与本 bench 完全对应，但 2025-10 开到现在未闭环。家族 2 的上游 PR 目前都在"降低概率"而非"彻底消除"层面。

### 6.3 家族 2 小结

- 症状报告密集，2026-Q1 连续合入多条根因补丁（#1560 / #1705 / #1733 / #1803 / #1762）
- 通用的 "ETIMEDOUT on stale peer QP reused from previous job" 场景在 #1088 / #919 / #1151 仍然 open
- **场景 A 的机制 ① 和 ②**（metadata chicken-and-egg / 单线程 accept）**没有任何上游修复**
- **场景 B 的 fail-fast / circuit breaker / 健康检查**也**没有任何上游修复**

---

## 7. 本地修复状态

### 7.1 已完成（本地 HEAD 已含 2026-Q1 家族 2 补丁）

- `#1560 / #1705 / #1733 / #1762 / #1803` 已全部合入 → `grep` 本地 `Mooncake/` 能找到对应改动
- `#398` 的 60s 超时机制在 `transfer_task.cpp:317` 可见

### 7.2 缺失（按优先级）

| # | 缺口 | 对应场景 / 机制 | 需本地改 / 向上游提 |
| --- | --- | --- | --- |
| 1 | **benchmark 客户端 ramp-up / warmup** | 场景 A（机制 ①②③④）| vLLM / vigil 侧，本地改 |
| 2 | **vigil 启动错峰（1-2 秒）** | 场景 A（降低 O(N²) 同步度）| vigil serving 脚本 |
| 3 | **Bug 3.9.a Fix：`SubmitTransfers` replica 0 失败 continue 而非 break** | 所有场景 + `num_replica≥2` | Mooncake 本地 `client_service.cpp` |
| 4 | **Bug 3.9.b Fix：`WaitForTransfers` 用 quorum 语义** | 所有场景 + `num_replica≥2` | Mooncake 本地 `client_service.cpp` |
| 5 | **Fail-fast on inactive endpoint**（`SubmitTransfers` submit 前 check `active()`） | 场景 B（消除 60s 尾延迟）| Mooncake 本地 |
| 6 | **后台自动重建 inactive endpoint** | 场景 B（真正自愈）| Mooncake 本地，改动大 |
| 7 | **Circuit breaker / peer 黑名单** | 场景 B（防止反复撞同一坏 peer） | Mooncake 本地 |
| 8 | **连接健康检查 / heartbeat 线程** | 场景 B（主动发现坏 peer） | Mooncake 本地 |
| 9 | **握手链路 RPC 级 retry + backoff** | 场景 A 机制 ④ | Mooncake 本地 / 向上游提 |
| 10 | **握手 RPC listener 改多线程** | 场景 A 机制 ② | Mooncake 本地 / 向上游提 |
| 11 | **passive re-establish 时 markFailed 在途 slices** | **场景 C**（彻底消除 C 类 60s 等待） | Mooncake 本地改 `disconnectUnlocked`（`rdma_endpoint.cpp:388-417`）——在清 `wr_depth_list_[i]` 的同时，对每个 in-flight slice 调 `markFailed()`，让 batch 立刻 fail 而不是等外层 60s |

---

## 8. 行动项

### 8.1 短期（我们自己能做，按 ROI 排序）

0. 🔜 **立刻可做（零代码改动，对场景 C 立竿见影）**：设 `MC_SLICE_TIMEOUT=10` 环境变量，把场景 C 的 60s 硬等变 10s（见 §3.12）。注意：**只救场景 C**，对场景 A/B 无效（A/B 的 slice 从没到 `ts>0` 那一步）。代价是合法的慢 transfer（≥10s）会被误杀，需评估本 bench 实际 slice 耗时分布再定具体阈值
1. ✅ **已做**：Python `batch_put failed` 增强（req_id / tp_rank / elapsed / codes / failed_samples），让 Group B 分析得以成立
2. 🔜 **Bench 客户端 ramp-up**：先以低并发（4-8）跑 5-10s，给 Mooncake 完成首批 endpoint 握手的时间窗口，然后爬坡到目标并发
3. 🔜 **显式 warmup**：bench 开始前对每个目标 segment 做一次 1-byte 小包 transfer，把所有 endpoint 推到 RTS
4. 🔜 **vigil 启动错峰**：prefill 以 1-2 秒错峰启动，降低 O(N²) 同步建连压力
5. 🔜 **Bug 3.9.a + 3.9.b 本地 patch**（~30 行 C++）：消除 "replica 0 失败 = 整 op 失败" 的放大效应

### 8.2 中期

1. **Fail-fast on inactive endpoint**（修法 D）：`SubmitTransfers` submit 前 check `endpoint->active()`，inactive 立刻返回错误 → 60s 变秒级
2. **后台自愈 inactive endpoint**：worker 线程发现 inactive 时主动触发重新 handshake，不等下次请求
3. **调短 `wait_for` 冷启动窗口**：启动窗口 60s → 10s，配合上层重试；稳态再切回 60s
4. **跟进上游 Issues #919 / #1088 / #1151**：贴本 bench 证据，推动根因修复
5. **nsys profiling 加回来**：trace 握手失败 → 业务升级的精确时序

### 8.3 长期

1. **Fix C：quorum early-return + cancel 其它 in-flight 传输**（消 tail 延迟）
2. **Circuit breaker / peer 黑名单**（场景 B 专用）
3. **连接健康检查线程**（场景 B 专用，类似 TCP keepalive）
4. **握手 RPC listener 改多线程 + RPC 级 retry**（消除机制 ② ④）
5. **Mooncake 客户端 API 暴露 endpoint 健康状态**：vLLM 侧可以做更智能的 replica 选择

---

## 9. 诊断 checklist

oncall 在新的 60s 超时事件里按序检查：

0. **三场景快速区分**（最先做这步，再走下面细节）：
   ```bash
   grep -cE "packet mismatch|mark it inactive|Failed to modify QP to RTR|Connection timed out \[110\]" *.log
   grep -c "Re-establish connection.*passive" *.log
   grep -c "Outstanding work requests found" *.log
   ```
   - 上一条非零（A/B 类 error 签名多）→ **场景 A 或 B**，继续第 1-7 步
   - 上一条全零 + `Re-establish connection (passive)` 非零 + `Outstanding work requests found` 非零 → **场景 C**，直接跳到第 8 步
1. **Python 侧 `batch_put failed` 的时间戳分布**：
   ```bash
   grep "batch_put failed" *.log | awk '{print $1, $2}' | sort
   ```
   集中在 bench 起跑头 120s 内 → **场景 A**；bench 启动 N 分钟后才开始 → **场景 B**。
2. **`elapsed` 直方图**：`elapsed ≈ 60s` / `120s` / `180s` 的 peak 都应对应 N×60s。其他数值可能是别的 timeout。
3. **有无 `mark it inactive`？指向几个 peer？**
   - 多个不同 peer_nic_path → 场景 A
   - 反复出现同一 peer_nic_path → 场景 B，**去查该 peer**
4. **底层错误是 `packet mismatch` 还是 `Connection timed out [110]`？**
   - `packet mismatch`：机制 ③（simultaneous-open），场景 A 主线
   - `Connection timed out [110]`：ETIMEDOUT，场景 A 机制 ①② 或场景 B
5. **有无 `Received same peer QP numbers, reusing connection`？**
   - 有 → 自愈机制工作，属 Group A（无业务影响）
   - 无 → 自愈失败，属 Group B（业务升级）
6. **看 `num_replica` 配置**：=1 时任意单 replica 失败都是 op 失败（Bug 3.9.a/b 会放大）；≥2 时修 3.9.a/b 有立竿见影效果
7. **场景 B 的下一步**：登录涉事 peer 机器，检查 `mooncake_master` 进程、`rocepXsY` RNIC 状态（`ibv_devinfo`、`ibstat`）、`dmesg`、`journalctl` 最近 10 分钟有无 kernel error
8. **场景 C 的下一步**：
   - 看 `Re-establish connection (passive)` 的时间戳和对端：
     ```bash
     grep "Re-establish connection.*passive" *.log | awk -F'peer ' '{print $1, $2}' | sort -u
     ```
   - 数 `Outstanding work requests found` 的条数——大致等于被丢弃的在途 WR 数，也是后续受影响的 batch 数上限
   - 确认后续 60s timeout 数量 ≤ Outstanding WR 数（如果远大于，可能有多轮 re-establish，grep 可能漏了）
   - 缓解：设 `MC_SLICE_TIMEOUT=10` 立刻生效；治本等 §7.2 第 11 行 patch

---

## 10. 相关但不属于本类

- **`metadata not found` (HTTP 404)**：HTTP metadata server 段元数据查不到。与本类 QP 握手失败是**两条独立链路**。属 bug 类别 1。见 [bug_class_metadata_not_found_cn.md](bug_class_metadata_not_found_cn.md)。
- **`OBJECT_NOT_FOUND(-704)`**：Master Service 对象元数据查不到。与本类无关。属 bug 类别 3。见 `bug_class_batch_put_transfer_fail_cn.md`。
- **可观测性缺口**（`-1 vs -800` 统计错误、缺 `req_id / tp_rank / elapsed` 字段）：见 bug 类别 3 的观测方案章节。
- **注意**：bug 类别 1 的 "原因 9 同节点重启 RDMA 参数变"**实际归属本类**（不是 metadata not found，是 QP handshake 失败）。
- **注意（场景 C）**：passive re-establishment 在日志上**完全不产生** A/B 的任何 error 签名（无 `packet mismatch` / `mark it inactive` / `Failed to modify QP to RTR` / `Connection timed out [110]`），但症状同样是 60s 超时。一定要按 §9 第 0 步先做三场景区分再往下走，否则按场景 A/B 的 checklist 查会查不到东西、误以为没事。

---

## 11. 验证命令 + 可观测性埋点路线

### 11.1 grep 验证命令

```bash
# 本地已含家族 2 修复 PR
cd Mooncake && git log --oneline --all | grep -E "1560|1705|1733|1803|1762|398"
# 预期 6 条全部命中

# ETIMEDOUT 错误点
grep -n "Failed to modify QP to RTR" mooncake-transfer-engine/src/transport/rdma_transport/rdma_endpoint.cpp
# 预期 :646

# packet mismatch 错误点
grep -n "packet mismatch" mooncake-transfer-engine/src/transport/rdma_transport/rdma_endpoint.cpp
# 预期 :283（或附近）

# 60s wait_for ★
grep -n "wait_for" mooncake-store/src/transfer_task.cpp
# 预期 :297（或 :341，不同版本行号略有差异）

# mark it inactive
grep -n "mark it inactive" mooncake-transfer-engine/src/transport/rdma_transport/worker_pool.cpp
# 预期 :244 或 :245

# 场景 C 核心：passive re-establishment + Outstanding WR 丢弃
grep -n "Re-establish connection\|Outstanding work requests found" mooncake-transfer-engine/src/transport/rdma_transport/rdma_endpoint.cpp
# 预期 :325（Re-establish 日志）+ :124（deconstruct 的 Outstanding）+ :406-407（disconnectUnlocked 的 Outstanding）

# 场景 C 机械根因：disconnectUnlocked RTS→RESET 直转
grep -n "IBV_QPS_RESET" mooncake-transfer-engine/src/transport/rdma_transport/rdma_endpoint.cpp
# 预期 :394 附近（attr.qp_state = IBV_QPS_RESET）

# slice_timeout 半成品机制（三场景共同上层）
grep -n "slice_timeout" mooncake-transfer-engine/src/multi_transport.cpp
# 预期 :163-176（getTransferStatus 的 slice 级 TIMEOUT 判定）
grep -n "slice_timeout" mooncake-transfer-engine/include/config.h
# 预期 :54（默认 -1）
grep -n "MC_SLICE_TIMEOUT" mooncake-transfer-engine/src/config.cpp
# 预期 :241-245（环境变量读取）

# client_service.cpp 两个放大 bug
grep -n "all_transfers_submitted\|all_transfers_succeeded" mooncake-store/src/client_service.cpp
# 预期可见 1531-1568 和 1572-1614 附近

# 握手 RPC 单线程 listener
grep -n "handshake_listen_backlog\|SO_RCVTIMEO" mooncake-transfer-engine/src/transfer_metadata_plugin.cpp
```

### 11.2 Phase 1 — 零改动验证（先跑，决定是否真需要改 Mooncake 源码）

只用现有日志就能立刻回答的四个问题：

| # | 问题 | 怎么查 | 若假设为真则应看到 |
|---|------|-------|---------------------|
| P1.1 | 失败真是集中在 bench 起跑前 60-120s？ | 把 `batch_put failed` 的 Python 日志按时间排序画直方图 | 起跑后 [0, 120s] 密集，之后几乎为零 |
| P1.2 | SIEVE eviction 是否在发生？ | `grep "evicted"` Mooncake worker 日志 | **该关键词应为 0**（若 >0 说明 SIEVE 路径也牵扯进来） |
| P1.3 | `packet mismatch` / `mark it inactive` / `Failed to modify QP to RTR` 是否在同一时间窗密集爆发？ | `grep -E "packet mismatch\|mark it inactive\|modify QP to RTR\|Connection timed out.*110"` 按时间聚合 | 全部集中在 bench 起跑后 [0, 60s] 窗口 |
| P1.4 | Python 侧 `elapsed` 分布 | awk 提取 `elapsed=` 字段作直方图 | 尖峰在 60 和 120 附近 |

**可以写一个 30 行的 analysis 脚本**（沿用 `scripts/mooncake/probe_observability_logs.py` 风格）来一次性跑完 P1.1-P1.4 并输出 CSV/plot。**如果 P1.1-P1.4 全部印证，就已经对"是握手风暴"有 80% 把握**，不需要再改 Mooncake 源。

### 11.3 Phase 2 — 最小 Mooncake 源码改动（K-1 ~ K-5 埋点）

在现有可观测性方案（`mooncake_observability_plan_20260415_cn.md` 方案 G）基础上再补 5 个点：

#### K-1：`doSetupConnection` 入口/出口计时

`rdma_endpoint.cpp:569` 开头 + `:671` 返回前：
```cpp
// 入口
auto t0 = std::chrono::steady_clock::now();
LOG(INFO) << "doSetupConnection enter: peer_nic_path='" << peer_nic_path_ << "'"
          << " cur_state=" << status_;
// 出口（所有 return 前加一次）
auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
    std::chrono::steady_clock::now() - t0).count();
LOG(INFO) << "doSetupConnection exit: peer_nic_path='" << peer_nic_path_ << "'"
          << " outcome=" << (rc == 0 ? "success" : "fail")
          << " rc=" << rc
          << " duration_ms=" << elapsed_ms
          << " final_state=" << status_;
```
**信号**：握手成功的 duration 约 10-50ms；失败的如果 duration 接近 3000ms / 60000ms，分别对应 metadata HTTP 超时 / RPC SO_RCVTIMEO。

#### K-2：握手 RPC listener 的 accept 速率和"等待时长"

`transfer_metadata_plugin.cpp:728-827` 的 listener 循环里：
```cpp
auto accepted_at = std::chrono::steady_clock::now();
int conn_fd = accept(listen_fd_, ...);
auto accept_returned_at = std::chrono::steady_clock::now();
// 处理完一次 connection 前后再各记一次
...
LOG(INFO) << "handshake_rpc accepted_and_processed"
          << " accept_wait_ms=" << ms(accept_returned_at - last_loop_top)
          << " process_duration_ms=" << ms(done_at - accept_returned_at);
```
**信号**：bench 起跑瞬间 `process_duration_ms` 会明显拉长，说明 listener 在排队；`accept_wait_ms` 在起跑瞬间应接近 0（持续有待处理连接）。

#### K-3：Metadata server 查询埋点

`transfer_metadata.cpp` 的 `getSegmentDescByName` 入口/出口：
```cpp
LOG(INFO) << "getSegmentDescByName: peer='" << peer_server_name << "'"
          << " cache_hit=" << cache_hit
          << " duration_ms=" << duration_ms
          << " outcome=" << (desc ? "found" : "not_found");
```
**信号**：bench 起跑时 `cache_hit=false` 且 `outcome=not_found` 的比例应显著升高，这直接证实 chicken-and-egg 假设（机制 ①）。

#### K-4：endpoint 生命周期 + 删除原因归因

`endpoint_store.cpp` 里两个不同的删除路径必须**区分标签**：

```cpp
// SIEVEEndpointStore::evictEndpoint (line 210)
LOG(INFO) << "endpoint_evicted reason=sieve_capacity"
          << " peer_nic_path='" << victim << "'"
          << " store_size=" << endpoint_map_.size()
          << " max_size=" << max_size_;

// SIEVEEndpointStore::deleteEndpoint (line 170)
LOG(INFO) << "endpoint_deleted reason=explicit_delete"
          << " peer_nic_path='" << peer_nic_path << "'"
          << " caller_context=<注入 worker_pool 调用者上下文>";
```
**信号**：bench 起跑期 `reason=explicit_delete` 应 ≫ `reason=sieve_capacity`。如果 sieve 那条也高，说明还有 SIEVE 压力（需另外考虑 `max_endpoints` 调大）。

#### K-5：QP 状态机每步计时

`rdma_endpoint.cpp:569-671` 的 `doSetupConnection` 里，三次 `ibv_modify_qp` 各自计时并打点：
```cpp
// RESET→INIT / INIT→RTR / RTR→RTS 三次前后
LOG(INFO) << "qp_state_transition"
          << " from=" << from_state << " to=" << to_state
          << " duration_ms=" << elapsed_ms
          << " rc=" << rc
          << " errno=" << errno_value;
```
**信号**：正常情况下每步都是亚毫秒级；失败时 `to=RTR` 的 `rc=EINVAL`/`errno=110` 是最常见的故障签名。

### 11.4 Phase 3 — 聚合指标（可选但强烈推荐）

把 K-1~K-5 的结构化日志同时 emit 成 Prometheus counters/histograms：

- `mooncake_endpoint_created_total{peer_nic_path}`
- `mooncake_endpoint_deleted_total{reason=sieve|explicit|idle}`
- `mooncake_handshake_duration_ms_bucket{outcome=success|timeout|packet_mismatch|metadata_missing}` (histogram)
- `mooncake_metadata_lookup_not_found_total`
- `mooncake_qp_transition_duration_ms_bucket{from, to, outcome}`

Grafana 看板只要一个图：上方 panel 画 bench 启动时间轴，下方 panel 把这五个指标叠在一起；"bench 起跑 → 指标尖峰 → 60s 超时连锁" 会一眼看出。

### 11.5 Phase 4 — 受控实验（A/B 验证）

有了 Phase 1 + 2 的数据后，跑三组对比：

| 实验 | 做法 | 假设正确时的预测 |
|------|------|------------------|
| E1 基线 | 直接跑 bench | 复现现象：[0, 120s] 内 `elapsed≈60n` 失败集中；K-2/K-3/K-5 指标尖峰 |
| E2 客户端 ramp-up | 先 5 并发跑 10s 再升到目标并发 | `batch_put failed` 集中窗口消失或大幅缩短；K-2/K-3 尖峰显著压低 |
| E3 显式 warmup | 开跑前对每个 peer 发 1 byte 预热 | `elapsed≈60n` 的失败消失；K-1 duration 分布 trailing-tail 消失 |

若 E2/E3 成功压制 E1 的现象 → **场景 A 根因确认**（是冷启动握手问题）。
若 E2/E3 对现象无影响 → 根因另查（可能是场景 B 或 SIEVE 压力，需看 K-4 的 `reason=sieve_capacity` 计数）。

### 11.6 Phase 5 — 最小可执行 checklist（给执行人用）

1. 写分析脚本 `scripts/mooncake/analyze_60s_timeout.py`：跑 P1.1-P1.4
2. 先跑一次 bench 作基线，用脚本跑一遍 Phase 1 → 决定是否进 Phase 2
3. 如果 Phase 1 数据不够确定，再 patch Mooncake 实装 K-1~K-5
4. 重启 Mooncake，跑 Phase 4 的 E1/E2/E3 三组
5. 出一份 markdown 报告说 "X 现象已被指标 Y 证实 / 未被证实"，就此 close 或 re-open 根因问题

---

## 12. 来源文档

| 文件 | 贡献章节 |
| --- | --- |
| `/home/aoshen/.claude/plans/sparkling-crafting-neumann.md`（60 秒超时根因分析 · 详细版） | **作为本文骨架**：§3（整个根因分析章节）+ §4.6（60s 其他来源排除表）+ §9 诊断 checklist + §11 可观测性 Phase 1-5 全部 |
| `prefill6_mooncake_error_analysis_20260415.md` + `prefill6_mooncake_error_analysis_20260415_cn.md` | §1 摘要的 prefill_6 证据 + §2.1 场景 A 症状链 + §4.1 prefill_6_mtc_18 主 run 全量证据（表格 + 热点主机 + 60s timeout 映射）+ §7 已完成修复列表 |
| `autosweep_20260417_135116_bug_census_cn.md` | §2.2 场景 B 症状链（ETIMEDOUT）+ §3.6 ETIMEDOUT 新机制 + §3.9 两个放大 bug + §4.2 跨 sweep 统计（Group A/B）+ §4.3–4.4 代表性时间线 + §7 缺失修复 + §8 行动项 |
| `mooncake_upstream_issues_prs_20260419_cn.md` 家族 2 章节（§427-453） | §6 整个上游 PR 状态章节（本地已含 / 未含 / 相关 issues 全部） |

**未来新证据追加指引**：
- 新 bench run 出现握手失败 → 在 §4 添加证据小节 `§4.N: <run_name>`，明确标注场景 A 或 B
- 新失败机制（机制 ⑤+） → 在 §3.5 添加
- 上游有新 PR → 更新 §6；若 cherry-pick 到本地，移到 §6.1 并写 commit 哈希
- 本类缺口被填上 → 更新 §7 + §8 相应行，保留历史记录
- 新的 60s 超时来源（非握手类）→ 在 §4.6 排除表追加
