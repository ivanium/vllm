# Mooncake 集成讨论 - 2026-04-07

## 参与者
- **乔一凡** — Mooncake 集成开发
- **马腾** — 架构讨论
- **yk** — SGLang 侧 / Mooncake 部署经验
- **p** — vLLM Ascend 侧
- **雷超** — Mooncake client 架构讨论

---

## 1. 进展更新（乔一凡）

- RDMA 已跑通，CPU offloading 基本没有明显性能下降
- Disk offloading 已测试，multi-turn 场景下 **80%+ 性能提升**

## 2. 问题讨论

### 问题一：多卡场景下 Mooncake client 架构

**现状**：当前代码基于 vllm-ascend 的方式，GPU worker 直接 embed 了 mooncake real client。单卡 CPU 和 disk offloading 没问题，但 **多卡会创建多个 real client，目前不支持**。

**SGLang 的做法（yk）**：用 dummy client 接 GPU worker，外部另起一个 mooncake real client worker。

**三种部署模式（yk 总结）**：

| 模式 | 架构 | 优点 |
|------|------|------|
| **1. 每 GPU 一个 real client** | 每个 real client 都持有 memory/ssd/nic 资源 | 设置和部署最简单 |
| **2. 每 GPU 一个 dummy client** | 本地所有 dummy client 共享一个 real client，real client 持有所有资源 | 只有 real client 持有资源，管理方便 |
| **3. 每 GPU 一个 real client，global segment=0** | 每个 real client 不持有 memory/ssd 资源，只持有 nic 资源；单独起一个 real client 持有 memory 和 ssd | Mooncake 的存储资源和 sglang 解耦，部署方便 |

**结论**：
- 乔一凡打算把 vllm 的 embed client 改成 dummy client 方式
- 马腾补充：需要支持 dummy client 特性，相当于 real client 管理 node 上所有资源，每个 dummy 对应每个 TP
- p 确认：vllm ascend 目前用多个 real client（每 worker 一个），mooncake 支持，后续演进方向也是 dummy real client 路线

**GPU Direct RDMA 问题（乔一凡）**：
- real worker 在单独 process 的话，没法直接注册 vllm worker 的 GPU memory
- 需要用 CUDA IPC 之类的方法做 GPU memory sharing
- 马腾确认：现在有方法做，就是上述 3 种部署模式

### 问题二：Disk Offloading 参数调优

**问题**：benchmark 时会报 `BatchPut failed due to insufficient space`

**yk 解答**：
- 这个错跟 disk 没关系，是内存里的空间不足
- 可能是 evict 的速度没跟上 put 的速度
- 建议把 **high water mark** 调低（默认 95%，改成 90% 或 80%）

**乔一凡测试**：
- 把 high water mark 调到 0.8，会好一些但还是没法完全跟上 put 速度
- 可能会调到大概 0.5

**进一步讨论**：

- **马腾建议**：evict 用两个 watermark 方式（类似 kernel mm），避免 evict 速率跟不上。或者用环境变量控制 eviction 速率，让它猛猛去 evict
- **乔一凡**：搞一个 low watermark 和一个 high watermark
- **yk 解释**：现在有两个参数控制——一个控制什么时候开始 evict，一个控制 evict 的比例

**内存容量问题**：
- 乔一凡开了 600GB CPU memory，EVICT_RATIO 是否也应该调大？
- yk 分析：确实跟 put 速率和内存容量有关。600GB memory × 0.8 = 480GB 时开始 eviction，evict 几十到几百毫秒就完成了，按理这么短时间内不会出现剩下的内存耗完
- yk 追问：往 mooncake 里写入的数据 size 都是一样的吗？如果一样的话内存碎片会很少，理论可存接近 600GB
- 乔一凡确认：用 llama-3-8B 测试的，感觉确实有点异常
- yk：如果数据 size 一样，他们这边也来试着复现

**后续**：乔一凡继续研究，如果明天还没搞定就把代码整理成脚本，让 yk 团队复现。

### 问题二补充：BatchPut 失败的根因（yk 后续跟进）

yk 和 @zzy 测了一下，发现另一种可能：
- **put 速度超过 SSD offloading 速度** → 大量内存数据来不及被 offload → 处于 offloading 状态的数据无法释放 → 新的 put 失败
- yk 确认：SSD offloading 这个 feature 刚开发完，确实问题还比较多

### 问题三：Mooncake C++ 层的三种部署方案详解（yk + 雷超）

雷超看到之前的三种部署模式讨论后，提了几个澄清问题，yk 做了非常详细的解释。

**yk 对 Mooncake C++ 代码层的精确描述**：

#### C++ 层的三个组件

```
┌─────────────────────────────────────────────────────────────────┐
│  1. Real Client (C++ class)                                     │
│     - 持有所有资源（memory, ssd, nic, transfer engine）           │
│     - 可以直接嵌入推理引擎进程                                     │
│     - 也可以作为独立进程运行                                       │
│                                                                  │
│  2. Dummy Client (C++ class)                                     │
│     - 什么资源都没有                                               │
│     - 只是嵌入推理引擎进程里的一个 CLI                              │
│     - 通过 IPC 把请求转发给 Real Client Server                    │
│     → 对应方案 2                                                  │
│                                                                  │
│  3. C++ Real Client Main                                         │
│     - 在 Real Client 外面包了一层                                  │
│     - 加了一个 RPC server，用于链接 Dummy Client 和 Master         │
│     - 必须和 Dummy Client 配合使用                                 │
│     → 对应方案 2                                                  │
│                                                                  │
│  4. mooncake.mooncake_store_service                              │
│     - 同样在 Real Client 外面包了一层                               │
│     - 可以作为独立进程启动，不嵌入推理引擎                            │
│     - 纯粹作为全局资源池的一部分                                     │
│     - 可以和推理引擎放同一台 server，也可以放其他 server              │
│     - 可以起多个                                                   │
│     → 对应方案 3                                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### 三种方案和 C++ 组件的对应关系

```
方案 1: 直接嵌入
┌──────────────────┐
│ vLLM Worker 进程  │
│  └── Real Client │  ← Real Client 直接嵌入，持有所有资源
│      (嵌入式)     │     global_segment 可以设为 0（不贡献存储）
└──────────────────┘     也可以设为 >0（同时当存储节点）

方案 2: Dummy + Real Client Main
┌──────────────────┐        IPC         ┌────────────────────────┐
│ vLLM Worker 进程  │  ←────────────→   │ Real Client Main 进程   │
│  └── Dummy Client│   (socket+SHM)    │  └── Real Client       │
│      (无资源)     │                   │  └── RPC Server        │
└──────────────────┘                   │      (链接 Dummy+Master)│
                                       └────────────────────────┘

方案 3: Real Client(seg=0) + Store Service
┌──────────────────┐                   ┌────────────────────────┐
│ vLLM Worker 进程  │    RDMA/TCP      │ Store Service 进程      │
│  └── Real Client │  ←────────────→   │  └── Real Client       │
│   (global_seg=0) │                   │  (持有 memory + ssd)   │
│   (只有 nic+TE)  │                   │  可以在同机或其他机器    │
└──────────────────┘                   └────────────────────────┘
```

#### Dummy Client 加 Transfer Engine？（雷超提问）

- 雷超问：后续是否有一种方式是给 dummy client 加上 transfer engine？
- yk 回答：dummy 加 TE，那就等于方案 3 里不持有存储资源的 real client 了
- **雷超补充**：比方案 3 更轻量，因为方案 3 每卡上的 real client 除了有 TE，还有不少额外线程开销（RPC server 等）
- yk 追问：额外线程开销是因为起了 RPC server 吗？
- 雷超确认：real client 和 dummy client 的区别，除了持有资源和创建 TE，还有其他线程创建
- **yk 解释**：如果要 DMA（GPU Direct RDMA），那只能是方案 1 或方案 2，因为 TE 需要能访问到 GPU memory
- 雷超提到阿里云的同学也在支持这个模式

#### 共享内存方案（雷超提出）

雷超提出了一种可能的 vLLM 实现思路：
- vLLM worker 注册一段内存（类似 SGLang 的 hicache）
- 这段内存和 real client 进程是共享内存
- real client 就可以 DMA 这段内存
- 这样针对 vLLM，应该在 vLLM worker 注册监控存，然后 real client 直接操作监控存

**yk 回应**：如果要 DMA，那只能是方案 1 或方案 2（real client 需要能访问 GPU memory）。

**后续**：下次周一会上可以再聊一下，阿里云的同学也在支持这个模式。

---

## 关键结论

1. **架构方向**：vLLM 侧从 embed real client 改为 dummy client + 外部 real client worker（方案 2）
2. **Disk offloading 调优**：调低 high water mark（0.8 → 0.5），考虑引入 low/high 双 watermark 机制
3. **BatchPut 失败根因**：put 速度 > SSD offloading 速度，导致内存数据无法及时释放。SSD offloading feature 尚不成熟
4. **GPU Direct RDMA 限制**：如果要 DMA，只能用方案 1（嵌入式 real client）或方案 2（dummy + real client main），方案 3（独立 store service）无法直接 DMA GPU memory
5. **潜在方案 4**：给 dummy client 加 TE（比方案 3 更轻量），阿里云同学在推进，下次周会讨论
