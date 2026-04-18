# vLLM vs SGLang: RL 训练支持深度对比分析

> 分析时间: 2026-04-08
> 代码版本: 本地 fork (verl-grounding/{vllm, sglang, slime})
> 数据来源: 源码分析 + GitHub Issues/PRs (vllm-project/vllm)

---

## 一、总览

| 维度 | SGLang | vLLM | 差距 | vLLM 弥补进度 |
|------|--------|------|------|---------------|
| 权重同步方式 | 3 路径 (disk/tensor/distributed) | 2 引擎 (NCCL/IPC) | 中等 | IPC 单 GPU 已合并，多 GPU #37476 进行中 |
| Co-locate 零拷贝 | `update_weights_from_tensor` | IPC engine (CUDA IPC handle) | 已有基础 | #34171 已合并，#37476 补多 GPU + chunked |
| 生成暂停/恢复 | pause + continue (3 mode) | pause + resume (3 mode) | 对等 | #32351 keep 模式已合并 |
| 内存管理 | release/resume + tags | sleep/wake | 中等 | CuMemAllocator 有 tag 但 CUDA graph 直接丢弃 |
| 权重版本追踪 | weight_version 参数 | 无 | **严重** | 无相关 PR |
| 会话路由 | consistent hashing | 无 | **严重** | 无相关 PR |
| 确定性推理 | `--enable-deterministic-inference` | Batch Invariance | 已追平 | #27433 大部分完成 |
| MoE 路由回放 | return_routed_experts | return_routed_experts | 对等 | #28284 已合并 |
| 负载均衡网关 | SGLang Model Gateway | 需外部 LB | **严重** | 无计划 |
| FP8 在线量化重载 | 基础支持 | QeRL 分层加载 | **vLLM 领先** | #32133 #38032 已合并 |
| DP+EP 架构 | 独立 MoE_DP + MoE_EP 组 | DPEP 锁步 + wave | 设计不同 | 各有取舍 |

---

## 二、权重同步

### 2.1 SGLang: 三条路径

**路径 1: `POST /update_weights_from_disk`** — 磁盘加载，最简单，适合弹性扩缩容。支持 `weight_version`、`flush_cache`、async 加载。

**路径 2: `POST /update_weights_from_tensor`** — 零拷贝内存更新，co-locate 核心。使用 `FlattenedTensorBucket` 将多 tensor 打包为单一 buffer，通过 `MultiprocessingSerializer` 走 GPU IPC 直传，零磁盘 I/O。支持 `direct` 和 `flattened_bucket` 两种 load_format。

**路径 3: `POST /update_weights_from_distributed`** — NCCL 广播，disaggregated 场景。三步走：`init_weights_update_group` → `update_weights_from_distributed` → `destroy_weights_update_group`。支持 `flattened_bucket` 优化。

### 2.2 vLLM: 可插拔 Weight Transfer Engine

**NCCL 引擎** (#31943 已合并): `POST /init_weight_transfer_engine` + `POST /update_weights`。支持 packed tensor broadcasting（双/三缓冲），基于 NeMo-RL 实现。

**IPC 引擎** (#34171 已合并): 训练进程调用 `reduce_tensor(weight)` 将 GPU tensor 转为 CUDA IPC handle，vLLM 调用 `rebuild_cuda_tensor(*args)` 在同一张 GPU 上直接重建 tensor 引用。同 GPU 零拷贝，不经 CPU 不经磁盘。关键代码 (`ipc_engine.py:182-191`):

```python
handle = ipc_handle[physical_gpu_id]
func, args = handle
list_args = list(args)
list_args[6] = device_index  # 修正逻辑 GPU index
weight = func(*list_args)    # rebuild_cuda_tensor → 同 GPU 零拷贝
```

**当前限制**: IPC 引擎只支持单 GPU co-locate。多 GPU FSDP + chunked packed 传输在 #37476 进行中。

**自定义引擎**: 通过 `WeightTransferEngineFactory.register_engine()` 注册。

### 2.3 对比

| | vLLM IPC Engine | SGLang update_weights_from_tensor |
|---|---|---|
| 传输方式 | CUDA IPC handle (`reduce_tensor`/`rebuild_cuda_tensor`) | `FlattenedTensorBucket` + `MultiprocessingSerializer` |
| 零拷贝 | 同 GPU 直接共享内存页 | 是，但走一层序列化 |
| 多 GPU | 仅单 GPU (#37476 进行中) | 支持 |
| Chunked | 一次全量 (#37476 进行中) | FlattenedTensorBucket 天然分桶 |
| weight_version | 无 | 每次更新可携带版本号 |

---

## 三、生成控制: 暂停/恢复

### 3.1 SGLang

```
POST /pause_generation   mode=abort|retract|in_place
POST /continue_generation
```

- **abort**: 立即终止所有 pending 请求（默认）
- **retract**: 将 running 请求移回 waiting 队列，KV cache 刷掉腾出内存，请求元数据保留。`continue_generation` 后 scheduler 自动重新 prefill 并继续 decode，对上层 RL 框架完全透明。适配 APRIL 论文的长尾缓解模式。
- **in_place**: 暂停但不改变请求状态，依赖 KV cache 保留

### 3.2 vLLM

```
POST /pause?mode=abort|wait|keep
POST /resume
```

- **abort**: 立即终止 in-flight 请求
- **wait**: 等待 in-flight 请求完成后暂停
- **keep** (#32351): 冻结请求队列和调度循环，恢复时继续

进行中: `/pause/step` (Draft #31741, Meta TorchRL) — 到达 step boundary 即暂停，不等 in-flight 请求，明确允许 KV cache 包含多权重版本条目。

### 3.3 retract vs abort vs keep

retract 和 abort 的本质区别不是"停了重新推理":

| | abort | retract | keep |
|---|---|---|---|
| 请求去向 | 返回给调用方 | 留在 scheduler 内部，移回 waiting queue | 冻结在 running queue |
| 上层感知 | 需处理重试 | 完全透明 | 完全透明 |
| KV cache | 按配置处理 | 刷掉（腾内存），resume 后自动 re-prefill | 保留 |
| 适用场景 | 快速停止 | 多轮 RL 长尾缓解 | 快速权重更新 |

---

## 四、内存管理: Sleep/Wake vs Release/Resume

### 4.1 pause 和 sleep 是正交机制

| | pause | sleep |
|---|---|---|
| 解决的问题 | 停止调度循环，防止推理和权重更新并发 | 释放 GPU 显存，腾给训练用 |
| 开销 | 几乎零，设 flag 停止 step 循环 | 重：权重 offload 到 CPU，KV cache 丢弃，CUDA graph 处理 |
| 恢复开销 | 零，清 flag 继续跑 | 重：权重拷回，可能需要 re-capture CUDA graph |
| 请求状态 | keep 模式保留请求和 KV cache | 全部丢失 |
| 适用场景 | disaggregated（训练和推理在不同 GPU） | co-locate（训练和推理共享 GPU） |

Co-locate 全流程: `pause` → `sleep` → 训练 → `wake_up` → `update_weights` → `resume`

Disaggregated 全流程: `pause` → `update_weights` → `resume`（不需要 sleep）

### 4.2 SGLang: torch_memory_saver

```
POST /release_memory_occupation  tags=["kv_cache", "weights", "cuda_graph"]
POST /resume_memory_occupation   tags=["kv_cache", "weights", "cuda_graph"]
```

需要 `--enable-memory-saver` flag。底层使用 `torch_memory_saver` 库，核心技巧：**释放 CUDA graph 的物理显存，但保留虚拟地址映射**。resume 时只需重新分配物理页，CUDA graph 可直接复用无需 re-capture。三种 tag 完全独立控制。

### 4.3 vLLM: CuMemAllocator

```
POST /sleep?level=1&mode=abort
POST /wake_up?tags=tag1&tags=tag2
```

底层使用 `CuMemAllocator` (`cumem.py:178-226`)：
- `offload_tags` 中的内存 → 备份到 CPU pinned memory (cudaMemcpy)
- 其余内存 → **直接 `unmap_and_release`，不备份，物理页释放**
- Level 1: `offload_tags=("weights",)` — 只有 weights 备份到 CPU
- **CUDA graph 和 KV cache 直接丢弃**

### 4.4 核心差距: CUDA graph 恢复代价

| | SGLang | vLLM |
|---|---|---|
| CUDA graph 处理 | `torch_memory_saver` 保留虚拟地址，释放物理页 | 直接 unmap + release，完全丢弃 |
| 恢复代价 | 重新映射物理页，无需 re-capture | **需要重新 capture CUDA graph**（昂贵） |
| Tag 粒度 | kv_cache / weights / cuda_graph 三者独立 | offload_tags 控制哪些备份，其余全丢 |

在 RL 训练每个 step 都要 sleep/wake 的 co-locate 场景下，SGLang 的累积优势非常显著。

---

## 五、分布式推理架构: DPEP vs 独立 DP+EP

这是两者最大的架构差异，直接影响 pause/resume 的实现方式。

### 5.1 vLLM: DPEP 锁步架构

vLLM 部署 MoE 模型时使用 DPEP (Data-Parallel Expert-Parallel) 架构，所有 DP rank **严格锁步**:

```
Engine 0:  step 1 → step 2 → ... → step 32 [AllReduce] → step 33 → ...
Engine 1:  step 1 → step 2 → ... → step 32 [AllReduce] → step 33 → ...
Engine 2:  step 1 → step 2 → ... → step 32 [AllReduce] → step 33 → ...
```

**关键机制** (`core.py:1696-1758`):

1. **每个 step 都锁步**: MoE All-to-All 本身是 barrier，每个 forward 隐式同步
2. **没有请求的 engine 跑 dummy batch**: `execute_dummy_batch()` 执行一个**真正的 model forward**（空输入），确保参与 All-to-All 集体通信
3. **每 32 步 AllReduce**: 检测全局是否还有未完成请求，决定 wave 是否结束
4. **Wave 机制**: wave 内所有 engine 锁步运行，wave 间所有 engine 同步停止

**Pause 在 DPEP 下的实现** (#34125): 非常简洁，不需要额外的 DP 协调协议:

```python
# scheduler.py:368-370 — pause 时 token_budget = 0，不调度真实请求
if self._pause_state == PauseState.PAUSED_ALL:
    token_budget = 0

# core.py:1716-1723 — 没有执行真实请求，但 wave 内继续跑 dummy batch
if not executed:
    if not local_unfinished_reqs and not self.engines_running:
        continue              # wave 间直接跳过
    self.execute_dummy_batch()  # wave 内执行 dummy forward 参与 All-to-All
```

被 pause 的 engine 不调度真实请求，但继续 `execute_dummy_batch()` 参与 All-to-All，所有 DP rank 保持锁步不死锁。

RFC #32103 曾提出复杂的 PAUSE_DP / RUN_UNTIL_STEP 协调协议，但实际实现 (#34125) 通过 scheduler 层的 `token_budget = 0` + dummy batch 简洁地解决了问题。#34544 (DPEP pause PR) 被关闭，因为核心功能已在 #34125 完成。

### 5.2 SGLang: 独立 DP + EP 组

SGLang 的 DP 和 EP 是**独立的并行组**:

- **MoE_DP 组**: 每个 DP rank 有独立的 scheduler 进程，**非锁步**，独立接收和调度请求
- **MoE_EP 组**: EP 层内的 all-to-all 通信
- **MoE_TP 组**: MoE 内的 tensor parallel

与 vLLM 的关键区别:

| | vLLM DPEP | SGLang MoE_DP + MoE_EP |
|---|---|---|
| 调度方式 | 所有 DP rank 锁步 | 每个 DP rank 独立 |
| Wave 机制 | 有，dummy batch 保持同步 | 无 |
| Pause 协调 | 自动（scheduler 层） | 需外部逐个调用每个 engine |
| 设计哲学 | 正确性优先（集体操作不会卡死） | 灵活性优先（各 rank 可独立扩缩） |

SGLang 的 pause_generation 是**单 engine 语义**，多 DP 场景需要应用层自行协调（逐个调用每个 engine 的 `/pause_generation`）。

### 5.3 各有取舍

- **vLLM DPEP**: 锁步保证正确性，但灵活性差。dummy batch 是浪费（空跑 forward），但保证 All-to-All 不卡死。
- **SGLang 独立 DP+EP**: 灵活，各 rank 可独立扩缩和调度，但 pause 时需要应用层协调，且 EP all-to-all 的同步方式依赖具体实现。

---

## 六、多轮生成与会话管理

### 6.1 SGLang 的优势

**Radix Tree 前缀缓存** (`radix_cache.py`): system prompt + context 在 rollout 之间自动共享，天然支持 RL 轨迹分支（同一前缀 → 多个 continuation），无需显式 fork/rollback。

**Session Controller** (`session_controller.py`): 支持 streaming/non-streaming sessions，请求替换/追加/丢弃，父子关系用于 rollout 分支。

**Consistent Hashing 路由**: SGLang Model Gateway 通过 `X-SMG-Routing-Key` header 实现会话亲和性，确保多轮对话始终路由到同一服务器。

### 6.2 vLLM 的现状

`StreamingInput` 异步生成器模式可实现迭代生成。有基础前缀缓存但非 radix tree。无会话路由，无分支语义。Roadmap 标注 "Study a way to enable multi-turn long horizon scheduling"，无实质 PR。

---

## 七、确定性推理

### 7.1 SGLang

`--enable-deterministic-inference` flag，减少跨 batch shape 的非确定性。与 miles 框架配合实现 "True On-Policy" 训练，消除 training-inference mismatch。

### 7.2 vLLM: 已基本追平

Batch Invariance (#27433) 已完成: 基础框架、FlashInfer、DeepSeek-V3、DeepGEMM on Blackwell、CUDA Graph、TRITON_MLA。还缺 FLASHINFER_MLA、NVFP4、AMD、DP 支持。Roadmap 标记完成。

---

## 八、FP8 在线量化重载 (vLLM 领先)

这是 vLLM 反超 SGLang 的领域。

**QeRL 系列** (RFC #30359): 分层加载 + 分层量化 (#32133 已合并)，逐层加载权重后立即量化，避免 2x 内存峰值。CUDA graph 指针复用，量化后数据 copy 到原 tensor data pointer，不需 re-capture。组合在线量化与量化重载 (#38032 已合并)。

| | vLLM | SGLang |
|---|---|---|
| 分层加载 | LayerwiseLoader | 无 |
| 在线量化 | FP8 layerwise process | 基础支持 (`post_process_weights`) |
| 内存效率 | 峰值 = 量化模型大小 | 需全量加载后量化 |

---

## 九、负载均衡

**SGLang Model Gateway**: 内置 async non-blocking 路由、cache-aware 负载均衡、故障容错、consistent hashing 会话亲和性。已在 GLM 4.5+ 训练中验证。

**vLLM**: 无内置 RL 感知 router。有 "Simple data parallel router for scale out"（Roadmap 已完成），但这是 DP 层面的基础路由，不是 cache-aware 的。文档明确说需要外部 LB。

---

## 十、Slime 框架的 SGLang 依赖分析

Slime 作为强绑定 SGLang 的 RL 框架，其依赖分三级:

### 硬依赖（迁移 vLLM 需重大重写）

1. **`update_weights_from_tensor`** — co-locate 权重同步核心，使用 `FlattenedTensorBucket` + `MultiprocessingSerializer`。vLLM IPC engine 可类似但 API 不同且仅单 GPU。
2. **`pause_generation` / `continue_generation`** — 原子权重更新保障。vLLM 有 pause/resume 但端点不同。
3. **`release/resume_memory_occupation`** — GPU 内存管理，tags 精确控制。vLLM sleep/wake 语义不同。
4. **`init/update/destroy_weights_update_group`** — 分离部署权重同步。vLLM 有 NCCL weight transfer 但 API 不同。

### 中等依赖（可适配）

5. `/get_weight_version` — 权重版本校验
6. `/flush_cache` — 显式缓存刷新
7. `/health_generate` — 健康检查
8. `/post_process_weights` — FP8 量化后处理
9. `sglang_router` worker 注册/注销

### 轻度依赖（vLLM 已有等价）

10. `/generate` + `return_logprob` — 两者都支持
11. `return_routed_experts` — 两者都支持

---

## 十一、vLLM 弥补 RL 不足的 GitHub 活动

### Q4 2025 Roadmap RL 专项 (#26376)

| 项目 | 状态 |
|------|------|
| Full determinism and batch invariance | ✅ 完成 |
| Testing for popular integrations | ✅ 完成 |
| Simple DP router | ✅ 完成 |
| Enhance weight loading speed | ✅ 完成 |
| Custom checkpoint loader | 进行中 |
| Multi-turn long horizon scheduling | **未完成，仅 study** |

### 已合并关键 PR

| 时期 | PR | 内容 |
|------|-----|------|
| 2025-11 | #28037 | 基础 pause/resume (abort + wait) |
| 2025-11 | #28284 | MoE 路由回放 return_routed_experts |
| 2026-01 | #31943 | NCCL Weight Transfer Engine |
| 2026-01 | #32133 | QeRL 分层加载 |
| 2026-01 | #32351 | pause keep 模式 |
| 2026-02 | #34125 | pause/resume 移入 engine 层（DPEP pause 自然支持）|
| 2026-02 | #34171 | IPC Weight Transfer Engine |
| 2026-03 | #36188 | RL 文档完善 |
| 2026-03 | #38032 | 在线量化 + 重载组合 |

### 进行中关键 PR/RFC

| PR | 内容 | 状态 |
|----|------|------|
| #37476 | IPC 多 GPU + chunked packed | 活跃 review |
| #31741 | `/pause/step` 快速暂停 (Meta TorchRL) | Draft |
| #31848 | 统一权重同步 API 设计 (Slime 团队参与) | RFC 讨论中 |
| #30359 | QeRL 在线量化重载 | 核心已合并 |

### 社区观察

- **Slime 团队 (@rizar) 直接参与 vLLM RFC** #31848，建议添加 bucketing/packing
- **Meta** 推动 `/pause/step` (#31741) 和 sharding metadata (#32066)
- **Anyscale** 主导 API 设计: pause/resume RFC、QeRL

---

## 十二、总结

### 已追平
1. 确定性推理 (Batch Invariance)
2. MoE 路由回放
3. 基础 pause/resume (abort + wait + keep)
4. NCCL 权重同步
5. Co-locate 零拷贝 (单 GPU IPC)
6. RL 文档
7. DPEP 场景 pause/resume

### 正在追赶
8. 多 GPU co-locate + chunked 传输 (#37476)
9. 统一权重同步 API (RFC #31848)

### 仍有明显差距
10. **权重版本追踪** — 无计划
11. **会话路由 (Consistent Hashing)** — 无计划
12. **内置 RL 负载均衡网关** — 无计划
13. **CUDA graph 感知内存管理** — wake 后需 re-capture，SGLang 不需要
14. **retract 暂停模式** — 对上层透明的请求回收，vLLM 无等价
15. **多轮长期调度** — 仅标记 study

### vLLM 领先
16. **FP8 在线量化重载 (QeRL)** — 分层加载 + 分层量化
17. **可插拔 Weight Transfer 架构** — 支持自定义引擎
18. **DPEP 锁步 pause** — scheduler 层自动协调，不需应用层介入
