# Deep Analysis of the Kimi K2.5 Decode Trace

> **Trace**: `vigil/logs/pd_kimi_70k_nsys_decode/2026-04-12/20260412_152636/decode-0/traces/nsys_profile.nsys-rep`
> **日期**: 2026-04-12
> **硬件**: 4× NVIDIA GB200 (GB100 chip, SM 10.0, 192GB HBM3e), NVL72 NVSwitch
> **模型**: nvidia/Kimi-K2.5-NVFP4 (61 layers, MLA + MoE, FP4 experts)
> **部署**: P/D 分离 (prefill: gb200-rack1-15, decode: gb200-rack1-16), TP=4
> **KV Transfer**: NixlConnector (RDMA/NVSwitch)
> **投机解码**: EAGLE3 (num_speculative_tokens=3)
> **CUDA Graph**: `FULL_AND_PIECEWISE`（decode worker 实际生效配置；`REPRODUCE.md` 中命令写的是 `FULL_DECODE_ONLY`，但与 worker log 不一致）
> **编译优化**: `fuse_allreduce_rms=true`, `enable_sp=false`, `fuse_gemm_comms=false`, `-O3`

## 0. 配置校正与问题定位

这份 trace **没有启用 sequence parallelism**。因此你期待看到的
`all_reduce -> reduce_scatter + all_gather` 改写，本次运行里根本没有前置条件。

### 0.1 生效配置（以 decode worker log 为准）

`decode-0/decode-0-gb200-rack1-16.log` 中的最终 `VllmConfig` 明确显示：

```text
compilation_config.pass_config = {
  'fuse_norm_quant': False,
  'fuse_act_quant': True,
  'fuse_attn_quant': False,
  'enable_sp': False,
  'fuse_gemm_comms': False,
  'fuse_allreduce_rms': True,
}
cudagraph_mode = FULL_AND_PIECEWISE
data_parallel_size = 1
```

对应源码路径：

- `vllm/config/vllm.py`: `PostGradPassManager` 只有在 `pass_config.enable_sp=True` 时才会注册 `SequenceParallelismPass`
- `vllm/compilation/passes/pass_manager.py`: 本次实际只注册了 `AllReduceFusionPass`，没有注册 `SequenceParallelismPass`
- `vllm/config/vllm.py`: `-O3` 的默认值当前把 `enable_sp` 绑定到 `IS_DENSE`，而该常量在当前源码里被硬编码为 `False`

### 0.2 为什么 trace 里看到的是 allreduce，而不是 RS/AG

因为这次启用的是 **FlashInfer allreduce+rms 融合**，不是 sequence parallelism：

```text
Enabled custom fusions: act_quant, allreduce_rms
Initialized FlashInfer Allreduce norm fusion workspace with backend=trtllm
```

`nsys_profile.sqlite` 的 kernel 统计也和这个结论一致：

| kernel | 次数 | 总时间(us) |
|---|---:|---:|
| `allreduce_fusion_kernel_oneshot_lamport` | 73,272 | 2,293,541.4 |
| `ncclDevKernel_AllGather_RING_LL` | 2,272 | 47,228.5 |
| `ReduceScatter` 相关 kernel | 0 | 0 |

也就是说：

- 主 TP 通信是 `allreduce_fusion_kernel_oneshot_lamport`
- trace 里 **没有** `ReduceScatter`
- 少量 `AllGather` 存在，但不是“SP 成对出现的 AG/RS”

### 0.3 Kimi-K2.5 / DeepSeekV2 模型侧也没有走到 MoE sequence parallel

`KimiK25ForConditionalGeneration` 的文本主干是 `DeepseekV2ForCausalLM`
（见 `vllm/model_executor/models/kimi_k25.py`）。

而 `DeepseekV2MoE` 中的模型侧 sequence parallel 开关来自：

```python
self.is_sequence_parallel = parallel_config.use_sequence_parallel_moe
```

`use_sequence_parallel_moe` 的条件是：

- `enable_expert_parallel=True`
- `data_parallel_size > 1`
- `tensor_parallel_size > 1`
- `all2all_backend` 属于支持列表

本次运行里 `data_parallel_size=1`，且没有启用 `--enable-expert-parallel`，
所以这条路径也不会被触发。

更重要的是，`DeepseekV2MoE.forward()` 里还有一条源码注释：

```python
# TODO: We can replace the all_reduce at the end of attn with a
# reduce_scatter instead of chunking here.
```

这说明即便将来启用模型侧的 MoE SP，当前实现也**不是**
“attention 后的 allreduce 已经稳定地被直接替换成 reduce_scatter”这一版本。

### 0.4 对本文其余分析的影响

- `REPRODUCE.md`/`slurm.out` 中虽然同时出现了：
  - `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'`
  - `--compilation_config.pass_config.fuse_allreduce_rms true`
- 但 `vllm/utils/argparse_utils.py:parse_args()` 会把同一根键
  `--compilation-config` 的 dotted 形式重新组装成一个新的 JSON 参数，
  并在参数列表末尾追加回去；这样在 argparse 的最终结果里，后追加的
  `{"pass_config":{"fuse_allreduce_rms":true}}` 会整体覆盖前面的
  `{"cudagraph_mode":"FULL_DECODE_ONLY"}`，而不是做 deep merge。
- 这正好解释了为什么 `slurm.out` 里命令看似带了 `FULL_DECODE_ONLY`，
  但 worker 的 `non-default args` 和最终 `VllmConfig` 里它消失了。

- “`FULL_DECODE_ONLY` + mixed batch 必然 eager” 这一前提需要重审。
- decode worker 实际日志显示本次运行捕获了：
  - `Capturing CUDA graphs (mixed prefill-decode, PIECEWISE): 17`
  - `Capturing CUDA graphs (decode, FULL): 9`
- 因此 Phase 3/4 中把 mixed batch 全部解释为 eager 的部分，和实际生效配置不完全一致。

**结论**: 这次 trace 的“异常点”不是“SP 开了但没拆成 RS/AG”，而是
**SP 根本没开，开的是 allreduce+rms 融合**；同时文档里对 cudagraph mode 的前提也写偏了。

---

## 目录

1. [Trace 总览与时序分解](#1-trace-总览与时序分解)
2. [单层 Forward Pass 的 Kernel 映射](#2-单层-forward-pass-的-kernel-映射)
3. [机制一：Lamport 负零哨兵协议](#3-机制一lamport-负零哨兵协议)
4. [机制二：CUDA Graph 捕获跨 GPU 通信](#4-机制二cuda-graph-捕获跨-gpu-通信)
5. [机制三：FP4 E2M1 两阶段 GEMM 设计](#5-机制三fp4-e2m1-两阶段-gemm-设计)
6. [关键配置参数](#6-关键配置参数)
7. [Trace-Derived Insights](#7-trace-derived-insights)
3. [机制一：Lamport 负零哨兵协议](#3-机制一lamport-负零哨兵协议)
4. [机制二：CUDA Graph 捕获跨 GPU 通信](#4-机制二cuda-graph-捕获跨-gpu-通信)
5. [机制三：FP4 E2M1 两阶段 GEMM 设计](#5-机制三fp4-e2m1-两阶段-gemm-设计)
6. [关键配置参数](#6-关键配置参数)

---

## 1. Trace 总览与时序分解

### 1.1 总体统计

| 指标 | 值 |
|------|-----|
| GPU 活动范围 | 742ms - 6065ms (5.32s) |
| 总 Kernel 数 | 1,114,464 |
| Memcpy 事件 | 19,144 |
| NVTX 标注 | 3,104 |
| CUDA Graph 中 kernel 占比 | 95% (1,058,584 / 1,114,464) |

### 1.2 六阶段时序

```
0ms        252ms      740ms        2120ms       2935ms       4190ms         6065ms
 │          │          │             │            │            │              │
 ▼          ▼          ▼             ▼            ▼            ▼              ▼
┃ Phase 1  ┃ Phase 2  ┃  Phase 3   ┃  Phase 4  ┃  Phase 5  ┃   Phase 6    ┃
┃ 预热     ┃ nixl KV  ┃  首次 decode┃  context→ ┃  Graph    ┃   稳态       ┃
┃ 空跑     ┃ 传输     ┃  eager 模式┃  gen 过渡 ┃  replay   ┃   decode     ┃
┃          ┃ 62 blocks┃  (非 graph)┃          ┃  预热     ┃   ~138 steps ┃
┃ ~250ms   ┃ ~490ms   ┃  ~1380ms   ┃  ~815ms   ┃  ~1255ms  ┃   ~1875ms   ┃
```

> **注**: CUDA Graph 在初始化阶段已预先 capture (`gpu_worker.py:588` → `capture_model()`),
> 覆盖了所有 `cudagraph_capture_sizes` 配置的 batch size。Phase 3 不是 graph capture，
> 而是 `FULL_DECODE_ONLY` 模式下混合 batch (含 context 请求) 的 eager 执行。

**Phase 1 (0-252ms): 初始化空跑**

NVTX: `execute_context_0(0)_generation_0(0)` — 0 个 context 请求, 0 个 generation 请求。

Decode worker 在主循环中反复调用 `schedule()` → `execute_model()`，但还没有请求到达，
`num_scheduled_tokens=0`，走提前返回路径处理 KV connector 的轮询:

```
Engine core 主循环                     GPU Worker                              GPU Model Runner
─────────────────                     ──────────                              ────────────────
v1/engine/core.py:380                 v1/worker/gpu_worker.py:748             v1/worker/gpu_model_runner.py:3830
                                      
def step():                           def execute_model(sched_out):           # execute_model() 内部:
  sched_out = scheduler.schedule()      with self.annotate_profile(...):      if not num_scheduled_tokens:
  #→ sched_out.total_num_scheduled      # NVTX: "execute_context_0(0)          if not has_kv_transfer_group():
  #   _tokens = 0 (无请求)              #   _generation_0(0)"                     return EMPTY_MODEL_RUNNER_OUTPUT
  future = executor.execute_model(      return model_runner.execute_model(      return self.kv_connector_no_forward(
    sched_out)                            sched_out)                               sched_out)
  model_output = future.result()                                               # ↑ 走 KV connector 轮询路径
                                                                               # 检查是否有来自 prefill 的 KV
```

NVTX 标注由 `gpu_worker.py:715-739` 的 `annotate_profile()` 生成:
```python
# gpu_worker.py:726-738
annotation = "execute_context_"
    + str(num_ctx_requests)      # 0
    + "(" + str(num_ctx_tokens)  # 0
    + ")_generation_"
    + str(num_gen_requests)      # 0
    + "(" + str(num_gen_tokens)  # 0
    + ")"
# → "execute_context_0(0)_generation_0(0)"
```

---

**Phase 2 (252-740ms): nixl KV Cache 传输**

NVTX: `nixl_read_blocks_for_req=chatcmpl`, `nixl_read_blocks_n=62`, `nixl_xfer_done_req=chatcmpl-...`

Prefill 节点完成计算后通知 decode 节点，NixlConnector 发起 RDMA 读取 KV blocks:

```
NixlConnector._read_blocks_for_req()              NixlConnector._read_blocks()
────────────────────────────────────              ──────────────────────────────
nixl_connector.py:2509-2569                       nixl_connector.py:2675-2694

def _read_blocks_for_req(req_id, meta):           # 准备 NIXL 传输描述符
  nvtx.range_push(                                nvtx.range_push(
    f"nixl_read_blocks_for_req={req_id[:8]}")       f"nixl_read_blocks_n=
  for remote_rank in remote_ranks:                     {len(local_block_descs_ids)}")
    self._read_blocks(                             # → "nixl_read_blocks_n=62"
      req_id, meta,
      remote_rank, ...)                            handle = nixl_wrapper.make_prepped_xfer(
  nvtx.range_pop()                                   "READ",           # RDMA READ
                                                     local_xfer_side,  # 本地 KV buffer
                                                     local_block_ids,  # 62 个 block 索引
                                                     remote_xfer_side, # 远端 prefill buffer
                                                     remote_block_ids)
                                                   nixl_wrapper.transfer(handle) # 异步启动
                                                   nvtx.range_pop()

NixlConnector.step() — 轮询传输状态
──────────────────────────────────
nixl_connector.py:2396-2406

for handle in handles:
  xfer_state = nixl_wrapper.check_xfer_state(handle)
  if xfer_state == "DONE":
    nvtx.range_push(f"nixl_xfer_done_req={req_id}")
    # → "nixl_xfer_done_req=chatcmpl-___prefill_addr_gb200-rack1-15:8000_..."
    nixl_wrapper.release_xfer_handle(handle)
    nvtx.range_pop()
```

Trace 实测: 首批 62 blocks 传输 360ms (252-612ms)，后续 3 组并发补传 53-58ms。

---

**Phase 3 (740-2120ms): 首次 Decode — Eager 模式 (非 CUDA Graph)**

NVTX: `execute_context_1(1)_generation_0(0)` — 1 个 **context** 请求 (KV 刚收完，首次 decode)。

KV 传输完成后，scheduler 首次将该请求调度。注意这里的 "context request" **不是指 prefill**
（prefill 已在 prefill 节点完成），而是指 **decode 节点还未为该请求产出过任何 token**
(`num_output_tokens == 0`, 见 `v1/core/sched/output.py:161-163`)。

**为什么 `num_output_tokens=0`？** 在 P/D 架构中，Router 将同一个请求分别发送给
prefill 和 decode 引擎。Decode 端通过 `Request.__init__` 创建请求对象时，
`_output_token_ids` 始终初始化为空列表 (`request.py:122`)，且
`from_engine_core_request()` (`request.py:185-201`) 的参数中不包含 output_token_ids。
Prefill 节点产出的第 1 个 token 由 router 直接返回给用户，不会回填到 decode 端。
因此 decode 端首次调度该请求时 `num_output_tokens=0`，处于 context phase。

首次 forward 后 decode 端产出第 2 个 token，`num_output_tokens` 变为 1，
下一轮即变为 "generation request"，开始使用预先 capture 好的 CUDA Graph。

`FULL_DECODE_ONLY` 模式下，含 context 请求的 batch 走 **eager 模式**，不使用 CUDA Graph:

```
CudagraphDispatcher.dispatch()                   CUDAGraphMode 选择逻辑
──────────────────────────────                   ──────────────────────
v1/cudagraph_dispatcher.py:301                   config/compilation.py:60

normalized_uniform =                             FULL_DECODE_ONLY = (FULL, NONE)
  uniform_decode                                 # decode_mode() = FULL  → 纯 decode 用 graph
  and cudagraph_mode.separate_routine()          # mixed_mode()  = NONE  → 混合 batch 不用 graph

# uniform_decode = False (因为有 context 请求)
# → normalized_uniform = False
# → dispatch 返回 mixed_mode() = NONE

gpu_model_runner.py:3877-3890
cudagraph_mode = CUDAGraphMode.NONE              # ★ Eager 模式!
                                                 # 不使用预先 capture 的 graph
gpu_model_runner.py:4020-4044
with set_forward_context(
    cudagraph_runtime_mode=CUDAGraphMode.NONE,   # 告诉所有层: 不使用 graph
    ...):
  model_output = self._model_forward(...)        # 61 层 eager forward
```

耗时 1378ms 因为:
1. **Eager 模式**: 每个 kernel 单独 launch，无 graph 优化 (~297 × 4us launch overhead)
2. **185ms allreduce 异常值**: Lamport 协议在 GPU 长时间空闲后首次 barrier，
   GPU 时钟不同步导致 spin-wait 极长
3. **cuBLAS/cuDNN plan 重选**: 长时间空闲后部分 handle 需要重新选择最优 plan

> **CUDA Graph 什么时候 capture 的？** 在初始化阶段 (profiling 开始前):
> ```
> gpu_worker.py:577-588
> for size in sorted(warmup_sizes, reverse=True):
>     model_runner._dummy_run(size)              # 预热各 batch size
> kernel_warmup(self)                            # kernel autotuning
> cuda_graph_memory_bytes = model_runner.capture_model()  # ★ 预先 capture 所有 graph
> ```
> `capture_model()` 遍历 `cudagraph_capture_sizes` 的所有 batch size，
> 对每个调用 `_warmup_and_capture()` 进行 warmup + capture。
> 这些 graph 在 Phase 5/6 的纯 decode batch 中被 replay。

---

**Phase 4 (2120-2935ms): Context → Generation 过渡**

Phase 3 执行完首次 decode 后，请求从 "context" 状态变为 "generation" 状态。
后续 scheduler step 中，该请求被标记为 generation request (`num_gen_requests=1`)。

此阶段的间歇性 kernel 活动 (302 个 kernel) 来自:
- 后续几轮 eager 模式 decode (请求从 context 转 generation 的过渡期)
- Triton kernel 首次编译 (JIT compile, 非 CUDA graph 相关)
- 可能的 EAGLE3 draft model 的首次执行

一旦 batch 变为纯 generation (0 个 context request)，dispatcher 返回
`decode_mode() = FULL`，开始使用初始化时预先 capture 好的 graph。

---

**Phase 5 (2935-4190ms): Graph Replay 预热**

NVTX: `execute_context_0(0)_generation_1(4)` — 0 个 context, 1 个 generation 请求 (4 tokens = 1 verified + 3 speculative)。

再次调用 `execute_model()` 时, batch descriptor 已存在, CUDAGraphWrapper 走 **replay 路径**:

```
CUDAGraphWrapper.__call__()
──────────────────────────
compilation/cuda_graph.py:341-356

entry = entries[batch_desc]

if entry.cudagraph is not None:       # ★ REPLAY 路径 (后续所有执行)
  # 验证 input tensor 地址未变 (debug mode)
  get_offloader().sync_prev_onload()
  entry.cudagraph.replay()            # 1 次 API 调用, ~1-2us CPU 开销
  return entry.output                 # 297 kernel 全部 GPU 端调度
```

首批 replay 仍需 ~760ms (GPU 侧 JIT warmup, Tensor Core 预热, L2 cache 未命中)。
4190ms 后收敛到稳态 ~9.8ms/step。

---

**Phase 6 (4190-6065ms): 稳态 Decode**

与 Phase 5 完全相同的代码路径, 但所有 warmup 已完成:

```
Engine core 主循环 (紧密循环)
─────────────────────────────
v1/engine/core.py:380-409

while scheduler.has_requests():
  sched_out = scheduler.schedule()                  # CPU: 调度 (~0.5ms)
  future = executor.execute_model(sched_out)        # CPU: enqueue graph.replay()
  model_output = future.result()                    # CPU: 等待 (async 模式下不等)
  if model_output is None:
    model_output = executor.sample_tokens(grammar)  # CPU: sampling + D2H copy
  scheduler.update_from_output(sched_out, output)   # CPU: 状态更新
```

GPU 侧: `graph.replay()` → 297 kernel (stream 19) + 420 kernel (stream 7109) → ~8.5ms GPU 时间。
CPU 侧: 调度 + sampling + 准备下一步 → ~1.3ms, 与 GPU 流水线化 (async scheduling)。

第二个请求 (5632ms) 到达时 nixl 传输仅 1.4ms (KV cache 已 warm), 且复用已有 graph,
`execute_context_1` 仅 41ms (无需 capture)。

### 1.3 稳态单步性能分解

单个 decode step, device 0, stream 19, graph 17417:

```
类别                 时间(us)    占比     次数    平均(us)
───────────────────────────────────────────────────────
AllReduce              1,674   19.8%     120       13.6
GEMM (全部)            3,863   45.7%     425        9.1
Attention (fmha)         667    7.9%      61       10.9
MoE routing+finalize   1,039   12.3%     240        4.3
其他 (norm/elem/kvc)   1,212   14.3%
───────────────────────────────────────────────────────
总 GPU 时间             8,456 us (8.46 ms)
Wall-clock             ~9.8 ms (含 CPU 调度间隙)

Shared experts (stream 7109, 并行隐藏): 1,656 us, 420 kernels
```

---

## 2. 单层 Forward Pass 的 Kernel 映射

### 2.1 代码结构

模型入口: `vllm/model_executor/models/kimi_linear.py`

```
KimiLinearForCausalLM.forward()
  └→ KimiLinearModel.forward()          ← 61 层循环
       └→ KimiDecoderLayer.forward()    ← 每层: MLA + MoE
            ├→ input_layernorm()         ← RMSNorm
            ├→ KimiMLAAttention()        ← Multi-head Latent Attention
            ├→ post_attention_layernorm() ← RMSNorm
            └→ KimiMoE()                ← Mixture of Experts (60/61 层)
                                            (第 1 层是 dense MLP)
```

### 2.2 完整单层 Kernel 序列 (device 0, stream 19)

以下是 trace 中一个 MoE 层的实测 kernel 序列 (23 kernels on main stream + 7 on aux stream):

```
┌────────────────────────────────────────────────────────────────────────────┐
│                                                                            │
│  ① input_layernorm (RMSNorm + fused residual add + 上层 allreduce)         │
│     代码: kimi_linear.py:369                                               │
│                                                                            │
│     triton_poi_fused_0           (3.0us)  ← 残差提取                       │
│     triton_poi_fused_1           (1.2us)  ← RMSNorm pointwise             │
│     allreduce_fusion_kernel_     (8.0us)  ← 上层output reduce + norm 融合  │
│       oneshot_lamport                        (fuse_allreduce_rms=true)     │
│                                                                            │
│  ② MLA Attention                                                           │
│     代码: model_executor/layers/mla.py MultiHeadLatentAttentionWrapper      │
│                                                                            │
│     fused_a_gemm_kernel          (7.5us)  ← QKV latent 压缩投影            │
│     triton_poi_fused_2           (2.6us)  ← kv_a_layernorm                │
│     triton_red_fused_3           (2.7us)  ← 归一化 reduce                  │
│     nvjet_sm100_tst_24x64_TNN   (6.1us)  ← kv_b_proj (KV latent→full KV) │
│     triton_poi_fused_add_clone_  (2.6us)  ← RoPE 位置编码                  │
│       copy_expand_index_mul_neg_                                           │
│       slice_split_stack                                                    │
│     concat_and_cache_mla_kernel  (2.7us)  ← KV Cache 写入 (paged)         │
│     nvjet_sm100_tst_64x8_NNT    (3.0us)  ← q_proj (Q latent→full Q)      │
│     triton_poi_fused__to_copy_   (1.7us)  ← Q/K scaling + concat          │
│       cat_clamp_mul_reciprocal_                                            │
│       view_0                                                               │
│     fmhaSm100fKernel_QkvE4m3O  (10.7us)  ← ★ Flash Attention SM100       │
│       Bfloat16HQk576HV512                    Paged-KV, FP8 QKV, BF16 out │
│     nvjet_sm100_tst_16x64_TNN   (3.8us)  ← o_proj (part 1)               │
│     nvjet_sm100_tst_64x8_TNT    (7.3us)  ← o_proj (part 2) + fused       │
│     allreduce_fusion_kernel      (8.0us)  ← o_proj TP AllReduce            │
│                                                                            │
│  ③ post_attention_layernorm                                                │
│     (fused into allreduce above by fuse_allreduce_rms)                     │
│                                                                            │
│  ④ MoE FFN ← 双流并行!                                                    │
│     代码: kimi_linear.py:162-177                                           │
│                                                                            │
│     ┌────────── Stream 19 (主流) ──────────┬─── Stream 7109 (辅助流) ───┐  │
│     │                                      │                           │  │
│     │ router_gemm_kernel_     (4.2us) Gate │ vectorized_elem  (2.1us)  │  │
│     │   float_output                       │ cvt_fp16_to_fp4  (2.5us)  │  │
│     │ vectorized_elementwise  (2.5us) Bias │ device_kernel    (9.0us)  │  │
│     │ cvt_fp16_to_fp4_sf_    (2.3us) Quant│   (shared gate_up GEMM)   │  │
│     │   major                              │ triton_poi_fused_(2.3us)  │  │
│     │ routingMainKernel      (4.2us) TopK  │   mul_silu_slice_0        │  │
│     │ routingIndicesCluster  (5.3us) 分发  │   (SiLU activation)       │  │
│     │                                      │ vectorized_elem  (1.2us)  │  │
│     │ bmm_E2m1_E2m1E2m1_   (22.0us) W1+W3│ cvt_fp16_to_fp4  (2.5us)  │  │
│     │   Fp32                   FP4 GEMM    │ device_kernel    (5.0us)  │  │
│     │ bmm_Bfloat16_E2m1E2m1 (13.0us) W2   │   (shared down GEMM)     │  │
│     │   _Fp32                  FP4 GEMM    │                           │  │
│     │ finalizeKernel         (3.4us) 合并  │ ← wait_stream 汇合       │  │
│     └──────────────────────────────────────┴───────────────────────────┘  │
│                                                                            │
│  ⑤ 残差相加 + TP AllReduce                                                 │
│     triton_poi_fused_add_mul_0   (1.5us)  ← shared + routed 加权求和      │
│     allreduce_fusion_kernel /    (12.0us) ← MoE output reduce             │
│       cross_device_reduce_1stage            (+ next layer RMSNorm fused)  │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

### 2.3 时序图: 主流 vs 辅助流并行

```
时间 ─────────────────────────────────────────────────────────────────────→

Stream 19:  ┃ar ┃fused_a┃norm┃kv_b┃rope┃kvc┃q_proj┃scale┃ fmha ┃o_p┃o_p┃ar ┃
(MLA部分)   ┃8us┃ 7.5us ┃5us ┃6us ┃3us ┃3us┃ 3us  ┃2us  ┃11us  ┃4us┃7us┃8us┃

Stream 19:  ┃gate┃bias┃fp4 ┃topK┃idx ┃  bmm_E2m1   ┃ bmm_Bf16  ┃fin┃add┃ allreduce ┃
(MoE部分)   ┃4us ┃3us ┃2us ┃4us ┃5us ┃    22us     ┃   13us    ┃3us┃2us┃   12us    ┃
                  ↓ wait_stream                                        ↑ wait_stream
Stream 7109:     ┃vec┃fp4┃ device_kernel ┃silu┃vec┃fp4┃devkernel┃────┘
(Shared exp)     ┃2us┃3us┃    9us        ┃2us ┃1us┃3us┃  5us    ┃
                 └── 总计 ~25us, 完全隐藏在主流 ~75us 后面 ──────┘
```

### 2.4 Kernel → 代码位置 映射表

| Kernel | 代码位置 | 模块 |
|--------|---------|------|
| `allreduce_fusion_kernel_oneshot_lamport` | FlashInfer `trtllm_allreduce_fusion.cuh` | TP 通信 |
| `fused_a_gemm_kernel` | `layers/mla.py` → fused QKV A-proj | MLA |
| `nvjet_sm100_tst_*_TNN` | CUTLASS SM100 GEMM (TNN layout) | 各种 Linear |
| `nvjet_sm100_tst_*_TNT` | CUTLASS SM100 GEMM (TNT layout) | o_proj / down_proj |
| `nvjet_sm100_tst_*_NNT` | CUTLASS SM100 GEMM (NNT layout) | q_proj |
| `fmhaSm100fKernel_*` | Flash Attention SM100 (Paged-KV) | MLA Attention |
| `concat_and_cache_mla_kernel` | `vllm/_custom_ops.py` | KV Cache |
| `device_kernel` | cuDNN fused GEMM (shared expert) | Shared MLP |
| `router_gemm_kernel_float_output` | FlashInfer MoE router | MoE Gate |
| `routingMainKernel` | FlashInfer top-K routing | MoE Dispatch |
| `routingIndicesClusterKernel` | FlashInfer token→expert 索引 | MoE Dispatch |
| `bmm_E2m1_E2m1E2m1_Fp32` | CUTLASS FP4 BMM (FP4×FP4→FP32) | MoE Expert W1+W3 |
| `bmm_Bfloat16_E2m1E2m1_Fp32` | CUTLASS FP4 BMM (BF16×FP4→FP32) | MoE Expert W2 |
| `finalizeKernel` | FlashInfer MoE combine | MoE Reduce |
| `cvt_fp16_to_fp4_sf_major` | `csrc/quantization/fp4/nvfp4_quant_kernels.cu` | 激活量化 |
| `triton_poi_fused_add_mul_0` | Triton JIT (残差相加) | 残差连接 |
| `triton_red_fused_2/3` | Triton JIT (RMSNorm reduce) | LayerNorm |
| `cross_device_reduce_1stage` | `csrc/custom_all_reduce.cuh` | vLLM 自定义 AR |

---

## 3. 机制一：Lamport 负零哨兵协议

### 3.1 问题：为什么不用 NCCL？

Decode 时每次 allreduce 的数据量极小：

```
payload = hidden_size × sizeof(bf16) = 7168 × 2 = 14 KB
NVSwitch 理论传输时间: 14KB / 900GB/s = 0.016 us
NCCL 实测延迟: ~15 us (调度 + 协议 + barrier)
实际数据传输只占延迟的 0.1%
```

每 step 120 次 allreduce × 15us = **1.8ms 纯通信开销**。问题不在带宽，在延迟。

### 3.2 解法：绕过 NCCL，用 P2P 内存 + 无 barrier 同步

**代码**: `csrc/custom_all_reduce.cuh`, FlashInfer `trtllm_allreduce_fusion.cuh`

#### 内存布局

每个 GPU 通过 `cudaIpcOpenMemHandle` 获得所有其他 GPU buffer 的虚拟地址映射：

```
              NVSwitch 统一地址空间 (MNNVL on GB200)
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│   GPU 0 的 IPC buffer         GPU 1 的 IPC buffer           │
│   ┌─────────────────┐        ┌─────────────────┐            │
│   │ data_buf[0] 8MB │←───────│ ptrs[0] 指向此  │            │
│   │ data_buf[1] 8MB │        │ data_buf[0] 8MB │←── GPU 0   │
│   │ data_buf[2] 8MB │        │ data_buf[1] 8MB │   直接读!  │
│   ├─────────────────┤        │ data_buf[2] 8MB │            │
│   │ Signal:         │        ├─────────────────┤            │
│   │  start[blk][rk] │        │ Signal:         │            │
│   │  end[blk][rk]   │        │  ...            │            │
│   │  _flag[blk]     │        └─────────────────┘            │
│   └─────────────────┘                                        │
│                                                              │
│   GPU 2 IPC buffer            GPU 3 IPC buffer              │
│   └─────────────────┘        └─────────────────┘            │
└──────────────────────────────────────────────────────────────┘
```

#### 负零哨兵协议

IEEE 754 浮点数有两个零：`+0.0 (0x0000)` 和 `-0.0 (0x8000)`。它们数值相等但二进制不同。利用这一点消除显式 barrier：

```
三缓冲轮换:
  Step 1 (flag=0): 写 buf[0], 清 buf[2] 为负零
  Step 2 (flag=1): 写 buf[1], 清 buf[0] 为负零
  Step 3 (flag=2): 写 buf[2], 清 buf[1] 为负零
  Step 4 (flag=3): 写 buf[0], 清 buf[2] 为负零 (循环)
```

每次 allreduce 的执行流程：

```cuda
// === Phase 1: PUSH ===
// 每个 GPU 写自己的数据，清除两轮前的旧 buffer
data_buf[flag % 3][my_offset] = my_data;         // 写真实数据
data_buf[(flag+2) % 3][my_offset] = -0.0f;       // 旧 buf 标记为未就绪

// === Phase 2: POLL + REDUCE ===
// 直接读其他 GPU 的 buffer，用负零判断是否就绪
float sum = 0;
for (int rank = 0; rank < 4; rank++) {
    float val;
    do {
        val = load_global_volatile(             // volatile 绕过 L2 cache
            peer_data_buf[rank][flag % 3][my_offset]
        );
    } while (has_negative_zero(val));           // 负零 = 未就绪, 继续自旋
    sum += val;                                 // 非负零 = 真实数据, 累加
}
output[my_offset] = sum;
```

**为什么用三个 buffer 而不是两个？** 防止 ABA 问题——当 GPU 0 开始第 N+2 轮写 buf[0] 时，如果只有两个 buffer，GPU 3 可能还在读第 N 轮的 buf[0]。三缓冲保证写入的 buffer 和任何 GPU 可能正在读的 buffer 永远不同。

#### 时序对比

```
                     NCCL allreduce              Lamport oneshot
──────────────────────────────────────────────────────────────────
kernel launch        需要调度新 kernel            已在运行的 kernel 内
拓扑协商             ring/tree 选择               直接 P2P
内存拷贝             数据拷到 NCCL buffer          原地读对方 buffer
同步机制             collective barrier           spin on 负零哨兵
最小延迟             ~10-15us                     ~5-8us
适用范围             任意大小                     < 512KB (4GPU)
```

#### 融合优化: allreduce + RMSNorm

编译 pass `allreduce_rms_fusion.py` 将两个操作融合到一个 kernel：

```
融合前 (两次 global memory 往返):
  kernel 1: allreduce(x)           → 写 global memory → barrier
  kernel 2: rmsnorm(allreduced_x)  → 写 global memory

融合后 (一次 global memory 往返):
  kernel: allreduce_fusion(x) {
    sum = poll_and_reduce(all_peers);     // allreduce
    norm = rsqrt(mean(sum * sum));        // RMSNorm
    output = sum * norm * weight;         // scale
  }
```

trace 中实测: 每层开头的 `allreduce_fusion_kernel_oneshot_lamport` 就是这个融合版本。

---

## 4. 机制二：CUDA Graph 捕获跨 GPU 通信

### 4.1 核心矛盾

CUDA Graph 是"录制一次，重放多次"——所有 kernel 参数在 capture 时固定。但 allreduce 需要：
1. 每次读到**新鲜数据**（不是 capture 时的）
2. 自旋等待其他 GPU **本次执行** 的 flag（不是 capture 时的）

### 4.2 解法：指针间接寻址 + IPC 事后注册

这里要先抓住一个容易绕晕的点：

- 普通 CUDA Graph 固定的是 **数据地址**，不是数据内容；replay 前只要把新数据写回同一地址即可。
- 但 custom allreduce 不只要读 `input` 自己的地址，还要读 **所有 rank 的 peer buffer 地址**。
- 因此 graph 里不能只固定 `input_ptr`，而是要固定 **“地址表的地址”**：kernel 先读 `RankData*`，再从 `RankData.ptrs[]` 里取出每个 rank 的 live buffer 地址。

也就是说，这里固定的是两层地址：

1. graph 参数里的 `&d_rank_data_base_[N]`
2. `d_rank_data_base_[N]` 里存的 `GPU0/GPU1/...` peer buffer 指针

第一层在 capture 时必须固定；第二层可以在 capture 后通过 IPC 打开远端显存再回填。

**关键数据结构**:

```cpp
// csrc/custom_all_reduce.cuh
struct RankData {
    const void* ptrs[8];  // 指向 8 个 rank 的数据缓冲区的指针
};

// d_rank_data_base_ 是 device memory 上的 RankData 数组
// Kernel 参数不是数据本身，而是 "指向数据指针的指针"
```

#### Capture 阶段

```cpp
// csrc/custom_all_reduce.cuh — allreduce() 被调用时
cudaStreamCaptureStatus status;
cudaStreamIsCapturing(stream, &status);

if (status == cudaStreamCaptureStatusActive) {
    // CAPTURE 模式: 记录 buffer 地址，稍后注册
    ptrs = &d_rank_data_base_[graph_unreg_buffers_.size()];
    graph_unreg_buffers_.push_back(input);  // 保存, 事后注册 IPC

    // Graph 录制的是: "launch kernel, arg = &d_rank_data_base_[N]"
    // d_rank_data_base_[N] 的内容此时未初始化!
}

// Kernel 内部通过 ptrs->ptrs[rank] 间接寻址
```

#### 事后注册阶段

```python
# custom_all_reduce.py:213-230 — graph capture 完成后
ipc_meta = custom_ar.get_graph_buffer_ipc_meta()   # 提取 IPC handles
all_ipc = dist.all_gather_object(ipc_meta)          # 跨 rank 交换
custom_ar.register_graph_buffers(all_handles, ...)  # 打开远端 handle
```

C++ 侧执行:

```cpp
// csrc/custom_all_reduce.cu — register_graph_buffers()
for (int buf = 0; buf < num_buffers; buf++) {
    RankData rd;
    for (int rank = 0; rank < world_size_; rank++) {
        if (rank == self_rank)
            rd.ptrs[rank] = self_buffer;          // 本地
        else {
            void* peer = cudaIpcOpenMemHandle(...); // 远端 IPC
            rd.ptrs[rank] = peer + offsets[rank];
        }
    }
    // 关键: 写入 device memory, 这是 graph replay 时 kernel 会读取的位置!
    cudaMemcpy(&d_rank_data_base_[buf], &rd, sizeof(RankData), H2D);
}
```

#### Replay 时的数据流

```
┌─────────────────────────────────────────────────────────────────┐
│  Graph 录制了:                                                   │
│    "launch kernel, arg = &d_rank_data_base_[0]"                 │
│                                                                 │
│  Capture 时  d_rank_data_base_[0] = {未初始化}                  │
│  注册后      d_rank_data_base_[0] = {GPU0: 0x1000,             │
│                                      GPU1: 0x2000,             │
│                                      GPU2: 0x3000,             │
│                                      GPU3: 0x4000}             │
│                                                                 │
│  Replay N:   kernel 从 0x1000 读 GPU0 的最新数据 ← live buffer │
│  Replay N+1: kernel 从 0x1000 读 GPU0 的最新数据 ← 每次都新鲜  │
│                                                                 │
│  指针永远指向 live buffer, 不是快照!                              │
└─────────────────────────────────────────────────────────────────┘
```

#### Lamport flag 跨 replay 的正确性

flag 计数器存在 `Signal` 结构中 (IPC 共享内存), 不是 graph 参数:

```cuda
// 每次 allreduce 执行:
uint32_t flag = self_sg->_flag[blockIdx.x] + 1;  // 读 device memory, 递增
// ... 执行 allreduce ...
self_sg->_flag[blockIdx.x] = flag;                 // 写回 device memory

// Replay 1: flag 0 → 1, spin-wait 直到所有 GPU flag=1
// Replay 2: flag 1 → 2, spin-wait 直到所有 GPU flag=2
// 永不混淆, 因为 flag 在 device memory 中递增
```

### 4.3 多流 CUDA Graph

CUDA Graph 捕获的不是 "kernel 列表"，而是一个 **DAG**:

```
Graph 17417 内部 (简化到一层):

Stream 19:  ─[norm]─[qkv]─[attn]─[o_proj]─[allreduce]─┬─[gate]─[route]─[bmm]─[final]─[ar]─→
                                                         │                                ↑
Stream 7109: ───────────────────────────────────────────┘─[cvt]─[gemm]─[silu]─[gemm]────┘
                                                        wait                           wait
```

`torch.cuda.stream()` 内的操作被捕获到对应 stream 上, `wait_stream` 被录制为 DAG 中的**依赖边**。Replay 时 GPU 硬件自动在对应流上并行执行。

### 4.4 性能收益

```
没有 CUDA Graph:
  297 kernel × ~4us launch overhead = 1.19ms 纯 CPU 开销
  + 120 次 allreduce 需要单独 NCCL kernel → +120 次额外 launch

有 CUDA Graph:
  1 次 cudaGraphLaunch (~1-2us)
  297 kernel + 跨流依赖 + 跨 GPU 同步 = 全部硬件级调度, CPU 零开销
```

---

## 5. 机制三：FP4 E2M1 两阶段 GEMM 设计

### 5.1 E2M1 格式

4-bit 浮点: 1 位符号 + 2 位指数 + 1 位尾数, 无 inf/NaN:

```
bits  sign exp  man    值     计算
0000   +   00   0    +0.0    subnormal
0001   +   00   1    +0.5    subnormal: 0.5 × 2^0
0010   +   01   0    +1.0    normal: 1.0 × 2^0
0011   +   01   1    +1.5    normal: 1.5 × 2^0
0100   +   10   0    +2.0    normal: 1.0 × 2^1
0101   +   10   1    +3.0    normal: 1.5 × 2^1
0110   +   11   0    +4.0    normal: 1.0 × 2^2
0111   +   11   1    +6.0    normal: 1.5 × 2^2

可表示值 (含负数): ±{0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}
存储: 2 个 FP4 值打包到 1 个 uint8 (nibble pair)
```

代码定义: `vllm/scalar_type.py:345`:
```python
float4_e2m1f = ScalarType.float_(2, 1, True, NanRepr.NONE)
```

### 5.2 Block Scale Factor 机制

8 个离散值的动态范围只有 [0, 6.0]，直接量化会崩溃。解法是 **block-wise scaling**:

```
原始数据 (BF16): [0.12, -3.7, 8.2, 0.001, ..., -15.3]  (每 16 个值一组)

Step 1: max_abs = max(|values|) = 15.3

Step 2: scale = global_scale × (max_abs / 6.0) = global_scale × 2.55

Step 3: scale → FP8 E4M3 格式 (8-bit, 省存储)

Step 4: 每个值量化:
         0.12 / scale → 0.047 → round → 0    (0000)
        -3.7 / scale → -1.45  → round → -1.5 (1011)
         8.2 / scale →  3.22  → round →  3.0 (0101)

存储:  ┌────────────────────┬──────────────┐
       │ 16 × FP4 = 8 bytes │ 1 × FP8 SF  │  ← "sf_major" 布局:
       │ (nibble-packed)     │ (E4M3 格式)  │    SF 连续存储, 数据连续存储
       └────────────────────┴──────────────┘    (对 Tensor Core MMA 更友好)
```

**为什么必须是 block-wise，而不是一个 tensor 只用一个 scale？**

因为激活张量里通常同时存在：

- 少量大值 / outlier
- 大量中小值

如果整块 tensor 只共用一个 scale，那么这个 scale 会被少量最大值主导，结果是：

- 大值不溢出
- 但大部分中小值会被压得非常接近 0
- 量化后大量元素直接掉桶，信息损失太大

反过来，如果每个元素单独配一个 scale，精度当然最好，但会有两个问题：

- 元数据开销过大，几乎失去 FP4 压缩意义
- 后续 Tensor Core 的 block-scaled GEMM 也不是按“每元素一个 scale”来吃数据

所以 block-wise 是折中：

- 比 per-tensor scale 细，能避免 outlier 把整块都拖坏
- 比 per-element scale 粗，元数据仍然很小
- 还能直接匹配后续 block-scaled Tensor Core GEMM 的输入格式

在这套实现里，粒度不是随便挑的，而是 **每 16 个值共享 1 个 SF**：

- `CVT_FP4_SF_VEC_SIZE = 16`
- `cvt_warp_fp16_to_fp4()` 先对这 16 个值取 `max_abs`
- 再写出 1 个 FP8 的 `SFout`

代码: `csrc/libtorch_stable/quantization/fp4/nvfp4_quant_kernels.cu`

### 5.3 两阶段 GEMM: 为什么 W1+W3 用 FP4 input 而 W2 用 BF16 input

```
                MoE Expert FFN 数据流

Hidden states (BF16)
        │
        ├──→ cvt_fp16_to_fp4_sf_major ──→ FP4 (给 routed experts)
        │                                    │
        │         ┌─────────────────────────┘
        │         ↓
        │    bmm_E2m1_E2m1E2m1_Fp32           ★ W1+W3: FP4 × FP4 → FP32
        │    Kernel含义: input=E2M1, weight=E2M1, output=FP32
        │         │
        │         ↓
        │    SiLU(gate) × up = activated       此时数据回到 BF16
        │         │
        │         ↓
        │    bmm_Bfloat16_E2m1E2m1_Fp32       ★ W2: BF16 × FP4 → FP32
        │    Kernel含义: input=BF16, weight=E2M1, output=FP32
        │         │
        │         ↓
        │    finalizeKernel → BF16 output
```

**为什么不对称量化？**

这是基于误差传播路径的精度权衡:

| 阶段 | Input 格式 | Weight 格式 | 理由 |
|------|-----------|-----------|------|
| W1+W3 (gate+up) | **FP4** | FP4 | 进入 SiLU 非线性前，FP4 精度损失对 gate 值影响有限 |
| W2 (down) | **BF16** | FP4 | SiLU 输出即将回到主残差路径，精度损失会被后续所有层放大 |

关键点: W2 的 output 直接通过 `finalizeKernel` 加权求和回 `hidden_states` 主路径。如果 W2 input 也 FP4：SiLU 压缩了值域 + FP4 只有 8 个离散值 → 两轮量化误差叠加 → 后续 60 层会放大这个误差。

**为什么不是“input 继续 FP4，但 output / accumulator 用高精度”就够了？**

因为这两件事解决的是两个完全不同的问题：

- **高精度 output / accumulator** 解决的是“求和时别再额外丢精度”
- **高精度 input** 解决的是“乘法用的操作数本身别太粗糙”

**再直白一点：矩阵乘法不是像 `a + b` 那样的一个原子操作。**

对 W2 来说，每个输出元素其实都是一整个点积：

`y[i,j] = Σ_k W2[i,k] * x[k]`

也就是说，一个输出值不是“乘一次再写回”，而是：

1. 先生成很多个中间乘积 `p_k = W2[i,k] * x[k]`
2. 再把这些 `p_k` 累加起来，得到最终的 `y[i,j]`

所以这里有两层精度问题：

- `p_k` 本身算得准不准，取决于 **输入操作数精度**
- `p_1 + p_2 + ... + p_k` 加得准不准，取决于 **accumulator 精度**

很多人会直觉地把 GEMM 想成“先算出一个结果，再用高精度 buffer 存住它”，但这只覆盖了第二层；它保护的是 running sum，不是每个乘积项本身。

如果 `x` 已经先被量化成 FP4，那么每个中间项实际上是：

`p_k = W2[i,k] * Q(x[k])`

而不是：

`p_k = W2[i,k] * x[k]`

所以误差是在 **生成每个 partial product** 的那一刻就进来了；后面即使把所有 `p_k` 都用 FP32 accumulator 累加，也只是“高精度地加一堆已经带误差的中间项”，并不能把它们恢复成原始的正确乘积。

如果 W2 输入已经先被量化成 FP4，那么 GEMM 真正算的是：

`W2 * Q(x)`，而不是 `W2 * x`

其中 `Q(x) = x + e` 是量化后的近似值，于是结果变成：

`W2 * Q(x) = W2 * x + W2 * e`

即使后面的累加全是 FP32，这个 `W2 * e` 也已经注入进结果里了；高精度 accumulator 只能减少“加法误差”，不能把前面乘法输入里已经丢掉的信息找回来。

而 W2 正是把专家输出投回主 hidden state 的最后一层线性层，所以这里更怕的是 **输入操作数误差**，不是只怕 accumulation 误差。

### 5.4 Tensor Core FP4 指令

GB200 (SM 10.0, Blackwell) Tensor Core 原生支持 FP4:

```
MMA 指令: mma.sync.aligned.m16n8k64.f32.e2m1.e2m1.f32

  - 输出 tile: 16×8
  - K 维度: 64 (64 个 FP4 = 32 bytes)
  - 每次: 16 × 8 × 64 = 8192 FP4 FLOPs / cycle / Tensor Core

对比 BF16: mma.sync.m16n8k16 → 2048 FLOPs / cycle
→ FP4 算力是 BF16 的 4 倍 (K=64 vs K=16)
```

这就是 NVFP4 的核心价值: 在相同芯片功耗下获得 **4× MoE expert 吞吐**。

### 5.5 端到端数据格式转换时序

从 trace 中 MoE 层的 kernel 可精确追溯每次格式转换:

```
时间(us)  Kernel                      Input → Output         说明
──────────────────────────────────────────────────────────────────────
 0.0     router_gemm_kernel           BF16 → FP32           门控 logit
 4.2     vectorized_elementwise       FP32 → FP32           score bias
 6.7     cvt_fp16_to_fp4_sf_major     BF16 → FP4+SF(FP8)   ★ 激活量化
 9.0     routingMainKernel            FP32 → INT32          Top-K 索引
13.2     routingIndicesCluster        INT32 → INT32         分发表
18.5     bmm_E2m1_E2m1E2m1_Fp32      FP4 × FP4 → FP32     ★ W1+W3 GEMM
40.5     bmm_Bfloat16_E2m1E2m1_Fp32  BF16 × FP4 → FP32    ★ W2 GEMM
53.5     finalizeKernel               FP32 → BF16           加权合并
56.9     triton_poi_fused_add_mul_0   BF16 + BF16 → BF16   残差相加
58.4     allreduce_fusion_kernel      BF16 → BF16           TP reduce
```

注意: **量化发生在 routing 之前** (`cvt_fp16_to_fp4` 在 `routingMainKernel` 之前)。FP4 数据直接被 dispatch 到 expert，expert GEMM 直接消费 FP4 input，省掉 dispatch 后再量化的开销。

---

## 6. 关键配置参数

### 6.1 Decode 端 vLLM 配置

来自 `pd_kimi_70k_nsys_decode.yaml`:

```yaml
# CUDA Graph
--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
--compilation_config.pass_config.fuse_allreduce_rms true  # allreduce+RMSNorm 融合
-O3                                                        # 最高编译优化等级

# MoE
VLLM_USE_FLASHINFER_MOE_FP4: "1"                 # 启用 FlashInfer FP4 MoE kernel
VLLM_FLASHINFER_MOE_BACKEND: latency             # 延迟优化 backend
VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE: "8192"       # 每 expert 最大 token 数
VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD: "8192" # 共享专家并行流阈值

# 投机解码
--speculative-config '{"model": "lightseekorg/kimi-k2.5-eagle3",
                       "method": "eagle3",
                       "num_speculative_tokens": 3}'

# Attention
--attention_config.flash_attn_version 4           # Flash Attention v4 (SM100)
--kv-cache-dtype fp8                              # KV Cache FP8 量化

# nsys 采集
--profiler-config '{"profiler":"cuda","delay_iterations":30,"max_iterations":200}'
```

### 6.2 Graph ID 与身份

| Graph ID | Kernel 数 | 内容 | 身份 |
|----------|----------|------|------|
| 17417 | 1,020,880 | 61层完整 forward (2 streams) | 主模型 decode |
| 17390 | 11,928 | allreduce + TNT GEMM + splitK | EAGLE3 验证头 |
| 17387 | 13,632 | allreduce + splitK GEMM | EAGLE3 draft model |

### 6.3 活跃 Stream

| Stream ID | 角色 | 每 step kernel 数 (dev0) |
|-----------|------|------------------------|
| 19 | 主计算流 (所有 CUDA Graph) | ~1,403 |
| 7109 | 共享专家辅助流 (aux_stream) | ~420 |
| 7119 | 共享专家辅助流 (dev 1/2/3 上的 aux_stream, 见 §7.0) | ~420 |
| async_output_copy | D2H 结果拷贝 | 少量 memcpy |

---

## 7. Trace-Derived Insights

以下发现**只能从 nsys trace 数据中观察到**，无法通过阅读代码得出。

### 7.0 为什么 4 个进程都看到 stream 19，以及为什么 aux 是 7109/7119

这里最容易误解的是：**nsys 里的 stream ID 不是跨进程全局唯一名字**。从 sqlite 可以直接看到：

```
globalPid            pid     device  contextId  streamId
295235766714368     820207    0         1         19
295235783491584     820208    1         1         19
295235800268800     820209    2         1         19
295235867377664     820213    3         1         19
```

也就是说，`streamId=19` 在 4 个 `VLLM::Worker_TP` 进程里都出现了，所以它绝不是
"4 个进程共享同一个 stream 对象"，而只是 **4 个不同进程里的本地 stream 编号刚好都叫 19**。

源码也解释了为什么它们这么一致：

- `current_stream()` (`vllm/utils/torch_utils.py:528`) 在第一次调用时不会返回 CUDA 默认流
- 它会显式执行 `torch.cuda.set_stream(torch.cuda.Stream())`
- 注释里写得很明确：CUDA Graph capture 不能用默认流，因此每个进程都创建一个 **dedicated stream per process**
- CUDA Graph capture 也确实绑定到了这个 stream 上：`torch.cuda.graph(..., stream=current_stream())`

所以：

- trace 里的 **main stream 19** 不是默认流
- 默认/null stream 在 sqlite 里其实是 `7`
- `19` 是 **每个 worker 进程自己创建的专用 compute stream**
- 因为 4 个 worker 的初始化顺序基本一致，所以这个“第一条关键用户流”在每个进程里都落成了本地编号 `19`

再看 aux stream：

- `aux_stream()` (`vllm/utils/torch_utils.py:577`) 也是 `torch.cuda.Stream()` 创建的
- `default_moe_runner.py` 只拿它来做 shared experts 并行流
- 它是 **每进程一个** 的全局 singleton，不是每层一个，也不是全局跨进程共享

那为什么会出现两个数字 `7109/7119`？

- 因为 aux stream 创建得比 main stream 晚得多
- 在 aux stream 创建之前，各 worker 进程里已经有很多别的 CUDA stream 被框架/库提前创建
- 这些“前置 stream”的创建顺序在不同 worker 上略有差异
- 所以同样是逻辑上的 aux stream，在 device 0 进程里拿到的是本地编号 `7109`，在 device 1/2/3 进程里拿到的是 `7119`

因此更准确的结论是：

```
Device 0: null stream 7, main stream 19, aux/shared-expert stream 7109
Device 1: null stream 7, main stream 19, aux/shared-expert stream 7119
Device 2: null stream 7, main stream 19, aux/shared-expert stream 7119
Device 3: null stream 7, main stream 19, aux/shared-expert stream 7119
```

所以：

- `19` 一样，不代表跨进程共享了同一条流
- `7109/7119` 不一样，也不代表存在两条不同逻辑的 aux 流
- 真正唯一能标识一个 trace 里的 CUDA stream 的，是至少要连同 **进程 / context / device** 一起看，而不能只看裸 `streamId`

---

### 7.1 AllReduce 跨 GPU 不对称: dev 2 等待时间是 dev 3 的 2 倍

**数据** (单步 4451-4462ms):
```
                allreduce 总时间   compute 总时间    allreduce 占比
  dev 0:          1,674 us          6,781 us          20%
  dev 1:          1,219 us          6,400 us          16%
  dev 2:          2,193 us          6,675 us          25%  ← 最慢
  dev 3:          1,083 us          6,661 us          14%  ← 最快
```

compute 部分几乎相同 (6400-6780us, 差异 < 6%)，但 allreduce 时间差 **2 倍**。

**首个 allreduce 的到达时序揭示了根本原因**:
```
                start time     end time      duration    等了谁?
  dev 0:        4451.815ms     4452.419ms     604us      等 dev 1,3
  dev 2:        4451.999ms     4452.420ms     420us      等 dev 1,3
  dev 1:        4452.361ms     4452.419ms      58us      等 dev 3
  dev 3:        4452.412ms     4452.419ms       7us      最后到, 不等
```

所有 GPU 在 ~4452.419ms 同时完成（end time 相差 < 1us），证明 Lamport 协议正确同步了。
但 **dev 0 比 dev 3 早到 597us**，这 597us 全部浪费在 spin-wait 上。

**原因分析**:

这不是 NVSwitch 拓扑问题（每层后续 allreduce 仅 6-10us）。根本原因是
**graph replay 之间的 CPU 调度间隙在各 GPU 上长度不同**:
- 每次 `graph.replay()` 由 CPU 发出，但 4 个 GPU 各有独立的 CUDA stream queue
- CPU 对 4 个 GPU 的 `replay()` 调用有微秒级时差
- 第一个 allreduce 吸收了这个初始偏差
- 一旦同步完成，后续 allreduce 几乎免费 (6-10us)，因为 GPU 已 lockstep

**影响**: 首个 allreduce "吸收偏差" 的代价约 200-600us/step，占 allreduce 总时间的 **30-50%**。
如果能让 4 个 GPU 的 graph replay 更同步地启动（例如用 CUDA event 同步 replay 起点），
可以将这部分开销压缩到接近零。

---

### 7.2 EAGLE3 Draft Model: 87% 时间在等 AllReduce (通信瓶颈)

**数据**:
```
                         总 GPU 时间   allreduce    allreduce 占比   非 AR kernel
主模型 (17417, 61层):     8,456 us      1,674 us       20%           6,782 us
EAGLE3 graph A (17387):    每次 ~370us   ~324 us       87%             ~46 us
EAGLE3 graph B (17390):    每次 ~189us   ~151 us       80%             ~38 us
```

EAGLE3 graph 17387 的完整 kernel profile (dev 0):
```
allreduce_fusion (324.8us avg)        ← 87% 的时间!
nvjet_sm100_tst splitK_TNT (31.2us)   ← lm_head GEMM
triton_red_fused_2 (3.7us)            ← RMSNorm reduce
triton_poi_fused_*                     ← 各种 pointwise
splitKreduce_kernel (3.1us)           ← splitK 归约
```

**这意味着什么**:

EAGLE3 draft model 只有 ~2 层 (lm_head + 投影层)，每层计算仅 ~20us。但每层仍需
TP allreduce，且 allreduce 的 latency floor (Lamport spin-wait ~8us + 首个 barrier ~300us)
不随计算量缩减。

```
Draft model 有效计算率:
  graph A: 46us / 370us = 12%   ← 88% 时间在通信!
  graph B: 38us / 189us = 20%

对比主模型:
  main:   6782us / 8456us = 80%  ← 20% 在通信
```

**结论**: EAGLE3 在 TP=4 下的投机解码是**极度通信瓶颈**的。每生成 3 个投机 token 的
开销 ~560us (graph A + B)，其中 475us 是纯 allreduce 等待。

**优化方向**:
- Draft model 用 TP=1 (不做 allreduce): 560us → ~84us, 节省 85%
- 或用更小的 draft model 跑在单 GPU 上，只在 verify 时做 TP

---

### 7.3 D2D Memcpy 揭示 EAGLE3 的 Token 接受/拒绝机制

**数据** (每步 3 组 D2D memcpy，发生在 graph replay 之间):
```
memcpy 1: 1920 KB × 4 GPU  (concurrent_kernels=0, 即 graph 之间)
memcpy 2:   56 KB × 4 GPU  = 4 tokens × 7168 × 2 bytes
memcpy 3:   14 KB × 4 GPU  = 1 token  × 7168 × 2 bytes
```

**解读**:

```
主模型 graph 17417 (verify + generate)
  │
  ├→ D2D 1920KB = 240 × 8KB: KV cache block 整理
  │   (EAGLE3 的 3 个投机 token 被 verify 后，接受的 token 的
  │    KV cache 需要从临时 slot 搬到正式 slot)
  │
  ├→ D2D 56KB = 4 tokens hidden states:
  │   (1 verified + 3 speculative 的 hidden_states 传给 EAGLE3 draft model)
  │
  ├→ D2D 14KB = 1 token hidden states:
  │   (最终确定的 1 个新 token 的 hidden state)
  │
  ↓
EAGLE3 graph 17390 → 17387 (draft 3 new speculative tokens)
```

**关键发现**: 这些 memcpy 发生在 graph replay 之间 (concurrent_kernels=0)，
是**串行的 GPU idle 时间**。虽然每次只有 ~2us，但 3 组 × 4 GPU = 24 次，加上
CPU 调度开销，这构成了步间 bubble 的一部分。

---

### 7.4 Shared Expert 的 SM 竞争: 与 bmm_E2m1 并行时反而更快

**数据** (stream 7109 device_kernel 时延 vs 主流并发 kernel):

```
主流并发 kernel                  aux device_kernel 耗时
─────────────────────────────────────────────────────
vectorized_elementwise (2us)         8-9 us  ← 慢
bmm_E2m1 (22us)                     4-5 us  ← 快!
bmm_Bfloat16 (13us)                11-12 us ← 慢
无 (gap)                             9 us   ← 基准
```

时延分布直方图:
```
 4us: ████████████ (12)         ← 与 bmm_E2m1 并行
 5us: █████████████████████ (21) ← 与 bmm_E2m1 并行
 8us: ███████████████████████████████████████████ (43) ← 与 small kernel 并行
 9us: █████████████████ (17)
11us: █████████ (9)              ← 与 bmm_Bfloat16 并行
```

**违反直觉**: 与大 kernel (bmm_E2m1, 22us) 并行时 shared expert 反而**更快** (4-5us vs 8-9us)。

**原因**: GB200 有 152 个 SM。`bmm_E2m1` 的 gridDim 可能只用了部分 SM（decode batch=1
时 expert GEMM 的矩阵很小），剩余 SM 可以充分服务 aux stream 的 `device_kernel`。
而 `vectorized_elementwise` 虽然执行快 (2us)，但它可能用了很多 SM blocks 来做
elementwise 操作（因为 elementwise kernel 通常 launch 很多 blocks 来保证延迟），
反而和 aux stream 抢 SM。

**推论**: 多流并行的效率取决于**并发 kernel 的 SM 占用率**，而不是单个 kernel 的执行时间。
小而宽的 kernel (多 block, 少计算) 反而比大而窄的 kernel (少 block, 多计算) 更影响并行效率。

---

### 7.5 Graph 内部的"两段式"结构: 1 层 Dense + 60 层 MoE

**数据**: 每个 graph replay 中 `triton_poi_fused_0` 出现 **2 次**，间距 ~0.5ms:

```
第 1 次 fused_0 @ 4451.809ms  ─→ 开始 sub-phase 1 (1 层 dense attention)
第 2 次 fused_0 @ 4452.474ms  ─→ 开始 sub-phase 2 (60 层 MoE)
下一步 fused_0 @ 4461.612ms  ─→ 间距 9.14ms = 一次完整 graph replay
```

Sub-phase 1 (layer 0, dense 层) kernel 序列:
```
triton_poi_fused_0           ← input RMSNorm (无上层 allreduce，因为是第一层)
triton_poi_fused_1           ← RMSNorm pointwise
allreduce_fusion             ← 上一 step 最后层 output 的 TP reduce + RMSNorm 融合
fused_a_gemm                 ← QKV latent projection
... MLA attention (7 kernels) ...
nvjet_TNT                    ← o_proj (attention output)
-- 无 MoE 层, 无 routing/bmm/finalize kernels --
```

Sub-phase 2 (layer 1-60, MoE 层) 开头:
```
triton_poi_fused_0           ← layer 0 dense MLP 后的 RMSNorm
allreduce_fusion             ← layer 0 output reduce (仅 6us, GPU 已同步)
device_kernel (20us)         ← layer 0 Dense MLP gate_up_proj (cuDNN)
silu_mul_cvt_fp16_to_fp4     ← SiLU + FP4 量化 (dense MLP, 非 MoE!)
device_kernel (10us)         ← layer 0 Dense MLP down_proj
allreduce_fusion             ← layer 0 MLP output reduce
fused_a_gemm                 ← layer 1 QKV projection (开始 MoE 层循环)
... 重复 60 次 MoE 层 ...
```

**关键发现**: Kimi K2.5 的第 0 层是 **dense MLP** (不是 MoE)。代码中
`kimi_linear.py:332-350` 的条件 `layer_idx >= config.first_k_dense_replace` 控制了
哪些层用 MoE，第 0 层不满足条件所以用 dense `KimiMLP`。

Dense MLP 用 cuDNN `device_kernel` (gate_up 20us + down 10us = 30us)，
不需要 routing/dispatch/finalize，也不用 aux_stream 并行——
因为 dense MLP 没有 shared expert 概念。

这解释了 trace 中的 "两段式" 结构: graph 先跑 1 层 dense，然后跑 60 层 MoE (后者包含
aux_stream 并行的 shared experts)。
