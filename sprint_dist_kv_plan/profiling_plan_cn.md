# Deep Profiling Plan

## 目标

基于 70K benchmark 三组对比结果，深入挖掘每个环节的延迟分布，建立完整的性能 mental model。

---

## 需要明确的指标

### A. 各 offload 路径的 CPU-GPU 带宽（分 p50/p95/p99）

| 指标 | 当前状态 | 需要做什么 |
|------|---------|-----------|
| MooncakeStore put (GPU→CPU) throughput 分布 | ✅ **已实现 p50/p95/p99** | `reduce()` 输出 `tp_p50/p95/p99_mib_s` 和 `lat_p50/p95/p99_ms` |
| MooncakeStore get (CPU→GPU) throughput 分布 | ✅ **已实现 p50/p95/p99** | 同上 |
| SimpleCPUOffload put/get throughput | ❌ 没有 | 需要在 SimpleCPUOffload 代码中加计时（非本项目范围） |
| Nixl P→D throughput 分布 | ✅ 有（avg/P90） | 已有（NixlConnector 内置） |
| C++ RDMA batch_put/get latency 分布 | ✅ 有 p95（C++ stderr） | 已有（`MC_STORE_CLIENT_METRIC_INTERVAL`） |

**已完成**：在 `mooncake_store_metrics.py` 的 `reduce()` 中增加了 per-transfer throughput 和 latency 的 p50/p95/p99。
每个 stats interval 输出 12 个新指标（put/get × throughput/latency × p50/p95/p99）。

### B. CPU memory 驱逐情况

**注意**：这些指标仅对 **MooncakeStoreConnector** 配置有意义。SimpleCPUOffloadConnector 的驱逐在 vLLM 内部的 block_pool LRU 中完成，没有暴露 counter。

| 指标 | 来源 | 当前状态 | 需要做什么 |
|------|------|---------|-----------|
| `master_successful_evictions_total` | Mooncake Master :9003 | ✅ Grafana 可看（167 次） | 已有 |
| `master_evicted_size_bytes` | Mooncake Master :9003 | ✅ Grafana 可看 | 已有 |
| `master_allocated_bytes` / `master_total_capacity_bytes` | Mooncake Master :9003 | ✅ Grafana 利用率 gauge | 已有 |
| Eviction 触发时机（CPU 利用率多高时） | Mooncake Master | ❌ 没有时间线 | 在 Grafana 中看 allocated_bytes 时间线和 eviction counter 的关联 |
| 驱逐前后的 get 延迟变化 | vLLM 日志 | ❌ 没有 | 通过时间线对比 `mooncake_store_get_lat_p95_ms`（新增的 p95 指标） |
| `MOONCAKE_NO_AVAILABLE_HANDLE` 背压 | vLLM prefill 日志 | ✅ 代码中有 warning | grep log 中是否出现 |

### C. 请求级别的延迟拆解

一个 70K 请求从到达到完成的完整生命周期：

```
请求到达 Router
  │
  ▼ Router routing delay (μs 级)
Prefill 端收到请求
  │
  ├─ [MooncakeStore lookup] prefix 查询延迟
  │   └─ batch_is_exist → Master RPC → 返回命中 token 数
  │
  ├─ [MooncakeStore get] prefix load 延迟（如果有 external cache hit）
  │   └─ CPU memory → RDMA → GPU VRAM
  │
  ├─ [Prefill compute] 剩余 tokens 的 prefill 计算
  │
  ├─ [MooncakeStore put] KV 写入 store（异步）
  │
  ├─ [Nixl transfer] KV 传输到 decode 端
  │   └─ GPU → RDMA → decode GPU
  │
  ▼
Decode 端开始 generation
  │
  ├─ [Decode forward] 每步 token 生成
  │
  └─ 完成
```

**需要做的**：
1. 在 prefill 日志中加 per-request 级别的计时：lookup 延迟、get 延迟、prefill 计算时间、Nixl post 时间
2. 或者用 nsys/pytorch profiler 抓 trace

### D. nsys profiling

```bash
# 在 YAML env 中加：
VLLM_TORCH_PROFILER_DIR: "{log_dir}/traces"
# 或者 vigil 的 profiling 配置：
profiling:
  mode: cuda
  # 或 mode: torch
```

nsys 能看到：
- GPU kernel 执行时间线
- RDMA DMA engine 活动（通过 CUDA event）
- cudaMemcpy / NCCL all-reduce
- Python GIL 竞争

**限制**：nsys trace 文件很大（GB 级），70K 请求会非常长。建议只 profile c=1 的单请求。

### E. PyTorch Profiler

```python
# vigil YAML 中加 profiling 配置
profiling:
  mode: torch
  # 自动 append --profiler-config 参数
```

PyTorch profiler 输出 Chrome trace（JSON），可以在 chrome://tracing 或 Perfetto 中看：
- 每个 forward step 的耗时
- attention kernel 时间
- KV transfer 时间（如果有 custom event）

### F. 还需要明确的内容

1. **Mooncake Store put 是否和 prefill forward 重叠**：当前 put 在后台线程，理论上和下一个请求的 prefill 重叠。需要确认实际是否有 GPU 资源竞争

2. **Mooncake Store get 和 prefill forward 的重叠**：get 在 recv thread，通过 RDMA DMA engine。但 RDMA 可能和 GPU forward 竞争 PCIe/NVLink 带宽

3. **Nixl 传输期间 decode 端在做什么**：Nixl 传 2.2 GB 要 521ms。这期间 decode 是在 idle 还是在处理其他请求的 generation？

4. **prefix lookup 延迟**：`batch_is_exist` 查询 Master 的 RPC 延迟是多少？每个请求都要查一次，如果 RPC 慢会直接加到 TTFT

5. **跨节点 vs 本地 RDMA 的带宽差异**：我们调研过 Mooncake 的 replica 是随机分配的。实际运行时有多少比例的 get 走了跨节点 RDMA vs 本地 loopback？

6. **decode 端 KV cache 89.5% 时的 attention kernel 性能退化**：大 batch + 接近满的 KV cache → attention 计算量大 → 和小 batch 相比的退化比例

---

## 实施优先级

| 优先级 | 任务 | 工作量 | 价值 |
|--------|------|--------|------|
| P0 | put/get 延迟 p50/p95/p99 输出 | 小（改 reduce()） | 高 |
| P0 | per-request 延迟拆解 log | 中（加 timer） | 高 |
| P1 | nsys profile c=1 单请求 | 小（加 YAML 配置） | 高 |
| P1 | prefix lookup RPC 延迟 | 小（加 timer） | 高 |
| P1 | 跨节点 vs 本地 RDMA 比例 | 中（需看 Master 分配日志） | 中 |
| P2 | pytorch profiler trace | 小（加 YAML 配置） | 中 |
| P2 | decode 端 attention kernel 退化分析 | 中（需 nsys） | 中 |

---

## 问题澄清

### Q2: `--kv-offloading-backend native` 和 `--kv-transfer-config` 不能同时用

`vllm/config/vllm.py:673` 中 `_post_init_kv_transfer_config()` 会**覆盖** `kv_transfer_config.kv_connector`：

```python
# line 668-673
if kv_offloading_backend == "native":
    config_connector = "SimpleCPUOffloadConnector"  # or OffloadingConnector
    self.kv_transfer_config.kv_connector = config_connector  # 覆盖！
```

这意味着如果之前 `--kv-transfer-config` 设了 NixlConnector，会被 `--kv-offloading-backend` 覆盖掉。`pd_tp.yaml` 里两个 `--kv-transfer-config` 参数 + `--kv-offloading-backend native` 的组合，最终 NixlConnector 被丢弃，实际只走 SimpleCPUOffload。

**PD 场景下正确的 offload 方式只有一种**：`MultiConnector` 嵌套，在 JSON 中同时包含 NixlConnector 和 offload connector。`--kv-offloading-backend` 只适用于非 PD 单实例场景。

### Q3: "External prefix cache hit" 的含义

`External prefix cache hit rate` 在 vLLM 中指的是**通过 KV connector 从外部获得的 computed tokens 占比**（`stats.py:283: self.external_kv_transfer += num_external_computed_tokens`）。

不同端含义不同：
- **Decode 端 100%**：所有 KV 都从 Nixl P→D 传输获得——这是 PD 分离的预期行为，不是从 decode 的 CPU memory 来
- **Prefill 端 74.8%**：74.8% 的 prefix tokens 从 MooncakeStore CPU memory 加载（不需要重新 prefill）——这才是 Mooncake 的 external cache hit

### Q3 补充: Mooncake Store 的 replica 在哪个节点

1P1D 配置中：
- 只有 prefill 端有 MooncakeStoreConnector → 只有 prefill 节点向 Master 注册了 CPU segment
- Decode 端只有 NixlConnector → decode 节点没有注册 Mooncake segment
- Master 的 `RandomAllocationStrategy` 只能从已注册的 segment 中分配
- **因此所有 replica 都在 prefill 节点的 CPU memory 上**，不会分到 decode 节点

如果 decode 端也配了 MooncakeStoreConnector，replica 可能被随机分配到 decode 节点（见 `kv_cache_data_paths_nvl72.md` 第 5 节分析）。

### Q5: 额外需要明确的内容

1. **Mooncake Store put 和 prefill forward 的 overlap 程度**
   - put 在后台 `KVCacheStoreSendingThread` 异步执行
   - 但 RDMA DMA 可能和 GPU forward 竞争 PCIe/NVLink 带宽
   - nsys trace 能看到 DMA engine 和 CUDA kernel 的时间线重叠
   - 关键问题：put 是否导致 forward 变慢？

2. **Mooncake Store get 和 prefill compute 的 overlap**
   - get 在 `KVCacheStoreRecvingThread` 异步执行
   - prefix load (195ms, 1.75 GiB) 和后续 prefill compute 是否真的重叠？
   - 如果 get 阻塞了 forward，TTFT 会增加

3. **跨节点 vs 本地 RDMA 的比例**
   - 我们的 1P1D 配置中，prefill 节点是唯一注册 segment 的节点
   - 所以 4 个 TP worker 的 CPU segment 都在同一节点上
   - Master RandomAllocationStrategy 从这 4 个 segment 中随机选
   - **所有 put/get 都是本地 RDMA loopback**（同节点内 GPU ↔ CPU）
   - 没有跨节点 RDMA（因为 decode 节点没注册 segment）

4. **Nixl 传输期间 decode 端是否 idle**
   - Nixl 传 2.2 GB 要 521ms
   - 这期间 decode 可能在处理其他请求的 generation（如果有多个并发请求）
   - 或者 idle（如果只有一个请求在等 KV）
   - 需要看 decode 端的 generation throughput 时间线

5. **Decode 端大 batch 时 attention kernel 的退化**
   - c=8 时 decode 端同时跑 5-6 个 70K 请求
   - KV cache usage 89.5%，attention 计算量 ~5x 于 c=1
   - TPOT 从 6.64ms (c=1) → 22.98ms (c=8)，~3.5x 退化
   - 需要 nsys 看 attention kernel 具体耗时

6. **prefix lookup（batch_is_exist）的 RPC 延迟**
   - 每个请求到 prefill 后第一步查 Master "这些 block 存在否"
   - 通过 Master gRPC 查询，延迟直接加到 TTFT
   - 70K tokens ≈ 4375 blocks（block_size=16），查询量不小
   - 需要在 lookup() 函数中加 timer 输出延迟
