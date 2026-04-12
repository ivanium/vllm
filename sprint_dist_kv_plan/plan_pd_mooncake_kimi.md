# PD 分离 + Mooncake CPU Offload + Kimi NVFP4 实施计划

## 最终目标

在 GB200 NVL72 集群上跑通 **PD 分离 + Mooncake 作为 L2 CPU offload 缓存** 的配置，模型为 `nvidia/Kimi-K2.5-NVFP4`。

## 当前状态

| 维度 | 现状 |
|------|------|
| 起始配置 | `vigil/examples/pd_1p1d_nixl_only_remote_01_03_ucx_py312_fix_less_post_serve_mooncake.yaml` |
| 模型 | Qwen/Qwen3-0.6B（小模型，用于快速验证） |
| 拓扑 | 1P1D，每实例 1GPU |
| KV 传输 | MultiConnector(NixlConnector + MooncakeStoreConnector) |
| PD 传输可观测性 | **已有** — nixl_connector.py 的 KV Transfer metrics |
| CPU-GPU offload 可观测性 | **待补齐** — Mooncake 侧的带宽/延迟指标尚未验证 |
| Mooncake 环境变量 | 已配置（MOONCAKE_ENABLE_OFFLOAD=1, CPU+Disk offload vars） |

### PD 传输已有指标（参考）

```
KV Transfer metrics: Num successful transfers=463, Avg xfer time (ms)=70.583,
  P90 xfer time (ms)=120.765, Avg post time (ms)=11.91, P90 post time (ms)=16.785,
  Avg MB per transfer=95.173, Throughput (MB/s)=1348.379, Avg number of descriptors=3045.5
```

---

## 阶段一：补齐 CPU-GPU Offload 可观测性（当前重点）

### 1.1 理解数据通路

当前 MooncakeStoreConnector 的 CPU offload 物理路径（参考 `sprint_dist_kv_docs/mooncake_store_cpu_offload_full_stack.md`）：

```
GPU→CPU offload:  GPU VRAM → ibv_reg_dmabuf_mr → IBV_WR_RDMA_WRITE → RNIC loopback → CPU Host DRAM
CPU→GPU load:     CPU Host DRAM → ibv_reg_mr → IBV_WR_RDMA_READ → RNIC loopback → GPU VRAM
```

> **关于 RDMA loopback vs NVLink-C2C**: 即便在 GB200 上有 NVLink-C2C (900 GB/s)，Mooncake 默认用的是 RDMA loopback，走 RNIC DMA engine。这不一定更慢 —— RDMA 由 NIC 的 DMA engine 发起，零 CPU 参与，延迟确定性更好；C2C 走的是 CPU 页表 + ATS 翻译，对小块随机访问反而可能有更高的软件栈开销。**关键是看 offload 场景是 latency-bound 还是 throughput-bound** —— 如果是大批量顺序搬运（throughput），C2C 的 900 GB/s 理论峰值远超 RDMA loopback；如果是按需的小块传输，RDMA 的 DMA offload 特性（不占 GPU SM / CPU core）反而更适合。先跑起来看真实数据再判断。

### 1.2 CPU-GPU 传输带宽：看哪个指标

**核心指标明确如下**：

| 指标名 | 来源 | 含义 | 测量范围 |
|--------|------|------|---------|
| **`gpu_to_cpu_store_throughput_mib_s`** | vLLM Python 层日志 | GPU→CPU offload 带宽 (MiB/s) | 纯 RDMA DMA 传输时间（不含排队） |
| **`cpu_to_gpu_load_throughput_mib_s`** | vLLM Python 层日志 | CPU→GPU load 带宽 (MiB/s) | 纯 RDMA DMA 传输时间（不含排队） |
| `mooncake_transfer_write_bytes` | Mooncake C++ Prometheus | 累计写入字节数 (counter) | RDMA 传输引擎级别 |
| `mooncake_transfer_read_bytes` | Mooncake C++ Prometheus | 累计读取字节数 (counter) | RDMA 传输引擎级别 |
| `mooncake_transfer_batch_put_latency` | Mooncake C++ Prometheus | 批量写延迟直方图 (μs) | RDMA 传输引擎级别 |
| `mooncake_transfer_batch_get_latency` | Mooncake C++ Prometheus | 批量读延迟直方图 (μs) | RDMA 传输引擎级别 |

**计时方式详解**（`mooncake_store_worker.py`）：

```python
# GPU→CPU (KVCacheStoreSendingThread, line 715-725):
cuda_event.synchronize()          # 先等 GPU 侧数据就绪
start_time = time.perf_counter()  # ← 计时开始
res = self.store.batch_put_from_multi_buffers(keys, addrs, sizes)  # RDMA DMA
elapsed_s = time.perf_counter() - start_time  # ← 计时结束
self.stats.record_transfer(successful_bytes, elapsed_s, "gpu_to_cpu_store")

# CPU→GPU (KVCacheStoreRecvingThread, line 974-986):
start_time = time.perf_counter()  # ← 计时开始
res = self.store.batch_get_into_multi_buffers(keys, addrs, sizes)  # RDMA DMA
elapsed_s = time.perf_counter() - start_time  # ← 计时结束
self.stats.record_transfer(successful_bytes, elapsed_s, "cpu_to_gpu_load")
```

**关键：计时测的是 `batch_put`/`batch_get` 的纯 DMA 传输时间，不包含**：
- CUDA event 同步等待
- 排队等待
- CPU 侧数据准备
- 事后的元数据更新

**throughput 计算（`mooncake_store_metrics.py:55`）**：
```python
throughput_mib_s = total_mib / total_time   # total_mib = total_bytes / 2^20
```

> 这个值反映的是 **RDMA DMA engine 的实际搬运速率**，是判断互联带宽利用率的核心数据。

### 1.3 可用的观测层（三层）与 Grafana 统一看板方案

#### 层 1：vLLM Python 层 — 文本日志 + Prometheus

**文件**: `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_metrics.py`

**文本日志（已有，自动输出）**：
通过 `KVConnectorLogging.log()` 周期性输出到 vLLM 日志：
```
KV Transfer metrics: gpu_to_cpu_store_num_transfers=X, gpu_to_cpu_store_throughput_mib_s=Y, ...
```

**Prometheus 导出（当前缺失，需要开发）**：
MooncakeStoreConnector **尚未实现 `build_prom_metrics()`**，所以这些指标目前 **只在文本日志里，不在 vLLM 的 `/metrics` Prometheus 端点中**。

NixlConnector 已有参考实现（`nixl_connector.py` 中的 `NixlPromMetrics` 类）。要把 Mooncake 指标也加到 vLLM Prometheus，需要：

```
实现路径：
  MooncakeStoreConnector
    └─ build_prom_metrics() → 返回 MooncakePromMetrics 实例
         └─ __init__(): 注册 Gauge/Counter/Histogram 到 prometheus_client
         └─ observe(): 从 transfer_stats_data 中读取值并更新 Prometheus metrics

注册链路：
  PrometheusStatLogger.__init__()
    → KVConnectorProm(vllm_config)
      → connector_cls.build_prom_metrics(...)
        → MooncakePromMetrics(...)    ← 新增

采集链路（每次 scheduler iteration 自动触发）：
  PrometheusStatLogger.record()
    → kv_connector_prom.observe(transfer_stats_data)
      → MooncakePromMetrics.observe(...)    ← 新增
```

**参考 NixlConnector 的实现**（`nixl_connector.py:490+, 3030+`）：
- `NixlPromMetrics` 在 `__init__` 中用 `self._histogram_cls()` 注册 histogram
- 用 `create_metric_per_engine()` 创建 per-engine 标签化的 metric 实例
- 在 `observe()` 中逐条调用 `.observe()` 更新 prometheus 值

#### 层 2：Mooncake C++ ClientMetric — RDMA 传输级指标

**文件**: `Mooncake/mooncake-store/include/client_metric.h`

这些指标运行在 **vLLM 进程内**（因为 Mooncake store client 是 in-process 的 C++ library），但走的是 C++ 自己的 metric 体系（ylt::metric），**不是 Python prometheus_client**。

| 指标 | 类型 | 说明 |
|------|------|------|
| `mooncake_transfer_read_bytes` | counter | 累计读字节数 |
| `mooncake_transfer_write_bytes` | counter | 累计写字节数 |
| `mooncake_transfer_batch_put_latency` | histogram (μs) | 批量写延迟 |
| `mooncake_transfer_batch_get_latency` | histogram (μs) | 批量读延迟 |

**如何启用**:
```yaml
env:
  MC_STORE_CLIENT_METRIC: "1"            # 默认已启用
  MC_STORE_CLIENT_METRIC_INTERVAL: "10"  # 每 10 秒输出 summary 到 stderr
```

> 注意：这些 C++ metrics 目前只输出到 stderr（日志），不暴露 HTTP 端点。要入 Grafana 需要 log parsing 或改造。

#### 层 3：Mooncake Master Prometheus 端点 — 全局缓存状态

**已原生支持 Prometheus**。Master 进程用 `ylt::coro_http_server` 在端口 9003 暴露 `/metrics`：

```
HTTP GET http://<master_ip>:9003/metrics
Content-Type: text/plain; version=0.0.4    ← 标准 Prometheus OpenMetrics 格式
```

关键指标（均为标准 Prometheus 格式，可直接被 Prometheus scrape）：
- `master_key_count` — 缓存中的 key 数量
- `master_value_size_bytes_sum` / `master_value_size_bytes_bucket` — 缓存占用字节数
- `master_allocated_bytes` — 已分配字节数
- `master_put_start_requests_total` — put 请求计数
- `master_evict_*` — 驱逐相关指标
- `master_offload_*` — offload 相关指标
- `segment_allocated_bytes` — 按 segment 的分配量（带 label）

Master 还有 `/metrics/summary`（人类可读）和 `/health`（JSON 健康检查）。

**已有工具**: `watch_offload.sh` 可实时拉取这些指标。

### 1.4 监控容器部署（Prometheus + Grafana）

**已创建**: `monitoring/` 目录，`docker compose up -d` 即可使用。

```bash
cd /home/aoshen/setup_new_cluster/monitoring/
docker compose up -d

# 访问：
#   Grafana:    http://<host>:3000  (admin/admin, 匿名可读)
#   Prometheus: http://<host>:9090
```

**目录结构**：
```
monitoring/
├── docker-compose.yml                          # Prometheus + Grafana
├── prometheus.yml                              # scrape config（Mooncake Master :9003）
└── grafana/
    ├── provisioning/
    │   ├── datasources/prometheus.yml          # 自动注册 Prometheus 数据源
    │   └── dashboards/default.yml              # 自动加载 dashboard
    └── dashboards/
        └── mooncake-overview.json              # 预配置 Mooncake 看板
```

**预配置 Dashboard 包含以下 Panel**：

| Panel | 指标 | 说明 |
|-------|------|------|
| Cache Keys | `master_key_count` | 当前缓存的 key 数量 |
| Allocated Memory | `master_allocated_bytes` | 已分配的 CPU 内存 |
| Total Capacity | `master_total_capacity_bytes` | 总容量 |
| Memory Utilization | allocated / capacity | 内存利用率（gauge 带阈值色） |
| Active Clients | `master_active_clients` | 活跃的 vLLM client 数 |
| Cache Hit Rate | `mem_cache_hit_nums_` / `file_cache_hit_nums_` | **内存 vs SSD 缓存命中率** |
| Eviction Activity | `master_successful_evictions_total` | LRU 驱逐活动 |
| Put/Get Request Rate | `master_put_start_*` / `master_get_replica_list_*` | 请求吞吐 |
| Failure Rate | `*_failures_total` | 错误率 |
| Value Size Distribution | `master_value_size_bytes_bucket` | KV block 大小分布 (heatmap) |
| Per-Segment Allocation | `segment_allocated_bytes` | 按 segment 的内存分配 |
| Memory vs File Cache Count | `mem_cache_nums_` / `file_cache_nums_` | **内存 vs SSD 中的对象数量** |
| Disk Offload Activity | `master_evict_disk_replica_*` | disk offload 请求频率 |

**部署后需要做的**：
1. 确认 Mooncake Master 已启动（`curl http://192.168.0.101:9003/metrics`）
2. 如果 Master 不在 192.168.0.101，修改 `prometheus.yml` 中的 target
3. 后续加 vLLM 实例时，取消注释 `prometheus.yml` 中的 vllm-prefill / vllm-decode job

**后续可选（需改 vLLM 代码）**: 把 Python 层的 offload throughput 也暴露到 vLLM `/metrics`

MooncakeStoreConnector 尚未实现 `build_prom_metrics()`。NixlConnector 已有参考实现 (`NixlPromMetrics`, `nixl_connector.py:490+,3030+`)，照着写一个即可把 `gpu_to_cpu_store_throughput_mib_s` 等打到 vLLM 的 Prometheus 端点。但这不是第一优先级——Master 端的指标已经能覆盖缓存状态和命中率。

### 1.5 Disk↔CPU 传输带宽：不可观测（当前限制）

**结论：Disk→CPU 传输带宽在当前 Mooncake 实现中是 opaque 的，没有单独的 metric。**

数据流是这样的：

```
offload 回读路径（SSD → GPU）:
  SSD NVMe ──(io_uring)──> ClientBuffer (CPU memory)  ← 不可观测，无 metric
       └─ FileStorage::BatchGet()
       └─ 只有 VLOG(1) 级日志（生产环境不开）

  ClientBuffer ──(RDMA DMA)──> GPU VRAM               ← 可观测
       └─ batch_get_into_multi_buffers()
       └─ 记录为 "cpu_to_gpu_load" metric

offload 写入路径（GPU → SSD）:
  GPU VRAM ──(RDMA DMA)──> Mooncake CPU segment        ← 可观测
       └─ batch_put_from_multi_buffers()
       └─ 记录为 "gpu_to_cpu_store" metric

  Mooncake CPU segment ──(io_uring)──> SSD NVMe        ← 不可观测
       └─ FileStorage::OffloadObjects() / BucketStorageBackend::BatchOffload()
       └─ 后台异步，无 metric
```

**从 vLLM 层面看到的 `cpu_to_gpu_load` 是 end-to-end 的**：当数据需要从 SSD 回读时，`batch_get_into_multi_buffers()` 内部会先 SSD→ClientBuffer 再 ClientBuffer→GPU，但 vLLM Python 层计时包含了两段，无法区分。

**间接判断方法**：
- 对比 `cpu_to_gpu_load_avg_time_ms`：如果某些 load 特别慢（比纯 RDMA 慢很多），大概率是 SSD 回读
- Grafana 中看 `mem_cache_hit_nums_` vs `file_cache_hit_nums_` 的比值：file hit 越多说明 SSD 参与越多
- Grafana 中看 `file_cache_nums_` 的增长：表示有多少对象被 offload 到了 SSD

### 1.6 重要指标全览

| 指标 | 来源 | 类型 | 含义 | 优先级 |
|------|------|------|------|--------|
| **`gpu_to_cpu_store_throughput_mib_s`** | vLLM 日志 | gauge | GPU→CPU RDMA 搬运带宽 | P0 |
| **`cpu_to_gpu_load_throughput_mib_s`** | vLLM 日志 | gauge | CPU→GPU RDMA 搬运带宽（含可能的 SSD 回读） | P0 |
| **`master_key_count`** | Master :9003 | gauge | 缓存中的 key 总数 | P0 |
| **`master_allocated_bytes`** | Master :9003 | gauge | CPU 内存池已使用量 | P0 |
| **`mem_cache_hit_nums_`** | Master :9003 | counter | 内存层（L2）缓存命中次数 | P0 |
| **`file_cache_hit_nums_`** | Master :9003 | counter | SSD 层（L3）缓存命中次数 | P0 |
| **`mem_cache_nums_`** | Master :9003 | gauge | 在内存中的对象数 | P1 |
| **`file_cache_nums_`** | Master :9003 | gauge | 在 SSD 中的对象数 | P1 |
| **`master_successful_evictions_total`** | Master :9003 | counter | 成功驱逐次数 | P1 |
| **`master_evicted_size_bytes`** | Master :9003 | counter | 驱逐的总字节数 | P1 |
| `master_evict_disk_replica_requests_total` | Master :9003 | counter | disk replica 驱逐请求 | P1 |
| `master_total_capacity_bytes` | Master :9003 | gauge | 总容量 | P1 |
| `master_active_clients` | Master :9003 | gauge | 活跃 client 数 | P2 |
| `master_put_start_requests_total` | Master :9003 | counter | put 请求数 | P2 |
| `master_put_start_discard_cnt` | Master :9003 | counter | 被丢弃的 put（背压指标） | P1 |
| `master_value_size_bytes` | Master :9003 | histogram | KV block 大小分布 | P2 |
| `segment_allocated_bytes{segment=...}` | Master :9003 | gauge | 按 segment 的分配量 | P2 |
| `mooncake_transfer_batch_put_latency` | C++ stderr | histogram (μs) | RDMA 批量写延迟 | P1（仅日志） |
| `mooncake_transfer_batch_get_latency` | C++ stderr | histogram (μs) | RDMA 批量读延迟 | P1（仅日志） |

> **Disk↔CPU 带宽没有直接指标**。只能通过 `file_cache_hit_nums_` 的 rate 和 `cpu_to_gpu_load` 延迟的分布间接推断。

### 1.7 验证步骤

#### Step 1: 确认 Mooncake Master 存活并可 scrape

```bash
# 检查 master 进程
ps aux | grep mooncake_master

# 验证 Prometheus 格式输出
curl -s http://192.168.0.101:9003/metrics | head -30
# 期望看到：# HELP master_key_count ...
#           # TYPE master_key_count gauge
#           master_key_count 0

# 检查 metadata server
curl -s http://192.168.0.101:8080/metadata?key=test
```

#### Step 2: 在起始 YAML 中增加 C++ 侧 metric reporting

在 prefill 和 decode 的 env 中添加：

```yaml
env:
  # ... 现有变量 ...
  # Mooncake C++ client metric reporting (输出到 stderr)
  MC_STORE_CLIENT_METRIC: "1"
  MC_STORE_CLIENT_METRIC_INTERVAL: "10"
  MC_STORE_CLUSTER_ID: "pd_1p1d_test"
```

#### Step 3: 用小模型跑一次，确认能看到以下日志

**期望在 vLLM 日志中看到（Python 层 — 文本日志）**:
```
KV Transfer metrics: gpu_to_cpu_store_num_transfers=..., gpu_to_cpu_store_throughput_mib_s=..., gpu_to_cpu_store_avg_time_ms=...
```

**期望在 vLLM 进程 stderr 中看到（C++ 层）**:
```
=== Transfer Metrics Summary ===
Total Read: X.XX GiB
Total Write: Y.YY GiB

=== Latency Summary (microseconds) ===
Batch Get: count=N, p95<Xμs, max<Yμs
Batch Put: count=N, p95<Xμs, max<Yμs
```

**期望在 Master Prometheus 端点看到**:
```bash
curl -s http://192.168.0.101:9003/metrics | grep master_key_count
# master_key_count > 0 （有数据被 offload 到 store 后）
```

#### Step 4: 配置 Prometheus scrape + Grafana dashboard

```bash
# 先手动验证 Mooncake master 的 /metrics 格式能被 Prometheus 解析
curl -s http://192.168.0.101:9003/metrics | promtool check metrics
```

然后在 Grafana 中创建 dashboard，分别查询两个 data source。

#### Step 5: 发送足够请求触发 offload

当前 post_serve 配置的 benchmark (100 prompts, speed-bench throughput_16k) 可能不足以触发 offload。
需要加大 working set：

```bash
# 单独手动发送长 prompt 测试
curl http://127.0.0.1:<router_port>/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-0.6B","messages":[{"role":"user","content":"<very_long_text>"}],"max_tokens":128}'
```

或修改 post_serve 中的 benchmark 配置，增大 num_prompts 和 max_concurrency。

### 1.8 潜在问题排查

| 问题 | 排查方法 |
|------|---------|
| Python 层没有 gpu_to_cpu_store 指标 | 检查 MooncakeStoreConnector 是否真的触发了 offload。GPU 显存未满时不会 offload |
| C++ 层没有输出 | 确认 MC_STORE_CLIENT_METRIC_INTERVAL > 0 |
| Master metrics 端点不可达 | 确认 master 进程存活，检查防火墙/端口 |
| Offload 从未触发 | 需要增大 working set 使 GPU HBM 不够用。小模型可能需要大量并发 |
| RDMA loopback 性能异常低 | 检查 `ibv_devinfo`，确认 RNIC 状态正常；注意 loopback 性能取决于 NIC DMA engine 能力，不等于远端 RDMA 性能 |
| throughput 数值解读 | `throughput_mib_s` 是纯 DMA 时间计算的，不含排队。如果批量够大、并发够高，实际端到端会低于此值 |

---

## 阶段二：确认 PD + Mooncake Offload 基本功能（小模型）

### 2.1 目标

用 Qwen3-0.6B 验证 PD 分离 + Mooncake CPU offload 全链路可运行、无 crash，且观测指标完整。

### 2.2 检查项

- [ ] prefill 和 decode 都能正常启动
- [ ] PD 之间的 KV 传输正常（NixlConnector metrics 正常）
- [ ] Mooncake offload 触发（看到 gpu_to_cpu_store 指标）
- [ ] Mooncake load 触发（看到 cpu_to_gpu_load 或类似指标）
- [ ] Master metrics 能看到 key_count > 0
- [ ] 无 MOONCAKE_NO_AVAILABLE_HANDLE (-200) 错误
- [ ] vmon 能采集到 GPU 利用率数据

### 2.3 简单 prompt 验证 cache hit

1. 发送一个 prompt A，等待完成
2. 再次发送相同的 prompt A
3. 第二次应该看到 external cache hit（从 Mooncake CPU 层命中）
4. 检查 LookupKeyServer 的日志或 prefix cache hit 指标

---

## 阶段三：逐步迁移到 Kimi NVFP4 配置

### 3.1 配置差异对比

从起始 YAML 到最终 Kimi 配置，需要逐步增加以下变更：

| 维度 | 起始配置 (Qwen3-0.6B) | 目标配置 (Kimi NVFP4) |
|------|----------------------|----------------------|
| 模型 | Qwen/Qwen3-0.6B | nvidia/Kimi-K2.5-NVFP4 |
| TP | 1 | 4 |
| GPU/instance | 1 | 4 |
| nodes/instance | 1 | 1 |
| gpu-memory-utilization | 0.85 (P) / 0.9 (D) | 0.85 (P) / 0.85 (D) |
| max-model-len | 默认 | 131072 |
| max-num-seqs | 默认 | 16 |
| max-num-batched-tokens | 默认 | 8192 |
| KV cache dtype | 默认 | fp8 |
| Attention | 默认 | FA4 + disable_flashinfer_prefill |
| Speculative decode | 无 | Eagle3 (kimi-k2.5-eagle3) |
| Compile mode | enforce-eager 两端 | P: enforce-eager, D: FULL_DECODE_ONLY |
| Prefix caching | 关 | P: 开 + chunked_prefill |
| Load format | 默认 | fastsafetensors |
| Language model only | 否 | 是 |
| MoE 环境变量 | 无 | VLLM_USE_FLASHINFER_MOE_FP4 等 |
| NCCL 配置 | 基础 | 完整 GB200 NVL72 配置 |
| Mooncake global_segment_size | 5GB | 需按 kimi 调整（每 rank ~175 GiB for 700G total） |
| Sleep mode | 有 | 有 |

### 3.2 分步迁移路径

> 每一步都要确认能正常启动、无 crash、观测指标完整后再继续下一步。

#### Step A: 模型切换 + TP 扩展

```diff
- model: Qwen/Qwen3-0.6B
+ model: nvidia/Kimi-K2.5-NVFP4

  roles:
    - role: prefill
-     gpus_per_node: 1
+     gpus_per_node: 4

    - role: decode
-     gpus_per_node: 1
+     gpus_per_node: 4

  # cmd 中添加：
+   -tp 4
+   --trust-remote-code
+   --language-model-only
+   --load-format fastsafetensors
+   --max-model-len 131072
+   --max-num-seqs 16
+   --max-num-batched-tokens 8192
```

#### Step B: Attention + KV Cache 优化

```diff
  # prefill 和 decode 都加：
+   --attention_config.flash_attn_version 4
+   --attention_config.disable_flashinfer_prefill true
+   --kv-cache-dtype fp8
+   --disable-hybrid-kv-cache-manager
```

#### Step C: Prefill 端 prefix caching + chunked prefill

```diff
  # 仅 prefill 端：
+   --enable-prefix-caching
+   --enable-chunked-prefill
```

#### Step D: Decode 端 cudagraph + compile 优化

```diff
  # 仅 decode 端：
-   --enforce-eager
+   -O3
+   --compilation_config.pass_config.fuse_allreduce_rms true
+   --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
```

#### Step E: Speculative decoding (Eagle3)

```diff
  # 两端都加：
+   --speculative-config '{"model": "lightseekorg/kimi-k2.5-eagle3", "method": "eagle3", "num_speculative_tokens": 3, "rejection_sample_method": "synthetic", "synthetic_acceptance_rate": 0.6}'
```

#### Step F: MoE 环境变量

```diff
  env:
+   VLLM_USE_FLASHINFER_MOE_FP4: "1"
+   VLLM_FLASHINFER_MOE_BACKEND: latency
+   VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE: "8192"
+   VLLM_ENABLE_MOE_DP_CHUNK: "0"
+   VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD: "8192"
```

#### Step G: GB200 NVL72 网络完整配置

```diff
  env:
+   NCCL_SOCKET_IFNAME: enp192s2
+   NCCL_IB_HCA: "mlx5_1,mlx5_2,mlx5_3,mlx5_4"
+   GLOO_SOCKET_IFNAME: enp192s2
    # 超时
+   VLLM_ENGINE_READY_TIMEOUT_S: "3600"
+   VLLM_RPC_TIMEOUT: "600000"
+   VLLM_LOG_STATS_INTERVAL: "1"
```

#### Step H: 调整 Mooncake global_segment_size

Kimi 模型的 KV cache 比 Qwen3-0.6B 大得多，需要增大 Mooncake CPU 内存池：

```json
// mooncake_config_gb200_rack1_01.json
{
  "global_segment_size": "175GB",  // 原来 5GB，需按 kimi 调整
  "local_buffer_size": "4GB",
  ...
}
```

> 注意：`global_segment_size` 是 **per rank** 的。TP4 下实际占 4 × 175 = 700 GiB CPU 内存。
> 需确认节点 CPU 内存足够（GB200 每节点 ~882 GiB LPDDR5X）。

---

## 阶段四：Benchmark 外部缓存命中率与性能提升

### 4.1 目标

量化 Mooncake CPU offload 带来的性能提升，特别是 external_cache_hit 率。

### 4.2 基线对比

需要跑三组配置进行对比：

| 配置 | 参考 YAML | 用途 |
|------|-----------|------|
| Non-PD baseline | `golden/tp4_eagle_fa4_offloading_c8.yaml` | SimpleCPUOffloadConnector, 单实例 |
| PD w/o offloading | `pd_tp_no_offload_fa4.yaml` | 纯 NixlConnector, 无 offload |
| PD w/ Mooncake offload | 本计划最终产出 | MultiConnector(Nixl + MooncakeStore) |

### 4.3 Benchmark 策略

**关键：要触发 offload 才能看到效果。** 需要足够大的 working set 使 GPU HBM 放不下所有 KV cache。

推荐 benchmark 参数：

```yaml
# 长上下文多轮对话 — 更容易触发 offload
- cmd: >-
    vllm-bench
    --backend openai-chat
    --base-url http://localhost:{router_port}
    --model nvidia/Kimi-K2.5-NVFP4
    --dataset-name random
    --random-input-len 70000
    --random-output-len 300
    --multi-turn
    --multi-turn-num-turns 10
    --sweep-max-concurrency 1,2,4,8
    --sweep-num-prompts-factor 3
    --multi-turn-delay-ms 250
    --multi-turn-prefix-global-ratio 0.15
    --multi-turn-prefix-conversation-ratio 0.75
    --percentile-metrics "ttft,tpot,itl,e2el"
    --result-dir {log_dir}
    --save-result
```

### 4.4 关键观测指标

| 指标 | 含义 | 期望趋势 |
|------|------|---------|
| `external_cache_hit` | 从 Mooncake CPU 层命中的 KV blocks 比例 | > 0, 越高越好 |
| `gpu_to_cpu_store_throughput_mib_s` | GPU→CPU offload 带宽 | 接近 RDMA loopback 理论值 |
| `master_key_count` | Mooncake 缓存的 key 数 | 随请求增长 |
| `master_evict_count` | LRU 驱逐次数 | 表示缓存已满在正常工作 |
| TTFT (P50) | 首 token 延迟 | PD+offload 应优于非 PD |
| TPOT (P50) | 每 token 延迟 | PD decode 端应低于非 PD |

---

## 阶段五：磁盘 Offload（低优先级）

### 5.1 限制

当前 Mooncake 实现的限制：
- 单节点多 GPU：**只有 CPU offload 能工作，disk offload 不行**
- 多节点单 GPU：CPU + disk 都能工作
- 多节点多 GPU：**只有 CPU offload 能工作**

因此 Kimi NVFP4 TP4 场景（单节点 4 GPU）**disk offload 不可用**。

### 5.2 如果未来需要

等 Mooncake 修复了多 GPU 节点的 disk offload 后，需要：
1. 确认 `/mnt/data` (RAID0, ~12T) 已挂载
2. 配置 disk offload 环境变量（当前 YAML 中已预设）
3. 用 io_uring 加速 (`MOONCAKE_USE_URING: "true"`)

---

## 阶段进度

| 阶段 | 状态 | 完成日期 |
|------|------|---------|
| 阶段一：可观测性 | ✅ 完成 | 2026-04-10 |
| 阶段二：小模型验证 | ✅ 完成 | 2026-04-10 |
| 阶段三：Kimi NVFP4 配置迁移 | ✅ 完成（iter 1-7） | 2026-04-10 |
| 阶段四：Benchmark + 性能对比 | ✅ 完成（70K benchmark） | 2026-04-11 |
| 阶段五：磁盘 Offload | ⏸️ 低优先级（单节点多GPU不支持） | — |

### 阶段四详细进度

- ✅ 去掉 `--load-format dummy`，真实权重加载
- ✅ 对齐全部配置（-O3, MoE FP4, NCCL, 超时等）
- ✅ Mooncake global_segment_size 调到 100GB/worker
- ✅ 70K 长上下文 multi-turn benchmark（sweep concurrency 1/2/4/8）
- ✅ 三组对比：MooncakeStore vs SimpleCPUOffload vs Native Offload
- ✅ 性能分析完成（TTFT/TPOT/吞吐/缓存命中率/传输带宽）

---

## 实验结果文档

| 文档 | 内容 |
|------|------|
| [`experiment_results_70k_benchmark.md`](vllm/sprint_dist_kv_docs/experiment_results_70k_benchmark.md) | 70K benchmark 三组对比：Mooncake vs SimpleCPU vs Native，含性能表格、缓存命中率、传输带宽、decode 负载分析 |
| [`profiling_plan.md`](vllm/sprint_dist_kv_docs/profiling_plan.md) | 深度 Profiling 计划：需要明确的指标、nsys/pytorch profiler 方案、问题澄清（offload 配置正确性、external cache hit 含义、RDMA 本地/远程比例） |
| [`kv_cache_data_paths_nvl72.md`](vllm/sprint_dist_kv_docs/kv_cache_data_paths_nvl72.md) | KV Cache 数据路径全解析：6 种来源、replica 选择、RDMA 批量机制、数据放置随机性 |
| [`mooncake_store_cpu_offload_full_stack.md`](vllm/sprint_dist_kv_docs/mooncake_store_cpu_offload_full_stack.md) | Mooncake Store CPU offload 完整调用栈 |

---

## 参考文件索引

| 文件 | 用途 |
|------|------|
| `vigil/examples/pd_kimi_bench_a_70k.yaml` | **Mooncake Store benchmark 配置** |
| `vigil/examples/pd_kimi_bench_a_70k_baseline_simplecpu.yaml` | **SimpleCPU baseline** |
| `vigil/examples/pd_kimi_bench_a_70k_baseline_native.yaml` | **Native offload baseline** |
| `vigil/examples/pd_kimi_iter7_eagle3_nccl.yaml` | 最终 iter7 配置（Mooncake + 全部优化） |
| `vigil/examples/pd_1p1d_nixl_only_remote_01_03_ucx_py312_fix_less_post_serve_mooncake.yaml` | **起始配置** |
| `vigil/recipes/crusoe/kimik25/low_latency/golden/tp4_eagle_fa4_offloading_c8.yaml` | Non-PD baseline (SimpleCPUOffload) |
| `vigil/recipes/crusoe/kimik25/low_latency/pd_tp_no_offload_fa4.yaml` | PD baseline (无 offload) |
| `vigil/recipes/crusoe/kimik25/low_latency/pd_tp_offload_fa4.yaml` | PD + SimpleCPUOffload (Nixl 传输) |
| `vigil/recipes/crusoe/kimik25/low_latency/pd_tp_offload_fa4_mooncake.yaml` | PD + SimpleCPUOffload (Mooncake 传输) |
| `vigil/recipes/crusoe/kimik25/low_latency/dev/tp4_eagle_fa4_mooncake_c8.yaml` | Non-PD + MooncakeStore (disk offload) |
| `vllm/sprint_dist_kv_docs/mooncake_store_cpu_offload_full_stack.md` | Mooncake CPU offload 调用栈分析 |
| `vllm/sprint_dist_kv_docs/doc.md` | GB200 集群架构与 KV cache 分层文档 |
| `vllm/scripts/mooncake/mooncake_config_gb200_rack1_01.json` | Mooncake 配置文件 |
| `vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_metrics.py` | Python 层 metrics |
| `vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_worker.py` | Worker 核心逻辑 |
| `Mooncake/mooncake-store/include/client_metric.h` | C++ 层 metrics 定义 |
| `watch_offload.sh` | Master metrics 实时监控脚本 |

---

## 核心发现与注意事项

1. **Mooncake Store 在高并发下吞吐领先 46%**（0.76 vs 0.52 req/s @c=8），TTFT 降低 81%（2688 vs 14336 ms）。核心优势是 74.8% external cache hit rate。

2. **TPOT 更高是 decode 端更忙的副作用**（22.98 vs 12.25 ms @c=8）。Mooncake prefill 更快 → 更多请求涌入 decode → 大 batch decode → TPOT 高。但总 E2EL 仍优于 baseline。

3. **Nixl P→D 传输是低并发时 TTFT 的主要成分**（占 63-77% @c=1）。每个 70K 请求搬 2.2 GB KV，平均 521ms。高并发时排队成为主因。

4. **Mooncake Store offload 带宽**：put 11.2 GiB/s（小批 60MiB/5ms），get 9.2 GiB/s（大批 1.75GiB/195ms）。Read:Write 比 12:1，prefix 被大量复用。

5. **Mooncake Master 需要保活**：Master 进程可能 crash（之前遇到过），需要在 rack1-01 上运行 `start_mooncake_master.sh --enable-offload --bg`。

6. **global_segment_size 不能太大**：175GB/worker × 4 = 700GB 超出节点可用内存（权重已占 139GB）。当前用 100GB/worker 稳定运行。

7. **Native OffloadingConnector 不能嵌在 MultiConnector 中**：会 crash。正确做法是 `--kv-offloading-backend native` + `VLLM_USE_SIMPLE_KV_OFFLOAD=1` 独立配置。

8. **FlashInfer MLA JIT 首次编译 32 分钟**：在 Blackwell 上首次运行极慢，之后缓存在共享文件系统上（16s profiling）。

9. **Prometheus 可观测性已完成**：vLLM 日志输出 `mooncake_store_put/get_throughput_mib_s`，`build_prom_metrics()` 已实现（可选入 Grafana），Master :9003/metrics 有 Grafana dashboard。
