# 计划：SimpleCPUOffloadConnector 可观测性

## A. CPU-GPU 传输带宽（~100 行，4 文件）

### A1. `copy_backend.py` — 在 copy 处加计时（~20 行）

在 `_copy_loop` 的 `copy_blocks()` 前后加 `time.perf_counter()`，把 (bytes, duration, is_store) 推到一个线程安全的 stats 列表。

```python
# _copy_loop 中：
t0 = time.perf_counter()
copy_blocks(src_blocks, dst_blocks, params)
elapsed = time.perf_counter() - t0
nbytes = len(src_blocks) * params.bytes_per_block
stats_list.append((nbytes, elapsed, is_store))
```

需要：给 `DmaCopyBackend` 加 `stats_list: list` + `bytes_per_block` 字段。

### A2. `worker.py` — 收集 stats 上报（~30 行）

`SimpleCPUOffloadWorker` 新增 `get_kv_connector_stats()`：
- 从 `_backend.stats_list` 中取出所有记录
- 分成 store (GPU→CPU) 和 load (CPU→GPU) 两组
- 构建 `SimpleCPUOffloadConnectorStats` 返回

### A3. 新建 `simple_cpu_offload_metrics.py`（~40 行）

照搬 `mooncake_store_metrics.py` 的模式：
- `SimpleCPUOffloadConnectorStats(KVConnectorStats)` — data dict 有 `put_bytes/put_duration/get_bytes/get_duration`
- `reduce()` 输出均值 + p50/p95/p99
- `SimpleCPUOffloadPromMetrics(KVConnectorPromMetrics)` — 4 个 histogram

### A4. `simple_cpu_offload_connector.py` — 接入 stats pipeline（~15 行）

加 3 个方法：
- `get_kv_connector_stats()` → 委托给 worker
- `build_kv_connector_stats(data)` → 构建 stats 对象
- `build_prom_metrics()` → 返回 PromMetrics 实例

---

## B. CPU Eviction Counter（~15 行，1 文件）

### B1. `manager.py` — 在 `get_new_blocks` 前后计数

`SimpleCPUOffloadScheduler.__init__` 加 `self.cpu_eviction_count = 0`。

两个 `get_new_blocks` 调用点（line 432, 550）前后对比 cached 数量：

```python
cached_before = len(self.cpu_block_pool.cached_block_hash_to_block)
cpu_blocks = cpu_pool.get_new_blocks(n)
evicted = cached_before - len(self.cpu_block_pool.cached_block_hash_to_block)
if evicted > 0:
    self.cpu_eviction_count += evicted
```

通过 A3 的 stats 对象把 `cpu_eviction_count` 带出去，或直接 `logger.info`。

---

## 文件清单

| 部分 | 文件 | 改动 |
|------|------|------|
| A1 | `vllm/v1/simple_kv_offload/copy_backend.py` | 加计时 + stats_list |
| A2 | `vllm/v1/simple_kv_offload/worker.py` | get_kv_connector_stats() |
| A3 | `vllm/distributed/.../v1/simple_cpu_offload_metrics.py` | **新建** Stats + PromMetrics |
| A4 | `vllm/distributed/.../v1/simple_cpu_offload_connector.py` | 3 个方法 |
| B1 | `vllm/v1/simple_kv_offload/manager.py` | eviction counter |
