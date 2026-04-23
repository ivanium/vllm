---
title: SGLang 默认 Mooncake 配置 & 环境变量全景调研
audience: 运维/部署工程师 + vLLM/SGLang Mooncake reviewer
last_verified: 2026-04-23
repo_path: vllm/third_partys/sglang
scope: |
  SGLang 一个 Mooncake worker 启动时读哪些 env / CLI / JSON,
  每个字段的默认值、优先级、HA vs 单 master 的差异;
  以及 Mooncake 在 SGLang 里的 5 个子系统视角 + 对 vLLM 的 cross-project 对比。
  所有代码引用均在本地 worktree grep 验证。
related_docs:
  - k8s_mooncake_ha_deployment_notes.md              (HA 上层说明)
  - mooncake_store_cpu_offload_full_stack_cn.md      (vLLM 侧 CPU offload 全栈)
  - vllm_mooncake_store_connector_system_design_v1_cn.md  (vLLM connector 设计)
---

# SGLang 默认 Mooncake 配置 & 环境变量全景调研

## §0 · 起点:生产 k8s YAML

```yaml
- name: MOONCAKE_TE_META_DATA_SERVER
  value: P2PHANDSHAKE
- name: MOONCAKE_MASTER
  value: etcd://etcd-client.mooncake-ha.svc.cluster.local:2379
- name: MOONCAKE_PROTOCOL
  value: rdma
- name: MOONCAKE_DEVICE
  value: mlx5_10,mlx5_11,mlx5_12,mlx5_13,mlx5_14,mlx5_15,mlx5_16,mlx5_17
- name: MOONCAKE_GLOBAL_SEGMENT_SIZE
  value: '0'
- name: MC_TE_METRIC
  value: 'true'
```

HA (多 master) 部署, `MOONCAKE_MASTER` 填 **etcd 集群地址**, master 通过 etcd 选主。单 master 直接写 `10.11.xx.xx:50051` (默认 port 50051)。

---

## §1 · Mooncake 在 SGLang 里的五重角色

理解 env 配置的前提: **Mooncake 在 SGLang 里出现在至少 5 个互不相关的子系统**, 有些共用同一个 `MooncakeTransferEngine` 进程级单例, 有些完全独立。不同 env 影响的是不同子系统。

| # | 子系统 | Mooncake 扮演角色 | 受哪些 env/CLI 影响 | 共享 TransferEngine 单例? |
|---|---|---|---|---|
| 1 | **HiCache L3 store** | 远端 KV object store | `MOONCAKE_*` 全套 | 是 (满足 §4.5 的 4-AND 时复用) |
| 2 | **HiCache L2 allocator** | pinned + RDMA-registered 本地内存池 | `MOONCAKE_STANDALONE_STORAGE` | 否 (只是 allocator, 不走 TE) |
| 3 | **P/D disaggregation** | Prefill↔Decode KV RDMA | `--mooncake-ib-device`, `SGLANG_MOONCAKE_CUSTOM_MEM_POOL`, `SGLANG_MOONCAKE_SEND_AUX_TCP` | 是 (在 `parallel_state.py` 早期初始化 TE) |
| 4 | **MoE token dispatcher** | EP all-to-all token 路由 | `SGLANG_MOONCAKE_EP_NUM_MAX_DISPATCH_TOKENS_PER_RANK`, `--moe-a2a-backend mooncake` | **否** (用独立的 `mooncake.mooncake_ep_buffer.Buffer`) |
| 5 | **Elastic EP weight sync** | 弹性扩缩容时 expert 权重 RDMA 同步 | 共用 P/D 的 env | 是 |

**关键洞察**:
- `MOONCAKE_PROTOCOL` / `MOONCAKE_TE_META_DATA_SERVER` **只影响路径 1** (HiCache Store)。P/D 的 TransferEngine 在 [mooncake_transfer_engine.py:192-197](../../third_partys/sglang/python/sglang/srt/distributed/device_communicators/mooncake_transfer_engine.py#L192) **硬编码 `rdma + P2PHANDSHAKE`**, 不读这两个 env
- 路径 1、3、5 共享同一个进程级 `MooncakeTransferEngine` 单例 ([mooncake_transfer_engine.py:264-281](../../third_partys/sglang/python/sglang/srt/distributed/device_communicators/mooncake_transfer_engine.py#L264)); 这也是为什么路径 1 的配置 (`MOONCAKE_DEVICE` / `MOONCAKE_PROTOCOL` / `MOONCAKE_TE_META_DATA_SERVER`) 要和路径 3 的 `--mooncake-ib-device` 对齐, 否则不满足 §4.5 的复用条件, 进程里会有**两个** TransferEngine 实例
- MoE dispatcher (路径 4) 用的是 Mooncake 另一个 C++ 组件 (`mooncake_ep_buffer.Buffer`), 和 TE 毫无关系。改 `MOONCAKE_PROTOCOL` 不影响它

**核心术语**:

| 术语 | 含义 |
|---|---|
| `MooncakeTransferEngine` | Mooncake 的 data plane 单例 (`mooncake.engine.TransferEngine` 的 Python 包装), 进程内 global |
| `MooncakeDistributedStore` | Mooncake 的 control+data plane 客户端 (`mooncake.store.MooncakeDistributedStore`), 含 Master RPC + TransferEngine |
| `standalone_storage` | 开关, `True` = dummy client / zero-copy 模式, SGLang 不自己做 Mooncake client, 通过 RPC 连本地 `mooncake_client` |
| `P2PHANDSHAKE` | Mooncake 的一种 metadata 模式, 不依赖中心 metadata server, 节点间直接 RPC 握手 |

---

## §2 · SGLang 承认的所有 Mooncake env

中心注册点: [environ.py:297-315](../../third_partys/sglang/python/sglang/srt/environ.py#L297)

| 环境变量 | 类型 | 默认值 | 作用 |
|---|---|---|---|
| `MOONCAKE_MASTER` | str | None | master 地址; `host:port` 或 `etcd://...` |
| `MOONCAKE_CLIENT` | str | None | dummy client 模式下 real client 的地址 |
| `MOONCAKE_LOCAL_HOSTNAME` | str | `"localhost"` | 本节点对外暴露的 hostname (fallback 到 `LOCAL_HOSTNAME` env) |
| `MOONCAKE_TE_META_DATA_SERVER` | str | `"P2PHANDSHAKE"` | metadata 模式: `P2PHANDSHAKE` / `http://...` / `etcd://...` |
| `MOONCAKE_GLOBAL_SEGMENT_SIZE` | str | `"4gb"` | 本节点贡献给 pool 的内存, 支持 `"Xgb"` 或 bytes; `"0"` = 不贡献 |
| `MOONCAKE_PROTOCOL` | str | `"tcp"` ⚠️ | 传输协议; **默认是 tcp, 生产必须显式 rdma** |
| `MOONCAKE_DEVICE` | str | `""` | NIC 列表, 空值 = auto-discover |
| `MOONCAKE_MASTER_METRICS_PORT` | int | `9003` | master metrics endpoint port |
| `MOONCAKE_CHECK_SERVER` | bool | `False` | 启动时是否 probe master 健康 |
| `MOONCAKE_STANDALONE_STORAGE` | bool | `False` | 是否用 dummy client / zero-copy 模式 |
| `SGLANG_HICACHE_MOONCAKE_CONFIG_PATH` | str | None | JSON 配置文件路径 (路径 2 触发器) |
| `SGLANG_HICACHE_MOONCAKE_REUSE_TE` | bool | `True` | 允许 HiCache Store 复用 P/D 已建 TransferEngine |
| `SGLANG_MOONCAKE_CUSTOM_MEM_POOL` | str | None | P/D custom mem pool: `NVLINK` / `BAREX` / `INTRA_NODE_NVLINK` |
| `SGLANG_MOONCAKE_SEND_AUX_TCP` | bool | `False` | P/D aux data 走 TCP 而非 RDMA |

**非 SGLang 注册, 但 Mooncake C++ 直接读** (SGLang Python 完全不感知):

| env | 作用 |
|---|---|
| `MC_TE_METRIC` | 打开 transfer-engine per-op metric (生产推荐 `true`) |
| `MC_MS_AUTO_DISC=1` | **强制覆盖** `MOONCAKE_DEVICE`, 改走自动发现 |
| `SGLANG_MOONCAKE_TRANS_THREAD` | transfer engine 工作线程数 (LWS PD YAML 观测到, 未进 environ.py) |
| `LOCAL_HOSTNAME` | `MOONCAKE_LOCAL_HOSTNAME` 未设时的遗留 fallback |

---

## §3 · 三条互斥的配置注入路径

[mooncake_store.py:246-266](../../third_partys/sglang/python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py#L246):

```python
if extra_config.get("master_server_address") or extra_config.get("client_server_address"):
    config = MooncakeStoreConfig.load_from_extra_config(extra_config)   # 路径 1 (CLI)
elif envs.SGLANG_HICACHE_MOONCAKE_CONFIG_PATH.is_set():
    config = MooncakeStoreConfig.from_file()                            # 路径 2 (JSON 文件)
else:
    config = MooncakeStoreConfig.load_from_env()                        # 路径 3 (env, k8s YAML 走这条)
```

| 路径 | 触发方式 | 覆盖性 |
|---|---|---|
| 1 | CLI `--hicache-storage-backend-extra-config` 的 JSON 字符串里含 master/client 地址 | **完全覆盖** env (env 不会被读) |
| 2 | env `SGLANG_HICACHE_MOONCAKE_CONFIG_PATH` 指向 JSON 文件 | 同上 |
| 3 | 直接设 `MOONCAKE_MASTER` 或 `MOONCAKE_CLIENT` env | fallback |

三条路径最终都落到同一个 dataclass `MooncakeStoreConfig` ([line 84-94](../../third_partys/sglang/python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py#L84)), JSON key 和 env 名的对应关系见 §5 表格。

⚠️ **混用陷阱**: 如果 YAML 设了 env 同时 CLI 传了 extra-config, env 不生效(不是按字段合并,而是整份 config 被覆盖)。

---

## §4 · 关键字段的行为细节

### 4.1 `MOONCAKE_MASTER` 的三种格式

| 格式 | 场景 | 例子 |
|---|---|---|
| `host:port` | 单 master | `10.11.22.33:50051` |
| `etcd://host:2379[,host2:...]` | HA, 通过 etcd 选主 | 用户 YAML |
| `etcd://host:2379/cluster-id` | HA + 显式隔离多 master 组 | — |

SGLang Python **不校验格式**, 直接透传给 Mooncake C++ client。HA client 会连 etcd → 读当前 leader → RPC 打到 leader, leader 挂掉时通过 etcd watch 切换。

`MOONCAKE_MASTER` 和 `MOONCAKE_CLIENT` **至少一个必填** ([mooncake_store.py:156-159](../../third_partys/sglang/python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py#L156)):
- 设 `MASTER` → SGLang 进程自己当 Mooncake client (普通模式)
- 设 `CLIENT` → standalone_storage 模式, SGLang 通过 RPC 连本地 `mooncake_client` real client

### 4.2 `MOONCAKE_PROTOCOL` 默认陷阱

```python
MOONCAKE_PROTOCOL = EnvStr("tcp")   # environ.py:311
```

**默认是 tcp, 不是 rdma**。不显式设 `rdma` 会在 IB 集群上跑出 TCP 带宽。合法取值: `rdma` / `tcp` / `ascend` (华为 NPU, 由 `ENABLE_ASCEND_TRANSFER_WITH_MOONCAKE=1` 切换)。

注意: P/D disagg 的 TransferEngine 在 [mooncake_transfer_engine.py:192-197](../../third_partys/sglang/python/sglang/srt/distributed/device_communicators/mooncake_transfer_engine.py#L192) **硬编码 `"rdma" + "P2PHANDSHAKE"`**, 不读 `MOONCAKE_PROTOCOL`。这个 env 只影响 HiCache Store 路径 (§1 子系统 1)。

### 4.3 `MOONCAKE_DEVICE` 三种格式

由 [get_ib_devices_for_gpu:15-90](../../third_partys/sglang/python/sglang/srt/distributed/device_communicators/mooncake_transfer_engine.py#L15) 解析:

| 格式 | 例子 | 语义 |
|---|---|---|
| 逗号分隔 | `mlx5_10,mlx5_11,...` (用户 YAML) | 所有 rank 共用, Mooncake 内部 NUMA-aware 选 NIC |
| JSON dict | `{"0": "ib0,ib1", "1": "ib2,ib3"}` | 显式 GPU→NIC 映射 (rail-aligned 需要) |
| JSON 文件 | `/etc/mooncake/ib_map.json` (`.json` 后缀) | 同上 dict, 写在文件里 |

空值 → Mooncake 自动发现。`MC_MS_AUTO_DISC=1` 会**强制覆盖**此 env, 改走 auto-discover。

另有一个独立 CLI `--mooncake-ib-device` ([server_args.py:5520](../../third_partys/sglang/python/sglang/srt/server_args.py#L5520)) 作用于 P/D disagg + Elastic EP, 理论上应和 `MOONCAKE_DEVICE` 一致 —— 不一致时 HiCache Store 会**拒绝复用** P/D 的 TransferEngine (见 §4.5), 进程里出现两个引擎实例。

### 4.4 `MOONCAKE_GLOBAL_SEGMENT_SIZE='0'` 的含义

- `"0"` → 本进程不贡献内存, 纯消费远端 store (**HA 推荐**, store 独立 pod, SGLang 重启不丢 cache)
- `"4gb"` / bytes → 贡献指定内存 (all-in-one 模式, SGLang 重启丢 cache)

TP 分片: 值是**整个 TP group 合计**, 平均分给每个 rank ([mooncake_store.py:299](../../third_partys/sglang/python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py#L299))。

### 4.5 TransferEngine 复用的 4-AND 条件

[mooncake_store.py:358-363](../../third_partys/sglang/python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py#L358):

```python
if (shared_engine is not None
    and device_name == shared_engine.get_ib_device()
    and self.config.metadata_server == "P2PHANDSHAKE"   # ← 硬编码字符串匹配
    and self.config.protocol == "rdma"):
    transfer_engine = shared_engine.get_engine()        # 复用
```

生产调优: 把 `MOONCAKE_DEVICE` / `MOONCAKE_TE_META_DATA_SERVER=P2PHANDSHAKE` / `MOONCAKE_PROTOCOL=rdma` 三者和 P/D 对齐, 让 HiCache Store 复用 P/D 的引擎, 省一份 RDMA 注册。不满足任一条 → 进程里同时存在两个 TransferEngine。

### 4.6 Layout 兼容陷阱

Mooncake **不支持** `layer_first` 的 HiCache mem layout (按 layer 再按 token 组织内存, 每个 KV page 内存不连续)。

- `register_mem_pool_host()` 的 assert 允许列表: `"page_first"`, `"page_first_direct"`, `"page_head"`, `"page_first_kv_spilt"` ([mooncake_store.py:~496-501](../../third_partys/sglang/python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py))
- server_args 会**静默自动转换** `layer_first` → `page_first` (`server_args._resolve_storage_layout_compatibility` 附近 line ~3125)

**后果**: `--hicache-mem-layout layer_first --hicache-storage-backend mooncake` **不会报错**, 但实际 layout 并不是 `layer_first`。调优 layout 的人如果以为自己跑的是 layer_first, 行为会和预期不一致。推荐直接显式写 `--hicache-mem-layout page_first`。

### 4.7 `MC_TE_METRIC`

SGLang Python 从不 `getenv("MC_TE_METRIC")`, 由 Mooncake C++ 在 `TransferEngine::initialize` 时直接读, 开启后输出 per-op bandwidth/latency/error metric。开销低, 生产推荐 `true`。

---

## §5 · 字段对照表 (env ↔ JSON key)

| 字段 | env | JSON key | 默认 | HA 典型值 | 单 master 典型值 |
|---|---|---|---|---|---|
| master 地址 | `MOONCAKE_MASTER` | `master_server_address` | None | `etcd://.../2379` | `10.11.22.33:50051` |
| metadata | `MOONCAKE_TE_META_DATA_SERVER` | `metadata_server` | `"P2PHANDSHAKE"` | `P2PHANDSHAKE` | `P2PHANDSHAKE` |
| segment | `MOONCAKE_GLOBAL_SEGMENT_SIZE` | `global_segment_size` | `"4gb"` | `"0"` | `"4gb"` |
| 协议 | `MOONCAKE_PROTOCOL` | `protocol` | `"tcp"` ⚠️ | `"rdma"` | `"rdma"` |
| NIC | `MOONCAKE_DEVICE` | `device_name` | `""` | `mlx5_10,...` | `""` |
| hostname | `MOONCAKE_LOCAL_HOSTNAME` | `local_hostname` | `"localhost"` | (默认) | (默认) |
| standalone | `MOONCAKE_STANDALONE_STORAGE` | `standalone_storage` | `False` | `False` | 可选 `True` |

---

## §6 · 错配诊断手册

| 症状 | 根因 | 处理 |
|---|---|---|
| `ValueError: Either 'MOONCAKE_MASTER' or 'MOONCAKE_CLIENT' is not set` | 走到路径 3 但两个 env 都没设, 或 CLI JSON key 拼错 | 检查 YAML env / CLI JSON |
| 带宽远低于 IB 上限 (< 20 GB/s on 200Gb IB) | `MOONCAKE_PROTOCOL` 走了默认 `tcp` | 显式设 `rdma`, 检查 `env \| grep MOONCAKE_PROTOCOL` |
| `MooncakeStore with standalone_storage=True requires MooncakeHostTensorAllocator` | Mooncake Python 版本 < 0.3.8.post1 被静默降级 | `pip install mooncake --upgrade` |
| `metadata not found` | 用了 http metadata server 模式 | 改 `MOONCAKE_TE_META_DATA_SERVER=P2PHANDSHAKE` |
| 日志出现两次 `Mooncake Transfer Engine initialized` | 不满足 §4.5 的 4-AND, HiCache Store 没复用 P/D 的引擎 | 对齐 device + P2PHANDSHAKE + rdma |
| HA 模式 client 拿不到 leader | Mooncake master 没启 etcd backend 或 etcd 里没 leader key | 查 master 启动 flag, 实测 `etcdctl get /mooncake/...` |
| `--hicache-mem-layout layer_first` 设了但没生效 | Mooncake 路径下静默转换成 `page_first` (§4.6) | 直接显式写 `page_first` |

---

## §7 · 推荐 k8s HA 部署模板

```yaml
env:
- name: MOONCAKE_TE_META_DATA_SERVER
  value: P2PHANDSHAKE                              # 必须, 才能复用 P/D TE
- name: MOONCAKE_MASTER
  value: etcd://etcd-client.mooncake-ha.svc.cluster.local:2379
- name: MOONCAKE_PROTOCOL
  value: rdma                                      # ⚠️ 默认是 tcp, 必须显式
- name: MOONCAKE_DEVICE
  value: mlx5_10,mlx5_11,mlx5_12,mlx5_13,mlx5_14,mlx5_15,mlx5_16,mlx5_17
- name: MOONCAKE_GLOBAL_SEGMENT_SIZE
  value: '0'                                        # store 独立 pod, SGLang 不贡献
- name: MC_TE_METRIC
  value: 'true'                                     # Mooncake C++ 直通, 开 per-op metric
```

CLI 侧要和 env 的 device 对齐, 才能复用 engine:

```bash
python -m sglang.launch_server \
    --enable-hierarchical-cache \
    --hicache-storage-backend mooncake \
    --hicache-mem-layout page_first \
    --mooncake-ib-device mlx5_10,mlx5_11,mlx5_12,mlx5_13,mlx5_14,mlx5_15,mlx5_16,mlx5_17 \
    --model-path ${MODEL_PATH} --tp-size 8
```

单 master 只需改两行:
```yaml
- name: MOONCAKE_MASTER
  value: 10.11.22.33:50051
- name: MOONCAKE_GLOBAL_SEGMENT_SIZE
  value: '4gb'                                      # SGLang 兼做 store
```

---

## §8 · 验证命令

```bash
# 在 pod 内看实际读到的 env
env | grep -E "^(MOONCAKE_|SGLANG_MOONCAKE_|MC_)" | sort

# 看走的是哪条配置路径
grep -E "Mooncake Configuration loaded (from env|from file|from extra_config)" sglang.log

# 看 TransferEngine 是否复用 (无此行 = 没复用, 对照 §4.5 排查)
grep "Reuse initialized mooncake transfer engine" sglang.log

# 看实际生效的 NIC
grep -iE "mooncake.*device|ib_device" sglang.log | head
```

---

## §9 · PR review 路由表 (SGLang Mooncake)

收到一个带 Mooncake 改动的 SGLang PR, 按文件路径定位它属于 §1 的哪个子系统:

| 文件路径 | 子系统 | Review 侧重 |
|---|---|---|
| `sgl-kernel/` | L1↔L2 kernel (常和子系统 2 相关) | CUDA kernel 正确性, memcpy batch API 兼容性 |
| `python/sglang/srt/mem_cache/storage/mooncake_store/` | §1 HiCache L3 store | config 解析、batch_put/get 语义、zero-copy 正确性 |
| `python/sglang/srt/mem_cache/memory_pool_host.py` | §2 L2 allocator / L1↔L2 kernel | allocator 能力声明、双重注册兼容性 |
| `python/sglang/srt/disaggregation/mooncake/` | §3 P/D disagg | bootstrap 协议、TP 错配、心跳/超时 |
| `python/sglang/srt/layers/moe/token_dispatcher/mooncake.py` | §4 MoE dispatcher | `mooncake_ep_buffer.Buffer` 的 dispatch/combine, **独立 buffer** |
| `python/sglang/srt/elastic_ep/` | §5 Elastic EP weight sync | `batch_transfer_sync_read` 语义、ZMQ 广播 |
| `python/sglang/srt/distributed/device_communicators/mooncake_transfer_engine.py` | **跨子系统** (§1/§3/§5 共用) | ⚠️ **需要跨路径 regression review**, 任何改动都可能同时影响 3 个子系统 |

最后一行最危险: TransferEngine 是 §1/§3/§5 共享的单例, 它的行为变更会同时影响多个路径, 必须多子系统回归测试。

---

## §10 · 与 vLLM Mooncake 集成的对比

（写给同时 review vLLM 和 SGLang 的 reviewer）

| 维度 | SGLang | vLLM |
|---|---|---|
| **KV cache L3** | HiCache `MooncakeStore` (内置抽象) | `MooncakeStoreConnector` (kv_connector v1 插件) |
| **KV cache L2 allocator** | `MooncakeHostTensorAllocator` 贴合 store 做 zero-copy | 无对应概念, L2 就是普通 pinned |
| **P/D disagg** | `disaggregation/mooncake/conn.py`, 独立子系统 | `MooncakeConnector` (kv_connector v1), direct P2P |
| **MoE dispatcher** | `MooncakeEPDispatcher` | 无 (vLLM 目前不用 Mooncake 做 MoE) |
| **Elastic EP** | `elastic_ep/expert_backup_*` | 无对应 (vLLM 暂无 elastic EP) |
| **Embedding store** | `MooncakeEmbeddingStore` | 无对应 |
| **TransferEngine 共享** | 进程内 global singleton, 多子系统复用 | 按 connector 实例粒度, 复用有限 |
| **Zero-copy 优化** | 有 (standalone_storage=True) | 无直接对应, v1 connector 一律走 staging |

**核心差异**: SGLang 的 Mooncake 集成**更深、更系统化** —— 不仅作为 L3 transport, 还把 Mooncake 的 memory allocator 和 MoE 专用 buffer 都吃到了核心路径里。vLLM 则通过 kv_connector v1 插件抽象保持了 Mooncake 的**可选性和可替换性**。

**Review 时的后果**:
- SGLang 里一个 Mooncake 的"局部改动"可能牵连多子系统 (§9 最后一行)
- vLLM 里 Mooncake connector 的改动基本只影响 connector 内部, 不会扩散到 HiCache / MoE

---

## §11 · 关键代码锚点

| 主题 | 文件 | 行 |
|---|---|---|
| env 注册中心 | `srt/environ.py` | 297-315 |
| `MooncakeStoreConfig` + 三条路径 | `srt/mem_cache/storage/mooncake_store/mooncake_store.py` | 84, 97, 148, 187, 246 |
| MASTER/CLIENT 必填校验 | 同上 | 156-159 |
| standalone_storage 分支 | 同上 | 331-342 |
| TransferEngine 复用 4-AND | 同上 | 358-363 |
| layout 允许列表 assert | 同上 | ~496-501 |
| TE 硬编码 `rdma + P2PHANDSHAKE` | `srt/distributed/device_communicators/mooncake_transfer_engine.py` | 192-197 |
| `get_ib_devices_for_gpu` 三格式 | 同上 | 15-90 |
| TE 单例 init | 同上 | 264-281 |
| `MooncakeEPDispatcher` (§1 子系统 4) | `srt/layers/moe/token_dispatcher/mooncake.py` | 286 |
| `ExpertBackupManager` (§1 子系统 5) | `srt/elastic_ep/expert_backup_manager.py` | 34 |
| `ExpertBackupClient` (§1 子系统 5) | `srt/elastic_ep/expert_backup_client.py` | 31 |
| `_resolve_storage_layout_compatibility` | `srt/server_args.py` | ~3125 |
| CLI `--hicache-storage-backend-extra-config` | `srt/server_args.py` | 5669-5672 |
| CLI `--mooncake-ib-device` | 同上 | 5520-5527 |
| 官方 README (上游权威) | `srt/mem_cache/storage/mooncake_store/README.md` | 全文 |
| LWS PD YAML (MC_TE_METRIC 实例) | `docs/references/multi_node_deployment/lws_pd/lws_pd_deploy.md` | 113, 258, 274 |
