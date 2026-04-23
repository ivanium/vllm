---
title: SGLang Default Mooncake Config & Environment Variable Deep Dive
audience: Deployment/SRE engineers + vLLM/SGLang Mooncake reviewers
last_verified: 2026-04-23
repo_path: vllm/third_partys/sglang
scope: |
  Which env vars / CLI flags / JSON keys a SGLang Mooncake worker reads at
  startup, each field's default, the precedence order, HA vs single-master
  differences — plus a 5-subsystem view of Mooncake's role in SGLang and a
  cross-project comparison against vLLM's Mooncake integration.
  All code refs grep-verified in the local worktree.
related_docs:
  - k8s_mooncake_ha_deployment_notes.md              (HA deployment top-level notes)
  - mooncake_store_cpu_offload_full_stack.md         (vLLM CPU offload full stack)
  - vllm_mooncake_store_connector_system_design_v1.md (vLLM connector design)
---

# SGLang Default Mooncake Config & Environment Variable Deep Dive

## §0 · Starting point: a production k8s YAML

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

HA (multi-master) deployment → `MOONCAKE_MASTER` points at the **etcd cluster address**; masters elect a leader through etcd. A single-master deployment just uses `10.11.xx.xx:50051` (default port 50051).

---

## §1 · Mooncake's five roles in SGLang

To read env config correctly, first understand this: **Mooncake appears in at least 5 unrelated subsystems** in SGLang — some share the same process-level `MooncakeTransferEngine` singleton, some are fully independent. Different envs affect different subsystems.

| # | Subsystem | Mooncake's role | Affected by which env/CLI | Shares TransferEngine singleton? |
|---|---|---|---|---|
| 1 | **HiCache L3 store** | Remote KV object store | full `MOONCAKE_*` set | Yes (reuse when §4.5's 4-AND is satisfied) |
| 2 | **HiCache L2 allocator** | pinned + RDMA-registered local mem pool | `MOONCAKE_STANDALONE_STORAGE` | No (allocator only, not a TE consumer) |
| 3 | **P/D disaggregation** | Prefill↔Decode KV over RDMA | `--mooncake-ib-device`, `SGLANG_MOONCAKE_CUSTOM_MEM_POOL`, `SGLANG_MOONCAKE_SEND_AUX_TCP` | Yes (initializes TE early in `parallel_state.py`) |
| 4 | **MoE token dispatcher** | EP all-to-all token routing | `SGLANG_MOONCAKE_EP_NUM_MAX_DISPATCH_TOKENS_PER_RANK`, `--moe-a2a-backend mooncake` | **No** (uses independent `mooncake.mooncake_ep_buffer.Buffer`) |
| 5 | **Elastic EP weight sync** | Expert weight RDMA sync during elastic scaling | shares P/D's env | Yes |

**Key insights**:
- `MOONCAKE_PROTOCOL` / `MOONCAKE_TE_META_DATA_SERVER` **only affect subsystem 1** (HiCache Store). P/D's TransferEngine at [mooncake_transfer_engine.py:192-197](../../third_partys/sglang/python/sglang/srt/distributed/device_communicators/mooncake_transfer_engine.py#L192) **hardcodes `rdma + P2PHANDSHAKE`** and doesn't read these envs.
- Subsystems 1, 3, 5 share one process-level `MooncakeTransferEngine` singleton ([mooncake_transfer_engine.py:264-281](../../third_partys/sglang/python/sglang/srt/distributed/device_communicators/mooncake_transfer_engine.py#L264)). This is why subsystem 1's config (`MOONCAKE_DEVICE` / `MOONCAKE_PROTOCOL` / `MOONCAKE_TE_META_DATA_SERVER`) must align with subsystem 3's `--mooncake-ib-device` — otherwise §4.5's reuse condition fails and the process ends up with **two** TransferEngine instances.
- The MoE dispatcher (subsystem 4) uses a different Mooncake C++ component (`mooncake_ep_buffer.Buffer`), unrelated to TE. Changing `MOONCAKE_PROTOCOL` does not affect it.

**Core terminology**:

| Term | Meaning |
|---|---|
| `MooncakeTransferEngine` | Mooncake's data-plane singleton (Python wrapper around `mooncake.engine.TransferEngine`), process-global |
| `MooncakeDistributedStore` | Mooncake's control+data-plane client (`mooncake.store.MooncakeDistributedStore`), contains Master RPC + TransferEngine |
| `standalone_storage` | A toggle; `True` = dummy-client / zero-copy mode, where SGLang does NOT act as the Mooncake client, but connects via RPC to a local `mooncake_client` |
| `P2PHANDSHAKE` | A Mooncake metadata mode that avoids a central metadata server — nodes exchange metadata via direct RPC handshake |

---

## §2 · Every Mooncake env var SGLang recognizes

Central registration: [environ.py:297-315](../../third_partys/sglang/python/sglang/srt/environ.py#L297)

| Env var | Type | Default | Purpose |
|---|---|---|---|
| `MOONCAKE_MASTER` | str | None | Master address; `host:port` or `etcd://...` |
| `MOONCAKE_CLIENT` | str | None | Real-client address in dummy-client mode |
| `MOONCAKE_LOCAL_HOSTNAME` | str | `"localhost"` | Externally-visible hostname (falls back to `LOCAL_HOSTNAME` env) |
| `MOONCAKE_TE_META_DATA_SERVER` | str | `"P2PHANDSHAKE"` | Metadata mode: `P2PHANDSHAKE` / `http://...` / `etcd://...` |
| `MOONCAKE_GLOBAL_SEGMENT_SIZE` | str | `"4gb"` | Memory this node contributes to the pool; accepts `"Xgb"` or bytes; `"0"` = no contribution |
| `MOONCAKE_PROTOCOL` | str | `"tcp"` ⚠️ | Transport; **default is tcp — production MUST set rdma explicitly** |
| `MOONCAKE_DEVICE` | str | `""` | NIC list; empty = auto-discover |
| `MOONCAKE_MASTER_METRICS_PORT` | int | `9003` | Master metrics endpoint port |
| `MOONCAKE_CHECK_SERVER` | bool | `False` | Probe master health at startup |
| `MOONCAKE_STANDALONE_STORAGE` | bool | `False` | Use dummy-client / zero-copy mode |
| `SGLANG_HICACHE_MOONCAKE_CONFIG_PATH` | str | None | JSON config file path (path 2 trigger) |
| `SGLANG_HICACHE_MOONCAKE_REUSE_TE` | bool | `True` | Allow HiCache Store to reuse P/D's TransferEngine |
| `SGLANG_MOONCAKE_CUSTOM_MEM_POOL` | str | None | P/D custom mem pool: `NVLINK` / `BAREX` / `INTRA_NODE_NVLINK` |
| `SGLANG_MOONCAKE_SEND_AUX_TCP` | bool | `False` | Send P/D aux data over TCP instead of RDMA |

**Not registered by SGLang, read directly by Mooncake C++** (SGLang Python is completely unaware):

| env | Purpose |
|---|---|
| `MC_TE_METRIC` | Enable transfer-engine per-op metrics (production: set `true`) |
| `MC_MS_AUTO_DISC=1` | **Force-overrides** `MOONCAKE_DEVICE` and switches to auto-discovery |
| `SGLANG_MOONCAKE_TRANS_THREAD` | Transfer engine worker thread count (seen in LWS PD YAML, not registered in environ.py) |
| `LOCAL_HOSTNAME` | Legacy fallback when `MOONCAKE_LOCAL_HOSTNAME` is unset |

---

## §3 · Three mutually-exclusive config-injection paths

[mooncake_store.py:246-266](../../third_partys/sglang/python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py#L246):

```python
if extra_config.get("master_server_address") or extra_config.get("client_server_address"):
    config = MooncakeStoreConfig.load_from_extra_config(extra_config)   # Path 1 (CLI)
elif envs.SGLANG_HICACHE_MOONCAKE_CONFIG_PATH.is_set():
    config = MooncakeStoreConfig.from_file()                            # Path 2 (JSON file)
else:
    config = MooncakeStoreConfig.load_from_env()                        # Path 3 (env; k8s YAML uses this)
```

| Path | Trigger | Override behavior |
|---|---|---|
| 1 | CLI `--hicache-storage-backend-extra-config` JSON contains master/client address | **Fully overrides** env (env is never read) |
| 2 | env `SGLANG_HICACHE_MOONCAKE_CONFIG_PATH` points to a JSON file | Same |
| 3 | Setting `MOONCAKE_MASTER` or `MOONCAKE_CLIENT` env directly | Fallback |

All three paths land in the same dataclass `MooncakeStoreConfig` ([line 84-94](../../third_partys/sglang/python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py#L84)); JSON-key ↔ env-name mapping is in §5.

⚠️ **Mixing trap**: if YAML sets envs AND CLI also passes extra-config, the env values are silently ignored — the whole config block is replaced, not merged field-by-field.

---

## §4 · Field behavior details

### 4.1 Three formats for `MOONCAKE_MASTER`

| Format | Scenario | Example |
|---|---|---|
| `host:port` | Single master | `10.11.22.33:50051` |
| `etcd://host:2379[,host2:...]` | HA, leader election via etcd | User's YAML |
| `etcd://host:2379/cluster-id` | HA with explicit cluster-id (multi-tenant on one etcd) | — |

SGLang Python **does not validate the format** — it's passed straight to the Mooncake C++ client. In HA mode the client connects to etcd → reads current leader → RPCs to the leader; on leader failure, etcd watches trigger a re-read.

`MOONCAKE_MASTER` and `MOONCAKE_CLIENT` — **at least one is required** ([mooncake_store.py:156-159](../../third_partys/sglang/python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py#L156)):
- Set `MASTER` → SGLang process acts as the Mooncake client itself (normal mode)
- Set `CLIENT` → standalone_storage mode; SGLang connects via RPC to a local `mooncake_client` real client

### 4.2 The `MOONCAKE_PROTOCOL` default trap

```python
MOONCAKE_PROTOCOL = EnvStr("tcp")   # environ.py:311
```

**Default is tcp, not rdma.** Forgetting to set `rdma` silently gives you TCP bandwidth on an IB cluster. Valid values: `rdma` / `tcp` / `ascend` (Huawei NPU, gated by `ENABLE_ASCEND_TRANSFER_WITH_MOONCAKE=1`).

Note: the P/D disagg TransferEngine at [mooncake_transfer_engine.py:192-197](../../third_partys/sglang/python/sglang/srt/distributed/device_communicators/mooncake_transfer_engine.py#L192) **hardcodes `"rdma" + "P2PHANDSHAKE"`** and does NOT read `MOONCAKE_PROTOCOL`. This env only affects the HiCache Store path (subsystem 1 in §1).

### 4.3 Three formats for `MOONCAKE_DEVICE`

Parsed by [get_ib_devices_for_gpu:15-90](../../third_partys/sglang/python/sglang/srt/distributed/device_communicators/mooncake_transfer_engine.py#L15):

| Format | Example | Semantics |
|---|---|---|
| Comma-separated | `mlx5_10,mlx5_11,...` (user's YAML) | Shared across ranks; Mooncake picks NUMA-locally |
| JSON dict | `{"0": "ib0,ib1", "1": "ib2,ib3"}` | Explicit GPU→NIC mapping (needed for rail-aligned topologies) |
| JSON file | `/etc/mooncake/ib_map.json` (must end in `.json`) | Same dict in a file |

Empty → Mooncake auto-discovers. `MC_MS_AUTO_DISC=1` **force-overrides** this env and also switches to auto-discovery.

Separately, `--mooncake-ib-device` ([server_args.py:5520](../../third_partys/sglang/python/sglang/srt/server_args.py#L5520)) applies to P/D disagg + Elastic EP and should match `MOONCAKE_DEVICE`. If they don't match, HiCache Store **refuses to reuse** P/D's TransferEngine (see §4.5) and the process ends up with two engine instances.

### 4.4 What `MOONCAKE_GLOBAL_SEGMENT_SIZE='0'` means

- `"0"` → this process contributes nothing; pure consumer of remote store (**HA recommended**: store lives in a separate pod, SGLang restarts don't drop cache)
- `"4gb"` / bytes → contribute the given amount (all-in-one mode; SGLang restarts drop cache)

TP sharding: the value is the **TP-group total**, split evenly per rank ([mooncake_store.py:299](../../third_partys/sglang/python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py#L299)).

### 4.5 The 4-AND condition for TransferEngine reuse

[mooncake_store.py:358-363](../../third_partys/sglang/python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py#L358):

```python
if (shared_engine is not None
    and device_name == shared_engine.get_ib_device()
    and self.config.metadata_server == "P2PHANDSHAKE"   # ← exact string match
    and self.config.protocol == "rdma"):
    transfer_engine = shared_engine.get_engine()        # reuse
```

Production tuning: align `MOONCAKE_DEVICE` / `MOONCAKE_TE_META_DATA_SERVER=P2PHANDSHAKE` / `MOONCAKE_PROTOCOL=rdma` with P/D so HiCache Store reuses P/D's engine and saves one round of RDMA memory registration. Missing any of these → two coexisting TransferEngines in the process.

### 4.6 Layout compatibility trap

Mooncake **does not support** the `layer_first` HiCache mem layout (memory organized by layer then token means each KV page is non-contiguous).

- `register_mem_pool_host()` asserts the allowed list: `"page_first"`, `"page_first_direct"`, `"page_head"`, `"page_first_kv_spilt"` ([mooncake_store.py:~496-501](../../third_partys/sglang/python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py))
- server_args **silently auto-converts** `layer_first` → `page_first` (`server_args._resolve_storage_layout_compatibility`, around line ~3125)

**Consequence**: `--hicache-mem-layout layer_first --hicache-storage-backend mooncake` **does NOT error**, but the effective layout isn't `layer_first`. If someone tuning layouts assumes they're running `layer_first`, behavior won't match expectation. Always spell out `--hicache-mem-layout page_first` explicitly.

### 4.7 `MC_TE_METRIC`

SGLang Python never calls `getenv("MC_TE_METRIC")`. Mooncake C++ reads it directly in `TransferEngine::initialize`; when on, per-op bandwidth/latency/error metrics are emitted. Low overhead, recommended `true` in production.

---

## §5 · Field mapping table (env ↔ JSON key)

| Field | env | JSON key | Default | HA typical | Single-master typical |
|---|---|---|---|---|---|
| Master addr | `MOONCAKE_MASTER` | `master_server_address` | None | `etcd://.../2379` | `10.11.22.33:50051` |
| Metadata | `MOONCAKE_TE_META_DATA_SERVER` | `metadata_server` | `"P2PHANDSHAKE"` | `P2PHANDSHAKE` | `P2PHANDSHAKE` |
| Segment size | `MOONCAKE_GLOBAL_SEGMENT_SIZE` | `global_segment_size` | `"4gb"` | `"0"` | `"4gb"` |
| Protocol | `MOONCAKE_PROTOCOL` | `protocol` | `"tcp"` ⚠️ | `"rdma"` | `"rdma"` |
| NIC | `MOONCAKE_DEVICE` | `device_name` | `""` | `mlx5_10,...` | `""` |
| Hostname | `MOONCAKE_LOCAL_HOSTNAME` | `local_hostname` | `"localhost"` | (default) | (default) |
| Standalone | `MOONCAKE_STANDALONE_STORAGE` | `standalone_storage` | `False` | `False` | optionally `True` |

---

## §6 · Misconfiguration playbook

| Symptom | Root cause | Fix |
|---|---|---|
| `ValueError: Either 'MOONCAKE_MASTER' or 'MOONCAKE_CLIENT' is not set` | Fell through to path 3 but neither env is set, or CLI JSON key is misspelled | Check YAML env / CLI JSON keys |
| Bandwidth far below IB limit (< 20 GB/s on 200Gb IB) | `MOONCAKE_PROTOCOL` hit the `tcp` default | Set `rdma` explicitly; verify with `env \| grep MOONCAKE_PROTOCOL` |
| `MooncakeStore with standalone_storage=True requires MooncakeHostTensorAllocator` | Mooncake Python version < 0.3.8.post1, silently downgraded | `pip install mooncake --upgrade` |
| `metadata not found` | Using http metadata-server mode | Switch to `MOONCAKE_TE_META_DATA_SERVER=P2PHANDSHAKE` |
| Two `Mooncake Transfer Engine initialized` lines in logs | §4.5's 4-AND not satisfied — HiCache Store did not reuse P/D's engine | Align device + P2PHANDSHAKE + rdma |
| HA mode: client can't find a leader | Mooncake master not started with etcd backend, or no leader key in etcd | Check master startup flags, try `etcdctl get /mooncake/...` |
| `--hicache-mem-layout layer_first` set but not effective | Silently converted to `page_first` under Mooncake (§4.6) | Spell out `page_first` explicitly |

---

## §7 · Recommended k8s HA deployment template

```yaml
env:
- name: MOONCAKE_TE_META_DATA_SERVER
  value: P2PHANDSHAKE                              # required to enable P/D TE reuse
- name: MOONCAKE_MASTER
  value: etcd://etcd-client.mooncake-ha.svc.cluster.local:2379
- name: MOONCAKE_PROTOCOL
  value: rdma                                      # ⚠️ default is tcp, must set explicitly
- name: MOONCAKE_DEVICE
  value: mlx5_10,mlx5_11,mlx5_12,mlx5_13,mlx5_14,mlx5_15,mlx5_16,mlx5_17
- name: MOONCAKE_GLOBAL_SEGMENT_SIZE
  value: '0'                                        # store lives in a separate pod; SGLang contributes nothing
- name: MC_TE_METRIC
  value: 'true'                                     # Mooncake C++ passthrough; enables per-op metrics
```

On the CLI side, keep `--mooncake-ib-device` identical to `MOONCAKE_DEVICE` so the engine can be reused:

```bash
python -m sglang.launch_server \
    --enable-hierarchical-cache \
    --hicache-storage-backend mooncake \
    --hicache-mem-layout page_first \
    --mooncake-ib-device mlx5_10,mlx5_11,mlx5_12,mlx5_13,mlx5_14,mlx5_15,mlx5_16,mlx5_17 \
    --model-path ${MODEL_PATH} --tp-size 8
```

Single-master deployment: change only two lines:
```yaml
- name: MOONCAKE_MASTER
  value: 10.11.22.33:50051
- name: MOONCAKE_GLOBAL_SEGMENT_SIZE
  value: '4gb'                                      # SGLang also acts as store
```

---

## §8 · Verification commands

```bash
# Inside the pod, see what env actually reached SGLang
env | grep -E "^(MOONCAKE_|SGLANG_MOONCAKE_|MC_)" | sort

# Which config path was taken
grep -E "Mooncake Configuration loaded (from env|from file|from extra_config)" sglang.log

# Was the TransferEngine reused? (absence of this line = not reused; see §4.5)
grep "Reuse initialized mooncake transfer engine" sglang.log

# Which NIC is actually in use
grep -iE "mooncake.*device|ib_device" sglang.log | head
```

---

## §9 · PR review routing table (SGLang Mooncake)

When reviewing a SGLang PR touching Mooncake, locate the affected subsystem by file path:

| File path | Subsystem | Review focus |
|---|---|---|
| `sgl-kernel/` | L1↔L2 kernel (often tied to subsystem 2) | CUDA kernel correctness, memcpy batch API compatibility |
| `python/sglang/srt/mem_cache/storage/mooncake_store/` | §1 HiCache L3 store | config parsing, batch_put/get semantics, zero-copy correctness |
| `python/sglang/srt/mem_cache/memory_pool_host.py` | §2 L2 allocator / L1↔L2 kernel | allocator capability declaration, double-registration compatibility |
| `python/sglang/srt/disaggregation/mooncake/` | §3 P/D disagg | bootstrap protocol, TP mismatch, heartbeat/timeout |
| `python/sglang/srt/layers/moe/token_dispatcher/mooncake.py` | §4 MoE dispatcher | `mooncake_ep_buffer.Buffer` dispatch/combine, **independent buffer** |
| `python/sglang/srt/elastic_ep/` | §5 Elastic EP weight sync | `batch_transfer_sync_read` semantics, ZMQ broadcast |
| `python/sglang/srt/distributed/device_communicators/mooncake_transfer_engine.py` | **cross-subsystem** (shared by §1/§3/§5) | ⚠️ **needs multi-path regression review**; any change can simultaneously affect 3 subsystems |

The last row is the most dangerous: TransferEngine is the shared singleton across §1/§3/§5 — behavior changes require multi-subsystem regression tests.

---

## §10 · Comparison with vLLM's Mooncake integration

(for reviewers working on both vLLM and SGLang)

| Dimension | SGLang | vLLM |
|---|---|---|
| **KV cache L3** | HiCache `MooncakeStore` (built-in abstraction) | `MooncakeStoreConnector` (kv_connector v1 plugin) |
| **KV L2 allocator** | `MooncakeHostTensorAllocator` bonded with store for zero-copy | no equivalent concept; L2 is plain pinned |
| **P/D disagg** | `disaggregation/mooncake/conn.py`, standalone subsystem | `MooncakeConnector` (kv_connector v1), direct P2P |
| **MoE dispatcher** | `MooncakeEPDispatcher` | n/a (vLLM doesn't use Mooncake for MoE) |
| **Elastic EP** | `elastic_ep/expert_backup_*` | n/a (no elastic EP in vLLM yet) |
| **Embedding store** | `MooncakeEmbeddingStore` | n/a |
| **TransferEngine sharing** | process-level global singleton, multi-subsystem reuse | per-connector-instance granularity, limited reuse |
| **Zero-copy optimization** | yes (standalone_storage=True) | no direct equivalent; v1 connector always stages |

**Core difference**: SGLang's Mooncake integration is **deeper and more systemic** — it uses Mooncake not just as an L3 transport, but also bakes Mooncake's memory allocator and MoE-specific buffer into the core path. vLLM keeps Mooncake **optional and swappable** through the kv_connector v1 plugin abstraction.

**Reviewer implication**:
- In SGLang, a "local" Mooncake change can propagate across subsystems (see last row of §9)
- In vLLM, Mooncake connector changes are mostly contained within the connector — minimal cross-component impact

---

## §11 · Key source anchors

| Topic | File | Line |
|---|---|---|
| Env registration center | `srt/environ.py` | 297-315 |
| `MooncakeStoreConfig` + three paths | `srt/mem_cache/storage/mooncake_store/mooncake_store.py` | 84, 97, 148, 187, 246 |
| MASTER/CLIENT required check | same | 156-159 |
| standalone_storage branch | same | 331-342 |
| TransferEngine reuse 4-AND | same | 358-363 |
| layout allowed-list assert | same | ~496-501 |
| TE hardcoded `rdma + P2PHANDSHAKE` | `srt/distributed/device_communicators/mooncake_transfer_engine.py` | 192-197 |
| `get_ib_devices_for_gpu` three formats | same | 15-90 |
| TE singleton init | same | 264-281 |
| `MooncakeEPDispatcher` (§1 subsystem 4) | `srt/layers/moe/token_dispatcher/mooncake.py` | 286 |
| `ExpertBackupManager` (§1 subsystem 5) | `srt/elastic_ep/expert_backup_manager.py` | 34 |
| `ExpertBackupClient` (§1 subsystem 5) | `srt/elastic_ep/expert_backup_client.py` | 31 |
| `_resolve_storage_layout_compatibility` | `srt/server_args.py` | ~3125 |
| CLI `--hicache-storage-backend-extra-config` | `srt/server_args.py` | 5669-5672 |
| CLI `--mooncake-ib-device` | same | 5520-5527 |
| Upstream README (authoritative) | `srt/mem_cache/storage/mooncake_store/README.md` | whole file |
| LWS PD YAML (MC_TE_METRIC in the wild) | `docs/references/multi_node_deployment/lws_pd/lws_pd_deploy.md` | 113, 258, 274 |
