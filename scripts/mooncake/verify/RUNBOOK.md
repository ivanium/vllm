# MooncakeStore Disk-Offload — Launch Runbook

How to launch vLLM with the Mooncake distributed-KV store backing it,
including the per-node disk-tier offload path. The supported launch flow
is the **zero-scripts** vigil YAML pipeline (`mndp_noscripts.yaml`), which
calls `mooncake_master` and `mooncake_client` directly with no `.sh` shims.

Validated on GB200 1-node × 4 GPU + Qwen3-8B.

## TL;DR

**Validated path (today):**
```bash
cd ~/repos/vllm-mooncake
~/repos/model-ci/.venv/bin/vigil -c mndp.yaml          # uses scripts/mooncake/*.sh wrappers
```

**Zero-scripts path (work in progress — see "Known gap" below):**
```bash
~/repos/model-ci/.venv/bin/vigil -c mndp_noscripts.yaml
```

Three processes start in this order:

1. `mooncake_master` (object-store metadata + HTTP-metadata peer-discovery + eviction)
2. `mooncake_client` (per-node owner — CPU pool + SSD tier)
3. `vllm serve` with `--kv-transfer-config '{"kv_connector":"MooncakeStoreConnector",...}'`

After the bench finishes, vigil's `post_serve` SIGTERMs the owner + master via PID files written during startup.

## Dual-mode topology — `real-client` vs `owner-client`

The vllm-side `MooncakeStoreConfig` now has an explicit `mode` field that
selects the deployment topology. **`mooncake_config.json` ships with
`mode: "owner-client"`** to match the way our pipelines (both verify and
legacy) spawn a separate `mooncake_client` owner per node.

```jsonc
// scripts/mooncake/mooncake_config.json
{
  "mode": "owner-client",          // vllm rank = pure requester
  "global_segment_size": 0,        // must be 0 in owner-client mode
  "local_buffer_size":   "4GB",    // RDMA scratch buffer on this rank
  "metadata_server":     "http://127.0.0.1:8080/metadata",
  "master_server_address":"127.0.0.1:50051",
  "protocol":            "rdma",
  "device_name":         ""
}
```

The connector validates at startup and raises if `mode` and
`global_segment_size` disagree:

| `mode` | Required `global_segment_size` | Failure if violated |
|---|---|---|
| `real-client` (default — PR-40900 baseline) | `> 0` | `ValueError: real-client mode requires global_segment_size > 0` |
| `owner-client` | `== 0` | `ValueError: owner-client mode requires global_segment_size == 0` |

Confirm which mode is live by grepping the worker log:
```
INFO [worker.py:875] Mooncake mode=owner-client (global_segment_size=0,
                     local_buffer_size=4294967296, preferred_segment=127.0.0.1:50053,
                     enable_offload=True)
```

If you want to run the PR-40900 baseline topology instead (every rank
contributes a CPU segment, no separate owner), set `mode: "real-client"`
in the JSON and `global_segment_size: "4GB"` (or similar). The pipelines
in `scripts/mooncake/verify/` will fail validation in that mode because
they spawn a separate owner — use the legacy CPU-only PR-40900 config
shape instead, or skip the per-node owner step.

See `~/repos/disk/design/dual-mode-design.md` and
`~/repos/disk/design/toggle-runbook.md` for the full design + toggle
procedure.

## ⚠️ Known gap (2026-05-13)

The zero-scripts `mndp_noscripts.yaml` **does not yet do per-rank RNIC
assignment**. With `-dp 4` on 4 GPUs the no-scripts yaml lets the Mooncake
transfer engine auto-discover RDMA devices — all 4 vllm ranks then contend
on the same NICs and the bench stalls under sustained offload pressure
(19 reqs queued, 0 running, 0 progress for >5 min, GPU OOM never
recovers).

The wrapper script (`run_vllm_with_mooncake_owner.sh` →
`rdma_config_utils.sh::get_worker_rdma_devices_csv`) queries each GPU's
PCI-affine RNIC via `ibdev2netdev` + `rdma link show`, then exports
`MOONCAKE_DEVICE=<rank0_nic>,<rank1_nic>,...` so the vllm-side
`rdma_utils.py` can pin each local rank to its dedicated NIC.

**Until we move that detection into Python**, the validated launch path is
still `mndp.yaml` (with the wrapper script). The zero-scripts yaml is
provided for reference / for setups where you can hardcode the per-rank
device list.

## Prerequisites

### 1. Build / install Mooncake (provides `mooncake_master` + `mooncake_client`)

```bash
git clone https://github.com/ivanium/Mooncake ~/repos/Mooncake
cd ~/repos/Mooncake
git checkout yifan/dev
./scripts/dev_compile.sh
./scripts/dev_install.sh    # installs into ~/repos/vllm-mooncake/.venv
```

After install, `mooncake_master` and `mooncake_client` are on `PATH` whenever the venv is active. Vigil activates the venv automatically when `repo_path: /home/$USER/repos/vllm-mooncake` is set on the step.

### 2. Install vLLM (already in the validation venv)

```bash
cd ~/repos/vllm-mooncake
# venv is .venv/; vllm binary is .venv/bin/vllm
```

### 3. SSD scratch path

The owner offloads CPU spillover to disk. The default path is
`/mnt/data/$USER/mooncake_offload/`. Create + chown if it doesn't exist:

```bash
sudo mkdir -p /mnt/data/$USER && sudo chown $USER /mnt/data/$USER
```

The path **must be a real local SSD**, not NFS / Lustre — Mooncake uses
`O_DIRECT` writev with 4 KiB alignment which networked filesystems either
silently fall back to buffered I/O or reject.

### 4. Discover your RDMA RNICs

The owner exposes its CPU pool over RDMA. Find your RNICs:

```bash
rdma link show
```

Edit `mndp_noscripts.yaml`'s `MC_OWNER_DEVICES` to match. The validated GB200 hardware uses:

```yaml
MC_OWNER_DEVICES: "rocep139s0,rocep140s0,rocep195s0,rocep196s0"
```

### 5. `nc` (netcat) on PATH

Used for port-readiness probes between phases.

```bash
which nc || sudo apt-get install netcat-openbsd
```

## Run

### Pre-flight cleanup

vigil's `post_serve` cleanup is best-effort. If a previous run crashed mid-flight, leftover `mooncake_master` / `mooncake_client` / `EngineCore` processes can leak. Always sweep before a new run:

```bash
pkill -9 -f mooncake_client mooncake_master 'vllm serve' EngineCore
sleep 2
nvidia-smi --query-gpu=memory.used --format=csv,noheader   # should all show 0 MiB
```

### Launch

```bash
cd ~/repos/vllm-mooncake
~/repos/model-ci/.venv/bin/vigil -c mndp_noscripts.yaml
```

Vigil orchestrates everything:

| Phase | What | Time |
|---|---|---|
| precheck | binary + venv existence | <1 s |
| collect-env | dump env_info.log | <1 s |
| pre_serve[0] | spawn `mooncake_master`, wait for RPC :50051 + HTTP :8080 | ~1.5 s |
| pre_serve[1] | spawn `mooncake_client`, wait for segment :50053 | ~10 s |
| serving | `vllm serve` ready (port 8100) | ~70 s |
| router | round-robin router on `{router_port}` | ~10 s |
| post_serve[0] | vmon start (resource tracker) | <1 s |
| post_serve[1] | `vllm-bench` — 100 conv × 3 turns × 32 concurrency, 16 k input | ~13 min |
| post_serve[2] | vmon stop | <1 s |
| post_serve[3] | kill owner + master via PID files | ~2 s |

### Watch progress

```bash
RUN=$(ls -td ~/repos/model-ci/logs/mndp_noscripts/$(date +%Y-%m-%d)/* | head -1)
tail -f $RUN/worker_0.log              # vllm + tier-log lines
tail -f $RUN/mooncake_master.log
tail -f $RUN/mooncake_owner.log
tail -f $RUN/hook_post_serve_1_vllm-bench.log   # bench progress
```

Watch for tier-log lines in `worker_0.log` like:

```
Mooncake load tier summary: req_id=… batch_keys=226 memory_keys=0
   disk_keys=226 unknown_keys=0 success_keys=226 failed_keys=0
   bytes_by_tier={'memory': 0, 'disk': 533200896, 'unknown': 0}
```

`disk_keys > 0` + `failed_keys = 0` means the disk-tier readback path is healthy.

### Expected results

Validation runs against `feat/mooncake-store-connector` + PR-47:

| Signal | Expected |
|---|---|
| Successful requests (100×3 turns) | 100/100 |
| Mean TTFT | 60–70 s |
| Median TTFT | 60–80 s |
| Failed requests / RPC_FAIL / tracebacks | 0 / 0 / 0 |
| Tier-log lines | 100–130 |
| Disk bytes returned | 50–70 GB |

## What can go wrong

### `mooncake_master crashed during startup`

The pre_serve readiness probe reports this when `mooncake_master` exits within the first 60 × 0.5 s. Causes:

- Port 50051 or 8080 already bound → check `ss -tlnp | grep -E ":(50051|8080)\s"`. Kill the holder, re-run.
- Binary not on PATH → confirm `command -v mooncake_master` returns a path. If not, the venv didn't activate; check `repo_path` on the step is set to `/home/$USER/repos/vllm-mooncake`.

### `mooncake_client crashed during startup` with "Duplicate rpc_meta key"

A previous owner registered its segment descriptor in the HTTP metadata server before crashing, and the descriptor outlives the process. Fix:

```bash
# Inspect what's stuck
curl -s "http://127.0.0.1:8080/metadata?key=mooncake/rpc_meta/127.0.0.1:50053"
# Delete it (or DELETE the whole rpc_meta tree)
curl -X DELETE "http://127.0.0.1:8080/metadata?key=mooncake/rpc_meta/127.0.0.1:50053"
curl -X DELETE "http://127.0.0.1:8080/metadata?key=mooncake/ram/127.0.0.1:50053"
```

Then re-run. The shell-script wrapper version (`run_vllm_with_mooncake_owner.sh`) did this cleanup automatically; the no-scripts yaml does not (yet). Long-term, switching peer discovery to P2PHANDSHAKE eliminates this failure mode entirely (see `P2P_VS_HTTP_METADATA.md`) — but at a measured performance cost on long benchmarks.

### CPU pool plateau at 95 %, no disk writes

`MC_LEASE_TTL` default is 30 min. Memory replicas can't be evicted until lease expires, so the disk-tier path never fires. The yaml hardcodes `-default_kv_lease_ttl=30000` (30 s) for benchmarking; if you change it, eviction reverts to the default.

### "RuntimeError: CUDA out of memory" on phase 2 startup

A previous `vllm serve` left EngineCore worker processes holding GPU memory. The `pkill` cleanup snippet at the top of this runbook is the fix.

### Disk bucket files accumulating

vigil's cleanup kills the owner but doesn't `rm -rf /mnt/data/$USER/mooncake_offload/`. After many runs you'll see residual GB on disk. Safe to delete manually between runs:

```bash
rm -rf /mnt/data/$USER/mooncake_offload/*.bucket
```

## Switching benchmark parameters

`mndp_noscripts.yaml` has these knobs you might want to tune:

| Field | Default | Notes |
|---|---|---|
| `model` (top of file) | `Qwen/Qwen3-8B` | Any HF model |
| `--gpu-memory-utilization` | `0.4` | Lower → smaller KV cache → more disk pressure |
| `--num-gpu-blocks-override` | `1024` | Caps KV cache size |
| `--max-model-len` | `16384` | Max sequence length |
| `MC_OWNER_CPU_MEM_GIB` | `4` | Owner CPU pool. Set small to force disk spillover |
| `MC_OWNER_DISK_GIB` | `1000` | Owner disk quota |
| `--random-input-len` (post_serve bench) | `16000` | Input length per turn |
| `--num-prompts` (post_serve bench) | `100` | Bench size |

## Multi-node

The yaml is single-node. For multi-node:

1. Run `mooncake_master` on one node only; have the others point at it via `--master_server_address=<master-host>:50051` and `--metadata_server=http://<master-host>:8080/metadata`.
2. Each compute node runs its own `mooncake_client` advertising its segment as `<node-host>:50053`.
3. Each vllm rank on that node sets `MOONCAKE_PREFERRED_SEGMENT=<node-host>:50053` so it routes to *its* local owner (round-trips stay node-local).
4. vigil supports `serving.roles[*].count > 1` for multi-instance launches.

## See also

- `P2P_VS_HTTP_METADATA.md` — why this runbook uses HTTP metadata, not P2PHANDSHAKE, and the perf trade-off
- `~/repos/vllm-mooncake/mndp_noscripts.yaml` — the actual pipeline config
- `~/repos/vllm-mooncake/scripts/mooncake/mooncake_config.json` — vllm-side connection params (read at runtime via `MOONCAKE_CONFIG_PATH`)
