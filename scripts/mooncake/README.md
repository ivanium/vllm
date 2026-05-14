# Mooncake KV Store for vLLM

## Two launch paths

| Path | Where | When to use |
|---|---|---|
| **`verify/`** (recommended) | `scripts/mooncake/verify/mndp_noscripts.yaml` | Production / new setups. Inlines `mooncake_master` + `mooncake_client` directly in the vigil yaml, RDMA NIC assignment done in Python via `vllm/.../mooncake/rdma_utils.py`. |
| **`legacy/`** | `scripts/mooncake/legacy/mndp.yaml` | Pre-existing wrapper-script flow. Useful for comparison / debugging — uses `start_mooncake_master.sh`, `run_vllm_with_mooncake_owner.sh`, etc. |

## Quickstart

```bash
cd ~/repos/vllm-mooncake

# Verify (no-scripts) path — recommended:
~/repos/model-ci/.venv/bin/vigil -c scripts/mooncake/verify/mndp_noscripts.yaml

# Legacy (wrapper-script) path — equivalent but with bash wrappers:
~/repos/model-ci/.venv/bin/vigil -c scripts/mooncake/legacy/mndp.yaml
```

See `verify/RUNBOOK.md` for the full step-by-step launch guide,
prerequisites, troubleshooting, and tuning knobs.

## Files at this directory

- `README.md` — this file
- `mooncake_config.json` — connection config consumed by `MOONCAKE_CONFIG_PATH`; referenced by both `verify/` and `legacy/`. `legacy/mooncake_config.json` is a symlink back here so the wrapper script's `${SCRIPT_DIR}/mooncake_config.json` default still resolves.

  Ships with **`"mode": "owner-client"`** + `"global_segment_size": 0` because both pipelines in this repo spawn a separate `mooncake_client` owner. To run the legacy PR-40900 baseline (every rank contributes a CPU segment, no separate owner), switch to `"mode": "real-client"` + `"global_segment_size": "4GB"`. See `~/repos/disk/design/dual-mode-design.md` for the full mode contract and `~/repos/disk/design/toggle-runbook.md` for the switch procedure.

## Install Mooncake

Both paths assume `mooncake_master` and `mooncake_client` are on PATH (i.e.
`mooncake-transfer-engine` wheel installed into the vllm-mooncake venv).
Build from source:

```shell
git clone https://github.com/ivanium/Mooncake
cd Mooncake
git checkout yifan/dev
./scripts/dev_compile.sh
./scripts/dev_install.sh
```

Build flags (set in `dev_compile.sh`):
- `-DUSE_CUDA=ON`
- `-DWITH_NVIDIA_PEERMEM=OFF` (required on GB200 — peermem kernel module won't load)
- `-DUSE_MNNVL=ON`

## See also

- `~/repos/disk/deploy/RUNBOOK.md` — the launch runbook (also copied into `verify/`)
- `~/repos/disk/deploy/P2P_VS_HTTP_METADATA.md` — peer-discovery mode trade-off
- `~/repos/disk/deploy/RDMA_UTILS.md` — why per-rank NIC pinning is necessary and how it's done
