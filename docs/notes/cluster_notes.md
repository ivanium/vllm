# Cluster Notes

Last updated: 2026-04-07

## Hardware

### GPUs
- 4x NVIDIA GB200 (Grace-Blackwell)
- VRAM: 189 GB each
- Driver: 580.126.09
- CUDA Version: 13.0 (nvcc: 13.1 at `/usr/local/cuda-13.1`)

### Storage: `/mnt/data`
- **Type**: 4x NVMe RAID-0 (`nvme_card0~3` via `/dev/md0`), XFS filesystem
- **Capacity**: 12 TB total, ~11 TB available
- **Sequential write**: ~5.7 GB/s
- **Sequential read**: ~6.7 GB/s
- **Root permissions**: `777` — anyone can create subdirectories
- Each user owns their own subdirectory (e.g. `/mnt/data/aoshen/`)

#### Existing directories in `/mnt/data`
| Directory | Owner |
|---|---|
| `mooncake_offload` | zijingliu (775, no write for others) |
| `jeejee/` | jeejeelee |
| `robin/` | robin |
| `rogerw/` | rogerw |
| `woosuk/` | woosuk |
| `yasong/` | yasong_wang |
| `yongye/` | yongye |

### Storage: `/mnt/lustre`
- Shared lustre filesystem
- HuggingFace model cache at `/mnt/lustre/hf-models`

---

## Software Environment

### Python / venv
- **uv** installed at `~/.local/bin/uv`
- **Python 3.12** venv at `~/code/uv_envs/py312` (general purpose)
- **vllm venv** at `~/setup_new_cluster/vllm/.venv` (use for all vllm/mooncake work)

### vllm
- **Repo**: `~/setup_new_cluster/vllm` — ivanium fork, branch `feat/mooncake-store-int`
  - This is the Inferact-customized vllm with Mooncake KV store integration
  - Upstream vllm clone is at `~/code/vllm` (not used for benchmarks)
- **Version**: `0.1.dev15437+g9998c1db5.precompiled`
- **Install command**:
  ```bash
  cd ~/setup_new_cluster/vllm
  source $HOME/.local/bin/env
  uv venv --python 3.12
  source .venv/bin/activate
  VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_VARIANT=cu130 \
    uv pip install -vvv -e . --reinstall-package vllm --torch-backend=cu130
  ```
- **PyTorch**: `2.10.0+cu130`, CUDA available: True
- **Sanity check**:
  ```bash
  source ~/setup_new_cluster/vllm/.venv/bin/activate
  python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
  python -c "import vllm.vllm_flash_attn; print('flash_attn OK')"
  ```

### Mooncake
- **Wheel** (pre-built for Grace-Blackwell aarch64):
  `~/setup_new_cluster/vllm/scripts/mooncake/mooncake_transfer_engine-0.3.10.post1-cp312-cp312-manylinux_2_39_aarch64.whl`
- **Install**:
  ```bash
  source ~/setup_new_cluster/vllm/.venv/bin/activate
  uv pip install scripts/mooncake/mooncake_transfer_engine-0.3.10.post1-cp312-cp312-manylinux_2_39_aarch64.whl
  ```
- **Verify**:
  ```bash
  bash scripts/mooncake/start_mooncake_master.sh --bg
  python scripts/mooncake/mooncake_example.py   # should print "Hello, Mooncake Store!"
  bash scripts/mooncake/start_mooncake_master.sh --stop
  ```

### Rust
- rustup installed, version 1.94.1 at `~/.cargo/bin/rustc`
- System rust at `/usr/bin/rustc` (1.75, old — use rustup version)

### Shell
- zsh at `/usr/bin/zsh`, oh-my-zsh installed
- `.bashrc` exec's zsh on login
- Key env vars set in both `.bashrc` and `.zshrc`:
  - `CUDA_HOME=/usr/local/cuda-13.1`
  - `~/.local/bin` in PATH (uv, vmon, vllm-bench)
  - `~/.cargo/env` sourced (rust)
  - `~/code/uv_envs/py312` activated

---

## Available Models (in `/mnt/lustre/hf-models/hub`)

| Model | Notes |
|---|---|
| `facebook/opt-125m` | Tiny, good for quick sanity checks |
| `Qwen/Qwen3-0.6B` | Small |
| `Qwen/Qwen3-8B` | Mid-size, used for benchmarks |
| `Qwen/Qwen3.5-9B` | |
| `Qwen/Qwen3.5-397B-A17B-FP8` | Large MoE |
| `nvidia/Qwen3.5-397B-A17B-NVFP4` | Large MoE, NVFP4 |
| `nvidia/Kimi-K2.5-NVFP4` | |
| `nvidia/DeepSeek-R1-0528-NVFP4-v2` | |
| `nvidia/DeepSeek-V3.2-NVFP4` | |
| `deepseek-ai/DeepSeek-V3.2` | |

Set `HF_HOME=/mnt/lustre/hf-models` to use cached models.

---

## Benchmark Setup

### Prerequisites
```bash
source ~/setup_new_cluster/vllm/.venv/bin/activate
export HF_HOME=/mnt/lustre/hf-models
export CUDA_HOME=/usr/local/cuda-13.1
export PATH=$HOME/.local/bin:$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
cd ~/setup_new_cluster/vllm
bash scripts/mooncake/start_mooncake_master.sh --bg
```

### Disk offload path
Use `/mnt/data/aoshen/mooncake_offload` (NOT `/mnt/data/mooncake_offload` — that's owned by zijingliu, no write permission).

### Run benchmark (trial: small)
```bash
MOONCAKE_CONFIG_PATH=scripts/mooncake/mooncake_config.json \
CPU_OFFLOAD_GIB=80 \
DISK_OFFLOAD_GIB=400 \
BACKENDS=baseline,mooncake \
bash scripts/mooncake/benchmark_cpu_offloading.sh Qwen/Qwen3-8B 2048 256 10
```

### Stop master
```bash
bash scripts/mooncake/start_mooncake_master.sh --stop
```

---

## TODOs
- [ ] Ask @Yasong Wang for secrets → put to `/mnt/shared/mqa/.env`
