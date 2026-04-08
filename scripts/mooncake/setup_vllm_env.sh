#!/usr/bin/env bash
# 为 vLLM 的 MooncakeStoreConnector 准备运行环境。
#
# 这个脚本的职责不是启动服务，而是“把当前 shell 会话里的环境准备好”，
# 供后续 `vllm serve ... --kv-transfer-config ...` 直接继承使用。
# 因此这里必须使用 `source` 执行，而不是直接 `bash setup_vllm_env.sh`：
# - `source`：导出的环境变量会留在当前 shell，后续启动的 vLLM 进程能读到。
# - `bash`：只会在子 shell 里生效，脚本结束后环境变量就丢失了。
#
# 根据 Mooncake / vLLM 侧代码，这个脚本主要完成两件事：
# 1. 设置 `MOONCAKE_CONFIG_PATH`，并把 JSON 配置里的 `global_segment_size`
#    改成当前节点愿意贡献给 Mooncake 全局内存池的 CPU 内存大小。
# 2. 在启用磁盘 offload 时，设置一组由 Mooncake Store 直接从环境变量读取的
#    SSD offload 参数，例如本地落盘目录、暂存缓冲区、总配额、淘汰策略等。
#
# 用法：
#   source setup_vllm_env.sh --cpu-mem-size 80
#       仅启用 CPU 内存池，当前节点向 Mooncake 全局池贡献 80 GB 内存。
#   source setup_vllm_env.sh --cpu-mem-size 80 --disk-size 400
#       在 80 GB CPU 内存池之外，再给本机配置 400 GB 的本地 SSD offload 空间。
#   source setup_vllm_env.sh --cpu-mem-size 80 --disk-size 400 --disk-path /nvme/offload
#       使用自定义 SSD 目录。该目录应为绝对路径；Mooncake 会拒绝相对路径、符号链接
#       或包含 `..` 的路径。
#
# 参数：
#   --cpu-mem-size <GB>  必填。写入 Mooncake JSON 的 `global_segment_size`。
#                        这表示当前节点贡献给分布式 KV 内存池的容量。
#   --disk-size    <GB>  选填。启用 Mooncake SSD offload，并设置磁盘总配额。
#   --disk-path    <dir> 选填。SSD offload 落盘目录，默认 /mnt/data/mooncake_offload。
#
set -euo pipefail

# 脚本所在目录。默认情况下，`mooncake_config.json` 与本脚本同目录放置。
SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- 解析命令行参数 ----------------------------------------------------------
# 这里先全部初始化为空字符串，便于后续判断用户是否显式传入了某个选项。
CPU_MEM_GB=""
DISK_GB=""
DISK_PATH=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cpu-mem-size)
            # 期望格式：--cpu-mem-size 80
            CPU_MEM_GB="$2"; shift 2 ;;
        --disk-size)
            # 期望格式：--disk-size 400
            DISK_GB="$2"; shift 2 ;;
        --disk-path)
            # 期望格式：--disk-path /absolute/path
            DISK_PATH="$2"; shift 2 ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Usage: source $0 --cpu-mem-size <GB> [--disk-size <GB>] [--disk-path <dir>]" >&2
            # 兼容两种调用方式：
            # - 若脚本是被 source 进来的，优先 return，避免直接退出用户当前 shell。
            # - 若脚本是被独立执行的，则退化为 exit。
            return 1 2>/dev/null || exit 1
            ;;
    esac
done

# `global_segment_size` 是 vLLM / Mooncake Store 初始化时的关键参数。
# Mooncake 的实现里会把它解析成字节数，并据此挂载本节点贡献给全局池的内存段。
# 因此这里把它设成必填项，避免以默认值误跑 benchmark。
if [[ -z "$CPU_MEM_GB" ]]; then
    echo "Error: --cpu-mem-size is required" >&2
    echo "Usage: source $0 --cpu-mem-size <GB> [--disk-size <GB>] [--disk-path <dir>]" >&2
    return 1 2>/dev/null || exit 1
fi

# --- Transfer Engine: TCP 连接池 ---------------------------------------------
# Mooncake Transfer Engine 的 TCP 传输层会读取 `MC_TCP_ENABLE_CONNECTION_POOL`。
# 开启后可复用持久连接，避免频繁创建/销毁 socket，尤其是在回退到 TCP 传输时能
# 显著减少连接管理开销。即使主路径是 RDMA，保留该设置也比较稳妥。
export MC_TCP_ENABLE_CONNECTION_POOL=1

# --- 更新 Mooncake JSON：声明本节点贡献的全局内存池大小 ----------------------
# vLLM 侧 `MooncakeStoreConfig.load_from_env()` 会优先读取环境变量
# `MOONCAKE_CONFIG_PATH`，再从这个 JSON 文件里拿到 `global_segment_size`、
# `local_buffer_size`、`protocol`、`master_server_address` 等配置。
#
# 如果外部已经提前设置了 `MOONCAKE_CONFIG_PATH`，这里尊重外部传入值；
# 否则默认使用脚本目录下的 `mooncake_config.json`。
export MOONCAKE_CONFIG_PATH="${MOONCAKE_CONFIG_PATH:-${SCRIPTS_DIR}/mooncake_config.json}"

# 这里直接原地改写 JSON，而不是再生成一个临时配置文件：
# - benchmark 脚本和 README 默认都引用这个固定路径；
# - Mooncake 支持 `"80GB"` 这种带单位的字符串格式；
# - 只改 `global_segment_size`，其余字段（master/metadata/protocol 等）保持不变。
python3 -c "
import json, sys

cfg_path, global_size_gb = sys.argv[1], sys.argv[2]
with open(cfg_path) as f:
    cfg = json.load(f)
# Mooncake / vLLM 都支持带单位的字符串，例如 \"4GB\"。
cfg['global_segment_size'] = global_size_gb + 'GB'
with open(cfg_path, 'w') as f:
    json.dump(cfg, f, indent=2)
    f.write('\n')
" "$MOONCAKE_CONFIG_PATH" "$CPU_MEM_GB"

echo "Mooncake env configured:"
echo "  Config:   $MOONCAKE_CONFIG_PATH"
echo "  CPU mem:  ${CPU_MEM_GB} GB (global segment)"

# --- SSD offload：这部分参数主要由 Mooncake Store 直接读环境变量 --------------
# 注意这里配置的是“客户端侧”的落盘行为：
# - vLLM 通过 MooncakeStoreConnector 把 KV 写入 Mooncake 的内存池；
# - 当 master 判断内存压力较大时，会通过 heartbeat 指示客户端把对象异步落到 SSD；
# - 缓存 miss 时，再从本地 SSD 回读。
# 也就是说，这里并不是让 vLLM 直接把 GPU KV 写盘，而是配置 Mooncake 的
# FileStorage / StorageBackend 行为。
if [[ -n "$DISK_GB" ]]; then
    # Mooncake 的路径校验要求使用绝对路径、真实目录、且必须可写。
    # 默认落在常见的数据盘挂载点下，用户也可以通过 --disk-path 覆盖。
    DISK_PATH="${DISK_PATH:-/mnt/data/mooncake_offload}"

    # 统一换算成字节。Mooncake 相关环境变量普遍以 bytes 为单位。
    DISK_BYTES=$(( DISK_GB * 1073741824 ))

    # 这里人为约束磁盘 offload 至少 10 GB：
    # - 小于这个值时，实际可用空间很快就会被 staging buffer、bucket 文件、
    #   元数据和淘汰预留空间吃掉；
    # - 对 benchmark 来说也很难观察到稳定行为。
    if [[ "$DISK_GB" -lt 10 ]]; then
        echo "Error: Disk size must be at least 10 GB" >&2
        return 1 2>/dev/null || exit 1
    fi

    # vLLM 侧会优先读取环境变量 `MOONCAKE_ENABLE_OFFLOAD`，
    # 只要值是 1 / true 即视为启用磁盘 offload。
    export MOONCAKE_ENABLE_OFFLOAD=1

    # Mooncake FileStorage 会直接从该变量读取 SSD 根目录。
    # 该目录必须存在且可写，因此后面会显式 `mkdir -p`。
    export MOONCAKE_OFFLOAD_FILE_STORAGE_PATH="$DISK_PATH"

    # 选择存储后端。当前默认使用 bucket backend：
    # - 把多个对象聚合到 bucket 文件里，减少小文件开销；
    # - 支持 FIFO / LRU 淘汰；
    # - 也是 Mooncake 文档里更推荐的通用后端。
    export MOONCAKE_OFFLOAD_STORAGE_BACKEND_DESCRIPTOR="${MOONCAKE_OFFLOAD_STORAGE_BACKEND_DESCRIPTOR:-bucket_storage_backend}"

    # bucket backend 的淘汰策略。这里默认使用 LRU，更符合 KV cache 的访问局部性。
    # 如果调用方已在外部设置，就保持外部值不动。
    export MOONCAKE_BUCKET_EVICTION_POLICY="${MOONCAKE_BUCKET_EVICTION_POLICY:-lru}"

    # 启用 io_uring 以降低本地文件 I/O 开销。
    # 如果固定缓冲区注册失败，Mooncake 会记录 warning 并回退到普通 I/O 路径，
    # 不会因此直接启动失败。
    export MOONCAKE_USE_URING="${MOONCAKE_USE_URING:-true}"

    # 本地 staging buffer，供 SSD 读写时做对齐和暂存使用。
    # vLLM 侧也会读取这个值来估算磁盘 offload 的可用缓冲预算。
    # 这里默认给 1 GB，低于 Mooncake 文档中的 1.25 GB 默认值，但更保守，
    # 也更容易避免因 memlock 限制导致的 io_uring 固定缓冲区注册失败。
    export MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES="${MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES:-1073741824}"

    # FileStorage 的总磁盘使用上限。对整个 offload 子系统来说，这是硬上限。
    export MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES="$DISK_BYTES"

    # bucket backend 自己的淘汰阈值。
    # 这里额外预留 40 GB 头寸，避免 bucket 文件、元数据恢复、文件系统开销等把盘写满。
    # 这是当前脚本的运维约定，并不是 Mooncake 框架强制要求。
    export MOONCAKE_BUCKET_MAX_TOTAL_SIZE=$(( DISK_BYTES - 42949672960 ))

    # heartbeat 周期。客户端会周期性把本地 offload 状态上报给 master，
    # master 也会借此下发需要落盘/回收的对象信息。
    # 这里用 3 秒，比 Mooncake 文档默认的 10 秒更激进，更适合 benchmark 快速触发。
    export MOONCAKE_OFFLOAD_HEARTBEAT_INTERVAL_SECONDS="${MOONCAKE_OFFLOAD_HEARTBEAT_INTERVAL_SECONDS:-3}"

    # 确保目录存在，避免被 Mooncake 的路径校验直接拒绝。
    mkdir -p "$MOONCAKE_OFFLOAD_FILE_STORAGE_PATH"

    echo "  Disk:     ${DISK_GB} GB (path=${DISK_PATH}, eviction=${MOONCAKE_BUCKET_EVICTION_POLICY})"
else
    # 不设置这些环境变量时，vLLM / Mooncake 将只使用内存池，不启用 SSD offload。
    echo "  Disk:     OFF (pass --disk-size <GB> to enable)"
fi
