# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

"""Worker-side logic for MooncakeStoreConnector.

Includes the store worker, transfer threads, lookup server,
and MooncakeDistributedStore integration.

本模块实现了 MooncakeStoreConnector 的 Worker 侧核心逻辑，包括：
  - MooncakeDistributedStore 的初始化与 KV 缓存注册
  - 后台异步发送/接收线程（KVCacheStoreSendingThread / KVCacheStoreRecvingThread）
  - 前缀查询服务（LookupKeyServer）
  - 磁盘 offload 分批策略
整体数据流：
  调度器下发 metadata → get_finished() 将请求分发到后台线程 → 后台线程通过
  MooncakeDistributedStore 的 batch_put / batch_get 完成跨节点 KV Cache 传输。
"""

import json
import os
import queue
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import regex as re
import torch
import zmq

from vllm.config import VllmConfig
from vllm.distributed import (
    get_dcp_group,
    get_pcp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.distributed.kv_events import BlockStored
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_store_data import (
    ChunkedTokenDatabase,
    KeyMetadata,
    MooncakeStoreConnectorMetadata,
    ReqMeta,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_store_metrics import (  # noqa: E501
    MooncakeStoreConnectorStats,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_store_scheduler import (  # noqa: E501
    get_zmq_rpc_path_lookup,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_utils import (
    get_mooncake_dp_engine_index,
)
from vllm.logger import init_logger
from vllm.utils.network_utils import get_ip, make_zmq_socket
from vllm.v1.core.kv_cache_utils import BlockHash, maybe_convert_block_hash
from vllm.v1.serial_utils import MsgpackDecoder

logger = init_logger(__name__)

# ============================================================
# 全局常量
# ============================================================

# MooncakeDistributedStore 中全局共享内存段的默认大小：4 GiB
DEFAULT_GLOBAL_SEGMENT_SIZE = 4 * 1024 * 1024 * 1024  # 4 GiB

# 本地传输缓冲区的默认大小：4 GiB（用于 RDMA/TCP 传输缓冲）
DEFAULT_LOCAL_BUFFER_SIZE = 4 * 1024 * 1024 * 1024  # 4 GiB

# Mooncake 返回此错误码表示 CPU/磁盘 offload 处于背压状态，暂时无可用 handle
MOONCAKE_NO_AVAILABLE_HANDLE = -200

# 磁盘 offload 暂存缓冲区的默认大小：1280 MiB
DEFAULT_MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE = 1280 * 1024 * 1024

# 磁盘 offload 可用预算占总预算的比例（留出 10% 安全裕量，避免边界情况溢出）
DISK_OFFLOAD_USABLE_BUDGET_RATIO = 0.9

# Direct I/O 要求的对齐粒度（4 KiB）
_DIRECT_IO_ALIGNMENT = 4096

# Direct I/O 写入时额外预留的对齐填充字节（两倍对齐，用于首尾对齐保护）
_DIRECT_IO_PADDING_BYTES = 2 * _DIRECT_IO_ALIGNMENT


# ============================================================
# 配置数据类
# ============================================================


@dataclass
class MooncakeStoreConfig:
    """MooncakeDistributedStore 的运行时配置。

    字段说明：
      metadata_server       : 元数据服务器地址（如 etcd/redis endpoint），
                              用于 Mooncake 节点发现。
      global_segment_size   : 全局共享内存段大小（字节），影响跨节点可见的
                              KV Cache 容量上限。
      local_buffer_size     : 本地传输缓冲区大小（字节），影响并发传输带宽。
      protocol              : 传输协议，支持 "tcp" / "rdma"。
      device_name           : RDMA 网卡设备名（TCP 模式下可为空）。
      master_server_address : Mooncake Master Server 地址，用于全局协调。
      enable_offload        : 是否启用 CPU/磁盘 offload（将 KV Cache 下沉到
                              CPU 内存或磁盘以扩大有效容量）。
    """

    metadata_server: str
    global_segment_size: int
    local_buffer_size: int
    protocol: str
    device_name: str
    master_server_address: str
    enable_offload: bool = False

    @staticmethod
    def from_file(file_path: str) -> "MooncakeStoreConfig":
        """从 JSON 配置文件加载 MooncakeStoreConfig。

        enable_offload 优先级：
          JSON 字段 > 环境变量 MOONCAKE_ENABLE_OFFLOAD（"1"/"true" 均视为启用）。
        """
        with open(file_path) as file:
            config = json.load(file)
        # 支持 JSON 字段或环境变量两种方式开启 offload
        enable_offload = config.get("enable_offload", False) or os.getenv(
            "MOONCAKE_ENABLE_OFFLOAD", ""
        ).lower() in ("1", "true")
        return MooncakeStoreConfig(
            metadata_server=config.get("metadata_server", ""),
            global_segment_size=_parse_size(
                config.get("global_segment_size", DEFAULT_GLOBAL_SEGMENT_SIZE)
            ),
            local_buffer_size=_parse_size(
                config.get("local_buffer_size", DEFAULT_LOCAL_BUFFER_SIZE)
            ),
            protocol=config.get("protocol", "tcp"),
            device_name=config.get("device_name", ""),
            master_server_address=config.get("master_server_address", ""),
            enable_offload=enable_offload,
        )

    @staticmethod
    def load_from_env() -> "MooncakeStoreConfig":
        """从环境变量 MOONCAKE_CONFIG_PATH 指向的 JSON 文件加载配置。

        若环境变量未设置则抛出 ValueError。
        """
        config_path = os.getenv("MOONCAKE_CONFIG_PATH")
        if not config_path:
            raise ValueError(
                "The environment variable 'MOONCAKE_CONFIG_PATH' is not set."
            )
        return MooncakeStoreConfig.from_file(config_path)


# ============================================================
# 辅助函数：磁盘 offload 预算计算
# ============================================================


def _get_disk_offload_buffer_budget_bytes(enable_offload: bool) -> int | None:
    """返回磁盘 offload 暂存缓冲区的原始预算（字节）。

    - 若 enable_offload 为 False，返回 None（不限制）。
    - 否则优先读取环境变量 MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES，
      未设置则使用默认值 DEFAULT_MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE。
    """
    if not enable_offload:
        return None
    value = os.getenv("MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES")
    if value is None:
        return DEFAULT_MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE
    return _parse_size(value)


def _parse_size(value: Any) -> int:
    """将存储大小字符串（支持 GB/MB/KB/B 单位）或数值解析为字节数。

    示例：
      "4GB"  → 4 * 1024^3
      "512mb" → 512 * 1024^2
      1024    → 1024（已是整数，直接返回）
    """
    if isinstance(value, int):
        return value
    if not isinstance(value, str):
        try:
            return int(value)
        except (TypeError, ValueError) as e:
            raise TypeError(f"Unsupported type for size: {type(value)}") from e

    cleaned = value.strip().lower()
    if not cleaned:
        raise ValueError("Size cannot be empty.")

    # 单位 → 字节倍数映射表
    unit_multipliers = {
        "gb": 1024**3,
        "mb": 1024**2,
        "kb": 1024,
        "b": 1,
    }
    match = re.match(r"^\s*([\d.]+)\s*(gb|mb|kb|b)?\s*$", cleaned)
    if not match:
        raise ValueError(f"Invalid format: '{value}'")

    number_str = match.group(1)
    unit = match.group(2) or "b"  # 无单位则默认按字节处理
    multiplier = unit_multipliers[unit]

    try:
        numeric_value = float(number_str)
    except ValueError as exc:
        raise ValueError(f"Invalid numeric value '{number_str}' in: '{value}'") from exc
    return int(numeric_value * multiplier)


def _align_up(value: int, alignment: int) -> int:
    """将 value 向上对齐到 alignment 的整数倍。

    用于 Direct I/O 场景：文件系统要求读写偏移和大小均对齐到块大小（通常 4 KiB）。
    """
    return ((value + alignment - 1) // alignment) * alignment


def _estimate_disk_offload_staging_bytes(size_list: list[int]) -> int:
    """估算将一组张量分片写入磁盘所需的暂存缓冲区大小（字节）。

    计算逻辑：
      1. 对所有分片字节数求和得到 data_size。
      2. 向上对齐到 _DIRECT_IO_ALIGNMENT（4 KiB）。
      3. 额外加上 _DIRECT_IO_PADDING_BYTES（8 KiB）作为首尾对齐保护。

    这是保守估算，确保 Direct I/O 暂存不会溢出预分配缓冲区。
    """
    data_size = sum(size_list)
    return _align_up(data_size, _DIRECT_IO_ALIGNMENT) + _DIRECT_IO_PADDING_BYTES


def _get_usable_disk_offload_buffer_budget_bytes(raw_budget_bytes: int) -> int:
    """将原始预算乘以安全系数（0.9），得到实际可用预算。

    保留 10% 裕量，避免因估算误差或并发写入导致缓冲区溢出。
    最小返回 1 以避免零预算。
    """
    return max(1, int(raw_budget_bytes * DISK_OFFLOAD_USABLE_BUDGET_RATIO))


def _get_usable_disk_offload_batch_key_count(num_keys: int) -> int:
    """将 key 总数乘以安全系数（0.9），得到单批次最大 key 数量。

    与字节预算类似，防止单批次 key 数量过多导致 offload 引擎过载。
    """
    return max(1, int(num_keys * DISK_OFFLOAD_USABLE_BUDGET_RATIO))


def _split_disk_offload_load_batches(
    keys: list[str],
    addrs: list[list[int]],
    sizes: list[list[int]],
    usable_budget_bytes: int,
    raw_budget_bytes: int,
) -> tuple[list[tuple[list[str], list[list[int]], list[list[int]]]], str | None]:
    """将一批 KV Cache load 请求按磁盘 offload 缓冲区预算拆分为若干子批次。

    目的：
      磁盘 offload 需要将 KV Cache 先暂存到内存缓冲区再写盘/读盘。若一批
      数据超出缓冲区容量，必须拆分为更小的批次分批加载，避免 OOM。

    参数：
      keys              : KV Cache 键列表。
      addrs             : 每个 key 对应的内存地址列表（多分片）。
      sizes             : 每个 key 对应的分片字节数列表。
      usable_budget_bytes : 可用缓冲区预算（= raw_budget * 0.9）。
      raw_budget_bytes  : 原始缓冲区预算（用于判断单个 key 是否超限）。

    返回：
      (batches, oversized_key)
        batches        : 拆分后的子批次列表，每个元素为 (keys, addrs, sizes)。
        oversized_key  : 若某个 key 本身超出原始预算（无法加载），返回该 key；
                         否则为 None。

    拆分策略：
      - 若单 key 超出 raw_budget → 无法加载，返回 ([], oversized_key)。
      - 若单 key 超出 usable_budget 但 ≤ raw_budget → 单独成批（特殊大块）。
      - 否则按 usable_budget 和最大 key 数量累积，超限时切批。
    """
    # 计算单批次最大 key 数（带安全系数）
    max_batch_keys = _get_usable_disk_offload_batch_key_count(len(keys))
    batches: list[tuple[list[str], list[list[int]], list[list[int]]]] = []
    batch_keys: list[str] = []
    batch_addrs: list[list[int]] = []
    batch_sizes: list[list[int]] = []
    batch_bytes = 0  # 当前批次累计字节数

    for key, addr, size in zip(keys, addrs, sizes, strict=True):
        key_bytes = _estimate_disk_offload_staging_bytes(size)

        # 情况 1：单个 key 超过原始预算上限 → 完全无法 offload，返回错误
        if key_bytes > raw_budget_bytes:
            return [], key

        # 情况 2：单个 key 超过可用预算（但不超过原始预算）→ 单独成批处理
        if key_bytes > usable_budget_bytes:
            if batch_keys:
                # 先将已积累的批次提交
                batches.append((batch_keys, batch_addrs, batch_sizes))
                batch_keys, batch_addrs, batch_sizes = [], [], []
                batch_bytes = 0
            batches.append(([key], [addr], [size]))
            continue

        # 情况 3：加入当前批次后超出预算或 key 数量上限 → 切批
        if batch_keys and (
            batch_bytes + key_bytes > usable_budget_bytes
            or len(batch_keys) >= max_batch_keys
        ):
            batches.append((batch_keys, batch_addrs, batch_sizes))
            batch_keys, batch_addrs, batch_sizes = [], [], []
            batch_bytes = 0

        # 将当前 key 加入批次
        batch_keys.append(key)
        batch_addrs.append(addr)
        batch_sizes.append(size)
        batch_bytes += key_bytes

    # 提交最后一个未满的批次
    if batch_keys:
        batches.append((batch_keys, batch_addrs, batch_sizes))
    return batches, None


# ============================================================
# Transfer Threads（传输后台线程）
# ============================================================


class KVTransferThread(threading.Thread):
    """KV Cache 异步传输线程的基类。

    设计思想：
      - 继承自 threading.Thread，以 daemon 模式运行，不阻塞主进程退出。
      - 使用 queue.Queue 作为请求队列，实现生产者-消费者模型：
          主线程（生产者）调用 add_request() 提交请求；
          后台线程（消费者）在 run() 中循环处理请求。
      - 完成的请求 ID 存入 finished_requests，主线程通过
        get_and_clear_finished_requests() 拉取结果。
      - KV 事件（BlockStored）通过 kv_events 列表异步收集，供上层消费。

    子类需实现 _handle_request() 来处理具体的 put/get 逻辑。
    """

    def __init__(
        self,
        store: Any,
        token_database: ChunkedTokenDatabase,
        block_size: int,
        tp_rank: int,
        ready_event: threading.Event,
        name: str,
    ):
        # daemon=True：主进程退出时自动终止此线程，无需手动 join
        super().__init__(daemon=True, name=name)
        # MooncakeDistributedStore 实例，提供 batch_put / batch_get 接口
        self.store = store
        # 线程启动完毕后 set，用于调用方等待线程就绪
        self.ready_event = ready_event
        # KV Cache 的逻辑块大小（token 数）
        self.block_size = block_size
        # 当前 Tensor Parallel rank（用于 key 哈希分片）
        self.tp_rank = tp_rank
        # 管理 token → KV Cache 地址映射的数据库
        self.token_database = token_database
        # 保护 finished_requests / stored_requests 的互斥锁
        self.done_task_lock = threading.Lock()
        # 待处理请求的阻塞队列
        self.request_queue: queue.Queue[Any] = queue.Queue()
        # 已完成传输的请求 ID 集合（由子类写入，主线程读取）
        self.finished_requests: set[str] = set()
        # 保护 kv_events 的互斥锁
        self.kv_event_lock = threading.Lock()
        # 已存储完毕的 Block 事件列表（用于通知上层缓存系统）
        self.kv_events: list[BlockStored] = []

    def add_request(self, request: ReqMeta) -> None:
        """将一个请求放入传输队列（线程安全，可从任意线程调用）。"""
        self.request_queue.put(request)

    def get_and_clear_finished_requests(self) -> set[str]:
        """原子地取走并清空已完成的请求 ID 集合。

        主线程每步调用一次，获取上一步已完成传输的请求集合。
        """
        with self.done_task_lock:
            finished = self.finished_requests.copy()
            self.finished_requests.clear()
        return finished

    def set_finished_request(self, req_id: str):
        """将 req_id 加入已完成集合（由子线程内部调用）。"""
        with self.done_task_lock:
            self.finished_requests.add(req_id)

    def run(self):
        """线程主循环：就绪后持续从队列取请求并处理。

        - 先 set ready_event，通知调用方线程已启动。
        - 收到 None 时记录警告但继续运行（非退出信号，避免意外终止）。
        - 所有异常均被捕获并记录，确保线程不因单个错误崩溃。
        """
        self.ready_event.set()
        while True:
            try:
                request_data = self.request_queue.get()
                if request_data is None:
                    logger.warning("Received a None request!")
                    self.request_queue.task_done()
                    continue
                self._handle_request(request_data)
            except Exception as e:
                logger.error("Error in %s: %s", self.name, e)

    def _handle_request(self, req_meta: Any):
        """处理单个请求的具体逻辑，由子类覆盖实现。"""
        pass

    def update_kv_event(self, events: list[BlockStored]):
        """线程安全地追加新的 BlockStored 事件（由子线程调用）。"""
        with self.kv_event_lock:
            self.kv_events.extend(events)

    def get_kv_events(self) -> list[BlockStored]:
        """原子地取走并清空待消费的 KV 事件列表（由主线程调用）。"""
        with self.kv_event_lock:
            events = self.kv_events.copy()
            self.kv_events.clear()
        return events


class KVCacheStoreSendingThread(KVTransferThread):
    """后台线程：将本节点的 KV Cache 块写入 MooncakeDistributedStore（put）。

    核心职责：
      1. 接收来自主线程的存储请求（ReqMeta）。
      2. 查询哪些 block 在 store 中已存在（去重），避免重复写入。
      3. 通过 batch_put_from_multi_buffers 将缺失的 block 批量写入 store。
      4. 感知 CPU/磁盘 offload 背压（MOONCAKE_NO_AVAILABLE_HANDLE 错误码），
         在压力期跳过后续写入以保护系统稳定性。
      5. 可选地生成 BlockStored KV 事件，供上层 KV 事件系统消费。

    put_step 与 TP 分片：
      当 num_kv_head < tp_size 时，多个 TP rank 共享同一组 KV head，
      此时 put_step > 1，每个 TP rank 仅负责写入其对应的 1/put_step 子集，
      避免重复写入同一份数据。
    """

    def __init__(
        self,
        store: Any,
        token_database: ChunkedTokenDatabase,
        block_size: int,
        tp_rank: int,
        put_step: int,
        kv_role: str,
        ready_event: threading.Event,
        enable_kv_event: bool = False,
        xfer_stats: MooncakeStoreConnectorStats | None = None,
        stats_lock: threading.Lock | None = None,
    ):
        super().__init__(
            store,
            token_database,
            block_size,
            tp_rank,
            ready_event,
            name="KVCacheStoreSendingThread",
        )
        # TP 写入步长：当 tp_size > num_kv_head 时 > 1，否则 = 1
        self.put_step = put_step
        # Worker 角色："kv_producer"（仅写）/ "kv_both"（读写）/ "kv_consumer"（仅读）
        self.kv_role = kv_role
        # 记录每个请求尚未完成的写入 job 数量（原子计数，归零即代表全部写完）
        self.stored_requests: defaultdict[str, int] = defaultdict(int)
        # 是否生成 BlockStored KV 事件
        self.enable_kv_event = enable_kv_event
        self.xfer_stats = xfer_stats
        self._stats_lock = stats_lock

        # 磁盘 offload 背压控制标志
        # _store_pressure_active: 当前是否处于背压状态
        self._store_pressure_active = False
        # _skip_store_requests: 背压期间应跳过写入的请求 ID 集合
        self._skip_store_requests: set[str] = set()

    def add_stored_request(self, req_id: str):
        """为 req_id 新增一个待完成的写入 job（引用计数 +1）。

        每次向队列提交一个写入请求前调用，用于跟踪该请求共有多少个
        并发写入任务，只有全部完成后才将 req_id 加入 finished_sending。
        """
        with self.done_task_lock:
            self.stored_requests[req_id] += 1

    def dec_stored_request(self, req_id: str):
        """完成一个写入 job 后将 req_id 的引用计数 -1。"""
        with self.done_task_lock:
            if req_id in self.stored_requests:
                self.stored_requests[req_id] -= 1

    def delete_finished_stored_request(self, req_id: str):
        """彻底移除 req_id 的所有写入跟踪记录（计数归零或被抢占时调用）。"""
        with self.done_task_lock:
            if req_id in self.stored_requests:
                del self.stored_requests[req_id]
            # 同时清理跳过集合，避免后续误判
            self._skip_store_requests.discard(req_id)

    def _should_skip_request(self, req_id: str) -> bool:
        """判断当前是否应跳过对 req_id 的写入（背压期间且已标记需跳过）。"""
        with self.done_task_lock:
            return self._store_pressure_active and req_id in self._skip_store_requests

    def _mark_request_skipped_for_pressure(self, req_id: str) -> bool:
        """将 req_id 标记为"因背压而跳过"，并激活背压状态。

        返回：该请求是否已经被标记过（True = 已标记，False = 首次标记）。
        首次标记时上层会打印 warning，避免重复日志。
        """
        with self.done_task_lock:
            already_skipped = req_id in self._skip_store_requests
            self._store_pressure_active = True
            self._skip_store_requests.add(req_id)
        return already_skipped

    def _clear_store_pressure(self) -> bool:
        """清除背压状态（在写入成功后调用）。

        返回：是否确实清除了背压（True = 之前有背压，False = 本来就无背压）。
        用于打印"背压已解除"的 info 日志，避免无意义的重复输出。
        """
        with self.done_task_lock:
            if not self._store_pressure_active and not self._skip_store_requests:
                return False
            self._store_pressure_active = False
            self._skip_store_requests.clear()
        return True

    def _handle_request(self, req_meta: ReqMeta):
        """处理单个 KV Cache 写入请求的核心逻辑。

        流程：
          1. 校验请求是否仍有效（可能已被取消/抢占）。
          2. 检查背压，若处于压力期则跳过本次写入。
          3. 遍历 token_database 将 token 范围转换为 (start, end, key) 三元组。
          4. 按 put_step 取子集（TP 分片写入）。
          5. batch_is_exist 查询去重，只写入 store 中不存在的 block。
          6. prepare_value 获取每个 block 的内存地址和大小。
          7. 同步 CUDA 事件，确保 GPU 计算已完成再读取 KV Cache。
          8. batch_put_from_multi_buffers 批量写入，处理失败和背压错误码。
          9. 若启用 kv_event，生成 BlockStored 事件链。
        """
        token_len = req_meta.token_len_chunk      # 本次需写入的 token 数量
        block_ids = req_meta.block_ids             # GPU 物理 block 索引列表
        req_id = req_meta.req_id                   # 请求唯一 ID
        current_event = req_meta.current_event     # CUDA 同步事件（可为 None）

        # 若请求已被主线程移除（抢占/取消），直接丢弃
        if req_id not in self.stored_requests:
            self.request_queue.task_done()
            return
        # 背压保护：跳过本批写入并递减引用计数
        if self._should_skip_request(req_id):
            logger.debug(
                "Skipping Mooncake store for request %s while CPU/disk offloading "
                "is under pressure",
                req_id,
            )
            self.dec_stored_request(req_id)
            self.request_queue.task_done()
            return

        # Step 1：将 token 范围转换为 (start, end, key) 列表
        starts = []
        ends = []
        keys = []
        block_hashes: list[BlockHash] = []
        for index, (start, end, key) in enumerate(
            self.token_database.process_tokens(token_len, req_meta.block_hashes)
        ):
            starts.append(start)
            ends.append(end)
            keys.append(key.to_string())
            block_hashes.append(req_meta.block_hashes[index])

        # Step 2：按 put_step 对列表做步进切片，实现 TP 级别的写入分片
        # 例如 tp_rank=1, put_step=2 → 取下标 1, 3, 5, ... 的 block
        starts = starts[self.tp_rank % self.put_step :: self.put_step]
        ends = ends[self.tp_rank % self.put_step :: self.put_step]
        keys = keys[self.tp_rank % self.put_step :: self.put_step]
        block_hashes = block_hashes[self.tp_rank % self.put_step :: self.put_step]

        if not keys:
            # 本 TP rank 无需写入任何 block（全部由其他 rank 负责）
            self.dec_stored_request(req_id)
            return

        # Step 3：去重检查，避免重复写入已存在的 block（节省带宽和 store 容量）
        exists_states = self.store.batch_is_exist(keys)
        missing_indices = [i for i, exists in enumerate(exists_states) if exists != 1]

        if not missing_indices:
            # 所有 block 在 store 中均已存在，无需写入
            self.dec_stored_request(req_id)
            return

        # 只保留缺失的 block
        starts = [starts[i] for i in missing_indices]
        ends = [ends[i] for i in missing_indices]
        keys = [keys[i] for i in missing_indices]
        block_hashes = [block_hashes[i] for i in missing_indices]

        logger.debug(
            "Storing KV cache for %d out of %d blocks "
            "(missing_count=%d) for request %s",
            len(keys),
            token_len // self.block_size,
            len(missing_indices),
            req_id,
        )

        # Step 4：为每个缺失 block 准备内存地址和字节大小
        addrs = []
        sizes = []
        stored_events: list[BlockStored] = []
        prev_key = None  # 用于构建 BlockStored 事件链（父子关系）
        # 将 BlockHash 转换为可序列化的标准格式
        new_block_hashes = [maybe_convert_block_hash(bh) for bh in block_hashes]

        for index, start in enumerate(starts):
            # prepare_value 返回：(base_addr, size_list, _)
            # base_addr 是 GPU KV Cache 对应分片的内存起始地址列表
            addr, size, _ = self.token_database.prepare_value(
                start, ends[index], block_ids
            )
            addrs.append(addr)
            sizes.append(size)

            if self.enable_kv_event:
                # 构建 BlockStored 事件，记录 token ids、block hash、存储介质等
                token_ids = (
                    req_meta.token_ids[start : ends[index]]
                    if req_meta.token_ids is not None
                    else None
                )
                stored_event = BlockStored(
                    block_hashes=[new_block_hashes[index]],
                    parent_block_hash=prev_key,   # 上一个 block 的 hash，构成链式前缀
                    token_ids=token_ids,
                    block_size=req_meta.original_block_size,
                    lora_id=None,
                    medium="cpu",  # KV Cache 写入 CPU/磁盘时标记为 "cpu"
                    lora_name=None,
                )
                stored_events.append(stored_event)
                prev_key = new_block_hashes[index]

        # Step 5：等待 CUDA 事件（确保 GPU 计算完成，KV Cache 数据已就绪）
        # current_event 由主线程在 get_finished() 中录制，所有写入请求共享同一事件
        if current_event is not None:
            current_event.synchronize()

        # Step 6：批量写入 MooncakeDistributedStore
        try:
            t0 = time.perf_counter()
            res = self.store.batch_put_from_multi_buffers(keys, addrs, sizes)
            elapsed = time.perf_counter() - t0
            ok_bytes = sum(
                (sum(size) if isinstance(size, list) else size)
                for size, ret in zip(sizes, res)
                if ret >= 0
            )
            if ok_bytes > 0 and self.xfer_stats is not None and self._stats_lock:
                with self._stats_lock:
                    self.xfer_stats.record_put(ok_bytes, elapsed)
            failed = [i for i, v in enumerate(res) if v < 0]
            if failed:
                # 统计本批次失败信息（错误码、字节数等）用于诊断
                total_bytes = sum(sum(s) if isinstance(s, list) else s for s in sizes)
                failed_codes = set(res[i] for i in failed)
                transfer_fail_keys = sum(1 for i in failed if res[i] == -1)
                no_available_handle_keys = sum(
                    1 for i in failed if res[i] == MOONCAKE_NO_AVAILABLE_HANDLE
                )
                other_failed_keys = (
                    len(failed) - transfer_fail_keys - no_available_handle_keys
                )
                if self.xfer_stats is not None and self._stats_lock:
                    with self._stats_lock:
                        self.xfer_stats.record_put_failures(
                            failed_batches=1,
                            failed_keys=len(failed),
                            transfer_fail_keys=transfer_fail_keys,
                            no_available_handle_keys=no_available_handle_keys,
                            other_failed_keys=other_failed_keys,
                        )
                logger.warning(
                    "batch_put failed: %d/%d keys failed "
                    "(codes=%s, batch_bytes=%d, num_keys=%d), "
                    "first_key=%s",
                    len(failed),
                    len(keys),
                    failed_codes,
                    total_bytes,
                    len(keys),
                    keys[0] if keys else "N/A",
                )
                # 若检测到 NO_AVAILABLE_HANDLE（offload 背压），激活背压保护
                # _mark_request_skipped_for_pressure 返回 True 说明已标记过，不重复 warning
                if (
                    MOONCAKE_NO_AVAILABLE_HANDLE in failed_codes
                    and not self._mark_request_skipped_for_pressure(req_id)
                ):
                    logger.warning(
                        "Detected Mooncake CPU/disk offloading pressure "
                        "(NO_AVAILABLE_HANDLE); skipping future store "
                        "batches for request %s until a later store "
                        "batch succeeds",
                        req_id,
                    )
            elif self._clear_store_pressure():
                # 写入成功且之前处于背压状态 → 背压解除，打印恢复日志
                logger.info(
                    "Mooncake CPU/offload pressure cleared after a "
                    "successful store batch"
                )
        except Exception as e:
            logger.error("Failed to put key %s, error: %s", keys, e)

        # Step 7：将 KV 事件推送给事件收集器
        if self.enable_kv_event and stored_events:
            self.update_kv_event(stored_events)

        # 递减引用计数，标记本次 job 完成
        self.dec_stored_request(req_id)
        self.request_queue.task_done()


class KVCacheStoreRecvingThread(KVTransferThread):
    """后台线程：从 MooncakeDistributedStore 拉取 KV Cache 块到本地 GPU（get）。

    核心职责：
      1. 接收来自主线程的加载请求（ReqMeta）。
      2. 查询 token_database 将 token 范围转换为 (key, addr, size) 列表。
      3. 按 tp_rank 做循环偏移（rotate）以实现负载均衡。
      4. 若启用磁盘 offload，按缓冲区预算将加载批次拆分为多个子批次。
      5. 通过 batch_get_into_multi_buffers 将远端 KV Cache 直接写入本地 GPU 内存。
      6. 完成后将 req_id 加入 finished_requests，由主线程消费。

    磁盘 offload 分批策略：
      磁盘 offload 需要暂存缓冲区，若一次加载量超出预算则调用
      _split_disk_offload_load_batches 拆分，分批串行加载。
    """

    def __init__(
        self,
        store: Any,
        token_database: ChunkedTokenDatabase,
        block_size: int,
        tp_rank: int,
        ready_event: threading.Event,
        disk_offload_buffer_budget_bytes: int | None = None,
        xfer_stats: MooncakeStoreConnectorStats | None = None,
        stats_lock: threading.Lock | None = None,
    ):
        super().__init__(
            store,
            token_database,
            block_size,
            tp_rank,
            ready_event,
            name="KVCacheStoreRecvingThread",
        )
        self.xfer_stats = xfer_stats
        self._stats_lock = stats_lock
        # 原始磁盘 offload 缓冲区预算（None 表示不限制 / 未启用 offload）
        self.disk_offload_buffer_budget_bytes = disk_offload_buffer_budget_bytes
        # 带安全系数（*0.9）的可用预算，避免边界溢出
        self.usable_disk_offload_buffer_budget_bytes = (
            None
            if disk_offload_buffer_budget_bytes is None
            else _get_usable_disk_offload_buffer_budget_bytes(
                disk_offload_buffer_budget_bytes
            )
        )

    def _handle_request(self, req_meta: ReqMeta):
        """处理单个 KV Cache 加载请求的核心逻辑。

        流程：
          1. 从 load_spec 中提取需加载的 token 数量和 vllm 本地缓存偏移量（mask_num）。
          2. 遍历 token_database 生成 (key, addr, size) 三元组。
          3. 按 tp_rank 对列表做循环偏移，分散热点请求的读取起点。
          4. 若启用 offload 且批次超出预算，拆分为多个子批次逐一加载。
          5. batch_get_into_multi_buffers 将远端数据直接写入本地 GPU 内存。
          6. 记录完成状态。
        """
        # load_spec.token_len：本次实际需要从 store 加载的 token 数
        token_len = req_meta.load_spec.token_len  # type: ignore[union-attr]
        req_id = req_meta.req_id
        # mask_num：vllm 本地 KV Cache 中已有的完整 block token 数，
        # 这些 block 无需从 store 加载（本地命中）
        mask_num = (
            req_meta.load_spec.vllm_cached_tokens  # type: ignore[union-attr]
            // self.block_size
            * self.block_size
        )

        # Step 1：生成所有待加载 block 的 (key, addr, size) 三元组
        addr_list = []
        size_list = []
        key_list = []
        for start, end, key in self.token_database.process_tokens(
            token_len, req_meta.block_hashes, mask_num
        ):
            # prepare_value 返回目标 GPU 内存地址和分片大小
            addr, size, _ = self.token_database.prepare_value(
                start, end, req_meta.block_ids
            )
            key_list.append(key.to_string())
            addr_list.append(addr)
            size_list.append(size)

        # Step 2：按 tp_rank 做循环偏移（rotate），实现跨请求的负载均衡
        # 例如 tp_rank=1，共 4 个 block → 顺序变为 [1,2,3,0]
        # 使得不同 TP rank 从不同起点开始加载，降低远端节点的集中读压力
        key_list_c = (
            key_list[self.tp_rank % len(key_list) :]
            + key_list[: self.tp_rank % len(key_list)]
        )
        addr_list_c = (
            addr_list[self.tp_rank % len(addr_list) :]
            + addr_list[: self.tp_rank % len(addr_list)]
        )
        size_list_c = (
            size_list[self.tp_rank % len(size_list) :]
            + size_list[: self.tp_rank % len(size_list)]
        )

        # Step 3：决定加载批次（默认一次加载全部；offload 场景下按预算拆分）
        load_batches = [(key_list_c, addr_list_c, size_list_c)]
        if self.usable_disk_offload_buffer_budget_bytes is not None:
            # 估算所有 block 的总暂存需求字节数
            total_staging_bytes = sum(
                _estimate_disk_offload_staging_bytes(size) for size in size_list_c
            )
            usable_batch_keys = _get_usable_disk_offload_batch_key_count(
                len(key_list_c)
            )
            # 若总量超出可用预算或 key 数过多，则需要拆批
            if (
                total_staging_bytes > self.usable_disk_offload_buffer_budget_bytes
                or len(key_list_c) > usable_batch_keys
            ):
                assert self.disk_offload_buffer_budget_bytes is not None
                load_batches, oversized_key = _split_disk_offload_load_batches(
                    key_list_c,
                    addr_list_c,
                    size_list_c,
                    self.usable_disk_offload_buffer_budget_bytes,
                    self.disk_offload_buffer_budget_bytes,
                )
                if oversized_key is not None:
                    # 单个 key 超出原始预算上限，无法加载 → 跳过整个请求
                    oversized_key_index = key_list_c.index(oversized_key)
                    oversized_key_bytes = _estimate_disk_offload_staging_bytes(
                        size_list_c[oversized_key_index]
                    )
                    logger.warning(
                        "Skipping Mooncake load for request %s because key %s "
                        "requires %d staging bytes, exceeding budget %d",
                        req_id,
                        oversized_key,
                        oversized_key_bytes,
                        self.disk_offload_buffer_budget_bytes,
                    )
                    self.set_finished_request(req_id)
                    self.request_queue.task_done()
                    return

        # Step 4：逐子批次从 store 拉取数据（串行，保证顺序写入 GPU 内存）
        current_batch_keys: list[str] = key_list_c
        try:
            for batch_keys, batch_addrs, batch_sizes in load_batches:
                current_batch_keys = batch_keys
                # batch_get_into_multi_buffers：零拷贝直接写入目标 GPU 内存地址
                t0 = time.perf_counter()
                res = self.store.batch_get_into_multi_buffers(
                    batch_keys, batch_addrs, batch_sizes
                )
                elapsed = time.perf_counter() - t0
                ok_bytes = sum(
                    (sum(size) if isinstance(size, list) else size)
                    for size, ret in zip(batch_sizes, res)
                    if ret >= 0
                )
                if ok_bytes > 0 and self.xfer_stats is not None and self._stats_lock:
                    with self._stats_lock:
                        self.xfer_stats.record_get(ok_bytes, elapsed)
                # 检查每个 key 的返回码，< 0 表示失败
                failed = [
                    (key, value)
                    for key, value in zip(batch_keys, res, strict=True)
                    if value < 0
                ]
                if failed:
                    logger.warning(
                        "Failed to get %d Mooncake keys from sub-batch "
                        "(batch_keys=%d, first_failures=%s)",
                        len(failed),
                        len(batch_keys),
                        failed[:3],  # 只打印前 3 个失败条目，避免日志过长
                    )
                    break  # 子批次失败时终止后续子批次，避免数据不一致
        except Exception as e:
            logger.warning(
                "Failed to get Mooncake sub-batch %s, error: %s",
                current_batch_keys[:3],
                e,
            )

        # 无论成功与否均标记请求完成（调用方根据实际 GPU 数据判断是否命中）
        self.set_finished_request(req_id)
        self.request_queue.task_done()


# ============================================================
# Store Worker（核心工作组件）
# ============================================================


class MooncakeStoreWorker:
    """Worker 侧的 MooncakeStoreConnector 核心组件。

    生命周期：
      1. __init__：初始化 MooncakeDistributedStore，建立与 metadata server 的连接，
                   计算 TP/PP/PCP/DCP 等并行维度的 rank 信息。
      2. register_kv_caches：GPU KV Cache 分配完毕后调用，
                   注册 GPU 内存到 store，并启动后台传输线程。
      3. get_finished：每个 forward step 调用，下发 I/O 请求并收取已完成结果。
      4. lookup：前缀查询，判断远端 store 中已存储了多少前缀 token 的 KV Cache。

    并行维度说明：
      tp_rank/tp_size  : Tensor Parallel（张量并行），同一层的 KV head 按 TP 切分。
      pp_rank/pp_size  : Pipeline Parallel（流水线并行），不同层分布在不同 PP stage。
      pcp_rank/pcp_size: Prefill Context Parallel，KV Cache 跨 prefill 节点聚合。
      dcp_rank/dcp_size: Decode Context Parallel，KV Cache 跨 decode 节点聚合。
      dp_rank          : Data Parallel rank（用于多实例部署中的实例标识）。
    """

    def __init__(self, vllm_config: VllmConfig):
        # 尝试导入 Mooncake Python 包，未安装则给出清晰的安装指引
        try:
            from mooncake.store import MooncakeDistributedStore  # type: ignore
        except ImportError as e:
            raise ImportError(
                "Please install mooncake by following the instructions at "
                "https://github.com/kvcache-ai/Mooncake/blob/main/doc/"
                "en/build.md to run vLLM with MooncakeStoreConnector."
            ) from e

        model_config = vllm_config.model_config
        parallel_config = vllm_config.parallel_config

        self.dp_rank = get_mooncake_dp_engine_index(parallel_config)
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.pp_size = parallel_config.pipeline_parallel_size
        # pp_rank：从全局 rank 推算当前所在的 Pipeline 阶段
        self.pp_rank = (parallel_config.rank // self.tp_size) % self.pp_size

        # PCP（Prefill Context Parallel）组信息
        self.pcp_size = get_pcp_group().world_size
        self.pcp_rank = get_pcp_group().rank_in_group if self.pcp_size > 1 else 0
        # DCP（Decode Context Parallel）组信息
        self.dcp_size = get_dcp_group().world_size
        self.dcp_rank = get_dcp_group().rank_in_group if self.dcp_size > 1 else 0

        # Worker 角色：kv_producer / kv_consumer / kv_both
        self.kv_role = vllm_config.kv_transfer_config.kv_role
        # 强制使用异步 load（compute-I/O overlap 必要条件）
        # NOTE(yifan): enforce load_async for now for better compute-I/O overlap.
        self.load_async = True

        # ── block_size 计算 ──────────────────────────────────────────────
        self.cache_config = vllm_config.cache_config
        self.original_block_size = self.cache_config.block_size
        self.block_size = self.cache_config.block_size
        # PCP/DCP > 1 时，多个节点合并 KV Cache，逻辑 block_size 相应扩大
        if self.pcp_size > 1:
            self.block_size *= self.pcp_size
        if self.dcp_size > 1:
            self.block_size *= self.dcp_size
        # 当前 PP stage 负责的 Transformer 层数
        self.num_layers = model_config.get_num_layers(parallel_config)

        # ── MLA（Multi-head Latent Attention）适配 ──────────────────────
        # MLA 将 KV 压缩为单个 latent head，num_kv_head 固定为 1
        self.use_mla = False
        if (
            hasattr(model_config, "use_mla")
            and isinstance(model_config.use_mla, bool)
            and model_config.use_mla
        ):
            self.use_mla = True

        if self.use_mla:
            self.num_kv_head = 1
        else:
            # 标准 MHA/GQA 场景：从模型配置获取总 KV head 数（所有 TP rank 的总和）
            self.num_kv_head = model_config.get_total_num_kv_heads()

        # ── put_step 计算（TP 写入分片策略）────────────────────────────
        # 当 KV head 数少于 TP 大小时（如极端 GQA），多个 TP rank 持有相同的 KV head，
        # 若全部写入会导致重复。put_step > 1 确保每份 KV head 只由一个 TP rank 写入。
        if self.num_kv_head < self.tp_size:
            self.put_step = self.tp_size // self.num_kv_head
            # 有效 KV head rank（合并持有相同 head 的多个 TP rank）
            self.head_or_tp_rank = self.tp_rank // self.put_step
        else:
            # 普通情况：每个 TP rank 写入自己的 KV head
            self.head_or_tp_rank = self.tp_rank
            self.put_step = 1

        # ── Key 元数据（用于生成全局唯一的 block key）───────────────────
        # key 格式示例："{model_name}@tp_rank:{rank}@pp_rank:{rank}@{block_hash}"
        self.metadata = KeyMetadata(
            model_name=model_config.model.rstrip("/").split("/")[-1],
            tp_rank=self.head_or_tp_rank,
            pcp_rank=self.pcp_rank,
            dcp_rank=self.dcp_rank,
            pp_rank=self.pp_rank,
        )

        # token → KV Cache 地址映射数据库（懒初始化，register_kv_caches 后才完整）
        self.token_database = ChunkedTokenDatabase(self.metadata, self.block_size)

        # ── 初始化 MooncakeDistributedStore ────────────────────────────
        store_config = MooncakeStoreConfig.load_from_env()
        self.store = MooncakeDistributedStore()

        # 组装 store 配置字典（使用本机 IP 作为 local_hostname）
        local_seg = get_ip()
        config_dict = {
            "local_hostname": local_seg,
            "metadata_server": store_config.metadata_server,
            "global_segment_size": str(store_config.global_segment_size),
            "local_buffer_size": str(store_config.local_buffer_size),
            "protocol": store_config.protocol,
            "rdma_devices": store_config.device_name,
            "master_server_addr": store_config.master_server_address,
        }
        if store_config.enable_offload:
            config_dict["enable_offload"] = "true"

        ret = self.store.setup(config_dict)
        if ret != 0:
            msg = "Initialize MooncakeDistributedStore failed."
            logger.error(msg)
            raise RuntimeError(msg)

        # 磁盘 offload 暂存缓冲区预算（None = 未启用 offload）
        self.disk_offload_buffer_budget_bytes = _get_disk_offload_buffer_budget_bytes(
            store_config.enable_offload
        )

        # ── KV 事件配置 ─────────────────────────────────────────────────
        kv_event_config = vllm_config.kv_events_config
        self.enable_kv_events = False
        if kv_event_config and kv_event_config.enable_kv_cache_events:
            self.enable_kv_events = True

        # 后台传输线程（在 register_kv_caches 中创建并启动）
        self.kv_send_thread: KVCacheStoreSendingThread | None = None
        self.kv_recv_thread: KVCacheStoreRecvingThread | None = None
        # 跟踪"已被 vllm 调度器完成但 store 写入尚未完全结束"的请求 ID
        self.finished_store_req: set[str] = set()
        self.xfer_stats = MooncakeStoreConnectorStats()
        self._stats_lock = threading.Lock()

    def get_kv_connector_stats(self) -> MooncakeStoreConnectorStats | None:
        with self._stats_lock:
            if self.xfer_stats.is_empty():
                return None
            return self.xfer_stats.clone_and_reset()

    def register_cross_layers_kv_caches(self, kv_cache: torch.Tensor) -> None:
        """Register a cross-layers KV cache tensor.

        Wraps the unified tensor in a single-entry dict so that the
        existing stride-based logic in register_kv_caches() produces
        the correct single-segment result (block_len = page_size * num_layers).
        """
        self.register_kv_caches({"__cross_layer__": kv_cache})

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """Register KV cache tensors and start transfer threads."""
        # TODO(yifan): we haven't supported HMA yet.
        first_kv_cache = next(iter(kv_caches.values()))

        # num_gpu_blocks 由 profiling 后的 cache_config 提供，是权威值
        assert self.cache_config.num_gpu_blocks is not None
        self.num_blocks = self.cache_config.num_gpu_blocks

        # ── 内存布局检测 ────────────────────────────────────────────────
        # 参考 simple_kv_offload/worker.py 中的 stride 推断方法
        #
        # 支持的后端布局：
        #   FlashAttn/ROCm : (2, num_blocks, ...) → K/V 在最外层
        #   FlashInfer/MLA : (num_blocks, ...)    → blocks 在最外层
        #
        # 推导：page_size_bytes = 每个物理 block 在存储中的字节大小，
        #       stride(d) * element_size > page_size_bytes → d 是段外维度。
        storage = first_kv_cache.untyped_storage()
        el = first_kv_cache.element_size()
        page_size_bytes = storage.nbytes() // self.num_blocks
        outer_dims = [
            d
            for d in range(first_kv_cache.ndim)
            if first_kv_cache.stride(d) * el > page_size_bytes
        ]

        # ── 注册 GPU 内存到 store 并记录 base_addr/block_len ────────────
        seen_ptrs: set[int] = set()          # 已注册过的内存起始地址（去重）
        self.kv_caches_base_addr: list[int] = []  # 各 layer/segment 的基地址
        self.block_len: list[int] = []            # 各 layer/segment 的单 block 字节数

        for cache in kv_caches.values():
            cache_storage = cache.untyped_storage()
            base_addr = cache_storage.data_ptr()
            region_len = cache_storage.nbytes()

            # 同一存储可能被多个 layer tensor 共享（如某些 GQA 实现），
            # 只注册一次避免重复 register_buffer 调用
            if base_addr not in seen_ptrs:
                seen_ptrs.add(base_addr)
                ret = self.store.register_buffer(base_addr, region_len)
                if ret != 0:
                    logger.error(
                        "register_buffer failed for addr %#x len %d: %d",
                        base_addr,
                        region_len,
                        ret,
                    )

            if not outer_dims:
                # Blocks-first 布局（FlashInfer / MLA）：整个 tensor 是一个连续段
                # kv_caches_base_addr 记录该层的 GPU 基地址
                # block_len = 每个物理 block 的字节数
                self.kv_caches_base_addr.append(base_addr)
                self.block_len.append(page_size_bytes)
            else:
                # K/V-first 布局（FlashAttn / ROCm）：K 和 V 分两段存储
                # outer_dims[0] 为 K/V 维度，shape[outer_dims[0]] = 2
                # 分别计算 K segment 和 V segment 的基地址
                seg_stride = cache.stride(outer_dims[0]) * el
                for idx in range(cache.shape[outer_dims[0]]):
                    self.kv_caches_base_addr.append(base_addr + idx * seg_stride)
                    # 每段内 block_len = 段的总字节数 / block 数
                    self.block_len.append(seg_stride // self.num_blocks)

        logger.info(
            "Registering KV_Caches. use_mla: %s, shape %s, "
            "num_blocks: %d, block_len: %s, "
            "per_key_bytes: %d, "
            "num_segments: %d",
            self.use_mla,
            first_kv_cache.shape,
            self.num_blocks,
            list(set(self.block_len)),
            sum(self.block_len),          # 每个 key（block）的总字节数
            len(self.kv_caches_base_addr),  # 段总数（layers × segments_per_layer）
        )

        # 将基地址和 block_len 注入 token_database，供传输线程调用 prepare_value
        self.token_database.set_kv_caches_base_addr(self.kv_caches_base_addr)
        self.token_database.set_block_len(self.block_len)

        # ── 启动后台传输线程 ────────────────────────────────────────────
        if self.kv_role in ["kv_producer", "kv_both"]:
            # Producer/Both 角色：启动写入线程
            ready_event_sending = threading.Event()
            self.kv_send_thread = KVCacheStoreSendingThread(
                self.store,
                self.token_database,
                self.block_size,
                self.tp_rank,
                self.put_step,
                self.kv_role,
                ready_event_sending,
                self.enable_kv_events,
                xfer_stats=self.xfer_stats,
                stats_lock=self._stats_lock,
            )
            self.kv_send_thread.start()
            # 注意：此处不等待 ready_event_sending，因为 send 线程可晚于 recv 启动

        # 所有角色均需启动接收线程（Consumer/Both 主动加载，Producer 也可能回读）
        ready_event_recving = threading.Event()
        self.kv_recv_thread = KVCacheStoreRecvingThread(
            self.store,
            self.token_database,
            self.block_size,
            self.tp_rank,
            ready_event_recving,
            disk_offload_buffer_budget_bytes=self.disk_offload_buffer_budget_bytes,
            xfer_stats=self.xfer_stats,
            stats_lock=self._stats_lock,
        )
        self.kv_recv_thread.start()
        # 等待接收线程就绪后再返回（确保后续 get_finished 调用时线程已运行）
        ready_event_recving.wait()

    def start_load_kv(
        self,
        metadata: MooncakeStoreConnectorMetadata,
    ):
        """空操作：load 请求在 get_finished() 中统一下发以实现 compute-I/O overlap。"""
        pass

    def wait_for_save(
        self,
        metadata: MooncakeStoreConnectorMetadata,
    ):
        """空操作：store 请求在 get_finished() 中统一下发以实现 compute-I/O overlap。"""
        pass

    def get_finished(
        self,
        finished_req_ids: set[str],
        meta: MooncakeStoreConnectorMetadata,
    ) -> tuple[set[str], set[str]]:
        """每步 forward 后调用：下发所有 I/O 请求，并返回本步已完成的传输结果。

        设计思路（compute-I/O overlap）：
          - 所有 load/store 请求均在模型计算启动后才下发（而非提前），
            使 GPU 计算与 KV Cache 传输尽可能并行。
          - load 请求：直接异步下发给 kv_recv_thread，不阻塞当前步。
          - store 请求：先录制 CUDA Event，下发给 kv_send_thread 后线程内
            synchronize() 等待 GPU 计算完成，再执行 CPU→store 的拷贝。
          - 完成集合：通过 _get_and_clear_finished_sending / get_and_clear_finished_requests
            收取上一步（或更早）已完成的传输结果。

        参数：
          finished_req_ids : 本步 vllm 调度器已完成（decode 结束）的请求 ID 集合。
          meta             : 本步的连接器元数据，包含所有活跃请求的 load/store 规格。

        返回：
          (done_sending, done_recving)
            done_sending : 本步已完成 store 写入的请求 ID 集合。
            done_recving : 本步已完成 load 读取的请求 ID 集合。
        """
        # ── 下发异步 Load 请求 ──────────────────────────────────────────
        for request in meta.requests:
            load_spec = request.load_spec
            if load_spec is None or not load_spec.can_load:
                continue  # 无 load 规格或不可加载，跳过

            token_len = request.token_len_chunk
            # 边界处理：若 kvpool_cached_tokens 不是 block_size 的整数倍，
            # 且恰好等于 token_len - 1，则加载到 kvpool_cached_tokens + 1
            # （处理最后一个非整 block 的特殊情况）
            if (load_spec.kvpool_cached_tokens % self.block_size != 0) and (
                load_spec.kvpool_cached_tokens == token_len - 1
            ):
                token_len = load_spec.kvpool_cached_tokens + 1
            else:
                token_len = load_spec.kvpool_cached_tokens
            load_spec.token_len = token_len

            assert self.kv_recv_thread is not None
            self.kv_recv_thread.add_request(request)  # 非阻塞投入队列

        assert self.load_async, "load_async must be True for better performance."

        # ── 下发异步 Store 请求 ─────────────────────────────────────────
        if self.kv_role in ["kv_producer", "kv_both"]:
            # 在所有 store 请求前录制一个共享的 CUDA Event，用于同步 GPU 计算完成
            # 只需录制一次，所有 store 请求共享同一个 event（它们都等待同一个计算步骤）
            current_event = None
            for request in meta.requests:
                if request.can_save:
                    current_event = torch.cuda.Event()
                    current_event.record()  # 录制当前 CUDA 流位置
                    break

            for request in meta.requests:
                if not request.can_save:
                    continue
                request.current_event = current_event
                assert self.kv_send_thread is not None
                # 先增加引用计数，再投入队列（防止计数为 0 时被误认为已完成）
                self.kv_send_thread.add_stored_request(request.req_id)
                self.kv_send_thread.add_request(request)

        # ── 收取已完成的传输结果 ────────────────────────────────────────
        # 收取发送完成集合（处理抢占和引用计数归零的逻辑）
        done_sending = (
            self._get_and_clear_finished_sending(finished_req_ids, meta)
            if self.kv_role in ["kv_producer", "kv_both"]
            else set()
        )

        # 收取接收完成集合（原子地取走并清空）
        done_recving = (
            self.kv_recv_thread.get_and_clear_finished_requests()
            if self.load_async and self.kv_recv_thread is not None
            else set()
        )

        logger.debug(
            "Completed send: %d, recv: %d, tp_rank: %d",
            len(done_sending),
            len(done_recving),
            self.tp_rank,
        )
        return done_sending, done_recving

    def _get_and_clear_finished_sending(
        self,
        finished_req_ids: set[str],
        meta: MooncakeStoreConnectorMetadata,
    ) -> set[str]:
        """收集已完成 KV Cache 写入的请求 ID 集合，并清理相关状态。

        完成判定逻辑（两个条件均满足才算完成）：
          1. stored_requests[req_id] == 0：该请求的所有写入 job 均已完成。
          2. req_id in finished_req_ids 或 req_id in finished_store_req：
             vllm 调度器已确认该请求结束（decode 完成或提前完成）。

        这种"双重检查"防止：请求已完成 decode 但还有 store job 在排队的竞态，
        或 store 已完成但调度器尚未宣布请求结束的竞态。

        抢占处理：
          preempted_req_ids 中的请求被强制中止，直接从 stored_requests 中删除，
          不再等待写入完成（抢占后 KV Cache 可能已失效）。
        """
        assert self.kv_send_thread is not None
        finished_sending: set[str] = set()

        # 清理被抢占请求的所有写入状态
        for req_id in meta.preempted_req_ids:
            self.kv_send_thread.delete_finished_stored_request(req_id)

        # 检查之前标记为"vllm 已完成但 store 未完成"的请求是否现在完成了
        for req_id in self.kv_send_thread.stored_requests.copy():
            if (
                self.kv_send_thread.stored_requests[req_id] == 0
                and req_id in self.finished_store_req
            ):
                self.finished_store_req.remove(req_id)
                finished_sending.add(req_id)
                self.kv_send_thread.delete_finished_stored_request(req_id)

        # 处理本步新完成的请求
        for req_id in finished_req_ids:
            req_remain_jobs = self.kv_send_thread.stored_requests.get(req_id)
            if req_remain_jobs == 0:
                # Store 已完成，且 vllm 也宣布完成 → 两侧均 ready
                finished_sending.add(req_id)
                self.kv_send_thread.delete_finished_stored_request(req_id)
            elif req_remain_jobs is not None:
                # vllm 完成了，但 store 还有 job 未结束 → 放入等待集合
                self.finished_store_req.add(req_id)

        return finished_sending

    def lookup(
        self,
        token_len: int,
        block_hashes: list[BlockHash],
    ) -> int:
        """查询远端 store 中已存储的最长前缀 token 数。

        用途：
          Prefill 节点在发送 KV Cache 前查询 Decode 节点 store 中已有多少前缀，
          避免重复传输已缓存的 KV block，节省带宽。

        查询策略：
          - 跨所有 TP rank 和 PP rank 并行查询（扩展 key 中的 tp_rank/pp_rank 字段）。
          - 对每个 TP rank/PP rank 的结果独立求出"第一个不存在的 block 下标"。
          - 取所有 rank 中最小的下标（最保守估计），作为安全可加载的前缀长度。

        返回：
          安全可加载的前缀 token 数（所有 TP/PP rank 中最短的命中前缀）。
          若出现异常，返回 0（降级为无前缀缓存）。
        """
        end = 0
        keys: list[str] = []
        try:
            starts: list[int] = []
            # 生成当前 tp_rank=0 / pp_rank=0 的 key 列表（基准 key）
            for start, end, key in self.token_database.process_tokens(
                token_len, block_hashes
            ):
                keys.append(key.to_string())
                starts.append(start)

            # 扩展到所有 TP rank（通过字符串替换 @tp_rank:0 → @tp_rank:i）
            multi_tp_keys = keys[:]
            for i in range(1, min(self.tp_size, self.num_kv_head)):
                for item in keys:
                    new_str = item.replace("@tp_rank:0", f"@tp_rank:{i}", 1)
                    multi_tp_keys.append(new_str)

            # 在 TP 扩展基础上，再扩展到所有 PP rank
            pp_base_keys = multi_tp_keys.copy()
            for i in range(1, self.pp_size):
                for item in pp_base_keys:
                    new_str = item.replace("@pp_rank:0", f"@pp_rank:{i}", 1)
                    multi_tp_keys.append(new_str)

            # 一次批量查询所有 TP × PP 组合的 key 存在状态
            res = self.store.batch_is_exist(multi_tp_keys)

            # 按 TP/PP rank 重新分组查询结果（每组 num_block 个 key）
            num_block = len(keys)
            multi_tp_values = [
                res[i * num_block : (i + 1) * num_block]
                for i in range(min(self.tp_size, self.num_kv_head) * self.pp_size)
            ]
            # 找出所有 rank 中最小的"第一个不存在 block"下标
            index = self._find_min_first_non_one_index(multi_tp_values)
            if index != -1:
                # 返回该 block 的 start token 偏移（即可安全加载的前缀长度）
                return starts[index]
        except Exception as e:
            logger.error("Remote connection failed in lookup: %s", e)
            return 0
        # 所有 rank 的所有 block 均存在 → 返回完整 token 长度
        return end

    @staticmethod
    def _find_min_first_non_one_index(
        arr: list[list[int]],
    ) -> int:
        """在二维数组 arr 中，找出所有行"第一个值不为 1 的下标"的最小值。

        用途：
          batch_is_exist 返回 1 表示 key 存在，非 1 表示不存在。
          此函数找出所有 TP/PP rank 中最早出现缓存 miss 的 block 下标，
          即跨所有 rank 的最短连续命中前缀长度。

        返回：
          最小的不存在 block 下标；若所有 block 均存在（全为 1），返回 -1。
        """
        try:
            return min(idx for row in arr for idx, val in enumerate(row) if val != 1)
        except ValueError:
            # 所有元素均为 1（全部存在），min() 对空迭代器抛 ValueError
            return -1

    def get_kv_events(self) -> list[BlockStored]:
        """获取并清空待消费的 BlockStored KV 事件列表。

        仅在 enable_kv_events=True 且有发送线程时返回实际事件，否则返回空列表。
        由框架的 KV 事件系统在每步 forward 后调用。
        """
        if self.enable_kv_events and self.kv_send_thread is not None:
            return self.kv_send_thread.get_kv_events()
        return []


# ============================================================
# Lookup Key Server（前缀查询服务）
# ============================================================


class LookupKeyServer:
    """Worker rank 0 上运行的 ZMQ REP 服务，响应调度器的前缀查询请求。

    架构说明：
      - 调度器（Scheduler）通过 ZMQ REQ-REP 模式向 Worker rank 0 查询
        "远端 store 中已存储了多少前缀 token 的 KV Cache"。
      - Worker rank 0 收到请求后调用 MooncakeStoreWorker.lookup() 查询 store，
        将结果（4 字节大端整数）返回给调度器。
      - 使用独立 daemon 线程处理请求，不阻塞主线程的 forward 流程。

    通信协议：
      请求：[token_len (4 bytes big-endian)] + [msgpack-encoded block_hashes (多帧)]
      响应：[lookup_result (4 bytes big-endian)]
    """

    def __init__(
        self,
        store_worker: MooncakeStoreWorker,
        vllm_config: VllmConfig,
    ):
        # msgpack 解码器，用于解析调度器发来的 block_hashes
        self.decoder = MsgpackDecoder()
        # ZMQ Context（每个进程一个，线程安全）
        self.ctx = zmq.Context()  # type: ignore[attr-defined]
        # 获取与调度器约定的 ZMQ socket 路径（通常为 unix domain socket）
        socket_path = get_zmq_rpc_path_lookup(vllm_config)
        # REP socket：接收请求并同步应答（Request-Reply 模式）
        self.socket = make_zmq_socket(
            self.ctx,
            socket_path,
            zmq.REP,  # type: ignore[attr-defined]
            bind=True,  # 服务端绑定地址
        )

        self.store_worker = store_worker
        self.running = True  # 控制服务线程退出的标志（预留，当前未使用停止逻辑）

        def process_request():
            """服务线程主循环：持续接收并处理前缀查询请求。

            协议解析：
              帧 0：token_len（4 字节大端整数）
              帧 1+：msgpack 编码的 block_hashes 列表
            """
            while self.running:
                # 接收多帧消息（zero-copy 模式，copy=False 减少内存拷贝）
                all_frames = self.socket.recv_multipart(copy=False)
                # 帧 0：token_len（4 字节大端整数）
                token_len = int.from_bytes(all_frames[0], byteorder="big")
                # 帧 1+：msgpack 编码的 block_hashes
                hash_frames = all_frames[1:]
                hashes_str = self.decoder.decode(hash_frames)
                # 调用 store worker 执行实际查询
                result = self.store_worker.lookup(token_len, hashes_str)
                # 将结果编码为 4 字节大端整数并返回
                response = result.to_bytes(4, "big")
                self.socket.send(response)

        # 以 daemon 线程启动服务，确保主进程退出时自动终止
        self.thread = threading.Thread(target=process_request, daemon=True)
        self.thread.start()

    def close(self):
        """关闭 ZMQ socket（linger=0 表示立即丢弃未发送消息，不等待）。"""
        self.socket.close(linger=0)
