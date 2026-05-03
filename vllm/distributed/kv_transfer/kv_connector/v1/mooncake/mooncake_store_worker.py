# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side logic for MooncakeStoreConnector.

Includes the store worker, transfer threads, lookup server,
and MooncakeDistributedStore integration.
"""

import json
import os
import queue
import threading
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from functools import reduce
from math import gcd, lcm
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
    GroupLayout,
    KeyMetadata,
    MooncakeStoreConnectorMetadata,
    ReqMeta,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_store_scheduler import (  # noqa: E501
    get_zmq_rpc_path_lookup,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_utils import (
    get_mooncake_dp_engine_index,
)
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.utils.network_utils import get_ip, make_zmq_socket
from vllm.v1.core.kv_cache_utils import BlockHash, maybe_convert_block_hash
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.serial_utils import MsgpackDecoder

from .mooncake_store_metrics import MooncakeStoreConnectorStats

logger = init_logger(__name__)

DEFAULT_GLOBAL_SEGMENT_SIZE = 4 * 1024 * 1024 * 1024  # 4 GiB
DEFAULT_LOCAL_BUFFER_SIZE = 4 * 1024 * 1024 * 1024  # 4 GiB
MOONCAKE_NO_AVAILABLE_HANDLE = -200
DEFAULT_MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE = 1280 * 1024 * 1024
DISK_OFFLOAD_USABLE_BUDGET_RATIO = 0.9
_DIRECT_IO_ALIGNMENT = 4096
_DIRECT_IO_PADDING_BYTES = 2 * _DIRECT_IO_ALIGNMENT


@dataclass
class MooncakeStoreConfig:
    """Configuration for MooncakeDistributedStore."""

    metadata_server: str
    global_segment_size: int
    local_buffer_size: int
    protocol: str
    device_name: str
    master_server_address: str
    enable_offload: bool = False

    @staticmethod
    def from_file(file_path: str) -> "MooncakeStoreConfig":
        with open(file_path) as file:
            config = json.load(file)
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
        config_path = os.getenv("MOONCAKE_CONFIG_PATH")
        if not config_path:
            raise ValueError(
                "The environment variable 'MOONCAKE_CONFIG_PATH' is not set."
            )
        return MooncakeStoreConfig.from_file(config_path)


def _get_disk_offload_buffer_budget_bytes(enable_offload: bool) -> int | None:
    if not enable_offload:
        return None
    value = os.getenv("MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES")
    if value is None:
        return DEFAULT_MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE
    return _parse_size(value)


def _parse_size(value: Any) -> int:
    """Parse storage size strings with units: GB, MB, KB, B."""
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
    unit = match.group(2) or "b"
    multiplier = unit_multipliers[unit]

    try:
        numeric_value = float(number_str)
    except ValueError as exc:
        raise ValueError(f"Invalid numeric value '{number_str}' in: '{value}'") from exc
    return int(numeric_value * multiplier)


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _estimate_disk_offload_staging_bytes(size_list: list[int]) -> int:
    data_size = sum(size_list)
    return _align_up(data_size, _DIRECT_IO_ALIGNMENT) + _DIRECT_IO_PADDING_BYTES


def _sum_batch_bytes(sizes: list[list[int]]) -> int:
    return sum(sum(size) for size in sizes)


def _get_usable_disk_offload_buffer_budget_bytes(raw_budget_bytes: int) -> int:
    return max(1, int(raw_budget_bytes * DISK_OFFLOAD_USABLE_BUDGET_RATIO))


def _get_usable_disk_offload_batch_key_count(num_keys: int) -> int:
    return max(1, int(num_keys * DISK_OFFLOAD_USABLE_BUDGET_RATIO))


def _split_disk_offload_load_batches(
    keys: list[str],
    addrs: list[list[int]],
    sizes: list[list[int]],
    usable_budget_bytes: int,
    raw_budget_bytes: int,
) -> tuple[list[tuple[list[str], list[list[int]], list[list[int]]]], str | None]:
    max_batch_keys = _get_usable_disk_offload_batch_key_count(len(keys))
    batches: list[tuple[list[str], list[list[int]], list[list[int]]]] = []
    batch_keys: list[str] = []
    batch_addrs: list[list[int]] = []
    batch_sizes: list[list[int]] = []
    batch_bytes = 0

    for key, addr, size in zip(keys, addrs, sizes, strict=True):
        key_bytes = _estimate_disk_offload_staging_bytes(size)
        if key_bytes > raw_budget_bytes:
            return [], key
        if key_bytes > usable_budget_bytes:
            if batch_keys:
                batches.append((batch_keys, batch_addrs, batch_sizes))
                batch_keys, batch_addrs, batch_sizes = [], [], []
                batch_bytes = 0
            batches.append(([key], [addr], [size]))
            continue
        if batch_keys and (
            batch_bytes + key_bytes > usable_budget_bytes
            or len(batch_keys) >= max_batch_keys
        ):
            batches.append((batch_keys, batch_addrs, batch_sizes))
            batch_keys, batch_addrs, batch_sizes = [], [], []
            batch_bytes = 0
        batch_keys.append(key)
        batch_addrs.append(addr)
        batch_sizes.append(size)
        batch_bytes += key_bytes

    if batch_keys:
        batches.append((batch_keys, batch_addrs, batch_sizes))
    return batches, None


# ============================================================
# Transfer Threads
# ============================================================


class KVTransferThread(threading.Thread):
    """Base class for async KV cache transfer threads."""

    def __init__(
        self,
        store: Any,
        token_database: ChunkedTokenDatabase,
        block_size: int,
        tp_rank: int,
        ready_event: threading.Event,
        name: str,
        record_operation: Callable[..., None] | None = None,
    ):
        super().__init__(daemon=True, name=name)
        self.store = store
        self.ready_event = ready_event
        self.block_size = block_size
        self.tp_rank = tp_rank
        self.token_database = token_database
        self._record_operation_cb = record_operation
        self.done_task_lock = threading.Lock()
        self.request_queue: queue.Queue[Any] = queue.Queue()
        self.finished_requests: set[str] = set()
        self.kv_event_lock = threading.Lock()
        self.kv_events: list[BlockStored] = []

    def add_request(self, request: ReqMeta) -> None:
        self.request_queue.put(request)

    def get_and_clear_finished_requests(self) -> set[str]:
        with self.done_task_lock:
            finished = self.finished_requests.copy()
            self.finished_requests.clear()
        return finished

    def set_finished_request(self, req_id: str):
        with self.done_task_lock:
            self.finished_requests.add(req_id)

    def run(self):
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
        pass

    def _record_operation(
        self,
        operation: str,
        start_time: float,
        num_keys: int,
        *,
        num_bytes: int = 0,
        status: str = "ok",
        num_failed_keys: int = 0,
    ) -> None:
        if self._record_operation_cb is None:
            return
        self._record_operation_cb(
            operation=operation,
            duration_seconds=time.perf_counter() - start_time,
            num_keys=num_keys,
            num_bytes=num_bytes,
            status=status,
            num_failed_keys=num_failed_keys,
        )

    def update_kv_event(self, events: list[BlockStored]):
        with self.kv_event_lock:
            self.kv_events.extend(events)

    def get_kv_events(self) -> list[BlockStored]:
        with self.kv_event_lock:
            events = self.kv_events.copy()
            self.kv_events.clear()
        return events


class KVCacheStoreSendingThread(KVTransferThread):
    """Background thread for storing KV cache blocks to the store."""

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
        record_operation: Callable[..., None] | None = None,
    ):
        super().__init__(
            store,
            token_database,
            block_size,
            tp_rank,
            ready_event,
            name="KVCacheStoreSendingThread",
            record_operation=record_operation,
        )
        self.put_step = put_step
        self.kv_role = kv_role
        self.stored_requests: defaultdict[str, int] = defaultdict(int)
        self.enable_kv_event = enable_kv_event

        # Pause store requests when CPU/disk offloading is under pressure.
        self._store_pressure_active = False
        self._skip_store_requests: set[str] = set()

    def add_stored_request(self, req_id: str):
        with self.done_task_lock:
            self.stored_requests[req_id] += 1

    def dec_stored_request(self, req_id: str):
        with self.done_task_lock:
            if req_id in self.stored_requests:
                self.stored_requests[req_id] -= 1

    def delete_finished_stored_request(self, req_id: str):
        with self.done_task_lock:
            if req_id in self.stored_requests:
                del self.stored_requests[req_id]
            self._skip_store_requests.discard(req_id)

    def _should_skip_request(self, req_id: str) -> bool:
        with self.done_task_lock:
            return self._store_pressure_active and req_id in self._skip_store_requests

    def _mark_request_skipped_for_pressure(self, req_id: str) -> bool:
        with self.done_task_lock:
            already_skipped = req_id in self._skip_store_requests
            self._store_pressure_active = True
            self._skip_store_requests.add(req_id)
        return already_skipped

    def _clear_store_pressure(self) -> bool:
        with self.done_task_lock:
            if not self._store_pressure_active and not self._skip_store_requests:
                return False
            self._store_pressure_active = False
            self._skip_store_requests.clear()
        return True

    def _handle_request(self, req_meta: ReqMeta):
        token_len = req_meta.token_len_chunk
        block_ids = req_meta.block_ids
        req_id = req_meta.req_id
        current_event = req_meta.current_event

        if req_id not in self.stored_requests:
            self.request_queue.task_done()
            return
        if self._should_skip_request(req_id):
            logger.debug(
                "Skipping Mooncake store for request %s while CPU/disk offloading "
                "is under pressure",
                req_id,
            )
            self.dec_stored_request(req_id)
            self.request_queue.task_done()
            return

        # HMA: assert per-group block_ids matches the registered group count.
        num_groups = len(self.token_database.groups) or 1
        if isinstance(block_ids, list) and block_ids and isinstance(block_ids[0], list):
            req_groups = len(block_ids)
        else:
            # Legacy flat list — treat as single group.
            req_groups = 1
        if req_groups != num_groups:
            logger.error(
                "req %s: KV group count mismatch on save: got %d, expected %d. "
                "Dropping save.",
                req_id,
                req_groups,
                num_groups,
            )
            self.dec_stored_request(req_id)
            self.request_queue.task_done()
            return

        starts: list[int] = []
        ends: list[int] = []
        groups_per_key: list[int] = []
        keys: list[str] = []
        block_hashes: list[BlockHash] = []
        # Per-group chunk counts (= cdiv(token_len, g_block_size)).
        # `len(req_meta.block_hashes)` is at hash_block_size granularity,
        # not per-group, so it can't be used here.
        group_block_sizes = self.token_database.group_block_sizes or [
            self.block_size
        ] * max(1, num_groups)
        g_total_chunks = [cdiv(token_len, gbs) for gbs in group_block_sizes]
        hash_bs = max(1, self.token_database.hash_block_size)
        per_group_savable = [0] * max(1, num_groups)
        for start, end, group_id, key in self.token_database.process_tokens(
            token_len, req_meta.block_hashes
        ):
            g_block_size = group_block_sizes[group_id]
            chunk_id = start // g_block_size
            if not self.token_database.is_chunk_in_window_per_request(
                chunk_id, block_ids, group_id, g_total_chunks[group_id]
            ):
                continue
            starts.append(start)
            ends.append(end)
            groups_per_key.append(group_id)
            keys.append(key.to_string())
            # Right-edge hash; same index `process_tokens` used for the key.
            block_hashes.append(req_meta.block_hashes[end // hash_bs - 1])
            per_group_savable[group_id] += 1

        self._save_debug_counter = getattr(self, "_save_debug_counter", 0) + 1
        if self._save_debug_counter % 50 == 1:
            logger.info(
                "[mooncake-save] call=%d savable=%d per_group=%s "
                "blocks_per_sw=%s g_total_chunks=%s g_block_sizes=%s",
                self._save_debug_counter,
                len(keys),
                per_group_savable,
                self.token_database.blocks_per_sw,
                g_total_chunks,
                group_block_sizes,
            )

        # Apply put_step striding for TP
        starts = starts[self.tp_rank % self.put_step :: self.put_step]
        ends = ends[self.tp_rank % self.put_step :: self.put_step]
        groups_per_key = groups_per_key[self.tp_rank % self.put_step :: self.put_step]
        keys = keys[self.tp_rank % self.put_step :: self.put_step]
        block_hashes = block_hashes[self.tp_rank % self.put_step :: self.put_step]

        if not keys:
            self.dec_stored_request(req_id)
            return

        # Check which blocks already exist (dedup)
        save_exists_start = time.perf_counter()
        try:
            exists_states = self.store.batch_is_exist(keys)
        except Exception:
            self._record_operation(
                "save_exists",
                save_exists_start,
                len(keys),
                status="error",
                num_failed_keys=len(keys),
            )
            raise
        self._record_operation(
            "save_exists",
            save_exists_start,
            len(keys),
        )
        missing_indices = [i for i, exists in enumerate(exists_states) if exists != 1]

        if not missing_indices:
            self.dec_stored_request(req_id)
            return

        starts = [starts[i] for i in missing_indices]
        ends = [ends[i] for i in missing_indices]
        groups_per_key = [groups_per_key[i] for i in missing_indices]
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

        addrs = []
        sizes = []
        stored_events: list[BlockStored] = []
        prev_key = None
        new_block_hashes = [maybe_convert_block_hash(bh) for bh in block_hashes]

        for index, start in enumerate(starts):
            gid = groups_per_key[index]
            addr, size, _ = self.token_database.prepare_value(
                start,
                ends[index],
                block_ids,
                gid,
                g_total_chunks[gid],
            )
            addrs.append(addr)
            sizes.append(size)

            if self.enable_kv_event:
                token_ids = (
                    req_meta.token_ids[start : ends[index]]
                    if req_meta.token_ids is not None
                    else None
                )
                stored_event = BlockStored(
                    block_hashes=[new_block_hashes[index]],
                    parent_block_hash=prev_key,
                    token_ids=token_ids,
                    block_size=req_meta.original_block_size,
                    lora_id=None,
                    medium="cpu",
                    lora_name=None,
                )
                stored_events.append(stored_event)
                prev_key = new_block_hashes[index]

        if current_event is not None:
            current_event.synchronize()

        batch_bytes = _sum_batch_bytes(sizes)
        put_start = time.perf_counter()
        try:
            res = self.store.batch_put_from_multi_buffers(keys, addrs, sizes)
            failed = [i for i, v in enumerate(res) if v < 0]
            self._record_operation(
                "save_put",
                put_start,
                len(keys),
                num_bytes=batch_bytes,
                status="partial_failure" if failed else "ok",
                num_failed_keys=len(failed),
            )
            if failed:
                failed_codes = set(res[i] for i in failed)
                # Log a sample (first failed key's first segment addr+size).
                # Cross-correlating addr against the registered ranges in the
                # startup "Registering KV_Caches" log is the fastest way to
                # spot the "RDMA address out of registered range" failure
                # mode (e.g., heterogeneous strides bug from DP=2 EP).
                fi = failed[0]
                first_addr = "N/A"
                first_size = "N/A"
                if fi < len(addrs):
                    a = addrs[fi]
                    s = sizes[fi]
                    if isinstance(a, list) and a:
                        first_addr = f"0x{a[0]:x}+{s[0] if isinstance(s, list) else s}"
                        first_size = f"n_segs={len(a)}"
                    elif isinstance(a, int):
                        first_addr = f"0x{a:x}"
                        first_size = str(s)
                logger.warning(
                    "batch_put failed: %d/%d keys failed "
                    "(codes=%s, batch_bytes=%d, num_keys=%d), "
                    "first_key=%s, first_failed_addr=%s, %s",
                    len(failed),
                    len(keys),
                    failed_codes,
                    batch_bytes,
                    len(keys),
                    keys[0] if keys else "N/A",
                    first_addr,
                    first_size,
                )
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
                logger.info(
                    "Mooncake CPU/offload pressure cleared after a "
                    "successful store batch"
                )
        except Exception as e:
            self._record_operation(
                "save_put",
                put_start,
                len(keys),
                num_bytes=batch_bytes,
                status="error",
                num_failed_keys=len(keys),
            )
            logger.error("Failed to put key %s, error: %s", keys, e)

        if self.enable_kv_event and stored_events:
            self.update_kv_event(stored_events)

        self.dec_stored_request(req_id)
        self.request_queue.task_done()


class KVCacheStoreRecvingThread(KVTransferThread):
    """Background thread for loading KV cache blocks from the store."""

    def __init__(
        self,
        store: Any,
        token_database: ChunkedTokenDatabase,
        block_size: int,
        tp_rank: int,
        ready_event: threading.Event,
        disk_offload_buffer_budget_bytes: int | None = None,
        record_operation: Callable[..., None] | None = None,
    ):
        super().__init__(
            store,
            token_database,
            block_size,
            tp_rank,
            ready_event,
            name="KVCacheStoreRecvingThread",
            record_operation=record_operation,
        )
        self.disk_offload_buffer_budget_bytes = disk_offload_buffer_budget_bytes
        self.usable_disk_offload_buffer_budget_bytes = (
            None
            if disk_offload_buffer_budget_bytes is None
            else _get_usable_disk_offload_buffer_budget_bytes(
                disk_offload_buffer_budget_bytes
            )
        )

    def _handle_request(self, req_meta: ReqMeta):
        token_len = req_meta.load_spec.token_len  # type: ignore[union-attr]
        req_id = req_meta.req_id
        # Mask aligns to scheduler frame (LCM); self.block_size is MIN
        # on HMA and would skip at hash-block granularity.
        sched_bs = getattr(self.token_database, "scheduler_block_size", self.block_size)
        mask_num = (
            req_meta.load_spec.vllm_cached_tokens  # type: ignore[union-attr]
            // sched_bs
            * sched_bs
        )

        # HMA: assert per-group block_ids matches the registered group count.
        num_groups = len(self.token_database.groups) or 1
        block_ids = req_meta.block_ids
        if isinstance(block_ids, list) and block_ids and isinstance(block_ids[0], list):
            req_groups = len(block_ids)
        else:
            req_groups = 1
        if req_groups != num_groups:
            logger.error(
                "req %s: KV group count mismatch on load: got %d, expected %d. "
                "Dropping load.",
                req_id,
                req_groups,
                num_groups,
            )
            return

        # Per-group load: skip (chunk, group) pairs outside the group's window.
        addr_list = []
        size_list = []
        key_list = []
        # Per-group native chunk counts; symmetric with the save path.
        load_group_block_sizes = self.token_database.group_block_sizes or [
            self.block_size
        ] * max(1, num_groups)
        load_g_total_chunks = [cdiv(token_len, gbs) for gbs in load_group_block_sizes]
        for start, end, group_id, key in self.token_database.process_tokens(
            token_len, req_meta.block_hashes, mask_num
        ):
            g_block_size = load_group_block_sizes[group_id]
            chunk_id = start // g_block_size
            if not self.token_database.is_chunk_in_window_per_request(
                chunk_id, block_ids, group_id, load_g_total_chunks[group_id]
            ):
                continue
            addr, size, _ = self.token_database.prepare_value(
                start,
                end,
                block_ids,
                group_id,
                load_g_total_chunks[group_id],
            )
            key_list.append(key.to_string())
            addr_list.append(addr)
            size_list.append(size)

        if not key_list:
            self.set_finished_request(req_id)
            self.request_queue.task_done()
            return

        # Rotate lists by tp_rank for load balancing
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

        load_batches = [(key_list_c, addr_list_c, size_list_c)]
        if self.usable_disk_offload_buffer_budget_bytes is not None:
            total_staging_bytes = sum(
                _estimate_disk_offload_staging_bytes(size) for size in size_list_c
            )
            usable_batch_keys = _get_usable_disk_offload_batch_key_count(
                len(key_list_c)
            )
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

        current_batch_keys: list[str] = key_list_c
        batch_bytes = 0
        load_get_start = time.perf_counter()
        try:
            for batch_keys, batch_addrs, batch_sizes in load_batches:
                current_batch_keys = batch_keys
                batch_bytes = 0
                load_get_start = time.perf_counter()
                batch_bytes = _sum_batch_bytes(batch_sizes)
                res = self.store.batch_get_into_multi_buffers(
                    batch_keys, batch_addrs, batch_sizes
                )
                failed = [
                    (key, value)
                    for key, value in zip(batch_keys, res, strict=True)
                    if value < 0
                ]
                self._record_operation(
                    "load_get",
                    load_get_start,
                    len(batch_keys),
                    num_bytes=batch_bytes,
                    status="partial_failure" if failed else "ok",
                    num_failed_keys=len(failed),
                )
                if failed:
                    logger.warning(
                        "Failed to get %d Mooncake keys from sub-batch "
                        "(batch_keys=%d, first_failures=%s)",
                        len(failed),
                        len(batch_keys),
                        failed[:3],
                    )
                    break
        except Exception as e:
            self._record_operation(
                "load_get",
                load_get_start,
                len(current_batch_keys),
                num_bytes=batch_bytes,
                status="error",
                num_failed_keys=len(current_batch_keys),
            )
            logger.warning(
                "Failed to get Mooncake sub-batch %s, error: %s",
                current_batch_keys[:3],
                e,
            )

        self.set_finished_request(req_id)
        self.request_queue.task_done()


# ============================================================
# Store Worker
# ============================================================


class MooncakeStoreWorker:
    """Worker-side component for MooncakeStoreConnector."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: "KVCacheConfig | None" = None,
    ):
        try:
            from mooncake.store import MooncakeDistributedStore  # type: ignore
        except ImportError as e:
            raise ImportError(
                "Please install mooncake by following the instructions at "
                "https://github.com/kvcache-ai/Mooncake/blob/main/doc/"
                "en/build.md to run vLLM with MooncakeStoreConnector."
            ) from e

        self.kv_cache_config = kv_cache_config
        model_config = vllm_config.model_config
        parallel_config = vllm_config.parallel_config

        self.dp_rank = get_mooncake_dp_engine_index(parallel_config)
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.pp_size = parallel_config.pipeline_parallel_size
        self.pp_rank = (parallel_config.rank // self.tp_size) % self.pp_size

        self.pcp_size = get_pcp_group().world_size
        self.pcp_rank = get_pcp_group().rank_in_group if self.pcp_size > 1 else 0
        self.dcp_size = get_dcp_group().world_size
        self.dcp_rank = get_dcp_group().rank_in_group if self.dcp_size > 1 else 0

        self.kv_role = vllm_config.kv_transfer_config.kv_role
        # NOTE(yifan): enforce load_async for now for better compute-I/O overlap.
        self.load_async = True
        self.cache_config = vllm_config.cache_config
        self.original_block_size = self.cache_config.block_size
        self.block_size = self.cache_config.block_size
        if self.pcp_size > 1:
            self.block_size *= self.pcp_size
        if self.dcp_size > 1:
            self.block_size *= self.dcp_size
        self.num_layers = model_config.get_num_layers(parallel_config)

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
            self.num_kv_head = model_config.get_total_num_kv_heads()

        if self.num_kv_head < self.tp_size:
            self.put_step = self.tp_size // self.num_kv_head
            self.head_or_tp_rank = self.tp_rank // self.put_step
        else:
            self.head_or_tp_rank = self.tp_rank
            self.put_step = 1

        self.metadata = KeyMetadata(
            model_name=model_config.model.rstrip("/").split("/")[-1],
            tp_rank=self.head_or_tp_rank,
            pcp_rank=self.pcp_rank,
            dcp_rank=self.dcp_rank,
            pp_rank=self.pp_rank,
        )

        self.token_database = ChunkedTokenDatabase(self.metadata, self.block_size)

        # Initialize MooncakeDistributedStore with its own TransferEngine
        store_config = MooncakeStoreConfig.load_from_env()
        self.store = MooncakeDistributedStore()

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

        self.disk_offload_buffer_budget_bytes = _get_disk_offload_buffer_budget_bytes(
            store_config.enable_offload
        )

        kv_event_config = vllm_config.kv_events_config
        self.enable_kv_events = False
        if kv_event_config and kv_event_config.enable_kv_cache_events:
            self.enable_kv_events = True

        self.kv_send_thread: KVCacheStoreSendingThread | None = None
        self.kv_recv_thread: KVCacheStoreRecvingThread | None = None
        self.finished_store_req: set[str] = set()
        self._kv_connector_stats_lock = threading.Lock()
        self.kv_connector_stats = MooncakeStoreConnectorStats()

    def register_cross_layers_kv_caches(self, kv_cache: torch.Tensor) -> None:
        """Register a cross-layers KV cache tensor.

        Wraps the unified tensor in a single-entry dict so that the
        existing stride-based logic in register_kv_caches() produces
        the correct single-segment result (block_len = page_size * num_layers).
        """
        self.register_kv_caches({"__cross_layer__": kv_cache})

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """Register KV cache tensors and start transfer threads.

        Buckets layers by kv_cache_config.kv_cache_groups so HMA models
        (mixed FullAttention + SlidingWindow) get one GroupLayout per
        group and prepare_value can resolve per-group block ids.
        """
        first_kv_cache = next(iter(kv_caches.values()))

        # num_blocks from cache_config is authoritative (set after
        # profiling, before KV cache allocation).
        assert self.cache_config.num_gpu_blocks is not None
        self.num_blocks = self.cache_config.num_gpu_blocks

        # Detect the KV cache memory layout using the stride-based
        # approach from simple_kv_offload/worker.py. Same logic as
        # before, but now applied per group.
        #
        # The physical layout varies across attention backends:
        #   FlashAttn/ROCm : (2, num_blocks, ...) → K/V outermost
        #   FlashInfer/MLA : (num_blocks, ...)    → blocks outermost
        # Layout detection now happens per-tensor inside the group loop
        # below (cache_outer_dims). The early `outer_dims` was dead code.

        # Bucket layers by KV cache group. If kv_cache_config is missing
        # (legacy / tests / no-arg construction), fall back to a single
        # group containing everything — preserves N=1 behavior for
        # DeepSeek/MLA.
        kv_cache_config = getattr(self, "kv_cache_config", None)
        is_cross_layer_registration = set(kv_caches) == {"__cross_layer__"}
        if kv_cache_config is not None:
            if is_cross_layer_registration:
                if len(kv_cache_config.kv_cache_groups) != 1:
                    raise ValueError(
                        "Cross-layer KV cache registration is only supported for "
                        "single-group KV cache configs."
                    )
                layer_to_group_idx: dict[str, int] = {"__cross_layer__": 0}
                num_groups = 1
            else:
                layer_to_group_idx = {}
                for g_idx, group in enumerate(kv_cache_config.kv_cache_groups):
                    for layer_name in group.layer_names:
                        layer_to_group_idx[layer_name] = g_idx
                num_groups = len(kv_cache_config.kv_cache_groups)
        else:
            layer_to_group_idx = {name: 0 for name in kv_caches}
            num_groups = 1

        # Bucket layers by group. Fail closed on unknown layer names when
        # `kv_cache_config` is present: silently defaulting to group 0
        # would misattribute window/block_size/block_ids on HMA.
        grouped_caches: list[list[tuple[str, torch.Tensor]]] = [
            [] for _ in range(num_groups)
        ]
        for name, cache in kv_caches.items():
            if kv_cache_config is not None and name not in layer_to_group_idx:
                raise ValueError(
                    f"KV cache layer '{name}' not in any kv_cache_group. "
                    f"Known groups: "
                    f"{[g.layer_names for g in kv_cache_config.kv_cache_groups]}"
                )
            g_idx = layer_to_group_idx.get(name, 0)
            grouped_caches[g_idx].append((name, cache))

        seen_ptrs: set[int] = set()
        groups: list[GroupLayout] = []
        flat_layer_to_group: list[int] = []

        for g_idx, layers in enumerate(grouped_caches):
            group_base_addrs: list[int] = []
            group_block_lens: list[int] = []
            for _, cache in layers:
                cache_storage = cache.untyped_storage()
                base_addr = cache_storage.data_ptr()
                region_len = cache_storage.nbytes()

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

                # Compute outer_dims and page_size_bytes PER TENSOR. HMA mixes
                # layer types (FA, SWA, indexer, compressor, ...) with
                # different shapes/strides; using globals from first_kv_cache
                # produces RDMA put addresses outside the registered range
                # for any tensor whose per-block stride differs from the
                # first one.
                cache_el = cache.element_size()
                cache_page_size_bytes = region_len // self.num_blocks
                cache_outer_dims = [
                    d
                    for d in range(cache.ndim)
                    if cache.stride(d) * cache_el > cache_page_size_bytes
                ]

                if not cache_outer_dims:
                    # Blocks-first layout (FlashInfer / MLA): one segment.
                    group_base_addrs.append(base_addr)
                    group_block_lens.append(cache_page_size_bytes)
                    flat_layer_to_group.append(g_idx)
                else:
                    # K/V-first layout (FlashAttn / ROCm): split segments.
                    seg_stride = cache.stride(cache_outer_dims[0]) * cache_el
                    for seg in range(cache.shape[cache_outer_dims[0]]):
                        group_base_addrs.append(base_addr + seg * seg_stride)
                        group_block_lens.append(seg_stride // self.num_blocks)
                        flat_layer_to_group.append(g_idx)
            groups.append(
                GroupLayout(
                    base_addrs=group_base_addrs,
                    block_lens=group_block_lens,
                )
            )

        self.token_database.set_groups(groups, flat_layer_to_group)
        self.kv_caches_base_addr = self.token_database.kv_caches_base_addr
        self.block_len = self.token_database.block_len

        # Per-group SWA window in blocks (0 = FA / no clip) and per-group
        # native block_size. HMA wraps per-layer specs in
        # `UniformTypeKVCacheSpecs` which hides `sliding_window` /
        # `attention_chunk_size` at the top level; drill into
        # `spec.kv_cache_specs` to recover it.
        if kv_cache_config is not None:
            blocks_per_sw: list[int] = []
            group_block_sizes: list[int] = []
            for group in kv_cache_config.kv_cache_groups:
                spec = group.kv_cache_spec
                inner_specs = getattr(spec, "kv_cache_specs", None)
                probe_spec = next(iter(inner_specs.values())) if inner_specs else spec
                g_block_size = getattr(spec, "block_size", None) or self.block_size
                group_block_sizes.append(g_block_size)
                window_tokens = getattr(probe_spec, "sliding_window", None) or getattr(
                    probe_spec, "attention_chunk_size", None
                )
                if window_tokens:
                    blocks_per_sw.append(cdiv(window_tokens, g_block_size) + 1)
                else:
                    blocks_per_sw.append(0)
        else:
            blocks_per_sw = [0] * num_groups
            group_block_sizes = [self.block_size] * num_groups
        self.token_database.set_blocks_per_sw(blocks_per_sw)
        # block_hashes granularity. Honor cache_config.hash_block_size if
        # set (mirroring vllm's resolver), else GCD(group_block_sizes).
        hash_block_size_attr = getattr(self.cache_config, "hash_block_size", None)
        if isinstance(hash_block_size_attr, int) and hash_block_size_attr > 0:
            hash_block_size = hash_block_size_attr
        elif group_block_sizes:
            hash_block_size = reduce(gcd, group_block_sizes)
        else:
            hash_block_size = self.block_size
        # Scheduler-frame alignment = LCM(group_block_sizes), then scale
        # by pcp/dcp. cache_config.block_size is mutated to MIN for HMA,
        # so don't read from it.
        if group_block_sizes:
            self.scheduler_block_size = reduce(lcm, group_block_sizes)
        else:
            self.scheduler_block_size = self.block_size
        pcp = getattr(self, "pcp_size", 1)
        dcp = getattr(self, "dcp_size", 1)
        if pcp > 1:
            self.scheduler_block_size *= pcp
        if dcp > 1:
            self.scheduler_block_size *= dcp
        self.token_database.set_group_block_sizes(
            group_block_sizes, hash_block_size, self.scheduler_block_size
        )
        self._blocks_per_sw = blocks_per_sw
        logger.info(
            "Per-group blocks_per_sw=%s group_block_sizes=%s "
            "hash_block_size=%d scheduler_block_size=%d "
            "(self.block_size=%d)",
            blocks_per_sw,
            group_block_sizes,
            hash_block_size,
            self.scheduler_block_size,
            self.block_size,
        )

        # Heterogeneous per-tensor block_lens are now handled correctly,
        # but logging them up front catches future regressions of the
        # DP=2 EP "first-tensor page_size for all" failure.
        unique_block_lens = sorted(set(self.block_len))
        if len(unique_block_lens) > 1:
            from collections import Counter

            dist = Counter(self.block_len)
            logger.info(
                "Mooncake KV layout has heterogeneous per-tensor block_lens %s "
                "(distribution=%s); per-tensor stride accounting is in effect.",
                unique_block_lens,
                dict(dist),
            )

        logger.info(
            "Registering KV_Caches. use_mla: %s, num_groups: %d, "
            "shape %s, num_blocks: %d, block_len: %s, "
            "per_key_bytes: %d, num_segments: %d",
            self.use_mla,
            num_groups,
            first_kv_cache.shape,
            self.num_blocks,
            list(set(self.block_len)),
            sum(self.block_len),
            len(self.kv_caches_base_addr),
        )

        # Start transfer threads
        if self.kv_role in ["kv_producer", "kv_both"]:
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
                self._record_kv_connector_operation,
            )
            self.kv_send_thread.start()

        ready_event_recving = threading.Event()
        self.kv_recv_thread = KVCacheStoreRecvingThread(
            self.store,
            self.token_database,
            self.block_size,
            self.tp_rank,
            ready_event_recving,
            disk_offload_buffer_budget_bytes=self.disk_offload_buffer_budget_bytes,
            record_operation=self._record_kv_connector_operation,
        )
        self.kv_recv_thread.start()
        ready_event_recving.wait()

    def start_load_kv(
        self,
        metadata: MooncakeStoreConnectorMetadata,
    ):
        """No-op: loads are issued in get_finished() for overlap."""
        pass

    def wait_for_save(
        self,
        metadata: MooncakeStoreConnectorMetadata,
    ):
        """No-op: stores are issued in get_finished() for overlap."""
        pass

    def get_finished(
        self,
        finished_req_ids: set[str],
        meta: MooncakeStoreConnectorMetadata,
    ) -> tuple[set[str], set[str]]:
        """Issue all I/O and get completed send/recv request IDs.

        All load and store I/O requests are issued here (after model
        compute is launched on the compute stream) for better
        compute-I/O overlap.
        """
        # Issue async loads
        for request in meta.requests:
            load_spec = request.load_spec
            if load_spec is None or not load_spec.can_load:
                continue

            token_len = request.token_len_chunk
            if (load_spec.kvpool_cached_tokens % self.block_size != 0) and (
                load_spec.kvpool_cached_tokens == token_len - 1
            ):
                token_len = load_spec.kvpool_cached_tokens + 1
            else:
                token_len = load_spec.kvpool_cached_tokens
            load_spec.token_len = token_len

            assert self.kv_recv_thread is not None
            self.kv_recv_thread.add_request(request)

        assert self.load_async, "load_async must be True for better performance."
        # Issue stores with CUDA event synchronization
        if self.kv_role in ["kv_producer", "kv_both"]:
            current_event = None
            for request in meta.requests:
                if request.can_save:
                    current_event = torch.cuda.Event()
                    current_event.record()
                    break

            for request in meta.requests:
                if not request.can_save:
                    continue
                request.current_event = current_event
                assert self.kv_send_thread is not None
                self.kv_send_thread.add_stored_request(request.req_id)
                self.kv_send_thread.add_request(request)

        # Check completion of previously queued transfers
        done_sending = (
            self._get_and_clear_finished_sending(finished_req_ids, meta)
            if self.kv_role in ["kv_producer", "kv_both"]
            else set()
        )

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

    def _record_kv_connector_operation(
        self,
        operation: str,
        duration_seconds: float,
        num_keys: int,
        *,
        num_bytes: int = 0,
        status: str = "ok",
        num_failed_keys: int = 0,
    ) -> None:
        with self._kv_connector_stats_lock:
            self.kv_connector_stats.record_operation(
                operation=operation,
                duration_seconds=duration_seconds,
                num_keys=num_keys,
                num_bytes=num_bytes,
                status=status,
                num_failed_keys=num_failed_keys,
            )

    def get_kv_connector_stats(self) -> MooncakeStoreConnectorStats | None:
        with self._kv_connector_stats_lock:
            if self.kv_connector_stats.is_empty():
                return None
            kv_connector_stats = self.kv_connector_stats
            self.kv_connector_stats = MooncakeStoreConnectorStats()
            return kv_connector_stats

    def _get_and_clear_finished_sending(
        self,
        finished_req_ids: set[str],
        meta: MooncakeStoreConnectorMetadata,
    ) -> set[str]:
        assert self.kv_send_thread is not None
        finished_sending: set[str] = set()

        for req_id in meta.preempted_req_ids:
            self.kv_send_thread.delete_finished_stored_request(req_id)

        for req_id in self.kv_send_thread.stored_requests.copy():
            if (
                self.kv_send_thread.stored_requests[req_id] == 0
                and req_id in self.finished_store_req
            ):
                self.finished_store_req.remove(req_id)
                finished_sending.add(req_id)
                self.kv_send_thread.delete_finished_stored_request(req_id)

        for req_id in finished_req_ids:
            req_remain_jobs = self.kv_send_thread.stored_requests.get(req_id)
            if req_remain_jobs == 0:
                finished_sending.add(req_id)
                self.kv_send_thread.delete_finished_stored_request(req_id)
            elif req_remain_jobs is not None:
                self.finished_store_req.add(req_id)

        return finished_sending

    def lookup(
        self,
        token_len: int,
        block_hashes: list[BlockHash],
    ) -> int:
        """Largest contiguous prefix (in tokens) that all groups + ranks supply.

        Per-(chunk, group) keys; a chunk hits iff every queried group
        returned 1. Single batch_is_exist round-trip.
        """
        keys: list[str] = []
        multi_tp_keys: list[str] = []
        lookup_start = time.perf_counter()
        try:
            hash_bs = max(1, self.token_database.hash_block_size)
            lookup_token_len = token_len // hash_bs * hash_bs
            if lookup_token_len <= 0:
                return 0

            # Per-group chunk counts. Each group's chunk_id indexes a
            # different token frame, so we keep (chunk_id, group_id) per
            # key and convert misses back to token positions before
            # aggregating.
            group_block_sizes = self.token_database.group_block_sizes or [
                self.block_size
            ] * max(1, len(self.token_database.groups) or 1)
            g_total_chunks_lk = [
                cdiv(lookup_token_len, gbs) for gbs in group_block_sizes
            ]
            chunk_groups: list[tuple[int, int]] = []
            for start, _end, group_id, key in self.token_database.process_tokens(
                lookup_token_len, block_hashes
            ):
                g_block = group_block_sizes[group_id]
                chunk_id = start // g_block
                if not self.token_database.is_chunk_in_window(
                    chunk_id, g_total_chunks_lk[group_id], group_id
                ):
                    continue
                keys.append(key.to_string())
                chunk_groups.append((chunk_id, group_id))

            if not keys:
                return 0

            # Expand keys for all TP ranks
            multi_tp_keys = keys[:]
            for i in range(1, min(self.tp_size, self.num_kv_head)):
                for item in keys:
                    new_str = item.replace("@tp_rank:0", f"@tp_rank:{i}", 1)
                    multi_tp_keys.append(new_str)

            # Expand keys for all PP ranks
            pp_base_keys = multi_tp_keys.copy()
            for i in range(1, self.pp_size):
                for item in pp_base_keys:
                    new_str = item.replace("@pp_rank:0", f"@pp_rank:{i}", 1)
                    multi_tp_keys.append(new_str)

            res = self.store.batch_is_exist(multi_tp_keys)
            self._record_kv_connector_operation(
                "lookup_exists",
                time.perf_counter() - lookup_start,
                len(multi_tp_keys),
            )

            n = len(keys)
            num_rows = min(self.tp_size, self.num_kv_head) * self.pp_size
            multi_tp_values = [res[i * n : (i + 1) * n] for i in range(num_rows)]

            # First-miss in token space: each missed (chunk_id, group)
            # maps to `cid * g_block_size`; the min across rows caps the
            # available prefix.
            def first_miss_token(row: list[int]) -> int:
                miss = lookup_token_len
                for (cid, gid), v in zip(chunk_groups, row):
                    if v != 1:
                        miss = min(miss, cid * group_block_sizes[gid])
                return miss

            per_row_first_miss = [first_miss_token(r) for r in multi_tp_values]
            min_token = min(per_row_first_miss) if per_row_first_miss else 0

            self._lookup_debug_counter = getattr(self, "_lookup_debug_counter", 0) + 1
            if self._lookup_debug_counter % 50 == 1:
                ones_per_row = [
                    sum(1 for v in row if v == 1) for row in multi_tp_values
                ]
                logger.info(
                    "[mooncake-lookup] call=%d num_keys=%d ones=%s "
                    "first_miss_token=%s min_first_miss_token=%d",
                    self._lookup_debug_counter,
                    len(multi_tp_keys),
                    ones_per_row,
                    per_row_first_miss,
                    min_token,
                )

            # Round to scheduler block boundary (LCM); self.block_size
            # is MIN on HMA, see `register_kv_caches`.
            sched_bs = getattr(self, "scheduler_block_size", self.block_size)
            if min_token >= lookup_token_len:
                return lookup_token_len
            return (min_token // sched_bs) * sched_bs
        except Exception as e:
            self._record_kv_connector_operation(
                "lookup_exists",
                time.perf_counter() - lookup_start,
                len(multi_tp_keys),
                status="error",
                num_failed_keys=len(multi_tp_keys),
            )
            logger.error("Remote connection failed in lookup: %s", e)
            return 0

    def get_kv_events(self) -> list[BlockStored]:
        if self.enable_kv_events and self.kv_send_thread is not None:
            return self.kv_send_thread.get_kv_events()
        return []


# ============================================================
# Lookup Key Server
# ============================================================


class LookupKeyServer:
    """ZMQ server on worker rank 0 for handling prefix lookup queries."""

    def __init__(
        self,
        store_worker: MooncakeStoreWorker,
        vllm_config: VllmConfig,
    ):
        self.decoder = MsgpackDecoder()
        self.ctx = zmq.Context()  # type: ignore[attr-defined]
        socket_path = get_zmq_rpc_path_lookup(vllm_config)
        self._ipc_path = socket_path.removeprefix("ipc://")
        if os.path.exists(self._ipc_path):
            os.unlink(self._ipc_path)
        self.socket = make_zmq_socket(
            self.ctx,
            socket_path,
            zmq.REP,  # type: ignore[attr-defined]
            bind=True,
        )

        self.store_worker = store_worker
        self.running = True

        def process_request():
            while self.running:
                all_frames = self.socket.recv_multipart(copy=False)
                token_len = int.from_bytes(all_frames[0], byteorder="big")
                hash_frames = all_frames[1:]
                hashes_str = self.decoder.decode(hash_frames)
                result = self.store_worker.lookup(token_len, hashes_str)
                response = result.to_bytes(4, "big")
                self.socket.send(response)

        self.thread = threading.Thread(target=process_request, daemon=True)
        self.thread.start()

    def close(self):
        self.socket.close(linger=0)
        if os.path.exists(self._ipc_path):
            os.unlink(self._ipc_path)
