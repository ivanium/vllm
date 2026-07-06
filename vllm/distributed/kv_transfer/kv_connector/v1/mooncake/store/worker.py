# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# The transfer-thread scaffolding (KVTransferThread, KVCacheStoreSendingThread,
# KVCacheStoreRecvingThread) is adapted from vllm-project/vllm-ascend
# (vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/).
"""Worker-side logic for MooncakeStoreConnector.

Includes the store worker, transfer threads, the admin (reset) server, the
lookup subprocess, and MooncakeDistributedStore integration.
"""

import contextlib
import dataclasses
import json
import os
import queue
import socket
import threading
import time
from collections import OrderedDict, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypeVar

import regex as re
import torch
import zmq

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed import (
    get_dcp_group,
    get_pcp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.distributed.kv_events import BlockStored
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake import rdma_utils
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.coordinator import (  # noqa: E501
    ExternalCachedBlockPool,
    MooncakeStoreCoordinator,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (  # noqa: E501
    BlobBlockHashes,
    ChunkedTokenDatabase,
    KeyMetadata,
    MooncakeStoreConnectorMetadata,
    PoolKey,
    ReqMeta,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.protocol import (  # noqa: E501
    LOOKUP_MSG,
    RECORD_BP_MSG,
    RESET_MSG,
    RESP_ERR,
    RESP_OK,
)
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.utils.network_utils import get_ip, make_zmq_socket
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    maybe_convert_block_hash,
    resolve_kv_cache_block_sizes,
)
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec

from .metrics import MooncakeStoreConnectorStats

logger = init_logger(__name__)

DEFAULT_GLOBAL_SEGMENT_SIZE = 4 * 1024 * 1024 * 1024  # 4 GiB
DEFAULT_LOCAL_BUFFER_SIZE = 4 * 1024 * 1024 * 1024  # 4 GiB

MOONCAKE_NO_AVAILABLE_HANDLE = -200
_T = TypeVar("_T")


def _rotate_list(values: list[_T], offset: int) -> list[_T]:
    return values[offset:] + values[:offset]


# Mirrors FileStorageConfig::local_buffer_size in Mooncake C++.
DEFAULT_MOONCAKE_DISK_STAGING_BUFFER_BYTES = 1280 * 1024 * 1024

# Mirrors DirectIO alignment in Mooncake's AllocateBatch.
_DIRECT_IO_ALIGNMENT = 4096
_DIRECT_IO_PADDING_BYTES = 2 * _DIRECT_IO_ALIGNMENT
SESSION_BREAKPOINT_MAX_SESSIONS = 100_000


MooncakeMode = Literal["embedded", "standalone-store"]


@dataclass
class SessionBreakpoint:
    """Validated store boundary for a request session."""

    aligned_token_len: int
    boundary_block_hash: bytes


class SessionBreakpointTracker:
    """Per-session history of stored boundaries for the lookup fast path.

    Owned by the lookup subprocess (single-threaded REP loop, so no
    locking). Entries are hints, not truth: every fast-path result is
    re-verified against the store, so entries staled by eviction, a store
    reset, or a not-yet-landed save self-heal by falling back to the full
    scan.
    """

    def __init__(
        self,
        hash_block_size: int,
        lcm_block_size: int,
        history_size: int,
    ):
        self._hash_block_size = hash_block_size
        self._lcm_block_size = lcm_block_size
        self._history_size = history_size
        self._sessions: OrderedDict[str, list[SessionBreakpoint]] = OrderedDict()

    def record(
        self,
        session_id: str | None,
        token_len: int,
        block_hashes: Sequence[BlockHash],
    ) -> None:
        if not session_id or token_len <= 0 or self._hash_block_size <= 0:
            return
        aligned_token_len = token_len // self._lcm_block_size * self._lcm_block_size
        if aligned_token_len <= 0:
            return
        boundary_block_idx = aligned_token_len // self._hash_block_size - 1
        if boundary_block_idx < 0 or boundary_block_idx >= len(block_hashes):
            return
        session_breakpoint = SessionBreakpoint(
            aligned_token_len=aligned_token_len,
            boundary_block_hash=bytes(block_hashes[boundary_block_idx]),
        )
        breakpoints = self._sessions.pop(session_id, [])
        breakpoints = [
            existing
            for existing in breakpoints
            if not (
                existing.aligned_token_len == session_breakpoint.aligned_token_len
                and existing.boundary_block_hash
                == session_breakpoint.boundary_block_hash
            )
        ]
        breakpoints.insert(0, session_breakpoint)
        del breakpoints[self._history_size :]
        self._sessions[session_id] = breakpoints
        while len(self._sessions) > SESSION_BREAKPOINT_MAX_SESSIONS:
            self._sessions.popitem(last=False)

    def best_len(
        self,
        session_id: str | None,
        token_len: int,
        block_hashes: Sequence[BlockHash],
    ) -> int | None:
        """Longest recorded boundary whose anchor hash still matches."""
        if not session_id or token_len <= 0 or self._hash_block_size <= 0:
            return None
        breakpoints = self._sessions.get(session_id)
        if not breakpoints:
            return None
        self._sessions.move_to_end(session_id)

        best_breakpoint_len: int | None = None
        for session_breakpoint in breakpoints:
            if session_breakpoint.aligned_token_len > token_len:
                continue
            breakpoint_len = session_breakpoint.aligned_token_len
            boundary_block_idx = breakpoint_len // self._hash_block_size - 1
            if boundary_block_idx < 0 or boundary_block_idx >= len(block_hashes):
                continue
            if (
                bytes(block_hashes[boundary_block_idx])
                != session_breakpoint.boundary_block_hash
            ):
                continue
            if best_breakpoint_len is None or breakpoint_len > best_breakpoint_len:
                best_breakpoint_len = breakpoint_len
        return best_breakpoint_len


@dataclass
class MooncakeStoreConfig:
    """Configuration for MooncakeDistributedStore.

    ``mode`` selects the topology: ``embedded`` (each rank contributes
    ``global_segment_size`` in-process) or ``standalone-store`` (rank
    contributes 0; an external ``mooncake_client`` process owns the pool
    and the SSD tier).
    """

    metadata_server: str
    master_server_address: str
    protocol: str
    device_name: str
    mode: MooncakeMode = "embedded"
    global_segment_size: int = DEFAULT_GLOBAL_SEGMENT_SIZE
    local_buffer_size: int = DEFAULT_LOCAL_BUFFER_SIZE
    enable_offload: bool = False

    def __post_init__(self) -> None:
        if self.mode not in ("embedded", "standalone-store"):
            raise ValueError(f"unknown Mooncake mode: {self.mode!r}")
        if self.local_buffer_size <= 0:
            raise ValueError("local_buffer_size must be > 0")
        if self.mode == "embedded" and self.global_segment_size == 0:
            raise ValueError("embedded mode requires global_segment_size > 0")
        if self.mode == "standalone-store" and self.global_segment_size != 0:
            raise ValueError("standalone-store mode requires global_segment_size == 0")

    @staticmethod
    def from_file(file_path: str) -> "MooncakeStoreConfig":
        with open(file_path) as file:
            config = json.load(file)
        return MooncakeStoreConfig(
            metadata_server=config.get("metadata_server", ""),
            master_server_address=config.get("master_server_address", ""),
            protocol=config.get("protocol", "rdma"),
            device_name=config.get("device_name", ""),
            mode=config.get("mode", "embedded"),
            global_segment_size=_parse_size(
                config.get("global_segment_size", DEFAULT_GLOBAL_SEGMENT_SIZE)
            ),
            local_buffer_size=_parse_size(
                config.get("local_buffer_size", DEFAULT_LOCAL_BUFFER_SIZE)
            ),
            enable_offload=bool(config.get("enable_offload", False)),
        )

    @staticmethod
    def load_from_config() -> "MooncakeStoreConfig":
        config_path = os.getenv("MOONCAKE_CONFIG_PATH")
        if not config_path:
            raise ValueError(
                "The environment variable 'MOONCAKE_CONFIG_PATH' is not set."
            )
        return MooncakeStoreConfig.from_file(config_path)


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
    return max(1, int(raw_budget_bytes * envs.VLLM_MOONCAKE_DISK_STAGING_USABLE_RATIO))


def _split_disk_offload_load_batches(
    keys: list[str],
    addrs: list[list[int]],
    sizes: list[list[int]],
    usable_budget_bytes: int,
    raw_budget_bytes: int,
) -> tuple[list[tuple[list[str], list[list[int]], list[list[int]]]], str | None]:
    """Split a GET into sub-batches that fit the owner's staging buffer.

    ``addrs[i]`` / ``sizes[i]`` are scatter-gather lists (K/V or multi-layer
    segments) for key ``i``. ``usable_budget_bytes`` caps a multi-key batch;
    ``raw_budget_bytes`` is the hard per-key cap.

    Returns ``(batches, oversize_key)``. Aborts with ``([], key)`` if any
    single key exceeds ``raw_budget_bytes``; otherwise ``oversize_key`` is
    ``None``.
    """
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
        if batch_keys and batch_bytes + key_bytes > usable_budget_bytes:
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


def _call_replica_predicate(replica_desc: Any, method_name: str) -> bool:
    method = getattr(replica_desc, method_name, None)
    if method is None:
        return False
    try:
        return bool(method())
    except Exception:
        return False


def _classify_replica_tier(replica_descs: Any) -> str:
    if not replica_descs:
        return "unknown"
    try:
        replica_desc = replica_descs[0]
    except (IndexError, KeyError, TypeError):
        return "unknown"

    if _call_replica_predicate(replica_desc, "is_memory_replica"):
        return "memory"
    if _call_replica_predicate(
        replica_desc, "is_disk_replica"
    ) or _call_replica_predicate(replica_desc, "is_local_disk_replica"):
        return "disk"
    return "unknown"


def _get_replica_tiers_by_key(store: Any, keys: list[str]) -> dict[str, str]:
    tiers_by_key = {key: "unknown" for key in keys}
    try:
        replica_descs_by_key = store.batch_get_replica_desc(keys)
    except Exception as e:
        logger.warning(
            "Failed to get Mooncake replica descriptors for tier logging "
            "(batch_keys=%d, error=%s); marking tiers unknown",
            len(keys),
            e,
        )
        return tiers_by_key

    for key in keys:
        if hasattr(replica_descs_by_key, "get"):
            replica_descs = replica_descs_by_key.get(key)
        else:
            try:
                replica_descs = replica_descs_by_key[key]
            except (KeyError, TypeError):
                replica_descs = None
        tiers_by_key[key] = _classify_replica_tier(replica_descs)
    return tiers_by_key


def _log_mooncake_load_tier_summary(
    req_id: str,
    batch_keys: list[str],
    load_results: list[int],
    tiers_by_key: dict[str, str],
) -> None:
    tier_counts = {"memory": 0, "disk": 0, "unknown": 0}
    bytes_by_tier = {"memory": 0, "disk": 0, "unknown": 0}
    success_keys = 0
    failed_keys = 0

    for index, key in enumerate(batch_keys):
        tier = tiers_by_key.get(key, "unknown")
        if tier not in tier_counts:
            tier = "unknown"
        tier_counts[tier] += 1

        value = load_results[index] if index < len(load_results) else -1
        if value >= 0:
            success_keys += 1
            bytes_by_tier[tier] += int(value)
        else:
            failed_keys += 1

    logger.info(
        "Mooncake load tier summary: req_id=%s batch_keys=%d "
        "memory_keys=%d disk_keys=%d unknown_keys=%d "
        "success_keys=%d failed_keys=%d bytes_by_tier=%s",
        req_id,
        len(batch_keys),
        tier_counts["memory"],
        tier_counts["disk"],
        tier_counts["unknown"],
        success_keys,
        failed_keys,
        bytes_by_tier,
    )


# ============================================================
# Transfer Threads
# ============================================================


class KVTransferThread(threading.Thread):
    """Base class for async KV cache transfer threads."""

    def __init__(
        self,
        store: Any,
        token_databases: list[ChunkedTokenDatabase],
        block_size: int,
        tp_rank: int,
        ready_event: threading.Event,
        name: str,
        record_operation: Callable[..., None] | None = None,
        request_queue: queue.Queue[Any] | None = None,
    ):
        super().__init__(daemon=True, name=name)
        self.store = store
        self.ready_event = ready_event
        self.block_size = block_size
        self.tp_rank = tp_rank
        self.token_databases = token_databases
        self._record_operation_cb = record_operation
        self.done_task_lock = threading.Lock()
        self.request_queue: queue.Queue[Any] = request_queue or queue.Queue()
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
            request_data = None
            try:
                request_data = self.request_queue.get()
                if request_data is None:
                    logger.warning("Received a None request!")
                    self.request_queue.task_done()
                    continue
                self._handle_request(request_data)
            except Exception:
                req_id = getattr(request_data, "req_id", "<unknown>")
                logger.exception("Error in %s (req=%s)", self.name, req_id)

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
        coord: MooncakeStoreCoordinator,
        token_databases: list[ChunkedTokenDatabase],
        block_size: int,
        tp_rank: int,
        put_step: int,
        kv_role: str,
        ready_event: threading.Event,
        enable_kv_event: bool = False,
        replicate_config: Any = None,
        record_operation: Callable[..., None] | None = None,
    ):
        super().__init__(
            store,
            token_databases,
            block_size,
            tp_rank,
            ready_event,
            name="KVCacheStoreSendingThread",
            record_operation=record_operation,
        )
        self.put_step = put_step
        self.coord = coord
        self.kv_role = kv_role
        self.stored_requests: defaultdict[str, int] = defaultdict(int)
        self.enable_kv_event = enable_kv_event
        # Caller always passes a non-None ReplicateConfig — see
        # MooncakeStoreWorker.__init__ where store_replicate_config is built.
        self.replicate_config = replicate_config

        # Pause store requests when CPU/disk offloading is under pressure.
        self._store_pressure_active = False
        self._skip_store_requests: set[str] = set()

        # Per-request high-water mark of tokens actually persisted; the next
        # batch resumes here, so pressure-skipped or failed ranges are retried.
        self._saved_offset: dict[str, int] = {}

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
            self._saved_offset.pop(req_id, None)

    def _record_saved(self, req_id: str, token_len: int) -> None:
        # Guard on liveness so a concurrent finish/preempt pop isn't
        # recreated. max(): the event requeue can reorder batches, and a
        # stale lower batch must not roll back the high-water mark (that
        # would re-enqueue redundant exist checks for already-saved ranges).
        with self.done_task_lock:
            if req_id in self.stored_requests:
                self._saved_offset[req_id] = max(
                    self._saved_offset.get(req_id, 0), token_len
                )

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
        # Cache hits are always a multiple of ``lcm_block_size`` tokens, which
        # is also ``store_mask``'s precondition.
        lcm_block_size = self.coord.lcm_block_size
        token_len = req_meta.token_len_chunk // lcm_block_size * lcm_block_size
        block_ids_per_group = req_meta.block_ids
        req_id = req_meta.req_id
        current_event = req_meta.current_event

        if req_id not in self.stored_requests:
            self.request_queue.task_done()
            return

        # Decrement the in-flight counter and signal task_done() in `finally`
        # so the scheduler can release the GPU blocks it pinned for this
        # request (via `delay_free_blocks`) even when the store path raises.
        try:
            if token_len == 0:
                return

            if self._should_skip_request(req_id):
                logger.debug(
                    "Skipping Mooncake store for request %s while CPU/disk "
                    "offloading is under pressure",
                    req_id,
                )
                return

            # Resume from where this rank left off; only the new suffix is saved.
            save_start = self._saved_offset.get(req_id, 0)

            # Within each lcm region only per-spec relevant chunks are loaded
            # (e.g., SWA or linear attn), so mask out irrelevant chunks.
            # Clamp the boundary to this batch's end so every chunked-save
            # batch also persists the sparse-group proof blocks for its own
            # boundary: lookups can then hit up to the last landed batch even
            # while later batches (incl. the prompt-end one) are still queued
            # behind deep send backlogs. Without this, proof blocks exist only
            # in the final batch and short-gap replays get 0 hits under load.
            boundary_tokens = req_meta.num_prompt_tokens
            if boundary_tokens is not None:
                boundary_tokens = min(boundary_tokens, token_len)
            store_masks = self.coord.store_mask(
                token_len,
                save_start,
                num_prompt_tokens=boundary_tokens,
            )

            starts: list[int] = []
            ends: list[int] = []
            keys: list[str] = []
            kv_event_block_hashes: list[BlockHash] = []
            group_indices: list[int] = []
            for g_idx, db in enumerate(self.token_databases):
                # Rotate the stride phase per group to balance load across ranks.
                put_step_rank = (self.tp_rank + g_idx) % self.put_step
                for start, end, block_hash in db.process_tokens(
                    token_len,
                    req_meta.block_hashes,
                    mask_num=save_start,
                    chunk_mask=store_masks[g_idx],
                    put_step=self.put_step,
                    put_step_rank=put_step_rank,
                ):
                    starts.append(start)
                    ends.append(end)
                    keys.append(db.key_for(block_hash))
                    if self.enable_kv_event:
                        kv_event_block_hashes.append(block_hash)
                    group_indices.append(g_idx)

            if not keys:
                self._record_saved(req_id, token_len)
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
            missing_indices = [
                i for i, exists in enumerate(exists_states) if exists != 1
            ]

            if not missing_indices:
                self._record_saved(req_id, token_len)
                return

            if len(missing_indices) != len(keys):
                starts = [starts[i] for i in missing_indices]
                ends = [ends[i] for i in missing_indices]
                keys = [keys[i] for i in missing_indices]
                if self.enable_kv_event:
                    kv_event_block_hashes = [
                        kv_event_block_hashes[i] for i in missing_indices
                    ]
                group_indices = [group_indices[i] for i in missing_indices]

            logger.debug(
                "Storing KV cache for %d blocks (groups=%s) for request %s",
                len(keys),
                set(group_indices),
                req_id,
            )

            addrs: list[list[int]] = []
            sizes: list[list[int]] = []
            stored_events: list[BlockStored] = []
            # parent_block_hash chains live within a group, not across.
            if self.enable_kv_event:
                prev_key_per_group: dict[int, Any] = {}
                new_block_hashes = [
                    maybe_convert_block_hash(bh) for bh in kv_event_block_hashes
                ]

            for idx, (s, e, g_idx) in enumerate(
                zip(starts, ends, group_indices, strict=True)
            ):
                db = self.token_databases[g_idx]
                addr, size, _ = db.prepare_value(s, e, block_ids_per_group[g_idx])
                addrs.append(addr)
                sizes.append(size)

                if self.enable_kv_event:
                    token_ids = (
                        req_meta.token_ids[s:e]
                        if req_meta.token_ids is not None
                        else None
                    )
                    stored_event = BlockStored(
                        block_hashes=[new_block_hashes[idx]],
                        parent_block_hash=prev_key_per_group.get(g_idx),
                        token_ids=token_ids,
                        block_size=db.block_size,
                        lora_id=None,
                        medium="cpu",
                        lora_name=None,
                        group_idx=g_idx,
                    )
                    stored_events.append(stored_event)
                    prev_key_per_group[g_idx] = new_block_hashes[idx]

            if current_event is not None:
                current_event.synchronize()

            batch_bytes = _sum_batch_bytes(sizes)
            put_start = time.perf_counter()
            try:
                res = self.store.batch_put_from_multi_buffers(
                    keys,
                    addrs,
                    sizes,
                    self.replicate_config,
                )
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
                    logger.warning(
                        "batch_put failed: %d/%d keys failed "
                        "(codes=%s, batch_bytes=%d, num_keys=%d), "
                        "first_key=%s",
                        len(failed),
                        len(keys),
                        failed_codes,
                        batch_bytes,
                        len(keys),
                        keys[0] if keys else "N/A",
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
                else:
                    self._record_saved(req_id, token_len)
                    if self._clear_store_pressure():
                        logger.info(
                            "Mooncake CPU/disk offloading pressure cleared "
                            "after a successful store batch"
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
        finally:
            self.dec_stored_request(req_id)
            self.request_queue.task_done()


class KVCacheStoreRecvingThread(KVTransferThread):
    """Background thread for loading KV cache blocks from the store."""

    def __init__(
        self,
        store: Any,
        coord: MooncakeStoreCoordinator,
        token_databases: list[ChunkedTokenDatabase],
        block_size: int,
        tp_rank: int,
        ready_event: threading.Event,
        disk_offload_buffer_budget_bytes: int | None = None,
        record_operation: Callable[..., None] | None = None,
        request_queue: queue.Queue[Any] | None = None,
    ):
        super().__init__(
            store,
            token_databases,
            block_size,
            tp_rank,
            ready_event,
            name="KVCacheStoreRecvingThread",
            record_operation=record_operation,
            request_queue=request_queue,
        )
        # _invalid_block_ids can be access by both the Worker and RecvingThread
        self._invalid_block_ids_lock = threading.Lock()
        self._invalid_block_ids: set[int] = set()
        self.disk_offload_buffer_budget_bytes = disk_offload_buffer_budget_bytes
        self.usable_disk_offload_buffer_budget_bytes = (
            None
            if disk_offload_buffer_budget_bytes is None
            else _get_usable_disk_offload_buffer_budget_bytes(
                disk_offload_buffer_budget_bytes
            )
        )
        self.coord = coord

    def _add_load_error_block_ids(self, block_ids: list[int]) -> None:
        with self._invalid_block_ids_lock:
            self._invalid_block_ids.update(block_ids)

    def get_and_clear_block_ids_with_load_errors(self) -> set[int]:
        with self._invalid_block_ids_lock:
            invalid_block_ids = self._invalid_block_ids.copy()
            self._invalid_block_ids.clear()
        return invalid_block_ids

    def _handle_request(self, req_meta: ReqMeta):
        token_len = req_meta.load_spec.token_len  # type: ignore[union-attr]
        req_id = req_meta.req_id
        mask_num = (
            req_meta.load_spec.vllm_cached_tokens  # type: ignore[union-attr]
            // self.block_size
            * self.block_size
        )

        # Skip chunks the consumer's per-group spec wouldn't populate
        # locally (e.g. SWA pre-window) even if the producer stored them.
        load_mask_per_group = self.coord.load_mask(req_meta.block_hashes, token_len)

        addr_list: list[list[int]] = []
        size_list: list[list[int]] = []
        key_list: list[str] = []
        block_id_list: list[int] = []
        for g_idx, db in enumerate(self.token_databases):
            mask = load_mask_per_group[g_idx]
            for start, end, block_hash in db.process_tokens(
                token_len, req_meta.block_hashes, mask_num
            ):
                chunk_idx = start // db.block_size
                if chunk_idx >= len(mask) or not mask[chunk_idx]:
                    continue
                addr, size, block_id = db.prepare_value(
                    start, end, req_meta.block_ids[g_idx]
                )
                key_list.append(db.key_for(block_hash))
                addr_list.append(addr)
                size_list.append(size)
                block_id_list.append(block_id)

        # Rotate aligned lists by tp_rank for load balancing.
        rotation = self.tp_rank % len(key_list)
        key_list_c = _rotate_list(key_list, rotation)
        addr_list_c = _rotate_list(addr_list, rotation)
        size_list_c = _rotate_list(size_list, rotation)
        block_id_list_c = _rotate_list(block_id_list, rotation)

        load_batches = [(key_list_c, addr_list_c, size_list_c, block_id_list_c)]
        if self.usable_disk_offload_buffer_budget_bytes is not None:
            total_staging_bytes = sum(
                _estimate_disk_offload_staging_bytes(size) for size in size_list_c
            )
            if total_staging_bytes > self.usable_disk_offload_buffer_budget_bytes:
                assert self.disk_offload_buffer_budget_bytes is not None
                split_batches, oversized_key = _split_disk_offload_load_batches(
                    key_list_c,
                    addr_list_c,
                    size_list_c,
                    self.usable_disk_offload_buffer_budget_bytes,
                    self.disk_offload_buffer_budget_bytes,
                )
                if oversized_key is not None:
                    oversized_key_index = key_list_c.index(oversized_key)
                    # Mark every block: we skip the whole request, and the
                    # tp_rank rotation means oversized_key isn't necessarily
                    # the first block in the request's original order.
                    self._add_load_error_block_ids(block_id_list_c)
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
                load_batches = []
                block_id_offset = 0
                for batch_keys, batch_addrs, batch_sizes in split_batches:
                    next_block_id_offset = block_id_offset + len(batch_keys)
                    batch_block_ids = block_id_list_c[
                        block_id_offset:next_block_id_offset
                    ]
                    load_batches.append(
                        (batch_keys, batch_addrs, batch_sizes, batch_block_ids)
                    )
                    block_id_offset = next_block_id_offset

        current_batch_keys: list[str] = key_list_c
        current_batch_block_ids: list[int] = block_id_list_c
        batch_bytes = 0
        try:
            for batch_keys, batch_addrs, batch_sizes, batch_block_ids in load_batches:
                current_batch_keys = batch_keys
                current_batch_block_ids = batch_block_ids
                batch_bytes = _sum_batch_bytes(batch_sizes)
                tiers_by_key: dict[str, str] | None = None
                if envs.VLLM_MOONCAKE_STORE_TIER_LOG:
                    tiers_by_key = _get_replica_tiers_by_key(self.store, batch_keys)
                # Reset so the recorded RPC duration excludes tier lookup.
                load_get_start = time.perf_counter()
                res = self.store.batch_get_into_multi_buffers(
                    batch_keys, batch_addrs, batch_sizes
                )
                if tiers_by_key is not None:
                    _log_mooncake_load_tier_summary(
                        req_id, batch_keys, res, tiers_by_key
                    )
                failed = [
                    (key, value, block_id)
                    for key, value, block_id in zip(
                        batch_keys, res, batch_block_ids, strict=True
                    )
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
                    self._add_load_error_block_ids(
                        [block_id for _, _, block_id in failed]
                    )
                    logger.warning(
                        "Failed to get %d Mooncake keys from sub-batch "
                        "(batch_keys=%d, first_failures=%s)",
                        len(failed),
                        len(batch_keys),
                        [(key, value) for key, value, _ in failed[:3]],
                    )
                    break
        except Exception as e:
            self._add_load_error_block_ids(current_batch_block_ids)
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
        kv_cache_config: KVCacheConfig,
    ):
        try:
            from mooncake.store import (  # type: ignore
                MooncakeDistributedStore,
                ReplicateConfig,
            )
        except ImportError as e:
            raise ImportError(
                "Please install mooncake by following the instructions at "
                "https://github.com/kvcache-ai/Mooncake/blob/main/doc/"
                "en/build.md to run vLLM with MooncakeStoreConnector."
            ) from e

        model_config = vllm_config.model_config
        parallel_config = vllm_config.parallel_config

        self.dp_rank = parallel_config.data_parallel_index
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.pp_size = parallel_config.pipeline_parallel_size
        self.pp_rank = (parallel_config.rank // self.tp_size) % self.pp_size

        self.pcp_size = get_pcp_group().world_size
        self.pcp_rank = get_pcp_group().rank_in_group if self.pcp_size > 1 else 0
        self.dcp_size = get_dcp_group().world_size
        self.dcp_rank = get_dcp_group().rank_in_group if self.dcp_size > 1 else 0

        assert vllm_config.kv_transfer_config is not None
        self.kv_role = vllm_config.kv_transfer_config.kv_role
        self.load_async = vllm_config.kv_transfer_config.kv_connector_extra_config.get(
            "load_async", True
        )
        self.cache_config = vllm_config.cache_config
        self.block_size, self.hash_block_size = resolve_kv_cache_block_sizes(
            kv_cache_config, vllm_config
        )
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

        if self.num_kv_head < self.tp_size and self.dcp_size <= 1:
            # Dedup: TP ranks holding the same KV heads stripe PUTs across
            # one shared key namespace. DCP splits the TP group, so with
            # DCP>1 those ranks have different `@dcpN` namespaces and
            # striping would leave keys unwritten (OBJECT_NOT_FOUND on
            # GET). PCP is outer to TP (pcp_rank is constant within a TP
            # group), so it needs no guard.
            self.put_step = self.tp_size // self.num_kv_head
            self.head_or_tp_rank = self.tp_rank // self.put_step
        else:
            self.head_or_tp_rank = self.tp_rank
            self.put_step = 1

        self.metadata = build_key_metadata(
            vllm_config,
            tp_rank=self.head_or_tp_rank,
            pcp_rank=self.pcp_rank,
            dcp_rank=self.dcp_rank,
            pp_rank=self.pp_rank,
        )

        # Initialize MooncakeDistributedStore with its own TransferEngine
        store_config = MooncakeStoreConfig.load_from_config()
        extra_config = (
            vllm_config.kv_transfer_config.kv_connector_extra_config
            if vllm_config.kv_transfer_config
            else {}
        )
        self.store = MooncakeDistributedStore()
        local_ip = get_ip()
        local_hostname = rdma_utils.get_requester_local_hostname(local_ip)
        ret = self.store.setup(
            local_hostname,
            store_config.metadata_server,
            store_config.global_segment_size,
            store_config.local_buffer_size,
            store_config.protocol,
            store_config.device_name,
            store_config.master_server_address,
        )
        if ret != 0:
            msg = "Initialize MooncakeDistributedStore failed."
            logger.error(msg)
            raise RuntimeError(msg)

        preferred_segment = rdma_utils.get_configured_preferred_segment(extra_config)
        self.preferred_segment = preferred_segment
        self.store_replicate_config = ReplicateConfig()
        if preferred_segment is not None:
            self.store_replicate_config.preferred_segment = preferred_segment

        logger.info(
            "Mooncake mode=%s (global_segment_size=%d, local_buffer_size=%d, "
            "preferred_segment=%s, enable_offload=%s)",
            store_config.mode,
            store_config.global_segment_size,
            store_config.local_buffer_size,
            preferred_segment or "<none>",
            store_config.enable_offload,
        )
        if store_config.mode == "embedded":
            if store_config.enable_offload and preferred_segment is None:
                logger.warning(
                    "enable_offload is set in embedded mode without "
                    "preferred_segment; SSD tier will only see puts that "
                    "happen to land on the owner segment."
                )
            if preferred_segment is not None:
                logger.warning(
                    "preferred_segment=%s with mode=embedded: rank-"
                    "contributed segments will be idle.",
                    preferred_segment,
                )
        elif (
            store_config.mode == "standalone-store" and not store_config.enable_offload
        ):
            logger.warning(
                "standalone-store mode without enable_offload: large prefills "
                "may exceed the owner DirectIO budget."
            )

        self.disk_offload_buffer_budget_bytes = (
            DEFAULT_MOONCAKE_DISK_STAGING_BUFFER_BYTES
            if store_config.enable_offload
            else None
        )
        # Admin (reset) server on rank 0; prefix lookups are served by the
        # scheduler-owned lookup subprocess, not the worker.
        self.admin_server: StoreAdminServer | None = None
        if vllm_config.parallel_config.rank == 0:
            self.admin_server = StoreAdminServer(self, vllm_config)

        kv_event_config = vllm_config.kv_events_config
        self.enable_kv_events = False
        if kv_event_config and kv_event_config.enable_kv_cache_events:
            self.enable_kv_events = True

        self.kv_send_thread: KVCacheStoreSendingThread | None = None
        # Pool of load-receive threads
        self.kv_recv_threads: list[KVCacheStoreRecvingThread] = []
        self.num_recv_threads = max(1, envs.VLLM_MOONCAKE_LOAD_RECV_THREADS)
        self.recv_request_queue: queue.Queue[ReqMeta] = queue.Queue()
        self.finished_store_req: set[str] = set()
        self._kv_connector_stats_lock = threading.Lock()
        self.kv_connector_stats = MooncakeStoreConnectorStats()

        self._kv_cache_config = kv_cache_config
        self._kv_cache_groups, self.coord = build_store_coordinator(
            vllm_config, kv_cache_config, self.block_size, self.hash_block_size
        )
        # One ChunkedTokenDatabase per group; addresses populated in
        # register_kv_caches once the kv-cache layout is known.
        self.token_dbs: list[ChunkedTokenDatabase] = [
            ChunkedTokenDatabase(
                dataclasses.replace(self.metadata, group_id=g_idx),
                g.kv_cache_spec.block_size,
                hash_block_size=self.hash_block_size,
            )
            for g_idx, g in enumerate(self._kv_cache_groups)
        ]

    def register_cross_layers_kv_caches(self, kv_cache: torch.Tensor) -> None:
        """Register a cross-layers KV cache tensor.

        Wraps the unified tensor in a single-entry dict so that the
        existing stride-based logic in register_kv_caches() produces
        the correct single-segment result (block_len = page_size * num_layers).
        """
        self.register_kv_caches({"__cross_layer__": kv_cache})

    def register_kv_caches(
        self,
        kv_caches: dict[str, torch.Tensor | list[torch.Tensor]],
    ) -> None:
        """Register KV cache tensors and start transfer threads."""
        if not kv_caches:
            logger.warning("No KV caches to offload.")
            return

        # Resolve each entry to a representative tensor for storage
        # deduplication. For attention layers the value is already a tensor;
        # for Mamba layers it is a list of tensors that all share the same
        # underlying raw storage, so we take the first one.
        def _repr_tensor(v: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
            assert isinstance(v, torch.Tensor | list)
            return v if isinstance(v, torch.Tensor) else v[0]

        assert self.cache_config.num_gpu_blocks is not None
        self.num_blocks = self.cache_config.num_gpu_blocks

        seen_ptrs: set[int] = set()
        addrs: list[int] = []
        block_lens: list[int] = []

        for value in kv_caches.values():
            cache = _repr_tensor(value)
            cache_storage = cache.untyped_storage()
            base_addr = cache_storage.data_ptr()
            if base_addr in seen_ptrs:
                continue
            seen_ptrs.add(base_addr)
            region_len = cache_storage.nbytes()

            ret = self.store.register_buffer(base_addr, region_len)
            if ret != 0:
                logger.error(
                    "register_buffer failed for addr %#x len %d: %d",
                    base_addr,
                    region_len,
                    ret,
                )

            # Detect layout via stride: a dim whose byte-stride exceeds
            # page_size_bytes is an outer segment dim (e.g. the K/V dim of
            # FlashAttn's (2, num_blocks, ...)). FlashInfer/MLA's blocks-
            # outermost layout has no such dim and yields a single segment.
            el = cache.element_size()
            page_size_bytes = region_len // self.num_blocks
            outer_dims = [
                d for d in range(cache.ndim) if cache.stride(d) * el > page_size_bytes
            ]
            if not outer_dims:
                # Blocks-first layout (FlashInfer / MLA): one segment.
                addrs.append(base_addr)
                block_lens.append(page_size_bytes)
            else:
                # K/V-first layout (FlashAttn / ROCm): split segments.
                seg_stride = cache.stride(outer_dims[0]) * el
                for idx in range(cache.shape[outer_dims[0]]):
                    addrs.append(base_addr + idx * seg_stride)
                    block_lens.append(seg_stride // self.num_blocks)

        logger.info(
            "Registered KV caches: num_groups=%d, num_segments=%d, num_blocks=%d",
            len(self.token_dbs),
            len(addrs),
            self.num_blocks,
        )

        for db in self.token_dbs:
            db.set_kv_caches_base_addr(addrs)
            db.set_block_len(block_lens)

        # Start transfer threads
        if self.kv_role in ["kv_producer", "kv_both"]:
            ready_event_sending = threading.Event()
            self.kv_send_thread = KVCacheStoreSendingThread(
                self.store,
                self.coord,
                self.token_dbs,
                self.block_size,
                self.tp_rank,
                self.put_step,
                self.kv_role,
                ready_event_sending,
                self.enable_kv_events,
                self.store_replicate_config,
                record_operation=self._record_kv_connector_operation,
            )
            self.kv_send_thread.start()

        self.kv_recv_threads = []
        ready_events_recving = []
        for i in range(self.num_recv_threads):
            ready_event_recving = threading.Event()
            recv_thread = KVCacheStoreRecvingThread(
                self.store,
                self.coord,
                self.token_dbs,
                self.block_size,
                self.tp_rank,
                ready_event_recving,
                disk_offload_buffer_budget_bytes=self.disk_offload_buffer_budget_bytes,
                record_operation=self._record_kv_connector_operation,
                request_queue=self.recv_request_queue,
            )
            recv_thread.name = f"KVCacheStoreRecvingThread-{i}"
            recv_thread.start()
            self.kv_recv_threads.append(recv_thread)
            ready_events_recving.append(ready_event_recving)
        for ready_event_recving in ready_events_recving:
            ready_event_recving.wait()
        logger.info(
            "Started %d Mooncake KV-load receive thread(s)", self.num_recv_threads
        )

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

            load_spec.token_len = load_spec.kvpool_cached_tokens
            self.recv_request_queue.put(request)

        assert self.load_async, "load_async must be True for better performance."
        # Issue stores with CUDA event synchronization
        if self.kv_role in ["kv_producer", "kv_both"]:
            current_event = None
            for request in meta.requests:
                if request.can_save:
                    # blocking=True: synchronize() sleeps instead of spinning.
                    current_event = torch.cuda.Event(blocking=True)
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

        done_recving: set[str] = set()
        if self.load_async:
            for recv_thread in self.kv_recv_threads:
                done_recving |= recv_thread.get_and_clear_finished_requests()

        logger.debug(
            "Completed send: %d, recv: %d, tp_rank: %d",
            len(done_sending),
            len(done_recving),
            self.tp_rank,
        )
        return done_sending, done_recving

    def get_block_ids_with_load_errors(self) -> set[int]:
        block_ids: set[int] = set()
        for recv_thread in self.kv_recv_threads:
            block_ids |= recv_thread.get_and_clear_block_ids_with_load_errors()
        return block_ids

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

    def get_kv_events(self) -> list[BlockStored]:
        if self.enable_kv_events and self.kv_send_thread is not None:
            return self.kv_send_thread.get_kv_events()
        return []

    def close(self) -> None:
        """Release the MooncakeDistributedStore handle on teardown.

        Closing the store frees its TransferEngine, the registered RDMA
        buffers, and the connection to the master server. Idempotent so it is
        safe to call from both the explicit shutdown path and ``__del__``.
        """
        admin_server = getattr(self, "admin_server", None)
        if admin_server is not None:
            if not admin_server.close():
                # A wedged RESET handler is still using the store; leak both
                # to process teardown rather than close the store under it.
                # Keeping the admin_server reference lets a later close()
                # (e.g. __del__ after shutdown) finish the job if the
                # handler has exited by then.
                return
            self.admin_server = None
        store = getattr(self, "store", None)
        if store is None:
            return
        self.store = None
        try:
            store.close()
        except Exception as e:
            logger.warning("Error closing MooncakeDistributedStore: %s", e)


# ============================================================
# Prefix-hit lookup
#
# Served by a dedicated subprocess. Lookups are pure-Python heavy
# (candidate-key construction + find_longest_cache_hit over up to ~100k
# chunk hashes), and every in-engine home for that work starves on the GIL
# under load: a worker-side thread competes with model execution (measured
# ~72s per lookup at high concurrency vs milliseconds when idle), an
# EngineCore thread competes with the scheduling loop, and even a
# ProcessPoolExecutor stalls because its manager thread lives in
# EngineCore. Hence: a spawn subprocess that owns its GIL, driven directly
# over ZMQ (the client's send is C-level, no dispatch thread), with its own
# store client so exist checks go straight to the master.
# ============================================================


@dataclass
class LookupContext:
    """Everything the lookup subprocess needs to answer prefix-hit queries."""

    coord: MooncakeStoreCoordinator
    kv_cache_groups: list[KVCacheGroupSpec]
    group_block_sizes: list[int]
    lookup_key_prefixes: tuple[tuple[str, ...], ...]
    # Keys per (group, hash) candidate; a hash counts as present only when
    # every TP/PP-expanded key exists.
    expected_per_key: int
    breakpoints: SessionBreakpointTracker


def build_key_metadata(
    vllm_config: VllmConfig,
    *,
    tp_rank: int = 0,
    pcp_rank: int = 0,
    dcp_rank: int = 0,
    pp_rank: int = 0,
) -> KeyMetadata:
    """Pool-key metadata, shared by the worker (real ranks) and the lookup
    subprocess (rank 0 everywhere, matching the worker-rank-0 placement of
    the lookup path it replaced)."""
    assert vllm_config.kv_transfer_config is not None
    return KeyMetadata(
        model_name=vllm_config.model_config.model.rstrip("/").split("/")[-1],
        tp_rank=tp_rank,
        pcp_rank=pcp_rank,
        dcp_rank=dcp_rank,
        pp_rank=pp_rank,
        cache_prefix=str(
            vllm_config.kv_transfer_config.kv_connector_extra_config.get(
                "cache_prefix", ""
            )
        ),
    )


def build_store_coordinator(
    vllm_config: VllmConfig,
    kv_cache_config: KVCacheConfig,
    scheduler_block_size: int,
    hash_block_size: int,
) -> tuple[list[KVCacheGroupSpec], MooncakeStoreCoordinator]:
    """Normalized KV groups + coordinator, shared by the worker and the
    lookup subprocess so the two can never disagree on group layout."""
    # Single-group + PCP/DCP > 1: scale the lone group's spec.block_size to
    # scheduler_block_size so the coordinator's
    # ``block_size % hash_block_size == 0`` invariant holds.
    groups = list(kv_cache_config.kv_cache_groups)
    if len(groups) == 1 and groups[0].kv_cache_spec.block_size != scheduler_block_size:
        g = groups[0]
        groups = [
            dataclasses.replace(
                g,
                kv_cache_spec=dataclasses.replace(
                    g.kv_cache_spec, block_size=scheduler_block_size
                ),
            )
        ]
    spec_cfg = getattr(vllm_config, "speculative_config", None)
    use_eagle = bool(
        spec_cfg.use_eagle()
        if spec_cfg is not None and callable(getattr(spec_cfg, "use_eagle", None))
        else False
    )
    coord = MooncakeStoreCoordinator(
        groups,
        scheduler_block_size=scheduler_block_size,
        hash_block_size=hash_block_size,
        use_eagle=use_eagle,
        retention_interval=envs.VLLM_PREFIX_CACHE_RETENTION_INTERVAL,
    )
    return groups, coord


def build_lookup_context(
    vllm_config: VllmConfig,
    kv_cache_config: KVCacheConfig,
) -> LookupContext:
    scheduler_block_size, hash_block_size = resolve_kv_cache_block_sizes(
        kv_cache_config, vllm_config
    )
    groups, coord = build_store_coordinator(
        vllm_config, kv_cache_config, scheduler_block_size, hash_block_size
    )
    metadata = build_key_metadata(vllm_config)
    model_config = vllm_config.model_config
    num_kv_head = (
        1
        if getattr(model_config, "use_mla", False)
        else model_config.get_total_num_kv_heads()
    )
    parallel_config = vllm_config.parallel_config
    tp_size = parallel_config.tensor_parallel_size
    pp_size = parallel_config.pipeline_parallel_size
    pcp_size = parallel_config.prefill_context_parallel_size
    dcp_size = parallel_config.decode_context_parallel_size
    # (tp_rank, pcp_rank, dcp_rank, pp_rank) namespaces
    if dcp_size > 1:
        # DCP reuses the TP workers and splits each TP group into
        # contiguous DCP groups, so dcp_rank == tp_rank % dcp_size.
        # Store/load paths do not apply KV-head dedup under DCP
        rank_namespaces = tuple(
            (tp_rank, pcp_rank, tp_rank % dcp_size, pp_rank)
            for pcp_rank in range(pcp_size)
            for tp_rank in range(tp_size)
            for pp_rank in range(pp_size)
        )
    else:
        # Without DCP, TP ranks that share a KV head write identical KV, so
        # lookup only needs one TP namespace per unique KV head.
        tp_count = min(tp_size, num_kv_head)
        rank_namespaces = tuple(
            (tp_rank, pcp_rank, 0, pp_rank)
            for pcp_rank in range(pcp_size)
            for tp_rank in range(tp_count)
            for pp_rank in range(pp_size)
        )
    prefixes = tuple(
        tuple(
            PoolKey.build_prefix(
                dataclasses.replace(metadata, group_id=g_idx),
                tp_rank=tp_rank,
                pcp_rank=pcp_rank,
                dcp_rank=dcp_rank,
                pp_rank=pp_rank,
            )
            for tp_rank, pcp_rank, dcp_rank, pp_rank in rank_namespaces
        )
        for g_idx in range(len(groups))
    )
    return LookupContext(
        coord=coord,
        kv_cache_groups=groups,
        group_block_sizes=[g.kv_cache_spec.block_size for g in groups],
        lookup_key_prefixes=prefixes,
        expected_per_key=len(rank_namespaces),
        breakpoints=SessionBreakpointTracker(
            hash_block_size,
            coord.lcm_block_size,
            envs.VLLM_MOONCAKE_SESSION_BREAKPOINT_HISTORY_SIZE,
        ),
    )


def run_lookup(
    ctx: LookupContext,
    exist_fn: Callable[[list[str]], list[int]],
    token_len: int,
    block_hashes: Sequence[BlockHash],
    session_id: str | None = None,
) -> int:
    """Prefix-hit lookup with the session-breakpoint fast path.

    A recorded breakpoint clamps the scan to one known boundary, cutting
    candidate keys. The clamped result is trusted only when fully verified
    against the store (or when the clamped scan is dense); otherwise fall
    back to the full scan.
    """
    if not block_hashes or token_len <= 0:
        return 0

    breakpoint_len = ctx.breakpoints.best_len(session_id, token_len, block_hashes)
    if breakpoint_len is not None:
        # TODO: Make this fast path EAGLE/MTP-aware. EAGLE may prune one
        # matched block, and sparse groups may need a peek block outside the
        # clamped boundary to validate a breakpoint.
        breakpoint_hit = _exist_lookup(
            ctx,
            exist_fn,
            breakpoint_len,
            block_hashes,
            aligned_boundary_token_len=breakpoint_len,
        )
        if breakpoint_hit == breakpoint_len or (
            breakpoint_len == token_len and _lookup_is_dense(ctx.coord, breakpoint_len)
        ):
            # A full breakpoint hit can skip the new tail. Exact-length
            # partial hits are only final when the clamped scan is dense;
            # sparse groups may have an earlier boundary for fallback.
            return breakpoint_hit

    return _exist_lookup(ctx, exist_fn, token_len, block_hashes)


def _exist_lookup(
    ctx: LookupContext,
    exist_fn: Callable[[list[str]], list[int]],
    token_len: int,
    block_hashes: Sequence[BlockHash],
    aligned_boundary_token_len: int | None = None,
) -> int:
    """How many prefix tokens exist in the store, across all TP/PP shards.

    ``aligned_boundary_token_len`` narrows sparse (SWA) groups to the blocks
    needed to prove one exact boundary; dense groups are unchanged.
    """
    # Per-(group, hash) candidate keys expanded across TP/PP;
    # candidate_meta keeps (group, hash_bytes) so the exist result slices
    # back to candidates.
    candidate_keys: list[str] = []
    candidate_meta: list[tuple[int, bytes]] = []
    lookup_masks = ctx.coord.lookup_mask(
        token_len,
        aligned_boundary_token_len=aligned_boundary_token_len,
    )
    for g_idx, spec_block_size in enumerate(ctx.group_block_sizes):
        lookup_mask = lookup_masks[g_idx]
        key_prefixes = ctx.lookup_key_prefixes[g_idx]
        group_hashes = ctx.coord.block_hashes_for_spec(
            block_hashes, ctx.kv_cache_groups[g_idx].kv_cache_spec
        )
        max_chunks = min(len(group_hashes), cdiv(token_len, spec_block_size))
        mask_limit = (
            max_chunks if lookup_mask is None else min(max_chunks, len(lookup_mask))
        )
        for chunk_id in range(mask_limit):
            if lookup_mask is not None and not lookup_mask[chunk_id]:
                continue
            h = group_hashes[chunk_id]
            hash_hex = h.hex()
            for key_prefix in key_prefixes:
                candidate_keys.append(PoolKey.build_key_string(key_prefix, hash_hex))
            candidate_meta.append((g_idx, bytes(h)))

    if not candidate_keys:
        return 0

    try:
        res = exist_fn(candidate_keys)
    except Exception as e:
        logger.error("Remote connection failed in lookup: %s", e)
        return 0

    ranks_per_candidate = ctx.expected_per_key
    exists_set = {
        (g_idx, hash_bytes)
        for i, (g_idx, hash_bytes) in enumerate(candidate_meta)
        if all(
            res[i * ranks_per_candidate + j] == 1 for j in range(ranks_per_candidate)
        )
    }
    _masks, hit_length = ctx.coord.find_longest_cache_hit(
        block_hashes, token_len, ExternalCachedBlockPool(exists_set)
    )
    return hit_length


def _lookup_is_dense(coord: MooncakeStoreCoordinator, token_len: int) -> bool:
    return all(
        mask is None
        for mask in coord.lookup_mask(
            token_len,
            aligned_boundary_token_len=token_len,
        )
    )


def _lookup_server_main(
    vllm_config: VllmConfig,
    kv_cache_config: KVCacheConfig,
    ipc_path: str,
) -> None:
    """Entry point of the lookup subprocess (see the section comment above
    for why this must be a subprocess)."""
    lookup_ctx = build_lookup_context(vllm_config, kv_cache_config)

    # Own store client straight to the master: batch_is_exist is a C++
    # master RPC, so lookups never touch a GIL-saturated engine process.
    # The tiny local segment is the price of a client handle.
    from mooncake.store import MooncakeDistributedStore

    store_config = MooncakeStoreConfig.load_from_config()
    store = MooncakeDistributedStore()
    local_hostname = rdma_utils.get_requester_local_hostname(get_ip())
    ret = store.setup(
        local_hostname,
        store_config.metadata_server,
        64 * 1024 * 1024,
        64 * 1024 * 1024,
        store_config.protocol,
        store_config.device_name,
        store_config.master_server_address,
    )
    if ret != 0:
        raise RuntimeError(f"lookup subprocess store setup failed: {ret}")

    ctx = zmq.Context()  # type: ignore[attr-defined]
    sock = make_zmq_socket(ctx, ipc_path, zmq.REP, bind=True)  # type: ignore[attr-defined]
    logger.info("Mooncake lookup subprocess serving on %s", ipc_path)
    _serve_lookup_requests(sock, lookup_ctx, store.batch_is_exist)


def _serve_lookup_requests(
    sock: Any,
    ctx: LookupContext,
    exist_fn: Callable[[list[str]], list[int]],
) -> None:
    """REP loop over the lookup wire format (see ``protocol.py``)."""
    while True:
        frames = sock.recv_multipart(copy=False)
        req_id = bytes(frames[0])
        msg_type = bytes(frames[1])
        token_len = int.from_bytes(bytes(frames[2]), "big")
        hash_len = int.from_bytes(bytes(frames[3]), "big")
        session_id = bytes(frames[5]).decode() or None
        block_hashes: Sequence[BlockHash] = (
            BlobBlockHashes(frames[4].buffer, hash_len) if hash_len else []
        )
        hit = 0
        try:
            if msg_type == LOOKUP_MSG:
                hit = run_lookup(
                    ctx, exist_fn, token_len, block_hashes, session_id=session_id
                )
            elif msg_type == RECORD_BP_MSG:
                ctx.breakpoints.record(session_id, token_len, block_hashes)
            else:
                logger.warning("Lookup subprocess: unknown msg_type %r", msg_type)
        except Exception:
            logger.exception("Lookup subprocess: %r handler failed", msg_type)
        sock.send_multipart([req_id, hit.to_bytes(4, "big")])


# ============================================================
# Store admin server (worker rank 0)
# ============================================================


class StoreAdminServer:
    """ZMQ REP server on worker rank 0 for admin commands.

    Currently handles one request type, tagged at frame 0:
    - ``RESET_MSG``: drains the send thread queue, then runs
      ``store.remove_all(force=True)``. Caller must have paused the
      scheduler first. Session breakpoints in the lookup subprocess need no
      reset: they are re-verified against the (now empty) store on every
      fast-path hit, so stale ones self-heal.
    """

    def __init__(
        self,
        store_worker: MooncakeStoreWorker,
        vllm_config: VllmConfig,
    ):
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
        self._closed = False
        # Periodic recv timeout so close() can stop the loop and then close
        # the socket from its own thread (zmq sockets are not thread-safe).
        self.socket.setsockopt(zmq.RCVTIMEO, 1000)  # type: ignore[attr-defined]

        def process_request():
            while self.running:
                try:
                    all_frames = self.socket.recv_multipart(copy=False)
                except zmq.Again:  # type: ignore[attr-defined]
                    continue
                msg_type = bytes(all_frames[0])

                if msg_type == RESET_MSG:
                    try:
                        # Drain in-flight puts before wiping the master;
                        # otherwise stale puts can repopulate it post-reset.
                        # Safe across HMA: store.remove_all wipes the underlying
                        # flat key space, clearing every (group_id, hash) entry.
                        if self.store_worker.kv_send_thread is not None:
                            self.store_worker.kv_send_thread.request_queue.join()
                        self.store_worker.store.remove_all(force=True)
                        logger.info("Mooncake store reset via remove_all succeeded.")
                        self.socket.send(RESP_OK)
                    except Exception as e:
                        logger.error("Mooncake remove_all failed: %s", e)
                        self.socket.send(RESP_ERR)

                else:
                    logger.warning(
                        "StoreAdminServer received unknown msg_type: %r",
                        msg_type,
                    )
                    self.socket.send(RESP_ERR)

        self.thread = threading.Thread(target=process_request, daemon=True)
        self.thread.start()

    def close(self) -> bool:
        """Idempotent: stop the REP loop, then release the socket/path.

        Waits for the loop thread to exit before touching the socket (zmq
        sockets are not thread-safe): an idle loop exits within RCVTIMEO,
        and an in-flight RESET is given time to finish so the socket and
        the worker's store aren't torn down under a live handler.

        Returns True once the thread has exited (resources released);
        False while it is still wedged in a handler (e.g. a stuck store
        RPC) — the caller must then leave shared state (the store handle)
        alone and may retry later. A wedged thread leaks its socket to
        process teardown rather than having it closed from another thread.
        """
        self.running = False
        self.thread.join(timeout=60)
        if self.thread.is_alive():
            logger.warning(
                "StoreAdminServer thread still busy after 60s (in-flight "
                "reset?); leaving its socket and the store to process "
                "teardown."
            )
            return False
        if not self._closed:
            self._closed = True
            self.socket.close(linger=0)
            if os.path.exists(self._ipc_path):
                os.unlink(self._ipc_path)
        return True


# ============================================================
# Lookup Key Client
# ============================================================


class LookupKeyClient:
    """Scheduler-side client for both MooncakeStoreConnector channels.

    - Lookup channel: DEALER to the lookup subprocess for prefix-hit
      queries and session-breakpoint records. Sending is a single C-level
      zmq call — deliberately no dispatch thread in this process, which
      would starve on the EngineCore GIL under load.
    - Admin channel: REQ to ``StoreAdminServer`` on worker rank 0 (reset).
    """

    def __init__(self, vllm_config: VllmConfig):
        self.ctx = zmq.Context()  # type: ignore[attr-defined]
        socket_path = get_zmq_rpc_path_lookup(vllm_config)
        self.socket = make_zmq_socket(
            self.ctx,
            socket_path,
            zmq.REQ,  # type: ignore[attr-defined]
            bind=False,
        )

        self._proc: Any = None
        self._sock: Any = None
        self._ipc_path: str | None = None
        self._proc_dead = False
        # Bound wait for blocking lookups: if the subprocess is alive but
        # unresponsive (still importing/binding, or the master is
        # unreachable), fail open instead of stalling the scheduler forever.
        self._block_timeout_s = 120.0
        # req_id -> wire id of the current in-flight attempt. Replies are
        # matched on the wire id (``<seq>|<req_id>``), so a reply from a
        # discarded attempt can't be served to a later request that reuses
        # the same request id.
        self._pending: dict[str, bytes] = {}
        self._results: dict[str, int] = {}
        self._lookup_seq = 0

    def start_lookup_subprocess(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
    ) -> None:
        """Spawn the lookup subprocess and connect a DEALER socket to it.

        The IPC path is unique per spawn (not just per parent PID): a
        terminated child may leave its bound socket file behind, and a new
        subprocess binding the same path would fail with EADDRINUSE.
        """
        import multiprocessing as mp
        import uuid

        ipc_path = (
            f"{get_zmq_rpc_path_lookup(vllm_config)}_lookup_"
            f"{os.getpid()}_{uuid.uuid4().hex[:8]}"
        )
        proc = mp.get_context("spawn").Process(
            target=_lookup_server_main,
            args=(vllm_config, kv_cache_config, ipc_path),
            daemon=True,
        )
        proc.start()
        self.connect_lookup(ipc_path, proc)

    def connect_lookup(self, ipc_path: str, proc: Any) -> None:
        self._proc = proc
        self._ipc_path = ipc_path
        self._sock = make_zmq_socket(
            self.ctx,
            ipc_path,
            zmq.DEALER,  # type: ignore[attr-defined]
            bind=False,
        )

    def _lookup_alive(self) -> bool:
        if self._proc_dead:
            return False
        if self._proc is None or not self._proc.is_alive():
            self._proc_dead = True
            logger.error(
                "Mooncake lookup subprocess is gone; external prefix-cache "
                "lookups disabled (failing open with 0 hits)."
            )
        return not self._proc_dead

    def _send(
        self,
        wire_id: bytes,
        msg_type: bytes,
        token_len: int,
        block_hashes: Sequence[BlockHash],
        session_id: str | None,
    ) -> None:
        hash_len = len(bytes(block_hashes[0])) if block_hashes else 0
        # Leading empty frame: the REQ/REP envelope delimiter a DEALER must
        # add by hand.
        self._sock.send_multipart(
            [
                b"",
                wire_id,
                msg_type,
                token_len.to_bytes(4, "big"),
                hash_len.to_bytes(2, "big"),
                b"".join(bytes(h) for h in block_hashes),
                (session_id or "").encode(),
            ],
            copy=False,
        )

    def _collect(self, timeout_ms: int) -> None:
        """File arrived replies under their req_id. Replies whose wire id is
        not the current attempt for that req_id — discarded/superseded
        attempts and breakpoint-record acks (empty wire id) — are dropped."""
        while self._sock.poll(timeout_ms):
            frames = self._sock.recv_multipart()
            timeout_ms = 0
            wire_id = bytes(frames[1])
            _seq, _, req_id_bytes = wire_id.partition(b"|")
            req_id = req_id_bytes.decode()
            if req_id and self._pending.get(req_id) == wire_id:
                del self._pending[req_id]
                self._results[req_id] = int.from_bytes(frames[2], "big")

    def lookup(
        self,
        req_id: str,
        token_len: int,
        block_hashes: Sequence[BlockHash],
        session_id: str | None = None,
        non_block: bool = False,
    ) -> int | None:
        """Prefix-hit lookup in the lookup subprocess.

        With ``non_block`` returns None until the reply arrives, so the
        caller retries on a later scheduler step. Fails open to 0 if the
        subprocess dies.
        """
        if not block_hashes or token_len <= 0:
            return 0
        if not self._lookup_alive():
            self._pending.pop(req_id, None)
            return 0
        self._collect(0)
        if req_id in self._results:
            return self._results.pop(req_id)
        if req_id not in self._pending:
            self._lookup_seq += 1
            wire_id = f"{self._lookup_seq}|{req_id}".encode()
            self._send(wire_id, LOOKUP_MSG, token_len, block_hashes, session_id)
            self._pending[req_id] = wire_id
        if non_block:
            return None
        deadline = time.monotonic() + self._block_timeout_s
        while req_id not in self._results:
            self._collect(1000)
            if req_id in self._results:
                break
            if not self._lookup_alive():
                self._pending.pop(req_id, None)
                return 0
            if time.monotonic() > deadline:
                logger.warning(
                    "Mooncake lookup for %s timed out after %.0fs (subprocess "
                    "alive but unresponsive — still starting up, or the "
                    "master is unreachable); failing open with 0 hits.",
                    req_id,
                    self._block_timeout_s,
                )
                self._pending.pop(req_id, None)
                return 0
        return self._results.pop(req_id)

    def record_session_breakpoint(
        self,
        session_id: str | None,
        token_len: int,
        block_hashes: Sequence[BlockHash],
    ) -> None:
        """Fire-and-forget session-breakpoint record; the subprocess's reply
        (empty req_id) is dropped by ``_collect``."""
        if not session_id or not block_hashes or token_len <= 0:
            return
        if not self._lookup_alive():
            return
        # Drain any queued replies (incl. earlier record acks) so
        # record-heavy phases with no interleaved lookups can't grow the
        # DEALER receive queue without bound.
        self._collect(0)
        self._send(b"", RECORD_BP_MSG, token_len, block_hashes, session_id)

    def discard(self, req_id: str) -> None:
        """Drop any cached/in-flight lookup for ``req_id`` (e.g. on abort).

        The in-flight attempt's wire id is forgotten, so its late reply is
        dropped even if a new request reuses the same request id."""
        self._pending.pop(req_id, None)
        self._results.pop(req_id, None)

    def reset(self) -> bool:
        """Trigger ``store.remove_all(force=True)`` on worker rank 0.

        Ordering assumption: caller MUST ensure no in-flight Mooncake
        lookups or transfers when invoking reset. In RL workflows this
        holds naturally at the step boundary after weight updates and
        rollout drain. Returns True on ACK, False on NACK.
        """
        self.socket.send(RESET_MSG)
        resp = self.socket.recv()
        return bytes(resp) == RESP_OK

    def close(self):
        """Idempotent: close both channels and reap the lookup subprocess
        (terminate, join, then kill) so repeated engine create/shutdown in
        one process doesn't accumulate zombie children or IPC sockets."""
        self.socket.close(linger=0)
        if self._sock is not None:
            self._sock.close(linger=0)
            self._sock = None
        proc, self._proc = self._proc, None
        if proc is not None:
            proc.terminate()
            proc.join(timeout=2)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=2)
        ipc_path, self._ipc_path = self._ipc_path, None
        if ipc_path:
            path = ipc_path.removeprefix("ipc://")
            if os.path.exists(path):
                with contextlib.suppress(OSError):
                    os.unlink(path)


def get_zmq_rpc_path_lookup(vllm_config: VllmConfig) -> str:
    """Construct IPC path for ZMQ lookup socket."""
    assert vllm_config.kv_transfer_config is not None
    dp_rank = vllm_config.parallel_config.data_parallel_index
    base_url = envs.VLLM_RPC_BASE_PATH
    rpc_port = 0
    hostname = socket.gethostname()
    extra_config = vllm_config.kv_transfer_config.kv_connector_extra_config
    if "lookup_rpc_port" in extra_config:
        rpc_port = extra_config["lookup_rpc_port"]
    logger.debug("Base URL: %s, RPC Port: %s", base_url, rpc_port)
    return (
        f"ipc://{base_url}/lookup_rpc_port_{rpc_port}_host_{hostname}_dp_rank{dp_rank}"
    )
