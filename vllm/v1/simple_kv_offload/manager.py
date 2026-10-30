# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-side manager for SimpleCPUOffloadConnector."""

import contextlib
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field, fields, replace
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

from vllm.config import VllmConfig
from vllm.distributed.kv_events import (
    MEDIUM_CPU,
    MEDIUM_STORAGE,
    BlockRemoved,
    BlockStored,
    KVCacheEvent,
)
from vllm.distributed.kv_transfer.kv_connector.utils import yield_req_data
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_coordinator import (
    KVCacheCoordinator,
    get_kv_cache_coordinator,
)
from vllm.v1.core.kv_cache_utils import (
    BlockHashWithGroupId,
    ExternalBlockHash,
    generate_block_hash_extra_keys,
    get_block_hash,
    get_group_id,
    maybe_convert_block_hash,
    resolve_dcp_kv_block_size,
    to_event_extra_keys,
)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (
    MambaSpec,
    SlidingWindowSpec,
    get_kv_cache_spec_kind,
    get_kv_cache_spec_sliding_window,
)
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.simple_kv_offload.disk_backend import plan_disk_layout
from vllm.v1.simple_kv_offload.metadata import (
    SimpleCPUOffloadMetadata,
    SimpleCPUOffloadWorkerMetadata,
)
from vllm.v1.simple_kv_offload.metrics import (
    OUTCOME_TO_FIELD,
    MetricName,
    SimpleCPUOffloadStats,
)

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.kv_cache_utils import BlockHash, KVCacheBlock
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass
class BlockStoreMeta:
    """Per-block metadata snapshot for BlockStored event emission.

    Captured at store-prep time (when the Request is available) and carried
    through TransferMeta to async completion, where the Request is gone.
    """

    token_ids: list[int]
    parent_block_hash: ExternalBlockHash | None
    lora_id: int | None
    lora_name: str | None
    extra_keys: tuple[Any, ...] | None


def _block_store_meta(
    request: "Request", block_idx: int, block_size: int, mm_idx: int
) -> tuple[BlockStoreMeta, int]:
    """Snapshot the event payload of the request's block ``block_idx``.

    Returns:
        The payload and the multi-modal index to start the next block's scan.

    """
    token_start = block_idx * block_size
    token_end = token_start + block_size
    extra_keys, mm_idx = generate_block_hash_extra_keys(
        request, token_start, token_end, mm_idx
    )
    lora_req = request.lora_request
    meta = BlockStoreMeta(
        token_ids=list(request.all_token_ids[token_start:token_end]),
        parent_block_hash=(
            None
            if block_idx == 0
            else maybe_convert_block_hash(request.block_hashes[block_idx - 1])
        ),
        lora_id=lora_req.adapter_id if lora_req else None,
        lora_name=lora_req.name if lora_req else None,
        extra_keys=extra_keys,
    )
    return meta, mm_idx


@dataclass
class TransferMeta:
    gpu_block_ids: list[int]
    cpu_block_ids: list[int]
    # KV cache group of each block; block ids are unique only within a group.
    group_ids: list[int]
    block_meta: list[BlockStoreMeta | None] | None = None
    # Event payloads for the partial re-key hashes the stored blocks carried
    # at selection time, keyed by hash; the primary-hash payloads are in
    # block_meta. None when kv cache events are disabled.
    secondary_meta: dict[BlockHashWithGroupId, BlockStoreMeta] | None = None


def _pool_blocks(
    pool: BlockPool, group_ids: list[int], block_ids: list[int]
) -> list["KVCacheBlock"]:
    return [pool.get_block(g, b) for g, b in zip(group_ids, block_ids, strict=True)]


@dataclass
class LoadRequestState:
    request: "Request"
    transfer_meta: TransferMeta
    load_event: int | None = None
    finished: bool = False


# NOTE: This per-request state is only used in eager mode.
@dataclass
class StoreRequestState:
    request: "Request"
    # Accumulated block IDs from scheduler_output via yield_req_data.
    block_ids: tuple[list[int], ...]
    # Per-group cursors tracking how many blocks have been stored/skipped.
    num_stored_blocks: list[int]
    store_events: set[int] = field(default_factory=set)
    finished: bool = False


class _StoreAdmission(Enum):
    READY = auto()
    NULL_BLOCK = auto()
    NOT_HASHED = auto()
    IN_FLIGHT = auto()
    ALREADY_CACHED = auto()


@dataclass
class BoundaryStoreStats:
    """Cumulative counters for boundary hand-off stores.

    Every branch that declines a hand-off is silent by design: the boundary is
    simply not offloaded and a later request misses. Without counters the only
    symptom is a lower cache hit rate with nothing in the logs, so each decline
    reason is counted and exposed through
    ``SimpleCPUOffloadConnector.get_boundary_store_stats()``. Reset by
    ``reset()``, which carries any not-yet-drained deltas; not otherwise
    cleared.
    """

    published: int = 0
    stored: int = 0
    dropped_cpu_full: int = 0
    dropped_request_gone: int = 0
    skipped_already_cached: int = 0
    skipped_in_flight: int = 0
    dropped_null_block: int = 0
    dropped_not_hashed: int = 0


class SimpleCPUOffloadScheduler:
    """Scheduler-side manager for CPU offloading."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: "KVCacheConfig | None",
        cpu_capacity_bytes: int,
        scheduler_block_size: int,
        block_size: int,
        lazy_offload: bool = False,
        disk_capacity_bytes: int = 0,
        use_page_cache: bool = False,
        num_cpu_blocks: int | None = None,
        aligned_num_cpu_blocks: int | None = None,
    ):
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        # When disk mode is active, the offload pool size is disk-based.
        offload_capacity = (
            disk_capacity_bytes if disk_capacity_bytes > 0 else cpu_capacity_bytes
        )
        self.enable_kv_cache_events = (
            vllm_config.kv_events_config is not None
            and vllm_config.kv_events_config.enable_kv_cache_events
        )
        dcp_world_size = vllm_config.parallel_config.decode_context_parallel_size
        self.cp_world_size = dcp_world_size
        self.scheduler_block_size = scheduler_block_size
        self.block_size = block_size
        # Derive a CPU KVCacheConfig from the GPU config and build a coordinator
        assert kv_cache_config is not None
        page_bytes = kv_cache_config.large_page_bytes
        if disk_capacity_bytes > 0:
            # O_DIRECT pads every slot to the driver alignment (see
            # plan_disk_layout); sizing the pool with the GPU page would make
            # the backing file outgrow disk_capacity_bytes.
            page_bytes = plan_disk_layout(
                page_bytes,
                [
                    kv_cache_config.group_geometry(g).block_bytes
                    for g in range(len(kv_cache_config.kv_cache_groups))
                ],
                use_page_cache,
            ).page_bytes
        self.cpu_kv_cache_config = self._derive_cpu_config(
            kv_cache_config,
            offload_capacity,
            page_bytes,
            num_cpu_blocks,
            aligned_num_cpu_blocks,
        )
        self.num_cpu_blocks = self.cpu_kv_cache_config.num_pages
        self.prefix_cacheable_group_ids = (
            self.cpu_kv_cache_config.prefix_cacheable_group_ids
        )
        self.kv_event_medium = MEDIUM_STORAGE if disk_capacity_bytes > 0 else MEDIUM_CPU
        # Must track kv_event_medium.
        self._info_labelvalues: tuple[str, ...] = (
            "disk" if disk_capacity_bytes > 0 else "cpu",
            str(use_page_cache).lower(),
            str(lazy_offload).lower(),
            str(self.num_cpu_blocks),
        )
        logger.info(
            "SimpleCPUOffloadScheduler: Allocating %d offload blocks "
            "(%.2f GB, mode=%s, backend=%s)",
            self.num_cpu_blocks,
            offload_capacity / (1024**3),
            "lazy" if lazy_offload else "eager",
            "disk" if disk_capacity_bytes > 0 else "cpu",
        )

        spec_config = vllm_config.speculative_config
        use_eagle_block_drop = (
            spec_config is not None and spec_config.use_eagle_block_drop()
        )
        self.cpu_coordinator: KVCacheCoordinator = get_kv_cache_coordinator(
            kv_cache_config=self.cpu_kv_cache_config,
            max_model_len=vllm_config.model_config.max_model_len,
            max_in_flight_tokens=vllm_config.max_in_flight_tokens,
            use_eagle=use_eagle_block_drop,
            enable_caching=True,
            enable_kv_cache_events=self.enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=1,
            scheduler_block_size=self.scheduler_block_size,
            num_prefill_lookahead=vllm_config.num_prefill_lookahead_tokens,
        )
        self.group_block_sizes = self.cpu_coordinator.group_block_sizes
        self.cpu_block_pool: BlockPool = self.cpu_coordinator.block_pool
        # GPU block pool reference - bound after scheduler builds kv_cache_manager
        self._gpu_block_pool: BlockPool | None = None

        # Load metadata
        self._reqs_to_load: dict[str, LoadRequestState] = {}
        # Inverse map: load_event_idx -> req_ids. Keyed by load_event_idx because
        # the worker reports completions by event index, not request id.
        self._load_event_to_reqs: dict[int, list[str]] = {}

        # Pending (cpu_hit_blocks, hit_length, num_computed_tokens) tuples from
        # find_longest_cache_hit, kept pinned via touch() while awaiting
        # update_state_after_alloc(). ``num_computed_tokens`` is the local
        # prefix the hit was resolved against; the load path needs it to place
        # external blocks and must not re-derive it from block metadata.
        self._pending_cpu_hits: dict[
            str, tuple[tuple[list[KVCacheBlock], ...], int, int]
        ] = {}

        # Store metadata
        self._lazy_mode = lazy_offload
        # Lazy mode: use a cursor to track the last scanned block in the GPU free queue.
        self._cursor: KVCacheBlock | None = None
        if self._lazy_mode:
            self._target_free = self._estimate_lazy_target_blocks(
                kv_cache_config,
                vllm_config.scheduler_config.max_num_batched_tokens,
                self.cp_world_size,
            )
        else:
            self._target_free = 0
        self._store_event_to_blocks: dict[int, TransferMeta] = {}
        self._abandoned_store_event_to_blocks: dict[int, TransferMeta] = {}
        # Eager mode only
        self._reqs_to_store: dict[str, StoreRequestState] = {}
        self._store_event_to_reqs: dict[int, list[str]] = {}
        # (group_id, gpu_block_id) of the blocks with a store in flight.
        self._in_flight_store_gpu_blocks: set[tuple[int, int]] = set()
        self._pending_finished_stores: list[TransferMeta] = []
        self._abandoned_reqs_to_load: dict[str, LoadRequestState] = {}

        # Event counters
        self._load_event_counter: int = 0
        self._store_event_counter: int = 0
        self.boundary_store_stats = BoundaryStoreStats()
        # Interval stats state drained by get_stats()
        self._boundary_stats_snapshot = BoundaryStoreStats()
        self._interval_load_blocks_completed = 0

        # For TP/PP: track partial store completions across steps.
        # Events must be reported by all world_size workers before considered complete.
        self._expected_worker_count = vllm_config.parallel_config.world_size
        self._store_event_pending_counts: dict[int, int] = {}

    @staticmethod
    def _derive_cpu_config(
        gpu_config: "KVCacheConfig",
        cpu_capacity_bytes: int,
        page_bytes: int,
        num_cpu_blocks: int | None = None,
        aligned_num_cpu_blocks: int | None = None,
    ) -> "KVCacheConfig":
        """Derive a CPU KVCacheConfig from the GPU config.

        Same kv_cache_groups and large pages as the GPU pool, so a CPU block
        has the GPU block geometry; one page costs ``page_bytes`` of offload
        capacity (O_DIRECT-padded for the disk file). An explicit
        ``num_cpu_blocks`` is checked against local capacity.
        ``aligned_num_cpu_blocks`` comes from the live worker allocations and
        is used as-is when no explicit count is configured.
        """
        local_capacity = cpu_capacity_bytes // page_bytes
        if num_cpu_blocks is None:
            num_cpu_blocks = (
                aligned_num_cpu_blocks
                if aligned_num_cpu_blocks is not None
                else max(1, local_capacity)
            )
        elif (
            type(num_cpu_blocks) is not int or not 0 < num_cpu_blocks <= local_capacity
        ):
            raise ValueError(
                f"num_cpu_blocks={num_cpu_blocks!r} must be between 1 and "
                f"local offload capacity {local_capacity}"
            )
        # The worker builds the CPU arena itself; the scheduler side never
        # reads tensors.
        pool_bytes = num_cpu_blocks * gpu_config.large_page_bytes
        return replace(
            gpu_config,
            num_pages=num_cpu_blocks,
            kv_cache_groups=[
                replace(
                    g,
                    num_blocks=pool_bytes // gpu_config.group_geometry(i).block_bytes,
                )
                for i, g in enumerate(gpu_config.kv_cache_groups)
            ],
            kv_cache_tensors=[],
        )

    @staticmethod
    def _estimate_lazy_target_blocks(
        kv_cache_config: "KVCacheConfig",
        max_num_batched_tokens: int,
        cp_world_size: int = 1,
    ) -> int:
        """GPU blocks to keep available (free/offloaded) per step in lazy mode."""
        WATERMARK_RATIO = 1.0  # Reserve larger space to avoid running out of GPU blocks
        target = 0
        for g in kv_cache_config.prefix_cacheable_groups:
            spec = g.kv_cache_spec
            block_size = resolve_dcp_kv_block_size(spec, cp_world_size)
            if isinstance(spec, MambaSpec):
                target += 2
            elif isinstance(spec, SlidingWindowSpec):
                target += cdiv(spec.sliding_window, block_size) + 1
            else:
                target += cdiv(max_num_batched_tokens, block_size)
        return int(target * (1 + WATERMARK_RATIO))

    def bind_gpu_block_pool(self, gpu_block_pool: BlockPool) -> None:
        """Bind GPU block pool so that we can touch blocks during stores.
        Called by Scheduler after kv_cache_manager is ready."""
        self._gpu_block_pool = gpu_block_pool

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        """Return (num_new_tokens, is_async) from consecutive CPU cache hits."""
        # Pins found CPU blocks so they survive LRU eviction until
        # update_state_after_alloc() consumes them. Any pin from an earlier
        # call on the same request (e.g. retry after a failed allocate_slots)
        # is dropped first.
        if stale := self._pending_cpu_hits.pop(request.request_id, None):
            self._free_pending_cpu_hit(stale)

        if request.skip_reading_prefix_cache:
            return 0, False

        if num_computed_tokens % self.scheduler_block_size != 0:
            # Transfers are whole-block copies, so an external suffix cannot
            # start in the middle of a destination block. Local hits are
            # hash-aligned, and the hash block size is finer than the
            # scheduler block size when a group keeps a private block size
            # (e.g. none-mode Mamba); such a request keeps its local hit and
            # skips the external one.
            logger.warning_once(
                "SimpleCPU external lookup requires scheduler-block-aligned "
                "local tokens, got %d tokens with scheduler block size %d.",
                num_computed_tokens,
                self.scheduler_block_size,
            )
            return 0, False

        num_skipped_hashes = num_computed_tokens // self.block_size
        remaining_hashes = request.block_hashes[num_skipped_hashes:]

        if not remaining_hashes:
            return 0, False
        # Must recompute at least the last token, matching the logic in
        # kv_cache_manager.get_computed_blocks().
        max_hit_len = request.num_tokens - 1 - num_computed_tokens
        if max_hit_len <= 0:
            return 0, False
        cpu_hit_blocks, hit_length, _ = self.cpu_coordinator.find_longest_cache_hit(
            remaining_hashes, max_hit_len
        )

        if hit_length > 0:
            pin_blocks = [
                blk for grp in cpu_hit_blocks for blk in grp if not blk.is_null
            ]
            self.cpu_block_pool.touch(pin_blocks)
            self._pending_cpu_hits[request.request_id] = (
                cpu_hit_blocks,
                hit_length,
                num_computed_tokens,
            )
            return hit_length, True
        return 0, False

    # TODO(yifan): this API now only matches the suffix part of the prefix cache. A more
    # general API should scan blocks in both GPU and CPU block pool in a single pass.
    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        req_id = request.request_id
        block_ids_by_group = blocks.get_block_ids()
        num_groups = len(block_ids_by_group)

        # Store tracking (eager mode only). Register the request;
        # block IDs are accumulated from scheduler_output in
        # _prepare_eager_store_specs via yield_req_data.
        if not self._lazy_mode and req_id not in self._reqs_to_store:
            self._reqs_to_store[req_id] = StoreRequestState(
                request=request,
                block_ids=tuple([] for _ in range(num_groups)),
                num_stored_blocks=[0] * num_groups,
            )

        # Pop the CPU hit cached by get_num_new_matched_tokens(). The
        # found blocks were pinned there to survive LRU eviction in the window
        # between get_num_new_matched_tokens() and this matching call.
        pending = self._pending_cpu_hits.pop(req_id, None)

        if num_external_tokens == 0:
            if pending is not None:
                logger.warning(
                    "SimpleCPUOffloadScheduler: update_state_after_alloc "
                    "called for req_id=%s with no external tokens but "
                    "get_num_new_matched_tokens() unexpectedly recorded "
                    "a pending CPU hit; releasing the stale pin.",
                    req_id,
                )
                self._free_pending_cpu_hit(pending)
            return

        if pending is None:
            logger.warning(
                "SimpleCPUOffloadScheduler: update_state_after_alloc called "
                "for req_id=%s with num_external_tokens=%d but no pending "
                "CPU hit from get_num_new_matched_tokens(); skipping load.",
                req_id,
                num_external_tokens,
            )
            return

        cpu_hit_blocks_full, _, num_computed_tokens = pending

        assert num_computed_tokens % self.scheduler_block_size == 0, (
            "SimpleCPU external loads must start at a scheduler-block boundary"
        )
        # External hits end on a hash boundary, which every prefix-cacheable
        # group can address.
        assert num_external_tokens % self.block_size == 0, (
            f"num_external_tokens={num_external_tokens} is not aligned to "
            f"block_size={self.block_size}"
        )

        # The scheduler may have accepted fewer blocks than
        # get_num_new_matched_tokens() reported.
        # (e.g. due to token budget in test_partial_gpu_prefix_plus_cpu_load).
        # Take only the leading N blocks per group matching num_external_tokens;
        # the rest will be released along with the temp pin below.
        cpu_hit_blocks: list[list[KVCacheBlock]] = []
        for g in range(num_groups):
            if g not in self.prefix_cacheable_group_ids:
                cpu_hit_blocks.append([])
                continue
            g_block_size = self.group_block_sizes[g]
            n_take_g = cdiv(num_external_tokens, g_block_size)
            cpu_hit_blocks.append(cpu_hit_blocks_full[g][:n_take_g])

        gpu_block_ids: list[int] = []
        cpu_block_ids: list[int] = []
        group_ids: list[int] = []
        cpu_blocks_to_touch: list[KVCacheBlock] = []

        for g in range(num_groups):
            cpu_blocks_g = cpu_hit_blocks[g]
            n_ext_g = len(cpu_blocks_g)
            if n_ext_g == 0:
                continue

            g_block_size = self.group_block_sizes[g]
            gpu_ext_start = num_computed_tokens // g_block_size
            group_gpu_ids = block_ids_by_group[g]

            for i, cpu_blk in enumerate(cpu_blocks_g):
                # Skip null blocks (e.g. sliding window or mamba padding).
                if cpu_blk.is_null:
                    continue
                gpu_block_ids.append(group_gpu_ids[gpu_ext_start + i])
                cpu_block_ids.append(cpu_blk.block_id)
                group_ids.append(g)
                cpu_blocks_to_touch.append(cpu_blk)

        # Touch CPU blocks to prevent eviction during async load.
        self.cpu_block_pool.touch(cpu_blocks_to_touch)
        # Release the temporary pin held since get_num_new_matched_tokens().
        self._free_pending_cpu_hit(pending)

        # Touch GPU blocks to prevent freeing during async load
        assert self._gpu_block_pool is not None
        self._gpu_block_pool.touch(
            _pool_blocks(self._gpu_block_pool, group_ids, gpu_block_ids)
        )

        assert self._reqs_to_load.get(req_id) is None
        self._reqs_to_load[req_id] = LoadRequestState(
            request=request,
            transfer_meta=TransferMeta(gpu_block_ids, cpu_block_ids, group_ids),
        )

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> SimpleCPUOffloadMetadata:
        # --- Stores ---
        store_event = -1
        store, store_req_ids = self.prepare_store_specs(scheduler_output)
        for transfer in self._pending_finished_stores:
            store.gpu_block_ids.extend(transfer.gpu_block_ids)
            store.cpu_block_ids.extend(transfer.cpu_block_ids)
            store.group_ids.extend(transfer.group_ids)
            if store.block_meta is not None:
                assert transfer.block_meta is not None
                store.block_meta.extend(transfer.block_meta)
                assert store.secondary_meta is not None
                assert transfer.secondary_meta is not None
                store.secondary_meta.update(transfer.secondary_meta)
        self._pending_finished_stores.clear()

        if store.gpu_block_ids:
            store_event = self._store_event_counter
            self._store_event_counter += 1
            self._store_event_to_blocks[store_event] = store
            if store_req_ids:  # For eager mode only, track req->blocks mapping
                self._store_event_to_reqs[store_event] = store_req_ids
                for req_id in store_req_ids:
                    store_state = self._reqs_to_store.get(req_id)
                    if store_state is not None:
                        store_state.store_events.add(store_event)

        # --- Loads ---
        load_event = -1
        load_gpu: list[int] = []
        load_cpu: list[int] = []
        load_groups: list[int] = []
        load_req_ids: list[str] = []
        for req_id, load_state in self._reqs_to_load.items():
            if load_state.load_event is not None:
                continue
            assert load_state.transfer_meta is not None
            load_gpu.extend(load_state.transfer_meta.gpu_block_ids)
            load_cpu.extend(load_state.transfer_meta.cpu_block_ids)
            load_groups.extend(load_state.transfer_meta.group_ids)
            load_req_ids.append(req_id)
        if load_req_ids:
            load_event = self._load_event_counter
            self._load_event_counter += 1
            for req_id in load_req_ids:
                self._reqs_to_load[req_id].load_event = load_event
            self._load_event_to_reqs[load_event] = load_req_ids

        result = SimpleCPUOffloadMetadata(
            load_event=load_event,
            load_gpu_blocks=load_gpu,
            load_cpu_blocks=load_cpu,
            load_group_ids=load_groups,
            load_event_to_reqs={
                event_idx: list(req_ids)
                for event_idx, req_ids in self._load_event_to_reqs.items()
            },
            store_event=store_event,
            store_gpu_blocks=store.gpu_block_ids,
            store_cpu_blocks=store.cpu_block_ids,
            store_group_ids=store.group_ids,
            need_flush=bool(scheduler_output.preempted_req_ids),
        )
        return result

    def prepare_store_specs(
        self, scheduler_output: SchedulerOutput
    ) -> tuple[TransferMeta, list[str]]:
        """Prepare the transfer of the store event and, in eager mode, the
        requests it stores for."""
        if self._lazy_mode:
            return self._prepare_lazy_store_specs()
        else:
            return self._prepare_eager_store_specs(scheduler_output)

    def _prepare_lazy_store_specs(
        self,
    ) -> tuple[TransferMeta, list[str]]:
        """Single-pass cursor walk: offload cached GPU blocks near eviction.

        Walks the GPU free queue of pages from the cursor, counting the blocks
        each page holds (safe for the allocator to evict once offloaded).
        Stops when target_free blocks are covered or the CPU pool is full.
        """
        gpu_pool = self._gpu_block_pool
        if gpu_pool is None or self._target_free <= 0:
            return TransferMeta([], [], []), []

        free_queue = gpu_pool.free_block_queue
        cpu_pool = self.cpu_block_pool
        num_groups = len(self.cpu_kv_cache_config.kv_cache_groups)

        # Validate cursor: stale if its page left the free queue (allocated or
        # touched out), which unlinks it from the queue's linked list.
        if self._cursor is not None and self._cursor.prev_free_block is None:
            self._cursor = None

        num_selected = [0] * num_groups
        gpu_ids: list[int] = []
        group_ids: list[int] = []
        last_visited = self._cursor
        cpu_full_groups: set[int] = set()
        covered = 0

        for node in free_queue.iter_blocks_after(self._cursor):
            if covered >= self._target_free:
                break

            last_visited = node
            page_blocks = gpu_pool.page_blocks(node)
            covered += len(page_blocks)

            candidates = [
                block
                for block in page_blocks
                if block.block_hash is not None
                and not block.is_null
                and block.group_id not in cpu_full_groups
                and block.group_id in self.prefix_cacheable_group_ids
                and cpu_pool.cached_block_hash_to_block.get_one_block(block.block_hash)
                is None
            ]
            if candidates:
                # A page holds the blocks of one group.
                g = candidates[0].group_id
                num_blocks = self._max_cpu_blocks(num_selected, g, len(candidates))
                if num_blocks < len(candidates):
                    cpu_full_groups.add(g)
                num_selected[g] += num_blocks
                gpu_ids.extend(block.block_id for block in candidates[:num_blocks])
                group_ids.extend([g] * num_blocks)

            if len(cpu_full_groups) == len(self.prefix_cacheable_group_ids):
                break

        self._cursor = last_visited

        # Batch-allocate CPU blocks.
        if gpu_ids:
            cpu_ids = self._new_cpu_block_ids(group_ids)
            # Touch GPU blocks to prevent eviction during async copy.
            gpu_pool.touch(_pool_blocks(gpu_pool, group_ids, gpu_ids))
        else:
            cpu_ids = []

        return TransferMeta(gpu_ids, cpu_ids, group_ids), []

    def _prepare_eager_store_specs(
        self, scheduler_output: SchedulerOutput
    ) -> tuple[TransferMeta, list[str]]:
        """Identify newly computed blocks to offload from scheduler requests.

        Only considers blocks whose KV data has been **confirmed computed** by
        the GPU. Blocks from the current step are stored on a later step, or by
        the finish-time flush if the request completes first.

        Returns:
            (transfer, req_ids) for the store event. ``transfer.block_meta`` is
            None when kv cache events are disabled.

        """
        events_enabled = self.enable_kv_cache_events
        transfer = TransferMeta(
            [], [], [], [] if events_enabled else None, {} if events_enabled else None
        )
        req_ids: list[str] = []

        gpu_block_pool = self._gpu_block_pool
        if gpu_block_pool is None:
            return transfer, []
        cpu_block_pool = self.cpu_block_pool
        num_groups = len(self.cpu_kv_cache_config.kv_cache_groups)
        # Dedup against blocks already scheduled.
        in_flight = self._in_flight_store_gpu_blocks

        block_state = scheduler_output.kv_connector_block_state
        boundary_offloads = (
            block_state.boundary_state_offloads if block_state is not None else {}
        )
        stats = self.boundary_store_stats
        preempted_req_ids = scheduler_output.preempted_req_ids or ()
        for req_id, entries in boundary_offloads.items():
            # The scheduler drains handoffs without filtering by request
            # liveness. A request that finished or was preempted in this step
            # is giving up its blocks, which may already have been reallocated
            # to another request; reading them would publish unrelated KV under
            # a valid hash. Drop conservatively, as the mooncake store does.
            store_state = self._reqs_to_store.get(req_id)
            if (
                store_state is None
                or store_state.finished
                or req_id in preempted_req_ids
                or req_id in scheduler_output.finished_req_ids
            ):
                stats.published += len(entries)
                stats.dropped_request_gone += len(entries)
                continue
            scheduled_for_req = False
            hash_index: dict[BlockHash, int] = {}
            # ``boundary_tokens`` is not needed here: the cache key comes from
            # the handed-off block itself, not from the offered boundary.
            for group_id, gpu_block_id, _ in entries:
                stats.published += 1
                gpu_block = gpu_block_pool.get_block(group_id, gpu_block_id)
                admission = self._classify_store_candidate(gpu_block)
                if admission is _StoreAdmission.NULL_BLOCK:
                    stats.dropped_null_block += 1
                    continue
                if admission is _StoreAdmission.NOT_HASHED:
                    stats.dropped_not_hashed += 1
                    continue
                if admission is _StoreAdmission.IN_FLIGHT:
                    stats.skipped_in_flight += 1
                    continue
                if admission is _StoreAdmission.ALREADY_CACHED:
                    stats.skipped_already_cached += 1
                    continue
                if cpu_block_pool.get_num_free_blocks(group_id) <= 0:
                    stats.dropped_cpu_full += 1
                    logger.warning_once(
                        "SimpleCPU dropped a boundary-state handoff because "
                        "the CPU block pool is full; this boundary will not be retried."
                    )
                    continue
                cpu_block = cpu_block_pool.get_new_blocks(1, group_id)[0]
                transfer.gpu_block_ids.append(gpu_block_id)
                transfer.cpu_block_ids.append(cpu_block.block_id)
                transfer.group_ids.append(group_id)
                if transfer.block_meta is not None:
                    # Keep the metadata list index-parallel with the block ids.
                    # The hand-off block's primary hash is resolved at
                    # completion, so its event keeps the payload-less form;
                    # only the partial re-key hashes already on the block
                    # carry a payload, snapshotted while the request lives.
                    transfer.block_meta.append(None)
                    assert transfer.secondary_meta is not None
                    transfer.secondary_meta.update(
                        self._secondary_store_meta(
                            store_state.request, group_id, gpu_block, hash_index
                        )
                    )
                in_flight.add((group_id, gpu_block_id))
                gpu_block_pool.touch([gpu_block])
                stats.stored += 1
                scheduled_for_req = True
            if scheduled_for_req:
                req_ids.append(req_id)

        # Preemption frees the request's blocks. It resumes with a full new
        # block table (MRV1 resumed-cached, MRV2 NewRequestData).
        for req_id in preempted_req_ids:
            state = self._reqs_to_store.get(req_id)
            if state is not None:
                state.block_ids = tuple([] for _ in range(num_groups))
                state.num_stored_blocks = [0] * num_groups

        for req_id, new_block_id_groups, _ in yield_req_data(scheduler_output):
            state = self._reqs_to_store.get(req_id)
            if state is None or state.finished:
                continue

            if new_block_id_groups:
                for g in range(min(num_groups, len(new_block_id_groups))):
                    if new_block_id_groups[g] is not None:
                        # Accumulate new block IDs.
                        state.block_ids[g].extend(new_block_id_groups[g])

            num_new_tokens = scheduler_output.num_scheduled_tokens.get(req_id, 0)
            if num_new_tokens == 0:
                continue

            block_ids_by_group = state.block_ids
            if not block_ids_by_group:
                continue

            gpu_block_ids, group_ids, advanced_per_group, block_meta, secondary_meta = (
                self._select_eager_blocks_to_store(state, block_ids_by_group)
            )

            if gpu_block_ids:
                req_ids.append(req_id)
                transfer.gpu_block_ids.extend(gpu_block_ids)
                transfer.cpu_block_ids.extend(self._new_cpu_block_ids(group_ids))
                transfer.group_ids.extend(group_ids)
                in_flight.update(zip(group_ids, gpu_block_ids))
                if transfer.block_meta is not None:
                    assert block_meta is not None
                    transfer.block_meta.extend(block_meta)
                    assert transfer.secondary_meta is not None
                    assert secondary_meta is not None
                    transfer.secondary_meta.update(secondary_meta)

                # Touch GPU blocks to prevent freeing during async copy
                gpu_block_pool.touch(
                    _pool_blocks(gpu_block_pool, group_ids, gpu_block_ids)
                )

                logger.debug(
                    "Request %s: Scheduling store of %d blocks to CPU (%d groups)",
                    req_id,
                    len(gpu_block_ids),
                    num_groups,
                )

            # Advance per-group cursors (includes cached hits + newly stored)
            for g in range(num_groups):
                state.num_stored_blocks[g] += advanced_per_group[g]

        # A request can contribute both boundary and positional blocks to the
        # same event. Completion tracking is set-based, so keep one request ID.
        req_ids = list(dict.fromkeys(req_ids))
        return transfer, req_ids

    def _max_cpu_blocks(self, num_selected: list[int], group_id: int, need: int) -> int:
        """Most blocks of a group, up to ``need``, that the CPU pool takes on
        top of the ``num_selected`` blocks per group."""
        num_blocks = list(num_selected)
        base = num_blocks[group_id]
        low = 0
        high = min(need, self.cpu_block_pool.get_num_free_blocks(group_id) - base)
        probe = high
        while low < high:
            num_blocks[group_id] = base + probe
            if self.cpu_block_pool.can_allocate(num_blocks, {}, 0):
                low = probe
            else:
                high = probe - 1
            probe = (low + high + 1) // 2
        return low

    def _new_cpu_block_ids(self, group_ids: list[int]) -> list[int]:
        """Allocate one CPU block of each listed group, index-parallel."""
        new_blocks = {
            g: iter(self.cpu_block_pool.get_new_blocks(n, g))
            for g, n in Counter(group_ids).items()
        }
        return [next(new_blocks[g]).block_id for g in group_ids]

    def _cached_gpu_block(
        self, block_hashes: "list[BlockHash]", block_idx: int, group_id: int
    ) -> "KVCacheBlock | None":
        """Return the GPU block still cached under this group's block hash."""
        if block_idx >= len(block_hashes):
            return None
        assert self._gpu_block_pool is not None
        blocks = self._gpu_block_pool.get_cached_block(
            block_hashes[block_idx], [group_id]
        )
        return blocks[0] if blocks else None

    def _select_eager_blocks_to_store(
        self,
        state: StoreRequestState,
        block_ids_by_group: tuple[list[int], ...],
    ) -> tuple[
        list[int],
        list[int],
        list[int],
        list[BlockStoreMeta | None] | None,
        dict[BlockHashWithGroupId, BlockStoreMeta] | None,
    ]:
        """Return confirmed eager blocks, their groups, cursor advances, and
        event metadata (primary payloads per block, re-key payloads by hash)."""
        assert self._gpu_block_pool is not None
        gpu_block_ids: list[int] = []
        group_ids: list[int] = []
        num_groups = len(self.cpu_kv_cache_config.kv_cache_groups)
        advanced_per_group = [0] * num_groups
        block_meta: list[BlockStoreMeta | None] | None = (
            [] if self.enable_kv_cache_events else None
        )
        secondary_meta: dict[BlockHashWithGroupId, BlockStoreMeta] | None = (
            {} if self.enable_kv_cache_events else None
        )
        request = state.request
        confirmed_tokens = request.num_computed_tokens - request.num_output_placeholders
        # The CPU blocks the selection needs per group; the store allocates
        # them in group order.
        num_selected = [0] * num_groups
        hash_index: dict[BlockHash, int] = {}

        for g, group_gpu_ids in enumerate(block_ids_by_group):
            if g not in self.prefix_cacheable_group_ids:
                continue
            group_manager = self.cpu_coordinator.single_type_managers[g]
            if not group_manager.has_positionally_stable_blocks:
                continue
            # FIXME (yifan): handle CPU cache eviction, where
            # num_stored_blocks can be stale and omit evicted blocks in
            # the middle of the request.
            group_size = self.group_block_sizes[g]
            # Mimic the GPU KV cache manager's cacheable prefix.
            cacheable_tokens = self.cpu_coordinator.get_num_cacheable_tokens(
                confirmed_tokens, g
            )
            ready = min(len(group_gpu_ids), cacheable_tokens // group_size)
            start = state.num_stored_blocks[g]
            if ready <= start:
                continue
            limit = self._max_cpu_blocks(num_selected, g, ready - start)
            if limit == 0:
                # A CPU hit needs every group, so the later groups' blocks
                # would serve no hit either.
                break
            curr_mm_idx = 0
            for i, gpu_block_id in enumerate(group_gpu_ids[start:ready], start=start):
                gpu_block = self._gpu_block_pool.get_block(g, gpu_block_id)
                if gpu_block.is_null:
                    # Sliding-window groups null pages that left the window
                    # before the connector sees the block table, but the
                    # retained prefix-cache tail stays hashed in the GPU free
                    # queue; store it from there.
                    recovered = self._cached_gpu_block(request.block_hashes, i, g)
                    if recovered is None:
                        advanced_per_group[g] += 1
                        continue
                    gpu_block = recovered
                    gpu_block_id = gpu_block.block_id
                if (
                    self._classify_store_candidate(gpu_block)
                    is not _StoreAdmission.READY
                ):
                    advanced_per_group[g] += 1
                    continue
                if num_selected[g] >= limit:
                    break
                primary_block_hash = gpu_block.block_hash
                assert primary_block_hash is not None
                gpu_block_ids.append(gpu_block_id)
                group_ids.append(g)
                num_selected[g] += 1
                advanced_per_group[g] += 1
                if block_meta is not None:
                    meta, curr_mm_idx = _block_store_meta(
                        request, i, group_size, curr_mm_idx
                    )
                    block_meta.append(meta)
                    assert secondary_meta is not None
                    secondary_meta.update(
                        self._secondary_store_meta(request, g, gpu_block, hash_index)
                    )

        return gpu_block_ids, group_ids, advanced_per_group, block_meta, secondary_meta

    def _secondary_store_meta(
        self,
        request: "Request",
        group_id: int,
        gpu_block: "KVCacheBlock",
        hash_index: "dict[BlockHash, int]",
    ) -> dict[BlockHashWithGroupId, BlockStoreMeta]:
        """Event payloads for the partial re-key hashes a stored block carries.

        Mamba-align checkpoints re-key a block mid-chunk (cache_partial_block),
        adding hashes beyond its primary. Their BlockStored payload is only
        resolvable while the storing request is alive, so snapshot it at
        selection; hashes registered later fall back to a payload-less event
        at completion.

        Args:
            request: The storing request.
            group_id: The group of the stored block.
            gpu_block: The stored block.
            hash_index: The position of each of the request's block hashes.
                Filled on first use; share it across the blocks of a request
                in one selection.

        """
        assert self._gpu_block_pool is not None
        hashes = self._gpu_block_pool.cached_block_hashes_by_block.get(
            (group_id, gpu_block.block_id),
            (),  # type: ignore[call-overload]
        )
        if not hashes:
            return {}
        if not hash_index:
            hash_index.update((bh, k) for k, bh in enumerate(request.block_hashes))
        meta: dict[BlockHashWithGroupId, BlockStoreMeta] = {}
        for block_hash in hashes:
            k = hash_index.get(get_block_hash(block_hash))
            if k is None:
                # Not the storing request's hash (registered by an earlier
                # request sharing the prefix); the payload-less fallback at
                # completion covers it.
                continue
            meta[block_hash], _ = _block_store_meta(request, k, self.block_size, 0)
        return meta

    def _secondary_store_events(
        self,
        secondary_hashes: Iterable[BlockHashWithGroupId],
        secondary_meta: dict[BlockHashWithGroupId, BlockStoreMeta] | None,
    ) -> list[tuple[BlockHashWithGroupId, BlockStoreMeta | None, int]]:
        """(hash, meta, block_size) for each partial re-key hash of a block.

        Hashes whose payload was snapshotted at selection time emit with
        ``self.block_size`` and their token ids; all others keep the
        payload-less form.
        """
        meta_by_hash = secondary_meta or {}
        events = []
        for block_hash in secondary_hashes:
            meta = meta_by_hash.get(block_hash)
            block_size = self.block_size if meta is not None else 0
            events.append((block_hash, meta, block_size))
        return events

    def _classify_store_candidate(self, gpu_block: "KVCacheBlock") -> _StoreAdmission:
        if gpu_block.is_null:
            return _StoreAdmission.NULL_BLOCK
        if gpu_block.block_hash is None:
            return _StoreAdmission.NOT_HASHED
        if (gpu_block.group_id, gpu_block.block_id) in self._in_flight_store_gpu_blocks:
            return _StoreAdmission.IN_FLIGHT
        if (
            self.cpu_block_pool.cached_block_hash_to_block.get_one_block(
                gpu_block.block_hash
            )
            is not None
        ):
            return _StoreAdmission.ALREADY_CACHED
        return _StoreAdmission.READY

    def get_boundary_store_stats(self) -> BoundaryStoreStats:
        return replace(self.boundary_store_stats)

    def get_stats(self) -> SimpleCPUOffloadStats:
        """Drain per-step stats for the connector's stats hooks."""
        stats = SimpleCPUOffloadStats()

        current = self.boundary_store_stats
        for outcome, field_name in OUTCOME_TO_FIELD.items():
            delta = getattr(current, field_name) - getattr(
                self._boundary_stats_snapshot, field_name
            )
            if delta > 0:
                stats.increase_counter(MetricName.SAVE_OUTCOMES, delta, (outcome,))
        self._boundary_stats_snapshot = replace(current)

        if self._interval_load_blocks_completed:
            stats.increase_counter(
                MetricName.LOAD_BLOCKS,
                self._interval_load_blocks_completed,
            )
            self._interval_load_blocks_completed = 0

        stats.set_gauge(
            MetricName.USED_BLOCKS,
            self.num_cpu_blocks - self.cpu_block_pool.free_block_queue.num_free_blocks,
        )
        pending = sum(
            len(t.cpu_block_ids)
            for t in (
                *self._store_event_to_blocks.values(),
                *self._pending_finished_stores,
                *self._abandoned_store_event_to_blocks.values(),
            )
        )
        stats.set_gauge(MetricName.PENDING_STORE_BLOCKS, pending)
        stats.set_gauge(MetricName.INFO, 1, self._info_labelvalues)
        return stats

    def update_connector_output(self, connector_output: KVConnectorOutput) -> None:
        """Handle async transfer completions from worker.

        Load completions arrive via finished_recving (real req_ids).
        Store completions arrive via kv_connector_worker_meta as
        per-event worker counts. We accumulate across steps and process
        a store event only when all workers have reported completion.
        """
        # --- Load completions ---
        for req_id in list(connector_output.finished_recving or []):
            completed_blocks = self._cleanup_load_request(req_id)
            if completed_blocks:
                self._interval_load_blocks_completed += completed_blocks

        # --- Store completions ---
        meta = connector_output.kv_connector_worker_meta
        if not isinstance(meta, SimpleCPUOffloadWorkerMetadata):
            return
        for event_idx, count in meta.completed_store_events.items():
            total = self._store_event_pending_counts.get(event_idx, 0) + count
            if total >= self._expected_worker_count:
                self._store_event_pending_counts.pop(event_idx, None)
                self._process_store_event(event_idx)
            else:
                self._store_event_pending_counts[event_idx] = total

    def _process_store_event(self, event_idx: int) -> None:
        """Process a fully-completed store event."""
        transfer = self._store_event_to_blocks.pop(event_idx, None)
        if transfer is None:
            transfer = self._abandoned_store_event_to_blocks.pop(event_idx, None)
            if transfer is None:
                return  # guard stale events from before a reset() call
            self._release_transfer_refs(transfer)
            return

        if not self._lazy_mode:
            self._in_flight_store_gpu_blocks.difference_update(
                zip(transfer.group_ids, transfer.gpu_block_ids)
            )

        self._process_store_completion(transfer)
        logger.debug(
            "Store event %d completed: cached %d blocks to CPU",
            event_idx,
            len(transfer.cpu_block_ids),
        )

        # Eager only: update per-req state
        if not self._lazy_mode:
            for req_id in self._store_event_to_reqs.pop(event_idx, []):
                state = self._reqs_to_store.get(req_id)
                if state is None:
                    continue
                state.store_events.discard(event_idx)
                if state.finished and not state.store_events:
                    self._cleanup_store_request(req_id)

    def _process_store_completion(self, transfer: TransferMeta) -> None:
        """Register copied blocks in the CPU prefix cache and release refs."""
        gpu_block_ids, block_meta = transfer.gpu_block_ids, transfer.block_meta
        assert len(transfer.cpu_block_ids) == len(gpu_block_ids)
        assert block_meta is None or len(block_meta) == len(gpu_block_ids)

        cpu_blocks = _pool_blocks(
            self.cpu_block_pool, transfer.group_ids, transfer.cpu_block_ids
        )
        assert self._gpu_block_pool is not None
        gpu_blocks = _pool_blocks(
            self._gpu_block_pool, transfer.group_ids, gpu_block_ids
        )
        for i, (gpu_block_id, gpu_block, cpu_block) in enumerate(
            zip(gpu_block_ids, gpu_blocks, cpu_blocks)
        ):
            primary_hash = gpu_block.block_hash
            assert primary_hash is not None
            self.cpu_block_pool._insert_block_hash(
                primary_hash,
                cpu_block,
                num_tokens=gpu_block.block_hash_num_tokens,
            )
            secondary_hashes = self._gpu_block_pool.cached_block_hashes_by_block.get(
                (gpu_block.group_id, gpu_block_id), ()
            )
            for block_hash in secondary_hashes:
                self.cpu_block_pool._insert_block_hash(
                    block_hash, cpu_block, num_tokens=None
                )
            if self.enable_kv_cache_events:
                primary_group_idx = get_group_id(primary_hash)
                events_to_emit: list[
                    tuple[BlockHashWithGroupId, BlockStoreMeta | None, int]
                ] = [
                    (
                        primary_hash,
                        block_meta[i] if block_meta is not None else None,
                        self.group_block_sizes[primary_group_idx],
                    )
                ]
                events_to_emit.extend(
                    self._secondary_store_events(
                        secondary_hashes, transfer.secondary_meta
                    )
                )

                for block_hash, meta, event_block_size in events_to_emit:
                    group_idx = get_group_id(block_hash)
                    spec = self.cpu_kv_cache_config.kv_cache_groups[
                        group_idx
                    ].kv_cache_spec
                    if meta is not None and len(meta.token_ids) != event_block_size:
                        # token_ids were sliced with a different g_block_size
                        # (e.g. Mamba+DCP where capture uses spec.block_size *
                        # cp_world_size but the event emits spec.block_size * 1).
                        # Emit empty metadata until #49962 fixes the sizing.
                        token_ids = []
                        parent_block_hash = None
                        extra_keys = None
                    else:
                        token_ids = meta.token_ids if meta else []
                        parent_block_hash = meta.parent_block_hash if meta else None
                        extra_keys = meta.extra_keys if meta else None
                    self.cpu_block_pool.kv_event_queue.append(
                        BlockStored(
                            block_hashes=[
                                maybe_convert_block_hash(get_block_hash(block_hash))
                            ],
                            parent_block_hash=parent_block_hash,
                            token_ids=token_ids,
                            block_size=event_block_size,
                            lora_id=meta.lora_id if meta else None,
                            medium=self.kv_event_medium,
                            lora_name=meta.lora_name if meta else None,
                            extra_keys=to_event_extra_keys(extra_keys and [extra_keys]),
                            group_idx=group_idx,
                            kv_cache_spec_kind=get_kv_cache_spec_kind(spec).value,
                            kv_cache_spec_sliding_window=(
                                get_kv_cache_spec_sliding_window(spec)
                            ),
                            locality="LOCAL",
                        )
                    )

        # Free CPU and GPU blocks' ref counts to turn them into prefix cache
        self.cpu_block_pool.free_blocks(cpu_blocks)
        self._gpu_block_pool.free_blocks(gpu_blocks)

    def _release_transfer_refs(self, transfer: TransferMeta) -> None:
        """Release transfer refs without making copied data cacheable."""
        self.cpu_block_pool.free_blocks(
            _pool_blocks(
                self.cpu_block_pool, transfer.group_ids, transfer.cpu_block_ids
            )
        )
        assert self._gpu_block_pool is not None
        self._gpu_block_pool.free_blocks(
            _pool_blocks(
                self._gpu_block_pool, transfer.group_ids, transfer.gpu_block_ids
            )
        )

    def has_pending_stores(self) -> bool:
        """Return True if a store transfer is queued or in flight."""
        return bool(
            self._pending_finished_stores
            or self._store_event_to_blocks
            or self._abandoned_store_event_to_blocks
        )

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        """Always returns (False, None). GPU blocks are protected by ref_cnt,
        so the scheduler can free blocks immediately."""
        req_id = request.request_id

        # Release any temp CPU hit pin from get_num_new_matched_tokens()
        # if request is canceled or preempted before update_state_after_alloc()
        pending = self._pending_cpu_hits.pop(req_id, None)
        if pending is not None:
            self._free_pending_cpu_hit(pending)

        # Handle load: defer cleanup if load is in-flight
        load_state = self._reqs_to_load.get(req_id)
        if load_state is not None:
            if load_state.load_event is not None:
                load_state.finished = True  # Defer: load in-flight
            else:
                self._cleanup_load_request(req_id)

        # Handle store (eager mode only): defer cleanup if stores in-flight
        if not self._lazy_mode:
            store_state = self._reqs_to_store.get(req_id)
            if store_state is not None:
                if store_state.store_events:
                    store_state.finished = True  # Defer: stores in-flight
                else:
                    self._cleanup_store_request(req_id)

        return False, None

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        self._queue_finished_eager_store(request, block_ids)
        return self.request_finished(request, block_ids=[])

    def _queue_finished_eager_store(
        self, request: "Request", block_ids: tuple[list[int], ...]
    ) -> None:
        """Queue confirmed eager blocks omitted after a request enters decode."""
        state = self._reqs_to_store.get(request.request_id)
        gpu_pool = self._gpu_block_pool
        if state is None or gpu_pool is None:
            return
        gpu_ids, group_ids, _, block_meta, secondary_meta = (
            self._select_eager_blocks_to_store(state, block_ids)
        )
        if not gpu_ids:
            return
        cpu_ids = self._new_cpu_block_ids(group_ids)
        self._pending_finished_stores.append(
            TransferMeta(gpu_ids, cpu_ids, group_ids, block_meta, secondary_meta)
        )
        self._in_flight_store_gpu_blocks.update(zip(group_ids, gpu_ids))
        gpu_pool.touch(_pool_blocks(gpu_pool, group_ids, gpu_ids))

    def _free_pending_cpu_hit(
        self, pending: tuple[tuple[list["KVCacheBlock"], ...], int, int]
    ) -> None:
        """Release the temporary CPU block pin taken in get_num_new_matched_tokens()."""
        cpu_hit_blocks, _hit_length, _num_computed_tokens = pending
        blocks_to_free = [
            blk for grp in cpu_hit_blocks for blk in grp if not blk.is_null
        ]
        if blocks_to_free:
            self.cpu_block_pool.free_blocks(blocks_to_free)

    def _cleanup_load_request(self, req_id: str) -> int:
        """Release all load resources for a request.

        Shared between request_finished() and update_connector_output() paths.
        Removes the request from _reqs_to_load, cleans up event mappings,
        and frees CPU/GPU touch refs.

        Returns the number of blocks in the load if it had been issued to
        the worker, else 0 (never-issued loads are not counted as completed
        by the caller).
        """
        state = self._reqs_to_load.pop(req_id, None)
        if state is None:
            state = self._abandoned_reqs_to_load.pop(req_id, None)
        if state is None:
            return 0
        # Remove from load event mapping (only this req, not whole event)
        if state.load_event is not None:
            reqs = self._load_event_to_reqs.get(state.load_event)
            if reqs is not None:
                with contextlib.suppress(ValueError):
                    reqs.remove(req_id)
                if not reqs:
                    self._load_event_to_reqs.pop(state.load_event, None)

        if state.transfer_meta is not None:
            # Free CPU touch refs
            transfer = state.transfer_meta
            self.cpu_block_pool.free_blocks(
                _pool_blocks(
                    self.cpu_block_pool, transfer.group_ids, transfer.cpu_block_ids
                )
            )
            # Free GPU touch refs
            assert self._gpu_block_pool is not None
            self._gpu_block_pool.free_blocks(
                _pool_blocks(
                    self._gpu_block_pool, transfer.group_ids, transfer.gpu_block_ids
                )
            )

        if state.load_event is None:
            return 0
        return len(state.transfer_meta.gpu_block_ids)

    def _cleanup_store_request(self, req_id: str) -> None:
        """Release store metadata for a request.

        Metadata-only cleanup but no block freeing. Job completion handles
        block caching and GPU ref freeing via _process_store_completion().
        """
        state = self._reqs_to_store.pop(req_id, None)
        if state is None:
            return
        for event_idx in list(state.store_events):
            if (reqs := self._store_event_to_reqs.get(event_idx)) is not None:
                with contextlib.suppress(ValueError):
                    reqs.remove(req_id)
                if not reqs:
                    self._store_event_to_reqs.pop(event_idx, None)
        state.store_events.clear()

    def take_events(self) -> Iterable[KVCacheEvent]:
        events = self.cpu_block_pool.take_events()
        for event in events:
            if isinstance(event, BlockRemoved):
                event.medium = self.kv_event_medium
                event.locality = "LOCAL"
        return events

    def reset(self) -> bool:
        """Abandon pending transfers and reset the CPU cache when safe.

        Worker-side DMA may still be using blocks after reset is requested.
        Keep those block refs pinned until the existing completion path reports
        the transfer finished, then release refs without caching abandoned
        store results.
        """
        for pending in self._pending_cpu_hits.values():
            self._free_pending_cpu_hit(pending)
        self._pending_cpu_hits.clear()

        self._abandoned_store_event_to_blocks.update(self._store_event_to_blocks)
        for transfer in self._pending_finished_stores:
            self._release_transfer_refs(transfer)
        self._pending_finished_stores.clear()

        self._store_event_to_blocks.clear()
        self._in_flight_store_gpu_blocks.clear()

        # Loads that have not been sent to the worker cannot have running DMA.
        # In-flight loads stay pinned and are cleaned up on completion.
        for req_id in list(self._reqs_to_load):
            state = self._reqs_to_load.pop(req_id)
            if state.load_event is None:
                self._reqs_to_load[req_id] = state
                self._cleanup_load_request(req_id)
            else:
                self._abandoned_reqs_to_load[req_id] = state

        self._reqs_to_store.clear()
        self._store_event_to_reqs.clear()
        self._store_event_pending_counts = {
            event_idx: count
            for event_idx, count in self._store_event_pending_counts.items()
            if event_idx in self._abandoned_store_event_to_blocks
        }
        self._cursor = None
        # Seed the fresh counters with outcomes since the last drain so a
        # mid-interval reset() does not drop them from the next report.
        undrained = BoundaryStoreStats(
            **{
                f.name: max(
                    0,
                    getattr(self.boundary_store_stats, f.name)
                    - getattr(self._boundary_stats_snapshot, f.name),
                )
                for f in fields(BoundaryStoreStats)
            }
        )
        self.boundary_store_stats = undrained
        self._boundary_stats_snapshot = BoundaryStoreStats()
        # NOTE: _load_event_counter / _store_event_counter are not
        # reset as they are monotonic and must stay ahead of the workers
        # high-water marks to avoid event index collisions

        if self._abandoned_store_event_to_blocks or self._abandoned_reqs_to_load:
            return False

        return self.cpu_block_pool.reset_prefix_cache()
