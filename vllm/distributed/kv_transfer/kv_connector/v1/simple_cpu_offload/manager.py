# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-side manager for SimpleCPUOffloadConnector."""

import contextlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from vllm.config import VllmConfig
from vllm.distributed.kv_events import KVCacheEvent
from vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload.metadata import (
    SimpleCPUOffloadMetadata,
)
from vllm.logger import init_logger
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_coordinator import (
    KVCacheCoordinator,
    get_kv_cache_coordinator,
)
from vllm.v1.core.kv_cache_utils import (
    get_block_hash,
    make_block_hash_with_group_id,
)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import KVConnectorOutput

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.kv_cache_utils import KVCacheBlock
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass
class TransferMeta:
    gpu_block_ids: list[int]
    cpu_block_ids: list[int]
    block_hashes: list[bytes]


@dataclass
class RequestState:
    """Consolidated per-request state for CPU offloading."""

    request: "Request"
    gpu_block_ids: tuple[list[int], ...]

    # Set when request_finished is called but transfers are still in-flight.
    # Defers block cleanup to the completion handler.
    finished: bool = False

    # Load tracking
    load_transfer: TransferMeta | None = None
    load_event: int | None = None

    # Store tracking (eager mode only)
    store_events: set[int] = field(default_factory=set)
    num_stored_blocks: int = 0


class SimpleCPUOffloadScheduler:
    """Scheduler-side manager for CPU offloading."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: "KVCacheConfig | None",
        cpu_capacity_bytes: int,
        lazy_offload: bool = False,
        min_lookahead_blocks: int = 8,
    ):
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        # NOTE: We use the same block size for both GPU and CPU.
        self.block_size = vllm_config.cache_config.block_size
        # Derive a CPU KVCacheConfig from the GPU config and build a coordinator
        self.cpu_kv_cache_config = self._derive_cpu_config(
            kv_cache_config, cpu_capacity_bytes
        )
        self.num_cpu_blocks = self.cpu_kv_cache_config.num_blocks

        logger.info(
            "SimpleCPUOffloadScheduler: Allocating %d CPU blocks "
            "(%.2f GB capacity, mode=%s)",
            self.num_cpu_blocks,
            cpu_capacity_bytes / (1024**3),
            "lazy" if lazy_offload else "eager",
        )

        self.cpu_coordinator: KVCacheCoordinator = get_kv_cache_coordinator(
            kv_cache_config=self.cpu_kv_cache_config,
            max_model_len=vllm_config.model_config.max_model_len,
            use_eagle=False,
            enable_caching=True,
            enable_kv_cache_events=False,
            dcp_world_size=1,
            pcp_world_size=1,
            hash_block_size=self.block_size,
        )
        self.cpu_block_pool = self.cpu_coordinator.block_pool

        # GPU block pool reference - injected after scheduler builds kv_cache_manager
        self._gpu_block_pool: BlockPool | None = None

        # Load metadata
        self._reqs_to_load: dict[str, RequestState] = {}
        # Inverse maps: job_idx -> req_ids. Keyed by job index because the
        # worker reports completions by job index, not request id.
        self._load_job_to_reqs: dict[int, list[str]] = {}

        # Store metadata
        self._lazy_mode = lazy_offload
        # Lazy store mode only
        self._min_lookahead_blocks = min_lookahead_blocks
        self._store_event_to_blocks: dict[int, TransferMeta] = {}
        # Eager mode only
        self._reqs_to_store: dict[str, RequestState] = {}
        self._store_job_to_reqs: dict[int, list[str]] = {}

        # Event counters
        self._load_event_counter: int = 0
        self._store_event_counter: int = 0

    @staticmethod
    def _derive_cpu_config(
        gpu_config: "KVCacheConfig", cpu_capacity_bytes: int
    ) -> "KVCacheConfig":
        """Derive a CPU KVCacheConfig from the GPU config.
        Same kv_cache_groups, num_blocks scaled by CPU/GPU memory ratio."""
        # Import here to avoid potential circular imports
        from vllm.v1.kv_cache_interface import KVCacheConfig as KVCacheConfigCls
        from vllm.v1.kv_cache_interface import KVCacheTensor

        page_sizes = {
            g.kv_cache_spec.page_size_bytes for g in gpu_config.kv_cache_groups
        }
        assert len(page_sizes) == 1, (
            f"Expected uniform page_size_bytes, got {page_sizes}"
        )
        page_size_bytes = next(iter(page_sizes))
        num_tensors = len(gpu_config.kv_cache_tensors)
        assert num_tensors > 0

        num_cpu_blocks = max(1, cpu_capacity_bytes // num_tensors // page_size_bytes)
        cpu_tensors = [
            KVCacheTensor(
                size=page_size_bytes * num_cpu_blocks,
                shared_by=list(t.shared_by),
            )
            for t in gpu_config.kv_cache_tensors
        ]

        return KVCacheConfigCls(
            num_blocks=num_cpu_blocks,
            kv_cache_tensors=cpu_tensors,
            kv_cache_groups=gpu_config.kv_cache_groups,
        )

    def bind_gpu_block_pool(self, gpu_block_pool: BlockPool) -> None:
        """Inject GPU block pool so that we can touch blocks during stores.
        Called by Scheduler after kv_cache_manager is ready."""
        self._gpu_block_pool = gpu_block_pool

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        """Return (num_new_tokens, is_async) from consecutive CPU cache hits."""
        skipped = num_computed_tokens // self.block_size
        remaining_hashes = request.block_hashes[skipped:]

        if not remaining_hashes:
            return 0, False

        max_hit_len = len(remaining_hashes) * self.block_size
        _, hit_length = self.cpu_coordinator.find_longest_cache_hit(
            remaining_hashes, max_hit_len
        )

        if hit_length > 0:
            logger.debug(
                "Request %s: CPU cache hit, %d external tokens can be loaded",
                request.request_id,
                hit_length,
            )
            return hit_length, True

        return 0, False

    # TODO (yifan): this function now assumes eager offloading and only matches
    # the suffix part of the prefix cache. Another interface is needed for lazy
    # offloading, which should check prefix cache hits in GPU block pool and CPU
    # block pool in a single pass.
    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        """Prepare load metadata after GPU block allocation."""

        def _get_num_skipped_blocks(block_list: Sequence["KVCacheBlock"]) -> int:
            return next(
                (i for i, b in enumerate(block_list) if b.block_hash is None),
                len(block_list),
            )

        req_id = request.request_id
        block_ids_by_group = blocks.get_block_ids()

        # Store tracking (eager mode only). Track here because this is the only
        # place we can see all scheduled requests. With chunked prefill, this
        # method is called multiple times for the same request across scheduling
        # steps, so update gpu_block_ids if the entry already exists.
        if not self._lazy_mode:
            existing = self._reqs_to_store.get(req_id)
            if existing is not None:
                existing.gpu_block_ids = block_ids_by_group
            else:
                self._reqs_to_store[req_id] = RequestState(
                    request=request, gpu_block_ids=block_ids_by_group
                )

        if num_external_tokens == 0:
            return

        num_blocks_to_load = num_external_tokens // self.block_size
        assert num_blocks_to_load > 0

        # Skip blocks already locally computed (GPU prefix cache hits).
        # TODO (yifan): better to pass num_computed_tokens from the coordinator and
        # avoid recomputing `skipped` here.
        # The current approach assumes 1st group is full attn and its block size
        # matches the hash_block_size. Beside, it is inefficient.
        # Use group 0 (full attention) for hash slicing.
        skipped = next(
            (i for i, b in enumerate(blocks.blocks[0]) if b.block_hash is None),
            len(blocks.blocks[0]),
        )
        hashes_to_load = request.block_hashes[skipped : skipped + num_blocks_to_load]

        max_hit_len = len(hashes_to_load) * self.block_size
        cpu_hit_blocks, hit_length = self.cpu_coordinator.find_longest_cache_hit(
            hashes_to_load, max_hit_len
        )
        assert hit_length == num_external_tokens, (
            f"Expected {num_external_tokens} hit tokens, got {hit_length}"
        )

        # Block IDs are shared across groups -- use group 0 for flat ID list
        gpu_block_ids = block_ids_by_group[0][skipped : skipped + num_blocks_to_load]
        cpu_block_ids = [b.block_id for b in cpu_hit_blocks[0]]

        # Touch ALL CPU blocks across all groups to prevent eviction during
        # async load
        all_cpu_blocks = [b for group_blocks in cpu_hit_blocks for b in group_blocks]
        self.cpu_block_pool.touch(all_cpu_blocks)

        # Touch GPU blocks to prevent freeing during async load
        if self._gpu_block_pool is not None:
            self._gpu_block_pool.touch(
                [self._gpu_block_pool.blocks[bid] for bid in gpu_block_ids]
            )

        assert req_id not in self._reqs_to_load
        self._reqs_to_load[req_id] = RequestState(
            request=request,
            gpu_block_ids=block_ids_by_group,
            load_transfer=TransferMeta(gpu_block_ids, cpu_block_ids, hashes_to_load),
        )

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> SimpleCPUOffloadMetadata:
        """Build metadata for worker to execute transfers this step."""
        # --- Stores ---
        store_event = -1
        store_gpu: list[int] = []
        store_cpu: list[int] = []
        block_hashes: list[bytes] = []
        store_req_ids: list[str] = []  # For eager mode only
        if self._lazy_mode:
            # Lazy: offload GPU blocks that are evicted from the GPU block pool
            store_gpu, store_cpu, block_hashes = self._prepare_lazy_store_specs()
        else:
            (store_gpu, store_cpu, block_hashes, store_req_ids) = (
                self._prepare_eager_store_specs(scheduler_output)
            )
        if store_gpu:
            store_event = self._store_event_counter
            self._store_event_counter += 1
            self._store_event_to_blocks[store_event] = TransferMeta(
                store_gpu,
                store_cpu,
                block_hashes,
            )
            if store_req_ids:  # For eager mode only
                self._store_job_to_reqs[store_event] = store_req_ids
                for req_id in store_req_ids:
                    state = self._reqs_to_store.get(req_id)
                    if state is not None:
                        state.store_events.add(store_event)

        # --- Loads ---
        load_event = -1
        load_gpu: list[int] = []
        load_cpu: list[int] = []
        load_req_ids: list[str] = []
        for req_id, state in self._reqs_to_load.items():
            if state.load_event is not None:
                continue
            transfer = state.load_transfer
            assert transfer is not None
            load_gpu.extend(transfer.gpu_block_ids)
            load_cpu.extend(transfer.cpu_block_ids)
            load_req_ids.append(req_id)
        if load_req_ids:
            load_event = self._load_event_counter
            self._load_event_counter += 1
            for req_id in load_req_ids:
                self._reqs_to_load[req_id].load_event = load_event
            self._load_job_to_reqs[load_event] = load_req_ids

        return SimpleCPUOffloadMetadata(
            load_event=load_event,
            load_gpu_blocks=load_gpu,
            load_cpu_blocks=load_cpu,
            # NOTE: passes reference, not copy. Safe because scheduler and
            # worker run in separate processes (metadata is serialized via ZMQ).
            load_job_to_reqs=self._load_job_to_reqs,
            store_event=store_event,
            store_gpu_blocks=store_gpu,
            store_cpu_blocks=store_cpu,
        )

    def _prepare_lazy_store_specs(
        self,
    ) -> tuple[list[int], list[int], list[bytes]]:
        """Pick LRU-front GPU eviction candidates, allocate CPU slots.

        Touches GPU blocks (ref_cnt 0->1) to prevent eviction during async copy.
        On completion, update_connector_output decrements back to 0.

        Returns:
            (gpu_block_ids, cpu_block_ids, block_hashes) for the store job.
        """
        total_tokens = self.vllm_config.scheduler_config.max_num_batched_tokens
        n_lookahead = max(total_tokens // self.block_size, self._min_lookahead_blocks)
        if self._gpu_block_pool is None or n_lookahead <= 0:
            return [], [], []

        gpu_ids: list[int] = []
        block_hashes: list[bytes] = []

        num_free = self.cpu_block_pool.get_num_free_blocks()
        all_group_ids = list(range(len(self.cpu_kv_cache_config.kv_cache_groups)))

        candidates = self._gpu_block_pool.get_eviction_candidates(n_lookahead)
        for gpu_block in candidates:
            bhash_with_group = gpu_block.block_hash
            if bhash_with_group is None:
                continue
            # Extract raw block hash (strip group_id suffix)
            raw_hash = get_block_hash(bhash_with_group)
            # Check all groups in CPU cache
            cpu_blocks = self.cpu_block_pool.get_cached_block(
                raw_hash,
                kv_cache_group_ids=all_group_ids,
            )
            if cpu_blocks is not None:
                continue
            if num_free <= 0:
                break
            num_free -= 1
            gpu_ids.append(gpu_block.block_id)
            block_hashes.append(raw_hash)

        # Batch allocate CPU blocks
        if gpu_ids:
            cpu_blocks_alloc = self.cpu_block_pool.get_new_blocks(len(gpu_ids))
            cpu_ids = [blk.block_id for blk in cpu_blocks_alloc]
            # Touch GPU blocks to prevent freeing during async copy
            self._gpu_block_pool.touch(
                [self._gpu_block_pool.blocks[bid] for bid in gpu_ids]
            )
        else:
            cpu_ids = []

        return gpu_ids, cpu_ids, block_hashes

    def _prepare_eager_store_specs(
        self,
        scheduler_output: SchedulerOutput,
    ) -> tuple[list[int], list[int], list[bytes], list[str]]:
        """Identify newly computed blocks to offload.

        Returns:
            (gpu_block_ids, cpu_block_ids, block_hashes, req_ids) for the
            store job.
        """
        merged_gpu_block_ids: list[int] = []
        merged_cpu_block_ids: list[int] = []
        merged_block_hashes: list[bytes] = []
        req_ids: list[str] = []

        cpu_pool = self.cpu_block_pool
        num_free = cpu_pool.get_num_free_blocks()
        all_group_ids = list(range(len(self.cpu_kv_cache_config.kv_cache_groups)))

        for req_id, num_new_tokens in scheduler_output.num_scheduled_tokens.items():
            if num_new_tokens == 0:
                continue

            state = self._reqs_to_store.get(req_id)
            if state is None or state.finished:
                continue

            request = state.request
            total_tokens = request.num_computed_tokens + num_new_tokens
            num_full_blocks = total_tokens // self.block_size

            gpu_blocks = state.gpu_block_ids[0]
            num_available_blocks = min(
                num_full_blocks,
                len(request.block_hashes),
                len(gpu_blocks),
            )
            # FIXME (yifan): handle CPU cache eviction, where num_stored_blocks can
            # be stale and omit evicted blocks in the middle of the request.
            num_already_stored = state.num_stored_blocks
            if num_available_blocks <= num_already_stored:
                continue

            new_block_hashes = request.block_hashes[
                num_already_stored:num_available_blocks
            ]
            new_gpu_blocks = gpu_blocks[num_already_stored:num_available_blocks]

            # --- Phase 1: Scan blocks, classify as cached vs to-store ---
            gpu_block_ids: list[int] = []
            block_hashes_to_store: list[bytes] = []
            num_cached_blocks = 0

            for gpu_block_id, block_hash in zip(new_gpu_blocks, new_block_hashes):
                # Check if already cached in CPU across all groups
                cpu_blocks = cpu_pool.get_cached_block(
                    block_hash,
                    kv_cache_group_ids=all_group_ids,
                )
                if cpu_blocks is not None:
                    num_cached_blocks += 1
                    continue

                if num_free <= 0:
                    logger.debug(
                        "Request %s: CPU cache full, cannot offload block",
                        req_id,
                    )
                    break
                num_free -= 1

                gpu_block_ids.append(gpu_block_id)
                block_hashes_to_store.append(block_hash)

            # --- Phase 2: Batch allocate CPU blocks ---
            n_to_alloc = len(gpu_block_ids)
            if n_to_alloc > 0:
                cpu_blocks_alloc = cpu_pool.get_new_blocks(n_to_alloc)
                cpu_block_ids = [blk.block_id for blk in cpu_blocks_alloc]
            else:
                cpu_block_ids = []

            if cpu_block_ids:
                req_ids.append(req_id)
                merged_gpu_block_ids.extend(gpu_block_ids)
                merged_cpu_block_ids.extend(cpu_block_ids)
                merged_block_hashes.extend(block_hashes_to_store)

                # Touch GPU blocks to prevent freeing during async copy
                if self._gpu_block_pool is not None:
                    self._gpu_block_pool.touch(
                        [self._gpu_block_pool.blocks[bid] for bid in gpu_block_ids]
                    )

                logger.debug(
                    "Request %s: Scheduling store of %d blocks to CPU",
                    req_id,
                    len(cpu_block_ids),
                )

            # Advance cursor past both cached hits and newly-stored blocks
            total_advanced = num_cached_blocks + len(cpu_block_ids)
            if total_advanced > 0:
                state.num_stored_blocks = num_already_stored + total_advanced

        return (
            merged_gpu_block_ids,
            merged_cpu_block_ids,
            merged_block_hashes,
            req_ids,
        )

    def update_connector_output(
        self,
        connector_output: KVConnectorOutput,
    ) -> None:
        """Handle async transfer completions from worker.

        The worker treats load and store differently:
        - For load which blocks are tightly coupled with requests, the worker reports
            finished_recving with the request ID.
        - For store which blocks are not tightly coupled with requests, the worker
            reports finished_sending with the job index, and the scheduler should
            update request metadata accordingly.

        The connector emits job-index sentinels (__load_done_N,
        __store_done_N). We translate those back to req_ids using our
        inverse maps, process completions, and mutate
        connector_output.finished_recving with real req_ids for the
        scheduler.
        """
        # --- Load completions ---
        # Unlike stores, loads always clean up unconditionally on completion.
        # A request has at most one load job, so completion means "done."
        for req_id in list(connector_output.finished_recving or []):
            self._cleanup_load_request(req_id)

        # --- Store completions ---
        for sentinel in connector_output.finished_sending or []:
            job_idx = int(sentinel[len("__store_done_") :])

            # Both lazy and eager: process job-level blocks
            transfer = self._store_event_to_blocks.pop(job_idx)
            self._process_store_completion(
                transfer.gpu_block_ids,
                transfer.cpu_block_ids,
                transfer.block_hashes,
            )
            logger.debug(
                "Store job %d completed: cached %d blocks to CPU",
                job_idx,
                len(transfer.cpu_block_ids),
            )

            # Eager only: update per-req state
            if not self._lazy_mode:
                for req_id in self._store_job_to_reqs.pop(job_idx, []):
                    state = self._reqs_to_store.get(req_id)
                    if state is None:
                        continue
                    state.store_events.discard(job_idx)
                    if state.finished and not state.store_events:
                        self._cleanup_store_request(req_id)

        # Scheduler doesn't need finished_sending since we protect blocks with ref_cnt.
        connector_output.finished_sending = None

    def _process_store_completion(
        self,
        gpu_block_ids: list[int],
        cpu_block_ids: list[int],
        block_hashes: list[bytes],
    ) -> None:
        """Cache CPU blocks for all groups and release GPU refs."""
        assert len(block_hashes) == len(cpu_block_ids) == len(gpu_block_ids)

        num_groups = len(self.cpu_kv_cache_config.kv_cache_groups)
        cpu_blocks = [self.cpu_block_pool.blocks[bid] for bid in cpu_block_ids]

        for block_hash, cpu_block in zip(block_hashes, cpu_blocks):
            for gid in range(num_groups):
                bhash_with_gid = make_block_hash_with_group_id(block_hash, gid)
                self.cpu_block_pool.cached_block_hash_to_block.insert(
                    bhash_with_gid, cpu_block
                )
            # block_hash attr stores the last group's hash; the hash map has
            # all groups so lookups still work for any group.
            cpu_block.block_hash = bhash_with_gid

        self.cpu_block_pool.free_blocks(cpu_blocks)
        if self._gpu_block_pool is not None:
            self._gpu_block_pool.free_blocks(
                self._gpu_block_pool.blocks[bid] for bid in gpu_block_ids
            )

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        """Always returns (False, None). GPU blocks are protected by ref_cnt,
        so the scheduler can free blocks immediately."""
        req_id = request.request_id

        # Handle load: defer cleanup if load is in-flight
        load_state = self._reqs_to_load.get(req_id)
        if load_state is not None:
            if load_state.load_event is not None:
                load_state.finished = True  # Defer: load in-flight
            else:
                self._cleanup_load_request(req_id)

        # Handle store (eager mode only): defer cleanup if stores in-flight
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
        return self.request_finished(request, block_ids=[])

    def _cleanup_load_request(self, req_id: str) -> None:
        """Release all load resources for a request.

        Shared between request_finished() and update_connector_output() paths.
        Removes the request from _reqs_to_load, cleans up job mappings,
        and frees CPU/GPU touch refs.
        """
        state = self._reqs_to_load.pop(req_id, None)
        if state is None:
            return
        # Remove from load job mapping (only this req, not whole job)
        if state.load_event is not None:
            reqs = self._load_job_to_reqs.get(state.load_event)
            if reqs is not None:
                with contextlib.suppress(ValueError):
                    reqs.remove(req_id)
                if not reqs:
                    self._load_job_to_reqs.pop(state.load_event, None)
        # Free CPU touch refs
        if state.load_transfer is not None:
            self.cpu_block_pool.free_blocks(
                self.cpu_block_pool.blocks[bid]
                for bid in state.load_transfer.cpu_block_ids
            )
        # Free GPU touch refs
        if state.load_transfer is not None and self._gpu_block_pool is not None:
            self._gpu_block_pool.free_blocks(
                self._gpu_block_pool.blocks[bid]
                for bid in state.load_transfer.gpu_block_ids
            )

    def _cleanup_store_request(self, req_id: str) -> None:
        """Release store metadata for a request.

        Metadata-only cleanup — no block freeing. Job completion handles
        block caching and GPU ref freeing via _process_store_completion().
        """
        state = self._reqs_to_store.pop(req_id, None)
        if state is None:
            return
        for job_idx in list(state.store_events):
            reqs = self._store_job_to_reqs.get(job_idx)
            if reqs is not None:
                with contextlib.suppress(ValueError):
                    reqs.remove(req_id)
                if not reqs:
                    self._store_job_to_reqs.pop(job_idx, None)
        state.store_events.clear()

    def take_events(self) -> Iterable[KVCacheEvent]:
        """Return KV cache events for telemetry."""
        return self.cpu_block_pool.take_events()
