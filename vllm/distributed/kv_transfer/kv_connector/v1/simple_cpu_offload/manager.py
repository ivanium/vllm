# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-side manager for SimpleCPUOffloadConnector."""

import contextlib
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from vllm.config import VllmConfig
from vllm.distributed.kv_events import KVCacheEvent
from vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload.metadata import (
    SimpleCPUOffloadMetadata,
)
from vllm.logger import init_logger
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import (
    BlockHashWithGroupId,
    get_block_hash,
    make_block_hash_with_group_id,
)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import KVConnectorOutput

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass
class ReqTransferMeta:
    gpu_block_ids: list[int]
    cpu_block_ids: list[int]
    block_hashes: list[BlockHashWithGroupId]


@dataclass
class RequestState:
    """Consolidated per-request state for CPU offloading."""

    request: "Request"
    gpu_block_ids: tuple[list[int], ...]

    # Set when request_finished is called but transfers are still in-flight.
    # Defers block cleanup to the completion handler.
    finished: bool = False

    # Load tracking
    load_transfer: ReqTransferMeta | None = None
    load_job_idx: int | None = None

    # Store tracking (eager mode only)
    store_job_idxs: set[int] = field(default_factory=set)
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

        cache_config = vllm_config.cache_config
        self.gpu_block_size = cache_config.block_size
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
        # Job-level block data: job_idx -> (gpu_ids, cpu_ids, block_hashes).
        # Used by both lazy and eager store paths for completion processing.
        self.store_job_to_blocks: dict[
            int, tuple[list[int], list[int], list[BlockHashWithGroupId]]
        ] = {}  # TODO (yifan): use ReqTransferMeta instead

        # Eager mode only
        self._reqs_to_store: dict[str, RequestState] = {}
        self._store_job_to_reqs: dict[int, list[str]] = {}

        # Job counters
        self._load_job_counter: int = 0
        self._store_job_counter: int = 0

        # Calculate number of CPU blocks based on capacity
        self.num_cpu_blocks = self._calculate_num_blocks(cpu_capacity_bytes)

        logger.info(
            "SimpleCPUOffloadScheduler: Allocating %d CPU blocks "
            "(%.2f GB capacity, mode=%s)",
            self.num_cpu_blocks,
            cpu_capacity_bytes / (1024**3),
            "lazy" if lazy_offload else "eager",
        )

        self.cpu_block_pool = BlockPool(
            num_gpu_blocks=self.num_cpu_blocks,
            enable_caching=True,
            hash_block_size=self.gpu_block_size,
            enable_kv_cache_events=False,
        )

    def _calculate_num_blocks(self, cpu_capacity_bytes: int) -> int:
        assert self.kv_cache_config is not None
        page_sizes = {
            group.kv_cache_spec.page_size_bytes
            for group in self.kv_cache_config.kv_cache_groups
        }
        assert len(page_sizes) == 1
        page_size_bytes = next(iter(page_sizes))
        num_tensors = len(self.kv_cache_config.kv_cache_tensors)
        assert num_tensors > 0

        return max(1, cpu_capacity_bytes // num_tensors // page_size_bytes)

    def bind_gpu_block_pool(self, gpu_block_pool: BlockPool) -> None:
        """Inject GPU block pool. Called by Scheduler after kv_cache_manager
        is ready. Required for ref_cnt-based block protection during stores."""
        self._gpu_block_pool = gpu_block_pool

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        """Return (num_new_tokens, is_async) from consecutive CPU cache hits."""
        # Skip blocks already locally computed (GPU prefix cache hits).
        skipped = num_computed_tokens // self.gpu_block_size

        num_matched_blocks = 0
        for block_hash in request.block_hashes[skipped:]:
            # TODO (yifan): support HMA
            cpu_blocks = self.cpu_block_pool.get_cached_block(
                block_hash, kv_cache_group_ids=[0]
            )
            if cpu_blocks is None:
                break
            num_matched_blocks += 1

        num_external_tokens = num_matched_blocks * self.gpu_block_size
        if num_external_tokens > 0:
            logger.debug(
                "Request %s: CPU cache hit, %d external tokens can be loaded",
                request.request_id,
                num_external_tokens,
            )
            return num_external_tokens, True

        return 0, False

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        """Prepare load metadata after GPU block allocation."""
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

        num_blocks_to_load = num_external_tokens // self.gpu_block_size
        assert num_blocks_to_load > 0, (
            f"Request {request.request_id}: No blocks to load for "
            f"{num_external_tokens} external tokens"
        )

        # TODO (yifan): support HMA
        # Skip blocks already locally computed (GPU prefix cache hits).
        skipped = sum(block.block_hash is not None for block in blocks.blocks[0])
        block_hashes_to_load = list(
            request.block_hashes[skipped : skipped + num_blocks_to_load]
        )

        # TODO (yifan): support HMA
        gpu_block_ids = block_ids_by_group[0][skipped : skipped + num_blocks_to_load]
        cpu_block_ids: list[int] = []
        touched_blocks = []
        for block_hash in block_hashes_to_load:
            # TODO (yifan): support HMA
            cpu_blocks = self.cpu_block_pool.get_cached_block(
                block_hash, kv_cache_group_ids=[0]
            )
            assert cpu_blocks is not None, (
                f"Request {request.request_id}: CPU cache miss for block "
                f"{block_hash} that was expected to be cached"
            )
            cpu_block = cpu_blocks[0]
            cpu_block_ids.append(cpu_block.block_id)
            touched_blocks.append(cpu_block)

        # Touch CPU blocks to prevent eviction during async load
        self.cpu_block_pool.touch(touched_blocks)
        # Touch GPU blocks to prevent freeing during async load
        if self._gpu_block_pool is not None:
            self._gpu_block_pool.touch(
                [self._gpu_block_pool.blocks[bid] for bid in gpu_block_ids]
            )

        # External KV cache hits. Track the request for load
        assert req_id not in self._reqs_to_load
        self._reqs_to_load[req_id] = RequestState(
            request=request,
            gpu_block_ids=block_ids_by_group,
            load_transfer=ReqTransferMeta(
                gpu_block_ids, cpu_block_ids, block_hashes_to_load
            ),
        )

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> SimpleCPUOffloadMetadata:
        """Build metadata for worker to execute transfers this step."""
        # --- Stores ---
        store_job_idx = -1
        store_gpu: list[int] = []
        store_cpu: list[int] = []
        block_hashes: list[BlockHashWithGroupId] = []
        store_req_ids: list[str] = []  # For eager mode only
        if self._lazy_mode:
            # Lazy: offload GPU blocks that are evicted from the GPU block pool
            store_gpu, store_cpu, block_hashes = self._prepare_lazy_store_specs()
        else:
            (store_gpu, store_cpu, block_hashes, store_req_ids) = (
                self._prepare_eager_store_specs(scheduler_output)
            )
        if store_gpu:
            store_job_idx = self._store_job_counter
            self._store_job_counter += 1
            self.store_job_to_blocks[store_job_idx] = (
                store_gpu,
                store_cpu,
                block_hashes,
            )
            if store_req_ids:  # For eager mode only
                self._store_job_to_reqs[store_job_idx] = store_req_ids
                for req_id in store_req_ids:
                    state = self._reqs_to_store.get(req_id)
                    if state is not None:
                        state.store_job_idxs.add(store_job_idx)

        # --- Loads ---
        load_job_idx = -1
        load_gpu: list[int] = []
        load_cpu: list[int] = []
        load_req_ids: list[str] = []
        for req_id, state in self._reqs_to_load.items():
            if state.load_job_idx is not None:
                continue
            transfer = state.load_transfer
            assert transfer is not None
            load_gpu.extend(transfer.gpu_block_ids)
            load_cpu.extend(transfer.cpu_block_ids)
            load_req_ids.append(req_id)
        if load_req_ids:
            load_job_idx = self._load_job_counter
            self._load_job_counter += 1
            for req_id in load_req_ids:
                self._reqs_to_load[req_id].load_job_idx = load_job_idx
            self._load_job_to_reqs[load_job_idx] = load_req_ids

        return SimpleCPUOffloadMetadata(
            load_job_idx=load_job_idx,
            load_gpu_blocks=load_gpu,
            load_cpu_blocks=load_cpu,
            # NOTE: passes reference, not copy. Safe because scheduler and
            # worker run in separate processes (metadata is serialized via ZMQ).
            load_job_to_reqs=self._load_job_to_reqs,
            store_job_idx=store_job_idx,
            store_gpu_blocks=store_gpu,
            store_cpu_blocks=store_cpu,
        )

    def _prepare_lazy_store_specs(
        self,
    ) -> tuple[list[int], list[int], list[BlockHashWithGroupId]]:
        """Pick LRU-front GPU eviction candidates, allocate CPU slots.

        Touches GPU blocks (ref_cnt 0->1) to prevent eviction during async copy.
        On completion, update_connector_output decrements back to 0.

        Returns:
            (gpu_block_ids, cpu_block_ids, block_hashes) for the store job.
        """
        total_tokens = self.vllm_config.scheduler_config.max_num_scheduled_tokens
        n_lookahead = max(
            total_tokens // self.gpu_block_size, self._min_lookahead_blocks
        )
        if self._gpu_block_pool is None or n_lookahead <= 0:
            return [], [], []

        gpu_ids: list[int] = []
        cpu_ids: list[int] = []
        block_hashes: list[BlockHashWithGroupId] = []

        candidates = self._gpu_block_pool.get_eviction_candidates(n_lookahead)
        for gpu_block in candidates:
            bhash_with_group = gpu_block.block_hash
            if bhash_with_group is None:
                continue
            plain_hash = get_block_hash(bhash_with_group)
            if self.cpu_block_pool.get_cached_block(plain_hash, [0]) is not None:
                continue
            if self.cpu_block_pool.get_num_free_blocks() == 0:
                break
            cpu_block = self.cpu_block_pool.get_new_blocks(1)[0]
            gpu_ids.append(gpu_block.block_id)
            cpu_ids.append(cpu_block.block_id)
            block_hashes.append(bhash_with_group)

        if gpu_ids:
            # Touch GPU blocks to prevent freeing during async copy
            self._gpu_block_pool.touch(
                [self._gpu_block_pool.blocks[bid] for bid in gpu_ids]
            )

        return gpu_ids, cpu_ids, block_hashes

    def _prepare_eager_store_specs(
        self,
        scheduler_output: SchedulerOutput,
    ) -> tuple[list[int], list[int], list[BlockHashWithGroupId], list[str]]:
        """Identify newly computed blocks to offload.

        Returns:
            (gpu_block_ids, cpu_block_ids, block_hashes) for the store job.
        """
        merged_gpu_block_ids: list[int] = []
        merged_cpu_block_ids: list[int] = []
        merged_block_hashes: list[BlockHashWithGroupId] = []
        req_ids: list[str] = []

        for req_id, num_new_tokens in scheduler_output.num_scheduled_tokens.items():
            if num_new_tokens == 0:
                continue

            state = self._reqs_to_store.get(req_id)
            if state is None or state.finished:
                continue

            request = state.request
            total_tokens = request.num_computed_tokens + num_new_tokens
            num_full_blocks = total_tokens // self.gpu_block_size

            # TODO (yifan): support HMA
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

            gpu_block_ids: list[int] = []
            cpu_block_ids: list[int] = []
            block_hashes_to_store: list[BlockHashWithGroupId] = []
            num_cached_blocks = 0
            for gpu_block_id, block_hash in zip(new_gpu_blocks, new_block_hashes):
                existing = self.cpu_block_pool.get_cached_block(
                    block_hash, kv_cache_group_ids=[0]
                )
                if existing is not None:
                    num_cached_blocks += 1
                    continue

                if self.cpu_block_pool.get_num_free_blocks() <= 0:
                    logger.debug(
                        "Request %s: CPU cache full, cannot offload block",
                        req_id,
                    )
                    break

                cpu_block = self.cpu_block_pool.get_new_blocks(1)[0]
                gpu_block_ids.append(gpu_block_id)
                cpu_block_ids.append(cpu_block.block_id)
                # TODO (yifan): support HMA — hardcoded group_id=0
                block_hashes_to_store.append(
                    make_block_hash_with_group_id(block_hash, 0)
                )

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
            gpu_block_ids, cpu_block_ids, bhashes_w_group = (
                self.store_job_to_blocks.pop(job_idx)
            )
            self._process_store_completion(
                gpu_block_ids, cpu_block_ids, bhashes_w_group
            )
            logger.debug(
                "Store job %d completed: cached %d blocks to CPU",
                job_idx,
                len(cpu_block_ids),
            )

            # Eager only: update per-req state
            if not self._lazy_mode:
                for req_id in self._store_job_to_reqs.pop(job_idx, []):
                    state = self._reqs_to_store.get(req_id)
                    if state is None:
                        continue
                    state.store_job_idxs.discard(job_idx)
                    if state.finished and not state.store_job_idxs:
                        self._cleanup_store_request(req_id)

        # Scheduler doesn't need finished_sending since we protect blocks with ref_cnt.
        connector_output.finished_sending = None

    def _process_store_completion(
        self,
        gpu_block_ids: list[int],
        cpu_block_ids: list[int],
        bhashes_w_group: list[BlockHashWithGroupId],
    ) -> None:
        """Cache CPU blocks and release refs for a completed store."""
        assert len(bhashes_w_group) == len(cpu_block_ids) == len(gpu_block_ids)

        cpu_blocks = [self.cpu_block_pool.blocks[bid] for bid in cpu_block_ids]
        for bhash_w_group, cpu_block in zip(bhashes_w_group, cpu_blocks):
            cpu_block.block_hash = bhash_w_group
            self.cpu_block_pool.cached_block_hash_to_block.insert(
                bhash_w_group, cpu_block
            )

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
            if load_state.load_job_idx is not None:
                load_state.finished = True  # Defer: load in-flight
            else:
                self._cleanup_load_request(req_id)

        # Handle store (eager mode only): defer cleanup if stores in-flight
        store_state = self._reqs_to_store.get(req_id)
        if store_state is not None:
            if store_state.store_job_idxs:
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
        if state.load_job_idx is not None:
            reqs = self._load_job_to_reqs.get(state.load_job_idx)
            if reqs is not None:
                with contextlib.suppress(ValueError):
                    reqs.remove(req_id)
                if not reqs:
                    self._load_job_to_reqs.pop(state.load_job_idx, None)
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
        for job_idx in list(state.store_job_idxs):
            reqs = self._store_job_to_reqs.get(job_idx)
            if reqs is not None:
                with contextlib.suppress(ValueError):
                    reqs.remove(req_id)
                if not reqs:
                    self._store_job_to_reqs.pop(job_idx, None)
        state.store_job_idxs.clear()

    def take_events(self) -> Iterable[KVCacheEvent]:
        """Return KV cache events for telemetry."""
        return self.cpu_block_pool.take_events()
