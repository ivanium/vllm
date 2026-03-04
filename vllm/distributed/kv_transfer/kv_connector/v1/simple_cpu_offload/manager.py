# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-side manager for SimpleCPUOffloadConnector."""

from collections import defaultdict
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from vllm.config import VllmConfig
from vllm.distributed.kv_events import KVCacheEvent
from vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload.metadata import (
    SimpleCPUOffloadMetadata,
)
from vllm.logger import init_logger
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    BlockHashWithGroupId,
    KVCacheBlock,
    get_block_hash,
    make_block_hash_with_group_id,
)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import KVConnectorOutput

# Sentinel prefix used in finished_sending to signal completed lazy store jobs.
# Format: f"{_LAZY_SENTINEL_PREFIX}{job_idx}"
_LAZY_SENTINEL_PREFIX = "__lazy_store_"

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


class SimpleCPUOffloadScheduler:
    """
    Scheduler-side manager for CPU offloading.

    Responsibilities:
    - Maintain hash->CPU_block_id mapping via BlockPool
    - LRU eviction of CPU blocks when cache is full
    - Prepare load/store metadata for worker
    - Track in-flight async transfers
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: "KVCacheConfig | None",
        cpu_capacity_bytes: int,
        lazy_offload: bool = False,
        min_lookahead_blocks: int = 8,
    ):
        """
        Initialize the scheduler-side manager.

        Args:
            vllm_config: vLLM configuration
            kv_cache_config: KV cache configuration
            cpu_capacity_bytes: CPU memory capacity in bytes
            lazy_offload: If True, use lazy (LRU-eviction-based) offloading
                instead of eager (newly-computed block) offloading.
            min_lookahead_blocks: Minimum number of LRU candidates to consider
                per step in lazy mode (floor for lookahead window).
        """
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config

        cache_config = vllm_config.cache_config
        self.gpu_block_size = cache_config.block_size

        # Lazy vs eager mode config
        self._lazy_mode = lazy_offload
        self._min_lookahead_blocks = min_lookahead_blocks

        # GPU block pool reference — injected after scheduler builds kv_cache_manager
        self._gpu_block_pool: BlockPool | None = None

        # Calculate number of CPU blocks based on capacity
        self.num_cpu_blocks = self._calculate_num_blocks(cpu_capacity_bytes)

        logger.info(
            "SimpleCPUOffloadScheduler: Allocating %d CPU blocks "
            "(%.2f GB capacity, mode=%s)",
            self.num_cpu_blocks,
            cpu_capacity_bytes / (1024**3),
            "lazy" if lazy_offload else "eager",
        )

        # Create CPU block pool for LRU management
        # BlockPool handles hash->block mapping and LRU eviction
        self.cpu_block_pool = BlockPool(
            num_gpu_blocks=self.num_cpu_blocks,
            enable_caching=True,
            hash_block_size=self.gpu_block_size,
            enable_kv_cache_events=False,
        )

        # Track requests for accessing block_hashes during store
        self._requests: dict[str, Request] = {}

        # Track GPU blocks for each request (for store operations)
        self._request_gpu_blocks: dict[str, list[list[int]]] = {}

        # Track in-flight transfers for async completion
        # Use lists to preserve ordering for correct block hash -> cpu_block_id mapping
        self._loading_requests: dict[str, list[BlockHash]] = defaultdict(list)
        self._storing_requests: dict[str, list[BlockHash]] = defaultdict(list)

        # Track number of blocks already stored to CPU per request
        self._num_stored_blocks: dict[str, int] = {}

        # Track touched CPU block IDs while async load is in-flight.
        self._pending_load_blocks: dict[str, list[int]] = defaultdict(list)

        # Track CPU block IDs allocated for storing (before caching)
        self._pending_cpu_blocks: dict[str, list[int]] = defaultdict(list)

        # Track GPU block IDs that have been touched for in-flight stores.
        # These blocks have an extra ref_cnt to prevent freeing during async copy.
        self._pending_gpu_store_blocks: dict[str, list[int]] = defaultdict(list)

        # Metadata for current step (load operations)
        self._reqs_to_load: dict[str, tuple[list[int], list[int]]] = {}

        # Monotonic job counters
        self._load_job_counter: int = 0
        self._store_job_counter: int = 0

        # req_id -> job_idx mappings (for snapshot sent to connector)
        self._req_to_load_job: dict[str, int] = {}
        self._req_to_store_jobs: dict[str, set[int]] = defaultdict(set)

        # --- Lazy-mode tracking ---
        # Accumulates (BlockHashWithGroupId, cpu_block, gpu_block_id) for lazy
        # stores in the current step; flushed into _pending_lazy_stores at the
        # end of build_connector_meta().
        self._pending_lazy_stores_current: list[
            tuple[BlockHashWithGroupId, KVCacheBlock, int]
        ] = []
        # job_idx -> list of (BlockHashWithGroupId, cpu_block, gpu_block_id)
        # pending registration
        self._pending_lazy_stores: dict[
            int, list[tuple[BlockHashWithGroupId, KVCacheBlock, int]]
        ] = {}

    def _calculate_num_blocks(self, cpu_capacity_bytes: int) -> int:
        """Calculate number of CPU blocks based on capacity."""
        assert self.kv_cache_config is not None
        # Calculate from KV cache config
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
        """Inject the GPU block pool reference for lazy offloading.

        Must be called by the Scheduler after kv_cache_manager is ready.
        Required for lazy mode; harmless in eager mode.

        Args:
            gpu_block_pool: The GPU-side BlockPool from KVCacheManager.
        """
        self._gpu_block_pool = gpu_block_pool

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        """
        Check CPU cache for prefix matches.

        Args:
            request: The request to check
            num_computed_tokens: Number of already computed tokens

        Returns:
            Tuple of (num_matched_tokens, is_async):
            - num_matched_tokens: Number of tokens that can be loaded from CPU
            - is_async: True if tokens will be loaded asynchronously
        """
        # Count consecutive cache hits from the start
        num_matched_blocks = 0
        for block_hash in request.block_hashes:
            # Check if block exists in CPU cache
            cpu_blocks = self.cpu_block_pool.get_cached_block(
                block_hash, kv_cache_group_ids=[0]
            )
            if cpu_blocks is None:
                break
            num_matched_blocks += 1

        num_matched_tokens = num_matched_blocks * self.gpu_block_size
        num_new_tokens = max(0, num_matched_tokens - num_computed_tokens)

        if num_new_tokens > 0:
            logger.debug(
                "Request %s: CPU cache hit, %d new tokens can be loaded",
                request.request_id,
                num_new_tokens,
            )
            return num_new_tokens, True

        return 0, False

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        """
        Prepare load metadata for worker after GPU block allocation.

        Args:
            request: The request
            blocks: Allocated GPU blocks
            num_external_tokens: Number of tokens to load from CPU
        """
        req_id = request.request_id

        # Store request for later access to block_hashes
        self._requests[req_id] = request

        # Store GPU block IDs for this request (for store operations later)
        block_ids_by_group = blocks.get_block_ids()
        self._request_gpu_blocks[req_id] = [list(ids) for ids in block_ids_by_group]

        if num_external_tokens == 0:
            return

        # Determine which blocks to load from CPU
        num_blocks_to_load = num_external_tokens // self.gpu_block_size
        block_hashes_to_load = list(request.block_hashes[:num_blocks_to_load])

        gpu_block_ids = block_ids_by_group[0]
        cpu_block_ids: list[int] = []
        dst_gpu_blocks: list[int] = []
        loaded_block_hashes: list[BlockHash] = []
        touched_blocks = []
        for block_idx, block_hash in enumerate(block_hashes_to_load):
            cpu_blocks = self.cpu_block_pool.get_cached_block(
                block_hash, kv_cache_group_ids=[0]
            )
            if cpu_blocks is None:
                logger.warning(
                    "CPU cache miss for block that was expected to be cached"
                )
                continue
            cpu_block = cpu_blocks[0]
            cpu_block_ids.append(cpu_block.block_id)
            dst_gpu_blocks.append(gpu_block_ids[block_idx])
            loaded_block_hashes.append(block_hash)
            touched_blocks.append(cpu_block)

        if len(cpu_block_ids) != num_blocks_to_load:
            logger.warning(
                "Request %s: Expected %d blocks but found %d in CPU cache",
                req_id,
                num_blocks_to_load,
                len(cpu_block_ids),
            )

        if not cpu_block_ids:
            return

        # Touch to update LRU ordering while these blocks are in-flight.
        self.cpu_block_pool.touch(touched_blocks)

        # Prepare load spec (first group only for now)
        self._reqs_to_load[req_id] = (dst_gpu_blocks, cpu_block_ids)

        # Track for async completion (use extend to preserve order)
        self._loading_requests[req_id].extend(loaded_block_hashes)
        self._pending_load_blocks[req_id].extend(cpu_block_ids)

    def _prepare_lazy_store_specs(
        self, n_lookahead: int
    ) -> tuple[list[int], list[int]]:
        """Identify LRU-front GPU blocks and allocate CPU slots for them.

        Called instead of _prepare_store_specs() in lazy mode. Peeks at the
        n_lookahead eviction candidates on the GPU free queue, skips blocks
        already in CPU cache, and allocates new CPU blocks for the rest.

        The (BlockHashWithGroupId, cpu_block) pairs are stashed in
        _pending_lazy_stores_current so build_connector_meta() can record them
        keyed by job_idx once the job counter is assigned.

        Args:
            n_lookahead: Number of eviction candidates to inspect.

        Returns:
            (gpu_block_ids, cpu_block_ids) — parallel lists for the store job.
        """
        if self._gpu_block_pool is None or n_lookahead <= 0:
            return [], []

        gpu_ids: list[int] = []
        cpu_ids: list[int] = []
        candidates = self._gpu_block_pool.get_eviction_candidates(n_lookahead)

        for gpu_block in candidates:
            bhash_with_group = gpu_block.block_hash  # BlockHashWithGroupId | None
            if bhash_with_group is None:
                continue
            # Extract plain BlockHash for cpu_block_pool lookup (group_id=0)
            plain_hash = get_block_hash(bhash_with_group)
            if self.cpu_block_pool.get_cached_block(plain_hash, [0]) is not None:
                continue  # already in CPU cache
            if self.cpu_block_pool.get_num_free_blocks() == 0:
                break  # CPU pool exhausted — best-effort
            cpu_block = self.cpu_block_pool.get_new_blocks(1)[0]
            gpu_ids.append(gpu_block.block_id)
            cpu_ids.append(cpu_block.block_id)
            # Stash for later registration on job completion
            self._pending_lazy_stores_current.append(
                (bhash_with_group, cpu_block, gpu_block.block_id)
            )

        # Touch GPU blocks to prevent eviction during copy.
        if gpu_ids and self._gpu_block_pool is not None:
            self._gpu_block_pool.touch(
                [self._gpu_block_pool.blocks[bid] for bid in gpu_ids]
            )

        return gpu_ids, cpu_ids

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> SimpleCPUOffloadMetadata:
        """
        Build metadata for worker to execute transfers.

        Assigns monotonic job_idxes to this step's load/store work and
        includes a complete snapshot of all in-flight jobs so the connector
        can translate watermarks back to req_ids.

        Args:
            scheduler_output: The scheduler output for this step

        Returns:
            Metadata containing per-job block lists and in-flight snapshot
        """
        # --- Stores (mode dispatch) ---
        store_job_idx = -1
        store_gpu: list[int] = []
        store_cpu: list[int] = []

        if self._lazy_mode:
            # Lazy: offload LRU eviction candidates instead of newly-computed
            # blocks. The lookahead window is at least _min_lookahead_blocks
            # and scales with the number of tokens scheduled this step.
            total_tokens = sum(scheduler_output.num_scheduled_tokens.values())
            n_lookahead = max(
                total_tokens // self.gpu_block_size,
                self._min_lookahead_blocks,
            )
            store_gpu, store_cpu = self._prepare_lazy_store_specs(n_lookahead)
            if store_gpu:
                store_job_idx = self._store_job_counter
                self._store_job_counter += 1
                # Record lazy entries keyed by job_idx for completion handling
                self._pending_lazy_stores[store_job_idx] = (
                    self._pending_lazy_stores_current.copy()
                )
                # Register a sentinel req_id so the connector can detect
                # completion of this lazy job via the watermark mechanism.
                sentinel = f"{_LAZY_SENTINEL_PREFIX}{store_job_idx}"
                self._req_to_store_jobs[sentinel].add(store_job_idx)
            self._pending_lazy_stores_current.clear()
        else:
            # Eager: offload newly-computed full blocks from scheduled requests
            reqs_to_store = self._prepare_store_specs(scheduler_output)
            if reqs_to_store:
                store_job_idx = self._store_job_counter
                self._store_job_counter += 1
                for req_id, (gpu, cpu) in reqs_to_store.items():
                    store_gpu.extend(gpu)
                    store_cpu.extend(cpu)
                    self._req_to_store_jobs[req_id].add(store_job_idx)

        # --- Loads ---
        load_job_idx = -1
        load_gpu: list[int] = []
        load_cpu: list[int] = []
        if self._reqs_to_load:
            load_job_idx = self._load_job_counter
            self._load_job_counter += 1
            for req_id, (gpu, cpu) in self._reqs_to_load.items():
                load_gpu.extend(gpu)
                load_cpu.extend(cpu)
                self._req_to_load_job[req_id] = load_job_idx

        # --- Complete snapshot for connector translation ---
        # Invert _req_to_load_job: job_idx → [req_ids]
        pending_load_jobs: dict[int, list[str]] = defaultdict(list)
        for req_id, job_idx in self._req_to_load_job.items():
            pending_load_jobs[job_idx].append(req_id)

        # Invert _req_to_store_jobs: job_idx → [req_ids]
        pending_store_jobs: dict[int, list[str]] = defaultdict(list)
        for req_id, job_idxes in self._req_to_store_jobs.items():
            for job_idx in job_idxes:
                pending_store_jobs[job_idx].append(req_id)

        self._reqs_to_load = {}

        return SimpleCPUOffloadMetadata(
            load_job_idx=load_job_idx,
            load_gpu_blocks=load_gpu,
            load_cpu_blocks=load_cpu,
            store_job_idx=store_job_idx,
            store_gpu_blocks=store_gpu,
            store_cpu_blocks=store_cpu,
            pending_load_jobs=dict(pending_load_jobs),
            pending_store_jobs=dict(pending_store_jobs),
        )

    def _prepare_store_specs(
        self,
        scheduler_output: SchedulerOutput,
    ) -> dict[str, tuple[list[int], list[int]]]:
        """
        Identify new blocks to offload to CPU.

        This method determines which newly computed blocks should be stored
        to CPU cache. It allocates CPU blocks and prepares the transfer specs.

        Args:
            scheduler_output: The scheduler output

        Returns:
            Dict mapping request_id to (gpu_block_ids, cpu_block_ids) for storing
        """
        reqs_to_store: dict[str, tuple[list[int], list[int]]] = {}

        for req_id, num_new_tokens in scheduler_output.num_scheduled_tokens.items():
            if num_new_tokens == 0:
                continue

            # Get GPU blocks for this request
            gpu_blocks_by_group = self._request_gpu_blocks.get(req_id)
            if gpu_blocks_by_group is None:
                continue

            # Get request to access block hashes.
            request = self._requests.get(req_id)
            if request is None:
                continue

            # Calculate how many blocks are now full and sourceable on GPU.
            total_tokens = request.num_computed_tokens + num_new_tokens
            num_full_blocks = total_tokens // self.gpu_block_size

            gpu_blocks = gpu_blocks_by_group[0]  # First group
            num_available_blocks = min(
                num_full_blocks,
                len(request.block_hashes),
                len(gpu_blocks),
            )
            if num_available_blocks <= 0:
                continue

            # Requests can be preempted/recomputed. Clamp stale stored counters
            # to the currently available block span.
            prev_num_stored = self._num_stored_blocks.get(req_id, 0)
            num_already_stored = min(prev_num_stored, num_available_blocks)
            if num_already_stored != prev_num_stored:
                self._num_stored_blocks[req_id] = num_already_stored

            if num_available_blocks <= num_already_stored:
                continue

            new_block_hashes = request.block_hashes[
                num_already_stored:num_available_blocks
            ]
            new_gpu_blocks = gpu_blocks[num_already_stored:num_available_blocks]

            # Allocate CPU blocks for new blocks and keep src/dst aligned.
            cpu_block_ids = []
            src_gpu_blocks = []
            block_hashes_to_store = []
            num_cached_blocks = 0
            for src_gpu_block, block_hash in zip(new_gpu_blocks, new_block_hashes):
                # Check if already cached (skip if so)
                existing = self.cpu_block_pool.get_cached_block(
                    block_hash, kv_cache_group_ids=[0]
                )
                if existing is not None:
                    # Already cached, skip
                    num_cached_blocks += 1
                    continue

                # Avoid exception-heavy fast path once pool is saturated.
                if self.cpu_block_pool.get_num_free_blocks() <= 0:
                    logger.debug(
                        "Request %s: CPU cache full, cannot offload block",
                        req_id,
                    )
                    break

                # Try to allocate a new CPU block
                new_blocks = self.cpu_block_pool.get_new_blocks(1)
                cpu_block = new_blocks[0]
                src_gpu_blocks.append(src_gpu_block)
                cpu_block_ids.append(cpu_block.block_id)
                block_hashes_to_store.append(block_hash)

            if cpu_block_ids:
                reqs_to_store[req_id] = (src_gpu_blocks, cpu_block_ids)

                # Track for async completion (use extend to preserve order)
                self._storing_requests[req_id].extend(block_hashes_to_store)
                self._pending_cpu_blocks[req_id].extend(cpu_block_ids)

                # Touch GPU source blocks to prevent them from being freed
                # while the async copy is in progress.
                if self._gpu_block_pool is not None:
                    self._gpu_block_pool.touch(
                        [self._gpu_block_pool.blocks[bid] for bid in src_gpu_blocks]
                    )
                self._pending_gpu_store_blocks[req_id].extend(b for b in src_gpu_blocks)

                logger.debug(
                    "Request %s: Scheduling store of %d blocks to CPU",
                    req_id,
                    len(cpu_block_ids),
                )

            # Count cached hits as already stored for this request so we keep
            # moving forward across full blocks on subsequent steps.
            total_advanced = num_cached_blocks + len(cpu_block_ids)
            if total_advanced > 0:
                self._num_stored_blocks[req_id] = num_already_stored + total_advanced

        return reqs_to_store

    def update_connector_output(
        self,
        connector_output: KVConnectorOutput,
    ) -> None:
        """
        Handle async transfer completions from worker.

        Args:
            connector_output: Output from worker containing completion info
        """
        # Mark loads complete
        for req_id in connector_output.finished_recving or []:
            self._req_to_load_job.pop(req_id, None)
            block_hashes = self._loading_requests.pop(req_id, [])
            load_block_ids = self._pending_load_blocks.pop(req_id, [])
            if load_block_ids:
                self.cpu_block_pool.free_blocks(
                    self.cpu_block_pool.blocks[block_id] for block_id in load_block_ids
                )
            logger.debug(
                "Request %s: Finished loading %d blocks from CPU",
                req_id,
                len(block_hashes),
            )

        # Mark stores complete and cache the blocks
        for req_id in connector_output.finished_sending or []:
            # Handle lazy-mode sentinel completions
            if req_id.startswith(_LAZY_SENTINEL_PREFIX):
                self._req_to_store_jobs.pop(req_id, None)
                job_idx = int(req_id[len(_LAZY_SENTINEL_PREFIX) :])
                lazy_entries = self._pending_lazy_stores.pop(job_idx, None)
                if lazy_entries:
                    for bhash_with_group, cpu_block, _gpu_bid in lazy_entries:
                        cpu_block.block_hash = bhash_with_group
                        self.cpu_block_pool.cached_block_hash_to_block.insert(
                            bhash_with_group, cpu_block
                        )
                    # Release scheduler-owned refs; blocks stay cached via hash map.
                    self.cpu_block_pool.free_blocks(
                        [entry[1] for entry in lazy_entries]
                    )
                    # Decrement GPU block ref_cnt for lazy stores.
                    if self._gpu_block_pool is not None:
                        gpu_bids = [entry[2] for entry in lazy_entries]
                        self._gpu_block_pool.free_blocks(
                            self._gpu_block_pool.blocks[bid] for bid in gpu_bids
                        )
                    logger.debug(
                        "Lazy store job %d: cached %d blocks to CPU",
                        job_idx,
                        len(lazy_entries),
                    )
                continue

            self._req_to_store_jobs.pop(req_id, None)
            block_hashes = self._storing_requests.pop(req_id, [])
            cpu_block_ids = self._pending_cpu_blocks.pop(req_id, [])

            # Cache the stored blocks in the CPU pool
            if block_hashes and cpu_block_ids:
                if len(block_hashes) != len(cpu_block_ids):
                    logger.warning(
                        "Request %s: completed store length mismatch: "
                        "hashes=%d, cpu_blocks=%d",
                        req_id,
                        len(block_hashes),
                        len(cpu_block_ids),
                    )

                # Get the actual blocks and cache them
                # block_hashes is now a list preserving insertion order
                cached_blocks = []
                for block_hash, cpu_block_id in zip(block_hashes, cpu_block_ids):
                    cpu_block = self.cpu_block_pool.blocks[cpu_block_id]
                    # The block pool will cache the block with its hash
                    block_hash_with_group = make_block_hash_with_group_id(
                        block_hash, group_id=0
                    )
                    cpu_block.block_hash = block_hash_with_group
                    self.cpu_block_pool.cached_block_hash_to_block.insert(
                        block_hash_with_group, cpu_block
                    )
                    cached_blocks.append(cpu_block)

                # Release scheduler-owned references from get_new_blocks().
                # Cached blocks remain discoverable via hash map and evictable by LRU.
                if cached_blocks:
                    self.cpu_block_pool.free_blocks(cached_blocks)
            elif cpu_block_ids:
                logger.warning(
                    "Request %s: completed store missing block hashes for %d "
                    "CPU blocks",
                    req_id,
                    len(cpu_block_ids),
                )

            # Release any unmatched CPU blocks to avoid leaks on mismatch.
            if len(cpu_block_ids) > len(block_hashes):
                self.cpu_block_pool.free_blocks(
                    self.cpu_block_pool.blocks[cpu_block_id]
                    for cpu_block_id in cpu_block_ids[len(block_hashes) :]
                )

            logger.debug(
                "Request %s: Finished storing %d blocks to CPU, cached",
                req_id,
                len(block_hashes),
            )

            # Decrement GPU block ref_cnt — blocks are now safe to free.
            gpu_block_ids = self._pending_gpu_store_blocks.pop(req_id, [])
            if gpu_block_ids and self._gpu_block_pool is not None:
                self._gpu_block_pool.free_blocks(
                    self._gpu_block_pool.blocks[bid] for bid in gpu_block_ids
                )

            # finished_sending is only reported after request completion.
            self._cleanup_request(req_id)

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Called when a request has finished.

        Args:
            request: The finished request
            block_ids: Block IDs being freed

        Returns:
            Tuple of (is_async, kv_transfer_params)
        """
        req_id = request.request_id

        # Check if storing in progress
        is_async = req_id in self._storing_requests

        # Cleanup if not async (otherwise cleaned up in update_connector_output)
        if not is_async:
            self._cleanup_request(req_id)

        return is_async, None

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Called when a request has finished for all KV cache groups.
        (SupportsHMA interface)

        Args:
            request: The finished request
            block_ids: Block IDs being freed for each group

        Returns:
            Tuple of (is_async, kv_transfer_params)
        """
        req_id = request.request_id
        is_async = req_id in self._storing_requests

        # Cleanup if not async
        if not is_async:
            self._cleanup_request(req_id)

        return is_async, None

    def _cleanup_request(self, req_id: str) -> None:
        """Clean up all state for a request."""
        self._req_to_load_job.pop(req_id, None)
        self._req_to_store_jobs.pop(req_id, None)
        pending_load_blocks = self._pending_load_blocks.pop(req_id, [])
        if pending_load_blocks:
            self.cpu_block_pool.free_blocks(
                self.cpu_block_pool.blocks[block_id] for block_id in pending_load_blocks
            )

        pending_cpu_blocks = self._pending_cpu_blocks.pop(req_id, [])
        if pending_cpu_blocks:
            self.cpu_block_pool.free_blocks(
                self.cpu_block_pool.blocks[block_id] for block_id in pending_cpu_blocks
            )

        # Release any GPU block refs held for in-flight stores.
        gpu_block_ids = self._pending_gpu_store_blocks.pop(req_id, [])
        if gpu_block_ids and self._gpu_block_pool is not None:
            self._gpu_block_pool.free_blocks(
                self._gpu_block_pool.blocks[bid] for bid in gpu_block_ids
            )

        self._requests.pop(req_id, None)
        self._request_gpu_blocks.pop(req_id, None)
        self._num_stored_blocks.pop(req_id, None)
        self._loading_requests.pop(req_id, None)
        self._storing_requests.pop(req_id, None)
        self._reqs_to_load.pop(req_id, None)

    def take_events(self) -> Iterable[KVCacheEvent]:
        """Return KV cache events for telemetry."""
        return self.cpu_block_pool.take_events()
