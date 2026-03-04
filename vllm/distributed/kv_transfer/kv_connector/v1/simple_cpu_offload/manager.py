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

        self.cpu_block_pool = BlockPool(
            num_gpu_blocks=self.num_cpu_blocks,
            enable_caching=True,
            hash_block_size=self.gpu_block_size,
            enable_kv_cache_events=False,
        )

        # Per-request state
        self._requests: dict[str, Request] = {}
        self._request_gpu_blocks: dict[str, list[list[int]]] = {}
        self._num_stored_blocks: dict[str, int] = {}

        # In-flight transfer tracking (lists preserve hash→block_id ordering)
        self._loading_requests: dict[str, list[BlockHash]] = defaultdict(list)
        self._storing_requests: dict[str, list[BlockHash]] = defaultdict(list)
        self._pending_load_blocks: dict[str, list[int]] = defaultdict(list)
        self._pending_cpu_blocks: dict[str, list[int]] = defaultdict(list)

        # GPU blocks with extra ref_cnt to prevent freeing during async copy
        self._pending_gpu_store_blocks: dict[str, list[int]] = defaultdict(list)

        # Load specs accumulated in update_state_after_alloc(), consumed in
        # build_connector_meta()
        self._reqs_to_load: dict[str, tuple[list[int], list[int]]] = {}

        # Monotonic job counters and job→req mappings
        self._load_job_counter: int = 0
        self._store_job_counter: int = 0
        self._req_to_load_job: dict[str, int] = {}
        self._req_to_store_jobs: dict[str, set[int]] = defaultdict(set)

        # Lazy-mode: job_idx → [(hash, cpu_block, gpu_block_id)]
        self._pending_lazy_stores: dict[
            int, list[tuple[BlockHashWithGroupId, KVCacheBlock, int]]
        ] = {}

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
        num_matched_blocks = 0
        for block_hash in request.block_hashes:
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
        """Prepare load metadata after GPU block allocation."""
        req_id = request.request_id
        self._requests[req_id] = request

        block_ids_by_group = blocks.get_block_ids()
        self._request_gpu_blocks[req_id] = [list(ids) for ids in block_ids_by_group]

        if num_external_tokens == 0:
            return

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

        self.cpu_block_pool.touch(touched_blocks)
        self._reqs_to_load[req_id] = (dst_gpu_blocks, cpu_block_ids)
        self._loading_requests[req_id].extend(loaded_block_hashes)
        self._pending_load_blocks[req_id].extend(cpu_block_ids)

    def _prepare_lazy_store_specs(
        self, n_lookahead: int
    ) -> tuple[
        list[int],
        list[int],
        list[tuple[BlockHashWithGroupId, KVCacheBlock, int]],
    ]:
        """Pick LRU-front GPU blocks, allocate CPU slots, touch GPU blocks.

        Returns:
            (gpu_block_ids, cpu_block_ids, lazy_entries) for the store job.
        """
        if self._gpu_block_pool is None or n_lookahead <= 0:
            return [], [], []

        gpu_ids: list[int] = []
        cpu_ids: list[int] = []
        lazy_entries: list[tuple[BlockHashWithGroupId, KVCacheBlock, int]] = []
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
            lazy_entries.append((bhash_with_group, cpu_block, gpu_block.block_id))

        if gpu_ids:
            self._gpu_block_pool.touch(
                [self._gpu_block_pool.blocks[bid] for bid in gpu_ids]
            )

        return gpu_ids, cpu_ids, lazy_entries

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> SimpleCPUOffloadMetadata:
        """Build metadata for worker to execute transfers this step."""
        # --- Stores ---
        store_job_idx = -1
        store_gpu: list[int] = []
        store_cpu: list[int] = []

        if self._lazy_mode:
            total_tokens = sum(scheduler_output.num_scheduled_tokens.values())
            n_lookahead = max(
                total_tokens // self.gpu_block_size,
                self._min_lookahead_blocks,
            )
            store_gpu, store_cpu, lazy_entries = self._prepare_lazy_store_specs(
                n_lookahead
            )
            if store_gpu:
                store_job_idx = self._store_job_counter
                self._store_job_counter += 1
                self._pending_lazy_stores[store_job_idx] = lazy_entries
                sentinel = f"{_LAZY_SENTINEL_PREFIX}{store_job_idx}"
                self._req_to_store_jobs[sentinel].add(store_job_idx)
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

        # Invert req→job maps into job→[req] snapshots for the connector
        pending_load_jobs: dict[int, list[str]] = defaultdict(list)
        for req_id, job_idx in self._req_to_load_job.items():
            pending_load_jobs[job_idx].append(req_id)
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
        """Identify newly computed blocks to offload. Returns {req_id: (gpu, cpu)}."""
        reqs_to_store: dict[str, tuple[list[int], list[int]]] = {}

        for req_id, num_new_tokens in scheduler_output.num_scheduled_tokens.items():
            if num_new_tokens == 0:
                continue

            gpu_blocks_by_group = self._request_gpu_blocks.get(req_id)
            if gpu_blocks_by_group is None:
                continue

            request = self._requests.get(req_id)
            if request is None:
                continue

            total_tokens = request.num_computed_tokens + num_new_tokens
            num_full_blocks = total_tokens // self.gpu_block_size

            gpu_blocks = gpu_blocks_by_group[0]
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

            cpu_block_ids = []
            src_gpu_blocks = []
            block_hashes_to_store = []
            num_cached_blocks = 0
            for src_gpu_block, block_hash in zip(new_gpu_blocks, new_block_hashes):
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
                src_gpu_blocks.append(src_gpu_block)
                cpu_block_ids.append(cpu_block.block_id)
                block_hashes_to_store.append(block_hash)

            if cpu_block_ids:
                reqs_to_store[req_id] = (src_gpu_blocks, cpu_block_ids)
                self._storing_requests[req_id].extend(block_hashes_to_store)
                self._pending_cpu_blocks[req_id].extend(cpu_block_ids)

                # Touch GPU blocks to prevent freeing during async copy
                if self._gpu_block_pool is not None:
                    self._gpu_block_pool.touch(
                        [self._gpu_block_pool.blocks[bid] for bid in src_gpu_blocks]
                    )
                self._pending_gpu_store_blocks[req_id].extend(src_gpu_blocks)

                logger.debug(
                    "Request %s: Scheduling store of %d blocks to CPU",
                    req_id,
                    len(cpu_block_ids),
                )

            # Advance cursor past both cached hits and newly-stored blocks
            total_advanced = num_cached_blocks + len(cpu_block_ids)
            if total_advanced > 0:
                self._num_stored_blocks[req_id] = num_already_stored + total_advanced

        return reqs_to_store

    def update_connector_output(
        self,
        connector_output: KVConnectorOutput,
    ) -> None:
        """Handle async transfer completions from worker."""
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

            if block_hashes and cpu_block_ids:
                if len(block_hashes) != len(cpu_block_ids):
                    logger.warning(
                        "Request %s: completed store length mismatch: "
                        "hashes=%d, cpu_blocks=%d",
                        req_id,
                        len(block_hashes),
                        len(cpu_block_ids),
                    )

                cached_blocks = []
                for block_hash, cpu_block_id in zip(block_hashes, cpu_block_ids):
                    cpu_block = self.cpu_block_pool.blocks[cpu_block_id]
                    block_hash_with_group = make_block_hash_with_group_id(
                        block_hash, group_id=0
                    )
                    cpu_block.block_hash = block_hash_with_group
                    self.cpu_block_pool.cached_block_hash_to_block.insert(
                        block_hash_with_group, cpu_block
                    )
                    cached_blocks.append(cpu_block)

                if cached_blocks:
                    self.cpu_block_pool.free_blocks(cached_blocks)
            elif cpu_block_ids:
                logger.warning(
                    "Request %s: completed store missing block hashes for %d "
                    "CPU blocks",
                    req_id,
                    len(cpu_block_ids),
                )

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

            self._cleanup_request(req_id)

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        """Always returns (False, None). GPU blocks are protected by ref_cnt,
        so the scheduler can free blocks immediately."""
        req_id = request.request_id
        if req_id not in self._storing_requests:
            self._cleanup_request(req_id)
        return False, None

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        """SupportsHMA interface — delegates to request_finished."""
        return self.request_finished(request, block_ids=[])

    def _cleanup_request(self, req_id: str) -> None:
        """Release all resources for a request (CPU blocks, GPU refs, state)."""
        self._req_to_load_job.pop(req_id, None)
        self._req_to_store_jobs.pop(req_id, None)

        pending_load_blocks = self._pending_load_blocks.pop(req_id, [])
        if pending_load_blocks:
            self.cpu_block_pool.free_blocks(
                self.cpu_block_pool.blocks[bid] for bid in pending_load_blocks
            )
        pending_cpu_blocks = self._pending_cpu_blocks.pop(req_id, [])
        if pending_cpu_blocks:
            self.cpu_block_pool.free_blocks(
                self.cpu_block_pool.blocks[bid] for bid in pending_cpu_blocks
            )

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
