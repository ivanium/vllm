# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import islice
from typing import TYPE_CHECKING, Any

import torch

from vllm.attention.layer import Attention
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.distributed.kv_events import BlockRemoved, BlockStored, KVCacheEvent
from vllm.distributed.kv_transfer.kv_connector.utils import yield_req_data
from vllm.distributed.kv_transfer.kv_connector.v1 import (
    KVConnectorBase_V1,
    KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
    SupportsHMA,
)
from vllm.forward_context import ForwardContext
from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionBackend, AttentionMetadata
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig, SlidingWindowSpec
from vllm.v1.kv_offload.abstract import OffloadingManager
from vllm.v1.kv_offload.factory import OffloadingSpecFactory
from vllm.v1.kv_offload.mediums import GPULoadStoreSpec
from vllm.v1.kv_offload.spec import OffloadingSpec
from vllm.v1.kv_offload.worker.worker import OffloadingWorker, TransferSpec
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request

if TYPE_CHECKING:
    pass

ReqId = str
# Block IDs for all KV cache groups (HMA support)
BlockIds = tuple[list[int], ...]

logger = init_logger(__name__)


@dataclass
class OffloadingConnectorMetadata(KVConnectorMetadata):
    reqs_to_load: dict[ReqId, TransferSpec]
    reqs_to_store: dict[ReqId, TransferSpec]


class OffloadingConnector(KVConnectorBase_V1, SupportsHMA):
    """
    KV Connector for CPU offloading with HMA (Hybrid Memory Allocator) support.

    Supports hybrid models with multiple KV cache groups (e.g., Full Attention
    + Sliding Window Attention layers). Each group has independent block tracking
    and offloading, with sliding window groups only storing blocks within their
    attention window.
    """

    @property
    def prefer_cross_layer_blocks(self) -> bool:
        ## TODO (YIFAN): double check and update this
        return False

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig | None = None,
    ):
        super().__init__(vllm_config, role, kv_cache_config)

        spec = OffloadingSpecFactory.create_spec(vllm_config)
        self._is_hma_enabled = (
            not vllm_config.scheduler_config.disable_hybrid_kv_cache_manager
        )

        self.connector_scheduler: OffloadingConnectorScheduler | None = None
        self.connector_worker: OffloadingConnectorWorker | None = None
        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = OffloadingConnectorScheduler(
                spec, kv_cache_config
            )
        elif role == KVConnectorRole.WORKER:
            self.connector_worker = OffloadingConnectorWorker(spec, kv_cache_config)

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        assert self.connector_worker is not None
        self.connector_worker.register_kv_caches(kv_caches)

    def register_cross_layers_kv_cache(
        self, kv_cache: torch.Tensor, attn_backend: type[AttentionBackend]
    ):
        assert self.connector_worker is not None
        self.connector_worker.register_cross_layers_kv_cache(kv_cache, attn_backend)

    def handle_preemptions(self, preempted_req_ids: set[str]):
        assert self.connector_worker is not None
        self.connector_worker.handle_preemptions(preempted_req_ids)

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        assert self.connector_worker is not None
        assert isinstance(self._connector_metadata, OffloadingConnectorMetadata)
        self.connector_worker.start_kv_transfers(self._connector_metadata)

    def wait_for_layer_load(self, layer_name: str) -> None:
        pass

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs,
    ) -> None:
        pass

    def wait_for_save(self):
        assert self.connector_worker is not None
        assert isinstance(self._connector_metadata, OffloadingConnectorMetadata)
        self.connector_worker.prepare_store_kv(self._connector_metadata)

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        assert self.connector_worker is not None
        return self.connector_worker.get_finished(finished_req_ids)

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int, bool]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.get_num_new_matched_tokens(
            request, num_computed_tokens
        )

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        assert self.connector_scheduler is not None
        return self.connector_scheduler.update_state_after_alloc(
            request, blocks, num_external_tokens
        )

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.build_connector_meta(scheduler_output)

    def update_connector_output(self, connector_output: KVConnectorOutput):
        assert self.connector_scheduler is not None
        self.connector_scheduler.update_connector_output(connector_output)

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished(request, block_ids)

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: BlockIds,
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Called exactly once when a request has finished for all kv cache groups
        (HMA mode), before its blocks are freed for each group.

        This method is required by the SupportsHMA interface.

        Args:
            request: The finished request.
            block_ids: Block IDs for all KV cache groups, as tuple of lists.

        Returns:
            Tuple of (delay_free, metadata):
            - delay_free: True if blocks should not be freed until the request_id
              is returned from get_finished() (async store in progress).
            - metadata: Optional KVTransferParams to include in request outputs.
        """
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished_all_groups(request, block_ids)

    def take_events(self) -> Iterable[KVCacheEvent]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.take_events()


class OffloadingConnectorScheduler:
    """Implementation of Scheduler side methods with HMA support."""

    def __init__(
        self, spec: OffloadingSpec, kv_cache_config: KVCacheConfig | None = None
    ):
        self.spec = spec
        self.gpu_block_size = spec.gpu_block_size
        self.offloaded_block_size = spec.offloaded_block_size
        self.block_size_factor = self.offloaded_block_size // self.gpu_block_size
        self.kv_cache_config = kv_cache_config

        # Determine number of KV cache groups and sliding window sizes
        self.num_kv_cache_groups = 1
        self.sw_sizes: list[int] = [0]  # Sliding window size in blocks per group

        if kv_cache_config is not None:
            self.num_kv_cache_groups = len(kv_cache_config.kv_cache_groups)
            # Compute sliding window sizes in blocks for each group
            # 0 means full attention (no sliding window)
            sw_sizes_tokens = [
                (
                    group.kv_cache_spec.sliding_window
                    if isinstance(group.kv_cache_spec, SlidingWindowSpec)
                    else 0
                )
                for group in kv_cache_config.kv_cache_groups
            ]
            self.sw_sizes = [
                n_tokens // self.gpu_block_size for n_tokens in sw_sizes_tokens
            ]

        # Get manager with appropriate number of groups
        # HybridOffloadingManager provides per-group tracking
        self.manager: OffloadingManager = spec.get_manager(
            num_groups=self.num_kv_cache_groups
        )

        # Request tracking
        self._requests: dict[ReqId, Request] = {}
        # list of GPU block IDs per request (uses group 0 for backward compatibility)
        self._request_block_ids: dict[ReqId, list[int]] = {}
        # requests to load for the current scheduler step
        self._reqs_to_load: dict[ReqId, TransferSpec] = {}
        # request blocks are stored in order
        # index of next block (of size offloaded_block_size) to offload
        self._next_stored_block_idx: dict[ReqId, int] = {}

        # request ID -> set(block hashes being stored/load)
        self._reqs_being_stored = defaultdict[ReqId, set[BlockHash]](set)
        self._reqs_being_loaded = defaultdict[ReqId, set[BlockHash]](set)

        logger.info(
            "OffloadingConnectorScheduler initialized with %d groups, sw_sizes=%s",
            self.num_kv_cache_groups,
            self.sw_sizes,
        )

    def get_sw_clipped_blocks(self, block_ids: BlockIds) -> BlockIds:
        """
        Clip the number of blocks to the sliding window size for each kv cache
        group that employs sliding window attention (SWA).

        This is necessary because the KV Cache manager allocates blocks for
        the entire sequence length initially, then cleans up blocks outside
        the window before the request_finished hook.

        For full attention groups (sw_sizes[i] == 0), all blocks are kept.
        For SWA groups, only the last sw_sizes[i] blocks are kept.

        Args:
            block_ids: Block IDs for all KV cache groups.

        Returns:
            Clipped block IDs for each group.
        """
        if len(block_ids) == 0:
            return block_ids
        if len(block_ids) != len(self.sw_sizes):
            # Single group mode or mismatch - return as-is
            return block_ids

        clipped = []
        for i, blocks in enumerate(block_ids):
            sw_size = self.sw_sizes[i]
            if sw_size == 0 or len(blocks) == 0:
                # Full attention or empty - keep all blocks
                clipped.append(blocks)
            else:
                # Sliding window - keep only last sw_size blocks
                clipped.append(blocks[-sw_size:])
        return tuple(clipped)

    def _get_block_hashes(
        self,
        req: Request,
        start_idx: int = 0,
        end_idx: int | None = None,
    ) -> Iterable[BlockHash]:
        return islice(
            req.block_hashes,
            self.block_size_factor * start_idx + self.block_size_factor - 1,
            self.block_size_factor * end_idx if end_idx else None,
            self.block_size_factor,
        )

    def get_num_new_matched_tokens(
        self, request: Request, num_computed_tokens: int
    ) -> tuple[int, bool]:
        """
        Get number of new tokens that can be loaded beyond the
        num_computed_tokens.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request

        Returns:
            A tuple with the following elements:
                - The number of tokens that can be loaded beyond what is
                  already computed.
                - `True` if tokens will be loaded asynchronously
                  (between scheduler steps).
        """
        num_blocks = request.num_tokens // self.offloaded_block_size

        assert len(request.block_hashes) // self.block_size_factor == num_blocks
        block_hashes = self._get_block_hashes(request)

        self.manager.touch(block_hashes)

        full_block_tokens = self.offloaded_block_size * num_blocks
        if full_block_tokens - num_computed_tokens < self.offloaded_block_size:
            # we can load less than a block, skip
            return 0, False

        start_block_idx = num_computed_tokens // self.offloaded_block_size
        hits = self.manager.lookup(
            self._get_block_hashes(request, start_idx=start_block_idx)
        )
        if hits == 0:
            return 0, False

        num_hit_tokens = (
            self.offloaded_block_size * (start_block_idx + hits) - num_computed_tokens
        )
        logger.debug(
            "Request %s hit %s offloaded tokens after %s GPU hit tokens",
            request.request_id,
            num_hit_tokens,
            num_computed_tokens,
        )
        if num_hit_tokens < self.offloaded_block_size:
            return 0, False

        return num_hit_tokens, True

    def update_state_after_alloc(
        self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int
    ):
        self._requests[request.request_id] = request
        # the block ids are updated in _get_reqs_to_store
        self._request_block_ids[request.request_id] = []

        if num_external_tokens == 0:
            return

        # For SWA (Sliding Window Attention), blocks may be padded with null
        # blocks for out-of-window positions. We need to:
        # 1. Filter out null blocks
        # 2. Identify which blocks need data loaded (pending blocks)
        # 3. Map block positions to the correct block hash indices

        # Get non-null blocks with their positions
        # Position is the index in the blocks list (corresponds to token range)
        pending_blocks_with_pos: list[tuple[int, int]] = []  # (position, block_id)
        num_computed_non_null = 0

        for i, block in enumerate(blocks.blocks[0]):
            if block.is_null:
                # Null blocks are placeholders for out-of-window positions
                continue
            if block.block_hash is not None:
                # This block has a hash = already computed/cached locally
                num_computed_non_null += 1
            else:
                # This block needs data loaded from external source (CPU)
                pending_blocks_with_pos.append((i, block.block_id))

        if not pending_blocks_with_pos:
            return  # Nothing to load

        pending_block_ids = [block_id for _, block_id in pending_blocks_with_pos]

        # For SWA, pending blocks are at the END of the allocated range
        # Their positions tell us which token range they cover
        first_pending_position = pending_blocks_with_pos[0][0]
        last_pending_position = pending_blocks_with_pos[-1][0]

        # Convert GPU block positions to offloaded block indices
        # Each offloaded block = block_size_factor GPU blocks
        offloaded_start_idx = first_pending_position // self.block_size_factor
        offloaded_end_idx = (
            last_pending_position + 1 + self.block_size_factor - 1
        ) // self.block_size_factor

        # Validate we have enough block hashes
        total_hashes = len(request.block_hashes) // self.block_size_factor
        if offloaded_end_idx > total_hashes:
            logger.warning(
                "Request %s: offloaded_end_idx=%d > total_hashes=%d, clamping",
                request.request_id,
                offloaded_end_idx,
                total_hashes,
            )
            offloaded_end_idx = total_hashes

        block_hashes = list(
            self._get_block_hashes(
                request, start_idx=offloaded_start_idx, end_idx=offloaded_end_idx
            )
        )

        src_spec = self.manager.prepare_load(block_hashes)
        dst_spec = GPULoadStoreSpec(pending_block_ids)

        self._reqs_to_load[request.request_id] = (src_spec, dst_spec)
        self._reqs_being_loaded[request.request_id].update(block_hashes)
        self._next_stored_block_idx[request.request_id] = offloaded_end_idx

    def _get_reqs_to_store(self, scheduler_output: SchedulerOutput):
        reqs_to_store: dict[ReqId, TransferSpec] = {}
        # iterate over both new and cached requests
        for req_id, new_block_id_groups, preempted in yield_req_data(scheduler_output):
            if preempted:
                self._request_block_ids[req_id] = []

            if new_block_id_groups:
                new_block_ids = new_block_id_groups[0]
                self._request_block_ids[req_id] += new_block_ids

            block_ids = self._request_block_ids[req_id]

            req = self._requests[req_id]
            new_tokens = scheduler_output.num_scheduled_tokens[req_id]
            total_tokens = req.num_computed_tokens + new_tokens
            num_blocks = total_tokens // self.offloaded_block_size
            start_block_idx = self._next_stored_block_idx.get(req_id, 0)
            num_new_blocks = num_blocks - start_block_idx

            if num_new_blocks <= 0:
                continue

            # NOTE: In async scheduling, placeholders may temporarily make
            # len(req.block_hashes) < num_blocks * self.block_size_factor.

            new_block_hashes = self._get_block_hashes(
                req, start_idx=start_block_idx, end_idx=num_blocks
            )
            store_output = self.manager.prepare_store(new_block_hashes)
            if store_output is None:
                logger.warning(
                    "Request %s: cannot store %s blocks", req_id, num_new_blocks
                )
                continue

            self._next_stored_block_idx[req_id] = num_blocks

            if not store_output.block_hashes_to_store:
                continue
            block_hashes_to_store = set(store_output.block_hashes_to_store)

            block_hashes = self._get_block_hashes(req, end_idx=num_blocks)
            self.manager.touch(block_hashes)

            new_block_hashes = self._get_block_hashes(
                req, start_idx=start_block_idx, end_idx=num_blocks
            )
            dst_spec = store_output.store_spec
            src_block_ids: list[int] = []
            for idx, blk_hash in enumerate(new_block_hashes):
                if blk_hash not in block_hashes_to_store:
                    continue
                offloaded_block_idx = start_block_idx + idx
                gpu_block_idx = offloaded_block_idx * self.block_size_factor
                for i in range(self.block_size_factor):
                    src_block_ids.append(block_ids[gpu_block_idx + i])
            src_spec = GPULoadStoreSpec(src_block_ids)

            reqs_to_store[req_id] = (src_spec, dst_spec)
            self._reqs_being_stored[req_id] |= block_hashes_to_store

            logger.debug(
                "Request %s offloading %s blocks starting from block #%d",
                req_id,
                len(block_hashes_to_store),
                start_block_idx,
            )

        return reqs_to_store

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        meta = OffloadingConnectorMetadata(
            reqs_to_load=self._reqs_to_load,
            reqs_to_store=self._get_reqs_to_store(scheduler_output),
        )
        self._reqs_to_load = {}

        # NOTE (orozery): we should move this logic to update_connector_output
        # once KVConnectorOutput allows us to report completed transfers
        for req_id in scheduler_output.preempted_req_ids or ():
            block_hashes = self._reqs_being_stored.get(req_id)
            if block_hashes:
                self.manager.complete_store(block_hashes)
                block_hashes.clear()

        return meta

    def update_connector_output(self, connector_output: KVConnectorOutput):
        """
        Update KVConnector state from worker-side connectors output.

        Args:
            connector_output (KVConnectorOutput): the worker-side
                connectors output.
        """
        for req_id in connector_output.finished_sending or []:
            block_hashes = self._reqs_being_stored.pop(req_id, None)
            if block_hashes:
                self.manager.complete_store(block_hashes)

        for req_id in connector_output.finished_recving or []:
            block_hashes = self._reqs_being_loaded.pop(req_id, None)
            if block_hashes:
                self.manager.complete_load(block_hashes)

    def request_finished(
        self,
        request: Request,
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Called when a request has finished, before its blocks are freed.

        Returns:
            True if the request is being saved/sent asynchronously and blocks
            should not be freed until the request_id is returned from
            get_finished().
            Optional KVTransferParams to be included in the request outputs
            returned by the engine.
        """
        req_id = request.request_id
        self._requests.pop(req_id, None)
        self._request_block_ids.pop(req_id, None)
        self._next_stored_block_idx.pop(req_id, None)

        request_being_stored = req_id in self._reqs_being_stored
        return request_being_stored, None

    def request_finished_all_groups(
        self,
        request: Request,
        block_ids: BlockIds,
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Called exactly once when a request has finished for all kv cache groups
        (HMA mode), before its blocks are freed for each group.

        This method handles sliding window clipping: for SWA groups, blocks
        outside the attention window are not tracked since the KV cache manager
        only allocates blocks within the window.

        Args:
            request: The finished request.
            block_ids: Block IDs for all KV cache groups.

        Returns:
            Tuple of (delay_free, metadata):
            - delay_free: True if blocks should not be freed yet (async store).
            - metadata: Optional KVTransferParams.
        """
        req_id = request.request_id
        self._requests.pop(req_id, None)
        self._request_block_ids.pop(req_id, None)
        self._next_stored_block_idx.pop(req_id, None)

        # Check if any blocks are still being stored
        request_being_stored = req_id in self._reqs_being_stored

        # Clip blocks to sliding window for SWA groups
        # This matches the KV cache manager's allocation strategy
        _ = self.get_sw_clipped_blocks(block_ids)

        return request_being_stored, None

    def take_events(self) -> Iterable[KVCacheEvent]:
        """Take the KV cache events from the connector.

        Returns:
            A list of KV cache events.
        """
        for event in self.manager.take_events():
            if event.removed:
                yield BlockRemoved(block_hashes=event.block_hashes, medium=event.medium)
            else:
                yield BlockStored(
                    block_hashes=event.block_hashes,
                    parent_block_hash=None,
                    token_ids=[],
                    lora_id=None,
                    block_size=event.block_size,
                    medium=event.medium,
                    lora_name=None,
                )


class OffloadingConnectorWorker:
    """Implementation of Worker side methods with HMA support."""

    def __init__(
        self, spec: OffloadingSpec, kv_cache_config: KVCacheConfig | None = None
    ):
        self.spec = spec
        self.kv_cache_config = kv_cache_config
        self.worker = OffloadingWorker()

        self._job_counter = 0

        # req_id -> (job_id, store)
        self._jobs: dict[int, tuple[ReqId, bool]] = {}
        # req_id -> active job IDs
        self._load_job: dict[ReqId, int] = {}
        # req_id -> set(active job IDs)
        self._store_jobs = defaultdict[ReqId, set[int]](set)
        # list of store jobs pending submission (job_id, transfer_spec)
        self._unsubmitted_store_jobs: list[tuple[int, TransferSpec]] = []

        self._finished_reqs_waiting_for_store: set[ReqId] = set()

    def _generate_job_id(self) -> int:
        job_id = self._job_counter
        self._job_counter = job_id + 1
        return job_id

    def _register_handlers(
        self,
        kv_caches: dict[str, torch.Tensor],
        attn_backends: dict[str, type[AttentionBackend]],
    ):
        """Register handlers for KV offloading transfers.

        Supports both single-group (backward compatible) and HMA modes.
        In HMA mode, registers per-group handlers for each KV cache group.
        """
        handler_results = list(
            self.spec.get_handlers(kv_caches, attn_backends, self.kv_cache_config)
        )

        for result in handler_results:
            if len(result) == 4:
                # HMA mode: 4-tuple (src_cls, dst_cls, handler, group_id)
                src_cls, dst_cls, handler, group_id = result
                self.worker.register_group_handler(src_cls, dst_cls, group_id, handler)
                logger.debug(
                    "Registered handler for group %d: %s -> %s",
                    group_id,
                    src_cls.medium(),
                    dst_cls.medium(),
                )
            else:
                # Single-group mode: 3-tuple (src_cls, dst_cls, handler)
                src_cls, dst_cls, handler = result
                self.worker.register_handler(src_cls, dst_cls, handler)
                logger.debug(
                    "Registered handler: %s -> %s",
                    src_cls.medium(),
                    dst_cls.medium(),
                )

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        layer_names = list(kv_caches.keys())
        layers = get_layers_from_vllm_config(
            self.spec.vllm_config, Attention, layer_names
        )
        attn_backends = {
            layer_name: layers[layer_name].get_attn_backend()
            for layer_name in layer_names
        }
        self._register_handlers(kv_caches, attn_backends)

    def register_cross_layers_kv_cache(
        self, kv_cache: torch.Tensor, attn_backend: type[AttentionBackend]
    ):
        cross_layer_name = "ALL_LAYERS"
        kv_caches = {cross_layer_name: kv_cache}
        attn_backends = {cross_layer_name: attn_backend}
        self._register_handlers(kv_caches, attn_backends)

    def handle_preemptions(self, preempted_req_ids: set[str]):
        for job_id, transfer_spec in self._unsubmitted_store_jobs:
            success = self.worker.transfer_async(job_id, transfer_spec)
            assert success
        self._unsubmitted_store_jobs.clear()

        for req_id in preempted_req_ids:
            job_ids = self._store_jobs.get(req_id)
            if job_ids:
                self.worker.wait(job_ids)

    def start_kv_transfers(self, metadata: OffloadingConnectorMetadata):
        for job_id, transfer_spec in self._unsubmitted_store_jobs:
            success = self.worker.transfer_async(job_id, transfer_spec)
            assert success
        self._unsubmitted_store_jobs.clear()

        for req_id, transfer_spec in metadata.reqs_to_load.items():
            job_id = self._generate_job_id()
            self._jobs[job_id] = (req_id, False)
            assert req_id not in self._load_job
            self._load_job[req_id] = job_id
            success = self.worker.transfer_async(job_id, transfer_spec)
            assert success

    def prepare_store_kv(self, metadata: OffloadingConnectorMetadata):
        for req_id, transfer_spec in metadata.reqs_to_store.items():
            job_id = self._generate_job_id()
            self._jobs[job_id] = (req_id, True)
            self._store_jobs[req_id].add(job_id)
            # NOTE(orozery): defer the store to the beginning of the next engine step,
            # so that offloading starts AFTER transfers related to token sampling,
            # thereby avoiding delays to token generation due to offloading.
            self._unsubmitted_store_jobs.append((job_id, transfer_spec))

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        """
        Notifies worker-side connector ids of requests that have
        finished generating tokens.
        Returns a list of request IDs that finished loading or storing.

        Returns:
            ids of requests that have finished asynchronous transfer
            tuple of (sending/saving ids, recving/loading ids).
        """
        finished_sending = set()
        finished_recving = set()
        for job_id, success in self.worker.get_finished():
            # we currently do not support job failures
            assert success
            req_id, store = self._jobs.pop(job_id)
            if store:
                req_jobs = self._store_jobs[req_id]
                req_jobs.remove(job_id)
                if req_jobs:
                    continue

                if req_id in self._finished_reqs_waiting_for_store:
                    self._finished_reqs_waiting_for_store.remove(req_id)
                    finished_sending.add(req_id)
                    del self._store_jobs[req_id]
            else:
                req_job = self._load_job[req_id]
                assert job_id == req_job
                del self._load_job[req_id]
                finished_recving.add(req_id)

        for req_id in finished_req_ids:
            pending_req_jobs = self._store_jobs.get(req_id)
            if pending_req_jobs:
                self._finished_reqs_waiting_for_store.add(req_id)
            elif pending_req_jobs is not None:
                finished_sending.add(req_id)
                del self._store_jobs[req_id]

        return finished_sending, finished_recving
