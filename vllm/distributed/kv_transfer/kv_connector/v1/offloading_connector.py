# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
KV cache offloading connector with full HMA (Hybrid Memory Allocator) support.

This connector supports hybrid models with multiple KV cache groups
(e.g., Full Attention + Mamba layers) by tracking and offloading
each group independently.
"""

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from itertools import islice
from typing import Any

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

ReqId = str
GroupId = int

logger = init_logger(__name__)


@dataclass
class OffloadingConnectorMetadata(KVConnectorMetadata):
    """
    Metadata for offloading connector with HMA support.

    For HMA, transfer specs are organized per request and per group:
    - reqs_to_load[req_id][group_id] = TransferSpec
    - reqs_to_store[req_id][group_id] = TransferSpec
    """

    # req_id -> group_id -> TransferSpec
    reqs_to_load: dict[ReqId, dict[GroupId, TransferSpec]] = field(default_factory=dict)
    reqs_to_store: dict[ReqId, dict[GroupId, TransferSpec]] = field(
        default_factory=dict
    )


class OffloadingConnector(KVConnectorBase_V1):
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

        # Store the number of KV cache groups for HMA support
        self.num_kv_cache_groups = (
            len(kv_cache_config.kv_cache_groups) if kv_cache_config else 1
        )

        self.connector_scheduler: OffloadingConnectorScheduler | None = None
        self.connector_worker: OffloadingConnectorWorker | None = None
        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = OffloadingConnectorScheduler(
                spec, self.num_kv_cache_groups, kv_cache_config
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
        # All loads are completed in start_load_kv, so nothing to do here.
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
        """
        Called when a request has finished (single KV cache group).

        For HMA support with multiple groups, use request_finished_all_groups.
        """
        assert self.connector_scheduler is not None
        # Wrap as single-group tuple for unified handling
        return self.connector_scheduler.request_finished(request, (block_ids,))

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        HMA-compatible method: Called when a request has finished for ALL
        KV cache groups, before its blocks are freed for each group.

        This is required by the SupportsHMA interface for hybrid models
        (e.g., Full Attention + Mamba).

        Args:
            request: The finished request.
            block_ids: Tuple of block ID lists, one per KV cache group.
                       e.g., (full_attn_blocks, mamba_blocks) for hybrid models.

        Returns:
            Tuple of (should_delay_free, optional_kv_transfer_params).
            If should_delay_free is True, blocks won't be freed until
            the request_id is returned from get_finished().
        """
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished(request, block_ids)

    def take_events(self) -> Iterable[KVCacheEvent]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.take_events()


class OffloadingConnectorScheduler:
    """
    Scheduler-side implementation with full HMA support.

    Tracks block IDs and transfer state per KV cache group, allowing
    independent offloading of different cache types (e.g., attention KV
    and Mamba states).
    """

    def __init__(
        self,
        spec: OffloadingSpec,
        num_kv_cache_groups: int = 1,
        kv_cache_config: KVCacheConfig | None = None,
    ):
        self.gpu_block_size = spec.gpu_block_size
        self.offloaded_block_size = spec.offloaded_block_size
        self.block_size_factor = self.offloaded_block_size // self.gpu_block_size
        # Use HybridOffloadingManager for multiple groups (HMA support)
        self.manager: OffloadingManager = spec.get_manager(
            num_groups=num_kv_cache_groups
        )
        self.num_kv_cache_groups = num_kv_cache_groups

        self.sliding_window_groups: set[int] = set()
        if kv_cache_config is not None:
            for group_id, group_spec in enumerate(kv_cache_config.kv_cache_groups):
                if isinstance(group_spec.kv_cache_spec, SlidingWindowSpec):
                    self.sliding_window_groups.add(group_id)
                    logger.info(
                        "Group %d is sliding window (window=%d), will skip offloading",
                        group_id,
                        group_spec.kv_cache_spec.sliding_window,
                    )

        self._requests: dict[ReqId, Request] = {}

        # HMA: Track block IDs per request per group
        # req_id -> group_id -> list of GPU block IDs
        self._request_block_ids: dict[ReqId, dict[GroupId, list[int]]] = {}

        # HMA: Track load requests per group
        # req_id -> group_id -> TransferSpec
        self._reqs_to_load: dict[ReqId, dict[GroupId, TransferSpec]] = {}

        # HMA: Track next stored block index per group
        # req_id -> group_id -> next block index to offload
        self._next_stored_block_idx: dict[ReqId, dict[GroupId, int]] = {}

        # HMA: Track block hashes being stored/loaded per group
        # req_id -> group_id -> set(block hashes)
        self._reqs_being_stored: dict[ReqId, dict[GroupId, set[BlockHash]]] = (
            defaultdict(lambda: defaultdict(set))
        )
        self._reqs_being_loaded: dict[ReqId, dict[GroupId, set[BlockHash]]] = (
            defaultdict(lambda: defaultdict(set))
        )

    def _init_request_state(self, req_id: ReqId) -> None:
        """Initialize per-group state for a new request."""
        self._request_block_ids[req_id] = {
            g: [] for g in range(self.num_kv_cache_groups)
        }

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

        full_block_tokens = self.offloaded_block_size * num_blocks
        if full_block_tokens - num_computed_tokens < self.offloaded_block_size:
            # we can load less than a block, skip
            return 0, False

        start_block_idx = num_computed_tokens // self.offloaded_block_size

        # For HMA: Check all groups and take the maximum hits.
        max_hits = 0
        for group_id in range(self.num_kv_cache_groups):
            block_hashes = self._get_block_hashes(request, start_idx=start_block_idx)
            hits = self.manager.lookup(block_hashes, group_id=group_id)
            logger.debug(
                "Request %s: CPU lookup group %d from block %d: %d hits",
                request.request_id,
                group_id,
                start_block_idx,
                hits,
            )
            if hits > max_hits:
                max_hits = hits
                # Touch blocks in this group (the one with most hits)
                block_hashes = self._get_block_hashes(request)
                self.manager.touch(block_hashes, group_id=group_id)

        if max_hits == 0:
            return 0, False

        num_hit_tokens = (
            self.offloaded_block_size * (start_block_idx + max_hits)
            - num_computed_tokens
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
        """
        Update state after block allocation for all KV cache groups.
        """
        req_id = request.request_id
        self._requests[req_id] = request
        self._init_request_state(req_id)

        if num_external_tokens == 0:
            return

        block_groups = blocks.get_block_ids()

        logger.debug(
            "update_state_after_alloc: req=%s, num_external_tokens=%d, "
            "num_groups=%d, block_groups_len=%d",
            req_id,
            num_external_tokens,
            self.num_kv_cache_groups,
            len(block_groups),
        )

        # Process each KV cache group
        for group_id, block_ids in enumerate(block_groups):
            # # Skip sliding window groups - they reuse GPU blocks circularly
            # # for old tokens, so loading them from CPU would be incorrect
            if group_id in self.sliding_window_groups:
                logger.info(
                    "Request %s group %d: (sliding window group)",
                    req_id,
                    group_id,
                )

            if not block_ids:
                logger.debug(
                    "Request %s group %d: skipped (empty block_ids)",
                    req_id,
                    group_id,
                )
                continue

            # Skip if this group doesn't have blocks to process
            if group_id >= len(blocks.blocks) or not blocks.blocks[group_id]:
                logger.debug(
                    "Request %s group %d: skipped (no blocks.blocks)",
                    req_id,
                    group_id,
                )
                continue

            num_computed_gpu_blocks = sum(
                block.block_hash is not None for block in blocks.blocks[group_id]
            )
            num_computed_tokens = num_computed_gpu_blocks * self.gpu_block_size
            full_block_tokens = num_computed_tokens + num_external_tokens

            logger.info(
                "Request %s group %d: num_computed_gpu_blocks=%d, "
                "num_external_tokens=%d, total_blocks=%d",
                req_id,
                group_id,
                num_computed_gpu_blocks,
                num_external_tokens,
                len(block_ids),
            )

            # Skip groups where tokens don't align with offloaded block size
            if full_block_tokens % self.offloaded_block_size != 0:
                logger.info(
                    "Request %s group %d: tokens %d not aligned to block size %d",
                    req_id,
                    group_id,
                    full_block_tokens,
                    self.offloaded_block_size,
                )
                continue

            num_pending_gpu_blocks = len(block_ids) - num_computed_gpu_blocks
            if num_external_tokens != num_pending_gpu_blocks * self.gpu_block_size:
                logger.info(
                    "Request %s group %d: skipped (external_tokens %d != pending %d)",
                    req_id,
                    group_id,
                    num_external_tokens,
                    num_pending_gpu_blocks * self.gpu_block_size,
                )
                continue

            start_block_idx = num_computed_tokens // self.offloaded_block_size
            num_blocks = full_block_tokens // self.offloaded_block_size

            if len(request.block_hashes) // self.block_size_factor < num_blocks:
                logger.debug(
                    "Request %s group %d: skipped (not enough block_hashes)",
                    req_id,
                    group_id,
                )
                continue

            # Get the GPU block IDs to load into
            gpu_block_ids = block_ids[num_computed_gpu_blocks:]

            # Skip groups with invalid GPU block IDs. Sliding window attention
            # groups have placeholder block ID 0 for tokens outside the window.
            # We detect this by checking if the first block ID is 0 but there
            # are more than one unique ID (meaning some blocks are valid).
            # If all blocks are 0, it's also invalid.
            num_zeros = sum(1 for b in gpu_block_ids if b == 0)

            # Skip if: all zeros, or mostly zeros (sliding window with placeholders)
            # A valid full-attention group should have unique IDs equal to count
            if not gpu_block_ids or num_zeros > 0:
                # This is a sliding window group or has placeholder blocks
                # Don't load from CPU as it would corrupt block 0
                logger.debug(
                    "Request %s group %d: skipped (has %d zero block IDs out of %d, "
                    "indicates sliding window or placeholder blocks)",
                    req_id,
                    group_id,
                    num_zeros,
                    len(gpu_block_ids),
                )
                continue

            block_hashes = self._get_block_hashes(
                request, start_idx=start_block_idx, end_idx=num_blocks
            )

            src_spec = self.manager.prepare_load(block_hashes, group_id=group_id)
            dst_spec = GPULoadStoreSpec(gpu_block_ids, group_id=group_id)

            block_hashes = self._get_block_hashes(
                request, start_idx=start_block_idx, end_idx=num_blocks
            )

            # Store per-group transfer spec
            if req_id not in self._reqs_to_load:
                self._reqs_to_load[req_id] = {}
            self._reqs_to_load[req_id][group_id] = (src_spec, dst_spec)
            self._reqs_being_loaded[req_id][group_id].update(block_hashes)

            if req_id not in self._next_stored_block_idx:
                self._next_stored_block_idx[req_id] = {}
            self._next_stored_block_idx[req_id][group_id] = num_blocks

            logger.debug(
                "Request %s group %d: prepared load of %d blocks "
                "(src CPU blocks: %s, dst GPU blocks: %s)",
                req_id,
                group_id,
                num_blocks - start_block_idx,
                src_spec.block_ids[:5].tolist() if len(src_spec.block_ids) > 0 else [],
                gpu_block_ids[:5] if len(gpu_block_ids) > 0 else [],
            )

    def _get_reqs_to_store(
        self, scheduler_output: SchedulerOutput
    ) -> dict[ReqId, dict[GroupId, TransferSpec]]:
        """
        Get requests to store with HMA support (all groups).
        """
        reqs_to_store: dict[ReqId, dict[GroupId, TransferSpec]] = {}

        # iterate over both new and cached requests
        for req_id, new_block_id_groups, preempted in yield_req_data(scheduler_output):
            if req_id not in self._request_block_ids:
                continue

            if preempted:
                # Reset all groups on preemption
                self._init_request_state(req_id)

            # Update block IDs for each group
            if new_block_id_groups:
                for group_id, new_block_ids in enumerate(new_block_id_groups):
                    if group_id in self._request_block_ids[req_id]:
                        self._request_block_ids[req_id][group_id] += list(new_block_ids)

            req = self._requests.get(req_id)
            if req is None:
                continue

            new_tokens = scheduler_output.num_scheduled_tokens.get(req_id, 0)
            total_tokens = req.num_computed_tokens + new_tokens
            num_blocks = total_tokens // self.offloaded_block_size

            # Process each group
            for group_id in range(self.num_kv_cache_groups):
                # Skip sliding window groups - they reuse GPU blocks circularly
                # for old tokens, so storing them would create invalid CPU cache
                if group_id in self.sliding_window_groups:
                    logger.info(
                        "Request %s group %d: store (sliding window group)",
                        req_id,
                        group_id,
                    )

                block_ids = self._request_block_ids[req_id].get(group_id, [])
                if not block_ids:
                    continue

                start_block_idx = self._next_stored_block_idx.get(req_id, {}).get(
                    group_id, 0
                )
                num_new_blocks = num_blocks - start_block_idx

                if num_new_blocks <= 0:
                    continue

                # NOTE: In async scheduling, placeholders may temporarily make
                # len(req.block_hashes) < num_blocks * self.block_size_factor.

                new_block_hashes = self._get_block_hashes(
                    req, start_idx=start_block_idx, end_idx=num_blocks
                )
                store_output = self.manager.prepare_store(
                    new_block_hashes, group_id=group_id
                )
                if store_output is None:
                    logger.warning(
                        "Request %s group %d: cannot store %s blocks",
                        req_id,
                        group_id,
                        num_new_blocks,
                    )
                    continue

                if req_id not in self._next_stored_block_idx:
                    self._next_stored_block_idx[req_id] = {}
                self._next_stored_block_idx[req_id][group_id] = num_blocks

                if not store_output.block_hashes_to_store:
                    continue
                block_hashes_to_store = set(store_output.block_hashes_to_store)

                block_hashes = self._get_block_hashes(req, end_idx=num_blocks)
                self.manager.touch(block_hashes, group_id=group_id)

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
                        if gpu_block_idx + i < len(block_ids):
                            src_block_ids.append(block_ids[gpu_block_idx + i])
                src_spec = GPULoadStoreSpec(src_block_ids, group_id=group_id)

                if req_id not in reqs_to_store:
                    reqs_to_store[req_id] = {}
                reqs_to_store[req_id][group_id] = (src_spec, dst_spec)
                self._reqs_being_stored[req_id][group_id] |= block_hashes_to_store

                logger.debug(
                    "Request %s group %d offloading %d blocks starting from #%d "
                    "(src GPU blocks: %s, dst CPU blocks: %s)",
                    req_id,
                    group_id,
                    len(block_hashes_to_store),
                    start_block_idx,
                    src_spec.block_ids[:5].tolist()
                    if len(src_spec.block_ids) > 0
                    else [],
                    dst_spec.block_ids[:5].tolist()
                    if len(dst_spec.block_ids) > 0
                    else [],
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
            req_groups = self._reqs_being_stored.pop(req_id, None)
            if req_groups:
                # Complete store for all groups
                for group_id, block_hashes in req_groups.items():
                    if block_hashes:
                        self.manager.complete_store(block_hashes, group_id=group_id)

        for req_id in connector_output.finished_recving or []:
            req_groups = self._reqs_being_loaded.pop(req_id, None)
            if req_groups:
                # Complete load for all groups
                for group_id, block_hashes in req_groups.items():
                    if block_hashes:
                        self.manager.complete_load(block_hashes, group_id=group_id)

    def request_finished(
        self,
        request: Request,
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Called when a request has finished for all KV cache groups,
        before its blocks are freed.

        Args:
            request: The finished request.
            block_ids: Tuple of block ID lists, one per KV cache group.

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

        # Check if ANY group has pending stores
        request_being_stored = req_id in self._reqs_being_stored and any(
            bool(hashes) for hashes in self._reqs_being_stored.get(req_id, {}).values()
        )
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
    """
    Worker-side implementation with HMA support.

    Handles transfer jobs for all KV cache groups, tracking completion
    across groups before marking a request as finished.
    """

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
        for result in self.spec.get_handlers(
            kv_caches, attn_backends, self.kv_cache_config
        ):
            if len(result) == 4:
                # HMA mode: 4-tuple (src_cls, dst_cls, handler, group_id)
                src_cls, dst_cls, handler, group_id = result
                self.worker.register_group_handler(src_cls, dst_cls, group_id, handler)
            else:
                # Non-HMA mode: 3-tuple (src_cls, dst_cls, handler)
                src_cls, dst_cls, handler = result
                self.worker.register_handler(src_cls, dst_cls, handler)

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
        Get finished requests across all groups.

        A request is only considered "finished" when ALL its groups
        have completed their transfers.

        Returns:
            ids of requests that have finished asynchronous transfer
            tuple of (sending/saving ids, recving/loading ids).
        """
        finished_sending: set[str] = set()
        finished_recving: set[str] = set()

        for job_id, success in self.worker.get_finished():
            # we currently do not support job failures
            assert success
            req_id, group_id, is_store = self._jobs.pop(job_id)

            if is_store:
                req_group_jobs = self._store_jobs[req_id][group_id]
                req_group_jobs.discard(job_id)

                # Check if ALL groups are done for this request
                all_groups_done = all(
                    len(jobs) == 0 for jobs in self._store_jobs[req_id].values()
                )
                if all_groups_done and req_id in self._finished_reqs_waiting_for_store:
                    self._finished_reqs_waiting_for_store.remove(req_id)
                    finished_sending.add(req_id)
                    del self._store_jobs[req_id]
            else:
                if group_id in self._load_jobs.get(req_id, {}):
                    del self._load_jobs[req_id][group_id]

                # Check if ALL groups are done loading
                if not self._load_jobs.get(req_id):
                    finished_recving.add(req_id)
                    self._load_jobs.pop(req_id, None)

        # Handle finished requests that may still have pending stores
        for req_id in finished_req_ids:
            has_pending = any(
                bool(jobs) for jobs in self._store_jobs.get(req_id, {}).values()
            )
            if has_pending:
                self._finished_reqs_waiting_for_store.add(req_id)
            elif req_id in self._store_jobs:
                finished_sending.add(req_id)
                del self._store_jobs[req_id]

        return finished_sending, finished_recving
