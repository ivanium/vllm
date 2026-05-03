# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-side logic for MooncakeStoreConnector."""

import os
from functools import reduce
from math import lcm
from typing import Any, cast

import zmq

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_store_data import (
    LoadSpec,
    MooncakeStoreConnectorMetadata,
    ReqMeta,
    RequestTracker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_utils import (
    get_mooncake_dp_engine_index,
)
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.utils.network_utils import make_zmq_socket
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    SlidingWindowSpec,
)
from vllm.v1.request import Request
from vllm.v1.serial_utils import MsgpackEncoder

logger = init_logger(__name__)


class MooncakeStoreScheduler:
    """Scheduler-side component for MooncakeStoreConnector."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig | None = None,
    ):
        self.kv_role = vllm_config.kv_transfer_config.kv_role
        self.load_async = vllm_config.kv_transfer_config.kv_connector_extra_config.get(
            "load_async", False
        )
        self.client = LookupKeyClient(vllm_config)

        self.load_specs: dict[str, LoadSpec] = {}
        self.pcp_size = vllm_config.parallel_config.prefill_context_parallel_size
        self.dcp_size = vllm_config.parallel_config.decode_context_parallel_size
        # Scheduler alignment = LCM(group_block_sizes); engine mutates
        # cache_config.block_size to MIN for HMA. Fall back to
        # cache_config.block_size for non-HMA / pre-spec callers.
        self.original_block_size = vllm_config.cache_config.block_size
        if kv_cache_config is not None and kv_cache_config.kv_cache_groups:
            self._block_size = reduce(
                lcm,
                (g.kv_cache_spec.block_size for g in kv_cache_config.kv_cache_groups),
            )
        else:
            self._block_size = vllm_config.cache_config.block_size
        if self.pcp_size > 1:
            self._block_size *= self.pcp_size
        if self.dcp_size > 1:
            self._block_size *= self.dcp_size

        self._request_trackers: dict[str, RequestTracker] = {}
        self._preempted_req_ids: set[str] = set()
        self._discard_partial_chunks = (
            vllm_config.kv_transfer_config.get_from_extra_config(
                "discard_partial_chunks", True
            )
        )
        self._unfinished_requests: dict[str, tuple[Request, list[list[int]]]] = {}
        self._unfinished_request_ids: set[str] = set()

        # HMA detection. blocks_per_sw[g] is non-zero only for SlidingWindowSpec
        # groups; FullAttentionSpec groups stay 0 and are never clipped.
        self._is_hma_required = False
        self.blocks_per_sw: list[int] = [0]
        if kv_cache_config is not None:
            self._is_hma_required = (
                not vllm_config.scheduler_config.disable_hybrid_kv_cache_manager
                and any(
                    not isinstance(g.kv_cache_spec, FullAttentionSpec)
                    for g in kv_cache_config.kv_cache_groups
                )
            )
            sw_sizes_tokens: list[tuple[int, int]] = [
                (g.kv_cache_spec.sliding_window, g.kv_cache_spec.block_size)
                if isinstance(g.kv_cache_spec, SlidingWindowSpec)
                else (0, self._block_size)
                for g in kv_cache_config.kv_cache_groups
            ]
            self.blocks_per_sw = [
                cdiv(n_tokens, block_size) + 1 if n_tokens else 0
                for n_tokens, block_size in sw_sizes_tokens
            ]

    def get_sw_clipped_blocks(
        self,
        block_ids: tuple[list[int], ...] | list[list[int]],
    ) -> list[list[int]]:
        """Clip per-group block IDs to the SWA window. No-op for non-HMA."""
        if len(block_ids) == 0 or not self._is_hma_required:
            return list(block_ids)
        return [
            blocks[-self.blocks_per_sw[i] :] if self.blocks_per_sw[i] > 0 else blocks
            for i, blocks in enumerate(block_ids)
        ]

    def _normalize_block_ids(
        self,
        block_ids: tuple[list[int], ...] | list[int] | list[list[int]],
    ) -> list[list[int]]:
        """Return a per-group list[list[int]] regardless of input shape."""
        if isinstance(block_ids, tuple):
            return self.get_sw_clipped_blocks(block_ids)
        if (
            isinstance(block_ids, list)
            and len(block_ids) > 0
            and isinstance(block_ids[0], list)
        ):
            return self.get_sw_clipped_blocks(cast(list[list[int]], block_ids))
        # Flat list[int] → wrap as single group, no clipping.
        flat = cast(list[int], block_ids)
        return [list(flat)] if flat else [[]]

    def get_num_new_matched_tokens(
        self,
        request: Request,
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        """Check for external KV cache hit."""
        if self._discard_partial_chunks:
            token_len = (
                len(request.prompt_token_ids) // self._block_size * self._block_size
            )
        else:
            token_len = len(request.prompt_token_ids)

        if token_len < self._block_size:
            return 0, False

        num_external_hit_tokens = self.client.lookup(token_len, request.block_hashes)

        if num_external_hit_tokens == request.num_tokens:
            num_external_hit_tokens -= 1

        if num_external_hit_tokens < num_computed_tokens:
            need_to_allocate = 0
        else:
            need_to_allocate = num_external_hit_tokens - num_computed_tokens

        logger.debug(
            "Reqid: %s, Total tokens %d, kvpool hit tokens: %d, need to load: %d",
            request.request_id,
            request.num_tokens,
            num_external_hit_tokens,
            need_to_allocate,
        )

        if need_to_allocate <= 0:
            return 0, False

        self.load_specs[request.request_id] = LoadSpec(
            vllm_cached_tokens=num_computed_tokens,
            kvpool_cached_tokens=num_external_hit_tokens,
            can_load=False,
        )

        return need_to_allocate, self.load_async

    def update_state_after_alloc(
        self,
        request: Request,
        blocks: KVCacheBlocks,
        num_external_tokens: int,
    ):
        """Update state after block allocation."""
        if num_external_tokens > 0:
            # Always per-group (tuple) → list[list[int]]; clip SWA groups.
            local_block_ids = self.get_sw_clipped_blocks(blocks.get_block_ids())
        else:
            # One empty group per detected KV cache group, so subsequent
            # .update() calls have correct shape.
            local_block_ids = [[] for _ in range(len(self.blocks_per_sw))]

        self._unfinished_requests[request.request_id] = (request, local_block_ids)
        self._unfinished_request_ids.add(request.request_id)

        if request.request_id not in self.load_specs:
            return

        if num_external_tokens == 0:
            self.load_specs[request.request_id].can_load = False
            return

        assert (
            num_external_tokens > 0
            and num_external_tokens
            == self.load_specs[request.request_id].kvpool_cached_tokens
            - self.load_specs[request.request_id].vllm_cached_tokens
        ), (
            f"Mismatch in number of tokens: {num_external_tokens} vs "
            f"{self.load_specs[request.request_id].kvpool_cached_tokens} - "
            f"{self.load_specs[request.request_id].vllm_cached_tokens}"
            f" for request {request.request_id}"
        )

        self.load_specs[request.request_id].can_load = True

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        """Build connector metadata for this scheduler step."""
        force_skip_save = self.kv_role == "kv_consumer"

        for finished_req_id in scheduler_output.finished_req_ids:
            self._request_trackers.pop(finished_req_id, None)
            self._unfinished_requests.pop(finished_req_id, None)
            self._unfinished_request_ids.discard(finished_req_id)
            self._preempted_req_ids.discard(finished_req_id)

        self._preempted_req_ids.update(scheduler_output.preempted_req_ids)
        for req_id in scheduler_output.preempted_req_ids:
            self._request_trackers.pop(req_id, None)
            self._unfinished_requests.pop(req_id, None)

        meta = MooncakeStoreConnectorMetadata(
            self._unfinished_request_ids,
            scheduler_output.preempted_req_ids,
        )

        # Handle new requests
        for request in scheduler_output.scheduled_new_reqs:
            load_spec = self.load_specs.pop(request.req_id, None)
            num_tokens_to_compute = (
                request.num_computed_tokens
                + scheduler_output.num_scheduled_tokens[request.req_id]
            )
            request_tuple = self._unfinished_requests.get(request.req_id)
            request_real = request_tuple[0]  # type: ignore[index]

            unfolded_block_ids = self._normalize_block_ids(request.block_ids)

            request_tracker = RequestTracker(
                req_id=request.req_id,
                token_len=num_tokens_to_compute,
                allocated_block_ids=unfolded_block_ids,
                num_saved_tokens=0,
                token_ids=request.prompt_token_ids[:num_tokens_to_compute].copy(),
            )
            self._request_trackers[request.req_id] = request_tracker

            last_chunk_tokens_num = (
                (len(request.prompt_token_ids) // self._block_size * self._block_size)
                if self._discard_partial_chunks
                else len(request.prompt_token_ids)
            )

            req_meta = ReqMeta.from_request_tracker(
                request_tracker,
                self._block_size,
                load_spec=load_spec,
                skip_save=force_skip_save,
                block_hashes=request_real.block_hashes,
                is_last_chunk=(request_tracker.token_len >= last_chunk_tokens_num),
                discard_partial_chunks=self._discard_partial_chunks,
                original_block_size=self.original_block_size,
            )
            if req_meta is not None:
                meta.add_request(req_meta)

        # Handle cached (running) requests
        cached_reqs = scheduler_output.scheduled_cached_reqs
        if not force_skip_save:
            for i, req_id in enumerate(cached_reqs.req_ids):
                new_block_ids = cached_reqs.new_block_ids[i]
                if not new_block_ids:
                    continue

                req_meta = None
                if req_id in self._preempted_req_ids:
                    # Resumed after preemption
                    new_block_ids = self._normalize_block_ids(new_block_ids)
                    self._preempted_req_ids.discard(req_id)
                    load_spec = self.load_specs.pop(req_id, None)
                    request_tuple = self._unfinished_requests.get(req_id)
                    request_real = request_tuple[0]  # type: ignore[index]
                    num_tokens_to_compute = (
                        request_real.num_computed_tokens
                        + scheduler_output.num_scheduled_tokens[req_id]
                    )
                    request_tracker = RequestTracker(
                        req_id=req_id,
                        token_len=num_tokens_to_compute,
                        allocated_block_ids=new_block_ids,
                        num_saved_tokens=0,
                        token_ids=request_real.prompt_token_ids[
                            :num_tokens_to_compute
                        ].copy(),
                    )
                    self._request_trackers[req_id] = request_tracker

                    last_chunk_tokens_num = (
                        (
                            len(request_real.prompt_token_ids)
                            // self._block_size
                            * self._block_size
                        )
                        if self._discard_partial_chunks
                        else len(request_real.prompt_token_ids)
                    )
                    req_meta = ReqMeta.from_request_tracker(
                        request_tracker,
                        self._block_size,
                        load_spec=load_spec,
                        skip_save=force_skip_save,
                        block_hashes=request_real.block_hashes,
                        is_last_chunk=(
                            request_tracker.token_len >= last_chunk_tokens_num
                        ),
                        discard_partial_chunks=self._discard_partial_chunks,
                        original_block_size=self.original_block_size,
                    )
                else:
                    # Decode/chunked request
                    request_tracker = self._request_trackers[req_id]
                    num_new_tokens = scheduler_output.num_scheduled_tokens[req_id]
                    req_tuple = self._unfinished_requests.get(req_id)
                    if req_tuple:
                        request = req_tuple[0]
                        num_current_tokens = request_tracker.token_len
                        new_token_ids = request.all_token_ids[
                            num_current_tokens : num_current_tokens + num_new_tokens
                        ]
                        request_tracker.token_len += len(new_token_ids)
                    else:
                        raise ValueError(
                            f"Request {req_id} is not in _unfinished_requests"
                        )
                    num_computed_token = cached_reqs.num_computed_tokens[i]
                    if num_computed_token >= len(request.prompt_token_ids):
                        continue
                    request_tracker.update(new_block_ids)

                    last_chunk_tokens_num = (
                        (
                            len(request.prompt_token_ids)
                            // self._block_size
                            * self._block_size
                        )
                        if self._discard_partial_chunks
                        else len(request.prompt_token_ids)
                    )
                    req_meta = ReqMeta.from_request_tracker(
                        request_tracker,
                        self._block_size,
                        load_spec=None,
                        skip_save=force_skip_save,
                        block_hashes=request.block_hashes,
                        is_last_chunk=(
                            request_tracker.token_len >= last_chunk_tokens_num
                        ),
                        discard_partial_chunks=self._discard_partial_chunks,
                        original_block_size=self.original_block_size,
                    )

                if req_meta is not None:
                    meta.add_request(req_meta)

        # Handle requests with pending load specs not yet scheduled
        request_ids = [req.req_id for req in scheduler_output.scheduled_new_reqs]
        for request_id, (
            request,
            block_ids,
        ) in self._unfinished_requests.items():
            if request_id not in request_ids and request_id not in cached_reqs.req_ids:
                load_spec = self.load_specs.pop(request_id, None)
                if not load_spec:
                    continue
                num_tokens_to_compute = load_spec.kvpool_cached_tokens
                if (num_tokens_to_compute % self._block_size != 0) and (
                    num_tokens_to_compute == len(request.prompt_token_ids) - 1
                ):
                    num_tokens_to_compute = num_tokens_to_compute + 1
                request_tracker = RequestTracker(
                    req_id=request_id,
                    token_len=num_tokens_to_compute,
                    allocated_block_ids=block_ids,
                    num_saved_tokens=0,
                )
                self._request_trackers[request_id] = request_tracker
                req_meta = ReqMeta.from_request_tracker(
                    request_tracker,
                    self._block_size,
                    load_spec=load_spec,
                    skip_save=None,
                    block_hashes=request.block_hashes,
                    discard_partial_chunks=self._discard_partial_chunks,
                )
                if req_meta is not None:
                    meta.add_request(req_meta)

        return meta

    def request_finished(
        self,
        request: Request,
        block_ids: tuple[list[int], ...] | list[int] | list[list[int]],
    ) -> tuple[bool, dict[str, Any] | None]:
        """Determine whether to delay freeing blocks for async save."""
        if self.kv_role == "kv_consumer":
            return False, None
        tracker = self._request_trackers.get(request.request_id)
        if tracker is not None and tracker.num_saved_tokens <= 0:
            return False, None
        normalized = self._normalize_block_ids(block_ids)
        delay_free_blocks = any(len(group) > 0 for group in normalized)
        if delay_free_blocks:
            logger.debug(
                "Delaying free of %s blocks for request %s",
                [len(g) for g in normalized],
                request.request_id,
            )
        return delay_free_blocks, None


class LookupKeyClient:
    """ZMQ client for querying prefix cache hits from worker."""

    def __init__(self, vllm_config: VllmConfig):
        self.encoder = MsgpackEncoder()
        self.ctx = zmq.Context()  # type: ignore[attr-defined]
        socket_path = get_zmq_rpc_path_lookup(vllm_config)
        self.socket = make_zmq_socket(
            self.ctx,
            socket_path,
            zmq.REQ,  # type: ignore[attr-defined]
            bind=False,
        )

    def lookup(self, token_len: int, block_hashes: list[BlockHash]) -> int:
        hash_strs = [h.hex() for h in block_hashes]
        hash_frames = self.encoder.encode(hash_strs)
        token_len_bytes = token_len.to_bytes(4, byteorder="big")
        all_frames = [token_len_bytes] + list(hash_frames)
        self.socket.send_multipart(all_frames, copy=False)
        resp = self.socket.recv()
        result = int.from_bytes(resp, "big")
        return result

    def close(self):
        self.socket.close(linger=0)


def get_zmq_rpc_path_lookup(vllm_config: VllmConfig) -> str:
    """Construct IPC path for ZMQ lookup socket."""
    dp_rank = get_mooncake_dp_engine_index(vllm_config.parallel_config)
    base_url = envs.VLLM_RPC_BASE_PATH
    rpc_port = 0
    extra_config = vllm_config.kv_transfer_config.kv_connector_extra_config
    if "lookup_rpc_port" in extra_config:
        rpc_port = extra_config["lookup_rpc_port"]
    uid = os.getuid()
    logger.debug("Base URL: %s, RPC Port: %s, UID: %s", base_url, rpc_port, uid)
    return f"ipc://{base_url}/lookup_rpc_port_{rpc_port}_uid{uid}_dp_rank{dp_rank}"
