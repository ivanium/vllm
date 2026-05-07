# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Data classes for MooncakeStoreConnector."""

from dataclasses import dataclass
from functools import reduce
from math import lcm
from typing import Optional, cast

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
)
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_utils import BlockHash

logger = init_logger(__name__)


@dataclass
class KeyMetadata:
    """Metadata for constructing pool keys."""

    model_name: str
    tp_rank: int
    pcp_rank: int
    dcp_rank: int
    pp_rank: int


@dataclass(order=True)
class PoolKey:
    """Key for addressing KV cache blocks in the distributed store.

    `group_id=None` keeps the pre-HMA wire format. HMA callers pass an
    id so `to_string()` appends `@grp:{g}`.
    """

    key_metadata: KeyMetadata
    chunk_hash: str
    group_id: int | None = None

    def __hash__(self):
        return hash(
            (
                self.key_metadata.model_name,
                self.key_metadata.tp_rank,
                self.key_metadata.pcp_rank,
                self.key_metadata.dcp_rank,
                self.key_metadata.pp_rank,
                self.chunk_hash,
                self.group_id,
            )
        )

    def to_string(self) -> str:
        base = (
            f"{self.key_metadata.model_name}"
            f"@tp_rank:{self.key_metadata.tp_rank}"
            f"@pcp{self.key_metadata.pcp_rank}"
            f"@dcp{self.key_metadata.dcp_rank}"
            f"@pp_rank:{self.key_metadata.pp_rank}"
            f"@{self.chunk_hash}"
        )
        if self.group_id is None:
            return base
        return f"{base}@grp:{self.group_id}"


@dataclass
class GroupLayout:
    """Per-KV-cache-group memory layout.

    base_addrs and block_lens are flattened across the group's layers and
    K/V segments; entry i is one (layer, K|V) pair.
    """

    base_addrs: list[int]
    block_lens: list[int]


@dataclass
class GroupDatabase:
    """Token/address database for one KV cache group.

    This owns the group's memory layout and block sizing. The parent
    ``ChunkedTokenDatabase`` holds the group list and flat layout views, while
    group-specific callers use this object directly.
    """

    metadata: KeyMetadata
    group_id: int
    layout: GroupLayout
    block_size: int
    hash_block_size: int
    emit_group_id: bool = False

    def _make_key_by_hash(self, chunk_hash: str) -> PoolKey:
        return PoolKey(
            self.metadata,
            chunk_hash,
            self.group_id if self.emit_group_id else None,
        )

    def make_key(self, chunk_hash: BlockHash | bytes | str) -> PoolKey:
        """Build this group's Mooncake key from a right-edge chunk hash."""
        if isinstance(chunk_hash, str):
            hash_str = chunk_hash
        else:
            hash_str = bytes(chunk_hash).hex()
        return self._make_key_by_hash(hash_str)

    def _select_block_ids(
        self,
        block_ids: list[list[int]] | list[int],
    ) -> list[int]:
        """Accept either full per-group block_ids or this group's local list."""
        if block_ids and isinstance(block_ids[0], list):
            per_group = cast(list[list[int]], block_ids)
            if self.group_id >= len(per_group):
                return []
            return per_group[self.group_id]
        return cast(list[int], block_ids)

    def prepare_value(
        self,
        start: int,
        end: int,
        block_ids: list[list[int]] | list[int],
        total_chunks: int,
    ) -> tuple[list[int], list[int], int]:
        """Memory addrs + sizes for ``[start, end)`` in this group."""
        group_block_ids = self._select_block_ids(block_ids)
        chunk_id = start // self.block_size
        local_i = chunk_id - (total_chunks - len(group_block_ids))

        addr_list: list[int] = []
        size_list: list[int] = []
        last_block_id = 0
        for base_addr, block_len in zip(
            self.layout.base_addrs,
            self.layout.block_lens,
            strict=True,
        ):
            if not (0 <= local_i < len(group_block_ids)):
                # No resident block for this logical chunk. Coordinated
                # transfer plans avoid this; preserve list shape defensively.
                addr_list.append(0)
                size_list.append(0)
                continue
            block_id = group_block_ids[local_i]
            addr = base_addr + block_id * block_len
            size = int(block_len / self.block_size * (end - start))
            addr_list.append(addr)
            size_list.append(size)
            last_block_id = block_id
        return addr_list, size_list, last_block_id


class ChunkedTokenDatabase:
    """Maps token positions to store keys and GPU memory addresses."""

    def __init__(self, metadata: KeyMetadata, block_size: int):
        self.metadata = metadata
        self.block_size = block_size
        # Flat views over groups. `kv_caches_base_addr[seg_idx]` and
        # `block_len[seg_idx]` are what `prepare_value` iterates;
        # `set_groups` populates them by flattening `groups`.
        self.kv_caches_base_addr: list[int] = []
        self.block_len: list[int] = []
        self.groups: list[GroupLayout] = []
        # layer_to_group[seg_idx] -> which group owns segment seg_idx.
        self.layer_to_group: list[int] = []
        # Per-group native block_size in tokens (HMA: mixed; else uniform).
        self.group_block_sizes: list[int] = []
        # block_hashes granularity = GCD(group_block_sizes). Each chunk's
        # key uses block_hashes[end_idx // hash_block_size - 1] (right-edge).
        self.hash_block_size: int = block_size
        # vllm allocator alignment = LCM(group_block_sizes). Used for
        # lookup return-value rounding and mask_num. NOT cache_config.block_size:
        # engine mutates that to MIN(group_block_sizes) for HMA.
        self.scheduler_block_size: int = block_size
        self.group_dbs: list[GroupDatabase] = []

    def set_groups(
        self,
        groups: list[GroupLayout],
        layer_to_group: list[int],
    ) -> None:
        """Group-aware setter — flattens groups into kv_caches_base_addr/block_len."""
        self.groups = groups
        self.layer_to_group = layer_to_group
        flat_addrs: list[int] = []
        flat_lens: list[int] = []
        for layout in groups:
            flat_addrs.extend(layout.base_addrs)
            flat_lens.extend(layout.block_lens)
        self.kv_caches_base_addr = flat_addrs
        self.block_len = flat_lens
        self._rebuild_group_dbs()

    def set_group_block_sizes(
        self,
        group_block_sizes: list[int],
        hash_block_size: int,
        scheduler_block_size: int | None = None,
    ) -> None:
        """Set per-group native block sizes, hash block size, scheduler block size.

        `scheduler_block_size` defaults to LCM(group_block_sizes).
        """
        if self.groups and len(group_block_sizes) != len(self.groups):
            raise ValueError(
                f"group_block_sizes length {len(group_block_sizes)} != "
                f"num_groups {len(self.groups)}"
            )
        self.group_block_sizes = list(group_block_sizes)
        self.hash_block_size = max(1, hash_block_size)
        if scheduler_block_size is not None and scheduler_block_size > 0:
            self.scheduler_block_size = scheduler_block_size
        elif group_block_sizes:
            self.scheduler_block_size = reduce(lcm, group_block_sizes)
        else:
            self.scheduler_block_size = self.block_size
        self._rebuild_group_dbs()

    def _g_block_size(self, group_id: int) -> int:
        """Return group `group_id`'s native block_size (falls back to global)."""
        if self.group_block_sizes and group_id < len(self.group_block_sizes):
            return self.group_block_sizes[group_id]
        return self.block_size

    def _rebuild_group_dbs(self) -> None:
        """Rebuild per-group databases from the current group geometry."""
        emit_group_id = len(self.groups) > 1
        self.group_dbs = [
            GroupDatabase(
                metadata=self.metadata,
                group_id=group_id,
                layout=layout,
                block_size=self._g_block_size(group_id),
                hash_block_size=self.hash_block_size,
                emit_group_id=emit_group_id,
            )
            for group_id, layout in enumerate(self.groups)
        ]

    def group(self, group_id: int) -> GroupDatabase:
        """Return the independent token database for one KV cache group."""
        if not self.group_dbs and group_id == 0:
            return GroupDatabase(
                metadata=self.metadata,
                group_id=0,
                layout=GroupLayout(
                    base_addrs=self.kv_caches_base_addr,
                    block_lens=self.block_len,
                ),
                block_size=self.block_size,
                hash_block_size=self.hash_block_size,
                emit_group_id=False,
            )
        return self.group_dbs[group_id]


@dataclass
class LoadSpec:
    """Specification for loading KV cache from external store."""

    vllm_cached_tokens: int
    kvpool_cached_tokens: int
    can_load: bool
    token_len: int = 0


@dataclass
class RequestTracker:
    """Tracks per-request state across scheduler ticks."""

    req_id: str
    token_len: int
    # One list of block ids per KV cache group. Single-group models use [[...]]
    allocated_block_ids: list[list[int]]
    num_saved_tokens: int = 0
    token_ids: list[int] | None = None

    def update(
        self,
        new_block_ids: tuple[list[int], ...] | list[list[int]] | list[int],
    ) -> None:
        """Append new blocks to each group.

        Accepts:
          - tuple/list of per-group block lists (HMA shape)
          - flat list[int] for single-group callers
        """
        if len(new_block_ids) == 0:
            return

        if isinstance(new_block_ids, tuple):
            per_group = list(new_block_ids)
        elif isinstance(new_block_ids, list) and isinstance(new_block_ids[0], list):
            per_group = cast(list[list[int]], new_block_ids)
        elif isinstance(new_block_ids, list):
            self.allocated_block_ids[0].extend(cast(list[int], new_block_ids))
            return
        else:
            raise ValueError(f"Unsupported new_block_ids type {type(new_block_ids)}")

        assert len(per_group) == len(self.allocated_block_ids), (
            f"KV group count mismatch on update: got {len(per_group)} "
            f"groups, tracker has {len(self.allocated_block_ids)}"
        )
        for g, group_blocks in enumerate(per_group):
            self.allocated_block_ids[g].extend(group_blocks)


@dataclass
class ReqMeta:
    """Per-request metadata for store put/get operations."""

    req_id: str
    token_len_chunk: int
    block_ids: list[list[int]]
    block_hashes: list[BlockHash]

    can_save: bool | None = None
    load_spec: LoadSpec | None = None
    is_last_chunk: bool | None = None
    current_event: torch.cuda.Event | None = None

    token_ids: list[int] | None = None
    original_block_size: int | None = None

    @staticmethod
    def from_request_tracker(
        tracker: RequestTracker,
        block_size: int,
        load_spec: LoadSpec | None = None,
        skip_save: bool | None = False,
        block_hashes: list[BlockHash] | None = None,
        is_last_chunk: bool | None = None,
        discard_partial_chunks: bool = True,
        original_block_size: int | None = None,
    ) -> Optional["ReqMeta"]:
        """Create ReqMeta from a RequestTracker."""
        if block_hashes is None:
            block_hashes = []
        input_token_len = tracker.token_len

        chunk_boundary = (
            cdiv(tracker.num_saved_tokens + 1, block_size) * block_size
            if discard_partial_chunks
            else 0
        )
        num_tokens_to_save = (
            (input_token_len // block_size * block_size)
            if discard_partial_chunks
            else input_token_len
        )

        skip_save = skip_save or num_tokens_to_save < chunk_boundary
        if skip_save and load_spec is None:
            return None

        if not skip_save:
            tracker.num_saved_tokens = num_tokens_to_save

        token_ids = None
        if tracker.token_ids:
            token_ids = tracker.token_ids

        if load_spec is not None and load_spec.can_load:
            logger.debug(
                "Scheduled to load %d tokens for request %s",
                load_spec.kvpool_cached_tokens,
                tracker.req_id,
            )
        else:
            load_spec = None

        logger.debug(
            "request:%s, meta save spec:%s, meta load spec:%s",
            tracker.req_id,
            not skip_save,
            load_spec,
        )
        return ReqMeta(
            req_id=tracker.req_id,
            token_len_chunk=num_tokens_to_save,
            block_ids=tracker.allocated_block_ids,
            can_save=not skip_save,
            load_spec=load_spec,
            block_hashes=block_hashes,
            is_last_chunk=is_last_chunk,
            token_ids=token_ids,
            original_block_size=original_block_size,
        )


class MooncakeStoreConnectorMetadata(KVConnectorMetadata):
    """Metadata passed from scheduler to worker."""

    def __init__(
        self,
        unfinished_request_ids: set[str],
        preempted_req_ids: set[str],
    ):
        self.requests: list[ReqMeta] = []
        self.unfinished_request_ids = unfinished_request_ids
        self.preempted_req_ids = preempted_req_ids

    def add_request(self, req_meta: ReqMeta) -> None:
        self.requests.append(req_meta)
