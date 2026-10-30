# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure request/metadata helpers for the LMCache connectors.

This module never imports lmcache so that unit tests exercise it without the
optional lmcache package.
"""

import enum
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheConfig
from vllm.v1.utils import ConstantList

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_utils import BlockHash
    from vllm.v1.request import Request


def validate_transfer_groups(kv_cache_config: KVCacheConfig) -> None:
    """Fail fast when a transferable group is not token-paged attention.

    Every KV cache group defaults to ``enable_kv_transfer=True``, so hybrid
    models (e.g. attention + Mamba) put non-token-paged groups into the
    transfer set. LMCache addresses each group with a per-token slot mapping,
    which only token-paged attention groups support; admitting anything else
    crashes mid-serving or sends ragged per-group block lists.

    Raises:
        ValueError: if any transfer group's spec is not an AttentionSpec.

    """
    unsupported = [
        f"{group.layer_names} ({type(group.kv_cache_spec).__name__})"
        for group in kv_cache_config.transfer_groups
        if not isinstance(group.kv_cache_spec, AttentionSpec)
    ]
    if unsupported:
        raise ValueError(
            "LMCache only supports token-paged attention KV cache groups, "
            f"but the transfer set includes {unsupported}. Disable the "
            "LMCache connector for hybrid models with non-attention groups."
        )


def lmcache_kv_shapes(
    kv_cache_config: KVCacheConfig,
    chunk_size: int,
    use_mla: bool,
    tp_size: int,
) -> list[tuple[int, int, int, int, int]]:
    """One LMCache KV shape per transferable group, from each group's spec.

    Args:
        kv_cache_config: the large-page KV cache config. Transfer groups must
            be token-paged attention (see ``validate_transfer_groups``).
        chunk_size: the LMCache chunk size.
        use_mla: whether the model uses MLA; MLA latent KV is not TP-sharded.
        tp_size: the tensor parallel size.

    Returns:
        ``(num_layers, 1 if use_mla else 2, chunk_size, num_kv_heads,
        head_size)`` per transfer group, in group order, with per-rank
        ``num_kv_heads`` derived from the group's own spec.

    """
    kv_shapes = []
    for group in kv_cache_config.transfer_groups:
        spec = group.kv_cache_spec
        assert isinstance(spec, AttentionSpec), (
            f"transfer group {group.layer_names} is not token-paged attention"
        )
        num_kv_heads = 1 if use_mla else max(1, spec.num_kv_heads // tp_size)
        kv_shapes.append(
            (
                len(group.layer_names),
                1 if use_mla else 2,
                chunk_size,
                num_kv_heads,
                spec.head_size,
            )
        )
    return kv_shapes


@dataclass
class LoadStoreOp:
    block_ids: list[list[int]]
    """Per-group manager block ids for the load/store operation"""

    token_ids: list[int] | None = None
    """Token IDs for the load/store operation (token mode)"""

    block_hashes: list[bytes] | None = None
    """Block hashes for the load/store operation (hash mode)"""

    start: int = 0
    """Start token index (token mode only)"""

    end: int = 0
    """End token index (token mode only)"""

    skip_first_n_tokens: int = 0
    """Tokens the server must skip at the start of the retrieve range"""

    def __len__(self) -> int:
        return len(self.block_ids[0]) if self.block_ids else 0


class LMCacheMPRequestState(enum.Enum):
    """State machine:
    PREFETCHING -- update_state_after_alloc --> WAITING_FOR_LOAD
    WAITING_FOR_LOAD -- process_loading_requests --> READY
    """

    PREFETCHING = enum.auto()
    WAITING_FOR_LOAD = enum.auto()
    READY = enum.auto()


@dataclass
class LMCacheMPRequestTracker:
    # NOTE: this class used vLLM data structures, should be part of
    # vLLM integration code

    request_id: str

    # Read-only lists to track the token ids and block hashes
    all_token_ids: ConstantList[int]
    block_hashes: ConstantList["BlockHash"]

    # Block ids and hashes will be updated at update_states_after_alloc and
    # during the generation. One list per KV cache group; block ids are
    # unique only within a group.
    allocated_block_ids: list[list[int]] = field(default_factory=list)

    # Number of scheduled tokens in this request. We keep tracking this to
    # avoid saving half-full blocks.
    num_scheduled_tokens: int = 0

    # Number of blocks stored will be initialized when lookup the external
    # hit tokens and will be updated when processing new requests and cached
    # requests.
    num_stored_blocks: int = 0

    # Staging load operation -- save vllm and lmcache hit tokens during lookup
    num_vllm_hit_blocks: int = 0
    num_lmcache_hit_blocks: int = 0

    # Main state
    state: LMCacheMPRequestState = LMCacheMPRequestState.PREFETCHING

    cache_salt: str = ""

    def __init__(self, request: "Request"):
        self.request_id = request.request_id
        self.cache_salt: str = request.cache_salt or ""
        self.all_token_ids = request.all_token_ids
        self.block_hashes = ConstantList(request.block_hashes)
        self.allocated_block_ids = []
        self.num_stored_blocks = 0
        self.num_vllm_hit_blocks = 0
        self.num_lmcache_hit_blocks = 0
        self.state = LMCacheMPRequestState.PREFETCHING

    @property
    def num_allocated_blocks(self) -> int:
        """Allocated blocks per group (every token-paged group has the same
        number)."""
        return len(self.allocated_block_ids[0]) if self.allocated_block_ids else 0

    ####
    # Check the state of the request
    ####
    def needs_retrieve(self) -> bool:
        """Check whether the current request needs retrieve, will be used
        update_stage_after_alloc"""
        return (
            self.num_lmcache_hit_blocks > self.num_vllm_hit_blocks
            and self.state != LMCacheMPRequestState.READY
        )

    def is_ready_for_retrieving(self) -> bool:
        """Check whether the current request is ready for retrieving,
        will be used in process_loading_requests"""
        return (
            self.state == LMCacheMPRequestState.WAITING_FOR_LOAD
            and self.needs_retrieve()
        )

    ####
    # Update internal states
    ####
    def increase_num_scheduled_tokens(self, num_new_tokens: int):
        self.num_scheduled_tokens += num_new_tokens

    def increase_num_stored_blocks(self, num_new_blocks: int):
        """Increase the number of stored blocks for the current request
        This function will be called when processing the cached requests.
        """
        self.num_stored_blocks += num_new_blocks

    def append_block_ids(
        self,
        new_block_ids: tuple[list[int], ...] | None,
    ):
        """Update the block ids for the current request
        This function will be called when processing the cached requests.
        """
        if not new_block_ids:
            return
        if not self.allocated_block_ids:
            self.allocated_block_ids = [[] for _ in new_block_ids]
        for group_ids, new_ids in zip(
            self.allocated_block_ids, new_block_ids, strict=True
        ):
            group_ids.extend(new_ids)

    ####
    # For debugging
    ####
    def __repr__(self) -> str:
        return (
            f"LMCacheMPRequestTracker(request_id={self.request_id}, "
            f"num_tokens={len(self.all_token_ids)}, "
            f"num_block_hashes={len(self.block_hashes)}, "
            f"num_allocated_blocks={self.num_allocated_blocks}, "
            f"num_stored_blocks={self.num_stored_blocks}, "
            f"vllm_hit_blocks={self.num_vllm_hit_blocks}, "
            f"lmcache_hit_blocks={self.num_lmcache_hit_blocks}, "
            f"state={self.state})"
        )

    def __str__(self) -> str:
        return self.__repr__()


@dataclass
class LMCacheMPRequestMetadata:
    request_id: str
    direction: Literal["STORE", "RETRIEVE"]
    op: LoadStoreOp
    cache_salt: str = ""

    @staticmethod
    def GetStoreMetadata(
        tracker: LMCacheMPRequestTracker,
        blocks_in_chunk: int,
        vllm_block_size: int,
    ) -> "LMCacheMPRequestMetadata | None":
        """Generate the store metadata for the current request tracker.

        Args:
            tracker: The request tracker to generate the metadata from.
            blocks_in_chunk: the number of blocks in a LMCache data chunk
            vllm_block_size: the block size used in vLLM

        """
        # Store the blocks that has block hashes
        # NOTE: the invariant here is that `num_stored_blocks` should
        # always be a multiple of `blocks_in_chunk`
        # TODO: This should be checked everytime we update the num_stored_blocks
        #
        # Why computed_blocks uses max(num_vllm_hit_blocks, num_lmcache_hit_blocks):
        #
        # Both values represent a prefix of blocks whose KV data is already
        # available (either from vLLM APC or from LMCache), so they must NOT
        # be summed (that would double-count the overlapping prefix).
        #
        # * num_lmcache_hit_blocks: LMCache-hit blocks are already counted in
        #   num_stored_blocks (set during lookup), so they must be included
        #   here to keep the upper bound consistent.  They are NOT re-stored.
        # * num_vllm_hit_blocks: LMCache stores in units of chunks (N blocks),
        #   so num_lmcache_hit_blocks is rounded DOWN to the nearest chunk
        #   boundary.  When vLLM APC hits more blocks than that rounded value
        #   (e.g. APC=44 blocks, LMCache=32 blocks after chunk alignment),
        #   using only num_lmcache_hit_blocks would set the upper bound too
        #   low and silently skip the APC-hit blocks that fall between the
        #   two values, causing under-storing.  Taking the max ensures we
        #   always use the tighter (larger) of the two hit counts.
        computed_blocks = tracker.num_scheduled_tokens // vllm_block_size + max(
            tracker.num_vllm_hit_blocks, tracker.num_lmcache_hit_blocks
        )
        min_available_blocks = min(
            len(tracker.block_hashes),
            tracker.num_allocated_blocks,
            computed_blocks,
        )
        num_staging_blocks = min_available_blocks - tracker.num_stored_blocks
        num_chunks = num_staging_blocks // blocks_in_chunk

        if num_chunks >= 1:
            start = tracker.num_stored_blocks
            end = start + num_chunks * blocks_in_chunk
            block_ids = [ids[start:end] for ids in tracker.allocated_block_ids]
            start_token_idx = start * vllm_block_size
            end_token_idx = end * vllm_block_size
            token_ids = list(tracker.all_token_ids)
            op = LoadStoreOp(
                token_ids=token_ids,
                block_ids=block_ids,
                start=start_token_idx,
                end=end_token_idx,
            )

            ret = LMCacheMPRequestMetadata(
                request_id=tracker.request_id,
                direction="STORE",
                op=op,
                cache_salt=tracker.cache_salt,
            )

            # Update the request tracker
            tracker.increase_num_stored_blocks(end - start)
            return ret

        return None

    @staticmethod
    def GetRetrieveMetadata(
        tracker: LMCacheMPRequestTracker,
        blocks_in_chunk: int,
        vllm_block_size: int,
    ) -> "LMCacheMPRequestMetadata | None":
        """Generate the retrieve metadata for the current request tracker.

        Args:
            tracker: The request tracker to generate the metadata from.
            blocks_in_chunk: the number of blocks in a LMCache data chunk
            vllm_block_size: the block size used in vLLM

        """
        if not tracker.is_ready_for_retrieving():
            return None

        # |---------------------|-----------------|----------------|
        # | num_vllm_hit_blocks |
        # | lmcache chunk 1   | lmcache chunk 2   |
        #                     |  need to retrieve |

        start = tracker.num_vllm_hit_blocks // blocks_in_chunk * blocks_in_chunk
        end = tracker.num_lmcache_hit_blocks
        assert end % blocks_in_chunk == 0, (
            "The number of LMCache hit blocks should be a multiple of the "
            "number of blocks in a lmcache chunk. "
        )
        assert len(tracker.block_hashes) >= end, (
            "The number of block hashes should be greater than or equal to the "
            "number of LMCache hit blocks. "
        )
        if end > start:
            block_ids = [ids[start:end] for ids in tracker.allocated_block_ids]
            start_token_idx = start * vllm_block_size
            end_token_idx = end * vllm_block_size
            token_ids = list(tracker.all_token_ids)

            # Compute how many tokens at the start of the retrieve range
            # overlap with APC-shared blocks. The server must skip writing
            # to these positions to avoid a cross-stream data race: the
            # retrieve writes on the LMCache CUDA stream while concurrent
            # requests may read these APC-shared blocks on the vLLM stream.
            apc_overlap_blocks = tracker.num_vllm_hit_blocks - start
            skip_first_n_tokens = apc_overlap_blocks * vllm_block_size

            op = LoadStoreOp(
                token_ids=token_ids,
                block_ids=block_ids,
                start=start_token_idx,
                end=end_token_idx,
                skip_first_n_tokens=skip_first_n_tokens,
            )

            ret = LMCacheMPRequestMetadata(
                request_id=tracker.request_id,
                direction="RETRIEVE",
                op=op,
                cache_salt=tracker.cache_salt,
            )
            return ret

        return None


class LMCacheMPConnectorMetadata(KVConnectorMetadata):
    def __init__(self):
        super().__init__()
        self.requests: list[LMCacheMPRequestMetadata] = []

    def add_request_metadata(self, request_metadata: LMCacheMPRequestMetadata):
        self.requests.append(request_metadata)

    def __len__(self):
        return len(self.requests)

    # For debugging
    def __str__(self):
        request_strs = []
        for req_meta in self.requests:
            request_strs.append(
                f"RequestMetadata(request_id={req_meta.request_id}, "
                f"direction={req_meta.direction}, "
                f"num_blocks={len(req_meta.op)}, "
                f"block_ids={req_meta.op.block_ids})"
            )
        return "[" + "\n".join(request_strs) + "]"

    def __repr__(self):
        return self.__str__()
