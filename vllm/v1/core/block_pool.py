# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from vllm.distributed.kv_events import (
    MEDIUM_GPU,
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    KVCacheEvent,
)
from vllm.logger import init_logger
from vllm.multimodal.utils import get_mm_features_in_window
from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    BlockHashWithGroupId,
    ExternalBlockHash,
    FreeKVCacheBlockQueue,
    KVCacheBlock,
    generate_block_hash_extra_keys,
    get_block_hash,
    get_group_id,
    make_block_hash_with_group_id,
    maybe_convert_block_hash,
    resolve_block_hashes,
    to_event_extra_keys,
)
from vllm.v1.kv_cache_interface import (
    large_page_block_range,
    min_blocks_per_large_page,
)
from vllm.v1.request import Request

logger = init_logger(__name__)


class BlockHashToBlockMap:
    """Cache of blocks that are used for prefix caching. It caches blocks
    from hash directly to a block or multiple blocks
    (i.e. {block_hash: KVCacheBlocks})
    - Mostly block_hash maps to a single KVCacheBlock, and KVCacheBlocks
        would simply be a KVCacheBlock.
    - Otherwise, KVCacheBlocks is a dict from {block_id: KVCacheBlock}

    A cached block is a full block with a block hash that can be used
    for prefix caching.
    The cached block may be used by running requests or in the
    free_block_queue that could potentially be evicted.

    NOTE #1: We currently don't de-duplicate the blocks in the cache,
    meaning that if a block becomes full and is cached, we don't check
    if there is already an identical block in the cache. This is because
    we want to make sure the allocated block IDs won't change so that
    block tables are append-only.
    NOTE #2: The union type is introduced in order to reduce GC costs
    from the inner dict.
    """

    def __init__(self):
        self._cache: dict[
            BlockHashWithGroupId, KVCacheBlock | dict[int, KVCacheBlock]
        ] = {}

    def get_one_block(self, key: BlockHashWithGroupId) -> KVCacheBlock | None:
        """Gets any block with the given block hash key."""
        blocks = self._cache.get(key)
        if blocks is not None:
            if isinstance(blocks, KVCacheBlock):
                return blocks
            if isinstance(blocks, dict):
                return next(iter(blocks.values()))
            self._unexpected_blocks_type(blocks)
        return None

    def contain(self, key: BlockHashWithGroupId, block_id: int) -> bool:
        """Checks whether the key maps to the given block ID."""
        blocks = self._cache.get(key)
        if blocks is None:
            return False
        if isinstance(blocks, KVCacheBlock):
            return blocks.block_id == block_id
        if isinstance(blocks, dict):
            return block_id in blocks
        self._unexpected_blocks_type(blocks)
        return False

    def insert(self, key: BlockHashWithGroupId, block: KVCacheBlock) -> None:
        """Inserts the KVCacheBlock to the cache"""
        blocks = self._cache.get(key)
        if blocks is None:
            # When key is not found, attach a single block to the key
            self._cache[key] = block
        elif isinstance(blocks, KVCacheBlock):
            # If there's a block with the same key, merge the original block
            # and the new block into a dict
            self._cache[key] = {blocks.block_id: blocks, block.block_id: block}
        elif isinstance(blocks, dict):
            # If it's already a dict, simply insert the block
            blocks[block.block_id] = block
        else:
            self._unexpected_blocks_type(blocks)

    def pop(self, key: BlockHashWithGroupId, block_id: int) -> KVCacheBlock | None:
        """Checks if block_hash exists and pop block_id from the cache"""
        blocks = self._cache.pop(key, None)
        if blocks is None:
            # block_hash not found in the cache
            return None
        # TODO(Jialin): If key is found, block_id should always present
        # in blocks. We currently keep the original behaviour for safety.
        #
        # Will add block_id == blocks.block_id assertion and
        # use del blocks[block_id] instead as followup.
        if isinstance(blocks, KVCacheBlock):
            if blocks.block_id == block_id:
                return blocks
            # If the single block ID doesn't match, we should put the
            # block back (it should happen rarely)
            self._cache[key] = blocks
            return None
        if isinstance(blocks, dict):
            # Try to pop block_id from the block dict, and if dict still
            # contain blocks, put back to the cache.
            block = blocks.pop(block_id, None)
            if len(blocks) > 0:
                self._cache[key] = blocks
            return block
        self._unexpected_blocks_type(blocks)
        return None

    def __len__(self) -> int:
        return len(self._cache)

    def _unexpected_blocks_type(self, blocks: Any) -> None:
        raise AssertionError(f"Invalid KV cache block type {type(blocks)}")


@dataclass(slots=True)
class _Page:
    """Bookkeeping of one page of the pool."""

    # The group with more blocks per page whose blocks the page holds; -1 when
    # the page is the one block of a group with one block per page.
    owner: int = -1
    # Blocks of the page with ref_cnt > 0.
    num_in_use: int = 0
    # Free blocks of the page that still carry a prefix-cache hash.
    num_cached: int = 0
    # While the page is in use, bit i is set if slot i is free and has no hash.
    free_mask: int = 0
    # The blocks of the page in slot order.
    slots: list[KVCacheBlock] = field(default_factory=list)


class BlockPool:
    """BlockPool that manages KVCacheBlocks.
    It provides methods to allocate, free and cache the kv cache blocks. The
    free_block_queue stores the free blocks in eviction order to enable
    allocation, free, and cache eviction. The cached_block_hash_to_block
    maps between block hash and cached block to support finding cached blocks
    by their block hash.

    The pool is a set of pages of ``large_page_bytes``. A page holds the
    blocks of the one group ``g`` that allocated it that lie wholly in it:
    block ``b`` spans bytes ``[b * block_bytes[g], (b + 1) * block_bytes[g])``
    (see ``large_page_block_range``). A group whose block fills a page (every
    group under the ``uniform_page`` allocator) uses the page as its block,
    so ``blocks`` and ``free_block_queue`` hold its blocks. A group with more
    blocks per page allocates, in order: free blocks on its in-use pages
    (fullest page first, so that pages empty and return to the pool), a
    page without cached blocks, its cached free blocks on in-use pages (LRU),
    then any page from the front of ``free_block_queue``, whose cached blocks
    are evicted. Taking clean pages before evicting cached blocks keeps the
    prefix cache while the pool has slack.

    Args:
        num_gpu_blocks: The number of pages in the pool.
        enable_caching: Whether to enable prefix caching.
        hash_block_size: The block size of which the block hashes are computed.
            The actual block size usually equals hash_block_size, but in cases
            where different KV cache groups have different block sizes, the
            actual block size can be a multiple of hash_block_size.
        enable_kv_cache_events: Whether to enable kv cache events.
        metrics_collector: Optional metrics collector for tracking block residency.
        medium: Storage medium reported in KV cache events.
        large_page_bytes: Bytes of one page.
        block_bytes: Bytes of one block of each KV cache group. Groups past
            the end of the sequence, or whose block fills a page, use the page
            as their block.

    """

    def __init__(
        self,
        num_gpu_blocks: int,
        enable_caching: bool,
        hash_block_size: int,
        enable_kv_cache_events: bool = False,
        metrics_collector: KVCacheMetricsCollector | None = None,
        medium: str = MEDIUM_GPU,
        large_page_bytes: int = 0,
        block_bytes: Sequence[int] = (),
    ):
        assert isinstance(num_gpu_blocks, int) and num_gpu_blocks > 0
        self.num_gpu_blocks = num_gpu_blocks
        self.medium = medium
        self.enable_caching = enable_caching
        self.hash_block_size = hash_block_size
        self.large_page_bytes = large_page_bytes
        # Bytes of one block of each group whose block is smaller than a page.
        self._block_bytes = {
            g: b for g, b in enumerate(block_bytes) if 0 < b < large_page_bytes
        }
        # Fewest blocks a page holds, per group with blocks smaller than a page.
        self.slots_per_page = {
            g: min_blocks_per_large_page(large_page_bytes, b)
            for g, b in self._block_bytes.items()
        }
        # Size of the block id space of those groups.
        self._num_group_blocks = {
            g: num_gpu_blocks * large_page_bytes // b
            for g, b in self._block_bytes.items()
        }
        # The first block of every page. For a group with more blocks per
        # page, it only links the page into free_block_queue.
        self.blocks: list[KVCacheBlock] = [
            KVCacheBlock(idx, pool=self, group_id=-1) for idx in range(num_gpu_blocks)
        ]
        self._pages = [_Page(slots=[block]) for block in self.blocks]
        # Free block queue that constructs and manipulates a doubly linked
        # list of pages with no block in use (including eviction candidates
        # when caching is enabled).
        self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)
        # Free blocks on in-use pages of groups with more blocks per page:
        # the number without a hash, the in-use pages that hold them by
        # their number of blocks in use, and the cached ones in LRU order.
        self._num_free_slots = dict.fromkeys(self.slots_per_page, 0)
        self._pages_by_use = {
            g: [FreeKVCacheBlockQueue([]) for _ in range(large_page_bytes // b)]
            for g, b in self._block_bytes.items()
        }
        self._cached_slots = {g: FreeKVCacheBlockQueue([]) for g in self._block_bytes}
        # Release counter for free_seq.
        self._free_seq = 0

        # Cache for block lookup
        self.cached_block_hash_to_block: BlockHashToBlockMap = BlockHashToBlockMap()
        # Secondary (non-primary) hash entries per block. Block ids are unique
        # only within a group, so the key is (group_id, block_id).
        self.cached_block_hashes_by_block: dict[
            tuple[int, int], set[BlockHashWithGroupId]
        ] = {}

        # To represent a placeholder block with block_id=0.
        # The ref_cnt of null_block is not maintained, needs special care to
        # avoid freeing it.
        self.null_block = self.free_block_queue.popleft()
        self.null_block.is_null = True

        self.enable_kv_cache_events = enable_kv_cache_events
        self.kv_event_queue: list[KVCacheEvent] = []

        self.metrics_collector = metrics_collector
        # Callbacks for blocks released with ``unpin_blocks`` whose contents
        # are still being read until the pool reuses them.
        self._reuse_watchers: dict[int, Callable[[KVCacheBlock], None]] = {}

    def get_block(self, group_id: int, block_id: int) -> KVCacheBlock:
        """The allocated block ``block_id`` of a group."""
        assert block_id < self._num_group_blocks.get(group_id, self.num_gpu_blocks), (
            f"Invalid block_id {block_id} of group {group_id}. This indicates "
            "a bug in the KV connector - workers should only report block IDs "
            "that were allocated by the scheduler."
        )
        if group_id not in self._block_bytes or block_id == 0:
            # Block 0 of every group is the null block.
            return self.blocks[block_id]
        page = self._pages[self._page_id(group_id, block_id)]
        index = block_id - page.slots[0].block_id
        assert page.owner == group_id and 0 <= index < len(page.slots), (
            f"Block {block_id} of group {group_id} is not allocated"
        )
        return page.slots[index]

    def _page_id(self, group_id: int, block_id: int) -> int:
        """The page that holds a block of a group with blocks smaller than a
        page."""
        return block_id * self._block_bytes[group_id] // self.large_page_bytes

    def get_cached_block(
        self, block_hash: BlockHash, kv_cache_group_ids: list[int]
    ) -> list[KVCacheBlock] | None:
        """Get the cached block by the block hash for each group in
        `kv_cache_group_ids`, or None if cache miss for any group.
        If there are duplicated blocks, we return the first block in the cache.

        Args:
            block_hash: The hash value of the block.
            kv_cache_group_ids: The ids of the KV cache groups.

        Returns:
            The cached blocks if exists, or None.

        """
        cached_blocks = []
        for group_id in kv_cache_group_ids:
            block_hash_with_group_id = make_block_hash_with_group_id(
                block_hash, group_id
            )
            block = self.cached_block_hash_to_block.get_one_block(
                block_hash_with_group_id
            )
            if not block:
                return None
            cached_blocks.append(block)
        return cached_blocks

    def cache_full_blocks(
        self,
        request: Request,
        blocks: list[KVCacheBlock],
        num_cached_blocks: int,
        num_full_blocks: int,
        block_size: int,
        kv_cache_group_id: int,
        block_mask: list[bool] | None = None,
    ) -> None:
        """Cache a list of full blocks for prefix caching.
        This function takes a list of blocks that will have their block hash
        metadata to be updated and cached. Given a request, it updates the
        metadata for each block and caching it in the
        `cached_block_hash_to_block`.
        The block hashes values are computed by the Request object immediately
        when it is created and when new tokens are appended.

        Args:
            request: The request to cache the blocks.
            blocks: All blocks in the request.
            num_cached_blocks: The number of blocks that are already cached.
            num_full_blocks: The number of blocks that are full and should
                be cached after this function.
            block_size: Number of tokens in each block.
            kv_cache_group_id: The id of the KV cache group.
            block_mask: Optional mask aligned with
                ``blocks[num_cached_blocks:num_full_blocks]``. When provided,
                blocks where the mask is False are skipped (treated like null
                blocks). Used by groups whose ``find_longest_cache_hit`` only
                consults a subset of blocks (e.g. SWA tail-window), so blocks
                that can never serve a hit stay out of the prefix-cache hash
                map.

        """
        if num_cached_blocks >= num_full_blocks:
            return
        new_full_blocks = blocks[num_cached_blocks:num_full_blocks]
        assert block_mask is None or len(block_mask) == len(new_full_blocks)
        block_hashes = resolve_block_hashes(
            request.block_hashes, self.hash_block_size, block_size
        )

        new_block_hashes = block_hashes[num_cached_blocks:]
        new_hashes: list[ExternalBlockHash] | None = (
            [] if self.enable_kv_cache_events else None
        )
        for i, blk in enumerate(new_full_blocks):
            # Some blocks may be null or masked out when enabling sparse attention
            # like sliding window attention, or Mamba models with prefix-caching
            # in align mode. We skip null blocks here.
            if blk.is_null or (block_mask is not None and not block_mask[i]):
                continue
            block_hash = new_block_hashes[i]
            num_hash_tokens = (num_cached_blocks + i + 1) * block_size

            # Update and added the full block to the cache.
            block_hash_with_group_id = make_block_hash_with_group_id(
                block_hash, kv_cache_group_id
            )
            if blk.block_hash is not None:
                # The only valid case where a "new full block" already has a
                # hash is partial->full promotion of the same cache block.
                assert (
                    blk.block_hash_num_tokens is not None
                    and blk.block_hash_num_tokens < num_hash_tokens
                )
                removed_hashes = self._remove_cached_block_hashes(blk)
                self._emit_block_removed_events(removed_hashes)
            self._insert_block_hash(
                block_hash_with_group_id,
                blk,
                num_tokens=num_hash_tokens,
            )
            if new_hashes is not None:
                new_hashes.append(maybe_convert_block_hash(block_hash))

        if self.enable_kv_cache_events:
            if num_cached_blocks == 0:
                parent_block_hash: ExternalBlockHash | None = None
            else:
                parent_block_hash = maybe_convert_block_hash(
                    block_hashes[num_cached_blocks - 1]
                )

            # Calculate token range for the blocks being cached
            start_token_idx = num_cached_blocks * block_size
            end_token_idx = num_full_blocks * block_size

            # Generate extra keys for each block individually.
            # Each block may have different extra_keys (e.g., different MM
            # features, or cache_salt only for the first block).
            # Skip null/masked-out blocks to match the length of new_hashes.
            extra_keys_list: list[tuple[Any, ...] | None] = []
            curr_mm_idx = 0
            for i in range(num_cached_blocks, num_full_blocks):
                if blocks[i].is_null:
                    continue
                if block_mask is not None and not block_mask[i - num_cached_blocks]:
                    continue
                block_start = i * block_size
                block_end = block_start + block_size
                extra_keys, curr_mm_idx = generate_block_hash_extra_keys(
                    request, block_start, block_end, curr_mm_idx
                )
                extra_keys_list.append(extra_keys)

            self.kv_event_queue.append(
                self._build_block_stored_event(
                    request,
                    block_hashes=new_hashes,
                    parent_block_hash=parent_block_hash,
                    start_token_idx=start_token_idx,
                    end_token_idx=end_token_idx,
                    block_size=block_size,
                    kv_cache_group_id=kv_cache_group_id,
                    extra_keys_list=extra_keys_list,
                )
            )

    def _build_block_stored_event(
        self,
        request: Request,
        block_hashes: list[ExternalBlockHash] | None,
        parent_block_hash: ExternalBlockHash | None,
        start_token_idx: int,
        end_token_idx: int,
        block_size: int,
        kv_cache_group_id: int,
        extra_keys_list: list[tuple[Any, ...] | None],
    ) -> BlockStored:
        """Build a ``BlockStored`` KV event for ``request``.

        Shared by ``cache_full_blocks`` (newly cached blocks) and
        ``emit_cached_block_events`` (prefix-cache-reused blocks) so both emit
        identical event shapes for downstream consumers.
        """
        return BlockStored(
            block_hashes=block_hashes,
            parent_block_hash=parent_block_hash,
            token_ids=request.all_token_ids[start_token_idx:end_token_idx],
            block_size=block_size,
            lora_id=request.lora_request.adapter_id if request.lora_request else None,
            medium=self.medium,
            lora_name=request.lora_request.name if request.lora_request else None,
            extra_keys=to_event_extra_keys(extra_keys_list),
            group_idx=kv_cache_group_id,
            session_id=request.session_id,
        )

    def emit_cached_block_events(
        self,
        request: Request,
        num_cached_blocks: int,
        block_size: int,
        kv_cache_group_id: int,
    ) -> None:
        """Generate BlockStored events for blocks reused from prefix cache.

        Unlike cache_full_blocks(), this does NOT modify block state —
        the blocks are already cached. It only generates events so that
        external consumers (e.g. gateway) can learn about reused blocks.

        Args:
            request: The request whose prefix cache blocks were reused.
            num_cached_blocks: Number of blocks that were cache hits.
            block_size: Number of tokens per block.
            kv_cache_group_id: The KV cache group ID.

        """
        if not self.enable_kv_cache_events or num_cached_blocks == 0:
            return

        block_hashes = resolve_block_hashes(
            request.block_hashes, self.hash_block_size, block_size
        )

        # Collect external hashes and extra_keys for cached blocks.
        cached_hashes: list[ExternalBlockHash] = []
        extra_keys_list: list[tuple[Any, ...] | None] = []
        curr_mm_idx = 0
        for i in range(num_cached_blocks):
            block_start = i * block_size
            block_end = block_start + block_size
            cached_hashes.append(maybe_convert_block_hash(block_hashes[i]))
            extra_keys, curr_mm_idx = generate_block_hash_extra_keys(
                request, block_start, block_end, curr_mm_idx
            )
            extra_keys_list.append(extra_keys)

        if not cached_hashes:
            return

        # Prefix-cache hits always form a contiguous prefix starting at block 0,
        # so the first (and thus the whole group's) parent block hash is None.
        parent_block_hash: ExternalBlockHash | None = None
        start_token_idx = 0
        end_token_idx = num_cached_blocks * block_size

        logger.debug(
            "EmitCachedBlock event: block_size=%d, "
            "num_cached_blocks=%d, parent_block_hash=%s, "
            "token_ids_len=%d, group_idx=%s",
            block_size,
            num_cached_blocks,
            parent_block_hash,
            len(request.all_token_ids[start_token_idx:end_token_idx]),
            kv_cache_group_id,
        )

        self.kv_event_queue.append(
            self._build_block_stored_event(
                request,
                block_hashes=cached_hashes,
                parent_block_hash=parent_block_hash,
                start_token_idx=start_token_idx,
                end_token_idx=end_token_idx,
                block_size=block_size,
                kv_cache_group_id=kv_cache_group_id,
                extra_keys_list=extra_keys_list,
            )
        )

    def cache_partial_block(
        self,
        request: Request,
        block: KVCacheBlock,
        num_tokens: int,
        kv_cache_group_id: int,
        block_size: int,
        replace_existing_hashes: bool = False,
    ) -> BlockHashWithGroupId | None:
        """Register a partial prefix-cache entry for an existing block.

        Prefix-cache keys normally identify full cache blocks. A partial entry
        makes an existing cache block reachable from a fine-grained prefix
        boundary inside that block without allocating or copying a new
        ``KVCacheBlock``.

        The partial entry is lookup metadata owned by ``block``. If ``block``
        has no primary hash, the key becomes its primary hash. If the block
        already has a primary hash, the partial entry is tracked in
        ``cached_block_hashes_by_block`` so eviction, reset, and promotion can
        remove every hash key that points to the block.

        Args:
            request: Request whose token IDs and block hashes define the
                partial entry.
            block: Existing cache block to make reachable from the partial
                prefix boundary.
            num_tokens: Prefix length represented by the partial entry. It
                must be a positive multiple of ``self.hash_block_size`` and
                cannot exceed the request's computed block hashes.
            kv_cache_group_id: KV cache group that owns the partial entry.
            block_size: Cache block size for the owning group. The partial
                entry hash itself is always the prefix-chain hash at
                ``num_tokens``; ``block_size`` must be a multiple of the hash
                block size.
            replace_existing_hashes: Whether the block contents were replaced
                and all existing cache entries must be removed before the new
                entry is registered.

        Returns:
            The hash key with group ID if a partial entry can be registered;
            otherwise ``None`` for null blocks.

        """
        if block.is_null:
            return None

        assert block_size % self.hash_block_size == 0
        block_hash = self._get_partial_block_hash(request, num_tokens)
        num_hash_blocks = num_tokens // self.hash_block_size
        block_hash_with_group_id = make_block_hash_with_group_id(
            block_hash, kv_cache_group_id
        )
        already_cached = block.block_hash == block_hash_with_group_id or (
            self.cached_block_hash_to_block.contain(
                block_hash_with_group_id, block.block_id
            )
        )
        if replace_existing_hashes:
            removed_hashes = self._remove_cached_block_hashes(block)
            self._emit_block_removed_events(removed_hashes)
            already_cached = False
        elif (
            not already_cached
            and block.block_hash is not None
            and block.block_hash_num_tokens is not None
            and block.block_hash_num_tokens < num_hash_blocks * self.hash_block_size
        ):
            removed_hashes = self._remove_cached_block_hashes(block)
            self._emit_block_removed_events(removed_hashes)
        self._insert_block_hash(
            block_hash_with_group_id,
            block,
            num_tokens=num_hash_blocks * self.hash_block_size,
        )
        if self.enable_kv_cache_events and not already_cached:
            parent_hash, block_start = self._get_partial_block_parent_hash_and_start(
                request, num_tokens
            )
            parent_block_hash = (
                maybe_convert_block_hash(parent_hash)
                if parent_hash is not None
                else None
            )
            block_end = num_tokens
            curr_mm_idx = 0
            mm_features = request.mm_features
            if block_start > 0 and mm_features:
                last_mm_pos = mm_features[-1].mm_position
                if last_mm_pos.offset + last_mm_pos.length > block_start:
                    curr_mm_idx, _ = get_mm_features_in_window(
                        mm_features, block_start, block_end
                    )
            extra_keys, _ = generate_block_hash_extra_keys(
                request, block_start, block_end, curr_mm_idx
            )
            self.kv_event_queue.append(
                BlockStored(
                    block_hashes=[maybe_convert_block_hash(block_hash)],
                    parent_block_hash=parent_block_hash,
                    token_ids=request.all_token_ids[block_start:block_end],
                    block_size=block_end - block_start,
                    lora_id=request.lora_request.adapter_id
                    if request.lora_request
                    else None,
                    medium=self.medium,
                    lora_name=request.lora_request.name
                    if request.lora_request
                    else None,
                    extra_keys=to_event_extra_keys([extra_keys]),
                    group_idx=kv_cache_group_id,
                    session_id=request.session_id,
                )
            )
        return block_hash_with_group_id

    def _get_partial_block_hash(
        self,
        request: Request,
        num_tokens: int,
    ) -> BlockHash:
        assert num_tokens % self.hash_block_size == 0
        num_hash_blocks = num_tokens // self.hash_block_size
        assert 0 < num_hash_blocks <= len(request.block_hashes)

        # Each hash_block_size hash chains over its full prefix, so the partial
        # entry for any group block size is the hash at that prefix boundary.
        return request.block_hashes[num_hash_blocks - 1]

    def _get_partial_block_parent_hash_and_start(
        self,
        request: Request,
        num_tokens: int,
    ) -> tuple[BlockHash | None, int]:
        num_hash_blocks = num_tokens // self.hash_block_size
        parent_hash = (
            request.block_hashes[num_hash_blocks - 2] if num_hash_blocks > 1 else None
        )
        block_start = (num_hash_blocks - 1) * self.hash_block_size
        return parent_hash, block_start

    def _remove_cached_block_hashes(
        self,
        block: KVCacheBlock,
    ) -> list[BlockHashWithGroupId]:
        block_hashes: list[BlockHashWithGroupId] = []
        if block.block_hash is not None:
            block_hashes.append(block.block_hash)
        block_hashes.extend(
            self.cached_block_hashes_by_block.pop((block.group_id, block.block_id), ())
        )
        if not block_hashes:
            return []

        removed_hashes: list[BlockHashWithGroupId] = []
        for block_hash in block_hashes:
            if (
                self.cached_block_hash_to_block.pop(block_hash, block.block_id)
                is not None
            ):
                removed_hashes.append(block_hash)
        block.reset_hash()
        return removed_hashes

    def _emit_block_removed_events(
        self,
        block_hashes: list[BlockHashWithGroupId],
    ) -> None:
        if not self.enable_kv_cache_events:
            return
        for block_hash in block_hashes:
            self.kv_event_queue.append(
                BlockRemoved(
                    block_hashes=[maybe_convert_block_hash(get_block_hash(block_hash))],
                    medium=self.medium,
                    group_idx=get_group_id(block_hash),
                )
            )

    def _insert_block_hash(
        self,
        block_hash_with_group_id: BlockHashWithGroupId,
        block: KVCacheBlock,
        num_tokens: int | None,
    ) -> None:
        if block.block_hash == block_hash_with_group_id:
            return

        if self.cached_block_hash_to_block.contain(
            block_hash_with_group_id, block.block_id
        ):
            return

        if block.block_hash is None:
            block.set_block_hash(block_hash_with_group_id, num_tokens=num_tokens)
        else:
            self.cached_block_hashes_by_block.setdefault(
                (block.group_id, block.block_id), set()
            ).add(block_hash_with_group_id)
        self.cached_block_hash_to_block.insert(block_hash_with_group_id, block)

    def move_block_hashes(
        self,
        src_block: KVCacheBlock,
        dst_block: KVCacheBlock,
    ) -> None:
        """Re-point ``src_block``'s prefix-cache entries to ``dst_block``.

        Used when the request owning ``src_block`` keeps writing into it
        : the prefix cache holds a private copy (``dst_block``)
        under the same hashes instead. Entries stay live; no events emitted.
        """
        assert dst_block.block_hash is None
        assert (
            dst_block.group_id,
            dst_block.block_id,
        ) not in self.cached_block_hashes_by_block
        num_tokens = src_block.block_hash_num_tokens
        for block_hash in self._remove_cached_block_hashes(src_block):
            # `num_tokens` only applies to the first (primary) insertion.
            self._insert_block_hash(block_hash, dst_block, num_tokens=num_tokens)

    def get_new_blocks(self, num_blocks: int, group_id: int = 0) -> list[KVCacheBlock]:
        """Get new blocks from the free block pool.

        Note that we do not check block cache in this function.

        Args:
            num_blocks: The number of blocks to allocate.
            group_id: The KV cache group that allocates.

        Returns:
            A list of new block.

        """
        if num_blocks > self.get_num_free_blocks(group_id):
            raise ValueError(f"Cannot get {num_blocks} free blocks from the pool")
        if group_id not in self._block_bytes:
            ret = self.free_block_queue.popleft_n(num_blocks)
            for block in ret:
                if self.slots_per_page and self._pages[block.block_id].owner != -1:
                    # The page holds blocks of a group with more blocks per page.
                    self._take_page(block.block_id, group_id)
                    continue
                if self.enable_caching:
                    self._maybe_evict_cached_block(block)
                block.group_id = group_id
        else:
            ret = self._get_new_slots(num_blocks, group_id)

        if self._reuse_watchers:
            self._notify_reuse(ret)

        for block in ret:
            assert block.ref_cnt == 0
            block.ref_cnt += 1
            if self.metrics_collector:
                self.metrics_collector.on_block_allocated(block)
        return ret

    def _get_new_slots(self, num_blocks: int, group_id: int) -> list[KVCacheBlock]:
        pages_by_use = self._pages_by_use[group_id]
        cached_slots = self._cached_slots[group_id]
        ret: list[KVCacheBlock] = []
        while len(ret) < num_blocks:
            if self._num_free_slots[group_id]:
                fullest = next(q for q in reversed(pages_by_use) if q.num_free_blocks)
                front = fullest.fake_free_list_head.next_free_block
                assert front is not None
                page_id = front.block_id
                page = self._pages[page_id]
                free_mask = page.free_mask
            elif cached_slots.num_free_blocks and not self._front_page_is_older(
                cached_slots.fake_free_list_head.next_free_block
            ):
                block = cached_slots.popleft()
                page_id = self._page_id(group_id, block.block_id)
                page = self._pages[page_id]
                page.num_cached -= 1
                if self.enable_caching:
                    self._maybe_evict_cached_block(block)
                self._update_page(page_id, page, page.num_in_use + 1, page.free_mask)
                ret.append(block)
                continue
            else:
                page_id = self.free_block_queue.popleft().block_id
                self._take_page(page_id, group_id)
                page = self._pages[page_id]
                free_mask = (1 << len(page.slots)) - 1
            # The page stays the fullest while it gives out its free blocks.
            num_in_use = page.num_in_use
            while free_mask and len(ret) < num_blocks:
                block = page.slots[(free_mask & -free_mask).bit_length() - 1]
                free_mask &= free_mask - 1
                num_in_use += 1
                ret.append(block)
            self._update_page(page_id, page, num_in_use, free_mask)
        return ret

    def _front_page(self) -> KVCacheBlock | None:
        """The first block of the page at the front of free_block_queue."""
        if self.free_block_queue.num_free_blocks == 0:
            return None
        return self.free_block_queue.fake_free_list_head.next_free_block

    def _has_clean_page(self) -> bool:
        """Whether free_block_queue starts with a page without cached blocks."""
        front = self._front_page()
        if front is None:
            return False
        page = self._pages[front.block_id]
        if page.owner != -1:
            return page.num_cached == 0
        return front.block_hash is None

    def _front_page_is_older(self, block: KVCacheBlock | None) -> bool:
        """Whether the front of free_block_queue is clean or was released
        before ``block``, so taking it keeps the LRU order."""
        front = self._front_page()
        if front is None:
            return False
        assert block is not None
        return self._has_clean_page() or front.free_seq < block.free_seq

    def _stamp(self, blocks: Iterable[KVCacheBlock]) -> None:
        # One batch joins its queue at once, so its blocks share a stamp.
        self._free_seq += 1
        for block in blocks:
            block.free_seq = self._free_seq

    def _take_page(self, page_id: int, group_id: int) -> None:
        """Give a page popped from free_block_queue to a group.

        Evicts the cached blocks of the page.
        """
        page = self._pages[page_id]
        # A page owned by a multi-block group counts its cached blocks.
        if self.enable_caching and (page.owner == -1 or page.num_cached):
            for block in page.slots:
                self._maybe_evict_cached_block(block)
        page.num_cached = 0
        if group_id not in self._block_bytes:
            block = self.blocks[page_id]
            block.group_id = group_id
            page.owner = -1
            page.slots = [block]
        elif page.owner != group_id:
            page.owner = group_id
            page.slots = [
                KVCacheBlock(block_id, pool=self, group_id=group_id)
                for block_id in large_page_block_range(
                    page_id, self.large_page_bytes, self._block_bytes[group_id]
                )
            ]

    # A group with one block per page keeps no page counts: its block is the
    # page. The helpers below serve groups with more blocks per page.

    def _update_page(
        self, page_id: int, page: _Page, num_in_use: int, free_mask: int
    ) -> None:
        """Set the blocks in use and the free mask of a page, and file the
        page by its blocks in use while it holds free blocks without a hash."""
        pages_by_use = self._pages_by_use[page.owner]
        if page.num_in_use and page.free_mask:
            pages_by_use[page.num_in_use].remove(self.blocks[page_id])
        if num_in_use and free_mask:
            pages_by_use[num_in_use].append(self.blocks[page_id])
        self._num_free_slots[page.owner] += (
            free_mask.bit_count() - page.free_mask.bit_count()
        )
        page.num_in_use = num_in_use
        page.free_mask = free_mask

    def _acquire(self, block: KVCacheBlock) -> None:
        """Take a free block out of the free queues before it is reused."""
        page_id = self._page_id(block.group_id, block.block_id)
        page = self._pages[page_id]
        if page.num_in_use == 0:
            self.free_block_queue.remove(self.blocks[page_id])
            free_mask = 0
            for i, slot in enumerate(page.slots):
                if slot is block:
                    continue
                if slot.block_hash is None:
                    free_mask |= 1 << i
                else:
                    self._cached_slots[block.group_id].append(slot)
        elif block.block_hash is None:
            free_mask = page.free_mask & ~(
                1 << (block.block_id - page.slots[0].block_id)
            )
        else:
            self._cached_slots[block.group_id].remove(block)
            free_mask = page.free_mask
        page.num_cached -= block.block_hash is not None
        self._update_page(page_id, page, page.num_in_use + 1, free_mask)

    def _release(self, block: KVCacheBlock) -> KVCacheBlock | None:
        """Account a block whose ref_cnt dropped to 0.

        Returns:
            The first block of the page if no block of the page is in use any
            more. Else None: the block joins the free blocks of its group.

        """
        page_id = self._page_id(block.group_id, block.block_id)
        page = self._pages[page_id]
        if block.block_hash is not None:
            page.num_cached += 1
            self._stamp([block])
        if page.num_in_use == 1:
            for slot in page.slots:
                if slot is not block and slot.block_hash is not None:
                    self._cached_slots[block.group_id].remove(slot)
            self._update_page(page_id, page, 0, 0)
            return self.blocks[page_id]
        free_mask = page.free_mask
        if block.block_hash is None:
            free_mask |= 1 << (block.block_id - page.slots[0].block_id)
        else:
            self._cached_slots[block.group_id].append(block)
        self._update_page(page_id, page, page.num_in_use - 1, free_mask)
        return None

    def _notify_reuse(self, blocks: Iterable[KVCacheBlock]) -> None:
        for block in blocks:
            watcher = self._reuse_watchers.pop(block.block_id, None)
            if watcher is not None:
                watcher(block)

    def unpin_blocks(
        self,
        blocks: Iterable[KVCacheBlock],
        on_reuse: Callable[[KVCacheBlock], None],
    ) -> None:
        """Release references to blocks that stay readable until reused.

        The blocks become last-resort eviction candidates regardless of prefix
        caching: they join the tail of the free queue and count as free.
        ``on_reuse`` fires when ``get_new_blocks`` hands a block out (or the
        cache is reset), so the caller can stop reading it.
        """
        released: list[KVCacheBlock] = []
        for block in blocks:
            assert block.ref_cnt > 0 and not block.is_null
            block.ref_cnt -= 1
            self._reuse_watchers[block.block_id] = on_reuse
            if block.ref_cnt == 0:
                if block.group_id in self._block_bytes:
                    page_block = self._release(block)
                    if page_block is not None:
                        released.append(page_block)
                else:
                    released.append(block)
        self._stamp(released)
        self.free_block_queue.append_n(released)

    def _maybe_evict_cached_block(self, block: KVCacheBlock) -> bool:
        """If a block is cached in `cached_block_hash_to_block`, we reset its hash
        metadata and evict it from the cache.

        Args:
            block: The block to evict.

        Returns:
            True if the block is evicted, False otherwise.

        """
        # Clean up metrics tracking first to prevent leaks
        if self.metrics_collector:
            self.metrics_collector.on_block_evicted(block)

        evicted_hashes = self._remove_cached_block_hashes(block)
        if not evicted_hashes:
            # The block doesn't have hash, eviction is not needed
            return False

        self._emit_block_removed_events(evicted_hashes)
        return True

    def touch(self, blocks: Sequence[KVCacheBlock]) -> None:
        """Touch a block increases its reference count by 1, and may remove
        the block from the free queue. This is used when a block is hit by
        another request with the same prefix.

        Args:
            blocks: A list of blocks to touch.

        """
        for block in blocks:
            if block.is_null:
                continue
            # ref_cnt=0 means this block is in the free list (i.e. eviction
            # candidate), so remove it.
            if block.ref_cnt == 0:
                if block.group_id in self._block_bytes:
                    self._acquire(block)
                else:
                    self.free_block_queue.remove(block)
            block.ref_cnt += 1
            if self.metrics_collector:
                self.metrics_collector.on_block_accessed(block)

    def is_block_writable(self, block: KVCacheBlock) -> bool:
        """Return whether a block can be mutated by its sole owner."""
        return not block.is_null and block.ref_cnt == 1 and block.block_hash is None

    def free_blocks(self, ordered_blocks: Iterable[KVCacheBlock]) -> None:
        """Free a list of blocks. The blocks should be ordered by their
        eviction priority, where the first block will be evicted first.

        Args:
            ordered_blocks: A list of blocks to free ordered by their eviction
                priority.

        """
        # Identify blocks with hash (LRU cache) and without it (never match APC)
        blocks_to_evict_last = []
        blocks_to_evict_first = []
        other_pools: dict[BlockPool, list[KVCacheBlock]] = {}
        for block in ordered_blocks:
            if block.is_null:  # never pooled; its ref_cnt is not maintained
                continue
            if block.pool is not None and block.pool is not self:
                other_pools.setdefault(block.pool, []).append(block)
                continue
            block.ref_cnt -= 1
            if block.ref_cnt == 0:
                if block.group_id in self._block_bytes:
                    # The page is freed with its last block in use, and
                    # counts as cached while any of its blocks is.
                    page_block = self._release(block)
                    if page_block is None:
                        continue
                    cached = self._pages[page_block.block_id].num_cached > 0
                    block = page_block
                else:
                    cached = block.block_hash is not None
                if not cached or not self.enable_caching:
                    # LIFO reuse of non-cached blocks for better GPU locality.
                    blocks_to_evict_first.append(block)
                else:
                    # FIFO reuse of cached blocks for LRU eviction behavior.
                    blocks_to_evict_last.append(block)

        # Blocks to reuse first are prepended to the front of the free queue.
        self.free_block_queue.prepend_n(blocks_to_evict_first)
        # Blocks to reuse last are appended to the end of the free queue.
        self._stamp(blocks_to_evict_last)
        self.free_block_queue.append_n(blocks_to_evict_last)
        for pool, blocks in other_pools.items():
            pool.free_blocks(blocks)

    def evict_blocks(self, block_ids: set[int], group_id: int = 0) -> None:
        """Evict blocks from the prefix cache by their block IDs.

        only evicts blocks that are currently cached (have a hash). blocks
        with ref_cnt > 0 are not freed from the block pool, only evicted
        from the prefix cache hash table.

        Args:
            block_ids: Set of block IDs to evict from cache.
            group_id: The KV cache group of the block IDs.

        """
        for block_id in block_ids:
            block = self.get_block(group_id, block_id)
            if (
                block.group_id in self._block_bytes
                and block.ref_cnt == 0
                and block.block_hash is not None
            ):
                page_id = self._page_id(group_id, block_id)
                page = self._pages[page_id]
                page.num_cached -= 1
                if page.num_in_use:
                    self._cached_slots[group_id].remove(block)
                    self._update_page(
                        page_id,
                        page,
                        page.num_in_use,
                        page.free_mask | 1 << (block_id - page.slots[0].block_id),
                    )
            self._maybe_evict_cached_block(block)

    def reset_prefix_cache(self) -> bool:
        """Reset prefix cache. This function may be used in RLHF
        flows to invalidate prefix caching after the weights are updated,
        or used for resetting prefix caching status for benchmarking.

        Returns:
            bool: True if the prefix cache is successfully reset,
            False otherwise.

        """
        num_used_blocks = self.num_gpu_blocks - self.free_block_queue.num_free_blocks
        if num_used_blocks != 1:  # The null block is always marked as used
            logger.warning(
                "Failed to reset prefix cache because some "
                "blocks (%d) are not freed yet",
                num_used_blocks - 1,
            )
            return False

        # Remove all hashes so that no new blocks will hit.
        self.cached_block_hash_to_block = BlockHashToBlockMap()
        self.cached_block_hashes_by_block.clear()
        if self._reuse_watchers:
            self._notify_reuse(
                [self.blocks[block_id] for block_id in list(self._reuse_watchers)]
            )

        # Remove all hashes from all blocks.
        for page in self._pages:
            page.num_cached = 0
            for block in page.slots:
                block.reset_hash()

        if self.metrics_collector:
            self.metrics_collector.reset()

        logger.info("Successfully reset prefix cache")

        if self.enable_kv_cache_events:
            self.kv_event_queue.append(AllBlocksCleared())

        return True

    def get_num_free_blocks(self, group_id: int = 0) -> int:
        """Get the number of blocks a group can allocate now.

        Args:
            group_id: The KV cache group.

        Returns:
            The number of free blocks.

        """
        n = self.slots_per_page.get(group_id, 1)
        num_free = n * self.free_block_queue.num_free_blocks
        if group_id in self._block_bytes:
            num_free += self._num_free_slots[group_id]
            num_free += self._cached_slots[group_id].num_free_blocks
        return num_free

    def can_allocate(
        self,
        num_blocks: Sequence[int],
        hit_blocks: Mapping[int, Sequence[KVCacheBlock]],
        num_reserved_pages: int,
    ) -> bool:
        """Whether every KV cache group can allocate its blocks after it
        touches its hit blocks, with ``num_reserved_pages`` pages left free.

        A group with one block per page takes one page per block. A group with
        more blocks per page takes pages only for what its free blocks on
        in-use pages do not cover, and evicts its cached free blocks on in-use
        pages only when no page is left. Allocations of different groups can
        interleave (a request's later Mamba blocks, the reservations of other
        requests), so the test passes only if every group covers its blocks
        after all other groups take the most pages they can.

        Args:
            num_blocks: Blocks to allocate per group, including the hit blocks
                that are free (ref_cnt 0) and so leave the free queues.
            hit_blocks: The cache-hit blocks each group with more blocks per
                page touches.
            num_reserved_pages: Pages to keep free.

        Returns:
            True if every group can allocate its blocks.

        """
        num_pages = self.free_block_queue.num_free_blocks - num_reserved_pages
        if not self.slots_per_page:
            # Every block, a free hit included, takes one page.
            return sum(num_blocks) <= num_pages
        if not any(num_blocks) and not any(hit_blocks.values()):
            # Nothing to allocate or touch (a decode step inside its blocks).
            return num_pages >= 0
        # Pages taken if every group takes the most it can, and the smallest
        # gap between the most and the least pages of a group that needs one.
        total = 0
        min_gap: int | None = None
        for group_id, num_group_blocks in enumerate(num_blocks):
            if group_id not in self._block_bytes:
                low = high = num_group_blocks
            else:
                num_free = self._num_free_slots[group_id]
                num_cached = self._cached_slots[group_id].num_free_blocks
                # Free hits pin their pages (or leave the group's cached
                # blocks) before any group allocates.
                hit_pages: dict[int, set[int]] = {}  # Page -> its free hit ids.
                for block in hit_blocks.get(group_id, ()):
                    if block.ref_cnt or block.is_null:
                        continue
                    num_group_blocks -= 1
                    page_id = self._page_id(group_id, block.block_id)
                    if self._pages[page_id].num_in_use:
                        num_cached -= 1
                    else:
                        hit_pages.setdefault(page_id, set()).add(block.block_id)
                # The other blocks of a pinned page join the group's free blocks.
                for page_id, hit_ids in hit_pages.items():
                    for slot in self._pages[page_id].slots:
                        if slot.block_id not in hit_ids:
                            if slot.block_hash is None:
                                num_free += 1
                            else:
                                num_cached += 1
                num_pages -= len(hit_pages)
                uncovered = max(0, num_group_blocks - num_free)
                n = self.slots_per_page[group_id]
                low = cdiv(max(0, uncovered - num_cached), n)
                high = cdiv(uncovered, n)
            total += high
            if low and (min_gap is None or high - low < min_gap):
                min_gap = high - low
        # A group that needs no page covers its blocks whatever others take.
        # Every other group must cover them with its least pages after all
        # other groups take their most.
        return num_pages >= 0 and (min_gap is None or total - min_gap <= num_pages)

    def get_usage(self) -> float:
        """Get the KV cache usage.

        Returns:
            The KV cache usage (between 0.0 and 1.0).

        """
        # Subtract 1 to account for null block.
        total_gpu_blocks = self.num_gpu_blocks - 1
        if not total_gpu_blocks:
            return 0
        return 1.0 - (self.free_block_queue.num_free_blocks / total_gpu_blocks)

    def take_events(self) -> list[KVCacheEvent]:
        """Atomically takes all events and clears the queue.

        Returns:
            A list of KV cache events.

        """
        if not self.enable_kv_cache_events:
            return []
        events = self.kv_event_queue
        self.kv_event_queue = []
        return events
