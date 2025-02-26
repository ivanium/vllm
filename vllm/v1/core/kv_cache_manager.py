from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

from vllm.logger import init_logger
from vllm.utils import cdiv
from vllm.v1.core.kv_cache_memory_pool import KVCacheMemPool
from vllm.v1.core.kv_cache_utils import (KVCacheBlock,
                                         generate_block_hash_extra_keys,
                                         hash_block_tokens,
                                         hash_request_tokens)
from vllm.v1.request import Request, RequestStatus

logger = init_logger(__name__)


class KVCacheManager:

    def __init__(
        self,
        block_size: int,
        num_gpu_blocks: int,
        max_model_len: int,
        sliding_window: Optional[int] = None,
        enable_caching: bool = True,
        num_preallocate_tokens: int = 64,
    ) -> None:
        self.block_size = block_size
        self.num_gpu_blocks = num_gpu_blocks
        self.max_model_len = max_model_len
        self.max_num_blocks_per_req = cdiv(max_model_len, block_size)
        self.sliding_window = sliding_window
        self.enable_caching = enable_caching
        # NOTE(woosuk): To avoid frequent block allocation, we preallocate some
        # blocks for each request. For example, when a request reaches the end
        # of its block table, we preallocate N blocks in advance. This way, we
        # reduce the overhead of updating free_block_ids and ref_cnts for each
        # request every step (at the cost of some memory waste).
        # NOTE(woosuk): This is different from the "lookahead" slots since this
        # does not guarantee that the request always has N empty blocks. After
        # the request gets N empty blocks, it starts to use the blocks without
        # further allocation. When it uses up all the N empty blocks, it gets
        # N new empty blocks.
        self.num_preallocate_tokens = num_preallocate_tokens
        self.num_preallocate_blocks = cdiv(num_preallocate_tokens, block_size)

        self.kv_block_pool = KVCacheMemPool(num_gpu_blocks)

        # Mapping from request ID to blocks to track the blocks allocated
        # for each request, so that we can free the blocks when the request
        # is finished.
        self.req_to_blocks: Dict[str, List[KVCacheBlock]] = {}

    def get_computed_blocks(
            self, request: Request) -> Tuple[List[KVCacheBlock], int]:
        """Get the computed (cached) blocks for the request.
        Note that the computed blocks must be full.

        Args:
            request: The request to get the computed blocks.

        Returns:
            A tuple containing:
                - A list of blocks that are computed for the request.
                - The number of computed tokens.
        """
        if not self.enable_caching:
            # Prefix caching is disabled.
            return [], 0

        computed_blocks = []

        # The block hashes for the request may already be computed
        # if the request was preempted and resumed.
        if not request.kv_block_hashes:
            request.set_kv_block_hashes(
                hash_request_tokens(self.block_size, request))
        block_hashes = request.kv_block_hashes

        for block_hash in block_hashes:
            # block_hashes is a chain of block hashes. If a block hash is not
            # in the cached_block_hash_to_id, the following block hashes are
            # not computed yet for sure.
            if cached_block := self.kv_block_pool.get_cached_block(block_hash):
                computed_blocks.append(cached_block)
            else:
                break

        # NOTE(woosuk): Since incomplete blocks are not eligible for
        # sharing, `num_computed_tokens` is always a multiple of
        # `block_size`.
        num_computed_tokens = len(computed_blocks) * self.block_size
        return computed_blocks, num_computed_tokens

    def append_slots(
        self,
        request: Request,
        num_tokens: int,
    ) -> Optional[List[KVCacheBlock]]:
        """Append slots to the block table of the request.
        We first append slots to already allocated blocks. If the allocated
        blocks are not enough, we allocate new blocks.

        Args:
            request: The request to append slots.
            num_tokens: The number of tokens to append.

        Returns:
            A list of new blocks if new blocks are allocated, or None
            if new blocks are required but cannot be allocated.
        """
        num_required_blocks = cdiv(request.num_computed_tokens + num_tokens,
                                   self.block_size)
        req_blocks = self.req_to_blocks[request.request_id]

        num_new_blocks = num_required_blocks - len(req_blocks)
        if num_new_blocks > self.kv_block_pool.num_free_blocks:
            # Need to allocate new blocks due to insufficient pre-allocated
            # slots, but we cannot allocate new blocks due to the limit.
            return None

        if num_new_blocks <= 0:
            # No new block is needed.
            new_blocks = []
        else:
            # Get new blocks from the free block pool considering
            # preallocated blocks.
            num_new_blocks = min(
                num_new_blocks + self.num_preallocate_blocks,
                self.kv_block_pool.num_free_blocks,
                # Should not exceed the maximum number of blocks per request.
                # This is especially because the block table has the shape
                # [..., max_num_blocks_per_req].
                # TODO(woosuk): Check and reject requests if
                # num_prompt_tokens + max_tokens > max_model_len.
                self.max_num_blocks_per_req - len(req_blocks),
            )
            assert num_new_blocks > 0

            new_blocks = self._get_new_blocks(num_new_blocks)
            req_blocks.extend(new_blocks)

        if not self.enable_caching:
            return new_blocks

        num_computed_full_blocks = (request.num_computed_tokens //
                                    self.block_size)

        # NOTE(rickyx): We are assuming the `num_tokens` are actual
        # tokens rather than lookahead slots (e.g. for speculative decoding).
        # TODO(rickyx): When supporting speculative decoding, we will need to
        # differentiate between them so that we can know how many blocks are
        # full after appending the actual tokens.
        num_full_blocks_after_append = (request.num_computed_tokens +
                                        num_tokens) // self.block_size
        assert num_full_blocks_after_append <= len(req_blocks)

        new_full_blocks = req_blocks[
            num_computed_full_blocks:num_full_blocks_after_append]
        if new_full_blocks:
            self._cache_full_blocks(
                request=request,
                blk_start_idx=num_computed_full_blocks,
                full_blocks=new_full_blocks,
                prev_block=req_blocks[num_computed_full_blocks - 1]
                if num_computed_full_blocks >= 1 else None,
            )

        return new_blocks

    def allocate_slots(
        self,
        request: Request,
        num_tokens: int,
        computed_blocks: List[KVCacheBlock],
    ) -> Optional[List[KVCacheBlock]]:
        """Allocate slots for a new request.

        Args:
            request: The request to allocate slots.
            num_tokens: The number of tokens to allocate. Note that this does
                not include the tokens that have already been computed.
            computed_blocks: A list of computed blocks.

        Returns:
            A list of new allocated blocks.
        """
        if num_tokens == 0:
            raise ValueError(
                f"num_tokens must be greater than 0, got {num_tokens}")

        # If a computed block of a request is an eviction candidate (in the
        # free queue and ref_cnt == 0), it cannot be counted as a free block
        # when allocating this request.
        num_evictable_computed_blocks = sum(1 for blk in computed_blocks
                                            if blk.ref_cnt == 0)

        num_required_blocks = cdiv(num_tokens, self.block_size)
        if (num_required_blocks > self.kv_block_pool.num_free_blocks -
                num_evictable_computed_blocks):
            # Cannot allocate new blocks.
            return None

        # Touch the computed blocks to make sure they won't be evicted.
        if self.enable_caching:
            self._touch(computed_blocks)
        else:
            assert not computed_blocks, (
                "Computed blocks should be empty when "
                "prefix caching is disabled")

        # Determine the number of new blocks to allocate considering
        # preallocated blocks.
        num_new_blocks = min(
            num_required_blocks + self.num_preallocate_blocks,
            self.kv_block_pool.num_free_blocks,
            # Should not exceed the maximum number of blocks per request.
            # This is especially because the block table has the shape
            # [..., max_num_blocks_per_req].
            # TODO(woosuk): Check and reject requests if
            # num_prompt_tokens + max_tokens > max_model_len.
            self.max_num_blocks_per_req - len(computed_blocks),
        )
        assert num_new_blocks > 0

        # Concatenate the computed block IDs and the new block IDs.
        new_blocks = self._get_new_blocks(num_new_blocks)
        self.req_to_blocks[request.request_id] = computed_blocks + new_blocks

        if not self.enable_caching:
            return new_blocks

        num_computed_tokens = len(computed_blocks) * self.block_size
        num_full_blocks = (num_computed_tokens + num_tokens) // self.block_size

        new_full_blocks = self.req_to_blocks[
            request.request_id][len(computed_blocks):num_full_blocks]
        if new_full_blocks:
            self._cache_full_blocks(
                request=request,
                blk_start_idx=len(computed_blocks),
                # The new full blocks are the full blocks that are not computed.
                full_blocks=new_full_blocks,
                prev_block=computed_blocks[-1] if computed_blocks else None,
            )

        return new_blocks

    def free(self, request: Request) -> None:
        """Free the blocks allocated for the request.
        When caching is enabled, we free the blocks in reverse order so that
        the tail blocks are evicted first.

        Args:
            request: The request to free the blocks.
        """
        # Default to [] in case a request is freed (aborted) before alloc.
        blocks = self.req_to_blocks.pop(request.request_id, [])
        ordered_blocks: Iterable[KVCacheBlock] = blocks
        if self.enable_caching:
            # Free blocks in reverse order so that the tail blocks are
            # freed first.
            ordered_blocks = reversed(blocks)

        for block in ordered_blocks:
            block.decr_ref()
            if block.ref_cnt == 0:
                self.kv_block_pool.free(block)

    def uncache_blocks(self, request: Request) -> int:
        """Uncache the blocks that are no longer full based on the
        num_computed_tokens in the given request. This happens when
        the blocks were full and cached due to speculative tokens, but the
        speculative tokens are not accepted.

        Args:
            request: The request.

        Returns:
            The number of uncached blocks.
        """
        blocks = self.req_to_blocks[request.request_id]
        num_computed_tokens = request.num_computed_tokens
        num_full_blocks = num_computed_tokens // self.block_size
        num_uncached_blocks = 0
        for block in blocks[num_full_blocks:]:
            # If the block is not cached, the following blocks are not cached.
            if not self.kv_block_pool.maybe_evict_cached_block(block):
                break
            num_uncached_blocks += 1
        return num_uncached_blocks

    def reset_prefix_cache(self) -> bool:
        """Reset prefix cache. This function may be used in RLHF
        flows to invalid prefix caching after the weights are updated,
        or used for resetting prefix caching status for benchmarking.

        Returns:
            bool: True if the prefix cache is successfully reset,
            False otherwise.
        """
        num_used_blocks = (self.num_gpu_blocks -
                           self.kv_block_pool.num_free_blocks)
        if num_used_blocks > 0:
            logger.warning(
                "Failed to reset prefix cache because some "
                "blocks (%d) are not freed yet", num_used_blocks)
            return False

        # Remove all hashes so that no new blocks will hit and remove all hashes
        # from all blocks.
        self.kv_block_pool.reset_prefix_cache()

        logger.info("Successfully reset prefix cache")
        return True

    def get_num_common_prefix_blocks(
        self,
        request: Request,
        num_running_requests: int,
    ) -> int:
        """Calculate the number of common prefix blocks shared by all requests
        in the RUNNING state.

        The function determines this by selecting any request and iterating
        through its blocks.  A block is considered a common prefix block if its
        `ref_cnt` equals the total number of requests in the RUNNING state.

        NOTE(woosuk): The number of requests in the RUNNING state is **greater
        than or equal to** the number of requests scheduled in the current step.
        This is because the RUNNING state only indicates that:
        1. The request has not yet finished, and
        2. The request holds its blocks unfreed.

        While all scheduled requests must be in the RUNNING state, the inverse
        is not necessarily true. There may be RUNNING requests that are not
        scheduled in the current step. As of 1/1/2025, the scheduler does not
        allow this case, but it is possible in the future, as we allow more
        flexible scheduling.

        This can result in an edge case where the number of common prefix blocks
        is 0, even though all scheduled requests share a common prefix. This
        occurs because there may be unscheduled RUNNING requests that do not
        share the common prefix. Currently, this case cannot be easily detected,
        so the function returns 0 in such cases.

        Args:
            request: Any request in the RUNNING state, used to identify the
                common prefix blocks.
            num_running_requests: The total number of requests in the RUNNING
                state. This can be different from the number of scheduled
                requests in the current step.

        Returns:
            int: The number of common prefix blocks.
        """
        assert request.status == RequestStatus.RUNNING
        blocks = self.req_to_blocks[request.request_id]
        num_common_blocks = 0
        for block in blocks:
            if block.ref_cnt == num_running_requests:
                num_common_blocks += 1
            else:
                break
        return num_common_blocks

    def _get_new_blocks(self, num_blocks: int) -> List[KVCacheBlock]:
        """Get new blocks from the free block pool.

        Note that we do not check block cache in this function.

        Args:
            num_blocks: The number of blocks to allocate.

        Returns:
            A list of new block.
        """
        if num_blocks > self.kv_block_pool.num_free_blocks:
            raise ValueError(
                f"Cannot get {num_blocks} free blocks from the pool")

        ret: List[KVCacheBlock] = []
        idx = 0
        while idx < num_blocks:
            # First allocate blocks.
            curr_block = self.kv_block_pool.allocate(1)[0]
            assert curr_block.ref_cnt == 0

            # If the block is cached, evict it.
            if self.enable_caching:
                self.kv_block_pool.maybe_evict_cached_block(curr_block)

            curr_block.incr_ref()
            ret.append(curr_block)
            idx += 1

        return ret

    def _touch(self, blocks: List[KVCacheBlock]) -> None:
        """Touch a block increases its reference count by 1, and may remove
        the block from the free queue. This is used when a block is hit by
        another request with the same prefix.

        Args:
            blocks: A list of blocks to touch.
        """
        for block in blocks:
            # ref_cnt=0 means this block is in the free list (i.e. eviction
            # candidate), so remove it.
            if block.ref_cnt == 0:
                self.kv_block_pool.remove(block)
            block.incr_ref()

    def _cache_full_blocks(
        self,
        request: Request,
        blk_start_idx: int,
        full_blocks: List[KVCacheBlock],
        prev_block: Optional[KVCacheBlock],
    ) -> None:
        """Cache a list of full blocks for prefix caching.

        This function takes a list of blocks that will have their block hash
        metadata to be updated and cached. Given a request, it computes the
        block hashes for the blocks starting from `blk_start_idx` to the end
        of the request's full blocks, updating the metadata for each block
        and caching them in the `kv_block_pool.cached_block_hash_to_block`.

        Args:
            request: The request to cache the blocks.
            blk_start_idx: The index of the first block in the request's blocks
                to cache.
            full_blocks: The list of blocks to update hash metadata.
            prev_block: The previous block in the chain.
        """
        num_cached_block_hashes = len(request.kv_block_hashes)

        # Update the new blocks with the block hashes through the chain.
        prev_block_hash_value = None
        if prev_block is not None:
            # Previous block must have a block hash because it must be
            # a full, cached block.
            assert prev_block.block_hash is not None
            prev_block_hash_value = prev_block.block_hash.hash_value

        for i, blk in enumerate(full_blocks):
            blk_idx = blk_start_idx + i

            if blk_idx < num_cached_block_hashes:
                # The block hash may already be computed in
                # "get_computed_blocks" if the tokens are not generated by
                # this request (either the prompt tokens or the previously
                # generated tokens with preemption). In this case we simply
                # reuse the block hash.
                block_hash = request.kv_block_hashes[blk_idx]
            else:
                # Otherwise compute the block hash and cache it in the request
                # in case it will be preempted in the future.
                start_token_idx = blk_idx * self.block_size
                end_token_idx = (blk_idx + 1) * self.block_size
                block_tokens = request.all_token_ids[
                    start_token_idx:end_token_idx]
                assert len(block_tokens) == self.block_size, (
                    f"Expected {self.block_size} tokens, got "
                    f"{len(block_tokens)} at {blk_idx}th block for request "
                    f"{request.request_id}({request})")

                # Generate extra keys for multi-modal inputs. Note that since
                # we reach to this branch only when the block is completed with
                # generated tokens, we only need to consider the last mm input.
                extra_keys, _ = generate_block_hash_extra_keys(
                    request, start_token_idx, end_token_idx, -1)

                # Compute the hash of the current block.
                block_hash = hash_block_tokens(prev_block_hash_value,
                                               block_tokens, extra_keys)
                request.append_kv_block_hashes(block_hash)

            # Update and added the full block to the cache.
            blk.block_hash = block_hash
            self.kv_block_pool.add_to_prefix_cache(blk)
            prev_block_hash_value = block_hash.hash_value
