from abc import ABC, abstractmethod
from collections import defaultdict

from typing import Dict, List, Union, Optional

from vllm.logger import init_logger
from vllm.v1.core.kv_cache_utils import (BlockHashType, FreeKVCacheBlockQueue,
                                         KVCacheBlock)

logger = init_logger(__name__)


class KVCacheBlockPoolBase(ABC):

    @abstractmethod
    def __init__(self, num_gpu_blocks: int):
        self.num_gpu_blocks = num_gpu_blocks

    @property
    @abstractmethod
    def num_free_blocks(self) -> int:
        """Return the number of free KVCacheBlocks."""
        raise NotImplementedError

    @abstractmethod
    def allocate(self, num_blocks: int) -> List[KVCacheBlock]:
        """Allocate a specified number of KVCacheBlocks."""
        raise NotImplementedError

    @abstractmethod
    def free(self, blocks: Union[KVCacheBlock, List[KVCacheBlock]]) -> None:
        """Free the specified KVCacheBlocks."""
        raise NotImplementedError

    # Prefix cache management methods.
    def reset_prefix_cache(self):
        """Reset prefix cache for all KVCacheBlocks."""
        return

    def maybe_evict_cached_block(self, block: KVCacheBlock) -> bool:
        """Maybe evict a cached block from the prefix cache.

        Args:
            block: The block to evict.

        Returns:
            True if the block is evicted, False otherwise.
        """
        return False

    def get_cached_block(self,
                         block_hash: BlockHashType) -> Optional[KVCacheBlock]:
        """Get a cached block by the block hash, or None if cache miss.

        Args:
            block_hash: The hash value of the block.

        Returns:
            The cached block if it exists, or None.
        """
        return None

    def add_to_prefix_cache(self, block: KVCacheBlock) -> None:
        """Add a block to the prefix cache.

        Args:
            block: The block to add to the prefix cache.
        """
        return None


class KVCacheBlockPool(KVCacheBlockPoolBase):

    def __init__(self, num_gpu_blocks: int):
        super().__init__(num_gpu_blocks)

        # A Block pool of all kv-cache blocks.
        self.block_pool: List[KVCacheBlock] = [
            KVCacheBlock(idx) for idx in range(num_gpu_blocks)
        ]
        # Free block queue that constructs and manipulates a doubly linked
        # list of free blocks (including eviction candidates when caching is
        # enabled).
        self.free_block_queue = FreeKVCacheBlockQueue(self.block_pool)
        # {block_hash: {block ID: block}}. A cached block is
        # a full block with a block hash that can be used for prefix caching.
        # The cached block may be used by running requests or in the
        # free_block_queue that could potentially be evicted.
        # NOTE: We currently don't de-duplicate the blocks in the cache,
        # meaning that if a block becomes full and is cached, we don't check
        # if there is already an identical block in the cache. This is because
        # we want to make sure the allocated block IDs won't change so that
        # block tables are append-only.
        self.cached_block_hash_to_block: Dict[BlockHashType, Dict[
            int, KVCacheBlock]] = defaultdict(dict)

    def allocate(self, num_blocks: int) -> List[KVCacheBlock]:
        return [self.free_block_queue.popleft() for _ in range(num_blocks)]

    def free(self, blocks: Union[KVCacheBlock, List[KVCacheBlock]]) -> None:
        if isinstance(blocks, KVCacheBlock):
            blocks = [blocks]
        for block in blocks:
            self.free_block_queue.append(block)

    @property
    def num_free_blocks(self) -> int:
        return self.free_block_queue.num_free_blocks

    def reset_prefix_cache(self):
        # Remove all hashes so that no new blocks will hit.
        self.cached_block_hash_to_block = defaultdict(dict)
        # Remove all hashes from all blocks.
        for block in self.block_pool:
            block.reset_hash()

    def maybe_evict_cached_block(self, block: KVCacheBlock) -> bool:
        """
        If a block is cached in `cached_block_hash_to_block`, we reset its hash
        metadata and evict it from the cache.

        Args:
            block: The block to evict.

        Returns:
            True if the block is evicted, False otherwise.
        """
        block_hash = block.block_hash
        if block_hash and block_hash in self.cached_block_hash_to_block:
            block.reset_hash()
            del self.cached_block_hash_to_block[block_hash][block.block_id]

            if len(self.cached_block_hash_to_block[block_hash]) == 0:
                del self.cached_block_hash_to_block[block_hash]

            return True
        return False

    def get_cached_block(self,
                         block_hash: BlockHashType) -> Optional[KVCacheBlock]:
        """Get a cached block by the block hash, or None if cache miss.
        If there are duplicated blocks, we return the first block in the cache.

        Args:
            block_hash: The hash value of the block.

        Returns:
            The cached block if it exists, or None.
        """
        if block_hash in self.cached_block_hash_to_block:
            first_block_id = list(
                self.cached_block_hash_to_block[block_hash].keys())[0]
            return self.cached_block_hash_to_block[block_hash][first_block_id]
        return None

    def add_to_prefix_cache(self, block: KVCacheBlock) -> None:
        """Add a block to the prefix cache.

        Args:
            block: The block to add to the prefix cache.
        """
        block_hash = block.block_hash
        if not block_hash:
            return

        self.cached_block_hash_to_block[block_hash][block.block_id] = block
