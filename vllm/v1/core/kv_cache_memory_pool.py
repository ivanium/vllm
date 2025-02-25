from abc import ABC, abstractmethod

from typing import List, Union

from vllm.logger import init_logger
from vllm.v1.core.kv_cache_utils import (FreeKVCacheBlockQueue, KVCacheBlock)

logger = init_logger(__name__)


class KVCacheBlockPoolBase(ABC):

    @abstractmethod
    def __init__(self, num_gpu_blocks: int):
        self.num_gpu_blocks = num_gpu_blocks

    @abstractmethod
    def allocate(self, num_blocks: int) -> List[KVCacheBlock]:
        """Allocate a specified number of KVCacheBlocks."""
        raise NotImplementedError

    @abstractmethod
    def free(self, blocks: Union[KVCacheBlock, List[KVCacheBlock]]) -> None:
        """Free the specified KVCacheBlocks."""
        raise NotImplementedError

    @property
    @abstractmethod
    def num_free_blocks(self) -> int:
        """Return the number of free KVCacheBlocks."""
        raise NotImplementedError

    @abstractmethod
    def reset_hash(self):
        """Reset the hash for all KVCacheBlocks."""
        raise NotImplementedError


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

    def reset_hash(self):
        for block in self.block_pool:
            block.reset_hash()
