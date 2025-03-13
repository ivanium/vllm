from abc import ABC, abstractmethod
from collections import defaultdict
import os

from typing import Dict, List, Union, Optional

from vllm.config import CacheConfig, ModelConfig
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_utils import (BlockHashType, FreeKVCacheBlockQueue,
                                         KVCacheBlock)

logger = init_logger(__name__)

MEM_PRESSURE_SIMU = True


class KVCacheMemPoolBase(ABC):

    @property
    @abstractmethod
    def num_free_blocks(self) -> int:
        """Return the number of free KVCacheBlocks."""
        raise NotImplementedError

    @abstractmethod
    def allocate(self, num_blocks: int) -> List[KVCacheBlock]:
        """Allocate new KVCacheBlocks."""
        raise NotImplementedError

    @abstractmethod
    def free(self, blocks: Union[KVCacheBlock, List[KVCacheBlock]]) -> None:
        """Free the specified KVCacheBlocks."""
        raise NotImplementedError

    # Prefix cache management methods.
    def reset_prefix_cache(self):
        """Reset prefix cache for all KVCacheBlocks."""
        return

    @abstractmethod
    def touch(self, blocks: List[KVCacheBlock]) -> None:
        """Touch a block increases its reference count by 1, and may remove
        the block from the free queue. This is used when a block is hit by
        another request with the same prefix.

        Args:
            blocks: A list of blocks to touch.
        """
        raise NotImplementedError

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


class KVCacheMemPool(KVCacheMemPoolBase):

    def __init__(self, num_gpu_blocks: int, enable_caching: bool = True):
        self.num_gpu_blocks = num_gpu_blocks
        self.enable_caching = enable_caching

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
        """Get new blocks from the free block pool.

        Note that we do not check block cache in this function.

        Args:
            num_blocks: The number of blocks to allocate.

        Returns:
            A list of new block.
        """
        if num_blocks > self.num_free_blocks:
            raise ValueError(
                f"Cannot get {num_blocks} free blocks from the pool")

        ret: List[KVCacheBlock] = []
        idx = 0
        while idx < num_blocks:
            # First allocate blocks.
            curr_block = self.free_block_queue.popleft()
            assert curr_block.ref_cnt == 0

            # If the block is cached, evict it.
            if self.enable_caching:
                self.maybe_evict_cached_block(curr_block)

            curr_block.incr_ref()
            ret.append(curr_block)
            idx += 1

        return ret

    def free(self, blocks: Union[KVCacheBlock, List[KVCacheBlock]]) -> None:
        """Free the specified blocks and add them to the free block queue.
        If a block is cached, we remove it from the prefix cache.
        Args:
            blocks: A list of blocks to free.
        """
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

    def touch(self, blocks: List[KVCacheBlock]) -> None:
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
                self.free_block_queue.remove(block)
            block.incr_ref()

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


class KVCacheElasticMemPool(KVCacheMemPoolBase):

    def __init__(self, num_gpu_blocks: int, enable_caching: bool,
                 block_size: int, token_bytes: int,
                 cache_config: Optional[CacheConfig],
                 model_config: Optional[ModelConfig]):
        self.block_size = block_size
        self.token_bytes = token_bytes
        self.block_bytes = self.block_size * self.token_bytes
        # assert block_bytes == (2 ** 21), "Only support 2MB block size for now."
        assert cache_config is not None and model_config is not None
        self.cache_config = cache_config
        self.model_config = model_config

        self.num_gpu_blocks = num_gpu_blocks
        self.enable_caching = enable_caching

        # A Block pool of all kv-cache blocks.
        self.block_pool: List[KVCacheBlock] = [
            KVCacheBlock(idx) for idx in range(num_gpu_blocks)
        ]

        # kvcached
        from kvcached.slab_allocator import PageAllocator, Page  # noqa: F401

        assert not self.enable_caching, "Caching is not supported in ElasticMemPool"
        PAGE_SIZE = 2 << 20  # 2MB
        assert PAGE_SIZE % self.block_bytes == 0
        assert self.num_gpu_blocks % (PAGE_SIZE // self.block_bytes) == 0
        GMEM_SIZE = (self.block_bytes * self.num_gpu_blocks + PAGE_SIZE -
                     1) // PAGE_SIZE * PAGE_SIZE
        self.page_allocator = PageAllocator(GMEM_SIZE, PAGE_SIZE)
        self.avail_pages: Dict[int, Page] = {}
        self.full_pages: Dict[int, Page] = {}

        # Adapt to available GPU memory.
        self.mem_file = "/tmp/kvcached_mem" if MEM_PRESSURE_SIMU else None

    def allocate(self, num_blocks: int) -> List[KVCacheBlock]:
        """Get new blocks from the free block pool.

        Note that we do not check block cache in this function.

        Args:
            num_blocks: The number of blocks to allocate.

        Returns:
            A list of new block.
        """
        if num_blocks > self.num_free_blocks:
            raise ValueError(
                f"Cannot get {num_blocks} free blocks from the pool")

        ret: List[KVCacheBlock] = []
        idx = 0
        while idx < num_blocks:
            page: Optional["Page"] = None
            if not self.avail_pages:
                page = self.page_allocator.alloc_page()
                page.init(self.block_bytes)
            else:
                _, page = self.avail_pages.popitem()
            assert page is not None
            block_id = page.alloc()
            if page.full():
                self.full_pages[page.page_id] = page
            else:
                self.avail_pages[page.page_id] = page
            # First allocate blocks.
            curr_block = self.block_pool[block_id]
            assert curr_block.ref_cnt == 0

            # # If the block is cached, evict it.
            # if self.enable_caching:
            #     self.maybe_evict_cached_block(curr_block)

            curr_block.incr_ref()
            ret.append(curr_block)
            idx += 1

        return ret

    def free(self, blocks: Union[KVCacheBlock, List[KVCacheBlock]]) -> None:
        """Free the specified blocks and add them to the free block queue.
        If a block is cached, we remove it from the prefix cache.
        Args:
            blocks: A list of blocks to free.
        """
        if isinstance(blocks, KVCacheBlock):
            blocks = [blocks]
        for block in blocks:
            page_id = self.page_allocator.get_page_id(block.block_id,
                                                      self.block_bytes)
            if page_id in self.full_pages:
                assert page_id not in self.avail_pages
                page = self.full_pages.pop(page_id)
                page.free(block.block_id)
                self.avail_pages[page_id] = page
            elif page_id in self.avail_pages:
                page = self.avail_pages[page_id]
                page.free(block.block_id)
                if page.empty():
                    del self.avail_pages[page_id]
                    self.page_allocator.free_page(page)
            else:
                raise ValueError("Block not found in any page")

    @property
    def num_free_blocks(self) -> int:
        num_free_blocks = sum(p.num_free_blocks()
                              for p in self.avail_pages.values())
        num_free_blocks += self.page_allocator.get_num_free_blocks(
            self.block_bytes)
        num_avail_blocks = self._get_avail_gpu_mem()  # available GPU memory

        return min(num_free_blocks, num_avail_blocks)

    def reset_prefix_cache(self):
        raise NotImplementedError

    def touch(self, blocks: List[KVCacheBlock]) -> None:
        raise NotImplementedError

    def maybe_evict_cached_block(self, block: KVCacheBlock) -> bool:
        raise NotImplementedError

    def get_cached_block(self,
                         block_hash: BlockHashType) -> Optional[KVCacheBlock]:
        raise NotImplementedError

    def add_to_prefix_cache(self, block: KVCacheBlock) -> None:
        raise NotImplementedError

    def _get_avail_gpu_mem(self):
        PAGE_SIZE = 2 << 20  # 2MB
        occupied_mem = self._read_mem_file()
        num_layers = int(getattr(self.model_config.hf_text_config,
                             "num_hidden_layers", 0))
        num_occupied_pages = (occupied_mem // num_layers + PAGE_SIZE - 1) // PAGE_SIZE
        num_occupied_blocks = num_occupied_pages * PAGE_SIZE // self.block_bytes
        if num_occupied_blocks > self.num_gpu_blocks:
            logger.warning(
                f"Occupied blocks {num_occupied_blocks} exceeds total GPU blocks {self.num_gpu_blocks}"
            )
            num_occupied_blocks = self.num_gpu_blocks
        num_avail_blocks = max(self.num_gpu_blocks - num_occupied_blocks, 0)
        return num_avail_blocks

    def _read_mem_file(self):
        if self.mem_file is None or not os.path.exists(self.mem_file):
            return 0

        try:
            with open(self.mem_file, 'r') as file:
                content = file.read().strip()

                try:
                    return int(content)
                except ValueError:
                    logger.error("Error: File does not contain a valid number.")
                    return 0
        except FileNotFoundError:
            logger.error(f"Error: File {self.mem_file} not found.")
            return 0
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            return 0
