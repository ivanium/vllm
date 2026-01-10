# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC

import numpy as np

from vllm.v1.kv_offload.abstract import LoadStoreSpec


class BlockIDsLoadStoreSpec(LoadStoreSpec, ABC):
    """
    Spec for loading/storing KV blocks from given block numbers.

    Attributes:
        block_ids: Array of block IDs to load/store.
        group_id: KV cache group ID for HMA (Hybrid Memory Allocator) support.
                  Different groups may have different block ID namespaces.
                  Default is 0 for backward compatibility with single-group models.
    """

    def __init__(self, block_ids: list[int], group_id: int = 0):
        self.block_ids = np.array(block_ids, dtype=np.int64)
        self.group_id = group_id

    def __repr__(self) -> str:
        if self.group_id == 0:
            return repr(self.block_ids)
        return f"group={self.group_id}, blocks={self.block_ids}"


class GPULoadStoreSpec(BlockIDsLoadStoreSpec):
    """
    Spec for loading/storing a KV block to GPU memory.
    """

    @staticmethod
    def medium() -> str:
        return "GPU"


class CPULoadStoreSpec(BlockIDsLoadStoreSpec):
    """
    Spec for loading/storing a KV block to CPU memory.
    """

    @staticmethod
    def medium() -> str:
        return "CPU"
