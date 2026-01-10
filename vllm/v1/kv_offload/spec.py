# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import TYPE_CHECKING

import torch

from vllm.attention.backends.abstract import AttentionBackend
from vllm.logger import init_logger
from vllm.v1.kv_offload.abstract import LoadStoreSpec, OffloadingManager
from vllm.v1.kv_offload.worker.worker import OffloadingHandler

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)


class OffloadingSpec(ABC):
    """Spec for an offloading connector"""

    def __init__(self, vllm_config: "VllmConfig"):
        logger.warning(
            "Initializing OffloadingSpec. This API is experimental and "
            "subject to change in the future as we iterate the design."
        )
        self.vllm_config = vllm_config

        kv_transfer_config = vllm_config.kv_transfer_config
        assert kv_transfer_config is not None
        self.extra_config = kv_transfer_config.kv_connector_extra_config

        self.gpu_block_size = vllm_config.cache_config.block_size
        self.offloaded_block_size = int(
            self.extra_config.get("block_size", self.gpu_block_size)
        )

        assert self.offloaded_block_size % self.gpu_block_size == 0

    @abstractmethod
    def get_manager(self, num_groups: int = 1) -> OffloadingManager:
        """
        Get an OffloadingManager that will be used
        by the scheduler-side offloading connector to track
        offloaded blocks and manage evictions.

        Args:
            num_groups: Number of KV cache groups for HMA support.
                       Default 1 for single-group models.

        Returns:
            OffloadingManager instance (or HybridOffloadingManager for
            multiple groups).
        """
        pass

    @abstractmethod
    def get_handlers(
        self,
        kv_caches: dict[str, torch.Tensor],
        attn_backends: dict[str, type[AttentionBackend]],
        kv_cache_config: "KVCacheConfig | None" = None,
    ) -> Iterator[
        tuple[type[LoadStoreSpec], type[LoadStoreSpec], OffloadingHandler]
        | tuple[type[LoadStoreSpec], type[LoadStoreSpec], OffloadingHandler, int]
    ]:
        """
        Get offloading handlers along with their respective src and dst types.

        Args:
            kv_caches: A dictionary of layer_name -> gpu_kv_cache tensor.
            attn_backends: A dictionary of layer_name -> AttentionBackend.
            kv_cache_config: Optional KV cache config for HMA support.
                            When provided with multiple groups, yields 4-tuples.

        Yields:
            3-tuples (src_type, dst_type, handler) for single-group mode.
            4-tuples (src_type, dst_type, handler, group_id) for HMA mode.
        """
        pass
