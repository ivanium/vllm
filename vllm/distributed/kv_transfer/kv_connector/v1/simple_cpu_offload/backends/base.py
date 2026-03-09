# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Abstract base class for KV cache transfer backends."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheConfig


class TransferBackend(ABC):
    """Abstract base class for KV cache transfer backends.

    A TransferBackend encapsulates the storage allocation and block copy
    mechanics for a particular destination medium (pinned CPU memory, disk,
    etc.). The worker uses a backend instance to move KV cache blocks
    between the GPU and the destination without knowing the underlying
    transfer mechanism.
    """

    @abstractmethod
    def setup(
        self,
        src_caches: dict[str, torch.Tensor],
        capacity_bytes: int,
        kv_cache_config: "KVCacheConfig | None",
    ) -> int:
        """Initialize backend storage.

        Args:
            src_caches: Unique GPU KV cache tensors (already deduplicated).
            capacity_bytes: Byte budget for destination storage.
            kv_cache_config: Optional KV cache configuration (used to
                determine the number of GPU blocks for reshaping raw
                tensors).

        Returns:
            Number of destination blocks allocated.
        """
        ...

    @abstractmethod
    def copy_blocks(
        self,
        src_block_ids: list[int],
        dst_block_ids: list[int],
        is_store: bool,
    ) -> None:
        """Copy blocks between source and destination.

        Args:
            src_block_ids: Source block indices.
            dst_block_ids: Destination block indices.
            is_store: If ``True`` the transfer direction is GPU -> dest;
                otherwise dest -> GPU.
        """
        ...

    @abstractmethod
    def record_event(self) -> Any:
        """Record a completion event after :meth:`copy_blocks`.

        Returns:
            An opaque event handle understood by :meth:`query_event` and
            :meth:`sync_event`.
        """
        ...

    @abstractmethod
    def query_event(self, event: Any) -> bool:
        """Non-blocking check whether *event* has completed.

        Returns:
            ``True`` if the event has completed, ``False`` otherwise.
        """
        ...

    @abstractmethod
    def sync_event(self, event: Any) -> None:
        """Block until *event* completes."""
        ...

    @abstractmethod
    def sync_all(self) -> None:
        """Block until **all** pending operations complete."""
        ...

    @property
    @abstractmethod
    def is_initialized(self) -> bool:
        """Whether the backend has been set up and is ready for transfers."""
        ...

    @abstractmethod
    def validate_block_ids(self, block_ids: list[int], is_src: bool) -> None:
        """Validate that *block_ids* are within bounds.

        Args:
            block_ids: Block indices to validate.
            is_src: If ``True``, validate against the source (GPU) block
                count; otherwise validate against the destination block
                count.

        Raises:
            ValueError: If any block index is out of range.
        """
        ...
