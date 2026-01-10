# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Hybrid Offloading Manager for HMA (Hybrid Memory Allocator) support.

This module provides a per-group OffloadingManager wrapper that enables
CPU offloading for hybrid models with multiple KV cache groups
(e.g., Full Attention + Mamba layers).
"""

from collections.abc import Callable, Iterable

from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_offload.abstract import (
    LoadStoreSpec,
    OffloadingEvent,
    OffloadingManager,
    PrepareStoreOutput,
)
from vllm.v1.kv_offload.backend import Backend


class HybridOffloadingManager(OffloadingManager):
    """
    Per-group OffloadingManager for HMA (Hybrid Memory Allocator) support.

    Each KV cache group has its own independent manager with its own:
    - Backend (e.g., CPU block pool)
    - Block tracking (LRU eviction per group)
    - Block ID namespace

    This design simplifies HMA support by treating each group as fully isolated.
    The scheduler-side connector is responsible for calling the correct group's
    methods based on the transfer context.

    Note: This design allocates CPU memory separately per group. A future
    optimization could share CPU blocks across groups for blocks with the same
    content hash, but this would require more complex deduplication logic.

    Args:
        num_groups: Number of KV cache groups.
        backend_factory: Callable that creates a Backend for each group.
            Takes group_id as argument and returns a Backend instance.
        manager_factory: Callable that creates an OffloadingManager for each group.
            Takes (backend, enable_events) as arguments and returns a manager.
        enable_events: Whether to track offloading events.
    """

    def __init__(
        self,
        num_groups: int,
        backend_factory: Callable[[int], Backend],
        manager_factory: Callable[[Backend, bool], OffloadingManager],
        enable_events: bool = False,
    ):
        self.num_groups = num_groups
        self.enable_events = enable_events
        self.managers: dict[int, OffloadingManager] = {}

        for group_id in range(num_groups):
            backend = backend_factory(group_id)
            self.managers[group_id] = manager_factory(backend, enable_events)

    def lookup(self, block_hashes: Iterable[BlockHash], group_id: int = 0) -> int:
        """
        Lookup blocks in the specified group's manager.

        Args:
            block_hashes: Block hashes to look up.
            group_id: KV cache group ID. Default is 0 for backward compatibility.

        Returns:
            Number of consecutive blocks found starting from the first.
        """
        if group_id not in self.managers:
            return 0
        return self.managers[group_id].lookup(block_hashes)

    def prepare_load(
        self, block_hashes: Iterable[BlockHash], group_id: int = 0
    ) -> LoadStoreSpec:
        """
        Prepare to load blocks from the specified group.

        Args:
            block_hashes: Block hashes to load.
            group_id: KV cache group ID.

        Returns:
            LoadStoreSpec with group_id set for routing.
        """
        manager = self.managers[group_id]
        spec = manager.prepare_load(block_hashes)
        # Ensure the spec has the correct group_id for worker-side routing
        if hasattr(spec, "group_id"):
            spec.group_id = group_id
        return spec

    def touch(self, block_hashes: Iterable[BlockHash], group_id: int = 0):
        """
        Mark blocks as recently used in the specified group.

        Args:
            block_hashes: Block hashes to touch.
            group_id: KV cache group ID.
        """
        if group_id in self.managers:
            self.managers[group_id].touch(block_hashes)

    def complete_load(self, block_hashes: Iterable[BlockHash], group_id: int = 0):
        """
        Mark blocks as done loading in the specified group.

        Args:
            block_hashes: Block hashes that finished loading.
            group_id: KV cache group ID.
        """
        if group_id in self.managers:
            self.managers[group_id].complete_load(block_hashes)

    def prepare_store(
        self, block_hashes: Iterable[BlockHash], group_id: int = 0
    ) -> PrepareStoreOutput | None:
        """
        Prepare to store blocks in the specified group.

        Args:
            block_hashes: Block hashes to store.
            group_id: KV cache group ID.

        Returns:
            PrepareStoreOutput with store_spec having group_id set,
            or None if blocks cannot be stored.
        """
        manager = self.managers[group_id]
        output = manager.prepare_store(block_hashes)
        if output is not None and hasattr(output.store_spec, "group_id"):
            output.store_spec.group_id = group_id
        return output

    def complete_store(
        self,
        block_hashes: Iterable[BlockHash],
        success: bool = True,
        group_id: int = 0,
    ):
        """
        Mark blocks as done storing in the specified group.

        Args:
            block_hashes: Block hashes that finished storing.
            success: Whether the store succeeded.
            group_id: KV cache group ID.
        """
        if group_id in self.managers:
            self.managers[group_id].complete_store(block_hashes, success)

    def take_events(self) -> Iterable[OffloadingEvent]:
        """
        Take events from all group managers.

        Yields:
            OffloadingEvent from all groups.
        """
        for manager in self.managers.values():
            yield from manager.take_events()

    # --- Convenience methods for single-group backward compatibility ---

    def get_manager(self, group_id: int = 0) -> OffloadingManager:
        """
        Get the underlying manager for a specific group.

        Useful for direct access when needed.

        Args:
            group_id: KV cache group ID.

        Returns:
            The OffloadingManager for the specified group.
        """
        return self.managers[group_id]
