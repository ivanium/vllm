# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for HybridOffloadingManager.

Tests per-group isolation for HMA (Hybrid Memory Allocator) support.
"""

import numpy as np

from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_offload.backends.cpu import CPUBackend
from vllm.v1.kv_offload.hybrid_manager import HybridOffloadingManager
from vllm.v1.kv_offload.lru_manager import LRUOffloadingManager
from vllm.v1.kv_offload.mediums import CPULoadStoreSpec


def to_hashes(int_hashes: list[int]) -> list[BlockHash]:
    """Convert integer hashes to BlockHash objects."""
    return [BlockHash(str(i).encode()) for i in int_hashes]


def test_hybrid_manager_basic():
    """
    Tests basic HybridOffloadingManager functionality with 2 groups.
    Each group should have independent block tracking.
    """
    num_groups = 2
    block_size = 256
    num_blocks = 4

    def backend_factory(group_id: int) -> CPUBackend:
        return CPUBackend(block_size=block_size, num_blocks=num_blocks)

    def manager_factory(backend, enable_events):
        return LRUOffloadingManager(backend, enable_events)

    manager = HybridOffloadingManager(
        num_groups=num_groups,
        backend_factory=backend_factory,
        manager_factory=manager_factory,
        enable_events=True,
    )

    # Both groups should be empty initially
    assert manager.lookup(to_hashes([1, 2]), group_id=0) == 0
    assert manager.lookup(to_hashes([1, 2]), group_id=1) == 0


def test_hybrid_manager_group_isolation():
    """
    Tests that different groups are fully isolated.
    The same block hash should have independent state in each group.
    """
    num_groups = 2
    block_size = 256
    num_blocks = 4

    def backend_factory(group_id: int) -> CPUBackend:
        return CPUBackend(block_size=block_size, num_blocks=num_blocks)

    def manager_factory(backend, enable_events):
        return LRUOffloadingManager(backend, enable_events)

    manager = HybridOffloadingManager(
        num_groups=num_groups,
        backend_factory=backend_factory,
        manager_factory=manager_factory,
        enable_events=False,
    )

    # Store blocks [1, 2] in group 0 only
    output0 = manager.prepare_store(to_hashes([1, 2]), group_id=0)
    assert output0 is not None
    assert output0.block_hashes_to_store == to_hashes([1, 2])
    manager.complete_store(to_hashes([1, 2]), success=True, group_id=0)

    # Group 0 should have blocks [1, 2]
    assert manager.lookup(to_hashes([1, 2]), group_id=0) == 2
    # Group 1 should NOT have blocks [1, 2]
    assert manager.lookup(to_hashes([1, 2]), group_id=1) == 0

    # Store blocks [1, 2] in group 1 as well
    output1 = manager.prepare_store(to_hashes([1, 2]), group_id=1)
    assert output1 is not None
    assert output1.block_hashes_to_store == to_hashes([1, 2])
    manager.complete_store(to_hashes([1, 2]), success=True, group_id=1)

    # Now both groups should have blocks [1, 2]
    assert manager.lookup(to_hashes([1, 2]), group_id=0) == 2
    assert manager.lookup(to_hashes([1, 2]), group_id=1) == 2


def test_hybrid_manager_independent_block_ids():
    """
    Tests that each group has independent CPU block ID namespace.
    The same content stored in different groups should get different block IDs.
    """
    num_groups = 2
    block_size = 256
    num_blocks = 4

    def backend_factory(group_id: int) -> CPUBackend:
        return CPUBackend(block_size=block_size, num_blocks=num_blocks)

    def manager_factory(backend, enable_events):
        return LRUOffloadingManager(backend, enable_events)

    manager = HybridOffloadingManager(
        num_groups=num_groups,
        backend_factory=backend_factory,
        manager_factory=manager_factory,
        enable_events=False,
    )

    # Store blocks [1, 2] in group 0
    output0 = manager.prepare_store(to_hashes([1, 2]), group_id=0)
    assert output0 is not None
    spec0 = output0.store_spec
    assert isinstance(spec0, CPULoadStoreSpec)
    # First allocations should be block IDs 0, 1
    assert np.array_equal(spec0.block_ids, np.array([0, 1], dtype=np.int64))
    assert spec0.group_id == 0
    manager.complete_store(to_hashes([1, 2]), success=True, group_id=0)

    # Store blocks [1, 2] in group 1
    output1 = manager.prepare_store(to_hashes([1, 2]), group_id=1)
    assert output1 is not None
    spec1 = output1.store_spec
    assert isinstance(spec1, CPULoadStoreSpec)
    # Group 1's first allocations should also be block IDs 0, 1
    # (independent namespace)
    assert np.array_equal(spec1.block_ids, np.array([0, 1], dtype=np.int64))
    assert spec1.group_id == 1
    manager.complete_store(to_hashes([1, 2]), success=True, group_id=1)


def test_hybrid_manager_prepare_load_with_group_id():
    """
    Tests that prepare_load returns specs with correct group_id.
    """
    num_groups = 2
    block_size = 256
    num_blocks = 4

    def backend_factory(group_id: int) -> CPUBackend:
        return CPUBackend(block_size=block_size, num_blocks=num_blocks)

    def manager_factory(backend, enable_events):
        return LRUOffloadingManager(backend, enable_events)

    manager = HybridOffloadingManager(
        num_groups=num_groups,
        backend_factory=backend_factory,
        manager_factory=manager_factory,
        enable_events=False,
    )

    # Store and complete in both groups
    for group_id in range(num_groups):
        output = manager.prepare_store(to_hashes([1, 2]), group_id=group_id)
        assert output is not None
        manager.complete_store(to_hashes([1, 2]), success=True, group_id=group_id)

    # Prepare load from group 0
    load_spec_0 = manager.prepare_load(to_hashes([1, 2]), group_id=0)
    assert isinstance(load_spec_0, CPULoadStoreSpec)
    assert load_spec_0.group_id == 0
    manager.complete_load(to_hashes([1, 2]), group_id=0)

    # Prepare load from group 1
    load_spec_1 = manager.prepare_load(to_hashes([1, 2]), group_id=1)
    assert isinstance(load_spec_1, CPULoadStoreSpec)
    assert load_spec_1.group_id == 1
    manager.complete_load(to_hashes([1, 2]), group_id=1)


def test_hybrid_manager_independent_eviction():
    """
    Tests that eviction in one group doesn't affect other groups.
    """
    num_groups = 2
    block_size = 256
    num_blocks = 2  # Small capacity to trigger eviction

    def backend_factory(group_id: int) -> CPUBackend:
        return CPUBackend(block_size=block_size, num_blocks=num_blocks)

    def manager_factory(backend, enable_events):
        return LRUOffloadingManager(backend, enable_events)

    manager = HybridOffloadingManager(
        num_groups=num_groups,
        backend_factory=backend_factory,
        manager_factory=manager_factory,
        enable_events=False,
    )

    # Fill group 0 with blocks [1, 2]
    output0 = manager.prepare_store(to_hashes([1, 2]), group_id=0)
    assert output0 is not None
    manager.complete_store(to_hashes([1, 2]), success=True, group_id=0)

    # Fill group 1 with blocks [1, 2]
    output1 = manager.prepare_store(to_hashes([1, 2]), group_id=1)
    assert output1 is not None
    manager.complete_store(to_hashes([1, 2]), success=True, group_id=1)

    # Store block [3] in group 0, which should evict block [1] in group 0 only
    output = manager.prepare_store(to_hashes([3]), group_id=0)
    assert output is not None
    assert len(output.block_hashes_evicted) == 1
    assert output.block_hashes_evicted[0] == to_hashes([1])[0]
    manager.complete_store(to_hashes([3]), success=True, group_id=0)

    # Group 0: block [1] evicted, blocks [2, 3] present
    assert manager.lookup(to_hashes([1]), group_id=0) == 0
    assert manager.lookup(to_hashes([2]), group_id=0) == 1
    assert manager.lookup(to_hashes([3]), group_id=0) == 1

    # Group 1: blocks [1, 2] still present (no eviction)
    assert manager.lookup(to_hashes([1]), group_id=1) == 1
    assert manager.lookup(to_hashes([2]), group_id=1) == 1
    assert manager.lookup(to_hashes([3]), group_id=1) == 0


def test_hybrid_manager_touch_per_group():
    """
    Tests that touch() affects only the specified group's LRU order.
    """
    num_groups = 2
    block_size = 256
    num_blocks = 2

    def backend_factory(group_id: int) -> CPUBackend:
        return CPUBackend(block_size=block_size, num_blocks=num_blocks)

    def manager_factory(backend, enable_events):
        return LRUOffloadingManager(backend, enable_events)

    manager = HybridOffloadingManager(
        num_groups=num_groups,
        backend_factory=backend_factory,
        manager_factory=manager_factory,
        enable_events=False,
    )

    # Fill both groups with blocks [1, 2]
    for group_id in range(num_groups):
        output = manager.prepare_store(to_hashes([1, 2]), group_id=group_id)
        assert output is not None
        manager.complete_store(to_hashes([1, 2]), success=True, group_id=group_id)

    # Touch block [1] in group 0 only (moves to end of LRU)
    manager.touch(to_hashes([1]), group_id=0)

    # Store block [3] in group 0 -> should evict block [2] (oldest now)
    output = manager.prepare_store(to_hashes([3]), group_id=0)
    assert output is not None
    assert output.block_hashes_evicted[0] == to_hashes([2])[0]
    manager.complete_store(to_hashes([3]), success=True, group_id=0)

    # Store block [3] in group 1 -> should evict block [1] (oldest, not touched)
    output = manager.prepare_store(to_hashes([3]), group_id=1)
    assert output is not None
    assert output.block_hashes_evicted[0] == to_hashes([1])[0]
    manager.complete_store(to_hashes([3]), success=True, group_id=1)


def test_hybrid_manager_events_aggregation():
    """
    Tests that take_events() aggregates events from all groups.
    """
    num_groups = 2
    block_size = 256
    num_blocks = 4

    def backend_factory(group_id: int) -> CPUBackend:
        return CPUBackend(block_size=block_size, num_blocks=num_blocks)

    def manager_factory(backend, enable_events):
        return LRUOffloadingManager(backend, enable_events)

    manager = HybridOffloadingManager(
        num_groups=num_groups,
        backend_factory=backend_factory,
        manager_factory=manager_factory,
        enable_events=True,
    )

    # Store blocks in both groups
    manager.prepare_store(to_hashes([1, 2]), group_id=0)
    manager.complete_store(to_hashes([1, 2]), success=True, group_id=0)

    manager.prepare_store(to_hashes([3, 4]), group_id=1)
    manager.complete_store(to_hashes([3, 4]), success=True, group_id=1)

    # Take events - should include events from both groups
    events = list(manager.take_events())
    assert len(events) == 2  # One store event per group

    # Verify event contents
    all_stored_hashes = set()
    for event in events:
        assert not event.removed
        assert event.block_size == block_size
        all_stored_hashes.update(event.block_hashes)

    expected_hashes = set(to_hashes([1, 2]) + to_hashes([3, 4]))
    assert all_stored_hashes == expected_hashes


def test_hybrid_manager_get_manager():
    """
    Tests get_manager() method for direct access to group managers.
    """
    num_groups = 3
    block_size = 256
    num_blocks = 4

    def backend_factory(group_id: int) -> CPUBackend:
        return CPUBackend(block_size=block_size, num_blocks=num_blocks)

    def manager_factory(backend, enable_events):
        return LRUOffloadingManager(backend, enable_events)

    manager = HybridOffloadingManager(
        num_groups=num_groups,
        backend_factory=backend_factory,
        manager_factory=manager_factory,
        enable_events=False,
    )

    # Get individual managers
    for group_id in range(num_groups):
        inner_manager = manager.get_manager(group_id)
        assert isinstance(inner_manager, LRUOffloadingManager)


def test_hybrid_manager_single_group_backward_compat():
    """
    Tests that HybridOffloadingManager works correctly with single group
    (backward compatibility with non-HMA models).
    """
    num_groups = 1
    block_size = 256
    num_blocks = 4

    def backend_factory(group_id: int) -> CPUBackend:
        return CPUBackend(block_size=block_size, num_blocks=num_blocks)

    def manager_factory(backend, enable_events):
        return LRUOffloadingManager(backend, enable_events)

    manager = HybridOffloadingManager(
        num_groups=num_groups,
        backend_factory=backend_factory,
        manager_factory=manager_factory,
        enable_events=False,
    )

    # Should work exactly like a regular LRUOffloadingManager
    output = manager.prepare_store(to_hashes([1, 2]))  # group_id defaults to 0
    assert output is not None
    manager.complete_store(to_hashes([1, 2]))

    assert manager.lookup(to_hashes([1, 2])) == 2

    load_spec = manager.prepare_load(to_hashes([1, 2]))
    assert isinstance(load_spec, CPULoadStoreSpec)
    assert load_spec.group_id == 0
    manager.complete_load(to_hashes([1, 2]))


def test_hybrid_manager_missing_group():
    """
    Tests behavior when accessing a non-existent group.
    """
    num_groups = 2
    block_size = 256
    num_blocks = 4

    def backend_factory(group_id: int) -> CPUBackend:
        return CPUBackend(block_size=block_size, num_blocks=num_blocks)

    def manager_factory(backend, enable_events):
        return LRUOffloadingManager(backend, enable_events)

    manager = HybridOffloadingManager(
        num_groups=num_groups,
        backend_factory=backend_factory,
        manager_factory=manager_factory,
        enable_events=False,
    )

    # Lookup in non-existent group should return 0 (no hits)
    assert manager.lookup(to_hashes([1, 2]), group_id=99) == 0

    # Touch in non-existent group should be a no-op
    manager.touch(to_hashes([1, 2]), group_id=99)

    # Complete load/store in non-existent group should be a no-op
    manager.complete_load(to_hashes([1, 2]), group_id=99)
    manager.complete_store(to_hashes([1, 2]), group_id=99)
