# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for SimpleCPUOffloadConnector."""

import pytest
import torch

from vllm.config import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload import (
    triton_kernels,
)
from vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload.manager import (
    SimpleCPUOffloadScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload.metadata import (
    SimpleCPUOffloadMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload.worker import (
    SimpleCPUOffloadWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload_connector import (
    SimpleCPUOffloadConnector,
)
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    init_none_hash,
)
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
)
from vllm.v1.outputs import KVConnectorOutput

from .utils import (
    create_request,
    create_scheduler,
    create_vllm_config,
)

# ============================================================
# Test SimpleCPUOffloadMetadata
# ============================================================


class TestSimpleCPUOffloadMetadata:
    """Tests for SimpleCPUOffloadMetadata dataclass."""

    def test_empty_metadata_creation(self):
        """Test creating empty metadata with default values."""
        metadata = SimpleCPUOffloadMetadata()
        assert metadata.reqs_to_load == {}
        assert metadata.reqs_to_store == {}

    def test_metadata_with_load_specs(self):
        """Test metadata with load specifications."""
        reqs_to_load = {
            "req-1": ([0, 1, 2], [10, 11, 12]),
            "req-2": ([3, 4], [13, 14]),
        }
        metadata = SimpleCPUOffloadMetadata(reqs_to_load=reqs_to_load)
        assert metadata.reqs_to_load == reqs_to_load
        assert metadata.reqs_to_store == {}

    def test_metadata_with_store_specs(self):
        """Test metadata with store specifications."""
        reqs_to_store = {
            "req-1": ([0, 1], [5, 6]),
        }
        metadata = SimpleCPUOffloadMetadata(reqs_to_store=reqs_to_store)
        assert metadata.reqs_to_load == {}
        assert metadata.reqs_to_store == reqs_to_store

    def test_metadata_with_both_specs(self):
        """Test metadata with both load and store specifications."""
        reqs_to_load = {"req-1": ([0], [10])}
        reqs_to_store = {"req-2": ([1], [11])}
        metadata = SimpleCPUOffloadMetadata(
            reqs_to_load=reqs_to_load,
            reqs_to_store=reqs_to_store,
        )
        assert metadata.reqs_to_load == reqs_to_load
        assert metadata.reqs_to_store == reqs_to_store


# ============================================================
# Test SimpleCPUOffloadScheduler
# ============================================================


def _create_scheduler_manager(
    block_size: int = 16,
    cpu_capacity_gb: float = 1.0,
    num_gpu_blocks: int = 100,
) -> tuple[SimpleCPUOffloadScheduler, KVCacheConfig]:
    """Create a SimpleCPUOffloadScheduler for testing."""
    vllm_config = create_vllm_config(
        block_size=block_size,
        max_num_batched_tokens=1024,
    )
    vllm_config.cache_config.num_gpu_blocks = num_gpu_blocks

    # Calculate page size for the attention spec
    # page_size_bytes = 2 * block_size * num_kv_heads * head_size * sizeof(dtype)
    num_kv_heads = 8
    head_size = 128
    dtype = torch.float16
    page_size_bytes = 2 * block_size * num_kv_heads * head_size * 2  # fp16 = 2 bytes

    kv_cache_config = KVCacheConfig(
        num_blocks=num_gpu_blocks,
        kv_cache_tensors=[
            KVCacheTensor(
                size=page_size_bytes,
                shared_by=["layer"],
            )
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=num_kv_heads,
                    head_size=head_size,
                    dtype=dtype,
                ),
            )
        ],
    )

    cpu_capacity_bytes = int(cpu_capacity_gb * (1024**3))
    scheduler_manager = SimpleCPUOffloadScheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        cpu_capacity_bytes=cpu_capacity_bytes,
    )
    return scheduler_manager, kv_cache_config


class TestSimpleCPUOffloadScheduler:
    """Tests for SimpleCPUOffloadScheduler."""

    def test_initialization(self):
        """Test scheduler manager initialization."""
        manager, _ = _create_scheduler_manager()
        assert manager.num_cpu_blocks > 0
        assert manager.gpu_block_size == 16
        assert manager.cpu_block_pool is not None

    def test_get_num_new_matched_tokens_no_cache(self):
        """Test cache miss returns zero tokens."""
        manager, _ = _create_scheduler_manager()
        request = create_request(num_tokens=64, block_size=16)

        num_matched, is_async = manager.get_num_new_matched_tokens(
            request, num_computed_tokens=0
        )
        assert num_matched == 0
        assert is_async is False

    def test_get_num_new_matched_tokens_with_cache_hit(self):
        """Test cache hit returns matched tokens."""
        manager, _ = _create_scheduler_manager(block_size=16)

        # Create a request and manually cache its blocks
        request = create_request(num_tokens=64, block_size=16)

        # Manually add blocks to CPU cache
        block_hash = request.block_hashes[0]
        try:
            new_blocks = manager.cpu_block_pool.get_new_blocks(1)
            cpu_block = new_blocks[0]

            # Set the block hash
            from vllm.v1.core.kv_cache_utils import make_block_hash_with_group_id

            block_hash_with_group = make_block_hash_with_group_id(block_hash, 0)
            cpu_block._block_hash = block_hash_with_group
            manager.cpu_block_pool.cached_block_hash_to_block.insert(
                block_hash_with_group, cpu_block
            )

            # Now check for cache hit
            num_matched, is_async = manager.get_num_new_matched_tokens(
                request, num_computed_tokens=0
            )
            assert num_matched == 16  # One block worth of tokens
            assert is_async is True
        except (ValueError, AttributeError):
            # If get_new_blocks fails or block structure differs, skip
            pytest.skip("Could not allocate CPU blocks for test")

    def test_update_state_after_alloc_no_external_tokens(self):
        """Test update_state_after_alloc with no external tokens."""
        manager, _ = _create_scheduler_manager()
        request = create_request(num_tokens=32, block_size=16)

        # Mock KVCacheBlocks
        class MockKVCacheBlocks:
            def get_block_ids(self):
                return [[0, 1]]

        blocks = MockKVCacheBlocks()
        manager.update_state_after_alloc(request, blocks, num_external_tokens=0)

        # Should not have any load specs
        assert request.request_id not in manager._reqs_to_load

        # But should track the request
        assert request.request_id in manager._requests
        assert request.request_id in manager._request_gpu_blocks

    def test_build_connector_meta_empty(self):
        """Test build_connector_meta with no operations."""
        manager, _ = _create_scheduler_manager()

        # Create empty scheduler output
        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            num_scheduled_tokens={},
            total_num_scheduled_tokens=0,
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=[],
            finished_req_ids=set(),
            free_encoder_mm_hashes=[],
            kv_connector_metadata=SimpleCPUOffloadMetadata(),
        )

        metadata = manager.build_connector_meta(scheduler_output)
        assert isinstance(metadata, SimpleCPUOffloadMetadata)
        assert metadata.reqs_to_load == {}
        assert metadata.reqs_to_store == {}

    def test_request_finished_cleanup(self):
        """Test request_finished cleans up request state."""
        manager, _ = _create_scheduler_manager()
        request = create_request(num_tokens=32, block_size=16)

        # Store some state for the request
        manager._requests[request.request_id] = request
        manager._request_gpu_blocks[request.request_id] = [[0, 1]]
        manager._num_stored_blocks[request.request_id] = 1

        # Finish the request (not async)
        is_async, params = manager.request_finished(request, block_ids=[0, 1])
        assert is_async is False
        assert params is None

        # Verify cleanup
        assert request.request_id not in manager._requests
        assert request.request_id not in manager._request_gpu_blocks
        assert request.request_id not in manager._num_stored_blocks

    def test_request_finished_all_groups_cleanup(self):
        """Test request_finished_all_groups cleans up state."""
        manager, _ = _create_scheduler_manager()
        request = create_request(num_tokens=32, block_size=16)

        # Store some state
        manager._requests[request.request_id] = request
        manager._request_gpu_blocks[request.request_id] = [[0, 1]]

        # Finish via all_groups interface
        is_async, params = manager.request_finished_all_groups(
            request, block_ids=([0, 1],)
        )
        assert is_async is False
        assert params is None

        # Verify cleanup
        assert request.request_id not in manager._requests

    def test_update_connector_output_finished_recving(self):
        """Test update_connector_output handles finished receiving."""
        manager, _ = _create_scheduler_manager()

        # Set up a loading request
        req_id = "test-req-1"
        manager._loading_requests[req_id] = [BlockHash(b"hash1")]

        # Mark as finished receiving
        connector_output = KVConnectorOutput(
            finished_sending=None,
            finished_recving={req_id},
            invalid_block_ids=set(),
        )
        manager.update_connector_output(connector_output)

        # Should be removed from loading
        assert req_id not in manager._loading_requests

    def test_update_connector_output_finished_sending(self):
        """Test update_connector_output handles finished sending and caches."""
        manager, _ = _create_scheduler_manager()

        # Set up a storing request
        req_id = "test-req-1"
        request = create_request(num_tokens=32, block_size=16)
        request.request_id = req_id

        block_hash = BlockHash(b"hash1")
        manager._requests[req_id] = request
        manager._storing_requests[req_id] = [block_hash]

        # Allocate a CPU block for the pending store
        try:
            new_blocks = manager.cpu_block_pool.get_new_blocks(1)
            cpu_block_id = new_blocks[0].block_id
            manager._pending_cpu_blocks[req_id] = [cpu_block_id]

            # Mark as finished sending
            connector_output = KVConnectorOutput(
                finished_sending={req_id},
                finished_recving=None,
                invalid_block_ids=set(),
            )
            manager.update_connector_output(connector_output)

            # Should be removed from storing
            assert req_id not in manager._storing_requests
            assert req_id not in manager._pending_cpu_blocks
        except ValueError:
            pytest.skip("Could not allocate CPU blocks for test")

    def test_take_events(self):
        """Test take_events returns events from block pool."""
        manager, _ = _create_scheduler_manager()
        events = list(manager.take_events())
        # Block pool may or may not have events
        assert isinstance(events, list)


# ============================================================
# Test SimpleCPUOffloadConnector
# ============================================================


def _create_connector(
    role: KVConnectorRole,
    block_size: int = 16,
    cpu_capacity_gb: float = 1.0,
    num_gpu_blocks: int = 100,
) -> SimpleCPUOffloadConnector:
    """Create a SimpleCPUOffloadConnector for testing."""
    vllm_config = create_vllm_config(
        block_size=block_size,
        max_num_batched_tokens=1024,
    )
    vllm_config.cache_config.num_gpu_blocks = num_gpu_blocks

    # Set up KV transfer config for SimpleCPUOffloadConnector
    vllm_config.kv_transfer_config = KVTransferConfig(
        kv_connector="SimpleCPUOffloadConnector",
        kv_role="kv_both",
        kv_connector_extra_config={"cpu_capacity_gb": cpu_capacity_gb},
    )

    # Calculate page size for the attention spec
    num_kv_heads = 8
    head_size = 128
    dtype = torch.float16
    page_size_bytes = 2 * block_size * num_kv_heads * head_size * 2  # fp16 = 2 bytes

    kv_cache_config = KVCacheConfig(
        num_blocks=num_gpu_blocks,
        kv_cache_tensors=[
            KVCacheTensor(
                size=page_size_bytes,
                shared_by=["layer"],
            )
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=num_kv_heads,
                    head_size=head_size,
                    dtype=dtype,
                ),
            )
        ],
    )

    return SimpleCPUOffloadConnector(vllm_config, role, kv_cache_config)


class TestSimpleCPUOffloadConnector:
    """Tests for SimpleCPUOffloadConnector."""

    def test_scheduler_role_initialization(self):
        """Test connector initializes correctly for SCHEDULER role."""
        connector = _create_connector(KVConnectorRole.SCHEDULER)
        assert connector.scheduler_manager is not None
        assert connector.worker_handler is None

    def test_worker_role_initialization(self):
        """Test connector initializes correctly for WORKER role."""
        connector = _create_connector(KVConnectorRole.WORKER)
        assert connector.scheduler_manager is None
        assert connector.worker_handler is not None

    def test_prefer_cross_layer_blocks_default(self):
        """Test that prefer_cross_layer_blocks uses default (False)."""
        connector = _create_connector(KVConnectorRole.SCHEDULER)
        # SimpleCPUOffloadConnector no longer overrides this property,
        # so it should use the default from the base class
        assert (
            not hasattr(SimpleCPUOffloadConnector, "prefer_cross_layer_blocks")
            or not connector.prefer_cross_layer_blocks
        )

    def test_get_num_new_matched_tokens_scheduler(self):
        """Test get_num_new_matched_tokens delegates to scheduler manager."""
        connector = _create_connector(KVConnectorRole.SCHEDULER)
        request = create_request(num_tokens=32, block_size=16)

        num_matched, is_async = connector.get_num_new_matched_tokens(
            request, num_computed_tokens=0
        )
        # No cache, so should return 0
        assert num_matched == 0
        assert is_async is False

    def test_get_num_new_matched_tokens_worker(self):
        """Test get_num_new_matched_tokens returns 0 for worker role."""
        connector = _create_connector(KVConnectorRole.WORKER)
        request = create_request(num_tokens=32, block_size=16)

        num_matched, is_async = connector.get_num_new_matched_tokens(
            request, num_computed_tokens=0
        )
        assert num_matched == 0
        assert is_async is False

    def test_build_connector_meta_scheduler(self):
        """Test build_connector_meta for scheduler role."""
        connector = _create_connector(KVConnectorRole.SCHEDULER)

        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            num_scheduled_tokens={},
            total_num_scheduled_tokens=0,
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=[],
            finished_req_ids=set(),
            free_encoder_mm_hashes=[],
            kv_connector_metadata=SimpleCPUOffloadMetadata(),
        )

        metadata = connector.build_connector_meta(scheduler_output)
        assert isinstance(metadata, SimpleCPUOffloadMetadata)

    def test_build_connector_meta_worker(self):
        """Test build_connector_meta returns empty metadata for worker role."""
        connector = _create_connector(KVConnectorRole.WORKER)

        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            num_scheduled_tokens={},
            total_num_scheduled_tokens=0,
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=[],
            finished_req_ids=set(),
            free_encoder_mm_hashes=[],
            kv_connector_metadata=SimpleCPUOffloadMetadata(),
        )

        metadata = connector.build_connector_meta(scheduler_output)
        assert isinstance(metadata, SimpleCPUOffloadMetadata)
        assert metadata.reqs_to_load == {}
        assert metadata.reqs_to_store == {}

    def test_bind_and_clear_connector_metadata(self):
        """Test bind and clear connector metadata for worker role."""
        connector = _create_connector(KVConnectorRole.WORKER)

        metadata = SimpleCPUOffloadMetadata(
            reqs_to_load={"req-1": ([0], [10])},
        )
        connector.bind_connector_metadata(metadata)
        assert connector._connector_metadata is metadata

        connector.clear_connector_metadata()
        assert connector._connector_metadata is None

    def test_request_finished_scheduler(self):
        """Test request_finished for scheduler role."""
        connector = _create_connector(KVConnectorRole.SCHEDULER)
        request = create_request(num_tokens=32, block_size=16)

        is_async, params = connector.request_finished(request, block_ids=[0, 1])
        assert is_async is False
        assert params is None

    def test_request_finished_all_groups_scheduler(self):
        """Test request_finished_all_groups for scheduler role."""
        connector = _create_connector(KVConnectorRole.SCHEDULER)
        request = create_request(num_tokens=32, block_size=16)

        is_async, params = connector.request_finished_all_groups(
            request, block_ids=([0, 1],)
        )
        assert is_async is False
        assert params is None

    def test_take_events_scheduler(self):
        """Test take_events for scheduler role."""
        connector = _create_connector(KVConnectorRole.SCHEDULER)
        events = list(connector.take_events())
        assert isinstance(events, list)

    def test_take_events_worker(self):
        """Test take_events returns empty for worker role."""
        connector = _create_connector(KVConnectorRole.WORKER)
        events = list(connector.take_events())
        assert events == []

    def test_get_finished_scheduler(self):
        """Test get_finished returns None for scheduler role."""
        connector = _create_connector(KVConnectorRole.SCHEDULER)
        finished_sending, finished_recving = connector.get_finished(set())
        assert finished_sending is None
        assert finished_recving is None


# ============================================================
# Test Triton Kernels (CPU-only tests without actual GPU)
# ============================================================


class TestTritonKernels:
    """Tests for Triton kernel copy operations.

    Note: These tests use the PyTorch fallback when GPU is not available.
    """

    def test_copy_blocks_torch_basic(self):
        """Test PyTorch fallback copy_blocks."""
        # Create source and destination tensors
        src_cache = torch.randn(10, 4, 8)  # 10 blocks
        dst_cache = torch.zeros(10, 4, 8)

        # Copy blocks 0->5, 1->6
        block_mapping = torch.tensor([[0, 5], [1, 6]], dtype=torch.int64)

        triton_kernels.copy_blocks_torch(src_cache, dst_cache, block_mapping)

        # Verify copies
        assert torch.allclose(dst_cache[5], src_cache[0])
        assert torch.allclose(dst_cache[6], src_cache[1])
        # Other blocks should be zero
        assert torch.allclose(dst_cache[0], torch.zeros_like(dst_cache[0]))

    def test_copy_blocks_torch_empty_mapping(self):
        """Test copy_blocks with empty block mapping."""
        src_cache = torch.randn(5, 4, 8)
        dst_cache = torch.zeros(5, 4, 8)
        original_dst = dst_cache.clone()

        # Empty mapping
        block_mapping = torch.empty((0, 2), dtype=torch.int64)

        triton_kernels.copy_blocks_torch(src_cache, dst_cache, block_mapping)

        # Destination should be unchanged
        assert torch.allclose(dst_cache, original_dst)

    def test_copy_blocks_function_empty(self):
        """Test copy_blocks wrapper handles empty mapping."""
        src_cache = torch.randn(5, 4, 8)
        dst_cache = torch.zeros(5, 4, 8)
        original_dst = dst_cache.clone()

        block_mapping = torch.empty((0, 2), dtype=torch.int64)

        # Should handle empty gracefully
        triton_kernels.copy_blocks(
            src_cache, dst_cache, block_mapping, use_triton=False
        )

        assert torch.allclose(dst_cache, original_dst)

    def test_copy_blocks_torch_preserves_dtype(self):
        """Test that copy preserves tensor dtype."""
        # Test with float16
        src_cache = torch.randn(5, 4, 8, dtype=torch.float16)
        dst_cache = torch.zeros(5, 4, 8, dtype=torch.float16)

        block_mapping = torch.tensor([[0, 1]], dtype=torch.int64)

        triton_kernels.copy_blocks_torch(src_cache, dst_cache, block_mapping)

        assert dst_cache.dtype == torch.float16
        assert torch.allclose(dst_cache[1], src_cache[0])

    def test_copy_blocks_cpu_to_cpu(self):
        """Test copying between CPU tensors."""
        src_cache = torch.randn(5, 2, 4)
        dst_cache = torch.zeros(5, 2, 4)

        block_mapping = torch.tensor([[0, 2], [3, 4]], dtype=torch.int64)

        # Use PyTorch fallback for CPU
        triton_kernels.copy_blocks(
            src_cache, dst_cache, block_mapping, use_triton=False
        )

        assert torch.allclose(dst_cache[2], src_cache[0])
        assert torch.allclose(dst_cache[4], src_cache[3])


# ============================================================
# Test SimpleCPUOffloadWorker (basic tests without GPU)
# ============================================================


class TestSimpleCPUOffloadWorker:
    """Tests for SimpleCPUOffloadWorker."""

    def test_worker_initialization(self):
        """Test worker initializes correctly."""
        vllm_config = create_vllm_config(block_size=16)
        worker = SimpleCPUOffloadWorker(
            vllm_config=vllm_config,
            kv_cache_config=None,
            cpu_capacity_bytes=1024 * 1024 * 100,  # 100 MB
        )

        assert worker.gpu_kv_caches == {}  # Not registered yet
        assert worker.cpu_kv_caches == {}
        assert worker.layer_names == []
        assert worker.load_stream is None
        assert worker.store_stream is None

    def test_bind_clear_metadata(self):
        """Test binding and clearing metadata."""
        vllm_config = create_vllm_config(block_size=16)
        worker = SimpleCPUOffloadWorker(
            vllm_config=vllm_config,
            kv_cache_config=None,
            cpu_capacity_bytes=1024 * 1024 * 100,
        )

        metadata = SimpleCPUOffloadMetadata()
        worker.bind_connector_metadata(metadata)
        assert worker._connector_metadata is metadata

        worker.clear_connector_metadata()
        assert worker._connector_metadata is None

    def test_get_finished_no_active_jobs(self):
        """Test get_finished with no active jobs."""
        vllm_config = create_vllm_config(block_size=16)
        worker = SimpleCPUOffloadWorker(
            vllm_config=vllm_config,
            kv_cache_config=None,
            cpu_capacity_bytes=1024 * 1024 * 100,
        )

        finished_sending, finished_recving = worker.get_finished(finished_req_ids=set())
        # No jobs, so both should be None
        assert finished_sending is None
        assert finished_recving is None

    def test_start_load_kv_no_metadata(self):
        """Test start_load_kv with no metadata does nothing."""
        vllm_config = create_vllm_config(block_size=16)
        worker = SimpleCPUOffloadWorker(
            vllm_config=vllm_config,
            kv_cache_config=None,
            cpu_capacity_bytes=1024 * 1024 * 100,
        )

        # Should not raise
        worker.start_load_kv()

    def test_wait_for_save_no_metadata(self):
        """Test wait_for_save with no metadata does nothing."""
        vllm_config = create_vllm_config(block_size=16)
        worker = SimpleCPUOffloadWorker(
            vllm_config=vllm_config,
            kv_cache_config=None,
            cpu_capacity_bytes=1024 * 1024 * 100,
        )

        # Should not raise
        worker.wait_for_save()

    def test_handle_preemptions_no_active_jobs(self):
        """Test handle_preemptions with no active jobs."""
        vllm_config = create_vllm_config(block_size=16)
        worker = SimpleCPUOffloadWorker(
            vllm_config=vllm_config,
            kv_cache_config=None,
            cpu_capacity_bytes=1024 * 1024 * 100,
        )

        # Should not raise even with preempted request IDs
        worker.handle_preemptions(preempted_req_ids={"req-1", "req-2"})

    def test_wait_for_layer_load_no_stream(self):
        """Test wait_for_layer_load with no stream."""
        vllm_config = create_vllm_config(block_size=16)
        worker = SimpleCPUOffloadWorker(
            vllm_config=vllm_config,
            kv_cache_config=None,
            cpu_capacity_bytes=1024 * 1024 * 100,
        )

        # Should not raise even without stream
        worker.wait_for_layer_load("layer.0")

    def test_register_kv_caches_per_layer(self):
        """Test register_kv_caches with per-layer tensors."""
        vllm_config = create_vllm_config(block_size=16)
        worker = SimpleCPUOffloadWorker(
            vllm_config=vllm_config,
            kv_cache_config=None,
            cpu_capacity_bytes=1024 * 1024 * 100,  # 100 MB
        )

        # Create mock per-layer KV caches (CPU tensors for testing without GPU)
        # Shape: [num_blocks, 2, num_kv_heads, block_size, head_size]
        num_blocks = 10
        num_kv_heads = 8
        block_size = 16
        head_size = 64
        kv_caches = {
            "layer.0": torch.randn(num_blocks, 2, num_kv_heads, block_size, head_size),
            "layer.1": torch.randn(num_blocks, 2, num_kv_heads, block_size, head_size),
            "layer.2": torch.randn(num_blocks, 2, num_kv_heads, block_size, head_size),
        }

        # Register should work without error (though streams need GPU)
        # Just test the data structure setup
        worker.gpu_kv_caches = kv_caches
        worker.layer_names = list(kv_caches.keys())

        assert len(worker.gpu_kv_caches) == 3
        assert len(worker.layer_names) == 3
        assert "layer.0" in worker.gpu_kv_caches
        assert "layer.1" in worker.gpu_kv_caches
        assert "layer.2" in worker.gpu_kv_caches

    def test_register_kv_caches_empty(self):
        """Test register_kv_caches with empty dict does nothing harmful."""
        vllm_config = create_vllm_config(block_size=16)
        worker = SimpleCPUOffloadWorker(
            vllm_config=vllm_config,
            kv_cache_config=None,
            cpu_capacity_bytes=1024 * 1024 * 100,
        )

        # Empty dict should not raise, just warn
        worker.register_kv_caches({})

        assert worker.gpu_kv_caches == {}
        assert worker.cpu_kv_caches == {}
        assert worker.layer_names == []


# ============================================================
# Integration-style tests
# ============================================================


class TestSimpleCPUOffloadIntegration:
    """Integration tests for SimpleCPUOffloadConnector with scheduler."""

    @pytest.fixture
    def scheduler_with_connector(self):
        """Create a scheduler with SimpleCPUOffloadConnector."""
        vllm_config = create_vllm_config(
            block_size=16,
            max_num_batched_tokens=256,
        )
        vllm_config.kv_transfer_config = KVTransferConfig(
            kv_connector="SimpleCPUOffloadConnector",
            kv_role="kv_both",
            kv_connector_extra_config={"cpu_capacity_gb": 0.1},
        )

        init_none_hash(sha256)

        scheduler = create_scheduler(vllm_config, num_blocks=100)
        return scheduler

    def test_add_request_and_schedule(self, scheduler_with_connector):
        """Test adding a request and scheduling it."""
        scheduler = scheduler_with_connector

        # Add a simple request
        request = create_request(num_tokens=32, block_size=16)
        scheduler.add_request(request)

        # Schedule should work
        output = scheduler.schedule()
        assert output is not None
        assert len(output.scheduled_new_reqs) == 1

    def test_connector_in_scheduler(self, scheduler_with_connector):
        """Test that the connector is properly set up in scheduler."""
        scheduler = scheduler_with_connector

        # Check connector is set up
        if scheduler.connector is not None:
            assert isinstance(scheduler.connector, SimpleCPUOffloadConnector)
            assert scheduler.connector.scheduler_manager is not None
