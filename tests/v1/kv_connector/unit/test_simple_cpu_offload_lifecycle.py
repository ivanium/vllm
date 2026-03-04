# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lifecycle and scheduling tests for SimpleCPUOffloadConnector internals."""

from types import SimpleNamespace
from unittest.mock import PropertyMock, patch

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload.manager import (
    SimpleCPUOffloadScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload.metadata import (
    SimpleCPUOffloadMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload.worker import (
    SimpleCPUOffloadWorker,
)
from vllm.v1.core.kv_cache_utils import BlockHash, make_block_hash_with_group_id
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
)
from vllm.v1.outputs import KVConnectorOutput

from .utils import create_request


def _build_scheduler_output(
    num_scheduled_tokens: dict[str, int],
    finished_req_ids: set[str] | None = None,
) -> SchedulerOutput:
    return SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens=num_scheduled_tokens,
        total_num_scheduled_tokens=sum(num_scheduled_tokens.values()),
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=finished_req_ids or set(),
        free_encoder_mm_hashes=[],
        kv_connector_metadata=SimpleCPUOffloadMetadata(),
    )


def _create_scheduler_manager(
    block_size: int = 16,
    cpu_capacity_gb: float = 1.0,
    num_gpu_blocks: int = 64,
) -> SimpleCPUOffloadScheduler:
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(
            block_size=block_size,
            num_gpu_blocks=num_gpu_blocks,
        )
    )

    num_kv_heads = 8
    head_size = 64
    dtype = torch.float16
    page_size_bytes = 2 * block_size * num_kv_heads * head_size * 2

    kv_cache_config = KVCacheConfig(
        num_blocks=num_gpu_blocks,
        kv_cache_tensors=[KVCacheTensor(size=page_size_bytes, shared_by=["layer"])],
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
    return SimpleCPUOffloadScheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        cpu_capacity_bytes=int(cpu_capacity_gb * (1024**3)),
    )


class _MockKVCacheBlocks:
    def __init__(self, block_ids: list[int]):
        self._block_ids = block_ids

    def get_block_ids(self):
        return [self._block_ids]


class _DoneEvent:
    def query(self) -> bool:
        return True


def test_accumulates_pending_cpu_blocks_across_steps():
    manager = _create_scheduler_manager()
    request = create_request(num_tokens=64, block_size=16)
    req_id = request.request_id

    manager._requests[req_id] = request
    manager._request_gpu_blocks[req_id] = [[0, 1, 2, 3]]

    request.num_computed_tokens = 0
    manager.build_connector_meta(_build_scheduler_output({req_id: 16}))

    request.num_computed_tokens = 16
    manager.build_connector_meta(_build_scheduler_output({req_id: 16}))

    assert len(manager._storing_requests[req_id]) == 2
    assert len(manager._pending_cpu_blocks[req_id]) == 2


def test_store_completion_releases_refs_and_cleans_request():
    manager = _create_scheduler_manager()
    request = create_request(num_tokens=32, block_size=16)
    req_id = request.request_id

    manager._requests[req_id] = request
    manager._request_gpu_blocks[req_id] = [[0, 1]]
    manager._num_stored_blocks[req_id] = 2

    free_before_alloc = manager.cpu_block_pool.get_num_free_blocks()
    blocks = manager.cpu_block_pool.get_new_blocks(2)
    block_ids = [block.block_id for block in blocks]
    assert manager.cpu_block_pool.get_num_free_blocks() == free_before_alloc - 2

    manager._storing_requests[req_id].extend(list(request.block_hashes[:2]))
    manager._pending_cpu_blocks[req_id].extend(block_ids)

    is_async, _ = manager.request_finished(request, block_ids=[0, 1])
    assert not is_async  # Always False with ref_cnt approach

    manager.update_connector_output(
        KVConnectorOutput(
            finished_sending={req_id},
            finished_recving=None,
            invalid_block_ids=set(),
        )
    )

    assert req_id not in manager._requests
    assert req_id not in manager._request_gpu_blocks
    assert req_id not in manager._num_stored_blocks
    assert manager.cpu_block_pool.get_num_free_blocks() == free_before_alloc
    assert all(
        manager.cpu_block_pool.blocks[block_id].ref_cnt == 0 for block_id in block_ids
    )

    for block_hash in request.block_hashes[:2]:
        cached = manager.cpu_block_pool.get_cached_block(
            block_hash, kv_cache_group_ids=[0]
        )
        assert cached is not None


def test_load_touch_refs_are_released_on_finished_recving():
    manager = _create_scheduler_manager()
    request = create_request(num_tokens=16, block_size=16)
    req_id = request.request_id

    cpu_block = manager.cpu_block_pool.get_new_blocks(1)[0]
    cpu_block.block_hash = make_block_hash_with_group_id(
        request.block_hashes[0], group_id=0
    )
    manager.cpu_block_pool.cached_block_hash_to_block.insert(
        cpu_block.block_hash, cpu_block
    )
    manager.cpu_block_pool.free_blocks([cpu_block])
    assert cpu_block.ref_cnt == 0

    manager.update_state_after_alloc(
        request=request,
        blocks=_MockKVCacheBlocks([7]),
        num_external_tokens=16,
    )
    assert cpu_block.ref_cnt == 1
    assert req_id in manager._pending_load_blocks

    manager.update_connector_output(
        KVConnectorOutput(
            finished_sending=None,
            finished_recving={req_id},
            invalid_block_ids=set(),
        )
    )
    assert cpu_block.ref_cnt == 0
    assert req_id not in manager._pending_load_blocks


def test_request_finished_releases_inflight_load_refs():
    manager = _create_scheduler_manager()
    request = create_request(num_tokens=16, block_size=16)
    req_id = request.request_id

    cpu_block = manager.cpu_block_pool.get_new_blocks(1)[0]
    cpu_block.block_hash = make_block_hash_with_group_id(
        request.block_hashes[0], group_id=0
    )
    manager.cpu_block_pool.cached_block_hash_to_block.insert(
        cpu_block.block_hash, cpu_block
    )
    manager.cpu_block_pool.free_blocks([cpu_block])

    manager.update_state_after_alloc(
        request=request,
        blocks=_MockKVCacheBlocks([9]),
        num_external_tokens=16,
    )
    assert cpu_block.ref_cnt == 1

    is_async, _ = manager.request_finished(request, block_ids=[9])
    assert not is_async
    assert cpu_block.ref_cnt == 0
    assert req_id not in manager._pending_load_blocks


def test_worker_wait_for_save_queues_store_jobs():
    vllm_config = SimpleNamespace(cache_config=SimpleNamespace(block_size=16))
    worker = SimpleCPUOffloadWorker(
        vllm_config=vllm_config,
        kv_cache_config=None,
        cpu_capacity_bytes=1024 * 1024,
    )

    worker.bind_connector_metadata(
        SimpleCPUOffloadMetadata(
            store_job_idx=0,
            store_gpu_blocks=[1, 2],
            store_cpu_blocks=[3, 4],
        )
    )

    with patch.object(
        type(worker), "_is_initialized", new_callable=PropertyMock, return_value=True
    ):
        worker.wait_for_save()

    assert len(worker._pending_store_jobs) == 1
    job_idx, src, dst = worker._pending_store_jobs[0]
    assert job_idx == 0
    assert src == [1, 2]
    assert dst == [3, 4]
    assert not worker._store_events


def test_worker_start_load_submits_pending_stores_without_metadata():
    vllm_config = SimpleNamespace(cache_config=SimpleNamespace(block_size=16))
    worker = SimpleCPUOffloadWorker(
        vllm_config=vllm_config,
        kv_cache_config=None,
        cpu_capacity_bytes=1024 * 1024,
    )
    worker.clear_connector_metadata()

    called = {"count": 0}

    def _fake_submit():
        called["count"] += 1

    worker._submit_pending_stores = _fake_submit  # type: ignore[method-assign]
    with patch.object(
        type(worker), "_is_initialized", new_callable=PropertyMock, return_value=True
    ):
        worker.start_load_kv()

    assert called["count"] == 1


def test_connector_emits_finished_sending_if_stores_complete_before_req_finishes():
    """Timing logic now lives in the connector, not the worker."""
    from unittest.mock import MagicMock

    from vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload_connector import (  # noqa: E501
        SimpleCPUOffloadConnector,
    )

    _ = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=16, num_gpu_blocks=64),
        kv_transfer_config=SimpleNamespace(
            kv_connector="SimpleCPUOffloadConnector",
            kv_role="kv_both",
            kv_connector_extra_config={},
        ),
    )
    connector = SimpleCPUOffloadConnector.__new__(SimpleCPUOffloadConnector)
    connector.scheduler_manager = None
    connector._connector_metadata = None
    connector._pending_load_wm_jobs = {}
    connector._pending_store_wm_jobs = {}
    connector._stores_completed_reqs = set()
    connector._finished_reqs_waiting_for_store = set()

    # Mock the worker_handler to return a known watermark.
    mock_worker = MagicMock()
    # job_idx=0 store has fired (store_wm=0), no loads.
    mock_worker.get_completed_watermarks.return_value = (-1, 0)
    connector.worker_handler = mock_worker

    req_id = "req-early-store-done"
    # Set up snapshot: job 0 is associated with req_id.
    connector._pending_store_wm_jobs = {0: [req_id]}

    # Store event fires, but request hasn't finished yet.
    finished_sending, finished_recving = connector.get_finished(set())
    assert finished_sending is None
    assert finished_recving is None
    assert req_id not in connector._pending_store_wm_jobs
    assert req_id in connector._stores_completed_reqs

    # Now the request finishes → should be emitted as finished_sending.
    finished_sending, finished_recving = connector.get_finished({req_id})
    assert finished_sending == {req_id}
    assert finished_recving is None
    assert req_id not in connector._stores_completed_reqs


def test_cached_blocks_advance_store_cursor():
    manager = _create_scheduler_manager()
    request = create_request(num_tokens=32, block_size=16)
    req_id = request.request_id

    # Pre-cache the first block hash so scheduler should skip storing it.
    cached_block = manager.cpu_block_pool.get_new_blocks(1)[0]
    cached_block.block_hash = make_block_hash_with_group_id(
        request.block_hashes[0], group_id=0
    )
    manager.cpu_block_pool.cached_block_hash_to_block.insert(
        cached_block.block_hash, cached_block
    )
    manager.cpu_block_pool.free_blocks([cached_block])

    manager._requests[req_id] = request
    manager._request_gpu_blocks[req_id] = [[0, 1]]
    request.num_computed_tokens = 0

    meta = manager.build_connector_meta(_build_scheduler_output({req_id: 32}))
    # Block 0 is already cached, so only block 1 should be stored.
    assert meta.store_gpu_blocks == [1]
    assert meta.store_job_idx >= 0
    assert manager._num_stored_blocks[req_id] == 2

    # Repeating the same scheduling span should not re-issue any store.
    meta2 = manager.build_connector_meta(_build_scheduler_output({req_id: 32}))
    assert meta2.store_job_idx == -1
    assert meta2.store_gpu_blocks == []


def test_store_touches_gpu_blocks_to_prevent_freeing():
    """When a store is submitted, GPU blocks should have ref_cnt incremented."""
    from vllm.v1.core.block_pool import BlockPool

    manager = _create_scheduler_manager()
    request = create_request(num_tokens=32, block_size=16)
    req_id = request.request_id

    # Create a GPU block pool and bind it.
    gpu_block_pool = BlockPool(
        num_gpu_blocks=64, enable_caching=True, hash_block_size=16
    )
    manager.bind_gpu_block_pool(gpu_block_pool)

    # Allocate GPU blocks 0 and 1 for this request.
    gpu_blocks = gpu_block_pool.get_new_blocks(2)
    gpu_block_ids = [b.block_id for b in gpu_blocks]
    assert all(gpu_block_pool.blocks[bid].ref_cnt == 1 for bid in gpu_block_ids)

    manager._requests[req_id] = request
    manager._request_gpu_blocks[req_id] = [gpu_block_ids]
    request.num_computed_tokens = 0

    # Build connector meta triggers _prepare_store_specs(), which should touch.
    manager.build_connector_meta(_build_scheduler_output({req_id: 32}))

    # GPU blocks that are being stored should have ref_cnt incremented to 2.
    stored_gpu_ids = manager._pending_gpu_store_blocks.get(req_id, [])
    assert len(stored_gpu_ids) > 0
    for bid in stored_gpu_ids:
        assert gpu_block_pool.blocks[bid].ref_cnt == 2


def test_store_completion_decrements_gpu_refcnt():
    """When a store completes, GPU blocks should have ref_cnt decremented."""
    from vllm.v1.core.block_pool import BlockPool

    manager = _create_scheduler_manager()
    request = create_request(num_tokens=32, block_size=16)
    req_id = request.request_id

    gpu_block_pool = BlockPool(
        num_gpu_blocks=64, enable_caching=True, hash_block_size=16
    )
    manager.bind_gpu_block_pool(gpu_block_pool)

    # Allocate GPU blocks.
    gpu_blocks = gpu_block_pool.get_new_blocks(2)
    gpu_block_ids = [b.block_id for b in gpu_blocks]

    manager._requests[req_id] = request
    manager._request_gpu_blocks[req_id] = [gpu_block_ids]
    request.num_computed_tokens = 0

    # Build connector meta triggers store, touching GPU blocks.
    manager.build_connector_meta(_build_scheduler_output({req_id: 32}))
    stored_gpu_ids = manager._pending_gpu_store_blocks[req_id]
    assert all(gpu_block_pool.blocks[bid].ref_cnt == 2 for bid in stored_gpu_ids)

    # Simulate store completion.
    manager.update_connector_output(
        KVConnectorOutput(
            finished_sending={req_id},
            finished_recving=None,
            invalid_block_ids=set(),
        )
    )

    # GPU blocks should have ref_cnt decremented back to 1
    # (still held by the kv_cache_manager allocation).
    for bid in stored_gpu_ids:
        assert gpu_block_pool.blocks[bid].ref_cnt == 1

    # Pending GPU store blocks should be cleaned up.
    assert req_id not in manager._pending_gpu_store_blocks


def test_lazy_store_touches_and_releases_gpu_blocks():
    """Lazy mode: GPU eviction candidates are touched during store,
    freed on completion."""
    from vllm.v1.core.block_pool import BlockPool

    manager = _create_scheduler_manager()
    manager._lazy_mode = True

    # Use a small GPU pool so that hashed free blocks are reachable by
    # get_eviction_candidates (which peeks from the LRU front with 2x
    # oversample). With 5 total blocks (1 null + 4 usable), after we
    # allocate all 4, set hashes on 2, and free them all, the hashed
    # blocks sit near the front of the free queue.
    gpu_block_pool = BlockPool(
        num_gpu_blocks=5, enable_caching=True, hash_block_size=16
    )
    manager.bind_gpu_block_pool(gpu_block_pool)

    # Allocate all usable blocks from the pool.
    all_blocks = gpu_block_pool.get_new_blocks(4)

    # Set block hashes on the first 2 (simulating cached prefix blocks).
    hashed_blocks = all_blocks[:2]
    unhashed_blocks = all_blocks[2:]
    for i, block in enumerate(hashed_blocks):
        block.block_hash = make_block_hash_with_group_id(
            BlockHash(b"hash" + str(i).encode()), group_id=0
        )
        gpu_block_pool.cached_block_hash_to_block.insert(block.block_hash, block)

    # Free hashed blocks first so they land at the LRU front,
    # then free unhashed blocks (which go after).
    gpu_block_pool.free_blocks(hashed_blocks)
    gpu_block_pool.free_blocks(unhashed_blocks)
    gpu_block_ids = [b.block_id for b in hashed_blocks]
    assert all(gpu_block_pool.blocks[bid].ref_cnt == 0 for bid in gpu_block_ids)

    # Build connector meta triggers lazy store.
    meta = manager.build_connector_meta(_build_scheduler_output({"some_req": 32}))

    # Lazy store should have picked up the hashed blocks.
    assert meta.store_job_idx >= 0, "Expected a lazy store job"

    # GPU blocks being stored should be touched (ref_cnt > 0).
    for bid in meta.store_gpu_blocks:
        assert gpu_block_pool.blocks[bid].ref_cnt > 0

    # Simulate lazy store completion via sentinel.
    sentinel = f"__lazy_store_{meta.store_job_idx}"
    manager.update_connector_output(
        KVConnectorOutput(
            finished_sending={sentinel},
            finished_recving=None,
            invalid_block_ids=set(),
        )
    )

    # GPU blocks should be back to ref_cnt=0.
    for bid in meta.store_gpu_blocks:
        assert gpu_block_pool.blocks[bid].ref_cnt == 0


def test_cleanup_request_releases_gpu_refcnt_on_abort():
    """If a request is cleaned up before store completion,
    GPU ref_cnt is released."""
    from vllm.v1.core.block_pool import BlockPool

    manager = _create_scheduler_manager()
    gpu_block_pool = BlockPool(
        num_gpu_blocks=64, enable_caching=True, hash_block_size=16
    )
    manager.bind_gpu_block_pool(gpu_block_pool)

    req_id = "aborted-req"
    gpu_blocks = gpu_block_pool.get_new_blocks(2)
    gpu_block_ids = [b.block_id for b in gpu_blocks]

    # Simulate connector having touched GPU blocks for a store.
    gpu_block_pool.touch(gpu_blocks)  # ref_cnt: 1 -> 2
    manager._pending_gpu_store_blocks[req_id] = gpu_block_ids

    # Cleanup should release the connector's ref.
    manager._cleanup_request(req_id)

    for bid in gpu_block_ids:
        assert gpu_block_pool.blocks[bid].ref_cnt == 1  # Back to original
    assert req_id not in manager._pending_gpu_store_blocks


def test_request_finished_returns_false_even_with_inflight_stores():
    """request_finished() always returns False now; ref_cnt protects blocks."""
    manager = _create_scheduler_manager()
    request = create_request(num_tokens=32, block_size=16)
    req_id = request.request_id

    # Set up state as if stores are in-flight.
    manager._requests[req_id] = request
    manager._request_gpu_blocks[req_id] = [[0, 1]]
    manager._storing_requests[req_id].extend(list(request.block_hashes[:2]))

    is_async, params = manager.request_finished(request, block_ids=[0, 1])
    assert is_async is False  # NEW: always False
    assert params is None

    # Connector state should NOT be cleaned up yet (stores still in-flight).
    assert req_id in manager._storing_requests
