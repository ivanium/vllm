# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lifecycle and scheduling tests for SimpleCPUOffloadConnector internals."""

from types import SimpleNamespace

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload.manager import (
    SimpleCPUOffloadScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload.metadata import (
    SimpleCPUOffloadMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload.worker import (
    SimpleCPUOffloadWorker,
    TransferJob,
)
from vllm.v1.core.kv_cache_utils import make_block_hash_with_group_id
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
    assert is_async

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

    worker.gpu_kv_cache = torch.zeros((8, 1), dtype=torch.float16)
    worker.cpu_kv_cache = torch.zeros((8, 1), dtype=torch.float16)
    worker.bind_connector_metadata(
        SimpleCPUOffloadMetadata(reqs_to_store={"req-1": ([1, 2], [3, 4])})
    )

    worker.wait_for_save()

    assert len(worker._unsubmitted_store_jobs) == 1
    assert worker._store_jobs["req-1"]
    assert not worker._active_jobs


def test_worker_start_load_submits_pending_stores_without_metadata():
    vllm_config = SimpleNamespace(cache_config=SimpleNamespace(block_size=16))
    worker = SimpleCPUOffloadWorker(
        vllm_config=vllm_config,
        kv_cache_config=None,
        cpu_capacity_bytes=1024 * 1024,
    )
    worker.gpu_kv_cache = torch.zeros((4, 1), dtype=torch.float16)
    worker.cpu_kv_cache = torch.zeros((4, 1), dtype=torch.float16)
    worker.clear_connector_metadata()

    called = {"count": 0}

    def _fake_submit():
        called["count"] += 1

    worker._submit_pending_store_jobs = _fake_submit  # type: ignore[method-assign]
    worker.start_load_kv()

    assert called["count"] == 1


def test_worker_emits_finished_sending_if_stores_complete_before_req_finishes():
    vllm_config = SimpleNamespace(cache_config=SimpleNamespace(block_size=16))
    worker = SimpleCPUOffloadWorker(
        vllm_config=vllm_config,
        kv_cache_config=None,
        cpu_capacity_bytes=1024 * 1024,
    )

    req_id = "req-early-store-done"
    worker._store_jobs[req_id].add(7)
    worker._active_jobs[7] = TransferJob(  # type: ignore[arg-type]
        req_id=req_id,
        is_store=True,
        event=_DoneEvent(),
    )

    finished_sending, finished_recving = worker.get_finished(set())
    assert finished_sending is None
    assert finished_recving is None
    assert req_id in worker._store_jobs
    assert not worker._store_jobs[req_id]

    finished_sending, finished_recving = worker.get_finished({req_id})
    assert finished_sending == {req_id}
    assert finished_recving is None
    assert req_id not in worker._store_jobs


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
    assert req_id in meta.reqs_to_store
    assert meta.reqs_to_store[req_id][0] == [1]
    assert manager._num_stored_blocks[req_id] == 2

    # Repeating the same scheduling span should not re-check from block 0.
    meta2 = manager.build_connector_meta(_build_scheduler_output({req_id: 32}))
    assert req_id not in meta2.reqs_to_store
