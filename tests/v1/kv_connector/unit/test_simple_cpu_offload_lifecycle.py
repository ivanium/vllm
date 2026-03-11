# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lifecycle tests for SimpleCPUOffloadScheduler.

Covers the core invariants:
  1. Eager store: tracks requests, advances cursors, touches/releases GPU refs
  2. Load: touches CPU+GPU refs, releases on completion or request_finished
  3. Deferred cleanup: request_finished defers when transfers are in-flight
  4. Lazy store: touches eviction candidates, releases on completion
"""

from types import SimpleNamespace

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload.manager import (
    RequestState,
    SimpleCPUOffloadScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload.metadata import (
    SimpleCPUOffloadMetadata,
)
from vllm.v1.core.block_pool import BlockPool
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

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BLOCK_SIZE = 16
NUM_KV_HEADS = 8
HEAD_SIZE = 64
DTYPE = torch.float16
PAGE_BYTES = 2 * BLOCK_SIZE * NUM_KV_HEADS * HEAD_SIZE * 2  # K+V


def _make_kv_cache_config(num_blocks: int = 64) -> KVCacheConfig:
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[KVCacheTensor(size=PAGE_BYTES, shared_by=["layer"])],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(
                    block_size=BLOCK_SIZE,
                    num_kv_heads=NUM_KV_HEADS,
                    head_size=HEAD_SIZE,
                    dtype=DTYPE,
                ),
            )
        ],
    )


def _make_vllm_config() -> SimpleNamespace:
    return SimpleNamespace(
        cache_config=SimpleNamespace(block_size=BLOCK_SIZE, num_gpu_blocks=64),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=256),
        model_config=SimpleNamespace(max_model_len=4096),
    )


def _make_manager(
    lazy: bool = False, num_gpu_blocks: int = 64
) -> SimpleCPUOffloadScheduler:
    return SimpleCPUOffloadScheduler(
        vllm_config=_make_vllm_config(),
        kv_cache_config=_make_kv_cache_config(num_gpu_blocks),
        cpu_capacity_bytes=1 * 1024**3,
        lazy_offload=lazy,
    )


def _make_gpu_pool(num_blocks: int = 64) -> BlockPool:
    return BlockPool(
        num_gpu_blocks=num_blocks, enable_caching=True, hash_block_size=BLOCK_SIZE
    )


def _sched_output(
    num_scheduled: dict[str, int] | None = None,
    finished: set[str] | None = None,
) -> SchedulerOutput:
    tokens = num_scheduled or {}
    return SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens=tokens,
        total_num_scheduled_tokens=sum(tokens.values()),
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=finished or set(),
        free_encoder_mm_hashes=[],
        kv_connector_metadata=SimpleCPUOffloadMetadata(),
    )


def _store_done(event_idx: int) -> KVConnectorOutput:
    return KVConnectorOutput(
        finished_sending={f"__store_done_{event_idx}"},
        finished_recving=None,
        invalid_block_ids=set(),
    )


def _load_done(req_id: str) -> KVConnectorOutput:
    return KVConnectorOutput(
        finished_sending=None,
        finished_recving={req_id},
        invalid_block_ids=set(),
    )


class _MockBlocks:
    """Minimal mock for KVCacheBlocks interface."""

    def __init__(self, block_ids: list[int], num_computed: int = 0):
        self._ids = block_ids
        self.blocks = [
            [
                SimpleNamespace(block_hash="h" if i < num_computed else None)
                for i in range(len(block_ids))
            ]
        ]

    def get_block_ids(self):
        return [self._ids]


def _cache_cpu_block(manager, block_hash):
    """Pre-cache a single block hash in the CPU pool."""
    blk = manager.cpu_block_pool.get_new_blocks(1)[0]
    blk.block_hash = make_block_hash_with_group_id(block_hash, group_id=0)
    manager.cpu_block_pool.cached_block_hash_to_block.insert(blk.block_hash, blk)
    manager.cpu_block_pool.free_blocks([blk])
    return blk


def _stamp_gpu_hashes(gpu_pool, gpu_block_ids, request):
    """Stamp block_hash on GPU blocks so _prepare_eager_store_specs can
    find them. In real usage kv_cache_manager does this after compute.
    Uses _block_hash to bypass the assert-None setter guard."""
    for i, bid in enumerate(gpu_block_ids):
        if i < len(request.block_hashes):
            gpu_pool.blocks[bid]._block_hash = make_block_hash_with_group_id(
                request.block_hashes[i], group_id=0
            )


def _setup_eager_store(mgr, req, n_blocks):
    """Common setup: create GPU pool, allocate blocks, stamp hashes, register."""
    gpu_pool = _make_gpu_pool()
    mgr.bind_gpu_block_pool(gpu_pool)
    gpu_blocks = gpu_pool.get_new_blocks(n_blocks)
    gpu_ids = [b.block_id for b in gpu_blocks]
    mgr.update_state_after_alloc(req, _MockBlocks(gpu_ids), num_external_tokens=0)
    return gpu_pool, gpu_blocks, gpu_ids


# ===================================================================
# Eager store tests
# ===================================================================


class TestEagerStore:
    """Eager mode: store blocks to CPU as they are computed."""

    def test_tracks_request_and_advances_cursor(self):
        """update_state_after_alloc tracks request; build_connector_meta
        advances per-group cursor across steps."""
        mgr = _make_manager()
        gpu_pool, gpu_blocks, gpu_ids = _setup_eager_store(
            mgr, req := create_request(num_tokens=32, block_size=BLOCK_SIZE), 2
        )

        state = mgr._reqs_to_store[req.request_id]
        assert isinstance(state, RequestState)
        assert state.num_stored_blocks == [0]

        # Stamp hash on block 0, step computes 16 tokens
        _stamp_gpu_hashes(gpu_pool, gpu_ids[:1], req)
        req.num_computed_tokens = 0
        mgr.build_connector_meta(_sched_output({req.request_id: 16}))
        assert state.num_stored_blocks == [1]

        # Stamp hash on block 1, step computes 16 more
        _stamp_gpu_hashes(gpu_pool, gpu_ids[:2], req)
        req.num_computed_tokens = 16
        mgr.build_connector_meta(_sched_output({req.request_id: 16}))
        assert state.num_stored_blocks == [2]

    def test_gpu_refs_touched_during_store_released_on_completion(self):
        """GPU blocks get ref_cnt bumped during store, decremented on
        completion."""
        mgr = _make_manager()
        req = create_request(num_tokens=32, block_size=BLOCK_SIZE)
        gpu_pool, gpu_blocks, gpu_ids = _setup_eager_store(mgr, req, 2)
        _stamp_gpu_hashes(gpu_pool, gpu_ids, req)
        req.num_computed_tokens = 0

        meta = mgr.build_connector_meta(_sched_output({req.request_id: 32}))
        assert meta.store_event >= 0
        for bid in meta.store_gpu_blocks:
            assert gpu_pool.blocks[bid].ref_cnt == 2  # 1 alloc + 1 touch

        mgr.update_connector_output(_store_done(meta.store_event))
        for bid in meta.store_gpu_blocks:
            assert gpu_pool.blocks[bid].ref_cnt == 1  # touch released

    def test_preemption_with_inflight_store(self):
        """Preemption frees alloc ref (2->1), store completion frees
        touch ref (1->0). Blocks return to free pool."""
        mgr = _make_manager()
        req = create_request(num_tokens=32, block_size=BLOCK_SIZE)
        gpu_pool, gpu_blocks, gpu_ids = _setup_eager_store(mgr, req, 2)
        _stamp_gpu_hashes(gpu_pool, gpu_ids, req)
        req.num_computed_tokens = 0

        meta = mgr.build_connector_meta(_sched_output({req.request_id: 32}))
        stored_ids = list(meta.store_gpu_blocks)

        # Preemption frees alloc ref: 2 -> 1
        gpu_pool.free_blocks(gpu_blocks)
        for bid in stored_ids:
            assert gpu_pool.blocks[bid].ref_cnt == 1

        free_before = gpu_pool.get_num_free_blocks()

        # Store completes: touch ref freed: 1 -> 0
        mgr.update_connector_output(_store_done(meta.store_event))
        for bid in stored_ids:
            assert gpu_pool.blocks[bid].ref_cnt == 0
        assert gpu_pool.get_num_free_blocks() > free_before

    def test_cached_blocks_skip_store(self):
        """Blocks already in CPU cache advance the cursor without
        re-storing."""
        mgr = _make_manager()
        req = create_request(num_tokens=32, block_size=BLOCK_SIZE)
        _cache_cpu_block(mgr, req.block_hashes[0])

        gpu_pool, gpu_blocks, gpu_ids = _setup_eager_store(mgr, req, 2)
        _stamp_gpu_hashes(gpu_pool, gpu_ids, req)
        req.num_computed_tokens = 0

        meta = mgr.build_connector_meta(_sched_output({req.request_id: 32}))
        # Only block 1 stored (block 0 already cached)
        assert len(meta.store_gpu_blocks) == 1
        assert meta.store_gpu_blocks[0] == gpu_ids[1]
        assert mgr._reqs_to_store[req.request_id].num_stored_blocks == [2]

        # Second call: nothing left
        meta2 = mgr.build_connector_meta(_sched_output({req.request_id: 32}))
        assert meta2.store_gpu_blocks == []
        assert meta2.store_event == -1


# ===================================================================
# Load tests
# ===================================================================


class TestLoad:
    """Load blocks from CPU cache to GPU."""

    def test_load_touches_and_releases_cpu_gpu_refs(self):
        """CPU and GPU blocks are touched during load setup, released on
        completion."""
        mgr = _make_manager()
        gpu_pool = _make_gpu_pool()
        mgr.bind_gpu_block_pool(gpu_pool)

        req = create_request(num_tokens=16, block_size=BLOCK_SIZE)
        cpu_blk = _cache_cpu_block(mgr, req.block_hashes[0])
        assert cpu_blk.ref_cnt == 0

        gpu_blk = gpu_pool.get_new_blocks(1)[0]
        assert gpu_pool.blocks[gpu_blk.block_id].ref_cnt == 1

        mgr.update_state_after_alloc(
            req, _MockBlocks([gpu_blk.block_id]), num_external_tokens=16
        )
        assert cpu_blk.ref_cnt == 1
        assert gpu_pool.blocks[gpu_blk.block_id].ref_cnt == 2

        meta = mgr.build_connector_meta(_sched_output())
        assert meta.load_event >= 0

        mgr.update_connector_output(_load_done(req.request_id))
        assert cpu_blk.ref_cnt == 0
        assert gpu_pool.blocks[gpu_blk.block_id].ref_cnt == 1
        assert req.request_id not in mgr._reqs_to_load

    def test_request_finished_releases_load_refs_before_submission(self):
        """If request finishes before load_event is assigned, cleanup
        is immediate."""
        mgr = _make_manager()
        gpu_pool = _make_gpu_pool()
        mgr.bind_gpu_block_pool(gpu_pool)

        req = create_request(num_tokens=16, block_size=BLOCK_SIZE)
        cpu_blk = _cache_cpu_block(mgr, req.block_hashes[0])
        gpu_blk = gpu_pool.get_new_blocks(1)[0]

        mgr.update_state_after_alloc(
            req, _MockBlocks([gpu_blk.block_id]), num_external_tokens=16
        )

        is_async, _ = mgr.request_finished(req, block_ids=[gpu_blk.block_id])
        assert not is_async
        assert cpu_blk.ref_cnt == 0
        assert gpu_pool.blocks[gpu_blk.block_id].ref_cnt == 1
        assert req.request_id not in mgr._reqs_to_load

    def test_load_skips_locally_computed_blocks(self):
        """External load skips blocks already computed on GPU."""
        mgr = _make_manager()
        req = create_request(num_tokens=64, block_size=BLOCK_SIZE)

        for bh in req.block_hashes[:4]:
            _cache_cpu_block(mgr, bh)

        num_new, is_async = mgr.get_num_new_matched_tokens(req, 32)
        assert num_new == 32
        assert is_async is True

        mgr.update_state_after_alloc(
            req,
            _MockBlocks([10, 11, 12, 13], num_computed=2),
            num_external_tokens=32,
        )

        state = mgr._reqs_to_load[req.request_id]
        assert state.load_transfer.gpu_block_ids == [12, 13]
        assert len(state.load_transfer.cpu_block_ids) == 2


# ===================================================================
# Deferred cleanup tests
# ===================================================================


class TestDeferredCleanup:
    """request_finished defers cleanup when transfers are in-flight."""

    def test_store_deferred_then_cleaned_on_completion(self):
        """In-flight store: finished=True until store completes."""
        mgr = _make_manager()
        req = create_request(num_tokens=32, block_size=BLOCK_SIZE)
        rid = req.request_id

        gpu_pool, gpu_blocks, gpu_ids = _setup_eager_store(mgr, req, 2)
        _stamp_gpu_hashes(gpu_pool, gpu_ids, req)
        req.num_computed_tokens = 0
        meta = mgr.build_connector_meta(_sched_output({rid: 32}))
        assert meta.store_event >= 0

        mgr.request_finished(req, block_ids=gpu_ids)
        assert rid in mgr._reqs_to_store
        assert mgr._reqs_to_store[rid].finished is True

        mgr.update_connector_output(_store_done(meta.store_event))
        assert rid not in mgr._reqs_to_store

    def test_load_deferred_then_cleaned_on_completion(self):
        """In-flight load: finished=True until load completes."""
        mgr = _make_manager()
        gpu_pool = _make_gpu_pool()
        mgr.bind_gpu_block_pool(gpu_pool)

        req = create_request(num_tokens=16, block_size=BLOCK_SIZE)
        rid = req.request_id
        cpu_blk = _cache_cpu_block(mgr, req.block_hashes[0])
        gpu_blk = gpu_pool.get_new_blocks(1)[0]

        mgr.update_state_after_alloc(
            req, _MockBlocks([gpu_blk.block_id]), num_external_tokens=16
        )
        meta = mgr.build_connector_meta(_sched_output())
        assert meta.load_event >= 0

        mgr.request_finished(req, block_ids=[gpu_blk.block_id])
        assert rid in mgr._reqs_to_load
        assert mgr._reqs_to_load[rid].finished is True

        mgr.update_connector_output(_load_done(rid))
        assert rid not in mgr._reqs_to_load
        assert cpu_blk.ref_cnt == 0
        assert gpu_pool.blocks[gpu_blk.block_id].ref_cnt == 1

    def test_no_inflight_cleans_immediately(self):
        """No in-flight transfers: cleanup is immediate."""
        mgr = _make_manager()
        req = create_request(num_tokens=32, block_size=BLOCK_SIZE)

        mgr.update_state_after_alloc(req, _MockBlocks([0, 1]), num_external_tokens=0)
        assert req.request_id in mgr._reqs_to_store

        mgr.request_finished(req, block_ids=[0, 1])
        assert req.request_id not in mgr._reqs_to_store


# ===================================================================
# Lazy store tests
# ===================================================================


class TestLazyStore:
    """Lazy mode: store GPU eviction candidates to CPU proactively."""

    def test_lazy_does_not_track_reqs_to_store(self):
        mgr = _make_manager(lazy=True)
        req = create_request(num_tokens=32, block_size=BLOCK_SIZE)
        mgr.update_state_after_alloc(req, _MockBlocks([0, 1]), num_external_tokens=0)
        assert req.request_id not in mgr._reqs_to_store

    def test_lazy_touches_and_releases_gpu_blocks(self):
        """Eviction candidates touched during store, freed on completion."""
        mgr = _make_manager(lazy=True)
        gpu_pool = _make_gpu_pool(num_blocks=5)
        mgr.bind_gpu_block_pool(gpu_pool)

        blocks = gpu_pool.get_new_blocks(4)
        hashed = blocks[:2]
        unhashed = blocks[2:]
        for i, b in enumerate(hashed):
            b.block_hash = make_block_hash_with_group_id(
                BlockHash(b"h" + str(i).encode()), group_id=0
            )
            gpu_pool.cached_block_hash_to_block.insert(b.block_hash, b)

        gpu_pool.free_blocks(hashed)
        gpu_pool.free_blocks(unhashed)
        gpu_ids = [b.block_id for b in hashed]
        assert all(gpu_pool.blocks[bid].ref_cnt == 0 for bid in gpu_ids)

        meta = mgr.build_connector_meta(_sched_output({"x": 32}))
        assert meta.store_event >= 0
        for bid in meta.store_gpu_blocks:
            assert gpu_pool.blocks[bid].ref_cnt > 0

        mgr.update_connector_output(_store_done(meta.store_event))
        for bid in meta.store_gpu_blocks:
            assert gpu_pool.blocks[bid].ref_cnt == 0
