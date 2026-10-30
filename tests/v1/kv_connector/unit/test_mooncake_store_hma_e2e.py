# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end save->lookup test for MooncakeStoreConnector on a hybrid
(SWA + Full) attention config, using a dict-backed mock store."""

import dataclasses
import sys
import threading
import types
from unittest.mock import MagicMock, patch

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store import (
    worker as mooncake_store_worker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.coordinator import (  # noqa: E501
    MooncakeStoreCoordinator,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (
    BoundaryPut,
    ChunkedTokenDatabase,
    KeyMetadata,
    LoadSpec,
    RankLocalStoreLayout,
    ReqMeta,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.worker import (  # noqa: E501
    KVCacheStoreRecvingThread,
    KVCacheStoreSendingThread,
)
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    GroupGeometry,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVCacheTensor,
    MambaSpec,
    SlidingWindowSpec,
)

from .utils import make_two_group_large_page_config


class _DictStore:
    """In-memory MooncakeDistributedStore stand-in."""

    def __init__(self):
        self._data: dict[str, bytes] = {}
        self.put_addrs: dict[str, list[int]] = {}

    def setup(self, *_args, **_kwargs):
        return 0

    def register_buffer(self, addr, length):
        return 0

    def batch_is_exist(self, keys):
        return [1 if k in self._data else 0 for k in keys]

    def batch_put_from_multi_buffers(self, keys, addrs, sizes, *_args, **_kwargs):
        for k, a in zip(keys, addrs):
            self._data[k] = b"x"
            self.put_addrs[k] = a
        return [0] * len(keys)

    def batch_get_into_multi_buffers(self, keys, addrs, sizes, *_args, **_kwargs):
        return [0 if k in self._data else -1 for k in keys]


def _minimal_vllm_config(cache_block_size=16):
    cfg = MagicMock()
    cfg.cache_config.block_size = cache_block_size
    cfg.cache_config.num_gpu_blocks = 4
    cfg.cache_config.enable_prefix_caching = True
    cfg.parallel_config.prefill_context_parallel_size = 1
    cfg.parallel_config.decode_context_parallel_size = 1
    cfg.parallel_config.pipeline_parallel_size = 1
    cfg.parallel_config.world_size = 1
    cfg.parallel_config.rank = 0
    cfg.parallel_config.data_parallel_rank_local = 0
    cfg.parallel_config.data_parallel_size_local = 1
    cfg.kv_transfer_config.kv_role = "kv_both"
    cfg.kv_transfer_config.kv_connector_extra_config = {}
    cfg.kv_events_config = None
    cfg.model_config.model = "/tmp/m"
    cfg.model_config.use_mla = False
    cfg.model_config.get_num_layers.return_value = 2
    cfg.model_config.get_total_num_kv_heads.return_value = 8
    cfg.model_config.max_model_len = 4096
    cfg.scheduler_config.max_num_batched_tokens = 8192
    # Without this, MagicMock's truthy use_eagle() triggers the coordinator's
    # "use_eagle && nothing annotated → flag all groups" fallback.
    cfg.speculative_config = None
    return cfg


def _build_worker_with_dict_store(vllm_config, kv_cache_config, store):
    """Build a MooncakeStoreWorker patching all distributed dependencies."""
    fake_mooncake_store = types.ModuleType("mooncake.store")
    fake_mooncake_store.MooncakeDistributedStore = lambda: store  # type: ignore[attr-defined]
    fake_mooncake_store.ReplicateConfig = MagicMock  # type: ignore[attr-defined]
    with (
        patch.dict(sys.modules, {"mooncake.store": fake_mooncake_store}),
        patch.object(mooncake_store_worker, "MooncakeStoreConfig") as MCfg,
        patch.object(mooncake_store_worker, "LookupKeyServer"),
    ):
        sc = MCfg.load_from_config.return_value
        sc.metadata_server = ""
        sc.global_segment_size = 1 << 20
        sc.local_buffer_size = 1 << 20
        sc.protocol = "tcp"
        sc.device_name = ""
        sc.master_server_address = ""
        sc.mode = "embedded"
        sc.enable_offload = False
        with (
            patch(
                "vllm.distributed.kv_transfer.kv_connector.v1.mooncake"
                ".store.worker.get_tensor_model_parallel_rank",
                return_value=0,
            ),
            patch(
                "vllm.distributed.kv_transfer.kv_connector.v1.mooncake"
                ".store.worker.get_tensor_model_parallel_world_size",
                return_value=1,
            ),
            patch(
                "vllm.distributed.kv_transfer.kv_connector.v1.mooncake"
                ".store.worker.get_pcp_group"
            ) as mock_pcp,
            patch(
                "vllm.distributed.kv_transfer.kv_connector.v1.mooncake"
                ".store.worker.get_dcp_group"
            ) as mock_dcp,
            patch(
                "vllm.distributed.kv_transfer.kv_connector.v1.mooncake"
                ".store.worker.get_ip",
                return_value="127.0.0.1",
            ),
        ):
            mock_pcp.return_value.world_size = 1
            mock_dcp.return_value.world_size = 1
            worker = mooncake_store_worker.MooncakeStoreWorker(
                vllm_config, kv_cache_config=kv_cache_config
            )
    return worker


# Large-page pool geometry for the worker-level tests. One backing pool of
# NUM_PAGES pages of PAGE_BYTES holds every group's blocks, as in production.
# The full-attention group packs one 16-token block as 2 kernel rows of
# 8 tokens (1 layer x 8 heads x 8 tokens x 64 x fp16 = 8192 B per row).
PAGE_BYTES = 34816
NUM_PAGES = 6
FULL_ROWS_PER_BLOCK, FULL_ROW_BYTES = 2, 8192


def _lp_group(
    layer_name: str,
    spec: KVCacheSpec,
    *,
    rows_per_block: int = 1,
    row_bytes: int,
    num_pages: int = NUM_PAGES,
) -> KVCacheGroupSpec:
    return KVCacheGroupSpec(
        [layer_name],
        spec,
        num_blocks=num_pages,
        kernel_rows_per_block=rows_per_block,
        row_bytes=row_bytes,
    )


def _lp_config(
    groups: list[KVCacheGroupSpec],
    *,
    block_size: int,
    num_pages: int = NUM_PAGES,
    page_bytes: int = PAGE_BYTES,
) -> KVCacheConfig:
    """Hand-built equivalent of the planner's large-page KVCacheConfig."""
    size = num_pages * page_bytes
    tensors = [
        KVCacheTensor(
            size=size,
            layers=[name],
            layer_stride=group.row_bytes,
            block_stride=group.row_bytes,
            offset=i * (group.row_bytes // len(group.layer_names)),
        )
        for group in groups
        for i, name in enumerate(group.layer_names)
    ]
    return KVCacheConfig(
        num_pages=num_pages,
        kv_cache_tensors=tensors,
        kv_cache_groups=groups,
        large_page_bytes=page_bytes,
        block_size=block_size,
    )


def _shared_pool_views(
    buf: torch.Tensor, group: KVCacheGroupSpec, dtype: torch.dtype
) -> dict[str, torch.Tensor]:
    """One strided view per layer of the group over the shared pool buffer.

    Each row of the group's row grid packs all of its layers, so layer i's
    view strides over the rows at a per-layer offset.
    """
    element_size = torch.empty((), dtype=dtype).element_size()
    row_elems = group.row_bytes // element_size
    per_layer = row_elems // len(group.layer_names)
    rows = buf.numel() * buf.element_size() // group.row_bytes
    typed = buf.view(dtype)
    return {
        name: torch.as_strided(
            typed, (rows, per_layer), (row_elems, 1), storage_offset=i * per_layer
        )
        for i, name in enumerate(group.layer_names)
    }


def _rank_local_db(
    metadata: KeyMetadata,
    spec: KVCacheSpec,
    *,
    base_addr: int = 0,
    rows_per_block: int = 2,
    row_bytes: int = 256,
) -> ChunkedTokenDatabase:
    """Token database with a rank-local layout over a large-page pool.

    Block ``b`` is at ``base_addr + b * rows_per_block * row_bytes``. The base
    address is set directly because the mock stores record addresses without
    dereferencing them.
    """
    layout = RankLocalStoreLayout(
        metadata, spec.block_size, geometry=GroupGeometry(rows_per_block, row_bytes)
    )
    layout.base_addr = base_addr
    return ChunkedTokenDatabase(metadata, spec.block_size, store_layout=layout)


def _fake_thread_init(*args, **kwargs):
    """Mock thread that sets all threading.Event args so waits don't block."""
    for v in list(args) + list(kwargs.values()):
        if isinstance(v, threading.Event):
            v.set()
    m = MagicMock()
    m.start = lambda: None
    return m


def test_e2e_swa_plus_full_save_then_lookup_hits():
    """E2E: build a SWA+Full hybrid worker, save all blocks via the sending
    thread (synchronously), then verify lookup returns the full hit length.
    Also verify that evicting SWA's early blocks (outside its window) still
    allows a full hit because the window covers the tail.

    The planner-built groups have different block strides (8704 B and
    4096 B), so each group's block addresses use its own stride.
    """
    cfg = make_two_group_large_page_config()
    # Retain SWA checkpoints densely: the CacheConfig default (0) retains only
    # semantic boundaries, so a one-block window would store nothing; this
    # test exercises addressing and hit semantics, not retention policy.
    cfg = dataclasses.replace(cfg, prefix_cache_retention_interval=None)
    vllm_config = _minimal_vllm_config(cache_block_size=16)
    store = _DictStore()

    worker = _build_worker_with_dict_store(vllm_config, cfg, store)
    worker.tp_size = 1
    worker.pp_size = 1
    worker.num_kv_head = 4

    # Register kv_caches using mocked thread classes so register_kv_caches
    # doesn't try to start real background threads (which set ready_event).
    # Both groups' views stride into one shared backing pool, as under the
    # large-page allocator.
    buf = torch.zeros(cfg.num_pages * cfg.large_page_bytes, dtype=torch.int8)
    kv_caches = {}
    for group in cfg.kv_cache_groups:
        kv_caches.update(_shared_pool_views(buf, group, torch.float16))

    with (
        patch.object(
            mooncake_store_worker,
            "KVCacheStoreSendingThread",
            side_effect=_fake_thread_init,
        ),
        patch.object(
            mooncake_store_worker,
            "KVCacheStoreRecvingThread",
            side_effect=_fake_thread_init,
        ),
    ):
        worker.register_kv_caches(kv_caches)

    # Now build a real sending thread (no .start()) over the worker's token_dbs
    # and the dict-backed store, so _handle_request runs synchronously.
    ready = threading.Event()
    send_thread = KVCacheStoreSendingThread(
        store=store,
        token_databases=worker.token_dbs,
        block_size=worker.scheduler_block_size,
        coord=worker.coord,
        tp_rank=worker.tp_rank,
        group_put_steps=worker._group_tp_replication_factors,
        kv_role=worker.kv_role,
        ready_event=ready,
        enable_kv_event=False,
    )

    hs = [BlockHash(bytes([i + 1]) * 4) for i in range(4)]
    # 1-based: block 0 is the reserved null block and the save skips it.
    save_req = ReqMeta(
        req_id="r0",
        token_len_chunk=64,
        # Block 0 is reserved as NULL_BLOCK_ID by the production block pool.
        block_ids=([1, 2, 3, 4], [1, 2, 3, 4]),
        block_hashes=hs,
        can_save=True,
        store_job_id=1,
    )
    # add_request also queues the job, so task_done() doesn't underflow.
    send_thread.add_request(save_req)
    send_thread._handle_request(send_thread.request_queue.get())

    # Every stored segment of block b starts at b * block_bytes of its group:
    # one 8704 B row for full attention, four 1024 B rows for SWA.
    base = buf.untyped_storage().data_ptr()
    for g_idx, block_bytes in enumerate((8704, 4096)):
        for i, block_id in enumerate([1, 2, 3, 4]):
            key = worker.token_dbs[g_idx].key_for(hs[i])
            assert store.put_addrs[key] == [base + block_id * block_bytes]

    # Point worker.store at the dict store (the worker constructor captured
    # the MagicMock; replace with the real dict store for lookup).
    worker.store = store

    # Both groups stored all 4 blocks -> full hit.
    assert worker.lookup(num_tokens=65, block_hashes=hs) == 64
    # Exact-multiple prompt: the full hit is re-derived one block lower,
    # where both groups' stored blocks still cover the SWA window.
    assert worker.lookup(num_tokens=64, block_hashes=hs) == 48

    # Evict SWA's first three blocks (outside its window of 16 tokens = 1
    # block).
    swa_keys_outside_window = [
        k
        for k in list(store._data.keys())
        if "@group:1" in k and any(("@" + h.hex()) in k for h in hs[:3])
    ]
    for k in swa_keys_outside_window:
        del store._data[k]

    # SWA window=16 -> only the last block must be present in the SWA group.
    # Full has all 4. Coordinator should still return 64.
    assert worker.lookup(num_tokens=65, block_hashes=hs) == 64
    # Exact-multiple prompt after eviction: the boundary one block lower
    # needs SWA block 2, which is gone — no usable stored boundary remains
    # (the pre-fix arithmetic clamp would have returned 48 and livelocked
    # on load failure -> recompute -> same lookup).
    assert worker.lookup(num_tokens=64, block_hashes=hs) == 0


def test_recv_skips_swa_blocks_before_window():
    """Producer stored every block for both groups; consumer must only fetch
    SWA blocks within the sliding window, not the head."""
    full = FullAttentionSpec(block_size=16, num_kv_heads=8, head_size=64, dtype=None)
    # sliding_window=32, block_size=16 → 2 contiguous blocks within window.
    swa = SlidingWindowSpec(
        block_size=16,
        num_kv_heads=8,
        head_size=64,
        dtype=None,
        sliding_window=32,
    )
    groups = [
        KVCacheGroupSpec(["L0"], full),
        KVCacheGroupSpec(["L1"], swa),
    ]
    md0 = KeyMetadata("m", 0, 0, 0, 0, group_id=0)
    md1 = KeyMetadata("m", 0, 0, 0, 0, group_id=1)
    # The SWA group has its own block stride: one 384 B row per block.
    db_full = _rank_local_db(md0, full, rows_per_block=2, row_bytes=512)
    db_swa = _rank_local_db(
        md1, swa, base_addr=1 << 20, rows_per_block=1, row_bytes=384
    )

    requested: list[tuple[str, list[int]]] = []

    class _CapturingStore:
        def batch_get_into_multi_buffers(self, keys, addrs, sizes):
            requested.extend(zip(keys, addrs, strict=True))
            return [0] * len(keys)

    ready = threading.Event()
    coord = MooncakeStoreCoordinator(groups, scheduler_block_size=16, block_size=16)
    recv = KVCacheStoreRecvingThread(
        store=_CapturingStore(),
        token_databases=[db_full, db_swa],
        block_size=16,
        tp_rank=0,
        ready_event=ready,
        coord=coord,
        group_kv_cache_ids=(0, 1),
    )

    hs = [BlockHash(bytes([i + 1]) * 4) for i in range(4)]
    req = ReqMeta(
        req_id="r0",
        token_len_chunk=64,
        block_ids=([0, 1, 2, 3], [0, 1, 2, 3]),
        block_hashes=hs,
        load_spec=LoadSpec(
            vllm_cached_tokens=0, kvpool_cached_tokens=64, can_load=True, token_len=64
        ),
    )
    recv.request_queue.put(req)
    recv._handle_request(recv.request_queue.get())

    full_keys = [k for k, _ in requested if "@group:0" in k]
    swa_keys = [k for k, _ in requested if "@group:1" in k]
    # Full attention: load every block (4).
    assert len(full_keys) == 4
    # SWA: only the 2 tail-window blocks (hashes hs[2], hs[3]).
    assert len(swa_keys) == 2
    swa_hashes = {k.rsplit("@", 1)[-1] for k in swa_keys}
    assert swa_hashes == {hs[2].hex(), hs[3].hex()}
    # The two tail-window loads target blocks 2 and 3.
    swa_addrs = [a for k, a in requested if "@group:1" in k]
    assert swa_addrs == [[(1 << 20) + 2 * 384], [(1 << 20) + 3 * 384]]


def test_offload_syncs_event_before_put():
    """An offload-carrying meta synchronizes its CoW-fence event before the
    store put reads the blocks, then completes in one pass and drains the
    completion counter."""
    full = FullAttentionSpec(block_size=16, num_kv_heads=8, head_size=64, dtype=None)
    mamba = MambaSpec(
        block_size=16,
        shapes=((1, 1),),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
    )
    groups = [
        KVCacheGroupSpec(["L0"], full),
        KVCacheGroupSpec(["L1"], mamba),
    ]
    coord = MooncakeStoreCoordinator(groups, scheduler_block_size=16, block_size=16)
    event = MagicMock()

    class _FencedStore(_DictStore):
        def batch_put_from_multi_buffers(self, keys, addrs, sizes, *a, **k):
            assert event.synchronize.called, "put must run after the event sync"
            return super().batch_put_from_multi_buffers(keys, addrs, sizes, *a, **k)

    store = _FencedStore()
    token_dbs = [
        _rank_local_db(
            KeyMetadata("m", 0, 0, 0, 0, group_id=g_idx),
            spec,
            base_addr=g_idx * 10_000,
        )
        for g_idx, spec in enumerate([full, mamba])
    ]

    send = KVCacheStoreSendingThread(
        store=store,
        coord=coord,
        token_databases=token_dbs,
        block_size=16,
        tp_rank=0,
        group_put_steps=[1, 1],
        kv_role="kv_both",
        ready_event=threading.Event(),
        replicate_config=MagicMock(),
    )

    hs = [BlockHash(bytes([i + 1]) * 4) for i in range(2)]
    req = ReqMeta(
        req_id="r1",
        token_len_chunk=0,
        block_ids=([1], [2]),
        block_hashes=hs,
        can_save=True,
        num_prompt_tokens=17,
        store_job_id=1,
        boundary_puts=[BoundaryPut(1, 7, 16)],
    )
    req.current_event = event

    send.add_request(req)
    send._handle_request(send.request_queue.get())
    assert send.request_queue.qsize() == 0
    assert store._data
    assert send.stored_requests["r1"] == set()
    event.synchronize.assert_called_once()


def test_worker_setup_tolerates_finer_scratch_group():
    """Setup and lookup must tolerate a non-prefix-cacheable scratch group.

    Regression: GLM-5.3-Flash carries a kpool-tail scratch group whose block
    size (``index_kpool`` tokens) is finer than the block size. Scratch groups
    never participate in store/load/lookup, so setup must skip them and lookup
    must still hit stored boundaries on the participating groups.
    """
    full = FullAttentionSpec(block_size=16, num_kv_heads=8, head_size=64, dtype=None)
    mamba = MambaSpec(
        block_size=16,
        shapes=((1, 1),),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
    )
    scratch = CircularBufferSpec(
        block_size=4,
        num_kv_heads=2,
        head_size=64,
        head_size_v=0,
        dtype=torch.bfloat16,
    )
    cfg = _lp_config(
        [
            _lp_group("L0", full, row_bytes=8 * 16 * 64 * 2),
            _lp_group("L1", mamba, row_bytes=4),
            _lp_group("L2", scratch, row_bytes=4 * 2 * 64 * 2),
        ],
        block_size=16,
    )
    vllm_config = _minimal_vllm_config(cache_block_size=16)
    store = _DictStore()

    worker = _build_worker_with_dict_store(vllm_config, cfg, store)
    worker.tp_size = 1
    worker.pp_size = 1
    worker.num_kv_head = 8
    # Only the participating full-attention and Mamba groups are registered.
    assert len(worker.token_dbs) == 2

    send_thread = KVCacheStoreSendingThread(
        store=store,
        token_databases=worker.token_dbs,
        block_size=worker.scheduler_block_size,
        coord=worker.coord,
        tp_rank=0,
        group_put_steps=[1, 1],
        kv_role="kv_both",
        ready_event=threading.Event(),
        replicate_config=MagicMock(),
        group_participates=[True, True],
    )

    # Persist both groups' first block (keyed by hs[0]).
    hs = [BlockHash(bytes([i + 1]) * 8) for i in range(2)]
    req = ReqMeta(
        req_id="r0",
        token_len_chunk=0,
        block_ids=([1], [2]),
        block_hashes=hs,
        can_save=True,
        num_prompt_tokens=17,
        boundary_puts=[BoundaryPut(0, 1, 16), BoundaryPut(1, 7, 16)],
    )
    send_thread._maybe_offload_boundary_states(req)

    worker.store = store

    # A 17-token prompt sharing the prefix must hit the first block.
    assert worker.lookup(num_tokens=17, block_hashes=hs) == 16
    # The scratch group's namespace never enters the store.
    assert not any("@group:2" in key for key in store._data)


def test_ring_scratch_group_is_never_stored_and_does_not_block_hits():
    """A per-request ring group (CircularBufferSpec, capacity as block_size)
    sits beside the paged group: worker setup tolerates its block size (the
    DeepSeek-V4.1 compressor ring is 8 rows), saving a request stores paged
    blocks only, and lookup hits on the paged group alone."""
    full = FullAttentionSpec(block_size=16, num_kv_heads=8, head_size=64, dtype=None)
    ring = CircularBufferSpec(
        block_size=8, num_kv_heads=1, head_size=64, head_size_v=0, dtype=torch.uint8
    )
    cfg = _lp_config(
        [
            _lp_group(
                "L0", full, rows_per_block=FULL_ROWS_PER_BLOCK, row_bytes=FULL_ROW_BYTES
            ),
            # 8 tokens x 1 head x 64 x uint8 per kernel row.
            _lp_group("L1", ring, row_bytes=512),
        ],
        block_size=16,
    )
    vllm_config = _minimal_vllm_config(cache_block_size=16)
    store = _DictStore()

    worker = _build_worker_with_dict_store(vllm_config, cfg, store)
    worker.tp_size = 1
    worker.pp_size = 1
    worker.num_kv_head = 8
    assert len(worker.token_dbs) == 1

    buf = torch.zeros(NUM_PAGES * PAGE_BYTES, dtype=torch.int8)
    kv_caches = _shared_pool_views(buf, cfg.kv_cache_groups[0], torch.float16)
    kv_caches.update(_shared_pool_views(buf, cfg.kv_cache_groups[1], torch.uint8))

    with (
        patch.object(
            mooncake_store_worker,
            "KVCacheStoreSendingThread",
            side_effect=_fake_thread_init,
        ),
        patch.object(
            mooncake_store_worker,
            "KVCacheStoreRecvingThread",
            side_effect=_fake_thread_init,
        ),
    ):
        worker.register_kv_caches(kv_caches)

    send_thread = KVCacheStoreSendingThread(
        store=store,
        token_databases=worker.token_dbs,
        block_size=worker.scheduler_block_size,
        coord=worker.coord,
        tp_rank=worker.tp_rank,
        group_put_steps=worker._group_tp_replication_factors,
        kv_role=worker.kv_role,
        ready_event=threading.Event(),
        enable_kv_event=False,
        group_participates=[True],
    )
    hs = [BlockHash(bytes([i + 1]) * 4) for i in range(4)]
    save_req = ReqMeta(
        req_id="r0",
        token_len_chunk=64,
        # Scheduler metadata is projected to the participating store groups.
        block_ids=([1, 2, 3, 4],),
        block_hashes=hs,
        can_save=True,
        store_job_id=1,
    )
    send_thread.add_request(save_req)
    send_thread._handle_request(send_thread.request_queue.get())
    worker.store = store

    assert worker.lookup(num_tokens=65, block_hashes=hs) == 64
    assert not any("@group:1" in key for key in store._data)
