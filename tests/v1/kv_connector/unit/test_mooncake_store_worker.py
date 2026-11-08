# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import itertools
import json
import logging
import math
import queue
import sys
import threading
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.distributed import mooncake_store
from vllm.distributed.kv_events import KVEventAggregator
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake import (
    rdma_utils,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store import (
    worker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store import (
    worker as mooncake_store_worker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (
    BlobBlockHashes,
    BoundaryPut,
    ChunkedTokenDatabase,
    KeyMetadata,
    LBHNCStoreLayout,
    LBNHCStoreLayout,
    LoadSpec,
    PoolKey,
    RankLocalStoreLayout,
    ReqMeta,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.metrics import (
    MooncakeStoreConnectorStats,
)
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
from vllm.v1.core.kv_cache_utils import BlockHash, maybe_convert_block_hash
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    GroupGeometry,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    RSWASpec,
)
from vllm.v1.kv_cache_layout import KVCacheLayout

_TP_SHARED_NAMESPACE = "@store_tp:4@store_pp:1@store_format:tp_shared_lbhnc"
_TP_SHARED_LBNHC_NAMESPACE = "@store_tp:4@store_pp:1@store_format:tp_shared_lbnhc"


def _tp_shared_prefix(tp_rank: int) -> str:
    return (
        f"test-model{_TP_SHARED_NAMESPACE}@tp_rank:{tp_rank}"
        "@pcp0@dcp0@pp_rank:0@group:0"
    )


class _RecordingBlockHashes:
    def __init__(self, values: list[bytes]):
        self.values = values
        self.accessed: list[int] = []

    def __len__(self):
        return len(self.values)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(len(self)))]
        if idx < 0:
            idx += len(self.values)
        if not 0 <= idx < len(self.values):
            raise IndexError(idx)
        self.accessed.append(idx)
        return self.values[idx]


def _default_send_coord() -> mooncake_store_worker.MooncakeStoreCoordinator:
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec

    spec = FullAttentionSpec(block_size=16, num_kv_heads=8, head_size=64, dtype=None)
    return mooncake_store_worker.MooncakeStoreCoordinator(
        [KVCacheGroupSpec(["layer0"], spec)],
        scheduler_block_size=16,
        block_size=16,
    )


def _make_rank_local_db(
    metadata: KeyMetadata,
    block_size: int,
    *,
    base_addr: int = 0,
    block_bytes: int = 256,
) -> ChunkedTokenDatabase:
    """Rank-local token db: manager block ``b`` occupies
    ``[base_addr + b * block_bytes, +block_bytes)``."""
    layout = RankLocalStoreLayout(
        metadata, block_size, geometry=GroupGeometry(2, block_bytes // 2)
    )
    layout.base_addr = base_addr
    return ChunkedTokenDatabase(metadata, block_size, store_layout=layout)


def _make_tp_shared_db(
    tp_rank: int = 0,
) -> tuple[ChunkedTokenDatabase, torch.Tensor]:
    num_blocks = 2
    local_heads = 4
    block_size = 16
    content_size = 4
    strides = (
        local_heads * block_size * content_size,
        block_size * content_size,
        content_size,
        1,
    )
    tensor = torch.empty_strided(
        (num_blocks, local_heads, block_size, content_size),
        strides,
        dtype=torch.float16,
    )
    metadata = KeyMetadata(
        "test-model",
        tp_rank,
        0,
        0,
        0,
        store_namespace=_TP_SHARED_NAMESPACE,
    )
    # fp16: one kernel row per block, row_bytes == stride(0) * element_size.
    row_bytes = strides[0] * tensor.element_size()
    layout = LBHNCStoreLayout(
        metadata,
        block_size,
        local_tp_size=2,
        store_tp_size=4,
        tp_rank=tp_rank,
        num_kv_heads=8,
        geometry=GroupGeometry(1, row_bytes),
    )
    db = ChunkedTokenDatabase(metadata, block_size, store_layout=layout)
    layout.register_kv_caches([tensor])
    return db, tensor


def _make_store_sending_thread(
    store: MagicMock,
    *,
    coord: mooncake_store_worker.MooncakeStoreCoordinator | None = None,
    token_databases: list[ChunkedTokenDatabase] | None = None,
    block_size: int = 16,
    tp_rank: int = 0,
    put_step: int = 1,
    replicate_config: object | None = None,
    enable_group_semantics: bool = False,
    supports_group_ids: bool = False,
) -> mooncake_store_worker.KVCacheStoreSendingThread:
    if coord is None:
        coord = _default_send_coord()
    if token_databases is None:
        db = _make_rank_local_db(
            KeyMetadata("test-model", 0, 0, 0, 0), 16, base_addr=0x1000
        )
        token_databases = [db]
    thread = mooncake_store_worker.KVCacheStoreSendingThread(
        store=store,
        token_databases=token_databases,
        block_size=block_size,
        coord=coord,
        tp_rank=tp_rank,
        group_put_steps=[put_step] * len(token_databases),
        kv_role="kv_producer",
        ready_event=threading.Event(),
        replicate_config=replicate_config,
        enable_group_semantics=enable_group_semantics,
        supports_group_ids=supports_group_ids,
    )
    thread.request_queue.task_done = MagicMock()
    return thread


def _make_store_recving_thread(
    store: MagicMock,
    *,
    tp_rank: int = 0,
    disk_offload_buffer_budget_bytes: int | None = None,
    is_hma_required: bool = False,
) -> mooncake_store_worker.KVCacheStoreRecvingThread:
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec

    token_database = _make_rank_local_db(
        KeyMetadata("test-model", 0, 0, 0, 0), 16, base_addr=0x1000
    )
    spec = FullAttentionSpec(block_size=16, num_kv_heads=8, head_size=64, dtype=None)
    coord = mooncake_store_worker.MooncakeStoreCoordinator(
        [KVCacheGroupSpec(["layer0"], spec)],
        scheduler_block_size=16,
        block_size=16,
    )
    thread = mooncake_store_worker.KVCacheStoreRecvingThread(
        store=store,
        token_databases=[token_database],
        block_size=16,
        tp_rank=tp_rank,
        ready_event=threading.Event(),
        coord=coord,
        disk_offload_buffer_budget_bytes=disk_offload_buffer_budget_bytes,
        is_hma_required=is_hma_required,
        group_kv_cache_ids=(0,),
    )
    thread.request_queue.task_done = MagicMock()
    return thread


def _make_load_req(
    req_id: str,
    block_hashes: list[bytes],
    *,
    token_len: int,
    vllm_cached_tokens: int = 0,
) -> ReqMeta:
    return ReqMeta(
        req_id=req_id,
        token_len_chunk=token_len,
        block_ids=(list(range(len(block_hashes))),),
        block_hashes=block_hashes,
        load_spec=LoadSpec(
            vllm_cached_tokens=vllm_cached_tokens,
            kvpool_cached_tokens=token_len,
            can_load=True,
            token_len=token_len,
        ),
    )


_TEST_SAVE_SEQ = itertools.count(1)


def _run_store_req(thread, req_meta: ReqMeta) -> None:
    """Register, enqueue and run a store job the way the worker does."""
    if req_meta.store_job_id is None:
        req_meta.store_job_id = next(_TEST_SAVE_SEQ)
    thread.add_request(req_meta)
    thread._handle_request(thread.request_queue.get())


def _make_store_req(req_id: str, block_hashes: list[bytes]) -> ReqMeta:
    # 1-based: block 0 is the reserved null block and the save skips it.
    return ReqMeta(
        req_id=req_id,
        token_len_chunk=32,
        block_ids=([1, 2],),
        block_hashes=block_hashes,
        can_save=True,
    )


def _make_multi_group_store_req(req_id: str, block_hashes: list[bytes]) -> ReqMeta:
    return ReqMeta(
        req_id=req_id,
        token_len_chunk=32,
        block_ids=([1, 2], [3, 4]),
        block_hashes=block_hashes,
        can_save=True,
    )


_DISK_OFFLOAD_SINGLE_KEY_BYTES = worker._estimate_disk_offload_staging_bytes([256])
_DISK_OFFLOAD_USABLE_BUDGET_RATIO = 0.9
_DISK_OFFLOAD_BUDGET_FOR_THREE_KEYS = 4 * _DISK_OFFLOAD_SINGLE_KEY_BYTES
_DISK_OFFLOAD_BUDGET_FOR_SPLIT = math.ceil(
    2 * _DISK_OFFLOAD_SINGLE_KEY_BYTES / _DISK_OFFLOAD_USABLE_BUDGET_RATIO
)  # Allows two 256-byte chunks but not the third.
_DISK_OFFLOAD_BUDGET_TOO_SMALL = (
    _DISK_OFFLOAD_SINGLE_KEY_BYTES - 1
)  # Smaller than a single 256-byte chunk.


class _FakeKVTransferConfig:
    def __init__(
        self,
        *,
        kv_role: str = "kv_both",
        extra_config: dict[str, object] | None = None,
    ) -> None:
        self.kv_role = kv_role
        self.kv_connector_extra_config = extra_config or {}

    def get_from_extra_config(self, key: str, default: object) -> object:
        return self.kv_connector_extra_config.get(key, default)


class _FakeModelConfig:
    model = "test-model"
    use_mla = False
    enable_sleep_mode = False
    enable_cumem_allocator = False

    def get_num_layers(self, parallel_config) -> int:
        return 1

    def get_total_num_kv_heads(self) -> int:
        return 1


def _make_vllm_config(
    *,
    extra_config: dict[str, object] | None = None,
    rank: int = 0,
    decode_context_parallel_size: int = 1,
    kv_role: str = "kv_both",
    pipeline_parallel_size: int = 1,
    kv_cache_layout: KVCacheLayout = KVCacheLayout.LBHNC,
    disable_hybrid_kv_cache_manager: bool = True,
) -> SimpleNamespace:
    cache_config = SimpleNamespace(block_size=16, num_gpu_blocks=10)
    cache_config.get_resolved_kv_cache_layout = lambda: kv_cache_layout
    return SimpleNamespace(
        model_config=_FakeModelConfig(),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=pipeline_parallel_size,
            rank=rank,
            data_parallel_index=0,
            decode_context_parallel_size=decode_context_parallel_size,
            prefill_context_parallel_size=1,
        ),
        kv_transfer_config=_FakeKVTransferConfig(
            kv_role=kv_role, extra_config=extra_config
        ),
        cache_config=cache_config,
        scheduler_config=SimpleNamespace(
            disable_hybrid_kv_cache_manager=disable_hybrid_kv_cache_manager
        ),
        kv_events_config=SimpleNamespace(enable_kv_cache_events=False),
        speculative_config=None,
    )


def _make_kv_cache_config(
    *,
    block_size: int = 16,
    prefix_cache_retention_interval: int | None = 0,
    dcp_size: int = 1,
) -> object:
    """Minimal single-group large-page KVCacheConfig for topology tests.

    ``block_size`` is the per-rank spec block size; DCP-sharded specs hold
    B // dcp tokens, so the hash block size is ``block_size * dcp_size``.
    """
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheConfig,
        KVCacheGroupSpec,
    )

    spec = FullAttentionSpec(
        block_size=block_size, num_kv_heads=8, head_size=64, dtype=None
    )
    return KVCacheConfig(
        num_pages=10,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer0"],
                spec,
                num_blocks=10,
                blocks_per_page=1,
                kernel_rows_per_block=2,
                row_bytes=128,
            )
        ],
        prefix_cache_retention_interval=prefix_cache_retention_interval,
        block_size=block_size * dcp_size,
        large_page_bytes=256,
    )


def _make_qsa_hybrid_kv_cache_config():
    def group(layer: str, spec, **geometry) -> KVCacheGroupSpec:
        return KVCacheGroupSpec([layer], spec, num_blocks=10, **geometry)

    return KVCacheConfig(
        num_pages=10,
        kv_cache_tensors=[],
        kv_cache_groups=[
            group(
                "full",
                FullAttentionSpec(
                    block_size=800,
                    num_kv_heads=8,
                    head_size=64,
                    dtype=None,
                ),
                blocks_per_page=1,
                kernel_rows_per_block=2,
                row_bytes=128,
            ),
            # Not prefix-cacheable: never a store group, so its geometry is
            # unused; keep it consistent anyway.
            group(
                "qsa",
                CircularBufferSpec(
                    block_size=8,
                    num_kv_heads=1,
                    head_size=64,
                    head_size_v=0,
                    dtype=torch.float16,
                ),
                blocks_per_page=1,
                kernel_rows_per_block=1,
                row_bytes=128,
            ),
            group(
                "mamba",
                MambaSpec(
                    block_size=800,
                    shapes=((1, 1),),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                ),
                blocks_per_page=1,
                kernel_rows_per_block=2,
                row_bytes=128,
            ),
        ],
        block_size=800,
        large_page_bytes=256,
    )


def _write_mooncake_config(tmp_path, config: dict[str, object]) -> str:
    config_path = tmp_path / "mooncake_config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return str(config_path)


def _install_fake_mooncake(monkeypatch, store_instance: MagicMock):
    class FakeReplicateConfig:
        def __init__(self) -> None:
            self.preferred_segment = ""

    fake_store_module = types.ModuleType("mooncake.store")
    fake_store_module.MooncakeDistributedStore = lambda: store_instance  # type: ignore[attr-defined]
    fake_store_module.ReplicateConfig = FakeReplicateConfig  # type: ignore[attr-defined]
    fake_store_module.ObjectDataType = SimpleNamespace(TENSOR=1)  # type: ignore[attr-defined]
    fake_mooncake_module = types.ModuleType("mooncake")
    fake_mooncake_module.store = fake_store_module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mooncake", fake_mooncake_module)
    monkeypatch.setitem(sys.modules, "mooncake.store", fake_store_module)
    return FakeReplicateConfig


def _patch_worker_runtime(
    monkeypatch,
    *,
    local_ip: str = "10.0.0.7",
    tp_rank: int = 0,
    tp_size: int = 1,
    pcp_size: int = 1,
    dcp_size: int = 1,
) -> None:
    pcp_group = SimpleNamespace(world_size=pcp_size, rank_in_group=0)
    # DCP groups are contiguous splits of the TP group (see
    # parallel_state.py), so dcp_rank == tp_rank % dcp_size.
    dcp_group = SimpleNamespace(world_size=dcp_size, rank_in_group=tp_rank % dcp_size)
    monkeypatch.setattr(worker, "get_tensor_model_parallel_rank", lambda: tp_rank)
    monkeypatch.setattr(worker, "get_tensor_model_parallel_world_size", lambda: tp_size)
    monkeypatch.setattr(worker, "get_pcp_group", lambda: pcp_group)
    monkeypatch.setattr(worker, "get_dcp_group", lambda: dcp_group)
    monkeypatch.setattr(worker, "get_ip", lambda: local_ip)
    monkeypatch.setattr(worker, "LookupKeyServer", MagicMock())


def test_create_mem_pool_returns_none_when_unconfigured():
    assert worker.MooncakeStoreWorker._create_mem_pool({}, _FakeModelConfig()) is None


def test_create_mem_pool_rejects_unknown_pool():
    with pytest.raises(
        ValueError,
        match="Unsupported custom_mem_pool='UNKNOWN'",
    ):
        worker.MooncakeStoreWorker._create_mem_pool(
            {"custom_mem_pool": "unknown"}, _FakeModelConfig()
        )


@pytest.mark.parametrize(
    "conflicting_option", ["enable_sleep_mode", "enable_cumem_allocator"]
)
def test_create_mem_pool_rejects_cumem_conflicts(conflicting_option):
    model_config = _FakeModelConfig()
    setattr(model_config, conflicting_option, True)

    with pytest.raises(ValueError, match="custom_mem_pool is incompatible"):
        worker.MooncakeStoreWorker._create_mem_pool(
            {"custom_mem_pool": "NVLINK"}, model_config
        )


@pytest.mark.parametrize(
    ("pool_type", "allocator_name"),
    [("nvlink", "NVLinkAllocator"), ("barex", "BarexAllocator")],
)
def test_create_mem_pool_uses_mooncake_allocator(
    monkeypatch, pool_type, allocator_name
):
    allocator_cls = MagicMock()
    allocator = allocator_cls.get_allocator.return_value
    allocator_module = types.ModuleType("mooncake.allocator")
    setattr(allocator_module, allocator_name, allocator_cls)
    mooncake_module = types.ModuleType("mooncake")
    mooncake_module.allocator = allocator_module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mooncake", mooncake_module)
    monkeypatch.setitem(sys.modules, "mooncake.allocator", allocator_module)

    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 2)
    mem_pool = MagicMock()
    mem_pool_ctor = MagicMock(return_value=mem_pool)
    monkeypatch.setattr(torch.cuda, "MemPool", mem_pool_ctor)

    result = worker.MooncakeStoreWorker._create_mem_pool(
        {"custom_mem_pool": pool_type}, _FakeModelConfig()
    )

    assert result is mem_pool
    allocator_cls.get_allocator.assert_called_once_with(torch.device("cuda", 2))
    allocator.allocator.assert_called_once_with()
    mem_pool_ctor.assert_called_once_with(allocator.allocator.return_value)


def test_pool_key_to_string_without_prefix_is_unchanged():
    """Default (empty) cache_prefix keeps keys byte-identical to the
    historical unprefixed format so existing deployments keep their hits."""
    key = PoolKey(KeyMetadata("test-model", 0, 0, 0, 0), "deadbeef")
    assert (
        key.to_string() == "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@group:0@deadbeef"
    )


def test_pool_key_store_tp_namespace():
    key = PoolKey(
        KeyMetadata(
            "test-model",
            2,
            0,
            0,
            0,
            store_namespace=_TP_SHARED_NAMESPACE,
        ),
        "deadbeef",
    )

    assert key.to_string() == f"{_tp_shared_prefix(2)}@deadbeef"


def test_tp_shared_sending_writes_each_store_shard():
    store = MagicMock()
    store.batch_is_exist.return_value = [0, 0, 0, 0]
    store.batch_put_from_multi_buffers.return_value = [1, 1, 1, 1]
    db, _ = _make_tp_shared_db()
    thread = _make_store_sending_thread(
        store,
        token_databases=[db],
        replicate_config=SimpleNamespace(),
    )
    req = _make_store_req("req", [b"h0", b"h1"])
    _run_store_req(thread, req)

    keys, addrs, sizes, _ = store.batch_put_from_multi_buffers.call_args.args
    assert keys == [
        f"{_tp_shared_prefix(0)}@6830",
        f"{_tp_shared_prefix(1)}@6830",
        f"{_tp_shared_prefix(0)}@6831",
        f"{_tp_shared_prefix(1)}@6831",
    ]
    assert len(addrs) == len(sizes) == 4


def test_tp_shared_kv_events_aggregate_once_per_logical_block():
    aggregator = KVEventAggregator(num_workers=2)

    for tp_rank in range(2):
        store = MagicMock()
        store.batch_is_exist.return_value = [0, 0]
        store.batch_put_from_multi_buffers.return_value = [256, 256]
        db, _ = _make_tp_shared_db(tp_rank)
        thread = _make_store_sending_thread(
            store,
            token_databases=[db],
            tp_rank=tp_rank,
            replicate_config=SimpleNamespace(),
        )
        thread.enable_kv_event = True

        _run_store_req(thread, _make_event_store_req(16))

        events = thread.get_kv_events()
        assert len(events) == 1
        aggregator.add_events(events)

    assert len(aggregator.get_common_events()) == 1


def test_tp_shared_kv_event_waits_for_all_missing_shards():
    store = MagicMock()
    store.batch_is_exist.return_value = [0, 0]
    store.batch_put_from_multi_buffers.return_value = [256, -1]
    db, _ = _make_tp_shared_db()
    thread = _make_store_sending_thread(
        store,
        token_databases=[db],
        replicate_config=SimpleNamespace(),
    )
    thread.enable_kv_event = True

    _run_store_req(thread, _make_event_store_req(16))

    assert thread.get_kv_events() == []


def test_tp_shared_kv_event_counts_existing_shards_as_satisfied():
    store = MagicMock()
    store.batch_is_exist.return_value = [1, 0]
    store.batch_put_from_multi_buffers.return_value = [256]
    db, _ = _make_tp_shared_db()
    thread = _make_store_sending_thread(
        store,
        token_databases=[db],
        replicate_config=SimpleNamespace(),
    )
    thread.enable_kv_event = True

    _run_store_req(thread, _make_event_store_req(16))

    assert len(thread.get_kv_events()) == 1


def test_tp_shared_receiving_reads_each_local_store_shard():
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec

    store = MagicMock()
    store.batch_get_into_multi_buffers.return_value = [1, 1, 1, 1]
    db, _ = _make_tp_shared_db()
    spec = FullAttentionSpec(block_size=16, num_kv_heads=8, head_size=64, dtype=None)
    coord = mooncake_store_worker.MooncakeStoreCoordinator(
        [KVCacheGroupSpec(["layer0"], spec)],
        scheduler_block_size=16,
        block_size=16,
    )
    thread = mooncake_store_worker.KVCacheStoreRecvingThread(
        store=store,
        token_databases=[db],
        block_size=16,
        tp_rank=0,
        ready_event=threading.Event(),
        coord=coord,
        group_kv_cache_ids=(0,),
    )
    thread.request_queue.task_done = MagicMock()

    thread._handle_request(_make_load_req("req", [b"h0", b"h1"], token_len=32))

    keys, addrs, sizes = store.batch_get_into_multi_buffers.call_args.args
    assert keys == [
        f"{_tp_shared_prefix(0)}@6830",
        f"{_tp_shared_prefix(1)}@6830",
        f"{_tp_shared_prefix(0)}@6831",
        f"{_tp_shared_prefix(1)}@6831",
    ]
    assert len(addrs) == len(sizes) == 4


def test_pool_key_cache_prefix_namespaces_and_disambiguates():
    """A non-empty cache_prefix is prepended, and two instances with
    different prefixes never collide on identical block hashes."""
    md_a = KeyMetadata("test-model", 0, 0, 0, 0, cache_prefix="depA")
    md_b = KeyMetadata("test-model", 0, 0, 0, 0, cache_prefix="depB")

    key_a = PoolKey(md_a, "deadbeef")
    key_b = PoolKey(md_b, "deadbeef")

    assert key_a.to_string() == (
        "depA@test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@group:0@deadbeef"
    )
    assert key_a.to_string() != key_b.to_string()
    assert hash(key_a) != hash(key_b)


def test_default_local_buffer_size_matches_pr40900():
    """PR-40900 shipped a 4 GiB default for local_buffer_size; the dual-mode
    patch preserves it (and the JSON key) so unchanged PR-40900 configs work."""
    assert mooncake_store.DEFAULT_LOCAL_BUFFER_SIZE == 4 * 1024**3


def test_get_requester_local_hostname_prefers_override(monkeypatch):
    monkeypatch.setenv("MOONCAKE_REQUESTER_LOCAL_HOSTNAME", "worker-a:50053")

    assert rdma_utils.get_requester_local_hostname("10.0.0.7") == "worker-a:50053"


def test_get_configured_preferred_segment_returns_explicit_override():
    assert (
        rdma_utils.get_configured_preferred_segment(
            {"preferred_segment": "10.0.0.7:50053"}
        )
        == "10.0.0.7:50053"
    )


def test_get_configured_preferred_segment_prefers_explicit_over_env(monkeypatch):
    monkeypatch.setenv("MOONCAKE_PREFERRED_SEGMENT", "10.0.0.8:50053")

    assert (
        rdma_utils.get_configured_preferred_segment(
            {"preferred_segment": "10.0.0.7:50053"}
        )
        == "10.0.0.7:50053"
    )


def test_get_configured_preferred_segment_returns_env_override(monkeypatch):
    monkeypatch.setenv("MOONCAKE_PREFERRED_SEGMENT", "10.0.0.8:50053")

    assert rdma_utils.get_configured_preferred_segment({}) == "10.0.0.8:50053"


def test_get_configured_preferred_segment_rejects_empty_override():
    with pytest.raises(ValueError, match="preferred_segment"):
        rdma_utils.get_configured_preferred_segment({"preferred_segment": "  "})


class _ReplicaDesc:
    def __init__(self, tier: str):
        self.tier = tier

    def is_memory_replica(self) -> bool:
        return self.tier == "memory"

    def is_disk_replica(self) -> bool:
        return self.tier == "disk"

    def is_local_disk_replica(self) -> bool:
        return self.tier == "disk"


def test_store_sending_thread_skips_request_during_cpu_pressure():
    store = MagicMock()
    store.batch_is_exist.side_effect = lambda keys: [0] * len(keys)
    store.batch_put_from_multi_buffers.side_effect = [
        [-200, -200],
        [256, 256],
        [256, 256],
    ]
    thread = _make_store_sending_thread(store)

    _run_store_req(thread, _make_store_req("req-a", [b"a0", b"a1"]))

    assert thread._store_pressure_active is True
    assert "req-a" in thread._skip_store_requests
    assert store.batch_put_from_multi_buffers.call_count == 1

    _run_store_req(thread, _make_store_req("req-a", [b"a2", b"a3"]))

    assert store.batch_put_from_multi_buffers.call_count == 1

    _run_store_req(thread, _make_store_req("req-b", [b"b0", b"b1"]))

    assert thread._store_pressure_active is False
    assert "req-a" not in thread._skip_store_requests
    assert store.batch_put_from_multi_buffers.call_count == 2

    _run_store_req(thread, _make_store_req("req-a", [b"a4", b"a5"]))

    assert store.batch_put_from_multi_buffers.call_count == 3


def test_store_sending_thread_records_mooncake_metrics():
    store = MagicMock()
    store.batch_is_exist.return_value = [0, 0]
    store.batch_put_from_multi_buffers.return_value = [256, 256]
    thread = _make_store_sending_thread(store)
    stats = MooncakeStoreConnectorStats()
    thread._record_operation_cb = stats.record_operation

    _run_store_req(thread, _make_store_req("req-a", [b"a0", b"a1"]))

    assert len(stats.data["save_exists"]) == 1
    assert stats.data["save_exists"][0]["num_keys"] == 2
    assert len(stats.data["save_put"]) == 1
    assert stats.data["save_put"][0]["num_bytes"] == 512
    assert stats.data["save_put"][0]["status"] == "ok"


def test_process_tokens_uses_mask_num_as_start_chunk():
    db = _make_rank_local_db(KeyMetadata("test-model", 0, 0, 0, 0), 32)
    block_hashes = _RecordingBlockHashes([bytes([i]) for i in range(4)])

    results = list(
        db.process_tokens(
            token_len=96,
            block_hashes=block_hashes,
            mask_num=64,
        )
    )

    assert block_hashes.accessed == [2]
    assert results == [(64, 96, bytes([2]))]


def test_process_tokens_applies_chunk_mask_before_hash_access():
    db = _make_rank_local_db(KeyMetadata("test-model", 0, 0, 0, 0), 32)
    block_hashes = _RecordingBlockHashes([bytes([i]) for i in range(4)])

    results = list(
        db.process_tokens(
            token_len=128,
            block_hashes=block_hashes,
            mask_num=64,
            chunk_mask=[False, True],
        )
    )

    assert block_hashes.accessed == [3]
    assert results == [(96, 128, bytes([3]))]


def test_process_tokens_applies_stride_before_hash_access():
    db = _make_rank_local_db(KeyMetadata("test-model", 0, 0, 0, 0), 32)
    block_hashes = _RecordingBlockHashes([bytes([i]) for i in range(4)])

    results = list(
        db.process_tokens(
            token_len=128,
            block_hashes=block_hashes,
            mask_num=64,
            put_step=2,
            put_step_rank=1,
        )
    )

    assert block_hashes.accessed == [3]
    assert results == [(96, 128, bytes([3]))]


def test_store_sending_thread_delta_saves_only_new_full_attention_chunks():
    store = MagicMock()
    store.batch_is_exist.side_effect = lambda keys: [0] * len(keys)
    store.batch_put_from_multi_buffers.return_value = [256, 256]
    thread = _make_store_sending_thread(store)

    thread._saved_offset["req-a"] = 32
    _run_store_req(
        thread,
        ReqMeta(
            req_id="req-a",
            token_len_chunk=64,
            block_ids=([1, 2, 3, 4],),
            block_hashes=[b"a0", b"a1", b"a2", b"a3"],
            can_save=True,
        ),
    )

    keys = store.batch_is_exist.call_args.args[0]
    assert keys == [
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@group:0@6132",
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@group:0@6133",
    ]
    assert store.batch_put_from_multi_buffers.call_args.args[0] == keys


def test_store_sending_thread_delta_strides_with_local_phase():
    store = MagicMock()
    store.batch_is_exist.side_effect = lambda keys: [0] * len(keys)
    store.batch_put_from_multi_buffers.return_value = [256]
    thread = _make_store_sending_thread(store, tp_rank=0, put_step=2)

    thread._saved_offset["req-a"] = 16
    _run_store_req(
        thread,
        ReqMeta(
            req_id="req-a",
            token_len_chunk=64,
            block_ids=([1, 2, 3, 4],),
            block_hashes=[b"a0", b"a1", b"a2", b"a3"],
            can_save=True,
        ),
    )

    keys = store.batch_is_exist.call_args.args[0]
    assert keys == [
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@group:0@6132",
    ]
    assert store.batch_put_from_multi_buffers.call_args.args[0] == keys


def test_tp_sharded_group_saves_every_block_on_every_rank():
    """Sharded ranks must write every block because peers hold different bytes."""
    store = MagicMock()
    store.batch_is_exist.side_effect = lambda keys: [0] * len(keys)
    store.batch_put_from_multi_buffers.side_effect = lambda keys, *a: [256] * len(keys)
    thread = _make_store_sending_thread(store, tp_rank=0, put_step=2)
    thread.group_put_steps = [1]

    _run_store_req(
        thread,
        ReqMeta(
            req_id="req-a",
            token_len_chunk=64,
            block_ids=([1, 2, 3, 4],),
            block_hashes=[b"a0", b"a1", b"a2", b"a3"],
            can_save=True,
        ),
    )

    keys = store.batch_is_exist.call_args.args[0]
    assert len(keys) == 4


def test_store_sending_thread_retries_skipped_range_after_pressure():
    store = MagicMock()
    store.batch_is_exist.side_effect = lambda keys: [0] * len(keys)
    store.batch_put_from_multi_buffers.side_effect = lambda keys, *a: [256] * len(keys)
    thread = _make_store_sending_thread(store, tp_rank=0, put_step=2)

    # Under pressure the request is skipped without persisting anything, so the
    # saved offset stays at 0.
    thread._store_pressure_active = True
    thread._skip_store_requests.add("req-a")

    _run_store_req(
        thread,
        ReqMeta(
            req_id="req-a",
            token_len_chunk=16,
            block_ids=([1],),
            block_hashes=[b"a0"],
            can_save=True,
        ),
    )

    store.batch_is_exist.assert_not_called()
    assert thread._saved_offset.get("req-a", 0) == 0

    thread._store_pressure_active = False
    thread._skip_store_requests.clear()

    # The next batch resumes from offset 0, re-covering the chunk skipped under
    # pressure (chunk 0) rather than losing it.
    _run_store_req(
        thread,
        ReqMeta(
            req_id="req-a",
            token_len_chunk=64,
            block_ids=([1, 2, 3, 4],),
            block_hashes=[b"a0", b"a1", b"a2", b"a3"],
            can_save=True,
        ),
    )

    keys = store.batch_is_exist.call_args.args[0]
    assert keys == [
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@group:0@6130",
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@group:0@6132",
    ]
    assert store.batch_put_from_multi_buffers.call_args.args[0] == keys


def _make_boundary_send_thread(
    store,
    *,
    replicate_config=None,
    enable_group_semantics=False,
    supports_group_ids=False,
):
    """Sending thread over a full-attention group 0 and the mamba "align"
    group 1 that boundary hand-offs reference, both at block size 16."""
    full = FullAttentionSpec(block_size=16, num_kv_heads=8, head_size=64, dtype=None)
    mamba = MambaSpec(
        block_size=16,
        shapes=((1, 1),),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
    )
    coord = mooncake_store_worker.MooncakeStoreCoordinator(
        [KVCacheGroupSpec(["L0"], full), KVCacheGroupSpec(["L1"], mamba)],
        scheduler_block_size=16,
        block_size=16,
    )
    db = _make_rank_local_db(
        KeyMetadata("test-model", 0, 0, 0, 0), 16, base_addr=0x1000
    )
    db_mamba = _make_rank_local_db(
        KeyMetadata("test-model", 0, 0, 0, 0, group_id=1), 16, base_addr=0x2000
    )
    return _make_store_sending_thread(
        store,
        coord=coord,
        token_databases=[db, db_mamba],
        replicate_config=replicate_config,
        enable_group_semantics=enable_group_semantics,
        supports_group_ids=supports_group_ids,
    )


def _make_boundary_req() -> ReqMeta:
    """Offload-only job for three mamba states at boundaries 16, 32 and 48."""
    return ReqMeta(
        req_id="req-a",
        token_len_chunk=0,
        block_ids=([1, 2, 3], [1]),
        block_hashes=[b"a0", b"a1", b"a2"],
        can_save=True,
        boundary_puts=[
            BoundaryPut(1, 7, 16),
            BoundaryPut(1, 8, 32),
            BoundaryPut(1, 9, 48),
        ],
        num_prompt_tokens=49,
    )


def test_store_sending_thread_skips_null_sparse_group_blocks():
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheGroupSpec,
    )

    store = MagicMock()
    store.batch_is_exist.side_effect = lambda keys: [0] * len(keys)
    store.batch_put_from_multi_buffers.side_effect = (
        lambda keys, addrs, sizes, replicate_config: [256] * len(keys)
    )
    full = FullAttentionSpec(block_size=16, num_kv_heads=8, head_size=64, dtype=None)
    sparse = FullAttentionSpec(block_size=16, num_kv_heads=8, head_size=64, dtype=None)
    coord = mooncake_store_worker.MooncakeStoreCoordinator(
        [
            KVCacheGroupSpec(["full"], full),
            KVCacheGroupSpec(["sparse"], sparse),
        ],
        scheduler_block_size=16,
        block_size=16,
    )

    db_full = _make_rank_local_db(
        KeyMetadata("test-model", 0, 0, 0, 0, group_id=0),
        16,
        base_addr=0x1000,
    )
    db_sparse = _make_rank_local_db(
        KeyMetadata("test-model", 0, 0, 0, 0, group_id=1),
        16,
        base_addr=0x2000,
        block_bytes=512,
    )

    thread = _make_store_sending_thread(
        store,
        coord=coord,
        token_databases=[db_full, db_sparse],
    )
    _run_store_req(
        thread,
        ReqMeta(
            req_id="req-a",
            token_len_chunk=48,
            block_ids=([1, 2, 3], [NULL_BLOCK_ID, 4, NULL_BLOCK_ID]),
            block_hashes=[b"a0", b"a1", b"a2"],
            can_save=True,
        ),
    )

    keys, addrs, _, _ = store.batch_put_from_multi_buffers.call_args.args
    sparse_keys = [key for key in keys if "@group:1" in key]
    assert [key.rsplit("@", 1)[-1] for key in sparse_keys] == [b"a1".hex()]
    assert all(addr[0] >= 0x2000 for key, addr in zip(keys, addrs) if "@group:1" in key)


def test_boundary_offload_skips_cap_omitted_mamba_group():
    """A Mamba group omitted by the handoff cap must not fall back to the
    connector's positional block table."""
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheGroupSpec,
        MambaSpec,
    )

    store = MagicMock()
    store.batch_is_exist.side_effect = lambda keys: [0] * len(keys)
    store.batch_put_from_multi_buffers.side_effect = lambda keys, *a: [256] * len(keys)

    full_spec = FullAttentionSpec(
        block_size=16, num_kv_heads=8, head_size=64, dtype=torch.float32
    )
    mamba_spec = MambaSpec(
        block_size=16,
        shapes=((1, 1),),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
    )
    coord = mooncake_store_worker.MooncakeStoreCoordinator(
        [
            KVCacheGroupSpec(["full"], full_spec),
            KVCacheGroupSpec(["mamba-1"], mamba_spec),
            KVCacheGroupSpec(["mamba-2"], mamba_spec),
        ],
        scheduler_block_size=16,
        block_size=16,
    )

    def make_db(group_id: int, block_size: int, base_addr: int):
        return _make_rank_local_db(
            KeyMetadata("test-model", 0, 0, 0, 0, group_id=group_id),
            block_size,
            base_addr=base_addr,
        )

    db_full = make_db(0, 16, 0x1000)
    db_mamba_1 = make_db(1, 16, 0x2000)
    db_mamba_2 = make_db(2, 16, 0x3000)
    thread = _make_store_sending_thread(
        store,
        coord=coord,
        token_databases=[db_full, db_mamba_1, db_mamba_2],
    )

    req = ReqMeta(
        req_id="req-a",
        token_len_chunk=0,
        block_ids=([1, 2, 3], [5], [9]),
        block_hashes=[b"a0", b"a1", b"a2"],
        can_save=True,
        # The scheduler accepted group 1 and omitted group 2 at boundary 48.
        boundary_puts=[BoundaryPut(1, 7, 48)],
        num_prompt_tokens=49,
    )
    assert thread._maybe_offload_boundary_states(req)

    keys, addrs, _sizes, _replicate_config = (
        store.batch_put_from_multi_buffers.call_args.args
    )
    puts = list(zip(keys, addrs, strict=True))
    boundary_hash = BlockHash(b"a2")
    assert (db_mamba_1.key_for(boundary_hash), [0x2000 + 7 * 256]) in puts
    assert (db_mamba_2.key_for(boundary_hash), [0x3000 + 9 * 256]) not in puts


def test_boundary_offload_replaces_stale_group_ids_after_filtering():
    store = MagicMock()
    store.batch_is_exist.return_value = [1, 0, 0]
    store.batch_put_from_multi_buffers.return_value = [256, 256]
    replicate_config = SimpleNamespace(group_ids=["stale"])
    thread = _make_boundary_send_thread(
        store,
        replicate_config=replicate_config,
        enable_group_semantics=True,
        supports_group_ids=True,
    )

    assert thread._maybe_offload_boundary_states(_make_boundary_req())

    keys, _addrs, _sizes, config = store.batch_put_from_multi_buffers.call_args.args
    assert keys == [
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@group:1@6131",
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@group:1@6132",
    ]
    assert config is replicate_config
    assert config.group_ids == [
        "vllm-mooncake-store:test-model@6131",
        "vllm-mooncake-store:test-model@6132",
    ]


def test_boundary_offload_honors_active_pressure_gate():
    store = MagicMock()
    thread = _make_boundary_send_thread(store)
    thread._store_pressure_active = True
    thread._skip_store_requests.add("req-a")

    _run_store_req(thread, _make_boundary_req())

    store.batch_is_exist.assert_not_called()
    store.batch_put_from_multi_buffers.assert_not_called()


def test_boundary_put_failure_activates_pressure_gate():
    store = MagicMock()
    store.batch_is_exist.side_effect = lambda keys: [0] * len(keys)
    store.batch_put_from_multi_buffers.return_value = [256, -200, 256]
    thread = _make_boundary_send_thread(store)

    _run_store_req(thread, _make_boundary_req())

    assert thread._store_pressure_active is True
    assert thread._skip_store_requests == {"req-a"}
    assert thread._saved_offset.get("req-a", 0) == 0

    _run_store_req(thread, _make_boundary_req())
    assert store.batch_put_from_multi_buffers.call_count == 1


def test_normal_save_excludes_mamba_group_and_null_blocks():
    """The positional normal save must never cover mamba chunks (align-mode
    block tables are not append-only) and must skip chunks whose block is the
    reserved null block, for any group."""
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheGroupSpec,
        MambaSpec,
    )

    store = MagicMock()
    store.batch_is_exist.side_effect = lambda keys: [0] * len(keys)
    store.batch_put_from_multi_buffers.side_effect = lambda keys, *a: [256] * len(keys)

    full_spec = FullAttentionSpec(
        block_size=16, num_kv_heads=8, head_size=64, dtype=None
    )
    mamba_spec = MambaSpec(
        block_size=16,
        shapes=((1, 1),),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
    )
    coord = mooncake_store_worker.MooncakeStoreCoordinator(
        [KVCacheGroupSpec(["L0"], full_spec), KVCacheGroupSpec(["L1"], mamba_spec)],
        scheduler_block_size=16,
        block_size=16,
    )
    db_full = _make_rank_local_db(
        KeyMetadata("test-model", 0, 0, 0, 0, group_id=0), 16, base_addr=0x1000
    )
    db_mamba = _make_rank_local_db(
        KeyMetadata("test-model", 0, 0, 0, 0, group_id=1), 16, base_addr=0x2000
    )
    thread = _make_store_sending_thread(
        store, coord=coord, token_databases=[db_full, db_mamba]
    )

    _run_store_req(
        thread,
        ReqMeta(
            req_id="r0",
            token_len_chunk=64,
            # FA has a null at chunk 1; the mamba table is align-mode shaped
            # (interior nulls + one state block) and must not be read at all.
            block_ids=([1, 0, 3, 4], [0, 0, 0, 9]),
            block_hashes=[b"a0", b"a1", b"a2", b"a3"],
            can_save=True,
        ),
    )

    keys, addrs, _sizes, _ = store.batch_put_from_multi_buffers.call_args.args
    assert all("@group:0@" in k for k in keys)
    assert [k.rsplit("@", 1)[-1] for k in keys] == ["6130", "6132", "6133"]
    assert addrs == [[0x1000 + 256], [0x1000 + 3 * 256], [0x1000 + 4 * 256]]


def test_block_aligned_snapshot_offload_uses_provided_block():
    """A block-aligned hand-off (sparse-retention mamba checkpoint) uploads
    exactly the core-pinned block under the boundary-end hash key; the
    positional block table entry for that chunk must be ignored."""
    store = MagicMock()
    store.batch_is_exist.side_effect = lambda keys: [0] * len(keys)
    store.batch_put_from_multi_buffers.side_effect = lambda keys, *a: [256] * len(keys)
    thread = _make_boundary_send_thread(store)

    hs = [bytes([i + 1]) * 4 for i in range(2)]  # 2 blocks = 32 tokens
    req = ReqMeta(
        req_id="req-a",
        token_len_chunk=0,
        block_ids=([1, 2, 3], [5]),
        block_hashes=hs,
        can_save=True,
        boundary_puts=[BoundaryPut(1, 7, 32)],
    )
    assert thread._maybe_offload_boundary_states(req)

    keys, addrs, _sizes, _ = store.batch_put_from_multi_buffers.call_args.args
    # One key, keyed by hs[32 // 16 - 1], read from the handed-off block 7 —
    # not from block_ids[1].
    assert keys == [thread.token_databases[1].key_for(BlockHash(hs[1]))]
    assert addrs == [[0x2000 + 7 * 256]]


def test_snapshot_offload_skips_null_handoff_block():
    """A hand-off the core could not materialize (null block) carries no
    committed state; persisting it would poison the boundary key."""
    store = MagicMock()
    thread = _make_boundary_send_thread(store)

    hs = [bytes([i + 1]) * 4 for i in range(2)]
    assert thread._maybe_offload_boundary_states(
        ReqMeta(
            req_id="req-a",
            token_len_chunk=0,
            block_ids=([1, 2, 3], [5]),
            block_hashes=hs,
            can_save=True,
            boundary_puts=[BoundaryPut(1, NULL_BLOCK_ID, 32)],
        )
    )
    store.batch_is_exist.assert_not_called()
    store.batch_put_from_multi_buffers.assert_not_called()


def test_worker_meta_is_absent_before_any_completed_job():
    send_thread = _make_boundary_send_thread(MagicMock())
    store_worker = object.__new__(mooncake_store_worker.MooncakeStoreWorker)
    store_worker.kv_send_thread = send_thread

    assert store_worker.build_connector_worker_meta() is None


def test_store_sending_thread_delta_start_rank_saves_second_local_chunk():
    store = MagicMock()
    store.batch_is_exist.side_effect = lambda keys: [0] * len(keys)
    store.batch_put_from_multi_buffers.return_value = [256, 256]
    thread = _make_store_sending_thread(store, tp_rank=1, put_step=2)

    thread._saved_offset["req-a"] = 16
    _run_store_req(
        thread,
        ReqMeta(
            req_id="req-a",
            token_len_chunk=64,
            block_ids=([1, 2, 3, 4],),
            block_hashes=[b"a0", b"a1", b"a2", b"a3"],
            can_save=True,
        ),
    )

    keys = store.batch_is_exist.call_args.args[0]
    assert keys == [
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@group:0@6131",
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@group:0@6133",
    ]
    assert store.batch_put_from_multi_buffers.call_args.args[0] == keys


def test_store_sending_thread_delta_saves_only_new_masked_chunks():
    store = MagicMock()
    store.batch_is_exist.side_effect = lambda keys: [0] * len(keys)
    store.batch_put_from_multi_buffers.side_effect = (
        lambda keys, addrs, sizes, replicate_config: [256] * len(keys)
    )
    coord = SimpleNamespace(
        lcm_block_size=16,
        store_mask=lambda token_len, start_token, num_prompt_tokens=None: (
            None,
            [True, False],
        ),
    )
    coord.store_mask = MagicMock(side_effect=coord.store_mask)

    db_full = _make_rank_local_db(
        KeyMetadata("test-model", 0, 0, 0, 0, group_id=0),
        16,
        base_addr=0x1000,
    )
    db_masked = _make_rank_local_db(
        KeyMetadata("test-model", 0, 0, 0, 0, group_id=1),
        16,
        base_addr=0x2000,
    )

    thread = _make_store_sending_thread(
        store,
        coord=coord,
        token_databases=[db_full, db_masked],
    )

    thread._saved_offset["req-a"] = 32
    _run_store_req(
        thread,
        ReqMeta(
            req_id="req-a",
            token_len_chunk=64,
            block_ids=([0, 1, 2, 3], [0, 1, 2, 3]),
            block_hashes=[b"a0", b"a1", b"a2", b"a3"],
            can_save=True,
            num_prompt_tokens=33,
            prefill_end_tokens=64,
        ),
    )

    keys = store.batch_is_exist.call_args.args[0]
    full_hashes = [k.rsplit("@", 1)[-1] for k in keys if "@group:0" in k]
    masked_hashes = [k.rsplit("@", 1)[-1] for k in keys if "@group:1" in k]

    assert full_hashes == [b"a2".hex(), b"a3".hex()]
    assert masked_hashes == [b"a2".hex()]
    coord.store_mask.assert_called_once_with(64, 32, num_prompt_tokens=64)


def test_store_sending_thread_prepares_missing_chunks_once_per_group():
    store = MagicMock()
    store.batch_is_exist.return_value = [0, 1, 0, 1, 0, 0]
    store.batch_put_from_multi_buffers.return_value = [256, 256, 512, 512]
    coord = SimpleNamespace(
        lcm_block_size=16,
        store_mask=lambda token_len, start_token, num_prompt_tokens=None: (
            None,
            None,
        ),
    )

    db0 = _make_rank_local_db(
        KeyMetadata("test-model", 0, 0, 0, 0, group_id=0),
        16,
        base_addr=0x1000,
    )
    db0.store_layout.prepare_values = MagicMock(wraps=db0.store_layout.prepare_values)

    db1 = _make_rank_local_db(
        KeyMetadata("test-model", 0, 0, 0, 0, group_id=1),
        16,
        base_addr=0x2000,
        block_bytes=512,
    )
    db1.store_layout.prepare_values = MagicMock(wraps=db1.store_layout.prepare_values)

    thread = _make_store_sending_thread(
        store,
        coord=coord,
        token_databases=[db0, db1],
    )
    _run_store_req(
        thread,
        ReqMeta(
            req_id="req-a",
            token_len_chunk=48,
            block_ids=([1, 2, 3], [3, 2, 1]),
            block_hashes=[b"a0", b"a1", b"a2"],
            can_save=True,
        ),
    )

    db0.store_layout.prepare_values.assert_called_once_with(
        [(0, 16), (32, 48)], [1, 2, 3], [0, 0]
    )
    db1.store_layout.prepare_values.assert_called_once_with(
        [(16, 32), (32, 48)], [3, 2, 1], [0, 0]
    )

    keys, addrs, sizes, _ = store.batch_put_from_multi_buffers.call_args.args
    assert [key.rsplit("@", 1)[-1] for key in keys] == [
        "6130",
        "6132",
        "6131",
        "6132",
    ]
    assert addrs == [[0x1100], [0x1300], [0x2400], [0x2200]]
    assert sizes == [[256], [256], [512], [512]]


def test_store_sending_thread_only_skips_on_no_available_handle():
    store = MagicMock()
    store.batch_is_exist.side_effect = lambda keys: [0] * len(keys)
    store.batch_put_from_multi_buffers.side_effect = [
        [-500, -500],
        [256, 256],
    ]
    thread = _make_store_sending_thread(store)

    _run_store_req(thread, _make_store_req("req-a", [b"a0", b"a1"]))

    assert thread._store_pressure_active is False
    assert "req-a" not in thread._skip_store_requests
    assert store.batch_put_from_multi_buffers.call_count == 1

    _run_store_req(thread, _make_store_req("req-a", [b"a2", b"a3"]))

    assert store.batch_put_from_multi_buffers.call_count == 2


@pytest.mark.parametrize(
    "failing_call", ["batch_is_exist", "batch_put_from_multi_buffers"]
)
def test_store_sending_thread_reports_job_when_store_raises(failing_call):
    # A store that blows up must still report its job, or the scheduler keeps
    # the job's GPU block references for the rest of the run.
    store = MagicMock()
    store.batch_is_exist.return_value = [0, 0]
    getattr(store, failing_call).side_effect = RuntimeError("mooncake down")
    thread = _make_store_sending_thread(store)
    req = _make_store_req("req-a", [b"a0", b"a1"])

    with contextlib.suppress(RuntimeError):
        _run_store_req(thread, req)

    assert thread.take_completed_saves() == {req.store_job_id: 1}


def test_store_sending_thread_reports_job_when_the_preamble_raises():
    # The report has to survive a failure before the first store call too, so
    # every dequeue leaves through the same exit.
    thread = _make_store_sending_thread(MagicMock())
    req = _make_store_req("req-a", [b"a0", b"a1"])
    req.token_len_chunk = None

    with contextlib.suppress(TypeError):
        _run_store_req(thread, req)

    assert thread.take_completed_saves() == {req.store_job_id: 1}
    thread.request_queue.task_done.assert_called_once()


def test_stale_store_job_cannot_touch_a_reused_request_id():
    # A preempted request resumes under its original id, so a job left over from
    # the retired generation carries a req_id that now belongs to a live one.
    thread = _make_store_sending_thread(MagicMock())
    stale = _make_store_req("req-a", [b"a0", b"a1"])
    stale.store_job_id = 1
    thread.add_request(stale)
    thread._record_saved(stale, 32)
    thread.delete_finished_stored_request("req-a")

    live = _make_store_req("req-a", [b"a0", b"a1"])
    live.store_job_id = 2
    thread.add_request(live)
    thread._retry_token_ids["req-a"] = (32, list(range(32, 64)))

    thread.finish_store_job(stale)
    thread._record_saved(stale, 64)
    thread._mark_request_skipped_for_pressure(stale)
    thread._update_retry_token_ids(stale, False, 0, list(range(32)))
    thread._update_retry_token_ids(stale, True, 0, None)

    assert thread.is_live_store_job(live)
    assert not thread.is_live_store_job(stale)
    assert thread._saved_offset.get("req-a") is None
    assert "req-a" not in thread._skip_store_requests
    assert thread._get_retry_token_ids(stale) is None
    assert thread._get_retry_token_ids(live) == (32, list(range(32, 64)))


def test_store_recving_thread_reports_failed_block_ids():
    store = MagicMock()
    store.batch_get_into_multi_buffers.return_value = [256, -5, -7]
    thread = _make_store_recving_thread(store)

    thread._handle_request(
        _make_load_req(
            "req-a",
            [b"a0", b"a1", b"a2"],
            token_len=48,
        )
    )

    assert thread.get_and_clear_finished_requests() == {"req-a"}
    assert thread.get_and_clear_block_ids_with_load_errors() == {(0, 1), (0, 2)}
    assert thread.get_and_clear_block_ids_with_load_errors() == set()


def test_store_recving_thread_reports_failed_block_ids_after_rotation():
    store = MagicMock()
    store.batch_get_into_multi_buffers.return_value = [256, -5, 256]
    thread = _make_store_recving_thread(store, tp_rank=1)

    thread._handle_request(
        _make_load_req(
            "req-a",
            [b"a0", b"a1", b"a2"],
            token_len=48,
        )
    )

    assert thread.get_and_clear_block_ids_with_load_errors() == {(0, 2)}


def test_store_recving_thread_reports_all_attempted_blocks_on_exception():
    store = MagicMock()
    store.batch_get_into_multi_buffers.side_effect = RuntimeError("boom")
    thread = _make_store_recving_thread(store)

    thread._handle_request(
        _make_load_req(
            "req-a",
            [b"a0", b"a1", b"a2"],
            token_len=48,
        )
    )

    assert thread.get_and_clear_finished_requests() == {"req-a"}
    assert thread.get_and_clear_block_ids_with_load_errors() == {(0, 0), (0, 1), (0, 2)}


def test_store_worker_get_block_ids_with_load_errors_delegates_to_recv_thread():
    recv_thread = MagicMock()
    recv_thread.get_and_clear_block_ids_with_load_errors.return_value = {3, 4}
    w = _make_bare_worker()
    w.kv_recv_threads = [recv_thread]

    assert w.get_block_ids_with_load_errors() == {3, 4}
    recv_thread.get_and_clear_block_ids_with_load_errors.assert_called_once_with()


def test_store_sending_thread_passes_replicate_config_when_preferred_segment_set():
    store = MagicMock()
    store.batch_is_exist.side_effect = lambda keys: [0] * len(keys)
    store.batch_put_from_multi_buffers.return_value = [256, 256]
    replicate_config = SimpleNamespace(preferred_segment="10.0.0.7:50053")
    thread = _make_store_sending_thread(store, replicate_config=replicate_config)

    _run_store_req(thread, _make_store_req("req-a", [b"a0", b"a1"]))

    assert store.batch_put_from_multi_buffers.call_count == 1
    call_args = store.batch_put_from_multi_buffers.call_args.args
    assert len(call_args) == 4
    assert call_args[3] is replicate_config


def test_store_sending_thread_passes_default_replicate_config_when_no_preferred_segment():  # noqa: E501
    """Without a preferred_segment the SendingThread still forwards a
    (default-constructed) ReplicateConfig so the C++ side always sees a
    well-defined config object."""
    store = MagicMock()
    store.batch_is_exist.side_effect = lambda keys: [0] * len(keys)
    store.batch_put_from_multi_buffers.return_value = [256, 256]
    replicate_config = SimpleNamespace()
    thread = _make_store_sending_thread(store, replicate_config=replicate_config)

    _run_store_req(thread, _make_store_req("req-a", [b"a0", b"a1"]))

    assert store.batch_put_from_multi_buffers.call_count == 1
    call_args = store.batch_put_from_multi_buffers.call_args.args
    assert len(call_args) == 4
    assert call_args[3] is replicate_config


def test_group_id_detection_uses_class_attribute_without_instantiating():
    class ReplicateConfigWithClassGroupIdsAndRequiredInit:
        group_ids = None

        def __init__(self, required):
            self.group_ids = required

    assert worker._replicate_config_supports_group_ids(
        ReplicateConfigWithClassGroupIdsAndRequiredInit,
        object(),
    )


def test_group_id_detection_uses_instance_attribute_when_class_lacks_attribute():
    class ReplicateConfigWithInstanceGroupIds:
        def __init__(self) -> None:
            self.group_ids = None

    replicate_config = ReplicateConfigWithInstanceGroupIds()

    assert worker._replicate_config_supports_group_ids(
        ReplicateConfigWithInstanceGroupIds,
        replicate_config,
    )


def test_group_id_detection_rejects_old_replicate_config():
    class ReplicateConfigWithoutGroupIds:
        __slots__ = ()

    assert not worker._replicate_config_supports_group_ids(
        ReplicateConfigWithoutGroupIds,
        ReplicateConfigWithoutGroupIds(),
    )


def test_store_sending_thread_sets_group_ids_when_enabled():
    store = MagicMock()
    store.batch_is_exist.side_effect = lambda keys: [0] * len(keys)
    store.batch_put_from_multi_buffers.return_value = [256, 256]
    replicate_config = SimpleNamespace(group_ids=None)
    thread = _make_store_sending_thread(
        store,
        replicate_config=replicate_config,
        enable_group_semantics=True,
        supports_group_ids=True,
    )

    _run_store_req(thread, _make_store_req("req-a", [b"a0", b"a1"]))

    assert store.batch_put_from_multi_buffers.call_count == 1
    keys, _addrs, _sizes, config = store.batch_put_from_multi_buffers.call_args.args
    assert config is replicate_config
    assert config.group_ids == [
        "vllm-mooncake-store:test-model@6130",
        "vllm-mooncake-store:test-model@6131",
    ]
    assert len(config.group_ids) == len(keys)


def test_store_sending_thread_leaves_group_ids_unchanged_when_flag_disabled():
    store = MagicMock()
    store.batch_is_exist.side_effect = lambda keys: [0] * len(keys)
    store.batch_put_from_multi_buffers.return_value = [256, 256]
    replicate_config = SimpleNamespace(group_ids=["existing"])
    thread = _make_store_sending_thread(
        store,
        replicate_config=replicate_config,
        enable_group_semantics=False,
        supports_group_ids=True,
    )

    _run_store_req(thread, _make_store_req("req-a", [b"a0", b"a1"]))

    assert store.batch_put_from_multi_buffers.call_count == 1
    assert store.batch_put_from_multi_buffers.call_args.args[3] is replicate_config
    assert replicate_config.group_ids == ["existing"]


def test_store_sending_thread_leaves_group_ids_unchanged_when_unsupported():
    store = MagicMock()
    store.batch_is_exist.side_effect = lambda keys: [0] * len(keys)
    store.batch_put_from_multi_buffers.return_value = [256, 256]
    replicate_config = SimpleNamespace(group_ids=["existing"])
    thread = _make_store_sending_thread(
        store,
        replicate_config=replicate_config,
        enable_group_semantics=True,
        supports_group_ids=False,
    )

    _run_store_req(thread, _make_store_req("req-a", [b"a0", b"a1"]))

    assert store.batch_put_from_multi_buffers.call_count == 1
    assert store.batch_put_from_multi_buffers.call_args.args[3] is replicate_config
    assert replicate_config.group_ids == ["existing"]


def test_mooncake_group_id_uses_logical_chunk_and_excludes_physical_ranks():
    group_id = worker._make_mooncake_group_id(
        KeyMetadata(
            "test-model",
            tp_rank=3,
            pcp_rank=5,
            dcp_rank=7,
            pp_rank=11,
            group_id=13,
        ),
        "abcdef",
    )

    assert group_id == "vllm-mooncake-store:test-model@abcdef"
    assert "tp_rank" not in group_id
    assert "pp_rank" not in group_id
    assert "pcp" not in group_id
    assert "dcp" not in group_id
    assert "group:" not in group_id


def test_mooncake_group_id_uses_cache_prefix_namespace():
    metadata_a = KeyMetadata("test-model", 0, 0, 0, 0, cache_prefix="deployment-a")
    metadata_b = KeyMetadata("test-model", 0, 0, 0, 0, cache_prefix="deployment-b")

    group_id_a = worker._make_mooncake_group_id(metadata_a, "abcdef")
    group_id_b = worker._make_mooncake_group_id(metadata_b, "abcdef")

    assert group_id_a == "vllm-mooncake-store:deployment-a@test-model@abcdef"
    assert group_id_b == "vllm-mooncake-store:deployment-b@test-model@abcdef"
    assert group_id_a != group_id_b


def test_store_sending_thread_group_id_excludes_physical_sharding():
    store = MagicMock()
    store.batch_is_exist.return_value = [0, 0]
    store.batch_put_from_multi_buffers.return_value = [256, 256]
    replicate_config = SimpleNamespace(group_ids=None)
    db = _make_rank_local_db(
        KeyMetadata(
            "test-model",
            tp_rank=2,
            pcp_rank=0,
            dcp_rank=0,
            pp_rank=0,
            group_id=0,
        ),
        16,
        base_addr=0x1000,
    )
    thread = _make_store_sending_thread(
        store,
        token_databases=[db],
        tp_rank=2,
        replicate_config=replicate_config,
        enable_group_semantics=True,
        supports_group_ids=True,
    )

    _run_store_req(thread, _make_store_req("req-a", [b"a0", b"a1"]))

    keys, _addrs, _sizes, config = store.batch_put_from_multi_buffers.call_args.args
    assert keys == [
        "test-model@tp_rank:2@pcp0@dcp0@pp_rank:0@group:0@6130",
        "test-model@tp_rank:2@pcp0@dcp0@pp_rank:0@group:0@6131",
    ]
    assert config.group_ids == [
        "vllm-mooncake-store:test-model@6130",
        "vllm-mooncake-store:test-model@6131",
    ]


def test_store_sending_thread_group_ids_share_across_kv_cache_groups():
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec

    store = MagicMock()
    store.batch_is_exist.return_value = [0, 0, 0, 0]
    store.batch_put_from_multi_buffers.return_value = [256, 256, 300, 300]
    replicate_config = SimpleNamespace(group_ids=None)
    spec = FullAttentionSpec(block_size=16, num_kv_heads=8, head_size=64, dtype=None)
    coord = mooncake_store_worker.MooncakeStoreCoordinator(
        [
            KVCacheGroupSpec(["layer0"], spec),
            KVCacheGroupSpec(["layer1"], spec),
        ],
        scheduler_block_size=16,
        block_size=16,
    )
    db0 = _make_rank_local_db(
        KeyMetadata("test-model", 0, 0, 0, 0, group_id=0), 16, base_addr=0x1000
    )
    # Group 1 has its own geometry (two 300-byte rows per block), so its put
    # addresses use its own block stride.
    metadata1 = KeyMetadata("test-model", 0, 0, 0, 0, group_id=1)
    layout1 = RankLocalStoreLayout(metadata1, 16, geometry=GroupGeometry(2, 300))
    layout1.base_addr = 0x3000
    db1 = ChunkedTokenDatabase(metadata1, 16, store_layout=layout1)
    thread = _make_store_sending_thread(
        store,
        coord=coord,
        token_databases=[db0, db1],
        replicate_config=replicate_config,
        enable_group_semantics=True,
        supports_group_ids=True,
    )

    _run_store_req(thread, _make_multi_group_store_req("req-a", [b"a0", b"a1"]))

    keys, addrs, sizes, config = store.batch_put_from_multi_buffers.call_args.args
    assert keys == [
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@group:0@6130",
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@group:0@6131",
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@group:1@6130",
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@group:1@6131",
    ]
    assert addrs == [
        [0x1000 + 256],
        [0x1000 + 2 * 256],
        [0x3000 + 3 * 600],
        [0x3000 + 4 * 600],
    ]
    assert sizes == [[256], [256], [600], [600]]
    # Different vLLM KV cache groups for the same prefix chunk share the
    # same Mooncake lifecycle group id.
    assert config.group_ids == [
        "vllm-mooncake-store:test-model@6130",
        "vllm-mooncake-store:test-model@6131",
        "vllm-mooncake-store:test-model@6130",
        "vllm-mooncake-store:test-model@6131",
    ]
    assert config.group_ids[0] == config.group_ids[2]
    assert config.group_ids[1] == config.group_ids[3]


def test_store_sending_thread_group_ids_follow_missing_key_filter():
    store = MagicMock()
    store.batch_is_exist.return_value = [1, 0]
    store.batch_put_from_multi_buffers.return_value = [256]
    replicate_config = SimpleNamespace(group_ids=None)
    thread = _make_store_sending_thread(
        store,
        replicate_config=replicate_config,
        enable_group_semantics=True,
        supports_group_ids=True,
    )

    _run_store_req(thread, _make_store_req("req-a", [b"a0", b"a1"]))

    keys, _addrs, _sizes, config = store.batch_put_from_multi_buffers.call_args.args
    assert keys == ["test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@group:0@6131"]
    assert config.group_ids == ["vllm-mooncake-store:test-model@6131"]


def test_estimate_disk_offload_staging_bytes_sums_multi_segment_sizes():
    assert worker._estimate_disk_offload_staging_bytes([256, 512]) == 12288


def test_recv_thread_uses_single_batch_when_no_disk_offload_budget(monkeypatch):
    monkeypatch.delenv("VLLM_MOONCAKE_STORE_TIER_LOG", raising=False)
    store = MagicMock()
    store.batch_get_into_multi_buffers.return_value = [256, 256, 256]
    thread = _make_store_recving_thread(store, disk_offload_buffer_budget_bytes=None)

    req = _make_load_req(
        "req-a",
        [b"a0", b"a1", b"a2"],
        token_len=48,
    )

    thread._handle_request(req)

    assert store.batch_get_into_multi_buffers.call_count == 1
    keys, addrs, sizes = store.batch_get_into_multi_buffers.call_args.args
    assert keys == [
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@group:0@6130",
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@group:0@6131",
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@group:0@6132",
    ]
    assert sizes == [[256], [256], [256]]
    store.batch_get_replica_desc.assert_not_called()


def test_recv_thread_logs_tier_summary_when_enabled(monkeypatch, caplog_vllm):
    monkeypatch.setenv("VLLM_MOONCAKE_STORE_TIER_LOG", "1")
    caplog_vllm.set_level(logging.INFO, logger=worker.logger.name)

    store = MagicMock()
    store.batch_get_into_multi_buffers.return_value = [256, 256, -10]
    thread = _make_store_recving_thread(store, disk_offload_buffer_budget_bytes=None)

    req = _make_load_req(
        "req-a",
        [b"a0", b"a1", b"a2"],
        token_len=48,
    )
    expected_keys = [
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@group:0@6130",
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@group:0@6131",
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@group:0@6132",
    ]
    store.batch_get_replica_desc.return_value = {
        expected_keys[0]: [_ReplicaDesc("memory")],
        expected_keys[1]: [_ReplicaDesc("disk")],
        expected_keys[2]: [],
    }

    thread._handle_request(req)

    assert store.batch_get_replica_desc.call_args.args == (expected_keys,)
    assert store.method_calls[0][0] == "batch_get_replica_desc"
    assert store.method_calls[1][0] == "batch_get_into_multi_buffers"

    messages = [record.getMessage() for record in caplog_vllm.records]
    assert any(
        "Mooncake load tier summary" in message
        and "req_id=req-a" in message
        and "batch_keys=3" in message
        and "memory_keys=1" in message
        and "disk_keys=1" in message
        and "unknown_keys=1" in message
        and "success_keys=2" in message
        and "failed_keys=1" in message
        and "bytes_by_tier={'memory': 256, 'disk': 256, 'unknown': 0}" in message
        for message in messages
    )


def test_recv_thread_records_partial_failure_metrics(monkeypatch):
    monkeypatch.delenv("VLLM_MOONCAKE_STORE_TIER_LOG", raising=False)
    store = MagicMock()
    store.batch_get_into_multi_buffers.return_value = [256, -10]
    thread = _make_store_recving_thread(store, disk_offload_buffer_budget_bytes=None)
    stats = MooncakeStoreConnectorStats()
    thread._record_operation_cb = stats.record_operation

    req = _make_load_req(
        "req-a",
        [b"a0", b"a1"],
        token_len=32,
    )

    thread._handle_request(req)

    assert len(stats.data["load_get"]) == 1
    assert stats.data["load_get"][0]["num_keys"] == 2
    assert stats.data["load_get"][0]["num_bytes"] == 512
    assert stats.data["load_get"][0]["status"] == "partial_failure"
    assert stats.data["load_get"][0]["num_failed_keys"] == 1


def test_recv_thread_uses_ratio_scaled_budget_for_first_pass_split():
    store = MagicMock()
    store.batch_get_into_multi_buffers.side_effect = [
        [256],
        [256],
    ]
    thread = _make_store_recving_thread(
        store,
        disk_offload_buffer_budget_bytes=2 * _DISK_OFFLOAD_SINGLE_KEY_BYTES,
    )

    req = _make_load_req(
        "req-a",
        [b"a0", b"a1"],
        token_len=32,
    )

    thread._handle_request(req)

    assert store.batch_get_into_multi_buffers.call_count == 2
    first_keys = store.batch_get_into_multi_buffers.call_args_list[0].args[0]
    second_keys = store.batch_get_into_multi_buffers.call_args_list[1].args[0]
    assert first_keys == [
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@group:0@6130",
    ]
    assert second_keys == [
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@group:0@6131",
    ]


def test_recv_thread_splits_disk_offload_loads_by_budget():
    store = MagicMock()
    store.batch_get_into_multi_buffers.side_effect = [
        [256, 256],
        [256],
    ]
    thread = _make_store_recving_thread(
        store,
        disk_offload_buffer_budget_bytes=_DISK_OFFLOAD_BUDGET_FOR_SPLIT,
    )

    req = _make_load_req(
        "req-a",
        [b"a0", b"a1", b"a2"],
        token_len=48,
    )

    thread._handle_request(req)

    assert store.batch_get_into_multi_buffers.call_count == 2

    first_keys = store.batch_get_into_multi_buffers.call_args_list[0].args[0]
    second_keys = store.batch_get_into_multi_buffers.call_args_list[1].args[0]
    first_addrs = store.batch_get_into_multi_buffers.call_args_list[0].args[1]
    second_addrs = store.batch_get_into_multi_buffers.call_args_list[1].args[1]
    first_sizes = store.batch_get_into_multi_buffers.call_args_list[0].args[2]
    second_sizes = store.batch_get_into_multi_buffers.call_args_list[1].args[2]
    assert first_keys == [
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@group:0@6130",
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@group:0@6131",
    ]
    assert second_keys == [
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@group:0@6132",
    ]
    layout = thread.token_databases[0].store_layout
    assert isinstance(layout, RankLocalStoreLayout)
    base_addr = layout.base_addr
    block_len = layout.block_bytes
    assert first_addrs == [[base_addr], [base_addr + block_len]]
    assert second_addrs == [[base_addr + 2 * block_len]]
    expected_size = block_len
    assert first_sizes == [[expected_size], [expected_size]]
    assert second_sizes == [[expected_size]]


def test_recv_thread_stops_after_first_failing_disk_offload_sub_batch():
    store = MagicMock()
    store.batch_get_into_multi_buffers.return_value = [-10, -10]
    thread = _make_store_recving_thread(
        store,
        disk_offload_buffer_budget_bytes=_DISK_OFFLOAD_BUDGET_FOR_SPLIT,
    )

    req = _make_load_req(
        "req-a",
        [b"a0", b"a1", b"a2"],
        token_len=48,
    )

    thread._handle_request(req)

    assert store.batch_get_into_multi_buffers.call_count == 1
    # The skipped split's blocks are never fetched; they must be reported as
    # load errors too, or the request would resume onto uninitialized blocks.
    assert thread.get_and_clear_block_ids_with_load_errors() == {
        (0, 0),
        (0, 1),
        (0, 2),
    }


def test_recv_thread_completes_load_when_all_keys_masked():
    """A hit whose every chunk is already vLLM-cached (the prefix cache filled
    between lookup and load) has no keys to fetch; the load must complete
    instead of dividing by zero in the tp_rank rotation and hanging."""
    store = MagicMock()
    thread = _make_store_recving_thread(store)

    thread._handle_request(
        _make_load_req("req-a", [b"h0"], token_len=16, vllm_cached_tokens=16)
    )

    store.batch_get_into_multi_buffers.assert_not_called()
    assert thread.get_and_clear_block_ids_with_load_errors() == set()
    assert thread.get_and_clear_finished_requests() == {"req-a"}


def test_recv_thread_skips_split_when_budget_holds_all_keys():
    """PR-36 removed the count-based split trigger; with budget for 3 keys,
    all three should be requested in a single call."""
    store = MagicMock()
    store.batch_get_into_multi_buffers.return_value = [256, 256, 256]
    thread = _make_store_recving_thread(
        store,
        disk_offload_buffer_budget_bytes=_DISK_OFFLOAD_BUDGET_FOR_THREE_KEYS,
    )

    req = _make_load_req(
        "req-a",
        [b"a0", b"a1", b"a2"],
        token_len=48,
    )

    thread._handle_request(req)

    assert store.batch_get_into_multi_buffers.call_count == 1
    assert store.batch_get_into_multi_buffers.call_args_list[0].args[0] == [
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@group:0@6130",
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@group:0@6131",
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@group:0@6132",
    ]


def test_recv_thread_reports_unsplittable_key_larger_than_budget():
    # Non-zero tp_rank exercises the rotation: the oversized key may not sit
    # at the original request's first block. Every block must still be marked
    # invalid, since none are loaded.
    store = MagicMock()
    thread = _make_store_recving_thread(
        store,
        tp_rank=2,
        disk_offload_buffer_budget_bytes=_DISK_OFFLOAD_BUDGET_TOO_SMALL,
    )

    req = _make_load_req(
        "req-a",
        [b"a0", b"a1", b"a2"],
        token_len=48,
    )

    thread._handle_request(req)

    assert store.batch_get_into_multi_buffers.call_count == 0
    assert thread.get_and_clear_block_ids_with_load_errors() == {(0, 0), (0, 1), (0, 2)}


def test_recv_thread_partial_load_failure_reports_block_ids_without_hma():
    store = MagicMock()
    # First key fails, second succeeds.
    store.batch_get_into_multi_buffers.return_value = [-1, 0]
    thread = _make_store_recving_thread(store)

    thread._handle_request(_make_load_req("req-a", [b"h0", b"h1"], token_len=32))

    assert thread.get_and_clear_block_ids_with_load_errors() == {(0, 0)}
    assert thread.get_and_clear_failed_requests() == set()
    assert thread.get_and_clear_finished_requests() == {"req-a"}


def test_recv_thread_partial_load_failure_reports_request_with_hma():
    store = MagicMock()
    store.batch_get_into_multi_buffers.return_value = [-1, 0]
    thread = _make_store_recving_thread(store, is_hma_required=True)

    thread._handle_request(_make_load_req("req-a", [b"h0", b"h1"], token_len=32))

    assert thread.get_and_clear_block_ids_with_load_errors() == set()
    assert thread.get_and_clear_failed_requests() == {"req-a"}
    assert thread.get_and_clear_finished_requests() == {"req-a"}


def test_recv_thread_oversized_key_reports_request_with_hma():
    store = MagicMock()
    thread = _make_store_recving_thread(
        store,
        tp_rank=2,
        disk_offload_buffer_budget_bytes=_DISK_OFFLOAD_BUDGET_TOO_SMALL,
        is_hma_required=True,
    )

    thread._handle_request(_make_load_req("req-a", [b"a0", b"a1", b"a2"], token_len=48))

    assert store.batch_get_into_multi_buffers.call_count == 0
    assert thread.get_and_clear_block_ids_with_load_errors() == set()
    assert thread.get_and_clear_failed_requests() == {"req-a"}
    assert thread.get_and_clear_finished_requests() == {"req-a"}


def test_worker_init_excludes_nonprefix_cache_groups(monkeypatch):
    store = MagicMock()
    store.setup.return_value = 0
    _install_fake_mooncake(monkeypatch, store)
    _patch_worker_runtime(monkeypatch)
    monkeypatch.setattr(
        worker.MooncakeStoreConfig,
        "load_from_config",
        staticmethod(lambda: _make_config()),
    )
    vllm_config = _make_vllm_config()
    vllm_config.cache_config.block_size = 800
    vllm_config.cache_config.enable_prefix_caching = True

    store_worker = worker.MooncakeStoreWorker(
        vllm_config, _make_qsa_hybrid_kv_cache_config()
    )

    assert [
        group.kv_cache_spec.block_size for group in store_worker._kv_cache_groups
    ] == [800, 800]
    assert [db.block_size for db in store_worker.token_dbs] == [800, 800]


def _make_two_full_attention_groups_kv_cache_config():
    """Two full-attention groups with different head sizes: no hybrid
    layers, but block-level failure reporting is still unsupported."""
    return KVCacheConfig(
        num_pages=10,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full0"],
                FullAttentionSpec(
                    block_size=800, num_kv_heads=8, head_size=64, dtype=None
                ),
                num_blocks=10,
                blocks_per_page=1,
                kernel_rows_per_block=2,
                row_bytes=128,
            ),
            KVCacheGroupSpec(
                ["full1"],
                FullAttentionSpec(
                    block_size=800, num_kv_heads=8, head_size=128, dtype=None
                ),
                num_blocks=10,
                blocks_per_page=1,
                kernel_rows_per_block=2,
                row_bytes=128,
            ),
        ],
        block_size=800,
        large_page_bytes=256,
    )


@pytest.mark.parametrize(
    ("make_config", "expected_is_hma_required"),
    [
        (lambda: _make_qsa_hybrid_kv_cache_config(), True),
        (lambda: _make_two_full_attention_groups_kv_cache_config(), True),
        (lambda: _make_kv_cache_config(block_size=800), False),
    ],
    ids=["hybrid", "two-full-attention-groups", "single-group"],
)
def test_worker_is_hma_required_from_kv_cache_groups(
    monkeypatch, make_config, expected_is_hma_required
):
    store = MagicMock()
    store.setup.return_value = 0
    _install_fake_mooncake(monkeypatch, store)
    _patch_worker_runtime(monkeypatch)
    monkeypatch.setattr(
        worker.MooncakeStoreConfig,
        "load_from_config",
        staticmethod(lambda: _make_config()),
    )
    vllm_config = _make_vllm_config()
    vllm_config.cache_config.block_size = 800
    vllm_config.cache_config.enable_prefix_caching = True

    store_worker = worker.MooncakeStoreWorker(vllm_config, make_config())

    assert store_worker._is_hma_required is expected_is_hma_required


def test_requester_worker_init_uses_positional_setup(tmp_path, monkeypatch):
    store = MagicMock()
    store.setup.return_value = 0
    _install_fake_mooncake(monkeypatch, store)
    _patch_worker_runtime(monkeypatch)
    monkeypatch.setenv(
        "MOONCAKE_CONFIG_PATH",
        _write_mooncake_config(
            tmp_path,
            {
                "metadata_server": "http://metadata/endpoint",
                "global_segment_size": "4gb",
                "local_buffer_size": "64mb",
                "protocol": "rdma",
                "device_name": "mlx5_0",
                "master_server_address": "10.0.0.7:50051",
                "enable_offload": True,
            },
        ),
    )
    w = worker.MooncakeStoreWorker(_make_vllm_config(), _make_kv_cache_config())

    assert not hasattr(w, "_isolate_offload_resources")
    assert store.setup.call_args.args == (
        "10.0.0.7",
        "http://metadata/endpoint",
        4 * 1024 * 1024 * 1024,  # global_segment_size: "4gb" honored
        64 * 1024 * 1024,
        "rdma",
        "mlx5_0",
        "10.0.0.7:50051",
    )
    assert w.coord.retention_interval == 0


def test_requester_worker_init_prefers_local_hostname_override(
    tmp_path,
    monkeypatch,
):
    store = MagicMock()
    store.setup.return_value = 0
    _install_fake_mooncake(monkeypatch, store)
    _patch_worker_runtime(monkeypatch)
    monkeypatch.setenv("MOONCAKE_REQUESTER_LOCAL_HOSTNAME", "worker-a:50053")
    monkeypatch.setenv(
        "MOONCAKE_CONFIG_PATH",
        _write_mooncake_config(
            tmp_path,
            {
                "metadata_server": "http://metadata/endpoint",
                "local_buffer_size": "64mb",
                "protocol": "tcp",
                "device_name": "",
                "master_server_address": "10.0.0.7:50051",
            },
        ),
    )
    worker.MooncakeStoreWorker(_make_vllm_config(rank=1), _make_kv_cache_config())

    assert store.setup.call_args.args[0] == "worker-a:50053"


def test_requester_worker_init_skips_disk_budget_when_offload_disabled(
    tmp_path,
    monkeypatch,
):
    """enable_offload=False zeroes out the disk budget so we don't generate
    redundant owner GET-RPCs."""
    store = MagicMock()
    store.setup.return_value = 0
    _install_fake_mooncake(monkeypatch, store)
    _patch_worker_runtime(monkeypatch)
    monkeypatch.setenv(
        "MOONCAKE_CONFIG_PATH",
        _write_mooncake_config(
            tmp_path,
            {
                "metadata_server": "http://metadata/endpoint",
                "protocol": "tcp",
                "device_name": "",
                "master_server_address": "10.0.0.7:50051",
                "enable_offload": False,
            },
        ),
    )
    w = worker.MooncakeStoreWorker(_make_vllm_config(), _make_kv_cache_config())

    assert w.disk_offload_buffer_budget_bytes is None


def test_worker_enables_lbhnc_store_tp_layout(tmp_path, monkeypatch):
    store = MagicMock()
    store.setup.return_value = 0
    _install_fake_mooncake(monkeypatch, store)
    _patch_worker_runtime(monkeypatch, tp_size=2)
    monkeypatch.setattr(_FakeModelConfig, "get_total_num_kv_heads", lambda _self: 8)
    monkeypatch.setenv(
        "MOONCAKE_CONFIG_PATH",
        _write_mooncake_config(
            tmp_path,
            {
                "metadata_server": "http://metadata/endpoint",
                "protocol": "tcp",
                "master_server_address": "10.0.0.7:50051",
            },
        ),
    )

    w = worker.MooncakeStoreWorker(
        _make_vllm_config(extra_config={"store_tp_size": 4}),
        _make_kv_cache_config(),
    )

    assert w.store_tp_size == 4
    assert w.token_dbs[0].metadata.store_namespace == _TP_SHARED_NAMESPACE
    assert isinstance(w.token_dbs[0].store_layout, LBHNCStoreLayout)
    assert w.token_dbs[0].store_layout.local_shard_ids == (0, 1)
    assert len(w._lookup_key_prefixes[0]) == 4


def test_worker_enables_lbnhc_store_tp_layout(tmp_path, monkeypatch):
    store = MagicMock()
    store.setup.return_value = 0
    _install_fake_mooncake(monkeypatch, store)
    _patch_worker_runtime(monkeypatch, tp_size=2)
    monkeypatch.setattr(_FakeModelConfig, "get_total_num_kv_heads", lambda _self: 8)
    monkeypatch.setenv(
        "MOONCAKE_CONFIG_PATH",
        _write_mooncake_config(
            tmp_path,
            {
                "metadata_server": "http://metadata/endpoint",
                "protocol": "tcp",
                "master_server_address": "10.0.0.7:50051",
            },
        ),
    )

    with patch.object(worker.logger, "warning_once") as warning_once:
        w = worker.MooncakeStoreWorker(
            _make_vllm_config(
                extra_config={"store_tp_size": 4},
                kv_cache_layout=KVCacheLayout.LBNHC,
            ),
            _make_kv_cache_config(),
        )

    warning_once.assert_called_once()
    assert w.store_tp_size == 4
    assert w.token_dbs[0].metadata.store_namespace == _TP_SHARED_LBNHC_NAMESPACE
    assert isinstance(w.token_dbs[0].store_layout, LBNHCStoreLayout)
    assert w.token_dbs[0].store_layout.local_shard_ids == (0, 1)
    assert len(w._lookup_key_prefixes[0]) == 4


def test_tp_sharded_layout_rejects_full_attention_subclasses():
    spec = RSWASpec(
        block_size=16,
        num_kv_heads=8,
        head_size=64,
        dtype=None,
        rswa_window=32,
    )
    store_worker = object.__new__(worker.MooncakeStoreWorker)
    store_worker.pcp_size = 1
    store_worker.dcp_size = 1
    store_worker._kv_cache_groups = [SimpleNamespace(kv_cache_spec=spec)]

    assert not store_worker._supports_tp_sharded_store_layout(LBHNCStoreLayout, {})


@pytest.mark.parametrize(
    ("extra_config", "expected"),
    [
        ({"store_tp_size": 4}, 4),
        (
            {
                "enable_store_tp_lcm": True,
                "prefill_tp_sizes": [3, 4, 6],
            },
            12,
        ),
        (
            {
                "enable_store_tp_lcm": False,
                "prefill_tp_sizes": [4, 2],
            },
            None,
        ),
        ({"enable_store_tp_lcm": True, "prefill_tp_sizes": []}, None),
        ({"enable_store_tp_lcm": True, "prefill_tp_sizes": [4, 0]}, None),
        ({"enable_store_tp_lcm": True, "prefill_tp_sizes": [4, 2.0]}, None),
        ({"store_tp_size": 0}, None),
        ({"store_tp_size": -1}, None),
        ({"store_tp_size": True}, None),
        ({"store_tp_size": 4.0}, None),
    ],
)
def test_resolve_store_tp_size(extra_config: dict[str, object], expected: int | None):
    assert worker.resolve_store_tp_size(extra_config) == expected


def test_lcm_store_tp_gives_prefill_and_decode_common_namespace(tmp_path, monkeypatch):
    store = MagicMock()
    store.setup.return_value = 0
    _install_fake_mooncake(monkeypatch, store)
    monkeypatch.setattr(_FakeModelConfig, "get_total_num_kv_heads", lambda _self: 8)
    monkeypatch.setenv(
        "MOONCAKE_CONFIG_PATH",
        _write_mooncake_config(
            tmp_path,
            {
                "metadata_server": "http://metadata/endpoint",
                "protocol": "tcp",
                "master_server_address": "10.0.0.7:50051",
            },
        ),
    )
    lcm_config = {
        "enable_store_tp_lcm": True,
        "prefill_tp_sizes": [4, 2],
    }

    workers = []
    topologies = [
        (4, 2, "kv_both", (2,)),
        (2, 1, "kv_both", (2, 3)),
        (2, 1, "kv_consumer", (2, 3)),
    ]
    for tp_size, tp_rank, kv_role, expected_shards in topologies:
        _patch_worker_runtime(
            monkeypatch,
            tp_size=tp_size,
            tp_rank=tp_rank,
        )
        extra_config = dict(lcm_config)
        if kv_role == "kv_consumer":
            extra_config["save_decode_cache"] = True
        store_worker = worker.MooncakeStoreWorker(
            _make_vllm_config(extra_config=extra_config, kv_role=kv_role),
            _make_kv_cache_config(),
        )
        assert store_worker.store_tp_size == 4
        assert isinstance(store_worker.token_dbs[0].store_layout, LBHNCStoreLayout)
        assert store_worker.token_dbs[0].store_layout.local_shard_ids == expected_shards
        workers.append(store_worker)

    shard_keys = [
        store_worker.token_dbs[0].store_layout.key_for(2, BlockHash(b"h"))
        for store_worker in workers
    ]
    assert len(set(shard_keys)) == 1
    assert workers[-1].can_put


@pytest.mark.parametrize(
    ("tp_size", "prefill_tp_sizes"),
    [
        (3, [4, 2]),
        (2, [4, 3]),
        (2, []),
        (2, [4, 0]),
    ],
)
def test_lcm_store_tp_falls_back_when_topology_is_incompatible(
    tmp_path,
    monkeypatch,
    tp_size: int,
    prefill_tp_sizes: list[int],
):
    store = MagicMock()
    store.setup.return_value = 0
    _install_fake_mooncake(monkeypatch, store)
    _patch_worker_runtime(monkeypatch, tp_size=tp_size)
    monkeypatch.setattr(_FakeModelConfig, "get_total_num_kv_heads", lambda _self: 8)
    monkeypatch.setenv(
        "MOONCAKE_CONFIG_PATH",
        _write_mooncake_config(
            tmp_path,
            {
                "metadata_server": "http://metadata/endpoint",
                "protocol": "tcp",
                "master_server_address": "10.0.0.7:50051",
            },
        ),
    )

    store_worker = worker.MooncakeStoreWorker(
        _make_vllm_config(
            extra_config={
                "enable_store_tp_lcm": True,
                "prefill_tp_sizes": prefill_tp_sizes,
            }
        ),
        _make_kv_cache_config(),
    )

    assert store_worker.store_tp_size is None
    assert type(store_worker.token_dbs[0]) is ChunkedTokenDatabase
    assert "@store_format:rank_local_tp" in store_worker.token_dbs[0].key_for(
        BlockHash(b"h")
    )


@pytest.mark.parametrize("store_tp_size", [1, 3, 6])
def test_worker_keeps_rank_local_layout_for_unsupported_store_tp(
    tmp_path,
    monkeypatch,
    store_tp_size: int,
):
    store = MagicMock()
    store.setup.return_value = 0
    _install_fake_mooncake(monkeypatch, store)
    _patch_worker_runtime(monkeypatch, tp_size=2)
    monkeypatch.setattr(_FakeModelConfig, "get_total_num_kv_heads", lambda _self: 8)
    monkeypatch.setenv(
        "MOONCAKE_CONFIG_PATH",
        _write_mooncake_config(
            tmp_path,
            {
                "metadata_server": "http://metadata/endpoint",
                "protocol": "tcp",
                "master_server_address": "10.0.0.7:50051",
            },
        ),
    )

    w = worker.MooncakeStoreWorker(
        _make_vllm_config(
            extra_config={"store_tp_size": store_tp_size},
            kv_cache_layout=KVCacheLayout.LBNHC,
        ),
        _make_kv_cache_config(),
    )

    assert w.store_tp_size is None
    assert type(w.token_dbs[0]) is ChunkedTokenDatabase
    key = w.token_dbs[0].key_for(BlockHash(b"h"))
    assert "@store_tp:" not in key
    assert "@store_pp:1" in key
    assert "@store_format:rank_local_tp2_layout_LBNHC_lp" in key


def test_store_tp_opt_in_isolates_different_pp_sizes(tmp_path, monkeypatch):
    store = MagicMock()
    store.setup.return_value = 0
    _install_fake_mooncake(monkeypatch, store)
    _patch_worker_runtime(monkeypatch, tp_size=2)
    monkeypatch.setattr(_FakeModelConfig, "get_total_num_kv_heads", lambda _self: 8)
    monkeypatch.setenv(
        "MOONCAKE_CONFIG_PATH",
        _write_mooncake_config(
            tmp_path,
            {
                "metadata_server": "http://metadata/endpoint",
                "protocol": "tcp",
                "master_server_address": "10.0.0.7:50051",
            },
        ),
    )

    keys = []
    for pp_size in (1, 2):
        w = worker.MooncakeStoreWorker(
            _make_vllm_config(
                extra_config={"store_tp_size": 4},
                pipeline_parallel_size=pp_size,
            ),
            _make_kv_cache_config(),
        )
        layout = w.token_dbs[0].store_layout
        keys.append(layout.key_for(layout.local_shard_ids[0], BlockHash(b"h")))

    assert "@store_pp:1" in keys[0]
    assert "@store_pp:2" in keys[1]
    assert keys[0] != keys[1]


def test_save_decode_cache_keeps_transfer_path_enabled(tmp_path, monkeypatch):
    store = MagicMock()
    store.setup.return_value = 0
    _install_fake_mooncake(monkeypatch, store)
    _patch_worker_runtime(monkeypatch)
    monkeypatch.setenv(
        "MOONCAKE_CONFIG_PATH",
        _write_mooncake_config(
            tmp_path,
            {
                "metadata_server": "http://metadata/endpoint",
                "protocol": "tcp",
                "device_name": "",
                "master_server_address": "10.0.0.7:50051",
            },
        ),
    )

    w = worker.MooncakeStoreWorker(
        _make_vllm_config(
            kv_role="kv_consumer",
            extra_config={
                "enable_lookup": False,
                "save_decode_cache": True,
            },
        ),
        _make_kv_cache_config(),
    )

    assert w.can_put is True
    assert w._capacity_only is False


def test_requester_worker_init_builds_replicate_config_for_preferred_segment(
    tmp_path,
    monkeypatch,
):
    store = MagicMock()
    store.setup.return_value = 0
    fake_replicate_config_cls = _install_fake_mooncake(monkeypatch, store)
    _patch_worker_runtime(monkeypatch)
    monkeypatch.setenv(
        "MOONCAKE_CONFIG_PATH",
        _write_mooncake_config(
            tmp_path,
            {
                "metadata_server": "http://metadata/endpoint",
                "protocol": "tcp",
                "device_name": "",
                "master_server_address": "10.0.0.7:50051",
            },
        ),
    )
    w = worker.MooncakeStoreWorker(
        _make_vllm_config(
            extra_config={
                "preferred_segment": "10.0.0.7:50053",
            }
        ),
        _make_kv_cache_config(),
    )

    assert isinstance(w.store_replicate_config, fake_replicate_config_cls)
    assert w.store_replicate_config.preferred_segment == "10.0.0.7:50053"


def test_worker_scales_uniform_attention_group_under_dcp(tmp_path, monkeypatch):
    from vllm.v1.kv_cache_interface import (
        KVCacheConfig,
    )

    store = MagicMock()
    store.setup.return_value = 0
    _install_fake_mooncake(monkeypatch, store)
    _patch_worker_runtime(monkeypatch, tp_size=4, dcp_size=4)
    monkeypatch.setenv(
        "MOONCAKE_CONFIG_PATH",
        _write_mooncake_config(
            tmp_path,
            {
                "metadata_server": "http://metadata/endpoint",
                "protocol": "tcp",
                "device_name": "",
                "master_server_address": "10.0.0.7:50051",
            },
        ),
    )
    spec = FullAttentionSpec(block_size=16, num_kv_heads=8, head_size=64, dtype=None)
    kv_cache_config = KVCacheConfig(
        num_pages=10,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer0", "layer1"],
                spec,
                num_blocks=10,
                blocks_per_page=1,
                kernel_rows_per_block=2,
                row_bytes=128,
            )
        ],
        # DCP-sharded specs hold B // dcp tokens per rank.
        block_size=64,
        large_page_bytes=256,
    )

    w = worker.MooncakeStoreWorker(
        _make_vllm_config(decode_context_parallel_size=4),
        kv_cache_config,
    )

    scaled_spec = w._kv_cache_groups[0].kv_cache_spec
    assert isinstance(scaled_spec, FullAttentionSpec)
    assert scaled_spec.block_size == 64
    assert w.coord.attention_groups[0].spec.block_size == 64
    assert w.token_dbs[0].block_size == 64


@pytest.mark.parametrize("dcp_size", [1, 4])
def test_worker_put_striding_covers_every_rank_get_namespace(
    tmp_path, monkeypatch, dcp_size
):
    """Every key a rank GETs must have been PUT by some rank.

    When num_kv_head < tp_size, ranks holding the same KV heads stripe
    their PUTs across one shared key namespace. That dedup is only valid
    when those ranks really share a namespace: with DCP > 1 each rank GETs
    every key from its own ``@dcpN`` namespace, so striding must be
    disabled.
    """
    tp_size = 4
    store = MagicMock()
    store.setup.return_value = 0
    _install_fake_mooncake(monkeypatch, store)
    monkeypatch.setenv(
        "MOONCAKE_CONFIG_PATH",
        _write_mooncake_config(
            tmp_path,
            {
                "metadata_server": "http://metadata/endpoint",
                "protocol": "tcp",
                "device_name": "",
                "master_server_address": "10.0.0.7:50051",
            },
        ),
    )

    # _FakeModelConfig has num_kv_head=1 < tp_size, which enables striding.
    num_chunks = 4
    put_keys: set[str] = set()
    get_keys_per_rank: dict[int, set[str]] = {}
    for tp_rank in range(tp_size):
        _patch_worker_runtime(
            monkeypatch, tp_rank=tp_rank, tp_size=tp_size, dcp_size=dcp_size
        )
        w = worker.MooncakeStoreWorker(
            _make_vllm_config(rank=tp_rank, decode_context_parallel_size=dcp_size),
            _make_kv_cache_config(dcp_size=dcp_size),
        )
        db = w.token_dbs[0]
        token_len = num_chunks * db.block_size
        block_hashes = [f"hash-{i}".encode() for i in range(token_len // db.block_size)]
        keys = [
            PoolKey(db.metadata, block_hash.hex()).to_string()
            for _, _, block_hash in db.process_tokens(token_len, block_hashes)
        ]
        assert len(keys) == num_chunks
        # PUT side: mirrors KVCacheStoreSendingThread's striding slice.
        put_step = w._group_tp_replication_factors[0]
        put_keys.update(keys[w.tp_rank % put_step :: put_step])
        # GET side: KVCacheStoreRecvingThread fetches every key.
        get_keys_per_rank[tp_rank] = set(keys)

    for tp_rank, rank_keys in get_keys_per_rank.items():
        missing = rank_keys - put_keys
        assert not missing, (
            f"tp_rank={tp_rank} would GET {len(missing)}/{len(rank_keys)} keys "
            f"that no rank PUT (Mooncake OBJECT_NOT_FOUND): {sorted(missing)}"
        )


def test_mqa_p4_to_d2_uses_shared_rank_zero_namespace(tmp_path, monkeypatch):
    store = MagicMock()
    store.setup.return_value = 0
    _install_fake_mooncake(monkeypatch, store)
    monkeypatch.setenv(
        "MOONCAKE_CONFIG_PATH",
        _write_mooncake_config(
            tmp_path,
            {
                "metadata_server": "http://metadata/endpoint",
                "protocol": "tcp",
                "master_server_address": "10.0.0.7:50051",
            },
        ),
    )

    block_hash = BlockHash(b"hash")
    put_keys: set[str] = set()
    get_keys: set[str] = set()
    for tp_size, target in ((4, put_keys), (2, get_keys)):
        for tp_rank in range(tp_size):
            _patch_worker_runtime(monkeypatch, tp_rank=tp_rank, tp_size=tp_size)
            w = worker.MooncakeStoreWorker(
                _make_vllm_config(
                    rank=tp_rank,
                    extra_config={"store_tp_size": 4},
                ),
                _make_kv_cache_config(),
            )

            assert w.store_tp_size is None
            assert w.token_dbs[0].metadata.tp_rank == 0
            key = w.token_dbs[0].key_for(block_hash)
            if tp_size == 4:
                put_step = w._group_tp_replication_factors[0]
                if tp_rank % put_step == 0:
                    target.add(key)
            else:
                target.add(key)

    assert (
        put_keys
        == get_keys
        == {
            (
                "test-model@store_pp:1@store_format:tp_shared_mqa_lp"
                "@tp_rank:0@pcp0@dcp0@pp_rank:0@group:0@68617368"
            )
        }
    )


def test_requester_worker_group_semantics_falls_back_without_group_ids(
    tmp_path,
    monkeypatch,
):
    store = MagicMock()
    store.setup.return_value = 0
    _install_fake_mooncake(monkeypatch, store)
    _patch_worker_runtime(monkeypatch)
    warning = MagicMock()
    monkeypatch.setattr(worker.logger, "warning", warning)
    monkeypatch.setenv(
        "MOONCAKE_CONFIG_PATH",
        _write_mooncake_config(
            tmp_path,
            {
                "metadata_server": "http://metadata/endpoint",
                "protocol": "tcp",
                "device_name": "",
                "master_server_address": "10.0.0.7:50051",
            },
        ),
    )

    w = worker.MooncakeStoreWorker(
        _make_vllm_config(
            extra_config={
                "enable_group_semantics": True,
            }
        ),
        _make_kv_cache_config(),
    )

    assert w.enable_group_semantics is True
    assert w._supports_group_ids is False
    assert any(
        "does not support ReplicateConfig.group_ids" in call.args[0]
        for call in warning.call_args_list
    )


def test_requester_worker_group_semantics_string_false_does_not_enable(
    tmp_path,
    monkeypatch,
):
    store = MagicMock()
    store.setup.return_value = 0
    _install_fake_mooncake(monkeypatch, store)
    _patch_worker_runtime(monkeypatch)
    warning = MagicMock()
    monkeypatch.setattr(worker.logger, "warning", warning)
    monkeypatch.setenv(
        "MOONCAKE_CONFIG_PATH",
        _write_mooncake_config(
            tmp_path,
            {
                "metadata_server": "http://metadata/endpoint",
                "protocol": "tcp",
                "device_name": "",
                "master_server_address": "10.0.0.7:50051",
            },
        ),
    )

    w = worker.MooncakeStoreWorker(
        _make_vllm_config(
            extra_config={
                "enable_group_semantics": "false",
            }
        ),
        _make_kv_cache_config(),
    )

    assert w.enable_group_semantics is False
    assert not any(
        "does not support ReplicateConfig.group_ids" in call.args[0]
        for call in warning.call_args_list
    )


def test_requester_worker_group_semantics_string_true_enables(
    tmp_path,
    monkeypatch,
):
    store = MagicMock()
    store.setup.return_value = 0

    class FakeReplicateConfig:
        def __init__(self) -> None:
            self.group_ids = None

    fake_store_module = types.ModuleType("mooncake.store")
    fake_store_module.MooncakeDistributedStore = lambda: store  # type: ignore[attr-defined]
    fake_store_module.ReplicateConfig = FakeReplicateConfig  # type: ignore[attr-defined]
    fake_store_module.ObjectDataType = SimpleNamespace(TENSOR=1)  # type: ignore[attr-defined]
    fake_mooncake_module = types.ModuleType("mooncake")
    fake_mooncake_module.store = fake_store_module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mooncake", fake_mooncake_module)
    monkeypatch.setitem(sys.modules, "mooncake.store", fake_store_module)
    _patch_worker_runtime(monkeypatch)
    monkeypatch.setenv(
        "MOONCAKE_CONFIG_PATH",
        _write_mooncake_config(
            tmp_path,
            {
                "metadata_server": "http://metadata/endpoint",
                "protocol": "tcp",
                "device_name": "",
                "master_server_address": "10.0.0.7:50051",
            },
        ),
    )

    w = worker.MooncakeStoreWorker(
        _make_vllm_config(
            extra_config={
                "enable_group_semantics": "true",
            }
        ),
        _make_kv_cache_config(),
    )

    assert w.enable_group_semantics is True
    assert w._supports_group_ids is True


# ---------------------------------------------------------------------------
# Helpers for register_kv_caches tests
# ---------------------------------------------------------------------------


def test_store_sending_thread_clamps_token_len_to_lcm():
    """Partial chunks past the last lcm boundary aren't stored — cache hits
    are always lcm-aligned (mirrors HybridKVCacheCoordinator)."""
    store = MagicMock()
    store.batch_is_exist.return_value = [0, 0]
    store.batch_put_from_multi_buffers.return_value = [256, 256]
    # Default coord: single full-attn block_size=16, lcm=16.
    # token_len_chunk=33 clamps to 32 → 2 chunks (not 3 with a partial 1-token chunk).
    thread = _make_store_sending_thread(store)

    _run_store_req(
        thread,
        ReqMeta(
            req_id="r0",
            token_len_chunk=33,
            block_ids=([1, 2, 3],),
            block_hashes=[b"a0", b"a1", b"a2"],
            can_save=True,
        ),
    )

    keys = store.batch_put_from_multi_buffers.call_args.args[0]
    assert len(keys) == 2


def test_store_sending_thread_skips_when_token_len_below_lcm():
    """Requests shorter than lcm_block_size cannot produce any aligned chunk,
    so neither the existence check nor the put should be issued."""
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec

    store = MagicMock()
    # lcm=64 via single full-attn block_size=64.
    spec = FullAttentionSpec(block_size=64, num_kv_heads=8, head_size=64, dtype=None)
    coord = mooncake_store_worker.MooncakeStoreCoordinator(
        [KVCacheGroupSpec(["L"], spec)],
        scheduler_block_size=64,
        block_size=64,
    )
    db = _make_rank_local_db(
        KeyMetadata("test-model", 0, 0, 0, 0, group_id=0),
        64,
        base_addr=0x1000,
        block_bytes=1024,
    )
    thread = _make_store_sending_thread(
        store, coord=coord, token_databases=[db], block_size=64
    )

    _run_store_req(
        thread,
        ReqMeta(
            req_id="r0",
            token_len_chunk=32,
            block_ids=([1, 2],),
            block_hashes=[b"a0", b"a1"],
            can_save=True,
        ),
    )

    store.batch_is_exist.assert_not_called()
    store.batch_put_from_multi_buffers.assert_not_called()


def test_store_sending_thread_kv_events_use_group_chunk_metadata():
    from vllm.v1.core.kv_cache_utils import BlockHash, maybe_convert_block_hash
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheGroupSpec,
        SlidingWindowSpec,
    )

    store = MagicMock()
    store.batch_is_exist.side_effect = lambda keys: [0] * len(keys)
    store.batch_put_from_multi_buffers.return_value = [256, 256]

    full_spec = FullAttentionSpec(
        block_size=32, num_kv_heads=8, head_size=64, dtype=None
    )
    swa_spec = SlidingWindowSpec(
        block_size=32,
        num_kv_heads=8,
        head_size=64,
        dtype=None,
        sliding_window=32,
    )
    coord = mooncake_store_worker.MooncakeStoreCoordinator(
        [KVCacheGroupSpec(["L0"], full_spec), KVCacheGroupSpec(["L1"], swa_spec)],
        scheduler_block_size=32,
        block_size=32,
    )

    db_full = _make_rank_local_db(
        KeyMetadata("test-model", 0, 0, 0, 0, group_id=0),
        32,
        base_addr=0x1000,
        block_bytes=512,
    )
    db_swa = _make_rank_local_db(
        KeyMetadata("test-model", 0, 0, 0, 0, group_id=1),
        32,
        base_addr=0x2000,
        block_bytes=128,
    )

    thread = _make_store_sending_thread(
        store,
        coord=coord,
        token_databases=[db_full, db_swa],
        block_size=32,
    )
    thread.enable_kv_event = True

    hs = [bytes([1]) * 4]
    _run_store_req(
        thread,
        ReqMeta(
            req_id="r0",
            token_len_chunk=32,
            block_ids=([1], [2]),
            block_hashes=hs,
            can_save=True,
            token_ids=list(range(32)),
        ),
    )

    full_event, swa_event = thread.get_kv_events()
    assert full_event.group_idx == 0
    assert full_event.block_size == 32
    assert full_event.token_ids == list(range(32))
    assert full_event.block_hashes == [maybe_convert_block_hash(BlockHash(hs[0]))]

    assert swa_event.group_idx == 1
    assert swa_event.block_size == 32
    assert swa_event.token_ids == list(range(32))
    assert swa_event.block_hashes == [maybe_convert_block_hash(BlockHash(hs[0]))]


def _make_event_store_req(token_len: int, token_ids_start: int = 0) -> ReqMeta:
    num_blocks = token_len // 16
    return ReqMeta(
        req_id="r0",
        token_len_chunk=token_len,
        block_ids=(list(range(1, num_blocks + 1)),),
        block_hashes=[f"a{i}".encode() for i in range(num_blocks)],
        can_save=True,
        token_ids=list(range(token_ids_start, token_len)),
        token_ids_start=token_ids_start,
    )


@pytest.mark.parametrize(
    ("saved_offset", "put_step", "exists", "stored_indices", "parent_indices"),
    [
        pytest.param(16, 1, [0, 0, 0], [1, 2, 3], [0, 1, 2], id="suffix"),
        pytest.param(0, 1, [1, 0, 1, 0], [1, 3], [0, 2], id="dedup-holes"),
        pytest.param(0, 2, [0, 0], [0, 2], [None, 1], id="tp-stride"),
    ],
)
def test_store_sending_thread_kv_events_use_request_chain_parents(
    saved_offset, put_step, exists, stored_indices, parent_indices
):
    store = MagicMock()
    store.batch_is_exist.return_value = exists
    store.batch_put_from_multi_buffers.return_value = [256] * len(stored_indices)
    thread = _make_store_sending_thread(store, put_step=put_step)
    thread.enable_kv_event = True

    thread._saved_offset["r0"] = saved_offset
    _run_store_req(thread, _make_event_store_req(64, saved_offset))

    events = thread.get_kv_events()
    assert [event.block_hashes for event in events] == [
        [maybe_convert_block_hash(BlockHash(f"a{i}".encode()))] for i in stored_indices
    ]
    assert [event.parent_block_hash for event in events] == [
        (None if i is None else maybe_convert_block_hash(BlockHash(f"a{i}".encode())))
        for i in parent_indices
    ]
    assert thread._retry_token_ids == {}


def test_store_sending_thread_kv_events_retry_without_covered_tokens():
    store = MagicMock()
    store.batch_is_exist.return_value = [0, 0]
    store.batch_put_from_multi_buffers.return_value = [256, 256]
    thread = _make_store_sending_thread(store)
    thread.enable_kv_event = True

    _run_store_req(thread, _make_event_store_req(32, 16))

    retry_event, suffix_event = thread.get_kv_events()
    assert retry_event.token_ids == []
    assert suffix_event.token_ids == list(range(16, 32))


def test_store_sending_thread_kv_events_recover_suffix_after_put_failure():
    store = MagicMock()
    store.batch_is_exist.side_effect = ([0], [0, 0])
    store.batch_put_from_multi_buffers.side_effect = ([-1], [256, 256])
    thread = _make_store_sending_thread(store)
    thread.enable_kv_event = True

    _run_store_req(thread, _make_event_store_req(16))

    assert thread.get_kv_events() == []
    assert thread._retry_token_ids["r0"] == (0, list(range(16)))

    _run_store_req(thread, _make_event_store_req(32, 16))

    retry_event, suffix_event = thread.get_kv_events()
    assert retry_event.token_ids == list(range(16))
    assert suffix_event.token_ids == list(range(16, 32))
    assert thread._retry_token_ids == {}
    assert thread._saved_offset["r0"] == 32


def _auto_set_ready_event(*args, **kwargs):
    """Side effect for mocked thread constructors that auto-sets ready_event."""
    for arg in args:
        if isinstance(arg, threading.Event):
            arg.set()
    for val in kwargs.values():
        if isinstance(val, threading.Event):
            val.set()
    return MagicMock()


def _register_with_mocked_threads(
    worker: mooncake_store_worker.MooncakeStoreWorker,
    kv_caches: dict[str, torch.Tensor],
) -> None:
    """Call register_kv_caches with the I/O transfer threads mocked out."""
    prefix = "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.worker."
    with (
        patch(prefix + "KVCacheStoreSendingThread", side_effect=_auto_set_ready_event),
        patch(prefix + "KVCacheStoreRecvingThread", side_effect=_auto_set_ready_event),
    ):
        worker.register_kv_caches(kv_caches)


def _refresh_group_tp_replication_factors(
    worker: mooncake_store_worker.MooncakeStoreWorker,
) -> None:
    worker._group_tp_replication_factors = (
        worker._compute_group_tp_replication_factors()
    )
    worker._init_lookup_key_prefixes()


def _make_bare_worker(
    *,
    num_gpu_blocks: int = 10,
    block_size: int = 16,
    kv_role: str = "kv_both",
    save_decode_cache: bool = False,
) -> mooncake_store_worker.MooncakeStoreWorker:
    """Construct a MooncakeStoreWorker via __new__, bypassing __init__.

    Sets only the attributes that register_kv_caches() reads so we can
    test the stride-based layout detection without a real
    MooncakeDistributedStore.
    """
    worker = object.__new__(mooncake_store_worker.MooncakeStoreWorker)
    worker.cache_config = MagicMock()
    worker.cache_config.num_gpu_blocks = num_gpu_blocks
    worker.store = MagicMock()
    worker.store.register_buffer.return_value = 0
    worker.kv_role = kv_role
    worker.can_put = kv_role in ("kv_producer", "kv_both") or save_decode_cache
    worker._capacity_only = False
    worker.scheduler_block_size = block_size
    worker._is_hma_required = False
    worker.tp_rank = 0
    worker.enable_kv_events = False
    worker.load_async = True
    worker.kv_send_thread = None
    worker.kv_recv_threads = []
    worker.num_recv_threads = 1
    worker.recv_request_queue = queue.Queue()
    worker.finished_store_req = set()
    worker.tp_size = 1
    worker.store_tp_size = None
    worker.num_kv_head = 1
    worker.pp_size = 1
    # Minimal single-full-attention-group config so the coordinator-based
    # lookup path works (the connector no longer carries a legacy single-group
    # path; everything flows through the coordinator).
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheConfig,
        KVCacheGroupSpec,
        KVCacheTensor,
    )

    worker.disk_offload_buffer_budget_bytes = None
    worker.store_replicate_config = SimpleNamespace()
    worker.enable_group_semantics = False
    worker._supports_group_ids = False
    worker._kv_connector_stats_lock = threading.Lock()
    worker.kv_connector_stats = MooncakeStoreConnectorStats()

    spec = FullAttentionSpec(
        block_size=block_size, num_kv_heads=8, head_size=64, dtype=None
    )
    # Affine large-page geometry: block b of the group starts at
    # pool_base + b * 256 (blocks_per_page=1, page == 2 rows * 128 B).
    group = KVCacheGroupSpec(
        ["layer0", "__cross_layer__"],
        spec,
        num_blocks=num_gpu_blocks,
        blocks_per_page=1,
        kernel_rows_per_block=2,
        row_bytes=128,
    )
    worker._kv_cache_config = KVCacheConfig(
        num_pages=num_gpu_blocks,
        kv_cache_tensors=[
            KVCacheTensor(
                size=0,
                layers=["layer0", "__cross_layer__"],
                layer_stride=0,
                block_stride=0,
            )
        ],
        kv_cache_groups=[group],
        block_size=block_size,
        large_page_bytes=256,
    )
    worker._kv_cache_groups = [group]
    worker.pcp_size = 1
    worker.dcp_size = 1
    # Pre-build a single-group token_dbs so lookup-only tests don't have to
    # call register_kv_caches.
    metadata = KeyMetadata("test-model", 0, 0, 0, 0, group_id=0)
    worker.token_dbs = [
        ChunkedTokenDatabase(
            metadata,
            block_size=block_size,
            store_layout=RankLocalStoreLayout(
                metadata,
                block_size,
                geometry=GroupGeometry(group.kernel_rows_per_block, group.row_bytes),
            ),
        )
    ]
    worker.coord = mooncake_store_worker.MooncakeStoreCoordinator(
        worker._kv_cache_groups,
        scheduler_block_size=block_size,
        block_size=block_size,
    )
    _refresh_group_tp_replication_factors(worker)
    return worker


def test_lookup_key_prefixes_cover_dcp_rank_namespaces():
    worker = _make_bare_worker()
    worker.tp_size = 4
    worker.num_kv_head = 1
    worker.dcp_size = 4
    _refresh_group_tp_replication_factors(worker)

    assert worker._lookup_key_prefixes[0] == (
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@group:0",
        "test-model@tp_rank:1@pcp0@dcp1@pp_rank:0@group:0",
        "test-model@tp_rank:2@pcp0@dcp2@pp_rank:0@group:0",
        "test-model@tp_rank:3@pcp0@dcp3@pp_rank:0@group:0",
    )


def test_lookup_key_prefixes_cover_store_tp_shards():
    worker = _make_bare_worker()
    worker.store_tp_size = 4
    metadata = KeyMetadata(
        "test-model",
        0,
        0,
        0,
        0,
        group_id=0,
        store_namespace=_TP_SHARED_NAMESPACE,
    )
    worker.token_dbs = [
        ChunkedTokenDatabase(
            metadata,
            block_size=16,
            store_layout=LBHNCStoreLayout(
                metadata,
                block_size=16,
                local_tp_size=2,
                store_tp_size=4,
                tp_rank=0,
                num_kv_heads=8,
                geometry=GroupGeometry(1, 512),
            ),
        )
    ]
    worker._init_lookup_key_prefixes()

    assert worker._lookup_key_prefixes[0] == (
        *(_tp_shared_prefix(tp_rank) for tp_rank in range(4)),
    )


def test_lookup_key_prefixes_cover_pcp_rank_namespaces():
    worker = _make_bare_worker()
    worker.tp_size = 4
    worker.num_kv_head = 1
    worker.pcp_size = 2
    worker.dcp_size = 1
    _refresh_group_tp_replication_factors(worker)

    assert worker._lookup_key_prefixes[0] == (
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@group:0",
        "test-model@tp_rank:0@pcp1@dcp0@pp_rank:0@group:0",
    )


def test_lookup_key_prefixes_expand_tp_sharded_groups_per_rank():
    """Replicated attention needs one namespace; sharded Mamba needs every rank."""
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheGroupSpec,
        MambaSpec,
    )

    worker = _make_bare_worker(block_size=16)
    worker.tp_size = 2
    worker.num_kv_head = 1
    fa = FullAttentionSpec(block_size=16, num_kv_heads=8, head_size=64, dtype=None)
    mamba = MambaSpec(
        block_size=16,
        shapes=((1, 1),),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
    )
    worker._kv_cache_groups = [
        KVCacheGroupSpec(["l0"], fa),
        KVCacheGroupSpec(["l1"], mamba),
    ]
    worker.token_dbs = [
        _make_rank_local_db(KeyMetadata("test-model", 0, 0, 0, 0, group_id=0), 16),
        _make_rank_local_db(KeyMetadata("test-model", 1, 0, 0, 0, group_id=1), 16),
    ]
    _refresh_group_tp_replication_factors(worker)

    assert worker._lookup_key_prefixes[0] == (
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@group:0",
    )
    assert worker._lookup_key_prefixes[1] == (
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@group:1",
        "test-model@tp_rank:1@pcp0@dcp0@pp_rank:0@group:1",
    )


def test_group_tp_replication_factors_mixed_mla_gqa_mamba():
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheGroupSpec,
        MambaSpec,
        MLAAttentionSpec,
    )

    worker = _make_bare_worker(block_size=16)
    worker.tp_size = 4
    worker.num_kv_head = 2
    mla = MLAAttentionSpec(block_size=16, num_kv_heads=1, head_size=64, dtype=None)
    gqa = FullAttentionSpec(block_size=16, num_kv_heads=8, head_size=64, dtype=None)
    mamba = MambaSpec(
        block_size=16,
        shapes=((1, 1),),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
    )
    worker._kv_cache_groups = [
        KVCacheGroupSpec(["l0"], mla),
        KVCacheGroupSpec(["l1"], gqa),
        KVCacheGroupSpec(["l2"], mamba),
    ]
    worker.token_dbs = [
        _make_rank_local_db(KeyMetadata("test-model", 0, 0, 0, 0, group_id=g_idx), 16)
        for g_idx in range(3)
    ]

    _refresh_group_tp_replication_factors(worker)
    assert worker._group_tp_replication_factors == (4, 2, 1)
    assert worker._lookup_key_prefixes[0] == (
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@group:0",
    )
    assert worker._lookup_key_prefixes[1] == (
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@group:1",
        "test-model@tp_rank:1@pcp0@dcp0@pp_rank:0@group:1",
    )
    assert worker._lookup_key_prefixes[2] == (
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@group:2",
        "test-model@tp_rank:1@pcp0@dcp0@pp_rank:0@group:2",
        "test-model@tp_rank:2@pcp0@dcp0@pp_rank:0@group:2",
        "test-model@tp_rank:3@pcp0@dcp0@pp_rank:0@group:2",
    )


def test_gqa_group_replication_factor_follows_worker_kv_head_count():
    worker = _make_bare_worker(block_size=16)
    worker.tp_size = 4
    worker.num_kv_head = 2
    worker._kv_cache_groups = [
        KVCacheGroupSpec(
            ["l0"],
            FullAttentionSpec(block_size=16, num_kv_heads=1, head_size=64, dtype=None),
        ),
    ]

    _refresh_group_tp_replication_factors(worker)

    assert worker._group_tp_replication_factors == (2,)
    assert worker._lookup_key_prefixes[0] == (
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@group:0",
        "test-model@tp_rank:1@pcp0@dcp0@pp_rank:0@group:0",
    )


def test_lookup_rejects_boundary_missing_one_mamba_shard():
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheGroupSpec,
        MambaSpec,
    )

    worker = _make_bare_worker(block_size=16)
    worker.tp_size = 2
    worker.num_kv_head = 1
    fa = FullAttentionSpec(block_size=16, num_kv_heads=8, head_size=64, dtype=None)
    mamba = MambaSpec(
        block_size=16,
        shapes=((1, 1),),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
    )
    worker._kv_cache_groups = [
        KVCacheGroupSpec(["l0"], fa),
        KVCacheGroupSpec(["l1"], mamba),
    ]
    worker.token_dbs = [
        _make_rank_local_db(KeyMetadata("test-model", 0, 0, 0, 0, group_id=0), 16),
        _make_rank_local_db(KeyMetadata("test-model", 1, 0, 0, 0, group_id=1), 16),
    ]
    worker.coord = mooncake_store_worker.MooncakeStoreCoordinator(
        worker._kv_cache_groups,
        scheduler_block_size=16,
        block_size=16,
    )
    _refresh_group_tp_replication_factors(worker)

    # 33 tokens for two 16-token blocks: the hit stops below the request end,
    # so the full-hit re-derivation stays out of the shard accounting.
    worker.store.batch_is_exist.side_effect = lambda keys: [1] * len(keys)
    assert worker.lookup(33, [b"h0", b"h1"]) == 32

    worker.store.batch_is_exist.side_effect = lambda keys: [
        0 if "tp_rank:1" in k and "group:1" in k else 1 for k in keys
    ]
    assert worker.lookup(33, [b"h0", b"h1"]) == 0


def test_lookup_requires_all_dcp_rank_namespaces():
    worker = _make_bare_worker(block_size=16)
    worker.tp_size = 4
    worker.num_kv_head = 1
    worker.dcp_size = 4
    _refresh_group_tp_replication_factors(worker)
    worker.store.batch_is_exist.return_value = [1, 1, 0, 1]

    assert worker.lookup(16, [b"a0"]) == 0
    assert worker.store.batch_is_exist.call_args.args[0] == [
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@group:0@6130",
        "test-model@tp_rank:1@pcp0@dcp1@pp_rank:0@group:0@6130",
        "test-model@tp_rank:2@pcp0@dcp2@pp_rank:0@group:0@6130",
        "test-model@tp_rank:3@pcp0@dcp3@pp_rank:0@group:0@6130",
    ]


def test_lookup_partial_prefix_returns_first_hit_length():
    worker = _make_bare_worker()
    worker.store.batch_is_exist.return_value = [1, 1, 0]
    assert worker.lookup(48, [b"a0", b"a1", b"a2"]) == 32


def test_lookup_full_hit_reuses_existing_boundary():
    """A full hit is re-derived below the request end without another RPC."""
    worker = _make_bare_worker(block_size=16)
    worker.store.batch_is_exist.return_value = [1, 1]

    assert worker.lookup(32, [b"h0", b"h1"]) == 16
    assert worker.store.batch_is_exist.call_count == 1


def test_lookup_full_hit_with_eagle_pops_once_not_twice():
    """Eagle already leaves the last block for the drafter, so a
    full-prompt re-derivation must never fire for eagle-governed hits:
    firing would anchor the search one block lower and pop a second
    block, regressing the hit by an extra producer boundary."""
    worker = _make_bare_worker(block_size=16)
    worker.coord = mooncake_store_worker.MooncakeStoreCoordinator(
        worker._kv_cache_groups,
        scheduler_block_size=16,
        block_size=16,
        use_eagle=True,
    )
    worker.store.batch_is_exist.return_value = [1, 1, 1, 1]

    # 64-token exact-multiple prompt, all 4 blocks stored: one eagle pop
    # gives 48; a spurious re-derivation (anchored at 48) would pop again
    # and return 32.
    assert worker.lookup(64, [b"h0", b"h1", b"h2", b"h3"]) == 48
    assert worker.store.batch_is_exist.call_count == 1


def test_lookup_full_hit_swa_degrades_when_no_stored_boundary_is_usable():
    """The motivating livelock: the producer of a 64-token prompt stored
    only its SWA tail window (blocks 2-3). The old arithmetic clamp turned
    the full hit into 48, whose SWA window needs the never-written block 1,
    so every load failed and the recompute re-entered the same lookup. The
    re-derivation must report that no stored boundary below the request end
    is usable."""
    from vllm.v1.kv_cache_interface import KVCacheGroupSpec, SlidingWindowSpec

    worker = _make_bare_worker(block_size=16)
    swa = SlidingWindowSpec(
        block_size=16, num_kv_heads=8, head_size=64, dtype=None, sliding_window=32
    )
    worker._kv_cache_groups = [KVCacheGroupSpec(["layer0"], swa)]
    worker.coord = mooncake_store_worker.MooncakeStoreCoordinator(
        worker._kv_cache_groups,
        scheduler_block_size=16,
        block_size=16,
    )
    worker.store.batch_is_exist.return_value = [0, 0, 1, 1]

    assert worker.lookup(64, [b"h0", b"h1", b"h2", b"h3"]) == 0
    assert worker.store.batch_is_exist.call_count == 1


def test_lookup_swa_single_group_returns_full_when_tail_window_present():
    """Single-SWA, sliding_window=32 (= 2 blocks): producer stored only the
    tail. Coordinator-driven lookup returns full prefix even though the
    pre-window blocks are absent."""
    from vllm.v1.kv_cache_interface import KVCacheGroupSpec, SlidingWindowSpec

    worker = _make_bare_worker(block_size=16)
    swa = SlidingWindowSpec(
        block_size=16, num_kv_heads=8, head_size=64, dtype=None, sliding_window=32
    )
    worker._kv_cache_groups = [KVCacheGroupSpec(["layer0"], swa)]
    worker.coord = mooncake_store_worker.MooncakeStoreCoordinator(
        worker._kv_cache_groups,
        scheduler_block_size=16,
        block_size=16,
    )
    worker.store.batch_is_exist.return_value = [0, 0, 1, 1]
    assert worker.lookup(65, [b"h0", b"h1", b"h2", b"h3"]) == 64


# ---------------------------------------------------------------------------
# register_kv_caches tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("save_decode_cache", [False, True])
def test_consumer_starts_send_thread_only_when_put_is_enabled(save_decode_cache):
    num_blocks = 10
    tensor = torch.zeros(num_blocks, 64, dtype=torch.float16)

    worker = _make_bare_worker(
        kv_role="kv_consumer", save_decode_cache=save_decode_cache
    )
    _register_with_mocked_threads(worker, {"layer0": tensor})
    assert (worker.kv_send_thread is not None) == save_decode_cache


def test_start_load_kv_queues_async_load():
    w = _make_bare_worker()
    req = _make_load_req(
        "load-req",
        [b"h0", b"h1"],
        token_len=32,
    )
    assert req.load_spec is not None
    req.load_spec.token_len = 0
    meta = mooncake_store_worker.MooncakeStoreConnectorMetadata(set(), set())
    meta.add_request(req)

    w.start_load_kv(meta)

    assert req.load_spec.token_len == 32
    assert w.recv_request_queue.get_nowait() is req


def test_putting_consumer_queues_decode_save():
    w = _make_bare_worker(kv_role="kv_consumer", save_decode_cache=True)
    send_thread = MagicMock()
    w.kv_send_thread = send_thread
    req = _make_store_req("decode-req", [b"h0", b"h1"])
    req.store_job_id = 1
    meta = mooncake_store_worker.MooncakeStoreConnectorMetadata(set(), set())
    meta.add_request(req)

    with patch.object(torch.cuda, "Event") as event_cls:
        event = event_cls.return_value
        w.wait_for_save(meta)

    event.record.assert_called_once_with()
    send_thread.add_request.assert_called_once_with(req)
    assert req.current_event is event


def _make_strided_layer_views(
    group: KVCacheGroupSpec,
    num_pages: int = 16,
    page_bytes: int = 256,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """One strided fp16 view per layer over a shared large-page pool buffer."""
    buf = torch.zeros(num_pages * page_bytes, dtype=torch.int8)
    row_elems = group.row_bytes // 2
    per_layer = row_elems // len(group.layer_names)
    rows = num_pages * page_bytes // group.row_bytes
    views = [
        torch.as_strided(
            buf.view(torch.float16),
            (rows, per_layer),
            (row_elems, 1),
            storage_offset=i * per_layer,
        )
        for i in range(len(group.layer_names))
    ]
    return buf, views


def test_register_kv_caches_shared_storage():
    """Registering pins the group's one backing allocation with the store and
    hands each layer's strided view to the group's store layout, which pins
    the pool base and owns block addressing."""
    worker = _make_bare_worker()
    group = worker._kv_cache_groups[0]
    buf, caches = _make_strided_layer_views(group)

    _register_with_mocked_threads(worker, dict(zip(group.layer_names, caches)))

    db = worker.token_dbs[0]
    layout = db.store_layout
    assert isinstance(layout, RankLocalStoreLayout)
    base = buf.untyped_storage().data_ptr()
    assert layout.base_addr == base
    assert layout.block_bytes == 256
    # Affine geometry: block b starts at base + b * block_bytes.
    assert db.prepare_value_for_block(3) == ([base + 3 * 256], [256])
    worker.store.register_buffer.assert_called_once_with(base, buf.nbytes)


def test_register_kv_caches_rejects_mismatched_row_stride():
    worker = _make_bare_worker()
    # 200-byte rows; the group's row_bytes is 128.
    bad = torch.zeros(4, 100, dtype=torch.float16)
    with pytest.raises(ValueError, match="stride rows"):
        _register_with_mocked_threads(worker, {"layer0": bad})


def test_register_kv_caches_excludes_nonprefix_cacheable_layers():
    num_blocks = 2
    worker = _make_bare_worker(num_gpu_blocks=num_blocks)
    store_layout = MagicMock()
    worker.token_dbs[0].store_layout = store_layout
    # Rows stride at the group's 128 row bytes.
    full_cache = torch.zeros(num_blocks, 64, dtype=torch.float16)
    qsa_cache = torch.zeros(num_blocks, 4, dtype=torch.float16)

    _register_with_mocked_threads(
        worker,
        {"layer0": full_cache, "qsa": qsa_cache},
    )

    store_layout.register_kv_caches.assert_called_once()
    (registered_caches,) = store_layout.register_kv_caches.call_args.args
    assert len(registered_caches) == 1
    assert registered_caches[0] is full_cache
    worker.store.register_buffer.assert_called_once_with(
        full_cache.data_ptr(), full_cache.untyped_storage().nbytes()
    )


# ---------------------------------------------------------------------------
# Dual-mode (embedded / standalone-store) config validation tests
# ---------------------------------------------------------------------------


def _make_config(**overrides):
    """Build a MooncakeStoreConfig with sensible defaults for validation tests.

    Required dataclass fields are populated; callers override only the field
    under test.
    """
    base = dict(
        metadata_server="http://metadata/endpoint",
        master_server_address="10.0.0.7:50051",
        protocol="rdma",
        device_name="mlx5_0",
    )
    base.update(overrides)
    return worker.MooncakeStoreConfig(**base)


def test_config_defaults_to_embedded():
    """A JSON without explicit mode parses as embedded with 4 GiB segment."""
    cfg = _make_config()
    assert cfg.mode == "embedded"
    assert cfg.global_segment_size == mooncake_store.DEFAULT_GLOBAL_SEGMENT_SIZE
    assert cfg.local_buffer_size == mooncake_store.DEFAULT_LOCAL_BUFFER_SIZE
    assert cfg.enable_offload is False
    assert cfg.tenant_id == mooncake_store.DEFAULT_TENANT_ID


def test_config_pr40900_unchanged(tmp_path):
    """A literal PR-40900 config (no mode, no enable_offload, no preferred_segment)
    parses without raising and resolves to embedded mode."""
    config_path = _write_mooncake_config(
        tmp_path,
        {
            "metadata_server": "http://metadata/endpoint",
            "global_segment_size": "4GB",
            "local_buffer_size": "4GB",
            "protocol": "rdma",
            "device_name": "mlx5_0",
            "master_server_address": "10.0.0.7:50051",
        },
    )
    cfg = worker.MooncakeStoreConfig.from_file(config_path)
    assert cfg.mode == "embedded"
    assert cfg.global_segment_size == 4 * 1024**3
    assert cfg.local_buffer_size == 4 * 1024**3
    assert cfg.enable_offload is False
    assert cfg.tenant_id == mooncake_store.DEFAULT_TENANT_ID


def test_config_from_file_normalizes_tenant_id(tmp_path):
    config_path = _write_mooncake_config(
        tmp_path,
        {
            "metadata_server": "http://metadata/endpoint",
            "master_server_address": "10.0.0.7:50051",
            "tenant_id": "  tenant-a  ",
        },
    )

    cfg = worker.MooncakeStoreConfig.from_file(config_path)

    assert cfg.tenant_id == "tenant-a"


def test_config_from_file_normalizes_empty_tenant_id_to_default(tmp_path):
    config_path = _write_mooncake_config(
        tmp_path,
        {
            "metadata_server": "http://metadata/endpoint",
            "master_server_address": "10.0.0.7:50051",
            "tenant_id": "   ",
        },
    )

    cfg = worker.MooncakeStoreConfig.from_file(config_path)

    assert cfg.tenant_id == mooncake_store.DEFAULT_TENANT_ID


def test_config_from_file_rejects_non_string_tenant_id(tmp_path):
    config_path = _write_mooncake_config(
        tmp_path,
        {
            "metadata_server": "http://metadata/endpoint",
            "master_server_address": "10.0.0.7:50051",
            "tenant_id": False,
        },
    )

    with pytest.raises(TypeError, match="tenant_id must be a string or null"):
        worker.MooncakeStoreConfig.from_file(config_path)


def test_config_embedded_rejects_zero_segment():
    with pytest.raises(
        ValueError, match=r"embedded mode requires global_segment_size > 0"
    ):
        _make_config(mode="embedded", global_segment_size=0)


def test_config_standalone_store_rejects_nonzero_segment():
    with pytest.raises(
        ValueError,
        match=r"standalone-store mode requires global_segment_size == 0",
    ):
        _make_config(mode="standalone-store", global_segment_size=4 * 1024**3)


def test_config_standalone_store_accepts_zero_segment():
    cfg = _make_config(mode="standalone-store", global_segment_size=0)
    assert cfg.mode == "standalone-store"
    assert cfg.global_segment_size == 0


def test_config_unknown_mode():
    with pytest.raises(ValueError, match=r"unknown Mooncake mode"):
        _make_config(mode="something-else")


def test_config_zero_local_buffer():
    with pytest.raises(ValueError, match=r"local_buffer_size must be > 0"):
        _make_config(local_buffer_size=0)


# ---------------------------------------------------------------------------
# End-to-end topology tests
# Covers the two supported recipes:
#   (A) standalone-store mode + disk offload (mode="standalone-store",
#       segment=0, enable_offload=true, preferred_segment set)
#   (B) embedded mode + CPU only      (mode default, segment>0,
#       enable_offload=false, no preferred_segment)
# ---------------------------------------------------------------------------


def test_topology_standalone_store_with_disk_offload(tmp_path, monkeypatch):
    """standalone-store + disk: global_segment_size=0, enable_offload=True,
    preferred_segment set. Assert setup() positional args, ReplicateConfig
    wiring, and that the disk-offload buffer budget is allocated."""
    store = MagicMock()
    store.setup.return_value = 0
    fake_replicate_config_cls = _install_fake_mooncake(monkeypatch, store)
    _patch_worker_runtime(monkeypatch)
    monkeypatch.setenv(
        "MOONCAKE_CONFIG_PATH",
        _write_mooncake_config(
            tmp_path,
            {
                "mode": "standalone-store",
                "metadata_server": "http://metadata/endpoint",
                "global_segment_size": 0,
                "local_buffer_size": "1GB",
                "protocol": "rdma",
                "device_name": "mlx5_0",
                "master_server_address": "10.0.0.7:50051",
                "enable_offload": True,
            },
        ),
    )

    w = worker.MooncakeStoreWorker(
        _make_vllm_config(extra_config={"preferred_segment": "10.0.0.7:50053"}),
        _make_kv_cache_config(),
    )

    # setup() receives global_segment_size=0 and the configured local buffer.
    assert store.setup.call_args.args == (
        "10.0.0.7",
        "http://metadata/endpoint",
        0,
        1024 * 1024 * 1024,
        "rdma",
        "mlx5_0",
        "10.0.0.7:50051",
    )
    assert store.setup.call_args.kwargs == {}
    # ReplicateConfig is built and carries the preferred_segment.
    assert isinstance(w.store_replicate_config, fake_replicate_config_cls)
    assert w.store_replicate_config.preferred_segment == "10.0.0.7:50053"
    # Disk-offload staging budget is allocated (enable_offload=True).
    assert w.disk_offload_buffer_budget_bytes is not None
    assert w.disk_offload_buffer_budget_bytes > 0


def test_topology_embedded_cpu_only(tmp_path, monkeypatch):
    """Embedded + CPU-only: no mode key (defaults to embedded),
    global_segment_size>0, enable_offload absent, no preferred_segment.
    This is the PR-40900 baseline recipe."""
    store = MagicMock()
    store.setup.return_value = 0
    fake_replicate_config_cls = _install_fake_mooncake(monkeypatch, store)
    _patch_worker_runtime(monkeypatch)
    monkeypatch.setenv(
        "MOONCAKE_CONFIG_PATH",
        _write_mooncake_config(
            tmp_path,
            {
                "metadata_server": "http://metadata/endpoint",
                "global_segment_size": "4GB",
                "local_buffer_size": "4GB",
                "protocol": "rdma",
                "device_name": "mlx5_0",
                "master_server_address": "10.0.0.7:50051",
            },
        ),
    )

    w = worker.MooncakeStoreWorker(_make_vllm_config(), _make_kv_cache_config())

    # setup() receives global_segment_size=4 GiB (rank contributes a segment).
    assert store.setup.call_args.args == (
        "10.0.0.7",
        "http://metadata/endpoint",
        4 * 1024 * 1024 * 1024,
        4 * 1024 * 1024 * 1024,
        "rdma",
        "mlx5_0",
        "10.0.0.7:50051",
    )
    assert store.setup.call_args.kwargs == {}
    # No preferred_segment — ReplicateConfig is default-constructed (so the
    # preferred_segment field keeps its default value).
    assert w.preferred_segment is None
    assert isinstance(w.store_replicate_config, fake_replicate_config_cls)
    assert w.store_replicate_config.preferred_segment == ""
    # No disk budget — enable_offload was absent (defaults to False).
    assert w.disk_offload_buffer_budget_bytes is None


def _create_store_client_for_tenant_test(consumer):
    if consumer == "ec":
        from vllm.distributed.ec_transfer.ec_connector.mooncake_store_embedding.store_client import (  # noqa: E501
            create_mooncake_embedding_store_client,
        )

        return create_mooncake_embedding_store_client()
    return worker.MooncakeStoreWorker(
        _make_vllm_config(rank=1), _make_kv_cache_config()
    )


@pytest.mark.parametrize("consumer", ["kv", "ec"])
def test_topology_forwards_non_default_tenant_id(tmp_path, monkeypatch, consumer):
    store = MagicMock()
    store.setup.return_value = 0
    _install_fake_mooncake(monkeypatch, store)
    _patch_worker_runtime(monkeypatch)
    monkeypatch.setenv(
        "MOONCAKE_CONFIG_PATH",
        _write_mooncake_config(
            tmp_path,
            {
                "metadata_server": "http://metadata/endpoint",
                "global_segment_size": "4GB",
                "local_buffer_size": "4GB",
                "protocol": "rdma",
                "device_name": "mlx5_0",
                "master_server_address": "10.0.0.7:50051",
                "tenant_id": "  tenant-a  ",
            },
        ),
    )

    _create_store_client_for_tenant_test(consumer)

    assert store.setup.call_args.kwargs == {"tenant_id": "tenant-a"}


@pytest.mark.parametrize("consumer", ["kv", "ec"])
def test_non_default_tenant_preserves_setup_type_error(tmp_path, monkeypatch, consumer):
    store = MagicMock()
    setup_error = TypeError(
        "setup(): incompatible function arguments; "
        "supported signature includes tenant_id"
    )
    store.setup.side_effect = setup_error
    _install_fake_mooncake(monkeypatch, store)
    _patch_worker_runtime(monkeypatch)
    monkeypatch.setenv(
        "MOONCAKE_CONFIG_PATH",
        _write_mooncake_config(
            tmp_path,
            {
                "metadata_server": "http://metadata/endpoint",
                "global_segment_size": "4GB",
                "local_buffer_size": "4GB",
                "protocol": "rdma",
                "device_name": "mlx5_0",
                "master_server_address": "10.0.0.7:50051",
                "tenant_id": "tenant-a",
            },
        ),
    )

    with pytest.raises(TypeError) as exc_info:
        _create_store_client_for_tenant_test(consumer)

    assert exc_info.value is setup_error


# ---------------------------------------------------------------------------
# Stats/metrics tests (PR-35 port)
# ---------------------------------------------------------------------------


def test_mooncake_store_stats_aggregate_reduce():
    stats = MooncakeStoreConnectorStats()
    stats.record_operation("save_put", 0.01, 2, num_bytes=128)
    other = MooncakeStoreConnectorStats()
    other.record_operation(
        "save_put",
        0.03,
        1,
        num_bytes=64,
        status="error",
        num_failed_keys=1,
    )

    reduced = stats.aggregate(other).reduce()

    assert reduced["save_put_count"] == 2
    assert reduced["save_put_total_keys"] == 3
    assert reduced["save_put_total_bytes"] == 192
    assert reduced["save_put_failed_keys"] == 1
    assert reduced["save_put_error_count"] == 1


def test_worker_get_kv_connector_stats_resets_after_read():
    worker = _make_bare_worker()
    worker._record_kv_connector_operation(
        "save_put",
        0.01,
        2,
        num_bytes=128,
    )

    stats = worker.get_kv_connector_stats()

    assert isinstance(stats, MooncakeStoreConnectorStats)
    assert stats.data["save_put"][0]["num_bytes"] == 128
    assert worker.get_kv_connector_stats() is None


def test_lookup_records_mooncake_metrics():
    worker = _make_bare_worker()
    worker.store.batch_is_exist.return_value = [1, 1]

    result = worker.lookup(33, [b"a0", b"a1"])
    stats = worker.get_kv_connector_stats()

    assert result == 32
    assert isinstance(stats, MooncakeStoreConnectorStats)
    assert len(stats.data["lookup_exists"]) == 1
    assert stats.data["lookup_exists"][0]["num_keys"] == 2


def test_store_worker_close_releases_store():
    worker = _make_bare_worker()
    store = worker.store

    worker.close()

    store.close.assert_called_once_with()
    assert worker.store is None


def test_store_worker_close_is_idempotent():
    worker = _make_bare_worker()
    store = worker.store

    worker.close()
    worker.close()

    # Second call short-circuits because store was already released.
    store.close.assert_called_once_with()


def test_store_worker_close_swallows_store_errors():
    worker = _make_bare_worker()
    worker.store.close.side_effect = RuntimeError("boom")

    # A failure tearing down the store must not propagate out of close().
    worker.close()

    assert worker.store is None


def test_blob_block_hashes_wire_roundtrip():
    """The lookup wire format sends a ``hash_len`` frame plus the raw hashes
    concatenated back-to-back; the server rebuilds them through a zero-copy
    ``BlobBlockHashes`` view over the frame buffer."""
    hashes = [BlockHash(bytes([i]) * 16) for i in range(5)]
    hash_len = len(hashes[0])

    # Client side (LookupKeyClient._lookup): flat payload frame.
    blob = b"".join(hashes)

    # Server side (LookupKeyServer): view over the frame buffer (a memoryview),
    # never materializing the full hash list upfront.
    view = BlobBlockHashes(memoryview(blob), hash_len)

    assert len(view) == 5
    assert list(view) == hashes  # default Sequence iter terminates via IndexError
    assert [bytes(h) for h in view] == hashes
    assert bytes(view[-1]) == hashes[-1]
    assert [bytes(h) for h in view[1:3]] == hashes[1:3]
    with pytest.raises(IndexError):
        _ = view[5]


def test_blob_block_hashes_empty():
    """Empty lookups send hash_len=0 and an empty payload."""
    view = BlobBlockHashes(memoryview(b""), 0)
    assert len(view) == 0
    assert list(view) == []
