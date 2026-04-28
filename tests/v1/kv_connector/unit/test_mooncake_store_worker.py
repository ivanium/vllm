# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import logging
import math
import sys
import threading
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake import (
    mooncake_store_worker,
    rdma_utils,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_store_data import (
    ChunkedTokenDatabase,
    KeyMetadata,
    LoadSpec,
    ReqMeta,
)


def _make_store_sending_thread(
    store: MagicMock,
    *,
    replicate_config: object | None = None,
) -> mooncake_store_worker.KVCacheStoreSendingThread:
    token_database = ChunkedTokenDatabase(
        KeyMetadata("test-model", 0, 0, 0, 0), block_size=16
    )
    token_database.set_kv_caches_base_addr([0x1000])
    token_database.set_block_len([256])
    thread = mooncake_store_worker.KVCacheStoreSendingThread(
        store=store,
        token_database=token_database,
        block_size=16,
        tp_rank=0,
        put_step=1,
        kv_role="kv_producer",
        ready_event=threading.Event(),
        replicate_config=replicate_config,
    )
    thread.request_queue.task_done = MagicMock()
    return thread


def _make_store_recving_thread(
    store: MagicMock,
    *,
    disk_offload_buffer_budget_bytes: int | None = None,
) -> mooncake_store_worker.KVCacheStoreRecvingThread:
    token_database = ChunkedTokenDatabase(
        KeyMetadata("test-model", 0, 0, 0, 0), block_size=16
    )
    token_database.set_kv_caches_base_addr([0x1000])
    token_database.set_block_len([256])
    thread = mooncake_store_worker.KVCacheStoreRecvingThread(
        store=store,
        token_database=token_database,
        block_size=16,
        tp_rank=0,
        ready_event=threading.Event(),
        disk_offload_buffer_budget_bytes=disk_offload_buffer_budget_bytes,
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
        block_ids=list(range(len(block_hashes))),
        block_hashes=block_hashes,
        load_spec=LoadSpec(
            vllm_cached_tokens=vllm_cached_tokens,
            kvpool_cached_tokens=token_len,
            can_load=True,
            token_len=token_len,
        ),
    )


def _make_store_req(req_id: str, block_hashes: list[bytes]) -> ReqMeta:
    return ReqMeta(
        req_id=req_id,
        token_len_chunk=32,
        block_ids=[0, 1],
        block_hashes=block_hashes,
        can_save=True,
        original_block_size=16,
    )


_DISK_OFFLOAD_SINGLE_KEY_BYTES = (
    mooncake_store_worker._estimate_disk_offload_staging_bytes([256])
)
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

    def get_num_layers(self, parallel_config) -> int:
        return 1

    def get_total_num_kv_heads(self) -> int:
        return 1


def _make_vllm_config(
    *, extra_config: dict[str, object] | None = None
) -> SimpleNamespace:
    return SimpleNamespace(
        model_config=_FakeModelConfig(),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1,
            rank=0,
        ),
        kv_transfer_config=_FakeKVTransferConfig(extra_config=extra_config),
        cache_config=SimpleNamespace(block_size=16, num_gpu_blocks=10),
        kv_events_config=SimpleNamespace(enable_kv_cache_events=False),
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
    fake_mooncake_module = types.ModuleType("mooncake")
    fake_mooncake_module.store = fake_store_module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mooncake", fake_mooncake_module)
    monkeypatch.setitem(sys.modules, "mooncake.store", fake_store_module)
    return FakeReplicateConfig


def _patch_worker_runtime(monkeypatch, *, local_ip: str = "10.0.0.7") -> None:
    single_rank_group = SimpleNamespace(world_size=1, rank_in_group=0)
    monkeypatch.setattr(
        mooncake_store_worker, "get_mooncake_dp_engine_index", lambda _: 0
    )
    monkeypatch.setattr(
        mooncake_store_worker, "get_tensor_model_parallel_rank", lambda: 0
    )
    monkeypatch.setattr(
        mooncake_store_worker, "get_tensor_model_parallel_world_size", lambda: 1
    )
    monkeypatch.setattr(
        mooncake_store_worker, "get_pcp_group", lambda: single_rank_group
    )
    monkeypatch.setattr(
        mooncake_store_worker, "get_dcp_group", lambda: single_rank_group
    )
    monkeypatch.setattr(mooncake_store_worker, "get_ip", lambda: local_ip)


def test_get_requester_local_buffer_size_uses_requester_default():
    assert (
        mooncake_store_worker._get_requester_local_buffer_size({})
        == mooncake_store_worker.DEFAULT_REQUESTER_LOCAL_BUFFER_SIZE
    )


def test_get_requester_local_hostname_prefers_override(monkeypatch):
    monkeypatch.setenv("MOONCAKE_LOCAL_HOSTNAME", "worker-a:50053")

    assert rdma_utils.get_requester_local_hostname("10.0.0.7") == "worker-a:50053"


def test_get_configured_preferred_segment_returns_explicit_override():
    assert (
        rdma_utils.get_configured_preferred_segment(
            {"preferred_segment": "10.0.0.7:50053"}
        )
        == "10.0.0.7:50053"
    )


def test_get_configured_preferred_segment_rejects_empty_override():
    with pytest.raises(ValueError, match="preferred_segment"):
        rdma_utils.get_configured_preferred_segment({"preferred_segment": "  "})


def test_get_configured_worker_rnic_prefers_explicit_device_name(monkeypatch):
    store_config = mooncake_store_worker.MooncakeStoreConfig(
        metadata_server="",
        requester_local_buffer_size=1,
        protocol="rdma",
        device_name="rocep139s0",
        master_server_address="",
    )

    assert (
        rdma_utils.get_configured_worker_rnic(
            protocol=store_config.protocol,
            configured_device=store_config.device_name,
        )
        == "rocep139s0"
    )


def test_get_configured_worker_rnic_selects_device_from_explicit_csv(monkeypatch):
    monkeypatch.setattr(
        rdma_utils,
        "get_current_physical_gpu_index",
        lambda: 1,
    )
    store_config = mooncake_store_worker.MooncakeStoreConfig(
        metadata_server="",
        requester_local_buffer_size=1,
        protocol="rdma",
        device_name="rocep139s0,rocep140s0",
        master_server_address="",
    )

    assert (
        rdma_utils.get_configured_worker_rnic(
            protocol=store_config.protocol,
            configured_device=store_config.device_name,
        )
        == "rocep140s0"
    )


def test_get_configured_worker_rnic_falls_back_to_mooncake():
    store_config = mooncake_store_worker.MooncakeStoreConfig(
        metadata_server="",
        requester_local_buffer_size=1,
        protocol="rdma",
        device_name="",
        master_server_address="",
    )

    assert (
        rdma_utils.get_configured_worker_rnic(
            protocol=store_config.protocol,
            configured_device=store_config.device_name,
        )
        == ""
    )


def test_get_configured_worker_rnic_rejects_short_explicit_csv(monkeypatch):
    monkeypatch.setattr(
        rdma_utils,
        "get_current_physical_gpu_index",
        lambda: 2,
    )
    with pytest.raises(ValueError, match="does not cover local GPU 2"):
        rdma_utils.get_configured_worker_rnic(
            protocol="rdma",
            configured_device="rocep139s0,rocep140s0",
        )


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

    thread.add_stored_request("req-a")
    thread._handle_request(_make_store_req("req-a", [b"a0", b"a1"]))

    assert thread._store_pressure_active is True
    assert "req-a" in thread._skip_store_requests
    assert store.batch_put_from_multi_buffers.call_count == 1

    thread.add_stored_request("req-a")
    thread._handle_request(_make_store_req("req-a", [b"a2", b"a3"]))

    assert store.batch_put_from_multi_buffers.call_count == 1

    thread.add_stored_request("req-b")
    thread._handle_request(_make_store_req("req-b", [b"b0", b"b1"]))

    assert thread._store_pressure_active is False
    assert "req-a" not in thread._skip_store_requests
    assert store.batch_put_from_multi_buffers.call_count == 2

    thread.add_stored_request("req-a")
    thread._handle_request(_make_store_req("req-a", [b"a4", b"a5"]))

    assert store.batch_put_from_multi_buffers.call_count == 3


def test_store_sending_thread_only_skips_on_no_available_handle():
    store = MagicMock()
    store.batch_is_exist.side_effect = lambda keys: [0] * len(keys)
    store.batch_put_from_multi_buffers.side_effect = [
        [-500, -500],
        [256, 256],
    ]
    thread = _make_store_sending_thread(store)

    thread.add_stored_request("req-a")
    thread._handle_request(_make_store_req("req-a", [b"a0", b"a1"]))

    assert thread._store_pressure_active is False
    assert "req-a" not in thread._skip_store_requests
    assert store.batch_put_from_multi_buffers.call_count == 1

    thread.add_stored_request("req-a")
    thread._handle_request(_make_store_req("req-a", [b"a2", b"a3"]))

    assert store.batch_put_from_multi_buffers.call_count == 2


def test_store_sending_thread_passes_replicate_config_when_preferred_segment_set():
    store = MagicMock()
    store.batch_is_exist.side_effect = lambda keys: [0] * len(keys)
    store.batch_put_from_multi_buffers.return_value = [256, 256]
    replicate_config = SimpleNamespace(preferred_segment="10.0.0.7:50053")
    thread = _make_store_sending_thread(store, replicate_config=replicate_config)

    thread.add_stored_request("req-a")
    thread._handle_request(_make_store_req("req-a", [b"a0", b"a1"]))

    assert store.batch_put_from_multi_buffers.call_count == 1
    call_args = store.batch_put_from_multi_buffers.call_args.args
    assert len(call_args) == 4
    assert call_args[3] is replicate_config


def test_store_sending_thread_uses_default_write_path_without_preferred_segment():
    store = MagicMock()
    store.batch_is_exist.side_effect = lambda keys: [0] * len(keys)
    store.batch_put_from_multi_buffers.return_value = [256, 256]
    thread = _make_store_sending_thread(store)

    thread.add_stored_request("req-a")
    thread._handle_request(_make_store_req("req-a", [b"a0", b"a1"]))

    assert store.batch_put_from_multi_buffers.call_count == 1
    call_args = store.batch_put_from_multi_buffers.call_args.args
    assert len(call_args) == 3


def test_get_disk_offload_buffer_budget_bytes_uses_requester_budget_override(
    monkeypatch,
):
    monkeypatch.setenv("MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES", "2mb")

    assert (
        mooncake_store_worker._get_disk_offload_buffer_budget_bytes() == 2 * 1024 * 1024
    )


def test_get_disk_offload_buffer_budget_bytes_uses_requester_default(monkeypatch):
    monkeypatch.delenv("MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES", raising=False)

    assert (
        mooncake_store_worker._get_disk_offload_buffer_budget_bytes()
        == mooncake_store_worker.DEFAULT_MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE
    )


def test_estimate_disk_offload_staging_bytes_sums_multi_segment_sizes():
    assert (
        mooncake_store_worker._estimate_disk_offload_staging_bytes([256, 512]) == 12288
    )


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
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@6130",
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@6131",
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@6132",
    ]
    assert sizes == [[256], [256], [256]]
    store.batch_get_replica_desc.assert_not_called()


def test_recv_thread_logs_tier_summary_when_enabled(monkeypatch, caplog_vllm):
    monkeypatch.setenv("VLLM_MOONCAKE_STORE_TIER_LOG", "1")
    caplog_vllm.set_level(logging.INFO, logger=mooncake_store_worker.logger.name)

    store = MagicMock()
    store.batch_get_into_multi_buffers.return_value = [256, 256, -10]
    thread = _make_store_recving_thread(store, disk_offload_buffer_budget_bytes=None)

    req = _make_load_req(
        "req-a",
        [b"a0", b"a1", b"a2"],
        token_len=48,
    )
    expected_keys = [
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@6130",
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@6131",
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@6132",
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
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@6130",
    ]
    assert second_keys == [
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@6131",
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
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@6130",
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@6131",
    ]
    assert second_keys == [
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@6132",
    ]
    base_addr = thread.token_database.kv_caches_base_addr[0]
    block_len = thread.token_database.block_len[0]
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


def test_recv_thread_uses_soft_key_cap_for_disk_offload_split():
    store = MagicMock()
    store.batch_get_into_multi_buffers.side_effect = [
        [256, 256],
        [256],
    ]
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

    assert store.batch_get_into_multi_buffers.call_count == 2
    assert store.batch_get_into_multi_buffers.call_args_list[0].args[0] == [
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@6130",
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@6131",
    ]
    assert store.batch_get_into_multi_buffers.call_args_list[1].args[0] == [
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@6132",
    ]


def test_recv_thread_reports_unsplittable_key_larger_than_budget():
    store = MagicMock()
    thread = _make_store_recving_thread(
        store,
        disk_offload_buffer_budget_bytes=_DISK_OFFLOAD_BUDGET_TOO_SMALL,
    )

    req = _make_load_req(
        "req-a",
        [b"a0"],
        token_len=16,
    )

    thread._handle_request(req)

    assert store.batch_get_into_multi_buffers.call_count == 0


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
    worker = mooncake_store_worker.MooncakeStoreWorker(_make_vllm_config())

    assert not hasattr(worker, "_isolate_offload_resources")
    assert store.setup.call_args.args == (
        "10.0.0.7",
        "http://metadata/endpoint",
        0,
        64 * 1024 * 1024,
        "rdma",
        "mlx5_0",
        "10.0.0.7:50051",
    )


def test_requester_worker_init_prefers_local_hostname_override(
    tmp_path,
    monkeypatch,
):
    store = MagicMock()
    store.setup.return_value = 0
    _install_fake_mooncake(monkeypatch, store)
    _patch_worker_runtime(monkeypatch)
    monkeypatch.setenv("MOONCAKE_LOCAL_HOSTNAME", "worker-a:50053")
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
    mooncake_store_worker.MooncakeStoreWorker(_make_vllm_config())

    assert store.setup.call_args.args[0] == "worker-a:50053"


def test_requester_worker_init_preserves_disk_budget_without_offload_ownership(
    tmp_path,
    monkeypatch,
):
    store = MagicMock()
    store.setup.return_value = 0
    _install_fake_mooncake(monkeypatch, store)
    _patch_worker_runtime(monkeypatch)
    monkeypatch.setenv("MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES", "4mb")
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
    worker = mooncake_store_worker.MooncakeStoreWorker(_make_vllm_config())

    assert worker.disk_offload_buffer_budget_bytes == 4 * 1024 * 1024


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
    worker = mooncake_store_worker.MooncakeStoreWorker(
        _make_vllm_config(
            extra_config={
                "preferred_segment": "10.0.0.7:50053",
            }
        )
    )

    assert isinstance(worker.store_replicate_config, fake_replicate_config_cls)
    assert worker.store_replicate_config.preferred_segment == "10.0.0.7:50053"


# ---------------------------------------------------------------------------
# Helpers for register_kv_caches tests
# ---------------------------------------------------------------------------


def _auto_set_ready_event(*args, **kwargs):
    """Side effect for mocked thread constructors that auto-sets ready_event."""
    for arg in args:
        if isinstance(arg, threading.Event):
            arg.set()
    for val in kwargs.values():
        if isinstance(val, threading.Event):
            val.set()
    return MagicMock()


def _make_bare_worker(
    *,
    num_gpu_blocks: int = 10,
    block_size: int = 16,
    kv_role: str = "kv_both",
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
    worker.use_mla = False
    worker.token_database = ChunkedTokenDatabase(
        KeyMetadata("test-model", 0, 0, 0, 0), block_size=block_size
    )
    worker.kv_role = kv_role
    worker.block_size = block_size
    worker.tp_rank = 0
    worker.put_step = 1
    worker.enable_kv_events = False
    worker.disk_offload_buffer_budget_bytes = None
    worker.kv_send_thread = None
    worker.kv_recv_thread = None
    return worker


# ---------------------------------------------------------------------------
# register_kv_caches tests
# ---------------------------------------------------------------------------


def test_register_kv_caches_blocks_first_single_segment():
    """Blocks-first layout (FlashInfer/MLA): one segment per layer."""
    num_blocks = 10
    page_size_elements = 64  # elements per block
    worker = _make_bare_worker(num_gpu_blocks=num_blocks)

    # Shape: (num_blocks, page_size_elements) — blocks outermost, no outer_dims
    tensor = torch.zeros(num_blocks, page_size_elements, dtype=torch.float16)

    with (
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake."
            "mooncake_store_worker.KVCacheStoreSendingThread",
            side_effect=_auto_set_ready_event,
        ),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake."
            "mooncake_store_worker.KVCacheStoreRecvingThread",
            side_effect=_auto_set_ready_event,
        ),
    ):
        worker.register_kv_caches({"layer0": tensor})

    assert len(worker.kv_caches_base_addr) == 1
    assert worker.kv_caches_base_addr[0] == tensor.untyped_storage().data_ptr()

    expected_block_len = tensor.untyped_storage().nbytes() // num_blocks
    assert len(worker.block_len) == 1
    assert worker.block_len[0] == expected_block_len

    worker.store.register_buffer.assert_called_once_with(
        tensor.untyped_storage().data_ptr(),
        tensor.untyped_storage().nbytes(),
    )


def test_register_kv_caches_kv_first_two_segments():
    """K/V-first layout (FlashAttn): two segments (K, V) per layer."""
    num_blocks = 10
    block_size_tokens = 16
    num_kv_heads = 4
    head_size = 8

    worker = _make_bare_worker(num_gpu_blocks=num_blocks)

    # Shape: (2, num_blocks, block_size, num_kv_heads, head_size) — K/V outermost
    tensor = torch.zeros(
        2,
        num_blocks,
        block_size_tokens,
        num_kv_heads,
        head_size,
        dtype=torch.float16,
    )

    with (
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake."
            "mooncake_store_worker.KVCacheStoreSendingThread",
            side_effect=_auto_set_ready_event,
        ),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake."
            "mooncake_store_worker.KVCacheStoreRecvingThread",
            side_effect=_auto_set_ready_event,
        ),
    ):
        worker.register_kv_caches({"layer0": tensor})

    # K/V-first: dim 0 has stride > page_size, so 2 segments
    assert len(worker.kv_caches_base_addr) == 2
    assert len(worker.block_len) == 2

    el = tensor.element_size()
    seg_stride = tensor.stride(0) * el  # stride of the K/V dim in bytes
    base = tensor.untyped_storage().data_ptr()
    assert worker.kv_caches_base_addr[0] == base
    assert worker.kv_caches_base_addr[1] == base + seg_stride
    assert worker.block_len[0] == seg_stride // num_blocks
    assert worker.block_len[1] == seg_stride // num_blocks


def test_register_kv_caches_cross_layer_single_segment():
    """Cross-layer tensor: single segment with block_len = page_size * num_layers."""
    num_blocks = 10
    num_layers = 4
    per_layer_page_elements = 64  # elements per layer per block

    worker = _make_bare_worker(num_gpu_blocks=num_blocks)

    # Cross-layer blocks-first tensor: all layers packed into a single
    # contiguous block.  Shape (num_blocks, num_layers * per_layer_page)
    # mimics the physical layout after stride reordering.
    total_page_elements = num_layers * per_layer_page_elements
    tensor = torch.zeros(num_blocks, total_page_elements, dtype=torch.float16)

    with (
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake."
            "mooncake_store_worker.KVCacheStoreSendingThread",
            side_effect=_auto_set_ready_event,
        ),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake."
            "mooncake_store_worker.KVCacheStoreRecvingThread",
            side_effect=_auto_set_ready_event,
        ),
    ):
        # Use the cross-layer wrapper key, same as register_cross_layers_kv_caches
        worker.register_kv_caches({"__cross_layer__": tensor})

    assert len(worker.kv_caches_base_addr) == 1
    assert worker.kv_caches_base_addr[0] == tensor.untyped_storage().data_ptr()

    expected_block_len = tensor.untyped_storage().nbytes() // num_blocks
    # block_len should be per_layer_page_size * num_layers
    assert (
        expected_block_len
        == num_layers * per_layer_page_elements * tensor.element_size()
    )
    assert len(worker.block_len) == 1
    assert worker.block_len[0] == expected_block_len

    # Also verify via register_cross_layers_kv_caches wrapper
    worker2 = _make_bare_worker(num_gpu_blocks=num_blocks)
    with (
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake."
            "mooncake_store_worker.KVCacheStoreSendingThread",
            side_effect=_auto_set_ready_event,
        ),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake."
            "mooncake_store_worker.KVCacheStoreRecvingThread",
            side_effect=_auto_set_ready_event,
        ),
    ):
        worker2.register_cross_layers_kv_caches(tensor)

    assert worker2.kv_caches_base_addr == worker.kv_caches_base_addr
    assert worker2.block_len == worker.block_len
