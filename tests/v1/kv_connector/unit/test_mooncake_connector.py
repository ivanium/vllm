# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import contextlib
import errno
import socket
import threading
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import msgspec
import pytest
import torch
import zmq.asyncio

from vllm import envs
from vllm.config import KVTransferConfig, set_current_vllm_config
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake import (
    mooncake_connector,
    mooncake_utils,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector import (
    KVConnectorRole,
    MooncakeConnector,
    MooncakeConnectorMetadata,
    MooncakeConnectorWorker,
    MooncakeXferMetadata,
    MooncakeXferResponse,
    MooncakeXferResponseStatus,
    PullReqMeta,
    SendBlockMeta,
    TransferRegion,
    _align_transfer_regions,
    _coalesce_contiguous_transfer_regions,
    _compute_sender_transfer_plan,
    _validate_asymmetric_region_lengths,
    get_mooncake_bootstrap_addr,
    should_launch_bootstrap_server,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_utils import (
    MooncakeBootstrapServer,
    RegisterWorkerPayload,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    GroupGeometry,
    KVCacheConfig,
    KVCacheGroupSpec,
    MLAAttentionSpec,
    SlidingWindowSpec,
)
from vllm.v1.request import RequestStatus

from .utils import (
    create_request,
    create_scheduler,
    create_vllm_config,
    maybe_update_block_size,
)


@pytest.mark.parametrize(
    ("remote_tp_size", "expected_dst_offsets"),
    [
        (1, [0, None, 65536, None, 131072, None, 196608, None]),
        (2, [0, None, 65536, None, 0, None, 65536, None]),
    ],
)
@pytest.mark.parametrize("local_tp_rank", range(8))
def test_sender_plan_replicated_gqa_tp8_to_smaller_tp(
    remote_tp_size,
    expected_dst_offsets,
    local_tp_rank,
):
    expected_dst_offset = expected_dst_offsets[local_tp_rank]
    plan = _compute_sender_transfer_plan(
        local_tp_rank=local_tp_rank,
        local_tp_size=8,
        remote_tp_rank=local_tp_rank // (8 // remote_tp_size),
        remote_tp_size=remote_tp_size,
        local_kv_block_len=65536,
        remote_kv_block_len=262144 // remote_tp_size,
        producer_cache_replicated=True,
        total_num_kv_heads=4,
    )
    if expected_dst_offset is None:
        assert not plan[0]
    else:
        assert plan == (True, 0, expected_dst_offset, 65536)


@pytest.mark.parametrize(
    ("local_tp_size", "expected_src_offsets"),
    [
        (1, [0, 0, 65536, 65536, 131072, 131072, 196608, 196608]),
        (2, [0, 0, 65536, 65536, 0, 0, 65536, 65536]),
        (4, [0] * 8),
    ],
)
@pytest.mark.parametrize("remote_tp_rank", range(8))
def test_sender_plan_gqa_to_replicated_tp8(
    local_tp_size,
    expected_src_offsets,
    remote_tp_rank,
):
    local_head_count = 4 // local_tp_size
    plan = _compute_sender_transfer_plan(
        local_tp_rank=remote_tp_rank // (8 // local_tp_size),
        local_tp_size=local_tp_size,
        remote_tp_rank=remote_tp_rank,
        remote_tp_size=8,
        local_kv_block_len=local_head_count * 65536,
        remote_kv_block_len=65536,
        producer_cache_replicated=False,
        total_num_kv_heads=4,
    )
    assert plan == (True, expected_src_offsets[remote_tp_rank], 0, 65536)


@pytest.mark.parametrize(
    "local_tp,remote_tp,local_len,remote_len,valid",
    [
        (1, 8, 262144, 65536, True),
        (8, 1, 65536, 262144, True),
        (1, 8, 262144, 131072, False),
        (8, 1, 131072, 262144, False),
    ],
)
def test_region_length_validation_checks_replicated_gqa_heads(
    local_tp, remote_tp, local_len, remote_len, valid
):
    """Replicated heads must have matching, whole per-head payloads."""
    local_region = TransferRegion("layer", 0, 0, local_len, local_len)
    remote_region = TransferRegion("layer", 0, 0, remote_len, remote_len)

    assert (
        _validate_asymmetric_region_lengths(
            local_regions=[local_region],
            remote_regions=[remote_region],
            local_tp_size=local_tp,
            remote_tp_size=remote_tp,
            producer_cache_replicated=local_tp > 4,
            total_num_kv_heads=4,
        )
        is None
    ) == valid


def _make_test_kv_cache_config() -> KVCacheConfig:
    """Single-group large-page config with the identity block mapping.

    One block per page and one kernel row per block, so manager block ids
    double as kernel row ids.
    """
    block_size = maybe_update_block_size(16)
    spec = FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=4,
        head_size=64,
        dtype=torch.float16,
    )
    # The row packs one kernel page of each of the group's four layers.
    row_bytes = 4 * spec.page_size_bytes
    return KVCacheConfig(
        num_pages=16,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                [
                    "model.layers.0.self_attn",
                    "model.layers.1.self_attn",
                    "model.layers.0.mla_attn",
                    "model.layers.1.eagle_attn",
                ],
                spec,
                num_blocks=16,
                blocks_per_page=1,
                kernel_rows_per_block=1,
                row_bytes=row_bytes,
            )
        ],
        large_page_bytes=row_bytes,
        block_size=spec.block_size,
    )


def _make_worker_vllm_config(kv_role: str):
    return create_vllm_config(
        kv_connector="MooncakeConnector",
        kv_role=kv_role,
        block_size=maybe_update_block_size(16),
    )


def _make_packed_mla_kv_cache_config(num_blocks: int) -> KVCacheConfig:
    """Single-group two-layer MLA large-page config (identity mapping)."""
    spec = MLAAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=64,
        dtype=torch.uint8,
    )
    row_bytes = 2 * spec.page_size_bytes
    return KVCacheConfig(
        num_pages=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["model.layers.0.mla_attn", "model.layers.1.indexer"],
                spec,
                num_blocks=num_blocks,
                blocks_per_page=1,
                kernel_rows_per_block=1,
                row_bytes=row_bytes,
            )
        ],
        large_page_bytes=row_bytes,
        block_size=spec.block_size,
    )


class FakeMooncakeWrapper:
    """Mock Mooncake TransferEngine for unit testing environments."""

    def __init__(self, *args, **kwargs):
        pass

    def initialize(self, local_hostname, metadata_server, protocol, device_name) -> int:
        return 0

    def get_rpc_port(self) -> int:
        return 12345

    def batch_transfer_sync_write(
        self, target_hostname, buffers, peer_buffer_addresses, lengths
    ) -> int:
        return 0

    def batch_register_memory(self, buffer_addresses, capacities) -> int:
        return 0

    def batch_unregister_memory(self, buffer_addresses) -> int:
        return 0


def test_align_transfer_regions_uses_layer_name_occurrences():
    """Repeated layer names should align by occurrence order."""
    local_regions = [
        TransferRegion(
            layer_name="model.layers.1.self_attn",
            layer_index=1,
            base_addr=0x1000,
            block_len=256,
            kv_block_len=128,
        ),
        TransferRegion(
            layer_name="model.layers.1.self_attn",
            layer_index=1,
            base_addr=0x1100,
            block_len=256,
            kv_block_len=128,
        ),
    ]
    remote_regions = [
        TransferRegion(
            layer_name="model.layers.0.self_attn",
            layer_index=0,
            base_addr=0xA000,
            block_len=256,
            kv_block_len=128,
        ),
        TransferRegion(
            layer_name="model.layers.1.self_attn",
            layer_index=1,
            base_addr=0xB000,
            block_len=256,
            kv_block_len=128,
        ),
        TransferRegion(
            layer_name="model.layers.1.self_attn",
            layer_index=1,
            base_addr=0xB100,
            block_len=256,
            kv_block_len=128,
        ),
    ]

    aligned_local, aligned_remote, err = _align_transfer_regions(
        local_regions, remote_regions
    )

    assert err is None
    assert [r.base_addr for r in aligned_local] == [0x1000, 0x1100]
    assert [r.base_addr for r in aligned_remote] == [0xB000, 0xB100]

    # Hetero PP with no shared layers is an empty success. Same PP fails closed.
    p_regions = [
        _region(
            0x1000,
            block_len=2048,
            kv_block_len=1024,
            row_offset=0,
            layer_name="model.layers.2.mla_attn",
            layer_index=2,
        )
    ]
    early_d = [
        _region(
            0xA000,
            block_len=2048,
            kv_block_len=1024,
            row_offset=0,
            layer_name="model.layers.0.mla_attn",
            layer_index=0,
        )
    ]
    none_p, none_d, none_err = _align_transfer_regions(
        p_regions, early_d, allow_partial_layers=True
    )
    assert none_err is None
    assert none_p == [] and none_d == []
    _, _, mismatch_err = _align_transfer_regions(
        p_regions, early_d, allow_partial_layers=False
    )
    assert mismatch_err is not None


@pytest.mark.asyncio
async def test_build_transfer_params_separates_prefill_pp_layers():
    """Each producer PP stage should send only its registered layer shard."""
    worker = MooncakeConnectorWorker.__new__(MooncakeConnectorWorker)
    worker.async_zmq_ctx = MagicMock()
    worker.is_kv_consumer = True
    worker.is_kv_producer = True
    worker.tp_rank = 0
    worker.tp_size = 1
    worker.use_mla = False
    worker.kv_cache_config = _make_test_kv_cache_config()
    worker.transfer_topo = SimpleNamespace(
        local_replicates_kv_cache=False,
        total_num_kv_heads=4,
    )

    block_len = 256
    remote_regions = [
        TransferRegion(
            layer_name=f"model.layers.{layer_index}.self_attn",
            layer_index=layer_index,
            base_addr=base_addr,
            block_len=block_len,
            kv_block_len=block_len,
        )
        for layer_index, base_addr in [
            (0, 0xA000),
            (1, 0xB000),
            (2, 0xC000),
            (3, 0xD000),
        ]
    ]
    producer_pp_regions = {
        0: [
            TransferRegion(
                layer_name="model.layers.0.self_attn",
                layer_index=0,
                base_addr=0x1000,
                block_len=block_len,
                kv_block_len=block_len,
            ),
            TransferRegion(
                layer_name="model.layers.1.self_attn",
                layer_index=1,
                base_addr=0x2000,
                block_len=block_len,
                kv_block_len=block_len,
            ),
        ],
        1: [
            TransferRegion(
                layer_name="model.layers.2.self_attn",
                layer_index=2,
                base_addr=0x3000,
                block_len=block_len,
                kv_block_len=block_len,
            ),
            TransferRegion(
                layer_name="model.layers.3.self_attn",
                layer_index=3,
                base_addr=0x4000,
                block_len=block_len,
                kv_block_len=block_len,
            ),
        ],
    }
    expected_by_pp_rank = {
        0: {
            "layers": [0, 1],
            "src_ptrs": [0x1000 + 10 * block_len, 0x2000 + 10 * block_len],
            "dst_ptrs": [0xA000 + 20 * block_len, 0xB000 + 20 * block_len],
        },
        1: {
            "layers": [2, 3],
            "src_ptrs": [0x3000 + 10 * block_len, 0x4000 + 10 * block_len],
            "dst_ptrs": [0xC000 + 20 * block_len, 0xD000 + 20 * block_len],
        },
    }

    transfer_id = "xfer-pp-split"
    send_meta = SendBlockMeta(
        p_req_id="p-req-pp",
        transfer_id=transfer_id,
        local_block_ids=[[10, 11]],
        ready=asyncio.Event(),
    )
    xfer_meta = _xfer_meta(remote_regions, {"d-req-pp": (transfer_id, [[20, 21]])})

    for pp_rank, local_regions in producer_pp_regions.items():
        aligned_local, aligned_remote, err = _align_transfer_regions(
            local_regions, remote_regions
        )

        assert err is None
        assert [r.layer_index for r in aligned_local] == (
            expected_by_pp_rank[pp_rank]["layers"]
        )
        assert [r.layer_index for r in aligned_remote] == (
            expected_by_pp_rank[pp_rank]["layers"]
        )

        (
            src_ptrs,
            dst_ptrs,
            lengths,
            err_reqs,
            err_msg,
        ) = await worker._build_transfer_params(
            ready_reqs=[("d-req-pp", send_meta)],
            agent_meta=xfer_meta,
            local_regions=aligned_local,
            remote_regions=aligned_remote,
        )

        assert err_reqs == []
        assert err_msg is None
        assert src_ptrs == expected_by_pp_rank[pp_rank]["src_ptrs"]
        assert dst_ptrs == expected_by_pp_rank[pp_rank]["dst_ptrs"]
        assert lengths == [2 * block_len, 2 * block_len]


@pytest.mark.asyncio
async def test_send_kv_to_decode_aligns_consumer_regions_by_layer_metadata(
    monkeypatch,
):
    """Producer sends its PP layer shard to the matching consumer layer address."""
    monkeypatch.setenv("VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT", "5")
    vllm_config = _make_worker_vllm_config("kv_producer")

    with set_current_vllm_config(vllm_config), patch_worker_dependencies():
        prefill_connector = MooncakeConnector(
            vllm_config,
            KVConnectorRole.WORKER,
            _make_test_kv_cache_config(),
        )
        prefill_worker = prefill_connector.connector_worker

        block_len = 4096
        prefill_worker.kv_caches_base_addr = [0x1000]
        prefill_worker.block_len_per_layer = [block_len]
        prefill_worker.kv_block_len_per_layer = [block_len]
        prefill_worker.registered_layer_names = ["model.layers.1.self_attn"]
        prefill_worker.registered_layer_indices = [1]

        class InlineSenderLoop:
            async def run_in_executor(self, executor, func, *args):
                return func(*args)

        origin_sender_loop = prefill_worker.sender_loop
        prefill_worker.sender_loop = InlineSenderLoop()

        transfer_id = "xfer-layer-align"
        send_meta = SendBlockMeta(
            p_req_id="p-req-layer-align",
            transfer_id=transfer_id,
            local_block_ids=[[10]],
            ready=asyncio.Event(),
        )
        prefill_worker.reqs_need_send[transfer_id] = send_meta
        send_meta.ready.set()

        xfer_meta = _xfer_meta(
            [
                _region(
                    0xA000,
                    block_len=block_len,
                    row_offset=-1,
                    layer_name="model.layers.0.self_attn",
                    layer_index=0,
                ),
                _region(
                    0xB000,
                    block_len=block_len,
                    row_offset=-1,
                    layer_name="model.layers.1.self_attn",
                    layer_index=1,
                ),
            ],
            {"d-req-layer-align": (transfer_id, [[20]])},
        )
        mock_socket = AsyncMock(spec=zmq.asyncio.Socket)
        mock_socket.send_multipart = AsyncMock()
        identity = b"consumer-layer-align"

        with patch.object(
            prefill_worker, "_send_blocks", return_value=0
        ) as mock_send_blocks:
            await prefill_worker.send_kv_to_decode(identity, mock_socket, xfer_meta)

        src_ptrs, dst_ptrs, lengths = mock_send_blocks.call_args[0][1:]
        assert src_ptrs == [0x1000 + 10 * block_len]
        assert dst_ptrs == [0xB000 + 20 * block_len]
        assert lengths == [block_len]

        sent_identity, sent_payload = mock_socket.send_multipart.call_args[0][0]
        assert sent_identity == identity
        response = prefill_worker._xfer_resp_decoder.decode(sent_payload)
        assert response.status == MooncakeXferResponseStatus.FINISH
        assert response.ok_reqs == ["d-req-layer-align"]

        prefill_worker.sender_loop = origin_sender_loop
        prefill_worker.shutdown()


def test_basic_interface():
    """Unit test for basic MooncakeConnector interface functionality."""
    vllm_config = create_vllm_config(
        kv_connector="MooncakeConnector", kv_role="kv_consumer"
    )
    scheduler = create_scheduler(
        vllm_config, kv_cache_config=_make_test_kv_cache_config()
    )

    # 2 Full Blocks and 1 Half Block.
    BLOCK_SIZE = vllm_config.cache_config.block_size
    NUM_EXTERNAL_FULL_BLOCKS = 2
    NUM_TOKENS = int(BLOCK_SIZE * (NUM_EXTERNAL_FULL_BLOCKS + 0.5))

    request = create_request(
        request_id=1,
        block_size=BLOCK_SIZE,
        num_tokens=NUM_TOKENS,
        do_remote_prefill=True,
    )
    request_id = request.request_id
    request.kv_transfer_params.update(
        {
            "transfer_id": request_id,
            "remote_bootstrap_addr": 54321,
        }
    )

    scheduler.add_request(request)

    # Remote Prefill, triggers NixlConnectorMetadata.
    scheduler_output = scheduler.schedule()
    kv_connector_metadata = scheduler_output.kv_connector_metadata
    assert kv_connector_metadata is not None
    assert isinstance(kv_connector_metadata, MooncakeConnectorMetadata)

    assert len(kv_connector_metadata.reqs_to_recv) == 1
    assert request_id in kv_connector_metadata.reqs_to_recv["my-engine-id"]
    req_meta = kv_connector_metadata.reqs_to_recv["my-engine-id"][request_id]

    # local_block_ids is list[list[int]] (per-group); flatten for comparison.
    all_block_ids = [bid for group in req_meta.local_block_ids for bid in group]
    for block_id, block in zip(
        all_block_ids,
        scheduler.kv_cache_manager.coordinator.single_type_managers[0].req_to_blocks[
            request_id
        ],
    ):
        assert block_id == block.block_id


def test_prompt_less_than_block_size():
    """Test that we can handle case where prompt is < block."""
    vllm_config = create_vllm_config(
        kv_connector="MooncakeConnector", kv_role="kv_consumer"
    )
    scheduler = create_scheduler(
        vllm_config, kv_cache_config=_make_test_kv_cache_config()
    )

    # Half of a block.
    BLOCK_SIZE = vllm_config.cache_config.block_size
    NUM_TOKENS = int(BLOCK_SIZE * 0.5)

    # Request will have 1 partial remote block.
    request = create_request(
        request_id=1,
        block_size=BLOCK_SIZE,
        num_tokens=NUM_TOKENS,
        do_remote_prefill=True,
        num_remote_blocks=1,
    )
    request.kv_transfer_params.update(
        {
            "transfer_id": request.request_id,
            "remote_bootstrap_addr": 54321,
        }
    )

    scheduler.add_request(request)
    scheduler_output = scheduler.schedule()

    # This request will read async.
    kv_connector_metadata = scheduler_output.kv_connector_metadata
    assert kv_connector_metadata is not None
    assert isinstance(kv_connector_metadata, MooncakeConnectorMetadata)
    assert len(kv_connector_metadata.reqs_to_recv["my-engine-id"]) == 1
    assert len(scheduler_output.scheduled_new_reqs) == 0


@pytest.fixture
def bootstrap_server():
    """Fixture to launch and cleanup a Mooncake Bootstrap HTTP Server."""
    server = MooncakeBootstrapServer("127.0.0.1", 0)
    try:
        server.start()
        yield server
    finally:
        server.shutdown()


def test_bootstrap_server_occupied_port_raises_to_caller():
    with socket.create_server(("127.0.0.1", 0)) as listener:
        server = MooncakeBootstrapServer("127.0.0.1", listener.getsockname()[1])
        try:
            with pytest.raises(OSError) as exc_info:
                server.start(timeout=0.05)
            assert exc_info.value.errno == errno.EADDRINUSE
        finally:
            server.shutdown()


def test_bootstrap_server_retains_and_releases_assigned_port(bootstrap_server):
    port = bootstrap_server.port
    assert port > 0
    with (
        pytest.raises(OSError) as exc_info,
        socket.create_server(("127.0.0.1", port)),
    ):
        pass
    assert exc_info.value.errno == errno.EADDRINUSE

    bootstrap_server.shutdown()
    with socket.create_server(("127.0.0.1", port)):
        pass


@pytest.mark.parametrize("failure", ["exit", "error", "timeout"])
def test_bootstrap_server_startup_failure_releases_port(monkeypatch, failure):
    stop = threading.Event()

    def run_without_starting(server, *, sockets):
        if failure == "error":
            raise SystemExit(1)
        if failure == "timeout":
            while not server.should_exit and not stop.wait(0.01):
                pass

    monkeypatch.setattr(mooncake_utils.uvicorn.Server, "run", run_without_starting)
    server = MooncakeBootstrapServer("127.0.0.1", 0)
    try:
        error = TimeoutError if failure == "timeout" else RuntimeError
        with pytest.raises(error):
            server.start(timeout=0.05 if failure == "timeout" else 1.0)
        assert server.port > 0
        with socket.create_server(("127.0.0.1", server.port)):
            pass
    finally:
        stop.set()
        server.shutdown()


@pytest.mark.parametrize("state", ["stopping", "exited"])
def test_bootstrap_server_rejects_stale_started_state(monkeypatch, state):
    release = threading.Event()

    def run_until_released(server, *, sockets):
        server.started = True
        release.wait()

    monkeypatch.setattr(mooncake_utils.uvicorn.Server, "run", run_until_released)
    server = MooncakeBootstrapServer("127.0.0.1", 0)
    try:
        server.start()
        thread = server.server_thread
        assert thread is not None
        if state == "stopping":
            with patch.object(thread, "join", return_value=None):
                server.shutdown()
        else:
            release.set()
            thread.join(timeout=5)
            assert not thread.is_alive()

        with pytest.raises(RuntimeError, match="still starting or stopping"):
            server.start()
    finally:
        release.set()
        server.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("external", [False, True])
async def test_register_worker_recovers_from_slow_bootstrap_server(
    monkeypatch, external
):
    """End-to-end: a real HTTP server that responds slower than the configured
    timeout must be retried on the wire, not treated as fatal."""
    original = MooncakeBootstrapServer.register_worker
    state = {"done": False, "calls": 0}

    async def slow_register(self, payload: RegisterWorkerPayload):
        state["calls"] += 1
        if not state["done"]:
            state["done"] = True
            # Record the worker first, then stall past the client timeout, so
            # the retry exercises the duplicate-registration path.
            response = await original(self, payload)
            await asyncio.sleep(1.5)
            return response
        return await original(self, payload)

    monkeypatch.setattr(MooncakeBootstrapServer, "register_worker", slow_register)

    monkeypatch.setenv("VLLM_MOONCAKE_CONNECTOR_TIMEOUT", "0.5")
    monkeypatch.setattr(mooncake_connector, "_BOOTSTRAP_MAX_ATTEMPTS", 5)

    server = MooncakeBootstrapServer("127.0.0.1", 0)
    server.start()
    try:
        monkeypatch.setenv("VLLM_MOONCAKE_BOOTSTRAP_PORT", str(server.port))
        worker = _make_local_register_worker_stub()
        if external:
            worker.vllm_config = _make_bootstrap_vllm_config(
                bootstrap_server_address=f"127.0.0.1:{server.port}"
            )
            assert not should_launch_bootstrap_server(worker.vllm_config)
        await MooncakeConnectorWorker.register_worker_with_bootstrap(worker)
        assert state["calls"] == 2

        async with httpx.AsyncClient() as client:
            response = await client.get(f"http://127.0.0.1:{server.port}/query")
            assert response.status_code == 200
            assert response.json()["0"]["engine_id"] == "eng-1"
    finally:
        server.shutdown()


@pytest.mark.asyncio
async def test_bootstrap_server(bootstrap_server: MooncakeBootstrapServer):
    """Tests the bootstrap server's api for worker registration and querying.

    Validates DP/TP/PP rank indexing and error handling for duplicate registrations.
    """
    import httpx

    base_url = f"http://127.0.0.1:{bootstrap_server.port}"

    # Query when empty
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{base_url}/query")
        assert response.status_code == 200
        assert response.json() == {}

    # Register multiple PP workers from the same producer engine.
    payload1 = {
        "engine_id": "eng-1",
        "dp_rank": 0,
        "tp_rank": 0,
        "pp_rank": 0,
        "addr": "tcp://1.1.1.1:1111",
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{base_url}/register", json=payload1)
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    payload2 = {
        "engine_id": "eng-1",
        "dp_rank": 0,
        "tp_rank": 0,
        "pp_rank": 1,
        "addr": "tcp://2.2.2.2:2222",
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{base_url}/register", json=payload2)
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    # Query after registration should preserve the PP dimension.
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{base_url}/query")
        assert response.status_code == 200
        data = response.json()
        assert "0" in data
        assert data["0"]["engine_id"] == "eng-1"
        assert data["0"]["worker_addr"]["0"]["0"] == "tcp://1.1.1.1:1111"
        assert data["0"]["worker_addr"]["0"]["1"] == "tcp://2.2.2.2:2222"

    # Re-registering the identical payload is idempotent.
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{base_url}/register", json=payload1)
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    # Test failure: same ranks, conflicting address
    payload1_conflict = dict(payload1, addr="tcp://9.9.9.9:9999")
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{base_url}/register", json=payload1_conflict)
        assert response.status_code == 400
        assert "is already registered" in response.text

    # Test failure: engine_id mismatch for same dp_rank
    payload3_fail = {
        "engine_id": "eng-2",
        "dp_rank": 0,
        "tp_rank": 1,
        "pp_rank": 0,
        "addr": "tcp://3.3.3.3:3333",
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{base_url}/register", json=payload3_fail)
        assert response.status_code == 400
        assert "Engine ID mismatch" in response.text


def _make_bootstrap_vllm_config(
    *,
    local_engines_only: bool = False,
    data_parallel_rank_local: int = 0,
    data_parallel_index: int = 0,
    nnodes_within_dp: int = 1,
    bootstrap_server_address: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        kv_transfer_config=KVTransferConfig(
            kv_connector="MooncakeConnector",
            kv_role="kv_producer",
            kv_connector_extra_config={
                "bootstrap_server_address": bootstrap_server_address
            },
        ),
        parallel_config=SimpleNamespace(
            local_engines_only=local_engines_only,
            data_parallel_rank_local=data_parallel_rank_local,
            data_parallel_index=data_parallel_index,
            nnodes_within_dp=nnodes_within_dp,
            master_addr="model-parallel-master",
            data_parallel_master_ip="data-parallel-master",
        ),
    )


def _make_local_register_worker_stub():
    return SimpleNamespace(
        vllm_config=SimpleNamespace(
            kv_transfer_config=None,
            parallel_config=SimpleNamespace(
                local_engines_only=False,
                data_parallel_rank_local=0,
                data_parallel_index=0,
                nnodes_within_dp=1,
                master_addr="127.0.0.1",
                data_parallel_master_ip="127.0.0.1",
            ),
        ),
        hostname="127.0.0.1",
        side_channel_port=1234,
        engine_id="eng-1",
        dp_rank=0,
        tp_rank=0,
        pp_rank=0,
    )


def _make_register_worker_stub():
    return SimpleNamespace(
        vllm_config=_make_bootstrap_vllm_config(),
        hostname="127.0.0.1",
        side_channel_port=1234,
        engine_id="eng-1",
        dp_rank=0,
        tp_rank=0,
        pp_rank=0,
    )


class _FlakyAsyncClient:
    """httpx.AsyncClient stub that read-times-out `failures` times first."""

    def __init__(self, failures: int, calls: list[int], **kwargs):
        self.failures = failures
        self.calls = calls

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    async def post(self, url, json=None):
        self.calls.append(1)
        if len(self.calls) <= self.failures:
            raise httpx.ReadTimeout("simulated slow bootstrap server")
        response = MagicMock()
        response.raise_for_status = MagicMock()
        return response


@pytest.mark.asyncio
async def test_register_worker_retries_on_read_timeout(monkeypatch):
    """A slow rank-0 bootstrap response must be retried, not treated as fatal."""
    monkeypatch.setattr(mooncake_connector, "_BOOTSTRAP_MAX_ATTEMPTS", 5)
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())

    calls: list[int] = []
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: _FlakyAsyncClient(2, calls),
    )

    worker = _make_register_worker_stub()
    await MooncakeConnectorWorker.register_worker_with_bootstrap(worker)

    assert len(calls) == 3


@pytest.mark.asyncio
async def test_register_worker_raises_after_max_attempts(monkeypatch):
    """Terminal registration failure must raise, not loop forever."""
    monkeypatch.setattr(mooncake_connector, "_BOOTSTRAP_MAX_ATTEMPTS", 3)
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())

    calls: list[int] = []
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: _FlakyAsyncClient(99, calls),
    )

    worker = _make_register_worker_stub()
    with pytest.raises(RuntimeError, match="after 3 attempts"):
        await MooncakeConnectorWorker.register_worker_with_bootstrap(worker)

    assert len(calls) == 3


@pytest.mark.asyncio
async def test_sender_listener_failure_propagates_to_caller(monkeypatch):
    """A terminal registration failure must surface via the Future rather than
    leaving register_kv_caches blocked on its ready event."""
    monkeypatch.setenv("VLLM_MOONCAKE_CONNECTOR_TIMEOUT", "0.1")
    monkeypatch.setattr(mooncake_connector, "_BOOTSTRAP_MAX_ATTEMPTS", 1)

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()

    async def failing_listener(ready_event):
        raise RuntimeError("simulated terminal registration failure")

    try:
        ready_event = threading.Event()
        fut = asyncio.run_coroutine_threadsafe(failing_listener(ready_event), loop)

        deadline = time.monotonic() + 10.0
        raised = None
        while not ready_event.wait(timeout=0.1):
            if fut.done():
                try:
                    fut.result()
                except RuntimeError as e:
                    raised = e
                break
            assert time.monotonic() < deadline, "caller blocked past its deadline"

        assert raised is not None
        assert "simulated terminal registration failure" in str(raised)
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()


@pytest.mark.parametrize(
    (
        "tp_rank",
        "pp_rank",
        "local_engines_only",
        "data_parallel_rank_local",
        "data_parallel_index",
        "expected",
    ),
    [
        (1, 0, False, 0, 0, False),
        (0, 1, False, 0, 0, False),
        (0, 0, True, 0, 1, True),
        (0, 0, True, 1, 0, False),
        (0, 0, False, 0, 0, True),
        (0, 0, False, 0, 1, False),
    ],
    ids=[
        "nonzero_tp_rank",
        "nonzero_pp_rank",
        "local_engine_rank_zero",
        "local_engine_nonzero_rank",
        "internal_lb_first_dp_engine",
        "internal_lb_nonzero_dp_engine",
    ],
)
def test_should_launch_bootstrap_server_selects_single_owner(
    tp_rank: int,
    pp_rank: int,
    local_engines_only: bool,
    data_parallel_rank_local: int,
    data_parallel_index: int,
    expected: bool,
):
    vllm_config = _make_bootstrap_vllm_config(
        local_engines_only=local_engines_only,
        data_parallel_rank_local=data_parallel_rank_local,
        data_parallel_index=data_parallel_index,
    )
    with (
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake."
            "mooncake_connector.get_tensor_model_parallel_rank",
            return_value=tp_rank,
        ),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake."
            "mooncake_connector.get_pp_group"
        ) as mock_pp_group,
    ):
        mock_pp_group.return_value.rank_in_group = pp_rank
        assert should_launch_bootstrap_server(vllm_config) is expected


@pytest.mark.parametrize(
    ("local_engines_only", "nnodes_within_dp", "expected_host"),
    [
        (True, 2, "127.0.0.1"),
        (False, 2, "model-parallel-master"),
        (False, 1, "data-parallel-master"),
    ],
    ids=["local_engine", "multi_node_tp_or_pp", "single_node_internal_lb"],
)
def test_get_mooncake_bootstrap_addr_selects_expected_host(
    local_engines_only: bool,
    nnodes_within_dp: int,
    expected_host: str,
):
    vllm_config = _make_bootstrap_vllm_config(
        local_engines_only=local_engines_only,
        nnodes_within_dp=nnodes_within_dp,
    )

    assert get_mooncake_bootstrap_addr(vllm_config) == (
        expected_host,
        envs.VLLM_MOONCAKE_BOOTSTRAP_PORT,
    )


@pytest.mark.parametrize(
    ("address", "expected"),
    [("external:12345", ("external", 12345)), ("[::1]:12345", ("::1", 12345))],
)
def test_get_mooncake_bootstrap_addr_uses_external_endpoint(address, expected):
    config = _make_bootstrap_vllm_config(bootstrap_server_address=address)
    assert get_mooncake_bootstrap_addr(config) == expected


@pytest.mark.parametrize("address", [":12345", "host:0", "host:65536"])
def test_get_mooncake_bootstrap_addr_rejects_invalid_external_address(address):
    config = _make_bootstrap_vllm_config(bootstrap_server_address=address)
    with pytest.raises(ValueError, match="bootstrap_server_address"):
        get_mooncake_bootstrap_addr(config)


def test_scheduler_request_finished():
    """Tests the scheduler-side logic when a request finishes.

    Differentiates between 'Finished' (requires transfer)
    and 'Aborted' (immediate free).
    """
    vllm_config = create_vllm_config(
        kv_connector="MooncakeConnector", kv_role="kv_producer"
    )
    scheduler = create_scheduler(
        vllm_config, kv_cache_config=_make_test_kv_cache_config()
    )
    scheduler_connector = scheduler.get_kv_connector().connector_scheduler

    request = create_request(request_id=1, do_remote_decode=True)
    request.kv_transfer_params["transfer_id"] = request.request_id

    # Case: Capped length (Successful prefill, need to send to decoder)
    request.status = RequestStatus.FINISHED_LENGTH_CAPPED
    delay_free, _ = scheduler_connector.request_finished(request, block_ids=([10, 11],))
    assert delay_free is True
    assert "id-1" in scheduler_connector._reqs_need_send
    assert scheduler_connector._reqs_need_send["id-1"][1] == [[10, 11]]

    # Case: Aborted (No need to transfer, free blocks immediately)
    scheduler_connector._reqs_need_send.clear()
    request.status = RequestStatus.FINISHED_ABORTED
    delay_free, _ = scheduler_connector.request_finished(request, block_ids=([12],))
    assert delay_free is False
    assert len(scheduler_connector._reqs_need_send) == 0
    assert "id-1" in scheduler_connector._reqs_not_processed


@contextlib.contextmanager
def patch_worker_dependencies():
    """Helper to mock all distributed and network dependencies for Worker tests."""
    with (
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector.TransferEngine",
            FakeMooncakeWrapper,
        ),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector.get_ip",
            return_value="127.0.0.1",
        ),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector.get_tensor_model_parallel_rank",
            return_value=0,
        ),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector.get_tensor_model_parallel_world_size",
            return_value=1,
        ),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector.get_pp_group"
        ) as mock_pp,
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector.should_launch_bootstrap_server",
            return_value=False,
        ),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector.make_zmq_socket"
        ) as mock_make_zmq,
        # The worker binds a CUDA device at init; keep the unit tests CPU-only.
        patch("torch.accelerator.current_device_index", return_value=0),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector.current_platform"
        ),
        patch("httpx.AsyncClient") as mock_async_client,
    ):
        # Mock PP group
        mock_pp_group = MagicMock()
        mock_pp_group.rank_in_group = 0
        mock_pp.return_value = mock_pp_group

        # Mock ZMQ socket
        mock_socket_object = AsyncMock()
        mock_socket_object.setsockopt = MagicMock()
        mock_socket_ctx = MagicMock()
        mock_socket_ctx.__enter__.return_value = mock_socket_object
        mock_make_zmq.return_value = mock_socket_ctx

        # Mock httpx client
        mock_http_client_instance = AsyncMock()
        mock_async_client.return_value = mock_http_client_instance

        yield {
            "mock_make_zmq": mock_make_zmq,
            "mock_socket_object": mock_socket_object,
            "mock_async_client": mock_async_client,
            "mock_http_client": mock_http_client_instance,
        }


def _region(
    base: int,
    *,
    block_len: int = 256,
    kv_block_len: int | None = None,
    row_offset: int | None = None,
    layer_name: str = "layer",
    layer_index: int = 0,
    group_index: int = 0,
) -> TransferRegion:
    """One transfer slice. `row_offset` defaults to `base` for in-row addresses."""
    return TransferRegion(
        layer_name=layer_name,
        layer_index=layer_index,
        base_addr=base,
        block_len=block_len,
        kv_block_len=block_len if kv_block_len is None else kv_block_len,
        group_index=group_index,
        row_offset=base if row_offset is None else row_offset,
    )


def _xfer_meta(
    regions: list[TransferRegion],
    req_blocks: dict,
    *,
    geometry: list[GroupGeometry] | None = None,
    **overrides,
) -> MooncakeXferMetadata:
    """Handshake metadata derived from the regions under test.

    ``geometry`` is the peer's per-group pool geometry. The default is one row
    per block.
    """
    num_groups = max((region.group_index for region in regions), default=0) + 1
    if geometry is None:
        row_bytes = [0] * num_groups
        for region in regions:
            row_bytes[region.group_index] = (
                row_bytes[region.group_index] or region.block_len
            )
        geometry = [GroupGeometry(1, b) for b in row_bytes]
    fields = dict(
        remote_hostname="consumer-host",
        remote_port=54321,
        remote_tp_size=1,
        remote_tp_rank=0,
        req_blocks=req_blocks,
        kv_caches_base_addr=[region.base_addr for region in regions],
        block_lens=[region.block_len for region in regions],
        kv_block_lens=[region.kv_block_len for region in regions],
        registered_layer_names=[region.layer_name for region in regions],
        registered_layer_indices=[region.layer_index for region in regions],
        registered_group_indices=[region.group_index for region in regions],
        registered_row_offsets=[region.row_offset for region in regions],
        group_geometries=geometry,
    )
    fields.update(overrides)
    return MooncakeXferMetadata(**fields)


@contextlib.contextmanager
def mooncake_register_worker(kv_cache_config, *, kv_role: str = "kv_consumer"):
    """Build a worker whose engine registration and sender thread are patched."""
    vllm_config = _make_worker_vllm_config(kv_role)
    connector_mod = (
        "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector"
    )
    with (
        set_current_vllm_config(vllm_config),
        patch_worker_dependencies(),
        patch(f"{connector_mod}.threading.Event"),
        patch(f"{connector_mod}.threading.Thread") as mock_thread,
    ):
        connector = MooncakeConnector(
            vllm_config, KVConnectorRole.WORKER, kv_cache_config
        )
        worker = connector.connector_worker
        mock_thread.return_value.is_alive.return_value = False
        with patch.object(
            worker.engine, "batch_register_memory", return_value=0
        ) as reg:
            yield worker, reg


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("local_pp_size", "local_pp_rank", "expected_addrs"),
    [
        (1, 0, ["tcp://producer-pp0:1234", "tcp://producer-pp1:1234"]),
        (2, 1, ["tcp://producer-pp1:1234"]),
    ],
    ids=["heterogeneous_pp_pulls_all_remote_pp", "matching_pp_pulls_same_rank"],
)
async def test_receive_kv_selects_remote_pp_workers(
    local_pp_size: int,
    local_pp_rank: int,
    expected_addrs: list[str],
):
    """Decode workers should not hard-code producer pp_rank 0."""
    vllm_config = _make_worker_vllm_config("kv_consumer")

    with set_current_vllm_config(vllm_config), patch_worker_dependencies():
        decode_connector = MooncakeConnector(
            vllm_config,
            KVConnectorRole.WORKER,
            _make_test_kv_cache_config(),
        )
        decode_worker = decode_connector.connector_worker
        decode_worker.pp_size = local_pp_size
        decode_worker.pp_rank = local_pp_rank
        decode_worker._remote_agents = {
            "p-engine": {
                0: {
                    0: "tcp://producer-pp0:1234",
                    1: "tcp://producer-pp1:1234",
                }
            }
        }
        decode_worker._tp_size["p-engine"] = 1

        pull_metas = {
            "d-req-1": PullReqMeta(
                d_req_id="d-req-1",
                transfer_id="xfer-req-1",
                local_block_ids=[[100, 101]],
                remote_engine_id="p-engine",
                remote_bootstrap_addr="http://bootstrap:33333",
            )
        }
        seen_addrs: list[str] = []

        async def fake_receive(worker_addr: str, metas: dict[str, PullReqMeta]):
            seen_addrs.append(worker_addr)
            for meta in metas.values():
                meta.pull_tasks_count -= 1

        with patch.object(
            decode_worker,
            "receive_kv_from_single_worker",
            side_effect=fake_receive,
        ):
            decode_worker.receive_kv("p-engine", pull_metas)
            await asyncio.sleep(0)

        assert seen_addrs == expected_addrs
        assert pull_metas["d-req-1"].pull_tasks_count == 0
        decode_worker.shutdown()


def test_resolve_need_send_accounts_for_remote_tp_fanout():
    """Producer-side completion waits for every paired consumer TP pull."""
    worker = MooncakeConnectorWorker.__new__(MooncakeConnectorWorker)
    worker.async_zmq_ctx = MagicMock()
    worker.is_kv_consumer = True
    worker.is_kv_producer = True
    send_meta = SendBlockMeta(
        p_req_id="p-req-1",
        transfer_id="xfer-req-1",
        local_block_ids=[[1]],
        ready=asyncio.Event(),
    )

    worker.resolve_need_send(send_meta, remote_tp_ranks=[0, 1])

    assert send_meta.need_send == 2


@pytest.mark.asyncio
async def test_heterogeneous_pp_waits_for_consumer_without_shared_layers():
    """P4/D2 must retain KV until both D stages finish, including a no-op pull."""
    config = _make_worker_vllm_config("kv_producer")
    with (
        set_current_vllm_config(config),
        patch_worker_dependencies(),
    ):
        worker = MooncakeConnector(
            config, KVConnectorRole.WORKER, _make_test_kv_cache_config()
        ).connector_worker
        try:
            worker.pp_size = 4
            worker.kv_caches_base_addr = [0x1000]
            worker.block_len_per_layer = [256]
            worker.kv_block_len_per_layer = [256]
            worker.registered_layer_names = ["model.layers.0.self_attn"]
            worker.registered_layer_indices = [0]
            send_meta = SendBlockMeta(
                p_req_id="p-req",
                transfer_id="transfer",
                local_block_ids=[[1]],
                ready=asyncio.Event(),
            )
            send_meta.ready.set()
            worker.reqs_need_send["transfer"] = send_meta
            sock = AsyncMock(spec=zmq.asyncio.Socket, send_multipart=AsyncMock())
            with (
                patch.object(worker, "sender_loop", asyncio.get_running_loop()),
                patch.object(worker, "_send_blocks", return_value=0) as send,
            ):
                for pp_rank in range(2):
                    metadata = _xfer_meta(
                        [
                            _region(
                                0x2000,
                                row_offset=-1,
                                layer_name=f"model.layers.{pp_rank}.self_attn",
                                layer_index=pp_rank,
                            )
                        ],
                        {"d-req": ("transfer", [[2]])},
                        remote_hostname="consumer",
                        remote_port=1234 + pp_rank,
                        remote_pp_size=2,
                    )
                    await worker.send_kv_to_decode(b"consumer", sock, metadata)
                    response = worker._xfer_resp_decoder.decode(
                        sock.send_multipart.call_args.args[0][1]
                    )
                    assert response.status == MooncakeXferResponseStatus.FINISH
                    assert response.ok_reqs == ["d-req"]
                    assert ("transfer" in worker.reqs_need_send) == (pp_rank == 0)
                    assert worker.finished_sending_reqs == (
                        set() if pp_rank == 0 else {"p-req"}
                    )
                send.assert_called_once_with("consumer:1234", [0x1100], [0x2200], [256])
        finally:
            worker.shutdown()
            worker.is_kv_consumer = True


@pytest.mark.asyncio
@patch(
    "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector.TransferEngine",
    FakeMooncakeWrapper,
)
async def test_kv_producer(monkeypatch):
    """Simulates a Producer Worker (Prefiller) receiving a transfer request
    from a Consumer (Decoder).

    Verifies memory offset calculation: ptr = base_addr + block_id * block_len.
    """
    monkeypatch.setenv("VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT", "5")
    vllm_config = _make_worker_vllm_config("kv_producer")

    with set_current_vllm_config(vllm_config), patch_worker_dependencies():
        prefill_connector = MooncakeConnector(
            vllm_config,
            KVConnectorRole.WORKER,
            _make_test_kv_cache_config(),
        )
        prefill_worker = prefill_connector.connector_worker
        prefill_worker.kv_caches_base_addr = [0x1000]
        block_len = 4096
        prefill_worker.block_len_per_layer = [block_len]
        prefill_worker.kv_block_len_per_layer = [block_len]
        prefill_worker.registered_layer_names = ["model.layers.0.self_attn"]
        prefill_worker.registered_layer_indices = [0]

        # Override loop to use current test loop
        origin_sender_loop = prefill_worker.sender_loop
        prefill_worker.sender_loop = asyncio.get_event_loop()

        # A request is finished on Producer and ready to be sent.
        transfer_id = "xfer-req-1"
        send_meta = SendBlockMeta(
            p_req_id="p-req-1",
            transfer_id=transfer_id,
            local_block_ids=[[10, 11]],
            ready=asyncio.Event(),
        )
        prefill_worker.reqs_need_send[transfer_id] = send_meta
        send_meta.ready.set()

        # Remote consumer request metadata
        xfer_meta = _xfer_meta(
            [
                _region(
                    0x2000,
                    block_len=block_len,
                    row_offset=-1,
                    layer_name="model.layers.0.self_attn",
                    layer_index=0,
                )
            ],
            {"d-req-1": (transfer_id, [[20, 21]])},
        )

        mock_socket = AsyncMock(spec=zmq.asyncio.Socket)
        mock_socket.send_multipart = AsyncMock()
        identity = b"consumer-id"

        with patch.object(
            prefill_worker, "_send_blocks", return_value=0
        ) as mock_send_blocks:

            def expected_transfers(src_base, dst_base, src_blocks, dst_blocks):
                n = len(src_blocks)
                return (
                    [src_base + src_blocks[0] * block_len],
                    [dst_base + dst_blocks[0] * block_len],
                    [n * block_len],
                )

            # Normal case: 2 blocks to 2 blocks
            await prefill_worker.send_kv_to_decode(identity, mock_socket, xfer_meta)
            src, dst, lens = expected_transfers(0x1000, 0x2000, [10, 11], [20, 21])
            mock_send_blocks.assert_called_once_with(
                "consumer-host:54321",
                src,
                dst,
                lens,
            )
            mock_socket.send_multipart.assert_called_once()

            # Verify the response sent back to the consumer
            sent_call = mock_socket.send_multipart.call_args[0][0]
            sent_identity, sent_payload = sent_call
            assert sent_identity == identity
            response = prefill_worker._xfer_resp_decoder.decode(sent_payload)
            assert response.status == MooncakeXferResponseStatus.FINISH
            assert response.ok_reqs == ["d-req-1"]

            # Verify internal state cleanup
            assert transfer_id not in prefill_worker.reqs_need_send
            assert "p-req-1" in prefill_worker.finished_sending_reqs

            # More cases:
            # Consumer only needs 1 block (less than P)
            mock_send_blocks.reset_mock()
            mock_socket.send_multipart.reset_mock()
            prefill_worker.reqs_need_send[transfer_id] = send_meta
            send_meta.sent = 0
            send_meta.ready.set()
            xfer_meta.req_blocks["d-req-1"] = (transfer_id, [[20]])
            # Worker processes the consumer's request
            await prefill_worker.send_kv_to_decode(identity, mock_socket, xfer_meta)
            # Verify transfer parameters are correct: 11 to 20
            src, dst, lens = expected_transfers(0x1000, 0x2000, [11], [20])
            mock_send_blocks.assert_called_once_with(
                "consumer-host:54321",
                src,
                dst,
                lens,
            )
            mock_socket.send_multipart.assert_called_once()

            # Consumer needs 3 blocks (more than P, error case)
            mock_send_blocks.reset_mock()
            mock_socket.send_multipart.reset_mock()
            prefill_worker.reqs_need_send[transfer_id] = send_meta
            send_meta.sent = 0
            send_meta.ready.set()
            xfer_meta.req_blocks["d-req-1"] = (transfer_id, [[20, 21, 22]])
            # Worker processes the consumer's request
            await prefill_worker.send_kv_to_decode(identity, mock_socket, xfer_meta)
            # This should not be called because error.
            mock_send_blocks.assert_not_called()
            mock_socket.send_multipart.assert_called_once()
            _, sent_payload = mock_socket.send_multipart.call_args[0][0]
            response = prefill_worker._xfer_resp_decoder.decode(sent_payload)
            assert response.err_msg == "P num blocks less than D"
            assert response.err_reqs == ["d-req-1"]

            # Timeout
            mock_send_blocks.reset_mock()
            mock_socket.send_multipart.reset_mock()
            prefill_worker.reqs_need_send[transfer_id] = send_meta
            send_meta.sent = 0
            send_meta.ready.clear()
            xfer_meta.req_blocks["d-req-1"] = (transfer_id, [[20, 21]])
            # Worker processes the consumer's request
            await prefill_worker.send_kv_to_decode(identity, mock_socket, xfer_meta)
            # This should not be called because timeout.
            mock_send_blocks.assert_not_called()
            mock_socket.send_multipart.assert_called_once()
            _, sent_payload = mock_socket.send_multipart.call_args[0][0]
            response = prefill_worker._xfer_resp_decoder.decode(sent_payload)
            assert response.err_msg == "Timeout waiting for P side ready."
            assert response.err_reqs == ["d-req-1"]

        # Transfer error
        with patch.object(
            prefill_worker, "_send_blocks", return_value=123
        ) as mock_send_blocks:
            mock_socket.send_multipart.reset_mock()
            prefill_worker.reqs_need_send[transfer_id] = send_meta
            send_meta.sent = 0
            send_meta.ready.set()
            xfer_meta.req_blocks["d-req-1"] = (transfer_id, [[20, 21]])
            # Worker processes the consumer's request
            await prefill_worker.send_kv_to_decode(identity, mock_socket, xfer_meta)
            mock_send_blocks.assert_called_once()
            mock_socket.send_multipart.assert_called_once()
            _, sent_payload = mock_socket.send_multipart.call_args[0][0]
            response = prefill_worker._xfer_resp_decoder.decode(sent_payload)
            assert response.err_msg == "Mooncake transfer engine returned 123"
            assert response.err_reqs == ["d-req-1"]

        # Clean up
        prefill_worker.sender_loop = origin_sender_loop
        prefill_worker.shutdown()


@pytest.mark.parametrize(
    ("local_block_ids", "expected_finished"),
    [
        ([], set()),
        ([[]], set()),
        ([[100]], {"d-req-1"}),
    ],
)
def test_pull_completion_requires_local_blocks(
    local_block_ids: list[list[int]],
    expected_finished: set[str],
):
    worker = MooncakeConnectorWorker.__new__(MooncakeConnectorWorker)
    worker.shutdown = MagicMock()
    worker.finished_recving_reqs = set()
    pull_meta = PullReqMeta(
        d_req_id="d-req-1",
        transfer_id="xfer-req-1",
        local_block_ids=local_block_ids,
        remote_engine_id="p-engine",
        remote_bootstrap_addr="http://bootstrap:33333",
        pull_tasks_count=1,
    )
    response = MooncakeXferResponse(
        status=MooncakeXferResponseStatus.FINISH,
        ok_reqs=["d-req-1"],
    )

    worker.process_pulling_result(response, {"d-req-1": pull_meta})

    assert pull_meta.pull_tasks_count == 0
    assert worker.finished_recving_reqs == expected_finished


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal", ["error", "exception", "finish"])
async def test_peer_failure_only_settles_its_pending_requests(terminal):
    """Partial and duplicate replies must not release another peer's targets."""
    with mooncake_register_worker(_make_test_kv_cache_config()) as (worker, _):
        metas = {
            f"d-req-{i}": PullReqMeta(
                d_req_id=f"d-req-{i}",
                transfer_id=f"xfer-{i}",
                local_block_ids=[[i]],
                remote_engine_id="p-engine",
                remote_bootstrap_addr="http://bootstrap:33333",
                pull_tasks_count=2,
            )
            for i in range(1, 4)
        }
        requests = list(metas.values())
        partial = MooncakeXferResponse(
            status=MooncakeXferResponseStatus.CONTINUE,
            ok_reqs=["d-req-1"],
            err_reqs=["d-req-2"],
            err_msg="one request failed",
        )
        end = (
            RuntimeError("connection lost")
            if terminal == "exception"
            else worker._encoder.encode(
                MooncakeXferResponse(
                    status=(
                        MooncakeXferResponseStatus.FINISH
                        if terminal == "finish"
                        else MooncakeXferResponseStatus.ERROR
                    ),
                    err_msg="peer finished",
                )
            )
        )
        sock = mooncake_connector.make_zmq_socket.return_value.__enter__.return_value
        sock.recv.side_effect = [worker._encoder.encode(partial)] * 2 + [end]
        await worker.receive_kv_from_single_worker("tcp://producer:1234", metas)
        assert [meta.pull_tasks_count for meta in requests] == [1, 1, 1]
        assert [meta.failed for meta in requests] == [False, True, True]
        assert worker.finished_recving_reqs == set()
        assert worker.get_block_ids_with_load_errors() == {(0, 2), (0, 3)}
        worker.process_pulling_result(
            MooncakeXferResponse(
                status=MooncakeXferResponseStatus.FINISH, ok_reqs=list(metas)
            ),
            metas,
        )
        assert worker.finished_recving_reqs == {meta.d_req_id for meta in requests}
        assert all(meta.pull_tasks_count == 0 for meta in requests)
        worker.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("first_peer_succeeds", [True, False])
async def test_expired_peer_finish_settles_pending_receive(first_peer_succeeds):
    """An expired peer's omitted result must not strand the decoder's blocks."""
    config = _make_test_kv_cache_config()
    with mooncake_register_worker(config, kv_role="kv_producer") as (producer, _):
        try:
            producer.kv_caches_base_addr = [0x1000]
            producer.block_len_per_layer = [256]
            producer.kv_block_len_per_layer = [256]
            producer.registered_layer_names = ["model.layers.0.self_attn"]
            producer.registered_layer_indices = [0]
            send_meta = SendBlockMeta(
                p_req_id="p-req-1",
                transfer_id="xfer-req-1",
                local_block_ids=[[1]],
                ready=asyncio.Event(),
                expire_time=time.perf_counter() - 1,
            )
            send_meta.ready.set()
            producer.reqs_need_send[send_meta.transfer_id] = send_meta

            async def expire_before_send():
                assert await producer.fetch_finished_sending_reqs() == {"p-req-1"}

            metadata = _xfer_meta(
                [
                    _region(
                        0x2000,
                        row_offset=-1,
                        layer_name=producer.registered_layer_names[0],
                    )
                ],
                {"d-req-1": (send_meta.transfer_id, [[100]])},
            )
            sock = AsyncMock(spec=zmq.asyncio.Socket, send_multipart=AsyncMock())
            with (
                patch.object(send_meta.ready, "wait", side_effect=expire_before_send),
                patch.object(producer, "_send_blocks") as send,
            ):
                await producer.send_kv_to_decode(b"consumer", sock, metadata)
                send.assert_not_called()
            assert send_meta.transfer_id not in producer.reqs_need_send
            sock.send_multipart.assert_awaited_once()
            _, payload = sock.send_multipart.call_args.args[0]
            response = producer._xfer_resp_decoder.decode(payload)
            assert response.status == MooncakeXferResponseStatus.FINISH
            assert not response.ok_reqs
            assert not response.err_reqs
        finally:
            producer.shutdown()
            producer.is_kv_consumer = True

    with mooncake_register_worker(config) as (worker, _):
        try:
            meta = PullReqMeta(
                d_req_id="d-req-1",
                transfer_id="xfer-req-1",
                local_block_ids=[[100]],
                remote_engine_id="p-engine",
                remote_bootstrap_addr="http://bootstrap:33333",
                pull_tasks_count=2,
            )
            worker.process_pulling_result(
                MooncakeXferResponse(
                    status=MooncakeXferResponseStatus.FINISH,
                    ok_reqs=[meta.d_req_id] if first_peer_succeeds else None,
                    err_reqs=None if first_peer_succeeds else [meta.d_req_id],
                ),
                {meta.d_req_id: meta},
            )
            assert meta.pull_tasks_count == 1
            assert worker.finished_recving_reqs == set()
            sock = (
                mooncake_connector.make_zmq_socket.return_value.__enter__.return_value
            )
            sock.recv.return_value = payload
            await worker.receive_kv_from_single_worker(
                "tcp://producer:1234", {meta.d_req_id: meta}
            )
            assert meta.pull_tasks_count == 0
            assert meta.failed
            assert worker.finished_recving_reqs == {meta.d_req_id}
            assert worker.get_block_ids_with_load_errors() == {(0, 100)}
            stats = worker.get_kv_connector_stats()
            assert stats is not None
            assert stats.reduce()["Num failed recvs"] == 1
        finally:
            worker.shutdown()


@pytest.mark.asyncio
async def test_kv_consumuer(monkeypatch):
    """Simulates a Consumer Worker (Decoder) initiating a pull from a Producer.

    Verifies that MooncakeXferMetadata is correctly serialized and sent via ZMQ.
    """
    vllm_config = _make_worker_vllm_config("kv_consumer")

    with set_current_vllm_config(vllm_config), patch_worker_dependencies() as mocks:
        kv_cache_config = _make_test_kv_cache_config()
        decode_connector = MooncakeConnector(
            vllm_config,
            KVConnectorRole.WORKER,
            kv_cache_config,
        )
        decode_worker = decode_connector.connector_worker
        decode_worker.kv_caches_base_addr = [0x1000]
        decode_worker.block_len_per_layer = [4096]
        decode_worker.kv_block_len_per_layer = [4096]
        decode_worker.registered_layer_names = ["model.layers.0.self_attn"]
        decode_worker.registered_layer_indices = [0]
        decode_worker.rpc_port = 54321

        # A request to pull data arrives.
        pull_metas = {
            "d-req-1": PullReqMeta(
                d_req_id="d-req-1",
                transfer_id="xfer-req-1",
                local_block_ids=[[100, 101]],
                remote_engine_id="p-engine",
                remote_bootstrap_addr="http://bootstrap:33333",
                pull_tasks_count=1,
            )
        }
        decode_worker._remote_agents = {"p-engine": {0: {0: "tcp://producer:1234"}}}
        decode_worker._tp_size["p-engine"] = 1

        # Mock the response from the producer.
        mock_response = MooncakeXferResponse(
            status=MooncakeXferResponseStatus.FINISH,
            ok_reqs=["d-req-1"],
        )
        encoded_response = decode_worker._encoder.encode(mock_response)
        mocks["mock_socket_object"].recv.return_value = encoded_response

        # Trigger the receive logic.
        decode_worker.receive_kv("p-engine", pull_metas)
        await asyncio.sleep(1)  # Allow async task to run

        # Verify the metadata sent to the producer.
        mocks["mock_make_zmq"].assert_called_with(
            decode_worker.async_zmq_ctx,
            "tcp://producer:1234",
            zmq.DEALER,
            bind=False,
            linger=0,
        )
        sent_payload = mocks["mock_socket_object"].send.call_args[0][0]
        sent_meta = decode_worker._xfer_meta_decoder.decode(sent_payload)

        assert sent_meta.remote_hostname == "127.0.0.1"
        assert sent_meta.remote_port == 54321
        assert sent_meta.req_blocks["d-req-1"] == ("xfer-req-1", [[100, 101]])
        assert sent_meta.kv_caches_base_addr == [0x1000]
        assert sent_meta.block_lens == [4096]
        assert sent_meta.registered_layer_names == ["model.layers.0.self_attn"]
        assert sent_meta.registered_layer_indices == [0]
        # The pull must carry the geometry the producer needs to expand the
        # consumer's manager block ids to kernel rows.
        assert sent_meta.group_geometries == [kv_cache_config.group_geometry(0)]

        # Verify internal state is updated correctly.
        assert "d-req-1" in decode_worker.finished_recving_reqs

        # Clean up
        decode_worker.shutdown()


@pytest.mark.asyncio
async def test_worker_get_finished_timeout(monkeypatch):
    """Tests the cleanup mechanism for requests."""
    vllm_config = _make_worker_vllm_config("kv_producer")
    with set_current_vllm_config(vllm_config), patch_worker_dependencies():
        prefill_connector = MooncakeConnector(
            vllm_config,
            KVConnectorRole.WORKER,
            _make_test_kv_cache_config(),
        )
        prefill_worker = prefill_connector.connector_worker

        # Add an expired request (expire_time is in the past).
        prefill_worker.reqs_need_send["tx-expired"] = SendBlockMeta(
            p_req_id="p-req-expired",
            transfer_id="tx-expired",
            local_block_ids=[[1, 2]],
            ready=MagicMock(),
            expire_time=time.perf_counter() - 100,
        )

        # Add a non-expired request.
        prefill_worker.reqs_need_send["tx-active"] = SendBlockMeta(
            p_req_id="p-req-active",
            transfer_id="tx-active",
            local_block_ids=[[3, 4]],
            ready=MagicMock(),
            expire_time=time.perf_counter() + 100,
        )

        finished_reqs = await prefill_worker.fetch_finished_sending_reqs()

        assert "p-req-expired" in finished_reqs
        assert "p-req-active" not in finished_reqs
        assert "tx-expired" not in prefill_worker.reqs_need_send
        assert "tx-active" in prefill_worker.reqs_need_send


def _lp_layer_views(
    pool: torch.Tensor, config: KVCacheConfig
) -> dict[str, torch.Tensor]:
    """Per-layer strided views over the pool, mirroring allocate_kv_cache.

    Each view holds the group's kernel rows: ``rows`` rows of the layer's
    kernel-page bytes with row stride ``row_bytes``, the layer at its offset
    within the group's row. Every group's row grid starts at the pool base.
    """
    views: dict[str, torch.Tensor] = {}
    for group in config.kv_cache_groups:
        kernel_block_size = (
            group.kv_cache_spec.block_size // group.kernel_rows_per_block
        )
        spec = group.kv_cache_spec.copy_with_new_block_size(kernel_block_size)
        rows = pool.nbytes // group.row_bytes
        offset = 0
        for name in group.layer_names:
            views[name] = torch.as_strided(
                pool, (rows, spec.page_size_bytes), (group.row_bytes, 1), offset
            )
            offset += spec.page_size_bytes
    return views


@pytest.mark.parametrize("head_strided", [False, True])
def test_register_kv_caches(head_strided: bool):
    """Large-page registration: per-layer kernel-row regions over the pool.

    A layer whose row payload is one contiguous span registers one region per
    layer; a head-strided row registers one region per head.
    """
    spec = FullAttentionSpec(
        block_size=maybe_update_block_size(16),
        num_kv_heads=4,
        head_size=64,
        dtype=torch.float16,
    )
    layer_names = [
        "model.layers.0.self_attn",
        "model.layers.1.self_attn",
    ]
    with mooncake_register_worker(_make_test_kv_cache_config()) as (worker, reg):
        for layer_name in layer_names:
            worker._layer_specs[layer_name] = spec
        page = spec.page_size_bytes
        row = 2 * page
        raw = torch.zeros(4 * row, dtype=torch.int8)
        if head_strided:
            # Gap between heads inside the row: the row payload is not one
            # contiguous span, but each head's slice of it is.
            head_bytes = page // spec.num_kv_heads
            head_stride = head_bytes + 64
            layer_row = spec.num_kv_heads * head_stride
            kv_caches = {
                name: torch.as_strided(
                    raw,
                    (4, spec.num_kv_heads, head_bytes),
                    (layer_row, head_stride, 1),
                    layer_idx * layer_row,
                )
                for layer_idx, name in enumerate(layer_names)
            }
        else:
            kv_caches = {
                name: torch.as_strided(raw, (4, page), (row, 1), layer_idx * page)
                for layer_idx, name in enumerate(layer_names)
            }
        worker.register_kv_caches(kv_caches)

        reg.assert_called_once()
        registered_ptrs, registered_lens = reg.call_args[0]
        assert registered_ptrs == [raw.data_ptr()]
        assert registered_lens == [raw.nbytes]

        n_blocks = 4
        storage_end = raw.data_ptr() + raw.nbytes
        for base, stride, kv_len in zip(
            worker.kv_caches_base_addr,
            worker.block_len_per_layer,
            worker.kv_block_len_per_layer,
        ):
            assert base + (n_blocks - 1) * stride + kv_len <= storage_end

        if head_strided:
            head_bytes = page // spec.num_kv_heads
            head_stride = head_bytes + 64
            layer_row = spec.num_kv_heads * head_stride
            expected_addrs = [
                cache[:, head_idx].data_ptr()
                for cache in kv_caches.values()
                for head_idx in range(spec.num_kv_heads)
            ]
            assert worker.kv_caches_base_addr == expected_addrs
            assert worker.block_len_per_layer == [layer_row] * len(expected_addrs)
            assert worker.kv_block_len_per_layer == [head_bytes] * len(expected_addrs)
            assert worker.registered_layer_names == [
                layer_name
                for layer_name in layer_names
                for _ in range(spec.num_kv_heads)
            ]
        else:
            assert len(worker.block_len_per_layer) == len(kv_caches)
            assert worker.block_len_per_layer == [row, row]
            assert worker.kv_block_len_per_layer == [page] * 2
            assert worker.registered_layer_names == list(kv_caches)
            assert worker.registered_layer_indices == [0, 1]
            assert worker.region_row_offsets == [0, page]


def test_packed_and_unpacked_region_lengths_fail_homogeneous_tp_handshake():
    packed = TransferRegion(
        layer_name="model.layers.0.mla_attn",
        layer_index=0,
        base_addr=0x1000,
        block_len=2048,
        kv_block_len=2048,
        group_index=0,
    )
    unpacked = TransferRegion(
        layer_name="model.layers.0.mla_attn",
        layer_index=0,
        base_addr=0x1000,
        block_len=64,
        kv_block_len=64,
        group_index=0,
    )
    err = _validate_asymmetric_region_lengths(
        local_regions=[packed],
        remote_regions=[unpacked],
        local_tp_size=1,
        remote_tp_size=1,
        producer_cache_replicated=False,
    )
    assert err is not None
    assert "length mismatch" in err

    count_err = _validate_asymmetric_region_lengths(
        local_regions=[packed],
        remote_regions=[unpacked, unpacked],
        local_tp_size=1,
        remote_tp_size=1,
        producer_cache_replicated=False,
    )
    assert count_err is not None
    assert "region counts" in count_err


def _two_group_lp_config() -> KVCacheConfig:
    """Full-attention + sliding-window groups over a 1792-byte page pool.

    Group 0 (two FA layers): R=2 kernel rows per block, 512-byte row, so
    block b starts at kernel row 2b. Group 1 (one SWA layer): one 768-byte
    row per block.
    """
    fa_spec = FullAttentionSpec(
        block_size=16, num_kv_heads=2, head_size=8, dtype=torch.uint8
    )
    swa_spec = SlidingWindowSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=24,
        dtype=torch.uint8,
        sliding_window=64,
    )
    g0_row = 2 * (fa_spec.page_size_bytes // 2)
    g1_row = swa_spec.page_size_bytes
    return KVCacheConfig(
        num_pages=4,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["model.layers.0.self_attn", "model.layers.1.self_attn"],
                fa_spec,
                num_blocks=7,
                blocks_per_page=1,
                kernel_rows_per_block=2,
                row_bytes=g0_row,
            ),
            KVCacheGroupSpec(
                ["model.layers.2.swa_attn"],
                swa_spec,
                num_blocks=9,
                blocks_per_page=1,
                kernel_rows_per_block=1,
                row_bytes=g1_row,
            ),
        ],
        large_page_bytes=1792,
        block_size=fa_spec.block_size,
    )


def test_register_kv_caches_two_groups_over_one_pool():
    """Two KV cache groups over one pool register per-group row regions.

    Both groups' row grids start at the pool base; the group-keyed dedup must
    not merge them, and the pull metadata must carry the per-group large-page
    geometry the sender needs to expand manager block ids.
    """
    config = _two_group_lp_config()
    g0, g1 = config.kv_cache_groups
    pool = torch.zeros(config.num_pages * config.large_page_bytes, dtype=torch.uint8)
    with mooncake_register_worker(config) as (worker, reg):
        worker.register_kv_caches(_lp_layer_views(pool, config))

        reg.assert_called_once()
        registered_ptrs, registered_lens = reg.call_args[0]
        assert registered_ptrs == [pool.data_ptr()]
        assert registered_lens == [pool.nbytes]

        assert worker.registered_layer_names == [
            "model.layers.0.self_attn",
            "model.layers.1.self_attn",
            "model.layers.2.swa_attn",
        ]
        assert worker.registered_group_indices == [0, 0, 1]
        assert worker.block_len_per_layer == [g0.row_bytes, g0.row_bytes, g1.row_bytes]
        assert worker.kv_block_len_per_layer == [
            g0.kv_cache_spec.page_size_bytes // 2,
            g0.kv_cache_spec.page_size_bytes // 2,
            g1.kv_cache_spec.page_size_bytes,
        ]
        # Both groups' first layer views sit at the pool base; the regions
        # must not merge into one.
        assert worker.kv_caches_base_addr == [
            pool.data_ptr(),
            pool.data_ptr() + 256,
            pool.data_ptr(),
        ]
        assert worker.region_row_offsets == [0, 256, 0]

        meta = worker._build_pull_metadata({}, include_layout=True)
        assert meta.group_geometries == [
            GroupGeometry(2, g0.row_bytes),
            GroupGeometry(1, g1.row_bytes),
        ]


def _two_group_transfer_worker():
    config = _two_group_lp_config()
    worker = MooncakeConnectorWorker.__new__(MooncakeConnectorWorker)
    worker.async_zmq_ctx = MagicMock()
    worker.is_kv_consumer = True
    worker.is_kv_producer = True
    worker.tp_rank = 0
    worker.tp_size = 1
    worker.use_mla = False
    worker.kv_cache_config = config
    worker.transfer_topo = SimpleNamespace(
        local_replicates_kv_cache=False,
        total_num_kv_heads=2,
    )
    return worker, config


@pytest.mark.asyncio
async def test_build_transfer_params_expands_two_group_kernel_rows():
    """Manager block ids expand to per-group kernel rows."""
    worker, config = _two_group_transfer_worker()
    g0, g1 = config.kv_cache_groups
    local_base, remote_base = 0x10000, 0xA0000
    local_regions = [
        _region(
            local_base,
            block_len=g0.row_bytes,
            kv_block_len=256,
            row_offset=0,
            layer_name="model.layers.0.self_attn",
        ),
        _region(
            local_base + 0x8000,
            block_len=g1.row_bytes,
            kv_block_len=g1.row_bytes,
            row_offset=0,
            layer_name="model.layers.2.swa_attn",
            layer_index=2,
            group_index=1,
        ),
    ]
    remote_regions = [
        _region(
            remote_base,
            block_len=g0.row_bytes,
            kv_block_len=256,
            row_offset=0,
            layer_name="model.layers.0.self_attn",
        ),
        _region(
            remote_base + 0x10000,
            block_len=g1.row_bytes,
            kv_block_len=g1.row_bytes,
            row_offset=0,
            layer_name="model.layers.2.swa_attn",
            layer_index=2,
            group_index=1,
        ),
    ]
    geometry = [GroupGeometry(2, g0.row_bytes), GroupGeometry(1, g1.row_bytes)]
    transfer_id = "xfer-lp"
    send_meta = SendBlockMeta(
        p_req_id="p-lp",
        transfer_id=transfer_id,
        local_block_ids=[[0, 1], [1]],
        ready=asyncio.Event(),
    )
    xfer_meta = _xfer_meta(
        remote_regions,
        {"d-lp": (transfer_id, [[0, 1], [1]])},
        geometry=geometry,
    )

    (
        src_ptrs,
        dst_ptrs,
        lengths,
        err_reqs,
        err_msg,
    ) = await worker._build_transfer_params(
        [("d-lp", send_meta)], xfer_meta, local_regions, remote_regions
    )

    assert err_reqs == []
    assert err_msg is None
    # Group 0 blocks 0 and 1 map to kernel rows [0, 1] and [2, 3].
    assert src_ptrs[:4] == [local_base + row * g0.row_bytes for row in range(4)]
    assert dst_ptrs[:4] == [remote_base + row * g0.row_bytes for row in range(4)]
    assert lengths[:4] == [256] * 4
    # Group 1 block 1 is kernel row 1.
    assert src_ptrs[4] == local_base + 0x8000 + 1 * g1.row_bytes
    assert dst_ptrs[4] == remote_base + 0x10000 + 1 * g1.row_bytes
    assert lengths[4] == g1.row_bytes


@pytest.mark.asyncio
async def test_build_transfer_params_uses_peer_geometry_for_remote_ids():
    """Destination ids expand with the consumer's advertised rows per block."""
    worker, config = _two_group_transfer_worker()
    g0 = config.kv_cache_groups[0]
    local_base, remote_base = 0x10000, 0xA0000
    region = dict(
        block_len=g0.row_bytes,
        kv_block_len=256,
        row_offset=0,
        layer_name="model.layers.0.self_attn",
    )
    local_regions = [_region(local_base, **region)]
    remote_regions = [_region(remote_base, **region)]
    geometry = [GroupGeometry(2, g0.row_bytes), GroupGeometry(1, 768)]
    transfer_id = "xfer-peer-geo"
    send_meta = SendBlockMeta(
        p_req_id="p-peer-geo",
        transfer_id=transfer_id,
        local_block_ids=[[0, 1], []],
        ready=asyncio.Event(),
    )
    xfer_meta = _xfer_meta(
        remote_regions,
        {"d-peer-geo": (transfer_id, [[3, 4], []])},
        geometry=geometry,
    )

    (
        src_ptrs,
        dst_ptrs,
        lengths,
        err_reqs,
        err_msg,
    ) = await worker._build_transfer_params(
        [("d-peer-geo", send_meta)], xfer_meta, local_regions, remote_regions
    )

    assert err_reqs == []
    assert err_msg is None
    assert src_ptrs == [local_base + row * g0.row_bytes for row in range(4)]
    assert dst_ptrs == [remote_base + row * g0.row_bytes for row in (6, 7, 8, 9)]
    assert lengths == [256] * 4


def test_prepare_transfer_regions_rejects_peer_without_group_geometry():
    """A peer that advertises no per-group geometry fails its layout."""
    worker, _ = _two_group_transfer_worker()
    legacy_meta = MooncakeXferMetadata(
        remote_hostname="consumer-host",
        remote_port=54321,
        remote_tp_size=1,
        remote_tp_rank=0,
        req_blocks={"d-no-geo": ("xfer-no-geo", [[0, 1], [1]])},
        kv_caches_base_addr=[],
        block_lens=[],
        kv_block_lens=[],
    )

    local_regions, remote_regions, err_msg = worker._prepare_transfer_regions(
        legacy_meta
    )

    assert local_regions == [] and remote_regions == []
    assert err_msg is not None and "pool geometry for 0 transfer groups" in err_msg


@pytest.mark.asyncio
async def test_build_transfer_params_sends_packed_region_once():
    """A full-row region coalesces contiguous blocks into one copy."""
    worker = MooncakeConnectorWorker.__new__(MooncakeConnectorWorker)
    worker.async_zmq_ctx = MagicMock()
    worker.is_kv_consumer = True
    worker.is_kv_producer = True
    worker.tp_rank = 0
    worker.tp_size = 1
    worker.use_mla = True
    worker.kv_cache_config = _make_packed_mla_kv_cache_config(num_blocks=4)
    worker.transfer_topo = SimpleNamespace(
        local_replicates_kv_cache=False,
        total_num_kv_heads=1,
    )

    block_len = 256
    local_region = _region(
        0x1000,
        block_len=block_len,
        row_offset=0,
        layer_name="model.layers.0.mla_attn",
    )
    remote_region = _region(
        0xA000,
        block_len=block_len,
        row_offset=0,
        layer_name="model.layers.0.mla_attn",
    )
    transfer_id = "xfer-packed"
    send_meta = SendBlockMeta(
        p_req_id="p-packed",
        transfer_id=transfer_id,
        local_block_ids=[[10, 11]],
        ready=asyncio.Event(),
    )
    xfer_meta = _xfer_meta([remote_region], {"d-packed": (transfer_id, [[20, 21]])})

    (
        src_ptrs,
        dst_ptrs,
        lengths,
        err_reqs,
        err_msg,
    ) = await worker._build_transfer_params(
        ready_reqs=[("d-packed", send_meta)],
        agent_meta=xfer_meta,
        local_regions=[local_region],
        remote_regions=[remote_region],
    )

    assert err_reqs == []
    assert err_msg is None
    assert src_ptrs == [0x1000 + 10 * block_len]
    assert dst_ptrs == [0xA000 + 20 * block_len]
    assert lengths == [2 * block_len]


def test_coalesce_promotes_padding_only_for_a_full_row():
    """A full-row merge includes the padding tail; a partial row does not."""
    page, row = 1024, 2560
    local = [
        _region(0, block_len=row, kv_block_len=page, row_offset=0),
        _region(page, block_len=row, kv_block_len=page, row_offset=page),
    ]
    remote = [
        _region(100, block_len=row, kv_block_len=page, row_offset=0),
        _region(100 + page, block_len=row, kv_block_len=page, row_offset=page),
    ]
    full_local, full_remote = _coalesce_contiguous_transfer_regions(
        local, remote, promote_full_row=True
    )
    assert len(full_local) == 1
    assert full_local[0].kv_block_len == full_remote[0].kv_block_len == row

    partial_local, _ = _coalesce_contiguous_transfer_regions(local, remote)
    assert partial_local[0].kv_block_len == 2 * page


def test_coalesce_spans_padding_between_pages():
    """Unpadded pages merge across the padding gap without a full-row lift."""
    row = 2560
    merged, _ = _coalesce_contiguous_transfer_regions(
        [
            _region(0, block_len=row, kv_block_len=1000),
            _region(1024, block_len=row, kv_block_len=1000),
        ],
        [
            _region(5000, block_len=row, kv_block_len=1000, row_offset=0),
            _region(6024, block_len=row, kv_block_len=1000, row_offset=1024),
        ],
    )
    assert len(merged) == 1
    assert merged[0].kv_block_len == 2024
    assert merged[0].row_offset == 0


def test_coalesce_promotes_each_group_that_starts_at_row_zero():
    """Groups overlay one row, so each row-start run is its own full-row copy."""
    page, gap, row = 1000, 1024, 4096

    def side(origin: int) -> list[TransferRegion]:
        return [
            _region(
                origin + base,
                block_len=row,
                kv_block_len=page,
                row_offset=offset,
                layer_name=f"group-{group}",
                layer_index=group,
                group_index=group,
            )
            for group, base, offset in (
                (0, 0, 0),
                (0, gap, gap),
                (1, 0, 0),
                (1, gap, gap),
            )
        ]

    merged_local, merged_remote = _coalesce_contiguous_transfer_regions(
        side(0), side(8000), promote_full_row=True
    )
    assert [region.group_index for region in merged_local] == [0, 1]
    assert [region.kv_block_len for region in merged_local] == [row, row]
    assert [region.kv_block_len for region in merged_remote] == [row, row]
    assert merged_local[1].base_addr == 0
    assert merged_remote[1].base_addr == 8000


def test_coalesce_does_not_promote_a_mid_row_run():
    """A matched PP slice that does not start at row offset 0 stays partial."""
    page, row = 1000, 4096
    merged, _ = _coalesce_contiguous_transfer_regions(
        [
            _region(
                1024, block_len=row, kv_block_len=page, row_offset=1024, layer_index=2
            ),
            _region(
                2048, block_len=row, kv_block_len=page, row_offset=2048, layer_index=2
            ),
        ],
        [
            _region(
                9000 + 1024,
                block_len=row,
                kv_block_len=page,
                row_offset=1024,
                layer_index=2,
            ),
            _region(
                9000 + 2048,
                block_len=row,
                kv_block_len=page,
                row_offset=2048,
                layer_index=2,
            ),
        ],
        promote_full_row=True,
    )
    assert len(merged) == 1
    assert merged[0].row_offset == 1024
    assert merged[0].kv_block_len == 1024 + page


def test_prepare_transfer_regions_reuses_success_and_error():
    """A peer layout is built once. Re-registering drops the cache."""
    worker, _, _, _ = _register_sliced_packed_mla([0, 1])
    regions = worker._get_transfer_regions(
        worker.kv_caches_base_addr,
        worker.block_len_per_layer,
        worker.kv_block_len_per_layer,
        worker.registered_layer_names,
        worker.registered_layer_indices,
        worker.registered_group_indices,
        worker.region_row_offsets,
    )
    meta = _xfer_meta(
        regions,
        {},
        remote_hostname="peer",
        remote_port=1,
        remote_tp_size=worker.tp_size,
        remote_pp_size=worker.pp_size,
    )
    calls = 0
    original = worker._get_transfer_regions

    def counting(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    worker._get_transfer_regions = counting
    prepared = worker._prepare_transfer_regions(meta)
    assert prepared[2] is None
    assert calls == 2
    assert worker._prepare_transfer_regions(meta) == prepared
    assert calls == 2

    mismatch = _xfer_meta(
        [
            _region(
                0x2000,
                block_len=1024,
                row_offset=0,
                layer_name="model.layers.9.mla_attn",
                layer_index=9,
            )
        ],
        {},
        remote_hostname="peer",
        remote_port=2,
        remote_tp_size=worker.tp_size,
        remote_pp_size=worker.pp_size,
    )
    failed = worker._prepare_transfer_regions(mismatch)
    assert failed[2] is not None
    assert calls == 4
    assert worker._prepare_transfer_regions(mismatch)[2] == failed[2]
    assert calls == 4

    page = 1024
    backing = torch.zeros((4, page * 2), dtype=torch.uint8)
    kv_caches = {
        "model.layers.0.mla_attn": backing[:, :page],
        "model.layers.1.mla_attn": backing[:, page:],
    }
    worker.is_kv_consumer = True
    # The first register built a producer. Its thread patch is already gone,
    # so stay on the consumer path and return before the sender thread starts.
    with patch.object(worker.engine, "batch_register_memory", return_value=0):
        worker.register_kv_caches(kv_caches)
    assert worker._prepared_transfer_regions == {}
    worker._prepare_transfer_regions(meta)
    assert calls == 6


def _short_xfer_meta(worker, layout_id: int, req_blocks: dict) -> MooncakeXferMetadata:
    # Steady-state pulls omit the registered layout but always carry the
    # geometry the sender needs to expand manager block ids.
    return MooncakeXferMetadata(
        remote_hostname="peer",
        remote_port=9,
        remote_tp_size=worker.tp_size,
        remote_tp_rank=0,
        remote_pp_size=worker.pp_size,
        layout_id=layout_id,
        req_blocks=req_blocks,
        kv_caches_base_addr=[],
        block_lens=[],
        kv_block_lens=[],
        group_geometries=worker.kv_cache_config.transfer_group_geometries,
    )


@pytest.mark.asyncio
async def test_steady_pull_omits_layout_and_unknown_id_asks_for_resend():
    """A later pull is much smaller, and an unseen layout_id is not guessed."""
    n = 146
    names = [f"model.layers.{i}.self_attn" for i in range(n)]
    req_blocks = {"d": ("xfer", [[1, 2, 3]])}
    full = MooncakeXferMetadata(
        remote_hostname="h",
        remote_port=1,
        remote_tp_size=4,
        remote_tp_rank=0,
        layout_id=1,
        req_blocks=req_blocks,
        kv_caches_base_addr=list(range(n)),
        block_lens=[4096] * n,
        kv_block_lens=[3000] * n,
        registered_layer_names=names,
        registered_layer_indices=list(range(n)),
        registered_group_indices=[i % 5 for i in range(n)],
        registered_row_offsets=[i * 100 for i in range(n)],
        group_geometries=[GroupGeometry(1, 4096)] * 5,
    )
    steady = MooncakeXferMetadata(
        remote_hostname="h",
        remote_port=1,
        remote_tp_size=4,
        remote_tp_rank=0,
        layout_id=1,
        req_blocks=req_blocks,
        kv_caches_base_addr=[],
        block_lens=[],
        kv_block_lens=[],
    )
    encoder = msgspec.msgpack.Encoder()
    full_bytes = len(encoder.encode(full))
    steady_bytes = len(encoder.encode(steady))
    assert steady_bytes < full_bytes * 0.2

    worker, _, regions, _ = _register_sliced_packed_mla([0, 1])
    calls = 0
    original = worker._get_transfer_regions

    def counting(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    worker._get_transfer_regions = counting

    def handshake(layout_id: int) -> MooncakeXferMetadata:
        return _xfer_meta(
            regions,
            req_blocks,
            remote_hostname="peer",
            remote_port=9,
            remote_tp_size=worker.tp_size,
            remote_pp_size=worker.pp_size,
            layout_id=layout_id,
        )

    prepared = worker._prepare_transfer_regions(handshake(1))
    assert prepared[2] is None
    assert calls == 2
    stored = worker._layout_by_peer[worker._peer_layout_key(handshake(1))]
    assert stored == (prepared[0], prepared[1])

    # Same layout under a new id hits the tuple cache and must still be stored.
    worker._prepare_transfer_regions(handshake(3))
    assert calls == 2
    assert worker._peer_layout_key(handshake(3)) in worker._layout_by_peer

    sock = MagicMock()
    sock.send_multipart = AsyncMock()
    await worker.send_kv_to_decode(b"d", sock, _short_xfer_meta(worker, 5, req_blocks))
    _, payload = sock.send_multipart.call_args[0][0]
    response = worker._xfer_resp_decoder.decode(payload)
    assert response.status == MooncakeXferResponseStatus.LAYOUT_MISS
    assert calls == 2


@pytest.mark.asyncio
async def test_peer_layouts_evict_least_recently_used(monkeypatch):
    """Stale layout ids are bounded, and an evicted peer is asked to resend."""
    monkeypatch.setattr(mooncake_connector, "_MAX_PEER_LAYOUTS", 2)
    worker, _, regions, _ = _register_sliced_packed_mla([0])
    req_blocks = {"d": ("xfer", [[1]])}

    def handshake(layout_id: int) -> MooncakeXferMetadata:
        return _xfer_meta(
            regions,
            req_blocks,
            remote_hostname="peer",
            remote_port=9,
            remote_tp_size=worker.tp_size,
            remote_pp_size=worker.pp_size,
            layout_id=layout_id,
        )

    worker._prepare_transfer_regions(handshake(1))
    worker._prepare_transfer_regions(handshake(3))
    assert worker._lookup_peer_layout(_short_xfer_meta(worker, 1, req_blocks))
    worker._prepare_transfer_regions(handshake(5))

    assert worker._lookup_peer_layout(_short_xfer_meta(worker, 1, req_blocks))
    assert worker._lookup_peer_layout(_short_xfer_meta(worker, 5, req_blocks))
    sock = MagicMock()
    sock.send_multipart = AsyncMock()
    await worker.send_kv_to_decode(b"d", sock, _short_xfer_meta(worker, 3, req_blocks))
    _, payload = sock.send_multipart.call_args[0][0]
    response = worker._xfer_resp_decoder.decode(payload)
    assert response.status == MooncakeXferResponseStatus.LAYOUT_MISS


class _ScriptedSock:
    def __init__(self, responses: list[bytes], sent: list[bytes]):
        self._responses = responses
        self._sent = sent

    def setsockopt(self, *_args, **_kwargs):
        return None

    async def send(self, data: bytes):
        self._sent.append(data)

    async def recv(self):
        return self._responses.pop(0)

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


def _single_pull() -> dict[str, PullReqMeta]:
    return {
        "d": PullReqMeta(
            d_req_id="d",
            transfer_id="t",
            local_block_ids=[[1]],
            remote_engine_id="p",
            remote_bootstrap_addr="http://bootstrap",
        )
    }


async def _pull_once(worker, addr: str, responses: list[MooncakeXferResponse]):
    sent: list[bytes] = []
    encoded = [worker._encoder.encode(response) for response in responses]
    with patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.mooncake."
        "mooncake_connector.make_zmq_socket",
        return_value=_ScriptedSock(encoded, sent),
    ):
        await worker.receive_kv_from_single_worker(addr, _single_pull())
    return sent


@pytest.mark.asyncio
async def test_pull_resends_layout_when_prefill_forgot_it():
    worker, _, _, _ = _register_sliced_packed_mla([0])
    addr = "tcp://peer:1"
    worker._acked_layout_peers.add(addr)
    sent = await _pull_once(
        worker,
        addr,
        [
            MooncakeXferResponse(
                status=MooncakeXferResponseStatus.LAYOUT_MISS,
            ),
            MooncakeXferResponse(
                status=MooncakeXferResponseStatus.FINISH,
                ok_reqs=["d"],
                layout_id=worker._kv_layout_id,
            ),
        ],
    )
    assert len(sent) == 2
    assert len(sent[1]) > len(sent[0])
    assert addr in worker._acked_layout_peers


@pytest.mark.asyncio
async def test_pull_keeps_layout_for_prefill_that_does_not_echo_it():
    """A prefill without layout_id support never gets a layout-less pull."""
    worker, _, _, _ = _register_sliced_packed_mla([0])
    addr = "tcp://old-peer:1"
    finish = MooncakeXferResponse(
        status=MooncakeXferResponseStatus.FINISH,
        ok_reqs=["d"],
    )
    first = await _pull_once(worker, addr, [finish])
    assert addr not in worker._acked_layout_peers
    second = await _pull_once(worker, addr, [finish])
    assert len(second[0]) == len(first[0])


def test_coalesce_keeps_a_run_inside_its_starting_row():
    """The row bound is the run start, not the previous slice."""
    row = 2500
    slices = [
        _region(base, block_len=row, kv_block_len=1000) for base in (0, 1000, 2000)
    ]
    merged, _ = _coalesce_contiguous_transfer_regions(slices, slices)
    assert len(merged) == 2
    assert merged[0].kv_block_len == 2000
    assert merged[1].base_addr == 2000
    assert merged[1].kv_block_len == 1000


def _layer_name(idx: int) -> str:
    return f"model.layers.{idx}.mla_attn"


def _register_sliced_packed_mla(
    layer_idxs: list[int], *, page: int = 1024, num_blocks: int = 4
):
    """Register per-layer views of one MLA group row. Returns worker fields."""
    spec = MLAAttentionSpec(
        block_size=maybe_update_block_size(16),
        num_kv_heads=1,
        head_size=64,
        dtype=torch.uint8,
    )
    row_bytes = page * len(layer_idxs)
    kv_cache_config = KVCacheConfig(
        num_pages=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                [_layer_name(i) for i in layer_idxs],
                spec,
                num_blocks=num_blocks,
                blocks_per_page=1,
                kernel_rows_per_block=1,
                row_bytes=row_bytes,
            )
        ],
        large_page_bytes=row_bytes,
        block_size=spec.block_size,
    )
    backing = torch.zeros((num_blocks, page * len(layer_idxs)), dtype=torch.uint8)
    kv_caches = {
        _layer_name(idx): backing[:, i * page : (i + 1) * page]
        for i, idx in enumerate(layer_idxs)
    }
    with mooncake_register_worker(kv_cache_config, kv_role="kv_producer") as (
        worker,
        _,
    ):
        worker.register_kv_caches(kv_caches)
    regions = worker._get_transfer_regions(
        worker.kv_caches_base_addr,
        worker.block_len_per_layer,
        worker.kv_block_len_per_layer,
        worker.registered_layer_names,
        worker.registered_layer_indices,
        worker.registered_group_indices,
        row_offsets=worker.region_row_offsets,
    )
    return worker, kv_cache_config, regions, backing.data_ptr()


def test_sliced_packed_mla_registers_each_layer_view():
    """Each layer slice of an MLA group row stays its own view, in row order."""
    page = 1024
    layer_idxs = [0, 1]
    spec = MLAAttentionSpec(
        block_size=maybe_update_block_size(16),
        num_kv_heads=1,
        head_size=64,
        dtype=torch.uint8,
    )
    row_bytes = page * len(layer_idxs)
    with mooncake_register_worker(
        KVCacheConfig(
            num_pages=4,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec(
                    [_layer_name(i) for i in layer_idxs],
                    spec,
                    num_blocks=4,
                    blocks_per_page=1,
                    kernel_rows_per_block=1,
                    row_bytes=row_bytes,
                )
            ],
            large_page_bytes=row_bytes,
            block_size=spec.block_size,
        ),
        kv_role="kv_producer",
    ) as (worker, reg):
        backing = torch.zeros((4, page * len(layer_idxs)), dtype=torch.uint8)
        worker.register_kv_caches(
            {
                _layer_name(idx): backing[:, i * page : (i + 1) * page]
                for i, idx in enumerate(layer_idxs)
            }
        )
    reg.assert_called_once()
    assert reg.call_args[0][0] == [backing.data_ptr()]
    assert worker.region_row_offsets == [0, page]


async def _copy_packed_slices(
    p_layers: list[int],
    d_layers: list[int],
    *,
    allow_partial: bool,
    local_ids: list[list[int]],
    remote_ids: list[list[int]],
    page: int = 1024,
):
    p_worker, p_config, p_regions, p_base = _register_sliced_packed_mla(
        p_layers, page=page
    )
    _, _, d_regions, d_base = _register_sliced_packed_mla(d_layers, page=page)
    aligned_p, aligned_d, err = _align_transfer_regions(
        p_regions, d_regions, allow_partial_layers=allow_partial
    )
    assert err is None
    coalesced_p, coalesced_d = _coalesce_contiguous_transfer_regions(
        aligned_p, aligned_d
    )
    transfer_id = "xfer-packed-slices"
    send_meta = SendBlockMeta(
        p_req_id="p-slices",
        transfer_id=transfer_id,
        local_block_ids=local_ids,
        ready=asyncio.Event(),
    )
    p_worker.kv_cache_config = p_config
    src, dst, lengths, err_reqs, err_msg = await p_worker._build_transfer_params(
        ready_reqs=[("d-slices", send_meta)],
        agent_meta=_xfer_meta(coalesced_d, {"d-slices": (transfer_id, remote_ids)}),
        local_regions=coalesced_p,
        remote_regions=coalesced_d,
    )
    return SimpleNamespace(
        p_regions=p_regions,
        d_regions=d_regions,
        coalesced_p=coalesced_p,
        coalesced_d=coalesced_d,
        p_base=p_base,
        d_base=d_base,
        src=src,
        dst=dst,
        lengths=lengths,
        err_reqs=err_reqs,
        err_msg=err_msg,
    )


@pytest.mark.asyncio
async def test_packed_hetero_pp_keeps_shared_layers():
    """P-PP4/D-PP2 copies the shared MLA span, not either side's full row."""
    page = 1024
    copied = await _copy_packed_slices(
        [2, 3],
        [0, 1, 2, 3],
        allow_partial=True,
        local_ids=[[10, 11]],
        remote_ids=[[10, 11]],
        page=page,
    )
    assert [region.layer_index for region in copied.p_regions] == [2, 3]
    assert [region.kv_block_len for region in copied.p_regions] == [page, page]
    assert copied.p_regions[0].block_len == 2 * page
    assert copied.d_regions[0].block_len == 4 * page
    assert len(copied.coalesced_p) == 1
    assert copied.coalesced_p[0].kv_block_len == 2 * page
    assert copied.coalesced_d[0].base_addr == copied.d_base + 2 * page
    assert copied.err_reqs == []
    assert copied.err_msg is None
    assert copied.lengths == [2 * page, 2 * page]
    assert copied.src == [
        copied.p_base + 10 * (2 * page),
        copied.p_base + 11 * (2 * page),
    ]
    assert copied.dst == [
        copied.d_base + 2 * page + 10 * (4 * page),
        copied.d_base + 2 * page + 11 * (4 * page),
    ]


@pytest.mark.asyncio
async def test_packed_matching_layers_coalesce_to_one_copy():
    """Identical packed layer sets still emit one copy for contiguous blocks."""
    page = 1024
    copied = await _copy_packed_slices(
        [0, 1],
        [0, 1],
        allow_partial=False,
        local_ids=[[10, 11]],
        remote_ids=[[20, 21]],
        page=page,
    )
    assert copied.err_reqs == []
    assert copied.err_msg is None
    assert len(copied.coalesced_p) == 1
    assert copied.coalesced_p[0].kv_block_len == copied.coalesced_p[0].block_len
    block_len = 2 * page
    assert copied.src == [copied.p_base + 10 * block_len]
    assert copied.dst == [copied.d_base + 20 * block_len]
    assert copied.lengths == [2 * block_len]


@pytest.mark.asyncio
@patch(
    "vllm.distributed.kv_transfer.kv_connector.v1.mooncake."
    "mooncake_connector.TransferEngine",
    FakeMooncakeWrapper,
)
@pytest.mark.parametrize("d_tp_size", [1, 4], ids=["p_tp2_d_tp1", "p_tp2_d_tp4"])
async def test_kv_producer_heterogeneous_tp(monkeypatch, d_tp_size):
    """Tests heterogeneous TP support in the producer transfer path.

    Verifies correct pointer and offset calculation when producer TP=2
    sends to consumer with TP=1 (P>D) or TP=4 (P<D).

    Parametrized cases:
    - P TP=2 > D TP=1: one D rank receives; dst_offset based on P rank
    - P TP=2 < D TP=4: two D ranks receive; src_offset based on D rank
    """
    P_TP_SIZE = 2
    P_TP_RANK = 0
    LOCAL_BLOCK_LEN = 4096

    local_block_len = LOCAL_BLOCK_LEN
    remote_block_len = LOCAL_BLOCK_LEN * P_TP_SIZE // d_tp_size

    monkeypatch.setenv("VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT", "5")
    vllm_config = _make_worker_vllm_config("kv_producer")

    with set_current_vllm_config(vllm_config), patch_worker_dependencies():
        prefill_connector = MooncakeConnector(
            vllm_config,
            KVConnectorRole.WORKER,
            _make_test_kv_cache_config(),
        )
        prefill_worker = prefill_connector.connector_worker

        # Override TP rank/size to simulate P TP=2
        prefill_worker.tp_rank = P_TP_RANK
        prefill_worker.tp_size = P_TP_SIZE
        prefill_worker._tp_size[prefill_worker.engine_id] = P_TP_SIZE
        prefill_worker.transfer_topo.tp_rank = P_TP_RANK
        prefill_worker.transfer_topo.tp_size = P_TP_SIZE

        prefill_worker.kv_caches_base_addr = [0x1000]
        prefill_worker.block_len_per_layer = [local_block_len]
        prefill_worker.kv_block_len_per_layer = [local_block_len]
        prefill_worker.registered_layer_names = ["model.layers.0.self_attn"]
        prefill_worker.registered_layer_indices = [0]

        origin_sender_loop = prefill_worker.sender_loop
        prefill_worker.sender_loop = asyncio.get_event_loop()

        transfer_id = "xfer-hetero-1"
        local_block_ids = [[10, 11]]
        send_meta = SendBlockMeta(
            p_req_id="p-req-h1",
            transfer_id=transfer_id,
            local_block_ids=local_block_ids,
            ready=asyncio.Event(),
        )
        prefill_worker.reqs_need_send[transfer_id] = send_meta
        send_meta.ready.set()

        # Compute target D ranks using the production code path
        target_d_ranks = prefill_worker.transfer_topo.handshake_target_ranks(d_tp_size)

        mock_socket = AsyncMock(spec=zmq.asyncio.Socket)
        mock_socket.send_multipart = AsyncMock()
        identity = b"consumer-hetero"

        # Assign different remote block IDs per D rank (nested per-group)
        d_rank_remote_blocks = {
            rank: [[20 + i * 10, 21 + i * 10]] for i, rank in enumerate(target_d_ranks)
        }

        with patch.object(
            prefill_worker, "_send_blocks", return_value=0
        ) as mock_send_blocks:
            for d_rank in target_d_ranks:
                remote_block_ids = d_rank_remote_blocks[d_rank]
                xfer_meta = _xfer_meta(
                    [
                        _region(
                            0x2000,
                            block_len=remote_block_len,
                            row_offset=-1,
                            layer_name="model.layers.0.self_attn",
                            layer_index=0,
                        )
                    ],
                    {
                        f"d-req-h1-r{d_rank}": (
                            transfer_id,
                            remote_block_ids,
                        )
                    },
                    remote_tp_size=d_tp_size,
                    remote_tp_rank=d_rank,
                )

                mock_send_blocks.reset_mock()
                mock_socket.send_multipart.reset_mock()

                await prefill_worker.send_kv_to_decode(identity, mock_socket, xfer_meta)

                # Verify _send_blocks was called
                mock_send_blocks.assert_called_once()
                call_args = mock_send_blocks.call_args[0]
                src_ptrs = call_args[1]
                dst_ptrs = call_args[2]
                lengths = call_args[3]

                # Flatten nested per-group block IDs for assertions
                flat_local = [b for g in local_block_ids for b in g]
                flat_remote = [b for g in remote_block_ids for b in g]
                num_blocks = len(flat_local)

                assert len(src_ptrs) == num_blocks
                assert len(dst_ptrs) == num_blocks
                assert len(lengths) == num_blocks

                if d_tp_size <= P_TP_SIZE:
                    tp_ratio = P_TP_SIZE // d_tp_size
                    expected_src_off = 0
                    expected_dst_off = (P_TP_RANK % tp_ratio) * local_block_len
                    expected_xfer_len = local_block_len
                else:
                    ratio_abs = d_tp_size // P_TP_SIZE
                    expected_src_off = (d_rank % ratio_abs) * remote_block_len
                    expected_dst_off = 0
                    expected_xfer_len = remote_block_len

                local_region_base = 0x1000
                remote_region_base = 0x2000
                for blk_idx, (lblk, rblk) in enumerate(zip(flat_local, flat_remote)):
                    assert src_ptrs[blk_idx] == (
                        local_region_base + lblk * local_block_len + expected_src_off
                    )
                    assert dst_ptrs[blk_idx] == (
                        remote_region_base + rblk * remote_block_len + expected_dst_off
                    )
                    assert lengths[blk_idx] == expected_xfer_len

                # Verify successful response sent back to consumer
                mock_socket.send_multipart.assert_called_once()
                _, sent_payload = mock_socket.send_multipart.call_args[0][0]
                response = prefill_worker._xfer_resp_decoder.decode(sent_payload)
                assert response.status == MooncakeXferResponseStatus.FINISH
                assert response.ok_reqs == [f"d-req-h1-r{d_rank}"]

        # After serving all D ranks, the request should be complete
        assert transfer_id not in prefill_worker.reqs_need_send
        assert "p-req-h1" in prefill_worker.finished_sending_reqs

        prefill_worker.sender_loop = origin_sender_loop
        prefill_worker.shutdown()


def test_sleep_unregisters_and_restores_kv_caches():
    spec = FullAttentionSpec(
        block_size=maybe_update_block_size(16),
        num_kv_heads=4,
        head_size=64,
        dtype=torch.float16,
    )
    layer_names = ["model.layers.0.self_attn", "model.layers.1.self_attn"]
    config = _make_test_kv_cache_config()
    with mooncake_register_worker(config) as (worker, reg):
        for layer_name in layer_names:
            worker._layer_specs[layer_name] = spec
        raw = torch.zeros(4 * config.large_page_bytes, dtype=torch.int8)
        worker.register_kv_caches(
            dict(zip(layer_names, _lp_layer_views(raw, config).values()))
        )
        reg.reset_mock()
        with patch.object(
            worker.engine, "batch_unregister_memory", return_value=0
        ) as unreg:
            worker.unregister_kv_caches()
            worker.unregister_kv_caches()
        unreg.assert_called_once_with([raw.data_ptr()])
        worker.restore_kv_caches()
        worker.restore_kv_caches()
        reg.assert_called_once_with([raw.data_ptr()], [raw.nbytes])
