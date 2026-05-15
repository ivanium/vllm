# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock, patch

from vllm.config import set_current_vllm_config
from vllm.distributed.kv_events import BlockStored
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake import (
    mooncake_store_connector,
    mooncake_store_scheduler,
    protocol,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_store_data import (
    MooncakeStoreConnectorMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_store_metrics import (  # noqa: E501
    MooncakeStoreConnectorStats,
)
from vllm.v1.outputs import KVConnectorOutput

from .utils import create_vllm_config


def _make_vllm_config():
    return create_vllm_config(
        kv_connector="MooncakeStoreConnector",
        kv_role="kv_both",
    )


def _make_block_stored() -> BlockStored:
    return BlockStored(
        block_hashes=[b"hash"],
        parent_block_hash=None,
        token_ids=[1, 2, 3],
        block_size=16,
        lora_id=None,
        medium="cpu",
        lora_name=None,
    )


def test_scheduler_role_initializes_store_scheduler_only():
    vllm_config = _make_vllm_config()

    with (
        set_current_vllm_config(vllm_config),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake."
            "mooncake_store_connector.MooncakeStoreScheduler"
        ) as mock_scheduler,
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake."
            "mooncake_store_connector.MooncakeStoreWorker"
        ) as mock_worker,
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake."
            "mooncake_store_connector.LookupKeyServer"
        ) as mock_lookup_server,
    ):
        connector = mooncake_store_connector.MooncakeStoreConnector(
            vllm_config, KVConnectorRole.SCHEDULER
        )

    mock_scheduler.assert_called_once_with(vllm_config)
    mock_worker.assert_not_called()
    mock_lookup_server.assert_not_called()
    assert connector.connector_scheduler is mock_scheduler.return_value
    assert connector.connector_worker is None


def test_worker_role_initializes_store_worker_and_lookup_server_on_rank0():
    vllm_config = _make_vllm_config()

    with (
        set_current_vllm_config(vllm_config),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake."
            "mooncake_store_connector.MooncakeStoreScheduler"
        ) as mock_scheduler,
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake."
            "mooncake_store_connector.MooncakeStoreWorker"
        ) as mock_worker,
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake."
            "mooncake_store_connector.LookupKeyServer"
        ) as mock_lookup_server,
    ):
        connector = mooncake_store_connector.MooncakeStoreConnector(
            vllm_config, KVConnectorRole.WORKER
        )

    mock_scheduler.assert_not_called()
    mock_worker.assert_called_once_with(vllm_config)
    mock_lookup_server.assert_called_once_with(mock_worker.return_value, vllm_config)
    assert connector.connector_scheduler is None
    assert connector.connector_worker is mock_worker.return_value


def test_worker_role_skips_lookup_server_on_nonzero_rank():
    vllm_config = _make_vllm_config()
    vllm_config.parallel_config.rank = 1

    with (
        set_current_vllm_config(vllm_config),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake."
            "mooncake_store_connector.MooncakeStoreWorker"
        ) as mock_worker,
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake."
            "mooncake_store_connector.LookupKeyServer"
        ) as mock_lookup_server,
    ):
        mooncake_store_connector.MooncakeStoreConnector(
            vllm_config, KVConnectorRole.WORKER
        )

    mock_worker.assert_called_once_with(vllm_config)
    mock_lookup_server.assert_not_called()


def test_lookup_rpc_path_uses_data_parallel_index_in_dense_dp():
    vllm_config = _make_vllm_config()
    vllm_config.parallel_config.data_parallel_rank = 0
    vllm_config.parallel_config.data_parallel_index = 3

    path = mooncake_store_scheduler.get_zmq_rpc_path_lookup(vllm_config)

    assert path.endswith("_dp_rank3")


def test_lookup_rpc_path_uses_local_rank_when_local_engines_only():
    vllm_config = _make_vllm_config()
    vllm_config.parallel_config.data_parallel_index = 7
    vllm_config.parallel_config.data_parallel_rank_local = 1
    vllm_config.parallel_config.data_parallel_hybrid_lb = True

    path = mooncake_store_scheduler.get_zmq_rpc_path_lookup(vllm_config)

    assert path.endswith("_dp_rank1")


def test_worker_methods_delegate_to_store_worker():
    vllm_config = _make_vllm_config()
    kv_caches = {"layer0": MagicMock()}
    metadata = MooncakeStoreConnectorMetadata(set(), set())
    finished_req_ids = {"req-1"}

    with (
        set_current_vllm_config(vllm_config),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake."
            "mooncake_store_connector.MooncakeStoreWorker"
        ) as mock_worker_cls,
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake."
            "mooncake_store_connector.LookupKeyServer"
        ),
    ):
        connector = mooncake_store_connector.MooncakeStoreConnector(
            vllm_config, KVConnectorRole.WORKER
        )

    worker = mock_worker_cls.return_value
    worker.get_finished.return_value = ({"req-1"}, {"req-2"})
    connector.bind_connector_metadata(metadata)

    connector.register_kv_caches(kv_caches)
    result = connector.get_finished(finished_req_ids)

    worker.register_kv_caches.assert_called_once_with(kv_caches)
    worker.get_finished.assert_called_once_with(finished_req_ids, metadata)
    assert result == ({"req-1"}, {"req-2"})


def test_get_kv_connector_kv_cache_events_returns_none_when_empty():
    vllm_config = _make_vllm_config()

    with (
        set_current_vllm_config(vllm_config),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake."
            "mooncake_store_connector.MooncakeStoreWorker"
        ) as mock_worker_cls,
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake."
            "mooncake_store_connector.LookupKeyServer"
        ),
    ):
        connector = mooncake_store_connector.MooncakeStoreConnector(
            vllm_config, KVConnectorRole.WORKER
        )

    mock_worker_cls.return_value.get_kv_events.return_value = []
    assert connector.get_kv_connector_kv_cache_events() is None


def test_get_kv_connector_stats_delegates_to_worker():
    vllm_config = _make_vllm_config()
    expected_stats = MooncakeStoreConnectorStats()
    expected_stats.record_operation("save_put", 0.01, 2, num_bytes=1024)

    with (
        set_current_vllm_config(vllm_config),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake."
            "mooncake_store_connector.MooncakeStoreWorker"
        ) as mock_worker_cls,
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake."
            "mooncake_store_connector.LookupKeyServer"
        ),
    ):
        connector = mooncake_store_connector.MooncakeStoreConnector(
            vllm_config, KVConnectorRole.WORKER
        )

    mock_worker_cls.return_value.get_kv_connector_stats.return_value = expected_stats
    stats = connector.get_kv_connector_stats()

    assert stats is expected_stats
    mock_worker_cls.return_value.get_kv_connector_stats.assert_called_once_with()


def test_build_kv_connector_stats_reconstructs_mooncake_stats():
    stats = mooncake_store_connector.MooncakeStoreConnector.build_kv_connector_stats(
        {
            "save_put": [
                {
                    "duration_seconds": 0.02,
                    "num_keys": 4,
                    "num_bytes": 2048,
                    "status": "ok",
                    "num_failed_keys": 0,
                }
            ]
        }
    )

    assert isinstance(stats, MooncakeStoreConnectorStats)
    assert stats.data["save_put"][0]["num_bytes"] == 2048


def test_get_kv_connector_kv_cache_events_wraps_worker_events():
    vllm_config = _make_vllm_config()
    event = _make_block_stored()

    with (
        set_current_vllm_config(vllm_config),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake."
            "mooncake_store_connector.MooncakeStoreWorker"
        ) as mock_worker_cls,
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake."
            "mooncake_store_connector.LookupKeyServer"
        ),
    ):
        connector = mooncake_store_connector.MooncakeStoreConnector(
            vllm_config, KVConnectorRole.WORKER
        )

    mock_worker_cls.return_value.get_kv_events.return_value = [event]
    kv_events = connector.get_kv_connector_kv_cache_events()

    assert isinstance(kv_events, mooncake_store_connector.MooncakeStoreKVEvents)
    assert kv_events.get_number_of_workers() == 1
    assert kv_events.get_all_events() == [event]


def test_prefer_cross_layer_blocks_from_config():
    # Default: disabled
    vllm_config = _make_vllm_config()
    with (
        set_current_vllm_config(vllm_config),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake."
            "mooncake_store_connector.MooncakeStoreScheduler"
        ),
    ):
        connector = mooncake_store_connector.MooncakeStoreConnector(
            vllm_config, KVConnectorRole.SCHEDULER
        )
    assert connector.prefer_cross_layer_blocks is False

    # Enabled via config
    vllm_config_enabled = create_vllm_config(
        kv_connector="MooncakeStoreConnector",
        kv_role="kv_both",
        kv_connector_extra_config={"enable_cross_layers_blocks": "true"},
    )
    with (
        set_current_vllm_config(vllm_config_enabled),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake."
            "mooncake_store_connector.MooncakeStoreScheduler"
        ),
    ):
        connector_enabled = mooncake_store_connector.MooncakeStoreConnector(
            vllm_config_enabled, KVConnectorRole.SCHEDULER
        )
    assert connector_enabled.prefer_cross_layer_blocks is True


def test_register_cross_layers_kv_cache_delegates_to_worker():
    vllm_config = _make_vllm_config()

    with (
        set_current_vllm_config(vllm_config),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake."
            "mooncake_store_connector.MooncakeStoreWorker"
        ) as mock_worker_cls,
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake."
            "mooncake_store_connector.LookupKeyServer"
        ),
    ):
        connector = mooncake_store_connector.MooncakeStoreConnector(
            vllm_config, KVConnectorRole.WORKER
        )

    fake_tensor = MagicMock()
    fake_backend = MagicMock()
    connector.register_cross_layers_kv_cache(fake_tensor, fake_backend)

    worker = mock_worker_cls.return_value
    worker.register_cross_layers_kv_caches.assert_called_once_with(fake_tensor)


def test_update_connector_output_and_take_events():
    vllm_config = _make_vllm_config()
    event = _make_block_stored()

    with (
        set_current_vllm_config(vllm_config),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake."
            "mooncake_store_connector.MooncakeStoreScheduler"
        ),
    ):
        connector = mooncake_store_connector.MooncakeStoreConnector(
            vllm_config, KVConnectorRole.SCHEDULER
        )

    kv_events = mooncake_store_connector.MooncakeStoreKVEvents(num_workers=1)
    kv_events.add_events([event])
    connector.update_connector_output(KVConnectorOutput(kv_cache_events=kv_events))

    assert connector._kv_cache_events is kv_events
    assert list(connector.take_events()) == [event]
    assert connector._kv_cache_events is None


# ============================================================
# reset_cache() — RL hard-reset path via typed LookupKey protocol
# ============================================================


def test_reset_cache_scheduler_role_delegates_to_reset_store():
    """SCHEDULER role reset_cache() routes to scheduler.reset_store()."""
    vllm_config = _make_vllm_config()

    with (
        set_current_vllm_config(vllm_config),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake."
            "mooncake_store_connector.MooncakeStoreScheduler"
        ) as mock_scheduler_cls,
    ):
        connector = mooncake_store_connector.MooncakeStoreConnector(
            vllm_config, KVConnectorRole.SCHEDULER
        )

    mock_scheduler_cls.return_value.reset_store.return_value = True
    assert connector.reset_cache() is True
    mock_scheduler_cls.return_value.reset_store.assert_called_once_with()


def test_reset_cache_scheduler_role_propagates_failure():
    """SCHEDULER role surfaces False when scheduler.reset_store() fails."""
    vllm_config = _make_vllm_config()

    with (
        set_current_vllm_config(vllm_config),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake."
            "mooncake_store_connector.MooncakeStoreScheduler"
        ) as mock_scheduler_cls,
    ):
        connector = mooncake_store_connector.MooncakeStoreConnector(
            vllm_config, KVConnectorRole.SCHEDULER
        )

    mock_scheduler_cls.return_value.reset_store.return_value = False
    assert connector.reset_cache() is False


def test_reset_cache_worker_role_returns_none():
    """WORKER role reset_cache() is a no-op; reset is driven via ZMQ admin."""
    vllm_config = _make_vllm_config()

    with (
        set_current_vllm_config(vllm_config),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake."
            "mooncake_store_connector.MooncakeStoreWorker"
        ),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake."
            "mooncake_store_connector.LookupKeyServer"
        ),
    ):
        connector = mooncake_store_connector.MooncakeStoreConnector(
            vllm_config, KVConnectorRole.WORKER
        )

    assert connector.reset_cache() is None


def test_scheduler_reset_store_returns_client_reset_result():
    """MooncakeStoreScheduler.reset_store() returns LookupKeyClient.reset()."""
    vllm_config = _make_vllm_config()

    with (
        set_current_vllm_config(vllm_config),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake."
            "mooncake_store_scheduler.LookupKeyClient"
        ) as mock_client_cls,
    ):
        sched = mooncake_store_scheduler.MooncakeStoreScheduler(vllm_config)

    mock_client_cls.return_value.reset.return_value = True
    assert sched.reset_store() is True
    mock_client_cls.return_value.reset.assert_called_once_with()


def test_scheduler_reset_store_handles_rpc_exception():
    """Exceptions from the ZMQ reset RPC are converted to False, not raised."""
    vllm_config = _make_vllm_config()

    with (
        set_current_vllm_config(vllm_config),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake."
            "mooncake_store_scheduler.LookupKeyClient"
        ) as mock_client_cls,
    ):
        sched = mooncake_store_scheduler.MooncakeStoreScheduler(vllm_config)

    mock_client_cls.return_value.reset.side_effect = RuntimeError("rpc timed out")
    assert sched.reset_store() is False


def test_lookup_key_client_reset_uses_admin_protocol():
    """LookupKeyClient.reset() sends RESET_MSG tag and parses RESP_OK/ERR."""
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.protocol import (
        RESET_MSG,
        RESP_ERR,
        RESP_OK,
    )

    vllm_config = _make_vllm_config()

    # Avoid touching real ZMQ during construction.
    with patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.mooncake."
        "mooncake_store_scheduler.make_zmq_socket"
    ) as mock_make_socket:
        client = mooncake_store_scheduler.LookupKeyClient(vllm_config)

    fake_socket = mock_make_socket.return_value
    fake_socket.recv.return_value = RESP_OK
    assert client.reset() is True

    sent_frame = fake_socket.send.call_args[0][0]
    assert bytes(sent_frame) == RESET_MSG

    # NACK path: server returns RESP_ERR, client surfaces False.
    fake_socket.recv.return_value = RESP_ERR
    assert client.reset() is False


def test_scheduler_reset_connector_cache_invokes_connector_reset():
    """Integration-shaped test for the cascade

    ``Scheduler.reset_prefix_cache(reset_connector=True)``
        -> ``Scheduler.reset_connector_cache()``
        -> ``MooncakeStoreConnector.reset_cache()``
        -> ``MooncakeStoreScheduler.reset_store()``.

    We construct a real (SCHEDULER role) ``MooncakeStoreConnector`` with
    ``MooncakeStoreScheduler`` patched out, then drive the cascade via a
    minimal stub that mimics the Scheduler's call site.
    """
    vllm_config = _make_vllm_config()

    with (
        set_current_vllm_config(vllm_config),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake."
            "mooncake_store_connector.MooncakeStoreScheduler"
        ) as mock_scheduler_cls,
    ):
        connector = mooncake_store_connector.MooncakeStoreConnector(
            vllm_config, KVConnectorRole.SCHEDULER
        )

    mock_scheduler_cls.return_value.reset_store.return_value = True

    # Mirror Scheduler.reset_connector_cache (scheduler.py:1917-1929):
    # the only relevant lines are
    #   if self.connector.reset_cache() is False:
    #       return False
    #   return True
    # We bypass the Scheduler class itself (heavy KVCacheManager setup)
    # and exercise the same call shape.
    class _StubScheduler:
        def __init__(self, c):
            self.connector = c
            self.log_stats = False

        def reset_connector_cache(self):
            if self.connector.reset_cache() is False:
                return False
            return True

    sched = _StubScheduler(connector)
    assert sched.reset_connector_cache() is True
    mock_scheduler_cls.return_value.reset_store.assert_called_once_with()

    # Negative branch: store reset fails -> reset_connector_cache returns False.
    mock_scheduler_cls.return_value.reset_store.reset_mock()
    mock_scheduler_cls.return_value.reset_store.return_value = False
    assert sched.reset_connector_cache() is False
