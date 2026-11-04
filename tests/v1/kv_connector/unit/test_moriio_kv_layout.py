# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MoRIIO transfer geometry over large-page kernel-row views.

Every layer view is the standardized ``[K, H, N, C]`` (one kernel block per
row, rows strided by the group's row bytes over the shared pool); manager
block ``b`` expands to kernel rows ``b * R`` to ``b * R + R - 1``.
"""

import importlib.util
import math
import threading
from collections import OrderedDict
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import torch

from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    GroupGeometry,
    MambaSpec,
    MLAAttentionSpec,
)

from .utils import make_moriio_writer

moriio_common = importlib.import_module(
    "vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_common"
)
moriio_engine = importlib.import_module(
    "vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_engine"
)
moriio_layout = importlib.import_module(
    "vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_layout"
)
msgpack = importlib.import_module("msgpack")
ssm_conv_transfer_utils = importlib.import_module(
    "vllm.distributed.kv_transfer.kv_connector.v1.ssm_conv_transfer_utils"
)
MambaConvSplitInfo = ssm_conv_transfer_utils.MambaConvSplitInfo

ROLE = moriio_common.ROLE
MoRIIOError = moriio_common.MoRIIOError
MoRIIOTransferAck = moriio_common.MoRIIOTransferAck
RemoteAllocInfo = moriio_common.RemoteAllocInfo
WriteTask = moriio_common.WriteTask
set_role = moriio_common.set_role
MoRIIOWrapper = moriio_engine.MoRIIOWrapper
MoRIIOWriter = moriio_engine.MoRIIOWriter

LayerTransferGeometry = moriio_layout.LayerTransferGeometry


def _full_spec(
    block_size: int = 16, num_kv_heads: int = 2, head_size: int = 2
) -> FullAttentionSpec:
    return FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=torch.bfloat16,
    )


def _mla_spec(block_size: int = 16) -> MLAAttentionSpec:
    return MLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.bfloat16,
    )


def _view(
    pool: torch.Tensor,
    row_bytes: int,
    num_rows: int,
    spec,
    offset: int = 0,
    head_major: bool = False,
) -> torch.Tensor:
    """One layer's [K, H, N, C] uint8 view at ``offset`` within each row."""
    heads, tokens, content = (
        spec.num_heads,
        spec.num_states,
        spec.state_content_size_bytes,
    )
    if head_major:
        strides = (row_bytes, content, heads * content, 1)
    else:
        strides = (row_bytes, tokens * content, content, 1)
    return torch.as_strided(
        pool, (num_rows, heads, tokens, content), strides, storage_offset=offset
    )


def test_layer_geometry_from_kernel_row_view():
    # row 1024; the layer holds 2 heads * 16 tokens * 8 content bytes per row.
    spec = _full_spec()
    pool = torch.zeros(2 * 16 * 1024, dtype=torch.uint8)
    view = _view(pool, 1024, 8, spec, offset=512)

    geometry = moriio_layout.get_layer_transfer_geometry("layer", view, {"layer": spec})

    assert geometry == LayerTransferGeometry(
        num_rows=8,
        tokens_per_row=16,
        row_len=2 * 16 * 8,
        row_stride=1024,
    )


def test_layer_geometry_rejects_non_large_page_views():
    spec = _full_spec()
    cache = torch.empty((2, 8, 16, 2, 8), dtype=torch.bfloat16)  # 5-D split K/V
    with pytest.raises(ValueError, match="must be \\[K, H, N, C\\]"):
        moriio_layout.get_layer_transfer_geometry("layer", cache, {"layer": spec})

    with pytest.raises(ValueError, match="Missing KV cache spec"):
        moriio_layout.get_layer_transfer_geometry("layer", cache, {})


def test_expand_block_kernel_rows_keeps_block_order():
    group = GroupGeometry(kernel_rows_per_block=2, row_bytes=512)
    rows = moriio_layout.expand_block_kernel_rows(group, 3, [2, 0])
    assert rows == [4, 5, 0, 1]


def test_expand_block_kernel_rows_bounds():
    group = GroupGeometry(kernel_rows_per_block=1, row_bytes=512)
    with pytest.raises(ValueError, match="block id 3 is outside"):
        moriio_layout.expand_block_kernel_rows(group, 3, [3])
    with pytest.raises(ValueError, match="block id -1 is outside"):
        moriio_layout.expand_block_kernel_rows(group, 3, [-1])


def test_block_transfer_offsets_pair_rows_across_peer_geometries():
    """Local and remote expand the same manager ids with their own geometry.

    Local: row 512, R=2 -> block 1 is rows 2, 3. Remote (half the heads):
    row 256, R=2 -> block 3 is rows 6, 7. Rows pair 1:1; sizes are the
    layer's per-row bytes.
    """
    spec = _full_spec()
    pool = torch.zeros(2 * 16 * 1280, dtype=torch.uint8)
    view = _view(pool, 512, 8, spec)
    geometry = moriio_layout.get_layer_transfer_geometry("layer", view, {"layer": spec})
    local_group = GroupGeometry(2, 512)
    remote_group = GroupGeometry(2, 256)

    local, remote, sizes = moriio_layout.compute_block_transfer_offsets(
        geometry, local_group, 3, remote_group, 4, [1], [3]
    )

    assert local == [2 * 512, 3 * 512]
    assert remote == [6 * 256, 7 * 256]
    assert sizes == [256, 256]


def test_block_transfer_offsets_merge_dense_consecutive_rows():
    # Single-layer group: one row is exactly the layer content, so adjacent
    # rows merge into one read on both sides.
    spec = _full_spec(num_kv_heads=1, head_size=8)
    row = spec.page_size_bytes
    pool = torch.zeros(2 * 16 * row, dtype=torch.uint8)
    view = _view(pool, row, 16, spec)
    geometry = moriio_layout.get_layer_transfer_geometry("layer", view, {"layer": spec})
    group = GroupGeometry(1, row)

    local, remote, sizes = moriio_layout.compute_block_transfer_offsets(
        geometry, group, 16, group, 16, [0, 1], [4, 5]
    )

    # Local blocks 0, 1 are rows 0, 1; remote blocks 4, 5 are rows 4, 5.
    assert local == [0]
    assert remote == [4 * row]
    assert sizes == [2 * row]


def test_block_transfer_offsets_rejects_kernel_row_split_mismatch():
    spec = _full_spec()
    pool = torch.zeros(2 * 16 * 1024, dtype=torch.uint8)
    view = _view(pool, 512, 8, spec)
    geometry = moriio_layout.get_layer_transfer_geometry("layer", view, {"layer": spec})
    local_group = GroupGeometry(2, 512)
    remote_group = GroupGeometry(1, 512)

    with pytest.raises(ValueError, match="kernel rows"):
        moriio_layout.compute_block_transfer_offsets(
            geometry, local_group, 4, remote_group, 8, [0], [0]
        )


def test_block_transfer_offsets_bounds_use_each_sides_group():
    spec = _full_spec()
    pool = torch.zeros(2 * 16 * 1024, dtype=torch.uint8)
    view = _view(pool, 512, 8, spec)
    geometry = moriio_layout.get_layer_transfer_geometry("layer", view, {"layer": spec})
    local_group = GroupGeometry(1, 512)
    # The peer's pool is smaller: its group has 4 blocks, not 8.
    remote_group = GroupGeometry(1, 512)

    with pytest.raises(ValueError, match="block id 8 is outside"):
        moriio_layout.compute_block_transfer_offsets(
            geometry, local_group, 8, remote_group, 4, [8], [0]
        )
    with pytest.raises(ValueError, match="block id 4 is outside"):
        moriio_layout.compute_block_transfer_offsets(
            geometry, local_group, 8, remote_group, 4, [0], [4]
        )


def test_local_block_ids_longer_than_remote_raises():
    # A longer local list is a genuine bug here and must fail loudly; the WRITE
    # producer save path trims lookahead blocks before they reach this layer.
    spec = _full_spec()
    pool = torch.zeros(2 * 16 * 1024, dtype=torch.uint8)
    view = _view(pool, 512, 8, spec)
    geometry = moriio_layout.get_layer_transfer_geometry("layer", view, {"layer": spec})
    group = GroupGeometry(1, 512)

    with pytest.raises(ValueError, match="longer than remote_block_ids"):
        moriio_layout.compute_block_transfer_offsets(
            geometry, group, 8, group, 8, [1, 3, 6, 7], [4, 5]
        )


def test_shorter_local_block_ids_still_transfers_only_what_decode_allocated():
    # READ-mode partial transfer keeps its shorter-local behavior.
    spec = _full_spec()
    pool = torch.zeros(2 * 16 * 1024, dtype=torch.uint8)
    view = _view(pool, 512, 8, spec)
    geometry = moriio_layout.get_layer_transfer_geometry("layer", view, {"layer": spec})
    group = GroupGeometry(1, 512)

    local, _, _ = moriio_layout.compute_block_transfer_offsets(
        geometry, group, 8, group, 8, [1], [4, 5]
    )
    assert len(local) == 1


def test_empty_local_block_ids_is_free_only_noop():
    spec = _full_spec()
    pool = torch.zeros(2 * 16 * 1024, dtype=torch.uint8)
    view = _view(pool, 512, 8, spec)
    geometry = moriio_layout.get_layer_transfer_geometry("layer", view, {"layer": spec})
    group = GroupGeometry(1, 512)

    assert moriio_layout.compute_block_transfer_offsets(
        geometry, group, 8, group, 8, [], [4, 5]
    ) == ([], [], [])


def test_registration_region_spans_strided_rows():
    spec = _full_spec()
    pool = torch.zeros(2 * 16 * 1024, dtype=torch.uint8)
    view = _view(pool, 1024, 8, spec, offset=512)

    regions = moriio_layout.iter_layer_registration_regions(
        "layer", view, {"layer": spec}
    )

    assert len(regions) == 1
    assert regions[0][0].data_ptr() == view.data_ptr()
    # Strided span: every row, ending at the last row's content.
    assert regions[0][1] == 7 * 1024 + 256


def test_head_slice_offsets_address_kernel_rows():
    # Head-major rows: [K, 4, 2, 16], head stride 16, token stride 64,
    # row 128.
    spec = FullAttentionSpec(
        block_size=2, num_kv_heads=4, head_size=4, dtype=torch.bfloat16
    )
    pool = torch.zeros(2 * 16 * 128, dtype=torch.uint8)
    view = _view(pool, 128, 16, spec, head_major=True)
    layout = moriio_layout.get_layer_head_layout("layer", view, {"layer": spec})
    assert layout.block_stride == 128

    # Write local heads [1, 3) of row 2 into remote heads [0, 2) of row 5:
    # one run of two adjacent heads per token.
    local, remote, sizes = moriio_layout.compute_head_slice_offsets(
        layout, layout, [2], [5], 1, 0, 2
    )

    assert local == [2 * 128 + 16, 2 * 128 + 64 + 16]
    assert remote == [5 * 128, 5 * 128 + 64]
    assert sizes == [32, 32]


def test_two_groups_expand_their_own_geometry():
    """Full + sliding-window groups: same block ids, different row grids.

    This is the large-page contract the old scalar-num_blocks handshake
    broke: each group's pool geometry (row bytes, block count) is its own.
    """
    full_spec = _full_spec()
    swa_spec = _full_spec(num_kv_heads=1)
    pool = torch.zeros(2 * 16 * 1280, dtype=torch.uint8)
    full_view = _view(pool, 512, 7, full_spec, offset=0)
    swa_view = _view(pool, 256, 15, swa_spec, offset=0)
    layer_to_spec = {"full0": full_spec, "swa0": swa_spec}
    full_geometry = moriio_layout.get_layer_transfer_geometry(
        "full0", full_view, layer_to_spec
    )
    swa_geometry = moriio_layout.get_layer_transfer_geometry(
        "swa0", swa_view, layer_to_spec
    )
    full_group = GroupGeometry(2, 512)
    swa_group = GroupGeometry(2, 256)

    full_offsets = moriio_layout.compute_block_transfer_offsets(
        full_geometry, full_group, 3, full_group, 3, [1], [2]
    )
    swa_offsets = moriio_layout.compute_block_transfer_offsets(
        swa_geometry, swa_group, 6, swa_group, 6, [1], [2]
    )

    # Block 1 is rows 2-3 of 512B rows in the full group and of 256B rows in
    # the swa group. Row bytes are the layers' per-row content (256 and 128).
    assert full_offsets == ([2 * 512, 3 * 512], [4 * 512, 5 * 512], [256, 256])
    assert swa_offsets == ([2 * 256, 3 * 256], [4 * 256, 5 * 256], [128, 128])
    # And the swa group admits block ids the full group does not have.
    assert moriio_layout.compute_block_transfer_offsets(
        swa_geometry, swa_group, 6, swa_group, 6, [5], [4]
    )[0] == [10 * 256, 11 * 256]
    with pytest.raises(ValueError, match="outside"):
        moriio_layout.compute_block_transfer_offsets(
            full_geometry, full_group, 3, full_group, 3, [5], [4]
        )


def _remote_meta() -> SimpleNamespace:
    return SimpleNamespace()


def _write_task(layer_name: str, transfer_id: str = "xfer") -> Any:
    return WriteTask(
        request_id="req",
        transfer_id=transfer_id,
        dst_engine_id="remote-engine",
        local_block_ids=[[1, 3]],
        remote_block_ids_hint=None,
        layer_name=layer_name,
        event=None,
        remote_notify_port=7000,
        remote_ip="127.0.0.1",
    )


def test_write_transfer_plan_caches_offsets_per_layer_geometry():
    kv_caches = {
        "dense0": torch.empty((8, 2, 4, 2, 3), dtype=torch.bfloat16),
        "dense1": torch.empty((8, 2, 4, 2, 3), dtype=torch.bfloat16),
        "indexer": torch.empty((8, 1, 4, 3), dtype=torch.bfloat16),
    }
    calls: list[str] = []

    class FakeWorker:
        _has_mamba: bool
        layer_to_group: dict[str, int]
        kv_caches: dict[str, torch.Tensor]
        layer_name_to_local_kv_cache_metadata: dict[str, list[Any]]

        def _region_session_indices(self, layer_name):
            return [list(self.kv_caches).index(layer_name)]

        def _compute_block_transfer_offsets(
            self,
            layer_name,
            local_block_ids,
            remote_block_ids,
            remote_moriio_meta,
            remote_engine_id=None,
        ):
            calls.append(layer_name)
            call_id = len(calls)
            return ([call_id], [call_id + 10], [call_id + 20])

    fake_worker = FakeWorker()
    fake_worker._has_mamba = False
    fake_worker.layer_to_group = dict.fromkeys(kv_caches, 0)
    fake_worker.kv_caches = kv_caches
    fake_worker.layer_name_to_local_kv_cache_metadata = {name: [] for name in kv_caches}
    writer = MoRIIOWriter.__new__(MoRIIOWriter)
    writer._worker_ref = lambda: fake_worker
    request_info = RemoteAllocInfo(block_ids=[[4, 5]])
    remote_meta = _remote_meta()

    dense0_plan = writer._prepare_transfer_plan(
        _write_task("dense0"),
        request_info,
        remote_meta,
    )
    dense1_plan = writer._prepare_transfer_plan(
        _write_task("dense1"),
        request_info,
        remote_meta,
    )
    indexer_plan = writer._prepare_transfer_plan(
        _write_task("indexer"),
        request_info,
        remote_meta,
    )

    assert calls == ["dense0", "dense1", "indexer"]
    assert dense0_plan.transfer_local_offsets == [1]
    assert dense1_plan.transfer_local_offsets == [2]
    assert indexer_plan.transfer_local_offsets == [3]
    assert len(request_info.transfer_offsets) == 3


def test_write_scheduler_deduplicates_layers_and_seals_expected_count():
    request_info = RemoteAllocInfo(block_ids=[[4, 5]])
    wrapper = _wrapper_for_messages()
    wrapper.done_remote_allocate_req_dict["xfer"] = request_info
    writer = make_moriio_writer(SimpleNamespace(moriio_wrapper=wrapper))

    assert writer.schedule_write(_write_task("dense0"))
    assert not writer.schedule_write(_write_task("dense0"))
    assert writer.schedule_write(_write_task("indexer"))

    assert writer._write_task_q.qsize() == 2
    writer.seal_pending_transfers()

    assert request_info.writes_expected == 2
    assert writer._sealed_writes["xfer"] == 2


def test_hybrid_write_attention_uses_its_own_group_not_the_first_group():
    worker = SimpleNamespace(
        _has_mamba=True,
        kv_caches={"draft": torch.empty((8, 1, 4, 3))},
        layer_to_group={"draft": 2},
        _region_session_indices=lambda _: [5],
        _compute_block_transfer_offsets=Mock(return_value=([1], [2], [3])),
    )
    writer = make_moriio_writer(worker)
    task = _write_task("draft")
    task.local_block_ids = [[10], [20], [30, 31]]
    remote = RemoteAllocInfo(block_ids=[[40], [50], [60, 61, 62]])
    meta = _remote_meta()
    plan = writer._prepare_transfer_plan(task, remote, meta)
    worker._compute_block_transfer_offsets.assert_called_once_with(
        "draft", [30, 31], [60, 61], meta, remote_engine_id=task.dst_engine_id
    )
    assert plan.sess_idx == 5


def test_hybrid_write_posts_both_regions_before_layer_completion():
    from unittest.mock import Mock

    from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector import (
        MoRIIOConnectorWorker,
    )

    wrapper = _wrapper_for_messages()
    info = RemoteAllocInfo(block_ids=[[], [7], [9]])
    wrapper.done_remote_allocate_req_dict["xfer"] = info
    wrapper.write_remote_data = Mock(side_effect=["conv-status", "ssm-status"])
    peer_meta = SimpleNamespace()
    worker = SimpleNamespace(
        moriio_wrapper=wrapper,
        tp_rank=0,
        get_engine_name_with_dp=lambda name, rank: name,
        _get_built_session=lambda name: (["conv-session", "ssm-session"], peer_meta),
        _is_mamba_layer=lambda name: True,
        _mamba_payload_index_by_layer={"kda": 2},
        _region_session_indices=lambda name: [0, 1],
        _compute_mamba_transfer_offsets=Mock(
            return_value=([10, 20, 30, 40], [50, 60, 70, 80], [2, 4, 8, 16], 3)
        ),
    )
    worker._mamba_blocks_for_layer = lambda name, blocks: (
        MoRIIOConnectorWorker._mamba_blocks_for_layer(worker, name, blocks)
    )
    writer = make_moriio_writer(worker)
    task = _write_task("kda")
    task.local_block_ids = [[], [1], [3]]
    task.remote_tp_size = 2
    task.event = Mock()
    writer._execute_write_task(task)

    task.event.synchronize.assert_called_once()
    worker._compute_mamba_transfer_offsets.assert_called_once_with(
        "kda", [3], [9], peer_meta, remote_tp_size=2
    )
    assert wrapper.write_remote_data.call_args_list[0].args == (
        [2, 4, 8],
        [10, 20, 30],
        [50, 60, 70],
        "conv-session",
    )
    assert wrapper.write_remote_data.call_args_list[1].args == (
        [16],
        [40],
        [80],
        "ssm-session",
    )
    assert info.transfer_statuses == ["conv-status", "ssm-status"]
    assert info.writes_done == 1
    assert not info.completion_notified


def test_write_completion_notifies_once_after_all_sealed_writes_finish():
    class FakeWrapper:
        def __init__(self):
            self.done_remote_allocate_req_dict = {}
            self.done_req_ids = []
            self.lock = threading.Lock()
            self.notifications = []
            self.wait_count = 0
            self.waited_statuses = []
            self._terminal_transfer_ids = OrderedDict()

        def waiting_for_transfer_complete(self, transfer_statuses=None):
            self.wait_count += 1
            self.waited_statuses.append(list(transfer_statuses or []))

        def _is_transfer_terminal_locked(self, transfer_id):
            return transfer_id in self._terminal_transfer_ids

        def _mark_transfer_terminal_locked(self, transfer_id):
            self._terminal_transfer_ids[transfer_id] = None

        def send_notify(
            self,
            transfer_id,
            remote_ip,
            remote_port,
            message_type=None,
            message_fields=None,
        ):
            self.notifications.append(
                (transfer_id, remote_ip, remote_port, message_type, message_fields)
            )

    wrapper = FakeWrapper()
    request_info = RemoteAllocInfo(block_ids=[[4, 5]], writes_expected=2)
    request_info.transfer_statuses.extend(["status-a", "status-b"])
    request_info.completion_request_id = "req"
    # Final notify port is resolved in _execute_write_task; finalize reads it
    # as-is, so set the already-offset value (base 7000 + tp_rank-2 offset).
    request_info.completion_notify_port = 7002
    request_info.completion_remote_ip = "127.0.0.1"
    wrapper.done_remote_allocate_req_dict["xfer"] = request_info
    writer = make_moriio_writer(SimpleNamespace(moriio_wrapper=wrapper, tp_rank=2))
    writer._scheduled_writes["xfer"] = 2
    writer._scheduled_layers["xfer"] = {"dense0", "indexer"}
    writer._sealed_writes["xfer"] = 2

    writer._mark_write_done("xfer", request_info)
    assert wrapper.notifications == []
    writer._mark_write_done("xfer", request_info)
    writer._finalize_if_complete("xfer", request_info)

    assert wrapper.notifications == [("xfer", "127.0.0.1", 7002, "write_done", None)]
    assert wrapper.done_req_ids == [MoRIIOTransferAck("xfer")]
    assert wrapper.done_remote_allocate_req_dict == {}
    assert wrapper.wait_count == 1
    assert wrapper.waited_statuses == [["status-a", "status-b"]]
    assert request_info.transfer_statuses == []
    assert wrapper._is_transfer_terminal_locked("xfer")


def test_moriio_wrapper_waits_scoped_statuses_without_global_drain():
    class FakeStatus:
        def __init__(self):
            self.checked = 0

        def Succeeded(self):
            self.checked += 1
            return True

        def Failed(self):
            return False

    wrapper = MoRIIOWrapper.__new__(MoRIIOWrapper)
    wrapper.lock = threading.Lock()
    wrapper._transfer_timeout = 1
    wrapper.moriio_engine = object()
    global_status = FakeStatus()
    scoped_status = FakeStatus()
    wrapper.transfer_status = [global_status]

    wrapper.waiting_for_transfer_complete([scoped_status])

    assert scoped_status.checked == 1
    assert global_status.checked == 0
    assert wrapper.transfer_status == [global_status]


def test_write_failure_marks_terminal_and_clears_scheduled_state():
    wrapper = _wrapper_for_messages()
    wrapper.done_remote_allocate_req_dict["xfer"] = RemoteAllocInfo(block_ids=[[4, 5]])
    writer = make_moriio_writer(SimpleNamespace(moriio_wrapper=wrapper))
    writer._scheduled_writes["xfer"] = 2
    writer._scheduled_layers["xfer"] = {"dense0", "indexer"}
    writer._sealed_writes["xfer"] = 2

    writer._mark_request_done("xfer")

    assert wrapper.done_req_ids == [MoRIIOTransferAck("xfer")]
    assert wrapper.done_remote_allocate_req_dict == {}
    assert wrapper._is_transfer_terminal_locked("xfer")
    assert "xfer" not in writer._scheduled_writes
    assert "xfer" not in writer._scheduled_layers
    assert "xfer" not in writer._sealed_writes


def test_schedule_write_rejects_terminal_transfer_without_recreating_state():
    wrapper = _wrapper_for_messages()
    wrapper.done_remote_allocate_req_dict["xfer"] = RemoteAllocInfo(block_ids=[[4, 5]])
    writer = make_moriio_writer(SimpleNamespace(moriio_wrapper=wrapper))
    writer._scheduled_writes["xfer"] = 1
    writer._scheduled_layers["xfer"] = {"dense0"}
    writer._sealed_writes["xfer"] = 1

    writer._mark_request_done("xfer")

    assert not writer.schedule_write(_write_task("indexer"))
    assert writer._write_task_q.empty()
    assert "xfer" not in writer._scheduled_writes
    assert "xfer" not in writer._scheduled_layers
    assert "xfer" not in writer._sealed_writes


def test_late_remote_blocks_message_is_ignored_after_transfer_done():
    set_role(ROLE.PRODUCER)
    wrapper = _wrapper_for_messages()
    with wrapper.lock:
        wrapper._mark_transfer_terminal_locked("xfer")

    wrapper._handle_message(
        msgpack.dumps(
            {
                "type": "remote_blocks",
                "req_id": "req",
                "transfer_id": "xfer",
                "block_notify_list": [4, 5],
                "decode_rank": 3,
            }
        )
    )

    assert "xfer" not in wrapper.done_remote_allocate_req_dict


def _wrapper_for_messages() -> Any:
    wrapper = MoRIIOWrapper.__new__(MoRIIOWrapper)
    wrapper.lock = threading.Lock()
    wrapper.done_remote_allocate_req_dict = {}
    wrapper.done_req_ids = []
    wrapper.done_write_cache_req_ids = []
    wrapper._write_done_counts = {}
    wrapper._terminal_transfer_ids = OrderedDict()
    return wrapper


@pytest.mark.parametrize(
    ("role", "payload", "expected"),
    [
        pytest.param(
            ROLE.PRODUCER,
            msgpack.dumps(
                {
                    "type": "remote_blocks",
                    "req_id": "req",
                    "transfer_id": "xfer",
                    "block_notify_list": [4, 5],
                    "decode_rank": 3,
                }
            ),
            "remote_blocks",
            id="remote-blocks",
        ),
        pytest.param(
            ROLE.CONSUMER,
            msgpack.dumps({"type": "write_done", "transfer_id": "xfer"}),
            "write_done",
            id="write-done",
        ),
        pytest.param(
            ROLE.PRODUCER,
            msgpack.dumps({"type": "release", "transfer_id": "xfer"}),
            MoRIIOTransferAck("xfer"),
            id="release",
        ),
        pytest.param(
            ROLE.PRODUCER,
            msgpack.dumps(
                {
                    "type": "release",
                    "transfer_id": "xfer",
                    "consumer_tp_size": 8,
                }
            ),
            MoRIIOTransferAck("xfer", 8),
            id="release-consumer-tp-size",
        ),
        pytest.param(None, b"xfer", "plain", id="plain-string"),
    ],
)
def test_moriio_wrapper_routes_valid_messages(role, payload, expected):
    wrapper = _wrapper_for_messages()
    completions: list[str] = []
    if role is not None:
        set_role(role)
    if expected == "plain":
        wrapper._handle_completion_message = completions.append

    wrapper._handle_message(payload)

    if expected == "remote_blocks":
        request_info = wrapper.done_remote_allocate_req_dict["xfer"]
        assert request_info.block_ids == [[4, 5]]
        assert request_info.decode_dp_rank == 3
    elif expected == "write_done":
        assert wrapper.done_write_cache_req_ids == ["xfer"]
    elif expected == "plain":
        assert completions == ["xfer"]
    else:
        assert wrapper.done_req_ids == [expected]
        assert wrapper._is_transfer_terminal_locked("xfer")


@pytest.mark.parametrize(
    ("role", "payload", "match"),
    [
        pytest.param(
            None,
            msgpack.dumps({"type": "unknown", "transfer_id": "xfer"}),
            "Unhandled structured message type",
            id="unknown-structured-type",
        ),
        pytest.param(
            ROLE.PRODUCER,
            msgpack.dumps(
                {
                    "type": "remote_blocks",
                    "req_id": "req",
                    "transfer_id": "xfer",
                    "block_notify_list": [],
                }
            ),
            "block_notify_list cannot be empty",
            id="empty-remote-blocks",
        ),
        pytest.param(
            None,
            b"",
            "Unhandled message format",
            id="empty-completion",
        ),
    ],
)
def test_moriio_wrapper_rejects_invalid_messages(role, payload, match):
    wrapper = _wrapper_for_messages()
    if role is not None:
        set_role(role)
    wrapper._handle_completion_message = lambda msg: None

    with pytest.raises(MoRIIOError, match=match):
        wrapper._handle_message(payload)


# ---------------------------------------------------------------------------
# Mamba/KDA (conv + ssm) transfer geometry
# ---------------------------------------------------------------------------


def _slot_strided(num_slots, per_slot_shape, slot_stride, dtype=torch.bfloat16):
    """Build a slot-strided view: shape (num_slots, *per_slot_shape) whose slot
    stride (dim 0) exceeds the per-slot element count, mimicking the K3 KDA
    conv/ssm cache tensors."""
    inner = 1
    for d in per_slot_shape:
        inner *= d
    assert slot_stride >= inner
    backing = torch.zeros(num_slots * slot_stride, dtype=dtype)
    inner_strides: list[int] = []
    acc = 1
    for d in reversed(per_slot_shape):
        inner_strides.insert(0, acc)
        acc *= d
    return backing.as_strided(
        (num_slots, *per_slot_shape), (slot_stride, *inner_strides)
    )


def _gdn_split_info(conv_rows=3, key_dim=4, value_dim=8, dtype_size=2):
    """A GDN-style conv split: sub-projections (key, key, value)."""
    conv_dim = 2 * key_dim + value_dim
    conv_state_bytes = conv_dim * conv_rows * dtype_size
    ssm_state_bytes = 64  # opaque here; ssm size is taken from the tensor
    return MambaConvSplitInfo(
        conv_rows=conv_rows,
        local_proj_dims=(key_dim, key_dim, value_dim),
        conv_dtype_size=dtype_size,
        ssm_sizes=(conv_state_bytes, ssm_state_bytes),
    )


def test_mamba_transfer_geometry_is_slot_strided():
    split = _gdn_split_info()
    conv_dim = sum(split.local_proj_dims)
    conv = _slot_strided(5, (conv_dim, split.conv_rows), slot_stride=200)
    ssm = _slot_strided(5, (2, 4, 4), slot_stride=64)

    geom = moriio_layout.get_mamba_transfer_geometry(conv, ssm)

    assert geom.conv_slot_stride == 200
    assert geom.ssm_slot_stride == 64
    assert geom.ssm_slot_bytes == (2 * 4 * 4) * ssm.element_size()
    # Region spans the full slot-strided extent (shape[0] * stride(0)), not just
    # numel(), so the highest-slot transfers stay in-bounds.
    assert geom.conv_region_len == conv.shape[0] * conv.stride(0) * conv.element_size()
    assert geom.ssm_region_len == ssm.shape[0] * ssm.stride(0) * ssm.element_size()


def test_kda_conv_ssm_unpacks_padded_pages_as_aliasing_strided_views():
    num_blocks = 4
    page_bytes = 128
    conv_shape = (3, 4)
    ssm_shape = (2, 2, 4)
    conv_bytes = math.prod(conv_shape) * torch.bfloat16.itemsize
    ssm_bytes = math.prod(ssm_shape) * torch.bfloat16.itemsize
    assert conv_bytes + ssm_bytes < page_bytes

    pages = torch.zeros((num_blocks, 1, 1, page_bytes), dtype=torch.uint8)
    spec = MambaSpec(
        block_size=1,
        shapes=(conv_shape, ssm_shape),
        dtypes=(torch.bfloat16, torch.bfloat16),
        page_size_padded=page_bytes,
    )

    conv, ssm = moriio_layout.kda_conv_ssm(pages, spec)

    assert conv.shape == (num_blocks, *conv_shape)
    assert ssm.shape == (num_blocks, *ssm_shape)
    assert conv.stride() == (page_bytes // 2, 4, 1)
    assert ssm.stride() == (page_bytes // 2, 8, 4, 1)
    assert conv.untyped_storage().data_ptr() == pages.untyped_storage().data_ptr()
    assert ssm.untyped_storage().data_ptr() == pages.untyped_storage().data_ptr()
    assert conv.storage_offset() == 0
    assert ssm.storage_offset() * ssm.element_size() == conv_bytes

    conv[1].fill_(1)
    ssm[2].fill_(2)
    assert torch.all(pages[1, 0, 0, :conv_bytes].view(torch.bfloat16) == 1)
    assert torch.all(
        pages[2, 0, 0, conv_bytes : conv_bytes + ssm_bytes].view(torch.bfloat16) == 2
    )


def test_kda_conv_ssm_accepts_strided_blocks_without_copying():
    backing = torch.zeros(4 * 256, dtype=torch.uint8)
    pages = backing.as_strided((4, 1, 1, 128), (256, 128, 128, 1))
    spec = MambaSpec(
        block_size=1,
        shapes=((3, 4), (2, 2, 4)),
        dtypes=(torch.bfloat16, torch.bfloat16),
        page_size_padded=128,
    )

    conv, ssm = moriio_layout.kda_conv_ssm(pages, spec)

    assert not pages.is_contiguous()
    assert conv.stride(0) == 128
    assert ssm.stride(0) == 128
    assert conv.untyped_storage().data_ptr() == backing.data_ptr()
    assert ssm.untyped_storage().data_ptr() == backing.data_ptr()


def test_kda_conv_ssm_rejects_noncontiguous_page_bytes():
    pages = torch.zeros((4, 1, 1, 256), dtype=torch.uint8)[..., ::2]
    spec = MambaSpec(
        block_size=1,
        shapes=((3, 4), (2, 2, 4)),
        dtypes=(torch.bfloat16, torch.bfloat16),
        page_size_padded=128,
    )

    with pytest.raises(ValueError, match="contiguous byte dimension"):
        moriio_layout.kda_conv_ssm(pages, spec)


def test_kda_conv_ssm_rejects_non_byte_packed_cache():
    pages = torch.zeros((4, 1, 1, 128), dtype=torch.bfloat16)
    spec = MambaSpec(
        block_size=1,
        shapes=((3, 4), (2, 2, 4)),
        dtypes=(torch.bfloat16, torch.bfloat16),
        page_size_padded=128,
    )

    with pytest.raises(ValueError, match="byte-sized dtype"):
        moriio_layout.kda_conv_ssm(pages, spec)


def test_mamba_transfer_geometry_requires_matching_slot_counts():
    conv = torch.empty((5, 4, 3), dtype=torch.bfloat16)
    ssm = torch.empty((4, 2, 4, 4), dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="conv/ssm slot count mismatch"):
        moriio_layout.get_mamba_transfer_geometry(conv, ssm)


def test_mamba_transfer_geometry_rejects_empty_caches():
    conv = torch.empty((0, 4, 3), dtype=torch.bfloat16)
    ssm = torch.empty((0, 2, 4, 4), dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="at least one slot"):
        moriio_layout.get_mamba_transfer_geometry(conv, ssm)


def test_mamba_conv_ssm_offsets_homogeneous_tp():
    split = _gdn_split_info()
    conv_dim = sum(split.local_proj_dims)
    conv = _slot_strided(6, (conv_dim, split.conv_rows), slot_stride=200)
    ssm = _slot_strided(6, (2, 4, 4), slot_stride=64)
    geom = moriio_layout.get_mamba_transfer_geometry(conv, ssm)

    local_slots = [1, 2]
    remote_slots = [1, 2]
    template = moriio_layout.build_mamba_offset_template(conv, ssm, split, tp_ratio=1)
    local_offs, remote_offs, sizes = moriio_layout.apply_mamba_offset_template(
        template, local_slots, remote_slots
    )

    n_conv = moriio_layout.compute_mamba_conv_split_count(local_slots, split)
    assert n_conv == len(local_slots) * len(split.local_proj_dims)
    assert len(local_offs) == n_conv + len(local_slots)

    # Homogeneous TP: remote conv slices equal local slices; since local==remote
    # slots, conv local and remote offsets coincide.
    loc_co = split.local_conv_offsets
    rem_co = split.remote_conv_offsets(0, 1)
    k = 0
    esz = conv.element_size()
    for ls, rs in zip(local_slots, remote_slots):
        lbase = ls * geom.conv_slot_stride * esz
        rbase = rs * geom.conv_slot_stride * esz
        for (loff, _), (roff, rsz) in zip(loc_co, rem_co):
            assert local_offs[k] == lbase + loff
            assert remote_offs[k] == rbase + roff
            assert sizes[k] == rsz
            k += 1

    # SSM: whole per-slot block, no head sharding at tp_ratio == 1.
    essz = ssm.element_size()
    for j, (ls, rs) in enumerate(zip(local_slots, remote_slots)):
        assert local_offs[n_conv + j] == ls * geom.ssm_slot_stride * essz
        assert remote_offs[n_conv + j] == rs * geom.ssm_slot_stride * essz
        assert sizes[n_conv + j] == geom.ssm_slot_bytes


def test_mamba_conv_ssm_offsets_heterogeneous_tp_is_gated():
    # Heterogeneous-TP mamba transfer is a documented follow-up (remote slot
    # stride + multi-rank fan-in gather). Until then the offset helper must fail
    # loudly rather than silently transfer wrong bytes, since the mamba path
    # bypasses the attention KV-head validator.
    split = _gdn_split_info()
    conv_dim = sum(split.local_proj_dims)
    conv = _slot_strided(4, (conv_dim, split.conv_rows), slot_stride=200)
    ssm = _slot_strided(4, (2, 4, 4), slot_stride=64)
    with pytest.raises(NotImplementedError):
        moriio_layout.build_mamba_offset_template(conv, ssm, split, tp_ratio=2)


def test_local_mamba_slots_must_fit_registered_regions():
    split = _gdn_split_info()
    conv_dim = sum(split.local_proj_dims)
    conv = _slot_strided(6, (conv_dim, split.conv_rows), slot_stride=200)
    ssm = _slot_strided(6, (2, 4, 4), slot_stride=64)
    template = moriio_layout.build_mamba_offset_template(conv, ssm, split, tp_ratio=1)

    with pytest.raises(ValueError, match="local mamba slot 6"):
        moriio_layout.apply_mamba_offset_template(template, [6], [0])


@pytest.mark.parametrize(
    ("local_slots", "remote_slots"),
    [([1, 2], [3]), ([1], [2, 3])],
)
def test_mamba_slot_counts_must_match(local_slots, remote_slots):
    split = _gdn_split_info()
    conv_dim = sum(split.local_proj_dims)
    conv = _slot_strided(6, (conv_dim, split.conv_rows), slot_stride=200)
    ssm = _slot_strided(6, (2, 4, 4), slot_stride=64)
    template = moriio_layout.build_mamba_offset_template(conv, ssm, split, tp_ratio=1)

    with pytest.raises(ValueError, match="slot counts must match"):
        moriio_layout.apply_mamba_offset_template(template, local_slots, remote_slots)


def test_remote_mamba_slots_must_be_non_negative():
    split = _gdn_split_info()
    conv_dim = sum(split.local_proj_dims)
    conv = _slot_strided(6, (conv_dim, split.conv_rows), slot_stride=200)
    ssm = _slot_strided(6, (2, 4, 4), slot_stride=64)
    template = moriio_layout.build_mamba_offset_template(conv, ssm, split, tp_ratio=1)

    with pytest.raises(ValueError, match="must be non-negative"):
        moriio_layout.apply_mamba_offset_template(template, [1], [-1])
