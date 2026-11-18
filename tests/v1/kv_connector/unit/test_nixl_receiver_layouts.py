# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The background pull receiver on a MiniMax-M3.1-shaped BLHNC KV plan (target
attention + indexer + DSpark drafter, with and without the drafter's own block
pool): its peer checks accept the plan, it prepares exactly the READs the
engine-thread path prepares, and replaying them copies every page of the
request to the right decode block and nothing else."""

from concurrent.futures import Future
from unittest.mock import MagicMock, patch

import msgspec
import numpy as np
import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.nixl import base_worker as bw
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    NixlAgentMetadata,
    NixlConnectorMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_receiver import ReadJob
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_worker import (
    NixlPullConnectorWorker,
    _PullReceiverBackend,
)
from vllm.v1.core import kv_cache_planning as planning
from vllm.v1.kv_cache_layout import KVCacheLayout
from vllm.v1.worker.utils import allocate_kv_cache

from .test_draft_kv_pool_nixl import BLOCK, _backend, _groups
from .test_nixl_desc_geometry import _RecordingNixl, _tag
from .utils import create_vllm_config

pytestmark = pytest.mark.cpu_test


class _NotifRecordingNixl(_RecordingNixl):
    """Also records each READ's notification and the posts."""

    def make_prepped_xfer(
        self, op, local_handle, local_ids, remote_handle, remote_ids, notif_msg=None
    ):
        handle = super().make_prepped_xfer(
            op, local_handle, local_ids, remote_handle, remote_ids, notif_msg
        )
        self.notifs = getattr(self, "notifs", []) + [notif_msg]
        return handle

    def transfer(self, handle):
        self.posted = getattr(self, "posted", []) + [handle]
        return "PROC"


def _side(engine_id, num_blocks, pool_blocks, monkeypatch):
    config = create_vllm_config(block_size=BLOCK, kv_role="kv_both")
    config.cache_config.kv_cache_layout = "BLHNC"
    config.kv_transfer_config.kv_buffer_device = "cuda"
    if pool_blocks is None:
        kv_cache_config = planning.get_kv_cache_config_from_groups(
            config,
            _groups(pool=False),
            num_blocks * planning._pool_bytes_per_block(config, _groups(False)),
        )
    else:
        monkeypatch.setenv(planning.DRAFT_KV_POOL_NUM_BLOCKS_ENV, str(pool_blocks))
        kv_cache_config = (
            planning.DefaultKVCacheConfigBuilder().get_kv_cache_config_from_groups(
                config, _groups(pool=True), num_blocks
            )
        )
    caches = allocate_kv_cache(
        kv_cache_config, torch.device("cpu"), KVCacheLayout.BLHNC
    )
    platform = MagicMock()
    platform.device_type = "cuda"
    platform.get_nixl_memory_type.return_value = "VRAM"
    with (
        patch.object(bw, "current_platform", platform),
        patch.object(bw, "NixlWrapper", _NotifRecordingNixl),
        patch.object(bw, "get_tensor_model_parallel_rank", return_value=0),
        patch.object(bw, "get_tensor_model_parallel_world_size", return_value=1),
        patch.object(bw, "get_current_attn_backends", return_value=[_backend()]),
        patch("threading.Thread"),
    ):
        worker = NixlPullConnectorWorker(config, engine_id, kv_cache_config)
        worker.register_kv_caches(caches)
    return worker, kv_cache_config, caches


def _pages(caches, layer, block):
    return caches[layer][block]


def _layers(group, caches):
    """Every layer of a transfer group (a group's spec map may list more)."""
    names = list(group.layer_names) + list(getattr(group, "kv_cache_specs", {}) or [])
    return [name for name in dict.fromkeys(names) if name in caches]


def _request_ids(kv_cache_config, first, step):
    """Two blocks per transfer group, numbered from `first` by `step`."""
    ids, nxt = [], first
    for _ in kv_cache_config.transfer_groups:
        ids.append([nxt, nxt + step])
        nxt += 2 * step
    return ids


def _resolved(nixl, xfers):
    """(local byte range, remote byte range) of every descriptor pair."""
    out = []
    for op, local_handle, local_ids, remote_handle, remote_ids in xfers:
        assert op == "READ"
        for li, ri in zip(local_ids.tolist(), remote_ids.tolist()):
            la, ll, _ = (int(x) for x in nixl.dlists[local_handle][li])
            ra, rl, _ = (int(x) for x in nixl.dlists[remote_handle][ri])
            out.append((la, ll, ra, rl))
    return out


def _handshake(backend, meta, engine_id):
    """Run the receiver's peer path with a handshake that already returned."""
    # CPU test caches register as DRAM; the engine's device KV is VRAM.
    meta.region_mem_types = ["VRAM"] * len(meta.region_mem_types or [])
    future: Future = Future()
    future.set_result(({(0, 0): meta}, 0.0))
    backend.worker._handshake_initiation_executor = MagicMock()
    backend.worker._handshake_initiation_executor.submit.return_value = future
    assert backend._peer(engine_id, "prefill-host", 5600, 1) is False
    assert backend._peer(engine_id, "prefill-host", 5600, 1) is True


@pytest.mark.parametrize(
    "p_pool,d_pool",
    [(None, None), (12, 13), (12, 30)],
    ids=["one-pool", "drafter-pool", "drafter-pool-unequal"],
)
def test_receiver_reads_every_page_like_the_engine_thread(p_pool, d_pool, monkeypatch):
    p, p_kv, p_caches = _side("P", 6, p_pool, monkeypatch)
    d, d_kv, d_caches = _side("D", 9, d_pool, monkeypatch)
    groups = d_kv.transfer_groups
    p_ids = _request_ids(p_kv, 1, 1)
    # D's blocks run down from each pool's last ones, unlike P's.
    d_ids = [
        [d_kv.num_blocks_of_pool(g.pool_id) - 1 - b for b in ids]
        for g, ids in zip(groups, _request_ids(d_kv, 0, 1))
    ]
    for g, ids in zip(groups, p_ids):
        for layer in _layers(g, p_caches):
            for pos, block in enumerate(ids):
                page = _pages(p_caches, layer, block)
                page.copy_(_tag(page, layer, pos))
    meta = msgspec.msgpack.Decoder(NixlAgentMetadata).decode(
        p.xfer_handshake_metadata.agent_metadata_bytes
    )

    # The receiver's own handshake path: geometry and transfer-plan checks.
    backend = _PullReceiverBackend(d)
    _handshake(backend, meta, "P")
    assert set(d._remote_agents["P"]) == {(0, 0)}

    def new_meta():
        metadata = NixlConnectorMetadata()
        metadata.add_new_req_to_recv(
            request_id="req",
            local_block_ids=tuple(list(ids) for ids in d_ids),
            kv_transfer_params={
                "remote_block_ids": [list(ids) for ids in p_ids],
                "remote_engine_id": "P",
                "remote_request_id": "prefill-req",
                "remote_host": "prefill-host",
                "remote_port": 5600,
                "tp_size": 1,
            },
        )
        return metadata.reqs_to_recv["req"]

    # The engine-thread path (base _read_blocks_for_req).
    legacy = new_meta()
    legacy.local_physical_block_ids = d._logical_to_kernel_block_ids(
        legacy.local_block_ids, d._physical_blocks_per_logical_kv_block
    )
    d._recving_metadata["req"] = legacy
    d._read_blocks_for_req("req", legacy)
    nixl = d.nixl_wrapper
    expected, expected_notifs = list(nixl.xfers), list(nixl.notifs)
    nixl.xfers.clear()
    nixl.notifs = []

    # The receiver's adapter, as the scheduler hands it a request.
    job_meta = new_meta()
    job_meta.receiver_generation, job_meta.receiver_is_async = 1, True
    job = ReadJob(("req", 1), job_meta)
    assert backend.ready(job)
    transfers = list(backend.transfers(job))
    for transfer in transfers:
        assert backend.post(transfer) == "PROC"
    got = list(nixl.xfers)
    assert len(transfers) == len(got) == len(expected) == 1
    for (op, lh, li, rh, ri), (eop, elh, eli, erh, eri) in zip(got, expected):
        assert (op, lh, rh) == (eop, elh, erh)
        assert np.array_equal(li, eli) and np.array_equal(ri, eri)
    assert nixl.notifs == expected_notifs == [b"prefill-req:1"]
    assert transfers[0].descriptors == len(expected[0][2])

    # Replay the receiver's READ as copies: every page lands, nothing else.
    import ctypes

    for la, ll, ra, rl in _resolved(nixl, got):
        assert ll == rl
        ctypes.memmove(la, ra, ll)
    wrong, checked = [], 0
    for g, ids in zip(groups, d_ids):
        for layer in _layers(g, d_caches):
            for pos, block in enumerate(ids):
                page = _pages(d_caches, layer, block)
                checked += 1
                if not torch.equal(page, _tag(page, layer, pos)):
                    wrong.append(f"{layer}[{pos}]")
                page.zero_()
    assert checked and not wrong, wrong[:8]
    storages = {t.untyped_storage().data_ptr(): t for t in d_caches.values()}
    for t in storages.values():
        raw = torch.empty(0, dtype=torch.uint8).set_(t.untyped_storage())
        assert not raw.any(), "D bytes outside the request's pages changed"


def test_receiver_refuses_a_peer_whose_pages_differ(monkeypatch):
    """A peer with another page geometry fails the handshake, before any READ."""
    p, _, _ = _side("P", 6, 12, monkeypatch)
    d, _, _ = _side("D", 9, 13, monkeypatch)
    meta = msgspec.msgpack.Decoder(NixlAgentMetadata).decode(
        p.xfer_handshake_metadata.agent_metadata_bytes
    )
    meta.block_lens = [n + 4 for n in meta.block_lens]
    backend = _PullReceiverBackend(d)
    with pytest.raises(ValueError, match="geometry"):
        _handshake(backend, meta, "P")
    assert "P" not in d._remote_agents
    assert not d.nixl_wrapper.xfers


def test_owned_request_copy_shares_no_block_list():
    """The receiver's copy of a request equals it and owns every block list."""
    from copy import deepcopy

    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
        copy_block_ids,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_worker import (
        _owned_req_meta,
    )

    metadata = NixlConnectorMetadata()
    metadata.add_new_req_to_recv(
        request_id="req",
        local_block_ids=([3, 4, 5], [7]),
        kv_transfer_params={
            "remote_block_ids": [[10, 11, 12], [13]],
            "remote_engine_id": "P",
            "remote_request_id": "prefill-req",
            "remote_host": "h",
            "remote_port": 1,
            "tp_size": 1,
        },
    )
    meta = metadata.reqs_to_recv["req"]
    meta.receiver_generation, meta.receiver_is_async = 4, True
    owned = _owned_req_meta(meta)
    assert owned == deepcopy(meta)
    assert owned.local_physical_block_ids is owned.local_block_ids
    for mine, theirs in [
        *zip(owned.local_block_ids, meta.local_block_ids),
        *zip(owned.remote.block_ids, meta.remote.block_ids),
    ]:
        assert mine == theirs and mine is not theirs
    meta.local_block_ids[0].append(99)
    meta.remote.block_ids[1].append(98)
    assert owned.local_block_ids[0] == [3, 4, 5] and owned.remote.block_ids[1] == [13]
    assert type(copy_block_ids(([1], [2]))) is tuple
    assert type(copy_block_ids([[1], [2]])) is list
