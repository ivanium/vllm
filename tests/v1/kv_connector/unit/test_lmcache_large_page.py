# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Large-page addressing tests for the LMCache connectors.

These tests exercise the pure helpers in ``lmcache_integration/metadata.py``
(and the builtin multi-process adapter behind a fake lmcache package), so
they run without the optional lmcache package.
"""

import logging
import sys
import types
from typing import Any

import pytest
import torch

from tests.v1.kv_connector.unit.utils import (
    create_request,
    make_kv_cache_config,
    make_two_group_large_page_config,
)
from vllm.distributed.kv_transfer.kv_connector.utils import group_slot_mapping
from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_integration.metadata import (
    LMCacheMPRequestMetadata,
    LMCacheMPRequestState,
    LMCacheMPRequestTracker,
    lmcache_kv_shapes,
    validate_transfer_groups,
)

BLOCK_SIZE = 16


def test_group_slot_mapping_is_block_id_times_block_size():
    slots = group_slot_mapping(BLOCK_SIZE, [0, 1, 5, 13], 3 * BLOCK_SIZE)
    # Blocks 0, 1, 5 -> slots 0..15, 16..31, 80..95, truncated to 48 tokens.
    assert slots.tolist() == list(range(0, 32)) + list(range(80, 96))
    assert slots.dtype == torch.long


def test_slot_mappings_are_per_transfer_group():
    # The assembly the native adapter inlines: one slot mapping per transfer
    # group, in transfer-group order, each truncated to the token count.
    config = make_two_group_large_page_config()
    # The request holds blocks 1 and 3 of the full-attention group and
    # blocks 2 and 5 of the sliding-window group; map 20 tokens.
    block_ids = ([1, 3], [2, 5])
    fa_slots, swa_slots = (
        group_slot_mapping(
            config.kv_cache_groups[g].kv_cache_spec.block_size, block_ids[g], 20
        )
        for g in config.transfer_group_ids
    )

    # fa: block 1 -> slots 16..31, block 3 -> slots 48..63; 20 tokens.
    assert fa_slots.tolist() == list(range(16, 32)) + list(range(48, 52))
    # swa: block 2 -> slots 32..47, block 5 -> slots 80..95; 20 tokens.
    assert swa_slots.tolist() == list(range(32, 48)) + list(range(80, 84))


def test_lmcache_kv_shapes_use_per_group_head_geometry():
    # (num_layers, k/v, chunk_size, num_kv_heads, head_size) per transfer
    # group; heads are TP-sharded and head sizes differ between groups, so a
    # model-level head geometry would size group 1 wrong.
    config = make_two_group_large_page_config()
    assert lmcache_kv_shapes(config, chunk_size=256, use_mla=False, tp_size=2) == [
        (1, 2, 256, 2, 34),
        (1, 2, 256, 2, 16),
    ]
    # MLA latent KV is not TP-sharded and stores a single tensor per layer.
    assert lmcache_kv_shapes(config, chunk_size=256, use_mla=True, tp_size=2) == [
        (1, 1, 256, 1, 34),
        (1, 1, 256, 1, 16),
    ]


def test_validate_transfer_groups_accepts_token_paged_attention():
    # Full attention + sliding window attention are both token-paged.
    config = make_kv_cache_config(block_size=BLOCK_SIZE, swa_enabled=True)
    validate_transfer_groups(config)


def test_validate_transfer_groups_rejects_mamba():
    config = make_kv_cache_config(block_size=BLOCK_SIZE, mamba_enabled=True)
    with pytest.raises(ValueError, match="MambaSpec"):
        validate_transfer_groups(config)


class _FakeMQFuture:
    def __init__(self, result):
        self._result = result

    def __class_getitem__(cls, item):
        return cls

    def result(self):
        return self._result


def _install_fake_lmcache_mp(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stand in the pieces of the lmcache package that the builtin
    multi-process adapter imports, recording every submitted request."""
    lmcache_utils: Any = types.ModuleType("lmcache.utils")
    lmcache_utils.init_logger = logging.getLogger
    lmcache_utils._lmcache_nvtx_annotate = lambda f: f

    custom_types: Any = types.ModuleType("lmcache.v1.multiprocess.custom_types")

    class CudaIPCWrapper:
        def __init__(self, tensor):
            self.tensor = tensor

    class IPCCacheEngineKey:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    custom_types.CudaIPCWrapper = CudaIPCWrapper
    custom_types.IPCCacheEngineKey = IPCCacheEngineKey
    custom_types.KVCache = list

    mq: Any = types.ModuleType("lmcache.v1.multiprocess.mq")

    class FakeMessageQueueClient:
        def __init__(self, server_url, context):
            self.submitted = []

        def submit_request(self, request_type, payloads, response_cls):
            self.submitted.append((request_type, payloads))
            result = 256 if request_type == "GET_CHUNK_SIZE" else True
            return _FakeMQFuture(result)

    mq.MessageQueueClient = FakeMessageQueueClient
    mq.MessagingFuture = _FakeMQFuture

    protocol: Any = types.ModuleType("lmcache.v1.multiprocess.protocol")

    class RequestType:
        GET_CHUNK_SIZE = "GET_CHUNK_SIZE"
        REGISTER_KV_CACHE = "REGISTER_KV_CACHE"
        UNREGISTER_KV_CACHE = "UNREGISTER_KV_CACHE"
        LOOKUP = "LOOKUP"
        STORE = "STORE"
        RETRIEVE = "RETRIEVE"
        END_SESSION = "END_SESSION"

    protocol.RequestType = RequestType
    protocol.get_response_class = lambda request_type: None

    fake_modules = {
        "lmcache": types.ModuleType("lmcache"),
        "lmcache.utils": lmcache_utils,
        "lmcache.v1": types.ModuleType("lmcache.v1"),
        "lmcache.v1.multiprocess": types.ModuleType("lmcache.v1.multiprocess"),
        "lmcache.v1.multiprocess.custom_types": custom_types,
        "lmcache.v1.multiprocess.mq": mq,
        "lmcache.v1.multiprocess.protocol": protocol,
    }
    for name, module in fake_modules.items():
        monkeypatch.setitem(sys.modules, name, module)


def test_register_kv_cache_payload_carries_protocol_version(monkeypatch):
    """The builtin registration payload pins its wire shape: instance id,
    wrapped layer views, per-group geometry dict, and the protocol version a
    co-released LMCache server checks."""
    _install_fake_lmcache_mp(monkeypatch)
    from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_integration import (
        multi_process_adapter,
    )

    adapter = multi_process_adapter.LMCacheMPWorkerAdapter(
        server_url="tcp://localhost:5555",
        context=None,
        model_name="test-model",
        vllm_block_size=BLOCK_SIZE,
        parallel_strategy=multi_process_adapter.ParallelStrategy(
            use_mla=False,
            kv_world_size=1,
            kv_worker_id=0,
            actual_world_size=1,
            actual_worker_id=0,
            tp_size=1,
            pp_size=1,
        ),
    )
    config = make_two_group_large_page_config()
    kv_caches = {"fa.0": torch.zeros(8), "swa.0": torch.zeros(8)}
    adapter.register_kv_caches(kv_caches, config)

    request_type, payloads = adapter.mq_client.submitted[-1]
    assert request_type == multi_process_adapter.RequestType.REGISTER_KV_CACHE
    instance_id, wrapped_caches, geometry, version = payloads
    assert instance_id == adapter.instance_id
    assert [c.tensor for c in wrapped_caches] == [kv_caches["fa.0"], kv_caches["swa.0"]]
    assert geometry == {
        "groups": [
            {
                "layer_names": ["fa.0"],
                "block_size": 16,
                "kernel_rows_per_block": 1,
                "row_bytes": 8704,
            },
            {
                "layer_names": ["swa.0"],
                "block_size": 16,
                "kernel_rows_per_block": 4,
                "row_bytes": 1024,
            },
        ],
    }
    assert version == multi_process_adapter.BUILTIN_MP_PROTOCOL_VERSION == 1


def _make_tracker(num_tokens: int = 64) -> LMCacheMPRequestTracker:
    request = create_request(request_id=0, num_tokens=num_tokens, block_size=BLOCK_SIZE)
    tracker = LMCacheMPRequestTracker(request)
    tracker.append_block_ids(([0, 1, 2, 3], [10, 11, 12, 13]))
    return tracker


def test_mp_tracker_keeps_block_ids_per_group():
    tracker = _make_tracker()
    assert tracker.allocated_block_ids == [[0, 1, 2, 3], [10, 11, 12, 13]]
    assert tracker.num_allocated_blocks == 4

    tracker.append_block_ids(([4], [14]))
    assert tracker.allocated_block_ids == [[0, 1, 2, 3, 4], [10, 11, 12, 13, 14]]
    assert tracker.num_allocated_blocks == 5

    tracker.append_block_ids(None)
    assert tracker.num_allocated_blocks == 5


def test_mp_store_metadata_slices_every_group():
    tracker = _make_tracker()
    tracker.increase_num_scheduled_tokens(64)

    meta = LMCacheMPRequestMetadata.GetStoreMetadata(
        tracker, blocks_in_chunk=2, vllm_block_size=BLOCK_SIZE
    )

    assert meta is not None
    assert meta.direction == "STORE"
    assert meta.op.block_ids == [[0, 1, 2, 3], [10, 11, 12, 13]]
    assert meta.op.start == 0
    assert meta.op.end == 64
    assert len(meta.op) == 4

    # Everything stored: a second call produces no metadata.
    assert (
        LMCacheMPRequestMetadata.GetStoreMetadata(
            tracker, blocks_in_chunk=2, vllm_block_size=BLOCK_SIZE
        )
        is None
    )


def test_mp_retrieve_metadata_slices_every_group():
    tracker = _make_tracker()
    tracker.num_vllm_hit_blocks = 1
    tracker.num_lmcache_hit_blocks = 4
    tracker.state = LMCacheMPRequestState.WAITING_FOR_LOAD

    meta = LMCacheMPRequestMetadata.GetRetrieveMetadata(
        tracker, blocks_in_chunk=2, vllm_block_size=BLOCK_SIZE
    )

    assert meta is not None
    assert meta.direction == "RETRIEVE"
    assert meta.op.block_ids == [[0, 1, 2, 3], [10, 11, 12, 13]]
    # The one APC-hit block overlaps the first chunk and must be skipped.
    assert meta.op.skip_first_n_tokens == BLOCK_SIZE
