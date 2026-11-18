# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NIXL regions and descriptors with a drafter KV pool (`VLLM_DRAFT_KV_POOL`):
each region carries its pool's block count, descriptors address the pool's
own rows, a peer without the pool pairs region by layer, and the chunked
registration's cuts (`VLLM_NIXL_CHUNKED_KV_REGISTRATION`) keep every
descriptor run whole at the decode and prefill pools' real sizes."""

from unittest.mock import MagicMock, patch

import msgspec
import numpy as np
import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.nixl import base_worker as bw
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    NixlAgentMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_worker import (
    NixlPullConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.registration import (
    KV_REGISTRATION_MAX_WHOLE_BYTES,
    RegionRun,
    check_registration_pieces,
    plan_registration_pieces,
    split_strided_descs,
)
from vllm.v1.core import kv_cache_planning as planning
from vllm.v1.core.single_type_kv_cache_manager import DFlashSlidingWindowManager
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.v1.kv_cache_layout import KVCacheLayout
from vllm.v1.worker.utils import allocate_kv_cache

from .test_nixl_desc_geometry import _RecordingNixl
from .utils import create_vllm_config, expand_strided_descs, make_kv_cache_group

pytestmark = pytest.mark.cpu_test

BLOCK = 16
TARGET = AttentionSpec(block_size=BLOCK, num_kv_heads=1, head_size=8, dtype=torch.uint8)
INDEXER = AttentionSpec(
    block_size=BLOCK, num_kv_heads=1, head_size=8, head_size_v=0, dtype=torch.uint8
)
# A drafter page (192 B) unlike either target page, as on MiniMax-M3.1.
DRAFT = AttentionSpec(block_size=BLOCK, num_kv_heads=1, head_size=6, dtype=torch.uint8)
TARGET_LAYERS = ["model.layers.0.attn", "model.layers.0.indexer", "model.layers.1.attn"]
DRAFT_LAYERS = ["model.layers.9.draft.attn", "model.layers.10.draft.attn"]


def _groups(pool: bool):
    target = make_kv_cache_group(TARGET_LAYERS[:1], TARGET)
    target.kv_cache_specs = {
        TARGET_LAYERS[0]: TARGET,
        TARGET_LAYERS[1]: INDEXER,
        TARGET_LAYERS[2]: TARGET,
    }
    drafter = make_kv_cache_group(
        DRAFT_LAYERS,
        DRAFT,
        DFlashSlidingWindowManager,
        DFlashSlidingWindowManager.config(sliding_window=32),
        pool_id=1 if pool else 0,
    )
    return [target, drafter]


def _worker(num_blocks: int, pool_blocks: int | None, monkeypatch):
    config = create_vllm_config(block_size=BLOCK, kv_role="kv_both")
    config.cache_config.kv_cache_layout = "BLHNC"
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
        assert kv_cache_config.pool_num_blocks == {1: pool_blocks}
    caches = allocate_kv_cache(
        kv_cache_config, torch.device("cpu"), KVCacheLayout.BLHNC
    )
    backend = _backend()
    with (
        patch.object(bw, "NixlWrapper", _RecordingNixl),
        patch.object(bw, "get_tensor_model_parallel_rank", return_value=0),
        patch.object(bw, "get_tensor_model_parallel_world_size", return_value=1),
        patch.object(bw, "get_current_attn_backends", return_value=[backend]),
        patch("threading.Thread"),
    ):
        worker = NixlPullConnectorWorker(config, "engine", kv_cache_config)
        worker.register_kv_caches(caches)
    return worker, kv_cache_config, caches


def _backend():
    backend = MagicMock()
    backend.get_supported_kernel_block_sizes.return_value = [BLOCK]
    backend.get_name.return_value = "FLASH_ATTN"
    backend.full_cls_name.return_value = "fake.FLASH_ATTN"
    return backend


def _region_of(worker) -> dict[str, int]:
    """Each layer's region."""
    return {
        name: region
        for region, names in enumerate(worker.region_members)
        for name in names
    }


def _resolve(worker, desc_ids) -> list[int]:
    """Start address of each local descriptor id."""
    rows = expand_strided_descs(worker.src_blocks_data)
    return [int(rows[i, 0]) for i in desc_ids]


@pytest.mark.parametrize("pool", [False, True], ids=["one-pool", "drafter-pool"])
def test_regions_carry_their_pool_and_address_its_rows(pool, monkeypatch):
    num_blocks, pool_blocks = 6, (12 if pool else None)
    worker, kv_cache_config, caches = _worker(num_blocks, pool_blocks, monkeypatch)
    region_of = _region_of(worker)
    target_block = 2 * 256 + 128
    draft_block = 2 * 192
    base = caches[TARGET_LAYERS[0]].untyped_storage().data_ptr()
    # Without the pool the drafter's pages alias the first bytes of the
    # target-sized blocks; with it they are rows of their own pool.
    pool_base = base + (num_blocks * target_block if pool else 0)
    for name in TARGET_LAYERS:
        r = region_of[name]
        assert worker.region_num_blocks[r] == num_blocks
        assert worker.block_stride_per_layer[r] == target_block
    for i, name in enumerate(DRAFT_LAYERS):
        r = region_of[name]
        assert worker.region_num_blocks[r] == (pool_blocks if pool else num_blocks)
        assert worker.block_stride_per_layer[r] == (
            draft_block if pool else target_block
        )
        assert worker.block_len_per_layer[r] == 192
        assert worker.kv_caches_base_addr["engine"][0][r] == pool_base + i * 192
    # Every descriptor row lies in the backing storage.
    rows = expand_strided_descs(worker.src_blocks_data)
    assert len(rows) == sum(worker.region_num_blocks)
    storage_end = base + kv_cache_config.kv_cache_tensors[0].size
    assert all(base <= a and a + n <= storage_end for a, n, _ in rows.tolist())

    # A read of target blocks [1, 4] and drafter blocks [2, 3] addresses block
    # b of each layer at its pool's row b.
    regions = [region_of[name] for name in TARGET_LAYERS + DRAFT_LAYERS]
    groups = [0] * len(TARGET_LAYERS) + [1] * len(DRAFT_LAYERS)
    desc_ids = worker._compute_desc_ids(
        block_ids=([1, 4], [2, 3]),
        dst_num_blocks=worker.num_blocks,
        block_size_ratio=None,
        physical_blocks_per_logical=1,
        pairs=(regions, groups),
        region_num_blocks=worker.region_num_blocks,
    )
    expected = []
    for name in TARGET_LAYERS + DRAFT_LAYERS:
        view = caches[name]
        blocks = [1, 4] if name in TARGET_LAYERS else [2, 3]
        expected += [view[b].data_ptr() for b in blocks]
    assert _resolve(worker, desc_ids) == expected
    if pool:
        drafter_rows = _resolve(worker, desc_ids[-4:])
        assert all(a >= pool_base for a in drafter_rows)


def test_handshake_advertises_each_regions_pool(monkeypatch):
    p, _, _ = _worker(6, 12, monkeypatch)
    meta = msgspec.msgpack.decode(
        p.xfer_handshake_metadata.agent_metadata_bytes, type=NixlAgentMetadata
    )
    assert meta.num_blocks == 6
    assert meta.region_num_blocks == p.region_num_blocks
    assert sorted(meta.region_num_blocks) == [6, 6, 6, 12, 12]


@pytest.mark.parametrize(
    "decode_pool", [None, 13], ids=["decode-one-pool", "decode-pool"]
)
def test_peer_without_or_with_the_pool_pairs_by_layer(decode_pool, monkeypatch):
    """Prefill with the pool, decode with or without it: regions pair by layer
    with the same page length, so each pair moves one layer's page."""
    p, _, _ = _worker(6, 12, monkeypatch)
    d, _, _ = _worker(9, decode_pool, monkeypatch)
    meta = msgspec.msgpack.decode(
        p.xfer_handshake_metadata.agent_metadata_bytes, type=NixlAgentMetadata
    )
    pairs = d._pair_regions(meta)
    assert len(pairs.local) == len(TARGET_LAYERS) + len(DRAFT_LAYERS)
    paired_layers = set()
    for remote, local, group in zip(pairs.remote, pairs.local, pairs.group):
        remote_region = pairs.remote_regions[remote]
        layers = set(meta.region_members[remote_region]) & set(d.region_members[local])
        assert len(layers) == 1
        (layer,) = layers
        paired_layers.add(layer)
        assert meta.block_lens[remote_region] == d.block_len_per_layer[local]
        is_draft = layer in DRAFT_LAYERS
        assert group == int(is_draft)
        assert meta.region_num_blocks[remote_region] == (12 if is_draft else 6)
        assert d.region_num_blocks[local] == ((decode_pool or 9) if is_draft else 9)
    assert paired_layers == set(TARGET_LAYERS + DRAFT_LAYERS)


def test_host_transfer_buffer_is_refused(monkeypatch):
    config = create_vllm_config(block_size=BLOCK, kv_role="kv_both")
    config.cache_config.kv_cache_layout = "BLHNC"
    config.kv_transfer_config.kv_buffer_device = "cpu"
    monkeypatch.setenv(planning.DRAFT_KV_POOL_NUM_BLOCKS_ENV, "12")
    kv_cache_config = (
        planning.DefaultKVCacheConfigBuilder().get_kv_cache_config_from_groups(
            config, _groups(pool=True), 6
        )
    )
    with (
        patch.object(bw, "NixlWrapper", _RecordingNixl),
        patch.object(bw, "get_tensor_model_parallel_rank", return_value=0),
        patch.object(bw, "get_tensor_model_parallel_world_size", return_value=1),
        patch.object(bw, "get_current_attn_backends", return_value=[_backend()]),
        patch("threading.Thread"),
        # A CPU worker never stages through the host, so pretend to be a GPU.
        patch.object(bw.current_platform, "device_type", "cuda"),
        patch.object(bw.current_platform, "get_nixl_memory_type", return_value=None),
        pytest.raises(NotImplementedError, match="host transfer buffer"),
    ):
        NixlPullConnectorWorker(config, "engine", kv_cache_config)


# ---- Chunked registration (dma-buf MRs under 4 GiB) at the real sizes ----

TARGET_ROW = 60 * 73_728 + 60 * 9_216  # 4,976,640 B
DRAFT_ROW = 5 * 131_072  # 655,360 B


def _runs(base, shared_blocks, pool_blocks, pool: bool):
    runs = []
    offset = 0
    for page, count in ((73_728, 60), (9_216, 60)):
        for _ in range(count):
            runs.append(RegionRun(base + offset, page, TARGET_ROW, shared_blocks))
            offset += page
    if pool:
        pool_base = base + shared_blocks * TARGET_ROW
        runs += [
            RegionRun(pool_base + i * 131_072, 131_072, DRAFT_ROW, pool_blocks)
            for i in range(5)
        ]
        end = pool_base + pool_blocks * DRAFT_ROW
    else:
        runs += [
            RegionRun(base + i * 131_072, 131_072, TARGET_ROW, shared_blocks)
            for i in range(5)
        ]
        end = base + shared_blocks * TARGET_ROW
    return runs, end


@pytest.mark.parametrize(
    "shared_blocks,pool_blocks",
    [(25_982, 2_993), (8_194, 2_313), (8_189, 27_745), (26_376, None)],
    ids=["decode", "prefill", "prefill-default-size", "decode-one-pool"],
)
def test_registration_pieces_keep_descriptors_whole(shared_blocks, pool_blocks):
    base = 0x7F00_0000_0000 + 0x1234_5000  # an unaligned dma-buf base
    runs, end = _runs(base, shared_blocks, pool_blocks, pool_blocks is not None)
    pieces = plan_registration_pieces(base, end, runs)
    check_registration_pieces(base, end, pieces, runs)
    assert all(hi - lo <= KV_REGISTRATION_MAX_WHOLE_BYTES for lo, hi in pieces)
    cuts = [lo for lo, _ in pieces[1:]]
    descs = np.array(
        [[r.base, r.block_len, 0, r.stride, r.num_blocks] for r in runs],
        dtype=np.uint64,
    )
    split = split_strided_descs(descs, cuts)
    # The same blocks in the same flat order, each run inside one piece.
    assert int(split[:, 4].sum()) == int(descs[:, 4].sum())
    starts = np.searchsorted(np.array([lo for lo, _ in pieces]), split[:, 0], "right")
    last = split[:, 0] + (split[:, 4] - 1) * split[:, 3] + split[:, 1]
    ends = np.searchsorted(np.array([lo for lo, _ in pieces]), last - 1, "right")
    assert (starts == ends).all()
    if pool_blocks is not None:
        # The drafter's pool is cut on its own rows, never inside a row.
        pool_base = base + shared_blocks * TARGET_ROW
        for cut in cuts:
            if cut > pool_base:
                assert (cut - pool_base) % DRAFT_ROW == 0
            else:
                assert (cut - base) % TARGET_ROW == 0
