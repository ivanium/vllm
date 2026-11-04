# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end NIXL descriptor geometry invariants for hybrid MLA+SSM models
under heterogeneous P/D block geometry (TP-sharded KDA-style state, so
the mamba-aligned logical block size differs between P and D while the
kernel-granularity pages stay equal).

The invariant under test: every LOCAL byte range a request's READ transfers
into must lie within that request's own blocks. A violation means an
incoming transfer can overwrite a co-resident request's KV or mamba state
mid-decode (silent corruption of an unrelated request).
"""

import math
from contextlib import nullcontext
from types import SimpleNamespace
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
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.push_worker import (
    NixlPushConnectorWorker,
)
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheLayout,
    KVCacheTensor,
    MLAAttentionSpec,
    create_kv_cache_views,
)

from .utils import (
    create_vllm_config,
    expand_strided_descs,
    make_kv_cache_config,
)


class _RecordingNixl:
    """Minimal NIXL wrapper stand-in that records descriptor lists and
    prepared transfers so tests can resolve desc ids to byte ranges."""

    def __init__(self, *args, **kwargs):
        self.dlists: dict[int, np.ndarray] = {}
        self.xfers: list[tuple] = []
        self.registered: list[tuple[list[tuple], object]] = []
        self._next_handle = 1

    def get_reg_descs(self, caches_data, mem_type):
        return caches_data

    def register_memory(self, descs, backends=None):
        self.registered.append((descs, backends))

    def deregister_memory(self, descs):
        pass

    def get_agent_metadata(self):
        return b"agent-meta"

    def get_xfer_descs(self, blocks_data, mem_type):
        return blocks_data

    def prep_xfer_dlist(self, agent, descs, mem_type=None):
        handle = self._next_handle
        self._next_handle += 1
        # Store per-block (addr, len, dev) rows so desc ids index directly.
        self.dlists[handle] = expand_strided_descs(np.asarray(descs, dtype=np.uint64))
        return handle

    def add_remote_agent(self, metadata):
        return "remote-agent"

    def make_prepped_xfer(
        self, op, local_handle, local_ids, remote_handle, remote_ids, notif_msg=None
    ):
        handle = self._next_handle
        self._next_handle += 1
        self.xfers.append(
            (
                op,
                local_handle,
                np.asarray(local_ids),
                remote_handle,
                np.asarray(remote_ids),
            )
        )
        return handle

    def transfer(self, handle):
        pass

    def check_xfer_state(self, handle):
        return "DONE"

    def get_xfer_telemetry(self, handle):
        from types import SimpleNamespace

        return SimpleNamespace(
            xferDuration=1.0, postDuration=1.0, totalBytes=1, descCount=1
        )

    def release_xfer_handle(self, handle):
        pass

    def release_dlist_handle(self, handle):
        pass

    def send_notif(self, agent, notif_msg=None):
        pass

    def check_remote_metadata(self, agent):
        return True

    def get_new_notifs(self):
        return {}

    def remove_remote_agent(self, agent):
        pass


def _make_packed_mla_view_worker(
    layouts,
    block_stride,
    *,
    num_blocks=4,
    pp_size=1,
    backend_names=("FLASHMLA",),
    worker_cls=NixlPushConnectorWorker,
):
    """Register real strided views; only NIXL and distributed runtime are fake.

    The config is a large-page pool whose pages are the packed rows, so a
    manager block id is also its kernel row id.
    """
    block_size = 16
    raw = torch.zeros(num_blocks * block_stride, dtype=torch.int8)
    tensors, caches = [], {}
    specs = {}
    for name, (offset, page_size) in layouts.items():
        spec = MLAAttentionSpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=page_size // block_size,
            dtype=torch.uint8,
        )
        specs[name] = spec
        tensor = KVCacheTensor(
            size=raw.nbytes,
            layers=[name],
            layer_stride=page_size,
            block_stride=block_stride,
            offset=offset,
        )
        tensors.append(tensor)
        (caches[name],) = create_kv_cache_views(
            raw, spec, num_blocks, KVCacheLayout.BLHNC, tensor
        )
    # One group when the layers share a page size (whole-row registration
    # dedups them into one region), else one group per layer.
    if len({spec.page_size_bytes for spec in specs.values()}) == 1:
        groups = [KVCacheGroupSpec(list(specs), next(iter(specs.values())))]
    else:
        groups = [KVCacheGroupSpec([name], spec) for name, spec in specs.items()]
    for group in groups:
        group.num_blocks = num_blocks
        group.blocks_per_page = 1
        group.kernel_rows_per_block = 1
        group.row_bytes = block_stride
    kv_cache_config = KVCacheConfig(
        num_pages=num_blocks,
        kv_cache_tensors=tensors,
        kv_cache_groups=groups,
        large_page_bytes=block_stride,
        block_size=block_size,
    )

    config = create_vllm_config(block_size=block_size)
    config.parallel_config.pipeline_parallel_size = pp_size
    config.kv_transfer_config.kv_role = "kv_producer"
    config.cache_config.kv_cache_layout = "BLHNC"
    backends = []
    for name in backend_names:
        backend = MagicMock()
        backend.get_supported_kernel_block_sizes.return_value = [block_size]
        backend.get_name.return_value = name
        backend.full_cls_name.return_value = f"fake.{name}"
        backends.append(backend)
    with (
        patch.object(bw, "NixlWrapper", _RecordingNixl),
        patch.object(bw, "get_tensor_model_parallel_rank", return_value=0),
        patch.object(bw, "get_tensor_model_parallel_world_size", return_value=1),
        patch.object(bw, "get_current_attn_backends", return_value=backends),
        patch("threading.Thread"),
    ):
        worker = worker_cls(
            config,
            "producer" if pp_size > 1 else "consumer",
            kv_cache_config,
        )
        worker.use_mla = True
        worker.register_kv_caches(caches)
    return worker, raw


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "worker_cls", [NixlPullConnectorWorker, NixlPushConnectorWorker]
)
@pytest.mark.parametrize("num_layers", [1, 2])
def test_packed_mla_pp1_transfers_whole_rows(num_layers, worker_cls):
    # BLHNC packs every layer's page of a block in one row: register the row
    # once so a block is one descriptor, not one per layer.
    layouts = {f"L{i}": (i * 128, 128) for i in range(num_layers)}
    stride = num_layers * 128
    worker, raw = _make_packed_mla_view_worker(layouts, stride, worker_cls=worker_cls)
    assert worker.num_regions == 1
    assert worker.num_descs == 4
    assert worker._registered_descs == [[(raw.data_ptr(), raw.nbytes, 0, "")]]
    # One packed region -> one run of 4 whole rows.
    assert worker.src_blocks_data.tolist() == [[raw.data_ptr(), stride, 0, stride, 4]]
    assert worker._transfer_layer_group_ids == ()
    metadata = msgspec.msgpack.decode(
        worker.xfer_handshake_metadata.agent_metadata_bytes, type=NixlAgentMetadata
    )
    assert metadata.packed_member_layouts == layouts


@pytest.mark.cpu_test
def test_per_layer_peer_addresses_pages_inside_packed_rows():
    # E.g. a HiSparse decoder reading a BLHNC prefiller's whole-row region.
    remote, raw = _make_packed_mla_view_worker({"L0": (64, 128)}, 192)
    metadata = msgspec.msgpack.decode(
        remote.xfer_handshake_metadata.agent_metadata_bytes, type=NixlAgentMetadata
    )
    assert metadata.block_lens == [192]
    local = SimpleNamespace(region_names=["L0"])
    bw.NixlBaseConnectorWorker._split_remote_packed_rows(local, metadata)
    assert metadata.kv_caches_base_addr == [raw.data_ptr() + 64]
    assert metadata.block_lens == [128]
    assert metadata.block_strides == [192]
    assert metadata.region_names == ["L0"]
    assert metadata.packed_member_layouts == {}


@pytest.mark.cpu_test
def test_packed_rows_reject_a_peer_packing_rows_differently():
    producer, _ = _make_packed_mla_view_worker({"L0": (0, 128), "L1": (128, 64)}, 192)
    consumer, _ = _make_packed_mla_view_worker({"L0": (64, 128), "L1": (0, 64)}, 192)
    metadata = msgspec.msgpack.decode(
        producer.xfer_handshake_metadata.agent_metadata_bytes, type=NixlAgentMetadata
    )
    metadata.engine_id = "remote"
    with pytest.raises(RuntimeError, match="Packed KV block rows differ"):
        consumer.add_remote_agent(metadata)


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "remote_backends, compatible",
    [
        (("INDEXER", "FLASHMLA"), True),
        (("FLASHMLA", "OTHER"), False),
        (("FLASHMLA",), False),
    ],
)
def test_packed_push_compatibility_hash_uses_all_backends_in_stable_order(
    remote_backends, compatible
):
    producer, _ = _make_packed_mla_view_worker(
        {"L0": (0, 128)},
        128,
        pp_size=2,
        backend_names=("FLASHMLA", "INDEXER"),
    )
    consumer, _ = _make_packed_mla_view_worker(
        {"L0": (0, 128)}, 128, backend_names=remote_backends
    )
    assert (producer.compat_hash == consumer.compat_hash) is compatible


@pytest.mark.cpu_test
@pytest.mark.parametrize("remote_block_size", [8, 32])
def test_packed_mla_rejects_unequal_block_sizes_before_peer_registration(
    remote_block_size,
):
    producer, _ = _make_packed_mla_view_worker({"L0": (0, 128)}, 128, pp_size=2)
    metadata = msgspec.msgpack.decode(
        producer.xfer_handshake_metadata.agent_metadata_bytes, type=NixlAgentMetadata
    )
    metadata.engine_id = "remote"
    metadata.block_size = remote_block_size
    with pytest.raises(NotImplementedError, match="identical P/D block sizes"):
        producer.add_remote_agent(metadata)
    assert len(producer.nixl_wrapper.dlists) == 1  # Only the local list exists.
    assert producer.tp_mappings == {}
    assert "remote" not in producer.dst_region_num_blocks
    with pytest.raises(KeyError):
        producer.transfer_topo.get_engine_info("remote")


@pytest.mark.cpu_test
def test_local_descriptors_follow_each_region_pool_capacity():
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.worker import (
        NixlConnectorWorker,
    )

    worker = object.__new__(NixlConnectorWorker)
    worker.transfer_topo = MagicMock()
    worker.device_id = 0
    worker.block_len_per_layer = [16, 16]
    worker.block_stride_per_layer = [16, 16]
    worker.region_num_blocks = [2, 3]
    worker.num_regions = 2
    worker._ssm_region_indices = []
    worker._ple_region_index = None
    worker._transfer_layer_region_indices = ()

    descriptors = worker._build_fa_local([100, 1000])

    # One (addr, len, dev, stride, count) run per region, sized to its pool.
    assert descriptors.tolist() == [[100, 16, 0, 16, 2], [1000, 16, 0, 16, 3]]


def _register_remote_agents(worker, metadata, tp_size):
    """Mirror the async handshake callback that publishes prepared agents."""
    worker._remote_agents[metadata.engine_id] = {
        (0, rank): worker.add_remote_agent(
            metadata, remote_tp_rank=rank, remote_tp_size=tp_size
        )
        for rank in range(tp_size)
    }


def _assert_local_writes_within(worker, owned_ranges):
    nixl = worker.nixl_wrapper
    assert nixl.xfers, "no transfers were posted"
    violations = []
    total_descs = 0
    for op, local_handle, local_ids, _, remote_ids in nixl.xfers:
        assert len(local_ids) == len(remote_ids)
        desc_arr = nixl.dlists[local_handle]
        for i in local_ids:
            addr, length, _dev = desc_arr[int(i)]
            addr, length = int(addr), int(length)
            total_descs += 1
            if not any(lo <= addr and addr + length <= hi for lo, hi in owned_ranges):
                violations.append((int(i), hex(addr), length))
    assert not violations, (
        f"{len(violations)}/{total_descs} local descriptors write outside "
        f"the request's own blocks: {violations[:10]}"
    )
    return total_descs


# ---------------------------------------------------------------------------
# large_page allocator: per-group kernel-row regions
# ---------------------------------------------------------------------------

# Shared geometry for the large-page fixtures: one pool of N pages of S
# bytes. S is a multiple of neither group's block, so some block ids cross a
# page end.
_LP_PAGE_BYTES = 1792  # S_r
_LP_NUM_PAGES = 4  # N


def fill_lp_geometry(config: KVCacheConfig) -> KVCacheConfig:
    """Fill in large-page geometry on a plain config, in place.

    Rows are whole blocks (kernel_rows_per_block=1) and each large page holds
    an integral number of every group's blocks, so manager block ids double
    as kernel block ids.
    """
    rows = [group.kv_cache_spec.page_size_bytes for group in config.kv_cache_groups]
    large_page_bytes = math.lcm(*rows)
    for group in config.kv_cache_groups:
        group.kernel_rows_per_block = 1
        group.row_bytes = group.kv_cache_spec.page_size_bytes
        group.blocks_per_page = large_page_bytes // group.row_bytes
        group.num_blocks = config.num_pages * group.blocks_per_page
    config.num_pages = config.num_pages
    config.large_page_bytes = large_page_bytes
    return config


def make_lp_kv_cache_config(
    block_size: int,
    swa_enabled: bool = False,
    mamba_enabled: bool = False,
    sw_size: int = 128,
    num_blocks: int = 100,
    mamba_cache_mode="none",
) -> KVCacheConfig:
    """make_kv_cache_config with the large-page geometry filled in."""
    return fill_lp_geometry(
        make_kv_cache_config(
            block_size=block_size,
            swa_enabled=swa_enabled,
            mamba_enabled=mamba_enabled,
            sw_size=sw_size,
            num_blocks=num_blocks,
            mamba_cache_mode=mamba_cache_mode,
        )
    )


def _large_page_pool_views(pool, views_spec):
    """as_strided layer views over the pool: (rows, page) with row stride."""
    return {
        name: torch.as_strided(pool, (rows, page), (stride, 1), offset)
        for name, (rows, page, stride, offset) in views_spec.items()
    }


def _make_large_page_worker(
    worker_cls=NixlPullConnectorWorker,
    *,
    with_mamba: bool = False,
    kv_buffer_device: str = "cuda",
    enable_permute_local_kv: bool = False,
    leading_non_transfer_group: bool = False,
):
    """A worker over one shared large-page pool with two transfer groups.

    Group 0 is a 2-layer head-sharded (GQA) attention group with R_g=2 and
    registers one kernel-row region per layer; group 1 is a single-layer MLA
    group with R_g=1 and registers one whole-row region. With ``with_mamba``
    group 1 is instead a single-layer Mamba group (per-layer conv/ssm
    sub-descriptors) and group 0 is the whole-row MLA group.

    With ``leading_non_transfer_group`` a non-transferable group (a
    HiSparse-style resident group, holding no registered layer) precedes the
    transferable ones, so transfer-group indices differ from full group ids.
    """
    from vllm.config import set_current_vllm_config
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheConfig,
        KVCacheGroupSpec,
        KVCacheTensor,
        MambaSpec,
        MLAAttentionSpec,
    )

    block_size = 16
    pool = torch.zeros(_LP_PAGE_BYTES * _LP_NUM_PAGES, dtype=torch.uint8)

    if with_mamba:
        from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum

        attn_spec = MLAAttentionSpec(
            block_size=block_size, num_kv_heads=1, head_size=24, dtype=torch.uint8
        )
        mamba_spec = MambaSpec(
            block_size=block_size,
            shapes=((8, 3), (1, 4, 4)),
            dtypes=(torch.float16, torch.float32),
            page_size_padded=128,
            mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
        )
        attn_row = attn_spec.page_size_bytes  # R=1
        mamba_row = mamba_spec.page_size_bytes
        attn_rows = (_LP_PAGE_BYTES * _LP_NUM_PAGES) // attn_row
        mamba_rows = (_LP_PAGE_BYTES * _LP_NUM_PAGES) // mamba_row
        groups = [
            KVCacheGroupSpec(
                ["attn.0"],
                attn_spec,
                num_blocks=2 * _LP_NUM_PAGES,
                blocks_per_page=2,
                kernel_rows_per_block=1,
                row_bytes=attn_row,
            ),
            KVCacheGroupSpec(
                ["ssm.0"],
                mamba_spec,
                num_blocks=2 * _LP_NUM_PAGES,
                blocks_per_page=2,
                kernel_rows_per_block=1,
                row_bytes=mamba_row,
            ),
        ]
        views_spec = {
            "attn.0": (attn_rows, attn_row, attn_row, 0),
            "ssm.0": (mamba_rows, mamba_row, mamba_row, 0),
        }
    else:
        g0_spec = FullAttentionSpec(
            block_size=block_size, num_kv_heads=2, head_size=8, dtype=torch.uint8
        )
        g1_spec = MLAAttentionSpec(
            block_size=block_size, num_kv_heads=1, head_size=24, dtype=torch.uint8
        )
        # Group 0: two layers, two kernel rows (R_g=2) per manager block.
        g0_layer_row = g0_spec.page_size_bytes // 2
        g0_row = 2 * g0_layer_row
        g1_row = g1_spec.page_size_bytes
        g0_rows = (_LP_PAGE_BYTES * _LP_NUM_PAGES) // g0_row
        g1_rows = (_LP_PAGE_BYTES * _LP_NUM_PAGES) // g1_row
        groups = [
            KVCacheGroupSpec(
                ["g0.l0", "g0.l1"],
                g0_spec,
                num_blocks=_LP_NUM_PAGES,
                blocks_per_page=1,
                kernel_rows_per_block=2,
                row_bytes=g0_row,
            ),
            KVCacheGroupSpec(
                ["g1.l0"],
                g1_spec,
                num_blocks=2 * _LP_NUM_PAGES,
                blocks_per_page=2,
                kernel_rows_per_block=1,
                row_bytes=g1_row,
            ),
        ]
        views_spec = {
            "g0.l0": (g0_rows, g0_layer_row, g0_row, 0),
            "g0.l1": (g0_rows, g0_layer_row, g0_row, g0_layer_row),
            "g1.l0": (g1_rows, g1_row, g1_row, 0),
        }

    if leading_non_transfer_group:
        resident_spec = FullAttentionSpec(
            block_size=block_size, num_kv_heads=1, head_size=8, dtype=torch.uint8
        )
        groups.insert(
            0,
            KVCacheGroupSpec(
                ["resident.0"],
                resident_spec,
                enable_kv_transfer=False,
                num_blocks=_LP_NUM_PAGES,
                blocks_per_page=1,
                kernel_rows_per_block=1,
                row_bytes=resident_spec.page_size_bytes,
            ),
        )

    kv_cache_config = KVCacheConfig(
        num_pages=_LP_NUM_PAGES,
        kv_cache_tensors=[
            KVCacheTensor(
                size=pool.nbytes,
                layers=list(views_spec),
                layer_stride=pool.nbytes,
                block_stride=0,
            )
        ],
        kv_cache_groups=groups,
        large_page_bytes=_LP_PAGE_BYTES,
        block_size=block_size,
    )

    vllm_config = create_vllm_config(
        block_size=block_size, enable_permute_local_kv=enable_permute_local_kv
    )
    vllm_config.cache_config.enable_prefix_caching = False
    vllm_config.kv_transfer_config.kv_buffer_device = kv_buffer_device

    fake_backend = MagicMock()
    fake_backend.get_supported_kernel_block_sizes.return_value = [block_size]
    fake_backend.get_name.return_value = "TEST_ATTN"
    fake_backend.full_cls_name.return_value = "test.AttentionBackend"

    mamba_layout_patch = (
        patch(
            "vllm.model_executor.layers.mamba.mamba_utils.get_conv_state_layout",
            return_value="DS",
        )
        if with_mamba
        else nullcontext()
    )
    with (
        patch.object(bw, "NixlWrapper", _RecordingNixl),
        patch.object(bw, "get_tensor_model_parallel_rank", return_value=0),
        patch.object(bw, "get_tensor_model_parallel_world_size", return_value=1),
        patch.object(bw, "get_current_attn_backends", return_value=[fake_backend]),
        patch("threading.Thread"),
        set_current_vllm_config(vllm_config),
        mamba_layout_patch,
    ):
        worker = worker_cls(vllm_config, "local-engine", kv_cache_config)
        worker.register_kv_caches(_large_page_pool_views(pool, views_spec))
    worker._test_pool = pool
    return worker


def _large_page_remote_meta(
    worker, engine_id="remote-engine", addr_offset=0, **overrides
):
    """This worker's advertised metadata as a peer's, bases shifted."""
    metadata = msgspec.msgpack.decode(
        worker.xfer_handshake_metadata.agent_metadata_bytes, type=NixlAgentMetadata
    )
    metadata.engine_id = engine_id
    metadata.kv_caches_base_addr = [
        addr + addr_offset for addr in metadata.kv_caches_base_addr
    ]
    for field_name, value in overrides.items():
        setattr(metadata, field_name, value)
    return metadata


def _large_page_owned_ranges(worker, group_manager_ids):
    """Byte ranges owned by the given per-group manager blocks: every kernel
    row of every block, b * R + j."""
    config = worker.kv_cache_config
    pool_addr = worker._test_pool.data_ptr()
    owned = []
    for group_id, ids in enumerate(group_manager_ids):
        group = config.kv_cache_groups[group_id]
        for block in ids:
            for j in range(group.kernel_rows_per_block):
                row = block * config.kv_cache_groups[group_id].kernel_rows_per_block + j
                owned.append(
                    (
                        pool_addr + row * group.row_bytes,
                        pool_addr + (row + 1) * group.row_bytes,
                    )
                )
    return owned


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "worker_cls", [NixlPullConnectorWorker, NixlPushConnectorWorker]
)
def test_large_page_registration_per_group_row_regions(worker_cls):
    worker = _make_large_page_worker(worker_cls)
    pool_addr = worker._test_pool.data_ptr()
    config = worker.kv_cache_config
    g0, g1 = config.kv_cache_groups

    # One region per layer of the head-sharded group, one whole-row region
    # for the MLA group.
    assert worker.region_names == ["g0.l0", "g0.l1", "g1.l0"]
    assert worker.region_group_ids == [0, 0, 1]
    assert worker.block_stride_per_layer == [g0.row_bytes, g0.row_bytes, g1.row_bytes]
    assert worker.block_len_per_layer == [
        g0.kv_cache_spec.page_size_bytes // 2,
        g0.kv_cache_spec.page_size_bytes // 2,
        g1.row_bytes,
    ]
    assert worker.region_num_blocks == [
        14,  # (1792 * 4) // 512
        14,
        18,  # (1792 * 4) // 384
    ]
    # Dedup regression: the group-0 layer-0 view and the group-1 row share
    # the pool base address but have different strides; they must not merge.
    bases = worker.kv_caches_base_addr[worker.engine_id][0]
    assert bases == [pool_addr, pool_addr + 256, pool_addr]
    # One memory registration over the whole pool.
    assert worker._registered_descs == [[(pool_addr, worker._test_pool.nbytes, 0, "")]]
    assert worker._packed_member_layouts == {"g1.l0": (0, g1.row_bytes)}

    metadata = msgspec.msgpack.decode(
        worker.xfer_handshake_metadata.agent_metadata_bytes, type=NixlAgentMetadata
    )
    assert [g.kernel_rows_per_block for g in metadata.group_geometries] == [2, 1]
    assert metadata.group_layer_names == [["g0.l0", "g0.l1"], ["g1.l0"]]
    assert metadata.region_num_blocks == worker.region_num_blocks


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "worker_cls", [NixlPullConnectorWorker, NixlPushConnectorWorker]
)
def test_large_page_desc_ids_match_affine_kernel_rows(worker_cls):
    worker = _make_large_page_worker(worker_cls)

    manager_ids = ([0, 1, 2, 3], [0, 1, 4])
    kernel_ids = worker._logical_to_kernel_block_ids(manager_ids)
    # Block b of a group with R kernel rows per block is rows b * R .. b * R
    # + R - 1: group 0 has R=2, group 1 has R=1.
    assert kernel_ids == [[0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 4]]

    # Descriptor ids index the region-major kernel-row space; every kernel
    # row of a manager block maps to that block's rows of the group's views.
    desc_ids = worker._compute_desc_ids(
        kernel_ids,
        region_num_blocks=worker.dst_region_num_blocks[worker.engine_id],
        region_group_ids=worker.region_group_ids,
    )
    descs = expand_strided_descs(worker.src_blocks_data)
    pool_addr = worker._test_pool.data_ptr()
    g0_row, g1_row = 512, 384
    expected_addrs = []
    for kid in kernel_ids[0]:  # group 0: both layer regions, row-major
        expected_addrs.append(pool_addr + kid * g0_row)  # g0.l0
    for kid in kernel_ids[0]:
        expected_addrs.append(pool_addr + 256 + kid * g0_row)  # g0.l1
    for kid in kernel_ids[1]:
        expected_addrs.append(pool_addr + kid * g1_row)  # g1 row
    assert descs[desc_ids, 0].tolist() == expected_addrs


@pytest.mark.cpu_test
def test_large_page_pull_writes_stay_within_request_blocks():
    """A pulled request's local descriptors must all land in the kernel rows
    of its own manager blocks."""
    worker = _make_large_page_worker()
    metadata = _large_page_remote_meta(worker, addr_offset=0x1000_0000)
    _register_remote_agents(worker, metadata, tp_size=1)

    local_ids = ([1, 3], [2])
    remote_ids = ([0, 2], [5])
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
        NixlConnectorMetadata,
    )

    connector_meta = NixlConnectorMetadata()
    connector_meta.add_new_req_to_recv(
        request_id="req-lp",
        local_block_ids=local_ids,
        kv_transfer_params={
            "remote_block_ids": remote_ids,
            "remote_engine_id": "remote-engine",
            "remote_request_id": "prefill-req-lp",
            "remote_host": "remote-host",
            "remote_port": 1234,
            "tp_size": 1,
        },
    )
    meta = connector_meta.reqs_to_recv["req-lp"]
    meta.local_physical_block_ids = worker._logical_to_kernel_block_ids(
        meta.local_block_ids
    )
    worker._recving_metadata["req-lp"] = meta

    worker._read_blocks_for_req("req-lp", meta)

    owned = _large_page_owned_ranges(worker, local_ids)
    assert _assert_local_writes_within(worker, owned) > 0


@pytest.mark.cpu_test
def test_large_page_load_error_tags_full_group_ids():
    """A failed pull reports (group_id, block_id) in the full kv_cache_groups
    space, which is where the scheduler evicts — not in transfer-group space
    (the two diverge when a non-transferable group comes first, e.g. a
    HiSparse resident group)."""
    worker = _make_large_page_worker(leading_non_transfer_group=True)
    assert worker.kv_cache_config.transfer_group_ids == (1, 2)
    metadata = _large_page_remote_meta(worker, addr_offset=0x1000_0000)
    _register_remote_agents(worker, metadata, tp_size=1)

    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
        NixlConnectorMetadata,
    )

    local_ids = ([1, 3], [2])
    connector_meta = NixlConnectorMetadata()
    connector_meta.add_new_req_to_recv(
        request_id="req-lp",
        local_block_ids=local_ids,
        kv_transfer_params={
            "remote_block_ids": ([0, 2], [5]),
            "remote_engine_id": "remote-engine",
            "remote_request_id": "prefill-req-lp",
            "remote_host": "remote-host",
            "remote_port": 1234,
            "tp_size": 1,
        },
    )
    worker.start_load_kv(connector_meta)
    worker.nixl_wrapper.check_xfer_state = lambda handle: "ERR"

    results = worker.get_transfer_results()
    assert results.failed_recving == {"req-lp"}
    # Transfer-group indices (0, 1) map to full group ids (1, 2).
    assert worker.get_block_ids_with_load_errors() == {(1, 1), (1, 3), (2, 2)}


@pytest.mark.cpu_test
def test_large_page_fewer_remote_groups_fails_with_clear_error():
    """A peer advertising fewer transfer groups than local must hit the
    descriptive group-count error, not a raw IndexError from indexing the
    advertised per-group lists."""
    worker = _make_large_page_worker()
    metadata = _large_page_remote_meta(worker, addr_offset=0x1000_0000)
    # Truncate the advertised per-group lists to one transfer group.
    metadata.group_layer_names = metadata.group_layer_names[:1]
    metadata.group_geometries = metadata.group_geometries[:1]

    with pytest.raises(RuntimeError, match="different number of transfer groups"):
        worker.add_remote_agent(metadata, remote_tp_rank=0, remote_tp_size=1)


@pytest.mark.cpu_test
def test_large_page_push_writes_stay_within_request_blocks():
    """The push-mode equivalent: a pushed request's local (P) descriptors
    must all land in the kernel rows of its own manager blocks."""
    worker = _make_large_page_worker(NixlPushConnectorWorker)
    metadata = _large_page_remote_meta(worker, addr_offset=0x1000_0000)
    _register_remote_agents(worker, metadata, tp_size=1)

    local_ids = ([1, 3], [2])
    remote_ids = ([0, 2], [5])
    worker._do_start_push_kv(
        "req-lp",
        local_ids,
        {
            "decode_engine_id": "remote-engine",
            "local_block_ids": remote_ids,
            "request_id": "decode-req-lp",
            "decode_host": "remote-host",
            "decode_port": 1234,
            "decode_tp_size": 1,
        },
    )

    owned = _large_page_owned_ranges(worker, local_ids)
    assert _assert_local_writes_within(worker, owned) > 0


@pytest.mark.cpu_test
def test_hisparse_host_group_pulls_dense_host_blocks():
    """A HiSparse host group transfers block b as row b of each source layer.

    The host pool keeps every source layer in its own contiguous region of one
    backing allocation, block b at ``b * page``; the device MLA group beside it
    keeps its strided rows. Both sides address the host blocks
    with that dense mapping.
    """
    from vllm.config import set_current_vllm_config
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
        NixlConnectorMetadata,
    )
    from vllm.v1.hisparse.layout import HiSparseLayout, add_hisparse_host_config
    from vllm.v1.kv_cache_interface import KVCacheGroupRole

    block_size = 16
    host_num_blocks = 6
    device_spec = MLAAttentionSpec(
        block_size=block_size, num_kv_heads=1, head_size=24, dtype=torch.uint8
    )
    host_spec = MLAAttentionSpec(
        block_size=block_size, num_kv_heads=1, head_size=32, dtype=torch.uint8
    )
    device_row = device_spec.page_size_bytes  # 384: 4 blocks per 1792-byte page
    host_page = host_spec.page_size_bytes  # 512
    pool = torch.zeros(_LP_PAGE_BYTES * _LP_NUM_PAGES, dtype=torch.uint8)
    device_group = KVCacheGroupSpec(
        ["dev.0"],
        device_spec,
        num_blocks=4 * _LP_NUM_PAGES,
        blocks_per_page=4,
        kernel_rows_per_block=1,
        row_bytes=device_row,
    )
    device_config = KVCacheConfig(
        num_pages=_LP_NUM_PAGES,
        kv_cache_tensors=[
            KVCacheTensor(
                size=pool.nbytes,
                layers=["dev.0"],
                layer_stride=device_row,
                block_stride=device_row,
            )
        ],
        kv_cache_groups=[device_group],
        large_page_bytes=_LP_PAGE_BYTES,
        block_size=block_size,
    )
    source_group = KVCacheGroupSpec(
        ["src.0", "src.1"],
        host_spec,
        host_resident=True,
        role=KVCacheGroupRole.HISPARSE_SOURCE,
        layer_kv_cache_specs={"src.0": host_spec, "src.1": host_spec},
    )
    kv_cache_config = add_hisparse_host_config(
        device_config,
        HiSparseLayout(
            source_group=source_group,
            host_num_blocks=host_num_blocks,
            host_block_stride=2 * host_page,
            shared_host_pool=False,
        ),
    )
    host_backing = torch.zeros(2 * host_num_blocks * host_page, dtype=torch.uint8)
    caches = {
        "dev.0": torch.as_strided(
            pool, (pool.nbytes // device_row, device_row), (device_row, 1)
        )
    }
    for tensor in kv_cache_config.kv_cache_tensors:
        if tensor.host_resident:
            (caches[tensor.layers[0]],) = create_kv_cache_views(
                host_backing, host_spec, host_num_blocks, KVCacheLayout.BLHNC, tensor
            )

    vllm_config = create_vllm_config(block_size=block_size)
    vllm_config.cache_config.enable_prefix_caching = False
    fake_backend = MagicMock()
    fake_backend.get_supported_kernel_block_sizes.return_value = [block_size]
    fake_backend.get_name.return_value = "TEST_ATTN"
    fake_backend.full_cls_name.return_value = "test.AttentionBackend"
    with (
        patch.object(bw, "NixlWrapper", _RecordingNixl),
        patch.object(bw, "get_tensor_model_parallel_rank", return_value=0),
        patch.object(bw, "get_tensor_model_parallel_world_size", return_value=1),
        patch.object(bw, "get_current_attn_backends", return_value=[fake_backend]),
        patch("threading.Thread"),
        set_current_vllm_config(vllm_config),
    ):
        worker = NixlPullConnectorWorker(vllm_config, "local-engine", kv_cache_config)
        worker.register_kv_caches(caches)

    remote_offset = 0x1000_0000
    metadata = _large_page_remote_meta(worker, addr_offset=remote_offset)
    _register_remote_agents(worker, metadata, tp_size=1)
    # Transfer groups: (host source, device).
    local_ids = ([1, 4], [1])
    remote_ids = ([2, 5], [5])
    connector_meta = NixlConnectorMetadata()
    connector_meta.add_new_req_to_recv(
        request_id="req-hs",
        local_block_ids=local_ids,
        kv_transfer_params={
            "remote_block_ids": remote_ids,
            "remote_engine_id": "remote-engine",
            "remote_request_id": "prefill-req-hs",
            "remote_host": "remote-host",
            "remote_port": 1234,
            "tp_size": 1,
        },
    )
    meta = connector_meta.reqs_to_recv["req-hs"]
    meta.local_physical_block_ids = worker._logical_to_kernel_block_ids(
        meta.local_block_ids
    )
    worker._recving_metadata["req-hs"] = meta
    worker._read_blocks_for_req("req-hs", meta)

    nixl = worker.nixl_wrapper
    local_ranges: set[tuple[int, int]] = set()
    remote_ranges: set[tuple[int, int]] = set()
    for _, local_handle, xfer_local_ids, remote_handle, xfer_remote_ids in nixl.xfers:
        for handle, ids, ranges in (
            (local_handle, xfer_local_ids, local_ranges),
            (remote_handle, xfer_remote_ids, remote_ranges),
        ):
            for desc_id in ids:
                addr, length, _ = nixl.dlists[handle][int(desc_id)]
                ranges.add((int(addr), int(length)))

    def expected_ranges(host_blocks, device_row_id, offset):
        host_base = host_backing.data_ptr() + offset
        return {
            (host_base + layer * host_num_blocks * host_page + b * host_page, host_page)
            for layer in range(2)
            for b in host_blocks
        } | {(pool.data_ptr() + offset + device_row_id * device_row, device_row)}

    assert local_ranges == expected_ranges([1, 4], 1, 0)
    assert remote_ranges == expected_ranges([2, 5], 5, remote_offset)


@pytest.mark.cpu_test
def test_large_page_mamba_per_layer_sub_descriptors():
    """Mamba groups keep per-layer conv/ssm sub-descriptors under large_page,
    with per-region kernel-row counts."""
    worker = _make_large_page_worker(with_mamba=True)
    config = worker.kv_cache_config
    attn_group, mamba_group = config.kv_cache_groups
    pool_addr = worker._test_pool.data_ptr()

    # Region 0: whole-row MLA; region 1: the mamba layer. Same pool base,
    # different strides -> distinct regions.
    assert worker.region_group_ids == [0, 1]
    assert worker._ssm_region_indices == [1]
    assert worker.block_stride_per_layer == [
        attn_group.row_bytes,
        mamba_group.row_bytes,
    ]
    assert worker.kv_caches_base_addr[worker.engine_id][0] == [
        pool_addr,
        pool_addr,
    ]

    # Local descs: 18 FA rows, then 4 sub-descriptor runs of 56 rows each.
    attn_rows = worker.region_num_blocks[0]
    mamba_rows = worker.region_num_blocks[1]
    assert (attn_rows, mamba_rows) == (18, 56)
    assert worker.src_blocks_data.shape == (1 + 4, 5)
    for run in worker.src_blocks_data[1:]:
        assert run[4] == mamba_rows

    kernel_ids = worker._logical_to_kernel_block_ids(([1], [2]))
    desc_ids = worker._compute_desc_ids(
        kernel_ids,
        region_num_blocks=worker.dst_region_num_blocks[worker.engine_id],
        region_group_ids=worker.region_group_ids,
    )
    # FA desc for the attention kernel row; then the 4 mamba sub-runs, each
    # mamba_rows long, at the mamba kernel row.
    attn_row_id = kernel_ids[0][0]
    ssm_row_id = kernel_ids[1][0]
    assert desc_ids.tolist() == [attn_row_id] + [
        attn_rows + sub * mamba_rows + ssm_row_id for sub in range(4)
    ]

    # A pull transfer resolves local descs into the request's own rows.
    metadata = _large_page_remote_meta(worker, addr_offset=0x1000_0000)
    _register_remote_agents(worker, metadata, tp_size=1)

    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
        NixlConnectorMetadata,
    )

    connector_meta = NixlConnectorMetadata()
    connector_meta.add_new_req_to_recv(
        request_id="req-lp-mamba",
        local_block_ids=([1], [2]),
        kv_transfer_params={
            "remote_block_ids": ([0], [1]),
            "remote_engine_id": "remote-engine",
            "remote_request_id": "prefill-req-lp-mamba",
            "remote_host": "remote-host",
            "remote_port": 1234,
            "tp_size": 1,
        },
    )
    meta = connector_meta.reqs_to_recv["req-lp-mamba"]
    meta.local_physical_block_ids = worker._logical_to_kernel_block_ids(
        meta.local_block_ids
    )
    worker._recving_metadata["req-lp-mamba"] = meta
    worker._read_blocks_for_req("req-lp-mamba", meta)

    owned = _large_page_owned_ranges(worker, ([1], [2]))
    assert _assert_local_writes_within(worker, owned) > 0


@pytest.mark.cpu_test
def test_large_page_host_staging_mirrors_device_geometry():
    """kv_buffer_device="cpu": host buffers mirror the device row layout per
    group, so registration advertises the same geometry as device buffers."""
    device_worker = _make_large_page_worker(with_mamba=True)
    host_worker = _make_large_page_worker(with_mamba=True, kv_buffer_device="cpu")
    assert host_worker.use_host_buffer
    assert host_worker.host_xfer_buffers

    # Per layer, the host view has the device view's shape, strides and
    # storage offset over one CPU backing buffer per transfer group.
    assert set(host_worker.host_xfer_buffers) == set(host_worker.device_kv_caches)
    for name, host_view in host_worker.host_xfer_buffers.items():
        device_view = host_worker.device_kv_caches[name]
        assert host_view.device.type == "cpu"
        assert host_view.shape == device_view.shape
        assert host_view.stride() == device_view.stride()
        assert host_view.storage_offset() == device_view.storage_offset()
    # One CPU backing buffer per shared device storage.
    assert len(
        {
            view.untyped_storage().data_ptr()
            for view in host_worker.host_xfer_buffers.values()
        }
    ) == len(
        {
            view.untyped_storage().data_ptr()
            for view in host_worker.device_kv_caches.values()
        }
    )

    # The registered regions and the advertised geometry are identical to the
    # device-buffer case, except the (host) base addresses.
    def handshake_geometry(worker):
        metadata = msgspec.msgpack.decode(
            worker.xfer_handshake_metadata.agent_metadata_bytes, type=NixlAgentMetadata
        )
        metadata.agent_metadata = b""
        metadata.kv_caches_base_addr = []
        return msgspec.msgpack.encode(metadata)

    assert handshake_geometry(host_worker) == handshake_geometry(device_worker)
    assert host_worker.region_names == device_worker.region_names
    assert host_worker.region_num_blocks == device_worker.region_num_blocks
    assert host_worker.block_len_per_layer == device_worker.block_len_per_layer
    assert host_worker.block_stride_per_layer == device_worker.block_stride_per_layer
    assert [desc[1] for desc in host_worker._registered_descs[0]] == [
        desc[1] for desc in device_worker._registered_descs[0]
    ]


@pytest.mark.cpu_test
def test_large_page_host_staging_copies_per_group():
    """The d2h/h2d staging copies run per transfer group, indexed by that
    group's kernel block ids, over views that mirror the device rows."""
    from vllm.distributed.kv_transfer.kv_connector.utils import copy_kv_blocks
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
        NixlConnectorMetadata,
    )

    worker = _make_large_page_worker(with_mamba=True, kv_buffer_device="cpu")
    config = worker.kv_cache_config
    pool = worker._test_pool
    pool.copy_(torch.arange(pool.nbytes, dtype=torch.int32).to(torch.uint8))

    copies = []

    def recording_copy(src, dst, src_ids, dst_ids, direction):
        copies.append((sorted(src), list(src_ids), list(dst_ids), direction))
        copy_kv_blocks(src, dst, src_ids, dst_ids, direction)

    worker.set_host_xfer_buffer_ops(recording_copy)

    # d2h: manager block ids expand to per-group kernel block ids.
    manager_ids = ([1], [2])
    save_meta = NixlConnectorMetadata()
    save_meta.add_new_req_to_save("req-host", manager_ids, {})
    worker.save_kv_to_host(save_meta)

    kernel_ids = [
        [block * config.kv_cache_groups[g].kernel_rows_per_block for block in group_ids]
        for g, group_ids in enumerate(manager_ids)
    ]
    assert copies == [
        (["attn.0"], kernel_ids[0], kernel_ids[0], "d2h"),
        (["ssm.0"], kernel_ids[1], kernel_ids[1], "d2h"),
    ]
    for g, name in enumerate(("attn.0", "ssm.0")):
        for row in kernel_ids[g]:
            assert torch.equal(
                worker.host_xfer_buffers[name][row], worker.device_kv_caches[name][row]
            )

    # h2d: after wiping the device pool, the received rows land back.
    pool.zero_()
    recv_meta = NixlConnectorMetadata()
    recv_meta.add_new_req_to_recv(
        request_id="req-host",
        local_block_ids=manager_ids,
        kv_transfer_params={
            "remote_block_ids": manager_ids,
            "remote_engine_id": "remote-engine",
            "remote_request_id": "prefill-req-host",
            "remote_host": "remote-host",
            "remote_port": 1234,
            "tp_size": 1,
        },
    )
    meta = recv_meta.reqs_to_recv["req-host"]
    meta.local_physical_block_ids = worker._logical_to_kernel_block_ids(
        meta.local_block_ids
    )
    worker.sync_recved_kv_to_device("req-host", meta)

    assert [call[3] for call in copies[2:]] == ["h2d", "h2d"]
    assert [sorted(call[0]) for call in copies[2:]] == [["attn.0"], ["ssm.0"]]
    for g, name in enumerate(("attn.0", "ssm.0")):
        for row in kernel_ids[g]:
            assert torch.equal(
                worker.device_kv_caches[name][row], worker.host_xfer_buffers[name][row]
            )


@pytest.mark.cpu_test
def test_large_page_host_staging_rejects_permute_local_kv():
    """Mirrored host views cannot express the LBNHC<->LBHNC physical permute."""
    with pytest.raises(NotImplementedError, match="enable_permute_local_kv"):
        _make_large_page_worker(kv_buffer_device="cpu", enable_permute_local_kv=True)


def _make_csa_linear_worker(worker_cls=NixlPullConnectorWorker):
    """A CSA-linear worker: a KDA group, a TP-replicated PLE conv-state group
    and a CircularBufferSpec ring group, preceded by a non-transferable
    resident group (so transfer-group indices differ from full group ids).

    Rows divide the page, so manager block ids double as kernel row ids.
    """
    from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
    from vllm.v1.kv_cache_interface import (
        CircularBufferSpec,
        FullAttentionSpec,
        MambaSpec,
    )

    block_size = 16
    page_bytes = 1024
    num_pages = 2
    pool = torch.zeros(page_bytes * num_pages, dtype=torch.uint8)

    resident_spec = FullAttentionSpec(
        block_size=block_size, num_kv_heads=1, head_size=8, dtype=torch.uint8
    )
    kda_spec = MambaSpec(
        block_size=block_size,
        shapes=((8, 3), (1, 4, 4)),
        dtypes=(torch.float16, torch.float32),
        page_size_padded=128,
        mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
    )
    ple_spec = MambaSpec(
        block_size=block_size,
        shapes=((4,), (8,)),
        dtypes=(torch.float16, torch.float16),
        page_size_padded=32,
        mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
        tp_replicated=True,
    )
    ring_spec = CircularBufferSpec(
        block_size=2, num_kv_heads=1, head_size=16, head_size_v=0, dtype=torch.float16
    )
    assert ring_spec.page_size_bytes == 64

    groups = [
        KVCacheGroupSpec(
            ["resident.0"],
            resident_spec,
            enable_kv_transfer=False,
            num_blocks=num_pages,
            blocks_per_page=1,
            kernel_rows_per_block=1,
            row_bytes=resident_spec.page_size_bytes,
        ),
        KVCacheGroupSpec(
            ["kda.0"],
            kda_spec,
            num_blocks=2 * num_pages * 4,
            blocks_per_page=8,
            kernel_rows_per_block=1,
            row_bytes=128,
        ),
        KVCacheGroupSpec(
            ["ple.0"],
            ple_spec,
            num_blocks=2 * num_pages * 16,
            blocks_per_page=32,
            kernel_rows_per_block=1,
            row_bytes=32,
        ),
        KVCacheGroupSpec(
            ["ring.0"],
            ring_spec,
            num_blocks=2 * num_pages * 8,
            blocks_per_page=16,
            kernel_rows_per_block=1,
            row_bytes=64,
        ),
    ]
    views = {
        # CSA-linear registers in dict order. All three views share the pool
        # base: only the (group, len, stride) key keeps them distinct.
        "kda.0": torch.as_strided(pool, (16, 128), (128, 1), 0),
        "ple.0": torch.as_strided(pool, (64, 32), (32, 1), 0),
        "ring.0": torch.as_strided(pool, (32, 64), (64, 1), 0),
    }
    kv_cache_config = KVCacheConfig(
        num_pages=num_pages,
        kv_cache_tensors=[
            KVCacheTensor(
                size=pool.nbytes,
                layers=list(views),
                layer_stride=pool.nbytes,
                block_stride=0,
            )
        ],
        kv_cache_groups=groups,
        large_page_bytes=page_bytes,
        block_size=block_size,
    )

    vllm_config = create_vllm_config(block_size=block_size)
    vllm_config.cache_config.enable_prefix_caching = False

    fake_backend = MagicMock()
    fake_backend.get_supported_kernel_block_sizes.return_value = [block_size]
    fake_backend.get_name.return_value = "TEST_ATTN"
    fake_backend.full_cls_name.return_value = "test.AttentionBackend"

    with (
        patch.object(bw, "NixlWrapper", _RecordingNixl),
        patch.object(bw, "get_tensor_model_parallel_rank", return_value=0),
        patch.object(bw, "get_tensor_model_parallel_world_size", return_value=1),
        patch.object(bw, "get_current_attn_backends", return_value=[fake_backend]),
        patch("threading.Thread"),
        patch(
            "vllm.model_executor.layers.mamba.mamba_utils.get_conv_state_layout",
            return_value="DS",
        ),
    ):
        worker = worker_cls(vllm_config, "local-engine", kv_cache_config)
        worker.register_kv_caches(views)
    worker._test_pool = pool
    return worker


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "worker_cls", [NixlPullConnectorWorker, NixlPushConnectorWorker]
)
def test_csa_linear_ple_registration_and_desc_geometry(worker_cls):
    """CSA-linear registration: the KDA group keeps per-layer conv/ssm
    sub-descriptors, the TP-replicated PLE conv state gets its own run with
    its own page length, and the ring group is a plain kernel-row region.

    _ple_group_index is in transfer-group space (its consumer,
    _compute_desc_ids, enumerates transfer groups): with a leading
    non-transferable group the two spaces diverge, and a full-space index
    would misroute the ring group's descriptors as PLE.
    """
    worker = _make_csa_linear_worker(worker_cls)

    assert worker._is_csa_linear
    assert worker.kv_cache_config.transfer_group_ids == (1, 2, 3)
    # Region order follows view insertion order for CSA-linear models.
    assert worker.region_names == ["kda.0", "ple.0", "ring.0"]
    assert worker.region_group_ids == [0, 1, 2]
    assert worker._ssm_region_indices == [0]
    assert worker._ple_region_index == 1
    assert worker._ple_group_index == 1  # transfer-group space
    assert worker._ple_block_len == 32
    assert worker._fa_region_indices() == [2]
    assert worker._region_is_mla == [False, False, False]

    metadata = msgspec.msgpack.decode(
        worker.xfer_handshake_metadata.agent_metadata_bytes, type=NixlAgentMetadata
    )
    assert metadata.ple_block_len == 32

    # Local descs: 32 ring rows, then 4 KDA sub-runs of 16 rows (3 conv
    # sub-projections + ssm), then one PLE run of 64 rows of 32 bytes.
    assert worker.region_num_blocks == [16, 64, 32]
    assert worker.num_descs == 32
    runs = worker.src_blocks_data.tolist()
    assert len(runs) == 1 + 4 + 1
    assert runs[0][1:] == [64, 0, 64, 32]  # ring: len, dev, stride, count
    assert [run[4] for run in runs[1:5]] == [16, 16, 16, 16]  # KDA sub-runs
    assert runs[5][1:] == [32, 0, 32, 64]  # PLE run: ple_block_len rows

    # Descriptor ids: FA region first, then the KDA sub-runs, then the PLE
    # run. Rows are whole pages here, so manager ids are kernel row ids.
    desc_ids = worker._compute_desc_ids(
        ([5], [7], [9]),
        region_num_blocks=worker.region_num_blocks,
        region_group_ids=worker.region_group_ids,
    )
    # KDA row 5 -> 32 + {0,16,32,48} + 5; PLE row 7 -> 96 + 7; ring row 9.
    assert desc_ids.tolist() == [37, 53, 69, 85, 103, 9]


@pytest.mark.cpu_test
def test_post_process_zeroes_only_the_untransferred_tail():
    """post_process_device_kv_on_receive with convert=False zeroes the blocks
    past the covered range (clipped out of the transfer) and preserves the
    covered blocks' contents."""
    worker = _make_large_page_worker()
    pool = worker._test_pool
    pool.fill_(1)
    caches = worker.device_kv_caches

    # Group 0 kernel rows of manager blocks [1, 2]: [4, 5, 7, 8]; the transfer
    # covered only the first two.
    to_device = lambda data, device, dtype: torch.tensor(data, dtype=dtype)  # noqa: E731
    with patch.object(bw, "async_tensor_h2d", to_device):
        worker.post_process_device_kv_on_receive([(0, [4, 5, 7, 8], 2)], convert=False)

    for name in ("g0.l0", "g0.l1"):
        assert caches[name][4].eq(1).all()
        assert caches[name][5].eq(1).all()
        assert caches[name][7].eq(0).all()
        assert caches[name][8].eq(0).all()
    # Rows outside the stale tail are untouched. (The pool is shared: the
    # zeroed bytes are only meaningful under group 0's row grid.)
    assert caches["g0.l0"][11].eq(1).all()
