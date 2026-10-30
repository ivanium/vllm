# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for MooncakeConnector hybrid FA + GDN support.

GDN is represented as a MambaSpec in vLLM, so these tests exercise the
Mooncake MambaSpec path with mamba_type=GDN_ATTN. Mamba2 is intentionally not
validated by this test module.
"""

import asyncio
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm.config import set_current_vllm_config
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector import (
    KVConnectorRole,
    MooncakeConnector,
    MooncakeConnectorScheduler,
    MooncakeConnectorWorker,
    MooncakeXferMetadata,
    SendBlockMeta,
    TransferRegion,
)
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    GroupGeometry,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)

from .test_mooncake_connector import patch_worker_dependencies
from .utils import create_request, create_vllm_config, maybe_update_block_size


def noop_shutdown():
    pass


def make_hybrid_gdn_kv_cache_config(block_size: int) -> KVCacheConfig:
    """FA + GDN groups over a shared large-page pool (identity mapping).

    The 704-byte page holds an integral number of each group's blocks, so
    manager block ids double as kernel row ids.
    """
    fa_spec = FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float16,
    )
    gdn_spec = MambaSpec(
        block_size=block_size,
        shapes=((6, 3), (1, 2, 2)),
        dtypes=(torch.float16, torch.float16),
        mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
    )
    assert gdn_spec.mamba_type == MambaAttentionBackendEnum.GDN_ATTN
    fa_row = fa_spec.page_size_bytes
    gdn_row = gdn_spec.page_size_bytes
    return KVCacheConfig(
        num_pages=16,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["model.layers.0.self_attn"],
                fa_spec,
                num_blocks=16 * 11,
                blocks_per_page=11,
                kernel_rows_per_block=1,
                row_bytes=fa_row,
            ),
            KVCacheGroupSpec(
                ["model.layers.1.linear_attn"],
                gdn_spec,
                num_blocks=16 * 16,
                blocks_per_page=16,
                kernel_rows_per_block=1,
                row_bytes=gdn_row,
            ),
        ],
        large_page_bytes=704,
        block_size=block_size,
    )


def make_hybrid_gdn_scheduler(kv_role: str) -> MooncakeConnectorScheduler:
    vllm_config = create_vllm_config(
        kv_connector="MooncakeConnector",
        kv_role=kv_role,
    )
    vllm_config.scheduler_config.disable_hybrid_kv_cache_manager = False
    return MooncakeConnectorScheduler(
        vllm_config=vllm_config,
        engine_id="test-engine",
        kv_cache_config=make_hybrid_gdn_kv_cache_config(
            vllm_config.cache_config.block_size
        ),
    )


@pytest.mark.cpu_test
def test_hybrid_gdn_remote_prefill_uses_mamba_n_minus_one():
    scheduler = make_hybrid_gdn_scheduler(kv_role="kv_consumer")
    request = create_request(num_tokens=10, do_remote_prefill=True)

    num_new_tokens, is_async = scheduler.get_num_new_matched_tokens(
        request, num_computed_tokens=0
    )

    assert num_new_tokens == request.num_prompt_tokens - 1
    assert is_async is True


@pytest.mark.cpu_test
def test_hybrid_gdn_remote_decode_truncates_prefill_before_cache_lookup():
    vllm_config = create_vllm_config(
        kv_connector="MooncakeConnector",
        kv_role="kv_producer",
    )
    vllm_config.scheduler_config.disable_hybrid_kv_cache_manager = False
    connector = MooncakeConnector(
        vllm_config,
        KVConnectorRole.SCHEDULER,
        make_hybrid_gdn_kv_cache_config(vllm_config.cache_config.block_size),
    )
    scheduler = connector.connector_scheduler
    assert scheduler is not None
    request = create_request(num_tokens=10, do_remote_decode=True)
    original_tokens = list(request.prompt_token_ids)

    connector.on_new_request(request)
    assert request.prompt_token_ids == original_tokens[:-1]
    assert request._all_token_ids == original_tokens[:-1]
    assert request.num_prompt_tokens == len(original_tokens) - 1
    assert request.max_tokens == 1
    assert request.kv_transfer_params["_p_side_truncated"] is True

    # Re-adding or rescheduling the request must not truncate another token.
    scheduler.on_new_request(request)
    assert request.prompt_token_ids == original_tokens[:-1]

    # Prefix-cache matching must not mutate the request after its local lookup.
    with patch.object(
        scheduler,
        "_truncate_mamba_request_for_prefill",
        side_effect=AssertionError("late Mamba truncation"),
    ):
        num_new_tokens, is_async = scheduler.get_num_new_matched_tokens(
            request, num_computed_tokens=0
        )

    assert num_new_tokens == 0
    assert is_async is False


def test_register_kv_caches_emits_fa_and_gdn_regions(monkeypatch):
    monkeypatch.setenv("VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT", "5")
    vllm_config = create_vllm_config(
        kv_connector="MooncakeConnector",
        kv_role="kv_consumer",
        block_size=maybe_update_block_size(16),
    )
    kv_cache_config = make_hybrid_gdn_kv_cache_config(
        vllm_config.cache_config.block_size
    )

    with set_current_vllm_config(vllm_config), patch_worker_dependencies():
        connector = MooncakeConnector(
            vllm_config,
            KVConnectorRole.WORKER,
            kv_cache_config,
        )
        worker = connector.connector_worker

        fa_group, gdn_group = kv_cache_config.kv_cache_groups
        pool = torch.empty(
            kv_cache_config.num_pages * kv_cache_config.large_page_bytes,
            dtype=torch.int8,
        )
        # Per-layer kernel-row views over the shared pool: each group's row
        # grid starts at the pool base.
        fa_cache = torch.as_strided(
            pool,
            (pool.nbytes // fa_group.row_bytes, fa_group.row_bytes),
            (fa_group.row_bytes, 1),
            0,
        )
        gdn_cache = torch.as_strided(
            pool,
            (pool.nbytes // gdn_group.row_bytes, gdn_group.row_bytes),
            (gdn_group.row_bytes, 1),
            0,
        )

        worker.register_kv_caches(
            {
                "model.layers.0.self_attn": fa_cache,
                "model.layers.1.linear_attn": gdn_cache,
            }
        )

        assert worker.transfer_topo.is_mamba is True
        assert worker.registered_layer_names == [
            "model.layers.0.self_attn",
            "model.layers.1.linear_attn",
        ]
        fa_spec = fa_group.kv_cache_spec
        gdn_spec = gdn_group.kv_cache_spec
        assert worker.block_len_per_layer == [fa_group.row_bytes, gdn_group.row_bytes]
        assert worker.kv_block_len_per_layer == [
            fa_spec.page_size_bytes,
            gdn_spec.page_size_bytes,
        ]
        assert worker.registered_group_indices == [0, 1]
        # Both groups' row grids start at the pool base; the regions must
        # not merge.
        assert worker.kv_caches_base_addr == [pool.data_ptr(), pool.data_ptr()]

        worker.shutdown()
        worker.shutdown = noop_shutdown
        connector.connector_worker = None


def test_register_kv_caches_scales_attention_len_to_kernel_block(monkeypatch):
    """A group with R>1 registers views whose payload is one kernel row."""
    monkeypatch.setenv("VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT", "5")
    vllm_config = create_vllm_config(
        kv_connector="MooncakeConnector",
        kv_role="kv_consumer",
        block_size=maybe_update_block_size(16),
    )
    kv_cache_config = make_hybrid_gdn_kv_cache_config(
        vllm_config.cache_config.block_size
    )
    fa_group = kv_cache_config.kv_cache_groups[0]
    fa_group.kernel_rows_per_block = 4
    fa_group.row_bytes = fa_group.kv_cache_spec.page_size_bytes // 4

    with set_current_vllm_config(vllm_config), patch_worker_dependencies():
        connector = MooncakeConnector(
            vllm_config,
            KVConnectorRole.WORKER,
            kv_cache_config,
        )
        worker = connector.connector_worker

        # The worker binds views at kernel-block granularity, so one row of
        # the view is one kernel block, not one manager block.
        fa_cache = torch.empty(64, fa_group.row_bytes, dtype=torch.uint8)
        worker.register_kv_caches({"model.layers.0.self_attn": fa_cache})

        assert worker.block_len_per_layer == [fa_group.row_bytes]
        assert worker.kv_block_len_per_layer == [fa_group.row_bytes]

        worker.shutdown()
        worker.shutdown = noop_shutdown
        connector.connector_worker = None


@pytest.mark.asyncio
async def test_build_transfer_params_uses_non_overlapping_physical_pages():
    kv_cache_config = make_hybrid_gdn_kv_cache_config(block_size=16)
    fa_group = kv_cache_config.kv_cache_groups[0]
    fa_group.kernel_rows_per_block = 4
    fa_group.row_bytes = fa_group.kv_cache_spec.page_size_bytes // 4
    worker = object.__new__(MooncakeConnectorWorker)
    worker.shutdown = noop_shutdown
    # __new__ skips __init__, where these caches are created.
    worker._prepared_transfer_regions = {}
    worker._layout_by_peer = OrderedDict()
    worker._acked_layout_peers = set()
    worker.use_mla = False
    worker.engine = SimpleNamespace(batch_register_memory=lambda *_: 0)
    worker.is_kv_consumer = True
    worker.tp_rank = 0
    worker.tp_size = 1
    worker.transfer_topo = SimpleNamespace(local_replicates_kv_cache=False)
    worker.kv_cache_config = kv_cache_config
    worker._layer_specs = {
        layer_name: group.kv_cache_spec
        for group in kv_cache_config.kv_cache_groups
        for layer_name in group.layer_names
    }

    ratio = 4
    fa_spec = kv_cache_config.kv_cache_groups[0].kv_cache_spec
    physical_page_bytes = fa_spec.page_size_bytes // ratio
    fa_cache = torch.empty(
        64,
        physical_page_bytes,
        dtype=torch.uint8,
    )
    worker.register_kv_caches({"model.layers.0.self_attn": fa_cache})

    local_regions = worker._get_transfer_regions(
        worker.kv_caches_base_addr,
        worker.block_len_per_layer,
        worker.kv_block_len_per_layer,
        worker.registered_layer_names,
        worker.registered_layer_indices,
        worker.registered_group_indices,
    )
    remote_base_addr = 0x100000
    remote_regions = [
        TransferRegion(
            layer_name="model.layers.0.self_attn",
            layer_index=0,
            base_addr=remote_base_addr,
            block_len=physical_page_bytes,
            kv_block_len=physical_page_bytes,
            group_index=0,
        )
    ]

    local_logical_block = 2
    remote_logical_block = 5
    transfer_id = "xfer-physical-attention-pages"
    send_meta = SendBlockMeta(
        p_req_id="p-physical-attention-pages",
        transfer_id=transfer_id,
        local_block_ids=[[local_logical_block], []],
        ready=asyncio.Event(),
    )
    # The consumer advertises the same per-group geometry: one manager block
    # expands to `ratio` kernel rows on both sides.
    xfer_meta = MooncakeXferMetadata(
        remote_hostname="consumer-host",
        remote_port=54321,
        remote_tp_size=1,
        remote_tp_rank=0,
        req_blocks={
            "d-physical-attention-pages": (
                transfer_id,
                [[remote_logical_block], []],
            )
        },
        kv_caches_base_addr=[],
        block_lens=[],
        kv_block_lens=[],
        group_geometries=[
            GroupGeometry(ratio, physical_page_bytes),
            GroupGeometry(1, 44),
        ],
    )

    (
        src_ptrs,
        dst_ptrs,
        lengths,
        err_reqs,
        err_msg,
    ) = await worker._build_transfer_params(
        [("d-physical-attention-pages", send_meta)],
        xfer_meta,
        local_regions,
        remote_regions,
    )

    assert err_reqs == []
    assert err_msg is None
    assert src_ptrs == [
        fa_cache.data_ptr() + local_logical_block * ratio * physical_page_bytes
    ]
    assert dst_ptrs == [
        remote_base_addr + remote_logical_block * ratio * physical_page_bytes
    ]
    assert lengths == [ratio * physical_page_bytes]

    def split_contiguous_blocks(src_ids, dst_ids):
        return [[block_id] for block_id in src_ids], [
            [block_id] for block_id in dst_ids
        ]

    with patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.mooncake."
        "mooncake_connector.group_concurrent_contiguous",
        side_effect=split_contiguous_blocks,
    ):
        (
            src_ptrs,
            dst_ptrs,
            lengths,
            err_reqs,
            err_msg,
        ) = await worker._build_transfer_params(
            [("d-physical-attention-pages", send_meta)],
            xfer_meta,
            local_regions,
            remote_regions,
        )

    expected_src_ptrs = [
        fa_cache.data_ptr()
        + (local_logical_block * ratio + offset) * physical_page_bytes
        for offset in range(ratio)
    ]
    expected_dst_ptrs = [
        remote_base_addr + (remote_logical_block * ratio + offset) * physical_page_bytes
        for offset in range(ratio)
    ]
    assert err_reqs == []
    assert err_msg is None
    assert src_ptrs == expected_src_ptrs
    assert dst_ptrs == expected_dst_ptrs
    assert lengths == [physical_page_bytes] * ratio
    assert all(
        next_ptr - ptr == physical_page_bytes
        for ptr, next_ptr in zip(src_ptrs, src_ptrs[1:])
    )
    assert all(
        next_ptr - ptr == physical_page_bytes
        for ptr, next_ptr in zip(dst_ptrs, dst_ptrs[1:])
    )


def test_register_kv_caches_deduplicates_shared_backing_memory(monkeypatch):
    monkeypatch.setenv("VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT", "5")
    vllm_config = create_vllm_config(
        kv_connector="MooncakeConnector",
        kv_role="kv_consumer",
        block_size=maybe_update_block_size(16),
    )
    kv_cache_config = make_hybrid_gdn_kv_cache_config(
        vllm_config.cache_config.block_size
    )

    with set_current_vllm_config(vllm_config), patch_worker_dependencies():
        connector = MooncakeConnector(
            vllm_config,
            KVConnectorRole.WORKER,
            kv_cache_config,
        )
        worker = connector.connector_worker

        # All groups share the one large-page pool backing.
        fa_group, gdn_group = kv_cache_config.kv_cache_groups
        pool = torch.empty(
            kv_cache_config.num_pages * kv_cache_config.large_page_bytes,
            dtype=torch.int8,
        )
        fa_cache = torch.as_strided(
            pool, (11, fa_group.row_bytes), (fa_group.row_bytes, 1), 0
        )
        gdn_cache = torch.as_strided(
            pool, (16, gdn_group.row_bytes), (gdn_group.row_bytes, 1), 0
        )

        with patch.object(
            worker.engine, "batch_register_memory", return_value=0
        ) as batch_register_memory:
            worker.register_kv_caches(
                {
                    "model.layers.0.self_attn": fa_cache,
                    "model.layers.1.linear_attn": gdn_cache,
                }
            )

        assert worker.kv_caches_base_addr == [
            fa_cache.data_ptr(),
            gdn_cache.data_ptr(),
        ]
        batch_register_memory.assert_called_once()
        registered_ptrs, registered_lens = batch_register_memory.call_args[0]
        assert registered_ptrs == [pool.data_ptr()]
        assert registered_lens == [pool.untyped_storage().nbytes()]

        worker.shutdown()
        worker.shutdown = noop_shutdown
        connector.connector_worker = None


def test_hybrid_gdn_transfer_params_preserve_group_identity(monkeypatch):
    monkeypatch.setenv("VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT", "5")
    vllm_config = create_vllm_config(
        kv_connector="MooncakeConnector",
        kv_role="kv_producer",
        block_size=maybe_update_block_size(16),
    )
    kv_cache_config = make_hybrid_gdn_kv_cache_config(
        vllm_config.cache_config.block_size
    )

    with set_current_vllm_config(vllm_config), patch_worker_dependencies():
        connector = MooncakeConnector(
            vllm_config,
            KVConnectorRole.WORKER,
            kv_cache_config,
        )
        worker = connector.connector_worker

        block_len = 0x100
        transfer_id = "xfer-hybrid-gdn"

        async def build_transfer_params():
            send_meta = SendBlockMeta(
                p_req_id="p-hybrid-gdn",
                transfer_id=transfer_id,
                local_block_ids=[
                    [10, 11],
                    [NULL_BLOCK_ID, 4],
                ],
                ready=asyncio.Event(),
            )
            return await worker._build_transfer_params(
                [("d-hybrid-gdn", send_meta)],
                xfer_meta,
                local_regions,
                remote_regions,
            )

        xfer_meta = MooncakeXferMetadata(
            remote_hostname="consumer-host",
            remote_port=54321,
            remote_tp_size=1,
            remote_tp_rank=0,
            req_blocks={
                "d-hybrid-gdn": (
                    transfer_id,
                    [
                        [30, 31],
                        [NULL_BLOCK_ID, 7],
                    ],
                )
            },
            kv_caches_base_addr=[],
            block_lens=[],
            kv_block_lens=[],
            group_geometries=[GroupGeometry(1, 64), GroupGeometry(1, 44)],
        )

        local_regions = [
            TransferRegion(
                layer_name="model.layers.1.linear_attn",
                layer_index=1,
                base_addr=0x5000,
                block_len=block_len,
                kv_block_len=block_len,
                group_index=1,
            ),
            TransferRegion(
                layer_name="model.layers.0.self_attn",
                layer_index=0,
                base_addr=0x1000,
                block_len=block_len,
                kv_block_len=block_len,
                group_index=0,
            ),
        ]
        remote_regions = [
            TransferRegion(
                layer_name="model.layers.1.linear_attn",
                layer_index=1,
                base_addr=0x6000,
                block_len=block_len,
                kv_block_len=block_len,
                group_index=1,
            ),
            TransferRegion(
                layer_name="model.layers.0.self_attn",
                layer_index=0,
                base_addr=0x2000,
                block_len=block_len,
                kv_block_len=block_len,
                group_index=0,
            ),
        ]

        src_ptrs, dst_ptrs, lengths, err_reqs, err_msg = asyncio.run(
            build_transfer_params()
        )

        assert err_reqs == []
        assert err_msg is None
        assert src_ptrs == [
            0x5000 + 4 * block_len,
            0x1000 + 10 * block_len,
        ]
        assert dst_ptrs == [
            0x6000 + 7 * block_len,
            0x2000 + 30 * block_len,
        ]
        assert lengths == [block_len, 2 * block_len]

        worker.shutdown()
        worker.shutdown = noop_shutdown
        connector.connector_worker = None


def test_hybrid_gdn_keeps_packed_fa_and_gdn_regions_whole(
    monkeypatch,
):
    monkeypatch.setenv("VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT", "5")
    vllm_config = create_vllm_config(
        kv_connector="MooncakeConnector",
        kv_role="kv_producer",
        block_size=maybe_update_block_size(16),
    )
    kv_cache_config = make_hybrid_gdn_kv_cache_config(
        vllm_config.cache_config.block_size
    )

    with set_current_vllm_config(vllm_config), patch_worker_dependencies():
        connector = MooncakeConnector(
            vllm_config,
            KVConnectorRole.WORKER,
            kv_cache_config,
        )
        worker = connector.connector_worker

        worker.transfer_topo = SimpleNamespace(is_kv_layout_blocks_first=False)
        regions = worker._get_transfer_regions(
            base_addrs=[0x1000, 0x2000],
            block_lens=[0x100, 0x100],
            kv_block_lens=[0x40, 0x100],
            layer_names=[
                "model.layers.0.self_attn",
                "model.layers.1.linear_attn",
            ],
            layer_indices=[0, 1],
            group_indices=[0, 1],
        )

        assert [
            (region.group_index, region.base_addr, region.kv_block_len)
            for region in regions
        ] == [
            (0, 0x1000, 0x40),
            (1, 0x2000, 0x100),
        ]

        worker.shutdown()
        worker.shutdown = noop_shutdown
        connector.connector_worker = None
