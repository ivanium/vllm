# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for NixlConnectorScheduler with HMA and Mamba N-1 prefill."""

import gc
import queue
import threading
from collections import defaultdict
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest
import torch

from tests.v1.attention.utils import MockMambaBuilder
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.v1.nixl import base_worker as bw
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.worker import (
    NixlConnectorWorker,
)
from vllm.v1.core.single_type_kv_cache_manager import (
    FullAttentionManager,
    SlidingWindowManager,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    GroupGeometry,
    KVCacheConfig,
    KVCacheGroupSpec,
    MLAAttentionSpec,
)

from .test_nixl_desc_geometry import make_lp_kv_cache_config
from .utils import (
    create_request,
    create_vllm_config,
    make_kv_cache_config,
    make_nixl_scheduler,
    maybe_update_block_size,
)


@pytest.fixture
def region_pull_worker():
    """Real page mapping and descriptors, with only the transport mocked.

    The large-page geometry is the identity: a large page holds one row of
    each group, so manager block ids double as kernel block ids.
    """
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.tp_mapping import TPMapping

    worker = object.__new__(NixlConnectorWorker)
    worker.engine_id = "D"
    worker.tp_rank = 0
    worker.block_size = 64
    worker.enable_permute_local_kv = False
    worker.enable_heterogeneous_attn_post_process = False
    worker._engine_last_active = {}
    worker._handshake_lock = threading.RLock()
    worker._handshake_futures = {}
    worker._transfer_layer_group_ids = ()
    worker._bidirectional_kv_xfer_enabled = False
    worker._recving_transfers = {}
    worker._failed_remote_engines = set()
    worker._invalid_remote_engines = set()
    worker._remote_registration_generations = {}
    worker.use_mla, worker._has_mamba = True, False
    worker._conv_decomp = None
    worker._ssm_region_indices = []
    worker._ple_region_index = None
    worker._ple_group_index = None
    worker.dcp_size = 1
    worker.dcp_rank = 0
    spec = MLAAttentionSpec(
        block_size=64, num_kv_heads=1, head_size=8, dtype=torch.bfloat16
    )
    groups = [
        KVCacheGroupSpec(["resident"], spec, enable_kv_transfer=False),
        KVCacheGroupSpec(["host"], spec),
        KVCacheGroupSpec(["indexer"], spec),
    ]
    for group in groups:
        group.num_blocks = 100
        group.blocks_per_page = 1
        group.kernel_rows_per_block = 1
        group.row_bytes = 1024
    worker.kv_cache_config = KVCacheConfig(
        num_pages=100,
        kv_cache_tensors=[],
        kv_cache_groups=groups,
        large_page_bytes=1024,
        block_size=64,
    )
    worker._group_spec_types = (MLAAttentionSpec, MLAAttentionSpec)
    worker.region_group_ids = [0, 1]
    worker.dst_region_group_ids = {"P": [0, 0]}
    worker._dst_group_geometries = {"P": [GroupGeometry(1, 1024)] * 2}
    worker.num_regions = 2
    worker.block_len_per_layer = [1024, 1024]
    worker.transfer_topo = MagicMock()
    worker.transfer_topo.tp_ratio.return_value = 1
    worker.transfer_topo.get_engine_info.return_value = SimpleNamespace(
        remote_tp_size=4,
        remote_dcp_size=1,
        remote_block_size=64,
    )
    worker.tp_mappings = {"P": TPMapping(((0,), (0,)), (0,), {0: 0}, 0)}
    worker._transfer_layer_group_ids = ()
    worker._mixed_mem_types = True
    worker.src_xfer_handle = 1
    worker._dram_src_handle = 2
    worker.dst_xfer_side_handles = {"P": {0: 3}}
    worker.dst_region_num_blocks = {"P": [100, 100], "D": [100, 100]}
    worker._remote_agents = {"P": {(0, 0): "P-rank0"}}
    worker._engine_by_address = {("localhost", 1): "P"}
    worker._read_blocks_mixed = MagicMock()  # type: ignore[method-assign]
    worker.nixl_wrapper = MagicMock()
    return worker


@pytest.mark.cpu_test
@pytest.mark.parametrize("num_pages", [0, 3, 19])
@pytest.mark.parametrize("region_groups", [(0, 1), (1, 0, 1)])
def test_dcp_region_pull(region_pull_worker, num_pages, region_groups):
    """Read each uncached page from its DCP rank; notify ranks with no pages."""
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
        RemoteMeta,
        ReqMeta,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.tp_mapping import TPMapping

    worker = region_pull_worker
    worker.region_group_ids = list(region_groups)
    worker.num_regions = len(region_groups)
    worker.dst_region_group_ids = {"P": [0] * worker.num_regions}
    worker.dst_region_num_blocks = {
        engine: [100] * worker.num_regions for engine in ("P", "D")
    }
    worker.block_len_per_layer = [1024] * worker.num_regions
    worker.dcp_rank = 0
    remote = worker.transfer_topo.get_engine_info.return_value
    remote.remote_tp_size = remote.remote_dcp_size = 8
    ranks = tuple(range(8))
    worker.tp_mappings["P"] = TPMapping((ranks, ranks), ranks, {r: r for r in ranks}, 8)
    worker.dst_xfer_side_handles = {"P": {r: 1000 + r for r in ranks}}
    worker._remote_agents = {"P": {(0, r): f"P-rank{r}" for r in ranks}}
    # Two regions with different prefix hits, plus one padding page each.
    local = (
        [list(range(31, 31 + num_pages)), list(range(42, 41 + num_pages))]
        if num_pages
        else []
    )
    meta = ReqMeta(
        local,
        local,
        tp_size=8,
        local_num_computed_blocks=(0, 1, 2),
        remote=RemoteMeta(
            [[10, 11, 12], []],
            "localhost",
            1,
            "P",
            "request-P",
            num_tokens=max(0, num_pages * 64 - 7),
        ),
    )
    worker._read_blocks_for_req("request", meta)
    reads = {
        call.kwargs["remote_xfer_side_handle"] - 1000: call.kwargs
        for call in worker._read_blocks_mixed.call_args_list
    }
    notified = {call.args[0] for call in worker.nixl_wrapper.send_notif.call_args_list}
    for rank in ranks:
        expected = [
            (region * 100 + 30 + group * 10 + page, region * 100 + 10 + page // 8)
            for region, group in enumerate(region_groups)
            for page in range(group + 1, num_pages)
            if page % 8 == rank
        ]
        if expected:
            read = reads[rank]
            assert (
                list(zip(read["local_block_descs_ids"], read["remote_block_descs_ids"]))
                == expected
            )
            assert f"P-rank{rank}" not in notified
        else:
            assert rank not in reads and f"P-rank{rank}" in notified
    assert meta.region_blocks_to_zero == (
        [[30 + group * 10 + num_pages] for group in region_groups]
        if num_pages
        else None
    )


@pytest.mark.cpu_test
def test_region_pull_completion_zeros_only_own_padding(region_pull_worker):
    """Completion must preserve neighboring layers and already imported pages."""
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
        RemoteMeta,
        ReqMeta,
    )

    worker = region_pull_worker
    # Two strided layer regions in one backing, with nonzero layer offsets.
    backing = torch.full((4, 3, 8), 7, dtype=torch.uint8)
    caches = {"host": backing[:, 1], "indexer": backing[:, 2]}
    worker.device_kv_caches = caches
    worker.region_names = list(caches)
    worker.kv_caches_base_addr = {"D": {0: [c.data_ptr() for c in caches.values()]}}
    worker.region_num_blocks = [4, 4]
    worker.block_len_per_layer = [8, 8]
    worker.block_stride_per_layer = [24, 24]
    worker._skip_dram_xfer = False
    worker._recving_metadata = {
        "request": ReqMeta(
            local_block_ids=[[1, 2], [0, 1]],
            local_physical_block_ids=[[1, 2], [0, 1]],
            tp_size=1,
            remote=RemoteMeta([[3]], "localhost", 1, "P", "request-P"),
            region_blocks_to_zero=[[2], [1]],
        )
    }
    worker._get_new_notifs = MagicMock(return_value=set())
    worker._pop_done_transfers = MagicMock(return_value=({"request"}, set()))
    worker._replicated_pcp_done_sending = set()
    worker._failed_recv_reqs = queue.Queue()
    worker._recv_failures = set()
    worker._send_pending_recv_notifs = MagicMock()
    worker._sync_device_after_direct_recv = MagicMock()
    worker.use_host_buffer = False
    worker._reqs_to_send = {}
    result = worker.get_transfer_results()
    expected = torch.full_like(backing, 7)
    expected[2, 1] = 0
    expected[1, 2] = 0
    assert torch.equal(backing, expected)
    assert result.finished_recving == {"request"}


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "swa_enabled,expected_sw_sizes",
    [
        # SWA enabled: FullAttentionSpec (0) + SlidingWindowSpec (2048/16=128)
        (True, [0, 128 + 1]),
        # SWA disabled: only FullAttentionSpec (0)
        (False, [0]),
    ],
)
@patch(
    "vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_scheduler.current_platform"
)
def test_sw_sizes(mock_platform, swa_enabled, expected_sw_sizes):
    """Test sw_sizes is correctly computed based on SWA enabled/disabled."""
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.scheduler import (
        NixlConnectorScheduler,
    )

    mock_platform.device_type = "cpu"

    block_size = 16
    vllm_config = create_vllm_config(block_size=block_size)
    # SW 2048 tokens=>128 blocks
    kv_cache_config = make_kv_cache_config(
        block_size=block_size, swa_enabled=swa_enabled, sw_size=2048
    )

    scheduler = NixlConnectorScheduler(
        vllm_config=vllm_config,
        engine_id="test-engine",
        kv_cache_config=kv_cache_config,
    )
    # in number of blocks
    assert scheduler.blocks_per_sw == expected_sw_sizes, (
        f"Expected sw_sizes={expected_sw_sizes}, got {scheduler.blocks_per_sw}"
    )


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "use_mla,source_ranks,tp_ratio,expected",
    [
        pytest.param(True, (0,), -2, False, id="pure_mla_with_remote_dcp"),
        pytest.param(True, (0, 1), -2, True, id="hybrid_mla_ssm"),
        pytest.param(False, (0,), -2, True, id="full_attention"),
        pytest.param(False, (0,), 2, False, id="local_tp_greater"),
    ],
)
def test_needs_split_local_xfer_handles(use_mla, source_ranks, tp_ratio, expected):
    """Handle creation and selection must use the same TP-mapping predicate.

    In pure MLA, DCP can target several physical remote workers even though
    the TP mapping has one replicated attention source. Those workers own
    disjoint blocks, so every read uses the whole local-region handle.
    """
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.worker import (
        NixlConnectorWorker,
    )

    worker = object.__new__(NixlConnectorWorker)
    worker.use_mla = use_mla
    plan = MagicMock()
    plan.all_source_ranks = source_ranks

    assert worker._needs_split_local_xfer_handles(tp_ratio, plan) is expected


@pytest.mark.cpu_test
def test_update_state_after_alloc_tracks_cached_blocks_per_group():
    """Hybrid SWA+FA requests can have different prefix-cache-hit counts per
    KV cache group. Each group's count must be tracked independently rather
    than collapsed to a single scalar, or DCP read-phase alignment
    (local_num_computed_blocks) would misalign one of the groups."""
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_scheduler import (
        NixlPullConnectorScheduler,
    )
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.kv_cache_utils import KVCacheBlock

    scheduler = object.__new__(NixlPullConnectorScheduler)
    scheduler._reqs_in_batch = set()
    scheduler._reqs_need_save = {}
    scheduler._reqs_need_recv = {}
    scheduler.use_host_buffer = False
    scheduler.is_bidirectional_kv_xfer_enabled = False
    scheduler._is_hma_required = False
    scheduler.kv_cache_config = MagicMock(
        select_transfer_block_ids=lambda block_ids: block_ids
    )

    def cached(block_id):
        return KVCacheBlock(block_id=block_id, _block_hash=object())

    def uncached(block_id):
        return KVCacheBlock(block_id=block_id)

    # Group 0 (full-attention): 2 of 3 blocks cached.
    # Group 1 (sliding-window): 1 of 3 blocks cached.
    blocks = KVCacheBlocks(
        blocks=(
            [cached(0), cached(1), uncached(2)],
            [cached(10), uncached(11), uncached(12)],
        )
    )

    request = create_request(do_remote_prefill=True)

    scheduler.update_state_after_alloc(request, blocks, num_external_tokens=2)

    _, _, local_num_computed_blocks, awaiting_kvs = scheduler._reqs_need_recv[
        request.request_id
    ]
    assert local_num_computed_blocks == (2, 1)
    # There are external tokens to pull, so the scheduler parks the request and
    # is waiting to see it in finished_recving.
    assert awaiting_kvs is True


@pytest.mark.cpu_test
def test_full_local_hit_is_not_awaited_by_the_scheduler():
    """A recv with num_external_tokens == 0 must not be flagged as awaited.

    The request keeps running -- there is nothing to pull, only a notification
    owed to the producer -- so the worker must not report it in
    finished_recving. Doing so trips `assert RequestStatus.is_finished` in
    _update_from_kv_xfer_finished, which only tolerates a finished recv for a
    request that is parked or already done."""
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_scheduler import (
        NixlPullConnectorScheduler,
    )
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.kv_cache_utils import KVCacheBlock

    scheduler = object.__new__(NixlPullConnectorScheduler)
    scheduler._reqs_in_batch = set()
    scheduler._reqs_need_save = {}
    scheduler._reqs_need_recv = {}
    scheduler.use_host_buffer = False
    scheduler.is_bidirectional_kv_xfer_enabled = False
    scheduler._is_hma_required = False
    scheduler.kv_cache_config = MagicMock(
        select_transfer_block_ids=lambda block_ids: block_ids
    )

    blocks = KVCacheBlocks(blocks=([KVCacheBlock(block_id=0, _block_hash=object())],))
    request = create_request(do_remote_prefill=True)

    scheduler.update_state_after_alloc(request, blocks, num_external_tokens=0)

    _, local_block_ids, _, awaiting_kvs = scheduler._reqs_need_recv[request.request_id]
    assert not local_block_ids  # full local hit: nothing to pull
    assert awaiting_kvs is False


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "local_ids,remote_ids,remote_rank,local_dcp_size,local_dcp_rank,"
    "remote_dcp_size,local_num_computed_blocks,expected_local,expected_remote",
    [
        # local more finely sharded than remote: remote's raw slice has an
        # extra tail block beyond what local's phase-aligned slice covers.
        pytest.param(
            [100, 101, 102],
            [200, 201],
            1,
            1,
            0,
            2,
            0,
            [101],
            [200],
            id="remote_tail_padding",
        ),
        # remote more finely sharded than local: local's raw slice has a
        # block beyond what remote's phase-aligned slice covers.
        pytest.param(
            [100],
            [200],
            0,
            2,
            1,
            1,
            0,
            [],
            [],
            id="local_tail_padding",
        ),
        # Asymmetric DCP with a partial prefix cache hit: the cached count
        # shifts both the local phase and the remote start position.
        pytest.param(
            [100],
            [200, 201],
            2,
            2,
            0,
            4,
            3,
            [100],
            [201],
            id="prefix_phase",
        ),
        # Equal (but active) DCP sizes still need to skip the locally cached
        # prefix from the front of remote's list.
        pytest.param(
            [100, 101],
            [200, 201, 202, 203],
            1,
            2,
            1,
            2,
            2,
            [100, 101],
            [202, 203],
            id="equal_dcp_with_prefix",
        ),
    ],
)
def test_apply_dcp_prefix_caching_matches_global_logical_positions(
    local_ids,
    remote_ids,
    remote_rank,
    local_dcp_size,
    local_dcp_rank,
    remote_dcp_size,
    local_num_computed_blocks,
    expected_local,
    expected_remote,
):
    """_apply_dcp_prefix_caching must operate on logical block IDs (DCP
    interleaves at logical-block granularity) and always return equal-length
    local/remote slices, so a later, unconditional _apply_prefix_caching
    trim is a guaranteed no-op rather than double-processing the read."""
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.worker import (
        NixlConnectorWorker,
    )

    worker = object.__new__(NixlConnectorWorker)

    actual_local, actual_remote = worker._apply_dcp_prefix_caching(
        local_ids,
        remote_ids,
        remote_rank=remote_rank,
        local_dcp_size=local_dcp_size,
        local_dcp_rank=local_dcp_rank,
        remote_dcp_size=remote_dcp_size,
        local_num_computed_blocks=local_num_computed_blocks,
    )

    assert actual_local == expected_local
    assert actual_remote == expected_remote
    assert len(actual_local) == len(actual_remote)


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "is_rocm,has_mamba,use_host_buffer,done_recving,expected_syncs",
    [
        (True, True, False, {"req"}, 1),
        (False, False, False, {"req"}, 0),
        (False, True, False, {"req"}, 0),
        (True, True, True, {"req"}, 0),
        (True, True, False, set(), 0),
    ],
)
def test_sync_device_after_direct_recv_gates(
    monkeypatch,
    is_rocm,
    has_mamba,
    use_host_buffer,
    done_recving,
    expected_syncs,
):
    """Only backends requiring visibility fences synchronize direct receives."""
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl import base_worker
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.worker import (
        NixlConnectorWorker,
    )

    worker = object.__new__(NixlConnectorWorker)
    worker._has_mamba = has_mamba
    worker.use_host_buffer = use_host_buffer

    sync_calls = []
    monkeypatch.setattr(base_worker.current_platform, "is_rocm", lambda: is_rocm)
    monkeypatch.setattr(
        base_worker.torch.accelerator,
        "synchronize",
        lambda: sync_calls.append(True),
    )

    worker._sync_device_after_direct_recv(done_recving)

    assert len(sync_calls) == expected_syncs


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "local_block_ids,remote_block_ids,expected_local,expected_remote",
    [
        # The local side reserved more trailing kernel blocks (e.g. the
        # recomputed final token); trim both to the shared prefix.
        pytest.param(
            [list(range(12)), [42]],
            [list(range(10)), [42]],
            [list(range(10)), [42]],
            [list(range(10)), [42]],
            id="trim_longer_local",
        ),
        # Prefix hit: the local side holds only the uncomputed suffix;
        # pair it with the remote's trailing blocks.
        pytest.param(
            [list(range(10)), [42]],
            [list(range(12)), [42]],
            [list(range(10)), [42]],
            [list(range(2, 12)), [42]],
            id="prefix_hit_tail_trim",
        ),
    ],
)
def test_apply_prefix_caching_mamba_hybrid(
    local_block_ids,
    remote_block_ids,
    expected_local,
    expected_remote,
):
    """_apply_prefix_caching trims FA groups to the shared prefix for
    Mamba hybrid models when the two sides reserved different tails.
    """
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.worker import (
        NixlConnectorWorker,
    )
    from vllm.v1.kv_cache_interface import MambaSpec

    worker = object.__new__(NixlConnectorWorker)
    worker._has_mamba = True
    worker._group_spec_types = (FullAttentionSpec, MambaSpec)
    worker.kv_cache_config = make_kv_cache_config(block_size=16, mamba_enabled=True)

    aligned_local, aligned_remote = worker._apply_prefix_caching(
        local_block_ids,
        remote_block_ids,
    )

    assert aligned_local == expected_local, (
        f"Expected local {expected_local}, got {aligned_local}"
    )
    assert aligned_remote == expected_remote, (
        f"Expected remote {expected_remote}, got {aligned_remote}"
    )


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "local_block_ids,remote_block_ids,expected_local,expected_remote",
    [
        # SSM prefix caching: remote has 3 placeholder + 1 real block,
        # local has only the 1 real block. FA blocks are equal (no trim).
        pytest.param(
            [list(range(10)), [42]],
            [list(range(10)), [40, 41, 42, 43]],
            [list(range(10)), [42]],
            [list(range(10)), [43]],
            id="ssm_prefix_trim_only",
        ),
        # FA partial prefix cache hit with homogeneous TP: local has 4 FA
        # blocks (prefix cached), remote has full 10. SSM equal (no trim).
        pytest.param(
            [list(range(6, 10)), [42]],
            [list(range(10)), [42]],
            [list(range(6, 10)), [42]],
            [list(range(6, 10)), [42]],
            id="fa_prefix_hit_homo_tp",
        ),
        # Both: FA partial prefix hit + SSM placeholder trim.
        # local FA=[6..9] (4 blocks, prefix cached), remote FA=[0..9]
        # local SSM=[99], remote SSM=[10, 20, 99] (2 placeholders + real)
        pytest.param(
            [[6, 7, 8, 9], [99]],
            [list(range(10)), [10, 20, 99]],
            [[6, 7, 8, 9], [99]],
            [[6, 7, 8, 9], [99]],
            id="fa_prefix_hit_and_ssm_trim",
        ),
    ],
)
def test_apply_prefix_caching_ssm_prefix_cache_hit(
    local_block_ids,
    remote_block_ids,
    expected_local,
    expected_remote,
):
    """_apply_prefix_caching end-trims SSM remote blocks to match the single
    local block (placeholders dropped) and end-trims FA remote blocks on
    partial prefix cache hits.
    """
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.worker import (
        NixlConnectorWorker,
    )
    from vllm.v1.kv_cache_interface import MambaSpec

    worker = object.__new__(NixlConnectorWorker)
    worker._has_mamba = True
    worker._group_spec_types = (FullAttentionSpec, MambaSpec)
    worker.kv_cache_config = make_kv_cache_config(block_size=16, mamba_enabled=True)

    aligned_local, aligned_remote = worker._apply_prefix_caching(
        local_block_ids,
        remote_block_ids,
    )

    assert aligned_local == expected_local, (
        f"Expected local {expected_local}, got {aligned_local}"
    )
    assert aligned_remote == expected_remote, (
        f"Expected remote {expected_remote}, got {aligned_remote}"
    )


@pytest.mark.cpu_test
def test_apply_prefix_caching_ssm_unpairable_slots_rejected():
    """Local SSM slots can only exceed the remote ones by the position D
    recomputes itself. A larger excess means the lists aren't
    position-aligned: fail loudly rather than transfer into wrong slots."""
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.worker import (
        NixlConnectorWorker,
    )
    from vllm.v1.kv_cache_interface import MambaSpec

    worker = object.__new__(NixlConnectorWorker)
    worker._has_mamba = True
    worker._group_spec_types = (FullAttentionSpec, MambaSpec)
    worker.kv_cache_config = make_kv_cache_config(block_size=16, mamba_enabled=True)

    with pytest.raises(AssertionError, match="unpairable SSM state slots"):
        worker._apply_prefix_caching(
            [list(range(10)), [4, 5, 6, 7]], [list(range(10)), [8, 9]]
        )


@pytest.mark.parametrize("model_name, sw_size", [("google/gemma-3-1b-it", 512)])
def test_fewer_blocks_with_hma(monkeypatch, model_name, sw_size):
    """Test that a prefill instance returns fewer "remote blocks" for the SWA groups
    when sequence exceeds the sliding window.
    """
    kv_transfer_config = KVTransferConfig(
        kv_connector="NixlConnector",
        kv_role="kv_consumer",
    )
    block_size = maybe_update_block_size(16)
    llm_kwargs = {
        "model": model_name,
        "enforce_eager": True,
        "gpu_memory_utilization": 0.3,
        "kv_transfer_config": kv_transfer_config,
        "max_model_len": 2048,
        "max_num_seqs": 1,
        "max_num_batched_tokens": 2048,
        "enable_prefix_caching": False,
        "block_size": block_size,
    }

    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    def run_hma_test(llm: LLM):
        remote_prefill_opts = {
            "do_remote_decode": True,
            "do_remote_prefill": False,
            "remote_engine_id": None,
            "remote_block_ids": None,
            "remote_host": None,
            "remote_port": None,
        }
        # Simulate sidecar request
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=1,
            extra_args={"kv_transfer_params": remote_prefill_opts},
        )
        engine_core_client = cast(Any, llm.llm_engine.engine_core)
        engine_core = engine_core_client.engine_core
        scheduler = engine_core.scheduler
        kv_managers = scheduler.kv_cache_manager.coordinator.single_type_managers
        # HMA enabled with FA + SWA groups
        assert len(kv_managers) > 2
        for kv_manager in kv_managers:
            assert isinstance(kv_manager, (SlidingWindowManager, FullAttentionManager))
        req_to_blocks = kv_managers[0].req_to_blocks
        assert len(req_to_blocks) == 0

        # Process some request with length exceeding the sliding window
        outputs = llm.generate(["hi" * 1401], sampling_params)
        kv_params = outputs[0].kv_transfer_params
        assert kv_params is not None

        # +1 to account for overlapping window across blocks.
        expected_num_remote_blocks = sw_size // block_size + 1
        remote_block_ids = kv_params["remote_block_ids"]
        assert (
            len(remote_block_ids[0])
            == expected_num_remote_blocks
            < len(remote_block_ids[-1])
        )
        for group_block_ids in remote_block_ids[:-1]:
            assert len(group_block_ids) == expected_num_remote_blocks

    def run_test_and_cleanup():
        gc.collect()
        torch.accelerator.empty_cache()
        llm = LLM(**llm_kwargs)
        try:
            run_hma_test(llm)
        finally:
            llm.llm_engine.engine_core.shutdown()

    run_test_and_cleanup()


@pytest.mark.cpu_test
def test_nixl_metadata_hma_block_ids_structure():
    """Test that NixlConnectorMetadata correctly stores block IDs for multiple
    KV cache groups when HMA is enabled.
    """
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
        NixlConnectorMetadata,
    )

    metadata = NixlConnectorMetadata()

    # Add request with block IDs for 2 groups (FA + SW)
    fa_blocks = [0, 1, 2, 3, 4, 5, 6, 7]  # 8 blocks for FA
    sw_blocks = [8, 9, 10, 11]  # 4 blocks for SW (clipped)

    metadata.add_new_req_to_recv(
        request_id="test-req-hma",
        local_block_ids=(fa_blocks, sw_blocks),
        kv_transfer_params={
            "remote_block_ids": ([10, 11, 12, 13, 14, 15, 16, 17], [18, 19, 20, 21]),
            "remote_engine_id": "remote-engine",
            "remote_request_id": "prefill-test-req-hma",
            "remote_host": "localhost",
            "remote_port": 1234,
            "tp_size": 1,
        },
    )

    assert "test-req-hma" in metadata.reqs_to_recv
    req_meta = metadata.reqs_to_recv["test-req-hma"]

    # Verify local block IDs structure
    assert len(req_meta.local_block_ids) == 2
    assert list(req_meta.local_block_ids[0]) == fa_blocks
    assert list(req_meta.local_block_ids[1]) == sw_blocks

    # Verify remote block IDs structure
    assert req_meta.remote is not None
    assert len(req_meta.remote.block_ids) == 2
    assert list(req_meta.remote.block_ids[0]) == [10, 11, 12, 13, 14, 15, 16, 17]
    assert list(req_meta.remote.block_ids[1]) == [18, 19, 20, 21]


def _make_mock_worker_for_desc_ids(has_mamba: bool, group_spec_types: tuple):
    """Build a mock NixlConnectorWorker with attrs needed by _compute_desc_ids."""
    from unittest.mock import MagicMock

    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.worker import (
        NixlConnectorWorker,
    )

    worker = MagicMock(spec=NixlConnectorWorker)
    worker._transfer_layer_group_ids = ()
    worker._has_mamba = has_mamba
    worker._group_spec_types = group_spec_types
    worker._conv_decomp = None
    worker._ssm_region_indices = []
    worker._ple_group_index = None
    worker._ple_region_index = None
    if has_mamba:
        from vllm.distributed.kv_transfer.kv_connector.v1.ssm_conv_transfer_utils import (  # noqa: E501
            MambaConvSplitInfo,
        )

        # Mamba2/GDN layout: 3 conv sub-projections -> 4 desc runs per layer.
        worker._conv_decomp = MambaConvSplitInfo(
            conv_rows=3,
            local_proj_dims=(1, 1, 1),
            conv_dtype_size=2,
            ssm_sizes=(0, 0),
        )
    worker._compute_desc_ids = NixlConnectorWorker._compute_desc_ids.__get__(
        worker, NixlConnectorWorker
    )
    return worker


@pytest.mark.cpu_test
def test_nixl_region_sort_preserves_pipeline_layer_order():
    """Numeric layer order prevents PP peers from pairing different regions."""
    names = [
        "model.layers.10.self_attn",
        "model.layers.2.self_attn",
        "model.layers.1.self_attn",
    ]

    assert sorted(names, key=bw._region_sort_key) == [
        "model.layers.1.self_attn",
        "model.layers.2.self_attn",
        "model.layers.10.self_attn",
    ]


@pytest.mark.cpu_test
def test_get_block_descs_ids_hybrid_ssm():
    """SSM desc ids follow the FA descs, one run per conv/ssm sub-descriptor."""
    from vllm.v1.kv_cache_interface import MambaSpec

    worker = _make_mock_worker_for_desc_ids(True, (FullAttentionSpec, MambaSpec))
    worker._ssm_region_indices = [1]

    result = worker._compute_desc_ids(
        block_ids=([3, 5], [1, 2]),
        region_num_blocks=[100, 100],
        region_group_ids=[0, 1],
    )

    # FA region 0 at offset 0; SSM region 1 at offset 100, with 4
    # sub-descriptor runs of 100 each.
    assert list(result) == [3, 5, 101, 102, 201, 202, 301, 302, 401, 402]


@pytest.mark.cpu_test
def test_get_block_descs_ids_selects_attention_regions_by_group():
    """Each attention group's blocks address only that group's regions."""
    worker = _make_mock_worker_for_desc_ids(
        False, (FullAttentionSpec, FullAttentionSpec)
    )

    result = worker._compute_desc_ids(
        block_ids=([1, 2], [7]),
        region_num_blocks=[10, 10, 10],
        region_group_ids=[0, 0, 1],
    )

    assert result.tolist() == [1, 2, 11, 12, 27]

    remapped = worker._compute_desc_ids(
        block_ids=([1, 2], [7]),
        region_num_blocks=[10, 10, 10],
        region_group_ids=[1, 1, 0],
    )
    assert remapped.tolist() == [21, 22, 7, 17]


@pytest.mark.cpu_test
def test_get_block_descs_ids_uses_per_region_pool_capacity():
    """Independent host and device pools use cumulative descriptor offsets."""
    worker = _make_mock_worker_for_desc_ids(
        False, (FullAttentionSpec, FullAttentionSpec)
    )

    result = worker._compute_desc_ids(
        block_ids=([4], [8]),
        region_num_blocks=[5, 10],
        region_group_ids=[0, 1],
    )

    assert result.tolist() == [4, 13]


def _bind_worker_method(worker, name):
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.worker import (
        NixlConnectorWorker,
    )

    method = getattr(NixlConnectorWorker, name)
    setattr(worker, name, method.__get__(worker, NixlConnectorWorker))


@pytest.mark.cpu_test
def test_nixl_metadata_hybrid_ssm_block_ids():
    """Test NixlConnectorMetadata correctly stores block IDs for FA + SSM
    groups with different block counts (kernel mismatch active)."""
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
        NixlConnectorMetadata,
    )

    metadata = NixlConnectorMetadata()

    # FA: 8 kernel blocks (2 logical * ratio=4), SSM: 2 logical blocks
    fa_blocks = [0, 1, 2, 3, 4, 5, 6, 7]
    ssm_blocks = [0, 1]

    metadata.add_new_req_to_recv(
        request_id="test-req-hybrid",
        local_block_ids=(fa_blocks, ssm_blocks),
        kv_transfer_params={
            "remote_block_ids": ([10, 11, 12, 13, 14, 15, 16, 17], [20, 21]),
            "remote_engine_id": "remote-engine",
            "remote_request_id": "prefill-test-req-hybrid",
            "remote_host": "localhost",
            "remote_port": 1234,
            "tp_size": 1,
        },
    )

    assert "test-req-hybrid" in metadata.reqs_to_recv
    req_meta = metadata.reqs_to_recv["test-req-hybrid"]

    # Verify local block IDs: different lengths per group
    assert len(req_meta.local_block_ids) == 2
    assert list(req_meta.local_block_ids[0]) == fa_blocks
    assert list(req_meta.local_block_ids[1]) == ssm_blocks
    assert len(req_meta.local_block_ids[0]) != len(req_meta.local_block_ids[1])

    # Verify remote block IDs: same asymmetry preserved
    assert req_meta.remote is not None
    assert len(req_meta.remote.block_ids) == 2
    assert list(req_meta.remote.block_ids[0]) == [10, 11, 12, 13, 14, 15, 16, 17]
    assert list(req_meta.remote.block_ids[1]) == [20, 21]
    assert len(req_meta.remote.block_ids[0]) != len(req_meta.remote.block_ids[1])


class _FakeBlock:
    def __init__(self, block_id):
        self.block_id = block_id


class _FakeSingleTypeManager:
    def __init__(self, kv_cache_group_id, records, block_size, block_ids):
        self.kv_cache_group_id = kv_cache_group_id
        self.records_new_block_ids = records
        self.block_size = block_size
        self.req_to_blocks = {"req-1": [_FakeBlock(b) for b in block_ids]}
        self.new_block_ids: list[int] = []

    def take_new_block_ids(self):
        ids = self.new_block_ids
        self.new_block_ids = []
        return ids


def _make_fake_kv_cache_manager():
    from unittest.mock import MagicMock

    from vllm.v1.core.kv_cache_manager import KVCacheManager

    manager = object.__new__(KVCacheManager)
    manager.coordinator = MagicMock()
    manager.kv_cache_config = MagicMock()
    manager.kv_cache_config.kv_cache_groups = (
        MagicMock(host_resident=False),
        MagicMock(host_resident=False),
    )
    manager.coordinator.single_type_managers = (
        _FakeSingleTypeManager(0, True, 16, [10, 11, 12, 13, 14, 15]),  # attention
        _FakeSingleTypeManager(1, False, 16, [20, 21, 22, 23, 24, 25]),  # mamba
    )
    return manager


@pytest.mark.cpu_test
def test_failed_load_rezeroes_unwritten_skipped_blocks():
    """A failed async load leaves zeroing-skipped blocks unwritten beyond
    the valid prefix; they must be zeroed before local recompute."""
    from unittest.mock import MagicMock

    from vllm.v1.core.sched.scheduler import Scheduler

    scheduler = object.__new__(Scheduler)
    scheduler.connector = MagicMock()
    scheduler.needs_kv_cache_zeroing = True
    scheduler.kv_cache_manager = _make_fake_kv_cache_manager()
    scheduler.kv_cache_manager.cache_blocks = MagicMock()  # type: ignore[method-assign]
    scheduler.failed_recving_kv_req_ids = {"req-1"}
    scheduler.finished_recving_kv_req_ids = {"req-1"}
    scheduler.prefix_replay_tokens = 0

    request = MagicMock()
    request.request_id = "req-1"
    request.num_computed_tokens = 48  # Truncated at the first invalid block.

    scheduler._update_waiting_for_remote_kv(request)

    # Attention blocks covering tokens >= 48 are re-recorded for zeroing
    # and flow into the next step's zero list; Mamba blocks are not.
    assert scheduler._get_new_block_ids_to_zero() == [[13, 14, 15], []]


# ── Mamba N-1 prefill tests ──────────────────────────────────────────────


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "has_mamba,is_hma_required,bounded_replay,expected_count",
    [
        (True, True, False, 9),
        (False, False, False, 10),
        (False, True, False, 10),
        (False, True, True, 9),
    ],
    ids=["mamba", "fa_only", "swa_only", "swa_bounded_replay"],
)
def test_mamba_n1_d_side(has_mamba, is_hma_required, bounded_replay, expected_count):
    """D-side: Mamba and SWA bounded replay get N-1 matched tokens, others N."""
    sched = make_nixl_scheduler(
        has_mamba=has_mamba,
        is_hma_required=is_hma_required,
        bounded_replay=bounded_replay,
    )
    req = create_request(num_tokens=10, do_remote_prefill=True)

    count, is_async = sched.get_num_new_matched_tokens(req, num_computed_tokens=0)
    assert count == expected_count
    assert is_async is True


@pytest.mark.cpu_test
def test_mamba_n1_d_side_builds_decode_metadata():
    req = create_request(num_tokens=10, do_remote_prefill=True)
    sched = make_nixl_scheduler(has_mamba=True, is_hma_required=True)

    num_computed_tokens, is_async = sched.get_num_new_matched_tokens(
        req, num_computed_tokens=0
    )

    assert num_computed_tokens == req.num_prompt_tokens - 1
    assert is_async is True

    vllm_config = create_vllm_config()
    metadata = MockMambaBuilder.build_mamba_metadata(
        vllm_config,
        seq_lens=[req.num_prompt_tokens],
        query_lens=[1],
        is_prefilling=[True],
    )

    assert metadata.num_decodes == 1
    assert metadata.num_prefills == 0


@pytest.mark.cpu_test
def test_mamba_n1_p_side_truncation():
    """P-side truncates to N-1 before the scheduler's prefix-cache lookup.

    Also verifies idempotency (calling again is a no-op) which is
    needed for preemption safety via the _p_side_truncated guard,
    and that non-Mamba models skip truncation entirely.
    """
    sched = make_nixl_scheduler(has_mamba=True, is_hma_required=True)
    req = create_request(num_tokens=10, do_remote_decode=True)
    req.max_tokens = 128
    assert req.prompt_token_ids is not None
    original_len = len(req.prompt_token_ids)

    sched.on_new_request(req)
    assert len(req.prompt_token_ids) == original_len - 1
    assert req.num_prompt_tokens == original_len - 1
    assert req.max_tokens == 1
    assert req.kv_transfer_params is not None
    assert req.kv_transfer_params["_p_side_truncated"] is True

    count, is_async = sched.get_num_new_matched_tokens(req, num_computed_tokens=0)

    assert count == 0
    assert is_async is False
    assert len(req.prompt_token_ids) == original_len - 1

    # Idempotency: re-adding a preempted request must not truncate further.
    sched.on_new_request(req)
    assert len(req.prompt_token_ids) == original_len - 1

    # Non-Mamba: truncation is skipped
    fa_sched = make_nixl_scheduler(has_mamba=False, is_hma_required=False)
    fa_req = create_request(num_tokens=10, do_remote_decode=True)
    assert fa_req.prompt_token_ids is not None
    fa_original = len(fa_req.prompt_token_ids)

    fa_sched.on_new_request(fa_req)
    assert len(fa_req.prompt_token_ids) == fa_original

    # SWA bounded replay: the window is not replayed after the load, so the
    # prefiller stops short of the last token too.
    swa_sched = make_nixl_scheduler(is_hma_required=True, bounded_replay=True)
    swa_req = create_request(num_tokens=10, do_remote_decode=True)
    swa_sched.on_new_request(swa_req)
    assert len(swa_req.prompt_token_ids) == 9

    # A request that skips reading the prefix cache (prompt logprobs) is not
    # cut: the decoder loads nothing and recomputes its whole prompt.
    logprobs_req = create_request(num_tokens=10, do_remote_decode=True)
    assert logprobs_req.sampling_params is not None
    logprobs_req.sampling_params.skip_reading_prefix_cache = True
    swa_sched.on_new_request(logprobs_req)
    assert len(logprobs_req.prompt_token_ids) == 10


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "swa_enabled,mamba_enabled,expected_has_mamba,expected_is_hma",
    [
        (True, True, True, True),
        (True, False, False, True),
        (False, False, False, False),
    ],
    ids=["fa_swa_mamba", "fa_swa_only", "fa_only"],
)
@patch(
    "vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_scheduler.current_platform"
)
def test_has_mamba_init(
    mock_platform,
    swa_enabled,
    mamba_enabled,
    expected_has_mamba,
    expected_is_hma,
):
    """Test _has_mamba / _is_hma_required derived from kv_cache_groups."""
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.scheduler import (
        NixlConnectorScheduler,
    )

    mock_platform.device_type = "cpu"

    block_size = 16
    vllm_config = create_vllm_config(block_size=block_size)
    # Explicitly enable HMA so we can test the scheduler's own derivation.
    vllm_config.scheduler_config.disable_hybrid_kv_cache_manager = False
    kv_cache_config = make_kv_cache_config(
        block_size=block_size,
        swa_enabled=swa_enabled,
        mamba_enabled=mamba_enabled,
    )

    scheduler = NixlConnectorScheduler(
        vllm_config=vllm_config,
        engine_id="test-engine",
        kv_cache_config=kv_cache_config,
    )
    assert scheduler._has_mamba is expected_has_mamba
    assert scheduler._is_hma_required is expected_is_hma


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "mamba_type,local_tp,conv_dim_local,conv_rows,temporal_shape,expected_proj_dims",
    [
        # nvidia/Nemotron-H-8B-Base-8K (Mamba2)
        # mamba_num_heads=128, head_dim=64, n_groups=8, ssm_state_size=128
        pytest.param(
            "mamba2",
            1,
            10240,
            3,
            (128, 64, 128),
            (8192, 1024, 1024),
            id="nemotron_h_8b_tp1",
        ),
        pytest.param(
            "mamba2",
            4,
            2560,
            3,
            (32, 64, 128),
            (2048, 256, 256),
            id="nemotron_h_8b_tp4",
        ),
        # nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B (Mamba2)
        # mamba_num_heads=64, head_dim=64, n_groups=8, ssm_state_size=128
        pytest.param(
            "mamba2",
            1,
            6144,
            3,
            (64, 64, 128),
            (4096, 1024, 1024),
            id="nemotron_nano_30b_tp1",
        ),
        # Qwen/Qwen3.5-0.8B (GDN, symmetric: num_v=num_k=16)
        # key_dim=2048, value_dim=2048, conv_dim=6144
        pytest.param(
            "gdn_attention",
            1,
            6144,
            3,
            (16, 128, 128),
            (2048, 2048, 2048),
            id="qwen35_08b_tp1",
        ),
        pytest.param(
            "gdn_attention",
            4,
            1536,
            3,
            (4, 128, 128),
            (512, 512, 512),
            id="qwen35_08b_tp4",
        ),
        # Qwen/Qwen3.5-4B (GDN, asymmetric: num_v=32, num_k=16, K:V=1:2)
        # key_dim=2048, value_dim=4096, conv_dim=8192
        pytest.param(
            "gdn_attention",
            1,
            8192,
            3,
            (32, 128, 128),
            (2048, 2048, 4096),
            id="qwen35_4b_tp1",
        ),
        # Qwen/Qwen3.5-27B (GDN, asymmetric: num_v=48, num_k=16, K:V=1:3)
        # key_dim=2048, value_dim=6144, conv_dim=10240
        pytest.param(
            "gdn_attention",
            1,
            10240,
            3,
            (48, 128, 128),
            (2048, 2048, 6144),
            id="qwen35_27b_tp1",
        ),
        pytest.param(
            "gdn_attention",
            8,
            1280,
            3,
            (6, 128, 128),
            (256, 256, 768),
            id="qwen35_27b_tp8",
        ),
        # ai21labs/AI21-Jamba2-Mini (Mamba1)
        # mamba d_inner = mamba_expand(2) * hidden_size(4096) = 8192
        # mamba_d_state=16, mamba_d_conv=4 → conv_rows=3.
        # Conv state holds only x: a single contiguous sub-projection.
        pytest.param(
            "mamba1",
            1,
            8192,
            3,
            (8192, 16),
            (8192,),
            id="jamba_mini_tp1",
        ),
        pytest.param(
            "mamba1",
            4,
            2048,
            3,
            (2048, 16),
            (2048,),
            id="jamba_mini_tp4",
        ),
        pytest.param(
            "mamba1",
            8,
            1024,
            3,
            (1024, 16),
            (1024,),
            id="jamba_mini_tp8",
        ),
    ],
)
def test_derive_mamba_conv_split(
    monkeypatch,
    mamba_type,
    local_tp,
    conv_dim_local,
    conv_rows,
    temporal_shape,
    expected_proj_dims,
):
    """Parametrized test for derive_mamba_conv_split with real model configs.

    Values generated by verify_conv_split.py which loads HuggingFace configs
    and calls vLLM's derive_mamba_conv_split directly.
    """
    from vllm.distributed.kv_transfer.kv_connector.v1.ssm_conv_transfer_utils import (
        derive_mamba_conv_split,
    )
    from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
    from vllm.v1.kv_cache_interface import MambaSpec

    _TYPE_MAP = {
        "mamba1": MambaAttentionBackendEnum.MAMBA1,
        "mamba2": MambaAttentionBackendEnum.MAMBA2,
        "gdn_attention": MambaAttentionBackendEnum.GDN_ATTN,
    }
    mamba_type_enum = _TYPE_MAP[mamba_type]

    monkeypatch.setenv("VLLM_SSM_CONV_STATE_LAYOUT", "DS")
    spec = MambaSpec(
        block_size=64,
        shapes=((conv_dim_local, conv_rows), temporal_shape),
        dtypes=(torch.bfloat16, torch.bfloat16),
        mamba_type=mamba_type_enum,
    )
    out = derive_mamba_conv_split(spec, local_tp=local_tp)
    assert out.local_proj_dims == expected_proj_dims
    assert out.conv_rows == conv_rows


@pytest.mark.cpu_test
def test_exchange_clipped_blocks_ssm_single_state():
    """SSM lists are reduced to the running state slot: speculative scratch
    slots, null placeholders and the previous step's state carry nothing.
    Attention groups pass through untouched."""
    sched = make_nixl_scheduler(has_mamba=True, is_hma_required=True)
    sched.blocks_per_sw = [0, 0]
    sched._ssm_spec_blocks = [None, 2]

    # Align-mode list: null placeholders, state block, 2 speculative slots.
    clipped = sched.get_exchange_clipped_blocks(([1, 2, 3], [0, 0, 7, 8, 9]))
    assert clipped == ([1, 2, 3], [7])

    # Same, still holding the previous step's state block (freed a step later).
    assert sched.get_exchange_clipped_blocks(([1], [0, 6, 7, 8, 9]))[1] == [7]

    # Default (mamba_block_size=max_model_len): state block, 2 scratch slots.
    assert sched.get_exchange_clipped_blocks(([1], [7, 8, 9]))[1] == [7]

    # Scratch slots not allocated: the state slot still survives.
    assert sched.get_exchange_clipped_blocks(([1], [5]))[1] == [5]

    assert sched.get_exchange_clipped_blocks(([1], [])) == ([1], [])
    # Per-step partial lists must not be interpreted as state + scratch slots.
    assert sched.get_exchange_clipped_blocks(([1], [7, 8]), clip_ssm=False) == (
        [1],
        [7, 8],
    )
    sched.blocks_per_sw = [2, 0]
    assert sched.get_exchange_clipped_blocks(([1, 2, 3], [7, 8, 9])) == ([2, 3], [7])

    # Non-mamba models pass through unchanged.
    fa_sched = make_nixl_scheduler(has_mamba=False)
    assert fa_sched.get_exchange_clipped_blocks(([1, 2],)) == ([1, 2],)


@pytest.mark.cpu_test
def test_push_write_hybrid_mla_replicates_attention():
    """Hybrid MLA+SSM push with P_TP < D_TP: attention blocks must be
    written to every covered D rank (replicated MLA latent) while SSM state
    is written per-rank through the split handles."""
    import threading
    from unittest.mock import MagicMock

    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.push_worker import (
        NixlPushConnectorWorker,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.tp_mapping import (
        TPMapping,
    )
    from vllm.v1.kv_cache_interface import MambaSpec, MLAAttentionSpec

    worker = object.__new__(NixlPushConnectorWorker)
    worker.shutdown = lambda: None  # type: ignore[method-assign]  # silence __del__
    worker.use_mla = True
    worker._has_mamba = True
    worker._is_csa_linear = False
    worker._group_spec_types = (MLAAttentionSpec, MambaSpec)
    worker.transfer_topo = MagicMock()
    worker.transfer_topo.tp_ratio.return_value = -2
    remote_info = MagicMock()
    remote_info.remote_block_size = 4
    worker.transfer_topo.get_engine_info.return_value = remote_info

    engine_id = "remote-engine"
    # Read-oriented mapping collapses the replicated attention group to one
    # source rank; the SSM state is sharded across both covered D ranks.
    worker.tp_mappings = {
        engine_id: TPMapping(
            source_ranks_per_group=((0,), (0, 1)),
            all_source_ranks=(0, 1),
            rank_to_attention_slot={0: 0, 1: 0},
            rank_offset_factor=0,
        )
    }
    worker.dst_xfer_side_handles = defaultdict(dict, {engine_id: {0: 100, 1: 101}})
    worker.src_xfer_handles_by_tp_ratio = {-2: [200, 201]}
    worker.src_xfer_handle = 300
    worker.region_group_ids = [0, 1]
    worker.dst_region_group_ids = {engine_id: [0, 1]}
    worker._sending_transfers = defaultdict(list)
    worker._sending_transfers_lock = threading.Lock()
    worker.kv_cache_config = make_lp_kv_cache_config(block_size=4, mamba_enabled=True)
    # The peer shares this worker's geometry; identity kernel mapping.
    worker._dst_group_geometries = {
        engine_id: worker.kv_cache_config.transfer_group_geometries
    }
    worker._xfer_blocks = MagicMock(return_value=1)  # type: ignore[method-assign]

    meta = MagicMock()
    meta.remote.engine_id = engine_id
    meta.remote.block_ids = [[7, 8], [3]]
    meta.local_physical_block_ids = [[1, 2], [5]]

    worker._xfer_blocks_for_req("req-1", meta)

    calls = worker._xfer_blocks.call_args_list
    assert len(calls) == 2
    for call, rank, local_handle, remote_handle in zip(
        calls, (0, 1), (200, 201), (100, 101)
    ):
        spec = call.kwargs["read_spec"]
        assert spec.remote_rank == rank
        # Attention group replicated to every rank, SSM by membership.
        assert spec.local_block_ids == [[1, 2], [5]]
        assert spec.remote_block_ids == [[7, 8], [3]]
        assert call.kwargs["local_xfer_side_handle"] == local_handle
        assert call.kwargs["remote_xfer_side_handle"] == remote_handle


def _make_host_buffer_worker(copy_op):
    """A worker stripped down to what the host-buffer copy paths touch."""
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.worker import (
        NixlConnectorWorker,
    )

    worker = object.__new__(NixlConnectorWorker)
    worker.use_host_buffer = True
    worker.copy_blocks = copy_op
    return worker


@pytest.mark.cpu_test
def test_sync_recved_kv_issues_one_copy_per_group():
    """h2d copies run per KV cache group, each with that group's own block
    ids (unique only within a group) and only that group's layers."""
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import ReqMeta
    from vllm.v1.kv_cache_interface import (
        KVCacheConfig,
        KVCacheGroupSpec,
        SlidingWindowSpec,
    )

    calls = []
    worker = _make_host_buffer_worker(
        lambda src, dst, src_ids, dst_ids, direction: calls.append(
            (sorted(src), src_ids, dst_ids, direction)
        )
    )
    spec = FullAttentionSpec(
        block_size=16, num_kv_heads=1, head_size=1, dtype=torch.float32
    )
    swa_spec = SlidingWindowSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        sliding_window=64,
    )
    worker.kv_cache_config = KVCacheConfig(
        num_pages=10,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["fa.0", "fa.1"], spec),
            KVCacheGroupSpec(["swa.0"], swa_spec),
        ],
        block_size=16,
    )
    worker.host_xfer_buffers = {"fa.0": None, "fa.1": None, "swa.0": None}
    worker.device_kv_caches = dict(worker.host_xfer_buffers)

    meta = ReqMeta(
        local_block_ids=([1, 2], [3]),
        local_physical_block_ids=([1, 2], [3]),
        tp_size=1,
    )
    worker.sync_recved_kv_to_device("req", meta)

    assert calls == [
        (["fa.0", "fa.1"], [1, 2], [1, 2], "h2d"),
        (["swa.0"], [3], [3], "h2d"),
    ]
