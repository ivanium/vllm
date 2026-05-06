# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for MooncakeStoreConnector HMA support."""

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_store_scheduler import (  # noqa: E501
    MooncakeStoreScheduler,
)

from .utils import create_vllm_config, make_kv_cache_config


def _make_scheduler(
    swa_enabled: bool,
    sw_size: int = 2048,
    block_size: int = 16,
    disable_hma: bool = False,
):
    vllm_config = create_vllm_config(
        kv_connector="MooncakeStoreConnector",
        kv_role="kv_both",
        block_size=block_size,
    )
    vllm_config.scheduler_config.disable_hybrid_kv_cache_manager = disable_hma
    kv_cache_config = make_kv_cache_config(
        block_size=block_size,
        swa_enabled=swa_enabled,
        sw_size=sw_size,
    )
    return MooncakeStoreScheduler(vllm_config, kv_cache_config=kv_cache_config)


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "swa_enabled,expected_blocks_per_sw",
    [
        (True, [0, 2048 // 16 + 1]),  # FA=0, SWA=128+1=129
        (False, [0]),
    ],
)
def test_blocks_per_sw(swa_enabled, expected_blocks_per_sw):
    scheduler = _make_scheduler(swa_enabled=swa_enabled)
    assert scheduler.blocks_per_sw == expected_blocks_per_sw


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "swa_enabled,disable_hma,expected_is_hma",
    [
        (True, False, True),
        (True, True, False),
        (False, False, False),
    ],
)
def test_is_hma_required(swa_enabled, disable_hma, expected_is_hma):
    scheduler = _make_scheduler(swa_enabled=swa_enabled, disable_hma=disable_hma)
    assert scheduler._is_hma_required is expected_is_hma


@pytest.mark.cpu_test
def test_get_sw_clipped_blocks_clips_swa_keeps_fa():
    # sw_size=128 tokens / block_size=16 = 8 blocks + 1 = 9 blocks_per_sw
    scheduler = _make_scheduler(swa_enabled=True, sw_size=128)
    assert scheduler.blocks_per_sw == [0, 9]

    fa_blocks = list(range(20))
    sw_blocks = list(range(100, 120))
    clipped = scheduler.get_sw_clipped_blocks((fa_blocks, sw_blocks))

    assert clipped[0] == fa_blocks  # FA untouched
    assert clipped[1] == sw_blocks[-9:]  # SWA clipped to last 9
    assert len(clipped[1]) == 9


@pytest.mark.cpu_test
def test_get_sw_clipped_blocks_noop_when_not_hma():
    scheduler = _make_scheduler(swa_enabled=False)
    assert scheduler._is_hma_required is False
    block_ids = ([1, 2, 3],)
    assert scheduler.get_sw_clipped_blocks(block_ids) == [[1, 2, 3]]


# ---------------------------------------------------------------------------
# Task 2: per-group block_ids end-to-end
# ---------------------------------------------------------------------------
from unittest.mock import MagicMock  # noqa: E402

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_store_data import (  # noqa: E402,E501
    RequestTracker,
)


@pytest.mark.cpu_test
def test_request_tracker_update_extends_each_group():
    """update() must extend the matching group, not flatten."""
    tracker = RequestTracker(
        req_id="r1",
        token_len=0,
        allocated_block_ids=[[1, 2], [10, 11]],
    )
    tracker.update(([3, 4], [12, 13]))
    assert tracker.allocated_block_ids == [[1, 2, 3, 4], [10, 11, 12, 13]]


@pytest.mark.cpu_test
def test_request_tracker_update_accepts_flat_list_as_single_group():
    """Legacy flat list extends group 0 (single-group compat)."""
    tracker = RequestTracker(
        req_id="r1",
        token_len=0,
        allocated_block_ids=[[1, 2]],
    )
    tracker.update([3, 4])
    assert tracker.allocated_block_ids == [[1, 2, 3, 4]]


@pytest.mark.cpu_test
def test_scheduler_request_finished_clips_swa_group():
    """request_finished must clip SWA group on the way to delay-free state."""
    scheduler = _make_scheduler(swa_enabled=True, sw_size=128, block_size=16)
    fa_blocks = list(range(20))
    sw_blocks = list(range(100, 120))

    request = MagicMock()
    request.request_id = "r-finished"
    request.kv_transfer_params = {}
    scheduler._request_trackers["r-finished"] = RequestTracker(
        req_id="r-finished",
        token_len=20 * 16,
        allocated_block_ids=[fa_blocks, sw_blocks],
        num_saved_tokens=20 * 16,
    )

    delay, _ = scheduler.request_finished(request, (fa_blocks, sw_blocks))
    assert delay is True


# ---------------------------------------------------------------------------
# Task 3: ChunkedTokenDatabase group-aware prepare_value
# ---------------------------------------------------------------------------
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_store_data import (  # noqa: E402,E501
    ChunkedTokenDatabase,
    GroupLayout,
    KeyMetadata,
)


@pytest.mark.cpu_test
def test_prepare_value_picks_right_group_block_ids():
    """A layer in the SWA group must address through SWA's clipped block ids."""
    block_size = 16
    metadata = KeyMetadata(
        model_name="m",
        tp_rank=0,
        pcp_rank=0,
        dcp_rank=0,
        pp_rank=0,
    )
    db = ChunkedTokenDatabase(metadata, block_size=block_size)

    fa_layout = GroupLayout(base_addrs=[0x1000], block_lens=[256])
    sw_layout = GroupLayout(base_addrs=[0x2000], block_lens=[256])
    db.set_groups([fa_layout, sw_layout], layer_to_group=[0, 1])

    # 20 chunks total. SWA holds last 9.
    fa_block_ids = list(range(20))
    sw_block_ids = list(range(100, 109))
    block_ids_per_group = [fa_block_ids, sw_block_ids]

    # Chunk 19 (last): FA local index = 19; SWA local index = 19 - (20 - 9) = 8 → 108.
    fa_addr, _, _ = db.prepare_value(
        start=19 * block_size,
        end=20 * block_size,
        block_ids=block_ids_per_group,
        group_id=0,
        total_chunks=20,
    )
    sw_addr, _, _ = db.prepare_value(
        start=19 * block_size,
        end=20 * block_size,
        block_ids=block_ids_per_group,
        group_id=1,
        total_chunks=20,
    )
    assert fa_addr == [0x1000 + 19 * 256]
    assert sw_addr == [0x2000 + 108 * 256]


@pytest.mark.cpu_test
def test_prepare_value_single_group_unchanged():
    """N=1 (single full-attention group) must produce the same addresses as before."""
    block_size = 16
    metadata = KeyMetadata(
        model_name="m",
        tp_rank=0,
        pcp_rank=0,
        dcp_rank=0,
        pp_rank=0,
    )
    db = ChunkedTokenDatabase(metadata, block_size=block_size)

    layout = GroupLayout(base_addrs=[0x1000, 0x2000], block_lens=[256, 256])
    db.set_groups([layout], layer_to_group=[0, 0])

    block_ids_per_group = [[5, 6, 7, 8]]
    addr_list, _, _ = db.prepare_value(
        start=2 * block_size,
        end=3 * block_size,
        block_ids=block_ids_per_group,
        group_id=0,
        total_chunks=4,
    )
    assert addr_list == [0x1000 + 7 * 256, 0x2000 + 7 * 256]


@pytest.mark.cpu_test
def test_is_chunk_in_window_per_request_swa_offset():
    """SWA group's per-request window check uses (total_chunks - len) offset."""
    block_size = 16
    db = ChunkedTokenDatabase(KeyMetadata("m", 0, 0, 0, 0), block_size=block_size)
    db.set_groups(
        [
            GroupLayout(base_addrs=[0x1000], block_lens=[256]),
            GroupLayout(base_addrs=[0x2000], block_lens=[256]),
        ],
        layer_to_group=[0, 1],
    )

    fa = list(range(20))
    sw = list(range(100, 109))  # SWA holds last 9: covers chunks [11, 20)
    block_ids = [fa, sw]

    # FA group: every chunk is in window (len matches total_chunks).
    for cid in (0, 5, 19):
        assert db.is_chunk_in_window_per_request(cid, block_ids, 0, 20) is True

    # SWA group: only the last 9 chunks are in window.
    assert db.is_chunk_in_window_per_request(10, block_ids, 1, 20) is False
    assert db.is_chunk_in_window_per_request(11, block_ids, 1, 20) is True
    assert db.is_chunk_in_window_per_request(19, block_ids, 1, 20) is True
    assert db.is_chunk_in_window_per_request(20, block_ids, 1, 20) is False


# ---------------------------------------------------------------------------
# Task 4: worker register_kv_caches buckets layers by KV cache group
# ---------------------------------------------------------------------------
from unittest.mock import patch  # noqa: E402

import torch  # noqa: E402


@pytest.mark.cpu_test
def test_register_kv_caches_buckets_by_kv_cache_group():
    """Mixed FA + SWA layers must be split into two GroupLayouts."""
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake import (
        mooncake_store_worker,
    )

    block_size = 16
    vllm_config = create_vllm_config(
        kv_connector="MooncakeStoreConnector",
        kv_role="kv_both",
        block_size=block_size,
    )
    kv_cache_config = make_kv_cache_config(
        block_size=block_size,
        swa_enabled=True,
    )
    # FA group claims layer0 + layer2; SWA group claims layer1 + layer3.

    worker = mooncake_store_worker.MooncakeStoreWorker.__new__(
        mooncake_store_worker.MooncakeStoreWorker,
    )
    worker.kv_cache_config = kv_cache_config
    worker.cache_config = vllm_config.cache_config
    worker.cache_config.num_gpu_blocks = 4
    worker.num_blocks = 4
    worker.block_size = block_size
    worker.use_mla = False
    worker.tp_size = 1
    worker.tp_rank = 0
    worker.put_step = 1
    worker.kv_role = "kv_both"
    worker.metadata = mooncake_store_worker.KeyMetadata(
        model_name="m",
        tp_rank=0,
        pcp_rank=0,
        dcp_rank=0,
        pp_rank=0,
    )
    worker.token_database = mooncake_store_worker.ChunkedTokenDatabase(
        worker.metadata,
        block_size=block_size,
    )
    worker.store = MagicMock()
    worker.store.register_buffer = MagicMock(return_value=0)
    worker.kv_send_thread = None
    worker.kv_recv_thread = None
    worker.enable_kv_events = False
    worker.disk_offload_buffer_budget_bytes = None
    worker.replicate_config = None
    worker.preferred_segment = None
    worker.store_replicate_config = None

    # MLA-style blocks-first layout: (num_blocks, block_size, head_size).
    fa_tensor = torch.zeros((4, 16, 64), dtype=torch.float16)
    sw_tensor = torch.zeros((4, 16, 64), dtype=torch.float16)
    kv_caches = {
        "layer0": fa_tensor,
        "layer2": fa_tensor.clone(),
        "layer1": sw_tensor,
        "layer3": sw_tensor.clone(),
    }
    with (
        patch.object(mooncake_store_worker, "KVCacheStoreSendingThread"),
        patch.object(mooncake_store_worker, "KVCacheStoreRecvingThread"),
        patch.object(mooncake_store_worker.threading, "Event") as mock_event,
    ):
        # Avoid blocking on ready_event_recving.wait().
        mock_event.return_value.wait = MagicMock()
        worker.register_kv_caches(kv_caches)

    db = worker.token_database
    assert len(db.groups) == 2
    assert len(db.groups[0].base_addrs) == 2  # FA: 2 layers
    assert len(db.groups[1].base_addrs) == 2  # SWA: 2 layers
    assert db.layer_to_group == [0, 0, 1, 1]


# ---------------------------------------------------------------------------
# Task: DSV4-shaped per-group hashing (group_block_sizes=[256, 64, 64, 4, 8])
# ---------------------------------------------------------------------------


def _dsv4_db():
    """Build ChunkedTokenDatabase with DSV4-Flash group block sizes."""
    metadata = KeyMetadata(
        model_name="dsv4", tp_rank=0, pcp_rank=0, dcp_rank=0, pp_rank=0
    )
    db = ChunkedTokenDatabase(metadata, block_size=256)
    # 5 groups: FA (256), SWA×2 (64), C4 (4), C8 (8).
    db.set_groups(
        [
            GroupLayout(base_addrs=[0x1000 * (i + 1)], block_lens=[256])
            for i in range(5)
        ],
        layer_to_group=[0, 1, 2, 3, 4],
    )
    db.set_blocks_per_sw([0, 3, 3, 3, 17])
    db.set_group_block_sizes([256, 64, 64, 4, 8], hash_block_size=4)
    return db


@pytest.mark.cpu_test
def test_dsv4_process_tokens_emits_per_group_native_chunks():
    """Each group walks its own native block_size; key hash is right-edge."""
    db = _dsv4_db()
    token_len = 12_000  # divisible by all group block sizes
    n_hashes = token_len // 4
    hashes = [f"h{i:04d}" for i in range(n_hashes)]

    by_group: dict[int, list[tuple[int, int, str]]] = {g: [] for g in range(5)}
    for start, end, g, key in db.process_tokens(token_len, hashes):
        by_group[g].append((start, end, key.chunk_hash))

    # cdiv(12000, gbs) per group.
    assert [len(by_group[g]) for g in range(5)] == [47, 188, 188, 3000, 1500]

    # FA chunk 0 [0, 256) → hash_idx 63 → h0063.
    assert by_group[0][0] == (0, 256, "h0063")
    assert by_group[0][1] == (256, 512, "h0127")
    # FA last chunk [11776, 12000) → hash_idx 2999 → h2999.
    # Note: token_len=12000, FA ceil(12000/256)=47, last chunk is partial
    # (12000-11776=224 tokens) but end_idx=12000 is hash-aligned.
    assert by_group[0][46] == (11776, 12000, "h2999")
    # SWA group 1 last chunk [11968, 12000) → same hash index.
    assert by_group[1][187] == (11968, 12000, "h2999")
    # C4 last chunk [11996, 12000) → same hash index.
    assert by_group[3][2999] == (11996, 12000, "h2999")


@pytest.mark.cpu_test
def test_dsv4_window_clipping_yields_47_3_3_3_17_in_window_keys():
    """Save-side window: FA all + SWA tails + compressor tail = 73 keys."""
    db = _dsv4_db()
    token_len = 12_000
    hashes = [f"h{i:04d}" for i in range(token_len // 4)]
    block_ids = [
        list(range(47)),
        list(range(3)),
        list(range(3)),
        list(range(3)),
        list(range(17)),
    ]
    g_total = [47, 188, 188, 3000, 1500]

    in_window = []
    for start, _end, g, _key in db.process_tokens(token_len, hashes):
        chunk_id = start // db.group_block_sizes[g]
        if db.is_chunk_in_window_per_request(chunk_id, block_ids, g, g_total[g]):
            in_window.append(g)

    per_group_count = [in_window.count(k) for k in range(5)]
    assert per_group_count == [47, 3, 3, 3, 17]


@pytest.mark.cpu_test
def test_dsv4_lookup_window_static_uses_blocks_per_sw():
    """Lookup-side window check (no block_ids) agrees with save-side."""
    db = _dsv4_db()
    g_total = [47, 188, 188, 3000, 1500]
    in_window = sum(
        1
        for g in range(5)
        for c in range(g_total[g])
        if db.is_chunk_in_window(c, g_total[g], g)
    )
    assert in_window == 73


@pytest.mark.cpu_test
def test_dsv4_scheduler_block_size_lcm_not_min():
    """scheduler_block_size = LCM (256), not MIN (4)."""
    db = _dsv4_db()
    assert db.scheduler_block_size == 256
    assert db.hash_block_size == 4
    assert db.group_block_sizes == [256, 64, 64, 4, 8]


@pytest.mark.cpu_test
def test_set_group_block_sizes_explicit_scheduler_block_size():
    """Explicit scheduler_block_size overrides the LCM default."""
    db = ChunkedTokenDatabase(KeyMetadata("m", 0, 0, 0, 0), block_size=256)
    db.set_groups(
        [GroupLayout([0x1000], [256]), GroupLayout([0x2000], [256])],
        layer_to_group=[0, 1],
    )
    db.set_group_block_sizes([256, 64], hash_block_size=64, scheduler_block_size=512)
    assert db.scheduler_block_size == 512


@pytest.mark.cpu_test
def test_dsv4_partial_last_chunk_indexes_safely():
    """Last chunk of every group uses end_idx // hash_bs - 1 in range."""
    db = _dsv4_db()
    token_len = 12_000
    hashes = [f"h{i:04d}" for i in range(token_len // 4)]

    by_group_last: dict[int, tuple[int, int, str]] = {}
    for start, end, g, key in db.process_tokens(token_len, hashes):
        by_group_last[g] = (start, end, key.chunk_hash)

    for g, (start, end, h) in by_group_last.items():
        assert h == f"h{end // 4 - 1:04d}", f"group {g} chunk [{start}, {end}): {h}"
