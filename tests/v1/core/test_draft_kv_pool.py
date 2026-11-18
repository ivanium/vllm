# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A DFlash/DSpark drafter's sliding-window KV in its own block pool
(`VLLM_DRAFT_KV_POOL`, vllm-project/vllm#64611): planning, layout, allocation
and accounting per pool, and the admission checks of each pool."""

import math
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from tests.v1.core.test_prefix_caching import make_kv_cache_manager, make_request
from tests.v1.core.utils import new_kv_cache_group
from tests.v1.spec_decode.utils import fake_speculative_config
from vllm.config import KVTransferConfig, ModelConfig, VllmConfig
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.utils.math_utils import cdiv
from vllm.v1.core import kv_cache_planning as planning
from vllm.v1.core.kv_cache_utils import (
    generate_scheduler_kv_cache_config,
    get_max_concurrency_for_kv_cache_config,
    init_none_hash,
)
from vllm.v1.core.single_type_kv_cache_manager import (
    DFlashSlidingWindowManager,
    FullAttentionManager,
)
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheConfig
from vllm.v1.request import Request
from vllm.v1.worker.utils import allocate_kv_cache

pytestmark = pytest.mark.cpu_test

# MiniMax-M3.1 NVFP4 page sizes at block 128, as GB300 serving logs report them.
MSA = AttentionSpec(
    block_size=128, num_kv_heads=1, head_size=288, dtype=torch.uint8
)  # 73,728 B
INDEXER = AttentionSpec(
    block_size=128, num_kv_heads=1, head_size=72, head_size_v=0, dtype=torch.uint8
)  # 9,216 B
DRAFTER = AttentionSpec(
    block_size=128, num_kv_heads=4, head_size=128, dtype=torch.float8_e4m3fn
)  # 131,072 B
WINDOW = 4096


def _layers(num_target: int = 60, num_draft: int = 5):
    layers = {}
    for i in range(num_target):
        layers[f"model.layers.{i}.attn"] = FullAttentionManager.layer_config(MSA)
        layers[f"model.layers.{i}.indexer"] = FullAttentionManager.layer_config(INDEXER)
    for i in range(num_draft):
        layers[f"draft.layers.{i}.attn"] = replace(
            DFlashSlidingWindowManager.layer_config(DRAFTER, sliding_window=WINDOW),
            is_draft=True,
        )
    return layers


def _config(
    *,
    max_num_seqs: int = 24,
    max_num_batched_tokens: int = 8192,
    max_num_inflight_kv_loads: int | None = 64,
    kv_transfer_config: KVTransferConfig | None = None,
    disable_hybrid: bool = False,
) -> VllmConfig:
    config = VllmConfig(model_config=ModelConfig(max_model_len=2048))
    # The real recipes serve 1M tokens; the model config caps the length, so set
    # it after validation (only the planner's arithmetic reads it here).
    config.model_config.max_model_len = 1 << 20
    config.cache_config.kv_cache_layout = "BLHNC"
    config.scheduler_config.max_num_seqs = max_num_seqs
    config.scheduler_config.max_num_batched_tokens = max_num_batched_tokens
    config.scheduler_config.max_num_inflight_kv_loads = max_num_inflight_kv_loads
    config.scheduler_config.disable_hybrid_kv_cache_manager = disable_hybrid
    config.speculative_config = fake_speculative_config(
        method="dspark",
        num_speculative_tokens=7,
        use_dflash=lambda: False,
        use_dspark=lambda: True,
        use_eagle=lambda: False,
        use_eagle_block_drop=lambda: False,
        use_multi_module_mtp=lambda: False,
        draft_model_config=SimpleNamespace(architectures=[]),
    )
    config.kv_transfer_config = kv_transfer_config
    return config


def _plan(config: VllmConfig, available_memory: int) -> KVCacheConfig:
    layers = _layers()
    groups = planning.get_global_kv_cache_groups(config, [layers])
    (worker_config,) = planning.get_kv_cache_configs(
        config, [layers], [available_memory], groups
    )
    return worker_config


TARGET_BLOCK = 60 * 73_728 + 60 * 9_216  # 4,976,640 B
DRAFT_BLOCK = 5 * 131_072  # 655,360 B
D_BUDGET = 131_266_019_328  # a decode instance's kv-cache-memory-bytes


@pytest.fixture
def pool_on(monkeypatch):
    monkeypatch.setenv(planning.DRAFT_KV_POOL_ENV, "1")
    monkeypatch.delenv(planning.DRAFT_KV_POOL_NUM_BLOCKS_ENV, raising=False)
    monkeypatch.delenv(planning.DRAFT_KV_POOL_CACHE_FRACTION_ENV, raising=False)


def test_switch_off_keeps_one_pool(monkeypatch):
    monkeypatch.delenv(planning.DRAFT_KV_POOL_ENV, raising=False)
    config = _plan(_config(), D_BUDGET)
    assert not config.pool_num_blocks and not config.pool_offsets
    assert [g.pool_id for g in config.kv_cache_groups] == [0, 0]
    # The drafter's pages sit inside the target-sized blocks.
    assert {t.block_stride for t in config.kv_cache_tensors} == {TARGET_BLOCK}
    assert config.num_blocks == D_BUDGET // TARGET_BLOCK
    assert all(t.pool_id == 0 for t in config.kv_cache_tensors)


def test_pool_groups_keep_ids_and_layers(monkeypatch):
    monkeypatch.delenv(planning.DRAFT_KV_POOL_ENV, raising=False)
    off = planning.get_global_kv_cache_groups(_config(), [_layers()])
    monkeypatch.setenv(planning.DRAFT_KV_POOL_ENV, "1")
    on = planning.get_global_kv_cache_groups(_config(), [_layers()])
    # Group 0 (target) and group 1 (drafter) keep their layers, order and
    # manager: Mooncake store keys (@group:N) and NIXL region pairing hold.
    assert [g.layer_names for g in on] == [g.layer_names for g in off]
    assert [g.manager_cls for g in on] == [g.manager_cls for g in off]
    assert [g.pool_id for g in on] == [0, 1]
    assert on[1].manager_cls is DFlashSlidingWindowManager


def test_pool_layout_and_size(pool_on):
    config = _config()
    kv = _plan(config, D_BUDGET)
    draft_group = kv.kv_cache_groups[1]
    cap = draft_group.max_allocated_blocks_per_request(config)
    # Async scheduling on D (2 batches of 8192 tokens) makes it 161 blocks.
    assert cap == cdiv(WINDOW - 1 + config.max_in_flight_tokens, 128) + 1
    # Upstream's default: (max_num_seqs + inflight loads) x cap x 1.5 + null.
    pool_blocks = 1 + math.ceil((24 + 64) * cap * 1.5)
    assert kv.pool_num_blocks == {1: pool_blocks}
    pool_bytes = pool_blocks * DRAFT_BLOCK
    assert kv.num_blocks == (D_BUDGET - pool_bytes) // TARGET_BLOCK
    shared_bytes = kv.num_blocks * TARGET_BLOCK
    assert kv.pool_offsets == {1: shared_bytes}
    # One backing allocation: the shared pool, then the drafter's pool.
    assert {t.size for t in kv.kv_cache_tensors} == {shared_bytes + pool_bytes}
    shared = [t for t in kv.kv_cache_tensors if t.pool_id == 0]
    private = [t for t in kv.kv_cache_tensors if t.pool_id == 1]
    assert {t.block_stride for t in shared} == {TARGET_BLOCK}
    assert {t.block_stride for t in private} == {DRAFT_BLOCK}
    assert min(t.offset for t in private) == shared_bytes
    assert sum(len(t.layers) for t in private) == 5
    assert kv.num_blocks_of_group(0) == kv.num_blocks
    assert kv.num_blocks_of_group(1) == pool_blocks


def test_pool_size_override_and_cache_fraction(pool_on, monkeypatch):
    config = _config()
    monkeypatch.setenv(planning.DRAFT_KV_POOL_CACHE_FRACTION_ENV, "0")
    kv = _plan(config, D_BUDGET)
    cap = kv.kv_cache_groups[1].max_allocated_blocks_per_request(config)
    assert cap == cdiv(WINDOW - 1 + config.max_in_flight_tokens, 128) + 1
    assert kv.pool_num_blocks == {1: 1 + 88 * cap}
    monkeypatch.setenv(planning.DRAFT_KV_POOL_NUM_BLOCKS_ENV, "2993")
    kv = _plan(config, D_BUDGET)
    assert kv.pool_num_blocks == {1: 2993}
    assert kv.num_blocks == (D_BUDGET - 2993 * DRAFT_BLOCK) // TARGET_BLOCK
    # It must admit one request at the drafter's admission cap.
    monkeypatch.setenv(planning.DRAFT_KV_POOL_NUM_BLOCKS_ENV, str(cap))
    with pytest.raises(ValueError, match="cannot admit one request"):
        _plan(config, D_BUDGET)


def test_shared_pool_override_leaves_the_pool(pool_on, monkeypatch):
    # P's Mooncake hbm_pool "auto" keeps one request's blocks with
    # num_gpu_blocks_override; the drafter's pool keeps its own size.
    monkeypatch.setenv(planning.DRAFT_KV_POOL_NUM_BLOCKS_ENV, "2313")
    config = _config(max_num_seqs=64, max_num_batched_tokens=16384)
    config.cache_config.num_gpu_blocks_override = 8194
    kv = _plan(config, 177 * 2**30)
    assert kv.num_blocks == 8194
    assert kv.pool_num_blocks == {1: 2313}
    assert kv.pool_offsets == {1: 8194 * TARGET_BLOCK}
    assert kv.kv_cache_tensors[0].size == 8194 * TARGET_BLOCK + 2313 * DRAFT_BLOCK


def test_scheduler_config_and_concurrency(pool_on, monkeypatch):
    monkeypatch.setenv(planning.DRAFT_KV_POOL_NUM_BLOCKS_ENV, "2993")
    config = _config()
    kv = _plan(config, D_BUDGET)
    scheduler_kv = generate_scheduler_kv_cache_config([kv, kv])
    assert scheduler_kv.pool_num_blocks == {1: 2993}
    # Max concurrency: the shared pool's own limit (the target only), unless
    # the drafter's pool is the tighter one.
    shared_per_request = kv.kv_cache_groups[0].max_allocated_blocks_per_request(config)
    assert get_max_concurrency_for_kv_cache_config(config, kv) == pytest.approx(
        kv.num_blocks / shared_per_request
    )
    cap = kv.kv_cache_groups[1].max_allocated_blocks_per_request(config)
    tight = replace(kv, pool_num_blocks={1: 200})
    assert get_max_concurrency_for_kv_cache_config(config, tight) == pytest.approx(
        200 / cap
    )
    # Different pool sizes on two workers would split the scheduler's view.
    with pytest.raises(AssertionError):
        generate_scheduler_kv_cache_config([kv, tight])


@pytest.mark.parametrize(
    "case",
    ["no_spec", "no_hma", "connector", "consumer_without_cap"],
)
def test_refused_configs(pool_on, case):
    config = _config()
    if case == "no_spec":
        config.speculative_config = None
        match = "DFlash or DSpark"
    elif case == "no_hma":
        config.scheduler_config.disable_hybrid_kv_cache_manager = True
        match = "hybrid KV cache manager"
    elif case == "connector":
        config.kv_transfer_config = KVTransferConfig(
            kv_connector="LMCacheConnectorV1", kv_role="kv_both"
        )
        match = "NixlConnector and MooncakeStoreConnector"
    else:
        config.kv_transfer_config = _p_d_connectors()
        config.scheduler_config.max_num_inflight_kv_loads = None
        match = "max-num-inflight-kv-loads"
    with pytest.raises((ValueError, NotImplementedError), match=match):
        planning.get_global_kv_cache_groups(config, [_layers()])


def _p_d_connectors() -> KVTransferConfig:
    return KVTransferConfig(
        kv_connector="MultiConnector",
        kv_role="kv_both",
        kv_connector_extra_config={
            "connectors": [
                {"kv_connector": "NixlConnector", "kv_role": "kv_both"},
                {"kv_connector": "MooncakeStoreConnector", "kv_role": "kv_both"},
            ]
        },
    )


def test_nixl_and_mooncake_store_are_supported(pool_on, monkeypatch):
    # P has no inflight-load cap: the explicit size stands in for it.
    monkeypatch.setenv(planning.DRAFT_KV_POOL_NUM_BLOCKS_ENV, "2313")
    config = _config(kv_transfer_config=_p_d_connectors())
    config.scheduler_config.max_num_inflight_kv_loads = None
    groups = planning.get_global_kv_cache_groups(config, [_layers()])
    assert [g.pool_id for g in groups] == [0, 1]


def test_allocated_views_index_their_pool(pool_on, monkeypatch):
    """The worker's views: block b of a drafter layer starts at the pool's
    offset plus b drafter blocks, never inside the shared pool."""
    monkeypatch.setenv(planning.DRAFT_KV_POOL_NUM_BLOCKS_ENV, "7")
    layers = {
        "t.0": FullAttentionManager.layer_config(replace(MSA, block_size=4)),
        "t.0.idx": FullAttentionManager.layer_config(replace(INDEXER, block_size=4)),
        "d.0": replace(
            DFlashSlidingWindowManager.layer_config(
                replace(DRAFTER, block_size=4), sliding_window=8
            ),
            is_draft=True,
        ),
        "d.1": replace(
            DFlashSlidingWindowManager.layer_config(
                replace(DRAFTER, block_size=4), sliding_window=8
            ),
            is_draft=True,
        ),
    }
    config = _config()
    config.model_config.max_model_len = 64
    config.scheduler_config.max_num_batched_tokens = 8
    groups = planning.get_global_kv_cache_groups(config, [layers])
    target_block = 4 * 576 + 4 * 72
    draft_block = 2 * 4 * 4 * 256
    (kv,) = planning.get_kv_cache_configs(
        config, [layers], [21 * target_block + 7 * draft_block], groups
    )
    assert (kv.num_blocks, kv.pool_num_blocks) == (21, {1: 7})
    caches = allocate_kv_cache(
        kv, torch.device("cpu"), config.cache_config.get_resolved_kv_cache_layout()
    )
    base = caches["t.0"].untyped_storage().data_ptr()
    assert caches["t.0"].shape[0] == 21 and caches["d.0"].shape[0] == 7
    for name, page in (("d.0", 0), ("d.1", 1)):
        view = caches[name]
        assert view.untyped_storage().nbytes() == 21 * target_block + 7 * draft_block
        for b in range(7):
            start = view[b].data_ptr() - base
            assert start == 21 * target_block + b * draft_block + page * 4096
    for b in range(21):
        assert caches["t.0"][b].data_ptr() - base == b * target_block


# ---- Allocation and accounting per pool (KVCacheManager) ----

BLOCK = 16
DRAFT_WINDOW = 64


def _manager(num_blocks: int, pool_blocks: int | None, **kwargs):
    spec = AttentionSpec(
        block_size=BLOCK, num_kv_heads=1, head_size=1, dtype=torch.float32
    )
    target = new_kv_cache_group(["target"], spec)
    drafter = new_kv_cache_group(
        ["drafter"],
        DFlashSlidingWindowManager.layer_config(spec, sliding_window=DRAFT_WINDOW),
    )
    if pool_blocks is not None:
        drafter = replace(drafter, pool_id=1)
    config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[target, drafter],
        pool_num_blocks={} if pool_blocks is None else {1: pool_blocks},
    )
    kwargs.setdefault("max_model_len", 4096)
    kwargs.setdefault("max_in_flight_tokens", 64)
    kwargs.setdefault("enable_caching", True)
    kwargs.setdefault("hash_block_size", BLOCK)
    return make_kv_cache_manager(config, **kwargs)


def _request(request_id: str, num_tokens: int) -> Request:
    return Request(
        request_id=request_id,
        prompt_token_ids=list(range(num_tokens)),
        sampling_params=SamplingParams(max_tokens=4096),
        pooling_params=None,
    )


def _used(pool) -> int:
    return pool.num_gpu_blocks - 1 - pool.get_num_free_blocks()


def _decode(manager, request, num_steps: int) -> list[tuple[int, int]]:
    """Prefill `request` in one chunk, then decode `num_steps` tokens; the used
    blocks of each pool after every step."""
    assert manager.allocate_slots(request, request.num_tokens) is not None
    request.num_computed_tokens = request.num_tokens
    pools = manager.coordinator.block_pools
    used = []
    for _ in range(num_steps):
        request.append_output_token_ids(0)
        assert manager.allocate_slots(request, 1) is not None
        request.num_computed_tokens += 1
        used.append(tuple(_used(pool) for pool in pools))
    return used


def test_drafter_blocks_come_from_their_pool():
    manager = _manager(num_blocks=200, pool_blocks=40)
    shared, private = manager.coordinator.block_pools
    request = _request("a", 300)
    blocks = manager.allocate_slots(request, 300)
    assert blocks is not None
    target_blocks, drafter_blocks = blocks.blocks
    assert all(b.pool() is shared for b in target_blocks)
    assert all(b.pool() is private for b in drafter_blocks if not b.is_null)
    # A one-chunk prefill backs every drafter block until the window moves.
    assert _used(shared) == cdiv(300, BLOCK)
    assert _used(private) == cdiv(300, BLOCK)
    manager.free(request)
    assert _used(shared) == 0 and _used(private) == 0


@pytest.mark.parametrize("pool_blocks", [None, 40])
def test_window_and_padding_accounting(pool_blocks):
    """Decode holds the drafter's window, (W - 1 + in-flight) / B + 1 blocks at
    most. In one pool those are shared-pool blocks; with the drafter's pool the
    shared pool holds the target's blocks only."""
    manager = _manager(num_blocks=200, pool_blocks=pool_blocks)
    request = _request("a", 300)
    used = _decode(manager, request, 200)
    window_blocks = cdiv(DRAFT_WINDOW - 1, BLOCK) + 1
    for step, counts in enumerate(used, start=1):
        target = cdiv(300 + step, BLOCK)
        if pool_blocks is None:
            (shared,) = counts
            assert target < shared <= target + window_blocks
        else:
            shared, private = counts
            assert shared == target
            assert 0 < private <= window_blocks


def test_padding_math_with_model_shapes(monkeypatch):
    """A drafter block in the shared pool costs a 4,976,640 B target block for
    its 655,360 B of pages; decode's window of 34 blocks pads 29.5 blocks."""
    layout = planning.KVCacheLayout.BLHNC
    monkeypatch.delenv(planning.DRAFT_KV_POOL_ENV, raising=False)
    target, drafter = planning.get_global_kv_cache_groups(_config(), [_layers()])
    assert planning._get_kv_cache_bytes_per_block([target, drafter], layout) == (
        TARGET_BLOCK
    )
    assert planning._get_kv_cache_bytes_per_block([target], layout) == TARGET_BLOCK
    assert planning._get_kv_cache_bytes_per_block([drafter], layout) == DRAFT_BLOCK
    # Decode's window: 4,095 tokens plus 2 in-flight steps of 1 + 7 drafts
    # plus 7 lookahead slots.
    window_blocks = cdiv(WINDOW - 1 + 2 * 8 + 7, 128) + 1
    assert window_blocks == 34
    padding = window_blocks * (TARGET_BLOCK - DRAFT_BLOCK) / TARGET_BLOCK
    assert padding == pytest.approx(29.52, abs=0.01)


def test_private_pool_refuses_on_its_own():
    """The shared pool has room; the drafter's pool does not: admission waits
    (no block of either pool is taken) and the refusal is counted."""
    manager = _manager(num_blocks=500, pool_blocks=12)
    shared, private = manager.coordinator.block_pools
    first = _request("a", 160)  # 10 drafter blocks
    assert manager.allocate_slots(first, 160) is not None
    second = _request("b", 64)  # 4 more drafter blocks; 1 is free
    assert manager.allocate_slots(second, 64) is None
    assert _used(shared) == 10 and _used(private) == 10
    assert manager._num_private_pool_refusals == 1
    manager.free(first)
    assert manager.allocate_slots(second, 64) is not None


def test_full_sequence_admission_checks_each_pool():
    manager = _manager(num_blocks=500, pool_blocks=12)
    request = _request("a", 400)
    # The first chunk fits both pools; the full sequence's drafter cap
    # (cdiv(63 + 64, 16) + 1 = 9 blocks) still fits 11 free blocks.
    assert manager.allocate_slots(request, 32, full_sequence_must_fit=True) is not None
    manager.free(request)
    small = _manager(num_blocks=500, pool_blocks=8)
    assert (
        small.allocate_slots(_request("b", 400), 32, full_sequence_must_fit=True)
        is None
    )


def test_watermark_and_reservation_per_pool():
    manager = _manager(num_blocks=500, pool_blocks=21, watermark=0.25)
    assert manager.watermark_blocks_by_pool == [125, 5]
    request = _request("a", 160)  # 10 blocks in each pool; 20 free in the private
    no_watermark = {"has_scheduled_reqs": False}
    # Private pool: reserving 11 leaves 9 < 10.
    assert (
        manager.allocate_slots(request, 160, reserved_blocks=[0, 11], **no_watermark)
        is None
    )
    # A waiting request with others scheduled also keeps each pool's watermark
    # free: 20 - 6 = 14 < 10 + 5, 20 - 5 = 15.
    assert manager.allocate_slots(request, 160, reserved_blocks=[0, 6]) is None
    assert manager.allocate_slots(request, 160, reserved_blocks=[0, 5]) is not None
    manager.free(request)
    # An int reserves shared-pool blocks only: 499 - 489 = 10.
    assert (
        manager.allocate_slots(request, 160, reserved_blocks=490, **no_watermark)
        is None
    )
    assert (
        manager.allocate_slots(request, 160, reserved_blocks=489, **no_watermark)
        is not None
    )


def test_one_pool_matches_without_the_pool_switch():
    """Without private pools every count is the shared pool's, as before."""
    manager = _manager(num_blocks=200, pool_blocks=None)
    assert len(manager.coordinator.block_pools) == 1
    assert manager.watermark_blocks_by_pool == [manager.watermark_blocks]
    by_pool = manager.coordinator.get_num_blocks_to_allocate_by_pool(
        request_id="a",
        num_tokens=300,
        new_computed_blocks=([], []),
        num_encoder_tokens=0,
        total_computed_tokens=0,
        num_local_computed_tokens=0,
        num_tokens_main_model=300,
    )
    assert by_pool == [2 * cdiv(300, BLOCK)]


def test_prefix_cache_hit_reuses_both_pools():
    """A finished request's blocks stay cached, each in its own pool; the next
    turn hits the target prefix and the drafter's window before its end."""
    init_none_hash(sha256)
    manager = _manager(num_blocks=200, pool_blocks=40)
    shared, private = manager.coordinator.block_pools
    first = make_request("a", list(range(320)), BLOCK, sha256)
    first_blocks = manager.allocate_slots(first, 320)
    assert first_blocks is not None
    first.num_computed_tokens = 320
    manager.free(first)
    assert _used(shared) == 0 and _used(private) == 0

    second = make_request("b", list(range(320)) + [7] * 20, BLOCK, sha256)
    blocks, num_hit, _ = manager.get_computed_blocks(second)
    assert num_hit == 320
    target_hit, drafter_hit = blocks.blocks
    assert [b.block_id for b in target_hit] == [
        b.block_id for b in first_blocks.blocks[0]
    ]
    # The drafter needs its blocks in [340 - 63, 320) only: 17, 18 and 19.
    first_read = (340 - (DRAFT_WINDOW - 1)) // BLOCK
    assert all(b.is_null for b in drafter_hit[:first_read])
    assert [b.block_id for b in drafter_hit[first_read:]] == [
        b.block_id for b in first_blocks.blocks[1][first_read:20]
    ]
    assert all(b.pool() is private for b in drafter_hit[first_read:])
    assert all(b.pool() is shared for b in target_hit)
