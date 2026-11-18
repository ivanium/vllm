# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mooncake store objects with a drafter KV pool (`VLLM_DRAFT_KV_POOL`): each
cache group's object is the same bytes in the same order as without the pool,
read from the group's own pool, so keys, object sizes (the size-class split's
input) and peers without the pool are unaffected."""

from types import SimpleNamespace

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake_store.layout import (
    KeyMetadata,
    RankLocalStoreLayout,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake_store.worker import (
    MooncakeStoreWorker,
)
from vllm.v1.core import kv_cache_planning as planning
from vllm.v1.kv_cache_layout import KVCacheLayout
from vllm.v1.worker.utils import allocate_kv_cache

from .test_draft_kv_pool_nixl import BLOCK, DRAFT_LAYERS, TARGET_LAYERS, _groups
from .utils import create_vllm_config

pytestmark = pytest.mark.cpu_test

NUM_BLOCKS, POOL_BLOCKS = 6, 12


def _register(pool: bool, monkeypatch):
    """Register both groups as the store worker does; returns the layouts,
    the caches and the config."""
    config = create_vllm_config(block_size=BLOCK, kv_role="kv_both")
    config.cache_config.kv_cache_layout = "BLHNC"
    groups = _groups(pool=pool)
    if pool:
        monkeypatch.setenv(planning.DRAFT_KV_POOL_NUM_BLOCKS_ENV, str(POOL_BLOCKS))
        kv_cache_config = (
            planning.DefaultKVCacheConfigBuilder().get_kv_cache_config_from_groups(
                config, groups, NUM_BLOCKS
            )
        )
    else:
        kv_cache_config = planning.get_kv_cache_config_from_groups(
            config, groups, NUM_BLOCKS * planning._pool_bytes_per_block(config, groups)
        )
    caches = allocate_kv_cache(
        kv_cache_config, torch.device("cpu"), KVCacheLayout.BLHNC
    )
    worker = SimpleNamespace(
        _kv_cache_config=kv_cache_config, cache_config=config.cache_config
    )
    layouts = []
    for group_id, group in enumerate(kv_cache_config.kv_cache_groups):
        layout = RankLocalStoreLayout(
            KeyMetadata("m", 0, 0, 0, 0, group_id=group_id), BLOCK, BLOCK
        )
        tensors = [caches[name] for name in group.layer_names]
        if pool:
            MooncakeStoreWorker._register_pool_group_kv_caches(
                worker,  # type: ignore[arg-type]
                group,
                SimpleNamespace(store_layout=layout),
                tensors,
            )
        else:
            # The worker's path without private pools.
            layout.register_kv_caches(tensors, kv_cache_config.num_blocks)
        layouts.append(layout)
    return layouts, caches, kv_cache_config


def _object(layout, block_id: int, base: int) -> list[tuple[int, int]]:
    """(offset from `base`, length) of each buffer of one block's object."""
    addrs, sizes, _ = layout.prepare_values([(0, BLOCK)], [block_id], [0])
    return [(a - base, n) for a, n in zip(addrs[0], sizes[0])]


@pytest.mark.parametrize("pool", [False, True], ids=["one-pool", "drafter-pool"])
def test_each_object_is_one_buffer_of_the_groups_pages(pool, monkeypatch):
    layouts, caches, kv = _register(pool, monkeypatch)
    base = caches[TARGET_LAYERS[0]].untyped_storage().data_ptr()
    target_row, draft_row = 2 * 256 + 128, 2 * 192
    target, drafter = layouts
    for b in (0, 1, 5):
        # The whole target row, as without the pool.
        assert _object(target, b, base) == [(b * target_row, target_row)]
    pool_base = NUM_BLOCKS * target_row if pool else 0
    stride = draft_row if pool else target_row
    for b in (0, 1, 5):
        # The drafter's pages of the block: 384 bytes from its row's start.
        assert _object(drafter, b, base) == [(pool_base + b * stride, draft_row)]
        first_page = caches[DRAFT_LAYERS[0]][b].data_ptr() - base
        assert first_page == pool_base + b * stride
    if pool:
        # Blocks past the shared pool's count exist only in the drafter's pool.
        b = POOL_BLOCKS - 1
        assert _object(drafter, b, base) == [(pool_base + b * stride, draft_row)]
        assert pool_base + (b + 1) * stride == kv.kv_cache_tensors[0].size


def test_object_bytes_do_not_change_with_the_pool(monkeypatch):
    """The size-class split sizes objects by group; both groups' sizes stay."""
    sizes = {}
    for pool in (False, True):
        layouts, _, _ = _register(pool, monkeypatch)
        sizes[pool] = [
            MooncakeStoreWorker._store_object_bytes(
                SimpleNamespace(store_layout=layout, chunk_size=BLOCK)
            )
            for layout in layouts
        ]
    assert sizes[False] == sizes[True] == [{640}, {384}]


def test_store_tp_layouts_are_refused(monkeypatch):
    _, caches, kv = _register(True, monkeypatch)
    worker = SimpleNamespace(_kv_cache_config=kv, cache_config=None)
    with pytest.raises(NotImplementedError, match="rank-local store layout"):
        MooncakeStoreWorker._register_pool_group_kv_caches(
            worker,  # type: ignore[arg-type]
            kv.kv_cache_groups[1],
            SimpleNamespace(store_layout=object()),
            [caches[DRAFT_LAYERS[0]]],
        )
