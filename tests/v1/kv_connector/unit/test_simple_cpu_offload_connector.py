# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for SimpleCPUOffloadConnector-level behavior."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from tests.v1.kv_connector.unit.utils import create_vllm_config
from tests.v1.simple_kv_offload.test_scheduler import (
    _BYTES_PER_BLOCK,
    BLOCK_SIZE,
    DTYPE,
    HEAD_SIZE,
    NUM_KV_HEADS,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload_connector import (
    SimpleCPUOffloadConnector,
)
from vllm.v1.engine import core as engine_core_module
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    SlidingWindowSpec,
)


def _make_kv_cache_config(num_blocks: int = 16) -> KVCacheConfig:
    fa_layers = ["layer0"]
    spec = FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=NUM_KV_HEADS,
        head_size=HEAD_SIZE,
        dtype=DTYPE,
    )
    groups = [KVCacheGroupSpec(fa_layers, spec)]
    size = _BYTES_PER_BLOCK * num_blocks
    tensors = [
        KVCacheTensor(
            size=size,
            layers=fa_layers,
            layer_stride=size,
            block_stride=_BYTES_PER_BLOCK,
        )
    ]
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=tensors,
        kv_cache_groups=groups,
    )


def _make_connector(
    extra_config: dict | None = None,
    num_gpu_blocks: int = 16,
    cpu_bytes: int = _BYTES_PER_BLOCK * 8,
) -> SimpleCPUOffloadConnector:
    kv_cache_config = _make_kv_cache_config(num_gpu_blocks)
    config_dict = {
        "cpu_bytes_to_use": cpu_bytes,
    }
    if extra_config:
        config_dict.update(extra_config)
    vllm_config = create_vllm_config(
        kv_connector="SimpleCPUOffloadConnector",
        kv_role="kv_both",
        kv_connector_extra_config=config_dict,
    )
    connector = SimpleCPUOffloadConnector(
        vllm_config=vllm_config,
        role=KVConnectorRole.SCHEDULER,
        kv_cache_config=kv_cache_config,
    )
    return connector


# Two-group large_page geometry: group 0 packs 2 kernel rows of 1024 B per
# block and gets 1 block per page; group 1 has one 768 B row, which does not
# divide the 2048 B page, so some of its block ids cross a page end.
LP_PAGE_BYTES = 2048
LP_NUM_PAGES = 8
LP_SLOTS_PER_PAGE = (1, 2)
LP_ROWS_PER_BLOCK = (2, 1)
LP_ROW_BYTES = (1024, 768)


def _make_large_page_kv_cache_config() -> KVCacheConfig:
    fa = FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=NUM_KV_HEADS,
        head_size=HEAD_SIZE,
        dtype=DTYPE,
    )
    swa = SlidingWindowSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=NUM_KV_HEADS,
        head_size=HEAD_SIZE,
        dtype=DTYPE,
        sliding_window=BLOCK_SIZE * 4,
    )
    groups = [
        KVCacheGroupSpec(
            ["fa"],
            fa,
            num_blocks=8,  # 8 * 2048 B // 2048 B
            slots_per_large_page=LP_SLOTS_PER_PAGE[0],
            kernel_rows_per_block=LP_ROWS_PER_BLOCK[0],
            row_bytes=LP_ROW_BYTES[0],
        ),
        KVCacheGroupSpec(
            ["swa"],
            swa,
            num_blocks=21,  # 8 * 2048 B // 768 B
            slots_per_large_page=LP_SLOTS_PER_PAGE[1],
            kernel_rows_per_block=LP_ROWS_PER_BLOCK[1],
            row_bytes=LP_ROW_BYTES[1],
        ),
    ]
    return KVCacheConfig(
        num_blocks=LP_NUM_PAGES,
        kv_cache_tensors=[],
        kv_cache_groups=groups,
        allocator="large_page",
        large_page_bytes=LP_PAGE_BYTES,
        hash_block_size=BLOCK_SIZE,
    )


def _make_large_page_connector(
    extra_config: dict | None = None,
    cpu_bytes: int = LP_PAGE_BYTES * 6,
) -> SimpleCPUOffloadConnector:
    config_dict = {"cpu_bytes_to_use": cpu_bytes}
    if extra_config:
        config_dict.update(extra_config)
    vllm_config = create_vllm_config(
        kv_connector="SimpleCPUOffloadConnector",
        kv_role="kv_both",
        kv_connector_extra_config=config_dict,
    )
    return SimpleCPUOffloadConnector(
        vllm_config=vllm_config,
        role=KVConnectorRole.SCHEDULER,
        kv_cache_config=_make_large_page_kv_cache_config(),
    )


def test_disk_mode_rejects_non_positive_capacity() -> None:
    """Disk mode with disk_capacity_bytes <= 0 raises ValueError.

    Without this guard, the worker operates in disk mode but the scheduler
    classifies events as MEDIUM_CPU instead of MEDIUM_STORAGE, silently
    mislabeling events.
    """
    with pytest.raises(ValueError, match="disk_capacity_bytes > 0"):
        _make_connector(
            extra_config={
                "kv_offload_backend": "disk",
                "disk_path": "/tmp/fake",
                "disk_capacity_bytes": 0,
            }
        )


def _make_ring_kv_cache_config(num_blocks: int = 16) -> KVCacheConfig:
    """A paged full-attention group beside a per-request ring group."""
    full = FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=NUM_KV_HEADS,
        head_size=HEAD_SIZE,
        dtype=DTYPE,
    )
    ring = CircularBufferSpec(
        block_size=6, num_kv_heads=1, head_size=32, head_size_v=0, dtype=DTYPE
    )
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            KVCacheTensor(
                size=_BYTES_PER_BLOCK * num_blocks,
                layers=["attn"],
                layer_stride=_BYTES_PER_BLOCK * num_blocks,
                block_stride=_BYTES_PER_BLOCK,
            ),
            KVCacheTensor(
                size=ring.page_size_bytes * num_blocks,
                layers=["ring"],
                layer_stride=ring.page_size_bytes * num_blocks,
                block_stride=ring.page_size_bytes,
            ),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(["attn"], full),
            KVCacheGroupSpec(["ring"], ring),
        ],
    )


def test_ring_scratch_group_is_accepted_and_never_offloaded() -> None:
    """The ring group's capacity is not a token granularity, it holds no
    hashed blocks, and it never participates in CPU hits."""
    vllm_config = create_vllm_config(
        kv_connector="SimpleCPUOffloadConnector",
        kv_role="kv_both",
        kv_connector_extra_config={"cpu_bytes_to_use": _BYTES_PER_BLOCK * 8},
    )
    connector = SimpleCPUOffloadConnector(
        vllm_config=vllm_config,
        role=KVConnectorRole.SCHEDULER,
        kv_cache_config=_make_ring_kv_cache_config(),
    )
    scheduler = connector.scheduler_manager
    assert scheduler.fa_gidx == 0
    cpu_groups = scheduler.cpu_kv_cache_config.kv_cache_groups
    assert [g.kv_cache_spec.prefix_cacheable for g in cpu_groups] == [True, False]
    # Only the paged group takes part in CPU-side hit lookup.
    attention_groups = scheduler.cpu_coordinator.attention_groups
    assert [g.group_ids for g in attention_groups] == [[0]]


@pytest.mark.parametrize(
    "connector, prefix_caching, worker_blocks, expected_blocks",
    [
        (None, True, 37, None),
        ("NixlConnector", True, 37, None),
        ("SimpleCPUOffloadConnector", False, 37, None),
        ("SimpleCPUOffloadConnector", True, 37, 37),
        ("MultiConnector", True, 37, 37),
        ("SimpleCPUOffloadConnector", True, None, None),
    ],
)
def test_engine_queries_offload_blocks_only_when_enabled(
    monkeypatch, connector, prefix_caching, worker_blocks, expected_blocks
) -> None:
    """Only active offload queries workers, which must return a block count."""
    config = create_vllm_config(
        kv_connector=connector or "SimpleCPUOffloadConnector",
        kv_role="kv_both",
        kv_connector_extra_config={
            "connectors": [
                {"kv_connector": "SimpleCPUOffloadConnector", "kv_role": "kv_both"}
            ]
        },
    )
    if connector is None:
        config.kv_transfer_config = None
    config.cache_config.enable_prefix_caching = prefix_caching
    kv_config = KVCacheConfig(num_blocks=16, kv_cache_tensors=[], kv_cache_groups=[])
    executor = MagicMock()
    executor.get_kv_cache_specs.return_value = [{}]
    executor.get_simple_cpu_offload_num_blocks.return_value = worker_blocks
    engine = object.__new__(engine_core_module.EngineCore)
    engine.model_executor = executor
    monkeypatch.setattr(
        engine_core_module, "get_kv_cache_configs", lambda *args: [kv_config]
    )

    if worker_blocks is None:
        with pytest.raises(AssertionError, match="workers must report"):
            engine._initialize_kv_caches(config)
        return
    engine._initialize_kv_caches(config)

    assert executor.get_simple_cpu_offload_num_blocks.call_count == (
        expected_blocks is not None
    )
    assert config.cache_config.simple_cpu_offload_num_blocks == expected_blocks


def test_large_page_cpu_pool_geometry() -> None:
    """The CPU side of a large_page config mirrors the GPU pool geometry.

    N_cpu = cpu_bytes / S pages of the GPU page size S, and group g gets the
    N_cpu * S // block_bytes_g block ids of those pages.
    """
    connector = _make_large_page_connector(cpu_bytes=LP_PAGE_BYTES * 6)
    scheduler = connector.scheduler_manager
    assert scheduler is not None
    cpu_config = scheduler.cpu_kv_cache_config
    assert cpu_config.allocator == "large_page"
    assert cpu_config.large_page_bytes == LP_PAGE_BYTES
    assert cpu_config.num_blocks == 6
    assert cpu_config.num_blocks == 6
    assert scheduler.num_cpu_blocks == 6
    groups = cpu_config.kv_cache_groups
    # 6 pages of 2048 B hold 6 blocks of 2048 B or 16 of 768 B.
    assert [g.num_blocks for g in groups] == [6, 16]
    assert [g.slots_per_large_page for g in groups] == list(LP_SLOTS_PER_PAGE)
    assert [g.kernel_rows_per_block for g in groups] == list(LP_ROWS_PER_BLOCK)
    assert [g.row_bytes for g in groups] == list(LP_ROW_BYTES)


def test_large_page_num_cpu_blocks_override_counts_pages() -> None:
    """Under large_page the num_cpu_blocks override means CPU large pages."""
    connector = _make_large_page_connector(extra_config={"num_cpu_blocks": 3})
    scheduler = connector.scheduler_manager
    assert scheduler is not None
    assert scheduler.num_cpu_blocks == 3
    groups = scheduler.cpu_kv_cache_config.kv_cache_groups
    assert [g.num_blocks for g in groups] == [3, 8]

    # 7 pages exceed the local capacity of 6 pages.
    with pytest.raises(ValueError, match="num_cpu_blocks"):
        _make_large_page_connector(extra_config={"num_cpu_blocks": 7})


@pytest.mark.parametrize(
    ("extra_config", "match"),
    [
        (
            {"kv_offload_backend": "disk", "disk_path": "/tmp/fake"},
            "disk.*not supported",
        ),
        ({"lazy_offload": True}, "lazy_offload.*not supported"),
    ],
)
def test_large_page_rejects_disk_backend_and_lazy_offload(
    extra_config: dict, match: str
) -> None:
    """v1 gaps under large_page fail at config time, not mid-transfer."""
    with pytest.raises(ValueError, match=match):
        _make_large_page_connector(extra_config=extra_config)
