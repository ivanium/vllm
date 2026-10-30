# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for SimpleCPUOffloadConnector-level behavior."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from tests.v1.kv_connector.unit.utils import (
    create_vllm_config,
    make_two_group_large_page_config,
)
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
    SlidingWindowSpec,
)


def _make_kv_cache_config(num_blocks: int = 16) -> KVCacheConfig:
    """One block per large page."""
    spec = FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=NUM_KV_HEADS,
        head_size=HEAD_SIZE,
        dtype=DTYPE,
    )
    return KVCacheConfig(
        num_pages=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer0"],
                spec,
                num_blocks=num_blocks,
                row_bytes=spec.page_size_bytes,
            )
        ],
        large_page_bytes=spec.page_size_bytes,
        block_size=BLOCK_SIZE,
    )


def _make_connector(
    extra_config: dict | None = None,
    num_gpu_blocks: int = 16,
    cpu_bytes: int = _BYTES_PER_BLOCK * 8,
    kv_cache_config: KVCacheConfig | None = None,
) -> SimpleCPUOffloadConnector:
    if kv_cache_config is None:
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


# Two-group large_page geometry from the shared real-planner fixture: group 0
# (full attention) has 8704 B blocks; group 1 (sliding window) has blocks of
# four 1024 B rows, which do not divide the 34816 B page.
LP_PAGE_BYTES = 34816
LP_ROWS_PER_BLOCK = (1, 4)
LP_ROW_BYTES = (8704, 1024)


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
        kv_cache_config=make_two_group_large_page_config(),
    )


def _make_swa_only_kv_cache_config(num_blocks: int = 16) -> KVCacheConfig:
    """A single prefix-cacheable sliding-window group, no full attention."""
    swa = SlidingWindowSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=NUM_KV_HEADS,
        head_size=HEAD_SIZE,
        dtype=DTYPE,
        sliding_window=BLOCK_SIZE * 4,
    )
    return KVCacheConfig(
        num_pages=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["swa"], swa, num_blocks=num_blocks, row_bytes=swa.page_size_bytes
            )
        ],
        large_page_bytes=swa.page_size_bytes,
        block_size=BLOCK_SIZE,
    )


def test_connector_init_without_full_attention_group() -> None:
    """A prefix-cacheable config with no FullAttention group must initialize.

    Connector init used to require a FullAttentionSpec group for the deleted
    sub-block partial-tail path; SWA-only models crashed on that assert.
    """
    connector = _make_connector(kv_cache_config=_make_swa_only_kv_cache_config())
    scheduler = connector.scheduler_manager
    assert scheduler is not None
    assert scheduler.prefix_cacheable_group_ids == (0,)


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
        num_pages=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["attn"], full, num_blocks=num_blocks, row_bytes=full.page_size_bytes
            ),
            KVCacheGroupSpec(
                ["ring"], ring, num_blocks=num_blocks, row_bytes=ring.page_size_bytes
            ),
        ],
        large_page_bytes=max(full.page_size_bytes, ring.page_size_bytes),
        block_size=BLOCK_SIZE,
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
    kv_config = KVCacheConfig(
        num_pages=16, kv_cache_tensors=[], kv_cache_groups=[], block_size=16
    )
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
    assert cpu_config.large_page_bytes == LP_PAGE_BYTES
    assert cpu_config.num_pages == 6
    assert cpu_config.num_pages == 6
    assert scheduler.num_cpu_blocks == 6
    groups = cpu_config.kv_cache_groups
    # 6 * 34816 B = 24 * 8704 B = 51 * 4096 B.
    assert [g.num_blocks for g in groups] == [24, 51]
    assert [g.kernel_rows_per_block for g in groups] == list(LP_ROWS_PER_BLOCK)
    assert [g.row_bytes for g in groups] == list(LP_ROW_BYTES)


def test_large_page_num_cpu_blocks_override_counts_pages() -> None:
    """Under large_page the num_cpu_blocks override means CPU large pages."""
    connector = _make_large_page_connector(extra_config={"num_cpu_blocks": 3})
    scheduler = connector.scheduler_manager
    assert scheduler is not None
    assert scheduler.num_cpu_blocks == 3
    groups = scheduler.cpu_kv_cache_config.kv_cache_groups
    assert [g.num_blocks for g in groups] == [12, 25]

    # 7 pages exceed the local capacity of 6 pages.
    with pytest.raises(ValueError, match="num_cpu_blocks"):
        _make_large_page_connector(extra_config={"num_cpu_blocks": 7})
