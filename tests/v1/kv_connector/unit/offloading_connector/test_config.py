# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for translating vLLM cache metadata to native offloading config."""

from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest
import torch

from tests.v1.kv_connector.unit.offloading_connector.utils import MockOffloadingSpec
from vllm.config import KVTransferConfig, ParallelConfig, VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.config import (
    build_offloading_config,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    SchedulerOffloadConfig,
)
from vllm.platforms import current_platform
from vllm.v1.core.kv_cache_planning import kv_cache_groups_tp_replicas
from vllm.v1.core.kv_cache_utils import generate_scheduler_kv_cache_config
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    HiddenStateCacheSpec,
    KVCacheConfig,
    KVCacheGroupRole,
    KVCacheGroupSpec,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    SlidingWindowSpec,
)
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry
from vllm.v1.kv_offload.file_mapper import FileMapper
from vllm.v1.kv_offload.tiering.spec import TieringOffloadingSpec


def _make_vllm_config(
    *,
    extra_config: dict[str, Any] | None = None,
    tensor_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
    prefill_context_parallel_size: int = 1,
    decode_context_parallel_size: int = 1,
) -> VllmConfig:
    config = MagicMock()
    config.cache_config.block_size = 16
    config.cache_config.enable_prefix_caching = True
    config.cache_config.cache_dtype = torch.float16
    config.cache_config.prefix_cache_retention_interval = None
    config.model_config.model = "test-model"
    config.model_config.use_mla = False
    config.model_config.get_total_num_kv_heads.return_value = 4
    world_size = (
        tensor_parallel_size * pipeline_parallel_size * prefill_context_parallel_size
    )
    with patch.object(current_platform, "device_count", return_value=world_size):
        config.parallel_config = ParallelConfig(
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            prefill_context_parallel_size=prefill_context_parallel_size,
            decode_context_parallel_size=decode_context_parallel_size,
        )
    config.kv_events_config = None
    config.kv_transfer_config = KVTransferConfig(
        kv_connector="OffloadingConnector",
        kv_role="kv_both",
        kv_connector_extra_config=dict(extra_config or {}),
    )
    return cast(VllmConfig, config)


def _lp_group(
    layer_names: list[str],
    spec: KVCacheSpec,
    num_pages: int = 4,
    row_bytes: int | None = None,
    **kwargs: Any,
) -> KVCacheGroupSpec:
    """A large-page group: one block per page, one kernel row per block."""
    if row_bytes is None:
        row_bytes = spec.page_size_bytes * len(layer_names)
    return KVCacheGroupSpec(
        list(layer_names),
        spec,
        num_blocks=num_pages,
        blocks_per_page=1,
        kernel_rows_per_block=1,
        row_bytes=row_bytes,
        **kwargs,
    )


def _lp_config(
    kv_cache_groups: list[KVCacheGroupSpec],
    num_pages: int = 4,
    block_size: int = 16,
) -> KVCacheConfig:
    return KVCacheConfig(
        num_pages=num_pages,
        kv_cache_tensors=[],
        kv_cache_groups=kv_cache_groups,
        large_page_bytes=max(g.row_bytes for g in kv_cache_groups),
        block_size=block_size,
    )


def _make_kv_cache_config() -> KVCacheConfig:
    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
    )
    return _lp_config([_lp_group(["layer"], spec, num_pages=16)], num_pages=16)


def _full_attention_spec(block_size: int = 16) -> FullAttentionSpec:
    return FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=4,
        head_size=128,
        dtype=torch.float32,
    )


def _mla_spec(
    block_size: int = 16,
    head_size: int = 512,
    dtype: torch.dtype = torch.float32,
    tokens_per_state: int = 1,
) -> MLAAttentionSpec:
    return MLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        max_tp_shards=1,
        head_size=head_size,
        dtype=dtype,
        tokens_per_state=tokens_per_state,
    )


_MAMBA_SPEC = MambaSpec(
    block_size=16,
    shapes=((16, 1),),
    dtypes=(torch.float32,),
)
# Page sizes of the specs the replicated-layout cases below are built from.
_MLA_PAGE = _mla_spec().page_size_bytes
_HALF_MLA_PAGE = _mla_spec(head_size=256).page_size_bytes


def _make_mla_kv_cache_config(
    layer_names: list[str] | None = None,
    head_size: int = 512,
    dtype: torch.dtype = torch.float32,
    num_pages: int = 4,
) -> KVCacheConfig:
    if layer_names is None:
        layer_names = ["layer0", "layer1"]
    spec = _mla_spec(head_size=head_size, dtype=dtype)
    return _lp_config([_lp_group(layer_names, spec, num_pages)], num_pages)


def _make_hybrid_kv_cache_config() -> KVCacheConfig:
    full_spec = _full_attention_spec(block_size=12)
    mla_spec = _mla_spec()
    return _lp_config(
        [
            _lp_group(["full_layer"], full_spec),
            _lp_group(["mla_layer"], mla_spec),
        ],
        block_size=4,
    )


def _mamba_spec() -> MambaSpec:
    return MambaSpec(
        block_size=16,
        shapes=((1, 1),),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
    )


def _make_mamba_hybrid_kv_cache_config() -> KVCacheConfig:
    return _lp_config(
        [
            _lp_group(["full_layer"], _full_attention_spec()),
            _lp_group(["mamba_layer"], _mamba_spec()),
        ]
    )


def _parallelism_agnostic(
    kv_cache_groups: list[KVCacheGroupSpec],
    *,
    canonical: bool = False,
    tensor_parallel_size: int = 1,
) -> bool:
    config = _make_vllm_config(
        extra_config={"canonical_layout": True} if canonical else None,
        tensor_parallel_size=tensor_parallel_size,
    )
    kv_cache_config = _lp_config(kv_cache_groups, num_pages=0)
    kv_cache_config.kv_tp_replicas = kv_cache_groups_tp_replicas(
        kv_cache_groups,
        tensor_parallel_size,
        config.parallel_config.decode_context_parallel_size,
    )
    return build_offloading_config(
        config, kv_cache_config
    ).parallel.is_parallelism_agnostic


def _replicated_layout(
    kv_cache_config: KVCacheConfig,
    *,
    tensor_parallel_size: int = 4,
    pipeline_parallel_size: int = 1,
    prefill_context_parallel_size: int = 1,
    decode_context_parallel_size: int = 1,
    use_mla: bool = True,
    use_v2_model_runner: bool = True,
    distributed_executor_backend: Any = "mp",
    nnodes: int = 1,
    world_size: int | None = None,
    kv_tp_replicas: int | None = None,
) -> bool:
    config = _make_vllm_config(
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        prefill_context_parallel_size=prefill_context_parallel_size,
        decode_context_parallel_size=decode_context_parallel_size,
    )
    config.model_config.use_mla = use_mla
    config.use_v2_model_runner = use_v2_model_runner
    config.parallel_config.distributed_executor_backend = distributed_executor_backend
    config.parallel_config.nnodes = nnodes
    if world_size is not None:
        config.parallel_config.world_size = world_size
    kv_cache_config.kv_tp_replicas = (
        kv_tp_replicas
        if kv_tp_replicas is not None
        else kv_cache_groups_tp_replicas(
            kv_cache_config.kv_cache_groups,
            config.parallel_config.tensor_parallel_size,
            config.parallel_config.decode_context_parallel_size,
        )
    )

    # Scheduler and Worker offload config
    worker_offload_config = build_offloading_config(config, kv_cache_config)
    scheduler_offload_config = build_offloading_config(
        config, generate_scheduler_kv_cache_config([kv_cache_config])
    )
    # Verify that the Worker and Scheduler agree on replication.
    # and worker-bytes-per-block
    assert (
        worker_offload_config.replicated_layout
        == scheduler_offload_config.replicated_layout
    )
    assert (
        worker_offload_config.worker_kv_bytes_per_block
        == scheduler_offload_config.worker_kv_bytes_per_block
    )

    return worker_offload_config.replicated_layout


def _make_swa_mla_kv_cache_config(
    num_pages: int = 4,
) -> KVCacheConfig:
    """Single SlidingWindowMLASpec group — DSV4 sliding-window layers."""
    spec = SlidingWindowMLASpec(
        block_size=16,
        num_kv_heads=1,
        max_tp_shards=1,
        head_size=512,
        dtype=torch.float32,
        sliding_window=128,
    )
    return _lp_config([_lp_group(["layer"], spec, num_pages)], num_pages)


def _make_dsv4_flash_kv_cache_config(num_pages: int = 4) -> KVCacheConfig:
    """Minimal fake of DeepSeek-V4-Flash's five MLA/SWA-MLA groups.

    The real model has the following groups (layer counts from runtime logs);
    2 per group is enough to exercise the structural properties that govern
    replicated_layout:
        group[0]: 22 SlidingWindowMLASpec layers, sliding_window=128
        group[1]: 21 SlidingWindowMLASpec layers, sliding_window=128
        group[2]: 62 MLAAttentionSpec layers  (MLA + indexer.k_cache)
        group[3]: 42 SlidingWindowMLASpec layers, sliding_window=8
        group[4]: 20 SlidingWindowMLASpec layers, sliding_window=128
    All specs have num_kv_heads=1 → every group is TP-replicated.
    """
    block_size = 16
    dtype = torch.float32
    head_size = 512

    def _swa_group(tag: str, sw: int) -> KVCacheGroupSpec:
        names = [f"{tag}_0", f"{tag}_1"]
        spec = SlidingWindowMLASpec(
            block_size=block_size,
            num_kv_heads=1,
            max_tp_shards=1,
            head_size=head_size,
            dtype=dtype,
            sliding_window=sw,
        )
        return _lp_group(names, spec, num_pages)

    def _mla_group(tag: str) -> KVCacheGroupSpec:
        names = [f"{tag}_0", f"{tag}_1"]
        spec = MLAAttentionSpec(
            block_size=block_size,
            num_kv_heads=1,
            max_tp_shards=1,
            head_size=head_size,
            dtype=dtype,
        )
        return _lp_group(names, spec, num_pages)

    groups = [
        _swa_group("swa_128a", 128),  # group[0]: SWA-MLA, sw=128
        _swa_group("swa_128b", 128),  # group[1]: SWA-MLA, sw=128
        _mla_group("mla"),  # group[2]: MLA + indexer layers
        _swa_group("swa_8", 8),  # group[3]: SWA-MLA, sw=8
        _swa_group("swa_128c", 128),  # group[4]: SWA-MLA, sw=128
    ]
    return _lp_config(groups, num_pages)


def _make_dsv3_2_kv_cache_config(num_pages: int = 4) -> KVCacheConfig:
    """Minimal fake of DeepSeek-V3.2-Exp's single group of MLA layers.

    The real model has 1 group (layer counts from runtime logs):
        group[0]: 122 MLAAttentionSpec layers (self_attn.attn + indexer.k_cache)
    All specs have num_kv_heads=1 → the group is TP-replicated.

    GLM-5.3 has the same topology (one group of MLAAttentionSpec layers,
    num_kv_heads=1) with 99 layers instead of 122, so this config covers both
    models.
    """
    names = [f"mla_{i}" for i in range(4)]  # 4 layers; real model has 122
    spec = MLAAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        max_tp_shards=1,
        head_size=512,
        dtype=torch.float32,
    )
    return _lp_config([_lp_group(names, spec, num_pages)], num_pages)


def test_worker_kv_bytes_sums_offloaded_group_rows():
    """worker_kv_bytes_per_block is this rank's per-block bytes summed over
    the offloaded groups: R_g kernel rows of row_g bytes each."""
    config = _make_vllm_config(
        extra_config={"block_size": 32},
        tensor_parallel_size=3,
        pipeline_parallel_size=2,
    )
    kv_cache_config = _lp_config(
        [
            _lp_group(["layer0", "layer1"], _full_attention_spec(), row_bytes=10),
            _lp_group(["mla"], _mla_spec(), row_bytes=6),
        ]
    )

    offloading_config = build_offloading_config(config, kv_cache_config)

    assert offloading_config.worker_kv_bytes_per_block == 16
    assert offloading_config.parallel.world_size == 6
    assert offloading_config.cache.blocks_per_chunk == 2


def test_offloading_skips_scratch_group():
    full_spec = _full_attention_spec()
    mamba = _mamba_spec()
    kv_cache_config = _lp_config(
        [
            _lp_group(["full_layer"], full_spec, row_bytes=128),
            _lp_group(
                ["scratch"],
                CircularBufferSpec(
                    block_size=4, num_kv_heads=1, head_size=128, dtype=torch.float32
                ),
                row_bytes=64,
            ),
            _lp_group(["mamba_layer"], mamba, row_bytes=8),
        ]
    )
    config = _make_vllm_config()

    offloading_config = build_offloading_config(config, kv_cache_config)
    scheduler_config = SchedulerOffloadConfig.from_spec(
        MockOffloadingSpec(offloading_config), config, kv_cache_config
    )

    # The scratch group is neither offloaded nor charged per block.
    assert [group.group_id for group in offloading_config.groups] == [0, 2]
    assert [group.group_idx for group in scheduler_config.kv_group_configs] == [0, 2]
    assert offloading_config.worker_kv_bytes_per_block == 136


def test_hisparse_offloads_only_indexer_group():
    source = _lp_group(["source"], _full_attention_spec(), host_resident=True)
    indexer = _lp_group(
        ["indexer"],
        _full_attention_spec(),
        role=KVCacheGroupRole.HISPARSE_INDEXER,
    )
    kv_cache_config = _lp_config([source, indexer])
    kv_cache_config.hisparse_host_num_blocks = 4
    config = _make_vllm_config()

    offloading_config = build_offloading_config(config, kv_cache_config)
    scheduler_config = SchedulerOffloadConfig.from_spec(
        MockOffloadingSpec(offloading_config), config, kv_cache_config
    )

    assert [
        (group.group_id, group.layer_names) for group in offloading_config.groups
    ] == [(1, ("indexer",))]
    assert [group.group_idx for group in scheduler_config.kv_group_configs] == [1]


def test_zero_blocks_gives_zero_worker_bytes():
    kv_cache_config = _lp_config([_lp_group(["layer"], _full_attention_spec())])
    kv_cache_config.num_pages = 0

    offloading_config = build_offloading_config(_make_vllm_config(), kv_cache_config)

    assert offloading_config.worker_kv_bytes_per_block == 0


def test_prefill_context_parallelism_does_not_scale_group_blocks():
    config = _make_vllm_config(
        extra_config={"block_size": 64},
        prefill_context_parallel_size=2,
    )

    offloading_config = build_offloading_config(config, _make_kv_cache_config())

    assert tuple(group.tokens_per_block for group in offloading_config.groups) == (16,)
    assert offloading_config.cache.tokens_per_hash == 16
    assert offloading_config.cache.blocks_per_chunk == 4


def test_dcp_scales_attention_but_not_mamba_group_blocks():
    config = _make_vllm_config(tensor_parallel_size=2, decode_context_parallel_size=2)
    config.speculative_config = None

    offloading_config = build_offloading_config(
        config, _make_mamba_hybrid_kv_cache_config()
    )

    assert tuple(group.tokens_per_block for group in offloading_config.groups) == (
        32,
        16,
    )
    scheduler_config = SchedulerOffloadConfig.from_spec(
        MockOffloadingSpec(offloading_config),
        config,
        _make_mamba_hybrid_kv_cache_config(),
    )
    mamba_group = scheduler_config.kv_group_configs[1]
    assert mamba_group.sliding_window_size_in_chunks == 1
    assert scheduler_config.alignment_tokens is not None
    manager_cls = KVCacheSpecRegistry.get_manager_class(mamba_group.kv_cache_spec)
    assert manager_cls is not None
    block_mask = manager_cls.reachable_block_mask(
        start_block=0,
        end_block=4,
        alignment_tokens=scheduler_config.alignment_tokens,
        kv_cache_spec=mamba_group.kv_cache_spec,
        use_eagle=mamba_group.is_eagle_group,
    )
    assert block_mask is None or [i for i in range(4) if block_mask[i]] == [1, 3]


@pytest.mark.parametrize("dcp_size,expected", [(1, 16), (2, 32)])
def test_dcp_scales_attention_group_blocks(dcp_size, expected):
    config = _make_vllm_config(
        tensor_parallel_size=2, decode_context_parallel_size=dcp_size
    )
    kv_cache_config = _lp_config(
        [
            _lp_group(
                ["mla_layer", "indexer_layer"],
                _mla_spec(head_size=128),
            )
        ]
    )

    offloading_config = build_offloading_config(config, kv_cache_config)

    assert offloading_config.groups[0].tokens_per_block == expected
    # Hash granularity stays at the global block size under large_page.
    assert offloading_config.cache.tokens_per_hash == 16


def test_preserves_data_parallel_config():
    config = _make_vllm_config()
    config.parallel_config.data_parallel_index = 2
    config.parallel_config.data_parallel_size = 4
    config.parallel_config.data_parallel_rank_local = 1

    offloading_config = build_offloading_config(config, _make_kv_cache_config())

    assert offloading_config.parallel.data_parallel_index == 2
    assert offloading_config.parallel.data_parallel_size == 4
    assert offloading_config.parallel.data_parallel_rank_local == 1


def test_resolves_heterogeneous_hybrid_block_sizes():
    config = _make_vllm_config()
    config.cache_config.block_size = 4

    offloading_config = build_offloading_config(config, _make_hybrid_kv_cache_config())

    assert tuple(group.tokens_per_block for group in offloading_config.groups) == (
        12,
        16,
    )
    assert offloading_config.cache.tokens_per_hash == 4
    assert offloading_config.cache.blocks_per_chunk == 1


@pytest.mark.parametrize("world_size", [2, 4, 8])
@pytest.mark.parametrize("use_v2_model_runner", [False, True], ids=["v1", "v2"])
def test_replicated_layout_enabled_for_pure_mla_tp_mp_single_node(
    world_size: int,
    use_v2_model_runner: bool,
):
    assert _replicated_layout(
        _make_mla_kv_cache_config(),
        tensor_parallel_size=world_size,
        use_v2_model_runner=use_v2_model_runner,
    )


@pytest.mark.parametrize("world_size", [2, 4])
def test_replicated_layout_enabled_for_swa_mla(world_size: int):
    """SlidingWindowMLASpec groups are also replicated and qualify."""
    assert _replicated_layout(
        _make_swa_mla_kv_cache_config(),
        tensor_parallel_size=world_size,
    )


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_replicated_layout_dsv4_flash(world_size: int):
    """DeepSeek-V4-Flash: all 5 UniformType groups are MLA/SWA-MLA with
    num_kv_heads=1, so replicated_layout must be True under TP/mp single-node."""
    assert _replicated_layout(
        _make_dsv4_flash_kv_cache_config(),
        tensor_parallel_size=world_size,
    )


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_replicated_layout_dsv3_2(world_size: int):
    """DeepSeek-V3.2 / GLM-5.3: single UniformType group of MLAAttentionSpec
    layers (122 and 99 respectively), num_kv_heads=1 → replicated_layout must
    be True under TP/mp single-node."""
    assert _replicated_layout(
        _make_dsv3_2_kv_cache_config(),
        tensor_parallel_size=world_size,
    )


def test_replicated_layout_enabled_for_mixed_size_mla_layers():
    """One group of MLA layers of different sizes qualifies: each layer is
    replicated and the row accounting sums the per-layer bytes."""
    mla_layers = ["layer0", "layer1"]
    spec = _mla_spec()
    row_bytes = _MLA_PAGE + _HALF_MLA_PAGE
    kv_cache_config = _lp_config([_lp_group(mla_layers, spec, row_bytes=row_bytes)])
    assert _replicated_layout(kv_cache_config)


@pytest.mark.parametrize(
    ("kv_cache_config", "case"),
    [
        (
            _lp_config(
                [
                    _lp_group(
                        ["layer"],
                        HiddenStateCacheSpec(
                            block_size=16,
                            num_kv_heads=1,
                            head_size=512,
                            dtype=torch.float32,
                        ),
                    )
                ]
            ),
            "hidden-state",
        ),
        (
            _lp_config(
                [
                    _lp_group(["mla"], _mla_spec()),
                    _lp_group(["full"], _full_attention_spec()),
                ]
            ),
            "mla-full-hybrid",
        ),
        (
            _lp_config(
                [
                    _lp_group(["mla"], _mla_spec()),
                    _lp_group(["mamba"], _MAMBA_SPEC),
                ]
            ),
            "mla-mamba-hybrid",
        ),
    ],
    ids=[
        "hidden-state",
        "mla-full-hybrid",
        "mla-mamba-hybrid",
    ],
)
def test_replicated_layout_excludes_unproven_cache_shapes(
    kv_cache_config: KVCacheConfig,
    case: str,
):
    assert not _replicated_layout(kv_cache_config), case


@pytest.mark.parametrize(
    ("kv_cache_config", "case"),
    [
        (
            _make_swa_mla_kv_cache_config(),
            "sliding-window-mla",
        ),
        (
            # One group, two MLA layer sizes: row bytes sum the layers.
            _lp_config(
                [
                    _lp_group(
                        ["layer0", "layer1"],
                        _mla_spec(),
                        row_bytes=_MLA_PAGE + _HALF_MLA_PAGE,
                    )
                ]
            ),
            "mixed-size-mla",
        ),
        (
            _lp_config(
                [
                    _lp_group(["layer0"], _mla_spec()),
                    _lp_group(["layer1"], _mla_spec()),
                ]
            ),
            "multi-group-mla",
        ),
    ],
    ids=[
        "sliding-window-mla",
        "mixed-size-mla",
        "multi-group-mla",
    ],
)
def test_replicated_layout_supported_cache_shapes(
    kv_cache_config: KVCacheConfig,
    case: str,
):
    assert _replicated_layout(kv_cache_config), case


@pytest.mark.parametrize(
    ("kv_tp_replicas", "expected"),
    [
        # DSA-style group: MLA + indexer layers, all TP-replicated.
        (4, True),
        # A group holding head-sharded layers is not replicated.
        (1, False),
    ],
    ids=["replicated-mla-indexer", "sharded-layers"],
)
def test_replicated_layout_for_dsa_style_groups(kv_tp_replicas: int, expected: bool):
    """Replicated layout follows the planner-computed kv_tp_replicas."""
    names = [f"main_{i}" for i in range(4)] + [f"add_{i}" for i in range(4)]
    spec = _mla_spec()
    kv_cache_config = _lp_config([_lp_group(names, spec)])

    assert (
        _replicated_layout(kv_cache_config, kv_tp_replicas=kv_tp_replicas) == expected
    )


@pytest.mark.parametrize(
    ("kwargs", "case"),
    [
        ({"tensor_parallel_size": 1}, "tp1"),
        ({"use_mla": False}, "use-mla-false"),
        ({"pipeline_parallel_size": 2, "world_size": 4}, "pp2"),
        ({"prefill_context_parallel_size": 2, "world_size": 4}, "pcp2"),
        ({"decode_context_parallel_size": 2}, "dcp2"),
        ({"world_size": 8}, "world-ne-tp"),
        ({"distributed_executor_backend": "ray"}, "ray"),
        ({"distributed_executor_backend": "uni"}, "uni"),
        ({"distributed_executor_backend": type("DummyExecutor", (), {})}, "class"),
        ({"nnodes": 2}, "multi-node"),
    ],
    ids=[
        "tp1",
        "use-mla-false",
        "pp2",
        "pcp2",
        "dcp2",
        "world-ne-tp",
        "ray",
        "uni",
        "class",
        "multi-node",
    ],
)
def test_replicated_layout_parallel_gate(kwargs: dict[str, Any], case: str):
    assert not _replicated_layout(_make_mla_kv_cache_config(), **kwargs), case


def test_parallelism_agnostic_for_single_full_attention_group():
    # A single rank trivially holds byte-identical blocks.
    assert _parallelism_agnostic([KVCacheGroupSpec(["l0"], _full_attention_spec())])


_SWA_SPEC = SlidingWindowSpec(
    block_size=16,
    num_kv_heads=4,
    head_size=128,
    dtype=torch.float32,
    sliding_window=128,
)
_SWA_MLA_SPEC = SlidingWindowMLASpec(
    block_size=16,
    num_kv_heads=1,
    max_tp_shards=1,
    head_size=576,
    dtype=torch.float32,
    sliding_window=128,
)


def _groups(*specs: KVCacheSpec) -> list[KVCacheGroupSpec]:
    return [KVCacheGroupSpec([f"l{i}"], spec) for i, spec in enumerate(specs)]


@pytest.mark.parametrize(
    "kv_cache_groups",
    [
        _groups(_SWA_SPEC),
        _groups(_full_attention_spec(), _full_attention_spec()),
    ],
)
def test_parallelism_agnostic_excluded(kv_cache_groups: list[KVCacheGroupSpec]):
    # Head-sharded groups hold different bytes per rank, so whole blocks are
    # not portable across topologies.
    assert not _parallelism_agnostic(kv_cache_groups, tensor_parallel_size=4)


@pytest.mark.parametrize(
    ("kv_cache_groups", "certified"),
    [
        pytest.param(_groups(_mla_spec(head_size=576)), True, id="mla-latent"),
        pytest.param(_groups(_SWA_MLA_SPEC), True, id="swa-mla"),
        pytest.param(
            _groups(_full_attention_spec(), _SWA_SPEC), False, id="attention-hybrid"
        ),
        pytest.param(
            list(_make_mamba_hybrid_kv_cache_config().kv_cache_groups),
            False,
            id="mamba-hybrid",
        ),
        pytest.param(
            _groups(_mla_spec(head_size=576), _mla_spec(head_size=128)),
            True,
            id="multi-group-mla",
        ),
    ],
)
def test_canonical_layout_gate(kv_cache_groups, certified):
    """Whole-block pages are portable exactly when every rank holds
    byte-identical KV (TP-replicated groups, no context parallelism)."""
    assert (
        _parallelism_agnostic(kv_cache_groups, canonical=True, tensor_parallel_size=4)
        is certified
    )


def test_canonical_layout_replication_follows_portability():
    """canonical_layout keeps the replicated single-copy arena exactly when
    the whole-block layout is portable (TP-replicated, no context
    parallelism)."""

    def _get_offloading_config(spec: KVCacheSpec, canonical: bool):
        config = _make_vllm_config(
            tensor_parallel_size=2,
            extra_config={"canonical_layout": True} if canonical else None,
        )
        config.model_config.use_mla = True

        kv_config = _lp_config([_lp_group(["layer"], spec)])
        kv_config.kv_tp_replicas = kv_cache_groups_tp_replicas(
            kv_config.kv_cache_groups,
            config.parallel_config.tensor_parallel_size,
            config.parallel_config.decode_context_parallel_size,
        )

        return build_offloading_config(config, kv_config)

    # TP-replicated MLA: canonical keeps the replicated arena.
    assert _get_offloading_config(_mla_spec(), canonical=False).replicated_layout
    assert _get_offloading_config(_mla_spec(), canonical=True).replicated_layout
    # Head-sharded groups are not portable: no replication under either layout.
    sharded = _get_offloading_config(_SWA_SPEC, canonical=True)
    assert not sharded.replicated_layout
    assert not sharded.parallel.is_parallelism_agnostic


@pytest.mark.parametrize("tp_size", [1, 2, 4])
@pytest.mark.parametrize("scheduler", [False, True])
def test_canonical_mla_dsa_rows_are_tp_independent(tp_size, scheduler):
    # DSv3.2-style group: 4 MLA layers + 3 indexer rows, all TP-replicated.
    layer_pages = [64 * 656] * 4 + [64 * 132] * 3  # uint8, block_size 64
    names = [f"l{i}" for i in range(len(layer_pages))]
    spec = _mla_spec(block_size=64, head_size=656, dtype=torch.uint8)
    kv_cache_config = _lp_config(
        [_lp_group(names, spec, row_bytes=sum(layer_pages))],
        block_size=64,
    )
    kv_cache_config.kv_tp_replicas = tp_size
    if scheduler:
        kv_cache_config = generate_scheduler_kv_cache_config([kv_cache_config])
    config = _make_vllm_config(
        tensor_parallel_size=tp_size,
        extra_config={"canonical_layout": True, "cpu_bytes_to_use": 196608 * 8},
    )
    config.cache_config.kv_cache_layout = "LBHNC"
    config.parallel_config.distributed_executor_backend = "mp"
    offloading_config = build_offloading_config(config, kv_cache_config)
    spec = TieringOffloadingSpec(offloading_config)
    assert spec.kv_bytes_per_chunk == 196608
    assert spec.num_chunks == 8
    mapper = FileMapper.from_offloading_spec("/cache", spec, parallel_agnostic=True)
    assert mapper.get_run_config()["tp_size"] == 1
    assert mapper.get_run_config()["replicated_layout"] is True


def test_accepts_blocks_per_chunk_for_heterogeneous_groups():
    config = _make_vllm_config(extra_config={"blocks_per_chunk": 2})

    offloading_config = build_offloading_config(config, _make_hybrid_kv_cache_config())

    assert tuple(group.tokens_per_block for group in offloading_config.groups) == (
        12,
        16,
    )
    assert offloading_config.cache.blocks_per_chunk == 2


def test_block_size_and_blocks_per_chunk_are_mutually_exclusive():
    config = _make_vllm_config(extra_config={"block_size": 64, "blocks_per_chunk": 2})

    with pytest.raises(ValueError, match="Specify only one"):
        build_offloading_config(config, _make_kv_cache_config())


def test_blocks_per_chunk_must_be_positive():
    config = _make_vllm_config(extra_config={"blocks_per_chunk": 0})

    with pytest.raises(ValueError, match="greater than 0"):
        build_offloading_config(config, _make_kv_cache_config())
