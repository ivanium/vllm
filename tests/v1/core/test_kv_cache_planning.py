# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import vllm.v1.core.kv_cache_planning as kv_cache_planning
import vllm.v1.core.kv_cache_utils as kv_cache_utils
from tests.v1.core.utils import (
    new_kv_cache_spec,
    new_mamba_spec,
    new_mla_spec,
    new_sliding_window_spec,
)
from vllm.config import (
    CacheConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.config.attention import AttentionConfig, HiSparseConfig
from vllm.config.kv_transfer import KVTransferConfig
from vllm.model_executor.layers.attention.mla_attention import MLAAttention
from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache
from vllm.utils.mem_constants import GiB_bytes
from vllm.v1.attention.backend import MultipleOf
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.core.kv_cache_config_builder import get_kv_cache_config_builder
from vllm.v1.core.kv_cache_planning import (
    get_kv_cache_configs,
    get_kv_cache_groups,
    get_kv_cache_groups_from_builder,
    is_kv_cache_spec_uniform,
    kv_cache_groups_tp_replicas,
    select_global_block_size,
)
from vllm.v1.core.kv_cache_utils import (
    generate_scheduler_kv_cache_config,
)
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.hisparse.binding import resolve_hisparse_specs
from vllm.v1.kv_cache_interface import (
    ChunkedLocalAttentionSpec,
    CircularBufferSpec,
    FullAttentionSpec,
    HiddenStateCacheSpec,
    HiSparseHotSpec,
    HiSparseResidentSpec,
    KVCacheConfig,
    KVCacheGroupRole,
    KVCacheGroupSpec,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    SlidingWindowSpec,
    SparseCacheRole,
    get_kv_quant_mode,
    large_page_block_range,
)
from vllm.v1.worker.utils import customize_attention_spec

pytestmark = pytest.mark.cpu_test


def _hisparse_dsv32_specs(
    vllm_config: VllmConfig,
    num_layers: int = 2,
    index_topk_freq: int = 1,
    backend_enum: AttentionBackendEnum = AttentionBackendEnum.FLASHMLA_SPARSE,
) -> dict[str, KVCacheSpec]:
    """DeepSeek-V3.2-style sparse MLA + indexer specs from the real producers.

    Layer i has an indexer when max(i - 1, 0) % index_topk_freq == 0, as in
    GLM-5.2; the layers after it reuse its top-k.
    """
    backend = backend_enum.get_class()
    specs: dict[str, KVCacheSpec] = {}
    for layer_idx in range(num_layers):
        has_indexer = max(layer_idx - 1, 0) % index_topk_freq == 0
        layer = SimpleNamespace(
            _vllm_config=vllm_config,
            attn_backend=backend,
            kv_cache_dtype=vllm_config.cache_config.cache_dtype,
            head_size=576,
            topk_buffer_width=None,
            sliding_window=None,
            indexer=SimpleNamespace() if has_indexer else None,
            non_causal_multi_token_decode=False,
        )
        specs[f"model.layers.{layer_idx}.self_attn"] = customize_attention_spec(
            backend, MLAAttention.get_kv_cache_spec(layer, vllm_config)
        )
        if not has_indexer:
            continue
        indexer = SimpleNamespace(
            cache_config=vllm_config.cache_config, head_dim=128, dtype=torch.uint8
        )
        specs[f"model.layers.{layer_idx}.self_attn.indexer"] = (
            DeepseekV32IndexerCache.get_kv_cache_spec(indexer, vllm_config)
        )
    return specs


def _hisparse_vllm_config(
    block_size: int = 64,
    max_model_len: int = 256,
    cache_dtype: str = "auto",
    index_topk: int = 128,
):
    config = VllmConfig(
        model_config=ModelConfig(
            model="facebook/opt-125m",
            max_model_len=max_model_len,
            hf_overrides={
                "index_topk": index_topk,
                "max_position_embeddings": max_model_len,
            },
            skip_tokenizer_init=True,
        ),
        cache_config=CacheConfig(block_size=block_size, cache_dtype=cache_dtype),
        attention_config=AttentionConfig(hisparse_config=HiSparseConfig()),
        kv_transfer_config=KVTransferConfig(
            kv_connector="HiSparseConnector",
            kv_role="kv_both",
            kv_connector_extra_config={"host_pool_gib": 1},
        ),
        parallel_config=ParallelConfig(),
        scheduler_config=SchedulerConfig(
            max_model_len=max_model_len,
            is_encoder_decoder=False,
            disable_hybrid_kv_cache_manager=False,
        ),
    )
    register_all_kvcache_specs(config)
    return config


def test_hisparse_large_page_planner():
    """The planner places HiSparse groups in the large-page pool and the host
    source group in its private pool, all at the global block size."""
    block_size = 64
    vllm_config = _hisparse_vllm_config(block_size=block_size)
    specs = _hisparse_dsv32_specs(vllm_config)
    # The worker-side spec resolution adds the derived resident/hot specs.
    specs = resolve_hisparse_specs(vllm_config, specs)
    (config,) = get_kv_cache_configs(
        vllm_config, [specs], [vllm_config.cache_config.block_size * 2**20]
    )

    groups = config.kv_cache_groups
    assert config.block_size == block_size
    assert all(group.kv_cache_spec.block_size == block_size for group in groups)

    host_group = groups[0]
    assert host_group.host_resident
    assert host_group.role == KVCacheGroupRole.HISPARSE_SOURCE
    assert set(host_group.layer_names) == {
        "model.layers.0.self_attn",
        "model.layers.1.self_attn",
    }
    # The host pool is a private, densely packed pool with one region per
    # source layer; its block table maps one-to-one to host blocks.
    source_page = next(iter(host_group.layer_kv_cache_specs.values())).page_size_bytes
    assert config.hisparse_host_num_blocks is not None
    assert config.hisparse_host_block_stride == 2 * source_page
    assert config.hisparse_host_num_blocks == 2**30 // (2 * source_page)
    assert host_group.num_blocks == config.hisparse_host_num_blocks
    host_tensors = [t for t in config.kv_cache_tensors if t.host_resident]
    assert len(host_tensors) == 2
    for tensor in host_tensors:
        assert tensor.block_stride == source_page
        assert tensor.layer_stride == source_page * config.hisparse_host_num_blocks
        assert tensor.size == config.hisparse_host_block_stride * (
            config.hisparse_host_num_blocks
        )
    host_rows = config.group_geometry(0).first_rows(np.arange(host_group.num_blocks))
    np.testing.assert_array_equal(host_rows, np.arange(host_group.num_blocks))

    indexer_group = groups[1]
    assert indexer_group.role == KVCacheGroupRole.HISPARSE_INDEXER
    assert not indexer_group.host_resident
    resident_groups = [
        group
        for group in groups
        if isinstance(group.kv_cache_spec, HiSparseResidentSpec)
    ]
    hot_groups = [
        group for group in groups if isinstance(group.kv_cache_spec, HiSparseHotSpec)
    ]
    assert len(resident_groups) == len(hot_groups) == 2
    # Paired hot/resident groups share one slot geometry, so a block id names
    # the same rows in both (the pool hands a page to one group at a time).
    for resident, hot in zip(resident_groups, hot_groups):
        assert resident.blocks_per_page == hot.blocks_per_page
        assert resident.row_bytes == hot.row_bytes
    num_pages = config.num_pages
    assert num_pages == config.num_pages > 0
    for group_id, group in enumerate(groups[1:], start=1):
        block_bytes = config.group_geometry(group_id).block_bytes
        assert group.num_blocks == num_pages * config.large_page_bytes // block_bytes


def test_hisparse_page_holds_whole_slots_of_every_group():
    """GLM-5.2-FP8 as served (78 layers, an indexer every 4, fp8 KV on
    FLASHINFER_MLA_SPARSE): the 21-layer indexer group has the largest slot,
    and no small multiple of it holds a 4-layer hot/resident slot without
    slack. The page is the LCM of the slots, so every device group tiles it
    exactly (a 4-layer hot group got one slot per page and wasted 57%)."""
    vllm_config = _hisparse_vllm_config(
        block_size=64, max_model_len=16384, cache_dtype="fp8_e4m3", index_topk=2048
    )
    specs = _hisparse_dsv32_specs(
        vllm_config,
        num_layers=78,
        index_topk_freq=4,
        backend_enum=AttentionBackendEnum.FLASHINFER_MLA_SPARSE,
    )
    specs = resolve_hisparse_specs(vllm_config, specs)
    (config,) = get_kv_cache_configs(vllm_config, [specs], [8 * GiB_bytes])

    device_groups = [g for g in config.kv_cache_groups if not g.host_resident]
    assert {len(g.layer_names) for g in device_groups} >= {4}
    for group in device_groups:
        slot = group.kernel_rows_per_block * group.row_bytes
        assert group.blocks_per_page * slot == config.large_page_bytes


def test_hisparse_requires_kernel_supported_block_size():
    """The host tier pages KV at the global block size, so every sparse-MLA
    backend must take it as a kernel block size."""
    vllm_config = _hisparse_vllm_config(block_size=64)
    specs = _hisparse_dsv32_specs(vllm_config)
    specs = {
        name: (
            replace(spec, kernel_block_sizes=(32,))
            if spec.cache_role is SparseCacheRole.SPARSE
            else spec
        )
        for name, spec in specs.items()
    }
    with pytest.raises(ValueError, match="kernel block size"):
        get_kv_cache_groups(vllm_config, specs)


def new_chunked_local_attention_spec(
    block_size=16,
    num_kv_heads=2,
    head_size=64,
    dtype=torch.float32,
    page_size_padded=None,
    attention_chunk_size=4,
):
    return ChunkedLocalAttentionSpec(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=dtype,
        page_size_padded=page_size_padded,
        attention_chunk_size=attention_chunk_size,
    )


def test_is_kv_cache_spec_uniform():
    kv_cache_spec = {
        "layer_1": new_kv_cache_spec(num_kv_heads=32),
        "layer_2": new_kv_cache_spec(num_kv_heads=32),
    }
    assert is_kv_cache_spec_uniform(kv_cache_spec)

    kv_cache_spec = {
        "layer_1": new_kv_cache_spec(num_kv_heads=32),
        "layer_2": new_kv_cache_spec(num_kv_heads=32, sliding_window=1),
    }
    assert is_kv_cache_spec_uniform(kv_cache_spec)

    kv_cache_spec = {
        "layer_1": new_kv_cache_spec(num_kv_heads=32),
        "layer_2": new_sliding_window_spec(num_kv_heads=32, sliding_window=1),
    }
    assert not is_kv_cache_spec_uniform(kv_cache_spec)

    kv_cache_spec = {
        "layer_1": new_sliding_window_spec(num_kv_heads=32, sliding_window=1),
        "layer_2": new_sliding_window_spec(num_kv_heads=32, sliding_window=1),
    }
    assert is_kv_cache_spec_uniform(kv_cache_spec)

    kv_cache_spec = {
        "layer_1": new_sliding_window_spec(num_kv_heads=32, sliding_window=1),
        "layer_2": new_sliding_window_spec(num_kv_heads=32, sliding_window=2),
    }
    assert not is_kv_cache_spec_uniform(kv_cache_spec)

    script = """
import sys

from vllm.v1.core.kv_cache_planning import is_kv_cache_spec_uniform
from vllm.v1.kv_cache_interface import KVCacheSpec

if sys.flags.optimize < 1:
    raise RuntimeError("subprocess is not running with optimization enabled")
specs = {
    "a": KVCacheSpec(block_size=1),
    "b": KVCacheSpec(block_size=2),
}
if is_kv_cache_spec_uniform(specs):
    raise RuntimeError("different specs were treated as uniform")
"""
    subprocess.run(
        [sys.executable, "-O", "-c", script],
        cwd=Path(__file__).resolve().parents[3],
        check=True,
        timeout=60,
    )


def test_get_kv_cache_configs_attention_free():
    kv_cache_specs: dict[str, KVCacheSpec] = {}
    vllm_config = VllmConfig(model_config=ModelConfig(max_model_len=16))
    vllm_config.cache_config.prefix_cache_retention_interval = None
    (kv_cache_config,) = get_kv_cache_configs(vllm_config, [kv_cache_specs], [0])
    assert kv_cache_config.num_pages == 1
    assert kv_cache_config.kv_cache_tensors == []
    assert kv_cache_config.kv_cache_groups == []
    assert kv_cache_config.num_pages == 1
    # Empty specs resolve to the default floor of the block size.
    assert kv_cache_config.block_size == vllm_config.cache_config.block_size == 64


def new_swa_mla_spec(head_size=576, sliding_window=128, model_version=None):
    return SlidingWindowMLASpec(
        block_size=16,
        num_kv_heads=1,
        max_tp_shards=1,
        head_size=head_size,
        dtype=torch.float32,
        sliding_window=sliding_window,
        model_version=model_version,
    )


def test_auto_fit_max_model_len():
    """Test that max_model_len=-1 auto-fits to available GPU memory."""
    kv_cache_specs = {
        "layer_1": new_kv_cache_spec(),
        "layer_2": new_kv_cache_spec(),
    }
    page_bytes = new_kv_cache_spec().page_size_bytes  # 16 KiB per layer

    # With enough memory, max_model_len stays at the derived max.
    vllm_config = _large_page_vllm_config(max_model_len=1024)
    vllm_config.model_config.original_max_model_len = -1
    get_kv_cache_configs(vllm_config, [kv_cache_specs], [page_bytes * 2 * 1024])
    assert vllm_config.model_config.max_model_len == 1024

    # B defaults to 64, so a page is one 2-layer slot of 4 * page_bytes each.
    # Memory for 8 such pages (7 usable after the reserved null page): one
    # request needs cdiv(len, 64) pages, so auto-fit lands on 7 blocks of 64.
    vllm_config = _large_page_vllm_config(max_model_len=1024)
    vllm_config.model_config.original_max_model_len = -1
    get_kv_cache_configs(vllm_config, [kv_cache_specs], [page_bytes * 2 * 32])
    assert vllm_config.model_config.max_model_len == 448


def test_auto_fit_max_model_len_respects_num_gpu_blocks_override():
    """Auto-fit must size max_model_len against the override-clamped pool, not
    the raw `available_memory`. Without this, auto-fit could pick a
    max_model_len that no longer fits once `num_gpu_blocks_override` is applied.
    """
    vllm_config = _large_page_vllm_config(max_model_len=16384)
    vllm_config.model_config.original_max_model_len = -1
    vllm_config.cache_config.num_gpu_blocks_override = 32

    kv_cache_specs = {
        "layer_1": new_kv_cache_spec(),
        "layer_2": new_kv_cache_spec(),
    }
    # Plenty of raw memory (1024 large pages would fit max_model_len=16384).
    large_available_memory = new_kv_cache_spec().page_size_bytes * 2 * 1024

    get_kv_cache_configs(vllm_config, [kv_cache_specs], [large_available_memory])

    # 32 pages, 31 usable: auto-fit must cap at 31 blocks of 64 tokens.
    assert vllm_config.model_config.max_model_len == 1984


def test_auto_fit_max_model_len_hybrid_mamba():
    """Auto-fit over a hybrid FA + align-mode Mamba pool: the Mamba group's
    resident blocks cost whole pages alongside the FA group's stream."""
    fa_page_bytes = new_kv_cache_spec().page_size_bytes  # 16 KiB per layer
    kv_cache_specs = {
        "fa.0": new_kv_cache_spec(),
        "fa.1": new_kv_cache_spec(),
        "mamba.0": new_mamba_spec(mamba_cache_mode="align"),
    }
    # At B=64 the page holds one 2-layer FA slot (128 KiB) or eight 1-layer
    # Mamba slots (16 KiB each). A request needs cdiv(len, 64) FA pages plus
    # cdiv(4, 8) = 1 Mamba page (2 resident states + 2 speculative blocks).
    vllm_config = _large_page_vllm_config(max_model_len=1024, mamba_align=True)
    vllm_config.model_config.original_max_model_len = -1
    get_kv_cache_configs(vllm_config, [kv_cache_specs], [fa_page_bytes * 2 * 32])
    # 8 pages, 7 usable: cdiv(len, 64) + 1 <= 7 fits len = 6 blocks.
    assert vllm_config.model_config.max_model_len == 384


def test_large_page_placement_pp_skewed_groups():
    """PP-skewed hybrid: the globally largest group (FA, 57 layers on rank 1)
    holds only 3 layers on rank 0, where another group (SWA, 20 layers) is
    bigger. Each group's row is its largest over ranks, so both ranks share
    one page that holds whole blocks of every group."""
    vllm_config = _large_page_vllm_config(max_model_len=1024)
    fa_spec = new_kv_cache_spec()
    swa_spec = new_sliding_window_spec(sliding_window=128)
    worker0 = {f"fa.{i}": fa_spec for i in range(3)}
    worker0.update({f"swa.{i}": swa_spec for i in range(20)})
    worker1 = {f"fa.{i}": fa_spec for i in range(3, 60)}

    configs = get_kv_cache_configs(
        vllm_config, [worker0, worker1], [GiB_bytes, GiB_bytes]
    )

    for config in configs:
        for group in config.kv_cache_groups:
            assert group.blocks_per_page >= 1
    assert configs[0].large_page_bytes == configs[1].large_page_bytes
    assert configs[0].num_pages == configs[1].num_pages
    for group0, group1 in zip(configs[0].kv_cache_groups, configs[1].kv_cache_groups):
        assert group0.row_bytes == group1.row_bytes


def test_check_enough_kv_cache_memory_respects_num_gpu_blocks_override():
    """Admission check must use the override-clamped pool size, not raw
    `available_memory`. Without this, startup could accept a max_model_len
    that does not actually fit in `num_gpu_blocks_override` blocks.
    """
    vllm_config = _large_page_vllm_config(max_model_len=16384)
    # 32 pages is far too small for max_model_len=16384 (would need 1025).
    vllm_config.cache_config.num_gpu_blocks_override = 32

    kv_cache_specs = {
        "layer_1": new_kv_cache_spec(),
        "layer_2": new_kv_cache_spec(),
    }
    # Plenty of raw memory: a bytes-only check against this would pass.
    large_available_memory = new_kv_cache_spec().page_size_bytes * 2 * 1024

    with pytest.raises(ValueError, match="max seq len"):
        get_kv_cache_configs(vllm_config, [kv_cache_specs], [large_available_memory])


def test_unify_hybrid_kv_cache_specs():
    # 1. has_full_attention and has_sliding_window
    before_spec_1 = new_kv_cache_spec(block_size=64)
    before_spec_2 = new_sliding_window_spec(
        block_size=16, page_size_padded=32 * 1024, sliding_window=1024
    )
    kv_cache_spec = {
        "layer_1": before_spec_1,
        "layer_2": before_spec_2,
    }
    kv_cache_spec["draft_layer_1"] = replace(before_spec_1, dcp_sharded=False)
    kv_cache_spec["draft_layer_2"] = replace(before_spec_2, dcp_sharded=False)
    kv_cache_planning.unify_hybrid_kv_cache_specs(kv_cache_spec)
    expected_spec_1 = new_kv_cache_spec(block_size=64)
    expected_spec_2 = new_kv_cache_spec(
        block_size=64, page_size_padded=64 * 1024, sliding_window=1024
    )
    assert kv_cache_spec["layer_1"] == expected_spec_1
    assert kv_cache_spec["layer_2"] == expected_spec_2
    assert kv_cache_spec["layer_2"].page_size_bytes == 64 * 1024
    assert kv_cache_spec["draft_layer_2"] == replace(expected_spec_2, dcp_sharded=False)

    # 2. has_full_attention and has_chunked_local_attention
    before_spec_1 = new_kv_cache_spec()
    before_spec_2 = new_chunked_local_attention_spec(
        page_size_padded=32 * 1024, attention_chunk_size=512
    )
    kv_cache_spec = {
        "layer_1": before_spec_1,
        "layer_2": before_spec_2,
    }
    kv_cache_planning.unify_hybrid_kv_cache_specs(kv_cache_spec)
    expected_spec_1 = new_kv_cache_spec()
    expected_spec_2 = new_kv_cache_spec(
        page_size_padded=32 * 1024, attention_chunk_size=512
    )

    assert kv_cache_spec["layer_1"] == expected_spec_1
    assert kv_cache_spec["layer_2"] == expected_spec_2

    # 3. has_full_attention, has_sliding_window and has_chunked_local_attention
    before_spec_1 = new_kv_cache_spec()
    before_spec_2 = new_sliding_window_spec(
        page_size_padded=32 * 1024, sliding_window=1024
    )
    before_spec_3 = new_chunked_local_attention_spec(
        page_size_padded=32 * 1024, attention_chunk_size=512
    )
    kv_cache_spec = {
        "layer_1": before_spec_1,
        "layer_2": before_spec_2,
        "layer_3": before_spec_3,
    }
    kv_cache_planning.unify_hybrid_kv_cache_specs(kv_cache_spec)
    expected_spec_1 = new_kv_cache_spec()
    expected_spec_2 = new_kv_cache_spec(page_size_padded=32 * 1024, sliding_window=1024)
    expected_spec_3 = new_kv_cache_spec(
        page_size_padded=32 * 1024, attention_chunk_size=512
    )
    assert kv_cache_spec["layer_1"] == expected_spec_1
    assert kv_cache_spec["layer_2"] == expected_spec_2
    assert kv_cache_spec["layer_3"] == expected_spec_3

    # 4. No FullAttentionSpec, should not convert
    kv_cache_spec = {
        "layer_1": new_sliding_window_spec(sliding_window=1024),
        "layer_2": new_chunked_local_attention_spec(attention_chunk_size=512),
    }

    with pytest.raises(ValueError):
        kv_cache_planning.unify_hybrid_kv_cache_specs(kv_cache_spec)

    # Replicated Mamba state still requires the hybrid cache manager.
    kv_cache_spec = {"attention": new_kv_cache_spec(), "mamba": new_mamba_spec()}
    with pytest.raises(ValueError):
        kv_cache_planning.unify_hybrid_kv_cache_specs(kv_cache_spec)


def _spec_decode_kv_cache_groups(worker_specs, use_eagle_block_drop=True):
    config = VllmConfig(model_config=ModelConfig(max_model_len=64))
    config.cache_config.kv_cache_layout = "BLHNC"
    config.speculative_config = SimpleNamespace(
        method="dflash",
        num_speculative_tokens=None,
        use_eagle=lambda: True,
        use_eagle_block_drop=lambda: use_eagle_block_drop,
        use_multi_module_mtp=lambda: False,
        parallel_drafting=False,
    )
    configs = get_kv_cache_configs(config, worker_specs, [2**30] * len(worker_specs))
    return generate_scheduler_kv_cache_config(configs).kv_cache_groups


def _as_draft(spec):
    return replace(spec, is_draft=True)


def _hybrid_specs(draft=True):
    """Qwen3.5-shaped [GDN x3, full-attn] x2, plus an MTP full-attn layer."""
    specs = {}
    for i in range(8):
        specs[f"layers.{i}"] = (
            new_kv_cache_spec()
            if i % 4 == 3
            else new_mamba_spec(mamba_cache_mode="align")
        )
    if draft:
        specs["mtp.0"] = _as_draft(new_kv_cache_spec())
    return specs


_DRAFT_SPEC_SHAPES = {
    "uniform": lambda: {
        "layers.0": new_kv_cache_spec(),
        "draft.0": _as_draft(new_kv_cache_spec()),
    },
    "hybrid": _hybrid_specs,
    "draft_spans_groups": lambda: {
        "layers.0": new_mla_spec(),
        "layers.1": new_swa_mla_spec(head_size=1024),
        "mtp.0": _as_draft(new_mla_spec()),
        "mtp.1": _as_draft(new_swa_mla_spec(head_size=1024)),
    },
}


@pytest.mark.parametrize(
    "make_specs", _DRAFT_SPEC_SHAPES.values(), ids=list(_DRAFT_SPEC_SHAPES)
)
@pytest.mark.parametrize("pp", [1, 2])
def test_eagle_groups_are_the_groups_holding_draft_layers(make_specs, pp):
    specs = make_specs()
    draft_layers = {name for name, spec in specs.items() if spec.is_draft}
    untagged = {name: replace(spec, is_draft=False) for name, spec in specs.items()}
    groups = _spec_decode_kv_cache_groups([specs])
    assert [(g.layer_names, g.kv_cache_spec) for g in groups] == [
        (g.layer_names, g.kv_cache_spec)
        for g in _spec_decode_kv_cache_groups([untagged])
    ]

    worker_specs = [
        {n: s for n, s in specs.items() if n not in draft_layers},
        {n: s for n, s in specs.items() if n in draft_layers},
    ]
    scheduler_groups = _spec_decode_kv_cache_groups(
        [specs] if pp == 1 else worker_specs
    )
    assert [g.is_eagle_group for g in scheduler_groups] == [
        bool(draft_layers & set(g.layer_names)) for g in groups
    ]


def test_no_eagle_groups_without_eagle_block_drop():
    groups = _spec_decode_kv_cache_groups([_hybrid_specs()], use_eagle_block_drop=False)
    assert not any(g.is_eagle_group for g in groups)


@pytest.mark.parametrize(
    "make_specs,warns",
    [
        (lambda: _hybrid_specs(draft=False), True),
        (
            lambda: {
                "layers.0": new_kv_cache_spec(),
                "draft.0": new_kv_cache_spec(),
            },
            False,
        ),
    ],
    ids=["untagged_hybrid", "untagged_uniform_attention"],
)
def test_warns_only_when_untagged_draft_meets_mamba(make_specs, warns, caplog_vllm):
    _spec_decode_kv_cache_groups([make_specs()])
    warned = "could be identified as the draft model's" in caplog_vllm.text
    assert warned == warns


_GQA_SPEC = FullAttentionSpec(
    block_size=16, num_kv_heads=8, head_size=64, dtype=torch.float32
)


_GQA_SWA_SPEC = SlidingWindowSpec(
    block_size=16, num_kv_heads=8, head_size=64, dtype=torch.float32, sliding_window=128
)


@pytest.mark.parametrize(
    "specs,expected",
    [
        pytest.param({"l.0": new_mla_spec(), "l.1": new_mla_spec()}, 4, id="mla"),
        pytest.param(
            {
                "l.0": new_mla_spec(),
                "l.1": new_mla_spec(),
                "s.0": new_swa_mla_spec(),
                "s.1": new_swa_mla_spec(),
            },
            4,
            id="mla-and-swa-mla",
        ),
        pytest.param({"l.0": _GQA_SPEC, "l.1": _GQA_SPEC}, 1, id="gqa"),
        pytest.param(
            {"l.0": new_mla_spec(), "l.1": _GQA_SPEC},
            1,
            id="mla-and-gqa-uniform-group",
        ),
        pytest.param(
            {
                "l.0": new_mla_spec(),
                "l.1": new_mla_spec(),
                "s.0": _GQA_SWA_SPEC,
                "s.1": _GQA_SWA_SPEC,
            },
            1,
            id="mla-and-gqa-swa",
        ),
    ],
)
def test_kv_tp_replicas(monkeypatch, specs, expected):
    """Resolved per layer before scheduler flattening hides mixed groups."""
    from vllm.config import ParallelConfig
    from vllm.platforms import current_platform

    monkeypatch.setattr(current_platform, "device_count", lambda: 4)
    parallel_config = ParallelConfig(tensor_parallel_size=4)
    vllm_config = VllmConfig(
        model_config=ModelConfig(max_model_len=16), parallel_config=parallel_config
    )
    vllm_config.cache_config.kv_cache_layout = "LBNHC"
    mem = sum(s.page_size_bytes for s in specs.values()) * 100
    configs = get_kv_cache_configs(vllm_config, [specs], [mem])
    assert configs[0].kv_tp_replicas == expected
    scheduler = generate_scheduler_kv_cache_config(configs)
    assert scheduler.kv_tp_replicas == expected


@pytest.mark.parametrize(
    "specs,tp_size,dcp_size,expected",
    [
        pytest.param([new_mla_spec()], 8, 1, 8, id="mla"),
        pytest.param([new_mla_spec()], 8, 2, 1, id="mla-dcp"),
        pytest.param(
            [replace(new_mla_spec(), max_tp_shards=None)], 8, 1, 1, id="mla-unset"
        ),
        pytest.param([replace(_GQA_SPEC, max_tp_shards=2)], 8, 1, 4, id="gqa-partial"),
        pytest.param([replace(_GQA_SPEC, max_tp_shards=8)], 4, 1, 1, id="gqa-sharded"),
        pytest.param(
            [new_mla_spec(), replace(_GQA_SWA_SPEC, max_tp_shards=4)],
            8,
            1,
            2,
            id="mla-and-gqa-partial",
        ),
        pytest.param(
            [
                HiddenStateCacheSpec(
                    block_size=16, num_kv_heads=1, head_size=64, dtype=torch.float32
                )
            ],
            8,
            1,
            1,
            id="hidden-state",
        ),
    ],
)
def test_kv_cache_groups_tp_replicas(specs, tp_size, dcp_size, expected):
    """Replicas are the gcd of each layer's tp_size // max_tp_shards."""
    groups = [KVCacheGroupSpec([f"l.{i}"], spec) for i, spec in enumerate(specs)]
    assert kv_cache_groups_tp_replicas(groups, tp_size, dcp_size) == expected


# ---------------------------------------------------------------------------
# Large-page placement
#
# Golden grouping / block-size / placement checks for the five target models.
# Spec maps are minimal reconstructions of the models' real worker-side specs
# (classes and numbers from the model builders); expected values come from the
# design table and the validated placement prototype (60 GiB KV budget/rank).
# ---------------------------------------------------------------------------

_LARGE_PAGE_KIMI = "moonshotai/Kimi-Linear-48B-A3B-Instruct"
_LARGE_PAGE_BUDGET = 60 * GiB_bytes


def _large_page_vllm_config(
    model="facebook/opt-125m", max_model_len=131072, mamba_align=False
):
    vllm_config = VllmConfig(
        model_config=ModelConfig(
            model=model,
            max_model_len=max_model_len,
            # opt-125m derives max_model_len=2048; let the test override it.
            hf_overrides={"max_position_embeddings": max_model_len},
            skip_tokenizer_init=True,
        ),
        cache_config=CacheConfig(
            mamba_cache_mode="align" if mamba_align else "none",
        ),
        parallel_config=ParallelConfig(),
        scheduler_config=SchedulerConfig(
            max_model_len=max_model_len, is_encoder_decoder=False
        ),
    )
    register_all_kvcache_specs(vllm_config)
    return vllm_config


def _dsv41_specs() -> dict[str, KVCacheSpec]:
    """DeepSeek-V4.1-Flash, dspark k=5: 43 SWA (bounded replay, private 32),
    4 compressed MLA + 4 indexer (tokens/state 2/2/2/1), 3 compressor rings."""
    specs: dict[str, KVCacheSpec] = {}
    swa = dict(
        bounded_replay=True,
        block_size=32,
        num_kv_heads=1,
        max_tp_shards=1,
        head_size=512,
        dtype=torch.uint8,
        sliding_window=128,
        cache_dtype_str="fp8_ds_mla",
        state_content_bytes=528,
        alignment=512,
        block_stride_alignment=528,
        model_version="deepseek_v4",
        kv_quant_mode=get_kv_quant_mode("fp8_ds_mla"),
        kernel_block_sizes=(MultipleOf(32),),
    )
    for i in range(40):
        specs[f"model.layers.{i}.attn.swa_cache"] = SlidingWindowMLASpec(**swa)
    for j in range(3):
        specs[f"draft_model.layers.{40 + j}.attn.swa_cache"] = SlidingWindowMLASpec(
            **swa, is_draft=True
        )
    for i in (2, 8, 14, 20):
        ratio = 1 if i == 20 else 2
        specs[f"model.layers.{i}.attn"] = MLAAttentionSpec(
            block_size=16,
            num_kv_heads=1,
            head_size=512,
            dtype=torch.uint8,
            tokens_per_state=ratio,
            cache_dtype_str="fp8_ds_mla",
            alignment=512,
            model_version="deepseek_v4",
            kv_quant_mode=get_kv_quant_mode("fp8_ds_mla"),
            state_content_bytes=528,
            kernel_block_sizes=(128,),
        )
        specs[f"model.layers.{i}.attn.indexer.k_cache"] = MLAAttentionSpec(
            block_size=16,
            num_kv_heads=1,
            head_size=132,
            dtype=torch.uint8,
            tokens_per_state=ratio,
            alignment=512,
            kernel_block_sizes=(128,),
        )
        if ratio > 1:
            specs[f"model.layers.{i}.attn.compressor.state_cache"] = CircularBufferSpec(
                block_size=8,
                num_kv_heads=1,
                head_size=1024,
                head_size_v=0,
                dtype=torch.float32,
            )
    return specs


def _glm53_specs(tp: int) -> dict[str, KVCacheSpec]:
    """GLM-5.3-Flash, MTP k=1: 34 KDA (mamba align), 12 sparse MLA + 12 indexer
    (11 target + 1 draft each), 12 tail rings."""
    specs: dict[str, KVCacheSpec] = {}
    kda_layers = [
        0,
        1,
        2,
        4,
        5,
        6,
        8,
        9,
        10,
        12,
        13,
        14,
        16,
        17,
        18,
        20,
        21,
        22,
        24,
        25,
        26,
        28,
        29,
        30,
        32,
        33,
        34,
        36,
        37,
        38,
        40,
        41,
        42,
        44,
    ]
    for i in kda_layers:
        specs[f"model.language_model.layers.{i}.self_attn"] = MambaSpec(
            block_size=16,
            shapes=((24576 // tp, 4), (64 // tp, 128, 128)),
            dtypes=(torch.bfloat16, torch.float32),
            mamba_cache_mode="align",
            num_speculative_blocks=1,
            num_prefill_checkpoint_blocks=1,
            prefill_checkpoint_alignment=16,
        )
    prefixes = [
        f"model.language_model.layers.{i}.self_attn"
        for i in (3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43)
    ] + ["draft.model.layers.45.mtp_block.self_attn"]
    for prefix in prefixes:
        is_draft = prefix.startswith("draft.")
        specs[f"{prefix}.attn"] = MLAAttentionSpec(
            block_size=16,
            num_kv_heads=1,
            max_tp_shards=1,
            head_size=512,
            dtype=torch.uint8,
            cache_dtype_str="fp8",
            kv_quant_mode=get_kv_quant_mode("fp8"),
            topk_buffer_width=2176,
            is_index_group_leader=True,
            block_stride_alignment=16384,
            kernel_block_sizes=(MultipleOf(32),),
            is_draft=is_draft,
        )
        specs[f"{prefix}.indexer.k_cache"] = MLAAttentionSpec(
            block_size=16,
            num_kv_heads=1,
            max_tp_shards=1,
            head_size=132,
            dtype=torch.uint8,
            cache_role=SparseCacheRole.INDEXER,
            tokens_per_state=4,
            block_stride_alignment=MultipleOf(256),
            kernel_block_sizes=(128, 256),
            is_draft=is_draft,
        )
        specs[f"{prefix}.indexer.tail_cache"] = CircularBufferSpec(
            block_size=8,
            num_kv_heads=2,
            head_size=128,
            head_size_v=0,
            dtype=torch.bfloat16,
            dcp_sharded=False,
            is_draft=is_draft,
        )
    return specs


def _kimilinear_specs(tp: int) -> dict[str, KVCacheSpec]:
    """Kimi-Linear-48B, ngram k=3: 20 KDA (mamba align), 7 MLA."""
    specs: dict[str, KVCacheSpec] = {}
    kda = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22, 24, 25]
    for i in range(27):
        name = f"model.layers.{i}.self_attn"
        if i in kda:
            specs[name] = MambaSpec(
                block_size=16,
                shapes=((6, 12288 // tp), (32 // tp, 128, 128)),
                dtypes=(torch.bfloat16, torch.float32),
                mamba_cache_mode="align",
                num_speculative_blocks=3,
            )
        else:
            specs[name] = MLAAttentionSpec(
                block_size=16,
                num_kv_heads=1,
                max_tp_shards=1,
                head_size=576,
                dtype=torch.bfloat16,
                block_stride_alignment=MultipleOf(64),
                kernel_block_sizes=(32, 64),
            )
    return specs


def _qwen38_specs(tp: int) -> dict[str, KVCacheSpec]:
    """Qwen3.8-Flash-Next, MTP k=1: 36 GDN + 1 PLE (mamba align), 13 QSA sets
    (full attention + raw ring + compressed, tokens/state 4)."""
    specs: dict[str, KVCacheSpec] = {}
    for i in range(48):
        if (i + 1) % 4 == 0:
            continue
        if i == 1:
            specs[f"model.layers.{i}.ple"] = MambaSpec(
                block_size=16,
                shapes=((10, 10240),),
                dtypes=(torch.bfloat16,),
                mamba_cache_mode="align",
                num_speculative_blocks=1,
                prefill_checkpoint_alignment=1,
                tp_replicated=True,
            )
        specs[f"model.layers.{i}.linear_attn"] = MambaSpec(
            block_size=16,
            shapes=((4, 10240 // tp), (48 // tp, 128, 128)),
            dtypes=(torch.bfloat16, torch.float32),
            mamba_cache_mode="align",
            num_speculative_blocks=1,
            prefill_checkpoint_alignment=64,
        )
    prefixes = [f"model.layers.{i}.self_attn" for i in range(3, 48, 4)] + [
        "mtp.layers.48.self_attn"
    ]
    for prefix in prefixes:
        is_draft = prefix.startswith("mtp.")
        specs[f"{prefix}.attn"] = FullAttentionSpec(
            block_size=16,
            num_kv_heads=max(2 // tp, 1),
            head_size=256,
            head_size_v=256,
            dtype=torch.bfloat16,
            kernel_block_sizes=(MultipleOf(16),),
            is_draft=is_draft,
        )
        specs[f"{prefix}.indexer.raw_key_cache"] = CircularBufferSpec(
            block_size=8,
            num_kv_heads=1,
            head_size=140,
            head_size_v=0,
            dtype=torch.bfloat16,
            is_draft=is_draft,
        )
        specs[f"{prefix}.indexer.compressed_key_cache"] = MLAAttentionSpec(
            block_size=16,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.bfloat16,
            tokens_per_state=4,
            kernel_block_sizes=(MultipleOf(1),),
            is_draft=is_draft,
        )
    return specs


def _minimax_specs(tp: int, draft: str = "mtp") -> dict[str, KVCacheSpec]:
    """MiniMax-M3: 3 dense + 57 sparse full attention, 57 indexer; the MTP
    draft merges into the target groups, the 64-head EAGLE3 draft does not."""
    kv_heads = max(1, 4 // tp)
    specs: dict[str, KVCacheSpec] = {}
    for i in range(60):
        name = f"language_model.model.layers.{i}.self_attn.attn"
        if i < 3:
            specs[name] = FullAttentionSpec(
                block_size=16,
                num_kv_heads=kv_heads,
                head_size=128,
                head_size_v=128,
                dtype=torch.bfloat16,
                kernel_block_sizes=(MultipleOf(16),),
            )
        else:
            specs[name] = FullAttentionSpec(
                block_size=16,
                num_kv_heads=kv_heads,
                head_size=128,
                head_size_v=128,
                dtype=torch.bfloat16,
                block_stride_alignment=MultipleOf(128),
                kernel_block_sizes=(128,),
            )
            specs[f"{name}.index_cache"] = MLAAttentionSpec(
                block_size=16,
                num_kv_heads=1,
                head_size=128,
                dtype=torch.bfloat16,
                cache_role=SparseCacheRole.INDEXER,
                block_stride_alignment=MultipleOf(128),
                kernel_block_sizes=(128,),
            )
    if draft == "mtp":
        specs["draft.model.layers.0.self_attn.attn"] = FullAttentionSpec(
            block_size=16,
            num_kv_heads=kv_heads,
            head_size=128,
            head_size_v=128,
            dtype=torch.bfloat16,
            block_stride_alignment=MultipleOf(128),
            kernel_block_sizes=(128,),
            is_draft=True,
        )
        specs["draft.model.layers.0.self_attn.attn.index_cache"] = MLAAttentionSpec(
            block_size=16,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.bfloat16,
            cache_role=SparseCacheRole.INDEXER,
            block_stride_alignment=MultipleOf(128),
            kernel_block_sizes=(128,),
            is_draft=True,
        )
    else:
        specs["draft.model.layers.0.self_attn.attn"] = FullAttentionSpec(
            block_size=16,
            num_kv_heads=64 // tp,
            head_size=128,
            head_size_v=128,
            dtype=torch.bfloat16,
            kernel_block_sizes=(MultipleOf(16),),
            is_draft=True,
        )
    return specs


def _large_page_group_table(
    config: KVCacheConfig,
) -> dict[tuple, KVCacheGroupSpec]:
    """(spec class, layers, block size, tokens/state, head size) -> group."""
    table = {}
    for group in config.kv_cache_groups:
        spec = group.kv_cache_spec
        key = (
            type(spec).__name__,
            len(group.layer_names),
            spec.block_size,
            str(getattr(spec, "tokens_per_state", None)),
            getattr(spec, "head_size", None),
        )
        assert key not in table, f"ambiguous group key {key}"
        table[key] = group
    return table


def _check_large_page_config(
    vllm_config: VllmConfig,
    config: KVCacheConfig,
    expected_block_size: int,
    expected_num_pages: int,
    expected_page_bytes: int,
    # group key -> (slots per large page, kernel rows per block, row bytes)
    expected_groups: dict[tuple, tuple[int, int, int]],
) -> None:
    block_size = vllm_config.cache_config.block_size
    assert block_size == expected_block_size
    assert config.block_size == block_size
    assert config.num_pages == expected_num_pages
    assert config.num_pages == expected_num_pages
    assert config.large_page_bytes == expected_page_bytes
    assert len(config.kv_cache_groups) == len(expected_groups)
    table = _large_page_group_table(config)
    for key, (n_g, rows_per_block, row_bytes) in expected_groups.items():
        group = table[key]
        assert group.blocks_per_page == n_g, key
        assert group.kernel_rows_per_block == rows_per_block, key
        assert group.row_bytes == row_bytes, key
        assert group.num_blocks == (
            expected_num_pages * expected_page_bytes // (rows_per_block * row_bytes)
        ), key
    # One tensor per layer, kernel-block (row) granularity, over the pool.
    num_layers = sum(len(g.layer_names) for g in config.kv_cache_groups)
    assert len(config.kv_cache_tensors) == num_layers
    group_of = {
        name: group for group in config.kv_cache_groups for name in group.layer_names
    }
    for tensor in config.kv_cache_tensors:
        (layer,) = tensor.layers
        group = group_of[layer]
        assert group.layer_kv_cache_specs is not None
        alignment = group.layer_kv_cache_specs[layer].get_block_stride_alignment()
        assert tensor.size == expected_num_pages * expected_page_bytes
        assert tensor.block_stride == group.row_bytes
        assert tensor.offset % alignment == 0


def test_large_page_dsv41():
    """DSV4.1-Flash (dspark k=5): B=128, 6 groups, SWA keeps its private 32."""
    vllm_config = _large_page_vllm_config()
    specs = _dsv41_specs()
    (config,) = get_kv_cache_configs(vllm_config, [specs], [_LARGE_PAGE_BUDGET])
    _check_large_page_config(
        vllm_config,
        config,
        expected_block_size=128,
        expected_num_pages=29558,
        expected_page_bytes=2179584,
        expected_groups={
            ("SlidingWindowMLASpec", 43, 32, "1", 512): (3, 1, 726528),
            ("MLAAttentionSpec", 3, 128, "2", 512): (21, 1, 101376),
            ("MLAAttentionSpec", 3, 128, "2", 132): (82, 1, 26112),
            ("CircularBufferSpec", 3, 8, "1", 1024): (21, 1, 98304),
            ("MLAAttentionSpec", 1, 128, "1", 512): (31, 1, 67584),
            ("MLAAttentionSpec", 1, 128, "1", 132): (129, 1, 16896),
        },
    )


def test_large_page_dsv41_pp2():
    """PP2 (split at layer 20): groups are global; a stage may hold zero layers
    of a group. Block ids must name the same page on both ranks, so the page
    size and every group's row are identical across ranks."""
    vllm_config = _large_page_vllm_config()
    specs = _dsv41_specs()

    def stage_of(name):
        idx = int(name.split("layers.")[1].split(".")[0])
        return 0 if idx < 20 else 1

    stages = [{n: s for n, s in specs.items() if stage_of(n) == r} for r in range(2)]
    configs = get_kv_cache_configs(vllm_config, stages, [_LARGE_PAGE_BUDGET] * 2)
    assert len(configs) == 2
    assert configs[0].num_pages == configs[1].num_pages == 27630
    assert configs[0].large_page_bytes == configs[1].large_page_bytes
    # Every layer is on exactly one rank.
    assert sum(len(g.layer_names) for g in configs[0].kv_cache_groups) + sum(
        len(g.layer_names) for g in configs[1].kv_cache_groups
    ) == len(specs)
    for group0, group1 in zip(configs[0].kv_cache_groups, configs[1].kv_cache_groups):
        assert group0.blocks_per_page == group1.blocks_per_page
        assert group0.kernel_rows_per_block == group1.kernel_rows_per_block
        assert group0.num_blocks == group1.num_blocks
        assert group0.row_bytes == group1.row_bytes > 0
    swa0 = configs[0].kv_cache_groups[0]
    assert len(swa0.layer_names) == 20
    # The SWA row is its 23-layer row on rank 1; six of them fill a page.
    assert swa0.blocks_per_page == 6


@pytest.mark.parametrize(
    "tp, expected",
    [
        (1, (431, 149291008, 1, 149291008, 189, 2944, 3036)),
        # TP4: a page of one KDA slot wastes >2% on the sparse-MLA row, so the
        # page holds 2 KDA slots.
        (4, (863, 74645504, 2, 37322752, 94, 1471, 1518)),
    ],
)
def test_large_page_glm53(tp, expected):
    """GLM-5.3-Flash (MTP k=1): B=128, 4 groups; the KDA mamba group rides at B
    and mamba_block_size is written back."""
    num_pages, page_bytes, n_kda, kda_row, n_mla, n_idx, n_ring = expected
    vllm_config = _large_page_vllm_config(mamba_align=True)
    specs = _glm53_specs(tp)
    (config,) = get_kv_cache_configs(vllm_config, [specs], [_LARGE_PAGE_BUDGET])
    assert vllm_config.cache_config.mamba_block_size == 128
    _check_large_page_config(
        vllm_config,
        config,
        expected_block_size=128,
        expected_num_pages=num_pages,
        expected_page_bytes=page_bytes,
        expected_groups={
            ("MambaSpec", 34, 128, "-1", None): (n_kda, 1, kda_row),
            ("MLAAttentionSpec", 12, 128, "1", 512): (n_mla, 1, 786432),
            ("MLAAttentionSpec", 12, 128, "4", 132): (n_idx, 1, 50688),
            ("CircularBufferSpec", 12, 8, "1", 128): (n_ring, 1, 49152),
        },
    )
    # KDA checkpoints (raw alignment 16) stay on the B-token prefix-cache grid.
    (kda,) = [
        g for g in config.kv_cache_groups if isinstance(g.kv_cache_spec, MambaSpec)
    ]
    assert kda.kv_cache_spec.prefill_checkpoint_alignment == 128


@pytest.mark.parametrize(
    "tp, expected",
    [
        (1, (1435, 44892160, 1, 44892160, 86)),
        # TP4 quarters the KDA row; three KDA slots per page keep the
        # demand-weighted page waste under 2%.
        (4, (1913, 33669120, 3, 11223040, 64)),
    ],
)
def test_large_page_kimilinear(tp, expected):
    """Kimi-Linear-48B (ngram k=3): B=64, the default floor; the KDA checkpoint
    alignment (16) and the MLA kernel blocks (32, 64) both divide it."""
    num_pages, page_bytes, n_kda, kda_row, n_mla = expected
    vllm_config = _large_page_vllm_config(model=_LARGE_PAGE_KIMI, mamba_align=True)
    assert vllm_config.model_config.use_mla
    specs = _kimilinear_specs(tp)
    (config,) = get_kv_cache_configs(vllm_config, [specs], [_LARGE_PAGE_BUDGET])
    _check_large_page_config(
        vllm_config,
        config,
        expected_block_size=64,
        expected_num_pages=num_pages,
        expected_page_bytes=page_bytes,
        expected_groups={
            ("MambaSpec", 20, 64, "-1", None): (n_kda, 1, kda_row),
            ("MLAAttentionSpec", 7, 64, "1", 576): (n_mla, 1, 516096),
        },
    )


@pytest.mark.parametrize(
    "tp, expected",
    [
        (1, (554, 116195328, (1, 116195328), 566, (67, 1703936), 3989, 2181)),
        # TP4 quarters the GDN row; two GDN slots per page.
        (4, (1108, 58097664, (2, 29048832), 282, (67, 851968), 1994, 1090)),
    ],
)
def test_large_page_qwen38(tp, expected):
    """Qwen3.8-Flash-Next (MTP k=1): B=64 from the raw GDN prefill-checkpoint
    alignment; the TP-replicated PLE group gets its own placement."""
    num_pages, page_bytes, (n_gdn, gdn_row), n_ple, main, n_ring, n_comp = expected
    vllm_config = _large_page_vllm_config(mamba_align=True)
    specs = _qwen38_specs(tp)
    (config,) = get_kv_cache_configs(vllm_config, [specs], [_LARGE_PAGE_BUDGET])
    _check_large_page_config(
        vllm_config,
        config,
        expected_block_size=64,
        expected_num_pages=num_pages,
        expected_page_bytes=page_bytes,
        expected_groups={
            ("MambaSpec", 36, 64, "-1", None): (n_gdn, 1, gdn_row),
            ("MambaSpec", 1, 64, "-1", None): (n_ple, 1, 204800),
            ("FullAttentionSpec", 13, 64, "1", 256): (main[0], 1, main[1]),
            ("CircularBufferSpec", 13, 8, "1", 140): (n_ring, 1, 29120),
            ("MLAAttentionSpec", 13, 64, "4", 128): (n_comp, 1, 53248),
        },
    )


@pytest.mark.parametrize(
    "tp, draft, expected",
    [
        # MTP: the draft layers merge into the target groups (2 groups).
        (1, "mtp", (2014, 31981568, (2, 61, 15990784), (15, 58))),
        (4, "mtp", (1611, 39976960, (10, 61, 3997696), (20, 58))),
        # EAGLE3: the 64-head draft layer is its own group; 4 main slots a page.
        (1, "eagle3", (1024, 62914560, (4, 60, 15728640), (32, 57))),
    ],
)
def test_large_page_minimax(tp, draft, expected):
    """MiniMax-M3: B=128 forced by exact kernel block sizes ([128]); dense and
    sparse full-attention layers merge into one group despite different
    block-stride alignments."""
    num_pages, page_bytes, (n_main, main_layers, main_row), (n_idx, idx_layers) = (
        expected
    )
    vllm_config = _large_page_vllm_config()
    specs = _minimax_specs(tp, draft)
    (config,) = get_kv_cache_configs(vllm_config, [specs], [_LARGE_PAGE_BUDGET])
    expected_groups = {
        ("FullAttentionSpec", main_layers, 128, "1", 128): (n_main, 1, main_row),
        ("MLAAttentionSpec", idx_layers, 128, "1", 128): (
            n_idx,
            1,
            idx_layers * 32768,
        ),
    }
    if draft == "eagle3":
        expected_groups[("FullAttentionSpec", 1, 128, "1", 128)] = (15, 1, 4194304)
    _check_large_page_config(
        vllm_config,
        config,
        expected_block_size=128,
        expected_num_pages=num_pages,
        expected_page_bytes=page_bytes,
        expected_groups=expected_groups,
    )


def test_large_page_splits_layers_with_different_kernel_blocks():
    """Layers whose backends resolve to different kernel blocks at B get
    separate groups, since a group has one row layout (MiniMax-M3: dense
    layers support kernel blocks up to 64, sparse layers only 128)."""
    vllm_config = _large_page_vllm_config()

    def layer(kernel_block_sizes):
        return FullAttentionSpec(
            block_size=16,
            num_kv_heads=4,
            head_size=128,
            dtype=torch.bfloat16,
            kernel_block_sizes=kernel_block_sizes,
        )

    specs = {
        **{f"dense.{i}": layer((16, 32, 64)) for i in range(3)},
        **{f"sparse.{i}": layer((128,)) for i in range(5)},
    }
    (config,) = get_kv_cache_configs(vllm_config, [specs], [_LARGE_PAGE_BUDGET])
    assert vllm_config.cache_config.block_size == 128
    rows = {
        tuple(group.layer_names): group.kernel_rows_per_block
        for group in config.kv_cache_groups
    }
    assert rows == {
        tuple(f"dense.{i}" for i in range(3)): 2,
        tuple(f"sparse.{i}" for i in range(5)): 1,
    }


def test_large_page_addressing_invariants():
    """Every block a page holds lies inside the page and is disjoint from the
    blocks of other pages under arbitrary page ownership; every page holds at
    least blocks_per_page blocks."""
    vllm_config = _large_page_vllm_config()
    specs = _dsv41_specs()
    (config,) = get_kv_cache_configs(vllm_config, [specs], [_LARGE_PAGE_BUDGET])
    num_pages = config.num_pages
    page_bytes = config.large_page_bytes

    def page_blocks(group_id: int, page: int) -> list[tuple[int, int]]:
        block_bytes = config.group_geometry(group_id).block_bytes
        return [
            (block_id * block_bytes, (block_id + 1) * block_bytes)
            for block_id in large_page_block_range(page, page_bytes, block_bytes)
        ]

    rng = np.random.default_rng(20261022)
    num_groups = len(config.kv_cache_groups)
    for _ in range(30):
        owners = rng.integers(0, num_groups, size=num_pages)
        intervals: list[tuple[int, int]] = []
        for page in rng.choice(np.arange(1, num_pages), size=64, replace=False):
            group_id = int(owners[page])
            blocks = page_blocks(group_id, int(page))
            assert len(blocks) >= config.kv_cache_groups[group_id].blocks_per_page
            for start, end in blocks:
                assert page * page_bytes <= start
                assert end <= (page + 1) * page_bytes
            intervals.extend(blocks)
        ordered = sorted(intervals)
        for (_, prev_end), (start, _) in zip(ordered, ordered[1:]):
            assert prev_end <= start

    # The per-layer views cover the group's highest block.
    for group in config.kv_cache_groups:
        num_kernel_blocks = num_pages * page_bytes // group.row_bytes
        assert group.num_blocks * group.kernel_rows_per_block <= num_kernel_blocks


@pytest.mark.parametrize("max_model_len", [1024, 131072])
def test_large_page_placement_independent_of_max_model_len(max_model_len):
    """n_g, S and N come from per-token/per-request rates only and must not
    move with max_model_len (auto-fit mutates it during its search)."""
    vllm_config = _large_page_vllm_config(
        model=_LARGE_PAGE_KIMI, mamba_align=True, max_model_len=max_model_len
    )
    specs = _kimilinear_specs(1)
    (config,) = get_kv_cache_configs(vllm_config, [specs], [_LARGE_PAGE_BUDGET])
    assert config.num_pages == 1435
    assert config.large_page_bytes == 44892160
    assert [g.blocks_per_page for g in config.kv_cache_groups] == [1, 86]


def test_large_page_user_block_size():
    """A user-set --block-size must be feasible as-is, else a ValueError
    naming the violated constraint."""
    specs = _dsv41_specs()
    vllm_config = _large_page_vllm_config()
    vllm_config.cache_config = CacheConfig(block_size=128)
    (config,) = get_kv_cache_configs(vllm_config, [specs], [_LARGE_PAGE_BUDGET])
    assert config.block_size == 128

    vllm_config = _large_page_vllm_config()
    vllm_config.cache_config = CacheConfig(block_size=64)
    with pytest.raises(ValueError, match="Invalid block_size=64"):
        get_kv_cache_configs(vllm_config, [specs], [_LARGE_PAGE_BUDGET])


def test_large_page_block_size_fits_dcp_sharded_kernel_blocks():
    """A DCP-sharded layer holds B / dcp tokens per rank, so its kernel block
    sizes constrain B / dcp, not B."""
    vllm_config = _large_page_vllm_config()
    vllm_config.parallel_config.decode_context_parallel_size = 2
    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=64,
        dtype=torch.bfloat16,
        kernel_block_sizes=(64,),
    )
    assert select_global_block_size(vllm_config, {"layer": spec}) == 128


def test_large_page_num_gpu_blocks_override():
    """num_gpu_blocks_override overrides N (large pages) directly."""
    vllm_config = _large_page_vllm_config(model=_LARGE_PAGE_KIMI, mamba_align=True)
    vllm_config.cache_config = CacheConfig(
        mamba_cache_mode="align",
        num_gpu_blocks_override=1000,
    )
    (config,) = get_kv_cache_configs(
        vllm_config, [_kimilinear_specs(1)], [_LARGE_PAGE_BUDGET]
    )
    assert config.num_pages == 1000
    assert config.num_pages == 1000
    assert config.kv_cache_groups[0].num_blocks == 1000
    assert config.kv_cache_groups[1].num_blocks == (
        1000 * config.large_page_bytes // config.group_geometry(1).block_bytes
    )


def test_large_page_sizer_rejects_tiny_pool():
    """One max-model-len request must fit: sum_g ceil(demand_g / n_g) <= N-1."""
    vllm_config = _large_page_vllm_config()
    vllm_config.cache_config = CacheConfig(num_gpu_blocks_override=4)
    with pytest.raises(ValueError, match="To serve at least one request"):
        get_kv_cache_configs(vllm_config, [_dsv41_specs()], [_LARGE_PAGE_BUDGET])


def test_large_page_attention_free():
    vllm_config = _large_page_vllm_config()
    (config,) = get_kv_cache_configs(vllm_config, [{}], [_LARGE_PAGE_BUDGET])
    assert config.num_pages == 1
    assert config.kv_cache_groups == []
    assert config.kv_cache_tensors == []


def test_large_page_none_mode_mamba_scheduler_block_size():
    """Without prefix caching a Mamba group keeps a private max_model_len
    block; the scheduler block size must still cover every group (the
    coordinator asserts it), while block hashes stay on the B grid."""
    vllm_config = _large_page_vllm_config(model=_LARGE_PAGE_KIMI, max_model_len=8192)
    vllm_config.cache_config.enable_prefix_caching = False
    specs = {
        name: replace(spec, block_size=8192, mamba_cache_mode="none")
        if isinstance(spec, MambaSpec)
        else spec
        for name, spec in _kimilinear_specs(4).items()
    }
    (config,) = get_kv_cache_configs(vllm_config, [specs], [_LARGE_PAGE_BUDGET])

    assert kv_cache_utils.resolve_scheduler_block_size(config, vllm_config) == 8192


def test_large_page_disable_hybrid_promotes_sliding_window():
    """--disable-hybrid-kv-cache-manager keeps its meaning:
    sliding-window layers are allocated as full attention, in one group."""
    vllm_config = _large_page_vllm_config()
    vllm_config.scheduler_config.disable_hybrid_kv_cache_manager = True
    attn = dict(num_kv_heads=8, head_size=128, dtype=torch.bfloat16, block_size=16)
    specs = {
        "full": FullAttentionSpec(**attn),
        "swa": SlidingWindowSpec(**attn, sliding_window=1024),
    }
    (config,) = get_kv_cache_configs(vllm_config, [specs], [_LARGE_PAGE_BUDGET])

    (group,) = config.kv_cache_groups
    assert type(group.kv_cache_spec) is FullAttentionSpec
    assert group.layer_names == ["full", "swa"]


def test_large_page_disable_hybrid_rejects_multi_group_model():
    """Promoted layers whose pages differ cannot share a large-page group, so
    the flag is an error rather than a silently multi-group plan."""
    vllm_config = _large_page_vllm_config()
    vllm_config.scheduler_config.disable_hybrid_kv_cache_manager = True
    attn = dict(num_kv_heads=8, dtype=torch.bfloat16, block_size=16)
    specs = {
        "full": FullAttentionSpec(**attn, head_size=128),
        "swa": SlidingWindowSpec(**attn, head_size=64, sliding_window=1024),
    }
    with pytest.raises(ValueError, match="one KV cache group"):
        get_kv_cache_configs(vllm_config, [specs], [_LARGE_PAGE_BUDGET])


def test_large_page_profiling_placement():
    """The worker-side profiling path plans a local placement through the
    builder with enough blocks of every group for min_blocks sequences."""
    vllm_config = _large_page_vllm_config()
    specs = _kimilinear_specs(1)
    builder = get_kv_cache_config_builder(vllm_config)
    groups = get_kv_cache_groups_from_builder(builder, vllm_config, dict(specs))
    config = builder.get_kv_cache_config_from_groups(vllm_config, groups, 64)
    assert config.num_pages >= 64
    for group in config.kv_cache_groups:
        assert group.num_blocks >= 64
    assert len(config.kv_cache_tensors) == 27
