# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import mmap
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
import vllm.v1.hisparse.runtime as hisparse_runtime_module
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
from vllm.config.attention import HiSparseConfig
from vllm.utils.mem_constants import GiB_bytes
from vllm.v1.attention.backend import MultipleOf
from vllm.v1.core.kv_cache_config_builder import get_kv_cache_config_builder
from vllm.v1.core.kv_cache_planning import (
    check_enough_kv_cache_memory,
    estimate_max_model_len,
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
from vllm.v1.hisparse.layout import (
    get_hisparse_kv_cache_groups,
)
from vllm.v1.kv_cache_interface import (
    ChunkedLocalAttentionSpec,
    CircularBufferSpec,
    FullAttentionSpec,
    HiddenStateCacheSpec,
    HiSparseHotSpec,
    HiSparseResidentSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVCacheTensor,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    SlidingWindowSpec,
    SparseCacheRole,
    UniformTypeKVCacheSpecs,
    get_kv_quant_mode,
    large_page_block_range,
)
from vllm.v1.kv_cache_layout import KVCacheLayout

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize("gpu_block_size", [32, 64])
@pytest.mark.parametrize("shared_host_pool", [False, True])
def test_hisparse_hma_uses_resolved_gpu_block_size(
    monkeypatch, gpu_block_size, shared_host_pool
):
    monkeypatch.setattr(
        hisparse_runtime_module.current_platform, "is_cuda_alike", lambda: True
    )
    specs = {
        "model.layers.0.self_attn": MLAAttentionSpec(
            block_size=gpu_block_size,
            num_kv_heads=1,
            head_size=576,
            dtype=torch.bfloat16,
            is_index_group_leader=True,
        ),
        "model.layers.0.self_attn.indexer": MLAAttentionSpec(
            block_size=gpu_block_size,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.bfloat16,
            cache_role=SparseCacheRole.INDEXER,
        ),
    }
    config = SimpleNamespace(
        attention_config=SimpleNamespace(hisparse_config=HiSparseConfig()),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(index_topk=128),
            max_model_len=gpu_block_size,
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=2 if shared_host_pool else 1,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            world_size=2 if shared_host_pool else 1,
            distributed_executor_backend="mp",
            nnodes_within_dp=1,
        ),
        cache_config=SimpleNamespace(
            num_gpu_blocks_override=7,
            prefix_cache_retention_interval=None,
            get_resolved_kv_cache_layout=lambda: KVCacheLayout.BLHNC,
        ),
    )
    monkeypatch.setattr(
        kv_cache_planning, "get_hisparse_host_pool_bytes", lambda _: 2**30
    )
    cache_config = kv_cache_planning.get_kv_cache_config_from_groups(
        config, get_hisparse_kv_cache_groups(config, specs), available_memory=2**30
    )
    assert cache_config.num_blocks == 7
    assert cache_config.hisparse_host_num_blocks is not None
    assert cache_config.hisparse_host_num_blocks > 7

    host_group, indexer_group, *auxiliary_groups = cache_config.kv_cache_groups
    assert cache_config.hisparse_shared_host_pool is shared_host_pool
    host_page = host_group.kv_cache_spec.page_size_bytes
    alignment = mmap.PAGESIZE if shared_host_pool else 1
    expected_host_stride = (host_page + alignment - 1) // alignment * alignment
    assert cache_config.hisparse_host_block_stride == expected_host_stride
    assert cache_config.hisparse_host_num_blocks == 2**30 // expected_host_stride
    assert host_group.host_resident
    assert not any(group.host_resident for group in [indexer_group, *auxiliary_groups])
    host_layers = set(host_group.layer_names)
    for tensor in cache_config.kv_cache_tensors:
        assert all(
            (name in host_layers) == tensor.host_resident for name in tensor.layers
        )
    host_tensor = next(t for t in cache_config.kv_cache_tensors if t.host_resident)
    host_spec = host_group.kv_cache_spec.kv_cache_specs[host_tensor.layers[0]]
    assert host_tensor.block_stride == host_spec.page_size_bytes
    assert host_tensor.layer_stride == (
        host_spec.page_size_bytes * cache_config.hisparse_host_num_blocks
    )
    assert host_group.kv_cache_spec.block_size == gpu_block_size
    assert indexer_group.kv_cache_spec.block_size == gpu_block_size
    host_specs = host_group.kv_cache_spec.kv_cache_specs
    gpu_indexer_specs = indexer_group.kv_cache_spec.kv_cache_specs
    assert set(host_specs) == {"model.layers.0.self_attn"}
    assert set(gpu_indexer_specs) == {"model.layers.0.self_attn.indexer"}
    assert indexer_group.kv_cache_spec.prefix_cacheable
    assert host_group.enable_kv_transfer
    auxiliary_specs = [group.kv_cache_spec for group in auxiliary_groups]
    assert any(isinstance(spec, HiSparseResidentSpec) for spec in auxiliary_specs)
    assert any(isinstance(spec, HiSparseHotSpec) for spec in auxiliary_specs)
    assert all(
        spec.block_size == gpu_block_size
        for spec in auxiliary_specs
        if isinstance(spec, (HiSparseResidentSpec, HiSparseHotSpec))
    )
    scheduler_block_size, hash_block_size = kv_cache_utils.resolve_kv_cache_block_sizes(
        cache_config,
        SimpleNamespace(
            cache_config=SimpleNamespace(
                block_size=16,
                enable_prefix_caching=True,
                prefix_match_unit=None,
            ),
            parallel_config=SimpleNamespace(decode_context_parallel_size=1),
            kv_transfer_config=object(),
        ),
    )
    assert scheduler_block_size == hash_block_size == gpu_block_size


@pytest.mark.parametrize(
    "max_model_len,num_gpu_blocks,ok",
    [
        (8192, 160, False),
        (8192, 1000, True),
        (32768, 656, False),
        (32768, 657, True),
        (-1, 657, True),
    ],
)
def test_hisparse_pool_must_fit_max_model_len(
    monkeypatch, max_model_len, num_gpu_blocks, ok
):
    """Each resident group takes its own blocks from HiSparse's shared GPU pool,
    so a pool that fits only the indexer pages must be rejected at startup.
    Resident pages are capped at the in-flight window: at 32768 tokens with 2048
    in flight, one request needs 512 indexer + 4 x 32 resident + 16 hot blocks,
    plus the null block. Auto-fit (-1) must pick a length that passes the same
    check."""
    monkeypatch.setattr(
        hisparse_runtime_module.current_platform, "is_cuda_alike", lambda: True
    )
    monkeypatch.setattr(
        kv_cache_planning, "get_hisparse_host_pool_bytes", lambda _: 2**30
    )
    specs: dict[str, KVCacheSpec] = {}
    for i in range(4):
        specs[f"model.layers.{i}.self_attn"] = MLAAttentionSpec(
            block_size=64,
            num_kv_heads=1,
            head_size=576,
            dtype=torch.bfloat16,
            is_index_group_leader=True,
        )
        specs[f"model.layers.{i}.self_attn.indexer"] = MLAAttentionSpec(
            block_size=64,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.bfloat16,
            cache_role=SparseCacheRole.INDEXER,
        )
    config = VllmConfig(model_config=ModelConfig(max_model_len=max_model_len))
    config.scheduler_config.max_num_batched_tokens = 2048
    # Async scheduling doubles the in-flight window, and with it the bound.
    config.scheduler_config.async_scheduling = False
    config.attention_config.hisparse_config = HiSparseConfig()
    config.model_config.hf_config.index_topk = 128
    config.cache_config.num_gpu_blocks_override = num_gpu_blocks
    config.cache_config.kv_cache_layout = "BLHNC"
    # Workers report the resident/hot caches HiSparse derives.
    specs |= {
        name: group.kv_cache_spec
        for group in get_hisparse_kv_cache_groups(config, specs) or []
        if isinstance(group.kv_cache_spec, (HiSparseResidentSpec, HiSparseHotSpec))
        for name in group.layer_names
    }

    if ok:
        kv_cache_planning.get_kv_cache_configs(config, [specs], [2**34])
    else:
        with pytest.raises(ValueError, match="max seq len"):
            kv_cache_planning.get_kv_cache_configs(config, [specs], [2**34])


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


def test_get_kv_cache_configs_multiple_workers():
    model_config = ModelConfig(max_model_len=16)
    vllm_config = VllmConfig(model_config=model_config)
    vllm_config.cache_config.kv_cache_layout = "LBNHC"
    vllm_config.cache_config.prefix_cache_retention_interval = None

    ref_kv_cache_spec = new_kv_cache_spec()
    same_kv_cache_specs = [
        {
            "layer1": new_kv_cache_spec(),
            "layer2": new_kv_cache_spec(),
        },
        {
            "layer1": new_kv_cache_spec(),
            "layer2": new_kv_cache_spec(),
        },
    ]

    # Basic case. All things are the same.
    kv_cache_configs = get_kv_cache_configs(
        vllm_config,
        same_kv_cache_specs,
        [
            ref_kv_cache_spec.page_size_bytes * 2 * 10,
            ref_kv_cache_spec.page_size_bytes * 2 * 10,
        ],
    )
    expected = KVCacheConfig(
        num_blocks=10,
        kv_cache_tensors=[
            KVCacheTensor(
                size=ref_kv_cache_spec.page_size_bytes * 10 * 2,
                layers=["layer1", "layer2"],
                layer_stride=ref_kv_cache_spec.page_size_bytes * 10,
                block_stride=ref_kv_cache_spec.page_size_bytes,
            ),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(["layer1", "layer2"], ref_kv_cache_spec),
        ],
    )
    assert kv_cache_configs == [expected, expected]

    # Different available memory. This is the case for TP.
    # Use the smallest memory available.
    kv_cache_configs = get_kv_cache_configs(
        vllm_config,
        same_kv_cache_specs,
        [
            ref_kv_cache_spec.page_size_bytes * 2 * 10,
            ref_kv_cache_spec.page_size_bytes * 2 * 20,
        ],
    )
    assert kv_cache_configs == [expected, expected]

    # Different KV cache specs. This is the case for PP.
    different_layer_specs = [
        {
            "layer1": new_kv_cache_spec(),
        },
        {
            "layer2": new_kv_cache_spec(),
            "layer3": new_kv_cache_spec(),
        },
    ]

    # Different workers have different layers.
    kv_cache_configs = get_kv_cache_configs(
        vllm_config,
        different_layer_specs,
        [
            ref_kv_cache_spec.page_size_bytes * 2 * 10,
            ref_kv_cache_spec.page_size_bytes * 2 * 10,
        ],
    )
    assert kv_cache_configs == [
        KVCacheConfig(
            num_blocks=10,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=ref_kv_cache_spec.page_size_bytes * 10,
                    layers=["layer1"],
                    layer_stride=ref_kv_cache_spec.page_size_bytes * 10,
                    block_stride=ref_kv_cache_spec.page_size_bytes,
                ),
            ],
            kv_cache_groups=[
                KVCacheGroupSpec(["layer1"], new_kv_cache_spec()),
            ],
        ),
        KVCacheConfig(
            num_blocks=10,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=ref_kv_cache_spec.page_size_bytes * 10 * 2,
                    layers=["layer2", "layer3"],
                    layer_stride=ref_kv_cache_spec.page_size_bytes * 10,
                    block_stride=ref_kv_cache_spec.page_size_bytes,
                ),
            ],
            kv_cache_groups=[
                KVCacheGroupSpec(["layer2", "layer3"], new_kv_cache_spec()),
            ],
        ),
    ]

    # Some layers are the same, some are different. This is the case for TP+PP
    tp_pp_kv_cache_specs = [
        {
            "layer1": new_kv_cache_spec(),
            "layer2": new_kv_cache_spec(),
        },
        {
            "layer1": new_kv_cache_spec(),
            "layer2": new_kv_cache_spec(),
        },
        {
            "layer3": new_kv_cache_spec(),
        },
        {
            "layer3": new_kv_cache_spec(),
        },
    ]

    kv_cache_configs = get_kv_cache_configs(
        vllm_config,
        tp_pp_kv_cache_specs,
        [
            ref_kv_cache_spec.page_size_bytes * 2 * 10,
            ref_kv_cache_spec.page_size_bytes * 2 * 10,
            ref_kv_cache_spec.page_size_bytes * 2 * 10,
            ref_kv_cache_spec.page_size_bytes * 2 * 10,
        ],
    )
    expected_12 = KVCacheConfig(
        num_blocks=10,
        kv_cache_tensors=[
            KVCacheTensor(
                size=ref_kv_cache_spec.page_size_bytes * 10 * 2,
                layers=["layer1", "layer2"],
                layer_stride=ref_kv_cache_spec.page_size_bytes * 10,
                block_stride=ref_kv_cache_spec.page_size_bytes,
            ),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(["layer1", "layer2"], ref_kv_cache_spec),
        ],
    )
    expected_3 = KVCacheConfig(
        num_blocks=10,
        kv_cache_tensors=[
            KVCacheTensor(
                size=ref_kv_cache_spec.page_size_bytes * 10,
                layers=["layer3"],
                layer_stride=ref_kv_cache_spec.page_size_bytes * 10,
                block_stride=ref_kv_cache_spec.page_size_bytes,
            ),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(["layer3"], ref_kv_cache_spec),
        ],
    )
    assert kv_cache_configs == [
        expected_12,
        expected_12,
        expected_3,
        expected_3,
    ]

    # Different workers have different types of layers. This is the case for
    # hybrid models + PP.
    different_type_layer_specs = [
        {
            "layer1": new_kv_cache_spec(),
            "layer2": new_kv_cache_spec(),
        },
        {
            "layer3": new_sliding_window_spec(),
            "layer4": new_sliding_window_spec(),
        },
    ]
    kv_cache_configs = get_kv_cache_configs(
        vllm_config,
        different_type_layer_specs,
        [
            ref_kv_cache_spec.page_size_bytes * 2 * 10,
            ref_kv_cache_spec.page_size_bytes * 2 * 10,
        ],
    )
    assert kv_cache_configs == [
        KVCacheConfig(
            num_blocks=10,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=ref_kv_cache_spec.page_size_bytes * 10 * 2,
                    layers=["layer1", "layer2"],
                    layer_stride=ref_kv_cache_spec.page_size_bytes * 10,
                    block_stride=ref_kv_cache_spec.page_size_bytes,
                ),
            ],
            kv_cache_groups=[
                KVCacheGroupSpec(["layer1", "layer2"], ref_kv_cache_spec),
                KVCacheGroupSpec([], new_sliding_window_spec()),
            ],
        ),
        KVCacheConfig(
            num_blocks=10,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=ref_kv_cache_spec.page_size_bytes * 10 * 2,
                    layers=["layer3", "layer4"],
                    layer_stride=ref_kv_cache_spec.page_size_bytes * 10,
                    block_stride=ref_kv_cache_spec.page_size_bytes,
                ),
            ],
            kv_cache_groups=[
                KVCacheGroupSpec([], ref_kv_cache_spec),
                KVCacheGroupSpec(["layer3", "layer4"], new_sliding_window_spec()),
            ],
        ),
    ]

    # When divided into multiple KVCacheGroups, need to ensure the number of
    # layers per group is similar.
    different_type_layer_specs = [
        {
            "layer1": new_kv_cache_spec(),
            "layer2": new_sliding_window_spec(),
            "layer3": new_sliding_window_spec(),
        },
        {
            "layer4": new_kv_cache_spec(),
            "layer5": new_sliding_window_spec(),
            "layer6": new_sliding_window_spec(),
        },
    ]
    kv_cache_configs = get_kv_cache_configs(
        vllm_config,
        different_type_layer_specs,
        [
            ref_kv_cache_spec.page_size_bytes * 10,
            ref_kv_cache_spec.page_size_bytes * 10,
        ],
    )
    assert kv_cache_configs == [
        KVCacheConfig(
            num_blocks=10,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=ref_kv_cache_spec.page_size_bytes * 10,
                    layers=["layer1"],
                    layer_stride=ref_kv_cache_spec.page_size_bytes * 10,
                    block_stride=ref_kv_cache_spec.page_size_bytes,
                ),
                KVCacheTensor(
                    size=ref_kv_cache_spec.page_size_bytes * 10,
                    layers=["layer2"],
                    layer_stride=ref_kv_cache_spec.page_size_bytes * 10,
                    block_stride=ref_kv_cache_spec.page_size_bytes,
                ),
                KVCacheTensor(
                    size=ref_kv_cache_spec.page_size_bytes * 10,
                    layers=["layer3"],
                    layer_stride=ref_kv_cache_spec.page_size_bytes * 10,
                    block_stride=ref_kv_cache_spec.page_size_bytes,
                ),
            ],
            kv_cache_groups=[
                KVCacheGroupSpec(["layer1"], ref_kv_cache_spec),
                KVCacheGroupSpec(["layer2"], new_sliding_window_spec()),
                KVCacheGroupSpec(["layer3"], new_sliding_window_spec()),
            ],
        ),
        KVCacheConfig(
            num_blocks=10,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=ref_kv_cache_spec.page_size_bytes * 10,
                    layers=["layer4"],
                    layer_stride=ref_kv_cache_spec.page_size_bytes * 10,
                    block_stride=ref_kv_cache_spec.page_size_bytes,
                ),
                KVCacheTensor(
                    size=ref_kv_cache_spec.page_size_bytes * 10,
                    layers=["layer5"],
                    layer_stride=ref_kv_cache_spec.page_size_bytes * 10,
                    block_stride=ref_kv_cache_spec.page_size_bytes,
                ),
                KVCacheTensor(
                    size=ref_kv_cache_spec.page_size_bytes * 10,
                    layers=["layer6"],
                    layer_stride=ref_kv_cache_spec.page_size_bytes * 10,
                    block_stride=ref_kv_cache_spec.page_size_bytes,
                ),
            ],
            kv_cache_groups=[
                KVCacheGroupSpec(["layer4"], ref_kv_cache_spec),
                KVCacheGroupSpec(["layer5"], new_sliding_window_spec()),
                KVCacheGroupSpec(["layer6"], new_sliding_window_spec()),
            ],
        ),
    ]

    # Have conflicting layers. Need to raise an error.
    conflicting_layer_specs = [
        {
            "layer1": new_kv_cache_spec(),
        },
        {
            "layer1": new_sliding_window_spec(),
        },
    ]
    with pytest.raises(AssertionError):
        get_kv_cache_configs(
            vllm_config,
            conflicting_layer_specs,
            [
                ref_kv_cache_spec.page_size_bytes * 2 * 10,
                ref_kv_cache_spec.page_size_bytes * 2 * 10,
            ],
        )


@pytest.mark.parametrize(
    "asymmetric_memory",
    [False, True],
    ids=["symmetric", "asymmetric"],
)
def test_get_kv_cache_configs_pp_sharding(asymmetric_memory):
    model_config = ModelConfig(max_model_len=512)
    vllm_config = VllmConfig(model_config=model_config)
    vllm_config.cache_config.kv_cache_layout = "LBNHC"
    vllm_config.cache_config.prefix_cache_retention_interval = None

    ref_kv_cache_spec = new_kv_cache_spec()
    pp_kv_cache_specs = [
        {"layer1": ref_kv_cache_spec},
        {"layer2": ref_kv_cache_spec},
    ]

    expected_num_blocks = model_config.max_model_len // ref_kv_cache_spec.block_size + 1
    avail_memory = ref_kv_cache_spec.page_size_bytes * expected_num_blocks

    # With per-worker validation, each worker only needs memory for its own
    # layers. Worker 2 having more memory shouldn't affect worker 1's config.
    available_memory = (
        [avail_memory, avail_memory * 2] if asymmetric_memory else [avail_memory] * 2
    )

    kv_cache_configs = get_kv_cache_configs(
        vllm_config,
        pp_kv_cache_specs,
        available_memory,
    )

    assert kv_cache_configs == [
        KVCacheConfig(
            num_blocks=expected_num_blocks,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=ref_kv_cache_spec.page_size_bytes * expected_num_blocks,
                    layers=["layer1"],
                    layer_stride=ref_kv_cache_spec.page_size_bytes
                    * expected_num_blocks,
                    block_stride=ref_kv_cache_spec.page_size_bytes,
                ),
            ],
            kv_cache_groups=[KVCacheGroupSpec(["layer1"], ref_kv_cache_spec)],
        ),
        KVCacheConfig(
            num_blocks=expected_num_blocks,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=ref_kv_cache_spec.page_size_bytes * expected_num_blocks,
                    layers=["layer2"],
                    layer_stride=ref_kv_cache_spec.page_size_bytes
                    * expected_num_blocks,
                    block_stride=ref_kv_cache_spec.page_size_bytes,
                ),
            ],
            kv_cache_groups=[KVCacheGroupSpec(["layer2"], ref_kv_cache_spec)],
        ),
    ]


def test_project_kv_cache_groups_to_worker():
    spec_a = new_kv_cache_spec()
    spec_b = new_kv_cache_spec(num_kv_heads=4)

    global_groups = [
        KVCacheGroupSpec(["layer1", "layer2", "layer3"], spec_a),
    ]
    worker_spec = {"layer1": spec_a, "layer2": spec_a}
    projected = kv_cache_planning._project_kv_cache_groups_to_worker(
        global_groups, worker_spec
    )
    assert len(projected) == 1
    assert projected[0].layer_names == ["layer1", "layer2"]
    assert projected[0].kv_cache_spec is spec_a

    projected = kv_cache_planning._project_kv_cache_groups_to_worker(
        global_groups, {"layer4": spec_a}
    )
    assert len(projected) == 1
    assert projected[0].layer_names == []
    assert projected[0].kv_cache_spec is spec_a

    uniform_spec = UniformTypeKVCacheSpecs(
        block_size=16,
        kv_cache_specs={"layer1": spec_a, "layer2": spec_b, "layer3": spec_a},
    )
    global_groups_uniform = [
        KVCacheGroupSpec(["layer1", "layer2", "layer3"], uniform_spec),
    ]
    projected = kv_cache_planning._project_kv_cache_groups_to_worker(
        global_groups_uniform, {"layer1": spec_a, "layer3": spec_a}
    )
    assert len(projected) == 1
    assert projected[0].layer_names == ["layer1", "layer3"]
    proj_spec = projected[0].kv_cache_spec
    assert isinstance(proj_spec, UniformTypeKVCacheSpecs)
    assert set(proj_spec.kv_cache_specs.keys()) == {"layer1", "layer3"}


@pytest.mark.parametrize("with_draft", [False, True])
@pytest.mark.parametrize("indexer_alignment", [None, 512])
def test_sparse_mla_preserves_physical_row_addressing(with_draft, indexer_alignment):
    from vllm.v1.attention.backends.mla.sparse_utils import flat_kv_row_view
    from vllm.v1.worker.utils import allocate_kv_cache

    config = VllmConfig(model_config=ModelConfig(max_model_len=1024))
    config.cache_config.kv_cache_layout = "BLHNC"
    config.parallel_config.decode_context_parallel_size = 4
    common = dict(block_size=64, num_kv_heads=1, dtype=torch.uint8)
    specs = {
        "target": MLAAttentionSpec(
            **common,
            head_size=576,
            state_content_bytes=656,
            block_stride_alignment=656,
            cache_dtype_str="fp8_ds_mla",
            is_index_group_leader=True,
        ),
        "indexer": MLAAttentionSpec(
            **common,
            head_size=132,
            cache_role=SparseCacheRole.INDEXER,
            block_stride_alignment=indexer_alignment,
        ),
        "draft": SlidingWindowSpec(
            block_size=64,
            num_kv_heads=64,
            head_size=64,
            dtype=torch.bfloat16,
            sliding_window=2048,
            dcp_sharded=False,
        ),
    }
    if not with_draft:
        del specs["draft"]
    groups = get_kv_cache_groups(config, specs)
    cache_config = kv_cache_planning.get_kv_cache_config_from_groups(
        config, groups, available_memory=8 * 1024 * 1024
    )
    caches = allocate_kv_cache(
        cache_config, torch.device("cpu"), KVCacheLayout.BLHNC, [64] * len(groups)
    )
    cache = caches["target"].squeeze(1)
    if indexer_alignment is not None:
        assert caches["indexer"].stride(0) % indexer_alignment == 0
    rows, stride_rows = flat_kv_row_view(cache, 64)
    cache[1, 0].fill_(7)
    torch.testing.assert_close(rows[stride_rows], cache[1, 0])
    assert torch.all(rows[stride_rows] == 7)


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


@pytest.mark.parametrize(
    ("model_id", "max_model_len", "want_estimated_max_len"),
    [
        ("Qwen/Qwen1.5-7B", 16385, 16384),
        ("Qwen/Qwen1.5-7B", 16383, 16383),
    ],
)
def test_estimate_max_model_len(model_id, max_model_len, want_estimated_max_len):
    # Create a VllmConfig
    model_config = ModelConfig(
        model_id,
        runner="generate",
        dtype="float16",
        max_model_len=max_model_len,
    )
    scheduler_config = SchedulerConfig(
        max_num_batched_tokens=32768,
        max_model_len=model_config.max_model_len,
        is_encoder_decoder=model_config.is_encoder_decoder,
    )

    vllm_config = VllmConfig(
        model_config=model_config,
        scheduler_config=scheduler_config,
    )

    # Create KV cache specs
    kv_cache_spec = {}
    for i in range(32):
        layer_name = f"layer_{i}"
        kv_cache_spec[layer_name] = FullAttentionSpec(
            block_size=16,
            num_kv_heads=32,
            head_size=128,
            dtype=torch.float16,
        )
    # Estimate the maximum model length, 16384 model_len need 8GB
    estimated_max_len = estimate_max_model_len(
        vllm_config, kv_cache_spec, 8 * GiB_bytes
    )
    assert estimated_max_len == want_estimated_max_len


def test_get_kv_cache_config_one_worker():
    # pass max_model_len to pass check_enough_kv_cache_memory
    model_config = ModelConfig(max_model_len=16)
    vllm_config = VllmConfig(model_config=model_config)
    vllm_config.cache_config.kv_cache_layout = "LBNHC"
    vllm_config.cache_config.prefix_cache_retention_interval = None

    mem_per_block_per_layer = 16 * 2 * 64 * 4 * 2
    # all layers are full attention -> single group
    kv_cache_specs_full = {
        "layer_1": new_kv_cache_spec(),
        "layer_2": new_kv_cache_spec(),
    }
    kv_cache_config_full = get_kv_cache_configs(
        vllm_config, [kv_cache_specs_full], [mem_per_block_per_layer * 2 * 32]
    )[0]
    print(kv_cache_config_full)
    assert kv_cache_config_full == KVCacheConfig(
        num_blocks=32,
        kv_cache_tensors=[
            KVCacheTensor(
                size=mem_per_block_per_layer * 32 * 2,
                layers=["layer_1", "layer_2"],
                layer_stride=mem_per_block_per_layer * 32,
                block_stride=mem_per_block_per_layer,
            ),
        ],
        kv_cache_groups=[KVCacheGroupSpec(["layer_1", "layer_2"], new_kv_cache_spec())],
    )

    # all layers are sliding window -> single group
    kv_cache_specs_sliding = {
        "layer_1": new_sliding_window_spec(),
        "layer_2": new_sliding_window_spec(),
    }
    kv_cache_config_sliding = get_kv_cache_configs(
        vllm_config, [kv_cache_specs_sliding], [mem_per_block_per_layer * 2 * 32]
    )[0]
    assert kv_cache_config_sliding == KVCacheConfig(
        num_blocks=32,
        kv_cache_tensors=[
            KVCacheTensor(
                size=mem_per_block_per_layer * 32 * 2,
                layers=["layer_1", "layer_2"],
                layer_stride=mem_per_block_per_layer * 32,
                block_stride=mem_per_block_per_layer,
            ),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(["layer_1", "layer_2"], new_sliding_window_spec())
        ],
    )

    # full + sliding, but disable_hybrid_kv_cache_manager
    vllm_config.scheduler_config.disable_hybrid_kv_cache_manager = True
    kv_cache_specs_hybrid = {
        "layer_1": new_kv_cache_spec(),
        "layer_2": new_sliding_window_spec(),
    }
    kv_cache_config_hybrid = get_kv_cache_configs(
        vllm_config, [kv_cache_specs_hybrid], [mem_per_block_per_layer * 2 * 32]
    )[0]
    assert kv_cache_config_hybrid == KVCacheConfig(
        num_blocks=32,
        kv_cache_tensors=[
            KVCacheTensor(
                size=mem_per_block_per_layer * 32 * 2,
                layers=["layer_1", "layer_2"],
                layer_stride=mem_per_block_per_layer * 32,
                block_stride=mem_per_block_per_layer,
            ),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer_1", "layer_2"], new_kv_cache_spec(sliding_window=1)
            ),
        ],
    )
    vllm_config.scheduler_config.disable_hybrid_kv_cache_manager = False

    # full + sliding, with hybrid_kv_cache_manager
    kv_cache_specs_hybrid = {
        "layer_1": new_kv_cache_spec(),
        "layer_2": new_sliding_window_spec(),
    }
    kv_cache_config_hybrid = get_kv_cache_configs(
        vllm_config, [kv_cache_specs_hybrid], [mem_per_block_per_layer * 2 * 32]
    )[0]
    assert kv_cache_config_hybrid == KVCacheConfig(
        num_blocks=64,
        kv_cache_tensors=[
            KVCacheTensor(
                size=mem_per_block_per_layer * 64,
                layers=["layer_1"],
                layer_stride=mem_per_block_per_layer * 64,
                block_stride=mem_per_block_per_layer,
            ),
            KVCacheTensor(
                size=mem_per_block_per_layer * 64,
                layers=["layer_2"],
                layer_stride=mem_per_block_per_layer * 64,
                block_stride=mem_per_block_per_layer,
            ),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(["layer_1"], new_kv_cache_spec()),
            KVCacheGroupSpec(["layer_2"], new_sliding_window_spec()),
        ],
    )

    # 2 full + 4 sliding, 2 layers per group
    kv_cache_specs_hybrid = {
        "layer_1": new_kv_cache_spec(),
        "layer_2": new_kv_cache_spec(),
        "layer_3": new_sliding_window_spec(),
        "layer_4": new_sliding_window_spec(),
        "layer_5": new_sliding_window_spec(),
        "layer_6": new_sliding_window_spec(),
    }
    kv_cache_config_hybrid = get_kv_cache_configs(
        vllm_config, [kv_cache_specs_hybrid], [mem_per_block_per_layer * 2 * 32]
    )[0]
    assert kv_cache_config_hybrid == KVCacheConfig(
        num_blocks=32,
        kv_cache_tensors=[
            KVCacheTensor(
                size=mem_per_block_per_layer * 32 * 2,
                layers=["layer_1", "layer_2"],
                layer_stride=mem_per_block_per_layer * 32,
                block_stride=mem_per_block_per_layer,
            ),
            KVCacheTensor(
                size=mem_per_block_per_layer * 32 * 2,
                layers=["layer_3", "layer_5"],
                layer_stride=mem_per_block_per_layer * 32,
                block_stride=mem_per_block_per_layer,
            ),
            KVCacheTensor(
                size=mem_per_block_per_layer * 32 * 2,
                layers=["layer_4", "layer_6"],
                layer_stride=mem_per_block_per_layer * 32,
                block_stride=mem_per_block_per_layer,
            ),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(["layer_1", "layer_2"], new_kv_cache_spec()),
            KVCacheGroupSpec(["layer_3", "layer_5"], new_sliding_window_spec()),
            KVCacheGroupSpec(["layer_4", "layer_6"], new_sliding_window_spec()),
        ],
    )

    # 3 full + 7 sliding, pad to 3 full + 9 sliding
    kv_cache_specs_hybrid = {
        "layer_1": new_kv_cache_spec(),
        "layer_2": new_kv_cache_spec(),
        "layer_3": new_kv_cache_spec(),
        "layer_4": new_sliding_window_spec(),
        "layer_5": new_sliding_window_spec(),
        "layer_6": new_sliding_window_spec(),
        "layer_7": new_sliding_window_spec(),
        "layer_8": new_sliding_window_spec(),
        "layer_9": new_sliding_window_spec(),
        "layer_10": new_sliding_window_spec(),
    }
    kv_cache_config_hybrid = get_kv_cache_configs(
        vllm_config, [kv_cache_specs_hybrid], [mem_per_block_per_layer * 3 * 32]
    )[0]
    assert kv_cache_config_hybrid == KVCacheConfig(
        num_blocks=32,
        kv_cache_tensors=[
            KVCacheTensor(
                size=mem_per_block_per_layer * 32 * 3,
                layers=["layer_1", "layer_2", "layer_3"],
                layer_stride=mem_per_block_per_layer * 32,
                block_stride=mem_per_block_per_layer,
            ),
            KVCacheTensor(
                size=mem_per_block_per_layer * 32 * 3,
                layers=["layer_4", "layer_7", "layer_10"],
                layer_stride=mem_per_block_per_layer * 32,
                block_stride=mem_per_block_per_layer,
            ),
            KVCacheTensor(
                size=mem_per_block_per_layer * 32 * 3,
                layers=["layer_5", "layer_8"],
                layer_stride=mem_per_block_per_layer * 32,
                block_stride=mem_per_block_per_layer,
            ),
            KVCacheTensor(
                size=mem_per_block_per_layer * 32 * 3,
                layers=["layer_6", "layer_9"],
                layer_stride=mem_per_block_per_layer * 32,
                block_stride=mem_per_block_per_layer,
            ),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(["layer_1", "layer_2", "layer_3"], new_kv_cache_spec()),
            KVCacheGroupSpec(
                ["layer_4", "layer_7", "layer_10"], new_sliding_window_spec()
            ),
            KVCacheGroupSpec(["layer_5", "layer_8"], new_sliding_window_spec()),
            KVCacheGroupSpec(["layer_6", "layer_9"], new_sliding_window_spec()),
        ],
    )

    # 6 full + 5 sliding, pad to 6 full + 6 sliding. This is a typical case for gpt-oss
    # eagle where there is only one more full attention layer than sliding window layers
    kv_cache_specs_hybrid = {
        "layer_1": new_kv_cache_spec(),
        "layer_2": new_kv_cache_spec(),
        "layer_3": new_kv_cache_spec(),
        "layer_4": new_kv_cache_spec(),
        "layer_5": new_kv_cache_spec(),
        "layer_6": new_kv_cache_spec(),
        "layer_7": new_sliding_window_spec(),
        "layer_8": new_sliding_window_spec(),
        "layer_9": new_sliding_window_spec(),
        "layer_10": new_sliding_window_spec(),
        "layer_11": new_sliding_window_spec(),
    }

    kv_cache_config_hybrid = get_kv_cache_configs(
        vllm_config, [kv_cache_specs_hybrid], [mem_per_block_per_layer * 6 * 32]
    )[0]
    print(kv_cache_config_hybrid)
    assert kv_cache_config_hybrid == KVCacheConfig(
        num_blocks=32,
        kv_cache_tensors=[
            KVCacheTensor(
                size=mem_per_block_per_layer * 32 * 6,
                layers=[
                    "layer_1",
                    "layer_2",
                    "layer_3",
                    "layer_4",
                    "layer_5",
                    "layer_6",
                ],
                layer_stride=mem_per_block_per_layer * 32,
                block_stride=mem_per_block_per_layer,
            ),
            KVCacheTensor(
                size=mem_per_block_per_layer * 32 * 6,
                layers=["layer_7", "layer_8", "layer_9", "layer_10", "layer_11"],
                layer_stride=mem_per_block_per_layer * 32,
                block_stride=mem_per_block_per_layer,
            ),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer_1", "layer_2", "layer_3", "layer_4", "layer_5", "layer_6"],
                new_kv_cache_spec(),
            ),
            KVCacheGroupSpec(
                ["layer_7", "layer_8", "layer_9", "layer_10", "layer_11"],
                new_sliding_window_spec(),
            ),
        ],
    )

    # different hidden size but same type, use UniformTypeKVCacheSpecs
    kv_cache_specs_hybrid = {
        "layer_1": new_kv_cache_spec(head_size=128),
        "layer_2": new_kv_cache_spec(head_size=64),
    }
    kv_cache_config_hybrid = get_kv_cache_configs(
        vllm_config, [kv_cache_specs_hybrid], [mem_per_block_per_layer * 3 * 32]
    )[0]
    # Layers of different page sizes pack densely into one allocation: the
    # 2x-larger layer_1 takes the first region, layer_2 the next.
    assert kv_cache_config_hybrid == KVCacheConfig(
        num_blocks=32,
        kv_cache_tensors=[
            KVCacheTensor(
                size=mem_per_block_per_layer * 32 * 3,
                layers=["layer_1"],
                layer_stride=mem_per_block_per_layer * 2 * 32,
                block_stride=mem_per_block_per_layer * 2,
            ),
            KVCacheTensor(
                size=mem_per_block_per_layer * 32 * 3,
                layers=["layer_2"],
                layer_stride=mem_per_block_per_layer * 32,
                block_stride=mem_per_block_per_layer,
                offset=mem_per_block_per_layer * 2 * 32,
            ),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer_1", "layer_2"],
                UniformTypeKVCacheSpecs(
                    block_size=16, kv_cache_specs=kv_cache_specs_hybrid
                ),
            )
        ],
    )

    # Different hidden size and different type, align by different block size
    kv_cache_specs_hybrid = {
        "layer_1": new_kv_cache_spec(head_size=64),
        "layer_2": new_sliding_window_spec(head_size=32),
    }
    kv_cache_config_hybrid = get_kv_cache_configs(
        vllm_config, [kv_cache_specs_hybrid], [mem_per_block_per_layer * 32]
    )[0]
    assert kv_cache_config_hybrid == KVCacheConfig(
        num_blocks=32,
        kv_cache_tensors=[
            KVCacheTensor(
                size=mem_per_block_per_layer * 32,
                layers=["layer_1"],
                layer_stride=mem_per_block_per_layer * 32,
                block_stride=mem_per_block_per_layer,
            ),
            KVCacheTensor(
                size=mem_per_block_per_layer * 32,
                layers=["layer_2"],
                layer_stride=mem_per_block_per_layer * 32,
                block_stride=mem_per_block_per_layer,
            ),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(["layer_1"], new_kv_cache_spec(head_size=64)),
            KVCacheGroupSpec(
                ["layer_2"], new_sliding_window_spec(head_size=32, block_size=32)
            ),
        ],
    )

    # different hidden size that cannot be aligned by using different block size,
    # but can be aligned by padding the smaller physical page.
    swa_spec = new_sliding_window_spec(head_size=96)
    kv_cache_specs_hybrid = {
        "layer_1": new_kv_cache_spec(head_size=64),
        "layer_2": swa_spec,
    }

    kv_cache_config_hybrid = get_kv_cache_configs(
        vllm_config, [kv_cache_specs_hybrid], [mem_per_block_per_layer * 2 * 32]
    )[0]
    padded_page_size = swa_spec.page_size_bytes
    assert kv_cache_config_hybrid == KVCacheConfig(
        num_blocks=42,
        kv_cache_tensors=[
            KVCacheTensor(
                size=padded_page_size * 42,
                layers=["layer_1"],
                layer_stride=padded_page_size * 42,
                block_stride=padded_page_size,
            ),
            KVCacheTensor(
                size=padded_page_size * 42,
                layers=["layer_2"],
                layer_stride=padded_page_size * 42,
                block_stride=padded_page_size,
            ),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer_1"],
                new_kv_cache_spec(
                    head_size=64,
                    page_size_padded=padded_page_size,
                ),
            ),
            KVCacheGroupSpec(
                ["layer_2"],
                new_sliding_window_spec(head_size=96),
            ),
        ],
    )

    # Test num_gpu_blocks_override
    vllm_config.cache_config.num_gpu_blocks_override = 16
    kv_cache_config_override_blocks = get_kv_cache_configs(
        vllm_config, [kv_cache_specs_full], [mem_per_block_per_layer * 2 * 32]
    )[0]
    assert kv_cache_config_override_blocks == KVCacheConfig(
        num_blocks=16,
        kv_cache_tensors=[
            KVCacheTensor(
                size=mem_per_block_per_layer * 16 * 2,
                layers=["layer_1", "layer_2"],
                layer_stride=mem_per_block_per_layer * 16,
                block_stride=mem_per_block_per_layer,
            ),
        ],
        kv_cache_groups=[KVCacheGroupSpec(["layer_1", "layer_2"], new_kv_cache_spec())],
    )


def test_get_kv_cache_configs_attention_free():
    kv_cache_specs: dict[str, KVCacheSpec] = {}
    vllm_config = VllmConfig(model_config=ModelConfig(max_model_len=16))
    vllm_config.cache_config.prefix_cache_retention_interval = None
    kv_cache_configs = get_kv_cache_configs(vllm_config, [kv_cache_specs], [0])
    assert kv_cache_configs == [
        KVCacheConfig(
            num_blocks=1,
            kv_cache_tensors=[],
            kv_cache_groups=[],
        )
    ]


def test_packed_groups_glm5_like_hybrid():
    vllm_config = VllmConfig(model_config=ModelConfig(max_model_len=8192))
    vllm_config.cache_config.kv_cache_layout = "BLHNC"
    kpool, block_size = 4, 1024
    kv_cache_spec: dict[str, KVCacheSpec] = {}
    for i in range(11):
        kv_cache_spec[f"layers.{i}.attn"] = MLAAttentionSpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=576,
            dtype=torch.bfloat16,
            block_stride_alignment=32 * 576 * 2,
        )
        kv_cache_spec[f"layers.{i}.indexer"] = MLAAttentionSpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=132,
            dtype=torch.uint8,
            tokens_per_state=kpool,
            block_stride_alignment=64 * 132,
        )
        kv_cache_spec[f"layers.{i}.tail"] = CircularBufferSpec(
            block_size=kpool,
            num_kv_heads=2,
            head_size=128,
            head_size_v=0,
            dtype=torch.bfloat16,
        )
    for i in range(34):
        kv_cache_spec[f"layers.{i}.kda"] = MambaSpec(
            shapes=((3, 6144), (16, 128, 128)),
            dtypes=(torch.bfloat16, torch.float32),
            block_size=block_size,
        )

    groups = kv_cache_planning.get_kv_cache_groups(vllm_config, kv_cache_spec)
    attn = [
        g
        for g in groups
        if isinstance(g.kv_cache_spec, UniformTypeKVCacheSpecs)
        and isinstance(g.kv_cache_spec.first_spec, MLAAttentionSpec)
    ]
    rings = [
        g
        for g in groups
        if isinstance(g.kv_cache_spec, UniformTypeKVCacheSpecs)
        and isinstance(g.kv_cache_spec.first_spec, CircularBufferSpec)
    ]
    mamba = [
        g
        for g in groups
        if isinstance(
            getattr(g.kv_cache_spec, "first_spec", g.kv_cache_spec), MambaSpec
        )
    ]
    assert len(attn) == 1 and len(attn[0].layer_names) == 22
    assert len(rings) == 1 and len(rings[0].layer_names) == 11
    # 11 * (1.18 MB MLA + 34 KB indexer) per block fits 12 fp32-SSM KDA states.
    assert len(mamba) == 3
    assert sum(len(g.layer_names) for g in mamba) == 34

    kv_cache_config = kv_cache_planning.get_kv_cache_config_from_groups(
        vllm_config, groups, available_memory=2 * 1024**3
    )
    strides = {t.block_stride for t in kv_cache_config.kv_cache_tensors}
    assert len(strides) == 1
    (block_stride,) = strides
    # Blocks sit a whole number of 32-row MLA pages and 64-state indexer pages
    # apart so both caches can be re-paged for their kernels.
    assert block_stride % (32 * 576 * 2) == 0
    assert block_stride % (64 * 132) == 0


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


def new_indexer_mla_spec(block_size=16):
    # Sparse-attention indexer k_cache: an MLAAttentionSpec with a much smaller
    # page size than the main MLA attention (uint8, small head), so their pages
    # cannot be unified.
    return MLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=132,
        dtype=torch.uint8,
    )


def test_mixed_page_size_groups_use_spec_compatibility():
    specs = {}
    for i in range(3):
        specs[f"mla.{i}"] = new_mla_spec()
        specs[f"indexer.{i}"] = new_indexer_mla_spec()
    specs.update({f"swa.{i}": new_swa_mla_spec(head_size=1024) for i in range(5)})

    config = _grouping_config()
    config.cache_config = CacheConfig()
    config.cache_config.kv_cache_layout = "BLNHC"
    groups = get_kv_cache_groups(config, specs)

    assert len(groups) == 3
    assert {name for group in groups for name in group.layer_names} == set(specs)
    assert sorted(len(group.layer_names) for group in groups) == [2, 3, 6]


def _grouping_config():
    cache_config = CacheConfig()
    cache_config.kv_cache_layout = "LBNHC"
    return SimpleNamespace(
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        speculative_config=None,
        cache_config=cache_config,
    )


def test_hidden_state_group_preserves_hybrid_prefix_cache_granularity():
    block_size = 544
    full_spec = FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        # FullAttentionSpec stores both K and V, so this produces a
        # 544 * 1 * (512 + 512) * 2 = 1,114,112-byte page.
        head_size=512,
        dtype=torch.bfloat16,
    )
    mamba_spec = MambaSpec(
        block_size=block_size,
        shapes=((557056,),),
        dtypes=(torch.bfloat16,),
        mamba_cache_mode="align",
    )
    hidden_spec = HiddenStateCacheSpec(
        block_size=block_size,
        num_kv_heads=3,
        head_size=1024,
        dtype=torch.bfloat16,
    )
    assert full_spec.page_size_bytes == mamba_spec.page_size_bytes

    groups = get_kv_cache_groups(
        _grouping_config(),
        {
            "model.full_attn": full_spec,
            "model.mamba": mamba_spec,
            "cache_only_layers.0": hidden_spec,
        },
    )

    hidden_group = next(
        group
        for group in groups
        if isinstance(group.kv_cache_spec, HiddenStateCacheSpec)
    )
    assert hidden_group.kv_cache_spec.block_size == 136

    kv_cache_config = KVCacheConfig(
        num_blocks=1,
        kv_cache_tensors=[],
        kv_cache_groups=groups,
    )
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(
            block_size=16,
            enable_prefix_caching=True,
            prefix_match_unit=None,
        ),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        kv_transfer_config=object(),
    )
    assert kv_cache_utils.resolve_kv_cache_block_sizes(
        kv_cache_config, vllm_config
    ) == (544, 136)


def test_multi_run_layer_compact_strides_place_hoisted_heads():
    """A layer-compact run region is its own dense allocation: under LHBNC the head
    groups sit between the layers and the blocks, so a run's block stride is one head
    group's slice, not the whole page (regression test for setStorage out-of-bounds
    on ROCm hybrid models)."""
    full = new_kv_cache_spec()
    swa = new_sliding_window_spec(sliding_window=full.block_size * 2)
    page = full.page_size_bytes
    assert swa.page_size_bytes == page

    vllm_config = VllmConfig(model_config=ModelConfig(max_model_len=16))
    vllm_config.cache_config.kv_cache_layout = "LHBNC"
    config = kv_cache_planning.get_kv_cache_config_from_groups(
        vllm_config,
        [
            KVCacheGroupSpec(["full.0", "full.1"], full),
            KVCacheGroupSpec(["swa.0", "swa.1"], swa),
        ],
        available_memory=4 * page * 8,
    )

    num_blocks = config.num_blocks
    head_group = full.block_size * full.state_content_size_bytes
    for tensor in config.kv_cache_tensors:
        # One head group per block, head groups between the layers and blocks.
        assert tensor.block_stride == head_group
        assert tensor.layer_stride == full.num_heads * num_blocks * head_group
        # The last block of the last layer's last head group must stay within the
        # allocation (the old page-based stride overran it).
        end = (
            tensor.offset
            + tensor.layer_stride * (len(tensor.layers) - 1)
            + (full.num_heads - 1) * num_blocks * head_group
            + tensor.block_stride * (num_blocks - 1)
            + head_group
        )
        assert end <= tensor.size


def test_mla_draft_prefers_standard_layout_when_pages_can_be_unified():
    specs = {
        "target.0.attn": new_mla_spec(),
        "draft.0": new_sliding_window_spec(num_kv_heads=1, head_size=288),
    }
    assert len({spec.page_size_bytes for spec in specs.values()}) == 1

    groups = get_kv_cache_groups(_grouping_config(), specs)

    assert len(groups) == 2
    assert all(
        not isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs) for group in groups
    )


def test_mla_with_incompatible_swa_uses_one_full_allocation_group(caplog_vllm):
    # Sparse MLA pages cannot be padded safely. Keeping the draft's attention
    # compute sliding-window while promoting only its allocation semantics lets
    # every layer share the target's block table and remain contiguous.
    draft = new_sliding_window_spec(block_size=16)
    specs = {
        "target.0.attn": new_mla_spec(block_size=64),
        "target.0.indexer": new_indexer_mla_spec(block_size=64),
        "draft.0": draft,
    }

    groups = get_kv_cache_groups(_grouping_config(), specs)
    assert len(groups) == 1
    assert set(groups[0].layer_names) == set(specs)
    group_spec = groups[0].kv_cache_spec
    assert isinstance(group_spec, UniformTypeKVCacheSpecs)
    assert group_spec.block_size == 64
    promoted_draft = group_spec.kv_cache_specs["draft.0"]
    assert isinstance(promoted_draft, FullAttentionSpec)
    assert not isinstance(promoted_draft, SlidingWindowSpec)
    assert promoted_draft.block_size == 64
    assert promoted_draft.sliding_window == draft.sliding_window
    assert specs["draft.0"] is draft
    assert "attention compute is unchanged" in caplog_vllm.text


def test_hidden_states_with_tp_scales_page_size():
    """When TP shrinks KV pages below the hidden-state per-token cost,
    get_kv_cache_groups must scale up target block sizes so that the
    common page accommodates the unsharded hidden states."""
    # Simulate TP=4 sharding a model with 8 KV heads → 2 per rank.
    # KV page = block_size(16) * num_kv_heads(2) * head_size(64) * dtype(2)
    #         = 16 * 2 * 64 * 2 = 4096 bytes.
    kv_spec = new_kv_cache_spec(
        block_size=16,
        num_kv_heads=2,
        head_size=64,
        dtype=torch.bfloat16,
    )
    # Hidden-state per-token cost = num_hidden_states(6) * hidden_size(512)
    #   * dtype(2) = 6144 bytes, which exceeds the 4096-byte KV page.
    hs_spec = HiddenStateCacheSpec(
        block_size=16,
        num_kv_heads=6,
        head_size=512,
        dtype=torch.bfloat16,
    )
    specs = {
        "target.0.attn": kv_spec,
        "target.1.attn": kv_spec,
        "cache_only_layers.48": hs_spec,
    }

    groups = get_kv_cache_groups(_grouping_config(), specs)

    # The hidden-state layer should be present and no assertion should fire.
    all_layers = {name for g in groups for name in g.layer_names}
    assert "cache_only_layers.48" in all_layers

    # The target group block sizes must have been scaled up.
    for g in groups:
        if "cache_only_layers.48" not in g.layer_names:
            assert g.kv_cache_spec.block_size > kv_spec.block_size


def test_auto_fit_max_model_len():
    """Test that max_model_len=-1 auto-fits to available GPU memory."""
    # Create config with original_max_model_len=-1 to trigger auto-fit
    model_config = ModelConfig(max_model_len=1024)
    # Simulate the user passing -1 by setting original_max_model_len
    model_config.original_max_model_len = -1
    vllm_config = VllmConfig(model_config=model_config)
    vllm_config.cache_config.kv_cache_layout = "LBNHC"

    mem_per_block_per_layer = 16 * 2 * 64 * 4 * 2  # 16KB per block per layer
    kv_cache_specs = {
        "layer_1": new_kv_cache_spec(),
        "layer_2": new_kv_cache_spec(),
    }

    # With enough memory, max_model_len stays at the derived max
    large_available_memory = mem_per_block_per_layer * 2 * 1024  # plenty of memory
    _kv_cache_configs = get_kv_cache_configs(
        vllm_config, [kv_cache_specs], [large_available_memory]
    )
    assert vllm_config.model_config.max_model_len == 1024

    # Reset for next test
    model_config = ModelConfig(max_model_len=1024)
    model_config.original_max_model_len = -1
    vllm_config = VllmConfig(model_config=model_config)
    vllm_config.cache_config.kv_cache_layout = "LBNHC"

    # With limited memory, max_model_len should be reduced
    # Need memory for at least max_model_len tokens
    # 32 blocks worth of memory for 2 layers = can fit 32*16=512 tokens
    limited_memory = mem_per_block_per_layer * 2 * 32
    _kv_cache_configs = get_kv_cache_configs(
        vllm_config, [kv_cache_specs], [limited_memory]
    )
    # Should be reduced to fit in memory
    assert vllm_config.model_config.max_model_len < 1024
    assert vllm_config.model_config.max_model_len > 0


def test_auto_fit_max_model_len_with_hybrid():
    """Test that auto-fit works with hybrid KV cache specs."""
    # Create config with original_max_model_len=-1 to trigger auto-fit
    model_config = ModelConfig(max_model_len=8192)
    # Simulate the user passing -1 by setting original_max_model_len
    model_config.original_max_model_len = -1
    vllm_config = VllmConfig(model_config=model_config)
    vllm_config.cache_config.kv_cache_layout = "LBNHC"

    mem_per_block_per_layer = 16 * 2 * 64 * 4 * 2  # 16KB per block per layer
    gamma = 2
    kv_cache_specs = {
        "layer_1": new_mamba_spec(num_speculative_blocks=gamma),
        "layer_2": new_kv_cache_spec(),
    }

    # One extra block on top of what a 1024-token request needs: the pool
    # reserves one block as the null block.
    available_memory = mem_per_block_per_layer * (1024 // 16 + 1 + gamma + 1)
    _kv_cache_configs = get_kv_cache_configs(
        vllm_config, [kv_cache_specs], [available_memory]
    )
    assert vllm_config.model_config.max_model_len == 1024


def test_auto_fit_max_model_len_not_triggered():
    """Test that auto-fit is not triggered when original_max_model_len is not -1."""
    model_config = ModelConfig(max_model_len=16)
    # original_max_model_len should be None by default, not -1
    vllm_config = VllmConfig(model_config=model_config)
    vllm_config.cache_config.kv_cache_layout = "LBNHC"

    mem_per_block_per_layer = 16 * 2 * 64 * 4 * 2
    kv_cache_specs = {
        "layer_1": new_kv_cache_spec(),
        "layer_2": new_kv_cache_spec(),
    }

    # This should work normally without auto-fit
    _kv_cache_configs = get_kv_cache_configs(
        vllm_config, [kv_cache_specs], [mem_per_block_per_layer * 2 * 32]
    )
    assert vllm_config.model_config.max_model_len == 16


def test_auto_fit_max_model_len_respects_num_gpu_blocks_override():
    """Auto-fit must size max_model_len against the override-clamped pool, not
    the raw `available_memory`. Without this, auto-fit could pick a
    max_model_len that no longer fits once `num_gpu_blocks_override` is applied.
    """
    model_config = ModelConfig(max_model_len=16384)
    model_config.original_max_model_len = -1  # request auto-fit
    vllm_config = VllmConfig(model_config=model_config)
    vllm_config.cache_config.kv_cache_layout = "LBNHC"
    # Cap the cache to 32 blocks regardless of available memory.
    vllm_config.cache_config.num_gpu_blocks_override = 32

    mem_per_block_per_layer = 16 * 2 * 64 * 4 * 2
    kv_cache_specs = {
        "layer_1": new_kv_cache_spec(),  # block_size=16
        "layer_2": new_kv_cache_spec(),
    }
    # Plenty of raw memory (1024 blocks per layer would fit max_model_len=16384).
    large_available_memory = mem_per_block_per_layer * 2 * 1024

    get_kv_cache_configs(vllm_config, [kv_cache_specs], [large_available_memory])

    # 32 blocks * block_size 16 = 512 token slots, so max_model_len must
    # auto-fit at or below that.
    assert 0 < vllm_config.model_config.max_model_len <= 32 * 16


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
            assert group.slots_per_large_page >= 1
    assert configs[0].large_page_bytes == configs[1].large_page_bytes
    assert configs[0].num_blocks == configs[1].num_blocks
    for group0, group1 in zip(configs[0].kv_cache_groups, configs[1].kv_cache_groups):
        assert group0.row_bytes == group1.row_bytes


def test_check_enough_kv_cache_memory_respects_num_gpu_blocks_override():
    """Admission check must use the override-clamped pool size, not raw
    `available_memory`. Without this, startup could accept a max_model_len
    that does not actually fit in `num_gpu_blocks_override` blocks.
    """
    model_config = ModelConfig(max_model_len=16384)
    vllm_config = VllmConfig(model_config=model_config)
    vllm_config.cache_config.kv_cache_layout = "LBNHC"
    # 32 blocks is far too small for max_model_len=16384 (would need 1024).
    vllm_config.cache_config.num_gpu_blocks_override = 32

    mem_per_block_per_layer = 16 * 2 * 64 * 4 * 2
    kv_cache_specs = {
        "layer_1": new_kv_cache_spec(),
        "layer_2": new_kv_cache_spec(),
    }
    # Plenty of raw memory: a bytes-only check against this would pass.
    large_available_memory = mem_per_block_per_layer * 2 * 1024

    with pytest.raises(ValueError, match="max seq len"):
        get_kv_cache_configs(vllm_config, [kv_cache_specs], [large_available_memory])


def test_unify_kv_cache_page_size_uses_padding_for_non_divisible_sizes():
    """DFlash drafters can have a smaller head size than the target model.

    For example, MiMo uses 192-dim target KV heads while its DFlash draft uses
    128-dim KV heads. The resulting page sizes are 3:2 rather than an integer
    block-size multiple, so the smaller page must be padded instead.
    """
    # Attention layers read padded pages through a strided view, so padding is
    # allowed.
    target_spec = new_kv_cache_spec(
        block_size=16,
        num_kv_heads=1,
        head_size=192,
        dtype=torch.bfloat16,
    )
    draft_spec = new_sliding_window_spec(
        block_size=16,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
        sliding_window=1024,
    )

    unified_specs = kv_cache_planning.unify_kv_cache_spec_page_size(
        {
            "target_attn": target_spec,
            "draft_attn": draft_spec,
        }
    )

    assert unified_specs["target_attn"] == target_spec
    unified_draft_spec = unified_specs["draft_attn"]
    assert unified_draft_spec.block_size == draft_spec.block_size
    assert unified_draft_spec.real_page_size_bytes == draft_spec.real_page_size_bytes
    assert unified_draft_spec.page_size_padded == target_spec.page_size_bytes
    assert unified_draft_spec.page_size_bytes == target_spec.page_size_bytes


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


def test_unify_kv_cache_spec_page_size_mamba():
    """Regression test for https://github.com/vllm-project/vllm/issues/43626.

    MambaSpec's page_size_bytes is determined by its state shapes and does not
    change with block_size, so unify_kv_cache_spec_page_size must pad the Mamba
    page instead of scaling its block_size. This situation arises when a layer
    with a page larger than the (already platform-aligned) Mamba page joins the
    specs, e.g. a dense draft model with more KV heads than the hybrid main
    model.
    """
    # 1. Hybrid main model (Mamba + full attention, pages already aligned at
    # 16KB) plus a dense draft model layer with a 2x larger page (32KB).
    # Reproduces the bare AssertionError from #43626: 32768 % 16384 == 0, so
    # the old code scaled the Mamba block_size, which left page_size_bytes
    # unchanged at 16384.
    mamba_spec = new_mamba_spec()  # page_size_bytes = 16384
    main_attn_spec = new_kv_cache_spec()  # page_size_bytes = 16384
    draft_attn_spec = new_kv_cache_spec(num_kv_heads=4)  # page_size_bytes = 32768
    assert mamba_spec.page_size_bytes == main_attn_spec.page_size_bytes == 16384
    assert draft_attn_spec.page_size_bytes == 32768

    unified = kv_cache_planning.unify_kv_cache_spec_page_size(
        {
            "mamba_layer": mamba_spec,
            "main_attn_layer": main_attn_spec,
            "draft_attn_layer": draft_attn_spec,
        }
    )
    # Mamba page is padded; block_size (caching granularity) is unchanged.
    assert unified["mamba_layer"].page_size_bytes == 32768
    assert unified["mamba_layer"].page_size_padded == 32768
    assert unified["mamba_layer"].block_size == mamba_spec.block_size
    # Attention layer with smaller page still unifies by scaling block_size.
    assert unified["main_attn_layer"].page_size_bytes == 32768
    assert unified["main_attn_layer"].block_size == 2 * main_attn_spec.block_size
    assert unified["main_attn_layer"].page_size_padded is None
    # Layer already at max page size is unchanged.
    assert unified["draft_attn_layer"] == draft_attn_spec

    # 2. Mamba page already padded by the platform (state smaller than the
    # padded page); the padding is re-applied at the new maximum.
    padded_mamba_spec = new_mamba_spec(
        shapes=((2, 256), (3, 32, 32)), page_size_padded=16384
    )
    assert padded_mamba_spec.page_size_bytes == 16384
    unified = kv_cache_planning.unify_kv_cache_spec_page_size(
        {
            "mamba_layer": padded_mamba_spec,
            "draft_attn_layer": draft_attn_spec,
        }
    )
    assert unified["mamba_layer"].page_size_bytes == 32768
    assert unified["mamba_layer"].page_size_padded == 32768

    # 3. Mamba page that does not evenly divide the maximum page size is
    # padded as well (the divisibility constraint only applies to block_size
    # scaling).
    odd_mamba_spec = new_mamba_spec(shapes=((6144,),))
    assert odd_mamba_spec.page_size_bytes == 24576
    assert 32768 % odd_mamba_spec.page_size_bytes != 0
    unified = kv_cache_planning.unify_kv_cache_spec_page_size(
        {
            "mamba_layer": odd_mamba_spec,
            "draft_attn_layer": draft_attn_spec,
        }
    )
    assert unified["mamba_layer"].page_size_bytes == 32768

    # 4. Attention layers with non-divisible page sizes are padded too: every
    # backend reads a padded page through the view's block stride, so there is
    # no longer a case that must raise.
    unified = kv_cache_planning.unify_kv_cache_spec_page_size(
        {
            "attn_layer": new_kv_cache_spec(block_size=24),  # 24576
            "draft_attn_layer": draft_attn_spec,  # 32768
        }
    )
    assert unified["attn_layer"].page_size_padded == 32768
    assert unified["attn_layer"].page_size_bytes == 32768
    assert unified["attn_layer"].block_size == 24

    # 5. Uniform page sizes are returned unchanged.
    specs = {
        "mamba_layer": new_mamba_spec(),
        "attn_layer": new_kv_cache_spec(),
    }
    assert kv_cache_planning.unify_kv_cache_spec_page_size(specs) == specs


@pytest.mark.parametrize("use_override", [True, False])
def test_kv_cache_reserves_null_block_for_max_model_len(use_override):
    """A KV cache sized to exactly the blocks a max_model_len request needs must
    be rejected. BlockPool keeps one block as the null block, so only
    num_blocks - 1 are usable; accepting num_blocks == needed would leave the
    request unschedulable and hang the engine. Covers both the
    num_gpu_blocks_override path and the memory-derived block count.
    """
    block_size = 16
    max_model_len = 512  # needs 512 / 16 = 32 blocks
    vllm_config = VllmConfig(model_config=ModelConfig(max_model_len=max_model_len))
    vllm_config.cache_config.kv_cache_layout = "LBNHC"
    spec = new_kv_cache_spec(block_size=block_size)

    # 32 blocks -> only 31 usable after the null block: one short -> reject.
    if use_override:
        # Ample raw memory; the override alone constrains the block count.
        vllm_config.cache_config.num_gpu_blocks_override = 32
        available_memory = [spec.page_size_bytes * 1024]
    else:
        available_memory = [spec.page_size_bytes * 32]

    with pytest.raises(ValueError, match="max seq len"):
        get_kv_cache_configs(vllm_config, [{"layer1": spec}], available_memory)


def test_auto_fit_max_model_len_reserves_null_block():
    """Auto-fit (max_model_len=-1) must size max_model_len against usable
    blocks, not the total pool. With memory for exactly 64 blocks, one is the
    null block, so auto-fit must settle on 63 * block_size; picking the full
    pool would leave a max-length request unschedulable and hang the engine.
    """
    block_size = 16
    model_config = ModelConfig(max_model_len=1024)
    model_config.original_max_model_len = -1
    vllm_config = VllmConfig(model_config=model_config)
    vllm_config.cache_config.kv_cache_layout = "LBNHC"
    spec = new_kv_cache_spec(block_size=block_size)

    # Exactly the 1024 / 16 = 64 blocks a full-length request would need.
    available_memory = [spec.page_size_bytes * 64]

    get_kv_cache_configs(vllm_config, [{"layer1": spec}], available_memory)

    assert vllm_config.model_config.max_model_len == 63 * block_size


def test_check_enough_kv_cache_memory_reserves_null_block():
    """The public admission check must reject a KV cache sized to exactly the
    blocks a max_model_len request needs. BlockPool keeps one block as the
    null block, so only num_blocks - 1 are usable; accepting
    num_blocks == needed would leave the request unschedulable and hang the
    engine.
    """
    block_size = 16
    max_model_len = 512  # needs 512 / 16 = 32 blocks
    vllm_config = VllmConfig(model_config=ModelConfig(max_model_len=max_model_len))
    vllm_config.cache_config.kv_cache_layout = "LBNHC"
    spec = new_kv_cache_spec(block_size=block_size)

    # 32 blocks -> only 31 usable after the null block: one short -> reject.
    with pytest.raises(ValueError, match="max seq len"):
        check_enough_kv_cache_memory(
            vllm_config, {"layer1": spec}, spec.page_size_bytes * 32
        )

    # 33 blocks -> 32 usable after the null block -> accept.
    check_enough_kv_cache_memory(
        vllm_config, {"layer1": spec}, spec.page_size_bytes * 33
    )


def _spec_decode_kv_cache_groups(worker_specs, use_eagle_block_drop=True):
    config = VllmConfig(model_config=ModelConfig(max_model_len=64))
    config.cache_config.kv_cache_layout = "BLHNC"
    config.speculative_config = SimpleNamespace(
        method="dflash",
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
    mem = sum(s.page_size_bytes for s in specs.values()) * 10
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
# Large-page allocator (kv_cache_allocator="large_page")
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
            kv_cache_allocator="large_page",
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
    assert config.allocator == "large_page"
    assert config.hash_block_size == block_size
    assert config.cache_hit_alignment_tokens == block_size
    assert config.num_blocks == expected_num_pages
    assert config.num_blocks == expected_num_pages
    assert config.large_page_bytes == expected_page_bytes
    assert len(config.kv_cache_groups) == len(expected_groups)
    table = _large_page_group_table(config)
    for key, (n_g, rows_per_block, row_bytes) in expected_groups.items():
        group = table[key]
        assert group.slots_per_large_page == n_g, key
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
    assert configs[0].num_blocks == configs[1].num_blocks == 27630
    assert configs[0].large_page_bytes == configs[1].large_page_bytes
    # Every layer is on exactly one rank.
    assert sum(len(g.layer_names) for g in configs[0].kv_cache_groups) + sum(
        len(g.layer_names) for g in configs[1].kv_cache_groups
    ) == len(specs)
    for group0, group1 in zip(configs[0].kv_cache_groups, configs[1].kv_cache_groups):
        assert group0.slots_per_large_page == group1.slots_per_large_page
        assert group0.kernel_rows_per_block == group1.kernel_rows_per_block
        assert group0.num_blocks == group1.num_blocks
        assert group0.row_bytes == group1.row_bytes > 0
    swa0 = configs[0].kv_cache_groups[0]
    assert len(swa0.layer_names) == 20
    # The SWA row is its 23-layer row on rank 1; six of them fill a page.
    assert swa0.slots_per_large_page == 6


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
    least slots_per_large_page blocks."""
    vllm_config = _large_page_vllm_config()
    specs = _dsv41_specs()
    (config,) = get_kv_cache_configs(vllm_config, [specs], [_LARGE_PAGE_BUDGET])
    num_pages = config.num_blocks
    page_bytes = config.large_page_bytes

    def page_blocks(group_id: int, page: int) -> list[tuple[int, int]]:
        group = config.kv_cache_groups[group_id]
        block_bytes = group.kernel_rows_per_block * group.row_bytes
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
            assert len(blocks) >= config.kv_cache_groups[group_id].slots_per_large_page
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
    assert config.num_blocks == 1435
    assert config.large_page_bytes == 44892160
    assert [g.slots_per_large_page for g in config.kv_cache_groups] == [1, 86]


def test_large_page_user_block_size():
    """A user-set --block-size must be feasible as-is, else a ValueError
    naming the violated constraint."""
    specs = _dsv41_specs()
    vllm_config = _large_page_vllm_config()
    vllm_config.cache_config = CacheConfig(
        kv_cache_allocator="large_page", block_size=128
    )
    (config,) = get_kv_cache_configs(vllm_config, [specs], [_LARGE_PAGE_BUDGET])
    assert config.hash_block_size == 128

    vllm_config = _large_page_vllm_config()
    vllm_config.cache_config = CacheConfig(
        kv_cache_allocator="large_page", block_size=64
    )
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


def test_large_page_prefix_match_unit_mismatch():
    """prefix_match_unit must equal the resolved global block size B."""
    vllm_config = _large_page_vllm_config()
    vllm_config.cache_config = CacheConfig(
        kv_cache_allocator="large_page", prefix_match_unit=16
    )
    with pytest.raises(ValueError, match="prefix_match_unit=16"):
        get_kv_cache_configs(vllm_config, [_dsv41_specs()], [_LARGE_PAGE_BUDGET])

    # A unit matching the resolved B is accepted even when it differs from
    # the requested (default) block size.
    vllm_config = _large_page_vllm_config()
    vllm_config.cache_config = CacheConfig(
        kv_cache_allocator="large_page", prefix_match_unit=128
    )
    (config,) = get_kv_cache_configs(
        vllm_config, [_dsv41_specs()], [_LARGE_PAGE_BUDGET]
    )
    assert config.hash_block_size == 128


def test_large_page_num_gpu_blocks_override():
    """num_gpu_blocks_override overrides N (large pages) directly."""
    vllm_config = _large_page_vllm_config(model=_LARGE_PAGE_KIMI, mamba_align=True)
    vllm_config.cache_config = CacheConfig(
        kv_cache_allocator="large_page",
        mamba_cache_mode="align",
        num_gpu_blocks_override=1000,
    )
    (config,) = get_kv_cache_configs(
        vllm_config, [_kimilinear_specs(1)], [_LARGE_PAGE_BUDGET]
    )
    assert config.num_blocks == 1000
    assert config.num_blocks == 1000
    assert config.kv_cache_groups[0].num_blocks == 1000
    group = config.kv_cache_groups[1]
    assert group.num_blocks == (
        1000
        * config.large_page_bytes
        // (group.kernel_rows_per_block * group.row_bytes)
    )


def test_large_page_sizer_rejects_tiny_pool():
    """One max-model-len request must fit: sum_g ceil(demand_g / n_g) <= N-1."""
    vllm_config = _large_page_vllm_config()
    vllm_config.cache_config = CacheConfig(
        kv_cache_allocator="large_page", num_gpu_blocks_override=4
    )
    with pytest.raises(ValueError, match="To serve at least one request"):
        get_kv_cache_configs(vllm_config, [_dsv41_specs()], [_LARGE_PAGE_BUDGET])


def test_large_page_attention_free():
    vllm_config = _large_page_vllm_config()
    (config,) = get_kv_cache_configs(vllm_config, [{}], [_LARGE_PAGE_BUDGET])
    assert config.num_blocks == 1
    assert config.kv_cache_groups == []
    assert config.kv_cache_tensors == []
    assert config.allocator == "large_page"


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

    assert kv_cache_utils.resolve_kv_cache_block_sizes(config, vllm_config) == (
        8192,
        config.hash_block_size,
    )


def test_large_page_disable_hybrid_promotes_sliding_window():
    """--disable-hybrid-kv-cache-manager keeps its uniform_page meaning:
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
    assert config.allocator == "large_page"
    assert config.num_blocks >= 64
    for group in config.kv_cache_groups:
        assert group.num_blocks >= 64
    assert len(config.kv_cache_tensors) == 27
