# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for HybridCpuGpuOffloadingHandlers.

Tests per-group handler creation and layer filtering for HMA support.
"""

from dataclasses import dataclass

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, KVCacheTensor
from vllm.v1.kv_offload.worker.hybrid_cpu_gpu import HybridCpuGpuOffloadingHandlers


@dataclass
class MockKVCacheSpec:
    """Mock KVCacheSpec for testing."""

    block_size: int = 16


def create_mock_kv_cache_config(
    groups: list[list[str]],
) -> KVCacheConfig:
    """
    Create a mock KVCacheConfig for testing.

    Args:
        groups: List of layer name lists, one per group.
                e.g., [["layer.0", "layer.1"], ["layer.2", "layer.3"]]

    Returns:
        KVCacheConfig with the specified groups.
    """
    kv_cache_groups = []
    all_layers = []
    for layer_names in groups:
        kv_cache_groups.append(
            KVCacheGroupSpec(
                layer_names=layer_names,
                kv_cache_spec=MockKVCacheSpec(),  # type: ignore
            )
        )
        all_layers.extend(layer_names)

    return KVCacheConfig(
        num_blocks=100,
        kv_cache_tensors=[KVCacheTensor(size=1024, shared_by=all_layers)],
        kv_cache_groups=kv_cache_groups,
    )


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="CUDA required for CpuGpuOffloadingHandlers",
)
@torch.inference_mode()
def test_hybrid_handlers_creation(default_vllm_config):
    """
    Tests that HybridCpuGpuOffloadingHandlers creates handlers for each group
    with only that group's layers.
    """
    # Create config with 2 groups
    kv_cache_config = create_mock_kv_cache_config(
        groups=[
            ["layer.0", "layer.1"],  # Group 0
            ["layer.2", "layer.3"],  # Group 1
        ]
    )

    # Create GPU caches for all layers
    num_blocks = 64
    block_size = 16
    num_heads = 8
    head_size = 64
    dtype = torch.bfloat16
    device = "cuda:0"

    gpu_caches = {}
    attn_backends = {}
    for layer_name in ["layer.0", "layer.1", "layer.2", "layer.3"]:
        gpu_cache_shape = FlashAttentionBackend.get_kv_cache_shape(
            num_blocks, block_size, num_heads, head_size
        )
        gpu_caches[layer_name] = torch.rand(gpu_cache_shape, dtype=dtype, device=device)
        attn_backends[layer_name] = FlashAttentionBackend

    # Create hybrid handlers
    handlers = HybridCpuGpuOffloadingHandlers(
        kv_cache_config=kv_cache_config,
        gpu_block_size=block_size,
        cpu_block_size=block_size * 2,  # 2 GPU blocks per CPU block
        num_cpu_blocks_per_group=32,
        gpu_caches=gpu_caches,
        attn_backends=attn_backends,
    )

    # Should have 2 groups
    assert len(handlers) == 2
    assert handlers.get_group_ids() == [0, 1]

    # Each group should have its own handlers
    for group_id in [0, 1]:
        gpu_to_cpu = handlers.get_gpu_to_cpu_handler(group_id)
        cpu_to_gpu = handlers.get_cpu_to_gpu_handler(group_id)
        assert gpu_to_cpu is not None
        assert cpu_to_gpu is not None

    # Group 0's handlers should only have 2 layers (layer.0, layer.1)
    group0_handlers = handlers.group_handlers[0]
    assert len(group0_handlers.gpu_to_cpu_handler.src_tensors) == 2

    # Group 1's handlers should only have 2 layers (layer.2, layer.3)
    group1_handlers = handlers.group_handlers[1]
    assert len(group1_handlers.gpu_to_cpu_handler.src_tensors) == 2


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="CUDA required for CpuGpuOffloadingHandlers",
)
@torch.inference_mode()
def test_hybrid_handlers_partial_layers(default_vllm_config):
    """
    Tests HybridCpuGpuOffloadingHandlers when not all group layers
    have GPU caches (e.g., pipeline parallelism).
    """
    # Create config with 2 groups
    kv_cache_config = create_mock_kv_cache_config(
        groups=[
            ["layer.0", "layer.1", "layer.2"],  # Group 0
            ["layer.3", "layer.4", "layer.5"],  # Group 1
        ]
    )

    # Create GPU caches for only some layers (simulating PP)
    num_blocks = 64
    block_size = 16
    num_heads = 8
    head_size = 64
    dtype = torch.bfloat16
    device = "cuda:0"

    # Only layers 0, 1, 3 are present (layers 2, 4, 5 on different PP rank)
    gpu_caches = {}
    attn_backends = {}
    for layer_name in ["layer.0", "layer.1", "layer.3"]:
        gpu_cache_shape = FlashAttentionBackend.get_kv_cache_shape(
            num_blocks, block_size, num_heads, head_size
        )
        gpu_caches[layer_name] = torch.rand(gpu_cache_shape, dtype=dtype, device=device)
        attn_backends[layer_name] = FlashAttentionBackend

    # Create hybrid handlers
    handlers = HybridCpuGpuOffloadingHandlers(
        kv_cache_config=kv_cache_config,
        gpu_block_size=block_size,
        cpu_block_size=block_size * 2,
        num_cpu_blocks_per_group=32,
        gpu_caches=gpu_caches,
        attn_backends=attn_backends,
    )

    # Both groups should still exist (each has at least one layer)
    assert len(handlers) == 2

    # Group 0 should have 2 layers (layer.0, layer.1)
    group0_handlers = handlers.group_handlers[0]
    assert len(group0_handlers.gpu_to_cpu_handler.src_tensors) == 2

    # Group 1 should have 1 layer (layer.3)
    group1_handlers = handlers.group_handlers[1]
    assert len(group1_handlers.gpu_to_cpu_handler.src_tensors) == 1


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="CUDA required for CpuGpuOffloadingHandlers",
)
@torch.inference_mode()
def test_hybrid_handlers_empty_group(default_vllm_config):
    """
    Tests that HybridCpuGpuOffloadingHandlers skips groups with no
    present layers (e.g., all layers on different PP rank).
    """
    # Create config with 3 groups
    kv_cache_config = create_mock_kv_cache_config(
        groups=[
            ["layer.0", "layer.1"],  # Group 0
            ["layer.2", "layer.3"],  # Group 1 (empty)
            ["layer.4", "layer.5"],  # Group 2
        ]
    )

    # Create GPU caches - skip group 1's layers entirely
    num_blocks = 64
    block_size = 16
    num_heads = 8
    head_size = 64
    dtype = torch.bfloat16
    device = "cuda:0"

    gpu_caches = {}
    attn_backends = {}
    for layer_name in ["layer.0", "layer.1", "layer.4", "layer.5"]:
        gpu_cache_shape = FlashAttentionBackend.get_kv_cache_shape(
            num_blocks, block_size, num_heads, head_size
        )
        gpu_caches[layer_name] = torch.rand(gpu_cache_shape, dtype=dtype, device=device)
        attn_backends[layer_name] = FlashAttentionBackend

    # Create hybrid handlers
    handlers = HybridCpuGpuOffloadingHandlers(
        kv_cache_config=kv_cache_config,
        gpu_block_size=block_size,
        cpu_block_size=block_size * 2,
        num_cpu_blocks_per_group=32,
        gpu_caches=gpu_caches,
        attn_backends=attn_backends,
    )

    # Should only have 2 groups (group 1 was skipped)
    assert len(handlers) == 2
    assert 0 in handlers.group_handlers
    assert 1 not in handlers.group_handlers  # Group 1 skipped
    assert 2 in handlers.group_handlers


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="CUDA required for CpuGpuOffloadingHandlers",
)
@torch.inference_mode()
def test_hybrid_handlers_single_group(default_vllm_config):
    """
    Tests HybridCpuGpuOffloadingHandlers with single group
    (backward compatibility with non-HMA models).
    """
    # Create config with 1 group (standard non-HMA model)
    kv_cache_config = create_mock_kv_cache_config(
        groups=[
            ["layer.0", "layer.1", "layer.2", "layer.3"],
        ]
    )

    # Create GPU caches for all layers
    num_blocks = 64
    block_size = 16
    num_heads = 8
    head_size = 64
    dtype = torch.bfloat16
    device = "cuda:0"

    gpu_caches = {}
    attn_backends = {}
    for layer_name in ["layer.0", "layer.1", "layer.2", "layer.3"]:
        gpu_cache_shape = FlashAttentionBackend.get_kv_cache_shape(
            num_blocks, block_size, num_heads, head_size
        )
        gpu_caches[layer_name] = torch.rand(gpu_cache_shape, dtype=dtype, device=device)
        attn_backends[layer_name] = FlashAttentionBackend

    # Create hybrid handlers
    handlers = HybridCpuGpuOffloadingHandlers(
        kv_cache_config=kv_cache_config,
        gpu_block_size=block_size,
        cpu_block_size=block_size * 2,
        num_cpu_blocks_per_group=32,
        gpu_caches=gpu_caches,
        attn_backends=attn_backends,
    )

    # Should have exactly 1 group
    assert len(handlers) == 1
    assert handlers.get_group_ids() == [0]

    # Group 0 should have all 4 layers
    group0_handlers = handlers.group_handlers[0]
    assert len(group0_handlers.gpu_to_cpu_handler.src_tensors) == 4


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="CUDA required for CpuGpuOffloadingHandlers",
)
@torch.inference_mode()
def test_hybrid_handlers_get_missing_group(default_vllm_config):
    """
    Tests that accessing a non-existent group raises KeyError.
    """
    kv_cache_config = create_mock_kv_cache_config(
        groups=[
            ["layer.0"],
        ]
    )

    num_blocks = 64
    block_size = 16
    dtype = torch.bfloat16
    device = "cuda:0"

    gpu_cache_shape = FlashAttentionBackend.get_kv_cache_shape(
        num_blocks, block_size, 8, 64
    )
    gpu_caches = {"layer.0": torch.rand(gpu_cache_shape, dtype=dtype, device=device)}
    attn_backends = {"layer.0": FlashAttentionBackend}

    handlers = HybridCpuGpuOffloadingHandlers(
        kv_cache_config=kv_cache_config,
        gpu_block_size=block_size,
        cpu_block_size=block_size * 2,
        num_cpu_blocks_per_group=32,
        gpu_caches=gpu_caches,
        attn_backends=attn_backends,
    )

    # Accessing existing group should work
    assert handlers.get_gpu_to_cpu_handler(0) is not None

    # Accessing non-existent group should raise KeyError
    with pytest.raises(KeyError):
        handlers.get_gpu_to_cpu_handler(1)

    with pytest.raises(KeyError):
        handlers.get_cpu_to_gpu_handler(99)
