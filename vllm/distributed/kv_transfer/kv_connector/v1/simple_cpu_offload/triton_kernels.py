# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton kernels for efficient GPU<->CPU block transfers."""

from typing import NamedTuple

import torch

from vllm.logger import init_logger
from vllm.triton_utils import triton

logger = init_logger(__name__)

tl = triton.language


class MultiLayerLaunchParams(NamedTuple):
    """Pre-computed launch parameters for copy_blocks_multi_layer."""

    src_ptr_table: torch.Tensor
    dst_ptr_table: torch.Tensor
    num_layers: int
    words_per_block: int
    block_size: int
    num_warps: int


@triton.jit
def _copy_blocks_kernel(
    src_ptr,
    dst_ptr,
    mapping_ptr,
    words_per_block: tl.constexpr,  # type: ignore[name-defined]
    BLOCK_SIZE: tl.constexpr,  # type: ignore[name-defined]
):
    """
    Triton kernel for copying blocks between GPU and CPU.

    Each program instance copies one block from src to dst.
    The mapping tensor specifies (src_block_id, dst_block_id) pairs.

    Args:
        src_ptr: Pointer to source tensor (flattened as int64)
        dst_ptr: Pointer to destination tensor (flattened as int64)
        mapping_ptr: Pointer to mapping tensor [N, 2] flattened
        words_per_block: Number of int64 words per block
        BLOCK_SIZE: Triton block size for vectorization
    """
    # Program ID corresponds to which block pair to copy
    pid = tl.program_id(0)

    # Load source and destination block IDs from mapping
    # mapping is [N, 2] flattened, so mapping[pid] = (src_id, dst_id)
    src_block = tl.load(mapping_ptr + pid * 2).to(tl.int64)
    dst_block = tl.load(mapping_ptr + pid * 2 + 1).to(tl.int64)

    # Compute base offsets for src and dst blocks
    src_off = src_block * words_per_block
    dst_off = dst_block * words_per_block

    # Copy in chunks of BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    for start in range(0, words_per_block, BLOCK_SIZE):
        idx = start + offsets
        mask = idx < words_per_block
        data = tl.load(src_ptr + src_off + idx, mask=mask, other=0)
        tl.store(dst_ptr + dst_off + idx, data, mask=mask)


def _compute_launch_params(
    words_per_block: int,
) -> tuple[int, int]:
    """Compute Triton launch parameters for block copy kernels."""
    block_size = min(triton.next_power_of_2(words_per_block), 1024)
    num_warps = min(max(block_size // 32, 1), 32)
    return block_size, num_warps


def copy_blocks_triton(
    src_cache: torch.Tensor,
    dst_cache: torch.Tensor,
    block_mapping: torch.Tensor,
) -> None:
    """
    Copy blocks using Triton kernel.

    This function copies blocks from src_cache to dst_cache based on
    the mapping specified in block_mapping.

    Args:
        src_cache: Source KV cache tensor [num_blocks, ...]
        dst_cache: Destination KV cache tensor [num_blocks, ...]
        block_mapping: [N, 2] tensor of (src_block_id, dst_block_id) pairs
    """
    if block_mapping.numel() == 0:
        return

    # Validate inputs
    assert src_cache.is_contiguous(), "Source cache must be contiguous"
    assert dst_cache.is_contiguous(), "Destination cache must be contiguous"
    assert block_mapping.shape[1] == 2, "Mapping must be [N, 2]"

    # Compute words per block (treating data as int64 for max bandwidth)
    block_numel = src_cache[0].numel()  # Elements per block
    bytes_per_block = block_numel * src_cache.element_size()
    words_per_block = bytes_per_block // 8  # int64 words

    if words_per_block == 0:
        logger.warning("Block size too small for Triton kernel")
        return

    # Check alignment for int64 view (bytes must be divisible by 8)
    total_src_bytes = src_cache.numel() * src_cache.element_size()
    total_dst_bytes = dst_cache.numel() * dst_cache.element_size()
    if total_src_bytes % 8 != 0 or total_dst_bytes % 8 != 0:
        logger.warning("Tensor size not aligned to 8 bytes, falling back to PyTorch")
        copy_blocks_torch(src_cache, dst_cache, block_mapping)
        return

    # Flatten tensors as int64 views for maximum transfer bandwidth
    src_flat = src_cache.view(-1).view(torch.int64)
    dst_flat = dst_cache.view(-1).view(torch.int64)

    # Ensure mapping is on CUDA and contiguous
    if not block_mapping.is_cuda:
        block_mapping = block_mapping.cuda()
    mapping_flat = block_mapping.to(torch.int64).contiguous().view(-1)

    # Launch kernel
    num_blocks = block_mapping.shape[0]
    BLOCK_SIZE, num_warps = _compute_launch_params(words_per_block)

    _copy_blocks_kernel[(num_blocks,)](
        src_flat,
        dst_flat,
        mapping_flat,
        words_per_block,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )


@triton.jit
def _copy_blocks_multi_layer_kernel(
    src_ptrs,
    dst_ptrs,
    mapping_ptr,
    words_per_block: tl.constexpr,  # type: ignore[name-defined]
    BLOCK_SIZE: tl.constexpr,  # type: ignore[name-defined]
):
    """
    Triton kernel for copying blocks across multiple layers in one launch.

    Uses pointer tables to handle non-contiguous (permuted) tensors by
    addressing each layer's storage via its base data_ptr and stride.

    Grid: (num_block_pairs, num_layers)

    Args:
        src_ptrs: Pointer to uint64 tensor [num_layers] of source base addrs
        dst_ptrs: Pointer to uint64 tensor [num_layers] of dest base addrs
        mapping_ptr: Pointer to int64 tensor [N * 2] of (src_id, dst_id) pairs
        words_per_block: Number of int64 words per block (stride-based)
        BLOCK_SIZE: Triton block size for vectorization
    """
    pair_id = tl.program_id(0)
    layer_id = tl.program_id(1)

    # Load source and destination block IDs from mapping
    src_block = tl.load(mapping_ptr + pair_id * 2).to(tl.int64)
    dst_block = tl.load(mapping_ptr + pair_id * 2 + 1).to(tl.int64)

    # Load base addresses from pointer tables and cast to typed pointers
    src_base = tl.load(src_ptrs + layer_id).to(tl.pointer_type(tl.int64))
    dst_base = tl.load(dst_ptrs + layer_id).to(tl.pointer_type(tl.int64))

    # Compute offsets using stride-based addressing
    src_off = src_block * words_per_block
    dst_off = dst_block * words_per_block

    # Copy in chunks of BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    for start in range(0, words_per_block, BLOCK_SIZE):
        idx = start + offsets
        mask = idx < words_per_block
        data = tl.load(src_base + src_off + idx, mask=mask, other=0)
        tl.store(dst_base + dst_off + idx, data, mask=mask)


def build_multi_layer_launch_params(
    src_caches: dict[str, torch.Tensor],
    dst_caches: dict[str, torch.Tensor],
) -> MultiLayerLaunchParams:
    """
    Pre-compute launch parameters for copy_blocks_multi_layer.

    Call once at init time and pass the results to copy_blocks_multi_layer
    to avoid per-call overhead of pointer table construction.

    Returns:
        MultiLayerLaunchParams with pointer tables, layer count, and
        kernel configuration.
    """
    assert list(src_caches.keys()) == list(dst_caches.keys()), (
        "src and dst cache dicts must have matching keys in the same order"
    )

    src_tensors = list(src_caches.values())
    dst_tensors = list(dst_caches.values())
    num_layers = len(src_tensors)

    first_tensor = src_tensors[0]
    words_per_block = first_tensor.stride(0) * first_tensor.element_size() // 8

    # Verify all layers have the same stride-based block size
    for t in src_tensors[1:]:
        wpb = t.stride(0) * t.element_size() // 8
        assert wpb == words_per_block, (
            f"Layer stride mismatch: expected {words_per_block} "
            f"int64 words/block, got {wpb}"
        )
    for t in dst_tensors:
        wpb = t.stride(0) * t.element_size() // 8
        assert wpb == words_per_block, (
            f"Layer stride mismatch: expected {words_per_block} "
            f"int64 words/block, got {wpb}"
        )

    src_ptr_table = torch.tensor(
        [t.data_ptr() for t in src_tensors],
        device="cuda",
        dtype=torch.uint64,
    )
    dst_ptr_table = torch.tensor(
        [t.data_ptr() for t in dst_tensors],
        device="cuda",
        dtype=torch.uint64,
    )

    block_size, num_warps = _compute_launch_params(words_per_block)
    return MultiLayerLaunchParams(
        src_ptr_table=src_ptr_table,
        dst_ptr_table=dst_ptr_table,
        num_layers=num_layers,
        words_per_block=words_per_block,
        block_size=block_size,
        num_warps=num_warps,
    )


def copy_blocks_multi_layer(
    src_caches: dict[str, torch.Tensor],
    dst_caches: dict[str, torch.Tensor],
    block_mapping: torch.Tensor,
    *,
    launch_params: MultiLayerLaunchParams | None = None,
) -> None:
    """
    Copy blocks across all layers in a single Triton kernel launch.

    Uses pointer tables and stride-based addressing to handle non-contiguous
    (permuted) tensors without requiring .view(-1).

    Args:
        src_caches: Dict mapping layer name to source KV cache tensor
        dst_caches: Dict mapping layer name to destination KV cache tensor
        block_mapping: [N, 2] tensor of (src_block_id, dst_block_id) pairs
        launch_params: Pre-computed launch parameters from
            build_multi_layer_launch_params(). If None, computed on the fly.
    """
    if block_mapping.numel() == 0:
        return

    if launch_params is None:
        launch_params = build_multi_layer_launch_params(src_caches, dst_caches)

    # Flatten mapping (already int64 contiguous from caller)
    mapping_flat = block_mapping.view(-1)

    # Launch kernel with grid (num_pairs, num_layers)
    num_pairs = block_mapping.shape[0]

    _copy_blocks_multi_layer_kernel[(num_pairs, launch_params.num_layers)](
        launch_params.src_ptr_table,
        launch_params.dst_ptr_table,
        mapping_flat,
        launch_params.words_per_block,
        BLOCK_SIZE=launch_params.block_size,
        num_warps=launch_params.num_warps,
    )


def copy_blocks_torch(
    src_cache: torch.Tensor,
    dst_cache: torch.Tensor,
    block_mapping: torch.Tensor,
) -> None:
    """
    Copy blocks using PyTorch indexing (fallback).

    This is a fallback implementation using standard PyTorch operations.
    It's simpler but may be slower than the Triton kernel.

    Args:
        src_cache: Source KV cache tensor [num_blocks, ...]
        dst_cache: Destination KV cache tensor [num_blocks, ...]
        block_mapping: [N, 2] tensor of (src_block_id, dst_block_id) pairs
    """
    if block_mapping.numel() == 0:
        return

    src_ids = block_mapping[:, 0].tolist()
    dst_ids = block_mapping[:, 1].tolist()

    for src_id, dst_id in zip(src_ids, dst_ids):
        dst_cache[dst_id].copy_(src_cache[src_id], non_blocking=True)


# Choose implementation based on availability
def copy_blocks(
    src_cache: torch.Tensor,
    dst_cache: torch.Tensor,
    block_mapping: torch.Tensor,
    use_triton: bool = True,
) -> None:
    """
    Copy blocks from src to dst based on mapping.

    Args:
        src_cache: Source KV cache tensor
        dst_cache: Destination KV cache tensor
        block_mapping: [N, 2] tensor of (src_id, dst_id) pairs
        use_triton: Whether to use Triton kernel (default: True)
    """
    if use_triton:
        try:
            copy_blocks_triton(src_cache, dst_cache, block_mapping)
        except Exception as e:
            logger.warning("Triton kernel failed, falling back to PyTorch: %s", e)
            copy_blocks_torch(src_cache, dst_cache, block_mapping)
    else:
        copy_blocks_torch(src_cache, dst_cache, block_mapping)
