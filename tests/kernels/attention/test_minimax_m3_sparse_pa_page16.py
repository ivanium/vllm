# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the MiniMax-M3 AITER page-16 cache geometry helpers.

The asm kernels themselves are ROCm-only, but the page numbering they are
fed -- the page-16 views, the writer's slot rebase, the attend's block-table
rebase -- is pure torch and must hold for both dense caches and the strided
per-layer windows the large-page allocator hands out.
"""

import pytest
import torch

from vllm.models.minimax_m3.amd.ops.sparse_pa import (
    minimax_m3_rebase_block_table_to_page16,
    minimax_m3_sparse_pa_page16_views,
)
from vllm.models.minimax_m3.common.sparse_attention import (
    minimax_m3_page16_pages_per_block,
    minimax_m3_rebase_slots_to_page16,
)

PAGE = 16
BLOCK_SIZE = 128
HEAD_DIM = 128
NUM_KV_HEADS = 1
CONTENT = NUM_KV_HEADS * HEAD_DIM
PAGES_IN_SIDE = BLOCK_SIZE // PAGE


def _large_page_layer_view(
    buf: torch.Tensor,
    num_blocks: int,
    num_layers: int,
    layer_idx: int,
) -> torch.Tensor:
    """One layer's ``[B, 2, N, C]`` window into a shared group buffer."""
    row = num_layers * 2 * BLOCK_SIZE * CONTENT
    return buf.as_strided(
        (num_blocks, 2, BLOCK_SIZE, CONTENT),
        (row, BLOCK_SIZE * CONTENT, CONTENT, 1),
        storage_offset=layer_idx * 2 * BLOCK_SIZE * CONTENT,
    )


def test_slot_rebase_dense_default_unchanged():
    slots = torch.tensor([0, 15, 16, 127, 128, 129, -1])
    actual = minimax_m3_rebase_slots_to_page16(slots, block_size=BLOCK_SIZE)
    expected = torch.tensor([0, 15, 16, 127, 256, 257, -1])
    assert torch.equal(actual, expected)


def test_slot_rebase_rescales_block_component_to_group_row():
    # A 2-layer packed group row spans 4 sides = 32 page-16 pages per block.
    slots = torch.tensor([0, 5, 127, 128, 200, 256, -1, -9])
    actual = minimax_m3_rebase_slots_to_page16(
        slots, block_size=BLOCK_SIZE, pages_per_block=32
    )
    expected = torch.tensor([0, 5, 127, 512, 584, 1024, -1, -9])
    assert torch.equal(actual, expected)


def test_slot_rebase_writes_into_out_buffer():
    slots = torch.tensor([128, -1])
    out = torch.zeros(2, dtype=torch.int64)
    actual = minimax_m3_rebase_slots_to_page16(
        slots, BLOCK_SIZE, pages_per_block=32, out=out
    )
    assert actual.data_ptr() == out.data_ptr()
    assert torch.equal(out, torch.tensor([512, -1]))


def test_pages_per_block16_dense_packed():
    kv = torch.zeros(3, 2, BLOCK_SIZE, CONTENT)
    assert minimax_m3_page16_pages_per_block(kv) == 2 * PAGES_IN_SIDE


def test_pages_per_block16_strided_group_row():
    num_blocks, num_layers = 3, 2
    buf = torch.zeros(num_blocks * num_layers * 2 * BLOCK_SIZE * CONTENT)
    kv = _large_page_layer_view(buf, num_blocks, num_layers, layer_idx=1)
    assert minimax_m3_page16_pages_per_block(kv) == 2 * num_layers * PAGES_IN_SIDE


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.uint8])
def test_page16_views_dense_packed_unchanged(dtype):
    num_blocks = 3
    kv = torch.zeros(num_blocks, 2, BLOCK_SIZE, CONTENT, dtype=dtype)
    k, v, pages_per_block = minimax_m3_sparse_pa_page16_views(
        kv, NUM_KV_HEADS, HEAD_DIM
    )
    x = PAGE // kv.element_size()
    num_phys16 = num_blocks * 2 * PAGES_IN_SIDE
    legacy_k = kv.view(num_phys16, NUM_KV_HEADS, HEAD_DIM // x, PAGE, x)
    legacy_v = kv.view(num_phys16, NUM_KV_HEADS, PAGE // x, HEAD_DIM, x)[PAGES_IN_SIDE:]
    assert pages_per_block == 2 * PAGES_IN_SIDE
    assert k.is_contiguous() and v.is_contiguous()
    assert k.shape == legacy_k.shape and k.stride() == legacy_k.stride()
    assert v.shape == legacy_v.shape and v.stride() == legacy_v.stride()
    assert k.data_ptr() == legacy_k.data_ptr()
    assert v.data_ptr() == legacy_v.data_ptr()


def test_page16_views_dense_separate_planes():
    num_blocks = 3
    raw = torch.zeros(2, num_blocks, BLOCK_SIZE, CONTENT)
    kv = raw.permute(1, 0, 2, 3)
    key_cache, value_cache = kv.unbind(1)
    assert key_cache.is_contiguous() and value_cache.is_contiguous()
    k, v, pages_per_block = minimax_m3_sparse_pa_page16_views(
        kv, NUM_KV_HEADS, HEAD_DIM
    )
    assert pages_per_block == PAGES_IN_SIDE
    assert k.shape[0] == num_blocks * PAGES_IN_SIDE
    assert v.shape[0] == k.shape[0]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.uint8])
def test_page16_views_large_page_block_addressing(dtype):
    num_blocks, num_layers = 3, 2
    row = num_layers * 2 * BLOCK_SIZE * CONTENT
    buf = torch.zeros(num_blocks * row, dtype=dtype)
    kv = _large_page_layer_view(buf, num_blocks, num_layers, layer_idx=1)
    k, v, pages_per_block = minimax_m3_sparse_pa_page16_views(
        kv, NUM_KV_HEADS, HEAD_DIM
    )
    assert pages_per_block == 2 * num_layers * PAGES_IN_SIDE
    assert k.shape[0] == (num_blocks - 1) * pages_per_block + 2 * PAGES_IN_SIDE
    assert v.shape[0] == k.shape[0] - PAGES_IN_SIDE

    # Page (block 2, page 3) of this layer's K side must land in layer 1's
    # K pages of group row 2, leaving the gaps (other layers) untouched.
    blk, j = 2, 3
    k[blk * pages_per_block + j].fill_(1)
    start = 2 * BLOCK_SIZE * CONTENT + blk * row + j * PAGE * CONTENT
    assert buf[start : start + PAGE * CONTENT].eq(1).all()
    assert buf[:start].eq(0).all()
    assert buf[start + PAGE * CONTENT :].eq(0).all()


def test_page16_views_large_page_shared_id_reaches_v_side():
    num_blocks, num_layers = 2, 2
    row = num_layers * 2 * BLOCK_SIZE * CONTENT
    buf = torch.zeros(num_blocks * row)
    kv = _large_page_layer_view(buf, num_blocks, num_layers, layer_idx=0)
    _, v, pages_per_block = minimax_m3_sparse_pa_page16_views(
        kv, NUM_KV_HEADS, HEAD_DIM
    )
    # The page id that reads K page j of a block reads V page j through the
    # offset V view.
    blk, j = 1, 5
    v[blk * pages_per_block + j].fill_(1)
    start = blk * row + (PAGES_IN_SIDE + j) * PAGE * CONTENT
    assert buf[start : start + PAGE * CONTENT].eq(1).all()
    assert buf[:start].eq(0).all()
    assert buf[start + PAGE * CONTENT :].eq(0).all()


def test_slot_rebase_lands_on_view_pages():
    num_blocks, num_layers = 2, 2
    row = num_layers * 2 * BLOCK_SIZE * CONTENT
    buf = torch.zeros(num_blocks * row)
    kv = _large_page_layer_view(buf, num_blocks, num_layers, layer_idx=0)
    k, _, pages_per_block = minimax_m3_sparse_pa_page16_views(
        kv, NUM_KV_HEADS, HEAD_DIM
    )
    # Block 1, page 3: slot = 1 * 128 + 3 * 16.
    slots = torch.tensor([BLOCK_SIZE + 3 * PAGE])
    rebased = minimax_m3_rebase_slots_to_page16(
        slots, BLOCK_SIZE, pages_per_block=pages_per_block
    )
    k[rebased[0].item() // PAGE].fill_(1)
    start = row + 3 * PAGE * CONTENT
    assert buf[start : start + PAGE * CONTENT].eq(1).all()
    assert buf[:start].eq(0).all()
    assert buf[start + PAGE * CONTENT :].eq(0).all()


def test_block_table_rebase_dense_default_unchanged():
    block_table = torch.tensor([[0, 1, 5]])
    actual = minimax_m3_rebase_block_table_to_page16(block_table)
    assert torch.equal(actual, block_table * 2)


def test_block_table_rebase_scales_to_physical_row():
    block_table = torch.tensor([[0, 1, 5]])
    actual = minimax_m3_rebase_block_table_to_page16(block_table, pages_per_block=32)
    assert torch.equal(actual, block_table * 4)
