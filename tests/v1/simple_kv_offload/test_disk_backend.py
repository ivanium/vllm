# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the disk backend's slot layout math (no CUDA needed)."""

import pytest

from vllm.v1.kv_cache_interface import large_page_block_range
from vllm.v1.simple_kv_offload.disk_backend import _ALIGNMENT, plan_disk_layout

# 2048-byte pages. FA group: one 2048-byte block per page; SWA group: 768-byte
# blocks, so block 2 crosses the end of page 0.
LARGE_PAGE_BYTES = 2048
BLOCK_BYTES = [2048, 768]


def test_plan_disk_layout_pads_slots_for_direct_io() -> None:
    """O_DIRECT needs aligned offsets and lengths: slot strides pad up, and
    the padded page size still fits every block id of a P-page pool."""
    layout = plan_disk_layout(LARGE_PAGE_BYTES, BLOCK_BYTES, use_page_cache=False)
    assert layout.slot_bytes == (4096, 4096)
    assert layout.buf_slot_bytes == 4096
    for g in (0, 1):
        for b in (0, 1, 3, 4, 5):
            assert layout.block_offset(g, b) % _ALIGNMENT == 0
        for num_pages in range(1, 8):
            num_blocks = num_pages * LARGE_PAGE_BYTES // BLOCK_BYTES[g]
            last_end = layout.block_offset(g, num_blocks - 1) + layout.slot_bytes[g]
            assert last_end <= num_pages * layout.page_bytes


def test_plan_disk_layout_page_cache_keeps_dense_slots() -> None:
    """Buffered I/O has no alignment constraint: slots are the blocks and the
    file is no larger than the GPU pool."""
    layout = plan_disk_layout(LARGE_PAGE_BYTES, BLOCK_BYTES, use_page_cache=True)
    assert layout.slot_bytes == (2048, 768)
    assert layout.page_bytes <= LARGE_PAGE_BYTES


@pytest.mark.parametrize("use_page_cache", [False, True])
@pytest.mark.parametrize("block_bytes", [BLOCK_BYTES, [4097, 1 << 20]])
def test_disk_blocks_stay_inside_their_page(use_page_cache, block_bytes) -> None:
    """Pages of different groups never overlap in the file, whatever padding
    each group's slots take."""
    page_bytes = max(block_bytes)
    layout = plan_disk_layout(page_bytes, block_bytes, use_page_cache)
    for g, b in enumerate(block_bytes):
        for page in range(4):
            for block_id in large_page_block_range(page, page_bytes, b):
                start = layout.block_offset(g, block_id)
                assert start >= page * layout.page_bytes
                assert start + layout.slot_bytes[g] <= (page + 1) * layout.page_bytes


def test_disk_block_offsets_within_a_group_are_disjoint() -> None:
    """Blocks of one group never overlap in the file."""
    layout = plan_disk_layout(LARGE_PAGE_BYTES, BLOCK_BYTES, use_page_cache=False)
    for g in (0, 1):
        offsets = sorted(layout.block_offset(g, b) for b in (0, 1, 3, 4, 5))
        for offset, next_offset in zip(offsets, offsets[1:]):
            assert offset + layout.slot_bytes[g] <= next_offset
