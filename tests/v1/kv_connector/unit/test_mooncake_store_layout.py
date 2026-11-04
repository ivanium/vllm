# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for Mooncake Store payload layouts under the large-page allocator.

A manager block of a group is ``kernel_rows_per_block`` contiguous kernel
rows of ``row_bytes`` in the group's shared backing allocation; block ``b``
starts at byte ``b * kernel_rows_per_block * row_bytes``.
"""

import ctypes

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (
    ChunkedTokenDatabase,
    KeyMetadata,
    LBHNCStoreLayout,
    LBNHCStoreLayout,
    RankLocalStoreLayout,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    GroupGeometry,
    KVCacheGroupSpec,
)

BLOCK_SIZE = 16
NUM_PAGES = 8
PAGE_BYTES = 2048


def _make_group(
    layer_names=("l0",),
    rows_per_block: int = 2,
    row_bytes: int = 768,
) -> KVCacheGroupSpec:
    spec = FullAttentionSpec(
        block_size=BLOCK_SIZE, num_kv_heads=4, head_size=24, dtype=torch.float16
    )
    return KVCacheGroupSpec(
        list(layer_names),
        spec,
        num_blocks=NUM_PAGES,
        kernel_rows_per_block=rows_per_block,
        row_bytes=row_bytes,
    )


def _make_views(
    group: KVCacheGroupSpec,
    num_pages: int = NUM_PAGES,
    page_bytes: int = PAGE_BYTES,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """One strided fp16 view per layer over a shared large-page pool buffer."""
    buf = torch.zeros(num_pages * page_bytes, dtype=torch.int8)
    row_elems = group.row_bytes // 2
    per_layer = row_elems // len(group.layer_names)
    rows = num_pages * page_bytes // group.row_bytes
    views = [
        torch.as_strided(
            buf.view(torch.float16),
            (rows, per_layer),
            (row_elems, 1),
            storage_offset=i * per_layer,
        )
        for i in range(len(group.layer_names))
    ]
    return buf, views


def _rank_local_layout(group: KVCacheGroupSpec) -> RankLocalStoreLayout:
    layout = RankLocalStoreLayout(
        KeyMetadata("test-model", 1, 0, 0, 0),
        BLOCK_SIZE,
        geometry=GroupGeometry(group.kernel_rows_per_block, group.row_bytes),
    )
    _, views = _make_views(group)
    layout.register_kv_caches(views)
    return layout


def test_rank_local_descriptors_match_reference():
    """Block b is one segment at base + b * block_bytes."""
    group = _make_group()  # rows_per_block=2, row_bytes=768
    buf, views = _make_views(group)
    layout = RankLocalStoreLayout(
        KeyMetadata("test-model", 1, 0, 0, 0),
        BLOCK_SIZE,
        geometry=GroupGeometry(group.kernel_rows_per_block, group.row_bytes),
    )
    layout.register_kv_caches(views)
    base = buf.untyped_storage().data_ptr()

    chunks = [(i * BLOCK_SIZE, (i + 1) * BLOCK_SIZE) for i in range(4)]
    addrs, sizes, first_block_ids = layout.prepare_values(chunks, [0, 1, 2, 3], [0] * 4)
    for chunk_addrs, chunk_sizes, block_id in zip(
        addrs, sizes, first_block_ids, strict=True
    ):
        assert chunk_addrs == [base + block_id * 1536]
        assert chunk_sizes == [1536]
        assert type(block_id) is int
        assert all(type(addr) is int for addr in chunk_addrs)

    # A multi-block chunk is one segment per block.
    addrs, sizes, first_block_ids = layout.prepare_values(
        [(2 * BLOCK_SIZE, 4 * BLOCK_SIZE)], [0, 1, 5, 3], [0]
    )
    assert addrs == [[base + 5 * 1536, base + 3 * 1536]]
    assert sizes == [[1536, 1536]]
    assert first_block_ids == [5]

    addrs, sizes = layout.prepare_value_for_block(3)
    assert addrs == [base + 3 * 1536]
    assert sizes == [1536]


def test_rank_local_descriptors_handle_empty_and_invalid_chunks():
    layout = _rank_local_layout(_make_group())
    assert layout.prepare_values([], [1, 2, 3], []) == ([], [], [])
    with pytest.raises(AssertionError):
        layout.prepare_values([(0, BLOCK_SIZE + 1)], [0, 1], [0])


@pytest.mark.parametrize(
    ("layout_cls", "store_format"),
    [
        (LBHNCStoreLayout, "tp_shared_lbhnc"),
        (LBNHCStoreLayout, "tp_shared_lbnhc"),
    ],
)
def test_tp_shared_layout_owns_store_namespace(layout_cls, store_format):
    assert (
        layout_cls.shared_namespace(4, 2)
        == f"@store_tp:4@store_pp:2@store_format:{store_format}"
    )


def test_rank_local_database_api_rejects_tp_shared_layout():
    metadata = KeyMetadata("test-model", 0, 0, 0, 0)
    layout = LBHNCStoreLayout(
        metadata,
        block_size=16,
        local_tp_size=2,
        store_tp_size=4,
        tp_rank=0,
        num_kv_heads=8,
        geometry=GroupGeometry(2, 768),
    )
    database = ChunkedTokenDatabase(metadata, 16, store_layout=layout)

    with pytest.raises(RuntimeError, match="rank-local"):
        database.prepare_value_for_block(0)


def _lp_view(
    buf: torch.Tensor,
    shape: tuple[int, ...],
    strides: tuple[int, ...],
) -> torch.Tensor:
    return torch.as_strided(buf.view(torch.float16), shape, strides)


@pytest.mark.parametrize(
    "layout_cls",
    [LBHNCStoreLayout, LBNHCStoreLayout],
)
@pytest.mark.parametrize("rows_per_block", [1, 2])
def test_tp_shared_layout_round_trip_across_tp_sizes(layout_cls, rows_per_block):
    """Shard bytes flow producer->store->consumer across TP sizes.

    With ``rows_per_block=2`` a block is two kernel rows; the payload must
    still arrive in block token order.
    """
    block_size = 16
    kernel_tokens = block_size // rows_per_block

    def packed_strides(local_heads: int) -> tuple[int, ...]:
        # Element strides of one rank's packed [rows, H, N, C] layer view.
        row = local_heads * kernel_tokens * 4
        if layout_cls is LBHNCStoreLayout:
            return (row, kernel_tokens * 4, 4, 1)
        return (row, 4, local_heads * 4, 1)

    def make_geometry(local_heads: int) -> GroupGeometry:
        return GroupGeometry(rows_per_block, local_heads * kernel_tokens * 4 * 2)

    stored: dict[int, bytes] = {}
    for tp_rank in range(4):
        # data[h, t, c]: the logical block's tokens in order.
        data = (
            torch.arange(2 * block_size * 4, dtype=torch.float16).view(2, block_size, 4)
            + tp_rank * 1000
        )
        buf = torch.zeros(1 << 20, dtype=torch.int8)
        tensor = _lp_view(buf, (rows_per_block, 2, kernel_tokens, 4), packed_strides(2))
        # View rows hold kernel blocks: row r covers tokens [r*k, (r+1)*k).
        tensor.copy_(data.view(2, rows_per_block, kernel_tokens, 4).permute(1, 0, 2, 3))
        metadata = KeyMetadata("test-model", tp_rank, 0, 0, 0)
        layout = layout_cls(
            metadata,
            block_size,
            local_tp_size=4,
            store_tp_size=4,
            tp_rank=tp_rank,
            num_kv_heads=8,
            geometry=make_geometry(2),
        )
        layout.register_kv_caches([tensor])
        addrs, sizes, _ = layout.prepare_values([(0, block_size)], [0], [tp_rank])
        stored[tp_rank] = b"".join(
            ctypes.string_at(addr, size)
            for addr, size in zip(addrs[0], sizes[0], strict=True)
        )

    for tp_rank in range(2):
        buf = torch.zeros(1 << 20, dtype=torch.int8)
        tensor = _lp_view(buf, (rows_per_block, 4, kernel_tokens, 4), packed_strides(4))
        tensor.zero_()
        metadata = KeyMetadata("test-model", tp_rank, 0, 0, 0)
        layout = layout_cls(
            metadata,
            block_size,
            local_tp_size=2,
            store_tp_size=4,
            tp_rank=tp_rank,
            num_kv_heads=8,
            geometry=make_geometry(4),
        )
        layout.register_kv_caches([tensor])
        shard_ids = layout.local_shard_ids
        addrs, sizes, _ = layout.prepare_values(
            [(0, block_size)] * len(shard_ids), [0], shard_ids
        )
        for shard_id, shard_addrs, shard_sizes in zip(
            shard_ids, addrs, sizes, strict=True
        ):
            offset = 0
            for addr, size in zip(shard_addrs, shard_sizes, strict=True):
                ctypes.memmove(addr, stored[shard_id][offset : offset + size], size)
                offset += size

        expected = torch.cat(
            [
                torch.arange(2 * block_size * 4, dtype=torch.float16).view(
                    2, block_size, 4
                )
                + r * 1000
                for r in (tp_rank * 2, tp_rank * 2 + 1)
            ],
            dim=0,
        )
        # Compare in logical (head, token) order.
        actual = (
            tensor.view(rows_per_block, 4, kernel_tokens, 4)
            .permute(1, 0, 2, 3)
            .reshape(4, block_size, 4)
        )
        torch.testing.assert_close(actual, expected)
