# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from types import SimpleNamespace

import pytest
import torch

from tests.v1.attention.utils import dense_kv_cache_views
from vllm.v1.kv_cache_interface import (
    ChunkedLocalAttentionSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheLayout,
    SlidingWindowSpec,
)
from vllm.v1.worker import utils as worker_utils
from vllm.v1.worker.utils import (
    AttentionGroup,
    KVBlockZeroer,
    _zero_kv_blocks_kernel,
)


def _large_page_two_group_config() -> KVCacheConfig:
    """Two KV cache groups under the large-page allocator.

    Group 0 (attention): 2 kernel rows per manager block, row of 1536 bytes,
    which does not divide the 4096-byte page. Group 1 (sliding window): 1 row
    per block, row of 512 bytes.
    """
    return KVCacheConfig(
        num_blocks=4,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=["a.0"],
                kv_cache_spec=FullAttentionSpec(
                    block_size=8, num_kv_heads=1, head_size=8, dtype=torch.float16
                ),
                num_blocks=5,
                slots_per_large_page=1,
                kernel_rows_per_block=2,
                row_bytes=1536,
            ),
            KVCacheGroupSpec(
                layer_names=["s.0"],
                kv_cache_spec=SlidingWindowSpec(
                    block_size=8,
                    num_kv_heads=1,
                    head_size=8,
                    dtype=torch.float16,
                    sliding_window=16,
                ),
                num_blocks=32,
                slots_per_large_page=8,
                kernel_rows_per_block=1,
                row_bytes=512,
            ),
        ],
        allocator="large_page",
        large_page_bytes=4096,
    )


def _large_page_zeroers(
    cfg: KVCacheConfig, device: torch.device
) -> tuple[list[KVBlockZeroer], list[torch.Tensor]]:
    """One zeroer per group on that group's kernel-row views (the runner's
    large-page wiring), plus the per-group views for content assertions."""
    kernel_block_sizes = [
        g.kv_cache_spec.block_size // g.kernel_rows_per_block
        for g in cfg.kv_cache_groups
    ]
    zeroers = []
    views = []
    for group_id, group in enumerate(cfg.kv_cache_groups):
        num_rows = cfg.num_blocks * cfg.large_page_bytes // group.row_bytes
        row_el = group.row_bytes // 4
        view = torch.ones((num_rows, row_el), dtype=torch.int32, device=device)
        views.append(view)
        layer_name = group.layer_names[0]
        zeroers.append(
            KVBlockZeroer(
                device,
                attn_groups_iter=[
                    AttentionGroup(None, [layer_name], group.kv_cache_spec, group_id)
                ],
                kernel_block_sizes=kernel_block_sizes,
                static_forward_context={layer_name: SimpleNamespace(kv_cache=view)},
                num_blocks=num_rows,
                kernel_rows_per_block=group.kernel_rows_per_block,
            )
        )
    return zeroers, views


def test_large_page_zeroer_maps_manager_ids_to_kernel_rows(monkeypatch):
    """Per-group zeroers map (group, manager block b) to the block's kernel
    rows b * R_g .. b * R_g + R_g - 1."""
    cfg = _large_page_two_group_config()
    device = torch.device("cpu")
    zeroers, _ = _large_page_zeroers(cfg, device)

    # Segments step by one kernel row, not by a manager block.
    for zeroer, group in zip(zeroers, cfg.kv_cache_groups):
        assert zeroer._meta is not None
        _, seg_block_strides, seg_page_sizes, *_ = zeroer._meta
        assert seg_block_strides.tolist() == [group.row_bytes // 4]
        assert seg_page_sizes.tolist() == [group.row_bytes // 4]

    uploaded: list[torch.Tensor] = []

    class FakeKernel:
        def __getitem__(self, grid):
            return lambda *args, **kwargs: None

    monkeypatch.setattr(worker_utils, "_zero_kv_blocks_kernel", FakeKernel())
    monkeypatch.setattr(
        worker_utils,
        "async_tensor_h2d",
        lambda values, **kwargs: uploaded.append(
            torch.as_tensor(values, dtype=torch.int64)
        ),
    )

    zeroers[0].zero_block_ids([1, 4])
    zeroers[1].zero_block_ids([3, 19])

    assert uploaded[0].tolist() == [2, 3, 8, 9]
    assert uploaded[1].tolist() == [3, 19]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_large_page_zeroing_covers_only_allocated_rows():
    """Zeroing a block of one group zeroes exactly that group's kernel rows
    for the block; other rows of the group and the other group are untouched,
    and block 0 of each group maps to the head of its row grid (page 0)."""
    cfg = _large_page_two_group_config()
    device = torch.device("cuda")
    zeroers, views = _large_page_zeroers(cfg, device)

    zeroers[0].zero_block_ids([3])
    zeroers[1].zero_block_ids([7])
    torch.accelerator.synchronize()

    for group_id, (view, block_id) in enumerate(zip(views, (3, 7))):
        group = cfg.kv_cache_groups[group_id]
        first_row = block_id * cfg.kv_cache_groups[group_id].kernel_rows_per_block
        expected = torch.ones_like(view)
        expected[first_row : first_row + group.kernel_rows_per_block] = 0
        assert torch.equal(view, expected), f"group {group_id}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_large_page_zero_kernel_rows_covers_page_zero():
    """zero_null_page re-zeroes the reserved page 0 after dummy writes, per
    group on that group's kernel-row grid. Only rows fully inside page 0 are
    zeroed (floor): a row crossing the page end would extend into page 1,
    which another group may own with live data. Page 0 stays byte-complete
    because a group whose row divides the page (group 1 here) tiles it."""
    cfg = _large_page_two_group_config()
    device = torch.device("cuda")
    zeroers, views = _large_page_zeroers(cfg, device)
    for view in views:
        view.fill_(7)

    worker_utils.zero_null_page(zeroers, cfg)
    torch.accelerator.synchronize()

    for group_id, (view, group) in enumerate(zip(views, cfg.kv_cache_groups)):
        rows_in_page = cfg.large_page_bytes // group.row_bytes
        # The null block's rows [0, R_g) stay covered.
        assert rows_in_page >= group.kernel_rows_per_block
        expected = torch.full_like(view, 7)
        expected[:rows_in_page] = 0
        assert torch.equal(view, expected), f"group {group_id}"
    # Group 0's row size does not divide the page: the row at the floor
    # boundary starts inside page 0 but extends past it, so it is untouched.
    boundary = cfg.large_page_bytes // cfg.kv_cache_groups[0].row_bytes
    assert cfg.large_page_bytes % cfg.kv_cache_groups[0].row_bytes != 0
    assert (views[0][boundary] == 7).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "spec",
    [
        SlidingWindowSpec(
            block_size=2,
            num_kv_heads=1,
            head_size=1,
            dtype=torch.uint8,
            sliding_window=4,
        ),
        ChunkedLocalAttentionSpec(
            block_size=2,
            num_kv_heads=1,
            head_size=1,
            dtype=torch.uint8,
            attention_chunk_size=4,
        ),
    ],
    ids=["sliding-window", "chunked-local"],
)
def test_attention_blocks_are_zeroed(spec):
    device = torch.device("cuda")
    storage = torch.ones((4, 1, 2, 2), dtype=torch.uint8, device=device)
    layer_name = "draft.self_attn"
    zeroer = KVBlockZeroer(
        device,
        attn_groups_iter=[AttentionGroup(None, [layer_name], spec, 0)],
        kernel_block_sizes=[2],
        static_forward_context={
            layer_name: SimpleNamespace(kv_cache=storage),
        },
        num_blocks=4,
    )

    zeroer.zero_block_ids([1])
    torch.accelerator.synchronize()

    expected = torch.ones_like(storage)
    expected[1] = 0
    assert torch.equal(storage, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_layers_in_one_group_may_use_different_kernel_pages_per_block():
    """Derive kernel pages per block from each layer's allocation."""
    device = torch.device("cuda")
    num_blocks = 4
    spec = SlidingWindowSpec(
        block_size=2,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.int32,
        sliding_window=2,
    )
    # Two kernel pages per logical block.
    wide = torch.ones((num_blocks * 2, 3), dtype=torch.int32, device=device)
    # One kernel page per logical block, carved out of a larger allocation so
    # an over-strided write lands in the guard region instead of faulting or
    # silently hitting another tensor.
    narrow_backing = torch.ones((num_blocks * 3, 5), dtype=torch.int32, device=device)
    narrow = narrow_backing[:num_blocks]

    zeroer = KVBlockZeroer(
        device,
        attn_groups_iter=[
            AttentionGroup(
                None,
                ["wide", "narrow"],
                spec,
                0,
            )
        ],
        kernel_block_sizes=[1],
        static_forward_context={
            "wide": SimpleNamespace(kv_cache=wide),
            "narrow": SimpleNamespace(kv_cache=narrow),
        },
        num_blocks=num_blocks,
    )

    zeroer.zero_block_ids([num_blocks - 1])
    torch.accelerator.synchronize()

    expected_wide = torch.ones_like(wide)
    expected_wide[2 * (num_blocks - 1) :] = 0
    assert torch.equal(wide, expected_wide)

    expected_narrow = torch.ones_like(narrow_backing)
    expected_narrow[num_blocks - 1] = 0
    assert torch.equal(narrow_backing, expected_narrow)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_block_ids_are_not_overwritten_while_copy_is_in_flight():
    device = torch.device("cuda")
    num_blocks = 4
    page_size_el = 4
    storage = torch.ones((num_blocks, page_size_el), dtype=torch.int32, device=device)

    # Build the minimal zeroer state directly so the test can focus on the
    # in-flight copy behavior without constructing model attention groups.
    zeroer = KVBlockZeroer.__new__(KVBlockZeroer)
    zeroer.device = device
    zeroer.kernel_rows_per_block = 1
    zeroer._meta = (
        torch.tensor([storage.data_ptr()], dtype=torch.uint64, device=device),
        torch.tensor([page_size_el], dtype=torch.int64, device=device),
        torch.tensor([page_size_el], dtype=torch.int64, device=device),
        page_size_el // page_size_el,  # max_chunks = 1
        page_size_el,  # blk_size
        1,  # n_segs
    )

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        # Keep the first nonblocking H2D copy pending while the host submits the
        # second call. Each call must stage from its own pinned source so the
        # first copy is not corrupted before it runs.
        torch.cuda._sleep(10_000_000)
        zeroer.zero_block_ids([1])
        zeroer.zero_block_ids([2])
    stream.synchronize()

    assert torch.all(storage[0] == 1)
    assert torch.all(storage[1] == 0)
    assert torch.all(storage[2] == 0)
    assert torch.all(storage[3] == 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_non_uniform_page_sizes():
    """Two segments with different page sizes (e.g. MLA + DSA indexer)."""
    device = torch.device("cuda")
    num_blocks = 4
    page_size_a = 10496  # int32 elements
    page_size_b = 2112

    storage_a = torch.ones((num_blocks, page_size_a), dtype=torch.int32, device=device)
    storage_b = torch.ones((num_blocks, page_size_b), dtype=torch.int32, device=device)

    zeroer = KVBlockZeroer.__new__(KVBlockZeroer)
    zeroer.device = device
    zeroer.kernel_rows_per_block = 1

    seg_page_sizes = [page_size_a, page_size_b]
    max_ps = max(seg_page_sizes)

    blk_size = min(1 << (max_ps - 1).bit_length(), 1024)

    zeroer._meta = (
        torch.tensor(
            [storage_a.data_ptr(), storage_b.data_ptr()],
            dtype=torch.uint64,
            device=device,
        ),
        torch.tensor(seg_page_sizes, dtype=torch.int64, device=device),
        torch.tensor(seg_page_sizes, dtype=torch.int64, device=device),
        (max_ps + blk_size - 1) // blk_size,
        blk_size,
        2,
    )

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        zeroer.zero_block_ids([1, 2])
    stream.synchronize()

    for storage in (storage_a, storage_b):
        assert torch.all(storage[0] == 1)
        assert torch.all(storage[1] == 0)
        assert torch.all(storage[2] == 0)
        assert torch.all(storage[3] == 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_packed_segment_zeros_only_its_last_block_page():
    """A packed KV segment steps by block stride but clears only its page."""
    device = torch.device("cuda")
    num_blocks = 4
    block_stride_el = 12
    page_size_el = 4
    page_offset_el = 3
    backing = torch.ones(
        (num_blocks, block_stride_el), dtype=torch.int32, device=device
    )

    zeroer = KVBlockZeroer.__new__(KVBlockZeroer)
    zeroer.device = device
    zeroer.kernel_rows_per_block = 1
    zeroer._meta = (
        torch.tensor(
            [backing.data_ptr() + page_offset_el * backing.element_size()],
            dtype=torch.uint64,
            device=device,
        ),
        torch.tensor([block_stride_el], dtype=torch.int64, device=device),
        torch.tensor([page_size_el], dtype=torch.int64, device=device),
        1,
        page_size_el,
        1,
    )

    zeroer.zero_block_ids([num_blocks - 1])
    torch.accelerator.synchronize()

    expected = torch.ones_like(backing)
    expected[-1, page_offset_el : page_offset_el + page_size_el] = 0
    assert torch.equal(backing, expected)


def test_large_dsv4_launch_geometry(monkeypatch):
    """Keep the failing DSV4 shape efficient and within launch limits."""
    device = torch.device("cpu")
    n_blocks, n_segs = 6870, 181
    layer_names = [f"layer.{i}" for i in range(n_segs)]
    page_sizes = [9344 if i % 2 == 0 else 292 for i in range(n_segs)]
    spec = SlidingWindowSpec(
        block_size=1,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.int32,
        sliding_window=1,
    )
    storages = {
        name: torch.ones((1, page_size), dtype=torch.int32)
        for name, page_size in zip(layer_names, page_sizes)
    }
    zeroer = KVBlockZeroer(
        device,
        attn_groups_iter=[
            AttentionGroup(None, [name], spec, group_id)
            for group_id, name in enumerate(layer_names)
        ],
        kernel_block_sizes=[1] * n_segs,
        static_forward_context={
            name: SimpleNamespace(kv_cache=storage)
            for name, storage in storages.items()
        },
        num_blocks=1,
    )

    assert zeroer._meta is not None
    _, _, seg_page_sizes, max_chunks, blk_size, n_segs = zeroer._meta
    assert seg_page_sizes.tolist() == page_sizes
    assert (max_chunks, blk_size, n_segs) == (10, 1024, 181)

    captured_grids = []

    class FakeKernel:
        def __getitem__(self, grid):
            captured_grids.append(grid)
            return lambda *args, **kwargs: None

    monkeypatch.setattr(worker_utils, "_zero_kv_blocks_kernel", FakeKernel())
    monkeypatch.setattr(
        worker_utils,
        "async_tensor_h2d",
        lambda values, **kwargs: torch.tensor(values, dtype=torch.int64),
    )

    zeroer.zero_block_ids(list(range(n_blocks)))

    old_max_chunks = max(page_sizes) // 4
    assert math.prod((n_blocks, n_segs, old_max_chunks)) > 2**31 - 1
    assert captured_grids == [(n_blocks, n_segs, max_chunks)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_warmup_compiles_for_all_block_counts():
    """After warmup, no launch should trigger a first-request JIT compile.

    The block count is carried by the launch grid, so changing it must reuse
    the warmup's compiled kernel.
    """
    device = torch.device("cuda")
    num_blocks = 64
    page_size_el = 4
    storage = torch.ones((num_blocks, page_size_el), dtype=torch.int32, device=device)

    zeroer = KVBlockZeroer.__new__(KVBlockZeroer)
    zeroer.device = device
    zeroer.kernel_rows_per_block = 1
    zeroer._meta = (
        torch.tensor([storage.data_ptr()], dtype=torch.uint64, device=device),
        torch.tensor([page_size_el], dtype=torch.int64, device=device),
        torch.tensor([page_size_el], dtype=torch.int64, device=device),
        1,  # max_chunks
        page_size_el,  # blk_size
        1,  # n_segs
    )

    def compiled_variants() -> set:
        return {
            key
            for caches in _zero_kv_blocks_kernel.device_caches.values()
            for key in caches[0]
        }

    zeroer.warmup(num_blocks)
    torch.accelerator.synchronize()
    warmed = compiled_variants()
    assert warmed

    for n_blocks in (1, 2, 3, 16, 32):
        zeroer.zero_block_ids(list(range(n_blocks)))
    torch.accelerator.synchronize()

    assert compiled_variants() == warmed


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_warmup_respects_available_block_count():
    """An empty KV cache must not be warmed with out-of-range block IDs."""
    device = torch.device("cuda")
    page_size_el = 4
    storage = torch.ones((1, page_size_el), dtype=torch.int32, device=device)

    zeroer = KVBlockZeroer.__new__(KVBlockZeroer)
    zeroer.device = device
    zeroer.kernel_rows_per_block = 1
    zeroer._meta = (
        torch.tensor([storage.data_ptr()], dtype=torch.uint64, device=device),
        torch.tensor([page_size_el], dtype=torch.int64, device=device),
        torch.tensor([page_size_el], dtype=torch.int64, device=device),
        1,
        page_size_el,
        1,
    )

    zeroer.warmup(0)
    torch.accelerator.synchronize()

    assert torch.all(storage == 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("layout", list(KVCacheLayout))
def test_zeroes_exactly_one_block_per_layer(layout: KVCacheLayout):
    """The zeroer must zero every byte of the target block in every layer and nothing
    outside it — per head-group region under LHBNC, and never past the target block's
    tile under block-major layouts (no out-of-bounds writes, no clobbering)."""
    device = torch.device("cuda")
    num_blocks, num_layers = 4, 2
    spec = FullAttentionSpec(
        block_size=4, num_kv_heads=2, head_size=8, dtype=torch.float32
    )
    raw = torch.empty(
        num_blocks * num_layers * spec.page_size_bytes,
        dtype=torch.int8,
        device=device,
    ).fill_(1)
    views = dense_kv_cache_views(raw, spec, num_blocks, num_layers, layout)
    groups = [
        AttentionGroup(
            backend=None,
            layer_names=[f"layer.{i}" for i in range(num_layers)],
            kv_cache_spec=spec,
            kv_cache_group_id=0,
        )
    ]
    ctx = {f"layer.{i}": SimpleNamespace(kv_cache=views[i]) for i in range(num_layers)}
    zeroer = KVBlockZeroer(
        device,
        attn_groups_iter=iter(groups),
        kernel_block_sizes=[spec.block_size],
        static_forward_context=ctx,
        num_blocks=num_blocks,
    )
    zeroer.zero_block_ids([2])
    torch.accelerator.synchronize()

    for view in views:
        assert (view[2] == 0).all(), layout
        for b in (0, 1, 3):
            assert (view[b].view(torch.int8) == 1).all(), layout
    zero_bytes = int((raw == 0).sum().item())
    assert zero_bytes == num_layers * spec.page_size_bytes, layout
