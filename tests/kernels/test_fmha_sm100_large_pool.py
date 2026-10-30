# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""fmha_sm100 paged sparse decode on pool views spanning more than 2^31 elements.

Under the large-page KV allocator, per-layer K/V views are strided slices of
one whole-pool buffer, so block-axis offsets exceed 2^31 elements. The TMA
descriptors compute those offsets in 64 bits. The page stride itself (one row)
stays far below 2^31 elements, the width of the kernel's stride fields.
"""

import pytest
import torch

from vllm.platforms import current_platform

if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
    pytest.skip(
        "fmha_sm100 requires an SM100 GPU.",
        allow_module_level=True,
    )

from vllm.third_party.fmha_sm100.api import fmha_sm100, fmha_sm100_plan

DEVICE = torch.device("cuda", torch.accelerator.current_device_index())
FP8 = torch.float8_e4m3fn
BATCH = 2
NUM_QO_HEADS = 16
NUM_KV_HEADS = 1
HEAD_DIM = 128
PAGE_SIZE = 128
TOPK = 16
PAGES_PER_REQ = 2


def _check_large_pool_pages(row_stride: int, num_pages: int, page_ids: list[int]):
    """Paged sparse decode on a strided pool view vs a dense packing.

    ``row_stride`` is the pool view's page stride in fp8 elements. ``page_ids``
    are the physical pages holding data; ``kv_indices`` maps each request's
    logical pages onto them. With ``num_pages * row_stride`` (or
    ``row_stride`` alone) past 2^31, any 32-bit stride truncation or
    page*stride product misaddresses the tail pages.
    """
    torch.manual_seed(0)
    kv_indices_list = (page_ids * BATCH)[: BATCH * PAGES_PER_REQ]
    need = num_pages * row_stride
    free, _ = torch.accelerator.get_memory_info()
    if free < 2 * need + (1 << 30):
        pytest.skip("not enough free device memory")

    k_pool = torch.empty(need, dtype=FP8, device=DEVICE)
    v_pool = torch.empty(need, dtype=FP8, device=DEVICE)
    view_shape = (num_pages, NUM_KV_HEADS, PAGE_SIZE, HEAD_DIM)
    view_strides = (row_stride, PAGE_SIZE * HEAD_DIM, HEAD_DIM, 1)
    k_big = k_pool.as_strided(view_shape, view_strides)
    v_big = v_pool.as_strided(view_shape, view_strides)

    k_ref = torch.randn(
        len(page_ids),
        NUM_KV_HEADS,
        PAGE_SIZE,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device=DEVICE,
    ).to(FP8)
    v_ref = torch.randn(
        len(page_ids),
        NUM_KV_HEADS,
        PAGE_SIZE,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device=DEVICE,
    ).to(FP8)
    for i, p in enumerate(page_ids):
        k_big[p].copy_(k_ref[i])
        v_big[p].copy_(v_ref[i])
    # Same physical page ids read back through the dense packing.
    ref_of_phys = {p: i for i, p in enumerate(page_ids)}
    kv_indices_ref_list = [ref_of_phys[p] for p in kv_indices_list]

    q = (
        torch.randn(BATCH, NUM_QO_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)
        * 0.5
    ).to(FP8)

    qo_lens = torch.ones(BATCH, dtype=torch.int32)
    kv_lens = torch.full((BATCH,), PAGES_PER_REQ * PAGE_SIZE, dtype=torch.int32)
    plan = fmha_sm100_plan(
        qo_lens,
        kv_lens,
        NUM_QO_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        qo_offset=kv_lens - qo_lens,
        page_size=PAGE_SIZE,
        output_maxscore=False,
        kv_block_num=TOPK,
        causal=True,
        sparse_kernel_mode="decode",
        use_fp8_kvcache=True,
        num_kv_splits=1,
        device=DEVICE,
    )

    kv_block_indexes = torch.full((BATCH, NUM_KV_HEADS, TOPK), -1, dtype=torch.int32)
    kv_block_indexes[:, :, :PAGES_PER_REQ] = torch.arange(
        PAGES_PER_REQ, dtype=torch.int32
    )
    kv_block_indexes = kv_block_indexes.to(DEVICE)

    def run(k, v, kv_indices_list_):
        out = torch.empty(
            BATCH, NUM_QO_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=DEVICE
        )
        fmha_sm100(
            q,
            k,
            v,
            plan,
            kv_indices=torch.tensor(kv_indices_list_, dtype=torch.int32, device=DEVICE),
            kv_block_indexes=kv_block_indexes,
            out=out,
            output_maxscore=False,
            output_o=True,
            sm_scale=HEAD_DIM**-0.5,
            q_scale=1.0,
            k_scale=1.0,
            v_scale=1.0,
            o_scale=1.0,
        )
        return out

    out_big = run(k_big, v_big, kv_indices_list)
    out_ref = run(k_ref, v_ref, kv_indices_ref_list)
    torch.accelerator.synchronize()
    assert torch.equal(out_big, out_ref)


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="Skipping CUDA-only tests."
)
@torch.inference_mode()
def test_fmha_sm100_paged_sparse_decode_page_span_over_int32():
    # ~2300 pages with a 1 MiB row stride: the view spans 2.4e9 fp8 elements,
    # past 2^31. Pages at both ends of the block axis are attended.
    row_stride = 1 << 20
    num_pages = 2304
    assert num_pages * row_stride > 1 << 31
    _check_large_pool_pages(
        row_stride, num_pages, page_ids=[0, 1, num_pages - 2, num_pages - 1]
    )

