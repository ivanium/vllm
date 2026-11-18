# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The low-batch sparse attention core of MiniMax-M3.1-Flash decode steps.

After the stock qkv projection and KV insert, for 1-32 query rows (up to 32
requests of up to 8 queries) of one TP rank (16 query heads on one KV head,
or 8 at TP8 with ``VLLM_MINIMAX_M3_LOW_BATCH_ATTN_PAD_HEADS=1``; one index
head, NVFP4 KV and index caches, E4M3 queries), three PDL-chained launches a
layer replace the stock indexer, top-k, sparse FMHA, o_proj input
quantization and GEMM:

1. ``indexer_cute.IndexScores``: every history page's index score, from the
   step's list of pages (``indexer_cute.IndexWork``, once a step in
   ``prepare_step``, which also writes each row's record);
2. ``sparse_cute.SparseAttention``: each row's top-15 pages and local page,
   the attention over them, o_proj's MXFP8 input;
3. ``o_proj``: FlashInfer's low-latency MXFP8 GEMM (only its activation
   loads wait for the attention; its weight loads do not, and past eight
   rows the attention prefetches them into L2).

Every launch is sized from the SM count and the step's request and query
counts; the context lengths and pages come from the device seq_lens and block
table only, so a CUDA graph captured once serves every later context length
up to 8192 pages.
"""

import math
from functools import cache
from typing import Any, NamedTuple

import torch

from vllm.logger import init_logger
from vllm.models.common.ops.moe_tail_fusion.primitives import current_cu_stream
from vllm.platforms import current_platform

from .. import (
    LOW_BATCH_ATTN_CLUSTER,
    LOW_BATCH_ATTN_PAD_HEADS,
    attention_cluster,
    attention_pads_heads,
)
from .o_proj import compile_all, mxfp8_weights, o_proj_gemm

logger = init_logger(__name__)

MAX_ROWS = 32
MAX_REQS = 32
MAX_QUERY_LEN = 8
HEADS = 16  # the kernels' query heads on one KV head
TP8_HEADS = 8  # a TP8 rank's, the first half of the kernels' 16
HEAD_DIM = 128
HIDDEN = 6144
PAGE = 128
MAX_PAGES = 8192
INDEX_CTAS_PER_SM = 4
ATTENTION_CTAS = 8  # CTAs a row, two pages each


class AttentionStep(NamedTuple):
    """One decode step's attention metadata, shared by every layer."""

    num_reqs: int
    query_len: int
    # The pages the indexer scores and the rows' records
    # (``indexer_cute.IndexWork``).
    work: torch.Tensor
    records: torch.Tensor


def prepare_step(
    block_table: torch.Tensor, seq_lens: torch.Tensor, query_len: int
) -> AttentionStep:
    """The step of ``seq_lens.shape[0]`` requests of ``query_len`` queries:
    ``block_table`` int32 [requests, pages] and ``seq_lens`` int32
    [requests] (0 for padding requests), both read in place. Lists the pages
    to score and each row's record on the current stream."""
    from cutlass import Int32, Int64

    num_reqs = seq_lens.shape[0]
    assert block_table.dtype == torch.int32 and seq_lens.dtype == torch.int32
    assert block_table.shape[0] == num_reqs and block_table.stride(1) == 1
    assert block_table.shape[1] <= MAX_PAGES and seq_lens.is_contiguous()
    assert 0 < num_reqs <= MAX_REQS and 0 < query_len <= MAX_QUERY_LEN
    scratch = _scratch(seq_lens.device)
    _step_kernels(seq_lens.device.index)[1](
        Int64(block_table.data_ptr()),
        Int32(block_table.stride(0)),
        Int64(seq_lens.data_ptr()),
        Int64(scratch.work.data_ptr()),
        Int64(scratch.records.data_ptr()),
        Int32(num_reqs),
        Int32(query_len),
        current_cu_stream(),
    )
    return AttentionStep(num_reqs, query_len, scratch.work, scratch.records)


class _Kernels(NamedTuple):
    sms: int
    work: Any
    indexer: Any
    attention: Any
    # Whether ``attention`` is the cluster kernel (LOW_BATCH_ATTN_CLUSTER).
    cluster: bool = False


@cache
def _kernels(device_index: int, heads: int = HEADS, cluster: bool = False) -> _Kernels:
    """The step's, the indexer's and (for ``heads`` query heads a row) the
    attention's compiled kernels: ``cluster``, each row's CTAs as one thread
    block cluster (``sparse_cluster_cute.py``), else ``SparseAttention``."""
    import cutlass.cute as cute
    from cutlass import Float32, Int32, Int64
    from cutlass.cute.runtime import make_fake_stream

    from .sparse_cute import SparseAttention

    sms, work, indexer = _step_kernels(device_index)
    args = (
        *(Int64(0),) * 11,
        Float32(0.0),
        *(Int64(0), Int32(0)) * 2,
        Int32(1),
        make_fake_stream(),
    )
    with torch.accelerator.device_index(device_index):
        if cluster:
            from .sparse_cluster_cute import ClusterSparseAttention

            try:
                attention = cute.compile(ClusterSparseAttention(heads), *args)
            except Exception as error:  # a DSL or driver without the cluster path
                logger.warning_once(
                    "MiniMax-M3 low-batch attention cluster kernel not used (%s=1): "
                    "%s: %s; each sparse attention CTA ranks the row's pages and the "
                    "row's last CTA merges them, as without the switch.",
                    LOW_BATCH_ATTN_CLUSTER,
                    type(error).__name__,
                    str(error)[:200],
                    scope="process",
                )
            else:
                logger.info_once(
                    "MiniMax-M3 low-batch attention: each row's %d sparse attention "
                    "CTAs run as one thread block cluster (%s=1): they rank the "
                    "row's pages together and merge through distributed shared "
                    "memory, %d query heads a row.",
                    ATTENTION_CTAS,
                    LOW_BATCH_ATTN_CLUSTER,
                    heads,
                    scope="process",
                )
                return _Kernels(sms, work, indexer, attention, True)
        attention = cute.compile(SparseAttention(ATTENTION_CTAS, heads), *args)
    return _Kernels(sms, work, indexer, attention)


@cache
def _step_kernels(device_index: int) -> tuple[int, Any, Any]:
    """The kernels every head count shares: the step's work list and the
    indexer (and o_proj's)."""
    import cutlass.cute as cute
    from cutlass import Int32, Int64
    from cutlass.cute.runtime import make_fake_stream

    from .indexer_cute import IndexScores, IndexWork

    sms = torch.cuda.get_device_properties(device_index).multi_processor_count
    with torch.accelerator.device_index(device_index):
        work = cute.compile(
            IndexWork(),
            Int64(0),
            Int32(0),
            *(Int64(0),) * 3,
            Int32(1),
            Int32(1),
            make_fake_stream(),
        )
        indexer = cute.compile(
            IndexScores(INDEX_CTAS_PER_SM * sms),
            *(Int64(0),) * 5,
            Int32(1),
            make_fake_stream(),
        )
    compile_all(device_index)
    return sms, work, indexer


class _Scratch(NamedTuple):
    """Buffers the layers reuse in turn, and the step's metadata."""

    # (FP32 index score, int32 physical page) [rows, pages].
    scores: torch.Tensor
    # o_proj's MXFP8 input and its scales (128x4 layout).
    act: torch.Tensor
    act_scale: torch.Tensor
    # The attention CTAs' partials of a row, and per row the CTAs done.
    partial: torch.Tensor
    counters: torch.Tensor
    work: torch.Tensor
    records: torch.Tensor


@cache
def _scratch(device: torch.device) -> _Scratch:
    from .indexer_cute import RECORD_BYTES, SCORE_STRIDE, work_entries
    from .sparse_cute import PARTIAL_BYTES

    ctas = (
        INDEX_CTAS_PER_SM
        * torch.cuda.get_device_properties(device).multi_processor_count
    )
    return _Scratch(
        scores=torch.empty(MAX_ROWS, SCORE_STRIDE, 2, dtype=torch.int32, device=device),
        act=torch.zeros(MAX_ROWS, HEADS * HEAD_DIM, dtype=torch.uint8, device=device),
        act_scale=torch.zeros(
            128 * HEADS * HEAD_DIM // 32, dtype=torch.uint8, device=device
        ),
        partial=torch.empty(PARTIAL_BYTES, dtype=torch.uint8, device=device),
        counters=torch.zeros(MAX_ROWS, dtype=torch.int32, device=device),
        work=torch.zeros(work_entries(ctas) + 1, 2, dtype=torch.int32, device=device),
        records=torch.zeros(MAX_ROWS, RECORD_BYTES, dtype=torch.uint8, device=device),
    )


class LowBatchAttention:
    """One sparse attention layer's low-batch path (module docstring)."""

    def __init__(self, layer, weight: torch.Tensor, weight_scale: torch.Tensor):
        # The attention prefetches both into L2 (16-byte aligned sizes).
        for t in (weight, weight_scale):
            assert t.data_ptr() % 16 == 0 and t.nbytes % 16 == 0
        self.logit_scale = (
            float(layer.impl.scale) * layer._q_scale_float * math.log2(math.e)
        )
        self.k_scale = layer._k_scale
        self.v_scale = layer._v_scale
        self.weight = weight
        self.weight_scale = weight_scale
        # 16, or 8 (TP8): o_proj's input is [rows, heads x 128] (its K).
        self.heads = layer.num_heads
        self.kernels = _kernels(weight.device.index, self.heads, attention_cluster())
        # Outside any capture: a buffer first made inside one would not be
        # zeroed for the graphs captured later.
        _scratch(weight.device)

    def accepts(self, num_reqs: int, query_len: int) -> bool:
        return (
            0 < query_len <= MAX_QUERY_LEN
            and 0 < num_reqs <= MAX_REQS
            and num_reqs * query_len <= MAX_ROWS
        )

    def run(
        self,
        step: AttentionStep,
        q: torch.Tensor,
        index_q: torch.Tensor,
        kv_cache: torch.Tensor,
        index_cache: torch.Tensor,
    ) -> torch.Tensor:
        """o_proj's BF16 output [rows, 6144] of the E4M3 queries ``q`` [rows,
        heads x 128] and index queries ``index_q`` [rows, 128] (request-major) on the
        NVFP4 ``kv_cache`` [pages, 2, 128, 72] and ``index_cache`` [pages,
        128, 72], on the current stream."""
        from cutlass import Float32, Int32, Int64

        rows = step.num_reqs * step.query_len
        q_size = self.heads * HEAD_DIM
        assert q.dtype == torch.float8_e4m3fn and q.shape == (rows, q_size)
        assert index_q.dtype == torch.float8_e4m3fn
        assert index_q.shape == (rows, HEAD_DIM)
        assert q.is_contiguous() and index_q.is_contiguous()
        assert kv_cache.shape[1:] == (2, PAGE, 72) and index_cache.shape[1:] == (
            PAGE,
            72,
        )
        assert kv_cache[0].is_contiguous() and index_cache[0].is_contiguous()
        scratch = _scratch(q.device)
        stream = current_cu_stream()
        self.kernels.indexer(
            Int64(index_q.data_ptr()),
            Int64(index_cache.data_ptr()),
            Int64(index_cache.stride(0)),
            Int64(step.work.data_ptr()),
            Int64(scratch.scores.data_ptr()),
            Int32(step.query_len),
            stream,
        )
        self.kernels.attention(
            Int64(q.data_ptr()),
            Int64(kv_cache.data_ptr()),
            Int64(kv_cache.stride(0)),
            Int64(step.records.data_ptr()),
            Int64(scratch.scores.data_ptr()),
            Int64(self.k_scale.data_ptr()),
            Int64(self.v_scale.data_ptr()),
            Int64(scratch.act.data_ptr()),
            Int64(scratch.act_scale.data_ptr()),
            Int64(scratch.partial.data_ptr()),
            Int64(scratch.counters.data_ptr()),
            Float32(self.logit_scale),
            Int64(self.weight.data_ptr()),
            Int32(self.weight.nbytes),
            Int64(self.weight_scale.data_ptr()),
            Int32(self.weight_scale.nbytes),
            Int32(rows),
            stream,
        )
        out = torch.empty(rows, HIDDEN, dtype=torch.bfloat16, device=q.device)
        o_proj_gemm(
            self.weight,
            self.weight_scale,
            scratch.act,
            scratch.act_scale,
            out,
            rows,
            stream,
        )
        return out


def query_heads() -> tuple[int, ...]:
    """The query heads per rank the low-batch attention takes."""
    return (HEADS, TP8_HEADS) if attention_pads_heads() else (HEADS,)


def unsupported_reason(layer) -> str | None:
    """Why ``layer`` (a ``MiniMaxM3SparseAttention``) cannot take the
    low-batch attention, or None."""
    if layer.num_heads not in query_heads():
        unset = f" ({LOW_BATCH_ATTN_PAD_HEADS} unset)"
        return f"{layer.num_heads} query heads per rank" + (
            unset if layer.num_heads == TP8_HEADS else ""
        )
    if (
        layer.num_kv_heads != 1
        or layer.num_idx_heads != 1
        or layer.head_dim != HEAD_DIM
        or layer.q_size != layer.num_heads * HEAD_DIM
    ):
        return "not one KV head and one index head of 128"
    if layer.kv_cache_dtype != "nvfp4" or layer.impl.topk_blocks != 16:
        return "not an NVFP4 KV cache with 16 top-k pages"
    if layer._k_scale_float != 1.0 or layer._v_scale_float != 1.0:
        return "KV global scales other than 1"
    if mxfp8_weights(layer.o_proj, HIDDEN, layer.num_heads * HEAD_DIM) is None:
        return "o_proj is not a FlashInfer cute-dsl MXFP8 linear without bias"
    return None


def build_low_batch_attention(layer) -> LowBatchAttention | None:
    """``layer``'s (a ``MiniMaxM3SparseAttention``) low-batch attention, or
    None when it does not have the shapes and caches the kernels serve (said
    once per process). Compiles the kernels once per process; call outside
    CUDA graph capture."""
    if not (
        current_platform.is_cuda() and current_platform.is_device_capability_family(100)
    ):
        return None
    reason = unsupported_reason(layer)
    if reason is not None:
        logger.warning_once(
            "MiniMax-M3 low-batch attention not used: %s; the sparse attention "
            "layers keep the stock attention.",
            reason,
            scope="process",
        )
        return None
    weights = mxfp8_weights(layer.o_proj, HIDDEN, layer.num_heads * HEAD_DIM)
    assert weights is not None
    return LowBatchAttention(layer, *weights)
