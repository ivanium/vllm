# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The MiniMax-M3 low-batch attention (``specialized_decode/attention``)
against the stock decode path it replaces (the MSA decode indexer with its
top-k, the MSA sparse decode and the MXFP8 o_proj), on random NVFP4 KV and
index caches of one TP rank (16 query heads, one KV head, one index head; or
a TP8 rank's 8 query heads with VLLM_MINIMAX_M3_LOW_BATCH_ATTN_PAD_HEADS=1).

``build_low_batch_attention`` reads a fixed set of the layer's attributes
(its o_proj and the attention constants); a full ``MiniMaxM3SparseAttention``
needs a model config, so the layer here carries those attributes around a
real ``RowParallelLinear`` with MXFP8 weights."""

import math
import os
import tempfile
from contextlib import suppress
from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not (
        current_platform.is_cuda() and current_platform.is_device_capability_family(100)
    ),
    reason="the low-batch attention kernels run on SM100-family GPUs",
)

HEADS, HEAD_DIM, HIDDEN, PAGE, TOPK, MAX_PAGES = 16, 128, 6144, 128, 16, 8192
DEVICE = "cuda"
PAD_HEADS = "VLLM_MINIMAX_M3_LOW_BATCH_ATTN_PAD_HEADS"


def _o_proj(quant, heads, raw=None, seed=0):
    """An MXFP8 ``RowParallelLinear`` from ``heads`` x 128 to the hidden size
    with random (or ``raw``: E4M3 weight and E8M0 scale bytes) weights,
    processed for its kernel."""
    from vllm.model_executor.layers.linear import RowParallelLinear
    from vllm.model_executor.layers.quantization.base_config import (
        QuantizeMethodBase,
    )

    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        o_proj = RowParallelLinear(
            heads * HEAD_DIM,
            HIDDEN,
            bias=False,
            quant_config=quant,
            reduce_results=False,
            prefix=f"layers.{seed}.self_attn.o_proj",
        ).to(DEVICE)
    finally:
        torch.set_default_dtype(default_dtype)
    if raw is None:
        generator = torch.Generator(device=DEVICE).manual_seed(seed)
        values = torch.randn(o_proj.weight.shape, generator=generator, device=DEVICE)
        weight = (values * 64).clamp(-448, 448).to(torch.float8_e4m3fn)
        scale = torch.randint(
            114, 118, o_proj.weight_scale.shape, generator=generator, device=DEVICE
        ).to(torch.uint8)
    else:
        weight, scale = raw
    o_proj.weight.data.copy_(weight)
    o_proj.weight_scale.data.copy_(scale.view(o_proj.weight_scale.dtype))
    raw_copy = (weight.clone(), scale.clone())
    if isinstance(o_proj.quant_method, QuantizeMethodBase):
        o_proj.quant_method.process_weights_after_loading(o_proj)
    return o_proj, raw_copy


def _attrs(o_proj, heads):
    return SimpleNamespace(
        o_proj=o_proj,
        num_heads=heads,
        num_kv_heads=1,
        num_idx_heads=1,
        head_dim=HEAD_DIM,
        q_size=heads * HEAD_DIM,
        kv_cache_dtype="nvfp4",
        impl=SimpleNamespace(scale=HEAD_DIM**-0.5, topk_blocks=TOPK),
        _q_scale_float=0.5,
        _k_scale_float=1.0,
        _v_scale_float=1.0,
        _k_scale=torch.ones((), dtype=torch.float32, device=DEVICE),
        _v_scale=torch.ones((), dtype=torch.float32, device=DEVICE),
    )


def _quant():
    from vllm.model_executor.layers.quantization.modelopt import ModelOptMxFp8Config

    return ModelOptMxFp8Config.from_config(
        {
            "quant_method": "mxfp8",
            "weight_block_size": [1, 32],
            "activation_scheme": "dynamic",
            "moe_quant_algo": "NVFP4",
            "moe_quant_format": "nvfp4-pack-quantized",
            "group_size": 16,
            "ignored_layers": [],
        }
    )


@pytest.fixture(scope="module", params=[16, 8], ids=["tp4", "tp8"])
def layer(request):
    """The layer's attributes around an MXFP8 o_proj with random weights, its
    config and its low-batch attention: a TP4 rank's 16 query heads, or a TP8
    rank's 8 (VLLM_MINIMAX_M3_LOW_BATCH_ATTN_PAD_HEADS=1)."""
    heads = request.param
    from vllm.config import (
        KernelConfig,
        ParallelConfig,
        VllmConfig,
        set_current_vllm_config,
    )
    from vllm.distributed import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.models.minimax_m3.nvidia.specialized_decode.attention import (
        build_low_batch_attention,
    )
    from vllm.v1.worker.workspace import init_workspace_manager

    config = VllmConfig(parallel_config=ParallelConfig(), kernel_config=KernelConfig())
    fd, path = tempfile.mkstemp()
    os.close(fd)
    default_dtype = torch.get_default_dtype()
    saved_switch = os.environ.get(PAD_HEADS)
    if heads == 8:
        os.environ[PAD_HEADS] = "1"
    try:
        with set_current_vllm_config(config):
            init_distributed_environment(
                world_size=1,
                rank=0,
                distributed_init_method=f"file://{path}",
                local_rank=0,
                backend="nccl",
            )
            initialize_model_parallel(1, 1)
            init_workspace_manager(torch.device(DEVICE))
            o_proj, raw = _o_proj(_quant(), heads)
            attrs = _attrs(o_proj, heads)
            attrs.raw_o_proj = raw
            yield attrs, config, build_low_batch_attention(attrs)
    finally:
        if saved_switch is None:
            os.environ.pop(PAD_HEADS, None)
        else:
            os.environ[PAD_HEADS] = saved_switch
        torch.set_default_dtype(default_dtype)
        cleanup_dist_env_and_memory()
        with suppress(FileNotFoundError):
            os.remove(path)


def _lengths(context, num_reqs, query_len, generator):
    """Request lengths: ``short`` cycles through one page and the 15- and
    16-history-page edges; ``medium`` draws 40-250 pages (rows ranked across
    the CTA's warps, one pair a thread) and ``long`` a few thousand (32 pairs
    a thread), both with a query window straddling two pages and a
    zero-length padding request last."""
    if context == "short":
        edges = (query_len, 100, 15 * PAGE + 64, 16 * PAGE + 64)
        return [edges[i % 4] for i in range(num_reqs)]
    low, high = (40, 250) if context == "medium" else (1000, 3000)
    lengths = (
        torch.randint(low * PAGE, high * PAGE, (num_reqs,), generator=generator)
        .add_(1)
        .tolist()
    )
    lengths[0] = lengths[0] // PAGE * PAGE + query_len // 2  # straddles two pages
    if num_reqs > 1:
        lengths[-1] = 0
    return lengths


def _randint(low, high, shape, generator):
    return torch.randint(
        low, high, shape, generator=generator, dtype=torch.uint8, device=DEVICE
    )


def _inputs(context, num_reqs, query_len, seed, warm=0, heads=HEADS):
    """Queries, caches, block table and seq_lens of one step. Fifteen pages of
    each request score far above its others (their index keys scaled up), so
    the stock top-k's 16-bit key resolution cannot change the selection;
    every page holds random values of the same scales, so each selected page
    carries about 1/16 of a row's output and a wrong one moves the row by
    about a third. With ``warm``, that many more pages score between the hot
    ones and the rest, and hot and warm pages all lie on 14 of the 32
    residues of the page index mod 32."""
    lengths = _lengths(
        context, num_reqs, query_len, torch.Generator().manual_seed(seed)
    )
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    rows = num_reqs * query_len
    pages = [max(math.ceil(n / PAGE), 1) for n in lengths]
    pool = sum(pages) + 4
    perm = torch.randperm(pool, generator=generator, device=DEVICE).to(torch.int32)
    # Unused block table entries point at random pages.
    block_table = torch.randint(
        0, pool, (num_reqs, MAX_PAGES), generator=generator, device=DEVICE
    ).to(torch.int32)
    kv = _randint(0, 256, (pool, 2, PAGE * 72), generator)
    index = _randint(0, 256, (pool, PAGE * 72), generator)
    codes = PAGE * HEAD_DIM // 2
    # E4M3 scales of K, V and the index keys between 0.25 and 0.9.
    kv[:, :, codes:] = _randint(0x28, 0x37, (pool, 2, PAGE * 8), generator)
    index[:, codes:] = _randint(0x28, 0x37, (pool, PAGE * 8), generator)
    cursor = 0
    for b, n in enumerate(lengths):
        phys = perm[cursor : cursor + pages[b]]
        cursor += pages[b]
        block_table[b, : pages[b]] = phys
        history = (max(n, 1) - query_len) // PAGE  # of the request's first row
        if history > TOPK - 1:
            pages_of = torch.arange(history, device=DEVICE)
            if warm:
                pages_of = pages_of[pages_of % 32 < 14]
            order = torch.randperm(len(pages_of), generator=generator, device=DEVICE)
            hot = pages_of[order]
            index[phys[hot[: TOPK - 1]].long(), codes:] = _randint(
                0x58, 0x68, (TOPK - 1, PAGE * 8), generator
            )
            if warm:
                # Scales 2 to 4: above every cold page, far below the hot ones.
                warm_pages = phys[hot[TOPK - 1 : TOPK - 1 + warm]].long()
                index[warm_pages, codes:] = _randint(
                    0x40, 0x48, (warm, PAGE * 8), generator
                )
    q = torch.randn(rows, heads * HEAD_DIM, generator=generator, device=DEVICE) * 2
    iq = torch.randn(rows, HEAD_DIM, generator=generator, device=DEVICE).abs() + 0.1
    return {
        "q": q.to(torch.float8_e4m3fn),
        "iq": iq.to(torch.float8_e4m3fn),
        "kv": kv.view(pool, 2, PAGE, 72),
        "index": index.view(pool, PAGE, 72),
        "block_table": block_table,
        "seq_lens": torch.tensor(lengths, dtype=torch.int32, device=DEVICE),
        "query_len": query_len,
        "live": torch.tensor(lengths).repeat_interleave(query_len) > 0,
    }


def _stock(attrs, inputs):
    """The stock decode path, as the model calls it for a FULL-graph step."""
    from vllm.models.minimax_m3.nvidia import msa_cutlass_sparse_decode as msa
    from vllm.models.minimax_m3.nvidia.indexer_nvfp4 import _msa_indexers

    block_table, seq_lens = inputs["block_table"], inputs["seq_lens"]
    num_reqs, query_len = seq_lens.shape[0], inputs["query_len"]
    rows = num_reqs * query_len
    heads = attrs.num_heads
    wrapper_cls = _msa_indexers()[0]
    score = torch.empty(
        wrapper_cls.score_buffer_size(num_reqs, MAX_PAGES, num_heads=1),
        dtype=torch.float32,
        device=DEVICE,
    )
    indexer = wrapper_cls(
        num_heads=1,
        use_cuda_graph=True,
        block_table_buffer=block_table,
        seq_lens_buffer=seq_lens,
        score_buffer=score,
    )
    indexer.plan(block_table, seq_lens, query_len=query_len)
    plan = msa.MSACutlassDecodePlanCache().prepare(
        block_table,
        seq_lens,
        seq_lens.cpu(),
        query_len,
        num_q_heads=heads,
        num_kv_heads=1,
        page_size=PAGE,
        topk_blocks=TOPK,
        q8kv4=True,
    )
    topk = torch.empty(
        max(4, (rows + 3) // 4 * 4), 1, TOPK, dtype=torch.int32, device=DEVICE
    )[:rows]
    indexer.run(inputs["iq"].view(-1, 1, HEAD_DIM), inputs["index"], out=topk)
    attn = torch.empty(rows, heads * HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)
    msa.msa_cutlass_sparse_decode(
        inputs["q"].view(-1, heads, HEAD_DIM),
        inputs["kv"],
        topk,
        attn.view(-1, heads, HEAD_DIM),
        plan,
        scale=attrs.impl.scale,
        q_scale_float=attrs._q_scale_float,
        k_scale_float=1.0,
        v_scale_float=1.0,
        k_scale=attrs._k_scale,
        v_scale=attrs._v_scale,
    )
    return attrs.o_proj(attn)[0]


def _check(out, expected, live):
    """Per live row, within the rounding of the two paths: FP8 against F16
    attention probabilities, and the MXFP8 rounding of o_proj's input (about
    4% at most between them). A wrong page moves a row by about a third (see
    ``_inputs``), a wrong mask or scale also far more."""
    for row in torch.nonzero(live).flatten().tolist():
        got, want = out[row].float(), expected[row].float()
        error = float((got - want).norm() / want.norm())
        assert error < 0.06, f"row {row}: relative error {error:.4f}"


def _run(low_batch, inputs):
    from vllm.models.minimax_m3.nvidia.specialized_decode.attention import (
        prepare_step,
    )

    step = prepare_step(inputs["block_table"], inputs["seq_lens"], inputs["query_len"])
    return low_batch.run(step, inputs["q"], inputs["iq"], inputs["kv"], inputs["index"])


@pytest.mark.parametrize("context", ["short", "medium", "long"])
@pytest.mark.parametrize(
    "num_reqs,query_len", [(1, 1), (4, 1), (32, 1), (1, 8), (4, 8)]
)
def test_low_batch_attention_matches_stock(layer, num_reqs, query_len, context):
    """Eager calls at each request count and query length the decode steps
    take, short contexts (every history page selected), medium ones (pages
    ranked across warps) and long ones."""
    from vllm.config import set_current_vllm_config
    from vllm.forward_context import set_forward_context

    attrs, config, low_batch = layer
    assert low_batch is not None and low_batch.accepts(num_reqs, query_len)
    assert low_batch.heads == attrs.num_heads
    inputs = _inputs(
        context,
        num_reqs,
        query_len,
        seed=num_reqs * 10 + query_len,
        heads=attrs.num_heads,
    )
    with (
        set_current_vllm_config(config),
        set_forward_context(None, config),
        torch.inference_mode(),
    ):
        out = _run(low_batch, inputs)
        expected = _stock(attrs, inputs)
        torch.accelerator.synchronize()
    _check(out, expected, inputs["live"])


@pytest.mark.parametrize("warm", [100, 400])
@pytest.mark.parametrize("num_reqs,query_len", [(1, 1), (1, 8)])
def test_low_batch_attention_crowded_ranking(layer, num_reqs, query_len, warm):
    """Long rows whose best pages crowd 14 residues mod 32 (``_inputs``): the
    ranking's bound, the 15th best of the 32 residues' best pages, lets
    about ``warm`` pages through; 100 are placed among each other, 400 are
    more than it holds and the top 15 are extracted one by one."""
    from vllm.config import set_current_vllm_config
    from vllm.forward_context import set_forward_context

    attrs, config, low_batch = layer
    inputs = _inputs(
        "long",
        num_reqs,
        query_len,
        seed=warm + query_len,
        warm=warm,
        heads=attrs.num_heads,
    )
    with (
        set_current_vllm_config(config),
        set_forward_context(None, config),
        torch.inference_mode(),
    ):
        out = _run(low_batch, inputs)
        expected = _stock(attrs, inputs)
        torch.accelerator.synchronize()
    _check(out, expected, inputs["live"])


def test_low_batch_attention_kv_global_scales(layer):
    """The device K and V global scales, which the stock decode reads, scale
    the logits and the output (0.75 and 1.25 here)."""
    from vllm.config import set_current_vllm_config
    from vllm.forward_context import set_forward_context
    from vllm.models.minimax_m3.nvidia.specialized_decode.attention import (
        build_low_batch_attention,
    )

    attrs, config, _ = layer
    scaled = SimpleNamespace(
        **{
            **vars(attrs),
            "_k_scale": torch.full((), 0.75, device=DEVICE),
            "_v_scale": torch.full((), 1.25, device=DEVICE),
        }
    )
    low_batch = build_low_batch_attention(scaled)
    inputs = _inputs("long", 4, 1, seed=5, heads=attrs.num_heads)
    with (
        set_current_vllm_config(config),
        set_forward_context(None, config),
        torch.inference_mode(),
    ):
        out = _run(low_batch, inputs)
        expected = _stock(scaled, inputs)
        torch.accelerator.synchronize()
    _check(out, expected, inputs["live"])


@pytest.mark.parametrize("num_reqs,query_len", [(4, 1), (4, 8)])
def test_low_batch_attention_graph_replay(layer, num_reqs, query_len):
    """One CUDA graph captured on one step and replayed on steps with other
    seq_lens, block tables, caches and queries copied into its buffers."""
    from vllm.config import set_current_vllm_config
    from vllm.forward_context import set_forward_context

    attrs, config, low_batch = layer
    steps = [
        _inputs(c, num_reqs, query_len, seed, heads=attrs.num_heads)
        for seed, c in enumerate(("long", "short", "long"))
    ]
    pool = max(s["kv"].shape[0] for s in steps)
    static = dict(steps[0])
    static["kv"] = torch.zeros(
        (pool, *steps[0]["kv"].shape[1:]), dtype=torch.uint8, device=DEVICE
    )
    static["index"] = torch.zeros(
        (pool, *steps[0]["index"].shape[1:]), dtype=torch.uint8, device=DEVICE
    )
    static["block_table"] = steps[0]["block_table"].clone()
    static["seq_lens"] = steps[0]["seq_lens"].clone()
    static["q"], static["iq"] = steps[0]["q"].clone(), steps[0]["iq"].clone()

    def load(step):
        for name in ("q", "iq", "block_table", "seq_lens"):
            static[name].copy_(step[name])
        static["kv"][: step["kv"].shape[0]].copy_(step["kv"])
        static["index"][: step["index"].shape[0]].copy_(step["index"])

    with (
        set_current_vllm_config(config),
        set_forward_context(None, config),
        torch.inference_mode(),
    ):
        load(steps[0])
        _run(low_batch, static)
        torch.accelerator.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = _run(low_batch, static)
        for step in steps[1:]:
            load(step)
            graph.replay()
            expected = _stock(attrs, dict(static, live=step["live"]))
            torch.accelerator.synchronize()
            _check(out, expected, step["live"])


def test_low_batch_attention_tp8_needs_its_switch(layer):
    """8 query heads per rank (TP8) are refused, and said so, without
    VLLM_MINIMAX_M3_LOW_BATCH_ATTN_PAD_HEADS=1; 16 are taken either way."""
    from vllm.models.minimax_m3.nvidia.specialized_decode.attention import (
        unsupported_reason,
    )

    attrs, _, _ = layer
    saved = os.environ.pop(PAD_HEADS, None)
    try:
        reason = unsupported_reason(attrs)
        if attrs.num_heads == 8:
            assert reason is not None and PAD_HEADS in reason
        else:
            assert reason is None
    finally:
        if saved is not None:
            os.environ[PAD_HEADS] = saved


@pytest.mark.parametrize("context", ["short", "medium", "long"])
@pytest.mark.parametrize("num_reqs,query_len", [(1, 8), (4, 8), (32, 1)])
def test_low_batch_attention_tp8_heads_are_16_head_rows(
    layer, num_reqs, query_len, context
):
    """A TP8 rank's 8 heads are bitwise the first 8 heads of a 16-head row
    whose other 8 hold any queries: o_proj's MXFP8 input and its scales; and
    o_proj on the rank's weights is a 16-head o_proj whose weight columns of
    the other heads are zero (up to the GEMM's summation order)."""
    from vllm.config import set_current_vllm_config
    from vllm.forward_context import set_forward_context
    from vllm.models.minimax_m3.nvidia.specialized_decode.attention import (
        LowBatchAttention,
        _scratch,
    )
    from vllm.models.minimax_m3.nvidia.specialized_decode.attention.o_proj import (
        mxfp8_weights,
    )

    attrs, config, low8 = layer
    if attrs.num_heads != 8:
        pytest.skip("the TP8 rank's heads")
    weight8, scale8 = attrs.raw_o_proj
    # The 16-head o_proj: the rank's columns, then zero columns (scale 2^0);
    # the bytes are joined as uint8.
    bytes8 = weight8.view(torch.uint8)
    raw16 = (
        torch.cat([bytes8, torch.zeros_like(bytes8)], dim=1).view(weight8.dtype),
        torch.cat([scale8, torch.full_like(scale8, 127)], dim=1),
    )
    o_proj16, _ = _o_proj(_quant(), HEADS, raw=raw16)
    attrs16 = _attrs(o_proj16, HEADS)
    low16 = LowBatchAttention(attrs16, *mxfp8_weights(o_proj16, HIDDEN, 2048))
    inputs = _inputs(context, num_reqs, query_len, seed=31 + num_reqs, heads=8)
    rows = num_reqs * query_len
    generator = torch.Generator(device=DEVICE).manual_seed(rows)
    other = (
        torch.randn(rows, 8 * HEAD_DIM, generator=generator, device=DEVICE) * 2
    ).to(torch.float8_e4m3fn)
    q16 = torch.cat(
        [inputs["q"].view(torch.uint8), other.view(torch.uint8)], dim=1
    ).view(torch.float8_e4m3fn)
    scratch = _scratch(inputs["q"].device)  # the buffers run() uses
    with (
        set_current_vllm_config(config),
        set_forward_context(None, config),
        torch.inference_mode(),
    ):
        scratch.act.zero_()
        scratch.act_scale.zero_()
        out8 = _run(low8, inputs)
        torch.accelerator.synchronize()
        act8 = scratch.act.view(-1)[: rows * 1024].view(rows, 1024).clone()
        scale_bytes8 = scratch.act_scale.clone()
        scratch.act.zero_()
        scratch.act_scale.zero_()
        out16 = _run(low16, dict(inputs, q=q16))
        torch.accelerator.synchronize()
        act16 = scratch.act.view(-1)[: rows * 2048].view(rows, 2048).clone()
        scale_bytes16 = scratch.act_scale.clone()
        expected = _stock(attrs, inputs)
        torch.accelerator.synchronize()
    assert torch.equal(act8, act16[:, :1024])
    # The first 32 MXFP8 blocks of each row (the rank's heads): 128x4 layout.
    offsets = torch.tensor(
        [
            (b // 4) * 512 + (r % 32) * 16 + (r // 32) * 4 + b % 4
            for r in range(rows)
            for b in range(32)
        ],
        device=DEVICE,
    )
    assert torch.equal(scale_bytes8[offsets], scale_bytes16[offsets])
    # Nothing past the rank's blocks is written at 8 heads.
    assert not bool(scale_bytes8[4096:].any())
    for row in torch.nonzero(inputs["live"]).flatten().tolist():
        got, want = out8[row].float(), out16[row].float()
        error = float((got - want).norm() / want.norm().clamp_min(1e-30))
        assert error < 1e-2, f"row {row}: o_proj 8 vs 16+zero heads {error:.5f}"
    _check(out8, expected, inputs["live"])
