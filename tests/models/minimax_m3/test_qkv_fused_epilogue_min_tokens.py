# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_MINIMAX_M3_QKV_FUSED_EPILOGUE_MIN_TOKENS on the CPU.

The variable sets the smallest batch whose qkv_proj runs the fused QKV+indexer
epilogue (512 when unset). These tests parse it and run the real
`MiniMaxM3SparseAttention.forward` with the compiled insert, the CuTe-DSL
epilogue, qkv_proj and attention replaced by recorders, checking which writer
each batch takes and that the unfused call is unchanged. The kernels' own
numerics are covered by test_qkv_index_fused_epilogue.py (SM100), which already
runs M = 1, 37 and 128.
"""

from types import SimpleNamespace as NS

import pytest
import torch

import vllm.models.minimax_m3.nvidia.model as m
from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp8Dynamic

ENV = "VLLM_MINIMAX_M3_QKV_FUSED_EPILOGUE_MIN_TOKENS"
NQ, NKV, NIQ, HD, BS, NB, HIDDEN = 64, 4, 4, 128, 128, 4, 6144


@pytest.mark.parametrize(
    "value, expected",
    [(None, 512), ("", 512), ("1", 1), ("192", 192), ("512", 512), ("4096", 4096)],
)
def test_min_tokens_parsing(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv(ENV, raising=False)
    else:
        monkeypatch.setenv(ENV, value)
    assert m._qkv_fused_epilogue_min_tokens(True) == expected
    assert m._qkv_fused_epilogue_min_tokens(False) == expected


@pytest.mark.parametrize("value", ["0", "-8", "abc", "1.5", " "])
def test_min_tokens_rejects_non_positive(monkeypatch, value):
    monkeypatch.setenv(ENV, value)
    with pytest.raises(ValueError, match=ENV):
        m._qkv_fused_epilogue_min_tokens(True)


class _Recorder:
    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))


def _layer(min_tokens: int, use_fused: bool = True):
    attn = object.__new__(m.MiniMaxM3SparseAttention)
    torch.nn.Module.__init__(attn)
    kv = torch.zeros((NB, 2 * NKV, BS, 72), dtype=torch.uint8)
    ic = torch.zeros((NB, BS, 72), dtype=torch.uint8)
    norm = NS(weight=torch.ones(HD), variance_epsilon=1e-6)
    attn.__dict__.update(
        layer_name="l0.attn",
        use_qkv_fused_epilogue=use_fused,
        qkv_fused_epilogue_min_tokens=min_tokens,
        num_heads=NQ,
        num_kv_heads=NKV,
        num_idx_heads=NIQ,
        q_size=NQ * HD,
        index_q_size=NIQ * HD,
        hidden_size=HIDDEN,
        kv_cache=kv,
        kv_cache_dtype="nvfp4",
        use_nvfp4_kv=True,
        _k_scale=torch.tensor(1.0),
        _v_scale=torch.tensor(1.0),
        _q_scale_float=1.0,
        _k_scale_float=1.0,
        _v_scale_float=1.0,
        impl=NS(use_cutlass_decode=True),
        rotary_emb=NS(cos_sin_cache=torch.zeros(8, 64), rotary_dim=64),
        q_norm=norm,
        k_norm=norm,
        index_q_norm=norm,
        index_k_norm=norm,
        indexer=NS(
            index_cache=NS(
                prefix="l0.idx", kv_cache=ic, query_dtype=torch.float8_e4m3fn
            )
        ),
        msa_mxfp8_output=False,
        use_sequence_parallel=False,
    )
    qkv_calls = _Recorder()

    def qkv_proj(h):
        qkv_calls(h)
        n = h.orig_shape[0] if isinstance(h, QuantizedActivation) else h.shape[0]
        return torch.zeros(n, (NQ + 2 * NKV + NIQ + 1) * HD, dtype=torch.bfloat16), None

    attn.qkv_proj = qkv_proj
    attn.o_proj = lambda x: (x, None)
    attn._run_attention = lambda q, iq, out: out
    return attn, qkv_calls


def _run(monkeypatch, attn, num_tokens, quantized=False, autotuning=False):
    insert, fused = _Recorder(), _Recorder()
    monkeypatch.setattr(m.ops, "fused_minimax_m3_qknorm_rope_kv_insert", insert)
    monkeypatch.setattr(attn, "_qkv_proj_fused_epilogue", fused)
    slots = torch.arange(num_tokens) % (NB * BS)
    monkeypatch.setattr(
        m,
        "get_forward_context",
        lambda: NS(slot_mapping={"l0.attn": slots, "l0.idx": slots}),
    )
    monkeypatch.setattr(m, "is_flashinfer_autotuning", lambda: autotuning)
    if quantized:
        hidden = QuantizedActivation(
            data=torch.zeros(num_tokens, HIDDEN, dtype=torch.float8_e4m3fn),
            scale=torch.zeros(-(-num_tokens // 128) * 128 * HIDDEN // 32).to(
                torch.uint8
            ),
            orig_dtype=torch.bfloat16,
            orig_shape=torch.Size((num_tokens, HIDDEN)),
            quant_key=kMxfp8Dynamic,
        )
    else:
        hidden = torch.zeros(num_tokens, HIDDEN, dtype=torch.bfloat16)
    out = m.MiniMaxM3SparseAttention.forward(attn, torch.zeros(num_tokens), hidden)
    assert out.shape == (num_tokens, NQ * HD)
    return insert.calls, fused.calls, slots


@pytest.mark.parametrize(
    "min_tokens, num_tokens, autotuning, expect_fused",
    [
        # 512 (unset): every decode batch keeps the unfused insert
        (512, 1, False, False),
        (512, 8, False, False),
        (512, 192, False, False),
        (512, 511, False, False),
        (512, 512, False, True),
        (512, 4096, False, True),
        # 1: every batch takes the fused epilogue
        (1, 1, False, True),
        (1, 8, False, True),
        (1, 192, False, True),
        (1, 4096, False, True),
        # FlashInfer autotune dummy runs keep the unfused path at any threshold
        (1, 192, True, False),
        (1, 8192, True, False),
        # an intermediate threshold splits at its value
        (64, 56, False, False),
        (64, 64, False, True),
    ],
)
@pytest.mark.parametrize("quantized", [False, True])
def test_forward_routes_by_min_tokens(
    monkeypatch, min_tokens, num_tokens, autotuning, expect_fused, quantized
):
    attn, qkv_calls = _layer(min_tokens)
    insert, fused, slots = _run(monkeypatch, attn, num_tokens, quantized, autotuning)
    assert len(fused) == int(expect_fused)
    assert len(insert) == int(not expect_fused)
    assert len(qkv_calls.calls) == int(not expect_fused)
    if expect_fused:
        args, kwargs = fused[0]
        _, positions, q, index_q, sm, ism = args
        assert q.shape == (num_tokens, NQ * HD) and q.dtype == torch.float8_e4m3fn
        assert index_q.shape == (num_tokens, NIQ * HD)
        assert sm is slots and ism is slots
    else:
        args, kwargs = insert[0]
        assert args[0].shape[0] == num_tokens  # qkv
        assert args[12] is slots and args[13] is slots
        # E4M3 q (CUTLASS decode): nothing reads qkv back, as before
        assert kwargs["write_qkv"] is False
        assert kwargs["q_fp8_out"].dtype == torch.float8_e4m3fn


def test_layer_without_fused_epilogue_ignores_threshold(monkeypatch):
    attn, _ = _layer(1, use_fused=False)
    insert, fused, _ = _run(monkeypatch, attn, 8)
    assert len(insert) == 1 and not fused
