# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fused attention front of the low-batch decode steps, enabled with
``VLLM_MINIMAX_M3_LOW_BATCH_FUSED_FRONT=1``.

On a sparse layer whose input is the previous MoE layer's unfinalized output,
one launch of vLLM #66324's front kernel (``kernels/front.py``) replaces the
stock input boundary (the MoE top-k finalize + all-reduce + Gemma RMSNorm +
MXFP8 quantization, ``ops/finalize_allreduce_norm.py``) and the qkv_proj GEMM
+ head norms + RoPE + KV and index-K insert ahead of the low-batch attention:
the boundary rows are all-reduced through peer memory, normed and quantized in
the kernel, the QKV projection is a split-K MXFP8 GEMV, and the epilogue writes
the E4M3 queries the low-batch attention reads and the NVFP4 K/V and index-K
pages the stock insert writes, byte for byte (unit K/V global scales, which the
low-batch attention requires too).

It serves steps of up to 16 rows (the kernel's bucket-8 and bucket-16 builds);
larger steps, steps without the low-batch attention and every other layer keep
the stock modules. The build is collective and fail closed: every TP rank
allocates the peer memory and builds the same layers, and a layer that failed
on any rank runs stock on every rank.
"""

import os
from typing import TYPE_CHECKING, NamedTuple

import torch
import torch.distributed as dist

from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.moe_output import (
    MoEOutput,
    UnfinalizedMoEOutput,
)

if TYPE_CHECKING:
    from ...model import MiniMaxM3DecoderLayer
    from .decode import SpecializedDecode

logger = init_logger(__name__)

ENV = "VLLM_MINIMAX_M3_LOW_BATCH_FUSED_FRONT"
HIDDEN = 6144
HEAD_DIM = 128
PAGE = 128
# The front's exchange buckets: rows 1-8 and 9-16 (kernels/_exchange.py).
BUCKETS = (8, 16)
MAX_TOP_K = 8
_ARENA_ALIGN = 256


def requested() -> bool:
    return os.environ.get(ENV, "0") == "1"


def _bf16(t: torch.Tensor) -> torch.Tensor:
    return t.detach().to(torch.bfloat16).contiguous()


class LayerFront(NamedTuple):
    """One sparse layer's front: a kernel instance per bucket, the operands it
    reads and the buffers it writes (row-prefix views per step)."""

    stages: dict
    weights: dict
    r_mid: torch.Tensor  # bf16 [16, H]: the residual after the input boundary
    q_fp8: torch.Tensor  # e4m3 [16, nq * 128]
    index_q: torch.Tensor  # e4m3 [16, niq * 128]


class FusedFront:
    """The built fronts of one executor's sparse layers."""

    def __init__(self, layers: dict[int, LayerFront], max_rows: int, transport):
        self.layers = layers
        self.max_rows = max_rows
        self.transport = transport

    def takes(
        self,
        layer: "MiniMaxM3DecoderLayer",
        pending,
        residual: torch.Tensor | None,
        positions: torch.Tensor,
    ) -> bool:
        """Whether ``layer``'s front runs on this input (host state only, the
        same on every TP rank): the previous MoE left its finalize open."""
        if id(layer) not in self.layers or residual is None:
            return False
        rows = positions.shape[0]
        if not 0 < rows <= self.max_rows or positions.dtype != torch.int64:
            return False
        if not isinstance(pending, MoEOutput) or pending.routed_scaling_factor != 1.0:
            return False
        routed = pending.routed
        if not isinstance(routed, UnfinalizedMoEOutput):
            return False
        fc2, perm = routed.gemm2_permuted, routed.expanded_idx_to_permuted_idx
        weights = routed.expert_weights
        top_k, rest = divmod(perm.numel(), rows)
        shared = pending.shared_output
        return (
            fc2.dtype == torch.bfloat16
            and fc2.dim() == 2
            and fc2.stride() == (HIDDEN, 1)
            and perm.dtype == torch.int32
            and weights.dtype in (torch.float32, torch.bfloat16)
            and weights.numel() == perm.numel()
            and rest == 0
            and 0 < top_k <= MAX_TOP_K
            and (
                shared is None
                or (shared.dtype == torch.bfloat16 and shared.shape == (rows, HIDDEN))
            )
            and residual.dtype == torch.bfloat16
            and residual.shape == (rows, HIDDEN)
        )

    def run(
        self,
        layer: "MiniMaxM3DecoderLayer",
        pending: MoEOutput,
        residual: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """The front of ``layer`` on the current stream: (E4M3 q, E4M3
        index_q, the new residual), after writing the NVFP4 K/V and index-K
        rows. None (nothing launched) when the caches do not have the layout
        the kernel writes."""
        from ..kernels._exchange import bucket
        from ..kernels.front import FinalizeIn

        lf = self.layers[id(layer)]
        attn = layer.self_attn
        rows = positions.shape[0]
        routed = pending.routed
        assert isinstance(routed, UnfinalizedMoEOutput)
        perm = routed.expanded_idx_to_permuted_idx.reshape(-1).contiguous()
        finalize = FinalizeIn(
            fc2=routed.gemm2_permuted,
            perm=perm,
            scales=routed.expert_weights.reshape(-1).contiguous(),
            topk=perm.numel() // rows,
        )
        shared = pending.shared_output
        slot_mappings = get_forward_context().slot_mapping
        assert isinstance(slot_mappings, dict)
        index_cache = attn.indexer.index_cache
        w = lf.weights
        try:
            # The stock input boundary after a deferred finalize rounds the
            # residual once (finalize + all-reduce + norm): single_round.
            lf.stages[bucket(rows)](
                rows,
                residual=residual.contiguous(),
                norm_w=w["input_norm"],
                out_residual=lf.r_mid[:rows],
                w=w["qkv"],
                wsf=w["qkv_sf"],
                q_w=w["q_norm"],
                k_w=w["k_norm"],
                iq_w=w["iq_norm"],
                ik_w=w["ik_norm"],
                cos_sin=w["cos_sin"],
                positions=positions,
                slot_mapping=slot_mappings[attn.layer_name][:rows],
                index_slot_mapping=slot_mappings[index_cache.prefix][:rows],
                kv_cache=attn.kv_cache,
                index_cache=index_cache.kv_cache,
                q_fp8=lf.q_fp8[:rows],
                index_q=lf.index_q[:rows],
                x2=None if shared is None else shared.contiguous(),
                finalize=finalize,
                single_round=True,
            )
        except ValueError as err:  # the argument checks, before any launch
            logger.warning_once(
                "MiniMax-M3 fused attention front not used on %s: %s; the layer "
                "keeps the stock input norm, qkv_proj and KV insert.",
                attn.layer_name,
                err,
            )
            return None
        return lf.q_fp8[:rows], lf.index_q[:rows], lf.r_mid[:rows]


def _no_effect(why: str) -> None:
    logger.warning_once(
        "%s=1 has no effect: %s; the sparse layers keep the stock input norm, "
        "qkv_proj and KV insert.",
        ENV,
        why,
    )


def _candidates(executor: "SpecializedDecode") -> list:
    """The local layers the front can serve: a sparse layer with the low-batch
    attention, after a MoE layer that defers its finalize (host state, the
    same on every rank)."""
    layers = executor.layers
    out = []
    for prev, layer in zip(layers, layers[1:]):
        attn = layer.self_attn
        moe = getattr(prev, "block_sparse_moe", None)
        if (
            getattr(attn, "low_batch_attention", None) is not None
            and getattr(prev, "is_moe_layer", False)
            and getattr(moe, "use_low_batch_ffn", False)
            and attn.kv_cache_dtype == "nvfp4"
            and getattr(attn, "indexer_kv_dtype", None) == "nvfp4"
            and attn._k_scale_float == 1.0
            and attn._v_scale_float == 1.0
        ):
            out.append(layer)
    return out


def _build_layer(layer, transport, buckets: tuple[int, ...]) -> LayerFront:
    from ..kernels._mxfp8 import mxfp8_linear_nk
    from ..kernels.front import QkvFrontStage

    attn = layer.self_attn
    heads = attn.num_heads + 2 * attn.num_kv_heads + attn.num_idx_heads + 1
    qkv, sf = mxfp8_linear_nk(
        attn.qkv_proj, f"{attn.layer_name}: qkv_proj", shape=(heads * HEAD_DIM, HIDDEN)
    )
    rope = attn.rotary_emb
    if rope.cos_sin_cache.dtype != torch.bfloat16:
        # The stock insert reads the same BF16 table (it requires the qkv dtype).
        raise ValueError(f"cos_sin_cache {rope.cos_sin_cache.dtype}, expected BF16")
    weights = dict(
        qkv=qkv,
        qkv_sf=sf,
        input_norm=_bf16(layer.input_layernorm.weight),
        q_norm=_bf16(attn.q_norm.weight),
        k_norm=_bf16(attn.k_norm.weight),
        iq_norm=_bf16(attn.index_q_norm.weight),
        ik_norm=_bf16(attn.index_k_norm.weight),
        cos_sin=rope.cos_sin_cache,
    )
    kwargs = dict(
        transport=transport,
        nq=attn.num_heads,
        nkv=attn.num_kv_heads,
        niq=attn.num_idx_heads,
        rot=rope.rotary_dim,
        block_size=PAGE,
        kv_cache_dtype="nvfp4",
        indexer_kv_dtype="nvfp4",
        prepushed=False,
        eps=layer.input_layernorm.variance_epsilon,
        # The stock insert norms all four head branches with q_norm's epsilon.
        head_eps=attn.q_norm.variance_epsilon,
        q_scale=float(attn._q_scale_float),
        with_step_meta=False,
    )
    stages = {b: QkvFrontStage(max_rows=b, **kwargs) for b in buckets}
    for stage in stages.values():
        # Compiled outside capture: the first launch is inside a FULL graph capture.
        stage.precompile()
    dev = qkv.device
    rows = max(buckets)
    return LayerFront(
        stages=stages,
        weights=weights,
        r_mid=torch.zeros(rows, HIDDEN, dtype=torch.bfloat16, device=dev),
        q_fp8=torch.zeros(
            rows, attn.num_heads * HEAD_DIM, dtype=torch.float8_e4m3fn, device=dev
        ),
        index_q=torch.zeros(
            rows, attn.num_idx_heads * HEAD_DIM, dtype=torch.float8_e4m3fn, device=dev
        ),
    )


def build_fused_front(executor: "SpecializedDecode") -> FusedFront | None:
    """The executor's fused fronts, or None (logged). Collective over the TP
    group: every rank calls it at the same point, after weight loading and
    before memory profiling and CUDA graph capture."""
    from vllm.distributed import get_tp_group

    from ..kernels import _arch, _dtypes, front
    from ..kernels._comm import make_transport
    from ..kernels._exchange import exchange_words, push_workspace

    group = get_tp_group()
    cpu_group = group.cpu_group
    world, rank = group.world_size, group.rank_in_group

    def all_ranks(flags: list[bool]) -> list[bool]:
        t = torch.tensor([int(f) for f in flags], dtype=torch.int32)
        if world > 1:
            dist.all_reduce(t, op=dist.ReduceOp.MIN, group=cpu_group)
        return [bool(v) for v in t.tolist()]

    layers = _candidates(executor)
    # Device code, target and residency (local; agreed below).
    try:
        _arch.device_spec()
        front._ext(_dtypes.CacheFormat.NVFP4, world)
        ready = True
    except Exception as err:
        logger.warning("MiniMax-M3 fused attention front unavailable: %s", err)
        ready = False
    if not all_ranks([ready])[0]:
        _no_effect("the kernel cannot run on some rank (CuTe DSL or target)")
        return None
    if not layers:
        _no_effect(
            "no sparse layer has the low-batch attention after a MoE layer that "
            "defers its finalize (NVFP4 KV and index-K, unit K/V scales)"
        )
        return None
    attn = layers[0].self_attn
    heads = attn.num_heads + 2 * attn.num_kv_heads + attn.num_idx_heads + 1
    # A boundary row per head cluster: a bucket needs as many clusters as rows.
    buckets = tuple(b for b in BUCKETS if b <= heads)
    # 1. Peer memory (collective; a failure raises on every rank together).
    capacity = sum(4 * exchange_words(world, b) + _ARENA_ALIGN for b in buckets)
    try:
        transport = make_transport(
            "auto",
            group=group.device_group,
            cpu_group=cpu_group,
            rank=rank,
            world=world,
            capacity=capacity,
        )
        for b in buckets:
            push_workspace(transport, b, "front")
        peer_ok = True
    except Exception:
        logger.exception("MiniMax-M3 fused attention front: peer memory setup failed")
        transport, peer_ok = None, False
    if not all_ranks([peer_ok])[0]:
        _no_effect("peer memory setup failed on some rank")
        return None
    # 2. The layers (local), then every rank keeps the layers that built everywhere.
    built: dict[int, LayerFront] = {}
    oks = []
    for layer in layers:
        try:
            built[id(layer)] = _build_layer(layer, transport, buckets)
            oks.append(True)
        except Exception:
            logger.exception(
                "MiniMax-M3 fused attention front did not build for %s",
                layer.self_attn.layer_name,
            )
            oks.append(False)
    agreed = all_ranks(oks)
    for layer, ok in zip(layers, agreed):
        if not ok:
            built.pop(id(layer), None)
    if not built:
        _no_effect("the front did not build on some rank")
        return None
    logger.info_once(
        "MiniMax-M3 specialized decode: the fused attention front (input "
        "all-reduce + norm, QKV, head norms, RoPE, NVFP4 KV and index-K stores in "
        "one launch) is built for %d of %d sparse layers, steps of up to %d rows, "
        "over a %s peer transport of %d ranks (%s=1).",
        len(built),
        len(executor.sparse),
        max(buckets),
        transport.kind,
        world,
        ENV,
        scope="process",
    )
    return FusedFront(built, max(buckets), transport)
