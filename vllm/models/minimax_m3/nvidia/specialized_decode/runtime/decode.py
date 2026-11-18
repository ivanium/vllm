# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The specialized decode executor.

``SpecializedDecode`` runs the decoder layers of ``MiniMaxM3Model`` for FULL
CUDA graph captures of pure decode steps of 1-32 rows (``moe.MAX_ROWS``).
``plan`` decides per forward whether it runs; the backbone then hands its
layers to ``DecodeStep.run`` after the embedding and keeps its final norm.
Every other forward (prefill, mixed, eager, PIECEWISE, larger batches) runs
the stock layers, which the executor leaves unchanged.

Per layer the executor runs the stock modules through its own steps:

- the input norm, which finalizes the previous MoE layer in the fused
  ``ops/finalize_allreduce_norm.py`` kernel;
- the attention: on the sparse layers, the stock qkv projection and KV
  insert, then ``attention.LowBatchAttention`` (indexer, sparse attention and
  o_proj) when the step's requests and queries fit it, the stock modules
  otherwise; with ``VLLM_MINIMAX_M3_LOW_BATCH_FUSED_FRONT=1``, steps of up to
  16 rows replace the input norm, qkv projection and KV insert ahead of the
  low-batch attention with one fused front launch (``front.py``);
- the fused all-reduce + post-attention norm;
- the FFN: ``moe.LowBatchFfn`` on the MoE layers that leave their finalize to
  the next layer, the stock modules on the others.

The fused kernels issue TP collectives, so every decision reads host state that
all TP ranks share (the step's rows, request and query counts and the capture
state); nothing is read back from the GPU.
"""

from functools import partial
from typing import TYPE_CHECKING, NamedTuple

import torch

from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_allreduce_gemma_rms_norm import (
    fused_allreduce_gemma_rms_norm,
)
from vllm.model_executor.layers.fused_moe.moe_output import MoEOutput
from vllm.models.minimax_m3.common.sparse_attention import MiniMaxM3SparseMetadata
from vllm.sequence import IntermediateTensors

from .. import INTERMEDIATE_PER_RANK, captures_pure_decode
from ..attention import HEADS, MAX_PAGES, AttentionStep, prepare_step
from ..moe import MAX_ROWS

if TYPE_CHECKING:
    from ...model import MiniMaxM3DecoderLayer, MiniMaxM3Model
    from .front import FusedFront

logger = init_logger(__name__)


def _same_tensor(a: torch.Tensor | None, b: torch.Tensor) -> bool:
    return (
        a is not None
        and a.data_ptr() == b.data_ptr()
        and a.shape == b.shape
        and a.stride() == b.stride()
    )


class DecodeStep(NamedTuple):
    """One captured decode step."""

    executor: "SpecializedDecode"
    # The low-batch attention's metadata, or None to run the stock attention.
    attention: AttentionStep | None

    def run(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        aux_hidden_states: list[torch.Tensor],
    ) -> tuple[torch.Tensor | MoEOutput, torch.Tensor]:
        """The decoder layers on the embedding ``hidden_states``; returns what
        the last layer leaves for the final norm."""
        return self.executor.run(
            hidden_states, positions, aux_hidden_states, self.attention
        )


class SpecializedDecode:
    """The decode executor over one backbone's local layers."""

    def __init__(self, backbone: "MiniMaxM3Model") -> None:
        self.backbone = backbone
        self.layers: list[MiniMaxM3DecoderLayer] = list(
            backbone.layers[backbone.start_layer : backbone.end_layer]
        )
        # The sparse layers build their low-batch attention once loaded.
        self.sparse = [
            layer.self_attn
            for layer in self.layers
            if hasattr(layer.self_attn, "use_low_batch_attention")
        ]
        for attn in self.sparse:
            attn.use_low_batch_attention = True
        self.front: FusedFront | None = None
        self._front_built = False

    def build_fused_front(self) -> None:
        """Build the fused attention fronts when requested (collective over
        the TP group; after weight loading, before profiling and capture)."""
        from . import front

        if self._front_built or not front.requested():
            return
        self._front_built = True
        self.front = front.build_fused_front(self)

    def plan(
        self,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
    ) -> DecodeStep | None:
        """The step this forward runs, or None to run the stock layers."""
        rows = positions.shape[0]
        if (
            intermediate_tensors is not None
            or not 0 < rows <= MAX_ROWS
            or not captures_pure_decode()
        ):
            return None
        step = DecodeStep(self, self._attention_step(rows))
        self._log_capture(rows, step.attention)
        return step

    def _log_capture(self, rows: int, attention: AttentionStep | None) -> None:
        """Which paths the captured steps take: the layers that built the
        low-batch paths (every rank), then once per captured step shape."""
        moe = [
            layer.block_sparse_moe
            for layer in self.layers
            if layer.is_moe_layer and layer.block_sparse_moe.use_low_batch_ffn
        ]
        logger.info_once(
            "MiniMax-M3 specialized decode: low-batch attention on %d of %d sparse "
            "attention layers, low-batch FFN on %d of %d deferring MoE layers.",
            sum(a.low_batch_attention is not None for a in self.sparse),
            len(self.sparse),
            sum(m.low_batch_ffn is not None for m in moe),
            len(moe),
            scope="process",
        )
        # The TP8 shapes (VLLM_MINIMAX_M3_LOW_BATCH_FFN_TP8 and _ATTN_PAD_HEADS).
        heads = {
            a.low_batch_attention.heads
            for a in self.sparse
            if a.low_batch_attention is not None
        }
        columns = {
            m.low_batch_ffn.intermediate for m in moe if m.low_batch_ffn is not None
        }
        if heads - {HEADS} or columns - {INTERMEDIATE_PER_RANK}:
            logger.info_once(
                "MiniMax-M3 specialized decode shapes: low-batch FFN at %s "
                "intermediate columns per rank, low-batch attention at %s query "
                "heads per rank (of the kernels' %d).",
                "/".join(map(str, sorted(columns))) or "no",
                "/".join(map(str, sorted(heads))) or "no",
                HEADS,
                scope="process",
            )
        if attention is None:
            logger.info_once(
                "MiniMax-M3 specialized decode: FULL-graph decode step of %d rows "
                "captured with the stock attention and the low-batch FFN.",
                rows,
            )
        else:
            logger.info_once(
                "MiniMax-M3 specialized decode: FULL-graph decode step of %d rows "
                "(%d requests x %d queries) captured with the low-batch attention "
                "and FFN.",
                rows,
                attention.num_reqs,
                attention.query_len,
            )

    def _attention_step(self, rows: int) -> AttentionStep | None:
        """The step's low-batch attention metadata, when its requests and
        queries fit and every layer with the low-batch path reads the same
        decode block table, of at most ``MAX_PAGES`` pages, for its KV and
        its index cache."""
        built = [a for a in self.sparse if a.low_batch_attention is not None]
        attn_metadata = get_forward_context().attn_metadata
        if not built or not isinstance(attn_metadata, dict):
            return None
        low_batch = built[0].low_batch_attention
        md = attn_metadata.get(built[0].layer_name)
        if not isinstance(md, MiniMaxM3SparseMetadata) or low_batch is None:
            return None
        decode = md.decode
        if decode is None or md.num_prefills or md.num_decode_tokens != rows:
            return None
        table = decode.block_table
        if table.shape[1] > MAX_PAGES or not low_batch.accepts(
            md.num_decodes, decode.decode_query_len
        ):
            return None
        for attn in built:
            main = attn_metadata.get(attn.layer_name)
            index = attn_metadata.get(attn.indexer.index_cache.prefix)
            # Metadata of one KV-cache group shares its slot mapping, so the
            # index cache's pages follow the same block table.
            if (
                not isinstance(main, MiniMaxM3SparseMetadata)
                or main.decode is None
                or not _same_tensor(main.decode.block_table, table)
                or not _same_tensor(
                    getattr(index, "slot_mapping", None), md.slot_mapping
                )
            ):
                return None
        return prepare_step(table, decode.seq_lens, decode.decode_query_len)

    def run(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        aux_hidden_states: list[torch.Tensor],
        attention: AttentionStep | None,
    ) -> tuple[torch.Tensor | MoEOutput, torch.Tensor]:
        aux_layers = self.backbone.aux_hidden_state_layers
        front = self.front if attention is not None else None
        fronts = 0
        residual = None
        for idx, layer in enumerate(self.layers, start=self.backbone.start_layer):
            low_batch = getattr(layer.self_attn, "low_batch_attention", None)
            fused = None
            if (
                front is not None
                and low_batch is not None
                and front.takes(layer, hidden_states, residual, positions)
            ):
                fused = front.run(layer, hidden_states, residual, positions)
            if fused is not None:
                # One launch for the input boundary, qkv_proj and the KV insert.
                q, index_q, residual = fused
                fronts += 1
                if idx in aux_layers:
                    aux_hidden_states.append(residual.clone())
                attn = layer.self_attn
                hidden_states = low_batch.run(
                    attention,
                    q,
                    index_q,
                    attn.kv_cache,
                    attn.indexer.index_cache.kv_cache,
                )
                hidden_states, residual = fused_allreduce_gemma_rms_norm(
                    hidden_states, residual, layer.post_attention_layernorm
                )
                hidden_states = self._ffn(layer, hidden_states)
                continue
            hidden_states, residual = layer.forward_input_norm(hidden_states, residual)
            # Aux hidden state k is the residual stream entering layer k.
            if idx in aux_layers:
                aux_hidden_states.append(residual.clone())
            if attention is not None and low_batch is not None:
                hidden_states = layer.self_attn(
                    positions=positions,
                    hidden_states=hidden_states,
                    low_batch=partial(low_batch.run, attention),
                )
            else:
                hidden_states = layer.self_attn(
                    positions=positions, hidden_states=hidden_states
                )
            hidden_states, residual = fused_allreduce_gemma_rms_norm(
                hidden_states, residual, layer.post_attention_layernorm
            )
            hidden_states = self._ffn(layer, hidden_states)
        if front is not None:
            logger.info_once(
                "MiniMax-M3 specialized decode: FULL-graph decode step of %d rows "
                "captured with the fused attention front on %d of %d sparse layers.",
                positions.shape[0],
                fronts,
                len(self.sparse),
            )
        return hidden_states, residual

    @staticmethod
    def _ffn(
        layer: "MiniMaxM3DecoderLayer", hidden_states: torch.Tensor
    ) -> torch.Tensor | MoEOutput:
        if not layer.is_moe_layer:
            return layer.mlp(hidden_states)
        moe = layer.block_sparse_moe
        if not moe.defers_finalize(hidden_states.shape[0]):
            return moe(hidden_states)
        ffn = moe.low_batch_ffn
        if ffn is not None and ffn.accepts(hidden_states):
            return ffn.run(hidden_states)
        return moe.forward_unfinalized(hidden_states)
