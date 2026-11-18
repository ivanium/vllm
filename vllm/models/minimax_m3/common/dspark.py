# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-M3.1 DSpark draft model.

The Qwen3 DSpark block drafter (context-KV precompute, non-causal query block,
Markov and confidence heads) built from MiniMax-M3 layers: Gemma RMSNorm
(`x * (1 + w)`) everywhere, partial RoPE, and a dense SwiGLU-OAI MLP. The
checkpoint ships no token embedding or LM head; both are shared with the target.
"""

import os
from collections.abc import Iterable

import torch
from torch import nn
from transformers import MiniMaxM3VLTextConfig

from vllm import _custom_ops as ops
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_tp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMulWithClamp
from vllm.model_executor.layers.attention.attention import get_attention_context
from vllm.model_executor.layers.fusion.quant_activation import (
    QuantizedActivation,
    get_input_quant_key,
)
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp8Dynamic
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
)
from vllm.model_executor.models.qwen3_dflash import (
    DFlashQwen3Attention,
    DFlashQwen3DecoderLayer,
    _resolve_layer_attention,
)
from vllm.model_executor.models.qwen3_dspark import (
    Qwen3DSparkForCausalLM,
    Qwen3DSparkModel,
)
from vllm.model_executor.models.utils import WeightsMapper
from vllm.models.minimax_m3.common.ops.dspark_context_kv import (
    DSparkContextKVWriter,
    rotary_dims_supported,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
)

logger = init_logger(__name__)

_CHECKPOINT_PREFIX = "language_model.model.dspark."


def _shared_slot_mapping(
    slot_mapping: torch.Tensor | list[torch.Tensor | None] | None,
) -> torch.Tensor | None:
    """The context slot mapping when every draft layer shares it, else None
    (dummy runs, and drafts whose layers span several KV cache groups)."""
    if slot_mapping is None or isinstance(slot_mapping, torch.Tensor):
        return slot_mapping
    first = slot_mapping[0]
    if first is None or any(
        s is None or s.data_ptr() != first.data_ptr() for s in slot_mapping
    ):
        return None
    return first


def _activation_dtype(x: torch.Tensor | QuantizedActivation) -> torch.dtype:
    """The dtype an activation stands for (a pre-quantized one's original)."""
    return x.orig_dtype if isinstance(x, QuantizedActivation) else x.dtype


def _gemma_rms_norm(hidden_size: int, eps: float) -> nn.Module:
    """The target's Gemma RMSNorm (fused FlashInfer kernels on CUDA).

    vLLM's GemmaRMSNorm hands the IR op an fp32 `1 + w` weight, which the
    vllm_c kernel rejects for bf16 inputs, so every call ran the unfused native
    path (about ten kernels).
    """
    if current_platform.is_rocm():
        from vllm.models.minimax_m3.amd.model import MiniMAXGemmaRMSNorm
    else:
        from vllm.models.minimax_m3.nvidia.model import MiniMAXGemmaRMSNorm
    return MiniMAXGemmaRMSNorm(hidden_size, eps=eps)


def dspark_qknorm_rope_kv_insert(
    qkv: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    rotary_dim: int,
    eps: float,
    q_fp8_out: torch.Tensor | None,
    layer_name: LayerNameType,
) -> None:
    """The fused QK-norm/RoPE on `qkv`, also storing the draft tokens' k/v.

    A custom op so the slot mapping is read when it runs: read in the compiled
    forward, it is fixed at trace time, when the draft layers have none.
    """
    _, layer, kv_cache, slot_mapping = get_attention_context(
        _resolve_layer_name(layer_name)
    )
    insert = slot_mapping is not None
    ops.fused_minimax_m3_qknorm_rope_kv_insert(
        qkv,
        q_norm_weight,
        k_norm_weight,
        cos_sin_cache,
        positions,
        num_heads,
        num_kv_heads,
        rotary_dim,
        eps,
        slot_mapping=slot_mapping,
        kv_cache=kv_cache if insert else None,
        block_size=kv_cache.size(2) if insert else 0,
        kv_cache_dtype="fp8" if insert and kv_cache.dtype == torch.uint8 else "auto",
        q_fp8_out=q_fp8_out,
        q_fp8_scale=layer._q_scale_float,
    )


def dspark_qknorm_rope_kv_insert_fake(
    qkv: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    rotary_dim: int,
    eps: float,
    q_fp8_out: torch.Tensor | None,
    layer_name: LayerNameType,
) -> None:
    return


direct_register_custom_op(
    op_name="minimax_m3_dspark_qknorm_rope_kv_insert",
    op_func=dspark_qknorm_rope_kv_insert,
    mutates_args=["qkv", "q_fp8_out"],
    fake_impl=dspark_qknorm_rope_kv_insert_fake,
)


class MiniMaxM3DSparkAttention(DFlashQwen3Attention):
    def __init__(
        self,
        config: MiniMaxM3VLTextConfig,
        sliding_window: int | None,
        causal: bool,
        cache_config: CacheConfig | None,
        quant_config: QuantizationConfig | None,
        prefix: str,
        query_pre_quantized: bool = False,
    ) -> None:
        super().__init__(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            rope_parameters={
                "rope_theta": config.rope_parameters["rope_theta"],
                "partial_rotary_factor": config.rope_parameters.get(
                    "partial_rotary_factor", 0.5
                ),
            },
            max_position=config.max_position_embeddings,
            head_dim=config.head_dim,
            rms_norm_eps=config.rms_norm_eps,
            sliding_window=sliding_window,
            causal=causal,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix,
            query_pre_quantized=query_pre_quantized,
        )
        self.q_norm = _gemma_rms_norm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = _gemma_rms_norm(self.head_dim, eps=config.rms_norm_eps)
        # With an FP8 KV cache the backend wants an FP8 query, and the fused
        # kernel below can write one. Producing it there instead of letting the
        # attention layer quantize afterwards saves a pass over q.
        self._emit_fp8_query = query_pre_quantized
        # FLASHINFER_TS writes an FP8 query's output as FP16 (FA4 only bf16);
        # an MXFP8 o_proj quantizes FP16 as readily and writes the model dtype.
        self._attn_output_dtype = (
            torch.float16
            if query_pre_quantized
            and self.attn.attn_backend.get_name() == "FLASHINFER_TS"
            and get_input_quant_key(self.o_proj) == kMxfp8Dynamic
            else None
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor | QuantizedActivation,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        # The target's fused per-head Gemma QK-norm + partial NeoX RoPE on q/k,
        # in place: one launch instead of the q-norm, k-norm and RoPE kernels.
        # q_fp8_out additionally stores the normed/roped q as E4M3, which is
        # what an FP8-KV backend consumes.
        q_fp8 = (
            torch.empty(
                (qkv.shape[0], self.q_size),
                dtype=current_platform.fp8_dtype(),
                device=qkv.device,
            )
            if self._emit_fp8_query
            else None
        )
        # Also stores the draft tokens' k/v, so the attention layer gets q alone.
        torch.ops.vllm.minimax_m3_dspark_qknorm_rope_kv_insert(
            qkv,
            self.q_norm.weight,
            self.k_norm.weight,
            self.rotary_emb.cos_sin_cache,
            positions,
            self.num_heads,
            self.num_kv_heads,
            self.rotary_emb.rotary_dim,
            self.q_norm.variance_epsilon,
            q_fp8,
            _encode_layer_name(self.attn.layer_name),
        )
        q = qkv[:, : self.q_size] if q_fp8 is None else q_fp8
        # output_dtype must be stated: it otherwise follows the query, which
        # may be FP8. The input may be the input norm's MXFP8 output.
        attn_output = self.attn(
            q,
            None,
            None,
            output_dtype=self._attn_output_dtype or _activation_dtype(hidden_states),
        )
        output, _ = self.o_proj(attn_output)
        return output


class MiniMaxM3DSparkMLP(nn.Module):
    """Dense SwiGLU-OAI MLP: gate * sigmoid(alpha * gate) * (up + beta)."""

    def __init__(
        self,
        config: MiniMaxM3VLTextConfig,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        intermediate_size = config.dspark_ffn_hidden_size
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = SiluAndMulWithClamp(
            swiglu_limit=config.swiglu_limit,
            alpha=config.swiglu_alpha,
            beta=1.0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x, _ = self.down_proj(self.act_fn(gate_up))
        return x


class MiniMaxM3DSparkDecoderLayer(DFlashQwen3DecoderLayer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        config: MiniMaxM3VLTextConfig,
        layer_idx: int,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        if not getattr(config, "dspark_use_dense_ffn", False):
            raise NotImplementedError("MiniMax DSpark supports dense FFN drafts only.")
        self.hidden_size = config.hidden_size
        sliding_window, causal = _resolve_layer_attention(config, layer_idx)
        # An FP8 draft cache means the backend reads an FP8 query, which the
        # fused QK-norm/RoPE kernel emits directly.
        kv_cache_dtype = getattr(cache_config, "cache_dtype", "auto")
        self.self_attn = MiniMaxM3DSparkAttention(
            config,
            sliding_window=sliding_window,
            causal=causal,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
            query_pre_quantized=str(kv_cache_dtype).startswith("fp8"),
        )
        self.mlp = MiniMaxM3DSparkMLP(config, quant_config, prefix=f"{prefix}.mlp")
        hidden_size, eps = config.hidden_size, config.rms_norm_eps
        self.input_layernorm = _gemma_rms_norm(hidden_size, eps=eps)
        self.post_attention_layernorm = _gemma_rms_norm(hidden_size, eps=eps)


class MiniMaxM3DSparkModel(Qwen3DSparkModel):
    decoder_layer_cls = MiniMaxM3DSparkDecoderLayer
    quant_config_mapper = WeightsMapper(
        orig_to_new_prefix={_CHECKPOINT_PREFIX: "model."}
    )

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        start_layer_id: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__(
            vllm_config=vllm_config, start_layer_id=start_layer_id, prefix=prefix
        )
        hidden_size, eps = self.config.hidden_size, self.config.rms_norm_eps
        self.hidden_norm = _gemma_rms_norm(hidden_size, eps=eps)
        self.norm = _gemma_rms_norm(hidden_size, eps=eps)
        # VLLM_ROCM_MINIMAX_M3_MONO_DRAFTER=1: supported draft steps run the draft
        # layers as FlyDSL launches (MI355X)
        self._mono_drafter = None
        if current_platform.is_rocm():
            from vllm.models.minimax_m3.amd.mono.envs import envs as mono_envs

            if mono_envs.MONO_DRAFTER:
                from vllm.models.minimax_m3.amd.mono.drafter.dispatch import (
                    DrafterDecode,
                )

                self._mono_drafter = DrafterDecode(self, vllm_config)

    def __call__(self, *args, **kwargs):
        # ahead of torch.compile's wrapper: the kernels launch outside the graph
        if self._mono_drafter is not None:
            out = self._mono_drafter.maybe_forward(*args, **kwargs)
            if out is not None:
                return out
        return super().__call__(*args, **kwargs)

    def _normalize_context_states(self, context_states: torch.Tensor) -> torch.Tensor:
        return self.hidden_norm(context_states)

    def _normalize_context_k(self, all_k: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [layer.self_attn.k_norm(k) for layer, k in zip(self.layers, all_k)]
        )

    def context_kv_modules(self) -> list[nn.Module]:
        k_norms = [layer.self_attn.k_norm for layer in self.layers]
        return [*super().context_kv_modules(), *k_norms]

    def _context_kv_writer(self) -> DSparkContextKVWriter | None:
        """The fused context K/V writer, or None where the base path runs."""
        if not hasattr(self, "_ctx_kv_writer"):
            if any(a.kv_cache.numel() == 0 for a in self._attn_layers):
                return None  # KV caches not bound yet: decide once they are
            rotary_dim = self._rope_cos_sin_cache.shape[-1]
            supported = (
                self._rope_is_neox
                and self.layers[0].self_attn.v_scale is None
                and rotary_dims_supported(rotary_dim, self._head_dim)
                and self._rope_cos_sin_cache.dtype == torch.bfloat16
                and DSparkContextKVWriter.supported(self._attn_layers)
            )
            self._ctx_kv_writer = (
                DSparkContextKVWriter(self._attn_layers) if supported else None
            )
        return self._ctx_kv_writer

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | list[torch.Tensor | None] | None = None,
    ) -> None:
        """One kernel norms, ropes and stores the context K/V of every layer,
        straight from the projection output."""
        slot_mapping = _shared_slot_mapping(context_slot_mapping)
        writer = (
            self._context_kv_writer()
            if slot_mapping is not None
            and hasattr(self, "_num_attn_layers")
            and context_states.dtype == torch.bfloat16
            else None
        )
        if writer is None:
            super().precompute_and_store_context_kv(
                context_states, context_positions, context_slot_mapping
            )
            return
        all_kv = self._project_context_kv_token_major(
            context_states,
            context_states.shape[0],
            self._num_attn_layers,
            self._num_kv_heads,
            self._head_dim,
        )
        writer(
            all_kv,
            self._k_norm_weights,
            self._rms_norm_eps,
            context_positions,
            self._rope_cos_sin_cache,
            self._rope_cos_sin_cache.shape[-1],
            slot_mapping,
        )


class _ShardedContextFc:
    """`combine_hidden_states` under VLLM_MINIMAX_M3_DSPARK_SHARDED_CONTEXT_FC=1:
    the context projection `fc` ([hidden, 5 x hidden], unquantized and
    replicated, so every rank streams all of it for the few rows of a decode
    step) split over the TP ranks. Each rank multiplies its 1/TP slice of the
    input features (strided views, no copies) into FP32; the FP32 partials are
    all-reduced and rounded to the model dtype once, as a split-K GEMM would.
    Larger (prefill) batches keep the replicated GEMM. As vLLM #66324's
    `ShardedContextFc` (MiniMax-M3 specialized decode)."""

    # Rows up to which the split runs: the decode steps' (8 requests x 8 tokens).
    MAX_ROWS = 64

    def __init__(self, model: nn.Module, stock, tp_size: int, tp_rank: int) -> None:
        self._model = model
        self._stock = stock
        self._tp_size = tp_size
        self._tp_rank = tp_rank
        self._logged = False

    @staticmethod
    def unsupported(model: nn.Module, tp_size: int) -> str | None:
        """Why the split cannot run (None when it can); the same on every rank."""
        inner = model.model
        fc = getattr(inner, "fc", None)
        if tp_size == 1:
            return "a single TP rank"
        if not getattr(inner, "use_aux_hidden_state", False) or fc is None:
            return "no context fc"
        if not isinstance(fc.quant_method, UnquantizedLinearMethod):
            return f"a context fc on {type(fc.quant_method).__name__}"
        if fc.weight.dtype not in (torch.bfloat16, torch.float16):
            return f"a {fc.weight.dtype} context fc"
        if getattr(fc, "bias", None) is not None:
            return "a context fc bias"
        if fc.input_size % tp_size:
            return f"{fc.input_size} input features over {tp_size} ranks"
        return None

    def __call__(self, hidden_states: torch.Tensor) -> torch.Tensor:
        from vllm.distributed import tensor_model_parallel_all_reduce

        weight = self._model.model.fc.weight
        if (
            hidden_states.dim() != 2
            or not 0 < hidden_states.shape[0] <= self.MAX_ROWS
            or hidden_states.shape[-1] != weight.shape[1]
        ):
            return self._stock(hidden_states)
        width = weight.shape[1] // self._tp_size
        cols = slice(self._tp_rank * width, (self._tp_rank + 1) * width)
        x = hidden_states.to(weight.dtype)[:, cols]
        partial = torch.mm(x, weight[:, cols].t(), out_dtype=torch.float32)
        out = tensor_model_parallel_all_reduce(partial).to(weight.dtype)
        if not self._logged:
            self._logged = True
            logger.info(
                "MiniMax-M3 DSpark: context fc of %d rows split over %d TP ranks.",
                hidden_states.shape[0],
                self._tp_size,
            )
        return out


class _DraftNormMxfp8:
    """`forward` of a draft Gemma RMSNorm under VLLM_MINIMAX_M3_DSPARK_NORM_MXFP8=1:
    when the linear it feeds takes FlashInfer's MXFP8 input (the 128x4 scale
    layout), the norm's kernel writes that input itself (`mxfp8=True`, as the
    target's input norms do), so the linear skips its own quantize launch. The
    residual update and the scales are those of the BF16 path; a code can be
    one E4M3 step off where the kernel's FP32 sum of squares, summed in another
    order in its MXFP8 variant, moves the BF16 norm across a rounding boundary
    (`ops/norm_mxfp8.py`). Otherwise the norm keeps its BF16 output. Checked
    per call, as the target checks its qkv_proj."""

    def __init__(
        self, name: str, norm: nn.Module, consumer, takes_mxfp8, status: dict
    ) -> None:
        self._name = name
        self._norm = norm
        self._consumer = consumer  # () -> the linear, or None while it cannot fuse
        self._takes_mxfp8 = takes_mxfp8  # nvidia.model._consumes_mxfp8_128x4
        self._status = status

    def __call__(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
        mxfp8: bool = False,
    ):
        norm = self._norm
        if not mxfp8 and x.dim() == 2:
            consumer = self._consumer()
            mxfp8 = (
                consumer is not None
                and x.dtype == getattr(consumer, "params_dtype", None)
                and self._takes_mxfp8(consumer)
            )
            self._record(mxfp8)
        return type(norm).forward(norm, x, residual, mxfp8=mxfp8)

    def _record(self, fused: bool) -> None:
        """Log, once per process, which norms fused at their first call."""
        status = self._status
        if self._name in status["first"]:
            return
        status["first"][self._name] = fused
        if len(status["first"]) == status["total"]:
            kept = [name for name, f in status["first"].items() if not f]
            logger.info(
                "MiniMax-M3 DSpark: %d of %d draft norms write their linear's "
                "MXFP8 input; BF16 kept for: %s.",
                status["total"] - len(kept),
                status["total"],
                ", ".join(kept) or "none",
            )


def _speculator_loop(speculator, num_reqs: int, head_hidden: torch.Tensor) -> bool:
    """`sample_sequential` fallback: False has the speculator run its loop."""
    return False


class _ShardedMarkovHook:
    """`sample_sequential` of the draft model under
    VLLM_MINIMAX_M3_DSPARK_SHARDED_MARKOV=1: the sharded chain, logging once
    per process the first draft step it takes."""

    def __init__(self, sampling) -> None:
        self._sampling = sampling
        self._logged = False

    def __call__(self, speculator, num_reqs: int, head_hidden: torch.Tensor) -> bool:
        taken = self._sampling(speculator, num_reqs, head_hidden)
        if taken and not self._logged:
            self._logged = True
            logger.info(
                "MiniMax-M3 DSpark: draft step of %d request(s) sampled by the "
                "sharded Markov chain.",
                num_reqs,
            )
        return taken


class MiniMaxM3DSparkForCausalLM(Qwen3DSparkForCausalLM):
    model_cls = MiniMaxM3DSparkModel

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self._attach_draft_step_kernels(vllm_config)
        # VLLM_MINIMAX_M3_DSPARK_SHARDED_MARKOV=1: sample the draft Markov chain
        # with the vocabulary-sharded cooperative kernel (one launch per draft
        # step) instead of the speculator's per-position loop, where the loop
        # would draw plain Gumbel noise into a full draft-logits cache.
        self._sharded_markov_off_reason: str | None = None
        self._sharded_markov_requested = (
            os.environ.get("VLLM_MINIMAX_M3_DSPARK_SHARDED_MARKOV", "0") == "1"
        )
        if not self._sharded_markov_requested:
            return
        spec_config = vllm_config.speculative_config
        reasons = []
        if not current_platform.is_cuda():
            reasons.append("a non-CUDA platform")
        if spec_config.draft_sample_method != "probabilistic":
            reasons.append(f"draft_sample_method={spec_config.draft_sample_method}")
        if spec_config.enable_adaptive_verification:
            reasons.append("adaptive verification")
        if vllm_config.model_config.use_fp64_gumbel:
            reasons.append("FP64 Gumbel sampling")
        if getattr(vllm_config, "watermark_config", None) is not None:
            reasons.append("watermarking")
        if vllm_config.parallel_config.enable_batch_sharded_sampling:
            reasons.append("batch-sharded sampling")
        if getattr(self.config, "dspark_draft_topk", None) is not None:
            reasons.append("top-k Markov sampling")
        if self.draft_id_to_target_id is not None:
            reasons.append("a reduced draft vocabulary")
        self._sharded_markov_off_reason = "; ".join(reasons) or None
        # The chain's settings: the draft logits cache is the speculator's
        # DraftLogitsCache in the head dtype (Speculator.draft_logits_spec).
        self.markov_cache_dtype = vllm_config.model_config.head_dtype
        self.max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        self.num_speculative_steps = spec_config.num_speculative_tokens

    def _attach_draft_step_kernels(self, vllm_config: VllmConfig) -> None:
        """The draft step's opt-in kernels, each behind its own variable:
        VLLM_MINIMAX_M3_DSPARK_SHARDED_CONTEXT_FC=1 splits the context fc over
        the TP ranks, VLLM_MINIMAX_M3_DSPARK_NORM_MXFP8=1 has the draft norms
        write the MXFP8 input of the linear they feed."""
        if os.environ.get("VLLM_MINIMAX_M3_DSPARK_SHARDED_CONTEXT_FC", "0") == "1":
            tp_group = get_tp_group()
            draft_tp = vllm_config.speculative_config.draft_tensor_parallel_size
            if not current_platform.is_cuda():
                reason = "a non-CUDA platform"
            elif draft_tp not in (None, tp_group.world_size):
                reason = f"a TP{draft_tp} draft in a TP{tp_group.world_size} group"
            else:
                reason = _ShardedContextFc.unsupported(self, tp_group.world_size)
            if reason is not None:
                logger.info_once(
                    "VLLM_MINIMAX_M3_DSPARK_SHARDED_CONTEXT_FC=1 has no effect: the "
                    "DSpark context fc stays replicated (%s).",
                    reason,
                    scope="process",
                )
            else:
                self.combine_hidden_states = _ShardedContextFc(
                    self,
                    self.combine_hidden_states,
                    tp_group.world_size,
                    tp_group.rank_in_group,
                )
                logger.info_once(
                    "MiniMax-M3 DSpark: the context fc of draft steps of up to %d "
                    "rows is split over the %d TP ranks (FP32 partials, one "
                    "all-reduce; VLLM_MINIMAX_M3_DSPARK_SHARDED_CONTEXT_FC=1).",
                    _ShardedContextFc.MAX_ROWS,
                    tp_group.world_size,
                    scope="process",
                )
        if os.environ.get("VLLM_MINIMAX_M3_DSPARK_NORM_MXFP8", "0") == "1":
            if not current_platform.is_cuda():
                logger.info_once(
                    "VLLM_MINIMAX_M3_DSPARK_NORM_MXFP8=1 has no effect on a non-CUDA "
                    "platform.",
                    scope="process",
                )
                return
            from vllm.models.minimax_m3.nvidia.model import (
                MiniMAXGemmaRMSNorm,
                _consumes_mxfp8_128x4,
            )

            inner = self.model
            # (name, norm, () -> the linear that is the norm output's only reader)
            pairs = []
            for i, layer in enumerate(inner.layers):
                attn = layer.self_attn
                pairs.append(
                    (
                        f"layers.{i}.input_layernorm",
                        layer.input_layernorm,
                        lambda attn=attn: attn.qkv_proj,
                    )
                )
                pairs.append(
                    (
                        f"layers.{i}.post_attention_layernorm",
                        layer.post_attention_layernorm,
                        lambda mlp=layer.mlp: mlp.gate_up_proj,
                    )
                )
            # The context states' norm feeds the context K/V projection unless the
            # BF16 fused K/V weight is there (_project_context_kv_token_major).
            pairs.append(
                (
                    "hidden_norm",
                    inner.hidden_norm,
                    lambda: (
                        inner._context_kv_proj
                        if inner._fused_kv_weight is None
                        else None
                    ),
                )
            )
            reason = next(
                (
                    f"a {type(norm).__name__} {name}"
                    for name, norm, _ in pairs
                    if type(norm) is not MiniMAXGemmaRMSNorm
                ),
                None,
            )
            if reason is not None:
                logger.info_once(
                    "VLLM_MINIMAX_M3_DSPARK_NORM_MXFP8=1 has no effect: the DSpark "
                    "draft norms keep their BF16 output (%s).",
                    reason,
                    scope="process",
                )
            else:
                status: dict = {"total": len(pairs), "first": {}}
                for name, norm, consumer in pairs:
                    norm.forward = _DraftNormMxfp8(
                        name, norm, consumer, _consumes_mxfp8_128x4, status
                    )
                logger.info_once(
                    "MiniMax-M3 DSpark: %d draft norms write the MXFP8 input of the "
                    "linear they feed where it takes it "
                    "(VLLM_MINIMAX_M3_DSPARK_NORM_MXFP8=1).",
                    len(pairs),
                    scope="process",
                )

    def draft_logits_shard(self) -> tuple[int, int] | None:
        """`(first column, width)` of `compute_draft_logits_local` within
        the full draft logits, or None when it is not a plain slice of them
        (vocabulary padding or added vocabulary in the shard)."""
        head = self.lm_head
        shard = head.shard_indices
        width = shard.num_org_elements
        if (
            self.logits_processor.logits_as_input
            or self.logits_processor.org_vocab_size != self.config.draft_vocab_size
            or shard.num_elements_padded != width
            or head.num_embeddings_per_partition != width
            or shard.org_vocab_start_index != head.tp_rank * width
        ):
            return None
        return shard.org_vocab_start_index, width

    def _markov_bias_unsupported(self) -> str | None:
        """Why `markov_bias` is not a plain BF16 product with the Markov
        head's `[vocab, rank]` weight, or None when it is."""
        head = self.model.markov_head.markov_w2
        processor = self.logits_processor
        # The draft's ModelOpt config gives an excluded head an unquantized
        # linear method; both run the same BF16 GEMM (LogitsProcessor._apply_head).
        if not isinstance(
            head.quant_method, (UnquantizedEmbeddingMethod, UnquantizedLinearMethod)
        ):
            return f"a Markov head on {type(head.quant_method).__name__}"
        if head.weight.dtype != torch.bfloat16:
            return f"a {head.weight.dtype} Markov head"
        if processor.scale != 1.0 or processor.soft_cap is not None:
            return "scaled or soft-capped draft logits"
        if processor.logits_as_input:
            return "logits as input"
        if processor.head_dtype not in (None, torch.bfloat16):
            return f"a {processor.head_dtype} head dtype"
        return None

    def markov_bias_weight(self) -> torch.Tensor | None:
        """The `[vocab, rank]` matrix `markov_bias` multiplies by, when
        `markov_bias` is a plain BF16 product with it (else None)."""
        if self._markov_bias_unsupported() is not None:
            return None
        return self.model.markov_head.markov_w2.weight

    def build_default_markov_sampling(self) -> None:
        """Install the sharded Markov chain as `sample_sequential` when
        VLLM_MINIMAX_M3_DSPARK_SHARDED_MARKOV=1. Called by the speculator after
        the draft weights load, before KV-cache allocation and CUDA graph
        capture, on every TP rank: the set-up is collective."""
        if not self._sharded_markov_requested:
            return
        reason = self._sharded_markov_off_reason or self._markov_bias_unsupported()
        if reason is not None:
            logger.info_once(
                "VLLM_MINIMAX_M3_DSPARK_SHARDED_MARKOV=1 has no effect: the "
                "DSpark Markov chain stays on the speculator's loop (%s).",
                reason,
                scope="process",
            )
            return
        from vllm.models.deepseek_v41.nvidia.specialized_decode.sampling.sharded_markov import (  # noqa: E501
            ShardedMarkovSampling,
            make_sharded_sampler,
        )

        tp_group = get_tp_group()
        sharded = make_sharded_sampler(
            self.model.markov_head.markov_w1.weight,
            self.markov_bias_weight(),
            self.config.draft_vocab_size,
            self.num_speculative_steps,
            self.markov_cache_dtype,
            tp_group.cpu_group if tp_group.world_size > 1 else None,
            self.max_num_reqs,
        )
        if sharded is None:
            # make_sharded_sampler logged why, on every rank.
            logger.info_once(
                "VLLM_MINIMAX_M3_DSPARK_SHARDED_MARKOV=1 has no effect: the "
                "DSpark Markov chain stays on the speculator's loop.",
                scope="process",
            )
            return
        local_logits = sharded.accepts_shard(self.draft_logits_shard())
        self.sample_sequential = _ShardedMarkovHook(
            ShardedMarkovSampling(sharded, _speculator_loop, local_logits)
        )
        logger.info_once(
            "MiniMax-M3 DSpark: the draft Markov chain of up to %d requests is "
            "sampled by the vocabulary-sharded kernel on %d ranks, one launch per "
            "draft step (%s draft logits); larger batches keep the speculator's "
            "loop (VLLM_MINIMAX_M3_DSPARK_SHARDED_MARKOV=1).",
            sharded.max_rows,
            sharded.world,
            "rank-local" if local_logits else "gathered",
            scope="process",
        )

    @staticmethod
    def _rename_checkpoint_weights(
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> Iterable[tuple[str, torch.Tensor]]:
        for name, tensor in weights:
            name = name.removeprefix(_CHECKPOINT_PREFIX)
            name = name.replace(".decoder_layer.", ".")
            if name.startswith("final_norm."):
                name = "norm." + name.removeprefix("final_norm.")
            # MiniMax stores the MXFP8 E8M0 block scales as weight_scale_inv.
            name = name.replace(".weight_scale_inv", ".weight_scale")
            yield name, tensor

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        return super().load_weights(self._rename_checkpoint_weights(weights))
