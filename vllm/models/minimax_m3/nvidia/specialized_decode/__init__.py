# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-M3 specialized decode on NVIDIA Blackwell GPUs, enabled with
``additional_config={"decode_backend": "latency_specialized"}``.

The executor (``runtime/decode.py``) runs the decoder layers of FULL CUDA
graph captures of pure decode steps of 1-32 rows (up to four requests with
the seven-token DSpark draft, whose verify steps are 8 rows a request), with
the low-batch sparse attention core after the KV insert (``attention/``) and
the low-batch FFN half of the MoE layers (``moe/``); every other forward
runs the stock layers.

At TP8 (two NVL72 trays) a rank has 384 intermediate columns and 8 query
heads; two opt-in switches, read when the model is built, extend the paths
to those shapes:

- ``VLLM_MINIMAX_M3_LOW_BATCH_FFN_TP8=1``: the low-batch FFN's kernels are
  also built for 384 intermediate columns per rank;
- ``VLLM_MINIMAX_M3_LOW_BATCH_ATTN_PAD_HEADS=1``: the low-batch attention
  takes 8 query heads per rank as the first 8 rows of its 16-head tile, the
  other 8 with zero queries and their outputs left out of o_proj's input
  (o_proj keeps the rank's own weights). Without it a TP8 rank keeps the
  stock attention and says so.
"""

import os

import torch

import vllm.envs as envs
from vllm.config import CompilationMode, CUDAGraphMode, VllmConfig
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

# One TP4 rank of MiniMax-M3.1-Flash: what the low-batch FFN's kernels are
# compiled for; and one TP8 rank's intermediate columns, which they are also
# built for with LOW_BATCH_FFN_TP8.
HIDDEN = 6144
EXPERTS = 128
TOP_K = 4
INTERMEDIATE_PER_RANK = 768
TP8_INTERMEDIATE_PER_RANK = 384
LOW_BATCH_FFN_TP8 = "VLLM_MINIMAX_M3_LOW_BATCH_FFN_TP8"
LOW_BATCH_ATTN_PAD_HEADS = "VLLM_MINIMAX_M3_LOW_BATCH_ATTN_PAD_HEADS"


def ffn_widths() -> tuple[int, ...]:
    """The intermediate columns per rank the low-batch FFN takes."""
    if os.environ.get(LOW_BATCH_FFN_TP8) == "1":
        return (INTERMEDIATE_PER_RANK, TP8_INTERMEDIATE_PER_RANK)
    return (INTERMEDIATE_PER_RANK,)


def attention_pads_heads() -> bool:
    """Whether the low-batch attention takes 8 query heads per rank (TP8)."""
    return os.environ.get(LOW_BATCH_ATTN_PAD_HEADS) == "1"


def unsupported(vllm_config: VllmConfig) -> list[str]:
    """Every requirement of the specialized decode the configuration misses
    (the fused MoE finalize is checked separately)."""
    from vllm.models.minimax_m3.nvidia.model import _use_mega_moe

    parallel = vllm_config.parallel_config
    compilation = vllm_config.compilation_config
    cudagraph_mode = compilation.cudagraph_mode
    config = vllm_config.model_config.hf_text_config
    tp = parallel.tensor_parallel_size
    widths = ffn_widths()
    width_text = f"{INTERMEDIATE_PER_RANK} intermediate columns per rank (TP4)"
    if TP8_INTERMEDIATE_PER_RANK in widths:
        width_text += f" or {TP8_INTERMEDIATE_PER_RANK} (TP8)"
    else:
        width_text += f"; {TP8_INTERMEDIATE_PER_RANK} at TP8 with {LOW_BATCH_FFN_TP8}=1"
    # Model Runner V2 is the only runner, and every MoE layer builds one shared
    # expert of intermediate_size (the low-batch FFN checks its shapes).
    requirements = {
        "an SM100-family GPU (B200, B300)": current_platform.is_cuda()
        and current_platform.is_device_capability_family(100),
        "no pipeline, data, expert or context parallelism or micro-batching": (
            parallel.pipeline_parallel_size == 1
            and parallel.data_parallel_size == 1
            and not parallel.enable_expert_parallel
            and parallel.decode_context_parallel_size == 1
            and parallel.prefill_context_parallel_size == 1
            and not parallel.use_ubatching
        ),
        f"MoE layers of hidden size {HIDDEN}, {EXPERTS} experts, top-{TOP_K} and "
        f"{width_text} for the routed and the shared expert": (
            config.hidden_size == HIDDEN
            and config.num_local_experts == EXPERTS
            and config.num_experts_per_tok == TOP_K
            and config.intermediate_size % tp == 0
            and config.intermediate_size // tp in widths
            and not _use_mega_moe(vllm_config)
        ),
        "FULL CUDA graphs for decode batches only (FULL_AND_PIECEWISE or "
        "FULL_DECODE_ONLY)": (
            cudagraph_mode is not None
            and cudagraph_mode.decode_mode() == CUDAGraphMode.FULL
            and cudagraph_mode.mixed_mode() != CUDAGraphMode.FULL
        ),
        "no torch.compile of the model": (
            compilation.mode != CompilationMode.STOCK_TORCH_COMPILE
        ),
        # The low-batch FFN binds weight addresses.
        "no sleep mode": not vllm_config.model_config.enable_sleep_mode,
        "no LoRA or batch invariance": (
            vllm_config.lora_config is None and not envs.VLLM_BATCH_INVARIANT
        ),
    }
    return [need for need, ok in requirements.items() if not ok]


def captures_pure_decode() -> bool:
    """Whether this forward is being captured into a FULL CUDA graph, which
    under the decode-only FULL graphs required above is a pure decode step.
    Model Runner V2 captures FULL graphs with the forward context's mode NONE
    and PIECEWISE ones, which report capturing too, with PIECEWISE."""
    return (
        torch.cuda.is_current_stream_capturing()
        and is_forward_context_available()
        and get_forward_context().cudagraph_runtime_mode != CUDAGraphMode.PIECEWISE
    )


def specialized_decode_config(vllm_config: VllmConfig) -> bool:
    """Whether the specialized decode is on (``decode_backend``).

    Raises with every missing requirement when requested. The configuration
    checks give the same answer on every TP rank, and the fused MoE finalize the
    low-batch FFN feeds (NVLink multicast) is decided over the TP group inside
    ``supports_fused_moe_finalize``, so every rank raises alike.
    """
    extra = vllm_config.additional_config
    backend = extra.get("decode_backend", "auto") if isinstance(extra, dict) else "auto"
    if backend == "auto":
        return False
    if backend != "latency_specialized":
        raise ValueError("Supported decode_backend values are auto/latency_specialized")
    from vllm.models.minimax_m3.nvidia.ops.finalize_allreduce_norm import (
        supports_fused_moe_finalize,
    )

    missing = unsupported(vllm_config)
    if not supports_fused_moe_finalize(vllm_config):
        missing.append(
            "tensor parallelism with the fused MoE finalize + all-reduce + norm "
            "(NVLink multicast on every rank)"
        )
    if missing:
        raise ValueError(
            f"decode_backend=latency_specialized needs {'; '.join(missing)}."
        )
    logger.info_once(
        "MiniMax-M3 specialized decode on: in FULL-graph decode steps of 1-32 "
        "rows, the sparse attention layers run the low-batch indexer, sparse "
        "attention and o_proj, and MoE layers that leave their finalize to the "
        "next layer run the low-batch FFN."
    )
    tp = vllm_config.parallel_config.tensor_parallel_size
    width = vllm_config.model_config.hf_text_config.intermediate_size // tp
    if width == TP8_INTERMEDIATE_PER_RANK:
        if attention_pads_heads():
            attention = "takes 8 query heads per rank as 8 of its 16 ({}=1)"
        else:
            attention = (
                "is not used: 8 query heads per rank ({} unset), so the sparse "
                "attention layers keep the stock attention"
            )
        logger.info_once(
            "MiniMax-M3 specialized decode at TP%d: the low-batch FFN runs at %d "
            "intermediate columns per rank (%s=1); the low-batch attention %s.",
            tp,
            width,
            LOW_BATCH_FFN_TP8,
            attention.format(LOW_BATCH_ATTN_PAD_HEADS),
            scope="process",
        )
    return True
