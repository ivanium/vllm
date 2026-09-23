# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import MethodType

from vllm.config import ModelConfig, VllmConfig
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.models import ModelRegistry
from vllm.model_executor.models.qwen3_dflash2 import DFlash2Qwen3ForCausalLM
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheSpec,
    SlidingWindowSpec,
    replace_as,
)


def configure_dflash(draft_config: ModelConfig) -> None:
    """Select GLM's allocation adapter without replacing the generic drafter."""
    if "DFlash2DraftModel" not in draft_config.architectures:
        return
    architecture = "Glm5NextDFlash2DraftModel"
    ModelRegistry.register_model(
        architecture, "vllm.models.glm5next.dflash:Glm5NextDFlash2ForCausalLM"
    )
    # Keep the canonical architecture for DFlash2 speculator selection.
    architectures = [
        architecture,
        *(arch for arch in draft_config.architectures if arch != architecture),
    ]
    draft_config.hf_config.architectures = architectures
    draft_config.model_arch_config.architectures = architectures


def _get_draft_kv_cache_spec(
    attention: Attention, vllm_config: VllmConfig
) -> KVCacheSpec | None:
    spec = Attention.get_kv_cache_spec(attention, vllm_config)
    if isinstance(spec, SlidingWindowSpec):
        return replace_as(
            spec,
            FullAttentionSpec,
            drop=("extra_retained_tokens",),
            block_size=vllm_config.cache_config.block_size,
            page_size_padded=None,
        )
    return spec


class Glm5NextDFlash2ForCausalLM(DFlash2Qwen3ForCausalLM):
    """Adapt only this drafter's SWA allocation to GLM's layer-outer layout."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        for layer in self.model.layers:
            attention = layer.self_attn.attn
            if attention.sliding_window is not None:
                attention.get_kv_cache_spec = MethodType(
                    _get_draft_kv_cache_spec, attention
                )
