# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV cache builder interface and resolution helpers."""

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec


class KVCacheConfigBuilder:
    """Strategy class for model- or platform-specific KV cache planning.

    Subclasses can override individual methods to customize planning while
    keeping the builder import surface small for implementers.
    """

    def get_kv_cache_configs(
        self,
        vllm_config: "VllmConfig",
        kv_cache_specs: list[dict[str, "KVCacheSpec"]],
        available_memory: list[int],
    ) -> list["KVCacheConfig"]:
        """Return the final normalized KV cache configs for all workers."""
        from vllm.v1.core.kv_cache_planning import build_kv_cache_configs

        return build_kv_cache_configs(
            self, vllm_config, kv_cache_specs, available_memory
        )

    def estimate_max_model_len(
        self,
        vllm_config: "VllmConfig",
        kv_cache_specs: list[dict[str, "KVCacheSpec"]],
        available_memory: list[int],
    ) -> int:
        """Return the largest max_model_len that fits across all workers."""
        from vllm.v1.core.kv_cache_planning import estimate_max_model_len

        return estimate_max_model_len(vllm_config, kv_cache_specs, available_memory)


def _load_builder(cls_path: str) -> KVCacheConfigBuilder:
    """Import and instantiate a builder from its fully-qualified class path."""
    module_path, cls_name = cls_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, cls_name)
    return cls()


def resolve_builder(vllm_config: "VllmConfig") -> KVCacheConfigBuilder:
    """Resolve the KV cache config builder to use.

    Priority: platform override > model declaration > default.
    """
    from vllm.platforms import current_platform

    platform_cls_path = current_platform.get_kv_cache_config_builder_cls(vllm_config)
    if platform_cls_path is not None:
        return _load_builder(platform_cls_path)

    model_cls_path = vllm_config.model_config.kv_cache_config_builder_cls
    if model_cls_path is not None:
        return _load_builder(model_cls_path)

    return KVCacheConfigBuilder()
