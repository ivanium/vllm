# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pluggable KV cache configuration builder.

Models and hardware platforms can provide custom builders to control how
KVCacheSpecs are translated into KVCacheConfig (layer grouping, memory
budgeting, tensor allocation).
"""

import importlib
from typing import TYPE_CHECKING

from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheSpec


class KVCacheConfigBuilder:
    """Strategy for converting KVCacheSpecs into KVCacheConfig.

    Subclass and override methods to customize grouping, memory budgeting,
    and tensor allocation for specific models or hardware platforms.

    The default implementations in this base class call the corresponding
    functions in ``kv_cache_utils``.  Override any method to replace just
    that phase of the pipeline.
    """

    def get_kv_cache_groups(
        self,
        vllm_config: "VllmConfig",
        kv_cache_spec: dict[str, "KVCacheSpec"],
    ) -> list[KVCacheGroupSpec]:
        """Decide how to group layers into KVCacheGroupSpecs."""
        from vllm.v1.core.kv_cache_utils import get_kv_cache_groups

        return get_kv_cache_groups(vllm_config, kv_cache_spec)

    def get_kv_cache_config_from_groups(
        self,
        vllm_config: "VllmConfig",
        kv_cache_groups: list[KVCacheGroupSpec],
        available_memory: int,
    ) -> KVCacheConfig:
        """Allocate blocks and tensors from groups."""
        from vllm.v1.core.kv_cache_utils import get_kv_cache_config_from_groups

        return get_kv_cache_config_from_groups(
            vllm_config, kv_cache_groups, available_memory
        )

    def max_memory_usage_bytes_from_groups(
        self,
        vllm_config: "VllmConfig",
        kv_cache_groups: list[KVCacheGroupSpec],
    ) -> int:
        """Memory needed for one request at max_model_len."""
        from vllm.v1.core.kv_cache_utils import (
            _max_memory_usage_bytes_from_groups,
        )

        return _max_memory_usage_bytes_from_groups(vllm_config, kv_cache_groups)


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

    # 1. Platform override (vendor customization, per-model)
    platform_cls_path = current_platform.get_kv_cache_config_builder_cls(vllm_config)
    if platform_cls_path is not None:
        return _load_builder(platform_cls_path)

    # 2. Model declaration
    model_cls_path = vllm_config.model_config.kv_cache_config_builder_cls
    if model_cls_path is not None:
        return _load_builder(model_cls_path)

    # 3. Default
    return KVCacheConfigBuilder()
