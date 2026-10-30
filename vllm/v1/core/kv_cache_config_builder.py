# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Interface and resolution for pluggable KV cache config builders."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from vllm.utils.import_utils import resolve_obj_by_qualname

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import (
        KVCacheConfig,
        KVCacheGroupSpec,
        KVCacheSpec,
    )


class KVCacheConfigBuilder(ABC):
    """Interface for platform KV cache placement.

    .. note::
        This interface is experimental and subject to change. The exact hooks
        and the ``num_blocks`` contract may evolve before it stabilizes.

    Subclasses normally inherit from ``DefaultKVCacheConfigBuilder`` and
    override only the hooks they need. The cross-worker planning flow
    (:func:`vllm.v1.core.kv_cache_planning.get_kv_cache_configs`) is Core-owned;
    this builder supplies the placement decisions: how layers are split into
    groups and how a group list is materialized into tensors for a given block
    count. Grouping coordination lives in ``DefaultKVCacheConfigBuilder``; a
    platform that replaces the builder fully owns its own grouping.
    """

    @abstractmethod
    def get_kv_cache_groups(
        self,
        vllm_config: "VllmConfig",
        kv_cache_spec: dict[str, "KVCacheSpec"],
    ) -> list["KVCacheGroupSpec"]:
        """Split the given layers into KV cache groups.

        Planning passes all workers' layers; profiling passes one worker's.
        Core rejects a result that does not put each layer in exactly one group.
        """
        raise NotImplementedError

    @abstractmethod
    def get_kv_cache_config_from_groups(
        self,
        vllm_config: "VllmConfig",
        kv_cache_groups: list["KVCacheGroupSpec"],
        num_blocks: int,
    ) -> "KVCacheConfig":
        """Materialize the groups for at least ``num_blocks`` blocks per group.

        This hook is the worker-local profiling path: ``num_blocks`` is the
        minimum block count each group must hold, and the placement (slots per
        large page, page bytes) is computed locally from the given groups —
        the engine re-plans with cross-rank memory after profiling.
        """
        raise NotImplementedError


def get_kv_cache_config_builder(
    vllm_config: "VllmConfig",
) -> KVCacheConfigBuilder:
    """Resolve the builder selected by the current platform.

    Resolution priority is owned by the platform hook
    (:meth:`vllm.platforms.interface.Platform.get_kv_cache_config_builder_cls`).
    Builders are stateless (every method takes ``vllm_config``), so a fresh
    instance is returned per call and engines with different configs in the
    same process always get the correct builder.
    """
    from vllm.platforms import current_platform

    builder_cls = resolve_obj_by_qualname(
        current_platform.get_kv_cache_config_builder_cls(vllm_config)
    )
    return builder_cls()
