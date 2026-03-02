# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
ZeroCPUOffloadConnector: A no-op connector with the same interface as
SimpleCPUOffloadConnector.

Used for benchmarking the connector API overhead in isolation — all
scheduler/worker hooks are called by the engine, but no CPU memory is
allocated and no data is transferred.
"""

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_events import KVCacheEvent
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload.metadata import (
    SimpleCPUOffloadMetadata,
)
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import KVConnectorOutput

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
    from vllm.v1.attention.backend import AttentionBackend, AttentionMetadata
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


class ZeroCPUOffloadConnector(KVConnectorBase_V1, SupportsHMA):
    """
    No-op CPU offloading connector for benchmarking connector API overhead.

    Implements the identical interface as SimpleCPUOffloadConnector but every
    method is a trivial no-op.  This lets you measure the cost the engine pays
    just for *having* a connector wired in (metadata creation, serialisation,
    hook dispatch) without any actual memory allocation or data movement.
    """

    @property
    def prefer_cross_layer_blocks(self) -> bool:
        # Set to False to use the standard per-layer KV cache layout.
        # Set to True to use the cross-layer layout (matches
        # SimpleCPUOffloadConnector, but adds ~10ms TPOT due to changed
        # memory strides in attention kernels).
        return False

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig | None" = None,
    ):
        super().__init__(vllm_config, role, kv_cache_config)
        logger.info(
            "ZeroCPUOffloadConnector: Initializing no-op connector (role=%s)",
            role.name,
        )

    # ------------------------------------------------------------------
    # Worker-side methods (all no-ops)
    # ------------------------------------------------------------------

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        pass

    def register_cross_layers_kv_cache(
        self,
        kv_cache: torch.Tensor,
        attn_backend: type["AttentionBackend"],
    ) -> None:
        pass

    def bind_connector_metadata(
        self,
        connector_metadata: KVConnectorMetadata,
    ) -> None:
        super().bind_connector_metadata(connector_metadata)

    def clear_connector_metadata(self) -> None:
        super().clear_connector_metadata()

    def handle_preemptions(self, preempted_req_ids: set[str]) -> None:
        pass

    def start_load_kv(
        self,
        forward_context: "ForwardContext",
        **kwargs: Any,
    ) -> None:
        pass

    def wait_for_layer_load(self, layer_name: str) -> None:
        pass

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> None:
        pass

    def wait_for_save(self) -> None:
        pass

    def get_finished(
        self,
        finished_req_ids: set[str],
    ) -> tuple[set[str] | None, set[str] | None]:
        return None, None

    # ------------------------------------------------------------------
    # Scheduler-side methods (all no-ops / trivial returns)
    # ------------------------------------------------------------------

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        return 0, False

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        pass

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        return SimpleCPUOffloadMetadata()

    def update_connector_output(
        self,
        connector_output: KVConnectorOutput,
    ) -> None:
        pass

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        return False, None

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        return False, None

    def take_events(self) -> Iterable[KVCacheEvent]:
        return []
