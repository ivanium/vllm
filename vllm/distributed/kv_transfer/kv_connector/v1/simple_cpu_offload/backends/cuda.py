# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA transfer backend using pinned CPU memory and Triton copy kernels."""

from typing import TYPE_CHECKING, Any

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload import (
    copy_ops,
)
from vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload.backends.base import (  # noqa: E501
    TransferBackend,
)
from vllm.logger import init_logger
from vllm.utils.platform_utils import is_pin_memory_available

if TYPE_CHECKING:
    from vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload.copy_ops import (  # noqa: E501
        LaunchParams,
    )
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)


class CudaTransferBackend(TransferBackend):
    """Transfer backend using pinned CPU memory and Triton copy kernels.

    This is the default backend extracted from
    :class:`SimpleCPUOffloadWorker`.  It allocates pinned host memory as
    the destination, uses low-priority CUDA streams for asynchronous
    transfers, and records :class:`torch.cuda.Event` objects for
    completion tracking.
    """

    def __init__(self) -> None:
        self.gpu_kv_caches: dict[str, torch.Tensor] | None = None
        self.cpu_kv_caches: dict[str, torch.Tensor] | None = None
        self.device: torch.device | None = None
        self.num_cpu_blocks: int = 0

        # Cached Triton launch params
        self._store_launch_params: LaunchParams | None = None
        self._load_launch_params: LaunchParams | None = None

        # CUDA streams for async transfers
        self.load_stream: torch.cuda.Stream | None = None
        self.store_stream: torch.cuda.Stream | None = None

    # ------------------------------------------------------------------
    # TransferBackend interface
    # ------------------------------------------------------------------

    def setup(
        self,
        src_caches: dict[str, torch.Tensor],
        capacity_bytes: int,
        kv_cache_config: "KVCacheConfig | None",
    ) -> int:
        """Allocate pinned CPU tensors and build Triton launch params.

        Args:
            src_caches: Unique GPU KV cache tensors (already deduplicated
                and reshaped to ``[num_blocks, page_size_bytes]``).
            capacity_bytes: Byte budget for pinned CPU memory.
            kv_cache_config: KV cache configuration (unused by this
                backend since *src_caches* are already reshaped).

        Returns:
            Number of CPU blocks allocated.
        """
        self.device = next(iter(src_caches.values())).device
        self.gpu_kv_caches = src_caches

        first = next(iter(src_caches.values()))
        bytes_per_block = first.stride(0) * first.element_size()
        num_unique_tensors = len(src_caches)

        self.num_cpu_blocks = max(
            1, capacity_bytes // (bytes_per_block * num_unique_tensors)
        )

        logger.info(
            "CudaTransferBackend: %d unique GPU KV tensors, "
            "allocating %d CPU blocks (%.2f GB)",
            num_unique_tensors,
            self.num_cpu_blocks,
            (self.num_cpu_blocks * bytes_per_block * num_unique_tensors) / (1024**3),
        )

        pin_memory = is_pin_memory_available()

        self.cpu_kv_caches = {}
        for name, gpu_tensor in src_caches.items():
            cpu_shape = (self.num_cpu_blocks,) + gpu_tensor.shape[1:]
            self.cpu_kv_caches[name] = torch.zeros(
                cpu_shape,
                dtype=gpu_tensor.dtype,
                device="cpu",
                pin_memory=pin_memory,
            )

        if not pin_memory:
            logger.warning(
                "Pinned memory not available. CPU offload performance may be degraded."
            )

        self._store_launch_params = copy_ops.build_launch_params(
            self.gpu_kv_caches, self.cpu_kv_caches
        )
        self._load_launch_params = copy_ops.build_launch_params(
            self.cpu_kv_caches, self.gpu_kv_caches
        )

        # Use lowest priority so KV cache I/O yields to compute streams.
        low_pri, _ = torch.cuda.Stream.priority_range()
        self.load_stream = torch.cuda.Stream(priority=low_pri)
        self.store_stream = torch.cuda.Stream(priority=low_pri)

        return self.num_cpu_blocks

    def copy_blocks(
        self,
        src_block_ids: list[int],
        dst_block_ids: list[int],
        is_store: bool,
    ) -> None:
        assert self.gpu_kv_caches is not None
        assert self.cpu_kv_caches is not None

        if is_store:
            src_caches = self.gpu_kv_caches
            dst_caches = self.cpu_kv_caches
        else:
            src_caches = self.cpu_kv_caches
            dst_caches = self.gpu_kv_caches

        first_src = next(iter(src_caches.values()))
        first_dst = next(iter(dst_caches.values()))
        self._validate_block_ids(src_block_ids, first_src.shape[0], "Source")
        self._validate_block_ids(dst_block_ids, first_dst.shape[0], "Dest")

        block_mapping = torch.tensor(
            list(zip(src_block_ids, dst_block_ids)),
            dtype=torch.int64,
            device=self.device,
        )

        launch_params = (
            self._store_launch_params if is_store else self._load_launch_params
        )

        copy_ops.copy_blocks(
            src_caches,
            dst_caches,
            block_mapping,
            launch_params=launch_params,
        )

    def record_event(self) -> torch.cuda.Event:
        event = torch.cuda.Event()
        event.record()
        return event

    def query_event(self, event: Any) -> bool:
        return event.query()

    def sync_event(self, event: Any) -> None:
        event.synchronize()

    def sync_all(self) -> None:
        if self.load_stream is not None:
            self.load_stream.synchronize()
        if self.store_stream is not None:
            self.store_stream.synchronize()

    @property
    def is_initialized(self) -> bool:
        return (
            self.gpu_kv_caches is not None
            and self.cpu_kv_caches is not None
            and self.load_stream is not None
            and self.store_stream is not None
        )

    def validate_block_ids(self, block_ids: list[int], is_src: bool) -> None:
        if is_src:
            assert self.gpu_kv_caches is not None
            first = next(iter(self.gpu_kv_caches.values()))
            label = "Source (GPU)"
        else:
            assert self.cpu_kv_caches is not None
            first = next(iter(self.cpu_kv_caches.values()))
            label = "Dest (CPU)"
        self._validate_block_ids(block_ids, first.shape[0], label)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_block_ids(block_ids: list[int], num_blocks: int, label: str) -> None:
        if not block_ids:
            return
        lo, hi = min(block_ids), max(block_ids)
        if lo < 0 or hi >= num_blocks:
            bad = lo if lo < 0 else hi
            raise ValueError(f"{label} block ID {bad} out of bounds [0, {num_blocks})")
