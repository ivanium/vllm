# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side handler for SimpleCPUOffloadConnector."""

from typing import TYPE_CHECKING

import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload import (
    triton_kernels,
)
from vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload.metadata import (
    SimpleCPUOffloadMetadata,
)
from vllm.logger import init_logger
from vllm.utils.platform_utils import is_pin_memory_available

if TYPE_CHECKING:
    from vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload.triton_kernels import (  # noqa: E501
        MultiLayerLaunchParams,
    )
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)


class SimpleCPUOffloadWorker:
    """Worker-side handler for CPU offloading transfers."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: "KVCacheConfig | None",
        cpu_capacity_bytes: int,
    ):
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        self.cpu_capacity_bytes = cpu_capacity_bytes

        cache_config = vllm_config.cache_config
        self.gpu_block_size = cache_config.block_size

        self.gpu_kv_caches: dict[str, torch.Tensor] | None = None
        self.cpu_kv_caches: dict[str, torch.Tensor] | None = None
        self.device: torch.device | None = None
        self._first_layer_name: str | None = None
        self.num_cpu_blocks: int = 0
        self._store_launch_params: MultiLayerLaunchParams | None = None
        self._load_launch_params: MultiLayerLaunchParams | None = None

        self.load_stream: torch.cuda.Stream | None = None
        self.store_stream: torch.cuda.Stream | None = None

        # Ordered (job_idx, event) — stream ordering lets us break early
        self._load_events: list[tuple[int, torch.cuda.Event]] = []
        self._store_events: list[tuple[int, torch.cuda.Event]] = []

        # Deferred stores: queued in wait_for_save(), flushed in start_load_kv()
        self._pending_store_jobs: list[tuple[int, list[int], list[int]]] = []
        self._connector_metadata: SimpleCPUOffloadMetadata | None = None

    @property
    def _is_initialized(self) -> bool:
        """Whether KV caches are registered and ready for transfers."""
        return (
            self.gpu_kv_caches is not None
            and self.cpu_kv_caches is not None
            and self.load_stream is not None
            and self.store_stream is not None
        )

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        """Register per-layer GPU KV caches and allocate pinned CPU tensors."""
        self.gpu_kv_caches = kv_caches
        self.device = next(iter(kv_caches.values())).device
        self._first_layer_name = next(iter(kv_caches))

        first_tensor = next(iter(kv_caches.values()))
        block_size_bytes = first_tensor[0].numel() * first_tensor.element_size()
        num_layers = len(kv_caches)
        self.num_cpu_blocks = max(
            1, self.cpu_capacity_bytes // (block_size_bytes * num_layers)
        )

        logger.info(
            "SimpleCPUOffloadWorker: %d per-layer GPU KV caches, "
            "allocating %d CPU blocks (%.2f GB)",
            num_layers,
            self.num_cpu_blocks,
            (self.num_cpu_blocks * block_size_bytes * num_layers) / (1024**3),
        )

        pin_memory = is_pin_memory_available()
        self.cpu_kv_caches = {}
        for name, gpu_tensor in kv_caches.items():
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

        self._store_launch_params = triton_kernels.build_launch_params(
            self.gpu_kv_caches, self.cpu_kv_caches
        )
        self._load_launch_params = triton_kernels.build_launch_params(
            self.cpu_kv_caches, self.gpu_kv_caches
        )

        self.load_stream = torch.cuda.Stream()
        self.store_stream = torch.cuda.Stream()

    def bind_connector_metadata(self, metadata: SimpleCPUOffloadMetadata) -> None:
        self._connector_metadata = metadata

    def clear_connector_metadata(self) -> None:
        self._connector_metadata = None

    def start_load_kv(self) -> None:
        """Flush deferred stores, then start async loads from CPU to GPU."""
        if not self._is_initialized:
            logger.warning("KV caches not registered, skipping load")
            return

        self._submit_pending_stores()

        if self._connector_metadata is None:
            return

        metadata = self._connector_metadata
        if not metadata.load_gpu_blocks:
            return

        assert self.load_stream is not None
        assert self.cpu_kv_caches is not None
        assert self.gpu_kv_caches is not None

        with torch.cuda.stream(self.load_stream):
            self._copy_blocks(
                src_caches=self.cpu_kv_caches,
                dst_caches=self.gpu_kv_caches,
                src_block_ids=metadata.load_cpu_blocks,
                dst_block_ids=metadata.load_gpu_blocks,
                is_store=False,
            )
            event = torch.cuda.Event()
            event.record(self.load_stream)

        self._load_events.append((metadata.load_job_idx, event))
        logger.debug(
            "Started loading %d blocks from CPU (job_idx=%d)",
            len(metadata.load_gpu_blocks),
            metadata.load_job_idx,
        )

    def wait_for_save(self) -> None:
        """Queue store jobs; actual submission deferred to start_load_kv()."""
        if self._connector_metadata is None:
            return

        if not self._is_initialized:
            return

        metadata = self._connector_metadata
        if not metadata.store_gpu_blocks:
            return

        self._pending_store_jobs.append(
            (
                metadata.store_job_idx,
                list(metadata.store_gpu_blocks),
                list(metadata.store_cpu_blocks),
            )
        )
        logger.debug(
            "Queued storing %d blocks to CPU (job_idx=%d)",
            len(metadata.store_gpu_blocks),
            metadata.store_job_idx,
        )

    def _submit_pending_stores(self) -> None:
        if not self._pending_store_jobs:
            return
        if not self._is_initialized:
            return

        assert self.store_stream is not None
        assert self.gpu_kv_caches is not None
        assert self.cpu_kv_caches is not None

        all_src: list[int] = []
        all_dst: list[int] = []
        for _, src, dst in self._pending_store_jobs:
            all_src.extend(src)
            all_dst.extend(dst)

        with torch.cuda.stream(self.store_stream):
            if all_src:
                self._copy_blocks(
                    src_caches=self.gpu_kv_caches,
                    dst_caches=self.cpu_kv_caches,
                    src_block_ids=all_src,
                    dst_block_ids=all_dst,
                    is_store=True,
                )
            # One event covers all batched jobs; they share the same completion point.
            event = torch.cuda.Event()
            event.record(self.store_stream)

        for job_idx, _, _ in self._pending_store_jobs:
            self._store_events.append((job_idx, event))
            logger.debug("Submitted deferred store to CPU (job_idx=%d)", job_idx)

        self._pending_store_jobs.clear()

    def get_completed_watermarks(self) -> tuple[int, int]:
        """Return (load_wm, store_wm): highest job_idx whose event fired."""
        return (
            self._drain_stream_events(self._load_events),
            self._drain_stream_events(self._store_events),
        )

    @staticmethod
    def _drain_stream_events(events: list[tuple[int, torch.cuda.Event]]) -> int:
        watermark = -1
        while events:
            job_idx, event = events[0]
            if event.query():
                watermark = job_idx
                events.pop(0)
            else:
                break  # Stream ordering: nothing after this can have fired.
        return watermark

    @staticmethod
    def _validate_block_ids(block_ids: list[int], num_blocks: int, label: str) -> None:
        if not block_ids:
            return
        lo, hi = min(block_ids), max(block_ids)
        if lo < 0 or hi >= num_blocks:
            bad = lo if lo < 0 else hi
            raise ValueError(f"{label} block ID {bad} out of bounds [0, {num_blocks})")

    def _copy_blocks(
        self,
        src_caches: dict[str, torch.Tensor],
        dst_caches: dict[str, torch.Tensor],
        src_block_ids: list[int],
        dst_block_ids: list[int],
        is_store: bool = True,
    ) -> None:
        """Execute multi-layer Triton kernel block transfer."""
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

        triton_kernels.copy_blocks(
            src_caches,
            dst_caches,
            block_mapping,
            launch_params=launch_params,
        )

    def handle_preemptions(self) -> None:
        """Sync all in-flight transfers before preempted blocks are reused."""
        self._submit_pending_stores()

        for _, event in self._load_events:
            event.synchronize()
        self._load_events.clear()

        for _, event in self._store_events:
            event.synchronize()
        self._store_events.clear()

    def wait_for_layer_load(self, layer_name: str) -> None:
        """Sync on first layer only (all layers transferred in one launch)."""
        if layer_name == self._first_layer_name and self.load_stream is not None:
            self.load_stream.synchronize()
