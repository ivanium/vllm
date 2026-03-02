# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side handler for SimpleCPUOffloadConnector."""

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload.metadata import (
    SimpleCPUOffloadMetadata,
)
from vllm.logger import init_logger
from vllm.utils.platform_utils import is_pin_memory_available

if TYPE_CHECKING:
    from vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload.triton_kernels import (  # noqa: E501
        MultiLayerLaunchParams,
    )
    from vllm.v1.attention.backend import AttentionBackend
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)


@dataclass
class TransferJob:
    """Tracks an in-flight transfer."""

    req_id: str
    is_store: bool  # True for GPU->CPU, False for CPU->GPU
    event: torch.cuda.Event


class SimpleCPUOffloadWorker:
    """
    Worker-side handler for CPU offloading transfers.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: "KVCacheConfig | None",
        cpu_capacity_bytes: int,
    ):
        """
        Initialize the worker-side handler.

        Args:
            vllm_config: vLLM configuration
            kv_cache_config: KV cache configuration
            cpu_capacity_bytes: CPU memory capacity in bytes
        """
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        self.cpu_capacity_bytes = cpu_capacity_bytes

        cache_config = vllm_config.cache_config
        self.gpu_block_size = cache_config.block_size

        # Will be set when KV cache is registered
        # Cross-layer mode (when prefer_cross_layer_blocks=True)
        self.gpu_kv_cache: torch.Tensor | None = None
        self.cpu_kv_cache: torch.Tensor | None = None
        # Per-layer mode (when prefer_cross_layer_blocks=False)
        self.gpu_kv_caches: dict[str, torch.Tensor] | None = None
        self.cpu_kv_caches: dict[str, torch.Tensor] | None = None
        self.num_cpu_blocks: int = 0
        # Cached Triton launch params for per-layer transfers (store/load)
        self._store_launch_params: MultiLayerLaunchParams | None = None
        self._load_launch_params: MultiLayerLaunchParams | None = None

        # CUDA streams for async transfers
        self.load_stream: torch.cuda.Stream | None = None
        self.store_stream: torch.cuda.Stream | None = None

        # Job tracking
        self._job_counter = 0
        self._active_jobs: dict[int, TransferJob] = {}
        self._load_jobs: dict[str, int] = {}  # req_id -> job_id
        self._store_jobs: dict[str, set[int]] = defaultdict(set)
        # Pending store jobs are deferred to the beginning of the next step.
        self._unsubmitted_store_jobs: list[tuple[int, str, list[int], list[int]]] = []

        # Track requests that have finished generating but still have pending stores
        self._finished_reqs_waiting_for_store: set[str] = set()

        # Current metadata (set per step)
        self._connector_metadata: SimpleCPUOffloadMetadata | None = None

    @property
    def _is_initialized(self) -> bool:
        """Whether KV caches are registered and ready for transfers."""
        return (self.gpu_kv_cache is not None and self.cpu_kv_cache is not None) or (
            self.gpu_kv_caches is not None and self.cpu_kv_caches is not None
        )

    @property
    def _is_per_layer(self) -> bool:
        """Whether per-layer KV cache mode is active."""
        return self.gpu_kv_caches is not None

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        """
        Register per-layer KV caches.

        Args:
            kv_caches: Dict mapping layer name to KV cache tensor
        """
        self.gpu_kv_caches = kv_caches

        # Compute per-layer block size from any layer
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

        # Allocate pinned CPU tensors matching each GPU layer
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

        # Pre-compute Triton launch params (pointer tables, block/warp config)
        from vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload import (
            triton_kernels,
        )

        self._store_launch_params = triton_kernels.build_multi_layer_launch_params(
            self.gpu_kv_caches, self.cpu_kv_caches
        )
        self._load_launch_params = triton_kernels.build_multi_layer_launch_params(
            self.cpu_kv_caches, self.gpu_kv_caches
        )

        # Initialize CUDA streams
        self.load_stream = torch.cuda.Stream()
        self.store_stream = torch.cuda.Stream()

    def register_cross_layers_kv_cache(
        self,
        kv_cache: torch.Tensor,
        attn_backend: type["AttentionBackend"],
    ) -> None:
        """
        Register cross-layer KV cache tensor.

        Args:
            kv_cache: Cross-layer KV cache tensor [num_blocks, ...]
            attn_backend: The attention backend
        """
        self.gpu_kv_cache = kv_cache

        # Calculate CPU block capacity
        # kv_cache shape: [num_blocks, num_layers, 2, num_heads, block_size, ...]
        # Shape may be permuted depending on attention backend
        block_size_bytes = kv_cache[0].numel() * kv_cache.element_size()
        self.num_cpu_blocks = max(1, self.cpu_capacity_bytes // block_size_bytes)

        logger.info(
            "SimpleCPUOffloadWorker: GPU KV cache shape %s, "
            "allocating %d CPU blocks (%.2f GB)",
            kv_cache.shape,
            self.num_cpu_blocks,
            (self.num_cpu_blocks * block_size_bytes) / (1024**3),
        )

        # Allocate pinned CPU tensor with same shape as single GPU block
        cpu_shape = (self.num_cpu_blocks,) + kv_cache.shape[1:]
        pin_memory = is_pin_memory_available()

        self.cpu_kv_cache = torch.zeros(
            cpu_shape,
            dtype=kv_cache.dtype,
            device="cpu",
            pin_memory=pin_memory,
        )

        if not pin_memory:
            logger.warning(
                "Pinned memory not available. CPU offload performance may be degraded."
            )

        # Initialize CUDA streams
        self.load_stream = torch.cuda.Stream()
        self.store_stream = torch.cuda.Stream()

    def bind_connector_metadata(
        self,
        metadata: SimpleCPUOffloadMetadata,
    ) -> None:
        """Bind metadata for the current step."""
        self._connector_metadata = metadata

    def clear_connector_metadata(self) -> None:
        """Clear metadata after the step."""
        self._connector_metadata = None

    def start_load_kv(self) -> None:
        """
        Start async loads from CPU to GPU.

        Called before the forward pass to overlap transfer with compute.
        """
        if not self._is_initialized:
            logger.warning("KV caches not registered, skipping load")
            return

        # Defer stores to start of the next step to avoid contention with
        # post-forward sampling work in the current step.
        self._submit_pending_store_jobs()

        if self._connector_metadata is None:
            return

        for req_id, (
            dst_gpu_blocks,
            src_cpu_blocks,
        ) in self._connector_metadata.reqs_to_load.items():
            if not dst_gpu_blocks or not src_cpu_blocks:
                continue

            job_id = self._job_counter
            self._job_counter += 1

            # Submit async load
            assert self.load_stream is not None
            with torch.cuda.stream(self.load_stream):
                if self.gpu_kv_caches is not None and self.cpu_kv_caches is not None:
                    self._transfer_blocks_per_layer(
                        src_caches=self.cpu_kv_caches,
                        dst_caches=self.gpu_kv_caches,
                        src_block_ids=src_cpu_blocks,
                        dst_block_ids=dst_gpu_blocks,
                        is_store=False,
                    )
                else:
                    self._transfer_blocks(
                        src_cache=self.cpu_kv_cache,
                        dst_cache=self.gpu_kv_cache,
                        src_block_ids=src_cpu_blocks,
                        dst_block_ids=dst_gpu_blocks,
                    )
                event = torch.cuda.Event()
                event.record(self.load_stream)

            self._active_jobs[job_id] = TransferJob(
                req_id=req_id,
                is_store=False,
                event=event,
            )
            self._load_jobs[req_id] = job_id

            logger.debug(
                "Request %s: Started loading %d blocks from CPU",
                req_id,
                len(src_cpu_blocks),
            )

    def wait_for_save(self) -> None:
        """
        Queue async stores from GPU to CPU.

        Called after the forward pass to stage computed KV cache stores.
        Stores are submitted in start_load_kv() of the next step.
        """
        if self._connector_metadata is None:
            return

        if not self._is_initialized:
            return

        for req_id, (
            src_gpu_blocks,
            dst_cpu_blocks,
        ) in self._connector_metadata.reqs_to_store.items():
            if not src_gpu_blocks or not dst_cpu_blocks:
                continue

            job_id = self._job_counter
            self._job_counter += 1

            self._store_jobs[req_id].add(job_id)
            self._unsubmitted_store_jobs.append(
                (job_id, req_id, list(src_gpu_blocks), list(dst_cpu_blocks))
            )

            logger.debug(
                "Request %s: Queued storing %d blocks to CPU",
                req_id,
                len(src_gpu_blocks),
            )

    def _submit_pending_store_jobs(self) -> None:
        """Submit deferred store jobs."""
        if not self._unsubmitted_store_jobs:
            return
        if not self._is_initialized:
            return

        assert self.store_stream is not None
        with torch.cuda.stream(self.store_stream):
            # Ensure source blocks are ready before store DMA begins.
            self.store_stream.wait_stream(torch.cuda.current_stream())
            for (
                job_id,
                req_id,
                src_gpu_blocks,
                dst_cpu_blocks,
            ) in self._unsubmitted_store_jobs:
                if self.gpu_kv_caches is not None and self.cpu_kv_caches is not None:
                    self._transfer_blocks_per_layer(
                        src_caches=self.gpu_kv_caches,
                        dst_caches=self.cpu_kv_caches,
                        src_block_ids=src_gpu_blocks,
                        dst_block_ids=dst_cpu_blocks,
                    )
                else:
                    self._transfer_blocks(
                        src_cache=self.gpu_kv_cache,
                        dst_cache=self.cpu_kv_cache,
                        src_block_ids=src_gpu_blocks,
                        dst_block_ids=dst_cpu_blocks,
                    )
                event = torch.cuda.Event()
                event.record(self.store_stream)
                self._active_jobs[job_id] = TransferJob(
                    req_id=req_id,
                    is_store=True,
                    event=event,
                )
                logger.debug(
                    "Request %s: Started deferred storing %d blocks to CPU",
                    req_id,
                    len(src_gpu_blocks),
                )

        self._unsubmitted_store_jobs.clear()

    @staticmethod
    def _validate_block_ids(
        block_ids: list[int],
        num_blocks: int,
        label: str,
    ) -> None:
        """Validate that all block IDs are within bounds."""
        if not block_ids:
            return
        lo, hi = min(block_ids), max(block_ids)
        if lo < 0 or hi >= num_blocks:
            bad = lo if lo < 0 else hi
            raise ValueError(f"{label} block ID {bad} out of bounds [0, {num_blocks})")

    def _transfer_blocks(
        self,
        src_cache: torch.Tensor,
        dst_cache: torch.Tensor,
        src_block_ids: list[int],
        dst_block_ids: list[int],
    ) -> None:
        """
        Execute block transfer using Triton kernel.

        Args:
            src_cache: Source KV cache tensor
            dst_cache: Destination KV cache tensor
            src_block_ids: Source block IDs
            dst_block_ids: Destination block IDs
        """
        from vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload import (
            triton_kernels,
        )

        self._validate_block_ids(src_block_ids, src_cache.shape[0], "Source")
        self._validate_block_ids(dst_block_ids, dst_cache.shape[0], "Dest")

        # Build block mapping tensor: [[src_id, dst_id], ...]
        block_mapping = torch.tensor(
            list(zip(src_block_ids, dst_block_ids)),
            dtype=torch.int64,
            device="cuda",
        )

        triton_kernels.copy_blocks(src_cache, dst_cache, block_mapping, use_triton=True)

    def _transfer_blocks_per_layer(
        self,
        src_caches: dict[str, torch.Tensor],
        dst_caches: dict[str, torch.Tensor],
        src_block_ids: list[int],
        dst_block_ids: list[int],
        is_store: bool = True,
    ) -> None:
        """
        Execute per-layer block transfer using a single multi-layer Triton
        kernel launch.

        Uses pointer tables and stride-based addressing to handle
        non-contiguous (permuted) GPU tensors without requiring .view(-1).

        Args:
            src_caches: Dict of layer name -> source KV cache tensor
            dst_caches: Dict of layer name -> destination KV cache tensor
            src_block_ids: Source block IDs
            dst_block_ids: Destination block IDs
            is_store: True for GPU->CPU, False for CPU->GPU
        """
        from vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload import (
            triton_kernels,
        )

        # Validate block IDs against first layer (all layers share block count)
        first_src = next(iter(src_caches.values()))
        first_dst = next(iter(dst_caches.values()))
        self._validate_block_ids(src_block_ids, first_src.shape[0], "Source")
        self._validate_block_ids(dst_block_ids, first_dst.shape[0], "Dest")

        # Build block mapping tensor once for all layers
        block_mapping = torch.tensor(
            list(zip(src_block_ids, dst_block_ids)),
            dtype=torch.int64,
            device="cuda",
        )

        # Use cached launch params when available
        launch_params = (
            self._store_launch_params if is_store else self._load_launch_params
        )

        try:
            triton_kernels.copy_blocks_multi_layer(
                src_caches,
                dst_caches,
                block_mapping,
                launch_params=launch_params,
            )
        except Exception as e:
            logger.warning(
                "Multi-layer Triton kernel failed, falling back to "
                "per-layer PyTorch copy: %s",
                e,
            )
            for layer_name in src_caches:
                triton_kernels.copy_blocks_torch(
                    src_caches[layer_name],
                    dst_caches[layer_name],
                    block_mapping,
                )

    def get_finished(
        self,
        finished_req_ids: set[str],
    ) -> tuple[set[str] | None, set[str] | None]:
        """
        Check for completed transfers and return finished request IDs.

        Args:
            finished_req_ids: Request IDs that have finished generating

        Returns:
            Tuple of (finished_sending, finished_recving)
            - finished_sending: Requests that finished generating AND completed
              all async stores
            - finished_recving: Requests that completed async loading
        """
        finished_sending: set[str] = set()
        finished_recving: set[str] = set()

        # Poll CUDA events for completion
        completed_job_ids = []
        for job_id, job in list(self._active_jobs.items()):
            if job.event.query():  # Non-blocking check
                completed_job_ids.append(job_id)

        # Process completed jobs
        for job_id in completed_job_ids:
            job = self._active_jobs.pop(job_id)

            if job.is_store:
                self._store_jobs[job.req_id].discard(job_id)
                if (
                    not self._store_jobs[job.req_id]
                    and job.req_id in self._finished_reqs_waiting_for_store
                ):
                    self._finished_reqs_waiting_for_store.remove(job.req_id)
                    finished_sending.add(job.req_id)
                    self._store_jobs.pop(job.req_id, None)
            else:
                self._load_jobs.pop(job.req_id, None)
                finished_recving.add(job.req_id)

        # Track requests that finished generating but still have pending stores
        for req_id in finished_req_ids:
            pending_store_jobs = self._store_jobs.get(req_id)
            if pending_store_jobs:
                # Request finished generating but has pending stores
                self._finished_reqs_waiting_for_store.add(req_id)
            elif pending_store_jobs is not None:
                # Request finished and no pending stores (empty set)
                finished_sending.add(req_id)
                del self._store_jobs[req_id]

        return (
            finished_sending if finished_sending else None,
            finished_recving if finished_recving else None,
        )

    def handle_preemptions(self, preempted_req_ids: set[str]) -> None:
        """
        Handle preempted requests before their blocks are overwritten.

        Args:
            preempted_req_ids: IDs of preempted requests
        """
        # Flush deferred stores first so waits below include them.
        self._submit_pending_store_jobs()

        # Wait for any in-flight loads for preempted requests
        # (to avoid overwriting GPU blocks still being loaded into)
        for req_id in preempted_req_ids:
            job_id = self._load_jobs.get(req_id)
            if job_id is not None:
                job = self._active_jobs.get(job_id)
                if job is not None:
                    job.event.synchronize()

        # Wait for any in-flight stores for preempted requests
        for req_id in preempted_req_ids:
            job_ids = self._store_jobs.get(req_id, set())
            for job_id in job_ids:
                job = self._active_jobs.get(job_id)
                if job is not None:
                    job.event.synchronize()

    def wait_for_layer_load(self, layer_name: str) -> None:
        """Block until the KV for a specific layer is loaded."""
        # For cross-layer cache, all layers are loaded together
        # Just sync the load stream
        if self.load_stream is not None:
            self.load_stream.synchronize()
