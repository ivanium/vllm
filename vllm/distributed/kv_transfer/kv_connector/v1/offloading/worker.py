# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.utils import group_pool_storage
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.canonical_mapping import (
    derive_canonical_mappings,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingConnectorMetadata,
    OffloadingWorkerMetadata,
    ReqId,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.config import (
    get_offloading_group_ids,
)
from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.kv_offload.base import (
    CanonicalKVCacheRef,
    CanonicalKVCaches,
    CanonicalKVCacheTensor,
    GPULoadStoreSpec,
    LoadStoreSpec,
    OffloadingSpec,
    OffloadingWorker,
)

logger = init_logger(__name__)


class OffloadingConnectorWorker:
    """Implementation of Worker side methods."""

    def __init__(
        self,
        spec: OffloadingSpec,
        vllm_config: "VllmConfig",
        kv_cache_config: KVCacheConfig,
    ):
        self.spec = spec
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        self.worker: OffloadingWorker | None = None
        # Non-writers still ack: pending_count waits for world_size per job.
        self._is_store_writer = (
            not self.spec.replicated_layout
            or self.spec.config.canonical_layout
            or self.spec.config.parallel.rank == 0
        )

        # job_id -> req_id for in-flight loads.
        self._load_jobs: dict[int, ReqId] = {}
        self._unsubmitted_store_jobs: list[
            tuple[int, GPULoadStoreSpec, LoadStoreSpec]
        ] = []
        self._connector_worker_meta = OffloadingWorkerMetadata()

    def _init_worker(self, kv_caches: CanonicalKVCaches) -> None:
        self.worker = self.spec.get_worker(kv_caches)

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """Register the GPU KV cache regions, one per offloaded group.

        Every layer view of a group strides over one shared backing
        allocation with stride(0) = the group's row_bytes, so manager block
        ``b`` of the group is one contiguous byte range at
        ``b * block_bytes`` holding all of the group's layers. The transfer
        unit is the whole manager block: one tensor and one data ref per
        group.
        """
        kv_cache_config = self.kv_cache_config
        selected_group_ids = get_offloading_group_ids(kv_cache_config)
        mappings = derive_canonical_mappings(
            self.vllm_config, kv_cache_config, selected_group_ids
        )

        tensors: list[CanonicalKVCacheTensor] = []
        group_data_refs: list[list[CanonicalKVCacheRef]] = []
        for group_id in selected_group_ids:
            group = kv_cache_config.kv_cache_groups[group_id]
            assert not group.host_resident
            storage = group_pool_storage(kv_caches, kv_cache_config, group_id)
            if storage is None:
                # This rank holds no layer of the group (pipeline parallelism).
                group_data_refs.append([])
                continue
            block_bytes = kv_cache_config.group_geometry(group_id).block_bytes
            tensor = torch.empty(0, dtype=torch.int8, device=storage.device).set_(
                storage, 0, (group.num_blocks, block_bytes)
            )
            tensors.append(
                CanonicalKVCacheTensor(tensor=tensor, page_size_bytes=block_bytes)
            )
            group_data_refs.append(
                [
                    CanonicalKVCacheRef(
                        tensor_idx=len(tensors) - 1,
                        page_size_bytes=block_bytes,
                        mapping=mappings.get(group_id),
                    )
                ]
            )

        self._init_worker(
            CanonicalKVCaches(tensors=tensors, group_data_refs=group_data_refs)
        )

    def handle_preemptions(self, kv_connector_metadata: OffloadingConnectorMetadata):
        assert self.worker is not None

        # Pop jobs_to_flush from store_jobs into _unsubmitted_store_jobs
        # so the existing submission loop below submits them before wait().
        if kv_connector_metadata.jobs_to_flush:
            for job_id in kv_connector_metadata.jobs_to_flush:
                entry = kv_connector_metadata.store_jobs.pop(job_id, None)
                if entry is not None:
                    if not self._is_store_writer:
                        self._connector_worker_meta.mark_completed(job_id)
                        continue
                    assert isinstance(entry.src_spec, GPULoadStoreSpec)
                    self._unsubmitted_store_jobs.append(
                        (job_id, entry.src_spec, entry.dst_spec)
                    )

        # Submit deferred stores from previous step (and jobs_to_flush above).
        for job_id, src_spec, dst_spec in self._unsubmitted_store_jobs:
            assert isinstance(src_spec, GPULoadStoreSpec)
            success = self.worker.submit_store(job_id, src_spec, dst_spec)
            assert success
        self._unsubmitted_store_jobs.clear()

        if kv_connector_metadata.jobs_to_flush:
            self.worker.wait(kv_connector_metadata.jobs_to_flush)

    def start_kv_transfers(self, metadata: OffloadingConnectorMetadata):
        assert self.worker is not None
        for job_id, src_spec, dst_spec in self._unsubmitted_store_jobs:
            success = self.worker.submit_store(job_id, src_spec, dst_spec)
            assert success
        self._unsubmitted_store_jobs.clear()

        for job_id, entry in metadata.load_jobs.items():
            self._load_jobs[job_id] = entry.req_id
            assert isinstance(entry.dst_spec, GPULoadStoreSpec)
            success = self.worker.submit_load(job_id, entry.src_spec, entry.dst_spec)
            assert success

    def prepare_store_kv(self, metadata: OffloadingConnectorMetadata):
        for job_id, entry in metadata.store_jobs.items():
            if not self._is_store_writer:
                # Gate before queueing: no _unsubmitted_store_jobs entry.
                self._connector_worker_meta.mark_completed(job_id)
                continue
            # NOTE(orozery): defer the store to the beginning of the next
            # engine step, so that offloading starts AFTER transfers related
            # to token sampling, thereby avoiding delays to token generation.
            assert isinstance(entry.src_spec, GPULoadStoreSpec)
            self._unsubmitted_store_jobs.append(
                (job_id, entry.src_spec, entry.dst_spec)
            )

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        """Returns:
        tuple of (finished_sending, finished_recving). Stores never
        emit finished_sending — the scheduler tracks store completion
        via kv_connector_worker_meta.completed_jobs and fences any
        block reuse via jobs_to_flush. Loads still emit
        finished_recving so the base scheduler can resume requests
        blocked on remote KV (and free aborted-during-load reqs).

        """
        assert self.worker is not None
        finished_recving: set[str] = set()
        for transfer_result in self.worker.get_finished():
            # we currently do not support job failures
            job_id = transfer_result.job_id
            assert transfer_result.success
            is_load = job_id in self._load_jobs
            if (
                transfer_result.transfer_time is not None
                and transfer_result.transfer_size is not None
            ):
                if is_load:
                    stats = self._connector_worker_meta.transfer_stats.load
                else:
                    stats = self._connector_worker_meta.transfer_stats.store
                stats.record(
                    transfer_result.transfer_size,
                    transfer_result.transfer_time,
                )

            self._connector_worker_meta.mark_completed(job_id)
            req_id = self._load_jobs.pop(job_id, None)
            if req_id is not None:
                finished_recving.add(req_id)

        return set(), finished_recving

    def build_connector_worker_meta(self) -> OffloadingWorkerMetadata | None:
        """Return completed transfer job IDs since the last call."""
        if not self._connector_worker_meta.completed_jobs:
            return None
        meta = self._connector_worker_meta
        self._connector_worker_meta = OffloadingWorkerMetadata()
        return meta

    def shutdown(self) -> None:
        self._unsubmitted_store_jobs.clear()
        self._load_jobs.clear()
        self._connector_worker_meta = OffloadingWorkerMetadata()
        if self.worker is not None:
            self.worker.shutdown()
