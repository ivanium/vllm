# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Metadata for SimpleCPUOffloadConnector."""

from dataclasses import dataclass, field

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata


@dataclass
class SimpleCPUOffloadMetadata(KVConnectorMetadata):
    """
    Metadata passed from scheduler to worker for CPU offload operations.

    The worker receives flat block lists keyed by a monotonic job_idx.
    The scheduler also ships a complete snapshot of all in-flight jobs
    (job_idx → req_ids) so the connector can translate watermarks to
    req_ids without the worker ever knowing about request identities.
    """

    # This step's load job (block-level, no req_ids).
    # load_job_idx == -1 means no load this step.
    load_job_idx: int = -1
    load_gpu_blocks: list[int] = field(default_factory=list)
    load_cpu_blocks: list[int] = field(default_factory=list)

    # This step's store job (block-level, no req_ids).
    # store_job_idx == -1 means no store this step.
    store_job_idx: int = -1
    store_gpu_blocks: list[int] = field(default_factory=list)
    store_cpu_blocks: list[int] = field(default_factory=list)

    # Complete snapshot of in-flight jobs (for connector translation).
    # Rebuilt from manager state each step; refreshed into connector in
    # bind_connector_metadata().
    pending_load_jobs: dict[int, list[str]] = field(default_factory=dict)
    pending_store_jobs: dict[int, list[str]] = field(default_factory=dict)
