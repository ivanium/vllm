# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Metadata for SimpleCPUOffloadConnector."""

from dataclasses import dataclass, field

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata


@dataclass
class SimpleCPUOffloadMetadata(KVConnectorMetadata):
    """
    Metadata passed from scheduler to worker for CPU offload operations.

    This metadata tells the worker which blocks to load from CPU and
    which blocks to store to CPU.
    """

    # Maps request_id -> (gpu_block_ids, cpu_block_ids) for loading
    # GPU blocks will receive data from corresponding CPU blocks
    reqs_to_load: dict[str, tuple[list[int], list[int]]] = field(default_factory=dict)

    # Maps request_id -> (gpu_block_ids, cpu_block_ids) for storing
    # CPU blocks will receive data from corresponding GPU blocks
    reqs_to_store: dict[str, tuple[list[int], list[int]]] = field(default_factory=dict)
