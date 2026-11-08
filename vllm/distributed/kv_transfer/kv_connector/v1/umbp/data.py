# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared data types for the UMBP KV connector.

The types in this module deliberately do not depend on a UMBP client.  They
describe vLLM's logical KV objects and the physical ranges a runtime adapter
will transfer.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from dataclasses import fields as dataclass_fields
from enum import Enum
from typing import Any, NamedTuple

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorHandshakeMetadata,
    KVConnectorMetadata,
    KVConnectorWorkerMetadata,
)
from vllm.v1.kv_cache_interface import (
    GroupGeometry,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
)


def _layer_specs(group: KVCacheGroupSpec) -> dict[str, KVCacheSpec]:
    """Map each layer of a cache group to its spec at the group block size."""
    if group.layer_kv_cache_specs is not None:
        return dict(group.layer_kv_cache_specs)
    return dict.fromkeys(group.layer_names, group.kv_cache_spec)


def _head_major_run(config: KVCacheConfig, group: KVCacheGroupSpec) -> int | None:
    """Bytes of a head-major (BHLNC) group's heads in one kernel row, or None
    when each layer's page is one contiguous run of the row.

    The planner lays such a group out as one tensor whose layer stride is one
    head of one layer.
    """
    if not group.is_head_major(config.kv_cache_layout):
        return None
    tensor = next(
        t for t in config.kv_cache_tensors if t.layers[0] in group.layer_names
    )
    return group.kv_cache_spec.num_heads * len(tensor.layers) * tensor.layer_stride


def _spec_fingerprint(spec: KVCacheSpec) -> dict[str, str]:
    """Every field that shapes a layer's stored bytes, in a stable form."""
    return {
        "kind": type(spec).__name__,
        **{item.name: str(getattr(spec, item.name)) for item in dataclass_fields(spec)},
    }


@dataclass(frozen=True)
class RankTopology:
    """Model-parallel ranks that participate in one KV object."""

    tp_rank: int = 0
    tp_size: int = 1
    pp_rank: int = 0
    pp_size: int = 1
    pcp_rank: int = 0
    pcp_size: int = 1
    dcp_rank: int = 0
    dcp_size: int = 1

    def __post_init__(self) -> None:
        for name in ("tp_size", "pp_size", "pcp_size", "dcp_size"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for rank, size, name in (
            (self.tp_rank, self.tp_size, "tp_rank"),
            (self.pp_rank, self.pp_size, "pp_rank"),
            (self.pcp_rank, self.pcp_size, "pcp_rank"),
            (self.dcp_rank, self.dcp_size, "dcp_rank"),
        ):
            if rank < 0 or rank >= size:
                raise ValueError(f"{name}={rank} is outside [0, {size})")

    @classmethod
    def from_vllm_config(cls, vllm_config: Any) -> RankTopology:
        parallel = vllm_config.parallel_config
        tp_size = getattr(parallel, "tensor_parallel_size", 1)
        pp_size = getattr(parallel, "pipeline_parallel_size", 1)
        pcp_size = getattr(parallel, "prefill_context_parallel_size", 1)
        # An engine's workers are laid out as [PP, PCP, TP], TP innermost
        # (initialize_model_parallel).
        rank = getattr(parallel, "rank", 0)
        tp_rank = rank % tp_size
        dcp_size = getattr(parallel, "decode_context_parallel_size", 1)
        return cls(
            tp_rank=tp_rank,
            tp_size=tp_size,
            pp_rank=rank // (tp_size * pcp_size) % pp_size,
            pp_size=pp_size,
            pcp_rank=rank // tp_size % pcp_size,
            pcp_size=pcp_size,
            # DCP partitions the TP workers; it does not create a second
            # Cartesian worker dimension. ParallelConfig is constructed before
            # process groups, so derive this stable identity from TP rank.
            dcp_rank=tp_rank % dcp_size,
            dcp_size=dcp_size,
        )

    @property
    def local_namespace(self) -> tuple[int, int, int, int]:
        return (self.tp_rank, self.pcp_rank, self.dcp_rank, self.pp_rank)

    @property
    def rank_count(self) -> int:
        return len(self.all_namespaces())

    def all_namespaces(self) -> tuple[tuple[int, int, int, int], ...]:
        if self.dcp_size > 1:
            if self.tp_size % self.dcp_size:
                raise ValueError("dcp_size must divide tp_size")
            return tuple(
                (tp, pcp, tp % self.dcp_size, pp)
                for pp in range(self.pp_size)
                for pcp in range(self.pcp_size)
                for tp in range(self.tp_size)
            )
        return tuple(
            (tp, pcp, dcp, pp)
            for pp in range(self.pp_size)
            for dcp in range(self.dcp_size)
            for pcp in range(self.pcp_size)
            for tp in range(self.tp_size)
        )


@dataclass(frozen=True)
class UMBPNamespace:
    """Compatibility namespace for objects stored outside the engine."""

    value: str

    @classmethod
    def configured(cls, vllm_config: Any) -> UMBPNamespace | None:
        """Return the explicit ``key_namespace``, or None when it is derived."""
        extra = vllm_config.kv_transfer_config.kv_connector_extra_config
        configured = extra.get("key_namespace", "auto")
        if configured == "auto":
            return None
        if not isinstance(configured, str) or not configured:
            raise ValueError("key_namespace must be a non-empty string or 'auto'")
        return cls(configured)

    @classmethod
    def from_vllm_config(
        cls,
        vllm_config: Any,
        kv_cache_config: KVCacheConfig,
    ) -> UMBPNamespace:
        """Derive the namespace from a worker's per-layer KV cache config.

        Only workers hold per-layer specs and group geometry, so the
        scheduler adopts the namespace its workers report instead.
        """
        if (configured := cls.configured(vllm_config)) is not None:
            return configured

        model_config = vllm_config.model_config
        parallel_config = vllm_config.parallel_config
        # Draft-model KV (e.g. DSpark/EAGLE layers) is stored alongside the
        # target's, so a different drafter must not reuse it.
        speculative_config = getattr(vllm_config, "speculative_config", None)
        draft_model_config = getattr(speculative_config, "draft_model_config", None)
        speculative = (
            None
            if speculative_config is None
            else {
                "method": getattr(speculative_config, "method", None),
                "model": getattr(draft_model_config, "model", None),
                "revision": getattr(draft_model_config, "revision", None),
            }
        )
        fields = {
            "model": getattr(model_config, "model", ""),
            "revision": getattr(model_config, "revision", None),
            "block_size": vllm_config.cache_config.block_size,
            "cache_layout": getattr(kv_cache_config, "kv_cache_layout", None),
            "groups": [
                {
                    "layers": group.layer_names,
                    "block_size": group.kv_cache_spec.block_size,
                    "kernel_rows_per_block": group.kernel_rows_per_block,
                    "row_bytes": group.row_bytes,
                    "layer_specs": {
                        name: _spec_fingerprint(spec)
                        for name, spec in _layer_specs(group).items()
                    },
                }
                for group in kv_cache_config.kv_cache_groups
            ],
            "tp": parallel_config.tensor_parallel_size,
            "pp": parallel_config.pipeline_parallel_size,
            "pcp": getattr(parallel_config, "prefill_context_parallel_size", 1),
            "dcp": getattr(parallel_config, "decode_context_parallel_size", 1),
            "layout_version": 3,
        }
        if speculative is not None:
            # Added only when set, so namespaces without a drafter are unchanged.
            fields["speculative"] = speculative
        payload = json.dumps(fields, sort_keys=True, separators=(",", ":")).encode()
        return cls(hashlib.sha256(payload).hexdigest()[:32])


@dataclass(frozen=True)
class UMBPHandshakeMetadata(KVConnectorHandshakeMetadata):
    """Namespace a worker derived, reported to the scheduler at startup."""

    namespace: str


@dataclass(frozen=True)
class BlockIdentityCodec:
    """Build stable rank-local keys from vLLM block hashes."""

    namespace: UMBPNamespace
    tp_rank: int = 0
    pp_rank: int = 0
    pcp_rank: int = 0
    dcp_rank: int = 0

    def key(self, block_hash: bytes, group_id: int = 0) -> str:
        return self.key_for_namespace(
            block_hash,
            group_id,
            (self.tp_rank, self.pcp_rank, self.dcp_rank, self.pp_rank),
        )

    def key_for_namespace(
        self,
        block_hash: bytes,
        group_id: int,
        namespace: tuple[int, int, int, int],
    ) -> str:
        if not isinstance(block_hash, bytes) or not block_hash:
            raise ValueError("block_hash must be non-empty bytes")
        if group_id < 0:
            raise ValueError("group_id must be non-negative")
        tp_rank, pcp_rank, dcp_rank, pp_rank = namespace
        return (
            f"umbp:vllm:v1:{self.namespace.value}:"
            f"tp{tp_rank}:pcp{pcp_rank}:dcp{dcp_rank}:"
            f"pp{pp_rank}:g{group_id}:"
            f"{block_hash.hex()}"
        )

    def keys_for_topology(
        self,
        block_hash: bytes,
        topology: RankTopology,
        group_ids: Sequence[int],
    ) -> tuple[str, ...]:
        return tuple(
            self.key_for_namespace(block_hash, group_id, namespace)
            for namespace in topology.all_namespaces()
            for group_id in group_ids
        )


@dataclass(frozen=True)
class KVRegion:
    """One layer's per-block region in a rank-local KV object.

    ``block_bytes`` is the layer's bytes per manager block; a manager block
    spans ``kernel_rows_per_block`` kernel rows of its group, and
    ``block_stride`` is the group's ``row_bytes`` (the stride between the
    layer's consecutive kernel rows).
    """

    layer_name: str
    group_id: int
    block_stride: int
    block_bytes: int
    object_offset: int
    block_size: int = 1

    def __post_init__(self) -> None:
        if self.block_stride <= 0 or self.block_bytes <= 0:
            raise ValueError("KV region sizes must be positive")
        if self.block_size <= 0:
            raise ValueError("KV region block_size must be positive")


@dataclass(frozen=True)
class KVLayoutDescriptor:
    """Backend-neutral description of one rank-local KV object layout."""

    regions: tuple[KVRegion, ...]
    topology: RankTopology
    layout_format: str = "rank_local"

    def __post_init__(self) -> None:
        if not self.layout_format:
            raise ValueError("layout_format must not be empty")
        names = [region.layer_name for region in self.regions]
        if len(names) != len(set(names)):
            raise ValueError("KV layout contains duplicate layer regions")

    @property
    def object_size(self) -> int:
        return sum(region.block_bytes for region in self.regions)


class KVLayoutPlanner:
    """Create a deterministic all-layer object layout from KVCacheConfig."""

    def __init__(
        self,
        regions: Sequence[KVRegion],
        group_geometry: Mapping[int, GroupGeometry],
    ) -> None:
        self.regions = tuple(regions)
        self.group_geometry = dict(group_geometry)
        for group_id, geometry in self.group_geometry.items():
            if geometry.kernel_rows_per_block <= 0 or geometry.row_bytes <= 0:
                raise ValueError(f"cache group {group_id} geometry must be positive")
        self._regions_by_group = {
            group_id: tuple(r for r in self.regions if r.group_id == group_id)
            for group_id in {r.group_id for r in self.regions}
        }
        for region in self.regions:
            region_geometry = self.group_geometry.get(region.group_id)
            if region_geometry is None:
                raise ValueError(f"no geometry for cache group {region.group_id}")
            if region.block_stride != region_geometry.row_bytes:
                raise ValueError(
                    f"KV region {region.layer_name!r} row stride "
                    f"{region.block_stride} != group row_bytes "
                    f"{region_geometry.row_bytes}"
                )
            if region.block_bytes % region_geometry.kernel_rows_per_block != 0:
                raise ValueError(
                    f"KV region {region.layer_name!r} block bytes "
                    f"{region.block_bytes} do not divide across "
                    f"{region_geometry.kernel_rows_per_block} kernel rows"
                )
        self._base_addresses: dict[str, int] = {}

    @classmethod
    def from_kv_cache_config(cls, config: KVCacheConfig) -> KVLayoutPlanner:
        group_geometry = {
            group_id: config.group_geometry(group_id)
            for group_id in config.transfer_group_ids
        }
        regions: list[KVRegion] = []
        offset = 0
        for group_id in config.transfer_group_ids:
            group = config.kv_cache_groups[group_id]
            if (run := _head_major_run(config, group)) is not None:
                # A head-major (BHLNC) row holds each head of every layer back
                # to back, so the group's heads are one region of the object.
                block_bytes = run * group.kernel_rows_per_block
                regions.append(
                    KVRegion(
                        layer_name=group.layer_names[0],
                        group_id=group_id,
                        block_stride=group.row_bytes,
                        block_bytes=block_bytes,
                        object_offset=offset,
                        block_size=group.kv_cache_spec.block_size,
                    )
                )
                offset += block_bytes
                continue
            for layer_name, spec in _layer_specs(group).items():
                regions.append(
                    KVRegion(
                        layer_name=layer_name,
                        group_id=group_id,
                        block_stride=group.row_bytes,
                        block_bytes=spec.page_size_bytes,
                        object_offset=offset,
                        block_size=spec.block_size,
                    )
                )
                offset += spec.page_size_bytes
        return cls(regions, group_geometry=group_geometry)

    @property
    def object_size(self) -> int:
        return sum(region.block_bytes for region in self.regions)

    def describe(
        self,
        topology: RankTopology,
        layout_format: str = "rank_local",
    ) -> KVLayoutDescriptor:
        return KVLayoutDescriptor(
            regions=self.regions,
            topology=topology,
            layout_format=layout_format,
        )

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        """Capture stable base addresses for the planned KV regions."""
        required = {region.layer_name for region in self.regions}
        missing = required - kv_caches.keys()
        if missing:
            raise ValueError(f"missing KV cache layers: {sorted(missing)}")
        addresses: dict[str, int] = {}
        for region in self.regions:
            if region.layer_name in addresses:
                continue
            cache = kv_caches[region.layer_name]
            if cache.ndim == 0:
                raise ValueError(f"KV cache {region.layer_name!r} must be a tensor")
            row_bytes = cache.stride(0) * cache.element_size()
            if row_bytes != region.block_stride:
                raise ValueError(
                    f"KV row stride mismatch for {region.layer_name!r}: "
                    f"config={region.block_stride}, tensor={row_bytes}"
                )
            addresses[region.layer_name] = cache.data_ptr()
        self._base_addresses = addresses

    def plan_registered_block(
        self,
        key: str,
        block_id: int,
        *,
        group_id: int | None = None,
        request_id: str | None = None,
    ) -> BlockTransferPlan:
        if not self._base_addresses:
            raise RuntimeError("KV caches must be registered before planning")
        return self.plan_for_block(
            key,
            block_id,
            self._base_addresses,
            group_id=group_id,
            request_id=request_id,
        )

    def plan_for_block(
        self,
        key: str,
        block_id: int,
        base_addresses: dict[str, int],
        *,
        group_id: int | None = None,
        request_id: str | None = None,
    ) -> BlockTransferPlan:
        """Lower one GPU block into a deterministic scatter/gather plan."""
        return BlockTransferPlan(
            key=key,
            block_id=block_id,
            ranges=self._ranges_for_block(block_id, base_addresses, group_id),
            request_id=request_id,
            group_id=group_id,
        )

    def materialize(self, plan: BlockTransferPlan) -> BlockTransferPlan:
        if not self._base_addresses:
            raise RuntimeError("KV caches must be registered before planning")
        return replace(
            plan,
            ranges=self._ranges_for_block(
                plan.block_id, self._base_addresses, plan.group_id
            ),
        )

    def _ranges_for_block(
        self,
        block_id: int,
        base_addresses: dict[str, int],
        group_id: int | None,
    ) -> tuple[KVRange, ...]:
        if block_id < 0:
            raise ValueError("block_id must be non-negative")
        regions = (
            self.regions
            if group_id is None
            else self._regions_by_group.get(group_id, ())
        )
        if group_id is not None and not regions:
            raise ValueError(f"KV layout does not contain cache group {group_id}")
        ranges: list[KVRange] = []
        object_offset = 0
        for region in regions:
            try:
                base_address = base_addresses[region.layer_name]
            except KeyError as exc:
                raise ValueError(
                    f"missing base address for layer {region.layer_name!r}"
                ) from exc
            geometry = self.group_geometry[region.group_id]
            rows = geometry.kernel_rows_per_block
            row_bytes = geometry.row_bytes
            part_bytes = region.block_bytes // rows
            block_address = base_address + block_id * geometry.block_bytes
            block_offset = (
                object_offset if group_id is not None else region.object_offset
            )
            for row in range(rows):
                ranges.append(
                    KVRange(
                        layer_name=region.layer_name,
                        group_id=region.group_id,
                        block_id=block_id,
                        base_address=block_address + row * row_bytes,
                        stride=row_bytes,
                        length=part_bytes,
                        object_offset=block_offset + row * part_bytes,
                    )
                )
            object_offset += region.block_bytes
        return tuple(ranges)


class KVRange(NamedTuple):
    """A source/destination range in a runtime transfer."""

    layer_name: str
    group_id: int
    block_id: int
    base_address: int
    stride: int
    length: int
    object_offset: int


@dataclass(frozen=True)
class BlockTransferPlan:
    """One complete logical block transfer."""

    key: str
    block_id: int
    ranges: tuple[KVRange, ...] = ()
    request_id: str | None = None
    group_id: int | None = None

    @property
    def group_block_id(self) -> tuple[int, int]:
        """The (group_id, block_id) pair; scheduler-built plans set group_id."""
        assert self.group_id is not None
        return self.group_id, self.block_id


@dataclass
class LoadSpec:
    """Scheduler decision for one externally loaded prefix."""

    local_tokens: int
    external_tokens: int

    @property
    def num_tokens_to_load(self) -> int:
        return max(self.external_tokens - self.local_tokens, 0)


@dataclass
class RequestTracker:
    """Core request state shared by every runtime mode."""

    saved_tokens: int = 0
    block_ids: tuple[list[int], ...] = ()
    load_spec: LoadSpec | None = None

    def reset(self) -> None:
        self.saved_tokens = 0
        self.block_ids = ()
        self.load_spec = None

    def update_blocks(self, block_ids: tuple[list[int], ...]) -> None:
        if not self.block_ids:
            self.block_ids = tuple(list(group) for group in block_ids)
            return
        if len(self.block_ids) != len(block_ids):
            raise ValueError("KV cache group count changed for a request")
        for current, new in zip(self.block_ids, block_ids, strict=True):
            if new[: len(current)] == current:
                current.extend(new[len(current) :])
            elif not new or current[-len(new) :] == new:
                continue
            else:
                current.extend(new)

    def replace_blocks(self, block_ids: tuple[list[int], ...]) -> None:
        """Replace stale pre-preemption tables with resumed full tables."""
        self.block_ids = tuple(list(group) for group in block_ids)
        self.saved_tokens = 0

    def mark_saved(self, token_count: int, block_size: int) -> int:
        """Advance the save watermark only to a complete block boundary."""
        complete_tokens = token_count // block_size * block_size
        if complete_tokens < self.saved_tokens:
            return self.saved_tokens
        self.saved_tokens = complete_tokens
        return complete_tokens


class TransferJobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TransferJobState:
    """Per-key completion state; one failed key does not fail its siblings."""

    plans: tuple[BlockTransferPlan, ...]
    status: TransferJobStatus = TransferJobStatus.PENDING
    completed_keys: set[str] = field(default_factory=set)
    failed_keys: set[str] = field(default_factory=set)
    error: str | None = None

    @property
    def keys(self) -> Sequence[str]:
        return tuple(plan.key for plan in self.plans)

    def start(self) -> None:
        if self.status != TransferJobStatus.PENDING:
            raise RuntimeError(f"cannot start job in state {self.status}")
        self.status = TransferJobStatus.RUNNING

    def complete(self, keys: Sequence[str] = ()) -> None:
        self.completed_keys.update(keys or self.keys)
        self._finish_if_done()

    def fail(self, keys: Sequence[str], error: str | None = None) -> None:
        self.failed_keys.update(keys)
        self.error = error
        self._finish_if_done()

    @property
    def failed_block_ids(self) -> set[tuple[int, int]]:
        return {
            plan.group_block_id for plan in self.plans if plan.key in self.failed_keys
        }

    def _finish_if_done(self) -> None:
        keys = set(self.keys)
        if self.completed_keys | self.failed_keys >= keys:
            self.status = (
                TransferJobStatus.FAILED
                if self.failed_keys
                else TransferJobStatus.COMPLETED
            )


@dataclass
class UMBPConnectorMetadata(KVConnectorMetadata):
    """Scheduler-to-worker metadata for one engine step."""

    async_load: bool = False
    store_plans: list[BlockTransferPlan] = field(default_factory=list)
    load_requests: dict[str, list[BlockTransferPlan]] = field(default_factory=dict)
    store_requests: dict[str, list[BlockTransferPlan]] = field(default_factory=dict)
    preempted_request_ids: set[str] = field(default_factory=set)
    store_event: int = -1


@dataclass
class StoreEventResult:
    """Number of workers that finished one store event."""

    completed_workers: int = 0

    def merge(self, other: StoreEventResult) -> None:
        self.completed_workers += other.completed_workers


@dataclass
class UMBPConnectorWorkerMetadata(KVConnectorWorkerMetadata):
    """Worker completion metadata aggregated across ranks."""

    store_events: dict[int, StoreEventResult] = field(default_factory=dict)

    def aggregate(
        self, other: KVConnectorWorkerMetadata
    ) -> UMBPConnectorWorkerMetadata:
        if not isinstance(other, UMBPConnectorWorkerMetadata):
            raise TypeError("cannot aggregate incompatible UMBP worker metadata")
        for event, result in other.store_events.items():
            self.store_events.setdefault(event, StoreEventResult()).merge(result)
        return self
