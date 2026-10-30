# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Base worker-side logic for the NIXL connector."""

import contextlib
import itertools
import logging
import math
import os
import queue
import threading
import time
import uuid
from collections import defaultdict
from collections.abc import Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, cast

import msgspec
import numpy as np
import regex as re
import torch
import zmq

from vllm.distributed.kv_transfer.kv_connector.utils import (
    BlockIds,
    EngineId,
    EngineTransferInfo,
    TransferTopology,
    get_current_attn_backends,
    kv_postprocess_layout_on_receive,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    CopyBlocksOp,
    KVConnectorTransferResults,
)
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    GET_META_MSG,
    NixlAgentMetadata,
    NixlConnectorMetadata,
    NixlHandshakePayload,
    ReqId,
    ReqMeta,
    TransferHandle,
    compute_nixl_compatibility_hash,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.stats import (
    NixlKVConnectorStats,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.tp_mapping import (
    TPMapping,
    _is_attention_spec,
    _is_ssm_spec,
    compute_tp_mapping,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.utils import (
    _NIXL_SUPPORTED_DEVICE,
)
from vllm.distributed.kv_transfer.kv_connector.v1.ssm_conv_transfer_utils import (
    MambaConvSplitInfo,
    derive_mamba_conv_split,
)
from vllm.distributed.nixl_utils import (
    NixlWrapper,
    check_nixl_version,
    nixl_agent_config,
)
from vllm.distributed.parallel_state import (
    get_pcp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.gpu_sync_debug import gpu_sync_allowed
from vllm.utils.math_utils import cdiv
from vllm.utils.network_utils import make_zmq_path, zmq_socket_ctx
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    CircularBufferSpec,
    FullAttentionSpec,
    GroupGeometry,
    KVCacheLayout,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)

# Re-handshake delay bounds for a pull peer whose local metadata keeps getting
# invalidated, eg due to a persistent RDMA fault.
_INVALID_PEER_MIN_BACKOFF_S = 1.0
_INVALID_PEER_MAX_BACKOFF_S = 60.0


def _region_sort_key(layer_name: str) -> tuple[tuple[int, int | str], ...]:
    """Sort transfer regions in model-layer order, then by cache name."""
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part)
        for part in re.split(r"(\d+)", layer_name)
    )


class NixlBaseConnectorWorker:
    """Base implementation of Worker side methods shared by pull and push."""

    # Transfer mode included in the NIXL compatibility hash so that a push
    # (WRITE) connector and a pull (READ) connector never handshake together.
    # Overridden by NixlPushConnectorWorker.
    _TRANSFER_MODE: str = "pull"

    # Layer-name routing is supported only by NixlPushConnector for HMA or
    # packed KV caches under PP.
    _supports_pp_hma = False

    def _compute_desc_ids(
        self,
        block_ids: BlockIds,
        region_num_blocks: list[int],
        region_group_ids: list[int],
    ) -> np.ndarray:
        """Descriptor ids for per-group kernel-row regions.

        ``block_ids`` holds kernel block ids per group (block b expanded to
        rows b*R .. b*R+R-1). The descriptor space is FA
        regions in region order, then each SSM region's conv sub-projection
        runs plus its SSM run, then the PLE run — mirroring ``_build_fa_local``
        and ``_build_mamba_local``.
        """
        if self._transfer_layer_group_ids:
            # Layer-name routing: per-layer regions in local layer order.
            offsets = np.cumsum([0, *region_num_blocks[:-1]])
            return np.concatenate(
                [
                    np.asarray(block_ids[g], dtype=np.int64) + offsets[k]
                    for k, g in enumerate(self._transfer_layer_group_ids)
                ]
            )

        ssm_positions = set(self._ssm_region_indices)
        ple_position = self._ple_region_index
        num_sub_descs = 0
        if self._has_mamba:
            assert self._conv_decomp is not None
            num_sub_descs = len(self._conv_decomp.local_conv_offsets) + 1
        region_offsets = [0] * len(region_num_blocks)
        pos = 0
        fa_positions = []
        for i in range(len(region_num_blocks)):
            if i in ssm_positions or i == ple_position:
                continue
            region_offsets[i] = pos
            pos += region_num_blocks[i]
            fa_positions.append(i)
        for i in sorted(ssm_positions):
            region_offsets[i] = pos
            pos += num_sub_descs * region_num_blocks[i]
        if ple_position is not None:
            region_offsets[ple_position] = pos

        all_descs: list[np.ndarray] = []
        for group_id, group in enumerate(block_ids):
            if not group:
                continue
            kernel_ids = np.asarray(group, dtype=np.int64)
            # Region-mapped reads (pure MLA with a different remote grouping)
            # index block ids per region, not per group; they never carry SSM
            # state, so only look the spec up for models that have any.
            if self._has_mamba and _is_ssm_spec(self._group_spec_types[group_id]):
                if ple_position is not None and group_id == self._ple_group_index:
                    all_descs.append(region_offsets[ple_position] + kernel_ids)
                    continue
                for i in sorted(ssm_positions):
                    if region_group_ids[i] != group_id:
                        continue
                    count = region_num_blocks[i]
                    all_descs.append(
                        np.concatenate(
                            [
                                region_offsets[i] + sub * count + kernel_ids
                                for sub in range(num_sub_descs)
                            ]
                        )
                    )
            else:
                all_descs.append(
                    np.concatenate(
                        [
                            region_offsets[i] + kernel_ids
                            for i in fa_positions
                            if region_group_ids[i] == group_id
                        ]
                    )
                )
        if not all_descs:
            return np.array([], dtype=np.int64)
        return np.concatenate(all_descs)

    def _fa_region_indices(self) -> list[int]:
        """Regions holding attention data (all but the SSM state regions)."""
        ssm = set(self._ssm_region_indices)
        if self._ple_region_index is not None:
            ssm.add(self._ple_region_index)
        return [i for i in range(self.num_regions) if i not in ssm]

    def _build_local_splits_from_plan(
        self,
        plan: TPMapping,
        src_blocks_data: np.ndarray,
        num_fa_descs: int,
    ) -> Iterator[np.ndarray]:
        """Build split handle data for P_TP > D_TP scenario.

        num_fa_descs is the boundary between FA and SSM descriptors.
        Split counts are derived from source_ranks_per_group lengths.
        FA uses rank_to_attention_slot for the slot offset;
        SSM uses the rank's positional index.
        Each Nx5 strided desc covers one region, so slicing it slices every block.
        """
        fa_idx = next(
            i for i, t in enumerate(self._group_spec_types) if _is_attention_spec(t)
        )
        fa_num_splits = len(plan.source_ranks_per_group[fa_idx])

        starts = self._strided_desc_starts(src_blocks_data)
        num_descs = int(src_blocks_data[:, 4].sum())
        has_ssm_descs = num_fa_descs < num_descs
        ssm_idx = next(
            (i for i, t in enumerate(self._group_spec_types) if _is_ssm_spec(t)),
            None,
        )
        ssm_num_splits = (
            len(plan.source_ranks_per_group[ssm_idx])
            if has_ssm_descs and ssm_idx is not None
            else 0
        )
        sharded_desc_end = num_descs - (
            self.region_num_blocks[self._ple_region_index]
            if self._ple_region_index is not None
            else 0
        )
        assert num_fa_descs <= sharded_desc_end

        # REPLICATE (MLA) descs are read whole on every rank; SPLIT (full-attn)
        # descs take this rank's head slice. SSM descs before the PLE pages are
        # sharded by the rank's positional index; PLE pages are copied whole.
        is_fa = starts < num_fa_descs
        fa_replicated = self._fa_desc_replicated(starts[is_fa], num_fa_descs)
        fa_split = is_fa.copy()
        fa_split[is_fa] = ~fa_replicated
        ssm_split = ~is_fa & (starts < sharded_desc_end)

        for p_idx, p_rank in enumerate(plan.all_source_ranks):
            fa_slot = plan.rank_to_attention_slot.get(p_rank, 0)
            handle = src_blocks_data.copy()
            handle[fa_split, 1] //= fa_num_splits
            handle[fa_split, 0] += fa_slot * handle[fa_split, 1]
            if ssm_num_splits:
                handle[ssm_split, 1] //= ssm_num_splits
                handle[ssm_split, 0] += p_idx * handle[ssm_split, 1]
            yield handle

    def _needs_split_local_xfer_handles(self, tp_ratio: int, plan: TPMapping) -> bool:
        """Whether reads need per-source slices of the local KV region.

        Pure MLA attention is replicated across TP ranks and writes the whole
        local region. Multiple physical remote workers may still participate
        because DCP assigns them disjoint blocks, but that does not require
        splitting the local region. Hybrid MLA+SSM is different: its mapping
        contains multiple source ranks for the sharded SSM state.
        """
        return tp_ratio < 0 and (not self.use_mla or len(plan.all_source_ranks) > 1)

    def _fa_desc_replicated(
        self,
        desc_starts: np.ndarray,
        num_fa_descs: int,
    ) -> np.ndarray:
        """Replicate flag per FA strided desc, given each one's flat block index.

        _build_fa_local emits regions in order, with ``region_num_blocks[i]``
        blocks per region. A strided desc never spans regions, so the region
        owning its first block decides.
        """
        assert self.transfer_topo is not None
        if len(self.block_len_per_layer) == 0 or self.num_regions == 0:
            return np.zeros(len(desc_starts), dtype=bool)
        if self._transfer_layer_region_indices:
            region_indices: Iterable[int] = self._transfer_layer_region_indices
        else:
            region_indices = self._fa_region_indices()
        region_ends = np.cumsum([self.region_num_blocks[i] for i in region_indices])
        assert region_ends[-1] == num_fa_descs, (
            f"FA region blocks {region_ends[-1]} != num_fa_descs {num_fa_descs}"
        )
        replicated = np.array([self._is_region_replicated(i) for i in region_indices])
        return replicated[np.searchsorted(region_ends, desc_starts, side="right")]

    def _is_region_replicated(self, region_idx: int) -> bool:
        """Whether region ``region_idx`` is transferred REPLICATE vs SPLIT.

        REPLICATE (MLA): identical on every rank, whole block read from one
        rank at offset 0, key-only. SPLIT (full-attn): head-sharded across TP.
        Defaults to SPLIT when the per-region map is unset (e.g. tests that set
        block_len_per_layer without register_kv_caches).
        """
        return region_idx < len(self._region_is_mla) and self._region_is_mla[region_idx]

    def _resolve_head_sharded_draft_kv_heads(self) -> int | None:
        """Total KV heads of a non-MLA draft under an MLA target, else None.

        Such a draft's KV is head-sharded while the target's is replicated."""
        if not self.use_mla or self._has_mamba or self._is_csa_linear:
            return None
        spec_config = self.vllm_config.speculative_config
        draft_config = spec_config.draft_model_config if spec_config else None
        if draft_config is None or draft_config.use_mla:
            return None
        return draft_config.get_total_num_kv_heads()

    def _get_head_sharded_draft_regions(self) -> set[int]:
        heads = self._head_sharded_draft_kv_heads
        if heads is None:
            return set()
        spec_config = self.vllm_config.speculative_config
        assert spec_config is not None and spec_config.draft_model_config is not None
        target_layers = self.model_config.get_total_num_hidden_layers()
        draft_layers = spec_config.draft_model_config.get_total_num_hidden_layers()
        regions = set[int]()
        for i, name in enumerate(self.region_names):
            if self._is_region_replicated(i):
                continue
            # DFlash numbers draft layers after the target instead of prefixing them.
            if not name.startswith("draft_model."):
                match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", name)
                if match is None or not (
                    target_layers <= int(match.group(1)) < target_layers + draft_layers
                ):
                    continue
            spec = self._layer_specs[name]
            assert isinstance(spec, AttentionSpec) and spec.num_kv_heads <= heads, (
                f"Draft region {name!r} is inconsistent with {heads} total KV heads"
            )
            regions.add(i)
        return regions

    def _set_region_layers(self, region_layers: list[list[str]]) -> None:
        layer_names = [layer_name for region in region_layers for layer_name in region]
        assert len(layer_names) == len(set(layer_names)), (
            "A KV cache layer spans multiple NIXL regions"
        )
        self.region_members: list[list[str]] = region_layers
        if not self._requires_layer_name_routing():
            return
        group_by_layer = self.kv_cache_config.transfer_group_index_by_layer
        ungrouped = [name for name in layer_names if name not in group_by_layer]
        assert not ungrouped, f"KV cache layers outside any local group: {ungrouped}"
        self._transfer_layer_names = tuple(layer_names)
        self._transfer_layer_region_indices = tuple(
            region_idx
            for region_idx, region in enumerate(region_layers)
            for _ in region
        )
        self._transfer_layer_group_ids = tuple(
            group_by_layer[name] for name in layer_names
        )

    def _tracks_region_layers(self) -> bool:
        """Whether regions advertise their member layer names.

        Push workers do so for HMA and packed layouts, so a PP producer can
        match layers by name instead of by region index.
        """
        return (
            self._supports_pp_hma
            and (self._is_hma_required or self._has_packed_cache)
            and not self._has_mamba
        )

    def _requires_layer_name_routing(self) -> bool:
        """Whether PP push must match HMA/packed layers by name, not region index."""
        return self.pp_size > 1 and self._tracks_region_layers()

    def _align_remote_regions_by_layer(
        self, nixl_agent_meta: NixlAgentMetadata
    ) -> None:
        """Select remote regions for this PP stage in local layer order."""
        remote_region_layers = nixl_agent_meta.region_members
        # Region-index routing here would silently transfer stale KV.
        assert remote_region_layers, "Remote advertised no region_members"
        assert (
            len(nixl_agent_meta.kv_caches_base_addr)
            == len(nixl_agent_meta.block_lens)
            == len(nixl_agent_meta.block_strides)
            == len(remote_region_layers)
        ), "Remote region metadata lengths disagree"
        assert all(
            values is None or len(values) == len(remote_region_layers)
            for values in (
                nixl_agent_meta.region_num_blocks,
                nixl_agent_meta.region_group_ids,
                nixl_agent_meta.region_names,
                nixl_agent_meta.region_mem_types,
            )
        ), "Remote region metadata lengths disagree"

        remote_region_by_layer: dict[str, int] = {}
        for region_idx, layer_names in enumerate(remote_region_layers):
            for layer_name in layer_names:
                assert layer_name not in remote_region_by_layer, (
                    f"Remote advertised layer {layer_name!r} in multiple regions"
                )
                remote_region_by_layer[layer_name] = region_idx

        missing = [
            name
            for name in self._transfer_layer_names
            if name not in remote_region_by_layer
        ]
        assert not missing, f"Remote is missing locally owned layers: {missing}"

        packed_regions = {
            remote_region_by_layer[name]
            for name in nixl_agent_meta.packed_member_layouts
        }
        remote_regions = [
            remote_region_by_layer[name] for name in self._transfer_layer_names
        ]
        nixl_agent_meta.kv_caches_base_addr = [
            nixl_agent_meta.kv_caches_base_addr[i] for i in remote_regions
        ]
        nixl_agent_meta.block_lens = [
            nixl_agent_meta.block_lens[i] for i in remote_regions
        ]
        nixl_agent_meta.block_strides = [
            nixl_agent_meta.block_strides[i] for i in remote_regions
        ]
        if nixl_agent_meta.region_num_blocks is not None:
            nixl_agent_meta.region_num_blocks = [
                nixl_agent_meta.region_num_blocks[i] for i in remote_regions
            ]
        if nixl_agent_meta.region_group_ids is not None:
            nixl_agent_meta.region_group_ids = [
                nixl_agent_meta.region_group_ids[i] for i in remote_regions
            ]
        if nixl_agent_meta.region_mem_types is not None:
            nixl_agent_meta.region_mem_types = [
                nixl_agent_meta.region_mem_types[i] for i in remote_regions
            ]
        if nixl_agent_meta.region_names is not None:
            nixl_agent_meta.region_names = list(self._transfer_layer_names)
        if nixl_agent_meta.packed_member_layouts:
            for i, layer_name in enumerate(self._transfer_layer_names):
                if remote_regions[i] not in packed_regions:
                    continue
                assert layer_name in nixl_agent_meta.packed_member_layouts, (
                    f"Remote packed layer {layer_name!r} has no layout"
                )
                offset, page_size = nixl_agent_meta.packed_member_layouts[layer_name]
                assert (
                    0 <= offset < offset + page_size <= nixl_agent_meta.block_lens[i]
                ), f"Remote packed layer {layer_name!r} escapes its block"
                local_region = self._transfer_layer_region_indices[i]
                assert page_size == self.block_len_per_layer[local_region], (
                    f"Packed MLA page sizes must match for layer {layer_name!r}"
                )
                nixl_agent_meta.kv_caches_base_addr[i] += offset
                nixl_agent_meta.block_lens[i] = page_size
            nixl_agent_meta.packed_member_layouts = {}
        # One layer per region keeps a second alignment pass a no-op.
        nixl_agent_meta.region_members = [[name] for name in self._transfer_layer_names]

    def _split_remote_packed_rows(self, nixl_agent_meta: NixlAgentMetadata) -> None:
        """Map the remote's whole-row region onto our per-layer regions.

        E.g. a BLHNC prefiller registers one region whose block is a full row
        holding every layer's page, while a HiSparse decoder registers one
        region per layer. Using the advertised `packed_member_layouts`, the
        remote row is rewritten in place as one virtual region per local layer:

            remote row b:  [ L0 page | L1 page | ... ]   base=B, stride=S
                             ^off0     ^off1

            remote metadata, before:  [(B, S, S)]               # (base, len, stride)
            remote metadata, after:   [(B + off0, page0, S),    # L0
                                       (B + off1, page1, S)]    # L1

        Remote regions outside the row (e.g. a separate indexer) pass through.
        """
        layouts = nixl_agent_meta.packed_member_layouts
        remote_names = nixl_agent_meta.region_names
        assert remote_names is not None, "Remote advertised no region_names"
        (row,) = [i for i, name in enumerate(remote_names) if name in layouts]
        remote_index = {name: i for i, name in enumerate(remote_names)}
        regions = [
            (row, *layouts[name])
            if name in layouts
            else (remote_index[name], 0, nixl_agent_meta.block_lens[remote_index[name]])
            for name in self.region_names
        ]
        nixl_agent_meta.kv_caches_base_addr = [
            nixl_agent_meta.kv_caches_base_addr[i] + offset for i, offset, _ in regions
        ]
        nixl_agent_meta.block_lens = [page_size for _, _, page_size in regions]
        nixl_agent_meta.block_strides = [
            nixl_agent_meta.block_strides[i] for i, _, _ in regions
        ]
        for attr in ("region_num_blocks", "region_group_ids", "region_mem_types"):
            values = getattr(nixl_agent_meta, attr)
            if values is not None:
                setattr(nixl_agent_meta, attr, [values[i] for i, _, _ in regions])
        nixl_agent_meta.region_names = list(self.region_names)
        nixl_agent_meta.packed_member_layouts = {}

    def __init__(
        self,
        vllm_config: "VllmConfig",
        engine_id: str,
        kv_cache_config: "KVCacheConfig",
    ):
        nixl_wrapper_cls = NixlWrapper
        if nixl_wrapper_cls is None:
            logger.error("NIXL is not available")
            raise RuntimeError("NIXL is not available")
        # Descriptor lists are prepared in NIXL's strided (Nx5) form.
        check_nixl_version("1.5.0")
        logger.info("Initializing NIXL wrapper")
        logger.info("Initializing NIXL worker %s", engine_id)

        # Config.
        self.vllm_config = vllm_config

        if vllm_config.kv_transfer_config is None:
            raise ValueError("kv_transfer_config must be set for NixlConnector")
        self.kv_transfer_config = vllm_config.kv_transfer_config

        self.nixl_backends = vllm_config.kv_transfer_config.get_from_extra_config(
            "backends", ["UCX"]
        )
        kv_lease_duration: int = vllm_config.kv_transfer_config.get_from_extra_config(
            "kv_lease_duration", 30
        )
        # NOTE (NickLucche): For now we use a hardcoded value for a simpler interface.
        self._lease_extension = kv_lease_duration * 2 // 3

        self._bidirectional_kv_xfer_enabled: bool = (
            vllm_config.kv_transfer_config.get_from_extra_config(
                "bidirectional_kv_xfer", False
            )
        )

        self.kv_cache_config = kv_cache_config
        transfer_block_sizes = [
            group.kv_cache_spec.block_size
            for group in kv_cache_config.transfer_groups
            if type(group.kv_cache_spec) not in (MambaSpec, CircularBufferSpec)
        ]
        self.block_size = (
            math.lcm(*transfer_block_sizes)
            if transfer_block_sizes
            else cast(int, vllm_config.cache_config.block_size)
        )
        self._layer_specs: dict[str, KVCacheSpec] = {
            layer_name: group.kv_cache_spec
            for group in kv_cache_config.transfer_groups
            for layer_name in group.layer_names
        }
        self._is_hma_required = (
            not vllm_config.scheduler_config.disable_hybrid_kv_cache_manager
            and any(
                not isinstance(spec, FullAttentionSpec)
                for spec in self._layer_specs.values()
            )
        )

        # ---- Model state (derived from model config) ----
        mamba_ssm_size = (0, 0)
        # Conv state sub-projection decomposition (None when no Mamba).
        # The transfer requires DS (dim, state_len) conv layout so that
        # conv sub-projections are contiguous in memory.
        self._conv_decomp: MambaConvSplitInfo | None = None
        self._has_mamba = any(
            isinstance(spec, MambaSpec) for spec in self._layer_specs.values()
        )
        self._is_csa_linear = any(
            type(group.kv_cache_spec) is CircularBufferSpec
            for group in kv_cache_config.kv_cache_groups
        )
        self._ple_group_index: int | None = None
        if self._is_csa_linear:
            # Transfer-group space, matching _compute_desc_ids.
            ple_groups = [
                (index, group)
                for index, group in enumerate(kv_cache_config.transfer_groups)
                if group.layer_names
                and isinstance(group.kv_cache_spec, MambaSpec)
                and group.kv_cache_spec.tp_replicated
            ]
            if len(ple_groups) > 1 or any(
                len(group.layer_names) != 1 for _, group in ple_groups
            ):
                raise ValueError(
                    "CSA-linear NIXL requires at most one PLE cache owner."
                )
            self._ple_group_index = ple_groups[0][0] if ple_groups else None

        if self._has_mamba:
            assert self._is_hma_required
            from vllm.model_executor.layers.mamba.mamba_utils import (
                is_conv_state_dim_first,
            )

            assert is_conv_state_dim_first(), (
                "3-read Mamba conv transfer requires DS conv state layout. "
                "Set VLLM_SSM_CONV_STATE_LAYOUT=DS"
            )
            mamba_spec = next(
                spec
                for spec in self._layer_specs.values()
                if isinstance(spec, MambaSpec)
                and (not self._is_csa_linear or not spec.tp_replicated)
            )
            self._conv_decomp = derive_mamba_conv_split(
                mamba_spec,
                vllm_config.parallel_config.tensor_parallel_size,
            )
            mamba_ssm_size = self._conv_decomp.ssm_sizes
        self._mamba_ssm_size = mamba_ssm_size

        # Agent.
        non_ucx_backends = [b for b in self.nixl_backends if b != "UCX"]
        # Configure NIXL num_threads to avoid UAR exhaustion on Mellanox NICs.
        # Each UCX thread allocates UARs (doorbell pages) via DevX, and
        # excessive NIXL UAR usage can exhaust NIC UAR space. This can cause
        # components like NVSHMEM (used by DeepEP kernels) to fail during RDMA
        # initialization with "mlx5dv_devx_alloc_uar" errors.
        # Ref: https://network.nvidia.com/files/doc-2020/ethernet-adapters-programming-manual.pdf#page=63
        num_threads = vllm_config.kv_transfer_config.get_from_extra_config(
            "num_threads", 4
        )
        if nixl_agent_config is None:
            config = None
        else:
            # Enable telemetry by default for NIXL 0.7.1 and above.
            config = (
                nixl_agent_config(backends=self.nixl_backends, capture_telemetry=True)
                if len(non_ucx_backends) > 0
                else nixl_agent_config(num_threads=num_threads, capture_telemetry=True)
            )

        sleep_mode = vllm_config.model_config.enable_sleep_mode and config is not None
        if sleep_mode:
            config = nixl_agent_config(backends=[], capture_telemetry=True)
        self.nixl_wrapper = nixl_wrapper_cls(str(uuid.uuid4()), config)
        if sleep_mode:
            # Keep slept pages unpinned, for this agent only; UCX_* env vars win.
            self.nixl_wrapper.create_backend(
                "UCX",
                {
                    "num_threads": str(num_threads),
                    "engine_config": "RCACHE_ENABLE=n,TLS=^cuda_ipc",
                },
            )
        # Map of engine_id -> {(pp_rank, tp_rank): agent_name, ...}.
        # non-PP remote uses pp_rank 0, i.e. (0, tp_rank).
        self._remote_agents: dict[EngineId, dict[tuple[int, int], str]] = defaultdict(
            dict
        )
        # Failed reads need a local metadata probe; invalid peers wait for drain.
        self._failed_remote_engines: set[EngineId] = set()
        self._invalid_remote_engines: set[EngineId] = set()
        # Invalid engine -> (backoff, earliest release time). Kept across release
        # so that repeat invalidations back off.
        self._invalid_engine_backoff: dict[EngineId, tuple[float, float]] = {}
        # Map of engine_id -> clock offset.
        self._engine_clock_offset: dict[EngineId, float] = {}

        # Metadata.
        self.engine_id: EngineId = engine_id
        self.tp_rank = get_tensor_model_parallel_rank()
        self.world_size = get_tensor_model_parallel_world_size()
        self.pcp_size = vllm_config.parallel_config.prefill_context_parallel_size
        self.dcp_size = vllm_config.parallel_config.decode_context_parallel_size
        self.pcp_rank = get_pcp_group().rank_in_group if self.pcp_size > 1 else 0

        # TP1 PCP+DCP owns distinct KV shards; replicated PCP uses rank zero.
        self.pcp_dcp_sharded = self.pcp_size > 1 and self.dcp_size > 1
        self.transfer_tp_size = (
            self.pcp_size if self.pcp_dcp_sharded else self.world_size
        )
        self.transfer_tp_rank = self.pcp_rank if self.pcp_dcp_sharded else self.tp_rank

        # MLA is either fully replicated or fully sharded across transfer ranks.
        self.dcp_rank = self.transfer_tp_rank % self.dcp_size

        self.enable_permute_local_kv = False
        self.enable_heterogeneous_attn_post_process = False

        # KV Caches and nixl tracking data.
        self.device_type = current_platform.device_type
        self.kv_buffer_device: str = vllm_config.kv_transfer_config.kv_buffer_device
        if self.device_type not in _NIXL_SUPPORTED_DEVICE:
            raise RuntimeError(f"{self.device_type} is not supported.")
        elif self.kv_buffer_device not in _NIXL_SUPPORTED_DEVICE[self.device_type]:
            raise RuntimeError(
                f"{self.device_type} with {self.kv_buffer_device} kv_buffer "
                "is not supported."
            )
        self.device_kv_caches: dict[str, torch.Tensor] = {}

        # cpu kv buffer for xfer
        # used when device memory can not be registered under nixl
        self.host_xfer_buffers: dict[str, torch.Tensor] = {}
        if self.device_type == "cpu":
            self.use_host_buffer = False
        else:
            self.use_host_buffer = self.kv_buffer_device == "cpu"

        # reserve different cores for start_load_kv() from model_forward()
        if self.device_type == "cpu":
            numa_core_list = current_platform.discover_numa_topology()
            # setup one last core in each numa for kv transfer.
            rsv_cores_for_kv = [
                max(each_numa_core_list) for each_numa_core_list in numa_core_list
            ]

            if rsv_cores_for_kv:
                if not hasattr(os, "sched_setaffinity"):
                    raise NotImplementedError(
                        "os.sched_setaffinity is not available on this platform"
                    )
                os.sched_setaffinity(0, rsv_cores_for_kv)

        # support for oot platform which can't register nixl memory
        # type based on kv_buffer_device
        nixl_memory_type = current_platform.get_nixl_memory_type()
        if nixl_memory_type is None:
            if self.kv_buffer_device in ["cuda", "xpu"]:
                nixl_memory_type = "VRAM"
            elif self.kv_buffer_device == "cpu":
                nixl_memory_type = "DRAM"
        if nixl_memory_type is None:
            raise RuntimeError(
                f"{self.device_type} with {self.kv_buffer_device} kv_buffer "
                "is not supported."
            )
        self.nixl_memory_type = nixl_memory_type

        # Note: host xfer buffer ops when use_host_buffer is True
        self.copy_blocks: CopyBlocksOp | None = None

        # Map of engine_id -> kv_caches_base_addr. For TP case, each local
        self.device_id: int = 0
        # Current rank may pull from multiple remote TP workers.
        # EngineId, dict[int, list[int]] -> engine_id, tp_rank, base_addr_for_layer
        self.kv_caches_base_addr = defaultdict[EngineId, dict[int, list[int]]](dict)

        # Number of NIXL regions: one kernel-row region per layer of a
        # head-sharded group, one whole-row region per TP-replicated group.
        self.num_regions = 0
        self.region_mem_types: list[str] = []
        self.region_group_ids: list[int] = []
        self.region_names: list[str] = []
        self.region_num_blocks: list[int] = []
        self._mixed_mem_types = False
        # True on TP ranks that share a HiSparse host pool with TP rank 0 and
        # leave its NIXL registration and DRAM reads to rank 0.
        self._skip_dram_xfer = False
        self._desc_is_dram: np.ndarray | None = None
        self._desc_pos: np.ndarray | None = None
        self._dram_src_handle: int | None = None

        # PP>1 (push mode): this worker holds a contiguous layer slice and
        # transfers into the matching sub-range of a PP=1 remote's regions.
        self.pp_size = vllm_config.parallel_config.pipeline_parallel_size
        if self.pp_size > 1:
            if self._is_hma_required and not self._supports_pp_hma:
                raise NotImplementedError(
                    "NixlConnector (pull) does not support pipeline_parallel_size "
                    "> 1 with hybrid KV cache layouts (HMA); use NixlPushConnector "
                    "for PP + HMA."
                )
            # PP push routes HMA (hybrid) attention layouts by layer name;
            # Mamba/SSM hybrids are not yet supported under PP.
            if self._has_mamba:
                raise NotImplementedError(
                    "NixlPushConnector does not support pipeline_parallel_size > 1 "
                    "with Mamba/SSM hybrid KV cache layouts yet."
                )
            # Decode-side PP is unsupported (completions counted per consumer rank).
            if vllm_config.kv_transfer_config.kv_role == "kv_consumer":
                raise NotImplementedError(
                    "NixlPushConnector consumer (decode) does not support "
                    "pipeline_parallel_size > 1."
                )
        # Keep heartbeat handshakes to a PP-sharded producer notif-only.
        self._hb_handshake_notif_only = False

        # nixl_prepped_dlist_handle of the local KV registration.
        self.src_xfer_handle: int | None = None
        # Populated dynamically during handshake based on remote configuration.
        # Per-source split handles, keyed by tp_ratio.
        self.src_xfer_handles_by_tp_ratio: dict[int, list[int]] = {}
        self._dram_src_handles_by_tp_ratio: dict[int, list[int | None]] = {}
        # Map of engine_id -> {tp_rank: nixl_prepped_dlist_handle (int)}.
        self.dst_xfer_side_handles = defaultdict[EngineId, dict[int, int]](dict)

        self.dst_region_num_blocks: dict[EngineId, list[int]] = {}
        self.dst_region_group_ids: dict[EngineId, list[int]] = {}
        self.dst_region_mem_types: dict[EngineId, list[str]] = {}
        # Per peer, the advertised pool geometry of each transfer group, used
        # to expand the peer's block ids to kernel block ids.
        self._dst_group_geometries: dict[EngineId, list[GroupGeometry]] = {}
        self._registered_descs: list[Any] = []
        self._kv_unregistered = False

        # In progress transfers.
        # [req_id -> list[handle]]
        self._recving_metadata: dict[ReqId, ReqMeta] = {}
        self._recving_transfers = defaultdict[ReqId, list[TransferHandle]](list)
        # Track the expiration time of requests that are waiting to be sent.
        self._reqs_to_send: dict[ReqId, float] = {}
        # Replicated-KV PCP ranks > 0 never transfer; requests they are asked
        # to send are reported finished immediately (see get_finished).
        self._replicated_pcp_done_sending: set[ReqId] = set()
        # Set of requests that have been part of a batch, regardless of status.
        self._reqs_to_process: set[ReqId] = set()

        # Invalid blocks from failed NIXL operations (thread-safe queue).
        # Holds (group_id, block_id) pairs: block ids are unique only within
        # a group.
        self._invalid_block_ids: queue.Queue[set[tuple[int, int]]] = queue.Queue()
        # requests that skipped transfer (handshake or transfer failures)
        # Uses Queue for thread-safe cross-thread coordination with the
        # background handshake thread, matching the _ready_requests pattern.
        self._failed_recv_reqs: queue.Queue[ReqId] = queue.Queue()
        self._recv_failures: set[ReqId] = set()
        self._pending_recv_notifs: dict[ReqId, list[tuple[str, bytes]]] = {}

        # Handshake metadata of this worker for NIXL transfers.
        self.xfer_handshake_metadata: NixlHandshakePayload | None = None
        # Background thread for initializing new NIXL handshakes.
        self._handshake_initiation_executor = ThreadPoolExecutor(
            # NIXL is not guaranteed to be thread-safe, limit 1 worker.
            max_workers=1,
            thread_name_prefix="vllm-nixl-handshake-initiator",
        )
        self._ready_requests = queue.Queue[tuple[ReqId, ReqMeta]]()
        self._handshake_futures: dict[
            EngineId, Future[tuple[dict[tuple[int, int], str], float]]
        ] = {}
        # Protects _handshake_futures and _remote_agents.
        self._handshake_lock = threading.RLock()

        # TTL-based eviction of stale remote engine state.
        self._engine_last_active: dict[EngineId, float] = {}
        self._engine_ttl: float = vllm_config.kv_transfer_config.get_from_extra_config(
            "engine_ttl", 3600.0
        )
        # Map of pull peer (host, port) -> engine_id currently serving it.
        self._engine_by_address: dict[tuple[str, int], EngineId] = {}

        self.model_config = vllm_config.model_config

        self.use_mla = self.model_config.use_mla

        # Get the attention backend from the first layer
        # NOTE (NickLucche) models with multiple backends are not supported yet
        self.attn_backends = get_current_attn_backends(vllm_config)
        self.backend_name = self.attn_backends[0].get_name()

        self.kv_cache_layout = (
            vllm_config.cache_config.get_resolved_kv_cache_layout().name
        )
        self.host_buffer_kv_cache_layout = self.kv_cache_layout
        logger.info(
            "Detected attention backend(s) %s",
            [backend.get_name() for backend in self.attn_backends],
        )
        logger.info("Detected kv cache layout %s", self.kv_cache_layout)

        # lazy initialized in register_kv_caches
        self.compat_hash: str | None = None
        self.transfer_topo: TransferTopology | None = None

        # With heterogeneous TP (or DCP), P must wait for all assigned D
        # workers to finish reading before safely freeing the blocks.
        self.consumer_notification_counts_by_req = defaultdict[ReqId, int](int)
        self.expected_consumer_notifications_by_req: dict[ReqId, int] = {}
        self.xfer_stats = NixlKVConnectorStats()

        self._group_spec_types = tuple(
            type(g.kv_cache_spec) for g in self.kv_cache_config.transfer_groups
        )
        # Per-region MLA flag, 1:1 with block_len_per_layer. True -> REPLICATE
        # (MLA), False -> SPLIT (head-sharded full-attn). Mixed only for models
        # combining both (e.g. GQA main + MLA Eagle-3 draft).
        self._region_is_mla = list[bool]()
        self._head_sharded_draft_regions = set[int]()
        self._ssm_region_indices = list[int]()
        self._ple_region_index: int | None = None
        # Page length of the PLE short-conv state; its region may be shared with
        # pages of a different (smaller) size, so it cannot use the region's.
        self._ple_block_len: int | None = None

        # Enable different block lengths for different layers *only* when MLA is used.
        # This is not used for SSM layers, which use the counterpart `mamba_ssm_size`.
        self.block_len_per_layer = list[int]()

        # Per-region block stride in bytes. Taken from the registered tensor's
        # stride(0) so it stays correct under layouts that interleave layers
        # within a block (BLHNC/BHLNC), where stride > block_len.
        self.block_stride_per_layer = list[int]()

        # Block-outermost layouts (BLHNC/BHLNC) pack every layer's page into one
        # block row of a shared allocation.
        self._has_packed_cache = KVCacheLayout[self.kv_cache_layout].is_block_outermost
        # Region layer membership is populated for HMA and packed layouts.
        self.region_members = []
        self._packed_member_layouts: dict[str, tuple[int, int]] = {}
        # Sorted layer names held by this worker per transfer group
        # (handshake layer-set check; advertised on the wire).
        self._group_layer_names: list[list[str]] = []
        # Local layer order, set by ``_set_region_layers`` only when
        # routing by layer name. Empty means region-index routing.
        self._transfer_layer_names = tuple[str, ...]()
        self._transfer_layer_region_indices = tuple[int, ...]()
        self._transfer_layer_group_ids = tuple[int, ...]()
        # Per-engine TP mappings. Generated during handshake.
        self.tp_mappings: dict[EngineId, TPMapping] = {}
        # GQA draft under an MLA target: its SPLIT regions use a draft TP mapping.
        self._head_sharded_draft_kv_heads = self._resolve_head_sharded_draft_kv_heads()
        self.draft_tp_mappings: dict[EngineId, TPMapping] = {}

        self.enforce_compat_hash = self.kv_transfer_config.get_from_extra_config(
            "enforce_handshake_compat", True
        )

    def _validate_remote_parallel_config(
        self, agent_metadata: NixlAgentMetadata
    ) -> None:
        local_pcp_size = self.pcp_size
        local_dcp_size = self.dcp_size
        remote_pcp_size = agent_metadata.pcp_size
        remote_dcp_size = agent_metadata.dcp_size
        if remote_pcp_size > 1 and remote_dcp_size not in (1, remote_pcp_size):
            raise NotImplementedError(
                "Remote NixlConnector PCP+DCP does not span the full PCP group. "
                f"Remote PCP/DCP={remote_pcp_size}/{remote_dcp_size}."
            )
        if (local_pcp_size > 1 and local_dcp_size == 1 and remote_dcp_size > 1) or (
            remote_pcp_size > 1 and remote_dcp_size == 1 and local_dcp_size > 1
        ):
            raise NotImplementedError(
                "Replicated PCP cannot be paired with a DCP-sharded NIXL peer. "
                f"Local PCP/DCP={local_pcp_size}/{local_dcp_size}; "
                f"remote PCP/DCP={remote_pcp_size}/{remote_dcp_size}."
            )

    def _validate_csa_linear_tp_layout(self, remote_tp_size: int) -> None:
        """Reject P/D pairs whose main-KV pages have different head layouts.

        A rank at TP < total KV heads holds several heads per page, while a
        rank at TP >= total KV heads holds exactly one (replicated beyond
        that). CSA-linear pages are shared tensors that transfer as whole
        block units, so endpoints on opposite sides of that boundary lay
        them out differently and cannot exchange them.
        """
        assert self.transfer_topo is not None
        total_kv_heads = self.transfer_topo.total_num_kv_heads
        if (self.world_size < total_kv_heads) != (remote_tp_size < total_kv_heads):
            raise ValueError(
                "CSA-linear NIXL requires both endpoints on the same side of "
                f"the KV-head sharding boundary ({total_kv_heads} KV heads): "
                f"got local TP {self.world_size}, remote TP {remote_tp_size}."
            )

    def _nixl_handshake(
        self,
        host: str,
        port: int,
        remote_tp_size: int,
        expected_engine_id: str,
        remote_dcp_size: int = 1,
        remote_pp_size: int = 1,
        notif_agents_only: bool = False,
    ) -> tuple[dict[tuple[int, int], str], float]:
        """Do a NIXL handshake with a remote instance."""
        if self._is_csa_linear:
            self._validate_csa_linear_tp_layout(remote_tp_size)

        # the first time we connect to a remote agent.
        # be careful, the handshake happens in a background thread.
        # it does not have an active cuda context until any cuda runtime
        # call is made. when UCX fails to find a valid cuda context, it will
        # disable any cuda ipc communication, essentially disabling any NVLink
        # communication.
        # when we are using device buffers, we need to set the device
        # explicitly to make sure the handshake background thread has a valid
        # cuda context.
        if not self.use_host_buffer:
            current_platform.set_device(self.device_id)

        # When target instance TP > local TP, we need to perform multiple
        # handshakes. Do it in a single background job for simplicity.
        # Regardless, only handshake with the remote TP rank(s) that current
        # local rank will read from. Note that With homogeneous TP,
        # this happens to be the same single rank_i.
        assert self.transfer_topo is not None
        p_remote_ranks = self.transfer_topo.handshake_target_ranks(
            remote_tp_size, remote_dcp_size
        )
        remote_rank_to_agent_name: dict[tuple[int, int], str] = {}
        path = make_zmq_path("tcp", host, port)
        # Clock offset to the peer, estimated from the handshake round-trip.
        # Keep the lowest-RTT sample: hop cost is ~uniform across ranks, so a
        # higher RTT is just noise that skews the midpoint estimate.
        best_rtt = float("inf")
        best_offset: float | None = None

        with zmq_socket_ctx(path, zmq.REQ, bind=False) as sock:
            for remote_pp_rank, remote_rank in itertools.product(
                range(remote_pp_size), p_remote_ranks
            ):
                logger.debug(
                    "Querying metadata on path: %s at remote pp rank %s, tp rank %s",
                    path,
                    remote_pp_rank,
                    remote_rank,
                )

                # Send query for the request.
                msg = msgspec.msgpack.encode(
                    (GET_META_MSG, remote_pp_rank, remote_rank)
                )
                # Set receive timeout to 5 seconds to avoid hanging on dead server
                sock.setsockopt(zmq.RCVTIMEO, 5000)  # milliseconds
                start_time = time.perf_counter()
                sock.send(msg)
                reply_parts = sock.recv_multipart()
                recv_time = time.perf_counter()
                assert len(reply_parts) == 2
                handshake_bytes = reply_parts[0]

                remote_perf = msgspec.msgpack.decode(reply_parts[1])
                rtt = recv_time - start_time
                if rtt < best_rtt:
                    best_rtt = rtt
                    best_offset = remote_perf - (start_time + recv_time) / 2

                # Decode handshake payload to get compatibility hash
                handshake_decoder = msgspec.msgpack.Decoder(NixlHandshakePayload)
                try:
                    handshake_payload = handshake_decoder.decode(handshake_bytes)
                except (msgspec.DecodeError, msgspec.ValidationError) as e:
                    raise RuntimeError(
                        f"Failed to decode NixlHandshakePayload. This likely indicates "
                        f"an incompatibility between connector version. Error: {e}"
                    ) from e

                got_metadata_time = time.perf_counter()
                logger.debug(
                    "NIXL handshake: get metadata took: %s",
                    got_metadata_time - start_time,
                )

                # Check compatibility hash BEFORE decoding agent metadata
                assert self.compat_hash is not None
                if (
                    self.enforce_compat_hash
                    and handshake_payload.compatibility_hash != self.compat_hash
                ):
                    raise RuntimeError(
                        f"NIXL compatibility hash mismatch. "
                        f"Local: {self.compat_hash}, "
                        f"Remote: {handshake_payload.compatibility_hash}. "
                        f"Prefill and decode instances have incompatible "
                        f"configurations. This may be due to: different vLLM versions,"
                        f" models, dtypes, KV cache layouts, attention backends, etc. "
                        f"Both instances must use identical configurations."
                        f"Disable this check using "
                        f'--kv-transfer-config \'{{"kv_connector_extra_config": '
                        f'{{"enforce_handshake_compat": false}}}}\''
                    )

                logger.info(
                    "NIXL compatibility check passed (hash: %s)",
                    handshake_payload.compatibility_hash,
                )

                # Decode agent metadata
                metadata_decoder = msgspec.msgpack.Decoder(NixlAgentMetadata)
                try:
                    metadata = metadata_decoder.decode(
                        handshake_payload.agent_metadata_bytes
                    )
                except (msgspec.DecodeError, msgspec.ValidationError) as e:
                    # This should not happen if hash matched
                    raise RuntimeError(
                        f"Failed to decode NixlAgentMetadata. Error: {e}"
                    ) from e

                self._validate_remote_parallel_config(metadata)

                # Ensure engine id matches.
                if metadata.engine_id != expected_engine_id:
                    raise RuntimeError(
                        f"Remote NIXL agent engine ID mismatch. "
                        f"Expected {expected_engine_id},"
                        f"received {metadata.engine_id}."
                    )

                # Register Remote agent.
                if notif_agents_only:
                    remote_agent_name = self._add_notif_only_remote_agent(
                        metadata, remote_tp_size, metadata.dcp_size
                    )
                else:
                    remote_agent_name = self.add_remote_agent(
                        metadata, remote_rank, remote_tp_size, metadata.dcp_size
                    )
                setup_agent_time = time.perf_counter()
                logger.debug(
                    "NIXL handshake: add agent took: %s (notif_agents_only=%s)",
                    setup_agent_time - got_metadata_time,
                    notif_agents_only,
                )
                remote_ranks = (remote_pp_rank, remote_rank)
                remote_rank_to_agent_name[remote_ranks] = remote_agent_name

        assert best_offset is not None
        return remote_rank_to_agent_name, best_offset

    def _add_notif_only_remote_agent(
        self, metadata: NixlAgentMetadata, remote_tp_size: int, remote_dcp_size: int = 1
    ) -> str:
        """Load a remote agent for notifs only on the push-mode decode side.

        Skips descriptor setup but records engine info for block accounting.
        """
        assert self.transfer_topo is not None
        self.transfer_topo.register_remote_engine(
            metadata.engine_id,
            EngineTransferInfo(
                remote_tp_size=remote_tp_size,
                remote_block_size=metadata.block_size,
                remote_block_len=metadata.block_lens[0],
                remote_dcp_size=remote_dcp_size,
            ),
        )
        return self.nixl_wrapper.add_remote_agent(metadata.agent_metadata)

    def initialize_host_xfer_buffer(self, kv_caches: dict[str, torch.Tensor]) -> None:
        """Initialize transfer buffers in CPU memory.

        Host buffers mirror the device row layout: each shared device backing
        storage gets one CPU buffer of the same size, and every layer's host
        view has the same shape, strides and storage offset as its device
        view. Registering the host buffers then advertises the same per-group
        row geometry as the device registration, and the d2h/h2d copies index
        both sides with the same per-group kernel block ids.
        """
        if self.kv_transfer_config.enable_permute_local_kv:
            raise NotImplementedError(
                "enable_permute_local_kv needs host buffers with a physical "
                "layout different from the device, which mirrored row-layout "
                "host staging cannot express."
            )
        xfer_buffers: dict[str, torch.Tensor] = {}
        backing_by_storage: dict[int, torch.UntypedStorage] = {}
        try:
            for layer_name, kv_cache in kv_caches.items():
                storage = kv_cache.untyped_storage()
                storage_key = storage.data_ptr()
                if storage_key not in backing_by_storage:
                    backing = torch.empty(
                        storage.nbytes(), dtype=torch.uint8, device="cpu"
                    )
                    backing_by_storage[storage_key] = backing.untyped_storage()
                xfer_buffers[layer_name] = torch.empty(
                    0, dtype=kv_cache.dtype, device="cpu"
                ).set_(
                    backing_by_storage[storage_key],
                    kv_cache.storage_offset(),
                    kv_cache.shape,
                    kv_cache.stride(),
                )
        except MemoryError as e:
            logger.error("NIXLConnectorWorker gets %s.", e)
            raise

        self.host_xfer_buffers = xfer_buffers

    def set_host_xfer_buffer_ops(self, copy_operation: CopyBlocksOp):
        """Assign copy (d2h, h2d) operations when host buffer is used."""
        # Set a no-op if the host buffer is not cpu.
        if self.kv_buffer_device != "cpu":
            return
        # Set a no-op if self.device_type is 'cpu'.
        if self.device_type == "cpu":
            return
        assert self.use_host_buffer
        self.copy_blocks = copy_operation

    def _log_failure(
        self,
        failure_type: str,
        req_id: str | None,
        msg: str = "",
        error: BaseException | None = None,
        meta: ReqMeta | None = None,
        **extra_context,
    ):
        """Log transfer failure with structured context for easier debugging."""
        context: dict[str, Any] = {
            "failure_type": failure_type,
            "request_id": req_id,
            "engine_id": self.engine_id,
        }
        if meta is None and req_id is not None:
            # Try to get metadata from in progress transfers when not provided
            meta = self._recving_metadata.get(req_id)

        if meta and meta.remote:
            context.update(
                {
                    "remote_engine_id": meta.remote.engine_id,
                    "remote_request_id": meta.remote.request_id,
                    "remote_host": meta.remote.host,
                    "remote_port": meta.remote.port,
                    "num_local_blocks": sum(
                        len(group) for group in meta.local_block_ids
                    ),
                    "num_remote_blocks": sum(
                        len(group) for group in meta.remote.block_ids
                    ),
                    "local_block_ids_sample": meta.local_block_ids[0][:10]
                    if meta.local_block_ids
                    else [],
                }
            )

        context.update(extra_context)
        if msg:
            failure_type = f"{failure_type}. {msg}"

        logger.error(
            "NIXL transfer failure: %s | Context: %s",
            failure_type,
            context,
            exc_info=error is not None,
            stacklevel=2,
        )

    def _ensure_handshake(
        self,
        engine_id: EngineId,
        host: str,
        port: int,
        tp_size: int,
        dcp_size: int = 1,
        pp_size: int = 1,
        notif_agents_only: bool = False,
    ) -> Future[tuple[dict[tuple[int, int], str], float]] | None:
        """Ensure a handshake is in-flight (or already done) for *engine_id*.

        Returns the ``Future`` if a handshake is pending (or was just
        started), or ``None`` if the handshake already completed
        successfully.  Callers can attach per-request callbacks to the
        returned future.
        Failures to handshake are logged and the request is marked as failed.
        """
        self._evict_stale_engines()
        with self._handshake_lock:
            if engine_id in self._remote_agents:
                return None
            fut = self._handshake_futures.get(engine_id)
            if fut is not None:
                return fut
            fut = self._handshake_initiation_executor.submit(
                self._nixl_handshake,
                host,
                port,
                tp_size,
                engine_id,
                dcp_size,
                pp_size,
                notif_agents_only,
            )
            self._handshake_futures[engine_id] = fut

            def done_callback(
                f: Future[tuple[dict[tuple[int, int], str], float]],
                eid=engine_id,
            ):
                with self._handshake_lock:
                    del self._handshake_futures[eid]
                    try:
                        remote_agents, clock_offset = f.result()
                        self._remote_agents[eid] = remote_agents
                        self._engine_clock_offset[eid] = clock_offset
                        self._engine_last_active[eid] = time.perf_counter()
                        self._track_remote_engine_replacement(eid, host, port)
                    except Exception as e:
                        self._log_failure(
                            failure_type="handshake_setup_failed",
                            req_id=None,
                            error=e,
                            remote_engine_id=eid,
                        )
                        # Count once per handshake, regardless of waiting requests.
                        self.xfer_stats.record_failed_handshake()

            fut.add_done_callback(done_callback)
            return fut

    def _background_nixl_handshake(
        self, req_id: str, remote_engine_id: EngineId, meta: ReqMeta
    ):
        # Do NIXL handshake in background and add to _ready_requests when done.
        assert meta.remote is not None
        fut = self._ensure_handshake(
            remote_engine_id,
            meta.remote.host,
            meta.remote.port,
            meta.tp_size,
            meta.dcp_size,
            meta.pp_size,
        )
        if fut is None:
            # Already handshaked — only happens if caller does not pre-check.
            self._ready_requests.put((req_id, meta))
            return

        # Check handshake success before proceeding with request.
        def request_ready(f: Future[Any], entry=(req_id, meta)):
            try:
                f.result()
                self._ready_requests.put(entry)
            except Exception as e:
                self._log_failure(
                    failure_type="handshake_failed",
                    req_id=req_id,
                    error=e,
                    meta=meta,
                )
                self._failed_recv_reqs.put(req_id)

        fut.add_done_callback(request_ready)

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """Register the KV Cache data in nixl."""
        use_layer_name_routing = self._requires_layer_name_routing()
        route_packed_layers = self._has_packed_cache and use_layer_name_routing
        self.transfer_topo = TransferTopology(
            tp_rank=self.transfer_tp_rank,
            tp_size=self.transfer_tp_size,
            engine_id=self.engine_id,
            is_mla=self.use_mla,
            total_num_kv_heads=1
            if self.use_mla
            else self.model_config.get_total_num_kv_heads(),
            attn_backends=self.attn_backends,
            dcp_size=self.dcp_size,
            # SSM States come in tuples (ssm, conv)
            tensor_shape=next(iter(kv_caches.values())).shape
            if not self._has_mamba
            else None,
            is_mamba=self._has_mamba,
        )
        backend_name = self.backend_name
        if self._supports_pp_hma and self._has_packed_cache:
            # PP can reorder the same backends across stages of a packed model.
            backend_name = ",".join(
                sorted({backend.get_name() for backend in self.attn_backends})
            )
        self.compat_hash = compute_nixl_compatibility_hash(
            self.vllm_config,
            backend_name,
            transfer_mode=self._TRANSFER_MODE,
        )

        if self._is_csa_linear and self.use_host_buffer:
            raise NotImplementedError(
                "NIXL host staging does not preserve CSA-linear shared tensors."
            )

        if self.use_host_buffer:
            self.initialize_host_xfer_buffer(kv_caches=kv_caches)
            assert len(self.host_xfer_buffers) == len(kv_caches), (
                f"host_buffer: {len(self.host_xfer_buffers)}, "
                f"kv_caches: {len(kv_caches)}"
            )
            xfer_buffers = self.host_xfer_buffers
        else:
            xfer_buffers = kv_caches
            assert not self.host_xfer_buffers, (
                "host_xfer_buffer should not be initialized when "
                f"kv_buffer_device is {self.kv_buffer_device}"
            )

        logger.info(
            "Registering KV_Caches. use_mla: %s, kv_buffer_device: %s, "
            "use_host_buffer: %s",
            self.use_mla,
            self.kv_buffer_device,
            self.use_host_buffer,
        )

        registration_ranges: dict[tuple[int, str], tuple[int, int, int]] = {}
        region_mem_types: list[str] = []
        region_base_addresses: list[int] = []
        seen_region_keys: list[Any] = []
        self._ssm_region_indices = []
        self._ple_block_len = None
        self._ple_region_index = None

        # Whole-row registration (one region per group) is limited to device
        # groups whose pages are TP-replicated, so a block never needs head
        # slicing inside a row. Head-sharded groups register one kernel-row
        # region per layer instead, which keeps all the heterogeneous-TP
        # splitting machinery per-layer. Host groups keep each layer in its own
        # dense region, so they register per layer too.
        packed_row_groups: set[int] = set()
        for index, tg in enumerate(self.kv_cache_config.transfer_groups):
            if not tg.host_resident and isinstance(
                tg.kv_cache_spec, (MLAAttentionSpec, SlidingWindowMLASpec)
            ):
                packed_row_groups.add(index)

        track_region_layers = self._tracks_region_layers()
        region_layers: list[list[str]] = []
        packed_member_layouts: dict[str, tuple[int, int]] = {}

        # K and V are packed into the content dim, so each attention layer is a
        # single NIXL region whose block transfers as one unit. Mamba layers instead
        # register separate conv/ssm sub-regions (see `_build_mamba_local`).
        # P and D may allocate equivalent transferable layers in different
        # cache-group orders. Keep their region lists aligned without putting
        # layers.10 before layers.2, which would break PP region slicing.
        layer_names = (
            xfer_buffers
            if self._is_csa_linear
            else sorted(
                xfer_buffers,
                key=_region_sort_key,
            )
        )
        for layer_name in layer_names:
            cache = xfer_buffers[layer_name]
            layer_spec = self._layer_specs.get(layer_name)
            if layer_spec is None:
                logger.debug(
                    "Skipping layer %s as no KVCache spec is present. "
                    "This is likely because the layer is sharing its KV cache",
                    layer_name,
                )
                continue
            group_index = self.kv_cache_config.transfer_group_index_by_layer[layer_name]
            group = self.kv_cache_config.transfer_groups[group_index]
            # Views arrive at kernel-row granularity: shape[0] is the
            # group's kernel-block count and stride(0) is the group row.
            layer_row_bytes = layer_spec.page_size_bytes // group.kernel_rows_per_block
            num_blocks = cache.shape[0]
            assert cache.ndim >= 2
            group_id = group_index
            is_mla_region = isinstance(
                layer_spec, (MLAAttentionSpec, SlidingWindowMLASpec)
            )
            logger.debug(
                "Registering layer %s with cache shape: %s", layer_name, cache.shape
            )
            storage = cache.untyped_storage()
            storage_addr = storage.data_ptr()
            is_host_resident = group.host_resident
            if cache.device.type == "cpu":
                mem_type = "DRAM"
                region_device_id = 0
            else:
                mem_type = self.nixl_memory_type
                region_device_id = max(cache.get_device(), 0)
                self.device_id = region_device_id
            if is_host_resident:
                registration_base = cache.data_ptr()
                registration_len = cache.nbytes
            else:
                registration_base = storage_addr
                registration_len = storage.nbytes()
            storage_key = (registration_base, mem_type)
            if storage_key not in registration_ranges:
                registration_ranges[storage_key] = (
                    registration_base,
                    registration_base + registration_len,
                    region_device_id,
                )

            # Kernel-row regions over the shared pool: stride is the group
            # row and desc ids index kernel blocks of the group's views.
            block_stride = cache.stride(0) * cache.element_size()
            if group_index in packed_row_groups and not use_layer_name_routing:
                # TP-replicated (MLA) group: register the whole row once so
                # a manager block is R_g consecutive descriptors.
                region_specs = [(registration_base, block_stride, block_stride)]
                offset = cache.data_ptr() - storage_addr
                assert offset >= 0 and offset + layer_row_bytes <= block_stride
                packed_member_layouts[layer_name] = (offset, layer_row_bytes)
            else:
                # Head-sharded attention and Mamba groups: one region per
                # layer view, so head slicing and conv/ssm sub-descriptors
                # keep working per layer.
                region_specs = [(cache.data_ptr(), layer_row_bytes, block_stride)]

            for base_addr, block_len, block_stride in region_specs:
                # A packed PP producer keeps one region per layer, so aliases at
                # one address (a layer and its SWA view) stay distinct.
                # Every group's row grid starts at the pool base, so the bare
                # address does not identify a region: key the dedup on the
                # group and the full region geometry instead. A merged region
                # then always belongs to a single group.
                region_key: Any = (group_id, base_addr, block_len, block_stride)
                if region_key in seen_region_keys and not route_packed_layers:
                    # A group's layers share one spec, so a merged region's
                    # geometry and MLA flag are the first layer's.
                    region_index = seen_region_keys.index(region_key)
                    assert region_mem_types[region_index] == mem_type
                else:
                    region_index = len(seen_region_keys)
                    seen_region_keys.append(region_key)
                    region_base_addresses.append(base_addr)
                    self.block_len_per_layer.append(block_len)
                    self.block_stride_per_layer.append(block_stride)
                    self.region_group_ids.append(group_id)
                    self.region_names.append(layer_name)
                    self.region_num_blocks.append(num_blocks)
                    self._region_is_mla.append(is_mla_region)
                    region_mem_types.append(mem_type)

                if track_region_layers:
                    if region_index == len(region_layers):
                        region_layers.append([layer_name])
                    else:
                        region_layers[region_index].append(layer_name)

                if isinstance(layer_spec, MambaSpec):
                    if layer_spec.tp_replicated:
                        assert self._is_csa_linear
                        assert self._ple_region_index in (None, region_index)
                        self._ple_region_index = region_index
                        self._ple_block_len = block_len
                    elif region_index not in self._ssm_region_indices:
                        self._ssm_region_indices.append(region_index)

        logger.debug(
            "Different block lengths collected: %s", set(self.block_len_per_layer)
        )
        assert (
            len(self.block_len_per_layer)
            == len(region_base_addresses)
            == len(self._region_is_mla)
            == len(self.block_stride_per_layer)
            == len(self.region_group_ids)
            == len(self.region_names)
            == len(self.region_num_blocks)
        )
        self._head_sharded_draft_regions = self._get_head_sharded_draft_regions()

        self.kv_caches_base_addr[self.engine_id][self.tp_rank] = region_base_addresses
        self.num_regions = len(region_base_addresses)
        self.region_mem_types = region_mem_types

        self._set_region_layers(region_layers)
        self._packed_member_layouts = packed_member_layouts

        if self.pp_size > 1 and not use_layer_name_routing:
            raise NotImplementedError(
                "Positional PP region slicing is undefined for per-group "
                "large-page regions; PP requires NixlPushConnector with "
                "layer-name routing."
            )

        # Total local FA descriptors (boundary between FA and mamba descs).
        xfer_region_num_blocks = (
            [self.region_num_blocks[i] for i in self._transfer_layer_region_indices]
            if self._transfer_layer_region_indices
            else self.region_num_blocks
        )
        if self._transfer_layer_region_indices:
            self.num_descs = sum(xfer_region_num_blocks)
        else:
            # SSM regions hold no attention data; their descriptors come from
            # _build_mamba_local.
            self.num_descs = sum(
                self.region_num_blocks[i] for i in self._fa_region_indices()
            )

        self._mixed_mem_types = len(set(region_mem_types)) > 1
        self._skip_dram_xfer = self._leaves_dram_xfer_to_rank0()
        if self._mixed_mem_types:
            assert self.use_mla and not self._has_mamba, (
                "Mixed-device KV registration is only supported for MLA "
                "models without Mamba layers."
            )
        for mem_type in sorted(set(region_mem_types)):
            if mem_type == "DRAM" and self._skip_dram_xfer:
                continue
            ranges_for_mem_type = [
                (start, end - start, device_id, "")
                for (_, cache_mem_type), (
                    start,
                    end,
                    device_id,
                ) in registration_ranges.items()
                if cache_mem_type == mem_type
            ]
            descs = self.nixl_wrapper.get_reg_descs(ranges_for_mem_type, mem_type)
            self.nixl_wrapper.register_memory(descs, backends=self.nixl_backends)
            self._registered_descs.append(descs)

        self.device_kv_caches = kv_caches
        self.dst_region_num_blocks[self.engine_id] = xfer_region_num_blocks
        self.dst_region_group_ids[self.engine_id] = self.region_group_ids
        self.dst_region_mem_types[self.engine_id] = self.region_mem_types

        if self._has_mamba:
            logger.info(
                "Hybrid SSM registration: num_regions=%s, "
                "num_descs=%s, mamba_ssm_size=%s, block_len_per_layer=%s",
                self.num_regions,
                self.num_descs,
                self._mamba_ssm_size,
                set(self.block_len_per_layer),
            )

        # Register local/src descr for NIXL xfer.
        self.src_xfer_handle, self.src_blocks_data = self.register_local_xfer_handler()

        self._group_layer_names = [
            sorted(
                name
                for name in group.layer_names
                if name in self._layer_specs and name in xfer_buffers
            )
            for group in self.kv_cache_config.transfer_groups
        ]

        # After KV Caches registered, listen for new connections.
        agent_metadata = NixlAgentMetadata(
            engine_id=self.engine_id,
            agent_metadata=self.nixl_wrapper.get_agent_metadata(),
            device_id=self.device_id,
            kv_caches_base_addr=self.kv_caches_base_addr[self.engine_id][self.tp_rank],
            block_lens=self.block_len_per_layer,
            block_strides=self.block_stride_per_layer,
            kv_cache_layout=self.kv_cache_layout
            if not self.use_host_buffer
            else self.host_buffer_kv_cache_layout,
            block_size=self.block_size,
            ssm_sizes=self._mamba_ssm_size,
            ple_block_len=self._ple_block_len,
            attn_backend_name=self.backend_name,
            region_num_blocks=self.region_num_blocks,
            region_group_ids=self.region_group_ids,
            region_names=self.region_names,
            region_mem_types=self.region_mem_types,
            dcp_size=self.dcp_size,
            pcp_size=self.pcp_size,
            region_members=self.region_members,
            packed_member_layouts=self._packed_member_layouts,
            group_geometries=self.kv_cache_config.transfer_group_geometries,
            group_layer_names=self._group_layer_names,
        )
        # Wrap metadata in payload with hash for defensive decoding
        assert self.compat_hash is not None
        encoder = msgspec.msgpack.Encoder()
        self.xfer_handshake_metadata = NixlHandshakePayload(
            compatibility_hash=self.compat_hash,
            agent_metadata_bytes=encoder.encode(agent_metadata),
        )

    def _ple_page_len(self) -> int:
        """Transfer length of one PLE state page.

        The PLE page shares its region with the first page of other cache
        groups (block-outer layout), whose block_len may be smaller; the
        descriptors must cover the whole PLE page or part of the conv state
        never reaches the decode side."""
        assert self._ple_region_index is not None
        assert self._ple_block_len is not None, (
            "PLE region registered without its page length"
        )
        return self._ple_block_len

    def unregister_kv_caches(self) -> None:
        if self._kv_unregistered or not self._registered_descs:
            return
        self._release_local_xfer_handles()
        # Remote engines hold local handles too; they handshake again on demand.
        for engine_id in list(self._remote_agents):
            self._cleanup_remote_engine(engine_id, log_eviction=False)
        for descs in self._registered_descs:
            self.nixl_wrapper.deregister_memory(descs)
        self._kv_unregistered = True

    def restore_kv_caches(self) -> None:
        if not self._kv_unregistered:
            return
        for descs in self._registered_descs:
            self.nixl_wrapper.register_memory(descs, backends=self.nixl_backends)
        self.src_xfer_handle, self.src_blocks_data = self.register_local_xfer_handler()
        # Peers load the new registration from the agent metadata.
        payload = self.xfer_handshake_metadata
        assert payload is not None
        agent_metadata = msgspec.msgpack.decode(
            payload.agent_metadata_bytes, type=NixlAgentMetadata
        )
        agent_metadata.agent_metadata = self.nixl_wrapper.get_agent_metadata()
        payload.agent_metadata_bytes = msgspec.msgpack.encode(agent_metadata)
        self._kv_unregistered = False

    def _build_mamba_local(self, base_addresses: list[int]) -> np.ndarray:
        """Build desc regions (conv sub-projections + ssm) per layer for
        local mamba blocks with DS conv layout, as Nx5 uint64 strided descs.

        A Mamba block interleaves conv and SSM state, which crucially differ in
        size, so the two are indexed as separate sub-regions. Attention blocks
        instead pack K and V into the content dim and transfer as a single unit.
        Reference diagram:
                            KVCacheTensor (Shared)
                               /       \\
                              /         \\
                             /           \\
        Attention (FlashInfer) View      Mamba View
                  |                          |
                  |                          |
           +-------------------+         +-------------------+
           | KVCacheTensor     |         | KVCacheTensor      |
           |                   |         |                    |
           |<----- page ------>|         |<----- page ------->|
           |       size        |         |       size         |
           |  Key 0  |  Val 0  |         |Conv 0  |   SSM 0   |
           |  Key 1  |  Val 1  |         |Conv 1  |   SSM 1   |
           |   ...   |   ...   |         |  ...   |    ...    |
           | Key N-2 | Val N-2 |         |Conv N-2|   SSM N-2 |
           | Key N-1 | Val N-1 |         |Conv N-1|   SSM N-1 |
           +-------------------+         +--------------------+
           |1st_split-2nd_split|         |1st_split-2nd_split |

        Mamba state blocks are indivisible (not token-extent data), so the
        descriptors always use the local page geometry.
        """
        assert base_addresses, "Local KV cache base addresses must not be empty."
        assert self._conv_decomp is not None
        conv_offsets = self._conv_decomp.local_conv_offsets
        conv_size, ssm_size = self._mamba_ssm_size
        device_id = self.device_id
        strided_descs: list[tuple[int, int, int, int, int]] = []

        region_indices = self._ssm_region_indices or range(len(base_addresses))
        for i in region_indices:
            base_addr = base_addresses[i]
            block_stride = self.block_stride_per_layer[i]
            # Each region keeps its own kernel-row count.
            count = self.region_num_blocks[i]
            for off, sz in conv_offsets:
                strided_descs.append(
                    (base_addr + off, sz, device_id, block_stride, count)
                )
            # SSM temporal state follows the conv state.
            strided_descs.append(
                (base_addr + conv_size, ssm_size, device_id, block_stride, count)
            )

        if (region_index := self._ple_region_index) is not None:
            block_len = self._ple_page_len()
            block_stride = self.block_stride_per_layer[region_index]
            base_addr = base_addresses[region_index]
            count = self.region_num_blocks[region_index]
            strided_descs.append((base_addr, block_len, device_id, block_stride, count))

        return self._pack_strided_descs(strided_descs)

    def _build_mamba_remote(
        self,
        nixl_agent_meta: NixlAgentMetadata,
        tp_ratio: int,
    ) -> np.ndarray:
        """Build remote desc regions (conv sub-projections + ssm) per layer.
        For hetero-TP, each D rank reads only its sub-projection slice from
        the P rank. Returns Nx5 uint64 strided descs."""
        assert nixl_agent_meta.kv_caches_base_addr, (
            "Remote KV cache base addresses must not be empty."
        )
        assert self._conv_decomp is not None
        effective_ratio = max(tp_ratio, 1)
        # Mamba conv state is always TP-sharded, even when attention KV
        # is replicated (num_kv_heads < tp_size).
        local_offset = self.tp_rank % effective_ratio
        conv_size_remote = nixl_agent_meta.ssm_sizes[0]

        conv_offsets = self._conv_decomp.remote_conv_offsets(local_offset, tp_ratio)
        if tp_ratio >= 1:
            ssm_read_size = self._mamba_ssm_size[1]
        else:
            ssm_read_size = nixl_agent_meta.ssm_sizes[1]

        # Each region keeps its own kernel-row count.
        assert nixl_agent_meta.region_num_blocks is not None
        remote_region_counts = nixl_agent_meta.region_num_blocks
        device_id = nixl_agent_meta.device_id

        strided_descs: list[tuple[int, int, int, int, int]] = []
        # NOTE (ZhanqiuHu): use per-layer block_lens[i], not [0], in case
        # block lengths vary across layers (e.g. MLA).
        region_indices = self._ssm_region_indices or range(
            len(nixl_agent_meta.kv_caches_base_addr)
        )
        for i in region_indices:
            base_addr = nixl_agent_meta.kv_caches_base_addr[i]
            block_stride = nixl_agent_meta.block_strides[i]
            count = remote_region_counts[i]
            for off, sz in conv_offsets:
                strided_descs.append(
                    (base_addr + off, sz, device_id, block_stride, count)
                )
            # SSM temporal state is also TP-sharded on the heads dimension.
            ssm_addr = base_addr + conv_size_remote + local_offset * ssm_read_size
            strided_descs.append(
                (ssm_addr, ssm_read_size, device_id, block_stride, count)
            )

        if (region_index := self._ple_region_index) is not None:
            local_block_len = self._ple_page_len()
            # Same NIXL_CONNECTOR_VERSION on both sides, so the field is set.
            assert nixl_agent_meta.ple_block_len is not None
            remote_block_len = nixl_agent_meta.ple_block_len
            if local_block_len != remote_block_len:
                raise ValueError(
                    "PLE pages require identical P/D geometry: "
                    f"local={local_block_len}, remote={remote_block_len}."
                )
            remote_block_stride = nixl_agent_meta.block_strides[region_index]
            addr = nixl_agent_meta.kv_caches_base_addr[region_index]
            count = remote_region_counts[region_index]
            strided_descs.append(
                (addr, remote_block_len, device_id, remote_block_stride, count)
            )

        return self._pack_strided_descs(strided_descs)

    @staticmethod
    def _pack_strided_descs(
        strided_descs: list[tuple[int, int, int, int, int]],
    ) -> np.ndarray:
        """Pack (addr, len, dev_id, stride, count) rows into an Nx5 uint64 array.

        Each row describes `count` blocks of `len` bytes whose starts are
        `stride` bytes apart; NIXL >= 1.5 prepares this layout directly.
        """
        return np.array(strided_descs, dtype=np.uint64).reshape(-1, 5)

    @staticmethod
    def _strided_desc_starts(strided_descs: np.ndarray) -> np.ndarray:
        """Flat block index of each strided desc's first block."""
        counts = strided_descs[:, 4]
        return np.cumsum(counts) - counts

    def _build_fa_local(self, base_addresses: list[int]) -> np.ndarray:
        """Build local FA descriptors as Nx5 uint64 strided descs."""
        assert self.transfer_topo is not None
        assert base_addresses, "Local KV cache base addresses must not be empty."
        if self._transfer_layer_region_indices:
            region_indices: Iterable[int] = self._transfer_layer_region_indices
        else:
            # SSM regions hold no attention data under per-group rows.
            region_indices = self._fa_region_indices()
        return self._pack_strided_descs(
            [
                (
                    base_addresses[i],
                    self.block_len_per_layer[i],
                    self.device_id,
                    self.block_stride_per_layer[i],
                    self.region_num_blocks[i],
                )
                for i in region_indices
            ]
        )

    def _build_fa_remote(
        self,
        plan: TPMapping,
        nixl_agent_meta: NixlAgentMetadata,
    ) -> np.ndarray:
        """Build remote FA descriptors as Nx5 uint64 strided descs."""
        assert self.transfer_topo is not None
        assert nixl_agent_meta.kv_caches_base_addr, (
            "Remote KV cache base addresses must not be empty."
        )
        fa_group_idx = next(
            i for i, t in enumerate(self._group_spec_types) if _is_attention_spec(t)
        )
        assert nixl_agent_meta.region_num_blocks is not None
        region_num_blocks = nixl_agent_meta.region_num_blocks
        device_id = nixl_agent_meta.device_id
        block_strides = nixl_agent_meta.block_strides
        strided_descs: list[tuple[int, int, int, int, int]] = []
        use_layer_name_routing = bool(self._transfer_layer_region_indices)
        # Remote region order matches local (region_names equality is checked
        # at handshake); SSM descs come from _build_mamba_remote, so the FA
        # descs skip the SSM state regions.
        skip_regions: set[int] = set()
        if not use_layer_name_routing:
            skip_regions = set(self._ssm_region_indices)
            if self._ple_region_index is not None:
                skip_regions.add(self._ple_region_index)
        for i, base_addr in enumerate(nixl_agent_meta.kv_caches_base_addr):
            if i in skip_regions:
                continue
            local_region = (
                self._transfer_layer_region_indices[i] if use_layer_name_routing else i
            )
            replicated = self._is_region_replicated(local_region)
            # Read our whole local region size from remote.
            local_block_len = self.block_len_per_layer[local_region]

            # REPLICATE reads the whole block once at offset 0; SPLIT gathers
            # its head slice from each source rank at a per-rank offset.
            if replicated:
                num_reads, rank_offset = 1, 0
            else:
                region_plan = (
                    self.draft_tp_mappings[nixl_agent_meta.engine_id]
                    if local_region in self._head_sharded_draft_regions
                    else plan
                )
                num_reads = len(region_plan.source_ranks_per_group[fa_group_idx])
                rank_offset = region_plan.rank_offset_factor * local_block_len
            local_block_len = local_block_len // num_reads

            addr = base_addr + rank_offset
            count = region_num_blocks[i]
            strided_descs.append(
                (addr, local_block_len, device_id, block_strides[i], count)
            )
        return self._pack_strided_descs(strided_descs)

    def register_local_xfer_handler(self) -> tuple[int, np.ndarray]:
        """Register the local xfer handler for the local KV regions."""
        assert self.transfer_topo is not None
        local_base_addresses = self.kv_caches_base_addr[self.engine_id][self.tp_rank]

        blocks_data = self._build_fa_local(local_base_addresses)
        logger.debug(
            "Created %s blocks for src engine %s and rank %s on device id %s",
            blocks_data[:, 4].sum(),
            self.engine_id,
            self.tp_rank,
            self.device_id,
        )
        if self._has_mamba:
            assert self.num_descs == blocks_data[:, 4].sum()
            # TODO (ZhanqiuHu): For homogeneous TP (tp_ratio == 1), the 3-descs split
            # is unnecessary — a single conv desc per block suffices.  Consider
            # adding a fast path that falls back to the standard 2-region
            # registration (_build_fa_local mamba=True) when no hetero-TP
            # remote has been seen.  Currently we always register 4 regions
            # because local descs are created before knowing the remote TP.
            logger.debug("Registering local Mamba descriptors (4 regions/layer)")
            mamba = self._build_mamba_local(local_base_addresses)
            blocks_data = np.concatenate([blocks_data, mamba])

        if self._mixed_mem_types:
            desc_is_dram = np.concatenate(
                [
                    np.full(
                        count,
                        mem_type == "DRAM",
                        dtype=bool,
                    )
                    for mem_type, count in zip(
                        self.region_mem_types,
                        self.region_num_blocks,
                        strict=True,
                    )
                ]
            )
            assert len(desc_is_dram) == blocks_data[:, 4].sum()
            desc_pos = np.empty(len(desc_is_dram), dtype=np.int64)
            dram_idx = np.where(desc_is_dram)[0]
            vram_idx = np.where(~desc_is_dram)[0]
            desc_pos[dram_idx] = np.arange(len(dram_idx), dtype=np.int64)
            desc_pos[vram_idx] = np.arange(len(vram_idx), dtype=np.int64)
            self._desc_is_dram = desc_is_dram
            self._desc_pos = desc_pos
            # DRAM descriptors are registered under CPU device 0.
            # A strided desc never spans regions, so its first block decides.
            strided_is_dram = desc_is_dram[self._strided_desc_starts(blocks_data)]
            blocks_data[strided_is_dram, 2] = 0
            if self._skip_dram_xfer:
                self._dram_src_handle = None
            else:
                self._dram_src_handle = self.nixl_wrapper.prep_xfer_dlist(
                    "NIXL_INIT_AGENT",
                    blocks_data[strided_is_dram],
                    mem_type="DRAM",
                )
            device_handle = self.nixl_wrapper.prep_xfer_dlist(
                "NIXL_INIT_AGENT",
                blocks_data[~strided_is_dram],
                mem_type=self.nixl_memory_type,
            )
            return device_handle, blocks_data

        # NIXL_INIT_AGENT to be used for preparations of local descs.
        handle = self.nixl_wrapper.prep_xfer_dlist(
            "NIXL_INIT_AGENT", blocks_data, mem_type=self.region_mem_types[0]
        )
        return handle, blocks_data

    def add_remote_agent(
        self,
        nixl_agent_meta: NixlAgentMetadata,
        remote_tp_rank: int = 0,
        remote_tp_size: int = 1,
        remote_dcp_size: int = 1,
    ) -> str:
        """Add the remote NIXL agent and prepare the descriptors for reading cache
        blocks from remote.

        In particular, handle both homogeneous and heterogeneous TP. The former
        requires local rank_i to read from remote rank_i.
        The latter, in the case of D.world_size < P.world_size, requires that a
        local (D) TP worker reads from multiple remote (P) TP workers.
        Conversely, assuming D.world_size > P.world_size, two or more local TP
        workers will read from a single remote TP worker.

        Here's an example for the last case described above (non-MLA):

        rank_offset     p_remote_tp_rank
        (kv split no)
        --------------------------------
            0                 0      Worker0  ---- 1st half of KV ----> Worker0  [ KV Cache ]
                                                                        /
            1                 0      Worker1  ---- 2nd half of KV -----/

            0                 1      Worker2  ---- 1st half of KV ----> Worker1  [ KV Cache ]
                                                                        /
            1                 1      Worker3  ---- 2nd half of KV -----/


                                Decoder TP workers                     Prefix TP workers
                                  (world_size=4)                         (world_size=2)
                                                 tp_ratio = 4 // 2 = 2

        Considering the KV Caches, if P-Worker_i has cache size [2, num_blocksP, kv_heads, block_size, head_dim]
        then D-Worker_j has [2, num_blocksD, kv_heads//tp_ratio, block_size, head_dim]. Mind the "LBHNC" layout format.
        Assuming num_blocksD >= num_blocksP, D-Worker0 reads from P-Worker0 by preparing the kv_heads//tp_ratio
        first heads from all the slots of all the blocks. D-Worker1 will do the same, but reading the second split
        along the kv_heads dimension, and so forth until "tp_ratio" D TP workers have pulled from P-Worker0.

        Note that the above will also hold true for the homogeneous TP case, where tp_ratio evaluates to 1.

        Regarding MLA case, the cache is replicated across TP workers so the rank_offset will just always be 0
        so that the whole cache is shared by "tp_ratio" D TP workers.

        For Mamba hetero-TP, both tp_ratio > 0 (D_TP > P_TP) and
        tp_ratio < 0 (P_TP > D_TP) are supported by the 3-read transfer.
        """  # noqa: E501
        engine_id = nixl_agent_meta.engine_id
        # TODO re-evaluate refreshing for scaling/recovery
        if (0, remote_tp_rank) in self._remote_agents.get(engine_id, {}):
            logger.debug(
                "Remote agent with engine_id %s and rank"
                "%s already exchanged metadata, skip handshake.",
                engine_id,
                remote_tp_rank,
            )
            return self._remote_agents[engine_id][(0, remote_tp_rank)]

        assert self.transfer_topo is not None
        transfer_topo = self.transfer_topo
        if self._transfer_layer_region_indices:
            if self.block_size != nixl_agent_meta.block_size:
                raise NotImplementedError(
                    "Attention-HMA push requires identical P/D block sizes."
                )
            if remote_tp_size > transfer_topo.tp_size and not self.use_mla:
                raise NotImplementedError(
                    "Attention-HMA push does not support decode TP greater than "
                    "prefill TP yet"
                )
            self._align_remote_regions_by_layer(nixl_agent_meta)
        elif nixl_agent_meta.packed_member_layouts and not self._packed_member_layouts:
            # With HiSparse, P may be on BL* while D is LB*, as only indexer is on HBM.
            # TODO (NickLucche) we may consider not allowing this case at all (P BL*)
            # and just clear this function.
            self._split_remote_packed_rows(nixl_agent_meta)

        # Both sides must agree on the global block size B.
        if nixl_agent_meta.block_size != self.block_size:
            raise RuntimeError(
                "NIXL transfers require equal P/D block sizes "
                f"(local={self.block_size}, "
                f"remote={nixl_agent_meta.block_size})."
            )

        if (
            self._packed_member_layouts
            and not self._transfer_layer_region_indices
            and self._packed_member_layouts != nixl_agent_meta.packed_member_layouts
        ):
            # Whole rows transfer as one unit, so both sides must pack them alike.
            raise RuntimeError(
                "Packed KV block rows differ between P and D. Use the same "
                "VLLM_KV_CACHE_LAYOUT and block size on both sides."
            )

        ### Register remote engine in TransferTopology (idempotent).
        transfer_info = EngineTransferInfo(
            remote_tp_size=remote_tp_size,
            remote_block_size=nixl_agent_meta.block_size,
            remote_block_len=nixl_agent_meta.block_lens[0],
            remote_dcp_size=remote_dcp_size,
        )
        transfer_topo.register_remote_engine(engine_id, transfer_info)
        logger.info("Transfer plan: %s", transfer_topo.describe(engine_id))

        self.tp_mappings[engine_id] = compute_tp_mapping(
            transfer_topology=transfer_topo,
            remote_tp_size=remote_tp_size,
            group_spec_types=self._group_spec_types,
            remote_dcp_size=remote_dcp_size,
        )
        if self._head_sharded_draft_regions:
            self.draft_tp_mappings[engine_id] = compute_tp_mapping(
                transfer_topology=transfer_topo,
                remote_tp_size=remote_tp_size,
                group_spec_types=self._group_spec_types,
                remote_dcp_size=remote_dcp_size,
                head_sharded_kv_heads=self._head_sharded_draft_kv_heads,
            )

        remote_agent_name = self.nixl_wrapper.add_remote_agent(
            nixl_agent_meta.agent_metadata
        )

        # Create dst descs and xfer side handles. TP workers have same #blocks
        # so we only register once per engine_id.
        if engine_id not in self.dst_region_num_blocks:
            # Per-region counts and group ids are mandatory.
            assert nixl_agent_meta.region_num_blocks is not None
            assert nixl_agent_meta.region_group_ids is not None
            assert nixl_agent_meta.region_mem_types is not None
            self.dst_region_num_blocks[engine_id] = nixl_agent_meta.region_num_blocks
            self.dst_region_group_ids[engine_id] = nixl_agent_meta.region_group_ids
            self.dst_region_mem_types[engine_id] = nixl_agent_meta.region_mem_types

        # Keep track of remote agent kv caches base addresses.
        self.kv_caches_base_addr[engine_id][remote_tp_rank] = (
            nixl_agent_meta.kv_caches_base_addr
        )
        self._validate_remote_agent_handshake(
            nixl_agent_meta, remote_tp_size, remote_dcp_size
        )
        self._dst_group_geometries.setdefault(
            engine_id, nixl_agent_meta.group_geometries
        )

        # This is 1 when P and D `--tensor-parallel-size` match. Otherwise,
        # this is the ratio between the two sizes.
        tp_ratio = transfer_topo.tp_ratio(remote_tp_size)

        logger.debug(
            "Registering remote agent (%s, rank %s) memory regions with tp_ratio %s",
            engine_id,
            remote_tp_rank,
            tp_ratio,
        )

        plan = self.tp_mappings[engine_id]

        ### (Optional) Register local agent memory regions. MLA is not split.
        if self._needs_split_local_xfer_handles(tp_ratio, plan) and (
            tp_ratio not in self.src_xfer_handles_by_tp_ratio
        ):
            # Remote tp_size > local tp_size: read from multiple remote ranks.
            # Logically "split" own regions into per-source chunks. Hybrid
            # MLA+SSM also needs this path: MLA is replicated and read once,
            # while the SSM state is sharded across every remote TP rank.
            # We only do this once per remote tp_size.
            self.src_xfer_handles_by_tp_ratio[tp_ratio] = []
            if self._mixed_mem_types:
                self._dram_src_handles_by_tp_ratio[tp_ratio] = []

            for handle_data in self._build_local_splits_from_plan(
                plan,
                self.src_blocks_data,
                self.num_descs,
            ):
                if self._mixed_mem_types:
                    assert self._desc_is_dram is not None
                    strided_is_dram = self._desc_is_dram[
                        self._strided_desc_starts(handle_data)
                    ]
                    if self._skip_dram_xfer:
                        dram_handle = None
                    else:
                        dram_handle = self.nixl_wrapper.prep_xfer_dlist(
                            "NIXL_INIT_AGENT",
                            handle_data[strided_is_dram],
                            mem_type="DRAM",
                        )
                    self._dram_src_handles_by_tp_ratio[tp_ratio].append(dram_handle)
                    handle_data = handle_data[~strided_is_dram]
                handle = self.nixl_wrapper.prep_xfer_dlist(
                    "NIXL_INIT_AGENT", handle_data, mem_type=self.nixl_memory_type
                )
                self.src_xfer_handles_by_tp_ratio[tp_ratio].append(handle)

        ### Register remote agent memory regions
        # With homogeneous TP, D pulls the whole kv cache from corresponding rank. With
        # heterogeneous TP, prepare the descriptors by splitting the P KV cache along
        # kv_head dim, of D worker's kv_head size (D>P).
        # Eg. PTP1 DTP2 => P0 KV:[block0-KV_0 | block0-KV_1..].

        # Register all remote blocks, but only the corresponding kv heads.
        blocks_data = self._build_fa_remote(plan, nixl_agent_meta)
        logger.debug(
            "Created %s blocks for dst engine %s with remote rank %s and local rank %s",
            blocks_data[:, 4].sum(),
            engine_id,
            remote_tp_rank,
            self.tp_rank,
        )
        if self._has_mamba:
            logger.debug(
                "Registering remote Mamba blocks for engine %s rank %s",
                engine_id,
                remote_tp_rank,
            )
            mamba = self._build_mamba_remote(nixl_agent_meta, tp_ratio)
            blocks_data = np.concatenate([blocks_data, mamba])

        remote_mem_types = set(self.dst_region_mem_types[engine_id])
        if len(remote_mem_types) != 1:
            raise NotImplementedError(
                "NIXL pull does not yet support a mixed-memory producer"
            )
        self.dst_xfer_side_handles[engine_id][remote_tp_rank] = (
            self.nixl_wrapper.prep_xfer_dlist(
                remote_agent_name, blocks_data, mem_type=remote_mem_types.pop()
            )
        )

        return remote_agent_name

    def _validate_remote_agent_handshake(
        self,
        nixl_agent_meta: NixlAgentMetadata,
        remote_tp_size: int,
        remote_dcp_size: int = 1,
    ):
        """Validate the remote agent handshake metadata ensuring the
        invariants hold true.
        """
        remote_engine_id = nixl_agent_meta.engine_id

        assert self.transfer_topo is not None
        remote_info = self.transfer_topo.get_engine_info(remote_engine_id)
        assert remote_info.remote_tp_size == remote_tp_size
        assert remote_info.remote_dcp_size == remote_dcp_size
        # DCP sizes must divide one another; this is what keeps the
        # read-slicing math in pull_worker a closed form.
        assert (
            self.dcp_size % remote_dcp_size == 0 or remote_dcp_size % self.dcp_size == 0
        ), (
            f"DCP sizes must divide one another: local={self.dcp_size}, "
            f"remote={remote_dcp_size} (engine {remote_engine_id})."
        )
        if self._has_mamba and self.dcp_size != remote_dcp_size:
            raise RuntimeError(
                "Hybrid MLA+Mamba NIXL transfers require matching DCP sizes, "
                f"got local={self.dcp_size}, remote={remote_dcp_size}."
            )

        tp_ratio = self.transfer_topo.tp_ratio(remote_tp_size)
        self._validate_transfer_groups(nixl_agent_meta)
        # num_kv_heads > tp_size with P_TP > D_TP not supported for non-mamba.
        # Mamba models can have replicated FA KV with tp_ratio < 0.
        # MLA models do not need to handle kv replication.
        if not self.use_mla and not self._has_mamba:
            assert not (
                tp_ratio < 0 and self.transfer_topo.is_kv_replicated(remote_engine_id)
            )
        has_draft_regions = bool(self._head_sharded_draft_regions)
        draft_local_heads = draft_remote_heads = 1
        if has_draft_regions:
            assert self._head_sharded_draft_kv_heads is not None
            draft_local_heads = max(
                1, self._head_sharded_draft_kv_heads // self.transfer_topo.tp_size
            )
            draft_remote_heads = max(
                1, self._head_sharded_draft_kv_heads // remote_tp_size
            )
            if tp_ratio < 0 or self.dcp_size > 1 or remote_dcp_size > 1:
                raise NotImplementedError(
                    "NIXL head-sharded draft KV under an MLA target requires "
                    "local TP >= remote TP and no DCP: "
                    f"local TP={self.transfer_topo.tp_size}, "
                    f"remote TP={remote_tp_size}, "
                    f"local DCP={self.dcp_size}, remote DCP={remote_dcp_size}, "
                    f"transfer_mode={self._TRANSFER_MODE}."
                )

        kv_cache_layout = (
            self.kv_cache_layout
            if not self.use_host_buffer
            else self.host_buffer_kv_cache_layout
        )
        if not self.use_mla and nixl_agent_meta.kv_cache_layout != kv_cache_layout:
            if (
                self.kv_transfer_config.enable_permute_local_kv
                and nixl_agent_meta.kv_cache_layout == "LBHNC"
                and kv_cache_layout == "LBNHC"
            ):
                logger.info(
                    "Remote is LBHNC and local is LBNHC, enabled additional permute "
                    "on local device KV."
                )
                assert not self._is_hma_required, (
                    "HMA does not support block size post processing"
                )
                self.enable_permute_local_kv = True
            else:
                raise RuntimeError(
                    "Heterogeneous TP expects same kv_cache_layout (local "
                    f"{kv_cache_layout}, remote {nixl_agent_meta.kv_cache_layout}). "
                    "Or enable experimental feature to receive LBHNC into a local "
                    "LBNHC cache by setting 'enable_permute_local_kv'=True "
                    "in --kv-transfer-config."
                )
        # if remote_agent used attn is not same as local,
        # hint heterogenuous attn post process
        if (
            nixl_agent_meta.attn_backend_name != self.backend_name
            and self.backend_name in ["CPU_ATTN"]
        ):
            if self._is_hma_required:
                raise RuntimeError(
                    "heterogeneous attn post process is not supported with HMA"
                )
            logger.info(
                "[Experimental] CPU_ATTN backend is used, "
                "hint heterogeneous attn post process"
            )
            self.enable_heterogeneous_attn_post_process = True

        # Heterogeneous TP requires head-splitting, which only works with
        # block-contiguous layouts (e.g. LBHNC). MLA and replicated-KV cases
        # don't split on heads. Mamba doesn't support heterogeneous TP.
        if (
            abs(tp_ratio) != 1
            and not self.use_mla
            and not self.transfer_topo.is_kv_replicated(remote_engine_id)
            and not KVCacheLayout[kv_cache_layout].is_block_contiguous
            and not self.enable_permute_local_kv
        ):
            raise RuntimeError(
                "Heterogeneous TP head-dimension splitting requires contiguous heads. "
                "Use a block-contiguous layout (e.g. LBHNC) on the prefill side."
            )
        # The MLA exemption above does not cover a head-sharded draft, which
        # slices the prefill draft block by head.
        if (
            has_draft_regions
            and draft_remote_heads != draft_local_heads
            and (
                nixl_agent_meta.kv_cache_layout != kv_cache_layout
                or not KVCacheLayout[kv_cache_layout].is_block_contiguous
            )
        ):
            raise RuntimeError(
                "Head-sharded draft KV under an MLA target needs the same "
                "block-contiguous kv_cache_layout (e.g. LBHNC) on prefill and "
                f"decode, got local={kv_cache_layout}, "
                f"remote={nixl_agent_meta.kv_cache_layout}."
            )

        # NVFP4 pages store all heads' data before all heads' scales, so a
        # head's bytes are not a contiguous slice of the page.
        if (
            not self.use_mla
            and self.transfer_topo.local_physical_heads
            != max(1, self.transfer_topo.total_num_kv_heads // remote_tp_size)
            and any(
                isinstance(spec, AttentionSpec) and spec.kv_quant_mode.is_nvfp4
                for spec in self._layer_specs.values()
            )
        ):
            raise RuntimeError(
                "NIXL KV-head re-sharding is not supported for NVFP4 KV cache "
                f"(local TP {self.transfer_topo.tp_size}, remote TP "
                f"{remote_tp_size}). Use matching TP sizes on prefill and decode."
            )

        # Per-region block_len validation enforcing the P/D invariant.
        # REPLICATE regions (MLA, or a whole-model MLA / replicated-KV transfer)
        # only allow the number of blocks to differ; SPLIT regions scale with
        # the per-rank KV head ratio rather than the raw tp_ratio, because GQA
        # replication caps per-rank heads at 1 when tp > total_kv_heads
        # (issue #45330). Mamba uses the ssm_sizes counterpart, so skip here.
        local_regions = self._transfer_layer_region_indices or range(
            len(self.block_len_per_layer)
        )
        if self._has_mamba and self.use_mla:
            # Hybrid MLA+SSM (e.g. KimiLinear's KDA+MLA): regions are
            # kernel-granularity views of the mamba-unified page. The MLA
            # per-token page is TP-independent, so the block lengths must
            # match even under heterogeneous TP.
            # SSM geometry is validated via ssm_sizes/conv offsets instead;
            # each SSM layer is its own region, whose length follows the
            # TP shard.
            regions = self._fa_region_indices()
            assert [self.block_len_per_layer[i] for i in regions] == [
                nixl_agent_meta.block_lens[i] for i in regions
            ], (
                "Hybrid MLA kernel-granularity block lengths must match "
                "between P and D: "
                f"local={self.block_len_per_layer}, "
                f"remote={nixl_agent_meta.block_lens}."
            )
        elif not self._has_mamba:
            assert len(local_regions) == len(nixl_agent_meta.block_lens), (
                "Number of KV layers must match between prefill and decode"
            )
            model_replicated = self.use_mla or self.transfer_topo.is_kv_replicated(
                remote_engine_id
            )
            total_kv_heads = self.transfer_topo.total_num_kv_heads
            for i, region_idx in enumerate(local_regions):
                local_len = self.block_len_per_layer[region_idx]
                remote_len = nixl_agent_meta.block_lens[i]
                if region_idx in self._head_sharded_draft_regions:
                    replicated = False
                    local_heads = draft_local_heads
                    remote_heads = draft_remote_heads
                else:
                    replicated = model_replicated or self._is_region_replicated(
                        region_idx
                    )
                    local_heads = self.transfer_topo.local_physical_heads
                    remote_heads = max(1, total_kv_heads // remote_tp_size)
                if replicated:
                    assert local_len == remote_len, (
                        "KV cache sizes must match between P and D when "
                        f"replicated (region {i}: local={local_len}, "
                        f"remote={remote_len})."
                    )
                else:
                    assert remote_len == local_len * remote_heads // local_heads, (
                        f"SPLIT region {i}: remote P KV block_len {remote_len} "
                        f"must equal local {local_len} * remote_heads "
                        f"{remote_heads} // local_heads {local_heads}."
                    )

        assert len(nixl_agent_meta.kv_caches_base_addr) == len(local_regions)
        num_remote_regions = len(nixl_agent_meta.kv_caches_base_addr)
        if nixl_agent_meta.region_num_blocks is not None:
            assert len(nixl_agent_meta.region_num_blocks) == num_remote_regions
        assert len(nixl_agent_meta.block_strides) == num_remote_regions
        assert all(
            stride >= block_len
            for stride, block_len in zip(
                nixl_agent_meta.block_strides,
                nixl_agent_meta.block_lens,
                strict=True,
            )
        )
        if nixl_agent_meta.region_group_ids is not None:
            assert len(nixl_agent_meta.region_group_ids) == num_remote_regions
        if nixl_agent_meta.region_names is not None:
            expected_names = (
                list(self._transfer_layer_names)
                if self._transfer_layer_names
                else self.region_names
            )
            assert nixl_agent_meta.region_names == expected_names, (
                "NIXL transfer regions must name the same model caches in the "
                "same order"
            )
        if nixl_agent_meta.region_mem_types is not None:
            assert len(nixl_agent_meta.region_mem_types) == num_remote_regions

    def _validate_transfer_groups(self, nixl_agent_meta: NixlAgentMetadata) -> None:
        """Per-group handshake invariants of the large-page KV pool."""
        # Per-group layer sets must match: a group's rows only carry the same
        # content when both sides pack the same layers into them. Whole-stage
        # PP gaps (one side holds none of the group's layers) pass; a
        # within-group subset mismatch is rejected.
        remote_layer_sets = nixl_agent_meta.group_layer_names
        num_groups = len(self._group_layer_names)
        if (
            len(remote_layer_sets) != num_groups
            or len(nixl_agent_meta.group_geometries) != num_groups
        ):
            raise RuntimeError(
                "NIXL peer advertised a different number of transfer groups: "
                f"local={num_groups}, remote={len(remote_layer_sets)} layer "
                f"sets and {len(nixl_agent_meta.group_geometries)} geometries."
            )
        for group_id, (local, remote) in enumerate(
            zip(self._group_layer_names, remote_layer_sets, strict=True)
        ):
            if local and remote and local != remote:
                raise RuntimeError(
                    f"NIXL layer-set mismatch in transfer group "
                    f"{group_id}: local={local}, remote={remote}."
                )

    def _group_kv_views(
        self, group_id: int, caches: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """The ``caches`` views owned by transfer group ``group_id``."""
        return {
            name: caches[name]
            for name in self.kv_cache_config.transfer_groups[group_id].layer_names
            if name in caches
        }

    def sync_recved_kv_to_device(self, req_id: str, meta: ReqMeta):
        """Copy recved kv from host buffer to device."""
        assert self.use_host_buffer
        assert self.copy_blocks is not None

        # The h2d block copies below are intentionally synchronous. Block ids
        # are unique only within a group, so each group's layers copy with
        # that group's own kernel block ids.
        with gpu_sync_allowed():
            for group_id, block_ids in enumerate(meta.local_physical_block_ids):
                if not block_ids:
                    continue
                self.copy_blocks(
                    self._group_kv_views(group_id, self.host_xfer_buffers),
                    self._group_kv_views(group_id, self.device_kv_caches),
                    block_ids,
                    block_ids,
                    "h2d",
                )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "synced recved kv of request[%s] to device kv buffer,"
                "local_block_ids: %s. ",
                req_id,
                ",".join(map(str, meta.local_physical_block_ids)),
            )

    def save_kv_to_host(self, metadata: NixlConnectorMetadata):
        """Copy kv from device to host buffer."""
        assert self.use_host_buffer
        assert self.copy_blocks is not None

        # The d2h block copies below are intentionally synchronous.
        with gpu_sync_allowed():
            for req_id, meta in metadata.reqs_to_save.items():
                meta.local_physical_block_ids = self._logical_to_kernel_block_ids(
                    meta.local_block_ids
                )
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "save_load_kv for request[%s] to host xfer buffer."
                        "local_block_ids: %s. ",
                        req_id,
                        ",".join(map(str, meta.local_physical_block_ids)),
                    )
                # blocking; per group, as in sync_recved_kv_to_device.
                for group_id, block_ids in enumerate(meta.local_physical_block_ids):
                    if not block_ids:
                        continue
                    self.copy_blocks(
                        self._group_kv_views(group_id, self.device_kv_caches),
                        self._group_kv_views(group_id, self.host_xfer_buffers),
                        block_ids,
                        block_ids,
                        "d2h",
                    )

    def post_process_device_kv_on_receive(
        self,
        block_ids_list: list[tuple[int, list[int], int]],
        convert: bool = True,
    ):
        """Post process device kv cache after receiving from remote.

        ``convert`` permutes the received blocks from HND to NHD when the
        remote sent a different layout (``enable_permute_local_kv``).

        The transfer only covers the first ``covered_blocks`` blocks of each
        request's local attention blocks; the rest was clipped by the front
        trim in ``_apply_prefix_caching``. Those blocks were excluded from the
        scheduler's alloc-time KV zeroing (which would race the RDMA write),
        so everything past the covered range is zeroed here. Stale bytes
        would otherwise surface as garbage or NaNs once decode grows into the
        untransferred tail.
        """
        if len(self.device_kv_caches) == 0:
            return
        assert self.transfer_topo is not None
        if convert:
            logger.debug(
                "Post-processing device kv cache on receive by permuting layout"
                "from HND to NHD."
            )
        else:
            logger.debug(
                "Post-processing device kv cache on receive by zeroing "
                "untransferred blocks."
            )

        for group, block_ids, covered_blocks in block_ids_list:
            attn_caches = [
                self.device_kv_caches[name]
                for name in self.kv_cache_config.transfer_groups[group].layer_names
            ]
            # Blocks the transfer didn't write: everything past the covered
            # range.
            has_stale = covered_blocks < len(block_ids)
            indices = None
            if convert or has_stale:
                indices = async_tensor_h2d(block_ids, attn_caches[0].device, torch.long)

            if convert:
                for cache in attn_caches:
                    kv_postprocess_layout_on_receive(cache, indices)
            if has_stale:
                assert indices is not None
                stale_ids = indices[covered_blocks:]
                for cache in attn_caches:
                    cache.index_fill_(0, stale_ids, 0)

    def post_process_device_kv_on_receive_heterogeneous_attn(
        self, block_ids: list[int]
    ):
        """Post process device kv cache after receiving from remote
        for heterogeneous attention.
        """
        assert self.enable_heterogeneous_attn_post_process

        indices = torch.tensor(block_ids, device=self.device_type, dtype=torch.long)

        for _, cache_or_caches in self.device_kv_caches.items():
            current_platform.pack_kv_cache(
                kv_cache=cache_or_caches,
                indices=indices,
            )

    def _leaves_dram_xfer_to_rank0(self) -> bool:
        """Whether this rank leaves the DRAM regions' NIXL work to TP rank 0.

        A HiSparse host pool shared by the local TP ranks (one mmap per DP
        rank) is registered with NIXL and filled by TP rank 0 alone: every
        rank would otherwise register the same pages on every NIC and READ the
        same host KV into them. Private pools are unaffected.
        """
        if not (
            self._mixed_mem_types and self.kv_cache_config.hisparse_shared_host_pool
        ):
            return False
        skip = self.tp_rank != 0
        logger.info(
            "HiSparse host pool: NIXL DRAM registration and reads on TP rank 0 "
            "only (tp_rank=%d, skip=%s)",
            self.tp_rank,
            skip,
        )
        return skip

    def _zero_region_blocks(self, block_ids: BlockIds) -> None:
        """Clear clipped physical pages in their owning region only."""
        if not any(block_ids):
            return
        bases = self.kv_caches_base_addr[self.engine_id][self.tp_rank]
        for region, blocks in enumerate(block_ids):
            if not blocks:
                continue
            if self._skip_dram_xfer and self.region_mem_types[region] == "DRAM":
                # The shared pool is rank 0's to write.
                continue
            cache = self.device_kv_caches[self.region_names[region]]
            storage = cache.untyped_storage()
            pages = torch.empty(0, dtype=torch.uint8, device=cache.device).set_(
                storage,
                bases[region] - storage.data_ptr(),
                (self.region_num_blocks[region], self.block_len_per_layer[region]),
                (self.block_stride_per_layer[region], 1),
            )
            for block in blocks:
                pages[block].zero_()

    def get_transfer_results(self) -> KVConnectorTransferResults:
        """Get transfers that completed on this specific worker.

        The scheduler process (via the MultiprocExecutor) will use this output
        to track which workers are done.
        """
        assert self.transfer_topo is not None
        done_sending = self._get_new_notifs()
        done_recving, newly_failed = self._pop_done_transfers(self._recving_transfers)
        if newly_failed:
            self._recv_failures.update(newly_failed)

        done_sending.update(self._replicated_pcp_done_sending)
        self._replicated_pcp_done_sending.clear()

        # Process receive failures reported by background threads. Each
        # producer already recorded a specific metric (handshake or
        # notification), so only the failure bookkeeping runs here.
        while not self._failed_recv_reqs.empty():
            try:
                req_id = self._failed_recv_reqs.get_nowait()
            except queue.Empty:
                break
            self._handle_failed_transfer(
                req_id, None, self._recv_failures, record_failed_transfer=False
            )

        failed_recv_reqs: set[ReqId] = set()
        if self._recv_failures:
            # Keep failure bookkeeping independent of healthy request count.
            for req_id in tuple(self._recv_failures):
                if req_id not in self._recving_metadata:
                    self._recv_failures.remove(req_id)
                elif req_id not in self._recving_transfers:
                    self._recv_failures.remove(req_id)
                    failed_recv_reqs.add(req_id)
            done_recving.update(failed_recv_reqs)

        if len(done_sending) > 0 or len(done_recving) > 0:
            logger.debug(
                "Rank %s, get_finished: %s requests done sending "
                "and %s requests done recving (%s failed)",
                self.tp_rank,
                len(done_sending),
                len(done_recving),
                len(failed_recv_reqs),
            )

        block_ids_for_post_process = list[tuple[int, list[int], int]]()
        block_ids_for_heterogeneous_attn_post_process = list[list[int]]()
        direct_device_recving = set[str]()
        for req_id in done_recving:
            # clean up metadata for completed requests
            meta = self._recving_metadata.pop(req_id, None)
            assert meta is not None, f"{req_id} not found in recving_metadata list"

            # Skip KV sync and post-processing for failed requests
            if req_id in failed_recv_reqs:
                self._pending_recv_notifs.pop(req_id, None)
                # TODO (NickLucche) handle failed transfer for HMA.
                # Block ids are unique only within a group; tag them.
                # local_block_ids is transfer-group space; the scheduler
                # indexes the full kv_cache_groups space.
                transfer_group_ids = self.kv_cache_config.transfer_group_ids
                self._invalid_block_ids.put(
                    {
                        (transfer_group_ids[g], block_id)
                        for g, group in enumerate(meta.local_block_ids)
                        for block_id in group
                    }
                )
                logger.warning(
                    "Skipping KV post-processing for failed request %s",
                    req_id,
                )
                continue

            self._send_pending_recv_notifs(req_id)
            assert meta.remote is not None
            if self.use_host_buffer:
                self.sync_recved_kv_to_device(req_id, meta)

            direct_device_recving.add(req_id)

            if meta.region_blocks_to_zero is not None:
                # P/D group positions differ. Use the actual region read plan,
                # including any local allocation padding the read did not cover.
                self._zero_region_blocks(meta.region_blocks_to_zero)
                continue

            # Post processing for layout conversion, and for blocks the
            # transfer clipped at kernel-block granularity when
            # _apply_prefix_caching front-trimmed to the minimum count
            # (hybrid heterogeneous TP).
            if self.enable_permute_local_kv:
                for g, local_group in enumerate(meta.local_physical_block_ids):
                    if not local_group or _is_ssm_spec(self._group_spec_types[g]):
                        continue
                    # Number of blocks the transfer covered; everything past
                    # this was clipped and must be zeroed.
                    covered_blocks = min(
                        len(local_group),
                        len(meta.remote.block_ids[g]),
                    )
                    block_ids_for_post_process.append((g, local_group, covered_blocks))
            # post processing for heterogeneous attention
            if self.enable_heterogeneous_attn_post_process:
                block_ids_for_heterogeneous_attn_post_process.append(
                    meta.local_physical_block_ids[0]
                )
        if block_ids_for_post_process:
            # MLA never needs the layout conversion, but its clipped blocks
            # still need zeroing.
            convert = not self.use_mla and self.enable_permute_local_kv
            self.post_process_device_kv_on_receive(block_ids_for_post_process, convert)

        for block_ids in block_ids_for_heterogeneous_attn_post_process:
            self.post_process_device_kv_on_receive_heterogeneous_attn(block_ids)

        self._sync_device_after_direct_recv(direct_device_recving)

        # Handle timeout to avoid stranding blocks on remote.
        self._reap_expired_send_leases(done_sending)

        self._cleanup_remote_engines()
        return KVConnectorTransferResults(
            finished_sending=done_sending,
            finished_recving=done_recving,
            failed_recving=failed_recv_reqs,
        )

    def get_finished(self) -> tuple[set[str], set[str]]:
        """Compatibility wrapper for the legacy completion API."""
        results = self.get_transfer_results()
        return results.finished_sending, results.finished_recving

    def _sync_device_after_direct_recv(self, done_recving: set[str]) -> None:
        """Make direct NIXL writes visible before model execution."""
        requires_sync = current_platform.is_rocm() and self._has_mamba
        if self.use_host_buffer or not done_recving or not requires_sync:
            return

        torch.accelerator.synchronize()

    def _get_new_notifs(self) -> set[str]:
        """Get req_ids which got a remote xfer notification.

        Subclasses must implement this to handle mode-specific notifications.
        """
        raise NotImplementedError

    def _reap_expired_send_leases(self, done_sending: set[str]) -> None:
        """Reclaim expired send-side KV leases into ``done_sending``.

        ``_reqs_to_send`` is not ordered by expiry: heartbeats update the
        deadline in place, and mixed TTLs share the map, so a live head
        entry can sit in front of already-expired ones. Scan every entry
        rather than stopping at the first still-live request.
        """
        if not self._reqs_to_send:
            return
        now = time.perf_counter()
        expired = [
            req_id for req_id, expires in self._reqs_to_send.items() if now >= expires
        ]
        for req_id in expired:
            count = self.consumer_notification_counts_by_req.pop(req_id, 0)
            self.expected_consumer_notifications_by_req.pop(req_id, None)
            self.xfer_stats.record_kv_expired_req()
            logger.warning(
                "Releasing expired KV blocks for request %s which were "
                "retrieved by %d remote worker(s) before lease expired.",
                req_id,
                count,
            )
            self._reqs_to_process.remove(req_id)
            del self._reqs_to_send[req_id]
            done_sending.add(req_id)

    def _handle_heartbeat(self, payload: str) -> None:
        """Extend leases for requests referenced in a heartbeat.

        Args:
            payload: comma-separated P-side request IDs, e.g.
                     "req_abc,req_def".

        """
        new_expiry = time.perf_counter() + self._lease_extension
        for req_id in payload.split(","):
            if req_id in self._reqs_to_send:
                old = self._reqs_to_send[req_id]
                self._reqs_to_send[req_id] = max(old, new_expiry)
                logger.debug(
                    "Heartbeat extended lease for request %s "
                    "by %ds (old_expiry=%.1f, new_expiry=%.1f)",
                    req_id,
                    self._lease_extension,
                    old,
                    new_expiry,
                )

    def _pop_done_transfers(
        self, transfers: dict[str, list[int]]
    ) -> tuple[set[str], set[str]]:
        """Poll transfers, retaining handles until they can be released.

        Args:
            transfers: Outstanding handles by request, updated in place.

        Returns:
            Requests with no outstanding handles and requests with observed
            failures. Failed requests may still have outstanding handles.

        """
        done_req_ids: set[str] = set()
        failed_req_ids: set[str] = set()
        for req_id, handles in list(transfers.items()):
            in_progress = []
            for handle in handles:
                try:
                    xfer_state = self.nixl_wrapper.check_xfer_state(handle)
                    if xfer_state == "DONE":
                        res = self.nixl_wrapper.get_xfer_telemetry(handle)
                        self.xfer_stats.record_transfer(res)
                        self.nixl_wrapper.release_xfer_handle(handle)
                    elif xfer_state == "PROC":
                        in_progress.append(handle)
                    else:
                        self._log_failure(
                            failure_type="transfer_failed",
                            req_id=req_id,
                            xfer_state=xfer_state,
                        )
                        if not self._handle_failed_transfer(
                            req_id, handle, failed_req_ids
                        ):
                            in_progress.append(handle)
                except Exception as e:
                    self._log_failure(
                        failure_type="transfer_exception",
                        req_id=req_id,
                        error=e,
                    )
                    if not self._handle_failed_transfer(req_id, handle, failed_req_ids):
                        in_progress.append(handle)

            if not in_progress:
                done_req_ids.add(req_id)
                del transfers[req_id]
            else:
                transfers[req_id] = in_progress
        return done_req_ids, failed_req_ids

    def _try_release_xfer_handle(self, req_id: str, handle: int) -> bool:
        """Release a handle, returning False if the caller must retain it."""
        try:
            self.nixl_wrapper.release_xfer_handle(handle)
        except Exception as e:
            # A status error does not guarantee that the backend stopped DMA.
            self._log_failure(
                failure_type="transfer_release_failed",
                msg="Retaining handle and blocks until release succeeds",
                req_id=req_id,
                error=e,
            )
            return False
        return True

    def _handle_failed_transfer(
        self,
        req_id: str,
        handle: int | None,
        failed_req_ids: set[str] | None = None,
        record_failed_transfer: bool = True,
    ) -> bool:
        """Record a failure and release its handle, returning False to retain it.

        Args:
            req_id: The request ID.
            handle: The transfer handle.
            failed_req_ids: Requests observed as failed; each stays listed
                until it has no outstanding handles.
            record_failed_transfer: Whether to count the failure toward the
                transport-failure metric. Callers that already recorded a more
                specific metric (eg KV expiry, reported separately from
                transport failures) pass False.

        """
        if record_failed_transfer:
            self.xfer_stats.record_failed_transfer()
            if (
                self._TRANSFER_MODE == "pull"
                and (meta := self._recving_metadata.get(req_id)) is not None
                and meta.remote is not None
            ):
                self._failed_remote_engines.add(meta.remote.engine_id)
        if failed_req_ids is not None:
            failed_req_ids.add(req_id)
        return handle is None or self._try_release_xfer_handle(req_id, handle)

    def _send_pending_recv_notifs(self, req_id: str) -> None:
        """Send notifications deferred by split DRAM/VRAM reads."""
        for agent_name, notif_id in self._pending_recv_notifs.pop(req_id, []):
            try:
                self.nixl_wrapper.send_notif(agent_name, notif_msg=notif_id)
            except Exception as e:
                self._log_failure(
                    failure_type="notification_failed",
                    msg="P worker blocks will be freed after timeout. "
                    "This may indicate network issues.",
                    req_id=req_id,
                    error=e,
                    remote_agent_name=agent_name,
                )
                self.xfer_stats.record_failed_notification()

    def _send_heartbeats(self, metadata: NixlConnectorMetadata) -> None:
        """Send heartbeats to remote engines, extending the lease on KV blocks."""
        for engine_id, hb_info in metadata.heartbeat_by_engine.items():
            # Proactive handshake (this request may still be in waiting queue) so
            # the **next** heartbeat for this remote can go through.
            if (
                self._ensure_handshake(
                    engine_id,
                    hb_info.host,
                    hb_info.port,
                    hb_info.tp_size,
                    hb_info.dcp_size,
                    hb_info.pp_size,
                    self._hb_handshake_notif_only and hb_info.pp_size > 1,
                )
                is not None
            ):
                continue  # handshake is still pending

            # Refresh the TTL: a heartbeat means the router still paired this P-D,
            # so requests are still waiting on this engine
            self._engine_last_active[engine_id] = time.perf_counter()
            # Build the heartbeat message: "HB:req1,req2,..."
            hb_msg = ("HB:" + ",".join(hb_info.req_ids)).encode()
            for agent_name in self._remote_agents[engine_id].values():
                try:
                    self.nixl_wrapper.send_notif(agent_name, notif_msg=hb_msg)
                except Exception:
                    logger.debug(
                        "Failed to send heartbeat to engine %s",
                        engine_id,
                        exc_info=True,
                    )

    @staticmethod
    def _expand_block_ids(
        block_ids: BlockIds, geometries: list[GroupGeometry]
    ) -> BlockIds:
        """Expand per-group block ids to kernel block ids."""
        if not block_ids:  # A request aborted before it was ever scheduled.
            return block_ids
        return [
            geometry.rows(ids).tolist()
            for geometry, ids in zip(geometries, block_ids, strict=True)
        ]

    def _remote_logical_to_kernel_block_ids(
        self, block_ids: BlockIds, remote_engine_id: str
    ) -> BlockIds:
        """``_logical_to_kernel_block_ids`` for a peer's block ids, with the
        peer's advertised pool geometry."""
        return self._expand_block_ids(
            block_ids, self._dst_group_geometries[remote_engine_id]
        )

    def _logical_to_kernel_block_ids(self, block_ids: BlockIds) -> BlockIds:
        """Expand the engine's per-group block ids to kernel block ids."""
        return self._expand_block_ids(
            block_ids, self.kv_cache_config.transfer_group_geometries
        )

    def _apply_dcp_prefix_caching(
        self,
        local_ids: list[int],
        remote_ids: list[int],
        remote_rank: int,
        local_dcp_size: int,
        local_dcp_rank: int,
        remote_dcp_size: int,
        local_num_computed_blocks: int,
        num_remote_blocks: int | None = None,
    ) -> tuple[list[int], list[int]]:
        """Match local and remote block IDs by global DCP position.

        ``local_ids`` excludes blocks already satisfied by the local prefix
        cache, while ``remote_ids`` contains the full transferable remote list.
        ``num_remote_blocks``, when provided, is the global block count before
        DCP partitioning and excludes allocation padding.

        Example:
            local DCP size 2, rank 0 owns *global positions*=[0, 2, 4, 6]
            cached_blocks=[0,2,4]
            to transfer: [6] (the rest is cached)

            remote DCP size 4
            remote rank 0 owns [0, 4] -> no more match, skip the read
            remote rank 2 owns [2, 6] -> match position 6 and read

        An empty result (rank 0 above) takes the notification-only path, allowing the
        remote to release its blocks without performing a transfer.

        """
        local_size, remote_size = local_dcp_size, remote_dcp_size
        if num_remote_blocks is not None:
            num_local = cdiv(num_remote_blocks - local_dcp_rank, local_size)
            local_ids = local_ids[: max(0, num_local - local_num_computed_blocks)]

        if local_size == remote_size:
            local_slice = local_ids
            remote_slice = remote_ids[local_num_computed_blocks:]
        elif local_size < remote_size:
            k = remote_size // local_size
            p = (remote_rank - local_dcp_rank) // local_size
            start_local = (p - local_num_computed_blocks) % k
            start_remote = (local_num_computed_blocks + start_local - p) // k
            local_slice = local_ids[start_local::k]
            remote_slice = remote_ids[start_remote:]
        else:
            k = local_size // remote_size
            remote_dcp_rank = remote_rank % remote_size
            c = (local_dcp_rank - remote_dcp_rank) // remote_size
            start_remote = c + local_num_computed_blocks * k
            local_slice = local_ids
            remote_slice = remote_ids[start_remote::k]

        matched_blocks = min(len(local_slice), len(remote_slice))
        if num_remote_blocks is not None and matched_blocks < len(local_slice):
            raise ValueError("Remote KV pages do not cover the requested range")
        return local_slice[:matched_blocks], remote_slice[:matched_blocks]

    @staticmethod
    def _block_ids_by_region(
        block_ids: BlockIds, region_group_ids: list[int]
    ) -> BlockIds:
        """Expand group block IDs into the corresponding region order."""
        return [list(block_ids[group_id]) for group_id in region_group_ids]

    def _apply_prefix_caching_by_region(
        self,
        decode_block_ids: BlockIds,
        prefill_block_ids: BlockIds,
        *,
        num_computed_blocks: list[int] | None = None,
        num_remote_blocks: int | None = None,
        remote_rank: int = 0,
        remote_dcp_size: int = 1,
    ) -> tuple[BlockIds, BlockIds]:
        """Pair an uncached decode suffix with the same prefill regions."""
        assert len(decode_block_ids) == len(prefill_block_ids)
        if not any(decode_block_ids):
            empty_regions: list[list[int]] = [[] for _ in decode_block_ids]
            return empty_regions, empty_regions.copy()

        if num_computed_blocks is not None:
            assert num_remote_blocks is not None
            matched_decode, matched_prefill = [], []
            for decode_region, prefill_region, start in zip(
                decode_block_ids, prefill_block_ids, num_computed_blocks, strict=True
            ):
                local, remote = self._apply_dcp_prefix_caching(
                    decode_region,
                    prefill_region,
                    remote_rank,
                    self.dcp_size,
                    self.dcp_rank,
                    remote_dcp_size,
                    start,
                    num_remote_blocks=num_remote_blocks,
                )
                matched_decode.append(local)
                matched_prefill.append(remote)
            return matched_decode, matched_prefill

        trimmed_prefill: list[list[int]] = []
        for decode_region, prefill_region in zip(
            decode_block_ids, prefill_block_ids, strict=True
        ):
            if len(decode_region) > len(prefill_region):
                raise ValueError(
                    "Decode allocated more KV pages than the prefill worker supplied"
                )
            trimmed_prefill.append(
                list(prefill_region[-len(decode_region) :]) if decode_region else []
            )
        return decode_block_ids, trimmed_prefill

    def _apply_prefix_caching(
        self,
        decode_block_ids: BlockIds,
        prefill_block_ids: BlockIds,
    ) -> tuple[BlockIds, BlockIds]:
        """Trim block ID lists so only the uncomputed suffix is transferred.

        Inputs are *kernel* block IDs, already expanded from manager block
        IDs with each side's pool geometry. Both pull and push call this after
        that expansion so the trim happens at kernel granularity.

        The prefix-cache hit is always on the decode (D) side, so ``decode``
        holds only its uncomputed blocks while prefill (P) holds the full
        sequence. This is mode-independent: pull passes its own (D) blocks as
        ``decode``, push passes the remote D registration as ``decode``.

        For non-Mamba models: end-trim ``prefill`` to match ``decode`` count, so
        already-cached prefix blocks are skipped in the transfer.

        For Mamba hybrid: SSM groups pair state slots by position and FA
        groups end-trim to the uncomputed suffix.
        """
        # Partial prefix cache hit: just transfer uncomputed blocks.
        # Skip mamba groups — their blocks represent full state (conv+ssm),
        # not per-token data, so trimming would corrupt the transfer.
        prefill_block_ids = list(prefill_block_ids)
        if not self._has_mamba:
            for i, prefill_group in enumerate(prefill_block_ids):
                num_decode_blocks = len(decode_block_ids[i])
                assert num_decode_blocks <= len(prefill_group)
                if num_decode_blocks < len(prefill_group):
                    prefill_block_ids[i] = prefill_group[-num_decode_blocks:]
        else:
            # (NOTE: ZhanqiuHu) The two sides can reserve a different number
            # of trailing kernel blocks per request (admission padding on
            # prefill; the recomputed final token on decode).
            decode_block_ids = list(decode_block_ids)
            for i, prefill_group in enumerate(prefill_block_ids):
                num_decode_blocks = len(decode_block_ids[i])
                num_prefill_blocks = len(prefill_group)
                if _is_ssm_spec(self._group_spec_types[i]):
                    if num_decode_blocks == num_prefill_blocks:
                        continue
                    # Only state-bearing slots reach here, normally just one
                    # (see get_exchange_clipped_blocks). A longer prefill list
                    # carries earlier positions the decode side already has
                    # (prefix hit) -> take its tail; a longer decode list holds
                    # the position D recomputes itself, which gets no prefill
                    # state.
                    assert num_decode_blocks - num_prefill_blocks <= 1, (
                        f"Group {i}: unpairable SSM state slots, "
                        f"decode={num_decode_blocks} prefill={num_prefill_blocks}"
                    )
                    num_blocks = min(num_decode_blocks, num_prefill_blocks)
                    if num_decode_blocks < num_prefill_blocks:
                        prefill_block_ids[i] = prefill_group[-num_blocks:]
                    else:
                        decode_block_ids[i] = decode_block_ids[i][:num_blocks]
                elif num_decode_blocks < num_prefill_blocks:
                    # Partial prefix cache hit for FA group.
                    prefill_block_ids[i] = prefill_group[-num_decode_blocks:]
                else:
                    # Each side pads by at most one kernel block (the
                    # recomputed final token on decode; the admission pad on
                    # prefill); anything larger indicates mismatched lists.
                    assert abs(num_decode_blocks - num_prefill_blocks) <= 2, (
                        f"Group {i}: |{num_decode_blocks} - {num_prefill_blocks}| > 2"
                    )
                    num_blocks = min(num_decode_blocks, num_prefill_blocks)
                    decode_block_ids[i] = decode_block_ids[i][:num_blocks]
                    prefill_block_ids[i] = prefill_group[:num_blocks]
        return decode_block_ids, prefill_block_ids

    def get_kv_connector_stats(self) -> KVConnectorStats | None:
        """Get the KV transfer stats for the connector."""
        # Clear stats for next iteration
        if not self.xfer_stats.is_empty():
            return self.xfer_stats.clone_and_reset()
        return None

    def get_block_ids_with_load_errors(self) -> set[tuple[int, int]]:
        """Return and clear the set of block IDs that failed to load.

        This is called by the scheduler to identify blocks that need
        to be retried after a NIXL transfer failure. Block ids are unique
        only within a group, so each entry is a ``(group_id, block_id)`` pair.
        """
        # Drain the queue (thread-safe, no lock needed).
        result: set[Any] = set()
        while not self._invalid_block_ids.empty():
            try:
                result.update(self._invalid_block_ids.get_nowait())
            except queue.Empty:
                break
        return result

    def _evict_stale_engines(self) -> None:
        """Scan for and evict remote engines that have exceeded their TTL.

        Called from the main thread in when a new remote engine appears.
        We can only go OOM as we discover and register a new remote, therefore we make
        sure we clean up stale engine data structures before then. This invariant
        prevents us from using background threads, though memory usage is not guaranteed
        to be "optimal" until a new handshake is performed.

        Engines with active transfers, heartbeats or pending handshakes cannot
        be stale:
        - Reads stamp _engine_last_active when they are issued, and engines a
          transfer is still reading from are held back explicitly, since the
          stamp is not refreshed while the read runs.
        - Heartbeats stamp it too, so an engine stays registered while this
          instance holds requests waiting on it.
        - Pending handshakes don't have an _engine_last_active entry yet
        """
        # NOTE (NickLucche): This does NOT currently prevent OOMing if a huge number
        # of remote engines is registered all at once (adding a background cleanup
        # thread wouldnt help either).
        # If that scenario is plausible, we can follow up with an LRU eviction policy.
        if self._engine_ttl <= 0:
            return

        now = time.perf_counter()
        busy = self._engines_with_inflight_transfers()
        for eid, last_active in list(self._engine_last_active.items()):
            if now - last_active > self._engine_ttl and eid not in busy:
                self._cleanup_remote_engine(eid)

    def _engines_with_inflight_transfers(self) -> set[EngineId]:
        """Remote engines a transfer is still reading from.

        The timestamp is stamped when a read is issued and not refreshed while
        it runs, so a transfer that outlives the TTL leaves its engine looking
        idle. A peer that has lost its NIC holds one indefinitely.
        """
        return {
            meta.remote.engine_id
            for req_id in self._recving_transfers
            if (meta := self._recving_metadata.get(req_id)) is not None
            and meta.remote is not None
        }

    def _track_remote_engine_replacement(
        self, engine_id: EngineId, host: str, port: int
    ) -> None:
        """Record the engine currently serving a pull peer address."""
        # TODO: Also handle push mode, which handshakes in both directions.
        if self._TRANSFER_MODE != "pull":
            return
        previous = self._engine_by_address.get((host, port))
        self._engine_by_address[(host, port)] = engine_id
        if previous is None or previous == engine_id:
            return
        logger.info(
            "Remote engine at %s:%d changed from %s to %s.",
            host,
            port,
            previous,
            engine_id,
        )

    def _cleanup_remote_engines(self) -> None:
        """Release replaced or locally invalid pull peers after their reads drain."""
        # TODO: Also handle push mode, which handshakes in both directions.
        if self._TRANSFER_MODE != "pull":
            return
        # A background handshake may be loading native metadata with the GIL
        # released. Defer both probing and cleanup until it has finished.
        with self._handshake_lock:
            if self._handshake_futures:
                return
            replaced = self._remote_agents.keys() - self._engine_by_address.values()
        now = time.perf_counter()
        for engine_id in self._failed_remote_engines - self._invalid_remote_engines:
            agents = self._remote_agents.get(engine_id, {}).values()
            try:
                # No descriptors: query local existence, not peer health.
                if not all(map(self.nixl_wrapper.check_remote_metadata, agents)):
                    self._mark_remote_engine_invalid(engine_id, now)
            except Exception:
                logger.warning(
                    "Could not check local NIXL metadata for engine %s.",
                    engine_id,
                    exc_info=True,
                )
        self._failed_remote_engines.clear()
        # Invalid peers keep rejecting new reads until their backoff has elapsed.
        invalid = {
            engine_id
            for engine_id in self._invalid_remote_engines
            if self._invalid_engine_backoff.get(engine_id, (0.0, 0.0))[1] <= now
        }
        if not replaced and not invalid:
            return
        # _recving_metadata is only accessed from this (main) thread.
        busy = {
            meta.remote.engine_id
            for meta in self._recving_metadata.values()
            if meta.remote is not None
        }
        # Replaced peers can still serve queued requests. Invalid peers reject
        # new reads, but must retain descriptors until outstanding handles end.
        invalid -= self._engines_with_inflight_transfers()
        for engine_id in (replaced - busy) | invalid:
            self._cleanup_remote_engine(engine_id, log_eviction=False)
            logger.info(
                "Released NIXL state for replaced or locally invalid remote engine %s.",
                engine_id,
            )

    def _mark_remote_engine_invalid(self, engine_id: EngineId, now: float) -> None:
        """Schedule release of an engine whose local NIXL metadata is gone.

        The first invalidation is released as soon as reads drain. Repeats
        before the peer has stayed healthy for the max backoff double the
        delay, during which reads fail fast instead of re-handshaking.
        """
        # Forget engines that have stayed healthy long enough.
        for eid, (_, release_at) in list(self._invalid_engine_backoff.items()):
            if now - release_at > _INVALID_PEER_MAX_BACKOFF_S:
                del self._invalid_engine_backoff[eid]
        prev = self._invalid_engine_backoff.get(engine_id)
        backoff = 0.0
        if prev is not None:
            backoff = min(
                max(2 * prev[0], _INVALID_PEER_MIN_BACKOFF_S),
                _INVALID_PEER_MAX_BACKOFF_S,
            )
        self._invalid_engine_backoff[engine_id] = (backoff, now + backoff)
        self._invalid_remote_engines.add(engine_id)
        logger.warning(
            "Local NIXL metadata for remote engine %s was invalidated; "
            "re-handshaking after reads drain%s.",
            engine_id,
            f" and a {backoff:.0f}s backoff" if backoff else "",
        )

    def _cleanup_remote_engine(
        self, engine_id: EngineId, *, log_eviction: bool = True
    ) -> None:
        """Remove all state for a single remote engine.

        Releases NIXL resources (dlist handles, remote agents) and clears
        all per-engine data structures. Used by TTL eviction, replaced/invalid
        peer cleanup and shutdown.
        """
        assert engine_id in self._remote_agents

        # Notif-only engines (push-mode D side) have no descriptor state.
        for handle in self.dst_xfer_side_handles.pop(engine_id, {}).values():
            self.nixl_wrapper.release_dlist_handle(handle)
        # Pop under the handshake lock; NIXL teardown stays outside it.
        with self._handshake_lock:
            agents = self._remote_agents.pop(engine_id)
            for address, eid in list(self._engine_by_address.items()):
                if eid == engine_id:
                    del self._engine_by_address[address]
        for agent_name in agents.values():
            self.nixl_wrapper.remove_remote_agent(agent_name)

        self._failed_remote_engines.discard(engine_id)
        self._invalid_remote_engines.discard(engine_id)

        self.kv_caches_base_addr.pop(engine_id, None)
        self.dst_region_num_blocks.pop(engine_id, None)
        self.dst_region_group_ids.pop(engine_id, None)
        self.dst_region_mem_types.pop(engine_id, None)
        self._dst_group_geometries.pop(engine_id, None)
        self.tp_mappings.pop(engine_id, None)
        self.draft_tp_mappings.pop(engine_id, None)
        if self.transfer_topo is not None:
            self.transfer_topo.unregister_remote_engine(engine_id)

        # Drop the cached clock offset; it is re-measured on the next handshake.
        self._engine_clock_offset.pop(engine_id, None)
        # A just-completed handshake may not have recorded activity yet, so
        # tolerate a missing entry.
        last_active = self._engine_last_active.pop(engine_id, None)
        if log_eviction and last_active is not None:
            logger.info(
                "Evicted stale remote engine %s (inactive for %.1fs).",
                engine_id,
                time.perf_counter() - last_active,
            )

    def __del__(self):
        with contextlib.suppress(Exception):
            self.shutdown()

    def _release_local_xfer_handles(self) -> None:
        try:
            if self.src_xfer_handle is not None:
                self.nixl_wrapper.release_dlist_handle(self.src_xfer_handle)
            for handles in self.src_xfer_handles_by_tp_ratio.values():
                for handle in handles:
                    self.nixl_wrapper.release_dlist_handle(handle)
            for dram_handles in self._dram_src_handles_by_tp_ratio.values():
                for dram_handle in dram_handles:
                    if dram_handle is not None:
                        self.nixl_wrapper.release_dlist_handle(dram_handle)
            if self._dram_src_handle is not None:
                self.nixl_wrapper.release_dlist_handle(self._dram_src_handle)
        finally:
            self.src_xfer_handle = None
            self.src_xfer_handles_by_tp_ratio.clear()
            self._dram_src_handles_by_tp_ratio.clear()
            self._dram_src_handle = None

    def _finish_shutdown(self) -> None:
        self._recving_transfers.clear()
        try:
            self._release_local_xfer_handles()
        except Exception:
            logger.exception("NIXL dlist-handle release failed at shutdown.")
        try:
            for engine_id in list(self._remote_agents):
                self._cleanup_remote_engine(engine_id, log_eviction=False)
        except Exception:
            logger.exception("NIXL remote-engine cleanup failed at shutdown.")
        try:
            if not self._kv_unregistered:
                for desc in self._registered_descs:
                    self.nixl_wrapper.deregister_memory(desc)
        finally:
            self._registered_descs.clear()
            # Drop cache references before their owners release registered
            # host memory; handshake futures may outlive model-runner shutdown.
            self.device_kv_caches = {}
            self.host_xfer_buffers = {}

    def shutdown(self) -> None:
        """Shutdown the connector worker."""
        if not hasattr(self, "_handshake_initiation_executor"):
            # error happens during init, no need to shutdown
            return
        self._handshake_initiation_executor.shutdown(wait=False)
        for handles in self._recving_transfers.values():
            for handle in handles:
                self.nixl_wrapper.release_xfer_handle(handle)
        self._recving_transfers.clear()
        self._finish_shutdown()
