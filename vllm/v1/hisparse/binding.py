# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side wiring of HiSparse caches to attention layers."""

from copy import copy
from typing import TYPE_CHECKING, Any

import torch

from vllm.config import VllmConfig
from vllm.v1.hisparse.layout import (
    HISPARSE_HOT_SUFFIX,
    HISPARSE_RESIDENT_SUFFIX,
    get_hisparse_kv_cache_groups,
)
from vllm.v1.hisparse.runtime import (
    HiSparseCacheHandle,
    HiSparseHostPool,
    drop_hisparse_host_pool,
    get_hisparse_host_pool,
    initialize_hisparse_runtime_buffers,
    release_pinned_state,
)
from vllm.v1.kv_cache_interface import (
    HiSparseHotSpec,
    HiSparseResidentSpec,
    KVCacheConfig,
    KVCacheLayout,
    KVCacheSpec,
    MLAAttentionSpec,
    create_kv_cache_views,
)
from vllm.v1.worker.utils import allocate_kv_cache

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.block_table import BlockTables


def resolve_hisparse_specs(
    vllm_config: VllmConfig,
    kv_cache_spec: dict[str, KVCacheSpec],
) -> dict[str, KVCacheSpec]:
    """Add the resident/hot specs HiSparse derives from the sparse-MLA specs.

    The planner plans every HiSparse group at the global block size, so the
    derived specs' block size is a placeholder resolved again at grouping time.
    """
    if not any(isinstance(spec, MLAAttentionSpec) for spec in kv_cache_spec.values()):
        return kv_cache_spec
    result = get_hisparse_kv_cache_groups(vllm_config, kv_cache_spec)
    if result is None:
        return kv_cache_spec
    groups, _ = result
    return kv_cache_spec | {
        layer_name: group.kv_cache_spec
        for group in groups
        if isinstance(group.kv_cache_spec, (HiSparseResidentSpec, HiSparseHotSpec))
        for layer_name in group.layer_names
    }


def allocate_hisparse_kv_caches(
    kv_cache_config: KVCacheConfig,
    device: torch.device,
    layout: KVCacheLayout,
    host_pool: HiSparseHostPool,
) -> dict[str, torch.Tensor]:
    """Allocate the host pool separately from the shared device backing."""
    device_config = copy(kv_cache_config)
    device_config.kv_cache_tensors = [
        tensor
        for tensor in kv_cache_config.kv_cache_tensors
        if not tensor.host_resident
    ]
    kv_caches = allocate_kv_cache(device_config, device, layout)
    host_tensors = [
        tensor for tensor in kv_cache_config.kv_cache_tensors if tensor.host_resident
    ]
    (host_size,) = {tensor.size for tensor in host_tensors}
    backing = host_pool.allocate(host_size)
    (host_group_id,) = kv_cache_config.host_group_ids
    layer_specs = kv_cache_config.kv_cache_groups[host_group_id].layer_kv_cache_specs
    assert layer_specs is not None
    host_num_blocks = kv_cache_config.hisparse_host_num_blocks
    assert host_num_blocks is not None
    for tensor in host_tensors:
        views = create_kv_cache_views(
            backing,
            layer_specs[tensor.layers[0]],
            host_num_blocks,
            layout,
            tensor,
        )
        kv_caches.update(zip(tensor.layers, views))
    return kv_caches


def init_hisparse_kv_cache(
    kv_cache_config: KVCacheConfig,
    device: torch.device,
    vllm_config: VllmConfig,
    forward_context: dict[str, Any],
    block_tables: "BlockTables",
) -> dict[str, torch.Tensor]:
    """Allocate and bind HiSparse caches within the caller's allocation context."""
    host_pool = get_hisparse_host_pool(vllm_config, kv_cache_config)
    try:
        kv_caches = allocate_hisparse_kv_caches(
            kv_cache_config,
            device,
            vllm_config.cache_config.get_resolved_kv_cache_layout(),
            host_pool,
        )
        cache_handles = bind_hisparse_kv_caches(
            forward_context=forward_context,
            kv_cache_config=kv_cache_config,
            kv_caches=kv_caches,
            block_tables=block_tables,
            host_pool=host_pool,
        )
        initialize_hisparse_runtime_buffers(
            cache_handles,
            max_num_reqs=vllm_config.scheduler_config.max_num_seqs,
        )
        return kv_caches
    except Exception:
        if host_pool.shared_region is not None:
            host_pool.shared_region.cleanup()
        elif host_pool.registered is not None:
            release_pinned_state([], [host_pool.registered])
        drop_hisparse_host_pool(vllm_config)
        raise


def _get_hisparse_cache(
    forward_context: dict[str, Any], layer_name: str
) -> HiSparseCacheHandle:
    attention_layer = forward_context[layer_name]
    hisparse_cache = attention_layer.hisparse_cache
    assert hisparse_cache is not None
    return hisparse_cache


def release_hisparse_profiling_cache(forward_context: dict[str, Any]) -> None:
    """Detach the profiling caches but keep the host pool for the real KV cache."""
    runtimes = {
        id(cache.runtime): cache.runtime
        for layer in forward_context.values()
        if (cache := getattr(layer, "hisparse_cache", None)) is not None
        and hasattr(cache.runtime, "_host_cache")
    }
    release_pinned_state(list(runtimes.values()), [])


def bind_hisparse_kv_caches(
    *,
    forward_context: dict[str, Any],
    kv_cache_config: KVCacheConfig,
    kv_caches: dict[str, torch.Tensor],
    block_tables: "BlockTables",
    host_pool: HiSparseHostPool,
) -> list[HiSparseCacheHandle]:
    """Bind existing cache storage and block tables; return the bound handles."""
    assert host_pool.registered is not None
    tensor_configs = {
        name: (
            tensor_config,
            tensor_config.offset + layer_index * tensor_config.layer_stride,
        )
        for tensor_config in kv_cache_config.kv_cache_tensors
        for layer_index, name in enumerate(tensor_config.layers)
    }
    resident_groups = [
        (group_id, group)
        for group_id, group in enumerate(kv_cache_config.kv_cache_groups)
        if isinstance(group.kv_cache_spec, HiSparseResidentSpec)
    ]
    residency: torch.Tensor | None = None
    if resident_groups:
        table_shapes = {
            block_tables.input_block_tables[group_id].shape
            for group_id, _ in resident_groups
        }
        assert len(table_shapes) == 1
        max_num_reqs, max_num_pages = table_shapes.pop()
        residency = torch.zeros(
            (max_num_reqs, len(resident_groups), max_num_pages),
            dtype=torch.int32,
            device=block_tables.input_block_tables[resident_groups[0][0]].device,
        )
    for resident_source_index, (group_id, group) in enumerate(resident_groups):
        assert residency is not None
        for cache_name in group.layer_names:
            assert cache_name.endswith(HISPARSE_RESIDENT_SUFFIX)
            layer_name = cache_name[: -len(HISPARSE_RESIDENT_SUFFIX)]
            tensor_config, byte_offset = tensor_configs[cache_name]
            assert not tensor_config.host_resident
            cache_handle = _get_hisparse_cache(forward_context, layer_name)
            cache_handle.bind_cache(
                kv_caches[cache_name],
                byte_offset=byte_offset,
                block_stride=tensor_config.block_stride,
                # The view spans the group's whole kernel-row grid.
                num_blocks=tensor_config.size // tensor_config.block_stride,
                block_size=group.kv_cache_spec.block_size,
                block_table=residency[:, resident_source_index],
                slot_mapping=block_tables.slot_mappings[group_id],
            )
            cache_handle.residency = residency
            assert cache_handle.view is not None
            kv_caches[cache_name] = cache_handle.view.cache
            cache_handle.runtime.resident_source_index = resident_source_index

    hot_backing: torch.Tensor | None = None
    cache_handles: list[HiSparseCacheHandle] = []
    for group_id, group in enumerate(kv_cache_config.kv_cache_groups):
        if not isinstance(group.kv_cache_spec, HiSparseHotSpec):
            continue
        for cache_name in group.layer_names:
            assert cache_name.endswith(HISPARSE_HOT_SUFFIX)
            raw_tensor = kv_caches[cache_name]
            if hot_backing is None:
                hot_backing = raw_tensor
            elif hot_backing.untyped_storage().data_ptr() != (
                raw_tensor.untyped_storage().data_ptr()
            ):
                raise RuntimeError("HiSparse hot tensors must share one GPU backing.")
            layer_name = cache_name[: -len(HISPARSE_HOT_SUFFIX)]
            cache_handle = _get_hisparse_cache(forward_context, layer_name)
            tensor_config, byte_offset = tensor_configs[cache_name]
            assert not tensor_config.host_resident
            cache_handle.runtime.bind_hot_cache(
                raw_tensor,
                byte_offset=byte_offset,
                block_stride=tensor_config.block_stride,
                num_blocks=tensor_config.size // tensor_config.block_stride,
                block_size=group.kv_cache_spec.block_size,
                block_table=block_tables.input_block_tables[group_id],
            )
            resident = cache_handle.view
            hot = cache_handle.runtime.hot
            assert resident is not None
            if (
                resident.cache.untyped_storage().data_ptr()
                != hot.cache.untyped_storage().data_ptr()
                or resident.cache.data_ptr() != hot.cache.data_ptr()
                or resident.cache.stride() != hot.cache.stride()
            ):
                raise RuntimeError("HiSparse resident and hot layouts must match.")
            source_cache = kv_caches[layer_name]
            assert source_cache.untyped_storage().data_ptr() == (
                host_pool.registered.untyped_storage().data_ptr()
            )
            cache_handle.draft_layer = forward_context[layer_name].is_draft_layer
            cache_handle.runtime.shared_host_region = host_pool.shared_region
            cache_handle.runtime.bind_source_cache(
                source_cache,
                registered_host_pool=host_pool.registered,
            )
            cache_handles.append(cache_handle)
            # The hot slab is raw storage owned by the runtime, not a layer
            # cache: nothing downstream binds or registers it.
            del kv_caches[cache_name]

    if hot_backing is None or not cache_handles:
        raise RuntimeError("HiSparse found no hot-cache handles.")
    (source_group_id,) = kv_cache_config.host_group_ids
    for cache_handle in cache_handles:
        cache_handle.source_block_table = block_tables.input_block_tables[
            source_group_id
        ]
        cache_handle.mirror_slot_mapping = block_tables.slot_mappings[source_group_id]
    return cache_handles
