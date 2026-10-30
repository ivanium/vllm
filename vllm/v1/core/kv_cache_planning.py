# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV cache planning: grouping, sizing, and memory-fit logic."""

import math
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from functools import partial, reduce
from typing import Any, NamedTuple

from vllm.config import SpeculativeConfig, VllmConfig
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv, round_up
from vllm.utils.mem_utils import format_gib
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.attention.backend import MultipleOf
from vllm.v1.core.kv_cache_config_builder import KVCacheConfigBuilder
from vllm.v1.core.kv_cache_utils import (
    _kv_cache_device_name,
    large_page_blocks_per_request,
)
from vllm.v1.hisparse.layout import (
    get_hisparse_host_pool_bytes,
    get_hisparse_kv_cache_config,
    get_hisparse_kv_cache_groups,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    ChunkedLocalAttentionSpec,
    CircularBufferSpec,
    FullAttentionSpec,
    HiddenStateCacheSpec,
    HiSparseHotSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheLayout,
    KVCacheSpec,
    KVCacheTensor,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
    compute_layout_strides,
    iter_layer_specs,
    min_blocks_per_large_page,
    replace_as,
)
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry

logger = init_logger(__name__)


def _check_enough_kv_cache_memory(
    available_memory: int,
    get_needed_memory: Callable[[], int],
    max_model_len: int,
    estimate_max_model_len: Callable[[int], int],
):
    if available_memory <= 0:
        raise ValueError(
            "No available memory for the cache blocks. "
            "Try increasing `gpu_memory_utilization` when initializing the engine "
            "(this flag also controls CPU memory reservation on the CPU "
            "backend, despite its name). "
            "See https://docs.vllm.ai/en/latest/configuration/conserving_memory/ "
            "for more details."
        )

    needed_memory = get_needed_memory()

    if needed_memory > available_memory:
        estimated_max_len = estimate_max_model_len(available_memory)
        estimated_msg = ""
        if estimated_max_len > 0:
            estimated_msg = (
                "Based on the available memory, "
                f"the estimated maximum model length is {estimated_max_len}. "
            )

        raise ValueError(
            f"To serve at least one request with the model's max seq len "
            f"({max_model_len}), ({format_gib(needed_memory)} GiB KV "
            f"cache is needed, which is larger than the available KV cache "
            f"memory ({format_gib(available_memory)} GiB). {estimated_msg}"
            f"Try increasing `gpu_memory_utilization` (which also controls "
            f"CPU memory on the CPU backend) or decreasing `max_model_len` "
            f"when initializing the engine. "
            f"See https://docs.vllm.ai/en/latest/configuration/conserving_memory/ "
            f"for more details."
        )


def max_memory_usage_bytes(
    vllm_config: VllmConfig, kv_cache_specs: Iterable[KVCacheSpec]
) -> int:
    """Get the maximum memory usage in bytes for the given KV cache specs."""
    return sum(spec.max_memory_usage_bytes(vllm_config) for spec in kv_cache_specs)


def estimate_max_model_len(
    vllm_config: VllmConfig,
    kv_cache_spec: dict[str, KVCacheSpec],
    available_memory: int,
) -> int:
    """Estimates the maximum model length that can fit in the available memory
    using binary search.

    This function temporarily modifies max_model_len during estimation but
    restores the original value before returning, ensuring no side effects.

    Args:
        vllm_config: The global VllmConfig
        kv_cache_spec: The kv cache spec of each attention layer in the model
        available_memory: Memory available for KV cache in bytes.

    Returns:
        The estimated maximum model length that can fit in the available memory.

    """
    # Save the original max_model_len to restore after estimation
    original_max_model_len = vllm_config.model_config.max_model_len

    # Define a function to check if a given model length fits in memory
    def fits_in_memory(model_len: int) -> bool:
        # Temporarily modify the max_model_len for this calculation
        vllm_config.model_config.max_model_len = model_len
        # Calculate memory needed for the given model length
        memory_needed = max_memory_usage_bytes(vllm_config, kv_cache_spec.values())
        return memory_needed <= available_memory

    try:
        # Binary search for the maximum model length
        left, right = 1, original_max_model_len

        # If even the smallest model length doesn't fit, return 0
        if not fits_in_memory(left):
            return 0

        # Binary search for the maximum model length that fits
        result = 1
        while left <= right:
            mid = (left + right) // 2
            if fits_in_memory(mid):
                result = mid
                left = mid + 1
            else:
                right = mid - 1
        return result
    finally:
        # Always restore the original max_model_len to avoid side effects
        vllm_config.model_config.max_model_len = original_max_model_len


def check_enough_kv_cache_memory(
    vllm_config: VllmConfig,
    kv_cache_spec: dict[str, KVCacheSpec],
    available_memory: int,
):
    """Checks whether `available_memory` is enough for the KV cache to hold at
    least one request with the model's max_model_len.

    Args:
        vllm_config: The global VllmConfig
        kv_cache_spec: The kv cache spec of each attention layer in the model
        available_memory: Memory available for KV cache in bytes.

    Raises:
        ValueError: If there is not enough memory available for the KV cache.

    """
    # No need to check for available memory if the kv_cache_spec is empty
    if kv_cache_spec:
        if vllm_config.cache_config.kv_cache_allocator == "large_page":
            groups = get_kv_cache_groups(vllm_config, dict(kv_cache_spec))
            plans, page_bytes = _plan_large_page_placement(
                groups, [kv_cache_spec], vllm_config, [available_memory]
            )
            _large_page_check_memory(
                vllm_config,
                plans,
                _usable_large_pages(available_memory // page_bytes),
                page_bytes,
            )
            return
        # Reserve the null block BlockPool permanently holds back, so the check
        # plans against usable blocks, as in get_kv_cache_configs. Group a copy
        # of the specs since grouping may unify them in-place.
        groups = get_kv_cache_groups(vllm_config, dict(kv_cache_spec))
        check_memory = (
            available_memory - _pool_bytes_per_block(vllm_config, groups)
            if groups
            else available_memory
        )
        _check_enough_kv_cache_memory(
            check_memory,
            lambda: max_memory_usage_bytes(vllm_config, kv_cache_spec.values()),
            vllm_config.model_config.max_model_len,
            lambda am: estimate_max_model_len(vllm_config, kv_cache_spec, am),
        )


def create_kv_cache_group_specs(
    kv_cache_spec: dict[str, KVCacheSpec], grouped_layer_names: list[list[str]]
) -> list[KVCacheGroupSpec]:
    """Create KVCacheGroupSpec object for each kv cache group layer.
    The layers in the same group should share the same
    KVCacheSpec.

    Args:
        kv_cache_spec:
            A mapping from each layer name to its corresponding KVCacheSpec.
        grouped_layer_names:
            A list of kv cache groups, where each element is a list of layer
            names that belong to the same group and should share the same
            KVCacheSpec.

    Returns:
        A list of KVCacheGroupSpec objects, one for each group.

    """
    kv_cache_groups = []
    for layer_names_one_group in grouped_layer_names:
        layer_specs = [
            kv_cache_spec[layer_name] for layer_name in layer_names_one_group
        ]
        merged_layer_spec = layer_specs[0].merge(layer_specs)
        kv_cache_groups.append(
            KVCacheGroupSpec(layer_names_one_group, merged_layer_spec)
        )
    return kv_cache_groups


def is_kv_cache_spec_uniform(kv_cache_spec: dict[str, KVCacheSpec]) -> bool:
    """Whether all layers in the given KVCacheSpec have the same KV cache spec.
    Note that we regard FullAttentionSpec with and without sliding window as
    the same type.

    Args:
        kv_cache_spec: The kv cache spec of each attention layer in the model

    Returns:
        True if all layers have the same type, False otherwise.

    """
    if not kv_cache_spec:
        # Encoder-only models do not have KV cache, kv_cache_type can be
        # regarded as uniform.
        return True
    try:
        kv_cache_spec_values = list(kv_cache_spec.values())
        _ = kv_cache_spec_values[0].merge(kv_cache_spec_values)
    except AssertionError:
        return False
    return True


def may_override_num_blocks(vllm_config: VllmConfig, num_blocks: int) -> int:
    """Override the number of kv cache blocks if `num_gpu_blocks_override` is set.
    The override is logged once, at the call site in `get_kv_cache_configs`.
    """
    if vllm_config.cache_config.num_gpu_blocks_override is not None:
        num_blocks = vllm_config.cache_config.num_gpu_blocks_override
    return num_blocks


def _pool_bytes_per_block(
    vllm_config: VllmConfig, kv_cache_groups: list[KVCacheGroupSpec]
) -> int:
    """Bytes consumed by one block in the worker's shared KV cache pool, mirroring
    the divisor used by `get_kv_cache_config_from_groups` to convert
    `available_memory` into `num_blocks`. Used to compute the effective KV cache
    capacity once `num_gpu_blocks_override` is applied.
    """
    return _get_kv_cache_bytes_per_block(
        [group for group in kv_cache_groups if not group.host_resident],
        vllm_config.cache_config.get_resolved_kv_cache_layout(),
    )


def _pool_bytes_per_block_from_builder(
    builder: KVCacheConfigBuilder,
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
) -> int:
    """Bytes consumed by one block in the worker's device KV cache pool.

    Core places one block with the builder and reads the size of its single
    device backing. A placement with several device allocations declares the
    value through ``KVCacheConfigBuilder._get_pool_bytes_per_block``. HiSparse
    keeps the pure computation, since probing its placement would check the host
    pool against ``max_model_len`` before auto-fit shortens it.
    ``num_gpu_blocks_override`` is suppressed so it cannot inflate the probe.
    """
    if not kv_cache_groups:
        return 0
    if vllm_config.attention_config.hisparse_config is not None:
        return _pool_bytes_per_block(vllm_config, kv_cache_groups)
    pool_bytes = builder._get_pool_bytes_per_block(vllm_config, kv_cache_groups)
    if pool_bytes is not None:
        return pool_bytes
    saved_override = vllm_config.cache_config.num_gpu_blocks_override
    vllm_config.cache_config.num_gpu_blocks_override = None
    try:
        probe = builder.get_kv_cache_config_from_groups(
            vllm_config, kv_cache_groups, num_blocks=1
        )
    finally:
        vllm_config.cache_config.num_gpu_blocks_override = saved_override
    backing_sizes = {
        tensor.size for tensor in probe.kv_cache_tensors if not tensor.host_resident
    }
    if len(backing_sizes) != 1:
        raise ValueError(
            "KV cache tensors must share one device backing allocation, but got "
            f"sizes {sorted(backing_sizes)}. Override "
            "KVCacheConfigBuilder._get_pool_bytes_per_block() for placements "
            "with several device allocations."
        )
    return backing_sizes.pop()


def get_uniform_page_size(kv_cache_specs: Iterable[KVCacheSpec]) -> int:
    """Get the page size of the KV cache."""
    page_sizes = {layer.page_size_bytes for layer in kv_cache_specs}
    assert len(page_sizes) == 1
    return page_sizes.pop()


def _get_kv_cache_groups_uniform_spec(
    kv_cache_specs: dict[str, KVCacheSpec],
) -> list[KVCacheGroupSpec]:
    """Generates the KV cache configuration for a model with the same KV cache
    spec for all layers.

    Args:
        kv_cache_specs: The kv cache spec of each attention layer in the model

    Returns:
        The generated KVCacheGroupSpecs

    """
    return create_kv_cache_group_specs(kv_cache_specs, [list(kv_cache_specs.keys())])


def _get_kv_cache_groups_uniform_type(
    spec: UniformTypeKVCacheSpecs,
) -> list[KVCacheGroupSpec]:
    """Generates the KV cache configuration for a model with one type of KV cache
    but different hidden sizes. All layers are merged into one group.

    Args:
        spec: The UniformTypeKVCacheSpecs of the model

    Returns:
        The generated KVCacheGroupSpecs

    """
    return [KVCacheGroupSpec(list(spec.kv_cache_specs.keys()), spec)]


def unify_kv_cache_spec_page_size(
    kv_cache_spec: dict[str, KVCacheSpec],
) -> dict[str, KVCacheSpec]:
    """Unify the page size of the given KVCacheSpec. If the page size of all layers
    are the same, return the original KVCacheSpec. If not same, unify the page
    size by increasing the block size of layers with smaller page size. Two
    cases cannot be unified by block size alone and pad their physical page to
    the maximum instead: Mamba layers, whose page size comes from state shapes
    and is independent of block size; and non-MLA attention layers whose page
    does not evenly divide the maximum (the padded page is read through a
    strided view). MLA is excluded because sparse MLA indexes the cache in
    whole token rows (see ``flat_kv_row_view``), so its block stride can only
    be padded by its own row-aligned ``alignment``, not to an arbitrary page
    size. Raise NotImplementedError if failed to unify the page size;
    ``get_kv_cache_groups`` catches it to try the full-allocation fallback
    (e.g. MLA next to an incompatible sliding-window draft).

    Args:
        kv_cache_spec: The KVCacheSpec of each attention layer in the model

    Returns:
        The updated KVCacheSpec with the same page_size_bytes.

    """
    page_sizes = {layer.page_size_bytes for layer in kv_cache_spec.values()}
    if len(page_sizes) <= 1:
        # All layers have the same page size, no need to unify.
        return kv_cache_spec

    max_page_size = max(page_sizes)
    new_kv_cache_spec = {}
    for layer_name, layer_spec in kv_cache_spec.items():
        if layer_spec.page_size_bytes == max_page_size:
            new_kv_cache_spec[layer_name] = layer_spec
        elif isinstance(layer_spec, MambaSpec):
            # MambaSpec's page size is determined by its state shapes and does
            # not scale with block_size, so pad the page instead. This is the
            # same padding mechanism the platform uses to align Mamba pages
            # with the main model's attention page size; it is needed here
            # when another layer (e.g. from a draft model) has a larger page
            # than the already-aligned Mamba page.
            new_spec: KVCacheSpec = replace(layer_spec, page_size_padded=max_page_size)
            assert new_spec.page_size_bytes == max_page_size
            new_kv_cache_spec[layer_name] = new_spec
        else:
            layer_page_size = layer_spec.page_size_bytes
            if max_page_size % layer_page_size == 0:
                ratio = max_page_size // layer_page_size
                new_block_size = layer_spec.block_size * ratio
                new_spec = replace(layer_spec, block_size=new_block_size)
            elif isinstance(layer_spec, AttentionSpec) and not isinstance(
                layer_spec, MLAAttentionSpec
            ):
                new_spec = replace(layer_spec, page_size_padded=max_page_size)
            else:
                raise NotImplementedError(
                    f"Layer {layer_name}: page size is not divisible by the "
                    "maximum page size and cannot be padded. Padding is only "
                    "supported for non-MLA attention layers."
                )
            assert new_spec.page_size_bytes == max_page_size
            new_kv_cache_spec[layer_name] = new_spec
    return new_kv_cache_spec


def is_kv_cache_type_attention_free(kv_cache_spec: dict[str, KVCacheSpec]) -> bool:
    # kv_cache_spec is an empty dict for attention free models
    return not kv_cache_spec


def _get_kv_cache_groups_uniform_page_size(
    kv_cache_spec: dict[str, KVCacheSpec],
) -> list[KVCacheGroupSpec]:
    """Generates the KV cache groups for hybrid models with multiple
    attention types but still with a uniform page size (physical memory per
    block per layer) for all layers.

    Detailed explanation about kv cache management of hybrid models:
    The layers in the models are repeated with some patterns, e.g., a model
    with 10 full attention layers and 20 sliding window attention layers can be
    regarded as repeating the pattern (1 * full, 2 * sw) 10 times.
    The KVCacheManager allocates different block tables for each of the 3 layers
    in the pattern, and repeats each of them 10 times to generate the
    block_table for the 30 layers in the model.
    Therefore, we can group the layers in the model into 3 kv_cache_groups, each
    of which contains 10 layers in the model.
    The KVCacheManager allocates the block_table for each group based on its
    kv_cache spec, and the model runner applies the block table to each layer
    in the group.
    For example:
    1. A model only uses full attention. The pattern is
    (num_hidden_layers * full), so there is only one group and the block table
    is shared by all layers. It is already handled by
    `_get_kv_cache_config_uniform_type`.
    2. A model with 10 full attention layers and 20 sliding window
    attention layers. There are 3 layers in the pattern (1 * full, 2 * sw), so
    there are 3 kv_cache_groups, each of which represents 10 layers.

    To simplify the implementation, we make the following assumptions:
    1. Physical memory per block: Must be the same across all KV cache groups.
    Breaking this assumption is non-trivial due to memory fragmentation concerns
    when allocating blocks of different sizes.
    2. Tokens per block (block_size): Currently, we directly use
    `CacheConfig.block_size` for all layers. It can be extended to vary by KV
    cache group, but within each KV cache group, all layers must share the same
    block size.
    3. Physical memory per token per layer: This property is decided by model
    config. Currently we only support models that have the same physical memory
    per token per layer for all layers. Can be relaxed with a simple extension,
    but still need to keep physical memory per block the same for all groups.
    4. Number of layers per group: Currently assumed the same for all layers.
    Can be relaxed with a simple extension, but still need to keep physical
    memory per block the same for all groups.
    5. Attention type within groups: All layers in a group must share the same
    attention type. One exception is that, when
    `--disable-hybrid-kv-cache-manager` is true, the single group for full
    attention layers may also include attention layers using sliding window or
    LLaMA 4 local attention. See `unify_hybrid_kv_cache_specs` for more details.
    6. Support for multiple attention types: The design for most components is
    general to an arbitrary number of attention types. But
    `find_longest_cache_hit` only supports one attention type or two
    types of full-attention plus exactly one another type. The general
    implementation of this function is feasible but we don't know how to
    implement it cleanly yet.

    As we assume tokens per block, physical memory per token per layer, and
    number of layers per group are the same now, we can ensure that physical
    memory per block is the same for all groups.

    Args:
        kv_cache_spec: The KVCacheSpec of each attention layer in the model
    Returns:
        The generated KVCacheGroupSpecs

    """
    # Group all layers by kv_cache_spec.
    # E.g., 2 full attention layers and 3 sliding window attention layers,
    # -> (full.0, full.1), (sw.0, sw.1, sw.2).
    same_type_layers: dict[KVCacheSpec, list[str]] = defaultdict(list)
    for layer_name, layer_spec in kv_cache_spec.items():
        same_type_layers[layer_spec].append(layer_name)

    # Attempt to further merge same-type layers based on whether their KV
    # cache specs can be merged, to minimize the group count. This benefits
    # situations where specs share a block layout and differ only in a
    # property it can reconcile (e.g. full attention layers differing only in
    # sliding window / attention chunk size).
    layer_buckets: list[list[str]] = []
    spec_buckets: list[list[KVCacheSpec]] = []
    for layer_spec, layer_names in same_type_layers.items():
        for names, specs in zip(layer_buckets, spec_buckets):
            try:
                # A raise means that the specs are incompatible.
                type(specs[0]).merge([*specs, layer_spec])
            except (AssertionError, ValueError):
                continue
            names.extend(layer_names)
            specs.append(layer_spec)
            break
        else:
            layer_buckets.append(list(layer_names))
            spec_buckets.append([layer_spec])

    # Split each group into smaller groups, to make the number of layers in each
    # group identical. Add padding to the last group of each type if necessary.
    # E.g., (full.0, full.1), (sw.0, sw.1, sw.2)
    # split to 3 groups with 2 layers each:
    # (full.0, full.1), (sw.0, sw.2), (sw.1, padding).
    # FIXME(Chen): At the moment of writing this code (2025-06-02), all
    # open-source hybrid model follows a n:1 pattern between different attention
    # types (e.g., Gemma3 5:1 between sw and full, LLaMA4 3:1 between local and
    # full), so we can use the "1" in the n:1 pattern as the group size, which
    # is the minimum number of layers among all attention types. Need a better
    # strategy if we want to support more complex patterns (e.g., 20 full + 30
    # sw, where the group size should be 10).
    min_num_layers = min([len(layers) for layers in layer_buckets])
    group_size = min_num_layers
    max_num_layers = max([len(layers) for layers in layer_buckets])
    if max_num_layers < min_num_layers * 1.5:
        # If the number of layers is not much larger than the minimum number of
        # layers, use the maximum number of layers as the group size to avoid
        # too many padding layers. A typical example is gpt-oss-20b + eagle,
        # with 12 sw + 13 full. We pad it to (13 sw, 13 full) instead of
        # (12 sw, 24 full). 1.5 is a heuristic to avoid too many padding
        # layers while accommodating speculative decoding drafters that add
        # extra layers to one attention type.
        group_size = max_num_layers
    grouped_layers = []
    for layers in layer_buckets:
        num_padding_layers = group_size - len(layers) % group_size
        if num_padding_layers != group_size:
            logger.warning(
                "Add %d padding layers, may waste at most %.2f%% KV cache memory",  # noqa
                num_padding_layers,
                num_padding_layers / len(layers) * 100,
            )
        num_groups = cdiv(len(layers), group_size)
        # In PP case, say if we have
        # - stage 0: full.0, sw.0, sw.1
        # - stage 1: full.1, sw.2, sw.3
        # We should have 3 groups: (full.0, full.1), (sw.0, sw.2), (sw.1, sw.3)
        # It can't be (full.0, full.1), (sw.0, sw.1), (sw.2, sw.3) because
        # the 3 groups in stage 0 will be (full.0), (sw.0, sw.1), (empty group)
        # and it will be padded to (full.0, padding), (sw.0, sw.1),
        # (padding, padding) to ensure the number of layers in each group is
        # the same and will cause memory waste.
        # To avoid this, we assign layers[i::num_groups] to the i-th group
        # instead of layers[i * group_size: (i + 1) * group_size]
        for i in range(num_groups):
            grouped_layers.append(layers[i::num_groups])
    return create_kv_cache_group_specs(kv_cache_spec, grouped_layers)


def _get_per_layer_spec(
    group: KVCacheGroupSpec,
    layer_name: str,
) -> KVCacheSpec:
    spec = group.kv_cache_spec
    if isinstance(spec, UniformTypeKVCacheSpecs):
        return spec.kv_cache_specs[layer_name]
    return spec


def _get_layer_block_stride(spec: KVCacheSpec) -> int:
    """A layer-outer block stride: the page padded to its alignment."""
    return round_up(spec.page_size_bytes, spec.get_block_stride_alignment())


def _get_kv_cache_bytes_per_block(
    kv_cache_groups: list[KVCacheGroupSpec], layout: KVCacheLayout
) -> int:
    """Return the largest cache group's bytes per block."""
    if not layout.is_block_outermost:
        # Each layer's block stride is its own padded page.
        return max(
            sum(
                _get_layer_block_stride(_get_per_layer_spec(group, layer_name))
                for layer_name in group.layer_names
            )
            for group in kv_cache_groups
        )
    bytes_per_block = max(
        sum(
            _get_per_layer_spec(group, layer_name).page_size_bytes
            for layer_name in group.layer_names
        )
        for group in kv_cache_groups
    )
    assert bytes_per_block > 0
    alignments = [
        group.kv_cache_spec.page_size_bytes
        for group in kv_cache_groups
        if isinstance(group.kv_cache_spec, HiSparseHotSpec)
    ]
    alignments.extend(
        _get_per_layer_spec(group, layer_name).get_block_stride_alignment()
        for group in kv_cache_groups
        for layer_name in group.layer_names
    )
    return round_up(bytes_per_block, math.lcm(*alignments))


def validate_kv_cache_layout(
    layout: KVCacheLayout,
    kv_cache_groups: list[KVCacheGroupSpec],
) -> None:
    """Validate that the resolved layout can express this model's packing.

    The layout was chosen once in the engine core from the backends' supported
    sets; a backend whose model packs pages side by side (e.g. the DeepSeek-V4
    indexer) declares block-outermost layouts there, so an inexpressible
    layout reaching this point is an error.
    """
    page_sizes = {
        _get_per_layer_spec(group, layer_name).page_size_bytes
        for group in kv_cache_groups
        for layer_name in group.layer_names
    }
    if len(page_sizes) == 1:
        # A rectangular layer dim exists; every layout can express it.
        return

    # Mixed page sizes pack pages side by side within a block, which needs each page
    # to be one contiguous chunk inside its block (a block-compact layout) and, with
    # multiple KV cache groups, the layer dim inside the block dim.
    if not layout.is_block_compact or (
        len(kv_cache_groups) > 1 and layout.is_layer_compact
    ):
        raise ValueError(
            f"KV cache layout {layout.name} cannot express this model's "
            f"mixed page sizes ({sorted(page_sizes)}); a backend should "
            "declare block-outermost supported layouts (e.g. BLHNC), or "
            "set VLLM_KV_CACHE_LAYOUT=BLHNC."
        )


def get_kv_cache_config_from_groups(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
    available_memory: int,
) -> KVCacheConfig:
    """Generate the KV cache configuration from the KV cache groups and spec
    of each layer.

    Args:
        vllm_config: The global VllmConfig
        kv_cache_groups: The KV cache groups
        available_memory: Memory available for KV cache in bytes
    Returns:
        The generated KVCacheConfig

    """
    if len(kv_cache_groups) == 0:
        # Attention free models do not have KV cache.
        # Return num_blocks=1 as BlockPool always needs a null_block.
        return KVCacheConfig(
            num_blocks=1,
            kv_cache_tensors=[],
            kv_cache_groups=kv_cache_groups,
            prefix_cache_retention_interval=(
                vllm_config.cache_config.prefix_cache_retention_interval
            ),
        )

    if vllm_config.attention_config.hisparse_config is not None:
        host_budget = get_hisparse_host_pool_bytes(vllm_config)
        return get_hisparse_kv_cache_config(
            vllm_config, kv_cache_groups, available_memory, host_budget
        )

    layout = vllm_config.cache_config.get_resolved_kv_cache_layout()
    validate_kv_cache_layout(layout, kv_cache_groups)
    bytes_per_block = _get_kv_cache_bytes_per_block(kv_cache_groups, layout)
    interleaved_block_stride = bytes_per_block if layout.is_block_outermost else None

    num_blocks = available_memory // bytes_per_block
    num_blocks = may_override_num_blocks(vllm_config, num_blocks)
    size = bytes_per_block * num_blocks

    # Groups alias from byte 0. Spec regions are laid out differently:
    #
    # block-outer (the same packing repeats for every block):
    # group 0: | blk 0 [ A | B  | pad ] | blk 1 [ A | B  | pad ] | ...
    # group 1: | blk 0 [  C  |    D   ] | blk 1 [  C  |    D   ] | ...
    #          |<--- bytes_per_block -->|
    #
    # layer-outer (only supported for uniform page sizes or single-group models):
    # group 0: | A [ blk 0 | blk 1 | ... ] | B [ blk 0 | blk 1 | ... ] |
    # group 1: | C [ blk 0 | blk 1 | ... ] | D [ blk 0 | blk 1 | ... ] |

    kv_cache_tensors = []
    for group in kv_cache_groups:
        group_spec = group.kv_cache_spec
        layers_by_spec: defaultdict[KVCacheSpec, list[str]] = defaultdict(list)
        if isinstance(group_spec, UniformTypeKVCacheSpecs):
            for layer_name, spec in group_spec.kv_cache_specs.items():
                layers_by_spec[spec].append(layer_name)
        elif group.layer_names:
            layers_by_spec[group_spec].extend(group.layer_names)

        byte_offset = 0
        for spec, layer_names in layers_by_spec.items():
            block_stride = interleaved_block_stride
            padded_page = _get_layer_block_stride(spec)
            if block_stride is None and padded_page != spec.page_size_bytes:
                assert layout.is_block_compact, f"{layout.name} cannot pad pages"
                block_stride = padded_page
            layer_stride, block_stride, _, _, _ = compute_layout_strides(
                spec,
                num_blocks,
                len(layer_names),
                layout,
                fixed_strides=(None, block_stride, None, None, None),
            )
            kv_cache_tensors.append(
                KVCacheTensor(
                    size=size,
                    layers=layer_names,
                    layer_stride=layer_stride,
                    block_stride=block_stride,
                    offset=byte_offset,
                )
            )
            # Block-outer layers share each block; layer-outer ones own regions.
            byte_offset += len(layer_names) * (
                spec.page_size_bytes if layout.is_block_outermost else layer_stride
            )

    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=kv_cache_tensors,
        kv_cache_groups=kv_cache_groups,
        prefix_cache_retention_interval=(
            vllm_config.cache_config.prefix_cache_retention_interval
        ),
    )


def _promote_local_kv_cache_specs(
    kv_cache_spec: dict[str, KVCacheSpec],
) -> dict[str, KVCacheSpec]:
    """Use full-attention allocation for local-attention cache specs.

    The returned specs affect KV cache management only. Attention modules keep
    their original sliding-window or chunked-local compute behavior.
    """
    promoted_specs = kv_cache_spec.copy()

    if is_kv_cache_spec_uniform(
        promoted_specs
    ) or UniformTypeKVCacheSpecs.is_uniform_type(promoted_specs):
        return promoted_specs

    has_full_attention = any(
        isinstance(spec, FullAttentionSpec) for spec in promoted_specs.values()
    )
    has_sliding_window = any(
        isinstance(spec, SlidingWindowSpec) for spec in promoted_specs.values()
    )
    has_chunked_local_attention = any(
        isinstance(spec, ChunkedLocalAttentionSpec) for spec in promoted_specs.values()
    )
    full_block_sizes = {
        spec.block_size
        for spec in promoted_specs.values()
        if isinstance(spec, FullAttentionSpec)
    }
    full_attention_block_size = (
        next(iter(full_block_sizes)) if len(full_block_sizes) == 1 else None
    )

    def promoted_page_size_padded(spec: AttentionSpec, block_size: int) -> int | None:
        if spec.page_size_padded is None:
            return None
        unpadded_page_size = (
            spec.unpadded_page_size_bytes * block_size // spec.block_size
        )
        return max(spec.page_size_padded, unpadded_page_size)

    promotions: dict[type[AttentionSpec], type[AttentionSpec]] = {
        SlidingWindowMLASpec: MLAAttentionSpec,
        SlidingWindowSpec: FullAttentionSpec,
        ChunkedLocalAttentionSpec: FullAttentionSpec,
    }

    if has_full_attention and (has_sliding_window or has_chunked_local_attention):
        for layer_name, spec in kv_cache_spec.items():
            target_cls = next(
                (promotions[c] for c in type(spec).__mro__ if c in promotions), None
            )
            if target_cls is None:
                continue
            assert isinstance(spec, AttentionSpec)
            block_size = full_attention_block_size or spec.block_size
            promoted_specs[layer_name] = replace_as(
                spec,
                target_cls,
                # Promoted specs allocate blocks for all tokens and never free
                # below the window, so the trailing-edge extension is moot.
                drop=("extra_retained_tokens",),
                block_size=block_size,
                page_size_padded=promoted_page_size_padded(spec, block_size),
            )

    if not (
        is_kv_cache_spec_uniform(promoted_specs)
        or UniformTypeKVCacheSpecs.is_uniform_type(promoted_specs)
    ):
        raise ValueError("Failed to promote local KV cache specs to one unified type.")

    return promoted_specs


def _try_get_full_allocation_fallback_groups(
    kv_cache_spec: dict[str, KVCacheSpec],
) -> list[KVCacheGroupSpec] | None:
    """Try a supported full-allocation fallback for local-attention layers."""
    if any(isinstance(spec, HiddenStateCacheSpec) for spec in kv_cache_spec.values()):
        return None
    if any(
        isinstance(spec, (SlidingWindowMLASpec, ChunkedLocalAttentionSpec))
        for spec in kv_cache_spec.values()
    ):
        return None

    has_mla = any(isinstance(spec, MLAAttentionSpec) for spec in kv_cache_spec.values())
    has_regular_swa = any(
        isinstance(spec, SlidingWindowSpec) for spec in kv_cache_spec.values()
    )
    if not (has_mla and has_regular_swa):
        return None

    try:
        promoted_specs = _promote_local_kv_cache_specs(kv_cache_spec)
    except ValueError:
        return None
    uniform_spec = UniformTypeKVCacheSpecs.from_specs(promoted_specs)
    if uniform_spec is None:
        return None
    logger.warning(
        "KV cache page sizes cannot be unified; treating sliding-window "
        "layers as full attention for cache allocation. Sliding-window "
        "attention compute is unchanged."
    )
    return _get_kv_cache_groups_uniform_type(uniform_spec)


def unify_hybrid_kv_cache_specs(kv_cache_spec: dict[str, KVCacheSpec]):
    """This function tries to convert the KV cache specs to one type if the model
    is a hybrid model with multiple type of KV cache. It will convert all
    SlidingWindowSpec to FullAttentionSpec if both types are present.

    Args:
        kv_cache_spec: The kv cache spec of each attention layer in the model

    """
    groups: defaultdict[bool, dict[str, KVCacheSpec]] = defaultdict(dict)
    for name, spec in kv_cache_spec.items():
        replicated = isinstance(spec, AttentionSpec) and not spec.dcp_sharded
        groups[replicated][name] = spec
    for specs in groups.values():
        promoted_specs = _promote_local_kv_cache_specs(specs)
        if promoted_specs == specs:
            continue
        logger.warning(
            "Hybrid KV cache manager is disabled for this hybrid model, "
            "This means we do not enable any optimizations for saving KV cache "
            "memory (e.g., dropping the KV cache outside the sliding window). "
            "The compute of layers like sliding window is still saved."
        )
        kv_cache_spec.update(promoted_specs)


def _approximate_gcd(values: Sequence[int], *, lower_bound: int | None = None) -> int:
    """Pick a chunk size that minimizes total upward padding.

    Each x is rounded up to a multiple of d:

      x -> ceil(x / d) * d

    Total padding is:

      pad(d) = sum_i (ceil(x_i / d) * d - x_i)

    We brute-force d in [lower_bound, max(values)] (fine for small lists / small
    maxima) and return the d with minimum padding. Ties prefer larger d.
    """
    if not values:
        raise ValueError("values must be non-empty")
    if any(x <= 0 for x in values):
        raise ValueError(f"values must be positive, got: {list(values)!r}")

    min_d = max(1, lower_bound if lower_bound is not None else 1)
    max_d = max(values)
    if min_d > max_d:
        return min_d

    best_d = min_d
    best_pad: int | None = None
    for d in range(min_d, max_d + 1):
        pad = sum((d - (x % d)) % d for x in values)
        if best_pad is None or pad < best_pad or (pad == best_pad and d > best_d):
            best_pad = pad
            best_d = d

    return best_d


def _get_packed_kv_cache_groups(
    vllm_config: VllmConfig,
    kv_cache_spec: dict[str, KVCacheSpec],
) -> list[KVCacheGroupSpec] | None:
    """Group mixed-page-size layers for contiguous block-outermost packing.

    Greedily buckets layers into uniform-type specs. Buckets with equal layer
    counts per page size are treated as a repeating layer pattern (one layer
    per page size) and split into groups covering the same number of pattern
    repeats (picked by ``_approximate_gcd`` to minimize padding), so all
    groups pack into the same per-block layout. Mamba buckets are additionally
    split to fit the block the attention buckets already need.
    Returns None when the layout is not block-outermost or all layers already
    share one page size.
    """
    layout = vllm_config.cache_config.get_resolved_kv_cache_layout()
    page_sizes = {spec.page_size_bytes for spec in kv_cache_spec.values()}
    if not layout.is_block_outermost or len(page_sizes) <= 1:
        return None

    buckets: list[dict[str, KVCacheSpec]] = []
    for name, spec in kv_cache_spec.items():
        for bucket in buckets:
            candidate = {**bucket, name: spec}
            if UniformTypeKVCacheSpecs.is_uniform_type(candidate):
                bucket[name] = spec
                break
        else:
            buckets.append({name: spec})

    bucketed = []
    for bucket in buckets:
        uniform_spec = UniformTypeKVCacheSpecs.from_specs(bucket)
        assert uniform_spec is not None
        page_size_layers: dict[int, list[str]] = defaultdict(list)
        for layer_name, layer_spec in bucket.items():
            page_size_layers[layer_spec.page_size_bytes].append(layer_name)
        # Only 1:1 patterns (one layer of each page size per repeat) are
        # supported; counts sharing a gcd > 1 (e.g. 2:1) could in principle
        # repeat too, but such buckets are emitted whole instead.
        balanced = len(set(map(len, page_size_layers.values()))) == 1
        bucketed.append((uniform_spec, page_size_layers, balanced))

    # Balanced buckets that mix page sizes must stay whole, so the largest one
    # sets a floor on the repeats per group; larger single-size buckets are
    # split down toward it. No such bucket means nothing needs packing.
    min_repeats_per_group = max(
        (
            spec.get_max_layers_per_page_size()
            for spec, page_size_layers, balanced in bucketed
            if balanced and len(page_size_layers) > 1
        ),
        default=0,
    )
    repeats_per_group = (
        _approximate_gcd(
            [
                spec.get_max_layers_per_page_size()
                for spec, _, balanced in bucketed
                if balanced
            ],
            lower_bound=min_repeats_per_group,
        )
        if min_repeats_per_group
        else None
    )

    def num_groups_for(spec: UniformTypeKVCacheSpecs, balanced: bool) -> int:
        if balanced and repeats_per_group is not None:
            return cdiv(spec.get_max_layers_per_page_size(), repeats_per_group)
        return 1

    def widest_group_bytes(page_size_layers: dict[int, list[str]], n: int) -> int:
        """Page bytes of the largest of the n groups a bucket splits into."""
        return sum(
            cdiv(len(names), n) * page for page, names in page_size_layers.items()
        )

    # Bytes a block must hold however the state buckets end up split: mamba,
    # circular-buffer and unsplit sliding-window buckets can go down to one
    # state per group, every other bucket's split is fixed by the repeat pattern.
    def is_state_bucket(spec: UniformTypeKVCacheSpecs) -> bool:
        if isinstance(spec.first_spec, (MambaSpec, CircularBufferSpec)):
            return True
        return repeats_per_group is None and isinstance(
            spec.first_spec, SlidingWindowSpec
        )

    anchor_bytes = max(
        (
            widest_group_bytes(
                page_size_layers,
                len(spec.kv_cache_specs)
                if is_state_bucket(spec)
                else num_groups_for(spec, balanced),
            )
            for spec, page_size_layers, balanced in bucketed
        ),
        default=0,
    )

    groups = []
    for spec, page_size_layers, balanced in bucketed:
        num_groups = num_groups_for(spec, balanced)
        # Cap a state group at the states a block already fits rather than let
        # it widen the block.
        if anchor_bytes and is_state_bucket(spec):
            states_per_block = max(anchor_bytes // spec.first_spec.page_size_bytes, 1)
            num_groups = max(
                num_groups, cdiv(len(spec.kv_cache_specs), states_per_block)
            )
        if num_groups == 1:
            groups.append(KVCacheGroupSpec(list(spec.kv_cache_specs), spec))
            continue

        pattern_repeats = list(zip(*page_size_layers.values()))
        for i in range(num_groups):
            group_layer_names = [
                name for repeat in pattern_repeats[i::num_groups] for name in repeat
            ]
            group_layer_specs = {
                name: spec.kv_cache_specs[name] for name in group_layer_names
            }
            group_spec = UniformTypeKVCacheSpecs.from_specs(group_layer_specs)
            assert group_spec is not None
            groups.append(KVCacheGroupSpec(group_layer_names, group_spec))

    return groups


def _warn_if_unannotated_eagle_mamba(
    spec_config: SpeculativeConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
) -> None:
    """Warn when no KV cache group could be identified as the draft model's.

    Args:
        spec_config: Config supplying the speculative method.
        kv_cache_groups: Groups as they will be handed to consumers.

    """
    if any(group.is_eagle_group for group in kv_cache_groups):
        return
    mamba_groups = [
        idx
        for idx, group in enumerate(kv_cache_groups)
        if any(
            isinstance(spec, MambaSpec)
            for spec in iter_layer_specs(group.kv_cache_spec)
        )
    ]
    if not mamba_groups:
        return
    logger.warning(
        "Speculative decoding (method=%s) is enabled but no KV cache group "
        "could be identified as the draft model's.",
        spec_config.method,
    )


def _largest_divisor_at_most(value: int, limit: int) -> int:
    for candidate in range(min(value, limit), 0, -1):
        if value % candidate == 0:
            return candidate
    return 1


def _ensure_min_page_size(
    groups: list[KVCacheGroupSpec],
    common_page: int,
    hidden_specs: dict[str, HiddenStateCacheSpec],
) -> tuple[list[KVCacheGroupSpec], int]:
    """Scale up group block sizes so the common page is at least as large as
    the biggest hidden-state per-token cost.

    This protects the hidden state extraction feature, where we store hidden states in
    the KV cache, and return them to clients using a custom hidden state connector.

    Returns the (possibly scaled) groups and updated common page size.
    """
    max_per_token = max(
        spec.num_kv_heads * spec.head_size * get_dtype_size(spec.dtype)
        for spec in hidden_specs.values()
    )
    if max_per_token <= common_page:
        return groups, common_page

    scale = cdiv(max_per_token, common_page)
    common_page *= scale
    scaled: list[KVCacheGroupSpec] = []
    for g in groups:
        s = g.kv_cache_spec
        kw: dict[str, Any] = {"block_size": s.block_size * scale}
        if isinstance(s, (AttentionSpec, MambaSpec)) and s.page_size_padded is not None:
            kw["page_size_padded"] = s.page_size_padded * scale
        scaled.append(KVCacheGroupSpec(g.layer_names, replace(s, **kw)))
    return scaled, common_page


# ---------------------------------------------------------------------------
# Large-page allocator planning (CacheConfig.kv_cache_allocator="large_page")
#
# One KV cache group per (normalized) spec; one global block size B for all
# prefix-cacheable token-paged groups. The pool is N large pages of S bytes;
# each page belongs to one group at a time. Block b of group g is kernel
# blocks b * R_g .. b * R_g + R_g - 1 of every layer view, so a page holds the
# blocks that lie wholly in it (large_page_block_range). S and row_g are the
# same on every rank (a rank with fewer layers of a group pads its rows), so
# block ids name the same page on every rank.

# The page size is the smallest multiple of a group's slot, at least the
# largest slot, whose demand-weighted waste is within this bound and that every
# rank's budget can hold one max-model-len request with. Smaller pages strand
# fewer bytes on pages that one group owns.
_LARGE_PAGE_MAX_PAGE_WASTE = 0.02
# Unless --block-size is set, B starts here: smaller blocks only add hashing
# and block-table work.
_DEFAULT_MIN_BLOCK_SIZE = 64


def _kernel_block_size_feasible(
    sizes: tuple[int | MultipleOf, ...], block_size: int
) -> bool:
    """Whether a block of ``block_size`` tokens splits into whole kernel
    blocks: some listed exact size divides it, or it is a multiple of some
    ``MultipleOf`` base."""
    return any(
        block_size % (size.base if isinstance(size, MultipleOf) else size) == 0
        for size in sizes
    )


def _kernel_block_size_supported(
    sizes: tuple[int | MultipleOf, ...], kernel_block_size: int
) -> bool:
    """Whether the listed sizes allow ``kernel_block_size`` as a kernel block:
    an exact ``int`` must be listed; ``MultipleOf(m)`` accepts multiples of m."""
    return any(
        kernel_block_size == size
        if isinstance(size, int)
        else kernel_block_size % size.base == 0
        for size in sizes
    )


def select_global_block_size(
    vllm_config: VllmConfig, kv_cache_spec: dict[str, KVCacheSpec]
) -> int:
    """The one block size B for all token-grid (prefix-cacheable, token-paged)
    groups under the large-page allocator.

    B is the smallest multiple of L0 (the LCM of the pure divisibility
    constraints: prefix-cacheable tokens_per_state, DCP for sharded groups,
    align-mode Mamba prefill-checkpoint alignments, ring capacities) at or
    above the requested block size (at least 64) at which every token-grid
    layer has a feasible kernel block. A user-set ``--block-size`` must be
    feasible as-is.
    """
    cache_config = vllm_config.cache_config
    dcp_size = vllm_config.parallel_config.decode_context_parallel_size
    constraints: dict[str, int] = {}
    # Layer -> (kernel block sizes, tokens of B the layer holds per rank).
    kernel_preds: dict[str, tuple[tuple[int | MultipleOf, ...], int]] = {}
    for layer_name, spec in kv_cache_spec.items():
        if isinstance(spec, CircularBufferSpec):
            # Rings keep their private block size (the ring capacity), which
            # must still divide the token grid.
            constraints[f"ring capacity of {layer_name}"] = spec.block_size
        elif isinstance(spec, MambaSpec):
            if spec.mamba_cache_mode == "align" and spec.prefill_checkpoint_alignment:
                constraints[f"prefill-checkpoint alignment of {layer_name}"] = (
                    spec.prefill_checkpoint_alignment
                )
        elif isinstance(spec, AttentionSpec) and spec.prefix_cacheable:
            if spec.kernel_block_sizes:
                shard = dcp_size if spec.dcp_sharded and dcp_size > 1 else 1
                kernel_preds[layer_name] = (spec.kernel_block_sizes, shard)
            tokens_per_state = spec.tokens_per_state
            if isinstance(tokens_per_state, int) and tokens_per_state > 1:
                constraints[f"tokens-per-state of {layer_name}"] = tokens_per_state
            if spec.dcp_sharded and dcp_size > 1:
                constraints[f"decode context parallelism of {layer_name}"] = dcp_size

    base_lcm = math.lcm(1, *constraints.values())
    requested = cache_config.block_size
    if cache_config.user_specified_block_size:
        problems = []
        if requested % base_lcm:
            detail = dict(sorted(constraints.items()))
            problems.append(f"it must be a multiple of {base_lcm} ({detail})")
        for layer_name, (sizes, shard) in kernel_preds.items():
            if not _kernel_block_size_feasible(sizes, requested // shard):
                problems.append(
                    f"layer {layer_name} supports kernel block sizes "
                    f"{list(sizes)}, none of which divides {requested // shard}"
                )
        if problems:
            raise ValueError(
                f"Invalid block_size={requested} with "
                "kv_cache_allocator='large_page': " + "; ".join(problems)
            )
        return requested
    block_size = round_up(max(requested, _DEFAULT_MIN_BLOCK_SIZE), base_lcm)
    for _ in range(1024):
        if all(
            _kernel_block_size_feasible(sizes, block_size // shard)
            for sizes, shard in kernel_preds.values()
        ):
            return block_size
        block_size += base_lcm
    raise ValueError(
        f"No feasible large-page block size: L0={base_lcm}, "
        f"kernel constraints={kernel_preds}"
    )


class _Bucket(NamedTuple):
    spec: KVCacheSpec
    kernel_block_size: int
    layer_names: list[str]


def _large_page_group_block_size(spec: KVCacheSpec, B: int, dcp_size: int) -> int:
    """The block size a spec is planned at under the large-page allocator.

    Prefix-cacheable token-paged specs use the global block size B (DCP-sharded
    specs hold B/dcp tokens per rank). Rings, none-mode Mamba, and
    non-prefix-cacheable window specs keep their own (private) block size.
    """
    if isinstance(spec, CircularBufferSpec):
        return spec.block_size
    if isinstance(spec, MambaSpec):
        return B if spec.mamba_cache_mode == "align" else spec.block_size
    if isinstance(spec, AttentionSpec):
        if not spec.prefix_cacheable:
            return spec.block_size
        if spec.dcp_sharded and dcp_size > 1:
            return B // dcp_size
        return B
    return spec.block_size


def _get_kv_cache_groups_large_page(
    vllm_config: VllmConfig,
    kv_cache_spec: dict[str, KVCacheSpec],
) -> list[KVCacheGroupSpec]:
    """Group layers by normalized spec under the large-page allocator.

    Bucket keys are the specs at their final block size with planner-written
    padding, per-layer backend alignment and kernel-block constraints stripped
    (the layers' own padding, e.g. MLA ``alignment``, is recomputed by
    ``__post_init__`` and kept). Buckets then combine when their specs merge,
    keeping the uniform-page planner's merge tolerance (window size,
    non-causal-multi-token-decode). Per-layer specs are kept on the group for
    placement and never enter ``merge``.
    """
    cache_config = vllm_config.cache_config
    disable_hybrid = vllm_config.scheduler_config.disable_hybrid_kv_cache_manager
    if disable_hybrid:
        unify_hybrid_kv_cache_specs(kv_cache_spec)
    B = select_global_block_size(vllm_config, kv_cache_spec)
    # Workers skip Platform.update_block_size_for_backend under this allocator;
    # the resolved block sizes reach them through KVCacheConfig (see
    # Worker.initialize_from_config).
    cache_config.block_size = B
    if cache_config.mamba_cache_mode == "align":
        if cache_config.mamba_block_size not in (None, B):
            logger.info(
                "kv_cache_allocator='large_page' sets mamba_block_size to the "
                "resolved block size %d, discarding %d",
                B,
                cache_config.mamba_block_size,
            )
        cache_config.mamba_block_size = B
    if cache_config.prefix_match_unit not in (None, B):
        raise ValueError(
            f"Invalid prefix_match_unit={cache_config.prefix_match_unit}; "
            "with kv_cache_allocator='large_page' prefix-cache hits land on "
            f"the resolved block size ({B}). Unset it or set it to {B}."
        )
    if not kv_cache_spec:
        return []

    dcp_size = vllm_config.parallel_config.decode_context_parallel_size
    # A group needs one row layout, so layers whose backends resolve to
    # different kernel blocks at their block size never share a group (e.g.
    # MiniMax-M3 dense layers at 64 vs sparse layers at 128).
    buckets: dict[tuple[KVCacheSpec, int], list[str]] = {}
    layer_specs: dict[str, KVCacheSpec] = {}
    for layer_name, spec in kv_cache_spec.items():
        key: KVCacheSpec
        if isinstance(spec, (AttentionSpec, MambaSpec)):
            block_size = _large_page_group_block_size(spec, B, dcp_size)
            if (
                isinstance(spec, MambaSpec)
                and spec.mamba_cache_mode == "align"
                and spec.prefill_checkpoint_alignment
            ):
                # Checkpoint offsets stay on the prefix-cache grid, the block
                # size, which the raw layer alignment divides.
                spec = replace(spec, prefill_checkpoint_alignment=block_size)
            final_spec = replace(spec, block_size=block_size, page_size_padded=None)
            layer_specs[layer_name] = final_spec
            key = replace(
                final_spec, block_stride_alignment=None, kernel_block_sizes=()
            )
            kernel_block_size = _largest_kernel_block_size(
                block_size, [_kernel_block_constraint(final_spec)]
            )
        else:
            # Non-attention, non-mamba spec types pass through unchanged.
            layer_specs[layer_name] = spec
            key = spec
            kernel_block_size = spec.block_size
        buckets.setdefault((key, kernel_block_size), []).append(layer_name)

    merged_buckets: list[_Bucket] = []
    for (key, kernel_block_size), layer_names in buckets.items():
        for i, bucket in enumerate(merged_buckets):
            # A subclass marker (e.g. HiddenStateCacheSpec) must not be
            # absorbed into a field-identical parent-class bucket.
            if (
                type(key) is not type(bucket.spec)
                or bucket.kernel_block_size != kernel_block_size
            ):
                continue
            try:
                merged = bucket.spec.merge([bucket.spec, key])
            except (AssertionError, ValueError):
                continue
            merged_buckets[i] = bucket._replace(
                spec=merged, layer_names=bucket.layer_names + layer_names
            )
            break
        else:
            merged_buckets.append(_Bucket(key, kernel_block_size, layer_names))

    kv_cache_groups = [
        KVCacheGroupSpec(
            bucket.layer_names,
            bucket.spec,
            layer_kv_cache_specs={
                name: layer_specs[name] for name in bucket.layer_names
            },
        )
        for bucket in merged_buckets
    ]
    for group in kv_cache_groups:
        assert group.layer_kv_cache_specs is not None
        page_sizes = {
            spec.page_size_bytes for spec in group.layer_kv_cache_specs.values()
        }
        assert len(page_sizes) == 1, (
            f"KV cache group has layers with different page sizes: {sorted(page_sizes)}"
        )
    if disable_hybrid and len(kv_cache_groups) > 1:
        raise ValueError(
            "--disable-hybrid-kv-cache-manager needs every layer in one KV cache "
            "group, but kv_cache_allocator='large_page' formed "
            f"{len(kv_cache_groups)} groups for this model."
        )
    return kv_cache_groups


@dataclass(frozen=True)
class _LargePageGroupPlan:
    """Placement of one KV cache group in the large-page pool."""

    group: KVCacheGroupSpec
    kernel_block_size: int
    # The largest row over ranks; every rank lays its rows out at this size.
    row_bytes: int
    # Per rank; empty where the group holds no layers.
    layer_offsets: list[dict[str, int]]

    @property
    def rows_per_block(self) -> int:
        """Kernel blocks (rows) per manager block of this group (R_g)."""
        return self.group.kv_cache_spec.block_size // self.kernel_block_size

    @property
    def slot_bytes(self) -> int:
        """Bytes of one manager block of this group."""
        return self.rows_per_block * self.row_bytes

    def slots_per_page(self, large_page_bytes: int) -> int:
        """Fewest blocks of this group one large page holds (n_g)."""
        return min_blocks_per_large_page(large_page_bytes, self.slot_bytes)


def _kernel_block_constraint(spec: KVCacheSpec) -> tuple[int | MultipleOf, ...]:
    """Kernel block sizes a layer's backend supports; empty for state and ring
    layers, which have no kernel constraint (one row is one block)."""
    if isinstance(spec, AttentionSpec) and not isinstance(spec, CircularBufferSpec):
        return spec.kernel_block_sizes
    return ()


def _largest_kernel_block_size(
    block_size: int, layer_sizes: list[tuple[int | MultipleOf, ...]]
) -> int:
    """The largest divisor of ``block_size`` that every listed layer supports."""
    layer_sizes = [sizes for sizes in layer_sizes if sizes]
    if not layer_sizes:
        return block_size
    divisors = sorted(
        {
            d
            for i in range(1, math.isqrt(block_size) + 1)
            if block_size % i == 0
            for d in (i, block_size // i)
        },
        reverse=True,
    )
    for kernel_block_size in divisors:
        if all(
            _kernel_block_size_supported(sizes, kernel_block_size)
            for sizes in layer_sizes
        ):
            return kernel_block_size
    raise ValueError(
        f"No common kernel block size for block size {block_size} "
        f"and kernel constraints {layer_sizes}."
    )


def _large_page_kernel_block_size(group: KVCacheGroupSpec) -> int:
    """Tokens per kernel block (one row) of a group."""
    return _largest_kernel_block_size(
        group.kv_cache_spec.block_size,
        [
            _kernel_block_constraint(spec)
            for spec in (group.layer_kv_cache_specs or {}).values()
        ],
    )


def _large_page_row_layout(
    layer_specs: dict[str, KVCacheSpec], kernel_block_size: int
) -> tuple[int, int, dict[str, int]]:
    """Bytes of one kernel block across all layers, the unit a row must be a
    multiple of, and per-layer offsets.

    Each layer offset is aligned to its block-stride alignment; a row that
    is a multiple of the LCM of those units keeps every row start, and so
    every layer's block, aligned.
    """
    offsets: dict[str, int] = {}
    units: list[int] = []
    offset = 0
    for layer_name, spec in layer_specs.items():
        unit = spec.get_block_stride_alignment()
        offset = round_up(offset, unit) if offset else 0
        offsets[layer_name] = offset
        if isinstance(spec, (AttentionSpec, MambaSpec)):
            offset += replace(
                spec, block_size=kernel_block_size, page_size_padded=None
            ).page_size_bytes
        else:
            offset += spec.page_size_bytes
        units.append(unit)
    return offset, math.lcm(*units), offsets


def _large_page_waste(
    plans: list[_LargePageGroupPlan],
    large_page_bytes: int,
    vllm_config: VllmConfig,
) -> float:
    """Share of the page bytes a max-model-len request takes that its blocks
    do not use. Requests share pages, so a request takes a 1 / slots_per_page
    share of a page per block."""
    used, taken = 0, 0.0
    for plan in plans:
        blocks = large_page_blocks_per_request(plan.group.kv_cache_spec, vllm_config)
        used += blocks * plan.slot_bytes
        taken += blocks * large_page_bytes / plan.slots_per_page(large_page_bytes)
    return 1 - used / taken


def _usable_large_pages(num_pages: int) -> int:
    """Pages of a pool of ``num_pages`` that requests can take: page 0 is the
    reserved null page."""
    return num_pages - 1


def _plan_large_page_placement(
    groups: list[KVCacheGroupSpec],
    worker_specs: list[dict[str, KVCacheSpec]],
    vllm_config: VllmConfig,
    available_memory: list[int] | None = None,
) -> tuple[list[_LargePageGroupPlan], int]:
    """Per-group plans and the large-page size S.

    Block ids address the same pages on every rank only if S and each
    group's row are the same on every rank, so a group's row is its largest
    over ranks. ``available_memory`` (per rank) rules out pages its budget
    cannot serve one max-model-len request with.
    """
    plans: list[_LargePageGroupPlan] = []
    for group in groups:
        kernel_block_size = _large_page_kernel_block_size(group)
        row = unit = 1
        offsets: list[dict[str, int]] = []
        for worker_spec in worker_specs:
            layer_specs = {
                name: group.layer_kv_cache_specs[name]  # type: ignore[index]
                for name in group.layer_names
                if name in worker_spec
            }
            layer_offsets: dict[str, int] = {}
            if layer_specs:
                rank_row, rank_unit, layer_offsets = _large_page_row_layout(
                    layer_specs, kernel_block_size
                )
                row = max(row, rank_row)
                unit = math.lcm(unit, rank_unit)
            offsets.append(layer_offsets)
        plans.append(
            _LargePageGroupPlan(group, kernel_block_size, round_up(row, unit), offsets)
        )

    max_slot = max(plan.slot_bytes for plan in plans)

    def fits_budget(large_page_bytes: int) -> bool:
        needed = _large_page_needed_pages(plans, vllm_config, large_page_bytes)
        return available_memory is None or all(
            needed <= _usable_large_pages(memory // large_page_bytes)
            for memory in available_memory
        )

    # A page loses at most two blocks of a group to its ends, so past this
    # size every page is within the waste bound.
    max_page = math.ceil(2 * max_slot / _LARGE_PAGE_MAX_PAGE_WASTE)
    candidates = sorted(
        {
            k * plan.slot_bytes
            for plan in plans
            for k in range(
                cdiv(max_slot, plan.slot_bytes), max_page // plan.slot_bytes + 1
            )
        }
    )
    choice: tuple[bool, float, int] | None = None
    for large_page_bytes in candidates:
        if any(plan.slots_per_page(large_page_bytes) < 1 for plan in plans):
            continue
        score = (
            not fits_budget(large_page_bytes),
            _large_page_waste(plans, large_page_bytes, vllm_config),
            large_page_bytes,
        )
        if not score[0] and score[1] <= _LARGE_PAGE_MAX_PAGE_WASTE:
            choice = score
            break
        if choice is None or score[:2] < choice[:2]:
            choice = score
    assert choice is not None, "a KV cache group does not fit into any large page"
    return plans, choice[2]


def _large_page_needed_pages(
    plans: list[_LargePageGroupPlan], vllm_config: VllmConfig, large_page_bytes: int
) -> int:
    """Usable pages one max-model-len request needs.

    Every group draws from the one page pool the scheduler manages, so a
    group counts even on PP ranks that hold none of its layers.
    """
    return sum(
        cdiv(
            large_page_blocks_per_request(plan.group.kv_cache_spec, vllm_config),
            plan.slots_per_page(large_page_bytes),
        )
        for plan in plans
    )


def _large_page_max_fitting_len(
    plans: list[_LargePageGroupPlan],
    vllm_config: VllmConfig,
    large_page_bytes: int,
    usable_pages: int,
) -> int:
    """Largest max_model_len whose request fits in ``usable_pages`` pages (0
    if even one token does not fit). Restores max_model_len before
    returning."""
    model_config = vllm_config.model_config
    original_max = model_config.max_model_len

    def fits() -> bool:
        needed = _large_page_needed_pages(plans, vllm_config, large_page_bytes)
        return needed <= usable_pages

    try:
        best = 0
        lo, hi = 1, original_max
        while lo <= hi:
            mid = (lo + hi) // 2
            model_config.max_model_len = mid
            if fits():
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        return best
    finally:
        model_config.max_model_len = original_max


def _large_page_check_and_fit_max_model_len(
    vllm_config: VllmConfig,
    plans: list[_LargePageGroupPlan],
    usable_pages: int,
    page_bytes: int,
) -> None:
    """Auto-fit max_model_len if requested, then check that one full
    max-model-len request fits in ``usable_pages`` pages."""
    model_config = vllm_config.model_config

    if model_config.original_max_model_len == -1:
        auto_fit_max = _large_page_max_fitting_len(
            plans, vllm_config, page_bytes, usable_pages
        )
        if auto_fit_max <= 0:
            raise ValueError(
                "Cannot auto-fit max_model_len: not enough KV cache memory "
                "to serve even a single token. Try increasing "
                "`gpu_memory_utilization`."
            )
        if auto_fit_max >= model_config.max_model_len:
            logger.info_once(
                "Auto-fit max_model_len: full model context length %d fits in "
                "available KV cache memory (%d large pages)",
                model_config.max_model_len,
                usable_pages,
            )
        else:
            logger.info_once(
                "Auto-fit max_model_len: reduced from %d to %d to fit in "
                "available KV cache memory (%d large pages)",
                model_config.max_model_len,
                auto_fit_max,
                usable_pages,
            )
            model_config.max_model_len = auto_fit_max

    _large_page_check_memory(vllm_config, plans, usable_pages, page_bytes)


def _large_page_check_memory(
    vllm_config: VllmConfig,
    plans: list[_LargePageGroupPlan],
    usable_pages: int,
    page_bytes: int,
) -> None:
    """Check that one max-model-len request fits in ``usable_pages`` pages."""
    _check_enough_kv_cache_memory(
        usable_pages * page_bytes,
        lambda: _large_page_needed_pages(plans, vllm_config, page_bytes) * page_bytes,
        vllm_config.model_config.max_model_len,
        lambda available: _large_page_max_fitting_len(
            plans, vllm_config, page_bytes, available // page_bytes
        ),
    )


def _build_large_page_kv_cache_config(
    vllm_config: VllmConfig,
    plans: list[_LargePageGroupPlan],
    rank: int,
    large_page_bytes: int,
    num_pages: int,
) -> KVCacheConfig:
    """One rank's KVCacheConfig: per-layer tensors at kernel-block (row)
    granularity over the shared pool of num_pages pages of large_page_bytes."""
    cache_config = vllm_config.cache_config
    groups: list[KVCacheGroupSpec] = []
    tensors: list[KVCacheTensor] = []
    size = num_pages * large_page_bytes
    for plan in plans:
        group = plan.group
        assert group.layer_kv_cache_specs is not None
        layer_names = [
            name for name in group.layer_names if name in plan.layer_offsets[rank]
        ]
        groups.append(
            KVCacheGroupSpec(
                layer_names,
                group.kv_cache_spec,
                is_eagle_group=group.is_eagle_group,
                host_resident=group.host_resident,
                enable_kv_transfer=group.enable_kv_transfer,
                role=group.role,
                num_blocks=size // plan.slot_bytes,
                slots_per_large_page=plan.slots_per_page(large_page_bytes),
                kernel_rows_per_block=plan.rows_per_block,
                row_bytes=plan.row_bytes,
                layer_kv_cache_specs={
                    name: group.layer_kv_cache_specs[name] for name in layer_names
                },
            )
        )
        for layer_name in layer_names:
            tensors.append(
                KVCacheTensor(
                    size=size,
                    layers=[layer_name],
                    layer_stride=plan.row_bytes,
                    block_stride=plan.row_bytes,
                    offset=plan.layer_offsets[rank][layer_name],
                )
            )
    return KVCacheConfig(
        num_blocks=num_pages,
        kv_cache_tensors=tensors,
        kv_cache_groups=groups,
        prefix_cache_retention_interval=cache_config.prefix_cache_retention_interval,
        allocator="large_page",
        large_page_bytes=large_page_bytes,
        hash_block_size=cache_config.block_size,
        cache_hit_alignment_tokens=cache_config.block_size,
    )


def get_large_page_kv_cache_config_from_groups(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
    min_blocks: int,
) -> KVCacheConfig:
    """Local large-page placement for worker-side profiling caches.

    The engine computes the real (cross-rank) placement after profiling; here
    one worker plans its own groups alone, with enough pages to give every
    group ``min_blocks`` blocks. Over-estimating is safe: the minimal cache is
    torn down after profiling.
    """
    worker_specs = [
        {
            name: spec
            for group in kv_cache_groups
            for name, spec in (group.layer_kv_cache_specs or {}).items()
        }
    ]
    plans, large_page_bytes = _plan_large_page_placement(
        kv_cache_groups, worker_specs, vllm_config
    )
    num_pages = max(
        cdiv(min_blocks, plan.slots_per_page(large_page_bytes)) for plan in plans
    )
    return _build_large_page_kv_cache_config(
        vllm_config, plans, 0, large_page_bytes, num_pages
    )


def _get_kv_cache_configs_large_page(
    vllm_config: VllmConfig,
    builder: KVCacheConfigBuilder,
    merged_kv_cache_specs: dict[str, KVCacheSpec],
    kv_cache_specs: list[dict[str, KVCacheSpec]],
    available_memory: list[int],
) -> list[KVCacheConfig]:
    """Large-page allocator variant of the cross-worker KV cache planning."""
    cache_config = vllm_config.cache_config
    global_kv_cache_groups = get_kv_cache_groups_from_builder(
        builder, vllm_config, merged_kv_cache_specs
    )
    # The global block size B, resolved by the grouping call.
    B = cache_config.block_size

    spec_config = vllm_config.speculative_config
    if spec_config is not None and spec_config.use_eagle_block_drop():
        draft_layer_names = {
            name for name, spec in merged_kv_cache_specs.items() if spec.is_draft
        }
        for group in global_kv_cache_groups:
            group.is_eagle_group = not draft_layer_names.isdisjoint(group.layer_names)
        _warn_if_unannotated_eagle_mamba(spec_config, global_kv_cache_groups)

    if not global_kv_cache_groups:
        # Attention-free models do not have KV cache. Return num_blocks=1 as
        # BlockPool always needs a null_block.
        return [
            KVCacheConfig(
                num_blocks=1,
                kv_cache_tensors=[],
                kv_cache_groups=[],
                prefix_cache_retention_interval=(
                    cache_config.prefix_cache_retention_interval
                ),
                allocator="large_page",
                hash_block_size=B,
                cache_hit_alignment_tokens=B,
            )
            for _ in kv_cache_specs
        ]

    plans, large_page_bytes = _plan_large_page_placement(
        global_kv_cache_groups, kv_cache_specs, vllm_config, available_memory
    )
    num_pages = min(available_memory) // large_page_bytes
    override = cache_config.num_gpu_blocks_override
    if override is not None:
        logger.info(
            "Overriding num_gpu_blocks=%d with num_gpu_blocks_override=%d",
            num_pages,
            override,
        )
        num_pages = override

    _large_page_check_and_fit_max_model_len(
        vllm_config, plans, _usable_large_pages(num_pages), large_page_bytes
    )

    kv_cache_configs = [
        _build_large_page_kv_cache_config(
            vllm_config, plans, rank, large_page_bytes, num_pages
        )
        for rank in range(len(kv_cache_specs))
    ]
    for kv_cache_config in kv_cache_configs:
        kv_cache_config.kv_tp_replicas = kv_cache_groups_tp_replicas(
            kv_cache_config.kv_cache_groups,
            vllm_config.parallel_config.tensor_parallel_size,
            vllm_config.parallel_config.decode_context_parallel_size,
        )
    return kv_cache_configs


def get_kv_cache_groups(
    vllm_config: VllmConfig,
    kv_cache_spec: dict[str, KVCacheSpec],
) -> list[KVCacheGroupSpec]:
    """Split the layers in the model into groups with the same KV cache spec.

    Args:
        vllm_config: The global VllmConfig
        kv_cache_spec: The kv cache spec of each attention layer in the model

    Returns:
        The generated KVCacheGroups

    """
    if vllm_config.cache_config.kv_cache_allocator == "large_page":
        return _get_kv_cache_groups_large_page(vllm_config, kv_cache_spec)

    if vllm_config.scheduler_config.disable_hybrid_kv_cache_manager:
        unify_hybrid_kv_cache_specs(kv_cache_spec)

    if is_kv_cache_type_attention_free(kv_cache_spec):
        # This returns an empty list to allow for the KVCacheManager to handle
        # attention free models.
        return []

    if hisparse_groups := get_hisparse_kv_cache_groups(vllm_config, kv_cache_spec):
        return hisparse_groups

    if is_kv_cache_spec_uniform(kv_cache_spec):
        # KV cache of all layers are the same, which is true for
        # most models. Allocate the same amount of memory for
        # each layer.
        return _get_kv_cache_groups_uniform_spec(kv_cache_spec)
    elif uniform_spec := UniformTypeKVCacheSpecs.from_specs(kv_cache_spec):
        # All layers need the same number of token slots (e.g., all layers are
        # full attention, or all layers are sliding window attention with the
        # same window size). Put all layers into one group.
        return _get_kv_cache_groups_uniform_type(uniform_spec)
    # Hidden-state layers use their own block table and must not be absorbed
    # into a compatible attention bucket.
    hidden_specs = {
        k: v for k, v in kv_cache_spec.items() if isinstance(v, HiddenStateCacheSpec)
    }
    filtered_spec = {
        k: v
        for k, v in kv_cache_spec.items()
        if not isinstance(v, HiddenStateCacheSpec)
    }

    if packed_groups := _get_packed_kv_cache_groups(vllm_config, filtered_spec):
        # Block-outermost blocks are strided by the widest group, so hidden
        # groups need no page alignment.
        packed_groups += [
            KVCacheGroupSpec([name], spec) for name, spec in hidden_specs.items()
        ]
        return packed_groups

    # Prefer preserving each layer's cache semantics. If physical pages cannot
    # be unified, try a supported allocation-only fallback before failing.
    try:
        filtered_spec = unify_kv_cache_spec_page_size(filtered_spec)
    except NotImplementedError:
        fallback_groups = _try_get_full_allocation_fallback_groups(kv_cache_spec)
        if fallback_groups is None:
            raise
        return fallback_groups
    groups = _get_kv_cache_groups_uniform_page_size(filtered_spec)

    # Add hidden-state layers back with page aligned to the common page.
    if hidden_specs:
        common_page = get_uniform_page_size([g.kv_cache_spec for g in groups])
        # TP may shrink the common page below hidden-state per-token cost.
        groups, common_page = _ensure_min_page_size(groups, common_page, hidden_specs)
        group_block_size = math.gcd(*(g.kv_cache_spec.block_size for g in groups))
        for name, spec in hidden_specs.items():
            per_token = spec.num_kv_heads * spec.head_size * get_dtype_size(spec.dtype)
            max_block_size = max(common_page // per_token, 1)
            new_bs = _largest_divisor_at_most(group_block_size, max_block_size)
            wasted_bytes = common_page - new_bs * per_token
            logger.info(
                "Using block size %d for hidden-state cache layer %s; "
                "page alignment wastes %d bytes (%.2f%%) per block",
                new_bs,
                name,
                wasted_bytes,
                wasted_bytes / common_page * 100,
            )
            aligned = replace(spec, block_size=new_bs, page_size_padded=common_page)
            groups.append(KVCacheGroupSpec([name], aligned))

    return groups


def get_kv_cache_groups_from_builder(
    builder: KVCacheConfigBuilder,
    vllm_config: VllmConfig,
    kv_cache_spec: dict[str, KVCacheSpec],
) -> list[KVCacheGroupSpec]:
    """Group the layers with the builder and check the result.

    Core checks every grouping, whether it comes from the default grouping, a
    model's ``SupportsKVCacheGrouping`` or a platform builder.
    """
    groups = builder.get_kv_cache_groups(vllm_config, kv_cache_spec)
    _check_exact_partition(groups, kv_cache_spec)
    return groups


def _check_exact_partition(
    groups: list[KVCacheGroupSpec],
    kv_cache_spec: dict[str, KVCacheSpec],
) -> None:
    """Ensure the groups partition every input layer exactly once."""
    grouped_layers: set[str] = set()
    for group in groups:
        assert not set(group.layer_names) & grouped_layers, (
            f"KV cache group overlaps: {set(group.layer_names) & grouped_layers}."
        )
        grouped_layers |= set(group.layer_names)
    missing = set(kv_cache_spec) - grouped_layers
    assert not missing, f"KV cache groups miss layers: {sorted(missing)}."


def _layer_tp_replicas(spec: KVCacheSpec, tp_size: int, dcp_size: int) -> int:
    if not isinstance(spec, AttentionSpec) or spec.max_tp_shards is None:
        return 1
    if spec.dcp_sharded and dcp_size > 1:
        return 1
    return max(1, tp_size // spec.max_tp_shards)


def kv_cache_groups_tp_replicas(
    groups: list[KVCacheGroupSpec], tp_size: int, dcp_size: int = 1
) -> int:
    """Consecutive TP ranks holding identical KV for every layer."""
    specs = [spec for g in groups for spec in iter_layer_specs(g.kv_cache_spec)]
    if not specs:
        return 1
    return reduce(
        math.gcd, (_layer_tp_replicas(s, tp_size, dcp_size) for s in specs), tp_size
    )


def _max_memory_usage_bytes_from_groups(
    builder: KVCacheConfigBuilder,
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
) -> int:
    """Calculate maximum memory usage in bytes from KV cache groups.

    This correctly accounts for padding in hybrid models. For example, if a
    model has 8 full attention layers and 9 sliding window layers, they will
    be padded to 9 full + 9 sliding window for uniform group sizes.

    Each group independently claims blocks from the shared pool, so a request consumes
    the sum of the per-group block counts, i.e. ``bytes_per_block * total_blocks``.
    """
    kv_cache_groups = [group for group in kv_cache_groups if not group.host_resident]
    if not kv_cache_groups:
        return 0

    bytes_per_block = _pool_bytes_per_block_from_builder(
        builder, vllm_config, kv_cache_groups
    )
    total_blocks = 0
    for group in kv_cache_groups:
        spec = group.kv_cache_spec
        if isinstance(spec, UniformTypeKVCacheSpecs):
            total_blocks += spec.max_memory_usage_pages(vllm_config)
        else:
            total_blocks += cdiv(
                spec.max_memory_usage_bytes(vllm_config),
                spec.page_size_bytes,
            )

    return bytes_per_block * total_blocks


def _estimate_max_model_len_from_groups(
    builder: KVCacheConfigBuilder,
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
    available_memory: int,
) -> int:
    """Binary search for the maximum model length that fits in available memory.
    Returns 0 if even 1 token doesn't fit.
    """
    original_max = vllm_config.model_config.max_model_len
    hisparse_enabled = (
        vllm_config.attention_config.hisparse_config is not None
        and bool(kv_cache_groups)
    )

    def fits(model_len: int) -> bool:
        vllm_config.model_config.max_model_len = model_len
        if hisparse_enabled:
            try:
                num_blocks = available_memory // _pool_bytes_per_block_from_builder(
                    builder, vllm_config, kv_cache_groups
                )
                builder.get_kv_cache_config_from_groups(
                    vllm_config, kv_cache_groups, num_blocks
                )
            except ValueError:
                return False
        return (
            _max_memory_usage_bytes_from_groups(builder, vllm_config, kv_cache_groups)
            <= available_memory
        )

    try:
        left, right = 1, original_max
        if not fits(left):
            return 0
        result = 1
        while left <= right:
            mid = (left + right) // 2
            if fits(mid):
                result = mid
                left = mid + 1
            else:
                right = mid - 1
        return result
    finally:
        vllm_config.model_config.max_model_len = original_max


def _auto_fit_max_model_len(
    builder: KVCacheConfigBuilder,
    vllm_config: VllmConfig,
    projected_groups_per_worker: list[list[KVCacheGroupSpec]],
    available_memory: list[int],
) -> None:
    """When max_model_len is set to -1, this function estimates the largest
    context length that can be supported with the available GPU memory.
    It uses binary search to find the maximum length that fits across all
    workers.

    Args:
        builder: The builder whose hooks drive block accounting.
        vllm_config: The global VllmConfig (will be modified in-place)
        projected_groups_per_worker: KV cache groups projected to each worker.
        available_memory: Memory available for KV cache in bytes for each
            worker.

    """
    original_max = vllm_config.model_config.max_model_len

    if all(not groups for groups in projected_groups_per_worker):
        # All workers have empty specs (attention-free model)
        logger.info_once(
            "Auto-fit max_model_len: attention-free model, "
            "using derived max_model_len=%d",
            original_max,
        )
        return

    # Find the max_model_len that fits across all workers.
    auto_fit_max = original_max
    limiting_worker_mem = available_memory[0]
    for groups, avail_mem in zip(projected_groups_per_worker, available_memory):
        if not groups:
            continue
        worker_max = _estimate_max_model_len_from_groups(
            builder, vllm_config, groups, avail_mem
        )
        if worker_max < auto_fit_max:
            auto_fit_max = worker_max
            limiting_worker_mem = avail_mem

    if auto_fit_max <= 0:
        raise ValueError(
            "Cannot auto-fit max_model_len: not enough GPU memory available "
            "to serve even a single token. Try increasing `gpu_memory_utilization`."
        )

    kv_cache_memory = (
        f"{format_gib(limiting_worker_mem)} GiB {_kv_cache_device_name(vllm_config)}"
    )
    if vllm_config.attention_config.hisparse_config is not None:
        host_pool_gib = format_gib(get_hisparse_host_pool_bytes(vllm_config))
        kv_cache_memory += f" + {host_pool_gib} GiB host"

    if auto_fit_max >= original_max:
        # The model's full context length fits in memory
        logger.info_once(
            "Auto-fit max_model_len: full model context length %d fits in "
            "available KV cache memory (%s)",
            original_max,
            kv_cache_memory,
        )
    else:
        # Need to reduce max_model_len to fit in memory
        vllm_config.model_config.max_model_len = auto_fit_max
        logger.info_once(
            "Auto-fit max_model_len: reduced from %d to %d to fit in "
            "available KV cache memory (%s)",
            original_max,
            auto_fit_max,
            kv_cache_memory,
        )


def _project_kv_cache_groups_to_worker(
    global_kv_cache_groups: list[KVCacheGroupSpec],
    worker_spec: dict[str, KVCacheSpec],
) -> list[KVCacheGroupSpec]:
    """Projects global KV cache groups onto a single worker's assigned layers.

    In pipeline parallelism, each worker only owns a subset of layers. This
    function filters the global groups to include only layers present on the
    given worker, adjusting UniformTypeKVCacheSpecs accordingly.

    Args:
        global_kv_cache_groups: The global KV cache groups for the whole model.
        worker_spec: The KV cache spec of each layer on this worker.

    Returns:
        The projected KV cache groups containing only this worker's layers.

    """
    projected_groups: list[KVCacheGroupSpec] = []
    for group in global_kv_cache_groups:
        worker_layer_names = [
            layer_name for layer_name in group.layer_names if layer_name in worker_spec
        ]
        if len(worker_layer_names) == len(group.layer_names):
            projected_groups.append(group)
            continue
        group_spec = group.kv_cache_spec
        if worker_layer_names and isinstance(group_spec, UniformTypeKVCacheSpecs):
            group_spec = UniformTypeKVCacheSpecs(
                block_size=group_spec.block_size,
                kv_cache_specs={
                    layer_name: group_spec.kv_cache_specs[layer_name]
                    for layer_name in worker_layer_names
                },
            )
        projected_groups.append(
            KVCacheGroupSpec(
                worker_layer_names,
                group_spec,
                is_eagle_group=group.is_eagle_group,
                host_resident=group.host_resident,
                enable_kv_transfer=group.enable_kv_transfer,
                role=group.role,
            )
        )
    return projected_groups


def get_kv_cache_configs(
    vllm_config: VllmConfig,
    kv_cache_specs: list[dict[str, KVCacheSpec]],
    available_memory: list[int],
) -> list[KVCacheConfig]:
    """Generates the KV cache configurations for a model.
    Since we use a shared centralized controller for all workers, we need the
    `kv_cache_config` to be consistent across all workers to make sure
    the KV cache allocation can be applied to all workers. However, different
    workers may have different memory available, and different type of layers
    (when pipeline parallel is enabled). To handle the difference between
    workers, the current implementation is:
    1. Merge the KV cache specs of all workers to get the KVCacheSpecs for
       the whole model.
    2. Generate the KV cache groups based on the layer ratio of the whole model.
       This also handles spec unification for hybrid models.
    3. Handle auto-fit max_model_len and memory checks using per-worker
       projected groups to account for PP sharding.
    4. Generate the KV cache configs for each worker based on the KV cache
       grouping strategy. (This is reasonable because the layer ratio of
       different PP stages are similar.)
    5. Change the num_blocks of each worker to the smallest among all workers
       and shrink tensor sizes proportionally to avoid allocating unused memory.

    Args:
        vllm_config: The global VllmConfig
        kv_cache_specs: List of dict[layer_name, KVCacheSpec] for each worker.
        available_memory: Memory available for KV cache in bytes for each
            worker.

    Returns:
        The generated KVCacheConfigs for each worker.

    """
    from vllm.v1.core.kv_cache_config_builder import get_kv_cache_config_builder

    builder = get_kv_cache_config_builder(vllm_config)

    # Merge the KV cache specs of all workers. Different PP stages may have
    # different layer names, and different TP ranks of the same PP stage should
    # have the same KV cache spec.
    merged_kv_cache_specs: dict[str, KVCacheSpec] = {}
    for kv_cache_spec_one_worker in kv_cache_specs:
        for layer_name, layer_spec in kv_cache_spec_one_worker.items():
            if layer_name not in merged_kv_cache_specs:
                merged_kv_cache_specs[layer_name] = layer_spec
            else:
                assert merged_kv_cache_specs[layer_name] == layer_spec, (
                    "The KV cache specs for the same layer are different "
                    "across workers. This is not supported yet."
                )

    # Check if the KV cache specs are registered correctly.
    # This is to prevent that some layers are initialized with unregistered specs.
    KVCacheSpecRegistry.check_kv_cache_spec_registry(merged_kv_cache_specs)

    # When speculating with more than 1 speculative module (e.g. multi-layered MTP)
    # tag every SlidingWindowSpec with how many extra tokens to retain in the window.
    extra_retained_tokens = max(0, vllm_config.num_prefill_lookahead_tokens - 1)
    for layer_name, layer_spec in merged_kv_cache_specs.items():
        if isinstance(layer_spec, SlidingWindowSpec):
            merged_kv_cache_specs[layer_name] = replace(
                layer_spec, extra_retained_tokens=extra_retained_tokens
            )

    # Get global KV cache groups. This also handles spec unification for
    # hybrid models when disable_hybrid_kv_cache_manager is enabled.
    # After this call, merged_kv_cache_specs may be modified in-place.
    if vllm_config.cache_config.kv_cache_allocator == "large_page":
        return _get_kv_cache_configs_large_page(
            vllm_config,
            builder,
            merged_kv_cache_specs,
            kv_cache_specs,
            available_memory,
        )
    global_kv_cache_groups = get_kv_cache_groups_from_builder(
        builder, vllm_config, merged_kv_cache_specs
    )
    spec_config = vllm_config.speculative_config
    if spec_config is not None and spec_config.use_eagle_block_drop():
        draft_layer_names = {
            name for name, spec in merged_kv_cache_specs.items() if spec.is_draft
        }
        for group in global_kv_cache_groups:
            group.is_eagle_group = not draft_layer_names.isdisjoint(group.layer_names)
        _warn_if_unannotated_eagle_mamba(spec_config, global_kv_cache_groups)

    # If original_max_model_len was -1, automatically
    # determine the maximum model length that fits in available GPU memory.
    # We use per-worker projected groups to account for PP sharding.
    projected_groups_per_worker = [
        _project_kv_cache_groups_to_worker(global_kv_cache_groups, worker_spec)
        for worker_spec in kv_cache_specs
    ]

    # If `num_gpu_blocks_override` is set, the cache size that will actually
    # be allocated is decoupled from the profiled `available_memory`. Reflect
    # that in `available_memory` here so auto-fit, the admission check, and
    # the per-worker config builder all plan against the same effective
    # capacity.
    override = vllm_config.cache_config.num_gpu_blocks_override
    if override is not None:
        adjusted_memory: list[int] = []
        for groups, avail_mem in zip(projected_groups_per_worker, available_memory):
            if not groups:
                adjusted_memory.append(avail_mem)
                continue
            bytes_per_block = _pool_bytes_per_block_from_builder(
                builder, vllm_config, groups
            )
            logger.info(
                "Overriding num_gpu_blocks=%d with num_gpu_blocks_override=%d",
                avail_mem // bytes_per_block,
                override,
            )
            adjusted_memory.append(override * bytes_per_block)
        available_memory = adjusted_memory

    # Reserve the null block BlockPool permanently holds back, so auto-fit and
    # the capacity check both plan against usable blocks. Allocation below
    # still uses the full memory.
    check_memory = [
        (
            avail_mem - _pool_bytes_per_block_from_builder(builder, vllm_config, groups)
            if groups
            else avail_mem
        )
        for groups, avail_mem in zip(projected_groups_per_worker, available_memory)
    ]

    if vllm_config.model_config.original_max_model_len == -1:
        _auto_fit_max_model_len(
            builder, vllm_config, projected_groups_per_worker, check_memory
        )

    # Check if the available memory is enough per worker.
    for groups, avail_mem in zip(projected_groups_per_worker, check_memory):
        if not groups:
            continue
        _check_enough_kv_cache_memory(
            avail_mem,
            partial(_max_memory_usage_bytes_from_groups, builder, vllm_config, groups),
            vllm_config.model_config.max_model_len,
            partial(_estimate_max_model_len_from_groups, builder, vllm_config, groups),
        )

    kv_cache_configs: list[KVCacheConfig] = []
    for projected_groups, kv_cache_spec_one_worker, available_memory_one_worker in zip(
        projected_groups_per_worker, kv_cache_specs, available_memory
    ):
        assert sum(len(group.layer_names) for group in projected_groups) == len(
            kv_cache_spec_one_worker
        ), "Some layers are not assigned to any group."
        bytes_per_block = _pool_bytes_per_block_from_builder(
            builder, vllm_config, projected_groups
        )
        num_blocks = (
            available_memory_one_worker // bytes_per_block if projected_groups else 1
        )
        kv_cache_configs.append(
            builder.get_kv_cache_config_from_groups(
                vllm_config, projected_groups, num_blocks
            )
        )

    # Change the num_blocks of each rank to the smallest among all ranks.
    # We also need to shrink the tensor size proportionally to avoid
    # allocating unused memory.
    min_num_blocks = min(
        kv_cache_config.num_blocks for kv_cache_config in kv_cache_configs
    )
    for i, kv_cache_config in enumerate(kv_cache_configs):
        if kv_cache_config.num_blocks == min_num_blocks:
            continue
        # Re-plan with exactly the smallest rank's block count, so strides and
        # offsets stay consistent with the shrunken allocation.
        kv_cache_configs[i] = builder.get_kv_cache_config_from_groups(
            vllm_config,
            kv_cache_config.kv_cache_groups,
            min_num_blocks,
        )

    for kv_cache_config in kv_cache_configs:
        kv_cache_config.kv_tp_replicas = kv_cache_groups_tp_replicas(
            kv_cache_config.kv_cache_groups,
            vllm_config.parallel_config.tensor_parallel_size,
            vllm_config.parallel_config.decode_context_parallel_size,
        )

    return kv_cache_configs


class DefaultKVCacheConfigBuilder(KVCacheConfigBuilder):
    """Default grouping and placement, backed by the module-level functions here.

    .. note::
        This class is part of the experimental ``KVCacheConfigBuilder``
        interface and is subject to change.

    Subclasses override only the hooks they need. The planning flow and its
    invariants (``num_gpu_blocks_override``, null-block reservation, auto-fit,
    admission checks, cross-rank convergence) stay in the Core-owned
    :func:`get_kv_cache_configs`.
    """

    def get_kv_cache_groups(
        self,
        vllm_config: VllmConfig,
        kv_cache_spec: dict[str, KVCacheSpec],
    ) -> list[KVCacheGroupSpec]:
        """See ``KVCacheConfigBuilder.get_kv_cache_groups``.

        A model implementing ``SupportsKVCacheGrouping`` owns the grouping
        decision; otherwise the default grouping is used.
        """
        if not vllm_config.model_config.supports_kv_cache_grouping:
            return get_kv_cache_groups(vllm_config, kv_cache_spec)
        if vllm_config.cache_config.kv_cache_allocator == "large_page":
            raise ValueError(
                "A model-declared get_kv_cache_groups cannot be combined "
                "with kv_cache_allocator='large_page', which owns grouping "
                "and placement. Remove one of them."
            )
        if vllm_config.attention_config.hisparse_config is not None:
            raise ValueError(
                "A model-declared get_kv_cache_groups cannot be combined "
                "with HiSparse, which owns its own KV cache grouping. "
                "Remove one of them."
            )
        model_cls, _ = vllm_config.model_config.registry.resolve_model_cls(
            vllm_config.model_config.architectures,
            vllm_config.model_config,
        )
        return model_cls.get_kv_cache_groups(vllm_config, kv_cache_spec)

    def get_kv_cache_config_from_groups(
        self,
        vllm_config: VllmConfig,
        kv_cache_groups: list[KVCacheGroupSpec],
        num_blocks: int,
    ) -> KVCacheConfig:
        """See ``KVCacheConfigBuilder.get_kv_cache_config_from_groups``."""
        if len(kv_cache_groups) == 0:
            # Attention free models do not have KV cache.
            # Return num_blocks=1 as BlockPool always needs a null_block.
            return KVCacheConfig(
                num_blocks=1,
                kv_cache_tensors=[],
                kv_cache_groups=kv_cache_groups,
                prefix_cache_retention_interval=(
                    vllm_config.cache_config.prefix_cache_retention_interval
                ),
            )
        if vllm_config.cache_config.kv_cache_allocator == "large_page":
            # Profiling path: a worker plans its own groups locally.
            return get_large_page_kv_cache_config_from_groups(
                vllm_config, kv_cache_groups, min_blocks=num_blocks
            )
        bytes_per_block = _pool_bytes_per_block(vllm_config, kv_cache_groups)
        return get_kv_cache_config_from_groups(
            vllm_config, kv_cache_groups, num_blocks * bytes_per_block
        )
