# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV cache planning helpers and default builder logic."""

from collections import defaultdict
from dataclasses import replace
from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.utils.mem_utils import format_gib
from vllm.v1.kv_cache_interface import (
    ChunkedLocalAttentionSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVCacheTensor,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.kv_cache_config_builder import KVCacheConfigBuilder

logger = init_logger(__name__)


def create_kv_cache_group_specs(
    kv_cache_spec: dict[str, KVCacheSpec],
    grouped_layer_names: list[list[str]],
) -> list[KVCacheGroupSpec]:
    """
    Create KVCacheGroupSpec object for each kv cache group layer.
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
    """
    Whether all layers in the given KVCacheSpec have the same KV cache spec.
    Note that we regard FullAttentionSpec with and without sliding window as
    the same type.

    Args:
        kv_cache_spec: The kv cache spec of each attention layer in the model

    Returns:
        True if all layers have the same type, False otherwise.
    """
    if not kv_cache_spec:
        # Encoder-only models do not have KV cache, so the spec can be
        # regarded as uniform.
        return True
    try:
        kv_cache_spec_values = list(kv_cache_spec.values())
        _ = kv_cache_spec_values[0].merge(kv_cache_spec_values)
    except AssertionError:
        return False
    return True


def _get_kv_cache_groups_uniform_spec(
    kv_cache_specs: dict[str, KVCacheSpec],
) -> list[KVCacheGroupSpec]:
    """
    Generates the KV cache configuration for a model with the same KV cache
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
    """
    Generates the KV cache configuration for a model with one type of KV cache
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
    """
    Unify the page size of the given KVCacheSpec. If the page size of all layers
    are the same, return the original KVCacheSpec. If not same, unify the page
    size by increasing the block size of layers with smaller page size. Raise
    NotImplementedError if failed to unify the page size.

    Args:
        kv_cache_spec: The KVCacheSpec of each attention layer in the model

    Returns:
        The updated KVCacheSpec with the same page_size_bytes.
    """
    page_sizes = {layer.page_size_bytes for layer in kv_cache_spec.values()}
    if len(page_sizes) <= 1:
        return kv_cache_spec

    max_page_size = max(page_sizes)
    new_kv_cache_spec = {}
    for layer_name, layer_spec in kv_cache_spec.items():
        if layer_spec.page_size_bytes == max_page_size:
            new_kv_cache_spec[layer_name] = layer_spec
        else:
            layer_page_size = layer_spec.page_size_bytes
            if max_page_size % layer_page_size != 0:
                raise NotImplementedError(
                    "The page size of the layer is not divisible by the "
                    "maximum page size. Cannot unify by adjusting block_size."
                )
            ratio = max_page_size // layer_page_size
            new_block_size = layer_spec.block_size * ratio
            new_spec = replace(layer_spec, block_size=new_block_size)
            assert new_spec.page_size_bytes == max_page_size
            new_kv_cache_spec[layer_name] = new_spec
    return new_kv_cache_spec


def is_kv_cache_type_attention_free(kv_cache_spec: dict[str, KVCacheSpec]) -> bool:
    # kv_cache_spec is an empty dict for attention free models
    return not kv_cache_spec


def _get_kv_cache_groups_uniform_page_size(
    kv_cache_spec: dict[str, KVCacheSpec],
) -> list[KVCacheGroupSpec]:
    """
    Generates the KV cache groups for hybrid models with multiple
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
    same_type_layers: dict[KVCacheSpec, list[str]] = defaultdict(list)
    for layer_name, layer_spec in kv_cache_spec.items():
        same_type_layers[layer_spec].append(layer_name)

    min_num_layers = min(len(layers) for layers in same_type_layers.values())
    group_size = min_num_layers
    max_num_layers = max(len(layers) for layers in same_type_layers.values())
    if max_num_layers < min_num_layers * 1.5:
        group_size = max_num_layers

    grouped_layers = []
    for layers in same_type_layers.values():
        num_padding_layers = group_size - len(layers) % group_size
        if num_padding_layers != group_size:
            logger.warning(
                "Add %d padding layers, may waste at most %.2f%% KV cache memory",
                num_padding_layers,
                num_padding_layers / len(layers) * 100,
            )
        num_groups = cdiv(len(layers), group_size)
        for i in range(num_groups):
            grouped_layers.append(layers[i::num_groups])
    return create_kv_cache_group_specs(kv_cache_spec, grouped_layers)


def unify_hybrid_kv_cache_specs(kv_cache_spec: dict[str, KVCacheSpec]) -> None:
    """
    This function tries to convert the KV cache specs to one type if the model
    is a hybrid model with multiple type of KV cache. It will convert all
    SlidingWindowSpec to FullAttentionSpec if both types are present.

    Args:
        kv_cache_spec: The kv cache spec of each attention layer in the model
    """
    if is_kv_cache_spec_uniform(
        kv_cache_spec
    ) or UniformTypeKVCacheSpecs.is_uniform_type(kv_cache_spec):
        return

    logger.warning(
        "Hybrid KV cache manager is disabled for this hybrid model, "
        "This means we do not enable any optimizations for saving KV cache "
        "memory (e.g., dropping the KV cache outside the sliding window). "
        "The compute of layers like sliding window is still saved."
    )

    has_full_attention = any(
        isinstance(spec, FullAttentionSpec) for spec in kv_cache_spec.values()
    )
    has_sliding_window = any(
        isinstance(spec, SlidingWindowSpec) for spec in kv_cache_spec.values()
    )
    has_chunked_local_attention = any(
        isinstance(spec, ChunkedLocalAttentionSpec) for spec in kv_cache_spec.values()
    )
    if has_full_attention and (has_sliding_window or has_chunked_local_attention):
        for layer_name, spec in kv_cache_spec.items():
            if isinstance(spec, SlidingWindowSpec):
                kv_cache_spec[layer_name] = FullAttentionSpec(
                    block_size=spec.block_size,
                    num_kv_heads=spec.num_kv_heads,
                    head_size=spec.head_size,
                    dtype=spec.dtype,
                    sliding_window=spec.sliding_window,
                    page_size_padded=spec.page_size_padded,
                )
            elif isinstance(spec, ChunkedLocalAttentionSpec):
                kv_cache_spec[layer_name] = FullAttentionSpec(
                    block_size=spec.block_size,
                    num_kv_heads=spec.num_kv_heads,
                    head_size=spec.head_size,
                    dtype=spec.dtype,
                    attention_chunk_size=spec.attention_chunk_size,
                    page_size_padded=spec.page_size_padded,
                )

    if not (
        is_kv_cache_spec_uniform(kv_cache_spec)
        or UniformTypeKVCacheSpecs.is_uniform_type(kv_cache_spec)
    ):
        raise ValueError(
            "Hybrid KV cache manager is disabled but failed to "
            "convert the KV cache specs to one unified type."
        )


def get_kv_cache_groups(
    vllm_config: "VllmConfig",
    kv_cache_spec: dict[str, KVCacheSpec],
) -> list[KVCacheGroupSpec]:
    """
    Split the layers in the model into groups with the same KV cache spec.

    Args:
        vllm_config: The global VllmConfig
        kv_cache_spec: The kv cache spec of each attention layer in the model

    Returns:
        The generated KVCacheGroups
    """
    if vllm_config.scheduler_config.disable_hybrid_kv_cache_manager:
        unify_hybrid_kv_cache_specs(kv_cache_spec)

    if is_kv_cache_type_attention_free(kv_cache_spec):
        return []

    if is_kv_cache_spec_uniform(kv_cache_spec):
        return _get_kv_cache_groups_uniform_spec(kv_cache_spec)

    if uniform_spec := UniformTypeKVCacheSpecs.from_specs(kv_cache_spec):
        return _get_kv_cache_groups_uniform_type(uniform_spec)

    kv_cache_spec = unify_kv_cache_spec_page_size(kv_cache_spec)
    return _get_kv_cache_groups_uniform_page_size(kv_cache_spec)


def _maybe_override_num_blocks(vllm_config: "VllmConfig", num_blocks: int) -> int:
    """
    Override the number of kv cache blocks if `num_gpu_blocks_override` is set.
    """
    if vllm_config.cache_config.num_gpu_blocks_override is not None:
        num_gpu_blocks_override = vllm_config.cache_config.num_gpu_blocks_override
        logger.info(
            "Overriding num_gpu_blocks=%d with num_gpu_blocks_override=%d",
            num_blocks,
            num_gpu_blocks_override,
        )
        num_blocks = num_gpu_blocks_override

    return num_blocks


def _get_num_blocks(
    vllm_config: "VllmConfig",
    num_layers: int,
    available_memory: int,
    page_size: int,
) -> int:
    """
    Get the number of kv cache blocks.

    Args:
        vllm_config: The global VllmConfig
        num_layers: The number of layers
        available_memory: Memory available for KV cache in bytes.
        page_size: The page size of the KV cache.
    """
    num_blocks = int(available_memory // page_size // num_layers)
    num_blocks = max(num_blocks, 0)
    num_blocks = _maybe_override_num_blocks(vllm_config, num_blocks)
    return num_blocks


def _get_uniform_page_size(kv_cache_specs: list[KVCacheSpec]) -> int:
    """
    Get the page size of the KV cache.
    """
    page_sizes = {layer.page_size_bytes for layer in kv_cache_specs}
    assert len(page_sizes) == 1
    return page_sizes.pop()


def _max_memory_usage_bytes_from_groups(
    vllm_config: "VllmConfig",
    kv_cache_groups: list[KVCacheGroupSpec],
) -> int:
    """
    Calculate maximum memory usage in bytes from KV cache groups.

    This correctly accounts for padding in hybrid models. For example, if a
    model has 8 full attention layers and 9 sliding window layers, they will
    be padded to 9 full + 9 sliding window for uniform group sizes.
    """
    if not kv_cache_groups:
        return 0

    if len(kv_cache_groups) == 1 and isinstance(
        kv_cache_groups[0].kv_cache_spec, UniformTypeKVCacheSpecs
    ):
        per_layer_specs = kv_cache_groups[0].kv_cache_spec.kv_cache_specs
        return sum(
            spec.max_memory_usage_bytes(vllm_config)
            for spec in per_layer_specs.values()
        )

    group_size = max(len(group.layer_names) for group in kv_cache_groups)
    page_size = _get_uniform_page_size(
        [group.kv_cache_spec for group in kv_cache_groups]
    )
    blocks_needed = sum(
        cdiv(group.kv_cache_spec.max_memory_usage_bytes(vllm_config), page_size)
        for group in kv_cache_groups
    )

    return group_size * page_size * blocks_needed


def _estimate_max_model_len_from_groups(
    vllm_config: "VllmConfig",
    kv_cache_groups: list[KVCacheGroupSpec],
    available_memory: int,
) -> int:
    """
    Binary search for the maximum model length that fits in available memory.
    Returns 0 if even 1 token doesn't fit.
    """
    original_max = vllm_config.model_config.max_model_len

    def fits(model_len: int) -> bool:
        vllm_config.model_config.max_model_len = model_len
        return (
            _max_memory_usage_bytes_from_groups(vllm_config, kv_cache_groups)
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


def _merge_kv_cache_specs(
    kv_cache_specs: list[dict[str, KVCacheSpec]],
) -> dict[str, KVCacheSpec]:
    """Merge per-worker specs into one global spec map."""
    merged: dict[str, KVCacheSpec] = {}
    for kv_cache_spec_one_worker in kv_cache_specs:
        for layer_name, layer_spec in kv_cache_spec_one_worker.items():
            if layer_name not in merged:
                merged[layer_name] = layer_spec
            else:
                assert merged[layer_name] == layer_spec, (
                    "The KV cache specs for the same layer are different "
                    "across workers. This is not supported yet."
                )
    return merged


def _project_kv_cache_groups_to_worker(
    global_kv_cache_groups: list[KVCacheGroupSpec],
    worker_spec: dict[str, KVCacheSpec],
) -> list[KVCacheGroupSpec]:
    """
    Projects global KV cache groups onto a single worker's assigned layers.

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
        group_spec = group.kv_cache_spec
        if worker_layer_names and isinstance(group_spec, UniformTypeKVCacheSpecs):
            group_spec = UniformTypeKVCacheSpecs(
                block_size=group_spec.block_size,
                kv_cache_specs={
                    layer_name: group_spec.kv_cache_specs[layer_name]
                    for layer_name in worker_layer_names
                },
            )
        projected_groups.append(KVCacheGroupSpec(worker_layer_names, group_spec))
    return projected_groups


def _get_projected_kv_cache_groups_per_worker(
    vllm_config: "VllmConfig",
    kv_cache_specs: list[dict[str, KVCacheSpec]],
) -> list[list[KVCacheGroupSpec]]:
    """Group merged specs once, then project those groups onto each worker."""
    merged_specs = _merge_kv_cache_specs(kv_cache_specs)
    global_groups = get_kv_cache_groups(vllm_config, merged_specs)
    return [
        _project_kv_cache_groups_to_worker(global_groups, worker_spec)
        for worker_spec in kv_cache_specs
    ]


def get_kv_cache_config_from_groups(
    vllm_config: "VllmConfig",
    kv_cache_groups: list[KVCacheGroupSpec],
    available_memory: int,
) -> KVCacheConfig:
    """
    Generate the KV cache configuration from the KV cache groups and spec
    of each layer.

    Args:
        vllm_config: The global VllmConfig
        kv_cache_groups: The KV cache groups
        available_memory: Memory available for KV cache in bytes
    Returns:
        The generated KVCacheConfig
    """
    if len(kv_cache_groups) == 0:
        return KVCacheConfig(
            num_blocks=1,
            kv_cache_tensors=[],
            kv_cache_groups=kv_cache_groups,
        )

    if len(kv_cache_groups) == 1 and isinstance(
        kv_cache_groups[0].kv_cache_spec, UniformTypeKVCacheSpecs
    ):
        num_blocks = (
            available_memory // kv_cache_groups[0].kv_cache_spec.page_size_bytes
        )
        num_blocks = _maybe_override_num_blocks(vllm_config, num_blocks)
        per_layer_specs = kv_cache_groups[0].kv_cache_spec.kv_cache_specs
        kv_cache_tensors = [
            KVCacheTensor(
                size=per_layer_specs[layer_name].page_size_bytes * num_blocks,
                shared_by=[layer_name],
            )
            for layer_name in kv_cache_groups[0].layer_names
        ]
    else:
        group_size = max(len(group.layer_names) for group in kv_cache_groups)
        page_size = _get_uniform_page_size(
            [group.kv_cache_spec for group in kv_cache_groups]
        )
        assert group_size > 0, "group_size must be greater than 0"
        num_blocks = _get_num_blocks(
            vllm_config, group_size, available_memory, page_size
        )
        kv_cache_tensors = []
        for i in range(group_size):
            shared_by = []
            for group in kv_cache_groups:
                if i < len(group.layer_names):
                    shared_by.append(group.layer_names[i])
            kv_cache_tensors.append(
                KVCacheTensor(size=page_size * num_blocks, shared_by=shared_by)
            )

    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=kv_cache_tensors,
        kv_cache_groups=kv_cache_groups,
    )


def _report_kv_cache_config(
    vllm_config: "VllmConfig",
    kv_cache_config: KVCacheConfig,
) -> None:
    """
    Log resolved KV cache configuration.

    Args:
        vllm_config: The global VllmConfig
        kv_cache_config: The resolved KV cache configuration
    """
    from vllm.v1.core.kv_cache_utils import get_max_concurrency_for_kv_cache_config

    min_block_size = min(
        group.kv_cache_spec.block_size for group in kv_cache_config.kv_cache_groups
    )

    num_tokens = (
        kv_cache_config.num_blocks
        // len(kv_cache_config.kv_cache_groups)
        * min_block_size
    )
    dcp_size = vllm_config.parallel_config.decode_context_parallel_size
    pcp_size = vllm_config.parallel_config.prefill_context_parallel_size
    if pcp_size * dcp_size > 1:
        num_tokens *= pcp_size * dcp_size
        logger.info(
            "Multiplying the GPU KV cache size by the cp_world_size %d "
            "(pcp_world_size %d * dcp_world_size %d).",
            pcp_size * dcp_size,
            pcp_size,
            dcp_size,
        )
    logger.info_once("GPU KV cache size: %s tokens", f"{num_tokens:,}", scope="local")
    max_concurrency = get_max_concurrency_for_kv_cache_config(
        vllm_config, kv_cache_config
    )
    logger.info_once(
        "Maximum concurrency for %s tokens per request: %.2fx",
        f"{vllm_config.model_config.max_model_len:,}",
        max_concurrency,
        scope="local",
    )


def estimate_max_model_len(
    vllm_config: "VllmConfig",
    kv_cache_specs: list[dict[str, KVCacheSpec]],
    available_memory: list[int],
) -> int:
    """
    Estimates the maximum model length that can fit in the available memory
    using binary search.

    This function temporarily modifies max_model_len during estimation but
    restores the original value before returning, ensuring no side effects.

    Args:
        vllm_config: The global VllmConfig
        kv_cache_specs: List of dict[layer_name, KVCacheSpec] for each worker.
        available_memory: Memory available for KV cache in bytes for each
            worker.

    Returns:
        The estimated maximum model length that can fit in the available memory
        across all workers.
    """
    projected_per_worker = _get_projected_kv_cache_groups_per_worker(
        vllm_config, kv_cache_specs
    )

    fit_across_workers = vllm_config.model_config.max_model_len
    for groups, avail_mem in zip(projected_per_worker, available_memory):
        if not groups:
            continue
        worker_fit = _estimate_max_model_len_from_groups(vllm_config, groups, avail_mem)
        fit_across_workers = min(fit_across_workers, worker_fit)
    return fit_across_workers


def build_kv_cache_configs(
    builder: "KVCacheConfigBuilder",
    vllm_config: "VllmConfig",
    kv_cache_specs: list[dict[str, KVCacheSpec]],
    available_memory: list[int],
) -> list[KVCacheConfig]:
    """
    Generates the KV cache configurations for a model.
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
    3. Handle memory checks using per-worker projected groups to account for
       PP sharding. If memory is insufficient, use the builder to estimate the
       largest max_model_len that can fit.
    4. Generate the KV cache configs for each worker based on the KV cache
       grouping strategy. (This is reasonable because the layer ratio of
       different PP stages are similar.)
    5. Change the num_blocks of each worker to the smallest among all workers
       and shrink tensor sizes proportionally to avoid allocating unused memory.

    Args:
        builder: The KV cache config builder driving this planning flow.
        vllm_config: The global VllmConfig
        kv_cache_specs: List of dict[layer_name, KVCacheSpec] for each worker.
        available_memory: Memory available for KV cache in bytes for each
            worker.

    Returns:
        The generated KVCacheConfigs for each worker.
    """
    projected_per_worker = _get_projected_kv_cache_groups_per_worker(
        vllm_config, kv_cache_specs
    )
    max_fit: int | None = None

    for groups, avail_mem in zip(projected_per_worker, available_memory):
        if not groups:
            continue
        if avail_mem <= 0:
            raise ValueError(
                "No available memory for the cache blocks. "
                "Try increasing `gpu_memory_utilization` when initializing "
                "the engine. See https://docs.vllm.ai/en/latest/configuration/"
                "conserving_memory/ for more details."
            )
        needed = _max_memory_usage_bytes_from_groups(vllm_config, groups)
        if needed > avail_mem:
            if max_fit is None:
                max_fit = builder.estimate_max_model_len(
                    vllm_config, kv_cache_specs, available_memory
                )
            estimated_msg = (
                f"Based on the available memory, "
                f"the estimated maximum model length is {max_fit}. "
                if max_fit > 0
                else ""
            )
            raise ValueError(
                f"To serve at least one request with the models's max seq "
                f"len ({vllm_config.model_config.max_model_len}), "
                f"({format_gib(needed)} GiB KV cache is needed, which is "
                f"larger than the available KV cache memory "
                f"({format_gib(avail_mem)} GiB). {estimated_msg}"
                f"Try increasing `gpu_memory_utilization` or decreasing "
                f"`max_model_len` when initializing the engine. "
                f"See https://docs.vllm.ai/en/latest/configuration/"
                f"conserving_memory/ for more details."
            )

    kv_cache_configs: list[KVCacheConfig] = []
    for groups, worker_spec, avail_mem in zip(
        projected_per_worker, kv_cache_specs, available_memory
    ):
        assert sum(len(group.layer_names) for group in groups) == len(worker_spec), (
            "Some layers are not assigned to any group."
        )
        kv_cache_configs.append(
            get_kv_cache_config_from_groups(vllm_config, groups, avail_mem)
        )

    min_num_blocks = min(
        kv_cache_config.num_blocks for kv_cache_config in kv_cache_configs
    )
    for kv_cache_config in kv_cache_configs:
        num_blocks_old = kv_cache_config.num_blocks
        kv_cache_config.num_blocks = min_num_blocks

        for tensor in kv_cache_config.kv_cache_tensors:
            assert tensor.size % num_blocks_old == 0
            tensor.size = tensor.size // num_blocks_old * min_num_blocks

        if kv_cache_config.kv_cache_groups:
            _report_kv_cache_config(vllm_config, kv_cache_config)

    return kv_cache_configs
