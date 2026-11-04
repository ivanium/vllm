# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

from vllm.config import VllmConfig
from vllm.config.kv_transfer import hisparse_host_pool_gib
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.v1.hisparse.runtime import (
    ResolvedHiSparseConfig,
    get_hisparse_host_block_stride,
    use_shared_hisparse_host_pool,
)
from vllm.v1.hisparse.types import ACTIVE_TAIL_PAGES
from vllm.v1.kv_cache_interface import (
    HiSparseHotSpec,
    HiSparseResidentSpec,
    KVCacheConfig,
    KVCacheGroupRole,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVCacheTensor,
    MLAAttentionSpec,
    SparseCacheRole,
    large_pages_needed,
)

logger = init_logger(__name__)

HISPARSE_HOT_SUFFIX = ".hisparse_hot"
HISPARSE_RESIDENT_SUFFIX = ".hisparse_resident"


def get_hisparse_host_pool_bytes(vllm_config: VllmConfig) -> int:
    host_pool_gib = hisparse_host_pool_gib(vllm_config.kv_transfer_config)
    if host_pool_gib is None:
        raise ValueError("HiSparse requires HiSparseConnector with host_pool_gib")
    return int(host_pool_gib * 2**30)


def _partition_hisparse_specs(
    mla_specs: dict[str, MLAAttentionSpec],
) -> tuple[dict[str, MLAAttentionSpec], dict[str, MLAAttentionSpec]]:
    if any(spec.model_version == "deepseek_v4" for spec in mla_specs.values()):
        raise ValueError("HiSparse does not support DeepSeek V4.")
    source_specs = {
        name: spec
        for name, spec in mla_specs.items()
        if spec.cache_role is SparseCacheRole.SPARSE
    }
    indexer_specs = {
        name: spec
        for name, spec in mla_specs.items()
        if spec.cache_role is SparseCacheRole.INDEXER
    }
    if not source_specs or not indexer_specs:
        raise ValueError("HiSparse requires sparse-MLA and indexer cache specs.")
    return source_specs, indexer_specs


def _lay_out_hisparse_groups(
    vllm_config: VllmConfig,
    source_specs: dict[str, MLAAttentionSpec],
    indexer_specs: dict[str, MLAAttentionSpec],
    block_size: int,
) -> list[KVCacheGroupSpec]:
    """The groups HiSparse allocates: the host source group, then the GPU
    groups sharing the device pool. Every HiSparse group is planned at the
    global block size: the host tier pages KV at it, and prefix-cache hashes
    are computed at it."""
    from vllm.v1.core.kv_cache_planning import (
        group_key_spec,
        kernel_block_size_supported,
    )

    for name, spec in source_specs.items():
        if spec.kernel_block_sizes and not kernel_block_size_supported(
            spec.kernel_block_sizes, block_size
        ):
            raise ValueError(
                f"HiSparse requires the resolved block size {block_size} to be "
                f"a kernel block size of every sparse-MLA backend (its host "
                f"tier pages KV at the block size), but {name} supports "
                f"{list(spec.kernel_block_sizes)}. Set --block-size to one of "
                "them."
            )
    config = ResolvedHiSparseConfig.from_vllm_config(
        vllm_config,
        max(
            spec.topk_buffer_width
            if spec.topk_buffer_width is not None
            else vllm_config.model_config.hf_config.index_topk
            for spec in source_specs.values()
        ),
        block_size,
    )
    assert config is not None

    indexer_layers = list(indexer_specs)
    assert len(set(indexer_specs.values())) == 1
    indexer_group = KVCacheGroupSpec(
        indexer_layers,
        group_key_spec(indexer_specs[indexer_layers[0]]),
        enable_kv_transfer=True,
        role=KVCacheGroupRole.HISPARSE_INDEXER,
        layer_kv_cache_specs=dict(indexer_specs),
    )

    indexer_page = sum(spec.page_size_bytes for spec in indexer_specs.values())
    hot_blocks_per_request = cdiv(config.device_buffer_size, block_size)
    hot_units: list[list[tuple[str, MLAAttentionSpec]]] = []
    for layer_name, layer_spec in source_specs.items():
        if layer_spec.is_index_group_leader or not hot_units:
            hot_units.append([])
        hot_units[-1].append((f"{layer_name}{HISPARSE_HOT_SUFFIX}", layer_spec))

    resident_groups: list[KVCacheGroupSpec] = []
    hot_groups: list[KVCacheGroupSpec] = []

    def append_hot_group(layers: list[tuple[str, MLAAttentionSpec]]) -> None:
        page_sizes = {spec.page_size_bytes for _, spec in layers}
        if len(page_sizes) != 1:
            raise ValueError(
                "HiSparse hot-cache groups require one page size, got "
                f"{sorted(page_sizes)}."
            )
        page_size = page_sizes.pop()
        names = [name for name, _ in layers]
        resident_names = [
            name[: -len(HISPARSE_HOT_SUFFIX)] + HISPARSE_RESIDENT_SUFFIX
            for name in names
        ]
        resident_spec = HiSparseResidentSpec(
            block_size=block_size,
            page_size=page_size,
        )
        resident_groups.append(
            KVCacheGroupSpec(
                resident_names,
                resident_spec,
                enable_kv_transfer=False,
                layer_kv_cache_specs=dict.fromkeys(resident_names, resident_spec),
            )
        )
        hot_spec = HiSparseHotSpec(
            block_size=block_size,
            page_size=page_size,
            blocks_per_request=hot_blocks_per_request,
        )
        hot_groups.append(
            KVCacheGroupSpec(
                names,
                hot_spec,
                enable_kv_transfer=False,
                layer_kv_cache_specs=dict.fromkeys(names, hot_spec),
            )
        )

    current: list[tuple[str, MLAAttentionSpec]] = []
    current_page = 0
    for unit in hot_units:
        unit_page = sum(spec.page_size_bytes for _, spec in unit)
        if current and current_page + unit_page > indexer_page:
            append_hot_group(current)
            current = []
            current_page = 0
        current.extend(unit)
        current_page += unit_page
    if current:
        append_hot_group(current)

    source_layers = list(source_specs)
    source_group = KVCacheGroupSpec(
        source_layers,
        group_key_spec(source_specs[source_layers[0]]),
        role=KVCacheGroupRole.HISPARSE_SOURCE,
        host_resident=True,
        enable_kv_transfer=True,
        layer_kv_cache_specs=dict(source_specs),
    )
    return [source_group, indexer_group, *resident_groups, *hot_groups]


def get_hisparse_kv_cache_groups(
    vllm_config: VllmConfig,
    kv_cache_spec: dict[str, KVCacheSpec],
    block_size: int | None = None,
) -> tuple[list[KVCacheGroupSpec], dict[str, KVCacheSpec]] | None:
    """Split HiSparse layers out of ``kv_cache_spec`` and group them.

    Returns the HiSparse groups (host source group first, then indexer,
    resident and hot groups) and the remaining per-layer specs, which the
    caller buckets with the generic grouping. Returns None when HiSparse is
    disabled or the model has no sparse-MLA layers. Derived resident/hot specs
    in ``kv_cache_spec`` (see ``resolve_hisparse_specs``) are laid out again
    from the attention specs they belong to.

    ``block_size`` is the planner's global block size B; resolved from the
    specs when not given (the worker's spec-resolution path).
    """
    attention_config = getattr(vllm_config, "attention_config", None)
    if attention_config is None or attention_config.hisparse_config is None:
        return None

    mla_specs: dict[str, MLAAttentionSpec] = {
        name: spec
        for name, spec in kv_cache_spec.items()
        if isinstance(spec, MLAAttentionSpec)
    }
    if not mla_specs:
        return None
    other_specs = {
        name: spec
        for name, spec in kv_cache_spec.items()
        if not isinstance(
            spec, (MLAAttentionSpec, HiSparseResidentSpec, HiSparseHotSpec)
        )
    }
    from vllm.v1.core.kv_cache_planning import (
        normalize_layer_spec,
        select_global_block_size,
    )

    if block_size is None:
        block_size = select_global_block_size(vllm_config, kv_cache_spec)
    # HiSparse MLA layers are prefix-cacheable and not DCP-sharded (HiSparse
    # rejects DCP at config time), so the global block size is theirs.
    mla_specs = {
        name: normalize_layer_spec(spec, block_size) for name, spec in mla_specs.items()
    }
    source_specs, indexer_specs = _partition_hisparse_specs(mla_specs)
    groups = _lay_out_hisparse_groups(
        vllm_config, source_specs, indexer_specs, block_size
    )
    return groups, other_specs


@dataclass(frozen=True)
class HiSparseLayout:
    source_group: KVCacheGroupSpec
    host_num_blocks: int
    host_block_stride: int
    shared_host_pool: bool


def create_hisparse_layout(
    vllm_config: VllmConfig,
    groups: list[KVCacheGroupSpec],
    host_budget: int,
) -> HiSparseLayout:
    """Size the host pool for groups laid out by `get_hisparse_kv_cache_groups`."""
    (source_group,) = [group for group in groups if group.host_resident]
    layer_specs = source_group.layer_kv_cache_specs
    assert layer_specs is not None
    group_page = sum(spec.page_size_bytes for spec in layer_specs.values())
    shared_host_pool = use_shared_hisparse_host_pool(vllm_config)
    host_block_stride = get_hisparse_host_block_stride(
        group_page,
        use_shared_host_pool=shared_host_pool,
    )
    host_num_blocks = host_budget // host_block_stride
    if host_num_blocks <= 0:
        raise ValueError("HiSparse has no allocatable host blocks.")
    # Every computed page needs a host block, so one request at max_model_len
    # must fit alongside the pool's null block.
    gpu_block_size = source_group.kv_cache_spec.block_size
    min_host_blocks = cdiv(vllm_config.model_config.max_model_len, gpu_block_size) + 1
    if host_num_blocks < min_host_blocks:
        raise ValueError(
            f"HiSparse host pool has {host_num_blocks} blocks but max_model_len "
            f"needs {min_host_blocks}; increase host_pool_gib."
        )

    return HiSparseLayout(
        source_group=source_group,
        host_num_blocks=host_num_blocks,
        host_block_stride=host_block_stride,
        shared_host_pool=shared_host_pool,
    )


def add_hisparse_host_config(
    kv_cache_config: KVCacheConfig,
    hisparse_layout: HiSparseLayout,
) -> KVCacheConfig:
    """Prepend the host source group and its tensors to a device-pool config.

    The host tier is a private, densely packed pool: one region per source
    layer, block ``b`` of a layer at ``b * page_size`` within its region. Its
    group holds no device rows (``row_bytes=0``); its block table maps
    one-to-one to host blocks (see ``KVCacheConfig.group_geometry``).
    """
    source_group = hisparse_layout.source_group
    layer_specs = source_group.layer_kv_cache_specs
    assert layer_specs is not None
    host_num_blocks = hisparse_layout.host_num_blocks
    host_size = hisparse_layout.host_block_stride * host_num_blocks
    host_tensors: list[KVCacheTensor] = []
    offset = 0
    for layer_name, spec in layer_specs.items():
        layer_bytes = spec.page_size_bytes * host_num_blocks
        host_tensors.append(
            KVCacheTensor(
                size=host_size,
                layers=[layer_name],
                layer_stride=layer_bytes,
                block_stride=spec.page_size_bytes,
                offset=offset,
                host_resident=True,
            )
        )
        offset += layer_bytes
    kv_cache_config.kv_cache_groups.insert(
        0,
        KVCacheGroupSpec(
            source_group.layer_names,
            source_group.kv_cache_spec,
            is_eagle_group=source_group.is_eagle_group,
            host_resident=True,
            enable_kv_transfer=source_group.enable_kv_transfer,
            role=source_group.role,
            num_blocks=host_num_blocks,
            layer_kv_cache_specs=dict(layer_specs),
        ),
    )
    kv_cache_config.kv_cache_tensors = [
        *host_tensors,
        *kv_cache_config.kv_cache_tensors,
    ]
    kv_cache_config.hisparse_host_num_blocks = host_num_blocks
    kv_cache_config.hisparse_host_block_stride = hisparse_layout.host_block_stride
    kv_cache_config.hisparse_shared_host_pool = hisparse_layout.shared_host_pool
    return kv_cache_config


def get_hisparse_steady_state_concurrency(
    vllm_config: VllmConfig, kv_cache_config: KVCacheConfig
) -> float:
    """Max concurrency at max_model_len once running requests read from host.

    A request reading from host pins only its active tail of each resident
    group, but admitting one still takes its full in-flight window, so the
    bound is all-but-one requests at steady state plus one being admitted.
    Device groups draw whole large pages (at least ``blocks_per_page``
    blocks each), so device demand is counted in pages.
    """
    admission_pages = 0
    steady_pages = 0
    host_blocks = 0
    for group in kv_cache_config.kv_cache_groups:
        spec = group.kv_cache_spec
        required = cdiv(spec.max_memory_usage_bytes(vllm_config), spec.page_size_bytes)
        if group.host_resident:
            host_blocks += required
            continue
        admission_pages += large_pages_needed(required, group.blocks_per_page)
        if isinstance(spec, HiSparseResidentSpec):
            required = min(required, ACTIVE_TAIL_PAGES)
        steady_pages += large_pages_needed(required, group.blocks_per_page)
    assert kv_cache_config.hisparse_host_num_blocks is not None
    num_pages = kv_cache_config.num_pages
    if num_pages < admission_pages:
        gpu_concurrency = num_pages / admission_pages
    else:
        gpu_concurrency = 1 + (num_pages - admission_pages) / steady_pages
    return min(gpu_concurrency, kv_cache_config.hisparse_host_num_blocks / host_blocks)
