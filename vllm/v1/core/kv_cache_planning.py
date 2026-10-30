# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV cache planning: grouping, sizing, and memory-fit logic."""

import math
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import reduce
from typing import NamedTuple, TypeVar

from vllm.config import SpeculativeConfig, VllmConfig
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv, round_up
from vllm.utils.mem_utils import format_gib
from vllm.v1.attention.backend import MultipleOf
from vllm.v1.core.kv_cache_config_builder import KVCacheConfigBuilder
from vllm.v1.core.kv_cache_utils import large_page_blocks_per_request
from vllm.v1.hisparse.layout import (
    add_hisparse_host_config,
    create_hisparse_layout,
    get_hisparse_host_pool_bytes,
    get_hisparse_kv_cache_groups,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    ChunkedLocalAttentionSpec,
    CircularBufferSpec,
    FullAttentionSpec,
    GroupGeometry,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVCacheTensor,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    SlidingWindowSpec,
    large_pages_needed,
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
    if not kv_cache_spec:
        # Attention-free models have no KV cache to check.
        return
    # Group a copy of the specs since grouping may unify them in-place.
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


def _is_uniform_type(kv_cache_spec: dict[str, KVCacheSpec]) -> bool:
    """Whether all layers have the same type of KV cache spec.

    Uses the registry to determine grouping base classes, so custom specs
    that inherit from FullAttentionSpec are treated as full attention.
    """
    block_sizes = {spec.block_size for spec in kv_cache_spec.values()}
    if len(block_sizes) > 1:
        return False
    if len({spec.dcp_sharded for spec in kv_cache_spec.values()}) > 1:
        return False
    first_spec = next(iter(kv_cache_spec.values()))
    return first_spec.is_uniform_with_collection(kv_cache_spec)


def _promote_local_kv_cache_specs(
    kv_cache_spec: dict[str, KVCacheSpec],
) -> dict[str, KVCacheSpec]:
    """Use full-attention allocation for local-attention cache specs.

    The returned specs affect KV cache management only. Attention modules keep
    their original sliding-window or chunked-local compute behavior.
    """
    promoted_specs = kv_cache_spec.copy()

    if is_kv_cache_spec_uniform(promoted_specs) or _is_uniform_type(promoted_specs):
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
        is_kv_cache_spec_uniform(promoted_specs) or _is_uniform_type(promoted_specs)
    ):
        raise ValueError("Failed to promote local KV cache specs to one unified type.")

    return promoted_specs


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
        if isinstance(group.kv_cache_spec, MambaSpec)
    ]
    if not mamba_groups:
        return
    logger.warning(
        "Speculative decoding (method=%s) is enabled but no KV cache group "
        "could be identified as the draft model's.",
        spec_config.method,
    )


# ---------------------------------------------------------------------------
# Pool geometry and placement.
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


def kernel_block_size_supported(
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
    groups.

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
            raise ValueError(f"Invalid block_size={requested}: " + "; ".join(problems))
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
        f"No feasible block size: L0={base_lcm}, kernel constraints={kernel_preds}"
    )


_PagedSpecT = TypeVar("_PagedSpecT", bound=AttentionSpec | MambaSpec)


def normalize_layer_spec(spec: _PagedSpecT, block_size: int) -> _PagedSpecT:
    """A layer's spec at its planned block size, without planner-written
    padding."""
    if (
        isinstance(spec, MambaSpec)
        and spec.mamba_cache_mode == "align"
        and spec.prefill_checkpoint_alignment
    ):
        # Checkpoint offsets stay on the prefix-cache grid, the block size,
        # which the raw layer alignment divides.
        spec = replace(spec, prefill_checkpoint_alignment=block_size)
    return replace(spec, block_size=block_size, page_size_padded=None)


def group_key_spec(spec: KVCacheSpec) -> KVCacheSpec:
    """The group spec of a normalized layer spec: per-layer backend alignment
    and kernel-block constraints live on ``layer_kv_cache_specs`` instead."""
    return replace(spec, block_stride_alignment=None, kernel_block_sizes=())


class _Bucket(NamedTuple):
    spec: KVCacheSpec
    kernel_block_size: int
    layer_names: list[str]


def _group_block_size(spec: KVCacheSpec, B: int, dcp_size: int) -> int:
    """The block size a spec is planned at.

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


def get_kv_cache_groups(
    vllm_config: VllmConfig,
    kv_cache_spec: dict[str, KVCacheSpec],
) -> list[KVCacheGroupSpec]:
    """Group layers by normalized spec.

    Bucket keys are the specs at their final block size with planner-written
    padding, per-layer backend alignment and kernel-block constraints stripped
    (the layers' own padding, e.g. MLA ``alignment``, is recomputed by
    ``__post_init__`` and kept). Buckets then combine when their specs merge,
    keeping the merge tolerance for window size and non-causal multi-token
    decode. Per-layer specs are kept on the group for placement and never enter
    ``merge``.

    Args:
        vllm_config: The global VllmConfig
        kv_cache_spec: The kv cache spec of each attention layer in the model

    Returns:
        The generated KVCacheGroups

    """
    disable_hybrid = vllm_config.scheduler_config.disable_hybrid_kv_cache_manager
    if disable_hybrid:
        unify_hybrid_kv_cache_specs(kv_cache_spec)
    B = select_global_block_size(vllm_config, kv_cache_spec)
    # The resolved block sizes reach workers through KVCacheConfig (see
    # Worker.initialize_from_config).
    vllm_config.cache_config.adopt_block_size(B)
    if not kv_cache_spec:
        return []

    # HiSparse owns the grouping of its sparse-MLA layers (one host source
    # group, one indexer group, and the derived resident/hot groups); the
    # remaining layers take the generic bucketing below.
    hisparse_groups: list[KVCacheGroupSpec] = []
    if hisparse := get_hisparse_kv_cache_groups(vllm_config, kv_cache_spec, B):
        hisparse_groups, kv_cache_spec = hisparse
        if not kv_cache_spec:
            return hisparse_groups

    dcp_size = vllm_config.parallel_config.decode_context_parallel_size
    # A group needs one row layout, so layers whose backends resolve to
    # different kernel blocks at their block size never share a group (e.g.
    # MiniMax-M3 dense layers at 64 vs sparse layers at 128).
    buckets: dict[tuple[KVCacheSpec, int], list[str]] = {}
    layer_specs: dict[str, KVCacheSpec] = {}
    for layer_name, spec in kv_cache_spec.items():
        key: KVCacheSpec
        if isinstance(spec, (AttentionSpec, MambaSpec)):
            block_size = _group_block_size(spec, B, dcp_size)
            final_spec = normalize_layer_spec(spec, block_size)
            layer_specs[layer_name] = final_spec
            key = group_key_spec(final_spec)
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
            f"group, but the planner formed {len(kv_cache_groups)} groups for "
            "this model."
        )
    return hisparse_groups + kv_cache_groups


@dataclass(frozen=True)
class _LargePageGroupPlan:
    """Placement of one KV cache group in the large-page pool."""

    group: KVCacheGroupSpec
    # The row is the largest over ranks; every rank lays its rows out at it.
    geometry: GroupGeometry
    # Per rank; empty where the group holds no layers.
    layer_offsets: list[dict[str, int]]

    def blocks_per_page(self, large_page_bytes: int) -> int:
        """Fewest blocks of this group one large page holds (n_g)."""
        return min_blocks_per_large_page(large_page_bytes, self.geometry.block_bytes)


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
            kernel_block_size_supported(sizes, kernel_block_size)
            for sizes in layer_sizes
        ):
            return kernel_block_size
    raise ValueError(
        f"No common kernel block size for block size {block_size} "
        f"and kernel constraints {layer_sizes}."
    )


def _group_kernel_block_size(group: KVCacheGroupSpec) -> int:
    """Tokens per kernel block (one row) of a group."""
    return _largest_kernel_block_size(
        group.kv_cache_spec.block_size,
        [
            _kernel_block_constraint(spec)
            for spec in (group.layer_kv_cache_specs or {}).values()
        ],
    )


def _group_row_layout(
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
    do not use. Requests share pages, so a request takes a 1 / blocks_per_page
    share of a page per block."""
    used, taken = 0, 0.0
    for plan in plans:
        blocks = large_page_blocks_per_request(plan.group.kv_cache_spec, vllm_config)
        used += blocks * plan.geometry.block_bytes
        taken += blocks * large_page_bytes / plan.blocks_per_page(large_page_bytes)
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

    HiSparse host groups live in a private host pool and take no large pages,
    so only device groups are planned. Block ids address the same pages on
    every rank only if S and each group's row are the same on every rank, so
    a group's row is its largest over ranks. ``available_memory`` (per rank)
    rules out pages its budget cannot serve one max-model-len request with.
    """
    plans: list[_LargePageGroupPlan] = []
    for group in groups:
        if group.host_resident:
            continue
        kernel_block_size = _group_kernel_block_size(group)
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
                rank_row, rank_unit, layer_offsets = _group_row_layout(
                    layer_specs, kernel_block_size
                )
                row = max(row, rank_row)
                unit = math.lcm(unit, rank_unit)
            offsets.append(layer_offsets)
        geometry = GroupGeometry(
            group.kv_cache_spec.block_size // kernel_block_size, round_up(row, unit)
        )
        plans.append(_LargePageGroupPlan(group, geometry, offsets))

    group_block_bytes = [plan.geometry.block_bytes for plan in plans]
    max_block = max(group_block_bytes)

    def fits_budget(large_page_bytes: int) -> bool:
        needed = _large_page_needed_pages(plans, vllm_config, large_page_bytes)
        return available_memory is None or all(
            needed <= _usable_large_pages(memory // large_page_bytes)
            for memory in available_memory
        )

    # A page loses at most two blocks of a group to its ends, so past this
    # size every page is within the waste bound.
    max_page = math.ceil(2 * max_block / _LARGE_PAGE_MAX_PAGE_WASTE)
    candidates = sorted(
        {
            k * block_bytes
            for block_bytes in group_block_bytes
            for k in range(cdiv(max_block, block_bytes), max_page // block_bytes + 1)
        }
    )
    choice: tuple[bool, float, int] | None = None
    for large_page_bytes in candidates:
        if any(plan.blocks_per_page(large_page_bytes) < 1 for plan in plans):
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
        large_pages_needed(
            large_page_blocks_per_request(plan.group.kv_cache_spec, vllm_config),
            plan.blocks_per_page(large_page_bytes),
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
    granularity over the pool of num_pages pages of large_page_bytes."""
    cache_config = vllm_config.cache_config
    groups: list[KVCacheGroupSpec] = []
    tensors: list[KVCacheTensor] = []
    size = num_pages * large_page_bytes
    for plan in plans:
        group = plan.group
        geometry = plan.geometry
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
                num_blocks=size // geometry.block_bytes,
                blocks_per_page=plan.blocks_per_page(large_page_bytes),
                kernel_rows_per_block=geometry.kernel_rows_per_block,
                row_bytes=geometry.row_bytes,
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
                    layer_stride=geometry.row_bytes,
                    block_stride=geometry.row_bytes,
                    offset=plan.layer_offsets[rank][layer_name],
                )
            )
    return KVCacheConfig(
        num_pages=num_pages,
        kv_cache_tensors=tensors,
        kv_cache_groups=groups,
        prefix_cache_retention_interval=cache_config.prefix_cache_retention_interval,
        large_page_bytes=large_page_bytes,
        block_size=cache_config.block_size,
    )


def _build_kv_cache_configs(
    vllm_config: VllmConfig,
    groups: list[KVCacheGroupSpec],
    worker_specs: list[dict[str, KVCacheSpec]],
    fit_pool: Callable[[list[_LargePageGroupPlan], int], int],
    available_memory: list[int] | None = None,
) -> list[KVCacheConfig]:
    """Place ``groups`` in the shared pool and build each worker's
    KVCacheConfig.

    ``fit_pool(plans, large_page_bytes)`` returns the pool's page count (page
    0 included). It may auto-fit max_model_len, so it runs before the HiSparse
    host pool, which is sized at the final max_model_len.
    """
    if not groups:
        # Attention-free models do not have KV cache. Return num_pages=1 as
        # BlockPool always needs a null page.
        return [
            KVCacheConfig(
                num_pages=1,
                kv_cache_tensors=[],
                kv_cache_groups=[],
                prefix_cache_retention_interval=(
                    vllm_config.cache_config.prefix_cache_retention_interval
                ),
                block_size=vllm_config.cache_config.block_size,
            )
            for _ in worker_specs
        ]
    plans, large_page_bytes = _plan_large_page_placement(
        groups, worker_specs, vllm_config, available_memory
    )
    num_pages = fit_pool(plans, large_page_bytes)
    kv_cache_configs = [
        _build_large_page_kv_cache_config(
            vllm_config, plans, rank, large_page_bytes, num_pages
        )
        for rank in range(len(worker_specs))
    ]
    if any(group.host_resident for group in groups):
        hisparse_layout = create_hisparse_layout(
            vllm_config, groups, get_hisparse_host_pool_bytes(vllm_config)
        )
        for kv_cache_config in kv_cache_configs:
            add_hisparse_host_config(kv_cache_config, hisparse_layout)
    return kv_cache_configs


def get_kv_cache_config_from_groups(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
    min_blocks: int,
) -> KVCacheConfig:
    """Local placement for worker-side profiling caches.

    The engine computes the real (cross-rank) placement after profiling; here
    one worker plans its own groups alone, with enough pages to give every
    group ``min_blocks`` blocks. Over-estimating is safe: the minimal cache is
    torn down after profiling.
    """
    worker_spec = {
        name: spec
        for group in kv_cache_groups
        for name, spec in (group.layer_kv_cache_specs or {}).items()
    }
    (kv_cache_config,) = _build_kv_cache_configs(
        vllm_config,
        kv_cache_groups,
        [worker_spec],
        lambda plans, large_page_bytes: max(
            large_pages_needed(min_blocks, plan.blocks_per_page(large_page_bytes))
            for plan in plans
        ),
    )
    return kv_cache_config


def get_kv_cache_groups_from_builder(
    builder: KVCacheConfigBuilder,
    vllm_config: VllmConfig,
    kv_cache_spec: dict[str, KVCacheSpec],
) -> list[KVCacheGroupSpec]:
    """Group the layers with the builder and check the result.

    Core checks every grouping, whether it comes from the default grouping or a
    platform builder.
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
    specs = [g.kv_cache_spec for g in groups]
    if not specs:
        return 1
    return reduce(
        math.gcd, (_layer_tp_replicas(s, tp_size, dcp_size) for s in specs), tp_size
    )


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
    2. Generate the KV cache groups from the merged specs. This resolves the
       one global block size B and handles spec unification for hybrid models
       when disable_hybrid_kv_cache_manager is enabled.
    3. Plan the per-group placement in the shared large-page pool, taking the
       smallest page count any worker's available memory affords.
    4. Auto-fit max_model_len if requested, then check that one full
       max-model-len request fits in the pool.
    5. Build each worker's KVCacheConfig from the shared placement.

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
    cache_config = vllm_config.cache_config

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

    def fit_pool(plans: list[_LargePageGroupPlan], large_page_bytes: int) -> int:
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
        return num_pages

    kv_cache_configs = _build_kv_cache_configs(
        vllm_config,
        global_kv_cache_groups,
        kv_cache_specs,
        fit_pool,
        available_memory,
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
    invariants (``num_gpu_blocks_override``, null-page reservation, auto-fit,
    admission checks, cross-rank convergence) stay in the Core-owned
    :func:`get_kv_cache_configs`.
    """

    def get_kv_cache_groups(
        self,
        vllm_config: VllmConfig,
        kv_cache_spec: dict[str, KVCacheSpec],
    ) -> list[KVCacheGroupSpec]:
        """See ``KVCacheConfigBuilder.get_kv_cache_groups``."""
        return get_kv_cache_groups(vllm_config, kv_cache_spec)

    def get_kv_cache_config_from_groups(
        self,
        vllm_config: VllmConfig,
        kv_cache_groups: list[KVCacheGroupSpec],
        num_blocks: int,
    ) -> KVCacheConfig:
        """See ``KVCacheConfigBuilder.get_kv_cache_config_from_groups``."""
        return get_kv_cache_config_from_groups(
            vllm_config, kv_cache_groups, min_blocks=num_blocks
        )
