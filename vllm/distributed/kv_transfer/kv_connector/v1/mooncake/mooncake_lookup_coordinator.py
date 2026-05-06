# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cache-hit aggregation for ``MooncakeStoreConnector.lookup`` that
delegates to vllm's per-manager ``find_longest_cache_hit`` by inheriting
from :class:`HybridKVCacheCoordinator` and overriding only ``__init__``.
Reuses upstream's iterative fixed-point loop, EAGLE handling, and
alignment math instead of reimplementing them.
"""

from dataclasses import dataclass
from math import lcm
from typing import Any

from vllm.v1.core.kv_cache_coordinator import HybridKVCacheCoordinator
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    make_block_hash_with_group_id,
)
from vllm.v1.core.single_type_kv_cache_manager import (
    SingleTypeKVCacheManager,
    spec_manager_map,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
)


@dataclass(slots=True)
class _LookupBlock:
    """Stand-in ``KVCacheBlock``: the per-manager ``find_longest_cache_hit``
    methods only count blocks and check ``is_null``."""

    block_id: int = -1
    is_null: bool = False


class LookupBlockPool:
    """Read-only ``BlockPool``-shaped view over Mooncake's exists results.

    Built once per ``lookup`` call from a ``batch_is_exist`` round-trip
    and assigned to ``self._lookup_coordinator.block_pool`` so the
    inherited ``find_longest_cache_hit`` probes Mooncake instead of a
    real GPU-backed pool.

    Hash shape: managers probe with ``make_block_hash_with_group_id(concat_hash,
    group_id)``, where ``concat_hash`` for HMA groups with ``block_size >
    hash_block_size`` spans ``scale = block_size / hash_block_size``
    consecutive entries (matching ``BlockHashListWithBlockSize``).
    ``MooncakeLookupCoordinator.build_block_pool`` pre-computes those
    concat hashes so probes here are an O(1) dict lookup.
    """

    def __init__(self, cached: dict[bytes, bool]):
        self._cached: dict[bytes, bool] = cached
        self.null_block = _LookupBlock(is_null=True)

    def get_cached_block(
        self,
        block_hash: BlockHash,
        kv_cache_group_ids: list[int],
    ) -> list[Any] | None:
        cached_blocks: list[Any] = []
        for group_id in kv_cache_group_ids:
            key = make_block_hash_with_group_id(block_hash, group_id)
            if not self._cached.get(bytes(key), False):
                return None
            cached_blocks.append(_LookupBlock())
        return cached_blocks


def _unwrap_kv_cache_config(config: KVCacheConfig) -> KVCacheConfig:
    """Replace ``UniformTypeKVCacheSpecs`` wrappers with their inner spec.

    Worker-side configs wrap groups in ``UniformTypeKVCacheSpecs``,
    which ``spec_manager_map`` can't dispatch on. ``is_eagle_group``
    is preserved across the unwrap — without it, EAGLE configs
    silently over-report hits by one block.
    """
    new_groups = []
    for g in config.kv_cache_groups:
        spec = g.kv_cache_spec
        inner_specs = getattr(spec, "kv_cache_specs", None)
        if inner_specs:
            inner = next(iter(inner_specs.values()))
            new_groups.append(
                KVCacheGroupSpec(g.layer_names, inner, is_eagle_group=g.is_eagle_group)
            )
        else:
            new_groups.append(g)
    return KVCacheConfig(
        num_blocks=config.num_blocks,
        kv_cache_tensors=config.kv_cache_tensors,
        kv_cache_groups=new_groups,
    )


class UnknownManagerSpecError(ValueError):
    """A kv_cache_group has a spec type missing from
    ``spec_manager_map``. Caller disables external hits to avoid
    over-reporting by skipping the unknown group."""


class HashBlockSizeMisalignedError(ValueError):
    """A kv_cache_group's ``block_size`` is not divisible by
    ``hash_block_size``."""


class MooncakeLookupCoordinator(HybridKVCacheCoordinator):
    """Inherits ``HybridKVCacheCoordinator.find_longest_cache_hit`` and
    skips the parent ``__init__`` so we don't pay for a GPU-backed
    ``BlockPool`` or per-group managers we don't need.

    Attrs ``find_longest_cache_hit`` reads, all populated below:
    ``kv_cache_config``, ``attention_groups``, ``lcm_block_size``,
    ``hash_block_size``, ``eagle_attn_group_indices``, ``block_pool``
    (overwritten per-call by the worker).
    """

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        hash_block_size: int,
        use_eagle: bool = False,
    ):
        # Skip both parent __init__s — they'd build a real GPU BlockPool
        # and per-group managers we don't need for read-only hit detection.
        self.kv_cache_config = _unwrap_kv_cache_config(kv_cache_config)
        self.hash_block_size = hash_block_size

        # Re-assert the parent's invariant since we skipped its __init__:
        # build_block_pool's scale = block_size // hash_block_size would
        # silently round otherwise.
        for i, g in enumerate(self.kv_cache_config.kv_cache_groups):
            bs = g.kv_cache_spec.block_size
            if bs % hash_block_size != 0:
                raise HashBlockSizeMisalignedError(
                    f"Group {i} block_size={bs} is not divisible by "
                    f"hash_block_size={hash_block_size}."
                )

        # Mirror KVCacheCoordinator.__init__'s eagle-group derivation: prefer
        # explicit per-group is_eagle_group flags; fall back to flagging all
        # groups when use_eagle=True but nothing is annotated, otherwise an
        # EAGLE config silently over-reports hits by one block.
        self.eagle_group_ids: set[int] = {
            i
            for i, g in enumerate(self.kv_cache_config.kv_cache_groups)
            if g.is_eagle_group
        }
        if use_eagle and not self.eagle_group_ids:
            self.eagle_group_ids = set(range(len(self.kv_cache_config.kv_cache_groups)))
        # Filled by _build_attention_groups once attention_groups exists.
        self.eagle_attn_group_indices: set[int] = set()
        # Worker swaps in a fresh LookupBlockPool per lookup call.
        self.block_pool: LookupBlockPool | None = None  # type: ignore[assignment]
        self._build_attention_groups()

    def _build_attention_groups(self) -> None:
        """Group kv_cache_groups by spec type for batch hit-lookup.

        Mirrors ``HybridKVCacheCoordinator.verify_and_split_kv_cache_groups``
        with two adjustments:

        * Allow single-group configs (no ``len > 1`` assert) so FA-only
          models work uniformly.
        * Raise :class:`UnknownManagerSpecError` on unmapped spec types
          so the worker can fail-closed; silently skipping would
          over-report hits by ignoring the unknown group.
        """
        groups: list[tuple[KVCacheSpec, list[int], type[SingleTypeKVCacheManager]]] = []
        for i, g in enumerate(self.kv_cache_config.kv_cache_groups):
            spec = g.kv_cache_spec
            manager_cls = spec_manager_map.get(type(spec))
            if manager_cls is None:
                raise UnknownManagerSpecError(
                    f"Unknown kv_cache_spec type {type(spec).__name__!r} "
                    f"at group {i}; coordinator path can't safely "
                    f"aggregate. Worker will disable external cache "
                    f"hits for this config."
                )
            for existing_spec, group_ids, existing_cls in groups:
                if existing_spec == spec:
                    assert manager_cls is existing_cls, (
                        "Same spec mapped to different manager classes"
                    )
                    group_ids.append(i)
                    break
            else:
                groups.append((spec, [i], manager_cls))

        # FA first (tightest initial bound for the iterative loop).
        groups.sort(key=lambda x: not isinstance(x[0], FullAttentionSpec))
        self.attention_groups = groups
        block_sizes = [s.block_size for s, _, _ in groups]
        self.lcm_block_size: int = lcm(*block_sizes) if block_sizes else 1
        # Mirror ``HybridKVCacheCoordinator.verify_and_split_kv_cache_groups``:
        # mark attention_groups containing any eagle group_id.
        self.eagle_attn_group_indices = {
            i
            for i, (_, group_ids, _) in enumerate(self.attention_groups)
            if any(gid in self.eagle_group_ids for gid in group_ids)
        }

    def build_block_pool(
        self,
        chunk_group_to_exists: dict[tuple[int, int], int],
        block_hashes: list[BlockHash],
    ) -> LookupBlockPool:
        """Build the per-call pool from Mooncake's exists bits.

        Each (chunk, group_id) with ``exists == 1`` is inserted under the
        same key shape upstream's ``BlockHashListWithBlockSize`` produces:
        the concat hash spans ``scale = block_size // hash_block_size``
        consecutive ``block_hashes`` entries (or one when ``scale == 1``).
        Misses (0 / -1) are simply absent from the dict.
        """
        cached: dict[bytes, bool] = {}
        groups = self.kv_cache_config.kv_cache_groups
        for (chunk_id, group_id), exists in chunk_group_to_exists.items():
            if exists != 1:
                continue
            spec = groups[group_id].kv_cache_spec
            scale = spec.block_size // self.hash_block_size
            base = chunk_id * scale
            end = base + scale
            if end > len(block_hashes):
                continue
            if scale == 1:
                concat_hash: BlockHash = block_hashes[base]
            else:
                concat_hash = BlockHash(b"".join(block_hashes[base:end]))
            key = make_block_hash_with_group_id(concat_hash, group_id)
            cached[bytes(key)] = True
        return LookupBlockPool(cached)
