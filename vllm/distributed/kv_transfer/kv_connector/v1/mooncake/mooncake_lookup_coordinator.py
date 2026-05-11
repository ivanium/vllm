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
from typing import Any, cast

from vllm.utils.math_utils import cdiv
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


@dataclass(frozen=True, slots=True)
class MooncakeChunk:
    """Logical group chunk that should be addressed in Mooncake.

    ``block_hash`` is the right-edge hash used by Mooncake's persisted key.
    The lookup coordinator may use a wider concat hash internally when calling
    vLLM's cache managers, but the transfer path stores/loads by this hash.
    """

    group_id: int
    chunk_id: int
    start: int
    end: int
    hash_index: int
    block_hash: BlockHash


@dataclass(slots=True)
class _LookupBlock:
    """Stand-in ``KVCacheBlock``: the per-manager ``find_longest_cache_hit``
    methods only count blocks and check ``is_null``.

    ``chunk`` is present for real Mooncake-backed blocks and absent for
    manager-produced null/skipped blocks.
    """

    block_id: int = -1
    is_null: bool = False
    chunk: MooncakeChunk | None = None


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

    def __init__(self, cached: dict[bytes, _LookupBlock]):
        self._cached: dict[bytes, _LookupBlock] = cached
        self.null_block = _LookupBlock(is_null=True)

    def get_cached_block(
        self,
        block_hash: BlockHash,
        kv_cache_group_ids: list[int],
    ) -> list[Any] | None:
        cached_blocks: list[Any] = []
        for group_id in kv_cache_group_ids:
            key = make_block_hash_with_group_id(block_hash, group_id)
            cached_block = self._cached.get(bytes(key))
            if cached_block is None:
                return None
            cached_blocks.append(cached_block)
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

    def normalize_block_hashes(
        self,
        block_hashes: list[BlockHash] | list[bytes] | list[str],
    ) -> list[BlockHash]:
        """Return byte-backed block hashes for manager and planner code."""
        if not block_hashes:
            return []
        if isinstance(block_hashes[0], str):
            return [BlockHash(bytes.fromhex(h)) for h in cast(list[str], block_hashes)]
        return [BlockHash(bytes(h)) for h in cast(list[bytes], block_hashes)]

    def _chunk_for_group(
        self,
        group_id: int,
        chunk_id: int,
        block_hashes: list[BlockHash],
        token_len: int,
    ) -> MooncakeChunk | None:
        spec = self.kv_cache_config.kv_cache_groups[group_id].kv_cache_spec
        start = chunk_id * spec.block_size
        if start >= token_len:
            return None
        end = min(start + spec.block_size, token_len)
        if end % self.hash_block_size != 0:
            return None
        hash_index = end // self.hash_block_size - 1
        if not (0 <= hash_index < len(block_hashes)):
            return None
        return MooncakeChunk(
            group_id=group_id,
            chunk_id=chunk_id,
            start=start,
            end=end,
            hash_index=hash_index,
            block_hash=block_hashes[hash_index],
        )

    def iter_group_chunks(
        self,
        token_len: int,
        block_hashes: list[BlockHash] | list[bytes] | list[str],
        mask_num: int = 0,
    ) -> list[MooncakeChunk]:
        """Return all hash-addressable chunks for every KV cache group."""
        hashes = self.normalize_block_hashes(block_hashes)
        chunks: list[MooncakeChunk] = []
        for group_id, group in enumerate(self.kv_cache_config.kv_cache_groups):
            total_chunks = cdiv(token_len, group.kv_cache_spec.block_size)
            for chunk_id in range(total_chunks):
                chunk = self._chunk_for_group(group_id, chunk_id, hashes, token_len)
                if chunk is not None and chunk.start >= mask_num:
                    chunks.append(chunk)
        return chunks

    def _resident_chunk_ids(
        self,
        token_len: int,
        block_ids: list[list[int]] | list[int],
    ) -> set[tuple[int, int]]:
        if block_ids and isinstance(block_ids[0], list):
            per_group = cast(list[list[int]], block_ids)
        else:
            per_group = [cast(list[int], block_ids)]

        resident: set[tuple[int, int]] = set()
        for group_id, group_block_ids in enumerate(per_group):
            if group_id >= len(self.kv_cache_config.kv_cache_groups):
                continue
            spec = self.kv_cache_config.kv_cache_groups[group_id].kv_cache_spec
            total_chunks = cdiv(token_len, spec.block_size)
            offset = total_chunks - len(group_block_ids)
            for local_i in range(len(group_block_ids)):
                chunk_id = offset + local_i
                if 0 <= chunk_id < total_chunks:
                    resident.add((chunk_id, group_id))
        return resident

    def plan_store_chunks(
        self,
        token_len: int,
        block_hashes: list[BlockHash] | list[bytes] | list[str],
        block_ids: list[list[int]] | list[int],
    ) -> list[MooncakeChunk]:
        """Return chunks that are physically resident and worth storing.

        The scheduler/coordinator path already normalizes block tables by KV
        spec. The worker consumes this generic plan instead of hard-coding
        sliding-window clipping.
        """
        resident = self._resident_chunk_ids(token_len, block_ids)
        return [
            chunk
            for chunk in self.iter_group_chunks(token_len, block_hashes)
            if (chunk.chunk_id, chunk.group_id) in resident
        ]

    def plan_load_chunks(
        self,
        token_len: int,
        block_hashes: list[BlockHash] | list[bytes] | list[str],
        block_ids: list[list[int]] | list[int],
        mask_num: int = 0,
    ) -> list[MooncakeChunk]:
        """Return resident chunks to load for an externally cached prefix."""
        resident = self._resident_chunk_ids(token_len, block_ids)
        return [
            chunk
            for chunk in self.iter_group_chunks(token_len, block_hashes, mask_num)
            if (chunk.chunk_id, chunk.group_id) in resident
        ]

    def plan_load_chunks_from_hit_blocks(
        self,
        hit_blocks: tuple[list[Any], ...],
        mask_num: int = 0,
    ) -> list[MooncakeChunk]:
        """Extract real Mooncake chunks from manager-produced hit blocks."""
        chunks: list[MooncakeChunk] = []
        seen: set[tuple[int, int]] = set()
        for blocks in hit_blocks:
            for block in blocks:
                if getattr(block, "is_null", False):
                    continue
                chunk = getattr(block, "chunk", None)
                if chunk is None or chunk.start < mask_num:
                    continue
                key = (chunk.group_id, chunk.chunk_id)
                if key in seen:
                    continue
                seen.add(key)
                chunks.append(chunk)
        return sorted(chunks, key=lambda c: (c.group_id, c.chunk_id))

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
        cached: dict[bytes, _LookupBlock] = {}
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
            chunk = self._chunk_for_group(
                group_id,
                chunk_id,
                block_hashes,
                token_len=len(block_hashes) * self.hash_block_size,
            )
            cached[bytes(key)] = _LookupBlock(chunk=chunk)
        return LookupBlockPool(cached)
