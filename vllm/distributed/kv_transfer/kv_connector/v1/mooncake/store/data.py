# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Adapted from vllm-project/vllm-ascend
# (vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/).
"""Data classes for MooncakeStoreConnector."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
    KVConnectorWorkerMetadata,
)
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_cache_interface import GroupGeometry

logger = init_logger(__name__)


class BlobBlockHashes(Sequence[BlockHash]):
    """Lazy view over a flat buffer of fixed-size block hashes to avoid the overhead
    of materializing all hashes upfront.
    """

    def __init__(self, blob: memoryview, hash_len: int):
        self._blob = blob
        self._hash_len = hash_len
        self._n = len(blob) // hash_len if hash_len else 0

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(self._n))]
        if idx < 0:
            idx += self._n
        if not 0 <= idx < self._n:
            raise IndexError(idx)
        off = idx * self._hash_len
        return BlockHash(self._blob[off : off + self._hash_len])


@dataclass
class KeyMetadata:
    """Metadata for constructing pool keys."""

    model_name: str
    tp_rank: int
    pcp_rank: int
    dcp_rank: int
    pp_rank: int
    group_id: int = 0
    # Optional namespace prepended to every key. Lets separate deployments
    # share one Mooncake master without colliding on identical block hashes.
    # Empty (the default) keeps keys byte-identical to the unprefixed format.
    cache_prefix: str = ""
    # Complete namespace for an opt-in Store payload format. Empty keeps
    # historical keys byte-identical.
    store_namespace: str = ""


@dataclass(order=True)
class PoolKey:
    """Key for addressing KV cache blocks in the distributed store."""

    key_metadata: KeyMetadata
    chunk_hash: str

    def __hash__(self):
        return hash(
            (
                self.key_metadata.cache_prefix,
                self.key_metadata.model_name,
                self.key_metadata.store_namespace,
                self.key_metadata.tp_rank,
                self.key_metadata.pcp_rank,
                self.key_metadata.dcp_rank,
                self.key_metadata.pp_rank,
                self.key_metadata.group_id,
                self.chunk_hash,
            )
        )

    @staticmethod
    def build_prefix(
        key_metadata: KeyMetadata,
        *,
        tp_rank: int | None = None,
        pcp_rank: int | None = None,
        dcp_rank: int | None = None,
        pp_rank: int | None = None,
    ) -> str:
        """Return the stable prefix for a Mooncake pool key."""
        prefix = f"{key_metadata.cache_prefix}@" if key_metadata.cache_prefix else ""
        return (
            f"{prefix}"
            f"{key_metadata.model_name}"
            f"{key_metadata.store_namespace}"
            f"@tp_rank:{key_metadata.tp_rank if tp_rank is None else tp_rank}"
            f"@pcp{key_metadata.pcp_rank if pcp_rank is None else pcp_rank}"
            f"@dcp{key_metadata.dcp_rank if dcp_rank is None else dcp_rank}"
            f"@pp_rank:{key_metadata.pp_rank if pp_rank is None else pp_rank}"
            f"@group:{key_metadata.group_id}"
        )

    @staticmethod
    def build_key_string(key_prefix: str, chunk_hash: str) -> str:
        return f"{key_prefix}@{chunk_hash}"

    def to_string(self) -> str:
        return self.build_key_string(
            self.build_prefix(self.key_metadata), self.chunk_hash
        )


StoreShardId = int
# Physical ranks encoded in a rank-local key: (TP, PCP, DCP, PP).
RankNamespace = tuple[int, int, int, int]


class StoreLayout:
    """Store payload layout for one KV-cache group on one local TP rank.

    Block ``b`` of the group is ``kernel_rows_per_block`` contiguous kernel
    rows of ``row_bytes`` (all layers of the group packed per row), at byte
    offset ``b * block_bytes`` of the shared backing allocation.
    """

    def __init__(
        self,
        metadata: KeyMetadata,
        block_size: int,
        geometry: GroupGeometry,
    ) -> None:
        self.metadata = metadata
        self.block_size = block_size
        self.geometry = geometry

    @property
    def block_bytes(self) -> int:
        """Bytes one manager block spans in a layer view of the group."""
        return self.geometry.block_bytes

    @property
    def local_shard_ids(self) -> tuple[StoreShardId, ...]:
        """Store objects contributed or consumed by this local rank."""
        raise NotImplementedError

    def key_for(self, shard_id: StoreShardId, chunk_hash: BlockHash) -> str:
        raise NotImplementedError

    def lookup_key_prefixes(
        self,
        rank_namespaces: Sequence[RankNamespace],
    ) -> tuple[str, ...]:
        """Prefixes that must exist for a logical block to be reusable."""
        raise NotImplementedError

    def register_kv_caches(
        self,
        kv_caches: Sequence[torch.Tensor],
    ) -> None:
        """Build descriptors from the group's local KV cache layer views.

        The views share one backing storage at the group's row stride, as
        ``group_pool_storage`` checks.
        """
        raise NotImplementedError

    def prepare_values(
        self,
        chunks: Sequence[tuple[int, int]],
        block_ids: list[int],
        shard_ids: Sequence[StoreShardId],
    ) -> tuple[list[list[int]], list[list[int]], list[int]]:
        """Build Store multi-buffer descriptors for logical chunks."""
        raise NotImplementedError


class RankLocalStoreLayout(StoreLayout):
    """Rank-local Store payload: one segment per manager block.

    The segment is the block's rows of the group's shared backing allocation,
    so one value carries every local layer of the group, packed per row.
    """

    def __init__(
        self,
        metadata: KeyMetadata,
        block_size: int,
        geometry: GroupGeometry,
    ) -> None:
        super().__init__(metadata, block_size, geometry)
        self._key_prefix = PoolKey.build_prefix(metadata)
        self.base_addr = 0

    @property
    def local_shard_ids(self) -> tuple[StoreShardId, ...]:
        return (0,)

    def key_for(self, shard_id: StoreShardId, chunk_hash: BlockHash) -> str:
        assert shard_id == 0
        return PoolKey.build_key_string(self._key_prefix, chunk_hash.hex())

    def lookup_key_prefixes(
        self,
        rank_namespaces: Sequence[RankNamespace],
    ) -> tuple[str, ...]:
        return tuple(
            PoolKey.build_prefix(
                self.metadata,
                tp_rank=tp_rank,
                pcp_rank=pcp_rank,
                dcp_rank=dcp_rank,
                pp_rank=pp_rank,
            )
            for tp_rank, pcp_rank, dcp_rank, pp_rank in rank_namespaces
        )

    def register_kv_caches(
        self,
        kv_caches: Sequence[torch.Tensor],
    ) -> None:
        self.base_addr = kv_caches[0].untyped_storage().data_ptr()

    def prepare_values(
        self,
        chunks: Sequence[tuple[int, int]],
        block_ids: list[int],
        shard_ids: Sequence[StoreShardId],
    ) -> tuple[list[list[int]], list[list[int]], list[int]]:
        if not chunks:
            return [], [], []
        assert len(chunks) == len(shard_ids) and all(
            shard_id == 0 for shard_id in shard_ids
        )
        addrs: list[list[int]] = []
        sizes: list[list[int]] = []
        first_block_ids: list[int] = []
        for start, end in chunks:
            span = end - start
            assert span % self.block_size == 0
            first = start // self.block_size
            chunk_block_ids = block_ids[first : first + span // self.block_size]
            addrs.append(
                [self.base_addr + b * self.block_bytes for b in chunk_block_ids]
            )
            sizes.append([self.block_bytes] * len(chunk_block_ids))
            first_block_ids.append(chunk_block_ids[0])
        return addrs, sizes, first_block_ids

    def prepare_value_for_block(self, block_id: int) -> tuple[list[int], list[int]]:
        """Return addresses and sizes for one physical block slot."""
        return [self.base_addr + block_id * self.block_bytes], [self.block_bytes]


class TPShardedStoreLayout(StoreLayout):
    """Store layout shared by divisible TP sizes.

    A shard's payload for one block is, for each local layer, the shard's head
    slice of every kernel row of the block, rows in token order. Concatenated
    over a block's ``kernel_rows_per_block`` rows this is byte-identical to
    the dense per-layer block slice of the historical format.
    """

    store_format: str

    @classmethod
    def shared_namespace(cls, store_tp_size: int, pp_size: int) -> str:
        return (
            f"@store_tp:{store_tp_size}@store_pp:{pp_size}"
            f"@store_format:{cls.store_format}"
        )

    def __init__(
        self,
        metadata: KeyMetadata,
        block_size: int,
        local_tp_size: int,
        store_tp_size: int,
        tp_rank: int,
        num_kv_heads: int,
        geometry: GroupGeometry,
    ) -> None:
        super().__init__(metadata, block_size, geometry)
        self.shards_per_rank = store_tp_size // local_tp_size
        first_shard = tp_rank * self.shards_per_rank
        self.store_shard_ids = tuple(
            range(first_shard, first_shard + self.shards_per_rank)
        )
        self.store_tp_size = store_tp_size
        self.heads_per_store_shard = num_kv_heads // store_tp_size
        self.local_num_kv_heads = num_kv_heads // local_tp_size
        self._shard_key_prefixes = {
            shard_id: PoolKey.build_prefix(metadata, tp_rank=shard_id)
            for shard_id in self.store_shard_ids
        }
        self._pool_base = 0
        # Per store shard: (byte offset from pool base, size) of each
        # (layer, in-row segment) template.
        self._shard_offsets = np.empty((self.shards_per_rank, 0, 0), dtype=np.int64)
        self._shard_sizes = np.empty((self.shards_per_rank, 0, 0), dtype=np.int64)

    @property
    def local_shard_ids(self) -> tuple[StoreShardId, ...]:
        return self.store_shard_ids

    def key_for(self, shard_id: StoreShardId, chunk_hash: BlockHash) -> str:
        return PoolKey.build_key_string(
            self._shard_key_prefixes[shard_id], chunk_hash.hex()
        )

    def lookup_key_prefixes(
        self,
        rank_namespaces: Sequence[RankNamespace],
    ) -> tuple[str, ...]:
        pp_ranks = sorted({namespace[3] for namespace in rank_namespaces})
        return tuple(
            PoolKey.build_prefix(self.metadata, tp_rank=shard_id, pp_rank=pp_rank)
            for shard_id in range(self.store_tp_size)
            for pp_rank in pp_ranks
        )

    def _has_packed_strides(
        self,
        head_stride: int,
        token_stride: int,
        content_bytes: int,
        kernel_tokens: int,
    ) -> bool:
        raise NotImplementedError

    def _shard_segments(
        self,
        head_start: int,
        head_stride: int,
        token_stride: int,
        content_bytes: int,
        kernel_tokens: int,
    ) -> Iterable[tuple[int, int]]:
        raise NotImplementedError

    def register_kv_caches(
        self,
        kv_caches: Sequence[torch.Tensor],
    ) -> None:
        # Per shard: one (offsets, sizes) pair per layer, each holding the
        # shard's in-row segments for that layer.
        templates: list[list[tuple[list[int], list[int]]]] = [
            [([], []) for _ in kv_caches] for _ in range(self.shards_per_rank)
        ]
        pool_base = kv_caches[0].untyped_storage().data_ptr()
        for layer_idx, cache in enumerate(kv_caches):
            if cache.ndim != 4 or cache.shape[1] != self.local_num_kv_heads:
                raise ValueError(
                    "TP-shared Mooncake store requires per-layer KV cache views "
                    "with logical shape (kernel_rows, local_kv_heads, "
                    "kernel_tokens, content)"
                )
            element_size = cache.element_size()
            head_stride, token_stride, content_stride = (
                stride * element_size for stride in cache.stride()[1:]
            )
            kernel_tokens = cache.shape[2]
            content_bytes = cache.shape[3] * element_size
            if content_stride != element_size or not self._has_packed_strides(
                head_stride, token_stride, content_bytes, kernel_tokens
            ):
                raise ValueError(
                    "TP-shared Mooncake store requires packed "
                    f"{self.store_format} KV layout"
                )

            layer_offset = cache.data_ptr() - pool_base
            for local_shard in range(self.shards_per_rank):
                offsets, sizes = templates[local_shard][layer_idx]
                head_start = local_shard * self.heads_per_store_shard
                for offset, size in self._shard_segments(
                    head_start, head_stride, token_stride, content_bytes, kernel_tokens
                ):
                    offsets.append(layer_offset + offset)
                    sizes.append(size)

        self._pool_base = pool_base
        self._shard_offsets = np.asarray(
            [[layer[0] for layer in shard] for shard in templates], dtype=np.int64
        )
        self._shard_sizes = np.asarray(
            [[layer[1] for layer in shard] for shard in templates], dtype=np.int64
        )

    def prepare_values(
        self,
        chunks: Sequence[tuple[int, int]],
        block_ids: list[int],
        shard_ids: Sequence[StoreShardId],
    ) -> tuple[list[list[int]], list[list[int]], list[int]]:
        if not chunks:
            return [], [], []
        if len(chunks) != len(shard_ids):
            raise ValueError("Each Store chunk must have one shard ID")

        count = len(chunks)
        starts = np.fromiter(
            (chunk[0] for chunk in chunks), dtype=np.int64, count=count
        )
        ends = np.fromiter((chunk[1] for chunk in chunks), dtype=np.int64, count=count)
        if np.any(ends - starts != self.block_size):
            raise ValueError("TP-shared Mooncake store requires full KV blocks")

        first_shard = self.store_shard_ids[0]
        local_shards = np.fromiter(
            (shard_id - first_shard for shard_id in shard_ids),
            dtype=np.int64,
            count=count,
        )
        if np.any(local_shards < 0) or np.any(local_shards >= self.shards_per_rank):
            raise ValueError("Store shard is not owned by this TP rank")
        chunk_block_ids = [
            block_ids[index] for index in (starts // self.block_size).tolist()
        ]
        # Segment order per (chunk, shard): layer-major, then the block's
        # kernel rows in token order, then the in-row segments — identical
        # byte order to the historical dense per-layer block slices.
        rows = self.geometry.kernel_rows_per_block
        offsets = self._shard_offsets[local_shards]  # [chunk, layer, segment]
        block_rows = self.geometry.rows(chunk_block_ids)
        row_offsets = (
            block_rows.reshape(count, rows)[:, None, :, None] * self.geometry.row_bytes
        )
        addrs = self._pool_base + offsets[:, :, None, :] + row_offsets
        addrs = addrs.reshape(count, -1)
        sizes = self._shard_sizes[local_shards]  # [chunk, layer, segment]
        sizes = np.broadcast_to(
            sizes[:, :, None, :], (count, sizes.shape[1], rows, sizes.shape[2])
        ).reshape(count, -1)
        return addrs.tolist(), sizes.tolist(), chunk_block_ids


class LBHNCStoreLayout(TPShardedStoreLayout):
    """Native head-major layout shared by divisible TP sizes."""

    store_format = "tp_shared_lbhnc"

    def _has_packed_strides(
        self,
        head_stride: int,
        token_stride: int,
        content_bytes: int,
        kernel_tokens: int,
    ) -> bool:
        return (
            token_stride == content_bytes
            and head_stride == kernel_tokens * content_bytes
        )

    def _shard_segments(
        self,
        head_start: int,
        head_stride: int,
        token_stride: int,
        content_bytes: int,
        kernel_tokens: int,
    ) -> Iterable[tuple[int, int]]:
        yield head_start * head_stride, self.heads_per_store_shard * head_stride


class LBNHCStoreLayout(TPShardedStoreLayout):
    """Native token-major layout shared by divisible TP sizes."""

    store_format = "tp_shared_lbnhc"

    def _has_packed_strides(
        self,
        head_stride: int,
        token_stride: int,
        content_bytes: int,
        kernel_tokens: int,
    ) -> bool:
        return (
            head_stride == content_bytes
            and token_stride == self.local_num_kv_heads * content_bytes
        )

    def _shard_segments(
        self,
        head_start: int,
        head_stride: int,
        token_stride: int,
        content_bytes: int,
        kernel_tokens: int,
    ) -> Iterable[tuple[int, int]]:
        for token_idx in range(kernel_tokens):
            yield (
                token_idx * token_stride + head_start * head_stride,
                self.heads_per_store_shard * content_bytes,
            )


class ChunkedTokenDatabase:
    """Enumerates logical token chunks and their hashes."""

    def __init__(
        self,
        metadata: KeyMetadata,
        block_size: int,
        store_layout: StoreLayout,
    ):
        self.metadata = metadata
        self.block_size = block_size
        self.store_layout = store_layout

    def _rank_local_layout(self) -> RankLocalStoreLayout:
        if not isinstance(self.store_layout, RankLocalStoreLayout):
            raise RuntimeError("This operation requires a rank-local Store layout")
        return self.store_layout

    def key_for(self, chunk_hash: BlockHash) -> str:
        return self._rank_local_layout().key_for(0, chunk_hash)

    def prepare_value_for_block(self, block_id: int) -> tuple[list[int], list[int]]:
        """Return addresses and sizes for one physical block slot."""
        return self._rank_local_layout().prepare_value_for_block(block_id)

    def process_tokens(
        self,
        token_len: int,
        block_hashes: list[BlockHash],
        mask_num: int = 0,
        *,
        chunk_mask: list[bool] | None = None,
        put_step: int = 1,
        put_step_rank: int = 0,
    ) -> Iterable[tuple[int, int, BlockHash]]:
        """Process tokens and yield (start_idx, end_idx, block_hash) tuples.

        When there are fewer KV heads than TP ranks, chunks are distributed
        across TP ranks to avoid duplicate load/store. The assignment keys off
        the absolute ``chunk_id`` so a given chunk always lands on the same
        rank regardless of where the processed suffix begins.

        Args:
            token_len: Total number of tokens. Must be block aligned and
                covered by ``block_hashes`` when hashes are present.
            block_hashes: One block hash per ``block_size`` chunk.
            mask_num: Number of tokens to skip from the beginning.
            chunk_mask: Optional mask relative to the first chunk after
                ``mask_num``. False entries are skipped before hash access.
            put_step: Stride for distributing chunks across ranks.
            put_step_rank: ``chunk_id % put_step`` value this rank stores.

        """
        assert put_step > 0
        if not block_hashes:
            return
        assert token_len % self.block_size == 0
        max_chunks = token_len // self.block_size
        assert max_chunks <= len(block_hashes)
        start_chunk = max(0, cdiv(mask_num, self.block_size))
        if chunk_mask is not None:
            max_chunks = min(max_chunks, start_chunk + len(chunk_mask))
        for chunk_id in range(start_chunk, max_chunks):
            if chunk_mask is not None and not chunk_mask[chunk_id - start_chunk]:
                continue
            if chunk_id % put_step != put_step_rank:
                continue
            start_idx = chunk_id * self.block_size
            yield start_idx, start_idx + self.block_size, block_hashes[chunk_id]


@dataclass
class LoadSpec:
    """Specification for loading KV cache from external store."""

    vllm_cached_tokens: int
    kvpool_cached_tokens: int
    can_load: bool
    token_len: int = 0


@dataclass
class RequestTracker:
    """Tracks per-request state across scheduler ticks."""

    req_id: str
    token_len: int
    allocated_block_ids: tuple[list[int], ...]
    num_saved_tokens: int = 0
    token_ids: list[int] | None = None
    has_pending_offload: bool = False
    # Snapshot of the prefill range length at tracker creation time.
    # For a fresh request this is len(prompt). For a resumed-from-preemption
    # request it includes previously-generated tokens, which are re-prefilled.
    prefill_end_tokens: int = 0

    def reset(self) -> None:
        self.token_len = 0
        self.allocated_block_ids = ()
        self.num_saved_tokens = 0
        self.token_ids = None
        self.has_pending_offload = False
        self.prefill_end_tokens = 0

    def update(
        self,
        new_block_ids: tuple[list[int], ...] | list[int],
    ) -> None:
        # Backward-compat: accept a single list (broadcast to single group).
        if isinstance(new_block_ids, list):
            new_block_ids = (new_block_ids,)
        if len(new_block_ids) != len(self.allocated_block_ids):
            raise ValueError(
                f"Group count mismatch: tracker has "
                f"{len(self.allocated_block_ids)} groups, update has "
                f"{len(new_block_ids)}"
            )
        for existing, new in zip(self.allocated_block_ids, new_block_ids, strict=True):
            if new:
                existing.extend(new)


class BoundaryPut(NamedTuple):
    """A block stored under the hash of the prefix ending at ``num_tokens``."""

    group_id: int
    block_id: int
    num_tokens: int


@dataclass
class ReqMeta:
    """Per-request metadata for store put/get operations."""

    req_id: str
    token_len_chunk: int
    block_ids: tuple[list[int], ...]
    block_hashes: list[BlockHash]

    can_save: bool | None = None
    load_spec: LoadSpec | None = None
    current_event: torch.cuda.Event | None = None

    token_ids: list[int] | None = None
    # Absolute request offset represented by token_ids[0].
    token_ids_start: int = 0
    num_prompt_tokens: int | None = None
    # Total prefill tokens, including generated tokens replayed after preemption.
    prefill_end_tokens: int | None = None
    # Identifies this store job for the engine's lifetime. A request id cannot
    # serve that purpose: it is reused once a preempted request resumes, so it
    # would release the wrong job's blocks.
    store_job_id: int | None = None
    # Core-provided mamba "align" state snapshots, stored under the hash of the
    # prefix they end rather than by position. The store-job reference keeps
    # each block alive until every worker rank finishes the job.
    boundary_puts: list[BoundaryPut] | None = None

    @staticmethod
    def from_request_tracker(
        tracker: RequestTracker,
        block_size: int,
        load_spec: LoadSpec | None = None,
        skip_save: bool | None = False,
        block_hashes: list[BlockHash] | None = None,
        num_prompt_tokens: int | None = None,
    ) -> "ReqMeta | None":
        """Create ReqMeta from a RequestTracker."""
        if block_hashes is None:
            block_hashes = []
        input_token_len = tracker.token_len

        token_ids_start = tracker.num_saved_tokens
        chunk_boundary = cdiv(token_ids_start + 1, block_size) * block_size
        num_tokens_to_save = input_token_len // block_size * block_size

        skip_save = skip_save or num_tokens_to_save < chunk_boundary
        # A ReqMeta must never carry both a save AND a load.
        # The save would also be wasted work — the bytes are being looked up
        # in the store right now. Later cached_reqs steps save new tokens
        # normally.
        if load_spec is not None and load_spec.can_load:
            skip_save = True
        if skip_save and load_spec is None:
            return None

        if not skip_save:
            tracker.num_saved_tokens = num_tokens_to_save

        token_ids = None
        if tracker.token_ids and not skip_save:
            # Scheduler tracking continues while this job is handled by an
            # asynchronous worker, so metadata must own a stable snapshot.
            token_ids = tracker.token_ids[token_ids_start:num_tokens_to_save]

        if load_spec is not None and load_spec.can_load:
            logger.debug(
                "Scheduled to load %d tokens for request %s",
                load_spec.kvpool_cached_tokens,
                tracker.req_id,
            )
        else:
            load_spec = None

        logger.debug(
            "request:%s, meta save spec:%s, meta load spec:%s",
            tracker.req_id,
            not skip_save,
            load_spec,
        )
        return ReqMeta(
            req_id=tracker.req_id,
            token_len_chunk=num_tokens_to_save,
            block_ids=tracker.allocated_block_ids,
            can_save=not skip_save,
            load_spec=load_spec,
            block_hashes=block_hashes,
            token_ids=token_ids,
            token_ids_start=token_ids_start,
            num_prompt_tokens=num_prompt_tokens,
            prefill_end_tokens=tracker.prefill_end_tokens,
        )


@dataclass
class MooncakeStoreWorkerMetadata(KVConnectorWorkerMetadata):
    """Maps ``ReqMeta.store_job_id`` to the number of ranks done with that job."""

    completed_saves: dict[int, int] = field(default_factory=dict)

    def aggregate(
        self, other: "KVConnectorWorkerMetadata"
    ) -> "MooncakeStoreWorkerMetadata":
        assert isinstance(other, MooncakeStoreWorkerMetadata)
        for store_job_id, count in other.completed_saves.items():
            self.completed_saves[store_job_id] = (
                self.completed_saves.get(store_job_id, 0) + count
            )
        return self


class MooncakeStoreConnectorMetadata(KVConnectorMetadata):
    """Metadata passed from scheduler to worker."""

    def __init__(
        self,
        unfinished_request_ids: set[str],
        preempted_req_ids: set[str],
    ):
        self.requests: list[ReqMeta] = []
        self.unfinished_request_ids = unfinished_request_ids
        self.preempted_req_ids = preempted_req_ids

    def add_request(self, req_meta: ReqMeta) -> None:
        self.requests.append(req_meta)
