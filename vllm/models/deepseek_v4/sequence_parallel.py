# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Sequence-sharded token split helpers for DeepSeek-V4 experiments.

The Ascend DSA-CP path splits query tokens across ranks, gathers hidden
states for full-cache producers, then restores the TP head layout with an
all-to-all. CUDA DeepSeek-V4 can use the same shape if we first define a
rank-local token packing that still maps back to normal per-request varlen
metadata.

The helpers in this module are intentionally framework-neutral: they compute
token ownership, local per-request metadata, and the Ulysses-style tensor
transposes needed by the correctness prototype.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TokenRange:
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class ShardQTokenSplit:
    """Rank-local view of a shardq equal split of flattened tokens.

    ``local_to_global`` stores token ids in the order the local rank should pack
    query rows. ``local_query_start_loc`` and ``local_seq_lens`` describe the
    same packed rows as pseudo-requests: if one original request intersects two
    shardq ranges, it contributes two local segments. ``segment_req_ids`` maps
    those pseudo-requests back to original request ids.
    """

    ranges: tuple[TokenRange, ...]
    local_to_global: torch.Tensor
    local_query_start_loc: torch.Tensor
    local_seq_lens: torch.Tensor
    segment_req_ids: torch.Tensor
    segment_offsets: torch.Tensor

    # Host-resident copies of the forward-constant segment metadata, computed
    # during the split so per-forward builders never round-trip GPU->host to
    # rebuild them. `seq_lens_cpu` is the full per-request KV length (indexed by
    # original request id); the others are per local segment.
    segment_req_ids_cpu: tuple[int, ...] = ()
    local_query_start_loc_cpu: tuple[int, ...] = (0,)
    local_seq_lens_cpu: tuple[int, ...] = ()
    seq_lens_cpu: tuple[int, ...] = ()


@dataclass(frozen=True)
class AllRankShardQTokenSplit:
    ranges: tuple[tuple[TokenRange, ...], ...]
    local_to_global: tuple[torch.Tensor, ...]
    counts: list[int]


def _merge_adjacent_ranges(ranges: list[TokenRange]) -> tuple[TokenRange, ...]:
    merged: list[TokenRange] = []
    for token_range in sorted(ranges, key=lambda r: r.start):
        if token_range.length <= 0:
            continue
        if merged and token_range.start <= merged[-1].end:
            merged[-1] = TokenRange(
                merged[-1].start, max(merged[-1].end, token_range.end)
            )
        else:
            merged.append(token_range)
    return tuple(merged)


def build_shardq_ranges(
    num_tokens: int, world_size: int, rank: int
) -> tuple[TokenRange, ...]:
    """Return the two shardq contiguous chunks assigned to ``rank``.

    The split first assigns each rank an equal total token budget, then folds
    that budget across the front and back of the flattened token stream. Empty
    ranges are omitted, so small batches keep compact metadata.
    """
    if world_size < 1:
        raise ValueError(f"world_size must be >= 1, got {world_size}")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}")
    if num_tokens < 0:
        raise ValueError(f"num_tokens must be >= 0, got {num_tokens}")

    if num_tokens == 0:
        return ()
    if world_size == 1:
        return (TokenRange(0, num_tokens),)

    base, remainder = divmod(num_tokens, world_size)
    counts = [base + int(r < remainder) for r in range(world_size)]
    front_lens = [(count + 1) // 2 for count in counts]
    back_lens = [count - front_len for count, front_len in zip(counts, front_lens)]

    front_start = sum(front_lens[:rank])
    first = TokenRange(front_start, front_start + front_lens[rank])

    back_end = num_tokens - sum(back_lens[:rank])
    second = TokenRange(back_end - back_lens[rank], back_end)
    return _merge_adjacent_ranges([first, second])


def build_shardq_local_to_global(
    num_tokens: int,
    world_size: int,
    rank: int,
    *,
    device: torch.device | None = None,
) -> tuple[tuple[TokenRange, ...], torch.Tensor]:
    ranges = build_shardq_ranges(num_tokens, world_size, rank)
    if not ranges:
        return ranges, torch.empty(0, dtype=torch.long, device=device)

    parts = [
        torch.arange(r.start, r.end, dtype=torch.long, device=device) for r in ranges
    ]
    return ranges, torch.cat(parts)


def build_all_rank_shardq_token_splits(
    num_tokens: int,
    world_size: int,
    *,
    device: torch.device | None = None,
) -> AllRankShardQTokenSplit:
    ranges: list[tuple[TokenRange, ...]] = []
    local_to_global: list[torch.Tensor] = []
    counts: list[int] = []
    for rank in range(world_size):
        rank_ranges, rank_indices = build_shardq_local_to_global(
            num_tokens, world_size, rank, device=device
        )
        ranges.append(rank_ranges)
        local_to_global.append(rank_indices)
        counts.append(rank_indices.numel())
    return AllRankShardQTokenSplit(
        ranges=tuple(ranges),
        local_to_global=tuple(local_to_global),
        counts=counts,
    )


def count_shardq_tokens_before(
    ranges: tuple[TokenRange, ...],
    boundary: int,
) -> int:
    """Count local shardq tokens whose global token id is less than boundary."""
    total = 0
    for token_range in ranges:
        total += max(0, min(token_range.end, boundary) - token_range.start)
    return total


def build_shardq_token_split(
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    num_input_tokens: int,
    world_size: int,
    rank: int,
) -> ShardQTokenSplit:
    """Build pseudo-request metadata for a shardq query-token split.

    Args:
        query_start_loc: Cumulative global query offsets, shape ``[R + 1]``.
        seq_lens: Full KV sequence lengths per original request, shape ``[R]``.
        num_input_tokens: Number of real flattened query tokens.
        world_size: Number of ranks splitting query tokens.
        rank: Current rank.

    Returns:
        ``ShardQTokenSplit`` with packed local token ids and varlen metadata.
    """
    if query_start_loc.dim() != 1:
        raise ValueError("query_start_loc must be 1D")
    if seq_lens.dim() != 1:
        raise ValueError("seq_lens must be 1D")
    if query_start_loc.numel() != seq_lens.numel() + 1:
        raise ValueError(
            "query_start_loc must have exactly one more element than seq_lens"
        )

    query_starts = query_start_loc.detach().cpu().tolist()
    seq_lens_cpu = seq_lens.detach().cpu().tolist()
    if query_starts[0] != 0:
        raise ValueError("query_start_loc must start at 0")
    if any(start > end for start, end in zip(query_starts[:-1], query_starts[1:])):
        raise ValueError("query_start_loc must be nondecreasing")
    if query_starts[-1] != num_input_tokens:
        raise ValueError(
            "num_input_tokens must match the final query_start_loc value, "
            f"got {num_input_tokens} and {query_starts[-1]}"
        )
    for req_id, (req_start, req_end) in enumerate(
        zip(query_starts[:-1], query_starts[1:])
    ):
        if seq_lens_cpu[req_id] < req_end - req_start:
            raise ValueError(
                "seq_lens must be at least the query length for every request"
            )

    device = query_start_loc.device
    ranges, local_to_global = build_shardq_local_to_global(
        num_input_tokens, world_size, rank, device=device
    )

    segment_lengths: list[int] = []
    segment_seq_lens: list[int] = []
    segment_req_ids: list[int] = []
    segment_offsets: list[int] = []
    for token_range in ranges:
        for req_id, (req_start, req_end) in enumerate(
            zip(query_starts[:-1], query_starts[1:])
        ):
            local_start = max(req_start, token_range.start)
            local_end = min(req_end, token_range.end)
            if local_start >= local_end:
                continue

            local_len = local_end - local_start
            # ``offset`` is the number of query tokens from this request that
            # live on later shardq chunks. Removing it gives the KV length
            # visible to the last local query token in this segment, matching
            # Ascend's contiguous-split metadata rule.
            offset = req_end - local_end
            segment_lengths.append(local_len)
            segment_seq_lens.append(max(seq_lens_cpu[req_id] - offset, 0))
            segment_req_ids.append(req_id)
            segment_offsets.append(local_start - req_start)

    if segment_lengths:
        lengths = torch.tensor(segment_lengths, dtype=torch.int32, device=device)
        local_query_start_loc = torch.empty(
            len(segment_lengths) + 1, dtype=torch.int32, device=device
        )
        local_query_start_loc[0] = 0
        local_query_start_loc[1:] = torch.cumsum(lengths, dim=0)
        local_seq_lens = torch.tensor(
            segment_seq_lens, dtype=torch.int32, device=device
        )
        req_ids = torch.tensor(segment_req_ids, dtype=torch.int32, device=device)
        offsets = torch.tensor(segment_offsets, dtype=torch.int32, device=device)
    else:
        local_query_start_loc = torch.zeros(1, dtype=torch.int32, device=device)
        local_seq_lens = torch.empty(0, dtype=torch.int32, device=device)
        req_ids = torch.empty(0, dtype=torch.int32, device=device)
        offsets = torch.empty(0, dtype=torch.int32, device=device)

    qsl_cpu = [0]
    for seg_len in segment_lengths:
        qsl_cpu.append(qsl_cpu[-1] + seg_len)

    return ShardQTokenSplit(
        ranges=ranges,
        local_to_global=local_to_global,
        local_query_start_loc=local_query_start_loc,
        local_seq_lens=local_seq_lens,
        segment_req_ids=req_ids,
        segment_offsets=offsets,
        segment_req_ids_cpu=tuple(segment_req_ids),
        local_query_start_loc_cpu=tuple(qsl_cpu),
        local_seq_lens_cpu=tuple(segment_seq_lens),
        seq_lens_cpu=tuple(seq_lens_cpu),
    )
