# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Register KV cache memory with NIXL in pieces a dma-buf MR cannot wrap.

RDMA stacks without linux 15fe76e23615 ("RDMA/umem: Fix truncation for block
sizes >= 4G") keep the DMA block iterator's step in 32 bits. When the driver
maps a contiguous dma-buf MR with blocks of 2^k >= 4 GiB, every block after an
aligned one repeats its address, so the MR wraps every 2^k bytes and RDMA ops
past the first block silently hit other memory (vllm-project/vllm#64048,
ai-dynamo/nixl#2512). An MR shorter than 4 GiB spans at most two such blocks,
and the one step between them is shorter than the MR, so it fits in 32 bits:
such an MR maps correctly wherever it lies, including across a 4 GiB boundary.

NIXL needs each strided descriptor run to lie inside one registration, so the
pieces are cut only where no descriptor crosses, and every run is split at the
same addresses before it is prepared. A split keeps the run's flat block
indices, so transfer requests index the prepared lists as before.

UCX's registration cache must be off for the pieces to stay separate MRs: it
registers a CUDA buffer's whole allocation and merges overlapping regions.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

CHUNKED_KV_REGISTRATION_ENV = "VLLM_NIXL_CHUNKED_KV_REGISTRATION"
# Size of the pieces a region is cut into.
KV_REGISTRATION_PIECE_BYTES = 1 << 31
# Longest registration kept whole: under 4 GiB with room for rounding the
# dma-buf range to pages at both ends.
KV_REGISTRATION_MAX_WHOLE_BYTES = (1 << 32) - (1 << 22)


_OFF = ("0", "n", "no", "off", "false")


def chunked_kv_registration_enabled() -> bool:
    value = os.environ.get(CHUNKED_KV_REGISTRATION_ENV, "1")
    return value.strip().lower() not in _OFF


def ucx_rcache_enabled(engine_config: str) -> bool:
    """Whether UCX keeps its registration cache. UCX_RCACHE_ENABLE in the
    environment wins over RCACHE_ENABLE in the NIXL UCX `engine_config`."""
    value = os.environ.get("UCX_RCACHE_ENABLE")
    if value is None:
        for item in engine_config.split(","):
            key, _, setting = item.partition("=")
            if key.strip() == "RCACHE_ENABLE":
                value = setting
    return value is None or value.strip().lower() not in _OFF


@dataclass(frozen=True)
class RegionRun:
    """Where a region's transfer descriptors may lie: block i's descriptors
    stay within [base + i * stride, base + i * stride + block_len)."""

    base: int
    block_len: int
    stride: int
    num_blocks: int

    @property
    def end(self) -> int:
        return self.base + (self.num_blocks - 1) * self.stride + self.block_len

    def block_containing(self, addr: int) -> int | None:
        """Start of the block extent `addr` lies strictly inside, if any."""
        if not self.base < addr < self.end:
            return None
        start = self.base + (addr - self.base) // self.stride * self.stride
        return start if start < addr < start + self.block_len else None


def _cut_inside(addr: int, runs: Sequence[RegionRun]) -> int | None:
    for run in runs:
        start = run.block_containing(addr)
        if start is not None:
            return start
    return None


def _row_cuts(lo: int, runs: list[RegionRun], piece_bytes: int) -> list[int] | None:
    """Equal cuts on the block rows every run shares, or None when the runs do
    not share rows. Equal counts keep NIXL's flat-index lookup cheap."""
    stride, num_blocks = runs[0].stride, runs[0].num_blocks
    if any(
        run.stride != stride
        or run.num_blocks != num_blocks
        or not 0 <= run.base - lo <= stride - run.block_len
        for run in runs
    ):
        return None
    rows_per_piece = max(1, piece_bytes // stride)
    min_pieces = -(-num_blocks // rows_per_piece)
    # Prefer pieces of exactly the same number of rows (up to 4x as many
    # pieces), so the split runs and the whole ones share a common count.
    fewest_rows = -(-num_blocks // (4 * min_pieces))
    rows = next(
        (r for r in range(rows_per_piece, fewest_rows - 1, -1) if num_blocks % r == 0),
        -(-num_blocks // min_pieces),
    )
    return list(range(lo + rows * stride, lo + num_blocks * stride, rows * stride))


def _greedy_cuts(
    lo: int, hi: int, runs: list[RegionRun], piece_bytes: int
) -> list[int]:
    """Cuts at the furthest point within piece_bytes that no descriptor
    crosses. Correct for any layout; sub-run counts may differ."""
    cuts = []
    cur = lo
    while hi - cur > piece_bytes:
        cut = cur + piece_bytes
        while (inside := _cut_inside(cut, runs)) is not None:
            cut = inside
        if cut <= cur:
            raise ValueError(
                f"no cut within {piece_bytes} bytes of {cur:#x}: a descriptor is longer"
            )
        cuts.append(cut)
        cur = cut
    return cuts


def plan_registration_pieces(
    start: int,
    end: int,
    runs: Sequence[RegionRun],
    piece_bytes: int = KV_REGISTRATION_PIECE_BYTES,
    max_whole_bytes: int = KV_REGISTRATION_MAX_WHOLE_BYTES,
) -> list[tuple[int, int]]:
    """Cut the registration [start, end) into (start, end) pieces.

    No descriptor of `runs` crosses a cut. A registration of at most
    `max_whole_bytes` stays whole. Region runs that do not interleave are
    grouped whole into pieces of at most `piece_bytes`; one that is longer
    but at most `max_whole_bytes` is a piece of its own, so its descriptor
    run is not split. Longer runs, alone or interleaved (packed block rows),
    are cut at block-row boundaries into equal pieces of at most
    `piece_bytes`. Raises ValueError when that is impossible.
    """
    if end - start <= max_whole_bytes:
        return [(start, end)]
    runs = [run for run in runs if run.num_blocks > 0]
    for run in runs:
        if not (start <= run.base and run.end <= end and run.block_len <= run.stride):
            raise ValueError(f"region run {run} is not inside [{start:#x}, {end:#x})")

    # Clusters of overlapping runs (packed rows interleave every run).
    clusters: list[tuple[int, int, list[RegionRun]]] = []
    for run in sorted(runs, key=lambda r: (r.base, r.end)):
        if clusters and run.base < clusters[-1][1]:
            lo, hi, members = clusters[-1]
            clusters[-1] = (lo, max(hi, run.end), members + [run])
        else:
            clusters.append((run.base, run.end, [run]))

    # Segments in address order: (lo, hi, splittable_with_neighbours).
    segments: list[tuple[int, int, bool]] = []
    cursor = start
    for lo, hi, members in clusters + [(end, end, [])]:
        # Bytes between clusters hold no descriptor: cut them anywhere.
        gap = cursor
        while lo - gap > piece_bytes:
            segments.append((gap, gap + piece_bytes, True))
            gap += piece_bytes
        if lo > gap:
            segments.append((gap, lo, True))
        if not members:
            break
        if hi - lo <= max_whole_bytes:
            segments.append((lo, hi, True))
        else:
            cuts = _row_cuts(lo, members, piece_bytes)
            if cuts is None:
                cuts = _greedy_cuts(lo, hi, members, piece_bytes)
            bounds = [lo] + cuts + [hi]
            # Pieces of one long cluster never merge with each other.
            segments.extend(
                (a, b, k in (0, len(bounds) - 2))
                for k, (a, b) in enumerate(zip(bounds, bounds[1:]))
            )
        cursor = hi

    # Merge neighbours while a piece stays within piece_bytes.
    pieces: list[list[int]] = []
    open_end = False
    for lo, hi, mergeable in segments:
        if pieces and open_end and mergeable and hi - pieces[-1][0] <= piece_bytes:
            pieces[-1][1] = hi
        else:
            pieces.append([lo, hi])
        open_end = mergeable
    result = [(lo, hi) for lo, hi in pieces]
    check_registration_pieces(start, end, result, runs, max_whole_bytes)
    return result


def check_registration_pieces(
    start: int,
    end: int,
    pieces: Sequence[tuple[int, int]],
    runs: Sequence[RegionRun],
    max_whole_bytes: int = KV_REGISTRATION_MAX_WHOLE_BYTES,
) -> None:
    """Raise ValueError unless the pieces tile [start, end), each is at most
    `max_whole_bytes` (when there is more than one) and no descriptor of
    `runs` crosses a cut."""
    if not pieces or pieces[0][0] != start or pieces[-1][1] != end:
        raise ValueError("pieces do not cover the registration")
    for i, (lo, hi) in enumerate(pieces):
        if lo >= hi or (i + 1 < len(pieces) and pieces[i + 1][0] != hi):
            raise ValueError(f"pieces are not contiguous at {hi:#x}")
        if len(pieces) > 1 and hi - lo > max_whole_bytes:
            raise ValueError(f"piece [{lo:#x}, {hi:#x}) is {hi - lo} bytes")
    for lo, _ in pieces[1:]:
        if _cut_inside(lo, runs) is not None:
            raise ValueError(f"a descriptor crosses the cut at {lo:#x}")


def split_strided_descs(descs: np.ndarray, cuts: Sequence[int]) -> np.ndarray:
    """Split each Nx5 (addr, len, dev_id, stride, count) row where its span
    crosses one of the sorted `cuts`, keeping the rows' flat block order.

    Raises ValueError if a cut falls inside one block."""
    if len(cuts) == 0 or len(descs) == 0:
        return descs
    cut_array = np.asarray(cuts, dtype=np.uint64)
    addr = descs[:, 0]
    span_end = addr + (descs[:, 4] - 1) * descs[:, 3] + descs[:, 1]
    first_cut = np.searchsorted(cut_array, addr, side="right")
    last_cut = np.searchsorted(cut_array, span_end, side="left")
    crossing = np.nonzero(last_cut > first_cut)[0]
    if crossing.size == 0:
        return descs
    parts = []
    done = 0
    for row in crossing.tolist():
        parts.append(descs[done:row])
        base, length, dev_id, stride, count = (int(x) for x in descs[row])
        first = 0
        for cut in cut_array[first_cut[row] : last_cut[row]].tolist():
            # First block starting at or after the cut; the one before it
            # must end at or before the cut.
            block = -(-(cut - base) // stride) if stride else count
            if block == 0 or base + (block - 1) * stride + length > cut:
                raise ValueError(
                    f"registration cut {cut:#x} falls inside a descriptor of "
                    f"the run at {base:#x} (len {length}, stride {stride})"
                )
            if block > first:
                parts.append(
                    np.array(
                        [
                            [
                                base + first * stride,
                                length,
                                dev_id,
                                stride,
                                block - first,
                            ]
                        ],
                        dtype=np.uint64,
                    )
                )
                first = block
        parts.append(
            np.array(
                [[base + first * stride, length, dev_id, stride, count - first]],
                dtype=np.uint64,
            )
        )
        done = row + 1
    parts.append(descs[done:])
    return _even_runs(np.concatenate(parts).astype(np.uint64, copy=False), len(descs))


def _even_runs(descs: np.ndarray, num_rows_before: int) -> np.ndarray:
    """Cut every run into runs of the counts' common divisor when that adds
    few rows. NIXL finds the run of a flat index with one division when all
    runs have the same count, and with a binary search otherwise."""
    counts = descs[:, 4].astype(np.int64)
    if (counts == counts[0]).all():
        return descs
    unit = int(np.gcd.reduce(counts))
    pieces = counts // unit
    if int(pieces.sum()) > 8 * max(num_rows_before, len(descs)):
        return descs
    even = np.repeat(descs, pieces, axis=0)
    first = np.repeat(np.cumsum(pieces) - pieces, pieces)
    offset = (np.arange(len(even)) - first).astype(np.uint64) * np.uint64(unit)
    even[:, 0] += offset * even[:, 3]
    even[:, 4] = unit
    return even


def spans_4gib_boundary(lo: int, hi: int) -> bool:
    return (lo >> 32) != ((hi - 1) >> 32)
