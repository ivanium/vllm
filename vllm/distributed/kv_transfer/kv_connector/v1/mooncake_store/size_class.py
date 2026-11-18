# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Store objects on Mooncake offset-allocator size classes.

Mooncake's default `offset` allocator rounds every allocation up to a size
class with 3 mantissa bits (`SmallFloat` in
`mooncake-store/src/offset_allocator.cpp`), while the master's `Mem Storage`
and its eviction watermark count the requested bytes. An object just above a
class costs up to 12.5% more than it reports, and allocations start failing
below the watermark. A 4,976,640-byte block, for example, takes 5,242,880.

With `VLLM_MOONCAKE_STORE_SIZE_CLASS_SPLIT=1`, every object of a cache group
whose objects are off a size class is stored as two: a head holding the largest
size class below its size, and a tail holding the rest (4,718,592 + 258,048
bytes, allocated as 4,718,592 + 262,144). The parts are keys of their own
(`<key>@part:0`, `<key>@part:1`) and are read and written from the same
registered buffers, so transfers stay zero-copy. A block is stored only while
both parts are: a lookup or a load that misses either part misses the block.
"""

from collections.abc import Iterable, Sequence
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

SIZE_CLASS_SPLIT_ENV = "VLLM_MOONCAKE_STORE_SIZE_CLASS_SPLIT"

# Mooncake OffsetAllocator: SmallFloat::MANTISSA_BITS.
_MANTISSA_BITS = 3
_MANTISSA_VALUE = 1 << _MANTISSA_BITS
_MANTISSA_MASK = _MANTISSA_VALUE - 1

PART_SUFFIXES = ("@part:0", "@part:1")


def _bin_round_up(size: int) -> int:
    """`SmallFloat::uintToFloatRoundUp`: the smallest bin that holds `size`."""
    if size < _MANTISSA_VALUE:
        return size
    mantissa_start_bit = size.bit_length() - 1 - _MANTISSA_BITS
    exp = mantissa_start_bit + 1
    mantissa = (size >> mantissa_start_bit) & _MANTISSA_MASK
    if size & ((1 << mantissa_start_bit) - 1):
        mantissa += 1
    # `+` lets a rounded-up mantissa carry into the exponent.
    return (exp << _MANTISSA_BITS) + mantissa


def _bin_round_down(size: int) -> int:
    """`SmallFloat::uintToFloatRoundDown`: the largest bin not above `size`."""
    if size < _MANTISSA_VALUE:
        return size
    mantissa_start_bit = size.bit_length() - 1 - _MANTISSA_BITS
    exp = mantissa_start_bit + 1
    mantissa = (size >> mantissa_start_bit) & _MANTISSA_MASK
    return (exp << _MANTISSA_BITS) | mantissa


def _bin_size(bin_index: int) -> int:
    """`SmallFloat::floatToUint`."""
    exp = bin_index >> _MANTISSA_BITS
    mantissa = bin_index & _MANTISSA_MASK
    if exp == 0:
        return mantissa
    return (mantissa | _MANTISSA_VALUE) << (exp - 1)


def size_class_ceil(size: int) -> int:
    """Bytes the offset allocator reserves for a `size`-byte object."""
    return _bin_size(_bin_round_up(size))


def size_class_floor(size: int) -> int:
    """The largest size class not above `size`."""
    return _bin_size(_bin_round_down(size))


def is_size_class(size: int) -> bool:
    return size_class_ceil(size) == size


def split_head_bytes(total: int) -> int:
    """Head bytes of a split `total`-byte object: the largest size class below
    it, so the tail is never empty."""
    if total < 2:
        raise ValueError(f"cannot split a {total}-byte object")
    return size_class_floor(total - 1)


def split_buffers(
    addrs: Sequence[int], sizes: Sequence[int], head_bytes: int
) -> tuple[list[int], list[int], list[int], list[int]]:
    """Split one object's buffers at byte `head_bytes` of their concatenation.

    Returns the head's and the tail's (addresses, sizes); a buffer the cut
    falls inside is shared between them.
    """
    head_addrs: list[int] = []
    head_sizes: list[int] = []
    tail_addrs: list[int] = []
    tail_sizes: list[int] = []
    remaining = head_bytes
    for addr, size in zip(addrs, sizes, strict=True):
        if remaining >= size:
            head_addrs.append(addr)
            head_sizes.append(size)
            remaining -= size
        elif remaining > 0:
            head_addrs.append(addr)
            head_sizes.append(remaining)
            tail_addrs.append(addr + remaining)
            tail_sizes.append(size - remaining)
            remaining = 0
        else:
            tail_addrs.append(addr)
            tail_sizes.append(size)
    if remaining or not tail_sizes:
        raise ValueError(
            f"cannot cut {sum(sizes)} bytes into a {head_bytes}-byte head and a tail"
        )
    return head_addrs, head_sizes, tail_addrs, tail_sizes


def key_prefix(key: str) -> str:
    """The key without its chunk hash (`PoolKey.build_key_string`)."""
    return key.rpartition("@")[0]


def _combine_status(values: Sequence[int]) -> int:
    """A put's or get's result for a split object: the first error, else the
    head's result plus the tail's (the bytes read, for a get)."""
    for value in values:
        if value < 0:
            return value
    return sum(values)


def _combine_exists(values: Sequence[int]) -> int:
    """1 when every part exists, 0 when one is missing, else the first error."""
    for value in values:
        if value < 0:
            return value
    return 1 if all(value == 1 for value in values) else 0


class SizeClassSplitStore:
    """`MooncakeDistributedStore` proxy that stores the objects under the given
    key prefixes as a size-class head and a tail.

    The methods the connector calls with keys (`batch_is_exist`,
    `batch_probe_key` without a lease policy, `is_exist`,
    `batch_put_from_multi_buffers`, `batch_get_into_multi_buffers` and
    `batch_get_replica_desc`) take and return one entry per caller key; every
    other attribute is the store's own.
    """

    def __init__(self, store: Any):
        self._store = store
        self._split_prefixes: frozenset[str] = frozenset()
        self._logged_first_put = False
        # Mirror optional store methods, so feature checks still see whether
        # the installed Mooncake has them.
        if callable(getattr(store, "batch_probe_key", None)):
            self.batch_probe_key = self._batch_probe_key
        if callable(getattr(store, "batch_get_replica_desc", None)):
            self.batch_get_replica_desc = self._batch_get_replica_desc

    def __getattr__(self, name: str) -> Any:
        if name == "_store":
            raise AttributeError(name)
        return getattr(self._store, name)

    @property
    def split_prefixes(self) -> frozenset[str]:
        return self._split_prefixes

    def set_split_prefixes(self, prefixes: Iterable[str]) -> None:
        self._split_prefixes = frozenset(prefixes)

    def is_split(self, key: str) -> bool:
        return bool(self._split_prefixes) and key_prefix(key) in self._split_prefixes

    @staticmethod
    def part_keys(key: str) -> tuple[str, str]:
        return (key + PART_SUFFIXES[0], key + PART_SUFFIXES[1])

    def _expand_keys(self, keys: Sequence[str]) -> tuple[list[str], list[int]]:
        """Each key's part keys, in order, and how many parts each key has."""
        expanded: list[str] = []
        counts: list[int] = []
        for key in keys:
            if self.is_split(key):
                expanded.extend(self.part_keys(key))
                counts.append(2)
            else:
                expanded.append(key)
                counts.append(1)
        return expanded, counts

    @staticmethod
    def _combine(values: Sequence[int], counts: Sequence[int], combine) -> list[int]:
        out: list[int] = []
        pos = 0
        for count in counts:
            if count == 1:
                out.append(values[pos])
            else:
                out.append(combine(values[pos : pos + count]))
            pos += count
        return out

    def _expand_buffers(
        self,
        keys: Sequence[str],
        all_buffer_ptrs: Sequence[Sequence[int]],
        all_sizes: Sequence[Sequence[int]],
    ) -> tuple[list[str], list[Sequence[int]], list[Sequence[int]], list[int]]:
        expanded_keys: list[str] = []
        expanded_ptrs: list[Sequence[int]] = []
        expanded_sizes: list[Sequence[int]] = []
        counts: list[int] = []
        for key, ptrs, sizes in zip(keys, all_buffer_ptrs, all_sizes, strict=True):
            if not self.is_split(key):
                expanded_keys.append(key)
                expanded_ptrs.append(ptrs)
                expanded_sizes.append(sizes)
                counts.append(1)
                continue
            head_bytes = split_head_bytes(sum(sizes))
            head_ptrs, head_sizes, tail_ptrs, tail_sizes = split_buffers(
                ptrs, sizes, head_bytes
            )
            expanded_keys.extend(self.part_keys(key))
            expanded_ptrs.extend((head_ptrs, tail_ptrs))
            expanded_sizes.extend((head_sizes, tail_sizes))
            counts.append(2)
        return expanded_keys, expanded_ptrs, expanded_sizes, counts

    def batch_is_exist(self, keys: Sequence[str], *args: Any, **kwargs: Any):
        expanded, counts = self._expand_keys(keys)
        if len(expanded) == len(keys):
            return self._store.batch_is_exist(keys, *args, **kwargs)
        values = self._store.batch_is_exist(expanded, *args, **kwargs)
        return self._combine(values, counts, _combine_exists)

    def _batch_probe_key(self, keys: Sequence[str], *args: Any, **kwargs: Any):
        expanded, counts = self._expand_keys(keys)
        if len(expanded) == len(keys):
            return self._store.batch_probe_key(keys, *args, **kwargs)
        if args or kwargs:
            # A lease policy leases groups of consecutive results, which the
            # parts would shift.
            raise ValueError(
                f"{SIZE_CLASS_SPLIT_ENV} supports batch_probe_key only "
                "without a lease policy"
            )
        values = self._store.batch_probe_key(expanded)
        return self._combine(values, counts, _combine_exists)

    def is_exist(self, key: str, *args: Any, **kwargs: Any):
        if not self.is_split(key):
            return self._store.is_exist(key, *args, **kwargs)
        values = self._store.batch_is_exist(list(self.part_keys(key)), *args, **kwargs)
        return _combine_exists(values)

    def _batch_get_replica_desc(self, keys: Sequence[str]):
        # A split object's replicas are reported by its head.
        heads = [self.part_keys(key)[0] if self.is_split(key) else key for key in keys]
        descs = self._store.batch_get_replica_desc(heads)
        if heads == list(keys):
            return descs
        return {key: descs.get(head) for key, head in zip(keys, heads, strict=True)}

    def batch_put_from_multi_buffers(
        self,
        keys: Sequence[str],
        all_buffer_ptrs: Sequence[Sequence[int]],
        all_sizes: Sequence[Sequence[int]],
        *args: Any,
        **kwargs: Any,
    ):
        expanded_keys, ptrs, sizes, counts = self._expand_buffers(
            keys, all_buffer_ptrs, all_sizes
        )
        if len(expanded_keys) == len(keys):
            return self._store.batch_put_from_multi_buffers(
                keys, all_buffer_ptrs, all_sizes, *args, **kwargs
            )
        if not self._logged_first_put:
            self._logged_first_put = True
            logger.info(
                "Mooncake store size-class split engaged: the first split put "
                "stores %d keys as %d objects.",
                len(keys),
                len(expanded_keys),
            )
        # Group ids name each key's lifecycle unit; both parts share it.
        config = args[0] if args else kwargs.get("config")
        group_ids: Any = getattr(config, "group_ids", None)
        grouped = (
            config
            if config is not None and group_ids and len(group_ids) == len(keys)
            else None
        )
        if grouped is not None:
            grouped.group_ids = [
                group_id
                for group_id, count in zip(group_ids, counts, strict=True)
                for _ in range(count)
            ]
        try:
            values = self._store.batch_put_from_multi_buffers(
                expanded_keys, ptrs, sizes, *args, **kwargs
            )
        finally:
            if grouped is not None:
                grouped.group_ids = group_ids
        return self._combine(values, counts, _combine_status)

    def batch_get_into_multi_buffers(
        self,
        keys: Sequence[str],
        all_buffer_ptrs: Sequence[Sequence[int]],
        all_sizes: Sequence[Sequence[int]],
        *args: Any,
        **kwargs: Any,
    ):
        expanded_keys, ptrs, sizes, counts = self._expand_buffers(
            keys, all_buffer_ptrs, all_sizes
        )
        if len(expanded_keys) == len(keys):
            return self._store.batch_get_into_multi_buffers(
                keys, all_buffer_ptrs, all_sizes, *args, **kwargs
            )
        values = self._store.batch_get_into_multi_buffers(
            expanded_keys, ptrs, sizes, *args, **kwargs
        )
        return self._combine(values, counts, _combine_status)
