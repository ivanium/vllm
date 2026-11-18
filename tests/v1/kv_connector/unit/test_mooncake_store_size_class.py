# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MooncakeStoreConnector objects on Mooncake offset-allocator size classes
(`VLLM_MOONCAKE_STORE_SIZE_CLASS_SPLIT`)."""

import ctypes
import random
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from tests.v1.kv_connector.unit.offloading_connector.utils import (
    make_kv_cache_group,
    sliding_window_layer,
)
from tests.v1.kv_connector.unit.test_mooncake_store_hma_e2e import (
    _build_worker_with_dict_store,
    _minimal_vllm_config,
    _register_with_fake_threads,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake_store.data import (
    LoadSpec,
    ReqMeta,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake_store.size_class import (  # noqa: E501
    SIZE_CLASS_SPLIT_ENV,
    SizeClassSplitStore,
    is_size_class,
    key_prefix,
    size_class_ceil,
    size_class_floor,
    split_buffers,
    split_head_bytes,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake_store.worker import (  # noqa: E501
    KVCacheStoreRecvingThread,
    KVCacheStoreSendingThread,
)
from vllm.v1.core.kv_cache_planning import get_kv_cache_config_from_groups
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.v1.kv_cache_layout import KVCacheLayout
from vllm.v1.worker.utils import allocate_kv_cache

# Mooncake ErrorCode values.
NO_AVAILABLE_HANDLE = -200
INVALID_PARAMS = -600
OBJECT_NOT_FOUND = -704


def _reference_size_classes(limit: int) -> list[int]:
    """Every size the offset allocator can reserve up to `limit`: 0-7, then
    m * 2**e for m in 8..15 (3 mantissa bits)."""
    classes = set(range(8))
    e = 0
    while (8 << e) <= limit:
        classes.update(m << e for m in range(8, 16))
        e += 1
    return sorted(classes)


class _BytesStore:
    """Byte-accurate MooncakeDistributedStore stand-in."""

    def __init__(self, *, probe: bool = True):
        self.objects: dict[str, bytes] = {}
        self.fail_keys: set[str] = set()
        self.put_calls: list[tuple[list[str], list[str] | None]] = []
        if probe:
            self.batch_probe_key = self._batch_probe_key

    def setup(self, *_args, **_kwargs):
        return 0

    def register_buffer(self, addr, length):
        return 0

    def remove_all(self, force=False):
        self.objects.clear()
        return 0

    def reserved_bytes(self) -> int:
        """What the offset allocator reserves for the stored objects."""
        return sum(size_class_ceil(len(value)) for value in self.objects.values())

    def requested_bytes(self) -> int:
        return sum(len(value) for value in self.objects.values())

    def batch_is_exist(self, keys):
        return [1 if key in self.objects else 0 for key in keys]

    def is_exist(self, key):
        return 1 if key in self.objects else 0

    def _batch_probe_key(self, keys, policy="none", candidate_size=1):
        return self.batch_is_exist(keys)

    def batch_get_replica_desc(self, keys):
        return {key: [f"memory:{key}"] for key in keys if key in self.objects}

    def batch_put_from_multi_buffers(self, keys, addrs, sizes, config=None):
        group_ids = getattr(config, "group_ids", None)
        self.put_calls.append((list(keys), list(group_ids) if group_ids else None))
        results = []
        for key, key_addrs, key_sizes in zip(keys, addrs, sizes, strict=True):
            if key in self.fail_keys:
                results.append(NO_AVAILABLE_HANDLE)
                continue
            # An existing object is an idempotent success, as in BatchPut.
            self.objects.setdefault(
                key,
                b"".join(
                    ctypes.string_at(a, n)
                    for a, n in zip(key_addrs, key_sizes, strict=True)
                ),
            )
            results.append(0)
        return results

    def batch_get_into_multi_buffers(self, keys, addrs, sizes, prefer=False):
        results = []
        for key, key_addrs, key_sizes in zip(keys, addrs, sizes, strict=True):
            data = self.objects.get(key)
            if data is None:
                results.append(OBJECT_NOT_FOUND)
                continue
            if sum(key_sizes) < len(data):
                results.append(INVALID_PARAMS)
                continue
            pos = 0
            for a, n in zip(key_addrs, key_sizes, strict=True):
                piece = data[pos : pos + n]
                ctypes.memmove(a, piece, len(piece))
                pos += len(piece)
            results.append(len(data))
        return results


def _buffer(nbytes: int, seed: int) -> ctypes.Array:
    rng = random.Random(seed)
    return ctypes.create_string_buffer(rng.randbytes(nbytes), nbytes)


def _segments(buf: ctypes.Array, sizes: list[int]) -> tuple[list[int], list[int]]:
    base = ctypes.addressof(buf)
    addrs, offset = [], 0
    for size in sizes:
        addrs.append(base + offset)
        offset += size
    return addrs, list(sizes)


PREFIX = "model@tp_rank:0@pcp0@dcp0@pp_rank:0@group:0"
OTHER_PREFIX = "model@tp_rank:0@pcp0@dcp0@pp_rank:0@group:1"


def _split_store(store: _BytesStore) -> SizeClassSplitStore:
    split = SizeClassSplitStore(store)
    split.set_split_prefixes([PREFIX])
    return split


# ----------------------------------------------------------------------------
# Size classes
# ----------------------------------------------------------------------------


def test_size_classes_match_mooncake_small_float():
    """The replica of SmallFloat rounds like the allocator: up to the
    smallest 3-mantissa-bit class, and is unchanged by a power-of-two
    allocation quantum (the multiplier bits of a large segment)."""
    classes = _reference_size_classes(1 << 24)
    rng = random.Random(0)
    samples = list(range(1, 4096)) + [rng.randrange(1, 1 << 24) for _ in range(20000)]
    for n in samples:
        assert size_class_ceil(n) == min(c for c in classes if c >= n), n
        assert size_class_floor(n) == max(c for c in classes if c <= n), n
        assert is_size_class(n) == (n in classes)
    # Multiplier bits 4-6 (45, 60, 123 and 180 GiB segments) on KV-sized,
    # quantum-aligned objects: the class in units is the class in bytes.
    for quantum_bits in (4, 5, 6):
        q = 1 << quantum_bits
        for n in (4_976_640, 5_632_000, 258_048, 389_120, 655_360, 4_718_592):
            assert size_class_ceil(-(-n // q)) * q == size_class_ceil(n)


@pytest.mark.parametrize(
    "total, head, tail, tail_reserved, unsplit_reserved",
    [
        # Drafter KV tail layout: target + index-K, 38,880 B per token x 128.
        (4_976_640, 4_718_592, 258_048, 262_144, 5_242_880),
        # Stock: every group in one object, 44,000 B per token x 128.
        (5_632_000, 5_242_880, 389_120, 393_216, 5_767_168),
    ],
)
def test_split_sizes(total, head, tail, tail_reserved, unsplit_reserved):
    assert not is_size_class(total)
    assert size_class_ceil(total) == unsplit_reserved
    assert split_head_bytes(total) == head and is_size_class(head)
    assert total - head == tail
    assert size_class_ceil(tail) == tail_reserved
    reserved = head + tail_reserved
    print(
        f"{total}: {head} + {tail} -> reserved {reserved} "
        f"(+{100 * (reserved / total - 1):.2f}%), unsplit {unsplit_reserved} "
        f"(+{100 * (unsplit_reserved / total - 1):.2f}%)"
    )
    assert reserved / total < 1.001


@pytest.mark.parametrize(
    "object_bytes, split",
    [
        (4_976_640, True),  # group 0: target K+V and index-K of a block
        (655_360, False),  # group 1: DSpark drafter K+V, a size class
        (5_632_000, True),  # stock: every layer in one object
        # The same layers if a layout stored them as objects of their own.
        (4_423_680, True),  # target K+V alone
        (552_960, True),  # index-K alone
        (368_640, True),  # an FP4 drafter
    ],
)
def test_every_object_size_reserves_what_it_requests(object_bytes, split):
    """Whatever object sizes a group has, what the worker stores reserves at
    most 0.1% more than it requests, so allocations cannot fail below a
    requested-byte watermark of 0.99."""
    assert (not is_size_class(object_bytes)) == split
    if split:
        head = split_head_bytes(object_bytes)
        reserved = size_class_ceil(head) + size_class_ceil(object_bytes - head)
    else:
        reserved = size_class_ceil(object_bytes)
    assert reserved / object_bytes < 1.001


def test_split_head_is_below_a_size_class_total():
    # The drafter object of that layout is a class: a group of such objects is not
    # split, and a forced split still leaves a non-empty tail.
    assert is_size_class(655_360)
    assert split_head_bytes(655_360) == 589_824
    with pytest.raises(ValueError):
        split_head_bytes(1)


def test_split_buffers_cuts_inside_a_buffer():
    addrs, sizes = [1000, 50_000, 90_000], [9216, 9216, 9216]
    head = split_head_bytes(sum(sizes))
    assert head == 26_624
    assert split_buffers(addrs, sizes, head) == (
        [1000, 50_000, 90_000],
        [9216, 9216, 8192],
        [98_192],
        [1024],
    )
    # A cut on a buffer boundary shares no buffer.
    assert split_buffers(addrs, sizes, 18_432) == (
        [1000, 50_000],
        [9216, 9216],
        [90_000],
        [9216],
    )
    with pytest.raises(ValueError):
        split_buffers(addrs, sizes, sum(sizes))


# ----------------------------------------------------------------------------
# The store proxy
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "segment_sizes",
    [
        [4_976_640],  # target key of that layout, one block-outermost run
        [5_632_000],  # stock key
        [9216, 9216, 9216],  # three layer-major segments, cut inside the third
    ],
)
def test_put_then_get_round_trip(segment_sizes):
    store = _BytesStore()
    split = _split_store(store)
    total = sum(segment_sizes)
    src = _buffer(total, seed=1)
    key = f"{PREFIX}@{'ab' * 32}"
    other = f"{OTHER_PREFIX}@{'ab' * 32}"
    other_src = _buffer(9216, seed=2)

    addrs, sizes = _segments(src, segment_sizes)
    other_addrs, other_sizes = _segments(other_src, [9216])
    assert split.batch_put_from_multi_buffers(
        [key, other], [addrs, other_addrs], [sizes, other_sizes], None
    ) == [0, 0]

    head = split_head_bytes(total)
    assert {k: len(v) for k, v in store.objects.items()} == {
        f"{key}@part:0": head,
        f"{key}@part:1": total - head,
        other: 9216,
    }
    assert store.objects[f"{key}@part:0"] + store.objects[f"{key}@part:1"] == bytes(
        src.raw
    )
    split_reserved = store.reserved_bytes() - 9216
    assert split_reserved == size_class_ceil(head) + size_class_ceil(total - head)
    assert split_reserved < size_class_ceil(total)

    dst = ctypes.create_string_buffer(total)
    other_dst = ctypes.create_string_buffer(9216)
    dst_addrs, _ = _segments(dst, segment_sizes)
    other_dst_addrs, _ = _segments(other_dst, [9216])
    assert split.batch_get_into_multi_buffers(
        [key, other], [dst_addrs, other_dst_addrs], [sizes, other_sizes]
    ) == [total, 9216]
    assert dst.raw == src.raw and other_dst.raw == other_src.raw
    assert split.batch_is_exist([key, other]) == [1, 1]


@pytest.mark.parametrize("evicted_part", [0, 1])
def test_a_missing_part_misses_the_object(evicted_part):
    store = _BytesStore()
    split = _split_store(store)
    keys = [f"{PREFIX}@{h * 64}" for h in "12"]
    srcs = [_buffer(27_648, seed=i) for i in range(2)]
    segs = [_segments(src, [9216, 9216, 9216]) for src in srcs]
    split.batch_put_from_multi_buffers(
        keys, [s[0] for s in segs], [s[1] for s in segs], None
    )
    del store.objects[f"{keys[1]}@part:{evicted_part}"]

    assert split.batch_is_exist(keys) == [1, 0]
    assert split.batch_probe_key(keys) == [1, 0]
    assert [split.is_exist(key) for key in keys] == [1, 0]
    dsts = [ctypes.create_string_buffer(27_648) for _ in keys]
    dst_segs = [_segments(dst, [9216, 9216, 9216]) for dst in dsts]
    res = split.batch_get_into_multi_buffers(
        keys, [s[0] for s in dst_segs], [s[1] for s in dst_segs]
    )
    assert res == [27_648, OBJECT_NOT_FOUND]
    assert dsts[0].raw == srcs[0].raw


def test_a_failed_part_fails_the_put_and_a_retry_completes_it():
    store = _BytesStore()
    split = _split_store(store)
    keys = [f"{PREFIX}@{h * 64}" for h in "12"]
    srcs = [_buffer(27_648, seed=i) for i in range(2)]
    segs = [_segments(src, [27_648]) for src in srcs]
    args = (keys, [s[0] for s in segs], [s[1] for s in segs], None)

    store.fail_keys = {f"{keys[1]}@part:1"}
    assert split.batch_put_from_multi_buffers(*args) == [0, NO_AVAILABLE_HANDLE]
    assert split.batch_is_exist(keys) == [1, 0]
    # The save path re-puts what the existence check reports missing; the
    # stored head is an idempotent success.
    store.fail_keys = set()
    assert split.batch_put_from_multi_buffers(*args) == [0, 0]
    assert split.batch_is_exist(keys) == [1, 1]


def test_group_ids_cover_both_parts_and_are_restored():
    store = _BytesStore()
    split = _split_store(store)
    keys = [f"{PREFIX}@{'1' * 64}", f"{OTHER_PREFIX}@{'1' * 64}"]
    src = _buffer(27_648, seed=3)
    seg = _segments(src, [27_648])
    config = SimpleNamespace(group_ids=["g1", "g2"])
    split.batch_put_from_multi_buffers(keys, [seg[0], seg[0]], [seg[1], seg[1]], config)
    assert store.put_calls[-1] == (
        [f"{keys[0]}@part:0", f"{keys[0]}@part:1", keys[1]],
        ["g1", "g1", "g2"],
    )
    assert config.group_ids == ["g1", "g2"]


def test_keys_outside_the_split_prefixes_pass_through():
    store = _BytesStore()
    split = SizeClassSplitStore(store)  # no prefixes yet
    store.batch_put_from_multi_buffers = MagicMock(return_value=[0])
    keys, addrs, sizes = [f"{PREFIX}@{'1' * 64}"], [[0x1000]], [[27_648]]
    assert split.batch_put_from_multi_buffers(keys, addrs, sizes, None) == [0]
    store.batch_put_from_multi_buffers.assert_called_once_with(keys, addrs, sizes, None)
    # Every other attribute is the store's own.
    assert split.remove_all is not None
    assert split.register_buffer(0x1000, 10) == 0


def test_optional_store_methods_are_mirrored():
    without = SizeClassSplitStore(_BytesStore(probe=False))
    assert getattr(without, "batch_probe_key", None) is None
    with_probe = _split_store(_BytesStore())
    with pytest.raises(ValueError):
        with_probe.batch_probe_key([f"{PREFIX}@{'1' * 64}"], "LastHitOnly", 2)
    # Unsplit keys keep any policy.
    assert with_probe.batch_probe_key([f"{OTHER_PREFIX}@1"], "LastHitOnly", 1) == [0]


def test_replica_descriptors_are_the_heads():
    store = _BytesStore()
    split = _split_store(store)
    key, other = f"{PREFIX}@{'1' * 64}", f"{OTHER_PREFIX}@{'1' * 64}"
    store.objects.update({f"{key}@part:0": b"h", f"{key}@part:1": b"t", other: b"o"})
    assert split.batch_get_replica_desc([key, other]) == {
        key: [f"memory:{key}@part:0"],
        other: [f"memory:{other}"],
    }


def test_a_reader_without_the_split_misses_split_objects():
    """Parts have keys of their own, so an instance without the switch sees
    a miss, never a head read into a whole object's buffers."""
    store = _BytesStore()
    split = _split_store(store)
    key = f"{PREFIX}@{'1' * 64}"
    src = _buffer(27_648, seed=4)
    seg = _segments(src, [27_648])
    split.batch_put_from_multi_buffers([key], [seg[0]], [seg[1]], None)
    assert store.batch_is_exist([key]) == [0]
    dst = ctypes.create_string_buffer(27_648)
    dst_seg = _segments(dst, [27_648])
    assert store.batch_get_into_multi_buffers([key], [dst_seg[0]], [dst_seg[1]]) == [
        OBJECT_NOT_FOUND
    ]


# ----------------------------------------------------------------------------
# The connector worker, through the planner's real KV layout
# ----------------------------------------------------------------------------

# As with MiniMax-M3's drafter KV tail layout: the target group's objects are off a size class
# (3 x 9,216 B per block), the drafter's sliding-window group's are on one
# (2 x 9,216 B).
_TARGET = AttentionSpec(block_size=16, num_kv_heads=4, head_size=72, dtype=torch.uint8)
_DRAFTER = sliding_window_layer(
    block_size=16, num_kv_heads=4, head_size=72, dtype=torch.uint8, sliding_window=32
)
_NUM_BLOCKS = 4
_HASHES = [BlockHash(bytes([i + 1]) * 4) for i in range(_NUM_BLOCKS - 1)]
_TOKENS = 16 * len(_HASHES)


def _worker(monkeypatch, layout: KVCacheLayout, split: bool):
    if split:
        monkeypatch.setenv(SIZE_CLASS_SPLIT_ENV, "1")
    else:
        monkeypatch.delenv(SIZE_CLASS_SPLIT_ENV, raising=False)
    groups = [
        make_kv_cache_group(["A0", "A1", "A2"], _TARGET),
        make_kv_cache_group(["S0", "S1"], _DRAFTER),
    ]
    planner_config = SimpleNamespace(
        cache_config=SimpleNamespace(
            get_resolved_kv_cache_layout=lambda: layout,
            num_gpu_blocks_override=_NUM_BLOCKS,
            prefix_cache_retention_interval=None,
        ),
        attention_config=SimpleNamespace(hisparse_config=None),
    )
    try:
        cfg = get_kv_cache_config_from_groups(planner_config, groups, 1 << 30)
        kv_caches = allocate_kv_cache(cfg, torch.device("cpu"), layout)
    except (ValueError, AssertionError):
        pytest.skip(f"{layout.name} cannot express the groups")
    store = _BytesStore()
    worker = _build_worker_with_dict_store(_minimal_vllm_config(16), cfg, store)
    _register_with_fake_threads(worker, kv_caches)
    first = next(iter(kv_caches.values()))
    raw = torch.empty(0, dtype=torch.int8).set_(first.untyped_storage())
    return worker, store, raw


def _save(worker, raw: torch.Tensor) -> None:
    generator = torch.Generator().manual_seed(7)
    raw.copy_(
        torch.randint(-128, 128, raw.shape, dtype=torch.int8, generator=generator)
    )
    thread = KVCacheStoreSendingThread(
        store=worker.store,
        token_databases=worker.token_dbs,
        block_size=worker.block_size,
        coord=worker.coord,
        tp_rank=worker.tp_rank,
        group_put_steps=worker._group_tp_replication_factors,
        kv_role=worker.kv_role,
        ready_event=threading.Event(),
    )
    thread.add_request(
        ReqMeta(
            req_id="r0",
            token_len_chunk=_TOKENS,
            block_ids=([1, 2, 3], [1, 2, 3]),
            block_hashes=_HASHES,
            can_save=True,
            store_job_id=1,
        )
    )
    thread._handle_request(thread.request_queue.get())


def _load(worker, raw: torch.Tensor, hit) -> list[int]:
    raw.fill_(7)
    thread = KVCacheStoreRecvingThread(
        store=worker.store,
        coord=worker.coord,
        token_databases=worker.token_dbs,
        block_size=worker.block_size,
        tp_rank=worker.tp_rank,
        ready_event=threading.Event(),
        group_participates=[True, True],
        is_hma_required=True,
    )
    req = ReqMeta(
        req_id="r1",
        token_len_chunk=hit.hit_length,
        block_ids=([1, 2, 3], [1, 2, 3]),
        block_hashes=_HASHES,
        load_spec=LoadSpec(
            vllm_cached_tokens=0,
            kvpool_cached_tokens=hit.hit_length,
            can_load=True,
            token_len=hit.hit_length,
            tail_key_boundaries=hit.tail_key_boundaries,
        ),
    )
    thread.request_queue.put(req)
    thread._handle_request(thread.request_queue.get())
    return sorted(thread.get_and_clear_failed_requests())


@pytest.mark.parametrize(
    "layout", [KVCacheLayout.LBNHC, KVCacheLayout.BLHNC], ids=lambda x: x.name
)
def test_worker_splits_off_class_groups_like_unsplit_objects(
    monkeypatch, caplog_vllm, layout
):
    """Against the same worker without the switch: the off-class group's
    objects become a size-class head and a tail, the on-class group's stay
    whole, and lookups and loads match byte for byte, also when either part
    of any object is gone."""
    with caplog_vllm.at_level("INFO"):
        worker, store, raw = _worker(monkeypatch, layout, split=True)
    control, control_store, control_raw = _worker(monkeypatch, layout, split=False)
    assert isinstance(worker.store, SizeClassSplitStore)
    assert not isinstance(control.store, SizeClassSplitStore)

    object_bytes = [worker._store_object_bytes(db) for db in worker.token_dbs]
    split_groups = [
        g for g, sizes in enumerate(object_bytes) if not all(map(is_size_class, sizes))
    ]
    if not split_groups:
        pytest.skip(f"{layout.name} puts every group on a size class")
    assert split_groups == [0]
    assert worker.store.split_prefixes == {
        key_prefix(worker.token_dbs[0].store_layout.key_for(0, BlockHash(b"")))
    }
    assert "stores each" in caplog_vllm.text

    _save(worker, raw)
    _save(control, control_raw)
    (target_bytes,) = object_bytes[0]
    head = split_head_bytes(target_bytes)
    assert len(store.objects) == len(control_store.objects) + len(_HASHES)
    for key, value in control_store.objects.items():
        if key_prefix(key) in worker.store.split_prefixes:
            parts = [store.objects[f"{key}@part:{i}"] for i in (0, 1)]
            assert [len(p) for p in parts] == [head, target_bytes - head]
            assert parts[0] + parts[1] == value
        else:
            assert store.objects[key] == value
    assert store.requested_bytes() == control_store.requested_bytes()
    assert store.reserved_bytes() < control_store.reserved_bytes()

    def lookup_and_load(w, s_raw):
        hit = w.lookup(num_tokens=_TOKENS + 1, block_hashes=_HASHES)
        failed = _load(w, s_raw, hit) if hit.hit_length else []
        return hit.hit_length, failed, s_raw.clone()

    hit, failed, loaded = lookup_and_load(worker, raw)
    assert hit == _TOKENS and not failed
    assert (hit, failed) == lookup_and_load(control, control_raw)[:2]
    assert torch.equal(loaded, control_raw)

    # Evict one object in the control and either part of it with the split.
    for key in list(control_store.objects):
        if key_prefix(key) not in worker.store.split_prefixes:
            continue
        for part in (0, 1):
            saved_split = dict(store.objects)
            saved_control = dict(control_store.objects)
            del store.objects[f"{key}@part:{part}"]
            del control_store.objects[key]
            got = lookup_and_load(worker, raw)
            want = lookup_and_load(control, control_raw)
            assert got[:2] == want[:2], (key, part)
            assert torch.equal(got[2], want[2])
            store.objects, control_store.objects = saved_split, saved_control


def test_worker_without_the_switch_keeps_the_store(monkeypatch):
    worker, store, raw = _worker(monkeypatch, KVCacheLayout.LBNHC, split=False)
    assert worker.store is store and worker._size_class_split is None
    _save(worker, raw)
    assert store.objects and not any("@part:" in key for key in store.objects)
