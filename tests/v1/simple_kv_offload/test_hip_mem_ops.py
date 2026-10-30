# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the HIP/ROCm paths in vllm.v1.simple_kv_offload.cuda_mem_ops.

Deliberately not skipped off ROCm: the version gate takes a plain version int
and the library loader is monkeypatched, so nothing here needs a HIP runtime
and the logic stays covered by the default (NVIDIA) CI that runs this
directory.
"""

import ctypes

import numpy as np
import pytest

from vllm.v1.simple_kv_offload import cuda_mem_ops
from vllm.v1.simple_kv_offload.cuda_mem_ops import (
    CopyGeometry,
    DriverMemcpyParams,
    MemcpyParams,
    _CUmemcpyAttributes,
    _load_hip_runtime,
    _num_attrs_for_hip_version,
    copy_blocks,
)


# hipMemcpyBatchAsync rejects numAttrs > 0 on ROCm 7.2.1-7.2.3 and accepts it
# on 7.13+. HIP encodes the version as major*10_000_000 + minor*100_000 + patch.
@pytest.mark.parametrize(
    ("version", "expected"),
    [
        (0, 0),  # unknown (hipRuntimeGetVersion failed) -> conservative 0
        (70_100_000, 0),  # ROCm 7.1
        (70_200_000, 0),  # ROCm 7.2.0
        (70_253_211, 0),  # ROCm 7.2.2 (patch 53211)
        (70_300_000, 0),  # ROCm 7.3
        (71_200_000, 0),  # ROCm 7.12 -- just below the 7.13 cutoff
        (71_300_000, 1),  # ROCm 7.13 -- first release accepting numAttrs > 0
        (71_326_193, 1),  # ROCm 7.13 nightly (patch 26193)
        (71_400_000, 1),  # ROCm 7.14
        (71_500_000, 1),  # ROCm 7.15
        (80_000_000, 1),  # ROCm 8.0
    ],
)
def test_num_attrs_for_hip_version(version: int, expected: int):
    assert _num_attrs_for_hip_version(version) == expected


@pytest.mark.parametrize(
    ("available", "expected"),
    [
        # Devel install: the unversioned symlink is present and preferred.
        ({"libamdhip64.so", "libamdhip64.so.7"}, "libamdhip64.so"),
        # Runtime-only / wheel-packaged ROCm: versioned soname only.
        ({"libamdhip64.so.7"}, "libamdhip64.so.7"),
        ({"libamdhip64.so.6"}, "libamdhip64.so.6"),
    ],
)
def test_load_hip_runtime_falls_back_to_versioned_soname(
    monkeypatch, available: set[str], expected: str
):
    loaded = []

    def fake_cdll(name, mode=0):
        if name not in available:
            raise OSError(f"{name}: cannot open shared object file")
        loaded.append(name)
        return name

    monkeypatch.setattr(cuda_mem_ops.ctypes, "CDLL", fake_cdll)
    assert _load_hip_runtime() == expected
    assert loaded == [expected]


def test_load_hip_runtime_reports_every_candidate_when_all_fail(monkeypatch):
    def fake_cdll(name, mode=0):
        raise OSError("cannot open shared object file")

    monkeypatch.setattr(cuda_mem_ops.ctypes, "CDLL", fake_cdll)
    with pytest.raises(OSError, match="could not load the HIP runtime") as exc:
        _load_hip_runtime()
    for name in ("libamdhip64.so", "libamdhip64.so.7", "libamdhip64.so.6"):
        assert name in str(exc.value)


def _fake_driver_params() -> DriverMemcpyParams:
    return DriverMemcpyParams(
        attrs=_CUmemcpyAttributes(),
        attrs_idx=ctypes.c_size_t(0),
        num_attrs=0,
        fail_idx=ctypes.c_size_t(0),
        stream_handle=0,
    )


def _fake_params() -> MemcpyParams:
    """Minimal params for driving copy_blocks() without a real device.

    Bases are 0 and the batch-memcpy function is monkeypatched, so no memory is
    ever dereferenced; only the descriptor bookkeeping (count/chunking) matters.
    """
    geometry = CopyGeometry(
        gpu_bases=np.zeros(1, dtype=np.int64),
        has_region=np.ones(1, dtype=bool),
        cpu_base=0,
        block_bytes=np.full(1, 512, dtype=np.int64),
    )
    return MemcpyParams(
        geometry=geometry, src_is_gpu=True, driver=_fake_driver_params()
    )


def _record_counts(monkeypatch) -> list[int]:
    """Patch the batch-memcpy call to record the descriptor count per call."""
    counts: list[int] = []

    def fake_fn(dst, src, sizes, count, *rest):
        counts.append(count)
        return 0

    monkeypatch.setattr(cuda_mem_ops, "_batch_memcpy", (fake_fn, 0))
    return counts


# One descriptor per block (a block's content is one contiguous range). ROCm
# chunks the descriptor batch at the cap (hipMemcpyBatchAsync faults above
# 8192/call); CUDA leaves the cap at 0 and issues a single call.
@pytest.mark.parametrize(
    ("max_desc", "num_blocks", "expected_counts"),
    [
        (8192, 5000, [5000]),  # below the cap -> single call
        (8192, 8192, [8192]),  # exactly at the cap -> single call
        (8192, 8193, [8192, 1]),  # one over -> spills into a second call
        (4096, 8235, [4096, 4096, 43]),  # bs=1 Kimi-scale block count
        (0, 8235, [8235]),  # CUDA uncapped -> single call
    ],
)
def test_copy_blocks_chunks_at_descriptor_cap(
    monkeypatch, max_desc, num_blocks, expected_counts
):
    counts = _record_counts(monkeypatch)
    monkeypatch.setattr(cuda_mem_ops, "_max_batch_descriptors", max_desc)

    ids = list(range(num_blocks))
    copy_blocks(ids, ids, [0] * num_blocks, _fake_params())

    assert counts == expected_counts
    assert sum(counts) == num_blocks


def test_copy_blocks_noop_on_empty(monkeypatch):
    counts = _record_counts(monkeypatch)
    monkeypatch.setattr(cuda_mem_ops, "_max_batch_descriptors", 8192)
    copy_blocks([], [], [], _fake_params())
    assert counts == []


# ---------------------------------------------------------------------------
# Block addresses
# ---------------------------------------------------------------------------


def _two_group_params(src_is_gpu: bool, has_region=(True, True)) -> tuple:
    """Two groups with different block bytes over one planner-built pool."""
    from tests.v1.kv_connector.unit.utils import make_two_group_large_page_config

    config = make_two_group_large_page_config()
    geometry = CopyGeometry(
        # Both groups are strided views of one backing allocation.
        gpu_bases=np.array([1 << 40, 1 << 40], dtype=np.int64),
        has_region=np.array(has_region),
        cpu_base=1 << 42,
        block_bytes=np.array(
            [
                config.group_geometry(g).block_bytes
                for g in range(len(config.kv_cache_groups))
            ],
            dtype=np.int64,
        ),
    )
    params = MemcpyParams(
        geometry=geometry, src_is_gpu=src_is_gpu, driver=_fake_driver_params()
    )
    return config, params


def _record_launches(monkeypatch) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Capture the (dst, src, sizes) descriptor arrays of each launch."""
    launched = []

    def fake_launch(dst_all, src_all, sz_all, params):
        launched.append((dst_all.copy(), src_all.copy(), sz_all.copy()))

    monkeypatch.setattr(cuda_mem_ops, "launch_batch", fake_launch)
    return launched


def test_copy_blocks_addresses(monkeypatch):
    """Block b of a group is at b * block_bytes on both sides, and a copy
    moves one block's content."""
    _, store_params = _two_group_params(src_is_gpu=True)
    _, load_params = _two_group_params(src_is_gpu=False)
    launched = _record_launches(monkeypatch)
    gpu_ids, cpu_ids, group_ids = [1, 5, 0, 3], [0, 2, 5, 1], [0, 0, 1, 1]

    copy_blocks(gpu_ids, cpu_ids, group_ids, store_params)
    copy_blocks(cpu_ids, gpu_ids, group_ids, load_params)

    (store_dst, store_src, store_sizes), (load_dst, load_src, _) = launched
    # Group 0 has 8704-byte blocks, group 1 has 4096-byte blocks (4 rows of
    # 1024 bytes).
    gpu_addr = [
        (1 << 40) + 8704,
        (1 << 40) + 5 * 8704,
        1 << 40,
        (1 << 40) + 3 * 4096,
    ]
    cpu_addr = [
        1 << 42,
        (1 << 42) + 2 * 8704,
        (1 << 42) + 5 * 4096,
        (1 << 42) + 4096,
    ]
    np.testing.assert_array_equal(store_src, gpu_addr)
    np.testing.assert_array_equal(store_dst, cpu_addr)
    np.testing.assert_array_equal(store_sizes, [8704, 8704, 4096, 4096])
    np.testing.assert_array_equal(load_src, cpu_addr)
    np.testing.assert_array_equal(load_dst, gpu_addr)


def test_copy_blocks_skips_groups_without_local_layers(monkeypatch):
    """Under PP a rank holds no layer of some groups; their blocks are not
    copied there."""
    _, params = _two_group_params(src_is_gpu=True, has_region=(False, True))
    launched = _record_launches(monkeypatch)

    copy_blocks([1], [0], [0], params)
    assert launched == []
    copy_blocks([1, 3], [0, 1], [0, 1], params)
    [(_, _, sizes)] = launched
    np.testing.assert_array_equal(sizes, [4096])
