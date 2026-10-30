# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for HF3FS KV Connector high-level components:
- TestHf3fsMockClient      : file-backed mock client I/O correctness
- TestHF3FSKVConnectorStats: metric collection, aggregation, serialisation
- TestHF3FSLargePageGeometry: per-group gather/scatter geometry over the
  strided layer views of the large-page allocator
- TestHF3FSLargePageGatherScatter: manager block ids map to kernel rows
- TestHF3FSLoadErrors      : failed loads surface (group_id, block_id) pairs
- TestHF3FSLargePageScheduler: per-group block ids through a real Scheduler
  with a two-group (full + sliding window) large-page config
"""

import os
from concurrent.futures import Future
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

import vllm.distributed.kv_transfer.kv_connector.v1.hf3fs.hf3fs_connector as hf3fs_mod
from tests.v1.kv_connector.unit.utils import (
    create_model_runner_output,
    create_request,
    create_scheduler,
    create_vllm_config,
    make_two_group_large_page_config,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.hf3fs.hf3fs_connector import (
    HF3FSKVConnector,
    HF3FSKVConnectorStats,
)
from vllm.distributed.kv_transfer.kv_connector.v1.hf3fs.utils import (
    gather_scatter_helper,
)
from vllm.distributed.kv_transfer.kv_connector.v1.hf3fs.utils.hf3fs_mock_client import (
    Hf3fsClient as MockHf3fsClient,
)
from vllm.v1.kv_cache_interface import KVCacheConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def hf3fs_stats():
    """Fresh HF3FSKVConnectorStats instance."""
    return HF3FSKVConnectorStats()


def _make_cuda_event():
    """Return a real CUDA event when available, otherwise a MagicMock."""
    if torch.cuda.is_available():
        return torch.cuda.Event()
    return MagicMock()


# ===========================================================================
# TestHf3fsMockClient
# ===========================================================================


class TestHf3fsMockClient:
    """Tests for hf3fs_mock_client.Hf3fsClient (file-backend mock)."""

    def test_init_creates_file(self, tmp_path):
        """Initializing the client should create the backing file."""
        path = str(tmp_path / "test_file")
        client = MockHf3fsClient(path=path, size=4096, bytes_per_page=512, entries=4)
        assert os.path.exists(path), "Backing file should be created on init"
        assert os.path.getsize(path) == 4096
        client.close()

    @pytest.mark.parametrize(
        "dtype, bytes_per_page",
        [
            (torch.float32, 512),
            (torch.float16, 256),
            (torch.bfloat16, 256),
        ],
        ids=["float32", "float16", "bfloat16"],
    )
    def test_batch_write_and_read_dtype(self, tmp_path, dtype, bytes_per_page):
        """Write a tensor of the given dtype and verify round-trip correctness."""
        path = str(tmp_path / f"rw_{dtype}")
        client = MockHf3fsClient(
            path=path, size=bytes_per_page * 8, bytes_per_page=bytes_per_page, entries=4
        )
        elem_size = torch.tensor([], dtype=dtype).element_size()
        numel = bytes_per_page // elem_size
        tensor_write = torch.arange(numel, dtype=dtype)
        event = _make_cuda_event()

        results = client.batch_write([0], [tensor_write], event)
        assert results == [bytes_per_page], f"Write should succeed, got {results}"

        tensor_read = torch.zeros(numel, dtype=dtype)
        results = client.batch_read([0], [tensor_read])
        assert results == [bytes_per_page], f"Read should succeed, got {results}"
        assert torch.equal(tensor_write, tensor_read), (
            "Read tensor should match written tensor"
        )
        client.close()

    def test_batch_read_empty_file_returns_error(self, tmp_path):
        """Reading out-of-bounds offset should return -1."""
        bytes_per_page = 128
        size = bytes_per_page * 4
        path = str(tmp_path / "empty_read")
        client = MockHf3fsClient(
            path=path, size=size, bytes_per_page=bytes_per_page, entries=4
        )
        numel = bytes_per_page // 4
        tensor_read = torch.zeros(numel, dtype=torch.float32)
        results = client.batch_read([size], [tensor_read])  # offset == size => OOB
        assert results[0] == -1, "Out-of-bounds read should return -1"
        client.close()

    def test_batch_write_out_of_bounds_returns_error(self, tmp_path):
        """Writing at an offset beyond file size should return -1."""
        bytes_per_page = 128
        size = bytes_per_page * 4
        path = str(tmp_path / "oob_write")
        client = MockHf3fsClient(
            path=path, size=size, bytes_per_page=bytes_per_page, entries=4
        )
        numel = bytes_per_page // 4
        tensor = torch.ones(numel, dtype=torch.float32)
        event = _make_cuda_event()
        results = client.batch_write([size], [tensor], event)  # OOB offset
        assert results[0] == -1, "Out-of-bounds write should return -1"
        client.close()

    def test_multiple_tensors_rw(self, tmp_path):
        """Write multiple tensors at different offsets, then read all back."""
        bytes_per_page = 128
        n = 4
        path = str(tmp_path / "multi_rw")
        client = MockHf3fsClient(
            path=path,
            size=bytes_per_page * n * 2,
            bytes_per_page=bytes_per_page,
            entries=8,
        )
        tensors_write = [
            torch.full((bytes_per_page // 4,), float(i), dtype=torch.float32)
            for i in range(n)
        ]
        offsets = [i * bytes_per_page for i in range(n)]
        event = _make_cuda_event()

        results = client.batch_write(offsets, tensors_write, event)
        assert all(r == bytes_per_page for r in results)

        tensors_read = [
            torch.zeros(bytes_per_page // 4, dtype=torch.float32) for _ in range(n)
        ]
        results = client.batch_read(offsets, tensors_read)
        assert all(r == bytes_per_page for r in results)

        for i, (tw, tr) in enumerate(zip(tensors_write, tensors_read)):
            assert torch.allclose(tw, tr), f"Tensor {i} mismatch after round-trip"
        client.close()

    def test_flush_and_close_no_error(self, tmp_path):
        """flush() and close() should not raise exceptions."""
        path = str(tmp_path / "flush_close")
        client = MockHf3fsClient(path=path, size=1024, bytes_per_page=128, entries=4)
        client.flush()
        client.close()


# ===========================================================================
# TestHF3FSKVConnectorStats
# ===========================================================================


class TestHF3FSKVConnectorStats:
    """Tests for HF3FSKVConnectorStats metric collection and aggregation."""

    def test_initial_is_empty(self, hf3fs_stats):
        """Fresh stats object should report is_empty() == True."""
        assert hf3fs_stats.is_empty() is True

    @pytest.mark.parametrize(
        "task_type, duration_key",
        [
            ("Saved", "save_duration"),
            ("Loaded", "load_duration"),
        ],
        ids=["save", "load"],
    )
    def test_record_success_duration(self, hf3fs_stats, task_type, duration_key):
        """Recording a successful task should update duration list and total count."""
        hf3fs_stats.record_success_task_duration(task_type, 0.5)
        assert not hf3fs_stats.is_empty()
        assert len(hf3fs_stats.data[duration_key]) == 1
        assert hf3fs_stats.data[duration_key][0] == pytest.approx(0.5)
        assert hf3fs_stats.data["num_transfer_task"] == 1

    @pytest.mark.parametrize(
        "task_type, failed_key",
        [
            ("Saved", "num_failed_save"),
            ("Loaded", "num_failed_load"),
        ],
        ids=["save", "load"],
    )
    def test_record_failed_task(self, hf3fs_stats, task_type, failed_key):
        """Recording a failed task should increment the corresponding counter."""
        hf3fs_stats.record_failed_task_count(task_type)
        assert hf3fs_stats.data[failed_key] == 1
        assert hf3fs_stats.data["num_transfer_task"] == 1

    def test_aggregate_two_stats(self):
        """aggregate() should merge save/load duration lists and sum counters."""
        stats1 = HF3FSKVConnectorStats()
        stats1.record_success_task_duration("Saved", 0.1)
        stats1.record_success_task_duration("Loaded", 0.2)

        stats2 = HF3FSKVConnectorStats()
        stats2.record_success_task_duration("Saved", 0.3)
        stats2.record_failed_task_count("Loaded")

        stats1.aggregate(stats2)
        assert stats1.data["save_duration"] == pytest.approx([0.1, 0.3])
        assert stats1.data["load_duration"] == pytest.approx([0.2])
        assert stats1.data["num_failed_load"] == 1
        assert stats1.data["num_transfer_task"] == 4

    def test_reduce_with_data(self):
        """reduce() computes correct averages when data is present."""
        stats = HF3FSKVConnectorStats()
        stats.record_success_task_duration("Saved", 1.0)
        stats.record_success_task_duration("Saved", 3.0)
        result = stats.reduce()
        assert result["Num save task success"] == pytest.approx(2.0, rel=0.01)
        assert result["Num save task failed"] == pytest.approx(0.0, rel=0.01)
        assert result["Avg save duration (ms)"] == pytest.approx(2000.0, rel=0.01)
        # Reduced values must be plain Python scalars so CLI logging renders
        # them without numpy reprs (eg np.float64(...)).
        assert all(not isinstance(v, np.generic) for v in result.values())

    def test_clone_and_reset(self, hf3fs_stats):
        """clone_and_reset() returns a copy with data and resets the original."""
        hf3fs_stats.record_success_task_duration("Saved", 0.7)
        hf3fs_stats.record_success_task_duration("Loaded", 0.4)

        clone = hf3fs_stats.clone_and_reset()
        assert clone.data["num_transfer_task"] == 2
        assert hf3fs_stats.is_empty()


# ===========================================================================
# Large-page layout: two-group geometry (full + sliding window)
# ===========================================================================

# Shared planner-built two-group config. Group 0 is full attention (layer
# "fa.0", one 8704 B row per block); group 1 is sliding window (layer
# "swa.0", four 1024 B rows per block). B = 16.
LP_BLOCK_SIZE = 16


def _make_two_group_config() -> KVCacheConfig:
    # 5 pages hold 20 full-attention blocks: the scheduler tests run two
    # requests needing up to 8 distinct full-attention blocks concurrently.
    return make_two_group_large_page_config(num_pages=5)


def _make_layer_views(config: KVCacheConfig) -> dict[str, torch.Tensor]:
    """Strided per-layer [K, H, N, C] views over one shared backing buffer.

    Mirrors what the worker binds under large_page: row stride = the group's
    row_bytes, layer l at element offset l * (row_bytes / num_layers).
    """
    buf = torch.zeros(
        config.num_pages * config.large_page_bytes // 2, dtype=torch.float16
    )  # fp16: 2 B/elem
    kv_caches: dict[str, torch.Tensor] = {}
    for group in config.kv_cache_groups:
        spec = group.kv_cache_spec
        kernel_block_size = spec.block_size // group.kernel_rows_per_block
        row_elems = group.row_bytes // 2
        layer_row_elems = row_elems // len(group.layer_names)
        num_rows = config.num_pages * config.large_page_bytes // group.row_bytes
        for layer_idx, name in enumerate(group.layer_names):
            kv_caches[name] = buf.as_strided(
                (num_rows, spec.num_kv_heads, kernel_block_size, spec.head_size),
                (row_elems, kernel_block_size * spec.head_size, spec.head_size, 1),
                storage_offset=layer_idx * layer_row_elems,
            )
    return kv_caches


def _make_worker_connector(config: KVCacheConfig) -> HF3FSKVConnector:
    vllm_config = create_vllm_config(
        kv_connector="HF3FSKVConnector",
        kv_role="kv_both",
        block_size=LP_BLOCK_SIZE,
    )
    return HF3FSKVConnector(vllm_config, KVConnectorRole.WORKER, config)


class _StubMetadataClient:
    """Stand-in for the external HF3FS metadata server."""

    exists: list[bool] = []

    def __init__(self, *args, **kwargs):
        pass

    def initialize(self, rank: int, num_pages: int = 0, role: str = "worker"):
        pass

    def batch_key_exists(self, keys: list[str]) -> list[bool]:
        return list(self.exists[: len(keys)]) + [False] * max(
            0, len(keys) - len(self.exists)
        )

    def allocate_pages_for_keys(self, rank: int, keys: list) -> list:
        return [(key, i) for i, (key, _) in enumerate(keys)]

    def confirm_write_for_keys(self, rank: int, confirmations: list, pages=None):
        pass

    def get_key_locations(self, rank: int, keys: list[str]) -> list:
        return [None] * len(keys)


@pytest.fixture
def stub_metadata_client(monkeypatch):
    monkeypatch.setattr(hf3fs_mod, "Hf3fsMetadataClient", _StubMetadataClient)
    _StubMetadataClient.exists = []
    return _StubMetadataClient


class TestHF3FSLargePageGeometry:
    """Worker-side registration over per-group strided layer views."""

    def test_per_group_geometry(self):
        config = _make_two_group_config()
        kv_caches = _make_layer_views(config)
        connector = _make_worker_connector(config)
        # register_kv_caches() also creates CUDA streams; exercise the
        # geometry derivation directly on CPU.
        connector._setup_kv_cache_config(kv_caches)

        # One storage page holds one 16-token block of every group.
        # Group 0: 1 layer * 4 heads * 16 tokens * 34 dim = 2176 elems.
        # Group 1: 1 layer * 4 heads * 16 tokens * 16 dim = 1024 elems.
        assert connector._bytes_per_page == (2176 + 1024) * 2
        assert connector._shape_per_page == [2176 + 1024]

        geom0, geom1 = connector._group_geometry
        assert geom0 is not None and geom1 is not None
        assert geom0.segment_offset == 0 and geom0.segment_numel == 2176
        assert geom1.segment_offset == 2176 and geom1.segment_numel == 1024
        # Per-layer pointers are the distinct view bases, strides carry the
        # group's row stride.
        assert geom0.ptrs.tolist() == [kv_caches["fa.0"].data_ptr()]
        assert geom1.ptrs.tolist() == [kv_caches["swa.0"].data_ptr()]
        assert geom0.strides == kv_caches["fa.0"].stride()
        assert geom1.strides == kv_caches["swa.0"].stride()

    def test_pp_group_without_local_layers_is_skipped(self):
        config = _make_two_group_config()
        kv_caches = _make_layer_views(config)
        connector = _make_worker_connector(config)
        connector._setup_kv_cache_config(
            {n: t for n, t in kv_caches.items() if n.startswith("fa")}
        )
        geom0, geom1 = connector._group_geometry
        assert geom0 is not None and geom1 is None
        assert connector._bytes_per_page == 2176 * 2

    def test_non_uniform_group_views_rejected(self):
        config = _make_two_group_config()
        kv_caches = _make_layer_views(config)
        # Uniformity is a per-layer property within a group; the shared
        # fixture has one layer per group, so give group 0 a second layer
        # tiled into the same 8704 B rows (2 * 2176 elems = 4352 = row).
        first = kv_caches["fa.0"]
        config.kv_cache_groups[0].layer_names = ["fa.0", "fa.1"]
        kv_caches["fa.1"] = first.as_strided(first.shape, first.stride(), 2176)
        # Corrupt the second layer with a different shape.
        kv_caches["fa.1"] = kv_caches["fa.1"].as_strided(
            (4, 4, 16, 34), first.stride(), 2176
        )
        connector = _make_worker_connector(config)
        with pytest.raises(AssertionError, match="uniform"):
            connector._setup_kv_cache_config(kv_caches)

    def test_row_stride_mismatch_rejected(self):
        config = _make_two_group_config()
        kv_caches = _make_layer_views(config)
        # Corrupt the group 0 row stride: 4096-elem rows instead of 4352.
        kv_caches["fa.0"] = kv_caches["fa.0"].as_strided(
            kv_caches["fa.0"].shape, (4096, 16 * 34, 34, 1)
        )
        connector = _make_worker_connector(config)
        with pytest.raises(ValueError, match="stride rows"):
            connector._setup_kv_cache_config(kv_caches)


class TestHF3FSLargePageGatherScatter:
    """Manager block b maps to tokens [b * B, (b + 1) * B) of each group."""

    def _capture(self, monkeypatch):
        calls = []

        def fake_gather(ptrs, total_tokens, dst, token_indices, *args):
            calls.append((dst, list(token_indices)))

        def fake_scatter(ptrs, total_tokens, src, token_indices, *args):
            calls.append((src, list(token_indices)))

        monkeypatch.setattr(gather_scatter_helper, "gather_kv_caches", fake_gather)
        monkeypatch.setattr(gather_scatter_helper, "scatter_kv_caches", fake_scatter)
        return calls

    def test_token_indices_follow_block_id(self, monkeypatch):
        config = _make_two_group_config()
        connector = _make_worker_connector(config)
        connector._setup_kv_cache_config(_make_layer_views(config))
        calls = self._capture(monkeypatch)

        buffer = torch.zeros(connector._shape_per_page, dtype=connector._dtype)
        connector._gather_or_scatter_kv_caches([[3, 5]], [buffer], "gather")

        assert len(calls) == 2
        # Group 0: 1 row of 16 tokens per block; block 3 is row 3 -> tokens
        # [48, 64).
        assert calls[0][1] == list(range(48, 64))
        # Group 1: 4 rows of 4 tokens per block; block 5 is rows 20-23 ->
        # tokens [80, 96).
        assert calls[1][1] == list(range(80, 96))
        for group_id, (segment, _) in enumerate(calls):
            geom = connector._group_geometry[group_id]
            assert geom is not None
            # The segment is the group's slice of the page buffer.
            assert segment.data_ptr() == (
                buffer.data_ptr() + geom.segment_offset * buffer.element_size()
            )
            assert segment.numel() == geom.segment_numel

    def test_scatter_skips_null_blocks(self, monkeypatch):
        config = _make_two_group_config()
        connector = _make_worker_connector(config)
        connector._setup_kv_cache_config(_make_layer_views(config))
        calls = self._capture(monkeypatch)

        buffer = torch.zeros(connector._shape_per_page, dtype=connector._dtype)
        # The sliding-window group holds no block for this token position.
        connector._gather_or_scatter_kv_caches([[3, 0]], [buffer], "scatter")
        assert len(calls) == 1

    def test_gather_rejects_null_blocks(self, monkeypatch):
        config = _make_two_group_config()
        connector = _make_worker_connector(config)
        connector._setup_kv_cache_config(_make_layer_views(config))
        self._capture(monkeypatch)

        buffer = torch.zeros(connector._shape_per_page, dtype=connector._dtype)
        with pytest.raises(AssertionError):
            connector._gather_or_scatter_kv_caches([[3, 0]], [buffer], "gather")


class TestHF3FSLoadErrors:
    """A failed load must surface its (group_id, block_id) pairs so the
    scheduler recomputes instead of resuming onto never-written blocks."""

    def _make_async_manager(self, connector, monkeypatch):
        # Streams/events/allocators need CUDA; the "blocks not found" failure
        # path under test never touches them.
        monkeypatch.setattr(
            hf3fs_mod.AsyncOperationManager, "_init_cuda_resources", lambda self: None
        )
        monkeypatch.setattr(
            hf3fs_mod.AsyncOperationManager, "_init_worker_threads", lambda self: None
        )
        connector._rank = 0
        connector._metadata_client = _StubMetadataClient()
        # Bound by register_kv_caches() in production.
        connector._async_manager = hf3fs_mod.AsyncOperationManager(connector)
        return connector._async_manager

    def test_failed_load_surfaces_invalid_block_ids(self, monkeypatch):
        config = _make_two_group_config()
        connector = _make_worker_connector(config)
        connector._setup_kv_cache_config(_make_layer_views(config))
        manager = self._make_async_manager(connector, monkeypatch)

        future: Future = Future()
        manager._load_futures["req-fail"] = future
        # Token block rows [group 0 block, group 1 block]; id 0 is the null
        # block, which the group intentionally does not hold.
        manager._handle_load_task(("req-fail", [[3, 5], [4, 0]], ["h1", "h2"], future))

        assert future.result() is False
        assert connector.get_block_ids_with_load_errors() == {(0, 3), (0, 4), (1, 5)}
        assert connector.get_block_ids_with_load_errors() == set()
        # The request is still reported finished receiving, per the
        # KVConnectorBase_V1 contract; the invalid blocks above drive
        # recompute.
        _, completed_loads = manager.get_finished_operations(set())
        assert completed_loads == {"req-fail"}


class TestHF3FSLargePageScheduler:
    """Per-group block ids through a real Scheduler (full + sliding window)."""

    def _make_scheduler(self, config: KVCacheConfig):
        vllm_config = create_vllm_config(
            kv_connector="HF3FSKVConnector",
            kv_role="kv_both",
            block_size=LP_BLOCK_SIZE,
            max_num_batched_tokens=256,
        )
        return create_scheduler(
            vllm_config, num_blocks=config.num_pages, kv_cache_config=config
        )

    def test_save_metadata_is_per_group(self, stub_metadata_client):
        scheduler = self._make_scheduler(_make_two_group_config())
        request = create_request(
            request_id=1, num_tokens=3 * LP_BLOCK_SIZE, block_size=LP_BLOCK_SIZE
        )
        scheduler.add_request(request)
        output = scheduler.schedule()
        metadata = output.kv_connector_metadata
        assert len(metadata.requests) == 1
        request_meta = metadata.requests[0]

        assert len(request_meta.block_ids) == 2
        full_ids, sw_ids = request_meta.block_ids
        # Both lists are index-aligned by token block; ids are group-local.
        assert len(full_ids) == len(sw_ids) == 3
        assert all(bid != 0 for bid in full_ids + sw_ids)
        manager_ids = scheduler.kv_cache_manager.get_block_ids("id-1")
        assert [full_ids, sw_ids] == [list(ids) for ids in manager_ids]

    def test_sliding_window_nulls_are_preserved(self, stub_metadata_client):
        scheduler = self._make_scheduler(_make_two_group_config())
        # A request with a cached prefix longer than the sliding window:
        # the SW group pads the out-of-window prefix positions with nulls.
        first = create_request(
            request_id=3,
            num_tokens=6 * LP_BLOCK_SIZE,
            common_prefix_len=6 * LP_BLOCK_SIZE,
            block_size=LP_BLOCK_SIZE,
        )
        scheduler.add_request(first)
        output = scheduler.schedule()
        scheduler.update_from_output(
            output, create_model_runner_output(list(scheduler.running))
        )

        second = create_request(
            request_id=4,
            num_tokens=8 * LP_BLOCK_SIZE,
            common_prefix_len=6 * LP_BLOCK_SIZE,
            block_size=LP_BLOCK_SIZE,
        )
        scheduler.add_request(second)
        output = scheduler.schedule()
        request_meta = next(
            r for r in output.kv_connector_metadata.requests if r.request_id == "id-4"
        )

        full_ids, sw_ids = request_meta.block_ids
        assert len(full_ids) == len(sw_ids) == 8
        # The SW group's window is one block, so only 5 of the 6 cached
        # prefix blocks count as computed; those 5 are not re-saved.
        assert request_meta.save_block_op is not None
        assert request_meta.save_block_op.skip_leading_blocks == 5
        # The full group holds real blocks everywhere; the SW group pads the
        # 4 out-of-window computed positions with its null block (id 0).
        assert all(bid != 0 for bid in full_ids)
        assert sw_ids[:4] == [0, 0, 0, 0]
        assert all(bid != 0 for bid in sw_ids[4:])

    def test_async_load_fetch_ids_are_per_group(self, stub_metadata_client):
        _StubMetadataClient.exists = [True, True]
        scheduler = self._make_scheduler(_make_two_group_config())
        request = create_request(
            request_id=2, num_tokens=4 * LP_BLOCK_SIZE, block_size=LP_BLOCK_SIZE
        )
        scheduler.add_request(request)
        output = scheduler.schedule()
        metadata = output.kv_connector_metadata
        assert len(metadata.requests) == 1
        request_meta = metadata.requests[0]

        assert request_meta.load_block_op is not None
        assert request_meta.load_block_op.num_blocks_to_load == 2
        # The fetch targets are the first two blocks of every group.
        manager_ids = scheduler.kv_cache_manager.get_block_ids("id-2")
        assert request_meta.block_ids == [list(ids[:2]) for ids in manager_ids]
