# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side unit tests for SimpleCPUOffloadConnector.

Covers the GPU->CPU store cross-stream synchronization: the store copy must be
ordered after the compute stream that writes the KV blocks, otherwise it can
read partially written / stale blocks and silently corrupt the CPU cache.
"""

from __future__ import annotations

import time
from contextlib import nullcontext
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import torch.distributed as dist

from vllm.platforms import current_platform

if not current_platform.is_cuda_alike():
    pytest.skip("Requires CUDA or ROCm", allow_module_level=True)

from tests.v1.kv_connector.unit.test_kv_connector_lifecycle import (
    _make_empty_scheduler_output,
)
from vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload_connector import (
    SimpleCPUOffloadConnector,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.simple_kv_offload.copy_backend import DmaCopyBackend
from vllm.v1.simple_kv_offload.cuda_mem_ops import (
    CU_MEMCPY_SRC_ACCESS_ORDER_ANY,
    CU_MEMCPY_SRC_ACCESS_ORDER_STREAM,
    CopyGeometry,
    build_params,
)
from vllm.v1.simple_kv_offload.disk_backend import DiskBackend, plan_disk_layout
from vllm.v1.simple_kv_offload.metadata import (
    SimpleCPUOffloadMetadata,
    SimpleCPUOffloadWorkerMetadata,
)
from vllm.v1.simple_kv_offload.sizing import sync_num_offload_blocks_across_workers
from vllm.v1.simple_kv_offload.worker import SimpleCPUOffloadWorker
from vllm.v1.worker.gpu.kv_connector import ActiveKVConnector
from vllm.v1.worker.kv_connector_model_runner_mixin import KVConnectorModelRunnerMixin

NUM_BLOCKS = 64
BLOCK_BYTES = 4096
ITERS = 30
# Keep the compute stream busy so the KV write lands late; this makes the
# store-vs-compute race deterministic instead of timing-dependent.
SLEEP_CYCLES = 50_000_000


def _make_worker(
    kv_cache_config: KVCacheConfig | None, cpu_capacity_bytes: int
) -> SimpleCPUOffloadWorker:
    return SimpleCPUOffloadWorker(None, kv_cache_config, cpu_capacity_bytes)


def _make_geometry(gpu: torch.Tensor, cpu: torch.Tensor) -> CopyGeometry:
    """Single-group geometry: 4096-byte blocks."""
    return CopyGeometry(
        gpu_bases=np.array([gpu.data_ptr()], dtype=np.int64),
        has_region=np.ones(1, dtype=bool),
        cpu_base=cpu.data_ptr(),
        block_bytes=np.full(1, BLOCK_BYTES, dtype=np.int64),
    )


def _make_backend() -> tuple[DmaCopyBackend, torch.Tensor, torch.Tensor]:
    gpu = torch.zeros((NUM_BLOCKS, BLOCK_BYTES), dtype=torch.int8, device="cuda")
    cpu = torch.zeros(
        NUM_BLOCKS * BLOCK_BYTES, dtype=torch.int8, device="cpu", pin_memory=True
    )
    low_pri, _ = torch.cuda.Stream.priority_range()
    backend = DmaCopyBackend()
    backend.init(
        _make_geometry(gpu, cpu),
        gpu.device,
        torch.cuda.Stream(priority=low_pri),
        torch.cuda.Stream(priority=low_pri),
    )
    return backend, gpu, cpu.view(NUM_BLOCKS, BLOCK_BYTES)


@pytest.mark.parametrize("use_v2", [False, True])
def test_no_forward_step_completes_cpu_store(monkeypatch, use_v2):
    """A finished request's final store must drain without another model step."""
    backend, gpu, cpu = _make_backend()
    worker = _make_worker(None, 0)
    worker._backend = backend
    connector = SimpleCPUOffloadConnector.__new__(SimpleCPUOffloadConnector)
    connector.worker_handler = worker
    output = _make_empty_scheduler_output()
    output.kv_connector_metadata = SimpleCPUOffloadMetadata(
        store_event=0,
        store_gpu_blocks=[0],
        store_cpu_blocks=[0],
        store_group_ids=[0],
    )
    module = "vllm.v1.worker.kv_connector_model_runner_mixin"
    monkeypatch.setattr(f"{module}.get_kv_transfer_group", lambda: connector)
    monkeypatch.setattr(f"{module}.set_forward_context", lambda *a: nullcontext())
    monkeypatch.setattr(f"{module}.get_forward_context", lambda: None)
    try:
        gpu.fill_(91)
        if use_v2:
            active = ActiveKVConnector.__new__(ActiveKVConnector)
            active.kv_connector = connector
            active._disabled = False
            active._pending_load_kwargs = None
            module = "vllm.v1.worker.gpu.kv_connector"
            monkeypatch.setattr(f"{module}.is_forward_context_available", lambda: True)
            monkeypatch.setattr(f"{module}.get_forward_context", lambda: None)
            result = active.no_forward(output)
        else:
            result = KVConnectorModelRunnerMixin.kv_connector_no_forward(output, None)
        completion = (
            result.kv_connector_output.kv_connector_worker_meta
            if result.kv_connector_output is not None
            else None
        )
        deadline = time.monotonic() + 5
        while completion is None and time.monotonic() < deadline:
            connector.get_finished(set())
            completion = connector.build_connector_worker_meta()
            time.sleep(0.001)
        assert completion is not None, "CPU store never completed on the empty step"
        assert isinstance(completion, SimpleCPUOffloadWorkerMetadata)
        assert completion.completed_store_events == {0: 1}
        assert torch.equal(cpu[0], gpu[0].cpu())
    finally:
        backend.shutdown()


def test_deferred_store_waits_for_draft_forward(monkeypatch):
    """Target completion must not submit the store before draft finalization."""
    backend, gpu, cpu = _make_backend()
    worker = _make_worker(None, 0)
    worker._backend = backend
    connector = SimpleCPUOffloadConnector.__new__(SimpleCPUOffloadConnector)
    connector.worker_handler = worker
    output = _make_empty_scheduler_output()
    output.kv_connector_metadata = SimpleCPUOffloadMetadata(
        store_event=0,
        store_gpu_blocks=[0],
        store_cpu_blocks=[0],
        store_group_ids=[0],
    )
    module = "vllm.v1.worker.kv_connector_model_runner_mixin"
    monkeypatch.setattr(f"{module}.get_kv_transfer_group", lambda: connector)
    monkeypatch.setattr(f"{module}.has_kv_transfer_group", lambda: True)
    monkeypatch.setattr(f"{module}.get_forward_context", lambda: None)
    launch = MagicMock(wraps=backend.launch_copy)
    monkeypatch.setattr(backend, "launch_copy", launch)
    try:
        with KVConnectorModelRunnerMixin._get_kv_connector_output(
            output, defer_finalize=True
        ):
            gpu.fill_(17)
        launch.assert_not_called()
        gpu.fill_(91)
        KVConnectorModelRunnerMixin.finalize_kv_connector()
        assert launch.call_count == 1
        completion = None
        deadline = time.monotonic() + 5
        while completion is None and time.monotonic() < deadline:
            connector.get_finished(set())
            completion = connector.build_connector_worker_meta()
            time.sleep(0.001)
        assert completion is not None
        assert completion.completed_store_events == {0: 1}
        assert torch.equal(cpu[0], gpu[0].cpu())
        assert launch.call_count == 1
    finally:
        backend.shutdown()


def _drive_store(
    backend: DmaCopyBackend,
    gpu: torch.Tensor,
    cpu: torch.Tensor,
    *,
    with_barrier: bool,
) -> int:
    """Run ITERS store cycles; return how many landed corrupted in the CPU pool.

    Each cycle writes a unique value on a compute stream (after a deliberate
    delay) and then issues the GPU->CPU store. The store is issued *after* the
    write in host program order, mirroring the connector's deferred-store
    assumption. Only the compute-done event creates a real device-side
    happens-before edge.
    """
    block_ids = list(range(gpu.shape[0]))
    compute_stream = torch.cuda.Stream()
    corrupt = 0
    for it in range(ITERS):
        val = (it % 126) + 1  # 1..126; distinct from the zero-initialized pool
        with torch.cuda.stream(compute_stream):
            torch.cuda._sleep(SLEEP_CYCLES)
            gpu.fill_(val)

        wait_event = None
        if with_barrier:
            wait_event = torch.Event()
            wait_event.record(compute_stream)

        store_events: list[tuple[int, torch.Event]] = []
        backend.launch_copy(
            block_ids,
            block_ids,
            [0] * len(block_ids),
            is_store=True,
            event_idx=it,
            events_list=store_events,
            wait_event=wait_event,
        )

        deadline = time.time() + 10.0
        while not store_events and time.time() < deadline:
            time.sleep(0.0005)
        assert store_events, "background copy was never enqueued"
        store_events[0][1].synchronize()

        if int((cpu[:, 0].to(torch.int32) != val).sum().item()):
            corrupt += 1

    # Drain the compute stream before returning: in the no-barrier control
    # phase the store never waits on compute, so the host loop runs far ahead
    # and leaves a backlog of sleep+fill kernels in flight. Without this, the
    # leftover control-phase fills race the barrier phase's fill->copy window
    # on the shared gpu tensor and flakily corrupt one iteration.
    compute_stream.synchronize()
    return corrupt


def test_store_orders_after_compute_write():
    """The store must wait for the compute event; without it, it races.

    Asserts both directions so the test is self-validating: the no-barrier
    control must actually corrupt (proving the race window is exercised), and
    the fixed path with the compute-done event must be clean.
    """
    backend, gpu, cpu = _make_backend()
    try:
        control = _drive_store(backend, gpu, cpu, with_barrier=False)
        fixed = _drive_store(backend, gpu, cpu, with_barrier=True)
    finally:
        backend.shutdown()

    assert control > 0, (
        "no-barrier store did not race the compute write; the test no longer "
        "exercises the hazard it is meant to guard"
    )
    assert fixed == 0, f"store raced compute even with the barrier: {fixed} corrupt"


class _RecordingBackend:
    """Captures launch_copy calls without touching the GPU."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def launch_copy(
        self,
        src_blocks,
        dst_blocks,
        group_ids,
        is_store,
        event_idx,
        events_list,
        wait_event=None,
    ) -> None:
        self.calls.append({"is_store": is_store, "wait_event": wait_event})


def test_transfer_hooks_pass_wait_event_for_store_only():
    """wait_for_save gates stores on a compute-done event; start_load_kv does not."""
    worker = _make_worker(
        kv_cache_config=None,
        cpu_capacity_bytes=0,
    )
    recording = _RecordingBackend()
    worker._backend = recording
    worker._connector_metadata = SimpleCPUOffloadMetadata(
        load_event=0,
        load_gpu_blocks=[0],
        load_cpu_blocks=[0],
        load_group_ids=[0],
        store_event=1,
        store_gpu_blocks=[1],
        store_cpu_blocks=[1],
        store_group_ids=[0],
    )

    worker.start_load_kv()
    worker.wait_for_save()

    store_calls = [c for c in recording.calls if c["is_store"]]
    load_calls = [c for c in recording.calls if not c["is_store"]]
    assert len(store_calls) == 1
    assert len(load_calls) == 1
    assert isinstance(store_calls[0]["wait_event"], torch.Event)
    assert load_calls[0]["wait_event"] is None


def test_disk_backend_init_lays_out_pages(tmp_path, monkeypatch):
    """The disk file holds num_pages pages of the (padded) slot layout."""
    monkeypatch.setattr(
        "vllm.v1.simple_kv_offload.disk_backend.pin_tensor", lambda t: None
    )
    gpu = torch.zeros((NUM_BLOCKS, BLOCK_BYTES), dtype=torch.int8, device="cuda")
    cpu = torch.zeros(NUM_BLOCKS * BLOCK_BYTES, dtype=torch.int8, device="cpu")
    backend = DiskBackend()
    disk_path = tmp_path / "kv-offload.bin"
    backend.init(
        _make_geometry(gpu, cpu),
        gpu.device,
        torch.cuda.Stream(),
        torch.cuda.Stream(),
        str(disk_path),
        plan_disk_layout(BLOCK_BYTES, [BLOCK_BYTES], use_page_cache=False),
        NUM_BLOCKS,
    )
    try:
        assert disk_path.stat().st_size == NUM_BLOCKS * BLOCK_BYTES
    finally:
        backend.shutdown()


def test_build_params_src_access_order():
    """build_params defaults to ANY and honors an explicit STREAM override."""
    geometry = CopyGeometry(
        gpu_bases=np.zeros(1, dtype=np.int64),
        has_region=np.ones(1, dtype=bool),
        cpu_base=0,
        block_bytes=np.full(1, 64, dtype=np.int64),
    )
    stream = torch.cuda.Stream()

    default = build_params(geometry, src_is_gpu=True, stream=stream)
    assert default.driver.attrs.srcAccessOrder == CU_MEMCPY_SRC_ACCESS_ORDER_ANY

    ordered = build_params(
        geometry,
        src_is_gpu=True,
        stream=stream,
        src_access_order=CU_MEMCPY_SRC_ACCESS_ORDER_STREAM,
    )
    assert ordered.driver.attrs.srcAccessOrder == CU_MEMCPY_SRC_ACCESS_ORDER_STREAM


def _make_lp_config_and_caches(
    num_layers: int, num_blocks: int
) -> tuple[KVCacheConfig, dict[str, torch.Tensor]]:
    """One group of layers as strided views over one backing allocation."""
    row_bytes = num_layers * BLOCK_BYTES
    names = [f"layer.{i}" for i in range(num_layers)]
    raw = torch.zeros(num_blocks * row_bytes, dtype=torch.int8, device="cuda")
    caches = {
        name: torch.as_strided(
            raw,
            (num_blocks, BLOCK_BYTES),
            (row_bytes, 1),
            storage_offset=i * BLOCK_BYTES,
        )
        for i, name in enumerate(names)
    }
    spec = FullAttentionSpec(
        block_size=256,
        num_kv_heads=2,
        head_size=2,
        dtype=torch.float16,
    )
    assert spec.page_size_bytes == BLOCK_BYTES
    config = KVCacheConfig(
        num_pages=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                names,
                spec,
                num_blocks=num_blocks,
                kernel_rows_per_block=1,
                row_bytes=row_bytes,
            )
        ],
        large_page_bytes=row_bytes,
        block_size=256,
    )
    return config, caches


@pytest.mark.parametrize("num_layers", [1, 2])
def test_explicit_cpu_blocks_bound_live_worker_allocation(monkeypatch, num_layers):
    """Different live KV strides must honor the same explicit CPU pool size."""
    config, caches = _make_lp_config_and_caches(num_layers, num_blocks=4)
    worker = SimpleCPUOffloadWorker(None, config, 8 * BLOCK_BYTES, num_cpu_blocks=4)
    monkeypatch.setattr("vllm.v1.simple_kv_offload.worker.PIN_MEMORY", False)
    try:
        worker.register_kv_caches(caches)
        assert worker.num_cpu_blocks == 4
        assert worker._cpu_arena is not None
        assert worker._cpu_arena.nbytes <= 8 * BLOCK_BYTES
    finally:
        if worker._backend is not None:
            worker._backend.shutdown()


@pytest.mark.parametrize(
    "capacity_bytes, count, local_capacity",
    [(8 * BLOCK_BYTES, 5, 4), (BLOCK_BYTES, 1, 0)],
)
def test_explicit_cpu_blocks_reject_heavier_worker_before_allocation(
    capacity_bytes, count, local_capacity
):
    """A count that fits the scheduler's stage must fail on a smaller worker."""
    config, caches = _make_lp_config_and_caches(num_layers=2, num_blocks=4)
    worker = SimpleCPUOffloadWorker(None, config, capacity_bytes, num_cpu_blocks=count)
    with pytest.raises(ValueError, match=f"local offload capacity {local_capacity}"):
        worker.register_kv_caches(caches)
    assert worker._cpu_arena is None


def test_sync_num_cpu_blocks_across_workers_noop_when_dist_uninitialized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "vllm.v1.simple_kv_offload.sizing.dist.is_initialized",
        lambda: False,
    )
    assert sync_num_offload_blocks_across_workers(42) == 42


def test_sync_num_cpu_blocks_across_workers_noop_when_world_size_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeGroup:
        world_size = 1

    monkeypatch.setattr(
        "vllm.v1.simple_kv_offload.sizing.dist.is_initialized",
        lambda: True,
    )
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_world_group",
        lambda: FakeGroup(),
    )
    assert sync_num_offload_blocks_across_workers(42) == 42


def test_sync_num_cpu_blocks_across_workers_all_reduce_min(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeGroup:
        world_size = 4
        cpu_group = object()

    def fake_all_reduce(tensor, group, op) -> None:
        assert op is dist.ReduceOp.MIN
        tensor.fill_(5)

    monkeypatch.setattr(
        "vllm.v1.simple_kv_offload.sizing.dist.is_initialized",
        lambda: True,
    )
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_world_group",
        lambda: FakeGroup(),
    )
    monkeypatch.setattr(
        "vllm.v1.simple_kv_offload.sizing.dist.all_reduce",
        fake_all_reduce,
    )
    assert sync_num_offload_blocks_across_workers(10) == 5
