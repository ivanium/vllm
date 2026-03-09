# Disk Backend V2 Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Redesign disk offloading to use MultiConnector composition instead of embedding disk logic in SimpleCPUOffloadConnector. The disk backend copies directly GPU↔Disk with internal staging buffers.

**Architecture:** Two `SimpleCPUOffloadConnector` instances (one for CPU, one for disk) composed via `MultiConnector`. The connector is parameterized by `backend_type` config. `DiskTransferBackend` is rewritten for direct GPU↔Disk with internal pinned CPU staging. All two-hop and disk-specific code is removed from the connector.

**Tech Stack:** Python, PyTorch (`.to("cuda", non_blocking=True)`, CUDA streams, pinned memory), `os.pwrite`/`os.pread`, `ThreadPoolExecutor`.

**Design doc:** `docs/plans/2026-03-10-disk-backend-v2-design.md`

---

**Note on MultiConnector:** The design doc lists changes to `multi_connector.py` (cross-wiring, `request_finished_all_groups`). These are Option B concerns — for Option A, MultiConnector's existing delegation pattern works correctly since each child connector is independently functional. Cross-wiring will be added in the follow-up PR.

**Note on CUDA streams:** The design doc specifies a dedicated low-priority CUDA stream in DiskTransferBackend for GPU↔staging copies. The initial implementation uses synchronous indexed copies which work on both CPU and GPU tensors. A follow-up optimization will add a dedicated CUDA stream for GPU to avoid blocking the compute pipeline. Unit tests use CPU tensors, so this has no impact on test correctness.

---

## Chunk 1: Clean Up Connector and Metadata

### Task 1: Add get_offload_block_pool to Base Class

Add the hook method that MultiConnector will use for future cross-wiring.

**Files:**

- Modify: `vllm/distributed/kv_transfer/kv_connector/v1/base.py`

- [ ] **Step 1: Add default method to KVConnectorBase_V1**

After `reset_cache()` (around line 636), add:

```python
    def get_offload_block_pool(self) -> Any:
        """Return the offload-tier block pool, if any.

        Used by MultiConnector cross-wiring to let one connector's
        offload pool serve as another connector's source pool.
        """
        return None
```

- [ ] **Step 2: Commit**

```bash
git add vllm/distributed/kv_transfer/kv_connector/v1/base.py
git commit -m "feat: add get_offload_block_pool hook to KVConnectorBase_V1"
```

---

### Task 2: Remove Disk Fields from Metadata

Remove the disk-specific fields from `SimpleCPUOffloadMetadata`. Each connector instance uses the standard fields independently via `MultiKVConnectorMetadata`.

**Files:**

- Modify: `vllm/distributed/kv_transfer/kv_connector/v1/simple_cpu_offload/metadata.py`
- Modify: `tests/v1/kv_connector/unit/test_simple_cpu_offload_connector.py`

- [ ] **Step 1: Remove disk fields from metadata**

In `metadata.py`, delete lines 34-42 (all `disk_*` fields):

```python
# DELETE these lines:
    # Disk transfer fields (used when disk backend is enabled)
    disk_load_event: int = INVALID_JOB_ID
    disk_load_cpu_blocks: list[int] = field(default_factory=list)
    disk_load_disk_blocks: list[int] = field(default_factory=list)
    disk_load_event_to_reqs: dict[int, list[str]] = field(default_factory=dict)

    disk_store_event: int = INVALID_JOB_ID
    disk_store_cpu_blocks: list[int] = field(default_factory=list)
    disk_store_disk_blocks: list[int] = field(default_factory=list)
```

- [ ] **Step 2: Remove TestDiskMetadata test class**

In the test file, delete the entire `TestDiskMetadata` class (tests for removed fields).

- [ ] **Step 3: Run tests**

Run: `pytest tests/v1/kv_connector/unit/test_simple_cpu_offload_connector.py::TestSimpleCPUOffloadMetadata -v -x`
Expected: All metadata tests PASS

- [ ] **Step 4: Commit**

```bash
git add vllm/distributed/kv_transfer/kv_connector/v1/simple_cpu_offload/metadata.py tests/v1/kv_connector/unit/test_simple_cpu_offload_connector.py
git commit -m "refactor: remove disk fields from SimpleCPUOffloadMetadata"
```

---

### Task 3: Strip Disk Code from SimpleCPUOffloadConnector

Remove all disk-specific logic from the connector: two-hop state machine, disk manager/worker, sentinel handling. Add `backend_type` parameterization and `get_offload_block_pool()`.

**Files:**

- Modify: `vllm/distributed/kv_transfer/kv_connector/v1/simple_cpu_offload_connector.py`
- Modify: `tests/v1/kv_connector/unit/test_simple_cpu_offload_connector.py`

- [ ] **Step 1: Rewrite the connector**

Replace the entire `simple_cpu_offload_connector.py` with the cleaned-up version. Key changes:

**Remove (~200 lines):**

- `_DISK_LOAD_DONE_PREFIX` constant (line 56)
- `_TwoHopLoadState` dataclass (lines 59-79)
- `self.disk_manager` and `self.disk_worker` fields
- `self._pending_disk_loads` and `self._two_hop_loads` dicts
- All disk manager/worker creation in `__init__` (lines 139-164)
- Disk metadata remapping in `bind_connector_metadata` (lines 190-204)
- Disk worker calls in `clear_connector_metadata`, `handle_preemptions`, `start_load_kv`, `wait_for_save`
- Disk sentinel merging in `get_finished` (lines 260-285)
- Disk hit fallback in `get_num_new_matched_tokens` (lines 305-319)
- Two-hop setup in `update_state_after_alloc` (lines 330-336)
- `_setup_two_hop_load()` method (lines 343-471)
- `_queue_hop2()` method (lines 473-503)
- Disk field merging in `build_connector_meta` (lines 515-563)
- Two-hop intercept in `update_connector_output` (lines 572-628)
- `_cleanup_two_hop_state()` method (lines 652-681)
- Two-hop cleanup calls in `request_finished` and `request_finished_all_groups`

**Add (~20 lines):**

- `backend_type` config reading and backend selection in `__init__`
- `get_offload_block_pool()` method

The resulting connector should be ~150 lines (down from ~687).

Here is the complete new connector code:

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SimpleCPUOffloadConnector: minimal KV cache offloading."""

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_events import KVCacheEvent
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload.backends import (
    CudaTransferBackend,
    DiskTransferBackend,
)
from vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload.manager import (
    SimpleCPUOffloadScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload.metadata import (
    SimpleCPUOffloadMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload.worker import (
    SimpleCPUOffloadWorker,
)
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import KVConnectorOutput

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
    from vllm.v1.attention.backend import AttentionMetadata
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)

# Default CPU capacity: 8 GB
DEFAULT_CPU_CAPACITY_BYTES = 8 * (1024**3)


class SimpleCPUOffloadConnector(KVConnectorBase_V1, SupportsHMA):
    """KV cache offloading with configurable transfer backend.

    Supports GPU↔CPU (CudaTransferBackend, default) and GPU↔Disk
    (DiskTransferBackend, via backend_type="disk" config).
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig | None" = None,
    ):
        super().__init__(vllm_config, role, kv_cache_config)

        enable_prefix_caching = vllm_config.cache_config.enable_prefix_caching
        extra_config = self._kv_transfer_config.kv_connector_extra_config or {}
        lazy_offload = bool(extra_config.get("lazy_offload", False))
        min_lookahead_blocks = int(extra_config.get("min_lookahead_blocks", 8))

        # Backend selection
        backend_type = str(extra_config.get("backend_type", "cuda"))
        if backend_type == "disk":
            capacity_bytes = int(extra_config.get("disk_bytes_to_use", 0))
            disk_path = str(
                extra_config.get("disk_path", "/tmp/vllm_kv_cache")
            )
            backend = DiskTransferBackend(disk_path=disk_path)
        else:
            capacity_bytes = int(
                extra_config.get(
                    "cpu_bytes_to_use", DEFAULT_CPU_CAPACITY_BYTES
                )
            )
            backend = CudaTransferBackend()

        self.scheduler_manager: SimpleCPUOffloadScheduler | None = None
        self.worker_handler: SimpleCPUOffloadWorker | None = None

        if not enable_prefix_caching:
            logger.warning(
                "Detected prefix caching disabled, disabling offload "
                "since it requires prefix caching."
            )
            return

        logger.info(
            "SimpleCPUOffloadConnector: role=%s, capacity=%.2f GB, "
            "mode=%s, backend=%s",
            role.name,
            capacity_bytes / (1024**3),
            "lazy" if lazy_offload else "eager",
            backend_type,
        )

        if role == KVConnectorRole.SCHEDULER:
            self.scheduler_manager = SimpleCPUOffloadScheduler(
                vllm_config,
                kv_cache_config,
                capacity_bytes,
                lazy_offload=lazy_offload,
                min_lookahead_blocks=min_lookahead_blocks,
            )
        elif role == KVConnectorRole.WORKER:
            self.worker_handler = SimpleCPUOffloadWorker(
                vllm_config,
                kv_cache_config,
                capacity_bytes,
                backend=backend,
            )

    # --- Worker-side methods ---

    def register_kv_caches(
        self,
        kv_caches: dict[str, torch.Tensor],
        kv_cache_raw_tensors: dict[str, torch.Tensor] | None = None,
    ) -> None:
        if self.worker_handler is not None:
            self.worker_handler.register_kv_caches(
                kv_caches, kv_cache_raw_tensors
            )

    def bind_connector_metadata(
        self,
        connector_metadata: KVConnectorMetadata,
    ) -> None:
        super().bind_connector_metadata(connector_metadata)
        if self.worker_handler is not None:
            assert isinstance(connector_metadata, SimpleCPUOffloadMetadata)
            self.worker_handler.bind_connector_metadata(connector_metadata)

    def clear_connector_metadata(self) -> None:
        super().clear_connector_metadata()
        if self.worker_handler is not None:
            self.worker_handler.clear_connector_metadata()

    def handle_preemptions(self, preempted_req_ids: set[str]) -> None:
        if self.worker_handler is not None:
            self.worker_handler.handle_preemptions()

    def start_load_kv(
        self,
        forward_context: "ForwardContext",
        **kwargs: Any,
    ) -> None:
        if self.worker_handler is not None:
            self.worker_handler.start_load_kv()

    def wait_for_layer_load(self, layer_name: str) -> None:
        pass

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> None:
        pass

    def wait_for_save(self) -> None:
        if self.worker_handler is not None:
            self.worker_handler.wait_for_save()

    def get_finished(
        self,
        finished_req_ids: set[str],
    ) -> tuple[set[str] | None, set[str] | None]:
        if self.worker_handler is not None:
            return self.worker_handler.get_finished(finished_req_ids)
        return None, None

    # --- Scheduler-side methods ---

    def get_offload_block_pool(self) -> "BlockPool | None":
        """Return the offload-tier block pool (for cross-connector wiring)."""
        if self.scheduler_manager is not None:
            return self.scheduler_manager.cpu_block_pool
        return None

    def bind_gpu_block_pool(self, gpu_block_pool: "BlockPool") -> None:
        if self.scheduler_manager is not None:
            self.scheduler_manager.bind_gpu_block_pool(gpu_block_pool)

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        if self.scheduler_manager is not None:
            return self.scheduler_manager.get_num_new_matched_tokens(
                request, num_computed_tokens
            )
        return 0, False

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        if self.scheduler_manager is not None:
            self.scheduler_manager.update_state_after_alloc(
                request, blocks, num_external_tokens
            )

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        if self.scheduler_manager is None:
            return SimpleCPUOffloadMetadata()
        return self.scheduler_manager.build_connector_meta(scheduler_output)

    def update_connector_output(
        self,
        connector_output: KVConnectorOutput,
    ) -> None:
        if self.scheduler_manager is not None:
            self.scheduler_manager.update_connector_output(connector_output)

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        if self.scheduler_manager is not None:
            return self.scheduler_manager.request_finished(request, block_ids)
        return False, None

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        if self.scheduler_manager is not None:
            return self.scheduler_manager.request_finished_all_groups(
                request, block_ids
            )
        return False, None

    def take_events(self) -> Iterable[KVCacheEvent]:
        if self.scheduler_manager is not None:
            return self.scheduler_manager.take_events()
        return []
```

- [ ] **Step 2: Remove disk-specific tests, update remaining tests**

Delete these test classes entirely:

- `TestDiskTierInitialization` — tests `disk_manager`/`disk_worker` which no longer exist
- `TestDiskTierStoreScheduler` — tests disk store fields which no longer exist
- `TestDiskTierTwoHopLoad` — tests two-hop load which no longer exists
- `TestDiskTierIntegration` — tests two-hop integration which no longer exists
- `_create_disk_connector` helper function

Also fix existing test that references removed attributes:

- `test_disk_tier_not_created_by_default`: references `connector.disk_manager` — delete this test
- `test_disk_tier_worker_not_created_by_default`: references `connector.disk_worker` — delete this test
- `test_bind_and_clear_connector_metadata`: references `_pending_load_job_indices` which should be `_pending_load_event_indices` — fix this

In `TestSimpleCPUOffloadConnector.test_bind_and_clear_connector_metadata`, change:

```python
# Old:
assert 0 in connector.worker_handler._pending_load_job_indices
# New:
assert 0 in connector.worker_handler._pending_load_event_indices
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/v1/kv_connector/unit/test_simple_cpu_offload_connector.py -v -x`
Expected: All remaining tests PASS

- [ ] **Step 4: Commit**

```bash
git add vllm/distributed/kv_transfer/kv_connector/v1/simple_cpu_offload_connector.py tests/v1/kv_connector/unit/test_simple_cpu_offload_connector.py
git commit -m "refactor: strip disk code from SimpleCPUOffloadConnector, add backend_type config"
```

---

## Chunk 2: Rewrite DiskTransferBackend

### Task 4: Rewrite DiskTransferBackend for Direct GPU↔Disk

Replace the current CPU↔Disk backend with one that handles GPU↔Disk directly, using internal pinned CPU staging buffers.

**Files:**

- Modify: `vllm/distributed/kv_transfer/kv_connector/v1/simple_cpu_offload/backends/` (directory with `base.py`, `cuda.py`, `disk.py`)
- Test: `tests/v1/kv_connector/unit/test_simple_cpu_offload_connector.py`

- [ ] **Step 1: Write tests for the new DiskTransferBackend**

Add to the test file:

```python
class TestDiskTransferBackend:
    """Tests for DiskTransferBackend (direct GPU↔Disk with staging)."""

    def test_setup_creates_file(self):
        """setup() creates a pre-allocated flat file and staging buffers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = DiskTransferBackend(
                disk_path=os.path.join(tmpdir, "kv_cache"),
            )
            # Create mock GPU KV caches (CPU tensors stand in for GPU in tests)
            src_caches = {
                "layer.0": torch.zeros(10, 64, dtype=torch.float16),
            }
            num_blocks = backend.setup(
                src_caches, capacity_bytes=10 * 64 * 2, kv_cache_config=None
            )
            assert num_blocks > 0
            assert backend.is_initialized
            assert os.path.exists(os.path.join(tmpdir, "kv_cache"))
            backend.shutdown()

    def test_store_and_load_roundtrip(self):
        """Writing blocks to disk and reading them back produces same data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = DiskTransferBackend(
                disk_path=os.path.join(tmpdir, "kv_cache"),
            )
            src_caches = {
                "layer.0": torch.arange(
                    640, dtype=torch.float16
                ).view(10, 64),
            }
            backend.setup(
                src_caches, capacity_bytes=10 * 64 * 2, kv_cache_config=None
            )

            # Store blocks 0,1 from GPU to disk blocks 0,1
            backend.copy_blocks([0, 1], [0, 1], is_store=True)
            event = backend.record_event()
            backend.sync_event(event)

            # Zero out blocks 2,3 to verify load works
            src_caches["layer.0"][2:4] = 0

            # Load disk blocks 0,1 back to GPU blocks 2,3
            backend.copy_blocks([0, 1], [2, 3], is_store=False)
            event = backend.record_event()
            backend.sync_event(event)

            assert torch.equal(
                src_caches["layer.0"][0], src_caches["layer.0"][2]
            )
            assert torch.equal(
                src_caches["layer.0"][1], src_caches["layer.0"][3]
            )
            backend.shutdown()

    def test_query_event_after_sync(self):
        """query_event returns True after sync_event."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = DiskTransferBackend(
                disk_path=os.path.join(tmpdir, "kv_cache"),
            )
            src_caches = {
                "layer.0": torch.zeros(10, 64, dtype=torch.float16),
            }
            backend.setup(
                src_caches, capacity_bytes=10 * 64 * 2, kv_cache_config=None
            )

            backend.copy_blocks([0], [0], is_store=True)
            event = backend.record_event()
            backend.sync_event(event)
            assert backend.query_event(event) is True
            backend.shutdown()

    def test_multi_layer_roundtrip(self):
        """Roundtrip works with multiple KV cache layers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = DiskTransferBackend(
                disk_path=os.path.join(tmpdir, "kv_cache"),
            )
            src_caches = {
                "layer.0": torch.randn(8, 32, dtype=torch.float16),
                "layer.1": torch.randn(8, 32, dtype=torch.float16),
            }
            # capacity for 8 blocks * 2 layers * 32 * 2 bytes
            backend.setup(
                src_caches, capacity_bytes=8 * 32 * 2 * 2,
                kv_cache_config=None,
            )

            original_0 = src_caches["layer.0"][0].clone()
            original_1 = src_caches["layer.1"][0].clone()

            backend.copy_blocks([0], [0], is_store=True)
            event = backend.record_event()
            backend.sync_event(event)

            # Corrupt source
            src_caches["layer.0"][1] = 0
            src_caches["layer.1"][1] = 0

            # Load back to block 1
            backend.copy_blocks([0], [1], is_store=False)
            event = backend.record_event()
            backend.sync_event(event)

            assert torch.equal(src_caches["layer.0"][1], original_0)
            assert torch.equal(src_caches["layer.1"][1], original_1)
            backend.shutdown()

    def test_shutdown_cleans_up_file(self):
        """shutdown() closes fd and unlinks the file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "kv_cache")
            backend = DiskTransferBackend(disk_path=path)
            src_caches = {
                "layer.0": torch.zeros(4, 16, dtype=torch.float16),
            }
            backend.setup(
                src_caches, capacity_bytes=4 * 16 * 2,
                kv_cache_config=None,
            )
            assert os.path.exists(path)
            backend.shutdown()
            assert not os.path.exists(path)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/v1/kv_connector/unit/test_simple_cpu_offload_connector.py::TestDiskTransferBackend -v -x`
Expected: FAIL (DiskTransferBackend doesn't match new interface yet)

- [ ] **Step 3: Rewrite DiskTransferBackend**

Replace the `DiskTransferBackend` class and `_DiskEvent` class in `backends/disk.py` with:

```python
class _DiskEvent:
    """Completion event for disk I/O operations."""

    __slots__ = ("_event",)

    def __init__(self) -> None:
        self._event = threading.Event()

    def set(self) -> None:
        self._event.set()

    def query(self) -> bool:
        return self._event.is_set()

    def wait(self) -> None:
        self._event.wait()


class DiskTransferBackend(TransferBackend):
    """Transfer backend for direct GPU↔Disk via internal CPU staging.

    Stores use GPU→staging(CPU)→Disk; loads use Disk→staging(CPU)→GPU.
    The staging buffers are internal and not visible to the scheduler.
    """

    DEFAULT_NUM_STAGING_BLOCKS = 64

    def __init__(
        self,
        disk_path: str,
        num_workers: int = 4,
        num_staging_blocks: int = DEFAULT_NUM_STAGING_BLOCKS,
    ) -> None:
        self._disk_path = disk_path
        self._num_workers = num_workers
        self._num_staging_blocks = num_staging_blocks

        # Populated by setup()
        self._fd: int | None = None
        self._gpu_kv_caches: dict[str, torch.Tensor] | None = None
        self._staging: dict[str, torch.Tensor] | None = None
        self._num_disk_blocks: int = 0
        self._bytes_per_block: int = 0
        self._num_layers: int = 0
        self._pool: ThreadPoolExecutor | None = None
        self._last_event: _DiskEvent | None = None

    @property
    def is_initialized(self) -> bool:
        return (
            self._fd is not None
            and self._gpu_kv_caches is not None
            and self._staging is not None
            and self._pool is not None
        )

    def setup(
        self,
        src_caches: dict[str, torch.Tensor],
        capacity_bytes: int,
        kv_cache_config: "KVCacheConfig | None",
    ) -> int:
        self._gpu_kv_caches = src_caches
        self._num_layers = len(src_caches)

        first = next(iter(src_caches.values()))
        self._bytes_per_block = first.stride(0) * first.element_size()

        self._num_disk_blocks = max(
            1, capacity_bytes // (self._bytes_per_block * self._num_layers)
        )

        total_bytes = (
            self._bytes_per_block * self._num_layers * self._num_disk_blocks
        )

        logger.info(
            "DiskTransferBackend: %d layers, %d disk blocks (%.2f GB), "
            "%d staging blocks, file=%s",
            self._num_layers,
            self._num_disk_blocks,
            total_bytes / (1024**3),
            self._num_staging_blocks,
            self._disk_path,
        )

        # Allocate pinned CPU staging buffers (is_pin_memory_available
        # is imported at the top of backends.py)
        pin = is_pin_memory_available()
        self._staging = {}
        for name, gpu_tensor in src_caches.items():
            shape = (self._num_staging_blocks,) + gpu_tensor.shape[1:]
            self._staging[name] = torch.zeros(
                shape, dtype=gpu_tensor.dtype, device="cpu", pin_memory=pin,
            )

        # Open backing file
        self._fd = os.open(
            self._disk_path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644,
        )
        try:
            os.posix_fallocate(self._fd, 0, total_bytes)
        except (OSError, AttributeError):
            os.ftruncate(self._fd, total_bytes)

        self._pool = ThreadPoolExecutor(max_workers=self._num_workers)
        return self._num_disk_blocks

    def copy_blocks(
        self,
        src_block_ids: list[int],
        dst_block_ids: list[int],
        is_store: bool,
    ) -> None:
        assert self._fd is not None
        assert self._gpu_kv_caches is not None
        assert self._staging is not None
        assert self._pool is not None

        event = _DiskEvent()
        self._last_event = event

        n = len(src_block_ids)
        if n == 0:
            event.set()
            return

        # Process in chunks of staging capacity
        chunk_size = self._num_staging_blocks
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            src_chunk = src_block_ids[start:end]
            dst_chunk = dst_block_ids[start:end]
            is_last_chunk = (end == n)

            if is_store:
                self._store_chunk(
                    src_chunk, dst_chunk,
                    event if is_last_chunk else None,
                )
            else:
                self._load_chunk(
                    src_chunk, dst_chunk,
                    event if is_last_chunk else None,
                )

    def _store_chunk(
        self,
        gpu_block_ids: list[int],
        disk_block_ids: list[int],
        event: _DiskEvent | None,
    ) -> None:
        """GPU → staging → Disk for one chunk."""
        assert self._gpu_kv_caches is not None
        assert self._staging is not None
        assert self._pool is not None

        n = len(gpu_block_ids)
        staging_ids = list(range(n))

        # GPU → staging (CPU copy via torch)
        for name in self._gpu_kv_caches:
            self._staging[name][staging_ids] = (
                self._gpu_kv_caches[name][gpu_block_ids]
            )

        # staging → Disk (async pwrite via thread pool)
        sem = threading.Semaphore(0)
        num_ios = n * self._num_layers
        fd = self._fd
        bpb = self._bytes_per_block

        for layer_idx, name in enumerate(self._staging):
            for i, disk_id in enumerate(disk_block_ids):
                data = self._staging[name][i].contiguous().numpy().tobytes()
                offset = self._file_offset(layer_idx, disk_id)
                self._pool.submit(self._pwrite_task, fd, data, offset, sem)

        # Wait for all pwrite I/Os, then signal event
        if event is not None:
            self._pool.submit(self._waiter_task, sem, num_ios, event)
        else:
            # Sync: must wait before reusing staging buffers
            for _ in range(num_ios):
                sem.acquire()

    def _load_chunk(
        self,
        disk_block_ids: list[int],
        gpu_block_ids: list[int],
        event: _DiskEvent | None,
    ) -> None:
        """Disk → staging → GPU for one chunk."""
        assert self._gpu_kv_caches is not None
        assert self._staging is not None
        assert self._pool is not None

        n = len(disk_block_ids)
        staging_ids = list(range(n))

        # Disk → staging (sync pread via thread pool, must complete before
        # GPU copy)
        sem = threading.Semaphore(0)
        num_ios = n * self._num_layers
        fd = self._fd
        bpb = self._bytes_per_block

        for layer_idx, name in enumerate(self._staging):
            tensor = self._staging[name]
            for i, disk_id in enumerate(disk_block_ids):
                offset = self._file_offset(layer_idx, disk_id)
                self._pool.submit(
                    self._pread_task, fd, tensor, i, offset, bpb, sem,
                )

        # Wait for all reads to complete
        for _ in range(num_ios):
            sem.acquire()

        # staging → GPU (CPU copy via torch)
        for name in self._gpu_kv_caches:
            self._gpu_kv_caches[name][gpu_block_ids] = (
                self._staging[name][staging_ids]
            )

        if event is not None:
            event.set()

    def _file_offset(self, layer_idx: int, block_id: int) -> int:
        return (
            layer_idx * self._num_disk_blocks * self._bytes_per_block
            + block_id * self._bytes_per_block
        )

    def record_event(self) -> _DiskEvent:
        assert self._last_event is not None
        event = self._last_event
        self._last_event = None
        return event

    def query_event(self, event: Any) -> bool:
        return event.query()

    def sync_event(self, event: Any) -> None:
        event.wait()

    def sync_all(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = ThreadPoolExecutor(max_workers=self._num_workers)

    def validate_block_ids(
        self, block_ids: list[int], is_src: bool
    ) -> None:
        if not block_ids:
            return
        if is_src:
            assert self._gpu_kv_caches is not None
            num = next(iter(self._gpu_kv_caches.values())).shape[0]
            label = "Source (GPU)"
        else:
            num = self._num_disk_blocks
            label = "Dest (Disk)"
        lo, hi = min(block_ids), max(block_ids)
        if lo < 0 or hi >= num:
            bad = lo if lo < 0 else hi
            raise ValueError(
                f"{label} block ID {bad} out of bounds [0, {num})"
            )

    def shutdown(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        # Clean up the backing file
        if os.path.exists(self._disk_path):
            os.unlink(self._disk_path)

    # --- I/O helpers ---

    @staticmethod
    def _pwrite_task(
        fd: int, data: bytes, offset: int, sem: threading.Semaphore,
    ) -> None:
        os.pwrite(fd, data, offset)
        sem.release()

    @staticmethod
    def _pread_task(
        fd: int,
        tensor: torch.Tensor,
        staging_idx: int,
        offset: int,
        nbytes: int,
        sem: threading.Semaphore,
    ) -> None:
        data = os.pread(fd, nbytes, offset)
        block = torch.frombuffer(bytearray(data), dtype=tensor.dtype)
        tensor[staging_idx].copy_(block.view(tensor[staging_idx].shape))
        sem.release()

    @staticmethod
    def _waiter_task(
        sem: threading.Semaphore, count: int, event: _DiskEvent,
    ) -> None:
        for _ in range(count):
            sem.acquire()
        event.set()
```

**Note on CPU-only testing:** The tests use regular CPU tensors (not CUDA tensors) as stand-ins for GPU tensors. The index-copy operations (`tensor[ids] = ...`) work identically on CPU, so the data path is fully exercised. When running on actual GPU, `torch.to("cuda", non_blocking=True)` would handle the DMA; in tests, the CPU path is equivalent.

- [ ] **Step 4: Run DiskTransferBackend tests**

Run: `pytest tests/v1/kv_connector/unit/test_simple_cpu_offload_connector.py::TestDiskTransferBackend -v -x`
Expected: All PASS

- [ ] **Step 5: Run all tests**

Run: `pytest tests/v1/kv_connector/unit/test_simple_cpu_offload_connector.py -v -x`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add vllm/distributed/kv_transfer/kv_connector/v1/simple_cpu_offload/backends/ tests/v1/kv_connector/unit/test_simple_cpu_offload_connector.py
git commit -m "feat: rewrite DiskTransferBackend for direct GPU↔Disk with staging"
```

---

## Chunk 3: Base Class Hook, Cleanup, and Linting

### Task 5: Update **init**.py Exports

Clean up exports to remove references to types that no longer need exporting.

**Files:**

- Modify: `vllm/distributed/kv_transfer/kv_connector/v1/simple_cpu_offload/__init__.py`

- [ ] **Step 1: Update exports**

The backends are now in a `backends/` subdirectory with their own `__init__.py`. The top-level `__init__.py` only needs to export `SimpleCPUOffloadMetadata`:

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Simple CPU/Disk offloading connector for KV cache."""

from vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload.metadata import (
    SimpleCPUOffloadMetadata,
)

__all__ = [
    "SimpleCPUOffloadMetadata",
]
```

- [ ] **Step 2: Check for external imports of removed exports**

Run: `grep -r "from.*simple_cpu_offload import.*TransferBackend\|from.*simple_cpu_offload import.*CudaTransferBackend\|from.*simple_cpu_offload import.*DiskTransferBackend" vllm/ --include="*.py"`

If any files import from `__init__.py`, update them to import directly from `backends.py`. The connector already imports directly from `backends.py`.

- [ ] **Step 3: Commit**

```bash
git add vllm/distributed/kv_transfer/kv_connector/v1/simple_cpu_offload/__init__.py
git commit -m "chore: simplify simple_cpu_offload exports"
```

---

### Task 6: Pre-commit Checks and Final Cleanup (IMPORTANT: review all code carefully)

**Files:**

- All modified files

- [ ] **Step 1: Run pre-commit on all modified files**

```bash
pre-commit run --files \
  vllm/distributed/kv_transfer/kv_connector/v1/simple_cpu_offload_connector.py \
  vllm/distributed/kv_transfer/kv_connector/v1/simple_cpu_offload/backends/base.py \
  vllm/distributed/kv_transfer/kv_connector/v1/simple_cpu_offload/backends/cuda.py \
  vllm/distributed/kv_transfer/kv_connector/v1/simple_cpu_offload/backends/disk.py \
  vllm/distributed/kv_transfer/kv_connector/v1/simple_cpu_offload/metadata.py \
  vllm/distributed/kv_transfer/kv_connector/v1/simple_cpu_offload/__init__.py \
  vllm/distributed/kv_transfer/kv_connector/v1/base.py \
  tests/v1/kv_connector/unit/test_simple_cpu_offload_connector.py
```

Expected: All checks pass. Fix any formatting/linting issues.

- [ ] **Step 2: Run full test suite one final time**

```bash
pytest tests/v1/kv_connector/unit/test_simple_cpu_offload_connector.py -v
```

Expected: All tests PASS

- [ ] **Step 3: Review code cleanliness**

Verify:

- No unused imports in any modified file
- No leftover references to `disk_manager`, `disk_worker`, `_two_hop_loads`, `_pending_disk_loads`, `_DISK_LOAD_DONE_PREFIX`
- No dead code or commented-out code
- The connector file is ~150 lines (clean delegation)
- DiskTransferBackend has clear store/load separation
- Tests are focused and non-redundant

```bash
grep -rn "disk_manager\|disk_worker\|_two_hop\|_pending_disk\|_DISK_LOAD_DONE" \
  vllm/distributed/kv_transfer/kv_connector/v1/simple_cpu_offload_connector.py \
  tests/v1/kv_connector/unit/test_simple_cpu_offload_connector.py
```

Expected: No matches

- [ ] **Step 4: Commit any fixes**

```bash
git add -u
git commit -m "chore: lint fixes and final cleanup"
```

---

### Task 7: Update Design Doc Status

- [ ] **Step 1: Mark design doc as implemented**

In `docs/plans/2026-03-10-disk-backend-v2-design.md`, change:

```markdown
**Status**: Approved
```

to:

```markdown
**Status**: Implemented (Option A)
```

- [ ] **Step 2: Commit**

```bash
git add docs/plans/2026-03-10-disk-backend-v2-design.md
git commit -m "docs: mark disk backend v2 design as implemented (Option A)"
```
