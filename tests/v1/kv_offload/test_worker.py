# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.v1.kv_offload.abstract import LoadStoreSpec
from vllm.v1.kv_offload.worker.worker import (
    OffloadingHandler,
    OffloadingWorker,
    TransferResult,
    TransferSpec,
)


class LoadStoreSpec1(LoadStoreSpec):
    def __init__(
        self,
        submit_success: bool = True,
        async_success: bool = True,
        exception: bool = False,
    ):
        self.finished = False
        self.submit_success = submit_success
        self.async_success = async_success
        self.exception = exception

    @staticmethod
    def medium() -> str:
        return "1"

    def __repr__(self):
        return f"{self.medium()}: {id(self)}"


class LoadStoreSpec2(LoadStoreSpec):
    @staticmethod
    def medium() -> str:
        return "2"

    def __repr__(self):
        return f"{self.medium()}: {id(self)}"


class OffloadingHandler1To2(OffloadingHandler):
    def __init__(self):
        self.transfers: dict[int, LoadStoreSpec1] = {}

    def transfer_async(self, job_id: int, spec: TransferSpec) -> bool:
        src, dst = spec
        assert isinstance(src, LoadStoreSpec1)
        assert isinstance(dst, LoadStoreSpec2)

        if src.exception:
            raise Exception("An expected exception. Don't worry!")
        if not src.submit_success:
            return False

        self.transfers[job_id] = src
        return True

    def get_finished(self) -> list[TransferResult]:
        finished = []
        for job_id, spec in list(self.transfers.items()):
            if spec.finished:
                finished.append((job_id, spec.async_success))
                del self.transfers[job_id]
        return finished


class OffloadingHandler2To1(OffloadingHandler):
    def __init__(self):
        self.transfers: dict[int, LoadStoreSpec1] = {}

    def transfer_async(self, job_id: int, spec: TransferSpec) -> bool:
        src, dst = spec
        assert isinstance(src, LoadStoreSpec2)
        assert isinstance(dst, LoadStoreSpec1)

        self.transfers[job_id] = dst
        return True

    def get_finished(self) -> list[TransferResult]:
        finished = []
        for job_id, spec in list(self.transfers.items()):
            if spec.finished:
                finished.append((job_id, spec.async_success))
                del self.transfers[job_id]
        return finished


def test_offloading_worker():
    """
    Tests OffloadingWorker with 2 handlers.
    One handler performs 1->2 transfers, and the other handles 2->1.
    """
    worker = OffloadingWorker()
    handler1to2 = OffloadingHandler1To2()
    handler2to1 = OffloadingHandler2To1()
    worker.register_handler(LoadStoreSpec1, LoadStoreSpec2, handler1to2)
    worker.register_handler(LoadStoreSpec2, LoadStoreSpec1, handler2to1)

    # 1st transfer 1->2 (exception)
    src1 = LoadStoreSpec1(exception=True)
    dst1 = LoadStoreSpec2()
    assert not worker.transfer_async(1, (src1, dst1))

    # 2ed transfer 1->2 (failure to submit)
    src2 = LoadStoreSpec1(submit_success=False)
    dst2 = LoadStoreSpec2()
    assert not worker.transfer_async(2, (src2, dst2))

    # 3rd transfer 1->2 (failure)
    src3 = LoadStoreSpec1(async_success=False)
    dst3 = LoadStoreSpec2()
    assert worker.transfer_async(3, (src3, dst3))

    # 4th transfer 1->2 (success)
    src4 = LoadStoreSpec1()
    dst4 = LoadStoreSpec2()
    worker.transfer_async(4, (src4, dst4))
    assert set(handler1to2.transfers.keys()) == {3, 4}

    # 5th transfer 2->1
    src5 = LoadStoreSpec2()
    dst5 = LoadStoreSpec1()
    worker.transfer_async(5, (src5, dst5))
    assert set(handler2to1.transfers.keys()) == {5}

    # no transfer completed yet
    assert worker.get_finished() == []

    # complete 3rd, 4th
    src3.finished = True
    src4.finished = True

    # 6th transfer 1->2
    src6 = LoadStoreSpec1()
    dst6 = LoadStoreSpec2()
    worker.transfer_async(6, (src6, dst6))

    # 7th transfer 2->1
    src7 = LoadStoreSpec2()
    dst7 = LoadStoreSpec1()
    worker.transfer_async(7, (src7, dst7))

    # 6th and 7th transfers started
    assert 6 in handler1to2.transfers
    assert 7 in handler2to1.transfers

    # verify result of 3rd and 4th transfers
    assert sorted(worker.get_finished()) == [(3, False), (4, True)]

    # complete 6th and 7th transfers
    src6.finished = True
    dst7.finished = True
    assert sorted(worker.get_finished()) == [(6, True), (7, True)]


class LoadStoreSpecWithGroup(LoadStoreSpec):
    """LoadStoreSpec that supports group_id for HMA testing."""

    def __init__(self, group_id: int = 0):
        self.group_id = group_id
        self.finished = False

    @staticmethod
    def medium() -> str:
        return "G"

    def __repr__(self):
        return f"Group{self.group_id}: {id(self)}"


class GroupHandler(OffloadingHandler):
    """Handler that tracks which group it belongs to."""

    def __init__(self, group_id: int):
        self.group_id = group_id
        self.transfers: dict[int, LoadStoreSpecWithGroup] = {}

    def transfer_async(self, job_id: int, spec: TransferSpec) -> bool:
        src, dst = spec
        assert isinstance(src, LoadStoreSpecWithGroup)
        # Verify routing - handler should only receive its group's transfers
        assert src.group_id == self.group_id, (
            f"Handler for group {self.group_id} received "
            f"transfer for group {src.group_id}"
        )
        self.transfers[job_id] = src
        return True

    def get_finished(self) -> list[TransferResult]:
        finished = []
        for job_id, spec in list(self.transfers.items()):
            if spec.finished:
                finished.append((job_id, True))
                del self.transfers[job_id]
        return finished


def test_offloading_worker_group_routing():
    """
    Tests OffloadingWorker with group-specific handlers for HMA support.
    Each group should have its own handler that only receives its transfers.
    """
    worker = OffloadingWorker()

    # Register handlers for 3 groups
    handlers = {}
    for group_id in range(3):
        handler = GroupHandler(group_id)
        handlers[group_id] = handler
        worker.register_group_handler(
            LoadStoreSpecWithGroup,
            LoadStoreSpecWithGroup,
            group_id,
            handler,
        )

    # Submit transfers for different groups
    specs = {}
    for group_id in range(3):
        src = LoadStoreSpecWithGroup(group_id=group_id)
        dst = LoadStoreSpecWithGroup(group_id=group_id)
        job_id = group_id * 10
        specs[job_id] = src
        assert worker.transfer_async(job_id, (src, dst))

    # Verify each handler received only its group's transfer
    for group_id in range(3):
        assert len(handlers[group_id].transfers) == 1
        job_id = group_id * 10
        assert job_id in handlers[group_id].transfers

    # Complete all transfers
    for src in specs.values():
        src.finished = True

    # Verify all finished
    finished = sorted(worker.get_finished())
    assert finished == [(0, True), (10, True), (20, True)]


class PermissiveGroupHandler(OffloadingHandler):
    """Handler that accepts transfers from any group (for fallback testing)."""

    def __init__(self):
        self.transfers: dict[int, LoadStoreSpecWithGroup] = {}

    def transfer_async(self, job_id: int, spec: TransferSpec) -> bool:
        src, dst = spec
        assert isinstance(src, LoadStoreSpecWithGroup)
        self.transfers[job_id] = src
        return True

    def get_finished(self) -> list[TransferResult]:
        finished = []
        for job_id, spec in list(self.transfers.items()):
            if spec.finished:
                finished.append((job_id, True))
                del self.transfers[job_id]
        return finished


def test_offloading_worker_group_fallback():
    """
    Tests that OffloadingWorker falls back to non-group handler
    when no group-specific handler is registered (backward compatibility).
    """
    worker = OffloadingWorker()

    # Register a non-group handler that accepts any group
    default_handler = PermissiveGroupHandler()
    worker.register_handler(
        LoadStoreSpecWithGroup,
        LoadStoreSpecWithGroup,
        default_handler,
    )

    # Submit transfer with group_id=5 (no specific handler registered)
    src = LoadStoreSpecWithGroup(group_id=5)
    dst = LoadStoreSpecWithGroup(group_id=5)
    assert worker.transfer_async(100, (src, dst))

    # Should have been handled by the default handler
    assert 100 in default_handler.transfers


def test_offloading_worker_group_priority():
    """
    Tests that group-specific handlers take priority over non-group handlers.
    """
    worker = OffloadingWorker()

    # Register non-group handler that accepts any group
    default_handler = PermissiveGroupHandler()
    worker.register_handler(
        LoadStoreSpecWithGroup,
        LoadStoreSpecWithGroup,
        default_handler,
    )

    # Register group-specific handler for group 0 only
    group0_handler = GroupHandler(group_id=0)
    worker.register_group_handler(
        LoadStoreSpecWithGroup,
        LoadStoreSpecWithGroup,
        0,
        group0_handler,
    )

    # Submit transfer for group 0 - should go to group0_handler
    src0 = LoadStoreSpecWithGroup(group_id=0)
    dst0 = LoadStoreSpecWithGroup(group_id=0)
    assert worker.transfer_async(1, (src0, dst0))
    assert 1 in group0_handler.transfers
    assert 1 not in default_handler.transfers

    # Submit transfer for group 1 - should fall back to default_handler
    src1 = LoadStoreSpecWithGroup(group_id=1)
    dst1 = LoadStoreSpecWithGroup(group_id=1)
    assert worker.transfer_async(2, (src1, dst1))
    assert 2 not in group0_handler.transfers
    assert 2 in default_handler.transfers


def test_offloading_worker_no_handler():
    """
    Tests that transfer fails gracefully when no handler is registered.
    """
    worker = OffloadingWorker()

    # No handlers registered
    src = LoadStoreSpecWithGroup(group_id=0)
    dst = LoadStoreSpecWithGroup(group_id=0)

    # Should return False (failure) not raise an exception
    assert not worker.transfer_async(1, (src, dst))
