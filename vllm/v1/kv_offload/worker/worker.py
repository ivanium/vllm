# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod

from vllm.logger import init_logger
from vllm.v1.kv_offload.abstract import LoadStoreSpec

# a single transfer spec (src_blocks_spec, dst_blocks_spec)
TransferSpec = tuple[LoadStoreSpec, LoadStoreSpec]
# transfers are forwarded to workers by (src_medium, dst_medium)
TransferType = tuple[str, str]
# transfer result (job_id, success)
TransferResult = tuple[int, bool]

logger = init_logger(__name__)


class OffloadingHandler(ABC):
    """
    OffloadingHandler class for managing asynchronous KV data transfers

    This class runs in the worker.
    It kicks off async KV data transfer requests, and allows
    collecting back completion statuses.

    The class provides the following primitives:
        transfer_async() - kicks off a new transfer job
        get_finished() - returns a list of newly finished job IDs.
    """

    @abstractmethod
    def transfer_async(self, job_id: int, spec: TransferSpec) -> bool:
        """
        Initiates an asynchronous transfer of KV data.

        Args:
            job_id: a unique ID that will be used when notifying back on
                transfer completion.
            spec: the (src, dst) spec of the KV data transfer.

        Returns:
            True if transfer was submitted successfully.
        """
        pass

    @abstractmethod
    def get_finished(self) -> list[TransferResult]:
        """
        Get transfers finished since last call.

        Returns:
            A list of (job_id, success) of transfers.
        """
        pass

    @abstractmethod
    def wait(self, job_ids: set[int]) -> None:
        """
        Wait for jobs to finish (blocking).

        Args:
            job_ids: The set of job IDs to wait for.
        """


class OffloadingWorker:
    """
    OffloadingWorker class for managing asynchronous KV data transfers
    using multiple OffloadingHandlers

    This class runs in the worker.
    It kicks off async KV data transfer requests, by delegating
    to one of its registered OffloadingHandlers, based on the transfer type.

    Supports HMA (Hybrid Memory Allocator) with per-group handlers:
    - Use register_group_handler() to register handlers for specific KV cache groups
    - transfer_async() routes to group-specific handlers when available
    - Falls back to non-group handlers for backward compatibility

    The class provides the following primitives:
        register_handler() - registers a handler for a transfer type
        register_group_handler() - registers a handler for a transfer type + group
        transfer_async() - kicks off a new transfer job
            using one of the registered handlers.
        get_finished() - returns a list of newly finished job IDs
            from all handlers.
    """

    def __init__(self):
        self.handlers: set[OffloadingHandler] = set()
        # Original: (src_medium, dst_medium) -> handler
        self.transfer_type_to_handler: dict[TransferType, OffloadingHandler] = {}
        # HMA: ((src_medium, dst_medium), group_id) -> handler
        self.group_transfer_handlers: dict[
            tuple[TransferType, int], OffloadingHandler
        ] = {}

    def register_handler(
        self,
        src_cls: type[LoadStoreSpec],
        dst_cls: type[LoadStoreSpec],
        handler: OffloadingHandler,
    ) -> None:
        """
        Registers a new handler for a transfer type.

        This is the original non-group-aware method, kept for backward
        compatibility with single-group models.

        Args:
            src_cls: the source type of transfers handled by this handler.
            dst_cls: the destination type of transfers handled by this handler.
            handler: the handler that will handle transfers.
        """
        transfer_type = (src_cls.medium(), dst_cls.medium())
        assert transfer_type not in self.transfer_type_to_handler
        self.handlers.add(handler)
        self.transfer_type_to_handler[transfer_type] = handler

    def register_group_handler(
        self,
        src_cls: type[LoadStoreSpec],
        dst_cls: type[LoadStoreSpec],
        group_id: int,
        handler: OffloadingHandler,
    ) -> None:
        """
        Registers a handler for a specific KV cache group (HMA support).

        Each group can have its own handler that only knows about
        that group's layer tensors. This enables per-group CPU offloading
        for hybrid models with multiple KV cache types.

        Args:
            src_cls: the source type of transfers handled by this handler.
            dst_cls: the destination type of transfers handled by this handler.
            group_id: the KV cache group ID this handler is for.
            handler: the handler that will handle transfers for this group.
        """
        transfer_type = (src_cls.medium(), dst_cls.medium())
        key = (transfer_type, group_id)
        assert key not in self.group_transfer_handlers, (
            f"Handler already registered for {transfer_type}, group {group_id}"
        )
        self.handlers.add(handler)
        self.group_transfer_handlers[key] = handler

    def transfer_async(self, job_id: int, spec: TransferSpec) -> bool:
        """
        Initiates an asynchronous transfer of KV data.

        Routes transfers to the appropriate handler:
        1. First tries group-specific handler based on spec's group_id (HMA)
        2. Falls back to non-group handler for backward compatibility

        Args:
            job_id: a unique ID that will be used when notifying back on
                transfer completion.
            spec: the (src, dst) spec of the KV data transfer.

        Returns:
            True if transfer was submitted successfully.
        """
        src, dst = spec
        transfer_type = (src.medium(), dst.medium())

        # Try group-specific handler first (HMA support)
        group_id = getattr(src, "group_id", 0)
        key = (transfer_type, group_id)
        handler = self.group_transfer_handlers.get(key)

        if handler is None:
            # Fall back to non-group handler (backward compatibility)
            handler = self.transfer_type_to_handler.get(transfer_type)

        if handler is None:
            logger.error(
                "No handler registered for %r transfer (group %d)",
                transfer_type,
                group_id,
            )
            return False

        try:
            success = handler.transfer_async(job_id, spec)
        except Exception as e:
            logger.warning(
                "Exception in %r transfer %d (group %d): %r",
                transfer_type,
                job_id,
                group_id,
                e,
                exc_info=True,
            )
            return False

        if not success:
            logger.warning(
                "Failed to submit %r transfer %d (group %d)",
                transfer_type,
                job_id,
                group_id,
            )
        else:
            logger.debug(
                "Submitted %r transfer %d (group %d)",
                transfer_type,
                job_id,
                group_id,
            )

        return success

    def get_finished(self) -> list[TransferResult]:
        """
        Get transfers finished since last call.

        Returns:
            A list of (job_id, success) of transfers.
        """
        finished = []
        for handler in self.handlers:
            finished.extend(handler.get_finished())
        return finished

    def wait(self, job_ids: set[int]) -> None:
        """
        Wait for jobs to finish (blocking).

        Args:
            job_ids: The set of job IDs to wait for.
        """
        for handler in self.handlers:
            handler.wait(job_ids)
