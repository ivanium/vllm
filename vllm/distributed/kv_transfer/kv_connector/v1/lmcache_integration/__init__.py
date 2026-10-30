# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""In-tree LMCache connector integration.

The submodules that talk to the external lmcache package are imported lazily
so that the pure helpers in ``metadata`` stay importable without lmcache.
"""

import importlib
from typing import TYPE_CHECKING, Any

from . import metadata

if TYPE_CHECKING:
    from . import multi_process_adapter, vllm_v1_adapter
    from .multi_process_adapter import (
        LMCacheMPSchedulerAdapter,
        LMCacheMPWorkerAdapter,
        ParallelStrategy,
    )

__all__ = [
    "vllm_v1_adapter",
    "multi_process_adapter",
    "metadata",
    "LMCacheMPSchedulerAdapter",
    "LMCacheMPWorkerAdapter",
    "LoadStoreOp",
    "ParallelStrategy",
]

_MP_ADAPTER_NAMES = {
    "LMCacheMPSchedulerAdapter",
    "LMCacheMPWorkerAdapter",
    "ParallelStrategy",
}


def __getattr__(name: str) -> Any:
    if name in ("vllm_v1_adapter", "multi_process_adapter"):
        return importlib.import_module(f".{name}", __name__)
    if name in _MP_ADAPTER_NAMES:
        module = importlib.import_module(".multi_process_adapter", __name__)
        return getattr(module, name)
    if name == "LoadStoreOp":
        return metadata.LoadStoreOp
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
