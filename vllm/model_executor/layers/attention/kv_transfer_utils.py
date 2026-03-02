# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import inspect
import os
import time
from collections.abc import Callable
from functools import wraps

from vllm.distributed.kv_transfer import (
    get_kv_transfer_group,
    has_kv_transfer_group,
    is_v1_kv_transfer_group,
)
from vllm.logger import init_logger

logger = init_logger(__name__)

# ── Per-layer profiling ──────────────────────────────────────────────
_PROFILE_KV_LAYER = os.environ.get("VLLM_PROFILE_KV_CONNECTOR", "0") == "1"
_PROFILE_INTERVAL = int(os.environ.get("VLLM_PROFILE_KV_INTERVAL", "100"))

# Accumulated timings (nanoseconds) reset after every _PROFILE_INTERVAL steps
_layer_stats: dict[str, int] = {}  # layer_name -> total overhead ns
_layer_call_count: int = 0  # total wrapper invocations since last report
_step_count: int = 0  # steps since last report
_step_overhead_ns: int = 0  # total per-layer overhead across the current step


def _report_layer_stats(num_steps: int = 0) -> None:
    """Log accumulated per-layer connector overhead and reset counters.

    Args:
        num_steps: Number of steps since last report (passed from worker
            profiler). Falls back to _step_count if 0.
    """
    global _layer_stats, _layer_call_count, _step_count, _step_overhead_ns
    if not _layer_stats:
        return
    steps = num_steps or _step_count or 1
    total_us = sum(_layer_stats.values()) / 1000
    avg_per_step_us = total_us / steps
    avg_per_call_us = total_us / max(_layer_call_count, 1)
    n_layers = len(_layer_stats)
    logger.info(
        "[KVConnector layer profiler] last %d steps: "
        "total_overhead=%.1f us, per_step=%.1f us, "
        "per_call=%.3f us (%d layers, %d calls)",
        steps,
        total_us,
        avg_per_step_us,
        avg_per_call_us,
        n_layers,
        _layer_call_count,
    )
    _layer_stats = {}
    _layer_call_count = 0
    _step_count = 0
    _step_overhead_ns = 0


def maybe_transfer_kv_layer(func: Callable) -> Callable:
    """Decorator that handles KV layer transfer prior and after execution of
    an attention layer, if enabled. Otherwise, the wrapper is a no-op.

    On entry: waits for the KV layer from the connector.
    On exit: saves the KV layer to the connector.
    """
    # Import at runtime to avoid circular dependency
    from vllm.model_executor.layers.attention.attention import get_attention_context

    # Inspect the signature ONCE when the decorator is applied.
    sig = inspect.signature(func)
    param_names = list(sig.parameters.keys())

    # Find the index of 'layer_name' parameter.
    try:
        layer_name_index = param_names.index("layer_name")
    except ValueError as e:
        raise TypeError(
            f"Function {func.__name__} must have a 'layer_name' parameter"
        ) from e

    @wraps(func)
    def wrapper(*args, **kwargs):
        if not has_kv_transfer_group() or not is_v1_kv_transfer_group():
            return func(*args, **kwargs)

        if _PROFILE_KV_LAYER:
            t0 = time.perf_counter_ns()

        layer_name: str = args[layer_name_index]

        # Extract attention context (metadata, layer, kv_cache, layer_slot_mapping)
        attn_metadata, _, kv_cache, _ = get_attention_context(layer_name)
        connector = get_kv_transfer_group()
        if attn_metadata is None or not connector.has_connector_metadata():
            if _PROFILE_KV_LAYER:
                _record_layer_time(layer_name, time.perf_counter_ns() - t0)
            return func(*args, **kwargs)

        # Wait for KV layer on entry
        connector.wait_for_layer_load(layer_name)

        if _PROFILE_KV_LAYER:
            # Exclude the actual attention computation from overhead timing.
            overhead_before = time.perf_counter_ns() - t0

        # Execute the function
        result = func(*args, **kwargs)

        if _PROFILE_KV_LAYER:
            t1 = time.perf_counter_ns()

        # Save KV cache layer on exit
        connector.save_kv_layer(layer_name, kv_cache, attn_metadata)

        if _PROFILE_KV_LAYER:
            overhead_after = time.perf_counter_ns() - t1
            _record_layer_time(layer_name, overhead_before + overhead_after)

        return result

    return wrapper


def _record_layer_time(layer_name: str, elapsed_ns: int) -> None:
    """Accumulate per-layer overhead (only called when profiling enabled)."""
    global _layer_call_count
    _layer_stats[layer_name] = _layer_stats.get(layer_name, 0) + elapsed_ns
    _layer_call_count += 1
