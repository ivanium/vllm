# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Minimal Mooncake requester config helpers."""

import os
import re
import socket
import urllib.request
from collections.abc import Mapping
from typing import Any

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_SEGMENT_METRIC_RE = re.compile(
    r'^segment_total_capacity_bytes\{segment="([^"]+)"\}\s+', re.MULTILINE
)
_AUTO_DETECT_TIMEOUT_SECONDS = 2.0


def normalize_string_override(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def get_current_physical_gpu_index() -> int | None:
    try:
        from vllm.platforms import current_platform
    except ImportError:
        return None

    try:
        device_index = torch.accelerator.current_device_index()
        physical_device_id = current_platform.device_id_to_physical_device_id(
            device_index
        )
        return int(physical_device_id)
    except Exception:
        return None


def get_requester_local_hostname(local_ip: str) -> str:
    override = normalize_string_override(os.getenv("MOONCAKE_LOCAL_HOSTNAME"))
    if override is not None:
        return override
    return local_ip


def _enumerate_local_ipv4_addresses() -> set[str]:
    """Return all local non-loopback IPv4 addresses across every NIC."""
    try:
        import psutil
    except ImportError:
        return set()
    addresses: set[str] = set()
    try:
        for _, addrs in psutil.net_if_addrs().items():
            for addr in addrs:
                if addr.family != socket.AF_INET:
                    continue
                if not addr.address or addr.address.startswith("127."):
                    continue
                addresses.add(addr.address)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Mooncake could not enumerate local IPs: %s", exc)
    return addresses


def _fetch_master_segments(metrics_url: str) -> list[str]:
    """Pull `segment_total_capacity_bytes` labels from the master /metrics page."""
    try:
        with urllib.request.urlopen(  # noqa: S310 - operator-controlled URL
            metrics_url, timeout=_AUTO_DETECT_TIMEOUT_SECONDS
        ) as resp:
            text = resp.read().decode("utf-8", errors="replace")
    except Exception as exc:
        logger.debug(
            "Mooncake could not fetch master metrics from %s: %s", metrics_url, exc
        )
        return []
    return _SEGMENT_METRIC_RE.findall(text)


def _auto_detect_preferred_segment(metrics_url: str) -> str | None:
    """Pick the master segment whose host matches a local IP, or None."""
    local_ips = _enumerate_local_ipv4_addresses()
    if not local_ips:
        return None
    for segment in _fetch_master_segments(metrics_url):
        host, _, _ = segment.partition(":")
        if host in local_ips:
            logger.info(
                "Mooncake auto-detected local owner segment %s from master "
                "/metrics (matched local IP %s)",
                segment,
                host,
            )
            return segment
    return None


def get_configured_preferred_segment(
    extra_config: Mapping[str, Any],
) -> str | None:
    preferred_segment = normalize_string_override(extra_config.get("preferred_segment"))
    if preferred_segment is not None:
        return preferred_segment
    if extra_config.get("preferred_segment") is not None:
        raise ValueError(
            "Mooncake preferred_segment override must be a non-empty string"
        )

    # Auto-detect: query master /metrics, find the segment whose host matches
    # any local IP. This is best-effort — failures fall back to None (random
    # allocator), so the worker still functions in deployments without an
    # accessible master HTTP endpoint.
    metrics_url = normalize_string_override(
        extra_config.get("master_metrics_url")
    ) or normalize_string_override(os.getenv("MOONCAKE_MASTER_METRICS_URL"))
    if metrics_url is None:
        return None
    return _auto_detect_preferred_segment(metrics_url)


def _get_explicit_worker_rnic(device_list: str) -> str:
    entries = [entry.strip() for entry in device_list.split(",")]
    if any(not entry for entry in entries):
        raise ValueError(
            "Mooncake worker device_name contains an empty RDMA device entry"
        )
    if len(entries) == 1:
        return entries[0]

    gpu_index = get_current_physical_gpu_index()
    if gpu_index is None:
        raise RuntimeError(
            "Mooncake RDMA requester could not determine the local physical GPU index"
        )
    if gpu_index >= len(entries):
        raise ValueError(
            "Mooncake worker device list does not cover local GPU "
            f"{gpu_index}: {device_list}"
        )
    device_name = entries[gpu_index]
    logger.info(
        "Mooncake selected worker RNIC %s from explicit device list for local GPU %s",
        device_name,
        gpu_index,
    )
    return device_name


def get_configured_worker_rnic(
    *,
    protocol: str,
    configured_device: str,
) -> str:
    normalized_device = normalize_string_override(configured_device)
    if normalized_device is not None:
        return _get_explicit_worker_rnic(normalized_device)

    if protocol not in {"rdma", "efa"}:
        return ""

    logger.warning(
        "Mooncake requester has no explicit worker RNIC configured; falling "
        "back to Mooncake auto-selection, which may be sub-optimal."
    )
    return ""
