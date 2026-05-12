# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Minimal Mooncake requester config helpers.

Includes per-GPU RDMA NIC auto-discovery so each vLLM DP rank pins to its
PCI-affine RNIC when the operator did not provide an explicit
``MOONCAKE_DEVICE=<csv>`` mapping. Without this, all DP ranks on a single
node would converge on whichever NIC Mooncake's transfer engine
auto-selects first, saturating one link and stalling under offload load.
"""

import functools
import os
import re
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_PREFERRED_SEGMENT_ENV = "MOONCAKE_PREFERRED_SEGMENT"
_GID_INDEX_ENV = "MC_GID_INDEX"
_INFINIBAND_ROOT = Path("/sys/class/infiniband")


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

    env_value = normalize_string_override(os.getenv(_PREFERRED_SEGMENT_ENV))
    if env_value is not None:
        logger.info(
            "Mooncake preferred_segment from %s: %s",
            _PREFERRED_SEGMENT_ENV,
            env_value,
        )
        return env_value
    return None


# ---------------------------------------------------------------------------
# Per-GPU RDMA NIC discovery
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _GpuInfo:
    index: int
    pci_bus: str  # normalized "DDDD:BB:DD.F"
    numa: int  # -1 if unknown


@dataclass(frozen=True)
class _RdmaDevice:
    name: str
    pci_bus: str
    numa: int
    netdevs: tuple[str, ...]


def _normalize_pci_bus(raw: str) -> str | None:
    """Return PCI bus id as ``DDDD:BB:DD.F`` (lowercase, 4-digit domain) or None.

    nvidia-smi emits 8-digit domains (e.g. ``00000000:89:00.0``) while sysfs
    paths use 4-digit (``0000:89:00.0``). Normalise to the 4-digit form.
    """
    if not raw:
        return None
    raw = raw.strip().lower()
    m = re.match(
        r"^(?:([0-9a-f]{4,8}):)?([0-9a-f]{2}):([0-9a-f]{2})\.([0-7])$", raw
    )
    if not m:
        return None
    domain_raw = m.group(1) or "0"
    # Strip leading zeros, then zero-pad to exactly 4 hex chars.
    domain = domain_raw.lstrip("0").zfill(4) or "0000"
    if len(domain) > 4:  # genuine non-zero 8-digit domain — leave intact
        return None
    return f"{domain}:{m.group(2)}:{m.group(3)}.{m.group(4)}"


def _pci_bus_distance(lhs: str, rhs: str) -> int:
    """Cheap distance between two PCI bus ids — abs(lhs_bus - rhs_bus) within
    same domain; if domains differ, large distance. Matches the bash impl."""
    try:
        lhs_dom, lhs_bus, _ = lhs.split(":", 2)
        rhs_dom, rhs_bus, _ = rhs.split(":", 2)
    except ValueError:
        return 1 << 30
    if lhs_dom != rhs_dom:
        return 1 << 30
    try:
        return abs(int(lhs_bus, 16) - int(rhs_bus, 16))
    except ValueError:
        return 1 << 30


def _read_numa(pci_bus: str) -> int:
    try:
        with open(f"/sys/bus/pci/devices/{pci_bus}/numa_node") as f:
            return int(f.read().strip())
    except OSError:
        return -1


def _gid_index_active(device: Path, port: str, gid_index: str) -> bool:
    """Return True if the device/port has a non-zero GID at the requested index."""
    gid_path = device / "ports" / port / "gids" / gid_index
    try:
        gid = gid_path.read_text().strip()
    except OSError:
        return False
    return bool(gid) and any(c not in "0:" for c in gid)


def _port_is_active(device: Path) -> bool:
    for port_path in (device / "ports").glob("*"):
        try:
            state = (port_path / "state").read_text()
        except OSError:
            continue
        # "4: ACTIVE" or "5: ACTIVE_DEFER" etc.; just check for "ACTIVE"
        if "ACTIVE" in state:
            return True
    return False


def _device_active_for_gid(device: Path, gid_index: str | None) -> bool:
    """Match the bash `device_is_active_for_gid_index` predicate."""
    for port_path in (device / "ports").glob("*"):
        try:
            state = (port_path / "state").read_text()
        except OSError:
            continue
        if "ACTIVE" not in state:
            continue
        if gid_index is None:
            return True
        if _gid_index_active(device, port_path.name, gid_index):
            return True
    return False


def _discover_local_gpus() -> list[_GpuInfo]:
    if not subprocess.run(
        ["which", "nvidia-smi"], capture_output=True, check=False
    ).stdout:
        return []
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,pci.bus_id",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout
    except (subprocess.SubprocessError, FileNotFoundError):
        return []

    gpus: list[_GpuInfo] = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        idx_str, bus_raw = (p.strip() for p in line.split(",", 1))
        try:
            idx = int(idx_str)
        except ValueError:
            continue
        bus = _normalize_pci_bus(bus_raw)
        if bus is None:
            continue
        gpus.append(_GpuInfo(index=idx, pci_bus=bus, numa=_read_numa(bus)))
    return gpus


def _discover_rdma_devices(gid_index: str | None) -> list[_RdmaDevice]:
    if not _INFINIBAND_ROOT.is_dir():
        return []

    devices: list[_RdmaDevice] = []
    for device_path in sorted(_INFINIBAND_ROOT.iterdir()):
        if not device_path.is_dir():
            continue
        if not _device_active_for_gid(device_path, gid_index):
            continue
        try:
            pci_link = (device_path / "device").resolve().name
        except OSError:
            continue
        pci_bus = _normalize_pci_bus(pci_link)
        if pci_bus is None:
            continue
        netdevs = tuple(
            sorted(
                p.name
                for p in (device_path / "device" / "net").glob("*")
                if p.is_dir()
            )
        )
        devices.append(
            _RdmaDevice(
                name=device_path.name,
                pci_bus=pci_bus,
                numa=_read_numa(pci_bus),
                netdevs=netdevs,
            )
        )
    return devices


def _gpu_matches_rnic_netdev(gpu_index: int, netdevs: tuple[str, ...]) -> bool:
    """Match the bash `gpu_matches_rnic_netdev` — netdev named like ``gpu0rdma0``."""
    pattern = re.compile(rf"^gpu{gpu_index}rdma[0-9]+$")
    return any(pattern.match(nd) for nd in netdevs)


def _assign_rnic_per_gpu(
    gpus: list[_GpuInfo], rnics: list[_RdmaDevice]
) -> dict[int, str]:
    """Map gpu_index → rnic_name. 3-phase: exact netdev → same-NUMA min PCI →
    global min PCI. Each RNIC is used by at most one GPU.

    Returns a partial map if any phase can't find a match; caller decides what
    to do with un-mapped GPUs.
    """
    by_name = {r.name: r for r in rnics}
    used: set[str] = set()
    mapping: dict[int, str] = {}

    # Phase A — exact netdev (gpu<N>rdma<M>) match
    for gpu in gpus:
        matches = [r for r in rnics if _gpu_matches_rnic_netdev(gpu.index, r.netdevs)]
        if not matches:
            continue
        chosen = sorted(m.name for m in matches)[0]
        mapping[gpu.index] = chosen
        used.add(chosen)

    # Phase B — among GPUs without a phase-A match, prefer same-NUMA min PCI distance
    for gpu in gpus:
        if gpu.index in mapping:
            continue
        candidates = [
            r for r in rnics if r.name not in used and r.numa == gpu.numa and gpu.numa != -1
        ]
        if not candidates:
            continue
        candidates.sort(
            key=lambda r: (_pci_bus_distance(gpu.pci_bus, r.pci_bus), r.name)
        )
        chosen = candidates[0].name
        mapping[gpu.index] = chosen
        used.add(chosen)

    # Phase C — global min PCI distance
    for gpu in gpus:
        if gpu.index in mapping:
            continue
        candidates = [r for r in rnics if r.name not in used]
        if not candidates:
            continue
        candidates.sort(
            key=lambda r: (_pci_bus_distance(gpu.pci_bus, r.pci_bus), r.name)
        )
        chosen = candidates[0].name
        mapping[gpu.index] = chosen
        used.add(chosen)

    _ = by_name  # silence unused
    return mapping


@functools.lru_cache(maxsize=1)
def _autodetect_gpu_rnic_map() -> dict[int, str]:
    """Cache the GPU→RNIC assignment for the lifetime of the process."""
    gid_index = normalize_string_override(os.getenv(_GID_INDEX_ENV))
    gpus = _discover_local_gpus()
    rnics = _discover_rdma_devices(gid_index)
    if not gpus or not rnics:
        return {}
    mapping = _assign_rnic_per_gpu(gpus, rnics)
    if mapping:
        logger.info(
            "Mooncake RDMA auto-discovery: %s (gid_index=%s)",
            ", ".join(f"gpu{i}->{n}" for i, n in sorted(mapping.items())),
            gid_index or "any",
        )
    return mapping


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


def _autodiscover_worker_rnic() -> str:
    """Fallback when the operator did not provide a ``MOONCAKE_DEVICE`` CSV.

    Maps the calling rank's local GPU to its PCI-affine RNIC by reading
    /sys/class/infiniband and ``nvidia-smi``. Returns "" if no mapping can
    be made (Mooncake's transfer engine will then fall back to its own
    auto-selection).
    """
    gpu_index = get_current_physical_gpu_index()
    if gpu_index is None:
        return ""
    mapping = _autodetect_gpu_rnic_map()
    rnic = mapping.get(gpu_index, "")
    if rnic:
        logger.info(
            "Mooncake auto-discovered worker RNIC %s for local GPU %d",
            rnic,
            gpu_index,
        )
    return rnic


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

    autodiscovered = _autodiscover_worker_rnic()
    if autodiscovered:
        return autodiscovered

    logger.warning(
        "Mooncake requester has no explicit worker RNIC configured and "
        "Python auto-discovery could not map the local GPU to an RNIC; "
        "falling back to Mooncake auto-selection, which may be sub-optimal."
    )
    return ""
