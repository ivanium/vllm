# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorPromMetrics,
    KVConnectorStats,
    PromMetric,
    PromMetricT,
)
from vllm.v1.metrics.utils import create_metric_per_engine

_BYTES_PER_MIB = 2**20


@dataclass
class MooncakeStoreConnectorStats(KVConnectorStats):
    """Transfer stats for MooncakeStoreConnector.

    Records per-transfer (bytes, duration, transfer_type) tuples.
    transfer_type is either "mooncake_store_put" or "mooncake_store_get".
    """

    def __post_init__(self):
        if not self.data:
            self.reset()

    def reset(self):
        self.data: dict[str, list] = {
            "mooncake_store_put_bytes": [],
            "mooncake_store_put_duration": [],
            "mooncake_store_get_bytes": [],
            "mooncake_store_get_duration": [],
        }

    def record_put(self, num_bytes: int, time_s: float):
        self.data["mooncake_store_put_bytes"].append(num_bytes)
        self.data["mooncake_store_put_duration"].append(time_s)

    def record_get(self, num_bytes: int, time_s: float):
        self.data["mooncake_store_get_bytes"].append(num_bytes)
        self.data["mooncake_store_get_duration"].append(time_s)

    def clone_and_reset(self) -> MooncakeStoreConnectorStats:
        snapshot = MooncakeStoreConnectorStats(data=copy.deepcopy(self.data))
        self.reset()
        return snapshot

    def aggregate(
        self, other: KVConnectorStats
    ) -> MooncakeStoreConnectorStats:
        if other.is_empty():
            return self
        for key in self.data:
            if key in other.data:
                self.data[key].extend(other.data[key])
        return self

    def is_empty(self) -> bool:
        return all(len(v) == 0 for v in self.data.values())

    @staticmethod
    def _percentile(sorted_vals: list[float], p: float) -> float:
        if not sorted_vals:
            return 0.0
        idx = int(len(sorted_vals) * p / 100)
        idx = min(idx, len(sorted_vals) - 1)
        return sorted_vals[idx]

    def reduce(self) -> dict[str, int | float]:
        reduced: dict[str, int | float] = {}
        for direction in ("put", "get"):
            bytes_key = f"mooncake_store_{direction}_bytes"
            dur_key = f"mooncake_store_{direction}_duration"
            bytes_list = self.data.get(bytes_key, [])
            dur_list = self.data.get(dur_key, [])

            n = len(bytes_list)
            if n == 0:
                continue

            total_bytes = sum(bytes_list)
            total_time = sum(dur_list)
            total_mib = total_bytes / _BYTES_PER_MIB

            reduced[f"mooncake_store_{direction}_num_transfers"] = n
            reduced[f"mooncake_store_{direction}_total_bytes"] = total_bytes
            reduced[f"mooncake_store_{direction}_avg_batch_mib"] = round(
                total_mib / n, 3
            )
            reduced[f"mooncake_store_{direction}_avg_time_ms"] = round(
                (total_time / n) * 1e3, 3
            )
            reduced[f"mooncake_store_{direction}_throughput_mib_s"] = round(
                total_mib / total_time if total_time > 0 else 0.0, 3
            )

            # Per-transfer throughput percentiles (MiB/s)
            per_xfer_tp = sorted(
                (b / _BYTES_PER_MIB) / t if t > 0 else 0.0
                for b, t in zip(bytes_list, dur_list)
            )
            reduced[f"mooncake_store_{direction}_tp_p50_mib_s"] = round(
                self._percentile(per_xfer_tp, 50), 3
            )
            reduced[f"mooncake_store_{direction}_tp_p95_mib_s"] = round(
                self._percentile(per_xfer_tp, 95), 3
            )
            reduced[f"mooncake_store_{direction}_tp_p99_mib_s"] = round(
                self._percentile(per_xfer_tp, 99), 3
            )

            # Per-transfer latency percentiles (ms)
            sorted_dur_ms = sorted(t * 1e3 for t in dur_list)
            reduced[f"mooncake_store_{direction}_lat_p50_ms"] = round(
                self._percentile(sorted_dur_ms, 50), 3
            )
            reduced[f"mooncake_store_{direction}_lat_p95_ms"] = round(
                self._percentile(sorted_dur_ms, 95), 3
            )
            reduced[f"mooncake_store_{direction}_lat_p99_ms"] = round(
                self._percentile(sorted_dur_ms, 99), 3
            )
        return reduced


class MooncakeStorePromMetrics(KVConnectorPromMetrics):
    """Prometheus metrics for MooncakeStoreConnector put/get transfers."""

    def __init__(
        self,
        vllm_config: Any,
        metric_types: dict[type[PromMetric], type[PromMetricT]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ):
        super().__init__(
            vllm_config, metric_types, labelnames, per_engine_labelvalues
        )

        duration_buckets = [
            0.0005, 0.001, 0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5,
            1.0, 5.0,
        ]
        bytes_buckets = [2 ** (10 + i) for i in range(1, 25, 2)]

        for direction in ("put", "get"):
            hist_dur = self._histogram_cls(
                name=f"vllm:mooncake_store_{direction}_duration_seconds",
                documentation=(
                    f"Histogram of Mooncake Store {direction} transfer "
                    f"duration in seconds."
                ),
                buckets=duration_buckets,
                labelnames=labelnames,
            )
            setattr(
                self,
                f"hist_{direction}_duration",
                create_metric_per_engine(hist_dur, self.per_engine_labelvalues),
            )

            hist_bytes = self._histogram_cls(
                name=f"vllm:mooncake_store_{direction}_bytes",
                documentation=(
                    f"Histogram of bytes transferred per Mooncake Store "
                    f"{direction} operation."
                ),
                buckets=bytes_buckets,
                labelnames=labelnames,
            )
            setattr(
                self,
                f"hist_{direction}_bytes",
                create_metric_per_engine(
                    hist_bytes, self.per_engine_labelvalues
                ),
            )

    def observe(
        self, transfer_stats_data: dict[str, Any], engine_idx: int = 0
    ):
        for direction in ("put", "get"):
            dur_key = f"mooncake_store_{direction}_duration"
            bytes_key = f"mooncake_store_{direction}_bytes"
            dur_hist = getattr(self, f"hist_{direction}_duration")
            bytes_hist = getattr(self, f"hist_{direction}_bytes")

            for val in transfer_stats_data.get(dur_key, []):
                dur_hist[engine_idx].observe(val)
            for val in transfer_stats_data.get(bytes_key, []):
                bytes_hist[engine_idx].observe(val)
