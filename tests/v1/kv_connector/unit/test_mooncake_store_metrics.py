# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from prometheus_client import Counter, Gauge, Histogram

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_store_metrics import (  # noqa: E501
    MooncakeStorePromMetrics,
)

from .utils import create_vllm_config


class _FakeMetricChild:
    def __init__(self):
        self.observations: list[float] = []
        self.increments: list[float] = []

    def observe(self, value: float):
        self.observations.append(value)

    def inc(self, value: float = 1):
        self.increments.append(value)


class _FakeMetric:
    def __init__(self, *args, **kwargs):
        self.children: dict[tuple[object, ...], _FakeMetricChild] = {}

    def labels(self, *labelvalues):
        if labelvalues not in self.children:
            self.children[labelvalues] = _FakeMetricChild()
        return self.children[labelvalues]


def test_mooncake_store_prom_metrics_observe_records_put_failures():
    prom_metrics = MooncakeStorePromMetrics(
        create_vllm_config(
            kv_connector="MooncakeStoreConnector",
            kv_role="kv_both",
        ),
        {
            Gauge: _FakeMetric,
            Counter: _FakeMetric,
            Histogram: _FakeMetric,
        },
        ["model_name", "engine_index"],
        {0: ["test-model", "0"]},
    )

    prom_metrics.observe(
        {
            "mooncake_store_put_duration": [0.25],
            "mooncake_store_put_bytes": [1024],
            "mooncake_store_get_duration": [0.5],
            "mooncake_store_get_bytes": [2048],
            "mooncake_store_put_failed_batches": [1, 1],
            "mooncake_store_put_failed_keys": [2, 1],
            "mooncake_store_put_transfer_fail_keys": [2, 0],
            "mooncake_store_put_no_available_handle_keys": [0, 1],
            "mooncake_store_put_other_failed_keys": [0, 0],
        }
    )

    assert prom_metrics.hist_put_duration[0].observations == [0.25]
    assert prom_metrics.hist_put_bytes[0].observations == [1024]
    assert prom_metrics.hist_get_duration[0].observations == [0.5]
    assert prom_metrics.hist_get_bytes[0].observations == [2048]
    assert prom_metrics.counter_put_failed_batches[0].increments == [1, 1]
    assert prom_metrics.counter_put_failed_keys[0].increments == [2, 1]
    assert prom_metrics.counter_put_transfer_fail_keys[0].increments == [2, 0]
    assert prom_metrics.counter_put_no_available_handle_keys[0].increments == [0, 1]
    assert prom_metrics.counter_put_other_failed_keys[0].increments == [0, 0]
