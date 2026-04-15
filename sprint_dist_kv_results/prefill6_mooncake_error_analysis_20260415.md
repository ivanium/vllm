# Prefill-6 Mooncake Error Analysis

Date: 2026-04-15

Scope:
- Benchmark run: `bench_results/pd_kimi_nsys_prefill6_light/prefill_6_mtc_18`
- Main error types:
  - `handshake timeout`
  - `transfer fail`
  - `batch_put failed`
- Supporting signals:
  - `received packet mismatch`
  - `mark it inactive`

Notes:
- All file paths below are repo-relative paths.
- All line numbers below are based on the local log copies in this workspace on 2026-04-15.
- This document summarizes the current logs plus local source inspection of vLLM and Mooncake.

## Executive Summary

These are more likely not three independent bugs. The most plausible chain is:

`startup handshake/path mismatch -> endpoint marked inactive -> later writes hit a bad endpoint -> Mooncake waits 60s and returns TRANSFER_FAIL(-800) -> vLLM logs batch_put failed`

Current best judgment:
- The most likely root cause is in Mooncake RDMA connection establishment or endpoint state management.
- `transfer fail` is more likely a downstream symptom of the startup connection issue than a separate root cause.
- `batch_put failed` in this run is a Python-side aggregation of lower-level `TRANSFER_FAIL(-800)` results, not a separate root cause.

## Error Count Summary

| Category | Count | Timing / Meaning |
| --- | ---: | --- |
| `handshake timeout` | 27 | All in startup window `2026-04-14 09:15:48-09:15:49` |
| `packet mismatch` | 14 | Supporting signal that handshake descriptors or endpoint paths mismatched |
| `inactive endpoint` | 27 | Endpoint setup failed and worker marked the endpoint inactive |
| `transfer timeout` | 9 | `Failed to complete transfers after 60 seconds` |
| `transfer fail key` | 9 | `Transfer failed for key ... TRANSFER_FAIL` |
| `operation fail` | 9 | `Operation for key ... failed: TRANSFER_FAIL` |
| `batch_put warn` | 6 | Python-side batch warnings covering 9 failed keys |

## 1. Handshake Timeout

### 1.1 Distribution by Log File

| Log file | Count | Representative location |
| --- | ---: | --- |
| `bench_results/pd_kimi_nsys_prefill6_light/prefill_6_mtc_18/prefill-1/prefill-1-gb200-rack1-04.log` | 17 | `:1340` |
| `bench_results/pd_kimi_nsys_prefill6_light/prefill_6_mtc_18/prefill-0/prefill-0-gb200-rack1-03.log` | 5 | `:1286` |
| `bench_results/pd_kimi_nsys_prefill6_light/prefill_6_mtc_18/prefill-3/prefill-3-gb200-rack1-08.log` | 3 | `:1912` |
| `bench_results/pd_kimi_nsys_prefill6_light/prefill_6_mtc_18/prefill-4/prefill-4-gb200-rack1-10.log` | 1 | `:1741` |
| `bench_results/pd_kimi_nsys_prefill6_light/prefill_6_mtc_18/prefill-5/prefill-5-gb200-rack1-11.log` | 1 | `:1583` |

### 1.2 Representative Log Locations

- `bench_results/pd_kimi_nsys_prefill6_light/prefill_6_mtc_18/prefill-1/prefill-1-gb200-rack1-04.log:1340`
- `bench_results/pd_kimi_nsys_prefill6_light/prefill_6_mtc_18/prefill-1/prefill-1-gb200-rack1-04.log:1372`
- `bench_results/pd_kimi_nsys_prefill6_light/prefill_6_mtc_18/prefill-0/prefill-0-gb200-rack1-03.log:1286`
- `bench_results/pd_kimi_nsys_prefill6_light/prefill_6_mtc_18/prefill-3/prefill-3-gb200-rack1-08.log:1912`
- `bench_results/pd_kimi_nsys_prefill6_light/prefill_6_mtc_18/prefill-4/prefill-4-gb200-rack1-10.log:1741`

### 1.3 Supporting Signals Around Handshake Failure

Representative `packet mismatch` lines:
- `bench_results/pd_kimi_nsys_prefill6_light/prefill_6_mtc_18/prefill-1/prefill-1-gb200-rack1-04.log:1364`
- `bench_results/pd_kimi_nsys_prefill6_light/prefill_6_mtc_18/prefill-1/prefill-1-gb200-rack1-04.log:1368`
- `bench_results/pd_kimi_nsys_prefill6_light/prefill_6_mtc_18/prefill-4/prefill-4-gb200-rack1-10.log:1739`

Representative `mark it inactive` lines:
- `bench_results/pd_kimi_nsys_prefill6_light/prefill_6_mtc_18/prefill-1/prefill-1-gb200-rack1-04.log:1351`
- `bench_results/pd_kimi_nsys_prefill6_light/prefill_6_mtc_18/prefill-1/prefill-1-gb200-rack1-04.log:1381`
- `bench_results/pd_kimi_nsys_prefill6_light/prefill_6_mtc_18/prefill-3/prefill-3-gb200-rack1-08.log:1916`

Top hosts in `inactive endpoint` logs:

| Host | Count |
| --- | ---: |
| `192.168.0.104` | 11 |
| `192.168.0.107` | 4 |
| `192.168.0.108` | 4 |
| `192.168.0.103` | 4 |
| `192.168.0.110` | 3 |
| `192.168.0.111` | 1 |

### 1.4 Analysis

Most likely causes, ordered by probability:

1. `active/passive simultaneous open` race or stale endpoint state during connection setup.
   - Mooncake source explicitly has simultaneous-open handling and connection reuse / re-establish logic.
   - Logs also contain `Received same peer QP numbers, reusing connection.` and `Re-establish connection`.

2. Multi-NIC path mismatch or endpoint advertisement mismatch.
   - Many `packet mismatch` logs show `peer.local_nic_path` and `peer.peer_nic_path` as empty.
   - That pattern looks more like “wrong handshake packet / wrong endpoint state” than a pure network outage.

3. A node- or NIC-local problem centered around host `192.168.0.104`.
   - `192.168.0.104` appears in `11/27` inactive-endpoint events and is the strongest hotspot.

4. Real RDMA parameter mismatch such as `MTU / GID / LID / QP`.
   - The old log format does not include enough fields to prove or disprove this.
   - This is now instrumented in local Mooncake source for the next rerun.

## 2. Transfer Fail

### 2.1 Meaning

The `transfer fail` cluster is more accurately:
- 9 lines of `Failed to complete transfers after 60 seconds for batch ...`
- 9 lines of `Transfer failed for key ...: TRANSFER_FAIL`
- 9 lines of `Operation for key ... failed: TRANSFER_FAIL`

This is a hard 60-second timeout in Mooncake transfer wait logic, not a random Python exception.

### 2.2 All Timeout Batches and Failed Keys

| Worker log | Batch timeout line | Batch ID | Key fail line | Failed key suffix |
| --- | --- | ---: | --- | --- |
| `prefill-0-gb200-rack1-03.log` | `:2163` | `80299849924864` | `:2164` | `16a380262664...` |
| `prefill-0-gb200-rack1-03.log` | `:2218` | `97737619846512` | `:2219` | `ac5bfcd52c34...` |
| `prefill-1-gb200-rack1-04.log` | `:2338` | `75355604427936` | `:2339` | `e4e2a74e4315...` |
| `prefill-1-gb200-rack1-04.log` | `:2340` | `82376130695408` | `:2341` | `9e80ff557a31...` |
| `prefill-1-gb200-rack1-04.log` | `:3100` | `75355605178320` | `:3101` | `e2fdd361cadd...` |
| `prefill-1-gb200-rack1-04.log` | `:3109` | `82376131834192` | `:3110` | `78548e37244b...` |
| `prefill-2-gb200-rack1-07.log` | `:2585` | `77278877289136` | `:2586` | `67fa0fb5de24...` |
| `prefill-3-gb200-rack1-08.log` | `:2941` | `103858685124736` | `:2942` | `3c35f017697c...` |
| `prefill-3-gb200-rack1-08.log` | `:3700` | `103858686344336` | `:3701` | `2800bcca4333...` |

Full `Transfer failed for key ...` log locations:
- `bench_results/pd_kimi_nsys_prefill6_light/prefill_6_mtc_18/prefill-0/prefill-0-gb200-rack1-03.log:2164`
- `bench_results/pd_kimi_nsys_prefill6_light/prefill_6_mtc_18/prefill-0/prefill-0-gb200-rack1-03.log:2219`
- `bench_results/pd_kimi_nsys_prefill6_light/prefill_6_mtc_18/prefill-1/prefill-1-gb200-rack1-04.log:2339`
- `bench_results/pd_kimi_nsys_prefill6_light/prefill_6_mtc_18/prefill-1/prefill-1-gb200-rack1-04.log:2341`
- `bench_results/pd_kimi_nsys_prefill6_light/prefill_6_mtc_18/prefill-1/prefill-1-gb200-rack1-04.log:3101`
- `bench_results/pd_kimi_nsys_prefill6_light/prefill_6_mtc_18/prefill-1/prefill-1-gb200-rack1-04.log:3110`
- `bench_results/pd_kimi_nsys_prefill6_light/prefill_6_mtc_18/prefill-2/prefill-2-gb200-rack1-07.log:2586`
- `bench_results/pd_kimi_nsys_prefill6_light/prefill_6_mtc_18/prefill-3/prefill-3-gb200-rack1-08.log:2942`
- `bench_results/pd_kimi_nsys_prefill6_light/prefill_6_mtc_18/prefill-3/prefill-3-gb200-rack1-08.log:3701`

### 2.3 Analysis

Most likely causes:

1. The transfer is hitting an endpoint or replica that was already broken during startup.
   - Transfer failures happen after the handshake failures, not before.
   - They are concentrated in `prefill-0/1/2/3`, not across every worker.

2. The write was submitted, but no completion arrived.
   - Mooncake waits for 60 seconds and then returns `TRANSFER_FAIL`.
   - This looks like a half-open connection, stale endpoint, or bad replica path.

3. This is not consistent with offload pressure.
   - The error code here is `-800 = TRANSFER_FAIL`.
   - The pressure / handle-exhaustion code would have been `-200 = NO_AVAILABLE_HANDLE`, which does not appear in these failures.

## 3. Batch Put Failed

### 3.1 All Python-side Batch Warnings

| Warning location | Failed keys in batch | Note |
| --- | ---: | --- |
| `bench_results/pd_kimi_nsys_prefill6_light/prefill_6_mtc_18/prefill-0/prefill-0-gb200-rack1-03.log:2167` | 1 | `codes={-800}` |
| `bench_results/pd_kimi_nsys_prefill6_light/prefill_6_mtc_18/prefill-0/prefill-0-gb200-rack1-03.log:2222` | 1 | `codes={-800}` |
| `bench_results/pd_kimi_nsys_prefill6_light/prefill_6_mtc_18/prefill-1/prefill-1-gb200-rack1-04.log:3106` | 2 | `codes={-800}` |
| `bench_results/pd_kimi_nsys_prefill6_light/prefill_6_mtc_18/prefill-1/prefill-1-gb200-rack1-04.log:3115` | 2 | `codes={-800}` |
| `bench_results/pd_kimi_nsys_prefill6_light/prefill_6_mtc_18/prefill-2/prefill-2-gb200-rack1-07.log:2589` | 1 | `codes={-800}` |
| `bench_results/pd_kimi_nsys_prefill6_light/prefill_6_mtc_18/prefill-3/prefill-3-gb200-rack1-08.log:3706` | 2 | `codes={-800}` |

### 3.2 Important Observability Caveat

The old Python warning logged `first_key`, not the first failed key. This can be misleading.

Concrete example:
- Actual failed key log:
  - `bench_results/pd_kimi_nsys_prefill6_light/prefill_6_mtc_18/prefill-0/prefill-0-gb200-rack1-03.log:2164`
- Old Python warning:
  - `bench_results/pd_kimi_nsys_prefill6_light/prefill_6_mtc_18/prefill-0/prefill-0-gb200-rack1-03.log:2167`

These two keys do not match. Therefore the old warning line should not be interpreted as “the shown first key is the failed key.”

### 3.3 Analysis

`batch_put failed` is not a separate root cause in this run.

It is the vLLM-side aggregation of lower-level Mooncake return codes:
- Mooncake returns `TRANSFER_FAIL(-800)` for one or more keys in the batch.
- vLLM collects the per-key return codes and emits one batch-level warning.

In this run:
- 6 warnings correspond to 6 batch-level warnings.
- Those 6 warnings together cover 9 failed keys.

## 4. Source Mapping

### 4.1 Mooncake Source

| Signal | Source location | Meaning |
| --- | --- | --- |
| `packet mismatch` | `third_partys/Mooncake/mooncake-transfer-engine/src/transport/rdma_transport/rdma_endpoint.cpp:282` | Handshake descriptor or endpoint path mismatch |
| `reuse existing connection` | `third_partys/Mooncake/mooncake-transfer-engine/src/transport/rdma_transport/rdma_endpoint.cpp:254` | Simultaneous-open / already-connected branch |
| `mark it inactive` | `third_partys/Mooncake/mooncake-transfer-engine/src/transport/rdma_transport/worker_pool.cpp:245` | Endpoint setup failed and worker marks endpoint inactive |
| `timeout after 60 seconds` | `third_partys/Mooncake/mooncake-store/src/transfer_task.cpp:341` | Hard transfer timeout |
| `Transfer failed for key` | `third_partys/Mooncake/mooncake-store/src/client_service.cpp:1681` | Per-key transfer failure reporting |

### 4.2 vLLM Source

| Signal | Source location | Meaning |
| --- | --- | --- |
| `batch_put failed` warning | `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_worker.py:739` | Python-side aggregation of per-key Mooncake return codes |
| failure stats aggregation | `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_metrics.py:55` | Classifies `TRANSFER_FAIL`, `NO_AVAILABLE_HANDLE`, and other failures |
| failure counters exported to Prometheus | `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_metrics.py:235` | Prometheus counters for batch/key failure classes |

## 5. Observability Work Added Locally

### 5.1 Added in vLLM

- `batch_put failed` now records:
  - `req_id`
  - `tp_rank`
  - `first_failed_key`
  - `failed_examples`
- File:
  - `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_worker.py`

- Prometheus counters now exist for:
  - `vllm:mooncake_store_put_failed_batches`
  - `vllm:mooncake_store_put_failed_keys`
  - `vllm:mooncake_store_put_transfer_fail_keys`
  - `vllm:mooncake_store_put_no_available_handle_keys`
  - `vllm:mooncake_store_put_other_failed_keys`
- File:
  - `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_metrics.py`

### 5.2 Added in Local Mooncake Source

- Handshake failure log now includes:
  - `endpoint`
  - `local_nic_path`
  - `peer_nic_path`
  - `local_port`
  - `local_gid_index`
  - `local_active_mtu`
  - `configured_mtu`
  - `effective_mtu`
  - `peer_gid`
  - `peer_lid`
  - `peer_qp_num`

- Transfer failure log now includes:
  - `pending_transfer_index`
  - `replica_index`
  - `strategy`
  - full replica descriptor

### 5.3 Validation Status

- Passed:
  - `.venv/bin/python -m pytest tests/v1/kv_connector/unit/test_mooncake_store_worker.py -q`
  - `.venv/bin/python -m pytest tests/v1/kv_connector/unit/test_mooncake_store_metrics.py -q`
- Not yet revalidated in this turn:
  - Local Mooncake C++ changes have not been rebuilt and rerun yet.

## 6. Grafana / Prometheus Status

Current local status:
- Grafana container is running.
- Prometheus container is not running.
- Grafana datasource points to `http://localhost:9090`, which is currently unreachable.
- Grafana dashboard search returns empty, and `mooncake-overview` is not currently loaded.

Most likely explanation:
- Grafana started on `2026-04-12 03:10 UTC`.
- Dashboard and provisioning files were last modified on `2026-04-13 13:41 UTC`.
- The Grafana provider config likely needs a restart to load the new file-provisioned dashboards.

Also note:
- Existing `mooncake-overview.json` focuses on Mooncake master metrics (`master_*`) and does not directly cover the new per-worker `TRANSFER_FAIL` / `batch_put failed` counters.

## 7. Final Judgment

If these errors are ranked by “how likely they are to be the actual root cause”:

1. `handshake timeout` and the associated `packet mismatch / inactive endpoint` signals are the most likely root cause.
2. `transfer fail` is the most likely downstream consequence of that root cause.
3. `batch_put failed` is the vLLM-side symptom of the `transfer fail` return codes.

The next most valuable rerun data would be:
- the enriched handshake failure log with `MTU / GID / LID / QP / NIC path`
- the enriched transfer failure log with `replica_index / strategy / transport_endpoint`

That should make it possible to decide whether the problem is:
- endpoint state race / simultaneous-open bug
- multi-NIC path mismatch
- or a true RDMA parameter mismatch on a specific host / NIC
