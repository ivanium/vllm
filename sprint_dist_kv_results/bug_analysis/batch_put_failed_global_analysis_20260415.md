# Issue 4: Observability, Error Accounting, and Diagnostic Blind Spots

Date: 2026-04-16

## Issue Card

| Field | Value |
| --- | --- |
| Issue Number | Issue 4 |
| Short Name | observability / error accounting |
| Current Status | `Confirmed` |
| Affected Runs | Cross-cutting issue; confirmed to affect diagnosis and monitoring for `decode_5_mtc_16`, `prefill_6_mtc_18`, `prefill_3/4/5`, and `p4_d4_mtc_14` |
| One-line Conclusion | At least two observability problems are already confirmed: the Python-side transfer-fail accounting was wrong, and the current logs / metrics still miss key context needed to separate `batch_put failed`, `TRANSFER_FAIL`, and `OBJECT_NOT_FOUND` into the correct issue families. |
| Related Issues | [Issue 1: Metadata / Stale Descriptor Visibility Failure](decode5_mtc16_mooncake_correlation_analysis_20260415.md); [Issue 2: RDMA Handshake and Endpoint State](prefill6_mooncake_error_analysis_20260415.md); [Issue 3: Processing Object Re-put and OBJECT_NOT_FOUND](prefill3_decode5_mooncake_deep_dive_20260415_cn.md) |

## 1. Problem Definition

This issue only covers why we still have trouble observing and distinguishing Issues 1, 2, and 3. It does not define a new data-plane root cause.

It consists of three categories:
- incorrect error accounting
- missing context in warnings and metrics
- decode-side blind spots that force root-cause analysis to rely on prefill-side logs

The core question is not:
- why a benchmark slowed down
- why a specific endpoint failed

The core question is:
- why the existing logs and metrics still make different failure families look deceptively similar

## 2. Current Status

- Evidence strength: `Confirmed`
- Priority: high, but as a supporting issue
- Single source of truth for the conclusion:
  - observability gaps do not directly create the data-plane failures, but they do distort monitoring, delay root-cause isolation, and make one symptom family look like several independent bugs.

Confirmed facts:
- Python-side transfer-fail accounting previously treated `-1` as transfer failure instead of `-800`
- old `batch_put failed` warnings were missing key request, rank, elapsed, and failed-key context
- in `decode5`, the decode-side hit-rate view cannot distinguish a real successful load from silent fallback / recompute

## 3. Affected Scope

Across the benchmark set, at least 6 runs show `batch_put failed` or closely related lower-level failures:

| Benchmark run | Affected logs | `batch_put failed` | `TRANSFER_FAIL`-related lines | Most likely mapped issue |
| --- | --- | ---: | ---: | --- |
| `decode_5_mtc_16` | `prefill-0-gb200-rack1-09.log` | 2205 | 106994 | [Issue 1](decode5_mtc16_mooncake_correlation_analysis_20260415.md) |
| `prefill_6_mtc_18` | 4 prefill logs | 6 | 18 | [Issue 2](prefill6_mooncake_error_analysis_20260415.md) |
| `prefill_3_mtc_12` | `prefill-2-gb200-rack1-13.log` | 2 | 16 | [Issue 3](prefill3_decode5_mooncake_deep_dive_20260415_cn.md) |
| `prefill_5_mtc_16` | `prefill-1-gb200-rack1-07.log` | 2 | 12 | weaker signal of Issue 2 or 3 |
| `prefill_4_mtc_14` | `prefill-0-gb200-rack1-01.log` | 1 | 4 | weaker signal of Issue 2 |
| `p4_d4_mtc_14` | `prefill-0-gb200-rack1-06.log` | 1 | 2 | weaker signal of Issue 2 |

Healthy controls:
- `decode_4_mtc_14`
- `decode_6_mtc_18`
- `prefill_2_mtc_*`

## 4. Main Symptoms and Diagnostic Fingerprints

The most characteristic symptom of this issue is ambiguity, not a single error chain.

The most misleading log shapes are:

1. `batch_put failed`
   - aggregation symptom, not a root cause

2. `codes={-800}`
   - without a code-name mapping, it is easy to miss that this is `TRANSFER_FAIL`

3. `first_key=...`
   - historically this was not necessarily the first failed key, only the first key in the batch

4. `External prefix cache hit rate: 100.0%`
   - this does not prove that the external KV load truly succeeded

## 5. Strongest Current Diagnostic Mechanism Chain

The mechanism here is diagnostic, not data-plane:

1. A real data-plane failure happens in Issue 1, 2, or 3.
2. Python and C++ logs do not carry enough request, key, endpoint, and state context.
3. Metrics may classify the failure under the wrong error bucket.
4. Decode-side observability cannot show whether a "hit" led to an actual successful external KV load.
5. As a result, `batch_put failed`, `TRANSFER_FAIL`, and `OBJECT_NOT_FOUND` can be misread as independent root causes instead of different layers of one issue chain.

## 6. Key Evidence

### 6.1 Confirmed transfer-fail accounting bug

Historical Python code included:

```python
transfer_fail_keys = sum(1 for i in failed if res[i] == -1)
```

But Mooncake C++ defines:
- `-1 = INTERNAL_ERROR`
- `-800 = TRANSFER_FAIL`

Impact:
- `vllm:mooncake_store_put_transfer_fail_keys` counted the wrong category
- real `TRANSFER_FAIL(-800)` keys were pushed into `other_failed_keys`

This is a confirmed accounting bug, not a hypothesis.

### 6.2 `batch_put failed` lacked the most important context

Old warnings reliably showed:
- `len(failed)` / `len(keys)`
- `failed_codes`
- `total_bytes`
- `keys[0]`

But they lacked:
- `req_id`
- `tp_rank`
- `elapsed`
- failed-key examples
- readable error-code names

Most importantly:
- `keys[0]` was not guaranteed to be a failed key

### 6.3 Decode-side hit-rate is not enough

In `decode_5_mtc_16`, decode logs reliably show:
- `External prefix cache hit rate: 100.0%`

But do not show:
- `batch_get failed`
- `load error`
- `fallback`
- `recompute`

That means the decode side still cannot answer the key question:
- did a hit actually turn into a successful external KV load?

### 6.4 C++ and Python still expose different layers of the same chain without a shared hierarchy

The same underlying failure can appear as:
- `metadata not found`
- `Failed to open segment`
- `Transfer submission failed for key`
- `TRANSFER_FAIL`
- `batch_put failed`

Without a stable symptom hierarchy, those can be mistaken for different bugs.

### 6.5 The full Mooncake error-code table should remain close to the docs

Source:
- `third_partys/Mooncake/mooncake-store/include/types.h:208-274`

| Code | Name | Meaning |
| ---: | --- | --- |
| 0 | `OK` | success |
| -1 | `INTERNAL_ERROR` | internal error |
| -10 | `BUFFER_OVERFLOW` | insufficient buffer |
| -100 | `SHARD_INDEX_OUT_OF_RANGE` | shard index out of range |
| -101 | `SEGMENT_NOT_FOUND` | no usable segment found |
| -102 | `SEGMENT_ALREADY_EXISTS` | segment already exists |
| -103 | `CLIENT_NOT_FOUND` | client not found |
| -200 | `NO_AVAILABLE_HANDLE` | handle allocation failed / offload pressure |
| -300 | `INVALID_VERSION` | invalid version |
| -400 | `INVALID_KEY` | invalid key |
| -500 | `WRITE_FAIL` | write failure |
| -600 | `INVALID_PARAMS` | invalid params |
| -601 | `ILLEGAL_CLIENT` | illegal client |
| -700 | `INVALID_WRITE` | invalid write |
| -701 | `INVALID_READ` | invalid read |
| -702 | `INVALID_REPLICA` | invalid replica operation |
| -703 | `REPLICA_IS_NOT_READY` | replica not ready |
| -704 | `OBJECT_NOT_FOUND` | object not found |
| -705 | `OBJECT_ALREADY_EXISTS` | object already exists |
| -706 | `OBJECT_HAS_LEASE` | object has lease |
| -707 | `LEASE_EXPIRED` | lease expired before data transfer |
| -708 | `OBJECT_HAS_REPLICATION_TASK` | object has ongoing replication task |
| -709 | `OBJECT_NO_REPLICATION_TASK` | object has no ongoing replication task |
| -710 | `REPLICA_NOT_FOUND` | replica not found |
| -711 | `REPLICA_ALREADY_EXISTS` | replica already exists |
| -712 | `REPLICA_IS_GONE` | replica existed but is now gone |
| -713 | `REPLICA_NOT_IN_LOCAL_MEMORY` | replica not in local memory |
| -714 | `OBJECT_REPLICA_BUSY` | replica refcount non-zero |
| -800 | `TRANSFER_FAIL` | transfer operation failed |
| -900 | `RPC_FAIL` | RPC failed |
| -1000 | `ETCD_OPERATION_ERROR` | etcd operation failed |
| -1001 | `ETCD_KEY_NOT_EXIST` | etcd key missing |
| -1002 | `ETCD_TRANSACTION_FAIL` | etcd transaction failed |
| -1003 | `ETCD_CTX_CANCELLED` | etcd context cancelled |
| -1004 | `OPLOG_ENTRY_NOT_FOUND` | oplog entry not found |
| -1010 | `UNAVAILABLE_IN_CURRENT_STATUS` | unavailable in current status |

### 6.6 Missing diagnostic fields should remain an explicit checklist

Old `batch_put failed` missing fields:

| Missing field | Why it matters |
| --- | --- |
| `req_id` | ties the failure to a request lifecycle |
| `tp_rank` | helps separate per-rank issues |
| `elapsed` | separates fast failure from 60s timeout from finalize-stage failure |
| failed-key samples | identifies exact block / rank / hash failures |
| readable code names | avoids raw `codes={-800}` ambiguity |

Old `batch_get failed` gaps:
- `req_id`
- `tp_rank`
- `elapsed`
- failed key / readable code name

Old exception-handler gaps:
- missing `req_id`
- missing `tp_rank`
- missing traceback

### 6.7 C++ source-side error origins are still worth keeping in one place

| Signal | Source location | Current meaning |
| --- | --- | --- |
| `metadata not found` | `http_metadata_server.cpp:39,87` | key missing from metadata store |
| `Failed to open segment` | `transfer_task.cpp:498,535,639` | `openSegment(handle.transport_endpoint_)` failed |
| `Transfer submission failed for key` | `client_service.cpp:1621,1992` | transfer submit failed and became `TRANSFER_FAIL` |
| `TRANSFER_FAIL` assignment points | `client_service.cpp:960,982,1102,1623,1994,2323,2396,2608,2628` | normalized transfer failure in multiple read/write paths |

## 7. Symptom-to-Issue Mapping

This table replaces the older "mode A / mode B" mix and should be the default routing reference.

| Symptom / keyword | Should not be treated as root cause directly | Most likely points to |
| --- | --- | --- |
| `batch_put failed` | yes | aggregation symptom of Issue 1 / 2 / 3 |
| `TRANSFER_FAIL(-800)` | yes | Issue 1 or 2, and sometimes the upstream trigger for Issue 3 |
| `OBJECT_NOT_FOUND(-704)` | yes | prioritize Issue 3 |
| `metadata not found` | no | prioritize Issue 1 |
| `Failed to open segment` | no | prioritize Issue 1 |
| `handshake timeout` | no | prioritize Issue 2 |
| `packet mismatch` | no | prioritize Issue 2 |
| `inactive endpoint` | no | prioritize Issue 2 |
| decode-side `100% hit rate` | yes | only means "counted as a hit"; must be interpreted through Issue 4 |

## 8. What This Is Not

This issue is not:

1. a new data-plane root cause
2. `batch_put failed` itself
3. a single-run-only problem

It is the cross-cutting explanation for why diagnosis remains slow and error-prone.

## 9. Relationship to Other Issues

- Relation to [Issue 1](decode5_mtc16_mooncake_correlation_analysis_20260415.md):
  - metadata `PUT/DELETE` and descriptor lifecycle logs are exactly what would most quickly upgrade Issue 1 from `Strong hypothesis` to `Confirmed`

- Relation to [Issue 2](prefill6_mooncake_error_analysis_20260415.md):
  - handshake-field enrichment is what would separate simultaneous-open, path mismatch, and real RDMA parameter mismatch

- Relation to [Issue 3](prefill3_decode5_mooncake_deep_dive_20260415_cn.md):
  - `ExistKey`, `PutStart discard`, and `WaitForTransfers` structure are what would close the final proof gap for the `OBJECT_NOT_FOUND` amplification chain

## 10. What Is Still Unproven

For Issue 4, the remaining unknown is not whether the blind spots exist, but exactly which fixes are already live versus still local-only.

Open items:
- which observability patches are already present in the exact runtime build
- whether decode-side "true load success / fallback" should be emitted at the connector, scheduler, or metrics layer
- whether metadata lifecycle logging should live in the metadata server, transfer metadata layer, or both

## 11. Fix Direction

### 11.1 Error accounting

- define `MOONCAKE_TRANSFER_FAIL = -800`
- fix `transfer_fail_keys` accounting
- export readable code names in addition to raw integers

### 11.2 Python warning context

At minimum, `batch_put failed` / `batch_get failed` should include:
- `req_id`
- `tp_rank`
- `elapsed`
- `first_failed_key`
- `failed_examples`
- readable code names

### 11.3 C++ state-point logging

Highest priority additions:
- `ExistKey()`: distinguish "metadata absent" from "processing but not completed"
- `PutStart()`: structured logging for the 30-second discard branch
- `WaitForTransfers()`: pending future count and current future index
- metadata server: successful `PUT / DELETE`
- transfer metadata: successful descriptor registration / removal

### 11.4 Decode-side true-result observability

At minimum:
- blocks counted as external-prefix hits
- load success count
- load failure count
- fallback / recompute count

Without this layer, `decode5`-style cases still have to be root-caused mostly from prefill logs.

### 11.5 Observability work already added locally should not be lost

vLLM-side additions already made locally:
- `batch_put failed` logs `req_id`
- `batch_put failed` logs `tp_rank`
- `batch_put failed` logs `first_failed_key`
- `batch_put failed` logs `failed_examples`
- counters for `put_failed_batches / put_failed_keys / put_transfer_fail_keys / put_no_available_handle_keys / put_other_failed_keys`

Mooncake-side additions already made locally:
- handshake failure logs enriched with `endpoint / NIC path / MTU / GID / LID / QP`
- transfer failure logs enriched with `pending_transfer_index / replica_index / strategy / full replica descriptor`

### 11.6 Validation and dashboard state should remain attached to the issue

Already validated:
- `.venv/bin/python -m pytest tests/v1/kv_connector/unit/test_mooncake_store_worker.py -q`
- `.venv/bin/python -m pytest tests/v1/kv_connector/unit/test_mooncake_store_metrics.py -q`

Not yet revalidated:
- local Mooncake C++ observability changes have not yet been rebuilt and rerun in this turn

Grafana / Prometheus state at the time of analysis:
- Grafana container running
- Prometheus container not running
- Grafana datasource points at `http://localhost:9090`, currently unreachable
- dashboard search empty; `mooncake-overview` not loaded

Meaning:
- even if code-level patches exist, dashboard-level observability is not yet in a stable usable state

## 12. Verification Plan

To consider this issue sufficiently addressed, all of the following should hold:

1. `TRANSFER_FAIL` Prometheus accounting matches actual `-800` log occurrences.
2. `batch_put failed` warnings directly show `req_id / tp_rank / elapsed / failed key samples`.
3. `batch_get failed` warnings include the same context.
4. `ExistKey`, `PutStart discard`, and `WaitForTransfers` logs are sufficient to prove or disprove Issue 3 directly.
5. metadata `PUT/DELETE` and descriptor registration / removal logs are sufficient to prove or disprove Issue 1 directly.
6. decode-side metrics can distinguish "counted as hit" from "real load success."

### 12.1 Reproduction entry points should remain here

Primary reproduction:
- `decode_5_mtc_16`
  - config: `bench_results/kimi_pd_p1d1to17_mooncake_nsys_autosweep/decode_5_mtc_16/attempt_1/rendered_config.yaml`
  - command: `vigil -c bench_results/kimi_pd_p1d1to17_mooncake_nsys_autosweep/decode_5_mtc_16/attempt_1/rendered_config.yaml`

Secondary reproduction:
- `prefill_6_mtc_18`
  - config: `bench_results/pd_kimi_nsys_prefill6_light/prefill_6_mtc_18/rendered_config.yaml`
  - command: `vigil -c bench_results/pd_kimi_nsys_prefill6_light/prefill_6_mtc_18/rendered_config.yaml`

## 13. Stable Conclusion Wording

Use the following wording as the single source of truth for this issue:

> At least two observability problems are already confirmed. First, transfer-fail accounting previously classified `TRANSFER_FAIL(-800)` under the wrong error bucket. Second, Python, C++, and decode-side logs still lack the fields needed to connect object, segment, endpoint, request, and real external-load outcome. These do not directly create the data-plane failures, but they do distort monitoring and delay the root-cause isolation of Issues 1, 2, and 3.
