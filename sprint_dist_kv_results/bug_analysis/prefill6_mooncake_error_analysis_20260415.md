# Issue 2: RDMA Handshake and Endpoint State Failure

Date: 2026-04-16

## Issue Card

| Field | Value |
| --- | --- |
| Issue Number | Issue 2 |
| Short Name | RDMA handshake / endpoint state |
| Current Status | `Strong hypothesis` |
| Affected Runs | Primarily `prefill_6_mtc_18`; weaker signals of the same family also appear in `prefill3/4/5` and `p4_d4` |
| One-line Conclusion | The strongest current judgment is that Mooncake RDMA connection establishment or endpoint state management is failing during the startup window, first as `handshake timeout / packet mismatch / inactive endpoint`, and later as `60s transfer timeout -> TRANSFER_FAIL -> batch_put failed`. |
| Related Issues | [Issue 1: Metadata / Stale Descriptor Visibility Failure](decode5_mtc16_mooncake_correlation_analysis_20260415.md); [Issue 3: Processing Object Re-put and OBJECT_NOT_FOUND](prefill3_decode5_mooncake_deep_dive_20260415_cn.md); [Issue 4: Observability, Error Accounting, and Diagnostic Blind Spots](batch_put_failed_global_analysis_20260415.md) |

## 1. Problem Definition

This issue refers to the following failure family:
- Mooncake hits low-level connection-establishment errors such as `handshake timeout`, `packet mismatch`, and `mark it inactive`
- some endpoints then enter a bad, inactive, or stale connection state
- later writes still hit those endpoints and block in `WaitForTransfers()` until the hard 60-second timeout
- Mooncake returns `TRANSFER_FAIL(-800)` and the Python side aggregates that into `batch_put failed`

This issue does not include:
- `metadata not found / Failed to open segment`, which belongs to [Issue 1](decode5_mtc16_mooncake_correlation_analysis_20260415.md)
- the downstream `OBJECT_NOT_FOUND(-704)` amplification chain, which belongs to [Issue 3](prefill3_decode5_mooncake_deep_dive_20260415_cn.md)
- logging and metrics blind spots, which belong to [Issue 4](batch_put_failed_global_analysis_20260415.md)

## 2. Current Status

- Evidence strength: `Strong hypothesis`
- Priority: high
- Single source of truth for the conclusion:
  - `prefill_6_mtc_18` is best explained as a startup-time RDMA handshake / endpoint state problem, not as an independent Python `batch_put` bug.

Facts that are already stable:
- `handshake timeout`, `packet mismatch`, and `inactive endpoint` are concentrated in the startup window
- the later `60s transfer timeout` events occur after those startup failures
- all Python-side `batch_put failed` warnings in this run carry `-800`
- the observed pattern does not fit offload pressure or the Issue 1 metadata-404 family

## 3. Affected Scope

Primary run:
- `bench_results/pd_kimi_nsys_prefill6_light/prefill_6_mtc_18`

Current strongest host hotspot:
- `192.168.0.104` accounts for `11/27` inactive-endpoint events

Secondary related runs:
- `prefill_3_mtc_12`
- `prefill_4_mtc_14`
- `prefill_5_mtc_16`
- `p4_d4_mtc_14`

Those runs show weaker signals such as re-establish / same-peer-QP reuse, but not the same concentrated and fully visible chain as `prefill6`.

## 4. Main Symptoms and Log Fingerprint

The most stable low-level fingerprint is:

```text
handshake timeout
  -> received packet mismatch
  -> mark it inactive
  -> Re-establish connection / reuse existing connection
  -> Failed to complete transfers after 60 seconds
  -> Transfer failed for key: TRANSFER_FAIL
  -> batch_put failed
```

Representative keywords:
- `handshake timeout`
- `received packet mismatch`
- `mark it inactive`
- `Re-establish connection`
- `Received same peer QP numbers, reusing connection`
- `Failed to complete transfers after 60 seconds`
- `Transfer failed for key`

## 5. Strongest Current Mechanism Chain

The strongest current explanation is:

1. Some RDMA endpoints enter an abnormal handshake path during startup.
2. The abnormal path surfaces as `handshake timeout`, `packet mismatch`, connection reuse / re-establish races, and inactive endpoints.
3. Later store writes still hit those bad or stale endpoint states.
4. Transfer completion never arrives, so the wait logic times out at 60 seconds.
5. Mooncake marks the keys as `TRANSFER_FAIL(-800)`.
6. vLLM aggregates the per-key results and logs `batch_put failed`.

What is fact vs inference:
- Steps 1, 2, 4, 5, and 6 are directly supported by logs and source.
- Step 3, specifically the stale / broken endpoint-state interpretation, remains the strongest current inference.

## 6. Key Evidence

### 6.1 Handshake failures by log file

| Log file | `handshake timeout` count | Representative location |
| --- | ---: | --- |
| `prefill-1-gb200-rack1-04.log` | 17 | `:1340` |
| `prefill-0-gb200-rack1-03.log` | 5 | `:1286` |
| `prefill-3-gb200-rack1-08.log` | 3 | `:1912` |
| `prefill-4-gb200-rack1-10.log` | 1 | `:1741` |
| `prefill-5-gb200-rack1-11.log` | 1 | `:1583` |

This already shows that the problem is not evenly distributed across all prefill workers.

### 6.2 Error count summary

| Category | Count | Timing / meaning |
| --- | ---: | --- |
| `handshake timeout` | 27 | all within `2026-04-14 09:15:48-09:15:49` |
| `packet mismatch` | 14 | handshake descriptor or path mismatch signal |
| `inactive endpoint` | 27 | endpoint setup failed and the worker marked it inactive |
| `transfer timeout` | 9 | `Failed to complete transfers after 60 seconds` |
| `Transfer failed for key` | 9 | `TRANSFER_FAIL` on 9 keys |
| `batch_put failed` | 6 | Python batch warnings covering those 9 keys |

This ordering matters:
- handshake failures first
- transfer timeouts later
- Python warnings last

### 6.3 Host hotspot distribution

| Host | Count |
| --- | ---: |
| `192.168.0.104` | 11 |
| `192.168.0.107` | 4 |
| `192.168.0.108` | 4 |
| `192.168.0.103` | 4 |
| `192.168.0.110` | 3 |
| `192.168.0.111` | 1 |

This is one of the strongest signs that the failure is not just uniform startup noise.

### 6.4 All 60-second timeout batches and failed keys

| Worker log | Timeout line | Batch ID | Failed key line | Failed key suffix |
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

This table preserves the exact timeout-batch to failed-key mapping and is important for later endpoint-pair timelines.

### 6.5 `TRANSFER_FAIL` here is a hard 60-second timeout

Relevant source:
- `third_partys/Mooncake/mooncake-store/src/transfer_task.cpp:341`
- `third_partys/Mooncake/mooncake-store/src/client_service.cpp:1681`

Meaning:
- this is not a random Python exception
- it is a deterministic Mooncake transfer wait timeout

### 6.6 `batch_put failed` is a symptom here, not a root cause

Python-side warnings all carry:
- `codes={-800}`

Meaning:
- the Python warning adds aggregation, not root-cause information
- the root-cause signals are already present in the lower-level handshake and transfer logs

## 7. Strongest State-Machine Explanations

The most useful next-level breakdown is not "the network is bad," but the following three state-machine candidates.

### 7.1 Candidate 1: simultaneous-open or stale connection reuse

Supporting evidence:
- logs contain `Received same peer QP numbers, reusing connection`
- logs contain `Re-establish connection`
- the issue is highly concentrated in startup, not spread uniformly over time

Current judgment:
- strongest candidate

### 7.2 Candidate 2: multi-NIC path mismatch or endpoint advertisement mismatch

Supporting evidence:
- `packet mismatch` is a direct handshake-stage signal
- some logs show empty or inconsistent `peer.local_nic_path` / `peer.peer_nic_path`

Current judgment:
- second strongest candidate

### 7.3 Candidate 3: real RDMA parameter mismatch

Potential dimensions:
- `MTU`
- `GID`
- `LID`
- `QP`

Current judgment:
- cannot be ruled out
- but currently weaker than the first two state-machine interpretations

## 8. What This Is Not

The current evidence does not support these explanations:

1. This is not the [Issue 1](decode5_mtc16_mooncake_correlation_analysis_20260415.md) metadata-404 family.
2. This is not an independent Python `batch_put failed` bug.
3. This is not offload pressure or `NO_AVAILABLE_HANDLE(-200)`.
4. This is not decode-side load variation or benchmark-side noise.

## 9. Relationship to Other Issues

- Relation to [Issue 1](decode5_mtc16_mooncake_correlation_analysis_20260415.md):
  - both can end in `TRANSFER_FAIL(-800)` and `batch_put failed`
  - but Issue 1 is driven by metadata / segment descriptor lookup failure
  - Issue 2 is driven by handshake / endpoint state failure

- Relation to [Issue 3](prefill3_decode5_mooncake_deep_dive_20260415_cn.md):
  - Issue 3 explains how some `TRANSFER_FAIL` chains can later amplify into `OBJECT_NOT_FOUND(-704)`
  - that is not the main contradiction in the `prefill6` run

- Relation to [Issue 4](batch_put_failed_global_analysis_20260415.md):
  - the Issue 4 observability work is what will separate simultaneous-open, path mismatch, and real RDMA parameter mismatch cleanly

## 10. What Is Still Unproven

The main remaining unknowns are:

1. Is this primarily simultaneous-open / stale reuse, or primarily path mismatch?
2. Is `192.168.0.104` a host-local hotspot, a NIC-local hotspot, or simply where the current state-machine bug becomes most visible?
3. Once an endpoint is marked inactive, why do later writes still hit a path that stalls for 60 seconds?

So the current state is:
- we already know the root cause is not the Python warning itself
- but we have not yet narrowed the state-machine failure to a single implementation bug

## 11. Fix Direction

First, strengthen state-machine evidence:
- print `endpoint / NIC path / MTU / GID / LID / QP` in handshake-failure logs
- print `replica_index / strategy / transport_endpoint` in transfer-failure logs
- build endpoint-pair timelines for the `192.168.0.104` hotspot

Then narrow logic fixes such as:
- stale connection reuse conditions
- simultaneous-open conflict handling
- endpoint reset / quarantine behavior after path mismatch

Also keep the following source map close to the issue:

| Signal | Source location | Role |
| --- | --- | --- |
| `packet mismatch` | `rdma_endpoint.cpp:282` | handshake descriptor or endpoint-path mismatch |
| `reuse existing connection` | `rdma_endpoint.cpp:254` | simultaneous-open / reuse branch |
| `mark it inactive` | `worker_pool.cpp:245` | endpoint setup failed, worker marks inactive |
| `timeout after 60 seconds` | `transfer_task.cpp:341` | hard transfer timeout |
| `Transfer failed for key` | `client_service.cpp:1681` | per-key transfer failure reporting |

## 12. Verification Plan

To consider this issue fixed, all of the following should hold:

1. The startup window no longer shows bursts of `handshake timeout`.
2. `packet mismatch` and `mark it inactive` largely disappear.
3. The `Failed to complete transfers after 60 seconds` events disappear.
4. `TRANSFER_FAIL(-800)` and Python `batch_put failed` warnings drop sharply or vanish together.
5. Under the same benchmark shape, `prefill_6_mtc_18` no longer enters the same post-startup write-path failure family.

## 13. Stable Conclusion Wording

Use the following wording as the single source of truth for this issue:

> `prefill_6_mtc_18` is best explained by a startup-time RDMA handshake / endpoint state failure. The low-level chain begins with `handshake timeout / packet mismatch / inactive endpoint` and later amplifies into `60s transfer timeout -> TRANSFER_FAIL(-800) -> batch_put failed`. In this issue, `batch_put failed` is only a downstream aggregation symptom, not a root cause.
