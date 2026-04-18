# Issue 3: Processing Object Re-put and OBJECT_NOT_FOUND Amplification

Date: 2026-04-16

## Issue Card

| Field | Value |
| --- | --- |
| Issue Number | Issue 3 |
| Short Name | processing object re-put / `OBJECT_NOT_FOUND` |
| Current Status | `Strong hypothesis` |
| Affected Runs | Primarily `prefill_3_mtc_12`; `OBJECT_NOT_FOUND(-704)` has not yet been established as the dominant contradiction in other runs |
| One-line Conclusion | The strongest current judgment is that `OBJECT_NOT_FOUND(-704)` in `prefill3` is not an independent master-side root cause, but a secondary failure created by `TRANSFER_FAIL` plus segment-level wait amplification, `ExistKey` semantic mismatch, and 30-second discard of stale processing metadata. |
| Related Issues | [Issue 1: Metadata / Stale Descriptor Visibility Failure](decode5_mtc16_mooncake_correlation_analysis_20260415.md); [Issue 2: RDMA Handshake and Endpoint State](prefill6_mooncake_error_analysis_20260415.md); [Issue 4: Observability, Error Accounting, and Diagnostic Blind Spots](batch_put_failed_global_analysis_20260415.md) |

## 1. Problem Definition

This issue refers to the following chain:
- some put operations first enter `TRANSFER_FAIL(-800)` or remain pending for a long time in the transfer phase
- upper-layer dedup / retry logic treats objects that are still in processing but not yet completed as "missing"
- a new `PutStart` that arrives after `put_start_discard_timeout_sec=30` can erase the old processing metadata
- the delayed `BatchPutEnd` from the original put then fails with `OBJECT_NOT_FOUND(-704)`

The key question here is not "why did transfer fail first?"

The key question is:
- why does finalize later surface what looks like an independent master-side `OBJECT_NOT_FOUND` failure?

This issue does not include:
- `metadata not found / Failed to open segment`, which belongs to [Issue 1](decode5_mtc16_mooncake_correlation_analysis_20260415.md)
- `handshake timeout / packet mismatch / inactive endpoint`, which belongs to [Issue 2](prefill6_mooncake_error_analysis_20260415.md)
- logging gaps and metric gaps, which belong to [Issue 4](batch_put_failed_global_analysis_20260415.md)

## 2. Current Status

- Evidence strength: `Strong hypothesis`
- Priority: medium-high
- Single source of truth for the conclusion:
  - `OBJECT_NOT_FOUND(-704)` in `prefill3` is best understood as a secondary state-machine effect after `TRANSFER_FAIL`, not as a fully independent new master bug.

Facts that are already stable:
- repeated 60-second timeout events appear from `08:07` to `08:11`
- `OBJECT_NOT_FOUND` appears only after multiple earlier `TRANSFER_FAIL` events have already accumulated
- later revoke lines on the same thread cover keys that timed out earlier on that same thread
- Mooncake source contains all three mechanisms needed to amplify this chain:
  - segment-level merged writes
  - serial waiting in `WaitForTransfers()`
  - 30-second discard of stale processing metadata in `PutStart()`

## 3. Affected Scope

Primary run:
- `bench_results/pd_kimi_nsys_prefill3_light/prefill_3_mtc_12/prefill-2/prefill-2-gb200-rack1-13.log`

Primary scenario:
- multi-instance prefill writes of external KV
- `prefer_alloc_in_same_node` merged-write path
- batches or neighboring batches with multiple pending / timed-out transfers

Main risk:
- `TRANSFER_FAIL` is later followed by `OBJECT_NOT_FOUND`
- this can easily be misread as a second independent master-side bug family

## 4. Main Symptoms and Log Fingerprint

The most recognizable pattern here is not one line, but a time chain:

```text
Failed to complete transfers after 60 seconds
  -> Transfer failed for key: TRANSFER_FAIL
  -> Continue waiting for other futures in the same op
  -> Later batches hit more timeouts
  -> Finalize put for key: OBJECT_NOT_FOUND
  -> Revoke earlier failed keys
```

Representative keywords:
- `Failed to complete transfers after 60 seconds`
- `Transfer failed for key`
- `Failed to finalize put for key`
- `OBJECT_NOT_FOUND`
- `revoke`

Representative locations:
- first timeout at `08:07`: `.../prefill-2-gb200-rack1-13.log:2490`
- second timeout at `08:08`: `.../prefill-2-gb200-rack1-13.log:3240`
- first mixed `TRANSFER_FAIL + OBJECT_NOT_FOUND` batch at `08:09`: `.../prefill-2-gb200-rack1-13.log:3990`
- fourth timeout at `08:10`: `.../prefill-2-gb200-rack1-13.log:4808`
- second mixed failure batch at `08:11`: `.../prefill-2-gb200-rack1-13.log:5579`

## 5. Strongest Current Mechanism Chain

The strongest current explanation is:

1. `BatchPutWhenPreferSameNode()` groups multiple logical keys by `transport_endpoint_` and submits transfer at the segment level.
2. A segment-level transfer future times out and the operation records `TRANSFER_FAIL(-800)`.
3. `WaitForTransfers()` does not stop after the first failure; it serially waits for the remaining futures in the same op, each with a 60-second timeout.
4. Finalization of a single op can therefore be delayed by 180, 240, or even 300 seconds instead of only 60 seconds.
5. Meanwhile, the vLLM sender runs `batch_is_exist(keys)`, while Mooncake `ExistKey()` returns `false` for objects that still exist in metadata but have no completed replica yet.
6. If a new `PutStart` hits those keys after 30 seconds, master discards the old processing metadata.
7. The original delayed `BatchPutEnd` finally runs against metadata that has already been erased and fails with `OBJECT_NOT_FOUND(-704)`.

What is fact vs inference:
- Steps 1, 3, 5, and 6 are directly supported by source.
- Steps 2, 4, and 7 match the observed log timeline very closely.
- the remaining missing piece is the final master-side lifecycle proof for the exact `PutStart -> discard -> delayed BatchPutEnd` sequence

## 6. Key Evidence

### 6.1 The `08:07` to `08:11` chain is cumulative, not a set of isolated failures

| Time | Thread / pid | Event | Meaning |
| --- | --- | --- | --- |
| `08:07:25` | `2137498` | timeout + `TRANSFER_FAIL` | first timeout |
| `08:07:25` | `2137533` | timeout + `TRANSFER_FAIL` | second thread also starts timing out |
| `08:08:25` | `2137498` | timeout + `TRANSFER_FAIL` | second round |
| `08:08:25` | `2137533` | timeout + `TRANSFER_FAIL` | second round |
| `08:09:25` | `2137498` | timeout + `TRANSFER_FAIL` | third round |
| `08:09:25` | `2137533` | timeout + `TRANSFER_FAIL` | third round |
| `08:09:25` | `2137533` | same batch hits `OBJECT_NOT_FOUND` | finalize has now entered the secondary failure stage |
| `08:09:25` | `2137533` | revoke 3 earlier failed keys | covers the earlier timeout keys on that thread |
| `08:10:25` | `2137498` | timeout + `TRANSFER_FAIL` | fourth round |
| `08:11:25` | `2137498` | timeout + `TRANSFER_FAIL` | fifth round |
| `08:11:25` | `2137498` | same batch hits `OBJECT_NOT_FOUND` | second finalize anomaly |
| `08:11:25` | `2137498` | revoke 5 earlier failed keys | covers the keys from `08:07` through `08:11` |

This is one of the strongest indicators that the state survives past the first 60-second timeout and continues to affect later finalize / revoke behavior.

### 6.2 Mooncake is not doing per-key transfer here; it merges by segment

Relevant source:
- `third_partys/Mooncake/mooncake-store/src/client_service.cpp:1941`

Important logic:
- keys are grouped by `buffer_descriptor.transport_endpoint_`
- transfer is submitted per merged segment group
- `WaitForTransfers()` operates on the merged ops, then writes state back to the original logical ops

This explains why:
- only a few `Transfer failed for key` lines may appear
- but a larger group of logical keys later fails together during finalize / revoke

### 6.3 `WaitForTransfers()` serially waits on every future

Relevant source:
- `third_partys/Mooncake/mooncake-store/src/client_service.cpp:1633`
- `third_partys/Mooncake/mooncake-store/src/transfer_task.cpp:297`
- `third_partys/Mooncake/mooncake-store/src/transfer_task.cpp:341`

Meaning:
- `WaitForTransfers()` calls `pending_transfers[i].get()` one by one
- it keeps waiting even after one future already failed
- each future can wait up to 60 seconds

This is what turns:
- one failed future into 60 seconds
- three pending futures into roughly 180 seconds
- five pending futures into roughly 300 seconds

### 6.4 `batch_is_exist` and `ExistKey` disagree on what "exists" means

vLLM sender behavior:
- `mooncake_store_worker.py:397-455`
- any key with `exists != 1` is treated as missing and is eligible for re-put

Mooncake `ExistKey()` behavior:
- `master_service.cpp:432`
- only returns `true` if there is at least one completed replica
- returns `false` for objects that still exist in metadata but only have processing replicas

This semantic mismatch is central:
- to vLLM the object appears missing
- to Mooncake the object may still exist, just not yet be completed

### 6.5 The 30-second discard branch can erase old processing metadata

Relevant source:
- `third_partys/Mooncake/mooncake-store/src/master_service.cpp:850`

Relevant config:
- `scripts/mooncake/mooncake_master.log:2`
- `put_start_discard_timeout_sec=30`
- `put_start_release_timeout_sec=600`

Meaning:
- if there is no completed replica
- and the old `put_start_time` is older than 30 seconds
- a new `PutStart` can discard the old processing metadata and proceed

### 6.6 This is weaker as a pure "network is slow" explanation

Around `08:09:25`, transfer-engine stats are not consistent with "the entire data plane is dead":
- throughput is still `586.74 MB/s`
- the latency distribution is still dominated by `200-1000us`, with only a minority of long-tail outliers

That makes a narrow state-machine / lifecycle explanation much stronger than a blanket "the whole network stalled" explanation.

## 7. What This Is Not

The current evidence does not support these explanations:

1. This is not a separately proven brand-new master bug family.
2. This is not just "the network was slow for 60 seconds."
3. This is not the [Issue 1](decode5_mtc16_mooncake_correlation_analysis_20260415.md) metadata-404 visibility family.
4. This is not explained by Python warnings themselves.

## 8. Relationship to Other Issues

- Relation to [Issue 1](decode5_mtc16_mooncake_correlation_analysis_20260415.md):
  - Issue 1 is driven by metadata / stale descriptor lookup failure
  - Issue 3 can begin from multiple kinds of upstream `TRANSFER_FAIL`; its center is how later state-machine behavior turns that into `OBJECT_NOT_FOUND`

- Relation to [Issue 2](prefill6_mooncake_error_analysis_20260415.md):
  - Issue 2 style `TRANSFER_FAIL` could also become an upstream trigger for Issue 3
  - but `prefill6` does not currently expose `OBJECT_NOT_FOUND` as the dominant contradiction

- Relation to [Issue 4](batch_put_failed_global_analysis_20260415.md):
  - `ExistKey`, `PutStart discard`, and `WaitForTransfers` instrumentation are the exact observability additions that would most likely upgrade this issue from `Strong hypothesis` to `Confirmed`

## 9. What Is Still Unproven

What is missing is not the overall mechanism, but the final master-side proof.

The most valuable missing evidence would be:
- the exact first `PutStart` for the key
- the object's processing state before completion
- the exact later `PutStart` that enters the 30-second discard branch
- the metadata erase event itself
- the delayed original `BatchPutEnd` subsequently failing with `OBJECT_NOT_FOUND`

That is why this issue is still `Strong hypothesis` rather than `Confirmed`.

## 10. Fix Direction

First, add the smallest observability needed to close the lifecycle gap:
- in `ExistKey()`, distinguish "metadata truly absent" from "processing but not completed"
- in `PutStart()`, log key, age, client ID, and replica counts when the discard branch fires
- in `WaitForTransfers()`, log the total future count and current index for the op

Then, if the lifecycle proof confirms the mechanism, consider logic fixes such as:
- preventing upper layers from treating processing objects as ordinary missing objects
- revisiting the contract between `batch_is_exist` and `ExistKey`
- improving how long merged ops can remain delayed before finalize

## 11. Verification Plan

To consider this issue fixed or fully proven, the following should hold:

1. Reproducing the same workload no longer shows `TRANSFER_FAIL` batches later turning into `OBJECT_NOT_FOUND`.
2. `WaitForTransfers()` logs directly reveal future count and cumulative wait behavior per op.
3. `ExistKey()` logs distinguish "processing not completed" from "metadata absent."
4. `PutStart discard` logs reveal whether old processing metadata was actually overwritten.
5. If a logic fix is applied, `BatchPutEnd` no longer hits `OBJECT_NOT_FOUND` due to discarded old metadata.

## 12. Stable Conclusion Wording

Use the following wording as the single source of truth for this issue:

> In `prefill3`, `OBJECT_NOT_FOUND(-704)` is best explained as a secondary state-machine failure after `TRANSFER_FAIL(-800)`. Segment-level merged writes and serial 60-second waits in `WaitForTransfers()` can delay finalize for minutes; meanwhile, vLLM treats "still processing but not completed" objects as missing, and a later `PutStart` can discard the old metadata after 30 seconds. The delayed original `BatchPutEnd` then runs into `OBJECT_NOT_FOUND`.
