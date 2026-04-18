# Issue 1: Metadata / Stale Descriptor Visibility Failure

Date: 2026-04-16

## Issue Card

| Field | Value |
| --- | --- |
| Issue Number | Issue 1 |
| Short Name | metadata / stale descriptor |
| Current Status | `Strong hypothesis` |
| Affected Runs | Primarily `decode_5_mtc_16`; not observed in the same form in `decode_4_mtc_14` or `decode_6_mtc_18` |
| One-line Conclusion | The strongest current judgment is that shared Mooncake metadata contains stale object / segment references that point to endpoints outside the active job, causing the prefill-side external KV publish path to fail with `metadata not found -> Failed to open segment -> TRANSFER_FAIL`. |
| Related Issues | [Issue 2: RDMA Handshake and Endpoint State](prefill6_mooncake_error_analysis_20260415.md); [Issue 3: Processing Object Re-put and OBJECT_NOT_FOUND](prefill3_decode5_mooncake_deep_dive_20260415_cn.md); [Issue 4: Observability, Error Accounting, and Diagnostic Blind Spots](batch_put_failed_global_analysis_20260415.md) |

## 1. Problem Definition

This issue refers to the following failure pattern:
- the prefill side publishes external prefix blocks through `MooncakeStoreConnector`
- object metadata ultimately points at a set of `transport_endpoint_` values that do not belong to the active job
- the transfer engine queries the shared metadata server for segment descriptors for those endpoints and repeatedly gets `404 metadata not found`
- the failure then propagates as `Failed to open segment`, `Transfer submission failed for key`, and finally `batch_put failed`

This issue does not include:
- the `handshake timeout / packet mismatch / inactive endpoint` chain, which belongs to [Issue 2](prefill6_mooncake_error_analysis_20260415.md)
- the `TRANSFER_FAIL -> OBJECT_NOT_FOUND(-704)` amplification chain, which belongs to [Issue 3](prefill3_decode5_mooncake_deep_dive_20260415_cn.md)
- logging gaps, metrics gaps, and error accounting problems, which belong to [Issue 4](batch_put_failed_global_analysis_20260415.md)

## 2. Current Status

- Evidence strength: `Strong hypothesis`
- Priority: high
- Single source of truth for the conclusion:
  - `decode_5_mtc_16` is best explained as shared Mooncake stale object / segment descriptor contamination, not as a transient in-run handshake wobble or a single broken decode worker.

Facts that are already stable:
- the failure window is driven by prefill-side Mooncake publish errors, not by a decode-side Mooncake error chain
- the failure window is strongly correlated with suppressed throughput and abnormal TTFT / E2E
- the metadata-miss endpoint set is stable and all endpoints lie outside the active job node set
- the endpoints that the job itself actually listens on are different from the repeatedly failing endpoints

## 3. Affected Scope

Primary run:
- `bench_results/kimi_pd_p1d1to17_mooncake_nsys_autosweep/decode_5_mtc_16/attempt_1`

Control runs:
- `decode_4_mtc_14`: no comparable failed-batch window
- `decode_6_mtc_18`: no comparable failed-batch window

Impact:
- directly affects the prefill-side external KV publish path
- indirectly affects multi-turn external prefix reuse on the decode side
- shows up at the benchmark level as a large regression in `Turn 2` and `Turn 3` TTFT / E2E

Role boundary:
- the prefill role uses `MultiConnector`, containing `NixlConnector` and `MooncakeStoreConnector`
- the decode role uses `NixlConnector` only
- therefore the root-cause logs live on the prefill side, while the decode side mostly shows consequences

## 4. Main Symptoms and Log Fingerprint

The most stable log fingerprint is:

```text
metadata not found
  -> Failed to retrieve segment descriptor
  -> Failed to open segment
  -> Transfer submission failed for key
  -> failed: TRANSFER_FAIL
  -> batch_put failed
```

Representative keywords:
- `metadata not found`
- `Failed to retrieve segment descriptor`
- `Failed to open segment`
- `Transfer submission failed for key`
- `failed: TRANSFER_FAIL`
- `batch_put failed`

Representative locations:
- first 404: `bench_results/kimi_pd_p1d1to17_mooncake_nsys_autosweep/decode_5_mtc_16/attempt_1/prefill-0/prefill-0-gb200-rack1-09.log:1413`
- local metadata server 404 example: `.../prefill-0-gb200-rack1-09.log:1550`
- first batch-level warning cluster: `.../prefill-0-gb200-rack1-09.log:1918`
- late-window 404: `.../prefill-0-gb200-rack1-09.log:648381`

## 5. Strongest Current Mechanism Chain

The strongest current explanation is:

1. `decode_5_mtc_16` relies on the prefill side to publish external prefix blocks through `MooncakeStoreConnector`.
2. Some object metadata resolves to endpoint descriptors that do not belong to the current job.
3. `openSegment()` asks the shared metadata server for segment descriptors for those endpoints.
4. The shared metadata server repeatedly returns `404 metadata not found`.
5. `submit_batch()` cannot open the segment and fails with `Transfer submission failed for key` and `TRANSFER_FAIL(-800)`.
6. Early-turn KV publish fails, so the next turns do not get the intended external prefix reuse and `Turn 2 / Turn 3` TTFT and E2E spike.

What is fact vs inference:
- Steps 1, 3, 4, and 5 are directly supported by source and logs.
- Step 2, specifically the stale / out-of-job descriptor interpretation, is the strongest current inference.
- Step 6 is a strong causal interpretation supported by turn-band reconstruction and comparison against neighboring runs.

## 6. Key Evidence

### 6.1 Benchmark-level anomaly

`decode_5_mtc_16` is the clear outlier in the decode sweep:

| Case | Req/s | Mean TTFT (s) | Mean E2E (s) | Turn 2 TTFT (s) | Turn 3 TTFT (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `decode_4_mtc_14` | 1.44 | 6.88 | 9.62 | 12.05 | 3.54 |
| `decode_5_mtc_16` | 0.96 | 13.73 | 16.48 | 35.81 | 33.29 |
| `decode_6_mtc_18` | 1.42 | 9.81 | 12.49 | 20.80 | 5.65 |

Important observation:
- `Turn 1` is not special
- `Turn 2` and `Turn 3` are where the benchmark diverges badly
- `Turn 5+` recovers to a much more normal range

### 6.2 Failure timeline versus throughput timeline

Observed failure window:
- first failure: `16:46:02`
- last failure: `16:48:50`

Within that window:
- `69` unique failed request IDs appear in `batch_put failed`
- router completion rate is only `68 / 169s = 0.402 req/s`

After the window:
- router completion rate rises to `252 / 161s = 1.565 req/s`

That is a near-`3.9x` throughput phase change immediately after the Mooncake failure window ends.

### 6.3 Failed requests almost exactly cover the first two turn-bands

Reconstructed turn-band mapping:

| Turn-band | Failed request count |
| --- | ---: |
| 1 | 32 |
| 2 | 32 |
| 3 | 5 |

Interpretation:
- every conversation's first request appears to fail the Mooncake publish path
- every conversation's second request also appears to fail it
- the failure window clips the earliest part of turn-band 3 before self-healing begins

### 6.4 Failures are evenly distributed across decode targets

| Turn-band | `rack1-10` | `rack1-11` | `rack1-12` | `rack1-13` | `rack1-14` |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 7 | 7 | 6 | 6 | 6 |
| 2 | 7 | 6 | 7 | 6 | 6 |
| 3 | 1 | 1 | 1 | 1 | 1 |

This rules out:
- a single broken decode worker
- router imbalance as the primary explanation

### 6.5 External prefix cache hit-rate milestones

| Case | First `>=10%` | First `>=20%` | First `>=50%` | First `>=70%` | Final / max |
| --- | --- | --- | --- | --- | ---: |
| `decode_4_mtc_14` | `16:32:29 (13.3%)` | `16:32:53 (27.8%)` | `16:33:05 (51.2%)` | `16:33:34 (70.0%)` | `79.3%` |
| `decode_5_mtc_16` | `16:49:57 (13.4%)` | `16:50:02 (20.0%)` | `16:50:44 (50.1%)` | never | `61.7%` |
| `decode_6_mtc_18` | `17:03:41 (16.0%)` | `17:03:44 (24.6%)` | `17:04:26 (52.4%)` | `17:05:04 (70.1%)` | `79.3%` |

This matters because it shows:
- `decode_5` starts worse
- `decode_5` recovers later
- `decode_5` also ends at a worse steady-state hit-rate plateau

### 6.6 The recovery is gradual, not binary

The failure window is not "fully broken" followed by "fully healthy."

Observed pattern:
- early stage: very high per-request failed-key ratios and near-zero external hit-rate
- middle stage: failures still happen, but some requests begin to complete
- late stage: failures get lighter and hit-rate begins to rise
- post-window: failed-batch metrics disappear, but the final hit-rate still does not catch up to healthy controls

Representative examples:
- early heavy failure:
  - `88/113 keys failed`
  - `97/108 keys failed`
- late lighter failure:
  - `1/6 keys failed`
  - `2/7 keys failed`

This is much more consistent with a gradually recovering publish path than with a one-shot open/close event.

### 6.7 The miss endpoints do not belong to the active job

`slurm.out` shows that this job ran on:
- `gb200-rack1-[09-14]`

Therefore the active job-side IP set is:
- `192.168.0.109`
- `192.168.0.110`
- `192.168.0.111`
- `192.168.0.112`
- `192.168.0.113`
- `192.168.0.114`

The prefill side itself actually listens on:
- `192.168.0.109:16961`
- `192.168.0.109:16663`
- `192.168.0.109:16571`
- `192.168.0.109:16725`

But the fixed metadata-miss endpoint set is:
- `192.168.0.101`
- `192.168.0.103`
- `192.168.0.104`
- `192.168.0.107`
- `192.168.0.117`
- `192.168.0.118`

Top miss endpoints:

| Count | Endpoint |
| ---: | --- |
| 6412 | `192.168.0.117:16379` |
| 6225 | `192.168.0.104:16234` |
| 6176 | `192.168.0.103:16462` |
| 6089 | `192.168.0.118:15418` |
| 5990 | `192.168.0.118:15530` |
| 5982 | `192.168.0.117:16214` |
| 5958 | `192.168.0.101:15446` |
| 5951 | `192.168.0.101:15426` |
| 5948 | `192.168.0.107:15220` |
| 5900 | `192.168.0.117:15721` |
| 5879 | `192.168.0.101:15580` |
| 5860 | `192.168.0.101:16860` |
| 5822 | `192.168.0.107:15843` |
| 5806 | `192.168.0.107:16944` |
| 5804 | `192.168.0.117:15945` |
| 5747 | `192.168.0.107:16075` |
| 5724 | `192.168.0.118:16719` |
| 5721 | `192.168.0.118:15089` |

This is one of the strongest reasons the stale-descriptor explanation is stronger than simple propagation delay.

### 6.8 The source-level 404 semantics are direct

Shared Mooncake config points to:
- metadata server: `http://192.168.0.101:8080/metadata`
- master server: `192.168.0.101:50051`

Relevant source chain:
- `http_metadata_server.cpp:25`: GET `/metadata?key=...` returns 404 if the key is not in the local `store_`
- `transfer_engine_impl.cpp:434`: `openSegment()` resolves the segment descriptor
- `transfer_metadata.cpp:878`: metadata backend lookup failure logs `Failed to retrieve segment descriptor`
- `transfer_task.cpp:481`: `submit_batch()` fails immediately if `openSegment()` fails

This means the precise statement is:
- not "the segment exists but the transfer engine cannot open it"
- but "the transfer engine cannot find the segment descriptor in the shared metadata store in the first place"

## 7. What This Is Not

The current evidence does not support these explanations:

1. This is not the [Issue 2](prefill6_mooncake_error_analysis_20260415.md) handshake-timeout family.
2. This is not a single decode node problem.
3. This is not ordinary decode scaling regression.
4. This is not a decode kernel or TPOT regression.

## 8. Relationship to Other Issues

- Relation to [Issue 2](prefill6_mooncake_error_analysis_20260415.md):
  - both can end in `TRANSFER_FAIL(-800)` and `batch_put failed`
  - but Issue 1 is driven by `metadata not found / open segment failed`
  - Issue 2 is driven by `handshake timeout / inactive endpoint`

- Relation to [Issue 3](prefill3_decode5_mooncake_deep_dive_20260415_cn.md):
  - Issue 3 explains how some `TRANSFER_FAIL` paths later become `OBJECT_NOT_FOUND(-704)`
  - that is not the center of the `decode5` failure family

- Relation to [Issue 4](batch_put_failed_global_analysis_20260415.md):
  - if metadata `PUT/DELETE` and descriptor lifecycle logging are added, this issue is one of the best candidates to be upgraded from `Strong hypothesis` to `Confirmed`

## 9. What Is Still Unproven

The biggest remaining ambiguity is:
- were these descriptors never successfully registered, or were they registered and later deleted?

What is still missing:
- successful metadata `PUT / DELETE` logs
- successful `updateSegmentDesc()` logs
- successful `removeSegmentDesc()` logs
- lifecycle traces that connect object metadata to exact endpoint descriptors

## 10. Fix Direction

Recommended order:

First, add the minimum observability needed to close the lifecycle gap:
- successful metadata `PUT / DELETE` logs with key and store size before / after
- successful descriptor registration / removal logs in `transfer_metadata.cpp`
- explicit linkage between object metadata and `transport_endpoint_`

Then, and only then, narrow logic fixes such as:
- stronger run isolation / namespace isolation for shared metadata
- debugging why object metadata points at out-of-job endpoints
- checking whether descriptor cleanup runs too early while objects still reference them

## 11. Verification Plan

To consider this issue fixed, all of the following should hold:

1. Re-running `decode_5_mtc_16` or an equivalent workload no longer produces the same out-of-job 404 endpoint set.
2. The chain `metadata not found -> Failed to open segment -> Transfer submission failed for key` disappears.
3. `mooncake_store_put_failed_batches` and `mooncake_store_put_failed_keys` no longer stay elevated in the early window.
4. `Turn 2 / Turn 3` TTFT returns to the range of the healthy neighbor runs.
5. External prefix hit-rate returns to a healthy plateau instead of topping out at `61.7%`.

## 12. Stable Conclusion Wording

Use the following wording as the single source of truth for this issue:

> `decode_5_mtc_16` is best explained by stale object / segment descriptor contamination in shared Mooncake metadata. The active job only runs on `192.168.0.109-114`, but the repeatedly failing endpoints all lie outside that set. On the prefill side, external KV publish repeatedly hits `metadata not found -> Failed to open segment -> TRANSFER_FAIL`, which aligns closely with the first two failed turn-bands and the large Turn 2 / Turn 3 latency spike.
