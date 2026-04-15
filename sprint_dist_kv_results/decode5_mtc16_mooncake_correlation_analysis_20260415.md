# Decode-5 MTC-16 Mooncake Error / Performance Correlation Analysis

Date: 2026-04-15

Related documents:
- `sprint_dist_kv_results/prefill6_mooncake_error_analysis_20260415.md`
- `bench_results/kimi_pd_p1d1to17_mooncake_nsys_autosweep/decode_5_mtc_16/attempt_1/*`

Scope:
- Main target run:
  - `bench_results/kimi_pd_p1d1to17_mooncake_nsys_autosweep/decode_5_mtc_16`
- Neighbor runs used for comparison:
  - `bench_results/kimi_pd_p1d1to17_mooncake_nsys_autosweep/decode_4_mtc_14`
  - `bench_results/kimi_pd_p1d1to17_mooncake_nsys_autosweep/decode_6_mtc_18`
- Goal:
  - Determine whether the Mooncake errors seen in `decode_5_mtc_16` are correlated with its abnormal performance.
  - Map the failing requests to the benchmark's multi-turn stages as far as local logs allow.
  - Compare prefill-side second-by-second metrics across `decode_4/5/6`.

Notes:
- All file paths below are repo-relative paths.
- All line numbers are based on the local workspace copies on 2026-04-15.
- This analysis uses both direct log inspection and small local counting scripts.
- The benchmark did not use `--save-detailed`, so no exact per-request benchmark trace exists for `conversation_id + turn_id` reconstruction.

## Executive Summary

`decode_5_mtc_16` shows a strong error/performance correlation.

Current best judgment:
- The abnormal slowdown is not a generic `decode=5` scaling issue.
- The dominant problem is a Mooncake write/publish failure window on the prefill side:
  - `metadata not found`
  - `Failed to open segment`
  - `Transfer submission failed for key`
  - `batch_put failed`
- Those failures cover almost exactly:
  - all failing requests in turn-band 1
  - all failing requests in turn-band 2
  - the earliest few requests in turn-band 3
- That timing lines up with the benchmark phase where early turns are supposed to seed external KV for later reuse.
- Neighbor runs `decode_4_mtc_14` and `decode_6_mtc_18` do not show the same failed-batch metrics and recover to materially higher external prefix hit rate.

Most likely causal chain:

`prefill-side Mooncake metadata / segment visibility failure -> KV publish fails for early requests -> next-turn external prefix reuse is unavailable or delayed -> Turn 2 / Turn 3 TTFT and E2E explode -> system gradually self-heals -> later turns recover`

## 1. Benchmark-Level Symptom

The high-level benchmark summary already shows that `decode_5` is the clear outlier relative to `decode_4` and `decode_6`.

| Case | Req/s | Out tok/s | Mean TTFT (s) | P90 TTFT (s) | Mean E2E (s) | P90 E2E (s) | Turn 1 TTFT (s) | Turn 2 TTFT (s) | Turn 3 TTFT (s) | Turn 4 TTFT (s) | Turn 5 TTFT (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `decode_4_mtc_14` | 1.44 | 430.53 | 6.88 | 13.31 | 9.62 | 15.58 | 27.18 | 12.05 | 3.54 | 3.77 | 3.61 |
| `decode_5_mtc_16` | 0.96 | 288.63 | 13.73 | 35.03 | 16.48 | 37.26 | 28.04 | 35.81 | 33.29 | 12.53 | 4.62 |
| `decode_6_mtc_18` | 1.42 | 426.24 | 9.81 | 26.93 | 12.49 | 31.53 | 31.07 | 20.80 | 5.65 | 5.65 | 5.80 |

Key observations:
- `decode_5` throughput drops sharply compared with both neighbors.
- `Turn 1` TTFT is not special. It is in the same ballpark as `decode_4` and `decode_6`.
- The real anomaly is `Turn 2` and `Turn 3`:
  - `decode_5 Turn 2 = 35.81s`
  - `decode_4 Turn 2 = 12.05s`
  - `decode_6 Turn 2 = 20.80s`
- `Turn 5+` of `decode_5` returns to a near-steady range around `4.5-4.8s`, which means the run is not permanently broken.

Relevant benchmark file:
- `bench_results/kimi_pd_p1d1to17_mooncake_nsys_autosweep/decode_5_mtc_16/attempt_1/hook_post_serve_0_vllm-bench.log:29`
- `bench_results/kimi_pd_p1d1to17_mooncake_nsys_autosweep/decode_5_mtc_16/attempt_1/hook_post_serve_0_vllm-bench.log:58`
- `bench_results/kimi_pd_p1d1to17_mooncake_nsys_autosweep/decode_5_mtc_16/attempt_1/hook_post_serve_0_vllm-bench.log:87`
- `bench_results/kimi_pd_p1d1to17_mooncake_nsys_autosweep/decode_5_mtc_16/attempt_1/hook_post_serve_0_vllm-bench.log:116`
- `bench_results/kimi_pd_p1d1to17_mooncake_nsys_autosweep/decode_5_mtc_16/attempt_1/hook_post_serve_0_vllm-bench.log:174`

## 2. What Error Type Appears in Decode-5

This run does not reproduce the exact `prefill6` handshake-timeout chain.

Not found in `decode_5_mtc_16`:
- `handshake timeout`
- `packet mismatch`
- `mark it inactive`
- `Failed to complete transfers after 60 seconds`

Found in `decode_5_mtc_16`:
- `metadata not found`
- `Failed to open segment`
- `Transfer submission failed for key`
- `failed: TRANSFER_FAIL`
- `batch_put failed`

Representative startup burst:
- `bench_results/kimi_pd_p1d1to17_mooncake_nsys_autosweep/decode_5_mtc_16/attempt_1/prefill-0/prefill-0-gb200-rack1-09.log:1413`
- `bench_results/kimi_pd_p1d1to17_mooncake_nsys_autosweep/decode_5_mtc_16/attempt_1/prefill-0/prefill-0-gb200-rack1-09.log:1415`
- `bench_results/kimi_pd_p1d1to17_mooncake_nsys_autosweep/decode_5_mtc_16/attempt_1/prefill-0/prefill-0-gb200-rack1-09.log:1416`

First batch-level failure cluster:
- `bench_results/kimi_pd_p1d1to17_mooncake_nsys_autosweep/decode_5_mtc_16/attempt_1/prefill-0/prefill-0-gb200-rack1-09.log:1918`
- `bench_results/kimi_pd_p1d1to17_mooncake_nsys_autosweep/decode_5_mtc_16/attempt_1/prefill-0/prefill-0-gb200-rack1-09.log:2033`

Interpretation:
- The failure mode here is closer to "Mooncake metadata / segment not visible when needed" than to the `prefill6` "handshake timeout then inactive endpoint" chain.

## 3. Failure Timeline vs Throughput Timeline

The prefill-side failure window lasts from:
- first observed failure: `16:46:02`
- last observed failure: `16:48:50`

Within that window:
- `69` unique failed request ids appear in `batch_put failed` warnings.
- Router completes only `68` total requests.

Using `router.log` completion timestamps:
- completion rate during failure window: `68 / 169s = 0.402 req/s`
- completion rate after failure window: `252 / 161s = 1.565 req/s`

That is a near-`3.9x` phase change in completion rate after the errors stop.

Interpretation:
- The correlation is not only temporal.
- The system enters a low-throughput phase exactly while Mooncake failures are active.
- Once the failure window ends, throughput rises sharply.

Important log references:
- last still-failing window:
  - `bench_results/kimi_pd_p1d1to17_mooncake_nsys_autosweep/decode_5_mtc_16/attempt_1/prefill-0/prefill-0-gb200-rack1-09.log:648304`
  - `bench_results/kimi_pd_p1d1to17_mooncake_nsys_autosweep/decode_5_mtc_16/attempt_1/prefill-0/prefill-0-gb200-rack1-09.log:648395`
- failure counters still present at:
  - `bench_results/kimi_pd_p1d1to17_mooncake_nsys_autosweep/decode_5_mtc_16/attempt_1/prefill-0/prefill-0-gb200-rack1-09.log:648397`

## 4. Mapping the 69 Failed Request IDs to Turn-Bands

### 4.1 Important Limitation

Exact mapping to `(conversation_id, turn_id)` is not possible from the available artifacts because:
- the benchmark result file is aggregated only
- proxy/router rewrites the original benchmark request id into:
  - `chatcmpl-___prefill_addr_<...>___decode_addr_<...>_<uuid>`
- the original benchmark `p1d5-mtc16-*` request ids are not preserved in the failure logs

Therefore this document uses a turn-band reconstruction:
- estimate which turn band a failed request belongs to by counting how many requests had fully completed before that failed request first appeared
- with `32` conversations:
  - completions `0-31` correspond to turn-band 1
  - completions `32-63` correspond to turn-band 2
  - completions `64-95` correspond to turn-band 3

### 4.2 Turn-Band Result

| Turn-band | Failed request count |
| --- | ---: |
| 1 | 32 |
| 2 | 32 |
| 3 | 5 |

This is the strongest single result in this analysis.

Interpretation:
- Every conversation's first request appears to fail its Mooncake publish path.
- Every conversation's second request also appears to fail its Mooncake publish path.
- The failure window then clips the earliest `5` requests of turn-band 3 before self-healing.

### 4.3 Distribution Across Decode Targets

| Turn-band | `rack1-10` | `rack1-11` | `rack1-12` | `rack1-13` | `rack1-14` |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 7 | 7 | 6 | 6 | 6 |
| 2 | 7 | 6 | 7 | 6 | 6 |
| 3 | 1 | 1 | 1 | 1 | 1 |

Interpretation:
- The failures are balanced across all decode workers.
- This is not consistent with a single decode node causing the benchmark anomaly.

## 5. Why This Matters for Multi-Turn Performance

This benchmark is specifically designed to benefit from prefix reuse:
- `multi-turn`
- `multi-turn-concurrency 16`
- `multi-turn-prefix-global-ratio 0.15`
- `multi-turn-prefix-conversation-ratio 0.75`
- `no_history_accumulation=true`

In other words:
- the benchmark repeatedly sends fixed-length prompts with large reusable prefixes
- if early-turn KV publish fails, the next turn cannot enjoy the intended external prefix reuse

That is exactly the pattern seen in the metrics:
- `Turn 1` is not abnormally worse than neighbors
- `Turn 2 / Turn 3` are the worst
- `Turn 5+` recovers after the write path has started healing

This strongly supports the causal interpretation:

`Turn 1 cache publish fails -> Turn 2 misses expected reuse -> Turn 2 cache publish also fails -> Turn 3 also suffers -> self-healing begins -> Turn 5+ mostly recovers`

## 6. Prefill Metrics Comparison: Decode-4 vs Decode-5 vs Decode-6

### 6.1 Failed Batch Metrics

| Case | Failed-metric seconds | First fail | Last fail | Max failed batches | Max failed keys |
| --- | ---: | --- | --- | ---: | ---: |
| `decode_4_mtc_14` | 0 | none | none | 0 | 0 |
| `decode_5_mtc_16` | 135 | `16:46:05` | `16:48:51` | 20 | 1761 |
| `decode_6_mtc_18` | 0 | none | none | 0 | 0 |

This is the cleanest second-by-second contrast between the three runs:
- `decode_4` and `decode_6` show no failed-batch metric seconds at all.
- `decode_5` shows a long contiguous failed-batch interval.

Representative `decode_5` metric lines:
- `bench_results/kimi_pd_p1d1to17_mooncake_nsys_autosweep/decode_5_mtc_16/attempt_1/prefill-0/prefill-0-gb200-rack1-09.log:38395`
- `bench_results/kimi_pd_p1d1to17_mooncake_nsys_autosweep/decode_5_mtc_16/attempt_1/prefill-0/prefill-0-gb200-rack1-09.log:52903`
- `bench_results/kimi_pd_p1d1to17_mooncake_nsys_autosweep/decode_5_mtc_16/attempt_1/prefill-0/prefill-0-gb200-rack1-09.log:648397`

### 6.2 External Prefix Cache Hit Rate Milestones

| Case | First `>=10%` | First `>=20%` | First `>=50%` | First `>=70%` | Final / max |
| --- | --- | --- | --- | --- | ---: |
| `decode_4_mtc_14` | `16:32:29 (13.3%)` | `16:32:53 (27.8%)` | `16:33:05 (51.2%)` | `16:33:34 (70.0%)` | `79.3%` |
| `decode_5_mtc_16` | `16:49:57 (13.4%)` | `16:50:02 (20.0%)` | `16:50:44 (50.1%)` | never | `61.7%` |
| `decode_6_mtc_18` | `17:03:41 (16.0%)` | `17:03:44 (24.6%)` | `17:04:26 (52.4%)` | `17:05:04 (70.1%)` | `79.3%` |

Interpretation:
- `decode_5` not only starts much worse.
- It also recovers more slowly and never reaches the same end-state reuse level as `decode_4` and `decode_6`.

Representative line references:
- `decode_4` reaches `79.3%`:
  - `bench_results/kimi_pd_p1d1to17_mooncake_nsys_autosweep/decode_4_mtc_14/attempt_1/prefill-0/prefill-0-gb200-rack1-09.log:5280`
- `decode_6` reaches `79.3%`:
  - `bench_results/kimi_pd_p1d1to17_mooncake_nsys_autosweep/decode_6_mtc_18/attempt_1/prefill-0/prefill-0-gb200-rack1-01.log:5112`
- `decode_5` is still `0.0%` at start:
  - `bench_results/kimi_pd_p1d1to17_mooncake_nsys_autosweep/decode_5_mtc_16/attempt_1/prefill-0/prefill-0-gb200-rack1-09.log:20662`
- `decode_5` climbs to `13.4%`, `20.0%`, and `50.1%` only much later:
  - `bench_results/kimi_pd_p1d1to17_mooncake_nsys_autosweep/decode_5_mtc_16/attempt_1/prefill-0/prefill-0-gb200-rack1-09.log:649504`
  - `bench_results/kimi_pd_p1d1to17_mooncake_nsys_autosweep/decode_5_mtc_16/attempt_1/prefill-0/prefill-0-gb200-rack1-09.log:649590`
  - `bench_results/kimi_pd_p1d1to17_mooncake_nsys_autosweep/decode_5_mtc_16/attempt_1/prefill-0/prefill-0-gb200-rack1-09.log:650358`
- `decode_5` ends at only `61.7%`:
  - `bench_results/kimi_pd_p1d1to17_mooncake_nsys_autosweep/decode_5_mtc_16/attempt_1/prefill-0/prefill-0-gb200-rack1-09.log:651071`

## 7. Self-Healing Pattern

The failure window does not stop all requests forever. The more precise pattern is:
- early stage:
  - very high per-request failed-key ratio
  - external prefix hit rate remains near `0%`
- middle stage:
  - failures continue, but some requests begin to complete
  - failed-key ratios remain high
- late failure stage:
  - failures become lighter
  - hit rate begins to rise
- post-failure stage:
  - failed-batch metrics disappear
  - completion rate increases sharply
  - external hit rate continues to climb, but not enough to catch up with `decode_4/6`

Example of early heavy failure:
- `88/113 keys failed`
  - `bench_results/kimi_pd_p1d1to17_mooncake_nsys_autosweep/decode_5_mtc_16/attempt_1/prefill-0/prefill-0-gb200-rack1-09.log:158220`
- `97/108 keys failed`
  - `bench_results/kimi_pd_p1d1to17_mooncake_nsys_autosweep/decode_5_mtc_16/attempt_1/prefill-0/prefill-0-gb200-rack1-09.log:158221`

Example of late lighter failure:
- `1/6 keys failed`
  - `bench_results/kimi_pd_p1d1to17_mooncake_nsys_autosweep/decode_5_mtc_16/attempt_1/prefill-0/prefill-0-gb200-rack1-09.log:648306`
- `2/7 keys failed`
  - `bench_results/kimi_pd_p1d1to17_mooncake_nsys_autosweep/decode_5_mtc_16/attempt_1/prefill-0/prefill-0-gb200-rack1-09.log:648304`

This suggests a gradual recovery in the publish path rather than a hard binary fail/open transition.

## 8. What This Is Not

This run does not support the following explanations:

1. "Router imbalance"
- Router selects decode workers evenly.
- Local counting from `router.log` shows `320` requests completed and `64` requests sent to each decode worker.

2. "A single decode node is broken"
- Failed request ids cover all five decode targets in near-equal counts.

3. "Decode kernel or TPOT regression"
- `mean TPOT` of `decode_5` remains near neighbors.
- Later turns recover to a reasonable steady range.

4. "Ordinary load scaling"
- `decode_4` and `decode_6` do not show the same failed-batch metrics.
- The outlier shape is therefore not explained by "slightly higher load" alone.

## 9. Most Likely Root Cause

Current best judgment:

The main issue is on the prefill-side Mooncake publish path, specifically around metadata / segment visibility or segment-open readiness.

Why this is the best fit:
- the first errors are `metadata not found`
- the next layer is `Failed to open segment`
- then `Transfer submission failed for key`
- then Python sees `batch_put failed`
- this persists exactly while throughput is suppressed and external hit rate stays near zero
- after the failure counters disappear, throughput and hit rate recover

The best concise root-cause statement is:

`decode_5_mtc_16` suffered a long prefill-side Mooncake publish failure window that prevented timely external KV reuse for the first two turn-bands, which in turn caused the benchmark's large Turn 2 / Turn 3 latency spike.

## 10. Recommended Next Steps

1. Re-run `decode_5_mtc_16` once with the same node set if possible.
- This checks whether the issue is transient metadata / readiness skew versus deterministic topology behavior.

2. Re-run with detailed request saving enabled if benchmark cost allows.
- `--save-detailed` would allow exact `(conversation_id, turn_id, request_id)` mapping.

3. Capture the same per-second prefill metrics on the rerun.
- In particular:
  - `mooncake_store_put_failed_batches`
  - `mooncake_store_put_failed_keys`
  - `External prefix cache hit rate`

4. Inspect why metadata 404 occurs for those 18 segment endpoints.
- This is the closest observable symptom to the likely root cause in this run.

5. If needed, compare `decode_5` against a second rerun of `decode_4` and `decode_6`.
- This would confirm whether `79.3%` external hit-rate is the expected healthy plateau for this benchmark family.

## Final Conclusion

Yes, the error and the performance anomaly are strongly correlated.

More strongly than that:
- the errors almost exactly cover the first two turn-bands
- those are the turns that should create and then consume reusable external KV
- neighbor runs do not show the same failed-batch metrics
- once the errors stop, throughput rises and later turns recover

So the best current conclusion is not merely "errors may contribute."

It is:

`decode_5_mtc_16` is best explained as a Mooncake publish/reuse failure episode during the first two multi-turn stages, and that episode is the main reason this point became the throughput and TTFT outlier of the decode-growth sweep.
