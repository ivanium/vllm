# Batch Size Sweet Spot Record: PD + Speculative Decoding

Date: 2026-04-14

## Purpose

Record one concrete observation about the batch-size/concurrency sweet spot for
PD serving on `nvidia/Kimi-K2.5-NVFP4` when comparing speculative decoding
against the non-speculative baseline.

This note is based on two vigil runs under:

- `third_partys/vigil/logs/pd_dev/2026-04-14/20260414_074458`
- `third_partys/vigil/logs/pd_dev/2026-04-14/20260414_082518`

## Setup

Shared setup across both runs:

- Model: `nvidia/Kimi-K2.5-NVFP4`
- Topology: PD disaggregation, `1P x 4 GPUs + 1D x 4 GPUs`
- Prefill: V1 runner, eager
- Decode: V2-style config with `FULL_DECODE_ONLY` cudagraph
- KV transfer: `NixlConnector` on decode, `NixlConnector + SimpleCPUOffloadConnector` on prefill
- Benchmark: `vllm-bench`, ShareGPT synthetic multi-turn workload
- Sweep: multi-turn concurrency `4, 8`

Key difference:

- `20260414_074458`: speculative decoding enabled with Eagle3, 3 draft tokens
- `20260414_082518`: speculative decoding disabled

Evidence:

- Spec run reproduce file:
  `third_partys/vigil/logs/pd_dev/2026-04-14/20260414_074458/REPRODUCE.md`
- Non-spec run reproduce file:
  `third_partys/vigil/logs/pd_dev/2026-04-14/20260414_082518/REPRODUCE.md`

## Main Result

For this PD setup, the sweet spot differs by concurrency:

- At lower batch pressure (`concurrency=4`), speculative decoding improves
  output throughput and TPOT, but hurts TTFT.
- At higher batch pressure (`concurrency=8`), speculative decoding is worse
  than the non-spec baseline on throughput, TTFT, TPOT, and E2EL.

In short:

- `concurrency=4` is still a viable region for speculation.
- `concurrency=8` is beyond the sweet spot for this speculative config.

## Benchmark Summary

### Concurrency = 4

Speculative (`20260414_074458`):

- Request throughput: `1.352 req/s`
- Output throughput: `399.47 tok/s`
- Mean TTFT: `960.72 ms`
- Mean TPOT: `6.55 ms`
- Mean ITL: `17.82 ms`
- Mean E2EL: `2854.32 ms`

Non-spec (`20260414_082518`):

- Request throughput: `1.195 req/s`
- Output throughput: `352.92 tok/s`
- Mean TTFT: `680.73 ms`
- Mean TPOT: `8.59 ms`
- Mean ITL: `8.60 ms`
- Mean E2EL: `3206.15 ms`

Interpretation:

- Spec improves output throughput by about `13%`
- Spec improves TPOT by about `24%`
- Spec reduces mean E2EL by about `11%`
- Spec worsens mean TTFT by about `41%`

### Concurrency = 8

Speculative (`20260414_074458`):

- Request throughput: `1.556 req/s`
- Output throughput: `459.87 tok/s`
- Mean TTFT: `1196.57 ms`
- Mean TPOT: `10.97 ms`
- Mean ITL: `30.19 ms`
- Mean E2EL: `4414.57 ms`

Non-spec (`20260414_082518`):

- Request throughput: `1.811 req/s`
- Output throughput: `535.08 tok/s`
- Mean TTFT: `773.31 ms`
- Mean TPOT: `9.60 ms`
- Mean ITL: `9.62 ms`
- Mean E2EL: `3599.09 ms`

Interpretation:

- Spec reduces output throughput by about `14%`
- Spec worsens TTFT by about `55%`
- Spec worsens TPOT by about `14%`
- Spec worsens mean E2EL by about `23%`

## Why This Happens

The speculative path is active and healthy in the spec-enabled run. Decode logs
show Eagle3 initialization and live acceptance metrics:

- Eagle3 auxiliary layers loaded
- Eagle speculator CUDA graph capture performed
- Average draft acceptance rate around `59%` to `61%`
- Mean acceptance length around `2.76` to `2.86`

This means speculation is not broken. It helps at lower concurrency because the
decode side gets faster token generation. But at higher concurrency, the extra
draft/verify work no longer pays for itself in this PD setup, so end-to-end
latency and throughput both regress.

Relevant log examples:

- `third_partys/vigil/logs/pd_dev/2026-04-14/20260414_074458/decode-0/decode-0-gb200-rack1-04.log`
- `third_partys/vigil/logs/pd_dev/2026-04-14/20260414_082518/decode-0/decode-0-gb200-rack1-08.log`

## Sweet Spot Takeaway

For this exact configuration:

- If optimizing for moderate concurrency, speculative decoding can be useful.
- If the operating point is closer to `concurrency=8`, the non-spec baseline is
  the better choice.

Operationally, the current sweet spot for speculation appears to be below
`concurrency=8`, with `concurrency=4` showing positive tradeoffs and
`concurrency=8` showing clear regression.

## Caveat

This is not a perfect node-controlled A/B test:

- Spec run used `gb200-rack1-03` and `gb200-rack1-04`
- Non-spec run used `gb200-rack1-07` and `gb200-rack1-08`

So node variance may contribute to the difference. Still, the observed trend is
consistent enough to treat this as a strong working hypothesis for batch-size
sweet-spot selection.

## Suggested Follow-up

- Repeat the same comparison on the same node pair
- Add one intermediate point, such as `concurrency=6`
- Test whether the sweet spot shifts with different speculative settings:
  `num_speculative_tokens`, draft model choice, or acceptance behavior
