# Handover: DeepSeek-V4.1 SWA bounded replay on ROCm

> Working notes for continuing on an AMD box. **Drop this commit before any PR.**
> Nothing has been posted to vllm-project; keep it that way until the AMD numbers are in.

## Goal

Re-enable encoder-side SWA bounded replay (#56227) on ROCm, which #57906 turned off after an HSA memory fault. Keep one implementation of the prefill index combine for all platforms, so the replay window clamp lives in exactly one kernel.

## Branch

`ivanium/vllm:feat/dsv41-encoder-swa-bounded-replay-amd`, based on main `d110c2f19c`.

| commit | what |
| --- | --- |
| `ca010c6e2d` | ROCm prefill calls the shared V4.1 Triton `combine_topk_swa_indices`; deletes the ROCm Torch combine and the unused V4-derived kernel copy; removes the ROCm guard in `DeepseekV4Attention.__init__`; widens the combine test. |
| `871752508b` | Shared kernel keeps local topk indices outside `[0, N)` at -1 (the Torch path did this; the shared kernel did not). New test `test_v41_combine_topk_swa_drops_invalid_topk`. **Cross-platform**: also changes the NVIDIA paths. |
| (this commit) | Handover notes. Drop it. |

## Why replay broke on ROCm

Everything replay needs already runs on ROCm through shared code, except one step:

| step | code | ROCm before this branch |
| --- | --- | --- |
| scheduler sets `replay_start` | `Scheduler._mark_prefix_replay` | shared ✓ |
| pad replayed slots in prefix-cacheable groups, hand `replay_start` to the SWA builders | `DeepseekV41ModelState`, `_pad_replayed_slots_kernel` (`amd/model.py` reuses it) | shared ✓ |
| compressed slot mapping skips PAD | `compressor_utils.py` | shared ✓ |
| prefill `gather_lens` start at `replay_start` | `ComputePrefillMetadataKernel` | shared ✓ |
| decode SWA indices clamp | `_compute_swa_indices_and_lens_kernel` | shared ✓ |
| **prefill topk + SWA index combine** | `CombineTopkSwaIndicesKernel` clamps the window at the gather start | **ROCm used its own Torch version, which did not clamp** ✗ |

Without the clamp, a replayed token's window reaches below the gathered SWA region. On most layers the index lands on the wrong rows and the output is silently wrong. On SWA-only layers (N = 0), for the first request of a chunk, the index goes negative, `build_ragged_indices_from_dense` turns it into -1, and OPUS (which does not check -1, see #58058) faults. That matches #57906's repro (multi-chunk prompt, which crosses OPUS's 1024-query gate). The OPUS link is inferred, not proven.

The Torch path existed because the V4-derived ROCm kernel computed `(pos + 1) // COMPRESS_RATIO` with V4.1's `compress_ratio == 0` SWA-only layers, which faulted on gfx950. The shared V4.1 kernel already guards that case (`cache_utils.py`, `if COMPRESS_RATIO > 0`), and it was already being warmed for `ROCM_FLASHMLA_SPARSE_DSV4` (`attention.py`, `register_warmup`) while never being called.

Torch vs Triton combine, one call on GB200 (4 prefills, 8192 tokens, topk 512): ~90 GPU ops + 7 host syncs, 1.7 ms host time, against 2 ops, 0 syncs, 49 µs. It runs once per layer per prefill chunk. #56638 measured -13% TTFT at 8K on MI355X from the same swap.

## Verified so far (GB200 only)

- `pytest tests/v1/attention/test_deepseek_v4_swa_visible.py`: 39 passed.
- Mutation check: with the filter removed from the kernel, only `test_v41_combine_topk_swa_drops_invalid_topk` fails.
- pre-commit (ruff, mypy, typos, …) clean on the touched files.
- **Nothing has run on ROCm yet.**

## To do on the AMD box, in order

The change is Python/Triton only, so no C++ rebuild is needed on top of a working ROCm dev install.

```bash
git fetch origin feat/dsv41-encoder-swa-bounded-replay-amd
git checkout feat/dsv41-encoder-swa-bounded-replay-amd
```

### 1. Unit tests, on gfx950 (MI355X) and gfx942 (MI300X) if available

```bash
python -m pytest tests/v1/attention/test_deepseek_v4_swa_visible.py -v
python -m pytest tests/kernels/attention/test_rocm_triton_attn_dsv4.py -v   # untouched V4 ROCm tests, regression only
```

Pass: all green. The combine test covers `compress_ratio` 0/1/2 (0 is the case that faulted the old kernel), a nonzero `query_start_loc` base, and 513-token requests (above the kernel's 256 workers per request).

### 2. Crash repro from #57906, replay on (the default)

Serve roughly as in #57906 (TP4, `--max-model-len 1048576`, `--max-num-batched-tokens 16384`, `--moe-backend aiter`, `--tokenizer-mode deepseek_v41`, add the DSpark `--speculative-config` you normally use). Check the server log does **not** print `SWA bounded replay is off on ROCm` (that message is gone on this branch; any other "SWA bounded replay is off" warning means another gate is active).

Then send prefix-sharing prompts, so every request after the first gets a prefix hit, a replay, and a multi-chunk prefill:

```python
import random, requests
MODEL = "deepseek-ai/DeepSeek-V4.1-Flash"
random.seed(0)
ids = [random.randint(1000, 30000) for _ in range(262144)]
for n in (4096, 32768, 131072, 262144):
    r = requests.post("http://127.0.0.1:8000/v1/completions",
                      json={"model": MODEL, "prompt": ids[:n], "max_tokens": 1})
    r.raise_for_status()
    print(n, "ok")
```

Pass: no `HSA_STATUS_ERROR_MEMORY_FAULT`, and `curl -s localhost:8000/metrics | grep prefix_cache_hits` is nonzero.

### 3. Accuracy: two-pass gsm8k, replay on vs off

Single-pass gsm8k barely exercises replay. Run the same eval twice against one server: pass 2 is all full-prompt prefix hits, so every request replays. Two arms: default (replay on) vs `--no-swa-bounded-replay`.

**Apply #58058 locally first.** Without it, partial prefix hits crash OPUS on -1 sentinels in both arms, which hides the signal:

```bash
git fetch upstream main pull/58058/head:pr-58058
git cherry-pick upstream/main..pr-58058   # local only, do not push
```

```bash
for pass in 1 2; do
  lm_eval --model local-completions \
    --model_args model=$MODEL,base_url=http://127.0.0.1:8000/v1/completions,num_concurrent=128,max_retries=3,tokenized_requests=False \
    --tasks gsm8k --num_fewshot 5 --output_path gsm8k_${ARM}_pass${pass}
done
curl -s localhost:8000/metrics | grep -E 'prefix_cache_(hits|queries)'
```

Pass: replay-on matches replay-off within noise, in both passes (±~1% at n=1319), with 0 request errors and a pass-2 hit rate around 90%. For reference, NVIDIA TP4 Flash gave replay 0.9272 / 0.9212 against baseline 0.9295 / 0.9196 (flexible-extract, pass 1 / pass 2). Compare the two arms on the same box, not against these numbers.

### 4. Where the invalid topk entries come from (optional, but it decides commit `871752508b`)

PR #58058 implies the ROCm indexer emits -1 inside `topk_len` under partial prefix hits; on NVIDIA we assume that never happens. To count them, temporarily paste this after the `combine_topk_swa_indices` call in `DeepseekV41ROCMAiterMLAAttention._forward_prefill`. It syncs, so use it for debugging only. After this branch the SWA part never yields -1, so any -1 within `combined_lens` comes from topk:

```python
cols = torch.arange(combined_indices.shape[1], device=q.device)
bad = int(((combined_indices < 0) & (cols[None] < combined_lens[:, None])).sum())
if bad:
    logger.warning("%s: %d invalid topk entries in a prefill chunk", self.prefix, bad)
```

- If entries show up: the filter is required on ROCm. The root cause is in the ROCm indexer prefill topk (`rocm_aiter_sparse_attn_indexer` / `top_k_per_row_prefill` row bounds) and deserves its own fix.
- If none show up even with prefix hits: the filter is defensive only. Decide whether to keep `871752508b` or drop it and keep the PR ROCm-only.

### 5. Perf

```bash
# prefix caching off isolates the combine swap (the #56638 setup); run main vs this branch
vllm bench serve --backend openai --base-url http://127.0.0.1:8000 --model $MODEL \
  --dataset-name random --random-input-len 8192 --random-output-len 1024 \
  --num-prompts 4 --max-concurrency 1 --seed $SEED --percentile-metrics ttft,tpot,e2el
```

Expect roughly #56638's -13% TTFT. Also do one run with prefix caching on and replay on to make sure the replayed chunk costs nothing unexpected.

## Before anything goes upstream

- **#56638** (JohnQinAMD, open) also re-enables a Triton combine on ROCm, but by fixing the ROCm-local kernel copy. It predates replay, has no gather-start clamp, and keeps two kernels. The two kernels have already drifted both ways: the copy lacks the cr=0 guard and the replay clamp, and the shared kernel lacked the topk filter. AGENTS.md requires coordinating before opening an overlapping PR. The plan is to comment on #56638 with the AMD numbers and propose the shared-kernel approach. The user decides when.
- If `871752508b` stays, rerun the NVIDIA two-pass gsm8k (GB200 TP4) before upstreaming, because it changes the FlashMLA / mega-attn paths too.
- Drop this handover commit, rebase on main, and write the PR per AGENTS.md: why it is not a duplicate (#56638, #57906), the tests run, the AMD eval table, and the AI-assistance line. No Claude session links.

## Out of scope (separate PRs, not needed for replay)

- Port the prefill chunk plan to V4.1 ROCm (#58405 does V4; its author plans the V4.1 port). At that point, pass a workspace `out=` into the combine like NVIDIA does. That also means growing the warmup reservation in `forward_mqa`'s `attn_metadata is None` branch.
- Emitting ragged indices straight from the combine, or caching per index source: saves only 3–4 small launches per layer. Not worth the complexity now.

## Misc

- On the GB200 box there is a local, unpushed stash `dsv41-amd-replay-before-rebase-20260922`: an earlier attempt that patched the clamp into the Torch path. It is superseded, ignore it.
- Push only to `origin` (the ivanium fork).
