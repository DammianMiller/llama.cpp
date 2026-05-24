# Gemma 4 31B — `draft-p-min` sweep (2026-05-24)

Target: `gemma-4-31B-it-IQ4_XS.gguf` (dense 31B, arch gemma4). Single RTX 3090 24 GB.
Binary: eagle3-port (PR #18039, b9039 base). Spec: ngram-mod, n-max 3.
KV q4_0/q4_0, ctx 131072, --parallel 1.
Workload: 3 prompts × 3 runs each (cold/warm/warm), 256 output tokens, temp 0.
Method: env-driven restart per `p_min` value; same prompts; total downtime ~6.5 min.

## Headline

**Current production `p_min = 0.80` is at the optimum plateau. Do NOT change.** A sweep across {0.65, 0.70, 0.75, 0.80} shows a flat curve between 0.70 and 0.80 (all within ~2% of each other) and a steep cliff below 0.70.

## Aggregate warm throughput (run 2 + run 3, avg)

| p_min | warm avg t/s | min | max | n  | vs 0.80 |
|------:|-------------:|----:|----:|---:|--------:|
| 0.65  |   21.82      | 11.84 | 37.38 | 6 | **−62%** ❌ |
| 0.70  |   57.44      | 35.57 |104.22 | 6 | +1% |
| 0.75  |   58.08      | 34.90 |105.77 | 6 | **+2%** (marginal best) |
| 0.80  |   56.90      | 35.74 |101.28 | 6 | baseline (current prod) |

## Per-prompt warm avg

| p_min | codegen-repetitive | codereview-medium | explain-novel |
|------:|-------------------:|------------------:|--------------:|
| 0.65  |  30.28             |  22.36            |  12.83        |
| 0.70  |  82.26             |  50.38            |  39.66        |
| 0.75  |  84.42             |  50.09            |  39.73        |
| 0.80  |  81.39             |  49.65            |  39.67        |

## Cold/warm gap (run 1 vs run 2+3, avg across prompts)

| p_min | cold t/s | warm t/s | gap |
|------:|---------:|---------:|----:|
| 0.65  | 13.97    | 21.82    |  7.85 |
| 0.70  | 25.27    | 57.44    | 32.16 |
| 0.75  | 29.80    | 58.08    | 28.29 |
| 0.80  | 29.59    | 56.90    | 27.31 |

Cold performance is dominated by no-cache state regardless of `p_min`. The cold-vs-warm gap of ~30 t/s at any p_min ≥ 0.70 is what PR #23398 (Gemma 4 MTP, [research note](../2026-05-24-gemma4-spec-research/)) is meant to close.

## Findings

1. **`p_min < 0.70` is a performance crater.** At 0.65, all three prompt types tank to roughly half the throughput of 0.70+. Likely mechanism: the lower threshold lets through low-probability drafts that mostly get rejected, eating compute without recovering tokens. The Tarlov L40 paper's "0.70 optimal" recommendation **does NOT replicate** on this 3090 + ngram-mod + IQ4_XS Gemma 4 31B stack — the curve below 0.70 falls off, not the other way around.

2. **`p_min` 0.70 → 0.75 → 0.80 is a flat plateau.** Differences (56.9, 57.4, 58.1 t/s) are well within run-to-run noise. `0.75` is the nominal arithmetic max but the ~2% gain over `0.80` is not worth a config change.

3. **Real-world agentic perf (`codereview-medium`, `explain-novel`) sits at ~40–50 t/s warm.** The 80–100 t/s numbers are only on the codegen-repetitive prompt — which is exactly the ngram-friendly regime where the May bench reported 129 t/s on an even-more-repetitive prompt. Sub-content matters.

4. **Cold runs (~25–30 t/s at any p_min ≥ 0.70)** confirm the prior assessment: ngram-mod cold-misses on novel content. Closing that gap is what an actual draft model (PR #23398 MTP) would do.

## Recommendation

**Keep `LLAMA_DRAFT_P_MIN=0.80`.** No production change.

The earlier research suggestion to "try 0.70 per Tarlov" was wrong for this hardware/config — the empirical sweep refutes it. The current value is on the optimum plateau and matches the May-bench-validated config.

**Future-proofing notes:**
- If model/quant/binary changes (e.g., PR #23398 lands and we switch to MTP), redo this sweep. The p_min curve will likely shift, and the new optimum could differ.
- If running on substantially novel content (less repetitive than the codegen prompt), expect ~40 t/s sustained — that's just the ngram-mod ceiling on cold content.

## Caveats

- The `draft acceptance rate` numbers in `results.jsonl` are cumulative-since-server-restart values from llama-server's own log; they show identical 141/185-type fractions across `p_min` values at the same run, which is implausible if `p_min` actually changes draft composition. The TPS numbers are wall-time-derived and are the ground truth for this report. The accept-rate parsing is suspect and would need a more careful per-prompt extraction in a future bench — but it doesn't affect the conclusion (TPS clearly varies with p_min in the expected way).
- Single prompt-set, single binary. A larger prompt suite would reduce noise on the 0.70/0.75/0.80 plateau, but the dramatic 0.65 collapse is robust enough to act on with the data we have.

## Files

- `prompts.json` — 3 prompts used
- `run_sweep.sh` — the sweep harness (env-driven restart per value, trap-based prod restore)
- `results.jsonl` — 36 rows (4 p_min × 3 prompts × 3 runs)
- `run.log` — wrapper log
- `run_full2.log` — full sweep stdout
