# Draft-param tuning at ctx 131072 + tree spec

**Date:** 2026-04-17
**Binary:** `.worktrees/upgrade-b8740-turbo/build-q8mma/bin/llama-server` (commit `4c91473bf`)
**Config:** ctx 131072, `--spec-parallel 2 --spec-ngram-mod-k 4`, f16 KV, Qwen3.6-35B-A3B
**Workload:** BST code-gen, `max_tokens=512 temperature=0.0`, median of 3 runs after 2-run warmup

## Results

| config                                                    | t/s   | accept | Δ vs prior |
|-----------------------------------------------------------|------:|-------:|-----------:|
| draft_min=2 draft_max=8 p_min=0.75 (first test)           | 51.7  | 35.7 % | —          |
| draft_min=1 draft_max=3 p_min=0.75                        | 80.1  | 59.1 % | **+55 %**  |
| draft_min=1 draft_max=3 p_min=0.80 (**current live**)     | **81.8** | **59.1 %** | **+58 %** |

## Why the tuning matters more than the model

Shrinking `draft_max` from 8 → 3 made the biggest single jump (+55 %):
- With 8 drafts per decode, we proposed a lot of speculative tokens. At
  ~35 % draft acceptance, that's ~5 wasted drafts per decode burning GPU
  time on attention that contributes zero accepted tokens.
- With 3 drafts per decode, the per-decode batch is much smaller (4 tokens
  vs 9), each decode is cheaper, and the draft-acceptance rate jumped to
  59 % because earlier drafts are the strongest predictions.
- Net: more real tokens per second despite fewer drafts per decode.

Raising `p_min` from 0.75 → 0.80 was a small win (+2 %) — p_min filters
draft-model continuations but is mostly a no-op for ngram-mod drafting
where the draft function doesn't emit calibrated probabilities. Kept
because it doesn't hurt.

## How this compares to the old prod baseline

| config                                        | ctx_alloc | n_ctx_per_seq | t/s   |
|-----------------------------------------------|----------:|--------------:|------:|
| linear @ 204800 (prior prod)                  | 204 800   | 204 800       | 59.6  |
| tree @ 131072 tuned (current live)            | 131 072   |  65 536       | **81.8** |

**+37 % vs prior prod throughput**, but **half the effective ctx per request**
(65 536 vs 204 800).

## Currently live

```
LLAMA_CTX_SIZE=131072
LLAMA_DRAFT_MAX=3
LLAMA_DRAFT_MIN=1
LLAMA_DRAFT_P_MIN=0.8
--spec-parallel 2
--spec-ngram-mod-k 4
--spec-ngram-persist ~/.cache/uap/ngram-mod-k4.bin
```

VRAM: 22.4 / 24.1 GiB, 1.7 GiB free. Lots of headroom vs 204k (374 MiB).

## Why tree-spec wins now

- `copy_cell` D2D fix (commit `4c91473bf`) eliminates the 3 × CPU round-trip
  overhead that previously dominated multi-seq hybrid workloads.
- Small draft budget (3) means the multi-branch batch is compact, keeping
  attention cost per decode close to non-spec.
- ngram-mod k=4 drafter's top-K branching gets exercised: each decode can
  accept tokens from whichever branch the target model agrees with.

## Things to try next (not done)

1. `--spec-parallel 4` with same draft budget — more branches, wider tree.
   May or may not help at ctx 131k since per-decode cost grows with batch.
2. `--spec-ngram-mod-k 6 or 8` — more candidates per bucket. Costs memory
   (k=8 = 128 MB ngram-mod.bin vs k=1 = 16 MB) but unlocks higher acceptance.
3. Running longer to warm the k=4 ngram-mod persisted cache — acceptance
   at 59 % is already strong; with months of prod traffic it could hit
   70-80 % like the old k=1 prod cache sometimes reports.
