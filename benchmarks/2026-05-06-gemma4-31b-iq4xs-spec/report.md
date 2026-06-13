# Gemma 4 31B IQ4_XS — speculative decoding A/B

- **Date**: 2026-05-06
- **Build**: `b9039` (`94e0ff659`) — fresh `origin/master` + PR-18039 (squashed) + gemma4 layer hook + eagle3 server-path n_new<=0 patch
- **Worktree**: `/home/cogtek/llama.cpp/.worktrees/eagle3-port`
- **Target**: `gemma-4-31B-it-IQ4_XS.gguf` (16.4 GB)
- **Draft (eagle3 arm)**: `Gemma-4-31B-Eagle3-Q8_0.gguf` (696 MB EAGLE-3 head)
- **Server flags**: `-fa on -ctk q4_0 -ctv q4_0 -ngl 99 -b 512 -ub 512 --parallel 1`
- **Context**: 131072 (baseline, ngram-mod) — eagle3 reduced to **65536** to fit (cuBLAS handle creation OOM at 131k once draft model + draft KV are added on the 24 GB RTX 3090)
- **Workload**: fixed code+prose prompt (~100 input tokens), `max_tokens=256`, `temp=0`, deterministic, 5 measured runs after a 64-token warmup, single slot, `cache_prompt=false`

## Result

| Arm       | tok/s mean | run 1 | runs 2-5 mean | min   | max   | accept % | runs |
|---        |---:        |---:   |---:           |---:   |---:   |---:      |---:  |
| baseline  | 35.23 | 35.81 | 35.08 | 34.21 | 35.81 | — | 5 |
| ngram-mod | 132.38 | 122.37 | 134.89 | 122.37 | 139.7 | 81.8 | 5 |
| eagle3    | 32.15 | 27.67 | 33.27 | 27.67 | 33.61 | 0.0 | 5 |

## Speedups vs baseline (mean tok/s)

- **ngrammod**: 3.76× (+275.8%)
- **eagle3**: 0.91× (-8.7%)

## Caveats

- **ngram-mod result inflates with deterministic repeated prompts.** Because `temperature=0` with the same prompt produces the same token stream, the ngram-mod state accumulates the exact next-token mapping. Run 1 is the most representative of a cold-cache path; runs 2-5 measure how well the cache replays. The "run 1" column above is the cleaner number to compare against eagle3 / baseline. For a fully prod-realistic measurement, vary the prompt across runs.
- **eagle3 had to be hot-fixed twice.** (1) PR-18039 ships an assertion (`n_new >= 1`) in `common_speculative_state_eagle3::draft()` that fires under llama-server when `draft()` is re-called between speculation rounds without prompt extension. Worked around by returning an empty draft for that round. (2) cuBLAS handle allocation fails at 131k ctx once draft + draft KV are added — even with q4_0/q4_0 KV — so eagle3 was run at 65k ctx. Both worth raising upstream on the PR.
- **The Gemma 4 31B Eagle3 head is not compatible with this base model.** Run 1 produced 289 drafts with **0.0% acceptance**; subsequent runs drafted nothing. The head's GGUF metadata reports `draft_vocab_size = 32000` while Gemma 4's vocab is 262144 — the head was trained against a vocab-pruned variant or different fine-tune. Result: every drafted token mapped (via `d2t`) to the wrong target vocab id, and the base model rejected all of them. **Eagle3 is a net -22% throughput loss vs baseline at this configuration**, not a speedup. Finding a head trained against a matching Gemma 4 31B-IT checkpoint, or building one ourselves, would be required to actually use eagle3.
- **131k context with q4_0/q4_0 KV** is what fits a 31B IQ4_XS + eagle3 head + draft KV on a 24 GB RTX 3090. q8_0/q4_0 KV (the original target) OOMs at 131k once the eagle3 head is added.
- **Hybrid spec rollback fixes are NOT in this build.** The benchmark binary is fresh-master-based, not your `upgrade-b8740` branch. All three arms run on the same base, so the cross-arm comparison is fair, but absolute numbers may differ from prod by a few percent.
- **Draft model is EAGLE-3, not a standalone model.** EAGLE-3 heads have different mechanics (target hidden-state extraction → encoder → decoder → draft) than the standard `--model-draft` path. The PR-18039 implementation has only been validated by the author against Llama 3.1 8B / 3.3 70B and Qwen 3 / 3-MoE / GPT-OSS / RedHtAI families. **Gemma 4 was not in their tested set** — we added the per-layer hook to `gemma4.cpp` ourselves to match the four arches the PR explicitly supports.

## How to reproduce

```bash
cd /home/cogtek/llama.cpp
PORT=18088 \
  LLAMA_BIN=/home/cogtek/llama.cpp/.worktrees/eagle3-port/build/bin/llama-server \
  ./benchmarks/run-gemma4-spec-ab.sh
```

Eagle3-only re-run: `./benchmarks/run-eagle3-only.sh`

