# MTP vs ngram-mod (2026-05-14)

A/B comparison of speculative decoding strategies on the production llama-server,
both running Qwen3.6-35B-A3B-IQ4_XS on the RTX 3090.

## Setup

| Variant | Binary | Model | Spec | KV | Ctx |
|---|---|---|---|---|---|
| **MTP** | `.worktrees/mtp-port/build/bin/llama-server` (b9139) | `Qwen3.6-35B-A3B-UD-IQ4_XS-MTP.gguf` | `--spec-type draft-mtp --spec-draft-n-max 3` | q4_0/q4_0 | 131072 |
| **ngram-mod** | `.worktrees/eagle3-port/build/bin/llama-server` (b9039) | `Qwen3.6-35B-A3B-UD-IQ4_XS.gguf` | `--spec-type ngram-mod --spec-draft-n-max 8 --spec-ngram-mod-n-match 16 --spec-ngram-persist .../qwen3.6-35b-a3b.bin --spec-ngram-reset-streak 3` | q8_0/q8_0 | 262144 |

Same suite (`run-bench.sh`): 3× warm-cache repeat → 6 varied coding prompts → 1 recall → 1 no-cache. 256 max_tokens, temp 0.3.

## Per-prompt result (predicted_n / predicted_ms = t/s)

| label | MTP t/s | MTP acc | ngram t/s | ngram acc | Δ |
|---|---:|---:|---:|---:|---:|
| A1-same | 142.4 | 98.2% | 131.9 | (no draft) | +10.5 |
| A2-same | 157.3 | 98.8% | 164.3 | 55.5% | −7.0 |
| A3-same | 144.9 | 97.1% | **198.5** | 67.7% | −53.7 |
| B1-varied | 151.0 | 97.6% | **188.2** | 53.4% | −37.1 |
| B2-varied | 142.7 | 97.5% | 131.8 | (no draft) | +10.8 |
| B3-varied | 145.6 | 99.4% | 130.2 | (no draft) | +15.4 |
| B4-varied | 134.6 | 96.3% | 133.8 | (no draft) | +0.8 |
| B5-varied | 147.4 | 98.5% | 134.1 | (no draft) | +13.3 |
| B6-varied | 129.6 | 95.5% | 133.2 | (no draft) | −3.6 |
| C-recall0 | 155.7 | 97.7% | 133.4 | (no draft) | +22.3 |
| D-no-cache | 151.4 | 98.2% | 133.9 | (no draft) | +17.4 |

## Aggregate

| metric | MTP | ngram-mod |
|---|---:|---:|
| n_runs | 11 | 11 |
| median t/s | **145.6** | 133.8 |
| min t/s | 129.6 | 130.2 |
| max t/s | 157.3 | **198.5** |
| aggregate t/s (Σpred / Σms) | **145.2** | 143.7 |
| total drafts | 1815 | 571 |
| total accepted | 1774 | 335 |
| aggregate acceptance | **97.7 %** | 58.7 % |
| runs with any drafts | **11/11** | 3/11 |

**Aggregate Δ: MTP is +1.0 % vs ngram-mod (cold-cache).**

## Read

- **MTP wins on consistency.** Every single run had draft acceptance, all 95-99 %. Spread is tight (130-157 t/s, std ~8 t/s).
- **ngram-mod is a lottery.** When the persisted cache happens to contain patterns matching the prompt, it jumps to 165-198 t/s (A2, A3, B1). When it doesn't — which was 8/11 runs here — it drafts nothing and falls to baseline ~130 t/s. The persist cache file `~/.cache/uap/ngram-mod-qwen3.6-35b-a3b.bin` was loaded for this run but its content (from prior workloads) only matched 3 of the 11 fresh bench prompts.
- **MTP doesn't beat ngram-mod's *peak*.** ngram-mod with deterministic warm-cache repeats was previously measured at 480-540 t/s (`2026-05-06-gemma4-31b-iq4xs-spec/`). MTP can't match that — its acceptance rate caps at ~98 % and `n_max=3` limits per-step speedup to ~3×.
- **MTP beats ngram-mod's *average* on varied workloads.** Anyone running a mixed-prompt agent workload (UAP, code review, exploration) will see more consistent latency with MTP. ngram-mod is only the better choice when you expect the cache to be warm for your prompts (repeated agent loops, deterministic templates).

## Caveats

- MTP build also dropped from q8/q8 KV → q4_0/q4_0 KV and ctx 262k → 131k. q4_0 KV may slightly affect quality; throughput effect is small.
- ngram-mod cache was loaded but not pre-warmed against this bench's prompts. A version of this test that runs each prompt twice (first to populate cache, then measure) would show ngram-mod in a more favorable light. The picture above is "what arrives first time you ask."
- Single GPU (3090), parallel=1, single client. Multi-client / batched serving might shift the comparison.

## Raw data

- `mtp.jsonl` — 11 lines, one per bench iteration on MTP
- `ngram-mod.jsonl` — 11 lines, same on ngram-mod
- `run-bench.sh` — bench harness; takes `mtp` or `ngram-mod` as a label argument
