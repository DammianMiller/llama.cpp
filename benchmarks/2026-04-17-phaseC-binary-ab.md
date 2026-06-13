# Phase C — merged binary vs old binary, real-workload A/B

**Date:** 2026-04-17
**Workload:** `curl -d '{"content":"Write a Python function that implements a binary search tree..."}' max_tokens=512 temperature=0.0`
**Model:** Qwen3.6-35B-A3B-UD-IQ4_XS.gguf, f16/f16 KV, ctx 204800
**Hardware:** RTX 3090, production systemd config (ngram-mod spec decoding)

## Results

### NEW binary — `.worktrees/upgrade-b8740-turbo/build-q8mma/bin/llama-server`
Commit: `b0e905ac1` (phase A merge + ngram-mod persist + Phase B revert)

| run | time | tokens | ms/tok |
|---|---:|---:|---:|
| 1 | 8.66s | 512 | 16.9 |
| 2 | 8.49s | 512 | 16.6 |
| 3 | 8.48s | 512 | 16.6 |
| median | 8.49s | 512 | **16.6 ms/tok (59.9 t/s)** |

### OLD binary — `/home/cogtek/llama.cpp/build/bin/llama-server`
Built 2026-04-14 from `upgrade-b8740` (master + spec decoding fixes, no turbo/planar merge)

| run | time | tokens | ms/tok |
|---|---:|---:|---:|
| 1 | 7.60s | 486 | 15.6 |
| 2 | 8.30s | 512 | 16.2 |
| 3 | 6.68s | 489 | 13.7 |
| median | 7.60s | 489 | **15.6 ms/tok (64 t/s)** |

## Finding

**Merged binary is ~7% slower per-token** on this workload vs the old. Also note:
- NEW hit `max_tokens=512` every run (still generating when cap hit)
- OLD hit EOS in 2/3 runs at 486 and 489 tokens

Both produce coherent output; NEW appears to produce slightly more verbose responses
(output len was 2207 vs 1885 chars in run 1). This alone doesn't explain the
per-token slowdown — that points to something in the merged code path.

Suspected causes (not isolated):
- deferred activation replay interaction with f16 KV
- GPU multi-position sampling overhead when draft acceptance is high
- ngram-mod cache file compatibility between builds (both pointed at
  `~/.cache/uap/ngram-mod.bin` but format compatibility not verified)

## Decision

Prod stays on OLD binary (`/home/cogtek/llama.cpp/build/bin/llama-server`).
Merged worktree (`.worktrees/upgrade-b8740-turbo`, branch `upgrade-b8740-turbo`)
is preserved for future investigation or when turbo/planarquant types are
needed. Don't deploy merged binary until the ~7% regression is isolated.

## What the merge delivered (code-wise)

- 103 planarquant commits integrated (turbo/pq types, hybrid spec fixes, etc.)
- PlanarQuant/TurboQuant KV types available at runtime (TURBO2/3/4_0, PQ3/4_0)
- Deferred activation replay for hybrid spec decoding
- GPU multi-position sampling for speculative decoding
- Tree speculation hybrid-model fixes (upstream `examples/speculative` unblocked)
- ngram-mod persistence (`--spec-ngram-persist`, `--spec-ngram-reset-streak`)

## What still needs work for real spec-decoding improvements

- **Tree speculation port to `common/speculative.cpp`**: 10-15 days per planarquant's
  `SPEC_DECODING_ROADMAP.md`. Expected gain +80-120%. Not started. Upstream
  `examples/speculative/speculative.cpp` already has the tree logic but it's
  not callable from the server, which uses `common/speculative.cpp` (linear only).
- **Vec kernel q8_0 optimization** (closes f16/q8 tg gap, 18% perf): weeks
  of CUDA kernel work. Out of scope for config-level changes.
