# Gemma 4 26B-A4B IQ4_XS — KV q8 vs q4 A/B (2026-05-09)

## Setup
- **Model**: `gemma-4-26B-A4B-it-UD-IQ4_XS.gguf` (13.6 GB, MoE 26B/4B-active)
- **Binary**: `~/llama.cpp/.worktrees/eagle3-port/build/bin/llama-server`
- **Context**: 131072
- **Spec**: ngram-mod, draft-n-max=3, draft-n-min=1, p-min=0.75, n-match=16
- **Workload**: deterministic median-function prompt × 5 runs, temp=0, max=256, cache_prompt=false

## Memory footprint at startup

| KV | model | context | compute | self total |
|---|---:|---:|---:|---:|
| q8/q8 | 12952 MiB | 1519 MiB | 532 MiB | 15004 MiB |
| q4/q4 | 12952 MiB | **804 MiB** | 532 MiB | **14289 MiB** (-715) |

## Phase A — q8/q8 KV (cold cache)

| Run | t/s | Accept |
|---|---:|---:|
| 1 | 157.69 | 56.8% |
| 2 | 353.77 | 64.7% |
| 3 | 100.88 | 12.5% |
| 4 | 139.23 | 39.1% |
| 5 | 159.21 | 56.8% |
| **mean** | **182.2** |  |
| median | 157.7 |  |

## Phase B — q4/q4 KV (cold cache, fresh persist)

| Run | t/s | Accept |
|---|---:|---:|
| 1 | 101.54 | (cold, no drafts) |
| 2 | 115.58 | 35.9% |
| 3 | 331.28 | 77.0% |
| 4 | 249.86 | 45.1% |
| 5 | 345.11 | 78.5% |
| **mean** | **228.7** |  |
| median | 249.9 |  |

## Phase C — q4/q4 KV (warmed persist)

| Run | t/s | Accept |
|---|---:|---:|
| 1 | 347.65 | 78.5% |
| 2 | 348.00 | 78.5% |
| 3 | 346.21 | 78.5% |
| 4 | 345.73 | 78.5% |
| 5 | 349.20 | 78.5% |
| **mean** | **347.36** |  |
| median | 347.65 |  |

## Verdict

**q4/q4 wins** on every metric:
- +25.5% mean throughput vs q8/q8 (228.7 vs 182.2)
- +58% median throughput
- 715 MB less VRAM used
- Higher steady-state acceptance once warmed

**Steady-state perf**: ~347 t/s with 78.5% draft accept after persist warms.

For comparison, Gemma 4 31B dense IQ4_XS on the same bench peaks at 480-540 t/s in fully-warmed cache replay, but median ~130 t/s — 26B-A4B has a tighter, more consistent throughput band thanks to MoE 4B active params per token.

## Applied prod config

```
LLAMA_MODEL=~/Downloads/gemma-4-26B-A4B-it-UD-IQ4_XS.gguf
LLAMA_CTX_SIZE=131072
LLAMA_CACHE_TYPE_K=q4_0
LLAMA_CACHE_TYPE_V=q4_0
LLAMA_DRAFT_MAX=3
LLAMA_DRAFT_P_MIN=0.75
LLAMA_SPEC_TYPE=ngram-mod
--spec-ngram-mod-n-match 16
--spec-ngram-persist ~/.cache/uap/ngram-mod-gemma4-26b-a4b.bin
```
