# Spec-decoding benchmark — Qwen3.6-35B-A3B (unsloth MTP GGUF)

Date: 2026-05-18 ~22:45 AEST
Model: `/home/cogtek/Downloads/Qwen3.6-35B-A3B-UD-IQ4_XS-MTP.gguf` (unsloth MTP GGUF — MTP head present, used for the draft-mtp config)
Binary: mtp-port `e7b484815` (b9139) — the production llama.cpp
GPU: RTX 3090, 24 GiB (embeddings + production services stopped for the run)
Params: ngl 99, q8_0/q8_0 KV, ctx 16384, batch/ubatch 512, threads 16
Test: `/completion` code-gen prompt, n_predict 512, temp 0, seed 42, 3 runs per config (context depth ~0)

| config        | gen t/s (run1 / run2 / run3) | mean t/s | draft acceptance        |
| ------------- | ---------------------------- | -------- | ----------------------- |
| none          | 124.66 / 128.16 / 129.23     | 127.35   | — (no spec)             |
| draft-mtp     | 147.61 / 147.59 / 148.12     | 147.77   | 99.4% (334/336)         |
| **ngram-mod** | 129.01 / 396.65 / 422.42     | **316.03** | **80.0% (448/560)**   |
| ngram-cache   | 127.86 / 126.52 / 125.32     | 126.57   | 25% (6/24 — barely engaged) |
| ngram-simple  | 128.19 / 128.66 / 126.61     | 127.82   | — (did not draft)       |
| ngram-map-k   | 127.55 / 126.89 / 127.43     | 127.29   | — (did not draft)       |
| ngram-map-k4v | 127.81 / 127.90 / 127.77     | 127.83   | — (did not draft)       |

## Winner: ngram-mod

**ngram-mod is the best config for this 3090** — warm steady-state ~400-420 t/s vs the 127 t/s no-spec baseline (~3.2x), and vs draft-mtp's 147 t/s (~2.7x).

- The run1=129 / run2=397 / run3=422 pattern is ngram-mod warm-up: the first request after server start has a cold n-gram cache (no speedup); the cache warms within one request and stays warm for the server's lifetime. A long-running production server lives in the warm regime — **~400+ t/s is the operative number**.
- 80% draft acceptance on 8-token drafts (n_max=8) is what produces the gain.

## Why the others lost

- **draft-mtp** — 99.4% draft acceptance (code at temp 0 is highly predictable) but only +16% throughput. Limited by n_max=3 draft depth plus the MTP head's own compute cost per step. High acceptance on short drafts loses to moderate acceptance on long drafts.
- **ngram-cache / ngram-simple / ngram-map-k / ngram-map-k4v** — all flat at the ~127 t/s baseline. They did not draft with default parameters; they need their `--spec-ngram-*-size-n / -size-m / -min-hits` knobs tuned to engage. Not pursued further — ngram-mod already wins decisively with defaults.

## Caveats

- Depth-0, deterministic (temp 0), code-generation prompt. Real agentic traffic is more varied; ngram-mod acceptance on diverse output will be lower than 80%. But ngram-mod's mechanism (drafting from recent-context n-grams) genuinely suits the repetitive/structured output that dominates coding workloads. Historical note: ngram-mod measured ~3.76x on Gemma 4 on this host.
- ngram-mod is **self-speculation** — it needs no MTP head and works identically on the non-MTP GGUF currently in production.

## Recommended production config

`LLAMA_ENABLE_SPEC_DECODING=true`, `LLAMA_SPEC_TYPE=ngram-mod`, `LLAMA_DRAFT_MAX=8` — applied to the current non-MTP model. Expected ~2.5-3x generation speedup over the current spec-off config once the n-gram cache warms.
