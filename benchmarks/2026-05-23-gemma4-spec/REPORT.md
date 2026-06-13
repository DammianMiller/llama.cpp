# Gemma 4 31B — speculative-decoding comparison (2026-05-23)

Target: `gemma-4-31B-it-IQ4_XS.gguf` (dense 31B, arch gemma4). Single RTX 3090 24 GB.
Workload: code-gen prompt, 256 tokens, temp 0, 3 runs/arm (run 1 cold → run 3 warm).
llama.cpp arms: ctx 131072, q4_0/q4_0 KV, fa on, parallel 1.

## Final ranking

| rank | engine / config | warm t/s | vs baseline | accept% | status |
|---|---|---:|---:|---:|---|
| **1** | **llama.cpp ngram-mod — eagle3-port binary** | **129** | **3.7×** | 100 | **DEPLOYED** |
| 2 | llama.cpp ngram-mod — mtp-port binary | 58 | 1.7× | 42 | superseded |
| 3 | llama.cpp ngram-cache | 36 | 1.0× | 12 | does not draft |
| 3 | llama.cpp ngram-simple | 36 | 1.0× | 1.8 | does not draft |
| — | llama.cpp baseline (no spec) | 35 | 1.0× | — | reference |
| — | llama.cpp draft-model (E4B) | 35 | 1.0× | 0 | non-viable (0 drafts) |
| — | vLLM (ngram or baseline) | — | — | — | cannot run on 24 GB |

## ★ Key finding — the binary dominates

Same model, same flags, same KV — only the llama-server **binary** differs:
`ngram-mod` runs at **129 t/s on the eagle3-port build vs 58 t/s on the mtp-port
build**. A **2.2× difference from the binary alone.** The eagle3-port build
carries local ngram-mod enhancements (persist / reset-streak WIP) the mtp-port
build (upstream PR #22673, MTP-focused) lacks. Production was on mtp-port; it is
now switched to **eagle3-port** — verified serving Gemma 4 reasoning + tool
calls cleanly end-to-end through the proxy.

## ngram variants

- **ngram-mod** is the only effective ngram type — 100% acceptance warm.
- ngram-simple (1.8%) and ngram-cache (12%) barely draft on default params — no speedup.
- ngram-mod n-max 3 vs n-max 8 — identical results; n-max does not matter here.

## Draft-model (non-ngram) — non-viable

Gemma 4 E4B Q4_K_M draft (vocab 262144, **matches** the 31B, unlike the dead
EAGLE-3 heads):
- **mtp-port binary**: rejected `--ctx-size-draft` (flag removed in that build).
- **eagle3-port binary**: loaded fine but **drafted 0 tokens** → 35 t/s (baseline).
  That binary's `--model-draft` path is wired for EAGLE-3 heads, not classic
  draft models.

llama.cpp classic draft-model speculative decoding is not usable for Gemma 4 31B
on either available binary. ngram-mod is the working path.

## vLLM — cannot run on this hardware

Model: `QuantTrio/gemma-4-31B-it-AWQ` (4-bit AWQ, 20 GB on disk — vLLM cannot
serve the IQ4_XS GGUF for spec decoding). vLLM 0.21.0 installed in
`~/vllm-venv`.

Three start attempts, all failed:
1. Default — `max_tokens_per_mm_item (2496) > max_num_batched_tokens (2048)`
   (Gemma 4 is multimodal). Fixed with `--max-num-batched-tokens 4096`.
2. `--gpu-memory-utilization 0.95` — `Free memory 20.95 GiB < desired 22.38 GiB`.
3. `--gpu-memory-utilization 0.88 --enforce-eager` — `Free memory 20.59 GiB <
   desired 20.73 GiB`.

**Root cause — hard hardware limit:** the 20 GB AWQ model + the desktop's
~3 GB resident VRAM use exceed what a 24 GB card can give vLLM, and even if it
squeezed in there would be ~0 GB left for vLLM's paged KV cache (which it
requires to function). vLLM is **not viable** for Gemma-4-31B on this 24 GB
RTX 3090. (llama.cpp's IQ4_XS GGUF — 16.4 GB — fits with room for KV; this is
why the production stack is llama.cpp.)

## Deployed config

Production: Gemma 4 31B IQ4_XS, **eagle3-port binary**, ngram-mod (n-max 3,
n-min 1, p-min 0.80, n-match 24), q4_0/q4_0 KV, ctx 131072, embedded chat
template, `--reasoning-format deepseek`. ~129 t/s warm code-gen, 3.7× baseline.
