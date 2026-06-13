# ik-llama Qwen3.6-27B two-stage  vs  old Qwen3.6-35B-A3B no-spec

**Date:** 2026-05-30 · **Hardware:** 1× RTX 3090 (24 GB) · **Endpoint:** `:8080`, temp 0, `cache_prompt:false`
**Method:** `bench-cmp.sh` — 3 decode prompts × 3 runs @256 tok + a prefill probe (6.4k & 23k input @64 tok). All metrics from the server's response `timings` (engine-agnostic). Raw rows in `results.jsonl`.

## Arms
| | NEW | OLD |
|---|---|---|
| Engine | ik-llama Docker (`cu13-server`) | self-built llama.cpp (mtp-port) systemd |
| Model | Qwen3.6-**27B** dense, IQ4_KS | Qwen3.6-**35B-A3B** MoE (3B active), IQ4_XS |
| Spec | two-stage ngram-mod + MTP | none |
| ctx / batch | 200000 / `-b 4096 -ub 1024` | 184320 / `-b 2048 -ub 2048` |
| KV | q4_0/q4_0 +khad/vhad | q4_0/q4_0 |

## Results

### Decode throughput (warm = runs 2–3 mean, t/s)
| prompt | NEW (accept) | NEW cold (run 1) | OLD | OLD cold |
|---|---|---|---|---|
| codegen-repetitive | **109.2** @92% | 77.2 | 104.5 | 104.7 |
| codereview-medium | 95.4 @84% | 63.1 | **105.1** | 104.3 |
| explain-novel | **110.6** @95% | 63.2 | 104.9 | 104.8 |
| **overall warm** | **105.1** | — | **104.9** | — |

Warm decode is a **dead tie (105.1 vs 104.9 t/s)**. But:
- **NEW is content-variable** (95–111) and pays a large **cold-start penalty** (63–77 t/s on the first call of a session) while the ngram/MTP caches warm.
- **OLD is dead-flat ~105 t/s** on every prompt, cold or warm — the 35B-A3B MoE only activates ~3B params, so raw decode is already fast and needs no spec.
- NEW's two-stage spec works (84–95% accept warm) but only buys it **parity** with OLD's no-spec MoE decode.

### Prefill throughput (t/s)
| input | NEW | OLD | OLD advantage |
|---|---|---|---|
| ~6.4k tok | 941.8 | **3255.2** | **3.5×** |
| ~23k tok | **CUDA OOM** (container wedged, needed restart) | **3048.1** (2 GB free, kept serving) | — |

**NEW prefills 3.5× slower and OOMs entirely at 23k.** At ctx 200000 the full KV is pre-allocated leaving ~318 MB; a large prompt's transient checkpoint + recurrent-ckpt + spec buffers blow past it → `CUDA error: out of memory` (`ggml-cuda.cu:442`), wedging the server. The compose's own header warns this: "set CTX_SIZE=180224 for heavy agentic use at the ceiling."

## Conclusion

**For this hardware the two configs decode equally fast warm, but the OLD config is decisively better for the real workload.**

Memory records that real Shannon traffic is **87–96% prefill-bound** at 24k–192k input. On that axis:
- OLD is **3.5× faster at prefill** and stable to 192k (memory `2026-05-28-prefill-investigation`).
- NEW **OOMs on a single 23k prompt** at ctx 200k — a hard stability failure on exactly the inputs that dominate the workload.

The new config's headline feature (two-stage spec decode) only matters in the decode-bound regime, which is <15% of real wall time here — and even there it merely matches the MoE's already-fast decode rather than beating it. The 27B *dense* model is inherently heavier to prefill than the 35B-A3B *MoE*.

### Recommendations
1. **If staying on the new config:** drop `CTX_SIZE` to **180224** (or 163840) per the compose's guidance to stop the prefill OOM. Re-benchmark prefill — it will still trail the MoE but won't crash.
2. **For the prefill-bound Shannon workload, the old 35B-A3B config is the stronger prod choice** (3.5× prefill, no OOM, equal warm decode, no cold-start penalty). Consider reverting, or running the new config only if decode-bound / short-prompt traffic becomes dominant.
