# Tree-speculation slowdown — CUDA profiling findings

**Date:** 2026-04-17
**Branch:** `upgrade-b8740-turbo` @ `3e6974248`
**Model:** Qwen3.6-35B-A3B-UD-IQ4_XS, ctx 16384, RTX 3090
**Tool:** Nsight Systems 2023.4.4, nvprof (unavailable on 8.6+)

## Observation

`--spec-parallel 2` on hybrid Qwen3.6 produces a clean, correct output but
runs ~4× slower than `--spec-parallel 1` on the same workload. 31.4 ms/tok
vs 8.52 ms/tok.

Curiously, **prompt eval is nearly unaffected** (4.71 vs 4.45 ms/token).
Only the single-token-generation (tg) path is slow.

## What was ruled OUT

| hypothesis | result |
|---|---|
| Unified KV flag | No — decoupling did not change behaviour |
| build_rs extra-copy ops | No — guard skipping 0-size ops did not help |
| CUDA graph capture disabled | No — instrumented logging shows both fast & slow use CUDA graphs at identical rates (91.5%) |
| spurious `begin()` rebuild | No — was still slow after fix (still a real correctness fix though) |
| Drafter overhead | No — drafter stats show fast execution (~1-2 ms total per request) |

## What the profile actually shows

`slow.qdstrm` imported successfully; `fast.qdstrm` did **not** (nsys 2023.4
bug: "Wrong event order has been detected"). So direct A/B kernel tables
are blocked by tooling.

From the slow trace alone:
- 22,096 kernel launches over ~5.2s wall window = 4,251 kernels/sec
- Total GPU busy time: 147.7 ms
- **GPU utilization: 2.8%** — GPU is idle 97% of the time
- Top kernels (slow):
  - `mul_mat_q`: 1,484 launches, 65.2 ms total, 43.9 µs avg
  - `mul_mat_vec_q`: 1,920 launches, 24.1 ms total, 12.6 µs avg
  - `mul_mat_vec_q_moe`: 351 launches, 5.1 ms total, 14.5 µs avg

The fact that CUDA graphs are active at 91.5% but total wall time is still
4× points to **kernels themselves taking longer per launch** rather than
excessive launches.

## Remaining hypothesis (best explanation given the data)

`n_seq_max > 1` halves `n_ctx_slot` (log confirms: 16384 → 8192 when n_seq_max=2)
but the ubatch runtime sizes are unchanged. The per-kernel compute work
grows subtly — possibly because tensors are reshaped to `[..., mem_size]` at
graph-build time and kernels operate on the larger shape even when only
one logical seq is active.

Can't pinpoint further without either:
- Fixing the nsys qdstrm import bug (upgrade to newer nsys)
- Adding NVTX ranges at graph-build sites in ggml-cuda for precise per-op timing
- Running `ncu --section LaunchStats --target-processes all` on specific kernels

Attempted: `ncu` requires tight control over child processes and the server's
CUDA graph capture interferes with ncu's kernel replay. Would need a
non-graph-capture variant of the test or a dedicated microbenchmark
exercising `--spec-parallel=2`'s graph without the server loop.

## Impact on tree speculation

- `--spec-parallel > 1` is not usable in prod on Qwen3.x hybrid until
  the per-kernel shape-scaling is resolved.
- The tree-spec drafter (ngram-cache, ngram-mod k=4) WORKS correctly and
  reaches useful acceptance rates (41% for ngram-mod k=4). Once the
  kernel-level issue is fixed, tree-spec should yield the expected
  throughput gains.
- On non-hybrid models (pure attention), tree-spec overhead is expected to
  be much smaller; worth testing when a non-hybrid model is available.

## Upstream action items

1. File an issue against llama.cpp describing the hybrid + n_seq_max>1
   cost, linking this benchmark file + the reproducer config.
2. Suggest profiling with newer nsys (>= 2024.x) to localize the exact
   kernel whose shape scales with `cparams.n_seq_max`.
3. Kernel-level fix likely lives in `src/llama-memory-recurrent.cpp`'s
   state tensor sizing or `src/llama-graph.cpp`'s `build_rs` reshape,
   but attribution needs that profile data.

## What ships from this session

- Graph-ops guard `if (n_rs > n_seqs)` (commit `3fde80ef4`) — safe, minor.
- `begin()` de-duplication (commit `3e6974248`) — real correctness fix for
  any ngram-based drafter whose `begin()` is not a no-op.
- Diagnostic instrumentation was added and then removed after proving
  CUDA graphs are not the bottleneck.
