# Tree-speculation 4× slowdown — deep profiling results

**Date:** 2026-04-17 (session 3)
**Branch:** `upgrade-b8740-turbo` @ `3e6974248`
**Tooling:** Nsight Systems 2023.4 (with qdstrm import bug), custom
instrumentation via GGML_CUDA_OP_TIMING env var.

## Executive summary

Root cause is **massive CPU↔GPU memcpy traffic** when `cparams.n_seq_max > 1`
on hybrid SSM+attention models. The slow case moves **58 GB of data** across
PCIe during a 5.2-second profile window — half the wall time is spent in
memcpys. Exact attribution beyond "memcpy flood" still unknown without
targeted memcpy instrumentation.

## What was tested and ruled out

| hypothesis | observation | status |
|---|---|---|
| Different kernel mix | Op counts identical: 16,151 in both cases | ❌ ruled out |
| Different tensor shapes | `op_shape` dump byte-for-byte identical | ❌ ruled out |
| CUDA graph capture disabled | Instrumented: 91.5% graph usage in both | ❌ ruled out |
| Recurrent-state checkpoints | `GGML_HYBRID_NO_CKPT=1` env bypass; no perf change | ❌ ruled out |
| Launch-overhead dominance | GPU only 2.8% busy in slow; dispatch-bound | ✅ confirmed but not root cause |

## What was confirmed

| observation | from |
|---|---|
| Slow has 53,901 memcpys, 58 GB total, 2.7s in memcpy | nsys CUPTI_ACTIVITY_KIND_MEMCPY query |
| Slow splits HtoD/DtoH ~50/50: 28.3 GB each way | sqlite breakdown by copyKind |
| Average copy size is 2.1 MB; 26,264 copies in the 1-10 MB bucket | size-bucket histogram |
| GPU utilization in slow: 2.8% — memcpy saturates PCIe | 147.7 ms compute / 5.2s wall |
| Prompt processing nearly unaffected (4.71 vs 4.45 ms/tok) — only TG slow | server timing logs |
| `ne[]` dimensions identical fast vs slow; `nb[3]` of K-cache halved (16M→8M) | op_shape dump during decode |

The `nb[3]` halving is consistent with n_stream=2 splitting the KV tensor
into two streams of half the storage each — but doesn't explain the 4×
wall-clock cost because only one stream is actually read at runtime.

## Likely sources of the memcpy flood (unverified)

1. **`ggml_backend_sched` weight shuttling** — with n_seq_max=2 the scheduler
   may fragment ops across streams, forcing host-staged transfers between
   compute stages. Strong suspect given the 1-10 MB copy size matches
   typical intermediate activation tensors.
2. **KV-write path fanning to both streams** — when a batch token has a
   single seq_id but n_stream=2, the cache may write to both streams
   (defensively) to keep the layout uniform, doubling write traffic.
3. **Logit/embedding read-back per-token** — nominal per-decode, but if the
   graph path changes under n_seq_max>1 the server might read back more
   tensors than the linear case.

## Tried-but-failed approaches

- **Nsys profile import**: `QdstrmImporter` (nsys 2023.4) fails on fast
  captures with "Wrong event order detected" — blocked per-kernel A/B
- **nvprof**: unsupported on CC 8.6 (3090)
- **In-kernel timing via cudaEvent**: crashed because events can't be
  placed inside CUDA graph capture scope
- **cc-level op count/shape diff**: proved shapes are identical,
  disproving the compute-shape hypothesis but didn't localize the memcpy source

## What was NOT tried (ran out of session budget)

1. **Instrument `ggml_backend_cuda_cpy_tensor_async`** directly: add a
   counter + size histogram + caller-id printf. Would pinpoint the memcpy
   source conclusively. ~30 lines of code, one rebuild.
2. **Upgrade Nsight Systems to 2024.x**: would fix the qdstrm import bug and
   allow direct per-kernel A/B. Requires system-level install.
3. **Capture with Nsight Compute (`ncu`) on a dedicated microbenchmark**:
   avoids the CUDA-graph-capture incompatibility nsys has with the live
   server loop.

## Commits that shipped from this session

- `3fde80ef4` — Graph-ops guard when `n_rs == n_seqs` (defensive, zero risk)
- `3e6974248` — Remove redundant `begin()` call in `common_speculative_draft_tree`
  (real correctness fix; avoids O(prompt) rescan per decode for ngram-mod)

Neither commit closes the 4× gap. Both are genuine improvements that
stand on their own merits.

## Next-session action items

1. Instrument `ggml_backend_cuda_cpy_tensor_async` with call-site attribution
   (caller pointer via backtrace or `__builtin_return_address(0)`). Count
   copies per source location, dump at exit. ~1 hour of work + rebuild.
2. Open an upstream llama.cpp issue against ggml-org/llama.cpp with the
   reproducer config and these findings. Title: "Hybrid model + n_seq_max > 1
   triggers 58 GB of PCIe memcpy traffic per decode burst on Qwen3.6".
3. Test on a non-hybrid model (pure attention) to confirm the cost is
   hybrid-specific. If true, the fix lives in `src/llama-memory-hybrid.cpp`
   or the backend scheduler's handling of hybrid graphs.
