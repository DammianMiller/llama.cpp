# Tree-speculation hybrid overhead — FIXED

**Date:** 2026-04-17
**Fix:** commit `4c91473bf` on `upgrade-b8740-turbo`
**File:** `src/llama-memory-recurrent.cpp`, function `copy_cell`

## Root cause

`llama_memory_recurrent::copy_cell(i_src, i_dst)` — which runs on every
hybrid seq_cp (e.g. tree-speculation branch fork, slot prompt cache ops) —
was using `ggml_backend_tensor_get` + `_set` to copy a single row *within the
same GPU tensor*. That forces a DtoH → HtoD PCIe round-trip per layer × per
R/S tensor pair (80 round-trips for Qwen3.6's 40 layers).

Tensor-name copy tracing confirmed: `cache_s_l*` was receiving 512 HtoD +
515 DtoH per 2-request run in the `--spec-parallel=2` case vs 3 total in
`--spec-parallel=1`. 58 GB of PCIe traffic in a 5.2 s window. GPU
utilization sat at 2.8% because the GPU was idle waiting on PCIe.

## Fix

Build a pair of stack-allocated `ggml_tensor` views that point at the src
and dst row offsets inside the same tensor, then call
`ggml_backend_tensor_copy`. On CUDA this routes to the backend's
`cpy_tensor` callback → `cudaMemcpyDeviceToDevice`. Data never leaves the
GPU.

46 insertions, 6 deletions. No new backend API, no header changes, no
abstraction violation.

## Measured impact

**Workload:** BST code-gen, 256 tokens, temperature=0.0, ctx 16384,
f16 KV, ngram-cache spec, RTX 3090, Qwen3.6-35B-A3B hybrid.

| config | before fix | after fix | speedup |
|---|---:|---:|---:|
| `--spec-parallel 1` (linear) | 2.36 s / 108 t/s | 2.34 s / 109 t/s | unchanged ✓ |
| `--spec-parallel 2` (tree)   | 8.46 s /  30 t/s | **2.80 s /  91 t/s** | **3.0×** |
| `--spec-parallel 4` (tree)   | 8.50 s /  30 t/s | **2.65 s /  97 t/s** | **3.2×** |
| `--spec-parallel 2` ngram-mod k=4 (41% draft accept) | 4.61 s / 55 t/s | **3.15 s / 81 t/s** | **1.5×** |

Tree-speculation is now viable on hybrid models.

The linear path is byte-identical before and after. The fix benefits
any hybrid-model code path that triggers `copy_cell`, not just
tree-speculation:
- `llama_memory_seq_cp` on hybrid memory (prompt-cache copy between slots, etc.)
- Checkpoint restore paths that move cells between slots
- Any future feature that needs to replicate recurrent state

## Reproducer

```bash
./build-q8mma/bin/llama-server \
  --model /path/to/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf \
  --ctx-size 16384 --cache-type-k f16 --cache-type-v f16 \
  --n-gpu-layers 99 --flash-attn on \
  --spec-type ngram-cache --spec-parallel 2 \
  --draft-max 8 --draft-min 2 --temp 0.0 \
  --mlock --prio 2
```

Then `curl` for 256-token completions — before fix was 8.4 s/request,
after fix is 2.8 s/request.

## Upstream-worthy

This fix is a pure llama.cpp bug and should be PR'd to `ggml-org/llama.cpp`.
It benefits anyone using hybrid SSM+attention models with any form of
multi-seq_id workload (speculation, multi-user slots, checkpointing).
