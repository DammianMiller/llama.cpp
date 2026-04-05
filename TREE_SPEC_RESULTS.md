# Tree Speculation Implementation Results

## Summary

**Tree speculation has been FIXED and is now WORKING for hybrid SSM+attention models** (Qwen3.5 family).

However, comprehensive benchmarking reveals that **speculative decoding in ANY form (linear or tree) is net-negative for this hardware+model combination** due to fundamental draft model overhead.

## What Was Fixed

### The Bug
`llama_memory_hybrid::split_equal()` hardcoded `sequential=true`, which asserts on coupled sequences (multiple seq_ids per token, required for tree speculation):

```
split_equal: sequential split is not supported when there are coupled 
sequences in the input batch (you may need to use the -kvu flag)
GGML_ASSERT(logits != nullptr) failed
```

### The Fix (commit `49f1ae889`)

1. Added `is_unified` member to `llama_memory_hybrid` and `llama_memory_hybrid_iswa`
2. Changed `split_equal(n_ubatch, true)` → `split_equal(n_ubatch, !is_unified)`
3. When `kv_unified=true`, non-sequential split is used (supports coupled sequences)
4. Force `kv_unified=true` in speculative example when `n_seq_dft > 1`

### Files Changed
- `src/llama-memory-hybrid.cpp` / `.h`
- `src/llama-memory-hybrid-iswa.cpp` / `.h`
- `examples/speculative/speculative.cpp`

**Tag**: `tree-spec-hybrid-fix-v1`

## Benchmark Results

**Hardware**: RTX 3090 24GB
**Target**: Qwen3.5-35B-A3B-UD-IQ4_XS (hybrid SSM+attention)
**Draft**: Qwen3.5-0.8B-Q8_0 (also hybrid SSM+attention)

### Baseline (no speculation)
- `llama-bench`: **122-124 tok/s** (pure decode)
- `llama-speculative` binary with draft-max=0: **98 tok/s** (includes spec harness overhead)
- Production UAP server: **108-112 tok/s**

### Linear Speculation (np=1)
| draft-max | draft-min | tok/s | accept |
|-----------|-----------|-------|--------|
| 1 | 1 | 78-83 | 42% |
| 2 | 1 | 58 | - |
| 4 | 2 | 29 | 31% |
| 8 | 2 | 20-29 | 35% |
| 16 | 2 | 17-20 | - |

### Tree Speculation (np>1)
| np | draft-max | tok/s | accept |
|----|-----------|-------|--------|
| 2 | 2 | 40 | 49% |
| 2 | 4 | 30 | - |
| 2 | 8 | **32** | 69% |
| 3 | 3 | 30 | 51% |
| 4 | 2 | 31-57 | 48% |
| 4 | 3 | **41** | 63% |
| 4 | 8 | 17-22 | 52% |

### Key Observations

1. **Tree spec acceptance rate DOUBLES** (35% → 69%) vs linear but...
2. **All speculation modes are SLOWER than baseline** (best: 83 tok/s vs 125 baseline)
3. **Overhead scales with draft-max** — more draft tokens = more overhead
4. **Wider trees (higher np)** don't help — draft model serializes through depth
5. **Smaller draft quant (IQ4_XS)** doesn't meaningfully speed things up

## Why Speculation Is Net-Negative Here

### Root Cause: Hybrid Draft Model Overhead

The draft model (Qwen3.5-0.8B) is a **hybrid SSM+attention model**. Each draft token requires:
- A separate `llama_decode()` call (~3.6ms)
- SSM state update (sequential, can't parallelize)
- Full layer forward pass through all 24 layers

For D=8 draft tokens: **~29ms of pure draft overhead per batch**

Compare to main model:
- Baseline tg1: ~8ms/token
- Main model must verify the D+1 batch in ~9-12ms

**Total per spec iteration**: 29ms draft + 12ms main = 41ms
**Tokens generated**: 1 + M_accepted ≈ 3 tokens
**Effective rate**: 73 tok/s

vs **baseline**: 8ms/token = 125 tok/s

### Why Tree Doesn't Help

Tree speculation improves **acceptance rate** (more paths = more chances to match) but **doesn't reduce draft model overhead**. Draft model still generates D tokens sequentially per branch, just with multiple parallel branches.

The math:
- Linear D=8: 8 sequential draft calls × 3.6ms = 29ms
- Tree np=2 D=8: Same 8 depth levels, but 2 tokens per level = still 8 × 3.6 × n_branches_avg

## What Would Actually Help

### 1. Non-Hybrid Draft Model (Most Impactful)

If we had a pure-attention model with compatible Qwen3.5 vocabulary:
- Decode time: ~1-2ms/token (vs 3.6ms for hybrid)
- D=8 drafts: ~8-16ms overhead
- Total per iteration: 20-28ms for 3 tokens = 107-150 tok/s

**Options**:
- Train a distilled pure-attention model from Qwen3.5-35B
- Use Qwen2.5-0.5B (different tokenizer, incompatible)

### 2. Medusa-Style Heads (Second Best)

Add multiple prediction heads to the main model itself:
- No separate draft model
- Heads share hidden states with main model
- Multiple next-token candidates in single forward pass
- Requires training/finetuning (weeks of GPU time)

### 3. Kernel-Level State Snapshots (Phase 3)

Modify `ggml/src/ggml-cuda/gated_delta_net.cu` to save intermediate SSM states:
- Eliminates activation replay overhead
- Enables zero-cost rollback for partial acceptance
- Complex CUDA kernel modification (~10-15 days)
- Benefit limited because main bottleneck is draft model

### 4. Concurrent Multi-User Serving (Aggregate)

Set `LLAMA_PARALLEL=4` in production:
- 4 users get ~80 tok/s each
- Aggregate throughput: 320 tok/s
- Single-user latency: same or worse

## Production Configuration (Current)

```
LLAMA_BIN=/home/cogtek/llama.cpp/.worktrees/turboquant-cuda-v2/build-pq/bin/llama-server
LLAMA_MODEL=/home/cogtek/Downloads/Qwen3.5-35B-A3B-UD-IQ4_XS.gguf
LLAMA_CACHE_TYPE_K=q8_0
LLAMA_CACHE_TYPE_V=q8_0
LLAMA_ENABLE_SPEC_DECODING=false  # <-- Correct decision
```

**Throughput: 108-112 tok/s, clean output, stable.**

## Commits & Tags on `feature/planarquant-kv`

| Tag | Commit | Description |
|-----|--------|-------------|
| `planarquant-v1.0-working` | 0179871c4 | PlanarQuant KV cache types |
| `planarquant-spec-fixed-v1` | 18633a70c | Hybrid spec activation replay |
| `planarquant-spec-deferred-v1` | b195cfb4f | Deferred replay optimization |
| **`tree-spec-hybrid-fix-v1`** | **49f1ae889** | **Tree speculation enabled for hybrid** |

## Conclusion

Tree speculation for hybrid models is now **correct and stable**. Users with matching draft models can benefit from it.

For the specific case of Qwen3.5-35B-A3B, speculative decoding is **not a path to throughput improvement** on current hardware because:
1. The only compatible draft model (Qwen3.5-0.8B) is also hybrid
2. Hybrid architecture makes each draft token expensive
3. Main model baseline is already GPU-bandwidth-limited

Real performance gains require either a purpose-built draft model (training task) or architectural changes to the main model (Medusa/EAGLE heads).
