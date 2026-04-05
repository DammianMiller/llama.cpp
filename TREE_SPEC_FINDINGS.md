# Tree Speculation - Implementation Findings

## Discovery

**Tree speculation IS already implemented** in llama.cpp at `examples/speculative/speculative.cpp` (the `llama-speculative` binary, NOT `llama-speculative-simple`).

Key features present:
- Multiple parallel draft sequences (`--parallel N` / `-np N`)
- Branch splitting based on probability threshold (`p_draft_split`)
- Batched draft model decodes (all branches in one llama_decode call)
- Multi-seq_id batches for the target model

## Status for Hybrid Models

**BLOCKER**: The existing tree-spec binary **crashes** on Qwen3.5 and other hybrid SSM+attention models.

Crash location: `common_sampler_sample()` at examples/speculative/speculative.cpp:499
Call stack:
```
#6 common_sampler_sample(common_sampler*, llama_context*, int, bool)
#7 main() at speculative.cpp
```

Root cause (suspected): The tree-spec binary uses `llama_memory_seq_cp` and `llama_memory_seq_keep` which interact with hybrid memory's checkpoint system in ways not covered by our fix. The draft model also needs multi-sequence support which may conflict with hybrid state tracking.

## What Would Be Required to Make Tree Spec Work for Hybrid

### Option A: Fix the Standalone Binary (Medium Effort)

1. Apply our activation-replay fix (`spec-decoding-fixed-v1` tag) to ensure draft and target models both handle hybrid rollback
2. Fix `llama_memory_seq_cp` for hybrid models - copying sequences needs to copy SSM states too
3. Debug the specific crash in common_sampler_sample

Estimated: 3-5 days

### Option B: Port Tree Spec to Common Library (Higher Effort)

The tree spec logic in `examples/speculative/speculative.cpp` isn't accessible from the server because the server uses `common/speculative.cpp` which implements only linear drafting.

Required changes:
1. Extract tree drafting into a new function in `common/speculative.cpp`:
   ```cpp
   bool common_speculative_draft_tree(
       common_speculative * spec,
       const common_params_speculative & params,
       const llama_tokens & prompt_tgt,
       llama_token id_last,
       int n_branches,
       std::vector<llama_tokens> & tree_drafts);  // D trees, each a linear path
   ```

2. Update server-context.cpp to:
   - Call tree draft function
   - Build multi-sequence target batch
   - Verify each branch separately, pick longest match
   - Handle deferred replay across tree branches

3. Verify hybrid memory works with multi-seq batches (requires Option A fix)

Estimated: 10-15 days

### Option C: Custom Attention Masks in Server (Highest Effort)

True tree speculation with custom attention masks:
- Each tree node attends only to its ancestors (not siblings)
- Requires modification to `src/llama-graph.cpp` mask building
- Requires exposing custom mask API in `llama.h`

Estimated: 15-20 days

## Current Measured Performance

Due to hybrid model rollback overhead, our measured server throughput:

| Config | Throughput | Notes |
|--------|-----------|-------|
| No spec (production) | 107 tok/s | Baseline, correct output |
| ngram-cache spec | 60-80 tok/s | Worse than baseline |
| Draft model (Qwen3.5-0.8B) | 73-77 tok/s | 65-78% acceptance but slower |
| Direct llama-speculative-simple | 87 tok/s | Confirms server overhead isn't the issue |

## Why Spec Decoding Is Currently Net-Negative

Three compounding factors on our hardware (RTX 3090, Qwen3.5-35B-A3B-IQ4_XS):

1. **Sequential draft generation**: D draft tokens require D separate `llama_decode()` calls on draft model. For D=8 at 3.6ms each = 29ms overhead per batch.

2. **Small-batch GPU inefficiency**: Main model batch verification of D+1 tokens only gets 327 tok/s (at pp8) vs 3050 tok/s at pp512. Parallelism factor is only ~2.5x at these sizes.

3. **Hybrid rollback cost**: Our activation replay fix adds correctness but not performance. Each partial rollback requires re-decoding accepted tokens.

## Tree Speculation Theoretical Benefit

Tree spec addresses problem #1 via batched draft generation across branches:
- Instead of D sequential decodes → single decode with tree width B
- Sibling nodes at same depth computed in parallel
- Tree of 32 nodes = fewer decode calls (D layers × B siblings batched)

Tree spec addresses problem #2 via larger verification batches:
- Linear D=8: batch=9 tokens at main model (pp9 ≈ 370 tok/s)
- Tree of 32: batch=32 tokens at main model (pp32 ≈ 900 tok/s)
- **2.4x GPU efficiency improvement**

Combined with high acceptance rate (multiple branches = more candidates),
expected throughput: 180-240 tok/s vs current 107 tok/s (+70-120%).

## Recommendation

**For production today**: Keep current config (no spec decoding, 107 tok/s, clean output).

**For future implementation**: Pursue **Option B** (port tree spec to common library). The existing implementation in `examples/speculative/speculative.cpp` provides a working reference - we just need to:
1. Fix hybrid crashes
2. Make it callable from the server
3. Handle our deferred replay infrastructure across tree branches

**Alternative near-term wins**:
- `LLAMA_PARALLEL=4`: 4x aggregate throughput for multi-user scenarios
- Custom distilled draft model trained specifically for Qwen3.5 output distribution: higher acceptance → better spec economics
