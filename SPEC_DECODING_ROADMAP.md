# Speculative Decoding Roadmap for Hybrid SSM+Attention Models

## Current State (Phase 1 Complete)

**Tag**: `planarquant-spec-deferred-v1` (commit `b195cfb4f`)

### What Works
- Correctness fix for hybrid spec decoding (Snakes & Ladders activation replay)
- Deferred activation replay (merges replay into next decode call)
- Clean tool-call JSON output with ngram-cache or draft model

### Performance
| Config | Speed | vs Baseline |
|--------|-------|-------------|
| No spec (baseline) | 107-118 tok/s | — |
| ngram-cache spec | 60-80 tok/s | -30% |
| Qwen3.5-0.8B draft (65-78% accept) | 73-77 tok/s | -30% |

### Why Spec Is Slower Than Baseline

Three root causes identified:

1. **Small batch inefficiency**: pp8=327 tok/s, pp16=511 tok/s vs tg1=126 tok/s.
   GPU parallelism at typical spec batch sizes gives only 2.5-4x speedup,
   not enough to amortize replay + draft generation overhead.

2. **Sequential draft generation**: Draft model requires D sequential
   `llama_decode()` calls (one per draft token). Each has ~3ms overhead.
   For D=8, that's 24ms per batch — exceeds the savings.

3. **Server loop overhead**: Sample/accept logic, seq_rm, batch
   construction add ~5ms per batch, significant at 8ms/token baseline.

---

## Phase 2: Tree Speculation (EAGLE-2 style) — RECOMMENDED NEXT

**Expected gain**: +80-120% (200-240 tok/s)

**Effort**: 10-15 days

### Design

```
Linear speculation (current):       Tree speculation:
  [draft_1, draft_2, draft_3, ...]    root
                                       ├── draft_1a
                                       │    ├── draft_2a
                                       │    └── draft_2b
                                       └── draft_1b
                                            ├── draft_2c
                                            └── draft_2d
```

### Key Implementation Tasks

#### Task 2.1: Tree Data Structures
```cpp
struct spec_tree_node {
    llama_token token;
    int32_t parent_idx;       // -1 for root
    int32_t depth;            // 0 = root
    int32_t batch_idx;        // position in verification batch
    float   prob;             // probability from draft model
};

struct spec_tree {
    std::vector<spec_tree_node> nodes;
    std::vector<std::vector<int32_t>> children_of;  // tree structure
    
    int32_t expand(int32_t parent, llama_token tok, float prob);
    std::vector<llama_pos> compute_positions() const;
    std::vector<int32_t> longest_match(const std::vector<llama_token>& accepted) const;
};
```

#### Task 2.2: Tree Drafting

Modify `common_speculative_draft` to return a tree:

```cpp
spec_tree common_speculative_draft_tree(
    struct common_speculative * spec,
    const common_params_speculative & params,
    const llama_tokens & prompt_tgt,
    llama_token id_last);
```

Approach:
1. Start with root (id_last)
2. BFS: sample top-K tokens from draft logits at each level
3. Prune branches below probability threshold
4. Stop at max_depth or max_nodes

Typical configuration:
- max_depth = 4-6
- branches_per_node = 2-3
- max_nodes = 32-64

#### Task 2.3: Tree Batch Construction

Pack tree nodes into verification batch:

```cpp
llama_batch build_tree_batch(const spec_tree & tree, llama_seq_id seq_id) {
    llama_batch batch = llama_batch_init(tree.nodes.size(), 0, 1);
    for (auto & node : tree.nodes) {
        batch.token[node.batch_idx] = node.token;
        batch.pos[node.batch_idx]   = node.depth;  // depth = position offset
        batch.n_seq_id[node.batch_idx] = 1;
        batch.seq_id[node.batch_idx][0] = seq_id;
        batch.logits[node.batch_idx] = 1;  // need logits at every node
    }
    return batch;
}
```

**Critical**: Custom attention mask. Each node must attend ONLY to its ancestors,
not to siblings. This requires modifying llama.cpp to support custom causal masks
or per-position attention scopes.

#### Task 2.4: Tree Verification & Path Selection

After main model forward pass:

```cpp
std::vector<int32_t> verify_and_select_path(
    const spec_tree & tree,
    common_sampler * smpl,
    llama_context * ctx) {
    
    std::vector<int32_t> accepted_path;
    int32_t current_node = 0;  // start at root
    
    while (true) {
        // Sample target model's prediction at current node
        llama_token target = common_sampler_sample(smpl, ctx, 
            tree.nodes[current_node].batch_idx);
        
        // Find child matching target
        int32_t matching_child = -1;
        for (int32_t child : tree.children_of[current_node]) {
            if (tree.nodes[child].token == target) {
                matching_child = child;
                break;
            }
        }
        
        if (matching_child < 0) {
            // No match - accepted path ends here, target becomes next token
            accepted_path.push_back(target);
            break;
        }
        
        accepted_path.push_back(tree.nodes[matching_child].token);
        current_node = matching_child;
        
        if (tree.children_of[current_node].empty()) break;
    }
    
    return accepted_path;
}
```

#### Task 2.5: Hybrid State Handling in Tree

For hybrid models, SSM state needs to match the accepted path:
- During forward pass, model computes state at EVERY tree node
- Accepted path determines which state vectors to keep
- Non-accepted nodes' states are discarded

Implementation options:
- **Simple**: seq_rm to prune non-accepted tree nodes, then activation replay to advance to accepted_path end
- **Advanced**: Modify SSM kernel to output per-token state, select directly

The simple approach reuses existing infrastructure and is sufficient for Phase 2.

### File Changes Required

| File | Changes |
|------|---------|
| `common/speculative.h` | Add `spec_tree` struct, tree draft function signatures |
| `common/speculative.cpp` | Implement tree drafting (new state type + draft fn) |
| `common/sampling.cpp` | Add tree verification + path selection |
| `tools/server/server-context.cpp` | Use tree batch construction + verification |
| `src/llama-batch.cpp` | Support custom attention masks for tree positions |
| `include/llama.h` | Expose tree position API |

### Testing Plan

1. **Unit tests**: tree construction, longest path selection
2. **Integration test**: single request with tree spec, verify output matches greedy
3. **Benchmark**: compare throughput vs baseline, linear spec, and deferred replay
4. **Correctness**: verify acceptance rates, no state drift

### Expected Outcomes

With 32-node tree, 65% per-token acceptance:
- Average accepted depth: 5-6 tokens
- Batch size: 32 tokens
- pp32 ≈ 900 tok/s
- Theoretical throughput: 32 × (0.65^5 × 5 + ...) / 32 × 900/32 ≈ 200-240 tok/s

---

## Phase 3: Mamba CUDA Kernel State Snapshots

**Expected gain**: +40-70% when combined with Phase 2 (250-300+ tok/s combined)

**Effort**: 10-15 days, requires CUDA kernel expertise

### Design

Modify Mamba SSM kernel to write intermediate states to global memory,
enabling zero-cost rollback.

### Current Kernel (Conceptual)

```cuda
__global__ void ssm_scan_kernel(
    const float* __restrict__ A, B, C, x, delta,
    float* __restrict__ out,
    float* __restrict__ state_final,  // only final state output
    int n_tokens, int d_state) {
    
    float state[d_state];
    // Load previous state
    load_state(state, state_init);
    
    for (int t = 0; t < n_tokens; t++) {
        // SSM update: state = A * state + B * x[t]
        update_state(state, A, B[t], x[t], delta[t]);
        // Compute output
        out[t] = C[t] @ state;
    }
    
    // Store only final state
    store_state(state_final, state);
}
```

### Modified Kernel

```cuda
__global__ void ssm_scan_kernel_checkpointed(
    const float* __restrict__ A, B, C, x, delta,
    float* __restrict__ out,
    float* __restrict__ state_final,
    float* __restrict__ checkpoint_states,  // [n_tokens, d_state]
    int checkpoint_every,  // 1 = checkpoint every token
    int n_tokens, int d_state) {
    
    float state[d_state];
    load_state(state, state_init);
    
    for (int t = 0; t < n_tokens; t++) {
        update_state(state, A, B[t], x[t], delta[t]);
        
        // NEW: checkpoint if requested
        if (t % checkpoint_every == 0) {
            store_state(&checkpoint_states[t * d_state], state);
        }
        
        out[t] = C[t] @ state;
    }
    
    store_state(state_final, state);
}
```

### Implementation Tasks

#### Task 3.1: Locate and Understand SSM Kernel
```bash
grep -rln "ssm_scan\|selective_scan" ggml/src/ggml-cuda/
```

Find files:
- `ggml/src/ggml-cuda/ssm-scan.cu` (or similar)
- `ggml/src/ggml-cuda/ssm-conv.cu`

Study existing implementation structure, memory access patterns.

#### Task 3.2: Add New ggml Operator

```cpp
// ggml/include/ggml.h
enum ggml_op {
    ...
    GGML_OP_SSM_SCAN_CHECKPOINTED,
    ...
};

// ggml/src/ggml.c
struct ggml_tensor * ggml_ssm_scan_checkpointed(
    struct ggml_context * ctx,
    struct ggml_tensor * s, x, dt, A, B, C,
    struct ggml_tensor * checkpoint_states,
    int checkpoint_every);
```

#### Task 3.3: CUDA Kernel Implementation

Modify existing Mamba kernel or create checkpointed variant:
- Add extra output tensor parameter
- Ensure coalesced writes (state is [d_state] per token)
- Handle boundary cases (empty batch, single token)

Memory cost: `n_tokens × d_state × sizeof(float) × n_layers`
For Qwen3.5 (d_state=128, 40 layers): 128 × 40 × 4 = 20KB per token
For D=8: 160KB per batch. Negligible.

#### Task 3.4: Integrate with llama_memory_hybrid

```cpp
// In llama_memory_hybrid
class llama_memory_hybrid {
    ...
    // GPU tensor for checkpoint states per layer
    std::vector<ggml_tensor *> checkpoint_state_r;
    std::vector<ggml_tensor *> checkpoint_state_s;
    
    // Read specific checkpoint position
    void restore_to_position(llama_seq_id seq_id, llama_pos target_pos) {
        // Copy from checkpoint_state_r[layer][target_pos] to active cell
        // Copy from checkpoint_state_s[layer][target_pos] to active cell
        // GPU-to-GPU copy, no CPU transfer
    }
};
```

#### Task 3.5: Update seq_rm Logic

Replace activation replay with direct state lookup:

```cpp
bool seq_rm(seq_id, p0, p1) {
    // No more "find checkpoint at or before p0-1"
    // Instead: direct lookup at position p0-1
    
    if (checkpoint_states_available(seq_id, p0-1)) {
        restore_to_position(seq_id, p0-1);  // O(1) GPU copy
        trim_attention_kv(seq_id, p0, p1);
        return true;
    }
    // Fallback to CPU checkpoint + activation replay
}
```

### File Changes Required

| File | Changes |
|------|---------|
| `ggml/include/ggml.h` | Add GGML_OP_SSM_SCAN_CHECKPOINTED |
| `ggml/src/ggml.c` | Implement new op and its CPU fallback |
| `ggml/src/ggml-cuda/ssm-scan.cu` | Modified CUDA kernel |
| `src/llama-memory-hybrid.h` | Add checkpoint tensor members |
| `src/llama-memory-hybrid.cpp` | Restore-from-tensor logic |
| `src/llama-graph.cpp` | Wire up checkpointed op for Mamba layers |

### Testing Plan

1. **Correctness**: Verify checkpointed kernel produces identical final state
2. **Numerical**: Compare checkpoint states against reference (non-GPU) impl
3. **Performance**: Measure kernel overhead (should be <5% for checkpoint writes)
4. **Integration**: Full hybrid spec decoding benchmark

### Expected Outcomes

With kernel state snapshots + tree spec:
- Zero activation replay cost
- Larger effective batches (tree spec)
- No GPU↔CPU checkpoint transfers

Projected: 250-300+ tok/s on RTX 3090 for Qwen3.5-35B-A3B-IQ4_XS

---

## Implementation Timeline

### Sprint 1 (Week 1-2): Tree Speculation Foundation
- Tree data structures & drafting algorithm
- Batch construction & custom attention masks
- Basic tree verification

### Sprint 2 (Week 2-3): Tree Integration
- Server integration
- Benchmark against baseline
- Iterate on tree shape (depth, branching)

### Sprint 3 (Week 3-4): Kernel Preparation
- Study existing Mamba CUDA kernel
- Design checkpointed variant
- Prototype in CPU first

### Sprint 4 (Week 4-5): Kernel Implementation
- Implement checkpointed CUDA kernel
- Add ggml operator
- Unit tests

### Sprint 5 (Week 5-6): Integration & Benchmark
- Wire into llama_memory_hybrid
- Full integration testing
- Performance measurements

---

## Reference Implementations

- **EAGLE-2**: https://github.com/SafeAILab/EAGLE
- **Medusa**: https://github.com/FasterDecoding/Medusa
- **Snakes & Ladders** (activation replay): NeurIPS 2024 paper

---

## Alternative Quick Wins (Not Phase 2/3)

If full tree speculation isn't feasible immediately:

1. **Parallel slots** (`LLAMA_PARALLEL=4`): 2-4x aggregate throughput for multi-user
2. **Pre-allocated batch buffers**: 5-10% overhead reduction
3. **Larger draft model on CPU**: Trade VRAM for potentially better drafts
4. **Custom Qwen3.5 draft model**: Train smaller specialist model
