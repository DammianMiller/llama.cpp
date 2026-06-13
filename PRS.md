# llama.cpp Upstream PR Plan

3 focused PRs to maximize acceptance likelihood. All patches are currently on branch `upgrade-b8740` (based on upstream `b8740`).

## Dependency graph

```
PR 1 (ngram-mod reset threshold)       [independent]
PR 2 (hybrid rollback via checkpoints)  ────► PR 3 (server activation replay)
```

PR 1 and PR 2 can be submitted in parallel. PR 3 depends on PR 2.

---

## PR 1 — `speculative: make ngram-mod low-acceptance reset threshold configurable`

**Scope:** Tiny, isolated, low-risk
**Files:** `common/speculative.cpp` (~10 lines)
**Risk:** Minimal (default preserves upstream behavior)

### Problem

`ngram-mod` has a hardcoded reset trigger in `accept()`:

```cpp
if (n_low >= 3) {
    LOG_WRN("%s: low acceptance streak (%d) – resetting ngram_mod\n", __func__, n_low);
    mod.reset();
    n_low = 0;
}
```

For models with naturally variable output (MoE, fine-tuned, uncensored variants), acceptance frequently dips below 50% for short runs without indicating the cache is actually stale. The hardcoded `n_low >= 3` causes **constant cache wiping** — the cache builds to 100+ drafts/call, then gets reset, then rebuilds, then gets reset again. This thrashing prevents the cache from reaching its productive steady state.

### Fix

Make the threshold configurable via env var. Default preserves upstream behavior.

```cpp
static const int reset_streak = []() {
    const char * env = std::getenv("NGRAM_MOD_RESET_STREAK");
    if (env && *env) {
        try { return std::stoi(env); } catch (...) {}
    }
    return 3;
}();
```

Set `NGRAM_MOD_RESET_STREAK=0` (or negative) to disable automatic resetting entirely.

### Benchmark data (Qwen3.5-35B-A3B-UD-IQ4_XS on RTX 3090)

| Setting | Avg acceptance | Peak acceptance | Gen tok/s |
|---------|---------------:|----------------:|----------:|
| Default (reset=3) | 26–69% (oscillating) | 72% | 55–97 |
| `NGRAM_MOD_RESET_STREAK=16` | 72–88% (stable) | 98.9% | 100–131 |

### Pre-submission checklist

- [x] Backward-compatible default
- [ ] Add unit test in `tests/test-speculative.cpp` exercising the threshold
- [ ] Document env var in `common/arg.cpp` help output OR add CLI flag `--spec-ngram-reset-streak`
- [ ] Benchmark table in commit message

### Branch / commit

Current implementation: `common/speculative.cpp` in branch `upgrade-b8740`.

---

## PR 2 — `llama-memory-hybrid: support speculative rollback via CPU state checkpoints`

**Scope:** Medium, self-contained
**Files:** `src/llama-memory-hybrid.{cpp,h}`, `common/speculative.cpp`
**Risk:** Medium — touches core memory handling

### Problem

Hybrid SSM+attention models (Qwen3.5-35B-A3B, Jamba, etc) fail `common_speculative_is_compat` because their recurrent layers can't support partial `seq_rm`. The upstream check tries to remove 1 token after a 2-token decode:

```cpp
if (!llama_memory_seq_rm(mem, 0, 1, -1)) {
    LOG_WRN("%s: the target context does not support partial sequence removal\n", __func__);
    res = false;
    goto done;
}
```

This immediately returns `false` for any hybrid model, disabling speculative decoding entirely on some of the best-performing local models available.

### Fix

Add a **CPU-side checkpoint system** to `llama_memory_hybrid`:

1. **Before** processing a short multi-token batch (likely speculative), save R/S tensor data to CPU RAM via `ggml_backend_tensor_get`
2. **During** `seq_rm` failure, detect hybrid model and test a checkpoint-based rollback path
3. **Restore** via `ggml_backend_tensor_set` when rollback is needed

Key components added:

```cpp
struct recurrent_checkpoint {
    llama_pos     pos = -1;
    int32_t       cell_id = -1;
    std::vector<std::vector<uint8_t>> r_data;  // per-layer R tensor data
    std::vector<std::vector<uint8_t>> s_data;  // per-layer S tensor data
    bool valid = false;
};
std::unordered_map<llama_seq_id, recurrent_checkpoint> cpu_checkpoints;

void save_recurrent_checkpoint(llama_seq_id seq_id);
bool restore_recurrent_checkpoint(llama_seq_id seq_id);
```

Only checkpoint for **short multi-token batches** (`n_tokens > 1 && n_tokens <= max_spec_checkpoint_tokens`) to avoid overhead on prompt prefill.

The `common_speculative_is_compat` check is extended:

```cpp
if (!llama_memory_seq_rm(mem, 0, 1, -1)) {
    const llama_model * model = llama_get_model(ctx_tgt);
    if (model && llama_model_is_hybrid(model)) {
        // Test with checkpoint rollback
        ...
    } else {
        // non-hybrid: preserve original failure behavior
    }
}
```

### Graceful seq_rm fallback (common/speculative.cpp)

The draft model's `seq_rm` calls also need a graceful fallback for hybrid models:

```cpp
if (!llama_memory_seq_rm(mem_dft, 0, 0, reuse_i)) {
    LOG_DBG("%s: leading seq_rm failed on draft, falling back to full clear\n", __func__);
    llama_memory_clear(mem_dft, false);
    prompt_dft.clear();
    reuse_n = 0;
}
```

### Risk mitigation

- All changes gated on `llama_model_is_hybrid(model)` — non-hybrid models are unaffected
- CPU-side memory usage bounded by `max_spec_checkpoint_tokens` (64) + number of recurrent layers
- Checkpoints are per-seq_id, cleared on memory reset

### Branch / commit

`src/llama-memory-hybrid.{cpp,h}` in branch `upgrade-b8740`. ~200 lines.

---

## PR 3 — `server: activation replay after partial speculative rollback for hybrid models`

**Scope:** Small, builds on PR 2
**Files:** `tools/server/server-context.cpp` (~40 lines)
**Depends on:** PR 2
**Risk:** Low — server-only, gated on hybrid model detection

### Problem (the subtle one that PR 2 alone doesn't fix)

After PR 2, `llama_memory_hybrid::seq_rm` can restore a checkpoint when it exists. But during real speculative decoding:

- Pre-speculation: checkpoint saved at position `K`
- Speculation: 8 drafts evaluated → SSM state now at `K+8`
- Partial accept: 5 drafts accepted, need to rollback to position `K+5`
- Rollback call: `seq_rm(seq_id, p0=K+6, -1)`

The old check was `checkpoint.pos == p0 - 1` → `K == K+5` → **false**. Exact match never fired.

Even after PR 2 relaxes the check to `checkpoint.pos <= p0 - 1`, the checkpoint is at `K` but the target is `K+5`. Restoring the checkpoint alone leaves the SSM state at position `K`, not `K+5`.

### Fix: Activation replay

After `seq_rm` restores an earlier checkpoint, the server re-decodes the tokens between `(cache_pos + 1)` and the target position via `llama_decode`. This is the **activation replay** technique from Snakes & Ladders (NeurIPS 2024).

```cpp
llama_memory_seq_rm(llama_get_memory(ctx), slot.id, slot.prompt.n_tokens(), -1);

// Activation replay for hybrid models: seq_rm may have restored the
// recurrent checkpoint to an earlier position than the target rollback,
// leaving the attention KV cache trimmed to match. Re-decode tokens
// from (cache_pos + 1) to target to bring both caches in sync.
{
    const llama_model * mdl = llama_get_model(ctx);
    if (mdl && llama_model_is_hybrid(mdl)) {
        const llama_pos cache_pos = llama_memory_seq_pos_max(llama_get_memory(ctx), slot.id);
        const llama_pos expected_pos = (llama_pos)slot.prompt.n_tokens() - 1;

        if (cache_pos >= 0 && cache_pos < expected_pos) {
            const int n_replay = (int)(expected_pos - cache_pos);
            if (n_replay > 0 && (size_t)(cache_pos + n_replay) < slot.prompt.tokens.size()) {
                llama_batch replay_batch = llama_batch_init(n_replay, 0, 1);
                replay_batch.n_tokens = n_replay;
                for (int ri = 0; ri < n_replay; ri++) {
                    const int prompt_idx = (int)(cache_pos + 1) + ri;
                    replay_batch.token[ri]     = slot.prompt.tokens[prompt_idx];
                    replay_batch.pos[ri]       = cache_pos + 1 + ri;
                    replay_batch.n_seq_id[ri]  = 1;
                    replay_batch.seq_id[ri][0] = slot.id;
                    replay_batch.logits[ri]    = (ri == n_replay - 1) ? 1 : 0;
                }
                const int ret = llama_decode(ctx, replay_batch);
                llama_batch_free(replay_batch);
            }
        }
    }
}
```

### Verified on

- Qwen3.5-35B-A3B-IQ4_XS + ngram-mod spec decoding: 88-98% draft acceptance
- Clean tool-call JSON output, no degenerate repetition
- 100+ tok/s sustained generation
- ~329 replay events across a real session (2-8 tokens each)

### Branch / commit

`tools/server/server-context.cpp` in branch `upgrade-b8740`.

---

## Submission order

1. **PR 1** (ngram-mod reset) — small, independent, tests the upstream review process
2. **PR 2** (hybrid checkpoints) — after PR 1 feedback, submit the larger change
3. **PR 3** (activation replay) — after PR 2 is merged or has reviewer approval

## Pre-submission checklist (all PRs)

- [ ] Rebase onto latest upstream `master` (current base is b8740)
- [ ] DCO sign-off on all commits
- [ ] `make test` passes
- [ ] Benchmark table with pre/post metrics
- [ ] Reference existing issues about hybrid spec decoding failures
- [ ] CI workflow passes (test-backend-ops, test-tokenizer, etc)
- [ ] No changes to unrelated files
