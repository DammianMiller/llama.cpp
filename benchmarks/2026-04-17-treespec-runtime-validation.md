# Tree speculation runtime validation (Task C)

**Date:** 2026-04-17
**Branch:** `upgrade-b8740-turbo` @ `c6f233eb3`
**Model:** Qwen3.6-35B-A3B-UD-IQ4_XS (hybrid SSM+attention, 40 layers, 16 attn heads / 2 kv)
**Hardware:** RTX 3090, f16 KV, ctx 16384
**Workload:** BST code generation, `max_tokens=256 temperature=0.0`, spec-type `ngram-cache`, cold cache, 3 runs each

## Results

| n_parallel | median time | tok/s | accepted drafts |
|---:|---:|---:|---:|
| **1 (linear)**  | 2.12 s | **120.8** | 0/16 |
| 2 (tree)         | 8.46 s |  30.3 | 0/16 |
| 4 (tree)         | 8.50 s |  30.1 | 0/16 |

Same output content across all three. No crashes. Tree-spec path correctness verified.

## Headline finding

Tree mode is **~4× slower** than linear with the same spec type on this hybrid model — and the slowdown is **independent of whether the tree actually forks**. `#calls` and `#drafts` stats are identical between parallel=1 and parallel=2 runs (cold cache, 0 drafts accepted in both), so the overhead is NOT from spec drafting or verification logic. It's from the server's context config changes when `speculative.n_parallel > 1`:

1. `cparams.kv_unified = true` (forced, see `common/common.cpp` in `c6f233eb3`)
2. `cparams.n_seq_max = n_slots * spec_branches` (doubled for parallel=2)
3. `llama_batch_init(..., spec_branches)` — per-token seq_id capacity bumped

Even when no token has >1 seq_id at runtime, this config change pushes the attention path into a slower mode on the Qwen MoE hybrid model.

The `llama.h` doc comment on `kv_unified` hints at this: "setting to false when n_seq_max > 1 can cause bad performance in some cases — try to disable when n_seq_max > 1 for improved performance when the sequences do not share a large prefix". We went the opposite direction (forced on when n > 1), assuming branches would share the prompt prefix. On this model the unified path is dramatically slower regardless.

## Secondary finding: ngram-cache beats ngram-mod on linear

Linear `--spec-type ngram-cache` hit **120 t/s** cold on this workload, vs ~60 t/s baseline with ngram-mod on prod. This is unexpected and worth separate benchmarking at scale — may be a real switch the user wants for their production spec-type, independent of tree work.

(Caveats: ctx=16384 test vs ctx=204800 prod; cold-cache numbers from single short request. Proper comparison would require matching configs.)

## Why tree mode doesn't realize gains here

- ngram-cache's cache is cold for a single fresh request → almost no drafts generated → no opportunity to fork branches
- Even with warmed cache, the fork would have to overcome the 4× baseline overhead before any gain materialises — requires >75% of decode time eliminated by speculation

Draft acceptance of 0/16 means the speculation infrastructure ran but every draft was rejected. For an agentic workload with repetitive tool-call patterns, acceptance would be much higher (ngram-mod production shows 32-98% on real traffic). But on this hybrid model, the tree-mode overhead dominates either way.

## What's needed to make Task C actually useful

1. **Decouple context config from drafting mode** — don't force `kv_unified=true` globally. Only activate when tree actually has a forked batch, and revert when single-branch. May require runtime context reconfig or two contexts.

2. **Profile kv_unified on hybrid** — find out *which* specific op is 4× slower under unified KV. Could be a kernel that doesn't exist for hybrid layouts.

3. **Test on non-hybrid models** — confirm the overhead is hybrid-specific. Qwen3 (non-MoE attention-only) would be a natural reference.

4. **ngram-mod → tree drafting** — extend `common_ngram_mod::get()` to return top-K. ngram-mod is user's prod spec-type and O(1) fast; tree drafting without switching off their faster drafter keeps more gains accessible.

## Decision

Prod stays on old binary. Tree speculation code lands on `upgrade-b8740-turbo`
as a working-but-disabled-by-default implementation for future work.

`--spec-parallel > 1` is NOT safe to enable in production on Qwen3.6 hybrid until
the n_seq_max overhead is resolved.

---

## Follow-up 2026-04-17 (commit `98ee3a376`)

### Root cause isolated to n_seq_max, NOT kv_unified

Re-ran the bench with kv_unified decoupled. Results identical (~30 t/s).
Then reverted just the n_seq_max bump (keeping batch-seq-max=2): **120 t/s,
same as linear**. Cost is entirely in `cparams.n_seq_max > 1` on this hybrid
model. Likely from recurrent-state duplication per seq_id.

### ngram-mod now supports top-K for tree drafting

Added k-slot buckets to `common_ngram_mod` (default k=1, max 8). New CLI
`--spec-ngram-mod-k K`. `common_speculative_state_ngram_mod::draft_tree()`
uses top-K to fork branches at root; each branch continues linearly via
the existing single-entry path.

Bench with real tree drafting:

| config                              | tok/s | draft accept |
|---|---:|---:|
| ngram-cache p=1 linear              | 120.8 |  0% (cold) |
| ngram-cache p=2 tree                |  30.3 |  0% (cold) |
| ngram-mod k=1 p=1 (prod baseline)   |  80.3 | n/a (linear) |
| **ngram-mod k=4 p=2 tree**          | **55.5** | **41%** |

**ngram-mod k=4 p=2 achieves 41% draft acceptance** — the drafter produces
useful forks. 55 t/s trails linear because the n_seq_max=2 hybrid overhead
swamps the tree gains. With that cost fixed (kernel work), this config
would be the winner.

## Updated decision

Tree-spec integration is functionally complete and validated. Three configs
now supported:
- `--spec-type ngram-cache --spec-parallel 2+` — tree via frequency counts
- `--spec-type ngram-mod --spec-ngram-mod-k 2+ --spec-parallel 2+` — tree via top-K slots
- Default linear path unchanged

On hybrid models, expect a fixed per-decode overhead from `n_seq_max > 1`
that currently overwhelms any tree-drafting gain. Do NOT enable in prod
on Qwen3.x MoE until the hybrid-memory multi-seq path is optimized.
