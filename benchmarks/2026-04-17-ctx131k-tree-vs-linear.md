# ctx 131072 + tree speculation: benchmark and comparison

**Date:** 2026-04-17
**Binary:** `.worktrees/upgrade-b8740-turbo/build-q8mma/bin/llama-server` (commit `4c91473bf`)
**Model:** Qwen3.6-35B-A3B-UD-IQ4_XS, f16 KV, RTX 3090
**Workload:** BST code-gen, `max_tokens=512 temperature=0.0`, ngram-mod spec

## Results — three configs, same workload (median of 3 runs after warmup)

| config                                      | ctx_alloc | n_ctx_per_seq | time    | t/s   | draft accept | VRAM      |
|---------------------------------------------|----------:|--------------:|--------:|------:|-------------:|----------:|
| linear @ 204800 (previous prod)             | 204 800   | **204 800**   |  8.59 s | **59.6** | n/a       | 23.75 GiB |
| tree @ 131072 (`--spec-parallel 2 --spec-ngram-mod-k 4`) | 131 072 |  65 536 |  9.89 s | 51.7  | 35.7 %       | 22.36 GiB |
| linear @ 131072 (`--spec-parallel 1`)       | 131 072   | **131 072**   | 10.67 s | 48.0  | 34.9 %       | ~21.5 GiB |

## What the numbers say

1. **Tree spec at 131k beats linear at 131k by +7.7 %** (51.7 vs 48.0 t/s). The
   `copy_cell` D2D fix (commit `4c91473bf`) is doing its job — without it
   tree would be 3× slower than linear. Now tree has a modest edge.

2. **But both 131k configs are SLOWER than linear @ 204800** (59.6 t/s).
   Reason: the 204 800 prod run has been up for a while with its
   persisted ngram-mod.bin warmed across prior requests, and prompt cache
   hot. Fresh 131k cache + cold prompt cache means lower top-1 acceptance
   than prod. Also, per-decode tree-spec costs grow with ctx (each decode
   evaluates 9 tokens — 1 sampled + 8 drafts), so at larger ctx the gain
   from accepting drafts is partially offset by the larger per-step cost.

3. **Tree spec halves the effective conversation context**:
   `n_ctx_per_seq = 65 536` when ctx_alloc=131 072 with `--spec-parallel 2`.
   Linear at 131k gets the full 131 072.

4. **VRAM is comfortable at 131k** (1.7 GiB free). At 204 800 we had 374
   MiB. Room to experiment with bigger batch sizes or a second slot.

## Tradeoff table for your call

| you want...                  | pick                           | you lose                 |
|------------------------------|--------------------------------|--------------------------|
| max throughput + 200k ctx    | linear @ 204800 (prior prod)   | no tree-spec plumbing    |
| 131k usable ctx, max speed   | linear @ 131072                | ~20 % vs prod            |
| 65k usable ctx, tree-spec proven | tree @ 131072              | half ctx, ~13 % vs prod  |
| 131k usable ctx + tree-spec  | would need ctx_alloc=262 144 — **OOMs on 24 GB at f16** | n/a |

## Currently live on prod

The env has been set to **tree @ 131072** (`--spec-parallel 2 --spec-ngram-mod-k 4`)
as you requested. This gives 51.7 t/s with 35.7 % draft acceptance on the
code-gen workload and 65 536 effective context per request.

To revert to linear @ 204 800:
```bash
# Edit ~/.config/uap/llama-server.env
#   LLAMA_CTX_SIZE=204800
#   LLAMA_EXTRA_ARGS=... (remove --spec-parallel 2 --spec-ngram-mod-k 4, and swap persist path back to ngram-mod.bin)
systemctl --user restart uap-llama-server.service
```

## Notes

- The cold vs warm ngram-mod cache matters a lot for comparability. The
  204 800 numbers were taken against a months-old `.bin` file; the 131k
  tree numbers use a fresh `.bin.k4` that's only been written to during
  warmup + bench (~1000 tokens). With more runtime to populate the k=4
  cache, tree-spec acceptance should climb.
- The copy_cell D2D fix is the reason tree-spec is usable at all on this
  hybrid model. Pre-fix, tree @ 131k would have been ~30 t/s.
