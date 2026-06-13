# llama.cpp Custom Patches

Custom patches for hybrid SSM+attention speculative decoding support.
Required for Qwen3.5-35B-A3B and other hybrid models to work correctly
with spec decoding enabled.

## Current State

- **Base:** upstream `b8740` (https://github.com/ggml-org/llama.cpp/releases/tag/b8740)
- **Active branch:** `upgrade-b8740`
- **Backup branch:** `backup-master-before-b8740`

## Patches

### 0001 — hybrid speculative: port rollback + checkpoint fixes

Adds CPU-side recurrent state checkpoint save/restore system to
`llama_memory_hybrid`:

- `save_recurrent_checkpoint(seq_id)` — snapshots R/S tensor state to
  CPU RAM before multi-token speculative batches
- `restore_recurrent_checkpoint(seq_id)` — writes it back on rollback
- `seq_rm` fallback path for recurrent models that can't positionally
  rewind SSM state
- Server-side graceful fallback when partial `seq_rm` fails

Files: `src/llama-memory-hybrid.{cpp,h}`, `common/speculative.cpp`,
`tools/server/server-context.cpp`

### 0002 — hybrid spec decoding activation replay for partial rollback

Extends patch 0001 with correct handling of partial speculative
acceptance:

- Accept any checkpoint at `pos <= p0-1` (not exact match)
- When checkpoint is further back than target, restore it and trim
  attention KV to match checkpoint position
- Server-side **activation replay**: after `seq_rm`, re-decode tokens
  from `(cache_pos + 1)` forward via `llama_decode` to bring both
  caches in sync

Implements "activation replay" technique from Snakes & Ladders
(NeurIPS 2024).

**Why it matters:** Without this, SSM state drifts every time a
speculative batch has partial acceptance (which is almost always).
The drift accumulates until the model produces degenerate output —
observable as looping tool calls in agentic workloads.

Files: `src/llama-memory-hybrid.cpp`, `tools/server/server-context.cpp`

## Upgrade Procedure

When a new upstream release (e.g. `b9000`) is available:

```bash
cd /home/cogtek/llama.cpp
git fetch origin --tags

# Save current state
git branch backup-$(git rev-parse --short HEAD)

# Create upgrade branch from new release
git checkout -b upgrade-b9000 b9000

# Apply patches (3-way merge handles small context shifts)
git am --3way patches/*.patch

# If conflicts: resolve manually, then `git am --continue`
# If a patch is already upstream: `git am --skip`

# Build and restart
cmake --build build --target llama-server -j$(nproc)
systemctl --user restart uap-llama-server.service

# Verify
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"x","stream":false,"messages":[{"role":"user","content":"hi"}],"max_tokens":10}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin).get('system_fingerprint'))"
```

## Regenerating Patches

After making changes on the upgrade branch:

```bash
rm -f patches/*.patch
git format-patch -o patches <base-release>..HEAD
```

## Verification

Run a test request and check for activation replay events:

```bash
curl -s http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"x","stream":false,"messages":[{"role":"user","content":"Write quicksort"}],"max_tokens":300}'

grep -c "activation replay" /home/cogtek/llama.cpp/llama-server.log
# Should show non-zero count when spec decoding is active on a hybrid model
```

## Tested on

- Qwen3.5-27B-IQ4_XS (dense, no SSM — replay never fires, safe no-op)
- Qwen3.5-35B-A3B-UD-IQ4_XS (hybrid — replay fires on every spec rollback)
- Qwen3.5-35B-A3B Q4_K_M (hybrid)
- Qwen3.5-35B-A3B-APEX-I-Compact (hybrid)
