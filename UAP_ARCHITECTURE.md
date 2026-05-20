# UAP llama.cpp Stack — Architecture, Process & Guardrail Documentation

Snapshot date: **2026-05-20**

This document covers the production llama.cpp + UAP anthropic-proxy stack on this host: how the components fit together, what each request goes through, and — the main thesis — **how the UAP proxy's guardrail stack lets an undersized open-weights model (Qwen3.6-35B-A3B, 3B-active MoE) sustainably handle agentic workloads at 200k context that the base model alone could not.**

---

## 1. System Architecture

```
                                                        ┌──────────────────────────────────────────────┐
                                                        │                  RTX 3090 (24 GiB)           │
                                                        │                                              │
   ┌──────────────────┐   Anthropic Messages API        │  ┌────────────────────────────────────┐      │
   │                  │   POST /v1/messages             │  │  uap-llama-server.service (:8080)  │      │
   │  Claude / SDK    │ ──────────────────────────────► │  │                                    │      │
   │  Claude Code     │                                 │  │  llama.cpp mtp-port  b9139         │      │
   │  Shannon / etc.  │ ◄────────────────────────────── │  │  e7b484815                         │      │
   │  (172.26.0.3,    │   Anthropic-shaped response     │  │                                    │      │
   │   127.0.0.1,     │                                 │  │  Model: Qwen3.6-35B-A3B-UD-IQ4_XS  │      │
   │   172.25.0.13)   │              ▲                  │  │         (non-MTP GGUF, 16.5 GiB)   │      │
   │                  │              │                  │  │                                    │      │
   └──────────────────┘              │                  │  │  • ctx 204800                      │      │
                                     │                  │  │  • q8_0 / q8_0 KV (~2125 MiB)      │      │
                                     │                  │  │  • --parallel 1  (single slot)     │      │
                                     │                  │  │  • --flash-attn on                 │      │
                                     │                  │  │  • ngram-mod spec decoding         │      │
                                     │                  │  │    (n_max=8, n_match=24)           │      │
       ┌─────────────────────────────┴─────────────────┐│  │  • --cache-ram 40960               │      │
       │  uap-anthropic-proxy.service (:4000)          ││  │  • --ctx-checkpoints 64 / 2048 tok │      │
       │  Python / FastAPI / httpx                     ││  │  • --slot-save-path …/llama-slots  │      │
       │                                               ││  │                                    │      │
       │  master @ #188 (write-tool-restore)           ││  └────────────────────────────────────┘      │
       │  Anthropic API (canonical) + OpenAI passthru  ││                                              │
       │                                               ││  ┌────────────────────────────────────┐      │
       │  ┌─── per-request pipeline (see §3) ───────┐  ││  │  nomic-embeddings.service (:8081)  │      │
       │  │  rate-limit · session monitor · context │  ││  │  nomic-embed-text-v2-moe Q8_0      │      │
       │  │  pruner · tool-schema sanitizer ·       │  ││  │  ctx 8192   (~2 GiB VRAM)          │      │
       │  │  tool narrowing · tool state-machine ·  │  ││  │  127.0.0.1 only                    │      │
       │  │  thinking strip · B1 recon-convergence  │  ││  └────────────────────────────────────┘      │
       │  │  + write-tool-restore · malformed-      │  │└──────────────────────────────────────────────┘
       │  │  payload defense · contamination loop · │  │
       │  │  guardrail · slot save/restore ·        │  │      ┌───────────────────────────────────┐
       │  │  generation-timeout deadline · response │  │      │  Disk                             │
       │  │  shape conversion                       │  │ ───► │  ~/.cache/uap/llama-slots/*.bin  │
       │  └─────────────────────────────────────────┘  │      │  (proxy-managed slot KV cache;    │
       │                                               │      │   LRU-capped at 12 files,         │
       │  Env config: ~/.config/uap/anthropic-proxy.env│      │   ~0.5–1 GiB per slot)            │
       │  Code:  …/universal-agent-protocol/tools/     │      └───────────────────────────────────┘
       │        agents/scripts/anthropic_proxy.py      │
       └───────────────────────────────────────────────┘
```

**Topology summary**

- **Three systemd user services**, all on `cogtek-MS-7C84`:
  - `uap-anthropic-proxy.service` (`:4000`) — the canonical entry point, listening on `0.0.0.0`. Code lives in `~/dev/miller-tech/universal-agent-protocol/tools/agents/scripts/anthropic_proxy.py`.
  - `uap-llama-server.service` (`:8080`) — llama.cpp inference server, runs the main 35B-A3B model.
  - `nomic-embeddings.service` (`:8081`, 127.0.0.1 only) — separate llama.cpp instance for embeddings. Shares the GPU; ~2 GiB VRAM.
- **One GPU slot** at the main llama-server (`--parallel 1`). All upstream calls serialize here.
- **Routing policy**: `ANTHROPIC_PASSTHROUGH_MODELS=__local_only__` — even for `claude-*` model IDs, every request stays local. Deliberate operator policy.
- **Persistence layer**: the proxy maintains a disk-backed slot cache so cross-session KV state survives both proxy restarts (LRU on disk) and short llama-server downtime.

---

## 2. Active Production Configuration

Verified live 2026-05-20.

| Component | Setting | Notes |
| --- | --- | --- |
| Model | `Qwen3.6-35B-A3B-UD-IQ4_XS.gguf` | non-MTP GGUF (unsloth). 34.66 B params, 3 B active (MoE). |
| Binary | `…/.worktrees/mtp-port/build/bin/llama-server` | b9139 / `e7b484815`. Branch `pr-22673` (MTP support — kept even though MTP off, the binary supports many spec types). |
| Context window | **204800** (200×1024) | bumped from 131072→196608→204800 across 2026-05-18. |
| KV quant | **q8_0 / q8_0** | Switched from q4_0 on 2026-05-18 for higher KV fidelity. KV ~2125 MiB on GPU. |
| Spec decoding | `ngram-mod` (n_max=8, n_match=24) | Chosen by benchmark — beat draft-mtp/ngram-cache/etc. |
| `--cache-ram` | **40 GiB** (raised from 16 GiB) | Avoids LRU thrash of the host-RAM prompt cache. |
| `--ctx-checkpoints` | 64 (every 2048 tok) | Fast partial-rollback during decode. |
| Chat template | `qwen3.5-enhanced.jinja` | qwen3_coder XML tool format; UAP's parser-G handles natively. |
| Proxy generation timeout | **1800 s** | Bumped from 1200 s on 2026-05-20 — most timeouts were queue-wait casualties, not slow generation. |
| Proxy context window | 204800 (mirrors server) | Used for pruning thresholds. |
| Proxy concurrency | semaphore limit=1, queue 900 s | Matches the single llama-server slot. |
| Slot save/restore | enabled, dir `~/.cache/uap/llama-slots`, LRU cap 12 | Eliminates the 60–96 s reprocess penalty on cross-session switches. |
| Driver/GPU | RTX 3090, 24 GiB VRAM | Server fits with ~1.5 GiB headroom alongside embeddings. |

Backups of every config change snapshot are kept in `~/.config/uap/*.pre-<change>-<timestamp>` — `ls -t` shows the rollback chain. Recent ones: `pre-cacheram40g`, `pre-mtpoff-q8-ctx200k`, `pre-nonmtp-model`, `pre-ngrammod`, `pre-timeout1800`.

---

## 3. Request Lifecycle (Sequence)

The single most important diagram: what happens to a `/v1/messages` request, end to end.

```
 Client                 anthropic-proxy (:4000)                                          llama-server (:8080)
  │                          │                                                                 │
  ├── POST /v1/messages ────►│  ① rate-limit  (CLIENT_RATE per remote addr)                    │
  │                          │  ② resolve session id  (fp:hash of first_user[:512])            │
  │                          │  ③ SessionMonitor  — context util, prune counter, no-write      │
  │                          │     streak, tool-call history, contamination tracking           │
  │                          │  ④ REQ log line                                                 │
  │                          │                                                                 │
  │                          │  ⑤ if msgs > 1: prune_conversation (B2/B3)                      │
  │                          │     — drop oldest with monotonic boundary (cache-stable),       │
  │                          │       inject summary breadcrumb of dropped block                │
  │                          │  ⑥ context overflow check; circuit-breaker after 4 prunes       │
  │                          │                                                                 │
  │                          │  ⑦ tools = anthropic_to_openai(body["tools"])                   │
  │                          │  ⑧ tool_schema_sanitize  — strip `pattern`/`format` keys        │
  │                          │     that break llama.cpp's auto-grammar                         │
  │                          │  ⑨ tool_narrowing (top 8 by token overlap w/ task prompt)       │
  │                          │     KEEP the full pre-narrowing list (for ⑭)                    │
  │                          │  ⑩ tool_choice policy: required vs auto vs none                 │
  │                          │     — force on cold start, finalize-turn = auto                 │
  │                          │  ⑪ thinking-block policy:                                       │
  │                          │     PROXY_DISABLE_THINKING_ON_TOOL_TURNS=on                     │
  │                          │  ⑫ tool-state-machine phase  (bootstrap → act → review →        │
  │                          │     finalize) — cycle detection, stagnation, dedup, ban         │
  │                          │  ⑬ tool-call grammar:  off (qwen3_coder XML, parser-G native)   │
  │                          │  ⑭ B1 recon-convergence:                                        │
  │                          │     if consecutive_no_write_turns ≥ 40 → inject firm directive  │
  │                          │     if ≥ 80 → inject hard "STOP, write NOW" directive           │
  │                          │     re-inject any write tool that ⑨ narrowed out                │
  │                          │     log:  restored_write_tools=['Edit','Write',…]               │
  │                          │                                                                 │
  │                          │  ⑮ acquire concurrency semaphore (limit=1, queue 900 s)         │
  │                          │  ⑯ if cross-session switch: POST /slots/0?action=save   ─────►  │
  │                          │     POST /slots/0?action=restore  for incoming session ─────►   │ load KV
  │                          │                                                                 │ (~1-3 s)
  │                          │  ⑰ POST /v1/chat/completions  (streaming if requested) ─────►   │ ╔══════════╗
  │                          │     wall-clock deadline: 1800 s                                 │ ║ slot     ║
  │                          │                                                                 │ ║ restore  ║
  │                          │                                                                 │ ║ ctx ckpt ║
  │                          │                                                                 │ ║ prompt   ║
  │                          │                                                                 │ ║ eval     ║
  │                          │                                                                 │ ║ (~1500   ║
  │                          │                                                                 │ ║  tok/s)  ║
  │                          │                                                                 │ ║ ngram-   ║
  │                          │                                                                 │ ║ mod      ║
  │                          │                                                                 │ ║ decode   ║
  │                          │                                                                 │ ║ (~445    ║
  │                          │                                                                 │ ║  tok/s   ║
  │                          │                                                                 │ ║  warm)   ║
  │                          │                                                                 │ ╚══════════╝
  │                          │  ◄─── streamed chunks / final response ──────────────────────── │
  │                          │                                                                 │
  │                          │  ⑱ response post-processing:                                    │
  │                          │     — extract think-block (truncation-safe — PR #177)           │
  │                          │     — parse qwen3_coder XML tool-calls (parser-G)               │
  │                          │     — malformed-payload defense:                                │
  │                          │         if pseudo-tool-payload detected → retry/clean (PR #169) │
  │                          │     — session-contamination check                               │
  │                          │     — Anthropic shape: msg_*, content blocks, stop_reason map   │
  │                          │     — tool_use.id `toolu_*` prefix normalisation                │
  │                          │  ⑲ record tool calls into SessionMonitor (feeds B1 streak)      │
  │                          │  ⑳ release semaphore                                            │
  │                          │                                                                 │
  │◄──── Anthropic response ─┤                                                                 │
  │                          │                                                                 │
```

**Key invariants enforced by the proxy**

- **One in-flight upstream call** at a time (semaphore = `--parallel 1`).
- **Slot KV state matches the conversation about to be processed** — cross-session switches do explicit save/restore so the slot's KV is the right session's history, not whatever was last there.
- **The model never sees a tool it can't call** — but with PR #188, the *write* tool is restored after narrowing so B1's "write your deliverable" is satisfiable.
- **The conversation never overflows the window** — monotonic prune boundary, summary breadcrumbs preserve findings, circuit breaker stops a death spiral.
- **Anthropic-spec response shape** — `msg_*` IDs, typed content blocks (`text`/`tool_use`/`thinking`), `stop_reason` mapping (`stop→end_turn`, `length→max_tokens`, `tool_calls→tool_use`), `toolu_*` IDs, `usage: {input_tokens, output_tokens}`.

---

## 4. The Guardrail Stack — How UAP Supercharges an Undersized Model

### Thesis

Qwen3.6-35B-A3B is a **3 B-active MoE** at IQ4_XS — small, cheap, fast, and not a top-tier agentic coding model on its own. Out of the box, on this hardware, you'd expect:

- modest tool-use reliability (Qwen3.6's tool-calling is competent but error-prone under pressure),
- frequent stuck-in-exploration failure modes on long agentic tasks,
- catastrophic latency on cross-session multi-tenancy because the single KV slot keeps getting evicted,
- broken responses whenever the model emits malformed pseudo-tool payloads, unclosed `<think>` blocks, or repeats itself.

The UAP proxy adds ~20 guardrails layered on top of the inference server. Each fixes one specific failure mode. The cumulative effect is that this 3B-active model **sustainably handles real multi-tenant agentic coding workloads at 200k context**, which it cannot do alone.

### The guardrails, by impact

```
┌──── Tier 1: throughput-changing ───────────────────────────────────────────────────────────────────┐
│                                                                                                    │
│  • ngram-mod self-speculative decoding              ~445 t/s warm vs ~127 t/s baseline   (~3.5×)   │
│    (proxy: not a proxy feature, but the proxy's bench harness chose it — see §5)                   │
│                                                                                                    │
│  • Slot save/restore  (PR #179/#180)                eliminates 60–96 s prompt-reprocess events     │
│    Cross-session KV state survives switches.        ~17% of requests were affected pre-fix.        │
│                                                                                                    │
│  • Tool narrowing                                   keeps prompt small under heavy toolsets        │
│    (caps 8 tools by relevance to task)              (sessions arrive with 14–26 tools)             │
│                                                                                                    │
│  • Concurrency semaphore  (PR #170)                 prevents queue thrash matching --parallel 1    │
│                                                                                                    │
└────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌──── Tier 2: behavioural correctness ───────────────────────────────────────────────────────────────┐
│                                                                                                    │
│  • B1 recon-convergence guardrail  (PR #185/#187)   detects ≥40 consecutive no-write turns and     │
│                                                     injects firm "switch to synthesis" directive.  │
│                                                     Hard tier at 80: "STOP, write NOW".            │
│                                                     PR #187: counts write-tool ABSENCE, not        │
│                                                     read-tool presence — fires on real agents.     │
│                                                                                                    │
│  • B1 write-tool-restore  (PR #188)                 narrowing was stripping all 3 write tools      │
│                                                     (Edit/Write/NotebookEdit) for recon agents     │
│                                                     — making the directive unsatisfiable. #188     │
│                                                     re-injects them when B1 fires.                 │
│                                                     Verified: 12/12 firings restored tools;        │
│                                                     post-fix peak streak 51 vs prior 90/110.       │
│                                                                                                    │
│  • B2 summarising pruner  (PR #186)                 oldest dropped block replaced by summary       │
│                                                     breadcrumb so the agent retains findings.      │
│                                                                                                    │
│  • B3 cache-stable monotonic prune  (PR #186)       prune boundary only moves forward — never      │
│                                                     re-prunes already-stable prefix. Plays nicely  │
│                                                     with slot save/restore.                        │
│                                                                                                    │
│  • Tool state-machine (act/review/finalize)         cycle detection, stagnation, dedup, ban;       │
│                                                     forces finalize when a session is stuck.       │
│                                                                                                    │
│  • Malformed-payload defense  (PR #169 — 4 layers)  vs Qwen3.6 stuck tool-required runaway:        │
│                                                     1) detect pseudo-tool text leakage             │
│                                                     2) retry with cleaner prompt                   │
│                                                     3) extract tool calls from text fallback       │
│                                                     4) return clean guardrail response             │
│                                                     Steady-state ~1 retry per 5–6 min, not a bug. │
│                                                                                                    │
│  • Session contamination loop guardrail             after 3 contamination resets, force finalize.  │
│                                                                                                    │
│  • Context-overflow circuit breaker                 after 4 consecutive prunes still at >100%,     │
│                                                     force finalize to prevent death spiral.        │
│                                                                                                    │
└────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌──── Tier 3: protocol & compatibility ──────────────────────────────────────────────────────────────┐
│                                                                                                    │
│  • Tool-schema sanitiser  (PR #182)                 strips `pattern` / `format` JSON Schema keys   │
│                                                     that break llama.cpp's auto-grammar.          │
│                                                                                                    │
│  • <think>-block truncation strip  (PR #177)        max_tokens cutting off mid-thinking no longer  │
│                                                     leaks raw `<think>` text into Anthropic        │
│                                                     `text` content blocks.                         │
│                                                                                                    │
│  • Qwen3.6-specific tuning:                                                                        │
│    PROXY_DISABLE_THINKING_ON_TOOL_TURNS=on          (Qwen's thinking trips its own tool format)    │
│    PROXY_FORCE_TOOL_CHOICE_ON_COLD_START=on         (Qwen ignores tool_choice=required cold)       │
│    PROXY_TOOL_CALL_GRAMMAR=off                      (qwen3_coder XML — parser-G handles it)        │
│                                                                                                    │
│  • Anthropic-spec response compliance               msg_* IDs, typed blocks, stop_reason mapping,  │
│                                                     toolu_* IDs, usage shape. SDKs that hit        │
│                                                     /v1/models see Shannon's canonical Claude IDs  │
│                                                     so they connect happily.                       │
│                                                                                                    │
│  • Slot save/restore HTTP timeouts  (PR #183)       180 s save / 300 s restore. Original 60/120 s  │
│                                                     was insufficient for the slower 35B-A3B's      │
│                                                     ~1 GiB KV state.                               │
│                                                                                                    │
│  • Generation/read timeouts                         retuned 600 → 1200 → 1800 s as model and       │
│                                                     workload changed. Final bump targeted queue-   │
│                                                     wait casualties rather than slow generation.   │
│                                                                                                    │
└────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Quantified impact — pre/post selected guardrails

| Failure mode | Without the guardrail | With the guardrail | Source |
| --- | --- | --- | --- |
| Cross-session prompt reprocessing | ~17% of requests, 60–96 s each | 0 events (post slot save/restore) | live monitoring 2026-05-15+ |
| Recon agent stuck in exploration | 664-turn run, no deliverable produced | Firm tier nudge at 40 turns; hard "STOP" at 80; runaways resolve | 2026-05-18 client incident, verified post-#188 |
| `pattern`/`format` schema → grammar failure | "failed to parse grammar" errors | sanitised; no grammar failures | PR #182 |
| Unclosed `<think>` block leaking to client | raw `<think>` text in `text` block | block stripped, `stop_reason=max_tokens` | PR #177 |
| Generation throughput (cold prompt) | ~127 t/s (no spec) | **~445 t/s warm with ngram-mod** | bench `~/llama.cpp/benchmarks/spec-decoding-bench-20260518.md` |
| GENERATION TIMEOUT rate (1200 s deadline) | ~0.7/hr avg, ~2.8/hr in bursts | ~0.33/hr (bump to 1800 s) | watcher 2026-05-20 |
| KV slot wedged from stale state | server appears responsive but no completions | wedge cron alerts within 30 min | added 2026-05-19 |

### The "supercharge" claim, summarised

A bare Qwen3.6-35B-A3B at IQ4_XS on a 3090 will reliably do:

- short conversations,
- shallow tool use,
- generation at ~127 t/s on novel content,
- many failure modes that look like "the model is dumb" but are actually format / tool-routing / queue / KV-thrash artefacts.

Add the UAP guardrail stack and the same model on the same hardware now:

- handles **200k-token agentic conversations** with sub-second slot reuse across session switches,
- decodes at **~445 t/s warm** (3.5× the raw baseline) on the dominant coding workload,
- **stays converged** on long recon/synthesis tasks instead of wandering for 600+ turns,
- **degrades gracefully** under heavy concurrent load (queue-bound, not wedge-bound),
- **speaks Anthropic Messages API** so any Claude-SDK client can use it as a drop-in.

That gap — between "raw inference server" and "production agentic platform" — is the proxy's value.

---

## 5. Recent Operational Changes (timeline)

A compressed view of the 2026-05-14 → 2026-05-20 work that produced the current state.

```
date       change                                                  driver
─────────  ──────────────────────────────────────────────────────  ─────────────────────────────
2026-05-14 model swap → Qwen3.6-35B-A3B-UD-IQ4_XS-MTP              try MTP for self-spec
           binary → mtp-port (b9139, PR #22673)
           ctx 262144 → 131072,  KV q8/q8 → q4/q4
2026-05-15 slot save/restore feature (PR #179)                    eliminate cross-session thrash
2026-05-17 generation/read timeouts 600 → 1200                    35B-A3B slower than 27B dense
           slot save/restore timeouts 60/120 → 180/300            larger KV state to serialise
2026-05-18 ctx 131072 → 196608                                    recon agent context overflow
           proxy PRs #184/#185/#186 (CTX log demote, B1, B2/B3)   recon-convergence work
           #187 — B1 inverts to write-tool-absence streak          B1 was inert as shipped
           cache-ram 16 → 40 GiB                                  internal RAM-cache LRU thrash
           MTP/spec-decoding DISABLED, KV q4 → q8, ctx → 204800   try q8 KV + wider window
           model swap → non-MTP GGUF                              clean (no unused MTP head)
2026-05-19 spec decoding benchmark; ngram-mod chosen              ~3.5× over no-spec baseline
           ngram-mod ENABLED                                       (after stale-slot wedge fix)
           ⚠ first attempt wedged: spec-type change needs slot    procedure now documented
                                    cache clear too                in project_active_server.md
           PR #188 — B1 restores write tools narrowing drops      diagnosed hard-tier runaways
2026-05-20 generation timeout 1200 → 1800 s                       most timeouts were queue waits
```

---

## 6. Operational Procedures

### Routine restart (config change that doesn't touch ctx/model/KV/spec)

```
systemctl --user restart uap-anthropic-proxy.service
# or
systemctl --user restart uap-llama-server.service
```

The proxy's startup auto-clears stale slot files. Safe with the normal config.

### ⚠ ctx-size / model / KV-quant / spec-type changes

A llama-server slot `.bin` file is shape-specific. **Restoring a slot file from a mismatched config hangs the decode** — the request reaches `prompt processing done` then never generates a token. Confirmed twice on 2026-05-18 (ctx bump, ngram-mod enable). **Any of these four changes requires the slot-cache clear procedure:**

```
systemctl --user stop  uap-anthropic-proxy.service
rm -f ~/.cache/uap/llama-slots/*
systemctl --user restart uap-llama-server.service
# verify a direct curl to :8080 completes (small prompt should return < 1 s)
curl -s --max-time 30 -X POST http://127.0.0.1:8080/v1/chat/completions \
     -H 'Content-Type: application/json' \
     -d '{"messages":[{"role":"user","content":"Reply with OK"}],"max_tokens":10}'
systemctl --user start uap-anthropic-proxy.service
```

### Rollback chain

Config backups in `~/.config/uap/`:
- `llama-server.env.pre-<change>-<timestamp>`
- `anthropic-proxy.env.pre-<change>-<timestamp>`

`ls -t` shows most-recent first. To roll back: `cp <backup> <live>` then restart per the procedure above.

### Monitoring

| Mechanism | What it watches | Cadence |
| --- | --- | --- |
| `systemd` service unit | process liveness; auto-restart on crash | continuous |
| llama-server `/health`, proxy `/health` | accept new requests | on-demand |
| Cron `281a9271` (session-only) | B1 RECON CONVERGENCE, errors, wedge signature | every 30 min |
| `cron` wedge signature | `REQ > 0 AND COMP == 0 AND slot.is_processing` | every 30 min |

The wedge signature is **REQ > 0 AND COMP == 0**. `REQ == 0` is just an idle period, not a wedge.

---

## 7. Open Items / Watch List

| Item | Status |
| --- | --- |
| `GENERATION TIMEOUT` rate post-1800 s bump | ~halved; watch for further drift |
| `--parallel 2` consideration | feasible only if VRAM math + workload patterns justify it (tight on 24 GiB) |
| `SESSION CONTAMINATION LOOP` recurrence | self-recovering; ~1 per 30–75 min during load; not actionable per-instance |
| Pre-existing test failures in `test_anthropic_proxy_streaming.py` | 5 failures appear local-env only; CI green; investigate if local dev needs clean |
| DFlash/DDTree port | parked — branch on `feature/hybrid-persist-cache`; targets hybrid SSM models; ngram-mod beat it on benchmark |

---

## 8. References

- **Active server config** snapshot: `~/.claude/projects/-home-cogtek-llama-cpp/memory/project_active_server.md`
- **Recon-convergence work** writeup: `~/.claude/projects/-home-cogtek-llama-cpp/memory/project_recon_convergence.md`
- **Spec-decoding benchmark**: `~/llama.cpp/benchmarks/spec-decoding-bench-20260518.md`
- **Non-MTP llama-bench**: `~/llama.cpp/benchmarks/qwen36-35b-a3b-nonmtp-q8kv-20260518-2210.md`
- **UAP proxy code**: `~/dev/miller-tech/universal-agent-protocol/tools/agents/scripts/anthropic_proxy.py`
- **Launch scripts**: `…/scripts/run-llama-server-continuity.sh` and `…/scripts/run-anthropic-proxy-continuity.sh`
- **PR history** (UAP): #169 (malformed-payload), #170 (concurrency), #172 (tool-turn tuning), #173 (docs), #174/#175 (CI), #176/#178/#181 (model id), #177 (think-strip), #179/#180 (slot save/restore), #182 (schema sanitiser), #183 (slot timeouts), #184 (log demote), #185/#187 (B1), #186 (B2/B3 pruner), #188 (B1 write-tool-restore)
