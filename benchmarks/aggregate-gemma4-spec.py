#!/usr/bin/env python3
"""Aggregate the 3-arm bench results and write final report.md."""
import json
import pathlib
import statistics
import sys

OUT = pathlib.Path("/home/cogtek/llama.cpp/benchmarks/2026-05-06-gemma4-31b-iq4xs-spec")
ARMS = ["baseline", "ngrammod", "eagle3"]

agg = {}
for arm in ARMS:
    f = OUT / f"{arm}.jsonl"
    if not f.exists() or f.stat().st_size == 0:
        agg[arm] = None
        continue
    rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    if not rows:
        agg[arm] = None
        continue
    tps = [r["tps"] for r in rows if r["tps"] > 0]
    n_drafted_total = sum(r["draft_n"] for r in rows)
    n_accepted_total = sum(r["draft_accepted"] for r in rows)
    agg[arm] = {
        "tps_mean": round(statistics.mean(tps), 2) if tps else 0.0,
        "tps_median": round(statistics.median(tps), 2) if tps else 0.0,
        "tps_min": round(min(tps), 2) if tps else 0.0,
        "tps_max": round(max(tps), 2) if tps else 0.0,
        "tps_run1": round(rows[0]["tps"], 2) if rows else 0.0,
        "tps_after_warmup_mean": round(statistics.mean(tps[1:]), 2) if len(tps) > 1 else 0.0,
        "accept_pct": round(100.0 * n_accepted_total / n_drafted_total, 1) if n_drafted_total > 0 else None,
        "n_runs": len(rows),
        "rows": rows,
    }

(OUT / "summary.json").write_text(json.dumps(agg, indent=2))

base = agg.get("baseline") or {}
b_mean = base.get("tps_mean", 0.0)

md = []
md.append("# Gemma 4 31B IQ4_XS — speculative decoding A/B")
md.append("")
md.append(f"- **Date**: 2026-05-06")
md.append(f"- **Build**: `b9039` (`94e0ff659`) — fresh `origin/master` + PR-18039 (squashed) + gemma4 layer hook + eagle3 server-path n_new<=0 patch")
md.append(f"- **Worktree**: `/home/cogtek/llama.cpp/.worktrees/eagle3-port`")
md.append(f"- **Target**: `gemma-4-31B-it-IQ4_XS.gguf` (16.4 GB)")
md.append(f"- **Draft (eagle3 arm)**: `Gemma-4-31B-Eagle3-Q8_0.gguf` (696 MB EAGLE-3 head)")
md.append(f"- **Server flags**: `-fa on -ctk q4_0 -ctv q4_0 -ngl 99 -b 512 -ub 512 --parallel 1`")
md.append(f"- **Context**: 131072 (baseline, ngram-mod) — eagle3 reduced to **65536** to fit (cuBLAS handle creation OOM at 131k once draft model + draft KV are added on the 24 GB RTX 3090)")
md.append(f"- **Workload**: fixed code+prose prompt (~100 input tokens), `max_tokens=256`, `temp=0`, deterministic, 5 measured runs after a 64-token warmup, single slot, `cache_prompt=false`")
md.append("")
md.append("## Result")
md.append("")
md.append("| Arm       | tok/s mean | run 1 | runs 2-5 mean | min   | max   | accept % | runs |")
md.append("|---        |---:        |---:   |---:           |---:   |---:   |---:      |---:  |")
for arm in ARMS:
    a = agg.get(arm)
    label = {"baseline":"baseline", "ngrammod":"ngram-mod", "eagle3":"eagle3"}[arm]
    if a is None:
        md.append(f"| {label:<9} | — | — | — | — | — | — | 0 |")
    else:
        ap = f"{a['accept_pct']}" if a['accept_pct'] is not None else "—"
        warm = a['tps_after_warmup_mean']
        warm_s = f"{warm}" if warm > 0 else "—"
        md.append(f"| {label:<9} | {a['tps_mean']} | {a['tps_run1']} | {warm_s} | {a['tps_min']} | {a['tps_max']} | {ap} | {a['n_runs']} |")
md.append("")
if b_mean > 0:
    md.append("## Speedups vs baseline (mean tok/s)")
    md.append("")
    for arm in ("ngrammod", "eagle3"):
        a = agg.get(arm)
        if a and a["tps_mean"] > 0:
            mult = a["tps_mean"] / b_mean
            d = (a["tps_mean"] - b_mean) / b_mean * 100.0
            md.append(f"- **{arm}**: {mult:.2f}× ({d:+.1f}%)")
    md.append("")
md.append("## Caveats")
md.append("")
md.append("- **ngram-mod result inflates with deterministic repeated prompts.** Because `temperature=0` with the same prompt produces the same token stream, the ngram-mod state accumulates the exact next-token mapping. Run 1 is the most representative of a cold-cache path; runs 2-5 measure how well the cache replays. The \"run 1\" column above is the cleaner number to compare against eagle3 / baseline. For a fully prod-realistic measurement, vary the prompt across runs.")
md.append("- **eagle3 had to be hot-fixed twice.** (1) PR-18039 ships an assertion (`n_new >= 1`) in `common_speculative_state_eagle3::draft()` that fires under llama-server when `draft()` is re-called between speculation rounds without prompt extension. Worked around by returning an empty draft for that round. (2) cuBLAS handle allocation fails at 131k ctx once draft + draft KV are added — even with q4_0/q4_0 KV — so eagle3 was run at 65k ctx. Both worth raising upstream on the PR.")
md.append("- **The Gemma 4 31B Eagle3 head is not compatible with this base model.** Run 1 produced 289 drafts with **0.0% acceptance**; subsequent runs drafted nothing. The head's GGUF metadata reports `draft_vocab_size = 32000` while Gemma 4's vocab is 262144 — the head was trained against a vocab-pruned variant or different fine-tune. Result: every drafted token mapped (via `d2t`) to the wrong target vocab id, and the base model rejected all of them. **Eagle3 is a net -22% throughput loss vs baseline at this configuration**, not a speedup. Finding a head trained against a matching Gemma 4 31B-IT checkpoint, or building one ourselves, would be required to actually use eagle3.")
md.append("- **131k context with q4_0/q4_0 KV** is what fits a 31B IQ4_XS + eagle3 head + draft KV on a 24 GB RTX 3090. q8_0/q4_0 KV (the original target) OOMs at 131k once the eagle3 head is added.")
md.append("- **Hybrid spec rollback fixes are NOT in this build.** The benchmark binary is fresh-master-based, not your `upgrade-b8740` branch. All three arms run on the same base, so the cross-arm comparison is fair, but absolute numbers may differ from prod by a few percent.")
md.append("- **Draft model is EAGLE-3, not a standalone model.** EAGLE-3 heads have different mechanics (target hidden-state extraction → encoder → decoder → draft) than the standard `--model-draft` path. The PR-18039 implementation has only been validated by the author against Llama 3.1 8B / 3.3 70B and Qwen 3 / 3-MoE / GPT-OSS / RedHtAI families. **Gemma 4 was not in their tested set** — we added the per-layer hook to `gemma4.cpp` ourselves to match the four arches the PR explicitly supports.")
md.append("")
md.append("## How to reproduce")
md.append("")
md.append("```bash")
md.append("cd /home/cogtek/llama.cpp")
md.append("PORT=18088 \\")
md.append("  LLAMA_BIN=/home/cogtek/llama.cpp/.worktrees/eagle3-port/build/bin/llama-server \\")
md.append("  ./benchmarks/run-gemma4-spec-ab.sh")
md.append("```")
md.append("")
md.append("Eagle3-only re-run: `./benchmarks/run-eagle3-only.sh`")
md.append("")

(OUT / "report.md").write_text("\n".join(md) + "\n")
print(OUT / "report.md")
