#!/bin/bash
set -euo pipefail

LLAMA_BIN="${LLAMA_BIN:-/home/cogtek/llama.cpp/.worktrees/eagle3-port/build/bin/llama-server}"
LLAMA_MODEL="${LLAMA_MODEL:-/home/cogtek/Downloads/gemma-4-31B-it-IQ4_XS.gguf}"
LLAMA_DRAFT="${LLAMA_DRAFT:-/home/cogtek/Downloads/Gemma-4-31B-Eagle3-Q8_0.gguf}"
PORT="${PORT:-8088}"
RUNS="${RUNS:-5}"
MAX_TOKENS="${MAX_TOKENS:-256}"
CTX="${CTX:-131072}"
STAMP="$(date +%Y-%m-%d)"
OUT_DIR="${OUT_DIR:-/home/cogtek/llama.cpp/benchmarks/${STAMP}-gemma4-31b-iq4xs-spec}"
mkdir -p "$OUT_DIR"

# Mixed code+prose prompt. Should yield ~256 tokens generation.
PROMPT='Write a complete Python function `median(values: list[int]) -> float` that returns the median of a list of integers. Requirements: (1) raise ValueError on empty list, (2) handle even and odd length lists correctly, (3) include type hints and a clear docstring with two examples, (4) implement without numpy, (5) follow PEP 8 style. Then briefly explain why median is more robust than mean for skewed distributions, in 3 short bullet points.'

SERVER_PID=""

start_server() {
  pkill -f "${LLAMA_BIN}" 2>/dev/null || true
  sleep 3
  "$LLAMA_BIN" \
    --model "$LLAMA_MODEL" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --threads 16 \
    --ctx-size "$CTX" \
    --cache-type-k q4_0 \
    --cache-type-v q4_0 \
    --gpu-layers 99 \
    --flash-attn on \
    --batch-size 512 \
    --ubatch-size 512 \
    --parallel 1 \
    --no-context-shift \
    --repeat-penalty 1.0 \
    "$@" \
    > "$OUT_DIR/server-${ARM:-x}.log" 2>&1 &
  SERVER_PID=$!
  for i in $(seq 1 120); do
    if curl -sf --max-time 2 "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
      echo "  server up after ${i}s"
      return 0
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "  server died during startup; tail log:"
      tail -30 "$OUT_DIR/server-${ARM:-x}.log"
      return 1
    fi
    sleep 1
  done
  echo "  server failed to start after 120s"
  tail -30 "$OUT_DIR/server-${ARM:-x}.log"
  return 1
}

stop_server() {
  if [[ -n "$SERVER_PID" ]]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
  fi
  pkill -f "${LLAMA_BIN}" 2>/dev/null || true
  sleep 4
}

run_bench() {
  local arm="$1"
  local out="$OUT_DIR/${arm}.jsonl"
  : > "$out"
  echo "  warmup..."
  curl -sf -m 180 "http://127.0.0.1:${PORT}/completion" \
    -H "Content-Type: application/json" \
    -d "$(jq -nc --arg p "$PROMPT" --argjson n 64 \
        '{prompt:$p, n_predict:$n, temperature:0, cache_prompt:false}')" \
    > /dev/null
  for i in $(seq 1 "$RUNS"); do
    local resp
    resp=$(curl -sf -m 300 "http://127.0.0.1:${PORT}/completion" \
      -H "Content-Type: application/json" \
      -d "$(jq -nc --arg p "$PROMPT" --argjson n "$MAX_TOKENS" \
            '{prompt:$p, n_predict:$n, temperature:0, cache_prompt:false}')")
    if [[ -z "$resp" ]]; then
      echo "  run $i: EMPTY RESPONSE"
      continue
    fi
    local n_pred pred_ms prompt_n prompt_ms n_drafted n_accepted
    n_pred=$(jq '.timings.predicted_n // 0' <<<"$resp")
    pred_ms=$(jq '.timings.predicted_ms // 0' <<<"$resp")
    prompt_n=$(jq '.timings.prompt_n // 0' <<<"$resp")
    prompt_ms=$(jq '.timings.prompt_ms // 0' <<<"$resp")
    n_drafted=$(jq '.timings.draft_n // 0' <<<"$resp")
    n_accepted=$(jq '.timings.draft_n_accepted // 0' <<<"$resp")
    local tps
    if awk "BEGIN{exit !($pred_ms > 0)}"; then
      tps=$(awk -v t="$pred_ms" -v n="$n_pred" 'BEGIN{printf "%.2f", n*1000.0/t}')
    else
      tps="0"
    fi
    local accept_pct
    if awk "BEGIN{exit !($n_drafted > 0)}"; then
      accept_pct=$(awk -v a="$n_accepted" -v d="$n_drafted" 'BEGIN{printf "%.1f", 100.0*a/d}')
    else
      accept_pct="N/A"
    fi
    echo "  run $i: pred=${n_pred}t in ${pred_ms}ms = ${tps} t/s | drafted=${n_drafted} accepted=${n_accepted} (${accept_pct}%)"
    jq -nc --argjson run "$i" --argjson npred "$n_pred" --argjson pms "$pred_ms" \
      --argjson promptn "$prompt_n" --argjson promptms "$prompt_ms" \
      --argjson nd "$n_drafted" --argjson na "$n_accepted" --arg tps "$tps" --arg apct "$accept_pct" \
      '{run:$run, n_pred:$npred, predicted_ms:$pms, prompt_n:$promptn, prompt_ms:$promptms, draft_n:$nd, draft_accepted:$na, tps:($tps|tonumber), accept_pct:$apct}' \
      >> "$out"
  done
}

cleanup() { stop_server; }
trap cleanup EXIT INT TERM

echo "== Gemma 4 31B IQ4_XS spec-decoding A/B =="
echo "  binary: $LLAMA_BIN"
echo "  model:  $LLAMA_MODEL"
echo "  draft:  $LLAMA_DRAFT"
echo "  ctx:    $CTX  runs:$RUNS  max_tokens:$MAX_TOKENS"
echo "  out:    $OUT_DIR"
echo

echo "[1/3] Baseline (no spec)"
ARM=baseline
start_server --spec-type none
run_bench baseline
stop_server

echo "[2/3] ngram-mod"
ARM=ngrammod
start_server \
  --spec-type ngram-mod \
  --spec-draft-n-max 3 \
  --spec-draft-n-min 1 \
  --spec-draft-p-min 0.80 \
  --spec-ngram-mod-n-match 24
run_bench ngrammod
stop_server

echo "[3/3] eagle3"
ARM=eagle3
start_server \
  --model-draft "$LLAMA_DRAFT" \
  --eagle3 \
  -ngld 99 \
  --cache-type-k-draft q4_0 \
  --cache-type-v-draft q4_0 \
  --spec-draft-n-max 5 \
  --spec-draft-n-min 1 \
  --spec-draft-p-min 0.50
run_bench eagle3
stop_server

# Aggregate
python3 - "$OUT_DIR" <<'PY'
import json, pathlib, sys, statistics

out = pathlib.Path(sys.argv[1])
arms = ["baseline", "ngrammod", "eagle3"]
agg = {}
for arm in arms:
    f = out / f"{arm}.jsonl"
    if not f.exists():
        agg[arm] = None
        continue
    rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    if not rows:
        agg[arm] = None
        continue
    tps = [r["tps"] for r in rows if r["tps"] > 0]
    accepts = [r["draft_accepted"] for r in rows if r["draft_n"] > 0]
    drafted = [r["draft_n"] for r in rows if r["draft_n"] > 0]
    agg[arm] = {
        "tps_mean": round(statistics.mean(tps), 2) if tps else 0.0,
        "tps_median": round(statistics.median(tps), 2) if tps else 0.0,
        "tps_min": round(min(tps), 2) if tps else 0.0,
        "tps_max": round(max(tps), 2) if tps else 0.0,
        "accept_pct": round(100.0 * sum(accepts) / sum(drafted), 1) if drafted and sum(drafted) > 0 else None,
        "n_runs": len(rows),
        "rows": rows,
    }
(out / "summary.json").write_text(json.dumps(agg, indent=2))

base = agg.get("baseline", {})
md = ["# Gemma 4 31B IQ4_XS — speculative decoding A/B", "",
      f"- binary: `{(out.parent.parent / '.worktrees' / 'eagle3-port' / 'build' / 'bin' / 'llama-server').relative_to(out.parent.parent)}` (b9039)",
      "- target: `gemma-4-31B-it-IQ4_XS.gguf` (16.4 GB)",
      "- draft:  `Gemma-4-31B-Eagle3.gguf` (1.3 GB FP16 EAGLE3 head, eagle3 arm only)",
      "- ctx: 131072  fa: on  ctk: q8_0  ctv: q4_0  -ngl: 99",
      "- workload: fixed code+prose prompt, max_tokens=256, temp=0, 5 runs after warmup",
      "",
      "| Arm       | tok/s mean | median | min | max | accept % | runs |",
      "|---        |---:        |---:    |---: |---: |---:      |---:  |"]
for arm in arms:
    a = agg.get(arm)
    if a is None:
        md.append(f"| {arm:<9} | — | — | — | — | — | 0 |")
    else:
        ap = f"{a['accept_pct']}" if a['accept_pct'] is not None else "—"
        md.append(f"| {arm:<9} | {a['tps_mean']} | {a['tps_median']} | {a['tps_min']} | {a['tps_max']} | {ap} | {a['n_runs']} |")
md.append("")
if base and base.get("tps_mean", 0) > 0:
    bm = base["tps_mean"]
    for arm in ("ngrammod", "eagle3"):
        a = agg.get(arm)
        if a and a["tps_mean"] > 0:
            d = (a["tps_mean"] - bm) / bm * 100.0
            md.append(f"- {arm} vs baseline: {d:+.1f}% tok/s")
(out / "report.md").write_text("\n".join(md) + "\n")
print(out / "report.md")
PY

echo
echo "=== REPORT ==="
cat "$OUT_DIR/report.md"
