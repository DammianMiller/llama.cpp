#!/bin/bash
# Gemma 4 31B IQ4_XS — llama.cpp speculative-decoding comparison (2026-05-23).
# Runs on the production mtp-port binary, deployed Gemma config (ctx 131072,
# q4_0/q4_0 KV). One server per arm. Requires the production stack STOPPED.
set -uo pipefail

BIN=/home/cogtek/llama.cpp/.worktrees/mtp-port/build/bin/llama-server
MODEL=/home/cogtek/Downloads/gemma-4-31B-it-IQ4_XS.gguf
PORT=18090
CTX=131072
RUNS=3
MAX_TOKENS=256
OUT=/home/cogtek/llama.cpp/benchmarks/2026-05-23-gemma4-spec
mkdir -p "$OUT"

PROMPT='Write a complete Python function `dedupe_preserve_order(items: list) -> list` that removes duplicates from a list while preserving first-seen order. Requirements: type hints, a docstring with two examples, handle unhashable elements gracefully by falling back to an O(n^2) scan, PEP 8. Then explain in 3 bullet points when the O(n^2) fallback would dominate runtime.'

SERVER_PID=""
start_server() {
  pkill -f "$BIN" 2>/dev/null || true
  sleep 4
  "$BIN" --model "$MODEL" --host 127.0.0.1 --port "$PORT" --threads 16 \
    --ctx-size "$CTX" --cache-type-k q4_0 --cache-type-v q4_0 --gpu-layers 99 \
    --flash-attn on --batch-size 512 --ubatch-size 512 --parallel 1 \
    --no-context-shift --repeat-penalty 1.0 "$@" \
    > "$OUT/server-${ARM}.log" 2>&1 &
  SERVER_PID=$!
  for i in $(seq 1 150); do
    curl -sf --max-time 2 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && { echo "  up after ${i}s"; return 0; }
    kill -0 "$SERVER_PID" 2>/dev/null || { echo "  SERVER DIED:"; tail -20 "$OUT/server-${ARM}.log"; return 1; }
    sleep 1
  done
  echo "  server timeout"; tail -20 "$OUT/server-${ARM}.log"; return 1
}
stop_server() {
  [[ -n "$SERVER_PID" ]] && { kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null; }
  pkill -f "$BIN" 2>/dev/null || true; sleep 4
}
run_bench() {
  local arm="$1" out="$OUT/${arm}.jsonl"; : > "$out"
  curl -sf -m 180 "http://127.0.0.1:${PORT}/completion" -H 'Content-Type: application/json' \
    -d "$(jq -nc --arg p "$PROMPT" '{prompt:$p, n_predict:64, temperature:0, cache_prompt:false}')" >/dev/null 2>&1
  echo "  warmup done"
  for i in $(seq 1 "$RUNS"); do
    local resp; resp=$(curl -sf -m 180 "http://127.0.0.1:${PORT}/completion" -H 'Content-Type: application/json' \
      -d "$(jq -nc --arg p "$PROMPT" --argjson n "$MAX_TOKENS" '{prompt:$p, n_predict:$n, temperature:0, cache_prompt:false}')")
    local np pms dn da tps apct
    np=$(jq '.timings.predicted_n // 0' <<<"$resp")
    pms=$(jq '.timings.predicted_ms // 1' <<<"$resp")
    dn=$(jq '.timings.draft_n // 0' <<<"$resp")
    da=$(jq '.timings.draft_n_accepted // 0' <<<"$resp")
    tps=$(awk -v n="$np" -v m="$pms" 'BEGIN{printf "%.2f", (m>0)?1000.0*n/m:0}')
    apct=$(awk -v a="$da" -v d="$dn" 'BEGIN{printf "%.1f", (d>0)?100.0*a/d:0}')
    echo "  $arm run $i: ${tps} t/s | drafted=${dn} accepted=${da} (${apct}%)"
    jq -nc --arg arm "$arm" --argjson run "$i" --argjson tps "$tps" --argjson dn "$dn" --argjson da "$da" --arg apct "$apct" \
      '{arm:$arm,run:$run,tps:$tps,draft_n:$dn,draft_accepted:$da,accept_pct:$apct}' >> "$out"
  done
}

echo "== Gemma 4 31B IQ4_XS — spec-decoding bench $(date) =="
declare -A ARMS=(
  [baseline]="--spec-type none"
  [ngram-mod-n3]="--spec-type ngram-mod --spec-draft-n-max 3 --spec-draft-n-min 1 --spec-draft-p-min 0.80"
  [ngram-mod-n8]="--spec-type ngram-mod --spec-draft-n-max 8 --spec-draft-n-min 1 --spec-draft-p-min 0.50"
  [ngram-simple]="--spec-type ngram-simple --spec-draft-n-max 3 --spec-draft-n-min 1 --spec-draft-p-min 0.80"
  [ngram-cache]="--spec-type ngram-cache --spec-draft-n-max 3 --spec-draft-n-min 1 --spec-draft-p-min 0.80"
)
for arm in baseline ngram-mod-n3 ngram-mod-n8 ngram-simple ngram-cache; do
  echo "[arm: $arm]  flags: ${ARMS[$arm]}"
  ARM="$arm"
  if start_server ${ARMS[$arm]}; then run_bench "$arm"; else echo "  SKIPPED (start failed)"; fi
  stop_server
done
echo "== bench done -> $OUT =="
