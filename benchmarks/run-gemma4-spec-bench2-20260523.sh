#!/bin/bash
# Gemma 4 31B — llama.cpp spec-decoding follow-up (2026-05-23).
# Arm 1: eagle3-port binary + ngram-mod (does it reproduce the 2026-05-06 132 t/s?)
# Arm 2: mtp-port binary + E4B draft model (the "non-ngram" draft-model arm).
# Draft arm runs at ctx 65536 — the 31B + E4B draft + draft KV will not fit
# 131072 on a 24 GB card (same constraint the 2026-05-06 eagle3 arm hit).
set -uo pipefail

MTP=/home/cogtek/llama.cpp/.worktrees/mtp-port/build/bin/llama-server
EAGLE3=/home/cogtek/llama.cpp/.worktrees/eagle3-port/build/bin/llama-server
MODEL=/home/cogtek/Downloads/gemma-4-31B-it-IQ4_XS.gguf
DRAFT=/home/cogtek/dflash-build/models/gemma4-e4b-draft/gemma-4-E4B-it-Q4_K_M.gguf
PORT=18090
RUNS=3
MAX_TOKENS=256
OUT=/home/cogtek/llama.cpp/benchmarks/2026-05-23-gemma4-spec
mkdir -p "$OUT"

PROMPT='Write a complete Python function `dedupe_preserve_order(items: list) -> list` that removes duplicates from a list while preserving first-seen order. Requirements: type hints, a docstring with two examples, handle unhashable elements gracefully by falling back to an O(n^2) scan, PEP 8. Then explain in 3 bullet points when the O(n^2) fallback would dominate runtime.'

SERVER_PID=""
start_server() {
  local bin="$1"; shift
  pkill -f "llama-server" 2>/dev/null || true
  sleep 5
  "$bin" --model "$MODEL" --host 127.0.0.1 --port "$PORT" --threads 16 \
    --cache-type-k q4_0 --cache-type-v q4_0 --gpu-layers 99 \
    --flash-attn on --batch-size 512 --ubatch-size 512 --parallel 1 \
    --no-context-shift --repeat-penalty 1.0 "$@" \
    > "$OUT/server-${ARM}.log" 2>&1 &
  SERVER_PID=$!
  for i in $(seq 1 180); do
    curl -sf --max-time 2 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && { echo "  up after ${i}s"; return 0; }
    kill -0 "$SERVER_PID" 2>/dev/null || { echo "  SERVER DIED:"; tail -25 "$OUT/server-${ARM}.log"; return 1; }
    sleep 1
  done
  echo "  server timeout"; tail -25 "$OUT/server-${ARM}.log"; return 1
}
stop_server() {
  [[ -n "$SERVER_PID" ]] && { kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null; }
  pkill -f "llama-server" 2>/dev/null || true; sleep 5
}
run_bench() {
  local arm="$1"
  local out="$OUT/${arm}.jsonl"
  : > "$out"
  curl -sf -m 200 "http://127.0.0.1:${PORT}/completion" -H 'Content-Type: application/json' \
    -d "$(jq -nc --arg p "$PROMPT" '{prompt:$p, n_predict:64, temperature:0, cache_prompt:false}')" >/dev/null 2>&1
  echo "  warmup done"
  for i in $(seq 1 "$RUNS"); do
    local resp; resp=$(curl -sf -m 200 "http://127.0.0.1:${PORT}/completion" -H 'Content-Type: application/json' \
      -d "$(jq -nc --arg p "$PROMPT" --argjson n "$MAX_TOKENS" '{prompt:$p, n_predict:$n, temperature:0, cache_prompt:false}')")
    local np pms dn da tps apct
    np=$(jq '.timings.predicted_n // 0' <<<"$resp"); pms=$(jq '.timings.predicted_ms // 1' <<<"$resp")
    dn=$(jq '.timings.draft_n // 0' <<<"$resp"); da=$(jq '.timings.draft_n_accepted // 0' <<<"$resp")
    tps=$(awk -v n="$np" -v m="$pms" 'BEGIN{printf "%.2f",(m>0)?1000.0*n/m:0}')
    apct=$(awk -v a="$da" -v d="$dn" 'BEGIN{printf "%.1f",(d>0)?100.0*a/d:0}')
    echo "  $arm run $i: ${tps} t/s | drafted=${dn} accepted=${da} (${apct}%)"
    echo "{\"arm\":\"$arm\",\"run\":$i,\"tps\":$tps,\"draft_n\":$dn,\"draft_accepted\":$da,\"accept_pct\":\"$apct\"}" >> "$out"
  done
}

echo "== Gemma 4 31B — spec follow-up bench $(date) =="

ARM=eagle3-draft-e4b
echo "[arm: $ARM]  eagle3-port binary + E4B draft model, ctx 65536"
if start_server "$EAGLE3" --ctx-size 65536 --model-draft "$DRAFT" \
     --gpu-layers-draft 99 \
     --cache-type-k-draft q4_0 --cache-type-v-draft q4_0 \
     --spec-draft-n-max 8 --spec-draft-n-min 1 --spec-draft-p-min 0.50; then
  run_bench "$ARM"
fi
stop_server

echo "== follow-up bench done -> $OUT =="
