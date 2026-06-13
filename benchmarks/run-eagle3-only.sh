#!/bin/bash
set -euo pipefail

LLAMA_BIN=/home/cogtek/llama.cpp/.worktrees/eagle3-port/build/bin/llama-server
LLAMA_MODEL=/home/cogtek/Downloads/gemma-4-31B-it-IQ4_XS.gguf
LLAMA_DRAFT=/home/cogtek/Downloads/RedHatAI-gemma-4-31B-it-speculator-eagle3-Q8_0.gguf
PORT=18088
RUNS=5
MAX_TOKENS=256
CTX=65536
OUT_DIR=/home/cogtek/llama.cpp/benchmarks/2026-05-06-gemma4-31b-iq4xs-spec

PROMPT='Write a complete Python function `median(values: list[int]) -> float` that returns the median of a list of integers. Requirements: (1) raise ValueError on empty list, (2) handle even and odd length lists correctly, (3) include type hints and a clear docstring with two examples, (4) implement without numpy, (5) follow PEP 8 style. Then briefly explain why median is more robust than mean for skewed distributions, in 3 short bullet points.'

pkill -f "$LLAMA_BIN" 2>/dev/null || true
sleep 3

"$LLAMA_BIN" \
  --model "$LLAMA_MODEL" \
  --model-draft "$LLAMA_DRAFT" \
  --eagle3 \
  --host 127.0.0.1 \
  --port "$PORT" \
  --threads 16 \
  --ctx-size "$CTX" \
  --cache-type-k q4_0 \
  --cache-type-v q4_0 \
  --cache-type-k-draft q4_0 \
  --cache-type-v-draft q4_0 \
  --gpu-layers 99 \
  -ngld 99 \
  --flash-attn on \
  --batch-size 512 \
  --ubatch-size 512 \
  --parallel 1 \
  --no-context-shift \
  --repeat-penalty 1.0 \
  --spec-draft-n-max 5 \
  --spec-draft-n-min 1 \
  --spec-draft-p-min 0.50 \
  > "$OUT_DIR/server-eagle3.log" 2>&1 &
PID=$!

for i in $(seq 1 180); do
  if curl -sf --max-time 2 "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "  server up after ${i}s"
    break
  fi
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "server died"
    tail -30 "$OUT_DIR/server-eagle3.log"
    exit 1
  fi
  sleep 1
done

OUT="$OUT_DIR/eagle3.jsonl"
: > "$OUT"

echo "  warmup..."
curl -sf -m 180 "http://127.0.0.1:${PORT}/completion" \
  -H "Content-Type: application/json" \
  -d "$(jq -nc --arg p "$PROMPT" --argjson n 64 '{prompt:$p, n_predict:$n, temperature:0, cache_prompt:false}')" \
  > /dev/null

for i in $(seq 1 "$RUNS"); do
  resp=$(curl -sf -m 300 "http://127.0.0.1:${PORT}/completion" \
    -H "Content-Type: application/json" \
    -d "$(jq -nc --arg p "$PROMPT" --argjson n "$MAX_TOKENS" '{prompt:$p, n_predict:$n, temperature:0, cache_prompt:false}')")
  n_pred=$(jq '.timings.predicted_n // 0' <<<"$resp")
  pred_ms=$(jq '.timings.predicted_ms // 0' <<<"$resp")
  prompt_n=$(jq '.timings.prompt_n // 0' <<<"$resp")
  prompt_ms=$(jq '.timings.prompt_ms // 0' <<<"$resp")
  n_drafted=$(jq '.timings.draft_n // 0' <<<"$resp")
  n_accepted=$(jq '.timings.draft_n_accepted // 0' <<<"$resp")
  if awk "BEGIN{exit !($pred_ms > 0)}"; then
    tps=$(awk -v t="$pred_ms" -v n="$n_pred" 'BEGIN{printf "%.2f", n*1000.0/t}')
  else
    tps="0"
  fi
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
    >> "$OUT"
done

kill "$PID" 2>/dev/null || true
wait "$PID" 2>/dev/null || true
echo "  done."
