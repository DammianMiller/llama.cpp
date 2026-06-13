#!/bin/bash
# Gemma 4 31B — vLLM speculative-decoding bench (2026-05-23).
# Model: QuantTrio/gemma-4-31B-it-AWQ (4-bit AWQ, ~20 GB) — vLLM cannot serve
# the IQ4_XS GGUF well, so the AWQ quant is used (different quant than the
# llama.cpp arms — noted in the report). Requires the production stack STOPPED.
set -uo pipefail

VLLM=/home/cogtek/vllm-venv/bin/vllm
MODEL=/home/cogtek/dflash-build/models/gemma4-31b-awq
PORT=18091
RUNS=4
MAXTOK=256
OUT=/home/cogtek/llama.cpp/benchmarks/2026-05-23-gemma4-spec
mkdir -p "$OUT"

PROMPT='Write a complete Python function `dedupe_preserve_order(items: list) -> list` that removes duplicates from a list while preserving first-seen order. Requirements: type hints, a docstring with two examples, handle unhashable elements gracefully by falling back to an O(n^2) scan, PEP 8. Then explain in 3 bullet points when the O(n^2) fallback would dominate runtime.'

VPID=""
start_vllm() {
  pkill -f "vllm serve" 2>/dev/null || true
  sleep 8
  "$VLLM" serve "$MODEL" --served-model-name gemma4 --host 127.0.0.1 --port "$PORT" \
    --gpu-memory-utilization 0.88 --max-model-len 4096 --max-num-seqs 1 \
    --max-num-batched-tokens 4096 --enforce-eager "$@" \
    > "$OUT/vllm-${ARM}.log" 2>&1 &
  VPID=$!
  for i in $(seq 1 600); do
    curl -sf --max-time 2 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && { echo "  vllm up after ${i}s"; return 0; }
    kill -0 "$VPID" 2>/dev/null || { echo "  VLLM DIED:"; tail -30 "$OUT/vllm-${ARM}.log"; return 1; }
    sleep 1
  done
  echo "  vllm startup timeout"; tail -30 "$OUT/vllm-${ARM}.log"; return 1
}
stop_vllm() {
  [[ -n "$VPID" ]] && { kill "$VPID" 2>/dev/null; wait "$VPID" 2>/dev/null; }
  pkill -f "vllm serve" 2>/dev/null || true; sleep 10
}
run_bench() {
  local arm="$1"
  local out="$OUT/vllm-${arm}.jsonl"
  : > "$out"
  # warmup
  curl -sf -m 200 "http://127.0.0.1:${PORT}/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "$(jq -nc --arg p "$PROMPT" '{model:"gemma4",messages:[{role:"user",content:$p}],max_tokens:64,temperature:0}')" >/dev/null 2>&1
  echo "  warmup done"
  for i in $(seq 1 "$RUNS"); do
    local t0 t1 resp ct elapsed tps
    t0=$(date +%s.%N)
    resp=$(curl -sf -m 200 "http://127.0.0.1:${PORT}/v1/chat/completions" -H 'Content-Type: application/json' \
      -d "$(jq -nc --arg p "$PROMPT" --argjson n "$MAXTOK" '{model:"gemma4",messages:[{role:"user",content:$p}],max_tokens:$n,temperature:0}')")
    t1=$(date +%s.%N)
    ct=$(jq '.usage.completion_tokens // 0' <<<"$resp")
    elapsed=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.3f", b-a}')
    tps=$(awk -v c="$ct" -v e="$elapsed" 'BEGIN{printf "%.2f", (e>0)?c/e:0}')
    echo "  $arm run $i: ${tps} t/s (${ct} tok in ${elapsed}s)"
    echo "{\"arm\":\"$arm\",\"run\":$i,\"tps\":$tps,\"completion_tokens\":$ct,\"elapsed_s\":$elapsed}" >> "$out"
  done
}

echo "== Gemma 4 31B AWQ — vLLM spec-decoding bench $(date) =="
"$VLLM" --version 2>&1 | head -1

ARM=baseline
echo "[arm: vllm-$ARM]  no speculative decoding"
if start_vllm; then run_bench baseline; fi
stop_vllm

ARM=ngram
echo "[arm: vllm-$ARM]  ngram (prompt-lookup) speculative decoding"
if start_vllm --speculative-config '{"method":"ngram","num_speculative_tokens":5,"prompt_lookup_max":4,"prompt_lookup_min":2}'; then
  run_bench ngram
fi
stop_vllm

echo "== vllm bench done -> $OUT =="
