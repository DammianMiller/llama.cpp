#!/bin/bash
set -euo pipefail
# Replicate the original 130+ t/s bench against the LIVE prod server.
# Same prompt × 5, temp=0, cache_prompt=false, n_predict=256.

PORT=${PORT:-8080}
RUNS=${RUNS:-5}
N=${N:-256}
OUT=/home/cogtek/llama.cpp/benchmarks/2026-05-06-gemma4-31b-iq4xs-spec/prod-deterministic.jsonl
LOG=/home/cogtek/llama.cpp/llama-server.log
: > "$OUT"

# EXACT prompt from run-gemma4-spec-ab.sh
PROMPT='Write a complete Python function `median(values: list[int]) -> float` that returns the median of a list of integers. Requirements: (1) raise ValueError on empty list, (2) handle even and odd length lists correctly, (3) include type hints and a clear docstring with two examples, (4) implement without numpy, (5) follow PEP 8 style. Then briefly explain why median is more robust than mean for skewed distributions, in 3 short bullet points.'

LMARK=$(wc -l < "$LOG" 2>/dev/null || echo 0)

echo "  warmup..."
curl -sf -m 180 "http://127.0.0.1:${PORT}/completion" \
  -H "Content-Type: application/json" \
  -d "$(jq -nc --arg p "$PROMPT" '{prompt:$p, n_predict:64, temperature:0, cache_prompt:false}')" \
  > /dev/null

for i in $(seq 1 "$RUNS"); do
  resp=$(curl -sf -m 300 "http://127.0.0.1:${PORT}/completion" \
    -H "Content-Type: application/json" \
    -d "$(jq -nc --arg p "$PROMPT" --argjson n "$N" '{prompt:$p, n_predict:$n, temperature:0, cache_prompt:false}')")
  np=$(jq '.timings.predicted_n // 0' <<<"$resp")
  pms=$(jq '.timings.predicted_ms // 0' <<<"$resp")
  dn=$(jq '.timings.draft_n // 0' <<<"$resp")
  da=$(jq '.timings.draft_n_accepted // 0' <<<"$resp")
  tps=$(awk -v t="$pms" -v n="$np" 'BEGIN{if(t>0) printf "%.2f", n*1000.0/t; else print "0"}')
  acc=$(awk -v a="$da" -v d="$dn" 'BEGIN{if(d>0) printf "%.1f", 100.0*a/d; else print "N/A"}')
  echo "  run $i: pred=${np}t in ${pms}ms = ${tps} t/s | drafted=${dn} acc=${da} (${acc}%)"
  jq -nc --argjson r "$i" --argjson n "$np" --argjson m "$pms" --argjson d "$dn" --argjson a "$da" --arg t "$tps" --arg p "$acc" \
    '{run:$r, n_pred:$n, predicted_ms:$m, draft_n:$d, draft_accepted:$a, tps:($t|tonumber), accept_pct:$p}' \
    >> "$OUT"
done

echo
echo "=== ngram_mod stats from server log ==="
sed -n "$((LMARK+1)),\$p" "$LOG" | grep "statistics ngram_mod" | tail -10
