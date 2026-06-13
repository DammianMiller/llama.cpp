#!/bin/bash
set -euo pipefail

# Re-bench ngram-mod against the LIVE prod server (port 8080, full prod flags).
# Verifies: (a) >=130 t/s warm, (b) no prompt reprocessing under cache pressure.

PORT=${PORT:-8080}
RUNS=${RUNS:-6}
MAX_TOKENS=${MAX_TOKENS:-256}
OUT=/home/cogtek/llama.cpp/benchmarks/2026-05-06-gemma4-31b-iq4xs-spec/rebench-prod.jsonl
LOG_FILE=/home/cogtek/llama.cpp/llama-server.log
: > "$OUT"

PROMPTS=(
  'Write a Python function `median(values: list[int]) -> float` returning the median. Raise ValueError on empty. Type hints + docstring with two examples. Then explain in 3 bullets why median is robust.'
  'Implement a binary search tree class in Python with insert, search, delete, and inorder traversal methods. Include docstrings and a small usage example.'
  'Write a Python decorator `@retry(times=3, delay=1)` that retries a function on exception with exponential backoff. Include type hints and a working example.'
  'Refactor this code for clarity:\n```\ndef f(l):\n    r=[]\n    for x in l:\n        if x>0: r.append(x*2)\n    return r\n```\nGive the refactored version with type hints and a brief explanation.'
  'Write a Python context manager that times the enclosed block and prints the duration on exit. Include __enter__, __exit__, and a usage example.'
  'Implement Fisher-Yates shuffle in Python without using random.shuffle. Include type hints, docstring, and a test case.'
)

# Mark log position before bench
LOG_MARK=$(wc -l < "$LOG_FILE" 2>/dev/null || echo 0)
echo "log mark: $LOG_MARK"

run_bench() {
  local label=$1; local prompt=$2; local cache_prompt=$3
  local resp
  resp=$(curl -sf -m 300 "http://127.0.0.1:${PORT}/completion" \
    -H "Content-Type: application/json" \
    -d "$(jq -nc --arg p "$prompt" --argjson n "$MAX_TOKENS" --argjson cp "$cache_prompt" \
        '{prompt:$p, n_predict:$n, temperature:0.3, cache_prompt:$cp}')")
  local n_pred pred_ms prompt_n prompt_ms n_drafted n_accepted n_cached
  n_pred=$(jq '.timings.predicted_n // 0' <<<"$resp")
  pred_ms=$(jq '.timings.predicted_ms // 0' <<<"$resp")
  prompt_n=$(jq '.timings.prompt_n // 0' <<<"$resp")
  prompt_ms=$(jq '.timings.prompt_ms // 0' <<<"$resp")
  n_drafted=$(jq '.timings.draft_n // 0' <<<"$resp")
  n_accepted=$(jq '.timings.draft_n_accepted // 0' <<<"$resp")
  n_cached=$(jq '.tokens_cached // 0' <<<"$resp")
  local tps prompt_tps acc
  tps=$(awk -v t="$pred_ms" -v n="$n_pred" 'BEGIN{if(t>0) printf "%.2f", n*1000.0/t; else print "0"}')
  prompt_tps=$(awk -v t="$prompt_ms" -v n="$prompt_n" 'BEGIN{if(t>0) printf "%.0f", n*1000.0/t; else print "0"}')
  acc=$(awk -v a="$n_accepted" -v d="$n_drafted" 'BEGIN{if(d>0) printf "%.1f", 100.0*a/d; else print "N/A"}')
  echo "  ${label}: pred=${n_pred}t in ${pred_ms}ms=${tps}t/s | prompt=${prompt_n}t in ${prompt_ms}ms=${prompt_tps}t/s | drafted=${n_drafted} acc=${n_accepted}(${acc}%) cached=${n_cached}"
  jq -nc --arg label "$label" --argjson npred "$n_pred" --argjson pms "$pred_ms" \
    --argjson promptn "$prompt_n" --argjson promptms "$prompt_ms" \
    --argjson nd "$n_drafted" --argjson na "$n_accepted" --argjson nc "$n_cached" \
    --arg tps "$tps" --arg ppt "$prompt_tps" --arg acc "$acc" \
    '{label:$label, n_pred:$npred, predicted_ms:$pms, prompt_n:$promptn, prompt_ms:$promptms, draft_n:$nd, draft_accepted:$na, tokens_cached:$nc, tps:($tps|tonumber), prompt_tps:($ppt|tonumber), accept_pct:$acc}' \
    >> "$OUT"
}

echo "[A] Same prompt × 3 (cache_prompt=true): tests prompt KV reuse"
for i in 1 2 3; do
  run_bench "A${i}-same" "${PROMPTS[0]}" true
done

echo "[B] 6 different prompts (cache_prompt=true): tests cache_reuse + cache eviction"
for i in {0..5}; do
  run_bench "B$((i+1))-varied" "${PROMPTS[i]}" true
done

echo "[C] Re-run prompt[0] (cache_prompt=true): is it still cached or evicted?"
run_bench "C-recall0" "${PROMPTS[0]}" true

echo "[D] cache_prompt=false (forces reprocess): comparison baseline"
run_bench "D-no-cache" "${PROMPTS[0]}" false

echo
echo "=== Server log entries since bench start (filtered) ==="
sed -n "$((LOG_MARK+1)),\$p" "$LOG_FILE" 2>/dev/null | \
  grep -E "n_past|update_slots:|prompt processing done|context shift|kv cache|cache evict|n_tokens|reused|create_check|task [0-9]+ \|" 2>/dev/null | \
  tail -60 || echo "(no matching log lines)"

echo
echo "=== /slots state ==="
curl -sf "http://127.0.0.1:${PORT}/slots" 2>/dev/null | jq '.[0] | {state, n_past, prompt: (.prompt|tostring|.[0:120])}' 2>&1 || echo "(slots endpoint unavailable)"
