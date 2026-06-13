#!/bin/bash
# MTP vs ngram-mod A/B bench. Uses the same suite as rebench-ngrammod-prod.sh.
# Caller passes label (mtp|ngram-mod). Writes JSONL to ${LABEL}.jsonl in this dir.
set -euo pipefail

LABEL=${1:?usage: $0 <label>}
PORT=${PORT:-8080}
MAX_TOKENS=${MAX_TOKENS:-256}
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${DIR}/${LABEL}.jsonl"
: > "$OUT"

PROMPTS=(
  'Write a Python function `median(values: list[int]) -> float` returning the median. Raise ValueError on empty. Type hints + docstring with two examples. Then explain in 3 bullets why median is robust.'
  'Implement a binary search tree class in Python with insert, search, delete, and inorder traversal methods. Include docstrings and a small usage example.'
  'Write a Python decorator `@retry(times=3, delay=1)` that retries a function on exception with exponential backoff. Include type hints and a working example.'
  'Refactor this code for clarity:\n```\ndef f(l):\n    r=[]\n    for x in l:\n        if x>0: r.append(x*2)\n    return r\n```\nGive the refactored version with type hints and a brief explanation.'
  'Write a Python context manager that times the enclosed block and prints the duration on exit. Include __enter__, __exit__, and a usage example.'
  'Implement Fisher-Yates shuffle in Python without using random.shuffle. Include type hints, docstring, and a test case.'
)

run_one() {
  local key=$1 prompt=$2 cache_prompt=$3
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
  printf "  %-12s pred=%4dt %7sms = %6st/s  prompt=%4dt = %4st/s  draft=%4d acc=%4d (%5s%%) cached=%d\n" \
    "$key" "$n_pred" "$pred_ms" "$tps" "$prompt_n" "$prompt_tps" "$n_drafted" "$n_accepted" "$acc" "$n_cached"
  jq -nc --arg label "$key" --argjson npred "$n_pred" --argjson pms "$pred_ms" \
    --argjson promptn "$prompt_n" --argjson promptms "$prompt_ms" \
    --argjson nd "$n_drafted" --argjson na "$n_accepted" --argjson nc "$n_cached" \
    --arg tps "$tps" --arg ppt "$prompt_tps" --arg acc "$acc" \
    '{label:$label, n_pred:$npred, predicted_ms:$pms, prompt_n:$promptn, prompt_ms:$promptms, draft_n:$nd, draft_accepted:$na, tokens_cached:$nc, tps:($tps|tonumber), prompt_tps:($ppt|tonumber), accept_pct:$acc}' \
    >> "$OUT"
}

echo "=== ${LABEL} ==="
echo "[A] Same prompt × 3 (cache_prompt=true): tests warm-cache repeat"
for i in 1 2 3; do run_one "A${i}-same" "${PROMPTS[0]}" true; done

echo "[B] 6 varied prompts (cache_prompt=true)"
for i in 0 1 2 3 4 5; do run_one "B$((i+1))-varied" "${PROMPTS[i]}" true; done

echo "[C] Re-run prompt[0] (recall test)"
run_one "C-recall0" "${PROMPTS[0]}" true

echo "[D] cache_prompt=false (forced reprocess)"
run_one "D-no-cache" "${PROMPTS[0]}" false

echo "Saved → $OUT"
