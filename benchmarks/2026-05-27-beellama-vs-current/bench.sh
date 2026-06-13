#!/usr/bin/env bash
# Single-arm bench: run all 3 prompts × 3 runs against http://127.0.0.1:8080.
# Writes results.jsonl with one row per (prompt, run).
# Usage: ./bench.sh <arm_label> <output_jsonl>
set -euo pipefail

ARM="$1"
OUT="$2"
PROMPTS="$(dirname "$0")/prompts.json"
LLAMA_LOG="$HOME/llama.cpp/llama-server.log"
N_TOKENS=256
RUNS=3

ts() { date +"%Y-%m-%d %H:%M:%S"; }
log() { printf '\033[1;34m[bench %s]\033[0m %s\n' "$(ts)" "$*"; }

[[ -f "$PROMPTS" ]] || { echo "Missing $PROMPTS"; exit 1; }
mapfile -t PROMPT_IDS < <(jq -r '.[].id' "$PROMPTS")

log "arm=$ARM prompts=${PROMPT_IDS[*]}"

for prompt_idx in $(seq 0 $((${#PROMPT_IDS[@]} - 1))); do
    prompt_id="${PROMPT_IDS[$prompt_idx]}"
    msgs=$(jq -c ".[$prompt_idx].messages" "$PROMPTS")

    for run in $(seq 1 $RUNS); do
        log "  $prompt_id run=$run"
        req=$(jq -nc --argjson messages "$msgs" --argjson n_predict "$N_TOKENS" \
            '{model:"local", messages:$messages, max_tokens:$n_predict, temperature:0, stream:false}')

        log_pre=$(wc -l < "$LLAMA_LOG" 2>/dev/null || echo 0)
        t0=$(date +%s.%N)
        resp=$(curl -s --max-time 300 -X POST -H "Content-Type: application/json" \
            http://127.0.0.1:8080/v1/chat/completions -d "$req")
        t1=$(date +%s.%N)
        wall=$(awk "BEGIN { print $t1 - $t0 }")

        n_out=$(echo "$resp" | jq -r '.usage.completion_tokens // 0')
        n_in=$(echo "$resp" | jq -r '.usage.prompt_tokens // 0')
        finish=$(echo "$resp" | jq -r '.choices[0].finish_reason // "?"')

        log_post=$(wc -l < "$LLAMA_LOG")
        stats=$(sed -n "${log_pre},${log_post}p" "$LLAMA_LOG" | python3 -c '
import re, sys, json
text = sys.stdin.read()
lines = [l for l in text.splitlines() if "draft acceptance" in l]
if not lines:
    print(json.dumps({"accept_pct": None, "accept_n": None, "accept_total": None}))
    sys.exit(0)
last = lines[-1]
m = re.search(r"=\s*([0-9.]+)\s*\(\s*(\d+)\s+accepted\s*/\s*(\d+)\s+generated\)", last)
if m:
    print(json.dumps({
        "accept_pct": float(m.group(1)),
        "accept_n": int(m.group(2)),
        "accept_total": int(m.group(3)),
    }))
else:
    print(json.dumps({"accept_pct": None, "accept_n": None, "accept_total": None}))
')
        accept_pct=$(echo "$stats" | jq -c '.accept_pct')
        accept_n=$(echo "$stats" | jq -c '.accept_n')
        accept_total=$(echo "$stats" | jq -c '.accept_total')
        tps=$(awk "BEGIN { if ($wall > 0) print $n_out / $wall; else print 0 }")

        jq -nc --arg arm "$ARM" --arg prompt "$prompt_id" --argjson run "$run" \
            --argjson tps "$tps" --argjson wall "$wall" \
            --argjson n_out "$n_out" --argjson n_in "$n_in" --arg finish "$finish" \
            --argjson accept_pct "$accept_pct" --argjson accept_n "$accept_n" --argjson accept_total "$accept_total" \
            '{arm:$arm, prompt:$prompt, run:$run, tps:$tps, wall_s:$wall,
              n_out:$n_out, n_in:$n_in, finish:$finish,
              accept_pct:$accept_pct, accept_n:$accept_n, accept_total:$accept_total}' \
            | tee -a "$OUT"
    done
done

log "arm $ARM complete — results in $OUT"
