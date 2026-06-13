#!/usr/bin/env bash
# Measure prefill_tps vs decode_tps as input length grows.
# Per request: parse `prompt eval time` and `eval time` from llama-server log.
# Output: per-arm JSONL with both rates.
#
# Usage: ./bench-prefill.sh <arm_label> <output_jsonl>
set -euo pipefail

ARM="$1"
OUT="$2"
LLAMA_LOG="$HOME/llama.cpp/llama-server.log"
N_OUT=128   # short output — we care about prefill, not decode length

ts() { date +"%Y-%m-%d %H:%M:%S"; }
log() { printf '\033[1;34m[bench %s]\033[0m %s\n' "$(ts)" "$*"; }

# Build prompts of escalating input size.
# Use a deterministic filler so prefill is comparable.
# Token count ≈ char count / 4 for English-ish text.
PYTHON_FILLER="def process(items):
    results = []
    for i, x in enumerate(items):
        if x is None or x < 0:
            continue
        results.append(x * 2)
    return results

"

for size_label in tiny short medium long verylong; do
    case "$size_label" in
        tiny)     reps=1     ;;   # ~50 tokens
        short)    reps=50    ;;   # ~2500 tokens
        medium)   reps=500   ;;   # ~25k tokens
        long)     reps=2000  ;;   # ~100k tokens
        verylong) reps=4000  ;;   # ~200k tokens (near ctx limit)
    esac
    log "building $size_label prompt (reps=$reps)"
    filler=""
    for _ in $(seq 1 $reps); do filler="${filler}${PYTHON_FILLER}"; done
    # Wrap as a question about the filler
    user_msg="Here is some Python code. Add a docstring to process().

\`\`\`python
${filler}\`\`\`

Just output the function with its new docstring."

    # Build request
    req=$(python3 -c "
import json, sys
msg = sys.stdin.read()
print(json.dumps({
    'model': 'local',
    'messages': [{'role': 'user', 'content': msg}],
    'max_tokens': $N_OUT,
    'temperature': 0,
    'stream': False,
}))
" <<< "$user_msg")

    log "  firing $size_label (input ~$((reps * 11)) chars)"
    log_pre=$(wc -l < "$LLAMA_LOG")
    t0=$(date +%s.%N)
    # Use stdin to avoid OS arg-list limit on large requests
    resp=$(printf '%s' "$req" | curl -s --max-time 600 -X POST -H "Content-Type: application/json" \
        http://127.0.0.1:8080/v1/chat/completions --data-binary @-)
    t1=$(date +%s.%N)
    wall=$(awk "BEGIN { print $t1 - $t0 }")

    n_out=$(echo "$resp" | jq -r '.usage.completion_tokens // 0')
    n_in=$(echo "$resp" | jq -r '.usage.prompt_tokens // 0')

    # Parse per-call timing from log
    log_post=$(wc -l < "$LLAMA_LOG")
    timings=$(sed -n "${log_pre},${log_post}p" "$LLAMA_LOG" | python3 -c '
import re, sys, json
text = sys.stdin.read()
out = {"prompt_eval_ms": None, "prompt_eval_tps": None, "prompt_eval_n": None,
       "decode_eval_ms": None, "decode_eval_tps": None, "decode_eval_n": None}
# llama-server timing lines look like:
#   prompt eval time =   X ms / N tokens (Y ms per token, Z tokens per second)
#         eval time =   X ms / N tokens (Y ms per token, Z tokens per second)
pe = re.search(r"prompt eval time =\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens\s*\(\s*[\d.]+\s*ms per token,\s*([\d.]+)\s*tokens per second", text)
if pe:
    out["prompt_eval_ms"]  = float(pe.group(1))
    out["prompt_eval_n"]   = int(pe.group(2))
    out["prompt_eval_tps"] = float(pe.group(3))
# strict non-"prompt" prefix for decode line
for m in re.finditer(r"(^|\n)\s*(?<!prompt )eval time =\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens\s*\(\s*[\d.]+\s*ms per token,\s*([\d.]+)\s*tokens per second", text):
    out["decode_eval_ms"]  = float(m.group(2))
    out["decode_eval_n"]   = int(m.group(3))
    out["decode_eval_tps"] = float(m.group(4))
print(json.dumps(out))
')

    pe_tps=$(echo "$timings" | jq -c '.prompt_eval_tps')
    pe_n=$(echo "$timings"   | jq -c '.prompt_eval_n')
    de_tps=$(echo "$timings" | jq -c '.decode_eval_tps')
    de_n=$(echo "$timings"   | jq -c '.decode_eval_n')

    jq -nc --arg arm "$ARM" --arg size "$size_label" \
        --argjson reps "$reps" --argjson wall "$wall" \
        --argjson n_in "$n_in" --argjson n_out "$n_out" \
        --argjson pe_tps "$pe_tps" --argjson pe_n "$pe_n" \
        --argjson de_tps "$de_tps" --argjson de_n "$de_n" \
        '{arm:$arm, size:$size, reps:$reps, wall_s:$wall, n_in:$n_in, n_out:$n_out,
          prompt_eval_tps:$pe_tps, prompt_eval_n:$pe_n,
          decode_eval_tps:$de_tps, decode_eval_n:$de_n}' \
        | tee -a "$OUT"
done

log "arm $ARM complete — results in $OUT"
