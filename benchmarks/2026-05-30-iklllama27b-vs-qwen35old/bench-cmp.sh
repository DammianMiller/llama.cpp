#!/usr/bin/env bash
# Compare-arm bench against http://127.0.0.1:8080 using response.timings.
# Decode suite: 3 prompts x 3 runs @256 tok. Prefill probe: 1 prompt x 2 runs @64 tok.
# Usage: ./bench-cmp.sh <arm_label> <output_jsonl>
set -euo pipefail
ARM="$1"; OUT="$2"; DIR="$(dirname "$0")"
ts(){ date +"%H:%M:%S"; }
run_one() { # prompts_file idx n_predict phase run
  local pf="$1" idx="$2" npred="$3" phase="$4" run="$5"
  local pid msgs req resp
  pid=$(jq -r ".[$idx].id" "$pf"); msgs=$(jq -c ".[$idx].messages" "$pf")
  req=$(jq -nc --argjson m "$msgs" --argjson n "$npred" '{model:"local",messages:$m,max_tokens:$n,temperature:0,stream:false,cache_prompt:false}')
  printf '\033[1;34m[%s]\033[0m %s %s run=%s\n' "$(ts)" "$ARM" "$pid" "$run" >&2
  resp=$(curl -s --max-time 600 -X POST -H "Content-Type: application/json" http://127.0.0.1:8080/v1/chat/completions -d "$req")
  echo "$resp" | python3 -c '
import sys,json
d=json.load(sys.stdin); t=d.get("timings",{}) or {}
row=dict(arm="'"$ARM"'",prompt="'"$pid"'",phase="'"$phase"'",run='"$run"',
  prompt_n=t.get("prompt_n"),prefill_tps=t.get("prompt_per_second"),
  predicted_n=t.get("predicted_n"),decode_tps=t.get("predicted_per_second"),
  draft_n=t.get("draft_n"),draft_acc=t.get("draft_n_accepted"),
  finish=(d.get("choices") or [{}])[0].get("finish_reason"))
print(json.dumps(row))' | tee -a "$OUT"
}
# Decode suite (warm: 3 runs)
for idx in 0 1 2; do for run in 1 2 3; do run_one "$DIR/prompts-decode.json" "$idx" 256 decode "$run"; done; done
# Prefill probe (2 runs)
for run in 1 2; do run_one "$DIR/prompts-prefill.json" 0 64 prefill "$run"; done
echo "[$(ts)] $ARM done -> $OUT" >&2
