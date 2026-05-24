#!/usr/bin/env bash
# Sweep LLAMA_DRAFT_P_MIN across {0.65, 0.70, 0.75, 0.80} on prod Gemma 4 31B
# stack. For each value: edit env, restart llama-server, fire 3 prompts × 3
# runs (cold/warm/warm), parse decode t/s + accept rate from server log,
# emit JSONL. Restore prod state at end.
#
# Production downtime ≈ 4 values × ~7 min ≈ 30 min. Run from a quiet window.
set -euo pipefail

BENCH_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="$HOME/.config/uap/llama-server.env"
ENV_BACKUP="${ENV_FILE}.bak-pmin-sweep-$(date +%Y%m%d-%H%M%S)"
LLAMA_LOG="$HOME/llama.cpp/llama-server.log"
SLOTS_DIR="$HOME/.cache/uap/llama-slots"
PROMPTS_FILE="$BENCH_DIR/prompts.json"

SWEEP_VALUES=(0.65 0.70 0.75 0.80)
N_TOKENS=256
RUNS_PER_PROMPT=3   # run1=cold, run2=warm, run3=warmer

ts() { date +"%Y-%m-%d %H:%M:%S"; }
log() { printf '\033[1;34m[sweep %s]\033[0m %s\n' "$(ts)" "$*"; }

# Sanity
[[ -f "$PROMPTS_FILE" ]] || { echo "Missing $PROMPTS_FILE"; exit 1; }
command -v jq >/dev/null || { echo "jq required"; exit 1; }

restore_prod() {
    log "RESTORING production config from $ENV_BACKUP"
    cp -f "$ENV_BACKUP" "$ENV_FILE"
    systemctl --user restart uap-llama-server.service
    sleep 3
    for i in 1 2 3 4 5 6 7 8 9 10; do
        out=$(curl -s --max-time 3 http://127.0.0.1:8080/health 2>/dev/null || true)
        [[ "$out" == '{"status":"ok"}' ]] && break
        sleep 2
    done
    systemctl --user start uap-anthropic-proxy.service
    log "production restored"
}
trap restore_prod EXIT

log "backing up env to $ENV_BACKUP"
cp "$ENV_FILE" "$ENV_BACKUP"

log "stopping proxy (paused for sweep)"
systemctl --user stop uap-anthropic-proxy.service

log "clearing slot cache (avoid mismatched-slot hangs)"
rm -f "$SLOTS_DIR"/*.bin 2>/dev/null || true

# Pre-extract prompts to messages JSON arrays
mapfile -t PROMPT_IDS < <(jq -r '.[].id' "$PROMPTS_FILE")
log "prompts: ${PROMPT_IDS[*]}"

OUT_JSONL="$BENCH_DIR/results.jsonl"
: > "$OUT_JSONL"   # truncate
RESULTS_LOG="$BENCH_DIR/run.log"
: > "$RESULTS_LOG"

for pmin in "${SWEEP_VALUES[@]}"; do
    log "═══ p_min = $pmin ═══"
    sed -i -E "s|^LLAMA_DRAFT_P_MIN=.*|LLAMA_DRAFT_P_MIN=${pmin}|" "$ENV_FILE"
    grep -E "^LLAMA_DRAFT_P_MIN" "$ENV_FILE" >> "$RESULTS_LOG"

    log "  restarting llama-server"
    systemctl --user restart uap-llama-server.service
    sleep 5
    for i in $(seq 1 30); do
        out=$(curl -s --max-time 3 http://127.0.0.1:8080/health 2>/dev/null || true)
        [[ "$out" == '{"status":"ok"}' ]] && { log "  ready after ${i}x2s"; break; }
        sleep 2
    done

    # Reset llama log marker for parsing
    LOG_MARK_LINE=$(wc -l < "$LLAMA_LOG" 2>/dev/null || echo 0)

    for prompt_idx in $(seq 0 $((${#PROMPT_IDS[@]} - 1))); do
        prompt_id="${PROMPT_IDS[$prompt_idx]}"
        msgs=$(jq -c ".[$prompt_idx].messages" "$PROMPTS_FILE")

        for run in $(seq 1 $RUNS_PER_PROMPT); do
            log "  $prompt_id  run=$run"
            req=$(jq -nc \
                --argjson messages "$msgs" \
                --argjson n_predict "$N_TOKENS" \
                '{model:"local", messages:$messages, max_tokens:$n_predict,
                  temperature:0, stream:false}')

            # Mark log position pre-call
            log_pre=$(wc -l < "$LLAMA_LOG")
            t0=$(date +%s.%N)
            resp=$(curl -s --max-time 300 -X POST \
                -H "Content-Type: application/json" \
                http://127.0.0.1:8080/v1/chat/completions \
                -d "$req")
            t1=$(date +%s.%N)
            wall=$(awk "BEGIN { print $t1 - $t0 }")

            # Output tokens from response usage
            n_out=$(echo "$resp" | jq -r '.usage.completion_tokens // 0')
            n_in=$(echo "$resp" | jq -r '.usage.prompt_tokens // 0')
            finish=$(echo "$resp" | jq -r '.choices[0].finish_reason // "?"')

            # Extract latest "draft acceptance rate" line emitted since this call.
            # Python parsing avoids set -e/pipefail headaches when no draft line exists.
            log_post=$(wc -l < "$LLAMA_LOG")
            stats=$(sed -n "${log_pre},${log_post}p" "$LLAMA_LOG" | python3 -c '
import re, sys, json
text = sys.stdin.read()
lines = [l for l in text.splitlines() if "draft acceptance rate" in l]
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
            accept_n=$(echo "$stats"   | jq -c '.accept_n')
            accept_total=$(echo "$stats" | jq -c '.accept_total')

            tps=$(awk "BEGIN { if ($wall > 0) print $n_out / $wall; else print 0 }")

            jq -nc \
                --arg pmin "$pmin" --arg prompt "$prompt_id" \
                --argjson run "$run" \
                --argjson tps "$tps" --argjson wall "$wall" \
                --argjson n_out "$n_out" --argjson n_in "$n_in" \
                --arg finish "$finish" \
                --argjson accept_pct "$accept_pct" \
                --argjson accept_n "$accept_n" \
                --argjson accept_total "$accept_total" \
                '{p_min:($pmin|tonumber), prompt:$prompt, run:$run,
                  tps:$tps, wall_s:$wall, n_out:$n_out, n_in:$n_in,
                  finish:$finish, accept_pct:$accept_pct,
                  accept_n:$accept_n, accept_total:$accept_total}' \
                | tee -a "$OUT_JSONL"
        done
    done
done

log "sweep complete — results in $OUT_JSONL"
log "(prod auto-restored by trap on exit)"
