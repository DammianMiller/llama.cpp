#!/usr/bin/env bash
# Spec-decoding A/B harness: same coding prompt + greedy (temp 0) decode against
# the live :8080 server. Run once per active spec config (MTP vs ngram-mod).
# Captures prompt/decode t/s + draft acceptance from the server timings block.
set -euo pipefail
LABEL="${1:-run}"
N="${2:-3}"          # iterations (first = cold, rest = warm)
MAXTOK="${3:-700}"
OUT="/tmp/bench-${LABEL}.jsonl"
: > "$OUT"

read -r -d '' PROMPT <<'P' || true
Implement a rate limiter in Python using the token bucket algorithm. Include a TokenBucket class with configurable capacity and refill rate, thread-safety using a lock, a consume(tokens=1) method returning True/False, and a decorator rate_limited(bucket) that blocks until tokens are available. Add docstrings and 3 pytest unit tests. Output only the code.
P

REQ=$(python3 -c "import json,sys; print(json.dumps({'model':'x','messages':[{'role':'user','content':sys.argv[1]}],'max_tokens':int(sys.argv[2]),'temperature':0,'cache_prompt':False}))" "$PROMPT" "$MAXTOK")

for i in $(seq 1 "$N"); do
  curl -s --max-time 180 http://127.0.0.1:8080/v1/chat/completions \
    -H 'Content-Type: application/json' -d "$REQ" \
  | python3 -c "
import json,sys
d=json.load(sys.stdin); t=d['timings']
dn=t.get('draft_n',0); da=t.get('draft_n_accepted',0)
acc=(100.0*da/dn) if dn else 0.0
print(json.dumps({'i':$i,'prompt_n':t.get('prompt_n'),'prompt_tps':round(t.get('prompt_per_second',0),1),
 'pred_n':t.get('predicted_n'),'decode_tps':round(t.get('predicted_per_second',0),2),
 'pred_ms':round(t.get('predicted_ms',0),1),'draft_n':dn,'draft_acc':da,'accept_pct':round(acc,1)}))
" | tee -a "$OUT"
done

echo "--- $LABEL summary (warm = runs 2..$N) ---"
python3 -c "
import json,sys
rows=[json.loads(l) for l in open('$OUT')]
warm=rows[1:] if len(rows)>1 else rows
import statistics as st
dt=[r['decode_tps'] for r in warm]
print('warm decode t/s: median=%.2f  min=%.2f  max=%.2f  (n=%d)'%(st.median(dt),min(dt),max(dt),len(dt)))
pn=warm[0]['pred_n']
if warm[0]['draft_n']:
    accs=[r['accept_pct'] for r in warm]
    print('draft accept %%: median=%.1f  drafted=%d accepted=%d (last warm run)'%(st.median(accs),warm[-1]['draft_n'],warm[-1]['draft_acc']))
else:
    print('draft accept: (not reported by this binary/spec)')
print('gen tokens per run: %s'%pn)
"
