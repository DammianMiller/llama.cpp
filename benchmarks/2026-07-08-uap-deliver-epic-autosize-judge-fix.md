# UAP deliver: epic context auto-size + acceptance-judge fix — live verification (2026-07-08)

Six live `uap deliver` runs on the local stack (llama.cpp b8780 `8974ec829`, Qwen3.6-35B-A3B-UD-IQ4_XS,
q8_0 KV, ctx 360000 = 2 rails × 180224, proxy :4000 window 180000) verifying the epic context
auto-size feature and diagnosing/fixing the acceptance-judge churn. Raw logs in
`2026-07-08-uap-deliver-epic-autosize-judge-fix/run{A..F}.log`.

UAP versions shipped during the exercise: v1.123.0 (PR #369 context auto-size) → v1.123.1
(PR #370 marker propagation) → v1.123.2 (PR #371 unchained sub-epics) → v1.124.0 (PR #372
churn breaker + --max-turns hard cap) → v1.124.1 (PR #373 judge evidence starvation).

## Run summary

| Run | UAP | Budget/session | Purpose | Outcome |
| --- | --- | --- | --- | --- |
| A | 1.123.0 | ~126k (rail-fit default) | auto-size smoke, default budget | budget line + budget-aware 4-phase plan + epics auto-engaged ✓; killed manually — acceptance judge rejected 100%-gate turns indefinitely (the churn this doc closes) |
| B | 1.123.0 | 5,734 (forced 8192×0.7) | hard-stop under pressure | stop fired live (`~6647/5734 est. tokens`, request never sent); marker feedback made the retry self-scope-down (read 1 file instead of 3) and deliver |
| C | 1.123.1 | 5,734 | full mission at tiny budget | SUCCESS: 7/7 hash-opaque tests green; one mid-run budget stop absorbed |
| D | 1.123.2 | 5,734 | engineered-impossible epic (3×10KB specs; mandatory reading alone > budget) | full split chain live: 5 budget-stopped sessions → `✂ epic read-specs outgrew its ~5,734-token session budget — re-planning as sub-epics` → namespaced sub-epics → cumulative assembly → parent accepted VIA split → mission success, 8/8 tests, exit 0 |
| E | 1.124.0 | ~126k | churn breaker + --max-turns hard cap | 6/6 epics accepted, exit 0; hard-cap notice printed; ZERO Turn-2 lines (was turns 2–5 per attempt in A); breaker fired 6× (judge still rejected every green turn → 2 attempts/epic) |
| F | 1.124.1 | ~126k | judge evidence fix | **6/6 epics accepted on FIRST attempt, first turn; zero breaker firings; zero extra turns; 16/16 tests green; exit 0** |

## Bugs found by live testing (all shipped)

1. **v1.123.1 — `[context-budget]` marker never reached the epic split path.** The epic runner's
   failure summary was goal-based, swallowing the executor's marker; unit tests stubbed the
   boundary and missed it. Fix: `IterationRecord.budgetStopped` tagging + marker appended to
   failed-run summaries.
2. **v1.123.2 — split sub-epics were dep-chained**, so the first piece failing full-project gates
   (inevitable mid-assembly) skipped the rest. Fix: unchained sequential pieces + a gates-green
   FINAL piece counts the parent epic as delivered (cumulative assembly across fresh sessions —
   this is what let run D succeed).
3. **v1.124.0 — acceptance churn breaker + explicit `--max-turns` hard cap.** The auto-optimizer
   silently enables the acceptance judge and escalation for complex missions; default-on
   until-delivered "extends past --max-turns" and the loop applied escalation `raiseMaxTurns`
   UNCAPPED when untilDelivered was off. Fixes: epic mode grades the epic GOAL (not the process
   prompt); secondary judge limited to `UAP_DELIVER_ACCEPTANCE_FLIP_LIMIT` (default 2) consecutive
   rejections of objectively-green turns, then gates win; CLI records the `--max-turns` option
   source and mirrors an explicit value into `maxTurnsCeiling`; the loop clamps `raiseMaxTurns`
   to the ceiling unconditionally.
4. **v1.124.1 — the judge's TRUE root cause: evidence starvation, not the qwen model.**
   `gatherEvidence` walked directories alphabetically (`data/` before `src/`); three 26KB
   `data/legacy-*.txt` files consumed the whole 60K evidence budget at 20K/file, so the judge
   never saw `package.json`/`src`/`test` and correctly reported every requirement "not visible".
   (Explains why runs C/D — no data files — never churned while A/E did.) Fix: priority-ordered
   evidence (configs+source > tests/structured > docs/data at a 1.5K head), candidate pool
   gathered before the file-count cap, and the secondary judge receives "objective gates ALL
   PASSED" as a runtime note. Probes on the finished project: setup + slugify specs 0.00 → 1.00
   (all criteria MET); negative control (spec for nonexistent modules) still rejects at 0.00.

## Judge before/after (same mission, same model, same budget)

| | Run A (pre-fix) | Run E (breaker) | Run F (evidence fix) |
| --- | --- | --- | --- |
| Epic acceptance | never (killed) | 2 attempts/epic (breaker) | **1 attempt, 1 turn** |
| Turns per attempt | 1 + 2–5 escalation | exactly 1 (hard cap) | exactly 1 |
| Breaker firings | n/a | 6 | **0** |
| Wall clock/epic | unbounded | ~5 min | ~2.5 min |

Net: the epic auto-size feature is live-verified end-to-end (budget resolution → budget-aware
planning → hard stop → self-scoping retry → ✂ split → cumulative sub-epic assembly), an explicit
`--max-turns` is a real hard cap, and the acceptance judge — with evidence it can actually see —
accepts green work on the first attempt while still rejecting missing work.
