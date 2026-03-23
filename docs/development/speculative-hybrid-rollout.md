# Hybrid Speculative Rollout Notes

This note documents the tested rollout approach for hybrid/recurrent models using n-gram speculative decoding, including the operational profile that gave the best balance of function and performance.

## Scope

- Target model family: Qwen3.5 variants
- Server mode: `llama-server`
- Speculative mode: `--spec-type ngram-cache`
- Context target: long-context operation (`--ctx-size 262144`) for agentic workflows

## Implementation Summary

The implementation in this branch keeps strict compatibility behavior between speculative rollback checks and recurrent/hybrid cache behavior by:

1. preserving compatibility probing in `common/speculative.cpp`;
2. using hybrid recurrent checkpoint save/restore in `src/llama-memory-hybrid.cpp`;
3. exposing and documenting checkpoint support in `src/llama-memory-hybrid.h`.

Operationally, the best tested profile came from balancing draft aggressiveness (not the most aggressive draft settings).

### Files changed in this branch

- `common/speculative.cpp`
- `src/llama-memory-hybrid.cpp`
- `src/llama-memory-hybrid.h`

### Behavioral contract

For hybrid/recurrent contexts, speculative compatibility is validated by rollback behavior in `common_speculative_is_compat()`. If rollback semantics drift, speculative should be treated as incompatible for that context.

In other words, correctness comes first: accepted drafts must never leave recurrent and attention memory out of sync.

## Operational Profiles Tested

### Baseline Fast Profile (historical)

```bash
--spec-type ngram-cache --draft-max 16 --draft-min 3 --draft-p-min 0.72
```

### Balanced Profile (best overall for function + throughput)

```bash
--spec-type ngram-cache --draft-max 12 --draft-min 2 --draft-p-min 0.80
```

The balanced profile reduced instability/loop pressure in agentic runs while keeping strong decode throughput.

### Suggested default startup flags (hybrid/recurrent)

```bash
--spec-type ngram-cache --draft-max 12 --draft-min 2 --draft-p-min 0.80
```

Use strict rollback mode for stability-first operation:

```bash
LLAMA_HYBRID_ROLLBACK_MODE=strict
```

## Benchmark and Validation Notes

### Qwen3.5-35B-A3B (long-context service)

- The balanced profile was selected after iterative canary runs with proxy tool-loop checks and server metrics enabled.
- In the final canary, critical errors stayed at zero in the sample window:
  - no `find_slot: non-consecutive token position` warnings,
  - no rollback failures,
  - no speculative auto-disable events.

Observed service-level metrics in tuned operation typically show:

- high prompt throughput (prefill),
- moderate decode throughput,
- stable acceptance without recurrent rollback failure churn.

### Qwen3.5-27B (Downloads model)

Measured at `ctx-size=262144`, `cache-type-k/v=q4_0`, CUDA full offload:

- no speculative: ~43 tok/s coding, ~41 tok/s pattern
- aggressive speculative (16/3/0.72): ~44 tok/s coding, ~102 tok/s pattern
- balanced speculative (12/2/0.80): ~43 tok/s coding, ~102 tok/s pattern

Interpretation:

- speculative uplift is workload-dependent;
- pattern/repetition-heavy generation benefits strongly;
- coding-style prompts are near throughput-neutral between aggressive and balanced settings, so balanced settings are preferred for reliability.

## Troubleshooting

### Low acceptance rate

Symptom:

- repeated low `draft acceptance rate` values.

Actions:

1. reduce `--draft-max`;
2. increase `--draft-p-min`;
3. re-check on representative prompts.

### Recurrent rollback warning / speculative disable

Symptom:

- `failed to rollback speculative draft tokens ...`
- `speculative decoding disabled for this slot ...`

Actions:

1. use strict rollback profile;
2. reduce draft aggressiveness;
3. confirm no stale context/process mix between runs.

### Context initialization OOM

Symptom:

- model loads but context or KV allocation fails at very large context.

Actions:

1. use quantized KV cache (`--cache-type-k q4_0 --cache-type-v q4_0`),
2. reduce `--ctx-size` for standalone benchmark process,
3. avoid running parallel GPU consumers during initialization.

## Recommended Rollout

1. Start with balanced speculative settings (`12/2/0.80`).
2. Keep metrics enabled and watch acceptance/rollback behavior.
3. Only move to more aggressive draft settings when acceptance is consistently high and function quality remains stable.
4. Re-validate with both coding-style and repetition-heavy prompts before promoting config changes.

## Credits and References

This work builds on existing llama.cpp speculative decoding and cache/checkpoint infrastructure maintained by the ggml-org/llama.cpp contributors.

Referenced work:

- Speculative decoding docs: `docs/speculative.md`
- n-gram cache/map lineage: #5479, #6828, #6848
- ngram-mod notes/video reference: #19164
- prompt/checkpoint behavior notes (SWA/hybrid implications):
  - https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055

Thanks to maintainers and contributors whose prior speculative and memory-path work enabled this targeted tuning and validation.
