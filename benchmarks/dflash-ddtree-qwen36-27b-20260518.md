# DFlash benchmark — Qwen3.6-27B (ddtree mode) — 2026-05-18

## Setup
- Engine: lucebox-hub `test_dflash` (built local, CUDA arch 86), driven by `scripts/bench_he.py`
- Target: `Qwen3.6-27B-Q4_K_M.gguf` (unsloth, 16 GB)
- Draft:  `dflash-draft-3.6-q8_0.gguf` (Lucebox, 1.8 GB)
- GPU: RTX 3090 24 GiB (production stack stopped for the run)
- Bench: HumanEval 10-prompt set, `--mode fast --ddtree-budget 22 --n-gen 128`
- Decode path: `--fast-rollback --ddtree --ddtree-budget=22`

## Background
First attempt used `--fast-rollback` alone (bench default) — all 10 prompts
crashed: `test_dflash` exits 1 right after `[step 0]`. Adding `--ddtree`
(lucebox's documented primary decode path) is this retry.

## Result

| prompt                  | steps | accept-len | accept% | decode tok/s |
|-------------------------|-------|-----------|---------|--------------|
| has_close_elements      |   14  |   9.14    |  57.1   |   129.10     |
| separate_paren_groups   |   20  |   6.40    |  40.0   |    89.47     |
| truncate_number         |    0  |   0.00    |   0.0   |     0.00  ✗  |
| below_zero              |   16  |   8.00    |  50.0   |   112.85     |
| mean_absolute_deviation |   35  |   3.66    |  22.9   |    51.45     |
| intersperse             |   26  |   4.92    |  30.8   |    69.22     |
| parse_nested_parens     |   17  |   7.53    |  47.1   |   105.93     |
| filter_by_substring     |   25  |   5.12    |  32.0   |    72.23     |
| sum_product             |    0  |   0.00    |   0.0   |     0.00  ✗  |
| rolling_max             |   25  |   5.12    |  32.0   |    71.72     |
|-------------------------|-------|-----------|---------|--------------|
| MEAN (incl. failures)   |       |   4.99    |  31.2   |    70.20     |
| MEAN (8 successful)     |       |  ~6.24    | ~39.0   |   ~87.75     |

## Verdict
- `--ddtree` mode **runs** where `--fast-rollback` alone crashed entirely.
- Still **not reliable**: 2/10 prompts (`truncate_number`, `sum_product`)
  hit the same `[step 0]` exit-1 crash. ~20% prompt failure rate.
- Successful prompts: ~88 tok/s mean decode, accept-len 3.7-9.1
  — broadly consistent with lucebox's published ~73-78 tok/s for this model.
- Production (Qwen3.6-35B-A3B + ngram-mod via UAP proxy) is unchanged and
  remains the deployed config. DFlash-via-lucebox is **not** production-ready
  at this failure rate; treat as an exploratory result only.

## Follow-up — `--mode batched` (pure tree-verify, no `--fast-rollback`)

Run: `bench_he.py --mode batched --ddtree-budget 22 --n-gen 128`

| prompt                  | accept-len | decode tok/s |
|-------------------------|-----------|--------------|
| has_close_elements      |   9.14    |   130.27     |
| separate_paren_groups   |   6.40    |    91.75     |
| truncate_number         |   0.00    |     0.00  ✗  |
| below_zero              |   8.00    |   115.05     |
| mean_absolute_deviation |   3.66    |    52.39     |
| intersperse             |   4.92    |    70.36     |
| parse_nested_parens     |   7.53    |   107.86     |
| filter_by_substring     |   5.12    |    73.35     |
| sum_product             |   0.00    |     0.00  ✗  |
| rolling_max             |   5.12    |    73.59     |
| MEAN (incl. failures)   |   4.99    |    71.46     |
| MEAN (8 successful)     |  ~6.24    |   ~89.33     |

**Conclusion:** batched mode is a wash vs ddtree-fast (~+1.5% on successful
prompts) — the chain `--fast-rollback` was never the speed cost. Critically,
the **same 2 prompts** (`truncate_number`, `sum_product`) fail at `[step 0]`
in batched mode too.

## Root cause — diagnosed 2026-05-22

Ran the 2 failing prompts directly through `test_dflash --ddtree
--ddtree-budget=22` with full instrumentation. It is **not a crash** — exit 0,
0 tokens generated. The decode loop hits the issue-#191 graceful break at
`test/test_dflash.cpp:2793` (`if (hit_eos || last_tok < 0 || ...) break;`).

Diagnostic output, both prompts identical:
```
[prefill] 86 tokens in 0.17 s, last_tok=0          <- pure target prefill
[step 0]  committed=86 last_tok=0 tree_N=23 accept=1 next=-1
[dflash]  generated 0 tokens  ->  0.00 tok/s   (exit 0)
```

- `[prefill] ... last_tok=0` — the **target model's prefill** (pure target,
  before any draft/speculation) argmaxes to token **0**.
- `[step 0] ... next=-1` — the verify-step argmax at the root tree slot = **-1**.
- `0` and `-1` are garbage argmax values (argmax of an all-zero / NaN logits
  row). `verify_compute=48 ms` confirms the graph ran — it produced bad logits.

**The target model's forward pass yields garbage logits for these 2 prompts.**
The failure is **upstream of speculative decoding** — that is why *both*
`--fast-rollback` and `--ddtree` modes fail the exact same 2 prompts. The
spec-decode layer is innocent; lucebox's **target graph** (q4_K CPU `tok_embd`
→ GPU migrate → 64-layer forward) is numerically unstable on specific prompt
content. Not length-related (131-tok prompt 08 fails; 128/135-tok prompts pass).

**Verdict:** a genuine numerical-correctness bug in lucebox's C++ target graph,
not a config/tokenizer/mode issue. Fixing it = debugging their CUDA forward
pass. DFlash-via-lucebox closed as exploratory; production (Qwen3.6-35B-A3B +
ngram-mod) is unaffected and faster.

## Layer-level pinpoint — 2026-05-22

Instrumented `build_qwen35_graph` (per-layer residual + block-output tagged
`ggml_set_output`, gated by env `DFLASH_NAN_PROBE`) and added a post-compute
NaN/Inf scan to test_dflash's prefill. Rebuilt `test_dflash`, ran both failing
prompts. Result — **identical for prompt 02 and prompt 08:**

```
[nanprobe] inp_embed   : clean
[nanprobe] >>> FIRST BAD LAYER = 3  (block=NaN/Inf, ffn/residual=NaN/Inf)
[nanprobe] final_norm  : NaN/Inf
[nanprobe] logits      : NaN/Inf
```

- Embeddings clean; layers **0, 1, 2 clean**; **layer 3 is the first NaN**.
- `block=NaN/Inf` — the NaN is present at the **attention/deltanet block
  output, before the FFN**. The FFN is not the origin.
- Layer 3 is the **first full-attention layer**: lucebox's qwen35 hybrid is
  full-attn when `(il+1) % 4 == 0` → il=3,7,11,…; layers 0-2 are Gated
  DeltaNet. So the NaN originates **inside `build_full_attn_block` at the
  first full-attention layer.**

**Root cause located: lucebox's full-attention block goes NaN at layer 3** for
specific prompt content (the 3 preceding DeltaNet layers are numerically fine).

## Op-level pinpoint — 2026-05-22

Instrumented inside `build_full_attn_block` — tagged every intermediate around
the attention (Q/K/V projections, q/k-norm, RoPE, Qfa, the flash-attn output,
gate, output projection) for layer 3. Rebuilt, re-ran both failing prompts.
Result — **identical for prompt 02 and prompt 08:**

```
[nanprobe]   L3.attn QGraw   : clean      <- Q+gate projection
[nanprobe]   L3.attn Qnorm   : clean      <- Q after q_norm
[nanprobe]   L3.attn gate    : clean
[nanprobe]   L3.attn Kraw    : clean      <- K projection
[nanprobe]   L3.attn Vraw    : clean      <- V projection
[nanprobe]   L3.attn Knorm   : clean      <- K after k_norm
[nanprobe]   L3.attn Qrope   : clean      <- Q after M-RoPE
[nanprobe]   L3.attn Krope   : clean      <- K after M-RoPE
[nanprobe]   L3.attn Qfa     : clean      <- Q permuted, FA-ready
[nanprobe]   L3.attn FAout   : NaN/Inf    <<< FIRST BAD TENSOR
[nanprobe]   L3.attn gatesig : clean
[nanprobe]   L3.attn Agated  : NaN/Inf    <- FAout * gatesig, inherits NaN
```

**THE OP: `ggml_flash_attn_ext` — `src/qwen35/qwen35_target_graph.cpp:619`.**

Every *input* to the flash attention is numerically clean — the query (`Qfa`),
and K/V (`Krope`/`Vraw`, clean before being written to the KV cache). The
flash-attention **output** is NaN/Inf. The NaN is **born inside lucebox's
flash-attention kernel.**

Since a correct causal prefill mask never yields an all-`-inf` query row
(position i always attends to ≥1 real key), the most likely mechanism is an
**f16 score-accumulator overflow** inside the FA kernel: these 2 prompts
contain a token pair whose Q·K dot product (head_dim=256) overflows the
kernel's f16 intermediate before `kq_scale` (1/16) is applied → Inf → softmax
→ NaN. The 8 working prompts lack that activation outlier. Classic Qwen-family
attention-logit outlier, manifesting in the kernel rather than in any tensor.

**Fileable upstream bug:** `ggml_flash_attn_ext` produces NaN on Qwen3.6-27B
layer 3 for HumanEval prompts `truncate_number` / `sum_product`, all inputs
finite — suggests an f16-overflow / softmax-stability issue in the FA kernel.
Going deeper (mask vs accumulator vs which CUDA kernel) needs device-side
instrumentation of lucebox's CUDA FA kernel — out of scope here.
