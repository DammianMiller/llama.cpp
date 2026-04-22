#pragma once

#include "llama.h"

#include <cstdint>

#define LLAMA_MAX_SEQ 256

struct llama_cparams {
    uint32_t n_ctx;           // context size used during inference
    uint32_t n_ctx_seq;       // context for a single sequence
    uint32_t n_batch;
    uint32_t n_ubatch;
    uint32_t n_seq_max;
    int32_t  n_threads;       // number of threads to use for generation
    int32_t  n_threads_batch; // number of threads to use for batch processing

    float rope_freq_base;
    float rope_freq_scale;

    uint32_t n_ctx_orig_yarn;
    // These hyperparameters are not exposed in GGUF, because all
    // existing YaRN models use the same values for them.
    float yarn_ext_factor;
    float yarn_attn_factor;
    float yarn_beta_fast;
    float yarn_beta_slow;

    bool embeddings;
    bool causal_attn;
    bool offload_kqv;
    bool flash_attn;
    bool auto_fa;
    bool fused_gdn_ar;       // use fused gated delta net (autoregressive)
    bool fused_gdn_ch;       // use fused gated delta net (chunked)
    bool auto_fgdn;
    bool no_perf;
    bool warmup;
    bool op_offload;
    bool kv_unified;
    bool pipeline_parallel;

    enum llama_pooling_type pooling_type;

    ggml_backend_sched_eval_callback cb_eval;
    void * cb_eval_user_data;

    // DDTree persist-verify cache. 0 = disabled. When set, the hybrid memory
    // allocates its ssm_intermediate + conv_input_cache at init time so the
    // scheduler's allocation plan accounts for them up front. Without this
    // path, allocating them post-init (via the late-binding
    // llama_memory_enable_verify_cache API) triggers CUDA-graph capture
    // warmup resets on subsequent decodes, costing ~10x decode tok/s.
    int32_t   verify_cache_max_tokens = 0;
    ggml_type verify_cache_type       = GGML_TYPE_F16;
};
