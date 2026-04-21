#pragma once

#include "llama-batch.h"
#include "llama-graph.h"
#include "llama-kv-cache.h"
#include "llama-memory.h"
#include "llama-memory-recurrent.h"

#include <memory>
#include <vector>
#include <unordered_map>

//
// llama_memory_hybrid
//

// utilizes instances of llama_memory_recurrent and llama_kv_cache to
//   support models where each layer may be either attention-based or recurrent

class llama_memory_hybrid : public llama_memory_i {
public:
    llama_memory_hybrid(
        const llama_model & model,
                            /* attn */
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                 uint32_t   kv_size,
                 uint32_t   n_pad,
                 uint32_t   n_swa,
           llama_swa_type   swa_type,
                            /* recurrent */
                ggml_type   type_r,
                ggml_type   type_s,
                 uint32_t   rs_size,
                            /* common */
                 uint32_t   n_seq_max,
                     bool   offload,
                     bool   unified,
                            /* layer filters */
    const layer_filter_cb & filter_attn = nullptr,
    const layer_filter_cb & filter_recr = nullptr);

    ~llama_memory_hybrid() = default;

    //
    // llama_memory_i
    //

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_context_ptr init_full() override;

    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    bool get_can_shift() const override;

    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id)                                                          override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    // state write/load

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0)       override;

    //
    // llama_memory_hybrid specific API
    //

    llama_kv_cache * get_mem_attn() const;
    llama_memory_recurrent * get_mem_recr() const;

    // Optional per-verify-token persist cache used by speculative decoding to
    // roll back the recurrent state without a replay forward. Idempotent:
    // calling enable_verify_cache() a second time with a larger budget grows
    // the buffers; a smaller budget is a no-op (we keep the bigger buffer).
    // Returns false on OOM or if this model has no recurrent layers.
    bool enable_verify_cache(int max_verify_tokens, ggml_type persist_type);
    void disable_verify_cache();

    bool          verify_cache_enabled() const { return m_verify_cache_enabled; }
    int           verify_cache_max_tokens() const { return m_max_verify_tokens; }
    ggml_type     verify_cache_type() const { return m_persist_type; }

    // persist_inter tensor for recurrent layer `il`, or nullptr if the layer is
    // attention-only or the verify cache is disabled.
    ggml_tensor * get_ssm_intermediate(int32_t il) const;

    // conv_input_cache tensor for recurrent layer `il`, or nullptr if the
    // layer is attention-only or the verify cache is disabled. Graph
    // builders write the full conv_input into this tensor during verify so
    // rollback can reconstruct the conv state for any accepted prefix.
    ggml_tensor * get_conv_input_cache(int32_t il) const;

    // Roll the ssm recurrent state of `seq_id` back to the intermediate slot
    // captured at the `commit_n`-th accepted token of the last verify forward.
    // commit_n is 1-based (1..max_verify_tokens). No-op if verify cache is
    // disabled, the buffer wasn't populated (no verify forward ran), or the
    // sequence has no active cell.
    void rollback_to_verify_slot(llama_seq_id seq_id, int n_verify, int commit_n);

private:
    const llama_hparams & hparams;

    const std::unique_ptr<llama_kv_cache> mem_attn;
    const std::unique_ptr<llama_memory_recurrent> mem_recr;

    // CPU-side checkpoint for speculative decoding rollback
    // Stores recurrent state (R/S tensors + cell position) in CPU RAM
    // before speculative batches, enabling rollback without extra GPU cells
    struct recurrent_checkpoint {
        llama_pos     pos = -1;
        int32_t       cell_id = -1;
        std::vector<std::vector<uint8_t>> r_data;  // per-layer R tensor data
        std::vector<std::vector<uint8_t>> s_data;  // per-layer S tensor data
        bool valid = false;
    };
    std::unordered_map<llama_seq_id, recurrent_checkpoint> cpu_checkpoints;

    // Save/restore recurrent state to/from CPU RAM
    void save_recurrent_checkpoint(llama_seq_id seq_id);
    bool restore_recurrent_checkpoint(llama_seq_id seq_id);
    bool has_recurrent_checkpoint(llama_seq_id seq_id) const;

    //
    // verify-cache persist buffers (Phase 2)
    //
    // ssm_intermediate[il]  shape [S_v*S_v, H, max_verify_tokens, n_seqs], f16/f32
    // conv_input_cache[il]  shape [(d_conv-1) + max_verify_tokens, conv_channels], f32
    // Layers that are not recurrent hold nullptr.
    //
    // Allocated lazily by enable_verify_cache(), freed by disable_verify_cache()
    // or on destruction. Buffers live on the same buffer-type as the layer's
    // recurrent state so gated_delta_net can write through on the same device.
    std::vector<ggml_tensor *> ssm_intermediate;
    std::vector<ggml_tensor *> conv_input_cache;

    // Per-layer buffer type captured at construction time so verify-cache
    // allocation can mirror the recurrent-state placement without needing a
    // stored reference to llama_model.
    std::vector<ggml_backend_buffer_type_t> layer_buft;

    // Owned ggml contexts + backend buffers backing the verify cache. Indexed
    // by buft (one pair per distinct buft across recurrent layers).
    std::vector<std::pair<ggml_context_ptr, ggml_backend_buffer_ptr>> verify_ctxs_bufs;

    bool      m_verify_cache_enabled = false;
    int       m_max_verify_tokens = 0;
    ggml_type m_persist_type = GGML_TYPE_F16;
    uint32_t  m_verify_n_seqs = 1;

    // Dimensions per recurrent layer (cached at construction so we don't have
    // to re-derive them when enabling the cache). Non-recurrent entries are 0.
    struct layer_dims {
        int64_t S_v;
        int64_t H;
        int64_t conv_channels;
        int64_t d_conv;
    };
    std::vector<layer_dims> layer_dims_arr;
};

class llama_memory_hybrid_context : public llama_memory_context_i {
public:
    using slot_info_vec_t = llama_kv_cache::slot_info_vec_t;

    // init failure
    explicit llama_memory_hybrid_context(llama_memory_status status);

    // init full
    explicit llama_memory_hybrid_context(llama_memory_hybrid * mem);

    // init update
    explicit llama_memory_hybrid_context(
        llama_memory_hybrid * mem,
              llama_context * lctx,
                       bool   optimize);

    // init success
    llama_memory_hybrid_context(
              llama_memory_hybrid * mem,
                  slot_info_vec_t   sinfos_attn,
        std::vector<llama_ubatch>   ubatches);

    ~llama_memory_hybrid_context() = default;

    bool next()  override;
    bool apply() override;

    llama_memory_status  get_status() const override;
    const llama_ubatch & get_ubatch() const override;

    //
    // llama_memory_hybrid_context
    //

    const llama_kv_cache_context * get_attn() const;
    const llama_memory_recurrent_context * get_recr() const;

    // Pass-through to llama_memory_hybrid::get_ssm_intermediate. Returns
    // nullptr when the verify cache is disabled. Exposed here so graph
    // builders that already hold a hybrid context can reach the persist buffer
    // without needing a second pointer.
    ggml_tensor * get_ssm_intermediate(int32_t il) const;

    // Pass-through to llama_memory_hybrid::get_conv_input_cache.
    ggml_tensor * get_conv_input_cache(int32_t il) const;

private:
    // the index of the next ubatch to process
    size_t i_next = 0;

    std::vector<llama_ubatch> ubatches;

    llama_memory_hybrid * mem = nullptr;

    const llama_memory_context_ptr ctx_attn;
    const llama_memory_context_ptr ctx_recr;

    const llama_memory_status status;
};
