#include "llama-memory-hybrid.h"

#include "llama-impl.h"
#include "llama-model.h"
#include "llama-context.h"

#include <limits>
#include <unordered_set>

//
// llama_memory_hybrid
//

llama_memory_hybrid::llama_memory_hybrid(
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
    const layer_filter_cb & filter_attn,
    const layer_filter_cb & filter_recr) :
    hparams(model.hparams),
    mem_attn(new llama_kv_cache(
        model,
        type_k,
        type_v,
        v_trans,
        offload,
        unified,
        kv_size,
        n_seq_max,
        n_pad,
        n_swa,
        swa_type,
        filter_attn == nullptr ?
            [&](int32_t il) { return !hparams.is_recurrent(il); }
            : filter_attn,
        nullptr
    )),
    mem_recr(new llama_memory_recurrent(
        model,
        type_r,
        type_s,
        offload,
        rs_size,
        n_seq_max,
        filter_recr == nullptr ?
            [&](int32_t il) { return hparams.is_recurrent(il); }
            : filter_recr
    ))
{
    // Cache per-layer buft and delta-net dimensions up front so enable_verify_cache()
    // can allocate persist buffers without needing a live llama_model reference.
    const int32_t n_layer = hparams.n_layer;
    layer_buft.assign(n_layer, nullptr);
    layer_dims_arr.assign(n_layer, layer_dims{0, 0, 0, 0});
    m_verify_n_seqs = n_seq_max;

    for (int32_t il = 0; il < n_layer; ++il) {
        if (!hparams.is_recurrent(il)) {
            continue;
        }
        ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();
        if (offload) {
            buft = ggml_backend_dev_buffer_type(model.dev_layer(il));
        }
        layer_buft[il] = buft;

        // Matches qwen35/qwen35moe/qwen3next graph builders: S_v = ssm_d_inner / ssm_dt_rank,
        // H = ssm_dt_rank, conv_channels = ssm_d_inner + 2 * ssm_n_group * ssm_d_state.
        // For non-delta-net recurrent layers (e.g. plain mamba), these aren't used;
        // enable_verify_cache() will simply skip allocating persist tensors when H == 0.
        const int64_t d_inner      = hparams.ssm_d_inner;
        const int64_t num_v_heads  = hparams.ssm_dt_rank;
        const int64_t head_v_dim   = num_v_heads > 0 ? d_inner / num_v_heads : 0;
        layer_dims_arr[il] = layer_dims{
            /*S_v          */ head_v_dim,
            /*H            */ num_v_heads,
            /*conv_channels*/ (int64_t) d_inner + 2 * (int64_t) hparams.ssm_n_group * (int64_t) hparams.ssm_d_state,
            /*d_conv       */ (int64_t) hparams.ssm_d_conv,
        };
    }

    ssm_intermediate.assign(n_layer, nullptr);
    conv_input_cache.assign(n_layer, nullptr);
}

llama_memory_context_ptr llama_memory_hybrid::init_batch(llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) {
    do {
        balloc.split_reset();

        // follow the recurrent pattern for creating the ubatch splits
        std::vector<llama_ubatch> ubatches;

        while (true) {
            llama_ubatch ubatch;

            if (embd_all) {
                // if all tokens are output, split by sequence
                ubatch = balloc.split_seq(n_ubatch);
            } else {
                // Use non-sequential split when KV cache is unified (needed for hellaswag/winogrande/multiple-choice)
                const bool unified = (mem_attn->get_n_stream() == 1);
                ubatch = balloc.split_equal(n_ubatch, !unified);
            }

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        // Save recurrent checkpoints before short multi-token speculative batches.
        // Avoid checkpointing long prompt-prefill batches because CPU<->GPU copies
        // of recurrent state are expensive and unnecessary there.
        bool has_speculative_batch = false;
        std::unordered_set<llama_seq_id> seqs_to_checkpoint;
        constexpr uint32_t max_spec_checkpoint_tokens = 64;

        for (const auto & ub : ubatches) {
            if (ub.n_tokens <= 1 || ub.n_tokens > max_spec_checkpoint_tokens) {
                continue;
            }

            has_speculative_batch = true;

            if (ub.seq_id == nullptr || ub.n_seq_id == nullptr) {
                continue;
            }

            for (uint32_t s = 0; s < ub.n_seqs; ++s) {
                const uint32_t i = s*ub.n_seq_tokens;
                if (ub.n_seq_id[i] == 0) {
                    continue;
                }
                const llama_seq_id seq_id = ub.seq_id[i][0];
                seqs_to_checkpoint.insert(seq_id);
            }
        }

        if (has_speculative_batch) {
            for (const auto seq_id : seqs_to_checkpoint) {
                save_recurrent_checkpoint(seq_id);
            }
        }

        // prepare the recurrent batches first
        if (!mem_recr->prepare(ubatches)) {
            // TODO: will the recurrent cache be in an undefined context at this point?
            LLAMA_LOG_ERROR("%s: failed to prepare recurrent ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        // prepare the attention cache
        auto heads_attn = mem_attn->prepare(ubatches);
        if (heads_attn.empty()) {
            LLAMA_LOG_ERROR("%s: failed to prepare attention ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        return std::make_unique<llama_memory_hybrid_context>(
                this, std::move(heads_attn), std::move(ubatches));
    } while(false);

    return std::make_unique<llama_memory_hybrid_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_memory_hybrid::init_full() {
    return std::make_unique<llama_memory_hybrid_context>(this);
}

llama_memory_context_ptr llama_memory_hybrid::init_update(llama_context * lctx, bool optimize) {
    return std::make_unique<llama_memory_hybrid_context>(this, lctx, optimize);
}

bool llama_memory_hybrid::get_can_shift() const {
    // Shifting is trivially supported for recurrent
    return mem_attn->get_can_shift();
}

void llama_memory_hybrid::clear(bool data) {
    mem_attn->clear(data);
    mem_recr->clear(data);
    cpu_checkpoints.clear();
}

void llama_memory_hybrid::save_recurrent_checkpoint(llama_seq_id seq_id) {
    // Find the cell for this sequence
    int32_t tail_id = -1;
    llama_pos best_pos = -1;
    for (uint32_t i = 0; i < mem_recr->size; ++i) {
        const auto & cell = mem_recr->cells[i];
        if (cell.has_seq_id(seq_id) && !cell.is_empty() && cell.pos > best_pos) {
            tail_id = (int32_t)i;
            best_pos = cell.pos;
        }
    }
    if (tail_id < 0) {
        cpu_checkpoints.erase(seq_id);
        return;
    }

    auto & ckpt = cpu_checkpoints[seq_id];
    ckpt.pos = mem_recr->cells[tail_id].pos;
    ckpt.cell_id = tail_id;
    ckpt.valid = true;

    // Save R/S tensor data to CPU RAM
    const uint32_t n_r = (uint32_t)mem_recr->r_l.size();
    const uint32_t n_s = (uint32_t)mem_recr->s_l.size();

    ckpt.r_data.resize(n_r);
    ckpt.s_data.resize(n_s);

    for (uint32_t il = 0; il < n_r; ++il) {
        ggml_tensor * t = mem_recr->r_l[il];
        if (t == nullptr) {
            ckpt.r_data[il].clear();
            continue;
        }
        const size_t row_size = ggml_row_size(t->type, hparams.n_embd_r());
        const size_t offset = (size_t) tail_id*row_size;
        ckpt.r_data[il].resize(row_size);
        ggml_backend_tensor_get(t, ckpt.r_data[il].data(), offset, row_size);
    }

    for (uint32_t il = 0; il < n_s; ++il) {
        ggml_tensor * t = mem_recr->s_l[il];
        if (t == nullptr) {
            ckpt.s_data[il].clear();
            continue;
        }
        const size_t row_size = ggml_row_size(t->type, hparams.n_embd_s());
        const size_t offset = (size_t) tail_id*row_size;
        ckpt.s_data[il].resize(row_size);
        ggml_backend_tensor_get(t, ckpt.s_data[il].data(), offset, row_size);
    }

    LLAMA_LOG_DEBUG("saved recurrent checkpoint for seq %d at pos %d (%u R + %u S tensors)\n",
        seq_id, ckpt.pos, n_r, n_s);
}

bool llama_memory_hybrid::restore_recurrent_checkpoint(llama_seq_id seq_id) {
    auto it = cpu_checkpoints.find(seq_id);
    if (it == cpu_checkpoints.end()) {
        return false;
    }

    auto & ckpt = it->second;

    // Find the cell for this sequence
    int32_t tail_id = -1;
    llama_pos best_pos = -1;
    for (uint32_t i = 0; i < mem_recr->size; ++i) {
        const auto & cell = mem_recr->cells[i];
        if (cell.has_seq_id(seq_id) && !cell.is_empty() && cell.pos > best_pos) {
            tail_id = (int32_t)i;
            best_pos = cell.pos;
        }
    }
    if (tail_id < 0) {
        return false;
    }

    // Restore R/S tensor data from CPU RAM
    const uint32_t n_r = (uint32_t)mem_recr->r_l.size();
    const uint32_t n_s = (uint32_t)mem_recr->s_l.size();

    if (ckpt.r_data.size() != n_r || ckpt.s_data.size() != n_s) {
        LLAMA_LOG_ERROR("checkpoint tensor count mismatch\n");
        return false;
    }

    for (uint32_t il = 0; il < n_r; ++il) {
        ggml_tensor * t = mem_recr->r_l[il];
        if (t == nullptr) {
            if (!ckpt.r_data[il].empty()) {
                LLAMA_LOG_ERROR("checkpoint R tensor mismatch at layer %u\n", il);
                return false;
            }
            continue;
        }
        const size_t row_size = ggml_row_size(t->type, hparams.n_embd_r());
        const size_t offset = (size_t) tail_id*row_size;
        if (ckpt.r_data[il].size() != row_size) {
            LLAMA_LOG_ERROR("checkpoint R tensor size mismatch at layer %u\n", il);
            return false;
        }
        ggml_backend_tensor_set(t, ckpt.r_data[il].data(), offset, row_size);
    }

    for (uint32_t il = 0; il < n_s; ++il) {
        ggml_tensor * t = mem_recr->s_l[il];
        if (t == nullptr) {
            if (!ckpt.s_data[il].empty()) {
                LLAMA_LOG_ERROR("checkpoint S tensor mismatch at layer %u\n", il);
                return false;
            }
            continue;
        }
        const size_t row_size = ggml_row_size(t->type, hparams.n_embd_s());
        const size_t offset = (size_t) tail_id*row_size;
        if (ckpt.s_data[il].size() != row_size) {
            LLAMA_LOG_ERROR("checkpoint S tensor size mismatch at layer %u\n", il);
            return false;
        }
        ggml_backend_tensor_set(t, ckpt.s_data[il].data(), offset, row_size);
    }

    // Reset cell position to checkpoint position
    mem_recr->cells[tail_id].pos = ckpt.pos;

    LLAMA_LOG_DEBUG("restored recurrent checkpoint for seq %d to pos %d\n", seq_id, ckpt.pos);
    return true;
}

bool llama_memory_hybrid::has_recurrent_checkpoint(llama_seq_id seq_id) const {
    const auto it = cpu_checkpoints.find(seq_id);
    return it != cpu_checkpoints.end() && it->second.valid;
}

bool llama_memory_hybrid::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    // Try removing from the recurrent cache first since it may fail. If it does
    // fail, the cache will not have been mutated.
    if (!mem_recr->seq_rm(seq_id, p0, p1)) {
        // For recurrent models, partial tail removal can fail because tensor state
        // is not directly rewindable. If we have a matching checkpoint for p0 - 1,
        // restore it and continue removing the attention cache.
        const bool rm_tail = p1 < 0 || p1 == std::numeric_limits<llama_pos>::max();
        if (p0 <= 0 || !rm_tail) {
            return false;
        }

        // Accept checkpoint at pos <= p0-1 (not just == p0-1). When checkpoint
        // is further back than target, restore it and trim attention KV to match.
        // Server's activation replay re-decodes tokens from (ckpt.pos + 1) to
        // (p0 - 1) to bring both caches in sync.
        auto it = cpu_checkpoints.find(seq_id);
        if (it != cpu_checkpoints.end() && it->second.valid && it->second.pos <= p0 - 1) {
            if (!restore_recurrent_checkpoint(seq_id)) {
                return false;
            }
            const llama_pos attn_trim_from = it->second.pos + 1;
            return mem_attn->seq_rm(seq_id, attn_trim_from, p1);
        } else {
            // Fallback: keep recurrent positions aligned with attention cache even if
            // we don't have a usable checkpoint.
            bool aligned = false;
            for (auto & cell : mem_recr->cells) {
                if (cell.has_seq_id(seq_id) && cell.pos >= p0) {
                    cell.pos = p0 - 1;
                    aligned = true;
                }
            }
            if (!aligned) {
                return false;
            }
        }

        return mem_attn->seq_rm(seq_id, p0, p1);
    }
    return mem_attn->seq_rm(seq_id, p0, p1);
}

void llama_memory_hybrid::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    mem_attn->seq_cp(seq_id_src, seq_id_dst, p0, p1);
    mem_recr->seq_cp(seq_id_src, seq_id_dst, p0, p1);
}

void llama_memory_hybrid::seq_keep(llama_seq_id seq_id) {
    mem_attn->seq_keep(seq_id);
    mem_recr->seq_keep(seq_id);
}

void llama_memory_hybrid::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    mem_attn->seq_add(seq_id, p0, p1, shift);
    mem_recr->seq_add(seq_id, p0, p1, shift);
}

void llama_memory_hybrid::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    mem_attn->seq_div(seq_id, p0, p1, d);
    mem_recr->seq_div(seq_id, p0, p1, d);
}

llama_pos llama_memory_hybrid::seq_pos_min(llama_seq_id seq_id) const {
    // the min of the total cache is the max of the two caches' min values
    return std::max(mem_attn->seq_pos_min(seq_id), mem_recr->seq_pos_min(seq_id));
}

llama_pos llama_memory_hybrid::seq_pos_max(llama_seq_id seq_id) const {
    // the max of the total cache is the min of the two caches' max values
    return std::min(mem_attn->seq_pos_max(seq_id), mem_recr->seq_pos_max(seq_id));
}

std::map<ggml_backend_buffer_type_t, size_t> llama_memory_hybrid::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> mb = mem_attn->memory_breakdown();
    for (const auto & buft_size : mem_recr->memory_breakdown()) {
        mb[buft_size.first] += buft_size.second;
    }
    return mb;
}

void llama_memory_hybrid::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        mem_attn->state_write(io, seq_id, flags);
    }
    mem_recr->state_write(io, seq_id, flags);
}

void llama_memory_hybrid::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        mem_attn->state_read(io, seq_id, flags);
    }
    mem_recr->state_read(io, seq_id, flags);
}

llama_kv_cache * llama_memory_hybrid::get_mem_attn() const {
    return mem_attn.get();
}

llama_memory_recurrent * llama_memory_hybrid::get_mem_recr() const {
    return mem_recr.get();
}

llama_memory_hybrid_context::llama_memory_hybrid_context(llama_memory_status status) : status(status) {}

llama_memory_hybrid_context::llama_memory_hybrid_context(llama_memory_hybrid * mem) :
    mem(mem),
    ctx_attn(mem->get_mem_attn()->init_full()),
    ctx_recr(mem->get_mem_recr()->init_full()),
    status(llama_memory_status_combine(ctx_attn->get_status(), ctx_recr->get_status())) {
}

llama_memory_hybrid_context::llama_memory_hybrid_context(
        llama_memory_hybrid * mem,
              llama_context * lctx,
                       bool   optimize) :
    mem(mem),
    ctx_attn(mem->get_mem_attn()->init_update(lctx, optimize)),
    ctx_recr(mem->get_mem_recr()->init_update(lctx, optimize)),
    status(llama_memory_status_combine(ctx_attn->get_status(), ctx_recr->get_status())) {
}

llama_memory_hybrid_context::llama_memory_hybrid_context(
              llama_memory_hybrid * mem,
                  slot_info_vec_t   sinfos_attn,
        std::vector<llama_ubatch>   ubatches) :
    ubatches(std::move(ubatches)),
    mem(mem),
    // note: here we copy the ubatches. not sure if this is ideal
    ctx_attn(new llama_kv_cache_context(mem->get_mem_attn(), std::move(sinfos_attn), this->ubatches)),
    ctx_recr(new llama_memory_recurrent_context(mem->get_mem_recr(), this->ubatches)),
    status(llama_memory_status_combine(ctx_attn->get_status(), ctx_recr->get_status())) {
}

bool llama_memory_hybrid_context::next() {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    ctx_attn->next();
    ctx_recr->next();

    if (++i_next >= ubatches.size()) {
        return false;
    }

    return true;
}

bool llama_memory_hybrid_context::apply() {
    assert(!llama_memory_status_is_fail(status));

    bool res = true;

    res = res & ctx_attn->apply();
    res = res & ctx_recr->apply();

    return res;
}

llama_memory_status llama_memory_hybrid_context::get_status() const {
    return status;
}

const llama_ubatch & llama_memory_hybrid_context::get_ubatch() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);
    return ubatches[i_next];
}

const llama_kv_cache_context * llama_memory_hybrid_context::get_attn() const {
    return static_cast<const llama_kv_cache_context *>(ctx_attn.get());
}

const llama_memory_recurrent_context * llama_memory_hybrid_context::get_recr() const {
    return static_cast<const llama_memory_recurrent_context *>(ctx_recr.get());
}


ggml_tensor * llama_memory_hybrid_context::get_ssm_intermediate(int32_t il) const {
    return mem ? mem->get_ssm_intermediate(il) : nullptr;
}

ggml_tensor * llama_memory_hybrid_context::get_conv_input_cache(int32_t il) const {
    return mem ? mem->get_conv_input_cache(il) : nullptr;
}

//
// verify-cache (Phase 2) — persist buffers for speculative decoding rollback
//

bool llama_memory_hybrid::enable_verify_cache(int max_verify_tokens, ggml_type persist_type) {
    if (max_verify_tokens <= 0) {
        LLAMA_LOG_WARN("%s: max_verify_tokens must be > 0\n", __func__);
        return false;
    }
    if (persist_type != GGML_TYPE_F32 && persist_type != GGML_TYPE_F16) {
        LLAMA_LOG_WARN("%s: persist_type must be F32 or F16\n", __func__);
        return false;
    }

    // Find recurrent layers that look like gated-delta-net (H > 0 && S_v > 0).
    std::vector<int32_t> gdn_layers;
    for (int32_t il = 0; il < (int32_t) layer_dims_arr.size(); ++il) {
        const auto & d = layer_dims_arr[il];
        if (d.H > 0 && d.S_v > 0) {
            gdn_layers.push_back(il);
        }
    }
    if (gdn_layers.empty()) {
        LLAMA_LOG_WARN("%s: model has no gated-delta-net layers; verify cache not enabled\n", __func__);
        return false;
    }

    // Idempotent: if already enabled with >= budget and same dtype, do nothing.
    if (m_verify_cache_enabled && persist_type == m_persist_type && max_verify_tokens <= m_max_verify_tokens) {
        return true;
    }

    // Reallocate: drop existing buffers first so the old ones are freed before
    // we try to allocate new ones.
    disable_verify_cache();

    const int32_t n_layer = hparams.n_layer;
    ssm_intermediate.assign(n_layer, nullptr);
    conv_input_cache.assign(n_layer, nullptr);

    struct buft_ctx {
        ggml_context_ptr ctx;
    };
    // key: buft pointer, value: ggml_context holding tensors for layers on that buft
    std::map<ggml_backend_buffer_type_t, buft_ctx> ctx_map;

    // Count per-buft tensors up front so we can size each ggml_context correctly
    // (2 tensors per gdn layer). Non-gdn recurrent layers get no persist tensor.
    std::map<ggml_backend_buffer_type_t, int32_t> tensors_per_buft;
    for (int32_t il : gdn_layers) {
        tensors_per_buft[layer_buft[il]] += 2;
    }

    for (auto & [buft, n_t] : tensors_per_buft) {
        ggml_init_params params = {
            /*.mem_size   =*/ size_t(n_t * ggml_tensor_overhead()),
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
        };
        ggml_context * ctx = ggml_init(params);
        if (!ctx) {
            LLAMA_LOG_ERROR("%s: failed to create ggml context for verify cache\n", __func__);
            disable_verify_cache();
            return false;
        }
        ctx_map[buft].ctx.reset(ctx);
    }

    for (int32_t il : gdn_layers) {
        const auto & d = layer_dims_arr[il];
        ggml_context * ctx = ctx_map[layer_buft[il]].ctx.get();

        // ssm_intermediate: one S_v*S_v state matrix per (head, token, seq).
        ggml_tensor * t_inter = ggml_new_tensor_4d(
            ctx, persist_type,
            d.S_v * d.S_v, d.H, (int64_t) max_verify_tokens, (int64_t) m_verify_n_seqs);
        ggml_format_name(t_inter, "verify_ssm_inter_l%d", il);
        ssm_intermediate[il] = t_inter;

        // conv_input_cache: [(d_conv - 1) + max_verify_tokens, conv_channels].
        // Always F32 — matches qkv_mixed dtype in the graph builder.
        const int64_t conv_window = (d.d_conv > 0 ? d.d_conv - 1 : 0) + max_verify_tokens;
        ggml_tensor * t_conv = ggml_new_tensor_2d(
            ctx, GGML_TYPE_F32,
            conv_window, d.conv_channels);
        ggml_format_name(t_conv, "verify_conv_in_l%d", il);
        conv_input_cache[il] = t_conv;
    }

    // Allocate backend buffers.
    for (auto & [buft, entry] : ctx_map) {
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(entry.ctx.get(), buft);
        if (!buf) {
            LLAMA_LOG_ERROR("%s: failed to allocate verify cache buffer for %s\n", __func__, ggml_backend_buft_name(buft));
            disable_verify_cache();
            return false;
        }
        ggml_backend_buffer_clear(buf, 0);
        LLAMA_LOG_INFO("%s: %10s verify cache buffer size = %8.2f MiB\n",
                __func__, ggml_backend_buffer_name(buf),
                ggml_backend_buffer_get_size(buf)/1024.0/1024.0);
        verify_ctxs_bufs.emplace_back(std::move(entry.ctx), ggml_backend_buffer_ptr(buf));
    }

    m_verify_cache_enabled = true;
    m_max_verify_tokens    = max_verify_tokens;
    m_persist_type         = persist_type;
    return true;
}

void llama_memory_hybrid::disable_verify_cache() {
    // ctxs_bufs destructor frees tensors + buffers.
    verify_ctxs_bufs.clear();
    std::fill(ssm_intermediate.begin(), ssm_intermediate.end(), nullptr);
    std::fill(conv_input_cache.begin(), conv_input_cache.end(), nullptr);
    m_verify_cache_enabled = false;
    m_max_verify_tokens    = 0;
}

ggml_tensor * llama_memory_hybrid::get_ssm_intermediate(int32_t il) const {
    if (!m_verify_cache_enabled) {
        return nullptr;
    }
    if (il < 0 || il >= (int32_t) ssm_intermediate.size()) {
        return nullptr;
    }
    return ssm_intermediate[il];
}

ggml_tensor * llama_memory_hybrid::get_conv_input_cache(int32_t il) const {
    if (!m_verify_cache_enabled) {
        return nullptr;
    }
    if (il < 0 || il >= (int32_t) conv_input_cache.size()) {
        return nullptr;
    }
    return conv_input_cache[il];
}

void llama_memory_hybrid::rollback_to_verify_slot(llama_seq_id seq_id, int n_verify, int commit_n) {
    if (!m_verify_cache_enabled) {
        return;
    }
    if (n_verify <= 0 || n_verify > m_max_verify_tokens) {
        LLAMA_LOG_WARN("%s: n_verify %d out of range [1, %d]\n", __func__, n_verify, m_max_verify_tokens);
        return;
    }
    if (commit_n <= 0 || commit_n > n_verify) {
        LLAMA_LOG_WARN("%s: commit_n %d out of range [1, %d]\n", __func__, commit_n, n_verify);
        return;
    }

    // Locate the cell for this sequence.
    int32_t tail_id = -1;
    llama_pos best_pos = -1;
    for (uint32_t i = 0; i < mem_recr->size; ++i) {
        const auto & cell = mem_recr->cells[i];
        if (cell.has_seq_id(seq_id) && !cell.is_empty() && cell.pos > best_pos) {
            tail_id = (int32_t) i;
            best_pos = cell.pos;
        }
    }
    if (tail_id < 0) {
        return;
    }

    // Slot index (0-based) inside the intermediate buffer.
    const int slot = commit_n - 1;

    // Scratch for f16 -> f32 conversion when persist_type != type_s.
    std::vector<uint8_t> scratch_src;
    std::vector<uint8_t> scratch_dst;

    for (int32_t il = 0; il < (int32_t) ssm_intermediate.size(); ++il) {
        ggml_tensor * inter = ssm_intermediate[il];
        if (inter == nullptr) {
            continue;
        }
        ggml_tensor * s_dst = mem_recr->s_l[il];
        if (s_dst == nullptr) {
            continue;
        }

        const auto & d = layer_dims_arr[il];
        const size_t slot_elems = (size_t) d.S_v * d.S_v * d.H * m_verify_n_seqs;
        const size_t slot_src_bytes = slot_elems * ggml_type_size(inter->type);

        // Offset within persist tensor: [S_v*S_v, H, slot, :] -> skip `slot * S_v*S_v * H` elems.
        const size_t inter_slot_stride = (size_t) d.S_v * d.S_v * d.H * ggml_type_size(inter->type);
        const size_t inter_offset = (size_t) slot * inter_slot_stride;

        // Offset within s_l: row `tail_id` of [n_embd_s, mem_size].
        const size_t s_row_size = ggml_row_size(s_dst->type, hparams.n_embd_s());
        const size_t s_offset   = (size_t) tail_id * s_row_size;

        if (inter->type == s_dst->type) {
            scratch_src.resize(slot_src_bytes);
            ggml_backend_tensor_get(inter, scratch_src.data(), inter_offset, slot_src_bytes);
            ggml_backend_tensor_set(s_dst, scratch_src.data(), s_offset, slot_src_bytes);
        } else {
            // Cross-dtype path — read persist, convert to dst dtype, write back.
            scratch_src.resize(slot_src_bytes);
            ggml_backend_tensor_get(inter, scratch_src.data(), inter_offset, slot_src_bytes);

            const size_t slot_dst_bytes = slot_elems * ggml_type_size(s_dst->type);
            scratch_dst.resize(slot_dst_bytes);

            if (inter->type == GGML_TYPE_F16 && s_dst->type == GGML_TYPE_F32) {
                ggml_fp16_to_fp32_row((const ggml_fp16_t *) scratch_src.data(),
                                      (float *) scratch_dst.data(), slot_elems);
            } else if (inter->type == GGML_TYPE_F32 && s_dst->type == GGML_TYPE_F16) {
                ggml_fp32_to_fp16_row((const float *) scratch_src.data(),
                                      (ggml_fp16_t *) scratch_dst.data(), slot_elems);
            } else {
                LLAMA_LOG_WARN("%s: unsupported persist/state dtype pair (layer %d)\n", __func__, il);
                continue;
            }
            ggml_backend_tensor_set(s_dst, scratch_dst.data(), s_offset, slot_dst_bytes);
        }

        // conv_input_cache -> conv_state slice: take rows [commit_n .. commit_n + d_conv - 1)
        // from conv_input_cache[il] and write into r_l[il] for this cell.
        ggml_tensor * conv_in = conv_input_cache[il];
        ggml_tensor * r_dst   = mem_recr->r_l[il];
        if (conv_in == nullptr || r_dst == nullptr) {
            continue;
        }
        const int64_t d_conv_m1 = d.d_conv > 0 ? d.d_conv - 1 : 0;
        if (d_conv_m1 == 0) {
            continue;
        }
        // conv_input_cache layout: [conv_window, conv_channels], F32, row-major on dim 0.
        // We want rows [slot+1 .. slot+1 + d_conv_m1) across all conv_channels columns.
        // The destination (r_l) stores each cell's conv state flattened as
        //   row_stride = n_embd_r() = d_conv_m1 * conv_channels  (dtype type_r).
        const size_t r_row_size = ggml_row_size(r_dst->type, hparams.n_embd_r());
        const size_t r_offset   = (size_t) tail_id * r_row_size;

        // Read the required window from persist (F32).
        const size_t conv_slot_elems = (size_t) d_conv_m1 * d.conv_channels;
        scratch_src.resize(conv_slot_elems * sizeof(float));
        // conv_in strides: ne[0] = conv_window, ne[1] = conv_channels. We need
        // a rectangular block [slot+1 .. slot+1+d_conv_m1) x [0 .. conv_channels).
        // ggml stores dim 0 contiguous, so columns are separated by ne[0] * sizeof(float).
        const size_t row0_offset = (size_t) ((slot + 1)) * sizeof(float);
        const size_t col_stride  = (size_t) conv_in->ne[0] * sizeof(float);
        for (int64_t c = 0; c < d.conv_channels; ++c) {
            ggml_backend_tensor_get(
                conv_in,
                scratch_src.data() + (size_t) c * d_conv_m1 * sizeof(float),
                row0_offset + (size_t) c * col_stride,
                (size_t) d_conv_m1 * sizeof(float));
        }

        if (r_dst->type == GGML_TYPE_F32) {
            ggml_backend_tensor_set(r_dst, scratch_src.data(), r_offset, r_row_size);
        } else if (r_dst->type == GGML_TYPE_F16) {
            scratch_dst.resize(conv_slot_elems * sizeof(ggml_fp16_t));
            ggml_fp32_to_fp16_row((const float *) scratch_src.data(),
                                  (ggml_fp16_t *) scratch_dst.data(), conv_slot_elems);
            ggml_backend_tensor_set(r_dst, scratch_dst.data(), r_offset, r_row_size);
        } else {
            LLAMA_LOG_WARN("%s: unsupported conv state dtype for layer %d\n", __func__, il);
        }
    }

    // Reset recurrent cell position to reflect that `commit_n` of the last
    // `n_verify` tokens have been accepted.
    const llama_pos new_pos = best_pos - (llama_pos) (n_verify - commit_n);
    mem_recr->cells[tail_id].pos = new_pos;

    // Trim the attention KV to match: drop any cells with pos > new_pos for
    // this sequence. This keeps both caches in sync, replacing the
    // seq_rm + activation-replay sequence the caller would otherwise run.
    // mem_attn->seq_rm may partially succeed on hybrid attn cells (full-attn
    // layers store their own KV), but the hybrid spec-decode flow never has
    // a checkpoint on the attn side so the contiguous trim path is fine.
    if (mem_attn) {
        mem_attn->seq_rm(seq_id, new_pos + 1, -1);
    }

    LLAMA_LOG_DEBUG("%s: seq %d rolled back to commit_n=%d / n_verify=%d slot (pos %d)\n",
            __func__, seq_id, commit_n, n_verify, new_pos);
}
