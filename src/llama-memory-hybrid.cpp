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
    )) {}

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
    ctx_attn(mem->get_mem_attn()->init_full()),
    ctx_recr(mem->get_mem_recr()->init_full()),
    status(llama_memory_status_combine(ctx_attn->get_status(), ctx_recr->get_status())) {
}

llama_memory_hybrid_context::llama_memory_hybrid_context(
        llama_memory_hybrid * mem,
              llama_context * lctx,
                       bool   optimize) :
    ctx_attn(mem->get_mem_attn()->init_update(lctx, optimize)),
    ctx_recr(mem->get_mem_recr()->init_update(lctx, optimize)),
    status(llama_memory_status_combine(ctx_attn->get_status(), ctx_recr->get_status())) {
}

llama_memory_hybrid_context::llama_memory_hybrid_context(
              llama_memory_hybrid * mem,
                  slot_info_vec_t   sinfos_attn,
        std::vector<llama_ubatch>   ubatches) :
    ubatches(std::move(ubatches)),
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
