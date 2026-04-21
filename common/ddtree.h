#pragma once

#include <cstdint>
#include <vector>

struct common_ngram_mod; // fwd

//
// ddtree : top-K draft distribution helper over common_ngram_mod
//
// Produces per-position top-K (id, log-prob) pairs that a downstream
// DDTree builder can grow a branching verify-tree from. This is a
// read-only view of a drafter — it does not mutate the ngram store.
//

// Draft top-K distribution at one position. ids[k] and log_probs[k] are
// sorted DESCENDING by log_prob. Empty entries are filled with id = -1 and
// log_prob = -INFINITY.
struct common_draft_topk_pos {
    static constexpr int K = 4;   // matches common_ngram_mod_slot::K
    int32_t ids[K];
    float   log_probs[K];
};

// Produce per-position top-K log-prob distributions from an ngram-mod
// drafter, given a tail of previously emitted tokens. Walks the ngram
// greedily: position 0 conditions on `last_tokens` (most recent first), and
// each subsequent position conditions on the top-1 id from the previous
// position. For depths up to `n_positions`.
//
// Log-probs are synthesised from hit counts with a Laplace prior:
//   p_k     = (hits_k + alpha) / (sum_hits + K * alpha)
//   log_p_k = log(p_k) / T          (temperature T sharpens/flattens)
//
// Parameters:
//   last_tokens_data: pointer to the tail of emitted tokens; must point to
//                     at least n entries, where n = ngram.get_n(). These n
//                     tokens are the hash key for position 0.
//   n_positions:      desired prediction depth
//   alpha:            Laplace prior (default 1.0f — a slot with 0 hits still
//                     gets a tiny nonzero probability)
//   temperature:      temperature. <1 sharpens, >1 flattens (default 0.5 to
//                     compensate for the typically flat ngram distribution)
//
// Returns a vector of length n_positions. If the ngram lookup at position k
// returns no hits at all (all slots empty), the corresponding entry's ids[]
// will all be -1 and log_probs[] all -INFINITY — callers should treat this
// as "drafter has nothing for you", and all subsequent positions are also
// empty-padded (the greedy chain terminates).
std::vector<common_draft_topk_pos> common_draft_topk_from_ngram(
    const common_ngram_mod & ngram,
    const int32_t *          last_tokens_data,
    int                      n_positions,
    float                    alpha       = 1.0f,
    float                    temperature = 0.5f);
