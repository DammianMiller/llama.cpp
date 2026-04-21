#pragma once

#include <cstdint>
#include <unordered_map>
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

//
// DDTree : branching verify-tree built from per-position top-K draft
// distributions. Node 0 of the flat tree is the implicit root (a bonus
// token carried over from the prior verify round); real draft nodes are
// indexed 1..n_nodes. parents[0] == -1 is the root sentinel.
//
// child_maps[i] maps a token id to the child flat-index of node i.
// visibility is a row-major (1+n_nodes) x (1+n_nodes) ancestor matrix:
// visibility[i*N + j] == 1 iff j is an ancestor of i (or i == j).
//
struct common_ddtree {
    int                                              n_nodes = 0;
    std::vector<int32_t>                             token_ids;    // size n_nodes
    std::vector<int>                                 depths;       // size n_nodes, 1..L
    std::vector<int>                                 parents;      // size 1+n_nodes, parents[0] = -1
    std::vector<std::unordered_map<int32_t, int>>    child_maps;   // size 1+n_nodes
    std::vector<uint8_t>                             visibility;   // (1+n_nodes)^2
};

// Build a DDTree from a vector of per-position top-K distributions.
//
// budget      : max number of draft nodes to keep (not counting the root)
// chain_seed  : if true, seed the tree with the top-1 chain of depth
//               min(L, budget) before opening the branching heap. this
//               matches the reference behaviour and keeps the "safe"
//               greedy chain always present.
//
// Returns a tree with n_nodes <= budget. If budget <= 0 or topk is empty,
// returns a degenerate root-only tree (n_nodes == 0).
common_ddtree common_ddtree_build(
    const std::vector<common_draft_topk_pos> & topk,
    int  budget,
    bool chain_seed = true);

// Kernel-style parent-id view. Returns a vector of length n_nodes where
// out[i] = tree.parents[i+1] - 1, with the convention that a node whose
// parent is the root (parents[i+1] == 0) maps to -1 (the pre-block state
// sentinel used by the verify kernel).
std::vector<int32_t> common_ddtree_parent_ids(const common_ddtree & tree);

// Build an F16 attention mask for the draft tree. Output is sized
// q_pad * kv_pad half-words (row-major, q as outer). For each query token
// t in [0, n_tokens) and key k in [0, kv_total):
//   - k < prompt_kv_start                 -> F16(0)            (prefix always visible)
//   - (k - prompt_kv_start) < n_tokens and
//     tree.visibility[t*n_tokens + (k-prompt_kv_start)] != 0  -> F16(0)
//   - otherwise                           -> F16(-INFINITY)    (0xFC00)
// Padding rows/cols are all -INFINITY. kv_pad rounds up to kq_mask_pad;
// q_pad rounds up to 32.
std::vector<uint16_t> common_ddtree_build_mask(
    const common_ddtree & tree,
    int                   prompt_kv_start,
    int                   kv_total,
    int                   n_tokens,
    int                   kq_mask_pad);

// Walk the tree along the argmax posterior at each accepted node.
// posterior[i] is the argmax token id sampled from logits of flat node i
// (including i == 0, the root/bonus slot). Starts at node 0 (always
// accepted), follows child_maps while the next sampled token is present,
// and writes the final diverging token to out_next_token. Returns the
// vector of accepted flat-node indices, always starting with 0.
std::vector<int> common_ddtree_follow_verified(
    const common_ddtree & tree,
    const int32_t *       posterior,
    int32_t &             out_next_token);
