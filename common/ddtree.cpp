#include "ddtree.h"

#include "ngram-mod.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <queue>
#include <utility>

static_assert(common_draft_topk_pos::K == common_ngram_mod_slot::K,
    "common_draft_topk_pos::K must match common_ngram_mod_slot::K");

static inline common_draft_topk_pos make_empty_pos() {
    common_draft_topk_pos p;
    for (int k = 0; k < common_draft_topk_pos::K; ++k) {
        p.ids      [k] = -1;
        p.log_probs[k] = -INFINITY;
    }
    return p;
}

std::vector<common_draft_topk_pos> common_draft_topk_from_ngram(
    const common_ngram_mod & ngram,
    const int32_t *          last_tokens_data,
    int                      n_positions,
    float                    alpha,
    float                    temperature) {

    std::vector<common_draft_topk_pos> out;
    if (n_positions <= 0) {
        return out;
    }
    out.reserve(n_positions);

    constexpr int K = common_draft_topk_pos::K;

    const size_t n = ngram.get_n();

    // Rolling window of size n that is re-hashed each iteration. For
    // position 0 it is a copy of last_tokens_data[0..n-1]; for each
    // subsequent position we drop the oldest token and append the previous
    // position's top-1 id.
    std::vector<int32_t> window(n);
    for (size_t i = 0; i < n; ++i) {
        window[i] = last_tokens_data[i];
    }

    // Guard against pathological temperature values. T<=0 is undefined; we
    // clamp to a small positive so division stays finite. Callers in the
    // DDTree path always pass a sane T, but we don't want a segfault from
    // a config typo.
    const float T = temperature > 0.0f ? temperature : 1e-6f;

    bool chain_terminated = false;

    for (int pos = 0; pos < n_positions; ++pos) {
        if (chain_terminated) {
            out.push_back(make_empty_pos());
            continue;
        }

        int32_t  ids [K];
        uint16_t hits[K];
        const int cnt = ngram.get_topk(window.data(), ids, hits);

        if (cnt == 0) {
            out.push_back(make_empty_pos());
            chain_terminated = true;
            continue;
        }

        // Laplace-smoothed probability denominator. cnt can be < K — in
        // that case we still use K*alpha in the denominator so the
        // distribution is well-defined over the full K-slot support.
        uint32_t sum_hits = 0;
        for (int k = 0; k < cnt; ++k) {
            sum_hits += hits[k];
        }
        const float denom = static_cast<float>(sum_hits) + static_cast<float>(K) * alpha;

        common_draft_topk_pos p = make_empty_pos();

        // ngram get_topk already returns entries in descending hit order,
        // so the resulting log-probs are automatically in descending order
        // (monotone-increasing transform of hits). Fill only the populated
        // entries; the rest stay as (-1, -INFINITY) from make_empty_pos.
        for (int k = 0; k < cnt; ++k) {
            const float pk = (static_cast<float>(hits[k]) + alpha) / denom;
            p.ids      [k] = ids[k];
            p.log_probs[k] = std::log(pk) / T;
        }

        out.push_back(p);

        // Advance the rolling window for the next position using this
        // position's top-1 id.
        for (size_t i = 0; i + 1 < n; ++i) {
            window[i] = window[i + 1];
        }
        window[n - 1] = p.ids[0];
    }

    return out;
}

//
// DDTree builder + helpers
//

namespace {

struct heap_entry {
    float            neg_logw;     // priority: smaller = higher logw
    std::vector<int> ranks;        // path of per-depth ranks (for tie-break reproducibility)
    int              parent_index; // flat-index of parent node in the tree
    int              depth;        // 1..L — depth of THIS candidate
    int              rank;         // sibling rank at this depth (0..K-1)
    float            logw;         // cumulative log-prob at this candidate
};

struct heap_cmp {
    bool operator()(const heap_entry & a, const heap_entry & b) const {
        return a.neg_logw > b.neg_logw;
    }
};

// F16(-INFINITY) bit pattern, used to mark "masked" positions in the
// attention mask. F16(0) is 0x0000. Hard-coded to avoid pulling ggml into
// this TU's public surface.
constexpr uint16_t F16_ZERO    = 0x0000;
constexpr uint16_t F16_NEG_INF = 0xFC00;

} // namespace

common_ddtree common_ddtree_build(
    const std::vector<common_draft_topk_pos> & topk,
    int  budget,
    bool chain_seed) {

    common_ddtree tree;

    // Root-only degenerate tree for zero-budget / empty input.
    const int L = static_cast<int>(topk.size());
    if (budget <= 0 || L <= 0) {
        tree.parents.push_back(-1);
        tree.child_maps.emplace_back();
        tree.visibility.assign(1, 1);
        return tree;
    }

    constexpr int K = common_draft_topk_pos::K;

    tree.parents.push_back(-1);
    tree.child_maps.emplace_back();

    std::priority_queue<heap_entry, std::vector<heap_entry>, heap_cmp> heap;

    if (chain_seed) {
        // Seed with the greedy top-1 chain up to min(L, budget) depth.
        const int chain_depth = std::min(L, budget);
        float cum_logw = 0.0f;
        int   prev_idx = 0;
        for (int d = 1; d <= chain_depth; ++d) {
            const common_draft_topk_pos & pos = topk[d - 1];
            const int32_t tok = pos.ids[0];

            // If the drafter ran out of predictions, bail out of the chain
            // early — subsequent positions are known-empty.
            if (tok < 0) {
                break;
            }

            cum_logw += pos.log_probs[0];
            const int cur = tree.n_nodes + 1;
            tree.token_ids.push_back(tok);
            tree.depths.push_back(d);
            tree.parents.push_back(prev_idx);
            tree.child_maps.emplace_back();
            tree.child_maps[prev_idx][tok] = cur;
            tree.n_nodes++;

            // Push the rank-1 sibling at this depth into the heap (if present).
            if (K > 1 && pos.ids[1] >= 0) {
                const float sib = cum_logw - pos.log_probs[0] + pos.log_probs[1];
                heap_entry e;
                e.neg_logw     = -sib;
                e.ranks        = { 1 };
                e.parent_index = prev_idx;
                e.depth        = d;
                e.rank         = 1;
                e.logw         = sib;
                heap.push(std::move(e));
            }

            prev_idx = cur;
        }
    } else {
        // Seed the heap with just the depth-1 rank-0 candidate.
        if (topk[0].ids[0] >= 0) {
            const float rlw = topk[0].log_probs[0];
            heap_entry e;
            e.neg_logw     = -rlw;
            e.ranks        = { 0 };
            e.parent_index = 0;
            e.depth        = 1;
            e.rank         = 0;
            e.logw         = rlw;
            heap.push(std::move(e));
        }
    }

    while (!heap.empty() && tree.n_nodes < budget) {
        heap_entry top = heap.top();
        heap.pop();

        const int dm1 = top.depth - 1;
        const common_draft_topk_pos & pos = topk[dm1];
        const int32_t tok = pos.ids[top.rank];

        // Defensive: if somehow an empty slot made it into the heap, skip.
        if (tok < 0) {
            continue;
        }

        const int cur = tree.n_nodes + 1;
        tree.token_ids.push_back(tok);
        tree.depths.push_back(top.depth);
        tree.parents.push_back(top.parent_index);
        tree.child_maps.emplace_back();
        tree.child_maps[top.parent_index][tok] = cur;
        tree.n_nodes++;

        // Push sibling (same parent, rank+1) if it exists.
        if (top.rank + 1 < K && pos.ids[top.rank + 1] >= 0) {
            const float sib = top.logw - pos.log_probs[top.rank] + pos.log_probs[top.rank + 1];
            heap_entry e;
            e.neg_logw     = -sib;
            e.ranks        = top.ranks;
            e.ranks.back() = top.rank + 1;
            e.parent_index = top.parent_index;
            e.depth        = top.depth;
            e.rank         = top.rank + 1;
            e.logw         = sib;
            heap.push(std::move(e));
        }

        // Push child (this node as parent, rank 0 at next depth) if we have
        // depth left and the next position has a top-1 candidate.
        if (top.depth < L && topk[top.depth].ids[0] >= 0) {
            const float chw = top.logw + topk[top.depth].log_probs[0];
            heap_entry e;
            e.neg_logw     = -chw;
            e.ranks        = top.ranks;
            e.ranks.push_back(0);
            e.parent_index = cur;
            e.depth        = top.depth + 1;
            e.rank         = 0;
            e.logw         = chw;
            heap.push(std::move(e));
        }
    }

    // Build the ancestor visibility matrix. visibility[i][j] == 1 iff j is
    // an ancestor of i (inclusive). Row-major, N = 1 + n_nodes.
    const int N = 1 + tree.n_nodes;
    tree.visibility.assign(static_cast<size_t>(N) * N, 0);
    tree.visibility[0] = 1;
    for (int i = 1; i < N; ++i) {
        const int p = tree.parents[i];
        for (int j = 0; j < i; ++j) {
            tree.visibility[static_cast<size_t>(i) * N + j] = tree.visibility[static_cast<size_t>(p) * N + j];
        }
        tree.visibility[static_cast<size_t>(i) * N + i] = 1;
    }

    return tree;
}

std::vector<int32_t> common_ddtree_parent_ids(const common_ddtree & tree) {
    std::vector<int32_t> out;
    out.reserve(tree.n_nodes);
    for (int i = 0; i < tree.n_nodes; ++i) {
        // parents[i+1] is the flat-tree parent. parent == 0 means the root
        // (pre-block state), which the verify kernel flags with -1.
        out.push_back(static_cast<int32_t>(tree.parents[i + 1]) - 1);
    }
    return out;
}

std::vector<uint16_t> common_ddtree_build_mask(
    const common_ddtree & tree,
    int                   prompt_kv_start,
    int                   kv_total,
    int                   n_tokens,
    int                   kq_mask_pad) {

    // Pad kv to a multiple of kq_mask_pad, and q to a multiple of 32.
    const int kv_pad = kq_mask_pad > 0
        ? ((kv_total + kq_mask_pad - 1) / kq_mask_pad) * kq_mask_pad
        : kv_total;
    const int q_pad  = ((n_tokens + 31) / 32) * 32;

    std::vector<uint16_t> mask(static_cast<size_t>(q_pad) * kv_pad, F16_NEG_INF);

    const int N = 1 + tree.n_nodes; // visibility side length

    for (int t = 0; t < n_tokens; ++t) {
        const size_t row = static_cast<size_t>(t) * kv_pad;

        for (int k = 0; k < kv_total; ++k) {
            if (k < prompt_kv_start) {
                mask[row + k] = F16_ZERO;
                continue;
            }

            const int local = k - prompt_kv_start;
            if (local >= n_tokens) {
                // leave as F16_NEG_INF
                continue;
            }

            // Guard against malformed (n_tokens, N) pairings — if the caller
            // passed n_tokens > N, the excess rows/cols stay masked out.
            if (t < N && local < N &&
                tree.visibility[static_cast<size_t>(t) * N + local] != 0) {
                mask[row + k] = F16_ZERO;
            }
        }
    }

    return mask;
}

std::vector<int> common_ddtree_follow_verified(
    const common_ddtree & tree,
    const int32_t *       posterior,
    int32_t &             out_next_token) {

    std::vector<int> accepted;
    accepted.reserve(tree.n_nodes + 1);
    accepted.push_back(0);

    int     cur = 0;
    int32_t nxt = posterior[0];

    while (true) {
        const auto & ch = tree.child_maps[cur];
        auto it = ch.find(nxt);
        if (it == ch.end()) {
            break;
        }
        cur = it->second;
        accepted.push_back(cur);
        nxt = posterior[cur];
    }

    out_next_token = nxt;
    return accepted;
}
