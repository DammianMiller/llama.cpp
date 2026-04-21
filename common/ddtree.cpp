#include "ddtree.h"

#include "ngram-mod.h"

#include <cmath>
#include <cstdint>

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
