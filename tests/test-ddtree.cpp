#include "ddtree.h"
#include "ngram-mod.h"

#include "ggml.h"   // ggml_fp16_to_fp32 for mask decoding

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

// Guaranteed-runtime check (asserts get stripped in Release builds).
#define TEST_CHECK(cond)                                                    \
    do {                                                                    \
        if (!(cond)) {                                                      \
            fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, #cond);\
            std::abort();                                                   \
        }                                                                   \
    } while (0)

static bool approx_eq(float a, float b, float tol = 1e-5f) {
    return std::fabs(a - b) <= tol;
}

// n=1: idx(tokens) collapses to `tokens[0] % size`, so picking tokens[0]=0
// always lands in slot 0 and successor is tokens[1]. Lets us build fully
// controlled fixtures.
static void test_k1_logprob_formula() {
    constexpr size_t N    = 1;
    constexpr size_t SIZE = 8;

    common_ngram_mod mod(N, SIZE);

    // add (0 -> 77) five times
    int32_t t[2] = { 0, 77 };
    for (int i = 0; i < 5; ++i) {
        mod.add(t);
    }

    const float alpha = 1.0f;
    const float T     = 0.5f;

    int32_t seed[1] = { 0 };
    auto dist = common_draft_topk_from_ngram(mod, seed, 1, alpha, T);
    TEST_CHECK(dist.size() == 1);
    TEST_CHECK(dist[0].ids[0] == 77);

    // expected: p = (5 + 1) / (5 + K*1), log(p)/T
    const float expected_p  = (5.0f + alpha) / (5.0f + common_draft_topk_pos::K * alpha);
    const float expected_lp = std::log(expected_p) / T;
    TEST_CHECK(approx_eq(dist[0].log_probs[0], expected_lp));

    // K-1 remaining entries are empty.
    for (int k = 1; k < common_draft_topk_pos::K; ++k) {
        TEST_CHECK(dist[0].ids[k] == -1);
        TEST_CHECK(std::isinf(dist[0].log_probs[k]) && dist[0].log_probs[k] < 0);
    }

    fprintf(stdout, "  test_k1_logprob_formula: ok\n");
}

static void test_sorted_invariant() {
    constexpr size_t N    = 1;
    constexpr size_t SIZE = 8;

    common_ngram_mod mod(N, SIZE);

    // multiple successors with distinct hit counts in slot 0
    int32_t t[2] = { 0, 0 };
    auto add = [&](int32_t s, int n) { t[0] = 0; t[1] = s; for (int i = 0; i < n; ++i) mod.add(t); };
    add(10, 7);
    add(20, 3);
    add(30, 2);
    add(40, 1);

    int32_t seed[1] = { 0 };
    auto dist = common_draft_topk_from_ngram(mod, seed, 1);
    TEST_CHECK(dist.size() == 1);

    const auto & p = dist[0];

    // descending: non-empty entries in strictly non-increasing log_prob order
    for (int k = 1; k < common_draft_topk_pos::K; ++k) {
        if (p.ids[k] != -1 && p.ids[k - 1] != -1) {
            TEST_CHECK(p.log_probs[k - 1] >= p.log_probs[k]);
        }
        if (p.ids[k - 1] == -1) {
            TEST_CHECK(p.ids[k] == -1);
        }
    }

    // top-1 should be the most-hit successor
    TEST_CHECK(p.ids[0] == 10);

    fprintf(stdout, "  test_sorted_invariant: ok\n");
}

static void test_empty_slot_becomes_empty_pos() {
    constexpr size_t N    = 1;
    constexpr size_t SIZE = 8;

    common_ngram_mod mod(N, SIZE); // never call add — every slot is empty

    int32_t seed[1] = { 0 };
    auto dist = common_draft_topk_from_ngram(mod, seed, 3);
    TEST_CHECK(dist.size() == 3);
    for (const auto & p : dist) {
        for (int k = 0; k < common_draft_topk_pos::K; ++k) {
            TEST_CHECK(p.ids[k] == -1);
            TEST_CHECK(std::isinf(p.log_probs[k]) && p.log_probs[k] < 0);
        }
    }
    fprintf(stdout, "  test_empty_slot_becomes_empty_pos: ok\n");
}

static void test_chain_termination() {
    constexpr size_t N    = 1;
    constexpr size_t SIZE = 8;

    common_ngram_mod mod(N, SIZE);

    // only slot 0 has data: (0 -> 99). starting from seed=0:
    //   pos 0 hits, top-1 = 99
    //   pos 1 hashes (99) -> slot (99 % 8) = 3, empty -> empty_pos + terminate
    //   pos 2 .. all empty
    int32_t t[2] = { 0, 99 };
    for (int i = 0; i < 4; ++i) {
        mod.add(t);
    }

    int32_t seed[1] = { 0 };
    auto dist = common_draft_topk_from_ngram(mod, seed, 4);
    TEST_CHECK(dist.size() == 4);

    // position 0: populated, top-1 == 99
    TEST_CHECK(dist[0].ids[0] == 99);
    TEST_CHECK(!std::isinf(dist[0].log_probs[0]));

    // positions 1..3: all empty.
    for (int pos = 1; pos < 4; ++pos) {
        for (int k = 0; k < common_draft_topk_pos::K; ++k) {
            TEST_CHECK(dist[pos].ids[k] == -1);
            TEST_CHECK(std::isinf(dist[pos].log_probs[k]) && dist[pos].log_probs[k] < 0);
        }
    }

    fprintf(stdout, "  test_chain_termination: ok\n");
}

static void test_roundtrip_against_get_topk() {
    constexpr size_t N    = 2;
    constexpr size_t SIZE = 64;

    common_ngram_mod mod(N, SIZE);

    // build a known history. add (1, 2 -> 3) four times, (1, 2 -> 4) twice,
    // (1, 2 -> 5) once. so the slot at key (1,2) has top-K = [3, 4, 5, EMPTY].
    int32_t buf[3];
    auto add = [&](int32_t a, int32_t b, int32_t c, int n) {
        buf[0] = a; buf[1] = b; buf[2] = c;
        for (int i = 0; i < n; ++i) mod.add(buf);
    };
    add(1, 2, 3, 4);
    add(1, 2, 4, 2);
    add(1, 2, 5, 1);

    // ground truth via get_topk
    int32_t  gt_ids [common_ngram_mod::K];
    uint16_t gt_hits[common_ngram_mod::K];
    int32_t key[2] = { 1, 2 };
    const int gt_cnt = mod.get_topk(key, gt_ids, gt_hits);
    TEST_CHECK(gt_cnt == 3);
    TEST_CHECK(gt_ids[0] == 3 && gt_hits[0] == 4);
    TEST_CHECK(gt_ids[1] == 4 && gt_hits[1] == 2);
    TEST_CHECK(gt_ids[2] == 5 && gt_hits[2] == 1);

    const float alpha = 1.0f;
    const float T     = 0.5f;
    auto dist = common_draft_topk_from_ngram(mod, key, 1, alpha, T);
    TEST_CHECK(dist.size() == 1);

    // ids[] must match get_topk's order.
    TEST_CHECK(dist[0].ids[0] == gt_ids[0]);
    TEST_CHECK(dist[0].ids[1] == gt_ids[1]);
    TEST_CHECK(dist[0].ids[2] == gt_ids[2]);
    TEST_CHECK(dist[0].ids[3] == -1);

    // log_probs[] must match the formula.
    const int K = common_draft_topk_pos::K;
    const float denom = (4.0f + 2.0f + 1.0f) + K * alpha;
    const float exp0 = std::log((4.0f + alpha) / denom) / T;
    const float exp1 = std::log((2.0f + alpha) / denom) / T;
    const float exp2 = std::log((1.0f + alpha) / denom) / T;
    TEST_CHECK(approx_eq(dist[0].log_probs[0], exp0));
    TEST_CHECK(approx_eq(dist[0].log_probs[1], exp1));
    TEST_CHECK(approx_eq(dist[0].log_probs[2], exp2));
    TEST_CHECK(std::isinf(dist[0].log_probs[3]) && dist[0].log_probs[3] < 0);

    // descending invariant re-checked here for good measure
    TEST_CHECK(dist[0].log_probs[0] > dist[0].log_probs[1]);
    TEST_CHECK(dist[0].log_probs[1] > dist[0].log_probs[2]);

    fprintf(stdout, "  test_roundtrip_against_get_topk: ok\n");
}

// bit-identical K=1 check: the top-1 id from common_draft_topk_from_ngram
// must equal mod.get() for the same hash key. this guards the scaffold
// against silent behaviour drift vs the existing single-token drafter.
static void test_k1_matches_get() {
    constexpr size_t N    = 2;
    constexpr size_t SIZE = 64;

    common_ngram_mod mod(N, SIZE);

    int32_t buf[3];
    auto add = [&](int32_t a, int32_t b, int32_t c, int n) {
        buf[0] = a; buf[1] = b; buf[2] = c;
        for (int i = 0; i < n; ++i) mod.add(buf);
    };
    add(7, 8, 100, 3);
    add(7, 8, 101, 1);

    int32_t key[2] = { 7, 8 };
    const int32_t get_top = mod.get(key);
    TEST_CHECK(get_top != common_ngram_mod::EMPTY);

    auto dist = common_draft_topk_from_ngram(mod, key, 1);
    TEST_CHECK(dist.size() == 1);
    TEST_CHECK(dist[0].ids[0] == get_top);

    fprintf(stdout, "  test_k1_matches_get: ok\n");
}

//
// DDTree builder/follow/mask tests
//

// Build a fully controlled `common_draft_topk_pos` fixture manually.
// Unpopulated slots default to (id=-1, log_prob=-INF).
static common_draft_topk_pos make_pos(std::initializer_list<std::pair<int32_t, float>> entries) {
    common_draft_topk_pos p;
    for (int k = 0; k < common_draft_topk_pos::K; ++k) {
        p.ids[k]       = -1;
        p.log_probs[k] = -INFINITY;
    }
    int k = 0;
    for (const auto & e : entries) {
        if (k >= common_draft_topk_pos::K) break;
        p.ids[k]       = e.first;
        p.log_probs[k] = e.second;
        ++k;
    }
    return p;
}

static void test_build_degenerate() {
    // budget = 0 — root-only tree, no nodes.
    std::vector<common_draft_topk_pos> topk;
    topk.push_back(make_pos({ { 11, -0.1f } }));

    const common_ddtree tree = common_ddtree_build(topk, 0, true);
    TEST_CHECK(tree.n_nodes == 0);
    TEST_CHECK(tree.token_ids.empty());
    TEST_CHECK(tree.depths.empty());
    TEST_CHECK(tree.parents.size() == 1);
    TEST_CHECK(tree.parents[0] == -1);
    TEST_CHECK(tree.child_maps.size() == 1);
    TEST_CHECK(tree.child_maps[0].empty());
    TEST_CHECK(tree.visibility.size() == 1);
    TEST_CHECK(tree.visibility[0] == 1);

    // Empty topk, positive budget — still degenerate.
    const common_ddtree tree2 = common_ddtree_build({}, 4, true);
    TEST_CHECK(tree2.n_nodes == 0);
    TEST_CHECK(tree2.visibility.size() == 1);
    TEST_CHECK(tree2.visibility[0] == 1);

    fprintf(stdout, "  test_build_degenerate: ok\n");
}

static void test_build_chain() {
    // topk with only top-1 populated — chain_seed produces a pure chain.
    std::vector<common_draft_topk_pos> topk;
    for (int d = 0; d < 6; ++d) {
        topk.push_back(make_pos({ { d + 1, -0.1f } }));
    }

    const int budget = 4;
    const common_ddtree tree = common_ddtree_build(topk, budget, true);

    TEST_CHECK(tree.n_nodes == budget);
    TEST_CHECK((int) tree.token_ids.size() == budget);
    TEST_CHECK((int) tree.depths.size()    == budget);
    TEST_CHECK((int) tree.parents.size()   == budget + 1);
    TEST_CHECK(tree.parents[0] == -1);

    // chain: parents[i+1] == i, token_ids[i] == i+1, depths[i] == i+1.
    for (int i = 0; i < budget; ++i) {
        TEST_CHECK(tree.parents[i + 1] == i);
        TEST_CHECK(tree.token_ids[i]   == i + 1);
        TEST_CHECK(tree.depths[i]      == i + 1);
    }

    // Visibility: each node sees root + its chain-of-ancestors + self.
    const int N = 1 + tree.n_nodes;
    for (int i = 0; i < N; ++i) {
        // root is an ancestor of every node.
        TEST_CHECK(tree.visibility[(size_t) i * N + 0] == 1);
        // self-visibility.
        TEST_CHECK(tree.visibility[(size_t) i * N + i] == 1);
        // ancestors on the chain.
        for (int j = 1; j <= i; ++j) {
            TEST_CHECK(tree.visibility[(size_t) i * N + j] == 1);
        }
        // no forward visibility.
        for (int j = i + 1; j < N; ++j) {
            TEST_CHECK(tree.visibility[(size_t) i * N + j] == 0);
        }
    }

    fprintf(stdout, "  test_build_chain: ok\n");
}

static void test_build_branching() {
    // Two depths both with K=2 real entries — budget=6 forces the heap to
    // expand siblings AND extend the chain, giving a branching tree.
    std::vector<common_draft_topk_pos> topk;
    topk.push_back(make_pos({ { 100, -0.1f }, { 101, -1.0f } }));
    topk.push_back(make_pos({ { 200, -0.2f }, { 201, -1.5f } }));
    topk.push_back(make_pos({ { 300, -0.3f }                 }));

    const int budget = 6;
    const common_ddtree tree = common_ddtree_build(topk, budget, true);

    TEST_CHECK(tree.n_nodes > 0);
    TEST_CHECK(tree.n_nodes <= budget);

    // There must be at least one non-linear parent relationship, i.e. some
    // node i (>= 2) whose parent is NOT i-1. chain-only would give
    // parents[i+1] == i for every i.
    bool has_branch = false;
    for (int i = 0; i < tree.n_nodes; ++i) {
        if (tree.parents[i + 1] != i) {
            has_branch = true;
            break;
        }
    }
    TEST_CHECK(has_branch);

    // Ancestry sanity: every node must be visible to itself and to the root.
    const int N = 1 + tree.n_nodes;
    for (int i = 0; i < N; ++i) {
        TEST_CHECK(tree.visibility[(size_t) i * N + 0] == 1);
        TEST_CHECK(tree.visibility[(size_t) i * N + i] == 1);
    }

    // Each child_map entry must resolve to a node whose parent is this node.
    for (int i = 0; i < N; ++i) {
        for (const auto & kv : tree.child_maps[i]) {
            const int child = kv.second;
            TEST_CHECK(child > 0 && child < N);
            TEST_CHECK(tree.parents[child] == i);
            TEST_CHECK(tree.token_ids[child - 1] == kv.first);
        }
    }

    fprintf(stdout, "  test_build_branching: ok\n");
}

static void test_parent_ids_sentinel() {
    // build_chain-style tree, but use a branching tree so there are multiple
    // nodes with parent == root (produce -1 sentinels).
    std::vector<common_draft_topk_pos> topk;
    topk.push_back(make_pos({ { 10, -0.1f }, { 11, -0.2f }, { 12, -0.3f } }));
    topk.push_back(make_pos({ { 20, -0.1f }                              }));

    const common_ddtree tree = common_ddtree_build(topk, 5, true);
    TEST_CHECK(tree.n_nodes > 0);

    const auto parent_ids = common_ddtree_parent_ids(tree);
    TEST_CHECK((int) parent_ids.size() == tree.n_nodes);

    int n_root_children = 0;
    for (int i = 0; i < tree.n_nodes; ++i) {
        const int p = tree.parents[i + 1];
        TEST_CHECK(parent_ids[i] == (int32_t) (p - 1));
        if (p == 0) {
            TEST_CHECK(parent_ids[i] == -1);
            ++n_root_children;
        }
    }
    TEST_CHECK(n_root_children >= 1);

    fprintf(stdout, "  test_parent_ids_sentinel: ok\n");
}

static void test_follow_full_match() {
    // 3-node chain; posterior at each node's argmax matches the next in
    // the chain, so the walk consumes everything.
    std::vector<common_draft_topk_pos> topk;
    topk.push_back(make_pos({ { 1, -0.1f } }));
    topk.push_back(make_pos({ { 2, -0.1f } }));
    topk.push_back(make_pos({ { 3, -0.1f } }));

    const common_ddtree tree = common_ddtree_build(topk, 3, true);
    TEST_CHECK(tree.n_nodes == 3);

    // posterior[0..2]: root predicts child token, node-at-depth-1 predicts
    // child at depth-2, etc. node at depth-3 predicts a "bonus" we set to 99.
    const int32_t posterior[4] = { 1, 2, 3, 99 };

    int32_t next_tok = -123;
    const auto accepted = common_ddtree_follow_verified(tree, posterior, next_tok);

    TEST_CHECK(accepted.size() == 4);            // root + 3 matched children
    TEST_CHECK(accepted[0] == 0);
    TEST_CHECK(accepted[1] == 1);
    TEST_CHECK(accepted[2] == 2);
    TEST_CHECK(accepted[3] == 3);
    TEST_CHECK(next_tok == 99);

    fprintf(stdout, "  test_follow_full_match: ok\n");
}

static void test_follow_mismatch() {
    // 3-node chain; posterior diverges at depth 3 (root=1, node1=2, node2=7).
    std::vector<common_draft_topk_pos> topk;
    topk.push_back(make_pos({ { 1, -0.1f } }));
    topk.push_back(make_pos({ { 2, -0.1f } }));
    topk.push_back(make_pos({ { 3, -0.1f } }));

    const common_ddtree tree = common_ddtree_build(topk, 3, true);
    TEST_CHECK(tree.n_nodes == 3);

    // root accepts 1, node1 accepts 2, node2 predicts 7 which doesn't match
    // any child of node2 -> stop. next_tok must be 7.
    const int32_t posterior[4] = { 1, 2, 7, 99 };

    int32_t next_tok = -123;
    const auto accepted = common_ddtree_follow_verified(tree, posterior, next_tok);

    TEST_CHECK(accepted.size() == 3);           // root + 2 matched children
    TEST_CHECK(accepted[0] == 0);
    TEST_CHECK(accepted[1] == 1);
    TEST_CHECK(accepted[2] == 2);
    TEST_CHECK(next_tok == 7);

    fprintf(stdout, "  test_follow_mismatch: ok\n");
}

static void test_mask_shape_and_values() {
    // tiny chain tree: 3 nodes, all in a line.
    std::vector<common_draft_topk_pos> topk;
    topk.push_back(make_pos({ { 1, -0.1f } }));
    topk.push_back(make_pos({ { 2, -0.1f } }));
    topk.push_back(make_pos({ { 3, -0.1f } }));

    const common_ddtree tree = common_ddtree_build(topk, 3, true);
    TEST_CHECK(tree.n_nodes == 3);

    const int prompt_kv_start = 2;
    const int kv_total        = 5;   // [0,1] = prompt prefix, [2..4] = block
    const int n_tokens        = 3;   // queries: root is excluded — callers
                                     // typically iterate over node flat 0..n_tokens-1.
    const int kq_mask_pad     = 32;

    const auto mask = common_ddtree_build_mask(tree, prompt_kv_start, kv_total, n_tokens, kq_mask_pad);

    const int kv_pad = ((kv_total + kq_mask_pad - 1) / kq_mask_pad) * kq_mask_pad;
    const int q_pad  = ((n_tokens + 31) / 32) * 32;
    TEST_CHECK(kv_pad == 32);
    TEST_CHECK(q_pad  == 32);
    TEST_CHECK((int) mask.size() == q_pad * kv_pad);

    const int N = 1 + tree.n_nodes;

    for (int t = 0; t < n_tokens; ++t) {
        for (int k = 0; k < kv_total; ++k) {
            const uint16_t raw = mask[(size_t) t * kv_pad + k];
            const float    val = ggml_fp16_to_fp32(raw);

            if (k < prompt_kv_start) {
                TEST_CHECK(raw == 0x0000);
                TEST_CHECK(val == 0.0f);
                continue;
            }

            const int local = k - prompt_kv_start;
            const bool visible = (local < n_tokens) &&
                                 (tree.visibility[(size_t) t * N + local] != 0);
            if (visible) {
                TEST_CHECK(raw == 0x0000);
                TEST_CHECK(val == 0.0f);
            } else {
                TEST_CHECK(raw == 0xFC00);
                TEST_CHECK(std::isinf(val) && val < 0.0f);
            }
        }

        // Positions in the KV padding past kv_total are -INF.
        for (int k = kv_total; k < kv_pad; ++k) {
            TEST_CHECK(mask[(size_t) t * kv_pad + k] == 0xFC00);
        }
    }

    // Padding rows past n_tokens are entirely -INF.
    for (int t = n_tokens; t < q_pad; ++t) {
        for (int k = 0; k < kv_pad; ++k) {
            TEST_CHECK(mask[(size_t) t * kv_pad + k] == 0xFC00);
        }
    }

    fprintf(stdout, "  test_mask_shape_and_values: ok\n");
}

int main() {
    fprintf(stdout, "test-ddtree: running...\n");

    test_k1_logprob_formula();
    test_sorted_invariant();
    test_empty_slot_becomes_empty_pos();
    test_chain_termination();
    test_roundtrip_against_get_topk();
    test_k1_matches_get();

    test_build_degenerate();
    test_build_chain();
    test_build_branching();
    test_parent_ids_sentinel();
    test_follow_full_match();
    test_follow_mismatch();
    test_mask_shape_and_values();

    fprintf(stdout, "test-ddtree: all tests passed\n");
    return 0;
}
