#include "ddtree.h"
#include "ngram-mod.h"

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

int main() {
    fprintf(stdout, "test-ddtree: running...\n");

    test_k1_logprob_formula();
    test_sorted_invariant();
    test_empty_slot_becomes_empty_pos();
    test_chain_termination();
    test_roundtrip_against_get_topk();
    test_k1_matches_get();

    fprintf(stdout, "test-ddtree: all tests passed\n");
    return 0;
}
