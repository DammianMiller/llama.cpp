#include "ngram-mod.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include <sys/types.h>
#include <unistd.h>

// Guaranteed-runtime check (asserts get stripped in Release builds).
#define TEST_CHECK(cond)                                                    \
    do {                                                                    \
        if (!(cond)) {                                                      \
            fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, #cond);\
            std::abort();                                                   \
        }                                                                   \
    } while (0)

// replicate the private hash from common_ngram_mod::idx so we can build a
// ground-truth "slot -> {token -> count}" table in the test.
static size_t ref_idx(const int32_t * tokens, size_t n, size_t size) {
    size_t res = 0;
    for (size_t i = 0; i < n; ++i) {
        res = res*6364136223846793005ULL + tokens[i];
    }
    return res % size;
}

static std::string make_tmp_path(const char * tag) {
    const char * tmpdir = std::getenv("TMPDIR");
    if (!tmpdir || !*tmpdir) {
        tmpdir = "/tmp";
    }
    std::string p = tmpdir;
    p += "/test-ngram-mod-";
    p += tag;
    p += "-";
    p += std::to_string(static_cast<long long>(getpid()));
    p += ".bin";
    return p;
}

static void test_add_get_topk() {
    constexpr size_t N    = 3;
    constexpr size_t SIZE = 64;
    constexpr int    ITER = 1000;

    common_ngram_mod mod(N, SIZE);

    // ground-truth: map slot index -> (token -> count)
    std::unordered_map<size_t, std::map<int32_t, int>> expected;

    std::mt19937 rng(0xC0FFEEu);
    std::uniform_int_distribution<int32_t> dist_tok(0, 31);

    int32_t tokens[N + 1];

    for (int i = 0; i < ITER; ++i) {
        for (size_t k = 0; k < N + 1; ++k) {
            tokens[k] = dist_tok(rng);
        }

        mod.add(tokens);

        const size_t si = ref_idx(tokens, N, SIZE);
        expected[si][tokens[N]]++;
    }

    // Re-run RNG, for every tuple read mod.get() and assert it's a member of that
    // slot's observed-successor set. Then check that get_topk returns entries sorted
    // desc with no duplicates, and that get()'s returned id has true count >=
    // the K-th highest true count for its slot (modulo top-K eviction).
    std::mt19937 rng2(0xC0FFEEu);
    std::uniform_int_distribution<int32_t> dist_tok2(0, 31);

    for (int i = 0; i < ITER; ++i) {
        for (size_t k = 0; k < N + 1; ++k) {
            tokens[k] = dist_tok2(rng2);
        }

        const int32_t top = mod.get(tokens);
        TEST_CHECK(top != common_ngram_mod::EMPTY);

        const size_t si = ref_idx(tokens, N, SIZE);
        const auto & counts = expected[si];
        TEST_CHECK(counts.find(top) != counts.end());

        int32_t  ids [common_ngram_mod::K];
        uint16_t hits[common_ngram_mod::K];
        const int cnt = mod.get_topk(tokens, ids, hits);
        TEST_CHECK(cnt >= 1 && cnt <= common_ngram_mod::K);
        TEST_CHECK(ids[0] == top);

        // entries must be sorted by hits desc; empty entries trail.
        for (int k = 1; k < common_ngram_mod::K; ++k) {
            if (ids[k - 1] == common_ngram_mod::EMPTY) {
                TEST_CHECK(ids[k] == common_ngram_mod::EMPTY);
                TEST_CHECK(hits[k] == 0);
            } else if (ids[k] != common_ngram_mod::EMPTY) {
                TEST_CHECK(hits[k - 1] >= hits[k]);
            }
        }

        // no duplicate ids among non-empty entries
        for (int a = 0; a < common_ngram_mod::K; ++a) {
            if (ids[a] == common_ngram_mod::EMPTY) continue;
            for (int b = a + 1; b < common_ngram_mod::K; ++b) {
                if (ids[b] == common_ngram_mod::EMPTY) continue;
                TEST_CHECK(ids[a] != ids[b]);
            }
        }

        // the returned top-1 must have count >= the K-th highest true count for this slot.
        std::vector<int> all_counts;
        all_counts.reserve(counts.size());
        for (const auto & tc : counts) {
            all_counts.push_back(tc.second);
        }
        std::sort(all_counts.begin(), all_counts.end(), std::greater<int>());
        const int kth = (int)all_counts.size() >= common_ngram_mod::K
            ? all_counts[common_ngram_mod::K - 1]
            : 0;

        auto it = counts.find(top);
        TEST_CHECK(it != counts.end());
        TEST_CHECK(it->second >= kth);
    }

    fprintf(stdout, "  test_add_get_topk: ok\n");
}

static void test_save_load_roundtrip() {
    constexpr size_t N    = 3;
    constexpr size_t SIZE = 64;

    common_ngram_mod a(N, SIZE);

    std::mt19937 rng(0x12345);
    std::uniform_int_distribution<int32_t> dist_tok(0, 63);

    int32_t tokens[N + 1];
    for (int i = 0; i < 500; ++i) {
        for (size_t k = 0; k < N + 1; ++k) {
            tokens[k] = dist_tok(rng);
        }
        a.add(tokens);
    }

    const std::string path = make_tmp_path("roundtrip");
    TEST_CHECK(a.save(path));

    common_ngram_mod b(N, SIZE);
    TEST_CHECK(b.load(path));

    TEST_CHECK(a.get_used() == b.get_used());
    TEST_CHECK(a.size()     == b.size());

    // replay same RNG stream and compare top-K per tuple
    std::mt19937 rng2(0x12345);
    std::uniform_int_distribution<int32_t> dist_tok2(0, 63);
    for (int i = 0; i < 500; ++i) {
        for (size_t k = 0; k < N + 1; ++k) {
            tokens[k] = dist_tok2(rng2);
        }
        int32_t  ids_a[common_ngram_mod::K], ids_b[common_ngram_mod::K];
        uint16_t hit_a[common_ngram_mod::K], hit_b[common_ngram_mod::K];
        const int ca = a.get_topk(tokens, ids_a, hit_a);
        const int cb = b.get_topk(tokens, ids_b, hit_b);
        TEST_CHECK(ca == cb);
        for (int k = 0; k < common_ngram_mod::K; ++k) {
            TEST_CHECK(ids_a[k] == ids_b[k]);
            TEST_CHECK(hit_a[k] == hit_b[k]);
        }
    }

    std::remove(path.c_str());
    fprintf(stdout, "  test_save_load_roundtrip: ok\n");
}

static void test_load_rejects_v1() {
    const std::string path = make_tmp_path("v1forge");

    // forge a v1-ish header: magic + u32 version=1 + zero-pad body. We pad the
    // remainder so the header read succeeds and the failure is specifically
    // the version check.
    {
        std::ofstream f(path, std::ios::binary | std::ios::trunc);
        TEST_CHECK(f.is_open());
        const char magic[4] = { 'N', 'G', 'M', 'D' };
        const uint32_t v1   = 1;
        const uint32_t pad32 = 0;
        const uint64_t pad64 = 0;
        f.write(magic, sizeof(magic));
        f.write(reinterpret_cast<const char *>(&v1),    sizeof(v1));
        f.write(reinterpret_cast<const char *>(&pad32), sizeof(pad32)); // where K would be
        f.write(reinterpret_cast<const char *>(&pad32), sizeof(pad32)); // where n would be
        f.write(reinterpret_cast<const char *>(&pad64), sizeof(pad64)); // where size would be
        TEST_CHECK(f.good());
    }

    common_ngram_mod m(3, 64);
    const bool ok = m.load(path);
    TEST_CHECK(!ok);

    std::remove(path.c_str());
    fprintf(stdout, "  test_load_rejects_v1: ok\n");
}

static void test_eviction_orders_by_hits() {
    // with n=1 the hash over one token is 0*mul + t = t, so tokens[0]=0 always
    // lands in slot 0. this lets us fully control eviction behaviour.
    constexpr size_t N    = 1;
    constexpr size_t SIZE = 7;

    common_ngram_mod m(N, SIZE);

    int32_t buf[2];
    auto add = [&](int32_t succ) { buf[0] = 0; buf[1] = succ; m.add(buf); };

    // 10x3, 20x2, 30x1, 40x1 -> slot full, then 50 evicts the lowest.
    add(10); add(10); add(10);
    add(20); add(20);
    add(30);
    add(40);
    add(50);

    int32_t  ids [common_ngram_mod::K];
    uint16_t hits[common_ngram_mod::K];
    buf[0] = 0;
    const int cnt = m.get_topk(buf, ids, hits);
    TEST_CHECK(cnt == common_ngram_mod::K);

    TEST_CHECK(ids[0] == 10 && hits[0] == 3);
    TEST_CHECK(ids[1] == 20 && hits[1] == 2);

    for (int k = 1; k < common_ngram_mod::K; ++k) {
        TEST_CHECK(hits[k - 1] >= hits[k]);
    }

    fprintf(stdout, "  test_eviction_orders_by_hits: ok\n");
}

static void test_saturation() {
    constexpr size_t N    = 1;
    constexpr size_t SIZE = 7;

    common_ngram_mod m(N, SIZE);

    int32_t buf[2] = { 0, 42 };
    for (int i = 0; i < 70000; ++i) { // > 65535
        m.add(buf);
    }

    int32_t  ids [common_ngram_mod::K];
    uint16_t hits[common_ngram_mod::K];
    buf[0] = 0;
    const int cnt = m.get_topk(buf, ids, hits);
    TEST_CHECK(cnt >= 1);
    TEST_CHECK(ids[0] == 42);
    TEST_CHECK(hits[0] == std::numeric_limits<uint16_t>::max());

    fprintf(stdout, "  test_saturation: ok\n");
}

int main() {
    fprintf(stdout, "test-ngram-mod: running...\n");

    test_add_get_topk();
    test_save_load_roundtrip();
    test_load_rejects_v1();
    test_eviction_orders_by_hits();
    test_saturation();

    fprintf(stdout, "test-ngram-mod: all tests passed\n");
    return 0;
}
