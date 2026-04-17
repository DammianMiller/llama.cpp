#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include <cstddef>
#include <string>

//
// common_ngram_mod
// ref: https://github.com/ggml-org/llama.cpp/pull/19164
//

// per-slot top-K successor record.
// ids[] is kept sorted by hits[] in descending order, with empty
// entries (ids[i] == EMPTY, hits[i] == 0) trailing.
struct common_ngram_mod_slot {
    static constexpr int K = 4;

    int32_t  ids [K];
    uint16_t hits[K];
};

static_assert(common_ngram_mod_slot::K == 4, "common_ngram_mod_slot::K must be 4 for the v2 save format");
static_assert(sizeof(common_ngram_mod_slot) == 4 * sizeof(int32_t) + 4 * sizeof(uint16_t),
    "common_ngram_mod_slot must be a tightly packed POD (24 bytes)");

// basic n-gram hasher with top-K successor tracking per slot
struct common_ngram_mod {
    using entry_t = int32_t;
    using slot_t  = common_ngram_mod_slot;

    static constexpr entry_t EMPTY = -1;
    static constexpr int     K     = slot_t::K;

    common_ngram_mod(uint16_t n, size_t size);

    size_t  idx(const entry_t * tokens) const;
    void    add(const entry_t * tokens);
    entry_t get(const entry_t * tokens) const; // returns EMPTY if slot empty

    // fills out_ids/out_hits with up to K entries in descending hit order.
    // returns the count of non-EMPTY entries written (0..K).
    int get_topk(const entry_t * tokens, int32_t out_ids[K], uint16_t out_hits[K]) const;

    void reset();

    size_t get_n()    const;
    size_t get_used() const;

    size_t size()       const;
    size_t size_bytes() const;

    // NGMD v2 format round-trip — save/load raw entries + metadata to disk.
    // Returns false on I/O or validation failure. File format:
    // magic "NGMD" + u32 version + u32 n + u64 size + u32 K + raw entries.
    // Used by both --spec-ngram-mod-preload (load at startup) and
    // --spec-ngram-persist (save on shutdown; turbo b0e905ac1).
    bool save(const std::string & path) const;
    bool load(const std::string & path);

private:
    size_t n; // ngram size to hash

    size_t used;

    std::vector<slot_t> entries;
};
