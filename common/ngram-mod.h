#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include <cstddef>

//
// common_ngram_mod
// ref: https://github.com/ggml-org/llama.cpp/pull/19164
//

// basic n-gram hasher
struct common_ngram_mod {
    using entry_t = int32_t;

    static constexpr entry_t EMPTY = -1;

    common_ngram_mod(uint16_t n, size_t size);

    size_t  idx(const entry_t * tokens) const;
    void    add(const entry_t * tokens);
    entry_t get(const entry_t * tokens) const; // return -1 if not found

    void reset();

    size_t get_n()    const;
    size_t get_used() const;

    size_t size()       const;
    size_t size_bytes() const;

    // persistence: save/load raw entries + metadata to disk
    // returns true on success. File format: magic "NGMD" + u32 n + u64 size + raw entries.
    bool save(const std::string & path) const;
    bool load(const std::string & path);

private:
    size_t n; // ngram size to hash

    size_t used;

    std::vector<entry_t> entries;
};
