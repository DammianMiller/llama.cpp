#include "ngram-mod.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>

//
// common_ngram_mod
//

static constexpr char     NGMD_MAGIC[4] = { 'N', 'G', 'M', 'D' };
static constexpr uint32_t NGMD_VERSION  = 2;

static inline uint16_t saturating_inc_u16(uint16_t v) {
    return v == std::numeric_limits<uint16_t>::max() ? v : static_cast<uint16_t>(v + 1);
}

// in-place insertion sort over the K=4 element slot, sorted by hits desc.
// empty entries (hits==0, ids==EMPTY) trail naturally since 0 sorts last.
// stable w.r.t. insertion order, which gives us "ties broken by last position"
// when we evict: the newest write sits at the lowest-hit tail.
static inline void slot_sort(common_ngram_mod_slot & s) {
    for (int i = 1; i < common_ngram_mod_slot::K; ++i) {
        const int32_t  id_i = s.ids [i];
        const uint16_t h_i  = s.hits[i];
        int j = i - 1;
        while (j >= 0 && s.hits[j] < h_i) {
            s.ids [j + 1] = s.ids [j];
            s.hits[j + 1] = s.hits[j];
            --j;
        }
        s.ids [j + 1] = id_i;
        s.hits[j + 1] = h_i;
    }
}

common_ngram_mod::common_ngram_mod(uint16_t n, size_t size) : n(n), used(0) {
    entries.resize(size);

    reset();
}

size_t common_ngram_mod::idx(const entry_t * tokens) const {
    size_t res = 0;

    for (size_t i = 0; i < n; ++i) {
        res = res*6364136223846793005ULL + tokens[i];
    }

    res = res % entries.size();

    return res;
}

void common_ngram_mod::add(const entry_t * tokens) {
    const size_t  i    = idx(tokens);
    const int32_t succ = tokens[n];

    slot_t & s = entries[i];

    // detect fully-empty slot (top entry is EMPTY -> whole slot is EMPTY by invariant)
    const bool was_empty = (s.ids[0] == EMPTY);

    // find an existing candidate
    int found = -1;
    for (int k = 0; k < slot_t::K; ++k) {
        if (s.ids[k] == succ) {
            found = k;
            break;
        }
    }

    if (found >= 0) {
        s.hits[found] = saturating_inc_u16(s.hits[found]);
    } else {
        // evict tail slot: by invariant, the last position has the lowest hits.
        // ties broken by last position -> overwrite index K-1.
        s.ids [slot_t::K - 1] = succ;
        s.hits[slot_t::K - 1] = 1;
    }

    slot_sort(s);

    if (was_empty) {
        used++;
    }
}

common_ngram_mod::entry_t common_ngram_mod::get(const entry_t * tokens) const {
    const size_t i = idx(tokens);

    return entries[i].ids[0];
}

int common_ngram_mod::get_topk(const entry_t * tokens, int32_t out_ids[K], uint16_t out_hits[K]) const {
    const size_t i = idx(tokens);
    const slot_t & s = entries[i];

    int count = 0;
    for (int k = 0; k < slot_t::K; ++k) {
        out_ids [k] = s.ids [k];
        out_hits[k] = s.hits[k];
        if (s.ids[k] != EMPTY) {
            count++;
        }
    }

    return count;
}

void common_ngram_mod::reset() {
    for (slot_t & s : entries) {
        for (int k = 0; k < slot_t::K; ++k) {
            s.ids [k] = EMPTY;
            s.hits[k] = 0;
        }
    }
    used = 0;
}

size_t common_ngram_mod::get_n() const {
    return n;
}

size_t common_ngram_mod::get_used() const {
    return used;
}

size_t common_ngram_mod::size() const {
    return entries.size();
}

size_t common_ngram_mod::size_bytes() const {
    return entries.size() * sizeof(slot_t);
}

bool common_ngram_mod::save(const std::string & path) const {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) {
        fprintf(stderr, "common_ngram_mod: save: could not open '%s' for writing\n", path.c_str());
        return false;
    }

    const uint32_t version = NGMD_VERSION;
    const uint32_t k       = static_cast<uint32_t>(slot_t::K);
    const uint32_t n_u32   = static_cast<uint32_t>(n);
    const uint64_t size    = static_cast<uint64_t>(entries.size());

    f.write(NGMD_MAGIC, sizeof(NGMD_MAGIC));
    f.write(reinterpret_cast<const char *>(&version), sizeof(version));
    f.write(reinterpret_cast<const char *>(&k),       sizeof(k));
    f.write(reinterpret_cast<const char *>(&n_u32),   sizeof(n_u32));
    f.write(reinterpret_cast<const char *>(&size),    sizeof(size));
    f.write(reinterpret_cast<const char *>(entries.data()), entries.size() * sizeof(slot_t));

    if (!f) {
        fprintf(stderr, "common_ngram_mod: save: write to '%s' failed\n", path.c_str());
        return false;
    }

    return true;
}

bool common_ngram_mod::load(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        fprintf(stderr, "common_ngram_mod: load: could not open '%s' for reading\n", path.c_str());
        return false;
    }

    char     magic[4]  = {};
    uint32_t version   = 0;
    uint32_t k         = 0;
    uint32_t n_u32     = 0;
    uint64_t size      = 0;

    f.read(magic, sizeof(magic));
    f.read(reinterpret_cast<char *>(&version), sizeof(version));
    f.read(reinterpret_cast<char *>(&k),       sizeof(k));
    f.read(reinterpret_cast<char *>(&n_u32),   sizeof(n_u32));
    f.read(reinterpret_cast<char *>(&size),    sizeof(size));

    if (!f) {
        fprintf(stderr, "common_ngram_mod: load: short header read from '%s'\n", path.c_str());
        return false;
    }

    if (std::memcmp(magic, NGMD_MAGIC, sizeof(NGMD_MAGIC)) != 0) {
        fprintf(stderr, "common_ngram_mod: load: bad magic in '%s' - rebuild the ngram cache\n", path.c_str());
        return false;
    }

    if (version != NGMD_VERSION) {
        fprintf(stderr, "common_ngram_mod: load: unsupported version %u in '%s' (expected %u) - rebuild the ngram cache\n",
                version, path.c_str(), NGMD_VERSION);
        return false;
    }

    if (k != static_cast<uint32_t>(slot_t::K)) {
        fprintf(stderr, "common_ngram_mod: load: K mismatch in '%s' (got %u, expected %d) - rebuild the ngram cache\n",
                path.c_str(), k, slot_t::K);
        return false;
    }

    if (n_u32 != static_cast<uint32_t>(n)) {
        fprintf(stderr, "common_ngram_mod: load: n mismatch in '%s' (got %u, expected %zu) - rebuild the ngram cache\n",
                path.c_str(), n_u32, n);
        return false;
    }

    if (size != static_cast<uint64_t>(entries.size())) {
        fprintf(stderr, "common_ngram_mod: load: size mismatch in '%s' (got %llu, expected %zu) - rebuild the ngram cache\n",
                path.c_str(), static_cast<unsigned long long>(size), entries.size());
        return false;
    }

    f.read(reinterpret_cast<char *>(entries.data()), entries.size() * sizeof(slot_t));
    if (!f) {
        fprintf(stderr, "common_ngram_mod: load: short body read from '%s'\n", path.c_str());
        return false;
    }

    // recompute used to keep it consistent with loaded contents
    used = 0;
    for (const slot_t & s : entries) {
        if (s.ids[0] != EMPTY) {
            used++;
        }
    }

    return true;
}


// NOTE: turbo's b0e905ac1 added a legacy K=2 (NGMD v1) save/load pair here.
// Dropped after cherry-pick conflict: the hybrid-persist NGMD v2 K=4 save/load
// above is the canonical round-trip path and is what --spec-ngram-mod-preload
// and --spec-ngram-persist now share. The old K=2 user-side dumps must be
// migrated via tools/ngram-mod-convert before use.
