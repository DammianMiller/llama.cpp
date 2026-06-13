#include "ngram-mod.h"

#include <cstdio>
#include <cstring>

//
// common_ngram_mod
//

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
    const size_t i = idx(tokens);

    if (entries[i] == EMPTY) {
        used++;
    }

    entries[i] = tokens[n];
}

common_ngram_mod::entry_t common_ngram_mod::get(const entry_t * tokens) const {
    const size_t i = idx(tokens);

    return entries[i];
}

void common_ngram_mod::reset() {
    std::fill(entries.begin(), entries.end(), EMPTY);
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
    return entries.size() * sizeof(entries[0]);
}


// File format:
//   char    magic[4] = 'N','G','M','D'
//   uint32  version  = 1
//   uint32  n
//   uint64  size (number of entries)
//   entry_t entries[size]
static constexpr char     NGMD_MAGIC[4] = {'N','G','M','D'};
static constexpr uint32_t NGMD_VERSION  = 1;

bool common_ngram_mod::save(const std::string & path) const {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) return false;
    bool ok = true;
    ok &= fwrite(NGMD_MAGIC, 1, 4, f) == 4;
    uint32_t ver = NGMD_VERSION;
    ok &= fwrite(&ver, sizeof(ver), 1, f) == 1;
    uint32_t n32 = (uint32_t) n;
    ok &= fwrite(&n32, sizeof(n32), 1, f) == 1;
    uint64_t sz = (uint64_t) entries.size();
    ok &= fwrite(&sz, sizeof(sz), 1, f) == 1;
    ok &= fwrite(entries.data(), sizeof(entry_t), entries.size(), f) == entries.size();
    fclose(f);
    return ok;
}

bool common_ngram_mod::load(const std::string & path) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) return false;
    char magic[4];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, NGMD_MAGIC, 4) != 0) { fclose(f); return false; }
    uint32_t ver = 0;
    if (fread(&ver, sizeof(ver), 1, f) != 1 || ver != NGMD_VERSION) { fclose(f); return false; }
    uint32_t n32 = 0;
    if (fread(&n32, sizeof(n32), 1, f) != 1 || n32 != (uint32_t) n) { fclose(f); return false; }
    uint64_t sz = 0;
    if (fread(&sz, sizeof(sz), 1, f) != 1 || sz != (uint64_t) entries.size()) { fclose(f); return false; }
    if (fread(entries.data(), sizeof(entry_t), entries.size(), f) != entries.size()) { fclose(f); return false; }
    fclose(f);
    // recount used (entries != EMPTY)
    used = 0;
    for (auto v : entries) if (v != EMPTY) used++;
    return true;
}
