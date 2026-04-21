// Convert an ngram-mod cache file from the user-side K=2 layout to the
// upstream K=4 layout used by common/ngram-mod.h in this branch.
//
// Input format  (NGMD v2, K=2, as written by the user's uncommitted
//                server-side code with `--spec-ngram-mod-k 2`):
//
//     char    magic[4]   = 'N','G','M','D'
//     uint32  version    = 2
//     uint32  n
//     uint32  size
//     uint32  reserved   = 0
//     uint32  K          = 2
//     int32   ids[2]     x size              (K=2 slots, 8 bytes each)
//
// Output format (NGMD v2, K=4, as accepted by common_ngram_mod::load in
//                this branch):
//
//     char    magic[4]   = 'N','G','M','D'
//     uint32  version    = 2
//     uint32  K          = 4
//     uint32  n
//     uint64  size
//     slot_t  entries    x size              (K=4, 24 bytes each)
//
// Migration strategy: K=2 slots don't carry hit counts, so we synthesise:
//   slot.ids[0..1]  = src.ids[0..1]  (preserve order: rank 0 stays rank 0)
//   slot.ids[2..3]  = -1
//   slot.hits[0]    = 2              (the rank-0 id gets a small head start)
//   slot.hits[1]    = 1              (rank-1 keeps second place)
//   slot.hits[2..3] = 0
// Synthesised hits are only weights for Laplace smoothing — real workload
// traffic will retrain them quickly. Top-1 identity is preserved.
//
// Usage:
//     ngram-mod-convert <input.bin> <output.bin>
//
// Compile standalone (not built by default):
//     g++ -O2 -std=c++17 -o ngram-mod-convert ngram-mod-convert.cpp

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static constexpr char     NGMD_MAGIC[4] = {'N','G','M','D'};
static constexpr uint32_t NGMD_VERSION  = 2;
static constexpr uint32_t SRC_K         = 2;
static constexpr uint32_t DST_K         = 4;

struct src_slot {
    int32_t ids[SRC_K];
};

struct dst_slot {
    int32_t  ids [DST_K];
    uint16_t hits[DST_K];
};

static_assert(sizeof(src_slot) == 8,  "src slot layout drift");
static_assert(sizeof(dst_slot) == 24, "dst slot layout drift");

int main(int argc, char ** argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s <input.bin> <output.bin>\n", argv[0]);
        return 2;
    }

    FILE * fi = fopen(argv[1], "rb");
    if (!fi) {
        fprintf(stderr, "error: failed to open input '%s'\n", argv[1]);
        return 1;
    }

    char magic[4];
    if (fread(magic, 1, 4, fi) != 4 || memcmp(magic, NGMD_MAGIC, 4) != 0) {
        fprintf(stderr, "error: '%s' is not an NGMD file\n", argv[1]);
        fclose(fi);
        return 1;
    }

    uint32_t version = 0;
    uint32_t n       = 0;
    uint32_t size    = 0;
    uint32_t pad     = 0;
    uint32_t k_in    = 0;
    if (fread(&version, sizeof(version), 1, fi) != 1
     || fread(&n,       sizeof(n),       1, fi) != 1
     || fread(&size,    sizeof(size),    1, fi) != 1
     || fread(&pad,     sizeof(pad),     1, fi) != 1
     || fread(&k_in,    sizeof(k_in),    1, fi) != 1) {
        fprintf(stderr, "error: failed to read input header\n");
        fclose(fi);
        return 1;
    }

    if (version != NGMD_VERSION) {
        fprintf(stderr, "error: unexpected version %u (expected %u)\n", version, NGMD_VERSION);
        fclose(fi);
        return 1;
    }
    if (k_in != SRC_K) {
        fprintf(stderr, "error: unexpected K=%u (expected %u)\n", k_in, SRC_K);
        fclose(fi);
        return 1;
    }

    fprintf(stderr, "input  : magic=NGMD version=%u n=%u size=%u K=%u\n",
            version, n, size, k_in);

    std::vector<src_slot> src(size);
    const size_t want = (size_t) size * sizeof(src_slot);
    if (fread(src.data(), 1, want, fi) != want) {
        fprintf(stderr, "error: failed to read %u src slots\n", size);
        fclose(fi);
        return 1;
    }
    fclose(fi);

    std::vector<dst_slot> dst(size);
    size_t n_non_empty = 0;
    for (uint32_t i = 0; i < size; ++i) {
        const src_slot & s = src[i];
        dst_slot & d = dst[i];
        d.ids[0] = s.ids[0];
        d.ids[1] = s.ids[1];
        d.ids[2] = -1;
        d.ids[3] = -1;
        d.hits[0] = (s.ids[0] != -1) ? 2 : 0;
        d.hits[1] = (s.ids[1] != -1) ? 1 : 0;
        d.hits[2] = 0;
        d.hits[3] = 0;
        if (s.ids[0] != -1 || s.ids[1] != -1) {
            ++n_non_empty;
        }
    }

    fprintf(stderr, "migrate: %zu non-empty slots -> K=4 destination\n", n_non_empty);

    FILE * fo = fopen(argv[2], "wb");
    if (!fo) {
        fprintf(stderr, "error: failed to open output '%s'\n", argv[2]);
        return 1;
    }

    const uint32_t version_out = NGMD_VERSION;
    const uint32_t K_out       = DST_K;
    const uint32_t n_out       = n;
    const uint64_t size_out    = size;

    if (fwrite(NGMD_MAGIC,   1, 4,                fo) != 4
     || fwrite(&version_out, sizeof(uint32_t), 1, fo) != 1
     || fwrite(&K_out,       sizeof(uint32_t), 1, fo) != 1
     || fwrite(&n_out,       sizeof(uint32_t), 1, fo) != 1
     || fwrite(&size_out,    sizeof(uint64_t), 1, fo) != 1
     || fwrite(dst.data(),   sizeof(dst_slot), dst.size(), fo) != dst.size()) {
        fprintf(stderr, "error: failed to write output\n");
        fclose(fo);
        return 1;
    }

    fclose(fo);

    fprintf(stderr, "output : %s NGMD v2 K=4 n=%u size=%llu (%.1f MiB)\n",
            argv[2], n_out, (unsigned long long) size_out,
            (double)(24 + size * sizeof(dst_slot)) / (1024.0 * 1024.0));
    return 0;
}
