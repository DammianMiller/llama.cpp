#pragma once

#include "common.cuh"

// PlanarQuant rotation lookup: direct table access (O(1) per pair)
// Tables pq_rot_cos[] and pq_rot_sin[] are defined in ggml-common.h
// and compiled into CUDA device memory via GGML_TABLE_BEGIN.

static __device__ __forceinline__ void pq4_get_rotation(int pair_idx, int dim, float & cos_out, float & sin_out) {
    (void)dim; // tables are precomputed for QK_PQ=128
    cos_out = pq_rot_cos[pair_idx];
    sin_out = pq_rot_sin[pair_idx];
}
