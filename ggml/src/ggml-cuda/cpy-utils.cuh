#pragma once

#include "ggml-common.h"
#include "convert.cuh"
#include "planarquant.cuh"

static __device__ __forceinline__ int best_index_int8(int n, const int8_t * val, float x) {
    if (x <= val[0]) return 0;
    if (x >= val[n-1]) return n-1;
    int ml = 0, mu = n-1;
    while (mu-ml > 1) {
        int mav = (ml+mu)/2;
        if (x < val[mav]) mu = mav; else ml = mav;
    }
    return x - val[mu-1] < val[mu] - x ? mu-1 : mu;
}

static __device__ void quantize_f32_q4_0_block(const float * __restrict__ x, block_q4_0 * __restrict__ y) {
    float amax = 0.0f;
    float vmax = 0.0f;

    for (int j = 0; j < QK4_0; ++j) {
        const float v = x[j];
        if (amax < fabsf(v)) {
            amax = fabsf(v);
            vmax = v;
        }
    }

    const float d  = vmax / -8;
    const float id = d ? 1.0f/d : 0.0f;

    y->d = d;

    for (int j = 0; j < QK4_0/2; ++j) {
        const float x0 = x[0       + j]*id;
        const float x1 = x[QK4_0/2 + j]*id;

        const uint8_t xi0 = min(15, (int8_t)(x0 + 8.5f));
        const uint8_t xi1 = min(15, (int8_t)(x1 + 8.5f));

        y->qs[j]  = xi0;
        y->qs[j] |= xi1 << 4;
    }
}

static __device__ void quantize_f32_q4_1_block(const float * __restrict__ x, block_q4_1 * __restrict__ y) {
    float vmin = FLT_MAX;
    float vmax = -FLT_MAX;

    for (int j = 0; j < QK4_1; ++j) {
        const float v = x[j];
        if (v < vmin) vmin = v;
        if (v > vmax) vmax = v;
    }

    const float d  = (vmax - vmin) / ((1 << 4) - 1);
    const float id = d ? 1.0f/d : 0.0f;

    y->dm.x = d;
    y->dm.y = vmin;

    for (int j = 0; j < QK4_1/2; ++j) {
        const float x0 = (x[0       + j] - vmin)*id;
        const float x1 = (x[QK4_1/2 + j] - vmin)*id;

        const uint8_t xi0 = min(15, (int8_t)(x0 + 0.5f));
        const uint8_t xi1 = min(15, (int8_t)(x1 + 0.5f));

        y->qs[j]  = xi0;
        y->qs[j] |= xi1 << 4;
    }
}

static __device__ void quantize_f32_q5_0_block(const float * __restrict__ x, block_q5_0 * __restrict__ y) {
    float amax = 0.0f;
    float vmax = 0.0f;

    for (int j = 0; j < QK5_0; ++j) {
        const float v = x[j];
        if (amax < fabsf(v)) {
            amax = fabsf(v);
            vmax = v;
        }
    }

    const float d  = vmax / -16;
    const float id = d ? 1.0f/d : 0.0f;

    y->d = d;

    uint32_t qh = 0;
    for (int j = 0; j < QK5_0/2; ++j) {
        const float x0 = x[0       + j]*id;
        const float x1 = x[QK5_0/2 + j]*id;

        const uint8_t xi0 = min(31, (int8_t)(x0 + 16.5f));
        const uint8_t xi1 = min(31, (int8_t)(x1 + 16.5f));

        y->qs[j]  = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_0/2);
    }
    memcpy(y->qh, &qh, sizeof(qh));
}

static __device__ void quantize_f32_q5_1_block(const float * __restrict__ x, block_q5_1 * __restrict__ y) {
    float min = x[0];
    float max = x[0];

    for (int j = 1; j < QK5_1; ++j) {
        const float v = x[j];
        min = v < min ? v : min;
        max = v > max ? v : max;
    }

    const float d  = (max - min) / 31;
    const float id = d ? 1.0f/d : 0.0f;

    y->dm.x = d;
    y->dm.y = min;

    uint32_t qh = 0;
    for (int j = 0; j < QK5_1/2; ++j) {
        const float x0 = (x[0       + j] - min)*id;
        const float x1 = (x[QK5_1/2 + j] - min)*id;

        const uint8_t xi0 = (uint8_t)(x0 + 0.5f);
        const uint8_t xi1 = (uint8_t)(x1 + 0.5f);

        y->qs[j]  = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_1/2);
    }
    memcpy(y->qh, &qh, sizeof(qh));
}

static __device__ void quantize_f32_q8_0_block(const float * __restrict__ x, block_q8_0 * __restrict__ y) {
    float amax = 0.0f; // absolute max

    for (int j = 0; j < QK8_0; j++) {
        const float v = x[j];
        amax = fmaxf(amax, fabsf(v));
    }

    const float d = amax / ((1 << 7) - 1);
    const float id = d ? 1.0f/d : 0.0f;

    y->d = d;

    for (int j = 0; j < QK8_0; ++j) {
        const float x0 = x[j]*id;
        y->qs[j] = roundf(x0);
    }
}

static __device__ void quantize_f32_iq4_nl_block(const float * __restrict__ x, block_iq4_nl * __restrict__ y) {
    float amax = 0.0f;
    float vmax = 0.0f;

    for (int j = 0; j < QK4_NL; ++j) {
        const float v = x[j];
        if (amax < fabsf(v)) {
            amax = fabsf(v);
            vmax = v;
        }
    }

    float d = vmax / kvalues_iq4nl[0];
    const float id = d ? 1.0f/d : 0.0f;

    float sumqx = 0, sumq2 = 0;
    for (int j = 0; j < QK4_NL/2; ++j) {
        const float x0 = x[0        + j]*id;
        const float x1 = x[QK4_NL/2 + j]*id;
        const uint8_t xi0 = best_index_int8(16, kvalues_iq4nl, x0);
        const uint8_t xi1 = best_index_int8(16, kvalues_iq4nl, x1);
        y->qs[j] = xi0 | (xi1 << 4);
        const float v0 = kvalues_iq4nl[xi0];
        const float v1 = kvalues_iq4nl[xi1];
        const float w0 = x[0        + j]*x[0        + j];
        const float w1 = x[QK4_NL/2 + j]*x[QK4_NL/2 + j];
        sumqx += w0*v0*x[j] + w1*v1*x[QK4_NL/2 + j];
        sumq2 += w0*v0*v0 + w1*v1*v1;
    }

    y->d = sumq2 > 0 ? sumqx/sumq2 : d;
}

// Wrapper functions for cpy.cu compatibility
static __device__ void cpy_blck_f32_q4_0(const char * cxi, char * cdsti) {
    quantize_f32_q4_0_block((const float *)cxi, (block_q4_0 *)cdsti);
}

static __device__ void cpy_blck_f32_q4_1(const char * cxi, char * cdsti) {
    quantize_f32_q4_1_block((const float *)cxi, (block_q4_1 *)cdsti);
}

static __device__ void cpy_blck_f32_q5_0(const char * cxi, char * cdsti) {
    quantize_f32_q5_0_block((const float *)cxi, (block_q5_0 *)cdsti);
}

static __device__ void cpy_blck_f32_q5_1(const char * cxi, char * cdsti) {
    quantize_f32_q5_1_block((const float *)cxi, (block_q5_1 *)cdsti);
}

static __device__ void cpy_blck_f32_q8_0(const char * cxi, char * cdsti) {
    quantize_f32_q8_0_block((const float *)cxi, (block_q8_0 *)cdsti);
}

static __device__ void cpy_blck_f32_iq4_nl(const char * cxi, char * cdsti) {
    quantize_f32_iq4_nl_block((const float *)cxi, (block_iq4_nl *)cdsti);
}

// PlanarQuant PQ4_0: quantize one block of QK_PQ floats
static __device__ void quantize_f32_pq4_0_block(const float * __restrict__ x, block_pq4_0 * __restrict__ y) {
    // Compute L2 norm
    float sum_sq = 0.0f;
    for (int j = 0; j < QK_PQ; j++) {
        sum_sq += x[j] * x[j];
    }
    float norm = sqrtf(sum_sq);
    y->d = __float2half(norm);

    float inv_norm = norm > 1e-30f ? 1.0f / norm : 0.0f;

    for (int j = 0; j < QK_PQ / 2; j++) {
        float a = x[2*j]     * inv_norm;
        float b = x[2*j + 1] * inv_norm;

        float cos_r, sin_r;
        pq4_get_rotation(j, QK_PQ, cos_r, sin_r);

        float a_rot =  cos_r * a + sin_r * b;
        float b_rot = -sin_r * a + cos_r * b;

        // Find nearest centroid (linear scan, 16 entries)
        uint8_t idx_a = 0, idx_b = 0;
        float best_a = fabsf(a_rot - pq4_centroids[0]);
        float best_b = fabsf(b_rot - pq4_centroids[0]);
        for (int i = 1; i < 16; i++) {
            float da = fabsf(a_rot - pq4_centroids[i]);
            float db = fabsf(b_rot - pq4_centroids[i]);
            if (da < best_a) { best_a = da; idx_a = i; }
            if (db < best_b) { best_b = db; idx_b = i; }
        }

        y->qs[j] = idx_a | (idx_b << 4);
    }
}

static __device__ void cpy_blck_f32_pq4_0(const char * cxi, char * cdsti) {
    quantize_f32_pq4_0_block((const float *)cxi, (block_pq4_0 *)cdsti);
}

// PlanarQuant PQ3_0: quantize one block with 3-bit packing
static __device__ void pq3_pack_device(uint8_t * qs, int idx, uint8_t val) {
    int bit_pos = idx * 3;
    int byte_pos = bit_pos / 8;
    int bit_off = bit_pos % 8;
    qs[byte_pos] = (qs[byte_pos] & ~(0x7 << bit_off)) | ((val & 0x7) << bit_off);
    if (bit_off > 5) {
        qs[byte_pos + 1] = (qs[byte_pos + 1] & ~(0x7 >> (8 - bit_off))) | ((val & 0x7) >> (8 - bit_off));
    }
}

static __device__ void quantize_f32_pq3_0_block(const float * __restrict__ x, block_pq3_0 * __restrict__ y) {
    float sum_sq = 0.0f;
    for (int j = 0; j < QK_PQ; j++) {
        sum_sq += x[j] * x[j];
    }
    float norm = sqrtf(sum_sq);
    y->d = __float2half(norm);

    float inv_norm = norm > 1e-30f ? 1.0f / norm : 0.0f;

    // Zero the packed buffer
    for (int j = 0; j < QK_PQ * 3 / 8; j++) {
        y->qs[j] = 0;
    }

    for (int j = 0; j < QK_PQ / 2; j++) {
        float a = x[2*j]     * inv_norm;
        float b = x[2*j + 1] * inv_norm;

        float cos_r, sin_r;
        pq4_get_rotation(j, QK_PQ, cos_r, sin_r);

        float a_rot =  cos_r * a + sin_r * b;
        float b_rot = -sin_r * a + cos_r * b;

        uint8_t idx_a = 0, idx_b = 0;
        float best_a = fabsf(a_rot - pq3_centroids[0]);
        float best_b = fabsf(b_rot - pq3_centroids[0]);
        for (int i = 1; i < 8; i++) {
            float da = fabsf(a_rot - pq3_centroids[i]);
            float db = fabsf(b_rot - pq3_centroids[i]);
            if (da < best_a) { best_a = da; idx_a = i; }
            if (db < best_b) { best_b = db; idx_b = i; }
        }

        pq3_pack_device(y->qs, 2*j,     idx_a);
        pq3_pack_device(y->qs, 2*j + 1, idx_b);
    }
}

static __device__ void cpy_blck_f32_pq3_0(const char * cxi, char * cdsti) {
    quantize_f32_pq3_0_block((const float *)cxi, (block_pq3_0 *)cdsti);
}

template<typename src_t, typename dst_t>
static __device__ void cpy_1_scalar(const char * cxi, char * cdsti) {
    *(dst_t *) cdsti = ggml_cuda_cast<dst_t>(*(const src_t *) cxi);
}
