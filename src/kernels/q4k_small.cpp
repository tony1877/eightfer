#include "kernels/q4k_small.h"

#include "ggml-cpu.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#if defined(__AVX512F__) || defined(_M_X64) || defined(__x86_64__)
#include <immintrin.h>
#define E8_X86 1
#endif

namespace e8::kernels {

namespace {

constexpr int QK = 256;

// ggml's block_q4_K / block_q8_K (ggml-common.h)
struct BlockQ4K {
    uint16_t d, dmin;  // fp16
    uint8_t  scales[12];
    uint8_t  qs[QK / 2];
};
static_assert(sizeof(BlockQ4K) == 144, "q4_K layout");

struct BlockQ8K {
    float   d;
    int8_t  qs[QK];
    int16_t bsums[QK / 16];
};
static_assert(sizeof(BlockQ8K) == 292, "q8_K layout");

// activations, re-laid out per 256 block: sub-blocks of 32 in the order [0,2 | 1,3 | 4,6 | 5,7] so that a 64-byte
// load of Q4_K nibbles (low: sub-blocks 2j, 2j+2; high: 2j+1, 2j+3) lines up with 64 contiguous bytes here
struct alignas(64) BlockQ8P {
    int8_t  qs[QK];
    float   d;
    int32_t bs[8];  // sum of each 32-element sub-block, natural order
    uint8_t pad[64 - 4 - 32];
};
static_assert(sizeof(BlockQ8P) == 320, "q8 permuted layout");

constexpr int kPermSrc[8] = { 0, 2, 1, 3, 4, 6, 5, 7 };

inline void scale_min(int j, const uint8_t * q, int & sc, int & m) {
    if (j < 4) {
        sc = q[j] & 63;
        m  = q[j + 4] & 63;
    } else {
        sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m  = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

// dst: [nblk * ncols] BlockQ8P (as I8 bytes); src[0] = x [K, ncols] F32
void quantize_op(ggml_tensor * dst, int ith, int nth, void *) {
    const ggml_tensor * x     = dst->src[0];
    const int64_t      K     = x->ne[0];
    const int64_t      ncols = x->ne[1];
    const int64_t      nblk  = K / QK;
    const int64_t      total = nblk * ncols;
    const auto *       tq    = ggml_get_type_traits_cpu(GGML_TYPE_Q8_K);
    BlockQ8P *         out   = (BlockQ8P *) dst->data;
    for (int64_t i = ith; i < total; i += nth) {
        const int64_t c = i / nblk, b = i % nblk;
        const float * src = (const float *) ((const char *) x->data + c * x->nb[1]) + b * QK;
        BlockQ8K      q;
        tq->from_float(src, &q, QK);
        BlockQ8P & o = out[c * nblk + b];
        o.d          = q.d;
        for (int s = 0; s < 8; s++) {
            std::memcpy(o.qs + s * 32, q.qs + kPermSrc[s] * 32, 32);
            o.bs[s] = q.bsums[2 * s] + q.bsums[2 * s + 1];
        }
    }
}

void gemm_rows_ref(const ggml_tensor * w, const BlockQ8P * xq, int64_t ncols, int64_t r0, int64_t r1, float * y,
                   int64_t ldy) {
    const int64_t nblk = w->ne[0] / QK;
    for (int64_t r = r0; r < r1; r++) {
        const BlockQ4K * row = (const BlockQ4K *) ((const char *) w->data + r * w->nb[1]);
        for (int64_t c = 0; c < ncols; c++) {
            double acc = 0;
            for (int64_t b = 0; b < nblk; b++) {
                const BlockQ4K & B = row[b];
                const BlockQ8P & X = xq[c * nblk + b];
                int sc[8], m[8];
                for (int j = 0; j < 8; j++) scale_min(j, B.scales, sc[j], m[j]);
                int64_t sumi = 0, summ = 0;
                for (int s = 0; s < 8; s++) {
                    // permuted position s holds natural sub-block kPermSrc[s]
                    const int sb = kPermSrc[s];
                    int       dot = 0;
                    for (int l = 0; l < 32; l++) {
                        const uint8_t byte = B.qs[(sb / 2) * 32 + l];
                        const int     q4   = (sb & 1) ? (byte >> 4) : (byte & 0xF);
                        dot += q4 * X.qs[s * 32 + l];
                    }
                    sumi += (int64_t) sc[sb] * dot;
                }
                for (int j = 0; j < 8; j++) summ += (int64_t) m[j] * X.bs[j];
                acc += (double) ggml_fp16_to_fp32(B.d) * X.d * sumi - (double) ggml_fp16_to_fp32(B.dmin) * X.d * summ;
            }
            y[c * ldy + r] = (float) acc;
        }
    }
}

#if defined(E8_X86)
#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx512f,avx512bw,avx512vl")))
#endif
void gemm_rows_vnni(const ggml_tensor * w, const BlockQ8P * xq, int64_t ncols, int64_t r0, int64_t r1, float * y,
                    int64_t ldy) {
    const int64_t nblk = w->ne[0] / QK;
    const __m512i m4   = _mm512_set1_epi8(0x0F);

    // one Q4_K block unpacked: nibbles (u8) in BlockQ8P's permuted sub-block order, int16 scale vectors for madd
    // (16 lanes per sub-block), mins, d and dmin
    struct Unpacked {
        __m512i q[4], s[4];
        __m256i m;
        float   d, dmin;
    };
    auto unpack = [&](const BlockQ4K & B, Unpacked & u) {
        int sc[8], mn[8];
        for (int j = 0; j < 8; j++) scale_min(j, B.scales, sc[j], mn[j]);
        const __m512i v0 = _mm512_loadu_si512((const void *) B.qs);
        const __m512i v1 = _mm512_loadu_si512((const void *) (B.qs + 64));
        u.q[0]           = _mm512_and_si512(v0, m4);                         // sub-blocks 0, 2
        u.q[1]           = _mm512_and_si512(_mm512_srli_epi16(v0, 4), m4);  // 1, 3
        u.q[2]           = _mm512_and_si512(v1, m4);                         // 4, 6
        u.q[3]           = _mm512_and_si512(_mm512_srli_epi16(v1, 4), m4);  // 5, 7
        static constexpr int pair[4][2] = { { 0, 2 }, { 1, 3 }, { 4, 6 }, { 5, 7 } };
        for (int i = 0; i < 4; i++) {
            u.s[i] = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_set1_epi16((short) sc[pair[i][0]])),
                                        _mm256_set1_epi16((short) sc[pair[i][1]]), 1);
        }
        u.m    = _mm256_loadu_si256((const __m256i *) mn);
        u.d    = ggml_fp16_to_fp32(B.d);
        u.dmin = ggml_fp16_to_fp32(B.dmin);
    };
    // sum over the block of scale_j * q4 * q8, as 16 int32 lanes: maddubs (u8 x s8 -> int16 pairs, max 3810) then
    // madd with the int16 scales (-> int32), so scaling costs nothing extra
    auto dot = [](const Unpacked & u, const __m512i * x) {
        __m512i s = _mm512_madd_epi16(_mm512_maddubs_epi16(u.q[0], x[0]), u.s[0]);
        s         = _mm512_add_epi32(s, _mm512_madd_epi16(_mm512_maddubs_epi16(u.q[1], x[1]), u.s[1]));
        s         = _mm512_add_epi32(s, _mm512_madd_epi16(_mm512_maddubs_epi16(u.q[2], x[2]), u.s[2]));
        s         = _mm512_add_epi32(s, _mm512_madd_epi16(_mm512_maddubs_epi16(u.q[3], x[3]), u.s[3]));
        return s;
    };
    auto hsum = [](__m512 a, __m256 m) {
        const __m256 lo = _mm512_castps512_ps256(a);
        const __m256 hi = _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(a), 1));
        __m256       t  = _mm256_sub_ps(_mm256_add_ps(lo, hi), m);
        __m128       q  = _mm_add_ps(_mm256_castps256_ps128(t), _mm256_extractf128_ps(t, 1));
        q               = _mm_add_ps(q, _mm_movehl_ps(q, q));
        q               = _mm_add_ss(q, _mm_movehdup_ps(q));
        return _mm_cvtss_f32(q);
    };

    // two rows at a time: each activation load serves both
    for (int64_t r = r0; r < r1; r += 2) {
        const bool       two = r + 1 < r1;
        const BlockQ4K * ra  = (const BlockQ4K *) ((const char *) w->data + r * w->nb[1]);
        const BlockQ4K * rb  = two ? (const BlockQ4K *) ((const char *) w->data + (r + 1) * w->nb[1]) : ra;
        __m512           fa[kQ4kMaxCols], fb[kQ4kMaxCols];
        __m256           ma[kQ4kMaxCols], mb[kQ4kMaxCols];
        for (int64_t c = 0; c < ncols; c++) {
            fa[c] = fb[c] = _mm512_setzero_ps();
            ma[c] = mb[c] = _mm256_setzero_ps();
        }
        for (int64_t b = 0; b < nblk; b++) {
            if (b + 2 < nblk) {
                _mm_prefetch((const char *) &ra[b + 2], _MM_HINT_T0);
                _mm_prefetch((const char *) &rb[b + 2], _MM_HINT_T0);
            }
            Unpacked ua, ub;
            unpack(ra[b], ua);
            unpack(rb[b], ub);
            for (int64_t c = 0; c < ncols; c++) {
                const BlockQ8P & X    = xq[c * nblk + b];
                const __m512i    x[4] = { _mm512_loadu_si512((const void *) X.qs), _mm512_loadu_si512((const void *) (X.qs + 64)),
                                          _mm512_loadu_si512((const void *) (X.qs + 128)), _mm512_loadu_si512((const void *) (X.qs + 192)) };
                const __m256i    bs   = _mm256_loadu_si256((const __m256i *) X.bs);
                fa[c] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(dot(ua, x)), _mm512_set1_ps(ua.d * X.d), fa[c]);
                ma[c] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_mullo_epi32(ua.m, bs)), _mm256_set1_ps(ua.dmin * X.d), ma[c]);
                fb[c] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(dot(ub, x)), _mm512_set1_ps(ub.d * X.d), fb[c]);
                mb[c] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_mullo_epi32(ub.m, bs)), _mm256_set1_ps(ub.dmin * X.d), mb[c]);
            }
        }
        for (int64_t c = 0; c < ncols; c++) {
            y[c * ldy + r] = hsum(fa[c], ma[c]);
            if (two) y[c * ldy + r + 1] = hsum(fb[c], mb[c]);
        }
    }
}
#endif

// dst: y [N, ncols] F32; src[0] = w (Q4_K [K, N]), src[1] = quantized activations
void gemm_op(ggml_tensor * dst, int ith, int nth, void *) {
    const ggml_tensor * w     = dst->src[0];
    const ggml_tensor * xq    = dst->src[1];
    const int64_t      N     = w->ne[1];
    const int64_t      ncols = dst->ne[1];
    // chunks of 16 rows keep each thread's share contiguous and the split even
    const int64_t      per   = ((N + nth - 1) / nth + 15) / 16 * 16;
    const int64_t      r0    = std::min(N, per * ith), r1 = std::min(N, r0 + per);
    float *            y     = (float *) dst->data;
    const int64_t      ldy   = dst->nb[1] / sizeof(float);
#if defined(E8_X86)
    static const bool vnni = ggml_cpu_has_avx512_vnni() != 0 && std::getenv("E8_SMALL_REF") == nullptr;
    if (vnni) {
        gemm_rows_vnni(w, (const BlockQ8P *) xq->data, ncols, r0, r1, y, ldy);
        return;
    }
#endif
    gemm_rows_ref(w, (const BlockQ8P *) xq->data, ncols, r0, r1, y, ldy);
}

} // namespace

bool q4k_small_supported(const ggml_tensor * w, const ggml_tensor * x) {
    return w->type == GGML_TYPE_Q4_K && x->type == GGML_TYPE_F32 && w->ne[0] % QK == 0 && x->ne[0] == w->ne[0] &&
           x->ne[1] >= 2 && x->ne[1] <= kQ4kMaxCols && x->ne[2] == 1 && x->ne[3] == 1 && ggml_is_contiguous(w);
}

ggml_tensor * q4k_mul_mat_small(ggml_context * ctx, ggml_tensor * w, ggml_tensor * x) {
    if (!ggml_is_contiguous(x)) {
        x = ggml_cont(ctx, x);
    }
    const int64_t ncols = x->ne[1], nblk = w->ne[0] / QK;
    ggml_tensor * qargs[1] = { x };
    ggml_tensor * xq = ggml_custom_4d(ctx, GGML_TYPE_I8, (int64_t) sizeof(BlockQ8P) * nblk * ncols, 1, 1, 1, qargs, 1,
                                      quantize_op, GGML_N_TASKS_MAX, nullptr);
    ggml_tensor * gargs[2] = { w, xq };
    return ggml_custom_4d(ctx, GGML_TYPE_F32, w->ne[1], ncols, 1, 1, gargs, 2, gemm_op, GGML_N_TASKS_MAX, nullptr);
}

} // namespace e8::kernels
