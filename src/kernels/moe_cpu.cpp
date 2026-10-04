#include "kernels/moe_cpu.h"

#include "ggml-cpu.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

namespace e8::kernels {

namespace {

std::atomic<long long> g_t_gu{0}, g_t_dn{0}, g_calls{0};  // microseconds
long long now_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct Pair {
    int64_t t, j;
    int32_t e;
    float   w;
};

std::vector<Pair> active_pairs(const ggml_tensor * ids, const ggml_tensor * w) {
    const int64_t k = ids->ne[0], n = ids->ne[1];
    std::vector<Pair> p;
    for (int64_t t = 0; t < n; t++) {
        for (int64_t j = 0; j < k; j++) {
            const float wt = ((const float *) ((const char *) w->data + t * w->nb[1]))[j];
            if (wt != 0.0f) {
                p.push_back({ t, j, ((const int32_t *) ((const char *) ids->data + t * ids->nb[1]))[j], wt });
            }
        }
    }
    return p;
}

// Ask the OS to read the weights of every active expert now, as large concurrent I/Os, instead of letting the
// compute threads fault them in page by page. Weights already resident cost nothing.
void prefetch_experts(const std::vector<Pair> & pairs, std::initializer_list<const ggml_tensor *> ts) {
#if defined(_WIN32)
    std::vector<WIN32_MEMORY_RANGE_ENTRY> r;
    std::vector<int32_t>                  seen;
    for (const Pair & p : pairs) {
        if (std::find(seen.begin(), seen.end(), p.e) != seen.end()) continue;
        seen.push_back(p.e);
        for (const ggml_tensor * t : ts) {
            if (t) r.push_back({ (void *) ((const char *) t->data + (size_t) p.e * t->nb[2]), t->nb[2] });
        }
    }
    if (!r.empty()) PrefetchVirtualMemory(GetCurrentProcess(), r.size(), r.data(), 0);
#else
    (void) pairs;
    (void) ts;
#endif
}

// row range of this thread, in chunks of 16 rows
void split_rows(int64_t rows, int ith, int nth, int64_t & r0, int64_t & r1) {
    const int64_t per = ((rows + nth - 1) / nth + 15) / 16 * 16;
    r0                = std::min(rows, per * ith);
    r1                = std::min(rows, r0 + per);
}

// dst H [n_ff, k, n] F32; src: x, ids, w, gate, up
void gate_up_op(ggml_tensor * dst, int ith, int nth, void *) {
    const long long t_start = now_us();
    struct Timer {
        int ith; long long t0;
        ~Timer() { if (ith == 0) { g_t_gu += now_us() - t0; g_calls++; } }
    } timer{ ith, t_start };
    const ggml_tensor * x = dst->src[0], * ids = dst->src[1], * w = dst->src[2], * gate = dst->src[3], * up = dst->src[4];
    const int64_t       ne = x->ne[0], ff = gate->ne[1], n = x->ne[1];
    int64_t             r0, r1;
    split_rows(ff, ith, nth, r0, r1);
    const auto pairs = active_pairs(ids, w);
    if (ith == 0) prefetch_experts(pairs, { gate, up, (const ggml_tensor *) dst->src[5] });  // down too: next op
    if (r0 >= r1 || pairs.empty()) return;

    const auto * tg = ggml_get_type_traits_cpu(gate->type);
    const auto * tu = ggml_get_type_traits_cpu(up->type);
    // activations in each weight type's vec_dot type (quantized per token, redundantly per thread: tiny)
    auto quant = [&](ggml_type vt, std::vector<uint8_t> & buf) {
        const size_t rs = ggml_row_size(vt, ne);
        buf.resize(rs * (size_t) n);
        for (int64_t t = 0; t < n; t++) {
            const float * xt = (const float *) ((const char *) x->data + t * x->nb[1]);
            if (vt == GGML_TYPE_F32) std::memcpy(buf.data() + rs * t, xt, sizeof(float) * ne);
            else ggml_get_type_traits_cpu(vt)->from_float(xt, buf.data() + rs * t, ne);
        }
        return rs;
    };
    std::vector<uint8_t> xg, xu;
    const size_t         rg = quant(tg->vec_dot_type, xg);
    const size_t         ru = tu->vec_dot_type == tg->vec_dot_type ? rg : quant(tu->vec_dot_type, xu);
    const uint8_t *      xu_p = tu->vec_dot_type == tg->vec_dot_type ? xg.data() : xu.data();
    // rows of inactive pairs are left unwritten: the down op skips the same pairs
    for (const Pair & p : pairs) {
        float *      h  = (float *) ((char *) dst->data + p.t * dst->nb[2] + p.j * dst->nb[1]);
        const char * gb = (const char *) gate->data + (size_t) p.e * gate->nb[2];
        const char * ub = (const char *) up->data + (size_t) p.e * up->nb[2];
        for (int64_t r = r0; r < r1; r++) {
            float g = 0, u = 0;
            tg->vec_dot((int) ne, &g, 0, gb + r * gate->nb[1], 0, xg.data() + rg * p.t, 0, 1);
            tu->vec_dot((int) ne, &u, 0, ub + r * up->nb[1], 0, xu_p + ru * p.t, 0, 1);
            h[r] = g / (1.0f + std::exp(-g)) * u;
        }
    }
}

// dst y [n_embd, n] F32; src: H, ids, w, down
void down_op(ggml_tensor * dst, int ith, int nth, void *) {
    struct Timer {
        int ith; long long t0;
        ~Timer() { if (ith == 0) g_t_dn += now_us() - t0; }
    } timer{ ith, now_us() };
    const ggml_tensor * H = dst->src[0], * ids = dst->src[1], * w = dst->src[2], * down = dst->src[3];
    const int64_t       ff = H->ne[0], ne = dst->ne[0], n = dst->ne[1];
    int64_t             r0, r1;
    split_rows(ne, ith, nth, r0, r1);
    if (r0 >= r1) return;
    for (int64_t t = 0; t < n; t++) {
        float * y = (float *) ((char *) dst->data + t * dst->nb[1]);
        for (int64_t r = r0; r < r1; r++) y[r] = 0.0f;
    }
    const auto pairs = active_pairs(ids, w);
    if (pairs.empty()) return;
    const auto *         td = ggml_get_type_traits_cpu(down->type);
    const ggml_type      vt = td->vec_dot_type;
    const size_t         rs = ggml_row_size(vt, ff);
    std::vector<uint8_t> hq(rs);
    for (const Pair & p : pairs) {
        const float * h = (const float *) ((const char *) H->data + p.t * H->nb[2] + p.j * H->nb[1]);
        if (vt == GGML_TYPE_F32) std::memcpy(hq.data(), h, sizeof(float) * ff);
        else ggml_get_type_traits_cpu(vt)->from_float(h, hq.data(), ff);
        const char * db = (const char *) down->data + (size_t) p.e * down->nb[2];
        float *      y  = (float *) ((char *) dst->data + p.t * dst->nb[1]);
        for (int64_t r = r0; r < r1; r++) {
            float s = 0;
            td->vec_dot((int) ff, &s, 0, db + r * down->nb[1], 0, hq.data(), 0, 1);
            y[r] += p.w * s;
        }
    }
}

} // namespace

ggml_tensor * moe_cpu_sparse(ggml_context * ctx, ggml_tensor * x, ggml_tensor * ids, ggml_tensor * w, ggml_tensor * gate,
                             ggml_tensor * up, ggml_tensor * down) {
    if (!ggml_is_contiguous(x)) x = ggml_cont(ctx, x);
    if (!ggml_is_contiguous(ids)) ids = ggml_cont(ctx, ids);
    if (!ggml_is_contiguous(w)) w = ggml_cont(ctx, w);
    const int64_t k = ids->ne[0], n = ids->ne[1];
    ggml_tensor * a1[6] = { x, ids, w, gate, up, down };  // down only for the prefetch
    ggml_tensor * H     = ggml_custom_4d(ctx, GGML_TYPE_F32, gate->ne[1], k, n, 1, a1, 6, gate_up_op, GGML_N_TASKS_MAX, nullptr);
    ggml_tensor * a2[4] = { H, ids, w, down };
    return ggml_custom_4d(ctx, GGML_TYPE_F32, x->ne[0], n, 1, 1, a2, 4, down_op, GGML_N_TASKS_MAX, nullptr);
}

void moe_cpu_times(double & gate_up_s, double & down_s, long long & calls) {
    gate_up_s = g_t_gu / 1e6;
    down_s    = g_t_dn / 1e6;
    calls     = g_calls;
}

} // namespace e8::kernels
