// Measures weight-space error of direct quants vs nested (base + quantized residual) on real BF16 tensors.
#include "ggml.h"
#include "ggml-cpu.h"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

static std::vector<float> load_bf16(const char * path, size_t n) {
    std::vector<uint16_t> raw(n);
    FILE * f = fopen(path, "rb");
    if (!f || fread(raw.data(), 2, n, f) != n) { fprintf(stderr, "read fail %s\n", path); exit(1); }
    fclose(f);
    std::vector<float> out(n);
    for (size_t i = 0; i < n; i++) { uint32_t u = (uint32_t)raw[i] << 16; memcpy(&out[i], &u, 4); }
    return out;
}

// quantize src (rows x cols) to type, then dequantize into dst; returns bits per weight
static double qdq(ggml_type t, const std::vector<float> & src, std::vector<float> & dst, int64_t rows, int64_t cols) {
    ggml_quantize_init(t);
    const size_t rs = ggml_row_size(t, cols);
    std::vector<uint8_t> q(rs * rows);
    dst.resize(src.size());
    const int nth = 4;
    std::vector<std::thread> th;
    for (int it = 0; it < nth; it++) {
        th.emplace_back([&, it] {
            int64_t r0 = rows * it / nth, r1 = rows * (it + 1) / nth;
            ggml_quantize_chunk(t, src.data(), q.data(), r0 * cols, r1 - r0, cols, nullptr);
            const auto * tr = ggml_get_type_traits(t);
            for (int64_t r = r0; r < r1; r++) tr->to_float(q.data() + r * rs, dst.data() + r * cols, cols);
        });
    }
    for (auto & x : th) x.join();
    return 8.0 * rs / cols;
}

static double rel_rmse(const std::vector<float> & a, const std::vector<float> & b) {
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size(); i++) { double d = (double)a[i] - b[i]; num += d * d; den += (double)a[i] * a[i]; }
    return std::sqrt(num / den);
}

int main(int argc, char ** argv) {
    if (argc != 4) { fprintf(stderr, "usage: %s file.bin rows cols\n", argv[0]); return 1; }
    const int64_t rows = atoll(argv[2]), cols = atoll(argv[3]);
    auto w = load_bf16(argv[1], (size_t)(rows * cols));
    printf("%s  [%lld x %lld]\n", argv[1], (long long)rows, (long long)cols);

    std::vector<float> d;
    double e_q8 = 0;
    for (ggml_type t : {GGML_TYPE_Q8_0, GGML_TYPE_Q6_K, GGML_TYPE_Q5_K, GGML_TYPE_Q4_K, GGML_TYPE_IQ4_XS}) {
        double bpw = qdq(t, w, d, rows, cols);
        double e = rel_rmse(w, d);
        if (t == GGML_TYPE_Q8_0) e_q8 = e;
        printf("  direct %-8s %5.2f bpw  rel_rmse %.5f  (%.2fx Q8_0)\n", ggml_type_name(t), bpw, e, e / e_q8);
    }
    for (ggml_type tb : {GGML_TYPE_IQ4_XS, GGML_TYPE_Q4_K}) {
        std::vector<float> base;
        double bpw_b = qdq(tb, w, base, rows, cols);
        std::vector<float> res(w.size());
        for (size_t i = 0; i < w.size(); i++) res[i] = w[i] - base[i];
        for (ggml_type tr : {GGML_TYPE_Q3_K, GGML_TYPE_Q4_0, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K}) {
            std::vector<float> rq;
            double bpw_r = qdq(tr, res, rq, rows, cols);
            for (size_t i = 0; i < w.size(); i++) rq[i] += base[i];
            double e = rel_rmse(w, rq);
            printf("  nested %-6s + %-5s %5.2f + %4.2f = %5.2f bpw  rel_rmse %.5f  (%.2fx Q8_0)\n",
                   ggml_type_name(tb), ggml_type_name(tr), bpw_b, bpw_r, bpw_b + bpw_r, e, e / e_q8);
        }
    }
    return 0;
}
