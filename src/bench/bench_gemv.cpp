// Matrix-vector throughput with ggml's own kernels, shaped like Qwen3.8-27B FFN weights ([5120 x 17408]).
// N = tokens per pass: N=1 is plain decode, N=k+1 is a speculative verify pass over k drafted tokens.

#include "bench/bench.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace e8::bench {

namespace {

constexpr int64_t kK = 5120;   // hidden size
constexpr int64_t kM = 17408;  // FFN width

struct Row {
    int    n;
    double ms;
    double gbps;
};

std::vector<uint8_t> make_tile(ggml_type type, int64_t rows) {
    std::mt19937                    rng(42);
    std::normal_distribution<float> nd(0.0f, 0.02f);
    std::vector<float>              f((size_t) (kK * rows));
    for (auto & v : f) {
        v = nd(rng);
    }
    ggml_quantize_init(type);
    std::vector<uint8_t> q(ggml_row_size(type, kK) * (size_t) rows);
    ggml_quantize_chunk(type, f.data(), q.data(), 0, rows, kK, nullptr);
    return q;
}

// Weights live in `wbuft`; activations/outputs in the backend's default buffer. Returns false + err on failure.
bool suite(ggml_backend_t be, ggml_backend_buffer_type_t wbuft, ggml_type type, size_t target_bytes,
           const std::vector<int> & ns, double seconds, std::vector<Row> & out, double & total_bytes, std::string & err) {
    const size_t mat_bytes = ggml_row_size(type, kK) * (size_t) kM;
    const int    n_mats    = std::max(1, (int) (target_bytes / mat_bytes));
    total_bytes            = (double) mat_bytes * n_mats;

    ggml_init_params wp = { ggml_tensor_overhead() * (size_t) (n_mats + 1), nullptr, true };
    ggml_context *   cw = ggml_init(wp);
    std::vector<ggml_tensor *> w;
    for (int i = 0; i < n_mats; i++) {
        w.push_back(ggml_new_tensor_2d(cw, type, kK, kM));
    }
    ggml_backend_buffer_t wb = ggml_backend_alloc_ctx_tensors_from_buft(cw, wbuft);
    if (!wb) {
        err = "weight allocation failed";
        ggml_free(cw);
        return false;
    }
    {
        // Replicate one quantized tile; repacking buffers need each tensor set whole, in one call.
        const auto           tile = make_tile(type, 256);
        std::vector<uint8_t> host(mat_bytes);
        for (size_t off = 0; off < mat_bytes; off += tile.size()) {
            std::memcpy(host.data() + off, tile.data(), std::min(tile.size(), mat_bytes - off));
        }
        for (auto * t : w) {
            ggml_backend_tensor_set(t, host.data(), 0, mat_bytes);
        }
    }

    std::mt19937                    rng(7);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    bool                            ok = true;
    for (int n : ns) {
        const int        gsize = n_mats + 8;
        ggml_init_params gp    = { ggml_tensor_overhead() * (size_t) gsize + ggml_graph_overhead_custom(gsize, false),
                                   nullptr, true };
        ggml_context *   cg    = ggml_init(gp);
        ggml_tensor *    x     = ggml_new_tensor_2d(cg, GGML_TYPE_F32, kK, n);
        ggml_cgraph *    gf    = ggml_new_graph_custom(cg, gsize, false);
        for (auto * t : w) {
            ggml_build_forward_expand(gf, ggml_mul_mat(cg, t, x));
        }
        ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
        if (!ggml_gallocr_alloc_graph(ga, gf)) {
            err = "activation allocation failed";
            ok  = false;
        } else {
            std::vector<float> xd((size_t) (kK * n));
            for (auto & v : xd) {
                v = nd(rng);
            }
            ggml_backend_tensor_set(x, xd.data(), 0, ggml_nbytes(x));
            for (int i = 0; i < 2; i++) {
                ggml_backend_graph_compute(be, gf);
            }
            int          runs = 0;
            const double t0   = now_s();
            double       el   = 0;
            do {
                if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) {
                    err = "graph compute failed";
                    ok  = false;
                    break;
                }
                runs++;
                el = now_s() - t0;
            } while (el < seconds || runs < 3);
            if (ok) {
                const double ms = el / runs * 1e3;
                out.push_back({ n, ms, total_bytes / (ms / 1e3) / GB });
            }
        }
        ggml_gallocr_free(ga);
        ggml_free(cg);
        if (!ok) {
            break;
        }
    }
    ggml_backend_buffer_free(wb);
    ggml_free(cw);
    return ok;
}

void report(const char * dev, const char * label, ggml_type type, int threads, const std::vector<Row> & rows) {
    for (const auto & r : rows) {
        char th[16] = "";
        if (threads > 0) {
            snprintf(th, sizeof(th), "t=%-2d ", threads);
        }
        printf("  %-4s %-7s %-10s %sN=%-2d %8.2f ms  %6.1f GB/s  %7.2f ms/token\n", dev, ggml_type_name(type), label, th,
               r.n, r.ms, r.gbps, r.ms / r.n);
    }
    fflush(stdout);
}

ggml_backend_buffer_type_t cpu_repack_buft() {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!dev) {
        return nullptr;
    }
    auto get_extra = (ggml_backend_dev_get_extra_bufts_t) ggml_backend_reg_get_proc_address(
        ggml_backend_dev_backend_reg(dev), "ggml_backend_dev_get_extra_bufts");
    if (!get_extra) {
        return nullptr;
    }
    for (ggml_backend_buffer_type_t * b = get_extra(dev); b && *b; ++b) {
        if (std::strstr(ggml_backend_buft_name(*b), "REPACK")) {
            return *b;
        }
    }
    return nullptr;
}

} // namespace

void gemv_cpu(const Options & opt, const sys::Info & si) {
    const int threads = opt.threads > 0 ? opt.threads : std::max(1, si.physical_cores);
    size_t    target  = 2 * GiB;
    if (si.ram_avail && target > si.ram_avail / 4) {
        target = si.ram_avail / 4;
    }

    ggml_backend_t be = ggml_backend_cpu_init();
    if (!be) {
        printf("[cpu-gemv] CPU backend init failed\n");
        return;
    }
    ggml_backend_buffer_type_t def    = ggml_backend_get_default_buffer_type(be);
    ggml_backend_buffer_type_t repack = cpu_repack_buft();

    struct Case {
        ggml_type                  type;
        ggml_backend_buffer_type_t buft;
        int                        threads;
        std::vector<int>           ns;
    };

    std::vector<Case> cases = {
        { GGML_TYPE_Q4_K,   def,    threads, { 1, 9 } },
        { GGML_TYPE_IQ4_XS, def,    threads, { 1 }    },
    };
    if (repack) {
        cases.push_back({ GGML_TYPE_Q4_K, repack, threads, { 1, 9 } });
        if (si.logical_cores > threads) {
            cases.push_back({ GGML_TYPE_Q4_K, repack, si.logical_cores, { 9 } });
        }
    }

    printf("[cpu-gemv] ggml CPU kernels, weights [%lld x %lld] per matrix, ~%.2f GB per pass\n", (long long) kK,
           (long long) kM, (double) target / GB);
    for (const auto & c : cases) {
        ggml_backend_cpu_set_n_threads(be, c.threads);
        std::vector<Row> rows;
        double           total = 0;
        std::string      err;
        if (!suite(be, c.buft, c.type, target, c.ns, opt.seconds, rows, total, err)) {
            printf("  cpu  %-7s %-10s failed: %s\n", ggml_type_name(c.type), ggml_backend_buft_name(c.buft), err.c_str());
        }
        report("cpu", ggml_backend_buft_name(c.buft), c.type, c.threads, rows);
    }
    ggml_backend_free(be);
}

void gemv_gpu(const Options & opt) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!dev) {
        printf("[gpu-gemv] no GPU device (built without CUDA?) - skipped\n");
        return;
    }
    size_t free_b = 0, total_b = 0;
    ggml_backend_dev_memory(dev, &free_b, &total_b);
    printf("[gpu-gemv] %s, VRAM free %.2f / %.2f GiB\n", ggml_backend_dev_description(dev), (double) free_b / GiB,
           (double) total_b / GiB);
    size_t target = std::min<size_t>(2 * GiB, free_b / 2);
    if (target < GiB / 2) {
        printf("  skipped: less than 1 GiB VRAM free. Close programs holding VRAM (check: nvidia-smi).\n");
        return;
    }
    ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
    if (!be) {
        printf("  backend init failed\n");
        return;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);

    struct Case {
        ggml_type        type;
        std::vector<int> ns;
    };

    const std::vector<Case> cases = {
        { GGML_TYPE_IQ4_XS, { 1, 2, 3, 4, 5, 6, 7, 8, 9, 16 } },
        { GGML_TYPE_Q4_K,   { 1, 9 }              },
        { GGML_TYPE_Q8_0,   { 1 }                 },
    };
    for (const auto & c : cases) {
        std::vector<Row> rows;
        double           total = 0;
        std::string      err;
        if (!suite(be, buft, c.type, target, c.ns, opt.seconds, rows, total, err)) {
            printf("  gpu  %-7s failed: %s\n", ggml_type_name(c.type), err.c_str());
        }
        report("gpu", "", c.type, 0, rows);
    }
    ggml_backend_free(be);
}

} // namespace e8::bench
