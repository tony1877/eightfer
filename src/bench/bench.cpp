#include "bench/bench.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <cstdio>

namespace e8::bench {

static void header(const sys::Info & si) {
    printf("shoehorn %s bench  (ggml %s @ %s)\n", SHOEHORN_VERSION, ggml_version(), ggml_commit());
    printf("[system] %s | %s | %d cores / %d threads | RAM %.1f GiB (%.1f GiB free)\n", si.os.c_str(), si.cpu.c_str(),
           si.physical_cores, si.logical_cores, (double) si.ram_total / GiB, (double) si.ram_avail / GiB);
    printf("[cpu] AVX2 %d  AVX512 %d  AVX512_VNNI %d  AVX512_BF16 %d\n", ggml_cpu_has_avx2(), ggml_cpu_has_avx512(),
           ggml_cpu_has_avx512_vnni(), ggml_cpu_has_avx512_bf16());
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev    = ggml_backend_dev_get(i);
        size_t             free_b = 0, total_b = 0;
        ggml_backend_dev_memory(dev, &free_b, &total_b);
        printf("[device %zu] %s: %s, %.2f / %.2f GiB free\n", i, ggml_backend_dev_name(dev), ggml_backend_dev_description(dev),
               (double) free_b / GiB, (double) total_b / GiB);
    }
    fflush(stdout);
}

int run(const Options & opt) {
    const sys::Info si = sys::query();
    header(si);
    if (opt.cpu) {
        ram(opt, si);
        gemv_cpu(opt, si);
    }
    if (opt.gpu) {
        gemv_gpu(opt);
    }
    if (opt.pcie) {
        pcie(opt, si);
    }
    if (opt.disk) {
        disk(opt);
    }
    printf("[done]\n");
    return 0;
}

} // namespace e8::bench
