// Pinned host memory and PCIe transfers, through ggml's CUDA host buffer type (cudaMallocHost)
// and ggml_backend_tensor_set/get (cudaMemcpyAsync + stream sync).

#include "bench/bench.h"
#include "io/async_reader.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

namespace e8::bench {

namespace {

enum class Dir { H2D, D2H };

double transfer_gbps(ggml_tensor * dev, void * host, size_t bytes, Dir dir, double seconds) {
    uint64_t     moved = 0;
    const double t0    = now_s();
    double       el    = 0;
    do {
        if (dir == Dir::H2D) {
            ggml_backend_tensor_set(dev, host, 0, bytes);
        } else {
            ggml_backend_tensor_get(dev, host, 0, bytes);
        }
        moved += bytes;
        el = now_s() - t0;
    } while (el < seconds);
    return (double) moved / el / GB;
}

} // namespace

void pcie(const Options & opt, const sys::Info & si) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!dev) {
        printf("[pcie] no GPU device - skipped\n");
        return;
    }
    ggml_backend_buffer_type_t host_buft = ggml_backend_dev_host_buffer_type(dev);
    if (!host_buft) {
        printf("[pcie] device has no pinned host buffer type - skipped\n");
        return;
    }

    // 1. How much pinned memory can we get, and how fast? (cudaMallocHost in 1 GiB chunks)
    double target_gib = opt.pinned_gib;
    if (si.ram_total) {
        target_gib = std::min(target_gib, (double) si.ram_total / GiB - 6.0);
    }
    std::vector<ggml_backend_buffer_t> chunks;
    double                             alloc_s = 0;
    bool                               refused = false;
    while ((double) chunks.size() < target_gib) {
        const double          t0 = now_s();
        ggml_backend_buffer_t b  = ggml_backend_buft_alloc_buffer(host_buft, GiB);
        alloc_s += now_s() - t0;
        // on failure ggml silently falls back to an ordinary (pageable) CPU buffer: detect it by type
        if (!b || ggml_backend_buffer_get_type(b) != host_buft) {
            if (b) {
                ggml_backend_buffer_free(b);
            }
            refused = true;
            break;
        }
        chunks.push_back(b);
    }
    printf("[pcie] pinned alloc: %zu GiB in %.2f s (%.0f ms/GiB)%s\n", chunks.size(), alloc_s,
           chunks.empty() ? 0.0 : alloc_s * 1e3 / (double) chunks.size(), refused ? "  <- refused beyond this" : "");
    if (chunks.empty()) {
        return;
    }
    while (chunks.size() > 1) {
        ggml_backend_buffer_free(chunks.back());
        chunks.pop_back();
    }
    void * pinned = ggml_backend_buffer_get_base(chunks[0]);
    std::memset(pinned, 3, GiB);

    // 2. A 1 GiB device tensor to copy into.
    ggml_init_params      ip  = { ggml_tensor_overhead() * 2, nullptr, true };
    ggml_context *        ctx = ggml_init(ip);
    ggml_tensor *         t   = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, (int64_t) GiB);
    ggml_backend_buffer_t db  = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_dev_buffer_type(dev));
    if (!db) {
        printf("  device allocation of 1 GiB failed (VRAM in use?) - skipped\n");
        ggml_free(ctx);
        ggml_backend_buffer_free(chunks[0]);
        return;
    }

    // 3. Copy bandwidth by transfer size. 2.5 MiB ~ one Flash-Next expert at 4-bit.
    const size_t sizes[] = { 5 * MiB / 2, 64 * MiB, GiB };
    for (size_t s : sizes) {
        const double h2d = transfer_gbps(t, pinned, s, Dir::H2D, opt.seconds);
        const double d2h = transfer_gbps(t, pinned, s, Dir::D2H, opt.seconds);
        printf("  pinned   %7.1f MiB  H2D %5.1f GB/s  D2H %5.1f GB/s\n", (double) s / MiB, h2d, d2h);
        fflush(stdout);
    }
    void * pageable = io::alloc_aligned(GiB);
    if (pageable) {
        std::memset(pageable, 5, GiB);
        printf("  pageable %7.1f MiB  H2D %5.1f GB/s\n", (double) GiB / MiB,
               transfer_gbps(t, pageable, GiB, Dir::H2D, opt.seconds));
        io::free_aligned(pageable);
    }

    // 4. CPU reads RAM while the GPU DMA-reads pinned RAM: is the total above the CPU-only figure?
    size_t cpu_bytes = 4 * GiB;
    if (si.ram_avail && cpu_bytes > si.ram_avail / 3) {
        cpu_bytes = si.ram_avail / 3 / MiB * MiB;
    }
    void * cpu_buf = io::alloc_aligned(cpu_bytes);
    if (cpu_buf) {
        std::memset(cpu_buf, 1, cpu_bytes);
        const int    nt       = std::max(1, si.physical_cores);
        const double dur      = opt.seconds * 2;
        const double cpu_solo = cpu_read_gbps(cpu_buf, cpu_bytes, nt, opt.seconds);
        const double dma_solo = transfer_gbps(t, pinned, GiB, Dir::H2D, opt.seconds);
        double       cpu_mix  = 0;
        std::thread  th([&] { cpu_mix = cpu_read_gbps(cpu_buf, cpu_bytes, nt, dur); });
        const double dma_mix = transfer_gbps(t, pinned, GiB, Dir::H2D, dur);
        th.join();
        printf("  concurrent: CPU read %.1f GB/s (solo %.1f) + H2D %.1f GB/s (solo %.1f) = %.1f GB/s from RAM\n", cpu_mix,
               cpu_solo, dma_mix, dma_solo, cpu_mix + dma_mix);
        io::free_aligned(cpu_buf);
    }

    ggml_backend_buffer_free(db);
    ggml_free(ctx);
    ggml_backend_buffer_free(chunks[0]);
}

} // namespace e8::bench
