// Pinned host memory and PCIe transfers, through ggml's CUDA host buffer type (cudaMallocHost)
// and ggml_backend_tensor_set/get (cudaMemcpyAsync + stream sync).

#include "bench/bench.h"
#include "io/async_reader.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <climits>
#include <unistd.h>
#endif

namespace e8::bench {

namespace {

ggml_backend_buffer_type_t pinned_buft() {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    return dev ? ggml_backend_dev_host_buffer_type(dev) : nullptr;
}

// Pinned memory in 1 GiB chunks; on refusal ggml silently falls back to an ordinary (pageable)
// CPU buffer, detected by its type. Returns false when the allocation was refused.
bool alloc_pinned_gib(ggml_backend_buffer_type_t buft, ggml_backend_buffer_t & out) {
    out = ggml_backend_buft_alloc_buffer(buft, GiB);
    if (!out || ggml_backend_buffer_get_type(out) != buft) {
        if (out) {
            ggml_backend_buffer_free(out);
        }
        out = nullptr;
        return false;
    }
    return true;
}

std::string self_exe() {
#if defined(_WIN32)
    wchar_t w[MAX_PATH * 4];
    const DWORD n = GetModuleFileNameW(nullptr, w, (DWORD) (sizeof(w) / sizeof(w[0])));
    const int   u = WideCharToMultiByte(CP_UTF8, 0, w, (int) n, nullptr, 0, nullptr, nullptr);
    std::string s((size_t) u, '\0');
    WideCharToMultiByte(CP_UTF8, 0, w, (int) n, s.data(), u, nullptr, nullptr);
    return s;
#else
    char          p[PATH_MAX];
    const ssize_t n = readlink("/proc/self/exe", p, sizeof(p) - 1);
    return n > 0 ? std::string(p, (size_t) n) : std::string();
#endif
}

struct ProbeResult {
    bool   ok      = false;
    double gib     = 0;
    double seconds = 0;
    bool   refused = false;
    int    status  = 0;
};

// Runs `eightfer pinned-probe <target>` and parses its PINNED line.
ProbeResult run_probe(double target_gib) {
    ProbeResult       r;
    const std::string exe = self_exe();
    if (exe.empty()) {
        return r;
    }
    char cmd[4096];
#if defined(_WIN32)
    // cmd.exe strips the outer quotes of a /c line, hence the extra pair
    snprintf(cmd, sizeof(cmd), "\"\"%s\" pinned-probe %.1f\"", exe.c_str(), target_gib);
    const int   wn = MultiByteToWideChar(CP_UTF8, 0, cmd, -1, nullptr, 0);
    std::wstring wcmd((size_t) wn, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, cmd, -1, wcmd.data(), wn);
    FILE * p = _wpopen(wcmd.c_str(), L"r");
#else
    snprintf(cmd, sizeof(cmd), "'%s' pinned-probe %.1f 2>/dev/null", exe.c_str(), target_gib);
    FILE * p = popen(cmd, "r");
#endif
    if (!p) {
        return r;
    }
    char line[256];
    while (fgets(line, sizeof(line), p)) {
        int refused = 0;
        if (sscanf(line, "PINNED %lf %lf %d", &r.gib, &r.seconds, &refused) == 3) {
            r.ok      = true;
            r.refused = refused != 0;
        }
    }
#if defined(_WIN32)
    r.status = _pclose(p);
#else
    r.status = pclose(p);
#endif
    return r;
}

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
#if defined(_WIN32)
        // WDDM caps the page-locked memory the GPU may use ("shared GPU memory") at half of system RAM:
        // on 31.2 GiB the 16th GiB was refused.
        const double wddm_cap = (double) si.ram_total / GiB / 2.0;
        target_gib            = std::min(target_gib, wddm_cap);
        printf("[pcie] Windows pinned-memory limit: half of RAM = %.1f GiB\n", wddm_cap);
#else
        target_gib = std::min(target_gib, (double) si.ram_total / GiB - 6.0);
#endif
    }
    if (si.ram_avail) {
        // leave the OS and everything else running at least 4 GiB
        target_gib = std::min(target_gib, (double) si.ram_avail / GiB - 4.0);
    }
    // The capacity probe runs in a child process. With ~15 GiB pinned on Windows (31 GiB RAM), freeing
    // the chunks failed - cudaFreeHost returned "out of memory" and ggml aborted the whole bench - even
    // when no allocation had been refused. The child reports and exits without freeing; the OS reclaims
    // its pinned memory. The transfer tests below then use one 1 GiB chunk, which frees cleanly.
    const ProbeResult probe = run_probe(target_gib);
    if (probe.ok) {
        printf("[pcie] pinned alloc: %.0f GiB in %.2f s (%.0f ms/GiB)%s\n", probe.gib, probe.seconds,
               probe.gib > 0 ? probe.seconds * 1e3 / probe.gib : 0.0, probe.refused ? "  <- refused beyond this" : "");
    } else {
        printf("[pcie] pinned-capacity probe failed (child exit status %d)\n", probe.status);
    }

    ggml_backend_buffer_t chunk = nullptr;
    if (!alloc_pinned_gib(host_buft, chunk)) {
        printf("[pcie] cannot pin 1 GiB for the transfer tests - skipped\n");
        return;
    }
    std::vector<ggml_backend_buffer_t> chunks = { chunk };
    void *                             pinned = ggml_backend_buffer_get_base(chunks[0]);
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

void pinned_probe(double target_gib) {
    ggml_backend_buffer_type_t         buft = pinned_buft();
    std::vector<ggml_backend_buffer_t> chunks;
    double                             secs    = 0;
    bool                               refused = false;
    while (buft && (double) (chunks.size() + 1) <= target_gib) {
        const double          t0 = now_s();
        ggml_backend_buffer_t b  = nullptr;
        const bool            ok = alloc_pinned_gib(buft, b);
        secs += now_s() - t0;
        if (!ok) {
            refused = true;
            break;
        }
        chunks.push_back(b);
    }
    printf("PINNED %zu %.3f %d\n", chunks.size(), secs, refused ? 1 : 0);
    fflush(stdout);
    // Exit without freeing: freeing near the pinned limit is what aborted the bench.
#if defined(_WIN32)
    TerminateProcess(GetCurrentProcess(), 0);
#endif
    _exit(0);
}

} // namespace e8::bench
