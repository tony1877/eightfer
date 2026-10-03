#include "bench/bench.h"
#include "io/async_reader.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <set>
#include <thread>

namespace e8::bench {

static std::atomic<uint64_t> g_sink{ 0 };

double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static uint64_t xor_words(const uint64_t * p, size_t n) {
    uint64_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    size_t   i  = 0;
    for (; i + 4 <= n; i += 4) {
        a0 ^= p[i];
        a1 ^= p[i + 1];
        a2 ^= p[i + 2];
        a3 ^= p[i + 3];
    }
    for (; i < n; i++) {
        a0 ^= p[i];
    }
    return a0 ^ a1 ^ a2 ^ a3;
}

double cpu_read_gbps(const void * buf, size_t bytes, int threads, double seconds) {
    const size_t             words = bytes / 8;
    std::vector<uint64_t>    done((size_t) threads, 0);
    std::vector<std::thread> th;
    const double             t0       = now_s();
    const double             deadline = t0 + seconds;
    for (int t = 0; t < threads; t++) {
        th.emplace_back([&, t] {
            sys::set_thread_high_perf();
            const size_t     w0  = words * (size_t) t / (size_t) threads;
            const size_t     w1  = words * (size_t) (t + 1) / (size_t) threads;
            const uint64_t * p   = static_cast<const uint64_t *>(buf) + w0;
            uint64_t         acc = 0;
            uint64_t         n   = 0;
            do {
                acc ^= xor_words(p, w1 - w0);
                n += (w1 - w0) * 8;
            } while (now_s() < deadline);
            done[(size_t) t] = n;
            g_sink += acc;
        });
    }
    for (auto & x : th) {
        x.join();
    }
    const double el    = now_s() - t0;
    uint64_t     total = 0;
    for (auto d : done) {
        total += d;
    }
    return (double) total / el / GB;
}

void ram(const Options & opt, const sys::Info & si) {
    size_t bytes = 4 * GiB;
    if (si.ram_avail && bytes > si.ram_avail / 3) {
        bytes = si.ram_avail / 3 / MiB * MiB;
    }
    void * buf = io::alloc_aligned(bytes);
    if (!buf) {
        printf("[ram] allocation of %.1f GiB failed\n", (double) bytes / GiB);
        return;
    }
    std::memset(buf, 1, bytes);

    std::set<int> counts = { 1, 2, 4, si.physical_cores, si.logical_cores };
    printf("[ram] read bandwidth, %.1f GiB buffer\n", (double) bytes / GiB);
    for (int nt : counts) {
        if (nt < 1 || nt > si.logical_cores) {
            continue;
        }
        printf("  %2d threads  %6.1f GB/s\n", nt, cpu_read_gbps(buf, bytes, nt, opt.seconds));
        fflush(stdout);
    }
    io::free_aligned(buf);
}

} // namespace e8::bench
