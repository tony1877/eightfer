// kprof: per-kernel GPU time for any CUDA program, through CUPTI activity tracing (no Nsight needed).
// Build:  cl /O2 /LD /EHsc kprof.cpp /I"%CUDA_PATH%\include" /I"%CUDA_PATH%\extras\CUPTI\include"
//         /link /LIBPATH:"%CUDA_PATH%\extras\CUPTI\lib64" /LIBPATH:"%CUDA_PATH%\lib\x64" cupti.lib cuda.lib
// Run:    set CUDA_INJECTION64_PATH=<path>\kprof.dll  (and KPROF_OUT=<file>, default kprof.txt), then the program.
// Every 3 s rewrites, per kernel name: launches, total / mean GPU time, and the share of all kernel time; plus the GPU
// span and how much of it no kernel was running (launch gaps).

#include <cupti.h>
#include <windows.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace {
struct Agg {
    unsigned long long n = 0, ns = 0;
};
std::mutex                                         mu;
std::map<std::string, Agg>                         agg;
std::vector<std::pair<unsigned long long, unsigned long long>> spans;  // kernel [start, end)

void CUPTIAPI buf_req(uint8_t ** buf, size_t * size, size_t * max_records) {
    *size        = 8 << 20;
    *buf         = (uint8_t *) _aligned_malloc(*size, 8);
    *max_records = 0;
}

void CUPTIAPI buf_done(CUcontext, uint32_t, uint8_t * buf, size_t, size_t valid) {
    CUpti_Activity * rec = nullptr;
    std::lock_guard<std::mutex> lk(mu);
    while (cuptiActivityGetNextRecord(buf, valid, &rec) == CUPTI_SUCCESS) {
        if (rec->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL || rec->kind == CUPTI_ACTIVITY_KIND_KERNEL) {
            auto *      k    = (CUpti_ActivityKernel9 *) rec;
            std::string name = k->name ? k->name : "?";
            if (name.size() > 90) name = name.substr(0, 90);
            Agg & a = agg[name];
            a.n++;
            a.ns += k->end - k->start;
            spans.push_back({ k->start, k->end });
        }
    }
    _aligned_free(buf);
}

void report() {
    cuptiActivityFlushAll(1);
    std::lock_guard<std::mutex> lk(mu);
    const char * out = getenv("KPROF_OUT") ? getenv("KPROF_OUT") : "kprof.txt";
    FILE *       f   = fopen(out, "w");
    if (!f) return;
    unsigned long long tot = 0, launches = 0;
    for (auto & [k, a] : agg) tot += a.ns, launches += a.n;
    std::vector<std::pair<std::string, Agg>> v(agg.begin(), agg.end());
    std::sort(v.begin(), v.end(), [](auto & x, auto & y) { return x.second.ns > y.second.ns; });
    // busy time = union of kernel intervals (kernels on several streams can overlap)
    std::sort(spans.begin(), spans.end());
    unsigned long long busy = 0, cs = 0, ce = 0;
    for (auto [s, e] : spans) {
        if (s > ce) { busy += ce - cs; cs = s; ce = e; }
        else ce = std::max(ce, e);
    }
    busy += ce - cs;
    const unsigned long long span = spans.empty() ? 0 : ce - spans.front().first;
    fprintf(f, "kernels: %llu launches, %.1f ms total kernel time, GPU busy %.1f of %.1f ms span\n", launches, tot / 1e6,
            busy / 1e6, span / 1e6);
    fprintf(f, "%10s %12s %10s %7s  %s\n", "launches", "total ms", "mean us", "share", "kernel");
    for (auto & [k, a] : v)
        fprintf(f, "%10llu %12.2f %10.2f %6.2f%%  %s\n", a.n, a.ns / 1e6, a.ns / 1e3 / a.n, 100.0 * a.ns / tot, k.c_str());
    fclose(f);
}
} // namespace

extern "C" __declspec(dllexport) int InitializeInjection(void) {
    cuptiActivityRegisterCallbacks(buf_req, buf_done);
    const CUptiResult r = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
    fprintf(stderr, "kprof: tracing kernels (%d)\n", (int) r);
    // rewrite the report every 3 s: exit-time hooks are unreliable here (CUDA teardown, loader lock)
    CreateThread(nullptr, 0, [](LPVOID) -> DWORD { for (;;) { Sleep(3000); report(); } }, nullptr, 0, nullptr);
    return 1;
}

