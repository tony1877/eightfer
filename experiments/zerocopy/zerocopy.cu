// Zero-copy gate for the streaming residual GEMM (plan idea 1): how fast can a few SMs read pinned, mapped host memory
// over PCIe, and how much does that slow a VRAM-bound kernel (the drafter's GEMV) running beside it? Compared with
// the copy engine (cudaMemcpyAsync H2D), which is what the verify uses today.
//
//   nvcc -O3 -std=c++17 -arch=sm_120a -o zerocopy.exe zerocopy.cu
//   zerocopy.exe

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CK(x)                                                                                     \
    do {                                                                                          \
        cudaError_t e_ = (x);                                                                     \
        if (e_ != cudaSuccess) {                                                                  \
            fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_));    \
            exit(1);                                                                              \
        }                                                                                         \
    } while (0)

constexpr size_t GiB = 1ull << 30;

// Each block reads its own contiguous span (as a row-split GEMM would), threads interleaved inside it, U loads of
// 16 bytes in flight per thread.
template <int U>
__global__ void read_span(const uint4 * __restrict__ p, size_t n16, int passes, unsigned * out) {
    const size_t span = (n16 + gridDim.x - 1) / gridDim.x;
    const size_t b0   = (size_t) blockIdx.x * span;
    const size_t b1   = min(n16, b0 + span);
    unsigned     acc  = 0;
    for (int ps = 0; ps < passes; ps++) {
        size_t i = b0 + threadIdx.x;
        for (; i + (U - 1) * blockDim.x < b1; i += U * blockDim.x) {
            uint4 v[U];
#pragma unroll
            for (int u = 0; u < U; u++) v[u] = __ldcs(p + i + (size_t) u * blockDim.x);
#pragma unroll
            for (int u = 0; u < U; u++) acc ^= v[u].x ^ v[u].y ^ v[u].z ^ v[u].w;
        }
        for (; i < b1; i += blockDim.x) {
            const uint4 v = __ldcs(p + i);
            acc ^= v.x ^ v.y ^ v.z ^ v.w;
        }
    }
    if (acc == 0x9e3779b9u) out[0] = acc;  // keeps the loads alive
}

struct Timer {
    cudaEvent_t a, b;
    Timer() { CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b)); }
    void start(cudaStream_t s) { CK(cudaEventRecord(a, s)); }
    void stop(cudaStream_t s) { CK(cudaEventRecord(b, s)); }
    float ms() { float t; CK(cudaEventSynchronize(b)); CK(cudaEventElapsedTime(&t, a, b)); return t; }
};

int main() {
    int dev = 0;
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, dev));
    const int sms = prop.multiProcessorCount;
    printf("[zerocopy] %s, %d SMs\n", prop.name, sms);

    const size_t host_bytes = 1 * GiB, vram_bytes = 2 * GiB;
    void *       h = nullptr;
    CK(cudaHostAlloc(&h, host_bytes, cudaHostAllocMapped | cudaHostAllocPortable));
    for (size_t i = 0; i < host_bytes / 8; i++) ((unsigned long long *) h)[i] = i * 0x9e3779b97f4a7c15ull;
    void * hd = nullptr;
    CK(cudaHostGetDevicePointer(&hd, h, 0));
    void *     dv = nullptr, *dst = nullptr;
    unsigned * out = nullptr;
    CK(cudaMalloc(&dv, vram_bytes));
    CK(cudaMalloc(&dst, host_bytes));
    CK(cudaMalloc(&out, 4));
    CK(cudaMemset(dv, 1, vram_bytes));

    cudaStream_t s1, s2;
    CK(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking));
    CK(cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking));
    Timer t1, t2;

    const size_t hn16 = host_bytes / 16, vn16 = vram_bytes / 16;

    // copy engine reference
    for (int w = 0; w < 2; w++) CK(cudaMemcpyAsync(dst, h, host_bytes, cudaMemcpyHostToDevice, s1));
    t1.start(s1);
    for (int r = 0; r < 3; r++) CK(cudaMemcpyAsync(dst, h, host_bytes, cudaMemcpyHostToDevice, s1));
    t1.stop(s1);
    const double ce = 3.0 * host_bytes / (t1.ms() * 1e-3) / 1e9;
    printf("  copy engine H2D 1 GiB             %6.1f GB/s\n", ce);

    // VRAM read alone (stand-in for the drafter's memory-bound GEMV), as vblocks spans: 8 per SM (3 MB each) or 8192
    // (256 KB each, closer to a GEMV's many row blocks)
    int  vblocks     = sms * 8;
    auto vram_launch = [&](cudaStream_t s) { read_span<4><<<vblocks, 256, 0, s>>>((const uint4 *) dv, vn16, 1, out); };
    const int vreps  = 20;
    float     vram_ms;
    double    vram_gbs;
    auto      vram_alone = [&]() {
        for (int w = 0; w < 3; w++) vram_launch(s2);
        t2.start(s2);
        for (int r = 0; r < vreps; r++) vram_launch(s2);
        t2.stop(s2);
        vram_ms  = t2.ms() / vreps;
        vram_gbs = vram_bytes / (vram_ms * 1e-3) / 1e9;
    };
    vram_alone();
    printf("  VRAM read, all SMs                %6.1f GB/s (%.2f ms per 2 GiB)\n", vram_gbs, vram_ms);

    // zero-copy read alone, by grid size and threads per block
    printf("  zero-copy read of mapped pinned RAM, alone:\n");
    printf("    blocks  threads  loads/thr   GB/s\n");
    struct Best { int g = 0, t = 0; double gbs = 0; };
    std::vector<int> grids = {1, 2, 4, 8, 12, 16, 24, 32, 48, sms, sms * 2, sms * 4};
    std::vector<std::pair<int, double>> best_by_g;
    for (int g : grids) {
        double best = 0;
        for (int thr : {256, 512, 1024}) {
            for (int u : {4, 8}) {
                auto launch = [&](int passes) {
                    if (u == 4) read_span<4><<<g, thr, 0, s1>>>((const uint4 *) hd, hn16, passes, out);
                    else read_span<8><<<g, thr, 0, s1>>>((const uint4 *) hd, hn16, passes, out);
                };
                launch(1);
                t1.start(s1);
                launch(2);
                t1.stop(s1);
                const double gbs = 2.0 * host_bytes / (t1.ms() * 1e-3) / 1e9;
                printf("    %6d  %7d  %9d  %6.1f\n", g, thr, u, gbs);
                best = std::max(best, gbs);
            }
        }
        best_by_g.push_back({g, best});
    }
    CK(cudaGetLastError());

    // zero-copy beside the VRAM reader: zero-copy launched first on G blocks, the VRAM reader fills the rest of the
    // GPU for about as long
    for (int vb : {sms * 8, 8192}) {
        vblocks = vb;
        vram_alone();
        for (int zthr : {1024, 256}) {
            printf("  together (zero-copy %d thr x 8 loads on G blocks; VRAM reader %d blocks, alone %.1f GB/s):\n",
                   zthr, vb, vram_gbs);
            printf("    blocks   zc GB/s  (alone)   VRAM GB/s   slowdown\n");
            for (auto [g, alone] : best_by_g) {
                if (g > 48) continue;
                const int    zpasses     = 3;
                const double zc_ms_alone = zpasses * host_bytes / (alone * 1e9) * 1e3;
                const int    n_v         = std::max(4, (int) (zc_ms_alone / vram_ms));
                t1.start(s1);
                read_span<8><<<g, zthr, 0, s1>>>((const uint4 *) hd, hn16, zpasses, out);
                t1.stop(s1);
                t2.start(s2);
                for (int r = 0; r < n_v; r++) vram_launch(s2);
                t2.stop(s2);
                const double zc = zpasses * host_bytes / (t1.ms() * 1e-3) / 1e9;
                const double vr = (double) n_v * vram_bytes / (t2.ms() * 1e-3) / 1e9;
                printf("    %6d  %8.1f  (%5.1f)   %9.1f   %7.1f%%\n", g, zc, alone, vr, 100.0 * (1 - vr / vram_gbs));
            }
        }
    }

    // what the verify does today: copy engine beside the VRAM reader (8192 blocks)
    {
        const int n_v = std::max(4, (int) (3.0 * host_bytes / (ce * 1e9) * 1e3 / vram_ms));
        t1.start(s1);
        for (int r = 0; r < 3; r++) CK(cudaMemcpyAsync(dst, h, host_bytes, cudaMemcpyHostToDevice, s1));
        t1.stop(s1);
        t2.start(s2);
        for (int r = 0; r < n_v; r++) vram_launch(s2);
        t2.stop(s2);
        const double c  = 3.0 * host_bytes / (t1.ms() * 1e-3) / 1e9;
        const double vr = (double) n_v * vram_bytes / (t2.ms() * 1e-3) / 1e9;
        printf("    copy engine %6.1f  (%5.1f)   %9.1f   %7.1f%%\n", c, ce, vr, 100.0 * (1 - vr / vram_gbs));
    }
    CK(cudaDeviceSynchronize());
    printf("[done]\n");
    return 0;
}
