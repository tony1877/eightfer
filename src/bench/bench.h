#pragma once

#include "sys/sysinfo.h"

#include <cstddef>
#include <string>
#include <vector>

namespace e8::bench {

struct Options {
    bool                     cpu  = true;
    bool                     gpu  = true;
    bool                     pcie = true;
    bool                     disk = true;
    std::vector<std::string> disks;              // directories (temp file is created) or existing files
    double                   disk_gib   = 4.0;   // temp file size per directory
    double                   pinned_gib = 16.0;  // pinned-allocation target
    int                      threads    = 0;     // CPU threads for GEMV; 0 = physical cores
    double                   seconds    = 1.5;   // duration of each measurement
};

int run(const Options & opt);

// Sections. Each prints its own report.
void ram(const Options & opt, const sys::Info & si);
void gemv_cpu(const Options & opt, const sys::Info & si);
void gemv_gpu(const Options & opt);
void pcie(const Options & opt, const sys::Info & si);
void disk(const Options & opt);

// Child-process entry for `shoehorn pinned-probe <GiB>`: pins memory in 1 GiB chunks up to the target,
// prints "PINNED <GiB> <seconds> <refused 0|1>" and exits without freeing (see bench_pcie.cpp).
[[noreturn]] void pinned_probe(double target_gib);

// Shared helpers.
double now_s();
// Streams `bytes` of `buf` with `threads` threads for about `seconds`; returns GB/s.
double cpu_read_gbps(const void * buf, size_t bytes, int threads, double seconds);

constexpr double GB  = 1e9;
constexpr size_t MiB = size_t(1) << 20;
constexpr size_t GiB = size_t(1) << 30;

} // namespace e8::bench
