// Unbuffered read throughput per drive, in the access patterns the engine will use:
//   sequential 4 MiB          upper bound
//   random 2.5 MiB, QD 1..64  one Flash-Next expert at 4-bit per read (expert streaming)
//   random 4 KiB              n-gram embedding rows (latency / IOPS)
// plus all drives at once, to see whether they share a bus (e.g. the chipset uplink).

#include "bench/bench.h"
#include "io/async_reader.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0A00
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace e8::bench {

namespace {

constexpr size_t kWriteChunk = 8 * MiB;

void fill_random(void * buf, size_t bytes) {
    uint64_t   s = 0x9E3779B97F4A7C15ull;
    uint64_t * p = static_cast<uint64_t *>(buf);
    for (size_t i = 0; i < bytes / 8; i++) {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        p[i] = s;
    }
}

// Make every 4 KiB page of the chunk unique so the drive can't deduplicate or compress it away.
void vary(void * buf, size_t bytes, uint64_t chunk) {
    uint64_t * p = static_cast<uint64_t *>(buf);
    for (size_t i = 0; i < bytes / 8; i += 512) {
        p[i] ^= (chunk + 1) * 0x9E3779B97F4A7C15ull;
    }
}

// Writes `bytes` (multiple of kWriteChunk) of pseudo-random data, bypassing the page cache. Returns GB/s, <0 on error.
double create_test_file(const std::string & path, uint64_t bytes, std::string & err) {
    void * buf = io::alloc_aligned(kWriteChunk);
    if (!buf) {
        err = "out of memory";
        return -1;
    }
    fill_random(buf, kWriteChunk);
    const double t0      = now_s();
    uint64_t     written = 0;
#if defined(_WIN32)
    const int    wn = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
    std::wstring wpath((size_t) wn, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, wpath.data(), wn);
    HANDLE h = CreateFileW(wpath.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS,
                           FILE_ATTRIBUTE_NORMAL | FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
    if (h == INVALID_HANDLE_VALUE) {
        err = io::os_error_string((int) GetLastError());
        io::free_aligned(buf);
        return -1;
    }
    for (uint64_t c = 0; written < bytes; c++) {
        vary(buf, kWriteChunk, c);
        DWORD w = 0;
        if (!WriteFile(h, buf, (DWORD) kWriteChunk, &w, nullptr) || w != kWriteChunk) {
            err = io::os_error_string((int) GetLastError());
            break;
        }
        written += kWriteChunk;
    }
    FlushFileBuffers(h);
    CloseHandle(h);
#else
    int fd = -1;
#ifdef O_DIRECT
    fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_DIRECT | O_CLOEXEC, 0644);
#endif
    if (fd < 0) {
        fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    }
    if (fd < 0) {
        err = io::os_error_string(errno);
        io::free_aligned(buf);
        return -1;
    }
    for (uint64_t c = 0; written < bytes; c++) {
        vary(buf, kWriteChunk, c);
        if (::write(fd, buf, kWriteChunk) != (ssize_t) kWriteChunk) {
            err = io::os_error_string(errno);
            break;
        }
        written += kWriteChunk;
    }
    fsync(fd);
    ::close(fd);
#endif
    io::free_aligned(buf);
    if (written < bytes) {
        return -1;
    }
    return (double) bytes / (now_s() - t0) / GB;
}

struct Stats {
    double                secs       = 0;
    uint64_t              bytes      = 0;
    uint64_t              ops        = 0;
    uint64_t              errors     = 0;
    double                lat_sum_us = 0;
    bool                  buffered   = false;
    std::vector<uint64_t> per_file;
    std::string           first_error;

    double gbps() const { return secs > 0 ? (double) bytes / secs / GB : 0; }

    double iops() const { return secs > 0 ? (double) ops / secs : 0; }

    double lat_us() const { return ops ? lat_sum_us / (double) ops : 0; }
};

// Keeps `qd_per_file` reads of `len` bytes in flight on every file for `seconds`.
Stats run_reads(const std::vector<std::string> & files, uint32_t len, int qd_per_file, bool sequential, double seconds) {
    Stats     st;
    const int nf = (int) files.size();
    const int qd = nf * qd_per_file;
    auto      rd = io::AsyncReader::create(qd);

    std::vector<int> fid((size_t) nf);
    uint32_t         align = 4096;
    for (int i = 0; i < nf; i++) {
        std::string err;
        fid[(size_t) i] = rd->open(files[(size_t) i], err);
        if (fid[(size_t) i] < 0) {
            st.errors      = 1;
            st.first_error = files[(size_t) i] + ": " + err;
            return st;
        }
        align       = std::max(align, rd->alignment(fid[(size_t) i]));
        st.buffered = st.buffered || !rd->unbuffered(fid[(size_t) i]);
    }
    len = (len + align - 1) / align * align;
    for (int i = 0; i < nf; i++) {
        if (rd->size(fid[(size_t) i]) < (uint64_t) len * 2) {
            st.errors      = 1;
            st.first_error = files[(size_t) i] + ": file too small";
            return st;
        }
    }

    std::vector<void *> bufs((size_t) qd, nullptr);
    for (auto & b : bufs) {
        b = io::alloc_aligned(len);
    }
    std::vector<uint64_t> next((size_t) nf, 0);
    std::vector<double>   t_sub((size_t) qd, 0);
    std::mt19937_64       rng(1234);
    st.per_file.assign((size_t) nf, 0);

    auto issue = [&](int slot) {
        const int      fi   = slot % nf;
        const uint64_t span = rd->size(fid[(size_t) fi]) / align * align;
        uint64_t       off  = 0;
        if (sequential) {
            off = next[(size_t) fi];
            next[(size_t) fi] += len;
            if (next[(size_t) fi] + len > span) {
                next[(size_t) fi] = 0;
            }
        } else {
            off = rng() % ((span - len) / align + 1) * align;
        }
        t_sub[(size_t) slot] = now_s();
        rd->submit(fid[(size_t) fi], off, len, bufs[(size_t) slot], (uint64_t) slot);
    };

    const double t0       = now_s();
    const double deadline = t0 + seconds;
    for (int s = 0; s < qd; s++) {
        issue(s);
    }
    std::vector<io::Completion> comp((size_t) qd);
    double                      t_end = t0;
    while (rd->inflight() > 0) {
        const int n = rd->wait(comp.data(), qd);
        if (n <= 0) {
            st.errors++;
            st.first_error = "wait() failed";
            break;
        }
        const double t = now_s();
        for (int i = 0; i < n; i++) {
            const int slot = (int) comp[(size_t) i].tag;
            if (comp[(size_t) i].error) {
                if (st.errors++ == 0) {
                    st.first_error = io::os_error_string(comp[(size_t) i].error);
                }
            } else {
                st.bytes += comp[(size_t) i].bytes;
                st.per_file[(size_t) (slot % nf)] += comp[(size_t) i].bytes;
                st.ops++;
                st.lat_sum_us += (t - t_sub[(size_t) slot]) * 1e6;
            }
            if (t < deadline && st.errors == 0) {
                issue(slot);
            }
        }
        t_end = t;
    }
    st.secs = t_end - t0;
    rd.reset();
    for (auto * b : bufs) {
        io::free_aligned(b);
    }
    return st;
}

struct Target {
    std::string label;
    std::string device;
    std::string file;
    std::string made_dir;  // directory we created (removed afterwards)
    bool        made_file  = false;
    double      write_gbps = 0;
};

bool check(const Stats & s) {
    if (s.errors) {
        printf("failed: %s\n", s.first_error.c_str());
        return false;
    }
    return true;
}

void test_target(const Target & t, double seconds) {
    printf("  %s  %s", t.label.c_str(), t.device.empty() ? "" : t.device.c_str());
    if (t.made_file) {
        printf("  (test file written at %.2f GB/s)", t.write_gbps);
    }
    printf("\n");

    printf("    seq 4 MiB      QD8   ");
    Stats s = run_reads({ t.file }, 4 * MiB, 8, true, seconds);
    if (!check(s)) {
        return;
    }
    printf("%6.2f GB/s%s\n", s.gbps(), s.buffered ? "  (BUFFERED: O_DIRECT unsupported, numbers include page cache)" : "");

    printf("    rand 2.5 MiB  ");
    for (int qd : { 1, 4, 16, 64 }) {
        s = run_reads({ t.file }, 5 * MiB / 2, qd, false, seconds);
        if (!check(s)) {
            return;
        }
        printf(" QD%-2d %5.2f GB/s", qd, s.gbps());
        if (qd == 1) {
            printf(" (%.2f ms) ", s.lat_us() / 1e3);
        }
        fflush(stdout);
    }
    printf("\n");

    printf("    rand 4 KiB    ");
    for (int qd : { 1, 32 }) {
        s = run_reads({ t.file }, 4096, qd, false, seconds);
        if (!check(s)) {
            return;
        }
        printf(" QD%-2d %7.1fk IOPS (%.0f us)", qd, s.iops() / 1e3, s.lat_us());
        fflush(stdout);
    }
    printf("\n");
}

} // namespace

void disk(const Options & opt) {
    std::vector<std::string> inputs = opt.disks;
    const bool               autodetect = inputs.empty();
    if (autodetect) {
        inputs = sys::fixed_drives();
        if (inputs.empty()) {
            printf("[disk] no --disk given - skipped\n");
            return;
        }
    }
    const uint64_t file_bytes = std::max<uint64_t>(1, (uint64_t) (opt.disk_gib * (double) GiB) / kWriteChunk) * kWriteChunk;
    printf("[disk] unbuffered reads, %.1f s per test%s\n", opt.seconds,
           autodetect ? ", all fixed drives (use --disk to choose)" : "");

    std::vector<Target> targets;
    for (const auto & in : inputs) {
        Target t;
        t.label  = in;
        t.device = sys::describe_drive(in);
        if (!sys::is_directory(in)) {
            t.file = in;  // existing file, read-only
            targets.push_back(t);
            continue;
        }
        // Drive roots (C:\ especially) let users create folders but not files, so always use a subfolder.
        const std::string dir = sys::join_path(in, "eightfer_bench_tmp");
        if (!sys::make_dir(dir)) {
            printf("  %s  skipped: cannot create %s\n", in.c_str(), dir.c_str());
            continue;
        }
        t.made_dir = dir;
        const int64_t free_b = sys::free_space(dir);
        if (free_b >= 0 && (uint64_t) free_b < file_bytes + 8 * (uint64_t) GiB) {
            printf("  %s  skipped: %.1f GiB free, need %.1f GiB + 8 GiB margin\n", in.c_str(), (double) free_b / GiB,
                   (double) file_bytes / GiB);
            if (!t.made_dir.empty()) {
                sys::remove_dir(t.made_dir);
            }
            continue;
        }
        t.file = sys::join_path(dir, "eightfer_bench.tmp");
        printf("  %s  writing %.1f GiB test file...\n", in.c_str(), (double) file_bytes / GiB);
        fflush(stdout);
        std::string err;
        t.write_gbps = create_test_file(t.file, file_bytes, err);
        if (t.write_gbps < 0) {
            printf("  %s  skipped: cannot write test file: %s\n", in.c_str(), err.c_str());
            sys::remove_file(t.file);
            if (!t.made_dir.empty()) {
                sys::remove_dir(t.made_dir);
            }
            continue;
        }
        t.made_file = true;
        targets.push_back(t);
    }

    for (const auto & t : targets) {
        test_target(t, opt.seconds);
    }

    if (targets.size() > 1) {
        std::vector<std::string> files;
        for (const auto & t : targets) {
            files.push_back(t.file);
        }
        printf("  all drives at once, rand 2.5 MiB QD32 each: ");
        const Stats s = run_reads(files, 5 * MiB / 2, 32, false, opt.seconds * 2);
        if (check(s)) {
            printf("%.2f GB/s total  (", s.gbps());
            for (size_t i = 0; i < targets.size(); i++) {
                printf("%s%s %.2f", i ? ", " : "", targets[i].label.c_str(), (double) s.per_file[i] / s.secs / GB);
            }
            printf(")\n");
        }
    }

    for (const auto & t : targets) {
        if (t.made_file) {
            sys::remove_file(t.file);
        }
        // a virus scanner may still hold the file for a moment, which keeps the folder non-empty
        for (int i = 0; !t.made_dir.empty() && !sys::remove_dir(t.made_dir) && i < 20; i++) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
}

} // namespace e8::bench
