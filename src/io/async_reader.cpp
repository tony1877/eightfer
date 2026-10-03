#include "io/async_reader.h"

#include <algorithm>
#include <cstring>
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

namespace e8::io {

static std::wstring widen(const std::string & s) {
    if (s.empty()) {
        return {};
    }
    const int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), (int) s.size(), nullptr, 0);
    std::wstring w((size_t) n, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.data(), (int) s.size(), w.data(), n);
    return w;
}

std::string os_error_string(int code) {
    char * buf = nullptr;
    const DWORD n = FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                                   nullptr, (DWORD) code, 0, (LPSTR) &buf, 0, nullptr);
    std::string s = n ? std::string(buf, n) : std::string("error");
    if (buf) {
        LocalFree(buf);
    }
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ' || s.back() == '.')) {
        s.pop_back();
    }
    return s + " (" + std::to_string(code) + ")";
}

void * alloc_aligned(size_t bytes) {
    return VirtualAlloc(nullptr, bytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
}

void free_aligned(void * p) {
    if (p) {
        VirtualFree(p, 0, MEM_RELEASE);
    }
}

namespace {

class WinReader final : public AsyncReader {
    // Completion key for errors raised synchronously by ReadFile, posted by hand so wait() reports them.
    static constexpr ULONG_PTR kSynthetic = ~(ULONG_PTR) 0;

    struct File {
        HANDLE   h;
        uint64_t size;
        uint32_t align;
    };

    struct Op {
        OVERLAPPED ov;  // first member: OVERLAPPED* from the port casts back to Op*
        uint64_t   tag;
        DWORD      err;
    };

    HANDLE                        iocp_ = nullptr;
    std::vector<File>             files_;
    std::vector<Op>               ops_;  // fixed size, addresses must stay stable while reads are pending
    std::vector<Op *>             free_;
    std::vector<OVERLAPPED_ENTRY> ents_;

public:
    explicit WinReader(int max_inflight) : ops_((size_t) max_inflight), ents_((size_t) max_inflight) {
        iocp_ = CreateIoCompletionPort(INVALID_HANDLE_VALUE, nullptr, 0, 1);
        for (auto & op : ops_) {
            free_.push_back(&op);
        }
    }

    ~WinReader() override {
        for (auto & f : files_) {
            CancelIoEx(f.h, nullptr);
        }
        std::vector<Completion> sink(ops_.size());
        while (inflight() > 0 && wait(sink.data(), (int) sink.size()) > 0) {
        }
        for (auto & f : files_) {
            CloseHandle(f.h);
        }
        if (iocp_) {
            CloseHandle(iocp_);
        }
    }

    int open(const std::string & path, std::string & err) override {
        if (!iocp_) {
            err = "CreateIoCompletionPort failed";
            return -1;
        }
        HANDLE h = CreateFileW(widen(path).c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                               nullptr, OPEN_EXISTING, FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED, nullptr);
        if (h == INVALID_HANDLE_VALUE) {
            err = os_error_string((int) GetLastError());
            return -1;
        }
        LARGE_INTEGER sz{};
        if (!GetFileSizeEx(h, &sz)) {
            err = os_error_string((int) GetLastError());
            CloseHandle(h);
            return -1;
        }
        // NO_BUFFERING needs sector-aligned offsets, sizes and buffers; 4 KiB covers 512e/4Kn drives.
        uint32_t          align = 4096;
        FILE_STORAGE_INFO si{};
        if (GetFileInformationByHandleEx(h, FileStorageInfo, &si, sizeof(si)) && si.LogicalBytesPerSector > align) {
            align = si.LogicalBytesPerSector;
        }
        if (!CreateIoCompletionPort(h, iocp_, (ULONG_PTR) files_.size(), 0)) {
            err = os_error_string((int) GetLastError());
            CloseHandle(h);
            return -1;
        }
        files_.push_back({ h, (uint64_t) sz.QuadPart, align });
        return (int) files_.size() - 1;
    }

    uint64_t size(int file) const override { return files_[(size_t) file].size; }

    uint32_t alignment(int file) const override { return files_[(size_t) file].align; }

    bool unbuffered(int) const override { return true; }

    bool submit(int file, uint64_t offset, uint32_t len, void * dst, uint64_t tag) override {
        if (free_.empty()) {
            return false;
        }
        Op * op = free_.back();
        free_.pop_back();
        ZeroMemory(&op->ov, sizeof(op->ov));
        op->ov.Offset     = (DWORD) (offset & 0xffffffffu);
        op->ov.OffsetHigh = (DWORD) (offset >> 32);
        op->tag           = tag;
        op->err           = 0;
        if (!ReadFile(files_[(size_t) file].h, dst, len, nullptr, &op->ov)) {
            const DWORD e = GetLastError();
            if (e != ERROR_IO_PENDING) {
                // failed synchronously: no packet gets queued for it, so queue one ourselves
                op->err = e;
                PostQueuedCompletionStatus(iocp_, 0, kSynthetic, &op->ov);
            }
        }
        return true;
    }

    int wait(Completion * out, int max) override {
        if (inflight() == 0 || max <= 0) {
            return 0;
        }
        ULONG       n    = 0;
        const ULONG want = (ULONG) std::min<size_t>((size_t) max, ents_.size());
        if (!GetQueuedCompletionStatusEx(iocp_, ents_.data(), want, &n, INFINITE, FALSE)) {
            return 0;
        }
        for (ULONG i = 0; i < n; i++) {
            Op *         op = reinterpret_cast<Op *>(ents_[i].lpOverlapped);
            Completion & c  = out[i];
            c.tag           = op->tag;
            if (ents_[i].lpCompletionKey == kSynthetic) {
                c.bytes = 0;
                c.error = (int) op->err;
            } else {
                DWORD      bytes = 0;
                const BOOL ok = GetOverlappedResult(files_[ents_[i].lpCompletionKey].h, &op->ov, &bytes, FALSE);
                c.bytes       = bytes;
                c.error       = ok ? 0 : (int) GetLastError();
            }
            free_.push_back(op);
        }
        return (int) n;
    }

    int inflight() const override { return (int) (ops_.size() - free_.size()); }
};

} // namespace

std::unique_ptr<AsyncReader> AsyncReader::create(int max_inflight) {
    return std::make_unique<WinReader>(std::max(1, max_inflight));
}

} // namespace e8::io

#else  // POSIX

#include <cerrno>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <fcntl.h>
#include <mutex>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>

namespace e8::io {

std::string os_error_string(int code) {
    return std::string(std::strerror(code)) + " (" + std::to_string(code) + ")";
}

void * alloc_aligned(size_t bytes) {
    void *       p    = nullptr;
    const size_t page = 4096;
    if (posix_memalign(&p, page, (bytes + page - 1) / page * page) != 0) {
        return nullptr;
    }
    return p;
}

void free_aligned(void * p) {
    std::free(p);
}

namespace {

// pread() worker pool; one worker per in-flight slot so the queue depth reaches the device.
class PosixReader final : public AsyncReader {
    struct File {
        int      fd;
        uint64_t size;
        uint32_t align;
        bool     direct;
    };

    struct Req {
        int      fd;
        uint64_t off;
        uint32_t len;
        void *   dst;
        uint64_t tag;
    };

    std::vector<File>        files_;
    mutable std::mutex       mu_;
    std::condition_variable  cv_req_;
    std::condition_variable  cv_done_;
    std::deque<Req>          reqs_;
    std::deque<Completion>   done_;
    std::vector<std::thread> workers_;
    bool                     stop_     = false;
    int                      inflight_ = 0;
    int                      max_;

    void worker() {
        for (;;) {
            Req r;
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_req_.wait(lk, [&] { return stop_ || !reqs_.empty(); });
                if (reqs_.empty()) {
                    return;
                }
                r = reqs_.front();
                reqs_.pop_front();
            }
            uint32_t got = 0;
            int      err = 0;
            while (got < r.len) {
                const uint32_t want = r.len - got;
                const ssize_t  n    = ::pread(r.fd, (char *) r.dst + got, want, (off_t) (r.off + got));
                if (n < 0) {
                    if (errno == EINTR) {
                        continue;
                    }
                    err = errno;
                    break;
                }
                got += (uint32_t) n;
                if ((uint32_t) n < want) {
                    break;  // EOF
                }
            }
            {
                std::lock_guard<std::mutex> lk(mu_);
                done_.push_back({ r.tag, got, err });
            }
            cv_done_.notify_one();
        }
    }

public:
    explicit PosixReader(int max_inflight) : max_(max_inflight) {
        for (int i = 0; i < max_inflight; i++) {
            workers_.emplace_back([this] { worker(); });
        }
    }

    ~PosixReader() override {
        {
            std::lock_guard<std::mutex> lk(mu_);
            stop_ = true;
        }
        cv_req_.notify_all();
        for (auto & t : workers_) {
            t.join();
        }
        for (auto & f : files_) {
            ::close(f.fd);
        }
    }

    int open(const std::string & path, std::string & err) override {
        int  fd     = -1;
        bool direct = false;
#ifdef O_DIRECT
        fd     = ::open(path.c_str(), O_RDONLY | O_DIRECT | O_CLOEXEC);
        direct = fd >= 0;
#endif
        if (fd < 0) {
            fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        }
        if (fd < 0) {
            err = os_error_string(errno);
            return -1;
        }
        struct stat st {};
        if (fstat(fd, &st) != 0) {
            err = os_error_string(errno);
            ::close(fd);
            return -1;
        }
        files_.push_back({ fd, (uint64_t) st.st_size, 4096, direct });
        return (int) files_.size() - 1;
    }

    uint64_t size(int file) const override { return files_[(size_t) file].size; }

    uint32_t alignment(int file) const override { return files_[(size_t) file].align; }

    bool unbuffered(int file) const override { return files_[(size_t) file].direct; }

    bool submit(int file, uint64_t offset, uint32_t len, void * dst, uint64_t tag) override {
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (inflight_ >= max_) {
                return false;
            }
            inflight_++;
            reqs_.push_back({ files_[(size_t) file].fd, offset, len, dst, tag });
        }
        cv_req_.notify_one();
        return true;
    }

    int wait(Completion * out, int max) override {
        std::unique_lock<std::mutex> lk(mu_);
        if (inflight_ == 0 || max <= 0) {
            return 0;
        }
        cv_done_.wait(lk, [&] { return !done_.empty(); });
        int n = 0;
        while (n < max && !done_.empty()) {
            out[n++] = done_.front();
            done_.pop_front();
        }
        inflight_ -= n;
        return n;
    }

    int inflight() const override {
        std::lock_guard<std::mutex> lk(mu_);
        return inflight_;
    }
};

} // namespace

std::unique_ptr<AsyncReader> AsyncReader::create(int max_inflight) {
    return std::make_unique<PosixReader>(std::max(1, max_inflight));
}

} // namespace e8::io

#endif
