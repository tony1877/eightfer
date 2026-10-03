#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace e8::io {

struct Completion {
    uint64_t tag;
    uint32_t bytes;  // bytes actually read
    int      error;  // 0 = ok, otherwise the OS error code
};

// Asynchronous reads that bypass the OS page cache.
//   Windows: FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED handles on one I/O completion port.
//   POSIX:   O_DIRECT (buffered fallback if the filesystem refuses it) + a pool of pread() workers.
// Offsets, lengths and destination buffers must be multiples of alignment(file).
// Not thread-safe: submit() and wait() must be called from one thread.
class AsyncReader {
public:
    static std::unique_ptr<AsyncReader> create(int max_inflight);
    virtual ~AsyncReader() = default;

    // Returns a file id >= 0, or -1 with err set.
    virtual int      open(const std::string & path, std::string & err) = 0;
    virtual uint64_t size(int file) const = 0;
    virtual uint32_t alignment(int file) const = 0;
    virtual bool     unbuffered(int file) const = 0;

    // Queues a read. Returns false (and queues nothing) if max_inflight reads are pending.
    // Errors, including immediate ones, are reported through wait().
    virtual bool submit(int file, uint64_t offset, uint32_t len, void * dst, uint64_t tag) = 0;

    // Blocks until at least one read finishes. Returns the number of completions written to out (<= max).
    virtual int wait(Completion * out, int max) = 0;

    virtual int inflight() const = 0;
};

// Page-aligned memory, valid as a destination for unbuffered reads.
void * alloc_aligned(size_t bytes);
void   free_aligned(void * p);

std::string os_error_string(int code);

} // namespace e8::io
