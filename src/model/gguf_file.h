#pragma once

// A GGUF file opened for metadata: typed key/value access and the tensor table (shapes, types and file offsets),
// without loading any tensor data. ggml's gguf API does the parsing.

#include "ggml.h"
#include "gguf.h"

#include <cstdint>
#include <string>
#include <vector>

namespace e8::model {

class GgufFile {
public:
    GgufFile() = default;
    ~GgufFile();
    GgufFile(const GgufFile &)             = delete;
    GgufFile & operator=(const GgufFile &) = delete;

    // Opens `path` and reads its header. Returns false with `err` set on failure.
    bool open(const std::string & path, std::string & err);

    const std::string & path() const { return path_; }

    std::string arch() const { return str("general.architecture", ""); }

    bool has(const std::string & key) const;
    // Scalar getters. Integer getters accept any integer type; `def` is returned when the key is missing.
    int64_t     i64(const std::string & key, int64_t def) const;
    float       f32(const std::string & key, float def) const;
    bool        b(const std::string & key, bool def) const;
    std::string str(const std::string & key, const std::string & def) const;
    // Integer or integer-array key as a vector (a scalar becomes one element). Empty when missing.
    std::vector<int64_t> i64_arr(const std::string & key) const;

    int64_t n_kv() const;
    // One line describing key/value `i`; arrays longer than `max_items` are summarized.
    std::string describe_kv(int64_t i, size_t max_items) const;

    int64_t             n_tensors() const;
    // Tensor metadata (shape, type, name) in a no_alloc ggml context; nullptr when the name is unknown.
    ggml_tensor *       tensor(const std::string & name) const;
    ggml_tensor *       tensor(int64_t i) const;
    // Absolute file offset of a tensor's data.
    uint64_t            data_offset(const ggml_tensor * t) const;
    // The underlying gguf context (e.g. to copy all key/values into a new file).
    const gguf_context * raw() const { return gguf_; }
    uint64_t            file_size() const { return file_size_; }

private:
    int64_t key_id(const std::string & key) const;

    std::string    path_;
    gguf_context * gguf_      = nullptr;
    ggml_context * meta_      = nullptr;
    uint64_t       file_size_ = 0;
};

} // namespace e8::model
