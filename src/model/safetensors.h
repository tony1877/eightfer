#pragma once

// Read-only access to a (sharded) Hugging Face safetensors checkpoint: model.safetensors.index.json + shards, or a
// single model.safetensors. Rows are converted to float32 on read.

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace e8::model {

struct StTensor {
    std::string          name;
    std::string          dtype;  // BF16, F16, F32
    std::vector<int64_t> shape;  // row-major, outermost first (PyTorch order)
    int                  file = -1;
    uint64_t             offset = 0;  // absolute byte offset of the data in the shard
    uint64_t             nbytes = 0;

    int64_t row_len() const { return shape.empty() ? 1 : shape.back(); }
    int64_t n_rows() const;
    int64_t n_elements() const;
};

class Safetensors {
public:
    ~Safetensors();
    bool open(const std::string & dir, std::string & err);

    const StTensor * find(const std::string & name) const;
    const std::map<std::string, StTensor> & tensors() const { return tensors_; }

    // Reads rows [row0, row0 + n) of `t` (rows of row_len() elements) as float32 into `out`.
    bool read_rows(const StTensor & t, int64_t row0, int64_t n, float * out, std::string & err) const;

private:
    std::vector<std::string>        files_;
    std::vector<void *>             fps_;  // FILE*
    std::map<std::string, StTensor> tensors_;
};

} // namespace e8::model
