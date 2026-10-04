#pragma once

// A GGUF model that may be split over several files (llama.cpp's "-00001-of-0000N.gguf" convention), memory-mapped.
// Key/values come from the first file; the tensor table is merged across all files.

#include "model/gguf_file.h"
#include "util/mmap.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace e8::model {

class GgufSet {
public:
    // Opens `path` (any split; the others are found by name) and maps every file.
    bool open(const std::string & path, std::string & err);

    const GgufFile & meta() const { return *files_[0]; }
    std::string      arch() const { return meta().arch(); }

    // Tensor metadata, nullptr when missing.
    const ggml_tensor * tensor(const std::string & name) const;
    // Pointer to the tensor's bytes inside the mapping.
    const void *        data(const ggml_tensor * t) const;
    std::vector<const ggml_tensor *> tensors() const;
    size_t              n_files() const { return files_.size(); }
    size_t              file_of(const ggml_tensor * t) const { return file_of_.at(t); }
    const util::MappedFile & map(size_t i) const { return *maps_[i]; }

private:
    struct Entry {
        const ggml_tensor * meta = nullptr;
        size_t              file = 0;
    };
    std::vector<std::unique_ptr<GgufFile>>         files_;
    std::vector<std::unique_ptr<util::MappedFile>> maps_;
    std::map<std::string, Entry>                   index_;
    std::map<const ggml_tensor *, size_t>          file_of_;
};

} // namespace e8::model
