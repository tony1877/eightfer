#include "model/gguf_set.h"

#include <cstdio>
#include <regex>

namespace e8::model {

bool GgufSet::open(const std::string & path, std::string & err) {
    // "<prefix>-00001-of-00003.gguf": open every split in order
    std::vector<std::string> paths;
    static const std::regex  re(R"((.*)-(\d{5})-of-(\d{5})\.gguf$)");
    std::smatch              m;
    if (std::regex_match(path, m, re)) {
        const int n = std::stoi(m[3].str());
        for (int i = 1; i <= n; i++) {
            char buf[32];
            snprintf(buf, sizeof buf, "-%05d-of-%05d.gguf", i, n);
            paths.push_back(m[1].str() + buf);
        }
    } else {
        paths.push_back(path);
    }
    for (const std::string & p : paths) {
        auto f = std::make_unique<GgufFile>();
        if (!f->open(p, err)) {
            return false;
        }
        auto mf = std::make_unique<util::MappedFile>();
        if (!mf->open(p, err)) {
            return false;
        }
        const size_t fi = files_.size();
        for (int64_t i = 0; i < f->n_tensors(); i++) {
            const ggml_tensor * t = f->tensor(i);
            index_[ggml_get_name(t)] = { t, fi };
            file_of_[t]             = fi;
        }
        files_.push_back(std::move(f));
        maps_.push_back(std::move(mf));
    }
    const int64_t want = meta().i64("split.tensors.count", -1);
    if (want >= 0 && (int64_t) index_.size() != want) {
        err = "split set has " + std::to_string(index_.size()) + " tensors, expected " + std::to_string(want);
        return false;
    }
    return true;
}

const ggml_tensor * GgufSet::tensor(const std::string & name) const {
    auto it = index_.find(name);
    return it == index_.end() ? nullptr : it->second.meta;
}

const void * GgufSet::data(const ggml_tensor * t) const {
    const size_t fi = file_of_.at(t);
    return maps_[fi]->data() + files_[fi]->data_offset(t);
}

std::vector<const ggml_tensor *> GgufSet::tensors() const {
    std::vector<const ggml_tensor *> v;
    for (auto & [n, e] : index_) v.push_back(e.meta);
    return v;
}

} // namespace e8::model
