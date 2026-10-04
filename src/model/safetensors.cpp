#include "model/safetensors.h"

#include "util/json.h"

#include "ggml.h"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <set>
#include <sstream>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

namespace e8::model {

namespace fs = std::filesystem;

namespace {

std::mutex g_io;

std::string utf8(const fs::path & p) {
    const auto u = p.u8string();
    return std::string(u.begin(), u.end());
}  // FILE* handles are shared; reads are serialized per call

bool read_text(const fs::path & p, std::string & out) {
    std::ifstream f(p, std::ios::binary);
    if (!f) return false;
    std::stringstream ss;
    ss << f.rdbuf();
    out = ss.str();
    return true;
}

bool pread_at(FILE * f, uint64_t off, void * dst, size_t n) {
#if defined(_WIN32)
    if (_fseeki64(f, (long long) off, SEEK_SET) != 0) return false;
#else
    if (fseeko(f, (off_t) off, SEEK_SET) != 0) return false;
#endif
    return fread(dst, 1, n, f) == n;
}

size_t dtype_size(const std::string & d) {
    if (d == "BF16" || d == "F16") return 2;
    if (d == "F32") return 4;
    return 0;
}

} // namespace

int64_t StTensor::n_rows() const {
    int64_t n = 1;
    for (size_t i = 0; i + 1 < shape.size(); i++) n *= shape[i];
    return n;
}

int64_t StTensor::n_elements() const {
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    return n;
}

Safetensors::~Safetensors() {
    for (void * f : fps_) {
        if (f) fclose((FILE *) f);
    }
}

bool Safetensors::open(const std::string & dir, std::string & err) {
    const fs::path d = fs::u8path(dir);
    std::vector<std::string> shards;
    std::string              text;
    if (read_text(d / "model.safetensors.index.json", text)) {
        json::Value idx;
        if (!json::parse(text, idx, err)) {
            err = "index json: " + err;
            return false;
        }
        const json::Value * wm = idx.get("weight_map");
        if (!wm || wm->kind != json::Value::Object) {
            err = "index json has no weight_map";
            return false;
        }
        std::set<std::string> uniq;
        for (auto & [k, v] : wm->obj) uniq.insert(v.str);
        shards.assign(uniq.begin(), uniq.end());
    } else if (fs::exists(d / "model.safetensors")) {
        shards.push_back("model.safetensors");
    } else {
        err = "no model.safetensors(.index.json) in " + dir;
        return false;
    }

    for (const std::string & s : shards) {
        const fs::path p = d / fs::u8path(s);
#if defined(_WIN32)
        FILE * f = _wfopen(p.wstring().c_str(), L"rb");
#else
        FILE * f = fopen(p.string().c_str(), "rb");
#endif
        if (!f) {
            err = "cannot open " + utf8(p);
            return false;
        }
        const int fi = (int) files_.size();
        files_.push_back(utf8(p));
        fps_.push_back(f);
        uint64_t hlen = 0;
        if (fread(&hlen, 8, 1, f) != 1 || hlen > (100u << 20)) {
            err = "bad safetensors header in " + s;
            return false;
        }
        std::string hdr(hlen, '\0');
        if (fread(hdr.data(), 1, hlen, f) != hlen) {
            err = "short header in " + s;
            return false;
        }
        json::Value h;
        if (!json::parse(hdr, h, err)) {
            err = s + ": " + err;
            return false;
        }
        for (auto & [name, v] : h.obj) {
            if (name == "__metadata__") continue;
            const json::Value * dt = v.get("dtype"), * sh = v.get("shape"), * of = v.get("data_offsets");
            if (!dt || !sh || !of || of->arr.size() != 2) {
                err = s + ": bad entry " + name;
                return false;
            }
            StTensor t;
            t.name  = name;
            t.dtype = dt->str;
            for (auto & x : sh->arr) t.shape.push_back(x.i);
            t.file   = fi;
            t.offset = 8 + hlen + (uint64_t) of->arr[0].i;
            t.nbytes = (uint64_t) (of->arr[1].i - of->arr[0].i);
            tensors_[name] = std::move(t);
        }
    }
    return true;
}

const StTensor * Safetensors::find(const std::string & name) const {
    auto it = tensors_.find(name);
    return it == tensors_.end() ? nullptr : &it->second;
}

bool Safetensors::read_rows(const StTensor & t, int64_t row0, int64_t n, float * out, std::string & err) const {
    const size_t es = dtype_size(t.dtype);
    if (!es) {
        err = t.name + ": unsupported dtype " + t.dtype;
        return false;
    }
    const int64_t len = t.row_len();
    if (row0 < 0 || row0 + n > t.n_rows()) {
        err = t.name + ": row range out of bounds";
        return false;
    }
    const size_t         nb = (size_t) (n * len) * es;
    std::vector<uint8_t> raw(es == 4 ? 0 : nb);
    void *               dst = es == 4 ? (void *) out : (void *) raw.data();
    {
        std::lock_guard<std::mutex> lk(g_io);
        if (!pread_at((FILE *) fps_[(size_t) t.file], t.offset + (uint64_t) row0 * len * es, dst, nb)) {
            err = t.name + ": read failed";
            return false;
        }
    }
    const size_t ne = (size_t) (n * len);
    if (t.dtype == "BF16") {
        const uint16_t * s = (const uint16_t *) raw.data();
        for (size_t i = 0; i < ne; i++) {
            const uint32_t u = (uint32_t) s[i] << 16;
            std::memcpy(out + i, &u, 4);
        }
    } else if (t.dtype == "F16") {
        ggml_fp16_to_fp32_row((const ggml_fp16_t *) raw.data(), out, (int64_t) ne);
    }
    return true;
}

} // namespace e8::model
