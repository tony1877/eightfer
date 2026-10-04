#include "model/gguf_file.h"

#include <cinttypes>
#include <cstdio>
#include <filesystem>

namespace e8::model {

GgufFile::~GgufFile() {
    if (gguf_) {
        gguf_free(gguf_);
    }
    if (meta_) {
        ggml_free(meta_);
    }
}

bool GgufFile::open(const std::string & path, std::string & err) {
    path_ = path;
    std::error_code ec;
    file_size_ = (uint64_t) std::filesystem::file_size(std::filesystem::u8path(path), ec);
    if (ec) {
        err = path + ": " + ec.message();
        return false;
    }
    gguf_init_params p = { /*no_alloc=*/true, /*ctx=*/&meta_ };
    gguf_              = gguf_init_from_file(path.c_str(), p);
    if (!gguf_) {
        err = path + ": not a readable GGUF file";
        return false;
    }
    return true;
}

int64_t GgufFile::key_id(const std::string & key) const {
    return gguf_ ? gguf_find_key(gguf_, key.c_str()) : -1;
}

bool GgufFile::has(const std::string & key) const {
    return key_id(key) >= 0;
}

static bool int_value(const gguf_context * g, int64_t id, gguf_type t, const void * data, size_t i, int64_t & out) {
    switch (t) {
        case GGUF_TYPE_UINT8:  out = ((const uint8_t *) data)[i]; return true;
        case GGUF_TYPE_INT8:   out = ((const int8_t *) data)[i]; return true;
        case GGUF_TYPE_UINT16: out = ((const uint16_t *) data)[i]; return true;
        case GGUF_TYPE_INT16:  out = ((const int16_t *) data)[i]; return true;
        case GGUF_TYPE_UINT32: out = ((const uint32_t *) data)[i]; return true;
        case GGUF_TYPE_INT32:  out = ((const int32_t *) data)[i]; return true;
        case GGUF_TYPE_UINT64: out = (int64_t) ((const uint64_t *) data)[i]; return true;
        case GGUF_TYPE_INT64:  out = ((const int64_t *) data)[i]; return true;
        case GGUF_TYPE_BOOL:   out = ((const int8_t *) data)[i] != 0; return true;
        default:               (void) g; (void) id; return false;
    }
}

int64_t GgufFile::i64(const std::string & key, int64_t def) const {
    const int64_t id = key_id(key);
    if (id < 0) {
        return def;
    }
    const gguf_type t = gguf_get_kv_type(gguf_, id);
    if (t == GGUF_TYPE_ARRAY) {
        return def;
    }
    int64_t v = def;
    return int_value(gguf_, id, t, gguf_get_val_data(gguf_, id), 0, v) ? v : def;
}

float GgufFile::f32(const std::string & key, float def) const {
    const int64_t id = key_id(key);
    if (id < 0) {
        return def;
    }
    switch (gguf_get_kv_type(gguf_, id)) {
        case GGUF_TYPE_FLOAT32: return gguf_get_val_f32(gguf_, id);
        case GGUF_TYPE_FLOAT64: return (float) gguf_get_val_f64(gguf_, id);
        default: {
            const int64_t v = i64(key, INT64_MIN);
            return v == INT64_MIN ? def : (float) v;
        }
    }
}

bool GgufFile::b(const std::string & key, bool def) const {
    const int64_t v = i64(key, -1);
    return v < 0 ? def : v != 0;
}

std::string GgufFile::str(const std::string & key, const std::string & def) const {
    const int64_t id = key_id(key);
    if (id < 0 || gguf_get_kv_type(gguf_, id) != GGUF_TYPE_STRING) {
        return def;
    }
    return gguf_get_val_str(gguf_, id);
}

std::vector<int64_t> GgufFile::i64_arr(const std::string & key) const {
    std::vector<int64_t> out;
    const int64_t        id = key_id(key);
    if (id < 0) {
        return out;
    }
    const gguf_type t = gguf_get_kv_type(gguf_, id);
    if (t != GGUF_TYPE_ARRAY) {
        int64_t v = 0;
        if (int_value(gguf_, id, t, gguf_get_val_data(gguf_, id), 0, v)) {
            out.push_back(v);
        }
        return out;
    }
    const gguf_type at = gguf_get_arr_type(gguf_, id);
    const size_t    n  = gguf_get_arr_n(gguf_, id);
    const void *    d  = gguf_get_arr_data(gguf_, id);
    for (size_t i = 0; i < n; i++) {
        int64_t v = 0;
        if (!int_value(gguf_, id, at, d, i, v)) {
            out.clear();
            return out;
        }
        out.push_back(v);
    }
    return out;
}

int64_t GgufFile::n_kv() const {
    return gguf_ ? gguf_get_n_kv(gguf_) : 0;
}

std::string GgufFile::describe_kv(int64_t i, size_t max_items) const {
    const char *    key = gguf_get_key(gguf_, i);
    const gguf_type t   = gguf_get_kv_type(gguf_, i);
    std::string     s   = key;
    s += " = ";
    char buf[128];
    if (t == GGUF_TYPE_STRING) {
        std::string v = gguf_get_val_str(gguf_, i);
        if (v.size() > 120) {
            v = v.substr(0, 117) + "...";
        }
        for (char & c : v) {
            if (c == '\n' || c == '\r') {
                c = ' ';
            }
        }
        return s + "\"" + v + "\"";
    }
    if (t == GGUF_TYPE_ARRAY) {
        const gguf_type at = gguf_get_arr_type(gguf_, i);
        const size_t    n  = gguf_get_arr_n(gguf_, i);
        snprintf(buf, sizeof(buf), "[%s x %zu]", gguf_type_name(at), n);
        s += buf;
        if (n <= max_items && at != GGUF_TYPE_STRING) {
            s += " {";
            const void * d = gguf_get_arr_data(gguf_, i);
            for (size_t j = 0; j < n; j++) {
                int64_t v = 0;
                if (at == GGUF_TYPE_FLOAT32) {
                    snprintf(buf, sizeof(buf), "%s%g", j ? ", " : "", ((const float *) d)[j]);
                } else if (int_value(gguf_, i, at, d, j, v)) {
                    snprintf(buf, sizeof(buf), "%s%" PRId64, j ? ", " : "", v);
                } else {
                    snprintf(buf, sizeof(buf), "%s?", j ? ", " : "");
                }
                s += buf;
            }
            s += "}";
        }
        return s;
    }
    if (t == GGUF_TYPE_FLOAT32 || t == GGUF_TYPE_FLOAT64) {
        snprintf(buf, sizeof(buf), "%g", f32(key, 0));
        return s + buf;
    }
    snprintf(buf, sizeof(buf), "%" PRId64, i64(key, 0));
    return s + buf;
}

int64_t GgufFile::n_tensors() const {
    return gguf_ ? gguf_get_n_tensors(gguf_) : 0;
}

ggml_tensor * GgufFile::tensor(const std::string & name) const {
    return meta_ ? ggml_get_tensor(meta_, name.c_str()) : nullptr;
}

ggml_tensor * GgufFile::tensor(int64_t i) const {
    return tensor(std::string(gguf_get_tensor_name(gguf_, i)));
}

uint64_t GgufFile::data_offset(const ggml_tensor * t) const {
    const int64_t id = gguf_find_tensor(gguf_, ggml_get_name(t));
    return id < 0 ? 0 : (uint64_t) (gguf_get_data_offset(gguf_) + gguf_get_tensor_offset(gguf_, id));
}

} // namespace e8::model
