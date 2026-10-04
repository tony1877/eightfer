#pragma once

// Minimal JSON reader: enough for safetensors headers and HF index files (objects, arrays, strings, numbers,
// true/false/null). Numbers are kept as double and as int64 when integral.

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace e8::json {

struct Value {
    enum Kind { Null, Bool, Number, String, Array, Object } kind = Null;
    bool                         b   = false;
    double                       num = 0;
    int64_t                      i   = 0;
    std::string                  str;
    std::vector<Value>           arr;
    std::map<std::string, Value> obj;

    const Value * get(const std::string & k) const {
        auto it = obj.find(k);
        return it == obj.end() ? nullptr : &it->second;
    }
};

// Parses `text`; returns false with `err` set on malformed input.
bool parse(const std::string & text, Value & out, std::string & err);

} // namespace e8::json
