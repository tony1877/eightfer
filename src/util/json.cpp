#include "util/json.h"

#include <cstdlib>

namespace e8::json {

namespace {

struct Parser {
    const std::string & s;
    size_t              p = 0;
    std::string         err;

    void ws() {
        while (p < s.size() && (s[p] == ' ' || s[p] == '\n' || s[p] == '\r' || s[p] == '\t')) p++;
    }
    bool fail(const char * m) {
        if (err.empty()) err = std::string(m) + " at offset " + std::to_string(p);
        return false;
    }
    bool lit(const char * w) {
        const size_t n = std::char_traits<char>::length(w);
        if (s.compare(p, n, w) != 0) return fail("bad literal");
        p += n;
        return true;
    }
    static void utf8(std::string & o, uint32_t c) {
        if (c < 0x80) o += (char) c;
        else if (c < 0x800) { o += (char) (0xC0 | (c >> 6)); o += (char) (0x80 | (c & 0x3F)); }
        else if (c < 0x10000) { o += (char) (0xE0 | (c >> 12)); o += (char) (0x80 | ((c >> 6) & 0x3F)); o += (char) (0x80 | (c & 0x3F)); }
        else { o += (char) (0xF0 | (c >> 18)); o += (char) (0x80 | ((c >> 12) & 0x3F)); o += (char) (0x80 | ((c >> 6) & 0x3F)); o += (char) (0x80 | (c & 0x3F)); }
    }
    bool hex4(uint32_t & c) {
        if (p + 4 > s.size()) return fail("short \\u escape");
        c = (uint32_t) std::strtoul(s.substr(p, 4).c_str(), nullptr, 16);
        p += 4;
        return true;
    }
    bool string(std::string & o) {
        if (s[p] != '"') return fail("expected string");
        p++;
        while (p < s.size() && s[p] != '"') {
            char c = s[p++];
            if (c != '\\') { o += c; continue; }
            if (p >= s.size()) return fail("bad escape");
            c = s[p++];
            switch (c) {
                case 'n': o += '\n'; break;
                case 't': o += '\t'; break;
                case 'r': o += '\r'; break;
                case 'b': o += '\b'; break;
                case 'f': o += '\f'; break;
                case 'u': {
                    uint32_t cp = 0;
                    if (!hex4(cp)) return false;
                    if (cp >= 0xD800 && cp < 0xDC00 && p + 6 <= s.size() && s[p] == '\\' && s[p + 1] == 'u') {
                        p += 2;
                        uint32_t lo = 0;
                        if (!hex4(lo)) return false;
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    }
                    utf8(o, cp);
                    break;
                }
                default: o += c;
            }
        }
        if (p >= s.size()) return fail("unterminated string");
        p++;
        return true;
    }
    bool value(Value & v) {
        ws();
        if (p >= s.size()) return fail("unexpected end");
        const char c = s[p];
        if (c == '{') {
            v.kind = Value::Object;
            p++;
            ws();
            if (p < s.size() && s[p] == '}') { p++; return true; }
            while (true) {
                ws();
                std::string k;
                if (!string(k)) return false;
                ws();
                if (p >= s.size() || s[p] != ':') return fail("expected ':'");
                p++;
                if (!value(v.obj[k])) return false;
                ws();
                if (p < s.size() && s[p] == ',') { p++; continue; }
                if (p < s.size() && s[p] == '}') { p++; return true; }
                return fail("expected ',' or '}'");
            }
        }
        if (c == '[') {
            v.kind = Value::Array;
            p++;
            ws();
            if (p < s.size() && s[p] == ']') { p++; return true; }
            while (true) {
                v.arr.emplace_back();
                if (!value(v.arr.back())) return false;
                ws();
                if (p < s.size() && s[p] == ',') { p++; continue; }
                if (p < s.size() && s[p] == ']') { p++; return true; }
                return fail("expected ',' or ']'");
            }
        }
        if (c == '"') { v.kind = Value::String; return string(v.str); }
        if (c == 't') { v.kind = Value::Bool; v.b = true; return lit("true"); }
        if (c == 'f') { v.kind = Value::Bool; v.b = false; return lit("false"); }
        if (c == 'n') { v.kind = Value::Null; return lit("null"); }
        char * end = nullptr;
        v.num      = std::strtod(s.c_str() + p, &end);
        if (end == s.c_str() + p) return fail("bad value");
        v.kind = Value::Number;
        v.i    = (int64_t) v.num;
        const std::string tok(s.c_str() + p, (size_t) (end - (s.c_str() + p)));
        if (tok.find_first_of(".eE") == std::string::npos) v.i = std::strtoll(tok.c_str(), nullptr, 10);
        p = (size_t) (end - s.c_str());
        return true;
    }
};

} // namespace

bool parse(const std::string & text, Value & out, std::string & err) {
    Parser ps{ text };
    if (!ps.value(out)) {
        err = ps.err;
        return false;
    }
    return true;
}

} // namespace e8::json
