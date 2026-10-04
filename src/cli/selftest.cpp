// `eightfer selftest <model.gguf> --tokens ids.txt [--half H] [--gpu-layers N]`: state carry-over check. For a set of
// layers, compares that layer's output for tokens [H, 2H) computed in one eval() of 2H tokens against two evals of H
// tokens each. Any difference beyond rounding means state carried between eval() calls (KV cache, conv state,
// delta-net state) is wrong; the first differing layer tells which kind.

#include "cli/commands.h"
#include "model/qwen35.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace e8::cli {

int selftest(const std::vector<std::string> & args) {
    std::string model, tokens_path;
    int         half = 64, gpu_layers = 0;
    for (size_t i = 2; i < args.size(); i++) {
        const std::string & k   = args[i];
        auto                val = [&]() -> std::string { return i + 1 < args.size() ? args[++i] : std::string(); };
        if (k == "--tokens") tokens_path = val();
        else if (k == "--half") half = std::atoi(val().c_str());
        else if (k == "--gpu-layers") gpu_layers = std::atoi(val().c_str());
        else if (model.empty() && k[0] != '-') model = k;
    }
    std::ifstream f(std::filesystem::u8path(tokens_path), std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    std::vector<int32_t> toks;
    const std::string    s = ss.str();
    for (size_t i = 0; i < s.size() && toks.size() < (size_t) half * 2;) {
        if (s[i] >= '0' && s[i] <= '9') {
            char * end = nullptr;
            toks.push_back((int32_t) std::strtol(s.c_str() + i, &end, 10));
            i = (size_t) (end - s.c_str());
        } else {
            i++;
        }
    }
    if (model.empty() || toks.size() < (size_t) half * 2 || half < 1) {
        fprintf(stderr, "usage: eightfer selftest <model.gguf> --tokens <ids.txt> [--half 64] [--gpu-layers N]\n");
        return 1;
    }

    model::LoadOptions o;
    o.n_gpu_layers = gpu_layers;
    o.n_ctx        = 512;
    o.n_ubatch     = half * 2;
    model::Qwen35 m;
    std::string   err;
    if (!m.load(model, o, err)) {
        fprintf(stderr, "load failed: %s\n", err.c_str());
        return 1;
    }
    const int64_t n_embd = m.hp().n_embd;
    printf("half %d, %d GPU layers\n", half, gpu_layers);
    for (int L : { 0, 1, 2, 3, 4, 7, 15, 31, (int) m.hp().n_layer - 1 }) {
        m.set_debug_layer(L);
        std::vector<float> a((size_t) n_embd * half * 2), b((size_t) n_embd * half * 2);
        m.reset();
        bool ok = m.eval(toks.data(), half * 2, a.data(), err);
        m.reset();
        ok = ok && m.eval(toks.data(), half, b.data(), err);
        ok = ok && m.eval(toks.data() + half, half, b.data() + (size_t) n_embd * half, err);
        if (!ok) {
            fprintf(stderr, "eval failed: %s\n", err.c_str());
            return 1;
        }
        double d1 = 0, n1 = 0, d2 = 0, n2 = 0, mx = 0;
        for (size_t i = 0; i < a.size(); i++) {
            const double d = (double) a[i] - b[i];
            if (i < (size_t) n_embd * half) {
                d1 += d * d;
                n1 += (double) a[i] * a[i];
            } else {
                d2 += d * d;
                n2 += (double) a[i] * a[i];
                mx = std::max(mx, std::fabs(d));
            }
        }
        printf("  layer %2d (%s): first half rel.err %.2e | second half rel.err %.2e, max abs %.3e\n", L,
               m.hp().is_recurrent(L) ? "delta-net" : "attention", std::sqrt(d1 / std::max(n1, 1e-30)),
               std::sqrt(d2 / std::max(n2, 1e-30)), mx);
    }
    return 0;
}

} // namespace e8::cli
