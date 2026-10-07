// `eightfer dump <base.gguf> --res <res.gguf> --tokens ids.txt --out file [-n 20480] [--chunk 256]`
//
// Experiment data for a drafter correction: per position, the last layer's output (before the final norm) of the base
// alone and of base + residual, both as f16 [n_embd], over chunks of `--chunk` tokens (each from an empty context).
// File: int32 n, int32 n_embd, then n rows of base hidden, then n rows of full hidden.

#include "cli/commands.h"
#include "model/qwen35.h"

#include "ggml.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace e8::cli {

int dump(const std::vector<std::string> & args) {
    std::string model, res, tokens_path, out;
    int         n = 20480, chunk = 256;
    for (size_t i = 2; i < args.size(); i++) {
        const std::string & a   = args[i];
        auto                val = [&]() -> std::string { return i + 1 < args.size() ? args[++i] : std::string(); };
        if (a == "--res") res = val();
        else if (a == "--tokens") tokens_path = val();
        else if (a == "--out") out = val();
        else if (a == "-n") n = std::atoi(val().c_str());
        else if (a == "--chunk") chunk = std::atoi(val().c_str());
        else if (model.empty() && a[0] != '-') model = a;
        else {
            fprintf(stderr, "unknown option: %s\n", a.c_str());
            return 1;
        }
    }
    const std::vector<int32_t> toks = read_token_ids(tokens_path);
    if (model.empty() || res.empty() || out.empty() || toks.empty()) {
        fprintf(stderr, "usage: eightfer dump <base.gguf> --res <res.gguf> --tokens ids.txt --out file [-n N] [--chunk C]\n");
        return 1;
    }
    n = std::min<int>(n, (int) toks.size() / chunk * chunk);
    model::LoadOptions o;
    o.n_gpu_layers  = 999;
    o.n_ctx         = chunk + 64;
    o.n_ubatch      = chunk;
    o.residual_path = res;
    o.mtp           = false;
    model::Qwen35 m;
    std::string   err;
    if (!m.load(model, o, err)) {
        fprintf(stderr, "load failed: %s\n", err.c_str());
        return 1;
    }
    const int64_t      ne = m.hp().n_embd;
    std::vector<float> h((size_t) chunk * ne);
    std::vector<ggml_fp16_t> hb((size_t) n * ne), hf((size_t) n * ne);
    m.set_debug_layer((int) m.hp().n_layer - 1);
    for (int c = 0; c < n; c += chunk) {
        for (int full = 0; full < 2; full++) {
            m.reset();
            model::EvalOpts eo;
            eo.residual = full == 1;
            if (!m.eval(toks.data() + c, chunk, eo, h.data(), nullptr, err)) {
                fprintf(stderr, "eval: %s\n", err.c_str());
                return 1;
            }
            ggml_fp32_to_fp16_row(h.data(), (full ? hf : hb).data() + (size_t) c * ne, (int64_t) chunk * ne);
        }
        if (c % (chunk * 16) == 0) fprintf(stderr, "dump: %d / %d\n", c + chunk, n);
    }
    FILE * f = fopen(out.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "cannot write %s\n", out.c_str());
        return 1;
    }
    const int32_t hdr[2] = { n, (int32_t) ne };
    fwrite(hdr, sizeof(hdr), 1, f);
    fwrite(hb.data(), sizeof(ggml_fp16_t), hb.size(), f);
    fwrite(hf.data(), sizeof(ggml_fp16_t), hf.size(), f);
    fclose(f);
    fprintf(stderr, "wrote %d positions to %s\n", n, out.c_str());
    return 0;
}

} // namespace e8::cli
