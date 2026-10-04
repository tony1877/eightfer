// `eightfer logits <model.gguf> --tokens ids.txt --out logits.f32 [--gpu-layers N] [--batch B] [--kv f16|f32]`: evaluates the token
// ids (whitespace/comma separated) from position 0 in batches of B and writes n_tokens * n_vocab float32 logits,
// row-major. Used to compare against reference implementations (tests/tiny).

#include "cli/commands.h"
#include "model/causal_lm.h"
#include "model/qwen35.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace e8::cli {

std::vector<int32_t> read_token_ids(const std::string & path) {
    std::ifstream     f(std::filesystem::u8path(path), std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    const std::string    s = ss.str();
    std::vector<int32_t> toks;
    for (size_t i = 0; i < s.size();) {
        if (s[i] >= '0' && s[i] <= '9') {
            char * end = nullptr;
            toks.push_back((int32_t) std::strtol(s.c_str() + i, &end, 10));
            i = (size_t) (end - s.c_str());
        } else {
            i++;
        }
    }
    return toks;
}

int logits(const std::vector<std::string> & args) {
    std::string model, tokens_path, out_path, res_path;
    std::string kv = "f16";
    int         gpu_layers = 0, batch = 512;
    for (size_t i = 2; i < args.size(); i++) {
        const std::string & k   = args[i];
        auto                val = [&]() -> std::string { return i + 1 < args.size() ? args[++i] : std::string(); };
        if (k == "--tokens") tokens_path = val();
        else if (k == "--out") out_path = val();
        else if (k == "--gpu-layers") gpu_layers = std::atoi(val().c_str());
        else if (k == "--batch") batch = std::atoi(val().c_str());
        else if (k == "--kv") kv = val();
        else if (k == "--res") res_path = val();
        else if (model.empty() && k[0] != '-') model = k;
    }
    const std::vector<int32_t> toks = read_token_ids(tokens_path);
    if (model.empty() || out_path.empty() || toks.empty() || batch < 1) {
        fprintf(stderr, "usage: eightfer logits <model.gguf> --tokens <ids.txt> --out <file> [--gpu-layers N] [--batch B]\n"
                        "                       [--kv f16|f32] [--res <pack .res.gguf>]\n");
        return 1;
    }
    model::LoadOptions o;
    o.n_gpu_layers = gpu_layers;
    o.output_gpu   = gpu_layers > 0;
    o.n_ctx        = (int) toks.size();
    o.n_ubatch     = batch;
    o.kv_type      = kv == "f32" ? GGML_TYPE_F32 : GGML_TYPE_F16;
    o.residual_path = res_path;
    std::string err;
    auto        mp = model::load_causal_lm(model, o, err);
    if (!mp) {
        fprintf(stderr, "load failed: %s\n", err.c_str());
        return 1;
    }
    model::CausalLM &  m  = *mp;
    const size_t       nv = (size_t) m.n_vocab();
    std::vector<float> all(toks.size() * nv);
    for (size_t i = 0; i < toks.size(); i += (size_t) batch) {
        const int n = (int) std::min(toks.size() - i, (size_t) batch);
        if (!m.eval(toks.data() + i, n, all.data() + i * nv, err)) {
            fprintf(stderr, "eval failed: %s\n", err.c_str());
            return 1;
        }
    }
    std::ofstream f(std::filesystem::u8path(out_path), std::ios::binary);
    f.write((const char *) all.data(), (std::streamsize) (all.size() * sizeof(float)));
    printf("%zu tokens x %zu vocab -> %s\n", toks.size(), nv, out_path.c_str());
    return f ? 0 : 1;
}

} // namespace e8::cli
