// `shoehorn decode <model.gguf> [--tokens prompt_ids.txt] [-n 64] [--gpu-layers N] [--ctx N] [--expert-cache-gb G]`:
// greedy decoding speed for any architecture (prefill the prompt, then one token per eval). Prints tok/s; with
// a Flash-Next model also the expert cache hit rate.

#include "cli/commands.h"
#include "model/causal_lm.h"
#include "model/qwen35.h"
#include "model/qwen4exp.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace e8::cli {

int decode(const std::vector<std::string> & args) {
    std::string model, tokens_path;
    int         n_gen = 64, gpu_layers = 999, n_ctx = 4096, threads = 0;
    int gpu_kv = -1;
    double      cache_gb = -1;
    for (size_t i = 2; i < args.size(); i++) {
        const std::string & a   = args[i];
        auto                val = [&]() -> std::string { return i + 1 < args.size() ? args[++i] : std::string(); };
        if (a == "--tokens") tokens_path = val();
        else if (a == "-n") n_gen = std::atoi(val().c_str());
        else if (a == "--gpu-layers") gpu_layers = std::atoi(val().c_str());
        else if (a == "--ctx") n_ctx = std::atoi(val().c_str());
        else if (a == "--gpu-kv") gpu_kv = std::atoi(val().c_str());
        else if (a == "--threads") threads = std::atoi(val().c_str());
        else if (a == "--expert-cache-gb") cache_gb = std::atof(val().c_str());
        else if (model.empty() && a[0] != '-') model = a;
        else {
            fprintf(stderr, "unknown option: %s\n", a.c_str());
            return 1;
        }
    }
    std::vector<int32_t> prompt = tokens_path.empty() ? std::vector<int32_t>{} : read_token_ids(tokens_path);
    if (model.empty() || n_gen < 1) {
        fprintf(stderr, "usage: shoehorn decode <model.gguf> [--tokens ids.txt] [-n 64] [--gpu-layers N] [--ctx N]\n"
                        "                       [--threads N] [--expert-cache-gb G]\n");
        return 1;
    }
    if (prompt.empty()) prompt = { 248045, 846, 198 };  // "<|im_start|>user\n"
    model::LoadOptions o;
    o.n_gpu_layers    = gpu_layers;
    o.n_ctx           = std::max(n_ctx, (int) prompt.size() + n_gen + 16);
    o.gpu_kv = gpu_kv;
    o.n_threads       = threads;
    o.expert_cache_gb = cache_gb;
    std::string  err;
    const auto   tl = std::chrono::steady_clock::now();
    auto         mp = model::load_causal_lm(model, o, err);
    if (!mp) {
        fprintf(stderr, "load failed: %s\n", err.c_str());
        return 1;
    }
    model::CausalLM & m = *mp;
    printf("loaded in %.1f s: %.2f GB GPU, %.2f GB CPU/mapped\n",
           std::chrono::duration<double>(std::chrono::steady_clock::now() - tl).count(), m.gpu_weight_bytes() / 1e9,
           m.cpu_weight_bytes() / 1e9);
    const int64_t      nv = m.n_vocab();
    std::vector<float> lg((size_t) nv * std::min<size_t>(prompt.size(), 512));
    auto               t0 = std::chrono::steady_clock::now();
    if (!m.prefill(prompt.data(), (int) prompt.size(), lg.data(), err)) {
        fprintf(stderr, "prefill failed: %s\n", err.c_str());
        return 1;
    }
    const double tp  = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    int32_t      tok = (int32_t) (std::max_element(lg.begin(), lg.begin() + nv) - lg.begin());
    printf("prefill: %zu tokens in %.1f s (%.1f tok/s), next token %d (logit %.4f)\n", prompt.size(), tp,
           prompt.size() / tp, tok, lg[(size_t) tok]);
    std::vector<int32_t> out;
    std::vector<double>  times;
    for (int i = 0; i < n_gen; i++) {
        const auto t1 = std::chrono::steady_clock::now();
        if (!m.eval(&tok, 1, lg.data(), err)) {
            fprintf(stderr, "decode failed: %s\n", err.c_str());
            return 1;
        }
        times.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count());
        out.push_back(tok);
        tok = (int32_t) (std::max_element(lg.begin(), lg.begin() + nv) - lg.begin());
    }
    double total = 0;
    for (double t : times) total += t;
    std::vector<double> s = times;
    std::sort(s.begin(), s.end());
    printf("prefill %zu tok in %.2f s; decode %d tok in %.2f s = %.2f tok/s (median %.1f ms, p90 %.1f ms)\n",
           prompt.size(), tp, n_gen, total, n_gen / total, 1e3 * s[s.size() / 2], 1e3 * s[s.size() * 9 / 10]);
    // second half only: once a cache has warmed up
    double tail = 0;
    for (size_t i = times.size() / 2; i < times.size(); i++) tail += times[i];
    printf("second half: %.2f tok/s\n", (times.size() - times.size() / 2) / tail);
    if (auto * q = dynamic_cast<model::Qwen4Exp *>(&m)) {
        printf("%s\n", q->cache_report().c_str());
    }
    printf("tokens:");
    for (int32_t t : out) printf(" %d", t);
    printf("\n");
    return 0;
}

} // namespace e8::cli
