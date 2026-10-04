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

// --rollback: two runs verify a 5-token batch at the same position and keep its first 2 tokens; the 3 discarded
// tokens differ between the runs. If rollback() leaves no trace of discarded tokens, the logits of the next token
// are bit-identical (same kernels, same batch shapes). A third run without any rollback (prefix + 2 tokens in one
// batch of 2) shows the ordinary batch-shape noise for scale.
static int rollback_test(const std::string & model, const std::string & res, const std::vector<int32_t> & toks,
                         int gpu_layers) {
    model::LoadOptions o;
    o.n_gpu_layers  = gpu_layers;
    o.n_ctx         = 512;
    o.residual_path = res;
    model::Qwen35 m;
    std::string   err;
    if (!m.load(model, o, err) || toks.size() < 48) {
        fprintf(stderr, "load failed or too few tokens: %s\n", err.c_str());
        return 1;
    }
    const size_t       nv = (size_t) m.hp().n_vocab;
    const int          P  = 32;
    std::vector<float> la(nv), lb(nv), lc(nv);
    auto run = [&](const int32_t * batch5, float * out, bool no_rollback) {
        m.reset();
        bool ok = m.eval(toks.data(), P, model::EvalOpts{}, nullptr, nullptr, err);
        if (no_rollback) {
            ok = ok && m.eval(batch5, 2, model::EvalOpts{}, nullptr, nullptr, err);
        } else {
            m.save_state();
            model::EvalOpts rec;
            rec.record = true;
            ok = ok && m.eval(batch5, 5, rec, nullptr, nullptr, err) && m.rollback(2, err);
        }
        return ok && m.eval(&toks[P + 10], 1, model::EvalOpts{}, out, nullptr, err);
    };
    const int32_t b1[5] = { toks[P], toks[P + 1], toks[P + 2], toks[P + 3], toks[P + 4] };
    const int32_t b2[5] = { toks[P], toks[P + 1], toks[P + 7], toks[P + 8], toks[P + 9] };
    if (!run(b1, la.data(), false) || !run(b2, lb.data(), false) || !run(b1, lc.data(), true)) {
        fprintf(stderr, "eval failed: %s\n", err.c_str());
        return 1;
    }
    double dab = 0, dac = 0;
    for (size_t i = 0; i < nv; i++) {
        dab = std::max(dab, (double) std::fabs(la[i] - lb[i]));
        dac = std::max(dac, (double) std::fabs(la[i] - lc[i]));
    }
    printf("rollback: max |logit diff| between runs with different discarded tokens: %.3g (%s)\n", dab,
           dab == 0 ? "bit-identical, PASS" : "FAIL");
    printf("          vs no-rollback run (batch-shape noise only): %.3g\n", dac);
    return dab == 0 ? 0 : 2;
}

int selftest(const std::vector<std::string> & args) {
    std::string model, tokens_path, res;
    int         half = 64, gpu_layers = 0;
    bool        rollback = false;
    for (size_t i = 2; i < args.size(); i++) {
        const std::string & k   = args[i];
        auto                val = [&]() -> std::string { return i + 1 < args.size() ? args[++i] : std::string(); };
        if (k == "--tokens") tokens_path = val();
        else if (k == "--half") half = std::atoi(val().c_str());
        else if (k == "--gpu-layers") gpu_layers = std::atoi(val().c_str());
        else if (k == "--res") res = val();
        else if (k == "--rollback") rollback = true;
        else if (model.empty() && k[0] != '-') model = k;
    }
    if (rollback) {
        return rollback_test(model, res, read_token_ids(tokens_path), gpu_layers);
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
