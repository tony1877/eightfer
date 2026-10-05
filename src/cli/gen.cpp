// `eightfer gen <base.gguf> [--res r.gguf] --tokens prompt_ids.txt [-n 128] [--spec K] [--temp T] [--top-p P]
//               [--top-k K] [--min-p P] [--seed S] [--gpu-layers N] [--ctx N] [--compare]`
//
// Generates from a prompt of token ids with self-speculative decoding (K drafts per cycle from the base, verified by
// base + residual; --spec 0 = plain decoding). Prints the generated ids and timing / acceptance statistics.
// --compare (greedy only) also decodes without speculation and reports whether the two outputs are identical.

#include "cli/commands.h"
#include "model/qwen35.h"
#include "runtime/spec.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace e8::cli {

namespace {

struct GenResult {
    std::vector<int32_t> toks;
    runtime::SpecStats   st;
    double               t_total = 0;
};

bool run_gen(model::Qwen35 & m, const std::vector<int32_t> & prompt, int n_gen, int k, bool adaptive,
             const runtime::SamplerParams & sp, GenResult & r, std::string & err) {
    m.reset();
    runtime::SpecDecoder dec(m, k, sp, adaptive);
    const auto           t0 = std::chrono::steady_clock::now();
    if (!dec.prefill(prompt, r.toks, err)) return false;
    while ((int) r.toks.size() < n_gen) {
        if (!dec.step(r.toks, err)) return false;
    }
    r.toks.resize((size_t) n_gen);
    r.t_total = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    r.st      = dec.stats();
    return true;
}

void report(const char * label, const GenResult & r, int k) {
    const auto & s      = r.st;
    const double decode = r.t_total - s.t_prefill;
    printf("%s: prefill %lld tok in %.2f s (%.1f tok/s); decode %lld tok in %.2f s = %.2f tok/s\n", label,
           (long long) s.prefill_tokens, s.t_prefill, s.prefill_tokens / std::max(s.t_prefill, 1e-9),
           (long long) (s.emitted - 1), decode, (s.emitted - 1) / std::max(decode, 1e-9));
    if (k > 0 && s.cycles > 0) {
        printf("  %lld cycles, acceptance %.3f (%lld/%lld), %.2f tokens/cycle; per cycle: draft %.1f ms, verify %.1f ms, "
               "rollback %.1f ms\n",
               (long long) s.cycles, (double) s.accepted / std::max<int64_t>(s.drafted, 1), (long long) s.accepted,
               (long long) s.drafted, (double) (s.emitted - 1) / s.cycles, 1e3 * s.t_draft / s.cycles,
               1e3 * s.t_verify / s.cycles, 1e3 * s.t_rollback / s.cycles);
    } else if (s.cycles > 0) {
        printf("  %.1f ms per token\n", 1e3 * s.t_verify / s.cycles);
    }
    if (k > 0) {
        printf("  k used:");
        for (int i = 1; i <= 16; i++) {
            if (s.k_hist[i]) printf(" %d x%lld", i, (long long) s.k_hist[i]);
        }
        printf("\n");
    }
}

} // namespace

int gen(const std::vector<std::string> & args) {
    std::string           model, res, tokens_path;
    int                   n_gen = 128, k = 6, gpu_layers = 999, n_ctx = 4096, threads = 0;
    int gpu_kv = -1;
    bool kv_q8 = false;
    bool                  compare = false, profile = false, adaptive = false;
    runtime::SamplerParams sp;
    for (size_t i = 2; i < args.size(); i++) {
        const std::string & a   = args[i];
        auto                val = [&]() -> std::string { return i + 1 < args.size() ? args[++i] : std::string(); };
        if (a == "--res") res = val();
        else if (a == "--tokens") tokens_path = val();
        else if (a == "-n") n_gen = std::atoi(val().c_str());
        else if (a == "--spec") {
            const std::string v = val();
            adaptive            = v == "auto";
            k                   = adaptive ? 12 : std::atoi(v.c_str());
        }
        else if (a == "--temp") sp.temp = (float) std::atof(val().c_str());
        else if (a == "--top-p") sp.top_p = (float) std::atof(val().c_str());
        else if (a == "--top-k") sp.top_k = std::atoi(val().c_str());
        else if (a == "--min-p") sp.min_p = (float) std::atof(val().c_str());
        else if (a == "--seed") sp.seed = (uint64_t) std::atoll(val().c_str());
        else if (a == "--gpu-layers") gpu_layers = std::atoi(val().c_str());
        else if (a == "--ctx") n_ctx = std::atoi(val().c_str());
        else if (a == "--gpu-kv") gpu_kv = std::atoi(val().c_str());
        else if (a == "--kv") kv_q8 = val() == "q8_0";
        else if (a == "--threads") threads = std::atoi(val().c_str());
        else if (a == "--compare") compare = true;
        else if (a == "--profile") profile = true;
        else if (model.empty() && a[0] != '-') model = a;
        else {
            fprintf(stderr, "unknown option: %s\n", a.c_str());
            return 1;
        }
    }
    const std::vector<int32_t> prompt = read_token_ids(tokens_path);
    if (model.empty() || prompt.empty() || n_gen < 1 || k < 0 || k > 15) {
        fprintf(stderr, "usage: eightfer gen <base.gguf> [--res r.gguf] --tokens <ids.txt> [-n 128] [--spec K (0..15) | auto]\n"
                        "                    [--temp T] [--top-p P] [--top-k K] [--min-p P] [--seed S] [--gpu-layers N]\n"
                        "                    [--ctx N] [--threads N] [--compare]\n");
        return 1;
    }
    model::LoadOptions o;
    o.n_gpu_layers  = gpu_layers;
    o.n_ctx         = std::max(n_ctx, (int) prompt.size() + n_gen + 32);
    o.gpu_kv = gpu_kv;
    if (kv_q8) o.kv_type = GGML_TYPE_Q8_0;
    o.n_threads     = threads;
    o.residual_path = res;
    o.max_record    = 16;
    model::Qwen35 m;
    std::string   err;
    const auto    tl = std::chrono::steady_clock::now();
    if (!m.load(model, o, err)) {
        fprintf(stderr, "load failed: %s\n", err.c_str());
        return 1;
    }
    printf("loaded in %.1f s: weights %.2f GB GPU + %.2f GB CPU, residual %.2f GB\n",
           std::chrono::duration<double>(std::chrono::steady_clock::now() - tl).count(), m.gpu_weight_bytes() / 1e9,
           m.cpu_weight_bytes() / 1e9, m.residual_bytes() / 1e9);

    if (profile) {
        // time a verify-sized eval (k+1 tokens) with and without the residual, from the same state each time
        m.reset();
        if (!m.eval(prompt.data(), (int) prompt.size(), model::EvalOpts{}, nullptr, nullptr, err)) {
            fprintf(stderr, "%s\n", err.c_str());
            return 1;
        }
        std::vector<int32_t> batch(prompt.end() - std::min<size_t>(prompt.size(), (size_t) k + 1), prompt.end());
        std::vector<float>   lg(batch.size() * (size_t) m.hp().n_vocab);
        for (int mode = 0; mode < 3; mode++) {
            model::EvalOpts o;
            o.residual = mode > 0;
            o.argmax   = mode == 0;
            o.record   = true;
            double best = 1e9, sum = 0;
            for (int rep = 0; rep < 5; rep++) {
                m.save_state();
                const auto t0 = std::chrono::steady_clock::now();
                int32_t    ids[16];
                if (!m.eval(batch.data(), (int) batch.size(), o, mode == 0 ? nullptr : lg.data(), ids, err)) {
                    fprintf(stderr, "%s\n", err.c_str());
                    return 1;
                }
                const double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                best = std::min(best, t);
                sum += t;
                m.restore_state();
            }
            static const char * names[] = { "base only, argmax", "base + residual, logits", "base + residual, logits" };
            printf("profile: %zu tokens, %s: best %.1f ms, mean %.1f ms\n", batch.size(), names[mode], best * 1e3, sum / 5 * 1e3);
        }
        return 0;
    }

    GenResult a;
    if (!run_gen(m, prompt, n_gen, k, adaptive, sp, a, err)) {
        fprintf(stderr, "generation failed: %s\n", err.c_str());
        return 1;
    }
    printf("tokens:");
    for (int32_t t : a.toks) printf(" %d", t);
    printf("\n");
    report(k > 0 ? "speculative" : "plain", a, k);

    if (compare && k > 0 && sp.temp <= 0) {
        GenResult b;
        if (!run_gen(m, prompt, n_gen, 0, false, sp, b, err)) {
            fprintf(stderr, "plain generation failed: %s\n", err.c_str());
            return 1;
        }
        report("plain", b, 0);
        size_t first = 0;
        while (first < a.toks.size() && a.toks[first] == b.toks[first]) first++;
        if (first == a.toks.size()) {
            printf("compare: IDENTICAL (%zu tokens); speedup %.2fx\n", a.toks.size(),
                   (b.t_total - b.st.t_prefill) / std::max(a.t_total - a.st.t_prefill, 1e-9));
        } else {
            printf("compare: first difference at generated token %zu (spec %d, plain %d)\n", first, a.toks[first], b.toks[first]);
            // near-tie or bug? replay the common prefix one token at a time and show the margin there
            std::vector<int32_t> ctx = prompt;
            ctx.insert(ctx.end(), a.toks.begin(), a.toks.begin() + (long long) first);
            m.reset();
            std::vector<float> lg((size_t) m.hp().n_vocab);
            for (size_t i = 0; i < ctx.size(); i++) {
                if (!m.eval(&ctx[i], 1, model::EvalOpts{}, i + 1 == ctx.size() ? lg.data() : nullptr, nullptr, err)) {
                    fprintf(stderr, "%s\n", err.c_str());
                    return 1;
                }
            }
            printf("  logits there (batch-1 replay): spec token %.5f, plain token %.5f, max %.5f\n", lg[(size_t) a.toks[first]],
                   lg[(size_t) b.toks[first]], *std::max_element(lg.begin(), lg.end()));
            return 3;
        }
    }
    return 0;
}

} // namespace e8::cli
