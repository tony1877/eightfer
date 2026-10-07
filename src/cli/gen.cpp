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
#include <cmath>
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

model::Qwen35 * g_shadow = nullptr;  // --pipe

bool run_gen(model::Qwen35 & m, const std::vector<int32_t> & prompt, int n_gen, int k, bool adaptive, int mtp, bool echo,
             const runtime::SamplerParams & sp, GenResult & r, std::string & err) {
    m.reset();
    runtime::SpecDecoder dec(m, k, sp, adaptive);
    dec.set_mtp(mtp);
    dec.set_echo(echo);
    if (k > 0) dec.set_shadow(g_shadow);
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
    if (s.rounds_echo + s.rounds_mtp + s.rounds_plain > 0) {
        printf("  draft rounds: echo %lld, mtp %lld, plain %lld; draft time: snapshots %.0f ms, base checks %.0f ms, "
               "rollbacks %.0f ms, mtp %.0f ms, long cycles %lld, reruns %lld\n",
               (long long) s.rounds_echo, (long long) s.rounds_mtp, (long long) s.rounds_plain, 1e3 * s.t_snap,
               1e3 * s.t_beval, 1e3 * s.t_rback, 1e3 * s.t_mtp, (long long) s.long_cycles, (long long) s.reruns);
    }
    if (s.pipe_rounds > 0) {
        printf("  pre-drafting: %lld tokens in %lld rounds, %.1f ms (%.1f ms per round, %.1f per token)\n",
               (long long) s.pipe_drafted, (long long) s.pipe_rounds, 1e3 * s.t_pipe, 1e3 * s.t_pipe / s.pipe_rounds,
               1e3 * s.t_pipe / std::max<int64_t>(s.pipe_drafted, 1));
    }
    if (s.echo_proposed > 0) {
        printf("  echo: %lld tokens proposed from the context, %.3f kept by the base\n", (long long) s.echo_proposed,
               (double) s.echo_accepted / (double) s.echo_proposed);
    }
    if (s.mtp_proposed > 0) {
        printf("  MTP: %lld proposals, %.3f kept by the base\n", (long long) s.mtp_proposed,
               (double) s.mtp_accepted / (double) s.mtp_proposed);
    }
    if (k > 0) {
        printf("  k used:");
        for (int i = 1; i <= runtime::SpecDecoder::kMaxK; i++) {
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
    int mtp    = 6;  // most MTP proposals per round (adaptive below that)
    bool echo  = true;
    bool kv_q8 = false;
    bool                  compare = false, profile = false, adaptive = false;
    int                   repeat  = 1;
    bool                  pipe    = false;
    bool                  tree_test = false;  // --tree-test: a 3-sequence tree verify vs each sequence alone  // --pipe: draft on a shadow model while verifying  // --repeat N: N runs with seeds seed..seed+N-1, aggregate decode speed
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
            k                   = adaptive ? runtime::SpecDecoder::kMaxK : std::atoi(v.c_str());
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
        else if (a == "--mtp") mtp = std::atoi(val().c_str());
        else if (a == "--echo") echo = std::atoi(val().c_str()) != 0;
        else if (a == "--profile") profile = true;
        else if (a == "--pipe") pipe = true;
        else if (a == "--tree-test") tree_test = true;
        else if (a == "--repeat") repeat = std::max(1, std::atoi(val().c_str()));
        else if (model.empty() && a[0] != '-') model = a;
        else {
            fprintf(stderr, "unknown option: %s\n", a.c_str());
            return 1;
        }
    }
    const std::vector<int32_t> prompt = read_token_ids(tokens_path);
    if (model.empty() || prompt.empty() || n_gen < 1 || k < 0 || k > runtime::SpecDecoder::kMaxK) {
        fprintf(stderr, "usage: eightfer gen <base.gguf> [--res r.gguf] --tokens <ids.txt> [-n 128] [--spec K (0..63) | auto]\n"
                        "                    [--temp T] [--top-p P] [--top-k K] [--min-p P] [--seed S] [--gpu-layers N]\n"
                        "                    [--ctx N] [--threads N] [--mtp N (MTP proposals per base pass, 0 = off)] [--compare]\n");
        return 1;
    }
    model::LoadOptions o;
    o.n_gpu_layers  = gpu_layers;
    o.n_ctx         = std::max(n_ctx, (int) prompt.size() + n_gen + 32);
    o.gpu_kv = gpu_kv;
    if (kv_q8) o.kv_type = GGML_TYPE_Q8_0;
    o.n_threads     = threads;
    o.residual_path = res;
    o.max_record    = std::max(k + 1, 128);  // echo proposals (64) and tree verifies (up to 128 tokens)
    o.mtp           = k > 0 && mtp > 0;
    if (const char * ub = std::getenv("E8_UBATCH")) o.n_ubatch = std::atoi(ub);
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

    model::Qwen35 shadow;
    if (pipe) {
        if (!shadow.make_shadow(m, err)) {
            fprintf(stderr, "shadow: %s\n", err.c_str());
            return 1;
        }
        g_shadow = &shadow;
    }
    if (tree_test) {
        // prompt, then 3 sequences of L tokens (from the prompt itself, shifted): tree logits vs one at a time
        const int L = 8, B = 3;
        std::vector<int32_t> seqs((size_t) (B * L));
        for (int b = 0; b < B; b++)
            for (int j = 0; j < L; j++) {
                const int bs = (b + (std::getenv("E8_TT_ROT") ? std::atoi(std::getenv("E8_TT_ROT")) : 0)) % B;
                seqs[(size_t) (b * L + j)] = prompt[(size_t) ((bs * 7 + j) % prompt.size())];
            }
        const int64_t      nv = m.hp().n_vocab;
        std::vector<float> lt((size_t) (B * L) * nv), ls((size_t) L * nv);
        m.reset();
        if (!m.eval(prompt.data(), (int) prompt.size(), model::EvalOpts{}, nullptr, nullptr, err)) { fprintf(stderr, "%s\n", err.c_str()); return 1; }
        model::EvalOpts to;
        to.dry = to.record = true;
        to.n_seqs = B;
        if (!m.eval(seqs.data(), B * L, to, lt.data(), nullptr, err)) { fprintf(stderr, "tree eval: %s\n", err.c_str()); return 1; }
        if (!m.commit(L, err, 2)) { fprintf(stderr, "commit: %s\n", err.c_str()); return 1; }
        std::vector<float> nt((size_t) nv), ns2((size_t) nv);
        const int32_t probe = prompt[0];
        if (!m.eval(&probe, 1, model::EvalOpts{}, nt.data(), nullptr, err)) { fprintf(stderr, "%s\n", err.c_str()); return 1; }
        double worst = 0;
        for (int bb = 0; bb < B + 1; bb++) {
            const int b = bb == 0 ? 0 : bb - 1;  // sequence 0 twice: the first reference run after the tree eval
            m.reset();
            if (!m.eval(prompt.data(), (int) prompt.size(), model::EvalOpts{}, nullptr, nullptr, err)) return 1;
            if (!m.eval(seqs.data() + b * L, L, model::EvalOpts{}, ls.data(), nullptr, err)) { fprintf(stderr, "%s\n", err.c_str()); return 1; }
            double d = 0;
            for (size_t i = 0; i < ls.size(); i++) d = std::max(d, (double) std::fabs(ls[i] - lt[(size_t) b * L * nv + i]));
            printf("tree-test: sequence %d max |logit diff| %.4f; per row (diff, argmax same):", b, d);
            for (int j = 0; j < L; j++) {
                const float * a = ls.data() + (size_t) j * nv, * c = lt.data() + ((size_t) b * L + j) * nv;
                double dj = 0;
                for (int64_t v = 0; v < nv; v++) dj = std::max(dj, (double) std::fabs(a[v] - c[v]));
                printf(" (%.2f,%d)", dj, (int) (std::max_element(a, a + nv) - a == std::max_element(c, c + nv) - c));
            }
            printf("\n");
            worst = std::max(worst, d);
            if (b == 2) {
                if (!m.eval(&probe, 1, model::EvalOpts{}, ns2.data(), nullptr, err)) return 1;
                double d2 = 0;
                for (size_t i = 0; i < nt.size(); i++) d2 = std::max(d2, (double) std::fabs(nt[i] - ns2[i]));
                printf("tree-test: next token after committing sequence 2, max |logit diff| %.4f\n", d2);
                worst = std::max(worst, d2);
            }
        }
        {  // reference noise: the copied-prompt sequence, batch of L vs one token at a time
            const int bs = std::getenv("E8_TT_ROT") ? (B - std::atoi(std::getenv("E8_TT_ROT")) % B) % B : 0;
            m.reset();
            m.eval(prompt.data(), (int) prompt.size(), model::EvalOpts{}, nullptr, nullptr, err);
            m.eval(seqs.data() + bs * L, L, model::EvalOpts{}, ls.data(), nullptr, err);
            m.reset();
            m.eval(prompt.data(), (int) prompt.size(), model::EvalOpts{}, nullptr, nullptr, err);
            double d = 0;
            std::vector<float> one((size_t) nv);
            for (int j = 0; j < L; j++) {
                m.eval(seqs.data() + bs * L + j, 1, model::EvalOpts{}, one.data(), nullptr, err);
                for (int64_t v = 0; v < nv; v++) d = std::max(d, (double) std::fabs(one[(size_t) v] - ls[(size_t) j * nv + v]));
            }
            printf("tree-test: same sequence %d alone, batch %d vs one at a time: max |logit diff| %.4f\n", bs, L, d);
        }
        printf("tree-test: %s\n", worst < 0.5 ? "PASS" : "FAIL");
        return worst < 0.5 ? 0 : 3;
    }
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

    if (repeat > 1) {
        double tdec = 0, tdraft = 0, tver = 0;
        long long ntok = 0, ncyc = 0, nfull = 0, ptest = 0, pkept = 0, rr[21] = {}, tcyc = 0, ttry = 0, tres = 0, rk[3][16] = {}, dflag = 0, ddraft = 0, dhit = 0;
        for (int r = 0; r < repeat; r++) {
            runtime::SamplerParams s2 = sp;
            s2.seed                   = sp.seed + (uint64_t) r;
            GenResult g;
            if (!run_gen(m, prompt, n_gen, k, adaptive, mtp, echo, s2, g, err)) {
                fprintf(stderr, "generation failed: %s\n", err.c_str());
                return 1;
            }
            const double d = g.t_total - g.st.t_prefill;
            printf("run %d (seed %llu): %.2f tok/s, %lld cycles\n", r, (unsigned long long) s2.seed, (g.st.emitted - 1) / d,
                   (long long) g.st.cycles);
            tdec += d;
            ntok += g.st.emitted - 1;
            ncyc += g.st.cycles;
            nfull += g.st.full_cycles;
            ptest += g.st.pipe_tested;
            tcyc += g.st.tree_cycles;
            dflag += g.st.dis_flags;
            ddraft += g.st.dis_drafts;
            dhit += g.st.dis_hit;
            for (int i = 0; i < 16; i++) {
                rk[0][i] += g.st.rk_tok[i];
                rk[1][i] += g.st.rk_top[i];
                rk[2][i] += g.st.rk_ent[i];
            }
            ttry += g.st.tree_tried;
            tres += g.st.tree_rescued;
            pkept += g.st.pipe_kept;
            tdraft += g.st.t_draft;
            tver += g.st.t_verify;
            for (int i = 0; i <= 20; i++) rr[i] += g.st.rej_rank[i];
        }
        printf("full-accept cycles: %.1f%%\n", 100.0 * nfull / std::max<long long>(ncyc, 1));
        printf("pipelined: %lld first tokens tested, %lld kept\n", ptest, pkept);
        printf("tree: %lld cycles, %lld alternatives tested, %lld accepted\n", tcyc, ttry, tres);
        printf("MTP disagreement: %.1f flagged drafts per rejected cycle (of %.1f), rejection lands on one in %.0f%% of cycles\n",
               (double) dflag / std::max(1LL, (long long) (rr[0] + rr[1] + rr[2] + rr[3] + rr[4] + rr[5] + rr[6] + rr[7] + rr[8] + rr[9] + rr[10] + rr[11] + rr[12] + rr[13] + rr[14] + rr[15] + rr[16] + rr[17] + rr[18] + rr[19] + rr[20])),
               (double) ddraft / std::max(1LL, (long long) (rr[0] + rr[1] + rr[2] + rr[3] + rr[4] + rr[5] + rr[6] + rr[7] + rr[8] + rr[9] + rr[10] + rr[11] + rr[12] + rr[13] + rr[14] + rr[15] + rr[16] + rr[17] + rr[18] + rr[19] + rr[20])),
               100.0 * dhit / std::max(1LL, (long long) (rr[0] + rr[1] + rr[2] + rr[3] + rr[4] + rr[5] + rr[6] + rr[7] + rr[8] + rr[9] + rr[10] + rr[11] + rr[12] + rr[13] + rr[14] + rr[15] + rr[16] + rr[17] + rr[18] + rr[19] + rr[20])));
        static const char * nm[3] = { "drafted-token prob", "top prob", "-entropy" };
        for (int s = 0; s < 3; s++) {
            long long tot = 0, c = 0;
            for (long long v : rk[s]) tot += v;
            printf("rejection at the least-confident draft by %s: cumulative", nm[s]);
            for (int i = 0; i < 8; i++) {
                c += rk[s][i];
                printf(" top%d %.0f%%", i + 1, 100.0 * c / std::max(tot, 1LL));
            }
            printf("\n");
        }
        long long rt = 0;
        for (long long v : rr) rt += v;
        printf("rejections %lld; correction is the base's candidate #", rt);
        for (int i = 0; i < 6; i++) printf(" %d: %.1f%%", i + 1, 100.0 * rr[i] / std::max(rt, 1LL));
        long long top4 = rr[0] + rr[1] + rr[2] + rr[3], top8 = top4 + rr[4] + rr[5] + rr[6] + rr[7];
        printf(" | within top 4: %.1f%%, top 8: %.1f%%, outside top 20: %.1f%%\n", 100.0 * top4 / std::max(rt, 1LL),
               100.0 * top8 / std::max(rt, 1LL), 100.0 * rr[20] / std::max(rt, 1LL));
        printf("repeat %d: decode %.2f tok/s, %.2f tokens/cycle, per cycle draft %.1f ms verify %.1f ms\n", repeat, ntok / tdec,
               (double) ntok / std::max<long long>(ncyc, 1), 1e3 * tdraft / std::max<long long>(ncyc, 1),
               1e3 * tver / std::max<long long>(ncyc, 1));
        return 0;
    }
    GenResult a;
    if (!run_gen(m, prompt, n_gen, k, adaptive, mtp, echo, sp, a, err)) {
        fprintf(stderr, "generation failed: %s\n", err.c_str());
        return 1;
    }
    printf("tokens:");
    for (int32_t t : a.toks) printf(" %d", t);
    printf("\n");
    report(k > 0 ? "speculative" : "plain", a, k);

    if (compare && k > 0 && sp.temp <= 0) {
        GenResult b;
        if (!run_gen(m, prompt, n_gen, 0, false, 0, false, sp, b, err)) {
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
