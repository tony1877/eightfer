// `eightfer multitest <base.gguf> --res <res.gguf> --tokens ids.txt [--ctx 65536] [--slots 3]`: checks the joint verify
// (Qwen35::eval_multi) against single-sequence verifies. Three slots get prompts of different lengths (past the VRAM
// window, so attention is the sparse RAM path in both); each slot's next tokens are verified alone (dry eval +
// commit(0)), then all together; the rows are compared (KLD, top-1). Then commit_seq() keeps a prefix of each and the
// next token's logits are compared with a fresh prefill of prompt + kept tokens.

#include "cli/commands.h"
#include "model/qwen35.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace e8::cli {
namespace {

// KL(p || q) for two logit rows, and whether their argmax agrees
double kld(const float * a, const float * b, int64_t n, bool & same_top) {
    double ma = a[0], mb = b[0];
    int64_t ia = 0, ib = 0;
    for (int64_t i = 1; i < n; i++) {
        if (a[i] > ma) ma = a[i], ia = i;
        if (b[i] > mb) mb = b[i], ib = i;
    }
    double sa = 0, sb = 0;
    for (int64_t i = 0; i < n; i++) {
        sa += std::exp(a[i] - ma);
        sb += std::exp(b[i] - mb);
    }
    const double la = ma + std::log(sa), lb = mb + std::log(sb);
    double       k  = 0;
    for (int64_t i = 0; i < n; i++) {
        const double pa = std::exp(a[i] - la);
        k += pa * ((a[i] - la) - (b[i] - lb));
    }
    same_top = ia == ib;
    return k;
}

}  // namespace

int multitest(const std::vector<std::string> & args) {
    std::string base, res, tokens;
    int         n_ctx = 65536, slots = 3, kall = 0;
    for (size_t i = 2; i < args.size(); i++) {
        const std::string & a   = args[i];
        auto                val = [&]() -> std::string { return i + 1 < args.size() ? args[++i] : std::string(); };
        if (a == "--res") res = val();
        else if (a == "--tokens") tokens = val();
        else if (a == "--ctx") n_ctx = std::atoi(val().c_str());
        else if (a == "--slots") slots = std::atoi(val().c_str());
        else if (a == "--k") kall = std::atoi(val().c_str());
        else if (base.empty()) base = a;
    }
    if (base.empty() || tokens.empty() || slots < 2) {
        fprintf(stderr, "usage: eightfer multitest <base.gguf> --res <res.gguf> --tokens ids.txt [--ctx 65536] [--slots 3]\n");
        return 1;
    }
    const std::vector<int32_t> toks = read_token_ids(tokens);
    model::LoadOptions         o;
    o.n_gpu_layers = 999;
    o.n_ctx        = n_ctx;
    o.kv_type      = GGML_TYPE_Q8_0;
    o.residual_path = res;
    o.max_record   = 128;
    o.n_slots      = slots;
    o.mtp          = true;
    model::Qwen35 m;
    std::string   err;
    if (!m.load(base, o, err)) {
        fprintf(stderr, "load failed: %s\n", err.c_str());
        return 1;
    }
    const int64_t nv = m.n_vocab();
    // per slot: prompt [start, start + len), then k "drafts" (the text's own continuation) and the kept prefix
    struct S {
        int start, len, k, keep;
    };
    std::vector<S> ss = { { 0, 9000, 24, 10 }, { 20000, 12500, 31, 31 }, { 40000, 6400, 17, 1 } };
    ss.resize((size_t) std::min(slots, 3));
    for (S & s : ss) {
        if (kall > 0) s.k = kall, s.keep = std::min(s.keep, kall);
    }
    for (const S & s : ss) {
        if ((size_t) (s.start + s.len + s.k + 2) > toks.size()) {
            fprintf(stderr, "need %d tokens in %s\n", s.start + s.len + s.k + 2, tokens.c_str());
            return 1;
        }
    }
    auto prefill = [&](const int32_t * t, int n) { return m.prefill(t, n, nullptr, err); };
    // prompts, then each slot's verify alone (dry, committing nothing)
    std::vector<std::vector<float>> ref(ss.size());
    double                          single_ms = 0;
    for (size_t b = 0; b < ss.size(); b++) {
        const S & s = ss[b];
        if (!m.select_slot((int) b, err) || (m.reset(), false) || !prefill(toks.data() + s.start, s.len)) {
            fprintf(stderr, "slot %zu prefill: %s\n", b, err.c_str());
            return 1;
        }
        ref[b].resize((size_t) s.k * (size_t) nv);
        model::EvalOpts vo;
        vo.record = vo.dry = true;
        double best = 1e9;
        for (int rep = 0; rep < 3; rep++) {  // best of 3 (the first allocates the graph)
            const auto t0 = std::chrono::steady_clock::now();
            if (!m.eval(toks.data() + s.start + s.len, s.k, vo, ref[b].data(), nullptr, err) || !m.commit(0, err)) {
                fprintf(stderr, "slot %zu single verify: %s\n", b, err.c_str());
                return 1;
            }
            best = std::min(best, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        }
        single_ms += best;
        fprintf(stderr, "slot %zu: prompt %d tokens, verify %d alone: %.0f ms\n", b, s.len, s.k, best);
    }
    // all together
    std::vector<model::MultiSeq> seqs;
    int                          N = 0;
    for (size_t b = 0; b < ss.size(); b++) {
        seqs.push_back({ (int) b, toks.data() + ss[b].start + ss[b].len, ss[b].k });
        N += ss[b].k;
    }
    std::vector<float> lg((size_t) N * (size_t) nv);
    for (size_t nb = 1; nb < ss.size(); nb++) {  // cost by number of sequences
        std::vector<model::MultiSeq> sub(seqs.begin(), seqs.begin() + (long) nb);
        double                       t = 1e9;
        for (int rep = 0; rep < 3; rep++) {
            const auto t0 = std::chrono::steady_clock::now();
            if (!m.eval_multi(sub, 0, lg.data(), nullptr, err)) {
                fprintf(stderr, "eval_multi: %s\n", err.c_str());
                return 1;
            }
            t = std::min(t, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
            for (size_t b = 0; b < nb; b++) m.commit_seq((int) b, 0, err);
        }
        printf("verify time: %zu sequence(s) jointly %.0f ms\n", nb, t);
    }
    double             joint_ms = 1e9;
    for (int rep = 0; rep < 3; rep++) {  // best of 3; the last one stays for commit_seq below
        for (size_t b = 0; rep > 0 && b < ss.size(); b++) {
            if (!m.commit_seq((int) b, 0, err)) {
                fprintf(stderr, "commit_seq: %s\n", err.c_str());
                return 1;
            }
        }
        const auto t0 = std::chrono::steady_clock::now();
        if (!m.eval_multi(seqs, 0, lg.data(), nullptr, err)) {
            fprintf(stderr, "eval_multi: %s\n", err.c_str());
            return 1;
        }
        joint_ms = std::min(joint_ms, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }
    printf("verify time: %d tokens jointly %.0f ms; the %zu sequences one by one %.0f ms in all\n", N, joint_ms, ss.size(),
           single_ms);
    if (std::getenv("E8_MULTI_TOPK")) {  // the server's path: top-k rows on the GPU
        const int            tk = std::atoi(std::getenv("E8_MULTI_TOPK"));
        std::vector<float>   tv((size_t) N * (size_t) tk);
        std::vector<int32_t> ti((size_t) N * (size_t) tk);
        for (size_t b = 0; b < ss.size(); b++) m.commit_seq((int) b, 0, err);
        const auto t0 = std::chrono::steady_clock::now();
        if (!m.eval_multi(seqs, tk, tv.data(), ti.data(), err)) {
            fprintf(stderr, "eval_multi topk: %s\n", err.c_str());
            return 1;
        }
        printf("verify time with top-%d rows: %.0f ms\n", tk,
               std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        for (size_t b = 0; b < ss.size(); b++) m.commit_seq((int) b, 0, err);
        if (!m.eval_multi(seqs, 0, lg.data(), nullptr, err)) return 1;
    }
    int  fails = 0, off = 0;
    for (size_t b = 0; b < ss.size(); b++) {
        double kmax = 0, ksum = 0;
        int    same = 0;
        for (int i = 0; i < ss[b].k; i++) {
            bool         st = false;
            const double k  = kld(ref[b].data() + (size_t) i * nv, lg.data() + (size_t) (off + i) * nv, nv, st);
            kmax = std::max(kmax, k);
            ksum += k;
            same += st;
        }
        off += ss[b].k;
        const bool ok = kmax < 0.02 && same >= ss[b].k - 1;
        fails += !ok;
        printf("joint vs alone, slot %zu: mean KLD %.6f, max %.6f, same top-1 %d/%d %s\n", b, ksum / ss[b].k, kmax, same,
               ss[b].k, ok ? "ok" : "FAIL");
    }
    // keep a prefix of each, then the next token vs a fresh prefill of prompt + kept tokens
    for (size_t b = ss.size(); b-- > 0;) {  // any order; the last slot evaluated is active
        if (!m.commit_seq((int) b, ss[b].keep, err)) {
            fprintf(stderr, "commit_seq %zu: %s\n", b, err.c_str());
            return 1;
        }
    }
    std::vector<float> a((size_t) nv), r((size_t) nv);
    for (size_t b = 0; b < ss.size(); b++) {
        const S &     s   = ss[b];
        const int32_t nxt = toks[(size_t) (s.start + s.len + s.keep)];
        if (!m.select_slot((int) b, err) || m.n_past() != s.len + s.keep || !m.eval(&nxt, 1, a.data(), err)) {
            fprintf(stderr, "slot %zu after commit: n_past %d (want %d) %s\n", b, m.n_past(), s.len + s.keep, err.c_str());
            return 1;
        }
        m.reset();
        if (!prefill(toks.data() + s.start, s.len + s.keep) || !m.eval(&nxt, 1, r.data(), err)) {
            fprintf(stderr, "slot %zu fresh: %s\n", b, err.c_str());
            return 1;
        }
        bool         st = false;
        const double k  = kld(r.data(), a.data(), nv, st);
        const bool   ok = k < 0.02 && st;
        fails += !ok;
        printf("after commit_seq(%d), slot %zu: next-token KLD vs fresh prefill %.6f, same top-1 %d %s\n", s.keep, b, k, st,
               ok ? "ok" : "FAIL");
    }
    printf(fails ? "FAILED\n" : "PASSED\n");
    return fails ? 1 : 0;
}

}  // namespace e8::cli
