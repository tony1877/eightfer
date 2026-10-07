#pragma once

// Self-speculative decoding (DESIGN.md section 4): the base weights alone draft k tokens greedily, base + residual
// verify them in one batch. Acceptance follows speculative sampling (Leviathan et al. 2023) specialised to a
// deterministic draft: accept draft d with probability p(d), else sample from p with d removed. The emitted tokens
// therefore follow the base + residual model's distribution exactly (greedy: identical tokens).

#include "model/qwen35.h"

#include <atomic>
#include <cstdint>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

namespace e8::runtime {

struct SamplerParams {
    float    temp  = 0.0f;  // 0 = greedy
    float    top_p = 1.0f;
    int      top_k = 0;     // 0 = off
    float    min_p = 0.0f;
    uint64_t seed  = 0;
    // OpenAI penalties over the tokens generated so far: logit -= frequency * count + presence * (count > 0)
    float    presence_penalty  = 0.0f;
    float    frequency_penalty = 0.0f;
};

using TokenCounts = std::unordered_map<int32_t, int>;

// applies the presence / frequency penalties for `counts` to a row of logits
void apply_penalties(float * logits, const SamplerParams & sp, const TokenCounts & counts);

struct SpecStats {
    int64_t full_cycles = 0;  // cycles whose verify accepted every draft
    int64_t cycles = 0, drafted = 0, accepted = 0, emitted = 0;
    int64_t k_hist[64] = {};  // cycles per chosen k (0..SpecDecoder::kMaxK)
    double  t_draft = 0, t_verify = 0, t_rollback = 0, t_prefill = 0;
    int64_t prefill_tokens = 0;
    int64_t mtp_proposed = 0, mtp_accepted = 0;  // MTP proposals checked by the base / kept
    int64_t echo_proposed = 0, echo_accepted = 0;  // tokens copied from the context as proposals / kept
    int64_t long_cycles = 0, reruns = 0;  // cycles extended past k by a long copy / long verifies re-evaluated
    // draft rounds by proposal source, and where their time goes (seconds)
    int64_t rounds_echo = 0, rounds_mtp = 0, rounds_plain = 0;
    double  t_snap = 0, t_beval = 0, t_rback = 0, t_mtp = 0;
    // pipelined cycles (shadow model): first pre-drafted tokens tested / kept, time spent pre-drafting
    int64_t pipe_tested = 0, pipe_kept = 0, pipe_drafted = 0, pipe_rounds = 0;
    double  t_pipe = 0;
};

class SpecDecoder {
public:
    // k = drafts per cycle; 0 disables speculation (plain decoding with base + residual). With `adaptive`, k is the
    // upper bound and each cycle picks the k with the best expected tokens per second from the running acceptance
    // rate and the measured draft / verify costs.
    SpecDecoder(model::Qwen35 & m, int k, const SamplerParams & sp, bool adaptive = false);
    // MTP proposals per base pass while drafting (when the model has its MTP block); 0 drafts with the base alone
    void set_mtp(int n) { mtp_n_ = n > 0 ? n : 0; }
    // echo drafting: propose continuations of earlier occurrences of the last tokens (on by default)
    void set_echo(bool on) { echo_ = on; }
    // the tokens before the first emitted one (the prompt), for echo drafting; prefill() sets it itself
    void set_context(const std::vector<int32_t> & prompt) {
        hist_       = prompt;
        ngram_upto_ = 0;
        ngram_.clear();
    }

    // Evaluates the prompt (base + residual) and samples the first token, which is appended to `out`.
    bool prefill(const std::vector<int32_t> & prompt, std::vector<int32_t> & out, std::string & err);
    // When the caller has evaluated the prompt itself: samples the first token from the last prompt position's
    // logits and appends it to `out`.
    void begin(const float * last_logits, std::vector<int32_t> & out);
    // One cycle: appends 1..k+1 tokens to `out`.
    bool step(std::vector<int32_t> & out, std::string & err);
    // drafts on `d` (a shadow of the verifying model, see Qwen35::make_shadow), pipelined with the verifies
    void set_shadow(model::Qwen35 * d) {
        d_       = d;
        d_dirty_ = true;
        pending_ = false;
    }

    const SpecStats & stats() const { return st_; }

    static constexpr int kMaxK = 63;  // most drafts per cycle (the model needs max_record >= kMaxK + 1)

private:
    // samples from logits (temperature/top-k/top-p/min-p); with `exclude` >= 0 that token's probability is removed
    // first (rejection resampling)
    int32_t sample(const float * logits, int32_t exclude);
    // probability of `tok` under the sampling distribution of these logits
    double  prob(const float * logits, int32_t tok);
    void    dist(const float * logits, std::vector<std::pair<float, int32_t>> & p);
    void    dist_topk(const float * v, const int32_t * id, int k, std::vector<std::pair<float, int32_t>> & p);
    void    finish_dist(std::vector<std::pair<float, int32_t>> & p);
    bool    no_penalties() const { return sp_.presence_penalty == 0 && sp_.frequency_penalty == 0; }
    // top-k to take on the GPU when sampling needs only that (penalties change logits on the CPU), else 0
    int     gpu_topk() const { return sp_.temp > 0 && sp_.top_k > 0 && sp_.top_k <= 64 && no_penalties() ? sp_.top_k : 0; }
    bool    step_plain(std::vector<int32_t> & out, std::string & err);
    bool    step_shadow(std::vector<int32_t> & out, std::string & err);
    int     choose_k() const;
    int     choose_mtp() const;
    double  verify_ms(int n) const;
    using Dist = std::vector<std::pair<float, int32_t>>;
    // drafts toks[1..k] (toks[0] = last_) with echo / MTP rounds checked by the base; qd gets the base's
    // distribution at each draft when sampling
    bool    draft_rounds(int k, std::vector<int32_t> & toks, std::vector<Dist> & qd, TokenCounts & cur, std::string & err,
                         bool extend = false);
    static uint64_t ngram_key(const int32_t * t);
    int     find_copy(const int32_t * tail);
    void    emit(std::vector<int32_t> & out, int32_t t) {
        out.push_back(t);
        counts_[t]++;
        hist_.push_back(t);
    }

    static constexpr int    kGpuBatch    = 32;    // ggml's default batch size for running host weights on the GPU
    static constexpr double kPriorHazard = 0.03;  // prior P(reject at a draft position | reached it), from the 27B
    static constexpr double kPriorWeight = 2;

    model::Qwen35 &    m_;
    int                k_;
    bool               adaptive_ = false;
    // adaptive k: decayed per-position draft counts (index = draft position), draft ms per token, decayed verify ms
    // per batch size (index = k + 1)
    double             seen_[kMaxK + 1] = {}, fail_[kMaxK + 1] = {}, draft_ms_ = 25;
    double             vt_[kMaxK + 2] = {}, vn_[kMaxK + 2] = {};
    SamplerParams      sp_;
    std::mt19937_64    rng_;
    int32_t            last_ = -1;  // sampled, not yet evaluated
    std::vector<float> logits_;
    TokenCounts        counts_;  // tokens emitted so far (penalties)
    int                mtp_n_ = 6;  // most MTP proposals per base pass (choose_mtp picks; 0 = draft with the base alone)
    static constexpr int kMaxMtp = 8;
    double               mseen_[kMaxMtp + 1] = {}, mfail_[kMaxMtp + 1] = {};  // MTP proposal survival by position
    double               check_ms_ = 23, step_ms_ = 2.5;                    // base check / MTP step cost (measured)
    static constexpr int kEchoN = 8;  // echo: tokens that must match to propose a copy
    static constexpr int kBigK  = 127;  // most drafts of a verify extended by a long copy
    bool                 last_echo_full_ = false;  // the last draft round was a copy the base kept entirely
    bool                 echo_  = true;
    std::vector<int32_t> hist_;  // prompt + emitted tokens
    std::unordered_map<uint64_t, int> ngram_;  // kEchoN-gram -> history position after its latest occurrence
    int                  ngram_upto_ = 0;
    SpecStats          st_;
    // pipelining with a shadow model
    model::Qwen35 *      d_       = nullptr;
    bool                 d_dirty_ = true;   // the shadow must take over the main model's state before drafting
    bool                 no_echo_ = false;  // while pre-drafting
    const std::atomic<bool> * stop_ = nullptr;  // pre-drafting stops between rounds once this is set
    bool                 pending_ = false;  // ptoks_[1..] were pre-drafted during a verify that accepted everything
    std::vector<int32_t> ptoks_;
    std::vector<std::vector<std::pair<float, int32_t>>> pqd_;
    std::vector<std::pair<float, int32_t>>              pend_p_;  // that verify's last row (sampling)
    int32_t              pend_top_ = -1;                          // its argmax (greedy)
};

} // namespace e8::runtime
