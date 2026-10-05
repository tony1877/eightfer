#pragma once

// Self-speculative decoding (DESIGN.md section 4): the base weights alone draft k tokens greedily, base + residual
// verify them in one batch. Acceptance follows speculative sampling (Leviathan et al. 2023) specialised to a
// deterministic draft: accept draft d with probability p(d), else sample from p with d removed. The emitted tokens
// therefore follow the base + residual model's distribution exactly (greedy: identical tokens).

#include "model/qwen35.h"

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
    int64_t cycles = 0, drafted = 0, accepted = 0, emitted = 0;
    int64_t k_hist[64] = {};  // cycles per chosen k (0..SpecDecoder::kMaxK)
    double  t_draft = 0, t_verify = 0, t_rollback = 0, t_prefill = 0;
    int64_t prefill_tokens = 0;
    int64_t mtp_proposed = 0, mtp_accepted = 0;  // MTP proposals checked by the base / kept
};

class SpecDecoder {
public:
    // k = drafts per cycle; 0 disables speculation (plain decoding with base + residual). With `adaptive`, k is the
    // upper bound and each cycle picks the k with the best expected tokens per second from the running acceptance
    // rate and the measured draft / verify costs.
    SpecDecoder(model::Qwen35 & m, int k, const SamplerParams & sp, bool adaptive = false);
    // MTP proposals per base pass while drafting (when the model has its MTP block); 0 drafts with the base alone
    void set_mtp(int n) { mtp_n_ = n > 0 ? n : 0; }

    // Evaluates the prompt (base + residual) and samples the first token, which is appended to `out`.
    bool prefill(const std::vector<int32_t> & prompt, std::vector<int32_t> & out, std::string & err);
    // When the caller has evaluated the prompt itself: samples the first token from the last prompt position's
    // logits and appends it to `out`.
    void begin(const float * last_logits, std::vector<int32_t> & out);
    // One cycle: appends 1..k+1 tokens to `out`.
    bool step(std::vector<int32_t> & out, std::string & err);

    const SpecStats & stats() const { return st_; }

    static constexpr int kMaxK = 63;  // most drafts per cycle (the model needs max_record >= kMaxK + 1)

private:
    // samples from logits (temperature/top-k/top-p/min-p); with `exclude` >= 0 that token's probability is removed
    // first (rejection resampling)
    int32_t sample(const float * logits, int32_t exclude);
    // probability of `tok` under the sampling distribution of these logits
    double  prob(const float * logits, int32_t tok);
    void    dist(const float * logits, std::vector<std::pair<float, int32_t>> & p);
    int     choose_k() const;
    double  verify_ms(int n) const;
    using Dist = std::vector<std::pair<float, int32_t>>;
    // drafts toks[1..k] (toks[0] = last_) with MTP rounds checked by the base; qd gets the base's distribution at
    // each draft when sampling
    bool    draft_mtp(int k, std::vector<int32_t> & toks, std::vector<Dist> & qd, TokenCounts & cur, std::string & err);
    void    emit(std::vector<int32_t> & out, int32_t t) {
        out.push_back(t);
        counts_[t]++;
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
    int                mtp_n_ = 3;  // MTP proposals per base pass (0 = draft with the base alone)
    SpecStats          st_;
};

} // namespace e8::runtime
