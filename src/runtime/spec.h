#pragma once

// Self-speculative decoding (DESIGN.md section 4): the base weights alone draft k tokens greedily, base + residual
// verify them in one batch. Acceptance follows speculative sampling (Leviathan et al. 2023) specialised to a
// deterministic draft: accept draft d with probability p(d), else sample from p with d removed. The emitted tokens
// therefore follow the base + residual model's distribution exactly (greedy: identical tokens).

#include "model/qwen35.h"

#include <cstdint>
#include <random>
#include <string>
#include <vector>

namespace e8::runtime {

struct SamplerParams {
    float    temp  = 0.0f;  // 0 = greedy
    float    top_p = 1.0f;
    int      top_k = 0;     // 0 = off
    float    min_p = 0.0f;
    uint64_t seed  = 0;
};

struct SpecStats {
    int64_t cycles = 0, drafted = 0, accepted = 0, emitted = 0;
    int64_t k_hist[17] = {};  // cycles per chosen k
    double  t_draft = 0, t_verify = 0, t_rollback = 0, t_prefill = 0;
    int64_t prefill_tokens = 0;
};

class SpecDecoder {
public:
    // k = drafts per cycle; 0 disables speculation (plain decoding with base + residual). With `adaptive`, k is the
    // upper bound and each cycle picks the k with the best expected tokens per second from the running acceptance
    // rate and the measured draft / verify costs.
    SpecDecoder(model::Qwen35 & m, int k, const SamplerParams & sp, bool adaptive = false);

    // Evaluates the prompt (base + residual) and samples the first token, which is appended to `out`.
    bool prefill(const std::vector<int32_t> & prompt, std::vector<int32_t> & out, std::string & err);
    // One cycle: appends 1..k+1 tokens to `out`.
    bool step(std::vector<int32_t> & out, std::string & err);

    const SpecStats & stats() const { return st_; }

private:
    // samples from logits (temperature/top-k/top-p/min-p); with `exclude` >= 0 that token's probability is removed
    // first (rejection resampling)
    int32_t sample(const float * logits, int32_t exclude);
    // probability of `tok` under the sampling distribution of these logits
    double  prob(const float * logits, int32_t tok);
    void    dist(const float * logits, std::vector<std::pair<float, int32_t>> & p);
    int     choose_k() const;

    model::Qwen35 &    m_;
    int                k_;
    bool               adaptive_ = false;
    // adaptive k: decayed acceptance counts, draft ms per token, verify ms ~ a + b*k (decayed least squares)
    double             acc_n_ = 9, rej_n_ = 1, draft_ms_ = 25;
    double             vs_ = 0, vsk_ = 0, vskk_ = 0, vst_ = 0, vskt_ = 0;
    SamplerParams      sp_;
    std::mt19937_64    rng_;
    int32_t            last_ = -1;  // sampled, not yet evaluated
    std::vector<float> logits_;
    SpecStats          st_;
};

} // namespace e8::runtime
