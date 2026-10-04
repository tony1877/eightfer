#pragma once

// What the generic tools (ppl, logits) need from a model, so they work for every architecture.

#include <cstdint>
#include <memory>
#include <string>

namespace e8::model {

struct LoadOptions;

class CausalLM {
public:
    virtual ~CausalLM() = default;

    // Runs `n` tokens at positions [n_past(), n_past() + n); writes n * n_vocab logits when `logits` is non-null.
    virtual bool eval(const int32_t * tokens, int n, float * logits, std::string & err) = 0;
    virtual void reset()                                                                 = 0;

    virtual int64_t  n_vocab() const          = 0;
    virtual int      n_past() const           = 0;
    virtual int      n_ctx() const            = 0;
    virtual int64_t  n_layer() const          = 0;
    virtual bool     add_bos() const          = 0;
    virtual int64_t  bos_token() const        = 0;
    virtual uint64_t gpu_weight_bytes() const = 0;
    virtual uint64_t cpu_weight_bytes() const = 0;
};

// Picks the implementation from general.architecture (qwen35, qwen4exp).
std::unique_ptr<CausalLM> load_causal_lm(const std::string & path, const LoadOptions & opt, std::string & err);

} // namespace e8::model
