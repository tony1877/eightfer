#pragma once

// Qwen3.8-Flash-Next ("qwen4exp" in GGUF), single sequence, base weights only (DESIGN.md M4). Mirrors llama.cpp's
// src/models/qwen4exp.cpp at the pinned commit and transformers' modeling_qwen4_exp.py:
//   - hyper-connections: the residual is hc parallel streams [n_embd, hc, T]; each block reads a gated mix of them
//     and writes back with per-stream injection weights; the final mixer replaces the output norm
//   - Gated DeltaNet in 3 of 4 layers (sigmoid output gate), gated full attention in the 4th with QSA sparse
//     attention: each token attends to the top budget/r blocks of r tokens, scored by a small indexer on pooled keys,
//     plus its own incomplete block (dense while the context fits the budget)
//   - MoE FFN (softmax router, top-k renormalized) plus a sigmoid-gated shared expert
//   - PLE: hashed n-gram embeddings (computed on the host from the token history) injected in one layer through a
//     gated value and a dilated causal depthwise conv
// Weights come from a memory-mapped (possibly split) GGUF. Tensors placed on the CPU are used in place from the
// mapping (no RAM copy), so the expert weights and the n-gram table can be far larger than RAM.

#include "model/causal_lm.h"
#include "model/gguf_set.h"
#include "model/qwen35.h"  // LoadOptions

#include "ggml-backend.h"
#include "ggml.h"

#include <cstdint>
#include <string>
#include <vector>

namespace e8::model {

struct Qwen4ExpHparams {
    int64_t n_layer = 0, n_embd = 0, n_head = 0, n_head_kv = 0, head_dim = 0, n_vocab = 0, n_ctx_train = 0, n_rot = 0;
    int     rope_sections[4] = { 0, 0, 0, 0 };
    float   rope_freq_base = 1e7f, rms_eps = 1e-6f, expert_weights_scale = 0.0f;
    // DeltaNet
    int64_t ssm_d_conv = 0, ssm_d_inner = 0, ssm_d_state = 0, ssm_n_v = 0, ssm_n_k = 0;
    // MoE
    int64_t n_expert = 0, n_expert_used = 0, n_ff_exp = 0, n_ff_shexp = 0;
    // hyper-connections
    int64_t hc = 0, hc_lr = 0;
    // QSA indexer
    int64_t idx_n_head = 0, idx_dim = 0, idx_budget = 0;
    std::vector<int64_t> compress;  // per layer, 0 = dense attention
    // PLE
    std::vector<bool>     ple_layer;
    int64_t               ple_ngram = 0, ple_per_gram = 0, ple_conv_kernel = 0, ple_eos = 0, ple_head_dim = 0;
    std::vector<uint64_t> ple_mult, ple_offset, ple_vocab;
    std::vector<bool>     recurrent;
    bool                  add_bos = false;
    int64_t               bos     = -1;

    int64_t conv_channels() const { return ssm_d_inner + 2 * ssm_n_k * ssm_d_state; }
    int64_t ple_n_heads() const { return (ple_ngram - 1) * ple_per_gram; }

    bool load(const GgufSet & f, std::string & err);
};

struct Qwen4ExpLayer {
    // hyper-connection mixers (before the token mixer, before the FFN)
    ggml_tensor * hca_norm = nullptr, * hca_down = nullptr, * hca_up = nullptr, * hca_inject = nullptr;
    ggml_tensor * hcf_norm = nullptr, * hcf_down = nullptr, * hcf_up = nullptr, * hcf_inject = nullptr;
    // attention + QSA indexer
    ggml_tensor * wq = nullptr, * wk = nullptr, * wv = nullptr, * wo = nullptr, * q_norm = nullptr, * k_norm = nullptr;
    ggml_tensor * idx_q = nullptr, * idx_k = nullptr, * idx_q_norm = nullptr, * idx_k_norm = nullptr;
    // gated delta net
    ggml_tensor * wqkv = nullptr, * wz = nullptr, * conv1d = nullptr, * dt_bias = nullptr, * a = nullptr;
    ggml_tensor * w_beta = nullptr, * w_alpha = nullptr, * ssm_norm = nullptr, * ssm_out = nullptr;
    // PLE
    ggml_tensor * ple_key = nullptr, * ple_value = nullptr, * ple_norm_key = nullptr, * ple_norm_query = nullptr;
    ggml_tensor * ple_norm_conv = nullptr, * ple_conv1d = nullptr;
    // MoE
    ggml_tensor * gate_inp = nullptr, * up_exps = nullptr, * gate_exps = nullptr, * down_exps = nullptr;
    ggml_tensor * gate_up_exps = nullptr;
    ggml_tensor * gate_inp_shexp = nullptr, * gate_shexp = nullptr, * up_shexp = nullptr, * down_shexp = nullptr;
    bool          on_gpu = false;
};

class Qwen4Exp : public CausalLM {
public:
    Qwen4Exp();
    ~Qwen4Exp() override;

    bool load(const std::string & path, const LoadOptions & opt, std::string & err);

    bool eval(const int32_t * tokens, int n, float * logits, std::string & err) override;
    void reset() override;

    int64_t  n_vocab() const override { return hp_.n_vocab; }
    int      n_past() const override { return n_past_; }
    int      n_ctx() const override { return n_ctx_; }
    int64_t  n_layer() const override { return hp_.n_layer; }
    bool     add_bos() const override { return hp_.add_bos; }
    int64_t  bos_token() const override { return hp_.bos; }
    uint64_t gpu_weight_bytes() const override { return gpu_bytes_; }
    uint64_t cpu_weight_bytes() const override { return cpu_bytes_; }
    const Qwen4ExpHparams & hp() const { return hp_; }

private:
    struct Inputs {
        ggml_tensor * embd = nullptr, * pos = nullptr, * mask = nullptr, * ple = nullptr;
        // QSA (sparse batches only)
        ggml_tensor * bvis = nullptr, * tail = nullptr, * pool_pos = nullptr;
    };
    ggml_cgraph * build_graph(ggml_context * ctx, int n, int n_kv, bool sparse, Inputs & in, ggml_tensor *& out);
    ggml_tensor * hc_mix(ggml_context * ctx, ggml_tensor * x, ggml_tensor * wn, ggml_tensor * wd, ggml_tensor * wu,
                         ggml_tensor * wi, ggml_tensor ** inject);
    ggml_tensor * hc_combine(ggml_context * ctx, ggml_tensor * res, ggml_tensor * out, ggml_tensor * inject);
    ggml_tensor * gdn(ggml_context * ctx, ggml_cgraph * gf, int64_t il, ggml_tensor * cur, int n);
    ggml_tensor * attn(ggml_context * ctx, ggml_cgraph * gf, int64_t il, ggml_tensor * cur, int n, int n_kv, bool sparse,
                       Inputs & in);
    ggml_tensor * moe(ggml_context * ctx, int64_t il, ggml_tensor * cur, int n);
    ggml_tensor * ple(ggml_context * ctx, ggml_cgraph * gf, int64_t il, ggml_tensor * res, ggml_tensor * emb, int n);
    // n-gram table rows for `n` tokens at the current position, gathered and dequantized: [n_embd per token]
    void          ple_rows(const int32_t * tokens, int n, std::vector<float> & out) const;

    Qwen4ExpHparams         hp_;
    GgufSet                 file_;
    ggml_backend_t          gpu_ = nullptr, cpu_ = nullptr;
    ggml_backend_sched_t    sched_ = nullptr;
    ggml_context *          wctx_[2] = { nullptr, nullptr };  // 0 = GPU copies, 1 = CPU tensors over the mapping
    ggml_backend_buffer_t   wbuf_gpu_ = nullptr;
    std::vector<ggml_backend_buffer_t> map_bufs_;  // one CPU buffer per mapped file
    ggml_context *          sctx_[2] = { nullptr, nullptr };
    ggml_backend_buffer_t   sbuf_[2] = { nullptr, nullptr };

    const ggml_tensor *     tok_embd_ = nullptr, * ple_table_ = nullptr;  // looked up on the host from the mapping
    ggml_tensor *           output_ = nullptr, * out_hc_norm_ = nullptr, * out_hc_down_ = nullptr, * out_hc_up_ = nullptr;
    std::vector<Qwen4ExpLayer> layers_;
    // state per layer: KV + indexer key caches (attention), conv/ssm state (DeltaNet), PLE conv history
    std::vector<ggml_tensor *> k_cache_, v_cache_, idx_raw_, idx_pool_, conv_state_, ssm_state_, ple_state_;
    std::vector<int32_t>    history_;  // every evaluated token, for the n-gram hash

    LoadOptions          opt_;
    int                  n_ctx_ = 0, n_past_ = 0, kpool_ = 0;
    uint64_t             gpu_bytes_ = 0, cpu_bytes_ = 0;
    std::vector<uint8_t> graph_meta_;
    ggml_tensor *        dbg_mask_ = nullptr, * dbg_scores_ = nullptr;
};

} // namespace e8::model
