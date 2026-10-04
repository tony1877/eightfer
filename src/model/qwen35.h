#pragma once

// Qwen3.5/3.6/3.8 dense ("qwen35" in GGUF): Gated DeltaNet linear attention in 3 of every 4 layers, gated full
// attention with partial interleaved M-RoPE in the 4th, a SwiGLU FFN in every layer. Mirrors llama.cpp's
// src/models/qwen35.cpp at the pinned commit, single sequence only. The MTP block is not loaded.
//
// Split precision (DESIGN.md section 3/4): with LoadOptions::residual_path set, every big matrix W of the base GGUF
// gets a residual R from `eightfer pack`, kept in pinned host memory. eval() with EvalOpts::residual computes
// x*W^T + x*R^T for those matrices (R on the CPU for small batches; ggml's scheduler streams R to the GPU for big
// ones). Without it the base alone runs (the draft model).
//
// Speculation support: save_state()/restore_state() snapshot the recurrent state (KV needs no snapshot: entries past
// n_past are masked and get overwritten). An eval with EvalOpts::record keeps the DeltaNet inputs of that batch, so
// rollback(keep) can restore the snapshot and replay only the first `keep` tokens through the recurrence.

#include "model/gguf_file.h"

#include "ggml-backend.h"
#include "ggml.h"

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace e8::model {

struct Qwen35Hparams {
    int64_t n_layer        = 0;  // trunk layers (block_count minus MTP layers)
    int64_t n_layer_nextn  = 0;  // MTP layers stored after the trunk
    int64_t n_embd         = 0;
    int64_t n_ff           = 0;
    int64_t n_head         = 0;
    int64_t n_head_kv      = 0;
    int64_t head_dim       = 0;  // key_length == value_length
    int64_t n_vocab        = 0;
    int64_t n_ctx_train    = 0;
    int64_t n_rot          = 0;  // rope.dimension_count (partial rotary)
    int     rope_sections[4] = { 0, 0, 0, 0 };
    float   rope_freq_base = 10000.0f;
    float   rms_eps        = 1e-6f;
    int64_t full_attn_interval = 4;
    // Gated DeltaNet
    int64_t ssm_d_conv  = 0;  // conv kernel
    int64_t ssm_d_inner = 0;  // value dim = head_v_dim * n_v_heads
    int64_t ssm_d_state = 0;  // head dim of k and v
    int64_t ssm_n_v     = 0;  // time_step_rank: value heads
    int64_t ssm_n_k     = 0;  // group_count: key heads
    int64_t bos = -1, eos = -1;
    bool    add_bos = false;

    std::vector<bool> recurrent;  // attention.recurrent_layers when the GGUF has it (newer converters)

    bool is_recurrent(int64_t il) const {
        return il < (int64_t) recurrent.size() ? recurrent[(size_t) il] : (il + 1) % full_attn_interval != 0;
    }
    int64_t conv_channels() const { return ssm_d_inner + 2 * ssm_n_k * ssm_d_state; }

    // Fills from a GGUF with general.architecture == "qwen35". Returns false with err set.
    bool load(const GgufFile & f, std::string & err);
};

struct Qwen35Layer {
    ggml_tensor * attn_norm = nullptr, * post_norm = nullptr;
    // full attention
    ggml_tensor * wq = nullptr, * wk = nullptr, * wv = nullptr, * wo = nullptr, * q_norm = nullptr, * k_norm = nullptr;
    // gated delta net
    ggml_tensor * wqkv = nullptr, * wz = nullptr, * conv1d = nullptr, * dt_bias = nullptr, * a = nullptr;
    ggml_tensor * w_beta = nullptr, * w_alpha = nullptr, * ssm_norm = nullptr, * ssm_out = nullptr;
    // ffn
    ggml_tensor * ffn_gate = nullptr, * ffn_up = nullptr, * ffn_down = nullptr;
    bool on_gpu = false;
};

struct EvalOpts {
    bool residual  = true;   // add the residual (when loaded)
    bool record    = false;  // keep DeltaNet inputs so rollback() can replay a prefix of this batch
    bool last_only = false;  // logits for the last token only (prefill)
    bool argmax    = false;  // return argmax token ids instead of logits
};

struct LoadOptions {
    int       n_gpu_layers = 0;     // trunk layers on the GPU, counted from the last layer (as llama.cpp does)
    bool      output_gpu   = true;  // output norm + LM head on the GPU (when a GPU exists)
    int       n_ctx        = 4096;  // KV capacity, rounded up to a multiple of 256
    ggml_type kv_type      = GGML_TYPE_F16;
    int       n_threads    = 0;     // CPU threads; 0 = physical cores
    int       n_ubatch     = 512;   // most tokens per eval() call
    std::string residual_path;      // `eightfer pack` .res.gguf; empty = base only
    int       max_record   = 16;    // most tokens per recorded eval (speculative verify batch)
};

class Qwen35 {
public:
    Qwen35();
    ~Qwen35();

    bool load(const std::string & path, const LoadOptions & opt, std::string & err);

    // Runs `n` tokens (n <= n_ubatch) at positions [n_past(), n_past() + n). Writes n * n_vocab logits to
    // `logits` (row i = token i) when it is non-null, otherwise only advances the state.
    bool eval(const int32_t * tokens, int n, float * logits, std::string & err);

    // General form. Writes n (or 1 with last_only) rows of n_vocab logits to `logits`, or with opts.argmax the
    // argmax ids to `ids`. Either output pointer may be null.
    bool eval(const int32_t * tokens, int n, const EvalOpts & opts, float * logits, int32_t * ids, std::string & err);

    // Snapshot / restore the recurrent state and n_past.
    void save_state();
    void restore_state();
    // After a recorded eval of n tokens: restore the snapshot taken before it and re-run only the first `keep`
    // tokens through the DeltaNet layers (attention KV is kept as written). n_past becomes snapshot + keep.
    bool rollback(int keep, std::string & err);

    bool has_residual() const { return !res_.empty(); }
    uint64_t residual_bytes() const { return res_bytes_; }

    // Empties the KV cache and zeroes the recurrent state.
    void reset();

    // Debug: when >= 0, eval() stops after this layer and writes its output (n * n_embd floats) instead of logits.
    void set_debug_layer(int il) { debug_layer_ = il; }

    int                   n_past() const { return n_past_; }
    int                   n_ctx() const { return n_ctx_; }
    const Qwen35Hparams & hp() const { return hp_; }
    // Weight bytes placed on the GPU and on the CPU.
    uint64_t gpu_weight_bytes() const { return gpu_bytes_; }
    uint64_t cpu_weight_bytes() const { return cpu_bytes_; }

private:
    ggml_cgraph * build_graph(ggml_context * ctx, int n, const EvalOpts & o, ggml_tensor *& inp_tok,
                              ggml_tensor *& inp_pos, ggml_tensor *& inp_mask, ggml_tensor *& out, int n_kv);
    // DeltaNet recurrence for layer il on pre-conv qkv [C, n, 1], g/beta [1, Hv, n, 1]: updates conv and ssm state,
    // returns the attention output [S, Hv, n, 1].
    ggml_tensor * gdn_core(ggml_context * ctx, ggml_cgraph * gf, int64_t il, ggml_tensor * qkv, ggml_tensor * g,
                           ggml_tensor * beta, int n);
    // x*W^T, plus x*R^T when the residual is active and W has one
    ggml_tensor * mm(ggml_context * ctx, ggml_tensor * w, ggml_tensor * x);
    bool load_residual(const std::string & path, std::string & err);
    bool compute(ggml_context * ctx, ggml_cgraph * gf, std::string & err);

    Qwen35Hparams            hp_;
    GgufFile                 file_;
    ggml_backend_t           gpu_ = nullptr, cpu_ = nullptr;
    ggml_backend_sched_t     sched_ = nullptr;
    // weights and state live in these contexts/buffers; index 0 = GPU, 1 = CPU
    ggml_context *           wctx_[2] = { nullptr, nullptr };
    ggml_backend_buffer_t    wbuf_[2] = { nullptr, nullptr };
    ggml_context *           sctx_[2] = { nullptr, nullptr };
    ggml_backend_buffer_t    sbuf_[2] = { nullptr, nullptr };

    ggml_tensor *            tok_embd_ = nullptr, * out_norm_ = nullptr, * output_ = nullptr;
    std::vector<Qwen35Layer> layers_;
    // per layer: KV cache (attention) or conv/ssm state (DeltaNet); unused entries are nullptr
    std::vector<ggml_tensor *> k_cache_, v_cache_, conv_state_, ssm_state_;
    // snapshot of conv/ssm state, and the recorded DeltaNet inputs of the last recorded eval
    std::vector<ggml_tensor *> conv_bak_, ssm_bak_, rec_qkv_, rec_g_, rec_beta_;

    // residual: base weight -> residual tensor (pinned host memory)
    std::unordered_map<const ggml_tensor *, ggml_tensor *> res_;
    ggml_context *        rctx_ = nullptr;
    ggml_backend_buffer_t rbuf_ = nullptr;
    uint64_t              res_bytes_ = 0;
    bool                  use_res_   = false;  // while building a graph

    LoadOptions opt_;
    int         n_ctx_  = 0;
    int         n_past_ = 0;
    int         saved_n_past_ = 0;
    int         recorded_n_ = 0;  // tokens in the last recorded eval (0 = none)
    int         debug_layer_ = -1;
    uint64_t    gpu_bytes_ = 0, cpu_bytes_ = 0;
    std::vector<uint8_t> graph_meta_;  // memory for the per-eval graph context
};

} // namespace e8::model
