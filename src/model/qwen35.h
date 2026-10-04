#pragma once

// Qwen3.5/3.6/3.8 dense ("qwen35" in GGUF): Gated DeltaNet linear attention in 3 of every 4 layers, gated full
// attention with partial interleaved M-RoPE in the 4th, a SwiGLU FFN in every layer. Mirrors llama.cpp's
// src/models/qwen35.cpp at the pinned commit, single sequence only. The MTP block is not loaded (M3).

#include "model/gguf_file.h"

#include "ggml-backend.h"
#include "ggml.h"

#include <cstdint>
#include <memory>
#include <string>
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

    bool    is_recurrent(int64_t il) const { return (il + 1) % full_attn_interval != 0; }
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

struct LoadOptions {
    int       n_gpu_layers = 0;     // trunk layers on the GPU, counted from the last layer (as llama.cpp does)
    bool      output_gpu   = true;  // output norm + LM head on the GPU (when a GPU exists)
    int       n_ctx        = 4096;  // KV capacity, rounded up to a multiple of 256
    ggml_type kv_type      = GGML_TYPE_F16;
    int       n_threads    = 0;     // CPU threads; 0 = physical cores
    int       n_ubatch     = 512;   // most tokens per eval() call
};

class Qwen35 {
public:
    Qwen35();
    ~Qwen35();

    bool load(const std::string & path, const LoadOptions & opt, std::string & err);

    // Runs `n` tokens (n <= n_ubatch) at positions [n_past(), n_past() + n). Writes n * n_vocab logits to
    // `logits` (row i = token i) when it is non-null, otherwise only advances the state.
    bool eval(const int32_t * tokens, int n, float * logits, std::string & err);

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
    ggml_cgraph * build_graph(ggml_context * ctx, int n, ggml_tensor *& inp_tok, ggml_tensor *& inp_pos,
                              ggml_tensor *& inp_mask, ggml_tensor *& out_logits, int n_kv);

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

    LoadOptions opt_;
    int         n_ctx_  = 0;
    int         n_past_ = 0;
    int         debug_layer_ = -1;
    uint64_t    gpu_bytes_ = 0, cpu_bytes_ = 0;
    std::vector<uint8_t> graph_meta_;  // memory for the per-eval graph context
};

} // namespace e8::model
