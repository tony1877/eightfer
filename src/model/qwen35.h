#pragma once

// Qwen3.5/3.6/3.8 dense ("qwen35" in GGUF): Gated DeltaNet linear attention in 3 of every 4 layers, gated full
// attention with partial interleaved M-RoPE in the 4th, a SwiGLU FFN in every layer. Mirrors llama.cpp's
// src/models/qwen35.cpp at the pinned commit, single sequence only.
//
// MTP drafting (LoadOptions::mtp): the GGUF's MTP block (blk.<n_layer>.*, one gated full-attention layer with nextn
// projections) is loaded on the GPU. Every eval keeps the trunk's final (normed) hidden states and writes the MTP
// layer's K/V for the positions whose next token is known; mtp_step() then proposes tokens from a hidden state and
// the token after it, chaining on its own output (as vLLM's Qwen3-Next MTP). Its KV is a ring of recent positions.
//
// Split precision (DESIGN.md section 3/4): with LoadOptions::residual_path set, every big matrix W of the base GGUF
// gets a residual R from `eightfer pack`, kept in pinned host memory. eval() with EvalOpts::residual computes
// x*W^T + x*R^T for those matrices (R on the CPU for small batches; ggml's scheduler streams R to the GPU for big
// ones). Without it the base alone runs (the draft model).
//
// Speculation support: save_state()/restore_state() snapshot the recurrent state (KV needs no snapshot: entries past
// n_past are masked and get overwritten). An eval with EvalOpts::record keeps the DeltaNet inputs of that batch, so
// rollback(keep) can restore the snapshot and replay only the first `keep` tokens through the recurrence.

#include "model/causal_lm.h"
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
    bool window_ok = false;  // attention may see only the VRAM window of recent tokens (speculative drafts)
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
    bool      mtp          = false; // load the MTP block for drafting (when the GGUF has one and a GPU exists)
    int       mtp_window   = 4096;  // MTP KV ring size
    int       gpu_kv       = -1;    // KV tokens kept in VRAM; fewer than n_ctx keeps the full KV in RAM and a
                                    // window of the most recent tokens in VRAM. -1 = auto (all if it fits)
    double    expert_cache_gb = -1; // MoE models: VRAM for the GPU expert cache; < 0 = auto, 0 = off
    bool      experts_gpu  = false; // MoE models: all experts of GPU layers in VRAM (small models / tests)
};

// one sparse-attention gather op's parameters (see Qwen35's sparse long-context notes)
struct Qwen35GatherInfo {
    const ggml_tensor * host = nullptr;
    int                 kp_rows = 0, page = 0, win_start = 0, nwin = 0;
};

class Qwen35 : public CausalLM {
public:
    Qwen35();
    ~Qwen35() override;

    bool load(const std::string & path, const LoadOptions & opt, std::string & err);

    // Runs `n` tokens (n <= n_ubatch) at positions [n_past(), n_past() + n). Writes n * n_vocab logits to
    // `logits` (row i = token i) when it is non-null, otherwise only advances the state.
    bool eval(const int32_t * tokens, int n, float * logits, std::string & err) override;

    // General form. Writes n (or 1 with last_only) rows of n_vocab logits to `logits`, or with opts.argmax the
    // argmax ids to `ids`. Either output pointer may be null.
    bool eval(const int32_t * tokens, int n, const EvalOpts & opts, float * logits, int32_t * ids, std::string & err);

    // Snapshot / restore the recurrent state and n_past. Two slots: 0 for the committed state of a speculative
    // cycle, 1 (only with MTP) for the base's own check of MTP proposals inside a draft.
    void save_state(int slot = 0);
    void restore_state(int slot = 0);
    // After a recorded eval of n tokens: restore the snapshot `slot` taken before it and re-run only the first `keep`
    // tokens through the DeltaNet layers (attention KV is kept as written). n_past becomes snapshot + keep.
    bool rollback(int keep, std::string & err, int slot = 0);

    bool has_mtp() const { return mtp_on_; }
    // row of the last eval's hidden states that belongs to position n_past() - 1, or -1 when unknown
    int  hidden_row() const { return hid_row_; }
    // One MTP draft step at position `pos`: reads `tok` (the token at pos + 1) and the hidden state at pos, which is
    // row `hid_row` of the last eval when >= 0, else the previous step's output. Writes the prediction for pos + 2:
    // n_vocab logits, or with `id` non-null and `logits` null the argmax.
    bool mtp_step(int32_t tok, int pos, int hid_row, float * logits, int32_t * id, std::string & err);

    bool has_residual() const { return !res_.empty(); }
    uint64_t residual_bytes() const { return res_bytes_; }

    // Empties the KV cache and zeroes the recurrent state.
    void reset() override;
    bool eval_last(const int32_t * tokens, int n, float * logits, std::string & err) override {
        EvalOpts o;
        o.last_only = true;
        return eval(tokens, n, o, logits, nullptr, err);
    }
    void checkpoint_save() override;
    void checkpoint_restore() override;

    // Debug: when >= 0, eval() stops after this layer and writes its output (n * n_embd floats) instead of logits.
    void set_debug_layer(int il) { debug_layer_ = il; }

    int                   n_past() const override { return n_past_; }
    int                   n_ctx() const override { return n_ctx_; }
    // largest batch a speculative verify should use (the sparse long-context attention takes up to 128 queries)
    int                   max_verify() const { return 64; }
    // largest verify for long copied drafts: rollback works within the first max_record tokens, else the kept prefix
    // is evaluated again
    int                   max_long_verify() const { return std::min(128, opt_.n_ubatch); }
    bool                  can_rollback(int keep) const { return keep == recorded_full_ || keep <= recorded_n_; }
    int                   max_record() const { return opt_.max_record; }
    int64_t               n_vocab() const override { return hp_.n_vocab; }
    int64_t               n_layer() const override { return hp_.n_layer; }
    bool                  add_bos() const override { return hp_.add_bos; }
    int64_t               bos_token() const override { return hp_.bos; }
    const Qwen35Hparams & hp() const { return hp_; }
    // Weight bytes placed on the GPU and on the CPU.
    uint64_t gpu_weight_bytes() const override { return gpu_bytes_; }
    uint64_t cpu_weight_bytes() const override { return cpu_bytes_; }
    // KV tokens in VRAM (== n_ctx unless the KV is split between RAM and a VRAM window)
    int gpu_kv() const { return W_; }

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
    // MTP: the layer input from embeddings and hidden states [n_embd, n]
    ggml_tensor * mtp_input(ggml_context * ctx, ggml_tensor * emb, ggml_tensor * hid);
    // MTP layer on x [n_embd, n] at positions `pos`: writes K/V to ring rows `kvidx`; with a mask, also attends and
    // returns the layer output (else nullptr)
    ggml_tensor * mtp_layer(ggml_context * ctx, ggml_cgraph * gf, ggml_tensor * x, int n, ggml_tensor * pos,
                            ggml_tensor * kvidx, ggml_tensor * mask);
    void embed(const int32_t * tokens, int n, std::vector<float> & out) const;

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
    std::vector<ggml_tensor *> conv_bak_, ssm_bak_, rec_qkv_, rec_g_, rec_beta_, conv_ck_, ssm_ck_;
    std::vector<ggml_tensor *> conv_bak1_, ssm_bak1_;  // snapshot slot 1 (MTP)
    int ck_n_past_ = 0;
    int saved_n_past1_ = 0;

    // MTP block (see the notes at the top)
    bool                 mtp_on_ = false;  // only when the full KV fits in VRAM next to it
    ggml_context *       mctx_ = nullptr;
    ggml_backend_buffer_t mbuf_ = nullptr;
    Qwen35Layer          mtp_;
    ggml_tensor *        mtp_eh_ = nullptr, * mtp_enorm_ = nullptr, * mtp_hnorm_ = nullptr, * mtp_norm_ = nullptr;
    ggml_tensor *        mtp_k_ = nullptr, * mtp_v_ = nullptr;  // KV ring [head_dim * n_head_kv, Wm_]
    ggml_tensor *        mtp_hid_ = nullptr;    // the last eval's trunk hidden states [n_embd, n_ubatch]
    ggml_tensor *        mtp_chain_ = nullptr;  // the last MTP step's output [n_embd, 1]
    int                  Wm_ = 0;
    std::vector<int>     mtp_slot_pos_;
    ggml_tensor *        inp_mpos_ = nullptr, * inp_mkvidx_ = nullptr;
    int                  hid_row_ = -1;

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
    int         recorded_n_ = 0;  // recorded tokens of the last recorded eval (0 = none)
    int         recorded_full_ = 0;  // its batch size (rollback beyond recorded_n_ is only possible to keep all)
    int         debug_layer_ = -1;
    uint64_t    gpu_bytes_ = 0, cpu_bytes_ = 0;
    std::vector<uint8_t> graph_meta_;  // memory for the per-eval graph context
    // KV rows written by this eval (I64 [n]): an input, not a view offset, so the graph is the same from token to
    // token and CUDA graphs can be replayed
    ggml_tensor *        inp_kvidx_ = nullptr;

    // Split KV (W_ < n_ctx_): k_cache_/v_cache_ are a VRAM ring of W_ slots (slot = pos % W_), and the full KV
    // lives in RAM per (attention layer, KV head) as [head_dim, n_ctx_] (hk_/hv_, index layer * n_head_kv + head).
    // Exact evals attend the RAM copy (CPU flash attention for small batches, GPU with a per-head copy for big
    // ones); drafts attend the ring. host_valid_: positions [0, host_valid_) are in RAM; exact_upto_: positions
    // [0, exact_upto_) were written by exact (non-draft) evals.
    int                        W_ = 0;
    std::vector<int>           slot_pos_;
    std::vector<ggml_tensor *> hk_, hv_;
    ggml_context *             hctx_ = nullptr;
    ggml_backend_buffer_t      hbuf_ = nullptr;
    int                        host_valid_ = 0, exact_upto_ = 0;
    int                        attn_mode_  = 0;  // while building: 0 = ring/plain KV, 1 = RAM KV
    int                        host_gpu_min_ = 1;  // RAM-KV attention on the GPU from this batch size (CPU below)
    ggml_tensor *              inp_hidx_ = nullptr;
    std::vector<ggml_tensor *> attn_nodes_;  // RAM-KV attention nodes, to place on CPU or GPU
    bool flush_ring(int from, int to, std::string & err);

    // Sparse long-context attention (small exact batches with the KV in RAM): per (attention layer, KV head) the
    // element-wise min/max of each page of kPage keys lives in VRAM. A batch's queries bound q.k per page with them,
    // take the top pages, and attend exactly over those pages plus the most recent tokens, gathered from RAM.
    using GatherInfo = Qwen35GatherInfo;
    std::vector<ggml_tensor *> pmid_;  // page midpoints of the keys, (min + max) / 2 per element
    // Draft far area: the first Kd_ rows of each VRAM ring hold, per KV head, the pages the last sparse verify
    // selected (positions < far_ws_), so drafts see the far context too. The ring itself is rows [Kd_, Kd_ + W_).
    int                        Kd_ = 0, far_rows_ = 0, far_ws_ = 0, ring_off_ = 0;
    std::vector<ggml_tensor *> sel_nodes_;  // per (attention layer, KV head): the sparse verify's page selection
    void fill_draft_far();
    ggml_context *             pctx_ = nullptr;
    ggml_backend_buffer_t      pbuf_ = nullptr;
    int                        sum_upto_ = 0;  // pages [0, sum_upto_) have summaries
    bool                       sparse_ = false;
    int                        sp_nfar_ = 0, sp_kp_ = 0, sp_ws_ = 0, sp_nwin_ = 0, sp_nsel_ = 0;
    std::vector<GatherInfo>    gi_;
    ggml_tensor *              inp_sbias_ = nullptr, * inp_smask_ = nullptr;
    void update_summaries(int upto);
};

} // namespace e8::model
