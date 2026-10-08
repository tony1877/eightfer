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

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <cstdlib>
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
    bool dry       = false;  // with record: leave the recurrent state and n_past as they were; commit(keep) then
                             // advances them by the first `keep` tokens (no snapshot needed)
    bool last_only = false;  // logits for the last token only (prefill)
    bool argmax    = false;  // return argmax token ids instead of logits
    int  topk      = 0;      // > 0 (not argmax): only each row's top-k logits, best first: `logits` gets n*topk values,
                             // `ids` n*topk token ids (sampling with top_k <= topk needs nothing else)
    bool window_ok = false;  // attention may see only the VRAM window of recent tokens (speculative drafts)
    // > 1 (tree verify): the n tokens are n_seqs sequences of n / n_seqs, each continuing the committed state on its
    // own (dry + record only; KV within the VRAM ring). commit(keep, err, seq) then keeps one of them.
    int  n_seqs    = 1;
};

struct LoadOptions {
    int       n_gpu_layers = 0;     // trunk layers on the GPU, counted from the last layer (as llama.cpp does)
    bool      output_gpu   = true;  // output norm + LM head on the GPU (when a GPU exists)
    int       n_ctx        = 4096;  // KV capacity, rounded up to a multiple of 256
    ggml_type kv_type      = GGML_TYPE_F16;
    ggml_type kv_type_v    = GGML_TYPE_COUNT;  // V cache type; GGML_TYPE_COUNT = kv_type
    bool      kv_lock      = false; // lock the RAM KV's pages in physical memory as they are taken (never paged out)
    int       n_slots      = 1;     // sequence slots (see Qwen35::select_slot); > 1 needs the KV in RAM
    double    kv_pool_gb   = 0;     // most RAM KV memory taken by all slots together; 0 = n_ctx tokens' worth
    int       n_threads    = 0;     // CPU threads; 0 = physical cores
    int       n_ubatch     = 1024;  // most tokens per eval() call (prefill batches; one pass over the weights each)
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
    // A second sequence over src's weights and KV cache (own recurrent state, CUDA stream, scheduler): drafts while
    // src verifies. sync_from() makes it continue from src's committed state.
    bool make_shadow(const Qwen35 & src, std::string & err);
    void sync_from(const Qwen35 & src);
    bool is_shadow() const { return shadow_; }

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
    // After a dry recorded eval: run the first `keep` tokens through the DeltaNet recurrence and advance n_past by keep
    bool commit(int keep, std::string & err, int seq = 0);
    // whether a tree verify of n tokens fits now (KV exact in the VRAM ring)
    bool can_tree(int n) const { return n_past_ + n <= W_ && n <= opt_.max_record && n <= opt_.n_ubatch; }

    bool has_mtp() const { return mtp_on_; }
    // row of the last eval's hidden states that belongs to position n_past() - 1, or -1 when unknown
    int  hidden_row() const { return hid_row_; }
    // One MTP draft step at position `pos`: reads `tok` (the token at pos + 1) and the hidden state at pos, which is
    // row `hid_row` of the last eval when >= 0, else the previous step's output. Writes the prediction for pos + 2:
    // n_vocab logits, or with `id` non-null and `logits` null the argmax.
    bool mtp_step(int32_t tok, int pos, int hid_row, float * logits, int32_t * id, std::string & err, int topk = 0);

    bool has_residual() const { return !res_.empty(); }
    uint64_t residual_bytes() const { return res_bytes_; }

    // Sequence slots: each slot is a sequence of its own (RAM KV region, a part of the VRAM ring and of the MTP ring,
    // recurrent state, page summaries, prompt-reuse checkpoint, KV bookkeeping), so several conversations keep their
    // state. One is active at a time; select_slot() parks the active one (recurrent state, page summaries and MTP
    // hidden row copied to RAM; its ring parts stay in VRAM) and resumes slot s, cheap enough to switch every decode
    // cycle. A slot's RAM KV takes memory only as positions are written; when all slots together would pass the pool
    // budget, the least recently used idle slots are emptied first (slot_epoch() changes when that happens).
    int      n_slots() const { return std::max(1, (int) slots_.size()); }
    int      active_slot() const { return cur_slot_; }
    bool     select_slot(int s, std::string & err);
    int      slot_n_past(int s) const { return s == cur_slot_ ? n_past_ : slots_[(size_t) s].n_past; }
    uint64_t slot_epoch(int s) const { return slots_.empty() ? 0 : slots_[(size_t) s].epoch; }
    uint64_t kv_committed_bytes() const { return committed_bytes_; }
    // a pinned slot (a conversation in progress) is never emptied to make room in the KV pool
    void     pin_slot(int s, bool on) { if (!slots_.empty()) slots_[(size_t) s].pinned = on; }

    // Empties the active slot's KV cache (its RAM pages go back to the OS) and zeroes the recurrent state.
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
    int                   max_batch() const { return opt_.n_ubatch; }
    // every residual product runs on the GPU (host weights staged by the ggml weight prefetch): a verify then costs
    // about the same at any size; else batches below 32 multiply the residual on the CPU
    bool                  res_gpu_all() const {
        static const bool off = std::getenv("E8_NO_PREFETCH") || std::getenv("E8_CPU_SMALL_VERIFY");
        return gpu_ && !off;
    }
    int                   verify_gpu_min() const { return res_gpu_all() ? 1 : 32; }
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
                           ggml_tensor * beta, int n, int ns = 1);
    // x*W^T, plus x*R^T when the residual is active and W has one
    ggml_tensor * mm(ggml_context * ctx, ggml_tensor * w, ggml_tensor * x);
    bool load_residual(const std::string & path, std::string & err);
    bool compute(ggml_context * ctx, ggml_cgraph * gf, std::string & err);
    bool replay(int keep, std::string & err, int row0 = 0);  // the first keep recorded tokens through the DeltaNet recurrence
    bool write_state_ = true;                  // while building: gdn_core updates the recurrent state
    ggml_tensor * out_ids_ = nullptr;          // while building: the top-k ids output (EvalOpts::topk)
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
    std::unordered_map<const ggml_tensor *, std::vector<ggml_tensor *>> res_chunks_;  // big residuals in row pieces
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
    bool        dry_pending_ = false;
    int         rec_seqs_ = 1;  // sequences of the last recorded eval (tree verify)  // the last recorded eval was dry: commit() before anything else
    int         debug_layer_ = -1;
    bool        shadow_ = false;
    uint64_t    gpu_bytes_ = 0, cpu_bytes_ = 0;
    std::vector<uint8_t> graph_meta_;  // memory for the per-eval graph context
    std::vector<uint8_t> graph_meta_draft_, graph_meta_mtp_;  // single-token draft evals / MTP steps (CUDA graph reuse)
    std::vector<uint8_t> graph_meta_small_[34];
    std::vector<ggml_tensor *> res_on_gpu_;  // residual products of the graph being built, pinned to the GPU  // base-only batches of 2..16 tokens, by size and dry/not
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
    void *                     hmem_ = nullptr;  // hbuf_'s memory when it comes straight from the OS (not pinned)
    size_t                     hmem_bytes_ = 0;
    // sequence slots (see select_slot); hk_/hv_/conv_ck_/ssm_ck_ are the active slot's
    struct KvSlot {
        std::vector<ggml_tensor *> hk, hv;              // RAM KV per (attention layer, KV head)
        std::vector<ggml_tensor *> conv, ssm;           // parked recurrent state (RAM)
        std::vector<ggml_tensor *> conv_ck, ssm_ck;     // prompt-reuse checkpoint (RAM)
        uint8_t *                  base = nullptr;      // this slot's part of hmem_
        size_t                     bytes = 0;
        int                        committed = 0;       // positions [0, committed) have memory in every head
        int                        n_past = 0, host_valid = 0, exact_upto = 0, ck_n_past = 0;
        uint64_t                   epoch = 0, used = 0;
        bool                       pinned = false;
        // its part of the VRAM ring and of the MTP ring (views into k_cache_ / mtp_k_ of slot 0's allocation)
        std::vector<ggml_tensor *> kr, vr;
        ggml_tensor *              mk = nullptr, * mv = nullptr;
        std::vector<int>           slot_pos, mtp_slot_pos;
        uint8_t *                  summ = nullptr;  // page summaries (RAM copy), per head n_ctx / kPage pages
        int                        sum_upto = 0, sum_saved = 0, far_rows = 0, far_ws = 0;
        std::vector<float>         hid;             // MTP: hidden state at n_past - 1 (empty = none)
    };
    std::vector<KvSlot>        slots_;
    int                        cur_slot_ = 0;
    uint64_t                   use_clock_ = 0;
    size_t                     kv_row_bytes_ = 0;      // RAM KV bytes per position (all heads, K and V)
    size_t                     pool_bytes_ = 0, committed_bytes_ = 0, locked_bytes_ = 0;
    ggml_context *             slctx_ = nullptr;       // parked states and checkpoints of slots 1..
    ggml_backend_buffer_t      slbuf_ = nullptr;
    void *                     slmem_ = nullptr;
    size_t                     slmem_bytes_ = 0;
    ggml_context *             rvctx_ = nullptr;       // the slots' ring views
    void *                     summem_ = nullptr;      // the slots' page summary copies
    size_t                     summem_bytes_ = 0;
    int                        sum_low_ = 0;           // lowest sum_upto_ since the active slot was resumed
    bool ensure_committed(int upto, std::string & err);  // the active slot's RAM KV has memory for [0, upto)
    // gives its RAM KV back to the OS and empties it; `forget` (an eviction, not the owner's reset) bumps its epoch
    void drop_slot(KvSlot & s, bool forget = true);
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
