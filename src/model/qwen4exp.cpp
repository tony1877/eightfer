#include "model/qwen4exp.h"

#include "ggml-alloc.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <thread>

namespace e8::model {

namespace {

constexpr int kGraphSize = 32768;
constexpr int kKvPad     = 256;

std::string key(const char * k) {
    return std::string("qwen4exp.") + k;
}

} // namespace

bool Qwen4ExpHparams::load(const GgufSet & s, std::string & err) {
    const GgufFile & f = s.meta();
    if (f.arch() != "qwen4exp") {
        err = "architecture \"" + f.arch() + "\" is not qwen4exp";
        return false;
    }
    n_layer              = f.i64(key("block_count"), 0) - f.i64(key("nextn_predict_layers"), 0);
    n_embd               = f.i64(key("embedding_length"), 0);
    n_head               = f.i64(key("attention.head_count"), 0);
    n_head_kv            = f.i64(key("attention.head_count_kv"), 0);
    head_dim             = f.i64(key("attention.key_length"), 0);
    n_ctx_train          = f.i64(key("context_length"), 0);
    n_rot                = f.i64(key("rope.dimension_count"), head_dim);
    rope_freq_base       = f.f32(key("rope.freq_base"), 1e7f);
    rms_eps              = f.f32(key("attention.layer_norm_rms_epsilon"), 1e-6f);
    expert_weights_scale = f.f32(key("expert_weights_scale"), 0.0f);
    ssm_d_conv           = f.i64(key("ssm.conv_kernel"), 0);
    ssm_d_inner          = f.i64(key("ssm.inner_size"), 0);
    ssm_d_state          = f.i64(key("ssm.state_size"), 0);
    ssm_n_v              = f.i64(key("ssm.time_step_rank"), 0);
    ssm_n_k              = f.i64(key("ssm.group_count"), 0);
    n_expert             = f.i64(key("expert_count"), 0);
    n_expert_used        = f.i64(key("expert_used_count"), 0);
    n_ff_exp             = f.i64(key("expert_feed_forward_length"), 0);
    n_ff_shexp           = f.i64(key("expert_shared_feed_forward_length"), n_ff_exp);
    hc                   = f.i64(key("hyper_connection.count"), 0);
    hc_lr                = f.i64(key("hyper_connection.low_rank"), 0);
    idx_n_head           = f.i64(key("attention.indexer.head_count"), 0);
    idx_dim              = f.i64(key("attention.indexer.key_length"), 0);
    idx_budget           = f.i64(key("attention.indexer.top_k"), 0);
    add_bos              = f.b("tokenizer.ggml.add_bos_token", false);
    bos                  = f.i64("tokenizer.ggml.bos_token_id", -1);

    const std::vector<int64_t> sec = f.i64_arr(key("rope.dimension_sections"));
    for (size_t i = 0; i < 4 && i < sec.size(); i++) rope_sections[i] = (int) sec[i];

    compress = f.i64_arr(key("attention.compress_ratios"));
    if (compress.size() == 1) compress.assign((size_t) n_layer, compress[0]);
    compress.resize((size_t) n_layer, 0);

    recurrent.assign((size_t) n_layer, false);
    const std::vector<int64_t> rl = f.i64_arr(key("attention.recurrent_layers"));
    const int64_t              interval = f.i64(key("full_attention_interval"), 4);
    for (int64_t il = 0; il < n_layer; il++) {
        recurrent[(size_t) il] = il < (int64_t) rl.size() ? rl[(size_t) il] != 0 : (il + 1) % interval != 0;
    }

    ple_layer.assign((size_t) n_layer, false);
    for (int64_t il : f.i64_arr(key("ple.layers"))) {
        if (il >= 0 && il < n_layer) ple_layer[(size_t) il] = true;
    }
    ple_ngram       = f.i64(key("ple.ngram_size"), 0);
    ple_per_gram    = f.i64(key("ple.heads_per_ngram"), 0);
    ple_conv_kernel = f.i64(key("ple.conv_kernel"), 0);
    ple_eos         = f.i64(key("ple.eos_token_id"), 0);
    ple_head_dim    = f.i64(key("embedding_length_per_layer_input"), 0);
    for (int64_t v : f.i64_arr(key("ple.layer_multipliers"))) ple_mult.push_back((uint64_t) v);
    for (int64_t v : f.i64_arr(key("ple.head_offsets"))) ple_offset.push_back((uint64_t) v);
    for (int64_t v : f.i64_arr(key("ple.head_vocab_sizes"))) ple_vocab.push_back((uint64_t) v);

    const ggml_tensor * emb = s.tensor("token_embd.weight");
    n_vocab                 = emb ? emb->ne[1] : 0;

    const bool has_ple = std::find(ple_layer.begin(), ple_layer.end(), true) != ple_layer.end();
    if (n_layer <= 0 || n_embd <= 0 || n_head <= 0 || n_head_kv <= 0 || head_dim <= 0 || n_vocab <= 0 || hc < 2 ||
        hc_lr <= 0 || n_expert <= 0 || n_expert_used <= 0 || ssm_d_state <= 0 || ssm_n_k <= 0 || ssm_n_v % ssm_n_k != 0 ||
        (has_ple && (ple_ngram < 2 || ple_mult.size() < (size_t) ple_ngram || ple_offset.size() < (size_t) ple_n_heads() ||
                     ple_vocab.size() < (size_t) ple_n_heads() || ple_head_dim * ple_n_heads() != n_embd))) {
        err = "incomplete or unsupported qwen4exp hyperparameters";
        return false;
    }
    return true;
}

Qwen4Exp::Qwen4Exp() = default;

Qwen4Exp::~Qwen4Exp() {
    if (sched_) ggml_backend_sched_free(sched_);
    if (wbuf_gpu_) ggml_backend_buffer_free(wbuf_gpu_);
    for (auto b : map_bufs_) {
        if (b) ggml_backend_buffer_free(b);
    }
    for (int i = 0; i < 2; i++) {
        if (sbuf_[i]) ggml_backend_buffer_free(sbuf_[i]);
        if (wctx_[i]) ggml_free(wctx_[i]);
        if (sctx_[i]) ggml_free(sctx_[i]);
    }
    if (gpu_) ggml_backend_free(gpu_);
    if (cpu_) ggml_backend_free(cpu_);
}

bool Qwen4Exp::load(const std::string & path, const LoadOptions & opt, std::string & err) {
    opt_ = opt;
    if (!file_.open(path, err) || !hp_.load(file_, err)) {
        return false;
    }
    const auto & h = hp_;

    cpu_ = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!cpu_) {
        err = "cannot initialize the CPU backend";
        return false;
    }
    ggml_backend_cpu_set_n_threads(cpu_, opt.n_threads > 0 ? opt.n_threads : (int) std::max(1u, std::thread::hardware_concurrency()));
    if (ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU)) {
        gpu_ = ggml_backend_dev_init(dev, nullptr);
    }
    const int  n_gpu_layers = gpu_ ? std::clamp(opt.n_gpu_layers, 0, (int) h.n_layer) : 0;
    const bool out_gpu      = gpu_ && opt.output_gpu && n_gpu_layers > 0;

    const size_t     n_tens = file_.tensors().size() + 8;
    ggml_init_params ip     = { ggml_tensor_overhead() * n_tens, nullptr, true };
    wctx_[0]                = ggml_init(ip);
    wctx_[1]                = ggml_init(ip);
    std::vector<std::pair<ggml_tensor *, const ggml_tensor *>> to_gpu, to_map;
    bool missing = false;
    auto make = [&](const std::string & name, bool gpu, bool required = true) -> ggml_tensor * {
        const ggml_tensor * m = file_.tensor(name);
        if (!m) {
            if (required) {
                err     = "tensor missing: " + name;
                missing = true;
            }
            return nullptr;
        }
        ggml_tensor * t = ggml_dup_tensor(wctx_[gpu ? 0 : 1], m);
        ggml_set_name(t, name.c_str());
        (gpu ? to_gpu : to_map).emplace_back(t, m);
        return t;
    };

    tok_embd_  = file_.tensor("token_embd.weight");
    ple_table_ = file_.tensor("per_layer_token_embd.weight");
    if (!tok_embd_) {
        err = "tensor missing: token_embd.weight";
        return false;
    }
    output_      = make("output.weight", out_gpu, false);
    if (!output_) output_ = make("token_embd.weight", out_gpu);
    out_hc_norm_ = make("output_hc_norm.weight", out_gpu);
    out_hc_down_ = make("output_hc_down.weight", out_gpu);
    out_hc_up_   = make("output_hc_up.weight", out_gpu);

    layers_.resize((size_t) h.n_layer);
    bool any_ple = false;
    for (int64_t il = 0; il < h.n_layer; il++) {
        Qwen4ExpLayer &   L = layers_[(size_t) il];
        const bool        g = il >= h.n_layer - n_gpu_layers;
        const std::string p = "blk." + std::to_string(il) + ".";
        L.on_gpu            = g;
        L.hca_norm   = make(p + "hc_attn_norm.weight", g);
        L.hca_down   = make(p + "hc_attn_down.weight", g);
        L.hca_up     = make(p + "hc_attn_up.weight", g);
        L.hca_inject = make(p + "hc_attn_inject.weight", g);
        L.hcf_norm   = make(p + "hc_ffn_norm.weight", g);
        L.hcf_down   = make(p + "hc_ffn_down.weight", g);
        L.hcf_up     = make(p + "hc_ffn_up.weight", g);
        L.hcf_inject = make(p + "hc_ffn_inject.weight", g);
        if (h.recurrent[(size_t) il]) {
            L.wqkv     = make(p + "attn_qkv.weight", g);
            L.wz       = make(p + "attn_gate.weight", g);
            L.conv1d   = make(p + "ssm_conv1d.weight", g);
            L.dt_bias  = make(p + "ssm_dt.bias", g);
            L.a        = make(p + "ssm_a", g);
            L.w_beta   = make(p + "ssm_beta.weight", g);
            L.w_alpha  = make(p + "ssm_alpha.weight", g);
            L.ssm_norm = make(p + "ssm_norm.weight", g);
            L.ssm_out  = make(p + "ssm_out.weight", g);
        } else {
            L.wq     = make(p + "attn_q.weight", g);
            L.wk     = make(p + "attn_k.weight", g);
            L.wv     = make(p + "attn_v.weight", g);
            L.wo     = make(p + "attn_output.weight", g);
            L.q_norm = make(p + "attn_q_norm.weight", g);
            L.k_norm = make(p + "attn_k_norm.weight", g);
            if (h.compress[(size_t) il] > 0) {
                L.idx_q      = make(p + "indexer.q_proj.weight", g);
                L.idx_k      = make(p + "indexer.k_proj.weight", g);
                L.idx_q_norm = make(p + "indexer.q_norm.weight", g);
                L.idx_k_norm = make(p + "indexer.k_norm.weight", g);
                if (kpool_ && kpool_ != h.compress[(size_t) il]) {
                    err = "QSA layers must share one compress ratio";
                    return false;
                }
                kpool_ = (int) h.compress[(size_t) il];
            }
        }
        if (h.ple_layer[(size_t) il]) {
            any_ple          = true;
            L.ple_key        = make(p + "ple_key.weight", g);
            L.ple_value      = make(p + "ple_value.weight", g);
            L.ple_norm_key   = make(p + "ple_norm_key.weight", g);
            L.ple_norm_query = make(p + "ple_norm_query.weight", g);
            L.ple_norm_conv  = make(p + "ple_norm_conv.weight", g);
            L.ple_conv1d     = make(p + "ple_conv1d.weight", g);
        }
        L.gate_inp     = make(p + "ffn_gate_inp.weight", g);
        L.gate_up_exps = make(p + "ffn_gate_up_exps.weight", false, false);  // experts stay in the mapping
        if (!L.gate_up_exps) {
            L.gate_exps = make(p + "ffn_gate_exps.weight", false);
            L.up_exps   = make(p + "ffn_up_exps.weight", false);
        }
        L.down_exps      = make(p + "ffn_down_exps.weight", false);
        L.gate_inp_shexp = make(p + "ffn_gate_inp_shexp.weight", g, false);
        L.gate_shexp     = make(p + "ffn_gate_shexp.weight", g, false);
        L.up_shexp       = make(p + "ffn_up_shexp.weight", g, false);
        L.down_shexp     = make(p + "ffn_down_shexp.weight", g, false);
        if (missing) return false;
    }
    if (missing) return false;
    if (any_ple && !ple_table_) {
        err = "tensor missing: per_layer_token_embd.weight";
        return false;
    }
    if (kpool_ > 0 && (h.idx_budget % kpool_ != 0 || h.idx_dim <= 0 || h.idx_n_head <= 0)) {
        err = "QSA budget must be a multiple of the compress ratio";
        return false;
    }

    // GPU copies
    if (gpu_ && ggml_get_first_tensor(wctx_[0])) {
        wbuf_gpu_ = ggml_backend_alloc_ctx_tensors_from_buft(wctx_[0], ggml_backend_get_default_buffer_type(gpu_));
        if (!wbuf_gpu_) {
            err = "not enough VRAM for the GPU layers (lower --gpu-layers)";
            return false;
        }
        ggml_backend_buffer_set_usage(wbuf_gpu_, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        gpu_bytes_ = ggml_backend_buffer_get_size(wbuf_gpu_);
        for (auto & [t, m] : to_gpu) ggml_backend_tensor_set(t, file_.data(m), 0, ggml_nbytes(t));
    }
    // CPU tensors: used in place from the mapping
    map_bufs_.assign(file_.n_files(), nullptr);
    for (size_t i = 0; i < file_.n_files(); i++) {
        const auto & mf = file_.map(i);
        map_bufs_[i]    = ggml_backend_cpu_buffer_from_ptr((void *) mf.data(), (size_t) mf.size());
        ggml_backend_buffer_set_usage(map_bufs_[i], GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    }
    for (auto & [t, m] : to_map) {
        if (ggml_backend_tensor_alloc(map_bufs_[file_.file_of(m)], t, (void *) file_.data(m)) != GGML_STATUS_SUCCESS) {
            err = std::string("cannot map tensor ") + ggml_get_name(t);
            return false;
        }
        cpu_bytes_ += ggml_nbytes(t);
    }

    // state
    n_ctx_                = std::max(kKvPad, (opt.n_ctx + kKvPad - 1) / kKvPad * kKvPad);
    ggml_init_params sp   = { ggml_tensor_overhead() * (size_t) (h.n_layer * 8 + 8), nullptr, true };
    sctx_[0]              = ggml_init(sp);
    sctx_[1]              = ggml_init(sp);
    const size_t nl       = (size_t) h.n_layer;
    k_cache_.assign(nl, nullptr);
    v_cache_.assign(nl, nullptr);
    idx_raw_.assign(nl, nullptr);
    idx_pool_.assign(nl, nullptr);
    conv_state_.assign(nl, nullptr);
    ssm_state_.assign(nl, nullptr);
    ple_state_.assign(nl, nullptr);
    for (int64_t il = 0; il < h.n_layer; il++) {
        ggml_context * c = sctx_[layers_[(size_t) il].on_gpu ? 0 : 1];
        const size_t   l = (size_t) il;
        if (h.recurrent[l]) {
            conv_state_[l] = ggml_new_tensor_1d(c, GGML_TYPE_F32, (h.ssm_d_conv - 1) * h.conv_channels());
            ssm_state_[l]  = ggml_new_tensor_1d(c, GGML_TYPE_F32, h.ssm_d_state * h.ssm_d_state * h.ssm_n_v);
        } else {
            k_cache_[l] = ggml_new_tensor_2d(c, opt.kv_type, h.head_dim * h.n_head_kv, n_ctx_);
            v_cache_[l] = ggml_new_tensor_2d(c, opt.kv_type, h.head_dim * h.n_head_kv, n_ctx_);
            if (h.compress[l] > 0) {
                idx_raw_[l]  = ggml_new_tensor_2d(c, GGML_TYPE_F32, h.idx_dim, n_ctx_);
                idx_pool_[l] = ggml_new_tensor_2d(c, GGML_TYPE_F32, h.idx_dim, n_ctx_ / kpool_ + 1);
            }
        }
        if (h.ple_layer[l]) {
            ple_state_[l] = ggml_new_tensor_2d(c, GGML_TYPE_F32, (h.ple_conv_kernel - 1) * h.ple_ngram, h.hc * h.n_embd);
        }
    }
    for (int i = 0; i < 2; i++) {
        ggml_backend_t be = i == 0 ? gpu_ : cpu_;
        if (!be || !ggml_get_first_tensor(sctx_[i])) continue;
        sbuf_[i] = ggml_backend_alloc_ctx_tensors_from_buft(sctx_[i], ggml_backend_get_default_buffer_type(be));
        if (!sbuf_[i]) {
            err = "not enough memory for the KV cache / recurrent state (lower --ctx)";
            return false;
        }
    }
    reset();

    std::vector<ggml_backend_t> bes;
    if (gpu_) bes.push_back(gpu_);
    bes.push_back(cpu_);
    sched_ = ggml_backend_sched_new(bes.data(), nullptr, (int) bes.size(), kGraphSize, false, true);
    graph_meta_.resize(ggml_tensor_overhead() * kGraphSize + ggml_graph_overhead_custom(kGraphSize, false));
    return true;
}

void Qwen4Exp::reset() {
    for (auto b : sbuf_) {
        if (b) ggml_backend_buffer_clear(b, 0);
    }
    history_.clear();
    n_past_ = 0;
}

// ---- building blocks -------------------------------------------------------------------------------------------

ggml_tensor * Qwen4Exp::hc_mix(ggml_context * ctx, ggml_tensor * x, ggml_tensor * wn, ggml_tensor * wd, ggml_tensor * wu,
                               ggml_tensor * wi, ggml_tensor ** inject) {
    const int64_t ne = hp_.n_embd, hc = hp_.hc, n = x->ne[2];
    // grouped RMSNorm: over each stream, gamma [n_embd, hc]
    ggml_tensor * xn = ggml_mul(ctx, ggml_rms_norm(ctx, x, hp_.rms_eps), ggml_reshape_2d(ctx, wn, ne, hc));
    ggml_tensor * x2 = ggml_reshape_2d(ctx, xn, ne * hc, n);
    ggml_tensor * lo = ggml_silu(ctx, ggml_scale(ctx, ggml_mul_mat(ctx, wd, x2), 1.0f / (float) hc));
    ggml_tensor * g  = ggml_mul(ctx, x2, ggml_sigmoid(ctx, ggml_mul_mat(ctx, wu, lo)));
    g                = ggml_reshape_3d(ctx, g, ne, hc, n);
    ggml_tensor * mixed = ggml_cont(ctx, ggml_view_2d(ctx, g, ne, n, g->nb[2], 0));
    for (int64_t c = 1; c < hc; c++) {
        mixed = ggml_add(ctx, mixed, ggml_view_2d(ctx, g, ne, n, g->nb[2], (size_t) c * g->nb[1]));
    }
    mixed = ggml_scale(ctx, mixed, 1.0f / (float) hc);
    if (inject) {
        *inject = ggml_mul_mat(ctx, wi, x2);  // [hc, n]
    }
    return mixed;
}

ggml_tensor * Qwen4Exp::hc_combine(ggml_context * ctx, ggml_tensor * res, ggml_tensor * out, ggml_tensor * inject) {
    const int64_t ne = hp_.n_embd, hc = hp_.hc, n = res->ne[2];
    // 2*sigmoid(inject/hc): a zero injection is a plain residual add
    ggml_tensor * w = ggml_scale(ctx, ggml_sigmoid(ctx, ggml_scale(ctx, inject, 1.0f / (float) hc)), 2.0f);
    w               = ggml_reshape_3d(ctx, w, 1, hc, n);
    ggml_tensor * b = ggml_repeat_4d(ctx, ggml_reshape_3d(ctx, out, ne, 1, n), ne, hc, n, 1);
    return ggml_add(ctx, res, ggml_mul(ctx, b, w));
}

ggml_tensor * Qwen4Exp::gdn(ggml_context * ctx, ggml_cgraph * gf, int64_t il, ggml_tensor * cur, int n) {
    const auto &          h = hp_;
    const Qwen4ExpLayer & L = layers_[(size_t) il];
    const int64_t S = h.ssm_d_state, Hk = h.ssm_n_k, Hv = h.ssm_n_v, C = h.conv_channels();
    const float   eps = h.rms_eps;

    ggml_tensor * qkv   = ggml_mul_mat(ctx, L.wqkv, cur);
    ggml_tensor * z     = ggml_mul_mat(ctx, L.wz, cur);
    ggml_tensor * beta  = ggml_reshape_4d(ctx, ggml_sigmoid(ctx, ggml_mul_mat(ctx, L.w_beta, cur)), 1, Hv, n, 1);
    ggml_tensor * alpha = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, L.w_alpha, cur), Hv, n, 1);
    ggml_tensor * g     = ggml_reshape_4d(ctx, ggml_mul(ctx, ggml_softplus(ctx, ggml_add(ctx, alpha, L.dt_bias)), L.a), 1, Hv, n, 1);

    ggml_tensor * cs   = conv_state_[(size_t) il];
    ggml_tensor * cin  = ggml_concat(ctx, ggml_reshape_3d(ctx, cs, h.ssm_d_conv - 1, C, 1),
                                     ggml_transpose(ctx, ggml_reshape_3d(ctx, qkv, C, n, 1)), 0);
    ggml_tensor * last = ggml_view_3d(ctx, cin, h.ssm_d_conv - 1, C, 1, cin->nb[1], cin->nb[2],
                                      ggml_row_size(cin->type, cin->ne[0] - (h.ssm_d_conv - 1)));
    ggml_build_forward_expand(gf, ggml_cpy(ctx, last, cs));

    ggml_tensor * conv = ggml_silu(ctx, ggml_ssm_conv(ctx, cin, L.conv1d));
    const size_t  nb1  = ggml_row_size(conv->type, C);
    ggml_tensor * q = ggml_view_4d(ctx, conv, S, Hk, n, 1, ggml_row_size(conv->type, S), nb1, nb1 * n, 0);
    ggml_tensor * k = ggml_view_4d(ctx, conv, S, Hk, n, 1, ggml_row_size(conv->type, S), nb1, nb1 * n, ggml_row_size(conv->type, S * Hk));
    ggml_tensor * v = ggml_view_4d(ctx, conv, S, Hv, n, 1, ggml_row_size(conv->type, S), nb1, nb1 * n, ggml_row_size(conv->type, 2 * S * Hk));
    q = ggml_scale(ctx, ggml_rms_norm(ctx, q, eps / (float) S), 1.0f / sqrtf((float) S));
    k = ggml_scale(ctx, ggml_rms_norm(ctx, k, eps / (float) S), 1.0f / sqrtf((float) S));

    ggml_tensor * ss  = ssm_state_[(size_t) il];
    ggml_tensor * res = ggml_gated_delta_net(ctx, q, k, v, g, beta, ggml_reshape_4d(ctx, ss, S, S, Hv, 1), 1);
    ggml_tensor * out = ggml_view_4d(ctx, res, S, Hv, n, 1, ggml_row_size(res->type, S), ggml_row_size(res->type, S * Hv),
                                     ggml_row_size(res->type, S * Hv * n), 0);
    ggml_build_forward_expand(gf, ggml_cpy(ctx, ggml_view_1d(ctx, res, S * S * Hv, ggml_row_size(res->type, S * Hv * n)), ss));

    // gated norm with a sigmoid gate (Qwen3.5 uses silu)
    ggml_tensor * o = ggml_mul(ctx, ggml_mul(ctx, ggml_rms_norm(ctx, out, eps), L.ssm_norm),
                               ggml_sigmoid(ctx, ggml_reshape_4d(ctx, z, S, Hv, n, 1)));
    return ggml_mul_mat(ctx, L.ssm_out, ggml_reshape_2d(ctx, o, S * Hv, n));
}

ggml_tensor * Qwen4Exp::attn(ggml_context * ctx, ggml_cgraph * gf, int64_t il, ggml_tensor * cur, int n, int n_kv,
                             bool sparse, Inputs & in) {
    const auto &          h  = hp_;
    const Qwen4ExpLayer & L  = layers_[(size_t) il];
    const int64_t         hd = h.head_dim;
    int sections[4] = { h.rope_sections[0], h.rope_sections[1], h.rope_sections[2], h.rope_sections[3] };
    auto rope = [&](ggml_tensor * x, ggml_tensor * pos) {
        return ggml_rope_multi(ctx, x, pos, nullptr, (int) h.n_rot, sections, GGML_ROPE_TYPE_IMROPE, (int) h.n_ctx_train,
                               h.rope_freq_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
    };
    auto norm = [&](ggml_tensor * x, ggml_tensor * w) { return ggml_mul(ctx, ggml_rms_norm(ctx, x, h.rms_eps), w); };

    ggml_tensor * mask = in.mask;
    if (L.idx_k) {
        // QSA indexer: raw keys into the cache, pool the blocks this batch completes, score blocks when sparse
        const int64_t d = h.idx_dim, r = kpool_;
        ggml_tensor * raw  = idx_raw_[(size_t) il];
        ggml_tensor * pool = idx_pool_[(size_t) il];
        ggml_build_forward_expand(gf, ggml_cpy(ctx, ggml_mul_mat(ctx, L.idx_k, cur),
                                               ggml_view_2d(ctx, raw, d, n, raw->nb[1], raw->nb[1] * (size_t) n_past_)));
        const int64_t b0 = n_past_ / r, b1 = (n_past_ + n) / r, nnew = b1 - b0;
        if (nnew > 0) {
            ggml_tensor * rows = ggml_view_3d(ctx, raw, d, r, nnew, raw->nb[1], raw->nb[1] * r, raw->nb[1] * (size_t) (r * b0));
            ggml_tensor * sum  = ggml_cont(ctx, ggml_view_2d(ctx, rows, d, nnew, rows->nb[2], 0));
            for (int64_t i = 1; i < r; i++) {
                sum = ggml_add(ctx, sum, ggml_view_2d(ctx, rows, d, nnew, rows->nb[2], (size_t) i * rows->nb[1]));
            }
            ggml_tensor * pk = norm(ggml_scale(ctx, sum, 1.0f / (float) r), L.idx_k_norm);
            pk               = rope(ggml_reshape_3d(ctx, pk, d, 1, nnew), in.pool_pos);
            ggml_build_forward_expand(gf, ggml_cpy(ctx, ggml_reshape_2d(ctx, pk, d, nnew),
                                                   ggml_view_2d(ctx, pool, d, nnew, pool->nb[1], pool->nb[1] * (size_t) b0)));
        }
        if (sparse) {
            const int64_t H = h.idx_n_head, NB = b1, K = h.idx_budget / r;
            ggml_tensor * q = norm(ggml_reshape_3d(ctx, ggml_mul_mat(ctx, L.idx_q, cur), d, H, n), L.idx_q_norm);
            q               = rope(q, in.pos);
            ggml_tensor * s = ggml_mul_mat(ctx, ggml_view_2d(ctx, pool, d, NB, pool->nb[1], 0), ggml_reshape_2d(ctx, q, d, H * n));
            s = ggml_relu(ctx, s);                                                           // [NB, H*n]
            s = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, s, NB, H, n), 1, 0, 2, 3));  // [H, NB, n]
            s = ggml_scale(ctx, ggml_reshape_2d(ctx, ggml_sum_rows(ctx, s), NB, n), 1.0f / sqrtf((float) d));
            s = ggml_add(ctx, s, in.bvis);                                                   // complete-and-visible blocks
            ggml_tensor * top = ggml_top_k(ctx, s, (int) K);                                 // [K, n]
            ggml_tensor * blk = ggml_fill(ctx, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, NB, n), -INFINITY);
            ggml_tensor * zer = ggml_fill(ctx, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, K, n), 0.0f);
            blk = ggml_set_rows(ctx, blk, zer, ggml_reshape_3d(ctx, top, K, n, 1));
            blk = ggml_add(ctx, ggml_reshape_2d(ctx, blk, NB, n), in.bvis);
            ggml_tensor * tok = ggml_reshape_2d(ctx, ggml_repeat_4d(ctx, ggml_reshape_3d(ctx, blk, 1, NB, n), r, NB, n, 1), r * NB, n);
            if (r * NB < n_kv) {
                tok = ggml_concat(ctx, tok, ggml_fill(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_kv - r * NB, n), -INFINITY), 0);
            }
            // union of the selected blocks and the token's own incomplete block (0/-inf masks)
            ggml_tensor * u = ggml_clamp(ctx, ggml_add(ctx, ggml_exp(ctx, tok), ggml_exp(ctx, in.tail)), 0.0f, 1.0f);
            mask            = ggml_cast(ctx, ggml_log(ctx, u), GGML_TYPE_F16);
            if (std::getenv("E8_QSA_DUMP") && !dbg_mask_) {  // debug: the first sparse layer's attention mask
                dbg_mask_ = mask;
                ggml_set_output(mask);
                dbg_scores_ = s;
                ggml_set_output(s);
            }
        }
    }

    ggml_tensor * qg = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, L.wq, cur), hd * 2, h.n_head, n);
    const size_t  es = ggml_element_size(qg);
    ggml_tensor * Q  = ggml_view_3d(ctx, qg, hd, h.n_head, n, es * hd * 2, es * hd * 2 * h.n_head, 0);
    ggml_tensor * gate = ggml_cont_2d(ctx, ggml_view_3d(ctx, qg, hd, h.n_head, n, es * hd * 2, es * hd * 2 * h.n_head, es * hd), hd * h.n_head, n);
    ggml_tensor * Kc = norm(ggml_reshape_3d(ctx, ggml_mul_mat(ctx, L.wk, cur), hd, h.n_head_kv, n), L.k_norm);
    ggml_tensor * Vc = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, L.wv, cur), hd, h.n_head_kv, n);
    Q  = rope(norm(Q, L.q_norm), in.pos);
    Kc = rope(Kc, in.pos);

    ggml_tensor * kc  = k_cache_[(size_t) il];
    ggml_tensor * vc  = v_cache_[(size_t) il];
    const int64_t row = hd * h.n_head_kv;
    ggml_build_forward_expand(gf, ggml_cpy(ctx, ggml_reshape_2d(ctx, Kc, row, n), ggml_view_2d(ctx, kc, row, n, kc->nb[1], kc->nb[1] * (size_t) n_past_)));
    ggml_build_forward_expand(gf, ggml_cpy(ctx, ggml_reshape_2d(ctx, Vc, row, n), ggml_view_2d(ctx, vc, row, n, vc->nb[1], vc->nb[1] * (size_t) n_past_)));
    ggml_tensor * K = ggml_view_3d(ctx, kc, hd, h.n_head_kv, n_kv, ggml_row_size(kc->type, hd), kc->nb[1], 0);
    ggml_tensor * V = ggml_view_3d(ctx, vc, hd, h.n_head_kv, n_kv, ggml_row_size(vc->type, hd), vc->nb[1], 0);
    ggml_tensor * a = ggml_flash_attn_ext(ctx, ggml_permute(ctx, Q, 0, 2, 1, 3), ggml_permute(ctx, K, 0, 2, 1, 3),
                                          ggml_permute(ctx, V, 0, 2, 1, 3), mask, 1.0f / sqrtf((float) hd), 0.0f, 0.0f);
    ggml_prec_set_acc(a, GGML_PREC_F32);
    a = ggml_reshape_2d(ctx, a, a->ne[0] * a->ne[1], a->ne[2] * a->ne[3]);
    a = ggml_mul(ctx, a, ggml_sigmoid(ctx, gate));
    return ggml_mul_mat(ctx, L.wo, a);
}

ggml_tensor * Qwen4Exp::moe(ggml_context * ctx, int64_t il, ggml_tensor * cur, int n) {
    const auto &          h  = hp_;
    const Qwen4ExpLayer & L  = layers_[(size_t) il];
    const int64_t         ne = h.n_embd, k = h.n_expert_used;

    ggml_tensor * probs = ggml_soft_max(ctx, ggml_mul_mat(ctx, L.gate_inp, cur));                 // [n_expert, n]
    ggml_tensor * sel   = ggml_argsort_top_k(ctx, probs, (int) k);                                 // [k, n]
    ggml_tensor * w     = ggml_get_rows(ctx, ggml_reshape_3d(ctx, probs, 1, h.n_expert, n), sel);  // [1, k, n]
    w = ggml_reshape_2d(ctx, w, k, n);
    w = ggml_div(ctx, w, ggml_clamp(ctx, ggml_sum_rows(ctx, w), 6.103515625e-5f, INFINITY));
    if (h.expert_weights_scale != 0.0f && h.expert_weights_scale != 1.0f) {
        w = ggml_scale(ctx, w, h.expert_weights_scale);
    }
    w = ggml_reshape_3d(ctx, w, 1, k, n);

    ggml_tensor * x3 = ggml_reshape_3d(ctx, cur, ne, 1, n);
    ggml_tensor * gt = nullptr, * up = nullptr;
    if (L.gate_up_exps) {
        ggml_tensor * gu = ggml_mul_mat_id(ctx, L.gate_up_exps, x3, sel);  // [2*n_ff, k, n]: gate | up
        const int64_t ff = gu->ne[0] / 2;
        gt = ggml_view_3d(ctx, gu, ff, k, n, gu->nb[1], gu->nb[2], 0);
        up = ggml_view_3d(ctx, gu, ff, k, n, gu->nb[1], gu->nb[2], ggml_row_size(gu->type, ff));
    } else {
        gt = ggml_mul_mat_id(ctx, L.gate_exps, x3, sel);
        up = ggml_mul_mat_id(ctx, L.up_exps, x3, sel);
    }
    ggml_tensor * down = ggml_mul_mat_id(ctx, L.down_exps, ggml_swiglu_split(ctx, gt, up), sel);  // [ne, k, n]
    down = ggml_mul(ctx, down, w);
    ggml_tensor * out = ggml_view_2d(ctx, down, ne, n, down->nb[2], 0);
    for (int64_t i = 1; i < k; i++) {
        out = ggml_add(ctx, out, ggml_view_2d(ctx, down, ne, n, down->nb[2], (size_t) i * down->nb[1]));
    }
    if (L.up_shexp) {
        ggml_tensor * sh = ggml_mul_mat(ctx, L.down_shexp, ggml_swiglu_split(ctx, ggml_mul_mat(ctx, L.gate_shexp, cur), ggml_mul_mat(ctx, L.up_shexp, cur)));
        if (L.gate_inp_shexp) {
            sh = ggml_mul(ctx, sh, ggml_sigmoid(ctx, ggml_mul_mat(ctx, ggml_reshape_2d(ctx, L.gate_inp_shexp, ne, 1), cur)));
        }
        out = ggml_add(ctx, out, sh);
    }
    return out;
}

ggml_tensor * Qwen4Exp::ple(ggml_context * ctx, ggml_cgraph * gf, int64_t il, ggml_tensor * res, ggml_tensor * emb, int n) {
    const auto &          h  = hp_;
    const Qwen4ExpLayer & L  = layers_[(size_t) il];
    const int64_t         ne = h.n_embd, hc = h.hc, hcd = hc * ne;
    auto gnorm = [&](ggml_tensor * x, ggml_tensor * w) {
        return ggml_mul(ctx, ggml_rms_norm(ctx, ggml_reshape_3d(ctx, x, ne, hc, n), h.rms_eps), ggml_reshape_2d(ctx, w, ne, hc));
    };
    ggml_tensor * key   = gnorm(ggml_mul_mat(ctx, L.ple_key, emb), L.ple_norm_key);  // [ne, hc, n]
    ggml_tensor * value = ggml_mul_mat(ctx, L.ple_value, emb);                       // [ne, n]
    ggml_tensor * query = gnorm(res, L.ple_norm_query);
    ggml_tensor * s     = ggml_scale(ctx, ggml_sum_rows(ctx, ggml_mul(ctx, key, query)), 1.0f / sqrtf((float) ne));  // [1, hc, n]
    ggml_tensor * mag   = ggml_sqrt(ctx, ggml_clamp(ctx, ggml_abs(ctx, s), 1e-6f, INFINITY));
    ggml_tensor * gate  = ggml_sigmoid(ctx, ggml_mul(ctx, ggml_sgn(ctx, s), mag));
    ggml_tensor * gated = ggml_mul(ctx, ggml_repeat_4d(ctx, ggml_reshape_3d(ctx, value, ne, 1, n), ne, hc, n, 1), gate);
    ggml_tensor * normed = ggml_reshape_2d(ctx, gnorm(ggml_reshape_2d(ctx, gated, hcd, n), L.ple_norm_conv), hcd, n);

    // dilated causal depthwise conv over [history | batch]: out[c, t] = sum_k w[k, c] * x[c, t - (K-1-k)*dilation]
    const int64_t K = h.ple_conv_kernel, dil = h.ple_ngram, hist = (K - 1) * dil;
    ggml_tensor * st     = ple_state_[(size_t) il];
    ggml_tensor * padded = ggml_concat(ctx, st, ggml_transpose(ctx, normed), 0);  // [hist + n, hcd]
    ggml_build_forward_expand(gf, ggml_cpy(ctx, ggml_view_2d(ctx, padded, hist, hcd, padded->nb[1], ggml_row_size(padded->type, n)), st));
    ggml_tensor * conv = nullptr;
    for (int64_t k = 0; k < K; k++) {
        const int64_t start   = hist - (K - 1 - k) * dil;
        ggml_tensor * shifted = ggml_cont(ctx, ggml_transpose(ctx, ggml_view_2d(ctx, padded, n, hcd, padded->nb[1], ggml_row_size(padded->type, start))));
        ggml_tensor * wk      = ggml_reshape_1d(ctx, ggml_cont(ctx, ggml_view_2d(ctx, L.ple_conv1d, 1, hcd, L.ple_conv1d->nb[1], (size_t) k * L.ple_conv1d->nb[0])), hcd);
        if (wk->type != GGML_TYPE_F32) wk = ggml_cast(ctx, wk, GGML_TYPE_F32);
        ggml_tensor * term = ggml_mul(ctx, shifted, wk);
        conv               = conv ? ggml_add(ctx, conv, term) : term;
    }
    conv = ggml_reshape_3d(ctx, ggml_silu(ctx, conv), ne, hc, n);
    return ggml_add(ctx, res, ggml_add(ctx, gated, conv));
}

void Qwen4Exp::ple_rows(const int32_t * tokens, int n, std::vector<float> & out) const {
    const auto &  h      = hp_;
    const int64_t ng     = h.ple_ngram, nh = h.ple_n_heads(), hd = h.ple_head_dim;
    const size_t  rs     = ggml_row_size(ple_table_->type, hd);
    const auto *  tt     = ggml_get_type_traits(ple_table_->type);
    const char *  table  = (const char *) file_.data(ple_table_);
    const int64_t rows   = ple_table_->ne[1];
    out.assign((size_t) (n * nh * hd), 0.0f);
    std::vector<int64_t> ctxv((size_t) ng);
    for (int t = 0; t < n; t++) {
        const int64_t p = n_past_ + t;
        auto tok_at     = [&](int64_t q) -> int64_t {  // -1 when before the sequence start
            if (q < 0) return -1;
            return q < n_past_ ? history_[(size_t) q] : tokens[q - n_past_];
        };
        ctxv[0]  = tokens[t];
        bool cut = false;
        for (int64_t s = 1; s < ng; s++) {
            const int64_t tk = cut ? -1 : tok_at(p - s);
            cut              = cut || tk < 0 || tk == h.ple_eos;
            ctxv[(size_t) s] = cut ? h.ple_eos : tk;
        }
        for (int64_t g = 2; g <= ng; g++) {
            uint64_t mixed = (uint64_t) ctxv[0] * h.ple_mult[0];
            for (int64_t j = 1; j < g; j++) mixed ^= (uint64_t) ctxv[(size_t) j] * h.ple_mult[(size_t) j];
            for (int64_t q = 0; q < h.ple_per_gram; q++) {
                const int64_t hi  = (g - 2) * h.ple_per_gram + q;
                const int64_t row = (int64_t) (mixed % h.ple_vocab[(size_t) hi] + h.ple_offset[(size_t) hi]);
                float *       dst = out.data() + ((size_t) t * nh + hi) * hd;
                if (row < 0 || row >= rows) continue;
                const char * src = table + rs * (size_t) row;
                if (ple_table_->type == GGML_TYPE_F32) std::memcpy(dst, src, sizeof(float) * hd);
                else tt->to_float(src, dst, hd);
            }
        }
    }
}

ggml_cgraph * Qwen4Exp::build_graph(ggml_context * ctx, int n, int n_kv, bool sparse, Inputs & in, ggml_tensor *& out) {
    const auto &  h  = hp_;
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, kGraphSize, false);

    in.embd = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, h.n_embd, n);
    ggml_set_input(in.embd);
    in.pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int64_t) n * 4);
    ggml_set_input(in.pos);
    in.mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_kv, n);
    ggml_set_input(in.mask);
    if (std::find(h.ple_layer.begin(), h.ple_layer.end(), true) != h.ple_layer.end()) {
        in.ple = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, h.n_embd, n);
        ggml_set_input(in.ple);
    }
    if (kpool_ > 0) {
        const int64_t nnew = (n_past_ + n) / kpool_ - n_past_ / kpool_;
        if (nnew > 0) {
            in.pool_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4 * nnew);
            ggml_set_input(in.pool_pos);
        }
        if (sparse) {
            in.bvis = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, (n_past_ + n) / kpool_, n);
            ggml_set_input(in.bvis);
            in.tail = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_kv, n);
            ggml_set_input(in.tail);
        }
    }

    // the wide residual starts as hc copies of the embedding
    ggml_tensor * res = ggml_repeat_4d(ctx, ggml_reshape_3d(ctx, in.embd, h.n_embd, 1, n), h.n_embd, h.hc, n, 1);
    for (int64_t il = 0; il < h.n_layer; il++) {
        const Qwen4ExpLayer & L = layers_[(size_t) il];
        if (h.ple_layer[(size_t) il]) {
            res = ple(ctx, gf, il, res, in.ple, n);
        }
        ggml_tensor * inject = nullptr;
        ggml_tensor * cur    = hc_mix(ctx, res, L.hca_norm, L.hca_down, L.hca_up, L.hca_inject, &inject);
        cur = h.recurrent[(size_t) il] ? gdn(ctx, gf, il, cur, n) : attn(ctx, gf, il, cur, n, n_kv, sparse, in);
        res = hc_combine(ctx, res, cur, inject);
        cur = hc_mix(ctx, res, L.hcf_norm, L.hcf_down, L.hcf_up, L.hcf_inject, &inject);
        cur = moe(ctx, il, cur, n);
        res = hc_combine(ctx, res, cur, inject);
    }
    // the final mixer is the output norm
    ggml_tensor * cur = hc_mix(ctx, res, out_hc_norm_, out_hc_down_, out_hc_up_, nullptr, nullptr);
    out               = ggml_mul_mat(ctx, output_, cur);
    ggml_set_output(out);
    ggml_build_forward_expand(gf, out);
    return gf;
}

bool Qwen4Exp::eval(const int32_t * tokens, int n, float * logits, std::string & err) {
    const auto & h = hp_;
    if (n <= 0 || n > opt_.n_ubatch) {
        err = "eval: batch size must be 1.." + std::to_string(opt_.n_ubatch);
        return false;
    }
    if (n_past_ + n > n_ctx_) {
        err = "eval: context full (" + std::to_string(n_ctx_) + " tokens)";
        return false;
    }
    const int n_kv = std::min(n_ctx_, (n_past_ + n + kKvPad - 1) / kKvPad * kKvPad);
    // QSA is dense while every token's complete blocks fit the budget
    bool sparse = false;
    if (kpool_ > 0) {
        const int64_t K = h.idx_budget / kpool_;
        sparse          = (n_past_ + n) / kpool_ > K;
    }

    ggml_init_params ip  = { graph_meta_.size(), graph_meta_.data(), true };
    ggml_context *   ctx = ggml_init(ip);
    Inputs           in;
    ggml_tensor *    out = nullptr;
    ggml_cgraph *    gf  = build_graph(ctx, n, n_kv, sparse, in, out);

    ggml_backend_sched_reset(sched_);
    if (!ggml_backend_sched_alloc_graph(sched_, gf)) {
        ggml_free(ctx);
        err = "eval: cannot allocate the compute graph";
        return false;
    }
    auto set = [](ggml_tensor * t, const void * d, size_t nb) {
        if (t && t->buffer) ggml_backend_tensor_set(t, d, 0, nb);
    };
    {  // token embeddings from the mapping
        const size_t       rs = ggml_row_size(tok_embd_->type, h.n_embd);
        const auto *       tt = ggml_get_type_traits(tok_embd_->type);
        const char *       tb = (const char *) file_.data(tok_embd_);
        std::vector<float> e((size_t) (h.n_embd * n));
        for (int i = 0; i < n; i++) {
            if (tokens[i] < 0 || tokens[i] >= h.n_vocab) {
                ggml_free(ctx);
                err = "eval: token id out of range";
                return false;
            }
            const char * row = tb + rs * (size_t) tokens[i];
            if (tok_embd_->type == GGML_TYPE_F32) std::memcpy(e.data() + (size_t) i * h.n_embd, row, sizeof(float) * h.n_embd);
            else tt->to_float(row, e.data() + (size_t) i * h.n_embd, h.n_embd);
        }
        set(in.embd, e.data(), e.size() * sizeof(float));
    }
    std::vector<int32_t> pos((size_t) n * 4, 0);
    for (int i = 0; i < n; i++) pos[(size_t) i] = pos[(size_t) n + i] = pos[(size_t) 2 * n + i] = n_past_ + i;
    set(in.pos, pos.data(), pos.size() * sizeof(int32_t));
    std::vector<ggml_fp16_t> mask((size_t) n_kv * n);
    const ggml_fp16_t        zero = ggml_fp32_to_fp16(0.0f), ninf = ggml_fp32_to_fp16(-INFINITY);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n_kv; j++) mask[(size_t) i * n_kv + j] = j <= n_past_ + i ? zero : ninf;
    }
    set(in.mask, mask.data(), mask.size() * sizeof(ggml_fp16_t));
    if (in.ple) {
        std::vector<float> pr;
        ple_rows(tokens, n, pr);
        set(in.ple, pr.data(), pr.size() * sizeof(float));
    }
    if (in.pool_pos) {  // each new pooled key is rotated to its block's first position
        const int64_t        b0 = n_past_ / kpool_, nnew = in.pool_pos->ne[0] / 4;
        std::vector<int32_t> pp((size_t) (4 * nnew), 0);
        for (int64_t i = 0; i < nnew; i++) pp[(size_t) i] = pp[(size_t) (nnew + i)] = pp[(size_t) (2 * nnew + i)] = (int32_t) ((b0 + i) * kpool_);
        set(in.pool_pos, pp.data(), pp.size() * sizeof(int32_t));
    }
    if (in.bvis) {
        const int64_t      NB = in.bvis->ne[0];
        std::vector<float> bv((size_t) (NB * n)), tl((size_t) n_kv * n);
        for (int i = 0; i < n; i++) {
            const int64_t p = n_past_ + i, nb = (p + 1) / kpool_;
            // visible blocks get -1e-30*b: ties (typically several blocks whose relu-summed score is exactly 0) go
            // to the lower index, as torch.topk does in the reference; nonzero scores are unaffected (below 1 ulp)
            for (int64_t b = 0; b < NB; b++) bv[(size_t) (i * NB + b)] = b < nb ? -1e-30f * (float) b : -INFINITY;
            for (int64_t j = 0; j < n_kv; j++) tl[(size_t) i * n_kv + j] = (j >= nb * kpool_ && j <= p) ? 0.0f : -INFINITY;
        }
        set(in.bvis, bv.data(), bv.size() * sizeof(float));
        set(in.tail, tl.data(), tl.size() * sizeof(float));
    }

    if (ggml_backend_sched_graph_compute(sched_, gf) != GGML_STATUS_SUCCESS) {
        ggml_free(ctx);
        err = "eval: graph compute failed";
        return false;
    }
    if (logits) ggml_backend_tensor_get(out, logits, 0, ggml_nbytes(out));
    if (dbg_mask_) {
        std::vector<ggml_fp16_t> mk((size_t) ggml_nelements(dbg_mask_));
        ggml_backend_tensor_get(dbg_mask_, mk.data(), 0, ggml_nbytes(dbg_mask_));
        std::vector<float> sc((size_t) ggml_nelements(dbg_scores_));
        ggml_backend_tensor_get(dbg_scores_, sc.data(), 0, ggml_nbytes(dbg_scores_));
        const int64_t nkv = dbg_mask_->ne[0], NB = dbg_scores_->ne[0];
        for (int i = 0; i < n && i < 64; i++) {
            fprintf(stderr, "qsa pos %d scores:", n_past_ + i);
            for (int64_t b = 0; b < NB; b++) fprintf(stderr, " %.4f", sc[(size_t) (i * NB + b)]);
            fprintf(stderr, "\n   selected:");
            for (int64_t j = 0; j < nkv; j++) {
                if (ggml_fp16_to_fp32(mk[(size_t) (i * nkv + j)]) == 0.0f) fprintf(stderr, " %lld", (long long) j);
            }
            fprintf(stderr, "\n");
        }
        dbg_mask_ = dbg_scores_ = nullptr;
    }
    ggml_free(ctx);
    history_.insert(history_.end(), tokens, tokens + n);
    n_past_ += n;
    return true;
}

// ---- factory -------------------------------------------------------------------------------------------------

std::unique_ptr<CausalLM> load_causal_lm(const std::string & path, const LoadOptions & opt, std::string & err) {
    GgufFile f;
    if (!f.open(path, err)) return nullptr;
    const std::string arch = f.arch();
    if (arch == "qwen35") {
        auto m = std::make_unique<Qwen35>();
        if (!m->load(path, opt, err)) return nullptr;
        return m;
    }
    if (arch == "qwen4exp") {
        auto m = std::make_unique<Qwen4Exp>();
        if (!m->load(path, opt, err)) return nullptr;
        return m;
    }
    err = "unsupported architecture \"" + arch + "\"";
    return nullptr;
}

} // namespace e8::model
