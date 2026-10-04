#include "model/qwen35.h"

#include "kernels/q4k_small.h"

#include "ggml-alloc.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

namespace e8::model {

namespace {

constexpr int    kGraphSize = 16384;
constexpr int    kKvPad     = 256;  // KV length handed to attention is padded to this (masked)
constexpr size_t kReadChunk = 64u << 20;

std::string key(const std::string & arch, const char * k) {
    return arch + "." + k;
}

FILE * open_read(const std::string & path) {
#if defined(_WIN32)
    const int    n = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
    std::wstring w((size_t) n, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, w.data(), n);
    return _wfopen(w.c_str(), L"rb");
#else
    return fopen(path.c_str(), "rb");
#endif
}

bool seek(FILE * f, uint64_t off) {
#if defined(_WIN32)
    return _fseeki64(f, (long long) off, SEEK_SET) == 0;
#else
    return fseeko(f, (off_t) off, SEEK_SET) == 0;
#endif
}

// Reads each (dst, file meta) tensor's bytes from `path` into its backend buffer.
bool read_tensors(const std::string & path, const GgufFile & file,
                  const std::vector<std::pair<ggml_tensor *, const ggml_tensor *>> & list, std::string & err) {
    FILE * fp = open_read(path);
    if (!fp) {
        err = "cannot open " + path;
        return false;
    }
    std::vector<uint8_t> chunk(kReadChunk);
    for (auto & [t, m] : list) {
        const size_t nb = ggml_nbytes(t);
        if (!seek(fp, file.data_offset(m))) {
            fclose(fp);
            err = std::string("seek failed for ") + ggml_get_name(t);
            return false;
        }
        for (size_t done = 0; done < nb;) {
            const size_t n = std::min(kReadChunk, nb - done);
            if (fread(chunk.data(), 1, n, fp) != n) {
                fclose(fp);
                err = std::string("short read for ") + ggml_get_name(t);
                return false;
            }
            ggml_backend_tensor_set(t, chunk.data(), done, n);
            done += n;
        }
    }
    fclose(fp);
    return true;
}

} // namespace

bool Qwen35Hparams::load(const GgufFile & f, std::string & err) {
    const std::string a = f.arch();
    if (a != "qwen35") {
        err = "architecture \"" + a + "\" is not qwen35";
        return false;
    }
    const int64_t blocks = f.i64(key(a, "block_count"), 0);
    n_layer_nextn        = f.i64(key(a, "nextn_predict_layers"), 0);
    n_layer              = blocks - n_layer_nextn;
    n_embd               = f.i64(key(a, "embedding_length"), 0);
    n_ff                 = f.i64(key(a, "feed_forward_length"), 0);
    n_head               = f.i64(key(a, "attention.head_count"), 0);
    n_head_kv            = f.i64(key(a, "attention.head_count_kv"), 0);
    head_dim             = f.i64(key(a, "attention.key_length"), 0);
    n_ctx_train          = f.i64(key(a, "context_length"), 0);
    n_rot                = f.i64(key(a, "rope.dimension_count"), head_dim);
    rope_freq_base       = f.f32(key(a, "rope.freq_base"), 10000.0f);
    rms_eps              = f.f32(key(a, "attention.layer_norm_rms_epsilon"), 1e-6f);
    full_attn_interval   = f.i64(key(a, "full_attention_interval"), 4);
    ssm_d_conv           = f.i64(key(a, "ssm.conv_kernel"), 0);
    ssm_d_inner          = f.i64(key(a, "ssm.inner_size"), 0);
    ssm_d_state          = f.i64(key(a, "ssm.state_size"), 0);
    ssm_n_v              = f.i64(key(a, "ssm.time_step_rank"), 0);
    ssm_n_k              = f.i64(key(a, "ssm.group_count"), 0);
    bos                  = f.i64("tokenizer.ggml.bos_token_id", -1);
    eos                  = f.i64("tokenizer.ggml.eos_token_id", -1);
    add_bos              = f.b("tokenizer.ggml.add_bos_token", false);

    const std::vector<int64_t> sec = f.i64_arr(key(a, "rope.dimension_sections"));
    for (size_t i = 0; i < 4 && i < sec.size(); i++) {
        rope_sections[i] = (int) sec[i];
    }
    recurrent.clear();
    for (int64_t v : f.i64_arr(key(a, "attention.recurrent_layers"))) {
        recurrent.push_back(v != 0);
    }
    const ggml_tensor * emb = f.tensor("token_embd.weight");
    n_vocab                 = emb ? emb->ne[1] : 0;

    if (n_layer <= 0 || n_embd <= 0 || n_ff <= 0 || n_head <= 0 || n_head_kv <= 0 || head_dim <= 0 || n_vocab <= 0 ||
        ssm_d_conv <= 1 || ssm_d_state <= 0 || ssm_n_v <= 0 || ssm_n_k <= 0 || ssm_n_v % ssm_n_k != 0 ||
        ssm_d_inner != ssm_d_state * ssm_n_v || f.i64(key(a, "attention.value_length"), head_dim) != head_dim) {
        err = "incomplete or unsupported qwen35 hyperparameters";
        return false;
    }
    return true;
}

Qwen35::Qwen35() = default;

Qwen35::~Qwen35() {
    if (sched_) {
        ggml_backend_sched_free(sched_);
    }
    if (rbuf_) ggml_backend_buffer_free(rbuf_);
    if (rctx_) ggml_free(rctx_);
    for (int i = 0; i < 2; i++) {
        if (wbuf_[i]) ggml_backend_buffer_free(wbuf_[i]);
        if (sbuf_[i]) ggml_backend_buffer_free(sbuf_[i]);
        if (wctx_[i]) ggml_free(wctx_[i]);
        if (sctx_[i]) ggml_free(sctx_[i]);
    }
    if (gpu_) ggml_backend_free(gpu_);
    if (cpu_) ggml_backend_free(cpu_);
}

bool Qwen35::load(const std::string & path, const LoadOptions & opt, std::string & err) {
    opt_ = opt;
    if (!file_.open(path, err) || !hp_.load(file_, err)) {
        return false;
    }
    const auto & h = hp_;

    // backends
    cpu_ = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!cpu_) {
        err = "cannot initialize the CPU backend";
        return false;
    }
    int threads = opt.n_threads > 0 ? opt.n_threads : (int) std::max(1u, std::thread::hardware_concurrency());
    ggml_backend_cpu_set_n_threads(cpu_, threads);
    if (ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU)) {
        gpu_ = ggml_backend_dev_init(dev, nullptr);
    }
    const int  n_gpu_layers = gpu_ ? std::clamp(opt.n_gpu_layers, 0, (int) h.n_layer) : 0;
    const bool out_gpu      = gpu_ && opt.output_gpu;

    // weight tensors: duplicated from the file's metadata into a GPU and a CPU context
    const size_t        n_tens = (size_t) file_.n_tensors() + 8;
    ggml_init_params    ip     = { ggml_tensor_overhead() * n_tens, nullptr, true };
    wctx_[0]                   = ggml_init(ip);
    wctx_[1]                   = ggml_init(ip);
    std::vector<std::pair<ggml_tensor *, const ggml_tensor *>> to_load;  // (dst, file meta)
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
        to_load.emplace_back(t, m);
        return t;
    };

    tok_embd_ = make("token_embd.weight", false);
    out_norm_ = make("output_norm.weight", out_gpu);
    output_   = make("output.weight", out_gpu, false);
    if (!output_) {
        output_ = make("token_embd.weight", out_gpu);  // tied embeddings
    }
    layers_.resize((size_t) h.n_layer);
    for (int64_t il = 0; il < h.n_layer; il++) {
        Qwen35Layer &     L   = layers_[(size_t) il];
        const bool        g   = il >= h.n_layer - n_gpu_layers;
        const std::string p   = "blk." + std::to_string(il) + ".";
        L.on_gpu              = g;
        L.attn_norm           = make(p + "attn_norm.weight", g);
        L.post_norm           = make(p + "post_attention_norm.weight", g);
        if (h.is_recurrent(il)) {
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
        }
        L.ffn_gate = make(p + "ffn_gate.weight", g);
        L.ffn_up   = make(p + "ffn_up.weight", g);
        L.ffn_down = make(p + "ffn_down.weight", g);
        if (missing) {
            return false;
        }
    }
    if (missing) {
        return false;
    }

    // allocate, then read every tensor's bytes from the file
    for (int i = 0; i < 2; i++) {
        ggml_backend_t be = i == 0 ? gpu_ : cpu_;
        if (!be || !ggml_get_first_tensor(wctx_[i])) {
            continue;
        }
        wbuf_[i] = ggml_backend_alloc_ctx_tensors_from_buft(wctx_[i], ggml_backend_get_default_buffer_type(be));
        if (!wbuf_[i]) {
            err = i == 0 ? "not enough VRAM for the GPU layers (lower --gpu-layers)" : "not enough RAM for the weights";
            return false;
        }
        ggml_backend_buffer_set_usage(wbuf_[i], GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        (i == 0 ? gpu_bytes_ : cpu_bytes_) = ggml_backend_buffer_get_size(wbuf_[i]);
    }
    if (!read_tensors(path, file_, to_load, err)) {
        return false;
    }
    if (!opt.residual_path.empty() && !load_residual(opt.residual_path, err)) {
        return false;
    }

    // state: KV cache for attention layers, conv + delta-net state for recurrent layers
    n_ctx_                 = std::max(kKvPad, (opt.n_ctx + kKvPad - 1) / kKvPad * kKvPad);
    ggml_init_params sp    = { ggml_tensor_overhead() * (size_t) (h.n_layer * 8 + 8), nullptr, true };
    sctx_[0]               = ggml_init(sp);
    sctx_[1]               = ggml_init(sp);
    k_cache_.assign((size_t) h.n_layer, nullptr);
    v_cache_.assign((size_t) h.n_layer, nullptr);
    conv_state_.assign((size_t) h.n_layer, nullptr);
    ssm_state_.assign((size_t) h.n_layer, nullptr);
    conv_bak_.assign((size_t) h.n_layer, nullptr);
    ssm_bak_.assign((size_t) h.n_layer, nullptr);
    rec_qkv_.assign((size_t) h.n_layer, nullptr);
    rec_g_.assign((size_t) h.n_layer, nullptr);
    rec_beta_.assign((size_t) h.n_layer, nullptr);
    const int64_t n_rec = std::max(1, opt.max_record);
    for (int64_t il = 0; il < h.n_layer; il++) {
        ggml_context * c = sctx_[layers_[(size_t) il].on_gpu ? 0 : 1];
        if (h.is_recurrent(il)) {
            conv_state_[(size_t) il] = ggml_new_tensor_1d(c, GGML_TYPE_F32, (h.ssm_d_conv - 1) * h.conv_channels());
            ssm_state_[(size_t) il]  = ggml_new_tensor_1d(c, GGML_TYPE_F32, h.ssm_d_state * h.ssm_d_state * h.ssm_n_v);
            conv_bak_[(size_t) il]   = ggml_dup_tensor(c, conv_state_[(size_t) il]);
            ssm_bak_[(size_t) il]    = ggml_dup_tensor(c, ssm_state_[(size_t) il]);
            rec_qkv_[(size_t) il]    = ggml_new_tensor_2d(c, GGML_TYPE_F32, h.conv_channels(), n_rec);
            rec_g_[(size_t) il]      = ggml_new_tensor_2d(c, GGML_TYPE_F32, h.ssm_n_v, n_rec);
            rec_beta_[(size_t) il]   = ggml_new_tensor_2d(c, GGML_TYPE_F32, h.ssm_n_v, n_rec);
        } else {
            k_cache_[(size_t) il] = ggml_new_tensor_2d(c, opt.kv_type, h.head_dim * h.n_head_kv, n_ctx_);
            v_cache_[(size_t) il] = ggml_new_tensor_2d(c, opt.kv_type, h.head_dim * h.n_head_kv, n_ctx_);
        }
    }
    for (int i = 0; i < 2; i++) {
        ggml_backend_t be = i == 0 ? gpu_ : cpu_;
        if (!be || !ggml_get_first_tensor(sctx_[i])) {
            continue;
        }
        sbuf_[i] = ggml_backend_alloc_ctx_tensors_from_buft(sctx_[i], ggml_backend_get_default_buffer_type(be));
        if (!sbuf_[i]) {
            err = "not enough memory for the KV cache / recurrent state (lower --ctx)";
            return false;
        }
    }
    reset();

    // scheduler: GPU first, CPU last; large CPU-weight matmuls may be offloaded to the GPU
    std::vector<ggml_backend_t> bes;
    if (gpu_) {
        bes.push_back(gpu_);
    }
    bes.push_back(cpu_);
    sched_ = ggml_backend_sched_new(bes.data(), nullptr, (int) bes.size(), kGraphSize, false, true);
    graph_meta_.resize(ggml_tensor_overhead() * kGraphSize + ggml_graph_overhead_custom(kGraphSize, false));
    return true;
}

void Qwen35::reset() {
    for (int i = 0; i < 2; i++) {
        if (sbuf_[i]) {
            ggml_backend_buffer_clear(sbuf_[i], 0);
        }
    }
    n_past_       = 0;
    saved_n_past_ = 0;
    recorded_n_   = 0;
}

bool Qwen35::load_residual(const std::string & path, std::string & err) {
    GgufFile rf;
    if (!rf.open(path, err)) {
        return false;
    }
    const std::string id_b = file_.str("eightfer.pack.id", ""), id_r = rf.str("eightfer.pack.id", "");
    if (rf.str("general.type", "") != "eightfer-residual" || id_r.empty()) {
        err = path + " is not an eightfer residual file";
        return false;
    }
    if (id_b != id_r) {
        err = "residual " + path + " was packed against a different base (pack id " + id_r + ", base has \"" + id_b + "\")";
        return false;
    }
    ggml_init_params ip = { ggml_tensor_overhead() * (size_t) (rf.n_tensors() + 8), nullptr, true };
    rctx_               = ggml_init(ip);
    std::vector<std::pair<ggml_tensor *, const ggml_tensor *>> to_load;
    for (int64_t i = 0; i < rf.n_tensors(); i++) {
        const ggml_tensor * m    = rf.tensor(i);
        ggml_tensor *       base = nullptr;
        for (ggml_context * c : wctx_) {
            if (c && !base) base = ggml_get_tensor(c, ggml_get_name(m));
        }
        if (!base) {
            continue;  // e.g. MTP tensors, which the trunk does not load
        }
        if (!ggml_are_same_shape(base, m)) {
            err = std::string("residual shape mismatch for ") + ggml_get_name(m);
            return false;
        }
        ggml_tensor * r = ggml_dup_tensor(rctx_, m);
        ggml_set_name(r, (std::string(ggml_get_name(m)) + ".res").c_str());
        res_[base] = r;
        to_load.emplace_back(r, m);
    }
    // pinned host memory when a GPU exists (fast GPU streaming for big batches); plain RAM otherwise or if the
    // pinned allocation is refused (Windows caps pinned memory at half of RAM)
    ggml_backend_buffer_type_t buft = nullptr;
    if (gpu_) {
        buft = ggml_backend_dev_host_buffer_type(ggml_backend_get_device(gpu_));
    }
    if (buft) {
        rbuf_ = ggml_backend_alloc_ctx_tensors_from_buft(rctx_, buft);
        if (!rbuf_) {
            fprintf(stderr, "warning: pinned allocation for the residual failed, using pageable RAM\n");
        }
    }
    if (!rbuf_) {
        rbuf_ = ggml_backend_alloc_ctx_tensors_from_buft(rctx_, ggml_backend_get_default_buffer_type(cpu_));
    }
    if (!rbuf_) {
        err = "not enough RAM for the residual";
        return false;
    }
    ggml_backend_buffer_set_usage(rbuf_, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    res_bytes_ = ggml_backend_buffer_get_size(rbuf_);
    return read_tensors(path, rf, to_load, err);
}

ggml_tensor * Qwen35::mm(ggml_context * ctx, ggml_tensor * w, ggml_tensor * x) {
    ggml_tensor * y = ggml_mul_mat(ctx, w, x);
    if (use_res_) {
        auto it = res_.find(w);
        if (it != res_.end()) {
            // small batches (decode, speculative verify): one-pass multi-column Q4_K kernel on the CPU; big batches:
            // ggml mul_mat, which the scheduler streams to the GPU
            static const bool small_ok = std::getenv("E8_NO_SMALL_GEMM") == nullptr;
            ggml_tensor *     r        = it->second;
            y = ggml_add(ctx, y, small_ok && kernels::q4k_small_supported(r, x) ? kernels::q4k_mul_mat_small(ctx, r, x)
                                                                                : ggml_mul_mat(ctx, r, x));
        }
    }
    return y;
}

void Qwen35::save_state() {
    for (size_t il = 0; il < conv_state_.size(); il++) {
        if (conv_state_[il]) {
            ggml_backend_tensor_copy(conv_state_[il], conv_bak_[il]);
            ggml_backend_tensor_copy(ssm_state_[il], ssm_bak_[il]);
        }
    }
    saved_n_past_ = n_past_;
}

void Qwen35::restore_state() {
    for (size_t il = 0; il < conv_state_.size(); il++) {
        if (conv_state_[il]) {
            ggml_backend_tensor_copy(conv_bak_[il], conv_state_[il]);
            ggml_backend_tensor_copy(ssm_bak_[il], ssm_state_[il]);
        }
    }
    n_past_ = saved_n_past_;
}

bool Qwen35::compute(ggml_context * ctx, ggml_cgraph * gf, std::string & err) {
    (void) ctx;
    ggml_backend_sched_reset(sched_);
    if (!ggml_backend_sched_alloc_graph(sched_, gf)) {
        err = "cannot allocate the compute graph";
        return false;
    }
    if (ggml_backend_sched_graph_compute(sched_, gf) != GGML_STATUS_SUCCESS) {
        err = "graph compute failed";
        return false;
    }
    return true;
}

bool Qwen35::rollback(int keep, std::string & err) {
    if (recorded_n_ <= 0 || keep < 0 || keep > recorded_n_) {
        err = "rollback: no recorded eval or keep out of range";
        return false;
    }
    const int n = recorded_n_;
    recorded_n_ = 0;
    if (keep == n) {
        return true;
    }
    restore_state();
    if (keep > 0) {
        const auto &     h   = hp_;
        ggml_init_params ip  = { graph_meta_.size(), graph_meta_.data(), true };
        ggml_context *   ctx = ggml_init(ip);
        ggml_cgraph *    gf  = ggml_new_graph_custom(ctx, kGraphSize, false);
        for (int64_t il = 0; il < h.n_layer; il++) {
            if (!h.is_recurrent(il)) {
                continue;
            }
            const size_t  l    = (size_t) il;
            ggml_tensor * qkv  = ggml_view_2d(ctx, rec_qkv_[l], h.conv_channels(), keep, rec_qkv_[l]->nb[1], 0);
            ggml_tensor * g    = ggml_view_2d(ctx, rec_g_[l], h.ssm_n_v, keep, rec_g_[l]->nb[1], 0);
            ggml_tensor * beta = ggml_view_2d(ctx, rec_beta_[l], h.ssm_n_v, keep, rec_beta_[l]->nb[1], 0);
            gdn_core(ctx, gf, il, ggml_reshape_3d(ctx, qkv, h.conv_channels(), keep, 1),
                     ggml_reshape_4d(ctx, g, 1, h.ssm_n_v, keep, 1), ggml_reshape_4d(ctx, beta, 1, h.ssm_n_v, keep, 1),
                     keep);
        }
        const bool ok = compute(ctx, gf, err);
        ggml_free(ctx);
        if (!ok) {
            err = "rollback: " + err;
            return false;
        }
    }
    n_past_ = saved_n_past_ + keep;
    return true;
}

ggml_tensor * Qwen35::gdn_core(ggml_context * ctx, ggml_cgraph * gf, int64_t il, ggml_tensor * qkv, ggml_tensor * g,
                               ggml_tensor * beta, int n) {
    const auto &        h   = hp_;
    const Qwen35Layer & L   = layers_[(size_t) il];
    const float         eps = h.rms_eps;
    const int64_t       S = h.ssm_d_state, Hk = h.ssm_n_k, Hv = h.ssm_n_v, C = h.conv_channels();

    // causal conv over [previous d_conv-1 inputs | this batch], then keep the last d_conv-1 inputs
    ggml_tensor * cs   = conv_state_[(size_t) il];
    ggml_tensor * cst  = ggml_reshape_3d(ctx, cs, h.ssm_d_conv - 1, C, 1);
    ggml_tensor * cin  = ggml_concat(ctx, cst, ggml_transpose(ctx, qkv), 0);  // [d_conv-1+n, C, 1]
    ggml_tensor * last = ggml_view_3d(ctx, cin, h.ssm_d_conv - 1, C, 1, cin->nb[1], cin->nb[2],
                                      ggml_row_size(cin->type, cin->ne[0] - (h.ssm_d_conv - 1)));
    ggml_build_forward_expand(gf, ggml_cpy(ctx, last, cs));

    ggml_tensor * conv = ggml_silu(ctx, ggml_ssm_conv(ctx, cin, L.conv1d));  // [C, n, 1]
    const size_t  nb1  = ggml_row_size(conv->type, C);
    ggml_tensor * q = ggml_view_4d(ctx, conv, S, Hk, n, 1, ggml_row_size(conv->type, S), nb1, nb1 * n, 0);
    ggml_tensor * k = ggml_view_4d(ctx, conv, S, Hk, n, 1, ggml_row_size(conv->type, S), nb1, nb1 * n,
                                   ggml_row_size(conv->type, S * Hk));
    ggml_tensor * v = ggml_view_4d(ctx, conv, S, Hv, n, 1, ggml_row_size(conv->type, S), nb1, nb1 * n,
                                   ggml_row_size(conv->type, 2 * S * Hk));
    // l2 norm = rms_norm(x, eps/n) / sqrt(n)  (llama.cpp build_gdn_l2_norm)
    q = ggml_scale(ctx, ggml_rms_norm(ctx, q, eps / (float) S), 1.0f / sqrtf((float) S));
    k = ggml_scale(ctx, ggml_rms_norm(ctx, k, eps / (float) S), 1.0f / sqrtf((float) S));

    ggml_tensor * ss    = ssm_state_[(size_t) il];
    ggml_tensor * state = ggml_reshape_4d(ctx, ss, S, S, Hv, 1);
    ggml_tensor * res   = ggml_gated_delta_net(ctx, q, k, v, g, beta, state, 1);
    ggml_tensor * out   = ggml_view_4d(ctx, res, S, Hv, n, 1, ggml_row_size(res->type, S),
                                       ggml_row_size(res->type, S * Hv), ggml_row_size(res->type, S * Hv * n), 0);
    ggml_tensor * new_state = ggml_view_1d(ctx, res, S * S * Hv, ggml_row_size(res->type, S * Hv * n));
    ggml_build_forward_expand(gf, ggml_cpy(ctx, new_state, ss));
    return out;
}

ggml_cgraph * Qwen35::build_graph(ggml_context * ctx, int n, const EvalOpts & o, ggml_tensor *& inp_tok,
                                  ggml_tensor *& inp_pos, ggml_tensor *& inp_mask, ggml_tensor *& out_logits, int n_kv) {
    const auto &  h   = hp_;
    ggml_cgraph * gf  = ggml_new_graph_custom(ctx, kGraphSize, false);
    const float   eps = h.rms_eps;
    const int64_t hd  = h.head_dim;

    // embeddings are looked up on the host (eval()), so a fully offloaded graph has no CPU split
    inp_tok = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, h.n_embd, n);
    ggml_set_input(inp_tok);
    inp_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int64_t) n * 4);
    ggml_set_input(inp_pos);
    inp_mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, n_kv, n, 1, 1);
    ggml_set_input(inp_mask);

    auto norm = [&](ggml_tensor * x, ggml_tensor * w) { return ggml_mul(ctx, ggml_rms_norm(ctx, x, eps), w); };

    ggml_tensor * inpL = inp_tok;
    int sections[4] = { h.rope_sections[0], h.rope_sections[1], h.rope_sections[2], h.rope_sections[3] };

    for (int64_t il = 0; il < h.n_layer; il++) {
        const Qwen35Layer & L   = layers_[(size_t) il];
        ggml_tensor *       cur = norm(inpL, L.attn_norm);

        if (h.is_recurrent(il)) {
            // ---- Gated DeltaNet (llama.cpp qwen35::build_layer_attn_linear, one sequence)
            const int64_t S = h.ssm_d_state, Hv = h.ssm_n_v, C = h.conv_channels();

            ggml_tensor * qkv = mm(ctx, L.wqkv, cur);  // [C, n]
            ggml_tensor * z   = mm(ctx, L.wz, cur);    // [d_inner, n]

            ggml_tensor * beta  = ggml_sigmoid(ctx, mm(ctx, L.w_beta, cur));  // [Hv, n]
            ggml_tensor * alpha = ggml_reshape_3d(ctx, mm(ctx, L.w_alpha, cur), Hv, n, 1);
            ggml_tensor * g     = ggml_mul(ctx, ggml_softplus(ctx, ggml_add(ctx, alpha, L.dt_bias)), L.a);
            if (o.record) {
                const size_t l = (size_t) il;
                ggml_build_forward_expand(gf, ggml_cpy(ctx, qkv, ggml_view_2d(ctx, rec_qkv_[l], C, n, rec_qkv_[l]->nb[1], 0)));
                ggml_build_forward_expand(gf, ggml_cpy(ctx, ggml_reshape_2d(ctx, g, Hv, n),
                                                       ggml_view_2d(ctx, rec_g_[l], Hv, n, rec_g_[l]->nb[1], 0)));
                ggml_build_forward_expand(gf, ggml_cpy(ctx, beta, ggml_view_2d(ctx, rec_beta_[l], Hv, n, rec_beta_[l]->nb[1], 0)));
            }
            ggml_tensor * out = gdn_core(ctx, gf, il, ggml_reshape_3d(ctx, qkv, C, n, 1), ggml_reshape_4d(ctx, g, 1, Hv, n, 1),
                                         ggml_reshape_4d(ctx, beta, 1, Hv, n, 1), n);

            ggml_tensor * z4 = ggml_reshape_4d(ctx, z, S, Hv, n, 1);
            ggml_tensor * on = ggml_mul(ctx, norm(out, L.ssm_norm), ggml_silu(ctx, z4));
            on               = ggml_reshape_2d(ctx, on, S * Hv, n);
            cur              = mm(ctx, L.ssm_out, on);
        } else {
            // ---- gated full attention (llama.cpp qwen35::build_layer_attn)
            ggml_tensor * qg = mm(ctx, L.wq, cur);  // [2*hd*n_head, n]: per head [q | gate]
            qg               = ggml_reshape_3d(ctx, qg, hd * 2, h.n_head, n);
            ggml_tensor * Kc = ggml_reshape_3d(ctx, mm(ctx, L.wk, cur), hd, h.n_head_kv, n);
            ggml_tensor * Vc = ggml_reshape_3d(ctx, mm(ctx, L.wv, cur), hd, h.n_head_kv, n);
            const size_t  es = ggml_element_size(qg);
            ggml_tensor * Qc = ggml_view_3d(ctx, qg, hd, h.n_head, n, es * hd * 2, es * hd * 2 * h.n_head, 0);
            ggml_tensor * gate = ggml_view_3d(ctx, qg, hd, h.n_head, n, es * hd * 2, es * hd * 2 * h.n_head, es * hd);
            gate               = ggml_cont_2d(ctx, gate, hd * h.n_head, n);
            Qc                 = norm(Qc, L.q_norm);
            Kc                 = norm(Kc, L.k_norm);
            Qc = ggml_rope_multi(ctx, Qc, inp_pos, nullptr, (int) h.n_rot, sections, GGML_ROPE_TYPE_IMROPE,
                                 (int) h.n_ctx_train, h.rope_freq_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
            Kc = ggml_rope_multi(ctx, Kc, inp_pos, nullptr, (int) h.n_rot, sections, GGML_ROPE_TYPE_IMROPE,
                                 (int) h.n_ctx_train, h.rope_freq_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);

            // write this batch's K/V at [n_past, n_past + n), then attend over [0, n_kv)
            ggml_tensor * kc = k_cache_[(size_t) il];
            ggml_tensor * vc = v_cache_[(size_t) il];
            const int64_t row = hd * h.n_head_kv;
            ggml_build_forward_expand(gf, ggml_cpy(ctx, ggml_reshape_2d(ctx, Kc, row, n),
                                                   ggml_view_2d(ctx, kc, row, n, kc->nb[1], kc->nb[1] * n_past_)));
            ggml_build_forward_expand(gf, ggml_cpy(ctx, ggml_reshape_2d(ctx, Vc, row, n),
                                                   ggml_view_2d(ctx, vc, row, n, vc->nb[1], vc->nb[1] * n_past_)));
            ggml_tensor * K = ggml_view_3d(ctx, kc, hd, h.n_head_kv, n_kv, ggml_row_size(kc->type, hd), kc->nb[1], 0);
            ggml_tensor * V = ggml_view_3d(ctx, vc, hd, h.n_head_kv, n_kv, ggml_row_size(vc->type, hd), vc->nb[1], 0);

            ggml_tensor * q = ggml_permute(ctx, Qc, 0, 2, 1, 3);
            ggml_tensor * k = ggml_permute(ctx, K, 0, 2, 1, 3);
            ggml_tensor * v = ggml_permute(ctx, V, 0, 2, 1, 3);
            ggml_tensor * a = ggml_flash_attn_ext(ctx, q, k, v, inp_mask, 1.0f / sqrtf((float) hd), 0.0f, 0.0f);
            ggml_prec_set_acc(a, GGML_PREC_F32);
            a   = ggml_reshape_2d(ctx, a, a->ne[0] * a->ne[1], a->ne[2] * a->ne[3]);
            a   = ggml_mul(ctx, a, ggml_sigmoid(ctx, gate));
            cur = mm(ctx, L.wo, a);
        }

        cur                 = ggml_add(ctx, cur, inpL);  // attention residual
        ggml_tensor * resid = cur;
        ggml_tensor * x     = norm(cur, L.post_norm);
        ggml_tensor * ff    = ggml_swiglu_split(ctx, mm(ctx, L.ffn_gate, x), mm(ctx, L.ffn_up, x));
        cur                 = ggml_add(ctx, mm(ctx, L.ffn_down, ff), resid);
        inpL                = cur;
        if (il == debug_layer_) {
            out_logits = inpL;
            ggml_set_output(out_logits);
            ggml_build_forward_expand(gf, out_logits);
            return gf;
        }
    }

    if (o.last_only && n > 1) {
        inpL = ggml_view_2d(ctx, inpL, inpL->ne[0], 1, inpL->nb[1], inpL->nb[1] * (size_t) (n - 1));
    }
    ggml_tensor * cur = norm(inpL, out_norm_);
    out_logits        = mm(ctx, output_, cur);
    if (o.argmax) {
        out_logits = ggml_argmax(ctx, out_logits);
    }
    ggml_set_output(out_logits);
    ggml_build_forward_expand(gf, out_logits);
    return gf;
}

bool Qwen35::eval(const int32_t * tokens, int n, float * logits, std::string & err) {
    return eval(tokens, n, EvalOpts{}, logits, nullptr, err);
}

bool Qwen35::eval(const int32_t * tokens, int n, const EvalOpts & opts, float * logits, int32_t * ids,
                  std::string & err) {
    if (n <= 0 || n > opt_.n_ubatch) {
        err = "eval: batch size must be 1.." + std::to_string(opt_.n_ubatch);
        return false;
    }
    if (opts.record && n > opt_.max_record) {
        err = "eval: recorded batch larger than max_record (" + std::to_string(opt_.max_record) + ")";
        return false;
    }
    if (n_past_ + n > n_ctx_) {
        err = "eval: context full (" + std::to_string(n_ctx_) + " tokens)";
        return false;
    }
    const int n_kv = std::min(n_ctx_, (n_past_ + n + kKvPad - 1) / kKvPad * kKvPad);

    use_res_             = opts.residual && has_residual();
    ggml_init_params ip  = { graph_meta_.size(), graph_meta_.data(), true };
    ggml_context *   ctx = ggml_init(ip);
    ggml_tensor *    inp_tok = nullptr, * inp_pos = nullptr, * inp_mask = nullptr, * out = nullptr;
    ggml_cgraph *    gf      = build_graph(ctx, n, opts, inp_tok, inp_pos, inp_mask, out, n_kv);
    use_res_                 = false;

    ggml_backend_sched_reset(sched_);
    if (!ggml_backend_sched_alloc_graph(sched_, gf)) {
        ggml_free(ctx);
        err = "eval: cannot allocate the compute graph";
        return false;
    }
    {
        const int64_t            ne  = hp_.n_embd;
        const size_t             rs  = ggml_row_size(tok_embd_->type, ne);
        const ggml_type_traits * tt  = ggml_get_type_traits(tok_embd_->type);
        std::vector<float>       emb((size_t) (ne * n));
        for (int i = 0; i < n; i++) {
            if (tokens[i] < 0 || tokens[i] >= hp_.n_vocab) {
                ggml_free(ctx);
                err = "eval: token id out of range";
                return false;
            }
            const char * row = (const char *) tok_embd_->data + rs * (size_t) tokens[i];
            if (tok_embd_->type == GGML_TYPE_F32) std::memcpy(emb.data() + (size_t) i * ne, row, sizeof(float) * ne);
            else tt->to_float(row, emb.data() + (size_t) i * ne, ne);
        }
        ggml_backend_tensor_set(inp_tok, emb.data(), 0, emb.size() * sizeof(float));
    }
    // text-only M-RoPE positions: three identical sections, the fourth zero (llama.cpp llm_graph_input_pos)
    std::vector<int32_t> pos((size_t) n * 4, 0);
    for (int i = 0; i < n; i++) {
        pos[(size_t) i] = pos[(size_t) n + i] = pos[(size_t) 2 * n + i] = n_past_ + i;
    }
    // an input no node uses (e.g. positions in a debug graph that stops before the first attention layer) is not
    // allocated by the scheduler
    if (inp_pos->buffer) {
        ggml_backend_tensor_set(inp_pos, pos.data(), 0, pos.size() * sizeof(int32_t));
    }
    std::vector<ggml_fp16_t> mask((size_t) n_kv * n);
    const ggml_fp16_t        zero = ggml_fp32_to_fp16(0.0f), ninf = ggml_fp32_to_fp16(-INFINITY);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n_kv; j++) {
            mask[(size_t) i * n_kv + j] = j <= n_past_ + i ? zero : ninf;
        }
    }
    if (inp_mask->buffer) {
        ggml_backend_tensor_set(inp_mask, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
    }

    const ggml_status st = ggml_backend_sched_graph_compute(sched_, gf);
    if (st != GGML_STATUS_SUCCESS) {
        ggml_free(ctx);
        err = "eval: graph compute failed";
        return false;
    }
    if (opts.argmax) {
        if (ids) ggml_backend_tensor_get(out, ids, 0, ggml_nbytes(out));
    } else if (logits) {
        ggml_backend_tensor_get(out, logits, 0, ggml_nbytes(out));
    }
    ggml_free(ctx);
    recorded_n_ = opts.record ? n : 0;
    n_past_ += n;
    return true;
}

} // namespace e8::model
