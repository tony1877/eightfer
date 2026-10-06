#include "model/qwen35.h"

#include "kernels/q4k_small.h"

#include "ggml-alloc.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <atomic>
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
constexpr int    kPage      = 64;   // sparse long-context attention: tokens per key-summary page
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
    if (pbuf_) ggml_backend_buffer_free(pbuf_);
    if (pctx_) ggml_free(pctx_);
    if (hbuf_) ggml_backend_buffer_free(hbuf_);
    if (hctx_) ggml_free(hctx_);
    if (rctx_) ggml_free(rctx_);
    if (mbuf_) ggml_backend_buffer_free(mbuf_);
    if (mctx_) ggml_free(mctx_);
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
    // MTP block: loaded after the KV is sized, only when it fits next to the full KV (see below)
    const std::string mtp_p = "blk." + std::to_string(h.n_layer) + ".";
    static const char * const kMtpNames[] = { "attn_norm.weight", "post_attention_norm.weight", "attn_q.weight",
        "attn_k.weight", "attn_v.weight", "attn_output.weight", "attn_q_norm.weight", "attn_k_norm.weight",
        "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight", "nextn.eh_proj.weight", "nextn.enorm.weight",
        "nextn.hnorm.weight", "nextn.shared_head_norm.weight" };
    bool   want_mtp = opt.mtp && h.n_layer_nextn > 0 && out_gpu;
    double mtp_w    = 0;
    for (const char * nm : kMtpNames) {
        const ggml_tensor * m = file_.tensor(mtp_p + nm);
        if (!m) want_mtp = false;
        else mtp_w += (double) ggml_nbytes(m) + 512;
    }
    if (opt.mtp && h.n_layer_nextn > 0 && !want_mtp) fprintf(stderr, "warning: no usable MTP block, drafting without it\n");

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
    W_                     = n_ctx_;
    int64_t n_attn = 0;
    bool    all_gpu = gpu_ != nullptr;
    for (int64_t il = 0; il < h.n_layer; il++) {
        if (!h.is_recurrent(il)) n_attn++;
        all_gpu = all_gpu && layers_[(size_t) il].on_gpu;
    }
    if (all_gpu && n_attn > 0) {
        const int64_t row     = h.head_dim * h.n_head_kv;
        const double  tok     = 2.0 * (double) n_attn * (double) ggml_row_size(opt.kv_type, row);
        int64_t       want    = opt.gpu_kv;
        if (want < 0) {
            size_t fr = 0, tot = 0;
            ggml_backend_dev_memory(ggml_backend_get_device(gpu_), &fr, &tot);
            // recurrent state and compute buffers (prefill batch, CUDA pools), then the RAM-KV attention staging:
            // one layer's per-head K/V copies, their F16 conversion and the mask
            const double state   = 2.0 * (double) (h.n_layer - n_attn) *
                                 (double) (h.ssm_d_state * h.ssm_d_state * h.ssm_n_v + h.conv_channels() * 4) * 4.0;
            // MTP: weights, a second state snapshot, its KV ring and the hidden-state buffer
            const double mtp_cost = mtp_w + state / 2.0 + 2.0 * (double) opt.mtp_window * (double) ggml_row_size(opt.kv_type, h.head_dim * h.n_head_kv) +
                                    4.0 * (double) h.n_embd * (double) (opt.n_ubatch + 1);
            const double reserve = (std::getenv("E8_VRAM_RESERVE_GB") ? std::atof(std::getenv("E8_VRAM_RESERVE_GB")) * 1e9 : 0.15e9) + state;  // measured: 256K ctx, 7.9K window peaked at 15.5 of 16.3 GB
            const double stage   = (double) n_ctx_ * (2.0 * (double) h.n_head_kv * (double) ggml_row_size(opt.kv_type, h.head_dim) +
                                                    4.0 * (double) h.head_dim + 2.0 * (double) opt.n_ubatch);
            if (want_mtp && (double) fr - reserve - mtp_cost >= tok * (double) n_ctx_) {
                want    = n_ctx_;
                mtp_on_ = true;
            } else if ((double) fr - reserve >= tok * (double) n_ctx_) {
                want = n_ctx_;
            } else {
                const double summ = (double) n_attn * (double) h.n_head_kv * (double) (n_ctx_ / kPage) * (double) h.head_dim * 2.0;
                want = (int64_t) (((double) fr - reserve - stage - summ) / tok);
            }
        }
        want = std::min<int64_t>(n_ctx_, want / kKvPad * kKvPad);
        if (opt.gpu_kv >= 0 && want >= n_ctx_) mtp_on_ = want_mtp;
        if (want_mtp && !mtp_on_) {
            fprintf(stderr, "MTP drafting off: its %.2f GB of VRAM would come out of the KV window\n",
                    (mtp_w + 0.17e9) / 1e9);
        }
        if (want < n_ctx_) {
            if (want < 2048) {
                err = "not enough VRAM for a KV window (" + std::to_string(want) + " tokens; lower --ctx or use --kv q8_0)";
                return false;
            }
            // the draft far area comes out of the window
            Kd_ = std::getenv("E8_DRAFT_FAR") ? std::atoi(std::getenv("E8_DRAFT_FAR")) / kPage * kPage : 4096;
            if (want - Kd_ < 2048) Kd_ = 0;
            W_ = (int) (want - Kd_);
        }
    }
    if (mtp_on_) {
        ggml_init_params mp = { ggml_tensor_overhead() * 24, nullptr, true };
        mctx_               = ggml_init(mp);
        std::vector<std::pair<ggml_tensor *, const ggml_tensor *>> ml;
        auto mk = [&](const char * nm) {
            const ggml_tensor * m = file_.tensor(mtp_p + nm);
            ggml_tensor *       t = ggml_dup_tensor(mctx_, m);
            ggml_set_name(t, (mtp_p + nm).c_str());
            ml.emplace_back(t, m);
            return t;
        };
        Qwen35Layer & L = mtp_;
        L.on_gpu        = true;
        L.attn_norm = mk("attn_norm.weight");
        L.post_norm = mk("post_attention_norm.weight");
        L.wq        = mk("attn_q.weight");
        L.wk        = mk("attn_k.weight");
        L.wv        = mk("attn_v.weight");
        L.wo        = mk("attn_output.weight");
        L.q_norm    = mk("attn_q_norm.weight");
        L.k_norm    = mk("attn_k_norm.weight");
        L.ffn_gate  = mk("ffn_gate.weight");
        L.ffn_up    = mk("ffn_up.weight");
        L.ffn_down  = mk("ffn_down.weight");
        mtp_eh_     = mk("nextn.eh_proj.weight");
        mtp_enorm_  = mk("nextn.enorm.weight");
        mtp_hnorm_  = mk("nextn.hnorm.weight");
        mtp_norm_   = mk("nextn.shared_head_norm.weight");
        mbuf_       = ggml_backend_alloc_ctx_tensors_from_buft(mctx_, ggml_backend_get_default_buffer_type(gpu_));
        if (!mbuf_) {
            err = "not enough VRAM for the MTP block (use --mtp 0)";
            return false;
        }
        ggml_backend_buffer_set_usage(mbuf_, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        gpu_bytes_ += ggml_backend_buffer_get_size(mbuf_);
        if (!read_tensors(path, file_, ml, err)) return false;
    }
    ggml_init_params sp    = { ggml_tensor_overhead() * (size_t) (h.n_layer * 14 + 16), nullptr, true };
    sctx_[0]               = ggml_init(sp);
    sctx_[1]               = ggml_init(sp);
    k_cache_.assign((size_t) h.n_layer, nullptr);
    v_cache_.assign((size_t) h.n_layer, nullptr);
    conv_state_.assign((size_t) h.n_layer, nullptr);
    ssm_state_.assign((size_t) h.n_layer, nullptr);
    conv_bak_.assign((size_t) h.n_layer, nullptr);
    ssm_bak_.assign((size_t) h.n_layer, nullptr);
    conv_bak1_.assign((size_t) h.n_layer, nullptr);
    ssm_bak1_.assign((size_t) h.n_layer, nullptr);
    rec_qkv_.assign((size_t) h.n_layer, nullptr);
    rec_g_.assign((size_t) h.n_layer, nullptr);
    rec_beta_.assign((size_t) h.n_layer, nullptr);
    conv_ck_.assign((size_t) h.n_layer, nullptr);
    ssm_ck_.assign((size_t) h.n_layer, nullptr);
    const int64_t n_rec = std::max(1, opt.max_record);
    for (int64_t il = 0; il < h.n_layer; il++) {
        ggml_context * c = sctx_[layers_[(size_t) il].on_gpu ? 0 : 1];
        if (h.is_recurrent(il)) {
            conv_state_[(size_t) il] = ggml_new_tensor_1d(c, GGML_TYPE_F32, (h.ssm_d_conv - 1) * h.conv_channels());
            ssm_state_[(size_t) il]  = ggml_new_tensor_1d(c, GGML_TYPE_F32, h.ssm_d_state * h.ssm_d_state * h.ssm_n_v);
            conv_bak_[(size_t) il]   = ggml_dup_tensor(c, conv_state_[(size_t) il]);
            ssm_bak_[(size_t) il]    = ggml_dup_tensor(c, ssm_state_[(size_t) il]);
            // snapshot slot 1 (multi-token proposal checks while drafting): VRAM beside MTP, else RAM (it is copied once
            // per checked proposal, ~0.16 GB, and the KV window needs the VRAM more)
            ggml_context * cb = mtp_on_ ? c : sctx_[1];
            conv_bak1_[(size_t) il] = ggml_dup_tensor(cb, conv_state_[(size_t) il]);
            ssm_bak1_[(size_t) il]  = ggml_dup_tensor(cb, ssm_state_[(size_t) il]);
            // prompt-reuse checkpoint: copied once per request, so it lives in RAM
            conv_ck_[(size_t) il]    = ggml_dup_tensor(sctx_[1], conv_state_[(size_t) il]);
            ssm_ck_[(size_t) il]     = ggml_dup_tensor(sctx_[1], ssm_state_[(size_t) il]);
            rec_qkv_[(size_t) il]    = ggml_new_tensor_2d(c, GGML_TYPE_F32, h.conv_channels(), n_rec);
            rec_g_[(size_t) il]      = ggml_new_tensor_2d(c, GGML_TYPE_F32, h.ssm_n_v, n_rec);
            rec_beta_[(size_t) il]   = ggml_new_tensor_2d(c, GGML_TYPE_F32, h.ssm_n_v, n_rec);
        } else {
            k_cache_[(size_t) il] = ggml_new_tensor_2d(c, opt.kv_type, h.head_dim * h.n_head_kv, Kd_ + W_);
            v_cache_[(size_t) il] = ggml_new_tensor_2d(c, opt.kv_type, h.head_dim * h.n_head_kv, Kd_ + W_);
        }
    }
    if (mtp_on_) {
        Wm_       = std::max(kKvPad, std::min(n_ctx_, opt.mtp_window / kKvPad * kKvPad));
        mtp_k_    = ggml_new_tensor_2d(sctx_[0], opt.kv_type, h.head_dim * h.n_head_kv, Wm_);
        mtp_v_    = ggml_new_tensor_2d(sctx_[0], opt.kv_type, h.head_dim * h.n_head_kv, Wm_);
        mtp_hid_  = ggml_new_tensor_2d(sctx_[0], GGML_TYPE_F32, h.n_embd, opt.n_ubatch);
        mtp_chain_ = ggml_new_tensor_2d(sctx_[0], GGML_TYPE_F32, h.n_embd, 1);
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
    if (W_ < n_ctx_) {
        ggml_init_params hp = { ggml_tensor_overhead() * (size_t) (h.n_layer * h.n_head_kv * 2 + 8), nullptr, true };
        hctx_               = ggml_init(hp);
        hk_.assign((size_t) (h.n_layer * h.n_head_kv), nullptr);
        hv_.assign((size_t) (h.n_layer * h.n_head_kv), nullptr);
        for (int64_t il = 0; il < h.n_layer; il++) {
            if (h.is_recurrent(il)) continue;
            for (int64_t j = 0; j < h.n_head_kv; j++) {
                hk_[(size_t) (il * h.n_head_kv + j)] = ggml_new_tensor_2d(hctx_, opt.kv_type, h.head_dim, n_ctx_);
                hv_[(size_t) (il * h.n_head_kv + j)] = ggml_new_tensor_2d(hctx_, opt.kv_type, h.head_dim, n_ctx_);
            }
        }
        if (std::getenv("E8_KV_PINNED") && std::atoi(std::getenv("E8_KV_PINNED")) != 0) {
            hbuf_ = ggml_backend_alloc_ctx_tensors_from_buft(hctx_, ggml_backend_dev_host_buffer_type(ggml_backend_get_device(gpu_)));
            if (!hbuf_) fprintf(stderr, "warning: pinned allocation for the KV failed, using pageable RAM\n");
        }
        if (!hbuf_) hbuf_ = ggml_backend_alloc_ctx_tensors_from_buft(hctx_, ggml_backend_get_default_buffer_type(cpu_));
        if (!hbuf_) {
            err = "not enough RAM for the KV cache (lower --ctx or use --kv q8_0)";
            return false;
        }
        ggml_backend_buffer_clear(hbuf_, 0);  // masked rows must still be finite for the GPU kernels
        ggml_init_params pp = { ggml_tensor_overhead() * (size_t) (h.n_layer * h.n_head_kv * 2 + 8), nullptr, true };
        pctx_               = ggml_init(pp);
        pmid_.assign(hk_.size(), nullptr);
        for (size_t i = 0; i < hk_.size(); i++) {
            if (!hk_[i]) continue;
            pmid_[i] = ggml_new_tensor_2d(pctx_, GGML_TYPE_F16, h.head_dim, n_ctx_ / kPage);
        }
        pbuf_ = ggml_backend_alloc_ctx_tensors_from_buft(pctx_, ggml_backend_get_default_buffer_type(gpu_));
        if (!pbuf_) {
            err = "not enough VRAM for the key page summaries";
            return false;
        }
        if (const char * e = std::getenv("E8_HOST_ATTN_GPU_MIN")) host_gpu_min_ = std::atoi(e);
        fprintf(stderr, "KV: %d tokens in RAM (%.2f GB), window of %d in VRAM\n", n_ctx_,
                (double) ggml_backend_buffer_get_size(hbuf_) / 1e9, W_);
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
    saved_n_past_ = saved_n_past1_ = 0;
    recorded_n_   = 0;
    hid_row_      = -1;
    mtp_slot_pos_.assign((size_t) Wm_, -1);
    slot_pos_.assign((size_t) W_, -1);
    host_valid_ = exact_upto_ = 0;
    sum_upto_ = 0;
    far_rows_ = 0;
}

void Qwen35::update_summaries(int upto) {
    // element-wise min/max of each new page's keys (from the RAM KV), computed in parallel, uploaded in order
    const auto & h = hp_;
    if (upto <= sum_upto_) return;
    const int64_t hd = h.head_dim, np = upto - sum_upto_;
    const auto *  tt = ggml_get_type_traits(opt_.kv_type);
    std::vector<size_t> ids;
    for (size_t i = 0; i < hk_.size(); i++) if (hk_[i]) ids.push_back(i);
    std::vector<std::vector<ggml_fp16_t>> mid(ids.size());
    std::atomic<size_t> next{ 0 };
    auto work = [&]() {
        std::vector<float> row((size_t) hd), lo((size_t) hd), hi((size_t) hd);
        for (size_t t; (t = next++) < ids.size();) {
            const ggml_tensor * hk = hk_[ids[t]];
            mid[t].resize((size_t) (np * hd));
            for (int64_t p = 0; p < np; p++) {
                std::fill(lo.begin(), lo.end(), INFINITY);
                std::fill(hi.begin(), hi.end(), -INFINITY);
                for (int r = 0; r < kPage; r++) {
                    const char * src = (const char *) hk->data + hk->nb[1] * (size_t) ((sum_upto_ + p) * kPage + r);
                    if (hk->type == GGML_TYPE_F32) std::memcpy(row.data(), src, sizeof(float) * hd);
                    else tt->to_float(src, row.data(), hd);
                    for (int64_t d = 0; d < hd; d++) {
                        lo[(size_t) d] = std::min(lo[(size_t) d], row[(size_t) d]);
                        hi[(size_t) d] = std::max(hi[(size_t) d], row[(size_t) d]);
                    }
                }
                for (int64_t d = 0; d < hd; d++) {
                    mid[t][(size_t) (p * hd + d)] = ggml_fp32_to_fp16(0.5f * (lo[(size_t) d] + hi[(size_t) d]));
                }
            }
        }
    };
    std::vector<std::thread> th;
    const int nt = std::max(1, std::min<int>((int) ids.size(), (int) std::thread::hardware_concurrency()));
    for (int i = 0; i < nt; i++) th.emplace_back(work);
    for (auto & x : th) x.join();
    const size_t off = sizeof(ggml_fp16_t) * (size_t) (sum_upto_ * hd);
    for (size_t t = 0; t < ids.size(); t++) {
        ggml_backend_tensor_set(pmid_[ids[t]], mid[t].data(), off, mid[t].size() * sizeof(ggml_fp16_t));
    }
    sum_upto_ = upto;
}

void Qwen35::fill_draft_far() {
    // after a sparse verify: copy its best-scoring pages (per KV head) from RAM into each ring's draft far area
    const auto &  h   = hp_;
    const int64_t nkv = h.n_head_kv;
    const size_t  hrs = ggml_row_size(opt_.kv_type, h.head_dim);
    const int     np  = std::min(Kd_ / kPage, sp_kp_);
    if (np <= 0 || sel_nodes_.empty()) return;
    std::vector<int32_t> sel((size_t) sp_kp_);
    std::vector<uint8_t> buf;
    for (int64_t il = 0; il < h.n_layer; il++) {
        if (h.is_recurrent(il)) continue;
        for (int kv = 0; kv < 2; kv++) {
            ggml_tensor * rc = (kv ? v_cache_ : k_cache_)[(size_t) il];
            buf.assign(rc->nb[1] * (size_t) Kd_, 0);
            for (int64_t j = 0; j < nkv; j++) {
                const size_t hi = (size_t) (il * nkv + j);
                ggml_backend_tensor_get(sel_nodes_[hi], sel.data(), 0, sel.size() * sizeof(int32_t));
                const ggml_tensor * ht = (kv ? hv_ : hk_)[hi];
                for (int p = 0; p < np; p++) {
                    for (int r = 0; r < kPage; r++) {
                        std::memcpy(buf.data() + rc->nb[1] * (size_t) (p * kPage + r) + hrs * (size_t) j,
                                    (const uint8_t *) ht->data + ht->nb[1] * (size_t) (sel[(size_t) p] * kPage + r), hrs);
                    }
                }
            }
            ggml_backend_tensor_set(rc, buf.data(), 0, buf.size());
        }
    }
    far_rows_ = np * kPage;
    far_ws_   = sp_ws_;
}

namespace {
// rows of a sparse attention's K or V: the selected pages (src[0] = page ids), then the recent window, then zeros
void gather_op(ggml_tensor * dst, int ith, int nth, void * ud) {
    const auto *  g    = (const Qwen35GatherInfo *) ud;
    const int64_t rows = dst->ne[1];
    const size_t  rs   = dst->nb[1];
    const int64_t per  = (rows + nth - 1) / nth, r0 = per * ith, r1 = std::min(rows, r0 + per);
    const int32_t * ids = g->kp_rows > 0 ? (const int32_t *) dst->src[0]->data : nullptr;
    for (int64_t r = r0; r < r1; r++) {
        int64_t src = -1;
        if (r < g->kp_rows) src = (int64_t) ids[r / g->page] * g->page + r % g->page;
        else if (r - g->kp_rows < g->nwin) src = g->win_start + (r - g->kp_rows);
        uint8_t * d = (uint8_t *) dst->data + rs * (size_t) r;
        if (src >= 0) std::memcpy(d, (const uint8_t *) g->host->data + g->host->nb[1] * (size_t) src, rs);
        else std::memset(d, 0, rs);
    }
}
} // namespace

bool Qwen35::flush_ring(int from, int to, std::string & err) {
    // copy positions [from, to) from the VRAM ring to the RAM KV, per KV head
    const auto &         h   = hp_;
    const size_t         hrs = ggml_row_size(opt_.kv_type, h.head_dim);
    std::vector<uint8_t> buf;
    for (int p = from; p < to;) {
        const int s = p % W_;
        if (slot_pos_[(size_t) s] != p) {
            err = "KV for position " + std::to_string(p) + " is no longer in VRAM";
            return false;
        }
        int e = p + 1;
        while (e < to && (e % W_) == s + (e - p) && slot_pos_[(size_t) (e % W_)] == e) e++;
        const int cnt = e - p;
        for (int64_t il = 0; il < h.n_layer; il++) {
            if (h.is_recurrent(il)) continue;
            for (int kv = 0; kv < 2; kv++) {
                ggml_tensor * rc = (kv ? v_cache_ : k_cache_)[(size_t) il];
                buf.resize(rc->nb[1] * (size_t) cnt);
                ggml_backend_tensor_get(rc, buf.data(), rc->nb[1] * (size_t) (Kd_ + s), buf.size());
                for (int64_t j = 0; j < h.n_head_kv; j++) {
                    ggml_tensor * ht = (kv ? hv_ : hk_)[(size_t) (il * h.n_head_kv + j)];
                    for (int r = 0; r < cnt; r++) {
                        std::memcpy((uint8_t *) ht->data + ht->nb[1] * (size_t) (p + r),
                                    buf.data() + rc->nb[1] * (size_t) r + hrs * (size_t) j, hrs);
                    }
                }
            }
        }
        p = e;
    }
    host_valid_ = std::max(host_valid_, to);
    return true;
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
        if (!base || std::string(ggml_get_name(m)).rfind("blk." + std::to_string(hp_.n_layer) + ".", 0) == 0) {
            continue;  // MTP tensors: drafting uses the base alone
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
    // E8_KV_PINNED=1 gives the pinned budget to a long-context KV instead (faster decode, slower prefill)
    const bool kv_pin = std::getenv("E8_KV_PINNED") && std::atoi(std::getenv("E8_KV_PINNED")) != 0;
    if (gpu_ && !kv_pin) {
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

void Qwen35::save_state(int slot) {
    auto & cb = slot ? conv_bak1_ : conv_bak_;
    auto & sb = slot ? ssm_bak1_ : ssm_bak_;
    for (size_t il = 0; il < conv_state_.size(); il++) {
        if (conv_state_[il] && cb[il]) {
            ggml_backend_tensor_copy(conv_state_[il], cb[il]);
            ggml_backend_tensor_copy(ssm_state_[il], sb[il]);
        }
    }
    (slot ? saved_n_past1_ : saved_n_past_) = n_past_;
}

void Qwen35::checkpoint_save() {
    for (size_t il = 0; il < conv_state_.size(); il++) {
        if (conv_state_[il]) {
            ggml_backend_tensor_copy(conv_state_[il], conv_ck_[il]);
            ggml_backend_tensor_copy(ssm_state_[il], ssm_ck_[il]);
        }
    }
    ck_n_past_ = n_past_;
}

void Qwen35::checkpoint_restore() {
    for (size_t il = 0; il < conv_state_.size(); il++) {
        if (conv_state_[il]) {
            ggml_backend_tensor_copy(conv_ck_[il], conv_state_[il]);
            ggml_backend_tensor_copy(ssm_ck_[il], ssm_state_[il]);
        }
    }
    n_past_     = ck_n_past_;
    recorded_n_ = 0;
    hid_row_    = -1;
}

void Qwen35::restore_state(int slot) {
    auto & cb = slot ? conv_bak1_ : conv_bak_;
    auto & sb = slot ? ssm_bak1_ : ssm_bak_;
    for (size_t il = 0; il < conv_state_.size(); il++) {
        if (conv_state_[il] && cb[il]) {
            ggml_backend_tensor_copy(cb[il], conv_state_[il]);
            ggml_backend_tensor_copy(sb[il], ssm_state_[il]);
        }
    }
    n_past_  = slot ? saved_n_past1_ : saved_n_past_;
    hid_row_ = -1;
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

bool Qwen35::rollback(int keep, std::string & err, int slot) {
    if (recorded_n_ <= 0 || keep < 0 || keep > recorded_n_) {
        err = "rollback: no recorded eval or keep out of range";
        return false;
    }
    const int n = recorded_n_;
    recorded_n_ = 0;
    if (keep == n) {
        return true;
    }
    restore_state(slot);
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
    n_past_  = (slot ? saved_n_past1_ : saved_n_past_) + keep;
    hid_row_ = keep - 1;
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
    inp_kvidx_ = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n);
    ggml_set_input(inp_kvidx_);
    inp_hidx_ = nullptr;
    attn_nodes_.clear();
    inp_sbias_ = inp_smask_ = nullptr;
    if (attn_mode_ == 1) {
        inp_hidx_ = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n);
        ggml_set_input(inp_hidx_);
        if (sparse_) {
            inp_smask_ = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, sp_nsel_, n);
            ggml_set_input(inp_smask_);
            if (sp_kp_ > 0) {
                inp_sbias_ = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, sp_nfar_, 1);
                ggml_set_input(inp_sbias_);
            }
            gi_.clear();
            gi_.reserve((size_t) (h.n_layer * h.n_head_kv * 2));  // stable addresses: the gather ops point at them
        }
    }

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
            ggml_build_forward_expand(gf, ggml_set_rows(ctx, kc, ggml_reshape_2d(ctx, Kc, row, n), inp_kvidx_));
            ggml_build_forward_expand(gf, ggml_set_rows(ctx, vc, ggml_reshape_2d(ctx, Vc, row, n), inp_kvidx_));
            ggml_tensor * q = ggml_permute(ctx, Qc, 0, 2, 1, 3);  // [hd, n, n_head]
            ggml_tensor * a = nullptr;
            if (attn_mode_ == 1 && sparse_) {
                const int64_t g  = h.n_head / h.n_head_kv, nkv = h.n_head_kv;
                const int64_t kr = (int64_t) sp_kp_ * kPage;
                std::vector<ggml_tensor *> wk((size_t) nkv), wv((size_t) nkv), sel((size_t) nkv, nullptr), qs((size_t) nkv);
                for (int64_t j = 0; j < nkv; j++) {  // write this batch to the RAM KV
                    const size_t  hi = (size_t) (il * nkv + j);
                    ggml_tensor * kj = ggml_cont(ctx, ggml_view_2d(ctx, Kc, hd, n, Kc->nb[2], Kc->nb[1] * (size_t) j));
                    ggml_tensor * vj = ggml_cont(ctx, ggml_view_2d(ctx, Vc, hd, n, Vc->nb[2], Vc->nb[1] * (size_t) j));
                    wk[(size_t) j]   = ggml_set_rows(ctx, hk_[hi], kj, inp_hidx_);
                    wv[(size_t) j]   = ggml_set_rows(ctx, hv_[hi], vj, inp_hidx_);
                    ggml_build_forward_expand(gf, wk[(size_t) j]);
                    ggml_build_forward_expand(gf, wv[(size_t) j]);
                    qs[(size_t) j] = ggml_view_3d(ctx, q, hd, n, g, q->nb[1], q->nb[2], q->nb[2] * (size_t) (j * g));
                }
                if (sp_kp_ > 0) {  // page selection on the GPU: sum over the batch's queries of the q.k upper bound
                    for (int64_t j = 0; j < nkv; j++) {
                        const size_t  hi = (size_t) (il * nkv + j);
                        ggml_tensor * q2 = ggml_reshape_2d(ctx, ggml_cont(ctx, qs[(size_t) j]), hd, n * g);
                        ggml_tensor * md = ggml_view_2d(ctx, pmid_[hi], hd, sp_nfar_, pmid_[hi]->nb[1], 0);
                        ggml_tensor * s  = ggml_mul_mat(ctx, md, q2);  // q . page midpoint: [nfar, n*g]
                        // per query: a distribution over pages (each query counts equally, whatever its scale), then
                        // summed over the batch's queries. (Midpoints beat the q.k upper bound from min/max: on text
                        // repeated 48K tokens back, PPL 1.0013 vs 1.069; exact 1.0004.)
                        s = ggml_soft_max_ext(ctx, s, nullptr, 1.0f / sqrtf((float) hd), 0.0f);
                        s = ggml_sum_rows(ctx, ggml_cont(ctx, ggml_transpose(ctx, s)));                            // [1, nfar]
                        s = ggml_add(ctx, ggml_reshape_2d(ctx, s, sp_nfar_, 1), inp_sbias_);
                        sel[(size_t) j] = ggml_cont(ctx, ggml_argsort_top_k(ctx, s, sp_kp_));  // best first
                        ggml_set_output(sel[(size_t) j]);
                        ggml_build_forward_expand(gf, sel[(size_t) j]);
                        if (sel_nodes_.size() < hk_.size()) sel_nodes_.resize(hk_.size(), nullptr);
                        sel_nodes_[hi] = sel[(size_t) j];
                    }
                }
                std::vector<ggml_tensor *> gk((size_t) nkv), gv((size_t) nkv);
                for (int64_t j = 0; j < nkv; j++) {  // gather the selected pages and the window from RAM
                    const size_t hi = (size_t) (il * nkv + j);
                    for (int kv = 0; kv < 2; kv++) {
                        gi_.push_back({ kv ? hv_[hi] : hk_[hi], (int) kr, kPage, sp_ws_, sp_nwin_ });
                        ggml_tensor * dep     = kv ? wv[(size_t) j] : wk[(size_t) j];
                        ggml_tensor * args[2] = { sel[(size_t) j] ? sel[(size_t) j] : dep, dep };
                        ggml_tensor * t       = ggml_custom_4d(ctx, opt_.kv_type, hd, sp_nsel_, 1, 1, args, 2, gather_op,
                                                               GGML_N_TASKS_MAX, &gi_.back());
                        ggml_build_forward_expand(gf, t);
                        (kv ? gv : gk)[(size_t) j] = t;
                    }
                }
                for (int64_t j = 0; j < nkv; j++) {
                    ggml_tensor * aj = ggml_flash_attn_ext(ctx, qs[(size_t) j], gk[(size_t) j], gv[(size_t) j], inp_smask_,
                                                           1.0f / sqrtf((float) hd), 0.0f, 0.0f);
                    ggml_prec_set_acc(aj, GGML_PREC_F32);
                    attn_nodes_.push_back(aj);
                    a = a ? ggml_concat(ctx, a, aj, 1) : aj;  // [hd, heads, n]
                }
            } else if (attn_mode_ == 1) {
                // full KV in RAM, one attention per KV head (its 'g' query heads), so a GPU run copies one head
                const int64_t g = h.n_head / h.n_head_kv;
                for (int64_t j = 0; j < h.n_head_kv; j++) {
                    ggml_tensor * hk = hk_[(size_t) (il * h.n_head_kv + j)];
                    ggml_tensor * hv = hv_[(size_t) (il * h.n_head_kv + j)];
                    ggml_tensor * kj = ggml_cont(ctx, ggml_view_2d(ctx, Kc, hd, n, Kc->nb[2], Kc->nb[1] * (size_t) j));
                    ggml_tensor * vj = ggml_cont(ctx, ggml_view_2d(ctx, Vc, hd, n, Vc->nb[2], Vc->nb[1] * (size_t) j));
                    ggml_build_forward_expand(gf, ggml_set_rows(ctx, hk, kj, inp_hidx_));
                    ggml_build_forward_expand(gf, ggml_set_rows(ctx, hv, vj, inp_hidx_));
                    ggml_tensor * k  = ggml_view_3d(ctx, hk, hd, n_kv, 1, hk->nb[1], hk->nb[1] * (size_t) n_kv, 0);
                    ggml_tensor * v  = ggml_view_3d(ctx, hv, hd, n_kv, 1, hv->nb[1], hv->nb[1] * (size_t) n_kv, 0);
                    ggml_tensor * qj = ggml_view_3d(ctx, q, hd, n, g, q->nb[1], q->nb[2], q->nb[2] * (size_t) (j * g));
                    ggml_tensor * aj = ggml_flash_attn_ext(ctx, qj, k, v, inp_mask, 1.0f / sqrtf((float) hd), 0.0f, 0.0f);
                    ggml_prec_set_acc(aj, GGML_PREC_F32);
                    attn_nodes_.push_back(aj);
                    a = a ? ggml_concat(ctx, a, aj, 1) : aj;  // [hd, heads, n]
                }
            } else {
                ggml_tensor * K = ggml_view_3d(ctx, kc, hd, h.n_head_kv, n_kv, ggml_row_size(kc->type, hd), kc->nb[1], kc->nb[1] * (size_t) ring_off_);
                ggml_tensor * V = ggml_view_3d(ctx, vc, hd, h.n_head_kv, n_kv, ggml_row_size(vc->type, hd), vc->nb[1], vc->nb[1] * (size_t) ring_off_);
                ggml_tensor * k = ggml_permute(ctx, K, 0, 2, 1, 3);
                ggml_tensor * v = ggml_permute(ctx, V, 0, 2, 1, 3);
                a = ggml_flash_attn_ext(ctx, q, k, v, inp_mask, 1.0f / sqrtf((float) hd), 0.0f, 0.0f);
                ggml_prec_set_acc(a, GGML_PREC_F32);
            }
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

    ggml_tensor * cur = nullptr;
    inp_mpos_ = inp_mkvidx_ = nullptr;
    if (mtp_on_) {
        // all rows: the MTP drafts read them (vLLM's Qwen3-Next MTP takes the target's normed hidden states)
        ggml_tensor * hall = norm(inpL, out_norm_);
        ggml_build_forward_expand(gf, ggml_cpy(ctx, hall, ggml_view_2d(ctx, mtp_hid_, h.n_embd, n, mtp_hid_->nb[1], 0)));
        if (n > 1) {  // MTP K/V of positions [0, n-1): their next token is this batch's next token
            inp_mpos_ = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int64_t) (n - 1) * 4);
            ggml_set_input(inp_mpos_);
            inp_mkvidx_ = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n - 1);
            ggml_set_input(inp_mkvidx_);
            ggml_tensor * e  = ggml_view_2d(ctx, inp_tok, h.n_embd, n - 1, inp_tok->nb[1], inp_tok->nb[1]);
            ggml_tensor * hh = ggml_view_2d(ctx, hall, h.n_embd, n - 1, hall->nb[1], 0);
            mtp_layer(ctx, gf, mtp_input(ctx, e, hh), n - 1, inp_mpos_, inp_mkvidx_, nullptr);
        }
        cur = o.last_only && n > 1 ? ggml_view_2d(ctx, hall, hall->ne[0], 1, hall->nb[1], hall->nb[1] * (size_t) (n - 1)) : hall;
    } else {
        if (o.last_only && n > 1) {
            inpL = ggml_view_2d(ctx, inpL, inpL->ne[0], 1, inpL->nb[1], inpL->nb[1] * (size_t) (n - 1));
        }
        cur = norm(inpL, out_norm_);
    }
    out_logits = mm(ctx, output_, cur);
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

void Qwen35::embed(const int32_t * tokens, int n, std::vector<float> & out) const {
    const int64_t            ne = hp_.n_embd;
    const size_t             rs = ggml_row_size(tok_embd_->type, ne);
    const ggml_type_traits * tt = ggml_get_type_traits(tok_embd_->type);
    out.resize((size_t) (ne * n));
    for (int i = 0; i < n; i++) {
        const char * row = (const char *) tok_embd_->data + rs * (size_t) tokens[i];
        if (tok_embd_->type == GGML_TYPE_F32) std::memcpy(out.data() + (size_t) i * ne, row, sizeof(float) * ne);
        else tt->to_float(row, out.data() + (size_t) i * ne, ne);
    }
}

ggml_tensor * Qwen35::mtp_input(ggml_context * ctx, ggml_tensor * emb, ggml_tensor * hid) {
    const float eps = hp_.rms_eps;
    ggml_tensor * e = ggml_mul(ctx, ggml_rms_norm(ctx, emb, eps), mtp_enorm_);
    ggml_tensor * x = ggml_mul(ctx, ggml_rms_norm(ctx, hid, eps), mtp_hnorm_);
    return ggml_mul_mat(ctx, mtp_eh_, ggml_concat(ctx, e, x, 0));  // fc([embedding | hidden])
}

ggml_tensor * Qwen35::mtp_layer(ggml_context * ctx, ggml_cgraph * gf, ggml_tensor * x, int n, ggml_tensor * pos,
                                ggml_tensor * kvidx, ggml_tensor * mask) {
    // the trunk's gated full attention + SwiGLU FFN, base weights only, KV in the MTP ring
    const auto &        h   = hp_;
    const Qwen35Layer & L   = mtp_;
    const float         eps = h.rms_eps;
    const int64_t       hd  = h.head_dim;
    int sections[4] = { h.rope_sections[0], h.rope_sections[1], h.rope_sections[2], h.rope_sections[3] };
    auto norm = [&](ggml_tensor * t, ggml_tensor * w) { return ggml_mul(ctx, ggml_rms_norm(ctx, t, eps), w); };
    ggml_tensor * cur = norm(x, L.attn_norm);
    ggml_tensor * Kc  = norm(ggml_reshape_3d(ctx, ggml_mul_mat(ctx, L.wk, cur), hd, h.n_head_kv, n), L.k_norm);
    ggml_tensor * Vc  = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, L.wv, cur), hd, h.n_head_kv, n);
    Kc = ggml_rope_multi(ctx, Kc, pos, nullptr, (int) h.n_rot, sections, GGML_ROPE_TYPE_IMROPE, (int) h.n_ctx_train,
                         h.rope_freq_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
    const int64_t row = hd * h.n_head_kv;
    ggml_build_forward_expand(gf, ggml_set_rows(ctx, mtp_k_, ggml_reshape_2d(ctx, Kc, row, n), kvidx));
    ggml_build_forward_expand(gf, ggml_set_rows(ctx, mtp_v_, ggml_reshape_2d(ctx, Vc, row, n), kvidx));
    if (!mask) return nullptr;
    ggml_tensor * qg   = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, L.wq, cur), hd * 2, h.n_head, n);
    const size_t  es   = ggml_element_size(qg);
    ggml_tensor * Qc   = ggml_view_3d(ctx, qg, hd, h.n_head, n, es * hd * 2, es * hd * 2 * h.n_head, 0);
    ggml_tensor * gate = ggml_cont_2d(ctx, ggml_view_3d(ctx, qg, hd, h.n_head, n, es * hd * 2, es * hd * 2 * h.n_head, es * hd),
                                      hd * h.n_head, n);
    Qc = ggml_rope_multi(ctx, norm(Qc, L.q_norm), pos, nullptr, (int) h.n_rot, sections, GGML_ROPE_TYPE_IMROPE,
                         (int) h.n_ctx_train, h.rope_freq_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
    ggml_tensor * K = ggml_view_3d(ctx, mtp_k_, hd, h.n_head_kv, Wm_, ggml_row_size(mtp_k_->type, hd), mtp_k_->nb[1], 0);
    ggml_tensor * V = ggml_view_3d(ctx, mtp_v_, hd, h.n_head_kv, Wm_, ggml_row_size(mtp_v_->type, hd), mtp_v_->nb[1], 0);
    ggml_tensor * a = ggml_flash_attn_ext(ctx, ggml_permute(ctx, Qc, 0, 2, 1, 3), ggml_permute(ctx, K, 0, 2, 1, 3),
                                          ggml_permute(ctx, V, 0, 2, 1, 3), mask, 1.0f / sqrtf((float) hd), 0.0f, 0.0f);
    ggml_prec_set_acc(a, GGML_PREC_F32);
    a = ggml_mul(ctx, ggml_reshape_2d(ctx, a, hd * h.n_head, n), ggml_sigmoid(ctx, gate));
    x = ggml_add(ctx, x, ggml_mul_mat(ctx, L.wo, a));
    ggml_tensor * y  = norm(x, L.post_norm);
    ggml_tensor * ff = ggml_swiglu_split(ctx, ggml_mul_mat(ctx, L.ffn_gate, y), ggml_mul_mat(ctx, L.ffn_up, y));
    return ggml_add(ctx, x, ggml_mul_mat(ctx, L.ffn_down, ff));
}

bool Qwen35::mtp_step(int32_t tok, int pos, int hid_row, float * logits, int32_t * id, std::string & err) {
    const auto & h = hp_;
    if (!mtp_on_ || tok < 0 || tok >= h.n_vocab || pos < 0 || (hid_row >= 0 && hid_row >= opt_.n_ubatch)) {
        err = "mtp_step: no MTP block or bad arguments";
        return false;
    }
    ggml_init_params ip  = { graph_meta_.size(), graph_meta_.data(), true };
    ggml_context *   ctx = ggml_init(ip);
    ggml_cgraph *    gf  = ggml_new_graph_custom(ctx, kGraphSize, false);
    ggml_tensor * inp_e  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, h.n_embd, 1);
    ggml_set_input(inp_e);
    ggml_tensor * inp_p  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4);
    ggml_set_input(inp_p);
    ggml_tensor * inp_kv = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 1);
    ggml_set_input(inp_kv);
    ggml_tensor * inp_m  = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, Wm_, 1, 1, 1);
    ggml_set_input(inp_m);
    ggml_tensor * hsrc = hid_row >= 0 ? ggml_view_2d(ctx, mtp_hid_, h.n_embd, 1, mtp_hid_->nb[1], mtp_hid_->nb[1] * (size_t) hid_row)
                                      : mtp_chain_;
    ggml_tensor * y  = mtp_layer(ctx, gf, mtp_input(ctx, inp_e, hsrc), 1, inp_p, inp_kv, inp_m);
    ggml_tensor * hn = ggml_mul(ctx, ggml_rms_norm(ctx, y, h.rms_eps), mtp_norm_);
    ggml_build_forward_expand(gf, ggml_cpy(ctx, hn, mtp_chain_));
    ggml_tensor * out = ggml_mul_mat(ctx, output_, hn);
    if (!logits) out = ggml_argmax(ctx, out);
    ggml_set_output(out);
    ggml_build_forward_expand(gf, out);

    ggml_backend_sched_reset(sched_);
    if (!ggml_backend_sched_alloc_graph(sched_, gf)) {
        ggml_free(ctx);
        err = "mtp_step: cannot allocate the compute graph";
        return false;
    }
    std::vector<float> emb;
    embed(&tok, 1, emb);
    ggml_backend_tensor_set(inp_e, emb.data(), 0, emb.size() * sizeof(float));
    const int32_t p4[4] = { pos, pos, pos, 0 };
    ggml_backend_tensor_set(inp_p, p4, 0, sizeof p4);
    const int64_t slot = pos % Wm_;
    mtp_slot_pos_[(size_t) slot] = pos;
    ggml_backend_tensor_set(inp_kv, &slot, 0, sizeof slot);
    std::vector<ggml_fp16_t> mask((size_t) Wm_);
    const ggml_fp16_t        zero = ggml_fp32_to_fp16(0.0f), ninf = ggml_fp32_to_fp16(-INFINITY);
    for (int j = 0; j < Wm_; j++) {
        const int sp = mtp_slot_pos_[(size_t) j];
        mask[(size_t) j] = sp >= 0 && sp <= pos && sp > pos - Wm_ ? zero : ninf;
    }
    ggml_backend_tensor_set(inp_m, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
    const bool ok = ggml_backend_sched_graph_compute(sched_, gf) == GGML_STATUS_SUCCESS;
    if (ok) {
        if (logits) ggml_backend_tensor_get(out, logits, 0, ggml_nbytes(out));
        else if (id) ggml_backend_tensor_get(out, id, 0, sizeof(int32_t));
    }
    ggml_free(ctx);
    if (!ok) err = "mtp_step: graph compute failed";
    return ok;
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
    // where this eval's attention reads from (see the split-KV notes in qwen35.h)
    const bool split = W_ < n_ctx_;
    host_valid_      = std::min(host_valid_, n_past_);
    exact_upto_      = std::min(exact_upto_, n_past_);
    bool ring_exact  = n_past_ + n <= W_;
    for (int p = 0; ring_exact && p < n_past_; p++) ring_exact = slot_pos_[(size_t) p] == p;
    const bool use_host = split && !ring_exact && !opts.window_ok;
    if (split && !ring_exact && host_valid_ < exact_upto_ && !flush_ring(host_valid_, exact_upto_, err)) {
        return false;
    }
    if (use_host && host_valid_ < n_past_) {
        err = "eval: KV for positions " + std::to_string(host_valid_) + ".." + std::to_string(n_past_) + " was lost";
        return false;
    }
    const bool window = split && !ring_exact && !use_host;  // a draft attending only the ring
    if (far_ws_ > n_past_) far_rows_ = 0;  // rolled back before the pages it was built from
    ring_off_         = window ? 0 : Kd_;
    const int  n_kv   = window ? Kd_ + W_ : std::min(use_host ? n_ctx_ : W_, (n_past_ + n + kKvPad - 1) / kKvPad * kKvPad);
    for (int i = 0; i < n; i++) slot_pos_[(size_t) ((n_past_ + i) % W_)] = n_past_ + i;
    attn_mode_ = use_host ? 1 : 0;
    sum_upto_  = std::min(sum_upto_, std::min(n_past_, host_valid_) / kPage);
    static const bool sparse_on = !(std::getenv("E8_SPARSE") && std::atoi(std::getenv("E8_SPARSE")) == 0);
    sparse_ = use_host && n <= 32 && sparse_on && !pmid_.empty();
    sel_nodes_.clear();
    if (sparse_) {
        static const int win   = std::getenv("E8_SPARSE_WINDOW") ? std::atoi(std::getenv("E8_SPARSE_WINDOW")) : 4096;
        static const int pages = std::getenv("E8_SPARSE_PAGES") ? std::atoi(std::getenv("E8_SPARSE_PAGES")) : 128;
        sp_ws_   = std::max(0, (n_past_ + n - std::max(win, n)) / kPage * kPage);
        sp_nfar_ = sp_ws_ / kPage;
        sp_kp_   = std::min(pages, sp_nfar_);
        sp_nwin_ = n_past_ + n - sp_ws_;
        sp_nsel_ = (sp_kp_ * kPage + sp_nwin_ + kKvPad - 1) / kKvPad * kKvPad;
        update_summaries(sp_nfar_);
    }

    use_res_             = opts.residual && has_residual();
    ggml_init_params ip  = { graph_meta_.size(), graph_meta_.data(), true };
    ggml_context *   ctx = ggml_init(ip);
    ggml_tensor *    inp_tok = nullptr, * inp_pos = nullptr, * inp_mask = nullptr, * out = nullptr;
    ggml_cgraph *    gf      = build_graph(ctx, n, opts, inp_tok, inp_pos, inp_mask, out, n_kv);
    use_res_                 = false;

    ggml_backend_sched_reset(sched_);
    for (ggml_tensor * a : attn_nodes_) {
        ggml_backend_sched_set_tensor_backend(sched_, a, gpu_ && n >= host_gpu_min_ ? gpu_ : cpu_);
    }
    if (!ggml_backend_sched_alloc_graph(sched_, gf)) {
        ggml_free(ctx);
        err = "eval: cannot allocate the compute graph";
        return false;
    }
    {
        for (int i = 0; i < n; i++) {
            if (tokens[i] < 0 || tokens[i] >= hp_.n_vocab) {
                ggml_free(ctx);
                err = "eval: token id out of range";
                return false;
            }
        }
        std::vector<float> emb;
        embed(tokens, n, emb);
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
        ggml_fp16_t * mr = mask.data() + (size_t) i * n_kv;
        const int     qp = n_past_ + i;
        if (window) {
            // ring slots holding positions in (qp - (W - n), qp]: the slots this batch overwrites are excluded
            // with a draft far area: its pages, then only ring positions past them (no token counted twice)
            const int lo = std::max(qp - (W_ - n), far_rows_ > 0 ? far_ws_ - 1 : -1);
            std::fill(mr, mr + Kd_, ninf);
            std::fill(mr, mr + far_rows_, zero);
            for (int j = 0; j < W_; j++) {
                const int sp  = slot_pos_[(size_t) j];
                mr[Kd_ + j] = sp >= 0 && sp <= qp && sp > lo ? zero : ninf;
            }
        } else {
            const int nz = std::min(n_kv, qp + 1);
            std::fill(mr, mr + nz, zero);
            std::fill(mr + nz, mr + n_kv, ninf);
        }
    }
    if (inp_kvidx_->buffer) {
        std::vector<int64_t> rows((size_t) n);
        for (int i = 0; i < n; i++) rows[(size_t) i] = Kd_ + (n_past_ + i) % W_;
        ggml_backend_tensor_set(inp_kvidx_, rows.data(), 0, rows.size() * sizeof(int64_t));
    }
    if (inp_hidx_ && inp_hidx_->buffer) {
        std::vector<int64_t> rows((size_t) n);
        for (int i = 0; i < n; i++) rows[(size_t) i] = n_past_ + i;
        ggml_backend_tensor_set(inp_hidx_, rows.data(), 0, rows.size() * sizeof(int64_t));
    }
    if (inp_mask->buffer) {
        ggml_backend_tensor_set(inp_mask, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
    }
    if (inp_mpos_ && inp_mpos_->buffer) {
        std::vector<int32_t> mp((size_t) (n - 1) * 4, 0);
        std::vector<int64_t> mr((size_t) (n - 1));
        for (int i = 0; i < n - 1; i++) {
            mp[(size_t) i] = mp[(size_t) (n - 1) + i] = mp[(size_t) 2 * (n - 1) + i] = n_past_ + i;
            mr[(size_t) i] = (n_past_ + i) % Wm_;
            mtp_slot_pos_[(size_t) mr[(size_t) i]] = n_past_ + i;
        }
        ggml_backend_tensor_set(inp_mpos_, mp.data(), 0, mp.size() * sizeof(int32_t));
        ggml_backend_tensor_set(inp_mkvidx_, mr.data(), 0, mr.size() * sizeof(int64_t));
    }
    if (inp_smask_ && inp_smask_->buffer) {
        // selected pages: visible; window rows: causal; padding: masked
        std::vector<ggml_fp16_t> sm((size_t) sp_nsel_ * n, ninf);
        const int kr = sp_kp_ * kPage;
        for (int i = 0; i < n; i++) {
            ggml_fp16_t * mr = sm.data() + (size_t) i * sp_nsel_;
            std::fill(mr, mr + kr, zero);
            std::fill(mr + kr, mr + kr + (n_past_ + i + 1 - sp_ws_), zero);
        }
        ggml_backend_tensor_set(inp_smask_, sm.data(), 0, sm.size() * sizeof(ggml_fp16_t));
    }
    if (inp_sbias_ && inp_sbias_->buffer) {
        std::vector<float> b((size_t) sp_nfar_, 0.0f);
        b[0] = 1e30f;  // the first page (attention sink) is always kept
        ggml_backend_tensor_set(inp_sbias_, b.data(), 0, b.size() * sizeof(float));
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
    if (sparse_ && Kd_ > 0 && sp_kp_ > 0) fill_draft_far();
    sel_nodes_.clear();
    ggml_free(ctx);
    recorded_n_ = opts.record ? n : 0;
    hid_row_    = mtp_on_ ? n - 1 : -1;
    if (!opts.window_ok) exact_upto_ = n_past_ + n;
    if (use_host) host_valid_ = n_past_ + n;
    n_past_ += n;
    return true;
}

} // namespace e8::model
