#include "model/qwen35.h"
#include "model/traffic.h"

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
#else
#include <sys/mman.h>
#endif

namespace e8::model {

namespace {

// Memory straight from the OS for the RAM KV. os_reserve() takes address space only; os_commit() backs a range
// (zero-filled, a page takes physical RAM once it is touched), os_decommit() gives a range back.
void * os_reserve(size_t n) {
#if defined(_WIN32)
    return VirtualAlloc(nullptr, n, MEM_RESERVE, PAGE_READWRITE);
#else
    void * p = mmap(nullptr, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    return p == MAP_FAILED ? nullptr : p;
#endif
}

bool os_commit(void * p, size_t n) {
#if defined(_WIN32)
    return VirtualAlloc(p, n, MEM_COMMIT, PAGE_READWRITE) != nullptr;
#else
    (void) p;
    (void) n;
    return true;
#endif
}

void os_decommit(void * p, size_t n) {
#if defined(_WIN32)
    VirtualFree(p, n, MEM_DECOMMIT);
#else
    madvise(p, n, MADV_DONTNEED);
#endif
}

void os_free(void * p, size_t n) {
#if defined(_WIN32)
    (void) n;
    VirtualFree(p, 0, MEM_RELEASE);
#else
    munmap(p, n);
#endif
}

// Keeps [p, p + n) in physical memory. Windows only locks pages within the process's minimum working set, so that
// is first set to `locked_total` (every byte locked so far, these included) plus a margin.
bool os_lock(void * p, size_t n, size_t locked_total) {
#if defined(_WIN32)
    static SIZE_T base_min = 0, base_max = 0;
    HANDLE        h        = GetCurrentProcess();
    if (!base_min) {
        DWORD fl = 0;
        if (!GetProcessWorkingSetSizeEx(h, &base_min, &base_max, &fl)) return false;
    }
    const SIZE_T mn = base_min + locked_total + (SIZE_T) (64u << 20);
    if (!SetProcessWorkingSetSizeEx(h, mn, std::max(base_max, mn) + (SIZE_T) (64u << 20),
                                    QUOTA_LIMITS_HARDWS_MIN_DISABLE | QUOTA_LIMITS_HARDWS_MAX_DISABLE)) {
        return false;
    }
    return VirtualLock(p, n) != 0;
#else
    (void) locked_total;
    return mlock(p, n) == 0;
#endif
}

size_t os_page() {
#if defined(_WIN32)
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return si.dwAllocationGranularity;  // 64 KiB: also the reservation granularity
#else
    return 65536;
#endif
}

constexpr int    kGraphSize = 16384;
constexpr int64_t kResChunkRows = 24832;  // residual rows per GPU-staged matmul piece (see load_residual)
// most queries per sparse long-context attention batch (bigger batches attend exactly over the RAM KV)
const int kSparseMaxN = std::getenv("E8_SPARSE_MAXN") ? std::atoi(std::getenv("E8_SPARSE_MAXN")) : 256;
// most tokens per sparse eval (one pass over the weights); its attention runs in sub-chunks of kSparseMaxN queries
const int kSparseBatch = std::getenv("E8_SPARSE_BATCH") ? std::atoi(std::getenv("E8_SPARSE_BATCH")) : 1024;
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
    if (rgbuf_) ggml_backend_buffer_free(rgbuf_);
    if (rgctx_) ggml_free(rgctx_);
    if (pbuf_) ggml_backend_buffer_free(pbuf_);
    if (pctx_) ggml_free(pctx_);
    if (hbuf_) ggml_backend_buffer_free(hbuf_);
    if (hmem_) os_free(hmem_, hmem_bytes_);
    if (slbuf_) ggml_backend_buffer_free(slbuf_);
    if (slmem_) os_free(slmem_, slmem_bytes_);
    if (slctx_) ggml_free(slctx_);
    if (slbuf2_) ggml_backend_buffer_free(slbuf2_);
    if (summem_) os_free(summem_, summem_bytes_);
    if (rvctx_) ggml_free(rvctx_);
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
    if (opt_.kv_type_v == GGML_TYPE_COUNT) opt_.kv_type_v = opt_.kv_type;
    const ggml_type kt = opt_.kv_type, vt = opt_.kv_type_v;
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
        const double  tok     = (double) n_attn * (double) (ggml_row_size(kt, row) + ggml_row_size(vt, row));
        int64_t       want    = opt.gpu_kv;
        if (want < 0) {
            size_t fr = 0, tot = 0;
            ggml_backend_dev_memory(ggml_backend_get_device(gpu_), &fr, &tot);
            // recurrent state and compute buffers (prefill batch, CUDA pools), then the RAM-KV attention staging:
            // one layer's per-head K/V copies, their F16 conversion and the mask
            const double state   = 2.0 * (double) (h.n_layer - n_attn) *
                                 (double) (h.ssm_d_state * h.ssm_d_state * h.ssm_n_v + h.conv_channels() * 4) * 4.0;
            // MTP: weights, a second state snapshot, its KV ring and the hidden-state buffer
            const double mtp_cost = mtp_w + (double) opt.mtp_window * (double) (ggml_row_size(kt, h.head_dim * h.n_head_kv) + ggml_row_size(vt, h.head_dim * h.n_head_kv)) +
                                    4.0 * (double) h.n_embd * (double) (opt_.n_ubatch + 1);
            double       reserve = (std::getenv("E8_VRAM_RESERVE_GB") ? std::atof(std::getenv("E8_VRAM_RESERVE_GB")) * 1e9 : 0.15e9) + state;  // measured: 256K ctx, 7.9K window peaked at 15.5 of 16.3 GB
            if (opt.n_slots > 1) {
                // slots: no state snapshot; records past 64 rows; the slot drafter: 32-row records, its graphs and
                // buffers (0.3 GB: 262K context, 4 slots peaked at 15.74 of 16.3 GB with 0.2; VRAM over-committed spills to
                // system memory, 10x slower) and
                // draft_batch - 1 more recurrent states
                const double rec_row = 4.0 * (double) (h.n_layer - n_attn) * (double) (h.conv_channels() + 2 * h.ssm_n_v);
                reserve += -state / 2.0 + rec_row * (double) (std::max(0, opt.max_record - 64) + 32) + 0.3e9;
                reserve += state / 2.0 * (double) std::max(0, opt.draft_batch - 1);
            }
            // with sparse attention big batches are split (see eval), so the compute buffer is the biggest verify's
            // (~0.35 GB at 88 tokens, measured; 0.5 for 128); else one layer's full per-head K/V copies, F16, mask
            const bool   sparse_env = !(std::getenv("E8_SPARSE") && std::atoi(std::getenv("E8_SPARSE")) == 0);
            const double stage   = sparse_env ? (opt.n_slots > 1 ? 0.35e9 : 0.5e9)  // (slots: joint verifies of <= 64 rows)
                                              : (double) n_ctx_ * ((double) h.n_head_kv * (double) (ggml_row_size(kt, h.head_dim) + ggml_row_size(vt, h.head_dim)) +
                                                                   4.0 * (double) h.head_dim + 2.0 * (double) opt_.n_ubatch);
            // prefill batches past 512 must not cost MTP or the whole KV in VRAM: then they drop to 512
            const double big = 0.15e9 * std::max(0, opt_.n_ubatch - 512) / 512.0;
            const double kv  = tok * (double) n_ctx_;
            // (with a split KV the sparse staging reserve below already covers a 1024-token batch: 0.38 GB measured)
            if (big > 0 && (double) fr - reserve - big < kv + (want_mtp ? mtp_cost : 0.0) &&
                (double) fr - reserve >= kv + (want_mtp ? mtp_cost : 0.0)) {
                opt_.n_ubatch = 512;
            } else if ((double) fr - reserve - big >= kv) {
                reserve += big;
            }
            if (want_mtp && (double) fr - reserve - mtp_cost >= tok * (double) n_ctx_) {
                want    = n_ctx_;
                mtp_on_ = true;
            } else if ((double) fr - reserve >= tok * (double) n_ctx_) {
                want = n_ctx_;
            } else {
                const double summ = opt.n_slots > 1 ? 0.0 :  // (slots keep them in RAM)
                                        (double) n_attn * (double) h.n_head_kv * (double) (n_ctx_ / kPage) * (double) h.head_dim * 2.0;
                want = (int64_t) (((double) fr - reserve - stage - summ) / tok);
                // MTP beside a split KV (E8_MTP_MIN_WINDOW=tokens): when the window left after it is still that big
                static const int64_t min_win = std::getenv("E8_MTP_MIN_WINDOW") ? std::atoll(std::getenv("E8_MTP_MIN_WINDOW")) : 4096;
                if (want_mtp && min_win > 0) {
                    const int64_t w2 = (int64_t) (((double) fr - reserve - stage - summ - mtp_cost) / tok);
                    if (std::getenv("E8_VRAM_DEBUG")) {
                        fprintf(stderr, "vram: free %.2f GB, reserve %.2f, stage %.2f, summaries %.2f, MTP %.2f; %.0f B per token -> "
                                        "window %lld with MTP\n", fr / 1e9, reserve / 1e9, stage / 1e9, summ / 1e9, mtp_cost / 1e9, tok,
                                (long long) w2);
                    }
                    // (slots: each slot's window is a share of the ring, so what MTP needs left is per slot)
                    if (opt.n_slots > 1 ? w2 >= (int64_t) opt.n_slots * 2560 : w2 - 4096 >= min_win) {
                        want    = w2;
                        mtp_on_ = true;
                    }
                }
            }
        }
        want = std::min<int64_t>(n_ctx_, want / kKvPad * kKvPad);
        if (opt.n_slots > 1 && want >= n_ctx_) want = n_ctx_ - kKvPad;  // slots keep their sequences in the RAM KV
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
        traffic::mtp_bytes = ggml_backend_buffer_get_size(mbuf_);
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
            if (opt.n_slots <= 1) {  // (slots never snapshot: verifies are dry, drafting has working states)
                conv_bak_[(size_t) il] = ggml_dup_tensor(c, conv_state_[(size_t) il]);
                ssm_bak_[(size_t) il]  = ggml_dup_tensor(c, ssm_state_[(size_t) il]);
            }
            // (proposal checks while drafting use dry evals + commit, so slot 1 needs no memory)
            // prompt-reuse checkpoint: copied once per request, so it lives in RAM
            conv_ck_[(size_t) il]    = ggml_dup_tensor(sctx_[1], conv_state_[(size_t) il]);
            ssm_ck_[(size_t) il]     = ggml_dup_tensor(sctx_[1], ssm_state_[(size_t) il]);
            rec_qkv_[(size_t) il]    = ggml_new_tensor_2d(c, GGML_TYPE_F32, h.conv_channels(), n_rec);
            rec_g_[(size_t) il]      = ggml_new_tensor_2d(c, GGML_TYPE_F32, h.ssm_n_v, n_rec);
            rec_beta_[(size_t) il]   = ggml_new_tensor_2d(c, GGML_TYPE_F32, h.ssm_n_v, n_rec);
        } else {
            k_cache_[(size_t) il] = ggml_new_tensor_2d(c, kt, h.head_dim * h.n_head_kv, Kd_ + W_);
            v_cache_[(size_t) il] = ggml_new_tensor_2d(c, vt, h.head_dim * h.n_head_kv, Kd_ + W_);
        }
    }
    if (mtp_on_) {
        Wm_       = std::max(kKvPad, std::min(n_ctx_, opt.mtp_window / kKvPad * kKvPad));
        mtp_k_    = ggml_new_tensor_2d(sctx_[0], kt, h.head_dim * h.n_head_kv, Wm_);
        mtp_v_    = ggml_new_tensor_2d(sctx_[0], vt, h.head_dim * h.n_head_kv, Wm_);
        mtp_hid_  = ggml_new_tensor_2d(sctx_[0], GGML_TYPE_F32, h.n_embd, opt_.n_ubatch);
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
        const int    ns    = std::max(1, opt.n_slots);
        const size_t nhead = (size_t) (h.n_layer * h.n_head_kv);
        side_on_           = opt.side_seq && ns == 1 && !(std::getenv("E8_KV_PINNED") && std::atoi(std::getenv("E8_KV_PINNED")) != 0);
        const int    nreg  = ns + (side_on_ ? 1 : 0);  // RAM KV regions
        ggml_init_params hp = { ggml_tensor_overhead() * (nhead * 2 * (size_t) nreg + 8), nullptr, true };
        hctx_               = ggml_init(hp);
        slots_.assign((size_t) ns, KvSlot{});
        std::vector<KvSlot *> regs;
        for (KvSlot & sl : slots_) regs.push_back(&sl);
        if (side_on_) regs.push_back(&side_.kv);
        for (KvSlot * slp : regs) {
            KvSlot & sl = *slp;
            sl.hk.assign(nhead, nullptr);
            sl.hv.assign(nhead, nullptr);
            for (int64_t il = 0; il < h.n_layer; il++) {
                if (h.is_recurrent(il)) continue;
                for (int64_t j = 0; j < h.n_head_kv; j++) {
                    sl.hk[(size_t) (il * h.n_head_kv + j)] = ggml_new_tensor_2d(hctx_, kt, h.head_dim, n_ctx_);
                    sl.hv[(size_t) (il * h.n_head_kv + j)] = ggml_new_tensor_2d(hctx_, vt, h.head_dim, n_ctx_);
                }
            }
        }
        hk_ = slots_[0].hk;
        hv_ = slots_[0].hv;
        for (size_t i = 0; i < nhead; i++) {
            if (hk_[i]) kv_row_bytes_ += hk_[i]->nb[1] + hv_[i]->nb[1];
        }
        // Masked rows must still be finite for the GPU kernels: zero. Pinned memory (one slot only) is cleared and
        // resident. Otherwise each slot's region is reserved address space that gets memory (zero-filled by the OS)
        // as positions are written (ensure_committed), so a slot takes RAM for the context it holds, not for n_ctx.
        if (ns == 1 && std::getenv("E8_KV_PINNED") && std::atoi(std::getenv("E8_KV_PINNED")) != 0) {
            hbuf_ = ggml_backend_alloc_ctx_tensors_from_buft(hctx_, ggml_backend_dev_host_buffer_type(ggml_backend_get_device(gpu_)));
            if (!hbuf_) fprintf(stderr, "warning: pinned allocation for the KV failed, using pageable RAM\n");
            else ggml_backend_buffer_clear(hbuf_, 0);
            slots_[0].committed = n_ctx_;
        }
        if (!hbuf_) {
            constexpr size_t kAlign = 64;
            const size_t     gran   = os_page();
            size_t           per    = 0;
            for (size_t i = 0; i < nhead; i++) {
                if (slots_[0].hk[i]) per += GGML_PAD(ggml_nbytes(slots_[0].hk[i]), kAlign) + GGML_PAD(ggml_nbytes(slots_[0].hv[i]), kAlign);
            }
            per         = (per + gran - 1) / gran * gran;
            hmem_bytes_ = per * (size_t) nreg;
            hmem_       = os_reserve(hmem_bytes_);
            if (!hmem_) {
                err = "cannot reserve address space for the RAM KV";
                return false;
            }
            hbuf_ = ggml_backend_cpu_buffer_from_ptr(hmem_, hmem_bytes_);
            for (int s = 0; s < nreg; s++) {
                KvSlot & sl = *regs[(size_t) s];
                sl.base     = (uint8_t *) hmem_ + per * (size_t) s;
                sl.bytes    = per;
                size_t off  = 0;
                for (size_t i = 0; i < nhead; i++) {
                    for (ggml_tensor * t : { sl.hk[i], sl.hv[i] }) {
                        if (!t) continue;
                        if (ggml_backend_tensor_alloc(hbuf_, t, sl.base + off) != GGML_STATUS_SUCCESS) {
                            err = "cannot place the RAM KV tensors";
                            return false;
                        }
                        off += GGML_PAD(ggml_nbytes(t), kAlign);
                    }
                }
            }
        }
        pool_bytes_ = opt.kv_pool_gb > 0 ? (size_t) (opt.kv_pool_gb * 1e9) : kv_row_bytes_ * (size_t) n_ctx_;
        if (side_on_ && opt.kv_pool_gb <= 0) pool_bytes_ += kv_row_bytes_ * 32768;  // (room for a side sequence)
        // slots 1..: parked recurrent state and a prompt-reuse checkpoint each, in RAM (pages on demand); slot 0's
        // checkpoint is conv_ck_/ssm_ck_ and its parked state lives here too
        slots_[0].conv_ck = conv_ck_;
        slots_[0].ssm_ck  = ssm_ck_;
        if (ns > 1) {
            ggml_init_params sp2 = { ggml_tensor_overhead() * (size_t) (h.n_layer * 4 * ns + 8), nullptr, true };
            slctx_               = ggml_init(sp2);
            for (int s = 0; s < ns; s++) {
                KvSlot & sl = slots_[(size_t) s];
                sl.conv.assign((size_t) h.n_layer, nullptr);
                sl.ssm.assign((size_t) h.n_layer, nullptr);
                if (s > 0) {
                    sl.conv_ck.assign((size_t) h.n_layer, nullptr);
                    sl.ssm_ck.assign((size_t) h.n_layer, nullptr);
                }
                for (size_t il = 0; il < (size_t) h.n_layer; il++) {
                    if (!conv_state_[il]) continue;
                    sl.conv[il] = ggml_dup_tensor(slctx_, conv_state_[il]);
                    sl.ssm[il]  = ggml_dup_tensor(slctx_, ssm_state_[il]);
                    if (s > 0) {
                        sl.conv_ck[il] = ggml_dup_tensor(slctx_, conv_state_[il]);
                        sl.ssm_ck[il]  = ggml_dup_tensor(slctx_, ssm_state_[il]);
                    }
                }
            }
            constexpr size_t kAlign = 64;
            for (ggml_tensor * t = ggml_get_first_tensor(slctx_); t; t = ggml_get_next_tensor(slctx_, t)) {
                slmem_bytes_ += GGML_PAD(ggml_nbytes(t), kAlign);
            }
            slmem_ = os_reserve(slmem_bytes_);
            if (!slmem_ || !os_commit(slmem_, slmem_bytes_)) {
                err = "not enough memory for the slots' recurrent states";
                return false;
            }
            slbuf_     = ggml_backend_cpu_buffer_from_ptr(slmem_, slmem_bytes_);
            size_t off = 0;
            for (ggml_tensor * t = ggml_get_first_tensor(slctx_); t; t = ggml_get_next_tensor(slctx_, t)) {
                ggml_backend_tensor_alloc(slbuf_, t, (uint8_t *) slmem_ + off);
                off += GGML_PAD(ggml_nbytes(t), kAlign);
            }
        }
        pmid_.assign(hk_.size(), nullptr);
        if (ns == 1) {  // (slots: each slot's summaries live in RAM, pmid_ points at the active slot's)
            ggml_init_params pp = { ggml_tensor_overhead() * (size_t) (h.n_layer * h.n_head_kv * 2 + 8), nullptr, true };
            pctx_               = ggml_init(pp);
            for (size_t i = 0; i < hk_.size(); i++) {
                if (!hk_[i]) continue;
                pmid_[i] = ggml_new_tensor_2d(pctx_, GGML_TYPE_F16, h.head_dim, n_ctx_ / kPage);
            }
            pbuf_ = ggml_backend_alloc_ctx_tensors_from_buft(pctx_, ggml_backend_get_default_buffer_type(gpu_));
            if (!pbuf_) {
                err = "not enough VRAM for the key page summaries";
                return false;
            }
        }
        if (ns > 1) {
            // each slot owns 1/ns of the VRAM ring (its own draft far area first, as the whole ring has) and of the MTP
            // ring: views into slot 0's tensors, swapped in by select_slot (all slots share W_ / Kd_ / Wm_)
            const int rs  = (Kd_ + W_) / ns / kKvPad * kKvPad;
            const int rms = mtp_on_ ? std::max(kKvPad, Wm_ / ns / kKvPad * kKvPad) : 0;
            if (rs < 2048) {
                err = "not enough VRAM for " + std::to_string(ns) + " slots' KV windows (" + std::to_string(rs) +
                      " tokens each; fewer --slots or a smaller --ctx)";
                return false;
            }
            ggml_init_params rp = { ggml_tensor_overhead() * (size_t) ((h.n_layer * 2 + 2 + h.n_layer * h.n_head_kv) * ns + 8), nullptr, true };
            rvctx_              = ggml_init(rp);
            auto view = [&](ggml_tensor * src, int rows, int s) {
                ggml_tensor * t = ggml_new_tensor_2d(rvctx_, src->type, src->ne[0], rows);
                ggml_backend_tensor_alloc(src->buffer, t, (uint8_t *) src->data + src->nb[1] * (size_t) rows * (size_t) s);
                return t;
            };
            const size_t pages = (size_t) (n_ctx_ / kPage), sum_head = pages * (size_t) h.head_dim * sizeof(ggml_fp16_t);
            size_t       nsum  = 0;
            for (ggml_tensor * t : hk_) nsum += t ? 1 : 0;
            summem_bytes_ = sum_head * nsum * (size_t) ns;
            summem_       = os_reserve(summem_bytes_);
            if (!summem_ || !os_commit(summem_, summem_bytes_)) {
                err = "not enough memory for the slots' page summaries";
                return false;
            }
            slbuf2_ = ggml_backend_cpu_buffer_from_ptr(summem_, summem_bytes_);
            for (int s = 0; s < ns; s++) {
                KvSlot & sl = slots_[(size_t) s];
                sl.kr.assign((size_t) h.n_layer, nullptr);
                sl.vr.assign((size_t) h.n_layer, nullptr);
                for (size_t il = 0; il < (size_t) h.n_layer; il++) {
                    if (!k_cache_[il]) continue;
                    sl.kr[il] = view(k_cache_[il], rs, s);
                    sl.vr[il] = view(v_cache_[il], rs, s);
                }
                if (mtp_on_) {
                    sl.mk = view(mtp_k_, rms, s);
                    sl.mv = view(mtp_v_, rms, s);
                }
                sl.summ = (uint8_t *) summem_ + sum_head * nsum * (size_t) s;
                sl.sumt.assign(hk_.size(), nullptr);
                size_t hi = 0;
                for (size_t i = 0; i < hk_.size(); i++) {
                    if (!hk_[i]) continue;
                    sl.sumt[i] = ggml_new_tensor_2d(rvctx_, GGML_TYPE_F16, h.head_dim, (int64_t) pages);
                    ggml_backend_tensor_alloc(slbuf2_, sl.sumt[i], sl.summ + sum_head * hi);
                    hi++;
                }
            }
            k_cache_ = slots_[0].kr;
            v_cache_ = slots_[0].vr;
            pmid_    = slots_[0].sumt;
            if (mtp_on_) {
                mtp_k_ = slots_[0].mk;
                mtp_v_ = slots_[0].mv;
                Wm_    = rms;
            }
            // the draft far area keeps drafts on long contexts useful (without it, 17K-token prompt: 4.2 vs 21.6 tokens
            // per cycle): as for one sequence, but shrunk (whole pages) to leave the ring at least 2048 rows
            Kd_ = Kd_ > 0 ? std::max(0, std::min(Kd_, rs - 2048)) / kPage * kPage : 0;
            W_  = rs - Kd_;
        }
        if (const char * e = std::getenv("E8_HOST_ATTN_GPU_MIN")) host_gpu_min_ = std::atoi(e);
        fprintf(stderr, "KV: %d tokens in RAM (%.2f GB, K %s V %s, %s), window of %d in VRAM", n_ctx_,
                (double) (kv_row_bytes_ * (size_t) n_ctx_) / 1e9, ggml_type_name(kt), ggml_type_name(vt),
                !hmem_ ? "pinned" : opt.kv_lock ? "locked as used" : "taken as used", W_);
        if (ns > 1) fprintf(stderr, " (+ %d draft far rows) per slot; %d slots, pool %.2f GB", Kd_, ns, (double) pool_bytes_ / 1e9);
        fprintf(stderr, "\n");
    } else if (opt.n_slots > 1) {
        err = "sequence slots need the KV in RAM (a GPU with all layers on it)";
        return false;
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
    graph_meta_draft_.resize(graph_meta_.size());
    graph_meta_mtp_.resize(graph_meta_.size());
    traffic::base_bytes = gpu_bytes_ - (mbuf_ ? ggml_backend_buffer_get_size(mbuf_) : 0);
    return true;
}

bool Qwen35::make_shadow(const Qwen35 & src, std::string & err) {
    // a second sequence over the same weights and KV: own recurrent state, records, MTP hidden rows, backends
    // (own CUDA stream, own CPU threads) and scheduler, so it can draft while `src` verifies
    const auto & h = src.hp_;
    shadow_        = true;
    hp_            = h;
    opt_           = src.opt_;
    opt_.n_ubatch  = std::min(src.opt_.n_ubatch, 128);  // drafts and their checks only
    opt_.residual_path.clear();
    n_ctx_ = src.n_ctx_;
    W_     = src.W_;
    Kd_    = src.Kd_;
    mtp_on_ = src.mtp_on_;
    Wm_     = src.Wm_;
    tok_embd_ = src.tok_embd_;
    out_norm_ = src.out_norm_;
    output_   = src.output_;
    layers_   = src.layers_;
    mtp_      = src.mtp_;
    mtp_eh_   = src.mtp_eh_;
    mtp_enorm_ = src.mtp_enorm_;
    mtp_hnorm_ = src.mtp_hnorm_;
    mtp_norm_  = src.mtp_norm_;
    mtp_k_     = src.mtp_k_;
    mtp_v_     = src.mtp_v_;
    k_cache_   = src.k_cache_;
    v_cache_   = src.v_cache_;
    hk_        = src.hk_;
    hv_        = src.hv_;
    pmid_      = src.pmid_;
    host_gpu_min_ = src.host_gpu_min_;
    cpu_ = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!cpu_) {
        err = "shadow: cannot initialize the CPU backend";
        return false;
    }
    ggml_backend_cpu_set_n_threads(cpu_, 4);
    if (src.gpu_) gpu_ = ggml_backend_dev_init(ggml_backend_get_device(src.gpu_), nullptr);
    ggml_init_params sp = { ggml_tensor_overhead() * (size_t) (h.n_layer * 14 + 16), nullptr, true };
    sctx_[0]            = ggml_init(sp);
    sctx_[1]            = ggml_init(sp);
    const size_t nl     = (size_t) h.n_layer;
    conv_state_.assign(nl, nullptr);
    ssm_state_.assign(nl, nullptr);
    conv_bak_.assign(nl, nullptr);
    ssm_bak_.assign(nl, nullptr);
    conv_bak1_.assign(nl, nullptr);
    ssm_bak1_.assign(nl, nullptr);
    conv_ck_.assign(nl, nullptr);
    ssm_ck_.assign(nl, nullptr);
    rec_qkv_.assign(nl, nullptr);
    rec_g_.assign(nl, nullptr);
    rec_beta_.assign(nl, nullptr);
    const int64_t n_rec = std::max(1, opt_.max_record);
    for (size_t il = 0; il < nl; il++) {
        if (!src.conv_state_[il]) continue;
        ggml_context * c = sctx_[src.layers_[il].on_gpu ? 0 : 1];
        conv_state_[il]  = ggml_dup_tensor(c, src.conv_state_[il]);
        ssm_state_[il]   = ggml_dup_tensor(c, src.ssm_state_[il]);
        rec_qkv_[il]     = ggml_new_tensor_2d(c, GGML_TYPE_F32, h.conv_channels(), n_rec);
        rec_g_[il]       = ggml_new_tensor_2d(c, GGML_TYPE_F32, h.ssm_n_v, n_rec);
        rec_beta_[il]    = ggml_new_tensor_2d(c, GGML_TYPE_F32, h.ssm_n_v, n_rec);
    }
    if (mtp_on_) {
        mtp_hid_   = ggml_new_tensor_2d(sctx_[0], GGML_TYPE_F32, h.n_embd, opt_.n_ubatch);
        mtp_chain_ = ggml_new_tensor_2d(sctx_[0], GGML_TYPE_F32, h.n_embd, 1);
    }
    for (int i = 0; i < 2; i++) {
        ggml_backend_t be = i == 0 ? gpu_ : cpu_;
        if (!be || !ggml_get_first_tensor(sctx_[i])) continue;
        sbuf_[i] = ggml_backend_alloc_ctx_tensors_from_buft(sctx_[i], ggml_backend_get_default_buffer_type(be));
        if (!sbuf_[i]) {
            err = "shadow: not enough VRAM for a second recurrent state";
            return false;
        }
    }
    reset();
    std::vector<ggml_backend_t> bes;
    if (gpu_) bes.push_back(gpu_);
    bes.push_back(cpu_);
    sched_ = ggml_backend_sched_new(bes.data(), nullptr, (int) bes.size(), kGraphSize, false, true);
    graph_meta_.resize(ggml_tensor_overhead() * kGraphSize + ggml_graph_overhead_custom(kGraphSize, false));
    graph_meta_draft_.resize(graph_meta_.size());
    graph_meta_mtp_.resize(graph_meta_.size());
    return true;
}

void Qwen35::sync_from(const Qwen35 & src) {
    // take over src's committed sequence: recurrent state, position, KV bookkeeping and the MTP hidden row
    for (size_t il = 0; il < conv_state_.size(); il++) {
        if (conv_state_[il]) {
            ggml_backend_tensor_copy(src.conv_state_[il], conv_state_[il]);
            ggml_backend_tensor_copy(src.ssm_state_[il], ssm_state_[il]);
        }
    }
    n_past_       = src.n_past_;
    slot_pos_     = src.slot_pos_;
    mtp_slot_pos_ = src.mtp_slot_pos_;
    host_valid_   = src.host_valid_;
    exact_upto_   = src.exact_upto_;
    sum_upto_     = src.sum_upto_;
    far_rows_     = src.far_rows_;
    far_ws_       = src.far_ws_;
    recorded_n_ = recorded_full_ = 0;
    dry_pending_ = false;
    hid_row_     = -1;
    if (mtp_on_ && src.hid_row_ >= 0 && src.mtp_hid_) {
        std::vector<float> row((size_t) hp_.n_embd);
        ggml_backend_tensor_get(src.mtp_hid_, row.data(), src.mtp_hid_->nb[1] * (size_t) src.hid_row_, row.size() * sizeof(float));
        ggml_backend_tensor_set(mtp_hid_, row.data(), 0, row.size() * sizeof(float));
        hid_row_ = 0;
    }
}

void Qwen35::reset() {
    if (slots_.size() > 1) {
        // the state buffers also hold the other slots' ring parts and slot 0's checkpoint: clear only this slot's
        for (size_t il = 0; il < conv_state_.size(); il++) {
            for (ggml_tensor * t : { conv_state_[il], ssm_state_[il], k_cache_[il], v_cache_[il] }) {
                if (t) ggml_backend_tensor_memset(t, 0, 0, ggml_nbytes(t));
            }
        }
        if (mtp_on_) {
            ggml_backend_tensor_memset(mtp_k_, 0, 0, ggml_nbytes(mtp_k_));
            ggml_backend_tensor_memset(mtp_v_, 0, 0, ggml_nbytes(mtp_v_));
        }
    } else {
        for (int i = 0; i < 2; i++) {
            if (sbuf_[i]) {
                ggml_backend_buffer_clear(sbuf_[i], 0);
            }
        }
    }
    n_past_       = 0;
    saved_n_past_ = saved_n_past1_ = 0;
    recorded_n_   = 0;
    hid_row_      = -1;
    mtp_slot_pos_.assign((size_t) Wm_, -1);
    slot_pos_.assign((size_t) W_, -1);
    host_valid_ = exact_upto_ = 0;
    sum_upto_ = sum_low_ = 0;
    far_rows_ = 0;
    ck_n_past_ = 0;
    if (!slots_.empty() && !shadow_ && cur_slot_ >= 0) drop_slot(slots_[(size_t) cur_slot_], false);
}

void Qwen35::drop_slot(KvSlot & s, bool forget) {
    if (s.base && s.committed > 0) {
        os_decommit(s.base, s.bytes);
        committed_bytes_ -= std::min(committed_bytes_, kv_row_bytes_ * (size_t) s.committed);
        if (opt_.kv_lock) locked_bytes_ -= std::min(locked_bytes_, kv_row_bytes_ * (size_t) s.committed);
        s.committed = 0;
    }
    s.n_past = s.host_valid = s.exact_upto = s.ck_n_past = 0;
    s.sum_upto = s.sum_saved = 0;
    s.far_rows = s.far_ws = 0;
    s.slot_pos.clear();
    s.mtp_slot_pos.clear();
    s.hid.clear();
    if (forget) s.epoch++;
}

bool Qwen35::ensure_committed(int upto, std::string & err) {
    if (slots_.empty() || cur_slot_ < 0) return true;
    return ensure_committed_slot(slots_[(size_t) cur_slot_], upto, err);
}

bool Qwen35::ensure_committed_slot(KvSlot & sl, int upto, std::string & err) {
    if (!sl.base || upto <= sl.committed) return true;
    constexpr int kChunk = 4096;  // positions committed at a time
    const int     to     = std::min(n_ctx_, (upto + kChunk - 1) / kChunk * kChunk);
    const size_t  need   = kv_row_bytes_ * (size_t) (to - sl.committed);
    // over the pool budget: empty the least recently used idle slots first
    while (committed_bytes_ + need > pool_bytes_) {
        KvSlot * lru = nullptr;
        for (KvSlot & o : slots_) {
            if (&o != &sl && !o.pinned && o.committed > 0 && (!lru || o.used < lru->used)) lru = &o;
        }
        if (!lru && side_on_ && &sl != &side_.kv && side_.kv.committed > 0) {
            fprintf(stderr, "KV pool: dropping the parked side sequence (%d tokens)\n", side_.n_past);
            drop_side();
            continue;
        }
        if (!lru) {
            err = "RAM KV pool full (" + std::to_string(pool_bytes_ / 1000000) + " MB, the other slots in use): lower the context or raise --kv-pool-gb";
            return false;
        }
        fprintf(stderr, "KV pool: emptying slot %d (%d tokens) for slot %d\n", (int) (lru - slots_.data()), lru->n_past,
                (int) (&sl - slots_.data()));
        drop_slot(*lru);
    }
    const size_t pg = 4096;
    for (size_t i = 0; i < sl.hk.size(); i++) {
        for (ggml_tensor * t : { sl.hk[i], sl.hv[i] }) {
            if (!t) continue;
            uint8_t * a = (uint8_t *) t->data + t->nb[1] * (size_t) sl.committed;
            uint8_t * b = (uint8_t *) t->data + t->nb[1] * (size_t) to;
            a           = (uint8_t *) ((uintptr_t) a / pg * pg);
            b           = (uint8_t *) (((uintptr_t) b + pg - 1) / pg * pg);
            if (!os_commit(a, (size_t) (b - a))) {
                err = "out of memory for the RAM KV";
                return false;
            }
            if (opt_.kv_lock && !os_lock(a, (size_t) (b - a), locked_bytes_ + need)) {
                static bool warned = false;
                if (!warned) fprintf(stderr, "warning: could not lock RAM KV pages in memory; they may be paged out\n");
                warned = true;
            }
        }
    }
    committed_bytes_ += need;
    if (opt_.kv_lock) locked_bytes_ += need;
    sl.committed = to;
    return true;
}

void Qwen35::park_active() {
    // the active slot's sequence into its slot: the ring's exact positions flushed to its RAM KV (so a parked slot's
    // RAM KV holds [0, n_past)), recurrent state, new page summaries and the MTP hidden row copied to RAM. The VRAM
    // state stays as it is, so the slot can go on as the active one.
    if (cur_slot_ < 0) return;  // detached: every slot is parked
    const auto & h     = hp_;
    const size_t pages = (size_t) (n_ctx_ / kPage), row = (size_t) h.head_dim * sizeof(ggml_fp16_t);
    KvSlot &     o     = slots_[(size_t) cur_slot_];
    std::string  e;
    const int    upto  = std::min(exact_upto_, n_past_);
    if (host_valid_ < upto && !flush_ring(host_valid_, upto, e)) fprintf(stderr, "slot %d: %s\n", cur_slot_, e.c_str());
    for (size_t il = 0; il < conv_state_.size(); il++) {
        if (!conv_state_[il]) continue;
        ggml_backend_tensor_copy(conv_state_[il], o.conv[il]);
        ggml_backend_tensor_copy(ssm_state_[il], o.ssm[il]);
    }
    o.n_past     = n_past_;
    o.host_valid = host_valid_;
    o.exact_upto = exact_upto_;
    o.ck_n_past  = ck_n_past_;
    o.far_rows   = far_rows_;
    o.far_ws     = far_ws_;
    o.sum_upto = o.sum_saved = sum_upto_;  // (its summaries are pmid_: already in its RAM copy)
    sum_low_   = sum_upto_;
    o.hid.clear();
    if (mtp_on_ && hid_row_ >= 0) {
        o.hid.resize((size_t) h.n_embd);
        ggml_backend_tensor_get(mtp_hid_, o.hid.data(), mtp_hid_->nb[1] * (size_t) hid_row_, o.hid.size() * sizeof(float));
    }
}

bool Qwen35::select_slot(int s, std::string & err) {
    if (slots_.empty() && s == 0) return true;
    if (s < 0 || s >= (int) slots_.size()) {
        err = "no sequence slot " + std::to_string(s);
        return false;
    }
    if (s == cur_slot_) {
        slots_[(size_t) s].used = ++use_clock_;
        return true;
    }
    if (dry_pending_) {
        err = "select_slot: a dry eval is not committed";
        return false;
    }
    const auto & h     = hp_;
    const size_t pages = (size_t) (n_ctx_ / kPage), row = (size_t) h.head_dim * sizeof(ggml_fp16_t);
    if (cur_slot_ >= 0) {
        park_active();
        KvSlot & o    = slots_[(size_t) cur_slot_];
        o.slot_pos     = std::move(slot_pos_);
        o.mtp_slot_pos = std::move(mtp_slot_pos_);
    }

    // resume slot s
    KvSlot & t = slots_[(size_t) s];
    cur_slot_  = s;
    t.used     = ++use_clock_;
    hk_        = t.hk;
    hv_        = t.hv;
    conv_ck_   = t.conv_ck;
    ssm_ck_    = t.ssm_ck;
    k_cache_   = t.kr;
    v_cache_   = t.vr;
    if (mtp_on_) {
        mtp_k_ = t.mk;
        mtp_v_ = t.mv;
    }
    for (size_t il = 0; il < conv_state_.size(); il++) {
        if (!conv_state_[il]) continue;
        if (t.n_past > 0) {
            ggml_backend_tensor_copy(t.conv[il], conv_state_[il]);
            ggml_backend_tensor_copy(t.ssm[il], ssm_state_[il]);
        } else {
            ggml_backend_tensor_memset(conv_state_[il], 0, 0, ggml_nbytes(conv_state_[il]));
            ggml_backend_tensor_memset(ssm_state_[il], 0, 0, ggml_nbytes(ssm_state_[il]));
        }
    }
    n_past_       = t.n_past;
    host_valid_   = t.host_valid;
    exact_upto_   = t.exact_upto;
    ck_n_past_    = t.ck_n_past;
    slot_pos_     = std::move(t.slot_pos);
    mtp_slot_pos_ = std::move(t.mtp_slot_pos);
    if ((int) slot_pos_.size() != W_) slot_pos_.assign((size_t) W_, -1);
    if ((int) mtp_slot_pos_.size() != Wm_) mtp_slot_pos_.assign((size_t) Wm_, -1);
    sum_upto_ = sum_low_ = t.sum_upto;
    pmid_     = t.sumt;
    // the hidden row goes to the last row of mtp_hid_: a joint verify's rows (from 0) stay intact for commit_seq
    hid_row_ = -1;
    if (mtp_on_ && !t.hid.empty()) {
        hid_row_ = opt_.n_ubatch - 1;
        ggml_backend_tensor_set(mtp_hid_, t.hid.data(), mtp_hid_->nb[1] * (size_t) hid_row_, t.hid.size() * sizeof(float));
    }
    saved_n_past_ = saved_n_past1_ = 0;
    recorded_n_ = recorded_full_ = 0;
    far_rows_   = t.far_rows;
    far_ws_     = t.far_ws;
    return true;
}

void Qwen35::drop_side() {
    if (!side_on_) return;
    drop_slot(side_.kv, true);
    side_.n_past = side_.host_valid = side_.exact_upto = side_.ck_n_past = side_.far_rows = side_.far_ws = side_.sum_upto = 0;
    side_.slot_pos.clear();
    side_.mtp_slot_pos.clear();
    std::vector<uint8_t>().swap(side_.blob);
}

bool Qwen35::swap_side(std::string & err) {
    if (!side_on_) {
        err = "no side sequence (one slot with the KV in RAM only)";
        return false;
    }
    if (dry_pending_) {
        err = "swap_side: a dry eval is not committed";
        return false;
    }
    // the VRAM state of a sequence of n_past positions, in a fixed order: recurrent state, prompt-reuse checkpoint,
    // ring rows in use (draft far area first), MTP ring rows in use, page summaries
    struct Part {
        ggml_tensor * t;
        size_t        n;
    };
    auto parts = [&](int n_past, int ck_n_past, int sum_upto) {
        std::vector<Part> v;
        if (n_past <= 0) return v;
        for (size_t il = 0; il < conv_state_.size(); il++) {
            if (!conv_state_[il]) continue;
            v.push_back({ conv_state_[il], ggml_nbytes(conv_state_[il]) });
            v.push_back({ ssm_state_[il], ggml_nbytes(ssm_state_[il]) });
            if (ck_n_past > 0) {
                v.push_back({ conv_ck_[il], ggml_nbytes(conv_ck_[il]) });
                v.push_back({ ssm_ck_[il], ggml_nbytes(ssm_ck_[il]) });
            }
        }
        const int rows = Kd_ + std::min(n_past, W_);
        for (size_t il = 0; il < k_cache_.size(); il++) {
            if (!k_cache_[il]) continue;
            v.push_back({ k_cache_[il], k_cache_[il]->nb[1] * (size_t) rows });
            v.push_back({ v_cache_[il], v_cache_[il]->nb[1] * (size_t) rows });
        }
        if (mtp_on_) {
            const int mr = std::min(n_past, Wm_);
            v.push_back({ mtp_k_, mtp_k_->nb[1] * (size_t) mr });
            v.push_back({ mtp_v_, mtp_v_->nb[1] * (size_t) mr });
        }
        for (ggml_tensor * t : pmid_) {
            if (t && sum_upto > 0) v.push_back({ t, t->nb[1] * (size_t) std::min<int64_t>(sum_upto, t->ne[1]) });
        }
        return v;
    };

    // park the active sequence
    SideSeq a;
    a.n_past     = n_past_;
    a.host_valid = host_valid_;
    a.exact_upto = exact_upto_;
    a.ck_n_past  = ck_n_past_;
    a.far_rows   = far_rows_;
    a.far_ws     = far_ws_;
    a.sum_upto   = sum_upto_;
    {
        const auto ps = parts(a.n_past, a.ck_n_past, a.sum_upto);
        size_t     tot = 0;
        for (auto & q : ps) tot += q.n;
        a.blob.resize(tot);
        size_t off = 0;
        for (auto & q : ps) {
            ggml_backend_tensor_get(q.t, a.blob.data() + off, 0, q.n);
            off += q.n;
        }
    }
    a.slot_pos     = std::move(slot_pos_);
    a.mtp_slot_pos = std::move(mtp_slot_pos_);
    a.kv           = std::move(slots_[0]);

    // resume the parked one (or an empty one)
    SideSeq b  = std::move(side_);
    slots_[0]  = std::move(b.kv);
    hk_        = slots_[0].hk;
    hv_        = slots_[0].hv;
    side_      = std::move(a);
    if (b.n_past <= 0) {
        // (reset() empties the region now active: already empty, or a sequence of no positions)
        reset();
        return true;
    }
    n_past_     = b.n_past;
    host_valid_ = b.host_valid;
    exact_upto_ = b.exact_upto;
    ck_n_past_  = b.ck_n_past;
    far_rows_   = b.far_rows;
    far_ws_     = b.far_ws;
    sum_upto_ = sum_low_ = b.sum_upto;
    {
        const auto ps  = parts(b.n_past, b.ck_n_past, b.sum_upto);
        size_t     off = 0;
        for (auto & q : ps) {
            if (off + q.n > b.blob.size()) {
                err = "swap_side: parked state is short";
                return false;
            }
            ggml_backend_tensor_set(q.t, b.blob.data() + off, 0, q.n);
            off += q.n;
        }
    }
    slot_pos_     = std::move(b.slot_pos);
    mtp_slot_pos_ = std::move(b.mtp_slot_pos);
    if ((int) slot_pos_.size() != W_) slot_pos_.assign((size_t) W_, -1);
    if ((int) mtp_slot_pos_.size() != Wm_) mtp_slot_pos_.assign((size_t) Wm_, -1);
    hid_row_      = -1;
    saved_n_past_ = saved_n_past1_ = 0;
    recorded_n_ = recorded_full_ = 0;
    return true;
}

bool Qwen35::save_prefix(PrefixSnap & s, std::string & err) {
    const int n = n_past_;
    if (!can_snapshot(n)) {
        err = "save_prefix: needs the KV in RAM and the whole prefix in the VRAM window";
        return false;
    }
    if (host_valid_ < n && !flush_ring(host_valid_, n, err)) return false;  // RAM rows [0, n) exact
    s = PrefixSnap{};
    s.n = n;
    for (size_t i = 0; i < hk_.size(); i++) {
        for (ggml_tensor * t : { hk_[i], hv_[i] }) {
            if (!t) continue;
            const size_t bytes = t->nb[1] * (size_t) n, o = s.kv.size();
            s.kv.resize(o + bytes);
            std::memcpy(s.kv.data() + o, t->data, bytes);
        }
    }
    for (size_t il = 0; il < conv_state_.size(); il++) {
        for (ggml_tensor * t : { conv_state_[il], ssm_state_[il] }) {
            if (!t) continue;
            const size_t o = s.rec.size();
            s.rec.resize(o + ggml_nbytes(t));
            ggml_backend_tensor_get(t, s.rec.data() + o, 0, ggml_nbytes(t));
        }
    }
    if (mtp_on_) {
        for (ggml_tensor * t : { mtp_k_, mtp_v_ }) {
            const size_t o = s.mtp.size();
            s.mtp.resize(o + ggml_nbytes(t));
            ggml_backend_tensor_get(t, s.mtp.data() + o, 0, ggml_nbytes(t));
        }
        s.mtp_slot_pos = mtp_slot_pos_;
    }
    return true;
}

bool Qwen35::load_prefix(const PrefixSnap & s, std::string & err) {
    const int n = s.n;
    if (slots_.empty() || W_ >= n_ctx_ || n <= 0 || n > W_ || dry_pending_) {
        err = "load_prefix: needs the KV in RAM and the whole prefix in the VRAM window";
        return false;
    }
    if (!ensure_committed(n, err)) return false;
    const auto & h = hp_;
    // RAM KV rows
    size_t off = 0;
    for (size_t i = 0; i < hk_.size(); i++) {
        for (ggml_tensor * t : { hk_[i], hv_[i] }) {
            if (!t) continue;
            const size_t bytes = t->nb[1] * (size_t) n;
            if (off + bytes > s.kv.size()) {
                err = "load_prefix: snapshot is short";
                return false;
            }
            std::memcpy(t->data, s.kv.data() + off, bytes);
            off += bytes;
        }
    }
    // the VRAM ring from the RAM rows: ring row = the KV heads' rows side by side, position p at ring row Kd_ + p
    std::vector<uint8_t> buf;
    for (int64_t il = 0; il < h.n_layer; il++) {
        if (h.is_recurrent(il)) continue;
        for (int kv = 0; kv < 2; kv++) {
            ggml_tensor * rc = (kv ? v_cache_ : k_cache_)[(size_t) il];
            buf.resize(rc->nb[1] * (size_t) n);
            for (int64_t j = 0; j < h.n_head_kv; j++) {
                const ggml_tensor * ht = (kv ? hv_ : hk_)[(size_t) (il * h.n_head_kv + j)];
                for (int r = 0; r < n; r++) {
                    std::memcpy(buf.data() + rc->nb[1] * (size_t) r + ht->nb[1] * (size_t) j,
                                (const uint8_t *) ht->data + ht->nb[1] * (size_t) r, ht->nb[1]);
                }
            }
            ggml_backend_tensor_set(rc, buf.data(), rc->nb[1] * (size_t) Kd_, buf.size());
        }
    }
    off = 0;
    for (size_t il = 0; il < conv_state_.size(); il++) {
        for (ggml_tensor * t : { conv_state_[il], ssm_state_[il] }) {
            if (!t) continue;
            ggml_backend_tensor_set(t, s.rec.data() + off, 0, ggml_nbytes(t));
            off += ggml_nbytes(t);
        }
    }
    if (mtp_on_ && !s.mtp.empty()) {
        ggml_backend_tensor_set(mtp_k_, s.mtp.data(), 0, ggml_nbytes(mtp_k_));
        ggml_backend_tensor_set(mtp_v_, s.mtp.data() + ggml_nbytes(mtp_k_), 0, ggml_nbytes(mtp_v_));
        mtp_slot_pos_ = s.mtp_slot_pos;
    } else {
        mtp_slot_pos_.assign((size_t) Wm_, -1);
    }
    slot_pos_.assign((size_t) W_, -1);
    for (int p = 0; p < n; p++) slot_pos_[(size_t) p] = p;
    n_past_     = n;
    host_valid_ = exact_upto_ = n;
    sum_upto_ = sum_low_ = 0;  // recomputed from the RAM rows when a sparse eval needs them
    far_rows_ = far_ws_ = 0;
    ck_n_past_  = 0;
    hid_row_    = -1;
    saved_n_past_ = saved_n_past1_ = 0;
    recorded_n_ = recorded_full_ = 0;
    return true;
}

std::vector<std::vector<ggml_fp16_t>> Qwen35::summarize(const std::vector<ggml_tensor *> & hk, int from, int upto) const {
    // element-wise min/max of each page's keys (from the RAM KV), midpoints as halves, heads in parallel
    const auto &  h  = hp_;
    const int64_t hd = h.head_dim, np = std::max(0, upto - from);
    const auto *  tt = ggml_get_type_traits(opt_.kv_type);
    std::vector<size_t> ids;
    for (size_t i = 0; i < hk.size(); i++) if (hk[i]) ids.push_back(i);
    std::vector<std::vector<ggml_fp16_t>> mid(ids.size());
    if (np == 0) return mid;
    std::atomic<size_t> next{ 0 };
    auto work = [&]() {
        std::vector<float> row((size_t) hd), lo((size_t) hd), hi((size_t) hd);
        for (size_t t; (t = next++) < ids.size();) {
            const ggml_tensor * k = hk[ids[t]];
            mid[t].resize((size_t) (np * hd));
            for (int64_t p = 0; p < np; p++) {
                std::fill(lo.begin(), lo.end(), INFINITY);
                std::fill(hi.begin(), hi.end(), -INFINITY);
                for (int r = 0; r < kPage; r++) {
                    const char * src = (const char *) k->data + k->nb[1] * (size_t) ((from + p) * kPage + r);
                    if (k->type == GGML_TYPE_F32) std::memcpy(row.data(), src, sizeof(float) * hd);
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
    return mid;
}

void Qwen35::update_summaries(int upto) {
    // the active slot's new pages, uploaded to the VRAM summaries in order
    if (upto <= sum_upto_) return;
    const auto   mid = summarize(hk_, sum_upto_, upto);
    const size_t off = sizeof(ggml_fp16_t) * (size_t) (sum_upto_ * hp_.head_dim);
    size_t       t   = 0;
    for (size_t i = 0; i < pmid_.size(); i++) {
        if (!pmid_[i]) continue;
        ggml_backend_tensor_set(pmid_[i], mid[t].data(), off, mid[t].size() * sizeof(ggml_fp16_t));
        t++;
    }
    sum_upto_ = upto;
}

int Qwen35::fill_far(const std::vector<ggml_tensor *> & kc, const std::vector<ggml_tensor *> & vc,
                     const std::vector<ggml_tensor *> & hk, const std::vector<ggml_tensor *> & hv,
                     const std::vector<ggml_tensor *> & sel_nodes, int kp) {
    // after a sparse verify: copy its best-scoring pages (per KV head) from RAM into the ring's draft far area
    const auto &  h   = hp_;
    const int64_t nkv = h.n_head_kv;
    const int     np  = std::min(Kd_ / kPage, kp);
    if (np <= 0 || sel_nodes.empty()) return 0;
    std::vector<int32_t> sel((size_t) kp);
    std::vector<uint8_t> buf;
    for (int64_t il = 0; il < h.n_layer; il++) {
        if (h.is_recurrent(il)) continue;
        for (int kv = 0; kv < 2; kv++) {
            ggml_tensor * rc = (kv ? vc : kc)[(size_t) il];
            buf.assign(rc->nb[1] * (size_t) Kd_, 0);
            for (int64_t j = 0; j < nkv; j++) {
                const size_t hi = (size_t) (il * nkv + j);
                ggml_backend_tensor_get(sel_nodes[hi], sel.data(), 0, sel.size() * sizeof(int32_t));
                const ggml_tensor * ht = (kv ? hv : hk)[hi];
                for (int p = 0; p < np; p++) {
                    for (int r = 0; r < kPage; r++) {
                        std::memcpy(buf.data() + rc->nb[1] * (size_t) (p * kPage + r) + ht->nb[1] * (size_t) j,
                                    (const uint8_t *) ht->data + ht->nb[1] * (size_t) (sel[(size_t) p] * kPage + r), ht->nb[1]);
                    }
                }
            }
            ggml_backend_tensor_set(rc, buf.data(), 0, buf.size());
        }
    }
    return np * kPage;
}

void Qwen35::fill_draft_far() {
    const int rows = fill_far(k_cache_, v_cache_, hk_, hv_, sel_nodes_, sp_kp_);
    if (rows <= 0) return;
    far_rows_ = rows;
    far_ws_   = sp_ws_;
}

namespace {
// the top k of each row of logits [n_vocab, rows], best first: values [1, k, rows], ids [k, rows]. In chunks of 32
// rows: the full-vocabulary argsort takes VRAM per row (a joint verify of 126 rows ran out at 128K context)
void rows_top_k(ggml_context * ctx, ggml_tensor * lg, int k, ggml_tensor *& vals, ggml_tensor *& ids) {
    const int64_t     nv = lg->ne[0], rows = lg->ne[1];
    constexpr int64_t kCh = 32;
    vals = ids = nullptr;
    for (int64_t r0 = 0; r0 < rows; r0 += kCh) {
        const int64_t r   = std::min(kCh, rows - r0);
        ggml_tensor * sub = ggml_view_2d(ctx, lg, nv, r, lg->nb[1], lg->nb[1] * (size_t) r0);
        ggml_tensor * id  = ggml_cont(ctx, ggml_argsort_top_k(ctx, sub, k));
        ggml_tensor * v   = ggml_get_rows(ctx, ggml_reshape_3d(ctx, sub, 1, nv, r), id);
        ids  = ids ? ggml_concat(ctx, ids, id, 1) : id;
        vals = vals ? ggml_concat(ctx, vals, v, 2) : v;
    }
}

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
    if (!ensure_committed(to, err)) return false;
    const auto &         h   = hp_;
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
                                    buf.data() + rc->nb[1] * (size_t) r + ht->nb[1] * (size_t) j, ht->nb[1]);
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
    // (files packed before the rename carry eightfer.* keys)
    const std::string id_b = file_.str("shoehorn.pack.id", file_.str("eightfer.pack.id", "")),
                      id_r = rf.str("shoehorn.pack.id", rf.str("eightfer.pack.id", ""));
    const std::string rtype = rf.str("general.type", "");
    if ((rtype != "shoehorn-residual" && rtype != "eightfer-residual") || id_r.empty()) {
        err = path + " is not a shoehorn residual file";
        return false;
    }
    if (id_b != id_r) {
        err = "residual " + path + " was packed against a different base (pack id " + id_r + ", base has \"" + id_b + "\")";
        return false;
    }
    ggml_init_params ip = { ggml_tensor_overhead() * (size_t) (rf.n_tensors() + 64), nullptr, true };
    rctx_               = ggml_init(ip);
    // VRAM budget for residual tensors: those go in VRAM in file order (layer by layer) while they fit, and a verify
    // reads them there instead of over PCIe
    size_t gpu_budget = 0;
    if (gpu_ && opt_.res_gpu_gb != 0) {
        if (opt_.res_gpu_gb > 0) {
            gpu_budget = (size_t) (opt_.res_gpu_gb * 1e9);
        } else {
            size_t vfree = 0, vtotal = 0;
            ggml_backend_dev_memory(ggml_backend_get_device(gpu_), &vfree, &vtotal);
            const size_t reserve = (size_t) (opt_.res_gpu_reserve_gb * 1e9);
            gpu_budget           = vfree > reserve ? vfree - reserve : 0;
        }
        if (gpu_budget > 0) rgctx_ = ggml_init(ip);
    }
    size_t gpu_used = 0;
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
        const size_t  nb     = ggml_nbytes(m);
        const bool    on_gpu = rgctx_ && gpu_used + nb + 256 <= gpu_budget;
        if (on_gpu) gpu_used += GGML_PAD(nb, 256);
        ggml_tensor * r = ggml_dup_tensor(on_gpu ? rgctx_ : rctx_, m);
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
    if (rgctx_ && gpu_used > 0) {
        rgbuf_ = ggml_backend_alloc_ctx_tensors_from_buft(rgctx_, ggml_backend_get_default_buffer_type(gpu_));
        if (!rgbuf_) {
            err = "not enough VRAM for " + std::to_string(gpu_used / 1e9).substr(0, 5) + " GB of residual (lower --res-gpu-gb)";
            return false;
        }
        ggml_backend_buffer_set_usage(rgbuf_, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        res_gpu_bytes_ = ggml_backend_buffer_get_size(rgbuf_);
    }
    const bool any_host = ggml_get_first_tensor(rctx_) != nullptr;
    if (!any_host) buft = nullptr;
    // E8_ZERO_COPY=1: the residual sits in mapped pinned RAM behind a CUDA buffer, so GPU kernels read it over PCIe
    // in place (no VRAM staging, no graph splits around it); see docs/SPEEDUP-PLAN.md idea 1
    static const bool zero_copy = std::getenv("E8_ZERO_COPY") && std::atoi(std::getenv("E8_ZERO_COPY")) != 0;
    if (buft && zero_copy) {
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(gpu_));
        auto set_mapped = (void (*)(bool)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_e8_alloc_mapped");
        if (set_mapped) {
            set_mapped(true);
            rbuf_ = ggml_backend_alloc_ctx_tensors_from_buft(rctx_, ggml_backend_get_default_buffer_type(gpu_));
            set_mapped(false);
        }
        if (rbuf_) fprintf(stderr, "residual: zero-copy (mapped pinned RAM read by GPU kernels)\n");
    }
    if (buft && !rbuf_) {
        rbuf_ = ggml_backend_alloc_ctx_tensors_from_buft(rctx_, buft);
        if (!rbuf_) {
            fprintf(stderr, "warning: pinned allocation for the residual failed, using pageable RAM\n");
        }
    }
    if (!rbuf_ && any_host) {
        rbuf_ = ggml_backend_alloc_ctx_tensors_from_buft(rctx_, ggml_backend_get_default_buffer_type(cpu_));
        if (!rbuf_) {
            err = "not enough RAM for the residual";
            return false;
        }
    }
    if (rbuf_) ggml_backend_buffer_set_usage(rbuf_, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    res_bytes_ = (rbuf_ ? ggml_backend_buffer_get_size(rbuf_) : 0) + res_gpu_bytes_;
    traffic::res_vram_bytes = res_gpu_bytes_;
    traffic::res_ram_bytes  = res_bytes_ - res_gpu_bytes_;
    if (res_gpu_bytes_) {
        fprintf(stderr, "residual: %.2f GB in VRAM, %.2f GB in RAM\n", res_gpu_bytes_ / 1e9, (res_bytes_ - res_gpu_bytes_) / 1e9);
    }
    if (!read_tensors(path, rf, to_load, err)) return false;
    // A big batch runs a host weight on the GPU by staging the whole tensor in VRAM. The output head's residual (248K
    // rows, 0.7 GB) would set the verify's VRAM need; as tensors over its row pieces (same memory) it needs a tenth.
    for (auto & [base, r] : res_) {
        if (r->ne[1] <= kResChunkRows || ggml_n_dims(r) != 2 || r->buffer != rbuf_) continue;
        auto & parts = res_chunks_[r];
        for (int64_t r0 = 0; r0 < r->ne[1]; r0 += kResChunkRows) {
            ggml_tensor * pt = ggml_new_tensor_2d(rctx_, r->type, r->ne[0], std::min<int64_t>(kResChunkRows, r->ne[1] - r0));
            if (ggml_backend_tensor_alloc(rbuf_, pt, (char *) r->data + r->nb[1] * (size_t) r0) != GGML_STATUS_SUCCESS) {
                err = "cannot place a residual row piece";
                return false;
            }
            parts.push_back(pt);
        }
    }
    return true;
}

ggml_tensor * Qwen35::mm(ggml_context * ctx, ggml_tensor * w, ggml_tensor * x) {
    ggml_tensor * y = ggml_mul_mat(ctx, w, x);
    if (use_res_) {
        auto it = res_.find(w);
        if (it != res_.end()) {
            // small batches (decode, speculative verify): one-pass multi-column Q4_K kernel on the CPU; big batches:
            // ggml mul_mat, which the scheduler streams to the GPU
            // (with the GPU staging host weights on its own stream, see patches/ggml-weight-prefetch.patch, a GPU run
            // costs one ~250 ms transfer at any batch size, so every residual product goes there: res_on_gpu_)
            static const bool small_ok = std::getenv("E8_NO_SMALL_GEMM") == nullptr;
            ggml_tensor *     r        = it->second;
            if (small_ok && !res_gpu_all() && r->buffer == rbuf_ && kernels::q4k_small_supported(r, x)) {
                y = ggml_add(ctx, y, kernels::q4k_mul_mat_small(ctx, r, x));
            } else if (auto ch = res_chunks_.find(r); ch != res_chunks_.end()) {
                // row pieces of a big residual (see load_residual), each staged in VRAM on its own
                ggml_tensor * acc = nullptr;
                for (ggml_tensor * part : ch->second) {
                    ggml_tensor * pm = ggml_mul_mat(ctx, part, x);
                    if (res_gpu_all()) res_on_gpu_.push_back(pm);
                    acc = acc ? ggml_concat(ctx, acc, pm, 0) : pm;
                }
                y = ggml_add(ctx, y, acc);
            } else {
                ggml_tensor * pm = ggml_mul_mat(ctx, r, x);
                if (res_gpu_all()) res_on_gpu_.push_back(pm);
                y = ggml_add(ctx, y, pm);
            }
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
    for (ggml_tensor * t : gpu_nodes_) {
        if (gpu_) ggml_backend_sched_set_tensor_backend(sched_, t, gpu_);
    }
    gpu_nodes_.clear();
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
    if (recorded_full_ <= 0 || keep < 0 || keep > recorded_full_) {
        err = "rollback: no recorded eval or keep out of range";
        return false;
    }
    const int n = recorded_full_, nr = recorded_n_;
    recorded_n_ = recorded_full_ = 0;
    if (keep == n) {
        return true;
    }
    if (keep > nr) {
        err = "rollback: past the recorded tokens";
        return false;
    }
    restore_state(slot);
    if (!replay(keep, err)) return false;
    n_past_  = (slot ? saved_n_past1_ : saved_n_past_) + keep;
    hid_row_ = keep - 1;
    return true;
}

bool Qwen35::replay(int keep, std::string & err, int row0, const std::vector<ggml_tensor *> * cs,
                    const std::vector<ggml_tensor *> * ss) {
    if (keep <= 0) return true;
    const auto &     h   = hp_;
    ggml_init_params ip  = { graph_meta_.size(), graph_meta_.data(), true };
    ggml_context *   ctx = ggml_init(ip);
    ggml_cgraph *    gf  = ggml_new_graph_custom(ctx, kGraphSize, false);
    gpu_nodes_.clear();
    for (int64_t il = 0; il < h.n_layer; il++) {
        if (!h.is_recurrent(il)) {
            continue;
        }
        const size_t  l    = (size_t) il;
        ggml_tensor * qkv  = ggml_view_2d(ctx, rec_qkv_[l], h.conv_channels(), keep, rec_qkv_[l]->nb[1], rec_qkv_[l]->nb[1] * (size_t) row0);
        ggml_tensor * g    = ggml_view_2d(ctx, rec_g_[l], h.ssm_n_v, keep, rec_g_[l]->nb[1], rec_g_[l]->nb[1] * (size_t) row0);
        ggml_tensor * beta = ggml_view_2d(ctx, rec_beta_[l], h.ssm_n_v, keep, rec_beta_[l]->nb[1], rec_beta_[l]->nb[1] * (size_t) row0);
        gdn_core(ctx, gf, il, ggml_reshape_3d(ctx, qkv, h.conv_channels(), keep, 1),
                 ggml_reshape_4d(ctx, g, 1, h.ssm_n_v, keep, 1), ggml_reshape_4d(ctx, beta, 1, h.ssm_n_v, keep, 1), keep, 1,
                 cs ? (*cs)[l] : nullptr, ss ? (*ss)[l] : nullptr);
    }
    const bool ok = compute(ctx, gf, err);
    ggml_free(ctx);
    if (!ok) err = "replay: " + err;
    return ok;
}

bool Qwen35::commit(int keep, std::string & err, int seq) {
    const int L = recorded_full_ / std::max(1, rec_seqs_), row0 = seq * L;
    if (!dry_pending_ || keep < 0 || seq < 0 || seq >= rec_seqs_ || keep > L || row0 + keep > recorded_n_) {
        err = "commit: no dry recorded eval or keep past the recorded tokens";
        return false;
    }
    dry_pending_ = false;
    recorded_n_ = recorded_full_ = 0;
    rec_seqs_   = 1;
    if (!replay(keep, err, row0)) return false;
    if (row0 > 0 && keep > 0) {
        // tree verify: the kept sequence's K/V sit at ring slots n_past + row0 ..; move them to n_past ..
        ggml_init_params ip  = { graph_meta_.size(), graph_meta_.data(), true };
        ggml_context *   ctx = ggml_init(ip);
        ggml_cgraph *    gf  = ggml_new_graph_custom(ctx, kGraphSize, false);
        for (size_t il = 0; il < k_cache_.size(); il++) {
            for (ggml_tensor * c : { k_cache_[il], v_cache_[il] }) {
                if (!c) continue;
                const size_t s = c->nb[1] * (size_t) (Kd_ + n_past_ + row0), d = c->nb[1] * (size_t) (Kd_ + n_past_);
                ggml_build_forward_expand(gf, ggml_cpy(ctx, ggml_view_2d(ctx, c, c->ne[0], keep, c->nb[1], s),
                                                       ggml_view_2d(ctx, c, c->ne[0], keep, c->nb[1], d)));
            }
        }
        const bool ok = compute(ctx, gf, err);
        ggml_free(ctx);
        if (!ok) return false;
    }
    n_past_ += keep;
    hid_row_ = row0 + keep - 1;
    return true;
}

ggml_tensor * Qwen35::gdn_core(ggml_context * ctx, ggml_cgraph * gf, int64_t il, ggml_tensor * qkv, ggml_tensor * g,
                               ggml_tensor * beta, int n, int ns, ggml_tensor * cs_in, ggml_tensor * ss_in) {
    // ns > 1: n tokens are ns sequences of n / ns, each from the same state (state not written)
    // cs_in / ss_in: this state instead of the active one (a parked slot's, in RAM): the ops that read it are placed
    // on the GPU (gpu_nodes_), the scheduler copies it there
    const int L = n / ns;
    const auto &        h   = hp_;
    const Qwen35Layer & Lw  = layers_[(size_t) il];
    const float         eps = h.rms_eps;
    const int64_t       S = h.ssm_d_state, Hk = h.ssm_n_k, Hv = h.ssm_n_v, C = h.conv_channels();

    // causal conv over [previous d_conv-1 inputs | this batch], then keep the last d_conv-1 inputs
    ggml_tensor * cs   = cs_in ? cs_in : conv_state_[(size_t) il];
    ggml_tensor * cst  = ggml_reshape_3d(ctx, cs, h.ssm_d_conv - 1, C, 1);
    if (ns > 1) {
        cst = ggml_repeat_4d(ctx, cst, h.ssm_d_conv - 1, C, ns, 1);
        qkv = ggml_reshape_3d(ctx, qkv, C, L, ns);
    }
    ggml_tensor * cin  = ggml_concat(ctx, cst, ggml_transpose(ctx, qkv), 0);  // [d_conv-1+L, C, ns]
    if (cs_in) gpu_nodes_.push_back(cin);
    ggml_tensor * last = ggml_view_3d(ctx, cin, h.ssm_d_conv - 1, C, 1, cin->nb[1], cin->nb[2],
                                      ggml_row_size(cin->type, cin->ne[0] - (h.ssm_d_conv - 1)));
    if (write_state_ && ns == 1) ggml_build_forward_expand(gf, ggml_cpy(ctx, last, cs));

    ggml_tensor * conv = ggml_silu(ctx, ggml_ssm_conv(ctx, cin, Lw.conv1d));  // [C, L, ns]
    const size_t  nb1  = ggml_row_size(conv->type, C);
    ggml_tensor * q = ggml_view_4d(ctx, conv, S, Hk, L, ns, ggml_row_size(conv->type, S), nb1, nb1 * L, 0);
    ggml_tensor * k = ggml_view_4d(ctx, conv, S, Hk, L, ns, ggml_row_size(conv->type, S), nb1, nb1 * L,
                                   ggml_row_size(conv->type, S * Hk));
    ggml_tensor * v = ggml_view_4d(ctx, conv, S, Hv, L, ns, ggml_row_size(conv->type, S), nb1, nb1 * L,
                                   ggml_row_size(conv->type, 2 * S * Hk));
    // l2 norm = rms_norm(x, eps/n) / sqrt(n)  (llama.cpp build_gdn_l2_norm)
    q = ggml_scale(ctx, ggml_rms_norm(ctx, q, eps / (float) S), 1.0f / sqrtf((float) S));
    k = ggml_scale(ctx, ggml_rms_norm(ctx, k, eps / (float) S), 1.0f / sqrtf((float) S));

    ggml_tensor * ss    = ss_in ? ss_in : ssm_state_[(size_t) il];
    ggml_tensor * state = ggml_reshape_4d(ctx, ss, S, S, Hv, 1);
    if (ns > 1) {
        state = ggml_repeat_4d(ctx, state, S, S, Hv, ns);
        g     = ggml_reshape_4d(ctx, g, 1, Hv, L, ns);
        beta  = ggml_reshape_4d(ctx, beta, 1, Hv, L, ns);
    }
    ggml_tensor * res   = ggml_gated_delta_net(ctx, q, k, v, g, beta, state, 1);
    if (ss_in) gpu_nodes_.push_back(res);
    ggml_tensor * out   = ggml_view_4d(ctx, res, S, Hv, L, ns, ggml_row_size(res->type, S),
                                       ggml_row_size(res->type, S * Hv), ggml_row_size(res->type, S * Hv * L), 0);
    if (ns > 1) return ggml_reshape_4d(ctx, out, S, Hv, n, 1);
    ggml_tensor * new_state = ggml_view_1d(ctx, res, S * S * Hv, ggml_row_size(res->type, S * Hv * n));
    if (write_state_) ggml_build_forward_expand(gf, ggml_cpy(ctx, new_state, ss));
    return out;
}

ggml_cgraph * Qwen35::build_graph(ggml_context * ctx, int n, const EvalOpts & o, ggml_tensor *& inp_tok,
                                  ggml_tensor *& inp_pos, ggml_tensor *& inp_mask, ggml_tensor *& out_logits, int n_kv) {
    const auto &  h   = hp_;
    ggml_cgraph * gf  = ggml_new_graph_custom(ctx, kGraphSize, false);
    const float   eps = h.rms_eps;
    res_on_gpu_.clear();
    gpu_nodes_.clear();
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
            gi_.reserve((size_t) (h.n_layer * h.n_head_kv * 2 * ((n + kSparseMaxN - 1) / kSparseMaxN)));  // stable addresses: the gather ops point at them
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
            if (o.record) {  // the first max_record tokens (a bigger batch can only be rolled back within them)
                const size_t  l  = (size_t) il;
                const int64_t nr = std::min<int64_t>(n, opt_.max_record);
                ggml_tensor * g2 = ggml_reshape_2d(ctx, g, Hv, n);
                ggml_build_forward_expand(gf, ggml_cpy(ctx, ggml_view_2d(ctx, qkv, C, nr, qkv->nb[1], 0),
                                                       ggml_view_2d(ctx, rec_qkv_[l], C, nr, rec_qkv_[l]->nb[1], 0)));
                ggml_build_forward_expand(gf, ggml_cpy(ctx, ggml_view_2d(ctx, g2, Hv, nr, g2->nb[1], 0),
                                                       ggml_view_2d(ctx, rec_g_[l], Hv, nr, rec_g_[l]->nb[1], 0)));
                ggml_build_forward_expand(gf, ggml_cpy(ctx, ggml_view_2d(ctx, beta, Hv, nr, beta->nb[1], 0),
                                                       ggml_view_2d(ctx, rec_beta_[l], Hv, nr, rec_beta_[l]->nb[1], 0)));
            }
            ggml_tensor * out = gdn_core(ctx, gf, il, ggml_reshape_3d(ctx, qkv, C, n, 1), ggml_reshape_4d(ctx, g, 1, Hv, n, 1),
                                         ggml_reshape_4d(ctx, beta, 1, Hv, n, 1), n, o.n_seqs);

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
                std::vector<ggml_tensor *> wk((size_t) nkv), wv((size_t) nkv);
                for (int64_t j = 0; j < nkv; j++) {  // write this batch to the RAM KV
                    const size_t  hi = (size_t) (il * nkv + j);
                    ggml_tensor * kj = ggml_cont(ctx, ggml_view_2d(ctx, Kc, hd, n, Kc->nb[2], Kc->nb[1] * (size_t) j));
                    ggml_tensor * vj = ggml_cont(ctx, ggml_view_2d(ctx, Vc, hd, n, Vc->nb[2], Vc->nb[1] * (size_t) j));
                    wk[(size_t) j]   = ggml_set_rows(ctx, hk_[hi], kj, inp_hidx_);
                    wv[(size_t) j]   = ggml_set_rows(ctx, hv_[hi], vj, inp_hidx_);
                    ggml_build_forward_expand(gf, wk[(size_t) j]);
                    ggml_build_forward_expand(gf, wv[(size_t) j]);
                }
                // queries in sub-chunks of kSparseMaxN, each with its own page selection, so a big batch (one pass
                // over the weights) attends like consecutive small batches; outputs concatenated in token order
                for (int64_t c0 = 0; c0 < n; c0 += kSparseMaxN) {
                    // per sub-chunk: all heads' page selections (GPU), then all their gathers (CPU), then attention
                    // (GPU), so a layer switches between the GPU and the CPU twice per sub-chunk, not twice per head
                    const int64_t m = std::min<int64_t>(kSparseMaxN, n - c0);
                    std::vector<ggml_tensor *> qc((size_t) nkv), sl((size_t) nkv, nullptr), gk((size_t) nkv), gv((size_t) nkv);
                    for (int64_t j = 0; j < nkv; j++) {
                        const size_t hi  = (size_t) (il * nkv + j);
                        qc[(size_t) j]   = ggml_view_3d(ctx, q, hd, m, g, q->nb[1], q->nb[2],
                                                        q->nb[2] * (size_t) (j * g) + q->nb[1] * (size_t) c0);
                        if (sp_kp_ > 0) {  // page selection on the GPU: sum over the sub-chunk's queries
                            ggml_tensor * q2 = ggml_reshape_2d(ctx, ggml_cont(ctx, qc[(size_t) j]), hd, m * g);
                            ggml_tensor * md = ggml_view_2d(ctx, pmid_[hi], hd, sp_nfar_, pmid_[hi]->nb[1], 0);
                            ggml_tensor * s  = ggml_mul_mat(ctx, md, q2);  // q . page midpoint: [nfar, m*g]
                            if (!slots_.empty()) gpu_nodes_.push_back(s);       // (slots: summaries from RAM)
                            // per query: a distribution over pages (each query counts equally, whatever its scale),
                            // then summed over the queries. (Midpoints beat the q.k upper bound from min/max: on text
                            // repeated 48K tokens back, PPL 1.0013 vs 1.069; exact 1.0004.)
                            s  = ggml_soft_max_ext(ctx, s, nullptr, 1.0f / sqrtf((float) hd), 0.0f);
                            s  = ggml_sum_rows(ctx, ggml_cont(ctx, ggml_transpose(ctx, s)));  // [1, nfar]
                            s  = ggml_add(ctx, ggml_reshape_2d(ctx, s, sp_nfar_, 1), inp_sbias_);
                            ggml_tensor * t = ggml_cont(ctx, ggml_argsort_top_k(ctx, s, sp_kp_));  // best first
                            ggml_set_output(t);
                            ggml_build_forward_expand(gf, t);
                            if (sel_nodes_.size() < hk_.size()) sel_nodes_.resize(hk_.size(), nullptr);
                            sel_nodes_[hi]  = t;  // the last sub-chunk's pages feed the drafts' far area
                            sl[(size_t) j] = t;
                        }
                    }
                    for (int64_t j = 0; j < nkv; j++) {  // gather the selected pages and the window from RAM
                        const size_t hi = (size_t) (il * nkv + j);
                        for (int kv = 0; kv < 2; kv++) {
                            gi_.push_back({ kv ? hv_[hi] : hk_[hi], (int) kr, kPage, sp_ws_, sp_nwin_ });
                            ggml_tensor * dep     = kv ? wv[(size_t) j] : wk[(size_t) j];
                            ggml_tensor * args[2] = { sl[(size_t) j] ? sl[(size_t) j] : dep, dep };
                            ggml_tensor * t       = ggml_custom_4d(ctx, kv ? opt_.kv_type_v : opt_.kv_type, hd, sp_nsel_, 1, 1, args, 2, gather_op,
                                                                   GGML_N_TASKS_MAX, &gi_.back());
                            ggml_build_forward_expand(gf, t);
                            (kv ? gv : gk)[(size_t) j] = t;
                        }
                    }
                    ggml_tensor * mk = ggml_view_2d(ctx, inp_smask_, sp_nsel_, m, inp_smask_->nb[1],
                                                    inp_smask_->nb[1] * (size_t) c0);
                    ggml_tensor * ac = nullptr;
                    for (int64_t j = 0; j < nkv; j++) {
                        ggml_tensor * aj = ggml_flash_attn_ext(ctx, qc[(size_t) j], gk[(size_t) j], gv[(size_t) j], mk,
                                                               1.0f / sqrtf((float) hd), 0.0f, 0.0f);
                        ggml_prec_set_acc(aj, GGML_PREC_F32);
                        attn_nodes_.push_back(aj);
                        ac = ac ? ggml_concat(ctx, ac, aj, 1) : aj;  // [hd, heads, m]
                    }
                    ggml_build_forward_expand(gf, ac);
                    a = a ? ggml_concat(ctx, a, ac, 2) : ac;  // [hd, heads, n]
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
        const int nm = n / o.n_seqs;  // tree verify: the first sequence's positions only
        if (nm > 1) {  // MTP K/V of positions [0, n-1): their next token is this batch's next token
            inp_mpos_ = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int64_t) (nm - 1) * 4);
            ggml_set_input(inp_mpos_);
            inp_mkvidx_ = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, nm - 1);
            ggml_set_input(inp_mkvidx_);
            ggml_tensor * e  = ggml_view_2d(ctx, inp_tok, h.n_embd, nm - 1, inp_tok->nb[1], inp_tok->nb[1]);
            ggml_tensor * hh = ggml_view_2d(ctx, hall, h.n_embd, nm - 1, hall->nb[1], 0);
            mtp_layer(ctx, gf, mtp_input(ctx, e, hh), nm - 1, inp_mpos_, inp_mkvidx_, nullptr);
        }
        cur = o.last_only && n > 1 ? ggml_view_2d(ctx, hall, hall->ne[0], 1, hall->nb[1], hall->nb[1] * (size_t) (n - 1)) : hall;
    } else {
        if (o.last_only && n > 1) {
            inpL = ggml_view_2d(ctx, inpL, inpL->ne[0], 1, inpL->nb[1], inpL->nb[1] * (size_t) (n - 1));
        }
        cur = norm(inpL, out_norm_);
    }
    out_logits = mm(ctx, output_, cur);
    out_ids_   = nullptr;
    if (o.argmax) {
        out_logits = ggml_argmax(ctx, out_logits);
    } else if (o.topk > 0) {  // the top-k of each row on the GPU: k values + ids come back instead of the vocabulary
        rows_top_k(ctx, out_logits, o.topk, out_logits, out_ids_);  // [k, rows], best first
        ggml_set_output(out_ids_);
        ggml_build_forward_expand(gf, out_ids_);
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

bool Qwen35::mtp_step(int32_t tok, int pos, int hid_row, float * logits, int32_t * id, std::string & err, int topk) {
    const auto & h = hp_;
    if (!mtp_on_ || tok < 0 || tok >= h.n_vocab || pos < 0 || (hid_row >= 0 && hid_row >= opt_.n_ubatch)) {
        err = "mtp_step: no MTP block or bad arguments";
        return false;
    }
    ggml_init_params ip  = { graph_meta_mtp_.size(), graph_meta_mtp_.data(), true };
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
    ggml_tensor * out  = ggml_mul_mat(ctx, output_, hn);
    ggml_tensor * oids = nullptr;
    if (!logits) {
        out = ggml_argmax(ctx, out);
    } else if (topk > 0) {
        oids = ggml_cont(ctx, ggml_argsort_top_k(ctx, out, topk));
        out  = ggml_get_rows(ctx, ggml_reshape_3d(ctx, out, 1, out->ne[0], 1), oids);
        ggml_set_output(oids);
        ggml_build_forward_expand(gf, oids);
    }
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
    if (ok) traffic::mtp();
    if (ok) {
        if (logits) ggml_backend_tensor_get(out, logits, 0, ggml_nbytes(out));
        if (oids && id) ggml_backend_tensor_get(oids, id, 0, ggml_nbytes(oids));
        else if (!logits && id) ggml_backend_tensor_get(out, id, 0, sizeof(int32_t));
    }
    ggml_free(ctx);
    if (!ok) err = "mtp_step: graph compute failed";
    return ok;
}

bool Qwen35::eval(const int32_t * tokens, int n, const EvalOpts & opts, float * logits, int32_t * ids,
                  std::string & err) {
    if (!multi_.empty()) {
        err = "eval: a joint verify is not committed (commit_seq)";
        return false;
    }
    if (!slots_.empty() && cur_slot_ < 0) {
        err = "eval: no slot selected (detached)";
        return false;
    }
    if (n <= 0 || n > opt_.n_ubatch) {
        err = "eval: batch size must be 1.." + std::to_string(opt_.n_ubatch);
        return false;
    }
    if (n_past_ + n > n_ctx_) {
        err = "eval: context full (" + std::to_string(n_ctx_) + " tokens)";
        return false;
    }
    const int ns = std::max(1, opts.n_seqs), L = n / ns;
    if (ns > 1 && (n % ns != 0 || !opts.dry || !opts.record || opts.last_only || !can_tree(n))) {
        err = "eval: a tree verify needs dry + record, equal sequences and room in the VRAM ring";
        return false;
    }
    // With the KV in RAM, a batch past the VRAM ring attends exactly over the whole RAM KV, which needs per-head
    // copies of all of it in VRAM; in pieces of kSparseMaxN it takes the sparse path instead (same quality at batch
    // 16-128: 32K text PPL 2.2903-2.2920 vs 2.2937 exact), so that VRAM is not reserved.
    static const bool sparse_env = !(std::getenv("E8_SPARSE") && std::atoi(std::getenv("E8_SPARSE")) == 0);
    if (W_ < n_ctx_ && sparse_env && !pmid_.empty() && !opts.window_ok && !opts.record && n > kSparseBatch &&
        n_past_ + n > W_) {
        for (int i = 0; i < n; i += kSparseBatch) {
            const int m    = std::min(kSparseBatch, n - i);
            const bool fin = i + m == n;
            float *   lo   = opts.last_only ? (fin ? logits : nullptr) : (logits ? logits + (size_t) i * hp_.n_vocab : nullptr);
            int32_t * io   = opts.last_only ? (fin ? ids : nullptr) : (ids ? ids + i : nullptr);
            if (!eval(tokens + i, m, opts, lo, io, err)) return false;
        }
        return true;
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
    sum_low_   = std::min(sum_low_, sum_upto_);
    static const bool sparse_on = !(std::getenv("E8_SPARSE") && std::atoi(std::getenv("E8_SPARSE")) == 0);
    sparse_ = use_host && n <= kSparseBatch && sparse_on && !pmid_.empty();
    // the active slot needs memory for the rows this eval writes (exact attention without sparse reads them all)
    if (use_host && !ensure_committed(sparse_ ? n_past_ + n : n_ctx_, err)) return false;
    sel_nodes_.clear();
    if (sparse_) {
        static const int win   = std::getenv("E8_SPARSE_WINDOW") ? std::atoi(std::getenv("E8_SPARSE_WINDOW")) : 4096;
        static const int pages = std::getenv("E8_SPARSE_PAGES") ? std::atoi(std::getenv("E8_SPARSE_PAGES")) : 128;
        // every sub-chunk sees at least `win` tokens back exactly
        sp_ws_   = std::max(0, (n_past_ + n - (std::max(win, n) + std::max(0, n - kSparseMaxN))) / kPage * kPage);
        sp_nfar_ = sp_ws_ / kPage;
        sp_kp_   = std::min(pages, sp_nfar_);
        sp_nwin_ = n_past_ + n - sp_ws_;
        sp_nsel_ = (sp_kp_ * kPage + sp_nwin_ + kKvPad - 1) / kKvPad * kKvPad;
        update_summaries(sp_nfar_);
    }

    use_res_             = opts.residual && has_residual();
    write_state_         = !(opts.dry && opts.record);
    // ggml-cuda caches a captured CUDA graph per address of the graph's first node: the hot single-token draft pass
    // gets its own scratch buffer so it keeps its cached graph while other shapes (checks, rollbacks) run in between
    // small base-only batches (draft checks, replays) get one buffer per (size, dry) for the same reason
    std::vector<uint8_t> * mp = &graph_meta_;
    if (n == 1 && !use_res_) mp = &graph_meta_draft_;
    else if (!use_res_ && n <= 16 && !std::getenv("E8_NO_SMALL_META")) {
        auto & v = graph_meta_small_[n * 2 + (write_state_ ? 0 : 1)];
        if (v.empty()) v.resize(graph_meta_.size());
        mp = &v;
    }
    auto &           meta = *mp;
    ggml_init_params ip  = { meta.size(), meta.data(), true };
    ggml_context *   ctx = ggml_init(ip);
    ggml_tensor *    inp_tok = nullptr, * inp_pos = nullptr, * inp_mask = nullptr, * out = nullptr;
    ggml_cgraph *    gf      = build_graph(ctx, n, opts, inp_tok, inp_pos, inp_mask, out, n_kv);
    const bool       with_res = use_res_;
    use_res_                 = false;
    write_state_             = true;

    ggml_backend_sched_reset(sched_);
    for (ggml_tensor * t : res_on_gpu_) ggml_backend_sched_set_tensor_backend(sched_, t, gpu_);
    res_on_gpu_.clear();
    for (ggml_tensor * t : gpu_nodes_) ggml_backend_sched_set_tensor_backend(sched_, t, gpu_);
    gpu_nodes_.clear();
    for (ggml_tensor * a : attn_nodes_) {
        ggml_backend_sched_set_tensor_backend(sched_, a, gpu_ && n >= host_gpu_min_ ? gpu_ : cpu_);
    }
    if (!ggml_backend_sched_alloc_graph(sched_, gf)) {
        ggml_free(ctx);
        err = "eval: cannot allocate the compute graph";
        return false;
    }
    if (std::getenv("E8_SCHED_DEBUG")) {
        size_t fr = 0, tot = 0;
        if (gpu_) ggml_backend_dev_memory(ggml_backend_get_device(gpu_), &fr, &tot);
        fprintf(stderr, "eval n=%d res=%d: GPU compute buffer %.3f GB, %d splits, %.2f GB free\n", n, (int) opts.residual,
                gpu_ ? ggml_backend_sched_get_buffer_size(sched_, gpu_) / 1e9 : 0.0, ggml_backend_sched_get_n_splits(sched_), fr / 1e9);
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
        pos[(size_t) i] = pos[(size_t) n + i] = pos[(size_t) 2 * n + i] = n_past_ + i % L;
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
        } else if (ns > 1) {
            // sequence i / L: the committed positions, then its own tokens up to this one (ring slots n_past + i - i % L ..)
            std::fill(mr, mr + n_kv, ninf);
            std::fill(mr, mr + std::min(n_kv, n_past_), zero);
            const int s0 = n_past_ + i - i % L;
            std::fill(mr + s0, mr + std::min(n_kv, n_past_ + i + 1), zero);
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
        const int            nm = L;
        std::vector<int32_t> mp((size_t) (nm - 1) * 4, 0);
        std::vector<int64_t> mr((size_t) (nm - 1));
        for (int i = 0; i < nm - 1; i++) {
            mp[(size_t) i] = mp[(size_t) (nm - 1) + i] = mp[(size_t) 2 * (nm - 1) + i] = n_past_ + i;
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
    if (st == GGML_STATUS_SUCCESS) traffic::pass(with_res);
    if (st != GGML_STATUS_SUCCESS) {
        ggml_free(ctx);
        err = "eval: graph compute failed";
        return false;
    }
    if (opts.argmax) {
        if (ids) ggml_backend_tensor_get(out, ids, 0, ggml_nbytes(out));
    } else {
        if (logits) ggml_backend_tensor_get(out, logits, 0, ggml_nbytes(out));
        if (opts.topk > 0 && ids && out_ids_) ggml_backend_tensor_get(out_ids_, ids, 0, ggml_nbytes(out_ids_));
    }
    if (sparse_ && Kd_ > 0 && sp_kp_ > 0) fill_draft_far();
    sel_nodes_.clear();
    ggml_free(ctx);
    recorded_n_    = opts.record ? std::min(n, opt_.max_record) : 0;
    rec_seqs_      = ns;
    recorded_full_ = opts.record ? n : 0;
    dry_pending_   = opts.dry && opts.record;
    hid_row_    = mtp_on_ ? n - 1 : -1;
    if (!opts.window_ok) exact_upto_ = n_past_ + n;
    if (use_host) host_valid_ = n_past_ + n;
    if (!dry_pending_) n_past_ += n;  // a dry eval moves on in commit()
    return true;
}

bool Qwen35::eval_multi(const std::vector<MultiSeq> & seqs, int topk, float * logits, int32_t * ids, std::string & err) {
    const auto &  h   = hp_;
    const int64_t hd  = h.head_dim, nkv = h.n_head_kv, g = h.n_head / h.n_head_kv;
    const float   eps = h.rms_eps;
    if (slots_.size() < 2 || seqs.empty()) {
        err = "eval_multi: needs sequence slots";
        return false;
    }
    if (dry_pending_ || !multi_.empty()) {
        err = "eval_multi: an earlier eval is not committed";
        return false;
    }
    // per sequence: its slot, rows, position and sparse attention shape (as eval() with the KV in RAM)
    struct P {
        KvSlot *      sl = nullptr;
        int           slot = 0, off = 0, n = 0, p = 0, ws = 0, nfar = 0, kp = 0, nwin = 0, nsel = 0;
        ggml_tensor * kvidx = nullptr, * hidx = nullptr, * smask = nullptr, * sbias = nullptr;
        std::vector<ggml_tensor *> sel;
    };
    static const int win   = std::getenv("E8_SPARSE_WINDOW") ? std::atoi(std::getenv("E8_SPARSE_WINDOW")) : 4096;
    static const int pages = std::getenv("E8_SPARSE_PAGES") ? std::atoi(std::getenv("E8_SPARSE_PAGES")) : 128;
    std::vector<P> ps(seqs.size());
    int            N = 0;
    park_active();  // every sequence then reads its slot's state, KV and summaries the same way
    for (size_t b = 0; b < seqs.size(); b++) {
        const MultiSeq & q = seqs[b];
        P &              x = ps[b];
        if (q.slot < 0 || q.slot >= (int) slots_.size() || q.n < 1 || q.n > kSparseMaxN || !q.tokens) {
            err = "eval_multi: bad sequence";
            return false;
        }
        for (size_t c = 0; c < b; c++) {
            if (ps[c].slot == q.slot) {
                err = "eval_multi: a slot twice";
                return false;
            }
        }
        for (int i = 0; i < q.n; i++) {
            if (q.tokens[i] < 0 || q.tokens[i] >= h.n_vocab) {
                err = "eval_multi: token id out of range";
                return false;
            }
        }
        x.sl   = &slots_[(size_t) q.slot];
        x.slot = q.slot;
        x.off  = N;
        x.n    = q.n;
        x.p    = x.sl->n_past;
        N += q.n;
        if (x.p + x.n > n_ctx_ || x.sl->host_valid < x.p) {
            err = "eval_multi: slot " + std::to_string(q.slot) + (x.p + x.n > n_ctx_ ? " is full" : ": its RAM KV is incomplete");
            return false;
        }
        if (!ensure_committed_slot(*x.sl, x.p + x.n, err)) return false;
        x.ws   = std::max(0, (x.p + x.n - (std::max(win, x.n) + std::max(0, x.n - kSparseMaxN))) / kPage * kPage);
        x.nfar = x.ws / kPage;
        x.kp   = std::min(pages, x.nfar);
        x.nwin = x.p + x.n - x.ws;
        x.nsel = (x.kp * kPage + x.nwin + kKvPad - 1) / kKvPad * kKvPad;
        // page summaries [0, nfar) in the slot's RAM copy
        KvSlot & sl = *x.sl;
        sl.sum_upto = std::min(sl.sum_upto, std::min(x.p, sl.host_valid) / kPage);
        sl.sum_saved = std::min(sl.sum_saved, sl.sum_upto);
        if (x.nfar > sl.sum_upto) {
            const size_t row = (size_t) hd * sizeof(ggml_fp16_t), npg = (size_t) (n_ctx_ / kPage);
            const auto   mid = summarize(sl.hk, sl.sum_upto, x.nfar);
            for (size_t t = 0; t < mid.size(); t++) {
                std::memcpy(sl.summ + (t * npg + (size_t) sl.sum_upto) * row, mid[t].data(), mid[t].size() * sizeof(ggml_fp16_t));
            }
            sl.sum_upto = sl.sum_saved = x.nfar;
        }
        if (x.slot == cur_slot_) sum_upto_ = sum_low_ = sl.sum_upto;
    }
    if (N > opt_.max_record || N >= opt_.n_ubatch) {
        err = "eval_multi: " + std::to_string(N) + " tokens, at most " + std::to_string(std::min(opt_.max_record, opt_.n_ubatch - 1));
        return false;
    }

    // ---- graph: the weights once over all N rows; recurrence and attention per sequence
    ggml_init_params ip  = { graph_meta_.size(), graph_meta_.data(), true };
    ggml_context *   ctx = ggml_init(ip);
    ggml_cgraph *    gf  = ggml_new_graph_custom(ctx, kGraphSize, false);
    gpu_nodes_.clear();
    res_on_gpu_.clear();
    attn_nodes_.clear();
    gi_.clear();
    gi_.reserve((size_t) (h.n_layer * nkv * 2) * ps.size());  // stable addresses: the gather ops point at them
    ggml_tensor * inp_tok = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, h.n_embd, N);
    ggml_set_input(inp_tok);
    ggml_tensor * inp_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int64_t) N * 4);
    ggml_set_input(inp_pos);
    for (P & x : ps) {
        x.kvidx = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, x.n);
        x.hidx  = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, x.n);
        x.smask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, x.nsel, x.n);
        for (ggml_tensor * t : { x.kvidx, x.hidx, x.smask }) ggml_set_input(t);
        if (x.kp > 0) {
            x.sbias = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, x.nfar, 1);
            ggml_set_input(x.sbias);
        }
        x.sel.assign(hk_.size(), nullptr);
    }
    use_res_     = has_residual();
    write_state_ = false;  // dry: commit_seq() advances the states
    auto norm = [&](ggml_tensor * t, ggml_tensor * w) { return ggml_mul(ctx, ggml_rms_norm(ctx, t, eps), w); };
    auto cols = [&](ggml_tensor * t, int64_t off, int64_t n) {  // rows [off, off + n) of a [.., N] tensor (contiguous)
        return ggml_view_2d(ctx, t, t->ne[0], n, t->nb[1], t->nb[1] * (size_t) off);
    };
    ggml_tensor * inpL = inp_tok;
    int sections[4] = { h.rope_sections[0], h.rope_sections[1], h.rope_sections[2], h.rope_sections[3] };
    for (int64_t il = 0; il < h.n_layer; il++) {
        const size_t        l   = (size_t) il;
        const Qwen35Layer & Lw  = layers_[l];
        ggml_tensor *       cur = norm(inpL, Lw.attn_norm);
        if (h.is_recurrent(il)) {
            const int64_t S = h.ssm_d_state, Hv = h.ssm_n_v, C = h.conv_channels();
            ggml_tensor * qkv   = mm(ctx, Lw.wqkv, cur);  // [C, N]
            ggml_tensor * z     = mm(ctx, Lw.wz, cur);
            ggml_tensor * beta  = ggml_sigmoid(ctx, mm(ctx, Lw.w_beta, cur));  // [Hv, N]
            ggml_tensor * alpha = mm(ctx, Lw.w_alpha, cur);
            ggml_tensor * gg    = ggml_mul(ctx, ggml_softplus(ctx, ggml_add(ctx, alpha, Lw.dt_bias)), Lw.a);  // [Hv, N]
            // record every row: commit_seq() replays a prefix of each sequence's rows
            ggml_build_forward_expand(gf, ggml_cpy(ctx, qkv, ggml_view_2d(ctx, rec_qkv_[l], C, N, rec_qkv_[l]->nb[1], 0)));
            ggml_build_forward_expand(gf, ggml_cpy(ctx, gg, ggml_view_2d(ctx, rec_g_[l], Hv, N, rec_g_[l]->nb[1], 0)));
            ggml_build_forward_expand(gf, ggml_cpy(ctx, beta, ggml_view_2d(ctx, rec_beta_[l], Hv, N, rec_beta_[l]->nb[1], 0)));
            ggml_tensor * out = nullptr;
            for (P & x : ps) {
                ggml_tensor * ob = gdn_core(ctx, gf, il, ggml_reshape_3d(ctx, cols(qkv, x.off, x.n), C, x.n, 1),
                                            ggml_reshape_4d(ctx, cols(gg, x.off, x.n), 1, Hv, x.n, 1),
                                            ggml_reshape_4d(ctx, cols(beta, x.off, x.n), 1, Hv, x.n, 1), x.n, 1,
                                            x.sl->conv[l], x.sl->ssm[l]);
                ob  = ggml_cont(ctx, ob);  // [S, Hv, n, 1]
                out = out ? ggml_concat(ctx, out, ob, 2) : ob;
            }
            ggml_tensor * z4 = ggml_reshape_4d(ctx, z, S, Hv, N, 1);
            ggml_tensor * on = ggml_mul(ctx, norm(out, Lw.ssm_norm), ggml_silu(ctx, z4));
            cur              = mm(ctx, Lw.ssm_out, ggml_reshape_2d(ctx, on, S * Hv, N));
        } else {
            ggml_tensor * qg = ggml_reshape_3d(ctx, mm(ctx, Lw.wq, cur), hd * 2, h.n_head, N);
            ggml_tensor * Kc = ggml_reshape_3d(ctx, mm(ctx, Lw.wk, cur), hd, nkv, N);
            ggml_tensor * Vc = ggml_reshape_3d(ctx, mm(ctx, Lw.wv, cur), hd, nkv, N);
            const size_t  es = ggml_element_size(qg);
            ggml_tensor * Qc   = ggml_view_3d(ctx, qg, hd, h.n_head, N, es * hd * 2, es * hd * 2 * h.n_head, 0);
            ggml_tensor * gate = ggml_cont_2d(ctx, ggml_view_3d(ctx, qg, hd, h.n_head, N, es * hd * 2, es * hd * 2 * h.n_head, es * hd),
                                              hd * h.n_head, N);
            Qc = ggml_rope_multi(ctx, norm(Qc, Lw.q_norm), inp_pos, nullptr, (int) h.n_rot, sections, GGML_ROPE_TYPE_IMROPE,
                                 (int) h.n_ctx_train, h.rope_freq_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
            Kc = ggml_rope_multi(ctx, norm(Kc, Lw.k_norm), inp_pos, nullptr, (int) h.n_rot, sections, GGML_ROPE_TYPE_IMROPE,
                                 (int) h.n_ctx_train, h.rope_freq_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
            Qc = ggml_cont(ctx, Qc);
            Kc = ggml_cont(ctx, Kc);
            // in phases over all sequences (K/V writes, page selections on the GPU, gathers on the CPU, attention on the
            // GPU), so a layer switches between the GPU and the CPU a few times, not a few times per sequence
            const int64_t row = hd * nkv;
            const size_t  nb  = ps.size();
            std::vector<ggml_tensor *> qs(nb);
            std::vector<std::vector<ggml_tensor *>> wk(nb, std::vector<ggml_tensor *>((size_t) nkv)),
                wv = wk, sl_(nb, std::vector<ggml_tensor *>((size_t) nkv, nullptr)), gk = wk, gv = wk, qc = wk;
            for (size_t b = 0; b < nb; b++) {  // this batch's K/V: the slot's ring part (drafts), then its RAM KV
                P &           x  = ps[b];
                KvSlot &      sl = *x.sl;
                ggml_tensor * Kb = ggml_view_3d(ctx, Kc, hd, nkv, x.n, Kc->nb[1], Kc->nb[2], Kc->nb[2] * (size_t) x.off);
                ggml_tensor * Vb = ggml_view_3d(ctx, Vc, hd, nkv, x.n, Vc->nb[1], Vc->nb[2], Vc->nb[2] * (size_t) x.off);
                ggml_tensor * Qb = ggml_view_3d(ctx, Qc, hd, h.n_head, x.n, Qc->nb[1], Qc->nb[2], Qc->nb[2] * (size_t) x.off);
                ggml_build_forward_expand(gf, ggml_set_rows(ctx, sl.kr[l], ggml_reshape_2d(ctx, ggml_cont(ctx, Kb), row, x.n), x.kvidx));
                ggml_build_forward_expand(gf, ggml_set_rows(ctx, sl.vr[l], ggml_reshape_2d(ctx, ggml_cont(ctx, Vb), row, x.n), x.kvidx));
                qs[b] = ggml_permute(ctx, Qb, 0, 2, 1, 3);  // [hd, n, n_head]
                for (int64_t j = 0; j < nkv; j++) {
                    const size_t  hi = (size_t) (il * nkv + j);
                    ggml_tensor * kj = ggml_cont(ctx, ggml_view_2d(ctx, Kb, hd, x.n, Kb->nb[2], Kb->nb[1] * (size_t) j));
                    ggml_tensor * vj = ggml_cont(ctx, ggml_view_2d(ctx, Vb, hd, x.n, Vb->nb[2], Vb->nb[1] * (size_t) j));
                    wk[b][(size_t) j] = ggml_set_rows(ctx, sl.hk[hi], kj, x.hidx);
                    wv[b][(size_t) j] = ggml_set_rows(ctx, sl.hv[hi], vj, x.hidx);
                    ggml_build_forward_expand(gf, wk[b][(size_t) j]);
                    ggml_build_forward_expand(gf, wv[b][(size_t) j]);
                }
            }
            for (size_t b = 0; b < nb; b++) {  // page selections over the slots' summaries (from RAM, on the GPU)
                P & x = ps[b];
                for (int64_t j = 0; j < nkv; j++) {
                    const size_t hi = (size_t) (il * nkv + j);
                    ggml_tensor * q = qs[b];
                    qc[b][(size_t) j] = ggml_view_3d(ctx, q, hd, x.n, g, q->nb[1], q->nb[2], q->nb[2] * (size_t) (j * g));
                    if (x.kp == 0) continue;
                    ggml_tensor * q2 = ggml_reshape_2d(ctx, ggml_cont(ctx, qc[b][(size_t) j]), hd, x.n * g);
                    ggml_tensor * md = ggml_view_2d(ctx, x.sl->sumt[hi], hd, x.nfar, x.sl->sumt[hi]->nb[1], 0);
                    ggml_tensor * s  = ggml_mul_mat(ctx, md, q2);
                    gpu_nodes_.push_back(s);
                    s = ggml_soft_max_ext(ctx, s, nullptr, 1.0f / sqrtf((float) hd), 0.0f);
                    s = ggml_sum_rows(ctx, ggml_cont(ctx, ggml_transpose(ctx, s)));
                    s = ggml_add(ctx, ggml_reshape_2d(ctx, s, x.nfar, 1), x.sbias);
                    ggml_tensor * t = ggml_cont(ctx, ggml_argsort_top_k(ctx, s, x.kp));
                    ggml_set_output(t);
                    ggml_build_forward_expand(gf, t);
                    x.sel[hi]         = t;
                    sl_[b][(size_t) j] = t;
                }
            }
            for (size_t b = 0; b < nb; b++) {  // gather the selected pages and the window from RAM
                P & x = ps[b];
                for (int64_t j = 0; j < nkv; j++) {
                    const size_t hi = (size_t) (il * nkv + j);
                    for (int kv = 0; kv < 2; kv++) {
                        gi_.push_back({ kv ? x.sl->hv[hi] : x.sl->hk[hi], x.kp * kPage, kPage, x.ws, x.nwin });
                        ggml_tensor * dep     = kv ? wv[b][(size_t) j] : wk[b][(size_t) j];
                        ggml_tensor * args[2] = { sl_[b][(size_t) j] ? sl_[b][(size_t) j] : dep, dep };
                        ggml_tensor * t = ggml_custom_4d(ctx, kv ? opt_.kv_type_v : opt_.kv_type, hd, x.nsel, 1, 1, args, 2, gather_op,
                                                         GGML_N_TASKS_MAX, &gi_.back());
                        ggml_build_forward_expand(gf, t);
                        (kv ? gv : gk)[b][(size_t) j] = t;
                    }
                }
            }
            ggml_tensor * a = nullptr;
            for (size_t b = 0; b < nb; b++) {  // attention
                P &           x  = ps[b];
                ggml_tensor * ab = nullptr;
                for (int64_t j = 0; j < nkv; j++) {
                    ggml_tensor * aj = ggml_flash_attn_ext(ctx, qc[b][(size_t) j], gk[b][(size_t) j], gv[b][(size_t) j], x.smask,
                                                           1.0f / sqrtf((float) hd), 0.0f, 0.0f);
                    ggml_prec_set_acc(aj, GGML_PREC_F32);
                    attn_nodes_.push_back(aj);
                    ab = ab ? ggml_concat(ctx, ab, aj, 1) : aj;  // [hd, heads, n]
                }
                a = a ? ggml_concat(ctx, a, ab, 2) : ab;  // [hd, heads, N]
            }
            a   = ggml_reshape_2d(ctx, a, a->ne[0] * a->ne[1], a->ne[2] * a->ne[3]);
            a   = ggml_mul(ctx, a, ggml_sigmoid(ctx, gate));
            cur = mm(ctx, Lw.wo, a);
        }
        cur                 = ggml_add(ctx, cur, inpL);
        ggml_tensor * resid = cur;
        ggml_tensor * xx    = norm(cur, Lw.post_norm);
        ggml_tensor * ff    = ggml_swiglu_split(ctx, mm(ctx, Lw.ffn_gate, xx), mm(ctx, Lw.ffn_up, xx));
        inpL                = ggml_add(ctx, mm(ctx, Lw.ffn_down, ff), resid);
    }
    ggml_tensor * hall = norm(inpL, out_norm_);
    if (mtp_on_) ggml_build_forward_expand(gf, ggml_cpy(ctx, hall, ggml_view_2d(ctx, mtp_hid_, h.n_embd, N, mtp_hid_->nb[1], 0)));
    ggml_tensor * out  = mm(ctx, output_, hall);
    ggml_tensor * oids = nullptr;
    if (topk > 0) {
        rows_top_k(ctx, out, topk, out, oids);
        ggml_set_output(oids);
        ggml_build_forward_expand(gf, oids);
    }
    ggml_set_output(out);
    ggml_build_forward_expand(gf, out);
    use_res_     = false;
    write_state_ = true;

    ggml_backend_sched_reset(sched_);
    for (ggml_tensor * t : res_on_gpu_) ggml_backend_sched_set_tensor_backend(sched_, t, gpu_);
    res_on_gpu_.clear();
    for (ggml_tensor * t : gpu_nodes_) ggml_backend_sched_set_tensor_backend(sched_, t, gpu_);
    gpu_nodes_.clear();
    for (ggml_tensor * t : attn_nodes_) ggml_backend_sched_set_tensor_backend(sched_, t, gpu_);
    if (!ggml_backend_sched_alloc_graph(sched_, gf)) {
        ggml_free(ctx);
        err = "eval_multi: cannot allocate the compute graph";
        return false;
    }
    {
        std::vector<int32_t> toks((size_t) N);
        for (const P & x : ps) std::copy(seqs[(size_t) (&x - ps.data())].tokens, seqs[(size_t) (&x - ps.data())].tokens + x.n, toks.begin() + x.off);
        std::vector<float> emb;
        embed(toks.data(), N, emb);
        ggml_backend_tensor_set(inp_tok, emb.data(), 0, emb.size() * sizeof(float));
    }
    std::vector<int32_t> pos((size_t) N * 4, 0);
    const ggml_fp16_t    zero = ggml_fp32_to_fp16(0.0f), ninf = ggml_fp32_to_fp16(-INFINITY);
    for (P & x : ps) {
        std::vector<int64_t> kvr((size_t) x.n), hr((size_t) x.n);
        for (int i = 0; i < x.n; i++) {
            const int qp = x.p + i;
            pos[(size_t) (x.off + i)] = pos[(size_t) (N + x.off + i)] = pos[(size_t) (2 * N + x.off + i)] = qp;
            kvr[(size_t) i] = Kd_ + qp % W_;
            hr[(size_t) i]  = qp;
        }
        ggml_backend_tensor_set(x.kvidx, kvr.data(), 0, kvr.size() * sizeof(int64_t));
        ggml_backend_tensor_set(x.hidx, hr.data(), 0, hr.size() * sizeof(int64_t));
        std::vector<ggml_fp16_t> sm((size_t) x.nsel * x.n, ninf);
        const int                kr = x.kp * kPage;
        for (int i = 0; i < x.n; i++) {
            ggml_fp16_t * mr = sm.data() + (size_t) i * x.nsel;
            std::fill(mr, mr + kr, zero);
            std::fill(mr + kr, mr + kr + (x.p + i + 1 - x.ws), zero);
        }
        ggml_backend_tensor_set(x.smask, sm.data(), 0, sm.size() * sizeof(ggml_fp16_t));
        if (x.sbias && x.sbias->buffer) {
            std::vector<float> bb((size_t) x.nfar, 0.0f);
            bb[0] = 1e30f;  // the first page (attention sink) is always kept
            ggml_backend_tensor_set(x.sbias, bb.data(), 0, bb.size() * sizeof(float));
        }
    }
    ggml_backend_tensor_set(inp_pos, pos.data(), 0, pos.size() * sizeof(int32_t));
    if (ggml_backend_sched_graph_compute(sched_, gf) != GGML_STATUS_SUCCESS) {
        ggml_free(ctx);
        err = "eval_multi: graph compute failed";
        return false;
    }
    traffic::pass(has_residual());
    if (logits) ggml_backend_tensor_get(out, logits, 0, ggml_nbytes(out));
    if (oids && ids) ggml_backend_tensor_get(oids, ids, 0, ggml_nbytes(oids));
    if (mtp_on_) {
        multi_hid_.resize((size_t) h.n_embd * N);
        ggml_backend_tensor_get(mtp_hid_, multi_hid_.data(), 0, multi_hid_.size() * sizeof(float));
    }
    // bookkeeping per slot: far areas from this verify's page selections, ring positions, RAM KV written
    multi_.clear();
    bool active_in = false;
    for (P & x : ps) {
        KvSlot &   sl  = *x.sl;
        const bool act = x.slot == cur_slot_;
        active_in      = active_in || act;
        if (x.kp > 0 && Kd_ > 0) {
            const int rows = fill_far(sl.kr, sl.vr, sl.hk, sl.hv, x.sel, x.kp);
            if (rows > 0) {
                (act ? far_rows_ : sl.far_rows) = rows;
                (act ? far_ws_ : sl.far_ws)     = x.ws;
            }
        }
        std::vector<int> & sp = act ? slot_pos_ : sl.slot_pos;
        if ((int) sp.size() != W_) sp.assign((size_t) W_, -1);
        for (int i = 0; i < x.n; i++) sp[(size_t) ((x.p + i) % W_)] = x.p + i;
        sl.host_valid = sl.exact_upto = x.p + x.n;
        if (act) host_valid_ = exact_upto_ = x.p + x.n;
        multi_.push_back({ x.slot, x.off, x.n, x.p, false });
    }
    if (mtp_on_ && !active_in && cur_slot_ >= 0) {  // the active slot sat out: its hidden row (overwritten above) back
        KvSlot & o = slots_[(size_t) cur_slot_];
        hid_row_   = -1;
        if (!o.hid.empty()) {
            hid_row_ = opt_.n_ubatch - 1;
            ggml_backend_tensor_set(mtp_hid_, o.hid.data(), mtp_hid_->nb[1] * (size_t) hid_row_, o.hid.size() * sizeof(float));
        }
    }
    gi_.clear();
    attn_nodes_.clear();
    ggml_free(ctx);
    recorded_n_ = recorded_full_ = 0;
    return true;
}

bool Qwen35::commit_seq(int slot, int keep, std::string & err) {
    MultiEntry * e = nullptr;
    for (MultiEntry & x : multi_) {
        if (x.slot == slot && !x.done) e = &x;
    }
    if (!e || keep < 0 || keep > e->n) {
        err = "commit_seq: no such sequence or keep out of range";
        return false;
    }
    KvSlot &   sl  = slots_[(size_t) slot];
    const bool act = slot == cur_slot_;
    // the recurrence over the kept rows, from the slot's state (VRAM when active, else its RAM copy)
    if (!replay(keep, err, e->off, act ? nullptr : &sl.conv, act ? nullptr : &sl.ssm)) return false;
    (act ? n_past_ : sl.n_past) = e->n_past + keep;
    if (mtp_on_ && keep > 0) {
        const float * hr = multi_hid_.data() + (size_t) hp_.n_embd * (size_t) (e->off + keep - 1);
        if (act) {
            hid_row_ = opt_.n_ubatch - 1;
            ggml_backend_tensor_set(mtp_hid_, hr, mtp_hid_->nb[1] * (size_t) hid_row_, (size_t) hp_.n_embd * sizeof(float));
        } else {
            sl.hid.assign(hr, hr + hp_.n_embd);
        }
    }
    e->done = true;
    if (std::all_of(multi_.begin(), multi_.end(), [](const MultiEntry & x) { return x.done; })) multi_.clear();
    return true;
}

void Qwen35::detach() {
    if (slots_.empty() || cur_slot_ < 0) return;
    park_active();
    KvSlot & o    = slots_[(size_t) cur_slot_];
    o.slot_pos     = std::move(slot_pos_);
    o.mtp_slot_pos = std::move(mtp_slot_pos_);
    cur_slot_      = -1;
    hid_row_       = -1;
}

bool Qwen35::make_slot_drafter(Qwen35 & d, int n_work, std::string & err) {
    // a second sequence over this model's weights, KV and VRAM recurrent state (and its snapshot), with its own
    // records, MTP hidden rows, CUDA stream, CPU threads and scheduler
    if (slots_.size() < 2) {
        err = "make_slot_drafter: needs sequence slots";
        return false;
    }
    const auto & h = hp_;
    d.shadow_      = true;
    d.owner_       = this;
    d.hp_          = h;
    d.opt_         = opt_;
    d.opt_.n_ubatch   = std::min(opt_.n_ubatch, 128);  // drafts and their checks only
    d.opt_.max_record = 32;  // a lockstep round (rows padded to 8 per member)
    d.opt_.residual_path.clear();
    d.n_ctx_    = n_ctx_;
    d.W_        = W_;
    d.Kd_       = Kd_;
    d.mtp_on_   = mtp_on_;
    d.Wm_       = Wm_;
    d.tok_embd_ = tok_embd_;
    d.out_norm_ = out_norm_;
    d.output_   = output_;
    d.layers_   = layers_;
    d.mtp_      = mtp_;
    d.mtp_eh_    = mtp_eh_;
    d.mtp_enorm_ = mtp_enorm_;
    d.mtp_hnorm_ = mtp_hnorm_;
    d.mtp_norm_  = mtp_norm_;
    d.pmid_      = pmid_;
    d.host_gpu_min_ = host_gpu_min_;
    d.conv_state_ = conv_state_;  // shared: this model is detached while the drafter is attached
    d.ssm_state_  = ssm_state_;
    d.conv_bak_   = conv_bak_;
    d.ssm_bak_    = ssm_bak_;
    d.conv_bak1_.assign(conv_state_.size(), nullptr);
    d.ssm_bak1_.assign(conv_state_.size(), nullptr);
    d.conv_ck_.assign(conv_state_.size(), nullptr);
    d.ssm_ck_.assign(conv_state_.size(), nullptr);
    d.cpu_ = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!d.cpu_) {
        err = "drafter: cannot initialize the CPU backend";
        return false;
    }
    ggml_backend_cpu_set_n_threads(d.cpu_, 4);
    if (gpu_) d.gpu_ = ggml_backend_dev_init(ggml_backend_get_device(gpu_), nullptr);
    ggml_init_params sp = { ggml_tensor_overhead() * (size_t) (h.n_layer * (4 + 2 * std::max(1, n_work)) + 16), nullptr, true };
    d.sctx_[0]          = ggml_init(sp);
    d.sctx_[1]          = ggml_init(sp);
    const size_t nl     = (size_t) h.n_layer;
    d.rec_qkv_.assign(nl, nullptr);
    d.rec_g_.assign(nl, nullptr);
    d.rec_beta_.assign(nl, nullptr);
    for (size_t il = 0; il < nl; il++) {
        if (!conv_state_[il]) continue;
        ggml_context * c = d.sctx_[layers_[il].on_gpu ? 0 : 1];
        d.rec_qkv_[il]   = ggml_new_tensor_2d(c, GGML_TYPE_F32, h.conv_channels(), d.opt_.max_record);
        d.rec_g_[il]     = ggml_new_tensor_2d(c, GGML_TYPE_F32, h.ssm_n_v, d.opt_.max_record);
        d.rec_beta_[il]  = ggml_new_tensor_2d(c, GGML_TYPE_F32, h.ssm_n_v, d.opt_.max_record);
    }
    if (mtp_on_) {
        d.mtp_hid_    = ggml_new_tensor_2d(d.sctx_[0], GGML_TYPE_F32, h.n_embd, d.opt_.n_ubatch);
        d.mtp_chain_  = ggml_new_tensor_2d(d.sctx_[0], GGML_TYPE_F32, h.n_embd, 1);
        d.mtp_chains_ = ggml_new_tensor_2d(d.sctx_[0], GGML_TYPE_F32, h.n_embd, std::max(1, n_work));  // per member
    }
    // working states for drafting several slots at once (draft_begin): the first is the shared state
    n_work = std::max(1, n_work);
    d.work_conv_.assign((size_t) n_work, std::vector<ggml_tensor *>(nl, nullptr));
    d.work_ssm_.assign((size_t) n_work, std::vector<ggml_tensor *>(nl, nullptr));
    d.work_conv_[0] = conv_state_;
    d.work_ssm_[0]  = ssm_state_;
    for (int w = 1; w < n_work; w++) {
        for (size_t il = 0; il < nl; il++) {
            if (!conv_state_[il]) continue;
            d.work_conv_[(size_t) w][il] = ggml_dup_tensor(d.sctx_[0], conv_state_[il]);
            d.work_ssm_[(size_t) w][il]  = ggml_dup_tensor(d.sctx_[0], ssm_state_[il]);
        }
    }
    for (int i = 0; i < 2; i++) {
        ggml_backend_t be = i == 0 ? d.gpu_ : d.cpu_;
        if (!be || !ggml_get_first_tensor(d.sctx_[i])) continue;
        d.sbuf_[i] = ggml_backend_alloc_ctx_tensors_from_buft(d.sctx_[i], ggml_backend_get_default_buffer_type(be));
        if (!d.sbuf_[i]) {
            err = "drafter: not enough VRAM for its records";
            return false;
        }
    }
    std::vector<ggml_backend_t> bes;
    if (d.gpu_) bes.push_back(d.gpu_);
    bes.push_back(d.cpu_);
    d.sched_ = ggml_backend_sched_new(bes.data(), nullptr, (int) bes.size(), kGraphSize, false, true);
    d.graph_meta_.resize(ggml_tensor_overhead() * kGraphSize + ggml_graph_overhead_custom(kGraphSize, false));
    d.graph_meta_draft_.resize(d.graph_meta_.size());
    d.graph_meta_mtp_.resize(d.graph_meta_.size());
    d.cur_slot_ = -1;
    return true;
}

bool Qwen35::attach_draft(int s, std::string & err) {
    // drafter: continue slot s's committed sequence (its parked state, ring parts and bookkeeping)
    if (!owner_ || s < 0 || s >= (int) owner_->slots_.size() || cur_slot_ >= 0) {
        err = "attach_draft: not a drafter, no such slot, or already attached";
        return false;
    }
    KvSlot & sl = owner_->slots_[(size_t) s];
    for (size_t il = 0; il < conv_state_.size(); il++) {
        if (!conv_state_[il]) continue;
        if (sl.n_past > 0) {
            ggml_backend_tensor_copy(sl.conv[il], conv_state_[il]);
            ggml_backend_tensor_copy(sl.ssm[il], ssm_state_[il]);
        } else {
            ggml_backend_tensor_memset(conv_state_[il], 0, 0, ggml_nbytes(conv_state_[il]));
            ggml_backend_tensor_memset(ssm_state_[il], 0, 0, ggml_nbytes(ssm_state_[il]));
        }
    }
    k_cache_ = sl.kr;
    v_cache_ = sl.vr;
    hk_      = sl.hk;
    hv_      = sl.hv;
    if (mtp_on_) {
        mtp_k_ = sl.mk;
        mtp_v_ = sl.mv;
    }
    n_past_       = sl.n_past;
    host_valid_   = sl.host_valid;
    exact_upto_   = sl.exact_upto;
    far_rows_     = sl.far_rows;
    far_ws_       = sl.far_ws;
    slot_pos_     = std::move(sl.slot_pos);
    mtp_slot_pos_ = std::move(sl.mtp_slot_pos);
    if ((int) slot_pos_.size() != W_) slot_pos_.assign((size_t) W_, -1);
    if ((int) mtp_slot_pos_.size() != Wm_) mtp_slot_pos_.assign((size_t) Wm_, -1);
    hid_row_ = -1;
    if (mtp_on_ && !sl.hid.empty()) {
        ggml_backend_tensor_set(mtp_hid_, sl.hid.data(), 0, sl.hid.size() * sizeof(float));
        hid_row_ = 0;
    }
    sum_upto_ = sum_low_ = 0;
    saved_n_past_ = saved_n_past1_ = 0;
    recorded_n_ = recorded_full_ = 0;
    dry_pending_ = false;
    cur_slot_    = s;
    return true;
}

void Qwen35::detach_draft() {
    // drafter: the ring bookkeeping back to the slot (the committed state never changed: drafts end restored)
    if (!owner_ || cur_slot_ < 0) return;
    KvSlot & sl    = owner_->slots_[(size_t) cur_slot_];
    sl.slot_pos     = std::move(slot_pos_);
    sl.mtp_slot_pos = std::move(mtp_slot_pos_);
    cur_slot_       = -1;
}

bool Qwen35::draft_begin(const std::vector<int> & slots, std::string & err) {
    // drafter: drafts for several slots at once, each on a working recurrent state of its own (loaded from the slot's
    // committed state); the slots' ring parts and bookkeeping are used in place
    if (!owner_ || cur_slot_ >= 0 || !members_.empty() || slots.empty() || slots.size() > work_conv_.size()) {
        err = "draft_begin: not a free drafter, or more slots than working states";
        return false;
    }
    members_.clear();
    for (size_t b = 0; b < slots.size(); b++) {
        const int s = slots[b];
        if (s < 0 || s >= (int) owner_->slots_.size()) {
            err = "draft_begin: no such slot";
            return false;
        }
        KvSlot & sl = owner_->slots_[(size_t) s];
        for (size_t il = 0; il < conv_state_.size(); il++) {
            if (!conv_state_[il]) continue;
            if (sl.n_past > 0) {
                ggml_backend_tensor_copy(sl.conv[il], work_conv_[b][il]);
                ggml_backend_tensor_copy(sl.ssm[il], work_ssm_[b][il]);
            } else {
                ggml_backend_tensor_memset(work_conv_[b][il], 0, 0, ggml_nbytes(work_conv_[b][il]));
                ggml_backend_tensor_memset(work_ssm_[b][il], 0, 0, ggml_nbytes(work_ssm_[b][il]));
            }
        }
        if ((int) sl.slot_pos.size() != W_) sl.slot_pos.assign((size_t) W_, -1);
        if ((int) sl.mtp_slot_pos.size() != Wm_) sl.mtp_slot_pos.assign((size_t) Wm_, -1);
        Member m;
        m.slot   = s;
        m.n_past = sl.n_past;
        if (mtp_on_ && !sl.hid.empty()) {  // the committed hidden row: rows from the end (draft_eval writes from 0)
            m.hid_row = opt_.n_ubatch - 1 - (int) b;
            ggml_backend_tensor_set(mtp_hid_, sl.hid.data(), mtp_hid_->nb[1] * (size_t) m.hid_row, sl.hid.size() * sizeof(float));
        }
        members_.push_back(m);
    }
    cur_slot_ = -2;  // busy with members
    return true;
}

bool Qwen35::draft_eval(const std::vector<std::vector<int32_t>> & ins, int topk, bool argmax, float * logits, int32_t * ids,
                        std::string & err) {
    // one base pass over every member's tokens (dry: rows recorded for draft_commit), attending each slot's ring part
    // and draft far area; K/V into the slot's ring and MTP ring; hidden rows kept for MTP
    const auto &  h   = hp_;
    const int64_t hd  = h.head_dim, nkv = h.n_head_kv;
    const float   eps = h.rms_eps;
    if (ins.size() != members_.size()) {
        err = "draft_eval: one token list per member";
        return false;
    }
    // each member's rows padded to P (8, or a power of two for long copies), so the graph keeps its shape from round
    // to round and its CUDA graph is reused (one graph scratch per shape). Padding rows lie past the member's tokens:
    // dry, never kept, overwritten before anything attends them
    int maxn = 0, nact = 0;
    for (const auto & v : ins) {
        maxn = std::max(maxn, (int) v.size());
        nact += !v.empty();
    }
    int P = 8;
    while (P < maxn) P *= 2;
    if (nact * P > opt_.max_record || nact * P > opt_.n_ubatch - (int) members_.size()) P = 0;  // no room: no padding
    int N = 0;
    for (size_t b = 0; b < ins.size(); b++) {
        Member & m = members_[b];
        m.off      = N;
        m.n        = (int) ins[b].size();
        m.np       = m.n == 0 ? 0 : P > 0 && m.n_past + P <= n_ctx_ ? P : m.n;
        if (m.n_past + m.n > n_ctx_) {
            err = "draft_eval: past the context";
            return false;
        }
        for (int32_t t : ins[b]) {
            if (t < 0 || t >= h.n_vocab) {
                err = "draft_eval: token id out of range";
                return false;
            }
        }
        N += m.np;
    }
    if (N < 1 || N > opt_.max_record || N > opt_.n_ubatch - (int) members_.size()) {
        err = "draft_eval: " + std::to_string(N) + " tokens is too many";
        return false;
    }
    const int n_kv = Kd_ + W_;
    auto & meta = draft_metas_[{ nact, P }];
    if (meta.empty()) meta.resize(graph_meta_.size());
    ggml_init_params ip  = { meta.size(), meta.data(), true };
    ggml_context *   ctx = ggml_init(ip);
    ggml_cgraph *    gf  = ggml_new_graph_custom(ctx, kGraphSize, false);
    gpu_nodes_.clear();
    res_on_gpu_.clear();
    ggml_tensor * inp_tok = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, h.n_embd, N);
    ggml_tensor * inp_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int64_t) N * 4);
    ggml_set_input(inp_tok);
    ggml_set_input(inp_pos);
    struct In {
        ggml_tensor * kvidx = nullptr, * mask = nullptr, * mpos = nullptr, * mkv = nullptr;
    };
    std::vector<In> pin(members_.size());
    for (size_t b = 0; b < members_.size(); b++) {
        const Member & m = members_[b];
        if (m.n == 0) continue;
        pin[b].kvidx     = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, m.np);
        pin[b].mask      = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, n_kv, m.np, 1, 1);
        ggml_set_input(pin[b].kvidx);
        ggml_set_input(pin[b].mask);
        if (mtp_on_ && m.np > 1) {
            pin[b].mpos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int64_t) (m.np - 1) * 4);
            pin[b].mkv  = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, m.np - 1);
            ggml_set_input(pin[b].mpos);
            ggml_set_input(pin[b].mkv);
        }
    }
    use_res_     = false;
    write_state_ = false;
    auto norm = [&](ggml_tensor * t, ggml_tensor * w) { return ggml_mul(ctx, ggml_rms_norm(ctx, t, eps), w); };
    auto cols = [&](ggml_tensor * t, int64_t off, int64_t n) {
        return ggml_view_2d(ctx, t, t->ne[0], n, t->nb[1], t->nb[1] * (size_t) off);
    };
    ggml_tensor * inpL = inp_tok;
    int sections[4] = { h.rope_sections[0], h.rope_sections[1], h.rope_sections[2], h.rope_sections[3] };
    for (int64_t il = 0; il < h.n_layer; il++) {
        const size_t        l   = (size_t) il;
        const Qwen35Layer & Lw  = layers_[l];
        ggml_tensor *       cur = norm(inpL, Lw.attn_norm);
        if (h.is_recurrent(il)) {
            const int64_t S = h.ssm_d_state, Hv = h.ssm_n_v, C = h.conv_channels();
            ggml_tensor * qkv  = mm(ctx, Lw.wqkv, cur);
            ggml_tensor * z    = mm(ctx, Lw.wz, cur);
            ggml_tensor * beta = ggml_sigmoid(ctx, mm(ctx, Lw.w_beta, cur));
            ggml_tensor * gg   = ggml_mul(ctx, ggml_softplus(ctx, ggml_add(ctx, mm(ctx, Lw.w_alpha, cur), Lw.dt_bias)), Lw.a);
            ggml_build_forward_expand(gf, ggml_cpy(ctx, qkv, ggml_view_2d(ctx, rec_qkv_[l], C, N, rec_qkv_[l]->nb[1], 0)));
            ggml_build_forward_expand(gf, ggml_cpy(ctx, gg, ggml_view_2d(ctx, rec_g_[l], Hv, N, rec_g_[l]->nb[1], 0)));
            ggml_build_forward_expand(gf, ggml_cpy(ctx, beta, ggml_view_2d(ctx, rec_beta_[l], Hv, N, rec_beta_[l]->nb[1], 0)));
            ggml_tensor * out = nullptr;
            for (size_t b = 0; b < members_.size(); b++) {
                const Member & m  = members_[b];
                if (m.n == 0) continue;
                ggml_tensor *  ob = gdn_core(ctx, gf, il, ggml_reshape_3d(ctx, cols(qkv, m.off, m.np), C, m.np, 1),
                                             ggml_reshape_4d(ctx, cols(gg, m.off, m.np), 1, Hv, m.np, 1),
                                             ggml_reshape_4d(ctx, cols(beta, m.off, m.np), 1, Hv, m.np, 1), m.np, 1,
                                             work_conv_[b][l], work_ssm_[b][l]);
                ob  = ggml_cont(ctx, ob);
                out = out ? ggml_concat(ctx, out, ob, 2) : ob;
            }
            ggml_tensor * on = ggml_mul(ctx, norm(out, Lw.ssm_norm), ggml_silu(ctx, ggml_reshape_4d(ctx, z, S, Hv, N, 1)));
            cur              = mm(ctx, Lw.ssm_out, ggml_reshape_2d(ctx, on, S * Hv, N));
        } else {
            ggml_tensor * qg = ggml_reshape_3d(ctx, mm(ctx, Lw.wq, cur), hd * 2, h.n_head, N);
            ggml_tensor * Kc = ggml_reshape_3d(ctx, mm(ctx, Lw.wk, cur), hd, nkv, N);
            ggml_tensor * Vc = ggml_reshape_3d(ctx, mm(ctx, Lw.wv, cur), hd, nkv, N);
            const size_t  es = ggml_element_size(qg);
            ggml_tensor * Qc   = ggml_view_3d(ctx, qg, hd, h.n_head, N, es * hd * 2, es * hd * 2 * h.n_head, 0);
            ggml_tensor * gate = ggml_cont_2d(ctx, ggml_view_3d(ctx, qg, hd, h.n_head, N, es * hd * 2, es * hd * 2 * h.n_head, es * hd),
                                              hd * h.n_head, N);
            Qc = ggml_cont(ctx, ggml_rope_multi(ctx, norm(Qc, Lw.q_norm), inp_pos, nullptr, (int) h.n_rot, sections, GGML_ROPE_TYPE_IMROPE,
                                                (int) h.n_ctx_train, h.rope_freq_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f));
            Kc = ggml_cont(ctx, ggml_rope_multi(ctx, norm(Kc, Lw.k_norm), inp_pos, nullptr, (int) h.n_rot, sections, GGML_ROPE_TYPE_IMROPE,
                                                (int) h.n_ctx_train, h.rope_freq_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f));
            const int64_t row = hd * nkv;
            ggml_tensor * a   = nullptr;
            for (size_t b = 0; b < members_.size(); b++) {
                const Member & m  = members_[b];
                if (m.n == 0) continue;
                KvSlot &       sl = owner_->slots_[(size_t) m.slot];
                ggml_tensor *  Kb = ggml_view_3d(ctx, Kc, hd, nkv, m.np, Kc->nb[1], Kc->nb[2], Kc->nb[2] * (size_t) m.off);
                ggml_tensor *  Vb = ggml_view_3d(ctx, Vc, hd, nkv, m.np, Vc->nb[1], Vc->nb[2], Vc->nb[2] * (size_t) m.off);
                ggml_tensor *  Qb = ggml_view_3d(ctx, Qc, hd, h.n_head, m.np, Qc->nb[1], Qc->nb[2], Qc->nb[2] * (size_t) m.off);
                ggml_build_forward_expand(gf, ggml_set_rows(ctx, sl.kr[l], ggml_reshape_2d(ctx, ggml_cont(ctx, Kb), row, m.np), pin[b].kvidx));
                ggml_build_forward_expand(gf, ggml_set_rows(ctx, sl.vr[l], ggml_reshape_2d(ctx, ggml_cont(ctx, Vb), row, m.np), pin[b].kvidx));
                ggml_tensor * K  = ggml_view_3d(ctx, sl.kr[l], hd, nkv, n_kv, ggml_row_size(sl.kr[l]->type, hd), sl.kr[l]->nb[1], 0);
                ggml_tensor * V  = ggml_view_3d(ctx, sl.vr[l], hd, nkv, n_kv, ggml_row_size(sl.vr[l]->type, hd), sl.vr[l]->nb[1], 0);
                ggml_tensor * ab = ggml_flash_attn_ext(ctx, ggml_permute(ctx, Qb, 0, 2, 1, 3), ggml_permute(ctx, K, 0, 2, 1, 3),
                                                       ggml_permute(ctx, V, 0, 2, 1, 3), pin[b].mask, 1.0f / sqrtf((float) hd), 0.0f, 0.0f);
                ggml_prec_set_acc(ab, GGML_PREC_F32);
                a = a ? ggml_concat(ctx, a, ab, 2) : ab;  // [hd, heads, N]
            }
            a   = ggml_reshape_2d(ctx, a, a->ne[0] * a->ne[1], a->ne[2] * a->ne[3]);
            cur = mm(ctx, Lw.wo, ggml_mul(ctx, a, ggml_sigmoid(ctx, gate)));
        }
        cur                 = ggml_add(ctx, cur, inpL);
        ggml_tensor * resid = cur;
        ggml_tensor * xx    = norm(cur, Lw.post_norm);
        ggml_tensor * ff    = ggml_swiglu_split(ctx, mm(ctx, Lw.ffn_gate, xx), mm(ctx, Lw.ffn_up, xx));
        inpL                = ggml_add(ctx, mm(ctx, Lw.ffn_down, ff), resid);
    }
    ggml_tensor * hall = norm(inpL, out_norm_);
    if (mtp_on_) {
        ggml_build_forward_expand(gf, ggml_cpy(ctx, hall, ggml_view_2d(ctx, mtp_hid_, h.n_embd, N, mtp_hid_->nb[1], 0)));
        ggml_tensor * keep_k = mtp_k_, * keep_v = mtp_v_;
        for (size_t b = 0; b < members_.size(); b++) {  // MTP K/V of positions whose next token is in this batch
            const Member & m = members_[b];
            if (m.np < 2) continue;
            KvSlot & sl = owner_->slots_[(size_t) m.slot];
            mtp_k_      = sl.mk;
            mtp_v_      = sl.mv;
            ggml_tensor * e  = ggml_view_2d(ctx, inp_tok, h.n_embd, m.np - 1, inp_tok->nb[1], inp_tok->nb[1] * (size_t) (m.off + 1));
            ggml_tensor * hh = ggml_view_2d(ctx, hall, h.n_embd, m.np - 1, hall->nb[1], hall->nb[1] * (size_t) m.off);
            mtp_layer(ctx, gf, mtp_input(ctx, e, hh), m.np - 1, pin[b].mpos, pin[b].mkv, nullptr);
        }
        mtp_k_ = keep_k;
        mtp_v_ = keep_v;
    }
    ggml_tensor * out  = mm(ctx, output_, hall);
    ggml_tensor * oids = nullptr;
    if (argmax) {
        out = ggml_argmax(ctx, out);
    } else if (topk > 0) {
        rows_top_k(ctx, out, topk, out, oids);
        ggml_set_output(oids);
        ggml_build_forward_expand(gf, oids);  // (the values do not depend on the concatenated ids)
    }
    ggml_set_output(out);
    ggml_build_forward_expand(gf, out);
    write_state_ = true;

    ggml_backend_sched_reset(sched_);
    for (ggml_tensor * t : gpu_nodes_) ggml_backend_sched_set_tensor_backend(sched_, t, gpu_);
    gpu_nodes_.clear();
    if (!ggml_backend_sched_alloc_graph(sched_, gf)) {
        ggml_free(ctx);
        err = "draft_eval: cannot allocate the compute graph";
        return false;
    }
    std::vector<int32_t> toks((size_t) N), pos((size_t) N * 4, 0);
    const ggml_fp16_t    zero = ggml_fp32_to_fp16(0.0f), ninf = ggml_fp32_to_fp16(-INFINITY);
    for (size_t b = 0; b < members_.size(); b++) {
        const Member & m  = members_[b];
        if (m.n == 0) continue;
        KvSlot &       sl = owner_->slots_[(size_t) m.slot];
        std::copy(ins[b].begin(), ins[b].end(), toks.begin() + m.off);
        std::fill(toks.begin() + m.off + m.n, toks.begin() + m.off + m.np, ins[b].back());
        std::vector<int64_t> kr((size_t) m.np);
        for (int i = 0; i < m.np; i++) {
            const int qp = m.n_past + i;
            pos[(size_t) (m.off + i)] = pos[(size_t) (N + m.off + i)] = pos[(size_t) (2 * N + m.off + i)] = qp;
            kr[(size_t) i]                          = Kd_ + qp % W_;
            sl.slot_pos[(size_t) (qp % W_)]         = qp;
        }
        ggml_backend_tensor_set(pin[b].kvidx, kr.data(), 0, kr.size() * sizeof(int64_t));
        // as eval()'s window mask: the far area's pages, then ring positions past them up to the query
        std::vector<ggml_fp16_t> mk((size_t) n_kv * m.np, ninf);
        for (int i = 0; i < m.np; i++) {
            ggml_fp16_t * mr = mk.data() + (size_t) i * n_kv;
            const int     qp = m.n_past + i;
            const int     lo = std::max(qp - (W_ - m.np), sl.far_rows > 0 ? sl.far_ws - 1 : -1);
            std::fill(mr, mr + sl.far_rows, zero);
            for (int j = 0; j < W_; j++) {
                const int sp = sl.slot_pos[(size_t) j];
                mr[Kd_ + j]  = sp >= 0 && sp <= qp && sp > lo ? zero : ninf;
            }
        }
        ggml_backend_tensor_set(pin[b].mask, mk.data(), 0, mk.size() * sizeof(ggml_fp16_t));
        if (pin[b].mpos && pin[b].mpos->buffer) {
            const int            nm = m.np - 1;
            std::vector<int32_t> mp((size_t) nm * 4, 0);
            std::vector<int64_t> mr((size_t) nm);
            for (int i = 0; i < nm; i++) {
                mp[(size_t) i] = mp[(size_t) (nm + i)] = mp[(size_t) (2 * nm + i)] = m.n_past + i;
                mr[(size_t) i]                         = (m.n_past + i) % Wm_;
                sl.mtp_slot_pos[(size_t) mr[(size_t) i]] = m.n_past + i;
            }
            ggml_backend_tensor_set(pin[b].mpos, mp.data(), 0, mp.size() * sizeof(int32_t));
            ggml_backend_tensor_set(pin[b].mkv, mr.data(), 0, mr.size() * sizeof(int64_t));
        }
    }
    {
        std::vector<float> emb;
        embed(toks.data(), N, emb);
        ggml_backend_tensor_set(inp_tok, emb.data(), 0, emb.size() * sizeof(float));
        ggml_backend_tensor_set(inp_pos, pos.data(), 0, pos.size() * sizeof(int32_t));
    }
    if (ggml_backend_sched_graph_compute(sched_, gf) != GGML_STATUS_SUCCESS) {
        ggml_free(ctx);
        err = "draft_eval: graph compute failed";
        return false;
    }
    traffic::pass(false);
    // the real rows of each member, in order (padding rows dropped)
    auto take = [&](ggml_tensor * t, void * dst) {  // (rows of out / oids: 1 id, topk values or ids, or n_vocab logits)
        const size_t rs = argmax ? sizeof(int32_t) : topk > 0 ? (size_t) topk * sizeof(int32_t) : (size_t) h.n_vocab * sizeof(float);
        size_t       o  = 0;
        for (const Member & m : members_) {
            if (m.n == 0) continue;
            ggml_backend_tensor_get(t, (uint8_t *) dst + o, rs * (size_t) m.off, rs * (size_t) m.n);
            o += rs * (size_t) m.n;
        }
    };
    if (argmax) {
        if (ids) take(out, ids);
    } else {
        if (logits) take(out, logits);
        if (oids && ids) take(oids, ids);
    }
    ggml_free(ctx);
    return true;
}

bool Qwen35::draft_commit(const std::vector<int> & keep, std::string & err) {
    // every member keeps its first keep[b] rows of the last draft_eval: one graph replays them into the working states
    const auto & h = hp_;
    if (keep.size() != members_.size()) {
        err = "draft_commit: one count per member";
        return false;
    }
    ggml_init_params ip  = { graph_meta_.size(), graph_meta_.data(), true };
    ggml_context *   ctx = ggml_init(ip);
    ggml_cgraph *    gf  = ggml_new_graph_custom(ctx, kGraphSize, false);
    gpu_nodes_.clear();
    bool any = false;
    for (size_t b = 0; b < members_.size(); b++) {
        const Member & m = members_[b];
        const int      k = keep[b];
        if (k < 0 || k > m.n) {
            ggml_free(ctx);
            err = "draft_commit: keep out of range";
            return false;
        }
        if (k == 0) continue;
        any = true;
        for (int64_t il = 0; il < h.n_layer; il++) {
            if (!h.is_recurrent(il)) continue;
            const size_t  l    = (size_t) il;
            ggml_tensor * qkv  = ggml_view_2d(ctx, rec_qkv_[l], h.conv_channels(), k, rec_qkv_[l]->nb[1], rec_qkv_[l]->nb[1] * (size_t) m.off);
            ggml_tensor * g    = ggml_view_2d(ctx, rec_g_[l], h.ssm_n_v, k, rec_g_[l]->nb[1], rec_g_[l]->nb[1] * (size_t) m.off);
            ggml_tensor * beta = ggml_view_2d(ctx, rec_beta_[l], h.ssm_n_v, k, rec_beta_[l]->nb[1], rec_beta_[l]->nb[1] * (size_t) m.off);
            gdn_core(ctx, gf, il, ggml_reshape_3d(ctx, qkv, h.conv_channels(), k, 1), ggml_reshape_4d(ctx, g, 1, h.ssm_n_v, k, 1),
                     ggml_reshape_4d(ctx, beta, 1, h.ssm_n_v, k, 1), k, 1, work_conv_[b][l], work_ssm_[b][l]);
        }
    }
    const bool ok = !any || compute(ctx, gf, err);
    ggml_free(ctx);
    if (!ok) return false;
    for (size_t b = 0; b < members_.size(); b++) {
        Member & m = members_[b];
        if (keep[b] == 0) continue;
        m.n_past += keep[b];
        m.hid_row = m.off + keep[b] - 1;
    }
    return true;
}

bool Qwen35::draft_mtp(int b, int32_t tok, int j, float * logits, int32_t * id, int topk, std::string & err) {
    // MTP step j of member b's chain (j == 0 reads its hidden row), on its slot's MTP ring
    if (b < 0 || b >= (int) members_.size()) {
        err = "draft_mtp: no such member";
        return false;
    }
    Member & m  = members_[(size_t) b];
    KvSlot & sl = owner_->slots_[(size_t) m.slot];
    if (j == 0 && m.hid_row < 0) {
        err = "draft_mtp: no hidden row";
        return false;
    }
    ggml_tensor * keep_k = mtp_k_, * keep_v = mtp_v_;
    mtp_k_ = sl.mk;
    mtp_v_ = sl.mv;
    std::swap(mtp_slot_pos_, sl.mtp_slot_pos);
    const bool ok = mtp_step(tok, m.n_past - 1 + j, j == 0 ? m.hid_row : -1, logits, id, err, topk);
    std::swap(mtp_slot_pos_, sl.mtp_slot_pos);
    mtp_k_ = keep_k;
    mtp_v_ = keep_v;
    return ok;
}

void Qwen35::draft_end() {
    members_.clear();
    if (cur_slot_ == -2) cur_slot_ = -1;
}

bool Qwen35::draft_mtp_multi(const std::vector<MtpItem> & items, int topk, bool argmax, float * logits, int32_t * ids,
                             std::string & err) {
    // one MTP step for several members at once: item i continues member items[i].member's chain (step j == 0 reads its
    // hidden row, later steps its own chain column), attends its slot's MTP ring; one row of output per item
    const auto &  h   = hp_;
    const int64_t hd  = h.head_dim, M = (int64_t) items.size();
    const float   eps = h.rms_eps;
    if (!mtp_on_ || items.empty() || !mtp_chains_) {
        err = "draft_mtp_multi: no MTP block or no items";
        return false;
    }
    for (const MtpItem & it : items) {
        if (it.member < 0 || it.member >= (int) members_.size() || it.tok < 0 || it.tok >= h.n_vocab ||
            (it.j == 0 && members_[(size_t) it.member].hid_row < 0)) {
            err = "draft_mtp_multi: bad item";
            return false;
        }
    }
    ggml_init_params ip  = { graph_meta_mtp_.size(), graph_meta_mtp_.data(), true };
    ggml_context *   ctx = ggml_init(ip);
    ggml_cgraph *    gf  = ggml_new_graph_custom(ctx, kGraphSize, false);
    ggml_tensor * inp_e = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, h.n_embd, M);
    ggml_tensor * inp_p = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, M * 4);
    ggml_set_input(inp_e);
    ggml_set_input(inp_p);
    std::vector<ggml_tensor *> kvi((size_t) M), msk((size_t) M);
    ggml_tensor *              hid = nullptr;
    for (int64_t i = 0; i < M; i++) {
        const MtpItem & it = items[(size_t) i];
        const Member &  m  = members_[(size_t) it.member];
        kvi[(size_t) i]    = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 1);
        msk[(size_t) i]    = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, Wm_, 1, 1, 1);
        ggml_set_input(kvi[(size_t) i]);
        ggml_set_input(msk[(size_t) i]);
        ggml_tensor * hs = it.j == 0 ? ggml_view_2d(ctx, mtp_hid_, h.n_embd, 1, mtp_hid_->nb[1], mtp_hid_->nb[1] * (size_t) m.hid_row)
                                     : ggml_view_2d(ctx, mtp_chains_, h.n_embd, 1, mtp_chains_->nb[1], mtp_chains_->nb[1] * (size_t) it.member);
        hid = hid ? ggml_concat(ctx, hid, hs, 1) : ggml_cont(ctx, hs);
    }
    // the MTP layer (as mtp_layer) with attention per item
    const Qwen35Layer & L = mtp_;
    int sections[4] = { h.rope_sections[0], h.rope_sections[1], h.rope_sections[2], h.rope_sections[3] };
    auto norm = [&](ggml_tensor * t, ggml_tensor * w) { return ggml_mul(ctx, ggml_rms_norm(ctx, t, eps), w); };
    ggml_tensor * x   = mtp_input(ctx, inp_e, hid);
    ggml_tensor * cur = norm(x, L.attn_norm);
    ggml_tensor * Kc  = norm(ggml_reshape_3d(ctx, ggml_mul_mat(ctx, L.wk, cur), hd, h.n_head_kv, M), L.k_norm);
    ggml_tensor * Vc  = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, L.wv, cur), hd, h.n_head_kv, M);
    Kc = ggml_cont(ctx, ggml_rope_multi(ctx, Kc, inp_p, nullptr, (int) h.n_rot, sections, GGML_ROPE_TYPE_IMROPE, (int) h.n_ctx_train,
                                        h.rope_freq_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f));
    ggml_tensor * qg   = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, L.wq, cur), hd * 2, h.n_head, M);
    const size_t  es   = ggml_element_size(qg);
    ggml_tensor * Qc   = ggml_view_3d(ctx, qg, hd, h.n_head, M, es * hd * 2, es * hd * 2 * h.n_head, 0);
    ggml_tensor * gate = ggml_cont_2d(ctx, ggml_view_3d(ctx, qg, hd, h.n_head, M, es * hd * 2, es * hd * 2 * h.n_head, es * hd),
                                      hd * h.n_head, M);
    Qc = ggml_cont(ctx, ggml_rope_multi(ctx, norm(Qc, L.q_norm), inp_p, nullptr, (int) h.n_rot, sections, GGML_ROPE_TYPE_IMROPE,
                                        (int) h.n_ctx_train, h.rope_freq_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f));
    const int64_t row = hd * h.n_head_kv;
    ggml_tensor * a   = nullptr;
    for (int64_t i = 0; i < M; i++) {
        KvSlot &      sl = owner_->slots_[(size_t) members_[(size_t) items[(size_t) i].member].slot];
        ggml_tensor * Ki = ggml_view_3d(ctx, Kc, hd, h.n_head_kv, 1, Kc->nb[1], Kc->nb[2], Kc->nb[2] * (size_t) i);
        ggml_tensor * Vi = ggml_view_3d(ctx, Vc, hd, h.n_head_kv, 1, Vc->nb[1], Vc->nb[2], Vc->nb[2] * (size_t) i);
        ggml_tensor * Qi = ggml_view_3d(ctx, Qc, hd, h.n_head, 1, Qc->nb[1], Qc->nb[2], Qc->nb[2] * (size_t) i);
        ggml_build_forward_expand(gf, ggml_set_rows(ctx, sl.mk, ggml_reshape_2d(ctx, ggml_cont(ctx, Ki), row, 1), kvi[(size_t) i]));
        ggml_build_forward_expand(gf, ggml_set_rows(ctx, sl.mv, ggml_reshape_2d(ctx, ggml_cont(ctx, Vi), row, 1), kvi[(size_t) i]));
        ggml_tensor * K  = ggml_view_3d(ctx, sl.mk, hd, h.n_head_kv, Wm_, ggml_row_size(sl.mk->type, hd), sl.mk->nb[1], 0);
        ggml_tensor * V  = ggml_view_3d(ctx, sl.mv, hd, h.n_head_kv, Wm_, ggml_row_size(sl.mv->type, hd), sl.mv->nb[1], 0);
        ggml_tensor * ai = ggml_flash_attn_ext(ctx, ggml_permute(ctx, Qi, 0, 2, 1, 3), ggml_permute(ctx, K, 0, 2, 1, 3),
                                               ggml_permute(ctx, V, 0, 2, 1, 3), msk[(size_t) i], 1.0f / sqrtf((float) hd), 0.0f, 0.0f);
        ggml_prec_set_acc(ai, GGML_PREC_F32);
        a = a ? ggml_concat(ctx, a, ai, 2) : ai;  // [hd, heads, M]
    }
    a = ggml_mul(ctx, ggml_reshape_2d(ctx, a, hd * h.n_head, M), ggml_sigmoid(ctx, gate));
    x = ggml_add(ctx, x, ggml_mul_mat(ctx, L.wo, a));
    ggml_tensor * y  = norm(x, L.post_norm);
    ggml_tensor * ff = ggml_swiglu_split(ctx, ggml_mul_mat(ctx, L.ffn_gate, y), ggml_mul_mat(ctx, L.ffn_up, y));
    y                = ggml_add(ctx, x, ggml_mul_mat(ctx, L.ffn_down, ff));
    ggml_tensor * hn = norm(y, mtp_norm_);  // [n_embd, M]: each item's chain goes on from here
    for (int64_t i = 0; i < M; i++) {
        const int b = items[(size_t) i].member;
        ggml_build_forward_expand(gf, ggml_cpy(ctx, ggml_view_2d(ctx, hn, h.n_embd, 1, hn->nb[1], hn->nb[1] * (size_t) i),
                                               ggml_view_2d(ctx, mtp_chains_, h.n_embd, 1, mtp_chains_->nb[1], mtp_chains_->nb[1] * (size_t) b)));
    }
    ggml_tensor * out  = ggml_mul_mat(ctx, output_, hn);
    ggml_tensor * oids = nullptr;
    if (argmax) {
        out = ggml_argmax(ctx, out);
    } else if (topk > 0) {
        rows_top_k(ctx, out, topk, out, oids);
        ggml_set_output(oids);
        ggml_build_forward_expand(gf, oids);  // (the values do not depend on the concatenated ids)
    }
    ggml_set_output(out);
    ggml_build_forward_expand(gf, out);
    ggml_backend_sched_reset(sched_);
    if (!ggml_backend_sched_alloc_graph(sched_, gf)) {
        ggml_free(ctx);
        err = "draft_mtp_multi: cannot allocate the compute graph";
        return false;
    }
    std::vector<int32_t> tok((size_t) M), pos((size_t) M * 4, 0);
    const ggml_fp16_t    zero = ggml_fp32_to_fp16(0.0f), ninf = ggml_fp32_to_fp16(-INFINITY);
    std::vector<ggml_fp16_t> mk((size_t) Wm_);
    for (int64_t i = 0; i < M; i++) {
        const MtpItem & it = items[(size_t) i];
        const Member &  m  = members_[(size_t) it.member];
        KvSlot &        sl = owner_->slots_[(size_t) m.slot];
        const int       p  = m.n_past - 1 + it.j;
        tok[(size_t) i]    = it.tok;
        pos[(size_t) i] = pos[(size_t) (M + i)] = pos[(size_t) (2 * M + i)] = p;
        const int64_t kv = p % Wm_;
        sl.mtp_slot_pos[(size_t) kv] = p;
        ggml_backend_tensor_set(kvi[(size_t) i], &kv, 0, sizeof kv);
        for (int j = 0; j < Wm_; j++) {
            const int sp = sl.mtp_slot_pos[(size_t) j];
            mk[(size_t) j] = sp >= 0 && sp <= p && sp > p - Wm_ ? zero : ninf;
        }
        ggml_backend_tensor_set(msk[(size_t) i], mk.data(), 0, mk.size() * sizeof(ggml_fp16_t));
    }
    {
        std::vector<float> emb;
        embed(tok.data(), (int) M, emb);
        ggml_backend_tensor_set(inp_e, emb.data(), 0, emb.size() * sizeof(float));
        ggml_backend_tensor_set(inp_p, pos.data(), 0, pos.size() * sizeof(int32_t));
    }
    const bool ok = ggml_backend_sched_graph_compute(sched_, gf) == GGML_STATUS_SUCCESS;
    if (ok) traffic::mtp();
    if (ok) {
        if (argmax) {
            if (ids) ggml_backend_tensor_get(out, ids, 0, ggml_nbytes(out));
        } else {
            if (logits) ggml_backend_tensor_get(out, logits, 0, ggml_nbytes(out));
            if (oids && ids) ggml_backend_tensor_get(oids, ids, 0, ggml_nbytes(oids));
        }
    }
    ggml_free(ctx);
    if (!ok) err = "draft_mtp_multi: graph compute failed";
    return ok;
}

} // namespace e8::model
