// `shoehorn pack`: builds the split-precision pair for a qwen35 model from its Hugging Face BF16 safetensors.
//
//   shoehorn pack --src <hf dir> --template <any qwen35 GGUF of the same model> --out <prefix>
//                 [--base iq4_xs] [--res q4_k|q3_k|q5_k|q6_k|none] [--threads N] [--check]
//
// Writes <prefix>.base.gguf (a complete, normal GGUF: every big matrix quantized to the base type, small tensors F32,
// token embedding BF16; loadable by shoehorn and llama.cpp) and <prefix>.res.gguf (for each big matrix, the residual
// W - deq(B) quantized to the residual type, same tensor names).
//
// The template GGUF supplies metadata (hyperparameters, tokenizer, chat template) and the tensor list, names and
// shapes. Its tensor data is only read with --check, which compares every converted source tensor against the
// template's dequantized values: a wrong name mapping or permutation shows up as a relative error near 1.
//
// HF -> GGUF conversion mirrors llama.cpp's convert_hf_to_gguf.py (Qwen3_5TextModel at the pinned commit):
// norm weights get +1 (except linear_attn.norm), A_log becomes -exp(A_log), conv1d is squeezed, and the DeltaNet
// value heads are reordered from grouped-by-key-head to tiled order (in_proj_qkv V rows, in_proj_z/a/b rows,
// A_log/dt_bias elements, conv1d V channels, out_proj columns).

#include "cli/commands.h"
#include "model/gguf_file.h"
#include "model/safetensors.h"

#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

namespace e8::cli {

namespace {

namespace fs = std::filesystem;

struct Source {
    std::string hf;              // HF tensor name
    enum Op { None, Plus1, NegExp } op = None;
    int64_t row_perm_from = -1;  // rows >= this index are V heads to reorder (-1: no row reorder)
    int64_t row_perm_hd   = 0;   // head size of those rows
    int64_t col_perm_hd   = 0;   // > 0: columns are V heads of this size, reorder them
};

struct Dims {
    int64_t n_layer = 0, hk = 0, hv = 0, s = 0;
};

// grouped [k][v_per_k][hd] -> tiled [v_per_k][k][hd]: source index of destination index d
int64_t reorder_src(int64_t d, int64_t hd, const Dims & m) {
    const int64_t vpk = m.hv / m.hk;
    const int64_t e = d % hd, t = d / hd, vi = t / m.hk, ki = t % m.hk;
    return (ki * vpk + vi) * hd + e;
}

bool map_name(const std::string & g, const Dims & m, Source & s) {
    auto ends = [&](const char * suf) {
        const size_t n = strlen(suf);
        return g.size() >= n && g.compare(g.size() - n, n, suf) == 0;
    };
    if (g == "token_embd.weight") { s.hf = "model.language_model.embed_tokens.weight"; return true; }
    if (g == "output_norm.weight") { s.hf = "model.language_model.norm.weight"; s.op = Source::Plus1; return true; }
    if (g == "output.weight") { s.hf = "lm_head.weight"; return true; }
    if (g.rfind("blk.", 0) != 0) return false;
    const size_t  dot = g.find('.', 4);
    const int64_t il  = std::atoll(g.substr(4, dot - 4).c_str());
    const std::string rest = g.substr(dot + 1);
    const bool        mtp  = il >= m.n_layer;
    const std::string p    = mtp ? "mtp.layers." + std::to_string(il - m.n_layer) + "."
                                 : "model.language_model.layers." + std::to_string(il) + ".";
    struct R { const char * g; const char * hf; Source::Op op; };
    static const R table[] = {
        { "attn_norm.weight", "input_layernorm.weight", Source::Plus1 },
        { "post_attention_norm.weight", "post_attention_layernorm.weight", Source::Plus1 },
        { "attn_q.weight", "self_attn.q_proj.weight", Source::None },
        { "attn_k.weight", "self_attn.k_proj.weight", Source::None },
        { "attn_v.weight", "self_attn.v_proj.weight", Source::None },
        { "attn_output.weight", "self_attn.o_proj.weight", Source::None },
        { "attn_q_norm.weight", "self_attn.q_norm.weight", Source::Plus1 },
        { "attn_k_norm.weight", "self_attn.k_norm.weight", Source::Plus1 },
        { "ffn_gate.weight", "mlp.gate_proj.weight", Source::None },
        { "ffn_up.weight", "mlp.up_proj.weight", Source::None },
        { "ffn_down.weight", "mlp.down_proj.weight", Source::None },
        { "attn_qkv.weight", "linear_attn.in_proj_qkv.weight", Source::None },
        { "attn_gate.weight", "linear_attn.in_proj_z.weight", Source::None },
        { "ssm_alpha.weight", "linear_attn.in_proj_a.weight", Source::None },
        { "ssm_beta.weight", "linear_attn.in_proj_b.weight", Source::None },
        { "ssm_conv1d.weight", "linear_attn.conv1d.weight", Source::None },
        { "ssm_dt.bias", "linear_attn.dt_bias", Source::None },
        { "ssm_a", "linear_attn.A_log", Source::NegExp },
        { "ssm_norm.weight", "linear_attn.norm.weight", Source::None },
        { "ssm_out.weight", "linear_attn.out_proj.weight", Source::None },
    };
    if (mtp) {
        if (rest == "nextn.eh_proj.weight") { s.hf = "mtp.fc.weight"; return true; }
        if (rest == "nextn.enorm.weight") { s.hf = "mtp.pre_fc_norm_embedding.weight"; s.op = Source::Plus1; return true; }
        if (rest == "nextn.hnorm.weight") { s.hf = "mtp.pre_fc_norm_hidden.weight"; s.op = Source::Plus1; return true; }
        if (rest == "nextn.shared_head_norm.weight") { s.hf = "mtp.norm.weight"; s.op = Source::Plus1; return true; }
    }
    for (const R & r : table) {
        if (rest != r.g) continue;
        s.hf = p + r.hf;
        s.op = r.op;
        if (m.hk != m.hv && m.hk > 0) {
            const int64_t qk = 2 * m.hk * m.s;
            if (ends("attn_qkv.weight")) { s.row_perm_from = qk; s.row_perm_hd = m.s; }
            else if (ends("attn_gate.weight")) { s.row_perm_from = 0; s.row_perm_hd = m.s; }
            else if (ends("ssm_alpha.weight") || ends("ssm_beta.weight")) { s.row_perm_from = 0; s.row_perm_hd = 1; }
            else if (ends("ssm_dt.bias") || ends("ssm_a")) { s.col_perm_hd = 1; }
            else if (ends("ssm_conv1d.weight")) { s.row_perm_from = qk; s.row_perm_hd = m.s; }
            else if (ends("ssm_out.weight")) { s.col_perm_hd = m.s; }
        }
        return true;
    }
    return false;
}

ggml_type parse_type(const std::string & s) {
    for (int t = 0; t < GGML_TYPE_COUNT; t++) {
        const char * n = ggml_type_name((ggml_type) t);
        if (n && s == n) return (ggml_type) t;
    }
    return GGML_TYPE_COUNT;
}

bool write_all(FILE * f, const void * p, size_t n) {
    return fwrite(p, 1, n, f) == n;
}

bool pad_to(FILE * f, size_t n, size_t align) {
    static const uint8_t zeros[64] = {};
    size_t pad = (align - n % align) % align;
    while (pad > 0) {
        const size_t k = std::min(pad, sizeof(zeros));
        if (!write_all(f, zeros, k)) return false;
        pad -= k;
    }
    return true;
}

FILE * open_w(const std::string & path) {
#if defined(_WIN32)
    return _wfopen(fs::u8path(path).wstring().c_str(), L"wb");
#else
    return fopen(path.c_str(), "wb");
#endif
}

FILE * open_r(const std::string & path) {
#if defined(_WIN32)
    return _wfopen(fs::u8path(path).wstring().c_str(), L"rb");
#else
    return fopen(path.c_str(), "rb");
#endif
}

struct Plan {
    std::string name;
    Source      src;
    const model::StTensor * st = nullptr;
    int64_t     ne[4] = { 1, 1, 1, 1 };
    int         n_dims = 1;
    ggml_type   base_type = GGML_TYPE_F32;
    bool        split = false;  // has a residual
    const ggml_tensor * tmpl = nullptr;
};

// llama.cpp imatrix GGUF (llama-imatrix output): "<weight>.in_sum2" [ne0] and "<weight>.counts" [1] per weight.
// Importance per input column = in_sum2 / count, as llama-quantize uses it.
bool load_imatrix(const std::string & path, std::map<std::string, std::vector<float>> & out, std::string & err) {
    ggml_context *   ctx = nullptr;
    gguf_init_params gp  = { false, &ctx };
    gguf_context *   g   = gguf_init_from_file(path.c_str(), gp);
    if (!g) {
        err = "cannot read imatrix " + path;
        return false;
    }
    std::map<std::string, std::pair<ggml_tensor *, ggml_tensor *>> sc;
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
        std::string n = t->name;
        if (n.size() > 8 && n.compare(n.size() - 8, 8, ".in_sum2") == 0) sc[n.substr(0, n.size() - 8)].first = t;
        else if (n.size() > 7 && n.compare(n.size() - 7, 7, ".counts") == 0) sc[n.substr(0, n.size() - 7)].second = t;
    }
    for (auto & [name, p] : sc) {
        if (!p.first || !p.second || p.first->type != GGML_TYPE_F32 || ggml_nelements(p.second) != 1) {
            continue;  // MoE (per-expert) entries are not used for dense models
        }
        const float        cnt = ((const float *) p.second->data)[0];
        const int64_t      n   = ggml_nelements(p.first);
        std::vector<float> v((size_t) n, 1.0f);
        if (cnt > 0) {
            for (int64_t i = 0; i < n; i++) v[(size_t) i] = ((const float *) p.first->data)[i] / cnt;
        }
        out[name] = std::move(v);
    }
    gguf_free(g);
    ggml_free(ctx);
    return true;
}

struct Stats {
    double w2 = 0, tmpl_e2 = 0, b_e2 = 0, br_e2 = 0;
};

} // namespace

int pack(const std::vector<std::string> & args) {
    std::string src_dir, tmpl_path, out, imatrix_path;
    std::string base_s = "iq4_xs", res_s = "q4_K";
    int         threads = (int) std::max(1u, std::thread::hardware_concurrency());
    bool        check   = false;
    for (size_t i = 2; i < args.size(); i++) {
        const std::string & k   = args[i];
        auto                val = [&]() -> std::string { return i + 1 < args.size() ? args[++i] : std::string(); };
        if (k == "--src") src_dir = val();
        else if (k == "--template") tmpl_path = val();
        else if (k == "--out") out = val();
        else if (k == "--base") base_s = val();
        else if (k == "--res") res_s = val();
        else if (k == "--threads") threads = std::atoi(val().c_str());
        else if (k == "--check") check = true;
        else if (k == "--imatrix") imatrix_path = val();
        else {
            fprintf(stderr, "unknown option %s\n", k.c_str());
            return 1;
        }
    }
    if (src_dir.empty() || tmpl_path.empty() || (out.empty() && !check)) {
        fprintf(stderr, "usage: shoehorn pack --src <hf dir> --template <qwen35.gguf> --out <prefix>\n"
                        "                     [--base iq4_xs] [--res q4_K|q3_K|q5_K|q6_K|none] [--threads N] [--check]\n"
                        "                     [--imatrix <llama-imatrix .gguf>]\n"
                        "  --check without --out only verifies the conversion against the template\n");
        return 1;
    }
    for (auto * s : { &base_s, &res_s }) {
        if (s->size() == 4 && (*s)[0] == 'q' && (*s)[2] == '_' && ((*s)[3] == 'k')) (*s)[3] = 'K';
    }
    const ggml_type base_t = parse_type(base_s);
    const bool      has_res = res_s != "none";
    const ggml_type res_t   = has_res ? parse_type(res_s) : GGML_TYPE_COUNT;
    if (base_t == GGML_TYPE_COUNT || (has_res && res_t == GGML_TYPE_COUNT)) {
        fprintf(stderr, "unknown quant type (use ggml names, e.g. iq4_xs, q4_K)\n");
        return 1;
    }
    ggml_quantize_init(base_t);
    if (has_res) ggml_quantize_init(res_t);
    const int64_t qblk = std::max(ggml_blck_size(base_t), has_res ? ggml_blck_size(res_t) : 1);

    std::string err;
    model::GgufFile tmpl;
    if (!tmpl.open(tmpl_path, err)) {
        fprintf(stderr, "template: %s\n", err.c_str());
        return 1;
    }
    if (tmpl.arch() != "qwen35") {
        fprintf(stderr, "template architecture is %s, expected qwen35\n", tmpl.arch().c_str());
        return 1;
    }
    std::map<std::string, std::vector<float>> imatrix;
    if (!imatrix_path.empty()) {
        if (!load_imatrix(imatrix_path, imatrix, err)) {
            fprintf(stderr, "%s\n", err.c_str());
            return 1;
        }
        printf("imatrix: %zu entries from %s\n", imatrix.size(), imatrix_path.c_str());
    }
    model::Safetensors st;
    if (!st.open(src_dir, err)) {
        fprintf(stderr, "source: %s\n", err.c_str());
        return 1;
    }
    Dims dm;
    dm.n_layer = tmpl.i64("qwen35.block_count", 0) - tmpl.i64("qwen35.nextn_predict_layers", 0);
    dm.hk      = tmpl.i64("qwen35.ssm.group_count", 0);
    dm.hv      = tmpl.i64("qwen35.ssm.time_step_rank", 0);
    dm.s       = tmpl.i64("qwen35.ssm.state_size", 0);

    // plan every template tensor
    std::vector<Plan> plans;
    uint64_t          src_bytes = 0;
    for (int64_t i = 0; i < tmpl.n_tensors(); i++) {
        const ggml_tensor * t = tmpl.tensor(i);
        Plan                p;
        p.name = ggml_get_name(t);
        p.tmpl = t;
        if (!map_name(p.name, dm, p.src)) {
            fprintf(stderr, "no source mapping for template tensor %s\n", p.name.c_str());
            return 1;
        }
        p.st = st.find(p.src.hf);
        if (!p.st && p.src.hf.rfind("model.language_model.", 0) == 0) {
            // text-only checkpoints (ForCausalLM) have no language_model level
            p.src.hf = "model." + p.src.hf.substr(strlen("model.language_model."));
            p.st     = st.find(p.src.hf);
        }
        if (!p.st) {
            fprintf(stderr, "source tensor %s (for %s) not found\n", p.src.hf.c_str(), p.name.c_str());
            return 1;
        }
        p.n_dims = ggml_n_dims(t);
        for (int d = 0; d < 4; d++) p.ne[d] = t->ne[d];
        if (p.st->n_elements() != ggml_nelements(t) || p.st->row_len() != t->ne[0]) {
            // conv1d is [C, 1, K] in HF: row length K matches ne0 after the squeeze; everything else must match
            if (!(p.st->n_elements() == ggml_nelements(t) && p.st->shape.back() == t->ne[0])) {
                fprintf(stderr, "shape mismatch for %s\n", p.name.c_str());
                return 1;
            }
        }
        if (p.name == "token_embd.weight") {
            p.base_type = GGML_TYPE_BF16;
        } else if (p.n_dims == 2 && t->ne[0] % qblk == 0 && t->ne[1] >= 256) {
            p.base_type = base_t;
            p.split     = has_res;
        } else {
            p.base_type = GGML_TYPE_F32;
        }
        src_bytes += p.st->nbytes;
        plans.push_back(p);
    }
    printf("pack: %zu tensors, %.2f GB source, base %s, residual %s, %d threads%s\n", plans.size(), src_bytes / 1e9,
           ggml_type_name(base_t), has_res ? ggml_type_name(res_t) : "none", threads, check ? ", checking vs template" : "");

    // output GGUF headers (metadata copied from the template; tensor infos in template order)
    const bool       write = !out.empty();
    gguf_context *   gb = nullptr, * gr = nullptr;
    ggml_context *   meta = nullptr;
    FILE *           fb = nullptr, * fr = nullptr;
    size_t           align_b = 32, align_r = 32;
    uint64_t         out_b = 0, out_r = 0;
    if (write) {
        ggml_init_params ip = { ggml_tensor_overhead() * (plans.size() * 2 + 8), nullptr, true };
        meta                = ggml_init(ip);
        gb                  = gguf_init_empty();
        gguf_set_kv(gb, tmpl.raw());
        for (int64_t k = gguf_get_n_kv(gb) - 1; k >= 0; k--) {
            const std::string key = gguf_get_key(gb, k);
            if (key.rfind("quantize.", 0) == 0) gguf_remove_key(gb, key.c_str());
        }
        std::mt19937_64 rng(std::random_device{}());
        char            id[32];
        snprintf(id, sizeof id, "%016llx", (unsigned long long) rng());
        const auto        fn = fs::u8path(src_dir).filename().u8string();
        const std::string src_name(fn.begin(), fn.end());
        gguf_set_val_u32(gb, "general.file_type", base_t == GGML_TYPE_IQ4_XS ? 30u : 0u);
        gguf_set_val_str(gb, "shoehorn.pack.id", id);
        gguf_set_val_str(gb, "shoehorn.pack.base_type", ggml_type_name(base_t));
        gguf_set_val_str(gb, "shoehorn.pack.residual_type", has_res ? ggml_type_name(res_t) : "none");
        gguf_set_val_str(gb, "shoehorn.pack.source", src_name.c_str());
        if (has_res) {
            gr = gguf_init_empty();
            gguf_set_val_str(gr, "general.architecture", "qwen35");
            gguf_set_val_str(gr, "general.type", "shoehorn-residual");
            gguf_set_val_str(gr, "shoehorn.pack.id", id);
            gguf_set_val_str(gr, "shoehorn.pack.base_type", ggml_type_name(base_t));
            gguf_set_val_str(gr, "shoehorn.pack.residual_type", ggml_type_name(res_t));
        }
        for (const Plan & p : plans) {
            ggml_tensor * t = ggml_new_tensor(meta, p.base_type, p.n_dims, p.ne);
            ggml_set_name(t, p.name.c_str());
            gguf_add_tensor(gb, t);
            if (p.split) {
                ggml_tensor * r = ggml_new_tensor(meta, res_t, p.n_dims, p.ne);
                ggml_set_name(r, p.name.c_str());
                gguf_add_tensor(gr, r);
            }
        }
        align_b = gguf_get_alignment(gb);
        fb      = open_w(out + ".base.gguf");
        if (!fb) {
            fprintf(stderr, "cannot write %s.base.gguf\n", out.c_str());
            return 1;
        }
        std::vector<uint8_t> m(gguf_get_meta_size(gb));
        gguf_get_meta_data(gb, m.data());
        write_all(fb, m.data(), m.size());
        if (gr) {
            align_r = gguf_get_alignment(gr);
            fr      = open_w(out + ".res.gguf");
            if (!fr) {
                fprintf(stderr, "cannot write %s.res.gguf\n", out.c_str());
                return 1;
            }
            std::vector<uint8_t> m2(gguf_get_meta_size(gr));
            gguf_get_meta_data(gr, m2.data());
            write_all(fr, m2.data(), m2.size());
        }
    }
    FILE * ft = check ? open_r(tmpl_path) : nullptr;
    std::mutex tmpl_io;

    Stats       total;
    double      worst_tmpl = 0;
    std::string worst_name;
    const auto  t_start = std::chrono::steady_clock::now();
    for (size_t pi = 0; pi < plans.size(); pi++) {
        const Plan &  p      = plans[pi];
        const int64_t ncols  = p.ne[0];
        const int64_t nrows  = ggml_nelements(p.tmpl) / ncols;
        const size_t  brow   = ggml_row_size(p.base_type, ncols);
        const size_t  rrow   = p.split ? ggml_row_size(res_t, ncols) : 0;
        std::vector<uint8_t> bdata(write ? brow * (size_t) nrows : 0), rdata(write ? rrow * (size_t) nrows : 0);
        std::vector<int64_t> colsrc;
        if (p.src.col_perm_hd > 0) {
            colsrc.resize((size_t) ncols);
            for (int64_t c = 0; c < ncols; c++) colsrc[(size_t) c] = reorder_src(c, p.src.col_perm_hd, dm);
        }
        auto src_row = [&](int64_t r) {
            if (p.src.row_perm_from < 0 || r < p.src.row_perm_from) return r;
            return p.src.row_perm_from + reorder_src(r - p.src.row_perm_from, p.src.row_perm_hd, dm);
        };
        const int64_t rows_per_blk = std::max<int64_t>(1, (1 << 20) / ncols);
        const int64_t n_blk        = (nrows + rows_per_blk - 1) / rows_per_blk;
        std::atomic<int64_t> next{ 0 };
        std::atomic<bool>    failed{ false };
        std::mutex           stat_mu;
        Stats                ts;
        const ggml_type_traits * tt_b = ggml_get_type_traits(p.base_type);
        const ggml_type_traits * tt_r = p.split ? ggml_get_type_traits(res_t) : nullptr;
        const ggml_type_traits * tt_t = ggml_get_type_traits(p.tmpl->type);
        const float *            imat = nullptr;
        if (auto it = imatrix.find(p.name); it != imatrix.end() && (int64_t) it->second.size() == ncols) {
            imat = it->second.data();
        }

        auto worker = [&]() {
            std::vector<float>   w, tmp, res, perm, tq;
            std::vector<uint8_t> traw;
            std::string          e;
            for (int64_t bi; (bi = next.fetch_add(1)) < n_blk && !failed;) {
                const int64_t r0 = bi * rows_per_blk, nr = std::min(rows_per_blk, nrows - r0);
                const size_t  ne = (size_t) (nr * ncols);
                w.resize(ne);
                // read source rows, coalescing runs that are contiguous in the source
                for (int64_t r = 0; r < nr;) {
                    const int64_t s0 = src_row(r0 + r);
                    int64_t       n  = 1;
                    while (r + n < nr && src_row(r0 + r + n) == s0 + n) n++;
                    if (!st.read_rows(*p.st, s0, n, w.data() + (size_t) (r * ncols), e)) {
                        fprintf(stderr, "%s\n", e.c_str());
                        failed = true;
                        return;
                    }
                    r += n;
                }
                if (!colsrc.empty()) {
                    perm.resize((size_t) ncols);
                    for (int64_t r = 0; r < nr; r++) {
                        float * row = w.data() + (size_t) (r * ncols);
                        for (int64_t c = 0; c < ncols; c++) perm[(size_t) c] = row[colsrc[(size_t) c]];
                        std::memcpy(row, perm.data(), sizeof(float) * (size_t) ncols);
                    }
                }
                if (p.src.op == Source::Plus1) for (float & x : w) x += 1.0f;
                if (p.src.op == Source::NegExp) for (float & x : w) x = -std::exp(x);

                Stats ls;
                for (float x : w) ls.w2 += (double) x * x;
                if (ft) {
                    const size_t trow = ggml_row_size(p.tmpl->type, ncols);
                    traw.resize(trow * (size_t) nr);
                    {
                        std::lock_guard<std::mutex> lk(tmpl_io);
                        const uint64_t off = tmpl.data_offset(p.tmpl) + trow * (uint64_t) r0;
#if defined(_WIN32)
                        _fseeki64(ft, (long long) off, SEEK_SET);
#else
                        fseeko(ft, (off_t) off, SEEK_SET);
#endif
                        if (fread(traw.data(), 1, traw.size(), ft) != traw.size()) {
                            fprintf(stderr, "template read failed for %s\n", p.name.c_str());
                            failed = true;
                            return;
                        }
                    }
                    tq.resize(ne);
                    if (p.tmpl->type == GGML_TYPE_F32) std::memcpy(tq.data(), traw.data(), ne * 4);
                    else tt_t->to_float(traw.data(), tq.data(), (int64_t) ne);
                    for (size_t i = 0; i < ne; i++) ls.tmpl_e2 += (double) (w[i] - tq[i]) * (w[i] - tq[i]);
                }
                if (write || p.split) {
                    uint8_t * bd = write ? bdata.data() + brow * (size_t) r0 : nullptr;
                    std::vector<uint8_t> local;
                    if (!bd) {
                        local.resize(brow * (size_t) nr);
                        bd = local.data();
                    }
                    if (p.base_type == GGML_TYPE_F32) std::memcpy(bd, w.data(), ne * 4);
                    else if (p.base_type == GGML_TYPE_BF16) ggml_fp32_to_bf16_row_ref(w.data(), (ggml_bf16_t *) bd, (int64_t) ne);
                    else ggml_quantize_chunk(p.base_type, w.data(), bd, 0, nr, ncols, imat);
                    if (p.split) {
                        tmp.resize(ne);
                        tt_b->to_float(bd, tmp.data(), (int64_t) ne);
                        res.resize(ne);
                        for (size_t i = 0; i < ne; i++) {
                            res[i] = w[i] - tmp[i];
                            ls.b_e2 += (double) res[i] * res[i];
                        }
                        std::vector<uint8_t> rl;
                        uint8_t * rd = write ? rdata.data() + rrow * (size_t) r0 : nullptr;
                        if (!rd) {
                            rl.resize(rrow * (size_t) nr);
                            rd = rl.data();
                        }
                        ggml_quantize_chunk(res_t, res.data(), rd, 0, nr, ncols, imat);
                        tt_r->to_float(rd, tmp.data(), (int64_t) ne);
                        for (size_t i = 0; i < ne; i++) {
                            const double d = (double) res[i] - tmp[i];
                            ls.br_e2 += d * d;
                        }
                    }
                }
                std::lock_guard<std::mutex> lk(stat_mu);
                ts.w2 += ls.w2;
                ts.tmpl_e2 += ls.tmpl_e2;
                ts.b_e2 += ls.b_e2;
                ts.br_e2 += ls.br_e2;
            }
        };
        std::vector<std::thread> pool;
        for (int i = 0; i < std::max(1, threads); i++) pool.emplace_back(worker);
        for (auto & th : pool) th.join();
        if (failed) return 1;

        if (write) {
            if (!write_all(fb, bdata.data(), bdata.size()) || !pad_to(fb, bdata.size(), align_b)) {
                fprintf(stderr, "write failed (disk full?)\n");
                return 1;
            }
            out_b += bdata.size();
            if (p.split) {
                if (!write_all(fr, rdata.data(), rdata.size()) || !pad_to(fr, rdata.size(), align_r)) {
                    fprintf(stderr, "write failed (disk full?)\n");
                    return 1;
                }
                out_r += rdata.size();
            }
        }
        const double rel_t = ts.w2 > 0 ? std::sqrt(ts.tmpl_e2 / ts.w2) : 0;
        if (check && rel_t > worst_tmpl) {
            worst_tmpl = rel_t;
            worst_name = p.name;
        }
        if (p.split) {
            total.w2 += ts.w2;
            total.b_e2 += ts.b_e2;
            total.br_e2 += ts.br_e2;
        }
        const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
        printf("[%4zu/%zu] %-36s %-7s%s", pi + 1, plans.size(), p.name.c_str(), ggml_type_name(p.base_type), imat ? "*" : " ");
        if (p.split) printf(" +%-5s B %.4f B+R %.4f", ggml_type_name(res_t), std::sqrt(ts.b_e2 / ts.w2), std::sqrt(ts.br_e2 / ts.w2));
        if (check) printf("  vs template(%s) %.4f%s", ggml_type_name(p.tmpl->type), rel_t, rel_t > 0.1 ? "  <-- MISMATCH" : "");
        printf("  %.0fs\n", el);
        fflush(stdout);
    }
    if (fb) fclose(fb);
    if (fr) fclose(fr);
    if (ft) fclose(ft);
    if (gb) gguf_free(gb);
    if (gr) gguf_free(gr);
    if (meta) ggml_free(meta);

    printf("\nbig matrices: rel. RMSE base %.4f, base+residual %.4f\n", std::sqrt(total.b_e2 / std::max(total.w2, 1e-30)),
           std::sqrt(total.br_e2 / std::max(total.w2, 1e-30)));
    if (check) printf("worst template mismatch: %.4f (%s)%s\n", worst_tmpl, worst_name.c_str(), worst_tmpl > 0.1 ? "  FAIL" : "  OK");
    if (write) printf("wrote %s.base.gguf (%.2f GB data)%s\n", out.c_str(), out_b / 1e9,
                      has_res ? (" and " + out + ".res.gguf (" + std::to_string(out_r / 1e9).substr(0, 5) + " GB data)").c_str() : "");
    return check && worst_tmpl > 0.1 ? 2 : 0;
}

} // namespace e8::cli
