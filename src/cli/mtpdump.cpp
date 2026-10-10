// `shoehorn mtpdump <base.gguf> --in conv.jsonl --out data.bin [--topk 32] [--batch 512] [--ctx 8192] [--max-seqs N]`
//
// Training data for the MTP draft head (docs/SPEEDUP-PLAN.md idea 2): what the head reads and what it should predict,
// taken from the 4-bit base, the model whose samples the head's proposals are checked against. Each line of conv.jsonl
// is {"text": "..."} (a whole conversation in the model's chat format; special tokens are parsed). Every sequence is
// read by the base alone (no residual) in batches; for every position t the file gets the token x_t, the trunk's final
// normed hidden state h_t (what the MTP layer reads, f16) and the base's top-k next-token logits at t (best first).
// The head at (h_t, x_{t+1}) should predict the base's distribution at t + 1.
//
// File: "MTPD", u32 version 1, u32 n_embd, u32 topk, then per sequence: u32 n, i32 tokens[n], f16 hidden[n][n_embd],
// i32 topk_ids[n][topk], f32 topk_logits[n][topk].

#include "cli/commands.h"
#include "model/qwen35.h"

#include "common.h"
#include "ggml.h"
#include "llama.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace e8::cli {

int mtpdump(const std::vector<std::string> & args) {
    std::string model, in_path, out_path;
    int         topk = 32, batch = 512, n_ctx = 8192, max_seqs = 1 << 30;
    for (size_t i = 2; i < args.size(); i++) {
        const std::string & a   = args[i];
        auto                val = [&]() -> std::string { return i + 1 < args.size() ? args[++i] : std::string(); };
        if (a == "--in") in_path = val();
        else if (a == "--out") out_path = val();
        else if (a == "--topk") topk = std::atoi(val().c_str());
        else if (a == "--batch") batch = std::atoi(val().c_str());
        else if (a == "--ctx") n_ctx = std::atoi(val().c_str());
        else if (a == "--max-seqs") max_seqs = std::atoi(val().c_str());
        else if (model.empty() && a[0] != '-') model = a;
        else {
            fprintf(stderr, "mtpdump: unknown option %s\n", a.c_str());
            return 1;
        }
    }
    if (model.empty() || in_path.empty() || out_path.empty() || topk < 1 || batch < 1) {
        fprintf(stderr, "usage: shoehorn mtpdump <base.gguf> --in conv.jsonl --out data.bin [--topk 32] [--batch 512] "
                        "[--ctx 8192] [--max-seqs N]\n");
        return 1;
    }

    // tokenizer: the GGUF's vocabulary (as the server loads it)
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.vocab_only         = true;
    llama_model * vm      = llama_model_load_from_file(model.c_str(), mp);
    if (!vm) {
        fprintf(stderr, "mtpdump: cannot read the vocabulary of %s\n", model.c_str());
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(vm);

    model::LoadOptions o;
    o.n_gpu_layers = 999;
    o.n_ctx        = n_ctx;
    o.mtp          = true;  // the hidden states are kept for the MTP layer
    o.n_ubatch     = std::max(o.n_ubatch, batch);
    model::Qwen35 m;
    std::string   err;
    if (!m.load(model, o, err)) {
        fprintf(stderr, "mtpdump: load failed: %s\n", err.c_str());
        return 1;
    }
    if (!m.has_mtp()) {
        fprintf(stderr, "mtpdump: the model has no MTP block on the GPU (hidden states are not kept)\n");
        return 1;
    }
    const int64_t E = m.hp().n_embd;

    std::ifstream in(std::filesystem::u8path(in_path));
    std::ofstream out(std::filesystem::u8path(out_path), std::ios::binary);
    if (!in || !out) {
        fprintf(stderr, "mtpdump: cannot open %s or %s\n", in_path.c_str(), out_path.c_str());
        return 1;
    }
    auto put_u32 = [&](uint32_t v) { out.write((const char *) &v, 4); };
    out.write("MTPD", 4);
    put_u32(1);
    put_u32((uint32_t) E);
    put_u32((uint32_t) topk);

    std::vector<float>       hid, lg;
    std::vector<int32_t>     ids;
    std::vector<ggml_fp16_t> h16;
    int64_t                  n_seq = 0, n_tok = 0;
    const auto               t0    = std::chrono::steady_clock::now();
    for (std::string line; n_seq < max_seqs && std::getline(in, line);) {
        if (line.find_first_not_of(" \t\r") == std::string::npos) continue;
        std::string text;
        try {
            text = nlohmann::json::parse(line).value("text", std::string());
        } catch (...) {
            fprintf(stderr, "mtpdump: skipping a line that is not JSON\n");
            continue;
        }
        std::vector<int32_t> toks = common_tokenize(vocab, text, /*add_special=*/false, /*parse_special=*/true);
        if ((int) toks.size() < 2) continue;
        if ((int) toks.size() > n_ctx) toks.resize((size_t) n_ctx);
        const int n = (int) toks.size();
        m.reset();
        std::vector<ggml_fp16_t> seq_h((size_t) n * E);
        std::vector<int32_t>     seq_ids((size_t) n * topk);
        std::vector<float>       seq_lg((size_t) n * topk);
        for (int p = 0; p < n; p += batch) {
            const int       b = std::min(batch, n - p);
            model::EvalOpts eo;
            eo.residual = false;  // the base: what the drafts are checked against
            eo.topk     = topk;
            lg.resize((size_t) b * topk);
            ids.resize((size_t) b * topk);
            if (!m.eval(toks.data() + p, b, eo, lg.data(), ids.data(), err)) {
                fprintf(stderr, "mtpdump: eval failed: %s\n", err.c_str());
                return 1;
            }
            hid.resize((size_t) b * E);
            if (!m.hidden_rows(b, hid.data(), err)) {
                fprintf(stderr, "mtpdump: %s\n", err.c_str());
                return 1;
            }
            ggml_fp32_to_fp16_row(hid.data(), seq_h.data() + (size_t) p * E, (int64_t) b * E);
            std::copy(ids.begin(), ids.end(), seq_ids.begin() + (size_t) p * topk);
            std::copy(lg.begin(), lg.end(), seq_lg.begin() + (size_t) p * topk);
        }
        put_u32((uint32_t) n);
        out.write((const char *) toks.data(), (std::streamsize) toks.size() * 4);
        out.write((const char *) seq_h.data(), (std::streamsize) seq_h.size() * sizeof(ggml_fp16_t));
        out.write((const char *) seq_ids.data(), (std::streamsize) seq_ids.size() * 4);
        out.write((const char *) seq_lg.data(), (std::streamsize) seq_lg.size() * 4);
        n_seq++;
        n_tok += n;
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        printf("seq %lld: %d tokens (total %lld, %.0f tok/s)\n", (long long) n_seq, n, (long long) n_tok, n_tok / s);
        fflush(stdout);
    }
    out.close();
    printf("wrote %lld sequences, %lld tokens to %s\n", (long long) n_seq, (long long) n_tok, out_path.c_str());
    llama_model_free(vm);
    return 0;
}

} // namespace e8::cli
