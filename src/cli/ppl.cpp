// `eightfer ppl`: perplexity over a pre-tokenized text, computed the way llama.cpp's llama-perplexity does it
// (tools/perplexity, default mode): the tokens are cut into n_ctx-sized chunks, each chunk starts from an empty
// state (with BOS in front when the model adds one), and only the second half of each chunk is scored. The
// output format matches, so the two can be compared chunk by chunk.

#include "cli/commands.h"
#include "model/causal_lm.h"
#include "model/qwen35.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace e8::cli {

namespace {

// All integers in the file, in order: accepts llama-tokenize's `--ids` list ("[1, 2, 3]") or one id per line.
bool read_tokens(const std::string & path, std::vector<int32_t> & out, std::string & err) {
    std::ifstream f(std::filesystem::u8path(path), std::ios::binary);
    if (!f) {
        err = "cannot open " + path;
        return false;
    }
    std::stringstream ss;
    ss << f.rdbuf();
    const std::string s = ss.str();
    for (size_t i = 0; i < s.size();) {
        if (s[i] >= '0' && s[i] <= '9') {
            char * end = nullptr;
            out.push_back((int32_t) std::strtol(s.c_str() + i, &end, 10));
            i = (size_t) (end - s.c_str());
        } else {
            i++;
        }
    }
    if (out.empty()) {
        err = path + ": no token ids found";
        return false;
    }
    return true;
}

// llama-perplexity --kl-divergence-base file: "_logits_", n_ctx, n_vocab, n_chunk, the tokens, then per chunk and
// scored position nv = 2*((n_vocab+1)/2)+4 uint16: two floats (scale, min_log_prob) and n_vocab quantized log-probs.
struct KldBase {
    std::ifstream        in;
    int32_t              n_ctx = 0, n_vocab = 0, n_chunk = 0;
    std::vector<int32_t> tokens;
    int                  nv = 0;

    bool open(const std::string & path, std::string & err) {
        in.open(std::filesystem::u8path(path), std::ios::binary);
        char magic[8] = {};
        if (!in || !in.read(magic, 8) || std::string(magic, 8) != "_logits_" ||
            !in.read((char *) &n_ctx, 4) || !in.read((char *) &n_vocab, 4) || !in.read((char *) &n_chunk, 4)) {
            err = path + ": not a llama-perplexity logits file";
            return false;
        }
        tokens.resize((size_t) n_chunk * n_ctx);
        if (!in.read((char *) tokens.data(), (std::streamsize) (tokens.size() * 4))) {
            err = path + ": truncated";
            return false;
        }
        nv = 2 * ((n_vocab + 1) / 2) + 4;
        return true;
    }

    // Reads one scored position's base log-probs into lp (n_vocab floats).
    bool next(std::vector<float> & lp) {
        std::vector<uint16_t> row((size_t) nv);
        if (!in.read((char *) row.data(), (std::streamsize) (row.size() * 2))) {
            return false;
        }
        float scale = 0, min_lp = 0;
        std::memcpy(&scale, row.data(), 4);
        std::memcpy(&min_lp, row.data() + 2, 4);
        lp.resize((size_t) n_vocab);
        for (int v = 0; v < n_vocab; v++) {
            lp[(size_t) v] = min_lp + scale * (float) row[(size_t) v + 4];
        }
        return true;
    }
};

ggml_type kv_type(const std::string & s) {
    if (s == "q8_0") return GGML_TYPE_Q8_0;
    if (s == "q4_0") return GGML_TYPE_Q4_0;
    if (s == "f32") return GGML_TYPE_F32;
    return GGML_TYPE_F16;
}

} // namespace

int ppl(const std::vector<std::string> & args) {
    std::string model, tokens_path, kld_path, res_path, kv = "f16";
    int         n_ctx = 512, n_chunks = -1, gpu_layers = 0, threads = 0, n_batch = 512;
    int gpu_kv = -1;
    for (size_t i = 2; i < args.size(); i++) {
        const std::string & k   = args[i];
        auto                val = [&]() -> std::string { return i + 1 < args.size() ? args[++i] : std::string(); };
        if (k == "--tokens") tokens_path = val();
        else if (k == "--kld-base") kld_path = val();
        else if (k == "--ctx") n_ctx = std::atoi(val().c_str());
        else if (k == "--gpu-kv") gpu_kv = std::atoi(val().c_str());
        else if (k == "--chunks") n_chunks = std::atoi(val().c_str());
        else if (k == "--gpu-layers") gpu_layers = std::atoi(val().c_str());
        else if (k == "--threads") threads = std::atoi(val().c_str());
        else if (k == "--batch") n_batch = std::atoi(val().c_str());
        else if (k == "--kv") kv = val();
        else if (k == "--res") res_path = val();
        else if (model.empty() && k[0] != '-') model = k;
        else {
            fprintf(stderr, "unknown option: %s\n", k.c_str());
            return 1;
        }
    }
    if (model.empty() || (tokens_path.empty() && kld_path.empty()) || n_ctx < 16 || n_batch < 1) {
        fprintf(stderr, "usage: eightfer ppl <model.gguf> --tokens <ids.txt> [--ctx 512] [--chunks N] [--gpu-layers N]\n"
                        "                    [--batch 512] [--kv f16|q8_0|q4_0] [--threads N] [--res <pack .res.gguf>]\n"
                        "       eightfer ppl <model.gguf> --kld-base <llama-perplexity logits file> [...]\n"
                        "                    (tokens and ctx from the file; adds KL divergence vs llama.cpp)\n");
        return 1;
    }

    std::string          err;
    std::vector<int32_t> toks;
    KldBase              base;
    if (!kld_path.empty()) {
        if (!base.open(kld_path, err)) {
            fprintf(stderr, "%s\n", err.c_str());
            return 1;
        }
        toks  = base.tokens;
        n_ctx = base.n_ctx;
    } else if (!read_tokens(tokens_path, toks, err)) {
        fprintf(stderr, "%s\n", err.c_str());
        return 1;
    }
    const int max_chunks = (int) (toks.size() / (size_t) n_ctx);
    if (max_chunks < 1) {
        fprintf(stderr, "need at least %d tokens, have %zu\n", n_ctx, toks.size());
        return 1;
    }
    n_chunks = n_chunks < 0 ? max_chunks : std::min(n_chunks, max_chunks);

    model::LoadOptions o;
    o.n_gpu_layers = gpu_layers;
    o.n_ctx        = n_ctx;
    o.gpu_kv = gpu_kv;
    o.n_threads    = threads;
    o.kv_type      = kv_type(kv);
    o.n_ubatch     = std::min(n_batch, n_ctx);
    o.residual_path = res_path;
    if (const char * ec = std::getenv("E8_EXPERT_CACHE_GB")) o.expert_cache_gb = std::atof(ec);
    o.experts_gpu = std::getenv("E8_EXPERTS_GPU") != nullptr;
    std::unique_ptr<model::CausalLM> mp;
    const auto    t_load = std::chrono::steady_clock::now();
    if (!(mp = model::load_causal_lm(model, o, err))) {
        fprintf(stderr, "load failed: %s\n", err.c_str());
        return 1;
    }
    model::CausalLM & m = *mp;
    struct {
        int64_t n_layer, n_vocab, bos;
        bool    add_bos;
    } hp = { m.n_layer(), m.n_vocab(), m.bos_token(), m.add_bos() };
    fprintf(stderr, "loaded in %.1f s: %lld layers (%d on GPU), weights %.2f GB GPU + %.2f GB CPU, vocab %lld, add_bos %d\n",
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t_load).count(), (long long) hp.n_layer,
            gpu_layers, (double) m.gpu_weight_bytes() / 1e9, (double) m.cpu_weight_bytes() / 1e9, (long long) hp.n_vocab,
            hp.add_bos ? 1 : 0);
    if (auto * q = dynamic_cast<model::Qwen35 *>(&m); q && q->has_residual()) {
        fprintf(stderr, "residual: %.2f GB (host)\n", (double) q->residual_bytes() / 1e9);
    }
    fprintf(stderr, "perplexity: %zu tokens, %d chunks of %d, batch %d\n", toks.size(), n_chunks, n_ctx, o.n_ubatch);

    const int          first = n_ctx / 2;
    // one batch of logits: positions are scored as their batch completes (a whole long chunk would not fit in RAM)
    std::vector<float> logits((size_t) hp.n_vocab * o.n_ubatch);
    double             nll = 0, nll2 = 0;
    int64_t            count = 0;
    // KL divergence vs the base, bucketed by position in the chunk (8 buckets over the scored half)
    const bool          kld = !kld_path.empty();
    constexpr int       kBuckets = 8;
    double              kl_sum[kBuckets] = {}, kl_max = 0, base_nll = 0;
    int64_t             kl_n[kBuckets] = {}, top_same = 0, kl_small = 0;
    int                 kl_max_chunk = 0, kl_max_pos = 0;
    std::vector<float>  blp;
    const auto         t0    = std::chrono::steady_clock::now();
    for (int c = 0; c < n_chunks; c++) {
        const size_t         start = (size_t) c * n_ctx;
        std::vector<int32_t> batch(toks.begin() + (long) start, toks.begin() + (long) (start + n_ctx));
        if (hp.add_bos && hp.bos >= 0) {
            batch[0] = (int32_t) hp.bos;
        }
        m.reset();
        for (int b = 0; b < n_ctx; b += o.n_ubatch) {
            const int n = std::min(o.n_ubatch, n_ctx - b);
            if (!m.eval(batch.data() + b, n, b + n > first ? logits.data() : nullptr, err)) {
                fprintf(stderr, "\n%s\n", err.c_str());
                return 1;
            }
        for (int i = std::max(first, b); i < std::min(b + n, n_ctx - 1); i++) {
            const float * row = logits.data() + (size_t) (i - b) * hp.n_vocab;
            float         mx  = row[0];
            for (int64_t v = 1; v < hp.n_vocab; v++) {
                mx = std::max(mx, row[v]);
            }
            double sum = 0;
            for (int64_t v = 0; v < hp.n_vocab; v++) {
                sum += std::exp((double) row[v] - mx);
            }
            const double lse = mx + std::log(sum);
            const double lp  = (double) row[toks[start + (size_t) i + 1]] - lse;
            nll -= lp;
            nll2 += lp * lp;
            count++;
            if (kld) {
                if (!base.next(blp)) {
                    fprintf(stderr, "\nlogits file ended early\n");
                    return 1;
                }
                double  kl = 0;
                int64_t a_base = 0, a_e8 = 0;
                for (int64_t v = 0; v < hp.n_vocab; v++) {
                    // as llama-perplexity: entries at the file's floor (max - 16) are not real probabilities
                    if (blp[(size_t) v] > -16.0f) {
                        const double pb = std::exp((double) blp[(size_t) v]);
                        kl += pb * ((double) blp[(size_t) v] - ((double) row[v] - lse));
                    }
                    if (blp[(size_t) v] > blp[(size_t) a_base]) a_base = v;
                    if (row[v] > row[a_e8]) a_e8 = v;
                }
                base_nll -= blp[(size_t) toks[start + (size_t) i + 1]];
                top_same += a_base == a_e8;
                kl_small += kl < 0.01;
                const int bk = (i - first) * kBuckets / (n_ctx - first);
                kl_sum[bk] += kl;
                kl_n[bk]++;
                if (kl > kl_max) {
                    kl_max       = kl;
                    kl_max_chunk = c + 1;
                    kl_max_pos   = i;
                }
            }
        }
        }
        if (c == 0) {
            const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            fprintf(stderr, "%.2f seconds per pass - ETA %.1f minutes\n", s, s * n_chunks / 60.0);
        }
        printf("[%d]%.4f,", c + 1, std::exp(nll / (double) count));
        fflush(stdout);
    }
    const double mean = nll / (double) count;
    const double var  = nll2 / (double) count - mean * mean;
    const double unc  = std::sqrt(std::max(0.0, var) / (double) (count - 1));
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    printf("\nFinal estimate: PPL = %.4f +/- %.5f\n", std::exp(mean), std::exp(mean) * unc);
    if (kld) {
        double tot = 0;
        for (int b = 0; b < kBuckets; b++) {
            tot += kl_sum[b];
        }
        printf("vs llama.cpp: base PPL %.4f, mean KLD %.6f, max KLD %.4f (chunk %d pos %d), same top-1 %.2f%%\n",
               std::exp(base_nll / (double) count), tot / (double) count, kl_max, kl_max_chunk, kl_max_pos,
               100.0 * (double) top_same / (double) count);
        printf("tokens with KLD < 0.01: %.2f%%\n", 100.0 * (double) kl_small / (double) count);
        printf("mean KLD by position:");
        for (int b = 0; b < kBuckets; b++) {
            printf("  %d-%d: %.5f", first + b * (n_ctx - first) / kBuckets, first + (b + 1) * (n_ctx - first) / kBuckets - 1,
                   kl_n[b] ? kl_sum[b] / (double) kl_n[b] : 0.0);
        }
        printf("\n");
    }
    fprintf(stderr, "%lld tokens scored, %.1f s, %.1f tokens/s evaluated\n", (long long) count, secs,
            (double) n_chunks * n_ctx / secs);
    return 0;
}

} // namespace e8::cli
