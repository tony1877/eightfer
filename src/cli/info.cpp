// `shoehorn info <model.gguf>`: architecture metadata and a tensor summary, without loading weights.

#include "cli/commands.h"
#include "model/gguf_file.h"

#include <cinttypes>
#include <cstdio>
#include <map>
#include <string>

namespace e8::cli {

int info(const std::vector<std::string> & args) {
    if (args.size() < 3) {
        fprintf(stderr, "usage: shoehorn info <model.gguf> [--tensors]\n");
        return 1;
    }
    const bool       all_tensors = args.size() > 3 && args[3] == "--tensors";
    model::GgufFile  f;
    std::string      err;
    if (!f.open(args[2], err)) {
        fprintf(stderr, "%s\n", err.c_str());
        return 1;
    }
    printf("%s  (%.2f GB, %" PRId64 " keys, %" PRId64 " tensors)\n", f.path().c_str(), (double) f.file_size() / 1e9,
           f.n_kv(), f.n_tensors());
    for (int64_t i = 0; i < f.n_kv(); i++) {
        const std::string line = f.describe_kv(i, 16);
        if (line.rfind("tokenizer.", 0) == 0 && line.find("[") != std::string::npos) {
            // vocabularies and merges: just their sizes
            printf("  %s\n", line.substr(0, line.find(']') + 1).c_str());
            continue;
        }
        printf("  %s\n", line.c_str());
    }

    struct Agg {
        int64_t  n     = 0;
        uint64_t bytes = 0;
    };
    std::map<std::string, Agg> by_type;
    uint64_t                   total = 0;
    for (int64_t i = 0; i < f.n_tensors(); i++) {
        const ggml_tensor * t  = f.tensor(i);
        const uint64_t      nb = ggml_nbytes(t);
        Agg &               a  = by_type[ggml_type_name(t->type)];
        a.n++;
        a.bytes += nb;
        total += nb;
        const std::string name = ggml_get_name(t);
        // one full-attention block, one linear-attention block and the non-block tensors by default
        const bool show = all_tensors || name.rfind("blk.", 0) != 0 || name.rfind("blk.0.", 0) == 0 ||
                          name.rfind("blk.3.", 0) == 0;
        if (show) {
            printf("  tensor %-36s %-8s [%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "]  %.1f MB\n",
                   name.c_str(), ggml_type_name(t->type), t->ne[0], t->ne[1], t->ne[2], t->ne[3], (double) nb / 1e6);
        }
    }
    printf("tensor data by type:\n");
    for (const auto & [type, a] : by_type) {
        printf("  %-8s %5" PRId64 " tensors  %8.2f GB\n", type.c_str(), a.n, (double) a.bytes / 1e9);
    }
    printf("  total          %8.2f GB\n", (double) total / 1e9);
    return 0;
}

} // namespace e8::cli
