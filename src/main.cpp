#include "bench/bench.h"
#include "cli/commands.h"
#include "sys/sysinfo.h"

#include "ggml.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

// ggml logs every repacked tensor and device detail at DEBUG/INFO; keep warnings and errors
// unless EIGHTFER_VERBOSE is set.
static void log_filter(ggml_log_level level, const char * text, void *) {
    static const bool verbose = std::getenv("EIGHTFER_VERBOSE") != nullptr;
    static int        last    = GGML_LOG_LEVEL_NONE;
    if (level != GGML_LOG_LEVEL_CONT) {
        last = level;
    }
    if (verbose || last >= GGML_LOG_LEVEL_WARN) {
        fputs(text, stderr);
    }
}

static void usage() {
    printf("eightfer %s\n\n"
           "usage: eightfer bench [options]\n"
           "  --only LIST     run only these sections, comma-separated: cpu,gpu,pcie,disk\n"
           "  --disk PATH     drive/folder (a temp file is written there) or existing big file; repeatable.\n"
           "                  Default on Windows: every fixed drive.\n"
           "  --disk-gb N     temp file size per drive (default 4)\n"
           "  --pinned-gb N   pinned host memory to try to allocate (default 16; on Windows at most\n"
           "                  half of RAM minus 1 GiB, and always leaving 4 GiB of RAM free)\n"
           "  --threads N     CPU threads for GEMV (default: physical cores)\n"
           "  --seconds S     duration of each measurement (default 1.5)\n",
           EIGHTFER_VERSION);
}

static bool parse_bench(const std::vector<std::string> & a, e8::bench::Options & o) {
    for (size_t i = 2; i < a.size(); i++) {
        const std::string & k   = a[i];
        auto                val = [&](const char * name) -> const char * {
            if (i + 1 >= a.size()) {
                fprintf(stderr, "missing value for %s\n", name);
                return nullptr;
            }
            return a[++i].c_str();
        };
        const char * v = nullptr;
        if (k == "--only") {
            if (!(v = val("--only"))) {
                return false;
            }
            const std::string s = std::string(",") + v + ",";
            o.cpu               = s.find(",cpu,") != std::string::npos;
            o.gpu               = s.find(",gpu,") != std::string::npos;
            o.pcie              = s.find(",pcie,") != std::string::npos;
            o.disk              = s.find(",disk,") != std::string::npos;
        } else if (k == "--disk") {
            if (!(v = val("--disk"))) {
                return false;
            }
            o.disks.push_back(v);
        } else if (k == "--disk-gb") {
            if (!(v = val("--disk-gb"))) {
                return false;
            }
            o.disk_gib = std::atof(v);
        } else if (k == "--pinned-gb") {
            if (!(v = val("--pinned-gb"))) {
                return false;
            }
            o.pinned_gib = std::atof(v);
        } else if (k == "--threads") {
            if (!(v = val("--threads"))) {
                return false;
            }
            o.threads = std::atoi(v);
        } else if (k == "--seconds") {
            if (!(v = val("--seconds"))) {
                return false;
            }
            o.seconds = std::atof(v);
        } else {
            fprintf(stderr, "unknown option: %s\n", k.c_str());
            return false;
        }
    }
    if (o.disk_gib <= 0 || o.seconds <= 0) {
        fprintf(stderr, "--disk-gb and --seconds must be > 0\n");
        return false;
    }
    return true;
}

int main(int argc, char ** argv) {
    ggml_log_set(log_filter, nullptr);
    const std::vector<std::string> args = e8::sys::utf8_args(argc, argv);
    if (args.size() < 2 || args[1] == "-h" || args[1] == "--help") {
        usage();
        return 0;
    }
    if (args[1] == "--version") {
        printf("eightfer %s\n", EIGHTFER_VERSION);
        return 0;
    }
    if (args[1] == "pinned-probe" && args.size() == 3) {  // internal: run by `bench` in a child process
        e8::bench::pinned_probe(std::atof(args[2].c_str()));
    }
    if (args[1] == "info") {
        return e8::cli::info(args);
    }
    if (args[1] == "ppl") {
        return e8::cli::ppl(args);
    }
    if (args[1] == "serve") {
        return e8::cli::serve(args);
    }
    if (args[1] == "decode") {
        return e8::cli::decode(args);
    }
    if (args[1] == "dump") {
        return e8::cli::dump(args);
    }
    if (args[1] == "gen") {
        return e8::cli::gen(args);
    }
    if (args[1] == "pack") {
        return e8::cli::pack(args);
    }
    if (args[1] == "logits") {
        return e8::cli::logits(args);
    }
    if (args[1] == "selftest") {
        return e8::cli::selftest(args);
    }
    if (args[1] == "bench") {
        e8::bench::Options opt;
        if (!parse_bench(args, opt)) {
            usage();
            return 1;
        }
        return e8::bench::run(opt);
    }
    usage();
    return 1;
}
