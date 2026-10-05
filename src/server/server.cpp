// `eightfer serve`: OpenAI-compatible HTTP server (DESIGN.md M6).
//
//   eightfer serve <model.gguf> [--res r.gguf] [--host 127.0.0.1] [--port 8090] [--alias NAME]
//                  [--api-key-file F] [--chat-template-file F] [--ctx N] [--kv f16|q8_0] [--spec auto|K]
//                  [--gpu-layers N] [--expert-cache-gb G] [--threads N]
//
// Endpoints: GET /health, GET /v1/models, POST /v1/chat/completions (stream or not). Chat templating (Jinja, tools,
// chat_template_kwargs such as enable_thinking) and output parsing (reasoning_content, tool_calls) come from
// llama.cpp's common library, so requests and responses look like llama-server's. One request runs at a time.
//
// Prompt reuse: the model state after a request covers prompt + output; a checkpoint is kept at the end of each
// prompt. A new prompt that extends either one only evaluates the new tokens (recurrent state cannot be cut back to an
// arbitrary prefix, so anything else starts over).

#include "cli/commands.h"
#include "model/causal_lm.h"
#include "model/qwen35.h"
#include "model/qwen4exp.h"
#include "runtime/spec.h"

#include "chat.h"
#include "common.h"
#include "llama.h"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <mutex>
#include <random>
#include <sstream>

namespace e8::cli {

namespace {

using json = nlohmann::ordered_json;

struct Server {
    std::unique_ptr<model::CausalLM> model;
    model::Qwen35 *                  q35 = nullptr;  // speculative decoding when it has a residual
    llama_model *                    vocab_model = nullptr;
    const llama_vocab *              vocab = nullptr;
    common_chat_templates_ptr        tmpls;
    std::string                      alias, api_key;
    int                              spec_k = 12;
    bool                             spec_auto = true;
    std::mutex                       mu;
    std::vector<int32_t>             state_tokens, ck_tokens;  // tokens in the model state / at the checkpoint
    std::mt19937_64                  rng{ std::random_device{}() };
};

std::string read_file(const std::string & p) {
    std::ifstream     f(p, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

std::string random_id(std::mt19937_64 & rng, const char * prefix, int n = 24) {
    static const char a[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    std::string       s   = prefix;
    for (int i = 0; i < n; i++) s += a[rng() % (sizeof a - 1)];
    return s;
}

// temperature / top-k / top-p / min-p sampling for the plain (non-speculative) path
int32_t sample(const float * lg, int64_t nv, const runtime::SamplerParams & sp, std::mt19937_64 & rng) {
    if (sp.temp <= 0) return (int32_t) (std::max_element(lg, lg + nv) - lg);
    std::vector<std::pair<float, int32_t>> p((size_t) nv);
    for (int64_t i = 0; i < nv; i++) p[(size_t) i] = { lg[i] / sp.temp, (int32_t) i };
    auto   desc = [](const auto & a, const auto & b) { return a.first > b.first; };
    size_t n    = p.size();
    if (sp.top_k > 0 && (size_t) sp.top_k < n) {
        std::partial_sort(p.begin(), p.begin() + sp.top_k, p.end(), desc);
        n = (size_t) sp.top_k;
    } else {
        std::sort(p.begin(), p.end(), desc);
    }
    p.resize(n);
    const float mx = p[0].first;
    double      sum = 0;
    for (auto & e : p) sum += (e.first = std::exp(e.first - mx));
    for (auto & e : p) e.first = (float) (e.first / sum);
    if (sp.min_p > 0) {
        while (p.size() > 1 && p.back().first < p[0].first * sp.min_p) p.pop_back();
    }
    if (sp.top_p < 1.0f) {
        double c = 0;
        for (size_t i = 0; i < p.size(); i++) {
            if ((c += p[i].first) >= sp.top_p) {
                p.resize(i + 1);
                break;
            }
        }
    }
    double tot = 0;
    for (auto & e : p) tot += e.first;
    double r = std::uniform_real_distribution<double>(0.0, tot)(rng);
    for (auto & e : p) {
        if ((r -= e.first) <= 0) return e.second;
    }
    return p.back().second;
}

json to_nl(const common_json & j) {
    return json::parse(j.dump());
}

struct Result {
    common_chat_msg msg;
    std::string     finish = "stop";
    int             n_prompt = 0, n_gen = 0, n_reused = 0;
    double          t_prompt = 0, t_gen = 0;
};

// Runs one chat request. `on_delta` (streaming) receives OpenAI delta objects as the output grows.
bool run_chat(Server & S, const json & body, Result & R, const std::function<void(const json &)> & on_delta,
              std::string & err) {
    // ---- template
    common_chat_templates_inputs in;
    in.messages = common_chat_msgs_parse_oaicompat(common_json::parse(body.at("messages").dump()));
    if (body.contains("tools") && !body["tools"].is_null()) {
        in.tools = common_chat_tools_parse_oaicompat(common_json::parse(body["tools"].dump()));
    }
    if (body.contains("tool_choice") && body["tool_choice"].is_string()) {
        in.tool_choice = common_chat_tool_choice_parse_oaicompat(body["tool_choice"].get<std::string>());
    }
    in.parallel_tool_calls   = body.value("parallel_tool_calls", false);
    in.add_generation_prompt = true;
    in.use_jinja             = true;
    in.reasoning_format      = COMMON_REASONING_FORMAT_DEEPSEEK;
    if (body.contains("chat_template_kwargs") && body["chat_template_kwargs"].is_object()) {
        for (auto & [k, v] : body["chat_template_kwargs"].items()) in.chat_template_kwargs[k] = v.dump();
    }
    auto et = in.chat_template_kwargs.find("enable_thinking");
    if (et != in.chat_template_kwargs.end()) in.enable_thinking = et->second == "true";
    if (body.value("reasoning_effort", std::string()) == "none") in.enable_thinking = false;
    const common_chat_params cp = common_chat_templates_apply(S.tmpls.get(), in);

    common_chat_parser_params pp(cp);
    pp.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
    pp.parse_tool_calls = true;
    if (!cp.parser.empty()) pp.parser.load(cp.parser);

    std::vector<std::string> stops = cp.additional_stops;
    if (body.contains("stop")) {
        if (body["stop"].is_string()) stops.push_back(body["stop"].get<std::string>());
        else if (body["stop"].is_array())
            for (auto & s : body["stop"]) stops.push_back(s.get<std::string>());
    }
    runtime::SamplerParams sp;
    sp.temp  = body.value("temperature", 1.0f);
    sp.top_p = body.value("top_p", 1.0f);
    sp.top_k = body.value("top_k", 0);
    sp.min_p = body.value("min_p", 0.0f);
    sp.seed  = body.value("seed", (uint64_t) 0);
    int max_tokens = body.value("max_completion_tokens", body.value("max_tokens", -1));

    // ---- prompt, with prefix reuse
    auto & m = *S.model;
    // The checkpoint goes before the generation prompt ("<|im_start|>assistant\n<think>\n"): the next turn re-renders
    // the assistant turn differently, but the conversation before it stays a prefix.
    std::string head = cp.prompt, tail;
    if (!cp.generation_prompt.empty() && head.size() > cp.generation_prompt.size() &&
        head.compare(head.size() - cp.generation_prompt.size(), std::string::npos, cp.generation_prompt) == 0) {
        tail = cp.generation_prompt;
        head.resize(head.size() - tail.size());
    }
    std::vector<int32_t> prompt = common_tokenize(S.vocab, head, /*add_special=*/m.add_bos(), /*parse_special=*/true);
    size_t               n_head = prompt.size();
    if (!tail.empty()) {
        const std::vector<int32_t> t = common_tokenize(S.vocab, tail, false, true);
        prompt.insert(prompt.end(), t.begin(), t.end());
    }
    if (n_head == prompt.size() && n_head > 0) n_head--;  // keep at least one token after the checkpoint
    if (prompt.empty() || (int) prompt.size() >= m.n_ctx() - 1) {
        err = "prompt of " + std::to_string(prompt.size()) + " tokens does not fit the context (" + std::to_string(m.n_ctx()) + ")";
        return false;
    }
    if (max_tokens < 0 || max_tokens > m.n_ctx() - (int) prompt.size() - 16) max_tokens = m.n_ctx() - (int) prompt.size() - 16;
    auto extends = [&](const std::vector<int32_t> & base) {
        return !base.empty() && base.size() < prompt.size() && std::equal(base.begin(), base.end(), prompt.begin());
    };
    size_t start = 0;
    if (extends(S.state_tokens) && (int) S.state_tokens.size() == m.n_past()) {
        start = S.state_tokens.size();
    } else if (extends(S.ck_tokens)) {
        m.checkpoint_restore();
        start = S.ck_tokens.size();
    } else {
        m.reset();
    }
    R.n_prompt = (int) prompt.size();
    R.n_reused = (int) start;
    const auto         t0 = std::chrono::steady_clock::now();
    std::vector<float> last((size_t) m.n_vocab());
    auto eval_range = [&](size_t a, size_t b) {
        for (size_t i = a; i < b; i += 512) {
            const int n = (int) std::min<size_t>(512, b - i);
            if (!m.eval_last(prompt.data() + i, n, last.data(), err)) {
                S.state_tokens.clear();
                S.ck_tokens.clear();
                return false;
            }
        }
        return true;
    };
    if (start < n_head) {
        if (!eval_range(start, n_head)) return false;
        start = n_head;
    }
    if (start == n_head) {
        m.checkpoint_save();
        S.ck_tokens.assign(prompt.begin(), prompt.begin() + (long long) n_head);
    }
    if (!eval_range(start, prompt.size())) return false;
    const auto t1 = std::chrono::steady_clock::now();
    R.t_prompt    = std::chrono::duration<double>(t1 - t0).count();

    // ---- generation
    std::vector<int32_t> out;  // every token produced (some may be past a stop)
    std::string          text;
    common_chat_msg      prev;
    std::vector<std::string> ids_cache;
    auto gen_id = [&]() { return random_id(S.rng, "call_"); };
    bool done   = false;
    size_t used = 0;  // tokens of `out` consumed into `text`
    auto consume = [&]() {
        for (; used < out.size() && !done; used++) {
            const int32_t t = out[used];
            if (llama_vocab_is_eog(S.vocab, t)) {
                done = true;
                break;
            }
            text += common_token_to_piece(S.vocab, t, true);
            R.n_gen++;
            for (const std::string & s : stops) {
                const size_t pos = s.empty() ? std::string::npos : text.find(s);
                if (pos != std::string::npos) {
                    text.resize(pos);
                    done = true;
                    break;
                }
            }
            if (R.n_gen >= max_tokens) {
                R.finish = "length";
                done     = true;
            }
        }
        if (on_delta) {
            try {
                common_chat_msg cur = common_chat_parse(text, !done, pp);
                cur.set_tool_call_ids(ids_cache, gen_id);
                for (const auto & d : common_chat_msg_diff::compute_diffs(prev, cur)) {
                    json delta = json::object();
                    if (!d.reasoning_content_delta.empty()) delta["reasoning_content"] = d.reasoning_content_delta;
                    if (!d.content_delta.empty()) delta["content"] = d.content_delta;
                    if (d.tool_call_index != std::string::npos) {
                        json tc = { { "index", d.tool_call_index } };
                        if (!d.tool_call_delta.id.empty()) {
                            tc["id"]   = d.tool_call_delta.id;
                            tc["type"] = "function";
                        }
                        json fn = json::object();
                        if (!d.tool_call_delta.name.empty()) fn["name"] = d.tool_call_delta.name;
                        if (!d.tool_call_delta.arguments.empty()) fn["arguments"] = d.tool_call_delta.arguments;
                        if (!fn.empty()) tc["function"] = fn;
                        delta["tool_calls"] = json::array({ tc });
                    }
                    on_delta(delta);
                }
                prev = cur;
            } catch (const std::exception &) {
                // partial text the parser cannot handle yet: wait for more tokens
            }
        }
    };

    const bool spec = S.q35 && S.q35->has_residual();
    if (spec) {
        runtime::SpecDecoder dec(*S.q35, S.spec_k, sp, S.spec_auto);
        dec.begin(last.data(), out);
        consume();
        while (!done && S.q35->n_past() + S.spec_k + 2 < m.n_ctx()) {
            if (!dec.step(out, err)) return false;
            consume();
        }
    } else {
        std::mt19937_64 rng(sp.seed ? sp.seed : S.rng());
        out.push_back(sample(last.data(), m.n_vocab(), sp, rng));
        consume();
        while (!done && m.n_past() + 2 < m.n_ctx()) {
            if (!m.eval_last(&out.back(), 1, last.data(), err)) return false;
            out.push_back(sample(last.data(), m.n_vocab(), sp, rng));
            consume();
        }
    }
    if (!done) R.finish = "length";
    // the model state now holds prompt + every produced token except the last (sampled, not evaluated)
    S.state_tokens = prompt;
    S.state_tokens.insert(S.state_tokens.end(), out.begin(), out.end() - 1);
    R.t_gen = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();

    R.msg = common_chat_parse(text, false, pp);
    R.msg.set_tool_call_ids(ids_cache, gen_id);
    if (!R.msg.tool_calls.empty() && R.finish == "stop") R.finish = "tool_calls";
    return true;
}

json usage(const Result & R) {
    return { { "prompt_tokens", R.n_prompt }, { "completion_tokens", R.n_gen }, { "total_tokens", R.n_prompt + R.n_gen },
             { "prompt_tokens_details", { { "cached_tokens", R.n_reused } } } };
}

json timings(const Result & R) {
    return { { "prompt_n", R.n_prompt - R.n_reused }, { "prompt_ms", R.t_prompt * 1e3 },
             { "prompt_per_second", (R.n_prompt - R.n_reused) / std::max(R.t_prompt, 1e-9) }, { "predicted_n", R.n_gen },
             { "predicted_ms", R.t_gen * 1e3 }, { "predicted_per_second", R.n_gen / std::max(R.t_gen, 1e-9) } };
}

} // namespace

int serve(const std::vector<std::string> & args) {
    std::string model_path, res, host = "127.0.0.1", key_file, tmpl_file, kv = "f16";
    int         port = 8090, n_ctx = 16384, gpu_layers = 999, threads = 0;
    int gpu_kv = -1;
    double      cache_gb = -1;
    auto        S = std::make_unique<Server>();
    for (size_t i = 2; i < args.size(); i++) {
        const std::string & a   = args[i];
        auto                val = [&]() -> std::string { return i + 1 < args.size() ? args[++i] : std::string(); };
        if (a == "--res") res = val();
        else if (a == "--host") host = val();
        else if (a == "--port") port = std::atoi(val().c_str());
        else if (a == "--alias") S->alias = val();
        else if (a == "--api-key-file") key_file = val();
        else if (a == "--chat-template-file") tmpl_file = val();
        else if (a == "--ctx") n_ctx = std::atoi(val().c_str());
        else if (a == "--gpu-kv") gpu_kv = std::atoi(val().c_str());
        else if (a == "--kv") kv = val();
        else if (a == "--gpu-layers") gpu_layers = std::atoi(val().c_str());
        else if (a == "--expert-cache-gb") cache_gb = std::atof(val().c_str());
        else if (a == "--threads") threads = std::atoi(val().c_str());
        else if (a == "--spec") {
            const std::string v = val();
            S->spec_auto        = v == "auto";
            S->spec_k           = S->spec_auto ? 12 : std::max(1, std::min(15, std::atoi(v.c_str())));
        } else if (model_path.empty() && a[0] != '-') model_path = a;
        else {
            fprintf(stderr, "unknown option: %s\n", a.c_str());
            return 1;
        }
    }
    if (model_path.empty()) {
        fprintf(stderr, "usage: eightfer serve <model.gguf> [--res r.gguf] [--host H] [--port 8090] [--alias NAME]\n"
                        "         [--api-key-file F] [--chat-template-file F] [--ctx 16384] [--kv f16|q8_0] [--spec auto|K]\n"
                        "         [--gpu-layers N] [--expert-cache-gb G] [--threads N]\n");
        return 1;
    }
    if (!key_file.empty()) {  // last non-comment line (same format as the llama-server key file)
        std::istringstream ks(read_file(key_file));
        for (std::string line; std::getline(ks, line);) {
            while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
            if (!line.empty() && line[0] != '#') S->api_key = line;
        }
    }
    if (S->alias.empty()) {
        S->alias = model_path.substr(model_path.find_last_of("/\\") + 1);
    }

    // tokenizer + chat templates (llama.cpp, vocab only)
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.vocab_only         = true;
    S->vocab_model        = llama_model_load_from_file(model_path.c_str(), mp);
    if (!S->vocab_model) {
        fprintf(stderr, "cannot load the vocabulary from %s\n", model_path.c_str());
        return 1;
    }
    S->vocab = llama_model_get_vocab(S->vocab_model);
    S->tmpls = common_chat_templates_init(S->vocab_model, tmpl_file.empty() ? "" : read_file(tmpl_file));

    model::LoadOptions o;
    o.n_gpu_layers    = gpu_layers;
    o.n_ctx           = n_ctx;
    o.gpu_kv = gpu_kv;
    o.n_threads       = threads;
    o.kv_type         = kv == "q8_0" ? GGML_TYPE_Q8_0 : GGML_TYPE_F16;
    o.residual_path   = res;
    o.expert_cache_gb = cache_gb;
    std::string err;
    S->model = model::load_causal_lm(model_path, o, err);
    if (!S->model) {
        fprintf(stderr, "load failed: %s\n", err.c_str());
        return 1;
    }
    S->q35 = dynamic_cast<model::Qwen35 *>(S->model.get());

    httplib::Server http;
    auto authorized = [&](const httplib::Request & req, httplib::Response & resp) {
        if (S->api_key.empty() || req.get_header_value("Authorization") == "Bearer " + S->api_key) return true;
        resp.status = 401;
        resp.set_content(json{ { "error", { { "message", "invalid API key" }, { "type", "authentication_error" } } } }.dump(),
                         "application/json");
        return false;
    };
    http.Get("/health", [](const httplib::Request &, httplib::Response & resp) {
        resp.set_content(R"({"status":"ok"})", "application/json");
    });
    http.Get("/v1/models", [&](const httplib::Request & req, httplib::Response & resp) {
        if (!authorized(req, resp)) return;
        json m = { { "id", S->alias }, { "object", "model" }, { "owned_by", "eightfer" }, { "created", 0 } };
        resp.set_content(json{ { "object", "list" }, { "data", json::array({ m }) } }.dump(), "application/json");
    });
    http.Post("/v1/chat/completions", [&](const httplib::Request & req, httplib::Response & resp) {
        if (!authorized(req, resp)) return;
        json body;
        try {
            body = json::parse(req.body);
        } catch (const std::exception & e) {
            resp.status = 400;
            resp.set_content(json{ { "error", { { "message", e.what() } } } }.dump(), "application/json");
            return;
        }
        const std::string id      = random_id(S->rng, "chatcmpl-");
        const int64_t     created = (int64_t) std::time(nullptr);
        const bool        stream  = body.value("stream", false);
        auto chunk = [&](const json & delta, const json & finish) {
            return json{ { "id", id }, { "object", "chat.completion.chunk" }, { "created", created }, { "model", S->alias },
                         { "choices", json::array({ { { "index", 0 }, { "delta", delta }, { "finish_reason", finish } } }) } };
        };
        if (!stream) {
            std::lock_guard<std::mutex> lk(S->mu);
            Result      R;
            std::string e;
            bool        ok = false;
            try {
                ok = run_chat(*S, body, R, nullptr, e);
            } catch (const std::exception & ex) {
                e = ex.what();
            }
            if (!ok) {
                resp.status = 400;
                resp.set_content(json{ { "error", { { "message", e } } } }.dump(), "application/json");
                return;
            }
            json msg = to_nl(R.msg.to_json_oaicompat());
            msg["role"] = "assistant";
            json r = { { "id", id }, { "object", "chat.completion" }, { "created", created }, { "model", S->alias },
                       { "choices", json::array({ { { "index", 0 }, { "message", msg }, { "finish_reason", R.finish } } }) },
                       { "usage", usage(R) }, { "timings", timings(R) } };
            resp.set_content(r.dump(), "application/json");
            return;
        }
        // runs after this handler returns: capture by value only
        Server * srv = S.get();
        resp.set_chunked_content_provider("text/event-stream", [srv, body, id, created](size_t, httplib::DataSink & sink) {
            Server & SS = *srv;
            auto chunk = [&](const json & delta, const json & finish) {
                return json{ { "id", id }, { "object", "chat.completion.chunk" }, { "created", created }, { "model", SS.alias },
                             { "choices", json::array({ { { "index", 0 }, { "delta", delta }, { "finish_reason", finish } } }) } };
            };
            std::lock_guard<std::mutex> lk(SS.mu);
            auto send = [&](const json & j) {
                const std::string s = "data: " + j.dump() + "\n\n";
                return sink.write(s.data(), s.size());
            };
            send(chunk({ { "role", "assistant" }, { "content", nullptr } }, nullptr));
            Result      R;
            std::string e;
            bool        ok = false;
            try {
                ok = run_chat(SS, body, R, [&](const json & d) { send(chunk(d, nullptr)); }, e);
            } catch (const std::exception & ex) {
                e = ex.what();
            }
            if (!ok) {
                send(json{ { "error", { { "message", e } } } });
            } else {
                json last     = chunk(json::object(), R.finish);
                last["usage"] = usage(R);
                last["timings"] = timings(R);
                send(last);
            }
            const std::string done = "data: [DONE]\n\n";
            sink.write(done.data(), done.size());
            sink.done();
            return true;
        });
    });

    printf("eightfer serve: %s on http://%s:%d (model id \"%s\", ctx %d, %s)\n", model_path.c_str(), host.c_str(), port,
           S->alias.c_str(), S->model->n_ctx(),
           S->q35 && S->q35->has_residual() ? "base + residual, speculative" : "plain decoding");
    fflush(stdout);
    if (!http.listen(host, port)) {
        fprintf(stderr, "cannot listen on %s:%d\n", host.c_str(), port);
        return 1;
    }
    return 0;
}

} // namespace e8::cli
