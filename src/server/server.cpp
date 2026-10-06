// `eightfer serve`: OpenAI-compatible HTTP server (DESIGN.md M6).
//
//   eightfer serve <model.gguf> [--res r.gguf] [--host 127.0.0.1] [--port 8090] [--alias NAME]
//                  [--api-key-file F] [--chat-template-file F] [--ctx N] [--kv f16|q8_0] [--spec auto|K]
//                  [--gpu-layers N] [--expert-cache-gb G] [--threads N] [--mtp N]
//
// Endpoints: GET /health, GET /v1/models, POST /v1/chat/completions and POST /v1/completions (stream or not),
// POST /unload (free the model; the next request loads it again). Several models (--also ALIAS=PATH[,RES]): the request's
// "model" picks one; it is loaded on demand and the previous one freed (one model in memory at a time). Chat
// templating (Jinja, tools, chat_template_kwargs such as enable_thinking) and output parsing (reasoning_content,
// tool_calls) come from llama.cpp's common library, so requests and responses look like llama-server's. Sampling:
// temperature, top_p, top_k, min_p, seed, presence_penalty, frequency_penalty, stop, max_tokens.
//
// Concurrency: requests are accepted in parallel and run one at a time, first come first served (the model holds one
// sequence). A request whose client disconnects stops at the next decode step, so it does not hold up the queue.
//
// Sharing the GPU (--idle-unload SEC, --unload-router URL): the model loads on the first request and is freed after SEC
// idle seconds; before loading, every model loaded by a llama-server router at URL is unloaded (same API key), so the
// two servers take turns on the GPU. The router is also proxied: /v1/models lists its models too, and a completion
// request naming one of them frees this model and is forwarded to the router (which loads it), so a client that only
// talks to this server can switch between all models. Requests of both kinds take turns.
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
#include "ggml-backend.h"
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
#include <condition_variable>
#include <mutex>
#include <random>
#include <sstream>
#include <thread>

namespace e8::cli {

namespace {

using json = nlohmann::ordered_json;

// a model this server can load: tokenizer and chat templates stay loaded, the weights only while it is active
struct Entry {
    std::string               alias, path, res;
    llama_model *             vocab_model = nullptr;
    const llama_vocab *       vocab       = nullptr;
    common_chat_templates_ptr tmpls;
};

struct Server {
    std::unique_ptr<model::CausalLM> model;
    model::Qwen35 *                  q35 = nullptr;  // speculative decoding when it has a residual
    std::vector<Entry>               entries;        // [0] = the main model (positional path, --alias, --res)
    int                              active = -1;    // entry whose weights are loaded
    const llama_vocab *              vocab  = nullptr;  // the active entry's
    common_chat_templates *          tmpl   = nullptr;
    std::string                      alias, api_key;
    int                              spec_k = 12;
    bool                             spec_auto = true;
    int                              mtp = 3;  // MTP proposals per base pass while drafting (0 = off)
    bool                             echo = true;  // echo drafting (copies from the context)
    // first-come-first-served turn taking: each request takes a ticket and runs when `serving` reaches it
    std::mutex                       mu;
    std::condition_variable          cv;
    uint64_t                         next_ticket = 0, serving = 0;
    std::vector<int32_t>             state_tokens, ck_tokens;  // tokens in the model state / at the checkpoint
    std::function<bool(int, std::string &)> load;  // loads entry i (lazy mode)
    std::string                      router_url;  // llama-server router to unload / forward to (empty = none)
    std::string                      timing_log;  // JSON lines of per-request timings (no content), when set
    std::chrono::system_clock::time_point last_end{};  // end of the previous request (idle time = tools / user)
    // sampling defaults for requests that do not set them (llama-server's --temperature/--top-k/... ; Qwen's thinking set)
    runtime::SamplerParams           defaults = [] {
        runtime::SamplerParams d;
        d.temp  = 1.0f;
        d.top_p = 0.95f;
        d.top_k = 20;
        d.min_p = 0.0f;
        return d;
    }();
    std::chrono::steady_clock::time_point last_used = std::chrono::steady_clock::now();
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

// holds the model for one request; waits for the requests that arrived earlier
class Turn {
public:
    explicit Turn(Server & S) : S_(S) {
        std::unique_lock<std::mutex> lk(S_.mu);
        const uint64_t t = S_.next_ticket++;
        S_.cv.wait(lk, [&] { return S_.serving == t; });
    }
    ~Turn() {
        {
            std::lock_guard<std::mutex> lk(S_.mu);
            S_.serving++;
        }
        S_.cv.notify_all();
    }
private:
    Server & S_;
};

// frees the model (the next request of ours loads it again); the caller holds a Turn
void unload_model(Server & S) {
    if (!S.model) return;
    S.q35 = nullptr;
    S.model.reset();
    S.active = -1;
    S.state_tokens.clear();
    S.ck_tokens.clear();
    fprintf(stderr, "model unloaded\n");
}

json to_nl(const common_json & j) {
    return json::parse(j.dump());
}

struct Result {
    std::string     spec;  // speculative decoding summary (log)
    json            stats = json::object();  // numbers for the timing log
    common_chat_msg msg;   // chat requests
    std::string     text;  // completion requests
    std::string     finish = "stop";
    int             n_prompt = 0, n_gen = 0, n_reused = 0;
    double          t_prompt = 0, t_gen = 0;
};

// Runs one request: chat (body has "messages") or completion ("prompt"). `on_delta` (streaming) receives OpenAI delta
// objects as the output grows (chat: {content, reasoning_content, tool_calls}; completion: {text}) and returns false
// when the client is gone, which stops generation (as does `cancelled`).
bool run_request(Server & S, const json & body, Result & R, const std::function<bool(const json &)> & on_delta,
                 const std::function<bool()> & cancelled, std::string & err) {
    int idx = 0;  // the entry this request names (the main model when it names none of ours)
    if (body.contains("model") && body["model"].is_string()) {
        for (size_t i = 0; i < S.entries.size(); i++) {
            if (S.entries[i].alias == body["model"].get<std::string>()) idx = (int) i;
        }
    }
    if (S.active != idx || !S.model) {
        unload_model(S);
        if (!S.load(idx, err)) return false;
        S.active = idx;
    }
    S.vocab = S.entries[(size_t) idx].vocab;
    S.tmpl  = S.entries[(size_t) idx].tmpls.get();
    S.last_used     = std::chrono::steady_clock::now();
    auto &     m    = *S.model;
    const bool chat = body.contains("messages");
    runtime::SamplerParams sp;
    const auto & dflt    = S.defaults;
    sp.temp              = body.value("temperature", dflt.temp);
    sp.top_p             = body.value("top_p", dflt.top_p);
    sp.top_k             = body.value("top_k", dflt.top_k);
    sp.min_p             = body.value("min_p", dflt.min_p);
    sp.seed              = body.value("seed", (uint64_t) 0);
    sp.presence_penalty  = body.value("presence_penalty", dflt.presence_penalty);
    sp.frequency_penalty = body.value("frequency_penalty", dflt.frequency_penalty);
    int max_tokens = body.value("max_completion_tokens", body.value("max_tokens", chat ? -1 : 16));
    std::vector<std::string> stops;
    if (body.contains("stop")) {
        if (body["stop"].is_string()) stops.push_back(body["stop"].get<std::string>());
        else if (body["stop"].is_array())
            for (auto & s : body["stop"]) stops.push_back(s.get<std::string>());
    }

    std::vector<int32_t>      prompt;
    size_t                    n_head = 0;  // the prompt-reuse checkpoint goes after this many tokens
    common_chat_parser_params pp;
    if (!chat) {
        // ---- completion: the prompt as given (a string, or token ids), no template
        const json & pr = body.contains("prompt") ? body["prompt"] : json("");
        if (pr.is_array() && !pr.empty() && pr[0].is_number_integer()) {
            for (auto & t : pr) prompt.push_back(t.get<int32_t>());
        } else {
            const std::string text = pr.is_array() ? (pr.empty() ? std::string() : pr[0].get<std::string>()) : pr.get<std::string>();
            prompt = common_tokenize(S.vocab, text, /*add_special=*/m.add_bos(), /*parse_special=*/true);
        }
        n_head = prompt.empty() ? 0 : prompt.size() - 1;
    } else {
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
    if (body.contains("reasoning_effort") && body["reasoning_effort"].is_string()) {
        // as llama-server: the effort goes to the chat template ("none" turns thinking off)
        const std::string re = body["reasoning_effort"].get<std::string>();
        if (re == "none") in.enable_thinking = false;
        else in.chat_template_kwargs["reasoning_effort"] = json(re).dump();
    }
    const common_chat_params cp = common_chat_templates_apply(S.tmpl, in);

    pp                  = common_chat_parser_params(cp);
    pp.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
    pp.parse_tool_calls = true;
    if (!cp.parser.empty()) pp.parser.load(cp.parser);

    stops.insert(stops.end(), cp.additional_stops.begin(), cp.additional_stops.end());

    // The checkpoint goes before the generation prompt ("<|im_start|>assistant\n<think>\n"): the next turn re-renders
    // the assistant turn differently, but the conversation before it stays a prefix.
    std::string head = cp.prompt, tail;
    if (!cp.generation_prompt.empty() && head.size() > cp.generation_prompt.size() &&
        head.compare(head.size() - cp.generation_prompt.size(), std::string::npos, cp.generation_prompt) == 0) {
        tail = cp.generation_prompt;
        head.resize(head.size() - tail.size());
    }
    prompt = common_tokenize(S.vocab, head, /*add_special=*/m.add_bos(), /*parse_special=*/true);
    n_head = prompt.size();
    if (!tail.empty()) {
        const std::vector<int32_t> t = common_tokenize(S.vocab, tail, false, true);
        prompt.insert(prompt.end(), t.begin(), t.end());
    }
    if (n_head == prompt.size() && n_head > 0) n_head--;  // keep at least one token after the checkpoint
    }

    // ---- prompt, with prefix reuse
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
        if (b > a && !m.prefill(prompt.data() + a, (int) (b - a), last.data(), err)) {
            S.state_tokens.clear();
            S.ck_tokens.clear();
            return false;
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
    runtime::TokenCounts counts;  // plain path: penalties
    std::string          text;
    common_chat_msg      prev;
    std::vector<std::string> ids_cache;
    auto gen_id = [&]() { return random_id(S.rng, "call_"); };
    bool done   = false;
    size_t used = 0;  // tokens of `out` consumed into `text`
    size_t sent = 0;  // completion streaming: bytes of `text` sent
    size_t hold = 0;  // completion streaming: a stop string may still be forming in the last hold bytes
    for (const std::string & s : stops) hold = std::max(hold, s.empty() ? 0 : s.size() - 1);
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
        if (cancelled && cancelled()) {
            R.finish = "cancelled";
            done     = true;
        }
        if (on_delta && !chat) {
            const size_t upto = done ? text.size() : (text.size() > hold ? text.size() - hold : 0);
            if (upto > sent) {
                // never split a UTF-8 character
                size_t e = upto;
                while (!done && e > sent && (text[e] & 0xC0) == 0x80) e--;
                if (e > sent && !on_delta(json{ { "text", text.substr(sent, e - sent) } })) {
                    R.finish = "cancelled";
                    done     = true;
                }
                sent = e;
            }
        } else if (on_delta) {
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
                    if (!on_delta(delta)) {
                        R.finish = "cancelled";
                        done     = true;
                    }
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
        dec.set_mtp(S.mtp);
        dec.set_echo(S.echo);
        dec.set_context(prompt);
        dec.begin(last.data(), out);
        consume();
        while (!done && S.q35->n_past() + S.spec_k + 2 < m.n_ctx()) {
            if (!dec.step(out, err)) return false;
            consume();
        }
        const auto & st = dec.stats();
        R.stats         = { { "cycles", st.cycles }, { "draft_s", st.t_draft }, { "verify_s", st.t_verify },
                            { "echo_proposed", st.echo_proposed }, { "echo_kept", st.echo_accepted },
                            { "mtp_proposed", st.mtp_proposed }, { "mtp_kept", st.mtp_accepted },
                            { "long_cycles", st.long_cycles }, { "reruns", st.reruns } };
        char         b[160];
        snprintf(b, sizeof b, "%lld cycles, %.1f tokens/cycle, draft %.0f / verify %.0f ms per cycle, echo %lld/%lld kept",
                 (long long) st.cycles, st.cycles ? (double) (st.emitted - 1) / st.cycles : 0.0,
                 st.cycles ? 1e3 * st.t_draft / st.cycles : 0.0, st.cycles ? 1e3 * st.t_verify / st.cycles : 0.0,
                 (long long) st.echo_accepted, (long long) st.echo_proposed);
        R.spec = b;
    } else {
        std::mt19937_64 rng(sp.seed ? sp.seed : S.rng());
        auto            next = [&]() {
            runtime::apply_penalties(last.data(), sp, counts);
            out.push_back(sample(last.data(), m.n_vocab(), sp, rng));
            counts[out.back()]++;
        };
        next();
        consume();
        while (!done && m.n_past() + 2 < m.n_ctx()) {
            if (!m.eval_last(&out.back(), 1, last.data(), err)) return false;
            next();
            consume();
        }
    }
    if (!done) R.finish = "length";
    // the model state now holds prompt + every produced token except the last (sampled, not evaluated)
    S.state_tokens = prompt;
    S.state_tokens.insert(S.state_tokens.end(), out.begin(), out.end() - 1);
    R.t_gen = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();

    S.last_used = std::chrono::steady_clock::now();
    if (!S.timing_log.empty()) {  // numbers only: never prompt or output text
        const auto now_sys = std::chrono::system_clock::now();
        const auto t0_sys  = now_sys - std::chrono::duration_cast<std::chrono::system_clock::duration>(
                                          std::chrono::duration<double>(R.t_prompt + R.t_gen));
        json line = { { "t", (double) std::chrono::duration_cast<std::chrono::milliseconds>(now_sys.time_since_epoch()).count() / 1e3 },
                      { "model", S.entries[(size_t) S.active].alias }, { "kind", chat ? "chat" : "completion" },
                      { "prompt_tokens", R.n_prompt }, { "reused_tokens", R.n_reused }, { "prefill_s", R.t_prompt },
                      { "gen_tokens", R.n_gen }, { "decode_s", R.t_gen }, { "finish", R.finish },
                      { "idle_before_s", S.last_end.time_since_epoch().count() == 0 ? -1.0
                                         : std::chrono::duration<double>(t0_sys - S.last_end).count() },
                      { "temp", sp.temp }, { "spec", R.stats } };
        S.last_end = now_sys;
        if (FILE * f = fopen(S.timing_log.c_str(), "ab")) {
            const std::string l = line.dump() + "\n";
            fwrite(l.data(), 1, l.size(), f);
            fclose(f);
        }
    }
    fprintf(stderr, "%s: prompt %d tokens (%d reused) in %.1f s, %d generated at %.1f tok/s (temp %.2f top_k %d top_p %.2f)%s%s\n",
            S.entries[(size_t) S.active].alias.c_str(), R.n_prompt, R.n_reused, R.t_prompt, R.n_gen,
            R.n_gen / std::max(R.t_gen, 1e-9), sp.temp, sp.top_k, sp.top_p, R.spec.empty() ? "" : "; ", R.spec.c_str());
    if (!chat) {
        R.text = text;
        return true;
    }
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
    int gpu_kv = -1, idle_unload = 0;
    std::string router_url;
    std::vector<Entry> extra;
    double      cache_gb = -1;
    auto        S = std::make_unique<Server>();
    for (size_t i = 2; i < args.size(); i++) {
        const std::string & a   = args[i];
        auto                val = [&]() -> std::string { return i + 1 < args.size() ? args[++i] : std::string(); };
        if (a == "--res") res = val();
        else if (a == "--host") host = val();
        else if (a == "--port") port = std::atoi(val().c_str());
        else if (a == "--alias") S->alias = val();
        else if (a == "--also") {  // ALIAS=PATH[,RES]
            const std::string v = val();
            const size_t      eq = v.find('='), cm = v.find(',', eq == std::string::npos ? 0 : eq);
            if (eq == std::string::npos) {
                fprintf(stderr, "--also needs ALIAS=PATH[,RES]\n");
                return 1;
            }
            Entry e;
            e.alias = v.substr(0, eq);
            e.path  = v.substr(eq + 1, cm == std::string::npos ? std::string::npos : cm - eq - 1);
            if (cm != std::string::npos) e.res = v.substr(cm + 1);
            extra.push_back(std::move(e));
        }
        else if (a == "--api-key-file") key_file = val();
        else if (a == "--chat-template-file") tmpl_file = val();
        else if (a == "--ctx") n_ctx = std::atoi(val().c_str());
        else if (a == "--gpu-kv") gpu_kv = std::atoi(val().c_str());
        else if (a == "--kv") kv = val();
        else if (a == "--gpu-layers") gpu_layers = std::atoi(val().c_str());
        else if (a == "--expert-cache-gb") cache_gb = std::atof(val().c_str());
        else if (a == "--threads") threads = std::atoi(val().c_str());
        else if (a == "--mtp") S->mtp = std::atoi(val().c_str());
        else if (a == "--echo") S->echo = std::atoi(val().c_str()) != 0;
        else if (a == "--idle-unload") idle_unload = std::atoi(val().c_str());
        else if (a == "--temperature" || a == "--temp") S->defaults.temp = (float) std::atof(val().c_str());
        else if (a == "--top-p") S->defaults.top_p = (float) std::atof(val().c_str());
        else if (a == "--top-k") S->defaults.top_k = std::atoi(val().c_str());
        else if (a == "--min-p") S->defaults.min_p = (float) std::atof(val().c_str());
        else if (a == "--presence-penalty") S->defaults.presence_penalty = (float) std::atof(val().c_str());
        else if (a == "--unload-router") router_url = val();
        else if (a == "--timing-log") S->timing_log = val();
        else if (a == "--spec") {
            const std::string v = val();
            S->spec_auto        = v == "auto";
            S->spec_k           = S->spec_auto ? runtime::SpecDecoder::kMaxK : std::max(1, std::min(runtime::SpecDecoder::kMaxK, std::atoi(v.c_str())));
        } else if (model_path.empty() && a[0] != '-') model_path = a;
        else {
            fprintf(stderr, "unknown option: %s\n", a.c_str());
            return 1;
        }
    }
    if (model_path.empty()) {
        fprintf(stderr, "usage: eightfer serve <model.gguf> [--res r.gguf] [--host H] [--port 8090] [--alias NAME]\n"
                        "         [--api-key-file F] [--chat-template-file F] [--ctx 16384] [--kv f16|q8_0] [--spec auto|K]\n"
                        "         [--gpu-layers N] [--expert-cache-gb G] [--threads N] [--mtp N (0 = off)]\n"
                        "         [--idle-unload SEC (load on demand, free after SEC idle)] [--unload-router URL]\n"
                        "         [--temperature 1.0] [--top-p 0.95] [--top-k 20] [--min-p 0] [--presence-penalty 0] (request defaults)\n"
                        "         [--timing-log FILE (JSON lines of per-request timings, no content)]\n");
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
    {
        Entry main;
        main.alias = S->alias;
        main.path  = model_path;
        main.res   = res;
        S->entries.push_back(std::move(main));
        for (auto & e : extra) S->entries.push_back(std::move(e));
    }
    for (auto & e : S->entries) {  // one chat template file (if given) for every model
        e.vocab_model = llama_model_load_from_file(e.path.c_str(), mp);
        if (!e.vocab_model) {
            fprintf(stderr, "cannot load the vocabulary from %s\n", e.path.c_str());
            return 1;
        }
        e.vocab = llama_model_get_vocab(e.vocab_model);
        e.tmpls = common_chat_templates_init(e.vocab_model, tmpl_file.empty() ? "" : read_file(tmpl_file));
    }

    model::LoadOptions o;
    o.n_gpu_layers    = gpu_layers;
    o.n_ctx           = n_ctx;
    o.gpu_kv = gpu_kv;
    o.n_threads       = threads;
    o.kv_type         = kv == "q8_0" ? GGML_TYPE_Q8_0 : GGML_TYPE_F16;
    o.residual_path   = res;
    o.expert_cache_gb = cache_gb;
    o.max_record      = S->spec_k + 1;
    o.mtp             = S->mtp > 0;
    S->router_url = router_url;
    Server * sp = S.get();
    S->load     = [sp, o, router_url](int idx, std::string & e) {
        bool unloaded = false;
        if (!router_url.empty()) {  // free the GPU: unload whatever the llama-server router has loaded
            httplib::Client  rc(router_url);
            httplib::Headers hd = { { "Authorization", "Bearer " + sp->api_key } };
            std::vector<std::string> asked;
            for (int tries = 0; tries < 60; tries++) {
                auto r = rc.Get("/v1/models", hd);
                if (!r || r->status != 200) {  // no router: nothing to free
                    fprintf(stderr, "router %s: %s\n", router_url.c_str(),
                            r ? ("HTTP " + std::to_string(r->status)).c_str() : httplib::to_string(r.error()).c_str());
                    break;
                }
                std::vector<std::string> busy;
                try {
                    const json list = json::parse(r->body);
                    for (auto & m : list.at("data")) {
                        const std::string st = m.contains("status") ? m["status"].value("value", std::string()) : std::string();
                        if (st == "loaded" || st == "loading") busy.push_back(m.value("id", std::string()));
                    }
                } catch (const std::exception &) {
                    break;
                }
                if (busy.empty()) break;
                for (auto & id : busy) {  // once per model; the router takes ~15 s to stop it
                    if (std::find(asked.begin(), asked.end(), id) != asked.end()) continue;
                    asked.push_back(id);
                    auto u = rc.Post("/models/unload", hd, json{ { "model", id } }.dump(), "application/json");
                    fprintf(stderr, "router: unloading %s (%s)\n", id.c_str(), u ? std::to_string(u->status).c_str() : "no response");
                }
                unloaded = true;
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
        }
        // the router reports a model unloaded before its process has released the VRAM: wait for the memory
        if (ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU); dev && unloaded) {
            size_t fr = 0, tot = 0, last = 0;
            for (int i = 0; i < 120; i++) {
                ggml_backend_dev_memory(dev, &fr, &tot);
                if (fr + (size_t) 1.5e9 >= tot || (i > 4 && fr == last)) break;
                last = fr;
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
            }
        }
        const auto t0 = std::chrono::steady_clock::now();
        model::LoadOptions lo = o;
        lo.residual_path      = sp->entries[(size_t) idx].res;
        sp->model             = model::load_causal_lm(sp->entries[(size_t) idx].path, lo, e);
        if (!sp->model) {
            e = "load failed: " + e;
            return false;
        }
        sp->q35 = dynamic_cast<model::Qwen35 *>(sp->model.get());
        sp->state_tokens.clear();
        sp->ck_tokens.clear();
        sp->active = idx;
        fprintf(stderr, "%s loaded in %.1f s\n", sp->entries[(size_t) idx].alias.c_str(),
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        return true;
    };
    std::string err;
    if (idle_unload <= 0 && !S->load(0, err)) {
        fprintf(stderr, "%s\n", err.c_str());
        return 1;
    }
    std::thread idle;
    if (idle_unload > 0) {  // free the model after idle_unload seconds without requests
        idle = std::thread([sp, idle_unload] {
            for (;;) {
                std::this_thread::sleep_for(std::chrono::seconds(5));
                if (!sp->model || std::chrono::steady_clock::now() - sp->last_used < std::chrono::seconds(idle_unload)) continue;
                Turn turn(*sp);
                if (sp->model && std::chrono::steady_clock::now() - sp->last_used >= std::chrono::seconds(idle_unload)) {
                    sp->q35 = nullptr;
                    sp->model.reset();
                    sp->state_tokens.clear();
                    sp->ck_tokens.clear();
                    fprintf(stderr, "idle for %d s: model unloaded\n", idle_unload);
                }
            }
        });
        idle.detach();
    }

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
        json data = json::array();
        for (size_t i = 0; i < S->entries.size(); i++) {
            data.push_back(json{ { "id", S->entries[i].alias }, { "object", "model" }, { "owned_by", "eightfer" }, { "created", 0 },
                                 { "status", { { "value", S->model && S->active == (int) i ? "loaded" : "unloaded" } } } });
        }
        if (!S->router_url.empty()) {  // and the router's models
            httplib::Client rc(S->router_url);
            rc.set_connection_timeout(2, 0);
            const httplib::Headers hd = { { "Authorization", req.get_header_value("Authorization") } };
            auto                   r  = rc.Get("/v1/models", hd);
            if (r && r->status == 200) {
                try {
                    const json list = json::parse(r->body);
                    for (const auto & m : list.at("data")) data.push_back(m);
                } catch (const std::exception &) {
                }
            }
        }
        resp.set_content(json{ { "object", "list" }, { "data", data } }.dump(), "application/json");
    });
    http.Post("/unload", [&](const httplib::Request & req, httplib::Response & resp) {  // free the GPU now
        if (!authorized(req, resp)) return;
        Turn turn(*S);
        const bool was = S->model != nullptr;
        unload_model(*S);
        resp.set_content(json{ { "success", true }, { "unloaded", was } }.dump(), "application/json");
    });
    // chat.completion(.chunk) or text_completion objects around a delta / message / text
    auto handler = [&](bool chat) {
        return [&, chat](const httplib::Request & req, httplib::Response & resp) {
            if (!authorized(req, resp)) return;
            json body;
            try {
                body = json::parse(req.body);
            } catch (const std::exception & e) {
                resp.status = 400;
                resp.set_content(json{ { "error", { { "message", e.what() } } } }.dump(), "application/json");
                return;
            }
            if (chat ? !body.contains("messages") : !body.contains("prompt")) {
                resp.status = 400;
                resp.set_content(json{ { "error", { { "message", chat ? "missing messages" : "missing prompt" } } } }.dump(),
                                 "application/json");
                return;
            }
            const std::string want = body.contains("model") && body["model"].is_string() ? body["model"].get<std::string>() : "";
            const bool ours = std::any_of(S->entries.begin(), S->entries.end(), [&](const Entry & e) { return e.alias == want; });
            if (!S->router_url.empty() && !want.empty() && !ours) {
                // another model: free ours and let the router serve it (it loads the model on demand)
                const std::string path = req.path, rbody = req.body, url = S->router_url;
                httplib::Headers  hd   = { { "Authorization", req.get_header_value("Authorization") } };
                Server *          srv  = S.get();
                auto client = [url] {
                    auto c = std::make_unique<httplib::Client>(url);
                    c->set_read_timeout(3600, 0);
                    c->set_write_timeout(600, 0);
                    return c;
                };
                if (!body.value("stream", false)) {
                    Turn turn(*S);
                    unload_model(*S);
                    auto r = client()->Post(path, hd, rbody, "application/json");
                    if (!r) {
                        resp.status = 502;
                        resp.set_content(json{ { "error", { { "message", "router: " + httplib::to_string(r.error()) } } } }.dump(),
                                         "application/json");
                        return;
                    }
                    resp.status = r->status;
                    resp.set_content(r->body, r->has_header("Content-Type") ? r->get_header_value("Content-Type") : "application/json");
                    return;
                }
                resp.set_chunked_content_provider("text/event-stream", [srv, path, rbody, hd, client](size_t, httplib::DataSink & sink) {
                    Turn turn(*srv);
                    unload_model(*srv);
                    auto r = client()->Post(path, hd, rbody, "application/json", [&](const char * d, size_t n) {
                        return sink.is_writable() && sink.write(d, n);
                    });
                    if (!r) {
                        const std::string e = "data: " + json{ { "error", { { "message", "router: " + httplib::to_string(r.error()) } } } }.dump() + "\n\n";
                        sink.write(e.data(), e.size());
                    }
                    sink.done();
                    return true;
                });
                return;
            }
            const std::string id      = random_id(S->rng, chat ? "chatcmpl-" : "cmpl-");
            const int64_t     created = (int64_t) std::time(nullptr);
            const std::string alias   = ours ? want : S->alias;
            auto obj = [id, created, alias, chat](const json & choice, bool chunk) {
                return json{ { "id", id }, { "object", chat ? (chunk ? "chat.completion.chunk" : "chat.completion") : "text_completion" },
                             { "created", created }, { "model", alias }, { "choices", json::array({ choice }) } };
            };
            if (!body.value("stream", false)) {
                Turn        turn(*S);
                Result      R;
                std::string e;
                bool        ok = false;
                try {
                    ok = run_request(*S, body, R, nullptr, [&] { return req.is_connection_closed(); }, e);
                } catch (const std::exception & ex) {
                    e = ex.what();
                }
                if (!ok) {
                    resp.status = 400;
                    resp.set_content(json{ { "error", { { "message", e } } } }.dump(), "application/json");
                    return;
                }
                json choice = { { "index", 0 } };
                if (chat) {
                    json msg    = to_nl(R.msg.to_json_oaicompat());
                    msg["role"] = "assistant";
                    choice["message"] = msg;
                } else {
                    choice["text"]     = R.text;
                    choice["logprobs"] = nullptr;
                }
                choice["finish_reason"] = R.finish;
                json r       = obj(choice, false);
                r["usage"]   = usage(R);
                r["timings"] = timings(R);
                resp.set_content(r.dump(), "application/json");
                return;
            }
            // runs after this handler returns: capture by value only
            Server * srv = S.get();
            resp.set_chunked_content_provider("text/event-stream", [srv, body, obj, chat](size_t, httplib::DataSink & sink) {
                Server & SS   = *srv;
                auto     send = [&](const json & j) {
                    const std::string s = "data: " + j.dump() + "\n\n";
                    return sink.is_writable() && sink.write(s.data(), s.size());
                };
                auto delta = [&](const json & d, const json & finish) {
                    json c = { { "index", 0 } };
                    if (chat) c["delta"] = d;
                    else {
                        c["text"]     = d.value("text", std::string());
                        c["logprobs"] = nullptr;
                    }
                    c["finish_reason"] = finish;
                    return obj(c, true);
                };
                Turn turn(SS);
                if (chat) send(delta({ { "role", "assistant" }, { "content", nullptr } }, nullptr));
                Result      R;
                std::string e;
                bool        ok = false;
                try {
                    ok = run_request(SS, body, R, [&](const json & d) { return send(delta(d, nullptr)); },
                                     [&] { return !sink.is_writable(); }, e);
                } catch (const std::exception & ex) {
                    e = ex.what();
                }
                if (!ok) {
                    send(json{ { "error", { { "message", e } } } });
                } else {
                    json last       = delta(json::object(), R.finish);
                    last["usage"]   = usage(R);
                    last["timings"] = timings(R);
                    send(last);
                }
                const std::string done = "data: [DONE]\n\n";
                sink.write(done.data(), done.size());
                sink.done();
                return true;
            });
        };
    };
    http.Post("/v1/chat/completions", handler(true));
    http.Post("/v1/completions", handler(false));

    printf("eightfer serve: %s on http://%s:%d (model id \"%s\", ctx %d, %s)\n", model_path.c_str(), host.c_str(), port,
           S->alias.c_str(), n_ctx,
           idle_unload > 0 ? "loaded on demand" : S->q35 && S->q35->has_residual() ? "base + residual, speculative" : "plain decoding");
    fflush(stdout);
    if (!http.listen(host, port)) {
        fprintf(stderr, "cannot listen on %s:%d\n", host.c_str(), port);
        return 1;
    }
    return 0;
}

} // namespace e8::cli
