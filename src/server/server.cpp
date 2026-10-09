// `shoehorn serve`: OpenAI-compatible HTTP server (DESIGN.md M6).
//
//   shoehorn serve <model.gguf> [--res r.gguf] [--host 127.0.0.1] [--port 8090] [--alias NAME]
//                  [--api-key-file F] [--chat-template-file F] [--ctx N] [--kv f16|q8_0|q4_0] [--kv-v TYPE] [--kv-lock] [--slots N] [--kv-pool-gb G] [--spec auto|K]
//                  [--gpu-layers N] [--expert-cache-gb G] [--threads N] [--mtp N]
//
// Endpoints: GET /health, GET /v1/models, POST /v1/chat/completions and POST /v1/completions (stream or not),
// POST /unload (free the model; the next request loads it again). Several models (--also ALIAS=PATH[,RES]): the request's
// "model" picks one; it is loaded on demand and the previous one freed (one model in memory at a time). Chat
// templating (Jinja, tools, chat_template_kwargs such as enable_thinking) and output parsing (reasoning_content,
// tool_calls) come from llama.cpp's common library, so requests and responses look like llama-server's. Sampling:
// temperature, top_p, top_k, min_p, seed, presence_penalty, frequency_penalty, stop, max_tokens.
//
// Concurrency: the model is used by one request at a time, in turns taken first come first served (Turn). A request
// holds the model for its prompt, then gives it up after every decode cycle, so requests on different sequence slots
// decode concurrently, a cycle each in turn. Model swaps and unloads wait until no request is decoding. A request
// whose client disconnects stops at the next decode cycle.
//
// Sequence slots (--slots N, 27B with the KV in RAM): the model keeps N sequences, each with its own prompt-reuse state,
// so N conversations (an agent and its subagents) run side by side without evicting each other's prompts. A slot
// serves one request at a time; with more requests than slots the others wait for one. Decoding requests go through
// two lanes that run at the same time: the draft lane (a slot drafter, Qwen35::make_slot_drafter, on its own CUDA
// stream: the requests waiting to draft, up to --draft-batch, drafted together in lockstep) and the verify lane (the model: one joint verify, Qwen35::eval_multi, of
// every request whose drafts are ready, then their acceptance and commits, run_round). A verify is mostly the
// residual crossing PCIe, so the GPU drafts for some requests while others' drafts are verified. Decoding requests
// form two groups that take turns: while one group's drafts are verified together, the other group drafts together
// (each lane waits, briefly, for its group to be complete). A request goes to the slot whose
// state or checkpoint its prompt extends the furthest, else to an empty slot, else to the least recently used one.
// Slots share one RAM KV budget (--kv-pool-gb, default the --ctx tokens' worth); a slot that needs more memory empties
// the least recently used idle slots.
//
// Side sequence (one slot, KV in RAM; --no-side turns it off): a request that continues neither the active sequence
// nor the parked one (a session title, a summary, another conversation) parks the active sequence and starts on an
// empty one, so it does not evict a conversation's prompt; a request that continues the parked sequence swaps it back
// (Qwen35::swap_side, ~0.1 s). The parked sequence is the one swapped out last: one conversation and one side request
// are kept, and the whole VRAM window always belongs to the running request.
//
// Sharing the GPU (--idle-unload SEC, --unload-router URL): the model loads on the first request and is freed after SEC
// idle seconds; before loading, every model loaded by a llama-server router at URL is unloaded (same API key), so the
// two servers take turns on the GPU. The router is also proxied: /v1/models lists its models too, and a completion
// request naming one of them frees this model and is forwarded to the router (which loads it), so a client that only
// talks to this server can switch between all models. Requests of both kinds take turns.
//
// Prompt reuse (per slot): the model state after a request covers prompt + output; a checkpoint is kept at the end of each
// prompt. A new prompt that extends either one only evaluates the new tokens (recurrent state cannot be cut back to an
// arbitrary prefix, so anything else starts over).

#include "cli/commands.h"
#include "model/causal_lm.h"
#include "model/qwen35.h"
#include "model/qwen4exp.h"
#include "runtime/spec.h"
#include "sys/sysinfo.h"

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
#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <optional>
#include <random>
#include <sstream>
#include <thread>

namespace e8::cli {

namespace {

using json = nlohmann::ordered_json;

// llama.cpp's Qwen XML tool-call grammar rejects a call whose optional parameters come before a required one (the
// model often writes str_replace_editor's file_text before path) and the call then vanishes: the turn looks like it
// ends mid-sentence. When the strict parse of the final text finds no call although the text has a <tool_call>, this
// reads <tool_call><function=NAME><parameter=P>VALUE</parameter>...</function></tool_call> blocks in any order,
// typing each value by the tool's JSON schema (strings as written, anything else parsed as JSON).
void lenient_tool_calls(const std::string & text, const std::vector<common_chat_tool> & tools, common_chat_msg & msg) {
    if (!msg.tool_calls.empty()) return;
    size_t p = text.find("<tool_call>");
    if (p == std::string::npos) return;
    const size_t first = p;
    auto trim1 = [](std::string v) {
        if (!v.empty() && v.front() == '\n') v.erase(0, 1);
        if (!v.empty() && v.back() == '\n') v.pop_back();
        return v;
    };
    while (p != std::string::npos) {
        const size_t end  = text.find("</tool_call>", p);
        const std::string blk = text.substr(p, end == std::string::npos ? std::string::npos : end - p);
        const size_t fn   = blk.find("<function=");
        if (fn == std::string::npos) break;
        const size_t fe   = blk.find('>', fn);
        if (fe == std::string::npos) break;
        const std::string name = blk.substr(fn + 10, fe - fn - 10);
        json schema = json::object();
        for (const auto & t : tools) {
            if (t.name == name) {
                try {
                    schema = json::parse(t.parameters).value("properties", json::object());
                } catch (...) {}
            }
        }
        json   args = json::object();
        size_t q    = fe;
        while ((q = blk.find("<parameter=", q)) != std::string::npos) {
            const size_t ne = blk.find('>', q);
            const size_t ve = blk.find("</parameter>", ne);
            if (ne == std::string::npos || ve == std::string::npos) break;
            const std::string pn = blk.substr(q + 11, ne - q - 11);
            const std::string v  = trim1(blk.substr(ne + 1, ve - ne - 1));
            const std::string ty = schema.contains(pn) && schema[pn].contains("type") && schema[pn]["type"].is_string()
                                       ? schema[pn]["type"].get<std::string>() : "string";
            if (ty == "string") {
                args[pn] = v;
            } else {
                try {
                    args[pn] = json::parse(v);
                } catch (...) {
                    args[pn] = v;
                }
            }
            q = ve + 12;
        }
        common_chat_tool_call tc;
        tc.name      = name;
        tc.arguments = args.dump();
        msg.tool_calls.push_back(tc);
        p = end == std::string::npos ? std::string::npos : text.find("<tool_call>", end);
    }
    if (!msg.tool_calls.empty()) {
        const size_t c = msg.content.find("<tool_call>");
        if (c != std::string::npos) msg.content.erase(c);
        while (!msg.content.empty() && (msg.content.back() == '\n' || msg.content.back() == ' ')) msg.content.pop_back();
        fprintf(stderr, "tool call recovered by the lenient parser (strict grammar rejected it): %s\n", msg.tool_calls[0].name.c_str());
    }
    (void) first;
}

// Reasoning stuck in a loop: a chunk of the last 400 characters seen twice before, or most of the last 40 lines
// repeating earlier lines. Real reasoning (new content) never trips it.
bool reasoning_loops(const std::string & text) {
    const size_t start = text.find("<think>") == std::string::npos ? 0 : text.find("<think>") + 7;
    const size_t end   = text.size();
    if (end - start < 4000) return false;
    // literal repeats of a recent 200-character chunk
    const std::string probe = text.substr(end - 300, 200);
    size_t            hits  = 0;
    for (size_t p = text.find(probe, start); p != std::string::npos && p < end - 300; p = text.find(probe, p + 1)) hits++;
    if (hits >= 2) return true;
    // recent lines that repeat earlier ones
    std::vector<std::string> lines;
    for (size_t a = start; a < end;) {
        size_t b = text.find('\n', a);
        if (b == std::string::npos) b = end;
        if (b - a > 20) lines.push_back(text.substr(a, b - a));
        a = b + 1;
    }
    if (lines.size() < 60) return false;
    std::unordered_map<std::string, int> seen;
    for (size_t i = 0; i + 40 < lines.size(); i++) seen[lines[i]]++;
    int dup = 0;
    for (size_t i = lines.size() - 40; i < lines.size(); i++) dup += seen.count(lines[i]) ? 1 : 0;
    return dup >= 24;
}

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
    float                            tool_temp = 0.6f;  // sampling temperature inside <tool_call> blocks (0 = as the request)
    bool                             spec_auto = true;
    int                              mtp = 6;  // most MTP proposals per base pass while drafting (adaptive; 0 = off)
    bool                             echo = true;  // echo drafting (copies from the context)
    // agent turns: earlier assistant turns' reasoning is not resent; no thinking right after a routine tool result
    // (errors, failures and tracebacks in the results still get a reasoning pass)
    bool                             drop_reasoning = true, think_after_tool = false;
    int                              think_budget = 32768;  // hard cap on reasoning tokens (0 = off); loops are cut earlier
    // first-come-first-served turn taking: each request takes a ticket and runs when `serving` reaches it
    std::mutex                       mu;
    std::condition_variable          cv;
    uint64_t                         next_ticket = 0, serving = 0;
    // per sequence slot: tokens in its model state / at its checkpoint; epoch = the model's slot_epoch() they belong to
    struct Slot {
        std::vector<int32_t> state_tokens, ck_tokens;
        uint64_t             epoch = 0, used = 0;
        bool                 busy = false;  // a request is using it
    };
    Slot                             park;         // the parked side sequence's tokens (one slot, Qwen35::swap_side)
    int                              streams = 0;  // requests holding a slot (the model must stay loaded)
    // decode rounds (sequence slots): requests in their decode loop, and the drafted cycles waiting for the verify
    struct Job {
        runtime::SpecDecoder * dec = nullptr;
        int                    slot = 0, group = 0;
        std::chrono::steady_clock::time_point since;
        std::vector<int32_t> * out = nullptr;
        bool                   done = false, ok = true;
        std::string            err;
    };
    int                              n_dec = 0;
    int                              group_n[2] = { 0, 0 };  // decoding requests per group
    int                              in_draft[2] = { 0, 0 };  // per group: requests being drafted
    std::vector<Job *>               pending;
    bool                             verifying = false;      // a request runs the verify lane
    // the draft lane: held by one request at a time (it drafts the queued requests together, or runs a prompt)
    struct DraftReq {
        runtime::SpecDecoder * dec = nullptr;
        int                    slot = 0, group = 0;
        std::chrono::steady_clock::time_point since;
        bool                   done = false, ok = true;
        std::string            err;
    };
    std::vector<DraftReq *>          draft_queue;
    bool                             draft_busy  = false;
    int                              draft_batch = 1;
    json                             config;                 // the dashboard's: --ctx, --kv, --kv-v, --spec
    std::unique_ptr<model::Qwen35>   drafter;                // sequence slots: drafts beside the verifies
    std::vector<Slot>                slots;
    uint64_t                         slot_clock = 0;
    std::function<bool(int, std::string &)> load;  // loads entry i (lazy mode)
    std::string                      router_url;  // llama-server router to unload / forward to (empty = none)
    std::string                      timing_log;  // JSON lines of per-request timings (no content), when set
    // dashboard (/dashboard, /stats): the last requests' timing lines and what is generating now
    std::mutex                       stats_mu;
    std::deque<json>                 recent;
    std::atomic<bool>                live_busy{ false };
    std::atomic<int>                 live_gen{ 0 }, live_prompt{ 0 };
    std::atomic<int64_t>             live_start_ms{ 0 };
    std::string                      live_model;
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

// holds the model; waits for the turns asked for earlier. release() / acquire() give it up and queue again (a decoding
// request does so after every cycle). An exclusive turn (unloading or swapping the model) also waits until no request
// holds a sequence slot.
class Turn {
public:
    explicit Turn(Server & S, bool exclusive = false) : S_(S), excl_(exclusive) { acquire(); }
    ~Turn() { release(); }
    void acquire() {
        if (held_) return;
        std::unique_lock<std::mutex> lk(S_.mu);
        for (;;) {
            if (excl_) S_.cv.wait(lk, [&] { return S_.streams == 0; });
            const uint64_t t = S_.next_ticket++;
            S_.cv.wait(lk, [&] { return S_.serving == t; });
            if (!excl_ || S_.streams == 0) break;
            S_.serving++;  // a request started in between: let it run, queue again
            S_.cv.notify_all();
        }
        held_ = true;
    }
    void release() {
        if (!held_) return;
        {
            std::lock_guard<std::mutex> lk(S_.mu);
            S_.serving++;
        }
        held_ = false;
        S_.cv.notify_all();
    }
    // gives up the turn until `ready` (checked under the lock) holds, then queues again
    template <class F> void wait_for(F ready) {
        release();
        {
            std::unique_lock<std::mutex> lk(S_.mu);
            S_.cv.wait(lk, ready);
        }
        acquire();
    }
private:
    Server & S_;
    bool     excl_ = false, held_ = false;
};

// holds the draft lane (the slot drafter and the VRAM recurrent state it shares with the model) for a prompt or a
// reasoning close. Holding the model and then the draft lane is fine; the other way round is not.
class DraftLane {
public:
    explicit DraftLane(Server & S) : S_(S) {
        std::unique_lock<std::mutex> lk(S_.mu);
        S_.cv.wait(lk, [&] { return !S_.draft_busy; });
        S_.draft_busy = true;
    }
    ~DraftLane() {
        {
            std::lock_guard<std::mutex> lk(S_.mu);
            S_.draft_busy = false;
        }
        S_.cv.notify_all();
    }
private:
    Server & S_;
};

// frees the model (the next request of ours loads it again); the caller holds a Turn
void unload_model(Server & S) {
    if (!S.model) return;
    S.drafter.reset();  // it shares the model's tensors
    S.q35 = nullptr;
    S.model.reset();
    S.active = -1;
    S.slots.clear();
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
    int             n_prompt = 0, n_gen = 0, n_reused = 0, slot = 0;
    int             side_swap = 0;
    int             n_tool = 0;     // tokens generated inside <tool_call> blocks (at --tool-temp)  // side sequence: 1 = parked the conversation, 2 = resumed the parked one
    double          t_prompt = 0, t_gen = 0;
};

// Runs one request: chat (body has "messages") or completion ("prompt"). `on_delta` (streaming) receives OpenAI delta
// objects as the output grows (chat: {content, reasoning_content, tool_calls}; completion: {text}) and returns false
// when the client is gone, which stops generation (as does `cancelled`).
// E8_LANE_LOG=1: a line per draft and per verify (ms since start, slot(s), duration), to see how the lanes overlap
double lane_ms() {
    static const auto t0 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}
const bool kLaneLog = std::getenv("E8_LANE_LOG") != nullptr;
// how long a lane waits for the rest of a group (E8_GROUP_WAIT_MS)
const int kGroupWaitMs = std::getenv("E8_GROUP_WAIT_MS") ? std::atoi(std::getenv("E8_GROUP_WAIT_MS")) : 250;

// One verify (the verify lane, holding the model): the pending drafts of `group` (all groups when -1) verified together,
// then each request's acceptance and commit.
void run_round(Server & S, int group = -1) {
    std::vector<Server::Job *> jobs;
    {  // the pending jobs (of `group`, unless -1), in order, as many as one verify holds (the rest wait for the next)
        std::lock_guard<std::mutex> lk(S.mu);
        const int cap = S.q35 ? S.q35->max_multi() : 1 << 30;
        int       n   = 0;
        for (size_t i = 0; i < S.pending.size();) {
            Server::Job * j = S.pending[i];
            const int     t = (int) j->dec->cycle_tokens().size();
            if (group >= 0 && j->group != group) {
                i++;
                continue;
            }
            if (!jobs.empty() && n + t > cap) break;
            n += t;
            jobs.push_back(j);
            S.pending.erase(S.pending.begin() + (long) i);
        }
    }
    if (jobs.empty()) return;
    {
        int vk = jobs[0]->dec->verify_topk();  // one row width for all (full logits when they differ)
        for (auto * j : jobs) {
            if (j->dec->verify_topk() != vk) vk = 0;
        }
        const int64_t                nv = S.q35->n_vocab();
        std::vector<model::MultiSeq> seqs;
        int                          N = 0;
        for (auto * j : jobs) {
            const auto & t = j->dec->cycle_tokens();
            seqs.push_back({ j->slot, t.data(), (int) t.size() });
            N += (int) t.size();
        }
        std::vector<float>   lg((size_t) N * (size_t) (vk > 0 ? vk : nv));
        std::vector<int32_t> ids(vk > 0 ? (size_t) N * (size_t) vk : 0);
        std::string          e;
        const auto           t0 = std::chrono::steady_clock::now();
        const double t_start = lane_ms();
        const bool ok = S.q35->eval_multi(seqs, vk, lg.data(), vk > 0 ? ids.data() : nullptr, e);
        const double tv = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (kLaneLog) {
            std::string sl;
            for (auto * j : jobs) sl += std::to_string(j->slot) + ":" + std::to_string(j->dec->cycle_tokens().size()) + " ";
            fprintf(stderr, "lane %9.0f verify [%s] %.0f ms\n", t_start, sl.c_str(), tv);
        }
        int off = 0;
        for (auto * j : jobs) {
            const int n = (int) j->dec->cycle_tokens().size();
            if (!ok) {
                j->ok  = false;
                j->err = e;
                continue;
            }
            j->ok = j->dec->finish_cycle(j->slot, lg.data() + (size_t) off * (size_t) (vk > 0 ? vk : nv),
                                         vk > 0 ? ids.data() + (size_t) off * (size_t) vk : nullptr, tv / (double) jobs.size(),
                                         *j->out, j->err);
            if (!j->ok) S.q35->commit_seq(j->slot, 0, e);  // leave nothing uncommitted
            off += n;
        }
    }
    {
        std::lock_guard<std::mutex> lk(S.mu);
        for (auto * j : jobs) j->done = true;
    }
    S.cv.notify_all();
}

// whether a request's model name names `alias`; names from before the rename (...-eightfer for ...-shoehorn) still do
bool names_alias(const std::string & alias, const std::string & want) {
    if (want == alias) return true;
    const std::string old = "-eightfer", now = "-shoehorn";
    return want.size() > old.size() && want.compare(want.size() - old.size(), old.size(), old) == 0 &&
           want.substr(0, want.size() - old.size()) + now == alias;
}

bool run_request(Server & S, Turn & turn, const json & body, Result & R, const std::function<bool(const json &)> & on_delta,
                 const std::function<bool()> & cancelled, std::string & err) {
    int idx = 0;  // the entry this request names (the main model when it names none of ours)
    if (body.contains("model") && body["model"].is_string()) {
        for (size_t i = 0; i < S.entries.size(); i++) {
            if (names_alias(S.entries[i].alias, body["model"].get<std::string>())) idx = (int) i;
        }
    }
    if (S.active != idx || !S.model) {
        if (S.model && S.streams > 0) {  // another model: wait until no request is decoding on this one
            turn.wait_for([&] { return S.streams == 0; });
            if (S.active == idx && S.model) goto loaded;
        }
        unload_model(S);
        if (!S.load(idx, err)) return false;
        S.active = idx;
    }
loaded:
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
    std::vector<common_chat_tool> req_tools;
    bool                          think_open = false;  // the prompt ends inside an open reasoning block
    int                           req_think_budget = 0;  // this request's reasoning cap from reasoning_effort (0: server default)
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
    json msgs = body.at("messages");
    if (S.drop_reasoning && msgs.is_array()) {
        // earlier assistant turns' reasoning: not resent (smaller, steadier prompts; the template may keep it)
        for (auto & mm : msgs) {
            if (!mm.is_object() || mm.value("role", "") != "assistant") continue;
            mm.erase("reasoning_content");
            mm.erase("reasoning");
            if (mm.contains("content") && mm["content"].is_string()) {
                std::string c = mm["content"].get<std::string>();
                for (size_t a; (a = c.find("<think>")) != std::string::npos;) {
                    const size_t b = c.find("</think>", a);
                    c.erase(a, b == std::string::npos ? std::string::npos : b + 8 - a);
                }
                while (!c.empty() && (c.front() == '\n' || c.front() == ' ')) c.erase(0, 1);
                mm["content"] = c;
            }
        }
    }
    const bool after_tool = msgs.is_array() && !msgs.empty() && msgs.back().is_object() && msgs.back().value("role", "") == "tool";
    // ... unless a tool result since the last assistant message reports trouble (then the model should reason about it)
    bool tool_trouble = false;
    if (after_tool) {
        static const char * const kTrouble[] = { "Traceback", "Error", "error:", "ERROR", "FAIL", "Failed", "failed",
                                                 "Exception", "exception", "assert", "not found", "No such file",
                                                 "denied", "exit code", "Exit code", "non-zero", "SyntaxError" };
        for (size_t i = msgs.size(); i-- > 0 && msgs[i].is_object() && msgs[i].value("role", "") == "tool";) {
            const json & c  = msgs[i].contains("content") ? msgs[i]["content"] : json();
            const std::string s = c.is_string() ? c.get<std::string>() : c.dump();
            for (const char * k : kTrouble) tool_trouble = tool_trouble || s.find(k) != std::string::npos;
        }
    }
    in.messages = common_chat_msgs_parse_oaicompat(common_json::parse(msgs.dump()));
    if (body.contains("tools") && !body["tools"].is_null()) {
        in.tools = common_chat_tools_parse_oaicompat(common_json::parse(body["tools"].dump()));
        if (const char * dir = std::getenv("E8_LOG_RAW")) {  // debugging: the tool schemas (not the messages)
            if (FILE * ft = fopen((std::string(dir) + "/tools.json").c_str(), "wb")) {
                json meta = { { "tools", body["tools"] } };
                if (body.contains("chat_template_kwargs")) meta["chat_template_kwargs"] = body["chat_template_kwargs"];
                if (body.contains("reasoning_effort")) meta["reasoning_effort"] = body["reasoning_effort"];
                const std::string s = meta.dump(1);
                fwrite(s.data(), 1, s.size(), ft);
                fclose(ft);
            }
        }
    }
    if (body.contains("tool_choice") && body["tool_choice"].is_string()) {
        in.tool_choice = common_chat_tool_choice_parse_oaicompat(body["tool_choice"].get<std::string>());
    }
    in.parallel_tool_calls   = body.value("parallel_tool_calls", true);  // several independent calls in one reply
    in.add_generation_prompt = true;
    in.use_jinja             = true;
    in.reasoning_format      = COMMON_REASONING_FORMAT_DEEPSEEK;
    if (body.contains("chat_template_kwargs") && body["chat_template_kwargs"].is_object()) {
        for (auto & [k, v] : body["chat_template_kwargs"].items()) in.chat_template_kwargs[k] = v.dump();
    }
    // The Qwen 3.x template adds a reasoning-effort sentence to the system prompt while thinking is on, except
    // for "medium" (its default is "xhigh"). Thinking toggles between agent steps, so any sentence there would
    // change the prompt 3 tokens in and void the prefix cache: always "medium" (no sentence); the reasoning budget
    // enforces the requested effort instead.
    in.chat_template_kwargs["reasoning_effort"] = "\"medium\"";
    auto et = in.chat_template_kwargs.find("enable_thinking");
    if (et != in.chat_template_kwargs.end()) in.enable_thinking = et->second == "true";
    if (after_tool && !tool_trouble && !S.think_after_tool) {
        // the turn after a tool result: act on it without a reasoning pass (agent loops)
        in.enable_thinking                        = false;
        in.chat_template_kwargs["enable_thinking"] = "false";
    }
    if (body.contains("thinking") && body["thinking"].is_object() && body["thinking"].value("type", "") == "disabled") {
        // DeepSeek-style switch (dsh sends it): no reasoning block
        in.enable_thinking                        = false;
        in.chat_template_kwargs["enable_thinking"] = "false";
    }
    if (body.contains("reasoning_effort") && body["reasoning_effort"].is_string()) {
        // "none"/"off" turns thinking off; low / medium / high cap the reasoning at 2K / 8K / 24K tokens (the
        // Qwen templates ignore an effort kwarg, so the budget is what makes the setting mean something)
        const std::string re = body["reasoning_effort"].get<std::string>();
        if (re == "none" || re == "off") {
            in.enable_thinking                        = false;
            in.chat_template_kwargs["enable_thinking"] = "false";
        } else {
            // not passed to the template: Qwen templates add an effort sentence to the system prompt only while
            // thinking is on, so the prompt (and the prefix cache) would change whenever thinking toggles between
            // agent steps. The budget below enforces the effort instead.
            if (re == "minimal" || re == "low") req_think_budget = 2048;
            else if (re == "medium") req_think_budget = 8192;
            else if (re == "high") req_think_budget = 24576;
        }
    }
    const common_chat_params cp = common_chat_templates_apply(S.tmpl, in);
    req_tools                   = in.tools;
    {
        const size_t tp = cp.prompt.rfind("<think>");
        think_open      = tp != std::string::npos && tp + 12 >= cp.prompt.size() &&
                     cp.prompt.find("</think>", tp) == std::string::npos;
    }

    pp                  = common_chat_parser_params(cp);
    pp.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
    pp.parse_tool_calls = true;
    if (!cp.parser.empty()) pp.parser.load(cp.parser);

    stops.insert(stops.end(), cp.additional_stops.begin(), cp.additional_stops.end());

    // The checkpoint goes before the generation prompt ("<|im_start|>assistant\n<think>\n"): the next turn re-renders
    // the assistant turn differently, but the conversation before it stays a prefix. The conversation part is always
    // rendered with thinking on: the template's system prompt differs with thinking off (its tool-call example has no
    // <think> block), and thinking toggles between agent steps (think_after_tool), which voided the prefix cache 4K
    // tokens in. Only the generation prompt follows this step's setting.
    std::string head = cp.prompt, tail;
    auto split = [](const common_chat_params & c, std::string & h, std::string & t) {
        h = c.prompt;
        t.clear();
        if (!c.generation_prompt.empty() && h.size() > c.generation_prompt.size() &&
            h.compare(h.size() - c.generation_prompt.size(), std::string::npos, c.generation_prompt) == 0) {
            t = c.generation_prompt;
            h.resize(h.size() - t.size());
            return true;
        }
        return false;
    };
    if (split(cp, head, tail) && !in.enable_thinking) {
        auto in_on                                = in;
        in_on.enable_thinking                     = true;
        in_on.chat_template_kwargs["enable_thinking"] = "true";
        std::string h_on, t_on;
        if (split(common_chat_templates_apply(S.tmpl, in_on), h_on, t_on)) head = h_on;
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
    // ---- sequence slot: the one whose state or checkpoint this prompt extends furthest, else an empty one, else the
    // least recently used
    const int n_slots = S.q35 ? S.q35->n_slots() : 1;
    if ((int) S.slots.size() != n_slots) S.slots.assign((size_t) n_slots, Server::Slot{});
    auto any_free = [&] {
        return S.model.get() != &m || std::any_of(S.slots.begin(), S.slots.end(), [](const Server::Slot & x) { return !x.busy; });
    };
    if (!any_free()) turn.wait_for(any_free);
    if (S.model.get() != &m || (int) S.slots.size() != n_slots) {
        err = "the model changed while this request waited for a slot";
        return false;
    }
    auto sync_slot = [&](int s) {  // forget tokens of a sequence the model has dropped
        auto & sl = S.slots[(size_t) s];
        if (S.q35 && S.q35->slot_epoch(s) != sl.epoch) {
            sl.state_tokens.clear();
            sl.ck_tokens.clear();
            sl.epoch = S.q35->slot_epoch(s);
        }
    };
    int    slot = -1;
    size_t best = 0;
    for (int s = 0; s < n_slots; s++) {
        sync_slot(s);
        const auto & sl  = S.slots[(size_t) s];
        if (sl.busy) continue;
        const int    np  = S.q35 ? S.q35->slot_n_past(s) : m.n_past();
        size_t       len = 0;
        if (extends(sl.state_tokens) && (int) sl.state_tokens.size() == np) len = sl.state_tokens.size();
        else if (extends(sl.ck_tokens)) len = sl.ck_tokens.size();
        if (len > best) {
            best = len;
            slot = s;
        }
    }
    if (slot < 0) {
        for (int s = 0; s < n_slots && slot < 0; s++) {
            const auto & x = S.slots[(size_t) s];
            if (!x.busy && x.state_tokens.empty() && x.ck_tokens.empty()) slot = s;
        }
    }
    if (slot < 0) {
        for (int s = 0; s < n_slots; s++) {
            if (!S.slots[(size_t) s].busy && (slot < 0 || S.slots[(size_t) s].used < S.slots[(size_t) slot].used)) slot = s;
        }
    }
    std::optional<DraftLane> dlane;
    if (S.drafter) dlane.emplace(S);
    if (S.q35 && !S.q35->select_slot(slot, err)) return false;
    sync_slot(slot);
    // hold the slot (and keep the model loaded) until this request ends
    struct Hold {
        Server & S;
        int      slot;
        Hold(Server & s, int i) : S(s), slot(i) {
            std::lock_guard<std::mutex> lk(S.mu);
            S.slots[(size_t) slot].busy = true;
            S.streams++;
            if (S.q35) S.q35->pin_slot(slot, true);
        }
        ~Hold() {
            {
                std::lock_guard<std::mutex> lk(S.mu);
                if (slot < (int) S.slots.size()) S.slots[(size_t) slot].busy = false;
                S.streams--;
                if (S.q35) S.q35->pin_slot(slot, false);
            }
            S.cv.notify_all();
        }
    } slot_hold(S, slot);
    auto & SL = S.slots[(size_t) slot];
    SL.used   = ++S.slot_clock;
    R.slot    = slot;
    // between decode cycles: let the other requests run a cycle, then continue on this slot
    auto yield = [&]() {
        turn.release();
        turn.acquire();
        S.last_used = std::chrono::steady_clock::now();
        return !S.q35 || S.q35->select_slot(slot, err);
    };

    // ---- side sequence (one slot): park the active sequence for a request that does not continue it, swap back for
    // one that continues the parked sequence
    if (S.q35 && S.q35->has_side() && n_slots == 1) {
        if (S.q35->side_epoch() != S.park.epoch) {  // dropped (KV pool full)
            S.park.state_tokens.clear();
            S.park.ck_tokens.clear();
            S.park.epoch = S.q35->side_epoch();
        }
        auto reuse = [&](const Server::Slot & x, int np) -> size_t {
            if (extends(x.state_tokens) && (int) x.state_tokens.size() == np) return x.state_tokens.size();
            if (extends(x.ck_tokens)) return x.ck_tokens.size();
            return 0;
        };
        constexpr size_t kSideMin = 2048;  // an active sequence worth keeping
        const size_t     ra = reuse(SL, m.n_past()), rp = reuse(S.park, S.q35->side_n_past());
        const size_t     ka = std::max(SL.state_tokens.size(), SL.ck_tokens.size());
        const bool       back = rp > ra, fresh = !back && ra == 0 && ka >= kSideMin;
        if (back || fresh) {
            const auto ts = std::chrono::steady_clock::now();
            if (fresh) {  // the new request starts on an empty sequence: forget the parked one
                S.q35->drop_side();
                S.park.state_tokens.clear();
                S.park.ck_tokens.clear();
            }
            if (!S.q35->swap_side(err)) {
                SL.state_tokens.clear();
                SL.ck_tokens.clear();
                return false;
            }
            std::swap(SL.state_tokens, S.park.state_tokens);
            std::swap(SL.ck_tokens, S.park.ck_tokens);
            SL.epoch     = S.q35->slot_epoch(0);
            S.park.epoch = S.q35->side_epoch();
            fprintf(stderr, "side sequence: %s (%zu tokens parked, %.0f ms)\n",
                    back ? "resumed the parked sequence" : "parked the conversation for an unrelated request",
                    std::max(S.park.state_tokens.size(), S.park.ck_tokens.size()),
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ts).count());
            R.side_swap = back ? 2 : 1;
        }
    }

    size_t start = 0;
    if (extends(SL.state_tokens) && (int) SL.state_tokens.size() == m.n_past()) {
        start = SL.state_tokens.size();
    } else if (extends(SL.ck_tokens)) {
        m.checkpoint_restore();
        start = SL.ck_tokens.size();
    } else {
        if (!SL.ck_tokens.empty()) {
            // reuse miss: where the new prompt leaves the checkpointed one (debugging cache hit rates)
            size_t d = 0;
            while (d < SL.ck_tokens.size() && d < prompt.size() && SL.ck_tokens[d] == prompt[d]) d++;
            auto piece = [&](const std::vector<int32_t> & v, size_t a, size_t b) {
                std::string s;
                for (size_t i = a; i < std::min(b, v.size()); i++) s += common_token_to_piece(S.vocab, v[i], true);
                for (auto & c : s) if (c == '\n') c = '|';
                return s.substr(0, 160);
            };
            fprintf(stderr, "prompt reuse miss: diverges at token %zu of %zu (checkpoint %zu)\n  before: %s\n  old:    %s\n  new:    %s\n",
                    d, prompt.size(), SL.ck_tokens.size(), piece(prompt, d > 25 ? d - 25 : 0, d).c_str(),
                    piece(SL.ck_tokens, d, d + 30).c_str(), piece(prompt, d, d + 30).c_str());
        }
        m.reset();
    }
    R.n_prompt = (int) prompt.size();
    R.n_reused = (int) start;
    const auto         t0 = std::chrono::steady_clock::now();
    std::vector<float> last((size_t) m.n_vocab());
    auto eval_range = [&](size_t a, size_t b) {
        if (b > a && !m.prefill(prompt.data() + a, (int) (b - a), last.data(), err)) {
            SL.state_tokens.clear();
            SL.ck_tokens.clear();
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
        SL.ck_tokens.assign(prompt.begin(), prompt.begin() + (long long) n_head);
    }
    if (!eval_range(start, prompt.size())) return false;
    const auto t1 = std::chrono::steady_clock::now();
    S.live_gen   = 0;
    S.live_busy  = true;
    S.live_prompt = R.n_prompt;
    S.live_start_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    {
        std::lock_guard<std::mutex> lk(S.stats_mu);
        S.live_model = S.entries[(size_t) S.active].alias;
    }
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
    auto last_sent = std::chrono::steady_clock::now();  // streaming: when the last delta went out
    // tool calls sample cooler: arguments (paths, numbers, copied values) need precision, prose and reasoning keep
    // the request's temperature. Switched between decode cycles, from the <tool_call> / </tool_call> tokens.
    const float base_temp = sp.temp;
    const float tool_temp = body.value("tool_temperature", S.tool_temp);
    auto special_id = [&](const char * s) {
        const auto v = common_tokenize(S.vocab, s, false, true);
        return v.size() == 1 ? v[0] : -1;
    };
    const int32_t tc_open = tool_temp > 0 && tool_temp < base_temp ? special_id("<tool_call>") : -1;
    const int32_t tc_close = tc_open >= 0 ? special_id("</tool_call>") : -1;
    auto consume = [&]() {
        for (; used < out.size() && !done; used++) {
            const int32_t t = out[used];
            if (t == tc_open) sp.temp = tool_temp;
            else if (t == tc_close && tc_close >= 0) sp.temp = base_temp;
            else if (tc_open >= 0 && sp.temp == tool_temp) R.n_tool++;
            if (llama_vocab_is_eog(S.vocab, t)) {
                done = true;
                break;
            }
            text += common_token_to_piece(S.vocab, t, true);
            R.n_gen++;
            S.live_gen = R.n_gen;
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
                if (done) lenient_tool_calls(text, req_tools, cur);
                cur.set_tool_call_ids(ids_cache, gen_id);
                const auto diffs = common_chat_msg_diff::compute_diffs(prev, cur);
                const auto now_t = std::chrono::steady_clock::now();
                if (!diffs.empty()) last_sent = now_t;
                else if (now_t - last_sent > std::chrono::seconds(5)) {
                    // nothing new to show (e.g. a tool call that only parses once complete): an empty delta keeps the
                    // stream alive, so clients with an idle timeout do not drop a long generation
                    last_sent = now_t;
                    if (!on_delta(json::object())) {
                        R.finish = "cancelled";
                        done     = true;
                    }
                }
                for (const auto & d : diffs) {
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
        auto make_dec = [&]() {
            auto d = std::make_unique<runtime::SpecDecoder>(*S.q35, S.spec_k, sp, S.spec_auto);
            d->set_mtp(S.mtp);
            d->set_echo(S.echo);
            return d;
        };
        int  next_loop_check = 512;
        auto dec_p = make_dec();
        // the two lanes (sequence slots with a drafter): the model is detached, this request holds neither the model
        // nor the draft lane except while it drafts, verifies, or (rarely) closes the reasoning
        const bool rounds = S.drafter != nullptr;
        struct Member {
            Server & S;
            bool     on;
            int      group = 0;
            Member(Server & s, bool r) : S(s), on(r) {
                if (!on) return;
                std::lock_guard<std::mutex> lk(S.mu);
                S.n_dec++;
                group = S.group_n[0] <= S.group_n[1] ? 0 : 1;
                S.group_n[group]++;
            }
            ~Member() {
                if (!on) return;
                {
                    std::lock_guard<std::mutex> lk(S.mu);
                    S.n_dec--;
                    S.group_n[group]--;
                }
                S.cv.notify_all();  // a lane may be waiting for this request's group
            }
        } member(S, rounds);
        struct Reacquire {  // run_request returns holding the model
            Turn & t;
            bool   on;
            ~Reacquire() {
                if (on) t.acquire();
            }
        } reacquire{ turn, rounds };
        dec_p->set_context(prompt);
        dec_p->begin(last.data(), out);
        consume();
        if (rounds) {
            S.q35->detach();
            dlane.reset();
            turn.release();
        }
        while (!done && S.q35->slot_n_past(slot) + S.spec_k + 2 < m.n_ctx()) {
            if (think_open && text.find("</think>") != std::string::npos) think_open = false;
            bool cut = false;
            const int budget = req_think_budget > 0 ? req_think_budget : S.think_budget;
            if (think_open && budget > 0 && R.n_gen >= budget) cut = true;
            if (think_open && !cut && R.n_gen >= next_loop_check) {
                next_loop_check = R.n_gen + 128;
                cut             = reasoning_loops(text);
            }
            if (cut) {
                think_open = false;
                {
                    // reasoning over budget: close it ourselves and let the model act (the last sampled token is not
                    // evaluated yet: it goes in with the closing tag)
                    const bool                 looped = R.n_gen < budget;
                    const std::vector<int32_t> f      = common_tokenize(
                        S.vocab, looped ? "\n\nI have gone in circles; I will act on the best plan so far.\n</think>\n\n"
                                        : "\n\nI have thought about this enough; I will act on the best plan so far.\n</think>\n\n",
                        false, true);
                    std::vector<int32_t>       ev(1, out.back());
                    ev.insert(ev.end(), f.begin(), f.end());
                    if (rounds) {  // the model for one eval: both lanes
                        turn.acquire();
                        DraftLane dl(S);
                        const bool ok = S.q35->select_slot(slot, err) &&
                                        S.q35->eval_last(ev.data(), (int) ev.size(), last.data(), err);
                        S.q35->detach();
                        turn.release();
                        if (!ok) return false;
                    } else if (!S.q35->eval_last(ev.data(), (int) ev.size(), last.data(), err)) {
                        return false;
                    }
                    out.insert(out.end(), f.begin(), f.end());
                    consume();
                    fprintf(stderr, "reasoning closed (%s) after %d tokens\n", R.n_gen >= budget ? "budget" : "repetition loop", R.n_gen);
                    std::vector<int32_t> ctx_now = prompt;
                    ctx_now.insert(ctx_now.end(), out.begin(), out.end());
                    dec_p = make_dec();
                    dec_p->set_context(ctx_now);
                    dec_p->begin(last.data(), out);
                    consume();
                    continue;
                }
            }
            if (rounds) {
                int ng;  // a verify holds one group's drafts
                {
                    std::lock_guard<std::mutex> lk(S.mu);
                    ng = std::max(1, S.group_n[member.group]);
                }
                dec_p->set_max_k(S.q35->max_multi() / ng - 1);
                {  // draft lane: queue up; whoever takes the lane drafts every queued request together
                    const double     tw = lane_ms();
                    Server::DraftReq rq;
                    rq.dec   = dec_p.get();
                    rq.slot  = slot;
                    rq.group = member.group;
                    rq.since = std::chrono::steady_clock::now();
                    std::unique_lock<std::mutex> lk(S.mu);
                    S.draft_queue.push_back(&rq);
                    S.cv.notify_all();
                    while (!rq.done) {
                        if (S.draft_busy) {
                            S.cv.wait(lk);
                            continue;
                        }
                        // the queue head's group, drafted together: wait (a while) for its requests that are being
                        // verified or about to queue, not for those waiting for a verify (they need this lane first)
                        const int  g   = S.draft_queue.front()->group;
                        const auto due = S.draft_queue.front()->since + std::chrono::milliseconds(kGroupWaitMs);
                        int        nq = 0, np = 0;
                        for (auto * r : S.draft_queue) nq += r->group == g;
                        for (auto * j : S.pending) np += j->group == g;
                        const int coming = S.group_n[g] - nq - np - S.in_draft[g];  // being verified, or between lanes
                        if (coming > 0 && nq < S.draft_batch && std::chrono::steady_clock::now() < due) {
                            S.cv.wait_until(lk, due);
                            continue;
                        }
                        S.draft_busy = true;
                        std::vector<Server::DraftReq *> grp;
                        for (size_t i = 0; i < S.draft_queue.size() && (int) grp.size() < std::max(1, S.draft_batch);) {
                            if (S.draft_queue[i]->group != g) {
                                i++;
                                continue;
                            }
                            grp.push_back(S.draft_queue[i]);
                            S.draft_queue.erase(S.draft_queue.begin() + (long) i);
                        }
                        S.in_draft[g] += (int) grp.size();
                        lk.unlock();
                        std::vector<runtime::SpecDecoder *> decs;
                        std::vector<int>                    sls;
                        for (auto * r : grp) {
                            decs.push_back(r->dec);
                            sls.push_back(r->slot);
                        }
                        const double td = lane_ms();
                        std::string  e;
                        const bool   ok = runtime::SpecDecoder::draft_lockstep(decs, sls, *S.drafter, e);
                        if (kLaneLog) {
                            std::string sl;
                            for (auto * r : grp) sl += std::to_string(r->slot) + ":" + std::to_string(r->dec->cycle_tokens().size()) + " ";
                            fprintf(stderr, "lane %9.0f draft  [%s] %.0f ms (waited %.0f)\n", td, sl.c_str(), lane_ms() - td, td - tw);
                        }
                        lk.lock();
                        for (auto * r : grp) {
                            r->ok   = ok;
                            r->err  = e;
                            r->done = true;
                        }
                        S.in_draft[g] -= (int) grp.size();
                        S.draft_busy = false;
                        S.cv.notify_all();
                    }
                    if (!rq.ok) {
                        err = rq.err;
                        return false;
                    }
                }
                // verify lane: verify everything pending (these drafts and others') unless a verify is running
                Server::Job job;
                job.dec  = dec_p.get();
                job.slot = slot;
                job.out  = &out;
                job.group = member.group;
                job.since = std::chrono::steady_clock::now();
                {
                    // the pending head's group, verified together once all its drafts are here (a verify of one
                    // request's drafts costs nearly as much as of three: the residual crosses PCIe once)
                    std::unique_lock<std::mutex> lk(S.mu);
                    S.pending.push_back(&job);
                    S.cv.notify_all();
                    while (!job.done) {
                        if (S.verifying || S.pending.empty()) {
                            S.cv.wait(lk);
                            continue;
                        }
                        // the head's group: wait (a while) for its requests being drafted right now, not for those
                        // queued for the draft lane (they may be waiting for this verify's requests)
                        const int  g   = S.pending.front()->group;
                        const auto due = S.pending.front()->since + std::chrono::milliseconds(kGroupWaitMs);
                        if (S.in_draft[g] > 0 && std::chrono::steady_clock::now() < due) {
                            S.cv.wait_until(lk, due);
                            continue;
                        }
                        S.verifying = true;
                        lk.unlock();
                        turn.acquire();
                        run_round(S, g);
                        turn.release();
                        lk.lock();
                        S.verifying = false;
                        S.cv.notify_all();
                    }
                }
                if (!job.ok) {
                    err = job.err;
                    return false;
                }
                S.last_used = std::chrono::steady_clock::now();
            } else if (!yield() || !dec_p->step(out, err)) {
                return false;
            }
            consume();
            dec_p->set_temp(sp.temp);  // inside or outside a tool call, for the next cycle
        }
        const auto & st = dec_p->stats();
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
            if (!yield() || !m.eval_last(&out.back(), 1, last.data(), err)) return false;
            next();
            consume();
        }
    }
    if (!done) R.finish = "length";
    // the model state now holds prompt + every produced token except the last (sampled, not evaluated)
    SL.state_tokens = prompt;
    SL.state_tokens.insert(SL.state_tokens.end(), out.begin(), out.end() - 1);
    R.t_gen = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
    if (chat) {  // final parse (also needed by the timing log below)
        R.msg = common_chat_parse(text, false, pp);
        lenient_tool_calls(text, req_tools, R.msg);
    }

    S.last_used = std::chrono::steady_clock::now();
    S.live_busy = false;
    {  // numbers only: never prompt or output text
        const auto now_sys = std::chrono::system_clock::now();
        const auto t0_sys  = now_sys - std::chrono::duration_cast<std::chrono::system_clock::duration>(
                                          std::chrono::duration<double>(R.t_prompt + R.t_gen));
        json line = { { "t", (double) std::chrono::duration_cast<std::chrono::milliseconds>(now_sys.time_since_epoch()).count() / 1e3 },
                      { "model", S.entries[(size_t) S.active].alias }, { "kind", chat ? "chat" : "completion" },
                      { "prompt_tokens", R.n_prompt }, { "reused_tokens", R.n_reused }, { "slot", R.slot }, { "prefill_s", R.t_prompt },
                      { "gen_tokens", R.n_gen }, { "decode_s", R.t_gen }, { "finish", R.finish },
                      { "idle_before_s", S.last_end.time_since_epoch().count() == 0 ? -1.0
                                         : std::chrono::duration<double>(t0_sys - S.last_end).count() },
                      { "temp", sp.temp }, { "spec", R.stats } };
        if (R.n_tool) line["tool_tokens"] = R.n_tool;
        if (R.side_swap) line["side"] = R.side_swap == 1 ? "parked" : "resumed";
        if (chat) {  // shape of the output (sizes and a repetition score, no text)
            auto rep = [](const std::string & s) {  // share of repeated lines (a reasoning loop shows as high)
                std::unordered_map<std::string, int> seen;
                int lines = 0, dup = 0;
                size_t a = 0;
                while (a < s.size()) {
                    size_t b = s.find('\n', a);
                    if (b == std::string::npos) b = s.size();
                    if (b - a > 8) {
                        lines++;
                        dup += seen[s.substr(a, b - a)]++ > 0;
                    }
                    a = b + 1;
                }
                return lines ? (double) dup / lines : 0.0;
            };
            line["reasoning_chars"] = R.msg.reasoning_content.size();
            line["content_chars"]   = R.msg.content.size();
            line["tool_calls"]      = R.msg.tool_calls.size();
            line["reasoning_repeat"] = rep(R.msg.reasoning_content);
            line["content_repeat"]  = rep(R.msg.content);
        }
        S.last_end = now_sys;
        {
            std::lock_guard<std::mutex> lk(S.stats_mu);
            S.recent.push_back(line);
            while (S.recent.size() > 3000) S.recent.pop_front();
        }
        if (FILE * f = S.timing_log.empty() ? nullptr : fopen(S.timing_log.c_str(), "ab")) {
            const std::string l = line.dump() + "\n";
            fwrite(l.data(), 1, l.size(), f);
            fclose(f);
        }
    }
    const std::string slot_tag = S.slots.size() > 1 ? " [slot " + std::to_string(R.slot) + "]" : std::string();
    fprintf(stderr, "%s%s: prompt %d tokens (%d reused) in %.1f s, %d generated at %.1f tok/s (temp %.2f top_k %d top_p %.2f)%s%s\n",
            S.entries[(size_t) S.active].alias.c_str(), slot_tag.c_str(), R.n_prompt, R.n_reused, R.t_prompt, R.n_gen,
            R.n_gen / std::max(R.t_gen, 1e-9), sp.temp, sp.top_k, sp.top_p, R.spec.empty() ? "" : "; ", R.spec.c_str());
    if (!chat) {
        R.text = text;
        return true;
    }
    R.msg = common_chat_parse(text, false, pp);
    lenient_tool_calls(text, req_tools, R.msg);
    R.msg.set_tool_call_ids(ids_cache, gen_id);
    if (!R.msg.tool_calls.empty() && R.finish == "stop") R.finish = "tool_calls";
    if (const char * dir = std::getenv("E8_LOG_RAW")) {
        // debugging: the raw generated text of chat turns that ended without a tool call (last 8 kept)
        static int seq = 0;
        if (R.msg.tool_calls.empty()) {
            const std::string path = std::string(dir) + "/raw-" + std::to_string(seq++ % 8) + ".txt";
            if (FILE * fr = fopen(path.c_str(), "wb")) {
                fprintf(fr, "finish=%s n_gen=%d\n----\n", R.finish.c_str(), R.n_gen);
                fwrite(text.data(), 1, text.size(), fr);
                fclose(fr);
            }
        }
    }
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
    std::string model_path, res, host = "127.0.0.1", key_file, tmpl_file, kv = "f16", kv_v;
    int         port = 8090, n_ctx = 16384, gpu_layers = 999, threads = 0;
    int gpu_kv = -1, idle_unload = 0;
    bool kv_lock = false, side = true;
    int n_slots = 1, draft_batch = 0;
    double kv_pool_gb = 0;
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
        else if (a == "--kv-v") kv_v = val();
        else if (a == "--kv-lock") kv_lock = true;
        else if (a == "--no-side") side = false;
        else if (a == "--slots") n_slots = std::max(1, std::atoi(val().c_str()));
        else if (a == "--kv-pool-gb") kv_pool_gb = std::atof(val().c_str());
        else if (a == "--draft-batch") draft_batch = std::max(1, std::atoi(val().c_str()));
        else if (a == "--gpu-layers") gpu_layers = std::atoi(val().c_str());
        else if (a == "--expert-cache-gb") cache_gb = std::atof(val().c_str());
        else if (a == "--threads") threads = std::atoi(val().c_str());
        else if (a == "--mtp") S->mtp = std::atoi(val().c_str());
        else if (a == "--echo") S->echo = std::atoi(val().c_str()) != 0;
        else if (a == "--drop-reasoning") S->drop_reasoning = std::atoi(val().c_str()) != 0;
        else if (a == "--think-after-tool") S->think_after_tool = std::atoi(val().c_str()) != 0;
        else if (a == "--think-budget") S->think_budget = std::atoi(val().c_str());
        else if (a == "--idle-unload") idle_unload = std::atoi(val().c_str());
        else if (a == "--temperature" || a == "--temp") S->defaults.temp = (float) std::atof(val().c_str());
        else if (a == "--top-p") S->defaults.top_p = (float) std::atof(val().c_str());
        else if (a == "--top-k") S->defaults.top_k = std::atoi(val().c_str());
        else if (a == "--min-p") S->defaults.min_p = (float) std::atof(val().c_str());
        else if (a == "--presence-penalty") S->defaults.presence_penalty = (float) std::atof(val().c_str());
        else if (a == "--unload-router") router_url = val();
        else if (a == "--timing-log") S->timing_log = val();
        else if (a == "--tool-temp") S->tool_temp = (float) std::atof(val().c_str());
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
        fprintf(stderr, "usage: shoehorn serve <model.gguf> [--res r.gguf] [--host H] [--port 8090] [--alias NAME]\n"
                        "         [--api-key-file F] [--chat-template-file F] [--ctx 16384] [--kv f16|q8_0|q4_0] [--spec auto|K]\n"
                        "         [--kv-v TYPE (V cache type, default: --kv)] [--kv-lock (keep the RAM KV in physical memory)]\n"
                        "         [--slots N (sequences kept, one per conversation)] [--kv-pool-gb G (RAM KV of all slots)]\n"
                        "         [--no-side (one slot: no parked side sequence for unrelated requests)]\n"
                        "         [--draft-batch N (slots: requests drafted together, default min(slots, 2))]\n"
                        "         [--gpu-layers N] [--expert-cache-gb G] [--threads N] [--mtp N (0 = off)]\n"
                        "         [--idle-unload SEC (load on demand, free after SEC idle)] [--unload-router URL]\n"
                        "         [--temperature 1.0] [--top-p 0.95] [--top-k 20] [--min-p 0] [--presence-penalty 0] (request defaults)\n"
                        "         [--tool-temp 0.6 (temperature inside <tool_call> blocks; 0 = the request's)]\n"
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
    auto kv_type = [](const std::string & t, ggml_type & out) {
        if (t == "f16") out = GGML_TYPE_F16;
        else if (t == "q8_0") out = GGML_TYPE_Q8_0;
        else if (t == "q4_0") out = GGML_TYPE_Q4_0;
        else return false;
        return true;
    };
    if (!kv_type(kv, o.kv_type) || (!kv_v.empty() && !kv_type(kv_v, o.kv_type_v))) {
        fprintf(stderr, "error: KV cache types are f16, q8_0 or q4_0\n");
        return 1;
    }
    S->config = { { "ctx", n_ctx }, { "kv", kv }, { "kv_v", kv_v.empty() ? kv : kv_v }, { "slots", n_slots } };
    o.kv_lock         = kv_lock;
    o.side_seq        = side && n_slots == 1;
    o.n_slots         = n_slots;
    o.kv_pool_gb      = kv_pool_gb;
    o.draft_batch     = n_slots > 1 ? (draft_batch > 0 ? std::min(draft_batch, n_slots) : std::min(n_slots, 2)) : 1;
    S->draft_batch    = o.draft_batch;
    o.residual_path   = res;
    o.expert_cache_gb = cache_gb;
    o.max_record      = n_slots > 1 ? 64 : S->spec_k + 1;  // slots: a joint verify's sequences (VRAM: see qwen35.cpp)
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
        sp->slots.clear();
        sp->drafter.reset();
        if (sp->q35 && sp->q35->n_slots() > 1 && sp->q35->has_residual()) {
            auto d = std::make_unique<model::Qwen35>();
            if (!sp->q35->make_slot_drafter(*d, std::max(1, o.draft_batch), e)) {
                e = "slot drafter: " + e;
                return false;
            }
            sp->drafter = std::move(d);
        }
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
                Turn turn(*sp, true);
                if (sp->model && std::chrono::steady_clock::now() - sp->last_used >= std::chrono::seconds(idle_unload)) {
                    sp->drafter.reset();
                    sp->q35 = nullptr;
                    sp->model.reset();
                    sp->slots.clear();
                    fprintf(stderr, "idle for %d s: model unloaded\n", idle_unload);
                }
            }
        });
        idle.detach();
    }

    if (!S->timing_log.empty()) {  // the dashboard starts with the last requests from the timing log
        std::ifstream tl(S->timing_log);
        std::deque<std::string> tail;
        for (std::string l; std::getline(tl, l);) {
            tail.push_back(l);
            if (tail.size() > 3000) tail.pop_front();
        }
        std::lock_guard<std::mutex> lk(S->stats_mu);
        for (auto & l : tail) {
            try {
                S->recent.push_back(json::parse(l));
            } catch (...) {}
        }
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
    // dashboard: numbers only (the timing-log lines of recent requests and the live state), no prompts or outputs
    http.Get("/dashboard", [](const httplib::Request &, httplib::Response & resp) {
        static const char * page =
#include "dashboard.inc"
            ;
        resp.set_content(page, "text/html; charset=utf-8");
    });
    // /stats?since=T: only the timing lines after T (the dashboard polls with its newest line's t)
    http.Get("/favicon.ico", [](const httplib::Request &, httplib::Response & resp) {  // the dashboard's mark
        resp.set_header("Cache-Control", "max-age=86400");
        resp.set_content(R"(<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32"><rect width="32" height="32" rx="7" fill="#08090a"/>)"
                         R"(<path d="M13.2 4h5.6a2 2 0 0 1 2 2v5c0 2.5 3.2 5.7 3.2 10.8 0 4.2-3.8 6.7-8 6.7s-8-2.5-8-6.7c0-5.1 3.2-8.3 3.2-10.8V6a2 2 0 0 1 2-2z" fill="#d9a54a"/>)"
                         R"(<circle cx="16" cy="7.6" r="1.5" fill="#08090a"/></svg>)", "image/svg+xml");
    });
    http.Get("/stats", [&](const httplib::Request & req, httplib::Response & resp) {
        const double since = req.has_param("since") ? std::atof(req.get_param_value("since").c_str()) : -1;
        const int64_t now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        json out;
        {
            std::lock_guard<std::mutex> lk(S->stats_mu);
            out["recent"] = json::array();
            for (auto & r : S->recent)
                if (r.value("t", 0.0) > since) out["recent"].push_back(r);
            out["live"] = { { "busy", S->live_busy.load() }, { "model", S->live_model }, { "gen", S->live_gen.load() },
                            { "prompt", S->live_prompt.load() }, { "elapsed_s", (now_ms - S->live_start_ms.load()) / 1e3 } };
            if (!S->recent.empty()) out["live"]["last_end_s"] = now_ms / 1e3 - S->recent.back().value("t", 0.0);
        }
        json models = json::array();
        for (auto & e : S->entries) models.push_back(e.alias);
        out["models"] = models;
        out["loaded"] = S->model ? S->entries[(size_t) S->active].alias : "";
        out["config"] = S->config;
        {  // where the model lives (the dashboard's memory view)
            json   mem = json::object();
            size_t fr = 0, tot = 0;
            if (ggml_backend_dev_t g = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU)) ggml_backend_dev_memory(g, &fr, &tot);
            const auto si = sys::query();
            mem["vram_total"] = tot;
            mem["vram_free"]  = fr;
            mem["ram_total"]  = si.ram_total;
            mem["ram_avail"]  = si.ram_avail;
            if (S->model) {
                mem["weights_gpu"] = S->model->gpu_weight_bytes();
                mem["weights_cpu"] = S->model->cpu_weight_bytes();
            }
            if (S->q35) {
                mem["residual"]   = S->q35->residual_bytes();
                mem["kv_ram"]     = S->q35->kv_committed_bytes();
                mem["side_parked"] = S->q35->has_side() ? S->q35->side_n_past() : 0;
            }
            out["mem"] = mem;
        }
        if (S->q35) out["config"]["window"] = S->q35->gpu_kv();
        resp.set_header("Cache-Control", "no-store");
        // pages served from this machine (the dsh plugin at 127.0.0.1:3080) may read the numbers; nothing else
        const std::string origin = req.get_header_value("Origin");
        if (origin.rfind("http://127.0.0.1:", 0) == 0 || origin.rfind("http://localhost:", 0) == 0) {
            resp.set_header("Access-Control-Allow-Origin", origin);
            resp.set_header("Vary", "Origin");
        }
        resp.set_content(out.dump(), "application/json");
    });
    http.Get("/v1/models", [&](const httplib::Request & req, httplib::Response & resp) {
        if (!authorized(req, resp)) return;
        json data = json::array();
        for (size_t i = 0; i < S->entries.size(); i++) {
            data.push_back(json{ { "id", S->entries[i].alias }, { "object", "model" }, { "owned_by", "shoehorn" }, { "created", 0 },
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
        Turn turn(*S, true);
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
            const bool ours = std::any_of(S->entries.begin(), S->entries.end(), [&](const Entry & e) { return names_alias(e.alias, want); });
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
                    Turn turn(*S, true);
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
                    Turn turn(*srv, true);
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
                    ok = run_request(*S, turn, body, R, nullptr, [&] { return req.is_connection_closed(); }, e);
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
                    ok = run_request(SS, turn, body, R, [&](const json & d) { return send(delta(d, nullptr)); },
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

    printf("shoehorn serve: %s on http://%s:%d (model id \"%s\", ctx %d, %s)\n", model_path.c_str(), host.c_str(), port,
           S->alias.c_str(), n_ctx,
           idle_unload > 0 ? "loaded on demand" : S->q35 && S->q35->has_residual() ? "base + residual, speculative" : "plain decoding");
    fflush(stdout);
    if (!http.listen(host, port)) {
        fprintf(stderr, "cannot listen on %s:%d\n", host.c_str(), port);
        return 1;
    }
    return 0;
}

// `shoehorn parsetest <model.gguf> --raw raw.txt --meta tools.json [--template t.jinja]`: runs the server's chat
// output parser on a logged raw turn (E8_LOG_RAW) with the request's tool schemas, and prints what it extracted.
int parsetest(const std::vector<std::string> & args) {
    std::string model, raw_path, meta_path, tmpl_file;
    for (size_t i = 2; i < args.size(); i++) {
        auto val = [&]() { return i + 1 < args.size() ? args[++i] : std::string(); };
        if (args[i] == "--raw") raw_path = val();
        else if (args[i] == "--meta") meta_path = val();
        else if (args[i] == "--template") tmpl_file = val();
        else model = args[i];
    }
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.vocab_only         = true;
    llama_model * vm      = llama_model_load_from_file(model.c_str(), mp);
    if (!vm) {
        fprintf(stderr, "cannot load %s\n", model.c_str());
        return 1;
    }
    auto        tmpls = common_chat_templates_init(vm, tmpl_file.empty() ? "" : read_file(tmpl_file));
    std::string text  = read_file(raw_path);
    const size_t sep  = text.find("----\n");
    if (sep != std::string::npos) text = text.substr(sep + 5);
    const json meta = json::parse(read_file(meta_path));
    common_chat_templates_inputs in;
    in.messages = common_chat_msgs_parse_oaicompat(common_json::parse(R"([{"role":"user","content":"x"}])"));
    in.tools    = common_chat_tools_parse_oaicompat(common_json::parse(meta.at("tools").dump()));
    in.add_generation_prompt = true;
    in.use_jinja             = true;
    in.reasoning_format      = COMMON_REASONING_FORMAT_DEEPSEEK;
    if (meta.contains("chat_template_kwargs")) {
        for (auto & [k, v] : meta["chat_template_kwargs"].items()) in.chat_template_kwargs[k] = v.dump();
    }
    const common_chat_params cp = common_chat_templates_apply(tmpls.get(), in);
    common_chat_parser_params pp(cp);
    pp.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
    pp.parse_tool_calls = true;
    if (!cp.parser.empty()) pp.parser.load(cp.parser);
    printf("format: %s, prompt ends: %s\n", common_chat_format_name(cp.format),
           json(cp.prompt.substr(cp.prompt.size() > 40 ? cp.prompt.size() - 40 : 0)).dump().c_str());
    try {
        common_chat_msg msg = common_chat_parse(text, false, pp);
        lenient_tool_calls(text, in.tools, msg);
        printf("reasoning: %zu chars, content: %s\ntool calls: %zu\n", msg.reasoning_content.size(),
               json(msg.content.substr(0, 300)).dump().c_str(), msg.tool_calls.size());
        for (auto & tc : msg.tool_calls) printf("  %s %s\n", tc.name.c_str(), tc.arguments.substr(0, 300).c_str());
    } catch (const std::exception & e) {
        printf("parse threw: %s\n", e.what());
    }
    return 0;
}
} // namespace e8::cli
