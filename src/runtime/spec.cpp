#include "runtime/spec.h"

#include <atomic>
#include <thread>

#include <algorithm>
#include <chrono>
#include <cmath>

namespace e8::runtime {

namespace {
double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
} // namespace

void apply_penalties(float * logits, const SamplerParams & sp, const TokenCounts & counts) {
    if (sp.presence_penalty == 0 && sp.frequency_penalty == 0) return;
    for (const auto & [t, c] : counts) {
        if (c > 0) logits[t] -= sp.frequency_penalty * (float) c + sp.presence_penalty;
    }
}

SpecDecoder::SpecDecoder(model::Qwen35 & m, int k, const SamplerParams & sp, bool adaptive) :
    m_(m), k_(std::min(std::max(0, k), kMaxK)), adaptive_(adaptive && k > 0), sp_(sp),
    rng_(sp.seed ? sp.seed : std::random_device{}()) {}

double SpecDecoder::verify_ms(int n) const {
    // measured cost of an n-token verify if there is one, else the nearest measured size of the same kind scaled by
    // the prior shape. Below the model's GPU batch the host residual is multiplied on the CPU (cost grows with n);
    // from it on the residual streams to the GPU (about one 14 GB transfer, nearly flat in n). With the weight
    // prefetch that is every size.
    const int gb    = m_.verify_gpu_min();
    auto      prior = [gb](int m) { return m < gb ? 250.0 + 38.0 * m : (gb <= 1 ? 255.0 : 340.0) + 0.5 * m; };
    if (vn_[n] > 0.3) return vt_[n] / vn_[n];
    int near = -1;
    for (int d = 1; d < kMaxK + 1 && near < 0; d++) {
        for (int m : { n - d, n + d }) {
            if (m >= 1 && m <= kMaxK + 1 && (m < gb) == (n < gb) && vn_[m] > 0.3) near = m;
        }
    }
    return near < 0 ? prior(n) : vt_[near] / vn_[near] * prior(n) / prior(near);
}

int SpecDecoder::choose_mtp() const {
    // MTP proposals per round: expected kept proposals (per-position survival, weak prior) + the base's own token,
    // per ms of one base check + the MTP steps; up to mtp_n_
    int    best = 1;
    double best_rate = 0, surv = 1, kept = 0;
    for (int m = 1; m <= std::min(mtp_n_, kMaxMtp); m++) {
        const double hazard = (mfail_[m] + 0.4 * 2.0) / (mseen_[m] + 2.0);
        surv *= 1 - std::clamp(hazard, 0.0, 1.0);
        kept += surv;
        const double rate = (kept + 1) / (check_ms_ + m * step_ms_);
        if (rate > best_rate) {
            best_rate = rate;
            best      = m;
        }
    }
    return best;
}

int SpecDecoder::choose_k() const {
    // expected accepted drafts at k = sum over draft positions j <= k of P(first j drafts accepted), from decayed
    // per-position counts with a weak prior (rejections cluster, so long runs are likelier than a fixed per-token rate
    // predicts); then pick the k with the best expected tokens per millisecond
    const int kmax = std::min(k_, m_.max_verify() - 1);
    // with a flat verify cost (every size streams the residual) at least 24 drafts: shorter verifies look cheap to
    // the estimate but measured slower (8 seeds x 4 prompts: 41.9 vs 40.8 tok/s mean; 32: 42.9, but prose 27.9)
    static const int kmin_env = std::getenv("E8_KMIN") ? std::atoi(std::getenv("E8_KMIN")) : -1;
    const int        kmin     = kmin_env >= 0 ? kmin_env : (m_.verify_gpu_min() <= 1 ? 32 : 1);
    int       best = std::min(kmin, kmax);
    double    best_rate = 0, surv = 1, accepted = 0;
    for (int k = 1; k <= kmax; k++) {
        const double hazard = (fail_[k] + kPriorHazard * kPriorWeight) / (seen_[k] + kPriorWeight);
        surv *= 1 - std::clamp(hazard, 0.0, 1.0);
        accepted += surv;
        const double rate = (accepted + 1) / (k * draft_ms_ + verify_ms(k + 1));
        if (k >= kmin && rate > best_rate) {
            best_rate = rate;
            best      = k;
        }
    }
    return best;
}

void SpecDecoder::dist(const float * logits, std::vector<std::pair<float, int32_t>> & p) {
    const int32_t nv = (int32_t) m_.hp().n_vocab;
    p.resize((size_t) nv);
    for (int32_t i = 0; i < nv; i++) p[(size_t) i] = { logits[i] / sp_.temp, i };
    auto   desc = [](const std::pair<float, int32_t> & a, const std::pair<float, int32_t> & b) { return a.first > b.first; };
    size_t n    = p.size();
    if (sp_.top_k > 0 && (size_t) sp_.top_k < n) {
        std::partial_sort(p.begin(), p.begin() + sp_.top_k, p.end(), desc);
        n = (size_t) sp_.top_k;
    } else if (sp_.top_p >= 1.0f && sp_.min_p <= 0.0f) {
        // nothing below needs the order: only put the most likely token first
        std::iter_swap(p.begin(), std::max_element(p.begin(), p.end(), [](const auto & a, const auto & b) { return a.first < b.first; }));
    } else {
        // the nucleus / min-p set lies among the likely tokens: sort only those when they provably hold the top_p mass
        const float mx = std::max_element(p.begin(), p.end(), [](const auto & a, const auto & b) { return a.first < b.first; })->first;
        const float lo = mx - 14.0f;  // logit cut, ~1e-6 of the top probability
        auto        mid = std::partition(p.begin(), p.end(), [lo](const auto & e) { return e.first >= lo; });
        double      in = 0, all = 0;
        for (auto it = p.begin(); it != p.end(); ++it) {
            const double e = std::exp((double) (it->first - mx));
            all += e;
            if (it < mid) in += e;
        }
        // the tail (below the cut) can stay unsorted when it is discarded anyway: the top_p cut ends inside the head
        // (with margin for float sums), or min_p removes everything under ~8e-7 of the top probability
        const bool head_only = (sp_.top_p < 1.0f && in >= (sp_.top_p + 1e-4) * all) || sp_.min_p > 1e-6f;
        std::sort(p.begin(), head_only ? mid : p.end(), desc);
    }
    p.resize(n);
    finish_dist(p);
}

void SpecDecoder::dist_topk(const float * v, const int32_t * id, int k, std::vector<std::pair<float, int32_t>> & p) {
    // the same distribution as dist() from a row's top-k (best first; k >= top_k)
    const int n = std::min(k, sp_.top_k);
    p.resize((size_t) n);
    for (int i = 0; i < n; i++) p[(size_t) i] = { v[i] / sp_.temp, id[i] };
    finish_dist(p);
}

void SpecDecoder::finish_dist(std::vector<std::pair<float, int32_t>> & p) {
    // softmax over the kept (sorted) entries, then min_p and top_p, renormalised
    const float mx  = p[0].first;
    double      sum = 0;
    for (auto & e : p) {
        e.first = std::exp(e.first - mx);
        sum += e.first;
    }
    for (auto & e : p) e.first = (float) (e.first / sum);
    if (sp_.min_p > 0) {
        const float th = p[0].first * sp_.min_p;
        while (p.size() > 1 && p.back().first < th) p.pop_back();
    }
    if (sp_.top_p < 1.0f) {
        double c    = 0;
        size_t keep = p.size();
        for (size_t i = 0; i < p.size(); i++) {
            c += p[i].first;
            if (c >= sp_.top_p) {
                keep = i + 1;
                break;
            }
        }
        p.resize(keep);
    }
    double s2 = 0;
    for (auto & e : p) s2 += e.first;
    for (auto & e : p) e.first = (float) (e.first / s2);
}

int32_t SpecDecoder::sample(const float * logits, int32_t exclude) {
    const int32_t nv = (int32_t) m_.hp().n_vocab;
    if (sp_.temp <= 0) {
        int32_t best = -1;
        for (int32_t i = 0; i < nv; i++) {
            if (i != exclude && (best < 0 || logits[i] > logits[best])) best = i;
        }
        return best;
    }
    std::vector<std::pair<float, int32_t>> p;
    dist(logits, p);
    double tot = 0;
    for (auto & e : p) tot += e.second == exclude ? 0.0 : e.first;
    if (tot <= 0) return p[0].second;  // only the excluded token had mass (cannot follow a rejection)
    std::uniform_real_distribution<double> u(0.0, tot);
    double r = u(rng_);
    for (auto & e : p) {
        if (e.second == exclude) continue;
        r -= e.first;
        if (r <= 0) return e.second;
    }
    for (auto it = p.rbegin(); it != p.rend(); ++it) {
        if (it->second != exclude) return it->second;
    }
    return p[0].second;
}

double SpecDecoder::prob(const float * logits, int32_t tok) {
    if (sp_.temp <= 0) {
        const int32_t nv   = (int32_t) m_.hp().n_vocab;
        int32_t       best = 0;
        for (int32_t i = 1; i < nv; i++) {
            if (logits[i] > logits[best]) best = i;
        }
        return best == tok ? 1.0 : 0.0;
    }
    std::vector<std::pair<float, int32_t>> p;
    dist(logits, p);
    for (auto & e : p) {
        if (e.second == tok) return e.first;
    }
    return 0.0;
}

bool SpecDecoder::prefill(const std::vector<int32_t> & prompt, std::vector<int32_t> & out, std::string & err) {
    const double  t0 = now();
    const int64_t nv = m_.hp().n_vocab;
    logits_.resize((size_t) nv);
    const size_t ub = (size_t) m_.max_batch();
    for (size_t i = 0; i < prompt.size(); i += ub) {
        const int       n     = (int) std::min(prompt.size() - i, ub);
        const bool      final = i + (size_t) n == prompt.size();
        model::EvalOpts o;
        o.last_only = true;
        if (!m_.eval(prompt.data() + i, n, o, final ? logits_.data() : nullptr, nullptr, err)) return false;
    }
    set_context(prompt);
    last_ = sample(logits_.data(), -1);
    emit(out, last_);
    st_.emitted++;
    st_.prefill_tokens += (int64_t) prompt.size();
    st_.t_prefill += now() - t0;
    return true;
}

void SpecDecoder::begin(const float * last_logits, std::vector<int32_t> & out) {
    logits_.assign(last_logits, last_logits + m_.hp().n_vocab);
    apply_penalties(logits_.data(), sp_, counts_);
    last_ = sample(logits_.data(), -1);
    emit(out, last_);
    st_.emitted++;
}

namespace {
using Dist = std::vector<std::pair<float, int32_t>>;
double at(const Dist & v, int32_t t) {
    for (auto & e : v) {
        if (e.second == t) return (double) e.first;
    }
    return 0.0;
}
int32_t draw(const Dist & p, std::mt19937_64 & rng) {
    double r = std::uniform_real_distribution<double>(0.0, 1.0)(rng);
    for (auto & e : p) {
        if ((r -= e.first) <= 0) return e.second;
    }
    return p.back().second;
}
// after rejecting a draft from q: a sample from max(0, p - q), normalised
int32_t draw_residual(const Dist & p, const Dist & q, std::mt19937_64 & rng) {
    std::unordered_map<int32_t, float> qm;  // q by token (q and p can hold the whole vocabulary)
    qm.reserve(q.size() * 2);
    for (auto & e : q) qm.emplace(e.second, e.first);
    std::vector<std::pair<double, int32_t>> r;
    double                                  tot = 0;
    for (auto & e : p) {
        const auto   it = qm.find(e.second);
        const double d  = e.first - (it == qm.end() ? 0.0 : (double) it->second);
        if (d > 0) {
            r.emplace_back(d, e.second);
            tot += d;
        }
    }
    if (tot <= 0) return p.front().second;
    double u = std::uniform_real_distribution<double>(0.0, tot)(rng);
    for (auto & e : r) {
        if ((u -= e.first) <= 0) return e.second;
    }
    return r.back().second;
}
} // namespace

uint64_t SpecDecoder::ngram_key(const int32_t * t) {
    uint64_t h = 1469598103934665603ull;  // FNV-1a over kEchoN token ids
    for (int i = 0; i < kEchoN; i++) {
        h ^= (uint32_t) t[i];
        h *= 1099511628211ull;
    }
    return h;
}

int SpecDecoder::find_copy(const int32_t * tail) {
    // index every kEchoN-gram of the history ending before its last position, then look up `tail`; returns the
    // history position right after the most recent earlier occurrence (where the copy continues), or -1
    const int h = (int) hist_.size();
    for (; ngram_upto_ < h; ngram_upto_++) {
        if (ngram_upto_ >= kEchoN) ngram_[ngram_key(hist_.data() + ngram_upto_ - kEchoN)] = ngram_upto_;
    }
    const auto it = ngram_.find(ngram_key(tail));
    if (it == ngram_.end()) return -1;
    const int e = it->second;
    if (e >= h || !std::equal(tail, tail + kEchoN, hist_.data() + e - kEchoN)) return -1;  // hash collision
    return e;
}

bool SpecDecoder::draft_rounds(int k, std::vector<int32_t> & toks, std::vector<Dist> & qd, TokenCounts & cur, std::string & err,
                               bool extend) {
    // Rounds of: proposals, then one base pass that checks them (speculative sampling against the proposals'
    // distribution, or argmax agreement when greedy) and adds one token of its own. Proposals come from
    //  - echo: when the last kEchoN tokens occurred earlier in the context, the tokens that followed them (up to 63;
    //    agents re-emit file contents, code and arguments); a deterministic proposal, q = 1 on the copied token
    //  - else the MTP block (mtp_n_ tokens), when loaded
    //  - else none: the base pass adds one token.
    // The kept tokens are exact samples from the base, so they and the base's distributions are the drafts for the
    // base + residual verify.
    model::Qwen35 &    dm      = d_ ? *d_ : m_;  // the shadow drafts when there is one
    const int64_t      nv      = dm.hp().n_vocab;
    const bool         sampled = sp_.temp > 0;
    const int          tk      = sampled ? gpu_topk() : 0;  // > 0: rows come back as their top-k only
    const int64_t      rw      = tk > 0 ? tk : nv;          // floats per row
    std::vector<float> ml(sampled ? (size_t) rw : 0), lg;
    std::vector<int32_t> mid(tk > 0 ? (size_t) tk : 1), lid;
    while ((int) toks.size() - 1 < k) {
        if (stop_ && stop_->load()) break;  // pre-drafting: the verify is done
        const int            need = k - ((int) toks.size() - 1);
        std::vector<int32_t> in(1, toks.back());
        std::vector<Dist>    pq;
        bool                 echo = false;
        if (echo_ && !no_echo_ && need > 1) {
            // the last kEchoN tokens of history + this cycle's drafts
            std::vector<int32_t> tail;
            const int            nd = (int) toks.size() - 1;  // drafts so far (toks[0] = last_, already in hist_)
            for (int i = kEchoN - 1; i >= 0; i--) {
                const int back = i - nd;  // >= 0: from hist_, else from toks
                if (back >= 0) {
                    if ((int) hist_.size() - 1 - back < 0) break;
                    tail.push_back(hist_[hist_.size() - 1 - (size_t) back]);
                } else {
                    tail.push_back(toks[(size_t) (-back)]);  // i tokens from the end = draft nd - i
                }
            }
            const int j = (int) tail.size() == kEchoN ? find_copy(tail.data()) : -1;
            if (j >= 0) {
                // the base's check of the copy must be rollback-able: at most max_record - 1 proposals
                const int len = std::min({ need - 1, 63, dm.max_record() - 1, (int) hist_.size() - j });
                for (int t = 0; t < len; t++) {
                    in.push_back(hist_[(size_t) (j + t)]);
                    if (sampled) pq.push_back(Dist{ { 1.0f, hist_[(size_t) (j + t)] } });
                }
                echo = len > 0;
            }
        }
        if (extend && !echo) break;  // extending a long copy: only further copies
        const double tm0 = now();
        if (!echo && mtp_n_ > 0 && dm.has_mtp() && dm.hidden_row() >= 0) {
            const int   mt  = std::min(choose_mtp(), need - 1);
            const int   row = dm.hidden_row();
            TokenCounts cm  = sampled ? cur : TokenCounts{};
            for (int j = 0; j < mt; j++) {  // MTP chain: position of toks.back() is n_past()
                int32_t id = -1;
                if (!dm.mtp_step(in.back(), dm.n_past() - 1 + j, j == 0 ? row : -1, sampled ? ml.data() : nullptr,
                                 tk > 0 ? mid.data() : &id, err, tk)) {
                    return false;
                }
                if (sampled) {
                    pq.emplace_back();
                    if (tk > 0) {
                        dist_topk(ml.data(), mid.data(), tk, pq.back());
                    } else {
                        apply_penalties(ml.data(), sp_, cm);
                        dist(ml.data(), pq.back());
                    }
                    id = draw(pq.back(), rng_);
                    cm[id]++;
                }
                in.push_back(id);
            }
        }
        const int mt = (int) in.size() - 1, nb = mt + 1;
        const double ts0 = now();
        st_.t_mtp += ts0 - tm0;
        (echo ? st_.rounds_echo : mt > 0 ? st_.rounds_mtp : st_.rounds_plain)++;
        // proposals: a dry check (recurrent state untouched), then commit only the kept tokens
        const double tb0 = now();
        st_.t_snap += tb0 - ts0;
        model::EvalOpts bo;
        bo.residual  = false;
        bo.window_ok = true;
        bo.record    = mt > 0;
        bo.dry       = mt > 0;
        bo.argmax    = !sampled;
        bo.topk      = tk;
        std::vector<int32_t> bid((size_t) nb);
        lg.resize(sampled ? (size_t) (nb * rw) : 0);
        lid.resize(tk > 0 ? (size_t) (nb * tk) : 0);
        if (!dm.eval(in.data(), nb, bo, sampled ? lg.data() : nullptr, tk > 0 ? lid.data() : bid.data(), err)) return false;
        st_.t_beval += now() - tb0;
        int     a    = 0;
        int32_t next = -1;
        if (!sampled) {
            while (a < mt && in[(size_t) a + 1] == bid[(size_t) a]) toks.push_back(in[(size_t) ++a]);
            next = bid[(size_t) a];
        } else {
            for (;; a++) {
                Dist pb;
                if (tk > 0) {
                    dist_topk(lg.data() + (size_t) a * tk, lid.data() + (size_t) a * tk, tk, pb);
                } else {
                    float * L = lg.data() + (size_t) a * nv;
                    apply_penalties(L, sp_, cur);
                    dist(L, pb);
                }
                if (a == mt) {  // all proposals kept (or none made): the base's own next token
                    next = draw(pb, rng_);
                    qd.push_back(std::move(pb));
                    break;
                }
                const int32_t d  = in[(size_t) a + 1];
                const double  qv = at(pq[(size_t) a], d);
                if (qv > 0 && std::uniform_real_distribution<double>(0.0, 1.0)(rng_) * qv < at(pb, d)) {
                    toks.push_back(d);
                    cur[d]++;
                    qd.push_back(std::move(pb));
                    continue;
                }
                next = draw_residual(pb, pq[(size_t) a], rng_);
                qd.push_back(std::move(pb));
                break;
            }
        }
        toks.push_back(next);
        cur[next]++;
        if (echo) {
            st_.echo_proposed += mt;
            st_.echo_accepted += a;
        } else if (mt > 0) {
            st_.mtp_proposed += mt;
            st_.mtp_accepted += a;
            // per-position survival of MTP proposals and the measured costs, for choose_mtp()
            for (int j = 1; j <= kMaxMtp; j++) {
                mseen_[j] *= 0.97;
                mfail_[j] *= 0.97;
            }
            for (int j = 1; j <= std::min(a + 1, mt); j++) mseen_[j] += 1;
            if (a < mt) mfail_[a + 1] += 1;
            check_ms_ = 0.9 * check_ms_ + 0.1 * 1e3 * (now() - tb0);
            step_ms_  = 0.9 * step_ms_ + 0.1 * 1e3 * (ts0 - tm0) / mt;
        }
        const double tr0 = now();
        if (mt > 0 && !dm.commit(a + 1, err)) return false;  // keep toks.back() before this round + the kept proposals
        st_.t_rback += now() - tr0;
        last_echo_full_ = echo && a == mt;
        if (extend && !last_echo_full_) break;
    }
    return true;
}

bool SpecDecoder::step(std::vector<int32_t> & out, std::string & err) {
    if (d_ && k_ > 0 && mtp_n_ > 0 && m_.has_mtp()) return step_shadow(out, err);
    return step_plain(out, err);
}

bool SpecDecoder::step_plain(std::vector<int32_t> & out, std::string & err) {
    const int64_t nv = m_.hp().n_vocab;
    int           k_ = adaptive_ ? choose_k() : this->k_;
    const int     base_past = m_.n_past();
    if (m_.n_past() + k_ + 1 > m_.n_ctx()) {
        err = "context full";
        return false;
    }
    st_.cycles++;
    st_.k_hist[k_]++;
    if (k_ == 0) {
        const double t0 = now();
        if (!m_.eval(&last_, 1, model::EvalOpts{}, logits_.data(), nullptr, err)) return false;
        st_.t_verify += now() - t0;
        apply_penalties(logits_.data(), sp_, counts_);
        last_ = sample(logits_.data(), -1);
        emit(out, last_);
        st_.emitted++;
        return true;
    }

    // draft k tokens with the base alone, from the committed state. Greedy: argmax on the GPU. Sampling: draw from the
    // base's own sampling distribution q and keep q for the acceptance test.
    double t0 = now();
    m_.save_state();
    std::vector<int32_t> toks(1, last_);
    model::EvalOpts      dopt;
    dopt.residual          = false;
    dopt.window_ok         = true;
    const bool sampled     = sp_.temp > 0;
    dopt.argmax            = !sampled;
    using Dist             = std::vector<std::pair<float, int32_t>>;
    std::vector<Dist> qd(sampled ? (size_t) k_ : 0);
    std::vector<float> dl(sampled ? (size_t) nv : 0);
    TokenCounts        cur = counts_;  // penalties at each position include the drafts before it
    const bool use_mtp = echo_ || (mtp_n_ > 0 && m_.has_mtp() && m_.hidden_row() >= 0);
    if (use_mtp) {
        qd.clear();
        last_echo_full_ = false;
        if (!draft_rounds(k_, toks, qd, cur, err)) return false;
        // long copies: while the drafts end in a copy the base kept entirely, keep copying (one bigger verify
        // instead of several cycles)
        const int big = std::min({ kBigK, m_.max_long_verify() - 1, m_.n_ctx() - base_past - 1 });
        if (echo_ && last_echo_full_ && big > k_) {
            if (!draft_rounds(big, toks, qd, cur, err, true)) return false;
            st_.long_cycles += (int) toks.size() - 1 > k_;
            k_ = (int) toks.size() - 1;
        }
    }
    for (int i = use_mtp ? k_ : 0; i < k_; i++) {
        int32_t id = -1;
        if (!m_.eval(&toks.back(), 1, dopt, sampled ? dl.data() : nullptr, &id, err)) return false;
        if (sampled) {
            apply_penalties(dl.data(), sp_, cur);
            dist(dl.data(), qd[(size_t) i]);
            std::uniform_real_distribution<double> uq(0.0, 1.0);
            double r = uq(rng_);
            id       = qd[(size_t) i].back().second;
            for (auto & e : qd[(size_t) i]) {
                r -= e.first;
                if (r <= 0) {
                    id = e.second;
                    break;
                }
            }
        }
        toks.push_back(id);
        cur[id]++;
    }
    m_.restore_state();
    const double td = now() - t0;
    st_.t_draft += td;
    draft_ms_ = 0.8 * draft_ms_ + 0.2 * (1e3 * td / k_);
    st_.drafted += k_;

    // verify all k + 1 positions with base + residual
    t0 = now();
    // top-k on the GPU when the sampler needs no more (greedy: the top token only)
    const int            vk = sampled ? gpu_topk() : (no_penalties() ? 1 : 0);
    const int64_t        vr = vk > 0 ? vk : nv;
    std::vector<float>   lg((size_t) ((k_ + 1) * vr));
    std::vector<int32_t> vid(vk > 0 ? (size_t) ((k_ + 1) * vk) : 0);
    model::EvalOpts      vopt;
    vopt.record = true;
    vopt.topk   = vk;
    if (!m_.eval(toks.data(), k_ + 1, vopt, lg.data(), vk > 0 ? vid.data() : nullptr, err)) return false;
    const double tv = 1e3 * (now() - t0);
    st_.t_verify += tv / 1e3;
    for (int n = 1; n <= kMaxK + 1; n++) {  // decayed verify ms per batch size
        vt_[n] *= 0.9;
        vn_[n] *= 0.9;
    }
    if (k_ + 1 <= kMaxK + 1) {
        vt_[k_ + 1] += tv;
        vn_[k_ + 1] += 1;
    }

    int                                    acc  = 0;
    int32_t                                next = -1;
    std::uniform_real_distribution<double> u(0.0, 1.0);
    cur = counts_;
    for (; acc < k_; acc++) {
        const int32_t d = toks[(size_t) acc + 1];
        if (vk > 0) {
            cur[d]++;
            if (!sampled) {  // top-1 only: accept iff it is the draft, else it is the next token
                const int32_t top = vid[(size_t) acc];
                if (top == d) {
                    emit(out, d);
                    continue;
                }
                next = top;
                break;
            }
            Dist pdist;
            dist_topk(lg.data() + (size_t) acc * vk, vid.data() + (size_t) acc * vk, vk, pdist);
            const Dist & q  = qd[(size_t) acc];
            const double pd = at(pdist, d), qv = at(q, d);
            if (qv > 0 && u(rng_) * qv < pd) {
                emit(out, d);
                continue;
            }
            next = draw_residual(pdist, q, rng_);
            {  // tree-draft study: rank of the verifier's correction among the base's candidates there
                int r = 0;
                while (r < (int) q.size() && q[(size_t) r].second != next) r++;
                st_.rej_rank[std::min(r, 20)]++;
            }
            break;
        }
        float * L = lg.data() + (size_t) acc * nv;
        apply_penalties(L, sp_, cur);
        cur[d]++;
        if (!sampled) {
            // deterministic draft (q = delta at d): accept with p(d), else resample from p without d
            const double pd = prob(L, d);
            if (pd >= 1.0 || (pd > 0 && u(rng_) < pd)) {
                emit(out, d);
                continue;
            }
            next = sample(L, d);
            break;
        }
        // sampled draft: accept with min(1, p(d)/q(d)), else resample from max(0, p - q) normalised
        Dist pdist;
        dist(L, pdist);
        const Dist & q  = qd[(size_t) acc];
        auto         at = [](const Dist & v, int32_t t) {
            for (auto & e : v) {
                if (e.second == t) return (double) e.first;
            }
            return 0.0;
        };
        const double pd = at(pdist, d), qv = at(q, d);
        if (qv > 0 && u(rng_) * qv < pd) {
            emit(out, d);
            continue;
        }
        next = draw_residual(pdist, q, rng_);
        break;
    }
    if (acc == k_) st_.full_cycles++;
    if (acc == k_) {  // all accepted: bonus token from the last position
        if (vk > 0) {
            if (!sampled) {
                next = vid[(size_t) k_];
            } else {
                Dist pb;
                dist_topk(lg.data() + (size_t) k_ * vk, vid.data() + (size_t) k_ * vk, vk, pb);
                next = draw(pb, rng_);
            }
        } else {
            apply_penalties(lg.data() + (size_t) k_ * nv, sp_, cur);
            next = sample(lg.data() + (size_t) k_ * nv, -1);
        }
    }
    emit(out, next);
    st_.accepted += acc;
    st_.emitted += acc + 1;
    for (int j = 1; j <= kMaxK; j++) {  // decayed per-position counts: drafts seen and first rejections
        seen_[j] *= 0.95;
        fail_[j] *= 0.95;
    }
    for (int j = 1; j <= std::min({ acc + 1, k_, kMaxK }); j++) seen_[j] += 1;
    if (acc < k_ && acc + 1 <= kMaxK) fail_[acc + 1] += 1;

    t0 = now();
    if (m_.can_rollback(acc + 1)) {
        if (!m_.rollback(acc + 1, err)) return false;  // keep last_ and the accepted drafts
    } else {
        // rejected past the recorded part of a long verify: back to the committed state, evaluate the kept prefix
        std::string e2;
        m_.rollback(k_ + 1, e2);  // clears the record
        m_.restore_state();
        if (!m_.eval(toks.data(), acc + 1, model::EvalOpts{}, nullptr, nullptr, err)) return false;
        st_.reruns++;
    }
    st_.t_rollback += now() - t0;
    last_ = next;
    return true;
}

// Pipelined cycle (with a shadow model): the shadow drafts; while the main model verifies those drafts on its own
// thread and CUDA stream, the shadow keeps drafting from the last draft as if all were accepted. When the verify
// accepts every draft, its last row (normally the bonus token's distribution) tests the first pre-drafted token
// instead, and the rest become the next cycle's drafts: that cycle skips its drafting. Exact: every emitted token
// is a draft accepted by speculative sampling against base + residual, or a token sampled from it.
bool SpecDecoder::step_shadow(std::vector<int32_t> & out, std::string & err) {
    const int64_t nv      = m_.hp().n_vocab;
    const bool    sampled = sp_.temp > 0;
    std::uniform_real_distribution<double> u(0.0, 1.0);
    auto at = [](const Dist & v, int32_t t) {
        for (auto & e : v) {
            if (e.second == t) return (double) e.first;
        }
        return 0.0;
    };
    std::vector<int32_t> toks;
    std::vector<Dist>    qd;
    TokenCounts          cur = counts_;
    if (pending_) {
        pending_         = false;
        const int32_t x1 = ptoks_[1];
        bool          ok;
        int32_t       corr = -1;
        if (!sampled) {
            ok   = x1 == pend_top_;
            corr = pend_top_;
        } else {
            const double pv = at(pend_p_, x1), qv = at(pqd_[0], x1);
            ok = qv > 0 && u(rng_) * qv < pv;
            if (!ok) corr = draw_residual(pend_p_, pqd_[0], rng_);
        }
        st_.drafted++;
        st_.pipe_tested++;
        if (!ok) {  // the pre-drafted tokens are off: the correction is the next token, the shadow starts over
            emit(out, corr);
            st_.emitted++;
            last_    = corr;
            d_dirty_ = true;
            return true;
        }
        st_.accepted++;
        st_.pipe_kept++;
        emit(out, x1);
        st_.emitted++;
        last_ = x1;
        toks.assign(ptoks_.begin() + 1, ptoks_.end());
        if (sampled) qd.assign(pqd_.begin() + 1, pqd_.end());
        cur = counts_;
        for (size_t i = 1; i < toks.size(); i++) cur[toks[i]]++;
    }
    const int room = m_.n_ctx() - m_.n_past() - 2;
    int       k    = std::min({ adaptive_ ? choose_k() : k_, room, m_.max_verify() - 1 });
    if (k < 1) {
        d_dirty_ = true;
        k_       = 0;
        return step_plain(out, err);
    }
    st_.cycles++;
    double t0 = now();
    if (toks.empty()) {
        if (d_dirty_) {
            d_->sync_from(m_);
            d_dirty_ = false;
        }
        toks.assign(1, last_);
    }
    // pre-drafted tokens kept: verify just those (this cycle has no drafting phase) when there are enough
    static const int pipe_min = std::getenv("E8_PIPE_MIN") ? std::atoi(std::getenv("E8_PIPE_MIN")) : 4;
    if (toks.size() > 1 && (int) toks.size() - 1 >= pipe_min) k = (int) toks.size() - 1;
    if ((int) toks.size() - 1 < k) {
        no_echo_ = false;
        if (!draft_rounds(k, toks, qd, cur, err)) return false;
    }
    const int kk = (int) toks.size() - 1;  // pre-drafted tokens may exceed k: verify them all
    st_.k_hist[std::min(kk, kMaxK)]++;
    const double td = now() - t0;
    st_.t_draft += td;
    st_.drafted += kk;

    // verify on the main model (own thread), pre-draft on the shadow meanwhile
    t0 = now();
    m_.save_state();
    const int            vk = sampled ? gpu_topk() : (no_penalties() ? 1 : 0);
    const int64_t        vr = vk > 0 ? vk : nv;
    std::vector<float>   lg((size_t) ((kk + 1) * vr));
    std::vector<int32_t> vid(vk > 0 ? (size_t) ((kk + 1) * vk) : 0);
    model::EvalOpts      vopt;
    vopt.record = true;
    vopt.topk   = vk;
    if (std::getenv("E8_TEST_VERIFY_NORES")) vopt.residual = false;  // timing experiment only (not exact)
    bool              vok = false;
    std::string       verr;
    std::atomic<bool> vdone{ false };
    std::thread       vt([&] {
        vok = m_.eval(toks.data(), kk + 1, vopt, lg.data(), vk > 0 ? vid.data() : nullptr, verr);
        vdone = true;
    });
    ptoks_.assign(1, toks.back());
    pqd_.clear();
    bool        pok = false;
    std::string perr;
    {
        const int kp = std::min({ std::max(k, 8), m_.max_verify() - 1, room - kk - 1 });
        if (kp >= 1) {
            TokenCounts pc = cur;
            no_echo_       = true;  // history lags the drafts here
            const double  tp0     = now();
            const int64_t rounds0 = st_.rounds_mtp + st_.rounds_plain + st_.rounds_echo;
            stop_            = &vdone;
            pok              = draft_rounds(kp, ptoks_, pqd_, pc, perr);
            stop_            = nullptr;
            st_.t_pipe += now() - tp0;
            st_.pipe_drafted += (int64_t) ptoks_.size() - 1;
            st_.pipe_rounds += st_.rounds_mtp + st_.rounds_plain + st_.rounds_echo - rounds0;
            no_echo_ = false;
        }
    }
    vt.join();
    if (!vok) {
        err = verr;
        return false;
    }
    const double tv = 1e3 * (now() - t0);
    st_.t_verify += tv / 1e3;
    for (int n = 1; n <= kMaxK + 1; n++) {
        vt_[n] *= 0.9;
        vn_[n] *= 0.9;
    }
    if (kk + 1 <= kMaxK + 1) {
        vt_[kk + 1] += tv;
        vn_[kk + 1] += 1;
    }
    draft_ms_ = 0.8 * draft_ms_ + 0.2 * (1e3 * (td > 0 ? td : 0) / std::max(kk, 1));

    int     acc  = 0;
    int32_t next = -1;
    cur          = counts_;
    auto row_dist = [&](int i, Dist & p) {
        if (vk > 0) {
            dist_topk(lg.data() + (size_t) i * vk, vid.data() + (size_t) i * vk, vk, p);
        } else {
            float * L = lg.data() + (size_t) i * nv;
            apply_penalties(L, sp_, cur);
            dist(L, p);
        }
    };
    for (; acc < kk; acc++) {
        const int32_t d = toks[(size_t) acc + 1];
        if (!sampled) {
            int32_t top;
            if (vk > 0) {
                top = vid[(size_t) acc];
            } else {
                float * L = lg.data() + (size_t) acc * nv;
                apply_penalties(L, sp_, cur);
                top = (int32_t) (std::max_element(L, L + nv) - L);
            }
            cur[d]++;
            if (top == d) {
                emit(out, d);
                continue;
            }
            next = top;
            break;
        }
        Dist pdist;
        row_dist(acc, pdist);
        cur[d]++;
        const Dist & q  = qd[(size_t) acc];
        const double pd = at(pdist, d), qv = at(q, d);
        if (qv > 0 && u(rng_) * qv < pd) {
            emit(out, d);
            continue;
        }
        next = draw_residual(pdist, q, rng_);
        break;
    }
    st_.accepted += acc;
    for (int j = 1; j <= kMaxK; j++) {
        seen_[j] *= 0.95;
        fail_[j] *= 0.95;
    }
    for (int j = 1; j <= std::min({ acc + 1, kk, kMaxK }); j++) seen_[j] += 1;
    if (acc < kk && acc + 1 <= kMaxK) fail_[acc + 1] += 1;

    t0 = now();
    if (acc == kk) {
        st_.full_cycles++;
        st_.emitted += acc;
        if (!m_.rollback(kk + 1, err)) return false;  // keeps everything (clears the record)
        if (pok && ptoks_.size() >= 2) {
            // the last row tests the first pre-drafted token next cycle; no bonus token
            if (!sampled) {
                if (vk > 0) {
                    pend_top_ = vid[(size_t) kk];
                } else {
                    float * L = lg.data() + (size_t) kk * nv;
                    apply_penalties(L, sp_, cur);
                    pend_top_ = (int32_t) (std::max_element(L, L + nv) - L);
                }
            } else {
                row_dist(kk, pend_p_);
            }
            pending_ = true;
            st_.t_rollback += now() - t0;
            return true;
        }
        Dist pb;
        if (!sampled) {
            if (vk > 0) {
                next = vid[(size_t) kk];
            } else {
                float * L = lg.data() + (size_t) kk * nv;
                apply_penalties(L, sp_, cur);
                next = (int32_t) (std::max_element(L, L + nv) - L);
            }
        } else {
            row_dist(kk, pb);
            next = draw(pb, rng_);
        }
        d_dirty_ = true;
    } else {
        st_.emitted += acc;
        d_dirty_ = true;
        if (m_.can_rollback(acc + 1)) {
            if (!m_.rollback(acc + 1, err)) return false;
        } else {
            std::string e2;
            m_.rollback(kk + 1, e2);
            m_.restore_state();
            if (!m_.eval(toks.data(), acc + 1, model::EvalOpts{}, nullptr, nullptr, err)) return false;
            st_.reruns++;
        }
    }
    emit(out, next);
    st_.emitted++;
    st_.t_rollback += now() - t0;
    last_ = next;
    return true;
}
} // namespace e8::runtime
