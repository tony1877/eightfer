#include "runtime/spec.h"

#include <algorithm>
#include <chrono>
#include <cmath>

namespace e8::runtime {

namespace {
double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
} // namespace

SpecDecoder::SpecDecoder(model::Qwen35 & m, int k, const SamplerParams & sp, bool adaptive) :
    m_(m), k_(std::min(std::max(0, k), 15)), adaptive_(adaptive && k > 0), sp_(sp),
    rng_(sp.seed ? sp.seed : std::random_device{}()) {}

int SpecDecoder::choose_k() const {
    // per-token acceptance from decayed counts of accepted drafts and first rejections (geometric model)
    const double alpha = std::clamp(acc_n_ / (acc_n_ + rej_n_), 0.05, 0.995);
    // verify cost a + b*k; before enough data, assume the measured 27B shape (b ~ 3% of a per token)
    double a = 350, b = 12;
    const double det = vs_ * vskk_ - vsk_ * vsk_;
    if (vs_ > 3 && det > 1e-6) {
        b = std::max(0.0, (vs_ * vskt_ - vsk_ * vst_) / det);
        a = std::max(1.0, (vst_ - b * vsk_) / vs_);
    }
    int    best = 1;
    double best_rate = 0;
    for (int k = 1; k <= k_; k++) {
        const double tokens = (1 - std::pow(alpha, k + 1)) / (1 - alpha);  // accepted drafts + 1
        const double rate   = tokens / (k * draft_ms_ + a + b * k);
        if (rate > best_rate) {
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
    } else {
        std::sort(p.begin(), p.end(), desc);
    }
    p.resize(n);
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
    const size_t ub = 512;
    for (size_t i = 0; i < prompt.size(); i += ub) {
        const int       n     = (int) std::min(prompt.size() - i, ub);
        const bool      final = i + (size_t) n == prompt.size();
        model::EvalOpts o;
        o.last_only = true;
        if (!m_.eval(prompt.data() + i, n, o, final ? logits_.data() : nullptr, nullptr, err)) return false;
    }
    last_ = sample(logits_.data(), -1);
    out.push_back(last_);
    st_.emitted++;
    st_.prefill_tokens += (int64_t) prompt.size();
    st_.t_prefill += now() - t0;
    return true;
}

void SpecDecoder::begin(const float * last_logits, std::vector<int32_t> & out) {
    logits_.assign(last_logits, last_logits + m_.hp().n_vocab);
    last_ = sample(logits_.data(), -1);
    out.push_back(last_);
    st_.emitted++;
}

bool SpecDecoder::step(std::vector<int32_t> & out, std::string & err) {
    const int64_t nv = m_.hp().n_vocab;
    const int     k_ = adaptive_ ? choose_k() : this->k_;
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
        last_ = sample(logits_.data(), -1);
        out.push_back(last_);
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
    for (int i = 0; i < k_; i++) {
        int32_t id = -1;
        if (!m_.eval(&toks.back(), 1, dopt, sampled ? dl.data() : nullptr, &id, err)) return false;
        if (sampled) {
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
    }
    m_.restore_state();
    const double td = now() - t0;
    st_.t_draft += td;
    draft_ms_ = 0.8 * draft_ms_ + 0.2 * (1e3 * td / k_);
    st_.drafted += k_;

    // verify all k + 1 positions with base + residual
    t0 = now();
    std::vector<float> lg((size_t) (k_ + 1) * nv);
    model::EvalOpts    vopt;
    vopt.record = true;
    if (!m_.eval(toks.data(), k_ + 1, vopt, lg.data(), nullptr, err)) return false;
    const double tv = 1e3 * (now() - t0);
    st_.t_verify += tv / 1e3;
    {
        const double d = 0.9;  // decayed least squares of verify ms on k
        vs_   = d * vs_ + 1;
        vsk_  = d * vsk_ + k_;
        vskk_ = d * vskk_ + (double) k_ * k_;
        vst_  = d * vst_ + tv;
        vskt_ = d * vskt_ + k_ * tv;
    }

    int                                    acc  = 0;
    int32_t                                next = -1;
    std::uniform_real_distribution<double> u(0.0, 1.0);
    for (; acc < k_; acc++) {
        const float * L = lg.data() + (size_t) acc * nv;
        const int32_t d = toks[(size_t) acc + 1];
        if (!sampled) {
            // deterministic draft (q = delta at d): accept with p(d), else resample from p without d
            const double pd = prob(L, d);
            if (pd >= 1.0 || (pd > 0 && u(rng_) < pd)) {
                out.push_back(d);
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
            out.push_back(d);
            continue;
        }
        std::vector<std::pair<double, int32_t>> resid;
        double                                  tot = 0;
        for (auto & e : pdist) {
            const double r = e.first - at(q, e.second);
            if (r > 0) {
                resid.emplace_back(r, e.second);
                tot += r;
            }
        }
        next = pdist.front().second;
        if (tot > 0) {
            double r = u(rng_) * tot;
            for (auto & e : resid) {
                r -= e.first;
                if (r <= 0) {
                    next = e.second;
                    break;
                }
            }
            if (r > 0) next = resid.back().second;
        }
        break;
    }
    if (acc == k_) {
        next = sample(lg.data() + (size_t) k_ * nv, -1);  // all accepted: bonus token from the last position
    }
    out.push_back(next);
    st_.accepted += acc;
    st_.emitted += acc + 1;
    acc_n_ = 0.9 * acc_n_ + acc;
    rej_n_ = 0.9 * rej_n_ + (acc < k_ ? 1 : 0);

    t0 = now();
    if (!m_.rollback(acc + 1, err)) return false;  // keep last_ and the accepted drafts
    st_.t_rollback += now() - t0;
    last_ = next;
    return true;
}

} // namespace e8::runtime
