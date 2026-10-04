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

SpecDecoder::SpecDecoder(model::Qwen35 & m, int k, const SamplerParams & sp) :
    m_(m), k_(std::max(0, k)), sp_(sp), rng_(sp.seed ? sp.seed : std::random_device{}()) {}

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

bool SpecDecoder::step(std::vector<int32_t> & out, std::string & err) {
    const int64_t nv = m_.hp().n_vocab;
    if (m_.n_past() + k_ + 1 > m_.n_ctx()) {
        err = "context full";
        return false;
    }
    st_.cycles++;
    if (k_ == 0) {
        const double t0 = now();
        if (!m_.eval(&last_, 1, model::EvalOpts{}, logits_.data(), nullptr, err)) return false;
        st_.t_verify += now() - t0;
        last_ = sample(logits_.data(), -1);
        out.push_back(last_);
        st_.emitted++;
        return true;
    }

    // draft k tokens with the base alone, from the committed state
    double t0 = now();
    m_.save_state();
    std::vector<int32_t> toks(1, last_);
    model::EvalOpts      dopt;
    dopt.residual = false;
    dopt.argmax   = true;
    for (int i = 0; i < k_; i++) {
        int32_t id = -1;
        if (!m_.eval(&toks.back(), 1, dopt, nullptr, &id, err)) return false;
        toks.push_back(id);
    }
    m_.restore_state();
    st_.t_draft += now() - t0;
    st_.drafted += k_;

    // verify all k + 1 positions with base + residual
    t0 = now();
    std::vector<float> lg((size_t) (k_ + 1) * nv);
    model::EvalOpts    vopt;
    vopt.record = true;
    if (!m_.eval(toks.data(), k_ + 1, vopt, lg.data(), nullptr, err)) return false;
    st_.t_verify += now() - t0;

    int                                    acc  = 0;
    int32_t                                next = -1;
    std::uniform_real_distribution<double> u(0.0, 1.0);
    for (; acc < k_; acc++) {
        const float * L  = lg.data() + (size_t) acc * nv;
        const int32_t d  = toks[(size_t) acc + 1];
        const double  pd = prob(L, d);
        if (pd >= 1.0 || (pd > 0 && u(rng_) < pd)) {
            out.push_back(d);
            continue;
        }
        next = sample(L, d);  // rejected: resample from p without d
        break;
    }
    if (acc == k_) {
        next = sample(lg.data() + (size_t) k_ * nv, -1);  // all accepted: bonus token from the last position
    }
    out.push_back(next);
    st_.accepted += acc;
    st_.emitted += acc + 1;

    t0 = now();
    if (!m_.rollback(acc + 1, err)) return false;  // keep last_ and the accepted drafts
    st_.t_rollback += now() - t0;
    last_ = next;
    return true;
}

} // namespace e8::runtime
