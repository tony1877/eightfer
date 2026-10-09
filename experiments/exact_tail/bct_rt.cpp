// bct_rt.cpp - bit-exact round trip of B-conditioned exact tails on real BF16 tensors.
// Encoder sees W and the packed base(s). Decoder sees ONLY the packed base bytes, the tail bitstream
// and the escape list, recomputes every cell from the base bytes, parses the self-delimiting code and
// rebuilds the BF16 bit pattern. Every weight is compared bit-for-bit.
//   scheme A: B=IQ4_XS, tail = V2 code of key(W) inside B's cell          (two-level: B | exact)
//   scheme C: B=IQ4_XS, T1 = j-bit uniform refinement (fixed width, no scales), T2 = V2 in sub-cell
//   scheme R: B=IQ4_XS, R=Q4_K of (W - deq B) (current plan), T2 = V2 in (B cell intersect R cell)
// V2 code: [1 sign bit if the cell straddles 0] [unary binade from the top, if >1 binade]
//          [mantissa offset, width ceil(log2(#codes in that binade intersect cell))]. All widths follow from
//          the base and the bits already read, so no entropy coder and no stored lengths are needed.
#include "ggml.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

static const int8_t KV4NL[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
static inline float h2f(uint16_t h) { return ggml_fp16_to_fp32(h); }
static inline float bf2f(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }
static inline int key_of(uint16_t b) { int mag = b & 0x7fff; return (b & 0x8000) ? (32767 - mag) : (32768 + mag); }
static inline uint16_t bits_of_key(int k) { return k >= 32768 ? (uint16_t)(k - 32768) : (uint16_t)(0x8000 | (32767 - k)); }
static double KVAL[65536];
static void init_kval() {
    for (int k = 0; k < 65536; k++) { uint16_t b = bits_of_key(k); int mag = b & 0x7fff;
        KVAL[k] = mag >= 0x7f80 ? ((b & 0x8000) ? -INFINITY : INFINITY) : bf2f(b); }
}
static inline int ceil_key(double x) { int lo = 0, hi = 65535; while (lo < hi) { int m = (lo + hi) >> 1; if (KVAL[m] >= x) hi = m; else lo = m + 1; } return lo; }
static inline int floor_key(double x) { int lo = 0, hi = 65535; while (lo < hi) { int m = (lo + hi + 1) >> 1; if (KVAL[m] <= x) lo = m; else hi = m - 1; } return lo; }
static inline int clog2(int64_t n) { int w = 0; while ((int64_t)1 << w < n) w++; return w; }

struct BitW { std::vector<uint64_t> v; uint64_t nbits = 0;
    void put(uint64_t x, int w) { for (int i = w - 1; i >= 0; i--) { if ((nbits >> 6) >= v.size()) v.push_back(0); if ((x >> i) & 1) v[nbits >> 6] |= 1ull << (63 - (nbits & 63)); nbits++; } } };
struct BitR { const std::vector<uint64_t> * v; uint64_t pos = 0;
    uint64_t get(int w) { uint64_t x = 0; for (int i = 0; i < w; i++) { x = (x << 1) | (((*v)[pos >> 6] >> (63 - (pos & 63))) & 1); pos++; } return x; } };

static void v2_enc(BitW & bw, int lk, int hk, int kw) {
    bool strad = lk < 32768 && hk >= 32768;
    if (strad) bw.put(kw >= 32768, 1);
    int ma, mb, mw;
    if (kw < 32768) { int b = std::min(hk, 32767); ma = 32767 - b; mb = 32767 - lk; mw = 32767 - kw; }
    else { int a = std::max(lk, 32768); ma = a - 32768; mb = hk - 32768; mw = kw - 32768; }
    int ea = ma >> 7, eb = mb >> 7, ew = mw >> 7;
    if (ea == eb) { bw.put(mw - ma, clog2(mb - ma + 1)); return; }
    for (int e = eb; e > ew; e--) bw.put(0, 1);
    if (ew > ea) bw.put(1, 1);
    int slo = std::max(ma, ew << 7), shi = std::min(mb, (ew << 7) + 127);
    bw.put(mw - slo, clog2(shi - slo + 1));
}
static int v2_dec(BitR & br, int lk, int hk) {
    bool strad = lk < 32768 && hk >= 32768;
    bool pos = strad ? br.get(1) : (lk >= 32768);
    int ma, mb;
    if (!pos) { int b = std::min(hk, 32767); ma = 32767 - b; mb = 32767 - lk; }
    else { int a = std::max(lk, 32768); ma = a - 32768; mb = hk - 32768; }
    int ea = ma >> 7, eb = mb >> 7, mw;
    if (ea == eb) mw = ma + (int)br.get(clog2(mb - ma + 1));
    else {
        int ew = eb;
        while (ew > ea && br.get(1) == 0) ew--;
        int slo = std::max(ma, ew << 7), shi = std::min(mb, (ew << 7) + 127);
        mw = slo + (int)br.get(clog2(shi - slo + 1));
    }
    return pos ? 32768 + mw : 32767 - mw;
}

struct Cell { double lo, hi, vhat; bool ok; };
static double MARGIN_IQ4 = 0.0625, MARGIN_Q4K = 1.0/1024;
static void cells_iq4xs(const uint8_t * q, int64_t nw, std::vector<Cell> & out, double ext) {
    for (int64_t s = 0; s < nw / 256; s++) {
        const uint8_t * b = q + s * 136; uint16_t dh, sh; memcpy(&dh, b, 2); memcpy(&sh, b + 2, 2);
        const uint8_t * sl = b + 4; const uint8_t * qs = b + 8; float d = h2f(dh);
        for (int ib = 0; ib < 8; ib++) {
            int ls = ((sl[ib / 2] >> 4 * (ib % 2)) & 0xf) | (((sh >> 2 * ib) & 3) << 4);
            float dl = d * (ls - 32); const uint8_t * qq = qs + 16 * ib; double D = dl;
            for (int j = 0; j < 32; j++) {
                int idx = j < 16 ? (qq[j] & 0xf) : (qq[j - 16] >> 4);
                double a = idx > 0 ? D * (KV4NL[idx - 1] + KV4NL[idx]) * 0.5 : D * (KV4NL[0] - ext * (KV4NL[1] - KV4NL[0]) * 0.5);
                double z = idx < 15 ? D * (KV4NL[idx] + KV4NL[idx + 1]) * 0.5 : D * (KV4NL[15] + ext * (KV4NL[15] - KV4NL[14]) * 0.5);
                double mg = std::fabs(D) * MARGIN_IQ4;
                out[s * 256 + ib * 32 + j] = {std::min(a, z) - mg, std::max(a, z) + mg, D * KV4NL[idx], dl != 0.0f};
            }
        }
    }
}
static void cells_q4k(const uint8_t * q, int64_t nw, std::vector<Cell> & out, double ext) {
    for (int64_t s = 0; s < nw / 256; s++) {
        const uint8_t * b = q + s * 144; uint16_t dh, mh; memcpy(&dh, b, 2); memcpy(&mh, b + 2, 2);
        const uint8_t * sc = b + 4; const uint8_t * qs = b + 16; float d = h2f(dh), dmin = h2f(mh); int is = 0;
        auto gsm = [&](int jj, uint8_t * dd, uint8_t * mm) {
            if (jj < 4) { *dd = sc[jj] & 63; *mm = sc[jj + 4] & 63; }
            else { *dd = (sc[jj + 4] & 0xF) | ((sc[jj - 4] >> 6) << 4); *mm = (sc[jj + 4] >> 4) | ((sc[jj - 0] >> 6) << 4); } };
        for (int j = 0; j < 256; j += 64) {
            uint8_t s1, m1, s2, m2; gsm(is, &s1, &m1); gsm(is + 1, &s2, &m2);
            float d1 = d * s1, mm1 = dmin * m1, d2 = d * s2, mm2 = dmin * m2;
            for (int l = 0; l < 64; l++) {
                int qv = l < 32 ? (qs[l] & 0xF) : (qs[l - 32] >> 4); float dd = l < 32 ? d1 : d2, mm = l < 32 ? mm1 : mm2;
                double g = (double)dd * qv - (double)mm;
                double mg = std::fabs((double)dd) * MARGIN_Q4K;
                out[s * 256 + j + l] = {g - (qv == 0 ? ext : 1.0) * dd * 0.5 - mg, g + (qv == 15 ? ext : 1.0) * dd * 0.5 + mg, g, dd != 0.0f};
            }
            qs += 32; is += 2;
        }
    }
}

static std::vector<uint8_t> quant(ggml_type t, const std::vector<float> & f, int64_t rows, int64_t cols, std::vector<float> & deq) {
    ggml_quantize_init(t); const size_t rs = ggml_row_size(t, cols);
    std::vector<uint8_t> q(rs * rows); deq.resize(f.size()); std::vector<std::thread> th;
    for (int it = 0; it < 4; it++) th.emplace_back([&, it] {
        int64_t r0 = rows * it / 4, r1 = rows * (it + 1) / 4;
        ggml_quantize_chunk(t, f.data(), q.data(), r0 * cols, r1 - r0, cols, nullptr);
        for (int64_t r = r0; r < r1; r++) ggml_get_type_traits(t)->to_float(q.data() + r * rs, deq.data() + r * cols, cols); });
    for (auto & x : th) x.join();
    return q;
}

// final cell for scheme: returns false if the base gives no usable cell (dead block)
struct Final { int lk, hk; bool ok; };

int main(int argc, char ** argv) {
    if (argc < 4) return 1;
    init_kval();
    int64_t rows = atoll(argv[2]), cols = atoll(argv[3]); size_t N = rows * cols;
    std::vector<uint16_t> raw(N); FILE * fp = fopen(argv[1], "rb"); if (fread(raw.data(), 2, N, fp) != N) return 1; fclose(fp);
    std::vector<float> f(N); for (size_t i = 0; i < N; i++) f[i] = bf2f(raw[i]);
    printf("%s [%lld x %lld]\n", argv[1], (long long)rows, (long long)cols);
    std::vector<float> deqB, deqR, deq8;
    auto qB = quant(GGML_TYPE_IQ4_XS, f, rows, cols, deqB);
    std::vector<float> res(N); for (size_t i = 0; i < N; i++) res[i] = f[i] - deqB[i];
    auto qR = quant(GGML_TYPE_Q4_K, res, rows, cols, deqR);
    auto q8 = quant(GGML_TYPE_Q8_0, f, rows, cols, deq8);
    double e8 = 0, ww = 0; for (size_t i = 0; i < N; i++) { e8 += (f[i] - deq8[i]) * (double)(f[i] - deq8[i]); ww += (double)f[i] * f[i]; }
    double rmse8 = std::sqrt(e8 / ww);
    // H(B index)
    { std::vector<double> h(16, 0); for (size_t s = 0; s < N / 256; s++) for (int j = 0; j < 128; j++) { uint8_t x = qB[s * 136 + 8 + j]; h[x & 15]++; h[x >> 4]++; }
      double H = 0; for (double x : h) if (x > 0) H -= x / N * std::log2(x / N); printf("  IQ4_XS index entropy %.3f bits (stored 4) + scales 0.25 -> H(B) <= %.3f\n", H, H + 0.25); }

    const double EXT = 2.0; if (argc > 4) MARGIN_IQ4 = atof(argv[4]);
    std::vector<Cell> cB(N), cR(N);
    // ---------------- encode (encoder may look at W) ----------------
    for (int scheme = 0; scheme < 3; scheme++) {
        const char * nm = scheme == 0 ? "A  B=IQ4_XS | exact V2 tail" : scheme == 1 ? "C  B=IQ4_XS | T1(4b) | T2 V2" : "R  B=IQ4_XS | R=Q4_K res | T2 V2";
        const int J = 4;
        BitW bw, bw1; std::vector<std::pair<uint32_t, uint16_t>> esc;
        double e_lvl2 = 0; size_t ndead = 0; double v3 = 0;
        auto final_cell = [&](size_t i, const std::vector<Cell> & B, const std::vector<Cell> & R, int scheme_, int t1, double & rec) -> Final {
            const Cell & b = B[i];
            if (!b.ok) return {0, 0, false};
            double lo = b.lo, hi = b.hi;
            if (scheme_ == 1) { double step = (hi - lo) / (1 << J); double slo = lo + t1 * step; rec = slo + 0.5 * step; lo = slo; hi = slo + step; }
            if (scheme_ == 2) { const Cell & r = R[i]; if (!r.ok) return {0, 0, false};
                double rlo = b.vhat + r.lo, rhi = b.vhat + r.hi; rec = b.vhat + r.vhat; lo = std::max(lo, rlo); hi = std::min(hi, rhi); }
            int lk = ceil_key(lo), hk = floor_key(hi);
            if (scheme_ == 1) { lk = std::max(lk, ceil_key(b.lo)); hk = std::min(hk, floor_key(b.hi)); }
            return {lk, hk, lk <= hk};
        };
        cells_iq4xs(qB.data(), N, cB, EXT);
        cells_q4k(qR.data(), N, cR, EXT);
        for (size_t i = 0; i < N; i++) {
            int kw = key_of(raw[i]); double w = f[i];
            int t1 = 0; double rec = cB[i].vhat;
            if (scheme == 1 && cB[i].ok) { double step = (cB[i].hi - cB[i].lo) / (1 << J); t1 = (int)std::floor((w - cB[i].lo) / step); t1 = std::max(0, std::min((1 << J) - 1, t1)); }
            Final fc = final_cell(i, cB, cR, scheme, t1, rec);
            if (!cB[i].ok || (scheme == 2 && !cR[i].ok)) { ndead++; bw.put(raw[i], 16); e_lvl2 += (w - rec) * (w - rec); continue; }  // dead block: raw inline, flagged by base
            if (!fc.ok || kw < fc.lk || kw > fc.hk) { esc.push_back({(uint32_t)i, raw[i]}); e_lvl2 += (w - rec) * (w - rec); if (scheme == 1) bw1.put(t1, J); continue; }
            if (scheme == 1) bw1.put(t1, J);
            e_lvl2 += (w - rec) * (w - rec);
            v2_enc(bw, fc.lk, fc.hk, kw);
            v3 += std::log2((KVAL[fc.hk + 1] - KVAL[fc.lk]) / (KVAL[kw + 1] - KVAL[kw]));
        }
        // ---------------- decode (decoder sees only qB, qR, streams, escape list) ----------------
        // wipe and recompute every cell from the packed base bytes only
        std::fill(cB.begin(), cB.end(), Cell{0, 0, 0, false}); std::fill(cR.begin(), cR.end(), Cell{0, 0, 0, false});
        cells_iq4xs(qB.data(), N, cB, EXT); cells_q4k(qR.data(), N, cR, EXT);
        std::vector<Cell> & dB = cB; std::vector<Cell> & dR = cR;
        BitR br{&bw.v}, br1{&bw1.v}; size_t ei = 0, bad = 0;
        for (size_t i = 0; i < N; i++) {
            uint16_t out;
            if (!dB[i].ok || (scheme == 2 && !dR[i].ok)) out = (uint16_t)br.get(16);
            else if (ei < esc.size() && esc[ei].first == i) { out = esc[ei].second; ei++; if (scheme == 1) br1.get(J); }
            else {
                int t1 = scheme == 1 ? (int)br1.get(J) : 0; double rec;
                Final fc = final_cell(i, dB, dR, scheme, t1, rec);
                out = bits_of_key(v2_dec(br, fc.lk, fc.hk));
            }
            if (out != raw[i]) bad++;
        }
        double tailbits = (double)bw.nbits / N, t1bits = (double)bw1.nbits / N, escb = esc.size() * 48.0 / N, offs = 16.0 / 256;
        double base = 4.25 + (scheme == 2 ? 4.5 : 0);
        printf("  [%s] bit-exact mismatches: %zu of %zu | out-of-cell escapes %.2e | dead-block weights (raw inline) %.2e\n", nm, bad, N, (double)esc.size() / N, (double)ndead / N);
        printf("      base %.2f + T1 %.2f + exact-tail %.3f + escapes %.3f + offsets %.3f = %.3f bpw total | level-2 relRMSE %.5f (%.2fx Q8_0)\n",
               base, t1bits, tailbits, escb, offs, base + t1bits + tailbits + escb + offs, std::sqrt(e_lvl2 / ww), std::sqrt(e_lvl2 / ww) / rmse8);
        printf("      exact tail with ideal value-uniform rANS instead of V2: %.3f (+dead-raw %.3f) -> total %.3f bpw\n", v3 / N, ndead * 16.0 / N, base + t1bits + v3 / N + ndead * 16.0 / N + escb + offs);
    }
    return 0;
}
