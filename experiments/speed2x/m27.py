"""Unified 27B decode model for the combined plan (section 10).
One timeline per decode cycle: verify (CPU slice of R + PCIe DMA slice of R + streamed KV, sharing one DRAM cap),
GPU drafting inside the verify window (VRAM time shared with verify kernels, derated), bubbles when no window is ready.
All hardware numbers are ASSUMED [A]; byte counts are DERIVED [D]. Units: GB (1e9), GB/s, ms unless noted.
"""
import random, math
from dataclasses import dataclass, replace

GB = 1e9
P_BIG = 25.62e9
B = P_BIG * 4.25 / 8 / GB          # 13.61 GB  IQ4_XS base, all big matrices incl. lm_head
R = P_BIG * 4.5 / 8 / GB           # 14.41 GB  Q4_K residual
KVB = 16 * 2 * 4 * 256 * 34 / 32   # 34816 B/token q8_0 (16 attn layers)
KVL = KVB / 16                     # 2176 B/token/layer


@dataclass
class HW:
    vram: float = 770.0     # effective VRAM read GB/s
    D: float = 46.0         # pinned H2D GB/s under WDDM
    cap: float = 68.0       # combined DRAM read cap (cores + DMA)
    cpu_mem: float = 60.0   # core-side DRAM read cap (CCD link)
    tmac: float = 0.525     # Q4_K x q8 TMAC/s on the 7 compute cores (custom VNNI kernel)
    h_us: float = 60.0      # CPU idle per CPU<->GPU handoff (sync + small copies + GPU add/norm)
    n_hand: int = 257
    gdn_ms: float = 0.05    # GPU-only GDN recurrence per GDN layer per verify (N<=24)
    fa_tflops: float = 50.0 # custom q8_0 split-KV attention kernel
    tail_ms: float = 3.0    # sampling over N positions, accept, GDN replay commit, next launch
    derate: float = 0.85    # usable fraction of leftover GPU time for drafting during verify
    col_ms: float = 0.12    # extra cost per extra column in a B pass (memory-bound GEMV)
    mtp_ms: float = 0.6     # one MTP step (MTP layer + 32k-row draft head)
    replay_ms: float = 0.60 # rootless GDN replay: per-hypothesis state materialise+write (3x151 MB/770)
    replay_fix: float = 1.2 # sequential replay over <=36 tokens x 48 layers (~0.7 us/token/layer)
    pass_misc: float = 0.5
    mtp: tuple = (0.75, 0.60, 0.50)   # chained MTP acceptance vs B (inner speculative sampling)


CENTRAL = HW()
PESS = HW(D=42, cap=60, tmac=0.26, h_us=100, fa_tflops=40, derate=0.80, mtp=(0.65, 0.50, 0.40))
OPT = HW(D=50, cap=75, tmac=0.875, h_us=40, fa_tflops=65, derate=0.90, mtp=(0.80, 0.70, 0.60))


def attn_layer_s(N, ctx, hw):
    flop = 4 * N * 24 * ctx * 256
    return max(flop / (hw.fa_tflops * 1e12), ctx * KVL / GB / hw.vram) + 0.05e-3


def verify(N, ctx, hw=CENTRAL, R_res=0.0, W=4096, stage=0.213, cpu_on=True, kvscale=1.0):
    """Return (T_v ms, CPU share GB, DMA GB, CPU-idle S ms, DRAM GB)."""
    R_ram = R - R_res
    kv_ram = max(0, ctx - W) * KVB / GB * kvscale
    c = min(hw.cpu_mem, hw.tmac * 1e12 * 0.5625 / N / GB) if cpu_on else 0.0
    S = hw.n_hand * hw.h_us * 1e-6 + 48 * hw.gdn_ms * 1e-3 + 16 * attn_layer_s(N, ctx, hw)
    if kv_ram > 0:
        S += 16 * max(0.0, kv_ram / 16 - stage) / hw.D       # KV that could not be pre-staged: DMA-only, CPU idle
    rate = min(c + hw.D, hw.cap)
    work = R_ram + kv_ram
    T = S + max(0.0, work - hw.D * S) / rate
    cpu_share = (T - S) * c * (rate / (c + hw.D)) if c > 0 else 0.0
    cpu_share = min(cpu_share, R_ram)
    dma = work - cpu_share
    return (T * 1e3 + hw.tail_ms, cpu_share, dma, S * 1e3, work)


def gpu_busy(N, ctx, hw, dma, R_res=0.0, W=4096):
    kv_ram = max(0, ctx - W) * KVB / GB
    g = B / hw.vram * 1e3 + hw.col_ms * (N - 1)            # B GEMM for N columns
    g += R_res / hw.vram * 1e3                              # resident R slice
    g += 2 * (dma - kv_ram) / hw.vram * 1e3                 # R ring: copy-engine write + GEMM read
    g += kv_ram / hw.vram * 1e3                             # KV staging write (read is inside attention)
    g += 16 * attn_layer_s(N, ctx, hw) * 1e3
    g += hw.n_hand * 0.015 + 48 * hw.gdn_ms + 1.0
    return g


def t_pass(w, ctx, hw, Wd=6144):
    cols = 4 * w
    t = B / hw.vram * 1e3 + hw.col_ms * (cols - 1) + hw.replay_fix + hw.replay_ms * w
    t += 3 * (hw.mtp_ms + 0.05 * (w - 1)) + hw.pass_misc
    t += min(ctx, Wd) * KVB / GB / hw.vram * 1e3
    return t


def tok_per_pass(rnd, mtp):
    k = 1
    for a in mtp:
        if rnd.random() < a:
            k += 1
        else:
            break
    return k


def sim(ctx, alpha, hw=CENTRAL, K_max=16, K_b=6, m_hedge=3, p_hard=0.30, a_easy=0.98, det=(0.75, 0.15),
        cov=(0.55, 0.40), roots=1, bonus_hedge=True, cycles=20000, seed=11, alpha_pen=0.0, R_res=0.0, W=4096, stage=0.213,
        pipeline=True, use_mtp=True, cpu_on=True, kvscale=1.0, return_stats=False):
    rnd = random.Random(seed)
    a_hard = (alpha - (1 - p_hard) * a_easy) / p_hard - alpha_pen / p_hard
    mtp = hw.mtp if use_mtp else ()

    def gen(npass, rnd=rnd):
        out = []
        for _ in range(npass):
            for _ in range(tok_per_pass(rnd, mtp)):
                h = rnd.random() < p_hard
                f = rnd.random() < (det[0] if h else det[1])
                out.append((h, f))
        return out

    tp1 = t_pass(1, ctx, hw)
    chain = []
    t = 0.0; tok = 0; bubbles = 0; Ns = 0; Tvs = 0.0
    cache = {}
    for _ in range(cycles):
        if not chain:
            bubbles += 1
            while len(chain) < K_b:
                t += tp1
                chain += gen(1)
        k = min(len(chain), K_max)
        window, extra = chain[:k], chain[k:]
        N = k + 1
        if N not in cache:
            Tv, x, dma, S, work = verify(N, ctx, hw, R_res, W, stage, cpu_on, kvscale)
            g = gpu_busy(N, ctx, hw, dma, R_res, W)
            cache[N] = (Tv, g)
        Tv, g = cache[N]
        Ns += N; Tvs += Tv
        hedged = [i for i, (h, f) in enumerate(window) if f][:m_hedge] if pipeline else []
        if pipeline:
            w = 1 + roots * len(hedged) + (roots if bonus_hedge else 0)
            npass = int(hw.derate * max(0.0, Tv - g) / t_pass(w, ctx, hw))
        else:
            npass = 0
        leaf = gen(npass)
        # outcome
        j = None
        for i, (h, f) in enumerate(window):
            if rnd.random() >= (a_hard if h else a_easy):
                j = i
                break
        if j is None:
            cont = extra + leaf
            if cont:
                h, f = cont[0]
                if rnd.random() < (a_hard if h else a_easy):
                    tok += k + 1
                    chain = cont[1:]
                else:
                    tok += k + 1
                    chain = gen(npass) if (pipeline and bonus_hedge and f and rnd.random() < (cov[0] if h else cov[1])) else []
            else:
                tok += k + 1
                chain = []
        else:
            tok += j + 1
            h = window[j][0]
            if j in hedged and rnd.random() < (cov[0] if h else cov[1]):
                chain = gen(npass)
            else:
                chain = []
        t += Tv
    rate = tok / t * 1e3
    if return_stats:
        return dict(tok_s=rate, tok_per_cycle=tok / cycles, bubble_frac=bubbles / cycles, mean_N=Ns / cycles,
                    mean_Tv=Tvs / cycles, ms_per_cycle=t / cycles)
    return rate


def best(ctx, alpha, hw=CENTRAL, Ks=(8, 10, 12, 14, 16, 20, 24), Kbs=(3, 4, 6, 8), **kw):
    b = None
    for K in Ks:
        for Kb in Kbs:
            if Kb > K:
                continue
            r = sim(ctx, alpha, hw, K_max=K, K_b=Kb, cycles=6000, **kw)
            if b is None or r > b[0]:
                b = (r, K, Kb)
    st = sim(ctx, alpha, hw, K_max=b[1], K_b=b[2], cycles=30000, return_stats=True, **kw)
    return st, b[1], b[2]


# ---------------- v0 baseline (current DESIGN.md, serial draft then verify) ----------------
def v0_verify(N, ctx, hw, W=4096, mode='best'):
    kv_ram = max(0, ctx - W) * KVB / GB
    c = min(hw.cpu_mem, hw.tmac * 8 / 7 * 1e12 * 0.5625 / N / GB)  # v0 may use all 8 cores
    S = hw.n_hand * hw.h_us * 1e-6 + 48 * hw.gdn_ms * 1e-3 + 16 * attn_layer_s(N, ctx, hw)
    t_cpu = S + R / c + kv_ram / hw.D                          # CPU does all R; KV still has to be DMA'd (serial)
    t_dma = (R + kv_ram) / hw.D + 48 * hw.gdn_ms * 1e-3 + 16 * attn_layer_s(N, ctx, hw) + 0.005
    if mode == 'doc':
        return 250.0
    return min(t_cpu, t_dma) * 1e3 + hw.tail_ms


def v0(ctx, alpha, hw=CENTRAL, mode='best', t_draft=18.5, alpha_pen=0.0):
    a = alpha - alpha_pen
    bestv = None
    for k in range(1, 25):
        Tv = v0_verify(k + 1, ctx, hw, mode=mode)
        E = (1 - a ** (k + 1)) / (1 - a)
        td = t_draft + (min(ctx, 6144) * KVB / GB / hw.vram * 1e3 if ctx > 4096 else 0)
        r = E / (k * td + Tv) * 1e3
        if bestv is None or r > bestv[0]:
            bestv = (r, k, Tv)
    return bestv


if __name__ == '__main__':
    for ctx in (4096, 200000):
        for N in (7, 12, 16, 20):
            Tv, x, dma, S, work = verify(N, ctx)
            g = gpu_busy(N, ctx, CENTRAL, dma)
            print(f"ctx {ctx:6d} N {N:2d}: Tv {Tv:5.0f} ms  CPU {x:5.2f} GB  DMA {dma:5.2f} GB  S {S:4.0f} ms  GPU busy {g:4.0f} ms")
    print('t_pass w=1 4k %.1f ms, w=4 %.1f ms; 200k w=1 %.1f' % (t_pass(1, 4096, CENTRAL), t_pass(4, 4096, CENTRAL),
                                                             t_pass(1, 200000, CENTRAL)))
