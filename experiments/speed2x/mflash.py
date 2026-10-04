"""Flash-Next decode model for section 10. GB = 1e9. [D] derived, [A] assumed."""
GB = 1e9
N_EXP = 24576
EXP_B = (2 * 2560 * 640 * 4.25 + 2560 * 640 * 4.5) / 8 / GB   # 2.662 MB: gate/up IQ4_XS, down IQ4_NL fallback [D, llama-quant.cpp:410]
EXP_R = (2 * 2560 * 640 * 4.5 + 2560 * 640 * 5.5) / 8 / GB    # 2.970 MB: Q4_K, down Q5_0 fallback [D, :416]
NONEXP_P = 36 * 57.9e6 + 12 * 51.4e6 + 48 * (6.2e6 + 13.2e6) + 32.8e6 + 636e6   # 4.30B excl. token embedding [D]
NONEXP = NONEXP_P * 8.5 / 8 / GB                               # 4.57 GB Q8_0
KV_TOK = 12 * 2 * 2 * 256 * 34 / 32                            # 13056 B/token q8_0
IDX_TOK = 12 * 128 * 2 / 4                                     # pooled indexer keys f16, per token
VRAM, RAM = 15.5, 22.0


def capacity(ctx, design='new', R_H=1.0, ram_total=RAM, kv_on_nvme=False):
    kv = ctx * KV_TOK / GB
    idx = ctx * IDX_TOK / GB
    v_fixed = 0.35 + NONEXP + 0.12 + 0.40 + R_H              # ctx reserve, non-expert, GDN/conv/PLE state, buffers, hot R
    r_fixed = 0.40 + 0.50                                     # host overhead, pinned staging
    if design == 'v0':                                        # DESIGN s5: KV in VRAM, token embedding BF16 in RAM
        v = VRAM - v_fixed - kv - idx
        r = ram_total - r_fixed - 1.27
    else:                                                     # KV (+0.1 GB hot blocks) out of VRAM; embedding on NVMe
        v = VRAM - v_fixed - idx - (0.10 if ctx > 16384 else kv)
        r = ram_total - r_fixed - 0.05 - (0 if (ctx <= 16384 or kv_on_nvme) else kv)
    nv, nr = int(v / EXP_B), int(r / EXP_B)
    return nv, nr, (nv + nr) / N_EXP, v, r


def tok_s(ctx, bw, design='new', u=None, h=None, chain_ms=27.0, p_ov=0.25, waste=1.05, R_H=1.0,
          ram_total=RAM, kv_on_nvme=False, miss_scale=1.0):
    nv, nr, c, v, r = capacity(ctx, design, R_H, ram_total, kv_on_nvme)
    if h is None:
        h = c                                                 # uniform routing: hit rate = cached share
    if u is None:
        u = 0.80 if design == 'v0' else 0.85
    miss = 480 * (1 - h) * EXP_B * waste * miss_scale         # GB per token
    t_io = miss / (bw * u) * 1e3
    t_ch = chain_ms + (1.5 if ctx > 16384 else 0) + (3.0 if kv_on_nvme and ctx > 16384 else 0)
    t = max(t_io, t_ch) + p_ov * min(t_io, t_ch)
    return 1e3 / t, dict(h=h, c=c, nv=nv, nr=nr, miss_MB=miss * 1e3, t_io=t_io, t_ch=t_ch)


if __name__ == '__main__':
    print(f"expert B {EXP_B*1e3:.3f} MB, R {EXP_R*1e3:.3f} MB, store {N_EXP*EXP_B:.1f} GB; non-expert {NONEXP_P/1e9:.2f}B = {NONEXP:.2f} GB Q8_0")
    for ctx in (4096, 200000):
        for d in ('v0', 'new'):
            nv, nr, c, v, r = capacity(ctx, d)
            print(f"  {d:3s} ctx {ctx:6d}: VRAM cache {v:5.2f} GB = {nv} experts, RAM cache {r:5.2f} GB = {nr}, cached share {c:.3f}")
    topo = {'T1 both NVMe on CPU lanes (+SATA bg)': 13.5, 'T2 one CPU + one chipset NVMe': 13.0,
            'T3 both behind chipset uplink': 6.8}
    for ctx in (4096, 200000):
        print(f"ctx {ctx}")
        b1, _ = tok_s(ctx, 6.8, 'v0')
        print(f"  v0 single drive / T3: {b1:5.1f} tok/s")
        for name, bw in topo.items():
            bv, _ = tok_s(ctx, bw, 'v0')
            bn, st = tok_s(ctx, bw, 'new')
            bn_eq, _ = tok_s(ctx, bw, 'new', u=0.80)
            bz, stz = tok_s(ctx, bw, 'new', h=min(1, st['c'] + 0.27))
            print(f"  {name:38s}: v0 {bv:5.1f} | new {bn:5.1f} (u equal: {bn_eq:5.1f}) x{bn/bv:.2f} vs v0-same-topology,"
                  f" x{bn/b1:.2f} vs single-drive v0 | io {st['t_io']:.0f} ms chain {st['t_ch']:.0f} | skewed h={stz['h']:.2f}: {bz:5.1f}")
        bn, st = tok_s(ctx, 13.0, 'new', kv_on_nvme=True)
        print(f"  T2 + KV tiered to NVMe: {bn:5.1f} (c {st['c']:.3f})")
        for R_H in (0.0, 2.0):
            bn, st = tok_s(ctx, 13.0, 'new', R_H=R_H)
            print(f"  T2 R_H={R_H}: {bn:5.1f} (c {st['c']:.3f})")
        bl, st = tok_s(ctx, 13.0, 'new', miss_scale=1.88 / 2.662)
        bl3, _ = tok_s(ctx, 6.8, 'new', miss_scale=1.88 / 2.662)
        print(f"  LOSSY cold experts IQ3_XXS: T2 {bl:5.1f}, T3 {bl3:5.1f}")
        for ram in (54.0,):
            b64, st = tok_s(ctx, 13.0, 'new', ram_total=ram, chain_ms=32.0)
            b64s, _ = tok_s(ctx, 6.8, 'new', ram_total=ram, chain_ms=32.0)
            print(f"  HW option 64 GB RAM (54 usable): c {st['c']:.2f}: T2 {b64:5.1f}, T3 {b64s:5.1f} tok/s")
