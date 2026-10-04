# Bit-level anatomy of real Qwen3.8-27B BF16 weights: entropy per field, nested "first-k-bits" heads, exact tails.
import numpy as np, sys
def H(counts):
    p = counts[counts > 0].astype(np.float64); p /= p.sum(); return float(-(p * np.log2(p)).sum())
def rel(a, b): return float(np.sqrt(np.mean((a - b) ** 2) / np.mean(a ** 2)))
for path, rows, cols in [("../nested_quant/w/10.mlp.down_proj.weight.bin", 5120, 17408),
                         ("../nested_quant/w/10.linear_attn.in_proj_qkv.weight.bin", 10240, 5120),
                         ("../nested_quant/w/11.self_attn.q_proj.weight.bin", 12288, 5120)]:
    u = np.fromfile(path, dtype=np.uint16)
    w = (u.astype(np.uint32) << 16).view(np.float32)
    s, e, m = u >> 15, (u >> 7) & 0xFF, u & 0x7F
    print(f"\n{path}  n={u.size/1e6:.1f}M")
    He = H(np.bincount(e, minlength=256)); Hfull = H(np.bincount(u, minlength=65536))
    print(f"  H(sign)={H(np.bincount(s)):.3f}  H(exp)={He:.3f} (8 raw)  H(mantissa)={H(np.bincount(m,minlength=128)):.3f} (7 raw)"
          f"  H(full 16-bit symbol)={Hfull:.3f} bits -> order-0 lossless {Hfull:.2f} bpw ({Hfull/16:.0%})")
    print(f"  exponent range used: {int(e[e>0].min())}..{int(e.max())}, top-8 exponents hold {np.sort(np.bincount(e))[-8:].sum()/u.size:.1%}")
    # nested heads: sign + exponent + top k mantissa bits (truncate) vs round-to-nearest; tail = exact remainder
    for k in range(0, 5):
        hi = m >> (7 - k)
        trunc = ((((s.astype(np.uint32) << 15) | (e.astype(np.uint32) << 7) | (hi.astype(np.uint32) << (7 - k))) << 16).astype(np.uint32)).view(np.float32)
        # round-to-nearest at k bits (may carry into exponent); signed tail stays exact
        lo = m & ((1 << (7 - k)) - 1)
        up = (lo >= (1 << (6 - k))) if k < 7 else np.zeros_like(lo, dtype=bool)
        mag = ((u & 0x7FFF).astype(np.int64) - lo + np.where(up, 1 << (7 - k), 0)).astype(np.uint32)
        rnd = ((((s.astype(np.uint32) << 15) | mag) << 16).astype(np.uint32)).view(np.float32)
        Hhead = H(np.bincount((s.astype(np.int64) << 12) | (e.astype(np.int64) << 4) | hi, minlength=1 << 13))
        print(f"  head e8m{k}: entropy-coded {Hhead:.2f} bpw | rel RMSE trunc {rel(w,trunc):.4f} round {rel(w,rnd):.4f} | exact tail {7-k} bits"
              f" (H={H(np.bincount(lo,minlength=1<<(7-k))):.2f}) | head+tail {Hhead+7-k:.2f} bpw")
    # MX-style: groups of 32 share the max exponent, k-bit signed fixed-point magnitudes (not lossless; for comparison)
    g = w.reshape(-1, 32); emax = np.floor(np.log2(np.abs(g).max(1, keepdims=True) + 1e-30))
    for k in (3, 4, 5, 8):
        step = 2.0 ** (emax + 1 - (k - 1)); q = np.clip(np.round(g / step), -(2 ** (k - 1)) + 1, 2 ** (k - 1) - 1) * step
        print(f"  MXINT{k} (shared exp/32): {k + 8/32:.2f} bpw rel RMSE {rel(g, q):.4f}")
print("\nreference (measured earlier, same tensors): IQ4_XS 4.25bpw 0.077-0.078 | Q4_K 4.5 0.072 | Q5_K 5.5 0.036 | Q6_K 6.56 0.018 | Q8_0 8.5 0.0054")
