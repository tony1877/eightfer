# Unit test: 3-level nested speculative sampling (B -> BR -> A) reproduces the exact target A's joint distribution.
# Toy vocabulary of 6 tokens, context = last 2 tokens; B, BR, A are different random conditionals.
import numpy as np, itertools
rng = np.random.default_rng(7)
Vn = 6
def table(sharp):
    t = rng.dirichlet(np.ones(Vn) * sharp, size=(Vn + 1, Vn + 1)); return t
A = table(0.7)
BR = 0.85 * A + 0.15 * table(0.7); BR /= BR.sum(-1, keepdims=True)     # close to A (like B+R vs BF16)
B = 0.5 * BR + 0.5 * table(0.7); B /= B.sum(-1, keepdims=True)         # further away
def dist(T, seq): a = seq[-2] if len(seq) >= 2 else Vn; b = seq[-1] if len(seq) >= 1 else Vn; return T[a, b]
def spec_step(seq, draft_fn, target, M):
    """one speculative round: draft_fn(seq) -> (token, q-vector); returns list of committed tokens (exact samples of target)."""
    drafts, qs, s = [], [], list(seq)
    for _ in range(M):
        x, q = draft_fn(s); drafts.append(x); qs.append(q); s.append(x)
    out, s = [], list(seq)
    for x, q in zip(drafts, qs):
        p = dist(target, s)
        if rng.random() < min(1.0, p[x] / q[x]):
            out.append(x); s.append(x)
        else:
            r = np.maximum(p - q, 0); r /= r.sum(); y = rng.choice(Vn, p=r); out.append(y); return out
    out.append(rng.choice(Vn, p=dist(target, s)))   # bonus sample
    return out
def level2_sampler():
    """returns a draft function that emits exact BR samples, produced by speculative B->BR rounds (buffered)."""
    buf = []
    def fn(s):
        nonlocal buf
        if not buf:
            buf = spec_step(s, lambda ss: (lambda q: (rng.choice(Vn, p=q), q))(dist(B, ss)), BR, 3)
        x = buf.pop(0)
        return x, dist(BR, s)
    return fn
L = 3; N = 60000
counts = {}
for n in range(N):
    seq = []
    f = level2_sampler()
    while len(seq) < L:
        seq += spec_step(seq, f, A, 4)
        f = level2_sampler()   # L2 buffer is invalid after an L3 round (prefix may have changed)
    k = tuple(seq[:L]); counts[k] = counts.get(k, 0) + 1
chi2, dof = 0.0, 0
for k in itertools.product(range(Vn), repeat=L):
    p = 1.0; s = []
    for x in k: p *= dist(A, s)[x]; s.append(x)
    e = N * p; o = counts.get(k, 0)
    if e > 5: chi2 += (o - e) ** 2 / e; dof += 1
print(f'chi2 {chi2:.1f} on {dof - 1} dof (expect ~{dof - 1} +- {np.sqrt(2 * (dof - 1)):.0f})')
# power check: the same counts tested against the B+R joint (what you would get without the exact level)
chi2b, dofb = 0.0, 0
for k in itertools.product(range(Vn), repeat=L):
    p = 1.0; s = []
    for x in k: p *= dist(BR, s)[x]; s.append(x)
    e = N * p; o = counts.get(k, 0)
    if e > 5: chi2b += (o - e) ** 2 / e; dofb += 1
print(f'power check vs B+R joint: chi2 {chi2b:.1f} on {dofb - 1} dof')
