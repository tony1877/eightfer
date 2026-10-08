"""Debug helper for M4: compares HF's QSA token selection at the first indexed layer with a numpy re-implementation
of the algorithm shoehorn uses (pooled block keys, relu-summed head scores, top budget/r blocks + own tail)."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import run4  # noqa: E402

model = run4.AutoModelForCausalLM.from_config(run4.make_config(), dtype=torch.float32, attn_implementation="eager").eval()
run4.randomize(model, 1234)
rng = np.random.default_rng(1234)
toks = rng.integers(1000, 200000, size=256).astype(np.int64)
toks[100] = run4.EOS
T = 40

cap = {}
layer = next(i for i, t in enumerate(model.config.layer_types) if t != "linear_attention")
idx = model.model.layers[layer].self_attn.indexer


def pre(mod, args, kwargs):
    cap["h"] = (args[0] if args else kwargs["hidden_states"]).detach()
    cap["pe"] = args[1] if len(args) > 1 else kwargs["position_embeddings"]


def post(mod, args, kwargs, out):
    cap["sel"] = out.detach()


idx.register_forward_pre_hook(pre, with_kwargs=True)
idx.register_forward_hook(post, with_kwargs=True)
with torch.no_grad():
    model(torch.from_numpy(toks[:T])[None])

h = cap["h"][0].float()
cos, sin = cap["pe"]
r, budget, d, H = idx.compress_ratio, idx.token_budget, idx.index_head_dim, idx.index_n_heads
K = budget // r
qk = h @ idx.index_qk_proj.weight.T
q, kraw = qk[:, : H * d].reshape(T, H, d), qk[:, H * d :]


def rms(x, w, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * (1 + w)


def rope(x, c, s):  # x [..., d], c/s [rot]
    rot = c.shape[-1]
    xr, xp = x[..., :rot], x[..., rot:]
    x1, x2 = xr[..., : rot // 2], xr[..., rot // 2 :]
    return torch.cat([xr * c + torch.cat([-x2, x1], -1) * s, xp], -1)


q = rms(q, idx.q_layernorm.weight)
c0, s0 = cos[0], sin[0]
q = torch.stack([rope(q[t], c0[t], s0[t]) for t in range(T)])
hf_sel = (cap["sel"][0, 0] == 0) if cap["sel"].is_floating_point() else cap["sel"][0, 0]

mism = 0
for p in range(T):
    nb = (p + 1) // r
    if nb == 0:
        continue
    blocks = kraw[: nb * r].reshape(nb, r, d).mean(1)
    pk = rms(blocks, idx.k_layernorm.weight)
    pk = torch.stack([rope(pk[b], c0[b * r], s0[b * r]) for b in range(nb)])
    sc = torch.relu(q[p] @ pk.T).sum(0) / d ** 0.5  # [nb]
    top = sc.topk(min(K, nb)).indices.sort().values.tolist()
    mine = set(j for b in top for j in range(b * r, b * r + r)) | set(range(nb * r, p + 1))
    hf = set(torch.nonzero(hf_sel[p]).flatten().tolist())
    if mine != hf:
        mism += 1
        if mism <= 3:
            print(f"pos {p}: nb {nb} scores {[round(v, 4) for v in sc.tolist()]}")
            print(f"   mine blocks {top}, hf positions {sorted(hf)}")
print("numpy re-implementation vs HF selection:", "MATCH" if mism == 0 else f"{mism} positions differ")

# per-position HF selection as block lists (for comparing with shoehorn's E8_QSA_DUMP output)
if len(sys.argv) > 1 and sys.argv[1] == "--list":
    for p in range(16, T):
        hf = sorted(torch.nonzero(hf_sel[p]).flatten().tolist())
        print(f"hf pos {p}: {hf}")
