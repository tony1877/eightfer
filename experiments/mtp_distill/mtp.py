"""The 27B's MTP draft layer in PyTorch, for distilling it against the 4-bit base (docs/SPEEDUP-PLAN.md idea 2).

Mirrors Qwen35::mtp_input / mtp_layer: x = eh_proj([rms(emb(x_{t+1})) * enorm | rms(h_t) * hnorm]); gated attention
(q proj = [Q | gate] per head, per-head q/k RMS norms, NEOX-style RoPE on the first 64 of 256 dims, causal over the
layer's own earlier positions); SwiGLU FFN; hn = rms(.) * shared_head_norm; logits = output.weight @ hn (frozen).

  python mtp.py eval  [--res]                 offline acceptance of the stored head on the held-out conversations
  python mtp.py train [--epochs 3] [--lr 2e-5] train on the pilot's first 32 conversations, report held-out acceptance
"""
import argparse, os, struct, sys, time
import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'third_party', 'llama.cpp', 'gguf-py'))
from gguf import GGUFReader                     # noqa: E402
from gguf.quants import dequantize              # noqa: E402

BASE = r'E:\shoehorn\orca27b.base.gguf'
RES = r'E:\shoehorn\orca27b.res.gguf'
DATA = r'E:\shoehorn\mtp\pilot.bin'
HEAD_CACHE = r'E:\shoehorn\mtp\head_bf16.pt'
DEV = 'cuda'
EPS, N_HEAD, N_KV, HD, N_ROT, ROPE_BASE = 1e-6, 24, 4, 256, 64, 1e7
TOP_K, TOP_P = 20, 0.95
MATS = ['attn_q', 'attn_k', 'attn_v', 'attn_output', 'ffn_gate', 'ffn_up', 'ffn_down', 'nextn.eh_proj']
NORMS = ['attn_norm', 'post_attention_norm', 'attn_q_norm', 'attn_k_norm', 'nextn.enorm', 'nextn.hnorm',
         'nextn.shared_head_norm']


def tensors(path):
    return {t.name: t for t in GGUFReader(path).tensors}


def deq(t):
    return torch.from_numpy(np.ascontiguousarray(dequantize(t.data, t.tensor_type))).float()


class MTP(torch.nn.Module):
    def __init__(self, with_res):
        super().__init__()
        b = tensors(BASE)
        r = tensors(RES) if with_res else {}
        p = 'blk.64.'
        self.w = torch.nn.ParameterDict()
        for m in MATS + NORMS:
            w = deq(b[p + m + '.weight'])
            if p + m + '.weight' in r:
                w = w + deq(r[p + m + '.weight'])
            self.w[m.replace('.', '_')] = torch.nn.Parameter(w)
        inv = ROPE_BASE ** (-torch.arange(0, N_ROT, 2, dtype=torch.float64) / N_ROT)
        self.register_buffer('inv_freq', inv.float(), persistent=False)

    def rope(self, x, pos):  # x [n, heads, HD]; NEOX pairs (i, i + N_ROT/2) on the first N_ROT dims
        ang = pos[:, None].float() * self.inv_freq[None, :]
        c, s = ang.cos()[:, None, :], ang.sin()[:, None, :]
        h = N_ROT // 2
        x1, x2, rest = x[..., :h], x[..., h:N_ROT], x[..., N_ROT:]
        return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s, rest], dim=-1)

    def forward(self, h, e):
        """h: trunk hidden at positions t (normed, as dumped) [n, E]; e: embeddings of x_{t+1} [n, E] -> hn [n, E]."""
        W = self.w
        rms = lambda t, w: F.rms_norm(t, (t.shape[-1],), w, EPS)
        x = F.linear(torch.cat([rms(e, W['nextn_enorm']), rms(h, W['nextn_hnorm'])], -1), W['nextn_eh_proj'])
        n = x.shape[0]
        pos = torch.arange(n, device=x.device)
        cur = rms(x, W['attn_norm'])
        qg = F.linear(cur, W['attn_q']).view(n, N_HEAD, 2 * HD)
        q, gate = qg[..., :HD], qg[..., HD:]
        k = F.linear(cur, W['attn_k']).view(n, N_KV, HD)
        v = F.linear(cur, W['attn_v']).view(n, N_KV, HD)
        q = self.rope(rms(q, W['attn_q_norm']), pos)
        k = self.rope(rms(k, W['attn_k_norm']), pos)
        rep = N_HEAD // N_KV
        k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        a = F.scaled_dot_product_attention(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1), is_causal=True)
        a = a.transpose(0, 1).reshape(n, N_HEAD * HD) * torch.sigmoid(gate.reshape(n, N_HEAD * HD))
        x = x + F.linear(a, W['attn_output'])
        y = rms(x, W['post_attention_norm'])
        x = x + F.linear(F.silu(F.linear(y, W['ffn_gate'])) * F.linear(y, W['ffn_up']), W['ffn_down'])
        return rms(x, W['nextn_shared_head_norm'])


def load_head_and_embd():
    b = tensors(BASE)
    if os.path.exists(HEAD_CACHE):
        head = torch.load(HEAD_CACHE)
    else:  # IQ4_XS output head -> bf16, in row chunks (the f32 copy would be 5 GB)
        t = b['output.weight']
        raw, rows = t.data, t.data.shape[0]
        head = torch.empty(rows, 5120, dtype=torch.bfloat16)
        for r0 in range(0, rows, 8192):
            head[r0:r0 + 8192] = torch.from_numpy(dequantize(raw[r0:r0 + 8192], t.tensor_type)).to(torch.bfloat16)
        torch.save(head, HEAD_CACHE)
    emb = torch.from_numpy(np.asarray(b['token_embd.weight'].data).view(np.int16)).view(torch.bfloat16)  # memmapped
    return head.to(DEV), emb


def read_data(path=DATA):
    f = open(path, 'rb')
    assert f.read(4) == b'MTPD'
    _, E, K = struct.unpack('<3I', f.read(12))
    seqs = []
    while (b := f.read(4)):
        n = struct.unpack('<I', b)[0]
        toks = np.frombuffer(f.read(4 * n), np.int32).copy()
        h = np.frombuffer(f.read(2 * n * E), np.float16).reshape(n, E).copy()
        ids = np.frombuffer(f.read(4 * n * K), np.int32).reshape(n, K).copy()
        lg = np.frombuffer(f.read(4 * n * K), np.float32).reshape(n, K).copy()
        seqs.append((toks, h, ids, lg))
    return seqs


def sampler(logits, ids):
    """The server's sampling distribution from best-first top logits: top_k 20, temp 1, top_p 0.95 -> (probs, ids)."""
    lg, ids = logits[:, :TOP_K], ids[:, :TOP_K]
    p = torch.softmax(lg.float(), -1)
    keep = (p.cumsum(-1) - p) < TOP_P  # the smallest best-first prefix reaching top_p
    p = p * keep
    return p / p.sum(-1, keepdim=True), ids


def accept(p_ids, p, q_ids, q):
    """sum_x min(p(x), q(x)) per row, both over their own small supports."""
    m = (p_ids[:, :, None] == q_ids[:, None, :]).float()
    return (torch.minimum(p[:, :, None], q[:, None, :]) * m).sum((1, 2))


def run_seq(model, head, emb, seq, train=False):
    """Positions t = 0..n-3: the head at (h_t, x_{t+1}) against the base's distribution at t + 1."""
    toks, h, ids, lg = seq
    n = len(toks) - 2
    hh = torch.from_numpy(h[:n]).to(DEV).float()
    ee = emb[torch.from_numpy(toks[1:n + 1]).long()].to(DEV).float()
    pb, pb_ids = sampler(torch.from_numpy(lg[1:n + 1]).to(DEV), torch.from_numpy(ids[1:n + 1]).to(DEV).long())
    with torch.autocast('cuda', dtype=torch.bfloat16):
        hn = model(hh, ee)
    loss_sum, acc_sum = 0.0, 0.0
    for c0 in range(0, n, 256):
        c1 = min(n, c0 + 256)
        logits = (hn[c0:c1].to(torch.bfloat16) @ head.T).float()
        if train:
            lp = torch.log_softmax(logits, -1)
            loss = -(pb[c0:c1] * lp.gather(1, pb_ids[c0:c1])).sum() / n
            loss.backward(retain_graph=c1 < n)
            loss_sum += float(loss)
        with torch.no_grad():
            top = logits.topk(TOP_K, -1)
            q, q_ids = sampler(top.values, top.indices)
            acc_sum += float(accept(pb_ids[c0:c1], pb[c0:c1], q_ids, q).sum())
    return loss_sum, acc_sum, n


def evaluate(model, head, emb, seqs):
    model.eval()
    a = c = 0
    with torch.no_grad():
        for s in seqs:
            _, acc, n = run_seq(model, head, emb, s)
            a, c = a + acc, c + n
    return a / c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['eval', 'train'])
    ap.add_argument('--res', action='store_true', help='start from base + residual (the ~8-bit head)')
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--lr', type=float, default=2e-5)
    a = ap.parse_args()
    t0 = time.time()
    seqs = read_data()
    train, held = seqs[:32], seqs[32:]
    head, emb = load_head_and_embd()
    model = MTP(a.res or a.mode == 'train').to(DEV)
    print(f'loaded in {time.time() - t0:.0f} s; held-out positions {sum(len(s[0]) - 2 for s in held)}', flush=True)
    if a.mode == 'eval':
        print(f'held-out first-step acceptance ({"base+res" if a.res else "base"} head): {evaluate(model, head, emb, held):.4f}')
        return
    print(f'start (base+res head): held-out acceptance {evaluate(model, head, emb, held):.4f}', flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0)
    for ep in range(a.epochs):
        model.train()
        tl = ta = tn = 0
        for i in np.random.default_rng(ep).permutation(len(train)):
            opt.zero_grad(set_to_none=True)
            loss, acc, n = run_seq(model, head, emb, train[i], train=True)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl, ta, tn = tl + loss * n, ta + acc, tn + n
        print(f'epoch {ep + 1}: train loss {tl / tn:.4f}, train acceptance {ta / tn:.4f}, '
              f'held-out acceptance {evaluate(model, head, emb, held):.4f}', flush=True)


if __name__ == '__main__':
    main()
