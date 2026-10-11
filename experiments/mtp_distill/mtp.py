"""The 27B's MTP draft layer in PyTorch, for distilling it against the 4-bit base (docs/SPEEDUP-PLAN.md idea 2).

Mirrors Qwen35::mtp_input / mtp_layer: x = eh_proj([rms(emb(x_{t+1})) * enorm | rms(h_t) * hnorm]); gated attention
(q proj = [Q | gate] per head, per-head q/k RMS norms, NEOX-style RoPE on the first 64 of 256 dims, causal over the
layer's own earlier positions); SwiGLU FFN; hn = rms(.) * shared_head_norm; logits = output.weight @ hn (frozen).

Chained steps (as the drafter runs them, Qwen35::mtp_step): step 1 reads the trunk's h_t and x_{t+1} and is judged
against the base at t + 1; step k > 1 reads step k-1's own hn and x_{t+k} (the token the drafter would have proposed,
when the earlier ones were kept) and is judged against the base at t + k. Step 1 writes the K/V entries of the real
positions; a later step's row t attends to those entries 0..t and to its own chain's entries (one per earlier later
step) plus itself, at positions t + k - 1, as the engine's MTP KV ring holds them during a drafting round.

  python mtp.py eval  [--res] [--steps 3] [--data f.bin ...] [--held 8]
                                             per-step offline acceptance of the stored head on the held-out conversations
  python mtp.py train [--epochs 3] [--lr 2e-5] [--steps 3] [--w 1,0.8,0.6] [--save out.pt] ...
                                             train on all but the last --held conversations, report held-out acceptance
"""
import argparse, os, struct, sys, time
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'third_party', 'llama.cpp', 'gguf-py'))
from gguf import GGUFReader                     # noqa: E402
from gguf.quants import dequantize              # noqa: E402

BASE = r'E:\shoehorn\orca27b.base.gguf'
RES = r'E:\shoehorn\orca27b.res.gguf'
DATA = r'E:\shoehorn\mtp\pilot.bin'
HEAD_CACHE = r'E:\shoehorn\mtp\head_bf16.pt'
DEV = 'cuda'
CHUNK = 128  # rows of vocab-wide logits at a time (16 GB: 256 runs out with a 3-step chain of ~1K rows)
EPS, N_HEAD, N_KV, HD, N_ROT, ROPE_BASE = 1e-6, 24, 4, 256, 64, 1e7
TOP_K, TOP_P = 20, 0.95
IM_END = 248046
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

    def step(self, h, e, pos, ctx=None, chain=()):
        """One MTP step over n rows. h [n, E]: the trunk's normed h_t (step 1) or the previous step's hn; e [n, E]:
        embeddings of the token after it; pos [n]: positions. Step 1 (ctx None) is causal over its own rows. A later
        step's row t attends to ctx's rows 0..t (step 1's K/V) and, on the same row, to each earlier chain step's K/V
        and its own. -> (hn [n, E], (k, v) of this step)"""
        W = self.w
        rms = lambda t, w: F.rms_norm(t, (t.shape[-1],), w, EPS)
        x = F.linear(torch.cat([rms(e, W['nextn_enorm']), rms(h, W['nextn_hnorm'])], -1), W['nextn_eh_proj'])
        n = x.shape[0]
        cur = rms(x, W['attn_norm'])
        qg = F.linear(cur, W['attn_q']).view(n, N_HEAD, 2 * HD)
        q, gate = qg[..., :HD], qg[..., HD:]
        k = F.linear(cur, W['attn_k']).view(n, N_KV, HD)
        v = F.linear(cur, W['attn_v']).view(n, N_KV, HD)
        q = self.rope(rms(q, W['attn_q_norm']), pos)
        k = self.rope(rms(k, W['attn_k_norm']), pos)
        if ctx is None:
            K, V, mask = k, v, None
        else:
            K = torch.cat([ctx[0], *(c[0] for c in chain), k])
            V = torch.cat([ctx[1], *(c[1] for c in chain), v])
            eye = torch.eye(n, dtype=torch.bool, device=x.device)
            mask = torch.cat([torch.ones(n, n, dtype=torch.bool, device=x.device).tril()] + [eye] * (len(chain) + 1), 1)
        rep = N_HEAD // N_KV
        K, V = K.repeat_interleave(rep, dim=1), V.repeat_interleave(rep, dim=1)
        a = F.scaled_dot_product_attention(q.transpose(0, 1), K.transpose(0, 1), V.transpose(0, 1), attn_mask=mask,
                                           is_causal=mask is None)
        a = a.transpose(0, 1).reshape(n, N_HEAD * HD) * torch.sigmoid(gate.reshape(n, N_HEAD * HD))
        x = x + F.linear(a, W['attn_output'])
        y = rms(x, W['post_attention_norm'])
        x = x + F.linear(F.silu(F.linear(y, W['ffn_gate'])) * F.linear(y, W['ffn_up']), W['ffn_down'])
        return rms(x, W['nextn_shared_head_norm']), (k, v)

    def forward(self, h, e):
        """Step 1 only: h: trunk hidden at positions t (normed, as dumped) [n, E]; e: embeddings of x_{t+1} -> hn."""
        return self.step(h, e, torch.arange(h.shape[0], device=h.device))[0]


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
    """The sequences of an mtpdump file as memory-mapped (toks, h, ids, logits) views (a 1M-token dump is ~10 GB)."""
    with open(path, 'rb') as f:
        assert f.read(4) == b'MTPD'
        _, E, K = struct.unpack('<3I', f.read(12))
    mm = np.memmap(path, np.uint8, 'r')
    seqs, off = [], 16
    while off < len(mm):
        n = int(mm[off:off + 4].view(np.uint32)[0])
        off += 4
        toks = mm[off:off + 4 * n].view(np.int32)
        off += 4 * n
        h = mm[off:off + 2 * n * E].view(np.float16).reshape(n, E)
        off += 2 * n * E
        ids = mm[off:off + 4 * n * K].view(np.int32).reshape(n, K)
        off += 4 * n * K
        lg = mm[off:off + 4 * n * K].view(np.float32).reshape(n, K)
        off += 4 * n * K
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


def run_seq(model, head, emb, seq, steps=1, train=False, weights=None, detach=False):
    """Rows t = 0..n-1 (n = len - 1 - steps): step k reads (step k-1's hn, or h_t for k = 1; x_{t+k}) and is judged
    against the base's distribution at t + k. -> (weighted loss per row, [acceptance sum per step], n)."""
    toks, h, ids, lg = seq
    n = len(toks) - 1 - steps
    if n < 8:
        return 0.0, [0.0] * steps, 0
    t = lambda a: torch.from_numpy(np.array(a)).to(DEV)
    hin = t(h[:n]).float()
    pos = torch.arange(n, device=DEV)
    ctx, chain, hns, grads = None, [], [], []
    loss_sum, acc = 0.0, []
    # one autocast region for the whole chain: each region casts the layer's weights to bf16 again, and every step's
    # graph keeps its copy for the backward (3 x 0.74 GB)
    ac = torch.autocast('cuda', dtype=torch.bfloat16)
    ac.__enter__()
    for k in range(1, steps + 1):
        ee = emb[torch.from_numpy(np.array(toks[k:k + n])).long()].to(DEV).float()
        pb, pb_ids = sampler(t(lg[k:k + n]), t(ids[k:k + n]).long())
        if train:  # activations recomputed in the backward: 16 GB holds a 3-step chain of ~1K rows only this way
            hn, kv = torch.utils.checkpoint.checkpoint(model.step, hin, ee, pos + (k - 1), ctx, tuple(chain),
                                                       use_reentrant=False)
        else:
            hn, kv = model.step(hin, ee, pos + (k - 1), ctx, chain)
        if k == 1:
            ctx = kv
        else:
            chain.append(kv)
        # the loss's gradient w.r.t. hn chunk by chunk (the logits are vocab-wide), then one backward for all steps
        x = hn.detach().requires_grad_(train)
        w = weights[k - 1] if weights else 1.0
        a_k = 0.0
        for c0 in range(0, n, CHUNK):
            c1 = min(n, c0 + CHUNK)
            with torch.set_grad_enabled(train):
                logits = (x[c0:c1].to(torch.bfloat16) @ head.T).float()
                if train:
                    lp = torch.log_softmax(logits, -1)
                    loss = -(pb[c0:c1] * lp.gather(1, pb_ids[c0:c1])).sum() * (w / n)
                    loss.backward()
                    loss_sum += float(loss)
            with torch.no_grad():
                top = logits.detach().topk(TOP_K, -1)
                q, q_ids = sampler(top.values, top.indices)
                a_k += float(accept(pb_ids[c0:c1], pb[c0:c1], q_ids, q).sum())
        acc.append(a_k)
        if train:
            hns.append(hn)
            grads.append(x.grad)
        hin = hn.detach() if detach else hn
    ac.__exit__(None, None, None)
    if train:
        torch.autograd.backward(hns, grads)
    return loss_sum, acc, n


def evaluate(model, head, emb, seqs, steps):
    """Held-out acceptance per step, sum min(p_base, q_mtp), averaged over positions."""
    model.eval()
    a, c = np.zeros(steps), 0
    with torch.no_grad():
        for s in seqs:
            _, acc, n = run_seq(model, head, emb, s, steps)
            a, c = a + acc, c + n
    return a / max(c, 1)


def fmt(acc):
    """Per-step acceptance, and the kept fraction of a full chain if each step's acceptance is independent."""
    kept, surv = 0.0, 1.0
    for x in acc:
        surv *= x
        kept += surv
    return ' '.join(f'a{i + 1} {x:.4f}' for i, x in enumerate(acc)) + f' | kept@{len(acc)} {kept / len(acc):.4f}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['eval', 'train'])
    ap.add_argument('--res', action='store_true', help='start from base + residual (the ~8-bit head)')
    ap.add_argument('--data', nargs='+', default=[DATA])
    ap.add_argument('--held', type=int, default=8, help='held-out conversations, from the end of --data')
    ap.add_argument('--steps', type=int, default=3)
    ap.add_argument('--w', default='1,0.8,0.6', help='loss weight per step')
    ap.add_argument('--detach', action='store_true', help='no gradient through the chained hn')
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--lr', type=float, default=2e-5)
    ap.add_argument('--warmup', type=int, default=50, help='optimizer steps of linear warmup, then cosine to 10%%')
    ap.add_argument('--accum', type=int, default=1, help='conversations per optimizer step')
    ap.add_argument('--eval-every', type=int, default=0, help='optimizer steps between held-out evals (0: per epoch)')
    ap.add_argument('--save', help='write the trained layer (bf16 state dict) here after every epoch')
    ap.add_argument('--load', help='start from a layer saved by --save instead of the GGUF')
    a = ap.parse_args()
    t0 = time.time()
    seqs = [s for p in a.data for s in read_data(p)]
    held = seqs[-a.held:]
    # the corpus repeats prompts: no training conversation shares its prompt (up to the first <|im_end|>) with held-out
    prompt = lambda s: bytes(s[0][:int(np.argmax(np.asarray(s[0]) == IM_END)) + 1])
    hp = {prompt(s) for s in held}
    train = [s for s in seqs[:-a.held] if prompt(s) not in hp]
    weights = [float(x) for x in a.w.split(',')][:a.steps]
    weights += [weights[-1]] * (a.steps - len(weights))
    head, emb = load_head_and_embd()
    model = MTP(a.res or a.mode == 'train' or bool(a.load)).to(DEV)
    if a.load:
        model.w.load_state_dict({k: v.float() for k, v in torch.load(a.load).items()})
    name = a.load or ('base+res' if a.res or a.mode == 'train' else 'base')
    print(f'loaded in {time.time() - t0:.0f} s; train {len(train)} conversations, {sum(len(s[0]) for s in train)} '
          f'tokens ({len(seqs) - a.held - len(train)} dropped: prompt in held-out); held-out {len(held)}, '
          f'{sum(len(s[0]) for s in held)} tokens', flush=True)
    if a.mode == 'eval':
        print(f'held-out ({name}): {fmt(evaluate(model, head, emb, held, a.steps))}')
        return
    print(f'start ({name}): held-out {fmt(evaluate(model, head, emb, held, a.steps))}', flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0, fused=True)  # no param-sized temporaries
    total = a.epochs * ((len(train) + a.accum - 1) // a.accum)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1.0, (i + 1) / a.warmup) *
                                              (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * min(1.0, i / total)))))
    it = 0
    for ep in range(a.epochs):
        model.train()
        tl, ta, tn = 0.0, np.zeros(a.steps), 0
        order = np.random.default_rng(ep).permutation(len(train))
        for b0 in range(0, len(order), a.accum):
            opt.zero_grad(set_to_none=True)
            for i in order[b0:b0 + a.accum]:
                loss, acc, n = run_seq(model, head, emb, train[i], a.steps, True, [w / a.accum for w in weights],
                                       a.detach)
                tl, ta, tn = tl + loss * n * a.accum, ta + acc, tn + n
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            it += 1
            if it in (1, 10):
                print(f'  step {it}: {time.time() - t0:.0f} s, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB',
                      flush=True)
            if a.eval_every and it % a.eval_every == 0:
                print(f'  step {it}/{total} ({time.time() - t0:.0f} s, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB): train loss {tl / max(tn, 1):.4f}, '
                      f'held-out {fmt(evaluate(model, head, emb, held, a.steps))}', flush=True)
                model.train()
        print(f'epoch {ep + 1} ({time.time() - t0:.0f} s): train loss {tl / tn:.4f}, train {fmt(ta / tn)}; '
              f'held-out {fmt(evaluate(model, head, emb, held, a.steps))}', flush=True)
        if a.save:
            torch.save({k: v.detach().to(torch.bfloat16).cpu() for k, v in model.w.items()}, a.save)


if __name__ == '__main__':
    main()
