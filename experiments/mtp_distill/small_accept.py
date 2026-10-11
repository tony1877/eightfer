"""Offline acceptance of a small standalone drafter (e.g. Qwen3.5-0.8B) against the 4-bit base, on mtpdump data.

At each held-out position t, the small model reads the true prefix x_0..x_t and proposes x_{t+1}; the base's
distribution at t comes from the dump (its stored top-32 logits). Both go through the server's sampler (temp 1,
top_k 20, top_p 0.95) and a_t = sum_x min(p_base, q_small). For a chain of k proposals starting at t, the expected
number kept is sum_i prod_{j<i} a_{t+j} (independent acceptances along the true path), comparable to the MTP head's
a1 + a1 a2 + a1 a2 a3 (1.42 for the stock head on the same held-out set).

  python small_accept.py <hf model dir> [--data corpus.bin] [--held 100]
"""
import argparse, time
import numpy as np
import torch
import mtp

THINK_END = 248069

ap = argparse.ArgumentParser()
ap.add_argument('model')
ap.add_argument('--data', default=r'E:\shoehorn\mtp\corpus.bin')
ap.add_argument('--held', type=int, default=100)
a = ap.parse_args()

from transformers import AutoModelForImageTextToText  # noqa: E402 (after mtp, which puts gguf-py on the path)
t0 = time.time()
model = AutoModelForImageTextToText.from_pretrained(a.model, dtype=torch.bfloat16).to(mtp.DEV).eval()
seqs = mtp.read_data(a.data)[-a.held:]
print(f'loaded {a.model} in {time.time() - t0:.0f} s; {len(seqs)} held-out conversations', flush=True)

acc_all, acc_think, acc_ans, chains = [], [], [], {k: [] for k in (1, 2, 3, 4, 5, 6)}
with torch.no_grad():
    for toks, h, ids, lg in seqs:
        x = torch.from_numpy(np.array(toks)).long().to(mtp.DEV)[None]
        logits = model(input_ids=x).logits[0, :-1].float()          # position t predicts x_{t+1}
        top = logits.topk(mtp.TOP_K, -1)
        q, q_ids = mtp.sampler(top.values, top.indices)
        p, p_ids = mtp.sampler(torch.from_numpy(np.array(lg[:-1])).to(mtp.DEV),
                               torch.from_numpy(np.array(ids[:-1])).long().to(mtp.DEV))
        acc = mtp.accept(p_ids, p, q_ids, q).cpu().numpy()
        tk = np.asarray(toks)
        end = np.where(tk == THINK_END)[0]
        cut = int(end[0]) if len(end) else 0                         # positions after </think> are the answer
        has_think = cut > 0 and (cut > np.where(tk == 248068)[0][0] + 3)
        acc_all.append(acc)
        if has_think:
            acc_think.append(acc[:cut])
        acc_ans.append(acc[cut:])
        n = len(acc)
        for k in chains:                                             # expected kept for a k-proposal chain at each t
            e, surv = np.zeros(n - k), np.ones(n - k)
            for i in range(k):
                surv = surv * acc[i:n - k + i]
                e += surv
            chains[k].append(e)

cat = lambda xs: np.concatenate(xs) if xs else np.zeros(1)
print(f'per-token acceptance vs the 4-bit base: all {cat(acc_all).mean():.4f} | answers {cat(acc_ans).mean():.4f} | '
      f'thinking {cat(acc_think).mean():.4f} ({sum(map(len, acc_think))} positions)')
print('expected kept per chain of k proposals: ' + ', '.join(f'k={k} {cat(v).mean():.3f}' for k, v in chains.items()))
print('stock MTP head on the same set: k=1 0.739, k=3 1.425 (a1 0.739, a2 0.606, a3 0.534)')
