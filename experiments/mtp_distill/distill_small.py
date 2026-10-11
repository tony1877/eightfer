"""Distills a small standalone drafter (e.g. Qwen3.5-0.8B) toward the 4-bit base on mtpdump data, then reports held-out
acceptance the way small_accept.py does. A quick probe of how far distillation moves acceptance, not the final recipe.

Loss per position t: cross-entropy of the small model's next-token distribution against the base's sampler distribution
at t (top_k 20 / top_p 0.95 of the stored top-32 logits, temp 1), i.e. KL up to a constant. The tied embedding / LM
head stays frozen (a quarter of the 0.8B's parameters); everything else trains in fp32 under bf16 autocast.

  python distill_small.py <hf model dir> [--epochs 1] [--lr 2e-5] [--accum 8] [--save out_dir]
"""
import argparse, time
import numpy as np
import torch
import mtp

ap = argparse.ArgumentParser()
ap.add_argument('model')
ap.add_argument('--data', default=r'E:\shoehorn\mtp\corpus.bin')
ap.add_argument('--held', type=int, default=100)
ap.add_argument('--epochs', type=int, default=1)
ap.add_argument('--lr', type=float, default=2e-5)
ap.add_argument('--accum', type=int, default=8)
ap.add_argument('--eval-every', type=int, default=60)
ap.add_argument('--save')
a = ap.parse_args()

from transformers import AutoModelForImageTextToText  # noqa: E402

seqs = mtp.read_data(a.data)
held = seqs[-a.held:]
prompt = lambda s: bytes(s[0][:int(np.argmax(np.asarray(s[0]) == mtp.IM_END)) + 1])
hp = {prompt(s) for s in held}
train = [s for s in seqs[:-a.held] if prompt(s) not in hp]
model = AutoModelForImageTextToText.from_pretrained(a.model, dtype=torch.float32).to(mtp.DEV)
model.gradient_checkpointing_enable()
for n, p in model.named_parameters():
    p.requires_grad = not ('embed_tokens' in n or 'lm_head' in n or 'visual' in n)
params = [p for p in model.parameters() if p.requires_grad]
print(f'train {len(train)} conversations, {sum(len(s[0]) for s in train)} tokens; trainable '
      f'{sum(p.numel() for p in params) / 1e6:.0f}M of {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M', flush=True)
opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0, fused=True)
total = a.epochs * len(train) // a.accum
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1.0, (i + 1) / 20) * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * min(1.0, i / total)))))
t_ = lambda x: torch.from_numpy(np.array(x)).to(mtp.DEV)


def target(seq):
    toks, h, ids, lg = seq
    p, p_ids = mtp.sampler(t_(lg[:-1]), t_(ids[:-1]).long())
    return t_(toks).long()[None], p, p_ids


def evaluate():
    model.eval()
    acc, n = 0.0, 0
    k3 = []
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for s in held:
            x, p, p_ids = target(s)
            top = model(input_ids=x).logits[0, :-1].float().topk(mtp.TOP_K, -1)
            q, q_ids = mtp.sampler(top.values, top.indices)
            ac = mtp.accept(p_ids, p, q_ids, q)
            acc, n = acc + float(ac.sum()), n + len(ac)
            a_ = ac.cpu().numpy()
            k3.append(a_[:-3] + a_[:-3] * a_[1:-2] + a_[:-3] * a_[1:-2] * a_[2:-1])
    model.train()
    return acc / n, float(np.concatenate(k3).mean())


t0 = time.time()
a0, k0 = evaluate()
print(f'start: held-out acceptance {a0:.4f}, kept for 3 proposals {k0:.3f}', flush=True)
model.train()
it = 0
for ep in range(a.epochs):
    order = np.random.default_rng(ep).permutation(len(train))
    tl, tn = 0.0, 0
    for b0 in range(0, len(order) - a.accum + 1, a.accum):
        opt.zero_grad(set_to_none=True)
        for i in order[b0:b0 + a.accum]:
            x, p, p_ids = target(train[i])
            with torch.autocast('cuda', dtype=torch.bfloat16):
                hid = model.model(input_ids=x).last_hidden_state[0, :-1]
            n = hid.shape[0]
            loss_sum = 0.0
            for c0 in range(0, n, 256):  # vocab-wide logits in chunks, gradients accumulated into hid
                c1 = min(n, c0 + 256)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    lgt = model.lm_head(hid[c0:c1]).float()
                lp = torch.log_softmax(lgt, -1)
                loss = -(p[c0:c1] * lp.gather(1, p_ids[c0:c1])).sum() / (n * a.accum)
                loss.backward(retain_graph=c1 < n)
                loss_sum += float(loss)
            tl, tn = tl + loss_sum * a.accum, tn + 1
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        it += 1
        if it % a.eval_every == 0 or it == total:
            ac, k3 = evaluate()
            print(f'step {it}/{total} ({time.time() - t0:.0f} s): train loss {tl / max(tn, 1):.4f}; held-out acceptance '
                  f'{ac:.4f}, kept for 3 proposals {k3:.3f}', flush=True)
            tl, tn = 0.0, 0
if a.save:
    model.save_pretrained(a.save)
print('stock MTP head on the same set: acceptance 0.739 (first step), kept for 3 proposals 1.425')
