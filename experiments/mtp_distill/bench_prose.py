"""Prose A/B through the server: 4 prompts not in the training corpus x 2 seeds, 500 tokens, thinking off, temp 1.0 /
top_k 20 / top_p 0.95. Reads each request's speculation stats from the server's timing log (--timing-log).

  python bench_prose.py [label]
"""
import json, os, sys, time, urllib.request

URL = 'http://127.0.0.1:8090/v1/chat/completions'
LOG = r'C:\models\bench\shoehorn-timing.jsonl'
KEY = [l for l in open(r'C:\Users\aless\.dsh\llama-api-key.txt').read().splitlines() if l and not l.startswith('#')][0]
PROMPTS = ["Write a short story about an old woman who repairs umbrellas in a seaside town.",
           "Describe a thunderstorm rolling over a wheat field at dusk, in rich sensory detail.",
           "Write a letter from a soldier to his younger sister about the first time he saw the sea.",
           "Write a reflective essay about what walking at night teaches us about a familiar city."]
SEEDS = [1, 2]
label = sys.argv[1] if len(sys.argv) > 1 else ''


def ask(prompt, seed):
    body = {"model": "qwen3.8-27b-uncensored-shoehorn", "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 500, "temperature": 1.0, "top_k": 20, "top_p": 0.95, "seed": seed,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={'Authorization': 'Bearer ' + KEY, 'Content-Type': 'application/json'})
    return json.load(urllib.request.urlopen(req, timeout=1200))


ask("Say hi.", 0)  # load the model / warm up
tot = {k: 0.0 for k in ('gen', 'dec', 'mp', 'mk', 'cyc')}
for p in PROMPTS:
    for s in SEEDS:
        size = os.path.getsize(LOG)
        ask(p, s)
        time.sleep(0.2)
        with open(LOG, 'rb') as f:
            f.seek(size)
            rec = [json.loads(l) for l in f.read().decode().splitlines() if l.strip()][-1]
        sp = rec['spec']
        tot['gen'] += rec['gen_tokens']
        tot['dec'] += rec['decode_s']
        tot['mp'] += sp['mtp_proposed']
        tot['mk'] += sp['mtp_kept']
        tot['cyc'] += sp['cycles']
        print(f"{rec['gen_tokens'] / rec['decode_s']:6.2f} tok/s, MTP kept {sp['mtp_kept']}/{sp['mtp_proposed']} "
              f"= {sp['mtp_kept'] / max(sp['mtp_proposed'], 1):.3f}, {rec['gen_tokens'] / max(sp['cycles'], 1):.1f} "
              f"tok/cycle | seed {s}: {p[:50]}", flush=True)
print(f"{label}: {tot['gen'] / tot['dec']:.2f} tok/s, MTP kept {tot['mk'] / tot['mp']:.3f} "
      f"({int(tot['mk'])}/{int(tot['mp'])}), {tot['gen'] / tot['cyc']:.1f} tokens/cycle, {int(tot['gen'])} tokens")
