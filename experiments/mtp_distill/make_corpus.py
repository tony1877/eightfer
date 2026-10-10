import json, random, sys, urllib.request

# Asks the shoehorn server (the B+R target) for answers and writes each conversation as {"text": ...} in Qwen chat
# format, for `shoehorn mtpdump`. Usage: make_corpus.py out.jsonl N [seed]
out_path, n = sys.argv[1], int(sys.argv[2])
rng = random.Random(int(sys.argv[3]) if len(sys.argv) > 3 else 1)
KEY = [l for l in open(r'C:\Users\aless\.dsh\llama-api-key.txt').read().splitlines() if l and not l.startswith('#')][0]

topics = ["a lighthouse in a storm", "a night market in Taipei", "an old violin maker", "the first snow in a mountain town",
          "a robot learning to cook", "a desert caravan at dawn", "a forgotten library", "a fisherman's last day at sea",
          "a city after a power outage", "a child's first day at school", "a garden through four seasons",
          "a train journey across Siberia", "a jazz club in 1950s Paris", "a beekeeper and her hives", "an abandoned space station",
          "a village festival in Sicily", "a long-distance friendship", "a bakery at 4 a.m.", "a museum guard at night",
          "a migrating flock of cranes", "a chess match between rivals", "a volcano island", "an apprentice blacksmith",
          "a rainy afternoon in a bookshop", "a mountain rescue team"]
concepts = ["how vaccines train the immune system", "why the sky is blue", "how compound interest works",
            "how a refrigerator moves heat", "what causes inflation", "how memory works in the brain",
            "why some bridges sway in the wind", "how plate tectonics shaped the continents", "how bread rises",
            "what a hash table is and why it is fast", "how GPS finds your position", "why we have seasons",
            "how noise-cancelling headphones work", "how coral reefs form", "what entropy means in everyday life"]
tasks = [
    lambda: f"Write a vivid short story about {rng.choice(topics)}.",
    lambda: f"Describe {rng.choice(topics)} in two rich, sensory paragraphs.",
    lambda: f"Write a heartfelt letter from someone connected to {rng.choice(topics)}.",
    lambda: f"Explain {rng.choice(concepts)} to a curious teenager, with an everyday analogy.",
    lambda: f"Write a reflective essay of a few paragraphs about {rng.choice(topics)} and what it teaches us.",
    lambda: f"Explain {rng.choice(concepts)} clearly and in depth, as for an educated adult.",
]

def chat(prompt, think):
    body = {"model": "qwen3.8-27b-uncensored-shoehorn", "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 700, "seed": rng.randrange(1, 1 << 30), "chat_template_kwargs": {"enable_thinking": think}}
    req = urllib.request.Request('http://127.0.0.1:8090/v1/chat/completions', data=json.dumps(body).encode(),
                                 headers={'Authorization': 'Bearer ' + KEY, 'Content-Type': 'application/json'})
    msg = json.load(urllib.request.urlopen(req, timeout=1200))['choices'][0]['message']
    return msg.get('reasoning_content') or '', msg.get('content') or ''

with open(out_path, 'a', encoding='utf-8') as f:
    for i in range(n):
        prompt = rng.choice(tasks)()
        think = rng.random() < 0.25
        reasoning, content = chat(prompt, think)
        text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n{reasoning.strip()}\n</think>\n\n{content}<|im_end|>\n"
        f.write(json.dumps({"text": text}) + "\n")
        f.flush()
        print(f"{i + 1}/{n}: {len(content)} chars, think={think}: {prompt[:70]}", flush=True)
