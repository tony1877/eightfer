import argparse, json, os, random, threading, urllib.request
from concurrent.futures import ThreadPoolExecutor

# Asks an OpenAI-compatible server for answers and writes each conversation as {"text": ...} in Qwen chat format, for
# `shoehorn mtpdump`. By default the local shoehorn server (the B+R target); for a large corpus, a vLLM pod serving the
# same fine-tune (--url, --key, --workers 64).
#
#   make_corpus.py out.jsonl N [seed] [--url http://host:port] [--key-file path | --key k] [--workers 1]
#
# Conversation i uses its own RNG (seed, i), so a run is reproducible whatever the order the workers finish in. Appends.
ap = argparse.ArgumentParser()
ap.add_argument('out')
ap.add_argument('n', type=int)
ap.add_argument('seed', type=int, nargs='?', default=1)
ap.add_argument('--url', default='http://127.0.0.1:8090')
ap.add_argument('--key-file', default=r'C:\Users\aless\.dsh\llama-api-key.txt')
ap.add_argument('--key', help='overrides --key-file')
ap.add_argument('--model', default='qwen3.8-27b-uncensored-shoehorn')
ap.add_argument('--workers', type=int, default=1)
ap.add_argument('--max-tokens', type=int, default=1024)
ap.add_argument('--start', type=int, default=0, help='first conversation index (to extend a corpus with new ones)')
a = ap.parse_args()
KEY = a.key or [l for l in open(a.key_file).read().splitlines() if l and not l.startswith('#')][0]

topics = ["a lighthouse in a storm", "a night market in Taipei", "an old violin maker", "the first snow in a mountain town",
          "a robot learning to cook", "a desert caravan at dawn", "a forgotten library", "a fisherman's last day at sea",
          "a city after a power outage", "a child's first day at school", "a garden through four seasons",
          "a train journey across Siberia", "a jazz club in 1950s Paris", "a beekeeper and her hives", "an abandoned space station",
          "a village festival in Sicily", "a long-distance friendship", "a bakery at 4 a.m.", "a museum guard at night",
          "a migrating flock of cranes", "a chess match between rivals", "a volcano island", "an apprentice blacksmith",
          "a rainy afternoon in a bookshop", "a mountain rescue team",
          "a lost dog finding its way home", "a retired astronaut tending tomatoes", "a ferry crossing in thick fog",
          "a hospital night shift", "a family reunion after twenty years", "a tailor in wartime Vienna",
          "a teenager's first summer job", "a cartographer mapping an unknown coast", "an orchard during a late frost",
          "a street musician in the subway", "a wedding that almost didn't happen", "an Arctic research station in winter",
          "a small-town newspaper's last issue", "a monastery in the Himalayas", "a power plant operator on New Year's Eve",
          "two strangers stuck in an elevator", "a pearl diver in the Gulf", "a puppeteer and his marionettes",
          "a long-haul truck driver at a diner", "a scientist waiting for results", "a coastal village before a hurricane",
          "a grandmother teaching a recipe", "a heist that goes sideways", "a ghost who wants to be remembered",
          "a vineyard during harvest", "a city seen from a hot-air balloon", "a broken-down car on a desert highway",
          "a marathon runner's last mile", "a lighthouse keeper's diary", "a school play gone wrong",
          "a clockmaker who can slow time", "an immigrant's first winter", "a dragon who hoards books instead of gold",
          "a detective who cannot smell", "a mapmaker of dreams", "a ship's cook on a whaling voyage",
          "a firefighter's day off", "the last bookstore on Earth", "a girl who talks to the wind",
          "a samurai who refuses to fight", "an AI that writes letters to its creator", "a bus driver's regular passengers",
          "a carnival closing for the season", "a boy and a stray cat in Istanbul", "a nurse in a field hospital",
          "a beach town in the off-season", "an archaeologist opening a tomb", "a theatre's opening night",
          "a farmer during a drought", "a ballet dancer's injury", "a radio operator hearing a strange signal",
          "a market in Marrakech", "a rooftop garden in New York", "a trapper in the Canadian north",
          "a family moving out of their childhood home", "a translator at a peace negotiation", "a glassblower in Murano",
          "a snowed-in mountain hut", "a medieval scribe copying a book", "a coral reef at night",
          "a chef losing her sense of taste", "a pilot's emergency landing", "a lonely lighthouse cat",
          "an old couple's last dance", "a submarine crew under the ice", "a poet in a noisy café",
          "a tea plantation in Darjeeling", "a hacker's moral dilemma", "the day the internet went down",
          "a beekeeper's first sting", "a kite festival in Gujarat", "a sculptor carving ice",
          "a rescued horse learning to trust", "a planet with two suns", "a night train to Lisbon"]
concepts = ["how vaccines train the immune system", "why the sky is blue", "how compound interest works",
            "how a refrigerator moves heat", "what causes inflation", "how memory works in the brain",
            "why some bridges sway in the wind", "how plate tectonics shaped the continents", "how bread rises",
            "what a hash table is and why it is fast", "how GPS finds your position", "why we have seasons",
            "how noise-cancelling headphones work", "how coral reefs form", "what entropy means in everyday life",
            "how a jet engine produces thrust", "why the moon has phases", "how photosynthesis works",
            "what a black hole is", "how antibiotics stop bacteria", "why prices rise when supply falls",
            "how the heart pumps blood", "how a neural network learns", "why ice floats",
            "how tides are caused by the moon", "what DNA does in a cell", "how a bill becomes a law",
            "why the Roman Empire fell", "how electricity reaches your home", "how sleep affects learning",
            "what public-key cryptography is", "how rainbows form", "why cats purr",
            "how a credit score is calculated", "what causes earthquakes", "how solar panels make electricity",
            "why the ocean is salty", "how the printing press changed Europe", "what the placebo effect is",
            "how airplanes stay in the air", "why leaves change colour in autumn", "how a nuclear reactor works",
            "what a recession is", "how muscles grow stronger", "how the internet routes a message",
            "why stars twinkle", "how the stock market works", "what causes the northern lights",
            "how a compiler turns code into machine instructions", "why we dream", "how batteries store energy",
            "what climate feedback loops are", "how bees communicate", "how language is learned by babies",
            "why bubbles are round", "how a lock and key work", "what the theory of relativity says about time",
            "how volcanoes erupt", "how a democracy balances power", "why some materials conduct electricity"]
audiences = ["a curious teenager", "a ten-year-old", "an educated adult", "a busy executive", "a grandparent",
             "a first-year university student", "someone who is sceptical"]
tones = ["warm", "melancholy", "humorous", "suspenseful", "hopeful", "quiet and reflective", "lyrical", "matter-of-fact"]
T, C, A, O = (lambda r: r.choice(topics)), (lambda r: r.choice(concepts)), (lambda r: r.choice(audiences)), (lambda r: r.choice(tones))
tasks = [
    lambda r: f"Write a vivid short story about {T(r)}.",
    lambda r: f"Write a {O(r)} short story about {T(r)}.",
    lambda r: f"Describe {T(r)} in two rich, sensory paragraphs.",
    lambda r: f"Write a heartfelt letter from someone connected to {T(r)}.",
    lambda r: f"Write a reflective essay of a few paragraphs about {T(r)} and what it teaches us.",
    lambda r: f"Write a scene, mostly dialogue, set during {T(r)}.",
    lambda r: f"Write a diary entry by someone living through {T(r)}.",
    lambda r: f"Write a poem about {T(r)}, then a short paragraph explaining its images.",
    lambda r: f"Write the opening chapter of a novel about {T(r)}.",
    lambda r: f"Write a news-style feature article about {T(r)}.",
    lambda r: f"Tell the story of {T(r)} from the point of view of an object in the scene.",
    lambda r: f"Write a {O(r)} monologue by a character in {T(r)}.",
    lambda r: f"Explain {C(r)} to {A(r)}, with an everyday analogy.",
    lambda r: f"Explain {C(r)} clearly and in depth, as for an educated adult.",
    lambda r: f"Write a short, engaging blog post about {C(r)}.",
    lambda r: f"Compare two common misconceptions about {C(r)} with what is actually true.",
    lambda r: f"Explain {C(r)} step by step, then summarise it in three sentences.",
    lambda r: f"Write a dialogue in which a teacher explains {C(r)} to {A(r)}.",
]

lock = threading.Lock()
done = 0


def chat(r, prompt, think):
    body = {"model": a.model, "messages": [{"role": "user", "content": prompt}], "max_tokens": a.max_tokens,
            "temperature": 1.0, "top_k": 20, "top_p": 0.95, "seed": r.randrange(1, 1 << 30),
            "chat_template_kwargs": {"enable_thinking": think}}
    req = urllib.request.Request(a.url.rstrip('/') + '/v1/chat/completions', data=json.dumps(body).encode(),
                                 headers={'Authorization': 'Bearer ' + KEY, 'Content-Type': 'application/json',
                                          'User-Agent': 'make_corpus'})  # RunPod's proxy refuses Python-urllib's
    msg = json.load(urllib.request.urlopen(req, timeout=1200))['choices'][0]['message']
    # vLLM's reasoning parser leaves the "\n\n" after </think> at the start of the content
    return msg.get('reasoning_content') or msg.get('reasoning') or '', (msg.get('content') or '').lstrip('\n')


def one(i, f):
    global done
    r = random.Random(a.seed * 1_000_003 + i)
    prompt = r.choice(tasks)(r)
    think = r.random() < 0.25
    for attempt in range(3):
        try:
            reasoning, content = chat(r, prompt, think)
            break
        except Exception as e:  # noqa: BLE001 (a dropped connection costs one conversation, not the run)
            if attempt == 2:
                print(f"#{i}: failed: {e}", flush=True)
                return
    text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n{reasoning.strip()}\n</think>\n\n{content}<|im_end|>\n"
    with lock:
        f.write(json.dumps({"text": text}) + "\n")
        f.flush()
        done += 1
        print(f"{done}/{a.n} (#{i}): {len(reasoning) + len(content)} chars, think={think}: {prompt[:70]}", flush=True)


with open(a.out, 'a', encoding='utf-8') as f, ThreadPoolExecutor(a.workers) as pool:
    for _ in pool.map(lambda i: one(i, f), range(a.start, a.start + a.n)):
        pass
