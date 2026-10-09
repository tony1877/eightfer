"""Agent reliability test: builds a fixture folder with 100 small tasks whose answers are known.

    python tests/agent_reliability/gen.py [--root C:\\tmp\\reltest] [--seed 7]

Writes the fixture under <root>, the prompt to paste into the agent at <root>\\PROMPT.txt, and the answer key
outside the fixture (<root>-key\\expected.json) so the agent never sees it. Grade with check.py afterwards.
Each task needs about one tool call: reading, searching, globbing, editing and writing files, PowerShell
computations, JSON lookups and background jobs.
"""
import argparse, hashlib, json, os, random, shutil

WORDS = ("amber basil cedar delta ember fjord garnet harbor indigo juniper kestrel lumen meadow nectar onyx pepper "
         "quartz raven saffron tundra umber velvet willow xenon yarrow zephyr").split()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"C:\tmp\reltest")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rnd = random.Random(a.seed)
    root, key_dir = a.root, a.root.rstrip("\\/") + "-key"
    for d in (root, key_dir):
        shutil.rmtree(d, ignore_errors=True)
    for sub in ("data", "edit", "out", "csv", "logs", "bin", "config", "tree"):
        os.makedirs(os.path.join(root, sub))
    os.makedirs(key_dir)
    w = lambda rel, text: open(os.path.join(root, rel), "w", encoding="utf-8", newline="\n").write(text)
    tasks = []  # {kind, text, answer | effect}

    # data files: 20 files of key = value lines, plus one unique code per file for grep tasks
    keys = [f"{x}_{y}" for x in WORDS[:10] for y in ("id", "level", "mode")]
    codes = {}
    for i in range(1, 21):
        vals = {k: str(rnd.randint(100, 99999)) for k in rnd.sample(keys, 18)}
        code = f"ZX-{rnd.randint(1000, 9999)}-{rnd.choice(WORDS).upper()}"
        codes[f"f{i:02}.txt"] = code
        lines = [f"{k} = {v}" for k, v in vals.items()]
        lines.insert(rnd.randint(0, len(lines)), f"# ref {code}")
        w(f"data/f{i:02}.txt", "\n".join(lines) + "\n")
        for k in rnd.sample(list(vals), 2 if i <= 10 else 1):
            if sum(t["kind"] == "read" for t in tasks) < 30:
                tasks.append({"kind": "read", "text": f"Read data\\f{i:02}.txt. What is the value of {k}?", "answer": vals[k]})
    for f, code in rnd.sample(sorted(codes.items()), 15):
        tasks.append({"kind": "grep", "text": f"Which file in data\\ contains the code {code}? Answer with the file name only.", "answer": f})

    # tree for glob tasks
    exts = ["log", "txt", "md", "json", "csv"]
    files = []
    for d in range(6):
        for _ in range(rnd.randint(3, 9)):
            files.append(f"tree/d{d}/{rnd.choice(WORDS)}{rnd.randint(1, 99)}.{rnd.choice(exts)}")
    for f in set(files):
        os.makedirs(os.path.dirname(os.path.join(root, f)), exist_ok=True)
        w(f, "x\n")
    files = sorted(set(files))
    for _ in range(10):
        if rnd.random() < .5:
            e = rnd.choice(exts)
            n = sum(f.endswith("." + e) for f in files)
            tasks.append({"kind": "glob", "text": f"How many .{e} files are there anywhere under tree\\ (all subfolders)?", "answer": str(n)})
        else:
            d, e = rnd.randrange(6), rnd.choice(exts)
            n = sum(f.startswith(f"tree/d{d}/") and f.endswith("." + e) for f in files)
            tasks.append({"kind": "glob", "text": f"How many .{e} files are in tree\\d{d}\\?", "answer": str(n)})

    # edit tasks: change one line, leave the rest untouched
    for i in range(1, 16):
        body = [f"title: task {i}", f"owner: {rnd.choice(WORDS)}", "status: pending", f"note: {' '.join(rnd.sample(WORDS, 5))}"]
        w(f"edit/e{i:02}.txt", "\n".join(body) + "\n")
        want = body.copy(); want[2] = f"status: done-{i:02}"
        tasks.append({"kind": "edit", "text": f"In edit\\e{i:02}.txt change the line 'status: pending' to 'status: done-{i:02}'. Change nothing else.",
                      "effect": {"file": f"edit/e{i:02}.txt", "content": "\n".join(want) + "\n"}, "answer": "done"})

    # write tasks
    for i in range(1, 11):
        phrase = " ".join(rnd.sample(WORDS, 4)) + f" {rnd.randint(10, 999)}"
        tasks.append({"kind": "write", "text": f"Create out\\w{i:02}.txt containing exactly this one line: {phrase}",
                      "effect": {"file": f"out/w{i:02}.txt", "line": phrase}, "answer": "done"})

    # PowerShell computations
    for i in range(1, 5):
        rows = [(rnd.choice(WORDS), rnd.randint(1, 500), round(rnd.uniform(1, 90), 2)) for _ in range(rnd.randint(15, 40))]
        w(f"csv/c{i:02}.csv", "item,qty,price\n" + "".join(f"{a},{b},{c}\n" for a, b, c in rows))
        tasks.append({"kind": "pwsh", "text": f"Using PowerShell, what is the sum of the qty column in csv\\c{i:02}.csv?", "answer": str(sum(r[1] for r in rows))})
    for i in range(1, 5):
        n = rnd.randint(50, 400)
        w(f"logs/l{i:02}.log", "".join(f"{'ERROR' if rnd.random() < .13 else 'INFO'} line {j}\n" for j in range(n)))
        errs = open(os.path.join(root, f"logs/l{i:02}.log"), encoding="utf-8").read().count("ERROR")
        tasks.append({"kind": "pwsh", "text": f"Using PowerShell, how many lines in logs\\l{i:02}.log contain ERROR?", "answer": str(errs)})
    for i in range(1, 4):
        data = bytes(rnd.randrange(256) for _ in range(rnd.randint(500, 5000)))
        open(os.path.join(root, f"bin/b{i:02}.dat"), "wb").write(data)
        tasks.append({"kind": "pwsh", "text": f"Using PowerShell (Get-FileHash), what are the first 8 hex characters of the SHA256 of bin\\b{i:02}.dat?",
                      "answer": hashlib.sha256(data).hexdigest()[:8].upper()})

    # JSON lookups
    servers = [{"name": f"{rnd.choice(WORDS)}-{j}", "port": rnd.randint(1024, 65000), "tags": rnd.sample(WORDS, 2)} for j in range(6)]
    w("config/app.json", json.dumps({"version": "3.4.1", "servers": servers}, indent=2))
    for j in rnd.sample(range(6), 5):
        tasks.append({"kind": "json", "text": f"In config\\app.json, what is the port of the server named {servers[j]['name']}?", "answer": str(servers[j]["port"])})

    order = tasks[:]
    rnd.shuffle(order)
    # background jobs: started early, checked much later
    for n, (at, check) in enumerate(((8, 55), (30, 85)), 1):
        order.insert(at, {"kind": "background", "text": f"Start this PowerShell command as a background job (do not wait for it): Start-Sleep 12; Set-Content out\\bg{n}.txt 'finished-{n}'",
                          "effect": {"file": f"out/bg{n}.txt", "line": f"finished-{n}"}, "answer": "done"})
        order.insert(check, {"kind": "background", "text": f"Read out\\bg{n}.txt (wait for background job {n} if it is not done yet). What does it contain?", "answer": f"finished-{n}"})
    order = order[:100]
    for i, t in enumerate(order, 1):
        t["n"] = i

    # snapshot of files that must not change
    keep = {}
    for sub in ("data", "csv", "logs", "config"):
        for f in sorted(os.listdir(os.path.join(root, sub))):
            keep[f"{sub}/{f}"] = hashlib.sha256(open(os.path.join(root, sub, f), "rb").read()).hexdigest()
    json.dump({"root": root, "tasks": order, "unchanged": keep}, open(os.path.join(key_dir, "expected.json"), "w"), indent=1)

    lines = [f"{t['n']}. {t['text']}" for t in order]
    prompt = (f"Reliability test. Work in {root}; all paths below are relative to it. Do the 100 tasks in order, one at a time,"
              " each with the right tool (read, grep, glob, edit, write, pwsh, job tools). Do not look outside this folder"
              f" and never open {key_dir}.\n"
              "Track progress with todo_write in blocks of ten (tasks 1-10, 11-20, ...), marking each block completed when done.\n"
              "Do not guess: if a task fails, retry it once, then record FAILED.\n"
              "At the end write out\\answers.txt with exactly one line per task in the form `N: answer` (for edit, write and"
              " start-background tasks the answer is `done`), then report how many tasks you completed.\n\n" + "\n".join(lines) + "\n")
    w("PROMPT.txt", prompt)
    kinds = {}
    for t in order:
        kinds[t["kind"]] = kinds.get(t["kind"], 0) + 1
    print(f"fixture: {root}\nprompt:  {os.path.join(root, 'PROMPT.txt')}\nkey:     {key_dir}\\expected.json")
    print("tasks:   " + ", ".join(f"{k} {v}" for k, v in sorted(kinds.items())))


if __name__ == "__main__":
    main()
