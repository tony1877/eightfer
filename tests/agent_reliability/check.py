"""Grades an agent reliability run (see gen.py).

    python tests/agent_reliability/check.py [--root C:\\tmp\\reltest]

Reads <root>\\out\\answers.txt and the files the tasks change, compares them with <root>-key\\expected.json, and
prints the score, a per-kind breakdown and every failure. Also flags files the agent should not have touched.
"""
import argparse, hashlib, json, os, re


def norm(s):
    s = s.strip().strip("`'\"").strip()
    s = re.sub(r"\s+", " ", s)
    return s.lower().rstrip(".")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"C:\tmp\reltest")
    a = ap.parse_args()
    key = json.load(open(a.root.rstrip("\\/") + "-key/expected.json"))
    root = key["root"]
    answers = {}
    path = os.path.join(root, "out", "answers.txt")
    if os.path.exists(path):
        for line in open(path, encoding="utf-8", errors="replace"):
            m = re.match(r"\s*(\d+)\s*[:.)-]\s*(.*)", line)
            if m: answers[int(m.group(1))] = m.group(2)
    else:
        print(f"no answers file at {path}: only file effects are graded")

    def read(rel):
        p = os.path.join(root, rel)
        return open(p, encoding="utf-8", errors="replace").read().replace("\r\n", "\n") if os.path.exists(p) else None

    ok, fails, kinds = 0, [], {}
    for t in key["tasks"]:
        good, why = True, ""
        got = answers.get(t["n"])
        eff = t.get("effect")
        if eff:
            body = read(eff["file"])
            if body is None:
                good, why = False, f"{eff['file']} missing"
            elif "content" in eff and body != eff["content"]:
                good, why = False, f"{eff['file']} content differs"
            elif "line" in eff and body.strip() != eff["line"]:
                good, why = False, f"{eff['file']} = {body.strip()[:60]!r}"
        else:
            want = t["answer"]
            if got is None:
                good, why = False, "no answer"
            elif norm(got) != norm(want) and not (norm(want) in norm(got) and len(norm(got)) <= len(norm(want)) + 12):
                good, why = False, f"answered {got.strip()[:60]!r}, expected {want!r}"
        k = kinds.setdefault(t["kind"], [0, 0])
        k[1] += 1
        if good:
            ok += 1; k[0] += 1
        else:
            fails.append((t["n"], t["kind"], why, t["text"][:90]))
    touched = [f for f, h in key["unchanged"].items()
               if not os.path.exists(os.path.join(root, f)) or hashlib.sha256(open(os.path.join(root, f), "rb").read()).hexdigest() != h]

    print(f"\nSCORE  {ok}/{len(key['tasks'])}   ({len(answers)} answers in answers.txt)")
    for kname, (g, n) in sorted(kinds.items()):
        print(f"  {kname:<11} {g}/{n}")
    if touched:
        print(f"\nFiles changed that should not have been ({len(touched)}): " + ", ".join(touched))
    if fails:
        print("\nFailures:")
        for n, kname, why, text in fails:
            print(f"  #{n:<3} {kname:<10} {why}\n       {text}")


if __name__ == "__main__":
    main()
