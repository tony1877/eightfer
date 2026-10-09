# Agent reliability test

100 small tasks with known answers, about one tool call each: read (30), grep (15), edit (15), write (10),
PowerShell computations (11), glob (10), JSON lookups (5) and background jobs (4).

```
python tests/agent_reliability/gen.py          # builds C:\tmp\reltest and C:\tmp\reltest-key
```

Paste the contents of `C:\tmp\reltest\PROMPT.txt` into a new session of the bundled agent (`scripts\agent.ps1`) or any
other agent client, wait for it to finish, then:

```
python tests/agent_reliability/check.py        # score, per-kind breakdown, every failure
```

`--seed N` gives a different but equally hard set; `--root` moves the fixture. The key lives outside the
fixture so the agent cannot read it, and the checker also flags fixture files the agent changed by mistake.

## Hard set

`gen_hard.py` builds 34 tasks meant to trip careless agents: multi-hop pointer chains with decoys, "last
occurrence wins" rules stated inside files, case-sensitive keys, file names with `[ ]`, `$` and double spaces
(PowerShell needs `-LiteralPath`), quoted CSV fields with commas, whole-word counts, byte-exact edits (CRLF,
duplicate lines in different sections, `$` and parentheses, UTF-8 accents, tabs, no trailing newline), an exact
JSON write, nested JSON, hidden and mixed-case files, date arithmetic, an instruction planted in a data file
that must be ignored, and a background job that fails with exit code 3.

```
python tests/agent_reliability/gen_hard.py                       # C:\tmp\hardtest
python tests/agent_reliability/check.py --root C:\tmp\hardtest
```
