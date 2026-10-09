# Agent reliability test

100 small tasks with known answers, about one tool call each: read (30), grep (15), edit (15), write (10),
PowerShell computations (11), glob (10), JSON lookups (5) and background jobs (4).

```
python tests/agent_reliability/gen.py          # builds C:\tmp\reltest and C:\tmp\reltest-key
```

Paste the contents of `C:\tmp\reltest\PROMPT.txt` into a new dsh session, wait for it to finish, then:

```
python tests/agent_reliability/check.py        # score, per-kind breakdown, every failure
```

`--seed N` gives a different but equally hard set; `--root` moves the fixture. The key lives outside the
fixture so the agent cannot read it, and the checker also flags fixture files the agent changed by mistake.
