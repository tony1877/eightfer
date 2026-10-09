---
name: project-memory
description: Use at the start of any multi-session task, and before concluding that work is unfinished or that you must re-investigate something. Explains the on-disk state-file pattern that carries progress across sessions and context compaction, so findings are recorded once and trusted rather than re-derived.
---

# Working across sessions

Your context does not survive between sessions, and compaction can drop
earlier tool results mid-session. Anything you must not lose belongs on
disk, not in your head.

The failure this prevents: re-reading files you already read, re-checking
whether a write succeeded, and re-deriving conclusions you already reached
- because the record of them was silently dropped from context.

## The pattern

1. **One state file per task**, in `docs/`. It is both the work queue and
   the memory.
2. **Read it first, every session.** It tells you what is done and what
   remains.
3. **Append before you act**, not after. Write `### WORKING: <item>` and
   your intent *before* making the change, so an interrupted session leaves
   a trace.
4. **Mark a terminal state immediately** when an item finishes: DONE with
   the files changed, or SKIPPED with an honest reason.
5. **Trust the file over your memory.** If it says an item is DONE, it is
   DONE - do not re-verify.

## Rules

- Never re-audit work the state file records as complete.
- Never re-open a file to double-check something already written down.
- If you are unsure whether a write landed, read the state file once and
  continue from what it says. Do not re-run earlier work.
- SKIPPED with a clear reason is a good outcome. It hands a decision back
  to the user instead of guessing.
- Never reason about what a future session will do. Record the current
  state accurately and stop.

## Durable project knowledge

Facts that outlive one task - build commands, conventions, environment
quirks, decisions already made - belong in the repository's agent notes
file, not in a task state file and not in your head. Update it when you
learn something durable, so the next session starts knowing it.
