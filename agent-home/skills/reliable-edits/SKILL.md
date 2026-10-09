---
name: reliable-edits
description: Use before creating or editing files, and right after any failed edit. New files and full rewrites are ONE `write` call with the complete content; changes are `edit` calls whose old_string is copied line for line from a fresh `read`; each failure has one fixed response. Verify by running the code, not by re-reading files.
---

# Editing files reliably

## The tools

| tool | params | notes |
|---|---|---|
| `read` | `file_path`, optional `offset` (1-based), `limit` | numbered lines; the text after the number is the exact line |
| `write` | `file_path`, `content` | creates or **overwrites** a file with the complete content |
| `edit` | `file_path`, `old_string`, `new_string`, optional `replace_all` | `old_string` must occur exactly once unless `replace_all` |
| `glob` / `grep` | patterns | find files / lines without reading whole files |

`edit` matches across CRLF and LF and keeps the file's own line endings: line endings are never the
cause of a failed edit.

## New file or full rewrite -> one `write`

Pass the complete file, first line to last, in a single `write`. Do not create an empty file and
fill it in pieces. A file too big to write in one go (over ~400 lines) is a sign to split the design
into smaller modules, each written whole.

Use `write` for an existing file too when more than about half of it changes: one rewrite beats
dozens of small edits.

## Change in an existing file -> `read`, then `edit` a copied block

1. `read` the region you will change (`offset`/`limit` around it), unless you read it since your
   last edit to that file. Use `grep` to find the line number first in a big file.
2. `old_string` = the lines you change, **copied exactly from that read**: whole lines, indentation
   included, no line numbers. Add an unchanged neighbouring line if needed to make it unique.
   Several lines are normal.
3. `new_string` = the replacement for exactly those lines.
4. One `edit` per separate region. Renaming a symbol everywhere: `replace_all: true`.

## When an edit fails: one fixed response each

| failure | do this |
|---|---|
| not found | `read` the region again and copy the lines fresh: they changed, or the copy was off |
| not unique / ambiguous | add one neighbouring unique line to `old_string` (or `replace_all` if every match should change) |
| not observed / stale | `read` the file, then redo the edit from what you see |

Never resend a failed `old_string` unchanged. Never count spaces or characters by hand: copy them.
Do not write file content through the shell.

## Verify by running, not by reading

After editing, run the program or the tests. A failing run tells you what is actually wrong;
re-reading a file you just wrote mostly wastes a turn. Read again only to make the next edit.
