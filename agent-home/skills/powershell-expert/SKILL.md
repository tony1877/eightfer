---
name: powershell-expert
description: Use before writing any non-trivial PowerShell command, and immediately when a shell command fails, prints nothing, hangs, or shows an error you do not recognise. Covers checking the shell first (version, language mode, dsh sandbox), running native exes and reading their exit codes, quoting and escaping, paths, bounded and untruncated output, real error handling, commands that hang waiting for input, environment variables, background jobs, and the bash habits that break in pwsh.
---

# PowerShell 7 like an expert

The shell is **pwsh 7** in a persistent terminal. It works like a person
typing into a console window, so anything that waits for input waits
forever. Most failures come from one of these sections. Find the matching
one before you change anything.

## 1. Check the shell first

Run this once per session, and again if something behaves strangely:

```powershell
"$($PSVersionTable.PSVersion) | $($ExecutionContext.SessionState.LanguageMode) | $PWD"
```

- Version should be 7.x. If it is 5.1, you are in Windows PowerShell and
  several rules below are different (encoding, `&&`, stderr handling).
- Language mode should be **FullLanguage**. If it says
  **ConstrainedLanguage**, read section 2 before doing anything else.

## 2. ConstrainedLanguage means the dsh sandbox, not a bug

dsh can run this shell under a Windows sandbox (`workspace-write`). pwsh
detects the restricted token and locks itself into ConstrainedLanguage.
These errors all mean the same thing:

| error | cause |
|---|---|
| `Method invocation is supported only on core types in this language mode` | ConstrainedLanguage |
| `Cannot create type. Only core types are supported in this language mode` | ConstrainedLanguage |
| `Access to a CIM resource was not available to the client` | the sandbox blocks WMI/CIM |

What works and what does not in the sandbox:
- Cmdlets that read files, text and processes: work.
- Native executables (`pnputil.exe`, `git.exe`, `cl.exe`): work. For device
  information use `pnputil /enum-devices`, not `Get-PnpDevice`.
- CIM/WMI cmdlets (`Get-PnpDevice`, `Get-CimInstance`, `Get-NetAdapter`):
  fail.
- .NET calls (`[IO.File]::...`, `$x.Method()` on most types): may fail.
- Writes outside the workspace: fail.

**Do not try to work around it.** You cannot escalate from this shell. Tell
the user: "This shell is sandboxed (ConstrainedLanguage). To run <command>,
switch the session to Full access. dsh only applies the mode when a shell
is created, so it has to be set before this session's shell starts, or in a
new session." Then continue with what does work.

## 3. Native executables

- Call them with `&` when the path is quoted or held in a variable:
  `& 'C:\Program Files\CMake\bin\cmake.exe' --version`
- Pass arguments as separate tokens, or splat an array:
  `$a = @('/nologo', '/W4', 'main.c'); & cl.exe @a`
- **Check `$LASTEXITCODE` after every native command that matters.** In
  pwsh 7, text on stderr does NOT mean failure and does not stop anything.
  Only the exit code tells you:
  ```powershell
  & cmake --build build; if ($LASTEXITCODE -ne 0) { "BUILD FAILED: $LASTEXITCODE" }
  ```
- `$?` covers the last command of any kind. `$LASTEXITCODE` only changes
  when a native exe runs, so an old value can stay around. Read it
  straight after the call.
- `2>&1` on a native exe turns stderr lines into ErrorRecord objects. To get
  plain text: `& git status 2>&1 | ForEach-Object { "$_" }`
- Use the `.exe` name when an alias could shadow the program:
  `where.exe` (`where` is Where-Object), `sc.exe`, `curl.exe`. The `.exe`
  form is never wrong.
- Batch files and `vcvars64.bat` need cmd: `cmd /c "call x.bat && set"`.
- If an exe mangles complex arguments, `--%` passes the rest of the line
  through unchanged: `& icacls.exe --% C:\dir /grant Users:(OI)(CI)R`

## 4. Quoting and escaping

| you write | you get |
|---|---|
| `'$x literal'` | exactly `$x literal`, with nothing expanded |
| `"value $x"` | variable expanded |
| `"$obj.Name"` | **wrong**: expands `$obj`, then appends `.Name` |
| `"$($obj.Name)"` | right: any expression goes in `$( )` |
| `"$drive:\x"` | **parse error**: `$drive:` is read as a scope/drive-qualified name |
| `"${drive}:\x"` | right |
| `` "a`tb" `` | backtick is the escape char (`` `t `` tab, `` `n `` newline, ``` `` ``` literal) |

- `-match`, `-replace` and `-split` take **regular expressions**.
  `-split '.'` splits on every character. Use `-split '\.'`, or
  `-split '.', 0, 'SimpleMatch'`.
- `-replace` capture groups go in single quotes: `-replace '(\d+)', '[$1]'`.
  In double quotes, `$1` is expanded as a variable and comes out empty.
- `-like` uses wildcards (`*`, `?`), not regex.
- Here-strings: `@'` ... `'@` (literal) or `@"` ... `"@` (expanding). The
  closing `'@` / `"@` must be at the **start of its line**, with no
  indentation.

## 5. Paths

- Paths with `[` `]` are wildcards to `-Path`. Use **`-LiteralPath`** for
  any path you did not write yourself.
- Build paths with `Join-Path`, not string concatenation.
- `.NET` calls resolve relative paths against the process directory, not
  your `cd`. Give them `(Resolve-Path 'f').Path` or an absolute path.
- `Test-Path -LiteralPath $p` before destructive operations.
- Quote every path that could contain spaces. `C:\Program Files` does.

## 6. Output that is bounded and not truncated

Output lands in your context, so size it on purpose.

- Limit it: `Select-Object -First 50`, `Get-Content f -Tail 50`,
  `Get-Content f -TotalCount 50`.
- `Format-Table` cuts columns with `...` to fit the width. For full values:
  `... | Format-Table -AutoSize | Out-String -Width 250`, or `Format-List`.
- **`Format-*` is only for display.** Never pipe Format-Table output into
  another command. Filter and select first, format last.
- `ConvertTo-Json` stops at depth 2 by default and prints a truncation
  warning. Pass `-Depth 10`.
- Count before dumping: `(Get-ChildItem -Recurse -File).Count`.
- Search text with Select-String, not by reading whole files:
  `Select-String -Path .\src\*.cs -Pattern 'Foo' -SimpleMatch -Context 2 | Select-Object -First 20`

## 7. Errors that are actually caught

- Most cmdlet errors are **non-terminating**. `try/catch` does not see them
  unless you add `-ErrorAction Stop`:
  ```powershell
  try { Get-Item -LiteralPath $p -ErrorAction Stop } catch { "ERR: $($_.Exception.Message)" }
  ```
- In a multi-step script, put `$ErrorActionPreference = 'Stop'` at the top
  so the first failure stops it.
- Native exes never throw. Check `$LASTEXITCODE` (section 3).
- To see the full last error: `$Error[0] | Format-List * -Force`.
- `-ErrorAction SilentlyContinue` hides the error. Use it only when failure
  is expected and harmless, never to make red text go away.

## 8. Nothing may wait for input

The terminal is interactive, so a prompt hangs the tool call until it
times out. Prevent every prompt:

| hangs | use instead |
|---|---|
| `Remove-Item dir` on a non-empty folder | `Remove-Item -LiteralPath dir -Recurse -Force` |
| cmdlets with a confirm step | add `-Confirm:$false` (and `-Force` where needed) |
| `git log`, `git diff`, `git show` (pager) | `git --no-pager log -n 20` |
| `Read-Host`, `Get-Credential`, `pause` | never use them. Ask the user in chat |
| installers, `winget` without flags | ask the user to run them |
| `more`, `less`, `Out-Host -Paging` | `Select-Object -First N` |

If a command could prompt and you are not sure, it will. Add the flags.

## 9. Environment variables

- `$env:NAME` is **this process only**. `$env:X = '1'` affects this shell
  and whatever it starts, nothing else.
- Persistent values live in the registry:
  `[Environment]::GetEnvironmentVariable('PATH', 'User')` (or `'Machine'`).
  Changes there do NOT reach an already-running shell.
- To pick up PATH changes made after this shell started:
  ```powershell
  $env:PATH = [Environment]::GetEnvironmentVariable('PATH','Machine') + ';' + [Environment]::GetEnvironmentVariable('PATH','User')
  ```
- Find a tool with `Get-Command cl.exe -ErrorAction SilentlyContinue`. If
  it returns nothing, the tool is not on PATH. That does not mean it is
  not installed (see static-analysis: `C:\Tools\bin` by full path).

## 10. Long-running work

- `Start-Job { ... }` runs in a **separate process**: variables from your
  session are not visible (use `$using:var`), and relative paths are
  unreliable. Use absolute paths inside the block.
- `Start-ThreadJob` is faster to start and also needs `$using:`.
- For a build or an import, write output to a file and read its tail:
  ```powershell
  $p = Start-Process pwsh -ArgumentList '-NoProfile','-Command','cmake --build C:\src\build *> C:\temp\build.log' -PassThru -WindowStyle Hidden
  Get-Content C:\temp\build.log -Tail 30
  $p.HasExited; $p.ExitCode
  ```
- `Start-Process -Wait -PassThru` gives you `.ExitCode`. Without
  `-PassThru` you have no exit code at all.

## 11. Arrays and loops

- `$a += $x` inside a loop copies the whole array every time. Collect
  loop output instead:
  ```powershell
  $rows = foreach ($f in Get-ChildItem -File) { [pscustomobject]@{ Name = $f.Name; KB = [int]($f.Length / 1KB) } }
  ```
- A command returning one item gives a scalar, not an array. Wrap it in
  `@(...)` before using `.Count` or indexing: `@(Get-ChildItem *.log).Count`
- `Get-ChildItem -Filter '*.cs'` is much faster than `-Include`. Add `-File`,
  and `-Depth N` with `-Recurse` on big trees.
- Compare with `-eq`, `-ne`, `-gt`, `-lt`. `>` is **redirection**:
  `if ($n > 5)` writes a file named `5`.
- Put `$null` on the left: `$null -eq $x` (on the right, an array compares
  element by element).

## 12. Bash habits that break here

| bash | pwsh 7 |
|---|---|
| `export X=1` / `X=1 cmd` | `$env:X = '1'; cmd` |
| `2>/dev/null` | `2>$null` |
| `ls -la` | `Get-ChildItem -Force` |
| `grep -r foo .` | `Get-ChildItem -Recurse -File \| Select-String -Pattern foo` |
| `which x` | `Get-Command x` |
| `touch f` | `New-Item -ItemType File -Path f` |
| `rm -rf d` | `Remove-Item -LiteralPath d -Recurse -Force` |
| `head -n 20` / `tail -n 20` | `Select-Object -First 20` / `-Last 20` |
| `` `cmd` `` substitution | `$(cmd)` (backtick is the escape character) |
| `a && b` / `a \|\| b` | same, works in pwsh 7 |
| `if [ -f x ]` | `if (Test-Path -LiteralPath x)` |
| `for f in *; do` | `foreach ($f in Get-ChildItem) { }` |

## Before you say a command failed

1. Did you read `$LASTEXITCODE` (native) or the error text (cmdlet)?
2. Is the language mode still FullLanguage? (section 1)
3. Did something prompt and hang? (section 8)
4. Is the output cut off by `Format-Table` or `-Depth`? (section 6)
5. Is a path or a `$` being interpreted where you meant it literally?
   (sections 4, 5)

Once you have the real cause, fix the command. Do not wrap it in retries or
write around it.
