# Sequence slots (`serve --slots N`): three conversations, each a ~15K-token document with its own secret code, are
# interleaved (A, B, C, then a follow-up turn on each). Every follow-up must land on its conversation's slot, reuse its
# prompt (cached_tokens) and still recall the code from the start of its document. With -Fresh, each follow-up is
# also sent to a single-slot server with nothing cached, for comparison.
#   .\tests\slots_smoke.ps1 -Model <base.gguf> -Res <res.gguf> [-Port 8097] [-Extra '--kv q8_0 --kv-v q4_0'] [-Fresh]
param([Parameter(Mandatory)][string]$Model, [string]$Res = '', [int]$Port = 8097, [string]$Extra = '--kv q8_0',
      [int]$Ctx = 65536, [int]$Slots = 3, [switch]$Fresh,
      [string]$Template = 'C:\models\templates\qwen-sharp-v22.5.0.jinja',
      [string]$Text = (Join-Path (Split-Path -Parent $PSScriptRoot) 'build-ref\eval.txt'))
$e8 = Join-Path (Split-Path -Parent $PSScriptRoot) 'build\bin\eightfer.exe'

function Start-Server([int] $port, [int] $slots) {
    $a = @('serve', $Model, '--port', "$port", '--alias', 'test', '--ctx', "$Ctx", '--slots', "$slots", '--spec', 'auto')
    if ($Res) { $a += @('--res', $Res) }
    if ($Template -and (Test-Path $Template)) { $a += @('--chat-template-file', $Template) }
    if ($Extra) { $a += $Extra -split ' ' | Where-Object { $_ } }
    $log = Join-Path $env:TEMP "e8-slots-$port.log"
    $p   = Start-Process -FilePath $e8 -ArgumentList $a -NoNewWindow -PassThru -RedirectStandardOutput $log -RedirectStandardError "$log.err"
    $deadline = (Get-Date).AddMinutes(5)
    while ((Get-Date) -lt $deadline) {
        try { Invoke-RestMethod "http://127.0.0.1:$port/health" -TimeoutSec 2 | Out-Null; break } catch { Start-Sleep 1; if ($p.HasExited) { break } }
    }
    return @{ p = $p; log = "$log.err"; url = "http://127.0.0.1:$port" }
}
function Chat($srv, $msgs, [int] $max = 40) {
    $body = @{ model = 'test'; messages = $msgs; max_tokens = $max; temperature = 0;
               chat_template_kwargs = @{ enable_thinking = $false } } | ConvertTo-Json -Depth 8
    $t0 = Get-Date
    $r  = Invoke-RestMethod "$($srv.url)/v1/chat/completions" -Method Post -Body $body -ContentType 'application/json' -TimeoutSec 3600
    return @{ text = [string]$r.choices[0].message.content; prompt = $r.usage.prompt_tokens;
              cached = $r.usage.prompt_tokens_details.cached_tokens; s = ((Get-Date) - $t0).TotalSeconds }
}

$txt   = Get-Content $Text -Raw
$convs = @(
    @{ name = 'A'; code = '7342-ALPHA'; doc = $txt.Substring(0, 60000) },
    @{ name = 'B'; code = '5519-BRAVO'; doc = $txt.Substring(60000, 60000) },
    @{ name = 'C'; code = '8803-CHARLIE'; doc = $txt.Substring(120000, 60000) }
)
foreach ($c in $convs) {
    $c.msgs = @(@{ role = 'user'; content = "Remember this: the secret code is $($c.code).`n`n$($c.doc)`n`nIn one sentence, what is this text about?" })
}
$fail = 0
$srv  = Start-Server $Port $Slots
try {
    foreach ($c in $convs) {  # first turns: one slot each
        $r = Chat $srv $c.msgs 60
        "{0} turn 1: prompt {1}, cached {2}, {3:n1} s: {4}" -f $c.name, $r.prompt, $r.cached, $r.s, $r.text.Trim()
        $c.msgs += @{ role = 'assistant'; content = $r.text }
        $c.msgs += @{ role = 'user'; content = 'What is the secret code stated at the very beginning? Reply with the code only.' }
    }
    foreach ($c in $convs) {  # follow-ups, each after the other two conversations ran
        $r  = Chat $srv $c.msgs 20
        $ok = $r.text -match [regex]::Escape($c.code)
        $hit = $r.cached -ge ($r.prompt - 200)
        "{0} turn 2: prompt {1}, cached {2}, {3:n1} s: {4}  [code {5}, prompt reuse {6}]" -f $c.name, $r.prompt, $r.cached,
            $r.s, $r.text.Trim(), ($(if ($ok) { 'ok' } else { 'WRONG' })), ($(if ($hit) { 'ok' } else { 'MISSED' }))
        if (-not $ok -or -not $hit) { $fail++ }
        $c.answer = $r.text
    }
    Get-Content $srv.log | Select-String 'KV:|slot|pool|dropped|error' | ForEach-Object { "log: $($_.Line)" }
} catch { "request failed: $_"; $fail++ } finally {
    Stop-Process -Id $srv.p.Id -Force -ErrorAction SilentlyContinue
}
if ($Fresh) {
    Start-Sleep 5
    $srv = Start-Server ($Port + 1) 1
    try {
        foreach ($c in $convs) {
            $r = Chat $srv $c.msgs 20
            "{0} fresh: prompt {1}, cached {2}: {3}  [{4}]" -f $c.name, $r.prompt, $r.cached, $r.text.Trim(),
                ($(if ($r.text.Trim() -eq $c.answer.Trim()) { 'same as the slot run' } else { 'differs' }))
        }
    } finally {
        Stop-Process -Id $srv.p.Id -Force -ErrorAction SilentlyContinue
    }
}
if ($fail) { "FAILED: $fail"; exit 1 } else { 'PASSED' }
