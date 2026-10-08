# Concurrent requests on sequence slots: sends -N requests at once (each a ~15K-token document from eval.txt with a
# summary to write, greedy) to `serve --slots S`, then each again alone. Checks that every concurrent request finishes,
# that they overlap in time, and compares each output with the one it got alone (greedy: the same up to near-ties).
#   .\tests\concurrent_smoke.ps1 -Model <base.gguf> -Res <res.gguf> [-Slots 3] [-N 4] [-Extra '--kv q8_0 --kv-v q4_0']
param([Parameter(Mandatory)][string]$Model, [string]$Res = '', [int]$Port = 8098, [int]$Slots = 3, [int]$N = 4,
      [int]$Ctx = 131072, [int]$MaxTokens = 300, [string]$Extra = '--kv q8_0',
      [string]$Template = 'C:\models\templates\qwen-sharp-v22.5.0.jinja',
      [string]$Text = (Join-Path (Split-Path -Parent $PSScriptRoot) 'build-ref\eval.txt'))
$e8  = Join-Path (Split-Path -Parent $PSScriptRoot) 'build\bin\eightfer.exe'
$a   = @('serve', $Model, '--port', "$Port", '--alias', 'test', '--ctx', "$Ctx", '--slots', "$Slots", '--spec', 'auto')
if ($Res) { $a += @('--res', $Res) }
if ($Template -and (Test-Path $Template)) { $a += @('--chat-template-file', $Template) }
if ($Extra) { $a += $Extra -split ' ' | Where-Object { $_ } }
$log = Join-Path $env:TEMP "e8-conc-$Port.log"
$p   = Start-Process -FilePath $e8 -ArgumentList $a -NoNewWindow -PassThru -RedirectStandardOutput $log -RedirectStandardError "$log.err"
$url = "http://127.0.0.1:$Port"
$txt = Get-Content $Text -Raw
$bodies = for ($i = 0; $i -lt $N; $i++) {
    $doc = $txt.Substring(($i * 45000) % ($txt.Length - 50000), 50000)
    @{ model = 'test'; max_tokens = $MaxTokens; temperature = 0; chat_template_kwargs = @{ enable_thinking = $false };
       messages = @(@{ role = 'user'; content = "Document $($i + 1):`n`n$doc`n`nSummarize this document in detail." }) } | ConvertTo-Json -Depth 6
}
$fail = 0
try {
    $dl = (Get-Date).AddMinutes(5)
    while ((Get-Date) -lt $dl) { try { Invoke-RestMethod "$url/health" -TimeoutSec 2 | Out-Null; break } catch { Start-Sleep 1; if ($p.HasExited) { break } } }
    $t0   = Get-Date
    $runs = 0..($N - 1) | ForEach-Object -ThrottleLimit $N -Parallel {
        $i = $_
        $s = ((Get-Date) - $using:t0).TotalSeconds
        try {
            $r = Invoke-RestMethod "$($using:url)/v1/chat/completions" -Method Post -Body ($using:bodies)[$i] -ContentType 'application/json' -TimeoutSec 3600
            [pscustomobject]@{ i = $i; start = $s; end = ((Get-Date) - $using:t0).TotalSeconds; text = [string]$r.choices[0].message.content;
                               n = [int]$r.usage.completion_tokens; tps = [double]$r.timings.predicted_per_second; err = '' }
        } catch {
            [pscustomobject]@{ i = $i; start = $s; end = 0; text = ''; n = 0; tps = 0; err = "$_" }
        }
    } | Sort-Object i
    $runs | Where-Object { $_.err } | ForEach-Object { "request $($_.i + 1) failed: $($_.err)" }
    $wall = ((Get-Date) - $t0).TotalSeconds
    for ($i = 0; $i -lt $N; $i++) {
        "concurrent {0}: {1:n1}-{2:n1} s, {3} tokens at {4:n1} tok/s" -f ($i + 1), $runs[$i].start, $runs[$i].end, $runs[$i].n, $runs[$i].tps
        if (-not $runs[$i].n) { $fail++ }
    }
    $overlap = ($runs | Measure-Object -Property start -Maximum).Maximum -lt ($runs | Measure-Object -Property end -Minimum).Minimum
    "wall {0:n1} s, all running at once at some point: {1}" -f $wall, $overlap
    $tsum = 0
    for ($i = 0; $i -lt $N; $i++) {  # alone, one after the other (cached prompts: same slots, same states)
        $t1 = Get-Date
        $r  = Invoke-RestMethod "$url/v1/chat/completions" -Method Post -Body $bodies[$i] -ContentType 'application/json' -TimeoutSec 3600
        $tsum += ((Get-Date) - $t1).TotalSeconds
        $same = [string]$r.choices[0].message.content -eq $runs[$i].text
        if (-not $same) {
            $x = [string]$r.choices[0].message.content; $y = $runs[$i].text; $d = 0
            while ($d -lt [math]::Min($x.Length, $y.Length) -and $x[$d] -eq $y[$d]) { $d++ }
            "alone {0}: {1} tokens at {2:n1} tok/s, output differs from char {3} of {4}" -f ($i + 1), $r.usage.completion_tokens,
                $r.timings.predicted_per_second, $d, $y.Length
        } else {
            "alone {0}: {1} tokens at {2:n1} tok/s, same output" -f ($i + 1), $r.usage.completion_tokens, $r.timings.predicted_per_second
        }
    }
    "alone total {0:n1} s (concurrent wall {1:n1} s)" -f $tsum, $wall
    Get-Content "$log.err" | Select-String 'KV:|error|failed|dropped|pool' | ForEach-Object { "log: $($_.Line)" }
} catch { "request failed: $_"; $fail++ } finally {
    Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
}
if ($fail) { "FAILED: $fail"; exit 1 } else { 'PASSED' }
