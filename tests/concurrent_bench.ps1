# Decode throughput with concurrent requests on sequence slots. Each of -N conversations (a ~12K-token document from
# eval.txt and a summary request) gets its prompt cached first (max_tokens 1), so the measured part is decoding: the
# requests one after another, then all at once. Prints aggregate tokens per second for both.
#   .\tests\concurrent_bench.ps1 -Model <base.gguf> -Res <res.gguf> [-N 3] [-Slots 3] [-Temp 1.0] [-Extra '...']
param([Parameter(Mandatory)][string]$Model, [string]$Res = '', [int]$Port = 8099, [int]$N = 3, [int]$Slots = 0,
      [int]$Ctx = 131072, [int]$MaxTokens = 500, [double]$Temp = 1.0, [string]$Extra = '--kv q8_0 --kv-v q4_0',
      [string]$Template = 'C:\models\templates\qwen-sharp-v22.5.0.jinja',
      [string]$Text = (Join-Path (Split-Path -Parent $PSScriptRoot) 'build-ref\eval.txt'))
if ($Slots -le 0) { $Slots = $N }
$e8 = Join-Path (Split-Path -Parent $PSScriptRoot) 'build\bin\shoehorn.exe'
$a  = @('serve', $Model, '--port', "$Port", '--alias', 'test', '--ctx', "$Ctx", '--slots', "$Slots", '--spec', 'auto')
if ($Res) { $a += @('--res', $Res) }
if ($Template -and (Test-Path $Template)) { $a += @('--chat-template-file', $Template) }
if ($Extra) { $a += $Extra -split ' ' | Where-Object { $_ } }
$log = Join-Path $env:TEMP "e8-bench-$Port.log"
$p   = Start-Process -FilePath $e8 -ArgumentList $a -NoNewWindow -PassThru -RedirectStandardOutput $log -RedirectStandardError "$log.err"
$url = "http://127.0.0.1:$Port"
$txt = Get-Content $Text -Raw
function Body([int] $i, [int] $max, [int] $seed) {
    $doc = $txt.Substring(($i * 37000) % ($txt.Length - 45000), 45000)
    @{ model = 'test'; max_tokens = $max; temperature = $Temp; seed = $seed; chat_template_kwargs = @{ enable_thinking = $false };
       messages = @(@{ role = 'user'; content = "Document $($i + 1):`n`n$doc`n`nSummarize this document in detail, section by section." }) } |
        ConvertTo-Json -Depth 6
}
try {
    $dl = (Get-Date).AddMinutes(5)
    while ((Get-Date) -lt $dl) { try { Invoke-RestMethod "$url/health" -TimeoutSec 2 | Out-Null; break } catch { Start-Sleep 1; if ($p.HasExited) { break } } }
    for ($i = 0; $i -lt $N; $i++) {  # cache each prompt in its slot
        Invoke-RestMethod "$url/v1/chat/completions" -Method Post -Body (Body $i 1 1) -ContentType 'application/json' -TimeoutSec 3600 | Out-Null
    }
    $tok = 0; $t0 = Get-Date
    for ($i = 0; $i -lt $N; $i++) {
        $r = Invoke-RestMethod "$url/v1/chat/completions" -Method Post -Body (Body $i $MaxTokens (100 + $i)) -ContentType 'application/json' -TimeoutSec 3600
        $tok += $r.usage.completion_tokens
    }
    $ts = ((Get-Date) - $t0).TotalSeconds
    "one after another: {0} tokens in {1:n1} s = {2:n1} tok/s" -f $tok, $ts, ($tok / $ts)
    $bodies = for ($i = 0; $i -lt $N; $i++) { Body $i $MaxTokens (200 + $i) }
    $t0     = Get-Date
    $runs   = 0..($N - 1) | ForEach-Object -ThrottleLimit $N -Parallel {
        $r = Invoke-RestMethod "$($using:url)/v1/chat/completions" -Method Post -Body ($using:bodies)[$_] -ContentType 'application/json' -TimeoutSec 3600
        [int]$r.usage.completion_tokens
    }
    $tc = ((Get-Date) - $t0).TotalSeconds
    $tk = ($runs | Measure-Object -Sum).Sum
    "all at once:       {0} tokens in {1:n1} s = {2:n1} tok/s ({3:n2}x)" -f $tk, $tc, ($tk / $tc), (($tk / $tc) / ($tok / $ts))
} catch { "request failed: $_" } finally {
    Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
}
Get-Content "$log.err" | Select-String 'KV:|error|CUDA' | ForEach-Object { "log: $($_.Line)" }
