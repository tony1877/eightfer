# Smoke test for `shoehorn serve`: starts it on a model, sends a chat request, a streamed one and a follow-up turn
# (prompt reuse), then stops it.  .\tests\server_smoke.ps1 -Model <gguf> [-Res <res.gguf>] [-Port 8095] [-Extra '...']
param([Parameter(Mandatory)][string]$Model, [string]$Res = '', [int]$Port = 8095, [string]$Extra = '',
      [string]$Template = 'C:\models\templates\qwen-sharp-v22.5.0.jinja', [int]$MaxTokens = 48)
$e8   = Join-Path (Split-Path -Parent $PSScriptRoot) 'build\bin\shoehorn.exe'
$args = @('serve', $Model, '--port', "$Port", '--alias', 'test', '--ctx', '4096')
if ($Res) { $args += @('--res', $Res) }
if ($Template -and (Test-Path $Template)) { $args += @('--chat-template-file', $Template) }
if ($Extra) { $args += $Extra -split ' ' }
$log = Join-Path $env:TEMP "e8-serve-$Port.log"
$p   = Start-Process -FilePath $e8 -ArgumentList $args -NoNewWindow -PassThru -RedirectStandardOutput $log -RedirectStandardError "$log.err"
try {
    $url = "http://127.0.0.1:$Port"
    $deadline = (Get-Date).AddMinutes(5)
    while ((Get-Date) -lt $deadline) { try { Invoke-RestMethod "$url/health" -TimeoutSec 2 | Out-Null; break } catch { Start-Sleep 1 } }
    (Invoke-RestMethod "$url/v1/models").data | ForEach-Object { "model: $($_.id)" }
    $msgs = @(@{ role = 'user'; content = 'Say hello in five words.' })
    $body = @{ model = 'test'; messages = $msgs; max_tokens = $MaxTokens; temperature = 0 } | ConvertTo-Json -Depth 8
    $r = Invoke-RestMethod "$url/v1/chat/completions" -Method Post -Body $body -ContentType 'application/json' -TimeoutSec 600
    "1st: finish=$($r.choices[0].finish_reason) usage=$($r.usage | ConvertTo-Json -Compress) tok/s=$([math]::Round($r.timings.predicted_per_second,1))"
    "    reasoning: $([string]$r.choices[0].message.reasoning_content | Select-Object -First 1)".Substring(0, [math]::Min(160, 15 + ([string]$r.choices[0].message.reasoning_content).Length))
    "    content:   $($r.choices[0].message.content)"
    $msgs += @{ role = 'assistant'; content = [string]$r.choices[0].message.content }
    $msgs += @{ role = 'user'; content = 'Now in three words.' }
    $body = @{ model = 'test'; messages = $msgs; max_tokens = $MaxTokens; temperature = 0; stream = $true } | ConvertTo-Json -Depth 8
    $raw = Invoke-WebRequest "$url/v1/chat/completions" -Method Post -Body $body -ContentType 'application/json' -TimeoutSec 600 -UseBasicParsing
    $lines = ($raw.Content -split "`n") | Where-Object { $_ -like 'data: *' }
    $last  = $lines | Where-Object { $_ -notmatch '\[DONE\]' } | Select-Object -Last 1
    "2nd (stream): $($lines.Count) events; last: $(($last -replace '^data: ','') | ConvertFrom-Json | ForEach-Object { 'finish=' + $_.choices[0].finish_reason + ' usage=' + ($_.usage | ConvertTo-Json -Compress) })"
    $tools = @(@{ type = 'function'; function = @{ name = 'get_weather'; description = 'Current weather for a city';
        parameters = @{ type = 'object'; properties = @{ city = @{ type = 'string' } }; required = @('city') } } })
    $body = @{ model = 'test'; messages = @(@{ role = 'user'; content = 'What is the weather in Rome? Use the tool.' });
        tools = $tools; max_tokens = 400; temperature = 0 } | ConvertTo-Json -Depth 10
    $r = Invoke-RestMethod "$url/v1/chat/completions" -Method Post -Body $body -ContentType 'application/json' -TimeoutSec 600
    "tools: finish=$($r.choices[0].finish_reason) tool_calls=$($r.choices[0].message.tool_calls | ConvertTo-Json -Compress -Depth 6)"

    # /v1/completions: plain prompt, no template; then streamed
    $body = @{ model = 'test'; prompt = 'The capital of France is'; max_tokens = 8; temperature = 0 } | ConvertTo-Json
    $r = Invoke-RestMethod "$url/v1/completions" -Method Post -Body $body -ContentType 'application/json' -TimeoutSec 600
    "completion: object=$($r.object) finish=$($r.choices[0].finish_reason) text=$($r.choices[0].text | ConvertTo-Json)"
    $body = @{ model = 'test'; prompt = 'Count: 1, 2, 3,'; max_tokens = 12; temperature = 0; stream = $true } | ConvertTo-Json
    $raw = Invoke-WebRequest "$url/v1/completions" -Method Post -Body $body -ContentType 'application/json' -TimeoutSec 600 -UseBasicParsing
    $ev  = ($raw.Content -split "`n") | Where-Object { $_ -like 'data: {*' } | ForEach-Object { ($_ -replace '^data: ', '') | ConvertFrom-Json }
    "completion (stream): $($ev.Count) events, text=$((($ev | ForEach-Object { $_.choices[0].text }) -join '') | ConvertTo-Json)"

    # penalties: a prompt that invites repetition, without and with a frequency penalty (distinct-token share)
    foreach ($fp in 0, 1.5) {
        $body = @{ model = 'test'; prompt = 'apple apple apple apple apple apple'; max_tokens = 40; temperature = 0; frequency_penalty = $fp } | ConvertTo-Json
        $r = Invoke-RestMethod "$url/v1/completions" -Method Post -Body $body -ContentType 'application/json' -TimeoutSec 600
        $w = ($r.choices[0].text -split '\s+') | Where-Object { $_ }
        "frequency_penalty=${fp}: $($w.Count) words, $(($w | Sort-Object -Unique).Count) distinct"
    }

    # two requests at once: both are served, one after the other
    $t0 = Get-Date
    $res = 1, 2 | ForEach-Object -Parallel {
        $b = @{ model = 'test'; prompt = "Request $_ says:"; max_tokens = 16; temperature = 0 } | ConvertTo-Json
        $r = Invoke-RestMethod "$using:url/v1/completions" -Method Post -Body $b -ContentType 'application/json' -TimeoutSec 600
        "$_ finish=$($r.choices[0].finish_reason) done at $([math]::Round(((Get-Date) - $using:t0).TotalSeconds, 1)) s"
    }
    "concurrent: $($res -join '; ')"

    # a streamed request whose client goes away: the next request must not wait for its 2000 tokens
    $hc = [System.Net.Http.HttpClient]::new(); $hc.Timeout = [TimeSpan]::FromSeconds(3)
    $b  = @{ model = 'test'; prompt = 'Write a very long story.'; max_tokens = 2000; temperature = 0; stream = $true } | ConvertTo-Json
    try { $hc.PostAsync("$url/v1/completions", [System.Net.Http.StringContent]::new($b, [Text.Encoding]::UTF8, 'application/json')).Result.Content.ReadAsStringAsync().Result | Out-Null } catch { }
    $hc.Dispose()
    $t0 = Get-Date
    $body = @{ model = 'test'; prompt = 'Hi'; max_tokens = 4; temperature = 0 } | ConvertTo-Json
    $r = Invoke-RestMethod "$url/v1/completions" -Method Post -Body $body -ContentType 'application/json' -TimeoutSec 600
    "after a dropped stream: next request done in $([math]::Round(((Get-Date) - $t0).TotalSeconds, 1)) s (finish=$($r.choices[0].finish_reason))"
} finally {
    Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
    Get-Content "$log.err" -ErrorAction SilentlyContinue | Select-String -Pattern 'error|failed' | Select-Object -First 5
}
