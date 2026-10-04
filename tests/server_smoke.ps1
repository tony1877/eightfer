# Smoke test for `eightfer serve`: starts it on a model, sends a chat request, a streamed one and a follow-up turn
# (prompt reuse), then stops it.  .\tests\server_smoke.ps1 -Model <gguf> [-Res <res.gguf>] [-Port 8095] [-Extra '...']
param([Parameter(Mandatory)][string]$Model, [string]$Res = '', [int]$Port = 8095, [string]$Extra = '',
      [string]$Template = 'C:\models\templates\qwen-sharp-v22.5.0.jinja', [int]$MaxTokens = 48)
$e8   = Join-Path (Split-Path -Parent $PSScriptRoot) 'build\bin\eightfer.exe'
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
} finally {
    Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
    Get-Content "$log.err" -ErrorAction SilentlyContinue | Select-String -Pattern 'error|failed' | Select-Object -First 5
}
