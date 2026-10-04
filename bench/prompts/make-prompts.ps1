# Tokenizes the M3 acceptance prompts (Qwen chat format, thinking on) with llama.cpp's llama-tokenize.
#   .\bench\prompts\make-prompts.ps1 -Model <any qwen35 gguf> [-Tokenize <llama-tokenize.exe>]
param(
    [Parameter(Mandatory)][string]$Model,
    [string]$Tokenize = "$PSScriptRoot\..\..\build-ref\bin\llama-tokenize.exe"
)
$prompts = [ordered]@{
    code      = 'Write a Python function that parses an ISO 8601 duration string like "P3DT4H12M" into a datetime.timedelta. Handle years and months by raising ValueError, and include a few doctest examples.'
    reasoning = 'A train leaves city A at 9:00 traveling at 80 km/h toward city B, 300 km away. Another train leaves city B at 9:30 traveling at 100 km/h toward city A. At what time do they meet, and how far from city A? Explain step by step.'
    prose     = 'Write a short, vivid paragraph describing a thunderstorm rolling over a mountain village at dusk, from the point of view of an old shepherd.'
    explain   = 'Explain how a hash map handles collisions, comparing separate chaining and open addressing, and when you would choose each one.'
}
foreach ($name in $prompts.Keys) {
    $text = "<|im_start|>user`n$($prompts[$name])<|im_end|>`n<|im_start|>assistant`n<think>`n"
    $tmp  = Join-Path $env:TEMP "e8-prompt-$name.txt"
    [IO.File]::WriteAllText($tmp, $text)
    $ids = & $Tokenize -m $Model -f $tmp --ids --log-disable 2>$null
    ($ids -replace '[\[\],]', ' ').Trim() -replace '\s+', ' ' | Set-Content (Join-Path $PSScriptRoot "$name.txt")
    "{0,-10} {1} tokens" -f $name, (((Get-Content (Join-Path $PSScriptRoot "$name.txt")) -split ' ').Count)
}
