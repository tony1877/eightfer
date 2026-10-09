# Starts the bundled agent (agent/, a fork of dsh) with its web UI, set up for shoehorn serve.
#
#   .\scripts\agent.ps1 [-AgentHome DIR] [-ApiKey KEY] [-- extra dsh web args, e.g. --no-open --trusted-host H:3080]
#
# AgentHome holds the agent's settings, sessions and memory: -AgentHome, else $env:DSH_HOME, else ~\.shoehorn\agent.
# A new one is filled from agent-home\ (settings, the `shoehorn` preset, skills); existing files are never overwritten.
# The API key is the one shoehorn serve was started with: -ApiKey, else $env:SHOEHORN_API_KEY, else "none".
param([string]$AgentHome = '', [string]$ApiKey = '', [Parameter(ValueFromRemainingArguments)][string[]]$Rest = @())
$ErrorActionPreference = 'Stop'
$root  = Split-Path -Parent $PSScriptRoot
$agent = Join-Path $root 'agent'

if (-not $AgentHome) { $AgentHome = if ($env:DSH_HOME) { $env:DSH_HOME } else { Join-Path $HOME '.shoehorn\agent' } }
$AgentHome = [IO.Path]::GetFullPath($AgentHome)
$seed = Join-Path $root 'agent-home'
foreach ($f in Get-ChildItem $seed -Recurse -File -Force) {
    $dst = Join-Path $AgentHome $f.FullName.Substring($seed.Length + 1)
    if (-not (Test-Path -LiteralPath $dst)) {
        New-Item -ItemType Directory -Force (Split-Path -Parent $dst) | Out-Null
        Copy-Item -LiteralPath $f.FullName $dst
        "seeded $dst"
    }
}
$env:DSH_HOME = $AgentHome

if (-not $ApiKey) { $ApiKey = if ($env:SHOEHORN_API_KEY) { $env:SHOEHORN_API_KEY } else { 'none' } }
$env:SHOEHORN_API_KEY = $ApiKey
# everything stays on this machine: no feedback telemetry to the upstream project
if (-not $env:DSH_TELEMETRY_MODE) { $env:DSH_TELEMETRY_MODE = 'DISABLED' }

if (-not (Get-Command pnpm -ErrorAction SilentlyContinue)) { throw 'pnpm not found: install Node 24 and run `corepack enable`' }
Push-Location $agent
try {
    if (-not (Test-Path (Join-Path $agent 'node_modules'))) { pnpm install --frozen-lockfile; if ($LASTEXITCODE) { throw 'pnpm install failed' } }
    if (-not (Test-Path (Join-Path $agent 'packages\client\ui-shoehorn\lib\index.js'))) { pnpm run build; if ($LASTEXITCODE) { throw 'agent build failed' } }
    pnpm dsh web @Rest
} finally { Pop-Location }
