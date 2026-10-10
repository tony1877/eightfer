# Keeps shoehorn (and anything else the start script starts) running: checks that every port in -Ports is listening,
# and when one has been down for -Grace seconds runs the -Start script, then waits a minute before looking again.
#
#   .\scripts\watchdog.ps1 -Start C:\path\to\start.ps1 [-Ports 8090,3080] [-Grace 30] [-Interval 5]
#
# It stays out of the way of deliberate stops: nothing happens while ~\.shoehorn\watchdog.pause exists (scripts\build.ps1
# creates it for the length of a build; a pause file older than an hour is ignored), and the grace period covers a
# restart from the Settings page (a few seconds). One watchdog per user (a named mutex). Log: ~\.shoehorn\logs\watchdog.log.
param([Parameter(Mandatory)][string]$Start, [string[]]$Ports = @('8090'), [int]$Grace = 30, [int]$Interval = 5)
$ErrorActionPreference = 'Continue'
# ports as text, split here: through -File, "8090,3080" would otherwise turn into the number 80903080
$Ports = @($Ports -split '[,\s]+' | Where-Object { $_ } | ForEach-Object { [int]$_ })
$home_ = Join-Path $HOME '.shoehorn'
$pause = Join-Path $home_ 'watchdog.pause'
$logDir = Join-Path $home_ 'logs'
New-Item -ItemType Directory -Force $logDir | Out-Null
$log = Join-Path $logDir 'watchdog.log'
function Say([string]$m) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $m" | Add-Content -LiteralPath $log }

$mutex = New-Object System.Threading.Mutex($false, 'Local\shoehorn-watchdog')
if (-not $mutex.WaitOne(0)) { Say 'another watchdog is running; exiting'; exit 0 }
Say "watching ports $($Ports -join ', '); start script $Start"

$downSince = $null
while ($true) {
    $down = @($Ports | Where-Object { -not (Get-NetTCPConnection -LocalPort $_ -State Listen -ErrorAction SilentlyContinue) })
    $paused = (Test-Path -LiteralPath $pause) -and ((Get-Date) - (Get-Item -LiteralPath $pause).LastWriteTime).TotalHours -lt 1
    if (-not $down.Count -or $paused) {
        $downSince = $null
    } elseif (-not $downSince) {
        $downSince = Get-Date
    } elseif (((Get-Date) - $downSince).TotalSeconds -ge $Grace) {
        Say "port(s) $($down -join ', ') down for $Grace s: running $Start"
        try {
            $out = & pwsh -NoProfile -ExecutionPolicy Bypass -File $Start 2>&1 | Select-Object -Last 3
            Say ("start script: " + (($out | ForEach-Object { "$_".Trim() }) -join ' / '))
        } catch { Say "start script failed: $_" }
        $downSince = $null
        Start-Sleep 60
        continue
    }
    Start-Sleep $Interval
}
