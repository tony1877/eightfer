<#
.SYNOPSIS
  Builds shoehorn on Windows: Visual Studio 2022 (MSVC) + CUDA + CMake/Ninja, Release.

.EXAMPLE
  .\scripts\build.ps1            # build with CUDA for RTX 50xx (sm_120)
  .\scripts\build.ps1 -Clean     # wipe .\build first
  .\scripts\build.ps1 -NoCuda    # CPU-only build
  .\scripts\build.ps1 -Portable  # no AVX-512 extensions (CPUs older than Zen 4 / Ice Lake)
#>
param(
    [switch]$Clean,
    [switch]$NoCuda,
    [switch]$Portable,
    [string]$CudaArch = '120a-real'  # Blackwell consumer GPUs (RTX 50xx)
)

$ErrorActionPreference = 'Stop'
$root  = Split-Path -Parent $PSScriptRoot
$build = Join-Path $root 'build'

function Fail([string]$msg) {
    Write-Host "ERROR: $msg" -ForegroundColor Red
    exit 1
}

$installVs = 'winget install --id Microsoft.VisualStudio.2022.BuildTools --override "--quiet --wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"'

# 1. Visual Studio 2022 with the x64 C++ toolset. CUDA 13.0 supports VS 2019/2022 as host compiler, not VS 2026.
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
if (-not (Test-Path $vswhere)) {
    Fail "Visual Studio Installer not found. Install the C++ Build Tools:`n  $installVs"
}
$vs = & $vswhere -products * -version '[17.0,18.0)' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -latest -property installationPath
if (-not $vs) {
    Fail "No Visual Studio 2022 with the C++ x64 toolset found. Install it:`n  $installVs"
}
$devcmd = Join-Path $vs 'Common7\Tools\VsDevCmd.bat'

# 2. Load the MSVC x64 environment into this session (via a temp .cmd to avoid cmd.exe quoting issues).
$tmp = Join-Path $env:TEMP ("shoehorn_vsenv_{0}.cmd" -f [guid]::NewGuid())
Set-Content -Path $tmp -Encoding ASCII -Value "@call `"$devcmd`" -arch=x64 -host_arch=x64 -no_logo`r`n@set"
try {
    $envDump = & cmd.exe /d /c $tmp
} finally {
    Remove-Item -Force $tmp -ErrorAction SilentlyContinue
}
foreach ($line in $envDump) {
    if ($line -match '^([^=]+)=(.*)$') {
        [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], 'Process')
    }
}

# 3. Tools.
foreach ($tool in 'cl', 'cmake', 'ninja', 'git') {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
        $hint = switch ($tool) {
            'cmake' { 'Add the "C++ CMake tools for Windows" component, or: winget install Kitware.CMake' }
            'ninja' { 'Add the "C++ CMake tools for Windows" component, or: winget install Ninja-build.Ninja' }
            'git'   { 'winget install Git.Git' }
            default { $installVs }
        }
        Fail "$tool not found after loading the VS environment. $hint"
    }
}
if (-not $NoCuda -and -not (Get-Command nvcc -ErrorAction SilentlyContinue)) {
    Fail 'nvcc not found. Install CUDA Toolkit 12.8 or newer (sm_120 support), or build with -NoCuda.'
}

# 4. ggml submodule.
if (-not (Test-Path (Join-Path $root 'third_party\llama.cpp\ggml\CMakeLists.txt'))) {
    git -C $root submodule update --init --depth 1
    if ($LASTEXITCODE -ne 0) { Fail 'git submodule update failed' }
}

# 4b. shoehorn's ggml patches (patches\ggml-*.patch), applied once: `git apply --check` fails when already applied.
foreach ($p in Get-ChildItem (Join-Path $root 'patches') -Filter 'ggml-*.patch' -ErrorAction SilentlyContinue | Sort-Object Name) {
    git -C (Join-Path $root 'third_party\llama.cpp') apply --check $p.FullName 2>$null
    if ($LASTEXITCODE -eq 0) {
        git -C (Join-Path $root 'third_party\llama.cpp') apply $p.FullName
        if ($LASTEXITCODE -ne 0) { Fail "could not apply $($p.Name)" }
        Write-Host "applied $($p.Name)"
    }
}

# 5. Configure and build. The first CUDA build compiles ggml's kernels and takes several minutes.
if ($Clean -and (Test-Path $build)) {
    Remove-Item -Recurse -Force $build
}
$cuda = if ($NoCuda) { 'OFF' } else { 'ON' }
# MSVC has no -march=native: ggml's MSVC detection enables AVX-512F but never VNNI, BF16 or VBMI, and MSVC
# defines no macros for them. Zen 4/5 (the target 9800X3D) and Ice Lake+ have all three, so enable them
# explicitly. The resulting binary needs such a CPU; -Portable leaves them off.
$on  = if ($Portable) { 'OFF' } else { 'ON' }
$isa = @("-DGGML_AVX512_VNNI=$on", "-DGGML_AVX512_BF16=$on", "-DGGML_AVX512_VBMI=$on")
if (-not $Portable) { $isa += '-DGGML_AVX512=ON' }  # -Portable leaves AVX-512F to ggml's own detection
# scripts\watchdog.ps1 leaves the server alone while this file exists (the build replaces shoehorn.exe)
$pause = Join-Path $HOME '.shoehorn\watchdog.pause'
New-Item -ItemType Directory -Force (Split-Path $pause) | Out-Null
Set-Content -LiteralPath $pause 'build.ps1'
try {
    cmake -S $root -B $build -G Ninja -DCMAKE_BUILD_TYPE=Release "-DSHOEHORN_CUDA=$cuda" "-DCMAKE_CUDA_ARCHITECTURES=$CudaArch" @isa
    if ($LASTEXITCODE -ne 0) { Fail 'cmake configure failed (see above)' }
    cmake --build $build --config Release
    if ($LASTEXITCODE -ne 0) { Fail 'build failed (see above)' }
} finally {
    Remove-Item -LiteralPath $pause -ErrorAction SilentlyContinue
}

Write-Host "Built: $(Join-Path $build 'bin\shoehorn.exe')" -ForegroundColor Green
