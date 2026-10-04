# Small-batch Q4_K residual kernel vs ggml's mul_mat (E8_NO_SMALL_GEMM=1), tiny model on the CPU.
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$t    = Join-Path $root 'tests\tiny\out'
$e8   = Join-Path $root 'build\bin\eightfer.exe'
$py   = Join-Path $root '.venv\Scripts\python.exe'
foreach ($b in 1, 5, 9, 16) {
    $env:E8_NO_SMALL_GEMM = '1'
    & $e8 logits "$t\tiny.base.gguf" --res "$t\tiny.res.gguf" --tokens "$t\tokens.txt" --out "$t\ggml-b$b.f32" --batch $b | Out-Null
    $env:E8_NO_SMALL_GEMM = $null
    & $e8 logits "$t\tiny.base.gguf" --res "$t\tiny.res.gguf" --tokens "$t\tokens.txt" --out "$t\small-b$b.f32" --batch $b | Out-Null
    & $py -c "import numpy as np, sys; a=np.fromfile(sys.argv[1],np.float32); b=np.fromfile(sys.argv[2],np.float32); print('batch %s: rel.err %.2e max %.2e' % (sys.argv[3], np.linalg.norm(a-b)/np.linalg.norm(a), np.abs(a-b).max()))" "$t\ggml-b$b.f32" "$t\small-b$b.f32" $b
}
