# eightfer

An inference engine for **Qwen3.8-27B** and **Qwen3.8-Flash-Next** on one consumer PC:
Windows 11, an RTX 5080 16 GB, a Ryzen 7 9800X3D and 32 GB DDR5, plus NVMe/SATA SSDs.

The idea is **8-bit quality at 4-bit residency**. Each weight is stored as a 4-bit base and a 4-bit residual:

- The base lives in VRAM. It drafts tokens at full GPU speed.
- The residual lives in system RAM or on NVMe. Base + residual verify the drafts.

Speculative sampling makes the result lossless relative to the ≈Q8 model.
For the 176B Flash-Next, experts stream from NVMe through a VRAM/RAM heat cache.
Its 51B n-gram table is read straight from disk at full BF16 precision.

- [`docs/DESIGN.md`](docs/DESIGN.md) — design, memory budgets, estimates, milestones.
- [`docs/SPEED2X.md`](docs/SPEED2X.md) — plan for ~2x decode speed (lossless) and 200k context on both models.
- [`docs/BITLEVEL.md`](docs/BITLEVEL.md) — bit-level lossless mode: exact BF16 rebuilt from the 4-bit draft's own bits.
- [`experiments/exact_tail`](experiments/exact_tail) — the bit-exact rebuild (0 mismatches over 204.5M real weights).
- [`experiments/nested_quant`](experiments/nested_quant) — first measurement on real Qwen3.8-27B weights.
  An IQ4_XS base plus a Q4_K residual has 0.89× the weight error of Q8_0.

Status: **M1 done**. `eightfer bench` measures the numbers the design depends on; the target box's results are in
[`bench/results/2026-10-04-rtx5080-9800x3d`](bench/results/2026-10-04-rtx5080-9800x3d/README.md). It doesn't run
models yet (M2 next).

## Build (Windows)

Needs Git, the CUDA Toolkit ≥ 12.8, and **Visual Studio 2022** Build Tools with the C++ workload.
CUDA 13.0 does not accept VS 2026 as host compiler. Install the Build Tools if `build.ps1` says they're missing:

```powershell
winget install --id Microsoft.VisualStudio.2022.BuildTools --override "--quiet --wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
```

Clone to a short path and build. The first build compiles ggml's CUDA kernels and takes several minutes. The build
enables AVX-512 VNNI/BF16/VBMI for Zen 4/5 and Ice Lake or newer; on an older CPU use `.\scripts\build.ps1 -Portable`.

```powershell
git clone --recurse-submodules --shallow-submodules -b claude/qwen-custom-inference-engine-jhydpg https://github.com/tony1877/eightfer C:\src\eightfer
cd C:\src\eightfer
.\scripts\build.ps1
```

## Bench

Close anything holding VRAM first; check with `nvidia-smi`. Then:

```powershell
.\build\bin\eightfer.exe bench 2>&1 | Tee-Object bench.txt
```

What it does (about 3–5 minutes):

- RAM read bandwidth, then ggml matrix-vector speed on the CPU and the GPU, 1 to 16 tokens per pass.
- Pins up to 16 GiB of RAM, then measures GPU upload/download speed, alone and while the CPU is also reading RAM.
- On every fixed drive, writes a 4 GiB temp file to `X:\eightfer_bench_tmp\`, measures unbuffered reads, and
  deletes the file.
- `--disk D:\some\folder` limits the disk test to chosen drives.
- `eightfer bench --help` lists all options.

Kernels come from [ggml](https://github.com/ggml-org/llama.cpp) (MIT), pinned as a submodule at `836d571`.
The runtime, scheduling, storage tiers and split-precision format are eightfer's.

## Serving (OpenAI-compatible)

```
eightfer serve <base.gguf> [--res <res.gguf>] --port 8090 --alias NAME --api-key-file KEYFILE                --chat-template-file TEMPLATE.jinja --ctx 16384 --kv q8_0 --spec auto
```

- Qwen3.8-27B: pack it first (`eightfer pack`), then serve the base with `--res`: base + residual quality,
  self-speculative decoding (14-16 tok/s at temp 1.0 on an RTX 5080 + 9800X3D).
- Flash-Next: serve the GGUF directly (experts stay memory-mapped; a GPU expert cache takes free VRAM;
  18.8 tok/s, 24 warm).
- Requests and responses follow llama-server: `chat_template_kwargs` (e.g. `enable_thinking`), `reasoning_content`,
  `tools` / `tool_calls`, streaming with usage and timings in the last chunk.
- Long context (up to the models' 262144): pass `--ctx 262144 --kv q8_0`. When the 27B's KV does not fit in VRAM next
  to the weights, the full KV (9.1 GB at 256K) lives in RAM and a window of recent tokens (auto, `--gpu-kv N`) stays
  in VRAM for the drafter. 27B at a 261,776-token prompt: prefill 416 tok/s (~10.5 min), decode 3.3 tok/s at a full
  context (4.4 with `E8_KV_PINNED=1`, which pins the KV instead of the residual: prefill 342 tok/s). Decode is
  bound by copying the RAM KV to the GPU each verify, so it is faster the less of the context is used.
- Flash-Next long prompts are processed layer by layer (65536-token chunks), so each layer's experts are read from
  disk once per chunk instead of once per 512 tokens: 6K-token prompt 110 tok/s (was 10.7); a 261,776-token prompt
  prefills at 95 tok/s (46 min) and then decodes at 9.0 tok/s, needle at position 0 recalled.
