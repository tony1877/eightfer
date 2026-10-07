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
eightfer serve <base.gguf> [--res <res.gguf>] --port 8090 --alias NAME --api-key-file KEYFILE \
               --chat-template-file TEMPLATE.jinja --ctx 16384 --kv q8_0 --spec auto [--mtp 3]
```

- Qwen3.8-27B: pack it first (`eightfer pack`), then serve the base with `--res`: base + residual quality,
  self-speculative decoding. On an RTX 5080 + 9800X3D, short context, 4 benchmark prompts: 27.8 tok/s at temp 1.0
  and 27.9 greedy (16-47 by prompt), vs 3.1 tok/s for plain base + residual decoding. Two things make it fast:
  - Long drafts. `--spec auto` picks 1-63 drafts per cycle from the measured acceptance by draft position and the
    measured verify cost per batch size. From 32 tokens ggml streams the residual to the GPU (~330 ms per verify,
    flat in the batch size, vs ~900 ms for 16 tokens on the CPU), so runs of 31-50 drafts pay off.
  - MTP-staged drafts (`--mtp N`, default 3, 0 = off). The model's own MTP block proposes N tokens; one base pass
    checks them by speculative sampling and adds a token of its own. The kept tokens are exact samples from the base,
    so they draft for base + residual as before. 13 ms per draft token vs 19 for the base alone.
  - Echo drafting (`--echo 1`, default). When the last 8 tokens occurred earlier in the context, the tokens that
    followed them (up to 63) are proposed instead of MTP's: agents re-emit file contents, code and tool arguments.
    The base checks the copy in one pass like an MTP proposal. Synthetic agent tasks at 256K context (file rewrite,
    str_replace call, refactor): 8.1 / 29.1 / 22.5 -> 36.3 / 49.4 / 29.8 tok/s, 2.8x end to end; it also fixes
    drafting once the copied text has left the drafts' VRAM window.
  - At 256K context MTP now fits beside the RAM KV: proposal checks run "dry" (recurrent state untouched, the kept
    tokens committed by replay) instead of snapshotting 0.16 GB of state; the output head's residual is staged in
    VRAM in row pieces (verify compute buffer 0.84 -> 0.23 GB); prompts past the VRAM ring run in 256-token sparse
    chunks instead of reserving 1.1 GB for exact staging. 256K: decode 18.2 -> 22.7 tok/s on a summary, prose
    ~13 -> 22 tok/s, prefill 403 -> 456 tok/s, needle recalled, 15.1 GB peak VRAM.
  - Residual uploads overlap compute ([`patches/ggml-weight-prefetch.patch`](patches/ggml-weight-prefetch.patch),
    applied by `build.ps1`): ggml's scheduler stages host weights in two VRAM slots from a second CUDA stream, the
    next layer's residual uploading while the current one computes. A verify costs ~255 ms at any size (was 330 ms
    from 32 tokens, 275-900 ms on the CPU below), so every verify runs on the GPU and drafts are at least 24 long.
    Prefill runs in 1024-token batches; with the KV in RAM their attention runs in 256-query sub-chunks, so the
    residual crosses PCIe once per 1024 tokens. Prefill: 6K prompt 725 -> 1170 tok/s, 64K prompt at 256K context
    495 -> 952, 261,776 tokens 456 -> 835 tok/s (586 -> 324 s, decode 24.4 tok/s at full context, needle recalled,
    15.4 GB peak VRAM). Decode, 8 seeds at temp 1.0: prose 28, explain 36-38, code 43-49, reasoning 59 tok/s. Prose stays drafter-bound: the 4-bit base's
    drafts are kept ~50% of the time, so about half the drafting is wasted (see `gen --repeat`).
  Output follows the base + residual distribution exactly (speculative sampling at both levels). Greedy output is
  token-identical to plain decoding up to rounding: verify batches of 16+ tokens can flip a near-tie (k=6 matches
  plain over 256 tokens).
- Flash-Next: serve the GGUF directly (experts stay memory-mapped; a GPU expert cache takes free VRAM;
  18.8 tok/s, 24 warm).
- Requests and responses follow llama-server: `chat_template_kwargs` (e.g. `enable_thinking`), `reasoning_content`,
  `tools` / `tool_calls`, streaming with usage and timings in the last chunk.
- Endpoints: `/v1/chat/completions`, `/v1/completions` (prompt string or token ids, no template), `/v1/models`,
  `/health`. Sampling: `temperature`, `top_p`, `top_k`, `min_p`, `seed`, `presence_penalty`, `frequency_penalty`
  (over the generated tokens, applied exactly inside speculative decoding), `stop`, `max_tokens`.
- Concurrent requests are accepted and run one at a time, first come first served (the model holds one sequence). A
  streamed request whose client disconnects stops at the next decode cycle. `tests/server_smoke.ps1` covers all of it.
- Long context (up to the models' 262144): pass `--ctx 262144 --kv q8_0`. When the 27B's KV does not fit in VRAM next
  to the weights, the full KV (9.1 GB at 256K) lives in RAM and small decode/verify batches use sparse attention:
  per 64-key page a midpoint key in VRAM ranks pages for the batch's queries, and attention is exact over the top 128
  pages, the first page and the last 4096 tokens, gathered from RAM. Drafts see the verify's best pages through a
  VRAM far area plus a ring of recent tokens. Quality: 32K text PPL 2.3301 vs 2.3317 exact; text repeated 48K tokens
  back 1.0006 vs 1.0004 exact (`E8_SPARSE=0` forces exact attention). 27B with a 261,776-token prompt: prefill
  403 tok/s (~11 min), decode 18.2 tok/s at full context with long drafts (10.6 with drafts of at most 15, 3.3 with
  exact attention); needle at position 0 recalled. MTP drafting turns itself off when the KV does not fit in VRAM
  (its 0.4 GB would come out of the KV window).
- Flash-Next long prompts are processed layer by layer (65536-token chunks), so each layer's experts are read from
  disk once per chunk instead of once per 512 tokens: 6K-token prompt 110 tok/s (was 10.7); a 261,776-token prompt
  prefills at 95 tok/s (46 min) and then decodes at 9.0 tok/s, needle at position 0 recalled.
