# M3 on the target box: orcarouter/Qwen3.8-27B-Uncensored, 2026-10-04

RTX 5080 16 GB (iGPU drives the displays), Ryzen 7 9800X3D, 32 GB DDR5-5600. eightfer at commit after `f50233e`.

## Pack

`eightfer pack` from the BF16 safetensors, imatrix from llama-imatrix on the Q8_0 (96 x 512 tokens of Python stdlib
source, `build-ref/calib.txt`). 12 min.

| File | Size | Contents |
|---|---|---|
| base | 16.48 GB | IQ4_XS big matrices (13.7 GB on GPU), BF16 token embedding (2.5 GB, CPU), F32 small tensors |
| residual | 14.64 GB | Q4_K of W - deq(B) (14.4 GB pinned RAM) |

Weight-space rel. RMSE over all big matrices: base 0.0779, base + residual 0.0051.

## Quality (ctx 512, 8 chunks of llama.cpp docs, 2040 scored tokens; reference = Q8_0 packed from the same BF16)

| Model | PPL | mean KLD vs Q8_0 | max KLD | same top-1 |
|---|---|---|---|---|
| Q8_0 (llama.cpp) | 4.0067 | - | - | - |
| base alone (IQ4_XS, imatrix) | 4.0430 | 0.0577 | 1.89 | 91.1% |
| **base + residual** | **4.0022** | **0.00199** | 0.086 | **97.7%** |

The residual cuts KLD 29x. The orca Q5_K_M comparison failed (llama-perplexity -ngl 99 ran out of VRAM) and is
still to be rerun with partial offload.

## Self-speculative decoding (greedy, 256 tokens, chat prompts in `bench/prompts`)

Plain base + residual decoding: **3.3 tok/s** (302 ms/token; the CPU pass over the 14.4 GB residual dominates).

| Prompt | k=4 | k=6 | k=8 | acceptance k=4 / 6 / 8 |
|---|---|---|---|---|
| code | 8.93 | 9.48 | 9.79 | 0.95 / 0.97 / 0.93 |
| reasoning | 9.23 | 9.56 | **10.38** | 0.99 / 0.98 / 0.97 |
| prose | 9.18 | 8.76 | 8.87 | 0.95 / 0.89 / 0.83 |
| explain | 9.02 | 8.97 | 9.52 | 0.97 / 0.92 / 0.90 |

tok/s, decode only. **2.7-2.8x over plain** at identical output: `--compare` gave IDENTICAL 256-token outputs for
code, reasoning and explain; prose diverged at token 35 on a near-tie (logits 23.585 vs 23.551, within the GPU's
batch-shape noise).

Per cycle (k=6): draft 143 ms (24 ms/token, all-GPU IQ4_XS), verify 575 ms, rollback < 1 ms.

## Versus the DESIGN estimate

Acceptance is at or above the DESIGN's best row (0.95). The verify pass is the gap: 440 ms at k=4 to 670 ms at k=8,
against 250 ms estimated, because the CPU matmul over the plain (non-repacked) Q4_K residual is compute-bound and
scales with the column count. Next: a faster residual GEMM (repacked layout or a custom AVX-512 VNNI kernel), and
splitting the verify so the GPU base pass and CPU residual pass overlap.
