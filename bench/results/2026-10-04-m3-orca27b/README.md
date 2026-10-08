# M3 on the target box: orcarouter/Qwen3.8-27B-Uncensored, 2026-10-04

RTX 5080 16 GB (iGPU drives the displays), Ryzen 7 9800X3D, 32 GB DDR5-5600. shoehorn at commit after `f50233e`.

## Pack

`shoehorn pack` from the BF16 safetensors, imatrix from llama-imatrix on the Q8_0 (96 x 512 tokens of Python stdlib
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

## After the verify fixes (same day)

Changes: a one-pass multi-column Q4_K kernel for the residual (`src/kernels/q4k_small.cpp`, maddubs/madd, two rows
per pass), 16 CPU threads by default (SMT gains ~10% on the residual pass), host-side embedding lookup, adaptive k
(`--spec auto`) and sampled drafts for temperature sampling (accept with min(1, p/q), resample from max(0, p - q)).

Verify of 7 tokens: 577 ms -> 383 ms. The residual pass is now at the CPU's all-core AVX-512 ceiling
(`shoehorn selftest --kernel-bench`: ~30 GB/s at 9 columns on 8 threads, ~41 GB/s on 16; it scales linearly to 4
cores and then flattens, so it is power/clock limited rather than memory limited).

| Prompt | greedy, auto k | temp 1.0 / top-p 0.95 / top-k 20, auto k | acceptance (sampled) | before (greedy, best k) |
|---|---|---|---|---|
| code | 14.1 tok/s | **14.4** | 0.90 | 9.8 |
| reasoning | 14.4 | **15.1** | 0.94 | 10.4 |
| prose | 11.2 | **13.0** | 0.81 | 9.2 |
| explain | 13.7 | **14.3** | 0.90 | 9.5 |

Plain base + residual decoding: 3.1 tok/s, so speculation gives 4.0-4.8x. Greedy outputs stay identical to plain
decoding (`--compare`), apart from near-ties within GPU batch-shape noise.

Before the sampled draft, a greedy draft under temp 1.0 accepted only 0.55-0.77 (7.8-12 tok/s).

Remaining: drafting is 24 ms/token (13.7 GB at ~570 GB/s vs 822 GB/s measured), and the GPU idles during the
~400 ms residual pass; drafting the next cycle during the verify would hide ~280 ms per cycle when all drafts are
accepted.

## CUDA graphs + graph-stable KV writes (2026-10-05)

shoehorn's ggml build had CUDA graphs off, and the KV write was a view at offset n_past, which changes the graph every
token so graphs could not replay. With both fixed, drafting went from 24 to 19.4 ms/token:

| Prompt (temp 1.0, top-p 0.95, top-k 20, `--spec auto`) | tok/s | acceptance |
|---|---|---|
| code | 15.75 | 0.90 |
| reasoning | 15.88 | 0.94 |
| prose | 13.96 | 0.81 |
| explain | 15.64 | 0.90 |
