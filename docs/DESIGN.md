# eightfer — design v0

Status: M1-M4 done; M5 and M6 usable, with open items listed below.

Legend: **[verified]** = checked against source/data in this repo or upstream files.
**[est.]** = arithmetic from bandwidth figures, not measured. **[measure]** = must be
measured on the target box before we rely on it.

---

## 1. Target box

Parts as reported by your `nvidia-smi`, `Get-PhysicalDisk`, `Win32_PhysicalMemory` and `nvcc` output:

| Part | Spec | Number we plan with |
|---|---|---|
| OS | Windows 11, PowerShell 7.6 | — |
| GPU | RTX 5080 16 GB GDDR7, PCIe **Gen5 x16** (no lane sharing), sm_120, driver 616.56 | **822–864 GB/s** VRAM read, 15.89 GiB total / 14.6 GiB free to ggml [measured]; H2D **57.5 GB/s** pinned, 22.4 pageable [measured] |
| CPU | Ryzen 7 9800X3D, 8C/16T Zen 5, AVX-512 (+VNNI/BF16/VBMI), single CCD | **62 GB/s** RAM read from cores [measured]; CPU + GPU DMA together **80 GB/s** [measured] |
| RAM | 2 × 16 GB DDR5-5600 CL30 (runs at 5600) | ~22 GB usable for model data [est.]; **pinned ≤ 15.6 GiB (WDDM: half of RAM)** [measured] |
| NVMe | WD_BLACK SN850X 1 TB (Gen4), Samsung 980 PRO 500 GB (Gen4), each on its own CPU root port | 7.05 / 5.88 GB/s at 2.5 MiB random; **12.45 GB/s** striped [measured] |
| SATA | Micron 5400 3.84 TB + 7.68 TB | ~0.55 GB/s each; fine for sparse 4 KiB reads [not measured] |
| Toolchain | CUDA 13.0, VS 2022 Build Tools (MSVC 19.44) | CUDA 13.0 accepts VS 2019/2022 as host compiler, not VS 2026 |

Bench output and analysis: [`bench/results/2026-10-04-rtx5080-9800x3d`](../bench/results/2026-10-04-rtx5080-9800x3d/README.md).

## 2. The models [verified: HF `config.json`, transformers `modeling_qwen4_exp.py`]

| | Qwen3.8-27B | Qwen3.8-Flash-Next |
|---|---|---|
| arch | `qwen3_5`, dense | `qwen4_exp`, MoE |
| layers | 64 = 48 Gated DeltaNet + 16 gated full attention | 48 = 36 Gated DeltaNet + 12 QSA sparse attention (indexer, budget 2048 tokens) |
| hidden | 5120 | 2560, widened to 4 hyper-connection streams |
| FFN | dense, 17408 | 512 experts × 640, top-10, + 1 shared expert |
| params | 25.6B in big matrices + 1.27B embedding | 125B main + 51.2B n-gram embedding table; ~6B active/token |
| extras | 1 MTP layer, vision tower | 1 MTP layer, vision tower, n-gram "PLE" injected at layer 1 |
| license | Apache-2.0 | Qwen Community License 1.0 |

Why a custom engine at all:

- 27B at Q8_0 is 29.1 GB. It does not fit 16 GB VRAM. Splitting it GPU/CPU gives ~4 tok/s [est.].
- Flash-Next's smallest GGUF (IQ1_S) is 70 GB, larger than VRAM + RAM combined. Every quant must stream from disk.
- ggml already has CUDA kernels for every op both models need [verified, llama.cpp `836d571`]:
  `ggml_gated_delta_net` (with K state snapshots), `ggml_lightning_indexer`, `ggml_dsv4_hc_pre_gated` /
  `ggml_dsv4_hc_post`, `ggml_ssm_conv`, `ggml_rope_multi`, `ggml_mul_mat_id`.
  The bottleneck is not kernels. It is where the bytes live and how often they move.
  eightfer reuses ggml for kernels and owns everything above them.

## 3. Core idea: split-precision weights ("8 for 4")

Every large weight matrix `W` is stored as two quantized tensors:

```
B = Q_base(W)            # IQ4_XS, 4.25 bpw  -> a normal 4-bit model on its own
R = Q_res(W - deq(B))    # Q4_K,   4.50 bpw  -> the correction
W ≈ deq(B) + deq(R)      # ~Q8_0 quality or better
```

Because `x·Wᵀ ≈ x·Bᵀ + x·Rᵀ`, the two halves can live in **different memories** and be computed by
**different processors**, then summed. The 4-bit half is also a free draft model: it costs zero extra memory.

Measured on three real Qwen3.8-27B BF16 tensors, ggml quantizers, no imatrix
(`experiments/nested_quant`) [verified]:

| Format | bpw | Weight rel. RMSE vs Q8_0 |
|---|---|---|
| Q8_0 | 8.50 | 1.00× |
| Q6_K | 6.56 | 3.3× |
| Q5_K | 5.50 | 6.7× |
| Q4_K | 4.50 | 13.2× |
| IQ4_XS | 4.25 | 14.2× |
| IQ4_XS + Q3_K residual | 7.69 | 1.9× |
| **IQ4_XS + Q4_K residual** | **8.75** | **0.89×** |
| IQ4_XS + Q5_K residual | 9.75 | 0.46× |

The residual width is a quality dial: Q3_K (smaller) → Q4_K (≈ Q8_0) → Q5_K (2× better than Q8_0).
Measured in M3: output KLD vs Q8_0 0.0020 for base + residual (0.058 base alone), top-1 agreement 97.7%; per-draft acceptance 0.80–0.94 at short drafts, 23–37 tokens per verify with long drafts.

## 4. Qwen3.8-27B: self-speculation across VRAM and RAM

Placement (decimal GB) [est. from param counts]:

| Where | What | Size |
|---|---|---|
| VRAM | B for all big matrices (IQ4_XS) | 13.6 GB |
| VRAM | KV cache, 16 attn layers, 32K ctx, q8_0 | 1.1 GB |
| VRAM | GDN recurrent state (fp32) + conv state | 0.16 GB |
| VRAM | compute buffers | ~0.4 GB |
| RAM (pinned) | R for all big matrices (Q4_K) | 14.4 GB |
| RAM | token embedding, BF16 (one row read per token) | 2.5 GB |

Decode loop:

1. **Draft**: GPU runs k tokens with B only. 13.6 GB per token at ~770 GB/s → ~18 ms/token [est.].
2. **Verify**: run the k+1 tokens through B+R in one pass. GPU computes `x·Bᵀ` and the CPU (AVX-512)
   computes `x·Rᵀ` at the same time. Partial sums are added per matmul. Decode is memory-bound, so reading R once
   serves all k+1 tokens: ~14.4 GB at ~60 GB/s → ~250 ms [est.].
3. **Accept** using speculative sampling (Leviathan et al. 2023; Chen et al. 2023). The output
   distribution is exactly that of the B+R model, so the speedup is **lossless** relative to ≈Q8.
4. **Rollback** on rejection:
   - Attention KV: truncate.
   - GDN: keep the committed state. During verify, record per-token `(q,k,v,g,β)` and the conv inputs.
     On accepting j tokens, replay j recurrent steps from the committed state (`ggml_gated_delta_net` on j tokens).
     This avoids holding k snapshots of a 151 MB state.

Throughput [est., draft 18.5 ms/token, verify 250 ms]:

| Accept rate α | Drafts k | Tokens/cycle | tok/s at ≈Q8 quality |
|---|---|---|---|
| 0.85 | 5 | 4.2 | ~12 |
| 0.90 | 6 | 5.2 | ~14 |
| 0.95 | 8 | 7.4 | ~19 |

Baselines [est.]:
- Q8_0 GGUF split across GPU and CPU: ~4 tok/s.
- IQ4_XS fully on the GPU: ~45–50 tok/s, at 4-bit quality.

Prefill runs at B+R precision with R streamed to the GPU over PCIe, layer by layer. Prefill is compute-bound, so the
transfer hides behind the matmuls for large chunks.

Optional drafter: the model's own MTP head. Drafts are cheaper but less accurate; we compare it against the B-only
drafter in M3.

## 5. Qwen3.8-Flash-Next: tiered expert streaming

Facts [verified]:
- One expert (gate+up+down) is 4.92M params, 2.61 MB at IQ4_XS.
- There are 512 × 48 = 24,576 experts, 64.2 GB at IQ4_XS.
- A token reads 10 × 48 = 480 experts, 1.25 GB.
- The n-gram table is read for 16 rows × 160 dims per token, 5 KB at BF16. Row addresses depend only on the last 3
  token ids, so they are known **before** the forward pass.

Placement [est.]:

| Tier | Holds |
|---|---|
| VRAM ~15.5 GB | Non-expert weights (attention, GDN, shared expert, routers, hyper-connections, lm_head) at Q8_0 ≈ 4.5 GB; KV ≈ 0.8 GB (64K, q8_0); GDN state 0.11 GB; **expert cache ≈ 9.5 GB ≈ 3,600 experts (15%)** |
| RAM ~22 GB pinned | **Expert cache ≈ 8,400 experts (34%)**. The CPU computes RAM hits directly. |
| NVMe (SN850X + 980 PRO) | Expert base store B (64 GB at IQ4_XS): 4 KiB-aligned, expert-major (one read per expert), striped across both drives in proportion to measured bandwidth |
| SATA (Micron 5400s) | N-gram table at **BF16** (102 GB), lossless, read sparsely. Expert residuals R: hot ones load once at startup. Download/source files. |

Mechanisms:

- **Heat cache**: per-layer LFU with decay across VRAM and RAM; admission and eviction by heat.
- **Lookahead prefetch**:
  - Apply layer l+1's router to layer l's hidden state and issue NVMe reads one layer early.
  - N-gram rows are prefetched as soon as the token is sampled.
- **Precision follows heat**: hot experts keep R resident (≈Q8); cold experts run B only.
  - A static plan from a calibration profile makes this a fixed, deterministic mixed-precision model.
  - `--lossless-q8` streams R for misses too, at 2× miss I/O.
- **Miss path**: Windows unbuffered overlapped reads on an IOCP into pinned staging buffers. Then either
  `cudaMemcpyAsync` to the GPU or a CPU compute in place.

I/O ceiling [est.]. This is not a speed prediction; compute and sync come on top.
Misses per token = 480 × (1 − h) × 2.61 MB.

| Cache hit rate h | 1× Gen4 (7 GB/s) | SN850X + 980 PRO striped (14 GB/s) |
|---|---|---|
| 0.5 | 11 tok/s | 22 tok/s |
| 0.6 | 14 tok/s | 28 tok/s |
| 0.7 | 19 tok/s | 37 tok/s |

Striping only doubles the rate if the two drives don't share the chipset uplink (PCIe 4.0 x4, ~7 GB/s).
The bench's "all drives at once" line answers this.

Speculation for Flash-Next is optional. MTP or cache-constrained self-drafting amortizes per-token sync and shared
experts. Whether it pays off under I/O load is an M5 measurement.

## 6. Engine layout (planned)

```
eightfer/
  third_party/llama.cpp   pinned submodule; we build only its ggml targets (+ libllama vocab-only for tokenizer)
  src/core/       GGUF reader, split-precision tensor (B,R), memory tiers, pinned pools
  src/io/         IOCP unbuffered reader (Windows), pread fallback (Linux), drive probe
  src/models/     qwen35 + qwen4exp graph builders on ggml
  src/runtime/    draft/verify scheduler, GDN replay rollback, expert cache, prefetcher
  src/sample/     samplers + speculative acceptance
  tools/          bench | plan | pack (residual builder) | run | serve (OpenAI-compatible)
```

Inputs:
- A stock GGUF runs in B-only mode from day one (bartowski/unsloth files), at short context.
- For B+R and for 200k context, `eightfer pack` builds a **pure** IQ4_XS base from the BF16 safetensors
  (13.61 GB; a stock IQ4_XS GGUF is 13.99–14.10 GB because llama.cpp upgrades output, attn_v and early
  ffn_down, and that doesn't fit at 200k) plus the residual pack against that exact base:
  `R = Q(W_bf16 − deq(B))`.

Build: CMake + MSVC 2022 + CUDA ≥ 12.8 (sm_120). Linux CPU build for CI and correctness tests.

## 7. Milestones

| # | Deliverable | Done when |
|---|---|---|
| M0 | This design + nested-quant measurement | ✅ |
| M1 | Skeleton + `eightfer bench` | ✅ Native Windows build and bench on the target box, 2026-10-04 ([results](../bench/results/2026-10-04-rtx5080-9800x3d/README.md)). |
| M2 | qwen35 graph + GGUF loader (B-only) | ✅ 2026-10-04. Tiny random model (8 layers, F32, CPU) vs transformers 5.18 (`tests/tiny/run.py`): batch 1–2 with F32 KV rel.err 8e-7, KLD 1e-12; batch ≥32 rel.err 9e-4, KLD 4e-7 (ggml's tiled CPU kernels). 27B Q5_K_M vs llama.cpp (36 GPU layers, ctx 512): batch 512 exact (PPL 3.5378 both, KLD 0.000000); batch 128 identical to llama.cpp `-ub 128`; batch 1 KLD 0.0011 vs llama.cpp `-ub 1` (llama.cpp's own ub1-vs-ub512 KLD is 0.0067; not repack, not fused-GDN). |
| M3 | `eightfer pack` + B+R verify + self-speculation | ✅ 2026-10-04 ([results](../bench/results/2026-10-04-m3-orca27b/README.md)): B+R KLD vs Q8_0 0.0020 (base alone 0.058), top-1 97.7%. 14-16 tok/s at temp 1.0 (sampled drafts, acceptance 0.80-0.94) and 11-14 greedy, vs 3.1 tok/s plain B+R (4-4.8x); greedy output identical to plain. 2026-10-05: long drafts (`--spec auto` up to 63; verify batches of 32+ stream R to the GPU, ~330 ms flat) and MTP-staged drafts (MTP → B → B+R, 13 vs 19 ms per draft token): 27.8 tok/s sampled / 27.9 greedy over the 4 prompts (16–47 by prompt). Greedy matches plain at k=6; bigger verify batches can flip near-ties (rounding). Drafting during the verify (a second instance with its own recurrent state drafting on another thread) was built and dropped: on 16 GB it fits only without MTP (the GPU verify's 0.77 GB compute buffer leaves 0.09 GB beside MTP), it slowed the verify 330 → 420 ms, and continuations were kept in 3–8 of ~19 cycles, so 16–17 tok/s vs 28 with MTP. |
| M4 | qwen4exp graph (B-only, mmap) | ✅ 2026-10-05 ([results](../bench/results/2026-10-05-m4-flashnext/README.md)): tiny random model vs transformers rel.err 3.4e-7 / KLD 1e-14 incl. QSA sparse selection, PLE n-gram hashing and hyper-connections. Real 85 GB IQ3_XXS: KLD 0.035 vs llama.cpp, within llama.cpp's own ub32-vs-ub512 KLD (0.032); same speed. |
| M5 | Expert store, heat cache, IOCP streaming, prefetch | 🟡 2026-10-05 ([results](../bench/results/2026-10-05-m5-flashnext-decode/README.md)): GPU expert cache (heat-based, 77% hit rate with 9.8 GB), sparse CPU MoE kernel, background prefetch, CUDA graphs: Flash-Next IQ3_XXS decodes at 18.8 tok/s (24.3 warm) vs llama.cpp 9.5. Not yet: IOCP + pinned RAM tier (OS page cache now), striping, planner. |
| M6 | OpenAI-compatible server, chat templates, vision | ✅ 2026-10-05 (vision dropped from scope): `eightfer serve` (/v1/chat/completions with streaming, /v1/models, API key; llama.cpp's common library for Jinja templates, reasoning_content and tool_calls; prompt reuse through a checkpoint before the generation prompt). Tested on the 27B (base + residual, speculative) and Flash-Next. Also /v1/completions, presence/frequency penalties (exact under speculation), concurrent requests queued first come first served with disconnect cancellation (`tests/server_smoke.ps1`). 256K context on both models (2026-10-05): 27B with the full KV in RAM and a VRAM window for drafts (261,776-token prompt: prefill 414 tok/s; decode 10.6 tok/s at full context with sparse page-selected attention, 3.3 exact; 18.2 with long drafts); Flash-Next with layer-major long-prompt prefill (95 tok/s, decode 9.0 tok/s at full context). |

## 8. Unknowns to measure (not assume)

- Draft acceptance α: IQ4_XS draft vs B+R target.
- Expert routing locality → cache hit rate h on real prompts.
- Sustained NVMe bandwidth and thermal throttling (the bench measures 1.5 s bursts).

Measured on 2026-10-04 ([results](../bench/results/2026-10-04-rtx5080-9800x3d/README.md)):
- CPU GEMV, Q4_K, 9 columns: compute-bound, as the sandbox hinted. Best 54 ms per pass (38.7 GB/s of 62,
  `CPU_REPACK`, 16 threads). The verify step needs SPEED2X's dual path.
- Pinned memory: WDDM caps it at half of RAM, 15.6 GiB here, ~113 ms/GiB to allocate. R (14.4 GB) fits; R plus
  pinned 200k KV does not.
- H2D under WDDM: 57.5 GB/s pinned at 1 GiB, 50.8 GB/s at 2.5 MiB, 22.4 GB/s pageable.
- CPU + GPU DMA together: 80.4 GB/s (43.9 + 36.5), above the 62 GB/s CPU-only figure. Hypothesis confirmed.
- The two NVMe drives sit on separate CPU root ports: 12.45 GB/s together.

## 9. Non-goals (for now)

Multi-user batching, training, non-NVIDIA GPUs, Linux io_uring path, vision before M6.

## 10. 2x speed and 200k context

Full plan, memory tables, review outcomes and milestones: [`docs/SPEED2X.md`](SPEED2X.md).
Models: [`experiments/speed2x/`](../experiments/speed2x/). All numbers are [est.] until the bench runs.

| | 27B @ 4k | 27B @ 200k | Flash-Next @ 4k | Flash-Next @ 200k |
|---|---|---|---|---|
| Current design, same assumptions | 13.6 tok/s | 9.4 | 12.7 (both NVMe on separate links) | 11.8 |
| **New plan, lossless (central)** | **26–27** | **16.1** (17.4 if VRAM allows) | **13.8** | **12.8** |
| Range over unmeasured inputs | 22–30 | 14–19 | 7.6 (drives share chipset) – 14.3 | 7.0 – 13.3 |
| 200k-token prompt prefill | | 265–300 s; ~1 s from prefix cache | | ~40–70 s; ~0.4 s from prefix cache |

27B mechanisms, all lossless:
- **Dual-path verify.** R is split by rows: the CPU multiplies its rows in place while the rest streams over PCIe and is
  multiplied on the GPU. This fixes the sandbox's "CPU is compute-bound at 9 tokens" problem.
- **MTP-staged drafting.** The model's own MTP head drafts for the 4-bit base, and the base drafts for B+R
  (MTP → B → B+R). Draft cost halves, from 18.5 to ~9 ms/token.
- **Draft during verify.** The GPU keeps drafting while the RAM-bound verify runs.
- **Up to 3 hedges.** At uncertain positions the drafter also continues from the base's second choice, used only when
  it matches the target's own correction.

200k context fits for both models:
- **27B:** q8_0 KV in pinned RAM, read exactly once per verify by a custom split-KV attention kernel. The drafter only
  sees a recent window plus a hot set. VRAM 15.50/15.5 GB, RAM 21.9/22 GB; the full-speed ceiling is ~203k tokens.
- **Flash-Next:** QSA KV in pinned RAM. Each step gathers only the 2048 tokens the indexer selected.
- **Both:** an NVMe prefix cache restores a 200k context in about 1 s instead of re-prefilling.

What is not reachable, stated plainly:
- **27B at 200k, 2x:** every verify must read ≥21.2 GB (R + KV) from DRAM, which caps it near 26 tok/s even with free
  drafting.
- **Flash-Next, software 2x:** there is no software path within 32 GB RAM. New lossless mechanisms add ~1.09x. The
  only lossless 2x is **2 × 32 GB RAM** (24–28 tok/s).

Opt-in, quality-gated options:
- Q3_K residual: ~29 tok/s at 4k. Weight error 1.9× Q8_0, still better than Q6_K.
- q4_0 far-context KV: 19.3 tok/s at 200k.
- Cold experts at IQ3_XXS: Flash-Next ~17.5 tok/s.

## 11. Bit-level lossless mode

See [`docs/BITLEVEL.md`](BITLEVEL.md). Exact BF16 = B (4.25 bpw, VRAM) + R (4.5, RAM) + T2 (~3, exact remainder).
The rebuild was bit-exact on 204.5M real weights. A draft → ≈Q8 → exact cascade gives BF16-exact output at ~18–19 tok/s
(4k), against ~28 for the default ≈Q8 target. All speeds are estimates.

## References

- Speculative decoding: Leviathan et al. 2023, *Fast Inference from Transformers via Speculative Decoding*;
  Chen et al. 2023, *Accelerating Large Language Model Decoding with Speculative Sampling*.
- Related nested-precision work: Park et al. 2024, *Any-Precision LLM*; Nair et al. 2025, *Matryoshka Quantization*.
- Model configs: `huggingface.co/Qwen/Qwen3.8-27B`, `huggingface.co/Qwen/Qwen3.8-Flash-Next`.
- Kernels: `github.com/ggml-org/llama.cpp` @ `836d571` (MIT).
