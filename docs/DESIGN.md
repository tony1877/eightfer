# eightfer — design v0

Status: design + first measurement. No engine code yet.

Legend: **[verified]** = checked against source/data in this repo or upstream files.
**[est.]** = arithmetic from bandwidth figures, not measured. **[measure]** = must be
measured on the target box before we rely on it.

---

## 1. Target box

| Part | Spec | Number we plan with |
|---|---|---|
| OS | Windows 11 | — |
| GPU | RTX 5080 16 GB GDDR7, PCIe 5.0 x16, sm_120 | ~770 GB/s effective VRAM read [est.], ~15.5 GB free VRAM [measure] |
| CPU | Ryzen 7 9800X3D, 8C/16T Zen 5, AVX-512, single CCD | ~60 GB/s RAM read from cores (single-CCD fabric cap) [est.] |
| RAM | 32 GB DDR5 | ~22 GB usable for model data [est.] |
| PCIe | 5.0 x16 | ~45–50 GB/s pinned H2D [measure] |
| Disks | NVMe + SATA SSDs | 7 GB/s per Gen4 NVMe, 12–14 GB/s per Gen5 NVMe, 0.55 GB/s SATA [measure] |

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
Not yet measured: output-level KLD and draft acceptance (milestone M3).

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
| NVMe | Expert store: B and R as separate files, 4 KiB-aligned, expert-major (one read per expert), striped across drives in proportion to each drive's measured bandwidth |
| Any SSD (SATA OK) | N-gram table at **BF16** (102 GB), lossless. It is read sparsely, so full precision costs disk space only. |

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

| Cache hit rate h | 1× Gen4 (7 GB/s) | 1× Gen5 (12 GB/s) | Gen5 + Gen4 striped (19 GB/s) |
|---|---|---|---|
| 0.5 | 11 tok/s | 19 tok/s | 30 tok/s |
| 0.6 | 14 tok/s | 24 tok/s | 38 tok/s |
| 0.7 | 19 tok/s | 32 tok/s | 51 tok/s |

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
- A stock GGUF as the base, so bartowski/unsloth files run in B-only mode from day one.
- An eightfer residual pack (a GGUF of R tensors), built locally from the BF16 safetensors
  against that exact base: `R = Q(W_bf16 − deq(B_stock))`.

Build: CMake + MSVC 2022 + CUDA ≥ 12.8 (sm_120). Linux CPU build for CI and correctness tests.

## 7. Milestones

| # | Deliverable | Done when |
|---|---|---|
| M0 | This design + nested-quant measurement | ✅ |
| M1 | Skeleton + `eightfer bench` | Builds on Windows (MSVC+CUDA). Bench reports VRAM BW, RAM BW, pinned H2D/D2H, per-drive unbuffered read BW at QD 1–64, free VRAM. You paste the output. |
| M2 | qwen35 graph + GGUF loader (B-only) | Tiny random-weight model on CPU matches transformers logits. Real 27B GGUF perplexity matches llama.cpp on your box. |
| M3 | `eightfer pack` + B+R verify + self-speculation | KLD of B+R vs BF16 ≤ Q8_0's. Acceptance α measured. tok/s measured. |
| M4 | qwen4exp graph (B-only, mmap) | Tiny random model on CPU matches transformers, including n-gram hashing and hyper-connections. |
| M5 | Expert store, heat cache, IOCP streaming, prefetch | Hit rate and tok/s from real traces. Planner picks the residual budget. |
| M6 | OpenAI-compatible server, chat templates, vision | Works with a standard OpenAI client. |

## 8. Unknowns to measure (not assume)

- Draft acceptance α: IQ4_XS draft vs B+R target.
- CPU GEMV throughput for Q4_K with k+1 ≤ 9 columns on the 9800X3D.
- Pinned-memory limits and allocation time for ~20 GB under Windows WDDM.
- Effective H2D bandwidth under WDDM.
- Combined DRAM bandwidth when the cores and GPU DMA read concurrently. Hypothesis: DMA bypasses the CCD link, so the
  total can exceed ~60 GB/s.
- Expert routing locality → cache hit rate h on real prompts.
- Sustained NVMe bandwidth, thermal throttling, and sector alignment per drive.
- Board lane sharing: on some AM5 boards, populating certain M.2 slots drops the GPU to x8.
  Check the manual or `nvidia-smi --query-gpu=pcie.link.width.max --format=csv`.

## 9. Non-goals (for now)

Multi-user batching, training, non-NVIDIA GPUs, Linux io_uring path, vision before M6.

## References

- Speculative decoding: Leviathan et al. 2023, *Fast Inference from Transformers via Speculative Decoding*;
  Chen et al. 2023, *Accelerating Large Language Model Decoding with Speculative Sampling*.
- Related nested-precision work: Park et al. 2024, *Any-Precision LLM*; Nair et al. 2025, *Matryoshka Quantization*.
- Model configs: `huggingface.co/Qwen/Qwen3.8-27B`, `huggingface.co/Qwen/Qwen3.8-Flash-Next`.
- Kernels: `github.com/ggml-org/llama.cpp` @ `836d571` (MIT).
