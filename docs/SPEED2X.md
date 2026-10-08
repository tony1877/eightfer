# shoehorn: 2x speed and 200k context (DESIGN §10)

Status: design. Produced by 4 independent designs, each attacked by an adversarial reviewer, combined, then corrected by a final critic (all 16 critic corrections applied). Labels in this section: **[M]** measured, **[V]** verified in source or config, **[D]** derived arithmetic from labelled inputs, **[A]** assumed (each one has a bench line in 10.8). GB = 10^9 bytes. tok/s figures come from one cycle-level model per model: a Monte Carlo over verify cycles for the 27B and a closed form for Flash-Next. Monte Carlo noise is about ±0.4 tok/s. The model scripts are in [`experiments/speed2x/`](../experiments/speed2x/).

### 10.1 Result first

| | 27B at 4k | 27B at 200k | Flash-Next at 4k | Flash-Next at 200k |
|---|---|---|---|---|
| Baseline as documented (§4 / task) | 14 tok/s (α 0.90, k 6, 250 ms verify) | does not fit (B + KV > VRAM) | 8–11 tok/s, one drive, h≈0.5 | not planned |
| Baseline, current design under this section's assumptions | 13.6 [D] | 9.4 [D] (v0 with KV streamed + windowed drafter) | 6.9 one drive / 12.7 striped T2 [D] | 6.4 / 11.8 [D] |
| **This plan, lossless (central)** | **27.0 [D]** (≈26.3 after the critic's VRAM corrections) | **16.1 [D]** (17.4 with VRAM surplus) | **13.8 [D]** (T2 topology) | **12.8 [D]** (T2) |
| Factor vs documented / vs same-assumption baseline | 1.93x / 1.98x | 1.15x of 14 / 1.71x | 1.09x of striped v0 (1.03x at equal u); no software 2x | 1.09x of striped v0 |
| Range (pessimistic … optimistic hardware, α 0.90) | 22.2 … 29.9 | 14.0 … 18.6 | 7.6 (T3) … 14.3 (T1) | 7.0 … 13.3 |
| 200k-token prefill | — | ~265–300 s (185–320) | — | ~40 s custom kernels; 45–70 s stock [A] |
| Quality | B+R target, lossless up to fp32 summation order, KV q8_0 | same | defined target (Q8_0 non-expert, IQ4_XS experts, static hot set with R) | same |

What this means:
- **27B at short context.** Central 26–27 tok/s = 1.9x of 14, lossless. 2x is not demonstrated: it needs α ≥ 0.91 or optimistic hardware, and the range over unmeasured inputs is 22–30 tok/s (1.6–2.1x). Treat 28 tok/s as a stretch target, gated at M3c. The opt-in Q3_K residual (10.5) reaches ~29 tok/s with a slightly different target.
- **27B at 200k.** 2x is **not reachable losslessly**. Each verify must read R (14.41 GB) plus the q8_0 KV (6.82 GB) from DRAM. At ≤68 GB/s that alone takes ≥312 ms, and a verify yields at most about 1/(1−α) ≈ 8 tokens. The ceiling with free drafting and no bubbles is about 26 tok/s. The plan reaches 16.1 tok/s (17.4 with the VRAM surplus). The lossy far-context q4_0 KV option (10.5) gives 19.3 tok/s.
- **Flash-Next.** Decode is NVMe-bound. The new lossless mechanisms add 1.09x (uniform routing) to 1.23x (skewed routing) over the current design on the same drives. The "2x of a single drive" comes from the second NVMe sitting on its own link, and §5 already stripes across both drives. If both drives sit behind the chipset uplink (T3), the best lossless result is 7.6/7.0 tok/s (1.09x). **There is no lossless software path to 2x within 15.5 GB VRAM / 22 GB RAM. The only lossless 2x on this box is a RAM upgrade to 64 GB** (10.4.4): 24–28 / 23–27 tok/s.
- **200k context fits for both models** within 15.5 GB VRAM and 22 GB RAM (10.3.4, 10.4.5). Both models are native 262k, so no RoPE scaling is used.

### 10.2 Ground rules for the numbers

Hardware scenarios (all [A] until bench output exists):

| Input | Pessimistic | **Central** | Optimistic |
|---|---|---|---|
| VRAM effective read | 770 GB/s | 770 | 770 |
| Pinned H2D under WDDM, D | 42 GB/s | **46** | 50 |
| Combined DRAM read, cores + DMA (hypothesis H1) | 60 GB/s (H1 false) | **68** | 75 |
| CPU Q4_K × q8 rate, 7 compute cores | 0.26 TMAC/s (stock ggml level) | **0.525** (custom VNNI kernel) | 0.875 |
| CPU idle per CPU↔GPU handoff (WDDM) | 100 µs | **60** | 40 |
| Split-KV q8_0 attention kernel | 40 TFLOPS | **50** | 65 |
| Draft time usable inside a verify (contention derate) | 0.80 | **0.85** | 0.90 |
| MTP chained acceptance vs B (steps 1/2/3) | 0.65/0.50/0.40 | **0.75/0.60/0.50** | 0.80/0.70/0.60 |

- CPU rate in R bytes: c(N) = TMAC × 562.5 / N GB/s (0.5625 B per Q4_K weight). Central at N=12: 24.6 GB/s [D]. The sandbox hint (8 MAC/cycle/core at N=9) scales to about 0.29 TMAC/s for stock ggml on 7 Zen5 cores, so the custom kernel is assumed to be ~1.8x stock [A].
- Acceptance [A]. Per-position α = 0.90 at 4k, as a mix of 70% easy positions (α 0.98) and 30% hard positions (α 0.713). At ≥128k, the drafter's sparse attention costs 0.02. A confidence detector flags hard positions with 75% recall and 15% false positives. A hedge root matches the target's corrected token with probability 0.55 (hard) or 0.40 (easy). The pure-IQ4_XS B also has an IQ4_XS lm_head, which costs about −0.005 to −0.01 α [A]; measured at M3a.
- **Sysmem fallback must be off.** NVIDIA Control Panel → Manage 3D settings → Program settings → shoehorn.exe → CUDA - Sysmem Fallback Policy = **Prefer No Sysmem Fallback**. The plan runs at 99.5% of VRAM, and the default driver policy silently spills overflow into system RAM over PCIe instead of failing. shoehorn compares cudaMemGetInfo with the plan after each allocation phase and refuses to run past it.
- Memory conventions. 15.5 GB VRAM includes a 0.35 GB reserve for the CUDA context, module images and pool slack. 22 GB RAM includes 0.40 GB of engine host overhead. The card is 16 GiB = 17.18 GB, so the measured free VRAM is probably larger than 15.5 GB; the surplus ladder in 10.3.4 uses it.
- "Same-assumption baseline" means the current §4 design (serial drafting with B, then verify) run on the central hardware. The 250 ms CPU verify in §4 needs ≥420 token·GB/s of CPU compute, which the sandbox hint contradicts. The §8 fallback (R streamed over PCIe) gives 325 ms and the best k is 13, so that baseline is 13.6 tok/s [D].

### 10.3 Qwen3.8-27B

#### 10.3.1 Mechanisms (default configuration, all lossless)

| # | Mechanism | Why it stays lossless | Marginal gain at 4k / 200k [D] |
|---|---|---|---|
| L1 | **Dual-path verify.** Each R matmul is split by rows. The CPU multiplies its rows in place (7 cores, AVX-512 VNNI). The rest of R streams over PCIe into a 4×32 MB VRAM ring and is multiplied on the GPU together with x·Bᵀ. At 200k the KV rides the same DMA stream. | Same sum x·deq(B)ᵀ + x·deq(R)ᵀ. **Requirement:** the CPU kernel quantizes activations to q8_1 (32-element blocks), the same as the GPU MMQ/MMVQ path, so results differ only in fp32 summation order. Gate: KLD vs the serial reference ≤1e-5. | serial: 13.6 → 15.3 (1.13x) / 9.4 → 10.7 |
| L2 | **MTP-staged drafter (MTP → B → B+R).** The MTP layer (0.226 GB IQ4_XS, with a 32k-row draft head) proposes 3 tokens. One B pass checks them by speculative sampling, so its output is exact samples from B. Those samples and their stored q are the drafts for the B+R verify. Cost: 22.3 ms per pass for 2.43 tokens = 9.2 ms/token, vs 18.5 ms for B alone. | The inner level reproduces B exactly. The outer test uses q = B's stored distribution. | 15.3 → 21.4 (1.40x) / 10.7 → 13.8 |
| L3 | **Draft during verify (leaf continuation).** A low-priority stream extends the chain past the current window while the verify runs; the verify's GPU work is only 52 of 215 ms at 4k. The first continuation token is accept-tested against p at the bonus position (PEARL-style). | Drafts are q-samples given their prefix. Scheduling only changes when they are produced. **Requirement:** q_B is first stored (f16, renormalized in fp32 on use). The inner MTP→B accept test and its residual use exactly that stored array as their target; the MTP proposal is sampled from its own stored array. The outer B+R accept test and residual use the same stored q_B, so a token whose stored q_B is 0 can never be a draft. Unit test: empirical χ² of the output against p over 1e7 tokens on a toy vocabulary. Re-basing the drafter after a commit only affects tokens not yet sampled. | 21.4 → 24.4 (1.14x) / 13.8 → 15.1 |
| L4 | **Hedges.** At up to 3 detector-flagged positions in the window (plus the bonus position), the drafter also extends a continuation from B's best alternative token. If the target's corrected token equals a hedge root, the next verify starts at once with that continuation. | The root is never accepted by a q-ratio test. It is used only when the target's own residual sample equals it, and its descendants are q-samples given the realized prefix. | 24.4 → 26.4 (1.08x) / 15.1 → 16.1 |
| L5 | **200k KV placement.** The q8_0 KV for all but the last 4096 tokens lives in pinned RAM and streams once per verify. Attention uses a custom split-KV kernel that reads q8_0 directly, outputs log-sum-exp, and merges the streamed part with the VRAM window exactly. Staging holds 2 of the 4 KV heads of one layer. The drafter attends to a draft-only VRAM view: the last 4096 tokens plus a 2048-token hot set per layer, picked from 128-token page-mean keys. | Every verify attends to all keys at q8_0. Sparse attention appears only in the drafter, so it can only lower α (−0.02 at ≥128k [A]). | makes 200k possible |
| L6 | **Resident R slice** in leftover VRAM at short context (0.33 GB at 4k after the corrected replay budget), sized from measured free VRAM. | Same weights in a different memory. | 26.4 → 27.0 (1.02x) / — |
| L7 | **NVMe prefix cache.** Content-addressed 4096-token KV pages plus a GDN+conv checkpoint every 8k tokens. Key = hash of (model, pack, KV type, kernel version, previous key, token ids). | Stores the raw bytes the target computed. A restore equals re-running the same prefill. | re-use of a 200k context: ~1.1 s instead of ~265 s |
| L8 | **Prefill.** Chunks of 2048 tokens. B and R run as two int8 MMQ GEMMs, with R streamed per chunk (0.31 s per chunk vs 2.6 s of compute). Attention uses the L5 kernel over earlier KV streamed from RAM. GDN prefill is chunked (llama.cpp `build_delta_net_chunking`). | Exact B+R and exact causal attention. | — |

Implementation constraints, all [V] at `836d571`:
- `ggml_gated_delta_net` needs F32 q/k/v/g/β and a contiguous [S,S,H,n_seqs] state (no stride-0 broadcast), and every sequence must have the same n_tokens. Drafter hypotheses ("rootless replay") therefore copy the committed root state per hypothesis per layer (3.15 MB). Short hypotheses are left-padded with identity steps: g = 0, which is a log-decay so the state factor is 1, and β = 0. The CUDA kernel runs tokens sequentially (`TODO: Add chunked kernel`).
- **The target's commit replay uses only the fp32 inputs recorded by the verify, never the drafter's.**
- ggml-cuda flash attention converts quantized K/V to an f16 temporary when there are more than 2 queries (`fattn-common.cuh` `need_f16_K/V`), and `ggml_flash_attn_ext` has no LSE output. The custom kernel in L5 is therefore required.
- `ggml_backend_sched` assigns whole ops to one backend, so L1 needs shoehorn's own verify scheduler: 257 joins per verify, spin-polled flags in host-mapped memory, and bulk DMA cut into 2–4 MB chunks. Copy engines ignore stream priority, so small x and partial-sum copies would otherwise wait behind 64 MB transfers. GPU-side waits on host flags must be bounded, with a fallback to event sync: a wait that outlives the WDDM TDR timeout (2 s) resets the driver. Check stream memory operations (`cuStreamWaitValue32` on host memory) under WDDM; without them each join is a host event sync plus relaunch.
- The R pack stays in native Q4_K layout and the custom CPU kernel reads it directly. The split moves freely only while all of R is pinned. Under the pinned-cap fallback, the CPU share is at least the unpinned rows (≥4.2–5.6 GB), so the 4k operating point (CPU 4.6 GB) and the 200k one (5.6 GB) share one boundary. Pinned need at 200k is ≈15.8 GB.

#### 10.3.2 One decode cycle, every shared resource (central, α 0.90 / 0.88)

Verify time [D]: `T_v = S + max(0, R_ram + KV_ram − D·S) / min(c(N) + D, cap) + 3 ms`. S is the time the CPU sits idle with no input: 257 handoffs × 60 µs, GDN 48 × 0.05 ms, attention 16 × max(FLOP/50 TF, staged-KV read), plus KV that could not be pre-staged. During S the DMA keeps streaming.

| Phase | Time | DRAM | PCIe H2D | CPU (7 cores) | GPU / VRAM | NVMe |
|---|---|---|---|---|---|---|
| **4k** verify, N≈11–12 | 217 ms (MC mean; 215 ms at N=12) | 14.0 GB (R − 0.4 resident): 4.6 GB by cores + 9.4 GB by DMA; average 65 GB/s against the 68 cap | 9.4 GB at about 44 GB/s (saturated) | R slice, compute-bound at 24.6 GB/s; idle S = 19 ms | verify work 52 ms (B GEMM 19, ring write+read 24, small kernels 9) + 5 draft passes × 26 ms (4 hypotheses); about 84% busy | idle |
| **4k** bubble (47% of cycles, about 81 ms each) | 38 ms per cycle on average | idle | idle | idle | 3–4 MTP-staged passes × 22 ms | idle |
| **4k cycle** | **255 ms → 6.88 tokens → 27.0 tok/s** | | | | | |
| **200k** verify, N≈12 | 355 ms (MC mean; 351 ms at N=12) | 21.2 GB: R 14.41 + KV 6.82; 5.6 GB by cores + 15.6 GB by DMA; average 60 GB/s | 15.6 GB at about 44 GB/s | R slice 5.6 GB; idle S = 112 ms (handoffs 15, GDN + attention 22, un-staged KV 74) | verify 78 ms + 8 draft passes × 26 ms | idle (KV spill above 203k) |
| **200k** bubble (50%, about 64 ms each) | 32 ms per cycle | idle | idle | idle | MTP-staged passes | idle |
| **200k cycle** | **387 ms → 6.23 tokens → 16.1 tok/s** | | | | | |

Mechanisms that share a resource are not multiplied. DRAM is at its cap during the verify, PCIe is saturated, and VRAM time is split between verify kernels and draft passes by the 0.85 derate. Any bubble leaves CPU, DRAM and PCIe idle. Full-layer KV staging (+0.21 GB VRAM) cuts the 200k S from 112 to 37 ms, so T_v goes from 351 to 327 ms and decode to 17.4 tok/s.

#### 10.3.3 Results

Waterfall (central hardware, k and bubble length re-optimized at each step) [D]:

| Step | 4k (α 0.90) | 200k (α 0.88) |
|---|---|---|
| Current design, same assumptions (serial, §8 DMA verify) | 13.6 | 9.4 |
| + L1 dual-path verify (still serial, B-only drafter) | 15.3 | 10.7 |
| + L2 MTP-staged drafter | 21.4 | 13.8 |
| + L3 draft during verify | 24.4 | 15.1 |
| + L4 hedges (default) | 26.4 | 16.1 |
| + L6 resident R (0.33 GB) / full-layer staging if VRAM allows | **27.0** | 16.1 (**17.4**) |
| Ablations from the full stack: no CPU slice / no MTP / no pipelining | 20.7 / 18.0 / 21.4 | 13.4 / 13.2 / 13.8 |

Context curve (central, α 0.90 minus the drafter penalty) [D]: 4k **27.0** · 32k 24.0 · 64k 22.8 · 128k 18.9 · 200k **16.1** · 262k ≈13.5 (with 2.05 GB of KV spilled to NVMe).

Hardware × α grid, tok/s at 4k / 200k [D]:

| | α 0.85 | α 0.90 | α 0.95 |
|---|---|---|---|
| Pessimistic | 17.8 / 10.9 | 22.2 / 14.0 | 29.9 / 19.2 |
| **Central** | 21.2 / 12.9 | **27.0 / 16.1** | 36.3 / 22.5 |
| Optimistic | 23.5 / 14.5 | 29.9 / 18.6 | 39.6 / 25.9 |
| Same-assumption v0 at 4k (pess / central / opt) | 10.3 / 11.2 / 12.7 | 13.0 / 13.6 / 15.1 | 18.2 / 19.0 / 19.7 |
| Documented v0 (§4 table) | 12 | 14 | 19 |

Single-input sensitivities at central, α 0.90, 4k (200k in parentheses):
- DRAM cap 60 → 25.0 (15.7); cap 75 → 27.4 (16.9).
- Stock-level CPU kernel → 24.5 (15.3).
- Weak MTP → 25.6. Strong MTP → 27.9.
- Handoff 100 µs → 26.5.
- D 40 → 26.1 (15.3); D 50 → 27.0 (17.1).
- Derate 0.70 → 25.7.
- Skeptical hedges (coverage 0.40/0.30, detector 0.65/0.20) → 25.6.
- Drafter α penalty 0.04 at 200k → (14.8).
- α needed for 28 tok/s (2x of 14): 0.91 gives 28.6; 0.92 gives 30.0.

#### 10.3.4 Memory plan [D]

| VRAM (15.5 GB) | 4k | 200k |
|---|---|---|
| B: shoehorn-built **pure** IQ4_XS incl. lm_head (`shoehorn pack`, or llama-quantize with `--output-tensor-type iq4_xs --tensor-type attn_v=iq4_xs --tensor-type ffn_down=iq4_xs`); R is built against this B. A stock IQ4_XS GGUF is 13.99–14.10 GB for these tensors (output Q6_K, attn_v Q5_K, early ffn_down Q5_K) and does not fit at 200k or in prefill. | 13.61 | 13.61 |
| CUDA context / modules / pool reserve [A] | 0.35 | 0.35 |
| GDN committed state fp32 (48×48×128×128×4) + conv | 0.157 | 0.157 |
| MTP layer + 32k-row draft head + MTP KV (4k window) | 0.324 | 0.324 |
| R DMA ring 4 × 32 MB | 0.128 | 0.128 |
| Compute buffers (verify N≤24, draft ≤64 columns, logits, split-KV partials) [A] | 0.30 | 0.30 |
| Draft replay inputs (drafter side bf16; verify-side record fp32 at 1.98 MB/token), stored f16 q (0.5 MB/token), ~80 live drafted tokens | 0.15 | 0.15 |
| KV q8_0: all 4096 tokens at 4k / exact recent window of 4096 at 200k (34,816 B/token) | 0.143 | 0.143 |
| Drafter hot set (2048 tokens/layer) + page-mean-key index | — | 0.123 |
| KV staging: 2 of 4 KV heads of one layer (195,904 × 2176 B / 2) | — | 0.213 |
| Resident R slice | 0.33 | — |
| **Total** | **15.49** | **15.50** (0.00 spare: deficit-ladder step 1 is the default at 200k until free VRAM is measured) |

| RAM (22 GB) | 4k | 200k |
|---|---|---|
| R Q4_K (25.62e9 × 4.5/8) | 14.41 | 14.41 |
| KV q8_0 outside the VRAM window (195,904 × 34,816 B) | — | 6.82 |
| Host overhead [A] / pinned staging | 0.40 / 0.15 | 0.40 / 0.15 |
| Token embedding BF16: full (2.54) at 4k; top-10k hot rows at 200k, rest on NVMe | 2.54 | 0.10 |
| **Total** | **17.50** | **21.88** (0.12 spare) |

- **Context ceiling at full speed** is (22 − 14.41 − 0.65)/34,816 B + 4096 ≈ **203k tokens** [D]. Beyond that, the oldest KV pages spill to NVMe and are re-read every verify. Each GB adds about 1 GB of DRAM write traffic, about +4.5% verify time [D]. At 262k the spill is 2.05 GB and decode drops to ≈13.5 tok/s. If usable RAM turns out to be 1–2 GB short, 200k still works through the spill at −5 to −9%.
- **Pinned memory.** Pinning R + KV takes 21.2 GB. If WDDM caps GPU-visible host memory near 16 GiB [A], pin only the DMA-sourced rows plus the KV (≤17.0 GB with a CPU share ≥4.2 GB) and keep the CPU rows VirtualLock'ed. If the cap is lower still, a static CPU share costs about 3–8% at 200k [A].
- **VRAM surplus ladder** (the bench measures free VRAM after the CUDA context):
  0. Keep the monitor on the motherboard (iGPU) output so DWM holds no VRAM on the 5080. Your `nvidia-smi` already shows `display_active = Disabled`.
  1. Full-layer KV staging, +0.21 GB: 16.1 → 17.4 tok/s at 200k.
  2. More resident R at 4k: about +0.5 tok/s per 0.8 GB.
  3. A longer exact window.
- **VRAM deficit ladder:**
  1. Drop the drafter hot set and index, −0.12 GB (α −0.01 [A]).
  2. Window 4096 → 2048, −0.07 GB.
  3. Use the B lm_head for MTP instead of the 32k head, −0.09 GB (+2 ms per draft pass).
- **NVMe (SN850X):** B pack 13.6, R pack 14.4, BF16 embedding 2.5, KV spill area ≤3, prefix cache (about 11 GB per 200k session: KV 6.96 + GDN checkpoints 3.9; budget ≤100 GB).
- **980 PRO:** optional replica of the packs.
- **SATA:** BF16 sources, cold tier of the prefix cache.

#### 10.3.5 Prefill of a 200k prompt [D from A]

- Linear layers: 2 GEMMs (B and R) × 2 × 24.35e9 (no lm_head) × 2e5 = 1.95e16 int8 ops. At 180 TOPS (40% of 450) that is 108 s.
- Causal attention: 16 × 24 × 256 × 4 × T²/2 = 7.86e15 FLOP. At 55 TFLOPS (FP16 with FP32 accumulate, GeForce half rate, ~50% efficiency) that is 143 s.
- GDN: 3 s chunked, 7.7 s sequential.
- Overhead: about 5 s.
- **Total ≈ 265–300 s (range 185–320 s).** 180 TOPS MMQ is [A] and above typical llama.cpp MMQ efficiency; at ≈135 TOPS it is ≈300 s.
- Hidden behind compute: the per-chunk R stream (0.31 s vs 2.6 s of compute) and re-reading the earlier KV (340 GB over the whole prompt, 7.4 s total at 46 GB/s).
- VRAM during prefill (requires the pure-IQ4_XS B): B 13.61 + reserve 0.35 + GDN 0.16 + chunk activations 0.6 + per-layer R double buffer 0.45 + KV stream ring 0.2 = 15.37 GB. The drafter buffers are freed during prefill.
- Prefill is compute-bound, so nothing in this section speeds it up losslessly. A re-used context restores from the prefix cache: 7.1 GB at 6.5 GB/s ≈ **1.1 s**.

### 10.4 Qwen3.8-Flash-Next

#### 10.4.1 Model and numbers

- Expert sizes [V]. `llama-quant.cpp:410/416`: rows of 640 are not a multiple of 256, so IQ4_XS falls back to IQ4_NL and Q4_K to Q5_0 for down_proj. One B expert is therefore 2.662 MB (store 65.4 GB) and one R expert is 2.970 MB.
- Non-expert weights [D from config]. 36 GDN × 57.9M + 12 attention × 51.4M + 48 × (router + shared expert 6.2M + hyper-connections 13.2M) + PLE 32.8M + lm_head 636M = 4.30B, excluding the token embedding. That is **4.57 GB at Q8_0**. The 3.84 GB used in one proposal was wrong.
- Defined target, the same for v0 and this plan: non-expert Q8_0; experts IQ4_XS (down IQ4_NL); a static hot set of **1.0 GB of R (337 experts at ≈Q8)** resident in VRAM, chosen by the planner. Each extra GB of R costs about 376 cache slots (−2 to −3% tok/s).
- Per-token time [D/A]: `t = max(t_io, t_chain) + 0.25·min(t_io, t_chain)`, where `t_io = 480·(1−h)·2.662 MB·1.05 / (BW·u)`.
  - t_chain = 27 ms [A]: 48 layers × (0.12 non-expert GPU + 0.05 router + 0.17 CPU GEMVs for RAM hits + 0.10 two handoffs + 0.05 HC) + lm_head and sampling.
  - At 200k, t_chain is +1.5 ms (indexer scan of 154 MB + 12 zero-copy gathers of 2.2 MB each).
  - Under uniform routing, h equals the cached share.
  - Drive utilization u is 0.80 for v0 and 0.85 for this plan [A]. With equal u the plan gives 13.1 instead of 13.8 at T2.
  - [A] The 0.25 overlap term needs 1-layer router lookahead (router l+1 applied to the layer-l hidden state) with high recall, so layer l's misses are issued before its router finishes. Without it, t ≈ t_io + t_chain: T2 10.8/10.1 (plan) vs 10.1/9.4 (v0). The ratio is unchanged; absolute numbers fall ~22%.

#### 10.4.2 Mechanisms (all lossless: they change where bytes live and when they move, never which experts are computed)

| # | Mechanism | Effect [D] |
|---|---|---|
| F1 | **Replicated expert store.** The 65.4 GB B store sits on both NVMe drives. The IOCP scheduler sends each miss to the drive with the earliest expected completion, demand misses jump the queue, and gate/up is issued before down. The SATA mirrors serve only low-priority prefetch: a 2.66 MB read takes 4.7 ms on SATA, longer than a layer. | u 0.80 → 0.85 [A]; balances unequal drives; SATA adds about +0.5 GB/s only in T1 |
| F2 | **QSA KV in pinned RAM** (13,056 B/token, 2.61 GB at 200k). Pooled indexer keys stay in VRAM (0.154 GB). Each layer gathers the 2048 selected tokens zero-copy (2.2 MB). There is a 0.1 GB VRAM hot-block cache. | +1.5 ms/token at 200k; frees 2.5 GB of VRAM for experts. Exact: gather-then-dense equals the masked reference. |
| F3 | **Token embedding to NVMe** (0.05 GB of hot rows stay in RAM) and **n-gram table BF16 to the SN850X** (102.4 GB) | +460 RAM slots; n-gram prefill 6.4 s instead of 17.8 s |
| F4 | **Prompt-seeded residency.** Prefill routing histograms seed the VRAM and RAM tiers by marginal hit gain, and are stored with the prefix cache. | 0 under uniform routing; about +0.03 h under skew [A] |
| F5 | **Layer-major prefill** in 100k mega-chunks, using a custom block-sparse QSA kernel, a batched top-k (ggml `top_k` uses CUB DeviceTopK per row only with CCCL ≥ 3.4.3; CUDA 13.0 ships CCCL 3.0, so it is a full per-row argsort [V `top-k.cu`]: vendor CCCL ≥ 3.4.3 or write a batched radix select; add ~0.5–1 ms/token at 200k decode until measured) and the fp32 HC residual (4.1 GB VRAM). The hot-set R is applied, as in decode. | 200k prefill ≈40 s; stock sparse-FA path 45–70 s [A, measure] |
| F6 | **Prefix cache** (KV pages + indexer keys + GDN/conv/PLE checkpoints every 8k; about 5.7 GB per 200k session) | restore ≈0.4 s plus background re-warm |
| F7 | **I/O-gated speculation**, off by default. It only pays if a·(1+α_mtp) > 2−ρ given the expert overlap ρ, i.e. once I/O per token falls below about 30 ms. | 0 at h≈0.45 |

#### 10.4.3 Results by drive topology (central) [D]

| tok/s at 4k / 200k | Current design (v0) | This plan | Plan / v0 on same drives | Plan / v0 on one drive |
|---|---|---|---|---|
| T1: both NVMe on CPU lanes (+SATA prefetch), 13.5 GB/s | 13.1 / 12.2 | 14.3 / 13.3 | 1.09x | 2.07x |
| **T2: one CPU-attached + one chipset NVMe, 13.0 GB/s (likely; M.2_1 is always CPU-attached)** | 12.7 / 11.8 | **13.8 / 12.8** | 1.09x | **2.0x** |
| T3: both behind the chipset x4 uplink, 6.8 GB/s | 6.9 / 6.4 | 7.6 / 7.0 | 1.09x | 1.09x |
| Skewed routing (h = cached share + 0.27), T2 | 22.7 / 20.1 | 27.8 / 24.1 | 1.23x / 1.20x | 2.2x |

- Cached share [D]. v0: 0.441 at 4k, 0.399 at 200k (KV in VRAM, embedding in RAM). This plan: 0.459 at 4k (3,381 VRAM + 7,906 RAM experts) and 0.416 at 200k (3,307 + 6,925).
- Under the task's own optimistic assumptions (h 0.5, 2.61 MB, 7 GB/s, u 0.75–1, no chain penalty), the single-drive v0 is 8.4–11.2 tok/s. The ratios above all come from one model.

#### 10.4.4 The lossless 2x on this box: 64 GB RAM (hardware option)

2×32 GB DDR5-5600 gives about 54 GB usable, so the cached share rises to 0.95 at 4k and 0.91 at 200k. The chain rises to 34–40 ms (18.8 ms GPU/sync plus 1.04 GB of RAM-hit expert reads per token at 50–68 GB/s): **24–28 / 23–27 tok/s (T2), 23–27 / 22–25 (T3)**, which is 1.9–2.2x of the striped v0. Use **2 × 32 GB** DIMMs: 4 DIMMs (2 per channel) lower the DDR5 speed and slow the DRAM-bound 27B. It is lossless: same model, more cache. For the 27B it gives no speed (that model is DRAM-bandwidth-bound), but it removes the 203k RAM ceiling and the pinned-memory squeeze.

#### 10.4.5 Memory plan at 200k [D] (4k differences in brackets)

| VRAM (15.5) | GB | RAM (22) | GB |
|---|---|---|---|
| Reserve (context/modules/pool) | 0.35 | Host overhead | 0.40 |
| Non-expert weights Q8_0 | 4.57 | Pinned I/O staging | 0.50 |
| GDN + conv + PLE conv state | 0.12 | KV q8_0, pinned [4k: 0.05 in VRAM] | 2.61 |
| Buffers + miss staging | 0.40 | Embedding hot rows | 0.05 |
| Hot-set R (target definition) | 1.00 | Expert B cache: 6,925 experts [4k: 7,906] | 18.44 [21.05] |
| Pooled indexer keys f16 | 0.154 | | |
| KV hot-block cache | 0.10 | | |
| Expert B cache: 3,307 experts [4k: 3,381] | 8.81 [9.00] | | |
| **Total** | **15.50** | **Total** | **22.00** |

- **NVMe SN850X:** expert B store 65.4, non-expert weights and packs about 6, n-gram BF16 102.4, token embedding 1.27, prefix cache ≤100. About 280 GB of 1 TB.
- **980 PRO:** expert B replica 65.4.
- **SATA:** sources, n-gram master copy, cold prefix tier.
- The full 262k context also fits: KV 3.42 GB, about 300 fewer cached experts.

#### 10.4.6 Prefill of a 200k prompt [D from A]

- Matmuls: 2 × 6.26B active × 2e5 = 2.5e15 ops at 160 TOPS = 15.7 s.
- Indexer: 6.1e13 FLOP plus the batched top-k, 3.5 s.
- Block-sparse attention: 1.2e14 FLOP with 5.35 TB of gathers (about half reused in L2), ≈3.5 s.
- GDN: 2–6 s.
- Overhead: about 5 s.
- Expert I/O: 2 × 44 GB at 11 GB/s = 8 s, overlapped. On T3 it is 16 s.
- **Total ≈ 40 s (30–60 s).**
- Stock llama.cpp at 836d571 already runs QSA as union-sparse FlashAttention (`ggml_flash_attn_ext_set_n_kv_max`, 8-query tiles), not dense FA plus mask. Estimated 45–70 s for 200k, limited by the n_kv × n_ubatch f16 mask, f16 conversion of the q8_0 KV, and up to 8x union overfetch. Measure it at M4 before writing the custom block-sparse kernel, which is an optimization (to ~40 s), not a requirement.

### 10.5 Lossy and target-changing options (opt-in, off by default, never in the headline)

| Option | Speed effect [D/A] | Expected quality impact [A] | Measurement gate |
|---|---|---|---|
| 27B: KV older than the last 16k tokens at q4_0 | 200k: 16.1 → **19.3** tok/s; RAM ceiling → about 360k | KLD 0.01–0.03 vs the q8_0-KV target; long-range retrieval at risk | KLD and top-1 on 100–200k documents; RULER and needle at 64k/128k/200k; ship only if KLD ≤ 0.01 and retrieval is within noise |
| 27B: bounded V-skip (drop V pages with total attention mass ≤ 1e-3 per query) | ≤ +6% at 200k if 50% of pages are skippable; adds a K-then-V round trip | error ≤ ε·max\|v\|, below q8_0 rounding; near-lossless, not exact | attention-mass sparsity per head at 200k; KLD |
| 27B: prefill attention with FP16 accumulate | prefill ≈265 → ≈195 s | accumulation error in 256-long dot products | KLD of the resulting KV and logits |
| Flash: missed (cold) experts read as IQ3_XXS | T2 13.8 → 18.8 (4k), 12.8 → 17.5 (200k); T3 7.6 → 10.5 | about 1.7x the IQ4_XS weight RMSE on about 54% of expert uses; KLD +0.005–0.02 | KLD vs the defined target over ≥1M tokens ≤ 0.005; per-layer sensitivity |
| Flash: skip routed experts ranked 9–10 when they miss | T2 13.8 → 16.9; T3 7.6 → 9.3 | changes routing; depends on the gate mass of ranks 9–10 (unmeasured) | gate-mass histograms; KLD ≤ 0.005; MMLU/GSM8K/code deltas |
| 27B: residual Q3_K instead of Q4_K | R 14.41 → 11.01 GB; T_v 215 → 165 ms (4k), 351 → 301 ms (200k); ~26.4 → 29.2 tok/s at 4k, 15.9 → 18.2 at 200k [D] | target moves from 0.89x to 1.9x Q8_0 weight RMSE [M], between Q8_0 and Q6_K; KLD unmeasured | adopt only if KLD(B+R_Q3 vs BF16) ≤ KLD(Q6_K vs BF16) and benchmarks are within noise |
| Target choice, not lossy against the bar: imatrix-calibrated B, R rebuilt against it | α +0.015 [A] → 4k 29.2 tok/s | changes the reference B+R | adopt at M3 only if KLD(B+R vs BF16) ≤ KLD(Q8_0 vs BF16) |
| Free and lossless, uncredited: the drafter also uses the resident R slice; suffix/prompt-lookup drafts | α ↑ (unmeasured); large gains on repetitive text | none | α with/without |

### 10.6 Review outcomes: corrections adopted and ideas dropped

| Disputed item | Proposer | Reviewer | Resolved here |
|---|---|---|---|
| 27B verify at 4k | 192–210 ms | 215–239 ms | **215 ms at N=12** (serial CPU-idle S = 19 ms modelled; cap 68). The reviewers were right that S is not overlapped. |
| 27B verify at 200k | 287–337 ms | 324–431 ms | **351 ms** with half-layer staging and a custom q8_0 kernel; 327 ms with full staging. The 431 ms figure assumed stock FA (an f16 temporary per head chunk), which this plan does not use. |
| Baseline | 14.3–14.5 tok/s | 12–13.9 | Report both: 14 as documented, **13.6** same-assumption (the §4 250 ms verify contradicts the CPU compute hint). |
| Multi-branch pre-drafting, M ≤ 16 | ×1.37 | +4% | Replaced by ≤3 single-root hedges: **+8%** in this model, go/no-go at M3d. |
| MTP acceptance | 0.80/0.70/0.60 | 0.70–0.75/… | 0.75/0.60/0.50 central; weak MTP gives 25.6 tok/s. |
| Flash non-expert size | 3.84 GB (overlap proposal) | — | **4.57 GB** [D from config]. |
| Half-expert split | 1.3 + 1.3 MB | 1.74 + 0.87 MB | Reviewer right; dropped (helps latency only). |
| QSA prefill gather traffic | — | 5.3e15 B (longctx review) | **5.35e12 B** (the bytes review was right; the other is off by 1000x). The stock path is union-sparse FA (45–70 s [A]), so the custom kernel is an optimization. |
| Flash u / η credit | 0.90–0.95 vs 0.75 | equal | 0.85 vs 0.80 [A], with equal-u numbers shown (13.1 vs 13.8). |
| Prompt-seeded residency | ×1.32 | ×1.05–1.1 | Credited only under skew. |
| "Bit-identical" dual path | claimed | Q8_K vs q8_1 differ | The CPU kernel uses q8_1 activations, so results are lossless up to fp32 order; KLD-gated. |
| Tree E(24) below chain | — | builder bug | A chain is a special tree; the builder must fall back. Tree verify deferred. |
| 200k "2.7x / 5.5x" | vs an infeasible v0 | strawman | Compare to 14 and to the 200k-capable v0 (9.4). |
| Prefill | 189–247 s / 22–31 s | 230–300 s / 30–50 s | **265 s / 40 s**. |

Dropped:
- Repair chains: ≤2% gain.
- 2–3-layer lookahead credit: unvalidated, and wasted prefetches spend the scarce NVMe bandwidth. (The 1-layer lookahead is required and assumed; its recall is measured in 10.8 item 10.)
- Entropy coding of B and R: measured 97–98% of raw [M], so ≤3% gain.
- CPU-side attention over the RAM KV: about 29 GFLOP per layer for N=12, about 19 ms per layer on the CPU.
- Layer-skipping drafters.
- Expert-union speculation for Flash at h≈0.45.
- 0.2 GB of resident R when VRAM is full.
- Deterministic tree nodes under recursive rejection sampling: biased unless roots are matched against the target's own sample, as L4 does.

### 10.7 Milestones (fit into §7)

| # | Adds for §10 | Done when |
|---|---|---|
| M1 (now) | Bench lines in 10.8 | Every [A] in 10.2 has a number from your box |
| M2 qwen35 graph | Custom split-KV q8_0 attention with LSE (decode N≤24 and 2048-query prefill); KV pages in pinned RAM with the VRAM recent window; chunked prefill (C=2048, R streamed); chunked GDN prefill; MTP layer graph (B-only) | Real 27B GGUF PPL matches llama.cpp at 4k/32k; the attention kernel matches ggml FA with KLD ≤1e-5 at 32k; a 200k prompt runs in B-only mode |
| M3a pack + serial verify | Residual pack; B+R verify on the §8 DMA path as the reference; speculative sampling with stored q; commit replay from the verify's own fp32 inputs | KLD(B+R vs BF16) ≤ KLD(Q8_0); α per position, B rank histogram, detector calibration, MTP acceptance vs B measured |
| M3b dual-path verify | VNNI Q4_K×q8_1 kernel (native layout, row ranges); own verify scheduler; chunked DMA; pin only the DMA rows | KLD vs the M3a reference ≤1e-5; T_v(4k, N=12) ≤ 230 ms |
| M3c MTP + pipelining | MTP-staged drafter; two streams with priorities; leaf continuation with the bonus-position test | ≥24 tok/s at 4k at the measured α |
| M3d hedges | Rootless-replay hypotheses (n_seqs, identity padding); ≤3 hedges | Keep if ≥ +5% measured, else remove |
| M3e 200k | Drafter sparse view; NVMe KV spill; prefix cache | 200k decode ≥15 tok/s; needle/RULER at 128k/200k equal to the q8_0-KV reference; restore ≤2 s |
| M4 qwen4exp graph | Block-sparse QSA prefill kernel; batched top-k; zero-copy KV gather; layer-major prefill | Tiny-model parity; 200k prefill ≤60 s |
| M5 expert streaming | Replicated store with least-loaded IOCP and demand priority; prompt-seeded residency; R hot-set planner; routing traces; F7 gate experiment | tok/s and h from traces; T2 ≥13 tok/s at 4k |
| M6 server | Per-session prefix cache; 200k defaults; per-request speed and acceptance stats; lossy options behind explicit flags | OpenAI client with a 200k prompt; restore ≤2 s |

### 10.8 What `shoehorn bench` must add

1. **Pinned memory.** `--pinned-gb 22` with cudaHostAlloc and with cudaHostRegister + VirtualLock: the WDDM ceiling and seconds per GB.
2. **H1 combined DRAM.** 7 cores streaming AVX-512 reads (and later running the Q4_K GEMM) while H2D DMA reads another pinned buffer, at 2/4/32/64 MB chunk sizes. This decides cap 60 vs 68–75 GB/s.
3. **Small copies behind bulk DMA.** Latency of 225 KB D2H/H2D copies while 64 MB vs 4 MB H2D chunks stream.
4. **Handoff round trip under WDDM.** Event sync vs spin on a host-mapped flag vs `cuStreamWaitValue32` on host-mapped memory, with and without a low-priority kernel stream; HAGS on and off.
5. **CPU Q4_K GEMM** for N = 1, 2, 4, 8, 12, 16, 20, 24 on 7 and 8 threads, sustained for 60 s. Stock repack now; shoehorn VNNI kernel at M3b.
6. **GPU IQ4_XS / Q4_K matmul** for N up to 64 (add 24, 32, 48, 64), alone and while the copy engine writes about 46 GB/s into VRAM.
7. **Free VRAM** after the CUDA context with ggml-cuda loaded (lazy module loading on/off), in decimal GB. Allocate up to free − 0.1 GB with the sysmem fallback policy on and off and confirm nothing is demoted to system memory.
8. **Attention and prefill throughput.** Stock ggml FA with q8_0 KV at 12–24 queries and 32k–200k context as the reference; the split-KV kernel once it exists; 2048-query prefill attention TFLOPS; int8 MMQ TOPS at batch 2048.
9. **Disks.** All drives at once (exists) plus which drive is chipset-attached; per-drive 10-minute sustained 2.66 MB reads (throttling); random 4 KiB IOPS at QD 64–256; NVMe reads concurrent with H2D DMA; free space per drive (the plan needs ≈410 GB on the SN850X for 27B + Flash data and 65.4 GB on the 980 PRO; if short, keep the n-gram table on SATA, which costs nothing in decode and 17.8 s of 200k prefill).
10. **Model instrumentation (M3/M5).**
    - 27B: end-to-end drafter pass (16 columns + rootless replay + MTP) and verify-kernel latency while draft passes run; α by position and context; target rank in B's top-4; detector recall/FP; hedge coverage; MTP chained acceptance fed B hidden states; drafter α loss with the sparse view at 128k/200k.
    - Flash: pooled indexer-key precision must match between prefill and decode (reference pools in bf16; top-k near-ties can flip). Hit rate vs capacity, per-layer skew, prompt-vs-generation histogram overlap, l+1 lookahead precision/recall, consecutive-token expert overlap ρ.

### 10.9 Re-plan triggers

- **H1 false (cap ≤ 62 GB/s):** 4k → about 25 tok/s. Keep the CPU slice; without it the result is 20.7.
- **Pinned ceiling below about 17 GB:** use a static CPU share at 200k (−3 to −8%), or 64 GB RAM.
- **Handoff above 100 µs:** give the CPU rows only in FFN gate/up/down (128 joins instead of 257).
- **MTP acceptance below 0.6/0.45/0.35:** expect about 24 tok/s at 4k; the B-only drafter with pipelining gives 18.
- **α below 0.87:** 4k drops below 24 tok/s. Evaluate the imatrix B under the M3 KLD gate.
- **Free VRAM below plan:** apply the deficit ladder (10.3.4).
- **Drives on T3:** move one NVMe to the CPU-attached M.2_1 slot (free), or add RAM (10.4.4).