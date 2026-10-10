# Software-only speedups for the 27B (plan, 2026-10-10)

No hardware changes. Every idea here is lossless: the output distribution stays B+R. Starting point, measured on the
RTX 5080 (see [DECODE-LIMITS.md](DECODE-LIMITS.md)):

- One cycle is a **verify** of about 255 ms plus **drafting** of 300-400 ms.
  - The verify is PCIe-bound: 14.4 GB of residual at about 54 GB/s.
  - Drafting is base passes of 19.5 ms each.
- Tokens per cycle: prose 15-17, code 26-50.

Already tried, and not repeated here: residual prefetch, lossless compression, a custom IQ4_XS GEMV, tree drafts
drafted one branch at a time, and pipelined drafting.

Pipelining was neutral for one reason: the verify slowed from 257 to 298 ms when drafting shared the GPU (commit
931444a). Idea 1 targets that cause.

| # | Idea | Where it helps | Expected | Gate before building |
|---|---|---|---|---|
| 1 | Zero-copy streaming GEMM for the residual | verify, and pipelining | +20% prose, +35% code with pipelining (MULTISEQ phase 2 estimate) | **passed**: 53.7 GB/s from 8-16 small CTAs, drafter -0.5% |
| 2 | Self-distilled MTP drafter | prose drafting | prose +10-20% | MTP acceptance gain on held-out prompts |
| 3 | Fused base pass (megakernel-lite) | every base pass | prose +5-9%, code +3-5% | ≥2 ms of non-GEMV time per pass in nsys |
| 4 | CPU slice of the residual (AVX-512 VNNI) | verify | verify -15-25% at N≤24 | ≥25 GB/s of R at N=16 on the CPU |
| 5 | Batched hedge drafting during the verify | tokens per cycle | explain +10-20%, prose +5-9% (from tree v1 numbers) | needs 1; tree v1 data in MULTISEQ.md |

Order: 1, then 2, 3, 4, 5. Idea 2 needs no CUDA work and can run in parallel with 1.

## 1. Zero-copy streaming GEMM

**Idea.** The residual never lands in VRAM. A custom CUDA kernel reads Q4_K straight from pinned, mapped RAM over PCIe
and multiplies it against the N≤64 verify rows. It uses about 12-16 persistent CTAs and leaves the rest of the GPU free.

**Why.** Today the copy engine stages R into two 71.5 MB VRAM slots (`patches/ggml-weight-prefetch.patch`), and MMQ
then reads it back. The new kernel removes:
- 28.8 GB of extra VRAM traffic per verify;
- 410 graph splits, each with a host sync point;
- the contention that slowed the verify while drafting ran beside it.

It also gives back:
- one GPU split for the whole verify, so it can be captured as a CUDA graph;
- 143 MB of VRAM, usable for resident R (`--res-gpu-gb`) or KV.

**Steps.**
1. **Bench**: done 2026-10-10, gate **passed** ([`experiments/zerocopy`](../experiments/zerocopy/README.md)).
   - 8-16 CTAs of 256 threads read mapped pinned RAM at 53.7 GB/s.
   - Next to them, a VRAM-bound stand-in for the drafter slows by 0.3-0.5%. Beside the copy engine it slows by 7.6%.
   - Design rule: use small CTAs. CTAs of 1024 threads hold SM slots and slow the reader by 10-47%.
   - Caveat: raw zero-copy is 7% below the copy engine (57.9 GB/s), so a verify on its own gets no faster. The gain
     has to come from overlap (step 4) and from removing the staging and split syncs.
2. **Zero-kernel prototype.** Expose `rbuf_`'s mapped pointer as a CUDA-visible buffer, so ggml's MMQ/MMVQ reads R
   over PCIe with no staging.
   - Wire it in `Qwen35::mm` (`src/model/qwen35.cpp:1588`) and at residual load (`qwen35.cpp:~1560`), behind
     `E8_ZERO_COPY=1`.
   - Prefill keeps the slot staging.
   - Measure the verify alone.
   - **Done 2026-10-10** (patch to `ggml-cuda.cu`: mapped allocation mode; `E8_ZERO_COPY=1`). Code prompt, greedy,
     `--spec 16`: same tokens per cycle (15.24); verify **418 ms against 278 ms staged**; 27.1 against 34.7 tok/s.
     PCIe only streams during the residual matmuls, at about 34 GB/s. Staging prefetches the next split while
     attention, GDN and B run. So step 3 must keep the stream running across the whole verify. Off by default.
3. **Next (chosen 2026-10-10): SM-driven staging.** Keep the two VRAM slots and the prefetch, but fill them with a
   copy kernel limited to 8-16 blocks of 256 threads, reading mapped pinned RAM, in place of `cudaMemcpyAsync`.
   - The patch already sends transfers up to 4 MB through a copy kernel, so this mostly extends that path.
   - Then rerun `gen --pipe` to test the hypothesis directly. Does the verify still slow from 257 to 298 ms when
     drafting runs beside it?
   - If yes, the cause is SM contention, not the copy engine: drop idea 1 and move to idea 2.
   - If no, keep pipelining on and continue with the full streaming kernel below only if it still pays.

   - **Result 2026-10-10: hypothesis rejected.** The change is `E8_SM_STAGE=G` in the ggml-cuda async upload. Prose,
     temp 1.0, 3 seeds, verify per cycle:

     | | without pipelining | with pipelining |
     |---|---|---|
     | copy engine | 276 ms | 302 ms |
     | 16-block SM copy | 299 ms | 322 ms |

     So the verify slows by the same ~25 ms with either copier. The cause is the drafter's kernels competing with the
     verify's compute for SMs, not the copy engine. Idea 1 is dropped; next is idea 2.

   **Later (dropped with idea 1): custom kernel** `src/kernels/stream_q4k.cu`, a persistent Q4_K × q8_1 GEMM.
   - int8 mma; the N≤64 activation rows held in shared memory; each weight byte read once; grid limited to G CTAs.
   - Add it as a GGML op through the ggml patch mechanism (`patches/`).
   - Dispatch it from `Qwen35::mm` for small-batch verifies.
   - Enable CUDA in our own CMake target, guarded by `SHOEHORN_CUDA`.
4. **Turn pipelining back on** (`gen --pipe`, the shadow model) and confirm the verify no longer slows while drafting.
   Then make it the default if tok/s improves.

## 2. Self-distilled MTP drafter

**Idea.** Fine-tune only the MTP layer, about 0.35B parameters with the trunk frozen, so that it predicts the
**IQ4_XS base's** distribution from **base** hidden states. Train it EAGLE-style over 3 chained steps.

**Why.** Prose is limited by MTP acceptance (42-44%, against 77% on code). The MTP head was trained on BF16 hidden
states against the data distribution. Here it receives IQ4_XS hidden states, and B judges it. A drafter change can
never change the output.

**Steps.**
1. **Data.** Add `gen --dump-mtp-data`, which writes base hidden states, next tokens and B's top-64 logits, taken
   from base passes on the user's prompts and `bench/prompts`.
   - About 20-50M tokens. Store hidden states in fp16 on NVMe.
2. **Trainer.** A PyTorch script under `experiments/mtp_distill/`.
   - Loss: KL(B ‖ MTP) over steps 1-3, each step fed the previous step's own output (training-time test).
   - Optimizer: bf16 with 8-bit Adam, about 3 GB of VRAM. Run it with the server stopped.
3. **Export.** Write the tuned layer back into the base GGUF as a new pack, so the original is untouched.
4. **Measure.**
   - MTP acceptance per step and tok/s with `gen --repeat 6` on held-out prompts.
   - Greedy output must stay identical to plain decoding.

## 3. Fused base pass (megakernel-lite)

**Idea.** Remove launch and small-kernel time from the base pass. A pass takes 19.5 ms against a floor of about
16 ms (13.9 GB at 872 GB/s).

**Steps.**
1. **Profile** one base pass with Nsight Systems (`gen --profile` under `nsys`). Split the time into GEMV and
   everything else. **Stop if everything else is under 2 ms.**
2. **Fuse the biggest non-GEMV chains** first, most likely:
   - RMSNorm into the q8_1 activation quantize that feeds the GEMV;
   - the GDN gate/beta/conv elementwise ops;
   - the residual adds.

   Do it as ggml-cuda patches in `patches/`.
3. Re-check: the drafts must be greedy-identical and the per-pass time must fall. The reference is llama.cpp's
   18.9 ms on the same file.

## 4. CPU slice of the residual

**Idea.** An AVX-512 VNNI Q4_K × q8_1 kernel on the Zen 5 cores multiplies part of R in place. The GPU streams the
rest at the same time.

**Why.** Measured (`bench4.txt`): CPU 43.9 + H2D 36.5 = **80.4 GB/s** read from RAM together, against 57.6 GB/s for
DMA alone. The old CPU kernel, `src/kernels/q4k_small.cpp`, was too slow (275-900 ms for all of R), so the GPU took
everything (commit 1dc7deb).

**Steps.**
1. **Kernel.** A new VNNI kernel that reads native Q4_K and uses q8_1 activations (the same as the GPU, so results
   differ only in summation order).
   - Bench at N = 1-24, 7-8 threads, sustained for 60 s.
   - **Gate:** ≥25 GB/s of R at N=16.
2. **Split by rows** per matmul. The CPU share is picked from N and the measured rates.
   - Joins go through mapped flags or events; check WDDM handoff latency first.
3. **Measure.**
   - Verify time at N = 8, 16, 24, 48.
   - KLD of the logits ≤1e-5 against the GPU-only path.

## 5. Batched hedge drafting during the verify

**Idea.** While the verify runs (once idea 1 frees the GPU), draft continuations from the two or three least
confident draft positions as well as from "all accepted". Draft every branch in the **same** batched base passes,
reusing `draft_lockstep` from multi-slot decoding. The next cycle then starts with drafts ready on most rejections,
not only on full accepts.

**Why.** Tree drafts v1 already showed more tokens per cycle (prose 17.4 -> 19.0, explain 20.0 -> 24.4) under exact
recursive rejection sampling. They were slower only because branches were drafted one after another (draft time
396 -> 671 ms). Batched and overlapped with the verify, that cost mostly disappears.

**Steps.**
1. Drive tree branches through `draft_lockstep`, one working recurrent state per branch.
2. Choose branch points where rejections actually happen. MTP disagreement predicts the verifier's rejections
   (commit d6013df); base confidence alone predicts them poorly (only 18% hits).
3. Measure tokens per cycle and tok/s with `E8_TREE` on and off, at 6 seeds.

## Verification for every idea

- **Correctness:**
  - logits KLD ≤1e-5 against the current path (`shoehorn logits`, `ppl`);
  - greedy output identical with `--spec 16`;
  - for sampled output, the existing exactness tests.
- **Speed:**
  - `gen --repeat 6` on prose, explain, code and reasoning at temp 1.0, before and after;
  - the dashboard Speed check;
  - the `--timing-log` verify/draft breakdown.
- **Regression:**
  - `tests/server_smoke.ps1` and `tests/concurrent_smoke.ps1`;
  - the `tests/agent_reliability` hard set (34/34).
