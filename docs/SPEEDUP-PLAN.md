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

## 2. Self-distilled MTP drafter (in progress, resume here)

**Why this is the one left for prose.** Measured through the server (prose, temp 1.0, top_k 20, top_p 0.95):
- drafting is 60% of each cycle (404 ms against 259 ms of verify);
- the MTP head's proposals are kept 41.6% of the time.

Running the head at ~8 bits (base + its residual, `E8_MTP_RES`, removed again) only lifted that to 44.8%, and the
heavier head made drafting 5% slower: 21.95 -> 21.43 tok/s. So precision is not the gap: what the head predicts
is. It was trained on BF16 hidden states to predict the data; here it reads the 4-bit base's hidden states and is
judged by the base's distribution.

**Done (2026-10-10, not committed):**
- `shoehorn mtpdump <base.gguf> --in conv.jsonl --out data.bin [--topk 32] [--batch 512] [--ctx 8192]`
  (`src/cli/mtpdump.cpp`, `Qwen35::hidden_rows`).
  - Input: each JSONL line is {"text": conversation in Qwen chat format}.
  - Process: the base alone reads each conversation.
  - Output per position: the token, the trunk's final normed hidden state (f16, what the MTP layer reads) and the
    base's top-k logits. Format in the file header comment.
  - Built in `build-zc/`; not run yet.
- `experiments/mtp_distill/make_corpus.py out.jsonl N [seed]`: asks the server (B+R) for answers to mixed prompts
  (stories, letters, descriptions, explanations; 25% with thinking on). The answers are the target's own text, which
  is what the drafter sees at inference.
- Pilot corpus: `experiments/mtp_distill/pilot.jsonl` (40 conversations).

**Next steps:**
1. **Dump the pilot.** Run `build-zc/bin/shoehorn.exe mtpdump E:/shoehorn/orca27b.base.gguf --in
   experiments/mtp_distill/pilot.jsonl --out <data>/pilot.bin`, with the server unloaded (it needs the GPU). Check
   that the hidden values are finite and that the top-1 logit's token matches the next token often (sanity).
2. **Full corpus.** Generate 300-500 conversations (about 150-250K tokens, 2-3 h through the server, in the
   background), then dump them. About 10.5 KB per token, 2-3 GB in total.
3. **PyTorch** (cu128 wheels for Blackwell) in `experiments/mtp_distill/.venv` (gitignored).
4. **MTP layer in PyTorch.** Use `third_party/llama.cpp/gguf-py` (`dequantize` works for IQ4_XS and Q4_K). Start
   from base + residual of `blk.64.*`, the ~8-bit head. Match `Qwen35::mtp_input` / `mtp_layer` exactly:
   - x = eh_proj([rms(emb(x_{t+1})) * enorm | rms(h_t) * hnorm]); emb is `token_embd` (BF16), and the concat order is
     embedding first.
   - Gated attention: q proj gives [Q | gate] per head (hd 256, 24 heads, 4 KV heads); q_norm / k_norm per head; IMROPE
     on the first 64 dims, freq base 1e7, sections [11, 11, 10, 0] (all equal positions for text); causal over the
     MTP's own K/V of earlier positions; out = wo(attn * sigmoid(gate)).
   - SwiGLU FFN with post_attention_norm; then hn = rms(.) * shared_head_norm; logits = output.weight (IQ4_XS,
     frozen; dequantize to bf16, 2.5 GB) @ hn.
   - Verify the port first: its logits should match `mtp_step` on the same inputs (top-1 equal, KLD near 0).
5. **Train** only the MTP layer (about 0.37B params), with the trunk and head frozen:
   - loss = KL(base's sampler distribution (top_k 20 / top_p 0.95 of the stored top-32, at temp 1) at t + 1 ||
     MTP softmax at (h_t, x_{t+1}));
   - bf16 with 8-bit Adam, about 8-10 GB of VRAM, with the server unloaded;
   - later, chained steps 2-3 (feed the head's own `hn` as the next h), as the drafter does.
6. **Export.** Quantize the trained layer back to IQ4_XS (or Q8_0 if VRAM allows: +0.2 GB, at the cost seen above)
   and write a copy of the base GGUF with only `blk.64.*` replaced. Never overwrite the original.
7. **Measure through the server.** The same 8 prose requests as the A/B above, comparing MTP kept and tok/s. Then
   check that greedy output is unchanged with `--spec 16` (the drafter must never change output).

**Gate:** MTP kept 41.6% -> at least 50% on held-out prompts, and prose tok/s up at least 8%.

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

**Result 2026-10-10: gate failed, dropped.** ggml's repacked Q4_K kernels (`bench --only cpu`, 16 threads)
already reach about 1.25 TMAC/s, a quarter of peak, and are compute-bound from N = 24 on:

| N | 16 | 24 | 32 | 48 | 64 |
|---|---|---|---|---|---|
| GB/s of R | 36.6 | 22.8 | 20.2 | 14.0 | 10.7 |

With the CPU reading, DMA falls to 36.5 GB/s, so a CPU slice pays only above about 21 GB/s. Verifies run at
N >= 25 (`choose_k` drafts at least 24), so the best case is about +3% at N = 24 and a loss beyond it. The existing
`q4k_small` kernel is slower still (20 GB/s at 16 columns).

**Steps (not done).**
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

**Result 2026-10-10, a smaller variant: dropped.** `E8_DRAFT_ALT=1` (off by default) tests the first MTP
proposal's runner-up in the same base pass, as a second sequence, using exact recursive rejection sampling. Results
over 6 seeds at temp 1.0 with `--spec auto`, 384 tokens:

| | Alternatives kept | Tokens/cycle | Draft ms/cycle | tok/s |
|---|---|---|---|---|
| prose | 197 of 558 | 16.33 -> 16.99 | 762 -> 848 | 14.80 -> 14.29 |
| explain | 91 of 188 | 21.05 -> 21.78 | 638 -> 681 | 21.37 -> 21.22 |
| code | 66 of 140 | 23.52 -> 24.74 | 567 -> 637 | 25.72 -> 25.05 |

Draft rounds fall about 10%, but each two-sequence base pass costs about 20% more than a single one: the tree eval
copies the recurrent state per sequence. One row alone costs much less (n = 1 -> 7 is +12%). Branching only pays
once a multi-sequence base pass costs about as much as one sequence with the same number of rows.

**Steps (full version, not done).**
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

## 6. Coupled (Gumbel-max) sampling to reuse drafts after a rejection (prose)

**Idea.** Sample every token as argmax(log p + Gumbel noise keyed by (seed, position, token)), in the drafts and the
verify alike. This is exact, and a position's noise is the same whatever its prefix. After a rejection, the discarded
drafts, and the verify's own picks under the old prefix, might come back at the same positions and be re-proposed
almost for free.

**Result 2026-10-10: gate failed, dropped.** Measured through the server (`coupled_sampling` per request), on 4 prose
prompts x 2 seeds, 500 tokens, thinking off, temp 1.0 / top_k 20 / top_p 0.95:

| | Standard | Coupled |
|---|---|---|
| tok/s | 21.2 | 19.2 |
| tokens/cycle | 14.3 | 13.0 |
| verify acceptance | 0.428 | 0.393 |
| MTP kept | 0.410 | 0.387 |
| tokens re-aligned per rejection (gate >= 3) | 0.26 | 0.46 (old drafts), 0.39 (verify picks) |
| first position after the correction matches | 17% | 13% |

After a different word, the next token's distribution moves enough that even the same noise picks a different token
87% of the time, so prose does not fall back onto the old path. Gumbel coupling also agrees slightly less than
speculative sampling's 1 - TV (unit test: 0.950 vs 0.960), which costs acceptance at every position. The pick itself
is exact (chi-squared 8.35 on 5 dof).
