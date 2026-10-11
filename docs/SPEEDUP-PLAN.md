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

**Pilot results (2026-10-10).**
- mtpdump on the 40 pilot conversations: 20,395 tokens at about 1,890 tok/s (`E:\shoehorn\mtp\pilot.bin`,
  214 MB). The base's top-1 equals the next token 71.9% of the time, and the next token is in its top 20 98.7% of
  the time.
- PyTorch 2.11 (cu128) is in `experiments/mtp_distill/.venv`. `mtp.py` holds the port, the offline acceptance
  metric (sum min(p_base, q_mtp) under top_k 20 / top_p 0.95, first step) and a trainer. The output head is cached
  as `E:\shoehorn\mtp\head_bf16.pt`.
- The port checks out. The stock 4-bit head's held-out first-step acceptance is 0.749 (base + residual: 0.7515).
  That is the design's assumed 0.75; with steps 2-3 decaying, it matches the engine's 41.6-44.8% per proposal.
- Training on 32 conversations (about 16K positions, lr 2e-5, 3 epochs) overfits:

  | | Train | Held-out |
  |---|---|---|
  | Epoch 1 | 0.744 | 0.7505 |
  | Epoch 2 | 0.830 | 0.7466 |
  | Epoch 3 | 0.849 | 0.7440 |

  So no gain can be seen at this data size.

**1.5M-token run (2026-10-11): gate missed, not exported.**
- Corpus: `make_corpus.py` against stock vLLM 0.31.0 serving `orcarouter/Qwen3.8-27B-Uncensored` at BF16 with the
  server's chat template, on one RunPod H100 (128 parallel requests): 2,300 conversations (26% with thinking, 1,394
  distinct prompts) in about 8 min, about $1.50 with setup. `E:\shoehorn\mtp\corpus.jsonl`.
- `mtpdump`: 1,495,443 tokens at 1,900 tok/s, `E:\shoehorn\mtp\corpus.bin` (15.7 GB). The base's top-1 equals the next
  token 71.5% of the time, and the next token is in its top 20 99.1% of the time (the pilot: 71.9%, 98.7%).
- `mtp.py` now trains the chain the way `Qwen35::mtp_step` runs it. Step k reads step k-1's `hn` and x_{t+k}, and is
  judged against the base at t + k. A later step's row attends to the real positions' K/V (from step 1) plus its own
  chain's entries. The loss is weighted 1 / 0.8 / 0.6, with full backprop through the chain.
  - Memory: per-step activation checkpointing and one autocast region for the whole chain. This peaks at 11.6 GiB
    on 16 GB; without it, a 3-step chain of about 1K rows runs out.
- Held-out: the last 100 conversations, with every training conversation that shares their prompt removed.
  Training: 2,067 conversations (1.32M tokens), 3 epochs, lr 3e-5 (warmup 30, cosine to 10%), 4 conversations per
  step, 33 min.

  | Held-out acceptance | Step 1 | Step 2 | Step 3 | Kept, 3-step chain |
  |---|---|---|---|---|
  | Stock 4-bit head (what the engine runs) | 0.7386 | 0.6055 | 0.5337 | 0.4748 |
  | Base + residual (the training start) | 0.7419 | 0.6107 | 0.5410 | 0.4800 |
  | Trained, epoch 1 | 0.7514 | 0.6247 | 0.5575 | 0.4942 |
  | Trained, epoch 2 | 0.7598 | 0.6420 | 0.5796 | 0.5101 |
  | Trained, epoch 3 | 0.7622 | 0.6473 | 0.5858 | 0.5148 |

  "Kept, 3-step chain" = (a1 + a1 a2 + a1 a2 a3) / 3, assuming each step's acceptance is independent. The pilot's
  earlier held-out gain shared prompts with its training set; this held-out set does not.
- Against the stock 4-bit head the trained layer is x1.084: +3.2% on step 1, +6.9% on step 2, +9.8% on step 3.
  Scaled to the engine, that gives 41.6% -> about 45%, before quantizing the layer back. The gate needs 50% (x1.2).
  At epoch 3, training (0.650) and held-out (0.515) have already diverged. Held-out gains shrank each epoch (+0.014,
  +0.016, +0.005), so 2-3x more data would plausibly add only another 0.01-0.02.
- Ready if needed: `export.py` (a copy of the base with `blk.64.*` replaced, as Q8_0 since gguf-py cannot write
  IQ4_XS; `--deq-pt` measures what the quantization costs offline) and `bench_prose.py` (8 prose requests through
  the server, reading MTP kept and tok/s from the timing log). The trained layer: `E:\shoehorn\mtp\mtp_chain.pt` (bf16).

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

## 7. Base calibration and residual width (2026-10-11)

Prose cycle today (`experiments/mtp_distill/bench_prose.py`: 4 prompts that are not in any corpus x 2 seeds, 500
tokens, thinking off, temp 1.0 / top_k 20 / top_p 0.95, through the server): 21.68 tok/s; per cycle, draft 404 ms,
verify 257 ms, 14.4 tokens.

**Prose-calibrated imatrix: no gain, dropped.**
- The production base's imatrix is 96 x 512 tokens of Python stdlib source. The 2026-10-04 sampled acceptance was
  lowest on prose (0.81, against 0.90-0.94 for code and reasoning), which suggested a calibration mismatch.
- New imatrix: llama-imatrix (build 11541) on the F16 GGUF, on one H100, about 30 min and $2.20. Input: 1,000 corpus
  conversations plus the same stdlib code, 73% prose and chat, 886K tokens (`E:\shoehorn\imatrix-prose-mix.gguf`).
  Re-packed from the BF16 safetensors (`E:\shoehorn\bf16`) as `orca27b-pm.*`.
- Base KLD against base + residual on held-out prose: 0.0262 -> 0.0254, same top-1 93.1% -> 93.0%. Against Q8_0 on
  the llama.cpp docs it got worse: 0.0577 -> 0.0610. The prose reference is `E:\shoehorn\mtp\prose-br.kld`: 64 x
  512 tokens of held-out corpus scored by the production base + residual.
- So IQ4_XS's error does not come from where the imatrix puts it.

**Q3_K residual: +7.5%, pending a quality decision.**
- `pack --res q3_K` with the production imatrix: the base's weights are byte-identical to production; only the
  residual changes, 14.64 -> 11.18 GB (`orca27b-r3.*`).
- Quality: KLD against Q8_0 0.0020 -> 0.0031 (same top-1 97.7% -> 97.4%). Against the Q4_K pair on prose: 0.0015.
  The base alone is 0.058.

  | | Verify per cycle | Draft per cycle | Tokens per cycle | tok/s |
  |---|---|---|---|---|
  | Q4_K residual | 257 ms | 404 ms | 14.4 | 21.68 |
  | Q3_K residual | 206 ms | 396 ms | 14.1 | 23.31 |

- The verify scales with the bytes streamed (0.80 against 0.76). Drafting is now two thirds of a cycle.

**Small standalone drafter (Qwen3.5-0.8B / 2B) in place of the MTP head: below the MTP head even after
distillation, dropped.**
- Both models have the 27B's tokenizer: the same 248,044 base tokens and merges. The 27B only adds 7 audio/TTS
  tokens. Both are the same `qwen35` architecture.
- Offline acceptance against the 4-bit base on the same 100 held-out conversations (`small_accept.py`;
  `distill_small.py` for the KD probe: the base's sampler distribution as target, embeddings frozen, 1 epoch, 1.32M
  tokens, 37 min locally):

  | Drafter | Per-token acceptance | Kept for 3 proposals |
  |---|---|---|
  | 0.8B, stock | 0.521 | 0.97 |
  | 2B, stock | 0.584 | 1.16 |
  | 0.8B, distilled | 0.588 | 1.18 |
  | MTP head, stock | 0.739 / 0.606 / 0.534 per step | 1.43 |

- Break-even with the MTP head is a flat acceptance of about 0.68. Distillation gains were shrinking (+0.028,
  +0.015, +0.014, +0.008 per 330K tokens), so 30-50x more data plausibly reaches 0.65-0.68. That is break-even, at a
  higher cost per step, with ~0.5 GB more VRAM than the head, and with a second model to roll back on rejections.
- The MTP head wins because it reads the 27B's own hidden state. A better trained drafter should keep that input and
  add capacity: an EAGLE-3-style head on features from several layers, not a separate small model.
