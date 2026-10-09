# shoehorn: bit-level lossless inference

Status: design plus experiments on real Qwen3.8-27B weights.

- **How it was produced:** three independent designs, all of which reached nearly the same numbers.
- **Review:** the panel's adversarial review was cut short by usage limits. Only these spot checks were done:
  - Exact-rebuild logs: 0 mismatches.
  - Statistical (chi-squared) test of the cascade.
- **Labels:** [M] measured, [D] derived, [A] assumed. Speed figures are [D/A] until `shoehorn bench` runs on the target box.

## 1. The idea in plain words

The idea: split every BF16 weight into pieces, multiply only the pieces you need, and still get an exact result.

**1. A BF16 weight carries only 10.5 bits of information [M].** The 8-bit exponent holds 2.55 bits; the 7 mantissa bits are
pure noise. Exact BF16 therefore needs ~10.5 bits/weight, not 16.

**2. Every weight is stored as three nested pieces:**

| Piece | Where it lives | Bits/weight | What it is |
|---|---|---|---|
| B | GPU | 4.25 | 4-bit sketch (IQ4_XS). Drafts tokens fast. |
| R | RAM | 4.5 | Refinement. B+R has 0.89x the error of Q8_0 [M]. |
| T2 | RAM / NVMe | ~3 | Exact remainder. B+R+T2 is the original BF16, bit for bit. |

- Exactness [M]: 0 mismatches over 204.5M real weights.
- Total size: 11.5-11.9 bits/weight; the theoretical floor is 10.5.

**3. How T2 stays so small.** Once B and R are known, the real weight is pinned to a tiny interval. BF16 bit patterns are
ordered like integers, so T2 only says "which of the n BF16 numbers in this interval" it is. Its width can be computed from
B and R, so no lengths are stored and the GPU can decode in parallel.

**4. Multiplying only the bits we need.**
- Drafting uses only the 4-bit sketch.
- The ~Q8 level (B+R) checks the drafts.
- The exact level (B+R+T2) checks a batch of ~32-64 tokens in one pass.
- Speculative sampling makes the output exactly the BF16 model's distribution. [M] chi-squared test on a toy vocabulary:
  185.6 on 166 dof, expected 166 +/- 18.
- Averaged over generated tokens, only ~1 bit per weight crosses the slow bus per token.

**5. Where bits cannot be skipped [M].**
- Inside the 64 layers, a provable "the skipped bits don't matter" bound is 37-151x too loose and compounds layer by layer.
- No attention key was ever exactly skippable: 0 of 24.6M pairs.
- So for an exact answer, every inner bit must be read. It just doesn't have to be read for every token.

**6. The one place bit-skipping is provable: the final 248k-row vocabulary layer.** The 4-bit scores plus an error bound
prove which rows can matter. Full bits are read only for those rows: ~8.5% per verify, for greedy and for Qwen's sampler
(T=1, top_k 20, top_p 0.95). The exact distribution was reproduced at 180/180 tested positions [M].

## 2. Speed and quality by mode (27B; estimates from three independent designs)

| Mode | Output is exactly | 27B @ 4k | 27B @ 200k | Flash-Next @ 4k |
|---|---|---|---|---|
| **Tier B (default)** | the B+R ~Q8 model | **~28 tok/s** | **~17 tok/s** | ~13.8 tok/s |
| **Tier A (exact)** | the BF16 model (bit-exact weights) | **~18-19 tok/s** | **~10-12 tok/s** | ~4 tok/s (not recommended) |

- **What exactness costs:** on this box, about a third of the speed.
- **The B+R target is already very close to fp32 [M, one easy 366-token text - needs broader testing]:**
  - KL vs the fp32-activation model: 5.2e-5 for B+R, against 1.9e-4 for HF's own BF16 inference of the same weights.
  - So by this one measurement, Tier B is closer to the true model than standard BF16 inference is.
  - Tier A matters when bit-exact weights are a requirement in themselves.
- **Tier A speed hinges on one assumption:** how often B+R agrees with exact BF16, assumed ~0.99 per token. On the one text
  above it measured 0.995. Code, math and near-ties may run lower (0.98 costs ~15%).
- **Flash-Next exact mode** needs ~2.7x more bytes per expert cache miss, and every token needs its own experts. Not worth it.

## 3. The exact remainder (technical)

- **Key:** for a BF16 pattern `b`, `key = 32768 + (b & 0x7FFF)` if positive, `32767 - (b & 0x7FFF)` if negative.
  It is strictly increasing in value.
- **Interval from B** (IQ4_XS sub-block scale `D = fp16(d)*(ls-32)`): the grid cell around `D*kv[k]`, widened by `|D|/16`.
  - The widening absorbs the fp16 scale rounding.
  - Escapes outside the cell: 0.04-0.11% [M], stored raw. Zero-scale blocks are stored raw.
- **Remainder:** `lk = min key >= lo`, `n = (max key <= hi) - lk + 1`, `t = key(W) - lk`. Codes for `t`:
  - V1: `ceil(log2 n)` bits.
  - V2: sign + binade code.
  - rANS: value-uniform model.
- **With R:** the interval is cell_B intersect cell_R, which leaves T2 at ~3 bits.
- **Measured sizes** (bits/weight, three tensors; files in `experiments/exact_tail`):

| Scheme | V2 code | ideal rANS |
|---|---|---|
| B + exact tail directly | 4.25 + 7.16-7.18 -> 11.5 | 10.9 |
| **B + R + T2** | 8.75 + 2.98-3.04 + 0.08 -> 11.8-11.9 | 11.5 |
| B + T1 (scale-free 4-bit refinement, 1.05x Q8_0 error) + T2 | 11.3 | 10.9 |
| DF11-style exact copy (no draft reuse) | 10.6 | - |

- **Why not just the most compact exact format:** DF11 (10.6) is smaller in total. But the nested scheme reuses the draft's
  bits, so the exact level reads only ~3 extra bits/weight instead of 10.6.

## 4. The cascade and what makes it exact

`MTP -> B (4-bit) -> B+R (~Q8) -> B+R+T2 (exact BF16)`.

Each level verifies the one below it with speculative sampling (Leviathan 2023; Chen 2023). By induction, the output has
the top level's distribution. Required conditions:
1. Every level's accept and residual tests use exactly the stored q array of the level below.
2. After a rejection, everything later is discarded and the lower levels restart from the exact state.
3. The exact level uses bit-exact weights and a pinned reference arithmetic: bf16 activation boundaries, KV dtype,
   GDN state dtype and logit rounding.
   - Two engines with different fp32 summation order differ by ~1e-6 relative [M]; that is the irreducible floor.
   - Greedy output is token-identical only with batch-invariant kernels (fixed reduction order).

The exact level verifies ~32-64 tokens per pass and reads T2 only then. The optimal batch length is
`L* ~ sqrt(2*deltaT / ((1-a3)*c2))`, where:
- deltaT = time of one exact pass
- a3 = per-token agreement between B+R and exact BF16
- c2 = time per token at the ~Q8 level

## 5. Memory (27B, Tier A) [D]

- **4k context:**
  - VRAM 15.5 GB: B 13.61 + the existing plan's buffers + ~0.3 for the exact level's GDN state and replay records.
    The resident R slice is dropped.
  - RAM 22 GB: R 14.4 + T2 ~7-8 + exact KV. About 1-2 GB of T2 spills to NVMe and is read once per exact pass.
- **200k context:** very tight.
  - Exact KV is stored as q8_0 KV plus a conditioned tail. Its size on real K/V is unmeasured: real K/V has outlier channels,
    and if the tail needs 5+ bits/elem it no longer fits.
  - Needs >=15.55 GB of measured free VRAM.

## 6. Ruled out [M]

- Provable bit-skipping inside the layers (bounds 37-151x loose, compounding over 64 layers).
- Exact skipping of attention keys.
- Deciding accept/reject from bounds alone.
- Exact 1-bit sign/magnitude planes: 15.4 bits/weight, worse than BF16's information content.
- Bit-prefix heads (sign + exponent + k mantissa bits) as drafters: worse than IQ4_XS per VRAM byte
  (`experiments/bitplanes`).
- An exact mode for Flash-Next.

## 7. Open

- **Agreement a3** between B+R and exact BF16, on varied text (code, math, long context).
- **T2 decode throughput on the GPU:** it must reach >=50 G weights/s inside the verify pass.
- **Exact-KV tail size** on real K/V at 200k.
- **The fourth design angle was not run:** CPU/GPU code that multiplies bit-planes directly. The question was whether it
  can keep the CPU at memory speed rather than compute speed at 12-24 tokens per pass.
- **Adversarial review** of the three designs; cut short by usage limits.
