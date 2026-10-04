# Exact BF16 from the 4-bit draft's own bits

Encoder + decoder that rebuild every BF16 weight **bit for bit** from:
`B` (IQ4_XS draft, 4.25 bpw) + `R` (Q4_K residual, 4.5 bpw, the ≈Q8 target) + `T2` (exact remainder, ~3 bpw).

How `T2` works: given `B` and `R`, each true weight lies in a known interval. BF16 bit patterns are ordered like integers,
so `T2` stores only "which of the n BF16 values in this interval" it is. The width (ceil log2 n) is computed from `B`/`R`
alone, so no lengths are stored and a GPU can decode in parallel. Rare out-of-interval weights (~0.05–0.1%) are stored raw.

Results on the three real Qwen3.8-27B tensors (`results_v2.txt`, `results_v1.txt`):

| Scheme | bits/weight | Bit-exact? |
|---|---|---|
| B + exact tail | 11.5 (V2 code) / 10.9 (ideal rANS) | 0 mismatches of 204.5M |
| B + R + T2 | 11.8–11.9 (V2) / 11.5 (rANS); B+R alone = 0.89× Q8_0 error | 0 mismatches |
| B + T1 (scale-free 4-bit) + T2 | 11.3 (V2) / 10.9 (rANS); B+T1 alone = 1.05× Q8_0 error | 0 mismatches |
| reference: information content of BF16 | 10.5 | — |

`cascade_test.py`: χ² test that chained speculative sampling (draft → ≈Q8 → exact) emits exactly the top model's distribution
on a toy vocabulary (185.6 on 166 dof, expected 166 ± 18; the same test rejects the ≈Q8 distribution at 6898.7).

Build: same as `experiments/nested_quant` (links ggml from the llama.cpp submodule), e.g.
`g++ -O2 -std=c++17 bct_rt2.cpp -I../../third_party/llama.cpp/ggml/include <ggml-cpu.a> <ggml-base.a> -lpthread -lm`.
