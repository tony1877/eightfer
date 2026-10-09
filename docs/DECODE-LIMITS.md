# Decode speed: where the time goes and what is left (2026-10-09)

Measured on the RTX 5080 (16 GB, PCIe 5.0 x16) with the 27B orca pair (IQ4_XS base 13.9 GB in VRAM, Q4_K residual
14.4 GB in pinned RAM), production settings. Every option below was measured before deciding; none of them changes
outputs.

## One check (speculative cycle)

| Part | Time | What bounds it |
|---|---|---|
| Verify (base + residual over the drafts) | ~265 ms | the residual crossing PCIe: 14.4 GB at ~54 GB/s |
| Drafting | ~300-400 ms | base passes (~19.5 ms each) checking MTP proposals, 1-7 tokens per pass |
| Everything else (commit, server, streaming) | ~2 ms | |

Tokens per check: code through a tool ~50, code ~26, prose ~15-17. Decode: code 44-78 tok/s, prose ~22-26 tok/s.
Tokens per base pass: code 4.5, prose 1.15 (MTP proposals kept: code 77%, prose 42-44%).

## What was tried, with numbers

| Idea | Measurement | Gain |
|---|---|---|
| Bigger prefill batches (512 -> 1024) | +5% prefill; 2048 does not fit next to the KV window | shipped |
| Base-only prefill except the last 4096 tokens (`--fast-prefill`) | KLD 0.0028 vs full, top-1 98.6%, hard agent test 34/34 | 1.6-1.75x prefill, shipped |
| Pipelining (draft the next check during the verify) | drafts past the end are usable only when the whole check is accepted, which is rare; old shadow path broken (1.5 tok/s) | ~0 |
| Prefetch residual during drafting | staging slots are 2 x 71.5 MB (410 splits); a bigger pool would come out of the KV window | ~1% |
| Base pass overhead | 19.5 ms vs llama.cpp 18.9 ms on the same file (llama-bench tg64 53 tok/s); CUDA graphs already save 4-6 ms | ~3% at most |
| Custom IQ4_XS GEMV for drafts | ggml's kernel: 872 GB/s at N=1 (91% of 960), 811-856 at N=2-6; dips only at N=7-8 (711 / 635 GB/s) | ~1-2% |
| Lossless residual compression + GPU decoder | residual Q4_K codes carry 3.96 of 4 bits (99%), scales 7.1 of 8 bits | ~1% |
| Prose at temperature 0.6 | 25.3 tok/s vs 25.9 at 1.0, same MTP acceptance | none: prose is bounded by the MTP head, not by sampling |

## What would move it

- More VRAM (24-32 GB): the residual stays on the GPU, a verify drops from ~265 ms to ~30 ms.
- A better draft model for prose than the built-in MTP head (multi-candidate drafting needs exact multi-draft
  rejection sampling to stay lossless; not done).
