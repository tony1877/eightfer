# Multi-sequence state: speculative full-model passes + batched streams

Both features need the 27B to hold more than one sequence state. Today `Qwen35` holds one: per layer
`conv_state_` / `ssm_state_` (DeltaNet), one KV ring `k_cache_` / `v_cache_` (+ RAM KV `hk_` / `hv_`), one
`n_past_`, `slot_pos_`, snapshot slots 0/1 and one record (`rec_*`).

## Measured premises (2026-10-07)

- A verify costs ~255 ms flat for 1-128 tokens (PCIe-bound, residual prefetch): extra sequences in one verify are
  nearly free.
- Cycles where the verify accepts every draft: prose 37% (kmin 12) / 21% (kmin 32), explain 48%, code 56%.
- Drafting a cycle: ~270-380 ms; the verify's GPU compute is ~60 ms of its 255 ms (the rest is copy-engine time).

## Phase 1: sequence slots

`SeqState { conv[], ssm[], n_past, ring offset }`: S slots of DeltaNet state (0.16 GB each) in VRAM; KV rows
per slot. Slot 0 is today's sequence; `eval(..., seq)` picks the slot. The DeltaNet ops already take n_seqs.
Attention: each slot owns a disjoint region of the KV ring (ring-exact contexts) so masks stay per slot.

## Phase 2: speculative full-model pass (single stream, exact)

Cycle c: launch the verify of drafts d[1..k] async on stream A (slot 0). Meanwhile on stream B, copy the
draft state after d[k] into slot 1 and keep drafting d[k+1..] (base + MTP), assuming all k accepted.
- verify accepts all k: slot 1's drafts become cycle c+1's drafts (no bonus token; exact). Saves one drafting
  phase. Expected: prose +20%, code +35%.
- else: discard slot 1.
Needs: a second ggml backend/sched (own CUDA stream), the drafts' KV written to slot 1's rows.

## Phase 3: batched streams (throughput, exact per stream)

The server runs up to B requests at once: each owns a slot; one verify evaluates all their drafts (B x k tokens,
still ~255 ms); drafting batches the B streams' base passes. Expected aggregate 4-8x at B = 4-8.

Each phase lands only after: greedy compare against plain decoding per slot, PPL/KLD unchanged, 256K needle,
server smoke test.
