# M5: Flash-Next decode with the GPU expert cache

orcarouter Qwen3.8-Flash-Next-Uncensored IQ3_XXS (85 GB, 53.5 GB of experts), RTX 5080 16 GB, 32 GB RAM, model on the
SN850X (C:). Greedy decode after the 81-token `bench/prompts/reasoning.txt` prompt, `eightfer decode`.

| Configuration | decode tok/s (256 tokens) | warm (2nd half) |
|---|---|---|
| llama.cpp `llama-bench` tg128, `-ngl 99 -ncmoe 46` (the router's setting) | 9.5-9.7 | |
| eightfer, experts mapped on the CPU, no cache, no CUDA graphs | 10.3-11.2 | 10.0-11.8 |
| + GPU expert cache (84 slots/layer, 9.8 GB), ggml mul_mat_id CPU half | 10.7-10.8 | 11.4-12.0 |
| + sparse CPU MoE kernel (only the routed-to-CPU pairs) | 11.4 | 13.7 |
| + CUDA graphs (were off in eightfer's ggml build) | 14.5 | 17.1 |
| + background prefetch thread, **lookahead off (default)** | **15.7** | **20.2** |
| same, lookahead prefetch on (`E8_LOOKAHEAD=1`) | 15.2 | 15.5 |
| + graph-stable KV/indexer writes (`set_rows` with row inputs, so CUDA graphs replay every token) | **18.8** | **24.3** |

With the last row: 2.0x llama.cpp overall, 2.5x warm. Expert cache hit rate: 77% of the 480 expert uses per token served from VRAM (heat-based admission, LFU with
decay). Per decode token (default config): graph build + alloc + inputs 5 ms, compute 53 ms (CPU expert ops 19 ms,
GPU + 96 GPU/CPU split boundaries the rest), cache update 6 ms.

Findings on the way:
- CUDA's `mul_mat_id` assumes each token uses an expert at most once (rows are compacted per token); repeated ids in a
  token row corrupt results or fault. The cache keeps ids distinct per row: per-position zero slots on the GPU side,
  and the split is only used below ggml's op-offload batch (32), where the CPU half runs on the CPU.
- `mul_mat_id` computes every selected row even at zero weight, so the CPU half needs its own sparse kernel.
- `PrefetchVirtualMemory` blocks its caller while it queues reads: on a compute thread it doubled the expert time.
- Lookahead (next layer's router on this layer's input) mispredicts enough that its extra reads cost more than the
  hits save on this box.

Not built yet from DESIGN section 5: explicit IOCP unbuffered reads into a pinned RAM tier (the OS page cache plays
that role now), drive striping, and a calibration-profile planner (the cache budget is "free VRAM minus 3 GB").
