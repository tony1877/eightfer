# eightfer

An inference engine for **Qwen3.8-27B** and **Qwen3.8-Flash-Next** on one consumer PC:
Windows 11, an RTX 5080 16 GB, a Ryzen 7 9800X3D and 32 GB DDR5, plus NVMe/SATA SSDs.

The idea is **8-bit quality at 4-bit residency**. Each weight is stored as a 4-bit base and a 4-bit residual:

- The base lives in VRAM. It drafts tokens at full GPU speed.
- The residual lives in system RAM or on NVMe. Base + residual verify the drafts.

Speculative sampling makes the result lossless relative to the ≈Q8 model.
For the 176B Flash-Next, experts stream from NVMe through a VRAM/RAM heat cache.
Its 51B n-gram table is read straight from disk at full BF16 precision.

Status: **design phase**.

- [`docs/DESIGN.md`](docs/DESIGN.md) — design, memory budgets, estimates, milestones.
- [`experiments/nested_quant`](experiments/nested_quant) — first measurement on real Qwen3.8-27B weights.
  An IQ4_XS base plus a Q4_K residual has 0.89× the weight error of Q8_0.

Kernels come from [ggml](https://github.com/ggml-org/llama.cpp) (MIT). The runtime, scheduling, storage tiers
and split-precision format are eightfer's.
