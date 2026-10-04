# Bench on the target box, 2026-10-04

RTX 5080 16 GB (PCIe Gen5 x16), Ryzen 7 9800X3D, 2 × 16 GB DDR5-5600 (configured 5600 MT/s), Windows 11 26200,
CUDA 13.0, VS 2022 Build Tools (MSVC 19.44). The llama-server model router was stopped for each run.
The disk test ran on C:\ (SN850X) and E:\ (980 PRO) only.

Raw output:
- [`cpu-gpu-pcie.txt`](cpu-gpu-pcie.txt) — final build (AVX-512 VNNI/BF16/VBMI on, pinned probe in a child process).
- [`disk.txt`](disk.txt) — disk section.
- [`cpu-before-vnni.txt`](cpu-before-vnni.txt) — CPU section of the first build, before VNNI/BF16 were enabled.

## Measured vs design

| What | DESIGN.md planned with | Measured |
|---|---|---|
| RAM read from cores | ~60 GB/s [est.] | **62 GB/s** (47 GB/s on 1 thread) |
| VRAM read, GEMV N=1 | ~770 GB/s [est.] | **822 GB/s** IQ4_XS, 834 Q4_K, 864 Q8_0 |
| H2D, pinned | ~50 GB/s | **57.5 GB/s** (1 GiB), 50.8 GB/s at 2.5 MiB; pageable 22.4 GB/s |
| CPU read + GPU DMA together | hypothesis: > 60 GB/s | **80.4 GB/s** (CPU 43.9 + H2D 36.5) — confirmed |
| Pinned memory | ~20 GB wanted | **15 GiB: WDDM caps it at half of RAM (15.6 GiB)** |
| Pinned alloc time | — | ~113 ms/GiB |
| SN850X, rand 2.5 MiB | ~7 GB/s | **7.05 GB/s** at QD4+ (5.24 at QD1, 0.50 ms) |
| 980 PRO, rand 2.5 MiB | ~7 GB/s | **5.88 GB/s** at QD4+ (3.86 at QD1, 0.68 ms) |
| Both NVMe at once | ~14 GB/s only if not sharing a link | **12.45 GB/s** (6.57 + 5.88): separate CPU root ports, no shared uplink |
| CPU GEMV Q4_K, 9 tokens | sandbox hint: compute-bound | **compute-bound**: best 54 ms / 9 tokens = 38.7 GB/s (`CPU_REPACK`, 16 threads) |

## What it means

- **27B verify must be dual-path, as SPEED2X planned.** At 9 tokens the CPU reaches 38.7 GB/s of the 62 GB/s it
  can read. Splitting R between CPU (in place) and GPU (over PCIe) uses the measured 80 GB/s combined, so streaming
  the 14.4 GB residual takes ~0.18 s instead of the 0.25 s DESIGN assumed.
- **Drafting is faster than assumed**: 13.6 GB of IQ4_XS at 822 GB/s is ~16.5 ms/token (DESIGN: 18.5).
- **Pinned memory is the binding constraint for 200k context.** R alone (14.4 GB) fits under the 15.6 GiB WDDM cap.
  R plus pinned q8_0 KV (21.9 GB in SPEED2X) does not. Options: KV in pageable RAM (H2D 22 GB/s), a smaller
  residual, or 64 GB of RAM (cap 32 GiB).
- **Flash-Next striping**: 12.45 GB/s rather than 14, so the striped column of DESIGN §5 drops by ~11%.

## AVX-512 VNNI / BF16

ggml's MSVC build enables AVX-512F but not VNNI/BF16/VBMI (MSVC has no `-march=native` and defines no macros for
them). `scripts/build.ps1` now enables them; `-Portable` turns them off.

| Q4_K GEMV, 9 tokens | without VNNI | with VNNI |
|---|---|---|
| `CPU`, 8 threads | 130.5 ms (16.1 GB/s) | **81.0 ms (26.0 GB/s), 1.6×** |
| `CPU_REPACK`, 8 threads | 60.8 ms | 60.2 ms |
| `CPU_REPACK`, 16 threads | 54.6 ms | 54.4 ms |

The repacked path, still the fastest, does not use VNNI in this ggml build, so the verify bottleneck is unchanged.

## Bench crash (fixed)

The first run aborted in `[pcie]`: after pinning 15 GiB, `cudaFreeHost` returned "out of memory" and ggml aborted
(exit `0xC0000409`). It happened again without any refused allocation, so the trigger is freeing with ~15 GiB pinned,
not probing past the limit. The capacity probe now runs in a child process (`eightfer pinned-probe <GiB>`) that
reports and exits without freeing; the transfer tests use one 1 GiB chunk. The default run completes.
