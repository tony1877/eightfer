# Zero-copy experiment

Question: can a few SMs read the pinned residual straight from RAM over PCIe, fast enough to replace the copy engine
plus VRAM staging, without slowing the drafter? This is the gate for idea 1 in
[`docs/SPEEDUP-PLAN.md`](../../docs/SPEEDUP-PLAN.md).

Method:
- Allocate 1 GiB of mapped pinned memory (`cudaHostAllocMapped`). G blocks read it through its device pointer with
  16-byte streaming loads, each block reading its own contiguous span.
- Compare with `cudaMemcpyAsync` H2D, which is what the verify uses today.
- Run both again next to a VRAM-bound reader that stands in for the drafter's GEMV. It reads 2 GiB as 672 spans of
  3 MB or 8192 spans of 256 KB.

Result (`results.txt`, RTX 5080, PCIe 5.0 x16, 2026-10-10):

| | GB/s | VRAM reader slowdown |
|---|---|---|
| Copy engine H2D | 57.9 | 7.6% |
| Zero-copy, 8-16 blocks of 256 threads | 53.7-53.8 | **0.3-0.5%** |
| Zero-copy, 2-8 blocks of 1024 threads | 53.7-53.8 | 0.3-0.9% |
| Zero-copy, ≥12 blocks of 1024 threads | 51-54 | 10-47% |

- A few SMs saturate PCIe. 1 block reaches 53.9 GB/s when it runs alone.
- The drafter is barely touched, as long as the blocks are small (256 threads). Large blocks hold SM thread slots
  for the whole transfer, and the reader's blocks straggle behind them.
- Raw zero-copy is **7% slower** than the copy engine (53.7 vs 57.9 GB/s). On its own, a zero-copy verify would take
  about 268 ms instead of about 249 ms for the 14.4 GB transfer. The gain has to come from what this removes: the
  VRAM write and read-back of the staged residual, the 410 split syncs, and the slowdown when the verify and drafting
  share the GPU (257 -> 298 ms measured).

Not measured here: dequantization and matmul inside the streaming kernel (step 3 of the plan), and the real
IQ4_XS GEMV in place of the stand-in reader.

Reproduce (Windows, CUDA 13.0, MSVC on PATH):

```powershell
nvcc -O3 -std=c++17 -arch=sm_120a -o zerocopy.exe zerocopy.cu
.\zerocopy.exe
```
