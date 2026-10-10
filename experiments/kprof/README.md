# kprof: per-kernel GPU time through CUPTI (do not use as is)

A CUPTI activity tracer loaded with `CUDA_INJECTION64_PATH`. It was meant to split a base pass into GEMV time and
everything else (idea 3 in [`docs/SPEEDUP-PLAN.md`](../../docs/SPEEDUP-PLAN.md)) without Nsight Systems.

**It hangs the traced process on exit** (Windows 11, driver with CUDA 13.0, 2026-10-10). The process stays stuck in
a kernel wait inside the driver, keeps about 400 MB of VRAM, and cannot be killed. Only a reboot frees it. It also
never wrote its report: neither the atexit hook nor the periodic thread produced a file before the hang.

Use Nsight Systems for this instead, or time ops inside the engine.
