#pragma once

// Small-batch Q4_K x F32 matmul for the CPU residual pass (speculative verify, decode).
//
// ggml's CPU mul_mat runs vec_dot once per (row, column), so each Q4_K block is unpacked once per column and a
// verify of k+1 tokens costs ~k+1 times the work of one. Here each block is unpacked once and dotted against all
// columns (AVX-512 VNNI), which keeps the pass memory-bound for up to kMaxCols columns.
//
// Two ggml custom ops: quantize the activations to Q8_K (in a layout matching the unpacked nibbles), then the GEMM.
// Integer dot products are identical to ggml's vec_dot_q4_K_q8_K; only float summation order differs.

#include "ggml.h"

namespace e8::kernels {

constexpr int kQ4kMaxCols = 16;

// y[N, ncols] = x[K, ncols] * w[K, N]^T for a Q4_K `w` and contiguous F32 `x` with ncols <= kQ4kMaxCols.
// Runs on the CPU backend (custom ops); `w` must live in host-accessible memory.
ggml_tensor * q4k_mul_mat_small(ggml_context * ctx, ggml_tensor * w, ggml_tensor * x);

bool q4k_small_supported(const ggml_tensor * w, const ggml_tensor * x);

} // namespace e8::kernels
