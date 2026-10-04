#pragma once

// Sparse MoE expert FFN on the CPU for small batches: computes only the (token, slot) pairs whose routing weight is
// nonzero. With the GPU expert cache, the CPU half of a MoE layer gets the full top-k selection with zero weight on
// the cached experts; ggml's mul_mat_id would still compute every selected row, this skips them.
//
//   y[:, t] = sum_j w[j, t] * down_e (silu(gate_e x_t) * (up_e x_t)),  e = ids[j, t], over pairs with w[j, t] != 0
//
// Two ggml custom ops (gate/up, then down), multi-threaded over output rows, using ggml's CPU vec_dot kernels for
// the expert weight types.

#include "ggml.h"

#include <vector>

namespace e8::kernels {

// Lookahead prefetch: the next layer's expert tensors and its GPU-cache map (experts with slot >= 0 need no read).
struct PrefetchHint {
    const ggml_tensor *      gate = nullptr, * up = nullptr, * down = nullptr;
    const std::vector<int> * slot_of = nullptr;
};

// x [n_embd, n] F32, ids [k, n] I32, w [k, n] F32 (contiguous); gate/up [n_embd, n_ff, n_expert], down
// [n_ff, n_embd, n_expert] in host memory. Returns y [n_embd, n] F32.
// With `next_ids` ([k, n] I32, the next layer's predicted selection) and `hint`, the op also starts reading the
// predicted uncached experts of the next layer, so their weights arrive while this layer and the next GPU part run.
ggml_tensor * moe_cpu_sparse(ggml_context * ctx, ggml_tensor * x, ggml_tensor * ids, ggml_tensor * w, ggml_tensor * gate,
                             ggml_tensor * up, ggml_tensor * down, ggml_tensor * next_ids = nullptr,
                             const PrefetchHint * hint = nullptr);

// Wall time spent in the two ops so far (measured on thread 0), seconds, and the number of calls.
void moe_cpu_times(double & gate_up_s, double & down_s, long long & calls);

} // namespace e8::kernels
