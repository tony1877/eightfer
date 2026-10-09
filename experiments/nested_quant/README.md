# Nested-quant experiment

Question: is a 4-bit base plus a quantized residual as accurate as Q8_0?

Method:
- Take three real BF16 tensors from Qwen3.8-27B: an MLP down_proj, a GDN in_proj_qkv and an attention q_proj.
- Quantize each directly with ggml (Q8_0, Q6_K, Q5_K, Q4_K, IQ4_XS).
- Separately, quantize a base (IQ4_XS or Q4_K), then quantize the residual `W - deq(base)`.
- Compare the relative RMSE of every variant against the original weights. No imatrix is used.

Result (`results.txt`): IQ4_XS + Q4_K residual (8.75 bpw) has **0.89x** the error of Q8_0 on all three tensors.
With a Q5_K residual it has 0.46x.

This is weight-space error only. Output-level KLD comes in milestone M3.

Reproduce on Linux or WSL. This uses ggml from llama.cpp `836d571`.

```bash
git clone https://github.com/ggml-org/llama.cpp && git -C llama.cpp checkout 836d571
cmake -S llama.cpp -B build-ggml -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DBUILD_SHARED_LIBS=OFF \
      -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_CURL=OFF
cmake --build build-ggml -j --target ggml-base ggml-cpu
g++ -O2 -std=c++17 nested.cpp -Illama.cpp/ggml/include \
    build-ggml/ggml/src/libggml-cpu.a build-ggml/ggml/src/libggml-base.a -lpthread -lm -o nested
python3 fetch_tensors.py          # ~410 MB of HTTP range requests
./nested w/10.mlp.down_proj.weight.bin 5120 17408
./nested w/10.linear_attn.in_proj_qkv.weight.bin 10240 5120
./nested w/11.self_attn.q_proj.weight.bin 12288 5120
```
