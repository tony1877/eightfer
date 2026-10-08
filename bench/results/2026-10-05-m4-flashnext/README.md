# M4: Flash-Next graph on the target box

## Tiny random model vs transformers 5.18 (`tests/tiny/run4.py`)

8 layers (6 Gated DeltaNet + 2 QSA attention), 8 experts top-2 + shared expert, 4 hyper-connection streams, PLE
n-gram embedding in layer 1, QSA budget 16 tokens in blocks of 4 (sparse from token 19 of 256), F32, CPU:

| shoehorn batch | rel. err | mean KLD | max KLD | same top-1 |
|---|---|---|---|---|
| 1, 7 | 3.4e-7 | 1.3e-14 | 2.1e-13 | 100% |
| 16 | 4.4e-5 | 2.1e-10 | 9.5e-10 | 99.6% |
| 64, 256 | 5.9e-4 | 3.7e-8 | 6.1e-8 | 99.2% |

Same model vs llama.cpp (pinned submodule): mean KLD 5e-6 on CPU and on GPU.

## Real model: orcarouter Qwen3.8-Flash-Next-Uncensored IQ3_XXS (85 GB, 2 splits, mmap)

Non-expert weights on the GPU, experts mapped on the CPU (llama.cpp: `-ngl 99 --cpu-moe`), ctx 512, 2 chunks of
llama.cpp docs:

| | PPL | KLD vs llama.cpp ub512 | same top-1 | s/pass |
|---|---|---|---|---|
| llama.cpp, ub 512 | 4.1164 | - | - | 37.4 |
| llama.cpp, ub 32 | 4.1694 | 0.0319 | 95.1% | |
| shoehorn, batch 512 | 4.1445 | 0.0352 | 96.1% | 38.9 |
| shoehorn, batch 31 | 4.1209 | 0.0324 | 96.3% | |

shoehorn is within llama.cpp's own batch-size noise on this model (top-10 routing over 512 experts on 3-bit weights
flips experts on tiny numeric differences), and runs at the same speed.
