# What a BF16 weight really contains

Question: if each weight is split into bits, how many bits carry information, and how good is "only the first k bits"?

Same three real Qwen3.8-27B tensors as `experiments/nested_quant` (fetch them with its `fetch_tensors.py` into
`../nested_quant/w/`), then `python3 bitstats.py` (needs numpy).

Results (`results.txt`, all three tensors agree):

| | bits per weight |
|---|---|
| BF16 as stored | 16 |
| Information actually in it (order-0 entropy) | **10.5** (sign 1.0, exponent 2.55 of 8, mantissa 6.97 of 7) |

| Keep only the top bits (sign + exponent + k mantissa bits) | size | error (rel. RMSE) | exact remainder |
|---|---|---|---|
| k=0 | 3.55 bpw | 0.196 | 7 bits |
| k=1 | 4.50 bpw | 0.104 | 6 bits |
| k=2 | 5.49 bpw | 0.053 | 5 bits |
| k=3 | 6.49 bpw | 0.027 | 4 bits |
| for comparison: IQ4_XS / Q4_K / Q6_K / Q8_0 | 4.25 / 4.5 / 6.56 / 8.5 | 0.077 / 0.072 / 0.018 / 0.0054 | - |

So exact BF16 can be stored in ~10.5 bits instead of 16, and the bits removed from a truncated weight are pure noise
(full entropy), which must be stored as-is for exactness.
