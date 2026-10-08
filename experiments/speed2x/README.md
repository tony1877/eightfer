# Speed and memory models behind docs/SPEED2X.md

Cycle-level models, not measurements. Every input is labelled in the scripts and in SPEED2X.md §10.2.
All of them must be replaced by `shoehorn bench` numbers from the target box.

| Script | What it computes |
|---|---|
| `m27.py` | 27B decode model: Monte Carlo over speculative verify cycles (dual-path verify, MTP-staged drafter, draft-during-verify, hedges, 200k KV streaming) |
| `run1.py` `run2.py` `run3.py` | Waterfall, hardware × α grid, sensitivities and lossy options, using `m27.py` |
| `mflash.py` | Flash-Next decode by drive topology (T1/T2/T3), cache shares, 64 GB RAM option |
| `mem.py` | VRAM/RAM plans at 4k and 200k, context ceiling, pinned-memory needs |
| `prefill.py` | 200k-token prefill time for both models, prefix-cache restore |
| `critic/indep27.py` `critic/flash_c.py` `critic/lossyq3.py` | The final critic's independent re-derivation (27B MC, Flash without lookahead, Q3_K residual) |

Run any of them with `python3 <script>` from this folder (stdlib only).
