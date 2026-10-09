# shoehorn

An inference engine that fits **Qwen3.8-27B** at near-8-bit quality onto a 16 GB GPU, and runs
**Qwen3.8-Flash-Next** (176B MoE) from NVMe. OpenAI-compatible server, a live dashboard, made for coding agents.

How it works: each weight is stored as a 4-bit base plus a 4-bit residual.

- The base (IQ4_XS, 13.9 GB) lives in VRAM and drafts tokens fast.
- The residual (Q4_K, 14.4 GB) lives in pinned system RAM. Base + residual check the drafts.
- Speculative sampling makes the output follow the base + residual distribution exactly (~Q8 quality;
  IQ4_XS + Q4_K has 0.89x the weight error of Q8_0).

## Status

Working and in daily use on the reference box: Windows 11, RTX 5080 16 GB, Ryzen 7 9800X3D, 32 GB DDR5.
Measured there (27B, production settings):

| What | Speed |
|---|---|
| Decode, code / tool calls | 44-78 tok/s |
| Decode, explanations | 36-38 tok/s |
| Decode, prose | 22-28 tok/s |
| Prefill, 6K prompt | ~1170 tok/s full, 1.6-1.75x that with `--fast-prefill` |
| 261,776-token prompt | prefill 835 tok/s, decode 24 tok/s at full context, 15.4 GB peak VRAM |
| Flash-Next | 18.8 tok/s (24 warm) |
| Agent reliability (`tests/agent_reliability`) | hard set 34/34 |

Where the decode time goes and what was tried: [`docs/DECODE-LIMITS.md`](docs/DECODE-LIMITS.md).

## What is supported

| | Supported | Notes |
|---|---|---|
| Models | GGUF architectures `qwen35` (Qwen3.8-27B and fine-tunes) and `qwen4exp` (Flash-Next) | Any other architecture is refused at load. Fine-tunes of these two work. |
| GPU | NVIDIA, CUDA 12.8+ | Default build targets Blackwell (`120a-real`). Other generations: `-CudaArch 89` (Ada), `86` (Ampere), etc. AMD and Intel GPUs are not supported. |
| VRAM | 16 GB minimum for the 27B | More VRAM can hold part or all of the residual (`--res-gpu-gb`), which shortens every verify. |
| RAM | ~32 GB for the 27B | 14.4 GB pinned residual + RAM KV (7-9 GB at 256K). Flash-Next streams experts from NVMe and uses RAM as a cache. |
| CPU | x86-64 | Default build uses AVX-512 (Zen 4/5, Ice Lake or newer). Any other x86-64 CPU: `-Portable`. |
| OS | Windows 10/11 | Linux is untested (some code paths exist; the build script is PowerShell). macOS: no. |
| PCIe | any | Verify speed is bound by host-to-GPU bandwidth: PCIe 5.0 x16 ~54 GB/s gives ~265 ms per verify; PCIe 4.0 roughly doubles it. |

**More VRAM.** `--res-gpu-gb G` keeps G GB of the residual in VRAM (layer by layer); only the rest crosses PCIe
on each verify. `--res-gpu-gb auto` takes all free VRAM but 3 GB (KV window, MTP, compute buffers). Output is
unchanged (greedy text identical). On the 16 GB reference card only ~1 GB fits next to a small KV: verify 259 ->
251 ms, in proportion to the bytes moved. The whole 14.4 GB residual needs a ~32 GB card; that case is not measured.

**Other models.** Other model families need their architecture ported (the layer graph lives in `src/model/`).

## Build (Windows)

Needs Git, the CUDA Toolkit >= 12.8, and **Visual Studio 2022** Build Tools with the C++ workload
(CUDA 13.0 does not accept VS 2026 as host compiler):

```powershell
winget install --id Microsoft.VisualStudio.2022.BuildTools --override "--quiet --wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
```

Clone to a short path and build. The first build compiles ggml's CUDA kernels and takes several minutes.

```powershell
git clone --recurse-submodules --shallow-submodules -b claude/qwen-custom-inference-engine-jhydpg https://github.com/tony1877/shoehorn C:\src\shoehorn
cd C:\src\shoehorn
.\scripts\build.ps1                 # options: -Portable (no AVX-512), -CudaArch <sm>, -NoCuda
```

## Quick start (27B)

1. **Pack** the model from Hugging Face BF16 safetensors. `--template` is any `qwen35` GGUF of the same model
   (for the tokenizer and metadata); `--imatrix` is optional but improves the base.

   ```powershell
   .\build\bin\shoehorn.exe pack --src D:\hf\Qwen3.8-27B --template D:\gguf\qwen3.8-27b-Q4_K_M.gguf `
       --out C:\models\q27 --base iq4_xs --res q4_K --imatrix D:\gguf\imatrix.gguf
   ```

   This writes `C:\models\q27.base.gguf` and `C:\models\q27.res.gguf`. `--res` can be `q3_K` (less RAM),
   `q5_K` / `q6_K` (more quality, more RAM and slower verify) or `none`. `--check` verifies the output.

2. **Serve**:

   ```powershell
   .\build\bin\shoehorn.exe serve C:\models\q27.base.gguf --res C:\models\q27.res.gguf `
       --port 8090 --alias qwen27 --ctx 262144 --kv q8_0 --kv-v q4_0 --spec auto --fast-prefill 4096
   ```

3. Point any OpenAI-compatible client at `http://127.0.0.1:8090/v1`. Open `http://127.0.0.1:8090/` for the
   dashboard (decode and prefill speed, accepted drafts, memory, PCIe traffic, per-request history).

Flash-Next needs no packing: serve its GGUF directly (experts stay memory-mapped, a GPU expert cache takes free VRAM).
Both models from one server: `--also ALIAS=PATH[,RES]`; the request's `model` picks one, the other is unloaded.

## Server flags

| Flag | Default | What |
|---|---|---|
| `--res FILE` | none | residual GGUF (27B) |
| `--res-gpu-gb G\|auto` | 0 | residual kept in VRAM; the rest streams from pinned RAM |
| `--host`, `--port` | 127.0.0.1, 8090 | listen address. Use `0.0.0.0` for LAN access, with `--api-key-file`. |
| `--alias NAME`, `--also ALIAS=PATH[,RES]` | | model names; extra models loaded on demand |
| `--api-key-file F`, `--chat-template-file F` | | bearer key; Jinja chat template override |
| `--idle-unload S` | off | free the GPU after S seconds idle (next request reloads) |
| `--ctx N` | 16384 | context length (up to 262144) |
| `--kv T`, `--kv-v T` | f16 | KV cache type (`q8_0`, `q4_0`); `--kv-v` sets V separately |
| `--kv-pool-gb G`, `--kv-lock` | | RAM KV pool size; lock it in physical memory |
| `--gpu-kv N`, `--gpu-layers N` | auto | VRAM KV window tokens; layers on GPU (Flash-Next) |
| `--expert-cache-gb G` | auto | Flash-Next GPU expert cache |
| `--spec auto\|K`, `--mtp N`, `--echo 0\|1` | auto, 3, 1 | draft length; MTP proposals per base pass; copy-from-context drafting |
| `--ubatch N` | 1024 | prefill batch |
| `--fast-prefill N` | off | read the prompt with the base only except its last N tokens: 1.6-1.75x prefill, KLD 0.0028 |
| `--slots N`, `--draft-batch N`, `--no-side` | 1 | concurrent conversations; side slot for title/summary requests |
| `--temp`, `--top-p`, `--top-k`, `--min-p`, `--presence-penalty` | model defaults | sampling defaults (requests can override) |
| `--tool-temp T` | 0.6 | temperature inside `<tool_call>` blocks |
| `--think-budget N`, `--think-after-tool 0\|1`, `--drop-reasoning 0\|1` | 32768 | reasoning cap and handling; a request's own `enable_thinking` or `think_after_tool` overrides `--think-after-tool` |
| `--sys-cache N`, `--sys-cache-gb G` | 4, 3 | system prompts kept in RAM for new conversations (one per client or agent persona) |
| `--hw-monitor URL` | http://127.0.0.1:8085 | LibreHardwareMonitor web server for the dashboard's CPU, RAM and VRAM temperatures (`""` = off) |
| `--threads N` | auto | CPU threads |
| `--timing-log FILE` | off | per-request JSONL with a decode time breakdown |

## Features

- **Self-speculative decoding.** The model's MTP head proposes tokens, the base checks them, base + residual verify
  up to 63 at once. `--spec auto` picks the draft length from measured acceptance and verify cost. Echo drafting
  copies repeated text (file contents, tool arguments) from the context.
- **Long context.** Up to 262K tokens. When the KV does not fit in VRAM it lives in RAM; decode uses exact attention
  over the top pages plus the last 4096 tokens (32K PPL 2.3301 vs 2.3317 exact; 256K needle recalled).
- **Prompt caches.** Each conversation reuses its prompt; a side slot keeps title/summary requests from evicting it;
  the system prompt prefix is kept in RAM and survives idle unloads.
- **Agent-friendly API.** llama-server compatible: `/v1/chat/completions`, `/v1/completions`, `/v1/models`,
  `/health`, `/stats`, `/unload`; tools / `tool_calls`, `reasoning_content`, `chat_template_kwargs`, streaming with
  usage and timings. Client disconnects stop generation.
- **Dashboard** at `/`, fed by `/stats`.

## Agent and web UI

`agent/` bundles a coding agent with a web UI: a fork of [dsh](https://github.com/deepseek-ai/deepseek-harness)
(MIT, DeepSeek), set up for shoehorn. It has a persistent PowerShell shell, file tools, background jobs, a task list,
durable memory, past-session search, web fetch, context compaction, skills and subagents, and it shows the server's
live stats in its header. Needs Node 24 and pnpm (`corepack enable`).

```powershell
$env:SHOEHORN_API_KEY = '<the key shoehorn serve was started with, if any>'
.\scripts\agent.ps1                 # installs and builds on first run, then prints the UI link
.\scripts\agent.ps1 --no-open --port 3080
```

Its settings, sessions and memory live in `~\.shoehorn\agent` (or `-AgentHome` / `$env:DSH_HOME`), filled on first
run from [`agent-home/`](agent-home): the routes to both models, the `shoehorn` agent preset and a few skills.
Upstream updates: `git subtree pull --prefix=agent <dsh repo> master --squash`. Any other OpenAI-compatible client
works the same; nothing in the server depends on the agent.

## Other commands

- `shoehorn bench`: RAM, GPU, PCIe and disk bandwidth (about 3-5 minutes; `--help` for options). Reference results:
  [`bench/results/2026-10-04-rtx5080-9800x3d`](bench/results/2026-10-04-rtx5080-9800x3d/README.md).
- `shoehorn ppl`: perplexity / KLD checks used for every quality claim above.
- `tests/server_smoke.ps1`, `tests/agent_reliability/`: server and agent tests.

## Docs

- [`docs/DESIGN.md`](docs/DESIGN.md): design and memory budgets.
- [`docs/DECODE-LIMITS.md`](docs/DECODE-LIMITS.md): decode time breakdown and measured dead ends.
- [`docs/SPEED2X.md`](docs/SPEED2X.md), [`docs/MULTISEQ.md`](docs/MULTISEQ.md), [`docs/BITLEVEL.md`](docs/BITLEVEL.md).

Kernels come from [ggml](https://github.com/ggml-org/llama.cpp) (MIT), pinned as a submodule. The runtime,
scheduling, storage tiers and split-precision format are shoehorn's.
