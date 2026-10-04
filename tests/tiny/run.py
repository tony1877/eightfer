"""M2 check: a tiny random-weight Qwen3.5 text model, run by HF transformers (float32, CPU) and by eightfer
(GGUF made by llama.cpp's convert_hf_to_gguf.py, F32, CPU). Prints logit agreement.

    .venv\Scripts\python tests\tiny\run.py [--eightfer build\bin\eightfer.exe] [--batch 64]

Needs: torch (CPU), transformers >= 5, numpy, safetensors, huggingface_hub (tokenizer files are fetched from the
27B repo because the converter needs the real tokenizer; vocab is the real 248320).
"""
import argparse, os, shutil, subprocess, sys
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests" / "tiny" / "out"
TOKENIZER_REPO = "Qwen/Qwen3.8-27B"
TOKENIZER_FILES = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "chat_template.jinja"]


def make_config():
    layer_types = ["linear_attention"] * 3 + ["full_attention"]
    return AutoConfig.for_model(
        "qwen3_5_text",
        vocab_size=248320, hidden_size=256, intermediate_size=512, num_hidden_layers=8,
        layer_types=layer_types * 2, full_attention_interval=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=64, attn_output_gate=True,
        linear_conv_kernel_dim=4, linear_key_head_dim=32, linear_value_head_dim=32,
        linear_num_key_heads=2, linear_num_value_heads=4,
        rms_norm_eps=1e-6, max_position_embeddings=4096, tie_word_embeddings=False,
        rope_parameters={"rope_type": "default", "rope_theta": 10000000.0, "partial_rotary_factor": 0.25,
                         "mrope_interleaved": True, "mrope_section": [3, 3, 2]},
        partial_rotary_factor=0.25, bos_token_id=248044, eos_token_id=248044,
        architectures=["Qwen3_5ForCausalLM"], dtype="float32",
    )


def randomize(model, seed):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "norm" in name:
                p.copy_(torch.empty_like(p).uniform_(-0.3, 0.3, generator=g))
            elif "A_log" in name:
                p.copy_(torch.empty_like(p).uniform_(-1.0, 1.0, generator=g))
            elif "dt_bias" in name:
                p.copy_(torch.randn(p.shape, generator=g) * 0.5)
            elif "embed_tokens" in name:
                p.copy_(torch.randn(p.shape, generator=g))
            else:
                fan_in = p.shape[-1] if p.dim() > 1 else 1
                p.copy_(torch.randn(p.shape, generator=g) / fan_in ** 0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eightfer", default=str(ROOT / "build" / "bin" / "eightfer.exe"))
    ap.add_argument("--batch", type=int, nargs="+", default=[64, 1, 256])
    ap.add_argument("--n-tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--kv", default="f16", choices=["f16", "f32"])
    ap.add_argument("--reuse", action="store_true", help="skip regenerating the model if the GGUF exists")
    a = ap.parse_args()

    hf_dir = OUT / "hf"
    gguf = OUT / "tiny-f32.gguf"
    torch.manual_seed(a.seed)
    cfg = make_config()
    model = AutoModelForCausalLM.from_config(cfg, dtype=torch.float32).eval()
    randomize(model, a.seed)

    if not (a.reuse and gguf.exists()):
        if hf_dir.exists():
            shutil.rmtree(hf_dir)
        model.save_pretrained(hf_dir, safe_serialization=True)
        for f in TOKENIZER_FILES:
            shutil.copy(hf_hub_download(TOKENIZER_REPO, f), hf_dir / f)
        env = dict(os.environ, PYTHONPATH=str(ROOT / "third_party" / "llama.cpp" / "gguf-py"))
        subprocess.run([sys.executable, str(ROOT / "third_party" / "llama.cpp" / "convert_hf_to_gguf.py"), str(hf_dir),
                        "--outtype", "f32", "--no-mtp", "--outfile", str(gguf)], check=True, env=env,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    rng = np.random.default_rng(a.seed)
    toks = rng.integers(1000, 200000, size=a.n_tokens).astype(np.int64)
    with torch.no_grad():
        ref = model(torch.from_numpy(toks)[None]).logits[0].float().numpy()
    (OUT / "tokens.txt").write_text(" ".join(map(str, toks)))

    ok = True
    for b in a.batch:
        out = OUT / f"e8-b{b}.f32"
        subprocess.run([a.eightfer, "logits", str(gguf), "--tokens", str(OUT / "tokens.txt"), "--out", str(out),
                        "--batch", str(b), "--kv", a.kv], check=True, stdout=subprocess.DEVNULL)
        e8 = np.fromfile(out, dtype=np.float32).reshape(ref.shape)
        rel = np.linalg.norm(e8 - ref) / np.linalg.norm(ref)
        mx = np.abs(e8 - ref).max()
        lp_r = torch.log_softmax(torch.from_numpy(ref).double(), -1)
        lp_e = torch.log_softmax(torch.from_numpy(e8).double(), -1)
        kld = (lp_r.exp() * (lp_r - lp_e)).sum(-1)
        top1 = (ref.argmax(-1) == e8.argmax(-1)).mean() * 100
        # random weights produce near-ties, so judge by distribution distance, not exact argmax
        good = rel < 2e-3 and kld.max() < 1e-4 and top1 >= 99.0
        ok &= good
        print(f"batch {b:4d}: rel.err {rel:.2e}  max|d| {mx:.2e}  mean KLD {kld.mean():.2e}  max KLD {kld.max():.2e}"
              f"  top-1 {top1:.1f}%  {'OK' if good else 'FAIL'}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
