"""M4 check: a tiny random-weight Qwen4-Exp (Flash-Next architecture) run by HF transformers (float32, CPU, eager) and
by eightfer (GGUF from llama.cpp's convert_hf_to_gguf.py, F32, CPU). Exercises hyper-connections, the n-gram (PLE)
embedding, MoE + shared expert, Gated DeltaNet with the sigmoid gate and QSA sparse attention (budget 16 tokens in
blocks of 4, so a 256-token sequence uses the block selection).

    .venv\\Scripts\\python tests\\tiny\\run4.py [--batch 64 1 256]
"""
import argparse, os, shutil, subprocess, sys
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests" / "tiny" / "out4"
TOKENIZER_REPO = "Qwen/Qwen3.8-27B"
TOKENIZER_FILES = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "chat_template.jinja"]
EOS = 248044


def make_config():
    return AutoConfig.for_model(
        "qwen4_exp_text",
        vocab_size=248320, hidden_size=256, num_hidden_layers=8,
        layer_types=(["linear_attention"] * 3 + ["full_attention"]) * 2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=64,
        linear_conv_kernel_dim=4, linear_key_head_dim=32, linear_value_head_dim=32,
        linear_num_key_heads=2, linear_num_value_heads=4,
        num_experts=8, num_experts_per_tok=2, moe_intermediate_size=64, shared_expert_intermediate_size=64,
        norm_topk_prob=True, output_gate_type="sigmoid",
        hc_count=4, hc_lowrank=32,
        ple_layer_ids=[2], ple_conv_kernel_size=4, ngram_size=3, heads_per_ngram=2,
        ngram_vocab_size_base=500, make_ngram_vocab_size_divisible_by=128, seed=1234, split_ngram_parts=4,
        indexer_n_heads=2, indexer_kv_heads=1, indexer_head_dim=32, indexer_budget=16, indexer_compress_ratio=4,
        rms_norm_eps=1e-6, max_position_embeddings=4096, tie_word_embeddings=False,
        rope_parameters={"rope_type": "default", "rope_theta": 10000000.0, "partial_rotary_factor": 0.25,
                         "mrope_interleaved": True, "mrope_section": [3, 3, 2]},
        partial_rotary_factor=0.25, bos_token_id=EOS, eos_token_id=EOS,
        architectures=["Qwen4ExpForCausalLM"], dtype="float32",
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
            elif "embed" in name:
                p.copy_(torch.randn(p.shape, generator=g))
            else:
                fan_in = p.shape[-1] if p.dim() > 1 else 1
                p.copy_(torch.randn(p.shape, generator=g) / fan_in ** 0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eightfer", default=str(ROOT / "build" / "bin" / "eightfer.exe"))
    ap.add_argument("--batch", type=int, nargs="+", default=[64, 1, 7, 256])
    ap.add_argument("--n-tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--reuse", action="store_true")
    a = ap.parse_args()

    hf_dir, gguf_file = OUT / "hf", OUT / "tiny4-f32.gguf"
    torch.manual_seed(a.seed)
    model = AutoModelForCausalLM.from_config(make_config(), dtype=torch.float32, attn_implementation="eager").eval()
    randomize(model, a.seed)
    if not (a.reuse and gguf_file.exists()):
        if hf_dir.exists():
            shutil.rmtree(hf_dir)
        model.save_pretrained(hf_dir, safe_serialization=True)
        # transformers saves "indexed_attention"; real checkpoints (and llama.cpp's converter) use "full_attention"
        cfg = hf_dir / "config.json"
        cfg.write_text(cfg.read_text().replace('"indexed_attention"', '"full_attention"'))
        for f in TOKENIZER_FILES:
            shutil.copy(hf_hub_download(TOKENIZER_REPO, f), hf_dir / f)
        env = dict(os.environ, PYTHONPATH=str(ROOT / "third_party" / "llama.cpp" / "gguf-py"))
        subprocess.run([sys.executable, str(ROOT / "third_party" / "llama.cpp" / "convert_hf_to_gguf.py"), str(hf_dir),
                        "--outtype", "f32", "--no-mtp", "--outfile", str(gguf_file)], check=True, env=env,
                       stdout=subprocess.DEVNULL)
        # the converter reads layer_types through transformers, which renames "full_attention" to
        # "indexed_attention", so it writes ratio 0 everywhere; set the QSA layers' ratio in place
        ratios = [model.config.indexer_compress_ratio if t != "linear_attention" else 0 for t in model.config.layer_types]
        subprocess.run([sys.executable, str(Path(__file__).parent / "set_ratios.py"), str(gguf_file),
                        ",".join(map(str, ratios))], check=True, env=env)

    rng = np.random.default_rng(a.seed)
    toks = rng.integers(1000, 200000, size=a.n_tokens).astype(np.int64)
    toks[100] = EOS  # an EOS inside the sequence resets the n-gram window
    with torch.no_grad():
        ref = model(torch.from_numpy(toks)[None]).logits[0].float().numpy()
    (OUT / "tokens.txt").write_text(" ".join(map(str, toks)))

    ok = True
    for b in a.batch:
        out = OUT / f"e8-b{b}.f32"
        subprocess.run([a.eightfer, "logits", str(gguf_file), "--tokens", str(OUT / "tokens.txt"), "--out", str(out),
                        "--batch", str(b), "--kv", "f32"], check=True, stdout=subprocess.DEVNULL)
        e8 = np.fromfile(out, dtype=np.float32).reshape(ref.shape)
        out.unlink()  # 250 MB per run
        rel = np.linalg.norm(e8 - ref) / np.linalg.norm(ref)
        lp_r = torch.log_softmax(torch.from_numpy(ref).double(), -1)
        lp_e = torch.log_softmax(torch.from_numpy(e8).double(), -1)
        kld = (lp_r.exp() * (lp_r - lp_e)).sum(-1)
        top1 = (ref.argmax(-1) == e8.argmax(-1)).mean() * 100
        # first position with a large error, to localise a mismatch (sparse attention starts at token 16)
        per_pos = np.linalg.norm(e8 - ref, axis=-1) / np.linalg.norm(ref, axis=-1)
        bad = np.nonzero(per_pos > 1e-2)[0]
        good = rel < 2e-3 and kld.max() < 1e-4 and top1 >= 99.0
        ok &= good
        print(f"batch {b:4d}: rel.err {rel:.2e}  mean KLD {kld.mean():.2e}  max KLD {kld.max():.2e}  top-1 {top1:.1f}%"
              f"  first bad pos {bad[0] if len(bad) else '-'}  {'OK' if good else 'FAIL'}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
