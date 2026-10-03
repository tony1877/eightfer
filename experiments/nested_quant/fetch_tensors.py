# Range-fetches three BF16 tensors from Qwen3.8-27B (one safetensors shard) without downloading the shard.
import json, os, struct, subprocess

OUT = "w"
URL = "https://huggingface.co/Qwen/Qwen3.8-27B/resolve/main/model-00004-of-00018.safetensors"
WANT = [
    "model.language_model.layers.10.mlp.down_proj.weight",
    "model.language_model.layers.10.linear_attn.in_proj_qkv.weight",
    "model.language_model.layers.11.self_attn.q_proj.weight",
]

def rng(a, b, out):
    subprocess.run(["curl", "-sS", "-L", "--max-time", "600", "-r", f"{a}-{b}", "-o", out, URL], check=True)

os.makedirs(OUT, exist_ok=True)
rng(0, 7, f"{OUT}/hlen.bin")
n = struct.unpack("<Q", open(f"{OUT}/hlen.bin", "rb").read())[0]
rng(8, 8 + n - 1, f"{OUT}/hdr.json")
hdr = json.load(open(f"{OUT}/hdr.json"))
for k in WANT:
    t = hdr[k]
    a, b = t["data_offsets"]
    print(k, t["dtype"], t["shape"], (b - a) / 1e6, "MB", flush=True)
    rng(8 + n + a, 8 + n + b - 1, f"{OUT}/" + k.split("layers.")[1] + ".bin")
