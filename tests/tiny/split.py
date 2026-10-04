"""M3 check on the tiny model: logits of the packed base alone and base + residual against the F32 original.
Run tests/tiny/run.py and `eightfer pack` on tests/tiny/out/hf first (see tests/tiny/README.md)."""
import subprocess, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests" / "tiny" / "out"
E8 = str(ROOT / "build" / "bin" / "eightfer.exe")


def logits(gguf, extra, tag):
    out = OUT / f"split-{tag}.f32"
    subprocess.run([E8, "logits", str(gguf), "--tokens", str(OUT / "tokens.txt"), "--out", str(out), "--batch", "64",
                    *extra], check=True, stdout=subprocess.DEVNULL)
    return np.fromfile(out, dtype=np.float32)


def kld(ref, x, nv):
    ref = ref.reshape(-1, nv).astype(np.float64)
    x = x.reshape(-1, nv).astype(np.float64)
    lr = ref - ref.max(-1, keepdims=True)
    lr -= np.log(np.exp(lr).sum(-1, keepdims=True))
    lx = x - x.max(-1, keepdims=True)
    lx -= np.log(np.exp(lx).sum(-1, keepdims=True))
    return (np.exp(lr) * (lr - lx)).sum(-1).mean(), (ref.argmax(-1) == x.argmax(-1)).mean() * 100


def main():
    nv = 248320
    ok = True
    for dev, extra in (("cpu", []), ("gpu", ["--gpu-layers", "99"])):
        ref = logits(OUT / "tiny-f32.gguf", extra, f"f32-{dev}")
        b = logits(OUT / "tiny.base.gguf", extra, f"b-{dev}")
        br = logits(OUT / "tiny.base.gguf", extra + ["--res", str(OUT / "tiny.res.gguf")], f"br-{dev}")
        kb, tb = kld(ref, b, nv)
        kbr, tbr = kld(ref, br, nv)
        print(f"{dev}: base KLD {kb:.2e} top-1 {tb:.1f}% | base+res KLD {kbr:.2e} top-1 {tbr:.1f}% | "
              f"improvement {kb / max(kbr, 1e-30):.0f}x")
        ok &= kbr < kb / 10
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
