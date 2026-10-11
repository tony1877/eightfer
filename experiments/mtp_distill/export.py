"""Writes a copy of the base GGUF with the MTP block (blk.64.*) replaced by a layer trained with mtp.py.

  python export.py trained.pt out.gguf [--type Q8_0] [--deq-pt check.pt]

The big matrices go in as --type (gguf-py can write Q8_0, Q5_0, Q4_0, BF16; not IQ4_XS), the norms as F32; every
other tensor and all metadata are copied unchanged. --deq-pt also saves the matrices as they will be loaded
(quantized, then dequantized), for `mtp.py eval --load` to measure what the quantization costs. Never overwrites the
base GGUF.
"""
import argparse, os, sys
import numpy as np
import torch
import mtp
from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType, GGUFValueType, Keys
from gguf.quants import quantize, dequantize

ap = argparse.ArgumentParser()
ap.add_argument('trained')
ap.add_argument('out')
ap.add_argument('--type', default='Q8_0')
ap.add_argument('--deq-pt')
a = ap.parse_args()
assert os.path.abspath(a.out) != os.path.abspath(mtp.BASE), 'refusing to overwrite the base GGUF'
qt = GGMLQuantizationType[a.type]
sd = {k: v.float().numpy() for k, v in torch.load(a.trained).items()}
P = 'blk.64.'

new, deq = {}, {}
for m in mtp.MATS + mtp.NORMS:
    w = sd[m.replace('.', '_')]
    if m in mtp.MATS:
        q = quantize(w, qt)
        new[P + m + '.weight'] = (q, qt)
        deq[m.replace('.', '_')] = torch.from_numpy(dequantize(q, qt).reshape(w.shape)).to(torch.bfloat16)
    else:
        new[P + m + '.weight'] = (w.astype(np.float32), GGMLQuantizationType.F32)
        deq[m.replace('.', '_')] = torch.from_numpy(w).to(torch.bfloat16)
    print(f'{m}: {w.shape} -> {new[P + m + ".weight"][1].name}', flush=True)
if a.deq_pt:
    torch.save(deq, a.deq_pt)

r = GGUFReader(mtp.BASE)
arch = r.fields[Keys.General.ARCHITECTURE].contents()
wr = GGUFWriter(a.out, arch)
if Keys.General.ALIGNMENT in r.fields:
    wr.data_alignment = r.fields[Keys.General.ALIGNMENT].contents()
for f in r.fields.values():
    if f.name == Keys.General.ARCHITECTURE or f.name.startswith('GGUF.'):
        continue
    vt = f.types[0]
    wr.add_key_value(f.name, f.contents(), vt, sub_type=f.types[-1] if vt == GGUFValueType.ARRAY else None)
missing = set(new)
for t in r.tensors:
    if t.name in new:
        d, ty = new[t.name]
        wr.add_tensor_info(t.name, d.shape, d.dtype, d.nbytes, ty)
        missing.discard(t.name)
    else:
        wr.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
assert not missing, missing
wr.write_header_to_file()
wr.write_kv_data_to_file()
wr.write_ti_data_to_file()
for t in r.tensors:
    wr.write_tensor_data(new[t.name][0] if t.name in new else t.data, tensor_endianess=r.endianess)
wr.close()
print(f'wrote {a.out}: {os.path.getsize(a.out) / 1e9:.2f} GB (base {os.path.getsize(mtp.BASE) / 1e9:.2f} GB)')
