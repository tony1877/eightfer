"""Sets qwen4exp.attention.compress_ratios in a GGUF in place: set_ratios.py <file.gguf> <r0,r1,...>"""
import sys

import gguf

rd = gguf.GGUFReader(sys.argv[1], "r+")
field = rd.fields["qwen4exp.attention.compress_ratios"]
ratios = [int(x) for x in sys.argv[2].split(",")]
for i, di in enumerate(field.data):
    field.parts[di][0] = ratios[i]
rd.data.flush()
