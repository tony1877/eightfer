T=200000
# 27B
lin=2*2*24.35e9*T            # B and R as two int8 GEMMs, lm_head only for last token
att=16*24*256*4*T*T/2        # causal, QK + PV
for tops,tf,gdn in ((250,80,3.0),(180,55,7.7),(150,45,7.7)):
    l=lin/(tops*1e12); a=att/(tf*1e12); tot=l+a+gdn+5
    print(f"27B prefill: linear {lin:.3g} ops @{tops} TOPS = {l:.0f} s; attn {att:.3g} FLOP @{tf} TF = {a:.0f} s; GDN {gdn} s; +5 s overhead => {tot:.0f} s")
C=2048; nchunk=T/C
print(f"  per chunk ({C}): R stream {14.41/46:.2f} s vs compute ~{(lin/(180e12)+att/(55e12))/nchunk:.2f} s; KV re-read total {T*T/2/C*34816/1e9:.0f} GB = {T*T/2/C*34816/1e9/46:.1f} s over PCIe (hidden)")
print(f"  prefix-cache restore: {(T*34816/1e9+0.157):.2f} GB at 6.5 GB/s = {(T*34816/1e9+0.157)/6.5:.1f} s")
# Flash
act=(4.30e9-0.636e9+11*4.92e6*48)
lin=2*act*T
idx=12*T*4*128*(T/4/2)*2
sp_flop=12*T*24*256*4*2048
sp_bytes=12*T*2051*1088
dense=12*24*256*4*T*T/2
for tops,io_bw,gdn,label in ((200,13*0.85,2.0,'opt'),(160,13*0.85,5.8,'central T2'),(130,6.8*0.8,5.8,'pess T3')):
    l=lin/(tops*1e12); i=idx/40e12+2.0; s=max(sp_flop/50e12, sp_bytes/770e9/2)
    io=2*44/io_bw
    comp=l+i+s+gdn+5
    print(f"Flash prefill [{label}]: matmul {lin:.3g} ops @{tops} = {l:.1f} s; indexer {idx:.3g} FLOP + top-k = {i:.1f} s; sparse attn {s:.1f} s (gather {sp_bytes/1e12:.2f} TB, half reused); GDN {gdn}; +5 => compute {comp:.0f} s; expert I/O {io:.0f} s overlapped => {max(comp,io)+3:.0f} s")
print(f"  stock-ggml dense-FA+mask QSA: {dense:.3g} FLOP @67 TF = {dense/67e12:.0f} s (why a custom block-sparse kernel is required)")
print(f"  n-gram prefill reads: {T*16/1e6:.1f}M x 4 KiB: NVMe @500k IOPS {T*16/5e5:.1f} s; SATA @2x90k {T*16/1.8e5:.1f} s")
