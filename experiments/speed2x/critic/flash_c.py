# Critic re-derivation of Flash-Next decode. GB=1e9. Inputs as in plan 10.4 unless noted.
GB=1e9; NE=24576
EB=(2*2560*640*4.25+2560*640*4.5)/8/GB   # 2.662 MB
NONEXP=(36*57.9e6+12*51.4e6+48*(6.2e6+13.2e6)+32.8e6+636e6)*8.5/8/GB
KVT=12*2*2*256*34/32
print(f"expert B {EB*1e3:.3f} MB  non-expert Q8_0 {NONEXP:.2f} GB  KV {KVT:.0f} B/tok -> 200k {200000*KVT/GB:.2f} GB")
def cap(ctx,ram=22.0,design='new'):
    idx=ctx*12*128*2/4/GB
    vfix=0.35+NONEXP+0.12+0.40+1.0
    if design=='v0':
        v=15.5-vfix-ctx*KVT/GB-idx; r=ram-0.9-1.27
    else:
        v=15.5-vfix-idx-(0.10 if ctx>16384 else ctx*KVT/GB); r=ram-0.95-(ctx*KVT/GB if ctx>16384 else 0)
    nv,nr=int(v/EB),int(r/EB); return nv,nr,(nv+nr)/NE
def tps(ctx,bw,u,h,chain,ov=0.25):
    tio=480*(1-h)*EB*1.05/(bw*u)*1e3
    tch=chain+(1.5 if ctx>16384 else 0)
    t=max(tio,tch)+ov*min(tio,tch)
    return 1e3/t,tio,tch
for ctx in (4096,200000):
    for d in ('v0','new'):
        nv,nr,c=cap(ctx,design=d)
        r=[tps(ctx,bw,0.80 if d=='v0' else 0.85,c,27.0)[0] for bw in (13.0,6.8)]
        print(f"{d:3s} ctx {ctx}: VRAM {nv} RAM {nr} cached {c:.3f} -> T2 {r[0]:.1f}  T3/single {r[1]:.1f}")
# the task's own single-drive baseline formula, no chain penalty
for h in (0.5,):
    for u in (0.75,1.0):
        print(f"task single-drive v0, h={h}, u={u}: {7*u/(480*(1-h)*2.61e-3):.1f} tok/s (I/O ceiling only)")
# 64 GB RAM: chain must include CPU reads of RAM hits
print("--- 64 GB RAM option (54 GB usable) ---")
for ctx in (4096,200000):
    nv,nr,c=cap(ctx,ram=54.0)
    ramhits=480*nr/NE; gb=ramhits*EB
    base_chain=27.0-48*0.17      # plan chain without the CPU GEMV share for 154 RAM hits
    for bwcpu,label in ((50,'CPU only ~50 GB/s'),(68,'CPU+GPU zero-copy split, 68 cap')):
        ch=base_chain+gb/bwcpu*1e3
        r2,tio,tch=tps(ctx,13.0,0.85,c,ch); r3,_,_=tps(ctx,6.8,0.85,c,ch)
        print(f"ctx {ctx}: cached {c:.3f}, RAM-hit experts/token {ramhits:.0f} = {gb:.2f} GB -> chain {ch:.1f} ms ({label}): T2 {r2:.1f}, T3 {r3:.1f}  (plan: chain 32 ms -> {'29.8/28.6' if ctx==4096 else '27.5/25.6'})")
# serialization: I/O for layer l cannot start before router l without lookahead
print("--- no-lookahead per-layer serialization (I/O and chain add) ---")
for ctx in (4096,200000):
    for d,u in (('v0',0.80),('new',0.85)):
        nv,nr,c=cap(ctx,design=d)
        for bw,name in ((13.0,'T2'),(6.8,'T3')):
            r,tio,tch=tps(ctx,bw,u,c,27.0,ov=1.0)
            print(f"  {d} ctx {ctx} {name}: {r:.1f} tok/s (io {tio:.0f} + chain {tch:.0f})")
