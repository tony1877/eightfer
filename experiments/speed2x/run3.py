from m27 import *
def go(ctx,a,hw,**kw):
    pen = kw.pop('alpha_pen', 0.0 if ctx<=16384 else (0.01 if ctx<=65536 else 0.02))
    st,K,Kb=best(ctx,a,hw,alpha_pen=pen,**kw); return st
# context curve, central, a=0.90
curve={4096:dict(W=4096,R_res=0.4),32768:dict(W=12288),65536:dict(W=8192),131072:dict(W=4096),200000:dict(W=4096)}
for ctx,kw in curve.items():
    st=go(ctx,0.90,CENTRAL,**kw)
    v=v0(ctx,0.90-(0 if ctx<=16384 else (0.01 if ctx<=65536 else 0.02)),CENTRAL)
    print(f"curve ctx {ctx:6d}: new {st['tok_s']:5.1f} tok/s (Tv {st['mean_Tv']:.0f}, N {st['mean_N']:.1f}, tok/cyc {st['tok_per_cycle']:.2f}) | v0 same-HW {v[0]:5.1f}")
print()
for name,hw in (('pess',PESS),('central',CENTRAL),('opt',OPT)):
    for a in (0.85,0.90,0.95):
        s4=go(4096,a,hw,R_res=0.4); s2=go(200000,a,hw)
        d=v0(4096,a,hw,'best'); d2=v0(200000,a-0.02,hw,'best')
        print(f"{name:7s} a={a:.2f}: 4k {s4['tok_s']:5.1f} (v0 same-HW {d[0]:5.1f}, x{s4['tok_s']/d[0]:.2f}; x{s4['tok_s']/14:.2f} of 14) | 200k {s2['tok_s']:5.1f} (v0 same-ctx {d2[0]:4.1f}, x{s2['tok_s']/d2[0]:.2f}; x{s2['tok_s']/14:.2f} of 14)")
print()
for a in (0.91,0.92,0.93):
    print(f"central a={a}: 4k {go(4096,a,CENTRAL,R_res=0.4)['tok_s']:.1f}")
print("200k a=0.90 drafter penalty 0.04:", round(go(200000,0.90,CENTRAL,alpha_pen=0.04)['tok_s'],1))
print("200k full-layer staging:", round(go(200000,0.90,CENTRAL,stage=0.43)['tok_s'],1))
kvs=((16000*34816)+(200000-16000)*18432)/((200000-4096)*34816)
print("LOSSY 200k far-KV q4_0 (kvscale %.2f):"%kvs, round(go(200000,0.90,CENTRAL,kvscale=kvs)['tok_s'],1))
print("262k with 2.05 GB NVMe spill (~+9%% Tv approx via kvscale):", round(go(262144,0.90,CENTRAL,kvscale=1.0+2.05/((262144-4096)*34816/1e9))['tok_s'],1))
