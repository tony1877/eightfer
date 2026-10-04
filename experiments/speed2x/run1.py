from m27 import *
import time
for name,hw in (('central',CENTRAL),('pess',PESS),('opt',OPT)):
    for a in (0.85,0.90,0.95):
        d=v0(4096,a,hw,'doc'); b=v0(4096,a,hw,'best')
        print(f"v0 {name} a={a}: doc(250ms) {d[0]:.1f} k={d[1]} | same-HW {b[0]:.1f} k={b[1]} Tv={b[2]:.0f}")
    b2=v0(200000,0.88,hw,'best'); print(f"v0 {name} 200k a=0.88 same-HW (KV streamed, windowed drafter) {b2[0]:.1f} k={b2[1]} Tv={b2[2]:.0f}")
t0=time.time()
for ctx,a,pen in ((4096,0.90,0.0),(200000,0.90,0.02)):
    st,K,Kb=best(ctx,a,CENTRAL,alpha_pen=pen)
    print(ctx,a,'central',K,Kb,{k:round(v,2) for k,v in st.items()})
print(time.time()-t0)
