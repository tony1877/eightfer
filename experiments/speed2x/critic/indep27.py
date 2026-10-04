# Independent re-derivation of the 27B decode numbers (critic). GB=1e9. All hardware inputs [A] as stated in plan 10.2.
import random
GB=1e9
P=25.62e9
B_pure=P*4.25/8/GB
R=P*4.5/8/GB
KVB=16*2*4*256*34/32
# --- stock IQ4_XS GGUF size of the same tensors (llama-quant.cpp rules at 836d571) ---
lm=248320*5120; attn_v=16*5120*1024; ffn_down=64*5120*17408
B_stock_imat = B_pure + lm*(6.5625-4.25)/8/GB + attn_v*(5.5-4.25)/8/GB
B_stock_noimat = B_stock_imat + (8/64)*ffn_down*(5.5-4.25)/8/GB
print(f"B pure IQ4_XS {B_pure:.3f} GB; stock IQ4_XS (imatrix: output Q6_K, attn_v Q5_K) {B_stock_imat:.3f}; stock no-imatrix (+ffn_down first 8 layers Q5_K) {B_stock_noimat:.3f}")
print(f"  R {R:.3f} GB; KV q8_0 {KVB:.0f} B/tok; 200k KV outside 4096 window {(200000-4096)*KVB/GB:.3f} GB")

def Tv(N,ctx,cap=68,D=46,tmac=0.525,h_us=60,fa=50,stage=0.213,Rres=0.0,W=4096,tail=3.0):
    Rram=R-Rres; kv=max(0,ctx-W)*KVB/GB
    c=min(60,tmac*1e12*0.5625/N/GB)
    attn=16*(max(4*N*24*ctx*256/(fa*1e12), ctx*KVB/16/GB/770)+0.05e-3)
    S=257*h_us*1e-6+48*0.05e-3+attn
    if kv>0: S+=16*max(0,kv/16-stage)/D
    rate=min(c+D,cap)
    T=S+max(0,Rram+kv-D*S)/rate
    return T*1e3+tail, S*1e3, (Rram+kv)/T
for ctx in (4096,200000):
    for N in (8,12):
        t,S,bw=Tv(N,ctx,Rres=0.4 if ctx==4096 else 0)
        print(f"Tv ctx {ctx} N {N}: {t:.0f} ms (S {S:.0f} ms, DRAM avg {bw:.1f} GB/s)")
# DRAM-only lower bound per verify and ceiling
for ctx,a in ((4096,0.90),(200000,0.88)):
    kv=max(0,ctx-4096)*KVB/GB
    tmin=(R+kv)/68
    print(f"ctx {ctx}: DRAM floor {tmin*1e3:.0f} ms/verify; ceiling with 1/(1-a)={1/(1-a):.1f} tok/verify -> {1/(1-a)/tmin:.1f} tok/s")

# --- independent Monte Carlo (own code; same assumptions as plan) ---
def run(ctx=4096,alpha=0.90,K=12,Kb=6,mtp=(0.75,0.60,0.50),derate=0.85,gbusy=52.,tpass1=22.3,tpassw=26.0,
        hedges=True,ph=0.30,ae=0.98,det=(0.75,0.15),cov=(0.55,0.40),Tv_ms=217.,Tv_scale=1.0,cycles=40000,seed=5):
    rnd=random.Random(seed)
    ah=(alpha-(1-ph)*ae)/ph
    def draft(npass):
        out=[]
        for _ in range(npass):
            n=1
            for a in mtp:
                if rnd.random()<a: n+=1
                else: break
            for _ in range(n):
                hard=rnd.random()<ph
                out.append((hard, rnd.random()<(det[0] if hard else det[1])))
        return out
    chain=[]; t=0.; tok=0
    T=Tv_ms*Tv_scale
    for _ in range(cycles):
        if not chain:
            while len(chain)<Kb:
                t+=tpass1; chain+=draft(1)
        k=min(len(chain),K); win=chain[:k]; rest=chain[k:]
        hed=[i for i,(h,f) in enumerate(win) if f][:3] if hedges else []
        w=1+len(hed)+(1 if hedges else 0)
        tp=tpassw if w>1 else tpass1
        npass=int(derate*max(0,T-gbusy)/tp)
        leaf=draft(npass)
        j=None
        for i,(h,f) in enumerate(win):
            if rnd.random()>=(ah if h else ae): j=i; break
        if j is None:
            cont=rest+leaf; tok+=k+1
            if cont:
                h,f=cont[0]
                if rnd.random()<(ah if h else ae): chain=cont[1:]
                else: chain=draft(npass) if (hedges and f and rnd.random()<(cov[0] if h else cov[1])) else []
            else: chain=[]
        else:
            tok+=j+1; h=win[j][0]
            chain=draft(npass) if (j in hed and rnd.random()<(cov[0] if h else cov[1])) else []
        t+=T
    return tok/t*1e3
print("indep MC 4k central (K12,Kb6):", round(run(),1))
print("indep MC 4k, best K/Kb:", max((round(run(K=K,Kb=Kb,cycles=15000),1),K,Kb) for K in (10,12,14,16) for Kb in (3,4,6)))
print("indep MC 4k no hedges:", round(run(hedges=False),1))
print("indep MC 4k, verify +10% from GPU contention:", round(run(Tv_scale=1.10),1))
print("indep MC 4k, MTP weak:", round(run(mtp=(0.65,0.5,0.4)),1))
print("indep MC 4k, alpha .88/.85:", round(run(alpha=0.88),1), round(run(alpha=0.85),1))
print("indep MC 4k, no resident R (stock-B VRAM), Tv=", round(Tv(12,4096)[0]), "->", round(run(Tv_ms=Tv(12,4096)[0]),1))
t200,S200,_=Tv(12,200000)
print("200k: Tv", round(t200), "S", round(S200))
print("indep MC 200k central a=.88 (gbusy 78, tpass 22.4/26.1):", round(run(alpha=0.88,Tv_ms=t200,gbusy=78,tpass1=22.4,tpassw=26.1),1))
t200n,S200n,_=Tv(12,200000,stage=0.0)
print("200k with NO KV staging (stock-B VRAM deficit): Tv", round(t200n), "->", round(run(alpha=0.87,Tv_ms=t200n,gbusy=78,tpass1=22.4,tpassw=26.1),1), "(alpha -0.01 hot set dropped)")
