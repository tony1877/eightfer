from m27 import *
def show(lbl, ctx, a, hw=CENTRAL, **kw):
    pen = kw.pop('alpha_pen', 0.02 if ctx>8192 else 0.0)
    st,K,Kb=best(ctx,a,hw,alpha_pen=pen,**kw)
    print(f"{lbl:48s} ctx {ctx:6d} a {a:.2f}: {st['tok_s']:5.1f} tok/s  K {K:2d} Kb {Kb} tok/cyc {st['tok_per_cycle']:.2f} bub {st['bubble_frac']:.2f} N {st['mean_N']:.1f} Tv {st['mean_Tv']:.0f} cyc {st['ms_per_cycle']:.0f}")
    return st['tok_s']
print("== waterfall (central HW, a=0.90 @4k, 0.88 @200k) ==")
for ctx in (4096,200000):
    show("1 dual-path verify, serial, B-only drafter",ctx,0.90,pipeline=False,use_mtp=False,m_hedge=0)
    show("2 + MTP-staged drafter (serial)",ctx,0.90,pipeline=False,use_mtp=True,m_hedge=0)
    show("3 + draft-during-verify, leaf only",ctx,0.90,pipeline=True,use_mtp=True,m_hedge=0)
    show("4 + hedges m=3 (default plan)",ctx,0.90,pipeline=True,use_mtp=True,m_hedge=3)
    show("4b hedges m=6",ctx,0.90,pipeline=True,use_mtp=True,m_hedge=6)
    if ctx==4096:
        show("5 + 0.4 GB resident R (VRAM slack)",ctx,0.90,m_hedge=3,R_res=0.4)
    else:
        show("5 + full-layer KV staging (needs +0.21 GB VRAM)",ctx,0.90,m_hedge=3,stage=0.43)
    show("  ablation: no CPU slice (DMA-only R)",ctx,0.90,m_hedge=3,cpu_on=False)
    show("  ablation: no MTP",ctx,0.90,m_hedge=3,use_mtp=False)
    show("  ablation: no pipelining (hedges impossible)",ctx,0.90,pipeline=False,m_hedge=0)
print("== hedge variants ==")
for ctx in (4096,200000):
    show("hedges m=3, no bonus hedge",ctx,0.90,m_hedge=3,bonus_hedge=False)
    show("hedges m=3, 2 roots cov .70/.50",ctx,0.90,m_hedge=3,roots=2,cov=(0.70,0.50))
    show("hedges m=3, skeptical cov .40/.30, det .65/.20",ctx,0.90,m_hedge=3,cov=(0.40,0.30),det=(0.65,0.20))
