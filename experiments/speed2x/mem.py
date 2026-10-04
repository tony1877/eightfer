KVB=34816; GB=1e9
B=25.62e9*4.25/8/GB; R=25.62e9*4.5/8/GB
mtp=0.226; head32k=32768*5120*4.25/8/GB; mtpkv=4096*2*4*256*34/32/GB
gdn=48*48*128*128*4/GB+0.006
def vram(ctx):
    it={'B IQ4_XS (all big matrices incl. lm_head)':B,'CUDA context/modules/pool reserve':0.35,'GDN committed state fp32 + conv':gdn,
        'MTP layer IQ4_XS + 32k-row draft head + MTP KV (4k)':mtp+head32k+mtpkv,'R DMA ring 4x32 MB':0.128,
        'compute buffers (verify N<=24, draft <=64 cols, logits, split-KV partials)':0.30,'draft replay inputs + hypothesis transients':0.08}
    if ctx<=16384:
        it['KV q8_0 full (%d tok)'%ctx]=ctx*KVB/GB
    else:
        it['exact recent KV window 4096 tok q8_0']=4096*KVB/GB
        it['drafter hot set 2048 tok/layer + page-mean-key index (128-tok pages)']=2048*KVB/GB+ctx/128*16*4*256*2/GB
        it['KV staging: 2 of 4 KV heads of one layer']=(ctx-4096)*2176/2/GB
    return it
def ram(ctx):
    it={'R Q4_K':R,'host overhead (CUDA runtime host allocs, graphs, tokenizer, threads)':0.40,'pinned staging (x, partials, replay log)':0.15}
    if ctx<=16384: it['token embedding BF16 (full)']=248320*5120*2/GB
    else:
        it['KV q8_0 outside VRAM window']=(ctx-4096)*KVB/GB
        it['token embedding hot rows BF16 (top 10k)']=0.10
    return it
for ctx in (4096,200000):
    v=vram(ctx); r=ram(ctx)
    print('ctx',ctx)
    for k,x in v.items(): print(f'   VRAM {x:6.3f}  {k}')
    sv=sum(v.values()); print(f'   VRAM total {sv:.3f} / 15.5 -> spare {15.5-sv:.3f}')
    for k,x in r.items(): print(f'   RAM  {x:6.3f}  {k}')
    sr=sum(r.values()); print(f'   RAM total {sr:.3f} / 22 -> spare {22-sr:.3f}')
maxctx=(22-R-0.40-0.15-0.10)/KVB*GB+4096
print('max ctx at full speed (RAM) %.0f'%maxctx)
print('pinned if all: R+KV200k %.2f; DMA-rows-only (CPU share>=4.2 GB) %.2f'%(R+(200000-4096)*KVB/GB, R-4.2+(200000-4096)*KVB/GB))
print('262k spill to NVMe: %.2f GB'%((262144-4096)*KVB/GB-(22-R-0.65)))
