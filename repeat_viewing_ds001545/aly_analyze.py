# Analysis per partly_cloudy/PREREG5.md
import numpy as np, glob, sys
sys.path.insert(0,'/home/claude/len'); from core import *
TOM=['DMPFC','MMPFC','VMPFC','PC','LTPJ','RTPJ']; PAIN=['AMCC','LInsula','RInsula','LS2','RS2','LMFG','RMFG']
D={}
for f in sorted(glob.glob('/home/claude/aly/out/*.npz')):
    s=f.split('/')[-1][:-4]; d=np.load(f); D[s]={'dice':float(d['dice']),'empty':[r for r in TOM+PAIN if int(d[f'{r}_n'])==0]}
    CL={}
    for r in (1,2,3):
        for i in range(6):
            lab=''.join(ch for ch in str(d[f'r{r}_c{i}_label']) if ch.isascii()).strip()
            if lab.startswith('Scramble Fix'): CL['fix']=lab
    D[s]['fixclip']=CL.get('fix')
    for clip in ['Intact A','Scramble Fix']:
        for net,regs in [('tom',TOM),('pain',PAIN)]:
            views=[]
            for r in (1,2,3):
                for i in range(6):
                    if ''.join(ch for ch in str(d[f'r{r}_c{i}_label']) if ch.isascii()).strip().startswith(clip):
                        ser=[zz(d[f'r{r}_c{i}_{g}']) for g in regs if int(d[f'{g}_n'])>0]
                        views.append(zz(np.mean(ser,0)))
            D[s][(clip,net)]=views
subs=sorted(D); print(f"subjects {len(subs)}; Dice range {min(D[s]['dice'] for s in subs):.2f}-{max(D[s]['dice'] for s in subs):.2f}")
from collections import Counter; print("empty ROIs (FOV):",Counter(r for s in subs for r in D[s]['empty']))
def run(clip,net,K,label,select=None,pos_ref=None):
    ps={};fa=[]
    for s in subs:
        peers=[o for o in subs if o!=s and (clip=='Intact A' or D[o]['fixclip']==D[s]['fixclip'])]
        ref=zz(np.mean([np.mean(D[o][(clip,net)],0) for o in peers],0)) if pos_ref is None else zz(np.mean([D[o][(clip,net)][pos_ref] for o in peers],0))
        v=D[s][(clip,net)]; x=zz(np.mean([v[select]] if select is not None else v[:K],0))
        Sg=surr(x,seed=hash((s,clip,net,K,select))%2**32); ps[s]=pval(x,ref,Sg); fa+=[pval(x,m,Sg)[1] for m in mism(ref)]
    c=calib(fa); k=sum(p<c for _,p in ps.values()); lo,hi=wilson(k,len(subs))
    print(f"  {label:22s} {k}/{len(subs)} = {k/len(subs):.0%} [{lo:.0%}-{hi:.0%}]  median r {np.median([o for o,_ in ps.values()]):.2f}  cut {c:.3f}  FA@.05 {np.mean(np.array(fa)<.05):.1%}")
    return {s:p<c for s,(o,p) in ps.items()}
for clip in ['Intact A','Scramble Fix']:
    for net in ['tom','pain']:
        print(f"{clip} / {net}:")
        for K in (1,2,3,6): run(clip,net,K,f"first {K} viewing(s)")
        for j in range(6): run(clip,net,None,f"viewing #{j+1} alone",select=j)

print("\nEXPLORATORY (not preregistered): each viewing alone vs a POSITION-MATCHED reference (others' viewing #j)")
for net in ['tom','pain']:
    print(f"Intact A / {net}:")
    for j in range(6): run('Intact A',net,None,f"viewing #{j+1} vs others #{j+1}",select=j,pos_ref=j)
    print(f"  cross: viewing #1 vs others' #6 / viewing #6 vs others' #1")
    run('Intact A',net,None,"viewing #1 vs others #6",select=0,pos_ref=5); run('Intact A',net,None,"viewing #6 vs others #1",select=5,pos_ref=0)
