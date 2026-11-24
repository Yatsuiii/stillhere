import numpy as np, json, csv, glob, sys
sys.path.insert(0,'/home/claude/len'); from core import *
exec(open('/home/claude/saxe/analyze.py').read().split("adults=[")[0])
A=[s for s in S if S[s]['adult']]; order=['3yo','4yo','5yo','7yo','8-12yo','Adult']
OUT={}
def ref(s,net='tom'): return zz(np.mean([S[a][net] for a in A if a!=s],0))
# ---- 1+2: ds000228 by length
LENS=[30,60,90,120,158]; per={}; fa={L:[] for L in LENS}
for s in S:
    r=ref(s); x=S[s]['tom']; per[s]={}
    for L in LENS:
        ps=[]
        for st in (range(0,158-L+1,10) if L<158 else [0]):
            xs,rs=x[st:st+L],r[st:st+L]; Sg=surr(zz(xs),seed=hash((s,L,st))%2**32)
            ps.append(pval(xs,rs,Sg)[1]); fa[L]+= [pval(xs,m,Sg)[1] for m in mism(rs)]
        per[s][L]=ps
cut={L:calib(fa[L]) for L in LENS}
print("calibrated cutoffs:",{L:round(c,4) for L,c in cut.items()}, " FA at .05:",{L:round(np.mean(np.array(fa[L])<.05),3) for L in LENS})
rng=np.random.default_rng(0)
print("\n1. full length (5.3 min), detection by group at calibrated cutoff [Wilson 95%] | at p<.05")
for g in order:
    G=[s for s in S if S[s]['grp']==g]; k=sum(per[s][158][0]<cut[158] for s in G); k5=sum(per[s][158][0]<.05 for s in G); lo,hi=wilson(k,len(G))
    print(f"  {g:7s} {k}/{len(G)} = {k/len(G):.0%} [{lo:.0%}-{hi:.0%}] | {k5}/{len(G)}")
miss=[s for s in A if per[s][158][0]>=cut[158]]
print("  missed adults:")
for s in miss:
    d=np.load(f'/home/claude/saxe/ts/{s}.npz'); R=d['R']; spk=sum(1 for j in range(R.shape[1]) if (R[:,j]!=0).sum()==1 and np.flatnonzero(R[:,j])[0]>=10)
    po,pp=pval(S[s]['pain'],ref(s,'pain'))
    print(f"   {s}: ToM r={pval(S[s]['tom'],ref(s))[0]:.2f} p={per[s][158][0]:.3f}; pain r={po:.2f} p={pp:.3f}; spikes(TR11-168)={spk}; VMPFC vox={int(d['VMPFC_nvox'])}")
print("\n2. by length: mean share of windows detected (calibrated) [bootstrap 95% CI over people]")
print("   group   "+"  ".join(f"{L*2/60:.0f}-{L*2/60:.1f}min".split('-')[1] for L in LENS))
for g in order:
    G=[s for s in S if S[s]['grp']==g]; row=[]
    for L in LENS:
        v=np.array([np.mean(np.array(per[s][L])<cut[L]) for s in G]); bs=[rng.choice(v,len(v)).mean() for _ in range(2000)]
        row.append(f"{v.mean():.0%}[{np.quantile(bs,.025):.0%}-{np.quantile(bs,.975):.0%}]")
    print(f"   {g:7s} "+"  ".join(row))
OUT['ds']={'cut':cut,'per':{s:{L:list(map(float,v)) for L,v in d.items()} for s,d in per.items()},'miss':miss}
# ---- 4: split half personal ROIs
TOM=['DMPFC','MMPFC','VMPFC','PC','LTPJ','RTPJ']
def Q(s,Y):
    d=np.load(f'/home/claude/saxe/ts/{s}.npz'); R=d['R']; X=np.column_stack([R,np.ones(168),np.linspace(-1,1,168)]); return Y-X@np.linalg.lstsq(X,Y,rcond=None)[0]
H=[np.arange(0,79),np.arange(79,158)]; sh={}; fah={'per':[],'grp':[]}
for s in S:
    v=np.load(f'/home/claude/saxe/vox/{s}.npz'); r=ref(s); sh[s]={}
    vox={g:Q(s,v[g].T).T[:,10:] for g in TOM}; vox={g:(Y-Y.mean(1,keepdims=True))/Y.std(1,keepdims=True) for g,Y in vox.items()}
    for i in (0,1):
        sel,tst=H[i],H[1-i]; per_=[];grp_=[]
        for g in TOM:
            Y=vox[g]; c=(zz(r[sel])*((Y[:,sel]-Y[:,sel].mean(1,keepdims=True))/Y[:,sel].std(1,keepdims=True))).mean(1)
            per_.append(zz(Y[np.argsort(c)[-50:]][:,tst].mean(0))); grp_.append(zz(Y[v[g+'_insphere']][:,tst].mean(0)))
        for nm,ser in [('per',zz(np.mean(per_,0))),('grp',zz(np.mean(grp_,0)))]:
            Sg=surr(ser,seed=hash((s,i,nm))%2**32); sh[s][(nm,1-i)]=pval(ser,r[tst],Sg); fah[nm]+=[pval(ser,m,Sg)[1] for m in mism(r[tst],5)]
ch={nm:calib(fah[nm]) for nm in fah}
print("\n4. split-half (half length), calibrated cutoffs",{k:round(v,4) for k,v in ch.items()})
for g in order:
    G=[s for s in S if S[s]['grp']==g]; f=lambda nm: np.mean([sh[s][(nm,h)][1]<ch[nm] for s in G for h in (0,1)])
    print(f"   {g:7s} sphere {f('grp'):.0%}  personal {f('per'):.0%}")
for s in miss: print("   ",s,{f"{nm}h{h}":f"{sh[s][(nm,h)][0]:.2f}{'*' if sh[s][(nm,h)][1]<ch[nm] else ''}" for nm in ('grp','per') for h in (0,1)})
json.dump(OUT,open('/home/claude/len/final_ds.json','w'),default=str)
