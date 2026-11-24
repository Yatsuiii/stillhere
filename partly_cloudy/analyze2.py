import numpy as np, csv, glob, json
W='/home/claude/saxe'; rng=np.random.default_rng(1)
TOM=['DMPFC','MMPFC','VMPFC','PC','LTPJ','RTPJ']; PAIN=['AMCC','LInsula','RInsula','LS2','RS2','LMFG','RMFG']
P={r['participant_id']:r for r in csv.DictReader(open(f'{W}/participants.tsv'),delimiter='\t')}
def z(x,ax=-1): return (x-x.mean(ax,keepdims=True))/x.std(ax,keepdims=True)
S={}
for f in sorted(glob.glob(f'{W}/ts/*.npz')):
    s=f.split('/')[-1][:-4]; d=np.load(f); R=d['R']; n=168
    X=np.column_stack([R,np.ones(n),np.linspace(-1,1,n)]); Q=lambda Y: Y-X@np.linalg.lstsq(X,Y,rcond=None)[0]
    spk=np.zeros(n,bool)
    for j in range(R.shape[1]):
        if (R[:,j]!=0).sum()==1: spk[R[:,j]!=0]=True
    v=np.load(f'{W}/vox/{s}.npz')
    S[s]=dict(adult=P[s]['Child_Adult']=='adult',grp=P[s]['AgeGroup'],spk=spk[10:],
        tom=z(np.mean([z(Q(d[r])[10:]) for r in TOM],0)),
        vox={r:z(Q(v[r].T).T[:,10:]) for r in TOM}, sph={r:v[r+'_insphere'] for r in TOM})
A=[s for s in S if S[s]['adult']]
def ref(s): return z(np.mean([S[a]['tom'] for a in A if a!=s],0))
def test(x,r,n=1000):
    o=np.corrcoef(x,r)[0,1]; T=len(x)
    nl=np.array([np.corrcoef(np.roll(x,k),r)[0,1] for k in rng.integers(min(8,T//4),T-min(8,T//4),n)]); return o,(1+(nl>=o).sum())/(n+1)
H=[np.arange(0,79),np.arange(79,158)]
order=['3yo','4yo','5yo','7yo','8-12yo','Adult']
out={}
for s,v in S.items():
    r=ref(s); o={}
    # A: split half
    for mode,rr in [('true',r),('rev',r[::-1])]:
        for i in (0,1):
            sel,tst=H[i],H[1-i]; per=[];grp=[]
            for reg in TOM:
                Y=v['vox'][reg]; c=(z(Y[:,sel])*z(rr[sel])).mean(1)
                per.append(z(Y[np.argsort(c)[-50:]][:,tst].mean(0))); grp.append(z(Y[v['sph'][reg]][:,tst].mean(0)))
            o[f'A_{mode}_per_h{1-i}']=test(z(np.mean(per,0)),rr[tst]); o[f'A_{mode}_grp_h{1-i}']=test(z(np.mean(grp,0)),rr[tst])
    # B: 20 mismatched refs
    o['B']=[test(v['tom'],np.roll(r[::-1],k),300)[1] for k in np.linspace(8,150,20).astype(int)]
    # C: scrub
    keep=~v['spk']; o['C']=test(z(v['tom'][keep]),z(r[keep])) if keep.sum()>40 else (np.nan,1.0); o['C_kept']=int(keep.sum())
    # D: lag
    o['D']=int(np.argmax([np.corrcoef(np.roll(v['tom'],L),r)[0,1] for L in range(-3,4)])-3)
    out[s]=o
json.dump({s:{k:(list(map(float,v)) if isinstance(v,(tuple,list)) else v) for k,v in o.items()} for s,o in out.items()},open(f'{W}/results2.json','w'))
prev={r['sub']:r for r in json.load(open(f'{W}/results.json'))}
miss=[s for s in S if S[s]['adult'] and prev[s]['tom_p']>=.05]
print("A. split-half detection (both halves pooled, % of 2n tests): group sphere | personal top-50 | personal vs REVERSED ref (selection false alarms)")
for g in order:
    G=[s for s in S if S[s]['grp']==g]; f=lambda m,t: np.mean([out[s][f'A_{m}_{t}_h{h}'][1]<.05 for s in G for h in (0,1)])
    print(f"  {g:7s} n={len(G):3d}  grp {f('true','grp'):.0%}  per {f('true','per'):.0%}  per-rev {f('rev','per'):.0%}  grp-rev {f('rev','grp'):.0%}")
print("  missed adults:")
for s in miss: print("   ",s,{k:round(out[s][k][0],2) for k in ['A_true_grp_h0','A_true_grp_h1','A_true_per_h0','A_true_per_h1']},'p',[round(out[s][k][1],3) for k in ['A_true_per_h0','A_true_per_h1']])
print("B. false alarms over 20 mismatched refs per person:")
for g in order:
    G=[s for s in S if S[s]['grp']==g]; print(f"  {g:7s} {np.mean([p<.05 for s in G for p in out[s]['B']]):.1%}")
print("C. scrubbed detection:")
for g in order:
    G=[s for s in S if S[s]['grp']==g]; print(f"  {g:7s} {np.mean([out[s]['C'][1]<.05 for s in G]):.0%}  (median kept TRs {np.median([out[s]['C_kept'] for s in G]):.0f})")
print("  missed adults scrubbed:",[(s,round(out[s]['C'][0],2),round(out[s]['C'][1],3)) for s in miss])
print("D. best lag distribution:")
from collections import Counter
for g in order: print(f"  {g:7s}",sorted(Counter(out[s]['D'] for s in S if S[s]['grp']==g).items()))
print("  missed adults lag:",[(s,out[s]['D']) for s in miss])
