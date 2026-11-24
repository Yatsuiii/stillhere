import numpy as np, json, sys
from math import sqrt
exec(open('/home/claude/saxe/analyze.py').read().split("adults=[")[0])
A=[s for s in S if S[s]['adult']]; rng=np.random.default_rng(5)
def zz(x): return (x-x.mean())/x.std()
def ptest(x,r,n=500):
    o=np.corrcoef(x,r)[0,1]; T=len(x); ks=rng.integers(5,T-5,n)
    nl=np.array([np.corrcoef(np.roll(x,k),r)[0,1] for k in ks]); return (1+(nl>=o).sum())/(n+1)
def wilson(k,n,z=1.96):
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*sqrt(p*(1-p)/n+z*z/(4*n*n))/d; return c-h,c+h
LENS=[30,60,90,120,158]; order=['3yo','4yo','5yo','7yo','8-12yo','Adult']
res={}
for s in S:
    ref=zz(np.mean([S[a]['tom'] for a in A if a!=s],0)); x=S[s]['tom']; res[s]={}
    for L in LENS:
        starts=list(range(0,158-L+1,10)) if L<158 else [0]; det=[];fa=[]
        for st in starts:
            xs=x[st:st+L]; rs=ref[st:st+L]; det.append(ptest(xs,rs)<.05)
            for k in np.linspace(3,L-3,5).astype(int): fa.append(ptest(xs,np.roll(rs[::-1],k))<.05)
        res[s][L]=(np.mean(det),np.mean(fa))
json.dump({s:{str(L):v for L,v in d.items()} for s,d in res.items()},open('/home/claude/len/len_ds000228.json','w'))
print("ds000228: mean detection per person over windows (FA in brackets); 95% CI is Wilson on people detected in >=50% of windows")
print("group  "+"  ".join(f"{L*2/60:.1f}min" for L in LENS))
for g in order:
    G=[s for s in S if S[s]['grp']==g]
    print(f"{g:7s}"+"  ".join(f"{np.mean([res[s][L][0] for s in G]):.0%}({np.mean([res[s][L][1] for s in G]):.0%})" for L in LENS))
