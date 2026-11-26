# PREREG7 (sha256 d660ffb09c074723...). Run once.
import numpy as np
from scipy import stats
exec(open('/home/claude/saxe/analyze.py').read().split("adults=[")[0])
rng=np.random.default_rng(20261126)
A=[s for s in S if S[s]['adult']]
H=[slice(0,79),slice(79,158)]
def sc(s,net,sl=slice(None)):
    ref=np.mean([S[a][net] for a in A],0)
    return np.arctanh(np.corrcoef(S[s][net][sl],ref[sl])[0,1])
def num(x):
    try: return float(x)
    except: return np.nan
def resid(y,X):
    X=np.column_stack([np.ones(len(y))]+X); return y-X@np.linalg.lstsq(X,y,rcond=None)[0]
def prho(x,y,cov):
    rx=resid(stats.rankdata(x),[stats.rankdata(c) for c in cov]); ry=resid(stats.rankdata(y),[stats.rankdata(c) for c in cov])
    return np.corrcoef(rx,ry)[0,1],rx,ry
def full_test(subs,beh,net,extra=False,nb=10000):
    x=np.array([sc(s,net) for s in subs]); y=np.array([num(P[s][beh]) for s in subs]); age=np.array([S[s]['age'] for s in subs])
    cov=[age]+([np.array([S[s]['spikes'] for s in subs],float)] if extra else [])
    r,rx,ry=prho(x,y,cov)
    perm=np.array([np.corrcoef(rx,rng.permutation(ry))[0,1] for _ in range(nb)])
    p=(1+(np.abs(perm)>=abs(r)).sum())/(nb+1)
    bs=[]
    for _ in range(nb):
        i=rng.integers(0,len(subs),len(subs)); bs.append(prho(x[i],y[i],[c[i] for c in cov])[0])
    return r,p,np.nanquantile(bs,[.025,.975]),x,y,age,cov
kids35=[s for s in S if S[s]['grp'] in ('3yo','4yo','5yo')]
kids=[s for s in S if not S[s]['adult']]
print(f"P1 sample n={len(kids35)}; S1 sample n={len(kids)}")
out={}
for lab,subs,beh in [('P1 3-5yo FB_Composite',kids35,'FB_Composite'),('S1 3-12yo Booklet-NOFB',kids,'ToM Booklet-Matched-NOFB')]:
    for net in ['tom','pain']:
        for ex in [False,True]:
            r,p,ci,*_=full_test(subs,beh,net,ex)
            k=f"{lab} | {net} | {'age+motion' if ex else 'age'}"; out[k]=(r,p,ci)
            print(f"{k}: partial rho {r:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]  p={p:.4f}")
# S2 specificity: bootstrap of difference tom - pain (age only)
for lab,subs,beh in [('P1',kids35,'FB_Composite'),('S1',kids,'ToM Booklet-Matched-NOFB')]:
    xt=np.array([sc(s,'tom') for s in subs]); xp=np.array([sc(s,'pain') for s in subs]); y=np.array([num(P[s][beh]) for s in subs]); age=np.array([S[s]['age'] for s in subs])
    d=[]
    for _ in range(10000):
        i=rng.integers(0,len(subs),len(subs)); d.append(prho(xt[i],y[i],[age[i]])[0]-prho(xp[i],y[i],[age[i]])[0])
    obs=prho(xt,y,[age])[0]-prho(xp,y,[age])[0]; lo,hi=np.nanquantile(d,[.025,.975])
    print(f"S2 {lab} ToM minus pain: {obs:+.3f} [{lo:+.3f}, {hi:+.3f}]")
# S4 reliability ceiling in P1 sample
def rel(subs):
    h1=[];h2=[]
    for g in sorted(set(S[s]['grp'] for s in subs)):
        G=[s for s in subs if S[s]['grp']==g]; a=np.array([sc(s,'tom',H[0]) for s in G]); b=np.array([sc(s,'tom',H[1]) for s in G])
        h1+=list(a-a.mean()); h2+=list(b-b.mean())
    r=np.corrcoef(h1,h2)[0,1]; return 2*r/(1+r)
x=np.array([sc(s,'tom') for s in kids35]); y=np.array([num(P[s]['FB_Composite']) for s in kids35]); age=np.array([S[s]['age'] for s in kids35])
rb=rel(kids35); robs=prho(x,y,[age])[0]
bs_rel=[];bs_dis=[]
for _ in range(5000):
    i=rng.integers(0,len(kids35),len(kids35)); sub=[kids35[j] for j in i]
    try:
        h1=[];h2=[]
        for g in sorted(set(S[s]['grp'] for s in sub)):
            G=[s for s in sub if S[s]['grp']==g]; a=np.array([sc(s,'tom',H[0]) for s in G]); b=np.array([sc(s,'tom',H[1]) for s in G]); h1+=list(a-a.mean()); h2+=list(b-b.mean())
        r=np.corrcoef(h1,h2)[0,1]; rr=2*r/(1+r)
    except Exception: continue
    bs_rel.append(rr); ro=prho(x[i],y[i],[age[i]])[0]
    bs_dis.append(ro/np.sqrt(rr*.71) if rr>0 else np.nan)
lo,hi=np.nanquantile(bs_rel,[.025,.975])
print(f"S4 brain reliability (3-5yo, SB) {rb:.3f} [{lo:.3f}, {hi:.3f}]; ceiling sqrt(rel*.71) = {np.sqrt(max(rb,0)*.71):.3f}")
if lo>0: dl,dh=np.nanquantile(bs_dis,[.025,.975]); print(f"S4 disattenuated {robs/np.sqrt(rb*.71):+.3f} [{dl:+.3f}, {dh:+.3f}]")
else: print("S4 disattenuated: not estimable (reliability CI includes 0)")
# S5
from math import atanh
for tr in [.2,.3,.5]:
    ro=tr*np.sqrt(max(rb,0)*.71); n=((1.959964+0.841621)/atanh(ro))**2+3 if ro>0 else float('inf')
    print(f"S5 true r {tr}: observable {ro:.3f}, n for 80% power ~ {n:.0f}")
# descriptive
print("FB_Composite distribution 3-5yo:",np.unique(y,return_counts=True))
