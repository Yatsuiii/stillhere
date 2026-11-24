import numpy as np, sys
sys.path.insert(0,'/home/claude/len'); from core import *
exec(open('/home/claude/ibc/analyze_ibc.py').read().split("n=min(")[0])
n=min(v['n'] for v in V.values()); subs=sorted(set(s for s,_ in V)); TAGS=['PainMovie_dir-ap','PainMovie_dir-pa']; T=n-5
LENS=[30,60,90,120,158,T]; rng=np.random.default_rng(0)
for t in ['tom_grp','tom_ind','pain_grp']:
    print(f"IBC {t}: share of windows detected at calibrated cutoff [bootstrap CI over people]; one viewing (24) vs two averaged (12)")
    for L in LENS:
        p1={};p2={};f1=[];f2=[]
        for s in subs:
            X={g:V[(s,g)][t][5:n] for g in TAGS}; R={g:zz(np.mean([V[(o,g)][t][5:n] for o in subs if o!=s],0)) for g in TAGS}
            for st in (range(0,T-L+1,10) if L<T else [0]):
                sl=slice(st,st+L)
                for g in TAGS:
                    Sg=surr(zz(X[g][sl]),seed=hash((s,g,t,L,st))%2**32); p1.setdefault((s,g),[]).append(pval(X[g][sl],R[g][sl],Sg)[1]); f1+=[pval(X[g][sl],m,Sg)[1] for m in mism(R[g][sl])]
                xa=zz(zz(X[TAGS[0]][sl])+zz(X[TAGS[1]][sl])); ra=zz(R[TAGS[0]][sl]+R[TAGS[1]][sl]); Sg=surr(xa,seed=hash((s,'avg',t,L,st))%2**32)
                p2.setdefault(s,[]).append(pval(xa,ra,Sg)[1]); f2+=[pval(xa,m,Sg)[1] for m in mism(ra)]
        c1,c2=calib(f1),calib(f2)
        v1=np.array([np.mean(np.array(p)<c1) for p in p1.values()]); v2=np.array([np.mean(np.array(p)<c2) for p in p2.values()])
        b=lambda v: (np.quantile([rng.choice(v,len(v)).mean() for _ in range(2000)],[.025,.975]))
        l1,h1=b(v1); l2,h2=b(v2)
        print(f"  {L*2/60:4.1f} min: one {v1.mean():.0%} [{l1:.0%}-{h1:.0%}] (cut {c1:.3f}, FA@.05 {np.mean(np.array(f1)<.05):.1%})   two {v2.mean():.0%} [{l2:.0%}-{h2:.0%}] (cut {c2:.3f}, FA@.05 {np.mean(np.array(f2)<.05):.1%})")
        if L==T: print("     full-length misses one-viewing:",[k for k,p in p1.items() if p[0]>=c1],' two:',[k for k,p in p2.items() if p[0]>=c2])
