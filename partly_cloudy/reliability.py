import numpy as np
exec(open('/home/claude/saxe/analyze.py').read().split("adults=[")[0])
A=[s for s in S if S[s]['adult']]; order=['3yo','4yo','5yo','7yo','8-12yo','Adult']; rng=np.random.default_rng(0)
H=[slice(0,79),slice(79,158)]
for net in ['tom','pain']:
    print(f"{net}: group  n  split-half r [95% CI]  Spearman-Brown  SD(full score)  mean full score")
    for g in order:
        G=[s for s in S if S[s]['grp']==g]; h1=[];h2=[];fu=[]
        for s in G:
            ref=np.mean([S[a][net] for a in A if a!=s],0)
            h1.append(np.corrcoef(S[s][net][H[0]],ref[H[0]])[0,1]); h2.append(np.corrcoef(S[s][net][H[1]],ref[H[1]])[0,1]); fu.append(np.corrcoef(S[s][net],ref)[0,1])
        h1,h2=np.array(h1),np.array(h2); r=np.corrcoef(h1,h2)[0,1]
        bs=[]
        for _ in range(5000):
            i=rng.integers(0,len(G),len(G))
            if np.std(h1[i])>0 and np.std(h2[i])>0: bs.append(np.corrcoef(h1[i],h2[i])[0,1])
        lo,hi=np.quantile(bs,[.025,.975]); sb=lambda x: 2*x/(1+x)
        print(f"  {g:7s}{len(G):3d}  {r:.2f} [{lo:.2f}, {hi:.2f}]   {sb(r):.2f} [{sb(lo):.2f}, {sb(hi):.2f}]   {np.std(fu):.2f}   {np.mean(fu):.2f}")

# EXPLORATORY (not in PREREG6): pooled children, scores residualised on age group, to get a tighter estimate
print("\nEXPLORATORY pooled children (age-group means removed):")
for net in ['tom','pain']:
    for label,grps in [('3-4yo',['3yo','4yo']),('5-12yo',['5yo','7yo','8-12yo']),('all children',['3yo','4yo','5yo','7yo','8-12yo'])]:
        h1=[];h2=[]
        for g in grps:
            G=[s for s in S if S[s]['grp']==g]; a=[];b=[]
            for s in G:
                ref=np.mean([S[x][net] for x in A],0); a.append(np.corrcoef(S[s][net][H[0]],ref[H[0]])[0,1]); b.append(np.corrcoef(S[s][net][H[1]],ref[H[1]])[0,1])
            h1+=list(np.array(a)-np.mean(a)); h2+=list(np.array(b)-np.mean(b))
        h1,h2=np.array(h1),np.array(h2); r=np.corrcoef(h1,h2)[0,1]
        bs=[np.corrcoef(h1[i],h2[i])[0,1] for i in (rng.integers(0,len(h1),len(h1)) for _ in range(5000))]
        lo,hi=np.quantile(bs,[.025,.975]); print(f"  {net} {label:12s} n={len(h1):3d} split-half r {r:.2f} [{lo:.2f}, {hi:.2f}]  Spearman-Brown {2*r/(1+r):.2f}")
