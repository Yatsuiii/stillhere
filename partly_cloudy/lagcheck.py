# EXPLORATORY: is the miss a timing offset? Estimate each person's lag from the PAIN network (independent of ToM), apply to ToM.
import numpy as np, json, glob, csv
exec(open('analyze.py').read().split("adults=[")[0])  # reuse S building (tom, pain per subject)
adults=[s for s in S if S[s]['adult']]
def ref(net,s): return z(np.mean([S[a][net] for a in adults if a!=s],0))
rng=np.random.default_rng(2)
def test(x,r):
    o=np.corrcoef(x,r)[0,1]; nl=np.array([np.corrcoef(np.roll(x,k),r)[0,1] for k in rng.integers(10,148,1000)]); return o,(1+(nl>=o).sum())/1001
prev={r['sub']:r for r in json.load(open('results.json'))}
rows=[]
for s in S:
    pr=ref('pain',s); L=int(np.argmax([np.corrcoef(np.roll(S[s]['pain'],k),pr)[0,1] for k in range(-3,4)])-3)
    tl=int(np.argmax([np.corrcoef(np.roll(S[s]['tom'],k),ref('tom',s))[0,1] for k in range(-3,4)])-3)
    r,p=test(np.roll(S[s]['tom'],L),ref('tom',s)); rows.append((s,S[s]['grp'],L,tl,prev[s]['tom_p']<.05,p<.05,round(r,2)))
order=['3yo','4yo','5yo','7yo','8-12yo','Adult']
for g in order:
    G=[x for x in rows if x[1]==g]; print(g,'det lag0',f"{np.mean([x[4] for x in G]):.0%}",'det with pain-lag',f"{np.mean([x[5] for x in G]):.0%}",'pain-lag==tom-lag',f"{np.mean([x[2]==x[3] for x in G]):.0%}")
print('missed adults (sub, pain lag, tom lag, detected with pain lag, r):',[(x[0],x[2],x[3],x[5],x[6]) for x in rows if x[1]=='Adult' and not x[4]])
