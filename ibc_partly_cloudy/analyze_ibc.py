import numpy as np, glob, json
exec(open('/home/claude/saxe/analyze.py').read().split("adults=[")[0].replace("W='/home/claude/saxe'","W='/home/claude/saxe'"))
A=[s for s in S if S[s]['adult']]
z=lambda x:(x-x.mean())/x.std()
REF={n:z(np.mean([S[a][n] for a in A],0)) for n in ['tom','pain']}
TOM=['DMPFC','MMPFC','VMPFC','PC','LTPJ','RTPJ']; PAIN=['AMCC','LInsula','RInsula','LS2','RS2','LMFG','RMFG']
rng=np.random.default_rng(3)
V={}
for f in sorted(glob.glob('/home/claude/ibc/out/*.npz')):
    s=f.split('/')[-1][:-4]; d=np.load(f)
    for tag in ['PainMovie_dir-ap','PainMovie_dir-pa']:
        mp=d[f'{tag}_mp']; n=mp.shape[0]; X=np.column_stack([mp,np.ones(n),np.linspace(-1,1,n)])
        Q=lambda y: z((y-X@np.linalg.lstsq(X,y,rcond=None)[0]))
        V[(s,tag)]={'tom_grp':z(np.mean([Q(d[f'{tag}_{r}_grp']) for r in TOM],0)),'tom_ind':z(np.mean([Q(d[f'{tag}_{r}_ind']) for r in TOM],0)),
                    'pain_grp':z(np.mean([Q(d[f'{tag}_{r}_grp']) for r in PAIN],0)),'n':n,
                    'loc':{r:(float(d[f'{r}_grp_tmean']),float(d[f'{r}_ind_tmean'])) for r in TOM}}
n=min(v['n'] for v in V.values()); L=158
gp=np.mean([v['pain_grp'][:n] for v in V.values()],0)
cs=[np.corrcoef(gp[k:k+L],REF['pain'])[0,1] for k in range(0,n-L+1)]; k=int(np.argmax(cs))
print(f"subjects={len(set(s for s,_ in V))} viewings={len(V)} run TRs={n}; alignment offset k={k} (group pain r={max(cs):.2f}; next best {sorted(cs)[-2]:.2f})")
def test(x,r):
    o=np.corrcoef(x,r)[0,1]; nl=np.array([np.corrcoef(np.roll(x,j),r)[0,1] for j in rng.integers(10,L-10,1000)]); return o,(1+(nl>=o).sum())/1001
rows={}
for key,v in V.items():
    rows[key]={t:test(v[t][k:k+L],REF['tom']) for t in ['tom_grp','tom_ind']}
    rows[key].update({t+'_rev':test(v[t][k:k+L],REF['tom'][::-1]) for t in ['tom_grp','tom_ind']})
    rows[key]['pain_grp']=test(v['pain_grp'][k:k+L],REF['pain'])
THR=0.05
for t in ['tom_grp','tom_ind','tom_grp_rev','tom_ind_rev','pain_grp']:
    print(f"{t:12s} detected {sum(r[t][1]<THR for r in rows.values())}/{len(rows)}  median r {np.median([r[t][0] for r in rows.values()]):.2f}")
b=sum(rows[x]['tom_ind'][1]<THR and rows[x]['tom_grp'][1]>=THR for x in rows); c=sum(rows[x]['tom_grp'][1]<THR and rows[x]['tom_ind'][1]>=THR for x in rows)
from math import comb
p=min(1,2*sum(comb(b+c,i) for i in range(0,min(b,c)+1))/2**(b+c)) if b+c else 1
print(f"McNemar: personal-only {b}, group-only {c}, exact p={p:.3f}")
subs=sorted(set(s for s,_ in V))
for s in subs:
    a,bb=V[(s,'PainMovie_dir-ap')],V[(s,'PainMovie_dir-pa')]
    print(s,'retest grp',round(np.corrcoef(a['tom_grp'],bb['tom_grp'])[0,1],2),'ind',round(np.corrcoef(a['tom_ind'],bb['tom_ind'])[0,1],2),
          '| r ap/pa grp',[round(rows[(s,t)]['tom_grp'][0],2) for t in ['PainMovie_dir-ap','PainMovie_dir-pa']],'ind',[round(rows[(s,t)]['tom_ind'][0],2) for t in ['PainMovie_dir-ap','PainMovie_dir-pa']],
          '| loc t grp/ind',round(np.nanmean([x[0] for x in a['loc'].values()]),1),round(np.mean([x[1] for x in a['loc'].values()]),1))
json.dump({f'{s}|{t}':{kk:list(map(float,vv)) for kk,vv in r.items()} for (s,t),r in rows.items()},open('/home/claude/ibc/results_ibc.json','w'))
