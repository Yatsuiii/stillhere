import numpy as np, glob
TOM=['DMPFC','MMPFC','VMPFC','PC','LTPJ','RTPJ']; PAIN=['AMCC','LInsula','RInsula','LS2','RS2','LMFG','RMFG']
z=lambda x:(x-x.mean())/x.std()
V={}
for f in sorted(glob.glob('/home/claude/ibc/out/*.npz')):
    s=f.split('/')[-1][:-4]; d=np.load(f)
    for tag in ['PainMovie_dir-ap','PainMovie_dir-pa']:
        mp=d[f'{tag}_mp']; n=mp.shape[0]; X=np.column_stack([mp,np.ones(n),np.arange(n)]); P=np.linalg.pinv(X)
        Q=lambda y: z(y-X@(P@y))
        V[(s,tag)]={t:z(np.mean([Q(d[f'{tag}_{r}_{suf}']) for r in rs],0)) for t,rs,suf in [('tom_grp',TOM,'grp'),('tom_ind',TOM,'ind'),('pain_grp',PAIN,'grp')]}
        assert all(np.isfinite(v).all() for v in V[(s,tag)].values())
subs=sorted(set(s for s,_ in V)); n=178
def ep(x,r,kmin=10):
    o=np.corrcoef(x,r)[0,1]; nl=np.array([np.corrcoef(np.roll(x,k),r)[0,1] for k in range(kmin,len(x)-kmin+1)]); return o,(1+(nl>=o).sum())/(len(nl)+1)
for lo,hi,lab in [(5,178,'TR5-end (as reported)'),(16,174,'TR16-173 (same film span as ds000228 TRs 11-168, k=16)'),(26,178,'TR26-end (drop 10 more)')]:
    print('==',lab)
    for t in ['tom_grp','tom_ind','pain_grp']:
        tab={}
        for s in subs:
            tab[s]=[]
            for tag in ['PainMovie_dir-ap','PainMovie_dir-pa']:
                ref=z(np.mean([V[(o,tag)][t] for o in subs if o!=s],0))
                tab[s].append(ep(V[(s,tag)][t][lo:hi],ref[lo:hi]))
        det=sum(p<.05 for s in subs for _,p in tab[s]); both=[s for s in subs if all(p>=.05 for _,p in tab[s])]
        print(f' {t}: {det}/24 detected; missed both: {both}; misses:',[(s[-2:],i,round(r,2),round(p,3)) for s in subs for i,(r,p) in enumerate(tab[s]) if p>=.05])
# reversed and mismatched calibration on internal ref
fa=[]
for s in subs:
    for tag in ['PainMovie_dir-ap','PainMovie_dir-pa']:
        ref=z(np.mean([V[(o,tag)]['tom_grp'] for o in subs if o!=s],0))[5:]
        for k in np.linspace(8,160,20).astype(int): fa.append(ep(V[(s,tag)]['tom_grp'][5:],np.roll(ref[::-1],k))[1]<.05)
print('internal tom_grp FA over 20 mismatched refs x24:',np.mean(fa))
# onset transient check: group mean ToM first 30 TRs
g=np.mean([V[k]['tom_grp'] for k in V],0); print('group ToM grp mean TR0-30:',np.round(g[:30],1))
