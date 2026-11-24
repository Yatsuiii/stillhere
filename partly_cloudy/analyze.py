import numpy as np, csv, glob, json
W='/home/claude/saxe'; rng=np.random.default_rng(0)
NETS={'tom':['DMPFC','MMPFC','VMPFC','PC','LTPJ','RTPJ'],'pain':['AMCC','LInsula','RInsula','LS2','RS2','LMFG','RMFG']}
P={r['participant_id']:r for r in csv.DictReader(open(f'{W}/participants.tsv'),delimiter='\t')}
def z(x): return (x-x.mean())/x.std()
S={}
for f in sorted(glob.glob(f'{W}/ts/*.npz')):
    s=f.split('/')[-1][:-4]; d=np.load(f); R=d['R']; n=int(d['nt'])
    X=np.column_stack([R,np.ones(n),np.linspace(-1,1,n)])
    out={}
    for net,rs in NETS.items():
        ser=[]
        for r in rs:
            y=d[r]; y=y-X@np.linalg.lstsq(X,y,rcond=None)[0]; ser.append(z(y[10:]))
        out[net]=z(np.mean(ser,0))
    spikes=int(sum(((R[:,j]!=0).sum()==1) for j in range(R.shape[1])))
    S[s]=dict(out,spikes=spikes,grp=P[s]['AgeGroup'],adult=P[s]['Child_Adult']=='adult',age=float(P[s]['Age']))
adults=[s for s in S if S[s]['adult']]
def ref(net,s): return z(np.mean([S[a][net] for a in adults if a!=s],0))
def test(x,r):
    obs=np.corrcoef(x,r)[0,1]; T=len(x)
    null=np.array([np.corrcoef(np.roll(x,k),r)[0,1] for k in rng.integers(10,T-10,1000)])
    return obs,(1+(null>=obs).sum())/1001
rows=[]
for s,v in S.items():
    row=dict(sub=s,grp=v['grp'],age=v['age'],spikes=v['spikes'])
    for net in NETS:
        r=ref(net,s); row[net+'_r'],row[net+'_p']=test(v[net],r)
        row[net+'_rev_r'],row[net+'_rev_p']=test(v[net],r[::-1])
    row['tom_spec']=row['tom_r']-np.corrcoef(v['tom'],ref('pain',s))[0,1]
    rows.append(row)
json.dump(rows,open(f'{W}/results.json','w'),indent=1,default=float)
order=['3yo','4yo','5yo','7yo','8-12yo','Adult']
grps=sorted(set(r['grp'] for r in rows),key=lambda g: order.index(g) if g in order else 99)
print(f"n={len(rows)} subjects; adults={len(adults)}")
print("group   n  ToM_det  pain_det  ToM_FA(rev)  pain_FA(rev)  median_ToM_r  median_spikes")
for g in grps:
    G=[r for r in rows if r['grp']==g]; f=lambda k: np.mean([r[k]<.05 for r in G])
    print(f"{g:7s}{len(G):3d}  {f('tom_p'):6.0%}   {f('pain_p'):6.0%}    {f('tom_rev_p'):6.0%}      {f('pain_rev_p'):6.0%}       {np.median([r['tom_r'] for r in G]):.2f}         {np.median([r['spikes'] for r in G]):.0f}")
K=[r for r in rows if r['grp']!='Adult' and not r['grp'].lower().startswith('adult')]
K=[r for r in rows if not S[r['sub']]['adult']]
q=np.quantile([r['spikes'] for r in K],[1/3,2/3])
for i,(lo,hi) in enumerate([(-1,q[0]),(q[0],q[1]),(q[1],1e9)]):
    G=[r for r in K if lo<r['spikes']<=hi]
    print(f"children motion tercile {i+1} (spikes {lo:.0f}-{hi:.0f}]: n={len(G)} ToM det {np.mean([r['tom_p']<.05 for r in G]):.0%}")
print("ToM specificity >0 by group:",{g:f"{np.mean([r['tom_spec']>0 for r in rows if r['grp']==g]):.0%}" for g in grps})
