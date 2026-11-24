import numpy as np, glob, csv, json
W='/home/claude/saxe'
TOM=['DMPFC','MMPFC','VMPFC','PC','LTPJ','RTPJ']; PAIN=['AMCC','LInsula','RInsula','LS2','RS2','LMFG','RMFG']
P={r['participant_id']:r for r in csv.DictReader(open(f'{W}/participants.tsv'),delimiter='\t')}
def zs(x): x=np.asarray(x,float); return (x-x.mean())/x.std()
D={}
for f in sorted(glob.glob(f'{W}/ts/*.npz')):
    s=f.split('/')[-1][:-4]; d=np.load(f); R=d['R']; n=R.shape[0]
    X=np.column_stack([R,np.ones(n),np.arange(n)])
    pinv=np.linalg.pinv(X)
    def clean(y): return y-X@(pinv@y)
    D[s]={net:zs(np.mean([zs(clean(d[r])[10:]) for r in rs],0)) for net,rs in [('tom',TOM),('pain',PAIN)]}
    D[s]['adult']=P[s]['Child_Adult']=='adult'; D[s]['grp']=P[s]['AgeGroup']
    D[s]['spk']=int(((R!=0).sum(0)==1).sum()); D[s]['spk_after10']=int(sum(np.flatnonzero(R[:,j])[0]>=10 for j in range(R.shape[1]) if (R[:,j]!=0).sum()==1))
A=[s for s in D if D[s]['adult']]
print('n',len(D),'adults',len(A))
def ref(net,s): 
    m=[D[a][net] for a in A if a!=s]; assert len(m)==(32 if D[s]['adult'] else 33); return zs(np.mean(m,0))
def exact_p(x,r,kmin=10):
    T=len(x); o=np.corrcoef(x,r)[0,1]; nl=np.array([np.corrcoef(np.roll(x,k),r)[0,1] for k in range(kmin,T-kmin+1)])
    return o,(1+(nl>=o).sum())/(len(nl)+1)
res={}
for s in D:
    res[s]={}
    for net in ['tom','pain']:
        r=ref(net,s); res[s][net]=exact_p(D[s][net],r); res[s][net+'_rev']=exact_p(D[s][net],r[::-1])
    r=ref('tom',s)
    res[s]['B']=[exact_p(D[s]['tom'],np.roll(r[::-1],k))[1] for k in np.linspace(8,150,20).astype(int)]
order=['3yo','4yo','5yo','7yo','8-12yo','Adult']
from math import sqrt
def wil(k,n,z=1.96):
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*sqrt(p*(1-p)/n+z*z/(4*n*n))/d; return c-h,c+h
for g in order:
    G=[s for s in D if D[s]['grp']==g]; k=sum(res[s]['tom'][1]<.05 for s in G); kp=sum(res[s]['pain'][1]<.05 for s in G)
    k33=sum(res[s]['tom'][1]<.033 for s in G)
    fa=np.mean([p<.05 for s in G for p in res[s]['B']]); fa33=np.mean([p<.033 for s in G for p in res[s]['B']])
    print(f"{g:7s} n={len(G)} ToM {k}/{len(G)}={k/len(G):.0%} CI {wil(k,len(G))[0]:.2f}-{wil(k,len(G))[1]:.2f} pain {kp}/{len(G)} revFA tom {sum(res[s]['tom_rev'][1]<.05 for s in G)} pain {sum(res[s]['pain_rev'][1]<.05 for s in G)}  B-FA .05 {fa:.1%} .033 {fa33:.1%}  ToM@.033 {k33/len(G):.0%} medr {np.median([res[s]['tom'][0] for s in G]):.2f}")
miss=[s for s in A if res[s]['tom'][1]>=.05]
print('missed adults',[(s,round(res[s]['tom'][0],2),round(res[s]['tom'][1],3),'pain',round(res[s]['pain'][0],2),round(res[s]['pain'][1],3),'spk',D[s]['spk'],D[s]['spk_after10']) for s in miss])
allB=[p for s in D for p in res[s]['B']]
print('overall B FA .05',np.mean(np.array(allB)<.05),'; quantile for 5%',np.quantile(allB,.05))
# compare with stored results.json
prev={r['sub']:r for r in json.load(open(f'{W}/results.json'))}
dis=[(s,round(prev[s]['tom_p'],3),round(res[s]['tom'][1],3)) for s in D if (prev[s]['tom_p']<.05)!=(res[s]['tom'][1]<.05)]
print('disagreements with results.json tom_p at .05:',dis)
print('max |r diff|',max(abs(prev[s]['tom_r']-res[s]['tom'][0]) for s in D))
json.dump({s:{k:(list(map(float,v)) if not isinstance(v,(int,bool)) else v) for k,v in res[s].items()} for s in res},open('/home/claude/review/recomp.json','w'))
