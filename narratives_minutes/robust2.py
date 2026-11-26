# Exploratory stress test (after Raghav's challenge): does a stronger readout cut the 5-minute miss rate?
import glob,zlib,numpy as np, nibabel.freesurfer as fs
ST=['black','forgot','merlin','sherlock','tunnel','21styear','slumlord']
l=fs.read_annot('atlas/lh.Schaefer2018_400Parcels_17Networks_order.annot'); r=fs.read_annot('atlas/rh.Schaefer2018_400Parcels_17Networks_order.annot')
N=[n.decode() for n in l[2][1:]]+[n.decode() for n in r[2][1:]]
pick=lambda hemis,keys:[i for i,n in enumerate(N) if any(h in n for h in hemis) and any(k in n for k in keys)]
LK=['TempPar','DefaultB_Temp','DefaultB_PFCv','DefaultB_PFCl']
SETS={'LANG left mean':('mean',pick(['_LH_'],LK)),'LANG both mean':('mean',pick(['_LH_','_RH_'],LK)),
      'LANG left parcelwise':('pw',pick(['_LH_'],LK)),'LANG both parcelwise':('pw',pick(['_LH_','_RH_'],LK)),
      'all 400 parcelwise':('pw',list(range(400)))}
D={st:{f.split('_')[-1][:-4]:np.load(f)['P'] for f in sorted(glob.glob(f'ts400/{st}_sub-*.npz'))} for st in ST}
z=lambda a:(a-a.mean(0))/a.std(0)
def surr(X,n,seed):  # same random phases for all columns: keeps cross-parcel covariance
    rng=np.random.default_rng(seed); F=np.fft.rfft(X-X.mean(0),axis=0); T=len(X)
    ph=rng.uniform(0,2*np.pi,(n,F.shape[0],1)); ph[:,0]=0
    if T%2==0: ph[:,-1]=0
    return np.fft.irfft(np.abs(F)[None]*np.exp(1j*(np.angle(F)[None]+ph)),n=T,axis=1)
def stat(X,R,kind):
    if kind=='mean': a=X.mean(1); b=R.mean(1); return float(np.corrcoef(a,b)[0,1])
    return float((z(X)*z(R)).mean(0).mean())
def run(a,b,name,kind,idx,nref=20):
    rng=np.random.default_rng(7); ps=[];fa=[]
    for st in ST:
        for s in D[st]:
            X=D[st][s][a:b][:,idx]; S=surr(X,500,zlib.crc32(f'{st}{s}'.encode()))
            def p(R):
                o=stat(X,R,kind); nl=np.array([stat(Sx,R,kind) for Sx in S]); return (1+(nl>=o).sum())/(len(nl)+1)
            pool=[o for o in D[st] if o!=s]; ids=pool if len(pool)<=nref else rng.choice(pool,nref,replace=False)
            ps.append(p(np.mean([D[st][o][a:b][:,idx] for o in ids],0)))
            for rs in ST:
                if rs==st: continue
                pool=[o for o in D[rs] if o!=s]; ids=pool if len(pool)<=nref else rng.choice(pool,nref,replace=False)
                fa.append(p(np.mean([D[rs][o][a:b][:,idx] for o in ids],0)))
    c=np.quantile(fa,.05); ps=np.array(ps); print(f'{name:28s} {a//40}-{b//40} min  miss {np.mean(ps>c):.1%} (n={len(ps)}, cutoff {c:.3f})',flush=True)
for name,(kind,idx) in SETS.items():
    for a,b in [(0,200),(0,400)]: run(a,b,name,kind,idx)
