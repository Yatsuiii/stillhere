import glob,sys,zlib,numpy as np
sys.path.insert(0,'/home/claude/stillhere/partly_cloudy'); from core import zz,surr,pval
ST=['black','forgot','merlin','sherlock','tunnel','21styear','slumlord']
D={st:{f.split('_')[-1][:-4]:np.load(f)['LANG'] for f in sorted(glob.glob(f'ts/{st}_sub-*.npz'))} for st in ST}
def run(a,b,nref=None,label=''):
    rng=np.random.default_rng(7); ps=[];fa=[]
    for st in ST:
        for s in D[st]:
            x=D[st][s][a:b]; S=surr(x,1000,zlib.crc32(f'{st}{s}{a}'.encode()))
            pool=[o for o in D[st] if o!=s]; ids=pool if nref is None or len(pool)<=nref else rng.choice(pool,nref,replace=False)
            ps.append(pval(x,zz(np.mean([D[st][o][a:b] for o in ids],0)),S)[1])
            for rs in ST:
                if rs==st: continue
                pool=[o for o in D[rs] if o!=s]; ids=pool if nref is None or len(pool)<=nref else rng.choice(pool,nref,replace=False)
                fa.append(pval(x,zz(np.mean([D[rs][o][a:b] for o in ids],0)),S)[1])
    c=np.quantile(fa,.05); print(f'{label:40s} miss {np.mean(np.array(ps)>c):.1%}  (n={len(ps)}, cutoff {c:.3f})',flush=True)
run(0,200,20,'first 5 min, 20 refs (original)')
run(200,400,20,'minutes 5-10, 20 refs')
run(0,200,None,'first 5 min, all other listeners as ref')
run(0,200,5,'first 5 min, 5 refs')
