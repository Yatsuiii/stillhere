# NOT PREREGISTERED. (1) alignment curve; (2) IBC-internal reference: same viewing direction, leave-one-subject-out, full run (no cross-dataset alignment)
import numpy as np
exec(open('analyze_ibc.py').read().split("rows={}")[0])
print('alignment curve r by k:',[round(c,2) for c in cs])
gt=np.mean([v['tom_grp'][:n] for v in V.values()],0); ct=[np.corrcoef(gt[j:j+L],REF['tom'])[0,1] for j in range(0,n-L+1)]
print('ToM group-mean curve by k:',[round(c,2) for c in ct])
subs=sorted(set(s for s,_ in V))
def test2(x,r):
    o=np.corrcoef(x,r)[0,1]; nl=np.array([np.corrcoef(np.roll(x,j),r)[0,1] for j in rng.integers(10,len(x)-10,1000)]); return o,(1+(nl>=o).sum())/1001
for t in ['tom_grp','tom_ind','pain_grp']:
    det=[];rev=[];rs=[]
    for s in subs:
        for tag in ['PainMovie_dir-ap','PainMovie_dir-pa']:
            ref=z(np.mean([V[(o,tag)][t][:n] for o in subs if o!=s],0)); x=V[(s,tag)][t][:n]
            o,p=test2(x[5:],ref[5:]); det.append(p<.05); rs.append(o); rev.append(test2(x[5:],ref[5:][::-1])[1]<.05)
    print(f'{t}: detected {sum(det)}/24, reversed FA {sum(rev)}/24, median r {np.median(rs):.2f}')
print('per-subject internal-ref detection (ap, pa) for tom_ind and tom_grp:')
for s in subs:
    out=[]
    for t in ['tom_ind','tom_grp']:
        for tag in ['PainMovie_dir-ap','PainMovie_dir-pa']:
            ref=z(np.mean([V[(o,tag)][t][:n] for o in subs if o!=s],0)); o,p=test2(V[(s,tag)][t][:n][5:],ref[5:]); out.append(f"{o:.2f}{'*' if p<.05 else ' '}")
    print(' ',s,'ind',out[:2],'grp',out[2:])
