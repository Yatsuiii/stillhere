import numpy as np
exec(open('analyze_ibc.py').read().split("rows={}")[0])
subs=sorted(set(s for s,_ in V))
for t in ['tom_grp','pain_grp']:
    a=z(np.mean([V[(s,'PainMovie_dir-ap')][t][:n] for s in subs],0)); b=z(np.mean([V[(s,'PainMovie_dir-pa')][t][:n] for s in subs],0))
    cc={L:round(np.corrcoef(a[10:-10],np.roll(b,L)[10:-10])[0,1],2) for L in range(-4,5)}; print(t,'ap vs pa shifted by L (pa rolled):',cc)
    # per subject
    ls=[]
    for s in subs:
        x=V[(s,'PainMovie_dir-ap')][t][:n]; y=V[(s,'PainMovie_dir-pa')][t][:n]
        ls.append(int(np.argmax([np.corrcoef(x[10:-10],np.roll(y,L)[10:-10])[0,1] for L in range(-4,5)])-4))
    print('  per-subject best lag',ls)
