import numpy as np
exec(open('/home/claude/review/recompute.py').read().split("res={}")[0])
for s in ['sub-pixar132','sub-pixar149','sub-pixar134','sub-pixar141','sub-pixar155','sub-pixar136']:
    for net in ['tom','pain']:
        r=ref(net,s); x=D[s][net]
        cc=[np.corrcoef(x[max(0,L):len(x)+min(0,L)], r[max(0,-L):len(r)-max(0,L)])[0,1] for L in range(-8,9)]
        print(s,net,' '.join(f'{c:+.2f}' for c in cc))
