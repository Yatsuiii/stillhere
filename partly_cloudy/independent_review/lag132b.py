import numpy as np
exec(open('/home/claude/review/recompute.py').read().split("res={}")[0])
s='sub-pixar132'
for net in ['tom','pain']:
    r=ref(net,s); x=D[s][net]
    cc=[(L,round(np.corrcoef(x[max(0,L):len(x)+min(0,L)], r[max(0,-L):len(r)-max(0,L)])[0,1],2)) for L in range(-20,1,2)]
    print(net,cc)
# null: how often does max over lags -20..20 of a subject exceed in detected adults
