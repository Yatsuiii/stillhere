# registration v3: EPI mean (masked) -> MNI template directly, SyN. (T1 route failed: no skull stripping available offline)
import numpy as np, ants, sys, os, shutil
exec(open('pipe.py').read().split("def run(s):")[0])
T=tpl(); bmT=ants.threshold_image(T,T.mean()*0.3,1e9); Tb=T*bmT
for s in sys.argv[1:]:
    cache=f'{W}/cache_{s}'; epi=ants.image_read(cache+'/epi.nii.gz')
    em=ants.get_mask(epi,cleanup=2); e=epi*em
    r=ants.registration(fixed=e,moving=Tb,type_of_transform='SyN')
    tl=[]
    for i,x in enumerate(r['fwdtransforms']):
        d=f'{cache}/v3_t{i}_'+os.path.basename(x); shutil.copy(x,d); tl.append(d)
    open(cache+'/tl.txt','w').write(' '.join(tl))
    w=ants.apply_transforms(fixed=epi,moving=bmT,transformlist=tl,interpolator='nearestNeighbor').numpy()
    em_n=em.numpy(); det=[np.linalg.det(np.array(ants.read_transform(x).parameters[:9]).reshape(3,3)) for x in tl if x.endswith('.mat')]
    print(s,'EPI mask L',round(em_n.sum()*27/1e6,2),'template brain in EPI L',round(w.sum()*27/1e6,2),'dice',round(2*(w*em_n).sum()/(w.sum()+em_n.sum()),2),'det',[round(d,2) for d in det],flush=True)
