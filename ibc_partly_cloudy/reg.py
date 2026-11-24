# redo registration only, reusing cached motion correction (registration bug: T1 was not skull-stripped, affine det 0.4-0.6)
import numpy as np, ants, sys, os, glob, shutil, urllib.request
exec(open('pipe.py').read().split("def run(s):")[0])
for s in sys.argv[1:]:
    cache=f'{W}/cache_{s}'; epi=ants.image_read(cache+'/epi.nii.gz'); T=tpl()
    k=[k for k in open(f'{W}/keys.txt').read().split() if f'/{s}/' in k and 'T1w' in k][0]; f=f'/tmp/{s}_T1w.nii.gz'
    if not os.path.exists(f): urllib.request.urlretrieve('https://s3.amazonaws.com/openneuro.org/'+k,f)
    t1=ants.image_read(f)
    r1=ants.registration(fixed=epi,moving=t1,type_of_transform='Rigid')
    em=ants.morphology(ants.threshold_image(epi,float(np.percentile(epi.numpy(),55)),1e9),'dilate',2)
    bm=ants.apply_transforms(fixed=t1,moving=em,transformlist=r1['invtransforms'],whichtoinvert=[True],interpolator='nearestNeighbor')
    r2=ants.registration(fixed=t1*bm,moving=T,type_of_transform='SyN')
    tl=r2['fwdtransforms']+r1['fwdtransforms']; tl2=[]
    for i,x in enumerate(tl):
        d=f'{cache}/v2_t{i}_'+os.path.basename(x); shutil.copy(x,d); tl2.append(d)
    open(cache+'/tl.txt','w').write(' '.join(tl2))
    bmT=ants.threshold_image(T,T.mean()*0.3,1e9); w=ants.apply_transforms(fixed=epi,moving=bmT,transformlist=tl2,interpolator='nearestNeighbor').numpy()
    dets=[np.linalg.det(np.array(ants.read_transform(x).parameters[:9]).reshape(3,3)) for x in tl2 if x.endswith('.mat')]
    print(s,'template brain warped to EPI (L):',round(w.sum()*27/1e6,2),'(template',round(bmT.numpy().sum()*8/1e6,2),') dets',[round(d,2) for d in dets],flush=True)
