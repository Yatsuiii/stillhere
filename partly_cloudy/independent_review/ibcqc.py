import numpy as np, ants, h5py, nibabel as nb, sys
W='/home/claude/ibc'; T=ants.image_read(f'{W}/mni2.nii.gz'); A=nb.load(f'{W}/mni2.nii.gz').affine; inv=np.linalg.inv(A)
print('template affine diag',np.diag(A)[:3],A[:3,3])
bmT=ants.threshold_image(T,T.mean()*0.3,1e9)
def roi(name):
    xyz=np.array(h5py.File(f'/home/claude/saxe/roi/{name}.mat')['roi_XYZmm']); a=np.zeros(T.shape,np.float32)
    v=np.round(inv[:3,:3]@xyz+inv[:3,3:]).astype(int); a[v[0],v[1],v[2]]=1
    return ants.from_numpy(a,origin=T.origin,spacing=T.spacing,direction=T.direction)
R={r:roi(r) for r in ['RTPJ','LTPJ','PC','MMPFC','VMPFC']}
for s in ['01','04','05','06','07','08','09','11','12','13','14','15']:
    c=f'{W}/cache_sub-{s}'; epi=ants.image_read(c+'/epi.nii.gz'); tl=open(c+'/tl.txt').read().split()
    em=ants.get_mask(epi,cleanup=2).numpy()
    w=ants.apply_transforms(fixed=epi,moving=bmT,transformlist=tl,interpolator='nearestNeighbor').numpy()
    dice=2*(w*em).sum()/(w.sum()+em.sum())
    # physical centroids
    def cen(m):
        idx=np.argwhere(m>0.5); 
        if len(idx)==0: return None
        pts=[epi.origin+np.array(epi.direction)@ (np.array(i)*np.array(epi.spacing)) for i in idx[::max(1,len(idx)//50)]]
        return np.mean(pts,0)
    bc=cen(em)
    out={}
    for r,m in R.items():
        wm=ants.apply_transforms(fixed=epi,moving=m,transformlist=tl,interpolator='linear').numpy()>0.5
        cc=cen(wm); out[r]=(None if cc is None else np.round(cc-bc).astype(int).tolist(), int(wm.sum()), round(float(em[wm].mean()) if wm.any() else 0,2))
    print(s,'dice',round(dice,3),'dir',np.round(np.diag(np.array(epi.direction)),0).tolist(),out,flush=True)
