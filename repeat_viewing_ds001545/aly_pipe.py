import sys, os, numpy as np, ants, h5py, nibabel as nb, urllib.request, shutil
from nilearn import datasets
W='/home/claude/aly'; SAXE='/home/claude/saxe/roi'; B='https://s3.amazonaws.com/openneuro.org/ds001545'
TOM=['DMPFC','MMPFC','VMPFC','PC','LTPJ','RTPJ']; PAIN=['AMCC','LInsula','RInsula','LS2','RS2','LMFG','RMFG']
CLIPS=[(5,64),(71,130),(137,196),(203,262),(269,328),(335,394)]  # 1-based volumes after dropping 3
def tpl():
    f=f'{W}/mni2.nii.gz'
    if not os.path.exists(f): datasets.load_mni152_template(resolution=2).to_filename(f)
    return ants.image_read(f)
def roi_mask(r):
    xyz=np.array(h5py.File(f'{SAXE}/{r}.mat')['roi_XYZmm']); m=nb.load(f'{W}/mni2.nii.gz'); inv=np.linalg.inv(m.affine)
    a=np.zeros(m.shape,np.float32); v=np.round(inv[:3,:3]@xyz+inv[:3,3:]).astype(int); a[v[0],v[1],v[2]]=1
    fn=f'/tmp/aroi_{os.getpid()}.nii.gz'; nb.Nifti1Image(a,m.affine).to_filename(fn); return ants.image_read(fn)
def get(u,f):
    for k in range(4):
        try: urllib.request.urlretrieve(u,f); return
        except Exception as e: err=e
    raise err
def run(s):
    out=f'{W}/out/{s}.npz'
    if os.path.exists(out): return
    T=tpl(); runs={}; mp={}; ev={}
    for r in (1,2,3):
        f=f'/tmp/{s}_r{r}.nii.gz'; get(f'{B}/{s}/func/{s}_task-movie_run-0{r}_bold.nii.gz',f)
        ev[r]=[l.split('\t')[2] for l in urllib.request.urlopen(f'{B}/{s}/func/{s}_task-movie_run-0{r}_events.tsv').read().decode('latin-1').strip().split('\n')[1:]]
        im=ants.image_read(f); sp=list(im.spacing); a=im.numpy()[...,3:]
        im=ants.from_numpy(a,origin=im.origin,spacing=im.spacing,direction=im.direction)
        runs[r]=ants.resample_image(im,(3,3,3,sp[3]),use_voxels=False,interp_type=0); del im,a; os.remove(f)
    ref=ants.get_average_of_timeseries(runs[1]); mc={}
    for r,im in runs.items():
        m=ants.motion_correction(im,fixed=ref,type_of_transform='Rigid'); mc[r]=m['motion_corrected'].numpy()
        mp[r]=np.array([np.ravel(ants.read_transform(q[0] if isinstance(q,(list,tuple)) else q).parameters) for q in m['motion_parameters']])
    del runs
    epi=ants.from_numpy(np.mean(mc[1],3),origin=ref.origin,spacing=ref.spacing,direction=ref.direction)
    em=ants.get_mask(epi,cleanup=2); bmT=ants.threshold_image(T,T.mean()*0.3,1e9)
    reg=ants.registration(fixed=epi*em,moving=T*bmT,type_of_transform='SyN'); tl=reg['fwdtransforms']
    w=ants.apply_transforms(fixed=epi,moving=bmT,transformlist=tl,interpolator='nearestNeighbor').numpy(); e=em.numpy()
    dice=2*(w*e).sum()/(w.sum()+e.sum())
    res={'dice':dice}
    for r in (1,2,3):
        Y=mc[r]; n=Y.shape[3]; X=np.column_stack([mp[r],np.ones(n),np.linspace(-1,1,n)])
        for roi in TOM+PAIN:
            g=ants.apply_transforms(fixed=epi,moving=roi_mask(roi),transformlist=tl,interpolator='linear').numpy()>0.5
            res[f'{roi}_n']=g.sum()
            y=Y[g].mean(0) if g.any() else np.full(n,np.nan); y=y-X@np.linalg.lstsq(X,y,rcond=None)[0] if g.any() else y
            for i,(a,b) in enumerate(CLIPS):
                res[f'r{r}_c{i}_{roi}']=y[a-1:b]
        for i,lab in enumerate(ev[r]): res[f'r{r}_c{i}_label']=lab
    np.savez(out,**res); print(s,'ok dice',round(dice,3),'min roi vox',min(int(res[f'{r}_n']) for r in TOM+PAIN),flush=True)
if __name__=='__main__':
    os.makedirs(f'{W}/out',exist_ok=True)
    for s in sys.argv[1:]:
        try: run(s)
        except Exception as e:
            import traceback; traceback.print_exc(); print(s,'FAILED',e,flush=True)
