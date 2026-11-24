import sys, os, numpy as np, nibabel as nb, h5py, urllib.request
W='/home/claude/saxe'; B='https://s3.amazonaws.com/openneuro.org/ds000228/derivatives/fmriprep'
TOM=['DMPFC','MMPFC','VMPFC','PC','LTPJ','RTPJ']
C={r:np.array(h5py.File(f'{W}/roi/{r}.mat')['roi_XYZmm']) for r in TOM}
def run(s):
    out=f'{W}/vox/{s}.npz'
    if os.path.exists(out): return
    nii=f'/tmp/{s}v.nii.gz'
    for k in range(4):
        try: urllib.request.urlretrieve(f'{B}/{s}/{s}_task-pixar_run-001_swrf_bold.nii.gz',nii); break
        except Exception as e: err=e
    else: print(s,'fail',err); return
    img=nb.load(nii); d=img.get_fdata(dtype=np.float32); A=img.affine
    g=np.stack(np.meshgrid(*[np.arange(n) for n in d.shape[:3]],indexing='ij'),-1).reshape(-1,3)
    mm=g@A[:3,:3].T+A[:3,3]; flat=d.reshape(-1,d.shape[3]); res={}
    inv=np.linalg.inv(A)
    for r,xyz in C.items():
        big=np.linalg.norm(mm-xyz.mean(1),axis=1)<=15
        v=np.round(inv[:3,:3]@xyz+inv[:3,3:]).astype(int); sm=np.zeros(d.shape[:3],bool); sm[v[0],v[1],v[2]]=True
        idx=np.flatnonzero(big); ts=flat[idx]; ok=ts.std(1)>0
        res[r]=ts[ok]; res[r+'_insphere']=sm.reshape(-1)[idx][ok]
    np.savez(out,**res); os.remove(nii); print(s,'ok',flush=True)
for s in sys.argv[1:]: run(s)
