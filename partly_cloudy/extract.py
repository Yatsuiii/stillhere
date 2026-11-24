import sys, os, numpy as np, nibabel as nb, h5py, scipy.io as sio, urllib.request
W='/home/claude/saxe'; B='https://s3.amazonaws.com/openneuro.org/ds000228/derivatives/fmriprep'
NETS={'tom':['DMPFC','MMPFC','VMPFC','PC','LTPJ','RTPJ'],'pain':['AMCC','LInsula','RInsula','LS2','RS2','LMFG','RMFG']}
ROIS={r:np.array(h5py.File(f'{W}/roi/{r}.mat')['roi_XYZmm']) for n in NETS.values() for r in n}
def run(s):
    out=f'{W}/ts/{s}.npz'
    if os.path.exists(out): return
    p=f'{B}/{s}/{s}_task-pixar_run-001'; nii=f'/tmp/{s}.nii.gz'; mat=f'/tmp/{s}.mat'
    for u,f in [(p+'_swrf_bold.nii.gz',nii),(p+'_ART_and_CompCor_nuisance_regressors.mat',mat)]:
        for k in range(4):
            try: urllib.request.urlretrieve(u,f); break
            except Exception as e: err=e
        else: print(s,'download fail',err); return
    img=nb.load(nii); d=img.get_fdata(dtype=np.float32); inv=np.linalg.inv(img.affine)
    reg={}
    for r,xyz in ROIS.items():
        v=np.round(inv[:3,:3]@xyz+inv[:3,3:]).astype(int)
        ok=(v>=0).all(0)&(v[0]<d.shape[0])&(v[1]<d.shape[1])&(v[2]<d.shape[2])
        ts=d[v[0,ok],v[1,ok],v[2,ok],:]; ts=ts[ts.std(1)>0]
        reg[r]=ts.mean(0); reg[r+'_nvox']=len(ts)
    R=np.array(h5py.File(mat)['R']); R=R if R.shape[0]==d.shape[3] else R.T
    np.savez(out,R=R,nt=d.shape[3],**reg)
    os.remove(nii); os.remove(mat); print(s,d.shape[3],R.shape,flush=True)
if __name__=='__main__':
    for s in sys.argv[1:]: run(s)
