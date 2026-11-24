import sys, os, glob, numpy as np, ants, h5py, nibabel as nb, urllib.request, json
from nilearn import datasets
W='/home/claude/ibc'; SAXE='/home/claude/saxe/roi'
TOM=['DMPFC','MMPFC','VMPFC','PC','LTPJ','RTPJ']; PAIN=['AMCC','LInsula','RInsula','LS2','RS2','LMFG','RMFG']
def tpl():
    f=f'{W}/mni2.nii.gz'
    if not os.path.exists(f): datasets.load_mni152_template(resolution=2).to_filename(f)
    return ants.image_read(f)
def roi_mask(t,name,radius=None):
    xyz=np.array(h5py.File(f'{SAXE}/{name}.mat')['roi_XYZmm'])
    a=np.zeros(t.shape,np.float32); inv=np.linalg.inv(nb.load(f'{W}/mni2.nii.gz').affine)
    if radius is None:
        v=np.round(inv[:3,:3]@xyz+inv[:3,3:]).astype(int); a[v[0],v[1],v[2]]=1
    else:
        c=xyz.mean(1); g=np.stack(np.meshgrid(*[np.arange(n) for n in t.shape],indexing='ij'),-1)
        mm=g@nb.load(f'{W}/mni2.nii.gz').affine[:3,:3].T+nb.load(f'{W}/mni2.nii.gz').affine[:3,3]
        a[np.linalg.norm(mm-c,axis=-1)<=radius]=1
    m=nb.load(f'{W}/mni2.nii.gz'); fn=f'/tmp/roi_{os.getpid()}.nii.gz'; nb.Nifti1Image(a,m.affine).to_filename(fn); return ants.image_read(fn)
def hrf(tr=2.0,n=16):
    from math import gamma
    t=np.arange(0,32,tr); g=lambda t,a: t**(a-1)*np.exp(-t)/gamma(a); h=g(t,6)-g(t,16)/6; return h/h.sum()
def run(s):
    out=f'{W}/out/{s}.npz'
    if os.path.exists(out): return
    keys=[k for k in open(f'{W}/keys.txt').read().split() if f'/{s}/' in k]
    rd=f'{W}/raw_{s}'; os.makedirs(rd,exist_ok=True)
    for k in keys:
        f=f'{rd}/{os.path.basename(k)}'
        if not os.path.exists(f): urllib.request.urlretrieve('https://s3.amazonaws.com/openneuro.org/'+k,f)
    cache=f'{W}/cache_{s}'
    if os.path.exists(cache+'/done'):
        z=np.load(cache+'/mc.npz'); mc={k[3:]:z[k] for k in z if k.startswith('mc_')}; mp={k[3:]:z[k] for k in z if k.startswith('mp_')}
        epi=ants.image_read(cache+'/epi.nii.gz'); T=tpl(); tl=open(cache+'/tl.txt').read().split()
    else:
      runs={}
      for tag in ['PainMovie_dir-ap','PainMovie_dir-pa','TheoryOfMind_dir-ap','TheoryOfMind_dir-pa']:
          f=glob.glob(f'{rd}/*task-{tag}_bold.nii.gz')[0]
          im=ants.image_read(f); sp=list(im.spacing); runs[tag]=ants.resample_image(im,(3,3,3,sp[3]),use_voxels=False,interp_type=0)
          del im
      ref=ants.get_average_of_timeseries(runs['PainMovie_dir-ap'])
      mc={};mp={}
      for tag,im in runs.items():
          r=ants.motion_correction(im,fixed=ref,type_of_transform='Rigid')
          mc[tag]=r['motion_corrected'].numpy(); mp[tag]=np.array([np.ravel(ants.read_transform(q[0] if isinstance(q,(list,tuple)) else q).parameters) for q in r['motion_parameters']])
      epi=ants.from_numpy(np.mean(mc['PainMovie_dir-ap'],3),origin=ref.origin,spacing=ref.spacing,direction=ref.direction)
      t1=ants.image_read(glob.glob(f'{rd}/*T1w.nii.gz')[0]); T=tpl()
      aff=ants.registration(fixed=T,moving=t1,type_of_transform='Affine')
      bm=ants.apply_transforms(fixed=t1,moving=ants.threshold_image(T,T.mean()*0.3,1e9),transformlist=aff['invtransforms'],whichtoinvert=[True],interpolator='nearestNeighbor')
      t1b=t1*ants.morphology(bm,'dilate',3)
      r2=ants.registration(fixed=t1b,moving=T,type_of_transform='SyN')    # T1 pts -> MNI pts
      r1=ants.registration(fixed=epi,moving=t1b,type_of_transform='Rigid') # EPI pts -> T1 pts
      tl=r2['fwdtransforms']+r1['fwdtransforms']

      import shutil; os.makedirs(cache,exist_ok=True); tl2=[]
      for i,f in enumerate(tl):
          d=f'{cache}/t{i}_'+os.path.basename(f); shutil.copy(f,d); tl2.append(d)
      tl=tl2; open(cache+'/tl.txt','w').write(' '.join(tl)); ants.image_write(epi,cache+'/epi.nii.gz')
      np.savez(cache+'/mc.npz',**{'mc_'+k:v for k,v in mc.items()},**{'mp_'+k:v for k,v in mp.items()}); open(cache+'/done','w').write('1')
    warp=lambda m: ants.apply_transforms(fixed=epi,moving=m,transformlist=tl,interpolator='linear').numpy()
    # localiser GLM
    tm=[];
    for tag in ['TheoryOfMind_dir-ap','TheoryOfMind_dir-pa']:
        Y=mc[tag]; n=Y.shape[3]; ev=[l.split('\t') for l in open(glob.glob(f'{rd}/*task-{tag}_events.tsv')[0]).read().strip().split('\n')[1:]]
        X=[]
        for cond in ['belief','photo']:
            b=np.zeros(n)
            for o,d,c in ev:
                if c==cond: b[int(float(o)/2):int((float(o)+float(d))/2)]=1
            X.append(np.convolve(b,hrf())[:n])
        X=np.column_stack(X+[mp[tag],np.ones(n),np.linspace(-1,1,n)]); tm.append((Y.reshape(-1,n).T,X))
    Yall=np.vstack([y-y.mean(0) for y,_ in tm]); n1=tm[0][1].shape[0]
    X1,X2=tm[0][1],tm[1][1]; Xall=np.zeros((X1.shape[0]+X2.shape[0],2+2*(X1.shape[1]-2)))
    Xall[:,:2]=np.vstack([X1[:,:2],X2[:,:2]]); k=X1.shape[1]-2
    Xall[:n1,2:2+k]=X1[:,2:]; Xall[n1:,2+k:]=X2[:,2:]
    B,res,_,_=np.linalg.lstsq(Xall,Yall,rcond=None); e=Yall-Xall@B; dof=Xall.shape[0]-np.linalg.matrix_rank(Xall)
    c=np.zeros(Xall.shape[1]);c[0]=1;c[1]=-1; v=(e**2).sum(0)/dof*(c@np.linalg.pinv(Xall.T@Xall)@c)
    tmap=(c@B)/np.sqrt(np.maximum(v,1e-12)); tmap=tmap.reshape(epi.shape)
    brain=epi.numpy()>np.percentile(epi.numpy(),60)
    res={}
    for net,names in [('tom',TOM),('pain',PAIN)]:
        for r in names:
            g=warp(roi_mask(T,r))>0.5; g&=brain
            res[f'{r}_grp_n']=g.sum()
            if net=='tom':
                big=(warp(roi_mask(T,r,15))>0.5)&brain; idx=np.flatnonzero(big)
                top=idx[np.argsort(tmap.ravel()[idx])[-50:]]; p=np.zeros(big.size,bool); p[top]=True; p=p.reshape(big.shape)
                res[f'{r}_ind_tmean']=tmap[p].mean(); res[f'{r}_grp_tmean']=tmap[g].mean() if g.any() else np.nan
            for tag in ['PainMovie_dir-ap','PainMovie_dir-pa']:
                Y=mc[tag]; n=Y.shape[3]
                res[f'{tag}_{r}_grp']=Y[g].mean(0) if g.any() else np.zeros(n)
                if net=='tom': res[f'{tag}_{r}_ind']=Y[p].mean(0)
    for tag in ['PainMovie_dir-ap','PainMovie_dir-pa']: res[f'{tag}_mp']=mp[tag]
    np.savez(out,**res); print(s,'ok',{k:round(float(v),2) for k,v in res.items() if k.endswith('tmean')},flush=True)
    import shutil; shutil.rmtree(rd)
if __name__=='__main__':
    for s in sys.argv[1:]:
        try: run(s)
        except Exception as e: import traceback; traceback.print_exc(); print(s,'FAILED',e,flush=True)
