import numpy as np, ants, h5py, nibabel as nb, urllib.request, os, sys
W='/home/claude/ibc'
src=open(f'{W}/pipe.py').read().split("def run(s):")[0].replace("from nilearn import datasets","")
exec(src)
for s in sys.argv[1:]:
    c=f'{W}/cache_{s}'; z=np.load(c+'/mc.npz'); epi=ants.image_read(c+'/epi.nii.gz'); tl=open(c+'/tl.txt').read().split(); T=tpl()
    ses=[k for k in open(f'{W}/keys.txt').read().split() if f'/{s}/' in k and 'TheoryOfMind' in k and k.endswith('events.tsv')]
    tm=[]
    for tag in ['TheoryOfMind_dir-ap','TheoryOfMind_dir-pa']:
        k=[x for x in ses if tag in x][0]; f=f'/home/claude/review/dl/{s}_{tag}.tsv'
        if not os.path.exists(f): urllib.request.urlretrieve('https://s3.amazonaws.com/openneuro.org/'+k,f)
        ev=[l.split('\t') for l in open(f).read().strip().split('\n')[1:]]
        Y=z['mc_'+tag]; n=Y.shape[3]; X=[]
        for cond in ['belief','photo']:
            b=np.zeros(n)
            for o,d,cc in ev:
                if cc==cond: b[int(float(o)/2):int((float(o)+float(d))/2)]=1
            X.append(np.convolve(b,hrf())[:n])
        X=np.column_stack(X+[z['mp_'+tag],np.ones(n),np.linspace(-1,1,n)]); tm.append((Y.reshape(-1,n).T,X))
    Yall=np.vstack([y-y.mean(0) for y,_ in tm]); n1=tm[0][1].shape[0]
    X1,X2=tm[0][1],tm[1][1]; k=X1.shape[1]-2; Xall=np.zeros((X1.shape[0]+X2.shape[0],2+2*k))
    Xall[:,:2]=np.vstack([X1[:,:2],X2[:,:2]]); Xall[:n1,2:2+k]=X1[:,2:]; Xall[n1:,2+k:]=X2[:,2:]
    B=np.linalg.lstsq(Xall,Yall,rcond=None)[0]; e=Yall-Xall@B; dof=Xall.shape[0]-np.linalg.matrix_rank(Xall)
    cv=np.zeros(Xall.shape[1]);cv[0]=1;cv[1]=-1; v=(e**2).sum(0)/dof*(cv@np.linalg.pinv(Xall.T@Xall)@cv)
    t=((cv@B)/np.sqrt(np.maximum(v,1e-12))).reshape(epi.shape)
    brain=epi.numpy()>np.percentile(epi.numpy(),60)
    tb=t[brain]; print(s,'rank',np.linalg.matrix_rank(Xall),'of',Xall.shape[1],'brain t: mean %.2f sd %.2f  frac>3.1 %.4f frac<-3.1 %.4f'%(tb.mean(),tb.std(),(tb>3.1).mean(),(tb<-3.1).mean()))
    for r in ['RTPJ','LTPJ','PC','DMPFC','MMPFC']:
        big=(ants.apply_transforms(fixed=epi,moving=roi_mask(T,r,15),transformlist=tl,interpolator='linear').numpy()>0.5)&brain
        tv=t[big]; print('  ',r,'N',big.sum(),'top50 +t %.2f  top50 -t %.2f  max %.1f'%(np.sort(tv)[-50:].mean(),np.sort(-tv)[-50:].mean(),tv.max()))
    # also: whole-brain belief-photo in a broad TPJ via group MNI peak spheres? done above
