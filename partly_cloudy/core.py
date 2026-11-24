import numpy as np
def zz(x): x=np.asarray(x,float); return (x-x.mean())/x.std()
def surr(x,n=2000,seed=0):
    rng=np.random.default_rng(seed); X=np.fft.rfft(x-x.mean()); T=len(x)
    ph=rng.uniform(0,2*np.pi,(n,len(X))); ph[:,0]=0
    if T%2==0: ph[:,-1]=0
    return np.fft.irfft(np.abs(X)*np.exp(1j*ph),n=T)
def pval(x,r,S=None):
    x=zz(x); r=zz(r); o=(x*r).mean()
    if S is None: S=surr(x)
    Sz=(S-S.mean(1,keepdims=True))/S.std(1,keepdims=True); nl=(Sz*r).mean(1)
    return o,(1+(nl>=o).sum())/(len(nl)+1)
def mism(r,k=20):
    T=len(r); return [np.roll(r[::-1],int(o)) for o in np.linspace(T*0.05,T*0.95,k)]
def wilson(k,n,z=1.96):
    from math import sqrt
    if n==0: return (np.nan,np.nan)
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*sqrt(p*(1-p)/n+z*z/(4*n*n))/d; return c-h,c+h
def calib(fa_ps,target=0.05):
    return float(np.quantile(np.asarray(fa_ps),target))
