import numpy as np, itertools
from cdlib import rref_basis, perp, extend
def Fv(A,B,bv,n):
    E=np.array(list(itertools.product([0,1],repeat=n))); q=(1-np.asarray(bv))/2
    prob=np.prod(np.where(E==1,q,1-q),axis=1); ext=extend(A,B); k=len(ext)
    s=(E@A.T%2)@(1<<np.arange(len(A))) if len(A) else np.zeros(len(E),dtype=int)
    l=(E@ext.T%2)@(1<<np.arange(k)) if k else np.zeros(len(E),dtype=int)
    J=np.bincount(s*(1<<k)+l,weights=prob,minlength=(1<<len(A))<<k).reshape(-1,1<<k); Ps=J.sum(1); ok=Ps>0
    return 2**k*np.sum((J[ok]**2).sum(1)/Ps[ok])
from cdlib import F
rng=np.random.default_rng(1); err=0; cnt=0
for tr in range(800):
    m=int(rng.integers(2,7)); n=m+1; dB=int(rng.integers(1,m)); dA=int(rng.integers(0,dB+1))
    B0=rref_basis(rng.integers(0,2,size=(dB,m))); A0=rref_basis(rng.integers(0,2,size=(dA,len(B0)))@B0%2) if dA else np.zeros((0,m),dtype=np.int64)
    ap=rng.integers(0,2,size=m)
    if len(rref_basis(np.vstack([B0,ap[None]])))==len(B0): continue
    pad=lambda M: np.hstack([M,np.zeros((len(M),1),dtype=np.int64)]); st=np.append(ap,1)
    A=rref_basis(np.vstack([pad(A0),st[None]])); B=rref_basis(np.vstack([pad(B0),st[None]]))
    A1=rref_basis(np.vstack([A0,ap[None]])); B1=rref_basis(np.vstack([B0,ap[None]]))
    a=rng.uniform(0,1); b=rng.uniform(0,1)
    for t,ref in [(0,F(A0,B0,b,m)),(1,F(A1,B1,b,m))]:
        err=max(err,abs(Fv(A,B,[b]*m+[t],n)-ref))
    for t,ref in [(0,F(perp(B1,m),perp(A1,m),a,m)),(1,F(perp(B0,m),perp(A0,m),a,m))]:
        err=max(err,abs(Fv(perp(B,n),perp(A,n),[a]*m+[t],n)-ref))
    cnt+=1
print(cnt,"instances, max endpoint mismatch:",err)
