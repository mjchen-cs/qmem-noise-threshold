import numpy as np
from cdlib import F, rref_basis, perp
rng=np.random.default_rng(5); Z=lambda r,n: np.zeros((0,n),dtype=np.int64)
def span(*Ms):
    M=np.vstack([m for m in Ms if len(m)]) if any(len(m) for m in Ms) else None
    return rref_basis(M) if M is not None else None
st=dict(Xint=0,Zint=0,prod=0,x_lo=9,x_hi=0,z_lo=9,z_hi=0)
for tr in range(1500):
    m=int(rng.integers(2,8)); n=m+1
    dB=int(rng.integers(1,m)); dA=int(rng.integers(0,dB+1))
    B0=rref_basis(rng.integers(0,2,size=(dB,m)))
    A0=rref_basis(rng.integers(0,2,size=(dA,len(B0)))@B0%2) if dA else Z(0,m)
    if len(A0)==0: A0=Z(0,m)
    ap=rng.integers(0,2,size=m)
    if len(rref_basis(np.vstack([B0,ap[None]])))==len(B0): continue   # a' notin B0
    k=len(B0)-len(A0)
    pad=lambda M: np.hstack([M,np.zeros((len(M),1),dtype=np.int64)])
    astar=np.append(ap,1)
    A=rref_basis(np.vstack([pad(A0),astar[None]])); B=rref_basis(np.vstack([pad(B0),astar[None]]))
    A1=rref_basis(np.vstack([A0,ap[None]])) if len(A0) else ap[None]; B1=rref_basis(np.vstack([B0,ap[None]]))
    a2=rng.uniform(0,1); b2=2/(1+a2)-1; a,b=np.sqrt(a2),np.sqrt(b2)
    X=F(A,B,b,n); X0=F(A0,B0,b,m); X1=F(A1,B1,b,m)
    Zf=F(perp(B,n),perp(A,n),a,n); Z1=F(perp(B0,m),perp(A0,m),a,m); Z0=F(perp(B1,m),perp(A1,m),a,m)
    st['Xint']=max(st['Xint'],X-((1-b2)*X0+b2*X1))
    st['Zint']=max(st['Zint'],Zf-((1-a2)*Z0+a2*Z1))
    st['prod']=max(st['prod'],((1-b2)*X0+b2*X1)*((1-a2)*Z0+a2*Z1)/2**k)
    x=X1/X0; z=Z1/Z0
    st['x_lo']=min(st['x_lo'],x); st['x_hi']=max(st['x_hi'],x); st['z_lo']=min(st['z_lo'],z); st['z_hi']=max(st['z_hi'],z)
print(st)
