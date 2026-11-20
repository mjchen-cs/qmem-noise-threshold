# per-w test: E_{s,sig}[E[chi_w|s,sig]^2] <= E_s[E[chi_w|s]^2] + E_s[E[chi_{w+v}|s]^2]
import numpy as np, itertools
from cdlib import rref_basis
rng=np.random.default_rng(7); worst=-9; cnt=0
for tr in range(4000):
    n=int(rng.integers(2,10)); dA=int(rng.integers(0,n-1))
    A=rref_basis(rng.integers(0,2,size=(dA,n))) if dA else np.zeros((0,n),dtype=np.int64)
    v=rng.integers(0,2,size=n); w=rng.integers(0,2,size=n)
    if not v.any() or (len(A) and len(rref_basis(np.vstack([A,v[None]])))==len(A)): continue
    b=rng.uniform(0,1); q=(1-b)/2
    E=np.array(list(itertools.product([0,1],repeat=n))); wt=E.sum(1); pr=q**wt*(1-q)**(n-wt)
    s=(E@A.T%2)@(1<<np.arange(len(A))) if len(A) else np.zeros(len(E),dtype=int)
    sg=E@v%2; cw=1-2*(E@w%2); cwv=1-2*(E@((w+v)%2)%2)
    def G(lab,chi):
        num=np.bincount(lab,weights=pr*chi); den=np.bincount(lab,weights=pr); ok=den>0
        return np.sum(num[ok]**2/den[ok])
    lhs=G(2*s+sg,cw); rhs=G(s,cw)+G(s,cwv)
    worst=max(worst,lhs-rhs); cnt+=1
print(cnt,"max lhs-rhs:",worst)
