# Which priors satisfy L3b? product (yes), mixtures (no). Test ferromagnetic Ising priors P(e) ~ exp(sum h_i S_i + sum J_ij S_i S_j), S=(-1)^e, h,J>=0.
import numpy as np, itertools
from cdlib import rref_basis
rng=np.random.default_rng(8)
def D(pr,E,A,v,w):
    s=(E@A.T%2)@(1<<np.arange(len(A))) if len(A) else np.zeros(len(E),dtype=int)
    sg=E@v%2; ch=1-2*(E@w%2); S=1<<len(A)
    P=np.zeros((S,2)); U=np.zeros((S,2)); np.add.at(P,(s,sg),pr); np.add.at(U,(s,sg),pr*ch)
    ok=(P>1e-300).all(1); mu=U[ok]/P[ok]; p=P[ok]/P[ok].sum(1,keepdims=True)
    return np.sum(P[ok].sum(1)*(p[:,0]-p[:,1])*(p[:,0]*mu[:,0]**2-p[:,1]*mu[:,1]**2))
res={'ferro':9,'ferro_weakJ':9}
for tr in range(3000):
    n=int(rng.integers(2,9)); dA=int(rng.integers(0,n-1))
    A=rref_basis(rng.integers(0,2,size=(dA,n))) if dA else np.zeros((0,n),dtype=np.int64)
    v=rng.integers(0,2,size=n); w=rng.integers(0,2,size=n)
    if not v.any() or (len(A) and len(rref_basis(np.vstack([A,v[None]])))==len(A)): continue
    E=np.array(list(itertools.product([0,1],repeat=n))); Sp=1-2*E
    for key,sc in [('ferro',1.0),('ferro_weakJ',0.1)]:
        h=rng.uniform(0,1.5,n); J=np.triu(rng.uniform(0,sc,(n,n)),1)
        en=Sp@h+np.einsum('ki,ij,kj->k',Sp,J,Sp); pr=np.exp(en-en.max()); pr/=pr.sum()
        res[key]=min(res[key],D(pr,E,A,v,w))
print(res)
