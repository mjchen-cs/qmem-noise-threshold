# Fourier-multiplier formulation: q = sum_t rho(t) U_t^2, rho(t)=2/P_s - 1/P_t (t=(s,sigma)).
# Certificate <=> rho in cone{ psi_i(t) psi_j(t) chi_t(y) } (+ nonneg functions), psi = Ginibre multipliers.
import sys; sys.path.insert(0,__import__('os').path.dirname(__import__('os').path.abspath(__file__)))
import numpy as np, itertools
from scipy.optimize import linprog
from cdlib import rref_basis
rng=np.random.default_rng(int(sys.argv[2])); d=int(sys.argv[1]); K=d+1
G0=[np.array(g) for g in itertools.product([0,1],repeat=K)]; T=G0
chi=np.array([[(-1)**(int(t@g)%2) for g in G0] for t in T])   # chi[t,g]
def law():
    while True:
        n=int(rng.integers(K,9)); M=rng.integers(0,2,size=(K,n))
        if len(rref_basis(M))==K: break
    b=rng.uniform(0,1,n); E=np.array(list(itertools.product([0,1],repeat=n)))
    pr=np.prod(np.where(E==1,(1-b)/2,(1+b)/2),axis=1)
    return np.array([np.sum(pr*(1-2*(E@(g@M%2)%2))) for g in G0])
gi={tuple(g):i for i,g in enumerate(G0)}
ok=0; trials=int(sys.argv[3])
for tr in range(trials):
    c=law(); P=chi@c/2**K                       # P_t
    Ps=np.array([P[i]+P[gi[tuple((t+np.eye(K,dtype=int)[-1])%2)]] for i,t in enumerate(T)])   # sigma = last bit
    rho=2/Ps-1/P
    psis=set()
    for z,zp in itertools.combinations_with_replacement(range(len(G0)),2):
        for e,ep in itertools.product([-1,1],repeat=2):
            zz=gi[tuple((G0[z]+G0[zp])%2)]
            psi=chi[:,zz]+e*c[z]*chi[:,zp]+ep*c[zp]*chi[:,z]+e*ep*c[zz]
            if np.abs(psi).max()>1e-12: psis.add(tuple(np.round(psi,12)))
    psis=[np.array(p) for p in psis]
    cols=[]
    for i in range(len(psis)):
        for j in range(i,len(psis)):
            pp=psis[i]*psis[j]
            for y in range(len(G0)): cols.append(pp*chi[:,y])
    cols+=list(np.eye(len(T)))            # nonneg functions (PSD part)
    A=np.array(cols).T
    r=linprog(np.zeros(A.shape[1]),A_eq=A,b_eq=rho,bounds=(0,None),method='highs')
    ok+= r.status==0
print(f"d={d}: multiplier-cone certificate exists in {ok}/{trials}  (#psi={len(psis)}, #cols={A.shape[1]})")
