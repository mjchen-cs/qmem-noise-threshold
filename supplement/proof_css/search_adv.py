import numpy as np, time, sys
from search_fastF import *
from scipy.optimize import minimize
exec(open(__import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)),'search_families.py')).read().split('res=[]')[0])
rng=np.random.default_rng(int(sys.argv[1]) if len(sys.argv)>1 else 0)
def R_vec(A,B,n,tv):
    tv=np.clip(tv,0,1); av=np.sqrt(np.maximum(2/(1+tv**2)-1,0)); return ratio(A,B,n,None,tv,av)
def opt_pervqubit(A,B,n,starts=4):
    best=0
    for s in range(starts):
        x0=rng.uniform(0.05,0.95,n)
        r=minimize(lambda x:-R_vec(A,B,n,x),x0,method='L-BFGS-B',bounds=[(0,1)]*n,options={'maxiter':200})
        best=max(best,-r.fun)
    return best
# (a) per-qubit biases on structured families (n<=12 to keep it fast)
t0=time.time(); worst=0; arg=None
for name,(n,CX,CZ) in fam.items():
    if n>12: continue
    A,B=css(CX,CZ,n)
    if len(rank_basis(B))-len(A)<=0: continue
    r=opt_pervqubit(A,B,n,3)
    if r>worst: worst,arg=r,name
print(f"(a) per-qubit biases, structured codes n<=12: max ratio {worst:.9f} ({arg})  [{time.time()-t0:.0f}s]",flush=True)
# (b) random CSS codes n=8..12, per-qubit biases
t0=time.time(); worst=0
for trial in range(60):
    n=int(rng.integers(8,13)); dB=int(rng.integers(2,n)); dA=int(rng.integers(0,dB))
    B=rank_basis([int(x) for x in rng.integers(1,1<<n,size=dB)]); 
    A=rank_basis([int(np.bitwise_xor.reduce([b for b in B if rng.random()<0.5] or [0])) for _ in range(dA)])
    if len(rank_basis(B))-len(A)<=0: continue
    r=opt_pervqubit(A,B,n,2); worst=max(worst,r)
print(f"(b) random codes n=8-12, per-qubit biases: max ratio {worst:.9f}  [{time.time()-t0:.0f}s]",flush=True)
