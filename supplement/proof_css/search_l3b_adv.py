import numpy as np, sys, time
from search_fastF import *
from scipy.optimize import minimize
rng=np.random.default_rng(int(sys.argv[1])); T=float(sys.argv[2]); dset=[int(x) for x in sys.argv[3].split(',')]
def D(A,v,w,n,b):
    E=errors(n); p=probs(n,np.clip(b,0,1)); s=parity_index(E,A); sg=(np.bitwise_count(E&v)&1).astype(np.int64)
    ch=1-2*(np.bitwise_count(E&w)&1).astype(float); S=1<<len(A)
    P=np.bincount(2*s+sg,weights=p,minlength=2*S).reshape(S,2); U=np.bincount(2*s+sg,weights=p*ch,minlength=2*S).reshape(S,2)
    ok=(P>1e-300).all(1); P,U=P[ok],U[ok]; Ps=P.sum(1); p0,p1=P[:,0]/Ps,P[:,1]/Ps; m0,m1=U[:,0]/P[:,0],U[:,1]/P[:,1]
    return float(np.sum(Ps*(p0-p1)*(p0*m0**2-p1*m1**2)))
worst=9; t0=time.time(); cnt=0
while time.time()-t0<T:
    d=int(rng.choice(dset)); n=int(rng.integers(d+2,13))
    A=rank_basis([int(x) for x in rng.integers(1,1<<n,size=d)])
    if len(A)<d: continue
    v=int(rng.integers(1,1<<n)); 
    if in_span(v,A): continue
    w=int(rng.integers(0,1<<n))
    best=9
    for st in range(3):
        x0=rng.uniform(0,1,n)
        r=minimize(lambda b: D(A,v,w,n,b),x0,method='L-BFGS-B',bounds=[(0,1)]*n,options={'maxiter':150}); best=min(best,r.fun)
    cnt+=1; worst=min(worst,best)
    if best<-1e-9: print("VIOLATION",n,d,A,v,w,best,flush=True)
print(f"d in {dset}: {cnt} instances (n<=12, biases optimised per qubit), min D = {worst:.3e}")
