import numpy as np, sys, time
from search_fastF import *
rng=np.random.default_rng(int(sys.argv[1])); n=int(sys.argv[2]); T=float(sys.argv[3])
ts=np.linspace(0.15,0.85,8)
def score(A,B):
    if len(rank_basis(B))-len(rank_basis(A))<=0: return -1
    return max(ratio(A,B,n,t) for t in ts)
def rand_code():
    dB=int(rng.integers(2,n)); dA=int(rng.integers(0,dB))
    B=[int(x) for x in rng.integers(1,1<<n,size=dB)]; A=B[:dA]
    return A,B
best_overall=0; t0=time.time(); it=0
while time.time()-t0<T:
    A,B=rand_code(); s=score(A,B); stall=0
    while stall<60 and time.time()-t0<T:
        A2,B2=list(A),list(B); r=rng.random()
        if r<0.4 and B2: i=int(rng.integers(len(B2))); B2[i]^=1<<int(rng.integers(n)); A2=[v for v in A2]  # mutate a B generator
        elif r<0.7 and A2: i=int(rng.integers(len(A2))); A2[i]=A2[i]^ B2[int(rng.integers(len(B2)))]
        elif r<0.85: B2.append(int(rng.integers(1,1<<n)))
        else:
            if B2: B2.pop(int(rng.integers(len(B2))))
        A2=[a for a in A2 if in_span(a,rank_basis(B2))] if B2 else []
        s2=score(A2,B2) if B2 else -1
        if s2>s: A,B,s=A2,B2,s2; stall=0
        else: stall+=1
        it+=1
    full=max(ratio(A,B,n,t) for t in np.linspace(0,1,41))
    best_overall=max(best_overall,full)
    if full>1+1e-9: print("VIOLATION",n,A,B,full,flush=True)
print(f"n={n}: {it} evaluations, best interior score climbed; max full-boundary ratio found = {best_overall:.10f}")
