import numpy as np, time
from search_fastF import *
rng=np.random.default_rng(5); worst=0; cnt=0; t0=time.time()
ts=np.linspace(0,1,21)
while time.time()-t0<400:
    n=int(rng.choice([14,15,16])); dB=int(rng.integers(2,n)); dA=int(rng.integers(0,dB))
    # sparse (LDPC-like) generators half the time
    def gen():
        if rng.random()<0.5: return int(sum(1<<int(i) for i in rng.choice(n,size=int(rng.integers(2,5)),replace=False)))
        return int(rng.integers(1,1<<n))
    B=rank_basis([gen() for _ in range(dB)]); A=rank_basis([int(np.bitwise_xor.reduce([b for b in B if rng.random()<0.4] or [0])) for _ in range(dA)])
    if len(B)-len(A)<=0: continue
    r=max(ratio(A,B,n,t) for t in ts); worst=max(worst,r); cnt+=1
print(f"random CSS pairs n=14-16 (half sparse): {cnt} codes, max ratio {worst:.10f}")
