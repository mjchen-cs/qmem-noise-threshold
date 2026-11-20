"""Fast exact F(A,B;b) = 2^k sum_{s,l} P(s,l)^2/P(s) for subspaces given as lists of int bit-vectors (n <= ~20)."""
import numpy as np
_cache={}
def errors(n):
    if n not in _cache: _cache[n]=np.arange(1<<n,dtype=np.int64)
    return _cache[n]
def probs(n,b):
    b=np.broadcast_to(np.asarray(b,float),(n,)); p=np.ones(1,float)
    for i in range(n):           # bit i of e  <->  coordinate i
        q=(1-b[i])/2; p=np.concatenate([p*(1-q),p*q])   # new bit is the highest so far
    return p
def rank_basis(vecs):
    basis=[]  # gaussian elimination over GF(2) on ints
    for v in vecs:
        for bb in basis: v=min(v,v^bb)
        if v: basis.append(v)
    return basis
def in_span(v,basis):
    for bb in basis: v=min(v,v^bb)
    return v==0
def parity_index(E,vecs):
    idx=np.zeros(len(E),dtype=np.int64)
    for j,v in enumerate(vecs):
        idx|=((np.bitwise_count(E&v)&1).astype(np.int64))<<j
    return idx
def F(A,B,n,b):
    A=rank_basis(A); ext=[]; cur=list(A)
    for v in B:
        if not in_span(v,cur): cur=rank_basis(cur+[v]); ext.append(v)
    k=len(ext); E=errors(n); p=probs(n,b)
    s=parity_index(E,A); l=parity_index(E,ext)
    J=np.bincount((s<<k)|l,weights=p,minlength=(1<<len(A))<<k).reshape(-1,1<<k)
    Ps=J.sum(1); ok=Ps>0
    return float((2**k)*np.sum((J[ok]**2).sum(1)/Ps[ok])), k
def perp(V,n):
    V=rank_basis(V); out=[]
    # brute force over a basis complement: solve via RREF on bit matrix
    M=np.array([[(v>>i)&1 for i in range(n)] for v in V],dtype=np.uint8).reshape(-1,n)
    r=0; piv=[]; M=M.copy()
    for c in range(n):
        pr=next((i for i in range(r,len(M)) if M[i,c]),None)
        if pr is None: continue
        M[[r,pr]]=M[[pr,r]]
        for i in range(len(M)):
            if i!=r and M[i,c]: M[i]^=M[r]
        piv.append(c); r+=1
    free=[c for c in range(n) if c not in piv]
    for f in free:
        x=[0]*n; x[f]=1
        for i,pc in enumerate(piv): x[pc]=int(M[i,f])
        out.append(sum(bit<<i for i,bit in enumerate(x)))
    return out
def ratio(A,B,n,t,bvec=None,avec=None):
    """F(A,B;b) F(B^perp,A^perp;a) / 2^k on the boundary (1+a^2)(1+b^2)=2 with b=t (or per-qubit vectors)."""
    if bvec is None:
        bvec=np.full(n,t); avec=np.sqrt(np.maximum(2/(1+bvec**2)-1,0))
    fx,k=F(A,B,n,bvec); fz,k2=F(perp(B,n),perp(A,n),n,avec); assert k==k2
    return fx*fz/2**k
