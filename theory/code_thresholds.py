# -*- coding: utf-8 -*-
"""Per-site factors of coded memories under depolarizing noise (fidelity eta), computed exactly with
Walsh-Hadamard transforms over the stabilizer group, and the thresholds eta* where they cross 1/2.
  q_s = 2^-r sum_{g in S} (-1)^{s.g} eta^wt(g),  A_s(sigma) = 2^-r sum_{g in S} (-1)^{s.g} eta^wt(g sigma)
  nu_soft = (sum_s A_s^2 / q_s)^(1/k)  (syndrome-aware decoding),  nu_ps = (A_0^2 / q_0)^(1/k)  (post-selection)
Codes: [[m,m-2,2]] (X^m, Z^m); [[5,1,3]]; quantum Hamming [[2^r-1, 2^r-1-2r, 3]]; [[2^r, 2^r-2r-2, 4]] = CSS(RM(1,r), RM(1,r)).
Usage: python code_thresholds.py"""
import numpy as np, itertools
# Pauli on m qubits = bool vector (x|z) of length 2m
def symp(a,b,m): return (np.sum(a[...,:m]&b[...,m:],-1)+np.sum(a[...,m:]&b[...,:m],-1))%2
def gf2_rank(M):
    M=M.copy().astype(np.uint8); r=0; rows,cols=M.shape
    for c in range(cols):
        p=np.nonzero(M[r:,c])[0]
        if len(p)==0: continue
        p=p[0]+r; M[[r,p]]=M[[p,r]]
        for i in range(rows):
            if i!=r and M[i,c]: M[i]^=M[r]
        r+=1
        if r==rows: break
    return r
def nullspace(M):   # vectors v with M v = 0 over GF(2)
    M=M.copy().astype(np.uint8); rows,cols=M.shape; piv=[]; r=0
    for c in range(cols):
        p=np.nonzero(M[r:,c])[0] if r<rows else []
        if len(p)==0: continue
        p=p[0]+r; M[[r,p]]=M[[p,r]]
        for i in range(rows):
            if i!=r and M[i,c]: M[i]^=M[r]
        piv.append(c); r+=1
        if r==rows: break
    free=[c for c in range(cols) if c not in piv]; basis=[]
    for f in free:
        v=np.zeros(cols,np.uint8); v[f]=1
        for i,c in enumerate(piv):
            if M[i,f]: v[c]=1
        basis.append(v)
    return np.array(basis)
def logicals(S,m):
    # normalizer: v with symp(v,s)=0 for all s: matrix Omega S
    Om=np.concatenate([S[:,m:],S[:,:m]],1)
    N=nullspace(Om)
    # remove stabilizer span, symplectic Gram-Schmidt
    cur=S.copy(); cand=[]
    for v in N:
        if gf2_rank(np.vstack([cur,v]))>gf2_rank(cur): cur=np.vstack([cur,v]); cand.append(v)
    cand=[c.copy() for c in cand]; X=[];Z=[]
    while cand:
        a=cand.pop(0); j=next((i for i,b in enumerate(cand) if symp(a,b,m)),None)
        b=cand.pop(j); X.append(a); Z.append(b)
        for i,c in enumerate(cand):
            ca=symp(c,b,m); cb=symp(c,a,m)
            if ca: c^=a
            if cb: c^=b
            cand[i]=c
    return np.array(X),np.array(Z)
def css(Hx,Hz):
    m=Hx.shape[1]
    return np.vstack([np.hstack([Hx,np.zeros_like(Hx)]),np.hstack([np.zeros_like(Hz),Hz])]).astype(np.uint8)
def code_lib():
    C={}
    for m in [4,6,8,12,16]:
        C[f"[[{m},{m-2},2]]"]=css(np.ones((1,m),np.uint8),np.ones((1,m),np.uint8))
    # [[5,1,3]]
    g=["XZZXI","IXZZX","XIXZZ","ZXIXZ"]; S=[]
    for s in g:
        x=[1 if c in "XY" else 0 for c in s]; z=[1 if c in "ZY" else 0 for c in s]; S.append(x+z)
    C["[[5,1,3]]"]=np.array(S,np.uint8)
    for r in [3,4,5]:
        H=np.array([[ (j>>i)&1 for j in range(1,2**r)] for i in range(r)],np.uint8); m=2**r-1
        C[f"[[{m},{m-2*r},3]]"]=css(H,H)
    for r in [4,5,6]:
        m=2**r; G=np.vstack([np.ones(m,np.uint8)]+[np.array([(j>>i)&1 for j in range(m)],np.uint8) for i in range(r)])
        C[f"[[{m},{m-2*(r+1)},4]]"]=css(G,G)
    return C
def group(S):
    r=len(S); m2=S.shape[1]; Gp=np.zeros((2**r,m2),np.uint8)
    for j in range(r):
        Gp[2**j:2**(j+1)]=Gp[:2**j]^S[j]
    return Gp   # index c (bit j = generator j) -> element
def wht(a):
    a=a.astype(np.float64).copy(); h=1; N=len(a)
    while h<N:
        v=a.reshape(-1,2,h); x=v[:,0].copy(); v[:,0]+=v[:,1]; v[:,1]=x-v[:,1]; h*=2
    return a
def weight(P,m): return np.sum(P[...,:m]|P[...,m:],-1)
def figures(S,LX,LZ,eta,nsig=48,rng=np.random.default_rng(0)):
    m=S.shape[1]//2; r=len(S); k=len(LX); Gp=group(S)
    q=wht(eta**weight(Gp,m))/2**r            # q_s indexed by syndrome bit pattern (WHT over c)
    out_soft=[];out_ps=[]
    sigs=[tuple(rng.integers(1,4,size=k)) for _ in range(nsig)] if 3**k>nsig else list(itertools.product([1,2,3],repeat=k))
    for sg in sigs:
        L=np.zeros(2*m,np.uint8)
        for i,s in enumerate(sg):
            if s in (1,2): L^=LX[i]
            if s in (3,2): L^=LZ[i]
        A=wht(eta**weight(Gp^L,m))/2**r
        ok=q>1e-15
        G=np.sum(A[ok]**2/q[ok]); out_soft.append(G**(1/k)); out_ps.append((A[0]**2/q[0])**(1/k))
    return np.array(out_soft),np.array(out_ps),q[0]


def threshold(fun, lo=0.3, hi=0.999):
    for _ in range(22):
        mid = (lo + hi) / 2; v = fun(mid); lo, hi = (mid, hi) if v < 0.5 else (lo, mid)
    return (lo + hi) / 2


if __name__ == "__main__":
    C = code_lib()
    print(f"{'code':14s} {'m/k':>5s}  {'eta* soft(gm)':>13s} {'eta* soft(min)':>14s} {'eta* postsel(min)':>17s}  nu_soft(1/3) max")
    print(f"{'Bell (bare)':14s} {1:5.2f}  {0.5**0.5:13.4f} {0.5**0.5:14.4f} {0.5**0.5:17.4f}  {1/9:.3f}")
    for name, S in C.items():
        m = S.shape[1] // 2; LX, LZ = logicals(S, m); k = len(LX)
        assert np.all(symp(S[:, None, :], S[None, :, :], m) == 0)
        f = lambda e, i, red: red(figures(S, LX, LZ, e)[i])
        a = threshold(lambda e: f(e, 0, lambda x: np.exp(np.mean(np.log(x)))))
        b = threshold(lambda e: f(e, 0, np.min)); c = threshold(lambda e: f(e, 1, np.min))
        print(f"{name:14s} {m / k:5.2f}  {a:13.4f} {b:14.4f} {c:17.4f}  {f(1 / 3, 0, np.max):.3f}", flush=True)
