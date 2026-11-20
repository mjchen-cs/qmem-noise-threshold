import numpy as np, itertools
from search_fastF import *
from scipy.optimize import minimize_scalar
def maxratio(A,B,n,grid=21):
    ts=np.linspace(0.0,1.0,grid); vals=[ratio(A,B,n,t) for t in ts]; i=int(np.argmax(vals))
    lo,hi=ts[max(i-1,0)],ts[min(i+1,grid-1)]
    r=minimize_scalar(lambda t:-ratio(A,B,n,t),bounds=(lo,hi),method='bounded',options={'xatol':1e-6})
    return max(max(vals),-r.fun), (r.x if -r.fun>=max(vals) else ts[i])
def css(CX,CZ,n):
    """A=C_Z, B=C_X^perp; checks C_X subset C_Z^perp."""
    for x in CX:
        for z in CZ: assert bin(x&z).count('1')%2==0, "not CSS"
    return rank_basis(CZ), perp(CX,n) if CX else [1<<i for i in range(n)]
def vec(bits): return sum(1<<i for i in bits)
def rep_checks(n,off=0): return [vec([off+i,off+i+1]) for i in range(n-1)]
fam={}
for n in range(2,17): fam[f"phase-flip rep n={n}"]=(n,[],rep_checks(n))
for n in range(2,17): fam[f"bit-flip rep n={n}"]=(n,rep_checks(n),[])
for n in range(4,17,2): fam[f"[[{n},{n-2},2]]"]=(n,[vec(range(n))],[vec(range(n))])
for n in range(2,17): fam[f"single Z-parity n={n} (k=n-1)"]=(n,[],[vec(range(n))])
for n1,n2 in [(2,2),(2,3),(3,2),(3,3),(2,4),(4,2),(3,4),(4,3),(2,5),(5,2),(2,6),(6,2),(2,7),(7,2),(2,8),(8,2),(4,4),(3,5),(5,3)]:
    n=n1*n2; CZ=[c for b in range(n2) for c in rep_checks(n1,b*n1)]; CX=[vec(range(b*n1,(b+2)*n1)) for b in range(n2-1)]
    fam[f"Shor {n1}x{n2}"]=(n,CX,CZ); fam[f"Shor-dual {n1}x{n2}"]=(n,CZ,CX)
# Hamming-based
def hamming_dual(m):
    n=2**m-1; return n,[sum(((j+1)>>r&1)<<j for j in range(n)) for r in range(m)]
for m in (3,4):
    n,H=hamming_dual(m); fam[f"quantum Hamming [[{n},{n-2*m}]]"]=(n,H,H)
    fam[f"Hamming X-only m={m}"]=(n,H,[]); fam[f"Hamming Z-only m={m}"]=(n,[],H)
# Reed-Muller m=4
def RM(r,m):
    n=2**m; mons=[s for d in range(r+1) for s in itertools.combinations(range(m),d)]
    return [sum((1 if all((x>>i)&1 for i in S) else 0)<<x for x in range(n)) for S in mons]
for r1 in range(0,4):
    for r2 in range(0,4):
        if r1<=4-r2-1: 
            CX,CZ=RM(r1,4),RM(r2,4); 
            if len(rank_basis(CX))+len(rank_basis(CZ))<16: fam[f"RM({r1},4)/RM({r2},4)"]=(16,CX,CZ)
# rotated surface d=3 [[9,1,3]] and toric 2x2/3x3 via hypergraph product of rep codes
def hgp(H1,n1,H2,n2):
    # H1: list of check-int over n1 bits (rows), H2 similarly; qubits = n1*n2 + m1*m2
    m1,m2=len(H1),len(H2); N=n1*n2+m1*m2
    q1=lambda i,j: i*n2+j; q2=lambda a,b: n1*n2+a*m2+b
    X=[];Z=[]
    for a in range(m1):          # X checks indexed (a, j)
        for j in range(n2):
            bits=[q1(i,j) for i in range(n1) if (H1[a]>>i)&1]+[q2(a,b) for b in range(m2) if (H2[b]>>j)&1]; X.append(vec(bits))
    for i in range(n1):          # Z checks indexed (i, b)
        for b in range(m2):
            bits=[q1(i,j) for j in range(n2) if (H2[b]>>j)&1]+[q2(a,b) for a in range(m1) if (H1[a]>>i)&1]; Z.append(vec(bits))
    return N,X,Z
for L in (2,3):
    rep=rep_checks(L); N,X,Z=hgp(rep,L,rep,L); fam[f"surface (HGP rep{L})"]=(N,X,Z)
    cyc=[vec([i,(i+1)%L]) for i in range(L)] if L>2 else [vec([0,1]),vec([0,1])]
    if L==3: N,X,Z=hgp(cyc,L,cyc,L); fam[f"toric 3x3"]=(N,X,Z)
res=[]
for name,(n,CX,CZ) in fam.items():
    A,B=css(CX,CZ,n); k=len(rank_basis(B))-len(A)
    if k<=0 or n>18: continue
    r,t=maxratio(A,B,n); res.append((r,name,n,k,t))
for r,name,n,k,t in sorted(res,reverse=True)[:25]: print(f"{r:.6f}  {name:32s} n={n:2d} k={k:2d} at b={t:.3f}")
print("families tested:",len(res),"  max ratio:",max(r for r,*_ in res))
