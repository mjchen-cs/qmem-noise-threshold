import sys; sys.path.insert(0,__import__('os').path.dirname(__import__('os').path.abspath(__file__)))
import numpy as np, itertools
from scipy.optimize import linprog, minimize
exec(open(__import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)),'counterexample_gks2.py')).read().split("for d in (1,2):")[0])
rng=np.random.default_rng(int(sys.argv[2]) if len(sys.argv)>2 else 3)
def gin_rows(c,d,G0,kmax):
    # constraint: E[(chi_x + e0 chi_x') prod_i (chi_{z_i} + e_i chi_{z_i}')] >= 0, x in w+G0 (index g), z_i in G0\{0}
    # expand: sum over subsets T of factors taken from Y' : coefficient prod(e for factors in T) * E[chi_{sum not T}] E[chi_{sum T}]
    rows=[]
    Gn=[g for g in G0 if any(g)]
    for k in range(1,kmax):
        for zs in itertools.combinations_with_replacement(Gn,k):
            for signs in itertools.product([-1,1],repeat=k+1):
                if sum(1 for s in signs if s<0)%2: continue
                for gi,g in enumerate(G0):
                    row=np.zeros(len(G0))
                    facs=[('w',g)]+[('z',z) for z in zs]
                    for T in itertools.product([0,1],repeat=k+1):
                        coef=np.prod([signs[i] if T[i] else 1 for i in range(k+1)])
                        def sumf(sel):
                            hasw=False; acc=(0,)*(d+1)
                            for i,(t,f) in enumerate(facs):
                                if sel[i]: acc=add(acc,f); hasw ^= (t=='w')
                            return hasw,acc
                        h1,a1=sumf(T); h0,a0=sumf([1-t for t in T])
                        # exactly one of h0,h1 is True
                        if h1: row[G0.index(a1)]+=coef*c[a0+(0,)]
                        else:  row[G0.index(a0)]+=coef*c[a1+(0,)]
                    rows.append(row)
    return np.array(rows)
d=int(sys.argv[1]) if len(sys.argv)>1 else 2
G0=[tuple(z) for z in itertools.product([0,1],repeat=d+1)]; nG=len(G0)
for kmax in [int(x) for x in sys.argv[3].split(",")] if len(sys.argv)>3 else (2,3,4):
    worst=9
    for tr in range(int(sys.argv[4]) if len(sys.argv)>4 else 60):
        c=law(d); R=gin_rows(c,d,G0,kmax)
        cons=[{'type':'ineq','fun':lambda W: R@W},{'type':'eq','fun':lambda W: W.sum()-1}]
        for st in range(4):
            r=linprog(rng.normal(size=nG),A_ub=-R,b_ub=np.zeros(len(R)),A_eq=np.ones((1,nG)),b_eq=[1],bounds=(0,None),method='highs')
            if r.status: continue
            res=minimize(lambda W: qform(c,d,W,G0),r.x,constraints=cons,bounds=[(0,None)]*nG,method='SLSQP')
            if (R@res.x).min()>-1e-9: worst=min(worst,res.fun)
    print(f"d={d} Ginibre up to {kmax} factors (one w-factor): min q on cone = {worst:.3e}",flush=True)
