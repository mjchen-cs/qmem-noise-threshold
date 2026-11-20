import sys; sys.path.insert(0,__import__('os').path.dirname(__import__('os').path.abspath(__file__)))
import numpy as np, itertools
from scipy.optimize import linprog, minimize
from cdlib import rref_basis
rng=np.random.default_rng(1)
def law(d):
    k=d+2
    while True:
        n=int(rng.integers(k,8)); M=rng.integers(0,2,size=(k,n))
        if len(rref_basis(M))==k: break
    b=rng.uniform(0,1,n); E=np.array(list(itertools.product([0,1],repeat=n)))
    pr=np.prod(np.where(E==1,(1-b)/2,(1+b)/2),axis=1)
    Z=np.array(list(itertools.product([0,1],repeat=k)))
    return {tuple(z):np.sum(pr*(1-2*((E@(z@M%2))%2))) for z in Z}
add=lambda a,b: tuple((x+y)%2 for x,y in zip(a,b))
def qform(c,d,W,G0):
    # Fourier over A (first d bits): h,g from G0 moments; x,y from W
    S=list(itertools.product([0,1],repeat=d)); D=0
    for s in S:
        ch=lambda a: (-1)**(sum(i*j for i,j in zip(a,s))%2)
        A=list(itertools.product([0,1],repeat=d))
        h=sum(ch(a)*c[a+(0,0)] for a in A); g=sum(ch(a)*c[a+(1,0)] for a in A)
        x=sum(ch(a)*W[G0.index(a+(0,))] for a in A); y=sum(ch(a)*W[G0.index(a+(1,))] for a in A)
        D+=g*(2*x*y*h-g*(x*x+y*y))/(h*(h*h-g*g))
    return D
for d in (1,2):
    G0=[tuple(z) for z in itertools.product([0,1],repeat=d+1)]; nG=len(G0); worst=9; worst_full=9
    for tr in range(200):
        c=law(d); cz=lambda g: c[g+(0,)]
        A_ub=[];b_ub=[]
        for g in G0:
            for z in G0:
                if any(z):
                    row=np.zeros(nG); row[G0.index(g)]+=cz(z); row[G0.index(add(g,z))]-=1; A_ub.append(row); b_ub.append(0)
        A_ub=np.array(A_ub)
        # also probabilities >=0 : law of (s,sig,om) Fourier inverse >=0  (homogeneous in W only for om-part -> |x|<=... ) include as linear in W with fixed c
        # minimise q over K ∩ {sum W = 1} by many random starts (SLSQP)
        cons=[{'type':'ineq','fun':lambda W: -A_ub@W},{'type':'eq','fun':lambda W: W.sum()-1}]
        for st in range(6):
            r=linprog(rng.normal(size=nG),A_ub=A_ub,b_ub=np.zeros(len(A_ub)),A_eq=np.ones((1,nG)),b_eq=[1],bounds=(0,None),method='highs')
            if r.status: continue
            res=minimize(lambda W: qform(c,d,W,G0),r.x,constraints=cons,bounds=[(0,None)]*nG,method='SLSQP')
            if (A_ub@res.x).max()<1e-9: worst=min(worst,res.fun)
    print("d",d,"min q on linear-GKS cone K (sum W=1):",worst)

# verify a d=2 counterexample explicitly
d=2; G0=[tuple(z) for z in itertools.product([0,1],repeat=d+1)]; nG=len(G0)
for tr in range(400):
    c=law(d); cz=lambda g: c[g+(0,)]
    A_ub=[]
    for g in G0:
        for z in G0:
            if any(z):
                row=np.zeros(nG); row[G0.index(g)]+=cz(z); row[G0.index(add(g,z))]-=1; A_ub.append(row)
    A_ub=np.array(A_ub); cons=[{'type':'ineq','fun':lambda W: -A_ub@W},{'type':'eq','fun':lambda W: W.sum()-1}]
    r=linprog(rng.normal(size=nG),A_ub=A_ub,b_ub=np.zeros(len(A_ub)),A_eq=np.ones((1,nG)),b_eq=[1],bounds=(0,None),method='highs')
    res=minimize(lambda W: qform(c,d,W,G0),r.x,constraints=cons,bounds=[(0,None)]*nG,method='SLSQP')
    if res.fun<-1e-3 and (A_ub@res.x).max()<1e-9: break
W=res.x*0.05
full=dict(c); 
for i,g in enumerate(G0): full[g+(1,)]=W[i]
Z=list(itertools.product([0,1],repeat=d+2)); Yp=Z
P=np.array([sum(full[z]*(-1)**(sum(a*b for a,b in zip(z,y))%2) for z in Z)/2**(d+2) for y in Yp])
print("min prob",P.min())
viol=min(full[add(a,b)]-full[a]*full[b] for a in Z for b in Z if any(a) and any(b) and a!=b)
print("min GKS-II slack",viol, " min moment",min(full.values()))
# direct L3b
Pr=P.reshape(2**d,2,2); D=0
for s in range(2**d):
    Ps=Pr[s].sum(); p=Pr[s].sum(1)/Ps; mu=(Pr[s,:,0]-Pr[s,:,1])/Pr[s].sum(1)
    D+=Ps*(p[0]-p[1])*(p[0]*mu[0]**2-p[1]*mu[1]**2)
print("direct L3b D =",D,"  q*scale^2 =",qform(c,d,W,G0)/2**d)

# Ginibre check on this law: E[prod_{i in S}(chi_i - chi_i') prod_{j in T}(chi_j + chi_j')] >= 0
Zn=[z for z in Z if any(z)]
def ginibre_min(P,maxk=4):
    Pm=P; worst=9; arg=None
    Ych=np.array([[(-1)**(sum(a*b for a,b in zip(z,y))%2) for z in Z] for y in Yp])  # rows y, cols z
    for k in range(2,maxk+1):
        for combo in itertools.combinations_with_replacement(range(1,len(Z)),k):
            for signs in itertools.product([-1,1],repeat=k):
                if sum(1 for s in signs if s<0)%2==1: continue   # odd number of minus -> antisymmetric, zero
                F=np.ones((len(Yp),len(Yp)))
                for zi,sg in zip(combo,signs): F=F*(Ych[:,zi][:,None]+sg*Ych[:,zi][None,:])
                val=Pm@F@Pm
                if val<worst: worst=val; arg=([Z[i] for i in combo],signs)
    return worst,arg
print("Ginibre min on counterexample:",ginibre_min(P))
