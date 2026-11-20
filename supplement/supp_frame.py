"""Third session (20 November): crossings of nu = 1/2 for the unencoded pair in the standard and in the best memory frame and for
the three-qubit code (readout-mitigated; power-law fits with parametric bootstrap, as in figures/fig4_bias.py); zero-parameter
models nu(1) a^(4/3) and nu(1) ((1+2a)/3)^2. Supplementary Section S16, Table S20, Fig. S6."""
import json, numpy as np
rng=np.random.default_rng(4)
import os, sys
DATA = os.environ.get('DATA_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data'))
d=json.load(open(sys.argv[1] if len(sys.argv)>1 else os.path.join(DATA,'MS_origin_frame_mitigate.json')))
def pts(d, layer, kind, fam="dz"):
    A,V,S=[],[],[]
    for k,v in d.items():
        ly,pol=k.split("_",1)
        if ly==layer and pol.startswith(f"{kind}_{fam}") and not pol.endswith("dz100"):
            A.append(int(pol.split(fam)[1])/100); V.append(v[0]["nu"]); S.append((v[0]["ci"][1]-v[0]["ci"][0])/3.92/v[0]["nu"])
    o=np.argsort(A); return np.array(A)[o],np.array(V)[o],np.array(S)[o]
def fit(A,V,S):
    p0,c0=np.polyfit(np.log(A),np.log(V),1,w=1/S)
    chi2=np.sum(((np.log(V)-(c0+p0*np.log(A)))/S)**2)/max(len(A)-2,1); S=S*np.sqrt(max(chi2,1.0))
    def cr(lv):
        p,c=np.polyfit(np.log(A),lv,1,w=1/S); return np.exp((np.log(0.5)-c)/p),p,c
    x,p,c=cr(np.log(V)); bs=np.array([cr(np.log(V)+rng.normal(0,S))[0] for _ in range(4000)]); return x,p,c,bs,chi2
out={}; BS={}
for kind in ("bare","bareF","rep3"):
    A,V,S=pts(d,"mit",kind); x,p,c,bs,chi2=fit(A,V,S); BS[kind]=bs
    out[kind]=dict(astar=x,ci=list(np.percentile(bs,[2.5,97.5])),p=p,chi2=chi2,nu1=d[f"mit_{kind}_dz100"][0]["nu"])
    print(f"{kind:6s} a*={x:.3f} [{out[kind]['ci'][0]:.3f},{out[kind]['ci'][1]:.3f}]  p={p:.2f} chi2/dof={chi2:.2f}  nu(1)={out[kind]['nu1']:.3f}")
best=(3/np.sqrt(2)-1)/2
for a_,b_ in (("bare","bareF"),("bareF","rep3"),("bare","rep3")):
    dd=BS[a_]-BS[b_]; print(f"shift {a_}->{b_}: {np.median(dd):.3f} [{np.percentile(dd,2.5):.3f},{np.percentile(dd,97.5):.3f}]  P(>0)={np.mean(dd>0):.4f}")
print("P(rep3 < 0.561)=",np.mean(BS['rep3']<best)," P(rep3<0.5)=",np.mean(BS['rep3']<0.5), " P(bareF>0.561)=",np.mean(BS['bareF']>best))
# zero-parameter model for bareF: nu(1)*((1+2a)/3)^2 ; bare: nu(1)*a^(4/3)
for kind,f in (("bareF",lambda a:((1+2*a)/3)**2),("bare",lambda a:a**(4/3))):
    A,V,S=pts(d,"mit",kind); m=out[kind]['nu1']*f(A); print(kind,"model/meas-1:",np.round(V/m-1,3), " model crossing", end=" ")
    from scipy.optimize import brentq; print(round(brentq(lambda a: out[kind]['nu1']*f(a)-0.5,0.3,1),4))
json.dump(dict(out=out,shifts={f"{a_}-{b_}":[float(np.median(BS[a_]-BS[b_])),float(np.percentile(BS[a_]-BS[b_],2.5)),float(np.percentile(BS[a_]-BS[b_],97.5))] for a_,b_ in (("bare","bareF"),("bareF","rep3"),("bare","rep3"))},
  p_rep3_below_bestframe=float(np.mean(BS['rep3']<best)),p_rep3_below_bound=float(np.mean(BS['rep3']<0.5))),open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'supp_frame_numbers.json'),'w'),indent=1,default=float)
