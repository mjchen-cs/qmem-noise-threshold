import sympy as sp
t,a,ap,c,cp,r=sp.symbols('t a ap c cp r',positive=True)
psi=lambda T: sum((((1+T)/2)**2*C+((1-T)/2)**2*Cp+2*((1+T)/2)*((1-T)/2)*r)/(((1+T)/2)*X+((1-T)/2)*Xp) for C,Cp,X,Xp in [(c,cp,a,ap),(cp,c,ap,a)])
gap=sp.factor(sp.simplify((1-t**2)*psi(0)+t**2*psi(1)-psi(t)))
print(gap)
target=t**2*(1-t**2)*(a-ap)**2*(a**2*cp+ap**2*c-2*a*ap*r)/(a*ap*(a+ap)*(a+ap-t*(a-ap))*(a+ap+t*(a-ap)))
print("identity holds:",sp.simplify(gap-target)==0)
