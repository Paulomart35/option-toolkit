import numpy as np
from scipy.stats import norm
def bs(S,K,T,r,s,q=0,kind='call'):
    d1=(np.log(S/K)+(r-q+.5*s*s)*T)/(s*np.sqrt(T)); d2=d1-s*np.sqrt(T)
    if kind=='call': return S*np.exp(-q*T)*norm.cdf(d1)-K*np.exp(-r*T)*norm.cdf(d2)
    return K*np.exp(-r*T)*norm.cdf(-d2)-S*np.exp(-q*T)*norm.cdf(-d1)
def delta(S,K,T,r,s,kind):
    d1=(np.log(S/K)+(r+.5*s*s)*T)/(s*np.sqrt(T)); return norm.cdf(d1) if kind=='call' else norm.cdf(d1)-1
def gamma(S,K,T,r,s):
    d1=(np.log(S/K)+(r+.5*s*s)*T)/(s*np.sqrt(T)); return norm.pdf(d1)/(S*s*np.sqrt(T))
print("T1 BS call/put 100,100,1,.05,.2:", bs(100,100,1,.05,.2), bs(100,100,1,.05,.2,kind='put'))
print("T2 HW4 BS 200,210,120/365,.05,.4:", bs(200,210,120/365,.05,.4))
print("T3 HW5 S40 K40 .3 .05 q.02 .5 call:", bs(40,40,.5,.05,.3,.02))
# hedge HW7 defaults
S,r,s=100,.03,.2; N=10
dw,gw=delta(S,100,1,r,s,'call'),gamma(S,100,1,r,s); dh,gh=delta(S,110,2,r,s,'call'),gamma(S,110,2,r,s)
nh=N*gw/gh; ns=N*dw-nh*dh
print("T4 delta hedge shares:",N*dw," | DG: nh",nh,"ns",ns, "dw gw dh gh",dw,gw,dh,gh)
# parity arbitrage example
S,K,T,r=100,100,1,.05; C=bs(S,K,T,r,.2); P=bs(S,K,T,r,.2,kind='put')
print("T5 parity C-P",C-P,"S-PV(K)",S-K*np.exp(-r*T),"PV(K)",K*np.exp(-r*T))
print("   if market call=12 -> profit today", 12-(P+S-K*np.exp(-r*T)))
# Hagan
def sabr_vol(K,F,T,a,b,rho,nu):
    K=np.asarray(K,float);FK=F*K;l=np.log(F/K);z=(nu/a)*FK**((1-b)/2)*l
    x=np.log((np.sqrt(1-2*rho*z+z*z)+z-rho)/(1-rho))
    zx=np.where(np.abs(z)<1e-7,1.0,z/np.where(np.abs(z)<1e-7,1.0,x))
    den=FK**((1-b)/2)*(1+(1-b)**2/24*l**2+(1-b)**4/1920*l**4)
    c=1+((1-b)**2/24*a*a/FK**(1-b)+rho*b*nu*a/(4*FK**((1-b)/2))+(2-3*rho**2)/24*nu*nu)*T
    return a/den*zx*c
print("T6 Hagan F=410 T=.5 a=.55 b=1 rho=-.4 nu=1.2 K=300,350,410,450,500:", sabr_vol([300,350,410,450,500],410,.5,.55,1,-.4,1.2))
# MC prof std_MC with r drift
def mc(S0,K,T,r,s0,nu,rho,npaths,nsteps,scheme,seed=42):
    rng=np.random.default_rng(seed); dt=T/nsteps; X=np.zeros(npaths); v=np.full(npaths,s0)
    for _ in range(nsteps):
        z1=rng.standard_normal(npaths); z2=rng.standard_normal(npaths)
        dW=rho*z1+np.sqrt(1-rho**2)*z2
        vp=np.maximum(v,0) if scheme!='euler' else v
        X+= (r-.5*vp**2)*dt+vp*dW*np.sqrt(dt)
        if scheme=='log': v=v*np.exp(nu*z1*np.sqrt(dt)-.5*nu*nu*dt)
        else: v=v+nu*v*z1*np.sqrt(dt)
    ST=S0*np.exp(X); pay=np.maximum(ST-K,0)
    return np.exp(-r*T)*pay.mean(), np.exp(-r*T)*pay.std()/np.sqrt(npaths), ST.mean()
print("T7 MC nu~0 (->BS 10.4506):", mc(100,100,1,.05,.2,1e-8,0,200000,100,'log'))
for sc in ['euler','full','log']:
    print("T8 MC",sc, mc(100,100,1,.05,.2,.5,-.4,200000,252,sc))
F=100*np.exp(.05); hv=sabr_vol(100,F,1,.2,1,-.4,.5)
d1=(np.log(F/100)+.5*hv**2)/hv; print("   Hagan vol",hv,"Black price",np.exp(-.05)*(F*norm.cdf(d1)-100*norm.cdf(d1-hv)))
