"""Stationary diagnostics for inventory probes whose support scales with budget."""
import json
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import brentq,minimize_scalar
import inventory_sequential as seq
from threshold_study import segment_integral

OUT=seq.inv.OUT/'budget_width'

def statistics(theta,h,n=128):
    q,gamma=theta;xx=[];ww=[];nodes,weights=leggauss(n)
    pieces=[(0.,2.)] if h==0 else [(0.,2.-h),(2.-h,2.)]
    for lo,hi in pieces:
        xx.extend((lo+hi)/2+(hi-lo)*nodes/2);ww.extend((hi-lo)*weights/2)
    x=np.array(xx);weights=np.array(ww)
    z=np.zeros_like(x) if h==0 else np.clip((x-2+h)/h,0,1)
    effort=z*z*(3-2*z)
    primitive=h*(2-h)*(z**3-.5*z**4)+h*h*(.75*z**4-.4*z**5)
    potential=(.4*x-q*x*x+2*gamma*primitive)/.25
    mass=weights*np.exp(potential);mass/=mass.sum()
    f=np.column_stack([-x,effort*x])/.5
    return np.einsum('n,ni,nj->ij',mass,f,f),float(mass@(effort*effort))

def boundary(q,target):
    return brentq(lambda g:seq.delta((q,g))-target,.01,.4,xtol=1e-14)

def design_rate(theta,matrix):
    sign=1 if seq.delta(theta)>0 else -1;target=-sign*.05
    def value(q):
        root=boundary(q,target)
        unconstrained=theta[1]-matrix[0,1]/matrix[1,1]*(q-theta[0])
        lo,hi=(.01,root) if sign>0 else (root,.4)
        g=np.clip(unconstrained,lo,hi);v=np.array([q,g])-theta
        return float(v@matrix@v/2)
    res=minimize_scalar(value,bounds=(.45,.6),method='bounded',options={'xatol':1e-14})
    return min(res.fun,value(theta[0]),value(.45),value(.6))

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    pstar=np.exp((.8-.6*4)/.25)/segment_integral(.6,0,2)
    kappa=1.1*35/(13*pstar);rows=[]
    for theta in [np.array([.5,.05]),np.array([.5,.3])]:
        baseline,_=statistics(theta,0.)
        opposite=boundary(theta[0],-.05 if theta[1]==.3 else .05)
        slope=8*(theta[1]-opposite)**2
        previous=None
        for width in [.5,.25,.125,.0625,.03125,.015625]:
            budget=width/kappa;matrix,cost=statistics(theta,width)
            refined,c2=statistics(theta,width,256)
            assert np.max(abs(matrix-refined))<1e-12 and abs(cost-c2)<1e-13
            assert cost>budget
            weight=budget/cost;mix=(1-weight)*baseline+weight*matrix
            rate=design_rate(theta,mix);gap=budget*slope-rate
            assert gap>=-1e-12
            rows.append(dict(theta=theta.tolist(),width=width,budget=budget,probe_cost=cost,
                             probe_time_weight=weight,design_rate=rate,unrestricted_upper=budget*slope,
                             relative_information_loss=gap/(budget*slope),gap_over_budget_squared=gap/budget**2))
    record={'kappa':kappa,'baseline_density_lower':pstar,'rows':rows,
            'scope':'Floating-point stationary diagnostics over scalar decision-boundary curves; their regularity and level-set structure are checked separately by verify_inventory_boundary.py.'}
    (OUT/'results.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2))

if __name__=='__main__':main()
