"""Interval verification of a regular preservation-comparison boundary."""
import json
import numpy as np
from mpmath import iv
import inventory_frontier as inv

def enclosure(rate,n=400):
    h=iv.mpf(2)/n;sums=[iv.mpf(0) for _ in range(4)]
    for j in range(n+1):
        x=j*h;f=iv.exp((iv.mpf('.4')*x-rate*x*x)/iv.mpf('.25'))
        w=1 if j in (0,n) else (4 if j%2 else 2)
        for k in range(4):sums[k]+=w*x**k*f
    # For r in [.05,.60]: exp(V)<=25, |V'|<=8, |V''|<=5.
    # Product differentiation bounds (x^k exp(V))'''' as follows.
    fourth=[153000,368000,883000,2106000]
    moments=[s*h/3+iv.mpf([-1,1])*iv.mpf(2)/180*h**4*bound
             for s,bound in zip(sums,fourth)]
    z,m1,m2,m3=moments;end=iv.exp((iv.mpf('.8')-4*rate)/iv.mpf('.25'))
    numerator=iv.mpf('.2')*m1+iv.mpf('.125')*(10+end)
    derivative_numerator=iv.mpf('.8')*m3+2*end
    return numerator/z,-(derivative_numerator*z-numerator*4*m2)/(z*z)

def main():
    iv.dps=35;bounds=[]
    for j in range(80):
        rate=iv.mpf([200+5*j,205+5*j])/1000
        _,value=enclosure(rate)
        lo=float(np.nextafter(float(value.a),-np.inf));hi=float(np.nextafter(float(value.b),np.inf))
        assert lo>0,(j,lo,hi)
        bounds.append([lo,hi])
    baseline=[];upper_preservation=[];low_edge=[]
    for j in range(30):
        q=iv.mpf([450+5*j,455+5*j])/1000
        j0,_=enclosure(q);jlow,_=enclosure(q-iv.mpf('.01'))
        baseline.append(float(np.nextafter(float(j0.a),-np.inf)))
        low_edge.append(float(np.nextafter(float((j0-jlow-iv.mpf('.2')).b),np.inf)))
        r=iv.mpf([50+5*j,55+5*j])/1000
        jt,_=enclosure(r)
        upper_preservation.append(float(np.nextafter(float((jt+iv.mpf('.2')).b),np.inf)))
    high_region_lower=min(baseline)-max(upper_preservation)
    assert high_region_lower>.05 and max(low_edge)<-.05
    result={'passed':True,'positive_derivative_rate_range':[.20,.60],'covered_inventory_box':{'q':[.45,.60],'gamma':[.01,.40],'eta':1},
            'derivative_interval_cells':80,'additional_cost_interval_cells':90,'simpson_subintervals':400,'interval_decimal_digits':35,
            'advantage_preservation_derivative_enclosure':[min(b[0] for b in bounds),max(b[1] for b in bounds)],
            'advantage_lower_when_q_minus_gamma_below_point2':high_region_lower,
            'advantage_upper_at_gamma_point01':max(low_edge),
            'consequence':'Both promised decision boundaries are unique regular gamma thresholds. In the higher-preservation region the advantage is already above +0.05.',
            'fourth_derivative_bounds':[153000,368000,883000,2106000],
            'basis':'Explicit derivative of the constant-preservation boundary-inclusive cost; outward interval Simpson moments through degree three.',
            'cell_enclosures':bounds}
    path=inv.OUT/'decision_boundary_interval.json';path.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='cell_enclosures'},indent=2))

if __name__=='__main__':main()
