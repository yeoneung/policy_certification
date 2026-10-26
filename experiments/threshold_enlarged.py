"""D-optimal design with a budget-calibrated threshold in its feedback library."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import quad
from scipy.optimize import minimize,linprog
import threshold_study as study

PROTOCOL=study.ROOT/'experiments/threshold_enlarged_protocol_20261026.json'
DIGEST=hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()

def moments(theta,cut,n=96):
    q,gamma=theta;nodes,weights=leggauss(n);xx=[];ww=[]
    for lo,hi in [(0.,cut),(cut,2.)]:
        xx.extend((lo+hi)/2+(hi-lo)*nodes/2);ww.extend((hi-lo)*weights/2)
    x=np.array(xx);weights=np.array(ww)
    p=np.exp((.4*x-q*x*x+gamma*np.maximum(0.,x*x-cut*cut))/.25)
    weights*=p;weights/=weights.sum();effort=(x>=cut).astype(float)
    f=np.column_stack([-x,effort*x])/.5
    return np.einsum('n,ni,nj->ij',weights,f,f),float(weights@effort)

def prepare():
    with np.load(study.OUT/'thresholds_B0.01.npz') as d:cuts=d['cuts'];qgrid=d['qgrid'];ggrid=d['ggrid']
    with np.load(study.seq.OUT/'designs_B0.01.npz') as d:old=d['tables'][0]
    tables=np.zeros((1,len(qgrid),len(ggrid),7));gaps=[];cost_errors=[]
    for i,q in enumerate(qgrid):
        for j,g in enumerate(ggrid):
            mats,cost=study.seq.moments(np.array([q,g]));tm,tc=moments((q,g),cuts[i,j])
            mats=np.concatenate([mats,tm[None]]);cost=np.r_[cost,tc];cap=.009
            cost_errors.append(abs(tc-cap))
            def objective(w):
                m=np.einsum('j,jab->ab',w,mats)
                return -np.linalg.slogdet(m)[1],-np.einsum('ab,jba->j',np.linalg.inv(m),mats)
            start=np.r_[old[i,j],0.]
            res=minimize(objective,start,jac=True,method='SLSQP',bounds=[(0,1)]*7,
                         constraints=[{'type':'eq','fun':lambda w:w.sum()-1,'jac':lambda w:np.ones(7)},
                                      {'type':'ineq','fun':lambda w:1-cost@w/cap,'jac':lambda w:-cost/cap}],
                         options={'ftol':1e-11,'maxiter':500})
            assert res.success,res.message
            w=np.maximum(0.,res.x);w/=w.sum()
            if cost@w>cap:
                w*=cap/(cost@w);w[0]+=1-w.sum()
            tables[0,i,j]=w
            gradient=-objective(w)[1]
            ref=linprog(-gradient,A_ub=cost[None],b_ub=[cap],A_eq=np.ones((1,7)),b_eq=[1.],bounds=(0,None),method='highs')
            assert ref.success;gaps.append(float(-ref.fun-gradient@w))
    assert max(gaps)<1e-4 and max(cost_errors)<1e-12
    np.savez_compressed(study.OUT/'enlarged_designs.npz',tables=tables,protocol_sha256=DIGEST)
    for theta in [(.5,.05),(.5,.3)]:
        for cut in [.8,1.4,1.9]:
            m,c=moments(theta,cut)
            f=lambda x:np.exp((.4*x-theta[0]*x*x+theta[1]*max(0.,x*x-cut*cut))/.25)
            den=quad(f,0,cut,epsabs=1e-13)[0]+quad(f,cut,2,epsabs=1e-13)[0]
            whole=(quad(lambda x:x*x*f(x),0,cut,epsabs=1e-13)[0]+quad(lambda x:x*x*f(x),cut,2,epsabs=1e-13)[0])/den/.25
            high=quad(lambda x:x*x*f(x),cut,2,epsabs=1e-13)[0]/den/.25
            assert np.allclose(m,[[whole,-high],[-high,high]],atol=1e-12)
    record={'passed':True,'designs':len(gaps),'max_logdet_first_order_gap':max(gaps),'max_threshold_cost_error':max(cost_errors)}
    (study.OUT/'enlarged_preflight.json').write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)

def run():
    with np.load(study.OUT/'thresholds_B0.01.npz') as d:cuts=d['cuts'];qgrid=d['qgrid'];ggrid=d['ggrid']
    with np.load(study.OUT/'enlarged_designs.npz') as d:tables=d['tables'];assert str(d['protocol_sha256'])==DIGEST
    cov=np.load(study.OUT/'covers.npz')
    for i,theta in enumerate([[.5,.05],[.5,.3]]):
        job=dict(id=f'enlarged_{i:03d}',group='class_ablation',theta=theta,budget=.01,delta=.05,
                 method='dopt_enlarged',reps=250,dt=.02,horizon=200000.,stream=i)
        path=study.OUT/(job['id']+'.json')
        if path.exists():
            d=json.loads(path.read_text());assert d['job']==job and d['protocol_sha256']==DIGEST;continue
        seeds=np.random.SeedSequence([2026102607,i]).generate_state(250,dtype=np.uint32)
        start=time.perf_counter()
        arr=study.paths(seeds,np.array(theta),4,.01,.05,.02,200000.,tables,cuts,qgrid,ggrid,cov['plus'],cov['minus'])
        advantage=study.seq.delta(theta);truth=1 if advantage>0 else -1;rr=[]
        for seed,row in zip(seeds,arr):
            r={k:(float(v) if np.isfinite(v) else None) for k,v in zip(study.FIELDS+['final_threshold'],row)}
            r.update(seed=int(seed),stopped=bool(r['answer']),error=bool(r['answer'] and r['answer']!=truth));rr.append(r)
        d=dict(job=job,protocol_sha256=DIGEST,advantage=advantage,smoke=False,wall_seconds=time.perf_counter()-start,runs=rr)
        path.write_text(json.dumps(d,indent=2,allow_nan=False));print(json.dumps({'method':'dopt_enlarged','theta':theta,'mean':float(arr[:,0].mean()),'seconds':d['wall_seconds']}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','run']);a=p.parse_args()
    (prepare if a.command=='prepare' else run)()
