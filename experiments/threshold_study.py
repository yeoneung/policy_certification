"""Inventory-triggered binary control under the common certification protocol."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
os.environ.setdefault('NUMBA_NUM_THREADS','4')
import numpy as np
from numba import njit, prange
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import ndtr
import inventory_sequential as seq
from followup_v4 import mixture_test, FIELDS

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/outputs/threshold_v5'
PROTOCOL=ROOT/'experiments/threshold_protocol_20261026.json'
DIGEST=hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
METHODS=['ab','dopt','policy','threshold']

def jobs():
    out=[]
    def add(group,theta,budget,method,reps,dt=.02,horizon=400000.):
        out.append(dict(id=f'threshold_{len(out):03d}',group=group,theta=theta,budget=budget,
                        delta=.05,method=method,reps=reps,dt=dt,horizon=horizon,stream=len(out)))
    for theta in [[.5,.05],[.5,.3]]:
        for method in METHODS:add('primary',theta,.01,method,250,horizon=200000.)
    for theta in [[.5,.05],[.5,.3]]:
        for budget in [.003,.03]:
            for method in METHODS:add('budget',theta,budget,method,100)
    for theta in [[.5,.05],[.5,.3]]:
        for method in ['policy','threshold']:add('step',theta,.01,method,100,dt=.01)
    assert sum(j['reps'] for j in out)==4000
    return out

def segment_integral(rate,lo,hi):
    """Integral of exp((.4 x - rate x^2)/.25) on [lo,hi]."""
    scale=np.sqrt(.25/(2*rate));mean=.2/rate
    return np.exp(.04/(.25*rate))*np.sqrt(2*np.pi)*scale*(ndtr((hi-mean)/scale)-ndtr((lo-mean)/scale))

def threshold_cost(theta,s):
    q,gamma=theta
    low=segment_integral(q,0.,s)
    high=np.exp(-gamma*s*s/.25)*segment_integral(q-gamma,s,2.)
    return high/(low+high)

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    record={'protocol_sha256':DIGEST,'jobs':jobs()}
    path=OUT/'jobs.json'
    if path.exists():assert json.loads(path.read_text())==record
    else:path.write_text(json.dumps(record,indent=2),encoding='utf-8')
    for budget in [.003,.01,.03]:
        seq.prepare(budget)
        path=OUT/f'thresholds_B{budget:g}.npz'
        if path.exists():
            with np.load(path) as d:assert str(d['protocol_sha256'])==DIGEST
            continue
        with np.load(seq.OUT/f'designs_B{budget:g}.npz') as d:
            qgrid=d['qgrid'];ggrid=d['ggrid']
        cuts=np.array([[brentq(lambda s:threshold_cost((q,g),s)-.9*budget,0.,2.,xtol=1e-13)
                        for g in ggrid] for q in qgrid])
        np.savez_compressed(path,cuts=cuts,qgrid=qgrid,ggrid=ggrid,protocol_sha256=DIGEST)
    cov=seq.class_cover()
    np.savez_compressed(OUT/'covers.npz',plus=cov[1],minus=cov[-1],protocol_sha256=DIGEST)
    print(json.dumps({'prepared':True,'cells':len(jobs()),'scheduled_runs':4000}),flush=True)

@njit(cache=True)
def one_path(seed,theta,method,budget,delta,dt,horizon,tables,cuts,qgrid,ggrid,plus,minus):
    np.random.seed(seed)
    x=.5;clock=0.;energy=0.;lowtotal=0.;hightotal=0.;operation=0.
    g00=0.;g01=0.;g11=0.;s0=0.;s1=0.;k=0;answer=0
    guard=0;steps_total=0;excess=0.;cut=0.
    target=np.zeros(7);actual=np.zeros(7)
    mp=-np.inf;mm=-np.inf;pp=-np.inf;pm=-np.inf
    while clock<horizon and answer==0:
        k+=1;duration=min(1.+.01*k,horizon-clock)
        v00=g00+seq.PRECISION[0];v11=g11+seq.PRECISION[1];det=v00*v11-g01*g01
        eq=seq.REF[0]+(v11*s0-g01*s1)/det
        eg=seq.REF[1]+(v00*s1-g01*s0)/det
        iq=int(np.round((min(qgrid[-1],max(qgrid[0],eq))-qgrid[0])/(qgrid[-1]-qgrid[0])*(len(qgrid)-1)))
        ig=int(np.round((min(ggrid[-1],max(ggrid[0],eg))-ggrid[0])/(ggrid[-1]-ggrid[0])*(len(ggrid)-1)))
        cut=cuts[iq,ig];w=np.zeros(7)
        if method==4:w[:]=tables[0,iq,ig]
        elif method==0:w[1]=.9*budget;w[0]=1-w[1]
        elif method==3:w[6]=1.
        else:w[:6]=tables[method-1,iq,ig]
        forced=k**(-.25);w*=1-forced
        w[1]+=forced*.45*budget;w[0]+=forced*(1-.45*budget)
        target+=duration*w;choice=int(np.argmax(target-actual));actual[choice]+=duration
        steps=int(np.ceil(duration/dt));h=duration/steps;root=np.sqrt(h)
        for j in range(steps):
            if choice==0:effort=0.
            elif choice==1:effort=1.
            elif choice==6:effort=1. if x>=cut else 0.
            else:
                z=min(1.,max(0.,(x-(2.-seq.WIDTHS[choice]))/seq.WIDTHS[choice]));effort=z*z*(3.-2.*z)
            credit=max(0.,budget*(clock+(j+1)*h)-energy);allowed=np.sqrt(credit/h)
            if effort>allowed:effort=allowed;guard+=1
            f0=-x/.5;f1=effort*x/.5;noise=np.random.normal()*root
            dy=(f0*(theta[0]-seq.REF[0])+f1*(theta[1]-seq.REF[1]))*h+noise
            raw=x+(.2-theta[0]*x+theta[1]*effort*x)*h+.5*noise
            nx=min(2.,max(0.,raw));correction=nx-raw
            lo=max(0.,correction);hi=max(0.,-correction)
            lowtotal+=lo;hightotal+=hi;operation+=(.2*x+.2*effort*effort)*h+10*lo+hi
            g00+=f0*f0*h;g01+=f0*f1*h;g11+=f1*f1*h;s0+=f0*dy;s1+=f1*dy
            energy+=effort*effort*h;steps_total+=1;x=nx
            excess=max(excess,energy-budget*(clock+(j+1)*h))
        clock+=duration;pp=mp;pm=mm
        answer,mp,mm=mixture_test(g00,g01,g11,s0,s1,seq.REF,seq.PRECISION,np.log(1/delta),plus,minus)
    return np.array([clock,energy,answer,lowtotal,hightotal,operation,k,guard,steps_total,excess,
                     g00,g01,g11,s0,s1,x,mp,mm,pp,pm,cut])

@njit(cache=True,parallel=True)
def paths(seeds,theta,method,budget,delta,dt,horizon,tables,cuts,qgrid,ggrid,plus,minus):
    out=np.zeros((len(seeds),21))
    for i in prange(len(seeds)):out[i]=one_path(seeds[i],theta,method,budget,delta,dt,horizon,tables,cuts,qgrid,ggrid,plus,minus)
    return out

def execute(job,smoke=False):
    target=OUT/(('smoke_' if smoke else '')+job['id']+'.json')
    if target.exists() and not smoke:
        d=json.loads(target.read_text());assert d['job']==job and d['protocol_sha256']==DIGEST
        return {'existing':job['id']}
    theta=np.array(job['theta']);cov=np.load(OUT/'covers.npz')
    with np.load(seq.OUT/f"designs_B{job['budget']:g}.npz") as d:tables=d['tables'];qgrid=d['qgrid'];ggrid=d['ggrid']
    with np.load(OUT/f"thresholds_B{job['budget']:g}.npz") as d:cuts=d['cuts']
    n=2 if smoke else job['reps']
    seeds=np.random.SeedSequence([2026102605,job['stream'],int(smoke)]).generate_state(n,dtype=np.uint32)
    start=time.perf_counter()
    arr=paths(seeds,theta,METHODS.index(job['method']),job['budget'],job['delta'],job['dt'],
              min(job['horizon'],100.) if smoke else job['horizon'],tables,cuts,qgrid,ggrid,cov['plus'],cov['minus'])
    advantage=seq.delta(theta);truth=1 if advantage>0 else -1;runs=[]
    for seed,row in zip(seeds,arr):
        r={key:(float(v) if np.isfinite(v) else None) for key,v in zip(FIELDS+['final_threshold'],row)}
        r.update(seed=int(seed),stopped=bool(r['answer']),error=bool(r['answer'] and r['answer']!=truth));runs.append(r)
    result={'job':job,'protocol_sha256':DIGEST,'advantage':advantage,'smoke':smoke,
            'wall_seconds':time.perf_counter()-start,'runs':runs}
    target.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    summary={'cell':job['id'],'method':job['method'],'mean':float(arr[:,0].mean()),
             'censored':sum(not r['stopped'] for r in runs),'errors':sum(r['error'] for r in runs),
             'seconds':result['wall_seconds']}
    print(json.dumps(summary),flush=True);return summary

def preflight():
    errors=[];residuals=[]
    for budget in [.003,.01,.03]:
        with np.load(OUT/f'thresholds_B{budget:g}.npz') as d:
            for i,q in enumerate(d['qgrid']):
                for j,g in enumerate(d['ggrid']):residuals.append(abs(threshold_cost((q,g),d['cuts'][i,j])-.9*budget))
    for q in [.45,.5,.6]:
        for g in [.01,.05,.3,.4]:
            for s in [.2,.9,1.5,1.95]:
                f=lambda x:np.exp((.4*x-q*x*x+g*max(0.,x*x-s*s))/.25)
                a=quad(f,0,s,epsabs=1e-13)[0];b=quad(f,s,2,epsabs=1e-13)[0]
                errors.append(abs(threshold_cost((q,g),s)-b/(a+b)))
    assert max(errors)<1e-12 and max(residuals)<1e-12
    seeds=[int(s) for j in jobs() for s in np.random.SeedSequence([2026102605,j['stream'],0]).generate_state(j['reps'],dtype=np.uint32)]
    assert len(set(seeds))==4000
    old=set()
    for folder in [seq.OUT,ROOT/'experiments/outputs/followup_v4']:
        for p in folder.glob('*.json'):
            d=json.loads(p.read_text())
            if isinstance(d,dict):old.update(r['seed'] for r in d.get('runs',[]) if 'seed' in r)
    assert not old.intersection(seeds)
    replay_differences=[]
    d=np.load(seq.OUT/'designs_B0.01.npz');cuts=np.load(OUT/'thresholds_B0.01.npz')['cuts'];cov=np.load(OUT/'covers.npz')
    for theta in [np.array([.5,.05]),np.array([.5,.3])]:
        for method in [0,1,2]:
            for seed in [81001,81002]:
                a=seq.one_path(seed,theta,method,.01,.02,2000.,.05,d['tables'],d['qgrid'],d['ggrid'],
                               np.zeros(6),cov['plus'],cov['minus'],.01,True)
                b=one_path(seed,theta,method,.01,.05,.02,2000.,d['tables'],cuts,d['qgrid'],d['ggrid'],cov['plus'],cov['minus'])
                assert np.allclose(a[:7],b[:7],rtol=1e-11,atol=1e-9),(theta,method,a[:7],b[:7])
                assert a[10]==b[7] and a[11]==b[8]
                replay_differences.append(float(np.max(np.abs(a[:7]-b[:7]))))
    record={'passed':True,'thresholds_checked':len(residuals),'stationary_cost_max_residual':max(residuals),
            'quadrature_crosschecks':len(errors),'stationary_formula_max_difference':max(errors),
            'unique_new_seeds':len(set(seeds)),'overlap_with_recorded_old_seeds':0,
            'existing_comparator_replays':len(replay_differences),'maximum_replay_summary_difference':max(replay_differences)}
    (OUT/'preflight.json').write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','preflight','smoke','run','cell'])
    p.add_argument('--index',type=int,default=0);p.add_argument('--workers',type=int,default=2)
    args=p.parse_args()
    if args.command=='prepare':prepare()
    elif args.command=='preflight':preflight()
    elif args.command=='smoke':
        for j in jobs()[:4]:execute(j,True)
    elif args.command=='cell':execute(jobs()[args.index])
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(subprocess.run,[sys.executable,str(Path(__file__)), 'cell','--index',str(i)],check=True)
                     for i in range(len(jobs()))]
            for f in as_completed(futures):f.result()

if __name__=='__main__':main()
