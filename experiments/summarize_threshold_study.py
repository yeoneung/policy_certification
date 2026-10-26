"""Complete summaries and planned comparisons for inventory-triggered controls."""
import csv
import json
from pathlib import Path
import numpy as np
from scipy.stats import beta,ttest_ind
import mpmath as mp
import threshold_study as study
import threshold_enlarged as enlarged
from validate_followup_v4 import log_min

def main():
    summaries=[];raw={};cov=np.load(study.OUT/'covers.npz');checked=0
    scheduled=[(job,study.DIGEST) for job in study.jobs()]
    for i in range(2):
        data=json.loads((study.OUT/f'enlarged_{i:03d}.json').read_text())
        scheduled.append((data['job'],enlarged.DIGEST))
    allseeds=[]
    for job,digest in scheduled:
        data=json.loads((study.OUT/(job['id']+'.json')).read_text())
        assert data['job']==job and data['protocol_sha256']==digest and not data['smoke']
        rr=data['runs'];assert len(rr)==job['reps'];truth=1 if data['advantage']>0 else -1
        for r in rr:
            allseeds.append(r['seed'])
            assert r['energy']<=job['budget']*r['time']+1e-7 and r['max_budget_excess']<1e-7
            assert r['stopped'] or abs(r['time']-job['horizon'])<1e-6
            assert r['error']==bool(r['answer'] and r['answer']!=truth)
            assert r['previous_plus_margin'] is None or r['previous_plus_margin']<=0
            assert r['previous_minus_margin'] is None or r['previous_minus_margin']<=0
            if r['stopped']:assert r['plus_margin' if r['answer']==1 else 'minus_margin']>0
        for i,r in enumerate(rr):
            if r['stopped'] and (i==0 or r['error']):
                boxes=cov['minus' if r['answer']==1 else 'plus']
                margin=log_min(r,boxes,study.seq.REF,study.seq.PRECISION)-mp.log(mp.mpf(1)/mp.mpf(job['delta']))
                assert margin>0;checked+=1
        times=np.array([r['time'] for r in rr]);errors=sum(r['error'] for r in rr)
        s=dict(job,advantage=data['advantage'],rmst=float(times.mean()),mcse=float(times.std(ddof=1)/np.sqrt(len(times))),
               censored=sum(not r['stopped'] for r in rr),errors=errors,
               error_before_horizon_upper95=float(beta.ppf(.95,errors+1,len(rr)-errors)),
               energy_per_time=sum(r['energy'] for r in rr)/times.sum(),
               guard_fraction=sum(r['guard_steps'] for r in rr)/sum(r['steps'] for r in rr),
               operating_cost_per_time=sum(r['operating_cost'] for r in rr)/times.sum(),
               mean_energy=float(np.mean([r['energy'] for r in rr])),max_budget_excess=max(r['max_budget_excess'] for r in rr))
        summaries.append(s);raw[job['id']]=times
    assert len(allseeds)==4500 and len(set(allseeds))==4500
    rng=np.random.default_rng(2026102606);comparisons=[]
    for t in [r for r in summaries if r['method']=='threshold']:
        for c in [r for r in summaries if r['method']!='threshold' and
                  all(r[k]==t[k] for k in ['group','theta','budget','delta','dt'])]:
            x=raw[c['id']];y=raw[t['id']]
            bx=x[rng.integers(len(x),size=(20000,len(x)))].mean(axis=1)
            by=y[rng.integers(len(y),size=(20000,len(y)))].mean(axis=1)
            comparisons.append(dict(group=t['group'],theta=t['theta'],budget=t['budget'],dt=t['dt'],
                                    comparator=c['method'],comparator_over_threshold=c['rmst']/t['rmst'],
                                    reduction_percent=100*(1-t['rmst']/c['rmst']),
                                    bootstrap95=np.quantile(bx/by,[.025,.975]).tolist(),
                                    welch_p=float(ttest_ind(x,y,equal_var=False).pvalue),holm_primary_p=None))
    for c in [r for r in summaries if r['method']=='dopt_enlarged']:
        t=next(r for r in summaries if r['group']=='primary' and r['theta']==c['theta'] and r['method']=='threshold')
        x=raw[c['id']];y=raw[t['id']]
        bx=x[rng.integers(len(x),size=(20000,len(x)))].mean(axis=1)
        by=y[rng.integers(len(y),size=(20000,len(y)))].mean(axis=1)
        comparisons.append(dict(group='class_ablation',theta=c['theta'],budget=.01,dt=.02,
                                comparator='dopt_enlarged',comparator_over_threshold=c['rmst']/t['rmst'],
                                reduction_percent=100*(1-t['rmst']/c['rmst']),
                                bootstrap95=np.quantile(bx/by,[.025,.975]).tolist(),
                                welch_p=float(ttest_ind(x,y,equal_var=False).pvalue),holm_primary_p=None))
    family=sorted([r for r in comparisons if r['group']=='primary'],key=lambda r:r['welch_p'])
    assert len(family)==6;running=0.
    for i,r in enumerate(family):
        running=max(running,min(1.,(6-i)*r['welch_p']));r['holm_primary_p']=running
    (study.OUT/'study_summary.json').write_text(json.dumps(summaries,indent=2))
    with (study.OUT/'study_summary.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(summaries[0]));writer.writeheader();writer.writerows(summaries)
    counts={'runs':sum(s['reps'] for s in summaries),'censored':sum(s['censored'] for s in summaries),'errors':sum(s['errors'] for s in summaries)}
    record={'protocol_sha256':study.DIGEST,'counts':counts,'comparisons':comparisons}
    (study.OUT/'comparisons.json').write_text(json.dumps(record,indent=2))
    validation=dict(passed=True,scheduled_runs=counts['runs'],censored=counts['censored'],errors=counts['errors'],
                    high_precision_terminal_exclusions=checked,maximum_budget_excess=max(s['max_budget_excess'] for s in summaries),
                    all_scheduled_cells_included=True)
    (study.OUT/'validation.json').write_text(json.dumps(validation,indent=2))
    print(json.dumps({'validation':validation,'primary_comparisons':family},indent=2))

if __name__=='__main__':main()
