"""Compute CSV and JSON summaries of generated experiment runs."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy.stats import beta,ttest_ind


def summarize(directory):
    records=[];data={}
    for path in sorted(directory.glob('*.json')):
        d=json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(d,dict) or 'runs' not in d:continue
        if d.get('smoke') or path.name.startswith(('pilot','smoke','merge_check')):continue
        runs=d['runs']
        if not runs:continue
        config=d.get('job',d)
        times=np.array([r['time'] for r in runs],dtype=float)
        energy=np.array([r['energy'] for r in runs],dtype=float)
        n=len(runs);errors=sum(bool(r.get('error')) for r in runs)
        row={k:config.get(k) for k in ['model','group','method','theta','budget','delta','dt','width','initial','horizon','max_time']}
        row.update(file=path.name,repetitions=n,censored=sum(not r['stopped'] for r in runs),errors=errors,
                   restricted_mean_time=float(times.mean()),time_mcse=float(times.std(ddof=1)/np.sqrt(n)) if n>1 else None,
                   mean_energy=float(energy.mean()),energy_per_time=float(energy.sum()/times.sum()),
                   error_before_horizon_upper95=float(beta.ppf(.95,errors+1,n-errors)) if errors<n else 1.)
        records.append(row);data[path.name]=times
    if not records:raise ValueError('No completed experiment records found')
    comparisons=[];rng=np.random.default_rng(2026102506)
    for a in [r for r in records if r['method']=='dopt']:
        matches=[r for r in records if r['method']=='policy' and
                 all(r[k]==a[k] for k in ['model','group','theta','budget','delta','dt'])]
        if len(matches)!=1:continue
        b=matches[0];x=data[a['file']];y=data[b['file']]
        xm=x[rng.integers(0,len(x),(20000,len(x)))].mean(axis=1)
        ym=y[rng.integers(0,len(y),(20000,len(y)))].mean(axis=1)
        comparisons.append({'dopt_file':a['file'],'policy_file':b['file'],
                            'model':a['model'],'group':a['group'],'theta':a['theta'],
                            'budget':a['budget'],'delta':a['delta'],'dt':a['dt'],
                            'dopt_over_policy_time':float(x.mean()/y.mean()),
                            'pointwise_ratio_bootstrap95':np.quantile(xm/ym,[.025,.975]).tolist(),
                            'welch_two_sided_p':float(ttest_ind(x,y,equal_var=False).pvalue),
                            'holm_target_p':None})
    family=[r for r in comparisons if r['model']=='bilinear' and r['group']=='primary' and r['theta']==[-.3,.2]]
    ordered=sorted(family,key=lambda r:r['welch_two_sided_p']);running=0.
    for i,r in enumerate(ordered):
        running=max(running,min(1.,(len(family)-i)*r['welch_two_sided_p']))
        r['holm_target_p']=running
    destination=directory/'analysis';destination.mkdir(exist_ok=True)
    with (destination/'summary.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    (destination/'summary.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    (destination/'comparisons.json').write_text(json.dumps(comparisons,indent=2),encoding='utf-8')
    print(json.dumps({'cells':len(records),'runs':sum(r['repetitions'] for r in records),
                      'censored':sum(r['censored'] for r in records),'errors':sum(r['errors'] for r in records),
                      'destination':str(destination)},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True,type=Path)
    summarize(p.parse_args().input.resolve())
