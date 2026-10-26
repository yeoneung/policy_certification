"""Run the inventory experiment configurations."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys
from scipy.optimize import brentq
import inventory_frontier as inv
ROOT = Path(__file__).resolve().parent
OUT = inv.OUT / 'sequential'
RERUN = False

def jobs(group):
    result = []
    if group == 'primary':
        for scenario, gamma in [('good', 0.3), ('bad', 0.05)]:
            for method in ['ab', 'dopt', 'policy', 'oracle']:
                result.append(dict(name=f'primary_{scenario}_{method}', method=method, theta=[0.5, gamma], reps=250, dt=0.02, budget=0.01, delta=0.05, max_time=200000, seed=20310001 + (scenario == 'bad')))
    else:
        for scenario, gamma in [('good', 0.3), ('bad', 0.05)]:
            base = dict(theta=[0.5, gamma], reps=100, dt=0.02, budget=0.01, delta=0.05, max_time=400000)
            settings = [('dt001', dict(dt=0.01)), ('delta001', dict(delta=0.01)), ('B003', dict(budget=0.003)), ('B03', dict(budget=0.03))]
            for setting, changes in settings:
                for method in ['ab', 'policy']:
                    result.append(dict(base, **{}))
                    result[-1].update(changes)
                    result[-1].update(name=f'sensitivity_{setting}_{scenario}_{method}', method=method, seed=20320001 + len(result))
        for sign, target in [('negative', -0.051), ('positive', 0.051)]:
            gamma = brentq(lambda g: inv.constant_cost(0.5, g, 0) - inv.constant_cost(0.5, g, 1) - target, 0.01, 0.3)
            for method in ['ab', 'policy']:
                result.append(dict(name=f'boundary_{sign}_{method}', method=method, theta=[0.5, gamma], reps=100, dt=0.02, budget=0.01, delta=0.05, max_time=400000, seed=20330001 + len(result)))
        for q in [0.46, 0.58]:
            for gamma in [0.05, 0.3]:
                for method in ['ab', 'policy']:
                    result.append(dict(name=f'robust_q{q:g}_g{gamma:g}_{method}', method=method, theta=[q, gamma], reps=100, dt=0.02, budget=0.01, delta=0.05, max_time=400000, seed=20340001 + len(result)))
    return result

def execute(job):
    target = OUT / (job['name'] + '.json')
    if target.exists() and (not RERUN):
        data = json.loads(target.read_text())
        for field in ['method', 'theta', 'reps', 'dt', 'budget', 'delta', 'max_time', 'seed']:
            assert data[field] == job[field], (target, field)
        return {'skipped_existing': job['name']}
    command = [sys.executable, str(ROOT / 'inventory_sequential.py'), 'simulate']
    for key in ['method', 'reps', 'dt', 'budget', 'delta', 'seed']:
        command.extend(['--' + key, str(job[key])])
    command.extend(['--max-time', str(job['max_time']), '--theta', *map(str, job['theta']), '--output', target.name])
    proc = subprocess.run(command, text=True, capture_output=True)
    if proc.returncode:
        raise RuntimeError(job['name'] + '\n' + proc.stdout + proc.stderr)
    return json.loads(proc.stdout.strip().splitlines()[-1])
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--group', choices=['primary', 'sensitivity'], default='primary')
    p.add_argument('--workers', type=int, default=3)
    p.add_argument('--rerun', action='store_true', help='Regenerate scheduled trajectories, retaining pilots')
    args = p.parse_args()
    selected = jobs(args.group)
    RERUN = args.rerun
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f'jobs_{args.group}.json').write_text(json.dumps(selected, indent=2), encoding='utf-8')
    for budget in sorted({j['budget'] for j in selected}):
        subprocess.run([sys.executable, str(ROOT / 'inventory_sequential.py'), 'prepare', '--budget', str(budget)], check=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(execute, j) for j in selected]
        for future in as_completed(futures):
            print(json.dumps(future.result()), flush=True)
