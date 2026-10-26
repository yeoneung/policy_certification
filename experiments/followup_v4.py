"""Run bilinear comparator and inventory localization experiments."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
os.environ.setdefault('NUMBA_NUM_THREADS', '4')
import numpy as np
from numba import njit, prange
from scipy.optimize import minimize
import policy_certification as pc
import inventory_sequential as inv
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/outputs/followup_v4'
PROTOCOL = ROOT / 'experiments/followup_protocol_20261025.json'
DIGEST = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
WIDTHS = np.array([0.0, 1.0, 0.5, 0.25, 0.1])
FIELDS = ['time', 'energy', 'answer', 'regulator_low', 'regulator_high', 'operating_cost', 'blocks', 'guard_steps', 'steps', 'max_budget_excess', 'g00', 'g01', 'g11', 's0', 's1', 'terminal_state', 'plus_margin', 'minus_margin', 'previous_plus_margin', 'previous_minus_margin']

def jobs():
    result = []

    def add(model, group, theta, budget, delta, method, reps, dt=0.02, width=0.0, initial=0.5):
        result.append(dict(id=f'{model}_{len(result):03d}', model=model, group=group, theta=theta, budget=budget, delta=delta, method=method, reps=reps, dt=dt, width=width, initial=initial, horizon=20000.0 if model == 'bilinear' else 400000.0, stream=len(result)))
    for theta in [[-0.3, 0.2], [1.0, 0.2]]:
        for budget in [0.5, 1.0]:
            for delta in [0.05, 0.001]:
                for method in ['dopt', 'policy']:
                    add('bilinear', 'primary', theta, budget, delta, method, 300, initial=0.0)
    for method in ['dopt', 'policy']:
        add('bilinear', 'dt_sensitivity', [-0.3, 0.2], 1.0, 0.001, method, 150, dt=0.01, initial=0.0)
    for theta in [[-0.25, 0.2], [-0.35, 0.2]]:
        for method in ['dopt', 'policy']:
            add('bilinear', 'neighbor', theta, 1.0, 0.001, method, 150, initial=0.0)
    for theta in [[0.5, 0.05], [0.5, 0.3]]:
        for budget in [0.003, 0.01, 0.03]:
            for delta in [0.05, 0.01]:
                for width in WIDTHS.tolist():
                    add('inventory', 'factorial', theta, budget, delta, 'budget_fill', 80, width=width)
    for theta in [[0.5, 0.05], [0.5, 0.3]]:
        for initial in [0.1, 1.9]:
            for width in [0.0, 0.5, 0.1]:
                add('inventory', 'initial_state', theta, 0.01, 0.05, 'budget_fill', 60, width=width, initial=initial)
    assert sum((j['reps'] for j in result)) == 11220
    return result

def save_manifest():
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'jobs.json'
    record = {'protocol_sha256': DIGEST, 'jobs': jobs()}
    if path.exists():
        assert json.loads(path.read_text()) == record
    else:
        path.write_text(json.dumps(record, indent=2), encoding='utf-8')
    return record['jobs']

def merge_boxes(boxes):

    def axis_merge(rows, axis):
        lo, hi = (0, 1) if axis == 0 else (2, 3)
        fixed = (2, 3) if axis == 0 else (0, 1)
        rows = sorted(rows, key=lambda b: (b[fixed[0]], b[fixed[1]], b[lo]))
        out = []
        for b in rows:
            if out and all((out[-1][i] == b[i] for i in fixed)) and (b[lo] <= out[-1][hi]):
                out[-1][hi] = max(out[-1][hi], b[hi])
            else:
                out.append(b.copy())
        return out
    previous = -1
    while previous != len(boxes):
        previous = len(boxes)
        boxes = axis_merge(axis_merge(boxes, 0), 1)
    return np.asarray(boxes)

def bilinear_cover():
    mesh = pc.ParameterMesh(0.005)
    n = np.ceil((pc.HIGH - pc.LOW) / 0.005).astype(int)
    result = {}
    for sign in [1, -1]:
        mask = mesh.possible[sign].reshape(tuple(n))
        boxes = []
        for i in range(n[0]):
            edges = np.diff(np.r_[False, mask[i], False].astype(int))
            for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
                boxes.append([pc.LOW[0] + i * mesh.width[0], pc.LOW[0] + (i + 1) * mesh.width[0], pc.LOW[1] + a * mesh.width[1], pc.LOW[1] + b * mesh.width[1]])
        result[sign] = merge_boxes(boxes)
    return result

def feasible_weights(w, cost, budget, baseline):
    w = np.maximum(w, 0.0)
    w /= w.sum()
    if cost @ w > budget:
        factor = budget / (cost @ w)
        w *= factor
        w[baseline] += 1.0 - factor
    return w

def prepare_bilinear(budget):
    path = OUT / f'bilinear_designs_B{budget:g}.npz'
    if path.exists():
        with np.load(path) as d:
            assert str(d['protocol_sha256']) == DIGEST
        return
    mesh = pc.ParameterMesh(0.01)
    qgrid = np.linspace(pc.LOW[0], pc.HIGH[0], 31)
    ggrid = np.linspace(pc.LOW[1], pc.HIGH[1], 25)
    tables = np.zeros((2, 31, 25, 5))
    cap = 0.9 * budget
    cost = pc.ACTIONS ** 2
    for i, q in enumerate(qgrid):
        for j, g in enumerate(ggrid):
            theta = np.array([q, g])
            mats = pc.information(theta)

            def objective(w):
                matrix = np.einsum('j,jab->ab', w, mats)
                return (-np.linalg.slogdet(matrix)[1], -np.einsum('ab,jba->j', np.linalg.inv(matrix), mats))
            start = pc.solve_design(theta, cap, mesh, 'balanced')['weights']
            res = minimize(objective, start, jac=True, method='SLSQP', bounds=[(0, 1)] * 5, constraints=[{'type': 'eq', 'fun': lambda w: w.sum() - 1, 'jac': lambda w: np.ones(5)}, {'type': 'ineq', 'fun': lambda w: 1 - cost @ w / cap, 'jac': lambda w: -cost / cap}], options={'ftol': 1e-10, 'maxiter': 500})
            if not res.success:
                raise RuntimeError((theta, res.message))
            tables[0, i, j] = feasible_weights(res.x, cost, cap, 2)
            w = pc.solve_design(theta, cap, mesh, 'policy', side='upper')['weights']
            tables[1, i, j] = feasible_weights(w, cost, cap, 2)
        if i % 10 == 0:
            print(json.dumps({'prepare_bilinear': budget, 'row': i}), flush=True)
    np.savez_compressed(path, tables=tables, qgrid=qgrid, ggrid=ggrid, protocol_sha256=DIGEST)

def prepare_inventory():
    path = OUT / 'inventory_costs.npz'
    if path.exists():
        with np.load(path) as d:
            assert str(d['protocol_sha256']) == DIGEST
        return
    qgrid = np.linspace(0.45, 0.6, 16)
    ggrid = np.linspace(0.01, 0.4, 27)
    costs = np.zeros((5, 16, 27))
    for wi, width in enumerate(WIDTHS):
        kind = 'candidate' if width == 0 else f'gate_{width}'
        for i, q in enumerate(qgrid):
            for j, g in enumerate(ggrid):
                costs[wi, i, j] = inv.inv.policy([1.0, q, g], kind, 2001)['spend']
    np.savez_compressed(path, costs=costs, qgrid=qgrid, ggrid=ggrid, protocol_sha256=DIGEST)

def prepare():
    save_manifest()
    cover_path = OUT / 'covers.npz'
    if not cover_path.exists():
        b = bilinear_cover()
        i = inv.class_cover()
        np.savez_compressed(cover_path, bplus=b[1], bminus=b[-1], iplus=i[1], iminus=i[-1], protocol_sha256=DIGEST)
    for budget in [0.5, 1.0]:
        prepare_bilinear(budget)
    prepare_inventory()

@njit(cache=True)
def mixture_test(g00, g01, g11, s0, s1, ref, precision, logdelta, plus, minus):
    det = g00 * g11 - g01 * g01
    if det <= 1e-10 or g00 <= 0 or g11 <= 0:
        return (0, -np.inf, -np.inf)
    b0 = (g11 * s0 - g01 * s1) / det
    b1 = (g00 * s1 - g01 * s0) / det
    v00 = g00 + precision[0]
    v11 = g11 + precision[1]
    vdet = v00 * v11 - g01 * g01
    score_g = s0 * b0 + s1 * b1
    score_v = (v11 * s0 * s0 - 2 * g01 * s0 * s1 + v00 * s1 * s1) / vdet
    r2 = 2 * logdelta + np.log(vdet / (precision[0] * precision[1])) + score_g - score_v
    c0 = ref[0] + b0
    c1 = ref[1] + b1
    positive = inv.rectangle_distance(g00, g01, g11, c0, c1, minus) - r2
    negative = inv.rectangle_distance(g00, g01, g11, c0, c1, plus) - r2
    if positive > 0:
        return (1, positive, negative)
    if negative > 0:
        return (-1, positive, negative)
    return (0, positive, negative)

@njit(cache=True)
def one_path(seed, model, theta, budget, delta, dt, horizon, initial, width, method, tables, costs, qgrid, ggrid, plus, minus):
    np.random.seed(seed)
    if model == 0:
        ref = np.array([0.375, 0.0])
        precision = np.array([1 / 1.125 ** 2, 1 / 0.6 ** 2])
        lower = -2.0
        upper = 2.0
        sigma = 0.7
        m = 5
    else:
        ref = np.array([0.525, 0.205])
        precision = np.array([1 / 0.075 ** 2, 1 / 0.195 ** 2])
        lower = 0.0
        upper = 2.0
        sigma = 0.5
        m = 2
    clock = 0.0
    energy = 0.0
    x = initial
    reg0 = 0.0
    regc = 0.0
    operation = 0.0
    g00 = 0.0
    g01 = 0.0
    g11 = 0.0
    s0 = 0.0
    s1 = 0.0
    k = 0
    answer = 0
    guard = 0
    steps_total = 0
    excess = 0.0
    target = np.zeros(m)
    actual = np.zeros(m)
    margin_plus = -np.inf
    margin_minus = -np.inf
    previous_plus = -np.inf
    previous_minus = -np.inf
    maxcost = costs.max()
    while clock < horizon and answer == 0:
        k += 1
        duration = min(0.5 * k if model == 0 else 1.0 + 0.01 * k, horizon - clock)
        v00 = g00 + precision[0]
        v11 = g11 + precision[1]
        det = v00 * v11 - g01 * g01
        estimate0 = ref[0] + (v11 * s0 - g01 * s1) / det
        estimate1 = ref[1] + (v00 * s1 - g01 * s0) / det
        iq = int(np.round((min(qgrid[-1], max(qgrid[0], estimate0)) - qgrid[0]) / (qgrid[-1] - qgrid[0]) * (len(qgrid) - 1)))
        ig = int(np.round((min(ggrid[-1], max(ggrid[0], estimate1)) - ggrid[0]) / (ggrid[-1] - ggrid[0]) * (len(ggrid) - 1)))
        forced = k ** (-0.25)
        if model == 0:
            w = tables[method, iq, ig].copy() * (1 - forced)
            w[0] += forced * budget / 4.0
            w[4] += forced * budget / 4.0
            w[2] += forced * (1 - budget / 2.0)
        else:
            nominal = min(1.0, 0.9 * budget / costs[iq, ig])
            safe = min(0.5, budget / (2 * maxcost))
            weight = (1 - forced) * nominal + forced * safe
            w = np.array([1 - weight, weight])
        target += duration * w
        choice = int(np.argmax(target - actual))
        actual[choice] += duration
        nsteps = int(np.ceil(duration / dt))
        h = duration / nsteps
        root = np.sqrt(h)
        for j in range(nsteps):
            if model == 0:
                action = pc.ACTIONS[choice]
            elif choice == 0:
                action = 0.0
            elif width == 0:
                action = 1.0
            else:
                z = min(1.0, max(0.0, (x - (2.0 - width)) / width))
                action = z * z * (3 - 2 * z)
            credit = max(0.0, budget * (clock + (j + 1) * h) - energy)
            allowed = np.sqrt(credit / h)
            if abs(action) > allowed:
                action = np.sign(action) * allowed
                guard += 1
            if model == 0:
                f0 = action / sigma
                f1 = (action - 0.5) * x / sigma
                drift = -x + theta[0] * action + (action - 0.5) * theta[1] * x
                operation += ((x - 0.7) ** 2 + 0.15 * action * action) * h
            else:
                f0 = -x / sigma
                f1 = action * x / sigma
                drift = 0.2 - theta[0] * x + theta[1] * action * x
                operation += (0.2 * x + 0.2 * action * action) * h
            noise = np.random.normal() * root
            dy = (f0 * (theta[0] - ref[0]) + f1 * (theta[1] - ref[1])) * h + noise
            raw = x + drift * h + sigma * noise
            nx = min(upper, max(lower, raw))
            correction = nx - raw
            low = max(0.0, correction)
            high = max(0.0, -correction)
            reg0 += low
            regc += high
            if model == 1:
                operation += 10 * low + high
            g00 += f0 * f0 * h
            g01 += f0 * f1 * h
            g11 += f1 * f1 * h
            s0 += f0 * dy
            s1 += f1 * dy
            energy += action * action * h
            x = nx
            steps_total += 1
            excess = max(excess, energy - budget * (clock + (j + 1) * h))
        clock += duration
        previous_plus = margin_plus
        previous_minus = margin_minus
        answer, margin_plus, margin_minus = mixture_test(g00, g01, g11, s0, s1, ref, precision, np.log(1 / delta), plus, minus)
    return np.array([clock, energy, answer, reg0, regc, operation, k, guard, steps_total, excess, g00, g01, g11, s0, s1, x, margin_plus, margin_minus, previous_plus, previous_minus])

@njit(cache=True, parallel=True)
def paths(seeds, model, theta, budget, delta, dt, horizon, initial, width, method, tables, costs, qgrid, ggrid, plus, minus):
    out = np.zeros((len(seeds), 20))
    for r in prange(len(seeds)):
        out[r] = one_path(seeds[r], model, theta, budget, delta, dt, horizon, initial, width, method, tables, costs, qgrid, ggrid, plus, minus)
    return out

def execute(job, smoke=False, rerun=False):
    path = OUT / (job['id'] + '.json')
    if path.exists() and (not smoke) and (not rerun):
        d = json.loads(path.read_text())
        assert d['job'] == job and d['protocol_sha256'] == DIGEST
        return {'existing': job['id']}
    cov = np.load(OUT / 'covers.npz')
    theta = np.array(job['theta'])
    model = 0 if job['model'] == 'bilinear' else 1
    if model == 0:
        d = np.load(OUT / f"bilinear_designs_B{job['budget']:g}.npz")
        tables = d['tables']
        costs = np.ones((1, 1))
        plus = cov['bplus']
        minus = cov['bminus']
        advantage = float(pc.advantage(theta))
        method = 0 if job['method'] == 'dopt' else 1
    else:
        d = np.load(OUT / 'inventory_costs.npz')
        tables = np.zeros((2, 1, 1, 2))
        costs = d['costs'][np.argmin(abs(WIDTHS - job['width']))]
        plus = cov['iplus']
        minus = cov['iminus']
        advantage = inv.delta(theta)
        method = 0
    assert abs(advantage) >= 0.05
    seeds = np.random.SeedSequence([2026102504, job['stream']]).generate_state(job['reps'], dtype=np.uint32)
    start = time.perf_counter()
    arr = paths(seeds, model, theta, job['budget'], job['delta'], job['dt'], job['horizon'], job['initial'], job['width'], method, tables, costs, d['qgrid'], d['ggrid'], plus, minus)
    truth = 1 if advantage > 0 else -1
    runs = []
    for seed, row in zip(seeds, arr):
        r = {key: float(value) if np.isfinite(value) else None for key, value in zip(FIELDS, row)}
        r.update(seed=int(seed), stopped=bool(r['answer']), error=bool(r['answer'] and r['answer'] != truth))
        runs.append(r)
    result = {'job': job, 'protocol_sha256': DIGEST, 'advantage': advantage, 'cover_counts': [len(plus), len(minus)], 'wall_seconds': time.perf_counter() - start, 'smoke': smoke, 'runs': runs}
    destination = OUT / ('smoke_' + job['id'] + '.json') if smoke else path
    destination.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    summary = {'cell': job['id'], 'model': job['model'], 'method': job['method'], 'B': job['budget'], 'delta': job['delta'], 'width': job['width'], 'reps': job['reps'], 'time': float(arr[:, 0].mean()), 'censored': sum((not r['stopped'] for r in runs)), 'errors': sum((r['error'] for r in runs)), 'wall': result['wall_seconds']}
    print(json.dumps(summary), flush=True)
    return summary

def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['prepare', 'run', 'smoke'])
    p.add_argument('--model', choices=['bilinear', 'inventory'], default='bilinear')
    p.add_argument('--rerun', action='store_true', help='Regenerate the fixed production jobs without changing the protocol')
    args = p.parse_args()
    selected = save_manifest()
    if args.command == 'prepare':
        prepare()
        return
    if args.command == 'smoke':
        for model in ['bilinear', 'inventory']:
            job = next((j.copy() for j in selected if j['model'] == model))
            job['reps'] = 3
            job['horizon'] = 1000.0
            execute(job, smoke=True)
        return
    for job in selected:
        if job['model'] == args.model:
            execute(job, rerun=args.rerun)
if __name__ == '__main__':
    main()
