"""Numerical checks of experiment inputs, allocation, and saved run statistics."""
import argparse
import hashlib
import json
from pathlib import Path
import mpmath as mp
import numpy as np
from scipy.optimize import linprog, minimize
import followup_v4 as study

def in_cover(point, boxes):
    x, y = point
    return bool(np.any((boxes[:, 0] <= x) & (x <= boxes[:, 1]) & (boxes[:, 2] <= y) & (y <= boxes[:, 3])))

def log_min(row, boxes, ref, precision):
    mp.mp.dps = 70
    a, b, c, s0, s1 = [mp.mpf(row[k]) for k in ['g00', 'g01', 'g11', 's0', 's1']]
    g = mp.matrix([[a, b], [b, c]])
    s = mp.matrix([s0, s1])
    p = mp.diag([mp.mpf(float(v)) for v in precision])
    z = (mp.log(mp.det(p)) - mp.log(mp.det(g + p)) + (s.T * (g + p) ** (-1) * s)[0]) / 2
    best = mp.inf
    centre = g ** (-1) * s
    for box in boxes:
        l, u, v, w = [mp.mpf(float(t)) for t in box]
        l -= float(ref[0])
        u -= float(ref[0])
        v -= float(ref[1])
        w -= float(ref[1])
        clip = lambda t, lo, hi: max(lo, min(hi, t))
        candidates = [(x, clip((s1 - b * x) / c, v, w)) for x in [l, u]]
        candidates += [(clip((s0 - b * y) / a, l, u), y) for y in [v, w]]
        if l <= centre[0] <= u and v <= centre[1] <= w:
            candidates.append((centre[0], centre[1]))
        best = min(best, *[z - s0 * x - s1 * y + (a * x * x + 2 * b * x * y + c * y * y) / 2 for x, y in candidates])
    return best

def preflight():
    rng = np.random.default_rng(2026102505)
    covers = np.load(study.OUT / 'covers.npz')
    allseeds = []
    max_gap = 0.0
    max_discrepancy = 0.0
    for job in study.jobs():
        allseeds.extend(map(int, np.random.SeedSequence([2026102504, job['stream']]).generate_state(job['reps'], dtype=np.uint32)))
    assert len(allseeds) == 11220 and len(set(allseeds)) == 11220
    for model in ['bilinear', 'inventory']:
        low, high = (study.pc.LOW, study.pc.HIGH) if model == 'bilinear' else (np.array([0.45, 0.01]), np.array([0.6, 0.4]))
        points = rng.uniform(low, high, (3000, 2))
        advantage = study.pc.advantage(points) if model == 'bilinear' else np.array([study.inv.delta(p) for p in points])
        for point, delta in zip(points, advantage):
            if abs(delta) >= 0.05:
                assert in_cover(point, covers[('b' if model == 'bilinear' else 'i') + ('plus' if delta > 0 else 'minus')])
    for budget in [0.5, 1.0]:
        table = np.load(study.OUT / f'bilinear_designs_B{budget:g}.npz')
        cap = 0.9 * budget
        cost = study.pc.ACTIONS ** 2
        for i, q in enumerate(table['qgrid']):
            for j, g in enumerate(table['ggrid']):
                mats = study.pc.information(np.array([q, g]))
                weights = table['tables'][:, i, j]
                assert np.all(weights >= 0) and np.allclose(weights.sum(axis=1), 1)
                assert np.all(weights @ cost <= cap + 1e-10)
                matrix = np.einsum('j,jab->ab', weights[0], mats)
                gradient = np.einsum('ab,jba->j', np.linalg.inv(matrix), mats)
                res = linprog(-gradient, A_ub=cost[None], b_ub=[cap], A_eq=np.ones((1, 5)), b_eq=[1], bounds=(0, None), method='highs')
                assert res.success
                gap = max(0.0, float(gradient @ (res.x - weights[0])))
                max_gap = max(max_gap, gap)
                assert gap < 0.0001, (budget, i, j, gap)
    ref = np.array([0.375, 0.0])
    precision = np.array([1 / 1.125 ** 2, 1 / 0.6 ** 2])
    for _ in range(50):
        a = rng.normal(size=(2, 2))
        g = (a.T @ a + np.eye(2)) * rng.uniform(1, 1000)
        s = rng.normal(size=2) * 100
        plus = np.array([[0.5, 1.0, -0.3, 0.3]])
        minus = np.array([[-0.7, -0.1, -0.5, 0.5]])
        answer, pm, mm = study.mixture_test(g[0, 0], g[0, 1], g[1, 1], *s, ref, precision, np.log(20), plus, minus)
        v = g + np.diag(precision)
        logz = 0.5 * (np.log(precision.prod()) - np.linalg.slogdet(v)[1] + s @ np.linalg.solve(v, s))

        def exact(box):
            lo = box[0, [0, 2]] - ref
            hi = box[0, [1, 3]] - ref
            fun = lambda z: logz - z @ s + 0.5 * z @ g @ z
            fit = minimize(fun, np.clip(np.linalg.solve(g, s), lo, hi), jac=lambda z: g @ z - s, method='L-BFGS-B', bounds=list(zip(lo, hi)), options={'ftol': 1e-14, 'gtol': 1e-10})
            return 2 * (fit.fun - np.log(20))
        discrepancy = max(abs(pm - exact(minus)), abs(mm - exact(plus)))
        max_discrepancy = max(max_discrepancy, discrepancy)
        assert discrepancy < 1e-06
    record = {'passed': True, 'unique_seeds': len(allseeds), 'class_cover_probe_points': 6000, 'dopt_designs_checked': 1550, 'maximum_dopt_logdet_first_order_gap': max_gap, 'mixture_optimization_crosschecks': 50, 'maximum_mixture_margin_discrepancy': max_discrepancy, 'cover_counts': {k: int(len(covers[k])) for k in ['bplus', 'bminus', 'iplus', 'iminus']}, 'protocol_sha256': study.DIGEST, 'scope': 'Numerical preflight, not a machine-verified coverage guarantee.'}
    (study.OUT / 'preflight.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(json.dumps(record, indent=2))

def results():
    cov = np.load(study.OUT / 'covers.npz')
    count = 0
    censored = 0
    errors = 0
    checked = 0
    max_excess = 0.0
    for job in study.jobs():
        data = json.loads((study.OUT / (job['id'] + '.json')).read_text())
        assert data['job'] == job and data['protocol_sha256'] == study.DIGEST and (not data['smoke'])
        assert len(data['runs']) == job['reps']
        for i, r in enumerate(data['runs']):
            count += 1
            errors += r['error']
            censored += not r['stopped']
            max_excess = max(max_excess, r['max_budget_excess'])
            assert r['energy'] <= job['budget'] * r['time'] + 1e-07 and r['max_budget_excess'] < 1e-07
            assert r['stopped'] or abs(r['time'] - job['horizon']) < 1e-06
            assert r['previous_plus_margin'] is None or r['previous_plus_margin'] <= 0
            assert r['previous_minus_margin'] is None or r['previous_minus_margin'] <= 0
            truth = 1 if data['advantage'] > 0 else -1
            assert r['error'] == bool(r['answer'] and r['answer'] != truth)
            if r['stopped']:
                assert r['plus_margin' if r['answer'] == 1 else 'minus_margin'] > 0
                if i == 0 or r['error']:
                    b = job['model'] == 'bilinear'
                    ref = np.array([0.375, 0.0]) if b else study.inv.REF
                    precision = np.array([1 / 1.125 ** 2, 1 / 0.6 ** 2]) if b else study.inv.PRECISION
                    boxes = cov[('b' if b else 'i') + ('minus' if r['answer'] == 1 else 'plus')]
                    margin = log_min(r, boxes, ref, precision) - mp.log(mp.mpf(1) / mp.mpf(job['delta']))
                    assert margin > 0, (job['id'], i, str(margin))
                    checked += 1
    assert count == 11220
    record = {'passed': True, 'scheduled_runs': count, 'censored': censored, 'errors': errors, 'high_precision_terminal_exclusions': checked, 'maximum_pathwise_budget_excess': max_excess, 'protocol_sha256': study.DIGEST, 'scope': 'Every saved run checked; representative exclusions recalculated at 70 digits. This is not a proof of uniform coverage.'}
    (study.OUT / 'validation.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(json.dumps(record, indent=2))
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['preflight', 'results'])
    args = p.parse_args()
    (preflight if args.command == 'preflight' else results)()
