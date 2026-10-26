"""Sequential inventory simulations with explicit resource accounting."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import time
os.environ.setdefault('NUMBA_NUM_THREADS', '4')
import numpy as np
from numba import njit, prange
from scipy.optimize import linprog, minimize
import inventory_frontier as inv
from policy_certification import quadratic_rows
OUT = inv.OUT / 'sequential'
QLOW, QHIGH = (0.45, 0.6)
GLOW, GHIGH = (0.01, 0.4)
REF = np.array([0.525, 0.205])
PRECISION = np.array([1 / 0.075 ** 2, 1 / 0.195 ** 2])
KINDS = ['baseline', 'candidate', 'gate_1.0', 'gate_0.5', 'gate_0.25', 'gate_0.1']
WIDTHS = np.array([0.0, 0.0, 1.0, 0.5, 0.25, 0.1])
METHODS = {'ab': 0, 'dopt': 1, 'policy': 2, 'oracle': 3}

def delta(theta):
    return float(inv.constant_cost(theta[0], theta[1], 0.0) - inv.constant_cost(theta[0], theta[1], 1.0))

def class_cover(epsilon=0.05, leaf=0.0005):
    """Cover each entire promised class by adaptively subdivided rectangles.

    A box wholly inside a promised class is accepted early. Boxes intersecting
    a class boundary are split to the leaf width and retained conservatively.
    Density-ratio bounds include both boundary regulator cost terms.
    """
    covers = {1: [], -1: []}
    stack = [np.array([QLOW, QHIGH, GLOW, GHIGH])]
    while stack:
        box = stack.pop()
        centre = np.array([(box[0] + box[1]) / 2, (box[2] + box[3]) / 2])
        half = np.array([(box[1] - box[0]) / 2, (box[3] - box[2]) / 2])
        j0 = float(inv.constant_cost(*centre, 0.0))
        j1 = float(inv.constant_cost(*centre, 1.0))
        osc0 = 16 * half[0]
        osc1 = 16 * sum(half)
        lo = np.exp(-osc0) * j0 - np.exp(osc1) * j1
        hi = np.exp(osc0) * j0 - np.exp(-osc1) * j1
        if lo >= epsilon:
            covers[1].append(box)
        elif hi <= -epsilon:
            covers[-1].append(box)
        elif lo > -epsilon and hi < epsilon:
            continue
        elif max(2 * half) <= leaf:
            if hi >= epsilon:
                covers[1].append(box)
            if lo <= -epsilon:
                covers[-1].append(box)
        else:
            axis = int(np.argmax(half))
            a, b = (box.copy(), box.copy())
            if axis == 0:
                a[1] = centre[0]
                b[0] = centre[0]
            else:
                a[3] = centre[1]
                b[2] = centre[1]
            stack.extend([a, b])

    def merge(boxes, axis):
        lo, hi = (0, 1) if axis == 0 else (2, 3)
        fixed = (2, 3) if axis == 0 else (0, 1)
        boxes = sorted(boxes, key=lambda b: (b[fixed[0]], b[fixed[1]], b[lo], b[hi]))
        result = []
        for box in boxes:
            if result and result[-1][fixed[0]] == box[fixed[0]] and (result[-1][fixed[1]] == box[fixed[1]]) and (box[lo] <= result[-1][hi]):
                result[-1][hi] = max(result[-1][hi], box[hi])
            else:
                result.append(box.copy())
        return result
    for sign in [1, -1]:
        while True:
            before = len(covers[sign])
            covers[sign] = merge(merge(covers[sign], 0), 1)
            if len(covers[sign]) == before:
                break
    return {s: np.asarray(v, dtype=float) for s, v in covers.items()}

def design_points(step=0.004):
    n = np.ceil((np.array([QHIGH, GHIGH]) - np.array([QLOW, GLOW])) / step).astype(int)
    w = (np.array([QHIGH, GHIGH]) - np.array([QLOW, GLOW])) / n
    q = QLOW + (np.arange(n[0]) + 0.5) * w[0]
    g = GLOW + (np.arange(n[1]) + 0.5) * w[1]
    xx, yy = np.meshgrid(q, g, indexing='ij')
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    dd = inv.constant_cost(pts[:, 0], pts[:, 1], 0.0) - inv.constant_cost(pts[:, 0], pts[:, 1], 1.0)
    return {1: pts[dd >= 0.05], -1: pts[dd <= -0.05]}

def moments(theta):
    stats = [inv.policy([1.0, *theta], kind, 2001) for kind in KINDS]
    return (np.array([s['matrix'][1:, 1:] for s in stats]), np.array([s['spend'] for s in stats]))

def policy_weights(theta, budget, points, mats, costs):
    sign = 1 if delta(theta) >= 0 else -1
    rows = quadratic_rows(np.asarray(theta), mats, points[-sign])
    cuts = list(np.unique(np.argmin(rows, axis=0)))
    for _ in range(100):
        aub = np.vstack([np.column_stack([-rows[cuts] / budget, np.ones(len(cuts))]), np.r_[costs / budget, 0.0]])
        res = linprog(np.r_[np.zeros(6), -1.0], A_ub=aub, b_ub=np.r_[np.zeros(len(cuts)), 1.0], A_eq=np.r_[np.ones(6), 0.0][None], b_eq=[1.0], bounds=[(0.0, None)] * 6 + [(None, None)], method='highs', options={'primal_feasibility_tolerance': 1e-09, 'dual_feasibility_tolerance': 1e-09})
        if not res.success:
            raise RuntimeError(res.message)
        vals = rows @ res.x[:-1]
        worst = int(np.argmin(vals))
        if vals[worst] >= budget * res.x[-1] - 1e-10:
            w = np.maximum(0.0, res.x[:-1])
            w /= sum(w)
            if costs @ w > budget:
                w[1:] *= budget / (costs @ w)
                w[0] = 1 - sum(w[1:])
            return w
        cuts.append(worst)
    raise RuntimeError('Design solver failed')

def dopt_weights(budget, mats, costs):
    start = np.zeros(6)
    start[1] = budget / 2
    start[0] = 1 - start[1]

    def objective(w):
        m = np.einsum('j,jab->ab', w, mats)
        sign, logdet = np.linalg.slogdet(m)
        if sign <= 0:
            return (1000000000000.0, np.zeros(6))
        inverse = np.linalg.inv(m)
        grad = -np.einsum('ab,jba->j', inverse, mats)
        return (-logdet, grad)
    res = minimize(objective, start, jac=True, method='SLSQP', bounds=[(0.0, 1.0)] * 6, constraints=[{'type': 'eq', 'fun': lambda w: sum(w) - 1, 'jac': lambda w: np.ones(6)}, {'type': 'ineq', 'fun': lambda w: 1 - costs @ w / budget, 'jac': lambda w: -costs / budget}], options={'ftol': 1e-10, 'maxiter': 300})
    if not res.success:
        raise RuntimeError('D-optimal solver: ' + res.message)
    w = np.maximum(res.x, 0.0)
    w /= sum(w)
    if costs @ w > budget:
        w[1:] *= budget / (costs @ w)
        w[0] = 1 - sum(w[1:])
    return w

def prepare(budget):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f'designs_B{budget:g}.npz'
    if path.exists():
        return path
    points = design_points()
    qgrid = np.linspace(QLOW, QHIGH, 16)
    ggrid = np.linspace(GLOW, GHIGH, 27)
    tables = np.zeros((2, len(qgrid), len(ggrid), 6))
    cap = 0.9 * budget
    start = time.perf_counter()
    for i, q in enumerate(qgrid):
        for j, g in enumerate(ggrid):
            theta = np.array([q, g])
            mats, costs = moments(theta)
            tables[0, i, j] = dopt_weights(cap, mats, costs)
            tables[1, i, j] = policy_weights(theta, cap, points, mats, costs)
        if i % 4 == 0:
            print(json.dumps({'preparing_budget': budget, 'q_index': i}), flush=True)
    np.savez_compressed(path, tables=tables, qgrid=qgrid, ggrid=ggrid)
    print(json.dumps({'prepared': str(path), 'seconds': time.perf_counter() - start}), flush=True)
    return path

@njit(cache=True)
def quadratic(g00, g01, g11, d0, d1):
    return g00 * d0 * d0 + 2 * g01 * d0 * d1 + g11 * d1 * d1

@njit(cache=True)
def rectangle_distance(g00, g01, g11, c0, c1, boxes):
    best = 1e+300
    for j in range(len(boxes)):
        a, b, c, d = boxes[j]
        if a <= c0 <= b and c <= c1 <= d:
            return 0.0
        for x in [a, b]:
            y = min(d, max(c, c1 - g01 / g11 * (x - c0)))
            best = min(best, quadratic(g00, g01, g11, x - c0, y - c1))
        for y in [c, d]:
            x = min(b, max(a, c0 - g01 / g00 * (y - c1)))
            best = min(best, quadratic(g00, g01, g11, x - c0, y - c1))
    return best

@njit(cache=True)
def certificate(g00, g01, g11, s0, s1, logdelta, plus, minus):
    det = g00 * g11 - g01 * g01
    if det < 1e-10 or g00 <= 0 or g11 <= 0:
        return 0
    b0 = (g11 * s0 - g01 * s1) / det
    b1 = (g00 * s1 - g01 * s0) / det
    v00 = g00 + PRECISION[0]
    v11 = g11 + PRECISION[1]
    vdet = v00 * v11 - g01 * g01
    score_g = s0 * b0 + s1 * b1
    score_v = (v11 * s0 * s0 - 2 * g01 * s0 * s1 + v00 * s1 * s1) / vdet
    radius2 = 2 * logdelta + np.log(vdet / (PRECISION[0] * PRECISION[1])) + score_g - score_v
    c0 = REF[0] + b0
    c1 = REF[1] + b1
    if rectangle_distance(g00, g01, g11, c0, c1, minus) > radius2:
        return 1
    if rectangle_distance(g00, g01, g11, c0, c1, plus) > radius2:
        return -1
    return 0

@njit(cache=True)
def one_path(seed, theta, method, budget, dt, horizon, delta_value, tables, qgrid, ggrid, oracle, plus, minus, block_scale, budget_guard):
    np.random.seed(seed)
    x = 0.5
    clock = 0.0
    energy = 0.0
    reg0 = 0.0
    regc = 0.0
    holding = 0.0
    g00 = 0.0
    g01 = 0.0
    g11 = 0.0
    s0 = 0.0
    s1 = 0.0
    target = np.zeros(6)
    actual = np.zeros(6)
    answer = 0
    curtailed = 0
    totalsteps = 0
    max_excess = 0.0
    k = 0
    while clock < horizon and answer == 0:
        k += 1
        duration = min(1.0 + block_scale * k, horizon - clock)
        v00 = g00 + PRECISION[0]
        v11 = g11 + PRECISION[1]
        det = v00 * v11 - g01 * g01
        estq = min(QHIGH, max(QLOW, REF[0] + (v11 * s0 - g01 * s1) / det))
        estg = min(GHIGH, max(GLOW, REF[1] + (v00 * s1 - g01 * s0) / det))
        w = np.zeros(6)
        if method == 0:
            w[1] = 0.9 * budget
            w[0] = 1 - w[1]
        elif method == 3:
            w = oracle.copy()
        else:
            iq = int(np.round((estq - QLOW) / (QHIGH - QLOW) * (len(qgrid) - 1)))
            ig = int(np.round((estg - GLOW) / (GHIGH - GLOW) * (len(ggrid) - 1)))
            w = tables[method - 1, iq, ig].copy()
        forced = k ** (-0.25)
        w *= 1 - forced
        w[1] += forced * 0.45 * budget
        w[0] += forced * (1 - 0.45 * budget)
        target += duration * w
        choice = int(np.argmax(target - actual))
        actual[choice] += duration
        steps = int(np.ceil(duration / dt))
        step = duration / steps
        root = np.sqrt(step)
        for j in range(steps):
            if choice == 0:
                effort = 0.0
            elif choice == 1:
                effort = 1.0
            else:
                z = min(1.0, max(0.0, (x - (2.0 - WIDTHS[choice])) / WIDTHS[choice]))
                effort = z * z * (3.0 - 2.0 * z)
            if budget_guard:
                credit = max(0.0, budget * (clock + (j + 1) * step) - energy)
                allowed = np.sqrt(credit / step)
                if effort > allowed:
                    effort = allowed
                    curtailed += 1
            f0 = -x / 0.5
            f1 = effort * x / 0.5
            noise = np.random.normal() * root
            dy = (f0 * (theta[0] - REF[0]) + f1 * (theta[1] - REF[1])) * step + noise
            increment = (0.2 - theta[0] * x + theta[1] * effort * x) * step + 0.5 * noise
            raw = x + increment
            nx = min(2.0, max(0.0, raw))
            correction = nx - raw
            reg0 += max(0.0, correction)
            regc += max(0.0, -correction)
            holding += 0.2 * x * step
            g00 += f0 * f0 * step
            g01 += f0 * f1 * step
            g11 += f1 * f1 * step
            s0 += f0 * dy
            s1 += f1 * dy
            energy += effort * effort * step
            totalsteps += 1
            max_excess = max(max_excess, energy - budget * (clock + (j + 1) * step))
            x = nx
        clock += duration
        answer = certificate(g00, g01, g11, s0, s1, np.log(1 / delta_value), plus, minus)
    operation = holding + 0.2 * energy + 10 * reg0 + regc
    return np.array([clock, energy, answer, reg0, regc, operation, k, estq, estg, max_excess, curtailed, totalsteps])

@njit(cache=True, parallel=True)
def run_paths(seeds, theta, method, budget, dt, horizon, delta_value, tables, qgrid, ggrid, oracle, plus, minus, block_scale, budget_guard):
    output = np.zeros((len(seeds), 12))
    for i in prange(len(seeds)):
        output[i] = one_path(seeds[i], theta, method, budget, dt, horizon, delta_value, tables, qgrid, ggrid, oracle, plus, minus, block_scale, budget_guard)
    return output

def simulate(args):
    data = np.load(prepare(args.budget))
    cover = class_cover(leaf=args.leaf)
    theta = np.array(args.theta)
    if abs(delta(theta)) < 0.05:
        raise ValueError('Simulation parameter is outside the promised classes')
    points = design_points()
    mats, costs = moments(theta)
    oracle = policy_weights(theta, 0.9 * args.budget, points, mats, costs)
    method = METHODS[args.method]
    seeds = np.random.SeedSequence([args.seed, method]).generate_state(args.reps, dtype=np.uint32)
    start = time.perf_counter()
    arr = run_paths(seeds, theta, method, args.budget, args.dt, args.max_time, args.delta, data['tables'], data['qgrid'], data['ggrid'], oracle, cover[1], cover[-1], args.block_scale, not args.no_guard)
    truth = 1 if delta(theta) >= 0.05 else -1
    runs = [{'time': float(r[0]), 'energy': float(r[1]), 'answer': int(r[2]), 'stopped': bool(r[2]), 'error': bool(r[2] and int(r[2]) != truth), 'shortage_regulator': float(r[3]), 'overflow_regulator': float(r[4]), 'operating_cost': float(r[5]), 'blocks': int(r[6]), 'estimate': [float(r[7]), float(r[8])], 'max_budget_excess': float(r[9]), 'curtailed_steps': int(r[10]), 'total_steps': int(r[11])} for r in arr]
    record = {'method': args.method, 'theta': theta.tolist(), 'advantage': delta(theta), 'budget': args.budget, 'design_budget': 0.9 * args.budget, 'delta': args.delta, 'dt': args.dt, 'max_time': args.max_time, 'reps': args.reps, 'seed': args.seed, 'block_scale': args.block_scale, 'leaf': args.leaf, 'budget_guard': not args.no_guard, 'plus_boxes': len(cover[1]), 'minus_boxes': len(cover[-1]), 'certificate': 'fixed Gaussian-prior likelihood mixture; conservative rectangle-class cover', 'controller_tracking': 'deterministic weighted deficits', 'model': 'projected Euler Gaussian regression with observed regulator', 'wall_seconds': time.perf_counter() - start, 'runs': runs}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / args.output).write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(json.dumps({'output': args.output, 'stops': sum((r['stopped'] for r in runs)), 'errors': sum((r['error'] for r in runs)), 'mean_capped_time': float(arr[:, 0].mean()), 'energy_time': float(arr[:, 1].sum() / arr[:, 0].sum()), 'wall_seconds': record['wall_seconds'], 'cover_boxes': [len(cover[1]), len(cover[-1])]}), flush=True)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['prepare', 'simulate'])
    p.add_argument('--method', choices=list(METHODS), default='policy')
    p.add_argument('--theta', type=float, nargs=2, default=[0.5, 0.3])
    p.add_argument('--budget', type=float, default=0.01)
    p.add_argument('--reps', type=int, default=100)
    p.add_argument('--dt', type=float, default=0.02)
    p.add_argument('--max-time', type=float, default=200000.0)
    p.add_argument('--delta', type=float, default=0.05)
    p.add_argument('--seed', type=int, default=20261025)
    p.add_argument('--leaf', type=float, default=0.0005)
    p.add_argument('--block-scale', type=float, default=0.01)
    p.add_argument('--output', default='pilot.json')
    p.add_argument('--no-guard', action='store_true')
    args = p.parse_args()
    if args.command == 'prepare':
        prepare(args.budget)
    else:
        simulate(args)
