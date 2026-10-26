"""Bilinear diffusion simulations and numerical design utilities."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
from scipy.optimize import linprog
from scipy.special import ndtr
ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs'
L = 2.0
SIGMA = 0.7
TARGET = 0.7
ACTION_PRICE = 0.15
CANDIDATE = 0.6
OFFSET = 0.5
EPSILON = 0.05
LOW = np.array([-0.75, -0.6])
HIGH = np.array([1.5, 0.6])
ACTIONS = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])

def stationary_moments(theta, action):
    """Exact truncated-Gaussian first and second moments (float evaluation)."""
    theta = np.asarray(theta)
    action = np.asarray(action)
    rate = 1.0 - (action - OFFSET) * theta[..., 1]
    mean = action * theta[..., 0] / rate
    scale = SIGMA / np.sqrt(2.0 * rate)
    aa, bb = ((-L - mean) / scale, (L - mean) / scale)
    phi_a = np.exp(-aa * aa / 2.0) / np.sqrt(2.0 * np.pi)
    phi_b = np.exp(-bb * bb / 2.0) / np.sqrt(2.0 * np.pi)
    mass = np.where(aa > 0, ndtr(-aa) - ndtr(-bb), ndtr(bb) - ndtr(aa))
    if np.any(mass <= 1e-300):
        raise ValueError('Stationary truncation mass underflow')
    zmean = (phi_a - phi_b) / mass
    zsecond = 1.0 + (aa * phi_a - bb * phi_b) / mass
    first = mean + scale * zmean
    second = mean * mean + 2.0 * mean * scale * zmean + scale * scale * zsecond
    return (first, second)

def policy_cost(theta, action):
    m, s = stationary_moments(theta, action)
    return s - 2.0 * TARGET * m + TARGET * TARGET + ACTION_PRICE * action * action

def advantage(theta):
    return policy_cost(theta, 0.0) - policy_cost(theta, CANDIDATE)

def information(theta):
    m, s = stationary_moments(np.asarray(theta)[None, :], ACTIONS)
    result = np.zeros((len(ACTIONS), 2, 2))
    result[:, 0, 0] = ACTIONS ** 2 / SIGMA ** 2
    result[:, 0, 1] = result[:, 1, 0] = ACTIONS * (ACTIONS - OFFSET) * m / SIGMA ** 2
    result[:, 1, 1] = (ACTIONS - OFFSET) ** 2 * s / SIGMA ** 2
    return result

class ParameterMesh:

    def __init__(self, step=0.01):
        n = np.ceil((HIGH - LOW) / step).astype(int)
        self.width = (HIGH - LOW) / n
        axes = [LOW[i] + (np.arange(n[i]) + 0.5) * self.width[i] for i in range(2)]
        x, y = np.meshgrid(*axes, indexing='ij')
        self.points = np.column_stack([x.ravel(), y.ravel()])
        self.radius = float(np.linalg.norm(self.width / 2.0))
        self.delta = advantage(self.points)
        cost_range = (L + abs(TARGET)) ** 2
        lipschitz_coordinates = np.array([abs(CANDIDATE) * cost_range * L / SIGMA ** 2, (abs(OFFSET) + abs(CANDIDATE - OFFSET)) * cost_range * L ** 2 / (4.0 * SIGMA ** 2)])
        self.delta_error = float(lipschitz_coordinates @ (self.width / 2.0))
        self.possible = {1: self.delta + self.delta_error >= EPSILON, -1: self.delta - self.delta_error <= -EPSILON}
        self.inner = {1: self.delta >= EPSILON, -1: self.delta <= -EPSILON}

    def alternative_points(self, answer, conservative=True):
        mask = self.possible[-answer] if conservative else self.inner[-answer]
        return self.points[mask]

def quadratic_rows(theta, matrices, points, radius=0.0):
    diff = points - theta
    md = np.einsum('kij,nj->nki', matrices, diff)
    rows = 0.5 * np.einsum('ni,nki->nk', diff, md)
    if radius:
        rows -= radius * np.linalg.norm(md, axis=2)
    return rows

def solve_design(theta, budget, mesh, criterion='policy', side='lower'):
    mats = information(theta)
    costs = ACTIONS ** 2
    if criterion == 'balanced':
        w = np.zeros(len(ACTIONS))
        w[0] = w[-1] = budget / 2.0
        w[2] = 1.0 - budget
        return {'weights': w, 'value': None, 'matrices': mats}
    if criterion == 'trace':
        res = linprog(-np.trace(mats, axis1=1, axis2=2), A_ub=costs[None], b_ub=[budget], A_eq=np.ones((1, len(ACTIONS))), b_eq=[1.0], bounds=(0.0, None), method='highs')
        if not res.success:
            raise RuntimeError(res.message)
        return {'weights': res.x, 'value': None, 'matrices': mats}
    answer = 1 if advantage(theta) >= 0.0 else -1
    points = mesh.alternative_points(answer, conservative=side == 'lower')
    radius = mesh.radius if side == 'lower' else 0.0
    rows = quadratic_rows(theta, mats, points, radius)
    cuts = list(np.unique(np.argmin(rows, axis=0)))
    res = None
    for _ in range(100):
        a_ub = np.column_stack([-rows[cuts], np.ones(len(cuts))])
        a_ub = np.vstack([a_ub, np.r_[costs, 0.0]])
        res = linprog(np.r_[np.zeros(len(ACTIONS)), -1.0], A_ub=a_ub, b_ub=np.r_[np.zeros(len(cuts)), budget], A_eq=np.r_[np.ones(len(ACTIONS)), 0.0][None], b_eq=[1.0], bounds=[(0.0, None)] * len(ACTIONS) + [(None, None)], method='highs')
        if not res.success:
            raise RuntimeError(res.message)
        values = rows @ res.x[:-1]
        worst = int(np.argmin(values))
        if values[worst] >= res.x[-1] - 1e-09:
            break
        cuts.append(worst)
    else:
        raise RuntimeError('Design cutting planes did not converge')
    return {'weights': res.x[:-1], 'value': float(max(0.0, np.min(values))), 'matrices': mats, 'cuts': len(cuts)}

def design_value(theta, weights, mesh):
    answer = 1 if advantage(theta) >= 0 else -1
    mat = np.einsum('k,kij->ij', weights, information(theta))
    lo = np.min(quadratic_rows(theta, mat[None], mesh.alternative_points(answer, True), mesh.radius))
    hi = np.min(quadratic_rows(theta, mat[None], mesh.alternative_points(answer, False)))
    return (max(0.0, float(lo)), float(hi))

def frontier():
    OUT.mkdir(exist_ok=True)
    records = []
    for step in [0.02, 0.01, 0.005, 0.0025]:
        mesh = ParameterMesh(step)
        for theta in [np.array([1.0, 0.2]), np.array([-0.3, 0.2])]:
            for budget in [0.1, 0.25, 0.5, 1.0]:
                lower = solve_design(theta, budget, mesh)
                upper = solve_design(theta, budget, mesh, side='upper')
                row = {'theta': theta.tolist(), 'advantage': float(advantage(theta)), 'budget': budget, 'mesh_step': step, 'offset': OFFSET, 'mesh_points': len(mesh.points), 'delta_cell_error': mesh.delta_error, 'gamma_lower': lower['value'], 'gamma_upper': upper['value'], 'weights': lower['weights'].tolist()}
                for method in ['balanced', 'trace']:
                    d = solve_design(theta, budget, mesh, method)
                    lo, hi = design_value(theta, d['weights'], mesh)
                    row[method + '_lower'] = lo
                    row[method + '_upper'] = hi
                    row[method + '_weights'] = d['weights'].tolist()
                records.append(row)
                print(json.dumps(row), flush=True)
    (OUT / 'frontier.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    return records

def stopping_decision(center, vmat, radius, mesh):
    eigmax = float(np.linalg.eigvalsh(vmat)[-1])
    cell_penalty = np.sqrt(eigmax) * mesh.radius
    for answer in [1, -1]:
        points = mesh.alternative_points(answer, True)
        d = points - center
        squared = np.einsum('ni,ij,nj->n', d, vmat, d)
        lower_distance = max(0.0, np.sqrt(max(0.0, float(np.min(squared)))) - cell_penalty)
        if lower_distance > radius:
            return answer
    return 0

def simulate(theta, budget, method, reps, dt, max_time, delta, seed, mesh_step=0.01):
    rng = np.random.default_rng(seed)
    theta = np.asarray(theta, dtype=float)
    mesh = ParameterMesh(mesh_step)
    ridge = 0.1
    parameter_radius = float(np.max(np.linalg.norm(np.array([[LOW[0], LOW[1]], [LOW[0], HIGH[1]], [HIGH[0], LOW[1]], [HIGH[0], HIGH[1]]]), axis=1)))
    x = np.zeros(reps)
    gram = np.zeros((reps, 2, 2))
    sufficient = np.zeros((reps, 2))
    energy = np.zeros(reps)
    stopped = np.zeros(reps, dtype=bool)
    result = [None] * reps
    cache = {}
    clock = 0.0
    block = 0
    ideal = solve_design(theta, budget, mesh, side='upper')['weights']
    safe = solve_design(theta, budget / 2.0, mesh, 'balanced')['weights']
    while clock < max_time and (not np.all(stopped)):
        block += 1
        active = np.flatnonzero(~stopped)
        choices = np.zeros(reps, dtype=int)
        for i in active:
            v = gram[i] + ridge * np.eye(2)
            center = np.linalg.solve(v, sufficient[i])
            estimate = np.clip(center, LOW, HIGH)
            if method == 'oracle':
                w = ideal
            elif method == 'balanced':
                w = solve_design(theta, budget, mesh, 'balanced')['weights']
            else:
                estimate = np.clip(np.round(estimate / 0.05) * 0.05, LOW, HIGH)
                key = tuple(estimate)
                if key not in cache:
                    cache[key] = solve_design(estimate, budget, mesh, 'trace' if method == 'trace' else 'policy', side='upper')['weights']
                w = cache[key]
            forced = block ** (-0.25)
            w = (1.0 - forced) * w + forced * safe
            w = np.maximum(w, 0.0)
            w /= w.sum()
            choices[i] = rng.choice(len(ACTIONS), p=w)
        a = ACTIONS[choices]
        duration = min(0.5 * block, max_time - clock)
        steps = max(1, int(np.ceil(duration / dt)))
        h = duration / steps
        for _ in range(steps):
            ii = np.flatnonzero(~stopped)
            xx, aa = (x[ii], a[ii])
            feature = np.column_stack([aa, (aa - OFFSET) * xx]) / SIGMA
            noise = rng.normal(size=len(ii)) * np.sqrt(h)
            dy = feature @ theta * h + noise
            raw = -xx * h + SIGMA * dy
            x[ii] = np.clip(xx + raw, -L, L)
            gram[ii] += np.einsum('ni,nj->nij', feature, feature) * h
            sufficient[ii] += feature * dy[:, None]
            energy[ii] += aa * aa * h
        clock += duration
        for i in np.flatnonzero(~stopped):
            v = gram[i] + ridge * np.eye(2)
            center = np.linalg.solve(v, sufficient[i])
            _, logdet = np.linalg.slogdet(v)
            radius = np.sqrt(2.0 * (np.log(1.0 / delta) + 0.5 * logdet - np.log(ridge))) + np.sqrt(ridge) * parameter_radius
            decision = stopping_decision(center, v, radius, mesh)
            if decision:
                stopped[i] = True
                truth = 1 if advantage(theta) >= EPSILON else -1
                result[i] = {'stopped': True, 'time': clock, 'energy': float(energy[i]), 'average_energy': float(energy[i] / clock), 'decision': decision, 'error': decision != truth}
        if block % 20 == 0:
            print(json.dumps({'progress': method, 'theta': theta.tolist(), 'dt': dt, 'time': clock, 'stopped': int(stopped.sum()), 'reps': reps}), flush=True)
    for i in np.flatnonzero(~stopped):
        result[i] = {'stopped': False, 'time': clock, 'energy': float(energy[i]), 'average_energy': float(energy[i] / clock), 'decision': 0, 'error': None}
    return {'theta': theta.tolist(), 'advantage': float(advantage(theta)), 'budget': budget, 'method': method, 'reps': reps, 'dt': dt, 'delta': delta, 'max_time': max_time, 'seed': seed, 'mesh_step': mesh_step, 'offset': OFFSET, 'design_cache_size': len(cache), 'design_precision': 'opposite-class centre mesh; estimate rounded to 0.05', 'stopping_precision': 'conservative parameter-cell exclusion', 'runs': result}

def validate():
    from scipy.integrate import quad
    checks = []
    for theta in [np.array([1.0, 0.2]), LOW, HIGH, np.array([-0.3, 0.2]), np.array([LOW[0], HIGH[1]]), np.array([HIGH[0], LOW[1]])]:
        for action in [-1.0, 0.0, 0.6, 1.0]:

            def unnorm(z):
                return np.exp((-(1 - (action - OFFSET) * theta[1]) * z * z + 2 * action * theta[0] * z) / SIGMA ** 2)
            norm = quad(unnorm, -L, L, epsabs=1e-10)[0]
            exact = stationary_moments(theta, action)
            quadrature = [quad(lambda z: z ** p * unnorm(z), -L, L, epsabs=1e-10)[0] / norm for p in [1, 2]]
            checks.append(max(abs(np.asarray(exact) - quadrature)) < 1e-08)
    mesh = ParameterMesh(0.02)
    for theta in [np.array([1.0, 0.2]), np.array([-0.3, 0.2])]:
        for b in [0.1, 0.5, 1.0]:
            lo = solve_design(theta, b, mesh)
            hi = solve_design(theta, b, mesh, side='upper')
            checks.extend([lo['value'] <= hi['value'] + 1e-08, abs(lo['weights'].sum() - 1) < 1e-08, lo['weights'] @ ACTIONS ** 2 <= b + 1e-08])
    rng = np.random.default_rng(10425)
    ncell = np.ceil((HIGH - LOW) / 0.02).astype(int)
    probes = rng.uniform(LOW, HIGH, size=(500, 2))
    indices = np.minimum(((probes - LOW) / mesh.width).astype(int), ncell - 1)
    centres = LOW + (indices + 0.5) * mesh.width
    checks.append(np.max(np.abs(advantage(probes) - advantage(centres))) <= mesh.delta_error + 1e-12)
    mats = information(np.array([1.0, 0.2]))
    lowers = quadratic_rows(np.array([1.0, 0.2]), mats, centres, mesh.radius)
    actual = quadratic_rows(np.array([1.0, 0.2]), mats, probes)
    checks.append(bool(np.all(lowers <= actual + 1e-10)))
    for truth in [np.array([1.0, 0.2]), np.array([-0.3, 0.2])]:
        sound = True
        sign = 1 if advantage(truth) >= EPSILON else -1
        for _ in range(20):
            a = rng.normal(size=(2, 2))
            v = a.T @ a + 0.01 * np.eye(2)
            center = truth + rng.normal(size=2)
            radius = np.sqrt((truth - center) @ v @ (truth - center)) + 1e-08
            sound &= stopping_decision(center, v, radius, mesh) != -sign
        checks.append(bool(sound))
    if not all(checks):
        raise AssertionError('Numerical validation failed')
    OUT.mkdir(exist_ok=True)
    record = {'checks_passed': len(checks), 'all_passed': True, 'scope': 'Quadrature including tail corners; design budgets and brackets; independent cell-cover/quadratic checks; no false exclusion of contained truths.'}
    (OUT / 'validation.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(json.dumps(record))
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['validate', 'frontier', 'simulate'])
    p.add_argument('--method', default='policy', choices=['policy', 'trace', 'balanced', 'oracle'])
    p.add_argument('--theta', nargs=2, type=float, default=[1.0, 0.2])
    p.add_argument('--budget', type=float, default=0.25)
    p.add_argument('--reps', type=int, default=30)
    p.add_argument('--dt', type=float, default=0.02)
    p.add_argument('--max-time', type=float, default=1500.0)
    p.add_argument('--delta', type=float, default=0.05)
    p.add_argument('--seed', type=int, default=527)
    p.add_argument('--mesh-step', type=float, default=0.01)
    p.add_argument('--output', default='simulation.json')
    args = p.parse_args()
    if args.command == 'validate':
        validate()
    elif args.command == 'frontier':
        frontier()
    else:
        start = time.perf_counter()
        record = simulate(args.theta, args.budget, args.method, args.reps, args.dt, args.max_time, args.delta, args.seed, args.mesh_step)
        record['wall_seconds'] = time.perf_counter() - start
        OUT.mkdir(exist_ok=True)
        (OUT / args.output).write_text(json.dumps(record, indent=2), encoding='utf-8')
        print(json.dumps({k: v for k, v in record.items() if k != 'runs'}), flush=True)
