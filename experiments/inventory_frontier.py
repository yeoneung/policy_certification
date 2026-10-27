"""Inventory diffusion models, stationary quadrature, and grid optimization."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.integrate import cumulative_trapezoid, simpson
from scipy.optimize import brentq, linprog
from scipy.special import ndtr
ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs' / 'inventory'
C = 2.0
SIGMA = 0.5
U0 = 1.0
DEMAND = 0.8
V0 = 0.0
OMEGA = 1.0
HOLDING = 0.2
SHORTAGE = 10.0
OVERFLOW = 1.0
ENERGY = 0.2

def gate(x, width, amplitude=1.0):
    """C1 smooth probe supported at inventory levels above C-width."""
    z = np.clip((x - (C - width)) / width, 0.0, 1.0)
    return amplitude * z * z * (3.0 - 2.0 * z)

def stationary(theta, x, u, v):
    eta, q, gamma = theta
    drift = eta * u - DEMAND - q * x + gamma * (v - V0) * x
    potential = cumulative_trapezoid(2.0 * drift / SIGMA ** 2, x, initial=0.0)
    density = np.exp(potential - np.max(potential))
    density /= simpson(density, x=x)
    features = np.column_stack([u, -x, (v - V0) * x]) / SIGMA
    matrix = np.einsum('ni,nj->nij', features, features)
    info = simpson(matrix * density[:, None, None], x=x, axis=0)
    spend = (u - U0) ** 2 + OMEGA * (v - V0) ** 2
    cost = float(simpson(spend * density, x=x))
    rate0 = SIGMA ** 2 * density[0] / 2.0
    ratec = SIGMA ** 2 * density[-1] / 2.0
    average = float(simpson((HOLDING * x + ENERGY * v * v) * density, x=x) + SHORTAGE * rate0 + OVERFLOW * ratec)
    signal = float(simpson(x * x * (v - V0) ** 2 * density, x=x))
    return {'matrix': info, 'spend': cost, 'cost': average, 'mean_inventory': float(simpson(x * density, x=x)), 'shortage_rate': float(rate0), 'overflow_rate': float(ratec), 'signal_per_spend': signal / cost if cost else 0.0, 'density': density}

def policy(theta, kind, n=4001):
    x = np.linspace(0.0, C, n)
    u = np.full_like(x, U0)
    if kind == 'baseline':
        v = np.full_like(x, V0)
    elif kind == 'candidate':
        v = np.ones_like(x)
    elif kind.startswith('gate_'):
        v = V0 + gate(x, float(kind[5:]))
    else:
        raise ValueError(kind)
    return stationary(theta, x, u, v)

def advantage(theta, n=4001):
    return policy(theta, 'baseline', n)['cost'] - policy(theta, 'candidate', n)['cost']

def explore():
    rows = []
    for gamma in [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.45]:
        theta = [1.0, 0.5, gamma]
        row = {'theta': theta, 'advantage': advantage(theta)}
        for kind in ['baseline', 'candidate', 'gate_1.0', 'gate_0.5', 'gate_0.25', 'gate_0.1']:
            d = policy(theta, kind)
            row[kind] = {k: v for k, v in d.items() if k not in ['matrix', 'density']}
        rows.append(row)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'exploration.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    print(json.dumps([{'gamma': r['theta'][2], 'advantage': r['advantage'], 'candidate_ratio': r['candidate']['signal_per_spend'], 'gate01_ratio': r['gate_0.1']['signal_per_spend']} for r in rows], indent=2))

def interval_checks():
    """Outward interval evaluation plus an explicit composite-Simpson remainder.

    The bounds below apply to A=.2, r in {.5,.45,.2}, x in [0,2].
    They establish a cost-order reversal and candidate information moments;
    arbitrary feedback statistics are checked separately by grid refinement.
    """
    from mpmath import iv
    iv.dps = 35
    n = 2000
    h = iv.mpf(2) / n
    fourth_bounds = [10875, 26874, 65904]

    def moments_and_cost(rate, v):
        sums = [iv.mpf(0), iv.mpf(0), iv.mpf(0)]
        rr = iv.mpf(str(rate))
        for i in range(n + 1):
            xx = h * i
            f = iv.exp((iv.mpf('.4') * xx - rr * xx * xx) / iv.mpf('.25'))
            w = 1 if i in [0, n] else 4 if i % 2 else 2
            for j in range(3):
                sums[j] += w * xx ** j * f
        zz = []
        for j in range(3):
            error = iv.mpf(2) / 180 * h ** 4 * fourth_bounds[j]
            zz.append(sums[j] * h / 3 + iv.mpf([-1, 1]) * error)
        end = iv.exp((iv.mpf('.8') - 4 * rr) / iv.mpf('.25'))
        cost = (iv.mpf('.2') * zz[1] + iv.mpf('.125') * (10 + end)) / zz[0]
        cost += iv.mpf('.2') * iv.mpf(str(v)) ** 2
        return (cost, zz[2] / zz[0])
    base, _ = moments_and_cost(0.5, 0)
    rows = []
    for gamma in [0.05, 0.3]:
        rate = 0.45 if gamma == 0.05 else 0.2
        candidate, second = moments_and_cost(rate, 1)
        diff = base - candidate

        def bounds(value):
            return [float(np.nextafter(float(value.a), -np.inf)), float(np.nextafter(float(value.b), np.inf))]
        rows.append({'gamma': gamma, 'advantage_enclosure': bounds(diff), 'candidate_second_moment_enclosure': bounds(second), 'baseline_cost_enclosure': bounds(base)})
    assert rows[0]['advantage_enclosure'][1] < -0.05
    assert rows[1]['advantage_enclosure'][0] > 0.05
    OUT.mkdir(parents=True, exist_ok=True)
    record = {'arithmetic': 'mpmath interval, 35 decimal digits', 'simpson_subintervals': n, 'fourth_derivative_bounds': fourth_bounds, 'scope': 'Constant-policy boundary costs and candidate information moments only', 'checks': rows}
    (OUT / 'interval_checks.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(json.dumps(record, indent=2))

def structural_study():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    widths = [2.0, 1.5, 1.0, 0.75, 0.5, 0.25, 0.1, 0.05, 0.025]
    rows = []
    for gamma in [0.05, 0.3]:
        theta = [1.0, 0.5, gamma]
        base = policy(theta, 'baseline', 16001)
        candidate = policy(theta, 'candidate', 16001)
        for width in widths:
            fine = policy(theta, f'gate_{width}', 16001)
            coarse = policy(theta, f'gate_{width}', 8001)
            rows.append({'gamma': gamma, 'width': width, 'gate_spend': fine['spend'], 'gate_information_ratio': fine['signal_per_spend'], 'support_lower_bound': (C - width) ** 2 / OMEGA, 'unrestricted_limit': C * C / OMEGA, 'candidate_information_ratio': candidate['signal_per_spend'], 'asymptotic_gate_over_ab': fine['signal_per_spend'] / candidate['signal_per_spend'], 'ratio_refinement_difference': abs(fine['signal_per_spend'] - coarse['signal_per_spend']), 'spend_refinement_difference': abs(fine['spend'] - coarse['spend']), 'advantage': base['cost'] - candidate['cost']})
    assert all((r['support_lower_bound'] - 1e-09 <= r['gate_information_ratio'] <= r['unrestricted_limit'] + 1e-09 for r in rows))
    assert max((r['ratio_refinement_difference'] for r in rows)) < 0.0001
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'structural_study.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), constrained_layout=True)
    for gamma, color in [(0.05, '#245e8e'), (0.3, '#ad4d32')]:
        rr = [r for r in rows if r['gamma'] == gamma]
        xx = [r['width'] for r in rr]
        axes[0].plot(xx, [r['gate_information_ratio'] for r in rr], 'o-', color=color, label=f'Probe, gamma={gamma}')
        axes[0].axhline(rr[0]['candidate_information_ratio'], color=color, linestyle=':', label=f'A/B, gamma={gamma}')
        axes[1].plot(xx, [r['gate_spend'] for r in rr], 'o-', color=color, label=f'gamma={gamma}')
    axes[0].axhline(C * C / OMEGA, color='black', linestyle='--', label='Capacity-based information reference')
    axes[0].set(xlabel='Probe support width h', ylabel='Signal squared per intervention expenditure')
    axes[1].set(xlabel='Probe support width h', ylabel='Probe intervention per physical time', yscale='log')
    for ax in axes:
        ax.invert_xaxis()
        ax.grid(alpha=0.2)
        ax.legend(fontsize=6)
    fig.savefig(OUT / 'inventory_geometry.png', dpi=160)
    plt.close(fig)
    print(json.dumps({'designs': len(rows), 'max_ratio_refinement_difference': max((r['ratio_refinement_difference'] for r in rows)), 'gate01_gains': [r['asymptotic_gate_over_ab'] for r in rows if r['width'] == 0.1]}))

def constant_cost(q, gamma, v):
    """Vectorized exact truncated-normal invariant cost for eta=1, u=1."""
    rate = q - gamma * v
    mu = (U0 - DEMAND) / rate
    sd = SIGMA / np.sqrt(2 * rate)
    aa, bb = (-mu / sd, (C - mu) / sd)
    phia = np.exp(-aa * aa / 2) / np.sqrt(2 * np.pi)
    phib = np.exp(-bb * bb / 2) / np.sqrt(2 * np.pi)
    mass = np.where(aa > 0, ndtr(-aa) - ndtr(-bb), ndtr(bb) - ndtr(aa))
    mean = mu + sd * (phia - phib) / mass
    rate0 = SIGMA ** 2 * phia / (2 * sd * mass)
    ratec = SIGMA ** 2 * phib / (2 * sd * mass)
    return HOLDING * mean + ENERGY * v * v + SHORTAGE * rate0 + OVERFLOW * ratec

def mesh_frontiers():
    from policy_certification import quadratic_rows
    all_rows = []
    kinds = ['baseline', 'candidate', 'gate_1.0', 'gate_0.5', 'gate_0.25', 'gate_0.1']
    for step in [0.002, 0.001, 0.0005, 0.00025]:
        low, high = (np.array([0.45, 0.01]), np.array([0.6, 0.4]))
        nn = np.ceil((high - low) / step).astype(int)
        width = (high - low) / nn
        axes = [low[i] + (np.arange(nn[i]) + 0.5) * width[i] for i in range(2)]
        qq, gg = np.meshgrid(*axes, indexing='ij')
        points = np.column_stack([qq.ravel(), gg.ravel()])
        j0 = constant_cost(points[:, 0], points[:, 1], 0)
        j1 = constant_cost(points[:, 0], points[:, 1], 1)
        osc0 = C * C / SIGMA ** 2 * width[0] / 2
        osc1 = C * C / SIGMA ** 2 * sum(width) / 2
        dlower = np.exp(-osc0) * j0 - np.exp(osc1) * j1
        dupper = np.exp(osc0) * j0 - np.exp(-osc1) * j1
        dd = j0 - j1
        radius = np.linalg.norm(width / 2)
        for gamma in [0.05, 0.3]:
            theta3 = np.array([1.0, 0.5, gamma])
            theta2 = theta3[1:]
            stats = [policy(theta3, kind, 16001) for kind in kinds]
            mats = np.array([s['matrix'][1:, 1:] for s in stats])
            spend = np.array([s['spend'] for s in stats])
            sign = 1 if advantage(theta3) > 0 else -1
            possible = dlower <= -0.05 if sign > 0 else dupper >= 0.05
            verified = dd <= -0.05 if sign > 0 else dd >= 0.05
            lowerrows = quadratic_rows(theta2, mats, points[possible], radius)
            upperrows = quadratic_rows(theta2, mats, points[verified])

            def solve(rows, budget, indices):
                rr = rows[:, indices]
                cc = spend[indices]
                cuts = list(np.unique(np.argmin(rr, axis=0)))
                for _ in range(200):
                    aub = np.vstack([np.column_stack([-rr[cuts] / budget, np.ones(len(cuts))]), np.r_[cc / budget, 0.0]])
                    res = linprog(np.r_[np.zeros(len(indices)), -1.0], A_ub=aub, b_ub=np.r_[np.zeros(len(cuts)), 1.0], A_eq=np.r_[np.ones(len(indices)), 0.0][None], b_eq=[1.0], bounds=[(0.0, None)] * len(indices) + [(None, None)], method='highs', options={'primal_feasibility_tolerance': 1e-10, 'dual_feasibility_tolerance': 1e-10})
                    if not res.success:
                        raise RuntimeError(res.message)
                    vals = rr @ res.x[:-1]
                    worst = int(np.argmin(vals))
                    if vals[worst] >= budget * res.x[-1] - 1e-12:
                        weights = np.maximum(res.x[:-1], 0.0)
                        weights /= weights.sum()
                        spending = float(cc @ weights)
                        if spending > budget:
                            weights[1:] *= budget / spending
                            weights[0] = 1.0 - weights[1:].sum()
                        primal = max(0.0, float(np.min(rr @ weights)))
                        rho = np.maximum(-res.ineqlin.marginals[:-1], 0.0)
                        rho /= rho.sum()
                        eta = max(0.0, float(-res.ineqlin.marginals[-1]))
                        dual = max(0.0, float(eta * budget + np.max(rho @ rr[cuts] - eta * cc)))
                        assert dual >= primal - 1e-10
                        return (primal, weights.tolist(), dual)
                    cuts.append(worst)
                raise RuntimeError('Inventory cutting planes did not converge')
            for budget in [1e-05, 0.0001, 0.001, 0.01]:
                flo, fw, _ = solve(lowerrows, budget, list(range(len(kinds))))
                _, _, fhi = solve(upperrows, budget, list(range(len(kinds))))
                alo, _, _ = solve(lowerrows, budget, [0, 1])
                _, _, ahi = solve(upperrows, budget, [0, 1])
                assert flo <= fhi + 1e-09 and alo <= ahi + 1e-09
                row = {'gamma': gamma, 'budget': budget, 'mesh_step': step, 'cells': len(points), 'full_lower': flo, 'full_upper': fhi, 'ab_lower': alo, 'ab_upper': ahi, 'full_lower_weights': fw, 'cell_corrected_information_gain': flo / ahi if ahi else None}
                all_rows.append(row)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'mesh_frontiers.json').write_text(json.dumps(all_rows, indent=2), encoding='utf-8')
    selected = [r for r in all_rows if r['mesh_step'] == 0.00025]
    print(json.dumps(selected, indent=2))
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['explore', 'interval', 'structure', 'frontier'])
    args = p.parse_args()
    {'explore': explore, 'interval': interval_checks, 'structure': structural_study, 'frontier': mesh_frontiers}[args.command]()
