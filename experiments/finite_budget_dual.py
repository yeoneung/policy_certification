"""Numerical polynomial optimization for resource and information bounds."""
from pathlib import Path
import argparse
import json
import numpy as np
from numpy.polynomial import Chebyshev
from scipy.optimize import brentq, linprog
from scipy import sparse
import inventory_frontier as inv

def advantage(q, g):
    return float(inv.constant_cost(q, g, 0) - inv.constant_cost(q, g, 1))

def value_interval(q, g):
    from mpmath import iv
    iv.dps = 35
    n = 2000
    h = iv.mpf(2) / n

    def cost(v):
        qi = iv.mpf([float(np.nextafter(q, -np.inf)), float(np.nextafter(q, np.inf))])
        gi = iv.mpf([float(np.nextafter(g, -np.inf)), float(np.nextafter(g, np.inf))])
        r = qi - gi * v
        sums = [iv.mpf(0), iv.mpf(0)]
        for i in range(n + 1):
            x = h * i
            f = iv.exp((iv.mpf('.4') * x - r * x * x) / iv.mpf('.25'))
            w = 1 if i in [0, n] else 4 if i % 2 else 2
            sums[0] += w * f
            sums[1] += w * x * f
        z = []
        for j, bound in enumerate([500000, 1200000]):
            error = iv.mpf(2) / 180 * h ** 4 * bound
            z.append(sums[j] * h / 3 + iv.mpf([-1, 1]) * error)
        end = iv.exp((iv.mpf('.8') - 4 * r) / iv.mpf('.25'))
        return iv.mpf('.2') * z[1] / z[0] + iv.mpf('.2') * v * v + iv.mpf('.125') * (10 + end) / z[0]
    d = cost(0) - cost(1)
    return [float(np.nextafter(float(d.a), -np.inf)), float(np.nextafter(float(d.b), np.inf))]

def solve(q, g, budget, degree=20, nstate=401, ncert=500001):
    sign = 1 if advantage(q, g) > 0 else -1
    alt = brentq(lambda z: advantage(q, z) + sign * 0.0501, 0.01, 0.4, xtol=1e-13)
    enclosure = value_interval(q, alt)
    assert enclosure[1] < -0.05 if sign > 0 else enclosure[0] > 0.05
    kappa = (g - alt) ** 2 / (2 * inv.SIGMA ** 2)
    x = np.linspace(0, inv.C, nstate)
    z = 2 * x / inv.C - 1
    factor = Chebyshev([0.5, 0, -0.5]) * inv.C ** 2 / 4
    derivatives = [factor * Chebyshev.basis(j) for j in range(degree)]
    fp = np.column_stack([p(z) for p in derivatives])
    fpp = np.column_stack([p.deriv()(z) * 2 / inv.C for p in derivatives])
    base = (0.2 - q * x)[:, None] * fp + 0.5 * inv.SIGMA ** 2 * fpp
    ident = sparse.eye(nstate, format='csr')
    zero = sparse.csr_matrix((nstate, nstate))
    a0 = sparse.hstack([sparse.csr_matrix(base), -np.ones((nstate, 1)), np.zeros((nstate, 1)), ident, ident], format='csr')
    du = np.linspace(-1, 1, 41)
    vv = np.linspace(0, 1, 41)
    au = sparse.hstack([sparse.csr_matrix((du[:, None, None] * fp[None, :, :]).reshape(-1, degree)), np.zeros((len(du) * nstate, 1)), np.repeat(-du ** 2, nstate)[:, None], sparse.vstack([-ident] * len(du), format='csr'), sparse.csr_matrix((len(du) * nstate, nstate))], format='csr')
    av = sparse.hstack([sparse.csr_matrix((g * vv[:, None, None] * x[None, :, None] * fp[None, :, :]).reshape(-1, degree)), np.zeros((len(vv) * nstate, 1)), np.repeat(-vv ** 2, nstate)[:, None], sparse.csr_matrix((len(vv) * nstate, nstate)), sparse.vstack([-ident] * len(vv), format='csr')], format='csr')
    aub = sparse.vstack([a0, au, av], format='csr')
    bub = np.r_[np.zeros(nstate * (1 + len(du))), (-kappa * vv[:, None] ** 2 * x[None, :] ** 2).ravel()]
    objective = np.zeros(degree + 2 + 2 * nstate)
    objective[degree] = 1 / budget
    objective[degree + 1] = 1.0
    bounds = [(None, None)] * len(objective)
    bounds[degree + 1] = (0.0, None)
    result = linprog(objective, A_ub=aub, b_ub=bub, bounds=bounds, method='highs', options={'primal_feasibility_tolerance': 1e-09, 'dual_feasibility_tolerance': 1e-09})
    if not result.success:
        raise RuntimeError(result.message)
    coeff = result.x[:degree]
    price = float(result.x[degree + 1])
    derivative = sum((c * p for c, p in zip(coeff, derivatives)), Chebyshev([0.0]))
    second = derivative.deriv() * 2 / inv.C
    third = second.deriv() * 2 / inv.C
    potential = derivative.integ() * inv.C / 2
    potential = potential - potential(-1.0)
    xx = np.linspace(0, inv.C, ncert)
    zz = 2 * xx / inv.C - 1
    up = derivative(zz)
    upp = second(zz)
    if price > 1e-14:
        udelta = np.clip(up / (2 * price), -1, 1)
        extra_u = udelta * up - price * udelta ** 2
    else:
        extra_u = np.abs(up)
    linear = g * xx * up
    quadratic = kappa * xx ** 2 - price
    extra_v = np.maximum(0.0, linear + quadratic)
    safe = np.where(quadratic < 0, quadratic, -1.0)
    vertex = -linear / (2 * safe)
    mask = (quadratic < 0) & (vertex > 0) & (vertex < 1)
    extra_v[mask] = np.maximum(extra_v[mask], linear[mask] * vertex[mask] + quadratic[mask] * vertex[mask] ** 2)
    values = (0.2 - q * xx) * up + 0.5 * inv.SIGMA ** 2 * upp + extra_u + extra_v
    norm1 = float(np.abs(derivative.coef).sum())
    norm2 = float(np.abs(second.coef).sum())
    norm3 = float(np.abs(third.coef).sum())
    maxdrift = max((abs(u - 0.8 - q * i + g * v * i) for u in [0, 2] for i in [0, 2] for v in [0, 1]))
    lipschitz = (q + abs(g)) * norm1 + maxdrift * norm2 + 0.5 * inv.SIGMA ** 2 * norm3 + 2 * kappa * inv.C
    correction = lipschitz * inv.C / (2 * (ncert - 1))
    zeta = float(values.max() + correction)
    minimum = float(potential(zz).min() - norm1 * inv.C / (2 * (ncert - 1)))
    span = float(potential(2 * 0.5 / inv.C - 1) - minimum)
    upper = zeta + price * budget
    assert upper > 0
    record = {'q': q, 'gamma': g, 'budget': budget, 'opposite_gamma': alt, 'opposite_advantage_interval': enclosure, 'price': price, 'zeta_grid': float(values.max()), 'zeta_corrected': zeta, 'derivative_lipschitz_bound': lipschitz, 'residual_correction': correction, 'initial_bias_bound': span, 'information_upper': upper, 'degree': degree, 'lp_state_nodes': nstate, 'verification_nodes': ncert, 'u_derivative_chebyshev_coefficients': derivative.coef.tolist(), 'p_chebyshev_coefficients': coeff.tolist(), 'action_scope': 'u in [0,2], v in [0,1]', 'arithmetic_scope': 'Analytic control maximization and derivative correction in real arithmetic; floating-point polynomial evaluation'}
    record['time_lower_bounds'] = {str(delta): max(0.0, (1 - 2 * delta) * np.log((1 - delta) / delta) - span) / upper for delta in [0.1, 0.05, 0.01, 0.001]}
    return record

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--degree', type=int, default=20)
    args = p.parse_args()
    rows = []
    for gamma in [0.05, 0.3]:
        for budget in [0.003, 0.01, 0.03]:
            row = solve(0.5, gamma, budget, args.degree)
            rows.append(row)
            print(json.dumps({k: row[k] for k in ['gamma', 'budget', 'information_upper', 'initial_bias_bound', 'residual_correction', 'time_lower_bounds']}), flush=True)
    inv.OUT.mkdir(parents=True, exist_ok=True)
    (inv.OUT / 'finite_budget_duals.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
if __name__ == '__main__':
    main()
