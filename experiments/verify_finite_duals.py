"""Outward-rounded arithmetic checks for computed polynomial bounds."""
from fractions import Fraction as F
import json
from pathlib import Path
import numpy as np
from mpmath import iv
import finite_budget_dual as dual
OUT = dual.inv.OUT
INF = np.inf

def down(x):
    return np.nextafter(x, -INF)

def up(x):
    return np.nextafter(x, INF)

def add(a, b):
    return (down(a[0] + b[0]), up(a[1] + b[1]))

def neg(a):
    return (-a[1], -a[0])

def sub(a, b):
    return add(a, neg(b))

def mul(a, b):
    p = [a[i] * b[j] for i in [0, 1] for j in [0, 1]]
    return (down(np.minimum.reduce(p)), up(np.maximum.reduce(p)))

def point(x):
    return (np.asarray(x), np.asarray(x))

def enclose_fraction(x):
    v = float(x)
    return (float(down(v)), float(up(v)))

def scalar_fraction_upper(x):
    return enclose_fraction(x)[1]

def derivative(c):
    n = len(c) - 1
    if n == 0:
        return [F(0)]
    d = [F(0)] * n
    d[n - 1] = 2 * n * c[n]
    if n >= 2:
        d[n - 2] = 2 * (n - 1) * c[n - 1]
    for j in range(n - 3, -1, -1):
        d[j] = d[j + 2] + 2 * (j + 1) * c[j + 1]
    d[0] /= 2
    return d

def antiderivative(c):
    d = [F(0)] * (len(c) + 1)
    d[1] += c[0]
    if len(c) > 1:
        d[2] += c[1] / 4
    for j in range(2, len(c)):
        d[j + 1] += c[j] / (2 * (j + 1))
        d[j - 1] -= c[j] / (2 * (j - 1))
    d[0] -= sum((v * (-1) ** j for j, v in enumerate(d)))
    return d

def defining_polynomial(coeff):
    p = [F.from_float(c) for c in coeff]
    f = [F(0)] * (len(p) + 2)
    for j, c in enumerate(p):
        f[j] += c / 2
        f[j + 2] -= c / 4
        f[abs(j - 2)] -= c / 4
    assert sum(f) == 0 and sum((v * (-1) ** j for j, v in enumerate(f))) == 0
    return f

def evaluate(c, z):
    bounds = [enclose_fraction(v) for v in c]
    b1 = point(np.zeros_like(z[0]))
    b2 = point(np.zeros_like(z[0]))
    for j in range(len(c) - 1, 0, -1):
        b0 = add(sub(mul(mul(point(2.0), z), b1), b2), bounds[j])
        b2, b1 = (b1, b0)
    return add(sub(mul(z, b1), b2), bounds[0])

def iv_bounds(value):
    return (float(down(float(value.a))), float(up(float(value.b))))

def verify(row, nodes=200001):
    iv.dps = 40
    opposite = dual.value_interval(row['q'], row['opposite_gamma'])
    sign = 1 if dual.advantage(row['q'], row['gamma']) > 0 else -1
    assert opposite[1] < -0.05 if sign > 0 else opposite[0] > 0.05
    q, g, alt = [F.from_float(float(row[k])) for k in ['q', 'gamma', 'opposite_gamma']]
    eta = F.from_float(float(row['price']))
    budget = F.from_float(float(row['budget']))
    kappa = (g - alt) ** 2 / F(1, 2)
    f = defining_polynomial(row['p_chebyshev_coefficients'])
    f1 = derivative(f)
    f2 = derivative(f1)
    potential = antiderivative(f)
    norm = [sum((abs(c) for c in p)) for p in [f, f1, f2]]
    maxdrift = max((abs(F(u) - F(4, 5) - q * x + g * v * x) for u in [0, 2] for x in [0, 2] for v in [0, 1]))
    lipschitz = (q + abs(g)) * norm[0] + maxdrift * norm[1] + F(1, 8) * norm[2] + 4 * kappa
    x = np.linspace(0.0, 2.0, nodes)
    z = sub(point(x), point(1.0))
    fp = evaluate(f, z)
    fpp = evaluate(f1, z)
    drift = sub(enclose_fraction(F(1, 5)), mul(enclose_fraction(q), point(x)))
    base = add(mul(drift, fp), mul(point(0.125), fpp))
    a = np.maximum(np.abs(fp[0]), np.abs(fp[1]))
    etalo, etahi = enclose_fraction(eta)
    extra_u = np.where(a <= 2 * etalo, up(up(a * a) / down(4 * etalo)), up(a - etalo))
    linear = mul(mul(enclose_fraction(g), point(x)), fp)
    quadratic = sub(mul(enclose_fraction(kappa), mul(point(x), point(x))), enclose_fraction(eta))
    l = linear[1]
    c = quadratic[1]
    extra_v = np.maximum(0.0, up(l + c))
    deriv1 = add(point(l), mul(point(2.0), point(c)))
    mask = (c < 0) & (l > 0) & (deriv1[0] < 0)
    candidate = np.zeros_like(l)
    candidate[mask] = up(up(l[mask] * l[mask]) / down(-4 * c[mask]))
    extra_v = np.maximum(extra_v, candidate)
    value = add(add(base, point(extra_u)), point(extra_v))[1]
    spacing = up(float(np.diff(x).max()))
    residual = up(up(scalar_fraction_upper(lipschitz) * spacing) / 2)
    zeta = up(float(value.max()) + residual)
    xb = np.linspace(0.0, 2.0, 2001)
    zb = sub(point(xb), point(1.0))
    pvals = evaluate(potential, zb)
    bias_spacing = up(float(np.diff(xb).max()))
    minimum = down(float(pvals[0].min()) - up(scalar_fraction_upper(norm[0]) * bias_spacing / 2))
    start = evaluate(potential, point(-0.5))[1]
    span = float(up(start - minimum))
    information = float(up(zeta + scalar_fraction_upper(eta * budget)))
    lower = {}
    for delta in ['0.1', '0.05', '0.01', '0.001']:
        d = iv.mpf(delta)
        kl = (1 - 2 * d) * iv.log((1 - d) / d)
        numerator = max(0.0, float(down(iv_bounds(kl)[0] - span)))
        lower[delta] = float(down(numerator / information)) if numerator else 0.0
    result = {k: row[k] for k in ['q', 'gamma', 'budget', 'opposite_gamma', 'opposite_advantage_interval', 'price']}
    result['opposite_advantage_interval'] = opposite
    result.update(zeta_upper=zeta, information_upper=information, initial_bias_upper=span, time_lower_bounds=lower, residual_correction_upper=residual, verification_nodes=nodes, p_chebyshev_coefficients=row['p_chebyshev_coefficients'], exact_neumann_endpoints=True, arithmetic_scope='Exact binary-rational polynomial coefficients; outward-rounded interval Clenshaw and analytic action maxima; global derivative correction; interval cost ordering. No formal arithmetic verification.')
    assert information > 0 and all((v >= 0 for v in lower.values()))
    return result
if __name__ == '__main__':
    rows = json.loads((OUT / 'finite_budget_duals.json').read_text())
    verified = []
    for row in rows:
        v = verify(row)
        verified.append(v)
        print(json.dumps({k: v[k] for k in ['gamma', 'budget', 'information_upper', 'initial_bias_upper', 'time_lower_bounds']}), flush=True)
    (OUT / 'verified_finite_duals.json').write_text(json.dumps(verified, indent=2), encoding='utf-8')
