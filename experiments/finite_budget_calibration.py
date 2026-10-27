"""Stationary threshold calibration and dimensionless inventory time scales.

Run from any directory. Stationary checks require no simulation records.
An available threshold-study summary also supplies normalized stopping times.
Generated JSON and CSV files stay under experiments/outputs/.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

from threshold_study import segment_integral, threshold_cost

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/outputs/inventory/finite_budget_calibration'
C, SIGMA2 = 2.0, 0.25


def quadrature_cost(q, gamma, s):
    low = quad(lambda x: np.exp((.4*x-q*x*x)/SIGMA2), 0, s,
               epsabs=1e-13, epsrel=1e-13)[0]
    high = quad(lambda x: np.exp((.4*x-(q-gamma)*x*x-gamma*s*s)/SIGMA2),
                s, C, epsabs=1e-13, epsrel=1e-13)[0]
    return high/(low+high)


def cost_derivative(q, gamma, s):
    low = segment_integral(q, 0, s)
    high = np.exp(-gamma*s*s/SIGMA2)*segment_integral(q-gamma, s, C)
    density = np.exp((.4*s-q*s*s)/SIGMA2)/(low+high)
    cost = high/(low+high)
    return -density-2*gamma*s/SIGMA2*cost*(1-cost)


def main(summary_path=None):
    OUT.mkdir(parents=True, exist_ok=True)
    pstar = np.exp((.4*C-.6*C*C)/SIGMA2)/segment_integral(.6, 0, C)
    kappa = 1.1*35/(13*pstar)
    rows = []
    for budget in [.003, .01, .03]:
        for gamma in [.05, .30]:
            q = .5
            s = brentq(lambda s: threshold_cost((q, gamma), s)-.9*budget,
                       0, C, xtol=1e-13)
            p0C = np.exp((.4*C-q*C*C)/SIGMA2)/segment_integral(q, 0, C)
            baseline_s = brentq(lambda s: threshold_cost((q, 0), s)-.9*budget,
                                0, C, xtol=1e-13)
            rows.append(dict(budget=budget, q=q, gamma=gamma,
                             uniform_smooth_linear_width=kappa*budget,
                             baseline_boundary_linear_width=.9*budget/p0C,
                             baseline_tail_width=C-baseline_s,
                             threshold=s, threshold_width=C-s,
                             stationary_spending=threshold_cost((q, gamma), s),
                             quadrature_spending=quadrature_cost(q, gamma, s)))

    # Independent adaptive quadrature and finite differences check the
    # analytic integral and the differentiated stationary distribution.
    errors, derivative_errors, derivatives = [], [], []
    for q in [.45, .5, .6]:
        for gamma in [.01, .05, .30, .4]:
            for s in [.05, .4, 1., 1.6, 1.95]:
                errors.append(abs(threshold_cost((q, gamma), s)-quadrature_cost(q, gamma, s)))
                d = cost_derivative(q, gamma, s)
                h = 1e-5
                fd = (quadrature_cost(q, gamma, s+h)-quadrature_cost(q, gamma, s-h))/(2*h)
                derivative_errors.append(abs(d-fd))
                derivatives.append(d)
    residual = max(abs(row['quadrature_spending']-.9*row['budget']) for row in rows)
    assert max(errors) < 1e-12 and max(derivative_errors) < 1e-8
    assert max(derivatives) < 0 and residual < 1e-12

    small_budget = []
    for gamma in [.05, .3]:
        p0C = np.exp((.4*C-.5*C*C)/SIGMA2)/segment_integral(.5, 0, C)
        for budget in [1e-5, 1e-6, 1e-7]:
            s = brentq(lambda s: threshold_cost((.5, gamma), s)-.9*budget,
                       0, C, xtol=1e-14)
            small_budget.append(dict(gamma=gamma, budget=budget,
                                     width_over_budget=(C-s)/budget,
                                     limiting_width_over_budget=.9/p0C))

    times = []
    summary = summary_path or ROOT/'experiments/outputs/threshold_v5/study_summary.csv'
    if summary_path is not None and not summary.is_file():
        raise FileNotFoundError(summary)
    if summary.is_file():
        with summary.open(newline='', encoding='utf-8') as source:
            for row in csv.DictReader(source):
                if row['group'] == 'primary':
                    times.append(dict(method=row['method'], theta=json.loads(row['theta']),
                                      rmst=float(row['rmst']),
                                      reference_reversion_periods=.5*float(row['rmst'])))
    result = dict(capacity=C, sigma_squared=SIGMA2, minimum_baseline_density=pstar,
                  uniform_smooth_kappa=kappa, geometric_budget_ceiling=C/kappa,
                  rows=rows, small_budget_check=small_budget,
                  reference_reversion_time=2.0, primary_times=times,
                  checks=dict(passed=True, quadrature_cases=len(errors),
                              max_cost_quadrature_error=max(errors),
                              max_derivative_finite_difference_error=max(derivative_errors),
                              max_spending_residual=residual))
    (OUT/'results.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    with (OUT/'calibration.csv').open('w', newline='', encoding='utf-8') as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path,
                        help='Optional threshold-study CSV for stopping-time normalization')
    args = parser.parse_args()
    main(args.summary)
