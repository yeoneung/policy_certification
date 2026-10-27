"""Summarize inventory decision times and cumulative intervention from CSVs.

This is descriptive postprocessing of the configured threshold and fixed-width
studies. It reads existing cell summaries and never generates trajectories.
Times and their MCSEs are multiplied by the reference decay rate; cumulative
effort retains its original model units. Decision margins are unchanged.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8') as source:
        return list(csv.DictReader(source))


def convert(row, reference_rate):
    theta = json.loads(row['theta'])
    result = dict(source_id=row['id'], group=row['group'],
                  q=float(theta[0]), gamma=float(theta[1]),
                  method=row['method'], budget=float(row['budget']),
                  delta=float(row['delta']), dt=float(row['dt']),
                  reps=int(row['reps']), horizon_model_time=float(row['horizon']),
                  rmst_model_time=float(row['rmst']),
                  time_mcse_model_time=float(row['mcse']),
                  rmst_periods=reference_rate*float(row['rmst']),
                  time_mcse_periods=reference_rate*float(row['mcse']),
                  mean_cumulative_effort=float(row['mean_energy']),
                  realized_effort_per_model_time=float(row['energy_per_time']),
                  censored=int(row['censored']), errors=int(row['errors']))
    if 'width' in row:
        result['width'] = float(row['width'])
    if not all(math.isfinite(v) for v in result.values() if isinstance(v, float)):
        raise ValueError(f"Nonfinite summary value in {row['id']}")
    return result


def unique(rows, **settings):
    selected = [row for row in rows if all(row[k] == v for k, v in settings.items())]
    if len(selected) != 1:
        raise ValueError(f'Expected one cell for {settings}; found {len(selected)}')
    return selected[0]


def contrast(first, second):
    # Administrative censoring changes the interpretation of both total effort
    # and stopping-time means. Report it for every contrast instead of dropping
    # censored records or silently interpreting capped observations as completion.
    if first['rmst_periods'] <= 0 or first['mean_cumulative_effort'] <= 0:
        raise ValueError('Positive reference time and cumulative effort required')
    return dict(from_id=first['source_id'], to_id=second['source_id'],
                from_budget=first['budget'], to_budget=second['budget'],
                from_delta=first['delta'], to_delta=second['delta'],
                time_reduction_percent=100*(1-second['rmst_periods']/first['rmst_periods']),
                cumulative_effort_increase_percent=100*(
                    second['mean_cumulative_effort']/first['mean_cumulative_effort']-1),
                all_paths_completed=first['censored'] == second['censored'] == 0,
                observed_errors=first['errors']+second['errors'])


def summarize(threshold_summary, allocation_summary, reference_rate=.5):
    if not math.isfinite(reference_rate) or reference_rate <= 0:
        raise ValueError('Reference rate must be positive and finite')
    threshold = [convert(r, reference_rate) for r in read_csv(threshold_summary)
                 if r['method'] == 'threshold' and r['group'] in {'primary', 'budget'}]
    confidence = [convert(r, reference_rate) for r in read_csv(allocation_summary)
                  if r['model'] == 'inventory' and r['group'] == 'factorial'
                  and r['method'] == 'budget_fill' and float(r['width']) == 1.]
    # Select all configured budgets and both primary truths, with no selection
    # based on the realized time, errors, or censoring.
    if len(threshold) != 6 or len(confidence) != 12:
        raise ValueError('Expected 6 threshold cells and 12 width-one cells')
    budget_contrasts, confidence_contrasts = [], []
    for gamma in [.05, .30]:
        for budget in [.003, .01, .03]:
            unique(threshold, q=.5, gamma=gamma, budget=budget, delta=.05, dt=.02)
            for delta in [.05, .01]:
                unique(confidence, q=.5, gamma=gamma, budget=budget, delta=delta, dt=.02)
        first = unique(threshold, gamma=gamma, budget=.01)
        second = unique(threshold, gamma=gamma, budget=.03)
        budget_contrasts.append(dict(gamma=gamma, **contrast(first, second)))
        for budget in [.003, .01, .03]:
            first = unique(confidence, gamma=gamma, budget=budget, delta=.01)
            second = unique(confidence, gamma=gamma, budget=budget, delta=.05)
            confidence_contrasts.append(dict(gamma=gamma, width=1., **contrast(first, second)))
    threshold.sort(key=lambda r: (r['budget'], r['gamma']))
    confidence.sort(key=lambda r: (r['budget'], r['gamma'], r['delta']))
    return dict(reference_rate=reference_rate,
                reference_period_in_model_time=1/reference_rate,
                interpretation='Descriptive comparisons of existing cell summaries; no extrapolation.',
                time_units='Reference reversion periods = reference_rate * model time.',
                effort_units='Original model units of integrated squared preservation effort.',
                decision_margin='Unchanged across the source studies; no margin sensitivity is estimated.',
                sources={name: dict(filename=Path(path).name,
                                    sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())
                         for name, path in [('threshold', threshold_summary),
                                            ('allocation', allocation_summary)]},
                budget_cells=threshold, confidence_cells=confidence,
                budget_contrasts=budget_contrasts, confidence_contrasts=confidence_contrasts)


def write_outputs(result, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output/'planning_summary.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    for name, key in [('budget_tradeoffs.csv', 'budget_cells'),
                      ('confidence_tradeoffs.csv', 'confidence_cells')]:
        rows = result[key]
        with (output/name).open('w', newline='', encoding='utf-8') as target:
            writer = csv.DictWriter(target, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threshold-summary', type=Path,
                        default=ROOT/'experiments/outputs/threshold_v5/study_summary.csv')
    parser.add_argument('--allocation-summary', type=Path,
                        default=ROOT/'experiments/outputs/allocation_study/study_summary.csv')
    parser.add_argument('--reference-rate', type=float, default=.5)
    parser.add_argument('--output', type=Path,
                        default=ROOT/'experiments/outputs/inventory/decision_horizon')
    args = parser.parse_args()
    result = summarize(args.threshold_summary, args.allocation_summary, args.reference_rate)
    write_outputs(result, args.output)
    print(json.dumps(dict(budget_cells=len(result['budget_cells']),
                          confidence_cells=len(result['confidence_cells']),
                          output=str(args.output))))


if __name__ == '__main__':
    main()
