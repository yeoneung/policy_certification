"""Command-line runner for numerical experiments and local result analysis."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
EXPERIMENTS=ROOT/'experiments'


def run(script,*arguments):
    subprocess.run([sys.executable,str(EXPERIMENTS/script),*map(str,arguments)],cwd=ROOT,check=True)


def summarize(directory):
    subprocess.run([sys.executable,str(ROOT/'summarize_results.py'),'--input',str(directory)],cwd=ROOT,check=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--suite',choices=['smoke','bilinear','inventory','extended','deterministic'],default='smoke')
    parser.add_argument('--rerun',action='store_true',help='Regenerate the fixed extended-study trajectories')
    args=parser.parse_args()
    if args.suite=='smoke':
        run('policy_certification.py','validate')
        run('followup_v4.py','prepare')
        run('validate_followup_v4.py','preflight')
        run('followup_v4.py','smoke')
    elif args.suite=='extended':
        run('followup_v4.py','prepare')
        run('validate_followup_v4.py','preflight')
        for model in ['bilinear','inventory']:
            run('followup_v4.py','run','--model',model,*(['--rerun'] if args.rerun else []))
        run('validate_followup_v4.py','results')
        summarize(EXPERIMENTS/'outputs/followup_v4')
    elif args.suite=='inventory':
        for group in ['primary','sensitivity']:
            run('inventory_run_study.py','--group',group,*(['--rerun'] if args.rerun else []))
        summarize(EXPERIMENTS/'outputs/inventory/sequential')
    elif args.suite=='bilinear':
        for label,theta,seed in [('positive',[1.,.2],527),('negative',[-.3,.2],528)]:
            for method in ['balanced','trace','policy','oracle']:
                run('policy_certification.py','simulate','--theta',*theta,'--method',method,
                    '--reps',100,'--dt',.02,'--max-time',3000,'--seed',seed,
                    '--output',f'{label}_{method}.json')
            for method in ['balanced','policy']:
                run('policy_certification.py','simulate','--theta',*theta,'--method',method,
                    '--reps',100,'--dt',.01,'--max-time',3000,'--seed',seed+1000,
                    '--output',f'{label}_{method}_dt001.json')
        summarize(EXPERIMENTS/'outputs')
    else:
        run('policy_certification.py','validate')
        run('policy_certification.py','frontier')
        for command in ['interval','structure','frontier']:
            run('inventory_frontier.py',command)
        run('finite_budget_dual.py')
        run('verify_finite_duals.py')


if __name__=='__main__':main()
