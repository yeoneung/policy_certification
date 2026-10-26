Controlled diffusion experiments
===============================

Python code for synthetic bilinear and inventory diffusion experiments:

* Stationary distributions, numerical quadrature, and information matrices.
* Resource-constrained policy-directed, D-optimal, and baseline allocations.
* Sequential stopping with Gaussian likelihood mixtures.
* Cumulative resource guards and deterministic allocation of controller time.
* Sensitivity experiments for confidence, resource budget, probe width,
  initial conditions, and integration step size.
* Polynomial optimization and outward-rounded numerical checks.

Installation
------------

Tested with Python 3.12.10. From this repository directory::

    python -m venv .venv

Activate the environment (``.venv\Scripts\Activate.ps1`` on Windows, or
``source .venv/bin/activate`` on a POSIX shell), then install dependencies::

    python -m pip install -r requirements.txt

Use that environment for the commands below.
The numerical simulator uses Numba; initial execution includes compilation.
``NUMBA_NUM_THREADS`` defaults to four and may be set before execution.

Quick checks
------------

Run numerical validation, build the design grids, and execute short simulations::

    python run_experiments.py --suite smoke

The short inventory simulation can reach its administrative horizon without
returning a decision. That is retained as a censored run, not a failed check.

Experiment suites
-----------------

Run the bilinear comparator and inventory probe-width configurations::

    python run_experiments.py --suite extended

The exact parameter grids, seeds, horizons and repetition counts are recorded
in ``experiments/followup_protocol_20261025.json`` and generated ``jobs.json``.
Matching cached designs and existing run files are reused. Regenerate fixed
trajectory jobs with::

    python run_experiments.py --suite extended --rerun

Run the inventory baseline, D-optimal, policy-directed and known-parameter
reference configurations, including sensitivity settings::

    python run_experiments.py --suite inventory

Run the bilinear balanced/trace/policy/reference configurations::

    python run_experiments.py --suite bilinear

Run stationary design grids, interval quadrature and polynomial bounds::

    python run_experiments.py --suite deterministic

These are full computational runs, rather than the short smoke checks.
For a smaller individual inventory simulation::

    python experiments/inventory_sequential.py simulate --method policy --theta 0.5 0.3 --reps 5 --max-time 1000 --output local_trial.json

Outputs and analysis
--------------------

All generated trajectories, design caches and summaries stay under
``experiments/outputs/``. They are ignored by Git. No precomputed trajectory
files or comparative outcomes are included in this repository.

The suite runner writes JSON and CSV summaries. Summarize a generated run
directory separately with::

    python summarize_results.py --input experiments/outputs/followup_v4

Summaries retain errors and administrative censoring. They report restricted
mean stopping time, Monte Carlo standard error, energy use, and pointwise
binomial bounds. Applicable D-optimal/policy comparisons also include
independent bootstrap ratio intervals and Welch tests. The targeted bilinear
comparison family uses Holm adjustment; other comparisons are descriptive.
No common error probability is estimated by pooling heterogeneous cells.

Model and numerical scope
-------------------------

These experiments use synthetic projected-Euler observations with their
projection regulator recorded. Regression uses the actual applied control
after any predictable resource attenuation. The state and operating-cost
calculations approximate a continuous diffusion. There is no field dataset.
Ordinary floating-point stopping calculations are distinct from the interval
arithmetic used for the separate numerical bounds. A known-parameter sampling
reference is not an implementable learner or a finite-time lower bound.

The repository contains executable experiment code, configuration files and
these operating instructions. It has no precomputed results or document
generation workflow. No authentication credential is required to run it.
