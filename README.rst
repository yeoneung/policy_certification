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
in ``experiments/allocation_protocol.json`` and generated ``jobs.json``.
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

    python summarize_results.py --input experiments/outputs/allocation_study

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

Inventory-triggered controls
---------------------------

Compare binary state-triggered preservation with controller-block allocation::

    python experiments/threshold_study.py prepare
    python experiments/threshold_study.py preflight
    python experiments/threshold_study.py smoke
    python experiments/threshold_study.py run

Add the same threshold to the D-optimal feedback library, then summarize every
configured cell::

    python experiments/threshold_enlarged.py prepare
    python experiments/threshold_enlarged.py run
    python experiments/summarize_threshold_study.py

The two JSON protocols specify the planned comparison family and the exploratory
feedback-class ablation. Run records, summaries and comparisons stay
under ``experiments/outputs/threshold_v5``. Existing matching records are reused.

Evaluate probes whose support scales with the resource budget and verify
the inventory cost-boundary derivatives with outward interval quadrature::

    python experiments/budget_width_study.py
    python experiments/verify_inventory_boundary.py

Finite-budget threshold calibration
-----------------------------------

Solve the stationary spending equation at the configured budgets, check its
derivative and integral against independent numerical calculations, and
compare exact threshold widths with the small-budget linear approximation::

    python experiments/finite_budget_calibration.py

This command runs without simulation records. It writes ``results.json`` and
``calibration.csv`` under
``experiments/outputs/inventory/finite_budget_calibration/``.
If ``experiments/outputs/threshold_v5/study_summary.csv`` is available, the
JSON also reports primary stopping times in reference reversion periods.
The reference baseline decay rate is 0.5, so one reference period equals
two model time units. To use a summary at another location::

    python experiments/finite_budget_calibration.py --summary path/to/study_summary.csv

The width comparison evaluates a uniform smooth-probe asymptotic coefficient
and the exact binary-threshold spending equation as separate constructions.
The generated files include spending residuals, derivative checks, and
small-budget convergence diagnostics. They also separate the local
boundary-density approximation, the integrated baseline-tail width, and
the threshold width under intervention.

Decision horizons and cumulative intervention
--------------------------------------------

After generating the threshold and allocation study summaries, compare
budget intensity, stopping time, and cumulative squared preservation effort::

    python experiments/summarize_inventory_planning.py

The command writes ``planning_summary.json``, ``budget_tradeoffs.csv``, and
``confidence_tradeoffs.csv`` under
``experiments/outputs/inventory/decision_horizon/``. It reports every configured
budget for the threshold rule and every budget/confidence combination for
the width-one smooth probe, retaining censoring and error counts. Time and
its Monte Carlo standard error are expressed in reference reversion periods;
cumulative effort retains its original model units. Comparisons are descriptive
and use the fixed decision margin in the source experiments.

To analyze summaries stored elsewhere::

    python experiments/summarize_inventory_planning.py --threshold-summary path/to/threshold.csv --allocation-summary path/to/allocation.csv --output path/to/analysis

The JSON records input hashes and the reference rate used for time conversion.
This command reads cell summaries and does not simulate additional paths.
