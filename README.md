# Gradient-Free Optimization for Cislunar SDA

MATLAB tools for designing cislunar observer constellations using estimation
performance. The fixed target scenarios are Lunar Gateway tracking, a
low-thrust transfer, and a 10 m/s prograde impulse applied at Gateway perilune
and propagated for 1.5 TU.

## Project layout

The top-level layout follows the Lunar Surface Sensor Network Optimization
project. Source folders here are ordinary MATLAB folders, not `+` packages, so
existing function names and calls remain unchanged.

| Location | Purpose |
| --- | --- |
| `run_opt.m` | Main optimization entry point; retained at the project root |
| `launch_optimization_gui.m` | GUI launcher and run monitor; retained at the root |
| `setup_project.m` | Adds code to the MATLAB path and resolves project directories |
| `src/orbitDynamics/` | CR3BP dynamics, Jacobian, and Sun position |
| `src/estimation/` | Extended Kalman filter |
| `src/measurements/` | Measurement models and Jacobians |
| `src/constraints/` | Unified visibility and legacy screening functions used by tests |
| `src/targetGeneration/` | Target truth builders and the low-thrust transfer solver |
| `src/optimization/` | Objective, cost calculation, and custom optimizer implementations |
| `scripts/` | Catalog preparation, plotting, and result-processing scripts |
| `scripts/batch/` | Windows PowerShell launchers for baseline and comparison studies |
| `tests/` | Project-path smoke test and trajectory visibility tests |
| `data/` | Local orbit catalog, optional raw CSV inputs, and generated caches |
| `results/` | Generated run folders, figures, reports, and logs |

## Visibility convention

`calc_visibility` applies one center-referenced angular framework to the
Earth, Moon, and Sun. For each body, the configured exclusion angle is an
absolute minimum line-of-sight separation from the body center. The effective
keep-out boundary is

```text
theta_keepout = max(theta_occultation, theta_exclusion).
```

Setting the exclusion angle to zero therefore recovers physical occultation.
The separate `calc_occlusion` and `calc_exclusion` functions are retained
only as independent regression references; optimization uses
`calc_visibility`.

## Getting started

Open the project root in MATLAB, then run:

```matlab
setup_project;
test_project_structure;
```

The structure test checks function resolution without loading the orbit catalog
or running a transfer solve. `setup_project` does not change the working
directory or recursively add data/results folders to the MATLAB path.

Place your existing `JPL_CR3BP_OrbitCatalog.mat` in `data/`. A catalog already at
the project root is also supported; if both exist, the one in `data/` is used.
There is no need to regenerate an existing catalog just because files moved.

The workflow uses MATLAB, Global Optimization Toolbox, Optimization Toolbox,
Statistics and Machine Learning Toolbox, and Parallel Computing Toolbox.
The Windows batch launchers use MATLAB on `PATH` when available and otherwise
try MATLAB R2026a. Pass `-MatlabExe` to select another installation.

## Entry points

```matlab
% Main optimization script (starts an optimization run).
run_opt

% Alternatively, configure and launch a run from the GUI.
launch_optimization_gui

% Visibility regression on Gateway and a generated low-thrust transfer.
% This invokes the transfer solver, but no constellation optimizer.
test_visibility_trajectories

% Study-definition figures and FE result processing.
plot_study_definition_figures
process_fe_convergence
```

Run `setup_project` first when calling functions or scripts directly. The main
runner, GUI, and moved scripts also initialize their own project paths, so the
scripts can be opened and run from their new locations.

`build_observer_orbit_catalog` is only needed to rebuild the observer catalog from raw JPL CSV
files. It reads `data/JPL_Data/`, with support for an existing root-level
`JPL_Data/`, and writes the catalog into `data/`. Its filtering and orbit
ordering are unchanged by this reorganization.

## Function-evaluation studies

Every supported optimizer uses `MAX_EVALS` as its only search stopping
criterion. The supported methods are GA, PSO, Bayesian optimization, ABC, and
ACO. The GUI and batch launchers do not set or fall back to `MAX_ITERS`.

The reviewer-facing full comparison uses a common 6000-FE budget, angles-only
measurements, three observers, all three cost terms, visibility screening, the
three fixed target cases, and optimizer seeds 0--19. Measurement noise uses the
fixed seed 1001 for every independent optimizer run. The scalable-method matrix
contains 240 runs: 4 methods x 3 target cases x 20 seeds. Bayesian optimization
is evaluated separately in the focused 1200-FE runtime study (5 methods x 20
seeds for Lunar Gateway), where its equal-FE runtime and objective benefit can
be assessed without committing to an impractically long 6000-FE campaign.

The GA baseline preserves the original sensitivity grid: both measurement
models, 3/5/7/10 observers, and 1/3/5 periods for the Gateway case. It adds the
low-thrust and Gateway-impulse cases and repeats every configuration for seeds
0--19 so baseline tables can report means and standard deviations.

```powershell
.\scripts\batch\run_baseline_soo.ps1
.\scripts\batch\run_comparison_soo.ps1
.\scripts\batch\run_runtime_comparison_1200_soo.ps1
.\scripts\batch\run_ga_objective_screening_soo.ps1
```

All four launchers accept `-MatlabExe`, `-EvalBudget`, `-Seeds`, and `-Pilot`.
Pilot and full runs use the same parallel-optimization path. They show
completed-run progress in PowerShell. Completed runs are skipped
safely; an incomplete run must be inspected or moved instead of being silently
overwritten.

Before starting the full studies, run the fast configuration/regression checks:

```matlab
setup_project;
test_project_structure;
test_fe_study_configuration;
test_ga_evaluation_counts(120);
test_visibility_keepout_definition;
test_low_thrust_transfer_case;
test_gateway_impulse_case;
test_results_processing;
```

Then exercise the exact batch entry points with their small pilot matrices:

```powershell
.\scripts\batch\run_baseline_soo.ps1 -Pilot
.\scripts\batch\run_comparison_soo.ps1 -Pilot
```

If Windows blocks direct `.ps1` execution, use a process-scoped bypass without
changing the machine or user policy:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\batch\run_baseline_soo.ps1 -Pilot
```

Without overrides, the baseline pilot performs three 120-FE GA runs (one per
fixed target case), and the comparison pilot performs fifteen 1200-FE runs
(five optimizers by three target cases), all with optimizer seed 0. Pilot output
is isolated under `BASELINE_PILOT` and `COMPARISON_PILOT_1200`, so it cannot be
mistaken for the manuscript study.

For one end-to-end check of the comparison pipeline, including all five
optimizers, all three fixed cases, and seeds 0--2, run:

```matlab
report = run_reviewer2_pipeline_pilot;
```

This launches or resumes 45 parallel runs at 1200 FE, processes and validates
every saved result, prints mean +/- sample-standard-deviation tables, and
creates EPS/PNG convergence, objective, runtime, and cost-component previews.
For each target case, it also identifies the lowest-objective run across all
optimizers and seeds and plots its truth/estimate trajectory and integrated EKF
error, +/- 3-sigma, and available-observer diagnostics. The selected runs are
recorded in `pilot_best_observed_runs.csv`; figures are saved under the newest
`COMPARISON_PILOT_1200/FE_DATA_*/paper_preview` folder. To reprocess completed
runs without launching MATLAB workers again, use `run_reviewer2_pipeline_pilot(false,true)`. These are explicitly pilot
statistics; the full 20-seed study must replace them in the manuscript.

To regenerate only the centered, label-safe best-run trajectory panels from the
newest processed 1200-FE pilot, run:

```matlab
plot_reviewer2_best_trajectories;
```

Process the one-seed comparison pilot manually with:

```matlab
paths = setup_project();
process_fe_convergence( ...
    fullfile(paths.runs,'COMPARISON_PILOT_1200'), ...
    "reviewer2_comparison_pilot_1200_v1",0,1200,false);
```

After the studies finish, run the manuscript processors. Each validates the
complete factorial, writes numeric and formatted CSV tables, prints the key
contrasts, and exports Times New Roman EPS/PNG figures with 12-point minimum
text:

```matlab
reports = run_reviewer2_results; % process all four completed studies

% Or inspect selected studies independently:
comparisonReport = run_reviewer2_results("comparison");
selectedReports = run_reviewer2_results(["runtime","objective_screening"]);
```

Convergence histories are aligned by cumulative function evaluations, not
iterations, and the manuscript curves show the across-seed mean with sample
standard-deviation bands. The objective/screening processor compares total
objective only for the matched combined-objective ON/OFF pair; J1/J2/J3-only
cases are compared using physical tracking, uncertainty, stability, coverage,
and orbit-family outcomes.

## Data, caches, and outputs

| Content | Default location |
| --- | --- |
| Observer orbit catalog | `data/JPL_CR3BP_OrbitCatalog.mat` |
| Fixed target cases | `data/TargetCaseDatabase.mat` |
| Raw JPL CSV files | `data/JPL_Data/` |
| Interpolated orbit cache | `data/cache/orbits/` |
| Transfer truth cache | `data/cache/transfers/` |
| Individual optimization runs | `results/<study>/...` |
| PowerShell baseline pilot | `results/BASELINE_PILOT/runs_GA/...` |
| PowerShell comparison pilot | `results/COMPARISON_PILOT_1200/runs_<algorithm>/...` |
| PowerShell baseline study | `results/BASELINE/runs_GA/...` |
| PowerShell comparison study | `results/COMPARISON/runs_<algorithm>/...` |
| GA objective/screening study | `results/GA_OBJECTIVE_SCREENING/...` |
| Focused Bayesian runtime study | `results/RUNTIME_COMPARISON_1200/...` |
| Orbit catalog figures | `results/database_figs/` |

An explicit `RUN_DIR` environment variable or GUI output-folder selection still
takes precedence over the default run location. Each run retains its existing
`data/`, `figs/`, and `logs/` subfolders.

The schema-v2 processors read the named study roots above. Legacy Excel/FIG
post-processors were removed so older result formats cannot be mistaken for the
equal-FE reviewer study.

The data and results contents are ignored by Git. Updating the repository does
not move or delete your local catalog, old caches, or previous results. New
caches are built in `data/cache/`; old root-level caches are not loaded
automatically. This avoids silently reusing a database built with the previous
slot definition. Do not move pre-correction caches into the new cache folders.

Historical results should still be interpreted using the mission, visibility,
noise, and stopping settings with which they were generated.

Fixed Gateway, low-thrust, and Gateway-impulse target definitions are stored in
`data/TargetCaseDatabase.mat`. The loader creates this compact database from
`scripts/build_target_case_database.m` when it is missing. Fixed target cases
never resolve observer catalog rows or observer slots.
