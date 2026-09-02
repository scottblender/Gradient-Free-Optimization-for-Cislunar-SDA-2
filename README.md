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
print_observer_ics_from_experiment_summary
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

The reviewer-facing comparison uses a common 6000-FE budget, angles-only
measurements, three observers, all three cost terms, visibility screening, the
three fixed target cases, and optimizer seeds 0--19. Measurement noise uses the
fixed seed 1001 for every independent optimizer run. The default matrix contains
300 runs: 5 methods x 3 target cases x 20 seeds.

The GA baseline preserves the original sensitivity grid: both measurement
models, 3/5/7/10 observers, and 1/3/5 periods for the Gateway case. It adds the
low-thrust and Gateway-impulse cases and repeats every configuration for seeds
0--19 so baseline tables can report means and standard deviations.

```powershell
.\scripts\batch\run_baseline_soo.ps1
.\scripts\batch\run_comparison_soo.ps1
```

Both launchers accept `-MatlabExe`, `-EvalBudget`, and `-Seeds`. Completed
runs are skipped safely; an incomplete run must be inspected or moved instead
of being silently overwritten.

After the studies finish, aggregate the data and convergence histories with:

```matlab
paths = setup_project();

[comparisonSummary, comparisonInventory] = process_fe_convergence( ...
    fullfile(paths.runs,'COMPARISON'), ...
    "reviewer2_comparison_v1",0:19,6000,false);

[baselineSummary, baselineInventory] = process_fe_convergence( ...
    fullfile(paths.runs,'BASELINE'), ...
    "reviewer2_baseline_v1",0:19,6000,false,"GA");
```

Convergence histories are aligned by cumulative function evaluations, not
iterations. The processor retains missing early checkpoints as `NaN` rather
than inventing values, and saves aggregate data without saving figures.

## Data, caches, and outputs

| Content | Default location |
| --- | --- |
| Observer orbit catalog | `data/JPL_CR3BP_OrbitCatalog.mat` |
| Fixed target cases | `data/TargetCaseDatabase.mat` |
| Raw JPL CSV files | `data/JPL_Data/` |
| Interpolated orbit cache | `data/cache/orbits/` |
| Transfer truth cache | `data/cache/transfers/` |
| Individual optimization runs | `results/runs/<timestamp>/...` |
| PowerShell baseline study | `results/runs/BASELINE/runs_GA/...` |
| PowerShell comparison study | `results/runs/COMPARISON/runs_<algorithm>/...` |
| Orbit catalog figures | `results/database_figs/` |

An explicit `RUN_DIR` environment variable or GUI output-folder selection still
takes precedence over the default run location. Each run retains its existing
`data/`, `figs/`, and `logs/` subfolders.

Post-processing scripts search the new results locations and retain support for
legacy root-level results folders. To select a specific existing study, make
its `runs_GA` or `runs` folder the MATLAB current folder and call the corresponding
post-processing script by name after `setup_project`. The observer-IC utility's
`RUN_ROOT` is relative to `results/`, with a project-root fallback.

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
