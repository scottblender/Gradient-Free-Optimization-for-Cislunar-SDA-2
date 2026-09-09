# Gradient-Free Optimization for Cislunar SDA

MATLAB tools for estimation-driven design of cislunar observer constellations in the Earth-Moon CR3BP. The fixed target cases are Lunar Gateway tracking, a low-thrust cislunar transfer, and a 10 m/s prograde impulse applied at Gateway perilune and propagated for 1.5 TU.

## Project layout

| Location | Purpose |
| --- | --- |
| `run_opt.m` | Main optimization entry point |
| `launch_optimization_gui.m` | GUI launcher and run monitor |
| `setup_project.m` | Adds source, scripts, and tests to the MATLAB path |
| `src/orbitDynamics/` | CR3BP dynamics, Jacobian, Jacobi constant, and Sun model |
| `src/estimation/` | Extended Kalman filter |
| `src/measurements/` | AO/AR measurement models and Jacobians |
| `src/constraints/` | Unified visibility/keep-out model and regression references |
| `src/targetGeneration/` | Gateway, low-thrust, and impulse truth generation |
| `src/optimization/` | Objective and gradient-free optimizers |
| `scripts/batch/` | Final PowerShell study launchers |
| `scripts/` | Catalog generation and Reviewer 2 result processing |
| `tests/` | Scientific/configuration regression tests |
| `data/` | Local catalog, target database, and caches |
| `results/` | Generated optimization runs and analyses |

## Setup

Open the repository root in MATLAB and run:

```matlab
setup_project;
test_project_structure;
```

The observer catalog is expected at `data/JPL_CR3BP_OrbitCatalog.mat`. Generated data and result folders are ignored by Git.

## Visibility convention

`calc_visibility` uses one center-referenced angular framework for Earth, Moon, and Sun. For each body,

```text
theta_keepout = max(theta_occultation, theta_exclusion).
```

Thus, a zero exclusion threshold recovers physical occultation. Optimization uses `calc_visibility`; the separate occlusion/exclusion functions remain as regression references.

## Reviewer 2 function-evaluation studies

Every optimizer is compared using cumulative objective-function evaluations (FE), not iterations.

### 1. Focused five-method runtime study

- GA, PSO, Bayesian optimization, ABC, ACO
- Lunar Gateway, AO, 3 observers, 1 period
- 1200 admitted FE
- optimizer seeds 0--19
- 100 runs total

Purpose: quantify Bayesian surrogate/runtime scaling at an equal FE budget and determine whether the extra runtime produces a solution-quality or convergence benefit.

Launcher:

```powershell
.\scripts\batch\run_runtime_comparison_1200_soo.ps1
```

### 2. Full optimizer comparison

- GA, PSO, ABC, ACO
- Lunar Gateway, low-thrust transfer, Gateway impulse
- AO, 3 observers, 1 Gateway period
- screening ON, `J1+J2+J3`
- 6000 admitted FE
- optimizer seeds 0--19
- 240 runs total

Launcher:

```powershell
.\scripts\batch\run_comparison_soo.ps1
```

### 3. GA baseline sensitivity

- GA, 6000 FE, seeds 0--19
- AO and AR
- 3, 5, 7, and 10 observers
- Gateway: 1, 3, and 5 periods
- low-thrust and impulse: fixed trajectory duration
- screening ON, `J1+J2+J3`
- 40 configurations / 800 runs total

Launcher:

```powershell
.\scripts\batch\run_baseline_soo.ps1
```

### 4. GA objective/screening sensitivity

- GA, 6000 FE, seeds 0--19
- AO, 3 observers, 1 Gateway period
- all three target cases
- five configurations:
  - `J111`, screening ON
  - `J111`, screening OFF
  - `J100`, screening ON
  - `J010`, screening ON
  - `J001`, screening ON
- 300 runs total

Launcher:

```powershell
.\scripts\batch\run_ga_objective_screening_soo.ps1
```

Total objective values from `J111`, `J100`, `J010`, and `J001` are different mathematical objectives and are not compared directly. Objective-component studies are interpreted through physical metrics such as RMSE, effective covariance sigma, stability, coverage, and orbit-family selection. The matched `J111` screening ON/OFF pair may be compared directly.

## Final Reviewer 2 processing

The four final processors validate the complete run factorials and write numeric/formatted CSV outputs:

```matlab
run_reviewer2_runtime_pipeline
run_reviewer2_comparison_pipeline
run_reviewer2_baseline_pipeline
run_reviewer2_objective_screening_pipeline
```

For the final paper, use the central runner:

```matlab
setup_project;
reports = run_reviewer2_results;
```

Or process only selected studies:

```matlab
reports = run_reviewer2_results("objective_screening");
reports = run_reviewer2_results(["runtime","comparison"]);
```

The central runner executes each scientific processor in data-only mode and then calls `make_reviewer2_paper_figures`. Curated manuscript figures are written under the newest

```text
results/<study>/FE_DATA_*/paper_final/
```

directory. Each study also receives `paper_figure_manifest.csv` describing the intended role of each figure.

### Figure conventions

Final Reviewer 2 figures use:

- Times New Roman;
- 12 pt minimum axis, tick, annotation, and legend text;
- 14 pt axis labels;
- consistent optimizer, measurement-model, and target-case colors;
- FE-aligned mean best-so-far convergence with sample-standard-deviation bands;
- compact 3-D geometry grids with common per-mission axis limits and camera;
- solid observer-orbit lines, with duplicate periodic orbits drawn once and all observer phase markers retained;
- no Earth in lunar-region result geometry plots;
- low-thrust departure/arrival periodic orbits and transfer start/end markers for context.

The final paper figure set is intentionally curated to support the conclusions rather than reproduce every intermediate diagnostic:

- **runtime:** equal-FE objective, runtime, BO slowdown, and convergence;
- **comparison:** overall/case optimizer metrics, convergence, rankings, and best-solution geometry;
- **baseline:** AO/AR, observer-count, tracking-duration, convergence, and geometry trends;
- **objective/screening:** matched screening sensitivity, objective-component physical metrics, family selection, convergence, and geometry changes.

## Recommended regression checks

Before processing final results:

```matlab
setup_project;
test_project_structure;
test_results_processing;
test_fe_study_configuration;
test_runtime_pipeline_configuration;
test_comparison_pipeline_configuration;
test_baseline_pipeline_configuration;
test_ga_objective_screening_configuration;
test_reviewer2_paper_figures_configuration;
test_visibility_keepout_definition;
test_low_thrust_transfer_case;
test_gateway_impulse_case;
```

## Data and outputs

| Content | Default location |
| --- | --- |
| Observer orbit catalog | `data/JPL_CR3BP_OrbitCatalog.mat` |
| Fixed target cases | `data/TargetCaseDatabase.mat` |
| Raw JPL CSV files | `data/JPL_Data/` |
| Orbit cache | `data/cache/orbits/` |
| Transfer cache | `data/cache/transfers/` |
| Runtime study | `results/RUNTIME_COMPARISON_1200/` |
| Full comparison | `results/COMPARISON/` |
| Baseline | `results/BASELINE/` |
| Objective/screening | `results/GA_OBJECTIVE_SCREENING/` |
| Final paper figures | `results/<study>/FE_DATA_*/paper_final/` |

Historical runs must be interpreted using the mission, visibility, noise, slot-definition, and stopping settings with which they were generated. Do not mix runs generated under different scientific configurations.
