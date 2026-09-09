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
| `results/` | Raw optimization studies |
| `COMPILED_REVIEWER_2_RESULTS/` | Timestamped processed Reviewer-2 tables and manuscript figures |

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

Total objective values from `J111`, `J100`, `J010`, and `J001` are different mathematical objectives and are not compared directly. Objective-component studies are interpreted primarily through tabulated RMSE, effective covariance sigma, stability, and selected orbit-family distributions. The matched `J111` screening ON/OFF pair may be compared directly.

## Final Reviewer 2 processing

The scientific processors validate the complete run factorials and produce aggregate data:

```matlab
run_reviewer2_runtime_pipeline
run_reviewer2_comparison_pipeline
run_reviewer2_baseline_pipeline
run_reviewer2_objective_screening_pipeline
```

For the paper, use the central runner:

```matlab
setup_project;
reports = run_reviewer2_results;
```

Or process selected studies:

```matlab
reports = run_reviewer2_results("comparison");
reports = run_reviewer2_results(["runtime","comparison"]);
```

`run_reviewer2_results` executes the scientific processors with historical previews hidden, moves each new analysis out of the raw-study tree, and calls `make_reviewer2_final_figures`, which routes through the curated manuscript renderer. Final CSVs, convergence MAT files, EPS figures, PNG figures, and the figure manifest are saved directly under:

```text
COMPILED_REVIEWER_2_RESULTS/runtime_1200_<timestamp>/
COMPILED_REVIEWER_2_RESULTS/comparison_<timestamp>/
COMPILED_REVIEWER_2_RESULTS/baseline_<timestamp>/
COMPILED_REVIEWER_2_RESULTS/objective_screening_<timestamp>/
```

The raw optimization runs remain under `results/RUNTIME_COMPARISON_1200/`, `results/COMPARISON/`, `results/BASELINE/`, and `results/GA_OBJECTIVE_SCREENING/`.

### Final figure conventions

Final Reviewer 2 figures use:

- Times New Roman;
- 12 pt minimum axis, tick, annotation, and legend text;
- 14 pt axis labels;
- one standalone metric figure per EPS/PNG so subfigures can be assembled in LaTeX;
- directly overlaid comparable convergence curves on one axes;
- no filled convergence uncertainty bands; sample standard deviation is shown only at the final FE point;
- no grid lines and no surrounding axes box;
- 20-run mean +/- sample standard deviation for quantitative comparisons;
- objective/cost comparison bars with the matched long-run AO GA baseline shown as a dashed reference;
- the same 7.6 x 7.0 inch centered 3-D layout used by the introductory tracking-case figures;
- solid observer-orbit lines, duplicate periodic orbits drawn once, no Earth, and low-thrust endpoint-orbit context.

The curated paper set intentionally omits redundant plots. In particular, 6000-FE optimization runtime remains in numerical tables rather than being repeated as a bar figure, and coverage-fraction figures are omitted. The focused 1200-FE runtime figure is retained because computational cost is the scientific purpose of that study.

The final paper figure set emphasizes:

- **runtime:** equal-1200-FE objective, runtime, and convergence;
- **comparison:** objective relative to the matched AO GA baseline, RMSE, effective uncertainty, stability, convergence, five-family orbit-selection summary, and representative optimizer geometry;
- **baseline:** AO/AR observer-count and Gateway-duration objective/RMSE/uncertainty trends, convergence, five-family orbit-selection summary, and representative 3/5/7/10-observer geometry;
- **screening ON/OFF:** convergence plus RMSE, effective sigma, stability, and rejected-measurement counts across all three target cases;
- **objective components:** numerical results in tables plus three mission-specific five-family selection figures with `Combined`, `J_1`, `J_2`, and `J_3` bars.

Trajectory figures are intentionally produced only for the **comparison** and **baseline** studies because those are the geometry comparisons used in the paper. The objective/screening study does not emit trajectory figures.

Orbit-family selection figures use all five manuscript families: `NHO`, `SHO`, `NNRHO`, `SNRHO`, and `DRO`.

## Baseline local Monte Carlo validation

The local baseline validation from the earlier manuscript is reproduced by a separate runner:

```matlab
mcReport = run_reviewer2_baseline_monte_carlo;
```

The no-argument default now reproduces both baseline AO Monte Carlo validation sets used in the manuscript:

- low-thrust transfer for 3, 5, 7, and 10 observers;
- Lunar Gateway for 3, 5, 7, and 10 observers over 1, 3, and 5 periods.

This gives 16 cases total. At the default 250 samples per case, the full Monte Carlo validation performs 4000 objective evaluations. For each configuration it selects the best observed 6000-FE GA baseline realization as the local reference. The first Monte Carlo sample is exactly the optimized design; orbit indices are perturbed uniformly by up to +/-10 orbit IDs and slot indices by up to +/-5 slot IDs. Measurement noise remains fixed by the saved baseline configuration so the validation isolates the local orbit/slot design neighborhood.

Outputs are saved to:

```text
COMPILED_REVIEWER_2_RESULTS/baseline_monte_carlo_<timestamp>/
```

including the sample/summary tables and separate EPS/PNG box-and-whisker figures such as:

```text
baseline_monte_carlo_samples.csv
baseline_monte_carlo_summary.csv
baseline_monte_carlo_figures.csv
baseline_mc_lt_ao_o3.eps/.png
baseline_mc_lt_ao_o5.eps/.png
baseline_mc_lt_ao_o7.eps/.png
baseline_mc_lt_ao_o10.eps/.png
baseline_mc_lg_ao_o3_p1.eps/.png
baseline_mc_lg_ao_o3_p3.eps/.png
baseline_mc_lg_ao_o3_p5.eps/.png
...
baseline_mc_lg_ao_o10_p5.eps/.png
```

Each case is exported as a separate box-and-whisker figure for LaTeX assembly. The red horizontal line is the optimized reference objective, and the legend identifies the Monte Carlo distribution and optimized reference. Monte Carlo figures follow the same no-grid/no-axes-box convention as the other final figures.

To run only the Lunar Gateway cases without repeating the low-thrust validation:

```matlab
mcLG = run_reviewer2_baseline_monte_carlo( ...
    'Mission',"LUNAR_GATEWAY", ...
    'GatewayPeriods',[1 3 5]);
```

`GATEWAY_IMPULSE` remains available through the same `Mission` option but is intentionally not part of the default Monte Carlo validation set.

The Monte Carlo reference design is intentionally seed-specific because the purpose is local-neighborhood validation of one optimized discrete solution. This does not replace the 20-run mean +/- sample-standard-deviation statistics used for optimizer and baseline performance claims.

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
test_reviewer2_legend_configuration;
test_baseline_monte_carlo_configuration;
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
| Raw runtime study | `results/RUNTIME_COMPARISON_1200/` |
| Raw full comparison | `results/COMPARISON/` |
| Raw baseline | `results/BASELINE/` |
| Raw objective/screening | `results/GA_OBJECTIVE_SCREENING/` |
| Processed final studies | `COMPILED_REVIEWER_2_RESULTS/<study>_<timestamp>/` |
| Baseline Monte Carlo | `COMPILED_REVIEWER_2_RESULTS/baseline_monte_carlo_<timestamp>/` |

Historical runs must be interpreted using the mission, visibility, noise, slot-definition, and stopping settings with which they were generated. Do not mix runs generated under different scientific configurations.
