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

For all manuscript figures, run the single root-level entry point:

```matlab
output = run_manuscript_figures;
% Inspect only the paired measurement diagrams (no catalog/results needed):
output = run_manuscript_figures("definitions",'DefinitionSections',"measurement");
% Regenerate selected results from cached processed reports:
output = run_manuscript_figures(["runtime","comparison"]);
% Reprocess raw runs after adding/changing results:
output = run_manuscript_figures("results",'Reprocess',true);
```

Selectors are `definitions`, `runtime`, `comparison`, `baseline`,
`objective_screening`, and `monte_carlo`; `all` includes all six. Definition
subsections are `catalog`, `slots`, `visibility`, `measurement`, and `cases`.
`Inspect=true` enables definition previews. Old runners remain callable for
backward compatibility; they are implementation helpers, not additional steps.
The first result request processes saved runs and caches reports; later requests
rerender those reports. Use `Reprocess=true` when the input results change.
Definition generation may propagate target trajectories; it does not run the
constellation optimizers. Monte Carlo selects the newest saved sample directory
(or `MonteCarloDirectory=folder`) and only replots it. Missing MC samples are
reported and skipped; new MC evaluations must be requested explicitly using the
validation runner described below.

Every invocation collects final EPS files, PNG previews, a figure manifest, and
`manuscript_tables.txt` / `manuscript_tables.tex` together in `MANUSCRIPT_OUTPUT/`.
The table printer uses this same folder when run separately. Matching output names
are overwritten on reruns; unselected figures are retained. Both entry points accept
`OutputDirectory` to use another shared folder. Per-study processing intermediates
and numerical source data retain their existing organization.

### Print all manuscript tables

```matlab
tables = print_manuscript_tables;
% Refresh summaries from saved raw runs when necessary (no new optimization):
tables = print_manuscript_tables('Reprocess',true);
% Pin a study to a particular processed directory for reproducibility:
tables = print_manuscript_tables('Directories',struct('baseline',folder));
```

The printer follows all 13 table labels in the supplied clean manuscript:
catalog parameter/geometric ranges, weights, algorithms, EKF/settings, target
ICs and LT reproduction, configurations, AO/AR baselines, 1200-FE and 6000-FE
comparisons, screening, and objective components. It prints copyable LaTeX rows,
mean +/- sample SD, and actual solver-call ranges in manuscript column order.
It reads processed CSVs and saved target/run data without running optimizations.
The four fixed descriptive/configuration tables are explicitly identified as
manuscript settings, stored in `scripts/templates/manuscript_configuration_tables.tex`;
update that template if the study settings change. Every data-derived table prints
its source path; missing data are reported per table. The target table reports
missing converged LT values separately and never substitutes the initial guess.

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
- 22 pt bold axes/legends, set when the plot is constructed;
- 24 pt bold axis labels;
- one standalone metric figure per EPS/PNG so subfigures can be assembled in LaTeX;
- directly overlaid comparable convergence curves on one axes;
- convergence figures show only the 20-run mean best-so-far curves; run-to-run variability is retained in the processed tables and metric figures;
- no grid lines and no surrounding axes box;
- 20-run mean +/- sample standard deviation for quantitative comparisons;
- objective/cost comparison bars with the matched long-run AO GA baseline shown as a dashed reference;
- matched canvas sizes within geometry, measurement, and metric panel groups;
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

### EPS placement and validation

EPS files use the restored September 10 print path and per-figure layouts.
Use equal LaTeX widths for paired panels. The runner copies completed outputs;
it does not enlarge fonts or reflow figures while exporting.

### Serial/parallel convergence and export cleanup

```matlab
setup_project;
results = test_parallel_speed(3); % Run serial + parallel LG GA, 6000 FE each
run_manuscript_figures("parallel"); % Export saved histories only
run_manuscript_figures(0);          % All manuscript figures; keep old exports
run_manuscript_figures(1);          % All manuscript figures; clear exports first
% Clear final exports and regenerate only the parallel comparison:
run_manuscript_figures("parallel",'ClearDirectory',1);
```

The benchmark saves completed runs, callback FE/objective/time histories, CSVs,
and a timing summary beneath `MANUSCRIPT_OUTPUT/parallel_speed_lunar_gateway_<timestamp>/`.
The master runner automatically includes the latest complete LG 6000-FE benchmark;
use `ParallelSpeedDirectory` to select a particular saved benchmark. Missing tests
are reported and skipped; figure generation never starts new optimizations.
Old 120-FE timing tests cannot supply the new histories and must be rerun.

Two separate EPS/PNG figures show mean best-so-far objective against FE and actual
optimization elapsed time. Callback time excludes pool startup and post-search
validation. Timing repetitions reuse seeds 0/1001 and alternate serial/parallel
execution order; they are not independent stochastic trials. Time curves use
previous-observation steps over each mode's common recorded time interval, without
extrapolating a completed run or inventing time-zero objective values.

Both figures and a copy of the numeric timing summary are exported directly to
`MANUSCRIPT_OUTPUT/`. The numeric first argument is `0` to retain final exports
(default), or `1` to clear final EPS/PNG files and generated table/summary files.
Benchmark subdirectories and numerical source data are preserved. When selecting
only one section with cleanup enabled, only that section's figures are regenerated.

### Formatting, final plot, EPS export

Formatting is applied by the plotters during construction using
`reviewer2_paper_style`: bold Times New Roman text, larger labels, reserved
margins, and legends with at most three columns. RA/Dec use the same canvas,
axis lengths, limits, and label positions. DRO shares the orbit-family canvas
and plotting rectangle. Each output remains a separate EPS for LaTeX assembly.

The occlusion geometry retains its existing canvas, fonts, geometry, and callouts.

The final plot is then passed to the working September 10 EPS print path.
Export functions do not resize fonts, rearrange legends, move axes, change
clipping/cameras, or rewrite bounding boxes. `format_manuscript_legend` is called
only during plot construction; the EPS writer never invokes it.

```matlab
run_manuscript_figures("definitions");
run_manuscript_figures(1); % Clear generated final exports and regenerate all
```

The master runner, table printer, saved parallel benchmark, and common output
parent are retained. `test_manuscript_figure_export` checks that the completed
styled scene is unchanged by EPS/PNG writing; it requires MATLAB graphics.

During construction, symmetric three-tick 3-D axes such as `[-0.05 0 0.05]`
retain only the two endpoint ticks. Data limits and 2-D zero baselines are
unchanged. Grouped bars use width 0.64, fixed outside padding, and 25-degree
rotation for long category labels; bar centers and error-bar coordinates stay
aligned. These settings are not applied during EPS export.
