function report = run_reviewer2_comparison_pipeline(saveFigures,runScreeningSensitivity)
%RUN_REVIEWER2_COMPARISON_PIPELINE Process the completed 6000-FE comparison.
%
% This pipeline does not launch or rerun any optimization. It processes:
%   4 optimizers x 3 target cases x 20 seeds = 240 optimization runs
%   GA, PSO, ABC, ACO
%   angles only, 3 observers, 1 Gateway period
%   6000 admitted search function evaluations per run
%
% In addition to the optimizer comparison, the pipeline can perform a
% post-optimization screening sensitivity study. The 20 GA designs from each
% target case are held fixed and their diagnostic EKF is re-evaluated with
% screening OFF. The saved screening-ON result is the paired control. Thus
% this isolates the measurement-availability gate; it is NOT a second
% optimization with screening disabled.
%
% Usage:
%   report = run_reviewer2_comparison_pipeline;
%   report = run_reviewer2_comparison_pipeline(false);       % display only
%   report = run_reviewer2_comparison_pipeline(true,false);  % skip sensitivity
%
% Inputs:
%   saveFigures            - save EPS/PNG previews (default true)
%   runScreeningSensitivity- build/reuse GA ON/OFF sensitivity (default true)

if nargin < 1 || isempty(saveFigures), saveFigures = true; end
if nargin < 2 || isempty(runScreeningSensitivity), runScreeningSensitivity = true; end
validateattributes(saveFigures,{'logical','numeric'},{'scalar'});
validateattributes(runScreeningSensitivity,{'logical','numeric'},{'scalar'});
saveFigures = logical(saveFigures);
runScreeningSensitivity = logical(runScreeningSensitivity);

paths = setup_project();
budget = 6000;
seeds = 0:19;
optimizers = ["GA","PSO","ABC","ACO"];
missions = ["LUNAR_GATEWAY","LOW_THRUST_TRANSFER","GATEWAY_IMPULSE"];
studyID = "reviewer2_comparison_v1";
comparisonRoot = fullfile(paths.runs,'COMPARISON');
expectedRuns = numel(optimizers)*numel(missions)*numel(seeds);

fprintf('\n--- Reviewer 2 6000-FE comparison pipeline ---\n');
fprintf('Target cases:               %d\n',numel(missions));
fprintf('Optimizers:                 %s\n',strjoin(cellstr(optimizers),', '));
fprintf('Independent seeds:          %s\n',mat2str(seeds));
fprintf('Measurement model:          ANGLES_ONLY\n');
fprintf('Observers:                  3\n');
fprintf('Gateway periods:            1\n');
fprintf('Search FE per run:          %d\n',budget);
fprintf('Expected optimization runs: %d\n\n',expectedRuns);

% Static study-definition checks fail before result processing.
test_project_structure();
test_fe_study_configuration();

assert(isfolder(comparisonRoot), ...
    'Comparison study root does not exist: %s',comparisonRoot);

[summary,inventory] = process_fe_convergence( ...
    comparisonRoot,studyID,seeds,budget,false,optimizers);

nonemptyRuns = inventory.run_file ~= "";
assert(sum(nonemptyRuns) == expectedRuns, ...
    'Expected %d saved optimization runs but found %d.', ...
    expectedRuns,sum(nonemptyRuns));
assert(~any(~inventory.valid), ...
    ['One or more comparison runs failed validation or are missing. ' ...
     'Inspect run_inventory.csv.']);
assert(height(summary) == numel(missions)*numel(optimizers), ...
    'Expected %d complete mission/optimizer groups but found %d.', ...
    numel(missions)*numel(optimizers),height(summary));
assert(numel(unique(summary.comparison_key)) == numel(missions), ...
    'Expected exactly one comparison configuration per target case.');
assert(all(ismember(summary.mission,missions)) && ...
    all(summary.measurement == "ANGLES_ONLY") && ...
    all(summary.num_observers == 3) && ...
    all(summary.n_runs == numel(seeds)) && ...
    all(summary.fe_budget == budget) && ...
    all(ismember(summary.optimizer,optimizers)), ...
    'Processed comparison metadata does not match the intended study.');

analysisDir = newest_analysis_directory(comparisonRoot);
metricsFile = fullfile(analysisDir,'final_run_metrics.csv');
assert(isfile(metricsFile),'Missing processed run metrics: %s',metricsFile);
runMetrics = readtable(metricsFile, ...
    'TextType','string','VariableNamingRule','preserve');
assert(height(runMetrics) == expectedRuns, ...
    'Expected %d processed metric rows but found %d.', ...
    expectedRuns,height(runMetrics));

requiredMetrics = [ ...
    "comparison_key","optimizer","seed","bestJ","search_fe", ...
    "solver_calls","parallel_overflow_evals","optimization_runtime_s", ...
    "budget_runtime_s","solver_wall_runtime_s","rmse_pos_km", ...
    "mean_effective_sigma_pos_km","mean_stability", ...
    "coverage_epoch_fraction","screening_count","run_file"];
assert(all(ismember(requiredMetrics,string(runMetrics.Properties.VariableNames))), ...
    'Processed comparison metrics are missing required fields.');
assert(all(runMetrics.search_fe == budget), ...
    'One or more processed runs do not report exactly 6000 admitted FE.');
assert(all(runMetrics.parallel_overflow_evals == 0), ...
    'The four-method comparison should not contain Bayesian overflow FE.');

results = build_comparison_results( ...
    summary,runMetrics,missions,optimizers,seeds,budget);
[objectiveTable,trackingTable] = format_comparison_tables(results);
rankings = build_rankings(results,missions,optimizers);

fprintf('\n--- 6000-FE objective/runtime results (mean +/- sample std) ---\n');
disp(objectiveTable);
fprintf('\n--- 6000-FE tracking/design results (mean +/- sample std) ---\n');
disp(trackingTable);
fprintf('\n--- Mission-wise objective/runtime rankings ---\n');
disp(rankings);

writetable(results,fullfile(analysisDir,'comparison_6000_results.csv'));
writetable(objectiveTable, ...
    fullfile(analysisDir,'comparison_6000_objective_runtime_formatted.csv'));
writetable(trackingTable, ...
    fullfile(analysisDir,'comparison_6000_tracking_formatted.csv'));
writetable(rankings,fullfile(analysisDir,'comparison_6000_rankings.csv'));

figureDir = "";
if saveFigures
    figureDir = string(fullfile(analysisDir,'paper_preview'));
    if ~isfolder(figureDir), mkdir(figureDir); end
end

% Core extended-comparison figures.
plot_comparison_convergence( ...
    analysisDir,missions,optimizers,budget,figureDir,saveFigures);
plot_grouped_metric(results,missions,optimizers, ...
    'BestJMean','BestJStd','Final best objective', ...
    'comparison_6000_objective',figureDir,saveFigures);
plot_grouped_metric(results,missions,optimizers, ...
    'RuntimeMean_s','RuntimeStd_s','Runtime to 6000 FE (s)', ...
    'comparison_6000_runtime',figureDir,saveFigures);
plot_grouped_metric(results,missions,optimizers, ...
    'RMSEPosMean_km','RMSEPosStd_km','Position RMSE (km)', ...
    'comparison_6000_position_rmse',figureDir,saveFigures);
plot_grouped_metric(results,missions,optimizers, ...
    'EffectiveSigmaPosMean_km','EffectiveSigmaPosStd_km', ...
    'Effective position sigma (km)', ...
    'comparison_6000_effective_sigma',figureDir,saveFigures);
plot_grouped_metric(results,missions,optimizers, ...
    'CoverageMean','CoverageStd','Coverage fraction', ...
    'comparison_6000_coverage',figureDir,saveFigures);
plot_grouped_metric(results,missions,optimizers, ...
    'ScreeningMean','ScreeningStd','Screened observer-epoch pairs', ...
    'comparison_6000_screening_count',figureDir,saveFigures);
plot_grouped_metric(results,missions,optimizers, ...
    'MeanStabilityMean','MeanStabilityStd','Mean observer stability index', ...
    'comparison_6000_stability',figureDir,saveFigures);

% Post-optimization screening ON/OFF sensitivity using GA reference designs.
screeningRuns = table();
screeningSummary = table();
if runScreeningSensitivity
    [screeningRuns,screeningSummary] = build_screening_sensitivity( ...
        comparisonRoot,summary,runMetrics,missions,seeds);
    fprintf('\n--- GA fixed-design screening ON/OFF sensitivity ---\n');
    disp(screeningSummary);
    writetable(screeningRuns, ...
        fullfile(analysisDir,'comparison_6000_screening_on_off_runs.csv'));
    writetable(screeningSummary, ...
        fullfile(analysisDir,'comparison_6000_screening_on_off_summary.csv'));

    plot_screening_metric(screeningSummary,missions, ...
        'ObjectiveMean','ObjectiveStd','Objective value', ...
        'comparison_6000_screening_on_off_objective',figureDir,saveFigures);
    plot_screening_metric(screeningSummary,missions, ...
        'RMSEPosMean_km','RMSEPosStd_km','Position RMSE (km)', ...
        'comparison_6000_screening_on_off_position_rmse',figureDir,saveFigures);
    plot_screening_metric(screeningSummary,missions, ...
        'EffectiveSigmaPosMean_km','EffectiveSigmaPosStd_km', ...
        'Effective position sigma (km)', ...
        'comparison_6000_screening_on_off_effective_sigma',figureDir,saveFigures);
end

report = struct();
report.studyID = studyID;
report.budget = budget;
report.seeds = seeds;
report.optimizers = optimizers;
report.missions = missions;
report.expectedRuns = expectedRuns;
report.analysisDirectory = string(analysisDir);
report.figureDirectory = figureDir;
report.summary = summary;
report.inventory = inventory;
report.runMetrics = runMetrics;
report.results = results;
report.objectiveTable = objectiveTable;
report.trackingTable = trackingTable;
report.rankings = rankings;
report.screeningRuns = screeningRuns;
report.screeningSummary = screeningSummary;

fprintf('\nReviewer 2 comparison pipeline passed.\n');
fprintf('Validated runs: %d/%d\n',sum(inventory.valid),expectedRuns);
fprintf('Processed data: %s\n',analysisDir);
if runScreeningSensitivity
    fprintf(['Screening sensitivity: GA designs held fixed; ' ...
        'diagnostic EKF re-evaluated with screening OFF.\n']);
end
if saveFigures
    fprintf('Paper-style previews: %s\n',figureDir);
else
    fprintf('Figures were displayed but not saved.\n');
end
end


function analysisDir = newest_analysis_directory(root)
directories = dir(fullfile(root,'FE_DATA_*'));
directories = directories([directories.isdir]);
assert(~isempty(directories), ...
    'No FE_DATA analysis directory was created under %s.',root);
[~,idx] = max([directories.datenum]);
analysisDir = fullfile(directories(idx).folder,directories(idx).name);
end


function results = build_comparison_results( ...
    summary,runMetrics,missions,optimizers,seeds,budget)

nRows = numel(missions)*numel(optimizers);
missionColumn = strings(nRows,1);
optimizerColumn = strings(nRows,1);
nRuns = nan(nRows,1);
searchFE = repmat(budget,nRows,1);
solverCallsMean = nan(nRows,1); solverCallsStd = nan(nRows,1);
bestJMean = nan(nRows,1); bestJStd = nan(nRows,1);
runtimeMean = nan(nRows,1); runtimeStd = nan(nRows,1);
solverWallMean = nan(nRows,1); solverWallStd = nan(nRows,1);
rmseMean = nan(nRows,1); rmseStd = nan(nRows,1);
sigmaMean = nan(nRows,1); sigmaStd = nan(nRows,1);
stabilityMean = nan(nRows,1); stabilityStd = nan(nRows,1);
coverageMean = nan(nRows,1); coverageStd = nan(nRows,1);
screeningMean = nan(nRows,1); screeningStd = nan(nRows,1);

row = 0;
for mission = missions
    for optimizer = optimizers
        row = row + 1;
        summaryRow = summary(summary.mission == mission & ...
            summary.optimizer == optimizer,:);
        assert(height(summaryRow) == 1, ...
            'Expected one summary row for %s/%s.',mission,optimizer);
        key = summaryRow.comparison_key;
        metricRows = runMetrics( ...
            runMetrics.comparison_key == key & ...
            runMetrics.optimizer == optimizer,:);
        metricRows = sortrows(metricRows,'seed');
        assert(height(metricRows) == numel(seeds) && ...
            isequal(metricRows.seed(:)',seeds), ...
            'Expected seeds %s for %s/%s.',mat2str(seeds),mission,optimizer);
        assert(all(metricRows.search_fe == budget), ...
            'Search FE mismatch for %s/%s.',mission,optimizer);

        tolerance = 1e-10*max(ones(height(metricRows),1), ...
            abs(metricRows.budget_runtime_s));
        assert(all(abs(metricRows.optimization_runtime_s- ...
            metricRows.budget_runtime_s) <= tolerance), ...
            'Optimization runtime is not equal-budget runtime for %s/%s.', ...
            mission,optimizer);
        assert(all(metricRows.solver_wall_runtime_s >= ...
            metricRows.budget_runtime_s), ...
            'Solver wall runtime precedes budget runtime for %s/%s.', ...
            mission,optimizer);

        missionColumn(row) = mission;
        optimizerColumn(row) = optimizer;
        nRuns(row) = height(metricRows);
        [solverCallsMean(row),solverCallsStd(row)] = ...
            sample_statistics(metricRows.solver_calls);
        [bestJMean(row),bestJStd(row)] = sample_statistics(metricRows.bestJ);
        [runtimeMean(row),runtimeStd(row)] = ...
            sample_statistics(metricRows.budget_runtime_s);
        [solverWallMean(row),solverWallStd(row)] = ...
            sample_statistics(metricRows.solver_wall_runtime_s);
        [rmseMean(row),rmseStd(row)] = ...
            sample_statistics(metricRows.rmse_pos_km);
        [sigmaMean(row),sigmaStd(row)] = ...
            sample_statistics(metricRows.mean_effective_sigma_pos_km);
        [stabilityMean(row),stabilityStd(row)] = ...
            sample_statistics(metricRows.mean_stability);
        [coverageMean(row),coverageStd(row)] = ...
            sample_statistics(metricRows.coverage_epoch_fraction);
        [screeningMean(row),screeningStd(row)] = ...
            sample_statistics(metricRows.screening_count);
    end
end

results = table( ...
    missionColumn,optimizerColumn,nRuns,searchFE, ...
    solverCallsMean,solverCallsStd,bestJMean,bestJStd, ...
    runtimeMean,runtimeStd,solverWallMean,solverWallStd, ...
    rmseMean,rmseStd,sigmaMean,sigmaStd,stabilityMean,stabilityStd, ...
    coverageMean,coverageStd,screeningMean,screeningStd, ...
    'VariableNames',{ ...
    'Mission','Optimizer','NRuns','SearchFE', ...
    'SolverCallsMean','SolverCallsStd','BestJMean','BestJStd', ...
    'RuntimeMean_s','RuntimeStd_s','SolverWallRuntimeMean_s', ...
    'SolverWallRuntimeStd_s','RMSEPosMean_km','RMSEPosStd_km', ...
    'EffectiveSigmaPosMean_km','EffectiveSigmaPosStd_km', ...
    'MeanStabilityMean','MeanStabilityStd','CoverageMean','CoverageStd', ...
    'ScreeningMean','ScreeningStd'});
end


function [objectiveTable,trackingTable] = format_comparison_tables(results)
caseName = mission_labels(results.Mission);
objectiveTable = table( ...
    caseName,results.Optimizer,results.NRuns,results.SearchFE, ...
    compose('%.5g +/- %.3g',results.SolverCallsMean,results.SolverCallsStd), ...
    compose('%.6g +/- %.3g',results.BestJMean,results.BestJStd), ...
    compose('%.5g +/- %.3g',results.RuntimeMean_s,results.RuntimeStd_s), ...
    'VariableNames',{ ...
    'Case','Optimizer','Runs','SearchFE','SolverCalls', ...
    'BestObjective','Runtime_s'});

trackingTable = table( ...
    caseName,results.Optimizer, ...
    compose('%.5g +/- %.3g',results.RMSEPosMean_km,results.RMSEPosStd_km), ...
    compose('%.5g +/- %.3g', ...
        results.EffectiveSigmaPosMean_km,results.EffectiveSigmaPosStd_km), ...
    compose('%.5g +/- %.3g', ...
        results.MeanStabilityMean,results.MeanStabilityStd), ...
    compose('%.4f +/- %.3f',results.CoverageMean,results.CoverageStd), ...
    compose('%.5g +/- %.3g',results.ScreeningMean,results.ScreeningStd), ...
    'VariableNames',{ ...
    'Case','Optimizer','RMSEPosition_km','EffectiveSigmaPosition_km', ...
    'MeanStability','CoverageFraction','ScreeningCount'});
end


function rankings = build_rankings(results,missions,optimizers)
rows = numel(missions)*numel(optimizers);
missionColumn = strings(rows,1); optimizerColumn = strings(rows,1);
objectiveRank = nan(rows,1); runtimeRank = nan(rows,1);
bestJMean = nan(rows,1); runtimeMean = nan(rows,1);
row = 0;
for mission = missions
    missionRows = results(results.Mission == mission,:);
    assert(height(missionRows) == numel(optimizers), ...
        'Incomplete ranking group for %s.',mission);
    [~,objectiveOrder] = sort(missionRows.BestJMean,'ascend');
    [~,runtimeOrder] = sort(missionRows.RuntimeMean_s,'ascend');
    objectiveRankLocal = nan(numel(optimizers),1);
    runtimeRankLocal = nan(numel(optimizers),1);
    objectiveRankLocal(objectiveOrder) = 1:numel(optimizers);
    runtimeRankLocal(runtimeOrder) = 1:numel(optimizers);
    for optimizer = optimizers
        row = row + 1;
        idx = find(missionRows.Optimizer == optimizer,1);
        missionColumn(row) = mission;
        optimizerColumn(row) = optimizer;
        objectiveRank(row) = objectiveRankLocal(idx);
        runtimeRank(row) = runtimeRankLocal(idx);
        bestJMean(row) = missionRows.BestJMean(idx);
        runtimeMean(row) = missionRows.RuntimeMean_s(idx);
    end
end
rankings = table(missionColumn,optimizerColumn,objectiveRank,runtimeRank, ...
    bestJMean,runtimeMean,'VariableNames',{ ...
    'Mission','Optimizer','ObjectiveRank','RuntimeRank','BestJMean','RuntimeMean_s'});
end


function [runs,summaryTable] = build_screening_sensitivity( ...
    comparisonRoot,summary,runMetrics,missions,seeds)
% Re-evaluate the 60 GA final designs with screening disabled. Screening-ON
% metrics are the saved validated results. Cache the expensive OFF EKF passes
% outside FE_DATA_* so reprocessing figures does not repeat them.

cacheDir = fullfile(comparisonRoot,'SCREENING_SENSITIVITY_GA_6000');
if ~isfolder(cacheDir), mkdir(cacheDir); end
cacheFile = fullfile(cacheDir,'screening_on_off_runs.csv');

source = select_ga_source_rows(summary,runMetrics,missions,seeds);
if isfile(cacheFile)
    candidate = readtable(cacheFile,'TextType','string', ...
        'VariableNamingRule','preserve');
    if valid_screening_cache(candidate,source)
        runs = candidate;
        summaryTable = summarize_screening_runs(runs,missions);
        fprintf('Reused screening sensitivity cache:\n%s\n',cacheFile);
        return;
    end
end

fprintf('\nBuilding fixed-design screening sensitivity (%d GA designs)...\n',height(source));
runs = table();
for k = 1:height(source)
    fprintf('  [%d/%d] %s seed %d\n', ...
        k,height(source),source.Mission(k),source.Seed(k));
    off = evaluate_screening_off(source.RunFile(k));

    onRow = table(source.Mission(k),"GA",source.Seed(k),true, ...
        source.RunFile(k),source.BestJ(k),source.RMSEPos_km(k), ...
        source.EffectiveSigmaPos_km(k),source.ScreeningCount(k), ...
        'VariableNames',{'Mission','Optimizer','Seed','UseScreening', ...
        'SourceRunFile','Objective','RMSEPos_km','EffectiveSigmaPos_km', ...
        'ScreeningCount'});
    offRow = table(source.Mission(k),"GA",source.Seed(k),false, ...
        source.RunFile(k),off.Objective,off.RMSEPos_km, ...
        off.EffectiveSigmaPos_km,off.ScreeningCount, ...
        'VariableNames',onRow.Properties.VariableNames);
    runs = [runs; onRow; offRow]; %#ok<AGROW>
end

writetable(runs,cacheFile);
summaryTable = summarize_screening_runs(runs,missions);
fprintf('Saved screening sensitivity cache:\n%s\n',cacheFile);
end


function source = select_ga_source_rows(summary,runMetrics,missions,seeds)
rowsExpected = numel(missions)*numel(seeds);
missionColumn = strings(rowsExpected,1); seedColumn = nan(rowsExpected,1);
runFile = strings(rowsExpected,1); bestJ = nan(rowsExpected,1);
rmse = nan(rowsExpected,1); sigma = nan(rowsExpected,1); screen = nan(rowsExpected,1);
row = 0;
for mission = missions
    summaryRow = summary(summary.mission == mission & summary.optimizer == "GA",:);
    assert(height(summaryRow) == 1,'Missing GA summary row for %s.',mission);
    key = summaryRow.comparison_key;
    metricRows = runMetrics(runMetrics.comparison_key == key & ...
        runMetrics.optimizer == "GA",:);
    metricRows = sortrows(metricRows,'seed');
    assert(height(metricRows) == numel(seeds) && ...
        isequal(metricRows.seed(:)',seeds), ...
        'Missing GA screening-sensitivity source runs for %s.',mission);
    for k = 1:height(metricRows)
        row = row + 1;
        missionColumn(row) = mission;
        seedColumn(row) = metricRows.seed(k);
        runFile(row) = metricRows.run_file(k);
        bestJ(row) = metricRows.bestJ(k);
        rmse(row) = metricRows.rmse_pos_km(k);
        sigma(row) = metricRows.mean_effective_sigma_pos_km(k);
        screen(row) = metricRows.screening_count(k);
    end
end
source = table(missionColumn,seedColumn,runFile,bestJ,rmse,sigma,screen, ...
    'VariableNames',{'Mission','Seed','RunFile','BestJ','RMSEPos_km', ...
    'EffectiveSigmaPos_km','ScreeningCount'});
end


function tf = valid_screening_cache(candidate,source)
tf = false;
required = ["Mission","Optimizer","Seed","UseScreening","SourceRunFile", ...
    "Objective","RMSEPos_km","EffectiveSigmaPos_km","ScreeningCount"];
if ~all(ismember(required,string(candidate.Properties.VariableNames)))
    return;
end
if height(candidate) ~= 2*height(source), return; end
for k = 1:height(source)
    rows = candidate(candidate.Mission == source.Mission(k) & ...
        candidate.Seed == source.Seed(k) & candidate.Optimizer == "GA",:);
    if height(rows) ~= 2 || numel(unique(rows.UseScreening)) ~= 2 || ...
            ~all(rows.SourceRunFile == source.RunFile(k))
        return;
    end
    on = rows(logical(rows.UseScreening),:);
    if height(on) ~= 1 || abs(on.Objective-source.BestJ(k)) > ...
            1e-9*max(1,abs(source.BestJ(k)))
        return;
    end
end
tf = true;
end


function out = evaluate_screening_off(runFile)
dataDir = fileparts(char(runFile));
trackingFile = fullfile(dataDir,'tracking_data.mat');
assert(isfile(runFile) && isfile(trackingFile), ...
    'Missing saved run or tracking file for screening sensitivity.');
S = load(runFile,'runState');
T = load(trackingFile,'tracking');
r = S.runState;
tracking = T.tracking;
s = r.settings;
assert(s.useScreening, ...
    'Screening sensitivity source run must be a screening-ON result.');
assert(isfield(r,'observers') && istable(r.observers), ...
    'Saved run is missing final observer data.');
observerICs = r.observers.initial_state;
stabilities = r.observers.stability_index;
measCfg = s.measurements;
sunFcn = @(t) sun_pos_bc4bp(t,s.LU,s.TU,s.theta0,s.i_sun);

[estimate,cov,screeningCount] = cr3bp_ekf( ...
    observerICs,tracking.truth,tracking.t_TU,s.P0,s.Q,s.R,s.mu,s.LU, ...
    sunFcn,s.sun_exclusion,s.moon_exclusion,s.earth_exclusion,false,measCfg);
[J,~,~,~] = compute_cost( ...
    tracking.truth,estimate,cov,stabilities,'SOO',s.costFlags,s.cost);
metrics = diagnostic_metrics(tracking.truth,estimate,cov,s.LU,s.TU);
out = struct('Objective',J,'RMSEPos_km',metrics.RMSEPos_km, ...
    'EffectiveSigmaPos_km',metrics.EffectiveSigmaPos_km, ...
    'ScreeningCount',screeningCount);
end


function m = diagnostic_metrics(truth,estimate,cov,LU,TU)
VU = LU/TU;
err = estimate-truth;
m.RMSEPos_km = sqrt(mean(sum(err(:,1:3).^2,2)))*LU;
m.RMSEVel_kms = sqrt(mean(sum(err(:,4:6).^2,2)))*VU;
N = size(cov,1);
effPos = zeros(N,1);
for k = 1:N
    P = squeeze(cov(k,:,:));
    P = 0.5*(P+P');
    effPos(k) = max(det(P(1:3,1:3)),realmin)^(1/6);
end
m.EffectiveSigmaPos_km = mean(effPos)*LU;
end


function summaryTable = summarize_screening_runs(runs,missions)
conditions = [true false];
rows = numel(missions)*numel(conditions);
missionColumn = strings(rows,1); useScreening = false(rows,1); nRuns = nan(rows,1);
objectiveMean = nan(rows,1); objectiveStd = nan(rows,1);
rmseMean = nan(rows,1); rmseStd = nan(rows,1);
sigmaMean = nan(rows,1); sigmaStd = nan(rows,1);
screenMean = nan(rows,1); screenStd = nan(rows,1);
row = 0;
for mission = missions
    for condition = conditions
        row = row + 1;
        x = runs(runs.Mission == mission & runs.UseScreening == condition,:);
        assert(height(x) == 20, ...
            'Expected 20 GA sensitivity runs for %s/screening=%d.',mission,condition);
        missionColumn(row) = mission;
        useScreening(row) = condition;
        nRuns(row) = height(x);
        [objectiveMean(row),objectiveStd(row)] = sample_statistics(x.Objective);
        [rmseMean(row),rmseStd(row)] = sample_statistics(x.RMSEPos_km);
        [sigmaMean(row),sigmaStd(row)] = sample_statistics(x.EffectiveSigmaPos_km);
        [screenMean(row),screenStd(row)] = sample_statistics(x.ScreeningCount);
    end
end
summaryTable = table(missionColumn,useScreening,nRuns, ...
    objectiveMean,objectiveStd,rmseMean,rmseStd,sigmaMean,sigmaStd, ...
    screenMean,screenStd,'VariableNames',{ ...
    'Mission','UseScreening','NRuns','ObjectiveMean','ObjectiveStd', ...
    'RMSEPosMean_km','RMSEPosStd_km','EffectiveSigmaPosMean_km', ...
    'EffectiveSigmaPosStd_km','ScreeningCountMean','ScreeningCountStd'});
end


function plot_comparison_convergence( ...
    analysisDir,missions,optimizers,budget,figureDir,saveFigures)
files = dir(fullfile(analysisDir,'convergence_*.mat'));
assert(numel(files) == numel(missions), ...
    'Expected one convergence file per target case.');
colors = lines(numel(optimizers));
plotStartFE = 60;
for mission = missions
    loaded = struct(); found = false;
    for k = 1:numel(files)
        candidate = load(fullfile(files(k).folder,files(k).name), ...
            'comparison','curves');
        if string(candidate.comparison.settings.mission.type) == mission
            loaded = candidate; found = true; break;
        end
    end
    assert(found,'No convergence data found for %s.',mission);
    fig = create_paper_figure(7.2,4.4); ax = axes(fig);
    hold(ax,'on'); box(ax,'on'); grid(ax,'on');
    curveOptimizers = upper(string({loaded.curves.optimizer}));
    lineHandles = gobjects(numel(optimizers),1);
    for a = 1:numel(optimizers)
        idx = find(curveOptimizers == optimizers(a),1);
        assert(~isempty(idx),'Missing %s convergence curve.',optimizers(a));
        curve = loaded.curves(idx);
        fe = double(curve.fe(:)); meanBest = double(curve.mean(:));
        valid = isfinite(meanBest) & fe >= plotStartFE;
        assert(any(valid),'No convergence data for %s/%s.',mission,optimizers(a));
        x = fe(valid); y = meanBest(valid);
        lineHandles(a) = stairs(ax,x,y,'Color',colors(a,:), ...
            'LineWidth',2.0,'DisplayName',optimizers(a));
        markerStride = max(1,round(numel(x)/12));
        markerIdx = unique([1:markerStride:numel(x),numel(x)]);
        plot(ax,x(markerIdx),y(markerIdx),'o','Color',colors(a,:), ...
            'MarkerFaceColor',colors(a,:),'MarkerSize',4, ...
            'HandleVisibility','off');
    end
    xlim(ax,[plotStartFE budget]);
    xticks(ax,unique([plotStartFE 1000:1000:budget budget]));
    xlabel(ax,'Function evaluations','FontWeight','bold');
    ylabel(ax,'Mean best-so-far objective','FontWeight','bold');
    apply_figure_style(ax);
    lgd = legend(ax,lineHandles,cellstr(optimizers), ...
        'Location','northoutside','Orientation','horizontal', ...
        'NumColumns',numel(optimizers));
    format_legend(lgd);
    export_preview(fig,figureDir, ...
        "comparison_6000_convergence_"+mission_code(mission),saveFigures);
end
end


function plot_grouped_metric(results,missions,optimizers, ...
    valueField,errorField,yLabel,stem,figureDir,saveFigures)
values = nan(numel(missions),numel(optimizers)); errors = values;
for m = 1:numel(missions)
    for a = 1:numel(optimizers)
        row = results(results.Mission == missions(m) & ...
            results.Optimizer == optimizers(a),:);
        assert(height(row) == 1,'Missing result for %s/%s.',missions(m),optimizers(a));
        values(m,a) = row.(valueField); errors(m,a) = row.(errorField);
    end
end
fig = create_paper_figure(7.4,4.8); ax = axes(fig);
hold(ax,'on'); box(ax,'on'); grid(ax,'on');
b = bar(ax,1:numel(missions),values,'grouped'); drawnow;
for a = 1:numel(optimizers)
    x = b(a).XEndPoints;
    lowerErrors = min(max(errors(:,a),0),max(values(:,a),0));
    errorbar(ax,x,values(:,a),lowerErrors,max(errors(:,a),0), ...
        'k.','LineWidth',1.25,'CapSize',8,'HandleVisibility','off');
    b(a).DisplayName = optimizers(a);
end
ax.XTick = 1:numel(missions); ax.XTickLabel = cellstr(mission_labels(missions));
xlabel(ax,'Target case','FontWeight','bold'); ylabel(ax,yLabel,'FontWeight','bold');
apply_figure_style(ax);
lgd = legend(ax,b,cellstr(optimizers),'Location','northoutside', ...
    'Orientation','horizontal','NumColumns',numel(optimizers));
format_legend(lgd);
export_preview(fig,figureDir,stem,saveFigures);
end


function plot_screening_metric(summaryTable,missions,valueField,errorField, ...
    yLabel,stem,figureDir,saveFigures)
conditions = [true false];
values = nan(numel(missions),2); errors = values;
for m = 1:numel(missions)
    for c = 1:2
        row = summaryTable(summaryTable.Mission == missions(m) & ...
            summaryTable.UseScreening == conditions(c),:);
        assert(height(row) == 1,'Missing screening sensitivity row.');
        values(m,c) = row.(valueField); errors(m,c) = row.(errorField);
    end
end
fig = create_paper_figure(7.4,4.8); ax = axes(fig);
hold(ax,'on'); box(ax,'on'); grid(ax,'on');
b = bar(ax,1:numel(missions),values,'grouped'); drawnow;
for c = 1:2
    x = b(c).XEndPoints;
    lowerErrors = min(max(errors(:,c),0),max(values(:,c),0));
    errorbar(ax,x,values(:,c),lowerErrors,max(errors(:,c),0), ...
        'k.','LineWidth',1.25,'CapSize',8,'HandleVisibility','off');
end
b(1).DisplayName = 'Screening ON'; b(2).DisplayName = 'Screening OFF';
ax.XTick = 1:numel(missions); ax.XTickLabel = cellstr(mission_labels(missions));
xlabel(ax,'Target case','FontWeight','bold'); ylabel(ax,yLabel,'FontWeight','bold');
apply_figure_style(ax);
lgd = legend(ax,b,{'Screening ON','Screening OFF'}, ...
    'Location','northoutside','Orientation','horizontal','NumColumns',2);
format_legend(lgd);
export_preview(fig,figureDir,stem,saveFigures);
end


function labels = mission_labels(missions)
missions = string(missions(:)); labels = strings(size(missions));
for k = 1:numel(missions), labels(k) = mission_label(missions(k)); end
end

function label = mission_label(mission)
switch upper(string(mission))
    case "LUNAR_GATEWAY", label = "Lunar Gateway";
    case "LOW_THRUST_TRANSFER", label = "Low-thrust transfer";
    case "GATEWAY_IMPULSE", label = "Gateway impulse";
    otherwise, label = string(mission);
end
end

function code = mission_code(mission)
switch upper(string(mission))
    case "LUNAR_GATEWAY", code = "lg";
    case "LOW_THRUST_TRANSFER", code = "lt";
    case "GATEWAY_IMPULSE", code = "gi";
    otherwise, code = lower(string(mission));
end
end

function fig = create_paper_figure(widthIn,heightIn)
fig = figure('Color','w','Units','inches','Position',[1 1 widthIn heightIn], ...
    'PaperUnits','inches','PaperSize',[widthIn heightIn], ...
    'PaperPosition',[0 0 widthIn heightIn],'PaperPositionMode','manual', ...
    'Renderer','painters','InvertHardcopy','off');
end

function apply_figure_style(ax)
set(ax,'FontName','Times New Roman','FontSize',12,'FontWeight','bold', ...
    'LineWidth',1.0,'TickDir','out');
ax.XLabel.FontSize = 14; ax.YLabel.FontSize = 14;
end

function format_legend(lgd)
lgd.FontName = 'Times New Roman'; lgd.FontSize = 12;
lgd.FontWeight = 'bold'; lgd.Box = 'off';
end

function export_preview(fig,figureDir,stem,saveFigures)
drawnow;
if ~saveFigures, return; end
assert(strlength(string(figureDir)) > 0,'Figure directory is empty.');
base = fullfile(char(figureDir),char(stem));
print(fig,[base '.eps'],'-depsc','-painters');
exportgraphics(fig,[base '.png'],'Resolution',300);
end

function [mu,sigma] = sample_statistics(values)
values = double(values(:)); mu = mean(values);
if numel(values) < 2, sigma = NaN; else, sigma = std(values); end
end
