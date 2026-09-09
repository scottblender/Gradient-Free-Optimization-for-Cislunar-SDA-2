function report = run_reviewer2_comparison_pipeline(saveFigures)
%RUN_REVIEWER2_COMPARISON_PIPELINE Process the completed 6000-FE comparison.
%
% This pipeline does not launch or rerun any optimization. It processes:
%   4 optimizers x 3 target cases x 20 seeds = 240 optimization runs
%   GA, PSO, ABC, ACO
%   angles only, 3 observers, 1 Gateway period
%   screening ON, J1+J2+J3
%   6000 admitted search function evaluations per run
%
% The earlier fixed-design post-processing screening ON/OFF diagnostic has
% been removed. Screening/objective sensitivity is now handled only by the
% separate GA_OBJECTIVE_SCREENING optimization study.
%
% In addition to statistical results, this pipeline selects the lowest-cost
% observed run for each optimizer/mission pair and exports observer-orbit
% geometry panels. Those panels show how the optimizers select genuinely
% different cislunar constellations for the same target case.
%
% Usage:
%   report = run_reviewer2_comparison_pipeline;
%   report = run_reviewer2_comparison_pipeline(false); % display only

if nargin < 1 || isempty(saveFigures), saveFigures = true; end
validateattributes(saveFigures,{'logical','numeric'},{'scalar'});
saveFigures = logical(saveFigures);

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
fprintf('Screening:                  ON\n');
fprintf('Objective:                  J1 + J2 + J3\n');
fprintf('Search FE per run:          %d\n',budget);
fprintf('Expected optimization runs: %d\n\n',expectedRuns);

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
    "coverage_epoch_fraction","mean_available_observers", ...
    "J1_weighted","J2_weighted","J3_weighted", ...
    "screening_count","run_file"];
assert(all(ismember(requiredMetrics,string(runMetrics.Properties.VariableNames))), ...
    'Processed comparison metrics are missing required fields.');
assert(all(runMetrics.search_fe == budget), ...
    'One or more processed runs do not report exactly 6000 admitted FE.');
assert(all(runMetrics.parallel_overflow_evals == 0), ...
    'The four-method comparison should not contain Bayesian overflow FE.');

results = build_comparison_results( ...
    summary,runMetrics,missions,optimizers,seeds,budget);
[objectiveTable,trackingTable] = format_comparison_tables(results);
[rankings,overallRanking,bestByMission] = ...
    build_rankings(results,missions,optimizers);
bestGeometryRuns = select_best_geometry_runs( ...
    summary,runMetrics,missions,optimizers,seeds);

fprintf('\n--- 6000-FE objective/runtime results (mean +/- sample std) ---\n');
disp(objectiveTable);
fprintf('\n--- 6000-FE tracking/design results (mean +/- sample std) ---\n');
disp(trackingTable);
fprintf('\n--- Mission-wise objective/runtime rankings ---\n');
disp(rankings);
fprintf('\n--- Overall optimizer ranking (mean mission rank) ---\n');
disp(overallRanking);
fprintf('\n--- Best optimizer for each target case ---\n');
disp(bestByMission);
fprintf('\n--- Best observed runs used for optimizer geometry panels ---\n');
disp(bestGeometryRuns(:,{'Mission','PanelLabel','Seed','BestObjective'}));

writetable(results,fullfile(analysisDir,'comparison_6000_results.csv'));
writetable(objectiveTable, ...
    fullfile(analysisDir,'comparison_6000_objective_runtime_formatted.csv'));
writetable(trackingTable, ...
    fullfile(analysisDir,'comparison_6000_tracking_formatted.csv'));
writetable(rankings,fullfile(analysisDir,'comparison_6000_rankings.csv'));
writetable(overallRanking, ...
    fullfile(analysisDir,'comparison_6000_overall_ranking.csv'));
writetable(bestByMission, ...
    fullfile(analysisDir,'comparison_6000_best_by_mission.csv'));
writetable(bestGeometryRuns, ...
    fullfile(analysisDir,'comparison_6000_geometry_selected_runs.csv'));

figureDir = "";
if saveFigures
    figureDir = string(fullfile(analysisDir,'paper_preview'));
    if ~isfolder(figureDir), mkdir(figureDir); end
end

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
    'ScreeningMean','ScreeningStd','Rejected measurement opportunities', ...
    'comparison_6000_screening_count',figureDir,saveFigures);
plot_grouped_metric(results,missions,optimizers, ...
    'MeanStabilityMean','MeanStabilityStd','Mean observer stability index', ...
    'comparison_6000_stability',figureDir,saveFigures);
plot_comparison_metric_panel(results,missions,optimizers,figureDir,saveFigures);
plot_comparison_component_panel(results,missions,optimizers,figureDir,saveFigures);
plot_overall_ranking(overallRanking,optimizers,figureDir,saveFigures);

geometryDetails = plot_reviewer2_constellation_geometry( ...
    bestGeometryRuns,figureDir,"comparison_geometry",saveFigures);
writetable(geometryDetails, ...
    fullfile(analysisDir,'comparison_6000_geometry_details.csv'));

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
report.overallRanking = overallRanking;
report.bestByMission = bestByMission;
report.bestGeometryRuns = bestGeometryRuns;
report.geometryDetails = geometryDetails;

fprintf('\nReviewer 2 comparison pipeline passed.\n');
fprintf('Validated runs: %d/%d\n',sum(inventory.valid),expectedRuns);
fprintf('Processed data: %s\n',analysisDir);
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
availableMean = nan(nRows,1); availableStd = nan(nRows,1);
screeningMean = nan(nRows,1); screeningStd = nan(nRows,1);
j1Mean = nan(nRows,1); j1Std = nan(nRows,1);
j2Mean = nan(nRows,1); j2Std = nan(nRows,1);
j3Mean = nan(nRows,1); j3Std = nan(nRows,1);

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
        [availableMean(row),availableStd(row)] = ...
            sample_statistics(metricRows.mean_available_observers);
        [screeningMean(row),screeningStd(row)] = ...
            sample_statistics(metricRows.screening_count);
        [j1Mean(row),j1Std(row)] = sample_statistics(metricRows.J1_weighted);
        [j2Mean(row),j2Std(row)] = sample_statistics(metricRows.J2_weighted);
        [j3Mean(row),j3Std(row)] = sample_statistics(metricRows.J3_weighted);
    end
end

results = table( ...
    missionColumn,optimizerColumn,nRuns,searchFE, ...
    solverCallsMean,solverCallsStd,bestJMean,bestJStd, ...
    runtimeMean,runtimeStd,solverWallMean,solverWallStd, ...
    rmseMean,rmseStd,sigmaMean,sigmaStd,stabilityMean,stabilityStd, ...
    coverageMean,coverageStd,availableMean,availableStd, ...
    screeningMean,screeningStd,j1Mean,j1Std,j2Mean,j2Std,j3Mean,j3Std, ...
    'VariableNames',{ ...
    'Mission','Optimizer','NRuns','SearchFE', ...
    'SolverCallsMean','SolverCallsStd','BestJMean','BestJStd', ...
    'RuntimeMean_s','RuntimeStd_s','SolverWallRuntimeMean_s', ...
    'SolverWallRuntimeStd_s','RMSEPosMean_km','RMSEPosStd_km', ...
    'EffectiveSigmaPosMean_km','EffectiveSigmaPosStd_km', ...
    'MeanStabilityMean','MeanStabilityStd','CoverageMean','CoverageStd', ...
    'AvailableObserversMean','AvailableObserversStd', ...
    'ScreeningMean','ScreeningStd','J1Mean','J1Std','J2Mean','J2Std', ...
    'J3Mean','J3Std'});
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
    compose('%.4f +/- %.3f', ...
        results.AvailableObserversMean,results.AvailableObserversStd), ...
    compose('%.5g +/- %.3g',results.ScreeningMean,results.ScreeningStd), ...
    'VariableNames',{ ...
    'Case','Optimizer','RMSEPosition_km','EffectiveSigmaPosition_km', ...
    'MeanStability','CoverageFraction','MeanAvailableObservers', ...
    'RejectedMeasurementOpportunities'});
end


function [rankings,overall,bestByMission] = build_rankings(results,missions,optimizers)
rows = numel(missions)*numel(optimizers);
missionColumn = strings(rows,1); optimizerColumn = strings(rows,1);
objectiveRank = nan(rows,1); runtimeRank = nan(rows,1);
bestJMean = nan(rows,1); runtimeMean = nan(rows,1);
row = 0;
for mission = missions
    missionRows = results(results.Mission == mission,:);
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

averageObjectiveRank = nan(numel(optimizers),1);
averageRuntimeRank = nan(numel(optimizers),1);
objectiveWins = zeros(numel(optimizers),1);
normalizedObjective = nan(numel(optimizers),1);
for k = 1:numel(optimizers)
    rows = rankings(rankings.Optimizer == optimizers(k),:);
    averageObjectiveRank(k) = mean(rows.ObjectiveRank);
    averageRuntimeRank(k) = mean(rows.RuntimeRank);
    objectiveWins(k) = sum(rows.ObjectiveRank == 1);

    normalized = nan(numel(missions),1);
    for m = 1:numel(missions)
        missionRows = results(results.Mission == missions(m),:);
        lo = min(missionRows.BestJMean); hi = max(missionRows.BestJMean);
        value = missionRows.BestJMean(missionRows.Optimizer == optimizers(k));
        if hi > lo
            normalized(m) = (value-lo)/(hi-lo);
        else
            normalized(m) = 0;
        end
    end
    normalizedObjective(k) = mean(normalized);
end
overall = table(optimizers(:),averageObjectiveRank,normalizedObjective, ...
    objectiveWins,averageRuntimeRank,'VariableNames',{ ...
    'Optimizer','MeanObjectiveRank','MeanNormalizedObjective', ...
    'MissionWins','MeanRuntimeRank'});
overall = sortrows(overall, ...
    {'MeanObjectiveRank','MeanNormalizedObjective','MeanRuntimeRank'}, ...
    {'ascend','ascend','ascend'});
overall.OverallRank = (1:height(overall))';

missionColumn = missions(:); bestOptimizer = strings(numel(missions),1);
bestObjectiveMean = nan(numel(missions),1); bestObjectiveStd = nan(numel(missions),1);
for m = 1:numel(missions)
    missionRows = results(results.Mission == missions(m),:);
    [bestObjectiveMean(m),idx] = min(missionRows.BestJMean);
    bestOptimizer(m) = missionRows.Optimizer(idx);
    bestObjectiveStd(m) = missionRows.BestJStd(idx);
end
bestByMission = table(missionColumn,bestOptimizer,bestObjectiveMean, ...
    bestObjectiveStd,'VariableNames',{ ...
    'Mission','BestOptimizer','BestObjectiveMean','BestObjectiveStd'});
end


function selected = select_best_geometry_runs( ...
    summary,runMetrics,missions,optimizers,seeds)
rows = numel(missions)*numel(optimizers);
missionColumn = strings(rows,1); panelKey = strings(rows,1); panelLabel = strings(rows,1);
runFile = strings(rows,1); bestObjective = nan(rows,1); seedColumn = nan(rows,1);
row = 0;
for mission = missions
    for optimizer = optimizers
        row = row + 1;
        summaryRow = summary(summary.mission == mission & ...
            summary.optimizer == optimizer,:);
        assert(height(summaryRow) == 1,'Missing summary row for geometry selection.');
        metricRows = runMetrics( ...
            runMetrics.comparison_key == summaryRow.comparison_key & ...
            runMetrics.optimizer == optimizer,:);
        assert(height(metricRows) == numel(seeds), ...
            'Incomplete geometry-selection group for %s/%s.',mission,optimizer);
        [value,idx] = min(metricRows.bestJ);
        missionColumn(row) = mission;
        panelKey(row) = lower(optimizer);
        panelLabel(row) = optimizer;
        runFile(row) = metricRows.run_file(idx);
        bestObjective(row) = value;
        seedColumn(row) = metricRows.seed(idx);
    end
end
selected = table(missionColumn,panelKey,panelLabel,runFile,bestObjective,seedColumn, ...
    'VariableNames',{'Mission','PanelKey','PanelLabel','RunFile','BestObjective','Seed'});
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
        deviation = double(curve.std(valid));
        bandIdx = unique(round(linspace(1,numel(x),min(240,numel(x)))));
        bandX = x(bandIdx); bandMean = y(bandIdx);
        bandStd = deviation(bandIdx);
        bandColor = 0.82*[1 1 1] + 0.18*colors(a,:);
        fill(ax,[bandX;flipud(bandX)], ...
            [max(0,bandMean-bandStd);flipud(bandMean+bandStd)], ...
            bandColor,'EdgeColor','none','HandleVisibility','off');
        lineHandles(a) = stairs(ax,x,y,'Color',colors(a,:), ...
            'LineWidth',2.0,'DisplayName',optimizer_label(optimizers(a)));
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
    lgd = legend(ax,lineHandles,cellstr(optimizer_labels(optimizers)), ...
        'Location','northoutside','Orientation','horizontal', ...
        'NumColumns',numel(optimizers));
    format_legend(lgd);
    export_preview(fig,figureDir, ...
        "comparison_6000_convergence_"+mission_code(mission),saveFigures);
end
end


function plot_comparison_metric_panel(results,missions,optimizers,figureDir,saveFigures)
specs = { ...
    'RMSEPosMean_km','RMSEPosStd_km','Position RMSE (km)'; ...
    'EffectiveSigmaPosMean_km','EffectiveSigmaPosStd_km', ...
        'Effective position sigma (km)'; ...
    'MeanStabilityMean','MeanStabilityStd','Mean stability index'; ...
    'AvailableObserversMean','AvailableObserversStd', ...
        'Mean available observers'};
fig = create_paper_figure(7.6,6.5);
tiled = tiledlayout(fig,2,2,'Padding','compact','TileSpacing','compact');
for q = 1:size(specs,1)
    ax = nexttile(tiled); hold(ax,'on'); box(ax,'on'); grid(ax,'on');
    [values,errors] = metric_arrays(results,missions,optimizers,specs{q,1},specs{q,2});
    b = bar(ax,1:numel(missions),values,'grouped'); drawnow;
    for a = 1:numel(optimizers)
        lower = min(max(errors(:,a),0),max(values(:,a),0));
        errorbar(ax,b(a).XEndPoints,values(:,a),lower,max(errors(:,a),0), ...
            'k.','LineWidth',1.0,'CapSize',6,'HandleVisibility','off');
        b(a).DisplayName = optimizer_label(optimizers(a));
    end
    ax.XTick = 1:numel(missions);
    ax.XTickLabel = cellstr(mission_labels(missions));
    ylabel(ax,specs{q,3},'FontWeight','bold');
    apply_figure_style(ax);
    if q == 1
        lgd = legend(ax,b,cellstr(optimizer_labels(optimizers)), ...
            'Orientation','horizontal','NumColumns',numel(optimizers));
        format_legend(lgd);
        lgd.Layout.Tile = 'north';
    end
end
xlabel(tiled,'Target case','FontName','Times New Roman', ...
    'FontSize',14,'FontWeight','bold');
export_preview(fig,figureDir,'comparison_6000_tracking_metrics',saveFigures);
end


function plot_comparison_component_panel(results,missions,optimizers,figureDir,saveFigures)
specs = { ...
    'J1Mean','J1Std','Weighted J_1'; ...
    'J2Mean','J2Std','Weighted J_2'; ...
    'J3Mean','J3Std','Weighted J_3'};
fig = create_paper_figure(7.6,6.4);
tiled = tiledlayout(fig,2,2,'Padding','compact','TileSpacing','compact');
for q = 1:3
    ax = nexttile(tiled); hold(ax,'on'); box(ax,'on'); grid(ax,'on');
    [values,errors] = metric_arrays(results,missions,optimizers,specs{q,1},specs{q,2});
    b = bar(ax,1:numel(missions),values,'grouped'); drawnow;
    for a = 1:numel(optimizers)
        lower = min(max(errors(:,a),0),max(values(:,a),0));
        errorbar(ax,b(a).XEndPoints,values(:,a),lower,max(errors(:,a),0), ...
            'k.','LineWidth',1.0,'CapSize',6,'HandleVisibility','off');
    end
    ax.XTick = 1:numel(missions);
    ax.XTickLabel = cellstr(mission_labels(missions));
    ylabel(ax,specs{q,3},'Interpreter','tex','FontWeight','bold');
    apply_figure_style(ax);
end
ax = nexttile(tiled); hold(ax,'on'); box(ax,'on'); grid(ax,'on');
[values,errors] = metric_arrays(results,missions,optimizers, ...
    'BestJMean','BestJStd');
b = bar(ax,1:numel(missions),values,'grouped'); drawnow;
for a = 1:numel(optimizers)
    lower = min(max(errors(:,a),0),max(values(:,a),0));
    errorbar(ax,b(a).XEndPoints,values(:,a),lower,max(errors(:,a),0), ...
        'k.','LineWidth',1.0,'CapSize',6,'HandleVisibility','off');
    b(a).DisplayName = optimizer_label(optimizers(a));
end
ax.XTick = 1:numel(missions); ax.XTickLabel = cellstr(mission_labels(missions));
ylabel(ax,'Total objective','FontWeight','bold'); apply_figure_style(ax);
lgd = legend(ax,b,cellstr(optimizer_labels(optimizers)), ...
    'Orientation','horizontal','NumColumns',numel(optimizers));
format_legend(lgd);
lgd.Layout.Tile = 'north';
export_preview(fig,figureDir,'comparison_6000_objective_components',saveFigures);
end


function plot_overall_ranking(overall,optimizers,figureDir,saveFigures)
values = nan(numel(optimizers),1);
for k = 1:numel(optimizers)
    values(k) = overall.MeanObjectiveRank(overall.Optimizer == optimizers(k));
end
fig = create_paper_figure(6.6,4.2); ax = axes(fig);
bar(ax,1:numel(optimizers),values,0.68,'FaceColor',[0.22 0.45 0.78]);
box(ax,'on'); grid(ax,'on');
ax.XTick = 1:numel(optimizers);
ax.XTickLabel = cellstr(optimizer_labels(optimizers));
ylim(ax,[0 numel(optimizers)+0.5]);
xlabel(ax,'Optimizer','FontWeight','bold');
ylabel(ax,'Mean objective rank','FontWeight','bold');
apply_figure_style(ax);
export_preview(fig,figureDir,'comparison_6000_overall_rank',saveFigures);
end


function [values,errors] = metric_arrays(results,missions,optimizers,valueField,errorField)
values = nan(numel(missions),numel(optimizers)); errors = values;
for m = 1:numel(missions)
    for a = 1:numel(optimizers)
        row = results(results.Mission == missions(m) & ...
            results.Optimizer == optimizers(a),:);
        assert(height(row) == 1,'Missing metric row for %s/%s.',missions(m),optimizers(a));
        values(m,a) = row.(valueField);
        errors(m,a) = row.(errorField);
    end
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


function labels = mission_labels(missions)
missions = string(missions(:)); labels = strings(size(missions));
for k = 1:numel(missions), labels(k) = mission_label(missions(k)); end
end

function labels = optimizer_labels(values)
values = string(values(:)); labels = strings(size(values));
for k = 1:numel(values), labels(k) = optimizer_label(values(k)); end
end

function label = optimizer_label(value)
if upper(string(value)) == "BAYESIAN", label = "BO"; else, label = upper(string(value)); end
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
close(fig);
end

function [mu,sigma] = sample_statistics(values)
values = double(values(:)); mu = mean(values);
if numel(values) < 2, sigma = NaN; else, sigma = std(values); end
end
