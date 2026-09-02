function report = run_reviewer2_pipeline_pilot(runOptimizations,saveFigures)
%RUN_REVIEWER2_PIPELINE_PILOT End-to-end three-seed FE study check.
%
% This is an integration pilot, not a quick unit test. By default it runs:
%   5 optimizers x 3 fixed target cases x 3 seeds = 45 optimization runs
%   120 search function evaluations per run
%
% The script then validates the saved schema, aligns convergence by function
% evaluations, prints mean +/- sample-standard-deviation tables, and creates
% paper-style preview figures. For each case, the lowest-objective run across
% every optimizer and seed supplies the trajectory and EKF diagnostic plots.
% Pilot output is isolated from the full study.
%
% Usage:
%   report = run_reviewer2_pipeline_pilot;
%   report = run_reviewer2_pipeline_pilot(false,true); % reprocess existing runs
%
% Inputs:
%   runOptimizations - launch/resume the 45 PowerShell runs (default true)
%   saveFigures      - save EPS and PNG previews (default true)

if nargin < 1 || isempty(runOptimizations), runOptimizations = true; end
if nargin < 2 || isempty(saveFigures), saveFigures = true; end
validateattributes(runOptimizations,{'logical','numeric'},{'scalar'});
validateattributes(saveFigures,{'logical','numeric'},{'scalar'});
runOptimizations = logical(runOptimizations);
saveFigures = logical(saveFigures);

paths = setup_project();
budget = 120;
seeds = 0:2;
optimizers = ["GA","PSO","BAYESIAN","ABC","ACO"];
missions = ["LUNAR_GATEWAY","LOW_THRUST_TRANSFER","GATEWAY_IMPULSE"];
studyID = "reviewer2_comparison_pilot_v1";
pilotRoot = fullfile(paths.runs,'COMPARISON_PILOT');

fprintf('\n--- Reviewer 2 full-pipeline pilot ---\n');
fprintf('Target cases:             %d\n',numel(missions));
fprintf('Optimizers:               %d\n',numel(optimizers));
fprintf('Independent seeds:        %s\n',mat2str(seeds));
fprintf('Search FE per run:        %d\n',budget);
fprintf('Expected optimization runs: %d\n', ...
    numel(missions)*numel(optimizers)*numel(seeds));
fprintf(['Pilot values verify the workflow only. Replace them with the ' ...
    'full-study statistics in the manuscript.\n\n']);

% Static checks fail quickly before launching an expensive batch.
test_project_structure();
test_fe_study_configuration();

if runOptimizations
    assert(ispc, ...
        'The supplied batch launcher is a Windows PowerShell script.');
    batchScript = fullfile(paths.root,'scripts','batch', ...
        'run_comparison_soo.ps1');
    assert(isfile(batchScript),'Missing comparison batch script: %s',batchScript);

    escapedScript = strrep(batchScript,'''','''''');
    command = sprintf([ ...
        'powershell.exe -NoProfile -ExecutionPolicy Bypass -Command ' ...
        '"& ''%s'' -Pilot -EvalBudget %d -Seeds @(0,1,2)"'], ...
        escapedScript,budget);

    fprintf('Launching/resuming the comparison pilot...\n');
    fprintf('%s\n\n',command);
    [status,commandOutput] = system(command,'-echo');
    assert(status == 0, ...
        'The pilot batch failed. Review its console output.\n%s', ...
        commandOutput);
end

assert(isfolder(pilotRoot), ...
    'Pilot root does not exist: %s',pilotRoot);

[summary,inventory] = process_fe_convergence( ...
    pilotRoot,studyID,seeds,budget,false,optimizers);

% The dedicated pilot root should contain exactly one complete identity for
% every case/optimizer/seed combination.
nonemptyRuns = inventory.run_file ~= "";
assert(~any(~inventory.valid(nonemptyRuns)), ...
    'One or more saved pilot runs failed validation. Inspect run_inventory.csv.');
assert(height(summary) == numel(missions)*numel(optimizers), ...
    'Expected %d complete case/optimizer groups but found %d.', ...
    numel(missions)*numel(optimizers),height(summary));
assert(all(summary.n_runs == numel(seeds)) && ...
    all(summary.fe_budget == budget), ...
    'The processed pilot does not contain three 120-FE runs per group.');

analysisDir = newest_analysis_directory(pilotRoot);
metricsFile = fullfile(analysisDir,'final_run_metrics.csv');
assert(isfile(metricsFile),'Missing processed run metrics: %s',metricsFile);
runMetrics = readtable(metricsFile, ...
    'TextType','string','VariableNamingRule','preserve');

paperResults = build_paper_results( ...
    summary,runMetrics,missions,optimizers,seeds,budget);
[objectiveTable,trackingTable] = format_paper_tables(paperResults);
bestRuns = select_best_observed_runs(summary,runMetrics,missions);

fprintf('\n--- Pilot objective/runtime results (mean +/- sample std) ---\n');
disp(objectiveTable);
fprintf('\n--- Pilot tracking/design results (mean +/- sample std) ---\n');
disp(trackingTable);
fprintf('\n--- Best observed runs used for trajectory/EKF plots ---\n');
disp(bestRuns(:, ...
    {'Mission','Optimizer','Seed','BestObjective'}));

writetable(paperResults,fullfile(analysisDir,'pilot_paper_results.csv'));
writetable(objectiveTable, ...
    fullfile(analysisDir,'pilot_objective_table_formatted.csv'));
writetable(trackingTable, ...
    fullfile(analysisDir,'pilot_tracking_table_formatted.csv'));
writetable(bestRuns, ...
    fullfile(analysisDir,'pilot_best_observed_runs.csv'));

figureDir = "";
if saveFigures
    figureDir = string(fullfile(analysisDir,'paper_preview'));
    if ~isfolder(figureDir), mkdir(figureDir); end
end

plot_convergence_previews( ...
    analysisDir,figureDir,missions,optimizers,budget,saveFigures);
plot_grouped_result(paperResults,missions,optimizers, ...
    'BestJMean','BestJStd','Final best objective', ...
    'pilot_final_objective',figureDir,saveFigures);
plot_grouped_result(paperResults,missions,optimizers, ...
    'RuntimeMean_s','RuntimeStd_s','Optimization runtime (s)', ...
    'pilot_runtime',figureDir,saveFigures);
plot_cost_component_preview( ...
    paperResults,missions,optimizers,figureDir,saveFigures);
plot_best_run_tracking(bestRuns,figureDir,saveFigures);

report = struct();
report.studyID = studyID;
report.budget = budget;
report.seeds = seeds;
report.expectedRuns = numel(missions)*numel(optimizers)*numel(seeds);
report.analysisDirectory = string(analysisDir);
report.figureDirectory = figureDir;
report.summary = summary;
report.inventory = inventory;
report.paperResults = paperResults;
report.bestObservedRuns = bestRuns;

fprintf('\nFull-pipeline pilot passed.\n');
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

function results = build_paper_results( ...
    summary,runMetrics,missions,optimizers,seeds,budget)

required = ["comparison_key","optimizer","seed","bestJ","search_fe", ...
    "solver_calls","optimization_runtime_s","rmse_pos_km", ...
    "mean_effective_sigma_pos_km","mean_stability", ...
    "coverage_epoch_fraction","screening_count"];
assert(all(ismember(required,string(runMetrics.Properties.VariableNames))), ...
    'Processed metrics are missing fields required by the paper preview.');

nRows = numel(missions)*numel(optimizers);
missionColumn = strings(nRows,1);
optimizerColumn = strings(nRows,1);
nRuns = nan(nRows,1);
searchFE = nan(nRows,1);
solverCallsMean = nan(nRows,1);
solverCallsStd = nan(nRows,1);
bestJMean = nan(nRows,1);
bestJStd = nan(nRows,1);
runtimeMean = nan(nRows,1);
runtimeStd = nan(nRows,1);
rmseMean = nan(nRows,1);
rmseStd = nan(nRows,1);
sigmaMean = nan(nRows,1);
sigmaStd = nan(nRows,1);
stabilityMean = nan(nRows,1);
stabilityStd = nan(nRows,1);
coverageMean = nan(nRows,1);
coverageStd = nan(nRows,1);
screeningMean = nan(nRows,1);
screeningStd = nan(nRows,1);

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
            'Expected seeds %s for %s/%s.', ...
            mat2str(seeds),mission,optimizer);
        assert(all(metricRows.search_fe == budget), ...
            'Search FE mismatch for %s/%s.',mission,optimizer);

        missionColumn(row) = mission;
        optimizerColumn(row) = optimizer;
        nRuns(row) = height(metricRows);
        searchFE(row) = budget;
        [solverCallsMean(row),solverCallsStd(row)] = ...
            sample_statistics(metricRows.solver_calls);
        [bestJMean(row),bestJStd(row)] = ...
            sample_statistics(metricRows.bestJ);
        [runtimeMean(row),runtimeStd(row)] = ...
            sample_statistics(metricRows.optimization_runtime_s);
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
    runtimeMean,runtimeStd,rmseMean,rmseStd,sigmaMean,sigmaStd, ...
    stabilityMean,stabilityStd,coverageMean,coverageStd, ...
    screeningMean,screeningStd, ...
    'VariableNames',{ ...
    'Mission','Optimizer','NRuns','SearchFE', ...
    'SolverCallsMean','SolverCallsStd','BestJMean','BestJStd', ...
    'RuntimeMean_s','RuntimeStd_s','RMSEPosMean_km','RMSEPosStd_km', ...
    'EffectiveSigmaPosMean_km','EffectiveSigmaPosStd_km', ...
    'MeanStabilityMean','MeanStabilityStd', ...
    'CoverageMean','CoverageStd','ScreeningMean','ScreeningStd'});
end

function [objectiveTable,trackingTable] = format_paper_tables(results)
caseName = strings(height(results),1);
for k = 1:height(results)
    caseName(k) = mission_label(results.Mission(k));
end
objectiveTable = table( ...
    caseName,results.Optimizer,results.NRuns,results.SearchFE, ...
    compose('%.4g +/- %.3g',results.SolverCallsMean,results.SolverCallsStd), ...
    compose('%.6g +/- %.3g',results.BestJMean,results.BestJStd), ...
    compose('%.4g +/- %.3g',results.RuntimeMean_s,results.RuntimeStd_s), ...
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
    compose('%.4g +/- %.3g',results.ScreeningMean,results.ScreeningStd), ...
    'VariableNames',{ ...
    'Case','Optimizer','RMSEPosition_km','EffectiveSigmaPosition_km', ...
    'MeanStability','CoverageFraction','ScreeningCount'});
end

function plot_convergence_previews( ...
    analysisDir,figureDir,missions,optimizers,budget,saveFigures)

files = dir(fullfile(analysisDir,'convergence_*.mat'));
assert(numel(files) == numel(missions), ...
    'Expected one convergence data file per target case.');
colors = lines(numel(optimizers));

% All methods are displayed at the same FE checkpoints. Population methods
% do not have a meaningful incumbent before their first batch completes, and
% BO should not receive a visual advantage merely because it records every FE.
checkpointStep = 60;
commonFE = (checkpointStep:checkpointStep:budget)';
if isempty(commonFE) || commonFE(end) ~= budget
    commonFE = unique([commonFE;budget]);
end

for mission = missions
    match = "";
    loaded = struct();
    for k = 1:numel(files)
        candidate = load(fullfile(files(k).folder,files(k).name), ...
            'comparison','curves');
        candidateMission = string(candidate.comparison.settings.mission.type);
        if candidateMission == mission
            match = string(fullfile(files(k).folder,files(k).name));
            loaded = candidate;
            break;
        end
    end
    assert(strlength(match) > 0, ...
        'No convergence data found for %s.',mission);

    fig = create_paper_figure(7.2,4.4, ...
        mission_label(mission)+" convergence");
    ax = axes(fig);
    hold(ax,'on');
    box(ax,'on');
    grid(ax,'on');

    curveOptimizers = upper(string({loaded.curves.optimizer}));
    lineHandles = gobjects(numel(optimizers),1);
    for a = 1:numel(optimizers)
        idx = find(curveOptimizers == optimizers(a),1);
        assert(~isempty(idx),'Missing %s convergence curve.',optimizers(a));
        curve = loaded.curves(idx);

        meanCommon = curve.mean(commonFE);
        stdCommon = curve.std(commonFE);
        validBand = isfinite(meanCommon) & isfinite(stdCommon);
        if any(validBand)
            xBand = commonFE(validBand);
            lowerBand = meanCommon(validBand)-stdCommon(validBand);
            upperBand = meanCommon(validBand)+stdCommon(validBand);
            fill(ax,[xBand;flipud(xBand)],[lowerBand;flipud(upperBand)], ...
                colors(a,:),'FaceAlpha',0.10,'EdgeColor','none', ...
                'HandleVisibility','off');
        end

        validLine = isfinite(meanCommon);
        xLine = commonFE(validLine);
        yLine = meanCommon(validLine);
        markerStep = max(1,round(numel(xLine)/10));
        lineHandles(a) = plot(ax,xLine,yLine,'-o', ...
            'Color',colors(a,:),'LineWidth',2.0,'MarkerSize',4.5, ...
            'MarkerIndices',1:markerStep:numel(xLine), ...
            'DisplayName',optimizers(a));
    end

    xlim(ax,[commonFE(1) commonFE(end)]);
    if budget <= 600
        xticks(ax,commonFE);
    end
    xlabel(ax,'Function evaluations','FontWeight','bold');
    ylabel(ax,'Mean best-so-far objective','FontWeight','bold');
    apply_figure_style(ax);

    lgd = legend(ax,lineHandles,cellstr(optimizers), ...
        'Location','northoutside','Orientation','horizontal', ...
        'NumColumns',numel(optimizers),'Box','on');
    format_legend(lgd,11);
    reserve_top_legend_space(ax);

    if saveFigures
        stem = fullfile(figureDir, ...
            "pilot_convergence_"+mission_code(mission));
        export_pilot_figure(fig,stem);
    end
end
end

function plot_grouped_result(results,missions,optimizers, ...
    meanVariable,stdVariable,yLabel,fileStem,figureDir,saveFigures)

fig = create_paper_figure(7.4,4.35,string(fileStem));
ax = axes(fig);
hold(ax,'on');
box(ax,'on');
grid(ax,'on');

[values,errors,missionLabels] = grouped_metric_arrays( ...
    results,missions,optimizers,meanVariable,stdVariable);

bars = bar(ax,values,'grouped','LineWidth',0.9);
colors = lines(numel(optimizers));
for a = 1:numel(optimizers)
    bars(a).FaceColor = colors(a,:);
    bars(a).DisplayName = optimizers(a);
    errorbar(ax,bars(a).XEndPoints,values(:,a),errors(:,a), ...
        'k.','LineWidth',1.15,'CapSize',7,'HandleVisibility','off');
end

xticks(ax,1:numel(missions));
xticklabels(ax,missionLabels);
xtickangle(ax,0);
ylabel(ax,yLabel,'FontWeight','bold');
apply_figure_style(ax);

lgd = legend(ax,bars,cellstr(optimizers), ...
    'Location','northoutside','Orientation','horizontal', ...
    'NumColumns',numel(optimizers),'Box','on');
format_legend(lgd,11);
reserve_top_legend_space(ax);

if saveFigures
    export_pilot_figure(fig,fullfile(figureDir,fileStem));
end
end

function plot_cost_component_preview( ...
    results,missions,optimizers,figureDir,saveFigures)

plot_tracking_metric_by_case(results,missions,optimizers, ...
    "RMSEPosMean_km","RMSEPosStd_km","Position RMSE (km)", ...
    "pilot_rmse_by_case",figureDir,saveFigures);
plot_tracking_metric_by_case(results,missions,optimizers, ...
    "EffectiveSigmaPosMean_km","EffectiveSigmaPosStd_km", ...
    "Effective position uncertainty (km)", ...
    "pilot_effective_sigma_by_case",figureDir,saveFigures);
plot_tracking_metric_by_case(results,missions,optimizers, ...
    "MeanStabilityMean","MeanStabilityStd","Mean stability index", ...
    "pilot_mean_stability_by_case",figureDir,saveFigures);
end

function plot_tracking_metric_by_case(results,missions,optimizers, ...
    meanVariable,stdVariable,yLabel,fileStem,figureDir,saveFigures)

fig = create_paper_figure(7.5,3.65,string(fileStem));
layout = tiledlayout(fig,1,numel(missions), ...
    'TileSpacing','compact','Padding','compact');
colors = lines(numel(optimizers));
legendBars = gobjects(numel(optimizers),1);

for m = 1:numel(missions)
    ax = nexttile(layout);
    hold(ax,'on');
    box(ax,'on');
    grid(ax,'on');

    rows = results(results.Mission == missions(m),:);
    [found,order] = ismember(optimizers,rows.Optimizer);
    assert(all(found),'Missing optimizer result for %s.',missions(m));
    rows = rows(order,:);

    values = rows.(meanVariable);
    deviations = rows.(stdVariable);
    bars = bar(ax,1:numel(optimizers),values,0.72,'LineWidth',0.85);
    bars.FaceColor = 'flat';
    bars.CData = colors;
    hold(ax,'on');
    errorbar(ax,1:numel(optimizers),values,deviations, ...
        'k.','LineWidth',1.0,'CapSize',6,'HandleVisibility','off');

    % Each mission receives an independent y-scale so a large low-thrust
    % value cannot visually flatten the Gateway or impulse bars.
    ymax = max(values+max(deviations,0),[],'omitnan');
    if isfinite(ymax) && ymax > 0
        ylim(ax,[0,1.10*ymax]);
    end

    xticks(ax,1:numel(optimizers));
    xticklabels(ax,[]);
    xlabel(ax,mission_label(missions(m)),'FontWeight','bold');
    if m == 1
        ylabel(ax,yLabel,'FontWeight','bold');
        for a = 1:numel(optimizers)
            legendBars(a) = bar(ax,nan,nan,0.72, ...
                'FaceColor',colors(a,:),'EdgeColor','none', ...
                'DisplayName',optimizers(a));
        end
    end
    apply_figure_style(ax);
end

lgd = legend(legendBars,cellstr(optimizers), ...
    'Orientation','horizontal','NumColumns',numel(optimizers),'Box','on');
lgd.Layout.Tile = 'north';
format_legend(lgd,11);

if saveFigures
    export_pilot_figure(fig,fullfile(figureDir,fileStem));
end
end

function [values,errors,missionLabels] = grouped_metric_arrays( ...
    results,missions,optimizers,meanVariable,stdVariable)

values = nan(numel(missions),numel(optimizers));
errors = nan(size(values));
missionLabels = strings(numel(missions),1);
for m = 1:numel(missions)
    rows = results(results.Mission == missions(m),:);
    [found,order] = ismember(optimizers,rows.Optimizer);
    assert(all(found),'Missing optimizer result for %s.',missions(m));
    rows = rows(order,:);
    values(m,:) = rows.(meanVariable).';
    errors(m,:) = rows.(stdVariable).';
    missionLabels(m) = mission_label(missions(m));
end
end

function bestRuns = select_best_observed_runs(summary,runMetrics,missions)
missionColumn = strings(numel(missions),1);
optimizerColumn = strings(numel(missions),1);
seedColumn = nan(numel(missions),1);
objectiveColumn = nan(numel(missions),1);
optimizationFile = strings(numel(missions),1);
trackingFile = strings(numel(missions),1);

for k = 1:numel(missions)
    mission = missions(k);
    keys = unique(summary.comparison_key(summary.mission == mission));
    assert(numel(keys) == 1, ...
        'Expected one comparison identity for %s.',mission);

    candidates = runMetrics(runMetrics.comparison_key == keys,:);
    assert(~isempty(candidates), ...
        'No completed runs are available for %s.',mission);
    candidates = sortrows(candidates,{'bestJ','optimizer','seed'});
    selected = candidates(1,:);

    optimizationPath = string(selected.run_file);
    trackingPath = string(fullfile(fileparts(optimizationPath), ...
        'tracking_data.mat'));
    assert(isfile(optimizationPath) && isfile(trackingPath), ...
        'The selected best run is missing saved tracking data.');

    missionColumn(k) = mission;
    optimizerColumn(k) = selected.optimizer;
    seedColumn(k) = selected.seed;
    objectiveColumn(k) = selected.bestJ;
    optimizationFile(k) = optimizationPath;
    trackingFile(k) = trackingPath;
end

bestRuns = table( ...
    missionColumn,optimizerColumn,seedColumn,objectiveColumn, ...
    optimizationFile,trackingFile, ...
    'VariableNames',{ ...
    'Mission','Optimizer','Seed','BestObjective', ...
    'OptimizationRunFile','TrackingDataFile'});
end

function plot_best_run_tracking(bestRuns,figureDir,saveFigures)
for k = 1:height(bestRuns)
    stateData = load(bestRuns.OptimizationRunFile(k),'runState');
    trackingData = load(bestRuns.TrackingDataFile(k),'tracking');
    runState = stateData.runState;
    tracking = trackingData.tracking;

    assert(size(tracking.truth,1) == numel(tracking.t_TU) && ...
        isequal(size(tracking.truth),size(tracking.estimate)), ...
        'Tracking truth and estimate arrays are inconsistent.');
    assert(size(tracking.covariance,1) == numel(tracking.t_TU), ...
        'Tracking covariance length is inconsistent.');

    trajectoryFig = plot_best_trajectory( ...
        runState,tracking,bestRuns(k,:));
    errorFig = plot_best_ekf_errors( ...
        runState,tracking,bestRuns(k,:));

    if saveFigures
        code = mission_code(bestRuns.Mission(k));
        export_pilot_figure(trajectoryFig, ...
            fullfile(figureDir,"pilot_best_trajectory_"+code));
        export_pilot_figure(errorFig, ...
            fullfile(figureDir,"pilot_best_ekf_errors_"+code));
    end
end
end

function fig = plot_best_trajectory(runState,tracking,bestRun)
truth = tracking.truth(:,1:3);
estimate = tracking.estimate(:,1:3);
mu = runState.settings.mu;
LU = runState.settings.LU;
moonCenter = [1-mu,0,0];
moonRadius = 1737.1/LU;
[xL1,xL2] = collinear_lagrange_points(mu);

% Match the manuscript study-definition target-case figures exactly:
% 7.2 x 6.5 in, perspective projection, view(-37.5,30).
fig = create_paper_figure(7.2,6.2, ...
    mission_label(bestRun.Mission)+" best-run trajectory");
ax = axes(fig);
hold(ax,'on');
box(ax,'on');
axis(ax,'equal');

hTruth = plot3(ax,truth(:,1),truth(:,2),truth(:,3), ...
    'Color',[0.85 0.20 0.15],'LineWidth',2.6, ...
    'DisplayName','Truth trajectory');
hEstimate = plot3(ax,estimate(:,1),estimate(:,2),estimate(:,3), ...
    '--','Color',[0.05 0.35 0.80],'LineWidth',2.0, ...
    'DisplayName','EKF estimate');

[sx,sy,sz] = sphere(30);
hMoon = surf(ax,moonCenter(1)+moonRadius*sx, ...
    moonCenter(2)+moonRadius*sy,moonCenter(3)+moonRadius*sz, ...
    'FaceColor',[0.72 0.72 0.72], ...
    'EdgeColor','none','FaceLighting','gouraud', ...
    'DisplayName','Moon');
camlight(ax,'headlight');
material(ax,'dull');

hL1 = plot3(ax,xL1,0,0,'^','MarkerSize',9, ...
    'MarkerFaceColor',[0.80 0.80 0.80], ...
    'MarkerEdgeColor','k','LineWidth',1.1, ...
    'DisplayName','L1');
hL2 = plot3(ax,xL2,0,0,'v','MarkerSize',9, ...
    'MarkerFaceColor',[0.80 0.80 0.80], ...
    'MarkerEdgeColor','k','LineWidth',1.1, ...
    'DisplayName','L2');
hStart = plot3(ax,truth(1,1),truth(1,2),truth(1,3),'o', ...
    'MarkerSize',8,'MarkerFaceColor',[0.20 0.70 0.25], ...
    'MarkerEdgeColor','k','LineWidth',1.0,'DisplayName','Start');
hEnd = plot3(ax,truth(end,1),truth(end,2),truth(end,3),'s', ...
    'MarkerSize',8,'MarkerFaceColor',[0.20 0.35 0.90], ...
    'MarkerEdgeColor','k','LineWidth',1.0,'DisplayName','End');

allPoints = [truth;estimate;moonCenter; ...
    moonCenter+[moonRadius 0 0];moonCenter-[moonRadius 0 0]; ...
    moonCenter+[0 moonRadius 0];moonCenter-[0 moonRadius 0]; ...
    moonCenter+[0 0 moonRadius];moonCenter-[0 0 moonRadius]; ...
    xL1 0 0;xL2 0 0];
xlim(ax,padded_limits(allPoints(:,1),0.08));
ylim(ax,padded_limits(allPoints(:,2),0.10));
zlim(ax,padded_limits(allPoints(:,3),0.10));
axis(ax,'vis3d');
ax.Projection = 'perspective';
view(ax,-37.5,30);
grid(ax,'off');

xlabel(ax,'x (LU)','FontWeight','bold');
ylabel(ax,'y (LU)','FontWeight','bold');
zlabel(ax,'z (LU)','FontWeight','bold');
apply_figure_style(ax);

lgd = legend(ax,[hTruth hEstimate hMoon hL1 hL2 hStart hEnd], ...
    {'Truth trajectory','EKF estimate','Moon','L1','L2','Start','End'}, ...
    'Location','northoutside','Orientation','horizontal', ...
    'NumColumns',4,'Box','on');
format_legend(lgd,11);
ax.PositionConstraint = 'outerposition';
ax.OuterPosition = [0.02 0.04 0.96 0.84];
drawnow;
ax.LooseInset = max(ax.TightInset,0.035);
end

function fig = plot_best_ekf_errors(runState,tracking,bestRun)
t = tracking.t_TU(:);
errors = tracking.estimate-tracking.truth;
N = numel(t);
LU = runState.settings.LU;
VU = LU/runState.settings.TU;
nObservers = runState.settings.mission.optimization.numObservers;

sigma = zeros(N,6);
for k = 1:N
    covariance = squeeze(tracking.covariance(k,:,:));
    covariance = (covariance+covariance')/2;
    sigma(k,:) = sqrt(max(real(diag(covariance)),0))';
end

errors(:,1:3) = errors(:,1:3)*LU;
errors(:,4:6) = errors(:,4:6)*VU;
sigma(:,1:3) = sigma(:,1:3)*LU;
sigma(:,4:6) = sigma(:,4:6)*VU;
bounds = 3*sigma;

available = tracking.availableObsCount(:);
assert(numel(available) == N, ...
    'Available-observer history length is inconsistent.');
if all(isnan(available))
    available = zeros(N,1);
else
    available = fillmissing(available,'previous');
    available = fillmissing(available,'next');
end
available = max(0,min(nObservers,available));

fig = create_paper_figure(7.5,5.55, ...
    mission_label(bestRun.Mission)+" best-run EKF errors");
layout = tiledlayout(fig,2,3, ...
    'TileSpacing','compact','Padding','compact');
axesHandles = gobjects(6,1);
coordinateNames = ["x","y","z","v_x","v_y","v_z"];
hErrorLegend = gobjects(1);
hBoundLegend = gobjects(1);

for component = 1:6
    ax = nexttile(layout);
    axesHandles(component) = ax;
    hold(ax,'on');
    box(ax,'on');

    yLimits = padded_limits([ ...
        errors(:,component);bounds(:,component);-bounds(:,component)],0.06);
    availabilityImage = image(ax, ...
        'XData',[t(1) t(end)], ...
        'YData',yLimits, ...
        'CData',[available';available'], ...
        'CDataMapping','scaled');
    availabilityImage.AlphaData = 1.0;
    set(ax,'YDir','normal');
    clim(ax,[0 max(1,nObservers)]);

    hError = plot(ax,t,errors(:,component), ...
        'Color',[0.12 0.40 0.68],'LineWidth',1.25, ...
        'DisplayName','EKF error');
    hBound = plot(ax,t,bounds(:,component), ...
        'Color',[0.75 0.18 0.13],'LineWidth',1.45, ...
        'DisplayName','+/- 3 sigma');
    plot(ax,t,-bounds(:,component), ...
        'Color',[0.75 0.18 0.13],'LineWidth',1.45, ...
        'HandleVisibility','off');
    if component == 1
        hErrorLegend = hError;
        hBoundLegend = hBound;
    end

    xlim(ax,[t(1) t(end)]);
    ylim(ax,yLimits);
    grid(ax,'on');
    if component <= 3
        ylabel(ax,sprintf('$e_{%s}$ (km)',coordinateNames(component)), ...
            'Interpreter','latex','FontWeight','bold');
    else
        ylabel(ax,sprintf('$e_{%s}$ (km/s)',coordinateNames(component)), ...
            'Interpreter','latex','FontWeight','bold');
        xlabel(ax,'t (TU)','FontWeight','bold');
    end
    apply_figure_style(ax);
end

availabilityGray = linspace(0.76,1.0,256)';
colormap(fig,repmat(availabilityGray,1,3));
colorbarHandle = colorbar(axesHandles(end));
colorbarHandle.Layout.Tile = 'east';
colorbarHandle.Ticks = 0:nObservers;
colorbarHandle.Label.String = 'Available observers';
colorbarHandle.Label.FontWeight = 'bold';
colorbarHandle.FontName = 'Times New Roman';
colorbarHandle.FontSize = 11;
colorbarHandle.FontWeight = 'bold';

lgd = legend([hErrorLegend hBoundLegend],{'EKF error','+/- 3 sigma'}, ...
    'Orientation','horizontal','NumColumns',2,'Box','on');
lgd.Layout.Tile = 'north';
format_legend(lgd,11);
end

function [xL1,xL2] = collinear_lagrange_points(mu)
equilibrium = @(x) x ...
    -(1-mu)*(x+mu)./abs(x+mu).^3 ...
    -mu*(x-1+mu)./abs(x-1+mu).^3;
xL1 = fzero(equilibrium,1-mu-0.15);
xL2 = fzero(equilibrium,1-mu+0.15);
end

function limits = padded_limits(values,fraction)
values = values(isfinite(values));
assert(~isempty(values),'Cannot size axes from empty data.');
lowerValue = min(values);
upperValue = max(values);
span = upperValue-lowerValue;
if span <= 100*eps(max(1,max(abs(values))))
    span = max(0.02,0.05*max(1,abs(mean(values))));
end
padding = fraction*span;
limits = [lowerValue-padding,upperValue+padding];
end

function fig = create_paper_figure(width,height,name)
fig = figure('Color','w','Name',char(name), ...
    'Units','inches','Position',[1 1 width height], ...
    'PaperUnits','inches','PaperSize',[width height], ...
    'PaperPosition',[0 0 width height], ...
    'PaperPositionMode','manual','Renderer','painters', ...
    'InvertHardcopy','off');
movegui(fig,'center');
end

function apply_figure_style(ax)
set(ax,'FontName','Times New Roman','FontSize',12, ...
    'FontWeight','bold','LineWidth',1.2,'Layer','top');
ax.XLabel.FontSize = 14;
ax.YLabel.FontSize = 14;
ax.ZLabel.FontSize = 14;
end

function format_legend(lgd,fontSize)
lgd.FontName = 'Times New Roman';
lgd.FontSize = fontSize;
lgd.FontWeight = 'bold';
lgd.ItemTokenSize = [16 9];
end

function reserve_top_legend_space(ax)
ax.Units = 'normalized';
ax.PositionConstraint = 'outerposition';
ax.OuterPosition = [0.04 0.06 0.92 0.82];
drawnow;
ax.LooseInset = max(ax.TightInset,0.025);
end

function export_pilot_figure(fig,stem)
drawnow;
stem = string(stem);
oldUnits = fig.Units;
fig.Units = 'inches';
position = fig.Position;
fig.PaperUnits = 'inches';
fig.PaperSize = position(3:4);
fig.PaperPosition = [0 0 position(3:4)];
fig.PaperPositionMode = 'manual';
fig.Units = oldUnits;
print(fig,char(stem+".eps"),'-depsc','-painters');
exportgraphics(fig,char(stem+".png"),'Resolution',300);
end

function [average,deviation] = sample_statistics(values)
values = values(isfinite(values));
assert(~isempty(values),'Cannot summarize an empty metric.');
average = mean(values);
if numel(values) < 2
    deviation = NaN;
else
    deviation = std(values,0);
end
end

function label = mission_label(mission)
switch string(mission)
    case "LUNAR_GATEWAY"
        label = "Lunar Gateway";
    case "LOW_THRUST_TRANSFER"
        label = "Low-thrust transfer";
    case "GATEWAY_IMPULSE"
        label = "Gateway impulse";
    otherwise
        label = string(mission);
end
end

function code = mission_code(mission)
switch string(mission)
    case "LUNAR_GATEWAY"
        code = "lg";
    case "LOW_THRUST_TRANSFER"
        code = "lt";
    case "GATEWAY_IMPULSE"
        code = "gi";
    otherwise
        code = lower(string(mission));
end
end
