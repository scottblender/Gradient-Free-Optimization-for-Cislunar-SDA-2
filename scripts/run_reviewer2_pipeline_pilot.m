function report = run_reviewer2_pipeline_pilot(runOptimizations,saveFigures)
%RUN_REVIEWER2_PIPELINE_PILOT End-to-end three-seed FE study check.
%
% This is an integration pilot, not a quick unit test. By default it runs:
%   5 optimizers x 3 fixed target cases x 3 seeds = 45 optimization runs
%   120 search function evaluations per run
%
% The script then validates the saved schema, aligns convergence by function
% evaluations, prints mean +/- sample-standard-deviation tables, and creates
% paper-style preview figures. Pilot output is isolated from the full study.
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

fprintf('\n--- Pilot objective/runtime results (mean +/- sample std) ---\n');
disp(objectiveTable);
fprintf('\n--- Pilot tracking/design results (mean +/- sample std) ---\n');
disp(trackingTable);

writetable(paperResults,fullfile(analysisDir,'pilot_paper_results.csv'));
writetable(objectiveTable, ...
    fullfile(analysisDir,'pilot_objective_table_formatted.csv'));
writetable(trackingTable, ...
    fullfile(analysisDir,'pilot_tracking_table_formatted.csv'));

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

    fig = figure('Color','w','Units','inches', ...
        'Position',[1 1 6.5 4.25]);
    ax = axes(fig);
    hold(ax,'on');
    box(ax,'on');
    grid(ax,'on');

    curveOptimizers = upper(string({loaded.curves.optimizer}));
    for a = 1:numel(optimizers)
        idx = find(curveOptimizers == optimizers(a),1);
        assert(~isempty(idx),'Missing %s convergence curve.',optimizers(a));
        curve = loaded.curves(idx);
        validBand = isfinite(curve.mean) & isfinite(curve.std);
        if any(validBand)
            xBand = curve.fe(validBand);
            lower = curve.mean(validBand)-curve.std(validBand);
            upper = curve.mean(validBand)+curve.std(validBand);
            fill(ax,[xBand;flipud(xBand)],[lower;flipud(upper)], ...
                colors(a,:),'FaceAlpha',0.12,'EdgeColor','none', ...
                'HandleVisibility','off');
        end
        stairs(ax,curve.fe,curve.mean,'Color',colors(a,:), ...
            'LineWidth',1.8,'DisplayName',optimizers(a));
    end

    xlim(ax,[1 budget]);
    xlabel(ax,'Function evaluations','FontWeight','bold');
    ylabel(ax,'Mean best-so-far objective','FontWeight','bold');
    title(ax,sprintf('%s: three-seed pilot',mission_label(mission)), ...
        'FontWeight','bold');
    legend(ax,'Location','best','FontSize',12,'FontWeight','bold');
    apply_figure_style(ax);

    if saveFigures
        stem = fullfile(figureDir, ...
            "pilot_convergence_"+mission_code(mission));
        export_pilot_figure(fig,stem);
    end
end
end

function plot_grouped_result(results,missions,optimizers, ...
    meanVariable,stdVariable,yLabel,fileStem,figureDir,saveFigures)

fig = figure('Color','w','Units','inches','Position',[1 1 7.5 3.5]);
layout = tiledlayout(fig,1,numel(missions), ...
    'TileSpacing','compact','Padding','compact');
colors = lines(numel(optimizers));

for m = 1:numel(missions)
    ax = nexttile(layout);
    rows = results(results.Mission == missions(m),:);
    [~,order] = ismember(optimizers,rows.Optimizer);
    rows = rows(order,:);

    values = rows.(meanVariable);
    errors = rows.(stdVariable);
    bars = bar(ax,1:numel(optimizers),values,0.72,'FaceColor','flat');
    bars.CData = colors;
    hold(ax,'on');
    errorbar(ax,1:numel(optimizers),values,errors, ...
        'k.','LineWidth',1.2,'CapSize',7);
    grid(ax,'on');
    box(ax,'on');
    xticks(ax,1:numel(optimizers));
    xticklabels(ax,optimizers);
    xtickangle(ax,25);
    title(ax,mission_label(missions(m)),'FontWeight','bold');
    if m == 1
        ylabel(ax,yLabel,'FontWeight','bold');
    end
    apply_figure_style(ax);
end
title(layout,'Three-seed, 120-FE pilot','FontWeight','bold','FontSize',13);

if saveFigures
    export_pilot_figure(fig,fullfile(figureDir,fileStem));
end
end

function plot_cost_component_preview( ...
    results,missions,optimizers,figureDir,saveFigures)

variables = ["RMSEPosMean_km","EffectiveSigmaPosMean_km","MeanStabilityMean"];
errors = ["RMSEPosStd_km","EffectiveSigmaPosStd_km","MeanStabilityStd"];
labels = ["Position RMSE (km)", ...
    "Effective position uncertainty (km)","Mean stability index"];
colors = lines(numel(optimizers));

fig = figure('Color','w','Units','inches','Position',[1 1 7.5 6.2]);
layout = tiledlayout(fig,numel(variables),numel(missions), ...
    'TileSpacing','compact','Padding','compact');

for v = 1:numel(variables)
    for m = 1:numel(missions)
        ax = nexttile(layout);
        rows = results(results.Mission == missions(m),:);
        [~,order] = ismember(optimizers,rows.Optimizer);
        rows = rows(order,:);

        values = rows.(variables(v));
        deviations = rows.(errors(v));
        bars = bar(ax,1:numel(optimizers),values,0.72,'FaceColor','flat');
        bars.CData = colors;
        hold(ax,'on');
        errorbar(ax,1:numel(optimizers),values,deviations, ...
            'k.','LineWidth',1.1,'CapSize',6);
        grid(ax,'on');
        box(ax,'on');
        xticks(ax,1:numel(optimizers));
        xticklabels(ax,optimizers);
        xtickangle(ax,25);
        if v == 1
            title(ax,mission_label(missions(m)),'FontWeight','bold');
        end
        if m == 1
            ylabel(ax,labels(v),'FontWeight','bold');
        end
        apply_figure_style(ax);
    end
end
title(layout,'Pilot objective components and design stability', ...
    'FontWeight','bold','FontSize',13);

if saveFigures
    export_pilot_figure(fig, ...
        fullfile(figureDir,'pilot_cost_components'));
end
end

function apply_figure_style(ax)
set(ax,'FontName','Times New Roman','FontSize',12, ...
    'FontWeight','bold','LineWidth',1.0);
end

function export_pilot_figure(fig,stem)
drawnow;
stem = string(stem);
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
