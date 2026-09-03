function report = run_reviewer2_runtime_pipeline(saveFigures)
%RUN_REVIEWER2_RUNTIME_PIPELINE Process the completed 1200-FE runtime study.
%
% This pipeline does not launch optimizations. It processes the focused
% Reviewer 2 runtime/scaling study:
%   5 optimizers x 20 seeds = 100 optimization runs
%   Lunar Gateway, angles only, 3 observers, 1 period
%   1200 admitted search function evaluations per run
%
% It validates the saved run schema and FE accounting, creates aggregate
% mean +/- sample-standard-deviation tables, quantifies the Bayesian runtime
% penalty relative to the other methods, and creates paper-style previews
% for convergence, final objective, and equal-budget optimization runtime.
%
% Usage:
%   report = run_reviewer2_runtime_pipeline;
%   report = run_reviewer2_runtime_pipeline(false); % process, display only
%
% Input:
%   saveFigures - save EPS and PNG previews (default true)

if nargin < 1 || isempty(saveFigures), saveFigures = true; end
validateattributes(saveFigures,{'logical','numeric'},{'scalar'});
saveFigures = logical(saveFigures);

paths = setup_project();
budget = 1200;
seeds = 0:19;
optimizers = ["GA","PSO","BAYESIAN","ABC","ACO"];
mission = "LUNAR_GATEWAY";
studyID = "reviewer2_runtime_comparison_1200_v1";
runtimeRoot = fullfile(paths.runs,'RUNTIME_COMPARISON_1200');
expectedRuns = numel(optimizers)*numel(seeds);

fprintf('\n--- Reviewer 2 1200-FE runtime pipeline ---\n');
fprintf('Mission:                    %s\n',mission);
fprintf('Measurement model:          ANGLES_ONLY\n');
fprintf('Observers:                  3\n');
fprintf('Gateway periods:            1\n');
fprintf('Optimizers:                 %d\n',numel(optimizers));
fprintf('Independent seeds:          %s\n',mat2str(seeds));
fprintf('Search FE per run:          %d\n',budget);
fprintf('Expected optimization runs: %d\n\n',expectedRuns);

% Static study-definition checks fail before any result processing.
test_project_structure();
test_fe_study_configuration();

assert(isfolder(runtimeRoot), ...
    'Runtime study root does not exist: %s',runtimeRoot);

[summary,inventory] = process_fe_convergence( ...
    runtimeRoot,studyID,seeds,budget,false,optimizers);

% The dedicated runtime-study root must contain exactly one complete run for
% every optimizer/seed identity and exactly one comparison configuration.
nonemptyRuns = inventory.run_file ~= "";
assert(sum(nonemptyRuns) == expectedRuns, ...
    'Expected %d saved optimization runs but found %d.', ...
    expectedRuns,sum(nonemptyRuns));
assert(~any(~inventory.valid), ...
    'One or more runtime-study runs failed validation. Inspect run_inventory.csv.');
assert(height(summary) == numel(optimizers), ...
    'Expected %d optimizer summary rows but found %d.', ...
    numel(optimizers),height(summary));
assert(numel(unique(summary.comparison_key)) == 1, ...
    'The runtime study contains more than one comparison configuration.');
assert(all(summary.mission == mission) && ...
    all(summary.measurement == "ANGLES_ONLY") && ...
    all(summary.num_observers == 3) && ...
    all(summary.n_runs == numel(seeds)) && ...
    all(summary.fe_budget == budget), ...
    'Processed runtime-study metadata does not match the intended design.');

analysisDir = newest_analysis_directory(runtimeRoot);
metricsFile = fullfile(analysisDir,'final_run_metrics.csv');
assert(isfile(metricsFile),'Missing processed run metrics: %s',metricsFile);
runMetrics = readtable(metricsFile, ...
    'TextType','string','VariableNamingRule','preserve');
assert(height(runMetrics) == expectedRuns, ...
    'Expected %d processed run-metric rows but found %d.', ...
    expectedRuns,height(runMetrics));

requiredMetrics = [ ...
    "comparison_key","optimizer","seed","bestJ","search_fe", ...
    "solver_calls","parallel_overflow_evals", ...
    "optimization_runtime_s","budget_runtime_s", ...
    "solver_wall_runtime_s","validation_runtime_s"];
assert(all(ismember(requiredMetrics,string(runMetrics.Properties.VariableNames))), ...
    'Processed runtime-study metrics are missing required fields.');
assert(all(runMetrics.search_fe == budget), ...
    'One or more processed runs do not report exactly 1200 admitted FE.');

runtimeResults = build_runtime_results( ...
    runMetrics,optimizers,seeds,budget);
formattedTable = format_runtime_table(runtimeResults);
slowdownTable = build_bo_slowdown(runtimeResults);
boDetails = build_bo_details(runMetrics,seeds,budget);

fprintf('\n--- 1200-FE objective/runtime results (mean +/- sample std) ---\n');
disp(formattedTable);
fprintf('\n--- Bayesian runtime ratio ---\n');
fprintf(['BOTimeRatio = mean BO equal-budget runtime / mean runtime of ' ...
    'the listed optimizer.\n']);
disp(slowdownTable);
fprintf('\n--- Bayesian FE/overflow audit ---\n');
disp(boDetails);

writetable(runtimeResults, ...
    fullfile(analysisDir,'runtime_comparison_1200_results.csv'));
writetable(formattedTable, ...
    fullfile(analysisDir,'runtime_comparison_1200_formatted.csv'));
writetable(slowdownTable, ...
    fullfile(analysisDir,'runtime_comparison_1200_bo_slowdown.csv'));
writetable(boDetails, ...
    fullfile(analysisDir,'runtime_comparison_1200_bo_audit.csv'));

figureDir = "";
if saveFigures
    figureDir = string(fullfile(analysisDir,'paper_preview'));
    if ~isfolder(figureDir), mkdir(figureDir); end
end

plot_runtime_convergence( ...
    analysisDir,optimizers,budget,figureDir,saveFigures);
plot_runtime_bar( ...
    runtimeResults,optimizers,figureDir,saveFigures);
plot_objective_bar( ...
    runtimeResults,optimizers,figureDir,saveFigures);

report = struct();
report.studyID = studyID;
report.budget = budget;
report.seeds = seeds;
report.optimizers = optimizers;
report.expectedRuns = expectedRuns;
report.analysisDirectory = string(analysisDir);
report.figureDirectory = figureDir;
report.summary = summary;
report.inventory = inventory;
report.runMetrics = runMetrics;
report.runtimeResults = runtimeResults;
report.formattedTable = formattedTable;
report.boSlowdown = slowdownTable;
report.boAudit = boDetails;

fprintf('\nReviewer 2 runtime pipeline passed.\n');
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


function results = build_runtime_results(runMetrics,optimizers,seeds,budget)

n = numel(optimizers);
optimizerColumn = strings(n,1);
nRuns = nan(n,1);
searchFE = repmat(budget,n,1);
bestJMean = nan(n,1);
bestJStd = nan(n,1);
budgetRuntimeMean = nan(n,1);
budgetRuntimeStd = nan(n,1);
solverWallMean = nan(n,1);
solverWallStd = nan(n,1);
solverCallsMean = nan(n,1);
solverCallsStd = nan(n,1);
overflowMean = nan(n,1);
overflowStd = nan(n,1);
overflowMax = nan(n,1);

for k = 1:n
    optimizer = optimizers(k);
    rows = runMetrics(runMetrics.optimizer == optimizer,:);
    rows = sortrows(rows,'seed');
    assert(height(rows) == numel(seeds) && ...
        isequal(rows.seed(:)',seeds), ...
        'Expected seeds %s for %s.',mat2str(seeds),optimizer);
    assert(all(rows.search_fe == budget), ...
        'Search FE mismatch for %s.',optimizer);

    tolerance = 1e-10*max(ones(height(rows),1),abs(rows.budget_runtime_s));
    assert(all(abs(rows.optimization_runtime_s-rows.budget_runtime_s) <= tolerance), ...
        'Optimization runtime is not the equal-budget runtime for %s.',optimizer);
    assert(all(rows.solver_wall_runtime_s >= rows.budget_runtime_s), ...
        'Solver wall runtime precedes budget runtime for %s.',optimizer);

    if optimizer == "BAYESIAN"
        assert(all(rows.parallel_overflow_evals >= 0), ...
            'Invalid Bayesian parallel-overflow count.');
    else
        assert(all(rows.parallel_overflow_evals == 0), ...
            'Unexpected parallel overflow for %s.',optimizer);
    end

    optimizerColumn(k) = optimizer;
    nRuns(k) = height(rows);
    [bestJMean(k),bestJStd(k)] = sample_statistics(rows.bestJ);
    [budgetRuntimeMean(k),budgetRuntimeStd(k)] = ...
        sample_statistics(rows.budget_runtime_s);
    [solverWallMean(k),solverWallStd(k)] = ...
        sample_statistics(rows.solver_wall_runtime_s);
    [solverCallsMean(k),solverCallsStd(k)] = ...
        sample_statistics(rows.solver_calls);
    [overflowMean(k),overflowStd(k)] = ...
        sample_statistics(rows.parallel_overflow_evals);
    overflowMax(k) = max(rows.parallel_overflow_evals);
end

results = table( ...
    optimizerColumn,nRuns,searchFE,bestJMean,bestJStd, ...
    budgetRuntimeMean,budgetRuntimeStd,solverWallMean,solverWallStd, ...
    solverCallsMean,solverCallsStd,overflowMean,overflowStd,overflowMax, ...
    'VariableNames',{ ...
    'Optimizer','NRuns','SearchFE','BestJMean','BestJStd', ...
    'BudgetRuntimeMean_s','BudgetRuntimeStd_s', ...
    'SolverWallRuntimeMean_s','SolverWallRuntimeStd_s', ...
    'SolverCallsMean','SolverCallsStd', ...
    'ParallelOverflowMean','ParallelOverflowStd','ParallelOverflowMax'});
end


function formatted = format_runtime_table(results)
formatted = table( ...
    results.Optimizer,results.NRuns,results.SearchFE, ...
    compose('%.6g +/- %.3g',results.BestJMean,results.BestJStd), ...
    compose('%.5g +/- %.3g', ...
        results.BudgetRuntimeMean_s,results.BudgetRuntimeStd_s), ...
    compose('%.5g +/- %.3g', ...
        results.SolverWallRuntimeMean_s,results.SolverWallRuntimeStd_s), ...
    compose('%.5g +/- %.3g', ...
        results.SolverCallsMean,results.SolverCallsStd), ...
    compose('%.3g +/- %.3g', ...
        results.ParallelOverflowMean,results.ParallelOverflowStd), ...
    'VariableNames',{ ...
    'Optimizer','Runs','SearchFE','BestObjective', ...
    'BudgetRuntime_s','SolverWallRuntime_s','SolverCalls', ...
    'ParallelOverflow'});
end


function slowdown = build_bo_slowdown(results)
idxBO = results.Optimizer == "BAYESIAN";
assert(sum(idxBO) == 1,'Expected exactly one Bayesian summary row.');
boRuntime = results.BudgetRuntimeMean_s(idxBO);
ratio = boRuntime ./ results.BudgetRuntimeMean_s;
relativeToFastest = results.BudgetRuntimeMean_s ./ ...
    min(results.BudgetRuntimeMean_s);
slowdown = table( ...
    results.Optimizer,results.BudgetRuntimeMean_s,ratio,relativeToFastest, ...
    'VariableNames',{ ...
    'Optimizer','BudgetRuntimeMean_s','BOTimeRatio','RuntimeVsFastest'});
end


function boDetails = build_bo_details(runMetrics,seeds,budget)
boDetails = runMetrics(runMetrics.optimizer == "BAYESIAN", ...
    {'seed','search_fe','solver_calls','parallel_overflow_evals', ...
    'budget_runtime_s','solver_wall_runtime_s','validation_runtime_s'});
boDetails = sortrows(boDetails,'seed');
assert(height(boDetails) == numel(seeds) && ...
    isequal(boDetails.seed(:)',seeds), ...
    'Expected all Bayesian seeds in the processed metrics.');
assert(all(boDetails.search_fe == budget), ...
    'Bayesian admitted FE does not equal the prescribed budget.');
end


function plot_runtime_convergence( ...
    analysisDir,optimizers,budget,figureDir,saveFigures)

files = dir(fullfile(analysisDir,'convergence_*.mat'));
assert(numel(files) == 1, ...
    'Expected exactly one convergence file for the focused runtime study.');
loaded = load(fullfile(files(1).folder,files(1).name), ...
    'comparison','curves');
assert(string(loaded.comparison.settings.mission.type) == "LUNAR_GATEWAY", ...
    'Runtime-study convergence file is not the Lunar Gateway case.');

fig = create_paper_figure(7.2,4.4);
ax = axes(fig);
hold(ax,'on');
box(ax,'on');
grid(ax,'on');
colors = lines(numel(optimizers));
curveOptimizers = upper(string({loaded.curves.optimizer}));
lineHandles = gobjects(numel(optimizers),1);
plotStartFE = 60;

for k = 1:numel(optimizers)
    idx = find(curveOptimizers == optimizers(k),1);
    assert(~isempty(idx),'Missing %s convergence curve.',optimizers(k));
    curve = loaded.curves(idx);
    fe = double(curve.fe(:));
    meanBest = double(curve.mean(:));
    valid = isfinite(meanBest) & fe >= plotStartFE;
    assert(any(valid),'No convergence data for %s at or after FE 60.',optimizers(k));

    x = fe(valid);
    y = meanBest(valid);
    lineHandles(k) = stairs(ax,x,y, ...
        'Color',colors(k,:),'LineWidth',2.0, ...
        'DisplayName',optimizers(k));

    markerStride = max(1,round(numel(x)/12));
    markerIdx = unique([1:markerStride:numel(x),numel(x)]);
    plot(ax,x(markerIdx),y(markerIdx),'o', ...
        'Color',colors(k,:),'MarkerFaceColor',colors(k,:), ...
        'MarkerSize',4,'HandleVisibility','off');
end

xlim(ax,[plotStartFE budget]);
xticks(ax,unique([plotStartFE 240 480 720 960 budget]));
xlabel(ax,'Function evaluations','FontWeight','bold');
ylabel(ax,'Mean best-so-far objective','FontWeight','bold');
apply_figure_style(ax);
lgd = legend(ax,lineHandles,cellstr(optimizers), ...
    'Location','northoutside','Orientation','horizontal', ...
    'NumColumns',numel(optimizers));
lgd.FontSize = 12;
lgd.Box = 'off';

export_preview(fig,figureDir,'runtime_comparison_1200_convergence',saveFigures);
end


function plot_runtime_bar(results,optimizers,figureDir,saveFigures)
values = nan(numel(optimizers),1);
errors = values;
for k = 1:numel(optimizers)
    row = results(results.Optimizer == optimizers(k),:);
    assert(height(row) == 1,'Missing runtime row for %s.',optimizers(k));
    values(k) = row.BudgetRuntimeMean_s;
    errors(k) = row.BudgetRuntimeStd_s;
end

fig = create_paper_figure(7.2,4.4);
ax = axes(fig);
hold(ax,'on');
box(ax,'on');
grid(ax,'on');
bar(ax,1:numel(optimizers),values,0.72);
lowerErrors = min(max(errors,0),max(values,0));
upperErrors = max(errors,0);
errorbar(ax,1:numel(optimizers),values,lowerErrors,upperErrors, ...
    'k.','LineWidth',1.35,'CapSize',9);
ax.XTick = 1:numel(optimizers);
ax.XTickLabel = cellstr(optimizers);
xlabel(ax,'Optimizer','FontWeight','bold');
ylabel(ax,'Runtime to 1200 FE (s)','FontWeight','bold');
apply_figure_style(ax);

export_preview(fig,figureDir,'runtime_comparison_1200_runtime',saveFigures);
end


function plot_objective_bar(results,optimizers,figureDir,saveFigures)
values = nan(numel(optimizers),1);
errors = values;
for k = 1:numel(optimizers)
    row = results(results.Optimizer == optimizers(k),:);
    assert(height(row) == 1,'Missing objective row for %s.',optimizers(k));
    values(k) = row.BestJMean;
    errors(k) = row.BestJStd;
end

fig = create_paper_figure(7.2,4.4);
ax = axes(fig);
hold(ax,'on');
box(ax,'on');
grid(ax,'on');
bar(ax,1:numel(optimizers),values,0.72);
lowerErrors = min(max(errors,0),max(values,0));
upperErrors = max(errors,0);
errorbar(ax,1:numel(optimizers),values,lowerErrors,upperErrors, ...
    'k.','LineWidth',1.35,'CapSize',9);
ax.XTick = 1:numel(optimizers);
ax.XTickLabel = cellstr(optimizers);
xlabel(ax,'Optimizer','FontWeight','bold');
ylabel(ax,'Final best objective','FontWeight','bold');
apply_figure_style(ax);

export_preview(fig,figureDir,'runtime_comparison_1200_objective',saveFigures);
end


function fig = create_paper_figure(widthIn,heightIn)
fig = figure('Color','w','Units','inches', ...
    'Position',[1 1 widthIn heightIn], ...
    'PaperUnits','inches','PaperSize',[widthIn heightIn], ...
    'PaperPosition',[0 0 widthIn heightIn], ...
    'PaperPositionMode','manual','Renderer','painters', ...
    'InvertHardcopy','off');
end


function apply_figure_style(ax)
ax.FontSize = max(12,ax.FontSize);
ax.LineWidth = 1.0;
ax.TickDir = 'out';
ax.XLabel.FontSize = max(14,ax.XLabel.FontSize);
ax.YLabel.FontSize = max(14,ax.YLabel.FontSize);
end


function export_preview(fig,figureDir,stem,saveFigures)
drawnow;
if ~saveFigures, return; end
assert(strlength(string(figureDir)) > 0,'Figure directory is empty.');
base = fullfile(char(figureDir),stem);
print(fig,[base '.eps'],'-depsc','-painters');
exportgraphics(fig,[base '.png'],'Resolution',300);
end


function [mu,sigma] = sample_statistics(values)
values = double(values(:));
mu = mean(values);
if numel(values) < 2
    sigma = NaN;
else
    sigma = std(values);
end
end
