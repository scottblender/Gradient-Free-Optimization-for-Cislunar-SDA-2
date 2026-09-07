function report = run_reviewer2_baseline_pipeline(saveFigures)
%RUN_REVIEWER2_BASELINE_PIPELINE Process and verify the completed GA baseline.
%
% Baseline definition:
%   GA only, 6000 FE, 20 seeds
%   ANGLES_ONLY and ANGLES_RANGE
%   3, 5, 7, 10 observers
%   Lunar Gateway: 1, 3, 5 periods
%   Low-thrust transfer and Gateway impulse: one fixed trajectory each
%   screening ON, J1+J2+J3
%
% This is 40 unique configurations x 20 seeds = 800 optimization runs.
% The script validates every saved run, aggregates mean +/- sample standard
% deviation metrics, generates measurement/observer-count/period sensitivity
% figures, and exports best-run observer-geometry panels for 3/5/7/10 sensors.
%
% Usage:
%   report = run_reviewer2_baseline_pipeline;
%   report = run_reviewer2_baseline_pipeline(false); % display only

if nargin < 1 || isempty(saveFigures), saveFigures = true; end
validateattributes(saveFigures,{'logical','numeric'},{'scalar'});
saveFigures = logical(saveFigures);

paths = setup_project();
budget = 6000;
seeds = 0:19;
optimizer = "GA";
measurements = ["ANGLES_ONLY","ANGLES_RANGE"];
missions = ["LUNAR_GATEWAY","LOW_THRUST_TRANSFER","GATEWAY_IMPULSE"];
observerCounts = [3 5 7 10];
gatewayPeriods = [1 3 5];
studyID = "reviewer2_baseline_v1";
baselineRoot = fullfile(paths.runs,'BASELINE');
expectedGroups = 2*(4*3 + 4 + 4);
expectedRuns = expectedGroups*numel(seeds);

fprintf('\n--- Reviewer 2 baseline pipeline ---\n');
fprintf('Optimizer:                   GA\n');
fprintf('Measurement models:          AO, AR\n');
fprintf('Observer counts:             %s\n',mat2str(observerCounts));
fprintf('Gateway periods:             %s\n',mat2str(gatewayPeriods));
fprintf('Independent seeds:           %s\n',mat2str(seeds));
fprintf('Search FE per run:           %d\n',budget);
fprintf('Expected configurations:     %d\n',expectedGroups);
fprintf('Expected optimization runs:  %d\n\n',expectedRuns);

test_project_structure();
test_fe_study_configuration();

assert(isfolder(baselineRoot),'Baseline root does not exist: %s',baselineRoot);
[summary,inventory] = process_fe_convergence( ...
    baselineRoot,studyID,seeds,budget,false,optimizer);

nonemptyRuns = inventory.run_file ~= "";
assert(sum(nonemptyRuns) == expectedRuns, ...
    'Expected %d saved baseline runs but found %d.',expectedRuns,sum(nonemptyRuns));
assert(~any(~inventory.valid), ...
    'One or more baseline runs failed validation. Inspect run_inventory.csv.');
assert(height(summary) == expectedGroups, ...
    'Expected %d complete baseline configurations but found %d.', ...
    expectedGroups,height(summary));
assert(all(summary.optimizer == optimizer) && all(summary.n_runs == numel(seeds)) && ...
    all(summary.fe_budget == budget), ...
    'Processed baseline summary does not match GA/20-seed/6000-FE design.');

analysisDir = newest_analysis_directory(baselineRoot);
metricsFile = fullfile(analysisDir,'final_run_metrics.csv');
assert(isfile(metricsFile),'Missing processed baseline metrics: %s',metricsFile);
runMetrics = readtable(metricsFile,'TextType','string', ...
    'VariableNamingRule','preserve');
assert(height(runMetrics) == expectedRuns, ...
    'Expected %d baseline metric rows but found %d.',expectedRuns,height(runMetrics));

results = build_baseline_results(summary,runMetrics,seeds,budget);
validate_baseline_factorial(results,measurements,missions,observerCounts,gatewayPeriods);
bestGeometryRuns = select_baseline_geometry_runs( ...
    results,runMetrics,missions,observerCounts,seeds);
formatted = format_baseline_table(results);

fprintf('\n--- Baseline aggregate results (mean +/- sample std) ---\n');
disp(formatted);
fprintf('\n--- Best AO/p1 runs used for observer-count geometry panels ---\n');
disp(bestGeometryRuns(:,{'Mission','PanelLabel','Seed','BestObjective'}));

writetable(results,fullfile(analysisDir,'baseline_6000_results.csv'));
writetable(formatted,fullfile(analysisDir,'baseline_6000_formatted.csv'));
writetable(bestGeometryRuns, ...
    fullfile(analysisDir,'baseline_6000_geometry_selected_runs.csv'));

figureDir = "";
if saveFigures
    figureDir = string(fullfile(analysisDir,'paper_preview'));
    if ~isfolder(figureDir), mkdir(figureDir); end
end

% Observer-count sensitivity at the nominal one-period Gateway duration.
metricSpecs = { ...
    'BestJMean','BestJStd','Final best objective','objective'; ...
    'RMSEPosMean_km','RMSEPosStd_km','Position RMSE (km)','position_rmse'; ...
    'EffectiveSigmaPosMean_km','EffectiveSigmaPosStd_km', ...
        'Effective position sigma (km)','effective_sigma'; ...
    'RuntimeMean_s','RuntimeStd_s','Runtime to 6000 FE (s)','runtime'; ...
    'CoverageMean','CoverageStd','Coverage fraction','coverage'};
for m = 1:numel(missions)
    for q = 1:size(metricSpecs,1)
        plot_observer_count_metric(results,missions(m),measurements,observerCounts, ...
            metricSpecs{q,1},metricSpecs{q,2},metricSpecs{q,3}, ...
            "baseline_"+metricSpecs{q,4}+"_vs_observers_"+mission_code(missions(m)), ...
            figureDir,saveFigures);
    end
end

% Gateway-duration sensitivity: one figure per measurement model and metric,
% with separate lines for the four observer counts.
periodSpecs = { ...
    'BestJMean','BestJStd','Final best objective','objective'; ...
    'RMSEPosMean_km','RMSEPosStd_km','Position RMSE (km)','position_rmse'; ...
    'EffectiveSigmaPosMean_km','EffectiveSigmaPosStd_km', ...
        'Effective position sigma (km)','effective_sigma'};
for meas = measurements
    for q = 1:size(periodSpecs,1)
        plot_gateway_period_metric(results,meas,observerCounts,gatewayPeriods, ...
            periodSpecs{q,1},periodSpecs{q,2},periodSpecs{q,3}, ...
            "baseline_gateway_"+periodSpecs{q,4}+"_vs_periods_"+measurement_code(meas), ...
            figureDir,saveFigures);
    end
end

geometryDetails = plot_reviewer2_constellation_geometry( ...
    bestGeometryRuns,figureDir,"baseline_geometry",saveFigures);
writetable(geometryDetails, ...
    fullfile(analysisDir,'baseline_6000_geometry_details.csv'));

report = struct();
report.studyID = studyID;
report.budget = budget;
report.seeds = seeds;
report.expectedGroups = expectedGroups;
report.expectedRuns = expectedRuns;
report.analysisDirectory = string(analysisDir);
report.figureDirectory = figureDir;
report.summary = summary;
report.inventory = inventory;
report.runMetrics = runMetrics;
report.results = results;
report.formattedTable = formatted;
report.bestGeometryRuns = bestGeometryRuns;
report.geometryDetails = geometryDetails;

fprintf('\nReviewer 2 baseline pipeline passed.\n');
fprintf('Validated runs: %d/%d\n',sum(inventory.valid),expectedRuns);
fprintf('Validated configurations: %d/%d\n',height(results),expectedGroups);
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
assert(~isempty(directories),'No FE_DATA directory created under %s.',root);
[~,idx] = max([directories.datenum]);
analysisDir = fullfile(directories(idx).folder,directories(idx).name);
end


function results = build_baseline_results(summary,runMetrics,seeds,budget)
n = height(summary);
comparisonKey = strings(n,1); mission = strings(n,1); measurement = strings(n,1);
numObservers = nan(n,1); nPeriods = nan(n,1); nRuns = nan(n,1);
bestJMean = nan(n,1); bestJStd = nan(n,1);
runtimeMean = nan(n,1); runtimeStd = nan(n,1);
rmseMean = nan(n,1); rmseStd = nan(n,1);
sigmaMean = nan(n,1); sigmaStd = nan(n,1);
stabilityMean = nan(n,1); stabilityStd = nan(n,1);
coverageMean = nan(n,1); coverageStd = nan(n,1);
screeningMean = nan(n,1); screeningStd = nan(n,1);

for k = 1:n
    key = summary.comparison_key(k);
    rows = runMetrics(runMetrics.comparison_key == key & runMetrics.optimizer == "GA",:);
    rows = sortrows(rows,'seed');
    assert(height(rows) == numel(seeds) && isequal(rows.seed(:)',seeds), ...
        'Incomplete baseline seed group for comparison key %s.',key);
    assert(all(rows.search_fe == budget),'Baseline FE mismatch for key %s.',key);

    S = load(rows.run_file(1),'runState');
    r = S.runState;
    s = r.settings;
    assert(s.useScreening && s.costFlags.J1 && s.costFlags.J2 && s.costFlags.J3, ...
        'Baseline must use screening ON and J1+J2+J3.');

    comparisonKey(k) = key;
    mission(k) = string(s.mission.type);
    measurement(k) = string(s.measurements.type);
    numObservers(k) = s.mission.optimization.numObservers;
    if mission(k) == "LUNAR_GATEWAY"
        nPeriods(k) = s.mission.gateway.Nperiods;
    else
        nPeriods(k) = 1;
    end
    nRuns(k) = height(rows);
    [bestJMean(k),bestJStd(k)] = sample_statistics(rows.bestJ);
    [runtimeMean(k),runtimeStd(k)] = sample_statistics(rows.budget_runtime_s);
    [rmseMean(k),rmseStd(k)] = sample_statistics(rows.rmse_pos_km);
    [sigmaMean(k),sigmaStd(k)] = sample_statistics(rows.mean_effective_sigma_pos_km);
    [stabilityMean(k),stabilityStd(k)] = sample_statistics(rows.mean_stability);
    [coverageMean(k),coverageStd(k)] = sample_statistics(rows.coverage_epoch_fraction);
    [screeningMean(k),screeningStd(k)] = sample_statistics(rows.screening_count);
end

results = table(comparisonKey,mission,measurement,numObservers,nPeriods,nRuns, ...
    bestJMean,bestJStd,runtimeMean,runtimeStd,rmseMean,rmseStd, ...
    sigmaMean,sigmaStd,stabilityMean,stabilityStd,coverageMean,coverageStd, ...
    screeningMean,screeningStd, ...
    'VariableNames',{'ComparisonKey','Mission','Measurement','NumObservers', ...
    'NPeriods','NRuns','BestJMean','BestJStd','RuntimeMean_s','RuntimeStd_s', ...
    'RMSEPosMean_km','RMSEPosStd_km','EffectiveSigmaPosMean_km', ...
    'EffectiveSigmaPosStd_km','MeanStabilityMean','MeanStabilityStd', ...
    'CoverageMean','CoverageStd','ScreeningMean','ScreeningStd'});
results = sortrows(results,{'Mission','Measurement','NumObservers','NPeriods'});
end


function validate_baseline_factorial(results,measurements,missions,observerCounts,gatewayPeriods)
assert(height(results) == 40,'Baseline must contain exactly 40 configurations.');
for meas = measurements
    for mission = missions
        for nObs = observerCounts
            periods = 1;
            if mission == "LUNAR_GATEWAY", periods = gatewayPeriods; end
            for nper = periods
                rows = results(results.Measurement == meas & results.Mission == mission & ...
                    results.NumObservers == nObs & results.NPeriods == nper,:);
                assert(height(rows) == 1, ...
                    'Missing/duplicate baseline configuration: %s/%s/o%d/p%d.', ...
                    meas,mission,nObs,nper);
                assert(rows.NRuns == 20,'Baseline configuration does not contain 20 runs.');
            end
        end
    end
end
end


function selected = select_baseline_geometry_runs( ...
    results,runMetrics,missions,observerCounts,seeds)
rowsExpected = numel(missions)*numel(observerCounts);
missionColumn = strings(rowsExpected,1); panelKey = strings(rowsExpected,1);
panelLabel = strings(rowsExpected,1); runFile = strings(rowsExpected,1);
bestObjective = nan(rowsExpected,1); seedColumn = nan(rowsExpected,1);
row = 0;
for mission = missions
    for nObs = observerCounts
        row = row + 1;
        config = results(results.Mission == mission & ...
            results.Measurement == "ANGLES_ONLY" & ...
            results.NumObservers == nObs & results.NPeriods == 1,:);
        assert(height(config) == 1,'Missing AO/p1 geometry baseline configuration.');
        metricRows = runMetrics(runMetrics.comparison_key == config.ComparisonKey,:);
        assert(height(metricRows) == numel(seeds),'Incomplete baseline geometry group.');
        [value,idx] = min(metricRows.bestJ);
        missionColumn(row) = mission;
        panelKey(row) = "o"+string(nObs);
        panelLabel(row) = string(nObs)+" observers";
        runFile(row) = metricRows.run_file(idx);
        bestObjective(row) = value;
        seedColumn(row) = metricRows.seed(idx);
    end
end
selected = table(missionColumn,panelKey,panelLabel,runFile,bestObjective,seedColumn, ...
    'VariableNames',{'Mission','PanelKey','PanelLabel','RunFile','BestObjective','Seed'});
end


function formatted = format_baseline_table(results)
formatted = table(mission_labels(results.Mission),measurement_labels(results.Measurement), ...
    results.NumObservers,results.NPeriods,results.NRuns, ...
    compose('%.6g +/- %.3g',results.BestJMean,results.BestJStd), ...
    compose('%.5g +/- %.3g',results.RMSEPosMean_km,results.RMSEPosStd_km), ...
    compose('%.5g +/- %.3g', ...
        results.EffectiveSigmaPosMean_km,results.EffectiveSigmaPosStd_km), ...
    compose('%.5g +/- %.3g',results.RuntimeMean_s,results.RuntimeStd_s), ...
    'VariableNames',{'Case','Measurement','Observers','Periods','Runs', ...
    'BestObjective','RMSEPosition_km','EffectiveSigmaPosition_km','Runtime_s'});
end


function plot_observer_count_metric(results,mission,measurements,observerCounts, ...
    valueField,errorField,yLabel,stem,figureDir,saveFigures)
fig = create_paper_figure(7.2,4.6); ax = axes(fig);
hold(ax,'on'); box(ax,'on'); grid(ax,'on');
colors = lines(numel(measurements)); handles = gobjects(numel(measurements),1);
for q = 1:numel(measurements)
    values = nan(size(observerCounts)); errors = values;
    for k = 1:numel(observerCounts)
        row = results(results.Mission == mission & ...
            results.Measurement == measurements(q) & ...
            results.NumObservers == observerCounts(k) & results.NPeriods == 1,:);
        assert(height(row) == 1,'Missing observer-count baseline point.');
        values(k) = row.(valueField); errors(k) = row.(errorField);
    end
    handles(q) = errorbar(ax,observerCounts,values,errors,'-o', ...
        'Color',colors(q,:),'LineWidth',1.8,'MarkerSize',6, ...
        'MarkerFaceColor',colors(q,:),'CapSize',8, ...
        'DisplayName',measurement_label(measurements(q)));
end
ax.XTick = observerCounts; xlim(ax,[min(observerCounts)-0.5 max(observerCounts)+0.5]);
xlabel(ax,'Number of observers','FontWeight','bold');
ylabel(ax,yLabel,'FontWeight','bold');
apply_figure_style(ax);
lgd = legend(ax,handles,'Location','northoutside','Orientation','horizontal');
format_legend(lgd);
export_preview(fig,figureDir,stem,saveFigures);
end


function plot_gateway_period_metric(results,measurement,observerCounts,gatewayPeriods, ...
    valueField,errorField,yLabel,stem,figureDir,saveFigures)
fig = create_paper_figure(7.2,4.6); ax = axes(fig);
hold(ax,'on'); box(ax,'on'); grid(ax,'on');
colors = lines(numel(observerCounts)); handles = gobjects(numel(observerCounts),1);
for q = 1:numel(observerCounts)
    values = nan(size(gatewayPeriods)); errors = values;
    for k = 1:numel(gatewayPeriods)
        row = results(results.Mission == "LUNAR_GATEWAY" & ...
            results.Measurement == measurement & ...
            results.NumObservers == observerCounts(q) & ...
            results.NPeriods == gatewayPeriods(k),:);
        assert(height(row) == 1,'Missing Gateway-period baseline point.');
        values(k) = row.(valueField); errors(k) = row.(errorField);
    end
    handles(q) = errorbar(ax,gatewayPeriods,values,errors,'-o', ...
        'Color',colors(q,:),'LineWidth',1.8,'MarkerSize',6, ...
        'MarkerFaceColor',colors(q,:),'CapSize',8, ...
        'DisplayName',string(observerCounts(q))+" observers");
end
ax.XTick = gatewayPeriods; xlim(ax,[0.75 5.25]);
xlabel(ax,'Lunar Gateway periods','FontWeight','bold');
ylabel(ax,yLabel,'FontWeight','bold');
apply_figure_style(ax);
lgd = legend(ax,handles,'Location','northoutside','Orientation','horizontal', ...
    'NumColumns',2);
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
function labels = measurement_labels(values)
values = string(values(:)); labels = strings(size(values));
for k = 1:numel(values), labels(k) = measurement_label(values(k)); end
end
function label = measurement_label(value)
switch upper(string(value))
    case "ANGLES_ONLY", label = "AO";
    case "ANGLES_RANGE", label = "AR";
    otherwise, label = string(value);
end
end
function code = measurement_code(value)
switch upper(string(value))
    case "ANGLES_ONLY", code = "ao";
    case "ANGLES_RANGE", code = "ar";
    otherwise, code = lower(string(value));
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
