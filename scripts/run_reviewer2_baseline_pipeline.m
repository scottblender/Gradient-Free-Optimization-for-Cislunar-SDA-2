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
% deviation metrics, generates FE convergence and baseline-sensitivity
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
plotStartFE = 60;
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
expectedConvergenceFigures = numel(missions)*numel(measurements) + numel(measurements);

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
requiredMetrics = ["rmse_pos_km","mean_effective_sigma_pos_km", ...
    "mean_stability","coverage_epoch_fraction", ...
    "mean_available_observers","screening_count"];
assert(all(ismember(requiredMetrics,string(runMetrics.Properties.VariableNames))), ...
    'Processed baseline metrics are missing required tracking fields.');

results = build_baseline_results(summary,runMetrics,seeds,budget);
validate_baseline_factorial(results,measurements,missions,observerCounts,gatewayPeriods);
bestGeometryRuns = select_baseline_geometry_runs( ...
    results,runMetrics,missions,observerCounts,seeds);
formatted = format_baseline_table(results);
trends = build_baseline_trends(results,measurements,missions,observerCounts);

fprintf('\n--- Baseline aggregate results (mean +/- sample std) ---\n');
disp(formatted);
fprintf('\n--- Best AO/p1 runs used for observer-count geometry panels ---\n');
disp(bestGeometryRuns(:,{'Mission','PanelLabel','Seed','BestObjective'}));
fprintf('\n--- Manuscript-ready baseline contrasts (percent change) ---\n');
disp(trends);

writetable(results,fullfile(analysisDir,'baseline_6000_results.csv'));
writetable(formatted,fullfile(analysisDir,'baseline_6000_formatted.csv'));
writetable(bestGeometryRuns, ...
    fullfile(analysisDir,'baseline_6000_geometry_selected_runs.csv'));
writetable(trends,fullfile(analysisDir,'baseline_6000_trends.csv'));

figureDir = "";
if saveFigures
    figureDir = string(fullfile(analysisDir,'paper_preview'));
    if ~isfolder(figureDir), mkdir(figureDir); end
end

% FE convergence for the observer-count comparison. Hold mission,
% measurement model, and duration fixed so only 3/5/7/10 observers vary.
for mission = missions
    for meas = measurements
        plot_observer_count_convergence(results,analysisDir,mission,meas, ...
            observerCounts,budget,plotStartFE, ...
            "baseline_convergence_vs_observers_"+mission_code(mission)+ ...
            "_"+measurement_code(meas),figureDir,saveFigures);
    end
end

% FE convergence for Gateway-duration sensitivity. Use the common 3-observer
% design so the only varying baseline factor is 1/3/5 target periods.
for meas = measurements
    plot_gateway_period_convergence(results,analysisDir,meas,3,gatewayPeriods, ...
        budget,plotStartFE, ...
        "baseline_convergence_vs_periods_"+measurement_code(meas), ...
        figureDir,saveFigures);
end

% Compact manuscript panels show AO/AR and observer-count trends together.
for m = 1:numel(missions)
    plot_baseline_observer_panel(results,missions(m),measurements,observerCounts, ...
        "baseline_metrics_vs_observers_"+mission_code(missions(m)), ...
        figureDir,saveFigures);
end

% Gateway-duration sensitivity, with all four observer counts in each panel.
for meas = measurements
    plot_baseline_period_panel(results,meas,observerCounts,gatewayPeriods, ...
        "baseline_gateway_metrics_vs_periods_"+measurement_code(meas), ...
        figureDir,saveFigures);
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
report.expectedConvergenceFigures = expectedConvergenceFigures;
report.analysisDirectory = string(analysisDir);
report.figureDirectory = figureDir;
report.summary = summary;
report.inventory = inventory;
report.runMetrics = runMetrics;
report.results = results;
report.formattedTable = formatted;
report.trends = trends;
report.bestGeometryRuns = bestGeometryRuns;
report.geometryDetails = geometryDetails;

fprintf('\nReviewer 2 baseline pipeline passed.\n');
fprintf('Validated runs: %d/%d\n',sum(inventory.valid),expectedRuns);
fprintf('Validated configurations: %d/%d\n',height(results),expectedGroups);
fprintf('Baseline convergence figures: %d\n',expectedConvergenceFigures);
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
availableMean = nan(n,1); availableStd = nan(n,1);
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
    [availableMean(k),availableStd(k)] = ...
        sample_statistics(rows.mean_available_observers);
    [screeningMean(k),screeningStd(k)] = sample_statistics(rows.screening_count);
end

results = table(comparisonKey,mission,measurement,numObservers,nPeriods,nRuns, ...
    bestJMean,bestJStd,runtimeMean,runtimeStd,rmseMean,rmseStd, ...
    sigmaMean,sigmaStd,stabilityMean,stabilityStd,coverageMean,coverageStd, ...
    availableMean,availableStd,screeningMean,screeningStd, ...
    'VariableNames',{'ComparisonKey','Mission','Measurement','NumObservers', ...
    'NPeriods','NRuns','BestJMean','BestJStd','RuntimeMean_s','RuntimeStd_s', ...
    'RMSEPosMean_km','RMSEPosStd_km','EffectiveSigmaPosMean_km', ...
    'EffectiveSigmaPosStd_km','MeanStabilityMean','MeanStabilityStd', ...
    'CoverageMean','CoverageStd','AvailableObserversMean', ...
    'AvailableObserversStd','ScreeningMean','ScreeningStd'});
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
    compose('%.4f +/- %.3f', ...
        results.AvailableObserversMean,results.AvailableObserversStd), ...
    compose('%.5g +/- %.3g',results.RuntimeMean_s,results.RuntimeStd_s), ...
    'VariableNames',{'Case','Measurement','Observers','Periods','Runs', ...
    'BestObjective','RMSEPosition_km','EffectiveSigmaPosition_km', ...
    'MeanAvailableObservers','Runtime_s'});
end


function trends = build_baseline_trends(results,measurements,missions,observerCounts)
factor = strings(0,1); missionColumn = strings(0,1);
measurementColumn = strings(0,1); contrast = strings(0,1);
objectiveChangePct = zeros(0,1); rmseChangePct = zeros(0,1);
sigmaChangePct = zeros(0,1); availableChangePct = zeros(0,1);

% AO -> AR at each mission/observer configuration (one-period Gateway).
for mission = missions
    for nObs = observerCounts
        ao = results(results.Mission == mission & results.Measurement == "ANGLES_ONLY" & ...
            results.NumObservers == nObs & results.NPeriods == 1,:);
        ar = results(results.Mission == mission & results.Measurement == "ANGLES_RANGE" & ...
            results.NumObservers == nObs & results.NPeriods == 1,:);
        [factor,missionColumn,measurementColumn,contrast,objectiveChangePct, ...
            rmseChangePct,sigmaChangePct,availableChangePct] = append_contrast( ...
            factor,missionColumn,measurementColumn,contrast,objectiveChangePct, ...
            rmseChangePct,sigmaChangePct,availableChangePct, ...
            "Measurement",mission,"AO to AR", ...
            "AO to AR at "+nObs+" observers",ao,ar);
    end
end

% 3 -> 10 observers, holding measurement model and duration fixed.
for mission = missions
    for measurement = measurements
        low = results(results.Mission == mission & results.Measurement == measurement & ...
            results.NumObservers == 3 & results.NPeriods == 1,:);
        high = results(results.Mission == mission & results.Measurement == measurement & ...
            results.NumObservers == 10 & results.NPeriods == 1,:);
        [factor,missionColumn,measurementColumn,contrast,objectiveChangePct, ...
            rmseChangePct,sigmaChangePct,availableChangePct] = append_contrast( ...
            factor,missionColumn,measurementColumn,contrast,objectiveChangePct, ...
            rmseChangePct,sigmaChangePct,availableChangePct, ...
            "Observer count",mission,measurement,"3 to 10 observers",low,high);
    end
end

% 1 -> 5 Gateway periods, holding observer count and measurement model fixed.
for measurement = measurements
    for nObs = observerCounts
        short = results(results.Mission == "LUNAR_GATEWAY" & ...
            results.Measurement == measurement & results.NumObservers == nObs & ...
            results.NPeriods == 1,:);
        long = results(results.Mission == "LUNAR_GATEWAY" & ...
            results.Measurement == measurement & results.NumObservers == nObs & ...
            results.NPeriods == 5,:);
        [factor,missionColumn,measurementColumn,contrast,objectiveChangePct, ...
            rmseChangePct,sigmaChangePct,availableChangePct] = append_contrast( ...
            factor,missionColumn,measurementColumn,contrast,objectiveChangePct, ...
            rmseChangePct,sigmaChangePct,availableChangePct, ...
            "Tracking duration","LUNAR_GATEWAY",measurement, ...
            "1 to 5 periods at "+nObs+" observers",short,long);
    end
end
trends = table(factor,missionColumn,measurementColumn,contrast, ...
    objectiveChangePct,rmseChangePct,sigmaChangePct,availableChangePct, ...
    'VariableNames',{'Factor','Mission','Measurement','Contrast', ...
    'ObjectiveChangePct','RMSEChangePct','EffectiveSigmaChangePct', ...
    'AvailableObserversChangePct'});
end


function varargout = append_contrast(factor,missionColumn,measurementColumn, ...
    contrast,objectiveChangePct,rmseChangePct,sigmaChangePct,availableChangePct, ...
    factorName,mission,measurement,contrastName,before,after)
assert(height(before) == 1 && height(after) == 1,'Baseline contrast is incomplete.');
factor(end+1,1) = string(factorName);
missionColumn(end+1,1) = string(mission);
measurementColumn(end+1,1) = string(measurement);
contrast(end+1,1) = string(contrastName);
objectiveChangePct(end+1,1) = percent_change(before.BestJMean,after.BestJMean);
rmseChangePct(end+1,1) = percent_change(before.RMSEPosMean_km,after.RMSEPosMean_km);
sigmaChangePct(end+1,1) = percent_change( ...
    before.EffectiveSigmaPosMean_km,after.EffectiveSigmaPosMean_km);
availableChangePct(end+1,1) = percent_change( ...
    before.AvailableObserversMean,after.AvailableObserversMean);
varargout = {factor,missionColumn,measurementColumn,contrast,objectiveChangePct, ...
    rmseChangePct,sigmaChangePct,availableChangePct};
end


function value = percent_change(before,after)
value = 100*(after-before)/max(abs(before),eps);
end


function plot_observer_count_convergence(results,analysisDir,mission,measurement, ...
    observerCounts,budget,plotStartFE,stem,figureDir,saveFigures)
fig = create_paper_figure(7.2,4.6); ax = axes(fig);
hold(ax,'on'); box(ax,'on'); grid(ax,'on');
colors = lines(numel(observerCounts)); handles = gobjects(numel(observerCounts),1);
for q = 1:numel(observerCounts)
    row = results(results.Mission == mission & ...
        results.Measurement == measurement & ...
        results.NumObservers == observerCounts(q) & results.NPeriods == 1,:);
    assert(height(row) == 1,'Missing observer-count convergence configuration.');
    curve = load_ga_curve(analysisDir,row.ComparisonKey,budget);
    valid = isfinite(curve.mean) & curve.fe >= plotStartFE;
    assert(any(valid),'No baseline convergence data for observer count %d.',observerCounts(q));
    x = double(curve.fe(valid)); y = double(curve.mean(valid));
    deviation = double(curve.std(valid));
    plot_uncertainty_band(ax,x,y,deviation,colors(q,:));
    handles(q) = stairs(ax,x,y,'Color',colors(q,:), ...
        'LineWidth',2.0,'DisplayName',string(observerCounts(q))+" observers");
    markerStride = max(1,round(numel(x)/12));
    markerIdx = unique([1:markerStride:numel(x),numel(x)]);
    plot(ax,x(markerIdx),y(markerIdx),'o','Color',colors(q,:), ...
        'MarkerFaceColor',colors(q,:),'MarkerSize',4,'HandleVisibility','off');
end
xlim(ax,[plotStartFE budget]);
xticks(ax,unique([plotStartFE 1000:1000:budget budget]));
xlabel(ax,'Function evaluations','FontWeight','bold');
ylabel(ax,'Mean best-so-far objective','FontWeight','bold');
apply_figure_style(ax);
lgd = legend(ax,handles,'Location','northoutside','Orientation','horizontal', ...
    'NumColumns',2);
format_legend(lgd);
export_preview(fig,figureDir,stem,saveFigures);
end


function plot_gateway_period_convergence(results,analysisDir,measurement,nObs, ...
    gatewayPeriods,budget,plotStartFE,stem,figureDir,saveFigures)
fig = create_paper_figure(7.2,4.6); ax = axes(fig);
hold(ax,'on'); box(ax,'on'); grid(ax,'on');
colors = lines(numel(gatewayPeriods)); handles = gobjects(numel(gatewayPeriods),1);
for q = 1:numel(gatewayPeriods)
    row = results(results.Mission == "LUNAR_GATEWAY" & ...
        results.Measurement == measurement & results.NumObservers == nObs & ...
        results.NPeriods == gatewayPeriods(q),:);
    assert(height(row) == 1,'Missing Gateway-period convergence configuration.');
    curve = load_ga_curve(analysisDir,row.ComparisonKey,budget);
    valid = isfinite(curve.mean) & curve.fe >= plotStartFE;
    assert(any(valid),'No Gateway convergence data for %d periods.',gatewayPeriods(q));
    x = double(curve.fe(valid)); y = double(curve.mean(valid));
    deviation = double(curve.std(valid));
    plot_uncertainty_band(ax,x,y,deviation,colors(q,:));
    label = string(gatewayPeriods(q))+" period";
    if gatewayPeriods(q) ~= 1, label = label+"s"; end
    handles(q) = stairs(ax,x,y,'Color',colors(q,:), ...
        'LineWidth',2.0,'DisplayName',label);
    markerStride = max(1,round(numel(x)/12));
    markerIdx = unique([1:markerStride:numel(x),numel(x)]);
    plot(ax,x(markerIdx),y(markerIdx),'o','Color',colors(q,:), ...
        'MarkerFaceColor',colors(q,:),'MarkerSize',4,'HandleVisibility','off');
end
xlim(ax,[plotStartFE budget]);
xticks(ax,unique([plotStartFE 1000:1000:budget budget]));
xlabel(ax,'Function evaluations','FontWeight','bold');
ylabel(ax,'Mean best-so-far objective','FontWeight','bold');
apply_figure_style(ax);
lgd = legend(ax,handles,'Location','northoutside','Orientation','horizontal', ...
    'NumColumns',numel(gatewayPeriods));
format_legend(lgd);
export_preview(fig,figureDir,stem,saveFigures);
end


function curve = load_ga_curve(analysisDir,comparisonKey,budget)
file = fullfile(analysisDir,"convergence_"+string(comparisonKey)+".mat");
assert(isfile(file),'Missing baseline convergence file: %s',file);
S = load(file,'curves');
assert(isfield(S,'curves') && numel(S.curves) == 1 && ...
    upper(string(S.curves(1).optimizer)) == "GA", ...
    'Baseline convergence file must contain exactly one GA curve.');
curve = S.curves(1);
assert(numel(curve.fe) == budget && curve.fe(end) == budget && ...
    numel(curve.mean) == budget, ...
    'Baseline convergence curve does not span the prescribed FE budget.');
end


function plot_baseline_observer_panel(results,mission,measurements,observerCounts, ...
    stem,figureDir,saveFigures)
specs = { ...
    'BestJMean','BestJStd','Final best objective'; ...
    'RMSEPosMean_km','RMSEPosStd_km','Position RMSE (km)'; ...
    'EffectiveSigmaPosMean_km','EffectiveSigmaPosStd_km', ...
        'Effective position sigma (km)'; ...
    'MeanStabilityMean','MeanStabilityStd','Mean stability index'; ...
    'AvailableObserversMean','AvailableObserversStd','Mean available observers'; ...
    'ScreeningMean','ScreeningStd','Rejected opportunities'};
fig = create_paper_figure(7.6,7.4);
tiled = tiledlayout(fig,3,2,'Padding','compact','TileSpacing','compact');
colors = lines(numel(measurements));
for q = 1:size(specs,1)
    ax = nexttile(tiled); hold(ax,'on'); box(ax,'on'); grid(ax,'on');
    handles = gobjects(numel(measurements),1);
    for m = 1:numel(measurements)
        values = nan(size(observerCounts)); errors = values;
        for k = 1:numel(observerCounts)
            row = results(results.Mission == mission & ...
                results.Measurement == measurements(m) & ...
                results.NumObservers == observerCounts(k) & results.NPeriods == 1,:);
            assert(height(row) == 1,'Missing observer-count baseline point.');
            values(k) = row.(specs{q,1}); errors(k) = row.(specs{q,2});
        end
        handles(m) = errorbar(ax,observerCounts,values,errors,'-o', ...
            'Color',colors(m,:),'LineWidth',1.7,'MarkerSize',5.5, ...
            'MarkerFaceColor',colors(m,:),'CapSize',7, ...
            'DisplayName',measurement_label(measurements(m)));
    end
    ax.XTick = observerCounts;
    xlim(ax,[min(observerCounts)-0.5 max(observerCounts)+0.5]);
    ylabel(ax,specs{q,3},'FontWeight','bold'); apply_figure_style(ax);
    if q == 1
        lgd = legend(ax,handles,'Orientation','horizontal');
        format_legend(lgd); lgd.Layout.Tile = 'north';
    end
end
xlabel(tiled,'Number of observers','FontName','Times New Roman', ...
    'FontSize',14,'FontWeight','bold');
export_preview(fig,figureDir,stem,saveFigures);
end


function plot_baseline_period_panel(results,measurement,observerCounts,gatewayPeriods, ...
    stem,figureDir,saveFigures)
specs = { ...
    'BestJMean','BestJStd','Final best objective'; ...
    'RMSEPosMean_km','RMSEPosStd_km','Position RMSE (km)'; ...
    'EffectiveSigmaPosMean_km','EffectiveSigmaPosStd_km', ...
        'Effective position sigma (km)'; ...
    'AvailableObserversMean','AvailableObserversStd','Mean available observers'};
fig = create_paper_figure(7.6,6.6);
tiled = tiledlayout(fig,2,2,'Padding','compact','TileSpacing','compact');
colors = lines(numel(observerCounts));
for q = 1:size(specs,1)
    ax = nexttile(tiled); hold(ax,'on'); box(ax,'on'); grid(ax,'on');
    handles = gobjects(numel(observerCounts),1);
    for m = 1:numel(observerCounts)
        values = nan(size(gatewayPeriods)); errors = values;
        for k = 1:numel(gatewayPeriods)
            row = results(results.Mission == "LUNAR_GATEWAY" & ...
                results.Measurement == measurement & ...
                results.NumObservers == observerCounts(m) & ...
                results.NPeriods == gatewayPeriods(k),:);
            assert(height(row) == 1,'Missing Gateway-duration baseline point.');
            values(k) = row.(specs{q,1}); errors(k) = row.(specs{q,2});
        end
        handles(m) = errorbar(ax,gatewayPeriods,values,errors,'-o', ...
            'Color',colors(m,:),'LineWidth',1.7,'MarkerSize',5.5, ...
            'MarkerFaceColor',colors(m,:),'CapSize',7, ...
            'DisplayName',string(observerCounts(m))+" observers");
    end
    ax.XTick = gatewayPeriods; xlim(ax,[0.75 5.25]);
    ylabel(ax,specs{q,3},'FontWeight','bold'); apply_figure_style(ax);
    if q == 1
        lgd = legend(ax,handles,'Orientation','horizontal','NumColumns',2);
        format_legend(lgd); lgd.Layout.Tile = 'north';
    end
end
xlabel(tiled,'Lunar Gateway periods','FontName','Times New Roman', ...
    'FontSize',14,'FontWeight','bold');
export_preview(fig,figureDir,stem,saveFigures);
end


function plot_uncertainty_band(ax,x,meanValues,stdValues,color)
x = double(x(:)); meanValues = double(meanValues(:)); stdValues = double(stdValues(:));
idx = unique(round(linspace(1,numel(x),min(240,numel(x)))));
x = x(idx); meanValues = meanValues(idx); stdValues = stdValues(idx);
bandColor = 0.82*[1 1 1] + 0.18*color;
fill(ax,[x;flipud(x)], ...
    [max(0,meanValues-stdValues);flipud(meanValues+stdValues)], ...
    bandColor,'EdgeColor','none','HandleVisibility','off');
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
close(fig);
end
function [mu,sigma] = sample_statistics(values)
values = double(values(:)); mu = mean(values);
if numel(values) < 2, sigma = NaN; else, sigma = std(values); end
end
