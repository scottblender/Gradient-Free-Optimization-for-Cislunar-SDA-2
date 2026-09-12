function manifest = make_reviewer2_curated_figures(reports,saveFigures)
%MAKE_REVIEWER2_CURATED_FIGURES Create only the manuscript figures needed.
%
% Curated Reviewer-2 output rules:
%   * all quantitative claims use 20-run mean +/- sample standard deviation;
%   * no coverage-fraction figures are produced;
%   * runtime is plotted only for the focused 1200-FE BO study;
%   * screening ON/OFF figures are limited to convergence, RMSE, effective
%     sigma, mean stability, and rejected-measurement counts;
%   * J111/J100/J010/J001 numerical comparisons are table-first; the only
%     objective-component figures are orbit-family-selection summaries;
%   * orbit-family summaries use NHO, SHO, NNRHO, SNRHO, and DRO;
%   * 3-D trajectory figures are produced only for comparison and baseline;
%   * result trajectories use the established introduction-figure renderer;
%   * all 2-D metric, bar, and convergence figures use the same manuscript
%     export size for reliable LaTeX subfigure alignment;
%   * convergence figures show only the 20-run mean best-so-far curves;
%     run-to-run variability remains in metric summaries/tables;
%   * all result axes use concise labels that state when a plotted quantity
%     is a 20-run mean;
%   * final paper figures contain no MATLAB figure titles;
%   * all final axes have grid lines off and the surrounding axes box off.

if nargin < 2 || isempty(saveFigures), saveFigures = true; end
saveFigures = logical(saveFigures);
assert(isstruct(reports),'reports must come from run_reviewer2_results.');
style = reviewer2_paper_style();
manifest = table(strings(0,1),strings(0,1),strings(0,1), ...
    'VariableNames',{'Study','FigureStem','Purpose'});

baselineResults = table();
if isfield(reports,'baseline') && isfield(reports.baseline,'results')
    baselineResults = reports.baseline.results;
end

%% Focused 1200-FE five-method runtime study
if isfield(reports,'runtime')
    r = reports.runtime;
    out = prepare_output(r.analysisDirectory,saveFigures);
    plot_runtime_metric(r,'BestJMean','BestJStd','Mean final best objective', ...
        "runtime_1200_objective",out,saveFigures,style,false,table());
    manifest = add_manifest(manifest,"runtime","runtime_1200_objective", ...
        "Equal-1200-FE mean final-best objective comparison.");

    plot_runtime_metric(r,'BudgetRuntimeMean_s','BudgetRuntimeStd_s', ...
        'Mean runtime to 1200 FE (s)',"runtime_1200_runtime",out,saveFigures,style,true,table());
    manifest = add_manifest(manifest,"runtime","runtime_1200_runtime", ...
        "Equal-1200-FE mean computational cost showing BO scaling penalty.");

    plot_runtime_convergence(r,out,saveFigures,style);
    manifest = add_manifest(manifest,"runtime","runtime_1200_convergence", ...
        "Five-method equal-FE mean convergence comparison.");
end

%% Full 6000-FE optimizer comparison
if isfield(reports,'comparison')
    r = reports.comparison;
    out = prepare_output(r.analysisDirectory,saveFigures);
    missions = string(r.missions);
    refs = matched_baselines(baselineResults,missions,3,1,'BestJMean','BestJStd');

    plot_comparison_metric(r,'BestJMean','BestJStd','Mean final best objective', ...
        "comparison_6000_objective",out,saveFigures,style,refs);
    manifest = add_manifest(manifest,"comparison","comparison_6000_objective", ...
        "Case-wise mean final-best objective with matched GA baselines.");

    specs = { ...
        'RMSEPosMean_km','RMSEPosStd_km','Mean position RMSE (km)','comparison_6000_position_rmse'; ...
        'EffectiveSigmaPosMean_km','EffectiveSigmaPosStd_km','Mean effective position sigma (km)','comparison_6000_effective_sigma'; ...
        'MeanStabilityMean','MeanStabilityStd','Mean observer stability index','comparison_6000_stability'};
    for q = 1:size(specs,1)
        metricRefs = matched_baselines(baselineResults,missions,3,1, ...
            specs{q,1},specs{q,2});
        plot_comparison_metric(r,specs{q,1},specs{q,2},specs{q,3}, ...
            string(specs{q,4}),out,saveFigures,style,metricRefs);
        manifest = add_manifest(manifest,"comparison",string(specs{q,4}), ...
            "Case-wise optimizer metric, mean +/- sample std., with matched GA baseline.");
    end

    for mission = missions
        stem = "comparison_6000_convergence_"+mission_code(mission);
        plot_comparison_convergence(r,mission,out,stem,saveFigures,style);
        manifest = add_manifest(manifest,"comparison",stem, ...
            "GA/PSO/ABC/ACO mean convergence overlaid for direct comparison.");
    end

    familyData = build_comparison_family_data(r);
    writetable(familyData,fullfile(char(r.analysisDirectory), ...
        'comparison_orbit_family_selection.csv'));
    plot_family_grouped_all_cases(familyData,"Optimizer", ...
        "comparison_orbit_family_selection",out,saveFigures,style);
    manifest = add_manifest(manifest,"comparison","comparison_orbit_family_selection", ...
        "All target-case/optimizer selections across NHO/SHO/NNRHO/SNRHO/DRO.");

    % Trajectory panels are intentionally retained only for the comparison
    % and baseline studies because they match the geometry figures in the paper.
    selection = select_reviewer2_representative_runs(r,"comparison");
    writetable(selection,fullfile(char(r.analysisDirectory), ...
        'comparison_6000_geometry_representative_runs.csv'));
    details = plot_reviewer2_geometry_grid(selection,out,"comparison_geometry",saveFigures);
    manifest = add_geometry_manifest(manifest,"comparison",details, ...
        "Representative optimizer geometry nearest each 20-run mean objective.");
end

%% GA baseline sensitivity
if isfield(reports,'baseline')
    r = reports.baseline;
    out = prepare_output(r.analysisDirectory,saveFigures);
    missions = ["LUNAR_GATEWAY","LOW_THRUST_TRANSFER","GATEWAY_IMPULSE"];

    observerSpecs = { ...
        'BestJMean','BestJStd','Mean final best objective','objective'; ...
        'RMSEPosMean_km','RMSEPosStd_km','Mean position RMSE (km)','position_rmse'; ...
        'EffectiveSigmaPosMean_km','EffectiveSigmaPosStd_km','Mean effective position sigma (km)','effective_sigma'};
    for mission = missions
        for q = 1:size(observerSpecs,1)
            stem = "baseline_observer_"+string(observerSpecs{q,4})+"_"+mission_code(mission);
            plot_baseline_observer_metric(r,mission,observerSpecs{q,1}, ...
                observerSpecs{q,2},observerSpecs{q,3},out,stem,saveFigures,style);
            manifest = add_manifest(manifest,"baseline",stem, ...
                "Angles-only / angles-plus-range observer-count trend using mean +/- sample std.");
        end
        for meas = ["ANGLES_ONLY","ANGLES_RANGE"]
            stem = "baseline_convergence_observers_"+mission_code(mission)+ ...
                "_"+measurement_code(meas);
            plot_baseline_observer_convergence(r,mission,meas,out,stem,saveFigures,style);
            manifest = add_manifest(manifest,"baseline",stem, ...
                "3/5/7/10-observer GA convergence comparison.");
        end
    end

    durationSpecs = { ...
        'BestJMean','BestJStd','Mean final best objective','objective'; ...
        'RMSEPosMean_km','RMSEPosStd_km','Mean position RMSE (km)','position_rmse'; ...
        'EffectiveSigmaPosMean_km','EffectiveSigmaPosStd_km','Mean effective position sigma (km)','effective_sigma'};
    for meas = ["ANGLES_ONLY","ANGLES_RANGE"]
        for q = 1:size(durationSpecs,1)
            stem = "baseline_gateway_duration_"+string(durationSpecs{q,4})+ ...
                "_"+measurement_code(meas);
            plot_baseline_duration_metric(r,meas,durationSpecs{q,1}, ...
                durationSpecs{q,2},durationSpecs{q,3},out,stem,saveFigures,style);
            manifest = add_manifest(manifest,"baseline",stem, ...
                "Gateway-duration trend for 3/5/7/10 observers.");
        end
        stem = "baseline_convergence_duration_"+measurement_code(meas);
        plot_baseline_duration_convergence(r,meas,out,stem,saveFigures,style);
        manifest = add_manifest(manifest,"baseline",stem, ...
            "1/3/5-period GA convergence comparison.");
    end

    familyData = build_baseline_family_data(r);
    writetable(familyData,fullfile(char(r.analysisDirectory), ...
        'baseline_orbit_family_selection.csv'));
    plot_family_grouped_all_cases(familyData,"Number of observers", ...
        "baseline_orbit_family_selection",out,saveFigures,style);
    manifest = add_manifest(manifest,"baseline","baseline_orbit_family_selection", ...
        "Angles-only, one-period 3/5/7/10-observer selections across all five orbit families.");

    selection = select_reviewer2_representative_runs(r,"baseline");
    writetable(selection,fullfile(char(r.analysisDirectory), ...
        'baseline_6000_geometry_representative_runs.csv'));
    details = plot_reviewer2_geometry_grid(selection,out,"baseline_geometry",saveFigures);
    manifest = add_geometry_manifest(manifest,"baseline",details, ...
        "Representative angles-only geometry nearest the mean objective for each observer count.");
end

%% GA screening/objective sensitivity
if isfield(reports,'objective_screening')
    r = reports.objective_screening;
    out = prepare_output(r.analysisDirectory,saveFigures);
    missions = ["LUNAR_GATEWAY","LOW_THRUST_TRANSFER","GATEWAY_IMPULSE"];

    % Screening ON/OFF: only the physical metrics requested for the paper.
    screeningSpecs = { ...
        'RMSEPosMean_km','RMSEPosStd_km','Mean position RMSE (km)','ga_screening_position_rmse'; ...
        'EffectiveSigmaPosMean_km','EffectiveSigmaPosStd_km','Mean effective position sigma (km)','ga_screening_effective_sigma'; ...
        'MeanStabilityMean','MeanStabilityStd','Mean observer stability index','ga_screening_stability'; ...
        'ScreeningMean','ScreeningStd','Mean screening violations','ga_screening_event_count'};
    for q = 1:size(screeningSpecs,1)
        plot_screening_metric_all_cases(r.results,screeningSpecs{q,1}, ...
            screeningSpecs{q,2},screeningSpecs{q,3},string(screeningSpecs{q,4}), ...
            out,saveFigures,style);
        manifest = add_manifest(manifest,"objective_screening",string(screeningSpecs{q,4}), ...
            "Screening ON/OFF metric across all three target cases.");
    end

    for mission = missions
        stem = "ga_screening_convergence_"+mission_code(mission);
        plot_screening_convergence(r,mission,out,stem,saveFigures,style);
        manifest = add_manifest(manifest,"objective_screening",stem, ...
            "Matched J111 screening ON/OFF convergence.");
    end

    % Objective-component performance remains table-first. Retain only the
    % family-selection summary requested for manuscript interpretation.
    familyData = build_objective_family_data(r);
    writetable(familyData,fullfile(char(r.analysisDirectory), ...
        'ga_objective_orbit_family_selection.csv'));
    for mission = missions
        stem = "ga_objective_orbit_family_selection_"+mission_code(mission);
        plot_objective_family_by_mission(familyData,mission,stem,out,saveFigures,style);
        manifest = add_manifest(manifest,"objective_screening",stem, ...
            "Objective-dependent NHO/SHO/NNRHO/SNRHO/DRO selection; combined case labeled simply Combined.");
    end

    % No objective/screening trajectory figures are generated by design.
end

write_manifests(reports,manifest);
end


%% Output/manifest helpers
function out = prepare_output(analysisDir,saveFigures)
out = string(analysisDir);
if saveFigures
    assert(isfolder(out),'Analysis directory does not exist: %s',out);
    clear_rendered_figures(out);
end
end

function clear_rendered_figures(out)
for pattern = ["*.eps","*.png"]
    files = dir(fullfile(char(out),char(pattern)));
    for k = 1:numel(files)
        delete(fullfile(files(k).folder,files(k).name));
    end
end
end

function manifest = add_manifest(manifest,study,stem,purpose)
manifest = [manifest;table(string(study),string(stem),string(purpose), ...
    'VariableNames',manifest.Properties.VariableNames)];
end

function manifest = add_geometry_manifest(manifest,study,details,purpose)
if isempty(details), return; end
for k = 1:height(details)
    manifest = add_manifest(manifest,study,string(details.FigureStem(k)),purpose);
end
end

function write_manifests(reports,manifest)
fields = ["runtime","comparison","baseline","objective_screening"];
for field = fields
    if ~isfield(reports,field), continue; end
    rows = manifest(manifest.Study == field,:);
    writetable(rows,fullfile(char(reports.(field).analysisDirectory), ...
        'paper_figure_manifest.csv'));
end
end


%% Baseline references
function ref = matched_baseline(B,mission,numObservers,nPeriods,valueField,stdField)
if nargin < 6, stdField = 'BestJStd'; end
if nargin < 5, valueField = 'BestJMean'; end
ref = table();
if isempty(B), return; end
rows = B(B.Mission == mission & B.Measurement == "ANGLES_ONLY" & ...
    B.NumObservers == numObservers & B.NPeriods == nPeriods,:);
if height(rows) ~= 1, return; end
ref = table(string(mission),rows.(valueField),rows.(stdField), ...
    'VariableNames',{'Mission','Mean','Std'});
end

function refs = matched_baselines(B,missions,numObservers,nPeriods,valueField,stdField)
if nargin < 6, stdField = 'BestJStd'; end
if nargin < 5, valueField = 'BestJMean'; end
refs = table();
for mission = string(missions(:)')
    row = matched_baseline(B,mission,numObservers,nPeriods,valueField,stdField);
    if ~isempty(row), refs = [refs;row]; end
end
end


%% Runtime/comparison figures
function plot_runtime_metric(r,valueField,stdField,yLabel,stem,out,saveFigures,style,annotateBO,baseline)
R = r.runtimeResults;
order = style.optimizerOrder(ismember(style.optimizerOrder,R.Optimizer));
R = sort_to_order(R,'Optimizer',order);
colors = colors_for_optimizers(R.Optimizer,style);
fig = paper_figure(style.metricFigureWidth,style.metricFigureHeight,style);
ax = axes(fig); hold(ax,'on'); box(ax,'off'); grid(ax,'off');
values = R.(valueField); errors = R.(stdField);
b = bar(ax,1:height(R),values,style.groupedBarWidth,'FaceColor','flat'); b.CData = colors;
errorbar(ax,1:height(R),values,errors,'k.','LineWidth',1.0, ...
    'CapSize',style.capSize,'HandleVisibility','off');
format_category_axis(ax,optimizer_labels(R.Optimizer),yLabel,style);
legendHandles = b; legendLabels = optimizer_labels(R.Optimizer);
if ~isempty(baseline)
    hBase = plot(ax,[0.55 height(R)+0.45],[baseline.Mean baseline.Mean],'--', ...
        'Color',[0.30 0.30 0.30],'LineWidth',1.5, ...
        'DisplayName','GA baseline');
    legendHandles = [legendHandles;hBase];
    legendLabels = [legendLabels;"GA baseline"];
end
% Keep the BO bar uncluttered. The runtime scale and standard-deviation bars
% already show the computational penalty without a ratio callout.
if annotateBO
    % Retained as an input for compatibility with the dedicated renderer.
end
lgd = legend(ax,legendHandles,cellstr(legendLabels),'Location','northoutside', ...
    'Orientation','horizontal','NumColumns',min(numel(legendLabels),6),'Box','off');
style_legend(lgd,ax,style); export_figure(fig,out,stem,saveFigures,style);
end

function plot_runtime_convergence(r,out,saveFigures,style)
files = dir(fullfile(char(r.analysisDirectory),'convergence_*.mat'));
assert(numel(files) == 1,'Expected one runtime convergence file.');
S = load(fullfile(files(1).folder,files(1).name),'curves');
optimizers = style.optimizerOrder(ismember(style.optimizerOrder,upper(string({S.curves.optimizer}))));
curves = cell(numel(optimizers),1);
for k = 1:numel(optimizers)
    curves{k} = S.curves(find(upper(string({S.curves.optimizer})) == optimizers(k),1));
end
plot_curve_overlay(curves,optimizer_labels(optimizers),colors_for_optimizers(optimizers,style), ...
    r.budget,out,"runtime_1200_convergence",saveFigures,style);
end

function plot_comparison_metric(r,valueField,stdField,yLabel,stem,out,saveFigures,style,baselineRefs)
R = r.results; missions = string(r.missions); optimizers = string(r.optimizers);
[values,errors] = grouped_values(R,missions,optimizers,valueField,stdField);
fig = paper_figure(style.metricFigureWidth,style.metricFigureHeight,style);
ax = axes(fig); hold(ax,'on'); box(ax,'off'); grid(ax,'off');
b = bar(ax,1:numel(missions),values,'grouped','BarWidth',style.groupedBarWidth); drawnow;
for k = 1:numel(optimizers)
    b(k).FaceColor = optimizer_color(optimizers(k),style);
    errorbar(ax,b(k).XEndPoints,values(:,k),errors(:,k),'k.', ...
        'LineWidth',0.9,'CapSize',style.capSize,'HandleVisibility','off');
end
ax.XTick = 1:numel(missions); ax.XTickLabel = cellstr(mission_labels(missions));
xlabel(ax,'Target case','FontWeight','bold');
ylabel(ax,yLabel,'FontWeight','bold'); style_axes(ax,style);
legendHandles = b(:); legendLabels = optimizer_labels(optimizers);
if ~isempty(baselineRefs)
    hBase = gobjects(1,1);
    for m = 1:numel(missions)
        row = baselineRefs(baselineRefs.Mission == missions(m),:);
        if height(row) ~= 1, continue; end
        h = plot(ax,[m-0.46 m+0.46],[row.Mean row.Mean],'--', ...
            'Color',[0.30 0.30 0.30],'LineWidth',1.5,'HandleVisibility','off');
        if ~isgraphics(hBase), hBase = h; end
    end
    if isgraphics(hBase)
        set(hBase,'HandleVisibility','on','DisplayName','GA baseline');
        legendHandles = [legendHandles;hBase];
        legendLabels = [legendLabels;"GA baseline"];
    end
end
lgd = legend(ax,legendHandles,cellstr(legendLabels),'Location','northoutside', ...
    'Orientation','horizontal','NumColumns',min(numel(legendLabels),5),'Box','off');
style_legend(lgd,ax,style); export_figure(fig,out,stem,saveFigures,style);
end

function plot_comparison_convergence(r,mission,out,stem,saveFigures,style)
key = comparison_key_for_mission(r,mission);
S = load(fullfile(char(r.analysisDirectory),"convergence_"+key+".mat"),'curves');
optimizers = string(r.optimizers); curves = cell(numel(optimizers),1);
for k = 1:numel(optimizers)
    idx = find(upper(string({S.curves.optimizer})) == optimizers(k),1);
    assert(~isempty(idx),'Missing convergence curve for %s.',optimizers(k));
    curves{k} = S.curves(idx);
end
plot_curve_overlay(curves,optimizer_labels(optimizers),colors_for_optimizers(optimizers,style), ...
    r.budget,out,stem,saveFigures,style);
end

function key = comparison_key_for_mission(r,mission)
rows = r.results(r.results.Mission == mission,:);
if ismember('ComparisonKey',rows.Properties.VariableNames)
    keys = unique(string(rows.ComparisonKey));
else
    s = r.summary(r.summary.mission == mission,:); keys = unique(string(s.comparison_key));
end
assert(numel(keys) == 1,'Expected one comparison key for %s.',mission); key = keys(1);
end


%% Baseline figures
function plot_baseline_observer_metric(r,mission,valueField,stdField,yLabel,out,stem,saveFigures,style)
R = r.results; measurements = ["ANGLES_ONLY","ANGLES_RANGE"]; counts = [3 5 7 10];
fig = paper_figure(style.metricFigureWidth,style.metricFigureHeight,style);
ax = axes(fig); hold(ax,'on'); box(ax,'off'); grid(ax,'off'); handles = gobjects(2,1);
for m = 1:2
    values = nan(size(counts)); errors = values;
    for k = 1:numel(counts)
        row = R(R.Mission == mission & R.Measurement == measurements(m) & ...
            R.NumObservers == counts(k) & R.NPeriods == 1,:);
        assert(height(row) == 1,'Missing baseline observer-count point.');
        values(k) = row.(valueField); errors(k) = row.(stdField);
    end
    c = style.measurementColors(m,:);
    handles(m) = errorbar(ax,counts,values,errors,'-o','Color',c,'LineWidth',style.lineWidth, ...
        'MarkerSize',style.markerSize,'MarkerFaceColor',c,'CapSize',style.capSize);
end
ax.XTick = counts; xlabel(ax,'Number of observers','FontWeight','bold');
ylabel(ax,yLabel,'FontWeight','bold'); style_axes(ax,style);
lgd = legend(ax,handles,{'Angles only','Angles + range'}, ...
    'Location','northoutside','Orientation','horizontal','Box','off');
style_legend(lgd,ax,style); export_figure(fig,out,stem,saveFigures,style);
end

function plot_baseline_duration_metric(r,measurement,valueField,stdField,yLabel,out,stem,saveFigures,style)
R = r.results; counts = [3 5 7 10]; periods = [1 3 5]; colors = lines(numel(counts));
fig = paper_figure(style.metricFigureWidth,style.metricFigureHeight,style);
ax = axes(fig); hold(ax,'on'); box(ax,'off'); grid(ax,'off'); handles = gobjects(numel(counts),1);
for k = 1:numel(counts)
    values = nan(size(periods)); errors = values;
    for p = 1:numel(periods)
        row = R(R.Mission == "LUNAR_GATEWAY" & R.Measurement == measurement & ...
            R.NumObservers == counts(k) & R.NPeriods == periods(p),:);
        assert(height(row) == 1,'Missing Gateway duration point.');
        values(p) = row.(valueField); errors(p) = row.(stdField);
    end
    handles(k) = errorbar(ax,periods,values,errors,'-o','Color',colors(k,:), ...
        'LineWidth',style.lineWidth,'MarkerSize',style.markerSize, ...
        'MarkerFaceColor',colors(k,:),'CapSize',style.capSize, ...
        'DisplayName',sprintf('%d observers',counts(k)));
end
ax.XTick = periods; xlabel(ax,'Tracking duration (Gateway periods)','FontWeight','bold');
ylabel(ax,yLabel,'FontWeight','bold'); style_axes(ax,style);
lgd = legend(ax,handles,'Location','northoutside','Orientation','horizontal','NumColumns',2,'Box','off');
style_legend(lgd,ax,style); export_figure(fig,out,stem,saveFigures,style);
end

function plot_baseline_observer_convergence(r,mission,measurement,out,stem,saveFigures,style)
counts = [3 5 7 10]; curves = cell(4,1); labels = strings(4,1); colors = lines(4);
for k = 1:4
    row = r.results(r.results.Mission == mission & r.results.Measurement == measurement & ...
        r.results.NumObservers == counts(k) & r.results.NPeriods == 1,:);
    assert(height(row) == 1,'Missing baseline convergence configuration.');
    curves{k} = load_ga_curve(r.analysisDirectory,row.ComparisonKey); labels(k) = string(counts(k))+" observers";
end
plot_curve_overlay(curves,labels,colors,r.budget,out,stem,saveFigures,style);
end

function plot_baseline_duration_convergence(r,measurement,out,stem,saveFigures,style)
periods = [1 3 5]; curves = cell(3,1); labels = strings(3,1); colors = lines(3);
for k = 1:3
    row = r.results(r.results.Mission == "LUNAR_GATEWAY" & r.results.Measurement == measurement & ...
        r.results.NumObservers == 3 & r.results.NPeriods == periods(k),:);
    assert(height(row) == 1,'Missing baseline duration convergence configuration.');
    curves{k} = load_ga_curve(r.analysisDirectory,row.ComparisonKey); labels(k) = string(periods(k))+" period"+plural_s(periods(k));
end
plot_curve_overlay(curves,labels,colors,r.budget,out,stem,saveFigures,style);
end


%% Screening figures
function plot_screening_metric_all_cases(R,valueField,stdField,yLabel,stem,out,saveFigures,style)
missions = ["LUNAR_GATEWAY","LOW_THRUST_TRANSFER","GATEWAY_IMPULSE"];
configs = ["combined_on","combined_off"]; values = nan(3,2); errors = values;
for m = 1:3
    for c = 1:2
        row = objective_result(R,missions(m),configs(c));
        values(m,c) = row.(valueField); errors(m,c) = row.(stdField);
    end
end
fig = paper_figure(style.metricFigureWidth,style.metricFigureHeight,style);
ax = axes(fig); hold(ax,'on'); box(ax,'off'); grid(ax,'off');
b = bar(ax,1:3,values,'grouped','BarWidth',style.groupedBarWidth); drawnow;
for c = 1:2
    b(c).FaceColor = style.configurationColors(c,:);
    errorbar(ax,b(c).XEndPoints,values(:,c),errors(:,c),'k.','LineWidth',0.9, ...
        'CapSize',style.capSize,'HandleVisibility','off');
end
ax.XTick = 1:3; ax.XTickLabel = cellstr(mission_labels(missions));
xlabel(ax,'Target case','FontWeight','bold');
ylabel(ax,yLabel,'FontWeight','bold'); style_axes(ax,style);
lgd = legend(ax,b,{'Screening ON','Screening OFF'},'Location','northoutside', ...
    'Orientation','horizontal','Box','off'); style_legend(lgd,ax,style);
export_figure(fig,out,stem,saveFigures,style);
end

function plot_screening_convergence(r,mission,out,stem,saveFigures,style)
configs = ["combined_on","combined_off"]; curves = cell(2,1);
for k = 1:2
    row = objective_result(r.results,mission,configs(k));
    curves{k} = load_ga_curve(r.analysisDirectory,row.ComparisonKey);
end
plot_curve_overlay(curves,["Screening ON","Screening OFF"], ...
    style.configurationColors(1:2,:),r.budget,out,stem,saveFigures,style);
end


%% Five-family selection summaries
function T = build_comparison_family_data(r)
missions = ["LUNAR_GATEWAY","LOW_THRUST_TRANSFER","GATEWAY_IMPULSE"];
optimizers = ["GA","PSO","ABC","ACO"]; T = empty_family_table();
for mission = missions
    for optimizer = optimizers
        row = r.results(r.results.Mission == mission & r.results.Optimizer == optimizer,:);
        assert(height(row) == 1,'Missing comparison family-selection group.');
        key = row.ComparisonKey;
        runs = r.runMetrics(r.runMetrics.comparison_key == key & r.runMetrics.optimizer == optimizer,:);
        assert(height(runs) == 20,'Expected 20 comparison runs for family summary.');
        T = [T; family_rows(mission,optimizer,optimizer,runs.run_file)];
    end
end
end

function T = build_baseline_family_data(r)
missions = ["LUNAR_GATEWAY","LOW_THRUST_TRANSFER","GATEWAY_IMPULSE"]; counts = [3 5 7 10];
T = empty_family_table();
for mission = missions
    for nObs = counts
        row = r.results(r.results.Mission == mission & r.results.Measurement == "ANGLES_ONLY" & ...
            r.results.NumObservers == nObs & r.results.NPeriods == 1,:);
        assert(height(row) == 1,'Missing baseline family-selection group.');
        runs = r.runMetrics(r.runMetrics.comparison_key == row.ComparisonKey,:);
        assert(height(runs) == 20,'Expected 20 baseline runs for family summary.');
        key = "o"+string(nObs); label = string(nObs);
        T = [T; family_rows(mission,key,label,runs.run_file)];
    end
end
end

function T = build_objective_family_data(r)
missions = ["LUNAR_GATEWAY","LOW_THRUST_TRANSFER","GATEWAY_IMPULSE"];
configs = ["combined_on","j1_only","j2_only","j3_only"]; labels = ["Combined","J_1","J_2","J_3"];
T = empty_family_table();
for mission = missions
    for k = 1:numel(configs)
        row = objective_result(r.results,mission,configs(k));
        runs = r.runMetrics(r.runMetrics.comparison_key == row.ComparisonKey,:);
        assert(height(runs) == 20,'Expected 20 objective-component runs for family summary.');
        T = [T; family_rows(mission,configs(k),labels(k),runs.run_file)];
    end
end
end

function T = empty_family_table()
T = table(strings(0,1),strings(0,1),strings(0,1),strings(0,1),zeros(0,1),zeros(0,1), ...
    'VariableNames',{'Mission','GroupKey','GroupLabel','Family','Count','Fraction'});
end

function T = family_rows(mission,key,label,runFiles)
families = ["NHO","SHO","NNRHO","SNRHO","DRO"]; selected = strings(0,1);
for j = 1:numel(runFiles)
    S = load(runFiles(j),'runState');
    assert(isfield(S.runState,'observers') && istable(S.runState.observers), ...
        'Run is missing selected observer table.');
    obs = S.runState.observers; names = string(obs.Properties.VariableNames);
    if ismember("orbit_family",names)
        raw = string(obs.orbit_family);
    elseif ismember("orbitFamily",names)
        raw = string(obs.orbitFamily);
    else
        error('Family:MissingObserverFamily','Selected observer table has no orbit-family field.');
    end
    for u = 1:numel(raw), selected(end+1,1) = manuscript_family(raw(u)); end
end
assert(~isempty(selected),'No selected observer families were found.');
T = empty_family_table();
for f = 1:numel(families)
    n = sum(selected == families(f));
    T = [T;table(string(mission),string(key),string(label),families(f),n,n/numel(selected), ...
        'VariableNames',T.Properties.VariableNames)];
end
end

function family = manuscript_family(raw)
name = upper(string(raw));
if name == "DRO" || contains(name,"DRO")
    family = "DRO";
elseif startsWith(name,"NNRH") || contains(name,"NNRHO")
    family = "NNRHO";
elseif startsWith(name,"SNRH") || contains(name,"SNRHO")
    family = "SNRHO";
elseif startsWith(name,"NH") || name == "NHO"
    family = "NHO";
elseif startsWith(name,"SH") || name == "SHO"
    family = "SHO";
else
    error('Family:UnknownObserverFamily','Unexpected selected observer family: %s',name);
end
end

function plot_family_grouped_all_cases(T,groupAxisLabel,stem,out,saveFigures,style)
missions = ["LUNAR_GATEWAY","LOW_THRUST_TRANSFER","GATEWAY_IMPULSE"];
families = ["NHO","SHO","NNRHO","SNRHO","DRO"];
% Preserve the first mission's group ordering and reuse it for all missions.
first = T(T.Mission == missions(1),:);
groupKeys = unique(first.GroupKey,'stable');
nPer = numel(groupKeys);

% Dense optimizer labels are spaced explicitly rather than rotated or shrunk.
% This keeps GA/PSO/ABC/ACO legible at manuscript scale while retaining clear
% visual separation between the LG, LT, and GI target-case groups.
withinGroupSpacing = 1.35;
caseGap = 2.20;
x = [];
V = [];
tickLabels = strings(0,1);
centers = zeros(3,1);
for m = 1:3
    startX = 1 + (m-1)*((nPer-1)*withinGroupSpacing + caseGap + withinGroupSpacing);
    xs = startX + (0:nPer-1)*withinGroupSpacing;
    centers(m) = mean(xs);
    x = [x xs];
    for g = 1:nPer
        rows = T(T.Mission == missions(m) & T.GroupKey == groupKeys(g),:);
        assert(height(rows) == 5,'Family summary must contain all five families.');
        values = zeros(1,5);
        for f = 1:5
            row = rows(rows.Family == families(f),:);
            assert(height(row) == 1,'Missing family-selection fraction.');
            values(f) = row.Fraction;
        end
        V(end+1,:) = 100*values;
        tickLabels(end+1,1) = rows.GroupLabel(1);
    end
end

fig = paper_figure(style.metricFigureWidth,style.metricFigureHeight,style);
ax = axes(fig); hold(ax,'on'); box(ax,'off'); grid(ax,'off');
b = bar(ax,x,V,'stacked','BarWidth',0.66); colors = lines(5);
for f = 1:5, b(f).FaceColor = colors(f,:); end
ax.XTick = x;
ax.XTickLabel = cellstr(tickLabels);
xlabel(ax,groupAxisLabel,'FontWeight','bold');
ylabel(ax,'Observer selections (%)','FontWeight','bold');
style_axes(ax,style);

% space_manuscript_bars applies the general dense-category rotation rule;
% override it here because the explicit x spacing makes horizontal optimizer
% labels readable and avoids the previous PSO/ABC/ACO overlap.
ax.XTick = x;
ax.XTickLabel = cellstr(tickLabels);
ax.XTickLabelRotation = 0;
xlim(ax,[min(x)-0.80*withinGroupSpacing,max(x)+0.80*withinGroupSpacing]);
ylim(ax,[0 112]);
for m = 1:3
    text(ax,centers(m),106,mission_short_label(missions(m)), ...
        'HorizontalAlignment','center','VerticalAlignment','middle', ...
        'FontName',style.fontName,'FontSize',style.fontSize,'FontWeight','bold');
end

lgd = legend(ax,b,cellstr(families),'Location','northoutside', ...
    'Orientation','horizontal','NumColumns',5,'Box','off');
style_legend(lgd,ax,style);
export_figure(fig,out,stem,saveFigures,style);
end

function plot_objective_family_by_mission(T,mission,stem,out,saveFigures,style)
families = ["NHO","SHO","NNRHO","SNRHO","DRO"];
keys = ["combined_on","j1_only","j2_only","j3_only"]; labels = ["Combined","J_1","J_2","J_3"];
V = zeros(4,5);
for g = 1:4
    rows = T(T.Mission == mission & T.GroupKey == keys(g),:);
    assert(height(rows) == 5,'Objective family figure requires all five families.');
    for f = 1:5, V(g,f) = 100*rows.Fraction(rows.Family == families(f)); end
end
fig = paper_figure(style.metricFigureWidth,style.metricFigureHeight,style);
ax = axes(fig); hold(ax,'on'); box(ax,'off'); grid(ax,'off');
b = bar(ax,1:4,V,'stacked','BarWidth',0.78); colors = lines(5);
for f = 1:5, b(f).FaceColor = colors(f,:); end
ax.XTick = 1:4; ax.XTickLabel = cellstr(labels); ylim(ax,[0 100]);
xlabel(ax,'Objective configuration','FontWeight','bold'); ylabel(ax,'Observer selections (%)','FontWeight','bold');
style_axes(ax,style); lgd = legend(ax,b,cellstr(families),'Location','northoutside', ...
    'Orientation','horizontal','NumColumns',5,'Box','off'); style_legend(lgd,ax,style);
export_figure(fig,out,stem,saveFigures,style);
end


%% Shared convergence/data helpers
function plot_curve_overlay(curves,labels,colors,budget,out,stem,saveFigures,style)
fig = paper_figure(style.convergenceFigureWidth,style.convergenceFigureHeight,style);
ax = axes(fig); hold(ax,'on'); box(ax,'off'); grid(ax,'off');
handles = gobjects(numel(curves),1); allY = zeros(0,1);
for k = 1:numel(curves)
    c = curves{k}; valid = c.fe >= 60 & isfinite(c.mean); assert(any(valid),'No convergence FE >= 60.');
    x = double(c.fe(valid)); y = double(c.mean(valid));
    handles(k) = stairs(ax,x,y,'Color',colors(k,:),'LineWidth',style.lineWidth,'DisplayName',string(labels(k)));
    allY = [allY;y];
end
allY = allY(isfinite(allY)); lo = min(allY); hi = max(allY); span = max(hi-lo,0.05*max(1,abs(hi)));
ylim(ax,[lo-0.06*span hi+0.08*span]); xlim(ax,[60 budget]);
xlabel(ax,'Function evaluations','FontWeight','bold'); ylabel(ax,'Mean best-so-far objective','FontWeight','bold');
style_axes(ax,style); lgd = legend(ax,handles,'Location','northoutside','Orientation','horizontal', ...
    'NumColumns',min(numel(handles),5),'Box','off'); style_legend(lgd,ax,style);
export_figure(fig,out,stem,saveFigures,style);
end

function curve = load_ga_curve(analysisDir,key)
S = load(fullfile(char(analysisDir),"convergence_"+string(key)+".mat"),'curves');
idx = find(upper(string({S.curves.optimizer})) == "GA",1); assert(~isempty(idx),'Missing GA curve.'); curve = S.curves(idx);
end

function [values,errors] = grouped_values(R,missions,optimizers,valueField,errorField)
values = nan(numel(missions),numel(optimizers)); errors = values;
for m = 1:numel(missions)
    for k = 1:numel(optimizers)
        row = R(R.Mission == missions(m) & R.Optimizer == optimizers(k),:); assert(height(row) == 1,'Missing comparison point.');
        values(m,k) = row.(valueField); errors(m,k) = row.(errorField);
    end
end
end

function row = objective_result(R,mission,config)
row = R(R.Mission == mission & string(R.Configuration) == config,:);
assert(height(row) == 1,'Missing objective/screening result for %s/%s.',mission,config);
end

function R = sort_to_order(R,field,order)
idx = nan(numel(order),1); for k = 1:numel(order), idx(k) = find(string(R.(field)) == order(k),1); end; R = R(idx,:);
end

function colors = colors_for_optimizers(optimizers,style)
colors = zeros(numel(optimizers),3); for k = 1:numel(optimizers), colors(k,:) = optimizer_color(optimizers(k),style); end
end

function c = optimizer_color(optimizer,style)
idx = find(style.optimizerOrder == upper(string(optimizer)),1); assert(~isempty(idx),'Unknown optimizer.'); c = style.optimizerColors(idx,:);
end


%% Figure styling/export
function fig = paper_figure(widthIn,heightIn,style)
fig = figure('Color','w','Units','inches','Position',[1 1 widthIn heightIn], ...
    'PaperUnits','inches','PaperSize',[widthIn heightIn], ...
    'PaperPosition',[0 0 widthIn heightIn],'PaperPositionMode','manual', ...
    'Renderer','painters','InvertHardcopy','off');
movegui(fig,'center'); set(fig,'DefaultAxesFontName',style.fontName,'DefaultAxesFontSize',style.fontSize);
end

function style_axes(ax,style)
set(ax,'Units','normalized','Position',style.metricPlotPosition);
set(ax,'FontName',style.fontName,'FontSize',style.fontSize,'FontWeight','bold', ...
    'LineWidth',style.axisLineWidth,'TickDir','out','Layer','top', ...
    'Box','off','XGrid','off','YGrid','off','ZGrid','off');
ax.XLabel.FontSize = style.labelFontSize; ax.YLabel.FontSize = style.labelFontSize;
wrap_manuscript_label(ax.XLabel); wrap_manuscript_label(ax.YLabel);
space_manuscript_bars(ax,style);
end

function style_legend(lgd,ax,style)
lgd.FontName = style.fontName; lgd.FontSize = style.legendFontSize; lgd.FontWeight = 'bold';
format_manuscript_legend(ax,lgd,style,style.metricPlotPosition);
end

function format_category_axis(ax,labels,yLabel,style)
ax.XTick = 1:numel(labels); ax.XTickLabel = cellstr(labels); ax.XTickLabelRotation = 18;
xlabel(ax,'Optimizer','FontWeight','bold');
ylabel(ax,yLabel,'FontWeight','bold'); style_axes(ax,style);
end

function export_figure(fig,out,stem,saveFigures,style)
drawnow; if ~saveFigures, return; end
base = fullfile(char(out),char(stem)); finalize_manuscript_figure(fig); print(fig,[base '.eps'],'-depsc2','-painters','-r600','-loose');
exportgraphics(fig,[base '.png'],'Resolution',style.exportDpi); close(fig);
end

function enforce_minimum_font_size(fig,minFontSize)
objects = findall(fig,'-property','FontSize');
for k = 1:numel(objects)
    try, if objects(k).FontSize < minFontSize, objects(k).FontSize = minFontSize; end, catch, end
end
end


%% Labels
function labels = optimizer_labels(values)
values = upper(string(values(:))); labels = values; labels(values == "BAYESIAN") = "BO";
end

function labels = mission_labels(values)
values = string(values(:)); labels = strings(size(values));
for k = 1:numel(values)
    switch values(k)
        case "LUNAR_GATEWAY", labels(k) = "Lunar Gateway";
        case "LOW_THRUST_TRANSFER", labels(k) = "Low-thrust transfer";
        case "GATEWAY_IMPULSE", labels(k) = "Gateway impulse";
        otherwise, labels(k) = values(k);
    end
end
end

function label = mission_short_label(mission)
switch string(mission)
    case "LUNAR_GATEWAY", label = "Lunar Gateway";
    case "LOW_THRUST_TRANSFER", label = "Low-thrust";
    case "GATEWAY_IMPULSE", label = "Gateway impulse";
    otherwise, label = string(mission);
end
end

function code = mission_code(mission)
switch string(mission)
    case "LUNAR_GATEWAY", code = "lg";
    case "LOW_THRUST_TRANSFER", code = "lt";
    case "GATEWAY_IMPULSE", code = "gi";
    otherwise, code = lower(string(mission));
end
end

function code = measurement_code(measurement)
if string(measurement) == "ANGLES_ONLY", code = "ao"; else, code = "ar"; end
end

function s = plural_s(value)
if value == 1, s = ""; else, s = "s"; end
end
