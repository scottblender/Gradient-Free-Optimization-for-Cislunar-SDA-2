function manifest = make_reviewer2_paper_figures(reports,saveFigures)
%MAKE_REVIEWER2_PAPER_FIGURES Create the curated Reviewer 2 manuscript plots.
%
% This function consumes report structs returned by the four final result
% processors. It does not run optimization. The figures are deliberately
% limited to plots that support the paper conclusions:
%   runtime: BO equal-FE runtime penalty and convergence/objective benefit;
%   comparison: overall/case optimizer quality, metrics, convergence, geometry;
%   baseline: AO/AR, observer-count, duration, convergence, geometry;
%   objective/screening: screening effects, component effects, families,
%                        convergence, and specialized constellation geometry.
%
% All axes/legends use Times New Roman with 12-point minimum text.

if nargin < 2 || isempty(saveFigures), saveFigures = true; end
validateattributes(saveFigures,{'logical','numeric'},{'scalar'});
saveFigures = logical(saveFigures);
assert(isstruct(reports),'reports must be the struct returned by run_reviewer2_results.');
style = reviewer2_paper_style();
manifest = table(strings(0,1),strings(0,1),strings(0,1), ...
    'VariableNames',{'Study','FigureStem','Purpose'});

if isfield(reports,'runtime')
    r = reports.runtime;
    out = prepare_output(r.analysisDirectory,saveFigures);
    plot_runtime_summary(r,out,saveFigures,style);
    manifest = add_manifest(manifest,"runtime","runtime_1200_summary", ...
        "Equal-FE final objective and runtime; highlights BO runtime penalty.");
    plot_runtime_convergence(r,out,saveFigures,style);
    manifest = add_manifest(manifest,"runtime","runtime_1200_convergence", ...
        "Five-method mean best-so-far convergence at the common 1200-FE budget.");
end

if isfield(reports,'comparison')
    r = reports.comparison;
    out = prepare_output(r.analysisDirectory,saveFigures);
    plot_comparison_summary(r,out,saveFigures,style);
    manifest = add_manifest(manifest,"comparison","comparison_6000_summary", ...
        "Case-wise objective, RMSE, uncertainty, and runtime comparison.");
    for mission = string(r.missions)
        stem = "comparison_6000_convergence_"+mission_code(mission);
        plot_comparison_convergence(r,mission,out,stem,saveFigures,style);
        manifest = add_manifest(manifest,"comparison",stem, ...
            "Four-method 6000-FE mean best-so-far convergence.");
    end
    plot_reviewer2_geometry_grid(r.bestGeometryRuns,out, ...
        "comparison_geometry",saveFigures);
    for mission = string(r.missions)
        stem = "comparison_geometry_"+mission_code(mission)+"_grid";
        manifest = add_manifest(manifest,"comparison",stem, ...
            "Best observed constellation geometry for each optimizer.");
    end
end

if isfield(reports,'baseline')
    r = reports.baseline;
    out = prepare_output(r.analysisDirectory,saveFigures);
    missions = ["LUNAR_GATEWAY","LOW_THRUST_TRANSFER","GATEWAY_IMPULSE"];
    for mission = missions
        stem = "baseline_observer_trends_"+mission_code(mission);
        plot_baseline_observer_trends(r,mission,out,stem,saveFigures,style);
        manifest = add_manifest(manifest,"baseline",stem, ...
            "AO/AR sensitivity to 3, 5, 7, and 10 observers.");
        for meas = ["ANGLES_ONLY","ANGLES_RANGE"]
            cstem = "baseline_convergence_observers_"+mission_code(mission)+ ...
                "_"+measurement_code(meas);
            plot_baseline_observer_convergence( ...
                r,mission,meas,out,cstem,saveFigures,style);
            manifest = add_manifest(manifest,"baseline",cstem, ...
                "GA convergence as observer count changes.");
        end
    end
    plot_baseline_duration_trends(r,out,"baseline_gateway_duration_trends", ...
        saveFigures,style);
    manifest = add_manifest(manifest,"baseline","baseline_gateway_duration_trends", ...
        "AO/AR sensitivity to one, three, and five Gateway periods.");
    for meas = ["ANGLES_ONLY","ANGLES_RANGE"]
        stem = "baseline_convergence_duration_"+measurement_code(meas);
        plot_baseline_duration_convergence(r,meas,out,stem,saveFigures,style);
        manifest = add_manifest(manifest,"baseline",stem, ...
            "GA convergence as Gateway tracking duration changes.");
    end
    plot_reviewer2_geometry_grid(r.bestGeometryRuns,out, ...
        "baseline_geometry",saveFigures);
    for mission = missions
        stem = "baseline_geometry_"+mission_code(mission)+"_grid";
        manifest = add_manifest(manifest,"baseline",stem, ...
            "Best observed geometry for 3, 5, 7, and 10 observers.");
    end
end

if isfield(reports,'objective_screening')
    r = reports.objective_screening;
    out = prepare_output(r.analysisDirectory,saveFigures);
    missions = ["LUNAR_GATEWAY","LOW_THRUST_TRANSFER","GATEWAY_IMPULSE"];
    for mission = missions
        stem = "ga_screening_summary_"+mission_code(mission);
        plot_screening_summary(r,mission,out,stem,saveFigures,style);
        manifest = add_manifest(manifest,"objective_screening",stem, ...
            "Matched combined-objective screening ON/OFF comparison.");
        stem = "ga_screening_convergence_"+mission_code(mission);
        plot_screening_convergence(r,mission,out,stem,saveFigures,style);
        manifest = add_manifest(manifest,"objective_screening",stem, ...
            "Matched combined-objective screening ON/OFF convergence.");
        stem = "ga_objective_components_"+mission_code(mission);
        plot_objective_component_summary(r,mission,out,stem,saveFigures,style);
        manifest = add_manifest(manifest,"objective_screening",stem, ...
            "Physical metric comparison for J111, J100, J010, and J001.");
        stem = "ga_objective_families_"+mission_code(mission);
        plot_objective_family_summary(r,mission,out,stem,saveFigures,style);
        manifest = add_manifest(manifest,"objective_screening",stem, ...
            "Selected orbit-family distribution by objective configuration.");
    end
    geometrySelection = select_objective_geometry_runs(r);
    writetable(geometrySelection,fullfile(char(r.analysisDirectory), ...
        'ga_objective_screening_geometry_selected_runs.csv'));
    plot_reviewer2_geometry_grid(geometrySelection,out, ...
        "ga_objective_geometry",saveFigures);
    for mission = missions
        stem = "ga_objective_geometry_"+mission_code(mission)+"_grid";
        manifest = add_manifest(manifest,"objective_screening",stem, ...
            "Best observed within-configuration geometry for screening/objective cases.");
    end
    trendTable = build_objective_screening_trends(r);
    writetable(trendTable,fullfile(char(r.analysisDirectory), ...
        'ga_objective_screening_trends.csv'));
end

write_manifests(reports,manifest);
end


function out = prepare_output(analysisDir,saveFigures)
out = string(fullfile(char(analysisDir),'paper_final'));
if saveFigures && ~isfolder(out), mkdir(out); end
end


function manifest = add_manifest(manifest,study,stem,purpose)
manifest = [manifest;table(string(study),string(stem),string(purpose), ...
    'VariableNames',manifest.Properties.VariableNames)]; %#ok<AGROW>
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


function plot_runtime_summary(r,out,saveFigures,style)
R = r.runtimeResults;
order = style.optimizerOrder(ismember(style.optimizerOrder,R.Optimizer));
R = sort_to_order(R,'Optimizer',order);
fig = paper_figure(style.figureWidth,style.figureHeight,style);
t = tiledlayout(fig,1,2,'Padding','loose','TileSpacing','compact');

ax1 = nexttile(t); hold(ax1,'on'); box(ax1,'on'); grid(ax1,'on');
colors = colors_for_optimizers(R.Optimizer,style);
b = bar(ax1,1:height(R),R.BestJMean,0.72,'FaceColor','flat'); b.CData = colors;
errorbar(ax1,1:height(R),R.BestJMean,R.BestJStd,'k.','LineWidth',1.0, ...
    'CapSize',style.capSize,'HandleVisibility','off');
format_category_axis(ax1,optimizer_labels(R.Optimizer), ...
    'Final best objective',style);
title(ax1,'(a) Equal-FE solution quality','FontSize',style.fontSize, ...
    'FontName',style.fontName,'FontWeight','bold');

ax2 = nexttile(t); hold(ax2,'on'); box(ax2,'on'); grid(ax2,'on');
b = bar(ax2,1:height(R),R.BudgetRuntimeMean_s,0.72,'FaceColor','flat'); b.CData = colors;
errorbar(ax2,1:height(R),R.BudgetRuntimeMean_s,R.BudgetRuntimeStd_s, ...
    'k.','LineWidth',1.0,'CapSize',style.capSize,'HandleVisibility','off');
format_category_axis(ax2,optimizer_labels(R.Optimizer), ...
    'Runtime to 1200 FE (s)',style);
title(ax2,'(b) Equal-FE computational cost','FontSize',style.fontSize, ...
    'FontName',style.fontName,'FontWeight','bold');
idxBO = find(R.Optimizer == "BAYESIAN",1);
if ~isempty(idxBO)
    ratio = R.BudgetRuntimeMean_s(idxBO)/min(R.BudgetRuntimeMean_s(R.Optimizer ~= "BAYESIAN"));
    text(ax2,idxBO,R.BudgetRuntimeMean_s(idxBO)+R.BudgetRuntimeStd_s(idxBO), ...
        sprintf('  %.1f\\times fastest',ratio),'FontName',style.fontName, ...
        'FontSize',style.fontSize,'FontWeight','bold','VerticalAlignment','bottom');
end
export_figure(fig,out,"runtime_1200_summary",saveFigures,style);
end


function plot_runtime_convergence(r,out,saveFigures,style)
files = dir(fullfile(char(r.analysisDirectory),'convergence_*.mat'));
assert(numel(files) == 1,'Expected one focused-runtime convergence file.');
S = load(fullfile(files(1).folder,files(1).name),'curves');
optimizers = style.optimizerOrder(ismember(style.optimizerOrder,string({S.curves.optimizer})));
fig = paper_figure(style.figureWidth,style.figureHeight,style); ax = axes(fig);
hold(ax,'on'); box(ax,'on'); grid(ax,'on'); handles = gobjects(numel(optimizers),1);
for k = 1:numel(optimizers)
    idx = find(upper(string({S.curves.optimizer})) == optimizers(k),1);
    curve = S.curves(idx); valid = curve.fe >= 60 & isfinite(curve.mean);
    x = double(curve.fe(valid)); y = double(curve.mean(valid)); d = double(curve.std(valid));
    c = optimizer_color(optimizers(k),style);
    uncertainty_band(ax,x,y,d,c,style);
    handles(k) = stairs(ax,x,y,'Color',c,'LineWidth',style.lineWidth, ...
        'DisplayName',optimizer_label(optimizers(k)));
end
xlim(ax,[60 r.budget]);
xlabel(ax,'Function evaluations','FontWeight','bold');
ylabel(ax,'Mean best-so-far objective','FontWeight','bold');
style_axes(ax,style);
lgd = legend(ax,handles,'Location','northoutside','Orientation','horizontal', ...
    'NumColumns',numel(handles)); style_legend(lgd,style);
export_figure(fig,out,"runtime_1200_convergence",saveFigures,style);
end


function plot_comparison_summary(r,out,saveFigures,style)
R = r.results;
missions = string(r.missions); optimizers = string(r.optimizers);
specs = { ...
    'BestJMean','BestJStd','Final best objective'; ...
    'RMSEPosMean_km','RMSEPosStd_km','Position RMSE (km)'; ...
    'EffectiveSigmaPosMean_km','EffectiveSigmaPosStd_km','Effective position sigma (km)'; ...
    'RuntimeMean_s','RuntimeStd_s','Runtime to 6000 FE (s)'};
fig = paper_figure(style.figureWidth,style.panelFigureHeight,style);
t = tiledlayout(fig,2,2,'Padding','loose','TileSpacing','compact');
legendHandles = gobjects(numel(optimizers),1);
for q = 1:4
    ax = nexttile(t); hold(ax,'on'); box(ax,'on'); grid(ax,'on');
    [values,errors] = grouped_values(R,missions,optimizers,specs{q,1},specs{q,2});
    b = bar(ax,1:numel(missions),values,'grouped');
    for k = 1:numel(optimizers)
        b(k).FaceColor = optimizer_color(optimizers(k),style);
        errorbar(ax,b(k).XEndPoints,values(:,k),errors(:,k),'k.', ...
            'LineWidth',0.9,'CapSize',style.capSize,'HandleVisibility','off');
        if q == 1, legendHandles(k) = b(k); end
    end
    ax.XTick = 1:numel(missions); ax.XTickLabel = cellstr(mission_labels(missions));
    ylabel(ax,specs{q,3},'FontWeight','bold'); style_axes(ax,style);
end
lgd = legend(legendHandles,cellstr(optimizer_labels(optimizers)), ...
    'Orientation','horizontal','NumColumns',numel(optimizers),'Box','off');
style_legend(lgd,style); lgd.Layout.Tile = 'north';
export_figure(fig,out,"comparison_6000_summary",saveFigures,style);
end


function plot_comparison_convergence(r,mission,out,stem,saveFigures,style)
R = r.results(r.results.Mission == mission,:);
key = string(R.ComparisonKey(1));
S = load(fullfile(char(r.analysisDirectory),"convergence_"+key+".mat"),'curves');
optimizers = string(r.optimizers);
fig = paper_figure(style.figureWidth,style.figureHeight,style); ax = axes(fig);
hold(ax,'on'); box(ax,'on'); grid(ax,'on'); handles = gobjects(numel(optimizers),1);
for k = 1:numel(optimizers)
    idx = find(upper(string({S.curves.optimizer})) == optimizers(k),1);
    assert(~isempty(idx),'Missing convergence curve for %s.',optimizers(k));
    curve = S.curves(idx); valid = curve.fe >= 60 & isfinite(curve.mean);
    x = double(curve.fe(valid)); y = double(curve.mean(valid)); d = double(curve.std(valid));
    c = optimizer_color(optimizers(k),style); uncertainty_band(ax,x,y,d,c,style);
    handles(k) = stairs(ax,x,y,'Color',c,'LineWidth',style.lineWidth, ...
        'DisplayName',optimizer_label(optimizers(k)));
end
xlim(ax,[60 r.budget]); xlabel(ax,'Function evaluations','FontWeight','bold');
ylabel(ax,'Mean best-so-far objective','FontWeight','bold'); style_axes(ax,style);
lgd = legend(ax,handles,'Location','northoutside','Orientation','horizontal', ...
    'NumColumns',numel(handles)); style_legend(lgd,style);
export_figure(fig,out,stem,saveFigures,style);
end


function plot_baseline_observer_trends(r,mission,out,stem,saveFigures,style)
R = r.results; measurements = ["ANGLES_ONLY","ANGLES_RANGE"]; counts = [3 5 7 10];
specs = { ...
    'BestJMean','BestJStd','Final best objective'; ...
    'RMSEPosMean_km','RMSEPosStd_km','Position RMSE (km)'; ...
    'EffectiveSigmaPosMean_km','EffectiveSigmaPosStd_km','Effective position sigma (km)'; ...
    'RuntimeMean_s','RuntimeStd_s','Runtime to 6000 FE (s)'};
fig = paper_figure(style.figureWidth,style.panelFigureHeight,style);
t = tiledlayout(fig,2,2,'Padding','loose','TileSpacing','compact');
legendHandles = gobjects(2,1);
for q = 1:4
    ax = nexttile(t); hold(ax,'on'); box(ax,'on'); grid(ax,'on');
    for m = 1:2
        values = nan(size(counts)); errors = values;
        for k = 1:numel(counts)
            row = R(R.Mission == mission & R.Measurement == measurements(m) & ...
                R.NumObservers == counts(k) & R.NPeriods == 1,:);
            assert(height(row) == 1,'Missing baseline observer-count point.');
            values(k) = row.(specs{q,1}); errors(k) = row.(specs{q,2});
        end
        c = style.measurementColors(m,:);
        h = errorbar(ax,counts,values,errors,'-o','Color',c, ...
            'LineWidth',style.lineWidth,'MarkerSize',style.markerSize, ...
            'MarkerFaceColor',c,'CapSize',style.capSize, ...
            'DisplayName',measurement_label(measurements(m)));
        if q == 1, legendHandles(m) = h; end
    end
    ax.XTick = counts; xlabel(ax,'Number of observers','FontWeight','bold');
    ylabel(ax,specs{q,3},'FontWeight','bold'); style_axes(ax,style);
end
lgd = legend(legendHandles,{'AO','AR'},'Orientation','horizontal','Box','off');
style_legend(lgd,style); lgd.Layout.Tile = 'north';
export_figure(fig,out,stem,saveFigures,style);
end


function plot_baseline_duration_trends(r,out,stem,saveFigures,style)
R = r.results; measurements = ["ANGLES_ONLY","ANGLES_RANGE"]; periods = [1 3 5];
specs = { ...
    'BestJMean','BestJStd','Final best objective'; ...
    'RMSEPosMean_km','RMSEPosStd_km','Position RMSE (km)'; ...
    'EffectiveSigmaPosMean_km','EffectiveSigmaPosStd_km','Effective position sigma (km)'; ...
    'RuntimeMean_s','RuntimeStd_s','Runtime to 6000 FE (s)'};
fig = paper_figure(style.figureWidth,style.panelFigureHeight,style);
t = tiledlayout(fig,2,2,'Padding','loose','TileSpacing','compact');
legendHandles = gobjects(2,1);
for q = 1:4
    ax = nexttile(t); hold(ax,'on'); box(ax,'on'); grid(ax,'on');
    for m = 1:2
        values = nan(size(periods)); errors = values;
        for k = 1:numel(periods)
            row = R(R.Mission == "LUNAR_GATEWAY" & R.Measurement == measurements(m) & ...
                R.NumObservers == 3 & R.NPeriods == periods(k),:);
            assert(height(row) == 1,'Missing Gateway-duration baseline point.');
            values(k) = row.(specs{q,1}); errors(k) = row.(specs{q,2});
        end
        c = style.measurementColors(m,:);
        h = errorbar(ax,periods,values,errors,'-o','Color',c, ...
            'LineWidth',style.lineWidth,'MarkerSize',style.markerSize, ...
            'MarkerFaceColor',c,'CapSize',style.capSize, ...
            'DisplayName',measurement_label(measurements(m)));
        if q == 1, legendHandles(m) = h; end
    end
    ax.XTick = periods; xlabel(ax,'Gateway tracking periods','FontWeight','bold');
    ylabel(ax,specs{q,3},'FontWeight','bold'); style_axes(ax,style);
end
lgd = legend(legendHandles,{'AO','AR'},'Orientation','horizontal','Box','off');
style_legend(lgd,style); lgd.Layout.Tile = 'north';
export_figure(fig,out,stem,saveFigures,style);
end


function plot_baseline_observer_convergence(r,mission,measurement,out,stem,saveFigures,style)
counts = [3 5 7 10]; fig = paper_figure(style.figureWidth,style.figureHeight,style);
ax = axes(fig); hold(ax,'on'); box(ax,'on'); grid(ax,'on'); colors = lines(4);
handles = gobjects(4,1);
for k = 1:4
    row = r.results(r.results.Mission == mission & r.results.Measurement == measurement & ...
        r.results.NumObservers == counts(k) & r.results.NPeriods == 1,:);
    assert(height(row) == 1,'Missing baseline convergence configuration.');
    curve = load_ga_curve(r.analysisDirectory,row.ComparisonKey);
    valid = curve.fe >= 60 & isfinite(curve.mean); x = double(curve.fe(valid));
    y = double(curve.mean(valid)); d = double(curve.std(valid));
    uncertainty_band(ax,x,y,d,colors(k,:),style);
    handles(k) = stairs(ax,x,y,'Color',colors(k,:),'LineWidth',style.lineWidth, ...
        'DisplayName',sprintf('%d observers',counts(k)));
end
xlim(ax,[60 r.budget]); xlabel(ax,'Function evaluations','FontWeight','bold');
ylabel(ax,'Mean best-so-far objective','FontWeight','bold'); style_axes(ax,style);
lgd = legend(ax,handles,'Location','northoutside','Orientation','horizontal', ...
    'NumColumns',2); style_legend(lgd,style);
export_figure(fig,out,stem,saveFigures,style);
end


function plot_baseline_duration_convergence(r,measurement,out,stem,saveFigures,style)
periods = [1 3 5]; fig = paper_figure(style.figureWidth,style.figureHeight,style);
ax = axes(fig); hold(ax,'on'); box(ax,'on'); grid(ax,'on'); colors = lines(3);
handles = gobjects(3,1);
for k = 1:3
    row = r.results(r.results.Mission == "LUNAR_GATEWAY" & ...
        r.results.Measurement == measurement & r.results.NumObservers == 3 & ...
        r.results.NPeriods == periods(k),:);
    assert(height(row) == 1,'Missing baseline duration convergence configuration.');
    curve = load_ga_curve(r.analysisDirectory,row.ComparisonKey);
    valid = curve.fe >= 60 & isfinite(curve.mean); x = double(curve.fe(valid));
    y = double(curve.mean(valid)); d = double(curve.std(valid));
    uncertainty_band(ax,x,y,d,colors(k,:),style);
    handles(k) = stairs(ax,x,y,'Color',colors(k,:),'LineWidth',style.lineWidth, ...
        'DisplayName',sprintf('%d period%s',periods(k),plural_s(periods(k))));
end
xlim(ax,[60 r.budget]); xlabel(ax,'Function evaluations','FontWeight','bold');
ylabel(ax,'Mean best-so-far objective','FontWeight','bold'); style_axes(ax,style);
lgd = legend(ax,handles,'Location','northoutside','Orientation','horizontal');
style_legend(lgd,style); export_figure(fig,out,stem,saveFigures,style);
end


function plot_screening_summary(r,mission,out,stem,saveFigures,style)
configs = ["combined_on","combined_off"];
specs = { ...
    'BestJMean','BestJStd','Final best objective'; ...
    'RMSEPosMean_km','RMSEPosStd_km','Position RMSE (km)'; ...
    'EffectiveSigmaPosMean_km','EffectiveSigmaPosStd_km','Effective position sigma (km)'; ...
    'AvailableObserversMean','AvailableObserversStd','Mean available observers'};
plot_configuration_panel(r.results,mission,configs,specs,out,stem,saveFigures,style);
end


function plot_objective_component_summary(r,mission,out,stem,saveFigures,style)
% Total objective is intentionally omitted because J111/J100/J010/J001 are
% different mathematical objectives and therefore are not directly comparable.
configs = ["combined_on","j1_only","j2_only","j3_only"];
specs = { ...
    'RMSEPosMean_km','RMSEPosStd_km','Position RMSE (km)'; ...
    'EffectiveSigmaPosMean_km','EffectiveSigmaPosStd_km','Effective position sigma (km)'; ...
    'MeanStabilityMean','MeanStabilityStd','Mean stability index'; ...
    'CoverageMean','CoverageStd','Coverage fraction'};
plot_configuration_panel(r.results,mission,configs,specs,out,stem,saveFigures,style);
end


function plot_configuration_panel(R,mission,configs,specs,out,stem,saveFigures,style)
fig = paper_figure(style.figureWidth,style.panelFigureHeight,style);
t = tiledlayout(fig,2,2,'Padding','loose','TileSpacing','compact');
colors = colors_for_configurations(configs,style);
for q = 1:4
    ax = nexttile(t); hold(ax,'on'); box(ax,'on'); grid(ax,'on');
    values = nan(numel(configs),1); errors = values;
    for k = 1:numel(configs)
        row = objective_result(R,mission,configs(k));
        values(k) = row.(specs{q,1}); errors(k) = row.(specs{q,2});
    end
    b = bar(ax,1:numel(configs),values,0.72,'FaceColor','flat'); b.CData = colors;
    errorbar(ax,1:numel(configs),values,errors,'k.','LineWidth',0.9, ...
        'CapSize',style.capSize,'HandleVisibility','off');
    ax.XTick = 1:numel(configs); ax.XTickLabel = cellstr(configuration_labels(configs));
    ax.XTickLabelRotation = 18; ylabel(ax,specs{q,3},'FontWeight','bold');
    style_axes(ax,style);
end
export_figure(fig,out,stem,saveFigures,style);
end


function plot_screening_convergence(r,mission,out,stem,saveFigures,style)
configs = ["combined_on","combined_off"];
colors = colors_for_configurations(configs,style);
fig = paper_figure(style.figureWidth,style.figureHeight,style); ax = axes(fig);
hold(ax,'on'); box(ax,'on'); grid(ax,'on'); handles = gobjects(2,1);
for k = 1:2
    row = objective_result(r.results,mission,configs(k));
    curve = load_ga_curve(r.analysisDirectory,row.ComparisonKey);
    valid = curve.fe >= 60 & isfinite(curve.mean); x = double(curve.fe(valid));
    y = double(curve.mean(valid)); d = double(curve.std(valid));
    uncertainty_band(ax,x,y,d,colors(k,:),style);
    handles(k) = stairs(ax,x,y,'Color',colors(k,:),'LineWidth',style.lineWidth, ...
        'DisplayName',configuration_label(configs(k)));
end
xlim(ax,[60 r.budget]); xlabel(ax,'Function evaluations','FontWeight','bold');
ylabel(ax,'Mean best-so-far objective','FontWeight','bold'); style_axes(ax,style);
lgd = legend(ax,handles,'Location','northoutside','Orientation','horizontal');
style_legend(lgd,style); export_figure(fig,out,stem,saveFigures,style);
end


function plot_objective_family_summary(r,mission,out,stem,saveFigures,style)
F = r.familySelection; configs = ["combined_on","j1_only","j2_only","j3_only"];
families = ["DRO","NRHO/rectilinear","Halo","Other"];
values = zeros(numel(configs),numel(families));
for k = 1:numel(configs)
    for f = 1:numel(families)
        row = F(F.Mission == mission & F.Configuration == configs(k) & ...
            F.FamilyGroup == families(f),:);
        assert(height(row) == 1,'Missing family-selection row.');
        values(k,f) = 100*row.Fraction;
    end
end
fig = paper_figure(style.figureWidth,style.figureHeight,style); ax = axes(fig);
b = bar(ax,1:numel(configs),values,'stacked'); box(ax,'on'); grid(ax,'on');
ax.XTick = 1:numel(configs); ax.XTickLabel = cellstr(configuration_labels(configs));
ax.XTickLabelRotation = 18; ylim(ax,[0 100]);
xlabel(ax,'Objective configuration','FontWeight','bold');
ylabel(ax,'Selected observers (%)','FontWeight','bold'); style_axes(ax,style);
lgd = legend(ax,b,cellstr(families),'Location','northoutside','Orientation','horizontal', ...
    'NumColumns',numel(families)); style_legend(lgd,style);
export_figure(fig,out,stem,saveFigures,style);
end


function selection = select_objective_geometry_runs(r)
R = r.results; M = r.runMetrics;
missions = ["LUNAR_GATEWAY","LOW_THRUST_TRANSFER","GATEWAY_IMPULSE"];
configs = ["combined_on","combined_off","j1_only","j2_only","j3_only"];
n = numel(missions)*numel(configs); missionColumn = strings(n,1);
panelKey = strings(n,1); panelLabel = strings(n,1); runFile = strings(n,1);
bestObjective = nan(n,1); seed = nan(n,1); rowOut = 0;
for mission = missions
    for config = configs
        rowOut = rowOut+1; rr = objective_result(R,mission,config);
        rows = M(M.comparison_key == string(rr.ComparisonKey),:);
        assert(~isempty(rows),'No run metrics for objective geometry selection.');
        [bestObjective(rowOut),idx] = min(rows.bestJ);
        missionColumn(rowOut) = mission; panelKey(rowOut) = config;
        panelLabel(rowOut) = configuration_label(config);
        runFile(rowOut) = rows.run_file(idx); seed(rowOut) = rows.seed(idx);
    end
end
selection = table(missionColumn,panelKey,panelLabel,runFile,bestObjective,seed, ...
    'VariableNames',{'Mission','PanelKey','PanelLabel','RunFile','BestObjective','Seed'});
end


function trends = build_objective_screening_trends(r)
missions = ["LUNAR_GATEWAY","LOW_THRUST_TRANSFER","GATEWAY_IMPULSE"];
configs = ["combined_off","j1_only","j2_only","j3_only"];
missionColumn = strings(numel(missions)*numel(configs),1);
configuration = strings(size(missionColumn)); rmsePct = nan(size(missionColumn));
sigmaPct = rmsePct; stabilityPct = rmsePct; coveragePct = rmsePct;
objectivePct = rmsePct; row = 0;
for mission = missions
    ref = objective_result(r.results,mission,"combined_on");
    for config = configs
        row = row+1; x = objective_result(r.results,mission,config);
        missionColumn(row) = mission; configuration(row) = config;
        rmsePct(row) = percent_change(ref.RMSEPosMean_km,x.RMSEPosMean_km);
        sigmaPct(row) = percent_change(ref.EffectiveSigmaPosMean_km,x.EffectiveSigmaPosMean_km);
        stabilityPct(row) = percent_change(ref.MeanStabilityMean,x.MeanStabilityMean);
        coveragePct(row) = percent_change(ref.CoverageMean,x.CoverageMean);
        if config == "combined_off"
            objectivePct(row) = percent_change(ref.BestJMean,x.BestJMean);
        end
    end
end
trends = table(missionColumn,configuration,objectivePct,rmsePct,sigmaPct, ...
    stabilityPct,coveragePct,'VariableNames',{'Mission','Configuration', ...
    'ObjectiveChangePct','RMSEChangePct','EffectiveSigmaChangePct', ...
    'StabilityChangePct','CoverageChangePct'});
end


function curve = load_ga_curve(analysisDir,key)
S = load(fullfile(char(analysisDir),"convergence_"+string(key)+".mat"),'curves');
idx = find(upper(string({S.curves.optimizer})) == "GA",1);
assert(~isempty(idx),'Missing GA convergence curve.'); curve = S.curves(idx);
end


function [values,errors] = grouped_values(R,missions,optimizers,valueField,errorField)
values = nan(numel(missions),numel(optimizers)); errors = values;
for m = 1:numel(missions)
    for k = 1:numel(optimizers)
        row = R(R.Mission == missions(m) & R.Optimizer == optimizers(k),:);
        assert(height(row) == 1,'Missing grouped comparison point.');
        values(m,k) = row.(valueField); errors(m,k) = row.(errorField);
    end
end
end


function row = objective_result(R,mission,config)
row = R(R.Mission == mission & string(R.Configuration) == config,:);
assert(height(row) == 1,'Missing objective/screening result for %s/%s.',mission,config);
end


function R = sort_to_order(R,field,order)
idx = nan(numel(order),1);
for k = 1:numel(order), idx(k) = find(string(R.(field)) == order(k),1); end
R = R(idx,:);
end


function colors = colors_for_optimizers(optimizers,style)
colors = zeros(numel(optimizers),3);
for k = 1:numel(optimizers), colors(k,:) = optimizer_color(optimizers(k),style); end
end


function c = optimizer_color(optimizer,style)
idx = find(style.optimizerOrder == upper(string(optimizer)),1);
assert(~isempty(idx),'Unknown optimizer color: %s',optimizer); c = style.optimizerColors(idx,:);
end


function colors = colors_for_configurations(configs,style)
colors = zeros(numel(configs),3);
for k = 1:numel(configs)
    idx = find(style.configurationOrder == string(configs(k)),1);
    assert(~isempty(idx)); colors(k,:) = style.configurationColors(idx,:);
end
end


function uncertainty_band(ax,x,y,d,color,style)
idx = unique(round(linspace(1,numel(x),min(180,numel(x)))));
x = x(idx); y = y(idx); d = d(idx);
bandColor = (1-style.alphaBand)*[1 1 1] + style.alphaBand*color;
fill(ax,[x;flipud(x)],[max(0,y-d);flipud(y+d)],bandColor, ...
    'EdgeColor','none','HandleVisibility','off');
end


function fig = paper_figure(widthIn,heightIn,style)
fig = figure('Color','w','Units','inches','Position',[1 1 widthIn heightIn], ...
    'PaperUnits','inches','PaperSize',[widthIn heightIn], ...
    'PaperPosition',[0 0 widthIn heightIn],'PaperPositionMode','manual', ...
    'Renderer','painters','InvertHardcopy','off');
movegui(fig,'center');
set(fig,'DefaultAxesFontName',style.fontName,'DefaultAxesFontSize',style.fontSize);
end


function style_axes(ax,style)
set(ax,'FontName',style.fontName,'FontSize',style.fontSize,'FontWeight','bold', ...
    'LineWidth',style.axisLineWidth,'TickDir','out','Layer','top');
ax.XLabel.FontSize = style.labelFontSize; ax.YLabel.FontSize = style.labelFontSize;
end


function style_legend(lgd,style)
lgd.FontName = style.fontName; lgd.FontSize = style.fontSize;
lgd.FontWeight = 'bold';
end


function format_category_axis(ax,labels,yLabel,style)
ax.XTick = 1:numel(labels); ax.XTickLabel = cellstr(labels);
ax.XTickLabelRotation = 18; ylabel(ax,yLabel,'FontWeight','bold'); style_axes(ax,style);
end


function export_figure(fig,out,stem,saveFigures,style)
drawnow; if ~saveFigures, return; end
base = fullfile(char(out),char(stem));
print(fig,[base '.eps'],'-depsc','-painters');
exportgraphics(fig,[base '.png'],'Resolution',style.exportDpi);
close(fig);
end


function labels = optimizer_labels(values)
values = string(values(:)); labels = strings(size(values));
for k = 1:numel(values), labels(k) = optimizer_label(values(k)); end
end

function label = optimizer_label(value)
switch upper(string(value))
    case "BAYESIAN", label = "BO";
    case "ABC", label = "ABC";
    otherwise, label = upper(string(value));
end
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

function label = measurement_label(value)
if string(value) == "ANGLES_ONLY", label = "AO"; else, label = "AR"; end
end

function labels = configuration_labels(values)
values = string(values(:)); labels = strings(size(values));
for k = 1:numel(values), labels(k) = configuration_label(values(k)); end
end

function label = configuration_label(value)
switch string(value)
    case "combined_on", label = "J_{111}, screening ON";
    case "combined_off", label = "J_{111}, screening OFF";
    case "j1_only", label = "J_{100}";
    case "j2_only", label = "J_{010}";
    case "j3_only", label = "J_{001}";
    otherwise, label = string(value);
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

function code = measurement_code(measurement)
if string(measurement) == "ANGLES_ONLY", code = "ao"; else, code = "ar"; end
end

function s = plural_s(value)
if value == 1, s = ''; else, s = 's'; end
end

function value = percent_change(reference,newValue)
value = 100*(double(newValue)-double(reference))/max(abs(double(reference)),eps);
end
