function manifest = make_reviewer2_paper_figures(reports,saveFigures)
%MAKE_REVIEWER2_PAPER_FIGURES Create the final Reviewer-2 manuscript figures.
%
% Design rules:
%   * statistical comparisons use 20-run mean +/- sample standard deviation;
%   * convergence uses small multiples, never overlapping uncertainty bands;
%   * geometry uses a representative realization nearest the group-mean
%     objective, never the lowest-cost seed as the statistical comparison;
%   * geometry grids are fixed at 6.5 x 6.5 inches and use common limits;
%   * Times New Roman, 12-point minimum ticks/legends, 14-point axis labels.

if nargin < 2 || isempty(saveFigures), saveFigures = true; end
saveFigures = logical(saveFigures);
assert(isstruct(reports),'reports must come from run_reviewer2_results.');
style = reviewer2_paper_style();
manifest = table(strings(0,1),strings(0,1),strings(0,1), ...
    'VariableNames',{'Study','FigureStem','Purpose'});

if isfield(reports,'runtime')
    r = reports.runtime; out = prepare_output(r.analysisDirectory,saveFigures);
    plot_runtime_summary(r,out,saveFigures,style);
    manifest = add_manifest(manifest,"runtime","runtime_1200_summary", ...
        "Mean +/- sample std final objective and equal-FE runtime; quantifies BO cost/benefit.");
    plot_runtime_convergence_small_multiples(r,out,saveFigures,style);
    manifest = add_manifest(manifest,"runtime","runtime_1200_convergence", ...
        "One optimizer per axes; mean convergence with final-FE sample standard deviation.");
end

if isfield(reports,'comparison')
    r = reports.comparison; out = prepare_output(r.analysisDirectory,saveFigures);
    plot_comparison_summary(r,out,saveFigures,style);
    manifest = add_manifest(manifest,"comparison","comparison_6000_summary", ...
        "Mean +/- sample std objective, RMSE, uncertainty, and runtime by target case.");
    plot_optimizer_ranking(r,out,saveFigures,style);
    manifest = add_manifest(manifest,"comparison","comparison_6000_optimizer_ranking", ...
        "Overall optimizer ranking computed from mission-wise mean objective, not individual seeds.");
    for mission = string(r.missions)
        stem = "comparison_6000_convergence_"+mission_code(mission);
        plot_comparison_convergence_small_multiples(r,mission,out,stem,saveFigures,style);
        manifest = add_manifest(manifest,"comparison",stem, ...
            "One optimizer per axes with shared limits; mean convergence and final-FE sample std.");
    end
    selection = select_reviewer2_representative_runs(r,"comparison");
    writetable(selection,fullfile(char(r.analysisDirectory), ...
        'comparison_6000_geometry_representative_runs.csv'));
    plot_reviewer2_geometry_grid(selection,out,"comparison_geometry",saveFigures);
    for mission = string(r.missions)
        stem = "comparison_geometry_"+mission_code(mission)+"_grid";
        manifest = add_manifest(manifest,"comparison",stem, ...
            "Representative optimizer geometry: realization nearest each 20-run mean objective.");
    end
end

if isfield(reports,'baseline')
    r = reports.baseline; out = prepare_output(r.analysisDirectory,saveFigures);
    missions = ["LUNAR_GATEWAY","LOW_THRUST_TRANSFER","GATEWAY_IMPULSE"];
    for mission = missions
        stem = "baseline_observer_trends_"+mission_code(mission);
        plot_baseline_observer_trends(r,mission,out,stem,saveFigures,style);
        manifest = add_manifest(manifest,"baseline",stem, ...
            "AO/AR observer-count trends using mean +/- sample standard deviation.");
        for meas = ["ANGLES_ONLY","ANGLES_RANGE"]
            cstem = "baseline_convergence_observers_"+mission_code(mission)+ ...
                "_"+measurement_code(meas);
            plot_baseline_observer_convergence_small_multiples( ...
                r,mission,meas,out,cstem,saveFigures,style);
            manifest = add_manifest(manifest,"baseline",cstem, ...
                "One observer-count case per axes with shared convergence limits.");
        end
    end
    for meas = ["ANGLES_ONLY","ANGLES_RANGE"]
        stem = "baseline_gateway_duration_objective_"+measurement_code(meas);
        plot_baseline_duration_metric(r,meas,"BestJMean","BestJStd", ...
            'Final best objective',out,stem,saveFigures,style);
        manifest = add_manifest(manifest,"baseline",stem, ...
            "Tracking-duration effect on mean objective for 3/5/7/10 observers.");
        stem = "baseline_gateway_duration_rmse_"+measurement_code(meas);
        plot_baseline_duration_metric(r,meas,"RMSEPosMean_km","RMSEPosStd_km", ...
            'Position RMSE (km)',out,stem,saveFigures,style);
        manifest = add_manifest(manifest,"baseline",stem, ...
            "Tracking-duration effect on mean position RMSE for 3/5/7/10 observers.");
        stem = "baseline_convergence_duration_"+measurement_code(meas);
        plot_baseline_duration_convergence_small_multiples(r,meas,out,stem,saveFigures,style);
        manifest = add_manifest(manifest,"baseline",stem, ...
            "One Gateway duration per axes with shared convergence limits.");
    end
    selection = select_reviewer2_representative_runs(r,"baseline");
    writetable(selection,fullfile(char(r.analysisDirectory), ...
        'baseline_6000_geometry_representative_runs.csv'));
    plot_reviewer2_geometry_grid(selection,out,"baseline_geometry",saveFigures);
    for mission = missions
        stem = "baseline_geometry_"+mission_code(mission)+"_grid";
        manifest = add_manifest(manifest,"baseline",stem, ...
            "Representative AO geometry nearest the mean objective for 3/5/7/10 observers.");
    end
end

if isfield(reports,'objective_screening')
    r = reports.objective_screening; out = prepare_output(r.analysisDirectory,saveFigures);
    missions = ["LUNAR_GATEWAY","LOW_THRUST_TRANSFER","GATEWAY_IMPULSE"];
    for mission = missions
        stem = "ga_screening_summary_"+mission_code(mission);
        plot_screening_summary(r,mission,out,stem,saveFigures,style);
        manifest = add_manifest(manifest,"objective_screening",stem, ...
            "Matched J111 screening ON/OFF mean +/- sample std comparison.");
        stem = "ga_screening_convergence_"+mission_code(mission);
        plot_screening_convergence_small_multiples(r,mission,out,stem,saveFigures,style);
        manifest = add_manifest(manifest,"objective_screening",stem, ...
            "Screening ON/OFF convergence shown on separate axes with shared limits.");
        stem = "ga_objective_components_"+mission_code(mission);
        plot_objective_component_summary(r,mission,out,stem,saveFigures,style);
        manifest = add_manifest(manifest,"objective_screening",stem, ...
            "Physical metrics for J111/J100/J010/J001 using mean +/- sample std.");
        stem = "ga_objective_families_"+mission_code(mission);
        plot_objective_family_summary(r,mission,out,stem,saveFigures,style);
        manifest = add_manifest(manifest,"objective_screening",stem, ...
            "Orbit-family distribution by objective configuration.");
    end

    selection = select_reviewer2_representative_runs(r,"objective_screening");
    writetable(selection,fullfile(char(r.analysisDirectory), ...
        'ga_objective_screening_geometry_representative_runs.csv'));
    screeningSelection = selection(ismember(selection.PanelKey,["combined_on","combined_off"]),:);
    componentSelection = selection(ismember(selection.PanelKey, ...
        ["combined_on","j1_only","j2_only","j3_only"]),:);
    plot_reviewer2_geometry_grid(screeningSelection,out,"ga_screening_geometry",saveFigures);
    plot_reviewer2_geometry_grid(componentSelection,out,"ga_objective_geometry",saveFigures);
    for mission = missions
        manifest = add_manifest(manifest,"objective_screening", ...
            "ga_screening_geometry_"+mission_code(mission)+"_grid", ...
            "Representative screening ON/OFF geometry nearest each group mean objective.");
        manifest = add_manifest(manifest,"objective_screening", ...
            "ga_objective_geometry_"+mission_code(mission)+"_grid", ...
            "Representative objective-component geometry nearest each group mean objective.");
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
colors = colors_for_optimizers(R.Optimizer,style);
fig = paper_figure(style.figureWidth,style.figureHeight,style);
t = tiledlayout(fig,1,2,'Padding','loose','TileSpacing','compact');

ax = nexttile(t); hold(ax,'on'); box(ax,'on'); grid(ax,'on');
b = bar(ax,1:height(R),R.BestJMean,0.72,'FaceColor','flat'); b.CData = colors;
errorbar(ax,1:height(R),R.BestJMean,R.BestJStd,'k.','LineWidth',1.0, ...
    'CapSize',style.capSize,'HandleVisibility','off');
format_category_axis(ax,optimizer_labels(R.Optimizer),'Final best objective',style);

ax = nexttile(t); hold(ax,'on'); box(ax,'on'); grid(ax,'on');
b = bar(ax,1:height(R),R.BudgetRuntimeMean_s,0.72,'FaceColor','flat'); b.CData = colors;
errorbar(ax,1:height(R),R.BudgetRuntimeMean_s,R.BudgetRuntimeStd_s,'k.', ...
    'LineWidth',1.0,'CapSize',style.capSize,'HandleVisibility','off');
format_category_axis(ax,optimizer_labels(R.Optimizer),'Runtime to 1200 FE (s)',style);
idxBO = find(R.Optimizer == "BAYESIAN",1);
if ~isempty(idxBO)
    fastest = min(R.BudgetRuntimeMean_s(R.Optimizer ~= "BAYESIAN"));
    ratio = R.BudgetRuntimeMean_s(idxBO)/fastest;
    text(ax,idxBO,R.BudgetRuntimeMean_s(idxBO)+R.BudgetRuntimeStd_s(idxBO), ...
        sprintf('%.1fx fastest',ratio),'HorizontalAlignment','center', ...
        'VerticalAlignment','bottom','FontName',style.fontName, ...
        'FontSize',style.fontSize,'FontWeight','bold');
end
export_figure(fig,out,"runtime_1200_summary",saveFigures,style);
end


function plot_runtime_convergence_small_multiples(r,out,saveFigures,style)
files = dir(fullfile(char(r.analysisDirectory),'convergence_*.mat'));
assert(numel(files) == 1,'Expected one runtime convergence file.');
S = load(fullfile(files(1).folder,files(1).name),'curves');
optimizers = style.optimizerOrder(ismember(style.optimizerOrder,upper(string({S.curves.optimizer}))));
curves = cell(numel(optimizers),1);
for k = 1:numel(optimizers)
    idx = find(upper(string({S.curves.optimizer})) == optimizers(k),1);
    curves{k} = S.curves(idx);
end
plot_curve_small_multiples(curves,optimizer_labels(optimizers), ...
    colors_for_optimizers(optimizers,style),r.budget,3,2,out, ...
    "runtime_1200_convergence",saveFigures,style);
end


function plot_comparison_summary(r,out,saveFigures,style)
R = r.results; missions = string(r.missions); optimizers = string(r.optimizers);
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
    b = bar(ax,1:numel(missions),values,'grouped'); drawnow;
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


function plot_optimizer_ranking(r,out,saveFigures,style)
T = r.overallRanking;
T = sortrows(T,'OverallRank','ascend');
colors = colors_for_optimizers(T.Optimizer,style);
fig = paper_figure(style.figureWidth,style.figureHeight,style);
t = tiledlayout(fig,1,2,'Padding','loose','TileSpacing','compact');
ax = nexttile(t); hold(ax,'on'); box(ax,'on'); grid(ax,'on');
b = bar(ax,1:height(T),T.MeanObjectiveRank,0.72,'FaceColor','flat'); b.CData = colors;
format_category_axis(ax,optimizer_labels(T.Optimizer),'Mean objective rank',style);
yline(ax,1,'k:','HandleVisibility','off');
ax = nexttile(t); hold(ax,'on'); box(ax,'on'); grid(ax,'on');
b = bar(ax,1:height(T),T.MissionWins,0.72,'FaceColor','flat'); b.CData = colors;
format_category_axis(ax,optimizer_labels(T.Optimizer),'Target-case wins',style);
export_figure(fig,out,"comparison_6000_optimizer_ranking",saveFigures,style);
end


function plot_comparison_convergence_small_multiples(r,mission,out,stem,saveFigures,style)
row = r.results(r.results.Mission == mission,:);
key = string(row.ComparisonKey(1));
S = load(fullfile(char(r.analysisDirectory),"convergence_"+key+".mat"),'curves');
optimizers = string(r.optimizers); curves = cell(numel(optimizers),1);
for k = 1:numel(optimizers)
    idx = find(upper(string({S.curves.optimizer})) == optimizers(k),1);
    assert(~isempty(idx),'Missing convergence curve for %s.',optimizers(k));
    curves{k} = S.curves(idx);
end
plot_curve_small_multiples(curves,optimizer_labels(optimizers), ...
    colors_for_optimizers(optimizers,style),r.budget,2,2,out,stem,saveFigures,style);
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


function plot_baseline_duration_metric(r,measurement,valueField,stdField,yLabel,out,stem,saveFigures,style)
R = r.results; counts = [3 5 7 10]; periods = [1 3 5];
fig = paper_figure(style.figureWidth,style.panelFigureHeight,style);
t = tiledlayout(fig,2,2,'Padding','loose','TileSpacing','compact');
colors = lines(numel(counts));
for k = 1:numel(counts)
    ax = nexttile(t); hold(ax,'on'); box(ax,'on'); grid(ax,'on');
    rows = R(R.Mission == "LUNAR_GATEWAY" & R.Measurement == measurement & ...
        R.NumObservers == counts(k),:);
    values = nan(size(periods)); errors = values;
    for p = 1:numel(periods)
        row = rows(rows.NPeriods == periods(p),:);
        assert(height(row) == 1,'Missing Gateway duration point.');
        values(p) = row.(valueField); errors(p) = row.(stdField);
    end
    errorbar(ax,periods,values,errors,'-o','Color',colors(k,:), ...
        'LineWidth',style.lineWidth,'MarkerSize',style.markerSize, ...
        'MarkerFaceColor',colors(k,:),'CapSize',style.capSize);
    ax.XTick = periods; xlabel(ax,'Gateway periods','FontWeight','bold');
    ylabel(ax,yLabel,'FontWeight','bold'); style_axes(ax,style);
    title(ax,string(counts(k))+" observers",'FontName',style.fontName, ...
        'FontSize',style.fontSize,'FontWeight','bold');
end
export_figure(fig,out,stem,saveFigures,style);
end


function plot_baseline_observer_convergence_small_multiples(r,mission,measurement,out,stem,saveFigures,style)
counts = [3 5 7 10]; curves = cell(4,1); labels = strings(4,1);
colors = lines(4);
for k = 1:4
    row = r.results(r.results.Mission == mission & r.results.Measurement == measurement & ...
        r.results.NumObservers == counts(k) & r.results.NPeriods == 1,:);
    assert(height(row) == 1,'Missing baseline convergence configuration.');
    curves{k} = load_ga_curve(r.analysisDirectory,row.ComparisonKey);
    labels(k) = string(counts(k))+" observers";
end
plot_curve_small_multiples(curves,labels,colors,r.budget,2,2,out,stem,saveFigures,style);
end


function plot_baseline_duration_convergence_small_multiples(r,measurement,out,stem,saveFigures,style)
periods = [1 3 5]; curves = cell(3,1); labels = strings(3,1); colors = lines(3);
for k = 1:3
    row = r.results(r.results.Mission == "LUNAR_GATEWAY" & ...
        r.results.Measurement == measurement & r.results.NumObservers == 3 & ...
        r.results.NPeriods == periods(k),:);
    assert(height(row) == 1,'Missing baseline duration convergence configuration.');
    curves{k} = load_ga_curve(r.analysisDirectory,row.ComparisonKey);
    labels(k) = string(periods(k))+" period"+plural_s(periods(k));
end
plot_curve_small_multiples(curves,labels,colors,r.budget,1,3,out,stem,saveFigures,style);
end


function plot_screening_summary(r,mission,out,stem,saveFigures,style)
configs = ["combined_on","combined_off"];
specs = { ...
    'BestJMean','BestJStd','Final best objective'; ...
    'RMSEPosMean_km','RMSEPosStd_km','Position RMSE (km)'; ...
    'EffectiveSigmaPosMean_km','EffectiveSigmaPosStd_km','Effective position sigma (km)'; ...
    'ScreeningMean','ScreeningStd','Rejected measurement opportunities'};
plot_configuration_panel(r.results,mission,configs,specs,out,stem,saveFigures,style);
end


function plot_screening_convergence_small_multiples(r,mission,out,stem,saveFigures,style)
configs = ["combined_on","combined_off"]; curves = cell(2,1);
for k = 1:2
    row = objective_result(r.results,mission,configs(k));
    curves{k} = load_ga_curve(r.analysisDirectory,row.ComparisonKey);
end
plot_curve_small_multiples(curves,configuration_labels(configs), ...
    colors_for_configurations(configs,style),r.budget,1,2,out,stem,saveFigures,style);
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


function plot_curve_small_multiples(curves,labels,colors,budget,nRows,nCols,out,stem,saveFigures,style)
% One curve per axes avoids unreadable overlap. Each axes shows the across-run
% mean best-so-far trace and a final-FE +/- sample-standard-deviation bar.
assert(numel(curves) == numel(labels));
fig = paper_figure(style.convergenceFigureWidth,style.convergenceFigureHeight,style);
t = tiledlayout(fig,nRows,nCols,'Padding','loose','TileSpacing','compact');
allY = zeros(0,1);
for k = 1:numel(curves)
    c = curves{k}; valid = c.fe >= 60 & isfinite(c.mean);
    y = double(c.mean(valid)); allY = [allY;y]; %#ok<AGROW>
    if any(valid)
        dEnd = double(c.std(find(valid,1,'last')));
        if isfinite(dEnd), allY = [allY;y(end)-dEnd;y(end)+dEnd]; end %#ok<AGROW>
    end
end
allY = allY(isfinite(allY));
lo = min(allY); hi = max(allY); span = max(hi-lo,0.05*max(1,abs(hi)));
yLimits = [lo-0.06*span,hi+0.08*span];
if yLimits(1) >= 0, yLimits(1) = max(0,yLimits(1)); end

for k = 1:numel(curves)
    ax = nexttile(t); hold(ax,'on'); box(ax,'on'); grid(ax,'on');
    c = curves{k}; valid = c.fe >= 60 & isfinite(c.mean);
    x = double(c.fe(valid)); y = double(c.mean(valid));
    stairs(ax,x,y,'Color',colors(k,:),'LineWidth',style.lineWidth);
    if ~isempty(x)
        dEnd = double(c.std(find(valid,1,'last')));
        if isfinite(dEnd)
            errorbar(ax,x(end),y(end),dEnd,'o','Color',colors(k,:), ...
                'MarkerFaceColor',colors(k,:),'MarkerSize',4.5, ...
                'LineWidth',1.0,'CapSize',style.capSize);
        else
            plot(ax,x(end),y(end),'o','Color',colors(k,:), ...
                'MarkerFaceColor',colors(k,:),'MarkerSize',4.5);
        end
    end
    xlim(ax,[60 budget]); ylim(ax,yLimits);
    title(ax,sprintf('(%c) %s',char('a'+k-1),string(labels(k))), ...
        'FontName',style.fontName,'FontSize',style.fontSize, ...
        'FontWeight','bold','Interpreter','none');
    style_axes(ax,style);
end
for k = numel(curves)+1:nRows*nCols
    ax = nexttile(t); axis(ax,'off');
end
xlabel(t,'Function evaluations','FontName',style.fontName, ...
    'FontSize',style.labelFontSize,'FontWeight','bold');
ylabel(t,'Mean best-so-far objective','FontName',style.fontName, ...
    'FontSize',style.labelFontSize,'FontWeight','bold');
export_figure(fig,out,stem,saveFigures,style);
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
lgd.FontName = style.fontName; lgd.FontSize = style.fontSize; lgd.FontWeight = 'bold';
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
values = upper(string(values(:))); labels = values; labels(values == "BAYESIAN") = "BO";
end

function label = optimizer_label(value)
label = optimizer_labels(value); label = label(1);
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
    case "combined_on", label = "Combined, screening ON";
    case "combined_off", label = "Combined, screening OFF";
    case "j1_only", label = "J_1 only";
    case "j2_only", label = "J_2 only";
    case "j3_only", label = "J_3 only";
    otherwise, label = string(value);
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

function value = percent_change(before,after)
value = 100*(double(after)-double(before))/max(abs(double(before)),eps);
end
