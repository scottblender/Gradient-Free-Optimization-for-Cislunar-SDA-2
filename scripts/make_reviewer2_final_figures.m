function manifest = make_reviewer2_final_figures(reports,saveFigures)
%MAKE_REVIEWER2_FINAL_FIGURES Create the final manuscript figure set.
%
% Final conventions:
%   * performance comparisons use 20-run mean +/- sample standard deviation;
%   * convergence curves that must be compared are overlaid on one axes;
%   * no filled convergence uncertainty bands are used;
%   * each bar/trend metric is a separate figure for LaTeX subfigure assembly;
%   * 6000-FE runtime is reported in tables, not repeated as a figure;
%   * objective/cost bars include the matched long-run AO GA baseline mean;
%   * 3-D geometry uses the established introduction-figure renderer;
%   * all text is Times New Roman with 12-point minimum and 14-point labels.

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

if isfield(reports,'runtime')
    r = reports.runtime;
    out = prepare_output(r.analysisDirectory,saveFigures);
    baseline = matched_baseline(baselineResults,"LUNAR_GATEWAY",3,1);

    plot_runtime_metric(r,'BestJMean','BestJStd','Final best objective', ...
        "runtime_1200_objective",out,saveFigures,style,false,baseline);
    manifest = add_manifest(manifest,"runtime","runtime_1200_objective", ...
        "Equal-1200-FE objective, mean +/- sample std, with matched 6000-FE AO GA baseline.");

    % Runtime is the central point of the focused BO study and therefore is
    % retained here. The 6000-FE comparison/baseline runtimes remain tables.
    plot_runtime_metric(r,'BudgetRuntimeMean_s','BudgetRuntimeStd_s', ...
        'Runtime to 1200 FE (s)',"runtime_1200_runtime",out,saveFigures,style,true,table());
    manifest = add_manifest(manifest,"runtime","runtime_1200_runtime", ...
        "Equal-1200-FE computational cost; BO slowdown is annotated.");

    plot_runtime_convergence_overlay(r,out,saveFigures,style);
    manifest = add_manifest(manifest,"runtime","runtime_1200_convergence", ...
        "All five equal-FE mean best-so-far curves overlaid; final-FE sample std only.");
end

if isfield(reports,'comparison')
    r = reports.comparison;
    out = prepare_output(r.analysisDirectory,saveFigures);
    missions = string(r.missions);

    refs = matched_baselines(baselineResults,missions,3,1);
    plot_comparison_metric(r,'BestJMean','BestJStd','Final best objective', ...
        "comparison_6000_objective",out,saveFigures,style,refs);
    manifest = add_manifest(manifest,"comparison","comparison_6000_objective", ...
        "Optimizer mean +/- sample std objective with case-matched AO GA baseline references.");

    comparisonSpecs = { ...
        'RMSEPosMean_km','RMSEPosStd_km','Position RMSE (km)','comparison_6000_position_rmse'; ...
        'EffectiveSigmaPosMean_km','EffectiveSigmaPosStd_km','Effective position sigma (km)','comparison_6000_effective_sigma'; ...
        'MeanStabilityMean','MeanStabilityStd','Mean observer stability index','comparison_6000_stability'};
    for q = 1:size(comparisonSpecs,1)
        plot_comparison_metric(r,comparisonSpecs{q,1},comparisonSpecs{q,2}, ...
            comparisonSpecs{q,3},string(comparisonSpecs{q,4}), ...
            out,saveFigures,style,table());
        manifest = add_manifest(manifest,"comparison",string(comparisonSpecs{q,4}), ...
            "Case-wise optimizer comparison using mean +/- sample std.");
    end

    for mission = missions
        stem = "comparison_6000_convergence_"+mission_code(mission);
        plot_comparison_convergence_overlay(r,mission,out,stem,saveFigures,style);
        manifest = add_manifest(manifest,"comparison",stem, ...
            "GA/PSO/ABC/ACO mean convergence overlaid for direct comparison.");
    end

    selection = select_reviewer2_representative_runs(r,"comparison");
    writetable(selection,fullfile(char(r.analysisDirectory), ...
        'comparison_6000_geometry_representative_runs.csv'));
    details = plot_reviewer2_geometry_grid(selection,out,"comparison_geometry",saveFigures);
    manifest = add_geometry_manifest(manifest,"comparison",details, ...
        "Representative optimizer geometry nearest each 20-run mean objective.");
end

if isfield(reports,'baseline')
    r = reports.baseline;
    out = prepare_output(r.analysisDirectory,saveFigures);
    missions = ["LUNAR_GATEWAY","LOW_THRUST_TRANSFER","GATEWAY_IMPULSE"];

    observerSpecs = { ...
        'BestJMean','BestJStd','Final best objective','objective'; ...
        'RMSEPosMean_km','RMSEPosStd_km','Position RMSE (km)','position_rmse'; ...
        'EffectiveSigmaPosMean_km','EffectiveSigmaPosStd_km','Effective position sigma (km)','effective_sigma'};
    for mission = missions
        for q = 1:size(observerSpecs,1)
            stem = "baseline_observer_"+string(observerSpecs{q,4})+"_"+mission_code(mission);
            plot_baseline_observer_metric(r,mission,observerSpecs{q,1}, ...
                observerSpecs{q,2},observerSpecs{q,3},out,stem,saveFigures,style);
            manifest = add_manifest(manifest,"baseline",stem, ...
                "AO/AR observer-count trend using mean +/- sample std.");
        end
        for meas = ["ANGLES_ONLY","ANGLES_RANGE"]
            stem = "baseline_convergence_observers_"+mission_code(mission)+ ...
                "_"+measurement_code(meas);
            plot_baseline_observer_convergence_overlay( ...
                r,mission,meas,out,stem,saveFigures,style);
            manifest = add_manifest(manifest,"baseline",stem, ...
                "3/5/7/10-observer GA mean convergence overlaid on one axes.");
        end
    end

    durationSpecs = { ...
        'BestJMean','BestJStd','Final best objective','objective'; ...
        'RMSEPosMean_km','RMSEPosStd_km','Position RMSE (km)','position_rmse'; ...
        'EffectiveSigmaPosMean_km','EffectiveSigmaPosStd_km','Effective position sigma (km)','effective_sigma'};
    for meas = ["ANGLES_ONLY","ANGLES_RANGE"]
        for q = 1:size(durationSpecs,1)
            stem = "baseline_gateway_duration_"+string(durationSpecs{q,4})+ ...
                "_"+measurement_code(meas);
            plot_baseline_duration_metric(r,meas,durationSpecs{q,1}, ...
                durationSpecs{q,2},durationSpecs{q,3},out,stem,saveFigures,style);
            manifest = add_manifest(manifest,"baseline",stem, ...
                "Gateway-duration effect for 3/5/7/10 observers, mean +/- sample std.");
        end
        stem = "baseline_convergence_duration_"+measurement_code(meas);
        plot_baseline_duration_convergence_overlay(r,meas,out,stem,saveFigures,style);
        manifest = add_manifest(manifest,"baseline",stem, ...
            "1/3/5-period GA mean convergence overlaid on one axes.");
    end

    selection = select_reviewer2_representative_runs(r,"baseline");
    writetable(selection,fullfile(char(r.analysisDirectory), ...
        'baseline_6000_geometry_representative_runs.csv'));
    details = plot_reviewer2_geometry_grid(selection,out,"baseline_geometry",saveFigures);
    manifest = add_geometry_manifest(manifest,"baseline",details, ...
        "Representative AO geometry nearest the mean objective for each observer count.");
end

if isfield(reports,'objective_screening')
    r = reports.objective_screening;
    out = prepare_output(r.analysisDirectory,saveFigures);
    missions = ["LUNAR_GATEWAY","LOW_THRUST_TRANSFER","GATEWAY_IMPULSE"];

    screeningSpecs = { ...
        'BestJMean','BestJStd','Final best objective','objective'; ...
        'RMSEPosMean_km','RMSEPosStd_km','Position RMSE (km)','position_rmse'; ...
        'EffectiveSigmaPosMean_km','EffectiveSigmaPosStd_km','Effective position sigma (km)','effective_sigma'; ...
        'ScreeningMean','ScreeningStd','Rejected measurement opportunities','screening_count'};
    componentSpecs = { ...
        'RMSEPosMean_km','RMSEPosStd_km','Position RMSE (km)','position_rmse'; ...
        'EffectiveSigmaPosMean_km','EffectiveSigmaPosStd_km','Effective position sigma (km)','effective_sigma'; ...
        'MeanStabilityMean','MeanStabilityStd','Mean stability index','stability'; ...
        'CoverageMean','CoverageStd','Coverage fraction','coverage'};

    for mission = missions
        for q = 1:size(screeningSpecs,1)
            stem = "ga_screening_"+string(screeningSpecs{q,4})+"_"+mission_code(mission);
            plot_configuration_metric(r.results,mission, ...
                ["combined_on","combined_off"],screeningSpecs{q,1}, ...
                screeningSpecs{q,2},screeningSpecs{q,3},out,stem,saveFigures,style);
            manifest = add_manifest(manifest,"objective_screening",stem, ...
                "Screening ON/OFF comparison using mean +/- sample std.");
        end

        stem = "ga_screening_convergence_"+mission_code(mission);
        plot_screening_convergence_overlay(r,mission,out,stem,saveFigures,style);
        manifest = add_manifest(manifest,"objective_screening",stem, ...
            "Matched J111 screening ON/OFF mean convergence overlaid on one axes.");

        % J111/J100/J010/J001 are different scalar objectives. Compare their
        % physical metrics rather than their total objective values.
        for q = 1:size(componentSpecs,1)
            stem = "ga_objective_"+string(componentSpecs{q,4})+"_"+mission_code(mission);
            plot_configuration_metric(r.results,mission, ...
                ["combined_on","j1_only","j2_only","j3_only"], ...
                componentSpecs{q,1},componentSpecs{q,2},componentSpecs{q,3}, ...
                out,stem,saveFigures,style);
            manifest = add_manifest(manifest,"objective_screening",stem, ...
                "J111/J100/J010/J001 physical-metric comparison using mean +/- sample std.");
        end

        stem = "ga_objective_families_"+mission_code(mission);
        plot_objective_family_summary(r,mission,out,stem,saveFigures,style);
        manifest = add_manifest(manifest,"objective_screening",stem, ...
            "Selected orbit-family distribution by objective configuration.");
    end

    selection = select_reviewer2_representative_runs(r,"objective_screening");
    writetable(selection,fullfile(char(r.analysisDirectory), ...
        'ga_objective_screening_geometry_representative_runs.csv'));
    screeningSelection = selection(ismember(selection.PanelKey, ...
        ["combined_on","combined_off"]),:);
    componentSelection = selection(ismember(selection.PanelKey, ...
        ["combined_on","j1_only","j2_only","j3_only"]),:);
    screeningDetails = plot_reviewer2_geometry_grid( ...
        screeningSelection,out,"ga_screening_geometry",saveFigures);
    componentDetails = plot_reviewer2_geometry_grid( ...
        componentSelection,out,"ga_objective_geometry",saveFigures);
    manifest = add_geometry_manifest(manifest,"objective_screening",screeningDetails, ...
        "Representative screening ON/OFF geometry nearest each group mean objective.");
    manifest = add_geometry_manifest(manifest,"objective_screening",componentDetails, ...
        "Representative objective-component geometry nearest each group mean objective.");
end

write_manifests(reports,manifest);
end


function out = prepare_output(analysisDir,saveFigures)
% Final CSVs, EPS files, PNG files, and the manifest all live together in
% results/<study>_<timestamp>/; no nested paper_final folder is created.
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
    'VariableNames',manifest.Properties.VariableNames)]; %#ok<AGROW>
end


function manifest = add_geometry_manifest(manifest,study,details,purpose)
if isempty(details), return; end
assert(ismember('FigureStem',details.Properties.VariableNames));
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


function ref = matched_baseline(B,mission,numObservers,nPeriods)
ref = table();
if isempty(B), return; end
rows = B(B.Mission == mission & B.Measurement == "ANGLES_ONLY" & ...
    B.NumObservers == numObservers & B.NPeriods == nPeriods,:);
if height(rows) ~= 1, return; end
ref = table(string(mission),rows.BestJMean,rows.BestJStd, ...
    'VariableNames',{'Mission','Mean','Std'});
end


function refs = matched_baselines(B,missions,numObservers,nPeriods)
refs = table();
if isempty(B), return; end
for mission = string(missions(:)')
    row = matched_baseline(B,mission,numObservers,nPeriods);
    if ~isempty(row), refs = [refs;row]; end %#ok<AGROW>
end
end


function plot_runtime_metric(r,valueField,stdField,yLabel,stem,out,saveFigures,style,annotateBO,baseline)
R = r.runtimeResults;
order = style.optimizerOrder(ismember(style.optimizerOrder,R.Optimizer));
R = sort_to_order(R,'Optimizer',order);
colors = colors_for_optimizers(R.Optimizer,style);
fig = paper_figure(style.figureWidth,style.figureHeight,style);
ax = axes(fig); hold(ax,'on'); box(ax,'on'); grid(ax,'on');
values = R.(valueField); errors = R.(stdField);
b = bar(ax,1:height(R),values,0.72,'FaceColor','flat'); b.CData = colors;
errorbar(ax,1:height(R),values,errors,'k.','LineWidth',1.0, ...
    'CapSize',style.capSize,'HandleVisibility','off');
format_category_axis(ax,optimizer_labels(R.Optimizer),yLabel,style);
legendHandles = b; legendLabels = optimizer_labels(R.Optimizer);

if ~isempty(baseline)
    hBase = plot(ax,[0.55 height(R)+0.45],[baseline.Mean baseline.Mean],'--', ...
        'Color',[0.30 0.30 0.30],'LineWidth',1.5,'DisplayName','Baseline AO');
    errorbar(ax,0.72,baseline.Mean,baseline.Std,'none','Color',[0.30 0.30 0.30], ...
        'LineWidth',1.0,'CapSize',style.capSize,'HandleVisibility','off');
    legendHandles = [legendHandles;hBase];
    legendLabels = [legendLabels;"Baseline AO"];
end

if annotateBO
    idxBO = find(R.Optimizer == "BAYESIAN",1);
    if ~isempty(idxBO)
        fastest = min(values(R.Optimizer ~= "BAYESIAN"));
        ratio = values(idxBO)/fastest;
        text(ax,idxBO,values(idxBO)+errors(idxBO),sprintf('%.1fx fastest',ratio), ...
            'HorizontalAlignment','center','VerticalAlignment','bottom', ...
            'FontName',style.fontName,'FontSize',style.fontSize,'FontWeight','bold');
    end
end

lgd = legend(ax,legendHandles,cellstr(legendLabels),'Location','northoutside', ...
    'Orientation','horizontal','NumColumns',min(numel(legendLabels),6),'Box','off');
style_legend(lgd,style);
export_figure(fig,out,stem,saveFigures,style);
end


function plot_runtime_convergence_overlay(r,out,saveFigures,style)
files = dir(fullfile(char(r.analysisDirectory),'convergence_*.mat'));
assert(numel(files) == 1,'Expected one runtime convergence file.');
S = load(fullfile(files(1).folder,files(1).name),'curves');
optimizers = style.optimizerOrder(ismember(style.optimizerOrder, ...
    upper(string({S.curves.optimizer}))));
curves = cell(numel(optimizers),1);
for k = 1:numel(optimizers)
    idx = find(upper(string({S.curves.optimizer})) == optimizers(k),1);
    curves{k} = S.curves(idx);
end
plot_curve_overlay(curves,optimizer_labels(optimizers), ...
    colors_for_optimizers(optimizers,style),r.budget,out, ...
    "runtime_1200_convergence",saveFigures,style);
end


function plot_comparison_metric(r,valueField,stdField,yLabel,stem,out,saveFigures,style,baselineRefs)
R = r.results; missions = string(r.missions); optimizers = string(r.optimizers);
[values,errors] = grouped_values(R,missions,optimizers,valueField,stdField);
fig = paper_figure(style.figureWidth,style.figureHeight,style);
ax = axes(fig); hold(ax,'on'); box(ax,'on'); grid(ax,'on');
b = bar(ax,1:numel(missions),values,'grouped'); drawnow;
for k = 1:numel(optimizers)
    b(k).FaceColor = optimizer_color(optimizers(k),style);
    errorbar(ax,b(k).XEndPoints,values(:,k),errors(:,k),'k.', ...
        'LineWidth',0.9,'CapSize',style.capSize,'HandleVisibility','off');
end
ax.XTick = 1:numel(missions);
ax.XTickLabel = cellstr(mission_labels(missions));
ylabel(ax,yLabel,'FontWeight','bold');
style_axes(ax,style);
legendHandles = b(:); legendLabels = optimizer_labels(optimizers);

if ~isempty(baselineRefs)
    hBase = gobjects(1,1);
    for m = 1:numel(missions)
        row = baselineRefs(baselineRefs.Mission == missions(m),:);
        if height(row) ~= 1, continue; end
        h = plot(ax,[m-0.46 m+0.46],[row.Mean row.Mean],'--', ...
            'Color',[0.30 0.30 0.30],'LineWidth',1.5,'HandleVisibility','off');
        errorbar(ax,m,row.Mean,row.Std,'none','Color',[0.30 0.30 0.30], ...
            'LineWidth',1.0,'CapSize',style.capSize,'HandleVisibility','off');
        if ~isgraphics(hBase), hBase = h; end
    end
    if isgraphics(hBase)
        set(hBase,'HandleVisibility','on','DisplayName','Baseline AO');
        legendHandles = [legendHandles;hBase];
        legendLabels = [legendLabels;"Baseline AO"];
    end
end

lgd = legend(ax,legendHandles,cellstr(legendLabels),'Location','northoutside', ...
    'Orientation','horizontal','NumColumns',min(numel(legendLabels),5),'Box','off');
style_legend(lgd,style);
export_figure(fig,out,stem,saveFigures,style);
end


function plot_comparison_convergence_overlay(r,mission,out,stem,saveFigures,style)
key = comparison_key_for_mission(r,mission);
S = load(fullfile(char(r.analysisDirectory),"convergence_"+key+".mat"),'curves');
optimizers = string(r.optimizers); curves = cell(numel(optimizers),1);
for k = 1:numel(optimizers)
    idx = find(upper(string({S.curves.optimizer})) == optimizers(k),1);
    assert(~isempty(idx),'Missing convergence curve for %s.',optimizers(k));
    curves{k} = S.curves(idx);
end
plot_curve_overlay(curves,optimizer_labels(optimizers), ...
    colors_for_optimizers(optimizers,style),r.budget,out,stem,saveFigures,style);
end


function key = comparison_key_for_mission(r,mission)
if ismember('ComparisonKey',r.results.Properties.VariableNames)
    rows = r.results(r.results.Mission == mission,:);
    keys = unique(string(rows.ComparisonKey));
else
    rows = r.summary(r.summary.mission == mission,:);
    keys = unique(string(rows.comparison_key));
end
assert(numel(keys) == 1,'Expected one comparison key for %s.',mission);
key = keys(1);
end


function plot_baseline_observer_metric(r,mission,valueField,stdField,yLabel,out,stem,saveFigures,style)
R = r.results; measurements = ["ANGLES_ONLY","ANGLES_RANGE"]; counts = [3 5 7 10];
fig = paper_figure(style.figureWidth,style.figureHeight,style);
ax = axes(fig); hold(ax,'on'); box(ax,'on'); grid(ax,'on');
handles = gobjects(2,1);
for m = 1:2
    values = nan(size(counts)); errors = values;
    for k = 1:numel(counts)
        row = R(R.Mission == mission & R.Measurement == measurements(m) & ...
            R.NumObservers == counts(k) & R.NPeriods == 1,:);
        assert(height(row) == 1,'Missing baseline observer-count point.');
        values(k) = row.(valueField); errors(k) = row.(stdField);
    end
    c = style.measurementColors(m,:);
    handles(m) = errorbar(ax,counts,values,errors,'-o','Color',c, ...
        'LineWidth',style.lineWidth,'MarkerSize',style.markerSize, ...
        'MarkerFaceColor',c,'CapSize',style.capSize, ...
        'DisplayName',measurement_label(measurements(m)));
end
ax.XTick = counts;
xlabel(ax,'Number of observers','FontWeight','bold');
ylabel(ax,yLabel,'FontWeight','bold');
style_axes(ax,style);
lgd = legend(ax,handles,{'AO','AR'},'Location','northoutside', ...
    'Orientation','horizontal','Box','off');
style_legend(lgd,style);
export_figure(fig,out,stem,saveFigures,style);
end


function plot_baseline_duration_metric(r,measurement,valueField,stdField,yLabel,out,stem,saveFigures,style)
R = r.results; counts = [3 5 7 10]; periods = [1 3 5]; colors = lines(numel(counts));
fig = paper_figure(style.figureWidth,style.figureHeight,style);
ax = axes(fig); hold(ax,'on'); box(ax,'on'); grid(ax,'on');
handles = gobjects(numel(counts),1);
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
ax.XTick = periods;
xlabel(ax,'Gateway tracking periods','FontWeight','bold');
ylabel(ax,yLabel,'FontWeight','bold');
style_axes(ax,style);
lgd = legend(ax,handles,'Location','northoutside','Orientation','horizontal', ...
    'NumColumns',2,'Box','off');
style_legend(lgd,style);
export_figure(fig,out,stem,saveFigures,style);
end


function plot_baseline_observer_convergence_overlay(r,mission,measurement,out,stem,saveFigures,style)
counts = [3 5 7 10]; curves = cell(4,1); labels = strings(4,1); colors = lines(4);
for k = 1:4
    row = r.results(r.results.Mission == mission & r.results.Measurement == measurement & ...
        r.results.NumObservers == counts(k) & r.results.NPeriods == 1,:);
    assert(height(row) == 1,'Missing baseline convergence configuration.');
    curves{k} = load_ga_curve(r.analysisDirectory,row.ComparisonKey);
    labels(k) = string(counts(k))+" observers";
end
plot_curve_overlay(curves,labels,colors,r.budget,out,stem,saveFigures,style);
end


function plot_baseline_duration_convergence_overlay(r,measurement,out,stem,saveFigures,style)
periods = [1 3 5]; curves = cell(3,1); labels = strings(3,1); colors = lines(3);
for k = 1:3
    row = r.results(r.results.Mission == "LUNAR_GATEWAY" & ...
        r.results.Measurement == measurement & r.results.NumObservers == 3 & ...
        r.results.NPeriods == periods(k),:);
    assert(height(row) == 1,'Missing baseline duration convergence configuration.');
    curves{k} = load_ga_curve(r.analysisDirectory,row.ComparisonKey);
    labels(k) = string(periods(k))+" period"+plural_s(periods(k));
end
plot_curve_overlay(curves,labels,colors,r.budget,out,stem,saveFigures,style);
end


function plot_configuration_metric(R,mission,configs,valueField,stdField,yLabel,out,stem,saveFigures,style)
colors = colors_for_configurations(configs,style);
values = nan(numel(configs),1); errors = values;
for k = 1:numel(configs)
    row = objective_result(R,mission,configs(k));
    values(k) = row.(valueField); errors(k) = row.(stdField);
end
fig = paper_figure(style.figureWidth,style.figureHeight,style);
ax = axes(fig); hold(ax,'on'); box(ax,'on'); grid(ax,'on');
b = bar(ax,1:numel(configs),values,0.72,'FaceColor','flat'); b.CData = colors;
errorbar(ax,1:numel(configs),values,errors,'k.','LineWidth',0.9, ...
    'CapSize',style.capSize,'HandleVisibility','off');
ax.XTick = 1:numel(configs);
ax.XTickLabel = cellstr(configuration_labels(configs));
ax.XTickLabelRotation = 18;
ylabel(ax,yLabel,'FontWeight','bold');
style_axes(ax,style);
export_figure(fig,out,stem,saveFigures,style);
end


function plot_screening_convergence_overlay(r,mission,out,stem,saveFigures,style)
configs = ["combined_on","combined_off"]; curves = cell(2,1);
for k = 1:2
    row = objective_result(r.results,mission,configs(k));
    curves{k} = load_ga_curve(r.analysisDirectory,row.ComparisonKey);
end
plot_curve_overlay(curves,configuration_labels(configs), ...
    colors_for_configurations(configs,style),r.budget,out,stem,saveFigures,style);
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
fig = paper_figure(style.figureWidth,style.figureHeight,style);
ax = axes(fig); hold(ax,'on'); box(ax,'on'); grid(ax,'on');
b = bar(ax,1:numel(configs),values,'stacked');
ax.XTick = 1:numel(configs);
ax.XTickLabel = cellstr(configuration_labels(configs));
ax.XTickLabelRotation = 18;
ylim(ax,[0 100]);
xlabel(ax,'Objective configuration','FontWeight','bold');
ylabel(ax,'Selected observers (%)','FontWeight','bold');
style_axes(ax,style);
lgd = legend(ax,b,cellstr(families),'Location','northoutside', ...
    'Orientation','horizontal','NumColumns',numel(families),'Box','off');
style_legend(lgd,style);
export_figure(fig,out,stem,saveFigures,style);
end


function plot_curve_overlay(curves,labels,colors,budget,out,stem,saveFigures,style)
assert(numel(curves) == numel(labels) && size(colors,1) == numel(curves));
fig = paper_figure(style.convergenceFigureWidth,style.convergenceFigureHeight,style);
ax = axes(fig); hold(ax,'on'); box(ax,'on'); grid(ax,'on');
handles = gobjects(numel(curves),1); allY = zeros(0,1);
for k = 1:numel(curves)
    c = curves{k}; valid = c.fe >= 60 & isfinite(c.mean);
    assert(any(valid),'Convergence curve contains no valid FE >= 60.');
    x = double(c.fe(valid)); y = double(c.mean(valid));
    handles(k) = stairs(ax,x,y,'Color',colors(k,:), ...
        'LineWidth',style.lineWidth,'DisplayName',string(labels(k)));
    dEnd = double(c.std(find(valid,1,'last')));
    if isfinite(dEnd)
        errorbar(ax,x(end),y(end),dEnd,'o','Color',colors(k,:), ...
            'MarkerFaceColor',colors(k,:),'MarkerSize',5.0, ...
            'LineWidth',1.0,'CapSize',style.capSize,'HandleVisibility','off');
        allY = [allY;y;y(end)-dEnd;y(end)+dEnd]; %#ok<AGROW>
    else
        plot(ax,x(end),y(end),'o','Color',colors(k,:), ...
            'MarkerFaceColor',colors(k,:),'MarkerSize',5.0,'HandleVisibility','off');
        allY = [allY;y]; %#ok<AGROW>
    end
end
allY = allY(isfinite(allY)); lo = min(allY); hi = max(allY);
span = max(hi-lo,0.05*max(1,abs(hi)));
yLimits = [lo-0.06*span,hi+0.08*span];
if yLimits(1) >= 0, yLimits(1) = max(0,yLimits(1)); end
xlim(ax,[60 budget]); ylim(ax,yLimits);
xlabel(ax,'Function evaluations','FontWeight','bold');
ylabel(ax,'Mean best-so-far objective','FontWeight','bold');
style_axes(ax,style);
lgd = legend(ax,handles,'Location','northoutside','Orientation','horizontal', ...
    'NumColumns',min(numel(handles),5),'Box','off');
style_legend(lgd,style);
export_figure(fig,out,stem,saveFigures,style);
end


function curve = load_ga_curve(analysisDir,key)
S = load(fullfile(char(analysisDir),"convergence_"+string(key)+".mat"),'curves');
idx = find(upper(string({S.curves.optimizer})) == "GA",1);
assert(~isempty(idx),'Missing GA convergence curve.');
curve = S.curves(idx);
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
assert(~isempty(idx),'Unknown optimizer color: %s',optimizer);
c = style.optimizerColors(idx,:);
end


function colors = colors_for_configurations(configs,style)
colors = zeros(numel(configs),3);
for k = 1:numel(configs)
    idx = find(style.configurationOrder == string(configs(k)),1);
    assert(~isempty(idx),'Unknown configuration color: %s',configs(k));
    colors(k,:) = style.configurationColors(idx,:);
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
enforce_minimum_font_size(fig,12); drawnow;
if ~saveFigures, return; end
base = fullfile(char(out),char(stem));
print(fig,[base '.eps'],'-depsc2','-painters','-r600');
exportgraphics(fig,[base '.png'],'Resolution',style.exportDpi);
close(fig);
end


function enforce_minimum_font_size(fig,minFontSize)
objects = findall(fig,'-property','FontSize');
for k = 1:numel(objects)
    try
        if objects(k).FontSize < minFontSize, objects(k).FontSize = minFontSize; end
    catch
    end
end
end


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