function report = run_reviewer2_objective_screening_pipeline(saveFigures)
%RUN_REVIEWER2_OBJECTIVE_SCREENING_PIPELINE Process the completed GA study.
%
% Study definition:
%   GA, 6000 FE, optimizer seeds 0:19, fixed measurement seed 1001
%   three target cases, angles only, three observers, one Gateway period
%   combined objective with screening ON/OFF
%   J1-only, J2-only, and J3-only with screening ON
%
% Total objective values are compared only for combined_on/combined_off,
% where the mathematical objective is identical. Objective-component cases
% are compared using RMSE, effective covariance sigma, stability, coverage,
% and selected orbit-family distributions.

if nargin < 1 || isempty(saveFigures), saveFigures = true; end
validateattributes(saveFigures,{'logical','numeric'},{'scalar'});
saveFigures = logical(saveFigures);

paths = setup_project();
root = fullfile(paths.runs,'GA_OBJECTIVE_SCREENING');
studyID = "reviewer2_ga_objective_screening_v1";
budget = 6000;
seeds = 0:19;
missions = ["LUNAR_GATEWAY","LOW_THRUST_TRANSFER","GATEWAY_IMPULSE"];
configurations = ["combined_on","combined_off","j1_only","j2_only","j3_only"];
expectedGroups = numel(missions)*numel(configurations);
expectedRuns = expectedGroups*numel(seeds);

fprintf('\n--- Reviewer 2 GA objective/screening pipeline ---\n');
fprintf('Configurations:              %s\n',strjoin(cellstr(configurations),', '));
fprintf('Target cases:                %d\n',numel(missions));
fprintf('Independent seeds:           %s\n',mat2str(seeds));
fprintf('Search FE per run:           %d\n',budget);
fprintf('Expected optimization runs:  %d\n\n',expectedRuns);

test_project_structure();
test_ga_objective_screening_configuration();
assert(isfolder(root),'GA objective/screening root does not exist: %s',root);
[summary,inventory] = process_fe_convergence( ...
    root,studyID,seeds,budget,false,"GA");

nonemptyRuns = inventory.run_file ~= "";
assert(sum(nonemptyRuns) == expectedRuns, ...
    'Expected %d saved runs but found %d.',expectedRuns,sum(nonemptyRuns));
assert(~any(~inventory.valid), ...
    'One or more GA objective/screening runs failed validation.');
assert(height(summary) == expectedGroups && all(summary.optimizer == "GA") && ...
    all(summary.n_runs == numel(seeds)) && all(summary.fe_budget == budget), ...
    'Processed study metadata does not match the intended factorial design.');

analysisDir = newest_analysis_directory(root);
runMetrics = readtable(fullfile(analysisDir,'final_run_metrics.csv'), ...
    'TextType','string','VariableNamingRule','preserve');
requiredMetrics = ["bestJ","rmse_pos_km","mean_effective_sigma_pos_km", ...
    "mean_stability","coverage_epoch_fraction","mean_available_observers", ...
    "screening_count","J1_weighted","J2_weighted","J3_weighted","run_file"];
assert(height(runMetrics) == expectedRuns && ...
    all(ismember(requiredMetrics,string(runMetrics.Properties.VariableNames))), ...
    'Processed run metrics are incomplete.');

results = build_results(summary,runMetrics,missions,configurations,seeds);
validate_factorial(results,missions,configurations);
formatted = format_results(results);
screeningContrasts = build_screening_contrasts(results,missions);
componentWinners = build_component_winners(results,missions);
familySelection = build_family_selection(results,runMetrics);

fprintf('\n--- GA objective/screening results (mean +/- sample std) ---\n');
disp(formatted);
fprintf(['\nNOTE: total objective is comparable only between combined_on and ' ...
    'combined_off.\n']);
fprintf('\n--- Screening ON relative to screening OFF (percent change) ---\n');
disp(screeningContrasts);
fprintf('\n--- Objective-component metric winners ---\n');
disp(componentWinners);

writetable(results,fullfile(analysisDir,'ga_objective_screening_results.csv'));
writetable(formatted,fullfile(analysisDir,'ga_objective_screening_formatted.csv'));
writetable(screeningContrasts, ...
    fullfile(analysisDir,'ga_screening_contrasts.csv'));
writetable(componentWinners, ...
    fullfile(analysisDir,'ga_objective_component_winners.csv'));
writetable(familySelection, ...
    fullfile(analysisDir,'ga_objective_family_selection.csv'));

figureDir = "";
if saveFigures
    figureDir = string(fullfile(analysisDir,'paper_preview'));
    if ~isfolder(figureDir), mkdir(figureDir); end
end
for mission = missions
    plot_screening_convergence(results,analysisDir,mission,budget,figureDir,saveFigures);
    plot_screening_metrics(results,mission,figureDir,saveFigures);
    plot_component_metrics(results,mission,figureDir,saveFigures);
    plot_family_selection(familySelection,mission,configurations,figureDir,saveFigures);
end

report = struct('studyID',studyID,'budget',budget,'seeds',seeds, ...
    'analysisDirectory',string(analysisDir),'figureDirectory',figureDir, ...
    'summary',summary,'inventory',inventory,'runMetrics',runMetrics, ...
    'results',results,'screeningContrasts',screeningContrasts, ...
    'componentWinners',componentWinners,'familySelection',familySelection);

fprintf('\nReviewer 2 GA objective/screening pipeline passed.\n');
fprintf('Validated runs: %d/%d\n',sum(inventory.valid),expectedRuns);
fprintf('Processed data: %s\n',analysisDir);
if saveFigures, fprintf('Paper-style previews: %s\n',figureDir); end
end


function analysisDir = newest_analysis_directory(root)
directories = dir(fullfile(root,'FE_DATA_*'));
directories = directories([directories.isdir]);
assert(~isempty(directories),'No FE_DATA analysis directory created under %s.',root);
[~,idx] = max([directories.datenum]);
analysisDir = fullfile(directories(idx).folder,directories(idx).name);
end


function results = build_results(summary,runMetrics,missions,configurations,seeds)
n = height(summary);
comparisonKey = strings(n,1); missionColumn = strings(n,1);
configuration = strings(n,1); screening = false(n,1);
useJ1 = false(n,1); useJ2 = false(n,1); useJ3 = false(n,1);
nRuns = nan(n,1); bestJMean = nan(n,1); bestJStd = nan(n,1);
rmseMean = nan(n,1); rmseStd = nan(n,1); sigmaMean = nan(n,1); sigmaStd = nan(n,1);
stabilityMean = nan(n,1); stabilityStd = nan(n,1);
coverageMean = nan(n,1); coverageStd = nan(n,1);
availableMean = nan(n,1); availableStd = nan(n,1);
screeningMean = nan(n,1); screeningStd = nan(n,1);
j1Mean = nan(n,1); j1Std = nan(n,1); j2Mean = nan(n,1); j2Std = nan(n,1);
j3Mean = nan(n,1); j3Std = nan(n,1);

for k = 1:n
    key = summary.comparison_key(k);
    rows = sortrows(runMetrics(runMetrics.comparison_key == key,:),'seed');
    assert(height(rows) == numel(seeds) && isequal(rows.seed(:)',seeds), ...
        'Incomplete GA sensitivity seed group for key %s.',key);
    S = load(rows.run_file(1),'runState'); s = S.runState.settings;
    code = configuration_code(s);
    assert(ismember(code,configurations),'Unexpected objective/screening configuration.');

    comparisonKey(k) = key; missionColumn(k) = string(s.mission.type);
    configuration(k) = code; screening(k) = logical(s.useScreening);
    useJ1(k) = logical(s.costFlags.J1); useJ2(k) = logical(s.costFlags.J2);
    useJ3(k) = logical(s.costFlags.J3); nRuns(k) = height(rows);
    [bestJMean(k),bestJStd(k)] = stats(rows.bestJ);
    [rmseMean(k),rmseStd(k)] = stats(rows.rmse_pos_km);
    [sigmaMean(k),sigmaStd(k)] = stats(rows.mean_effective_sigma_pos_km);
    [stabilityMean(k),stabilityStd(k)] = stats(rows.mean_stability);
    [coverageMean(k),coverageStd(k)] = stats(rows.coverage_epoch_fraction);
    [availableMean(k),availableStd(k)] = stats(rows.mean_available_observers);
    [screeningMean(k),screeningStd(k)] = stats(rows.screening_count);
    [j1Mean(k),j1Std(k)] = stats(rows.J1_weighted);
    [j2Mean(k),j2Std(k)] = stats(rows.J2_weighted);
    [j3Mean(k),j3Std(k)] = stats(rows.J3_weighted);
end
results = table(comparisonKey,missionColumn,configuration,screening,useJ1,useJ2,useJ3, ...
    nRuns,bestJMean,bestJStd,rmseMean,rmseStd,sigmaMean,sigmaStd, ...
    stabilityMean,stabilityStd,coverageMean,coverageStd,availableMean,availableStd, ...
    screeningMean,screeningStd,j1Mean,j1Std,j2Mean,j2Std,j3Mean,j3Std, ...
    'VariableNames',{'ComparisonKey','Mission','Configuration','Screening', ...
    'UseJ1','UseJ2','UseJ3','NRuns','BestJMean','BestJStd','RMSEPosMean_km', ...
    'RMSEPosStd_km','EffectiveSigmaPosMean_km','EffectiveSigmaPosStd_km', ...
    'MeanStabilityMean','MeanStabilityStd','CoverageMean','CoverageStd', ...
    'AvailableObserversMean','AvailableObserversStd','ScreeningMean','ScreeningStd', ...
    'J1Mean','J1Std','J2Mean','J2Std','J3Mean','J3Std'});
results.Configuration = categorical(results.Configuration,configurations,'Ordinal',true);
results = sortrows(results,{'Mission','Configuration'});
end


function code = configuration_code(s)
flags = [logical(s.costFlags.J1),logical(s.costFlags.J2),logical(s.costFlags.J3)];
if all(flags)
    if logical(s.useScreening), code = "combined_on"; else, code = "combined_off"; end
elseif isequal(flags,[true false false]), code = "j1_only";
elseif isequal(flags,[false true false]), code = "j2_only";
elseif isequal(flags,[false false true]), code = "j3_only";
else, error('Study:UnexpectedCostFlags','Unexpected J1/J2/J3 flag combination.');
end
end


function validate_factorial(results,missions,configurations)
assert(height(results) == numel(missions)*numel(configurations));
for mission = missions
    rows = results(results.Mission == mission,:);
    assert(height(rows) == numel(configurations));
    assert(isequal(string(rows.Configuration(:))',configurations));
    assert(all(rows.NRuns == 20));
end
end


function formatted = format_results(results)
formatted = table(mission_labels(results.Mission),configuration_labels(results.Configuration), ...
    results.NRuns,compose('%.6g +/- %.3g',results.BestJMean,results.BestJStd), ...
    compose('%.5g +/- %.3g',results.RMSEPosMean_km,results.RMSEPosStd_km), ...
    compose('%.5g +/- %.3g',results.EffectiveSigmaPosMean_km, ...
        results.EffectiveSigmaPosStd_km), ...
    compose('%.5g +/- %.3g',results.MeanStabilityMean,results.MeanStabilityStd), ...
    compose('%.4f +/- %.3f',results.CoverageMean,results.CoverageStd), ...
    compose('%.4f +/- %.3f',results.AvailableObserversMean, ...
        results.AvailableObserversStd), ...
    compose('%.5g +/- %.3g',results.ScreeningMean,results.ScreeningStd), ...
    'VariableNames',{'Case','Configuration','Runs','BestObjective', ...
    'RMSEPosition_km','EffectiveSigmaPosition_km','MeanStability', ...
    'CoverageFraction','MeanAvailableObservers','RejectedMeasurementOpportunities'});
end


function contrasts = build_screening_contrasts(results,missions)
missionColumn = missions(:); objectiveChangePct = nan(numel(missions),1);
rmseChangePct = objectiveChangePct; sigmaChangePct = objectiveChangePct;
coverageChangePct = objectiveChangePct; availableChangePct = objectiveChangePct;
for k = 1:numel(missions)
    on = get_result(results,missions(k),"combined_on");
    off = get_result(results,missions(k),"combined_off");
    objectiveChangePct(k) = pct(off.BestJMean,on.BestJMean);
    rmseChangePct(k) = pct(off.RMSEPosMean_km,on.RMSEPosMean_km);
    sigmaChangePct(k) = pct(off.EffectiveSigmaPosMean_km,on.EffectiveSigmaPosMean_km);
    coverageChangePct(k) = pct(off.CoverageMean,on.CoverageMean);
    availableChangePct(k) = pct(off.AvailableObserversMean,on.AvailableObserversMean);
end
contrasts = table(missionColumn,objectiveChangePct,rmseChangePct,sigmaChangePct, ...
    coverageChangePct,availableChangePct,'VariableNames',{'Mission', ...
    'ObjectiveChangePct','RMSEChangePct','EffectiveSigmaChangePct', ...
    'CoverageChangePct','AvailableObserversChangePct'});
end


function winners = build_component_winners(results,missions)
configs = ["combined_on","j1_only","j2_only","j3_only"];
metric = ["Position RMSE";"Effective position sigma";"Mean stability"];
fields = ["RMSEPosMean_km","EffectiveSigmaPosMean_km","MeanStabilityMean"];
missionColumn = strings(numel(missions)*numel(fields),1);
metricColumn = strings(size(missionColumn)); winner = strings(size(missionColumn));
meanValue = nan(size(missionColumn)); row = 0;
for mission = missions
    subset = results(results.Mission == mission & ...
        ismember(string(results.Configuration),configs),:);
    for q = 1:numel(fields)
        row = row+1; [meanValue(row),idx] = min(subset.(fields(q)));
        missionColumn(row) = mission; metricColumn(row) = metric(q);
        winner(row) = string(subset.Configuration(idx));
    end
end
winners = table(missionColumn,metricColumn,winner,meanValue, ...
    'VariableNames',{'Mission','Metric','WinningConfiguration','MeanValue'});
end


function familySelection = build_family_selection(results,runMetrics)
families = ["DRO","NRHO/rectilinear","Halo","Other"];
missionColumn = strings(0,1); configuration = strings(0,1);
familyColumn = strings(0,1); count = zeros(0,1); fraction = zeros(0,1);
for k = 1:height(results)
    rows = runMetrics(runMetrics.comparison_key == results.ComparisonKey(k),:);
    familyValues = strings(0,1);
    for j = 1:height(rows)
        S = load(rows.run_file(j),'runState');
        assert(isfield(S.runState,'observers') && istable(S.runState.observers), ...
            'Run is missing selected observer data.');
        raw = string(S.runState.observers.orbit_family);
        for u = 1:numel(raw), familyValues(end+1,1) = family_group(raw(u)); end %#ok<AGROW>
    end
    for f = 1:numel(families)
        missionColumn(end+1,1) = results.Mission(k); %#ok<AGROW>
        configuration(end+1,1) = string(results.Configuration(k)); %#ok<AGROW>
        familyColumn(end+1,1) = families(f); %#ok<AGROW>
        count(end+1,1) = sum(familyValues == families(f)); %#ok<AGROW>
        fraction(end+1,1) = count(end)/numel(familyValues); %#ok<AGROW>
    end
end
familySelection = table(missionColumn,configuration,familyColumn,count,fraction, ...
    'VariableNames',{'Mission','Configuration','FamilyGroup','Count','Fraction'});
end


function group = family_group(value)
value = upper(string(value));
if contains(value,"DRO"), group = "DRO";
elseif contains(value,"NRHO") || contains(value,"RECT"), group = "NRHO/rectilinear";
elseif contains(value,"HALO") || ~isempty(regexp(value,'^[NS]H?L[12]$','once')), group = "Halo";
else, group = "Other";
end
end


function plot_screening_convergence(results,analysisDir,mission,budget,figureDir,saveFigures)
configs = ["combined_on","combined_off"];
colors = [0.00 0.45 0.74;0.85 0.33 0.10];
fig = paper_figure(7.0,4.4); ax = axes(fig); hold(ax,'on'); box(ax,'on'); grid(ax,'on');
handles = gobjects(2,1);
for k = 1:2
    row = get_result(results,mission,configs(k));
    S = load(fullfile(analysisDir,"convergence_"+string(row.ComparisonKey)+".mat"),'curves');
    curve = S.curves(1); valid = curve.fe >= 60 & isfinite(curve.mean);
    x = double(curve.fe(valid)); y = double(curve.mean(valid)); d = double(curve.std(valid));
    uncertainty_band(ax,x,y,d,colors(k,:));
    handles(k) = stairs(ax,x,y,'Color',colors(k,:),'LineWidth',2, ...
        'DisplayName',configuration_label(configs(k)));
end
xlim(ax,[60 budget]); xticks(ax,unique([60 1000:1000:budget budget]));
xlabel(ax,'Function evaluations','FontWeight','bold');
ylabel(ax,'Mean best-so-far objective','FontWeight','bold'); style_axes(ax);
lgd = legend(ax,handles,'Location','northoutside','Orientation','horizontal'); style_legend(lgd);
export_figure(fig,figureDir,"ga_screening_convergence_"+mission_code(mission),saveFigures);
end


function plot_screening_metrics(results,mission,figureDir,saveFigures)
configs = ["combined_on","combined_off"];
specs = {'BestJMean','BestJStd','Final best objective'; ...
    'RMSEPosMean_km','RMSEPosStd_km','Position RMSE (km)'; ...
    'EffectiveSigmaPosMean_km','EffectiveSigmaPosStd_km','Effective position sigma (km)'; ...
    'AvailableObserversMean','AvailableObserversStd','Mean available observers'};
plot_metric_panel(results,mission,configs,specs, ...
    "ga_screening_metrics_"+mission_code(mission),figureDir,saveFigures);
end


function plot_component_metrics(results,mission,figureDir,saveFigures)
configs = ["combined_on","j1_only","j2_only","j3_only"];
specs = {'RMSEPosMean_km','RMSEPosStd_km','Position RMSE (km)'; ...
    'EffectiveSigmaPosMean_km','EffectiveSigmaPosStd_km','Effective position sigma (km)'; ...
    'MeanStabilityMean','MeanStabilityStd','Mean stability index'; ...
    'AvailableObserversMean','AvailableObserversStd','Mean available observers'};
plot_metric_panel(results,mission,configs,specs, ...
    "ga_objective_metrics_"+mission_code(mission),figureDir,saveFigures);
end


function plot_metric_panel(results,mission,configs,specs,stem,figureDir,saveFigures)
fig = paper_figure(7.6,6.5);
tiled = tiledlayout(fig,2,2,'Padding','compact','TileSpacing','compact');
colors = lines(numel(configs));
for q = 1:4
    ax = nexttile(tiled); hold(ax,'on'); box(ax,'on'); grid(ax,'on');
    values = nan(numel(configs),1); errors = values;
    for k = 1:numel(configs)
        row = get_result(results,mission,configs(k));
        values(k) = row.(specs{q,1}); errors(k) = row.(specs{q,2});
    end
    b = bar(ax,1:numel(configs),values,0.72,'FaceColor','flat');
    b.CData = colors; lower = min(max(errors,0),max(values,0));
    errorbar(ax,1:numel(configs),values,lower,max(errors,0), ...
        'k.','LineWidth',1.0,'CapSize',7,'HandleVisibility','off');
    ax.XTick = 1:numel(configs);
    ax.XTickLabel = cellstr(configuration_labels(configs));
    ax.XTickLabelRotation = 20;
    ylabel(ax,specs{q,3},'FontWeight','bold'); style_axes(ax);
end
export_figure(fig,figureDir,stem,saveFigures);
end


function plot_family_selection(familySelection,mission,configs,figureDir,saveFigures)
families = ["DRO","NRHO/rectilinear","Halo","Other"];
values = zeros(numel(configs),numel(families));
for k = 1:numel(configs)
    for f = 1:numel(families)
        row = familySelection(familySelection.Mission == mission & ...
            familySelection.Configuration == configs(k) & ...
            familySelection.FamilyGroup == families(f),:);
        assert(height(row) == 1); values(k,f) = 100*row.Fraction;
    end
end
fig = paper_figure(7.2,4.8); ax = axes(fig);
b = bar(ax,1:numel(configs),values,'stacked');
ax.XTick = 1:numel(configs); ax.XTickLabel = cellstr(configuration_labels(configs));
ax.XTickLabelRotation = 20; ylim(ax,[0 100]); box(ax,'on'); grid(ax,'on');
xlabel(ax,'Objective/screening configuration','FontWeight','bold');
ylabel(ax,'Selected observers (%)','FontWeight','bold'); style_axes(ax);
lgd = legend(ax,b,cellstr(families),'Location','northoutside', ...
    'Orientation','horizontal','NumColumns',numel(families)); style_legend(lgd);
export_figure(fig,figureDir,"ga_objective_families_"+mission_code(mission),saveFigures);
end


function row = get_result(results,mission,configuration)
row = results(results.Mission == mission & ...
    string(results.Configuration) == configuration,:);
assert(height(row) == 1,'Missing result for %s/%s.',mission,configuration);
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

function uncertainty_band(ax,x,y,d,color)
idx = unique(round(linspace(1,numel(x),min(240,numel(x)))));
x = x(idx); y = y(idx); d = d(idx);
bandColor = 0.82*[1 1 1] + 0.18*color;
fill(ax,[x;flipud(x)],[max(0,y-d);flipud(y+d)],bandColor, ...
    'EdgeColor','none','HandleVisibility','off');
end

function fig = paper_figure(width,height)
fig = figure('Color','w','Units','inches','Position',[1 1 width height], ...
    'PaperUnits','inches','PaperSize',[width height], ...
    'PaperPosition',[0 0 width height],'PaperPositionMode','manual', ...
    'Renderer','painters','InvertHardcopy','off');
end

function style_axes(ax)
set(ax,'FontName','Times New Roman','FontSize',12,'FontWeight','bold', ...
    'LineWidth',1.0,'TickDir','out');
ax.XLabel.FontSize = 14; ax.YLabel.FontSize = 14;
end

function style_legend(lgd)
lgd.FontName = 'Times New Roman'; lgd.FontSize = 12;
lgd.FontWeight = 'bold'; lgd.Box = 'off';
end

function export_figure(fig,figureDir,stem,saveFigures)
drawnow; if ~saveFigures, return; end
base = fullfile(char(figureDir),char(stem));
print(fig,[base '.eps'],'-depsc','-painters');
exportgraphics(fig,[base '.png'],'Resolution',300);
close(fig);
end

function [mu,sigma] = stats(values)
values = double(values(:)); mu = mean(values);
if numel(values) < 2, sigma = NaN; else, sigma = std(values); end
end

function value = pct(before,after)
value = 100*(double(after)-double(before))/max(abs(double(before)),eps);
end
