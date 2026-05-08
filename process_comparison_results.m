%% process_comparison_results_cleaned_eps.m
% Minimal comparison-study postprocessing script.
%
% Outputs only the requested paper figures/tables:
%   1) Trajectory panels for LaTeX subfigure grids.
%   2) Two average-screening-event bar charts total, one per maneuver.
%   3) Individual metric panels for a LaTeX 2x2 figure per maneuver:
%      total cost, RMSE, runtime, and screening events.
%   4) Two J111 cost-component breakdown bar charts, one per maneuver.
%   5) Cost-component / cost-combination information in Excel tables only.
%   6) Baseline comparison using total cost only.
%   7) Orbit-family selection count plots by optimizer and study group.
%
% All figures are exported as EPS files.
% MATLAB does not generate LaTeX. Use LaTeX subfigure/subcaption to arrange
% the exported EPS panels.

clear; clc; close all;

%% ========================= FIND runs =========================

if isfolder(fullfile(pwd, "runs"))
    rootDir = fullfile(pwd, "runs");
elseif strcmpi(string(getCurrentFolderName()), "runs")
    rootDir = pwd;
else
    error("Could not find runs folder. Put this script inside runs or one folder above it, or hard-code rootDir.");
end

fprintf("Using comparison-study runs folder:\n%s\n", rootDir);

%% ========================= SETTINGS =========================

cfg = struct();

cfg.outDir = fullfile(rootDir, "COMPARISON_REPORT_OUTPUT_CLEANED_EPS");

cfg.optimizerOrder = ["GA","PSO","BAYESIAN","ABC","ACO"];
cfg.observerOrder = [3 5 7 10];
cfg.measurementShortOrder = ["ao","ar"];

cfg.paperObserverCount = 3;
cfg.paperScreeningFlag = 1;
cfg.paperCostCombos = ["J111","J110","J101","J011","J100","J010","J001"];

% These control the qualitative trajectory and orbit-family plots.
cfg.displayMeasurementShort = "ao";
cfg.lgPeriods = 1;

% Appearance for metric/bar panels.
cfg.fontName = "Times New Roman";
cfg.fontWeight = "bold";
cfg.axisFontSize = 26;
cfg.labelFontSize = 26;
cfg.legendFontSize = 26;
cfg.legendLocation = "northoutside";
cfg.legendOrientation = "horizontal";
cfg.titleFontSize = 20;
cfg.removeTitles = true;
cfg.axisLineWidth = 1.4;
cfg.lineWidth = 2.2;
cfg.markerSize = 8;
cfg.epsResolution = 600;

% Individual trajectory panel source size. These are designed to be shrunk
% in LaTeX subfigure grids.
cfg.singleFigWidthIn = 12.5;
cfg.singleFigHeightIn = 9.5;

cfg.trajPanelWidthIn = cfg.singleFigWidthIn;
cfg.trajPanelHeightIn = cfg.singleFigHeightIn;
cfg.trajAxisFontSize = 32;
cfg.trajLabelFontSize = 36;
cfg.trajLineWidth = 2.8;
cfg.trajMarkerSize = 60;

% Individual metric/cost panel sizes for LaTeX subfigure layouts.
cfg.metricPanelWidthIn = 8.8;
cfg.metricPanelHeightIn = 6.2;

% Orbit-family plotting controls.
cfg.orbitFamilyUseAllCostCombos = false;       % paper case: J111 only
cfg.orbitFamilyUseAllObserverCounts = true;    % count across 3, 5, 7, 10
cfg.orbitFamilyUseBothScreeningFlags = false;  % paper case: screening ON only

%% ========================= OUTPUT FOLDERS =========================

makeDir(cfg.outDir);
figDir = fullfile(cfg.outDir, "figures"); makeDir(figDir);
trajDir = fullfile(figDir, "trajectory_panels"); makeDir(trajDir);
screenDir = fullfile(figDir, "screening"); makeDir(screenDir);
metricDir = fullfile(figDir, "metric_2x2_panels"); makeDir(metricDir);
costBarDir = fullfile(figDir, "cost_breakdown_bars"); makeDir(costBarDir);
baseDir = fullfile(figDir, "baseline_cost_only"); makeDir(baseDir);
familyDir = fullfile(figDir, "orbit_family_counts"); makeDir(familyDir);
tabDir = fullfile(cfg.outDir, "tables"); makeDir(tabDir);

summaryXlsx = fullfile(tabDir, "Comparison_Consolidated_Summary_CLEANED_EPS.xlsx");
if exist(summaryXlsx, "file")
    delete(summaryXlsx);
end

%% ========================= OPTIONAL BASELINE SUMMARY =========================

baselineRaw = table();
baselineAgg = table();

baselineSummaryXlsx = findBaselineSummaryXlsx(rootDir);
if strlength(baselineSummaryXlsx) > 0 && isfile(baselineSummaryXlsx)
    fprintf("Found baseline summary:\n%s\n", baselineSummaryXlsx);
    try
        baselineRaw = readtable(baselineSummaryXlsx, "Sheet", "RunSummary", "VariableNamingRule", "preserve");
    catch
        try
            baselineRaw = readtable(baselineSummaryXlsx, "Sheet", 1, "VariableNamingRule", "preserve");
        catch ME
            warning("Could not read baseline summary. Baseline comparisons will be skipped.\n%s", ME.message);
            baselineRaw = table();
        end
    end

    if ~isempty(baselineRaw)
        baselineRaw = standardizeSummaryColumns(baselineRaw);
        baselineRaw = addBaselineRunFoldersIfPossible(baselineRaw, baselineSummaryXlsx);
        if all(ismember(["use_J1","use_J2","use_J3"], string(baselineRaw.Properties.VariableNames)))
            baselineRaw.cost_combo = makeCostCombo(baselineRaw.use_J1, baselineRaw.use_J2, baselineRaw.use_J3);
            baselineRaw.all_costs_on = sameNum(baselineRaw.use_J1,1) & sameNum(baselineRaw.use_J2,1) & sameNum(baselineRaw.use_J3,1);
            baselineRaw = baselineRaw(baselineRaw.all_costs_on,:);
        end
        baselineAgg = aggregateSummary(baselineRaw, ...
            {'mission','measurement_short','period_key','num_observers'}, ...
            {'min_cost','rmse_pos_km','mean_detPpos_km6','mean_stability','runtime_s','screening_events'});
        writetable(baselineRaw, summaryXlsx, "Sheet", "BaselineRaw_J111");
        writetable(baselineAgg, summaryXlsx, "Sheet", "BaselineAgg_J111");
    end
else
    fprintf("No baseline summary found. Baseline cost-only plots will be skipped.\n");
end

%% ========================= FIND COMPARISON WORKBOOKS =========================

xlsxFiles = dir(fullfile(rootDir, "**", "ExperimentSummary*.xlsx"));
if ~isempty(xlsxFiles)
    paths = strings(numel(xlsxFiles),1);
    for i = 1:numel(xlsxFiles)
        paths(i) = string(fullfile(xlsxFiles(i).folder, xlsxFiles(i).name));
    end
    xlsxFiles = xlsxFiles(~contains(paths, "COMPARISON_REPORT_OUTPUT"));
end

if isempty(xlsxFiles)
    error("No ExperimentSummary*.xlsx files found under: %s", rootDir);
end

fprintf("Found %d comparison-study ExperimentSummary files.\n", numel(xlsxFiles));

%% ========================= BUILD RAW SUMMARY TABLE =========================

rawSummary = table();
observerSelections = table();

for k = 1:numel(xlsxFiles)
    xlsxPath = fullfile(xlsxFiles(k).folder, xlsxFiles(k).name);
    runDir = inferRunDirFromExcel(xlsxPath);

    fprintf("[%d/%d] Processing: %s\n", k, numel(xlsxFiles), xlsxPath);

    runInfo = parseRunInfo(runDir, xlsxPath);
    runInfo.runDir = string(runDir);

    try
        Tsum = readtable(xlsxPath, "Sheet", "Summary", "VariableNamingRule", "preserve");
    catch
        try
            Tsum = readtable(xlsxPath, "Sheet", 1, "VariableNamingRule", "preserve");
        catch ME
            warning("Could not read summary sheet for %s\n%s", xlsxPath, ME.message);
            continue;
        end
    end

    if isempty(Tsum)
        continue;
    end

    S = Tsum(1,:);
    runInfo = fillRunInfoFromSummary(runInfo, S);

    row = table();
    row.run_name = string(runInfo.runName);
    row.run_folder = string(runDir);
    row.excel_file = string(xlsxPath);
    row.optimizer = upper(string(getVarValue(S, ["optimizer","Optimizer","solver","Solver"])));
    if row.optimizer == "" || ismissing(row.optimizer)
        row.optimizer = upper(string(runInfo.optimizer));
    end
    row.measurement_short = string(runInfo.measurementShort);
    row.measurement_model = string(runInfo.measurementModel);
    row.mission = string(runInfo.mission);
    row.num_observers = runInfo.numObservers;
    row.periods = runInfo.periods;
    row.seed = getNumericVarValue(S, ["seed","Seed"]);
    row.screening_flag = getNumericVarValue(S, ["use_screening","screening_flag","Screening"]);
    row.use_J1 = getNumericVarValue(S, ["use_J1","J1","cost_J1"]);
    row.use_J2 = getNumericVarValue(S, ["use_J2","J2","cost_J2"]);
    row.use_J3 = getNumericVarValue(S, ["use_J3","J3","cost_J3"]);
    row.min_cost = getNumericVarValue(S, ["min_cost","J_total","cost","best_cost"]);
    row.runtime_s = getNumericVarValue(S, ["runtime_s","runtime","Runtime","Runtime_s"]);
    row.rmse_pos_km = getNumericVarValue(S, ["rmse_pos_km","RMSE_pos_km","rmse"]);
    row.rmse_vel_kms = getNumericVarValue(S, ["rmse_vel_kms","RMSE_vel_kms"]);
    row.mean_detPpos_km6 = getNumericVarValue(S, ["mean_detPpos_km6","mean_detPpos","detP","det"]);
    row.mean_stability = getNumericVarValue(S, ["mean_stability","stability","mean_stab"]);
    row.screening_events = getNumericVarValue(S, ["screeningCount_final","screening_events","ScreeningEvents"]);
    row.selected_families_text = string(getVarValue(S, ["selected_families","selectedFamilies","orbit_families","family_names","best_family_names","selected_orbit_families"]));

    rawSummary = appendTableUnion(rawSummary, row);

    obsRows = readObserverSelectionsFromExcel(xlsxPath, row, "comparison");
    observerSelections = appendTableUnion(observerSelections, obsRows);
end

if isempty(rawSummary)
    error("No valid comparison-study summary rows were extracted.");
end

rawSummary = standardizeSummaryColumns(rawSummary);
rawSummary.cost_combo = makeCostCombo(rawSummary.use_J1, rawSummary.use_J2, rawSummary.use_J3);
rawSummary.all_costs_on = sameNum(rawSummary.use_J1,1) & sameNum(rawSummary.use_J2,1) & sameNum(rawSummary.use_J3,1);
rawSummary = rawSummary(ismember(rawSummary.optimizer, cfg.optimizerOrder), :);

coreSummary = rawSummary(rawSummary.all_costs_on, :);

writetable(rawSummary, summaryXlsx, "Sheet", "Raw_All");
writetable(coreSummary, summaryXlsx, "Sheet", "Raw_J111");
if ~isempty(observerSelections)
    observerSelections = standardizeSummaryColumns(observerSelections);
    writetable(observerSelections, summaryXlsx, "Sheet", "ObserverSelections");
end

%% ========================= AGGREGATED TABLES =========================

aggJ111 = aggregateSummary(coreSummary, ...
    {'mission','measurement_short','measurement_model','period_key','screening_flag','optimizer','num_observers'}, ...
    {'min_cost','runtime_s','rmse_pos_km','mean_detPpos_km6','mean_stability','screening_events'});

writetable(aggJ111, summaryXlsx, "Sheet", "Agg_J111");

costAgg = aggregateSummary(rawSummary, ...
    {'mission','measurement_short','measurement_model','period_key','screening_flag','optimizer','cost_combo','num_observers'}, ...
    {'min_cost','runtime_s','rmse_pos_km','mean_detPpos_km6','mean_stability','screening_events'});

writetable(costAgg, summaryXlsx, "Sheet", "Agg_AllCostCombos");

costContributionSummary = computeCostContributionSummary(costAgg);
if ~isempty(costContributionSummary)
    writetable(costContributionSummary, summaryXlsx, "Sheet", "CostContribSummary");
end

costComboTable = costAgg;
if ismember("screening_flag", string(costComboTable.Properties.VariableNames))
    costComboTable = costComboTable(sameNum(costComboTable.screening_flag, cfg.paperScreeningFlag), :);
end
if ismember("num_observers", string(costComboTable.Properties.VariableNames))
    costComboTable = costComboTable(sameNum(costComboTable.num_observers, cfg.paperObserverCount), :);
end
writetable(costComboTable, summaryXlsx, "Sheet", "CostComboTable_3obs");

%% ========================= PLOT LIST =========================

maneuverSpecs = [ ...
    makeSimpleCase("lg", "Lunar Gateway", cfg.displayMeasurementShort, cfg.lgPeriods), ...
    makeSimpleCase("lt", "Low-thrust Transfer", cfg.displayMeasurementShort, NaN) ...
];

trajIndex = table();
for i = 1:numel(maneuverSpecs)
    rows = exportTrajectoryGridPanels(coreSummary, maneuverSpecs(i), cfg, trajDir);
    trajIndex = appendTableUnion(trajIndex, rows);
end
if ~isempty(trajIndex)
    writetable(trajIndex, summaryXlsx, "Sheet", "TrajectoryPanels");
end

screenIndex = table();
for i = 1:numel(maneuverSpecs)
    row = makeScreeningBarFigure(coreSummary, maneuverSpecs(i), cfg, screenDir);
    screenIndex = appendTableUnion(screenIndex, row);
end
if ~isempty(screenIndex)
    writetable(screenIndex, summaryXlsx, "Sheet", "ScreeningPlots");
end

metricIndex = table();
for i = 1:numel(maneuverSpecs)
    rows = exportMetric2x2Panels(aggJ111, maneuverSpecs(i), cfg, metricDir);
    metricIndex = appendTableUnion(metricIndex, rows);
end
if ~isempty(metricIndex)
    writetable(metricIndex, summaryXlsx, "Sheet", "Metric2x2Panels");
end

costBarIndex = table();
for i = 1:numel(maneuverSpecs)
    row = makeCostContributionBarFigure(costContributionSummary, maneuverSpecs(i), cfg, costBarDir);
    costBarIndex = appendTableUnion(costBarIndex, row);
end
if ~isempty(costBarIndex)
    writetable(costBarIndex, summaryXlsx, "Sheet", "CostContributionBars");
end

baselineIndex = table();
if ~isempty(baselineAgg)
    for i = 1:numel(maneuverSpecs)
        row = makeBaselineCostOnlyFigure(aggJ111, baselineAgg, maneuverSpecs(i), cfg, baseDir);
        baselineIndex = appendTableUnion(baselineIndex, row);
    end
    if ~isempty(baselineIndex)
        writetable(baselineIndex, summaryXlsx, "Sheet", "BaselineCostOnly");
    end
end

familyLong = observerSelections;
if ~isempty(familyLong)
    writetable(familyLong, summaryXlsx, "Sheet", "OrbitFamilies_Comparison");
    familyIndex = makeOrbitFamilyPlots(familyLong, cfg, familyDir, "comparison");
    if ~isempty(familyIndex)
        writetable(familyIndex, summaryXlsx, "Sheet", "OrbitFamilyPlots_Comp");
    end
else
    fprintf("No comparison orbit-family selections found. Orbit-family comparison plots skipped.\n");
end

baselineFamilyLong = table();
if ~isempty(baselineRaw)
    baselineFamilyLong = collectObserverSelectionsFromExcelFiles(baselineRaw, "baseline");
    if ~isempty(baselineFamilyLong)
        writetable(baselineFamilyLong, summaryXlsx, "Sheet", "OrbitFamilies_Baseline");
        baselineFamilyIndex = makeOrbitFamilyPlots(baselineFamilyLong, cfg, familyDir, "baseline");
        if ~isempty(baselineFamilyIndex)
            writetable(baselineFamilyIndex, summaryXlsx, "Sheet", "OrbitFamilyPlots_Base");
        end
    else
        fprintf("No baseline orbit-family selections found. Baseline orbit-family plots skipped.\n");
    end
end

fprintf("\nDone.\nSummary workbook:\n%s\n", summaryXlsx);
fprintf("Trajectory panels:       %s\n", trajDir);
fprintf("Screening charts:        %s\n", screenDir);
fprintf("2x2 metric panels:       %s\n", metricDir);
fprintf("Cost breakdown bars:     %s\n", costBarDir);
fprintf("Baseline cost only:      %s\n", baseDir);
fprintf("Orbit-family plots:      %s\n", familyDir);

%% ========================================================================
%% LOCAL FUNCTIONS
%% ========================================================================

function S = makeSimpleCase(mission, label, measurementShort, periods)
S = struct();
S.mission = lower(string(mission));
S.label = string(label);
S.measurement_short = lower(string(measurementShort));
S.periods = periods;
end

function folderName = getCurrentFolderName()
[~, folderName] = fileparts(pwd);
end

function makeDir(d)
if ~exist(d, 'dir')
    mkdir(d);
end
end

function runDir = inferRunDirFromExcel(xlsxPath)
[dataDir, ~, ~] = fileparts(xlsxPath);
[parentDir, dataFolderName] = fileparts(dataDir);
if strcmpi(dataFolderName, "data")
    runDir = parentDir;
else
    runDir = dataDir;
end
end

function T = addBaselineRunFoldersIfPossible(T, baselineSummaryXlsx)
if isempty(T)
    return;
end
if ismember("run_folder", string(T.Properties.VariableNames))
    return;
end
[tabDir, ~, ~] = fileparts(baselineSummaryXlsx);
searchRoot = fileparts(fileparts(tabDir));
if ~isfolder(searchRoot)
    searchRoot = fileparts(tabDir);
end
T.run_folder = strings(height(T),1);
for i = 1:height(T)
    runName = "";
    if ismember("run_name", string(T.Properties.VariableNames))
        runName = string(T.run_name(i));
    elseif ismember("run", string(T.Properties.VariableNames))
        runName = string(T.run(i));
    end
    if strlength(runName) > 0
        hits = dir(fullfile(searchRoot, "**", char(runName)));
        hits = hits([hits.isdir]);
        if ~isempty(hits)
            T.run_folder(i) = string(fullfile(hits(1).folder, hits(1).name));
        end
    end
end
end

function T = standardizeSummaryColumns(T)
if isempty(T)
    return;
end
if ismember('optimizer', T.Properties.VariableNames)
    T.optimizer = upper(string(T.optimizer));
end
if ismember('measurement_short', T.Properties.VariableNames)
    T.measurement_short = lower(string(T.measurement_short));
end
if ismember('measurement_model', T.Properties.VariableNames)
    T.measurement_model = upper(string(T.measurement_model));
end
if ismember('mission', T.Properties.VariableNames)
    T.mission = lower(string(T.mission));
end
if ismember('periods', T.Properties.VariableNames)
    T.period_key = makePeriodKey(T.periods);
end
end

function periodKey = makePeriodKey(periods)
periods = double(periods);
periodKey = strings(numel(periods),1);
for i = 1:numel(periods)
    if isfinite(periods(i))
        periodKey(i) = "p" + string(round(periods(i)));
    else
        periodKey(i) = "pNaN";
    end
end
end

function runInfo = parseRunInfo(runDir, xlsxPath)
txt = lower(string(runDir) + " " + string(xlsxPath));
[~, runName] = fileparts(runDir);
runInfo = struct();
runInfo.runName = string(runName);
runInfo.optimizer = "unknown";
optNames = ["GA","PSO","BAYESIAN","ABC","ACO","GAMULTIOBJ","DMOPSO"];
for i = 1:numel(optNames)
    if contains(upper(txt), optNames(i))
        runInfo.optimizer = optNames(i);
        break;
    end
end
if contains(txt, filesep + "ao" + filesep) || contains(txt, "_ao_") || contains(txt, "angles_only") || contains(txt, "anglesonly")
    runInfo.measurementShort = "ao";
    runInfo.measurementModel = "ANGLES_ONLY";
elseif contains(txt, filesep + "ar" + filesep) || contains(txt, "_ar_") || contains(txt, "angles_range") || contains(txt, "anglesrange")
    runInfo.measurementShort = "ar";
    runInfo.measurementModel = "ANGLES_RANGE";
else
    runInfo.measurementShort = "unknown";
    runInfo.measurementModel = "unknown";
end
if contains(txt, filesep + "lg" + filesep) || contains(txt, "_lg_") || contains(txt, "lunar_gateway") || contains(txt, "gateway")
    runInfo.mission = "lg";
elseif contains(txt, filesep + "lt" + filesep) || contains(txt, "_lt_") || contains(txt, "low_thrust") || contains(txt, "lowthrust") || contains(txt, "transfer")
    runInfo.mission = "lt";
else
    runInfo.mission = "unknown";
end
tok = regexp(txt, "_o(\d+)", "tokens", "once");
if isempty(tok)
    tok = regexp(txt, "o(\d+)", "tokens", "once");
end
if ~isempty(tok)
    runInfo.numObservers = str2double(tok{1});
else
    runInfo.numObservers = NaN;
end
tokP = regexp(txt, "_p(\d+)", "tokens", "once");
if isempty(tokP)
    tokP = regexp(txt, "p(\d+)", "tokens", "once");
end
if ~isempty(tokP)
    runInfo.periods = str2double(tokP{1});
else
    runInfo.periods = NaN;
end
end

function runInfo = fillRunInfoFromSummary(runInfo, S)
opt = string(getVarValue(S, ["optimizer","Optimizer","solver","Solver"]));
if strlength(opt) > 0 && opt ~= "<missing>"
    runInfo.optimizer = upper(opt);
end
meas = string(getVarValue(S, ["measurement_model","meas_model","MEAS_MODEL"]));
if runInfo.measurementModel == "unknown" && strlength(meas) > 0 && meas ~= "<missing>"
    meas = upper(meas);
    if contains(meas, "ANGLES_ONLY")
        runInfo.measurementModel = "ANGLES_ONLY";
        runInfo.measurementShort = "ao";
    end
    if contains(meas, "ANGLES_RANGE")
        runInfo.measurementModel = "ANGLES_RANGE";
        runInfo.measurementShort = "ar";
    end
end
missionVal = string(getVarValue(S, ["mission","MISSION_TYPE","mission_type"]));
if runInfo.mission == "unknown" && strlength(missionVal) > 0 && missionVal ~= "<missing>"
    mv = upper(missionVal);
    if contains(mv, "LUNAR_GATEWAY")
        runInfo.mission = "lg";
    end
    if contains(mv, "LOW_THRUST") || contains(mv, "TRANSFER")
        runInfo.mission = "lt";
    end
end
if isnan(runInfo.numObservers)
    n = getNumericVarValue(S, ["num_observers","numObservers","observers"]);
    if isfinite(n)
        runInfo.numObservers = n;
    end
end
if isnan(runInfo.periods)
    p = getNumericVarValue(S, ["periods","Nperiods","nperiods"]);
    if isfinite(p)
        runInfo.periods = p;
    elseif runInfo.mission == "lt"
        runInfo.periods = NaN;
    end
end
end

function val = getVarValue(T, candidateNames)
val = missing;
vars = string(T.Properties.VariableNames);
for i = 1:numel(candidateNames)
    idx = strcmpi(vars, candidateNames(i));
    if any(idx)
        tmp = T.(vars(find(idx,1)));
        if iscell(tmp)
            tmp = tmp{1};
        end
        val = tmp;
        return;
    end
end
end

function val = getNumericVarValue(T, candidateNames)
val = NaN;
vars = string(T.Properties.VariableNames);
for i = 1:numel(candidateNames)
    idx = strcmpi(vars, candidateNames(i));
    if any(idx)
        tmp = T.(vars(find(idx,1)));
        if iscell(tmp)
            tmp = tmp{1};
        end
        if isstring(tmp) || ischar(tmp)
            tmp = str2double(tmp);
        end
        if isnumeric(tmp) || islogical(tmp)
            val = double(tmp(1));
        else
            val = NaN;
        end
        return;
    end
end
end

function combo = makeCostCombo(J1,J2,J3)
J1 = double(J1);
J2 = double(J2);
J3 = double(J3);
combo = strings(numel(J1),1);
for i = 1:numel(J1)
    a = ternary(isfinite(J1(i)) && J1(i) ~= 0, "1", "0");
    b = ternary(isfinite(J2(i)) && J2(i) ~= 0, "1", "0");
    c = ternary(isfinite(J3(i)) && J3(i) ~= 0, "1", "0");
    combo(i) = "J" + a + b + c;
end
end

function out = ternary(cond, a, b)
if cond
    out = a;
else
    out = b;
end
end

function Tout = appendTableUnion(Tout,Tin)
if isempty(Tin)
    return;
end
if isempty(Tout)
    Tout = Tin;
    return;
end
varsOut = string(Tout.Properties.VariableNames);
varsIn = string(Tin.Properties.VariableNames);
allVars = unique([varsOut varsIn], "stable");
for i = 1:numel(allVars)
    v = allVars(i);
    if ~ismember(v, varsOut)
        Tout.(v) = makeMissingColumn(height(Tout), Tin.(v));
    end
    if ~ismember(v, varsIn)
        Tin.(v) = makeMissingColumn(height(Tin), Tout.(v));
    end
end
Tout = Tout(:, cellstr(allVars));
Tin = Tin(:, cellstr(allVars));
Tout = [Tout; Tin];
end

function col = makeMissingColumn(n, exampleCol)
if isstring(exampleCol)
    col = strings(n,1);
    col(:) = missing;
elseif iscell(exampleCol)
    col = cell(n,1);
    col(:) = {[]};
elseif iscategorical(exampleCol)
    col = categorical(strings(n,1));
    col(:) = categorical(missing);
elseif isdatetime(exampleCol)
    col = NaT(n,1);
elseif isduration(exampleCol)
    col = seconds(nan(n,1));
elseif islogical(exampleCol)
    col = false(n,1);
elseif isnumeric(exampleCol)
    col = nan(n, size(exampleCol,2));
else
    col = strings(n,1);
    col(:) = missing;
end
end

function Agg = aggregateSummary(T, groupVars, metricVars)
if isempty(T)
    Agg = table();
    return;
end
groupVars = groupVars(ismember(groupVars, T.Properties.VariableNames));
metricVars = metricVars(ismember(metricVars, T.Properties.VariableNames));
if isempty(groupVars) || isempty(metricVars)
    Agg = table();
    return;
end
[G, groupTable] = findgroups(T(:, groupVars));
Agg = groupTable;
Agg.n_runs = splitapply(@numel, ones(height(T),1), G);
for i = 1:numel(metricVars)
    v = metricVars{i};
    x = double(T.(v));
    Agg.([v '_mean']) = splitapply(@safeMean, x, G);
    Agg.([v '_std']) = splitapply(@safeStd, x, G);
end
end

function m = safeMean(x)
x = double(x);
if all(~isfinite(x))
    m = NaN;
else
    m = mean(x, "omitnan");
end
end

function s = safeStd(x)
x = double(x);
if sum(isfinite(x)) <= 1
    s = 0;
else
    s = std(x, "omitnan");
end
end

function tf = sameNum(a,b)
a = double(a);
b = double(b);
tf = isfinite(a) & isfinite(b) & abs(a-b) < 1e-12;
end

function baselineSummaryXlsx = findBaselineSummaryXlsx(rootDir)
baselineSummaryXlsx = "";
searchRoot = fileparts(rootDir);
hits = dir(fullfile(searchRoot, "**", "Baseline_Consolidated_Summary*.xlsx"));
if isempty(hits)
    return;
end
[~,idx] = max([hits.datenum]);
baselineSummaryXlsx = string(fullfile(hits(idx).folder, hits(idx).name));
end

%% ========================= TRAJECTORY PANELS =========================

function figIndex = exportTrajectoryGridPanels(coreSummary, caseSpec, cfg, outDir)
figIndex = table();
T = coreSummary;
T = T(T.mission == caseSpec.mission,:);
T = T(T.measurement_short == caseSpec.measurement_short,:);
T = T(sameNum(T.screening_flag, cfg.paperScreeningFlag),:);
if caseSpec.mission == "lg"
    T = T(sameNum(T.periods, caseSpec.periods),:);
end
T = T(ismember(T.optimizer, cfg.optimizerOrder),:);
T = T(ismember(double(T.num_observers), cfg.observerOrder),:);

if isempty(T)
    warning("No trajectory panels found for %s.", caseSpec.label);
    return;
end

caseDir = fullfile(outDir, char(caseSpec.mission + "_" + caseSpec.measurement_short));
makeDir(caseDir);

for r = 1:numel(cfg.optimizerOrder)
    for c = 1:numel(cfg.observerOrder)
        optName = cfg.optimizerOrder(r);
        nObs = cfg.observerOrder(c);
        Ts = T(T.optimizer == optName & sameNum(T.num_observers, nObs), :);
        if isempty(Ts)
            continue;
        end

        [bestVal, idx] = min(Ts.min_cost, [], "omitnan");
        if isempty(idx) || ~isfinite(bestVal)
            continue;
        end

        figPath = findTrajectoryFigForRun(char(Ts.run_folder(idx)));
        if strlength(figPath) == 0
            warning("Could not find trajectory .fig for %s, %s, %d observers.", caseSpec.label, optName, nObs);
            continue;
        end

        fileStem = "traj_" + caseSpec.mission + "_" + caseSpec.measurement_short + "_" + optName + "_o" + string(nObs);
        if caseSpec.mission == "lg"
            fileStem = fileStem + "_p" + string(caseSpec.periods);
        end
        outPath = fullfile(caseDir, fileStem + ".eps");
        exportTrajectoryPanelFromFig(figPath, outPath, cfg);

        row = table(string(fileStem+".eps"), string(outPath), string(caseSpec.label), optName, nObs, ...
            'VariableNames', {'figure_name','figure_path','case_label','optimizer','num_observers'});
        figIndex = appendTableUnion(figIndex, row);
    end
end
end

function figPath = findTrajectoryFigForRun(runDir)
figPath = "";
hits = dir(fullfile(runDir, "figs", "fig_traj3d*.fig"));
if isempty(hits)
    hits = dir(fullfile(runDir, "**", "fig_traj3d*.fig"));
end
if ~isempty(hits)
    figPath = string(fullfile(hits(1).folder, hits(1).name));
end
end

function exportTrajectoryPanelFromFig(figPath, outPath, cfg)
fig = openfig(figPath, "invisible");

set(fig, "Color", "w");
set(fig, "Units", "inches");
set(fig, "Position", [1 1 cfg.singleFigWidthIn cfg.singleFigHeightIn]);
set(fig, "Renderer", "opengl");

applyFigureStyleForTrajectory(fig, cfg);
setLegendVisibility(fig, "off");

if isfield(cfg, "removeTitles") && cfg.removeTitles
    removeAllTitles(fig);
end

drawnow;
exportFigureAsImageEPS(fig, outPath, cfg);
close(fig);
end

function applyFigureStyleForTrajectory(fig, cfg)
ax = findall(fig, "Type", "axes");

for i = 1:numel(ax)
    if strcmpi(ax(i).Tag, "legend")
        continue;
    end

    try
        set(ax(i), ...
            "FontName", cfg.fontName, ...
            "FontSize", cfg.trajAxisFontSize, ...
            "FontWeight", cfg.fontWeight, ...
            "LineWidth", cfg.axisLineWidth);

        ax(i).XLabel.FontSize = cfg.trajLabelFontSize;
        ax(i).YLabel.FontSize = cfg.trajLabelFontSize;
        ax(i).ZLabel.FontSize = cfg.trajLabelFontSize;

        ax(i).XLabel.FontWeight = cfg.fontWeight;
        ax(i).YLabel.FontWeight = cfg.fontWeight;
        ax(i).ZLabel.FontWeight = cfg.fontWeight;

        title(ax(i), "");

        % Dense trajectory grids are too small for per-panel axis text.
        % Remove labels and tick labels so the plotted orbits use the space.
        xlabel(ax(i), "");
        ylabel(ax(i), "");
        zlabel(ax(i), "");
        ax(i).XTickLabel = [];
        ax(i).YTickLabel = [];
        ax(i).ZTickLabel = [];
        ax(i).TickLength = [0.01 0.01];

        grid(ax(i), "on");
        box(ax(i), "on");
    catch
    end
end

lines = findall(fig, "Type", "line");
for i = 1:numel(lines)
    try
        lines(i).LineWidth = cfg.trajLineWidth;
    catch
    end
end

scat = findall(fig, "Type", "scatter");
for i = 1:numel(scat)
    try
        scat(i).SizeData = max(scat(i).SizeData, cfg.trajMarkerSize);
    catch
    end
end
end

function setLegendVisibility(fig, visibilityState)
lgd = findall(fig, "Type", "legend");
for i = 1:numel(lgd)
    try
        lgd(i).Visible = visibilityState;
    catch
    end
end
end

%% ========================= SCREENING BAR CHARTS =========================

function row = makeScreeningBarFigure(coreSummary, caseSpec, cfg, outDir)
row = table();
T = coreSummary;
T = T(T.mission == caseSpec.mission,:);
T = T(T.measurement_short == caseSpec.measurement_short,:);
if caseSpec.mission == "lg"
    T = T(sameNum(T.periods, caseSpec.periods),:);
end

if isempty(T)
    warning("No screening data found for %s.", caseSpec.label);
    return;
end

Agg = aggregateSummary(T, {'optimizer','screening_flag'}, {'screening_events'});
optList = cfg.optimizerOrder(ismember(cfg.optimizerOrder, unique(Agg.optimizer)));
if isempty(optList)
    optList = unique(Agg.optimizer, 'stable');
end

yOff = nan(numel(optList),1);
yOn = nan(numel(optList),1);
for i = 1:numel(optList)
    idx0 = find(Agg.optimizer == optList(i) & sameNum(Agg.screening_flag,0), 1);
    idx1 = find(Agg.optimizer == optList(i) & sameNum(Agg.screening_flag,1), 1);
    if ~isempty(idx0)
        yOff(i) = Agg.screening_events_mean(idx0);
    end
    if ~isempty(idx1)
        yOn(i) = Agg.screening_events_mean(idx1);
    end
end

fig = figure("Color","w","Units","inches","Position",[1 1 8.5 5.5]);
ax = axes(fig);
hold(ax,'on');
grid(ax,'on');
box(ax,'on');
bar(ax, categorical(optList,optList), [yOff yOn], 'grouped');
setCommonAxes(ax, cfg);
lblArgs = commonLabelArgs(cfg);
xlabel(ax, "Optimizer", lblArgs{:});
ylabel(ax, "Average screening events", lblArgs{:});
ttlArgs = commonTitleArgs(cfg);
title(ax, caseSpec.label + " (" + upper(caseSpec.measurement_short) + ")", ttlArgs{:});
applyConsistentLegend(ax, {"Screening OFF","Screening ON"}, cfg);

fileStem = "screening_bar_" + caseSpec.mission + "_" + caseSpec.measurement_short;
outPath = fullfile(outDir, fileStem + ".eps");
exportFigure(fig, outPath, cfg);
close(fig);

row = table(string(fileStem+".eps"), string(outPath), string(caseSpec.label), ...
    'VariableNames', {'figure_name','figure_path','case_label'});
end

%% ========================= METRIC 2x2 PANELS =========================

function figIndex = exportMetric2x2Panels(aggJ111, caseSpec, cfg, outDir)
figIndex = table();
T = aggJ111;
T = T(T.mission == caseSpec.mission,:);
T = T(sameNum(T.screening_flag, cfg.paperScreeningFlag),:);
T = T(sameNum(T.num_observers, cfg.paperObserverCount),:);
if caseSpec.mission == "lg"
    T = T(T.period_key == "p" + string(caseSpec.periods), :);
end
T = T(ismember(T.measurement_short, cfg.measurementShortOrder), :);

if isempty(T)
    warning("No 2x2 metric panel data found for %s.", caseSpec.label);
    return;
end

metricDefs = { ...
    'min_cost_mean', 'cost', 'Total cost'; ...
    'rmse_pos_km_mean', 'rmse', 'RMSE position (km)'; ...
    'runtime_s_mean', 'runtime', 'Runtime (s)'; ...
    'screening_events_mean', 'screening', 'Average screening events'};

caseDir = fullfile(outDir, char(caseSpec.mission));
makeDir(caseDir);

for m = 1:size(metricDefs,1)
    metricVar = metricDefs{m,1};
    stem = metricDefs{m,2};
    yLabel = metricDefs{m,3};
    fileStem = "metric_panel_" + caseSpec.mission + "_" + stem;
    outPath = fullfile(caseDir, fileStem + ".eps");
    exportGroupedMetricPanel(T, cfg, metricVar, yLabel, outPath);
    row = table(string(fileStem+".eps"), string(outPath), string(caseSpec.label), string(metricVar), ...
        'VariableNames', {'figure_name','figure_path','case_label','metric'});
    figIndex = appendTableUnion(figIndex, row);
end
end

function exportGroupedMetricPanel(T, cfg, metricVar, yLabelText, outPath)
fig = figure("Color","w","Units","inches","Position",[1 1 cfg.metricPanelWidthIn cfg.metricPanelHeightIn]);
ax = axes(fig);
hold(ax,'on');
grid(ax,'on');
box(ax,'on');
plotGroupedBarsByMeasurement(ax, T, cfg, metricVar, yLabelText);
exportFigure(fig, outPath, cfg);
close(fig);
end

function barHandles = plotGroupedBarsByMeasurement(ax, T, cfg, metricVar, yLabelText)
if ~ismember(metricVar, T.Properties.VariableNames)
    barHandles = gobjects(0);
    text(ax, 0.5, 0.5, 'Metric unavailable', 'HorizontalAlignment','center');
    axis(ax,'off');
    return;
end
optList = cfg.optimizerOrder(ismember(cfg.optimizerOrder, unique(T.optimizer)));
if isempty(optList)
    optList = unique(T.optimizer, 'stable');
end
measList = cfg.measurementShortOrder(ismember(cfg.measurementShortOrder, unique(T.measurement_short)));
Y = nan(numel(optList), numel(measList));
for i = 1:numel(optList)
    for j = 1:numel(measList)
        idx = find(T.optimizer == optList(i) & T.measurement_short == measList(j), 1);
        if ~isempty(idx)
            Y(i,j) = T.(metricVar)(idx);
        end
    end
end
barHandles = bar(ax, categorical(optList,optList), Y, 'grouped');
setCommonAxes(ax, cfg);
lblArgs = commonLabelArgs(cfg);
xlabel(ax, 'Optimizer', lblArgs{:});
ylabel(ax, yLabelText, lblArgs{:});
if nargout == 0
    applyConsistentLegend(ax, upper(cellstr(measList)), cfg);
end
end

%% ========================= COST CONTRIBUTION =========================

function Tcontrib = computeCostContributionSummary(costAgg)
Tcontrib = table();
if isempty(costAgg)
    return;
end
groupVars = {'mission','measurement_short','measurement_model','period_key','screening_flag','optimizer','num_observers'};
groupVars = groupVars(ismember(groupVars, costAgg.Properties.VariableNames));
[G, groupTable] = findgroups(costAgg(:,groupVars));
for i = 1:height(groupTable)
    Ts = costAgg(G==i,:);
    V = getCostComboMap(Ts);
    [phi1, phi2, phi3] = estimateShapleyContributions(V);
    row = groupTable(i,:);
    row.contrib_J1 = phi1;
    row.contrib_J2 = phi2;
    row.contrib_J3 = phi3;
    Tcontrib = appendTableUnion(Tcontrib, row);
end
end

function M = getCostComboMap(Ts)
combos = ["J111","J110","J101","J011","J100","J010","J001"];
values = nan(numel(combos),1);
for i = 1:numel(combos)
    idx = find(Ts.cost_combo == combos(i), 1);
    if ~isempty(idx) && ismember('min_cost_mean', Ts.Properties.VariableNames)
        values(i) = Ts.min_cost_mean(idx);
    end
end
M.combos = combos;
M.values = values;
end

function val = getMapValue(M, combo)
idx = find(M.combos == combo, 1);
if isempty(idx)
    val = NaN;
else
    val = M.values(idx);
end
end

function [phi1, phi2, phi3] = estimateShapleyContributions(V)
v000 = 0;
v100 = getMapValue(V, "J100");
v010 = getMapValue(V, "J010");
v001 = getMapValue(V, "J001");
v110 = getMapValue(V, "J110");
v101 = getMapValue(V, "J101");
v011 = getMapValue(V, "J011");
v111 = getMapValue(V, "J111");
phi1 = weightedAverageMarginals([v100-v000, v110-v010, v101-v001, v111-v011], [1/3,1/6,1/6,1/3]);
phi2 = weightedAverageMarginals([v010-v000, v110-v100, v011-v001, v111-v101], [1/3,1/6,1/6,1/3]);
phi3 = weightedAverageMarginals([v001-v000, v101-v100, v011-v010, v111-v110], [1/3,1/6,1/6,1/3]);
end

function out = weightedAverageMarginals(vals, weights)
vals = double(vals(:));
weights = double(weights(:));
idx = isfinite(vals) & isfinite(weights);
if ~any(idx)
    out = NaN;
else
    out = sum(vals(idx).*weights(idx)) / sum(weights(idx));
end
end

function row = makeCostContributionBarFigure(Tcontrib, caseSpec, cfg, outDir)
row = table();
T = Tcontrib;
T = T(T.mission == caseSpec.mission,:);
T = T(T.measurement_short == caseSpec.measurement_short,:);
T = T(sameNum(T.screening_flag, cfg.paperScreeningFlag),:);
T = T(sameNum(T.num_observers, cfg.paperObserverCount),:);
if caseSpec.mission == "lg"
    T = T(T.period_key == "p" + string(caseSpec.periods), :);
end
T = T(ismember(T.optimizer, cfg.optimizerOrder), :);

if isempty(T)
    warning("No cost-contribution data found for %s.", caseSpec.label);
    return;
end

optList = cfg.optimizerOrder(ismember(cfg.optimizerOrder, unique(T.optimizer)));
Y = nan(numel(optList), 3);
for i = 1:numel(optList)
    idx = find(T.optimizer == optList(i), 1);
    if ~isempty(idx)
        Y(i,1) = T.contrib_J1(idx);
        Y(i,2) = T.contrib_J2(idx);
        Y(i,3) = T.contrib_J3(idx);
    end
end

fig = figure("Color","w","Units","inches","Position",[1 1 8.5 5.5]);
ax = axes(fig);
hold(ax,'on');
grid(ax,'on');
box(ax,'on');
bar(ax, categorical(optList,optList), Y, 'grouped');
setCommonAxes(ax, cfg);
lblArgs = commonLabelArgs(cfg);
xlabel(ax, 'Optimizer', lblArgs{:});
ylabel(ax, 'Contribution to total cost', lblArgs{:});
ttlArgs = commonTitleArgs(cfg);
title(ax, sprintf('%s: J111 cost-component breakdown (%s, 3 obs.)', caseSpec.label, upper(caseSpec.measurement_short)), ttlArgs{:});
applyConsistentLegend(ax, {'J1','J2','J3'}, cfg);

fileStem = "cost_breakdown_bar_" + caseSpec.mission;
outPath = fullfile(outDir, fileStem + ".eps");
exportFigure(fig, outPath, cfg);
close(fig);

row = table(string(fileStem+".eps"), string(outPath), string(caseSpec.label), ...
    'VariableNames', {'figure_name','figure_path','case_label'});
end

%% ========================= BASELINE COST ONLY =========================

function row = makeBaselineCostOnlyFigure(aggJ111, baselineAgg, caseSpec, cfg, outDir)
row = table();
T = aggJ111;
T = T(T.mission == caseSpec.mission,:);
T = T(sameNum(T.screening_flag, cfg.paperScreeningFlag),:);
T = T(sameNum(T.num_observers, cfg.paperObserverCount),:);
if caseSpec.mission == "lg"
    T = T(T.period_key == "p" + string(caseSpec.periods), :);
end
T = T(ismember(T.measurement_short, cfg.measurementShortOrder), :);

B = baselineAgg;
B = B(B.mission == caseSpec.mission,:);
B = B(sameNum(B.num_observers, cfg.paperObserverCount),:);
if caseSpec.mission == "lg"
    B = B(B.period_key == "p" + string(caseSpec.periods), :);
end
B = B(ismember(B.measurement_short, cfg.measurementShortOrder), :);

if isempty(T) || isempty(B)
    warning("No baseline cost-only data found for %s.", caseSpec.label);
    return;
end

fig = figure("Color","w","Units","inches","Position",[1 1 8.5 5.5]);
ax = axes(fig);
hold(ax,'on');
grid(ax,'on');
box(ax,'on');
barHandles = plotGroupedBarsByMeasurement(ax, T, cfg, 'min_cost_mean', 'Total cost');

measList = cfg.measurementShortOrder(ismember(cfg.measurementShortOrder, unique(B.measurement_short)));
baseHandles = gobjects(0);
baseLabels = strings(0);
baseStyles = {'--', '-.'};          % AO dashed, AR dash-dot
baseColors = [0 0 0; 0.45 0.45 0.45];

for j = 1:numel(measList)
    idx = find(B.measurement_short == measList(j), 1);

    if ~isempty(idx) && isfinite(B.min_cost_mean(idx))
        h = yline(ax, B.min_cost_mean(idx), baseStyles{j}, ...
            'LineWidth', cfg.lineWidth + 0.4, ...
            'Color', baseColors(j,:), ...
            'DisplayName', "Baseline " + upper(measList(j)));

        baseHandles(end+1) = h;
        baseLabels(end+1) = "Baseline " + upper(measList(j));
    end
end

barLabels = upper(cellstr(cfg.measurementShortOrder));
barLabels = barLabels(1:numel(barHandles));
legendHandles = [barHandles(:); baseHandles(:)];
legendLabels = [string(barLabels(:)); baseLabels(:)];
applyConsistentLegendWithHandles(ax, legendHandles, cellstr(legendLabels), cfg);

ttlArgs = commonTitleArgs(cfg);
title(ax, sprintf('%s: optimizer cost vs baseline', caseSpec.label), ttlArgs{:});

fileStem = "baseline_cost_only_" + caseSpec.mission;
outPath = fullfile(outDir, fileStem + ".eps");
exportFigure(fig, outPath, cfg);
close(fig);

row = table(string(fileStem+".eps"), string(outPath), string(caseSpec.label), ...
    'VariableNames', {'figure_name','figure_path','case_label'});
end

%% ========================= OBSERVER SELECTION SHEET READER =========================

function obsLong = readObserverSelectionsFromExcel(xlsxPath, metaRow, studyLabel)
obsLong = table();

sheets = getWorkbookSheetNames(xlsxPath);
if isempty(sheets)
    return;
end

obsSheets = sheets(endsWith(lower(sheets), "obs"));
if isempty(obsSheets)
    obsSheets = findSheetsWithOrbitFamilyColumn(xlsxPath, sheets);
end

if isempty(obsSheets)
    return;
end

for s = 1:numel(obsSheets)
    try
        Tobs = readtable(xlsxPath, "Sheet", obsSheets(s), "VariableNamingRule", "preserve");
    catch
        continue;
    end

    if isempty(Tobs)
        continue;
    end

    vars = string(Tobs.Properties.VariableNames);
    famVar = findVarName(vars, ["orbit_family","orbit family","family","orbitFamily","family_name"]);
    orbVar = findVarName(vars, ["orbit_index","orbit index","orbit_idx","orbitIdx"]);
    slotVar = findVarName(vars, ["slot_index","slot index","slot_idx","slotIdx"]);
    obsVar = findVarName(vars, ["observer_id","observer id","observer","obs_id"]);
    periodVar = findVarName(vars, ["period_TU","period TU","period","Period (TU)"]);
    stabVar = findVarName(vars, ["stability_index","stability index","Stability index"]);

    if strlength(famVar) == 0
        continue;
    end

    fams = cleanFamilyNames(forceStringVector(Tobs.(famVar)));
    keep = strlength(fams) > 0 & fams ~= "Missing" & fams ~= "Unknown";
    if ~any(keep)
        continue;
    end

    Tobs = Tobs(keep,:);
    fams = fams(keep);

    n = height(Tobs);
    block = table();
    block.study_label = repmat(string(studyLabel), n, 1);
    block.mission = repmat(safeMetaString(metaRow, "mission"), n, 1);
    block.measurement_short = repmat(safeMetaString(metaRow, "measurement_short"), n, 1);
    block.measurement_model = repmat(safeMetaString(metaRow, "measurement_model"), n, 1);
    block.periods = repmat(safeMetaDouble(metaRow, "periods"), n, 1);
    block.period_key = makePeriodKey(block.periods);
    block.optimizer = repmat(safeMetaString(metaRow, "optimizer"), n, 1);
    block.num_observers = repmat(safeMetaDouble(metaRow, "num_observers"), n, 1);
    block.screening_flag = repmat(safeMetaDouble(metaRow, "screening_flag"), n, 1);
    block.cost_combo = repmat(makeCostCombo(safeMetaDouble(metaRow, "use_J1"), safeMetaDouble(metaRow, "use_J2"), safeMetaDouble(metaRow, "use_J3")), n, 1);
    block.family = fams;
    block.excel_file = repmat(string(xlsxPath), n, 1);
    block.observer_id = getOptionalNumericColumn(Tobs, obsVar, n);
    block.orbit_index = getOptionalNumericColumn(Tobs, orbVar, n);
    block.slot_index = getOptionalNumericColumn(Tobs, slotVar, n);
    block.period_TU = getOptionalNumericColumn(Tobs, periodVar, n);
    block.stability_index = getOptionalNumericColumn(Tobs, stabVar, n);

    obsLong = appendTableUnion(obsLong, block);
end
end

function obsLong = collectObserverSelectionsFromExcelFiles(summaryTable, studyLabel)
obsLong = table();
if isempty(summaryTable)
    return;
end
if ~ismember("excel_file", string(summaryTable.Properties.VariableNames))
    return;
end
for i = 1:height(summaryTable)
    xlsxPath = string(summaryTable.excel_file(i));
    if strlength(xlsxPath) == 0 || ~isfile(xlsxPath)
        continue;
    end
    obsRows = readObserverSelectionsFromExcel(xlsxPath, summaryTable(i,:), studyLabel);
    obsLong = appendTableUnion(obsLong, obsRows);
end
end

function sheets = getWorkbookSheetNames(xlsxPath)
sheets = strings(0,1);
try
    sheets = string(sheetnames(xlsxPath));
    sheets = sheets(:);
    return;
catch
end
try
    [~, sh] = xlsfinfo(xlsxPath);
    sheets = string(sh(:));
catch
    sheets = strings(0,1);
end
end

function obsSheets = findSheetsWithOrbitFamilyColumn(xlsxPath, sheets)
obsSheets = strings(0,1);
for i = 1:numel(sheets)
    try
        T = readtable(xlsxPath, "Sheet", sheets(i), "VariableNamingRule", "preserve");
        vars = lower(string(T.Properties.VariableNames));
        if any(vars == "orbit_family") || any(contains(vars, "orbit") & contains(vars, "family"))
            obsSheets(end+1,1) = sheets(i);
        end
    catch
    end
end
end

function name = findVarName(vars, candidates)
name = "";
varsClean = lower(regexprep(string(vars), '\s+', '_'));
candClean = lower(regexprep(string(candidates), '\s+', '_'));
for i = 1:numel(candClean)
    idx = find(varsClean == candClean(i), 1);
    if ~isempty(idx)
        name = string(vars(idx));
        return;
    end
end
for i = 1:numel(candClean)
    idx = find(contains(varsClean, candClean(i)), 1);
    if ~isempty(idx)
        name = string(vars(idx));
        return;
    end
end
end

function val = safeMetaString(T, varName)
if ismember(varName, string(T.Properties.VariableNames))
    val = string(T.(varName)(1));
else
    val = "unknown";
end
end

function val = safeMetaDouble(T, varName)
if ismember(varName, string(T.Properties.VariableNames))
    val = double(T.(varName)(1));
else
    val = NaN;
end
end

function x = getOptionalNumericColumn(T, varName, n)
x = nan(n,1);
if strlength(varName) == 0
    return;
end
try
    tmp = T.(varName);
    if isnumeric(tmp) || islogical(tmp)
        x = double(tmp(:));
    else
        x = str2double(string(tmp(:)));
    end
catch
    x = nan(n,1);
end
end

%% ========================= ORBIT FAMILY SELECTIONS =========================

function familyLong = collectOrbitFamilySelections(summaryTable, studyLabel)
familyLong = table();
if isempty(summaryTable)
    return;
end

for i = 1:height(summaryTable)
    fams = strings(0,1);

    if ismember("selected_families_text", string(summaryTable.Properties.VariableNames))
        fams = parseFamilyText(summaryTable.selected_families_text(i));
    end

    if isempty(fams) && ismember("run_folder", string(summaryTable.Properties.VariableNames))
        runDir = string(summaryTable.run_folder(i));
        if strlength(runDir) > 0 && isfolder(runDir)
            fams = tryLoadSelectedFamilies(runDir);
        end
    end

    if isempty(fams)
        continue;
    end

    for k = 1:numel(fams)
        row = table();
        row.study_label = string(studyLabel);
        row.mission = safeTableString(summaryTable, "mission", i);
        row.measurement_short = safeTableString(summaryTable, "measurement_short", i);
        row.period_key = safeTableString(summaryTable, "period_key", i);
        row.optimizer = safeTableString(summaryTable, "optimizer", i);
        row.num_observers = safeTableDouble(summaryTable, "num_observers", i);
        row.screening_flag = safeTableDouble(summaryTable, "screening_flag", i);
        row.cost_combo = safeTableString(summaryTable, "cost_combo", i);
        row.family = string(fams(k));
        familyLong = appendTableUnion(familyLong, row);
    end
end

if ~isempty(familyLong)
    familyLong.family = cleanFamilyNames(familyLong.family);
    familyLong = familyLong(strlength(familyLong.family) > 0 & familyLong.family ~= "missing", :);
end
end

function val = safeTableString(T, varName, rowIdx)
if ismember(varName, string(T.Properties.VariableNames))
    val = string(T.(varName)(rowIdx));
else
    val = "unknown";
end
end

function val = safeTableDouble(T, varName, rowIdx)
if ismember(varName, string(T.Properties.VariableNames))
    val = double(T.(varName)(rowIdx));
else
    val = NaN;
end
end

function fams = parseFamilyText(txt)
fams = strings(0,1);
if ismissing(txt)
    return;
end
txt = string(txt);
if strlength(txt) == 0 || lower(txt) == "missing"
    return;
end
txt = erase(txt, "[");
txt = erase(txt, "]");
txt = erase(txt, "{");
txt = erase(txt, "}");
txt = erase(txt, "'");
txt = erase(txt, '"');
parts = split(txt, [",",";","|",newline,char(9)]);
parts = strtrim(parts);
parts = parts(strlength(parts) > 0);
fams = parts(:);
end

function fams = tryLoadSelectedFamilies(runDir)
fams = strings(0,1);
matFiles = dir(fullfile(runDir, '**', '*.mat'));
if isempty(matFiles)
    return;
end

fieldNames = [ ...
    "selected_families", ...
    "selectedFamilies", ...
    "selected_orbit_families", ...
    "selectedOrbitFamilies", ...
    "orbit_families", ...
    "orbitFamilies", ...
    "family_names", ...
    "familyNames", ...
    "best_family_names", ...
    "bestFamilyNames", ...
    "selectedFamilyNames"];

for i = 1:numel(matFiles)
    f = fullfile(matFiles(i).folder, matFiles(i).name);
    try
        vars = whos('-file', f);
        vnames = string({vars.name});

        for j = 1:numel(fieldNames)
            if any(vnames == fieldNames(j))
                S = load(f, char(fieldNames(j)));
                fams = forceStringVector(S.(char(fieldNames(j))));
                fams = cleanFamilyNames(fams);
                fams = fams(strlength(fams) > 0);
                if ~isempty(fams)
                    return;
                end
            end
        end

        for j = 1:numel(vnames)
            S = load(f, char(vnames(j)));
            X = S.(char(vnames(j)));
            fams = findFamiliesInsideObject(X, fieldNames);
            fams = cleanFamilyNames(fams);
            fams = fams(strlength(fams) > 0);
            if ~isempty(fams)
                return;
            end
        end
    catch
    end
end
end

function fams = findFamiliesInsideObject(X, fieldNames)
fams = strings(0,1);
if isstruct(X)
    for j = 1:numel(fieldNames)
        if isfield(X, fieldNames(j))
            fams = forceStringVector(X.(fieldNames(j)));
            if ~isempty(fams)
                return;
            end
        end
    end
    flds = string(fieldnames(X));
    for i = 1:numel(flds)
        try
            fams = findFamiliesInsideObject(X.(flds(i)), fieldNames);
            if ~isempty(fams)
                return;
            end
        catch
        end
    end
elseif istable(X)
    vars = string(X.Properties.VariableNames);
    for j = 1:numel(fieldNames)
        idx = strcmpi(vars, fieldNames(j));
        if any(idx)
            fams = forceStringVector(X.(vars(find(idx,1))));
            if ~isempty(fams)
                return;
            end
        end
    end
end
end

function s = forceStringVector(x)
try
    if iscell(x)
        s = string(x(:));
    elseif isstring(x)
        s = x(:);
    elseif ischar(x)
        s = string(x);
    elseif iscategorical(x)
        s = string(x(:));
    elseif isnumeric(x)
        s = string(x(:));
    elseif isstruct(x)
        s = strings(0,1);
    else
        s = string(x(:));
    end
catch
    s = strings(0,1);
end
end

function fams = cleanFamilyNames(fams)
fams = string(fams(:));
fams = strtrim(fams);
fams = replace(fams, "_", " ");
fams = replace(fams, "-", " ");
fams = regexprep(fams, '\s+', ' ');
fams = upper(fams);
end

function figIndex = makeOrbitFamilyPlots(familyLong, cfg, outDir, studyLabel)
figIndex = table();
if isempty(familyLong)
    return;
end

T = familyLong;
T = T(ismember(T.optimizer, cfg.optimizerOrder), :);

% Keep orbit-family plots consistent with the two paper cases used in the
% trajectory/metric figures: LG at cfg.lgPeriods and LT, both using
% cfg.displayMeasurementShort.
if ismember("measurement_short", string(T.Properties.VariableNames))
    T = T(T.measurement_short == cfg.displayMeasurementShort, :);
end
if ismember("mission", string(T.Properties.VariableNames)) && ismember("period_key", string(T.Properties.VariableNames))
    isLGKeep = T.mission == "lg" & T.period_key == "p" + string(cfg.lgPeriods);
    isLTKeep = T.mission == "lt";
    T = T(isLGKeep | isLTKeep, :);
end
if isempty(T)
    return;
end

if ~cfg.orbitFamilyUseAllCostCombos && ismember("cost_combo", string(T.Properties.VariableNames))
    T = T(T.cost_combo == "J111", :);
end
if ~cfg.orbitFamilyUseAllObserverCounts && ismember("num_observers", string(T.Properties.VariableNames))
    T = T(sameNum(T.num_observers, cfg.paperObserverCount), :);
end
if ~cfg.orbitFamilyUseBothScreeningFlags && ismember("screening_flag", string(T.Properties.VariableNames))
    T = T(sameNum(T.screening_flag, cfg.paperScreeningFlag), :);
end

Tlg = T(T.mission == "lg", :);
if ~isempty(Tlg)
    measList = unique(Tlg.measurement_short, 'stable');
    periodList = unique(Tlg.period_key, 'stable');
    for m = 1:numel(measList)
        for p = 1:numel(periodList)
            for o = 1:numel(cfg.optimizerOrder)
                optName = cfg.optimizerOrder(o);
                Ts = Tlg(Tlg.measurement_short == measList(m) & Tlg.period_key == periodList(p) & Tlg.optimizer == optName, :);
                if isempty(Ts)
                    continue;
                end
                caseLabel = sprintf('%s LG | %s | %s | %s', upper(studyLabel), upper(measList(m)), periodList(p), optName);
                fileStem = "orbit_families_" + string(studyLabel) + "_lg_" + lower(measList(m)) + "_" + lower(periodList(p)) + "_" + upper(optName);
                outPath = fullfile(outDir, fileStem + ".eps");
                exportOrbitFamilyBar(Ts, cfg, caseLabel, outPath);
                row = table(string(fileStem+".eps"), string(outPath), string(caseLabel), ...
                    'VariableNames', {'figure_name','figure_path','case_label'});
                figIndex = appendTableUnion(figIndex, row);
            end
        end
    end
end

Tlt = T(T.mission == "lt", :);
if ~isempty(Tlt)
    measList = unique(Tlt.measurement_short, 'stable');
    for m = 1:numel(measList)
        for o = 1:numel(cfg.optimizerOrder)
            optName = cfg.optimizerOrder(o);
            Ts = Tlt(Tlt.measurement_short == measList(m) & Tlt.optimizer == optName, :);
            if isempty(Ts)
                continue;
            end
            caseLabel = sprintf('%s LT | %s | %s', upper(studyLabel), upper(measList(m)), optName);
            fileStem = "orbit_families_" + string(studyLabel) + "_lt_" + lower(measList(m)) + "_" + upper(optName);
            outPath = fullfile(outDir, fileStem + ".eps");
            exportOrbitFamilyBar(Ts, cfg, caseLabel, outPath);
            row = table(string(fileStem+".eps"), string(outPath), string(caseLabel), ...
                'VariableNames', {'figure_name','figure_path','case_label'});
            figIndex = appendTableUnion(figIndex, row);
        end
    end
end
end

function exportOrbitFamilyBar(Ts, cfg, titleText, outPath)
fams = categorical(string(Ts.family));
[G, famNames] = findgroups(fams);
counts = splitapply(@numel, ones(numel(G),1), G);
[famNames, idx] = sort(string(famNames));
counts = counts(idx);

fig = figure("Color","w","Units","inches","Position",[1 1 8.5 5.5]);
ax = axes(fig);
hold(ax,'on');
grid(ax,'on');
box(ax,'on');
bar(ax, categorical(famNames, famNames), counts);
setCommonAxes(ax, cfg);
xtickangle(ax, 35);
lblArgs = commonLabelArgs(cfg);
xlabel(ax, 'Orbit family', lblArgs{:});
ylabel(ax, 'Selection count', lblArgs{:});
ttlArgs = commonTitleArgs(cfg);
title(ax, titleText, ttlArgs{:});
exportFigure(fig, outPath, cfg);
close(fig);
end

%% ========================= EXPORT / STYLE HELPERS =========================

function exportFigure(fig, outPath, cfg)
[folder, name, ext] = fileparts(outPath);
if ~strcmpi(ext, '.eps')
    outPath = fullfile(folder, name + ".eps");
end

if isfield(cfg, "removeTitles") && cfg.removeTitles
    removeAllTitles(fig);
end

set(fig, 'Color', 'w', 'InvertHardcopy', 'off', 'Renderer', 'opengl');
drawnow;
exportFigureAsImageEPS(fig, outPath, cfg);
end

function removeAllTitles(fig)
ax = findall(fig, "Type", "axes");
for i = 1:numel(ax)
    try
        title(ax(i), "");
    catch
    end
end
end

function exportFigureAsImageEPS(fig, outEps, cfg)
try
    exportgraphics(fig, outEps, ...
        'ContentType', 'image', ...
        'Resolution', cfg.epsResolution, ...
        'BackgroundColor', 'white');
catch
    warning('exportgraphics EPS image export failed. Falling back to print -depsc -opengl.');
    print(fig, outEps, '-depsc', '-image', sprintf('-r%d', cfg.epsResolution));
end
end

function setCommonAxes(ax, cfg)
set(ax, 'FontName', cfg.fontName, 'FontSize', cfg.axisFontSize, 'FontWeight', cfg.fontWeight, 'LineWidth', cfg.axisLineWidth);
end

function lgd = applyConsistentLegend(ax, labels, cfg)
lgArgs = commonLegendArgs(cfg);
lgd = legend(ax, labels, 'Location', cfg.legendLocation, 'Orientation', cfg.legendOrientation, lgArgs{:});
lgd.Box = 'on';
lgd.AutoUpdate = 'off';
end

function lgd = applyConsistentLegendWithHandles(ax, handles, labels, cfg)
lgArgs = commonLegendArgs(cfg);
lgd = legend(ax, handles, labels, 'Location', cfg.legendLocation, 'Orientation', cfg.legendOrientation, lgArgs{:});
lgd.Box = 'on';
lgd.AutoUpdate = 'off';
end

function args = commonLabelArgs(cfg)
args = {'FontName', cfg.fontName, 'FontSize', cfg.labelFontSize, 'FontWeight', cfg.fontWeight};
end

function args = commonTitleArgs(cfg)
args = {'FontName', cfg.fontName, 'FontSize', cfg.titleFontSize, 'FontWeight', cfg.fontWeight};
end

function args = commonLegendArgs(cfg)
args = {'FontName', cfg.fontName, 'FontSize', cfg.legendFontSize, 'FontWeight', cfg.fontWeight};
end
