function postprocess_constellation_studies()
% POSTPROCESS_CONSTELLATION_STUDIES
% Cleanly summarizes completed constellation studies, generates JOTA-style
% figures, and writes one Excel workbook with multiple sheets.
%
% What this script does:
%   1) Summarizes and plots observer count vs RMSE, det(P_pos), and stability.
%   2) Determines the best-performing constellations for each model group,
%      along with overall best and overall minima for each metric.
%   3) Summarizes orbit families selected across all studies.
%   4) Prints key tables to the console and writes one Excel workbook.
%
% Notes:
%   - A CSV file cannot contain multiple sheets. This script writes a single
%     Excel workbook (.xlsx) with multiple sheets instead.
%   - The script expects run directories produced by run_opt.m and the batch
%     runner structure shown in the study setup.
%
% Author: ChatGPT

clearvars -except ans;
close all;
clc;

% ---------------- Figure defaults ----------------
set(groot, ...
    'defaultAxesFontSize',16, ...
    'defaultAxesFontWeight','bold', ...
    'defaultAxesFontName','Times New Roman', ...
    'defaultTextFontSize',12, ...
    'defaultTextFontWeight','bold', ...
    'defaultTextFontName','Times New Roman', ...
    'defaultLegendFontSize',11, ...
    'defaultLegendFontWeight','bold', ...
    'defaultAxesLabelFontSizeMultiplier',1.0, ...
    'defaultAxesTitleFontSizeMultiplier',1.0, ...
    'defaultLineLineWidth',1.8);

% ---------------- User options ----------------
thisFile = mfilename('fullpath');
thisDir  = fileparts(thisFile);

% Prefer the current working directory when it looks like the project root.
projectRoot = pwd;
if ~exist(fullfile(projectRoot, 'runs'), 'dir')
    projectRoot = thisDir;
end

% Search all study results under runs by default. Change this if desired.
searchRoot = fullfile(projectRoot, 'runs');

% Output directory for the consolidated post-processing products.
outDir = fullfile(projectRoot, 'postprocess_summary');
figDir = fullfile(outDir, 'figs');
if ~exist(outDir, 'dir'), mkdir(outDir); end
if ~exist(figDir, 'dir'), mkdir(figDir); end

outWorkbook = fullfile(outDir, 'ConstellationStudy_PostProcess.xlsx');

fprintf('POST-PROCESS START: %s\n', string(datetime('now')));
fprintf('Search root: %s\n', searchRoot);
fprintf('Output dir : %s\n\n', outDir);

% ---------------- Discover experiment files ----------------
excelFiles = recursive_dir(searchRoot, 'ExperimentSummary_*.xlsx');
excelFiles = excelFiles(~contains(string(excelFiles), string(outWorkbook)));

if isempty(excelFiles)
    error('No ExperimentSummary_*.xlsx files were found under: %s', searchRoot);
end

fprintf('Found %d ExperimentSummary files.\n', numel(excelFiles));

% ---------------- Read all run summaries and observer tables ----------------
runsCell = {};
obsCell  = {};

for i = 1:numel(excelFiles)
    file = excelFiles{i};
    try
        [runTbl, obsTbl] = read_one_experiment_file(file);
        if ~isempty(runTbl)
            runsCell{end+1,1} = runTbl; 
        end
        if ~isempty(obsTbl)
            obsCell{end+1,1} = obsTbl; 
        end
    catch ME
        warning('Skipping file due to read error:\n  %s\n  %s', file, ME.message);
    end
end

if isempty(runsCell)
    error('No valid run summaries were parsed from the discovered files.');
end

runsTbl = vertcat(runsCell{:});
if isempty(obsCell)
    obsTbl = table();
else
    obsTbl = vertcat(obsCell{:});
end

runsTbl = sortrows(runsTbl, {'mission_type','measurement_model','nperiods','num_observers'});

% ---------------- Build aggregated observer-count summary ----------------
obsCountSummaryTbl = build_observer_count_summary(runsTbl);

% ---------------- Determine best constellations ----------------
bestTbl = build_best_constellation_summary(runsTbl, obsTbl);

% ---------------- Orbit family summary ----------------
[familySummaryTbl, familyByGroupTbl] = build_orbit_family_summary(obsTbl);

% ---------------- Print to console ----------------
print_run_summary_console(runsTbl);
print_observer_count_console(obsCountSummaryTbl);
print_best_constellations_console(bestTbl);
print_orbit_family_console(familySummaryTbl, familyByGroupTbl);

% ---------------- Write workbook ----------------
if isfile(outWorkbook)
    delete(outWorkbook);
end

writetable(runsTbl, outWorkbook, 'Sheet', 'Runs');
writetable(obsCountSummaryTbl, outWorkbook, 'Sheet', 'ObserverCountSummary');
writetable(bestTbl, outWorkbook, 'Sheet', 'BestConstellations');
writetable(familySummaryTbl, outWorkbook, 'Sheet', 'OrbitFamilySummary');
if ~isempty(familyByGroupTbl)
    writetable(familyByGroupTbl, outWorkbook, 'Sheet', 'OrbitFamilyByGroup');
end
if ~isempty(obsTbl)
    writetable(obsTbl, outWorkbook, 'Sheet', 'ObserverSelections');
end

fprintf('\nWrote workbook:\n  %s\n', outWorkbook);

% ---------------- Generate figures ----------------
plot_observer_count_metric_by_mission(obsCountSummaryTbl, figDir, 'rmse_pos_km_mean', ...
    'Position RMSE (km)', 'observer_count_vs_rmse');
plot_observer_count_metric_by_mission(obsCountSummaryTbl, figDir, 'mean_detPpos_km6_mean', ...
    'Mean det(P_{pos}) (km^6)', 'observer_count_vs_detPpos');
plot_observer_count_metric_by_mission(obsCountSummaryTbl, figDir, 'mean_stability_mean', ...
    'Mean stability index', 'observer_count_vs_stability');
plot_orbit_family_bar_by_mission(obsTbl, figDir);

fprintf('\nPOST-PROCESS END: %s\n', string(datetime('now')));
end

% ========================================================================
% Helpers
% ========================================================================

function fileList = recursive_dir(rootDir, pattern)
    d = dir(fullfile(rootDir, '**', pattern));
    fileList = fullfile({d.folder}, {d.name});
    fileList = reshape(fileList, [], 1);
end

function [runTbl, obsTbl] = read_one_experiment_file(file)
    sheetList = sheetnames(file);

    runTbl = table();
    obsTbl = table();

    % ---------------- Summary sheet ----------------
    if any(strcmpi(sheetList, 'Summary'))
        runTbl = readtable(file, 'Sheet', 'Summary', 'TextType', 'string');
    else
        return;
    end

    if isempty(runTbl)
        return;
    end

    % Summary sheet is expected to be one row per file.
    runTbl = runTbl(1,:);

    % ---------------- Path-derived metadata ----------------
    meta = parse_metadata_from_path(file, runTbl);

    % Ensure expected fields exist.
    if ~ismember('measurement_model', runTbl.Properties.VariableNames)
        runTbl.measurement_model = string(missing);
    end
    if ~ismember('optimizer', runTbl.Properties.VariableNames)
        runTbl.optimizer = string(missing);
    end
    if ~ismember('min_cost', runTbl.Properties.VariableNames)
        runTbl.min_cost = NaN;
    end
    if ~ismember('rmse_pos_km', runTbl.Properties.VariableNames)
        runTbl.rmse_pos_km = NaN;
    end
    if ~ismember('mean_detPpos_km6', runTbl.Properties.VariableNames)
        runTbl.mean_detPpos_km6 = NaN;
    end
    if ~ismember('mean_stability', runTbl.Properties.VariableNames)
        runTbl.mean_stability = NaN;
    end
    if ~ismember('runtime_s', runTbl.Properties.VariableNames)
        runTbl.runtime_s = NaN;
    end
    if ~ismember('num_function_evals', runTbl.Properties.VariableNames)
        runTbl.num_function_evals = NaN;
    end

    runTbl.file_path      = string(file);
    runTbl.case_name      = string(meta.case_name);
    runTbl.run_dir        = string(meta.run_dir);
    runTbl.mission_type   = string(meta.mission_type);
    runTbl.measurement_model = string(runTbl.measurement_model);
    runTbl.num_observers  = meta.num_observers;
    runTbl.nperiods       = meta.nperiods;
    runTbl.model_group    = string(meta.model_group);
    runTbl.case_label     = string(meta.case_label);

    % ---------------- Function evaluations from SECOND sheet ----------------
    if numel(sheetList) >= 2
        try
            evalTbl = readtable(file, 'Sheet', sheetList{2}, 'TextType', 'string');
            runTbl.num_function_evals = height(evalTbl);
        catch
            runTbl.num_function_evals = NaN;
        end
    else
        runTbl.num_function_evals = NaN;
    end

    % ---------------- Observer sheet ----------------
    obsSheetName = "";
    for k = 1:numel(sheetList)
        s = sheetList{k};
        if strcmpi(s, 'Summary')
            continue;
        end
        try
            T = readtable(file, 'Sheet', s, 'TextType', 'string');
        catch
            continue;
        end
        names = string(T.Properties.VariableNames);
        needed = ["observer_id","orbit_index","slot_index","orbit_family","period_TU","stability_index"];
        if all(ismember(needed, names))
            obsSheetName = s;
            obsTbl = T;
            break;
        end
    end

    if ~isempty(obsTbl)
        obsTbl.file_path          = repmat(string(file), height(obsTbl), 1);
        obsTbl.case_name          = repmat(string(meta.case_name), height(obsTbl), 1);
        obsTbl.run_dir            = repmat(string(meta.run_dir), height(obsTbl), 1);
        obsTbl.mission_type       = repmat(string(meta.mission_type), height(obsTbl), 1);
        obsTbl.measurement_model  = repmat(string(runTbl.measurement_model), height(obsTbl), 1);
        obsTbl.num_observers      = repmat(meta.num_observers, height(obsTbl), 1);
        obsTbl.nperiods           = repmat(meta.nperiods, height(obsTbl), 1);
        obsTbl.model_group        = repmat(string(meta.model_group), height(obsTbl), 1);
        obsTbl.obs_sheet_name     = repmat(string(obsSheetName), height(obsTbl), 1);
    end
end

function meta = parse_metadata_from_path(file, runTbl)
    % Expected structure example:
    % .../runs/BASELINE/runs_GA/ar/lg/b_ga600_ar_o10_p5/data/ExperimentSummary_....xlsx

    dataDir = fileparts(file);
    caseDir = fileparts(dataDir);
    missionDir = fileparts(caseDir);

    [~, caseName] = fileparts(caseDir);
    [~, missionCode] = fileparts(missionDir);

    meta = struct();
    meta.case_name = caseName;
    meta.run_dir   = caseDir;
    meta.num_observers = parse_token(caseName, '_o(\d+)', NaN);
    meta.nperiods      = parse_token(caseName, '_p(\d+)', 1);

    if isnan(meta.num_observers)
        meta.num_observers = parse_token(string(file), '_o(\d+)', NaN);
    end
    if isnan(meta.nperiods)
        meta.nperiods = 1;
    end

    switch lower(string(missionCode))
        case "lt"
            meta.mission_type = "LOW_THRUST_TRANSFER";
        case "lg"
            meta.mission_type = "LUNAR_GATEWAY";
        case "po"
            meta.mission_type = "PERIODIC_ORBIT";
        case "tt"
            meta.mission_type = "TIME_OPT_TRANSFER";
        case "ft"
            meta.mission_type = "FUEL_OPT_TRANSFER";
        otherwise
            if ismember('run_tag', runTbl.Properties.VariableNames)
                rt = string(runTbl.run_tag(1));
                if contains(rt, 'lt', 'IgnoreCase', true)
                    meta.mission_type = "LOW_THRUST_TRANSFER";
                elseif contains(rt, 'lg', 'IgnoreCase', true)
                    meta.mission_type = "LUNAR_GATEWAY";
                else
                    meta.mission_type = upper(string(missionCode));
                end
            else
                meta.mission_type = upper(string(missionCode));
            end
    end

    meas = "";
    if ismember('measurement_model', runTbl.Properties.VariableNames)
        meas = string(runTbl.measurement_model(1));
    end
    if strlength(meas) == 0 || ismissing(meas)
        if contains(caseName, '_ao_', 'IgnoreCase', true)
            meas = "ANGLES_ONLY";
        elseif contains(caseName, '_ar_', 'IgnoreCase', true)
            meas = "ANGLES_RANGE";
        else
            meas = "UNKNOWN";
        end
    end

    if strcmpi(meta.mission_type, 'LUNAR_GATEWAY') || strcmpi(meta.mission_type, 'PERIODIC_ORBIT')
        meta.model_group = sprintf('%s | %d period(s) | %s', meta.mission_type, meta.nperiods, meas);
    else
        meta.model_group = sprintf('%s | %s', meta.mission_type, meas);
    end

    if strcmpi(meta.mission_type, 'LUNAR_GATEWAY') || strcmpi(meta.mission_type, 'PERIODIC_ORBIT')
        meta.case_label = sprintf('%s | o%d | p%d | %s', meta.mission_type, meta.num_observers, meta.nperiods, meas);
    else
        meta.case_label = sprintf('%s | o%d | %s', meta.mission_type, meta.num_observers, meas);
    end
end

function val = parse_token(txt, expr, defaultVal)
    tok = regexp(char(txt), expr, 'tokens', 'once');
    if isempty(tok)
        val = defaultVal;
    else
        val = str2double(tok{1});
    end
end

function summaryTbl = build_observer_count_summary(runsTbl)
    G = findgroups(runsTbl.model_group, runsTbl.mission_type, runsTbl.measurement_model, runsTbl.nperiods, runsTbl.num_observers);

    summaryTbl = table();
    summaryTbl.model_group = splitapply(@(x) x(1), runsTbl.model_group, G);
    summaryTbl.mission_type = splitapply(@(x) x(1), runsTbl.mission_type, G);
    summaryTbl.measurement_model = splitapply(@(x) x(1), runsTbl.measurement_model, G);
    summaryTbl.nperiods = splitapply(@(x) x(1), runsTbl.nperiods, G);
    summaryTbl.num_observers = splitapply(@(x) x(1), runsTbl.num_observers, G);
    summaryTbl.n_runs = splitapply(@numel, runsTbl.num_observers, G);

    summaryTbl.min_cost_mean = splitapply(@nanmean, runsTbl.min_cost, G);
    summaryTbl.min_cost_std  = splitapply(@nanstd,  runsTbl.min_cost, G);

    summaryTbl.rmse_pos_km_mean = splitapply(@nanmean, runsTbl.rmse_pos_km, G);
    summaryTbl.rmse_pos_km_std  = splitapply(@nanstd,  runsTbl.rmse_pos_km, G);

    summaryTbl.mean_detPpos_km6_mean = splitapply(@nanmean, runsTbl.mean_detPpos_km6, G);
    summaryTbl.mean_detPpos_km6_std  = splitapply(@nanstd,  runsTbl.mean_detPpos_km6, G);

    summaryTbl.mean_stability_mean = splitapply(@nanmean, runsTbl.mean_stability, G);
    summaryTbl.mean_stability_std  = splitapply(@nanstd,  runsTbl.mean_stability, G);

    summaryTbl.runtime_s_mean = splitapply(@nanmean, runsTbl.runtime_s, G);
    summaryTbl.runtime_s_std  = splitapply(@nanstd,  runsTbl.runtime_s, G);

    summaryTbl = sortrows(summaryTbl, {'mission_type','measurement_model','nperiods','num_observers'});
end

function bestTbl = build_best_constellation_summary(runsTbl, obsTbl)
    rows = {};

    % Best by min cost within each model group.
    modelGroups = unique(runsTbl.model_group, 'stable');
    for i = 1:numel(modelGroups)
        g = modelGroups(i);
        idx = runsTbl.model_group == g;
        Tg = runsTbl(idx,:);
        if isempty(Tg)
            continue;
        end

        rows{end+1,1} = make_best_row(Tg, obsTbl, 'GROUP_BEST_MIN_COST', g, 'min_cost'); 
        rows{end+1,1} = make_best_row(Tg, obsTbl, 'GROUP_LOWEST_RMSE',   g, 'rmse_pos_km'); 
        rows{end+1,1} = make_best_row(Tg, obsTbl, 'GROUP_LOWEST_DET',    g, 'mean_detPpos_km6'); 
        rows{end+1,1} = make_best_row(Tg, obsTbl, 'GROUP_LOWEST_STAB',   g, 'mean_stability'); 
    end

    % Overall rows across all studies.
    rows{end+1,1} = make_best_row(runsTbl, obsTbl, 'OVERALL_BEST_MIN_COST', 'ALL', 'min_cost'); 
    rows{end+1,1} = make_best_row(runsTbl, obsTbl, 'OVERALL_LOWEST_RMSE',   'ALL', 'rmse_pos_km'); 
    rows{end+1,1} = make_best_row(runsTbl, obsTbl, 'OVERALL_LOWEST_DET',    'ALL', 'mean_detPpos_km6'); 
    rows{end+1,1} = make_best_row(runsTbl, obsTbl, 'OVERALL_LOWEST_STAB',   'ALL', 'mean_stability'); 

    bestTbl = struct2table(vertcat(rows{:}), 'AsArray', true);
    bestTbl = sortrows(bestTbl, {'scope','model_group'});
end

function S = make_best_row(Tcand, obsTbl, scopeName, modelGroup, metricName)
    vals = Tcand.(metricName);
    vals(~isfinite(vals)) = NaN;

    [~, idxMin] = min(vals, [], 'omitnan');
    if isempty(idxMin) || isnan(idxMin)
        idxMin = 1;
    end
    r = Tcand(idxMin,:);

    if isempty(obsTbl)
        obsRun = table();
    else
        obsRun = obsTbl(obsTbl.file_path == r.file_path, :);
        if ismember('observer_id', obsRun.Properties.VariableNames)
            obsRun = sortrows(obsRun, 'observer_id');
        end
    end

    [orbitVec, slotVec, famVec, meanObsStab] = constellation_strings(obsRun);

    S = struct();
    S.scope = string(scopeName);
    S.model_group = string(modelGroup);
    S.metric_used = string(metricName);
    S.case_name = string(r.case_name);
    S.mission_type = string(r.mission_type);
    S.measurement_model = string(r.measurement_model);
    S.num_observers = double(r.num_observers);
    S.nperiods = double(r.nperiods);
    S.optimizer = string(r.optimizer);
    S.file_path = string(r.file_path);
    S.min_cost = double(r.min_cost);
    S.rmse_pos_km = double(r.rmse_pos_km);
    S.mean_detPpos_km6 = double(r.mean_detPpos_km6);
    S.mean_stability = double(r.mean_stability);
    S.runtime_s = double(r.runtime_s);
    S.orbit_indices = orbitVec;
    S.slot_indices = slotVec;
    S.orbit_families = famVec;
    S.mean_observer_stability_from_obs_sheet = meanObsStab;
end

function [orbitStr, slotStr, famStr, meanObsStab] = constellation_strings(obsRun)
    if isempty(obsRun)
        orbitStr = "";
        slotStr  = "";
        famStr   = "";
        meanObsStab = NaN;
        return;
    end

    orbitStr = join(string(obsRun.orbit_index(:)).', ', ');
    slotStr  = join(string(obsRun.slot_index(:)).', ', ');
    famStr   = join(string(obsRun.orbit_family(:)).', ' | ');

    stabVals = double(obsRun.stability_index);
    meanObsStab = mean(stabVals, 'omitnan');
end

function [familySummaryTbl, familyByGroupTbl] = build_orbit_family_summary(obsTbl)
    if isempty(obsTbl)
        familySummaryTbl = table();
        familyByGroupTbl = table();
        return;
    end

    fam = string(obsTbl.orbit_family);
    fam(strlength(strtrim(fam)) == 0 | ismissing(fam)) = "UNSPECIFIED";
    obsTbl.orbit_family = fam;

    G = findgroups(obsTbl.orbit_family);
    familySummaryTbl = table();
    familySummaryTbl.orbit_family = splitapply(@(x) x(1), obsTbl.orbit_family, G);
    familySummaryTbl.count = splitapply(@numel, obsTbl.orbit_family, G);
    familySummaryTbl.percent = 100 * familySummaryTbl.count / sum(familySummaryTbl.count);
    familySummaryTbl = sortrows(familySummaryTbl, 'count', 'descend');

    G2 = findgroups(obsTbl.model_group, obsTbl.orbit_family);
    familyByGroupTbl = table();
    familyByGroupTbl.model_group = splitapply(@(x) x(1), obsTbl.model_group, G2);
    familyByGroupTbl.orbit_family = splitapply(@(x) x(1), obsTbl.orbit_family, G2);
    familyByGroupTbl.count = splitapply(@numel, obsTbl.orbit_family, G2);
    familyByGroupTbl = sortrows(familyByGroupTbl, {'model_group','count'}, {'ascend','descend'});
end

function print_run_summary_console(runsTbl)
    fprintf('\n============================================================\n');
    fprintf('ENRICHED RUN SUMMARY\n');
    fprintf('============================================================\n');

    showVars = intersect({ ...
        'case_name','mission_type','measurement_model','nperiods','num_observers', ...
        'min_cost','rmse_pos_km','mean_detPpos_km6','mean_stability','runtime_s', ...
        'num_function_evals'}, ...
        runsTbl.Properties.VariableNames, 'stable');

    disp(runsTbl(:, showVars));
end

function print_observer_count_console(T)
    fprintf('\n============================================================\n');
    fprintf('OBSERVER COUNT SUMMARY\n');
    fprintf('============================================================\n');

    showVars = intersect({ ...
        'model_group','num_observers','n_runs', ...
        'rmse_pos_km_mean','mean_detPpos_km6_mean','mean_stability_mean', ...
        'runtime_s_mean','min_cost_mean'}, ...
        T.Properties.VariableNames, 'stable');

    disp(T(:, showVars));
end

function print_best_constellations_console(T)
    fprintf('\n============================================================\n');
    fprintf('BEST CONSTELLATIONS\n');
    fprintf('============================================================\n');

    showVars = intersect({ ...
        'scope','model_group','case_name','num_observers','nperiods', ...
        'min_cost','rmse_pos_km','mean_detPpos_km6','mean_stability', ...
        'orbit_indices','slot_indices','orbit_families'}, ...
        T.Properties.VariableNames, 'stable');

    disp(T(:, showVars));
end

function print_orbit_family_console(familySummaryTbl, familyByGroupTbl)
    fprintf('\n============================================================\n');
    fprintf('ORBIT FAMILY SUMMARY\n');
    fprintf('============================================================\n');

    if isempty(familySummaryTbl)
        fprintf('No observer family data were found.\n');
        return;
    end

    disp(familySummaryTbl);

    fprintf('\n------------------------------------------------------------\n');
    fprintf('ORBIT FAMILY SUMMARY BY MODEL GROUP\n');
    fprintf('------------------------------------------------------------\n');
    disp(familyByGroupTbl);
end

function plot_observer_count_metric_by_mission(T, figDir, yField, yLabel, fileStem)
    if isempty(T)
        return;
    end

    missionTypes = unique(T.mission_type, 'stable');
    for m = 1:numel(missionTypes)
        mission = missionTypes(m);
        idxMission = T.mission_type == mission;
        Tm = T(idxMission,:);
        if isempty(Tm)
            continue;
        end

        modelGroups = unique(Tm.model_group, 'stable');
        nGroups = numel(modelGroups);

        cmap = lines(max(nGroups, 1));
        markers = {'o','s','d','^','v','>','<','p','h'};

        figW = 7.5;
        figH = 5.8;
        f = figure('Color','w','Units','inches','Position',[1 1 figW figH], ...
                   'PaperUnits','inches','PaperPosition',[0 0 figW figH]);

        ax = axes(f);
        hold(ax, 'on');
        box(ax, 'on');
        set(ax, 'TickLabelInterpreter', 'tex', 'Layer', 'top');
        grid(ax, 'on');
        ax.GridAlpha = 0.25;     
        ax.MinorGridAlpha = 0.15;
        ax.LineWidth = 1.2; 

        legHandles = gobjects(nGroups,1);
        legLabels = strings(nGroups,1);

        for i = 1:nGroups
            idx = Tm.model_group == modelGroups(i);
            Tg = sortrows(Tm(idx,:), 'num_observers');
            mk = markers{mod(i-1, numel(markers))+1};

            legHandles(i) = plot(ax, Tg.num_observers, Tg.(yField), ['-' mk], ...
                'Color', cmap(i,:), ...
                'LineWidth', 1.8, ...
                'MarkerSize', 7, ...
                'MarkerFaceColor', cmap(i,:), ...
                'DisplayName', char(modelGroups(i)));

            legLabels(i) = make_group_legend_label(Tg(1,:));
        end

        xlabel(ax, 'Number of observers');
        ylabel(ax, yLabel);
        xticks(ax, unique(Tm.num_observers));
        xlim(ax, [min(Tm.num_observers)-0.3, max(Tm.num_observers)+0.3]);

        if strcmp(yField, 'rmse_pos_km_mean') || strcmp(yField, 'mean_detPpos_km6_mean')
            set(ax, 'YScale', 'log');
        end

        ax.Units = 'normalized';
        ax.PositionConstraint = 'innerposition';
        ax.Position = [0.125 0.14 0.80 0.80];
        ax.LooseInset = ax.TightInset + [0.02 0.02 0.02 0.02];

        lgd = legend(ax, legHandles, cellstr(legLabels), 'Location', 'eastoutside');
        lgd.Box = 'on';
        lgd.ItemTokenSize = [18 12];
        lgd.NumColumns = 1;

        missionCode = mission_code_for_filename(mission);
        exportgraphics(f, fullfile(figDir, sprintf('%s_%s.pdf', fileStem, missionCode)), 'ContentType', 'image');
        savefig(f, fullfile(figDir, sprintf('%s_%s.fig', fileStem, missionCode)));
        close(f);
    end
end

function label = make_group_legend_label(Trow)
    mission = string(Trow.mission_type(1));
    meas = string(Trow.measurement_model(1));
    nper = double(Trow.nperiods(1));

    switch upper(mission)
        case "LOW_THRUST_TRANSFER"
            missionShort = "Low-thrust";
        case "LUNAR_GATEWAY"
            missionShort = "Lunar Gateway";
        case "PERIODIC_ORBIT"
            missionShort = "Periodic orbit";
        otherwise
            missionShort = mission;
    end

    switch upper(meas)
        case "ANGLES_ONLY"
            measShort = "AO";
        case "ANGLES_RANGE"
            measShort = "AR";
        otherwise
            measShort = meas;
    end

    if strcmpi(mission, 'LUNAR_GATEWAY') || strcmpi(mission, 'PERIODIC_ORBIT')
        label = sprintf('p%d, %s', nper, measShort);
    else
        label = sprintf('%s', measShort);
    end
end

function plot_orbit_family_bar_by_mission(obsTbl, figDir)
    if isempty(obsTbl)
        return;
    end

    missionTypes = unique(obsTbl.mission_type, 'stable');
    for m = 1:numel(missionTypes)
        mission = missionTypes(m);
        Tm = obsTbl(obsTbl.mission_type == mission, :);
        if isempty(Tm)
            continue;
        end

        fam = string(Tm.orbit_family);
        fam(strlength(strtrim(fam)) == 0 | ismissing(fam)) = "UNSPECIFIED";
        Tm.orbit_family = fam;

        G = findgroups(Tm.orbit_family);
        Tplot = table();
        Tplot.orbit_family = splitapply(@(x) x(1), Tm.orbit_family, G);
        Tplot.count = splitapply(@numel, Tm.orbit_family, G);
        Tplot = sortrows(Tplot, 'count', 'descend');

        figW = 10.0;
        figH = 5.4;
        f = figure('Color','w','Units','inches','Position',[1 1 figW figH], ...
                   'PaperUnits','inches','PaperPosition',[0 0 figW figH]);

        ax = axes(f);
        hold(ax, 'on');
        box(ax, 'on');
        set(ax, 'TickLabelInterpreter', 'none', 'Layer', 'top');

        bar(ax, categorical(Tplot.orbit_family), Tplot.count, 'FaceColor', [0.25 0.25 0.25]);
        ylabel(ax, 'Count Across All Studies');
        xlabel(ax, 'Orbit family');
        ax.XTickLabelRotation = 30;

        ax.Units = 'normalized';
        ax.PositionConstraint = 'innerposition';
        ax.Position = [0.10 0.22 0.86 0.72];
        ax.LooseInset = ax.TightInset + [0.02 0.02 0.02 0.02];

        missionCode = mission_code_for_filename(mission);
        exportgraphics(f, fullfile(figDir, sprintf('orbit_family_counts_%s.pdf', missionCode)), 'ContentType', 'image');
        savefig(f, fullfile(figDir, sprintf('orbit_family_counts_%s.fig', missionCode)));
        close(f);
    end
end

function missionCode = mission_code_for_filename(mission)
    switch upper(string(mission))
        case "LOW_THRUST_TRANSFER"
            missionCode = 'lt';
        case "LUNAR_GATEWAY"
            missionCode = 'lg';
        case "PERIODIC_ORBIT"
            missionCode = 'po';
        case "TIME_OPT_TRANSFER"
            missionCode = 'tt';
        case "FUEL_OPT_TRANSFER"
            missionCode = 'ft';
        otherwise
            missionCode = lower(regexprep(char(string(mission)), '[^a-zA-Z0-9]+', '_'));
    end
end