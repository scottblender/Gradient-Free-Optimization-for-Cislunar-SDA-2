%% process_baseline_results.m
% Baseline post-processing for LaTeX subfigure workflows.
%
% This version is designed for:
%   1) Saving INDIVIDUAL trajectory EPS files (one file per case)
%   2) Saving INDIVIDUAL Monte Carlo EPS files (one file per case)
%   3) Saving shared legend-only EPS files
%   4) Writing one consolidated Excel summary
%
% Expected LaTeX workflow:
%   - Use the saved individual EPS files inside subfigure environments
%   - Use the shared legend EPS once below the subfigure grid
%
% Requirements:
%   - objective_wrapper.m on path
%   - build_target_truth.m on path
%   - JPL_CR3BP_OrbitCatalog.mat accessible in project directory
%
% Figures are exported as EPS with image/raster content.

clear; clc; close all;

%% ========================= FIND runs_GA =========================

if isfolder(fullfile(pwd, "runs_GA"))
    rootDir = fullfile(pwd, "runs_GA");
elseif strcmpi(string(getCurrentFolderName()), "runs_GA")
    rootDir = pwd;
else
    error("Could not find runs_GA. Put this script inside runs_GA or one folder above it.");
end

fprintf("Using runs_GA folder:\n%s\n", rootDir);

%% ========================= FIND PROJECT DIRECTORY =========================

projectDir = findProjectDir(rootDir);
addpath(genpath(projectDir));

fprintf("Using project directory:\n%s\n", projectDir);

catalogPath = fullfile(projectDir, "JPL_CR3BP_OrbitCatalog.mat");
if ~isfile(catalogPath)
    error("Could not find JPL_CR3BP_OrbitCatalog.mat in projectDir: %s", projectDir);
end

%% ========================= USER SETTINGS =========================

outDir = fullfile(rootDir, "BASELINE_REPORT_OUTPUT");

% -----------------------------------------------------------------
% GLOBAL FONT / STYLE CONTROLS
% -----------------------------------------------------------------
cfg.fontName   = "Times New Roman";
cfg.fontWeight = "bold";

% -----------------------------------------------------------------
% TRAJECTORY FIGURE CONTROLS
% -----------------------------------------------------------------
% These are used only when exporting the individual 3D trajectory EPS
% files for the LaTeX subfigure workflow. Keep these smaller than the
% Monte Carlo fonts because 3D axis labels are much easier to clip.
cfg.trajFigWidthIn      = 9.0;
cfg.trajFigHeightIn     = 8.0;
cfg.trajAxisFontSize    = 37;
cfg.trajLabelFontSize   = 37;
cfg.trajTitleFontSize   = 30;
cfg.trajLegendFontSize  = 30;
cfg.trajSharedLegendSize = 28;
cfg.trajAnnotationFontSize = 30;

% Padded axes box used in the temporary trajectory export figure:
% [left bottom width height].
% This keeps the tight subfigure look while reserving space for 3D labels.
% If labels crop, increase left/bottom or reduce width/height slightly.
cfg.trajPaddedAxesPosition = [0.18 0.18 0.70 0.60];
cfg.trajXLabelPosition = [0.42 -0.10 0];
cfg.trajYLabelPosition = [0.98 -0.05 0];

cfg.trajLineWidth       = 3.6;
cfg.trajAxisLineWidth   = 1.6;
cfg.trajMinScatterSize  = 80;
cfg.trajEpsResolution   = 600;

% Quick preview of each trajectory export. This writes and briefly opens
% a PNG that uses the exact same padded export figure as the EPS.
cfg.previewTrajectoryExport = false;
cfg.previewPauseSeconds = 5;
cfg.previewResolution = 600;

% -----------------------------------------------------------------
% MONTE CARLO FIGURE CONTROLS
% -----------------------------------------------------------------
% These are separate from the trajectory settings. Font size 50 worked
% well for the MC plots, so it is kept here.
cfg.mcFigWidthIn        = 8.5;
cfg.mcFigHeightIn       = 6.0;
cfg.mcAxisFontSize      = 50;
cfg.mcLabelFontSize     = 50;
cfg.mcLegendFontSize    = 42;
cfg.mcSharedLegendSize = 28;
cfg.mcLineWidth         = 3.8;
cfg.mcAxisLineWidth     = 1.8;
cfg.mcScatterSize       = 28;
cfg.mcMarkerLineWidth   = 3.5;

% Shared legend-only figure controls
cfg.legendFigWidthIn    = 7.5;
cfg.legendFigHeightIn   = 0.9;

% MC y-axis padding
cfg.mcLowerYPadFrac = 0.15;
cfg.mcUpperYPadFrac = 0.08;

% EPS raster/image export
cfg.mcEpsResolution = 600;

% Figure handling
cfg.removeTitles = true;

% IMPORTANT FOR LATEX SUBFIGURE WORKFLOW:
% Save the individual figure panels WITHOUT legends.
cfg.singleTrajShowLegend = false;
cfg.singleMCShowLegend   = false;

% Save shared legend-only EPS files
cfg.makeLegendOnlyFiles = false;

% -----------------------------------------------------------------
% TRUE MONTE CARLO SETTINGS
% -----------------------------------------------------------------
% Master/table switch: set this true if you want to compute/load the MC
% samples and write the MC summary/sample tables even when no figures are
% requested.
cfg.makeTrueMonteCarloPlots = true;

% Figure switches: these independently control which MC figures are saved.
% IMPORTANT: If either one is true, the script automatically runs/loads the
% MC validation even when cfg.makeTrueMonteCarloPlots is false.
cfg.makeTrueMCScatterPlots = false;
cfg.makeTrueMCBoxPlots     = true;

% Effective MC run switch. This is what the processing loop uses.
% This means these cases all work:
%   scatter=true,  box=false  -> saves only scatter EPS files
%   scatter=false, box=true   -> saves only box-and-whisker EPS files
%   scatter=true,  box=true   -> saves both EPS file types
%   scatter=false, box=false, makeTrueMonteCarloPlots=true -> MC tables only
cfg.runTrueMonteCarloValidation = ...
    cfg.makeTrueMonteCarloPlots || ...
    cfg.makeTrueMCScatterPlots || ...
    cfg.makeTrueMCBoxPlots;

cfg.mcNumSamples = 250;
cfg.mcOrbitRadius = 10;
cfg.mcSlotRadius  = 5;
cfg.mcSamplingMode = "UNIFORM_BOX";

cfg.EKF_DT = 0.01;
cfg.slots_per_orbit = 50;
cfg.useOrbitCache = true;
cfg.useTransferCache = true;

cfg.mcSeed = 1;
cfg.useMCCache = true;
cfg.forceRecomputeMC = false;

%% ========================= OUTPUT FOLDERS =========================

makeDir(outDir);

figOutDir   = fullfile(outDir, "figures");
tabOutDir   = fullfile(outDir, "tables");
trajOutDir  = fullfile(figOutDir, "trajectory_eps");
mcOutDir    = fullfile(figOutDir, "true_mc_eps");
legendOutDir = fullfile(figOutDir, "legend_eps");
mcCacheDir  = fullfile(outDir, "mc_cache");

makeDir(figOutDir);
makeDir(tabOutDir);
makeDir(trajOutDir);
makeDir(mcOutDir);
makeDir(legendOutDir);
makeDir(mcCacheDir);

cfg.mcCacheDir = mcCacheDir;

% Search the current cache folder and legacy cache folders so cached MC
% results from earlier versions of this script are reused instead of
% recomputed only because the output/filename convention changed.
cfg.mcCacheSearchDirs = [
    string(mcCacheDir)
    string(fullfile(rootDir, "BASELINE_REPORT_OUTPUT", "mc_cache"))
    string(fullfile(rootDir, "BASELINE_REPORT_OUTPUT_TRUE_MC", "mc_cache"))
    string(fullfile(rootDir, "BASELINE_REPORT_OUTPUT_LATEX_SUBFIGS", "mc_cache"))
];
cfg.mcCacheSearchDirs = unique(cfg.mcCacheSearchDirs, "stable");

%% ========================= LOAD STATIC CATALOG CONTEXT =========================

baseCtx = buildBaseCatalogContext(projectDir, catalogPath, cfg);

%% ========================= FIND EXCEL FILES =========================

xlsxFiles = dir(fullfile(rootDir, "**", "ExperimentSummary*.xlsx"));

if isempty(xlsxFiles)
    error("No ExperimentSummary*.xlsx files found under: %s", rootDir);
end

fprintf("Found %d ExperimentSummary files.\n", numel(xlsxFiles));

%% ========================= TABLES / INDEXES =========================

runSummaryAll = table();
observerAll   = table();
bestEvalAll   = table();
mcSummaryAll  = table();
mcSamplesAll  = table();
trajIndex     = table();
legendFiles   = table();

trajLegendTextMaster = "";

%% ========================= PROCESS EACH RUN =========================

for k = 1:numel(xlsxFiles)

    xlsxPath = fullfile(xlsxFiles(k).folder, xlsxFiles(k).name);
    runDir   = inferRunDirFromExcel(xlsxPath);

    fprintf("\n[%d/%d] Processing:\n%s\n", k, numel(xlsxFiles), xlsxPath);

    runInfo = parseRunInfo(runDir, xlsxPath);
    runInfo.runDir = string(runDir);

    %% ---------- Read Summary Sheet ----------
    try
        Tsum = readtable(xlsxPath, "Sheet", "Summary", ...
            "VariableNamingRule", "preserve");
    catch
        try
            Tsum = readtable(xlsxPath, "Sheet", 1, ...
                "VariableNamingRule", "preserve");
        catch
            warning("Could not read summary sheet: %s", xlsxPath);
            continue;
        end
    end

    if isempty(Tsum)
        warning("Empty summary sheet: %s", xlsxPath);
        continue;
    end

    S = Tsum(1,:);

    %% ---------- Read Sheet Names ----------
    try
        sheetNames = sheetnames(xlsxPath);
    catch
        [~, tmpSheets] = xlsfinfo(xlsxPath);
        sheetNames = string(tmpSheets);
    end

    %% ---------- Read Evaluation Sheet ----------
    Teval = table();
    evalSheetName = "";

    if numel(sheetNames) >= 2
        evalSheetName = sheetNames(2);
        try
            Teval = readtable(xlsxPath, "Sheet", evalSheetName, ...
                "VariableNamingRule", "preserve");
        catch
            warning("Could not read evaluation sheet %s in %s.", evalSheetName, xlsxPath);
            Teval = table();
        end
    end

    %% ---------- Read Observer Sheet ----------
    Tobs = table();
    obsSheetName = "";

    if numel(sheetNames) >= 3
        obsSheetName = sheetNames(3);
        try
            Tobs = readtable(xlsxPath, "Sheet", obsSheetName, ...
                "VariableNamingRule", "preserve");
        catch
            warning("Could not read observer sheet %s in %s.", obsSheetName, xlsxPath);
            Tobs = table();
        end
    end

    %% ---------- Fill runInfo from Summary if needed ----------
    runInfo = fillRunInfoFromSummary(runInfo, S);

    %% ---------- Function evaluation count ----------
    if ~isempty(Teval)
        nFuncEval = height(Teval);
    else
        nFuncEval = NaN;
    end

    %% ---------- Best evaluation row ----------
    [evalCostCol, hasEvalCost] = findCostColumn(Teval);

    if hasEvalCost
        Jhist = double(Teval.(evalCostCol));
        [bestCostEval, idxBest] = min(Jhist, [], "omitnan");
    else
        idxBest = NaN;
        bestCostEval = NaN;
        evalCostCol = "";
    end

    bestEvalMeta = table( ...
        string(runInfo.runName), ...
        string(runInfo.measurementShort), ...
        string(runInfo.measurementModel), ...
        string(runInfo.mission), ...
        runInfo.numObservers, ...
        runInfo.periods, ...
        idxBest, ...
        bestCostEval, ...
        string(evalSheetName), ...
        string(evalCostCol), ...
        'VariableNames', { ...
        'run_name','measurement_short','measurement_model', ...
        'mission','num_observers','periods','best_eval_index', ...
        'best_eval_cost','eval_sheet','eval_cost_column'});

    bestEvalAll = appendTableUnion(bestEvalAll, bestEvalMeta);

    %% ---------- Consolidated summary row ----------
    row = table();

    row.run_name          = string(runInfo.runName);
    row.run_folder        = string(runDir);
    row.excel_file        = string(xlsxPath);
    row.measurement_short = string(runInfo.measurementShort);
    row.measurement_model = string(runInfo.measurementModel);
    row.mission           = string(runInfo.mission);
    row.num_observers     = runInfo.numObservers;
    row.periods           = runInfo.periods;
    row.function_evals    = nFuncEval;

    row.run_tag           = getVarValue(S, ["run_tag","RunTag","tag"]);
    row.optimizer         = getVarValue(S, ["optimizer","Optimizer","solver","Solver"]);
    row.seed              = getNumericVarValue(S, ["seed","Seed"]);
    row.use_screening     = getNumericVarValue(S, ["use_screening","screening_flag","Screening"]);

    row.use_J1            = getNumericVarValue(S, ["use_J1","J1","cost_J1"]);
    row.use_J2            = getNumericVarValue(S, ["use_J2","J2","cost_J2"]);
    row.use_J3            = getNumericVarValue(S, ["use_J3","J3","cost_J3"]);

    row.runtime_s         = getNumericVarValue(S, ["runtime_s","runtime","Runtime","Runtime_s"]);
    row.screening_events  = getNumericVarValue(S, ["screeningCount_final","screening_events","ScreeningEvents"]);
    row.rmse_pos_km       = getNumericVarValue(S, ["rmse_pos_km","RMSE_pos_km","rmse"]);
    row.rmse_vel_kms      = getNumericVarValue(S, ["rmse_vel_kms","RMSE_vel_kms"]);
    row.mean_detPpos_km6  = getNumericVarValue(S, ["mean_detPpos_km6","mean_detPpos","detP","det"]);
    row.mean_stability    = getNumericVarValue(S, ["mean_stability","stability","mean_stab"]);
    row.min_cost          = getNumericVarValue(S, ["min_cost","J_total","cost","best_cost"]);
    row.best_eval_cost    = bestCostEval;

    runSummaryAll = appendTableUnion(runSummaryAll, row);

    %% ---------- Observer selections ----------
    if ~isempty(Tobs)
        Tobs.run_name          = repmat(string(runInfo.runName), height(Tobs), 1);
        Tobs.measurement_short = repmat(string(runInfo.measurementShort), height(Tobs), 1);
        Tobs.measurement_model = repmat(string(runInfo.measurementModel), height(Tobs), 1);
        Tobs.mission           = repmat(string(runInfo.mission), height(Tobs), 1);
        Tobs.num_observers     = repmat(runInfo.numObservers, height(Tobs), 1);
        Tobs.periods           = repmat(runInfo.periods, height(Tobs), 1);
        Tobs.excel_file        = repmat(string(xlsxPath), height(Tobs), 1);

        observerAll = appendTableUnion(observerAll, Tobs);
    end

    %% ---------- Export individual trajectory EPS ----------
    figsDir = fullfile(runDir, "figs");
    trajFigs = dir(fullfile(figsDir, "fig_traj3d*.fig"));

    if isempty(trajFigs)
        trajFigs = dir(fullfile(runDir, "**", "fig_traj3d*.fig"));
    end

    if ~isempty(trajFigs)

        trajFigPath = fullfile(trajFigs(1).folder, trajFigs(1).name);

        outStem = makeFigureStem(runInfo, "traj");
        outEps  = fullfile(trajOutDir, outStem + ".eps");

        try
            legendText = exportEnlargedFigAsImageEPS( ...
                trajFigPath, outEps, cfg, cfg.singleTrajShowLegend);

            if strlength(trajLegendTextMaster) == 0 && strlength(legendText) > 0
                trajLegendTextMaster = legendText;
            end

           newRow = table( ...
            string(runInfo.runName), ...
            string(runInfo.measurementShort), ...
            string(runInfo.measurementModel), ...
            string(runInfo.mission), ...
            runInfo.numObservers, ...
            runInfo.periods, ...
            string(outEps), ...
            string(trajFigPath), ...
            string(legendText), ...
            'VariableNames', { ...
            'run_name','measurement_short','measurement_model', ...
            'mission','num_observers','periods','eps_path', ...
            'fig_path','legend_text'});

            trajIndex = appendTableUnion(trajIndex, newRow);

        catch ME
            warning("Could not export trajectory fig:\n%s\n%s", trajFigPath, ME.message);
        end

    else
        warning("No trajectory .fig file found for run: %s", runInfo.runName);
    end

    %% ---------- TRUE Monte Carlo validation ----------
    if cfg.runTrueMonteCarloValidation
        try
            [mcSampleTbl, mcSummary] = makeTrueMonteCarloValidationPlot( ...
                Teval, Tobs, S, runInfo, baseCtx, cfg, mcOutDir);

            mcSummaryAll = appendTableUnion(mcSummaryAll, mcSummary);

            if ~isempty(mcSampleTbl)
                mcSamplesAll = appendTableUnion(mcSamplesAll, mcSampleTbl);
            end

        catch ME
            warning("TRUE Monte Carlo validation failed for %s\n%s", ...
                runInfo.runName, ME.message);
        end
    end
end

%% ========================= SORT TABLES =========================

if ~isempty(runSummaryAll)
    runSummaryAll = sortrows(runSummaryAll, ...
        {'mission','measurement_short','num_observers','periods'});
end

if ~isempty(observerAll)
    obsSortVars = intersect( ...
        {'mission','measurement_short','num_observers','periods','observer_id','Observer'}, ...
        observerAll.Properties.VariableNames, 'stable');

    if ~isempty(obsSortVars)
        observerAll = sortrows(observerAll, obsSortVars);
    end
end

if ~isempty(bestEvalAll)
    bestEvalAll = sortrows(bestEvalAll, ...
        {'mission','measurement_short','num_observers','periods'});
end

if ~isempty(mcSummaryAll)
    mcSummaryAll = sortrows(mcSummaryAll, ...
        {'mission','measurement_short','num_observers','periods'});
end

if ~isempty(trajIndex)
    trajIndex = sortrows(trajIndex, ...
        {'mission','measurement_short','num_observers','periods'});
end

%% ========================= CREATE LEGEND-ONLY EPS FILES =========================

if cfg.makeLegendOnlyFiles

    % ---- Trajectory legend ----
    if strlength(trajLegendTextMaster) > 0
        trajLegendPath = fullfile(legendOutDir, "legend_trajectory.eps");

        try
            createTrajectoryLegendOnly(trajLegendTextMaster, trajLegendPath, cfg);

            tmp = table("trajectory", string(trajLegendPath), ...
                'VariableNames', {'legend_type','eps_path'});
            legendFiles = appendTableUnion(legendFiles, tmp);

            fprintf("Saved trajectory legend EPS:\n%s\n", trajLegendPath);
        catch ME
            warning("Could not create trajectory legend-only EPS: %s", ME.message);
        end
    end

    % ---- MC legend ----
    mcLegendPath = fullfile(legendOutDir, "legend_monte_carlo.eps");
    try
        createMCLegendOnly(mcLegendPath, cfg);

        tmp = table("monte_carlo", string(mcLegendPath), ...
            'VariableNames', {'legend_type','eps_path'});
        legendFiles = appendTableUnion(legendFiles, tmp);

        fprintf("Saved Monte Carlo legend EPS:\n%s\n", mcLegendPath);
    catch ME
        warning("Could not create Monte Carlo legend-only EPS: %s", ME.message);
    end
end

%% ========================= ADD RANKINGS =========================

if ~isempty(runSummaryAll)

    runSummaryAll.cost_rank_overall = tiedrankSafe(runSummaryAll.min_cost);
    runSummaryAll.rmse_rank_overall = tiedrankSafe(runSummaryAll.rmse_pos_km);
    runSummaryAll.det_rank_overall  = tiedrankSafe(runSummaryAll.mean_detPpos_km6);
    runSummaryAll.time_rank_overall = tiedrankSafe(runSummaryAll.runtime_s);

    groupKey = strcat(runSummaryAll.mission, "_", ...
                      runSummaryAll.measurement_short, "_p", ...
                      string(runSummaryAll.periods));

    runSummaryAll.group_key = groupKey;

    runSummaryAll.best_cost_in_group = false(height(runSummaryAll),1);
    runSummaryAll.best_rmse_in_group = false(height(runSummaryAll),1);
    runSummaryAll.best_det_in_group  = false(height(runSummaryAll),1);
    runSummaryAll.fastest_in_group   = false(height(runSummaryAll),1);

    groups = unique(groupKey);

    for i = 1:numel(groups)

        idx = groupKey == groups(i);
        idxLocal = find(idx);

        if isempty(idxLocal)
            continue;
        end

        [~, a] = min(runSummaryAll.min_cost(idx), [], "omitnan");
        [~, b] = min(runSummaryAll.rmse_pos_km(idx), [], "omitnan");
        [~, c] = min(runSummaryAll.mean_detPpos_km6(idx), [], "omitnan");
        [~, d] = min(runSummaryAll.runtime_s(idx), [], "omitnan");

        if ~isempty(a) && ~isnan(a)
            runSummaryAll.best_cost_in_group(idxLocal(a)) = true;
        end
        if ~isempty(b) && ~isnan(b)
            runSummaryAll.best_rmse_in_group(idxLocal(b)) = true;
        end
        if ~isempty(c) && ~isnan(c)
            runSummaryAll.best_det_in_group(idxLocal(c)) = true;
        end
        if ~isempty(d) && ~isnan(d)
            runSummaryAll.fastest_in_group(idxLocal(d)) = true;
        end
    end
end

%% ========================= WRITE CONSOLIDATED EXCEL =========================

summaryXlsx = fullfile(tabOutDir, "Baseline_Consolidated_Summary_LATEX_SUBFIGS.xlsx");

if exist(summaryXlsx, "file")
    delete(summaryXlsx);
end

if ~isempty(runSummaryAll)
    writetable(runSummaryAll, summaryXlsx, "Sheet", "RunSummary");
end

if ~isempty(observerAll)
    writetable(observerAll, summaryXlsx, "Sheet", "ObserverSelections");
end

if ~isempty(bestEvalAll)
    writetable(bestEvalAll, summaryXlsx, "Sheet", "BestEvalRows");
end

if ~isempty(mcSummaryAll)
    writetable(mcSummaryAll, summaryXlsx, "Sheet", "TrueMCSummary");
end

if ~isempty(mcSamplesAll)
    writetable(mcSamplesAll, summaryXlsx, "Sheet", "TrueMCSamples");
end

if ~isempty(trajIndex)
    writetable(trajIndex, summaryXlsx, "Sheet", "TrajectoryFigureIndex");
end

if ~isempty(legendFiles)
    writetable(legendFiles, summaryXlsx, "Sheet", "LegendFiles");
end

%% ========================= DONE =========================

fprintf("\nDone.\n");
fprintf("Summary Excel:\n%s\n", summaryXlsx);
fprintf("Trajectory EPS folder:\n%s\n", trajOutDir);
fprintf("Monte Carlo EPS folder:\n%s\n", mcOutDir);
fprintf("Legend EPS folder:\n%s\n", legendOutDir);
fprintf("MC cache folder:\n%s\n", mcCacheDir);

%% ========================================================================
%% LOCAL FUNCTIONS
%% ========================================================================

function folderName = getCurrentFolderName()
    [~, folderName] = fileparts(pwd);
end

function makeDir(d)
    if ~exist(d, 'dir')
        mkdir(d);
    end
end

function projectDir = findProjectDir(rootDir)

    candidates = strings(0,1);

    candidates(end+1) = string(pwd);
    candidates(end+1) = string(rootDir);
    candidates(end+1) = string(fileparts(rootDir));

    tmp = string(rootDir);
    for i = 1:5
        tmp = string(fileparts(tmp));
        if strlength(tmp) > 0
            candidates(end+1) = tmp;
        end
    end

    candidates = unique(candidates, "stable");

    for i = 1:numel(candidates)
        if isfile(fullfile(candidates(i), "JPL_CR3BP_OrbitCatalog.mat"))
            projectDir = char(candidates(i));
            return;
        end
    end

    error("Could not locate project directory containing JPL_CR3BP_OrbitCatalog.mat.");
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

function stem = makeFigureStem(runInfo, kind)

    kind = lower(string(kind));

    if runInfo.mission == "lg"
        if isfinite(runInfo.periods)
            stem = sprintf("%s_%s_%s_o%d_p%d", ...
                char(kind), char(runInfo.mission), char(runInfo.measurementShort), ...
                runInfo.numObservers, runInfo.periods);
        else
            stem = sprintf("%s_%s_%s_o%d", ...
                char(kind), char(runInfo.mission), char(runInfo.measurementShort), ...
                runInfo.numObservers);
        end
    else
        stem = sprintf("%s_%s_%s_o%d", ...
            char(kind), char(runInfo.mission), char(runInfo.measurementShort), ...
            runInfo.numObservers);
    end

    stem = string(matlab.lang.makeValidName(string(stem)));
end

function baseCtx = buildBaseCatalogContext(projectDir, catalogPath, cfg)

    fprintf("\nLoading JPL catalog...\n");

    S = load(catalogPath);
    T1 = S.T;

    baseCtx = struct();
    baseCtx.projectDir = projectDir;
    baseCtx.catalogPath = catalogPath;
    baseCtx.S = S;
    baseCtx.T1 = T1;

    baseCtx.mu = 1.215058560962404E-2;
    baseCtx.LU = 384400;
    baseCtx.TU = 375695;
    baseCtx.VU = baseCtx.LU / baseCtx.TU;

    baseCtx.ode_opts = odeset('RelTol', 1e-13, 'AbsTol', 1e-13);

    baseCtx.num_orbits = height(T1);
    baseCtx.slots_per_orbit = cfg.slots_per_orbit;

    baseCtx.tf          = T1.("Period (TU) ");
    baseCtx.states      = T1.("state");
    baseCtx.times       = T1.("time");
    baseCtx.stabilities = T1.("Stability index  ");

    OrbitCacheDir = fullfile(projectDir, "orbit_cache");
    if ~exist(OrbitCacheDir, 'dir')
        mkdir(OrbitCacheDir);
    end

    orbitDbCacheFile = fullfile(OrbitCacheDir, ...
        sprintf('orbit_database_slots_%d_halfopen_v1.mat', cfg.slots_per_orbit));

    rebuildOrbitDb = true;

    if cfg.useOrbitCache && isfile(orbitDbCacheFile)
        try
            C = load(orbitDbCacheFile, 'orbit_database', 'cacheMeta');
            if isfield(C, 'orbit_database') && numel(C.orbit_database) == baseCtx.num_orbits && ...
                    isfield(C, 'cacheMeta') && isfield(C.cacheMeta, 'slot_definition') && ...
                    string(C.cacheMeta.slot_definition) == "equal_time_half_open_v1" && ...
                    isfield(C.cacheMeta, 'slots_per_orbit') && ...
                    C.cacheMeta.slots_per_orbit == cfg.slots_per_orbit
                baseCtx.orbit_database = C.orbit_database;
                rebuildOrbitDb = false;
                fprintf("Loaded cached orbit database:\n%s\n", orbitDbCacheFile);
            end
        catch ME
            warning("Failed to load orbit database cache: %s", ME.message);
            rebuildOrbitDb = true;
        end
    end

    if rebuildOrbitDb
        fprintf("Building orbit database with %d slots/orbit...\n", cfg.slots_per_orbit);

        orbit_database = cell(baseCtx.num_orbits, 1);

        times = baseCtx.times;
        states = baseCtx.states;
        tf = baseCtx.tf;
        slots_per_orbit = cfg.slots_per_orbit;

        parfor i = 1:baseCtx.num_orbits
            t_raw  = times{i};
            s_raw  = states{i};
            period = tf(i);

            t_slots = orbit_slot_times(period, slots_per_orbit);

            [t_unique, idx_u] = unique(t_raw);
            s_unique = s_raw(idx_u, :);

            F = griddedInterpolant(t_unique, s_unique, 'spline');
            s_slots = F(t_slots);

            orbit_database{i} = s_slots;
        end

        baseCtx.orbit_database = orbit_database;

        cacheMeta = struct();
        cacheMeta.created = string(datetime('now'));
        cacheMeta.num_orbits = baseCtx.num_orbits;
        cacheMeta.slots_per_orbit = cfg.slots_per_orbit;
        cacheMeta.slot_definition = "equal_time_half_open_v1";

        try
            save(orbitDbCacheFile, 'orbit_database', 'cacheMeta', '-v7.3');
            fprintf("Saved orbit database cache:\n%s\n", orbitDbCacheFile);
        catch ME
            warning("Could not save orbit database cache: %s", ME.message);
        end
    end

    baseCtx.TransferCacheDir = fullfile(projectDir, "transfer_cache");
    if ~exist(baseCtx.TransferCacheDir, 'dir')
        mkdir(baseCtx.TransferCacheDir);
    end
end

function runInfo = parseRunInfo(runDir, xlsxPath)

    txt = lower(string(runDir) + " " + string(xlsxPath));
    [~, runName] = fileparts(runDir);

    runInfo = struct();
    runInfo.runName = string(runName);

    if contains(txt, filesep + "ao" + filesep) || ...
       contains(txt, "_ao_") || ...
       contains(txt, "angles_only") || ...
       contains(txt, "anglesonly")
        runInfo.measurementShort = "ao";
        runInfo.measurementModel = "ANGLES_ONLY";
    elseif contains(txt, filesep + "ar" + filesep) || ...
           contains(txt, "_ar_") || ...
           contains(txt, "angles_range") || ...
           contains(txt, "anglesrange")
        runInfo.measurementShort = "ar";
        runInfo.measurementModel = "ANGLES_RANGE";
    else
        runInfo.measurementShort = "unknown";
        runInfo.measurementModel = "unknown";
    end

    if contains(txt, filesep + "lg" + filesep) || ...
       contains(txt, "_lg_") || ...
       contains(txt, "lunar_gateway") || ...
       contains(txt, "gateway")
        runInfo.mission = "lg";
    elseif contains(txt, filesep + "lt" + filesep) || ...
           contains(txt, "_lt_") || ...
           contains(txt, "low_thrust") || ...
           contains(txt, "lowthrust") || ...
           contains(txt, "transfer")
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

    meas = string(getVarValue(S, ["measurement_model","meas_model","MEAS_MODEL"]));
    if runInfo.measurementModel == "unknown" && strlength(meas) > 0 && meas ~= "<missing>"
        meas = upper(meas);
        if contains(meas, "ANGLES_ONLY")
            runInfo.measurementModel = "ANGLES_ONLY";
            runInfo.measurementShort = "ao";
        elseif contains(meas, "ANGLES_RANGE")
            runInfo.measurementModel = "ANGLES_RANGE";
            runInfo.measurementShort = "ar";
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
        elseif runInfo.mission == "lg"
            runInfo.periods = 1;
        end
    end
end

function ctx = buildObjectiveContextForRun(runInfo, S, baseCtx, cfg)

    % Never reinterpret a legacy solution's slot indices on the new grid.
    if ~ismember('slot_definition', S.Properties.VariableNames) || ...
            string(S.slot_definition(1)) ~= "equal_time_half_open_v1" || ...
            ~ismember('slots_per_orbit', S.Properties.VariableNames) || ...
            S.slots_per_orbit(1) ~= cfg.slots_per_orbit
        error('slots:IncompatibleSummary', ...
            ['Cannot recompute this run with the half-open slot grid. ' ...
             'Use its original code/grid for legacy results, or rerun optimization.']);
    end

    ctx = struct();

    ctx.mu = baseCtx.mu;
    ctx.LU = baseCtx.LU;
    ctx.TU = baseCtx.TU;
    ctx.VU = baseCtx.VU;

    ctx.orbit_database = baseCtx.orbit_database;
    ctx.stabilities    = baseCtx.stabilities;

    ctx.T1     = baseCtx.T1;
    ctx.times  = baseCtx.times;
    ctx.states = baseCtx.states;

    ctx.sun_min  = deg2rad(20);
    ctx.moon_min = deg2rad(10);

    theta0 = 0;
    i_sun  = deg2rad(0);
    LU = ctx.LU;
    TU = ctx.TU;

    ctx.sunFcn = @(t) sun_pos_bc4bp(t, LU, TU, theta0, i_sun);

    useScreeningVal = getNumericVarValue(S, ["use_screening","screening_flag","Screening"]);
    if isfinite(useScreeningVal)
        ctx.useScreening = logical(useScreeningVal);
    else
        ctx.useScreening = true;
    end

    ctx.costFlags = struct('J1', true, 'J2', true, 'J3', true);

    j1 = getNumericVarValue(S, ["use_J1","J1","cost_J1"]);
    j2 = getNumericVarValue(S, ["use_J2","J2","cost_J2"]);
    j3 = getNumericVarValue(S, ["use_J3","J3","cost_J3"]);

    if isfinite(j1), ctx.costFlags.J1 = logical(j1); end
    if isfinite(j2), ctx.costFlags.J2 = logical(j2); end
    if isfinite(j3), ctx.costFlags.J3 = logical(j3); end

    ctx.measCfg = struct();
    ctx.measCfg.type = upper(string(runInfo.measurementModel));

    if ctx.measCfg.type ~= "ANGLES_ONLY" && ctx.measCfg.type ~= "ANGLES_RANGE"
        ctx.measCfg.type = "ANGLES_ONLY";
    end

    ctx.missionCfg = buildMissionCfg(runInfo);

    [t_target, s_target, truthInfo] = buildOrLoadTargetTruth(ctx.missionCfg, baseCtx, cfg);

    ctx.t_target_truth = t_target;
    ctx.s_target_truth = s_target;
    ctx.truthInfo = truthInfo;

    EKF_DT = cfg.EKF_DT;

    t_truth = t_target(:);
    s_truth = s_target;

    t_target_ekf = (t_truth(1):EKF_DT:t_truth(end)).';
    if isempty(t_target_ekf) || t_target_ekf(end) < t_truth(end)
        t_target_ekf = [t_target_ekf; t_truth(end)];
    end

    [t_unique_truth, idx_u_truth] = unique(t_truth);
    s_unique_truth = s_truth(idx_u_truth, :);

    F_truth = griddedInterpolant(t_unique_truth, s_unique_truth, 'spline');
    s_target_ekf = F_truth(t_target_ekf);

    ctx.t_target = t_target_ekf;
    ctx.s_target = s_target_ekf;

    [ctx.P0, ctx.Q, ctx.R, ctx.costCfg] = buildEkfAndCostConfig(ctx.missionCfg, ctx.measCfg, ctx.LU, ctx.VU);

    ctx.opt_flag = 'SOO';

    optVal = string(getVarValue(S, ["optimizer","Optimizer","solver","Solver"]));
    if strlength(optVal) > 0 && optVal ~= "<missing>"
        ctx.solverName = upper(optVal);
    else
        ctx.solverName = "GA";
    end
end

function missionCfg = buildMissionCfg(runInfo)

    missionCfg = struct();

    switch lower(string(runInfo.mission))
        case "lg"
            missionCfg.type = "LUNAR_GATEWAY";
        case "lt"
            missionCfg.type = "LOW_THRUST_TRANSFER";
        otherwise
            error("Cannot determine mission type for run %s.", runInfo.runName);
    end

    missionCfg.optimization.numObservers = round(runInfo.numObservers);

    switch missionCfg.type

        case "LUNAR_GATEWAY"
            missionCfg.gateway.s0 = [1.02202108343387, 0, -0.182096487798513, ...
                                     0, -0.103255420206012, 0]';
            missionCfg.gateway.period   = 1.51110546287394;
            missionCfg.gateway.dt       = 0.001;

            if isfinite(runInfo.periods) && runInfo.periods > 0
                missionCfg.gateway.Nperiods = round(runInfo.periods);
            else
                missionCfg.gateway.Nperiods = 1;
            end

        case "LOW_THRUST_TRANSFER"
            missionCfg.transfer.depOrbitIndex = 52;
            missionCfg.transfer.depSlot       = 10;
            missionCfg.transfer.arrOrbitIndex = 400;
            missionCfg.transfer.arrSlot       = 1;
            missionCfg.transfer.dt            = 0.001;
            missionCfg.transfer.solverMode    = "LOW_THRUST_CLASS";

            missionCfg.transfer.lowthrust.sigma            = 1.0;
            missionCfg.transfer.lowthrust.m0               = 1.0;
            missionCfg.transfer.lowthrust.Tmax             = 0.3672;
            missionCfg.transfer.lowthrust.ve               = 39.8;
            missionCfg.transfer.lowthrust.tf_guess         = 2.0;
            missionCfg.transfer.lowthrust.tf_lb            = 0.1;
            missionCfg.transfer.lowthrust.tf_ub            = 12.0;

            missionCfg.transfer.lowthrust.lambda_guess     = [
               -0.25
                0.75
                0.35
               -0.20
                0.40
                0.10
                0.05
            ];

            missionCfg.transfer.lowthrust.lambda_lb        = -20 * ones(7,1);
            missionCfg.transfer.lowthrust.lambda_ub        =  20 * ones(7,1);

            missionCfg.transfer.lowthrust.w_pos_indirect   = 1;
            missionCfg.transfer.lowthrust.w_vel_indirect   = 1;
            missionCfg.transfer.lowthrust.w_norm_indirect  = 1;
            missionCfg.transfer.lowthrust.w_mass_indirect  = 1;
    end
end

function [t_target, s_target, truthInfo] = buildOrLoadTargetTruth(missionCfg, baseCtx, cfg)

    useTransferCache = cfg.useTransferCache && contains(string(missionCfg.type), "TRANSFER");

    if useTransferCache
        cacheKey  = make_transfer_cache_key(missionCfg, baseCtx.slots_per_orbit);
        cacheFile = fullfile(baseCtx.TransferCacheDir, cacheKey + ".mat");

        if isfile(cacheFile)
            try
                C = load(cacheFile, 't_target', 's_target', 'truthInfo');
                t_target = C.t_target;
                s_target = C.s_target;
                truthInfo = C.truthInfo;
                return;
            catch ME
                warning("Failed to load transfer cache, rebuilding: %s", ME.message);
            end
        end
    end

    [t_target, s_target, truthInfo] = build_target_truth( ...
        missionCfg, baseCtx.T1, baseCtx.orbit_database, ...
        baseCtx.times, baseCtx.states, baseCtx.mu, baseCtx.ode_opts);

    if useTransferCache
        try
            cacheMeta = struct();
            cacheMeta.cacheKey = cacheKey;
            cacheMeta.created = string(datetime('now'));
            cacheMeta.slot_definition = "equal_time_half_open_v1";

            save(cacheFile, 't_target', 's_target', 'truthInfo', 'cacheMeta', '-v7.3');
        catch ME
            warning("Could not save transfer cache: %s", ME.message);
        end
    end
end

function [P0, Q, R, costCfg] = buildEkfAndCostConfig(missionCfg, measCfg, LU, VU)

    switch upper(string(missionCfg.type))

        case "LOW_THRUST_TRANSFER"
            pos_var  = (1 / LU)^2;
            vel_var  = (10 / (VU * 1000))^2;
            P0 = diag([pos_var, pos_var, pos_var, vel_var, vel_var, vel_var]);

            q_pos = 6.25e-4;
            q_vel = 6.25e-4;
            Q = diag([q_pos q_pos q_pos q_vel q_vel q_vel]);

            r_ang   = 1e-8;
            r_range = (1 / LU)^2;

            switch upper(string(measCfg.type))
                case "ANGLES_ONLY"
                    R = diag([r_ang r_ang]);
                case "ANGLES_RANGE"
                    R = diag([r_ang r_ang r_range]);
            end

            costCfg = struct();
            costCfg.weights = [1, 1, 0.1];
            costCfg.pos_rmse_acc = 100 / LU;
            costCfg.vel_rmse_acc = 0.1 / VU;
            costCfg.sigma_pos_acc = 100 / LU;
            costCfg.sigma_vel_acc = 0.1 / VU;
            costCfg.stability_acc = 1.0;

        case "LUNAR_GATEWAY"
            pos_var  = (1 / LU)^2;
            vel_var  = (10 / (VU * 1000))^2;
            P0 = diag([pos_var, pos_var, pos_var, vel_var, vel_var, vel_var]);

            q_pos = 1e-8;
            q_vel = 1e-8;
            Q = diag([q_pos q_pos q_pos q_vel q_vel q_vel]);

            r_ang   = 1e-8;
            r_range = (1 / LU)^2;

            switch upper(string(measCfg.type))
                case "ANGLES_ONLY"
                    R = diag([r_ang r_ang]);
                case "ANGLES_RANGE"
                    R = diag([r_ang r_ang r_range]);
            end

            costCfg = struct();
            costCfg.weights = [1, 1, 0.1];
            costCfg.pos_rmse_acc = 1 / LU;
            costCfg.vel_rmse_acc = 1.0e-3 / VU;
            costCfg.sigma_pos_acc = 1 / LU;
            costCfg.sigma_vel_acc = 1.0e-3 / VU;
            costCfg.stability_acc = 1.0;

        otherwise
            error("Unsupported mission type: %s", string(missionCfg.type));
    end
end

function [mcTable, mcSummary] = makeTrueMonteCarloValidationPlot(Teval, Tobs, S, runInfo, baseCtx, cfg, outDir)

    fprintf("  Building objective context for %s...\n", runInfo.runName);
    ctx = buildObjectiveContextForRun(runInfo, S, baseCtx, cfg);

    JgaMin = getGaMinimumCostFromExcel(Teval, S);

    xBest = extractFinalDesign(Teval, Tobs, runInfo);
    xBest = round(xBest);

    nVars = numel(xBest);
    numObs = nVars / 2;

    lb = repmat([1 1], 1, numObs);
    ub = repmat([baseCtx.num_orbits baseCtx.slots_per_orbit], 1, numObs);

    [cacheFile, cacheFileToSave] = findExistingOrNewMCCacheFile(runInfo, xBest, cfg);

    fprintf("  Looking for MC cache:\n  %s\n", cacheFile);
    fprintf("  Primary save cache path:\n  %s\n", cacheFileToSave);

    loadedFromCache = false;

    if cfg.useMCCache && ~cfg.forceRecomputeMC && isfile(cacheFile)
        try
            C = load(cacheFile, "Xmc", "Jmc", "JgaMin", "cacheMeta");

            if isfield(C, "cacheMeta") && isMCCacheCompatible(C.cacheMeta, runInfo, xBest, cfg)
                Xmc = C.Xmc;
                Jmc = C.Jmc;
                JgaMin = C.JgaMin;

                loadedFromCache = true;
                fprintf("  Loaded TRUE MC cache:\n  %s\n", cacheFile);
            else
                fprintf("  MC cache exists but settings changed. Recomputing.\n");
            end
        catch ME
            warning("Failed to load MC cache for %s. Recomputing. Reason: %s", ...
                runInfo.runName, ME.message);
        end
    end

    if ~loadedFromCache

        rng(cfg.mcSeed, "twister");

        Xmc = sampleAroundBestDesign(xBest, lb, ub, cfg.mcNumSamples, ...
            cfg.mcOrbitRadius, cfg.mcSlotRadius, cfg.mcSamplingMode);

        Jmc = nan(cfg.mcNumSamples, 1);

        fprintf("  Running TRUE MC for %s: %d objective evaluations...\n", ...
            runInfo.runName, cfg.mcNumSamples);

        dq = [];

        for i = 1:cfg.mcNumSamples

            xSample = Xmc(i,:);

            J = objective_wrapper( ...
                xSample, ...
                ctx.orbit_database, ...
                ctx.stabilities, ...
                ctx.s_target, ...
                ctx.t_target, ...
                ctx.P0, ...
                ctx.Q, ...
                ctx.R, ...
                ctx.mu, ...
                ctx.LU, ...
                ctx.sunFcn, ...
                ctx.sun_min, ...
                ctx.moon_min, ...
                ctx.opt_flag, ...
                ctx.solverName, ...
                dq, ...
                ctx.useScreening, ...
                ctx.costFlags, ...
                ctx.costCfg, ...
                ctx.measCfg);

            if numel(J) > 1
                J = sum(J(:), "omitnan");
            end

            Jmc(i) = double(J);

            if mod(i, max(1, round(cfg.mcNumSamples/10))) == 0
                fprintf("    %d/%d samples complete.\n", i, cfg.mcNumSamples);
            end
        end

        if cfg.useMCCache
            cacheMeta = makeMCCacheMeta(runInfo, xBest, cfg);

            try
                save(cacheFileToSave, ...
                    "Xmc", "Jmc", "JgaMin", ...
                    "xBest", "lb", "ub", "cacheMeta", "-v7.3");

                cacheFile = cacheFileToSave;

                fprintf("  Saved TRUE MC cache:\n  %s\n", cacheFile);
            catch ME
                warning("Failed to save MC cache for %s. Reason: %s", ...
                    runInfo.runName, ME.message);
            end
        end
    end

    nSamples = numel(Jmc);

    mcTable = table();
    mcTable.run_name = repmat(string(runInfo.runName), nSamples, 1);
    mcTable.measurement_short = repmat(string(runInfo.measurementShort), nSamples, 1);
    mcTable.measurement_model = repmat(string(runInfo.measurementModel), nSamples, 1);
    mcTable.mission = repmat(string(runInfo.mission), nSamples, 1);
    mcTable.num_observers = repmat(runInfo.numObservers, nSamples, 1);
    mcTable.periods = repmat(runInfo.periods, nSamples, 1);
    mcTable.sample = (1:nSamples).';
    mcTable.total_cost = Jmc(:);
    mcTable.ga_min_cost = repmat(JgaMin, nSamples, 1);
    mcTable.loaded_from_cache = repmat(loadedFromCache, nSamples, 1);

    for j = 1:nVars
        mcTable.("x_" + string(j)) = Xmc(:,j);
    end

    outStem = makeFigureStem(runInfo, "mc");

    if ~isfield(cfg, "makeTrueMCScatterPlots")
        cfg.makeTrueMCScatterPlots = true;
    end
    if ~isfield(cfg, "makeTrueMCBoxPlots")
        cfg.makeTrueMCBoxPlots = false;
    end

    scatterEps = "";
    boxEps     = "";

    % ------------------------------------------------------------
    % Optional MC scatter plot
    % ------------------------------------------------------------
    if cfg.makeTrueMCScatterPlots
        scatterEps = fullfile(outDir, outStem + ".eps");

        fig = createTrueMCFigure(mcTable.sample, mcTable.total_cost, ...
            JgaMin, cfg, cfg.singleMCShowLegend);

        exportMCFigureAsImageEPS(fig, scatterEps, cfg);
        close(fig);
    end

    % ------------------------------------------------------------
    % Optional MC box-and-whisker plot
    % ------------------------------------------------------------
    if cfg.makeTrueMCBoxPlots
        boxStem = makeFigureStem(runInfo, "mc_box");
        boxEps  = fullfile(outDir, boxStem + ".eps");

        fig = createTrueMCBoxFigure(mcTable.total_cost, ...
            JgaMin, cfg, cfg.singleMCShowLegend);

        exportMCFigureAsImageEPS(fig, boxEps, cfg);
        close(fig);
    end

    mcSummary = table();

    mcSummary.run_name            = string(runInfo.runName);
    mcSummary.measurement_short   = string(runInfo.measurementShort);
    mcSummary.measurement_model   = string(runInfo.measurementModel);
    mcSummary.mission             = string(runInfo.mission);
    mcSummary.num_observers       = runInfo.numObservers;
    mcSummary.periods             = runInfo.periods;
    mcSummary.mode_used           = string("true_local_monte_carlo_objective_wrapper");
    mcSummary.num_samples         = nSamples;
    mcSummary.loaded_from_cache   = loadedFromCache;
    mcSummary.cache_file          = string(cacheFile);
    mcSummary.ga_min_cost         = JgaMin;
    mcSummary.mc_min_cost         = min(Jmc, [], "omitnan");
    mcSummary.mc_mean_cost        = mean(Jmc, "omitnan");
    mcSummary.mc_median_cost      = median(Jmc, "omitnan");
    mcSummary.mc_std_cost         = std(Jmc, "omitnan");
    mcSummary.mc_percent_below_ga_min = 100 * mean(Jmc < JgaMin, "omitnan");

    for j = 1:nVars
        mcSummary.("xbest_" + string(j)) = xBest(j);
    end

    mcSummary.scatter_plot_file = string(scatterEps);
    mcSummary.box_plot_file     = string(boxEps);

    % Backward-compatible generic plot field. Prefer the scatter plot
    % if it exists; otherwise use the box-and-whisker plot.
    if strlength(string(scatterEps)) > 0
        mcSummary.plot_file = string(scatterEps);
    elseif strlength(string(boxEps)) > 0
        mcSummary.plot_file = string(boxEps);
    else
        mcSummary.plot_file = "";
    end
end

function JgaMin = getGaMinimumCostFromExcel(Teval, S)

    JgaMin = NaN;

    summaryCost = getNumericVarValue(S, ["min_cost","J_total","cost","best_cost"]);
    if isfinite(summaryCost)
        JgaMin = summaryCost;
        return;
    end

    [costCol, hasCost] = findCostColumn(Teval);
    if hasCost
        Jhist = double(Teval.(costCol));
        JgaMin = min(Jhist, [], "omitnan");
    end

    if ~isfinite(JgaMin)
        error("Could not determine GA minimum cost from Excel.");
    end
end

function cacheFile = makeMCCacheFile(runInfo, xBest, cfg)

    cacheMeta = makeMCCacheMeta(runInfo, xBest, cfg);

    % Keep the legacy low-thrust naming convention as pNaN so caches that
    % already exist with names like ..._pNaN_... are reused directly.
    if isnan(cacheMeta.periods)
        periodStr = "pNaN";
    else
        periodStr = "p" + string(round(cacheMeta.periods));
    end

    rawName = sprintf("mc_%s_%s_%s_o%d_%s_n%d_ro%d_rs%d_dt%s_seed%d_%s", ...
        char(cacheMeta.run_name), ...
        char(cacheMeta.mission), ...
        char(cacheMeta.measurement_short), ...
        cacheMeta.num_observers, ...
        char(periodStr), ...
        cacheMeta.mcNumSamples, ...
        cacheMeta.mcOrbitRadius, ...
        cacheMeta.mcSlotRadius, ...
        char(local_num_str(cacheMeta.EKF_DT)), ...
        cacheMeta.mcSeed, ...
        char(cacheMeta.xBestHash));

    rawName = string(matlab.lang.makeValidName(string(rawName)));
    cacheFile = fullfile(cfg.mcCacheDir, rawName + "_halfopen_v1_sunlu_v2.mat");
end

function [cacheFile, cacheFileToSave] = findExistingOrNewMCCacheFile(runInfo, xBest, cfg)

    % Current expected cache filename in the primary cache folder.
    cacheFileToSave = makeMCCacheFile(runInfo, xBest, cfg);
    [~, cacheName, cacheExt] = fileparts(cacheFileToSave);

    cacheBaseNames = strings(0,1);
    cacheBaseNames(end+1) = string(cacheName + cacheExt);

    % Legacy period naming variants.
    cacheBaseNames(end+1) = replace(string(cacheName + cacheExt), "pNA", "pNaN");
    cacheBaseNames(end+1) = replace(string(cacheName + cacheExt), "pNaN", "pNA");
    cacheBaseNames = unique(cacheBaseNames, "stable");

    cacheMeta = makeMCCacheMeta(runInfo, xBest, cfg);
    hashStr = string(cacheMeta.xBestHash);

    searchDirs = string(cfg.mcCacheDir);
    if isfield(cfg, "mcCacheSearchDirs")
        searchDirs = unique([searchDirs(:); string(cfg.mcCacheSearchDirs(:))], "stable");
    end

    % 1) Exact filename search in all known cache directories.
    for d = 1:numel(searchDirs)
        for n = 1:numel(cacheBaseNames)
            candidate = fullfile(searchDirs(d), cacheBaseNames(n));
            if isfile(candidate)
                cacheFile = candidate;
                return;
            end
        end
    end

    % 2) Fuzzy legacy search by run metadata, MC settings, and xBest hash.
    % This catches older names such as:
    % mc_b_ga600_ar_o10_lt_ar_o10_pNaN_n250_ro10_rs5_dt0p01_seed1_HASH.mat
    patterns = [
        sprintf("*%s*%s*%s*o%d*n%d*ro%d*rs%d*dt%s*seed%d*%s*.mat", ...
            char(runInfo.runName), ...
            char(runInfo.mission), ...
            char(runInfo.measurementShort), ...
            round(runInfo.numObservers), ...
            cfg.mcNumSamples, ...
            cfg.mcOrbitRadius, ...
            cfg.mcSlotRadius, ...
            char(local_num_str(cfg.EKF_DT)), ...
            cfg.mcSeed, ...
            char(hashStr))

        sprintf("*%s*o%d*n%d*ro%d*rs%d*dt%s*seed%d*%s*.mat", ...
            char(runInfo.runName), ...
            round(runInfo.numObservers), ...
            cfg.mcNumSamples, ...
            cfg.mcOrbitRadius, ...
            cfg.mcSlotRadius, ...
            char(local_num_str(cfg.EKF_DT)), ...
            cfg.mcSeed, ...
            char(hashStr))
    ];

    for d = 1:numel(searchDirs)
        for p = 1:numel(patterns)
            hits = dir(fullfile(searchDirs(d), patterns(p)));
            if ~isempty(hits)
                cacheFile = fullfile(hits(1).folder, hits(1).name);
                return;
            end
        end
    end

    % If no existing cache was found, save to the current primary path.
    cacheFile = cacheFileToSave;
end

function cacheMeta = makeMCCacheMeta(runInfo, xBest, cfg)

    cacheMeta = struct();

    cacheMeta.created = string(datetime("now"));
    cacheMeta.run_name = string(runInfo.runName);
    cacheMeta.mission = string(runInfo.mission);
    cacheMeta.measurement_short = string(runInfo.measurementShort);
    cacheMeta.measurement_model = string(runInfo.measurementModel);
    cacheMeta.num_observers = double(runInfo.numObservers);
    cacheMeta.periods = double(runInfo.periods);

    cacheMeta.mcNumSamples = double(cfg.mcNumSamples);
    cacheMeta.mcOrbitRadius = double(cfg.mcOrbitRadius);
    cacheMeta.mcSlotRadius = double(cfg.mcSlotRadius);
    cacheMeta.mcSamplingMode = string(cfg.mcSamplingMode);
    cacheMeta.mcSeed = double(cfg.mcSeed);
    cacheMeta.EKF_DT = double(cfg.EKF_DT);
    cacheMeta.slots_per_orbit = double(cfg.slots_per_orbit);
    cacheMeta.slot_definition = "equal_time_half_open_v1";
    cacheMeta.sun_model_version = "sun_lu_v2";

    cacheMeta.xBest = double(xBest(:).');
    cacheMeta.xBestHash = simpleVectorHash(cacheMeta.xBest);
end

function tf = isMCCacheCompatible(cacheMeta, runInfo, xBest, cfg)

    tf = false;
    if ~isfield(cacheMeta, 'slot_definition') || ...
            string(cacheMeta.slot_definition) ~= "equal_time_half_open_v1" || ...
            ~isfield(cacheMeta, 'sun_model_version') || ...
            string(cacheMeta.sun_model_version) ~= "sun_lu_v2"
        return;
    end

    checks = [
        string(cacheMeta.run_name) == string(runInfo.runName)
        string(cacheMeta.mission) == string(runInfo.mission)
        string(cacheMeta.measurement_short) == string(runInfo.measurementShort)
        string(cacheMeta.measurement_model) == string(runInfo.measurementModel)

        sameNumber(cacheMeta.num_observers, runInfo.numObservers)
        sameNumber(cacheMeta.periods, runInfo.periods)

        sameNumber(cacheMeta.mcNumSamples, cfg.mcNumSamples)
        sameNumber(cacheMeta.mcOrbitRadius, cfg.mcOrbitRadius)
        sameNumber(cacheMeta.mcSlotRadius, cfg.mcSlotRadius)

        string(cacheMeta.mcSamplingMode) == string(cfg.mcSamplingMode)
        sameNumber(cacheMeta.mcSeed, cfg.mcSeed)
        sameNumber(cacheMeta.EKF_DT, cfg.EKF_DT)
        sameNumber(cacheMeta.slots_per_orbit, cfg.slots_per_orbit)

        string(cacheMeta.xBestHash) == string(simpleVectorHash(xBest))
    ];

    tf = all(checks);
end

function tf = sameNumber(a, b)
%SAMENUMBER True if two numeric values are equal, treating NaN as equal.

    a = double(a);
    b = double(b);

    if isnan(a) && isnan(b)
        tf = true;
    else
        tf = isequal(a, b);
    end
end

function h = simpleVectorHash(x)

    x = double(x(:).');
    s = char(sprintf("%.12g_", x));

    v = uint32(2166136261);

    for k = 1:numel(s)
        v = bitxor(v, uint32(s(k)));
        v = uint32(mod(double(v) * 16777619, 2^32));
    end

    h = string(sprintf("%08X", v));
end

function xBest = extractFinalDesign(Teval, Tobs, runInfo)

    try
        xBest = getDesignFromObserverSheet(Tobs, runInfo.numObservers);
        return;
    catch
    end

    if isempty(Teval)
        error("Cannot extract final design: both observer sheet and evaluation sheet are unavailable.");
    end

    [costCol, hasCost] = findCostColumn(Teval);
    if ~hasCost
        error("Cannot extract best design: no cost column found in evaluation sheet.");
    end

    J = double(Teval.(costCol));
    [~, idxBest] = min(J, [], "omitnan");

    [~, xColNames] = findDesignColumns(Teval);
    if isempty(xColNames)
        error("Cannot extract design: no x_1, x_2, ... columns found.");
    end

    xBest = nan(1, numel(xColNames));
    for j = 1:numel(xColNames)
        xBest(j) = double(Teval.(xColNames(j))(idxBest));
    end

    xBest = round(xBest);
end

function xBest = getDesignFromObserverSheet(Tobs, numObservers)

    if isempty(Tobs)
        error("Observer sheet is empty.");
    end

    vars = string(Tobs.Properties.VariableNames);
    cleanVars = lower(regexprep(vars, "[^a-zA-Z0-9]", ""));

    orbitCandidates = ["orbitindex","orbitidx","orbit","selectedorbit"];
    slotCandidates  = ["slotindex","slotidx","slot","selectedslot"];

    orbitCol = "";
    slotCol  = "";

    for i = 1:numel(orbitCandidates)
        hit = cleanVars == orbitCandidates(i);
        if any(hit)
            orbitCol = vars(find(hit,1));
            break;
        end
    end

    for i = 1:numel(slotCandidates)
        hit = cleanVars == slotCandidates(i);
        if any(hit)
            slotCol = vars(find(hit,1));
            break;
        end
    end

    if orbitCol == "" || slotCol == ""
        error("Could not find orbit_index and slot_index columns.");
    end

    n = min(round(numObservers), height(Tobs));
    xBest = nan(1, 2*n);

    for k = 1:n
        xBest(2*k-1) = double(Tobs.(orbitCol)(k));
        xBest(2*k)   = double(Tobs.(slotCol)(k));
    end

    if any(~isfinite(xBest))
        error("Invalid observer design from observer sheet.");
    end

    xBest = round(xBest);
end

function Xmc = sampleAroundBestDesign(xBest, lb, ub, nSamples, orbitRadius, slotRadius, mode)

    nVars = numel(xBest);
    Xmc = nan(nSamples, nVars);

    switch upper(string(mode))

        case "UNIFORM_BOX"
            for i = 1:nSamples
                x = xBest;

                for j = 1:nVars
                    if mod(j,2) == 1
                        rad = orbitRadius;
                    else
                        rad = slotRadius;
                    end

                    x(j) = xBest(j) + randi([-rad rad]);
                    x(j) = max(lb(j), min(ub(j), x(j)));
                end

                Xmc(i,:) = round(x);
            end

        case "RANDOM_WALK_SORTED"
            D = nan(nSamples, nVars);

            for i = 1:nSamples
                for j = 1:nVars
                    if mod(j,2) == 1
                        rad = orbitRadius;
                    else
                        rad = slotRadius;
                    end
                    D(i,j) = randi([-rad rad]);
                end
            end

            dist = sum(abs(D), 2);
            [~, ord] = sort(dist, "ascend");
            D = D(ord,:);

            for i = 1:nSamples
                x = xBest + D(i,:);
                x = max(lb, min(ub, x));
                Xmc(i,:) = round(x);
            end

        otherwise
            error("Unknown MC sampling mode: %s", mode);
    end

    Xmc(1,:) = round(xBest);
end

function fig = createTrueMCFigure(xSample, yCost, JgaMin, cfg, showLegend)

    fig = figure("Color","w", "Units","inches", ...
        "Position",[1 1 cfg.mcFigWidthIn cfg.mcFigHeightIn]);

    set(fig, "Renderer", "opengl");

    hold on; grid on; box on;

    scatter(xSample, yCost, cfg.mcScatterSize, ...
        'o', ...
        'MarkerEdgeColor', 'b', ...
        'MarkerFaceColor', 'none', ...
        'LineWidth', cfg.mcMarkerLineWidth);

    yline(JgaMin, "r-", "LineWidth", cfg.mcLineWidth);

    yAll = [yCost(:); JgaMin];
    yMin = min(yAll, [], "omitnan");
    yMax = max(yAll, [], "omitnan");

    if isfinite(yMin) && isfinite(yMax)
        if yMax > yMin
            yRange = yMax - yMin;
            lowerPad = cfg.mcLowerYPadFrac * yRange;
            upperPad = cfg.mcUpperYPadFrac * yRange;
        else
            lowerPad = max(1e-6, 0.05 * abs(yMin) + 1e-6);
            upperPad = lowerPad;
        end
        ylim([yMin - lowerPad, yMax + upperPad]);
    end

    xlabel("All Samples", ...
        "FontName", cfg.fontName, ...
        "FontSize", cfg.mcLabelFontSize, ...
        "FontWeight", cfg.fontWeight);

    ylabel("Total Cost", ...
        "FontName", cfg.fontName, ...
        "FontSize", cfg.mcLabelFontSize, ...
        "FontWeight", cfg.fontWeight);

    set(gca, ...
        "FontName", cfg.fontName, ...
        "FontSize", cfg.mcAxisFontSize, ...
        "FontWeight", cfg.fontWeight, ...
        "LineWidth", cfg.mcAxisLineWidth);

    title("");

    if showLegend
        legend({"Monte Carlo Samples", "GA Minimum"}, ...
            "Location", "northeast", ...
            "FontName", cfg.fontName, ...
            "FontSize", cfg.mcLegendFontSize, ...
            "FontWeight", cfg.fontWeight);
    end
end

function fig = createTrueMCBoxFigure(yCost, JgaMin, cfg, showLegend)

    fig = figure("Color","w", "Units","inches", ...
        "Position",[1 1 cfg.mcFigWidthIn cfg.mcFigHeightIn]);

    set(fig, "Renderer", "opengl");

    hold on; grid on; box on;

    yCost = yCost(:);
    yCost = yCost(isfinite(yCost));

    if isempty(yCost)
        warning("No finite Monte Carlo costs available for box plot.");
        yCost = NaN;
    end

    % Box-and-whisker representation of the MC objective values.
    % boxchart is preferred when available; boxplot is used as a fallback.
    try
        b = boxchart(ones(size(yCost)), yCost, ...
            "BoxWidth", 0.45, ...
            "MarkerStyle", "o", ...
            "LineWidth", cfg.mcMarkerLineWidth);

        try
            b.MarkerSize = max(4, round(cfg.mcScatterSize / 8));
        catch
        end
    catch
        boxplot(yCost, ...
            "Labels", {""}, ...
            "Widths", 0.45, ...
            "Symbol", "o");
    end

    % GA minimum reference line.
    yline(JgaMin, "r-", "LineWidth", cfg.mcLineWidth);

    yAll = [yCost(:); JgaMin];
    yMin = min(yAll, [], "omitnan");
    yMax = max(yAll, [], "omitnan");

    if isfinite(yMin) && isfinite(yMax)
        if yMax > yMin
            yRange = yMax - yMin;
            lowerPad = cfg.mcLowerYPadFrac * yRange;
            upperPad = cfg.mcUpperYPadFrac * yRange;
        else
            lowerPad = max(1e-6, 0.05 * abs(yMin) + 1e-6);
            upperPad = lowerPad;
        end
        ylim([yMin - lowerPad, yMax + upperPad]);
    end

    xlim([0.5 1.5]);
    xticks(1);
    xticklabels("MC Samples");

    xlabel("", ...
        "FontName", cfg.fontName, ...
        "FontSize", cfg.mcLabelFontSize, ...
        "FontWeight", cfg.fontWeight);

    ylabel("Total Cost", ...
        "FontName", cfg.fontName, ...
        "FontSize", cfg.mcLabelFontSize, ...
        "FontWeight", cfg.fontWeight);

    set(gca, ...
        "FontName", cfg.fontName, ...
        "FontSize", cfg.mcAxisFontSize, ...
        "FontWeight", cfg.fontWeight, ...
        "LineWidth", cfg.mcAxisLineWidth);

    title("");

    if showLegend
        % Dummy handles keep the legend clean and consistent.
        hBox = plot(nan, nan, "k-", "LineWidth", cfg.mcMarkerLineWidth);
        hGA  = plot(nan, nan, "r-", "LineWidth", cfg.mcLineWidth);

        legend([hBox hGA], {"Monte Carlo Distribution", "GA Minimum"}, ...
            "Location", "northeast", ...
            "FontName", cfg.fontName, ...
            "FontSize", cfg.mcLegendFontSize, ...
            "FontWeight", cfg.fontWeight);
    end
end

function legendText = exportEnlargedFigAsImageEPS(figPath, outEps, cfg, showLegend)

    fig = openfig(figPath, "invisible");

    set(fig, "Color", "w");
    set(fig, "Units", "inches");
    set(fig, "Position", [1 1 cfg.trajFigWidthIn cfg.trajFigHeightIn]);
    set(fig, "PaperUnits", "inches");
    set(fig, "PaperPosition", [0 0 cfg.trajFigWidthIn cfg.trajFigHeightIn]);
    set(fig, "PaperSize", [cfg.trajFigWidthIn cfg.trajFigHeightIn]);
    set(fig, "InvertHardcopy", "off");
    set(fig, "Renderer", "opengl");

    legendText = extractLegendText(fig);

    applyTrajectoryFigureStyle(fig, cfg);

    if showLegend
        setLegendVisibility(fig, "on");
    else
        setLegendVisibility(fig, "off");
    end

    drawnow;
    pause(0.05);

    exportTrajectoryEPS(fig, outEps, cfg);
    close(fig);
end

function applyTrajectoryFigureStyle(fig, cfg)

    ax = findall(fig, "Type", "axes");

    for i = 1:numel(ax)

        if strcmpi(ax(i).Tag, "legend")
            continue;
        end

        set(ax(i), ...
            "Units", "normalized", ...
            "FontName", cfg.fontName, ...
            "FontSize", cfg.trajAxisFontSize, ...
            "FontWeight", cfg.fontWeight, ...
            "LineWidth", cfg.trajAxisLineWidth);

        ax(i).XLabel.FontSize = cfg.trajLabelFontSize;
        ax(i).YLabel.FontSize = cfg.trajLabelFontSize;
        ax(i).ZLabel.FontSize = cfg.trajLabelFontSize;

        ax(i).XLabel.FontWeight = cfg.fontWeight;
        ax(i).YLabel.FontWeight = cfg.fontWeight;
        ax(i).ZLabel.FontWeight = cfg.fontWeight;

        ax(i).XLabel.FontName = cfg.fontName;
        ax(i).YLabel.FontName = cfg.fontName;
        ax(i).ZLabel.FontName = cfg.fontName;

        % Keep 3D geometry stable. The actual export padding is handled
        % in exportTrajectoryEPS using cfg.trajPaddedAxesPosition.
        axis(ax(i), "vis3d");
        box(ax(i), "on");
        grid(ax(i), "off");

        if cfg.removeTitles
            title(ax(i), "");
        else
            ax(i).Title.FontSize = cfg.trajTitleFontSize;
            ax(i).Title.FontWeight = cfg.fontWeight;
            ax(i).Title.FontName = cfg.fontName;
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
            scat(i).SizeData = max(scat(i).SizeData, cfg.trajMinScatterSize);
        catch
        end
    end

    lgd = findall(fig, "Type", "legend");
    for i = 1:numel(lgd)
        try
            set(lgd(i), ...
                "FontName", cfg.fontName, ...
                "FontSize", cfg.trajLegendFontSize, ...
                "FontWeight", cfg.fontWeight, ...
                "Box", "on");
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

function legendText = extractLegendText(fig)

    legendText = "";

    lgd = findall(fig, "Type", "legend");

    if isempty(lgd)
        return;
    end

    try
        txt = string(lgd(1).String);
        txt = txt(strlength(txt) > 0);
        legendText = strjoin(txt, " | ");
    catch
        legendText = "";
    end
end

function createTrajectoryLegendOnly(legendText, outEps, cfg)

    labels = split(string(legendText), " | ");
    labels = labels(strlength(labels) > 0);

    if isempty(labels)
        error("Trajectory legend text is empty.");
    end

    fig = figure("Color","w", "Units","inches", ...
        "Position",[1 1 cfg.legendFigWidthIn 0.95]);

    ax = axes(fig, "Visible", "off");
    hold(ax, "on");

    h = gobjects(numel(labels),1);

    for i = 1:numel(labels)
        h(i) = plot(ax, nan, nan, "-", "LineWidth", cfg.trajLineWidth);
    end

    lgd = legend(ax, h, labels, ...
        "Orientation", "horizontal", ...
        "Location", "south", ...
        "Box", "on");

    lgd.FontName = cfg.fontName;
    lgd.FontSize = cfg.trajSharedLegendSize;
    lgd.FontWeight = cfg.fontWeight;

    exportFigureAsImageEPS(fig, outEps, cfg);
    close(fig);
end

function createMCLegendOnly(outEps, cfg)

    fig = figure("Color","w", "Units","inches", ...
        "Position",[1 1 cfg.legendFigWidthIn cfg.legendFigHeightIn]);

    ax = axes(fig, "Visible", "off");
    hold(ax, "on");

    h1 = scatter(ax, nan, nan, cfg.mcScatterSize, ...
        'o', ...
        'MarkerEdgeColor', 'b', ...
        'MarkerFaceColor', 'none', ...
        'LineWidth', cfg.mcMarkerLineWidth);

    h2 = plot(ax, nan, nan, "r-", "LineWidth", cfg.mcLineWidth);

    lgd = legend(ax, [h1 h2], ...
        ["Monte Carlo Samples", "GA Minimum"], ...
        "Orientation", "horizontal", ...
        "Location", "south", ...
        "Box", "on");

    lgd.FontName = cfg.fontName;
    lgd.FontSize = cfg.mcSharedLegendSize;
    lgd.FontWeight = cfg.fontWeight;

    exportFigureAsImageEPS(fig, outEps, cfg);
    close(fig);
end

function exportTrajectoryEPS(fig, outEps, cfg)
%EXPORTTRAJECTORYEPS Export the main 3D trajectory axes through a padded
%temporary figure. This is tighter than exporting the full original .fig,
%but safer than axes-only export because it reserves whitespace for labels.

    ax = findall(fig, "Type", "axes");

    if isempty(ax)
        error("No axes found for trajectory export.");
    end

    keep = true(numel(ax), 1);
    for i = 1:numel(ax)
        if strcmpi(ax(i).Tag, "legend")
            keep(i) = false;
        end
    end
    ax = ax(keep);

    if isempty(ax)
        error("No trajectory axes found for export.");
    end

    % Pick the largest axes as the main trajectory axes.
    areas = zeros(numel(ax), 1);
    for i = 1:numel(ax)
        try
            pos = ax(i).Position;
            areas(i) = pos(3) * pos(4);
        catch
            areas(i) = 0;
        end
    end

    [~, idx] = max(areas);
    mainAx = ax(idx);

    drawnow;

    exportFig = figure( ...
        "Color", "w", ...
        "Units", "inches", ...
        "Position", [1 1 cfg.trajFigWidthIn cfg.trajFigHeightIn], ...
        "PaperUnits", "inches", ...
        "PaperPosition", [0 0 cfg.trajFigWidthIn cfg.trajFigHeightIn], ...
        "PaperSize", [cfg.trajFigWidthIn cfg.trajFigHeightIn], ...
        "InvertHardcopy", "off", ...
        "Renderer", "opengl", ...
        "Visible", "off");

    newAx = copyobj(mainAx, exportFig);

    set(newAx, ...
        "Units", "normalized", ...
        "Position", cfg.trajPaddedAxesPosition, ...
        "ActivePositionProperty", "position");

    axis(newAx, "vis3d");
    box(newAx, "on");
    grid(newAx, "off");

    set(newAx, ...
        "FontName", cfg.fontName, ...
        "FontSize", cfg.trajAxisFontSize, ...
        "FontWeight", cfg.fontWeight, ...
        "LineWidth", cfg.trajAxisLineWidth);

    newAx.XLabel.FontName = cfg.fontName;
    newAx.YLabel.FontName = cfg.fontName;
    newAx.ZLabel.FontName = cfg.fontName;

    newAx.XLabel.FontSize = cfg.trajLabelFontSize;
    newAx.YLabel.FontSize = cfg.trajLabelFontSize;
    newAx.ZLabel.FontSize = cfg.trajLabelFontSize;

    newAx.XLabel.FontWeight = cfg.fontWeight;
    newAx.YLabel.FontWeight = cfg.fontWeight;
    newAx.ZLabel.FontWeight = cfg.fontWeight;

    % Use capital axis labels for final trajectory export
    newAx.XLabel.String = "X (LU)";
    newAx.YLabel.String = "Y (LU)";
    newAx.ZLabel.String = "Z (LU)";

    try
    newAx.XLabel.Units = "normalized";
    newAx.YLabel.Units = "normalized";

    newAx.XLabel.Position = cfg.trajXLabelPosition;
    newAx.YLabel.Position = cfg.trajYLabelPosition;
    catch
    end

    if cfg.removeTitles
        title(newAx, "");
    end

    lines = findall(newAx, "Type", "line");
    for i = 1:numel(lines)
        try
            lines(i).LineWidth = cfg.trajLineWidth;
        catch
        end
    end

    scat = findall(newAx, "Type", "scatter");
    for i = 1:numel(scat)
        try
            scat(i).SizeData = max(scat(i).SizeData, cfg.trajMinScatterSize);
        catch
        end
    end

    drawnow;
    pause(0.05);

    % Quick preview before EPS export.
    if isfield(cfg, "previewTrajectoryExport") && cfg.previewTrajectoryExport
        previewPng = replace(string(outEps), ".eps", "_PREVIEW.png");

        exportgraphics(exportFig, previewPng, ...
            "ContentType", "image", ...
            "Resolution", cfg.previewResolution, ...
            "BackgroundColor", "white");

        try
            im = imread(previewPng);
            previewFig = figure( ...
                "Color", "w", ...
                "Name", "Trajectory EPS Preview", ...
                "NumberTitle", "off");
            imshow(im, "Border", "tight");
            title("Preview of trajectory export", ...
                "FontName", cfg.fontName, ...
                "FontWeight", cfg.fontWeight);
            drawnow;

            if isfield(cfg, "previewPauseSeconds")
                pause(cfg.previewPauseSeconds);
            else
                pause(0.75);
            end

            try
                close(previewFig);
            catch
            end
        catch ME
            warning("Could not display trajectory preview: %s", ME.message);
        end
    end

    try
        exportgraphics(exportFig, outEps, ...
            "ContentType", "image", ...
            "Resolution", cfg.trajEpsResolution, ...
            "BackgroundColor", "white");
    catch
        warning("Trajectory exportgraphics failed. Falling back to print.");
        print(exportFig, outEps, "-depsc", "-image", "-opengl", ...
            sprintf("-r%d", cfg.trajEpsResolution));
    end

    close(exportFig);
end

function exportMCFigureAsImageEPS(fig, outEps, cfg)
%EXPORTMCFIGUREASIMAGEEPS Keep Monte Carlo export as full-figure export.

    try
        exportgraphics(fig, outEps, ...
            "ContentType", "image", ...
            "Resolution", cfg.mcEpsResolution, ...
            "BackgroundColor", "white");
    catch
        warning("MC exportgraphics failed. Falling back to print.");
        print(fig, outEps, "-depsc", "-image", "-opengl", ...
            sprintf("-r%d", cfg.mcEpsResolution));
    end
end

function exportFigureAsImageEPS(fig, outEps, cfg)
%EXPORTFIGUREASIMAGEEPS Generic helper retained for legend-only files.

    if isfield(cfg, "trajEpsResolution")
        res = cfg.trajEpsResolution;
    elseif isfield(cfg, "mcEpsResolution")
        res = cfg.mcEpsResolution;
    else
        res = 600;
    end

    try
        exportgraphics(fig, outEps, ...
            "ContentType", "image", ...
            "Resolution", res, ...
            "BackgroundColor", "white");
    catch
        warning("exportgraphics EPS image export failed. Falling back to print.");
        print(fig, outEps, "-depsc", "-image", "-opengl", sprintf("-r%d", res));
    end
end

function [costCol, found] = findCostColumn(T)

    costCol = "";
    found = false;

    if isempty(T)
        return;
    end

    vars = string(T.Properties.VariableNames);

    candidates = [
        "J_total"
        "J"
        "cost"
        "Cost"
        "total_cost"
        "TotalCost"
        "objective"
        "Objective"
        "min_cost"
    ];

    for i = 1:numel(candidates)
        idx = strcmpi(vars, candidates(i));
        if any(idx)
            costCol = vars(find(idx,1));
            found = true;
            return;
        end
    end

    for i = 1:numel(vars)
        v = vars(i);
        if contains(lower(v), "j_total") || contains(lower(v), "cost")
            try
                if isnumeric(T.(v))
                    costCol = v;
                    found = true;
                    return;
                end
            catch
            end
        end
    end
end

function [xCols, xColNames] = findDesignColumns(T)

    xCols = [];
    xColNames = strings(0,1);

    if isempty(T)
        return;
    end

    vars = string(T.Properties.VariableNames);
    cleanVars = lower(regexprep(vars, "[^a-zA-Z0-9]", ""));

    nums = nan(numel(vars),1);

    for i = 1:numel(vars)
        tok = regexp(cleanVars(i), "^x(\d+)$", "tokens", "once");
        if ~isempty(tok)
            nums(i) = str2double(tok{1});
        end
    end

    valid = isfinite(nums);

    if ~any(valid)
        return;
    end

    idx = find(valid);
    [~, ord] = sort(nums(valid), "ascend");

    xCols = idx(ord);
    xColNames = vars(xCols);
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

function cacheKey = make_transfer_cache_key(missionCfg, slots_per_orbit)

    tr = missionCfg.transfer;
    solverMode = upper(string(tr.solverMode));

    depOrb  = get_field_or_default(tr, 'depOrbitIndex', 0);
    arrOrb  = get_field_or_default(tr, 'arrOrbitIndex', 0);
    depSlot = get_field_or_default(tr, 'depSlot', 0);
    arrSlot = get_field_or_default(tr, 'arrSlot', 0);
    dtVal   = get_field_or_default(tr, 'dt', 0);

    switch solverMode
        case "LOW_THRUST_CLASS"
            lt = tr.lowthrust;

            cacheKey = sprintf('lt_d%d_a%d_ds%d_as%d_dt%s_sl%d_tf%s', ...
                depOrb, arrOrb, depSlot, arrSlot, ...
                local_num_str(dtVal), slots_per_orbit, ...
                local_num_str(get_field_or_default(lt, 'tf_guess', 0)));

        otherwise
            cacheKey = sprintf('tr_d%d_a%d_ds%d_as%d_dt%s_sl%d', ...
                depOrb, arrOrb, depSlot, arrSlot, ...
                local_num_str(dtVal), slots_per_orbit);
    end

    cacheKey = regexprep(cacheKey, '[^A-Za-z0-9_]', '_') + "_halfopen_v1";
end

function v = get_field_or_default(s, fieldName, defaultVal)
    if isfield(s, fieldName) && ~isempty(s.(fieldName))
        v = s.(fieldName);
    else
        v = defaultVal;
    end
end

function s = local_num_str(x)
    if isempty(x) || ~isfinite(x)
        s = "0";
        return;
    end
    s = string(x);
    s = replace(s, ".", "p");
    s = replace(s, "-", "m");
end

function r = tiedrankSafe(x)

    x = double(x);
    r = nan(size(x));

    finiteIdx = isfinite(x);

    if ~any(finiteIdx)
        return;
    end

    try
        r(finiteIdx) = tiedrank(x(finiteIdx));
    catch
        [~, ord] = sort(x(finiteIdx));
        rr = nan(sum(finiteIdx),1);
        rr(ord) = 1:sum(finiteIdx);
        r(finiteIdx) = rr;
    end
end

function Tout = appendTableUnion(Tout, Tin)
%APPENDTABLEUNION Vertically concatenate tables with different variables.

    if isempty(Tin)
        return;
    end

    if isempty(Tout)
        Tout = Tin;
        return;
    end

    varsOut = string(Tout.Properties.VariableNames);
    varsIn  = string(Tin.Properties.VariableNames);

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
    Tin  = Tin(:, cellstr(allVars));

    Tout = [Tout; Tin];
end

function col = makeMissingColumn(n, exampleCol)

    if isstring(exampleCol)
        col = strings(n, 1);
        col(:) = missing;

    elseif iscell(exampleCol)
        col = cell(n, 1);
        col(:) = {[]};

    elseif iscategorical(exampleCol)
        col = categorical(strings(n, 1));
        col(:) = categorical(missing);

    elseif isdatetime(exampleCol)
        col = NaT(n, 1);

    elseif isduration(exampleCol)
        col = seconds(nan(n, 1));

    elseif islogical(exampleCol)
        col = false(n, 1);

    elseif isnumeric(exampleCol)
        col = nan(n, size(exampleCol, 2));

    else
        col = strings(n, 1);
        col(:) = missing;
    end
end
