% ---- print_observer_ics_from_experiment_summary.m ---- %
clear; close all; clc;

% ============================================================
% Purpose:
%   Read a saved optimizer observer configuration from an
%   ExperimentSummary_*.xlsx file and print a simplified table:
%
%       observer_id
%       orbit_family
%       period_TU
%       observer_IC_nondimensional
%
%   where:
%
%       observer_IC_nondimensional = [x, y, z, vx, vy, vz]
%
%   No files are saved.
% ============================================================


%% ---------------- User inputs ----------------

% Choose run root:
%   "runs_GA" for baseline
%   "runs"    for comparison
RUN_ROOT = "runs/20260624_141925";

% Configuration from the ExperimentSummary filename
% Example:
%   ExperimentSummary_GA_scr1_AO_J111_seed000_lg_o5.xlsx
CONFIG_TAG = "GA_scr1_AO_J111_seed000_lg_o3";

% Period case from the parent folder name.
% Example folders:
%   b_ga600_ao_o5_p1
%   b_ga600_ao_o5_p3
%   b_ga600_ao_o5_p5
NUM_PERIODS = 1;
PERIOD_TAG = "p" + string(NUM_PERIODS);

% This should match run_opt.m
slots_per_orbit = 50;

% Your observer sheet is the last sheet in the ExperimentSummary file.
READ_LAST_SHEET = true;


%% ---------------- Constants ----------------

mu = 1.215058560962404E-2;

LU = 384400;       % km
TU = 375695;       % seconds
VU = LU / TU;      % km/s


%% ---------------- Paths ----------------

thisFile = mfilename('fullpath');
thisDir  = fileparts(thisFile);

addpath(genpath(thisDir));

runRootPath = fullfile(thisDir, RUN_ROOT);

if ~isfolder(runRootPath)
    error('Run root folder does not exist:\n%s', runRootPath);
end

catalogPath = fullfile(thisDir, 'JPL_CR3BP_OrbitCatalog.mat');

if ~isfile(catalogPath)
    error('Could not find JPL_CR3BP_OrbitCatalog.mat at:\n%s', catalogPath);
end


%% ---------------- Find ExperimentSummary file ----------------

excelFile = find_experiment_summary_file(runRootPath, CONFIG_TAG, PERIOD_TAG);

fprintf('\nSelected ExperimentSummary file:\n  %s\n', excelFile);


%% ---------------- Read observer configuration from Excel ----------------

sheetNames = sheetnames(excelFile);

fprintf('\nSheets found in Excel file:\n');
disp(sheetNames(:));

if READ_LAST_SHEET
    obsSheet = sheetNames(end);
else
    obsSheet = find_observer_sheet(excelFile, sheetNames);
end

fprintf('\nReading observer configuration from sheet:\n  %s\n', obsSheet);

observer_input = readtable(excelFile, 'Sheet', obsSheet);
observer_input = normalize_observer_table(observer_input);

fprintf('\nObserver configuration read from Excel:\n');
disp(observer_input);


%% ---------------- Load JPL orbit catalog ----------------

S = load(catalogPath);

T1     = S.T;
times  = T1.("time");
states = T1.("state");
tf     = T1.("Period (TU) ");

num_orbits = height(T1);

fprintf('\nLoaded JPL catalog with %d orbits.\n', num_orbits);


%% ---------------- Build/load orbit slot database ----------------

OrbitCacheDir = fullfile(thisDir, "orbit_cache");

if ~exist(OrbitCacheDir, 'dir')
    mkdir(OrbitCacheDir);
end

orbitDbCacheFile = fullfile(OrbitCacheDir, ...
    sprintf('orbit_database_slots_%d.mat', slots_per_orbit));

rebuildOrbitDb = true;

if isfile(orbitDbCacheFile)
    try
        C = load(orbitDbCacheFile, 'orbit_database', 'cacheMeta');

        if isfield(C, 'orbit_database') && numel(C.orbit_database) == num_orbits
            orbit_database = C.orbit_database;
            rebuildOrbitDb = false;
            fprintf('\nLoaded cached orbit database from:\n  %s\n', orbitDbCacheFile);
        end

    catch ME
        warning('Failed to load orbit database cache: %s', ME.message);
        rebuildOrbitDb = true;
    end
end

if rebuildOrbitDb
    fprintf('\nBuilding orbit database with %d slots per orbit...\n', slots_per_orbit);

    orbit_database = cell(num_orbits, 1);

    for i = 1:num_orbits
        t_raw  = times{i};
        s_raw  = states{i};
        period = tf(i);

        t_slots = (0:slots_per_orbit-1)' * period / slots_per_orbit;

        [t_unique, idx_u] = unique(t_raw);
        s_unique = s_raw(idx_u, :);

        F = griddedInterpolant(t_unique, s_unique, 'spline');
        s_slots = F(t_slots);

        orbit_database{i} = s_slots;
    end

    cacheMeta = struct();
    cacheMeta.created         = string(datetime('now'));
    cacheMeta.num_orbits      = num_orbits;
    cacheMeta.slots_per_orbit = slots_per_orbit;

    save(orbitDbCacheFile, 'orbit_database', 'cacheMeta', '-v7.3');

    fprintf('Saved orbit database cache to:\n  %s\n', orbitDbCacheFile);
end


%% ---------------- Convert orbit/slot rows to observer ICs ----------------

num_obs = height(observer_input);

orbit_indices = observer_input.orbit_index;
slot_indices  = observer_input.slot_index;

observer_ICs = zeros(num_obs, 6);
period_TU_catalog = zeros(num_obs, 1);

for k = 1:num_obs
    iOrb  = round(orbit_indices(k));
    iSlot = round(slot_indices(k));

    iOrb  = max(1, min(iOrb, numel(orbit_database)));
    iSlot = max(1, min(iSlot, size(orbit_database{iOrb}, 1)));

    orbit_indices(k) = iOrb;
    slot_indices(k)  = iSlot;

    observer_ICs(k, :) = orbit_database{iOrb}(iSlot, :);
    period_TU_catalog(k) = tf(iOrb);
end


%% ---------------- Build simplified output table ----------------

if ismember("orbit_family", observer_input.Properties.VariableNames)
    orbit_family = string(observer_input.orbit_family);
else
    orbit_family = strings(num_obs, 1);
end

observer_IC_nondimensional = strings(num_obs, 1);

for k = 1:num_obs
    observer_IC_nondimensional(k) = sprintf( ...
        '[%.10g, %.10g, %.10g, %.10g, %.10g, %.10g]', ...
        observer_ICs(k,1), ...
        observer_ICs(k,2), ...
        observer_ICs(k,3), ...
        observer_ICs(k,4), ...
        observer_ICs(k,5), ...
        observer_ICs(k,6));
end

simple_IC_table = table( ...
    observer_input.observer_id, ...
    orbit_family, ...
    period_TU_catalog, ...
    observer_IC_nondimensional, ...
    'VariableNames', { ...
        'observer_id', ...
        'orbit_family', ...
        'period_TU', ...
        'observer_IC_nondimensional' ...
    });


%% ---------------- Display simplified table only ----------------

fprintf('\n============================================================\n');
fprintf('Simplified observer IC table\n');
fprintf('RUN_ROOT:    %s\n', RUN_ROOT);
fprintf('CONFIG_TAG:  %s\n', CONFIG_TAG);
fprintf('PERIOD_TAG:  %s\n', PERIOD_TAG);
fprintf('Excel file:  %s\n', excelFile);
fprintf('Sheet:       %s\n', obsSheet);
fprintf('============================================================\n\n');

disp(simple_IC_table);

fprintf('\nEach observer IC is in nondimensional CR3BP units:\n');
fprintf('  [x, y, z, vx, vy, vz]\n\n');



%% ============================================================
% Helper functions
% ============================================================

function excelFile = find_experiment_summary_file(runRootPath, configTag, periodTag)

    pattern = fullfile(runRootPath, "**", "ExperimentSummary_*.xlsx");
    files = dir(pattern);

    if isempty(files)
        error('No ExperimentSummary_*.xlsx files found under:\n%s', runRootPath);
    end

    configTag = string(configTag);
    periodTag = string(periodTag);

    names = string({files.name});

    % First match by ExperimentSummary filename
    configMatched = files(contains(names, configTag));

    if isempty(configMatched)
        fprintf('\nAvailable ExperimentSummary files:\n');

        for i = 1:numel(files)
            fprintf('  %s\n', fullfile(files(i).folder, files(i).name));
        end

        error('No ExperimentSummary file found containing CONFIG_TAG:\n%s', configTag);
    end

    % Then match by folder period tag.
    % This avoids picking p1 when you meant p3 or p5.
    folders = string({configMatched.folder});

    % Match folder pieces like:
    %   _p1
    %   _p3
    %   _p5
    periodPattern = "_" + periodTag;

    periodMatched = configMatched(contains(folders, periodPattern));

    if isempty(periodMatched)
        fprintf('\nFiles matched CONFIG_TAG, but none matched PERIOD_TAG = %s.\n', periodTag);
        fprintf('\nCONFIG_TAG matches were:\n');

        for i = 1:numel(configMatched)
            fprintf('  %s\n', fullfile(configMatched(i).folder, configMatched(i).name));
        end

        error('No ExperimentSummary file found for CONFIG_TAG = %s and PERIOD_TAG = %s.', ...
            configTag, periodTag);
    end

    if numel(periodMatched) > 1
        fprintf('\nMultiple matching files found for CONFIG_TAG = %s and PERIOD_TAG = %s.\n', ...
            configTag, periodTag);
        fprintf('Using most recently modified file:\n');

        for i = 1:numel(periodMatched)
            fprintf('  %s\n', fullfile(periodMatched(i).folder, periodMatched(i).name));
        end

        [~, idxNewest] = max([periodMatched.datenum]);
        periodMatched = periodMatched(idxNewest);
    end

    excelFile = fullfile(periodMatched.folder, periodMatched.name);
end


function obsSheet = find_observer_sheet(excelFile, sheetNames)

    requiredCols = ["observer_id", "orbit_index", "slot_index"];

    obsSheet = "";

    for i = 1:numel(sheetNames)
        try
            Ttest = readtable(excelFile, 'Sheet', sheetNames(i), 'NumRows', 1);
            names = string(Ttest.Properties.VariableNames);

            if all(ismember(requiredCols, names))
                obsSheet = sheetNames(i);
            end
        catch
        end
    end

    if strlength(obsSheet) == 0
        obsSheet = sheetNames(end);
        warning('No observer-looking sheet found. Falling back to last sheet: %s', obsSheet);
    end
end


function T = normalize_observer_table(T)

    names = string(T.Properties.VariableNames);

    required = ["observer_id", "orbit_index", "slot_index"];

    if all(ismember(required, names))
        return;
    end

    lowerNames = lower(names);

    for i = 1:numel(required)
        target = required(i);
        idx = find(lowerNames == lower(target), 1);

        if ~isempty(idx)
            T.Properties.VariableNames{idx} = char(target);
        end
    end

    names = string(T.Properties.VariableNames);

    if ~ismember("observer_id", names)
        T.observer_id = (1:height(T))';
    end

    if ~ismember("orbit_index", names)
        error('Observer table does not contain an orbit_index column.');
    end

    if ~ismember("slot_index", names)
        error('Observer table does not contain a slot_index column.');
    end

    if ~isnumeric(T.orbit_index)
        T.orbit_index = str2double(string(T.orbit_index));
    end

    if ~isnumeric(T.slot_index)
        T.slot_index = str2double(string(T.slot_index));
    end

    if ~isnumeric(T.observer_id)
        T.observer_id = str2double(string(T.observer_id));
    end

    if any(isnan(T.orbit_index)) || any(isnan(T.slot_index))
        error('orbit_index or slot_index contains NaN after conversion.');
    end
end